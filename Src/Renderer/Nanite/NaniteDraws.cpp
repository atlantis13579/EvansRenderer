#include "Renderer/Nanite/NaniteDraws.h"
#include "Renderer/Geometry/Instancing.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        constexpr double TransformRoundoff=32*std::numeric_limits<float>::epsilon();
        constexpr double TransformUnderflow=32*double(std::numeric_limits<float>::min());
        bool Affine(const Matrix4& m)
        {
            return std::all_of(m.Values.begin(),m.Values.end(),[](float v){return std::isfinite(v);}) &&
                m.Values[12]==0 && m.Values[13]==0 && m.Values[14]==0 && m.Values[15]==1;
        }
        bool PositionInterval(const Matrix4& world,Float3 position,double* lo,double* hi)
        {
            const double p[]{position.X,position.Y,position.Z,1};
            for (unsigned row=0;row<3;++row)
            {
                double value=0,magnitude=0;
                for (unsigned col=0;col<4;++col) { const auto term=world.Values[row*4+col]*p[col]; value+=term; magnitude+=std::abs(term); }
                const double error=TransformRoundoff*magnitude+TransformUnderflow;
                lo[row]=value-error; hi[row]=value+error;
                if (!std::isfinite(magnitude) || magnitude>std::numeric_limits<float>::max() ||
                    lo[row]<-std::numeric_limits<float>::max() || hi[row]>std::numeric_limits<float>::max()) return false;
            }
            return true;
        }
    }
    bool NaniteDrawPipelineSupported(const PipelineState& p)
    {
        return p.Fill==FillMode::Solid && p.DepthTest && p.DepthWrite && !p.BlendEnabled &&
            !p.OutlineExtrusion && !p.Parallax.Enabled;
    }
    bool SameNaniteDraw(const NaniteClusterDraw& a,const NaniteClusterDraw& b)
    {
        return a.VertexBuffer==b.VertexBuffer && a.IndexBuffer==b.IndexBuffer && a.FirstIndex==b.FirstIndex &&
            a.IndexCount==b.IndexCount && a.World.Values==b.World.Values && a.ViewProjection.Values==b.ViewProjection.Values &&
            a.LightViewProjection.Values==b.LightViewProjection.Values && SameInstancedPipeline(a.Pipeline,b.Pipeline);
    }
    bool BuildNaniteDrawBins(const std::vector<NaniteClusterDraw>& draws,NaniteBinLayout& output,std::string& error)
    {
        if (draws.empty() || draws.size()>NaniteVisibilityMaximumQueries)
        { error="Nanite bin candidate count is invalid"; return false; }
        NaniteBinLayout result; std::uint64_t totalIndices=0,blocks=0;
        for (unsigned i=0;i<draws.size();++i)
        {
            const auto& draw=draws[i];
            if (!draw.IndexCount || draw.IndexCount%3 || !NaniteDrawPipelineSupported(draw.Pipeline) ||
                std::uint64_t(draw.FirstIndex)+draw.IndexCount>UINT32_MAX)
            { error="Nanite bin requires valid opaque triangle ranges"; return false; }
            bool merge=false;
            if (!result.Bins.empty())
            {
                const auto& previous=draws[i-1]; auto comparison=draw;
                comparison.FirstIndex=previous.FirstIndex; comparison.IndexCount=previous.IndexCount;
                merge=std::uint64_t(previous.FirstIndex)+previous.IndexCount==draw.FirstIndex && SameNaniteDraw(previous,comparison);
            }
            if (!merge) result.Bins.push_back({i,0,draw.FirstIndex,0,unsigned(totalIndices),0,0});
            auto& bin=result.Bins.back(); ++bin.DrawCount; bin.IndexCount+=draw.IndexCount; totalIndices+=draw.IndexCount;
            if (totalIndices>NaniteMaximumDrawBytes/sizeof(unsigned)) { error="Nanite bin source indices exceed the frame budget"; return false; }
        }
        for (auto& bin:result.Bins)
        { bin.FirstBlock=unsigned(blocks); bin.BlockCount=(bin.DrawCount+NaniteVisibilityThreads-1)/NaniteVisibilityThreads; blocks+=bin.BlockCount; }
        const std::uint64_t candidateWords=draws.size()*NaniteBinCandidateWords,blockWords=blocks*NaniteBinBlockWords;
        const std::uint64_t metadataWords=candidateWords+blockWords+result.Bins.size()*NaniteBinDescriptorWords;
        const std::uint64_t argumentOffset=draws.size()+2*blocks,indexOffset=argumentOffset+result.Bins.size()*NaniteDrawArgumentWords;
        const std::uint64_t outputWords=indexOffset+totalIndices;
        if ((metadataWords+outputWords+totalIndices)*sizeof(unsigned)>NaniteMaximumDrawBytes)
        { error="Nanite bin scratch exceeds the frame budget"; return false; }
        result.Metadata.resize(std::size_t(metadataWords)); result.OutputWords=unsigned(outputWords); result.SourceIndexWords=unsigned(totalIndices);
        auto& c=result.Constants; c.Counts={unsigned(draws.size()),unsigned(blocks),unsigned(result.Bins.size()),0};
        c.Layout={0,unsigned(candidateWords),unsigned(candidateWords+blockWords),0};
        c.Source={0,unsigned(draws.size()),unsigned(draws.size()+blocks),unsigned(argumentOffset)};
        c.Target={unsigned(indexOffset),unsigned(totalIndices),0,0};
        for (unsigned binId=0;binId<result.Bins.size();++binId)
        {
            const auto& bin=result.Bins[binId]; const auto at=c.Layout.Z+binId*NaniteBinDescriptorWords;
            const unsigned descriptor[]{bin.FirstDraw,bin.DrawCount,bin.FirstBlock,bin.BlockCount,bin.OutputIndex,bin.IndexCount,0,0};
            std::copy(std::begin(descriptor),std::end(descriptor),result.Metadata.begin()+at);
            for (unsigned block=0;block<bin.BlockCount;++block)
            {
                const unsigned first=bin.FirstDraw+block*NaniteVisibilityThreads,count=std::min(NaniteVisibilityThreads,bin.FirstDraw+bin.DrawCount-first);
                const unsigned record[]{first,count,binId,0}; std::copy(std::begin(record),std::end(record),result.Metadata.begin()+c.Layout.Y+(bin.FirstBlock+block)*NaniteBinBlockWords);
            }
            for (unsigned i=0;i<bin.DrawCount;++i)
            {
                const auto& draw=draws[bin.FirstDraw+i]; const unsigned record[]{bin.OutputIndex+draw.FirstIndex-bin.FirstIndex,draw.IndexCount,bin.FirstBlock+i/NaniteVisibilityThreads,binId};
                std::copy(std::begin(record),std::end(record),result.Metadata.begin()+c.Layout.X+(bin.FirstDraw+i)*NaniteBinCandidateWords);
            }
        }
        output=std::move(result); error.clear(); return true;
    }
    bool ValidateNaniteClusterDraw(const NaniteClusterDraw& draw,const NaniteVisibilityQuery& query,
        const BufferDesc& vertexDesc,const std::vector<std::uint8_t>& vertices,
        const BufferDesc& indexDesc,const std::vector<std::uint8_t>& indices,std::string& error)
    {
        if (!NaniteDrawPipelineSupported(draw.Pipeline) || !Affine(draw.World) || draw.ViewProjection.Values!=query.Current.Values ||
            vertexDesc.Usage!=BufferUsage::Vertex || indexDesc.Usage!=BufferUsage::Index || vertices.size()%sizeof(Vertex) || indices.size()%sizeof(unsigned) ||
            !draw.IndexCount || draw.IndexCount%3 || draw.FirstIndex>indices.size()/sizeof(unsigned) ||
            draw.IndexCount>indices.size()/sizeof(unsigned)-draw.FirstIndex)
        { error="Nanite draw requires matching projection, affine world, opaque solid material and valid triangle buffers"; return false; }
        const float boundsLo[]{query.Minimum.X,query.Minimum.Y,query.Minimum.Z},boundsHi[]{query.Maximum.X,query.Maximum.Y,query.Maximum.Z};
        for (unsigned i=0;i<draw.IndexCount;++i)
        {
            unsigned index=0; std::memcpy(&index,indices.data()+(std::size_t(draw.FirstIndex)+i)*sizeof(unsigned),sizeof(index));
            if (index>=vertices.size()/sizeof(Vertex)) { error="Nanite draw index exceeds decoded vertices"; return false; }
            Vertex vertex; std::memcpy(&vertex,vertices.data()+std::size_t(index)*sizeof(Vertex),sizeof(vertex)); double lo[3],hi[3];
            if (!PositionInterval(draw.World,vertex.Position,lo,hi)) { error="Nanite draw world transform overflows"; return false; }
            for (unsigned axis=0;axis<3;++axis) if (lo[axis]<boundsLo[axis] || hi[axis]>boundsHi[axis])
            { error="Nanite query does not conservatively enclose the drawn world-space geometry"; return false; }
        }
        return true;
    }
    bool BuildNaniteDrawQuery(Float3 minimum,Float3 maximum,const Matrix4& world,const Matrix4& previousWorld,
        const Matrix4& viewProjection,const Matrix4& previousViewProjection,bool historyValid,unsigned cluster,unsigned instance,
        NaniteVisibilityQuery& output,std::string& error)
    {
        if (!Affine(world) || (historyValid && !Affine(previousWorld))) { error="Nanite draw query requires affine world transforms"; return false; }
        const float low[]{minimum.X,minimum.Y,minimum.Z},high[]{maximum.X,maximum.Y,maximum.Z};
        for (unsigned axis=0;axis<3;++axis) if (!std::isfinite(low[axis]) || !std::isfinite(high[axis]) || low[axis]>high[axis])
        { error="Nanite draw query has invalid source bounds"; return false; }
        double lower[3]{INFINITY,INFINITY,INFINITY},upper[3]{-INFINITY,-INFINITY,-INFINITY};
        for (unsigned frame=0;frame<(historyValid?2u:1u);++frame) for (unsigned corner=0;corner<8;++corner)
        {
            const Float3 point{(corner&1)?high[0]:low[0],(corner&2)?high[1]:low[1],(corner&4)?high[2]:low[2]}; double lo[3],hi[3];
            if (!PositionInterval(frame?previousWorld:world,point,lo,hi)) { error="Nanite draw query transform overflows"; return false; }
            for (unsigned axis=0;axis<3;++axis) { lower[axis]=std::min(lower[axis],lo[axis]); upper[axis]=std::max(upper[axis],hi[axis]); }
        }
        float lo[3],hi[3];
        for (unsigned axis=0;axis<3;++axis) { lo[axis]=std::nextafter(float(lower[axis]),-INFINITY); hi[axis]=std::nextafter(float(upper[axis]),INFINITY); }
        NaniteVisibilityQuery q; q.Minimum={lo[0],lo[1],lo[2],0}; q.Maximum={hi[0],hi[1],hi[2],0};
        q.Current=viewProjection; q.Previous=previousViewProjection; q.Identity={historyValid?1u:0u,0,cluster,instance};
        NaniteVisibilityWork validation;
        if (!BuildNaniteFrameCull({q},1,1,NaniteVisibilityPhase::Main,validation,error)) return false;
        output=q; return true;
    }
}
