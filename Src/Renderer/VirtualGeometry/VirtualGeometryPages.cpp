#include "Renderer/VirtualGeometry/VirtualGeometryPages.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>
#include <type_traits>

namespace Evans::Renderer
{
    namespace
    {
        constexpr unsigned Word(VirtualClusterWord value) { return static_cast<unsigned>(value); }
        constexpr std::uint64_t HashSeed=14695981039346656037ull,HashPrime=1099511628211ull;
        std::uint32_t Bits(float value) { std::uint32_t bits; std::memcpy(&bits,&value,sizeof(bits)); return bits; }
        float Scalar(std::uint32_t value) { float scalar; std::memcpy(&scalar,&value,sizeof(scalar)); return scalar; }
        bool Finite(Float3 p) { return std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z); }
        bool Same(Float3 a,Float3 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z; }
        void Bounds(Float3 p,Float3& lo,Float3& hi)
        {
            lo={std::min(lo.X,p.X),std::min(lo.Y,p.Y),std::min(lo.Z,p.Z)};
            hi={std::max(hi.X,p.X),std::max(hi.Y,p.Y),std::max(hi.Z,p.Z)};
        }
        std::uint64_t Signature(const VirtualGeometryPages& data)
        {
            std::uint64_t hash=HashSeed; auto mix=[&](std::uint32_t value){hash^=value; hash*=HashPrime;};
            mix(data.SourceVertexCount); mix(data.SourceTriangleCount); mix(static_cast<std::uint32_t>(data.Pages.size()));
            for (const auto& page:data.Pages) for (auto value:page.Words) mix(value);
            mix(static_cast<std::uint32_t>(data.Clusters.size()));
            for (const auto& cluster:data.Clusters)
            {
                mix(cluster.Page); mix(cluster.Word); mix(cluster.Material); mix(cluster.Node);
                for (float value:{cluster.BoundsMinimum.X,cluster.BoundsMinimum.Y,cluster.BoundsMinimum.Z,cluster.BoundsMaximum.X,cluster.BoundsMaximum.Y,cluster.BoundsMaximum.Z}) mix(Bits(value));
            }
            mix(static_cast<std::uint32_t>(data.SourceVertices.size())); for (auto value:data.SourceVertices) mix(value);
            mix(static_cast<std::uint32_t>(data.SourceTriangles.size())); for (auto value:data.SourceTriangles) mix(value); return hash;
        }
        bool PageHeader(const VirtualGeometryPage& page,std::string& error)
        {
            const auto& words=page.Words;
            if (words[0]!=VirtualGeometryPageMagic || words[1]!=VirtualGeometryPageVersion ||
                words[2]<VirtualGeometryPageHeaderWords+VirtualGeometryClusterHeaderWords || words[2]>VirtualGeometryPageWords || !words[3] ||
                (words[4]&~VirtualGeometryPinnedPage) || words[5] || words[6] || words[7])
            { error="Invalid virtual geometry source-page header"; return false; }
            return true;
        }
        struct Corner { std::uint32_t Material,Vertex,Triangle; };
        struct Adjacency { std::uint64_t Key; std::uint32_t Begin,End; bool Expanded=false; };
        std::uint64_t CornerKey(std::uint32_t material,std::uint32_t vertex) { return (std::uint64_t(material)<<32)|vertex; }
    }

    bool EncodeVirtualGeometryPages(const std::vector<Vertex>& vertices,std::uint32_t sourceTriangleCount,
        const std::vector<VirtualGeometryClusterInput>& clusters,const VirtualGeometryPageLimits& limits,
        VirtualGeometryPages& output,std::string& error)
    {
        static_assert(std::is_trivially_copyable_v<Vertex>,"Virtual pages copy complete vertex words without conversion");
        const GeometryPreparationLimits native;
        if (clusters.empty() || !sourceTriangleCount || vertices.size()>std::min(limits.Source.MaximumVertices,native.MaximumVertices) ||
            std::uint64_t(sourceTriangleCount)*3>std::min(limits.Source.MaximumIndices,native.MaximumIndices) || clusters.size()>limits.MaximumClusters ||
            !limits.MaximumPages || limits.MaximumPages>std::numeric_limits<std::uint32_t>::max()/VirtualGeometryPageWords ||
            clusters.size()>std::numeric_limits<std::uint32_t>::max())
        { error="Virtual geometry source or page/cluster dimensions exceed their budgets"; return false; }
        if (!ValidateTriangleGeometry(vertices,clusters.front().Indices,error)) return false;
        VirtualGeometryPages candidate; candidate.SourceVertexCount=static_cast<std::uint32_t>(vertices.size()); candidate.SourceTriangleCount=sourceTriangleCount;
        for (const auto& input:clusters)
        {
            if (input.Indices.empty() || input.Indices.size()%3 || input.Indices.size()>VirtualGeometryClusterTriangles*3 ||
                (!input.SourceTriangles.empty() && input.SourceTriangles.size()!=input.Indices.size()/3))
            { error="Virtual cluster requires bounded indexed triangles and matching source keys"; return false; }
            std::vector<std::uint32_t> sourceVertices,localIndices;
            for (auto index:input.Indices)
            {
                if (index>=vertices.size()) { error="Virtual cluster references a missing source vertex"; return false; }
                auto found=std::find(sourceVertices.begin(),sourceVertices.end(),index);
                if (found==sourceVertices.end()) { sourceVertices.push_back(index); localIndices.push_back(static_cast<std::uint32_t>(sourceVertices.size()-1)); }
                else localIndices.push_back(static_cast<std::uint32_t>(found-sourceVertices.begin()));
            }
            for (auto key:input.SourceTriangles) if (key>=sourceTriangleCount) { error="Virtual cluster source triangle key is outside the source"; return false; }
            const std::size_t payload=VirtualGeometryClusterHeaderWords+sourceVertices.size()*VirtualGeometryVertexWords+localIndices.size();
            if (payload+VirtualGeometryPageHeaderWords>VirtualGeometryPageWords ||
                candidate.SourceVertices.size()+sourceVertices.size()>std::numeric_limits<std::uint32_t>::max() ||
                candidate.SourceTriangles.size()+input.Indices.size()/3>std::numeric_limits<std::uint32_t>::max())
            { error="Virtual cluster payload or source-key tables exceed native addressing"; return false; }
            const auto pageFlags=input.Pinned?VirtualGeometryPinnedPage:0;
            if (candidate.Pages.empty() || candidate.Pages.back().Words[2]+payload>VirtualGeometryPageWords || candidate.Pages.back().Words[4]!=pageFlags)
            {
                if (candidate.Pages.size()>=limits.MaximumPages) { error="Virtual geometry exceeded its encoded page budget"; return false; }
                candidate.Pages.emplace_back(); auto& page=candidate.Pages.back().Words;
                page[0]=VirtualGeometryPageMagic; page[1]=VirtualGeometryPageVersion; page[2]=VirtualGeometryPageHeaderWords; page[4]=pageFlags;
            }
            auto& page=candidate.Pages.back().Words; const auto start=page[2]; auto* header=page.data()+start;
            VirtualGeometryCluster cluster; cluster.Page=static_cast<std::uint32_t>(candidate.Pages.size()-1); cluster.Word=start; cluster.Material=input.Material; cluster.Node=input.Node;
            cluster.BoundsMinimum=cluster.BoundsMaximum=vertices[sourceVertices.front()].Position;
            for (auto index:sourceVertices) Bounds(vertices[index].Position,cluster.BoundsMinimum,cluster.BoundsMaximum);
            header[Word(VirtualClusterWord::VertexCount)]=static_cast<std::uint32_t>(sourceVertices.size());
            header[Word(VirtualClusterWord::IndexCount)]=static_cast<std::uint32_t>(localIndices.size());
            header[Word(VirtualClusterWord::Vertices)]=start+VirtualGeometryClusterHeaderWords;
            header[Word(VirtualClusterWord::Indices)]=header[Word(VirtualClusterWord::Vertices)]+static_cast<std::uint32_t>(sourceVertices.size())*VirtualGeometryVertexWords;
            header[Word(VirtualClusterWord::Material)]=input.Material; header[Word(VirtualClusterWord::Cluster)]=static_cast<std::uint32_t>(candidate.Clusters.size());
            header[Word(VirtualClusterWord::Flags)]=input.SourceTriangles.empty()?0:VirtualGeometrySourceTriangleKeys;
            const float bounds[]{cluster.BoundsMinimum.X,cluster.BoundsMinimum.Y,cluster.BoundsMinimum.Z,cluster.BoundsMaximum.X,cluster.BoundsMaximum.Y,cluster.BoundsMaximum.Z};
            for (unsigned axis=0;axis<6;++axis) header[Word(VirtualClusterWord::MinimumX)+axis]=Bits(bounds[axis]);
            header[Word(VirtualClusterWord::SourceVertices)]=static_cast<std::uint32_t>(candidate.SourceVertices.size());
            header[Word(VirtualClusterWord::SourceTriangles)]=static_cast<std::uint32_t>(candidate.SourceTriangles.size());
            for (std::size_t vertex=0;vertex<sourceVertices.size();++vertex)
                std::memcpy(page.data()+header[Word(VirtualClusterWord::Vertices)]+vertex*VirtualGeometryVertexWords,&vertices[sourceVertices[vertex]],sizeof(Vertex));
            std::copy(localIndices.begin(),localIndices.end(),page.begin()+header[Word(VirtualClusterWord::Indices)]);
            candidate.SourceVertices.insert(candidate.SourceVertices.end(),sourceVertices.begin(),sourceVertices.end());
            if (input.SourceTriangles.empty()) candidate.SourceTriangles.insert(candidate.SourceTriangles.end(),input.Indices.size()/3,VirtualGeometryInvalid);
            else candidate.SourceTriangles.insert(candidate.SourceTriangles.end(),input.SourceTriangles.begin(),input.SourceTriangles.end());
            candidate.Clusters.push_back(cluster); page[2]+=static_cast<std::uint32_t>(payload); ++page[3];
        }
        candidate.Signature=Signature(candidate);
        if (!ValidateVirtualGeometryPages(candidate,error)) return false;
        output=std::move(candidate); error.clear(); return true;
    }

    bool BuildVirtualGeometryPages(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
        const std::vector<std::uint32_t>& triangleMaterials,const VirtualGeometryPageLimits& limits,VirtualGeometryPages& output,std::string& error)
    {
        if (!ValidateTriangleGeometry(vertices,indices,error)) return false;
        const GeometryPreparationLimits native; const std::size_t count=indices.size()/3;
        if (vertices.size()>std::min(limits.Source.MaximumVertices,native.MaximumVertices) || indices.size()>std::min(limits.Source.MaximumIndices,native.MaximumIndices) ||
            (!triangleMaterials.empty() && triangleMaterials.size()!=count))
        { error="Virtual geometry source exceeds its budget or has inconsistent material tags"; return false; }
        auto material=[&](std::uint32_t face){return triangleMaterials.empty()?0:triangleMaterials[face];};
        std::vector<Corner> corners; corners.reserve(indices.size()); std::vector<std::uint32_t> order;
        for (std::uint32_t face=0;face<count;++face)
        { order.push_back(face); for (unsigned corner=0;corner<3;++corner) corners.push_back({material(face),indices[face*3+corner],face}); }
        std::sort(corners.begin(),corners.end(),[](const Corner& a,const Corner& b){return std::tie(a.Material,a.Vertex,a.Triangle)<std::tie(b.Material,b.Vertex,b.Triangle);});
        std::sort(order.begin(),order.end(),[&](auto a,auto b){return std::make_pair(material(a),a)<std::make_pair(material(b),b);});
        std::vector<Adjacency> adjacency;
        for (std::size_t begin=0;begin<corners.size();)
        {
            std::size_t end=begin+1; const auto key=CornerKey(corners[begin].Material,corners[begin].Vertex);
            while (end<corners.size() && CornerKey(corners[end].Material,corners[end].Vertex)==key) ++end;
            adjacency.push_back({key,static_cast<std::uint32_t>(begin),static_cast<std::uint32_t>(end)}); begin=end;
        }
        std::vector<std::uint8_t> state(count); std::vector<std::uint32_t> frontier; std::size_t seed=0,read=0;
        std::vector<VirtualGeometryClusterInput> clusters;
        auto next=[&]() { while (seed<order.size() && state[order[seed]]) ++seed; return seed<order.size()?order[seed]:VirtualGeometryInvalid; };
        auto enqueue=[&](std::uint32_t face){if (!state[face]) { state[face]=1; frontier.push_back(face); }};
        while (read<frontier.size() || next()!=VirtualGeometryInvalid)
        {
            if (read==frontier.size()) { frontier.clear(); read=0; enqueue(next()); }
            if (clusters.size()>=limits.MaximumClusters) { error="Virtual geometry exceeded its cluster budget"; return false; }
            VirtualGeometryClusterInput cluster; cluster.Material=material(frontier[read]);
            while (cluster.SourceTriangles.size()<VirtualGeometryClusterTriangles)
            {
                if (read==frontier.size())
                {
                    const auto face=next(); if (face==VirtualGeometryInvalid || material(face)!=cluster.Material) break;
                    enqueue(face);
                }
                const auto face=frontier[read++]; state[face]=2; cluster.SourceTriangles.push_back(face);
                for (unsigned corner=0;corner<3;++corner)
                {
                    const auto vertex=indices[face*3+corner]; cluster.Indices.push_back(vertex); const auto key=CornerKey(cluster.Material,vertex);
                    auto range=std::lower_bound(adjacency.begin(),adjacency.end(),key,[](const Adjacency& a,std::uint64_t b){return a.Key<b;});
                    if (!range->Expanded)
                    { range->Expanded=true; for (auto entry=range->Begin;entry<range->End;++entry) enqueue(corners[entry].Triangle); }
                }
            }
            clusters.push_back(std::move(cluster));
        }
        return EncodeVirtualGeometryPages(vertices,static_cast<std::uint32_t>(count),clusters,limits,output,error);
    }

    bool DecodeVirtualGeometryCluster(const VirtualGeometryPages& data,std::uint32_t clusterIndex,VirtualGeometryDecodedCluster& output,std::string& error)
    {
        if (clusterIndex>=data.Clusters.size()) { error="Virtual cluster is outside the directory"; return false; }
        const auto& cluster=data.Clusters[clusterIndex];
        if (cluster.Page>=data.Pages.size() || !Finite(cluster.BoundsMinimum) || !Finite(cluster.BoundsMaximum)) { error="Virtual cluster page or bounds are invalid"; return false; }
        const auto& page=data.Pages[cluster.Page]; if (!PageHeader(page,error)) return false;
        if (cluster.Word<VirtualGeometryPageHeaderWords || std::uint64_t(cluster.Word)+VirtualGeometryClusterHeaderWords>page.Words[2])
        { error="Virtual cluster header is outside its page payload"; return false; }
        const auto* header=page.Words.data()+cluster.Word; const auto vertices=header[Word(VirtualClusterWord::VertexCount)],indices=header[Word(VirtualClusterWord::IndexCount)];
        const auto vertexStart=header[Word(VirtualClusterWord::Vertices)],indexStart=header[Word(VirtualClusterWord::Indices)];
        const auto sourceVertices=header[Word(VirtualClusterWord::SourceVertices)],sourceTriangles=header[Word(VirtualClusterWord::SourceTriangles)];
        if (!vertices || vertices>VirtualGeometryMaximumClusterVertices || !indices || indices%3 || indices>VirtualGeometryClusterTriangles*3 ||
            vertexStart!=cluster.Word+VirtualGeometryClusterHeaderWords || std::uint64_t(vertexStart)+std::uint64_t(vertices)*VirtualGeometryVertexWords!=indexStart ||
            std::uint64_t(indexStart)+indices>page.Words[2] || std::uint64_t(sourceVertices)+vertices>data.SourceVertices.size() ||
            std::uint64_t(sourceTriangles)+indices/3>data.SourceTriangles.size() || header[Word(VirtualClusterWord::Material)]!=cluster.Material ||
            header[Word(VirtualClusterWord::Cluster)]!=clusterIndex || (header[Word(VirtualClusterWord::Flags)]&~VirtualGeometrySourceTriangleKeys) || header[Word(VirtualClusterWord::Reserved)])
        { error="Virtual cluster dimensions, offsets or directory identity are invalid"; return false; }
        Float3 lo{Scalar(header[Word(VirtualClusterWord::MinimumX)]),Scalar(header[Word(VirtualClusterWord::MinimumY)]),Scalar(header[Word(VirtualClusterWord::MinimumZ)])};
        Float3 hi{Scalar(header[Word(VirtualClusterWord::MaximumX)]),Scalar(header[Word(VirtualClusterWord::MaximumY)]),Scalar(header[Word(VirtualClusterWord::MaximumZ)])};
        if (!Same(lo,cluster.BoundsMinimum) || !Same(hi,cluster.BoundsMaximum)) { error="Virtual cluster header bounds disagree with its directory"; return false; }
        VirtualGeometryDecodedCluster candidate; candidate.Material=cluster.Material; candidate.Node=cluster.Node;
        candidate.Vertices.resize(vertices); std::memcpy(candidate.Vertices.data(),page.Words.data()+vertexStart,std::size_t(vertices)*sizeof(Vertex));
        candidate.Indices.assign(page.Words.begin()+indexStart,page.Words.begin()+indexStart+indices);
        if (!ValidateTriangleGeometry(candidate.Vertices,candidate.Indices,error)) return false;
        candidate.SourceVertices.assign(data.SourceVertices.begin()+sourceVertices,data.SourceVertices.begin()+sourceVertices+vertices);
        candidate.SourceTriangles.assign(data.SourceTriangles.begin()+sourceTriangles,data.SourceTriangles.begin()+sourceTriangles+indices/3);
        auto keys=candidate.SourceVertices; std::sort(keys.begin(),keys.end());
        if (keys.back()>=data.SourceVertexCount || std::adjacent_find(keys.begin(),keys.end())!=keys.end()) { error="Virtual cluster source vertex keys are invalid or duplicated"; return false; }
        const bool keyed=(header[Word(VirtualClusterWord::Flags)]&VirtualGeometrySourceTriangleKeys)!=0;
        for (auto key:candidate.SourceTriangles) if (keyed?key>=data.SourceTriangleCount:key!=VirtualGeometryInvalid)
        { error="Virtual cluster source triangle keys disagree with its mapping flags"; return false; }
        if (keyed)
        {
            auto triangles=candidate.SourceTriangles; std::sort(triangles.begin(),triangles.end());
            if (std::adjacent_find(triangles.begin(),triangles.end())!=triangles.end()) { error="A virtual cluster repeats a source triangle key"; return false; }
        }
        Float3 actualLo=candidate.Vertices.front().Position,actualHi=actualLo;
        for (const auto& vertex:candidate.Vertices) Bounds(vertex.Position,actualLo,actualHi);
        if (!Same(lo,actualLo) || !Same(hi,actualHi)) { error="Virtual cluster bounds do not match its decoded positions"; return false; }
        output=std::move(candidate); error.clear(); return true;
    }

    bool ValidateVirtualGeometryPages(const VirtualGeometryPages& data,std::string& error)
    {
        const GeometryPreparationLimits native;
        if (!data.SourceVertexCount || data.SourceVertexCount>native.MaximumVertices || !data.SourceTriangleCount ||
            std::uint64_t(data.SourceTriangleCount)*3>native.MaximumIndices || data.Pages.empty() || data.Clusters.empty() ||
            data.Pages.size()>std::numeric_limits<std::uint32_t>::max()/VirtualGeometryPageWords || data.Clusters.size()>std::numeric_limits<std::uint32_t>::max() ||
            data.SourceVertices.size()>std::numeric_limits<std::uint32_t>::max() || data.SourceTriangles.size()>std::numeric_limits<std::uint32_t>::max())
        { error="Virtual dataset has invalid source or encoded dimensions"; return false; }
        std::vector<std::uint8_t> seen(data.SourceTriangleCount); std::size_t directory=0,vertexKeys=0,triangleKeys=0;
        for (std::uint32_t pageIndex=0;pageIndex<data.Pages.size();++pageIndex)
        {
            const auto& page=data.Pages[pageIndex]; if (!PageHeader(page,error)) return false; std::uint32_t word=VirtualGeometryPageHeaderWords,clusters=0;
            while (directory<data.Clusters.size() && data.Clusters[directory].Page==pageIndex)
            {
                if (data.Clusters[directory].Word!=word) { error="Virtual cluster directory has a gap, overlap or invalid page order"; return false; }
                VirtualGeometryDecodedCluster decoded; if (!DecodeVirtualGeometryCluster(data,static_cast<std::uint32_t>(directory),decoded,error)) return false;
                const auto* header=page.Words.data()+word;
                if (header[Word(VirtualClusterWord::SourceVertices)]!=vertexKeys || header[Word(VirtualClusterWord::SourceTriangles)]!=triangleKeys)
                { error="Virtual source-key tables have a gap or overlap"; return false; }
                for (auto key:decoded.SourceTriangles) if (key!=VirtualGeometryInvalid)
                { if (seen[key]) { error="A virtual leaf repeats a source triangle"; return false; } seen[key]=1; }
                vertexKeys+=decoded.SourceVertices.size(); triangleKeys+=decoded.SourceTriangles.size();
                word=header[Word(VirtualClusterWord::Indices)]+header[Word(VirtualClusterWord::IndexCount)]; ++directory; ++clusters;
            }
            if (word!=page.Words[2] || clusters!=page.Words[3]) { error="Virtual page payload or cluster count is inconsistent"; return false; }
            for (std::size_t padding=word;padding<page.Words.size();++padding) if (page.Words[padding]) { error="Virtual page padding must be zero"; return false; }
        }
        if (directory!=data.Clusters.size() || vertexKeys!=data.SourceVertices.size() || triangleKeys!=data.SourceTriangles.size() ||
            std::find(seen.begin(),seen.end(),0)!=seen.end()) { error="Virtual dataset omits source triangles or leaves unowned directory/key data"; return false; }
        if (data.Signature!=Signature(data)) { error="Virtual dataset content signature is stale"; return false; }
        error.clear(); return true;
    }
}
