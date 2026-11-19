#include "Renderer/Nanite/NaniteInstances.h"

#include <algorithm>
#include <cstring>
#include <map>

namespace Evans::Renderer
{
    bool BuildNaniteInstanceWork(const VisibilityBufferDesc& frame,
        const std::vector<NaniteVisibilityGeometryConstants>& constants,NaniteInstanceWork& output,std::string& error)
    {
        NaniteInstanceWork result;
        if (!frame.GpuInstances || !frame.Scene || !frame.Scene->HasNativeGeometry())
        { output=std::move(result); error.clear(); return true; }
        if (!frame.NaniteRaster) { error="Nanite instance scheduling requires the hybrid raster producer"; return false; }
        if (!ValidateVisibilityScene(*frame.Scene,error)) return false;
        const auto& scene=*frame.Scene;
        if (constants.size()!=scene.NativeMeshes.size()) { error="Nanite instance constants do not match native sources"; return false; }
        std::vector<ShaderUInt4> queries(constants.size());
        for (const auto& draw:frame.Draws)
        {
            if (draw.Range>=queries.size()) { error="Nanite instance draw is outside the source table"; return false; }
            queries[draw.Range].Z=1;
        }
        if (bool(frame.NaniteVisibility)!=!frame.NaniteClusters.empty())
        { error="Nanite instance culling requires matching cluster ranges"; return false; }
        for (unsigned i=0;i<frame.NaniteClusters.size();++i)
        {
            const auto& cluster=frame.NaniteClusters[i];
            if (cluster.Range>=queries.size() || !queries[cluster.Range].Z)
            { error="Nanite instance cluster has no submitted source"; return false; }
            auto& query=queries[cluster.Range]; const auto& range=scene.Draws[cluster.Range];
            if (!query.Y) query.X=i;
            if (query.X+query.Y!=i || cluster.FirstTriangle!=range.FirstTriangle+query.W || !cluster.TriangleCount ||
                cluster.TriangleCount>range.TriangleCount-query.W)
            { error="Nanite instance clusters must partition each source in order"; return false; }
            ++query.Y; query.W+=cluster.TriangleCount;
        }
        std::map<std::pair<std::uint64_t,std::uint64_t>,unsigned> families;
        std::size_t totalWords=0;
        for (unsigned i=0;i<constants.size();++i)
        {
            const auto& mesh=scene.NativeMeshes[i]; const auto& range=scene.Draws[i];
            const auto& c=constants[i]; auto& query=queries[i];
            if (c.Range.X!=range.FirstTriangle || c.Range.Y!=range.TriangleCount || c.Range.Z!=mesh.FirstIndex || c.Dispatch.Z!=range.Material ||
                (frame.NaniteVisibility && query.Z && query.W!=range.TriangleCount))
            { error="Nanite instance metadata does not cover the native source"; return false; }
            const auto key=std::make_pair(mesh.VertexBuffer.Value,mesh.IndexBuffer.Value);
            const auto inserted=families.emplace(key,unsigned(result.Batches.size()));
            if (inserted.second) { NaniteInstanceBatch batch; batch.Vertex=mesh.VertexBuffer; batch.Index=mesh.IndexBuffer; result.Batches.push_back(std::move(batch)); }
            result.Batches[inserted.first->second].Sources.push_back(i);
        }
        for (auto& batch:result.Batches)
        {
            const auto count=batch.Sources.size(); const auto words=NaniteInstanceHeaderWords+(NaniteInstanceRecordWords+1)*count;
            totalWords+=words;
            if (totalWords>NaniteInstanceMaximumBytes/sizeof(unsigned)) { error="Nanite instance schedule exceeds the memory budget"; return false; }
            batch.Upload.resize(words,0); batch.Parameters={unsigned(count),unsigned(NaniteInstanceHeaderWords+NaniteInstanceRecordWords*count),NaniteInstanceHeaderWords,unsigned(bool(frame.NaniteVisibility))};
            unsigned groups=0;
            for (unsigned local=0;local<count;++local)
            {
                const unsigned source=batch.Sources[local]; auto* record=batch.Upload.data()+NaniteInstanceHeaderWords+local*NaniteInstanceRecordWords;
                std::memcpy(record,&constants[source],sizeof(constants[source]));
                auto query=queries[source]; query.W=0; std::memcpy(record+sizeof(constants[source])/sizeof(unsigned),&query,sizeof(query));
                groups+=(constants[source].Range.Y+NaniteRasterThreads-1)/NaniteRasterThreads;
            }
            batch.MaximumGroupsX=std::min(groups,NaniteRasterDispatchColumns);
            batch.MaximumGroupsY=(groups+batch.MaximumGroupsX-1)/batch.MaximumGroupsX;
        }
        output=std::move(result); error.clear(); return true;
    }
}
