#pragma once

#include "Renderer/Nanite/NaniteRaster.h"

namespace Evans::Renderer
{
    inline constexpr unsigned NaniteInstanceHeaderWords=4, NaniteInstanceRecordWords=44;
    inline constexpr std::size_t NaniteInstanceMaximumBytes=128ull*1024*1024;
    struct NaniteInstanceBatch
    {
        BufferHandle Vertex,Index;
        ShaderUInt4 Parameters; // Source count, prefix offset, record offset, culling enabled.
        std::vector<unsigned> Sources,Upload;
        unsigned MaximumGroupsX=0,MaximumGroupsY=0;
    };
    struct NaniteInstanceWork { std::vector<NaniteInstanceBatch> Batches; };
    // Shared decoded streams form a binding family, even across nonadjacent
    // instances/materials. Stable destination ranges keep primitive IDs intact.
    // GPU counts include MainVisible and Deferred: history cannot discard work
    // needed by the post-occlusion phase. A prefix scan emits indirect dispatch
    // dimensions; assembly binary-searches the inclusive ends (including zeros).
    [[nodiscard]] bool BuildNaniteInstanceWork(const VisibilityBufferDesc&,
        const std::vector<NaniteVisibilityGeometryConstants>&,NaniteInstanceWork&,std::string&);
}
