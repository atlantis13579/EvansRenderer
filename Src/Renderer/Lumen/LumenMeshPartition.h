#pragma once

#include "Renderer/Types.h"
#include "Maths/AABB.h"

#include <functional>

namespace Evans::Renderer
{
    inline constexpr unsigned LumenPartitionMaximumParts=1024;
    inline constexpr std::size_t LumenPartitionMaximumTriangles=2*1024*1024;
    struct LumenMeshPartitionSettings { unsigned MaximumParts=64; };
    struct LumenMeshPart
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
        std::vector<std::uint32_t> SourceVertices,SourceTriangles;
        Maths::AABB Bounds;
        unsigned Components=0;
        double SurfaceArea=0;
    };
    struct LumenMeshPartition
    {
        std::vector<LumenMeshPart> Parts;
        std::vector<std::uint32_t> TriangleParts;
        unsigned ConnectedComponents=0;
    };
    using LumenPartitionCancellation=std::function<bool()>;

    // Preprocessing only: local positions identify edge-connected components;
    // original vertex attributes, winding and triangle identities stay intact.
    // A bounded spatial partition merges whole components when necessary. It
    // never cuts a closed shell into open pieces to satisfy a proxy budget.
    // Exact positional welding joins UV/normal seams without closing small gaps.
    // Unused vertices are omitted. All indexed triangles, including degenerate
    // ones, remain addressable. Failure/cancellation leaves output unchanged.
    bool BuildLumenMeshPartition(const std::vector<Vertex>& vertices,const std::vector<std::uint32_t>& indices,
        const LumenMeshPartitionSettings& settings,LumenMeshPartition& output,std::string& error,
        const LumenPartitionCancellation& cancelled={});
}
