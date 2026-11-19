#pragma once

#include "Renderer/Geometry/GeometryPreparation.h"
#include "Renderer/Geometry/LevelOfDetail.h"

#include <limits>

namespace Evans::Renderer
{
    inline constexpr std::uint32_t NaniteInvalid=std::numeric_limits<std::uint32_t>::max();
    struct NaniteHierarchySettings
    {
        GeometryPreparationLimits Source;
        LodBuildSettings Simplification; // One reduction per group; MaximumLevels is ignored.
        std::uint32_t ClusterTriangles=128,ClusterVertices=256,ClustersPerGroup=8,MaximumLevels=24;
        std::size_t MaximumClusters=256*1024,MaximumGroups=128*1024,MaximumStoredIndices=12'000'000;
    };
    struct NaniteCluster
    {
        std::vector<std::uint32_t> Indices; // Source endpoints; attributes remain exact.
        std::vector<std::uint32_t> SourceTriangles; // Present only on lossless leaves.
        std::uint32_t Material=0,Level=0,GeneratingGroup=NaniteInvalid,ParentGroup=NaniteInvalid;
        Float3 BoundsMinimum,BoundsMaximum;
    };
    struct NaniteClusterGroup
    {
        std::vector<std::uint32_t> Children,Parents;
        std::uint32_t Level=0,Material=0;
        Float3 BoundsMinimum,BoundsMaximum; // Contains every child's inherited LOD bounds.
        double LocalDisplacement=0,MaximumDisplacement=0;
    };
    struct NaniteHierarchy
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> SourceIndices,SourceMaterials;
        std::vector<NaniteCluster> Clusters;
        std::vector<NaniteClusterGroup> Groups;
        std::vector<std::uint32_t> Roots;
        std::uint32_t ClusterTriangleLimit=0,ClusterVertexLimit=0,GroupClusterLimit=0;
        std::uint64_t Signature=0;
    };
    // Karis, Stubbe and Wihlidal, "A Deep Dive into Nanite Virtualized Geometry"
    // (SIGGRAPH 2021): cluster -> group -> simplify -> recluster. Repartitioning
    // makes this a many-to-many replacement DAG, rather than a fixed-region tree.
    // This preprocessing format is independent of Unreal and of the existing
    // VirtualGeometry page format. NanitePages adds compressed group bundles;
    // native visibility and rasterization consume the DAG separately.
    // Failed builds preserve output. Irreducible groups retain their fine clusters.
    [[nodiscard]] bool BuildNaniteHierarchy(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices,const std::vector<std::uint32_t>& materials,
        const NaniteHierarchySettings& settings,NaniteHierarchy& output,std::string& error);
    [[nodiscard]] bool ValidateNaniteHierarchy(const NaniteHierarchy& data,std::string& error);
}
