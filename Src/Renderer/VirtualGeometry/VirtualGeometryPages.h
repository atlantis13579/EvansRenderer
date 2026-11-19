#pragma once

#include "Renderer/Geometry/GeometryPreparation.h"

#include <array>
#include <limits>

namespace Evans::Renderer
{
    inline constexpr std::uint32_t VirtualGeometryInvalid=std::numeric_limits<std::uint32_t>::max();
    inline constexpr std::uint32_t VirtualGeometryPageBytes=8*1024;
    inline constexpr std::uint32_t VirtualGeometryPageWords=VirtualGeometryPageBytes/sizeof(std::uint32_t);
    inline constexpr std::uint32_t VirtualGeometryPageHeaderWords=8,VirtualGeometryClusterHeaderWords=16;
    inline constexpr std::uint32_t VirtualGeometryVertexWords=12,VirtualGeometryClusterTriangles=48;
    inline constexpr std::uint32_t VirtualGeometryMaximumClusterVertices=VirtualGeometryClusterTriangles*3;
    inline constexpr std::uint32_t VirtualGeometryPageMagic=std::uint32_t('E')|(std::uint32_t('V')<<8)|(std::uint32_t('G')<<16)|(std::uint32_t('P')<<24);
    inline constexpr std::uint32_t VirtualGeometryPageVersion=1,VirtualGeometryPinnedPage=1,VirtualGeometrySourceTriangleKeys=1;
    static_assert(sizeof(Vertex)==VirtualGeometryVertexWords*sizeof(std::uint32_t),"Virtual pages require the canonical packed vertex");
    static_assert(VirtualGeometryPageHeaderWords+VirtualGeometryClusterHeaderWords+
        VirtualGeometryMaximumClusterVertices*VirtualGeometryVertexWords+VirtualGeometryClusterTriangles*3<=VirtualGeometryPageWords,
        "Even an unshared triangle cluster must fit in one page");

    // Page header: magic, version, used words, cluster count, flags, reserved XYZ.
    // Runtime physical copies may replace reserved X/Y with page ID/generation;
    // source pages remain immutable. No quantization or cross-API shader input.
    enum class VirtualClusterWord : unsigned
    {
        VertexCount,IndexCount,Vertices,Indices,Material,Cluster,Flags,Reserved,
        MinimumX,MinimumY,MinimumZ,MaximumX,MaximumY,MaximumZ,SourceVertices,SourceTriangles
    };
    struct VirtualGeometryPage { std::array<std::uint32_t,VirtualGeometryPageWords> Words{}; };
    static_assert(sizeof(VirtualGeometryPage)==VirtualGeometryPageBytes,"Physical page pools require an exact page stride");
    struct VirtualGeometryCluster
    {
        std::uint32_t Page=0,Word=0,Material=0,Node=VirtualGeometryInvalid;
        Float3 BoundsMinimum,BoundsMaximum;
    };
    struct VirtualGeometryClusterInput
    {
        std::vector<std::uint32_t> Indices; // original vertex IDs, at most 48 triangles
        std::vector<std::uint32_t> SourceTriangles; // empty for a coarser representation
        std::uint32_t Material=0,Node=VirtualGeometryInvalid;
        bool Pinned=false;
    };
    struct VirtualGeometryPages
    {
        std::uint32_t SourceVertexCount=0,SourceTriangleCount=0;
        std::vector<VirtualGeometryPage> Pages;
        std::vector<VirtualGeometryCluster> Clusters;
        // Diagnostic/source keys stay on the CPU, outside the streamed payload.
        std::vector<std::uint32_t> SourceVertices,SourceTriangles;
        std::uint64_t Signature=0; // deterministic content identity, not a security hash
    };
    struct VirtualGeometryPageLimits
    {
        GeometryPreparationLimits Source;
        std::size_t MaximumPages=16*1024,MaximumClusters=256*1024;
    };
    struct VirtualGeometryDecodedCluster
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices,SourceVertices,SourceTriangles;
        std::uint32_t Material=0,Node=VirtualGeometryInvalid;
    };

    // This is the page/cluster data layer, not a virtualized draw implementation.
    // Hierarchical selection and streaming draw production remain separate.
    // References for the later architecture: Hoppe, "View-dependent refinement
    // of progressive meshes", SIGGRAPH 1997, https://hhoppe.com/proj/vdrpm/;
    // Karis, Stubbe and Wihlidal, "A Deep Dive into Nanite Virtualized Geometry",
    // SIGGRAPH 2021, https://www.wihlidal.com/projects/nanite-deepdive/.
    //
    // Partitioning expands each (material, vertex) adjacency once and retains a
    // frontier across cluster boundaries. High-valence input cannot turn every
    // new cluster into another full scan of the same vertex fan. Seeds, corners
    // and material order are deterministic; attributes and winding are unchanged.
    [[nodiscard]] bool BuildVirtualGeometryPages(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices,const std::vector<std::uint32_t>& triangleMaterials,
        const VirtualGeometryPageLimits& limits,VirtualGeometryPages& output,std::string& error);
    // Hierarchy builders may provide coarse clusters without source-triangle
    // keys, followed by lossless leaves. Every source triangle key must occur once
    // among keyed leaves. A page never mixes pinned and evictable clusters.
    [[nodiscard]] bool EncodeVirtualGeometryPages(const std::vector<Vertex>& vertices,std::uint32_t sourceTriangleCount,
        const std::vector<VirtualGeometryClusterInput>& clusters,const VirtualGeometryPageLimits& limits,
        VirtualGeometryPages& output,std::string& error);
    [[nodiscard]] bool ValidateVirtualGeometryPages(const VirtualGeometryPages& data,std::string& error);
    // Local decoding validates the selected page/cluster, not unrelated pages.
    // Full validation is required when accepting a new immutable dataset.
    [[nodiscard]] bool DecodeVirtualGeometryCluster(const VirtualGeometryPages& data,std::uint32_t cluster,
        VirtualGeometryDecodedCluster& output,std::string& error);
}
