#pragma once

#include "Renderer/VirtualGeometry/VirtualGeometryHierarchy.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    inline constexpr std::uint32_t VirtualGeometryGroupWidth=64,VirtualGeometryNodeWords=20,VirtualGeometryViewWords=36,VirtualGeometryClusterWords=8;
    inline constexpr std::uint32_t VirtualGeometryNodeStateWords=8,VirtualGeometryClusterStateWords=4,VirtualGeometryBlockWords=4,VirtualGeometryStatusWords=4;
    inline constexpr std::uint32_t VirtualGeometryPreviousSelection=1;
    inline constexpr std::uint32_t VirtualGeometryMissingPage=1,VirtualGeometryStalePage=2,VirtualGeometryBadPayload=4,VirtualGeometryOutputOverflow=8;
    enum class VirtualGeometryGpuPass : unsigned { Evaluate,Propagate,Select,ClusterScan,BlockScan,Decode,Count };
    struct alignas(16) VirtualGeometryGpuConstants
    {
        ShaderUInt4 Counts; // nodes, views, virtual pages, physical slots
        ShaderUInt4 Offsets; // node, child, cluster-reference, page-reference word offsets
        ShaderUInt4 Tables; // cluster directory, views, height order, metadata word count
        ShaderUInt4 Dispatch; // first height-order element, invocation count, group count X, reserved
        ShaderUInt4 Limits; // clusters, state words, physical words, reserved
        ShaderUInt4 Capacity; // maximum drawable vertices/indices, source triangles, reserved
        ShaderUInt4 Identity; // cache epoch low/high, revision low/high
        Float4 Quality; // downward-rounded refine/coarsen pixel limits, reserved XY
    };
    static_assert(sizeof(VirtualGeometryGpuConstants)==128 && offsetof(VirtualGeometryGpuConstants,Quality)==112,"Virtual geometry frame ABI is eight vectors");
    struct VirtualGeometryGpuLevel { std::uint32_t First=0,Count=0; };
    struct VirtualGeometryGpuUpload
    {
        VirtualGeometryGpuConstants Constants;
        std::vector<std::uint32_t> Metadata,InitialState,ClusterIds;
        std::vector<ShaderUInt4> PageTable; // slot, generation, epoch low/high
        std::vector<VirtualGeometryPage> PhysicalPages;
        std::vector<VirtualGeometryGpuLevel> Levels; // includes height zero
        std::uint64_t Source=0;
    };
    struct VirtualGeometryGpuReadback
    {
        std::vector<std::uint32_t> State;
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
    };
    struct VirtualGeometryGpuDispatch
    {
        VirtualGeometryGpuConstants Constants;
        VirtualGeometryGpuPass Pass=VirtualGeometryGpuPass::Evaluate;
        std::uint32_t GroupsX=0,GroupsY=0,Reserved=0;
    };
    static_assert(sizeof(VirtualGeometryGpuDispatch)==144,"Virtual dispatch records have explicit vector alignment");
    struct VirtualGeometryMaterialRange { std::uint32_t Material=0,FirstIndex=0,IndexCount=0; };
    struct VirtualGeometryGpuSelection
    {
        std::uint32_t VertexCount=0,IndexCount=0,ActiveClusters=0;
        std::vector<std::uint32_t> Nodes;
        std::vector<VirtualGeometryMaterialRange> Materials;
    };
    struct VirtualGeometryBuffers
    {
        BufferHandle VertexBuffer,IndexBuffer;
        VirtualGeometryGpuReadback Readback; // authoritative mirrors of the drawable allocations
        VirtualGeometryGpuSelection Selection;
        unsigned DispatchCount=0;
    };

    // Six storage buffers: metadata, page table, physical pages, mutable state,
    // drawable vertices and drawable indices. Nodes occupy five vectors; cluster
    // records occupy two. Material-sorted cluster ordinals produce contiguous
    // draw ranges while retaining original page-directory IDs for validation.
    // View rows carry coefficient uncertainty. Shader interval bounds use L1
    // displacement/screen radii, conservative under nonuniform scale and shear.
    // Near-plane uncertainty requests refinement. Vertices are copied losslessly.
    [[nodiscard]] bool PrepareVirtualGeometryGpu(const VirtualGeometryHierarchy& hierarchy,const VirtualGeometryResidency& residency,
        const std::vector<LodView>& views,const VirtualGeometryCutSettings& settings,const std::vector<std::uint32_t>& previousCut,
        const GeometryPreparationLimits& limits,VirtualGeometryGpuUpload& output,std::string& error);
    [[nodiscard]] bool BuildVirtualGeometryGpuDispatch(const VirtualGeometryGpuUpload& upload,VirtualGeometryGpuPass pass,unsigned height,
        VirtualGeometryGpuConstants& constants,std::uint32_t& groupsX,std::uint32_t& groupsY) noexcept;
    [[nodiscard]] bool ValidateVirtualGeometryGpuUpload(const VirtualGeometryGpuUpload& upload,std::string& error);
    [[nodiscard]] bool BuildVirtualGeometryGpuSchedule(const VirtualGeometryGpuUpload& upload,std::vector<VirtualGeometryGpuDispatch>& output,std::string& error);
    // Validate native feedback and stream sizes without running CPU selection or
    // decoding. These checks gate publication; they never repair a failed job.
    [[nodiscard]] bool ValidateVirtualGeometryGpuState(const VirtualGeometryGpuUpload& upload,const std::vector<std::uint32_t>& state,
        VirtualGeometryGpuSelection& output,std::string& error);
    [[nodiscard]] bool ValidateVirtualGeometryGpuReadback(const VirtualGeometryGpuUpload& upload,const VirtualGeometryGpuReadback& readback,
        VirtualGeometryGpuSelection& output,std::string& error);
    [[nodiscard]] bool AdmitVirtualGeometryGpuFeedback(const VirtualGeometryHierarchy& hierarchy,const VirtualGeometryResidency& residency,
        const VirtualGeometryGpuUpload& upload,const std::vector<std::uint32_t>& state,VirtualGeometryCut& output,std::string& error);
}
