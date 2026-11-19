#pragma once

#include "Renderer/Nanite/NaniteStreaming.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    inline constexpr unsigned NaniteGpuWidth=64,NaniteGpuGroupWords=16,NaniteGpuClusterWords=16,NaniteGpuViewWords=36;
    inline constexpr unsigned NaniteGpuGroupStateWords=4,NaniteGpuClusterStateWords=4,NaniteGpuBlockWords=4,NaniteGpuStatusWords=4;
    inline constexpr unsigned NaniteGpuPageIdentityWords=4,NaniteGpuPageWords=NanitePageWords+NaniteGpuPageIdentityWords;
    inline constexpr unsigned NaniteGpuPageBytes=NaniteGpuPageWords*sizeof(std::uint32_t);
    inline constexpr unsigned NaniteGpuStalePage=1,NaniteGpuBadPayload=2,NaniteGpuOverflow=4,NaniteGpuMissingPage=8;
    enum class NaniteGpuPass : unsigned { Evaluate,Refine,Clusters,Blocks,Decode,Count };
    struct alignas(16) NaniteGpuConstants
    {
        ShaderUInt4 Counts; // groups, views, logical pages, physical slots
        ShaderUInt4 Offsets; // groups, child references, parent references, clusters
        ShaderUInt4 Tables; // views, height order, metadata words, cluster count
        ShaderUInt4 Dispatch; // first height-order entry, invocation count, workgroups X, reserved
        ShaderUInt4 Limits; // state words, physical words, prefix blocks, position exponent bits
        ShaderUInt4 Capacity; // maximum vertices/indices, source triangles, reserved
        ShaderUInt4 Identity; // cache epoch low/high, revision low/high
        Float4 Quality; // downward-rounded pixel threshold, quantization displacement, reserved XY
    };
    static_assert(sizeof(NaniteGpuConstants)==128,"Nanite GPU constants have eight aligned vectors");
    struct NaniteGpuPhysicalPage { ShaderUInt4 Identity; NanitePage Payload; };
    static_assert(sizeof(NaniteGpuPhysicalPage)==NaniteGpuPageWords*sizeof(std::uint32_t),"Nanite physical identity precedes each fixed payload");
    struct NaniteGpuLevel { std::uint32_t First=0,Count=0; };
    struct NaniteGpuUpload
    {
        NaniteGpuConstants Constants;
        std::vector<std::uint32_t> Metadata,InitialState,ClusterIds;
        std::vector<ShaderUInt4> PageTable;
        std::vector<NaniteGpuPhysicalPage> PhysicalPages;
        std::vector<NaniteGpuLevel> Levels;
        std::uint64_t Source=0,Signature=0;
        bool ReadbackGeometry=false; // Diagnostic copy policy; not part of GPU input identity.
    };
    struct NaniteGpuDispatch
    {
        NaniteGpuConstants Constants;
        NaniteGpuPass Pass=NaniteGpuPass::Evaluate;
        std::uint32_t GroupsX=0,GroupsY=0,Reserved=0;
    };
    struct NaniteGpuMaterialRange { std::uint32_t Material=0,FirstIndex=0,IndexCount=0; };
    struct NaniteGpuSelection
    {
        std::uint32_t VertexCount=0,IndexCount=0,ActiveClusters=0;
        std::vector<std::uint32_t> Clusters,RefinedGroups;
        std::vector<NaniteGpuMaterialRange> Materials;
    };
    struct NaniteGpuReadback
    {
        std::vector<std::uint32_t> State;
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
    };
    struct NaniteGeometryTask
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
    };
    struct NaniteAsyncPlan
    {
        std::vector<NaniteGpuDispatch> Dispatches;
        std::size_t VertexBytes=0,IndexBytes=0,ReservedBytes=0;
    };
    // Resident clusters bound every possible cut, so prefix and decode can share
    // one submission. Poll normally reads selection state only. Explicit geometry
    // diagnostics retain the full-copy path; neither mode waits inside a poll.
    [[nodiscard]] bool BuildNaniteAsyncPlan(const NaniteGpuUpload&,NaniteAsyncPlan&,std::string&);
    NaniteGeometryTask NewNaniteGeometryTask() noexcept;
    struct NaniteGeometryBuffers
    {
        BufferHandle VertexBuffer,IndexBuffer;
        NaniteGpuReadback Readback;
        NaniteGpuSelection Selection;
        unsigned DispatchCount=0;
        [[nodiscard]] std::uint64_t ReadbackByteCount() const noexcept
        { return std::uint64_t(Readback.State.size()+Readback.Indices.size())*sizeof(std::uint32_t)+std::uint64_t(Readback.Vertices.size())*sizeof(Vertex); }
    };

    // Separate level dispatches order DAG dependencies without device-wide
    // barriers inside a kernel. Evaluation propagates shared error upward;
    // refinement propagates complete ancestor decisions downward. Material
    // ordered prefix scans produce exact drawable allocations, then each
    // cluster workgroup decodes its random-access compressed bit fields.
    // CPU routines prepare/validate protocol data; they do not replace GPU work.
    [[nodiscard]] bool PrepareNaniteGpu(const NaniteHierarchy&,const NanitePages&,const VirtualGeometryResidency&,
        const std::vector<LodView>&,double maximumErrorPixels,const GeometryPreparationLimits&,NaniteGpuUpload&,std::string& error);
    [[nodiscard]] bool ValidateNaniteGpuUpload(const NaniteGpuUpload&,std::string& error);
    [[nodiscard]] bool BuildNaniteGpuSchedule(const NaniteGpuUpload&,std::vector<NaniteGpuDispatch>&,std::string& error);
    [[nodiscard]] bool ValidateNaniteGpuState(const NaniteGpuUpload&,const std::vector<std::uint32_t>&,NaniteGpuSelection&,std::string& error);
    [[nodiscard]] bool ValidateNaniteGpuReadback(const NaniteGpuUpload&,const NaniteGpuReadback&,NaniteGpuSelection&,std::string& error);
    // Native errors drive admission; CPU projection is used only to report the
    // remaining leaf quantization floor. The actual resident cut must agree
    // with authoritative GPU feedback before requests can be published.
    [[nodiscard]] bool AdmitNaniteGpuFeedback(const NaniteHierarchy&,const NanitePages&,const VirtualGeometryResidency&,
        const std::vector<LodView>&,const NaniteGpuUpload&,const std::vector<std::uint32_t>&,NaniteResidentCut&,std::string& error);
}
