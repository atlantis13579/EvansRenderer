#pragma once

#include "Renderer/Geometry/Instancing.h"

namespace Evans::Renderer
{
    // One workgroup owns one compatible material/geometry batch. Homogeneous
    // AABB tests produce visibility bits; a stable scan compacts the existing
    // instance ABI and writes an indexed indirect command. No count readback
    // is needed to submit the draw. CPU-selected LOD and batch order remain.
    // Harris et al., "Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3, ch. 39:
    // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    //
    // Dispatch before opening the graphics pass (required by Vulkan/Metal).
    // Each pending draw owns its output until GPU completion. Make compute
    // writes visible to vertex uniforms and indirect-command reads; DX11 must
    // copy the UAV output into a separate constant buffer after unbinding it.
    // Sorted transparency, selection outlines and path tracing are not eligible.
    // The frame token prevents an indirect draw from reusing a previous frame's
    // allocation after its storage has been recycled.
    struct GpuDrivenBatchHandle
    {
        std::uint64_t Frame = 0;
        std::uint32_t Index = 0;
    };
    struct GpuDrivenCullConstants
    {
        Matrix4 CullingViewProjection;
        Matrix4 RenderViewProjection;
        Float4 BoundsMinimum;
        Float4 BoundsMaximum;
        Float4 Material;
        ShaderUInt4 Draw; // Index count, first index, signed base-vertex bits, candidate count.
        ShaderUInt4 Control; // Frustum test enabled, reserved YZW.
    };
    static_assert(sizeof(GpuDrivenCullConstants) == 208 &&
        offsetof(GpuDrivenCullConstants, Material) == 160 &&
        offsetof(GpuDrivenCullConstants, Draw) == 176 &&
        offsetof(GpuDrivenCullConstants, Control) == 192, "GPU visibility uniform ABI changed");

    // D3D11/12, Vulkan, ES 3.1 and Metal share these five packed words. ES calls
    // the last word reservedMustBeZero; all consumers use a zero first instance.
    struct GpuDrivenIndexedArguments
    {
        std::uint32_t IndexCount = 0;
        std::uint32_t InstanceCount = 0;
        std::uint32_t FirstIndex = 0;
        std::int32_t BaseVertex = 0;
        std::uint32_t FirstInstance = 0;
    };
    static_assert(sizeof(GpuDrivenIndexedArguments) == 20 &&
        offsetof(GpuDrivenIndexedArguments, BaseVertex) == 12 &&
        offsetof(GpuDrivenIndexedArguments, FirstInstance) == 16, "Indexed indirect ABI changed");
    static_assert(sizeof(InstanceConstantBlock) == 15440, "GPU compaction output ABI changed");

    struct GpuDrivenBatchDesc
    {
        Float3 BoundsMinimum, BoundsMaximum;
        Matrix4 CullingViewProjection, RenderViewProjection;
        Float4 Material = {1, 0, 0, 0};
        std::uint32_t FirstIndex = 0, IndexCount = 0;
        std::int32_t BaseVertex = 0;
        bool FrustumCulling = true;
        bool Wireframe = false; // ES expands triangle edges; other APIs use native fill mode.
        std::vector<DrawInstance> Instances;
    };
    struct GpuDrivenBatchUpload
    {
        GpuDrivenCullConstants Constants;
        std::array<InstanceConstants, MaximumDrawInstances> Candidates;
    };

    // Validation/packing only: offscreen objects must survive this CPU step.
    // A failed upload is cleared so a caller cannot accidentally reuse a draw.
    [[nodiscard]] bool BuildGpuDrivenBatch(const GpuDrivenBatchDesc& desc,
        GpuDrivenBatchUpload& output);

    [[nodiscard]] inline bool GpuDrivenDrawMatches(const GpuDrivenBatchDesc& desc,
        const Matrix4& world, const Matrix4& camera, const PipelineState& pipeline, const PassDesc& pass)
    {
        return !desc.Instances.empty() && desc.Instances.front().WorldTransform.Values == world.Values &&
            desc.RenderViewProjection.Values == camera.Values && desc.Material.X == pipeline.Color.W * pipeline.MaterialOpacity &&
            desc.Wireframe == (pipeline.Fill == FillMode::Wireframe) && !pipeline.BlendEnabled &&
            !pipeline.OutlineExtrusion && pass.Type != PassType::Outline && !pass.DepthPeelingLayer;
    }
}
