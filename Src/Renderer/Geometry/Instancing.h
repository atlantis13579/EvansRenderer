#pragma once

#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/FrameHistory.h"

namespace Evans::Renderer
{
    // Three matrices and vegetation parameters fit 64 objects in the 16 KiB native
    // uniform-block budget. A zero previous matrix encodes invalid history;
    // its interpolated clip W is zero and the velocity pixel stage rejects it.
    // SV_InstanceID/gl_InstanceID/instance_id select a record without duplicating vertices.
    inline constexpr std::size_t MaximumDrawInstances = 64;
    enum class InstanceVertexProgram : std::size_t { Forward, Gouraud, Deferred, Shadow, Velocity, Count };
    inline constexpr auto InstanceVertexProgramCount = static_cast<std::size_t>(InstanceVertexProgram::Count);
    struct InstanceConstants
    {
        Matrix4 World;
        Matrix4 NormalWorld;
        Matrix4 PreviousWorldViewProjection;
        Float4 Wind{0,0,0,0}, Gust{0,0,0,0}, VegetationHistory{0,0,0,0};
    };
    static_assert(sizeof(InstanceConstants) == 240 &&
        offsetof(InstanceConstants, PreviousWorldViewProjection) == 128,
        "Native instance uniform packing changed");
    struct InstanceConstantBlock
    {
        Matrix4 ViewProjection;
        Float4 Material; // Combined object/material opacity for the velocity pass.
        std::array<InstanceConstants, MaximumDrawInstances> Instances;
    };
    static_assert(offsetof(InstanceConstantBlock, Instances) == 80, "Instance array offset changed");
    static_assert(sizeof(InstanceConstantBlock) <= 16 * 1024, "Instance block exceeds portable uniform capacity");

    [[nodiscard]] bool BuildInstanceConstants(const std::vector<DrawInstance>& instances,
        const Matrix4& representativeWorld, InstanceConstantBlock& output);
    [[nodiscard]] bool SameInstancedPipeline(const PipelineState& left, const PipelineState& right);
}
