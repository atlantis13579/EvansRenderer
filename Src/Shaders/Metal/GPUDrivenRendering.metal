#include <metal_stdlib>
using namespace metal;

// Independent Metal compute producer, encoded before the graphics encoder.
// Stable scan preserves draw order and all motion/normal transforms. The output
// command matches MTLDrawIndexedPrimitivesIndirectArguments, with baseInstance 0.
// Harris et al., "Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3, chapter 39.
// Explicit rows preserve the host ABI without depending on MSL matrix layout.
constant uint InstanceCapacity = 64;
constant uint MatrixRowCount = 4;
constant uint AabbCornerCount = 8;
constant uint CornerX = 1, CornerY = 2, CornerZ = 4;
constant float BoundaryTolerance = 1.0e-5f;

struct InstanceRecord
{
    float4 World[MatrixRowCount];
    float4 NormalWorld[MatrixRowCount];
    float4 PreviousClip[MatrixRowCount];
    float4 Wind, Gust, VegetationHistory;
};
struct VisibilityConstants
{
    float4 CullRows[MatrixRowCount];
    float4 RenderRows[MatrixRowCount];
    float4 BoundsMinimum, BoundsMaximum, Material;
    uint4 Draw, Control;
};
struct InstanceBlock
{
    float4 Camera[MatrixRowCount];
    float4 Material;
    InstanceRecord Records[InstanceCapacity];
};
struct IndexedCommand
{
    uint IndexCount, InstanceCount, IndexStart;
    int BaseVertex;
    uint BaseInstance;
};

bool VisibleBox(constant VisibilityConstants& parameters, const device InstanceRecord& object)
{
    bool4 outsideSides = bool4(true);
    bool2 outsideDepth = bool2(true);
    for (uint corner = 0; corner < AabbCornerCount; ++corner)
    {
        float4 point((corner & CornerX) ? parameters.BoundsMaximum.x : parameters.BoundsMinimum.x,
            (corner & CornerY) ? parameters.BoundsMaximum.y : parameters.BoundsMinimum.y,
            (corner & CornerZ) ? parameters.BoundsMaximum.z : parameters.BoundsMinimum.z, 1.0f);
        float4 world(dot(object.World[0], point), dot(object.World[1], point), dot(object.World[2], point), dot(object.World[3], point));
        float4 clip(dot(parameters.CullRows[0], world), dot(parameters.CullRows[1], world),
            dot(parameters.CullRows[2], world), dot(parameters.CullRows[3], world));
        float4 sides(clip.w + clip.x, clip.w - clip.x, clip.w + clip.y, clip.w - clip.y);
        float2 depth(clip.z, clip.w - clip.z);
        if (!all(isfinite(sides)) || !all(isfinite(depth))) return true;
        outsideSides = outsideSides && (sides < -BoundaryTolerance);
        outsideDepth = outsideDepth && (depth < -BoundaryTolerance);
    }
    return !any(outsideSides) && !any(outsideDepth);
}

kernel void EvansGpuDrivenCull(constant VisibilityConstants& parameters [[buffer(0)]],
    const device InstanceRecord* candidates [[buffer(1)]], device InstanceBlock& output [[buffer(2)]],
    device IndexedCommand& command [[buffer(3)]], uint lane [[thread_index_in_threadgroup]])
{
    // Dispatch exactly one complete group of InstanceCapacity threads. Every
    // lane, including inactive candidates, must participate in every barrier.
    threadgroup uint prefix[InstanceCapacity];
    bool visible = false;
    if (lane < min(parameters.Draw.w, InstanceCapacity)) visible = parameters.Control.x == 0 || VisibleBox(parameters, candidates[lane]);
    prefix[lane] = visible ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint distance = 1; distance < InstanceCapacity; distance *= 2)
    {
        uint preceding = lane >= distance ? prefix[lane - distance] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix[lane] += preceding;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (visible)
    {
        uint destination = prefix[lane] - 1;
        for (uint row = 0; row < MatrixRowCount; ++row)
        {
            output.Records[destination].World[row] = candidates[lane].World[row];
            output.Records[destination].NormalWorld[row] = candidates[lane].NormalWorld[row];
            output.Records[destination].PreviousClip[row] = candidates[lane].PreviousClip[row];
        }
        output.Records[destination].Wind=candidates[lane].Wind;
        output.Records[destination].Gust=candidates[lane].Gust;
        output.Records[destination].VegetationHistory=candidates[lane].VegetationHistory;
    }
    if (lane == 0)
    {
        for (uint row = 0; row < MatrixRowCount; ++row) output.Camera[row] = parameters.RenderRows[row];
        output.Material = parameters.Material;
        command.IndexCount = parameters.Draw.x;
        command.InstanceCount = prefix[InstanceCapacity - 1];
        command.IndexStart = parameters.Draw.y;
        command.BaseVertex = as_type<int>(parameters.Draw.z);
        command.BaseInstance = 0;
    }
}
