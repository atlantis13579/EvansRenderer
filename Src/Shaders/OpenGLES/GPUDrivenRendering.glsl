#version 310 es
precision highp float;
precision highp int;
// ES 3.1 compute visibility and stable stream compaction. Bind the instance
// output as a UBO and the command output as GL_DRAW_INDIRECT_BUFFER after a
// GL_UNIFORM_BARRIER_BIT | GL_COMMAND_BARRIER_BIT visibility barrier.
// Hillis-Steele inclusive scan, GPU Gems 3, chapter 39 (Harris et al., 2007).
// Inputs retain renderer clip depth; ES's Z remap is done by the vertex consumer.
// ES 3.1 requires a literal layout qualifier; one named macro owns both uses.
#define EVANS_GPU_DRIVEN_GROUP_SIZE 64
const uint BatchSize = uint(EVANS_GPU_DRIVEN_GROUP_SIZE);
const uint MatrixRowCount = 4u;
const uint BoxCornerCount = 8u;
const uint XBit = 1u, YBit = 2u, ZBit = 4u;
const float ClipPlaneTolerance = 1.0e-5;
layout(local_size_x = EVANS_GPU_DRIVEN_GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

struct ObjectTransforms
{
    vec4 World[MatrixRowCount];
    vec4 NormalWorld[MatrixRowCount];
    vec4 PreviousWorldViewProjection[MatrixRowCount];
    vec4 Wind, Gust, VegetationHistory;
};
layout(std140, binding = 0) uniform CullParameters
{
    vec4 CullRows[MatrixRowCount];
    vec4 RenderRows[MatrixRowCount];
    vec4 LocalMinimum, LocalMaximum, Coverage;
    uvec4 DrawData;
    uvec4 Options;
};
layout(std430, binding = 0) readonly buffer InputTransforms { ObjectTransforms Objects[]; } Source;
layout(std430, binding = 1) writeonly buffer VisibleTransforms
{
    vec4 ViewProjection[MatrixRowCount];
    vec4 Material;
    ObjectTransforms Objects[];
} Destination;
layout(std430, binding = 2) writeonly buffer DrawElementsCommand
{
    uint Count, InstanceCount, FirstIndex, BaseVertexBits, ReservedMustBeZero;
} Command;
shared uint VisiblePrefix[BatchSize];

bool TestBox(uint objectIndex)
{
    ivec4 allSideCornersOutside = ivec4(1);
    ivec2 allDepthCornersOutside = ivec2(1);
    for (uint corner = 0u; corner < BoxCornerCount; ++corner)
    {
        vec4 point = vec4((corner & XBit) != 0u ? LocalMaximum.x : LocalMinimum.x,
            (corner & YBit) != 0u ? LocalMaximum.y : LocalMinimum.y,
            (corner & ZBit) != 0u ? LocalMaximum.z : LocalMinimum.z, 1.0);
        vec4 world = vec4(dot(Source.Objects[objectIndex].World[0], point), dot(Source.Objects[objectIndex].World[1], point),
            dot(Source.Objects[objectIndex].World[2], point), dot(Source.Objects[objectIndex].World[3], point));
        vec4 clip = vec4(dot(CullRows[0], world), dot(CullRows[1], world), dot(CullRows[2], world), dot(CullRows[3], world));
        vec4 sides = vec4(clip.w + clip.x, clip.w - clip.x, clip.w + clip.y, clip.w - clip.y);
        vec2 depth = vec2(clip.z, clip.w - clip.z);
        if (any(isnan(sides)) || any(isinf(sides)) || any(isnan(depth)) || any(isinf(depth))) return true;
        allSideCornersOutside *= ivec4(lessThan(sides, vec4(-ClipPlaneTolerance)));
        allDepthCornersOutside *= ivec2(lessThan(depth, vec2(-ClipPlaneTolerance)));
    }
    return all(equal(allSideCornersOutside, ivec4(0))) && all(equal(allDepthCornersOutside, ivec2(0)));
}

void main()
{
    uint slot = gl_LocalInvocationIndex;
    bool keep = false;
    if (slot < min(DrawData.w, BatchSize)) keep = Options.x == 0u || TestBox(slot);
    VisiblePrefix[slot] = keep ? 1u : 0u;
    barrier();
    for (uint distance = 1u; distance < BatchSize; distance *= 2u)
    {
        uint prior = slot >= distance ? VisiblePrefix[slot - distance] : 0u;
        barrier();
        VisiblePrefix[slot] += prior;
        barrier();
    }
    if (keep)
    {
        uint outputIndex = VisiblePrefix[slot] - 1u;
        // Explicit vector stores also avoid aggregate-array lowering differences
        // in ES drivers; the record remains the same three row-major matrices.
        for (uint row = 0u; row < MatrixRowCount; ++row)
        {
            Destination.Objects[outputIndex].World[row] = Source.Objects[slot].World[row];
            Destination.Objects[outputIndex].NormalWorld[row] = Source.Objects[slot].NormalWorld[row];
            Destination.Objects[outputIndex].PreviousWorldViewProjection[row] = Source.Objects[slot].PreviousWorldViewProjection[row];
        }
        Destination.Objects[outputIndex].Wind=Source.Objects[slot].Wind;
        Destination.Objects[outputIndex].Gust=Source.Objects[slot].Gust;
        Destination.Objects[outputIndex].VegetationHistory=Source.Objects[slot].VegetationHistory;
    }
    if (slot == 0u)
    {
        for (uint row = 0u; row < MatrixRowCount; ++row) Destination.ViewProjection[row] = RenderRows[row];
        Destination.Material = Coverage;
        Command.Count = DrawData.x;
        Command.InstanceCount = VisiblePrefix[BatchSize - 1u];
        Command.FirstIndex = DrawData.y;
        Command.BaseVertexBits = DrawData.z;
        Command.ReservedMustBeZero = 0u;
    }
}
