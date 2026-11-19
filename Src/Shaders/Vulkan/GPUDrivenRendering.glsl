#version 450
// Vulkan-native compute, compiled from this source at runtime. One workgroup
// compacts one ordered batch. Barrier storage writes for vertex uniform reads
// and indirect-command reads before vkCmdDrawIndexedIndirect; dispatch outside
// the render pass. Positive-height viewport Y conversion belongs to the vertex
// consumer, not this renderer-space 0 <= Z <= W visibility test.
// Harris et al., "Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3, chapter 39.
const uint InstanceCapacity = 64u;
const uint MatrixRows = 4u;
const uint CornerCount = 8u;
const uint CornerX = 1u, CornerY = 2u, CornerZ = 4u;
const uint LeftPlane = 1u, RightPlane = 2u, BottomPlane = 4u;
const uint TopPlane = 8u, NearPlane = 16u, FarPlane = 32u;
const uint AllPlanes = LeftPlane | RightPlane | BottomPlane | TopPlane | NearPlane | FarPlane;
const float ClipTolerance = 1.0e-5;
layout(local_size_x = InstanceCapacity, local_size_y = 1, local_size_z = 1) in;

struct InstanceRecord
{
    vec4 World[MatrixRows];
    vec4 NormalWorld[MatrixRows];
    vec4 PreviousClip[MatrixRows];
    vec4 Wind, Gust, VegetationHistory;
};
layout(set = 0, binding = 0, std140) uniform VisibilityParameters
{
    vec4 CullingCamera[MatrixRows];
    vec4 DrawingCamera[MatrixRows];
    vec4 BoundsMinimum, BoundsMaximum, Material;
    uvec4 Draw;
    uvec4 Control;
} Parameters;
layout(set = 0, binding = 1, std430) readonly buffer CandidateBuffer { InstanceRecord Records[]; } Candidates;
layout(set = 0, binding = 2, std430) writeonly buffer CompactedBuffer
{
    vec4 Camera[MatrixRows];
    vec4 Material;
    InstanceRecord Records[];
} Compacted;
layout(set = 0, binding = 3, std430) writeonly buffer IndexedIndirectBuffer
{
    uint IndexCount, InstanceCount, FirstIndex, BaseVertexBits, FirstInstance;
} Arguments;
shared uint Prefix[InstanceCapacity];

bool BoxIntersectsFrustum(uint objectIndex)
{
    uint outside = AllPlanes;
    for (uint corner = 0u; corner < CornerCount; ++corner)
    {
        vec4 local = vec4((corner & CornerX) != 0u ? Parameters.BoundsMaximum.x : Parameters.BoundsMinimum.x,
            (corner & CornerY) != 0u ? Parameters.BoundsMaximum.y : Parameters.BoundsMinimum.y,
            (corner & CornerZ) != 0u ? Parameters.BoundsMaximum.z : Parameters.BoundsMinimum.z, 1.0);
        vec4 world = vec4(dot(Candidates.Records[objectIndex].World[0], local), dot(Candidates.Records[objectIndex].World[1], local),
            dot(Candidates.Records[objectIndex].World[2], local), dot(Candidates.Records[objectIndex].World[3], local));
        vec4 clip = vec4(dot(Parameters.CullingCamera[0], world), dot(Parameters.CullingCamera[1], world),
            dot(Parameters.CullingCamera[2], world), dot(Parameters.CullingCamera[3], world));
        vec4 sides = vec4(clip.w + clip.x, clip.w - clip.x, clip.w + clip.y, clip.w - clip.y);
        vec2 depth = vec2(clip.z, clip.w - clip.z);
        if (any(isnan(sides)) || any(isinf(sides)) || any(isnan(depth)) || any(isinf(depth))) return true;
        uint cornerOutside = (sides.x < -ClipTolerance ? LeftPlane : 0u) | (sides.y < -ClipTolerance ? RightPlane : 0u) |
            (sides.z < -ClipTolerance ? BottomPlane : 0u) | (sides.w < -ClipTolerance ? TopPlane : 0u) |
            (depth.x < -ClipTolerance ? NearPlane : 0u) | (depth.y < -ClipTolerance ? FarPlane : 0u);
        outside &= cornerOutside;
    }
    return outside == 0u;
}

void main()
{
    uint lane = gl_LocalInvocationIndex;
    bool visible = false;
    if (lane < min(Parameters.Draw.w, InstanceCapacity)) visible = Parameters.Control.x == 0u || BoxIntersectsFrustum(lane);
    Prefix[lane] = visible ? 1u : 0u;
    barrier();
    for (uint offset = 1u; offset < InstanceCapacity; offset *= 2u)
    {
        uint preceding = lane >= offset ? Prefix[lane - offset] : 0u;
        barrier();
        Prefix[lane] += preceding;
        barrier();
    }
    if (visible) Compacted.Records[Prefix[lane] - 1u] = Candidates.Records[lane];
    if (lane == 0u)
    {
        for (uint row = 0u; row < MatrixRows; ++row) Compacted.Camera[row] = Parameters.DrawingCamera[row];
        Compacted.Material = Parameters.Material;
        Arguments.IndexCount = Parameters.Draw.x;
        Arguments.InstanceCount = Prefix[InstanceCapacity - 1u];
        Arguments.FirstIndex = Parameters.Draw.y;
        Arguments.BaseVertexBits = Parameters.Draw.z;
        Arguments.FirstInstance = 0u;
    }
}
