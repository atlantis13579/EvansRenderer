// Native D3D12 visibility producer. Structured input preserves all three rigid
// transforms; raw outputs become a CBV and an indexed ExecuteIndirect argument.
// Transition both UAVs to their graphics consumer states after this dispatch.
// Stable Hillis-Steele scan: Harris et al., GPU Gems 3, chapter 39 (2007).
static const uint BatchCapacity = 64;
static const uint MatrixRows = 4;
static const uint VectorBytes = 16;
static const uint MatricesPerInstance = 3;
static const uint InstanceStride = (MatricesPerInstance * MatrixRows + 3) * VectorBytes;
static const uint TransformArrayOffset = (MatrixRows + 1) * VectorBytes;
static const uint AabbCorners = 8;
static const uint XCornerBit = 1, YCornerBit = 2, ZCornerBit = 4;
static const uint ArgumentWordBytes = 4;
static const uint IndexCountWord = 0, InstanceCountWord = 1, FirstIndexWord = 2, BaseVertexWord = 3, FirstInstanceWord = 4;
static const float BoundaryTolerance = 1.0e-5;

struct RigidInstance
{
    float4 World[MatrixRows];
    float4 NormalWorld[MatrixRows];
    float4 PreviousClip[MatrixRows];
    float4 Wind, Gust, VegetationHistory;
};
cbuffer CullParameters : register(b0)
{
    float4 CullingCamera[MatrixRows];
    float4 DrawingCamera[MatrixRows];
    float4 BoxMin, BoxMax, Coverage;
    uint4 IndexedDraw;
    uint4 Options;
};
StructuredBuffer<RigidInstance> InputInstances : register(t0);
RWByteAddressBuffer VisibleInstances : register(u0);
RWByteAddressBuffer IndexedCommand : register(u1);
groupshared uint InclusiveCount[BatchCapacity];

bool IntersectsCamera(uint objectIndex)
{
    RigidInstance object = InputInstances[objectIndex];
    bool4 rejectedSides = true;
    bool2 rejectedDepth = true;
    [unroll] for (uint corner = 0; corner < AabbCorners; ++corner)
    {
        float4 local = float4((corner & XCornerBit) != 0 ? BoxMax.x : BoxMin.x,
            (corner & YCornerBit) != 0 ? BoxMax.y : BoxMin.y,
            (corner & ZCornerBit) != 0 ? BoxMax.z : BoxMin.z, 1.0);
        float4 world = float4(dot(object.World[0], local), dot(object.World[1], local), dot(object.World[2], local), dot(object.World[3], local));
        float4 clip = float4(dot(CullingCamera[0], world), dot(CullingCamera[1], world), dot(CullingCamera[2], world), dot(CullingCamera[3], world));
        float4 sides = float4(clip.w + clip.x, clip.w - clip.x, clip.w + clip.y, clip.w - clip.y);
        float2 nearFar = float2(clip.z, clip.w - clip.z);
        if (!all(isfinite(sides)) || !all(isfinite(nearFar))) return true;
        rejectedSides = rejectedSides && sides < -BoundaryTolerance;
        rejectedDepth = rejectedDepth && nearFar < -BoundaryTolerance;
    }
    return !(any(rejectedSides) || any(rejectedDepth));
}

[numthreads(BatchCapacity, 1, 1)]
void CSCull(uint threadIndex : SV_GroupIndex)
{
    bool accepted = false;
    if (threadIndex < min(IndexedDraw.w, BatchCapacity)) accepted = Options.x == 0 || IntersectsCamera(threadIndex);
    InclusiveCount[threadIndex] = accepted ? 1 : 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint distance = 1; distance < BatchCapacity; distance *= 2)
    {
        uint earlier = threadIndex >= distance ? InclusiveCount[threadIndex - distance] : 0;
        // Finish every read before any thread overwrites this scan level.
        GroupMemoryBarrierWithGroupSync();
        InclusiveCount[threadIndex] += earlier;
        GroupMemoryBarrierWithGroupSync();
    }
    if (accepted)
    {
        RigidInstance object = InputInstances[threadIndex];
        uint address = TransformArrayOffset + (InclusiveCount[threadIndex] - 1) * InstanceStride;
        for (uint row = 0; row < MatrixRows; ++row)
        {
            VisibleInstances.Store4(address + row * VectorBytes, asuint(object.World[row]));
            VisibleInstances.Store4(address + (MatrixRows + row) * VectorBytes, asuint(object.NormalWorld[row]));
            VisibleInstances.Store4(address + (2 * MatrixRows + row) * VectorBytes, asuint(object.PreviousClip[row]));
        }
        VisibleInstances.Store4(address+192,asuint(object.Wind));
        VisibleInstances.Store4(address+208,asuint(object.Gust));
        VisibleInstances.Store4(address+224,asuint(object.VegetationHistory));
    }
    if (threadIndex == 0)
    {
        for (uint row = 0; row < MatrixRows; ++row) VisibleInstances.Store4(row * VectorBytes, asuint(DrawingCamera[row]));
        VisibleInstances.Store4(MatrixRows * VectorBytes, asuint(Coverage));
        IndexedCommand.Store(IndexCountWord * ArgumentWordBytes, IndexedDraw.x);
        IndexedCommand.Store(InstanceCountWord * ArgumentWordBytes, InclusiveCount[BatchCapacity - 1]);
        IndexedCommand.Store(FirstIndexWord * ArgumentWordBytes, IndexedDraw.y);
        IndexedCommand.Store(BaseVertexWord * ArgumentWordBytes, IndexedDraw.z);
        IndexedCommand.Store(FirstInstanceWord * ArgumentWordBytes, 0);
    }
}
