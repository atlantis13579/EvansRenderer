// Independent D3D11 compute producer: conservative box visibility, stable
// instance compaction and DrawIndexedInstancedIndirect arguments. Copy the raw
// instance UAV into a constant buffer before the existing instanced vertex stage.
// Harris et al., GPU Gems 3, chapter 39, "Parallel Prefix Sum (Scan) with CUDA".
// Hillis-Steele uses two group barriers per scan step; no wave-size assumption.
static const uint InstanceCapacity = 64;
static const uint MatrixRowCount = 4;
static const uint RowsPerInstance = 12;
static const uint RowBytes = 16;
static const uint InstanceBytes = RowsPerInstance * RowBytes;
static const uint InstanceHeaderBytes = (MatrixRowCount + 1) * RowBytes;
static const uint BoxCornerCount = 8;
static const uint CornerX = 1, CornerY = 2, CornerZ = 4;
static const uint IndexCountOffset = 0;
static const uint InstanceCountOffset = 4;
static const uint FirstIndexOffset = 8;
static const uint BaseVertexOffset = 12;
static const uint FirstInstanceOffset = 16;
static const float ClipBoundaryTolerance = 1.0e-5;

cbuffer VisibilityParameters : register(b0)
{
    float4 CullRows[MatrixRowCount];
    float4 RenderRows[MatrixRowCount];
    float4 BoundsMinimum, BoundsMaximum, Material;
    uint4 Draw; // Index count, first index, base vertex bits, candidate count.
    uint4 Control;
};
ByteAddressBuffer Candidates : register(t0);
RWByteAddressBuffer CompactedInstances : register(u0);
RWByteAddressBuffer IndirectArguments : register(u1);
groupshared uint Prefix[InstanceCapacity];

bool VisibleBox(uint instance)
{
    bool4 outsideXY = true;
    bool2 outsideZ = true;
    uint objectAddress = instance * InstanceBytes;
    float4 worldRow0 = asfloat(Candidates.Load4(objectAddress));
    float4 worldRow1 = asfloat(Candidates.Load4(objectAddress + RowBytes));
    float4 worldRow2 = asfloat(Candidates.Load4(objectAddress + 2 * RowBytes));
    float4 worldRow3 = asfloat(Candidates.Load4(objectAddress + 3 * RowBytes));
    [unroll] for (uint corner = 0; corner < BoxCornerCount; ++corner)
    {
        float4 localPosition = float4((corner & CornerX) ? BoundsMaximum.x : BoundsMinimum.x,
            (corner & CornerY) ? BoundsMaximum.y : BoundsMinimum.y,
            (corner & CornerZ) ? BoundsMaximum.z : BoundsMinimum.z, 1.0);
        float4 world = float4(dot(worldRow0, localPosition), dot(worldRow1, localPosition), dot(worldRow2, localPosition), dot(worldRow3, localPosition));
        float4 clip = float4(dot(CullRows[0], world), dot(CullRows[1], world), dot(CullRows[2], world), dot(CullRows[3], world));
        float4 side = float4(clip.w + clip.x, clip.w - clip.x, clip.w + clip.y, clip.w - clip.y);
        float2 depth = float2(clip.z, clip.w - clip.z);
        // Overflow cannot justify rejecting geometry. Keep it conservatively.
        if (!all(isfinite(side)) || !all(isfinite(depth))) return true;
        outsideXY = outsideXY && side < -ClipBoundaryTolerance;
        outsideZ = outsideZ && depth < -ClipBoundaryTolerance;
    }
    return !any(outsideXY) && !any(outsideZ);
}

[numthreads(InstanceCapacity, 1, 1)]
void CSCull(uint lane : SV_GroupIndex)
{
    bool visible = false;
    if (lane < min(Draw.w, InstanceCapacity)) visible = Control.x == 0 || VisibleBox(lane);
    Prefix[lane] = visible ? 1 : 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < InstanceCapacity; offset *= 2)
    {
        uint predecessor = lane >= offset ? Prefix[lane - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        Prefix[lane] += predecessor;
        GroupMemoryBarrierWithGroupSync();
    }
    if (visible)
    {
        uint destination = InstanceHeaderBytes + (Prefix[lane] - 1) * InstanceBytes;
        for (uint row = 0; row < RowsPerInstance; ++row)
            CompactedInstances.Store4(destination + row * RowBytes, Candidates.Load4(lane * InstanceBytes + row * RowBytes));
    }
    if (lane == 0)
    {
        for (uint row = 0; row < MatrixRowCount; ++row) CompactedInstances.Store4(row * RowBytes, asuint(RenderRows[row]));
        CompactedInstances.Store4(MatrixRowCount * RowBytes, asuint(Material));
        IndirectArguments.Store(IndexCountOffset, Draw.x);
        IndirectArguments.Store(InstanceCountOffset, Prefix[InstanceCapacity - 1]);
        IndirectArguments.Store(FirstIndexOffset, Draw.y);
        IndirectArguments.Store(BaseVertexOffset, Draw.z);
        IndirectArguments.Store(FirstInstanceOffset, 0);
    }
}
