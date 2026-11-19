// D3D12 native material tile scheduling. Headers contain draw arguments and
// compact-list offsets; the CPU never reads GPU material coverage or tile counts.
// Wihlidal, "Nanite GPU Driven Materials", GDC 2024.
static const uint TileExtent=8u, Lanes=64u, HeaderWords=8u, BytesPerWord=4u;
static const uint TriangleStride=192u, TriangleMaterialOffset=176u;
static const uint VertexCount=6u, InstanceField=1u, OffsetField=4u, CursorField=5u;
static const uint InvalidBin=0xffffffffu, TriangleMask=0x7fffffffu;
cbuffer Scheduling : register(b0) { uint4 Image, Bins, Memory; };
ByteAddressBuffer SourceBins : register(t0);
ByteAddressBuffer Geometry : register(t1);
Texture2D<uint> References : register(t2);
RWByteAddressBuffer Schedule : register(u0);
groupshared uint LocalBins[Lanes];
[numthreads(Lanes,1,1)]
void CSClear(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x>=Bins.x) return;
    uint address=thread.x*HeaderWords*BytesPerWord;
    Schedule.Store4(address,uint4(VertexCount,0,0,0));
    Schedule.Store4(address+OffsetField*BytesPerWord,uint4(0,0,0,0));
}
[numthreads(Lanes,1,1)]
void CSClassify(uint3 tile : SV_GroupID,uint lane : SV_GroupIndex)
{
    uint2 texel=tile.xy*TileExtent+uint2(lane%TileExtent,lane/TileExtent);
    uint bin=InvalidBin;
    [branch] if (all(texel<Image.xy))
    {
        uint reference=References.Load(int3(texel,0))&TriangleMask;
        [branch] if (reference!=0u && reference<=Image.z)
        {
            uint source=Geometry.Load((reference-1u)*TriangleStride+TriangleMaterialOffset);
            [branch] if (source<Image.w) bin=SourceBins.Load(source*BytesPerWord);
        }
    }
    LocalBins[lane]=bin; GroupMemoryBarrierWithGroupSync();
    bool unique=bin<Bins.x;
    for (uint earlier=0u;earlier<lane && unique;++earlier) unique=LocalBins[earlier]!=bin;
    uint location=Bins.w+(tile.y*Bins.y+tile.x)*Lanes+lane;
    Schedule.Store(location*BytesPerWord,unique?bin:InvalidBin);
    if (unique) { uint oldCount; Schedule.InterlockedAdd((bin*HeaderWords+InstanceField)*BytesPerWord,1u,oldCount); }
}
[numthreads(Lanes,1,1)]
void CSAllocate(uint lane : SV_GroupIndex)
{
    if (lane!=0u) return;
    uint allocation=Memory.y;
    for (uint bin=0u;bin<Bins.x;++bin)
    {
        Schedule.Store((bin*HeaderWords+OffsetField)*BytesPerWord,allocation);
        allocation+=Schedule.Load((bin*HeaderWords+InstanceField)*BytesPerWord);
    }
}
[numthreads(Lanes,1,1)]
void CSScatter(uint3 group : SV_GroupID,uint lane : SV_GroupIndex)
{
    uint tile=group.y*Bins.y+group.x;
    uint bin=Schedule.Load((Bins.w+tile*Lanes+lane)*BytesPerWord);
    [branch] if (bin<Bins.x)
    {
        uint local; Schedule.InterlockedAdd((bin*HeaderWords+CursorField)*BytesPerWord,1u,local);
        uint list=Schedule.Load((bin*HeaderWords+OffsetField)*BytesPerWord);
        Schedule.Store((list+local)*BytesPerWord,tile);
    }
}
