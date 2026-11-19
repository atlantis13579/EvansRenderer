// D3D11 material work classification. A lane owns one pixel of an 8x8 tile;
// the first lane for each material appends that tile exactly once. Prefix
// allocation bounds total tile references by padded image pixels, not bins*pixels.
// Wihlidal, "Nanite GPU Driven Materials", GDC 2024.
static const uint TileWidth=8u, TileLanes=64u, BinWords=8u, WordBytes=4u;
static const uint QuadVertices=6u, CountWord=1u, ListWord=4u, CursorWord=5u;
static const uint NoMaterial=0xffffffffu, ReferenceMask=0x7fffffffu;
cbuffer MaterialFrame : register(b0) { uint4 Counts, Layout, Storage; };
ByteAddressBuffer MaterialMap : register(t0);
static const uint TriangleAttributeVectors=11u;
struct TriangleSource { float4 Attributes[TriangleAttributeVectors]; uint4 Source; };
StructuredBuffer<TriangleSource> Triangles : register(t1);
Texture2D<uint> Visibility : register(t2);
RWByteAddressBuffer Work : register(u0);
groupshared uint TileMaterials[TileLanes];
[numthreads(TileLanes,1,1)]
void CSClear(uint3 id : SV_DispatchThreadID)
{
    if (id.x>=Layout.x) return;
    uint base=id.x*BinWords*WordBytes;
    Work.Store4(base,uint4(QuadVertices,0,0,0));
    Work.Store4(base+ListWord*WordBytes,uint4(0,0,0,0));
}
[numthreads(TileLanes,1,1)]
void CSClassify(uint3 group : SV_GroupID,uint lane : SV_GroupIndex)
{
    uint2 pixel=group.xy*TileWidth+uint2(lane%TileWidth,lane/TileWidth);
    uint material=NoMaterial;
    [branch] if (all(pixel<Counts.xy))
    {
        uint reference=Visibility.Load(int3(pixel,0))&ReferenceMask;
        [branch] if (reference>0u && reference<=Counts.z)
        {
            uint source=Triangles[reference-1u].Source.x;
            [branch] if (source<Counts.w) material=MaterialMap.Load(source*WordBytes);
        }
    }
    TileMaterials[lane]=material; GroupMemoryBarrierWithGroupSync();
    bool first=material<Layout.x;
    for (uint previous=0u;previous<lane && first;++previous) first=TileMaterials[previous]!=material;
    uint temporary=Layout.w+(group.y*Layout.y+group.x)*TileLanes+lane;
    Work.Store(temporary*WordBytes,first?material:NoMaterial);
    if (first) { uint ignored; Work.InterlockedAdd((material*BinWords+CountWord)*WordBytes,1u,ignored); }
}
[numthreads(TileLanes,1,1)]
void CSAllocate(uint lane : SV_GroupIndex)
{
    // A small serial header scan avoids another hierarchy and never touches pixels.
    if (lane!=0u) return;
    uint start=Storage.y;
    for (uint bin=0u;bin<Layout.x;++bin)
    { Work.Store((bin*BinWords+ListWord)*WordBytes,start); start+=Work.Load((bin*BinWords+CountWord)*WordBytes); }
}
[numthreads(TileLanes,1,1)]
void CSScatter(uint3 group : SV_GroupID,uint lane : SV_GroupIndex)
{
    uint tile=group.y*Layout.y+group.x;
    uint bin=Work.Load((Layout.w+tile*TileLanes+lane)*WordBytes);
    [branch] if (bin<Layout.x)
    {
        uint cursor; Work.InterlockedAdd((bin*BinWords+CursorWord)*WordBytes,1u,cursor);
        uint start=Work.Load((bin*BinWords+ListWord)*WordBytes);
        Work.Store((start+cursor)*WordBytes,tile);
    }
}
