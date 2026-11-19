#version 310 es
precision highp float;
precision highp int;
// ES-native classify/allocate/scatter. Tile Y follows GL texture coordinates;
// resolve uses the same orientation, including partial tiles at the image edge.
// Wihlidal, "Nanite GPU Driven Materials", GDC 2024.
layout(local_size_x=64) in;
const uint TileWidth=8u, TileThreads=64u, HeaderWords=8u, QuadVertices=6u;
const uint InstanceWord=1u, ListWord=4u, CursorWord=5u;
const uint TriangleWords=48u, MaterialWord=44u;
const uint Missing=0xffffffffu, ReferenceMask=0x7fffffffu;
layout(std140,binding=0) uniform Frame { uvec4 Image, Bins, Memory; };
layout(std430,binding=0) readonly buffer SourceMaterials { uint MaterialBins[]; };
layout(std430,binding=1) readonly buffer TriangleData { uint Geometry[]; };
// Separate blocks retain global addresses beyond the per-block device limit.
#if defined(EVANS_VISIBILITY_BANK_WORDS)
layout(std430,binding=4) readonly buffer GeometryOverflow { uint overflowWords[]; };
#endif
uint readGeometryWord(uint word)
{
#if defined(EVANS_VISIBILITY_BANK_WORDS)
    const uint BankWords=EVANS_VISIBILITY_BANK_WORDS;
    if (word>=BankWords) return overflowWords[word-BankWords];
#endif
    return Geometry[word];
}
layout(std430,binding=2) buffer Scheduling { uint Work[]; };
layout(binding=0) uniform highp usampler2D Visibility;
layout(r32ui,binding=0) writeonly uniform highp uimage2D TileList;
shared uint LocalMaterial[TileThreads];
ivec2 TileAddress(uint word) { return ivec2(word%Memory.w,word/Memory.w); }
void main()
{
    uint lane=gl_LocalInvocationIndex;
#if EVANS_MATERIAL_PASS == 0
    uint bin=gl_GlobalInvocationID.x;
    if (bin>=Bins.x) return;
    for (uint word=0u;word<HeaderWords;++word) Work[bin*HeaderWords+word]=word==0u?QuadVertices:0u;
#elif EVANS_MATERIAL_PASS == 1
    uvec2 pixel=gl_WorkGroupID.xy*TileWidth+uvec2(lane%TileWidth,lane/TileWidth);
    uint bin=Missing;
    if (all(lessThan(pixel,Image.xy)))
    {
        uint reference=texelFetch(Visibility,ivec2(pixel),0).r&ReferenceMask;
        if (reference>0u && reference<=Image.z)
        {
            uint source=readGeometryWord((reference-1u)*TriangleWords+MaterialWord);
            if (source<Image.w) bin=MaterialBins[source];
        }
    }
    LocalMaterial[lane]=bin; barrier();
    bool first=bin<Bins.x;
    for (uint previous=0u;previous<lane && first;++previous) first=LocalMaterial[previous]!=bin;
    uint tile=gl_WorkGroupID.y*Bins.y+gl_WorkGroupID.x;
    Work[Bins.w+tile*TileThreads+lane]=first?bin:Missing;
    if (first) atomicAdd(Work[bin*HeaderWords+InstanceWord],1u);
#elif EVANS_MATERIAL_PASS == 2
    if (lane!=0u) return;
    uint start=Memory.y;
    for (uint bin=0u;bin<Bins.x;++bin)
    {
        Work[bin*HeaderWords+ListWord]=start;
        imageStore(TileList,TileAddress(bin),uvec4(start-Memory.y,0,0,0));
        start+=Work[bin*HeaderWords+InstanceWord];
    }
#else
    uint tile=gl_WorkGroupID.y*Bins.y+gl_WorkGroupID.x;
    uint bin=Work[Bins.w+tile*TileThreads+lane];
    if (bin<Bins.x)
    {
        uint cursor=atomicAdd(Work[bin*HeaderWords+CursorWord],1u);
        uint at=Work[bin*HeaderWords+ListWord]+cursor;
        imageStore(TileList,TileAddress(Bins.x+at-Memory.y),uvec4(tile,0,0,0));
    }
#endif
}
