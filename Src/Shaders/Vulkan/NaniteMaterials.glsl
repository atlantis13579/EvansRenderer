#version 450
// Vulkan GLSL source is compiled at runtime. Image writes are published before
// classification; storage writes become vertex reads and indirect-command reads.
// Wihlidal, "Nanite GPU Driven Materials", GDC 2024.
layout(local_size_x=64) in;
const uint TileSide=8u, TilePixels=64u, BinStride=8u, QuadVertexCount=6u;
const uint CountOffset=1u, ListOffset=4u, CursorOffset=5u;
const uint TriangleStride=48u, SourceOffset=44u, IndexMask=0x7fffffffu;
const uint InvalidMaterial=0xffffffffu;
layout(set=0,binding=0,std140) uniform Frame { uvec4 Counts, Layout, Storage; } FrameData;
layout(set=0,binding=1,std430) readonly buffer Materials { uint Bin[]; } MaterialsData;
layout(set=0,binding=2,std430) readonly buffer Geometry { uint Words[]; } Triangles;
layout(set=0,binding=3,std430) buffer Schedule { uint Words[]; } OutputData;
layout(set=0,binding=4) uniform usampler2D Visibility;
shared uint TileBins[TilePixels];
void main()
{
    uint local=gl_LocalInvocationIndex;
#if EVANS_MATERIAL_PASS == 0
    uint material=gl_GlobalInvocationID.x;
    if (material>=FrameData.Layout.x) return;
    uint base=material*BinStride;
    for (uint field=0u;field<BinStride;++field) OutputData.Words[base+field]=field==0u?QuadVertexCount:0u;
#elif EVANS_MATERIAL_PASS == 1
    uvec2 location=gl_WorkGroupID.xy*TileSide+uvec2(local%TileSide,local/TileSide);
    uint material=InvalidMaterial;
    if (all(lessThan(location,FrameData.Counts.xy)))
    {
        uint primitive=texelFetch(Visibility,ivec2(location),0).r&IndexMask;
        if (primitive!=0u && primitive<=FrameData.Counts.z)
        {
            uint source=Triangles.Words[(primitive-1u)*TriangleStride+SourceOffset];
            if (source<FrameData.Counts.w) material=MaterialsData.Bin[source];
        }
    }
    TileBins[local]=material; barrier();
    bool owner=material<FrameData.Layout.x;
    for (uint other=0u;other<local && owner;++other) owner=TileBins[other]!=material;
    uint tile=gl_WorkGroupID.y*FrameData.Layout.y+gl_WorkGroupID.x;
    OutputData.Words[FrameData.Layout.w+tile*TilePixels+local]=owner?material:InvalidMaterial;
    if (owner) atomicAdd(OutputData.Words[material*BinStride+CountOffset],1u);
#elif EVANS_MATERIAL_PASS == 2
    if (local!=0u) return;
    uint offset=FrameData.Storage.y;
    for (uint material=0u;material<FrameData.Layout.x;++material)
    {
        OutputData.Words[material*BinStride+ListOffset]=offset;
        offset+=OutputData.Words[material*BinStride+CountOffset];
    }
#else
    uint tile=gl_WorkGroupID.y*FrameData.Layout.y+gl_WorkGroupID.x;
    uint material=OutputData.Words[FrameData.Layout.w+tile*TilePixels+local];
    if (material<FrameData.Layout.x)
    {
        uint position=atomicAdd(OutputData.Words[material*BinStride+CursorOffset],1u);
        uint list=OutputData.Words[material*BinStride+ListOffset];
        OutputData.Words[list+position]=tile;
    }
#endif
}
