#include <metal_stdlib>
using namespace metal;
// Independent Metal material tile scheduler. Separate tracked compute encoders
// publish atomic counts, compact offsets and indirect arguments without readback.
// Wihlidal, "Nanite GPU Driven Materials", GDC 2024.
constant uint TileSide=8u, TileThreads=64u, HeaderSize=8u, QuadVertices=6u;
constant uint InstanceCount=1u, ListStart=4u, WriteCursor=5u;
constant uint TriangleSize=48u, TriangleSource=44u;
constant uint NoBin=0xffffffffu, TriangleMask=0x7fffffffu;
struct MaterialFrame { uint4 Image, Bins, Memory; };
kernel void EvansMaterialClear(constant MaterialFrame& frame [[buffer(0)]],
    device atomic_uint* work [[buffer(3)]],uint thread [[thread_position_in_grid]])
{
    if (thread>=frame.Bins.x) return;
    for (uint field=0u;field<HeaderSize;++field)
        atomic_store_explicit(work+thread*HeaderSize+field,field==0u?QuadVertices:0u,memory_order_relaxed);
}
kernel void EvansMaterialClassify(constant MaterialFrame& frame [[buffer(0)]],
    const device uint* materialBins [[buffer(1)]],const device uint* geometry [[buffer(2)]],
    device atomic_uint* work [[buffer(3)]],texture2d<uint,access::read> visibility [[texture(0)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    threadgroup uint localBins[TileThreads];
    uint2 pixel=group.xy*TileSide+uint2(lane%TileSide,lane/TileSide); uint bin=NoBin;
    if (all(pixel<frame.Image.xy))
    {
        uint reference=visibility.read(pixel).r&TriangleMask;
        if (reference!=0u && reference<=frame.Image.z)
        {
            uint source=geometry[(reference-1u)*TriangleSize+TriangleSource];
            if (source<frame.Image.w) bin=materialBins[source];
        }
    }
    localBins[lane]=bin; threadgroup_barrier(mem_flags::mem_threadgroup);
    bool unique=bin<frame.Bins.x;
    for (uint earlier=0u;earlier<lane && unique;++earlier) unique=localBins[earlier]!=bin;
    uint tile=group.y*frame.Bins.y+group.x;
    atomic_store_explicit(work+frame.Bins.w+tile*TileThreads+lane,unique?bin:NoBin,memory_order_relaxed);
    if (unique) atomic_fetch_add_explicit(work+bin*HeaderSize+InstanceCount,1u,memory_order_relaxed);
}
kernel void EvansMaterialAllocate(constant MaterialFrame& frame [[buffer(0)]],
    device atomic_uint* work [[buffer(3)]],uint lane [[thread_index_in_threadgroup]])
{
    if (lane!=0u) return;
    uint offset=frame.Memory.y;
    for (uint bin=0u;bin<frame.Bins.x;++bin)
    {
        atomic_store_explicit(work+bin*HeaderSize+ListStart,offset,memory_order_relaxed);
        offset+=atomic_load_explicit(work+bin*HeaderSize+InstanceCount,memory_order_relaxed);
    }
}
kernel void EvansMaterialScatter(constant MaterialFrame& frame [[buffer(0)]],
    device atomic_uint* work [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint tile=group.y*frame.Bins.y+group.x;
    uint bin=atomic_load_explicit(work+frame.Bins.w+tile*TileThreads+lane,memory_order_relaxed);
    if (bin<frame.Bins.x)
    {
        uint cursor=atomic_fetch_add_explicit(work+bin*HeaderSize+WriteCursor,1u,memory_order_relaxed);
        uint start=atomic_load_explicit(work+bin*HeaderSize+ListStart,memory_order_relaxed);
        atomic_store_explicit(work+start+cursor,tile,memory_order_relaxed);
    }
}
