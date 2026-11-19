#include <metal_stdlib>
using namespace metal;
// Matches MTLDrawIndexedPrimitivesIndirectArguments. A compute encoder writes
// commands before the ordinary material render encoder consumes them.
struct Parameters { uint4 counts,layoutInfo,source,target; };
struct IndexedCommand
{
    uint indexCount,instanceCount,indexStart;
    int baseVertex;
    uint baseInstance;
};
constant uint GroupWidth=64,MainVisible=2,PostVisible=4;
kernel void EvansNaniteDraws(constant Parameters& p [[buffer(0)]],
    device const uint2* ranges [[buffer(1)]],device IndexedCommand* commands [[buffer(2)]],
    device const uint* decisions [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint cluster=(group.y*p.target.w+group.x)*GroupWidth+lane;
    if (cluster>=p.counts.x) return;
    const uint2 range=ranges[cluster];
    const uint accepted=p.counts.w==0?MainVisible:PostVisible;
    commands[cluster]={range.y,decisions[cluster]==accepted?1u:0u,range.x,0,0};
}

constant uint CandidateStride=4,BlockStride=4,BinStride=8,CommandStride=5;
inline void inclusivePrefix(threadgroup uint* values,uint lane,uint count)
{
    values[lane]=count; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint distance=1;distance<GroupWidth;distance*=2)
    {
        uint previous=lane>=distance?values[lane-distance]:0;
        threadgroup_barrier(mem_flags::mem_threadgroup); values[lane]+=previous;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void EvansNaniteBinCount(constant Parameters& p [[buffer(0)]],device const uint* metadata [[buffer(1)]],
    device uint* output [[buffer(2)]],device const uint* states [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint block=group.y*p.target.w+group.x; if (block>=p.target.z) return;
    uint record=p.layoutInfo.y+block*BlockStride,query=metadata[record]+lane,count=0;
    uint length=metadata[record+1];
    if (lane<length && states[query]==(p.counts.w==0?MainVisible:PostVisible)) count=metadata[p.layoutInfo.x+query*CandidateStride+1];
    threadgroup uint scratch[GroupWidth]; inclusivePrefix(scratch,lane,count);
    if (lane<length) output[p.source.x+query]=scratch[lane]-count;
    if (lane==GroupWidth-1) output[p.source.y+block]=scratch[lane];
}
kernel void EvansNaniteBinScan(constant Parameters& p [[buffer(0)]],device const uint* metadata [[buffer(1)]],
    device uint* output [[buffer(2)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint bin=group.y*p.target.w+group.x; if (bin>=p.target.z) return;
    uint record=p.layoutInfo.z+bin*BinStride,firstBlock=metadata[record+2],blockCount=metadata[record+3],running=0;
    threadgroup uint scratch[GroupWidth];
    for (uint first=0;first<blockCount;first+=GroupWidth)
    {
        uint block=first+lane,count=block<blockCount?output[p.source.y+firstBlock+block]:0;
        inclusivePrefix(scratch,lane,count);
        if (block<blockCount) output[p.source.z+firstBlock+block]=running+scratch[lane]-count;
        uint batch=scratch[GroupWidth-1]; threadgroup_barrier(mem_flags::mem_threadgroup); running+=batch;
    }
    if (lane==0)
    {
        uint command=p.source.w+bin*CommandStride;
        output[command]=running; output[command+1]=running?1u:0u; output[command+2]=p.target.x+metadata[record+4];
        output[command+3]=output[command+4]=0;
    }
}
kernel void EvansNaniteBinScatter(constant Parameters& p [[buffer(0)]],device const uint* metadata [[buffer(1)]],
    device uint* output [[buffer(2)]],device const uint* states [[buffer(3)]],device const uint* indices [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint query=group.y*p.target.w+group.x; if (query>=p.target.z) return;
    if (states[query]!=(p.counts.w==0?MainVisible:PostVisible)) return;
    uint record=p.layoutInfo.x+query*CandidateStride,first=metadata[record],count=metadata[record+1];
    uint block=metadata[record+2],bin=metadata[record+3];
    uint destination=p.target.x+metadata[p.layoutInfo.z+bin*BinStride+4]+output[p.source.x+query]+output[p.source.z+block];
    for (uint corner=lane;corner<count;corner+=GroupWidth) output[destination+corner]=indices[first+corner];
}
