#version 310 es
precision highp float;
precision highp int;
// ES 3.1 indexed indirect commands use absolute indices. Instance count zero
// suppresses a cluster; the reserved final word must remain zero on ES.
layout(local_size_x=64) in;
layout(std140,binding=0) uniform NaniteDrawParameters
{
    uvec4 counts;
    uvec4 layoutInfo;
    uvec4 source;
    uvec4 target;
};
layout(std430,binding=0) readonly buffer Ranges { uint rangeWords[]; };
layout(std430,binding=1) buffer Commands { uint argumentWords[]; };
layout(std430,binding=2) readonly buffer Visibility { uint decisionWords[]; };
const uint Threads=64u,RangeStride=2u,ArgumentStride=5u;
const uint MainVisibleState=2u,PostVisibleState=4u;
#ifndef EVANS_NANITE_BIN_PASS
void main()
{
    uint index=(gl_WorkGroupID.y*target.w+gl_WorkGroupID.x)*Threads+gl_LocalInvocationIndex;
    if (index>=counts.x) return;
    uint range=index*RangeStride;
    uint command=index*ArgumentStride;
    uint phaseState=counts.w==0u?MainVisibleState:PostVisibleState;
    argumentWords[command]=rangeWords[range+1u];
    argumentWords[command+1u]=uint(decisionWords[index]==phaseState);
    argumentWords[command+2u]=rangeWords[range];
    argumentWords[command+3u]=0u;
    argumentWords[command+4u]=0u;
}
#else
layout(std430,binding=3) readonly buffer IndexSource { uint sourceIndices[]; };
const uint CandidateWords=4u,BlockWords=4u,BinWords=8u;
shared uint localScan[Threads];
void blockPrefix(uint lane,uint count)
{
    localScan[lane]=count; barrier();
    for (uint step=1u;step<Threads;step*=2u)
    {
        uint neighbor=0u; if (lane>=step) neighbor=localScan[lane-step];
        barrier(); localScan[lane]+=neighbor; barrier();
    }
}
void main()
{
    uint work=gl_WorkGroupID.y*target.w+gl_WorkGroupID.x;
    uint lane=gl_LocalInvocationIndex;
    if (work>=target.z) return;
    uint accepted=counts.w==0u?MainVisibleState:PostVisibleState;
#if EVANS_NANITE_BIN_PASS == 0
    uint block=layoutInfo.y+work*BlockWords;
    uint first=rangeWords[block],length=rangeWords[block+1u],query=first+lane,count=0u;
    if (lane<length) if (decisionWords[query]==accepted) count=rangeWords[layoutInfo.x+query*CandidateWords+1u];
    blockPrefix(lane,count);
    if (lane<length) argumentWords[source.x+query]=localScan[lane]-count;
    if (lane==Threads-1u) argumentWords[source.y+work]=localScan[lane];
#elif EVANS_NANITE_BIN_PASS == 1
    // A serial scan of block totals needs no barrier in SSBO-dependent control
    // flow, remaining valid on ES translators with strict uniformity analysis.
    // Each block already represents up to 64 candidate clusters.
    if (lane==0u)
    {
        uint descriptor=layoutInfo.z+work*BinWords;
        uint first=rangeWords[descriptor+2u],length=rangeWords[descriptor+3u],total=0u;
        for (uint block=0u;block<length;++block)
        {
            argumentWords[source.z+first+block]=total;
            total+=argumentWords[source.y+first+block];
        }
        uint command=source.w+work*ArgumentStride;
        argumentWords[command]=total; argumentWords[command+1u]=uint(total!=0u);
        argumentWords[command+2u]=target.x+rangeWords[descriptor+4u];
        argumentWords[command+3u]=0u; argumentWords[command+4u]=0u;
    }
#else
    if (decisionWords[work]!=accepted) return;
    uint candidate=layoutInfo.x+work*CandidateWords;
    uint first=rangeWords[candidate],length=rangeWords[candidate+1u];
    uint block=rangeWords[candidate+2u],bin=rangeWords[candidate+3u];
    uint destination=target.x+rangeWords[layoutInfo.z+bin*BinWords+4u]+argumentWords[source.x+work]+argumentWords[source.z+block];
    for (uint index=lane;index<length;index+=Threads) argumentWords[destination+index]=sourceIndices[first+index];
#endif
}
#endif
