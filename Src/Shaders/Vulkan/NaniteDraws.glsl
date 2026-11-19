#version 450
// VkDrawIndexedIndirectCommand: indexCount, instanceCount, firstIndex,
// signed vertexOffset, firstInstance. Main and post accept disjoint states.
layout(local_size_x=64) in;
layout(set=0,binding=0,std140) uniform Parameters
{
    uvec4 counts; uvec4 layoutInfo; uvec4 source; uvec4 target;
};
#ifndef EVANS_NANITE_BIN_PASS
layout(set=0,binding=1,std430) readonly buffer ClusterRanges { uvec2 ranges[]; };
#else
layout(set=0,binding=1,std430) readonly buffer BinMetadata { uint metadata[]; };
layout(set=0,binding=4,std430) readonly buffer OriginalIndexStream { uint originalIndices[]; };
#endif
layout(set=0,binding=2,std430) buffer IndirectCommands { uint commands[]; };
layout(set=0,binding=3,std430) readonly buffer Classifications { uint states[]; };
const uint GroupWidth=64u,CommandWords=5u;
const uint MainVisible=2u,PostVisible=4u;
#ifndef EVANS_NANITE_BIN_PASS
void main()
{
    uint ordinal=(gl_WorkGroupID.y*target.w+gl_WorkGroupID.x)*GroupWidth+gl_LocalInvocationIndex;
    if (ordinal>=counts.x) return;
    uvec2 range=ranges[ordinal];
    uint accepted=counts.w==0u?MainVisible:PostVisible;
    uint base=ordinal*CommandWords;
    commands[base]=range.y;
    commands[base+1u]=states[ordinal]==accepted?1u:0u;
    commands[base+2u]=range.x;
    commands[base+3u]=0u;
    commands[base+4u]=0u;
}
#else
const uint CandidateWords=4u,BlockWords=4u,BinWords=8u;
shared uint prefix[GroupWidth];
uvec4 recordAt(uint word)
{ return uvec4(metadata[word],metadata[word+1u],metadata[word+2u],metadata[word+3u]); }
void scan(uint lane,uint value)
{
    prefix[lane]=value; barrier();
    for (uint distance=1u;distance<GroupWidth;distance*=2u)
    {
        uint previous=0u; if (lane>=distance) previous=prefix[lane-distance];
        barrier(); prefix[lane]+=previous; barrier();
    }
}
void main()
{
    uint job=gl_WorkGroupID.y*target.w+gl_WorkGroupID.x,lane=gl_LocalInvocationIndex;
    if (job>=target.z) return;
    uint accepted=counts.w==0u?MainVisible:PostVisible;
#if EVANS_NANITE_BIN_PASS == 0
    uvec4 block=recordAt(layoutInfo.y+job*BlockWords);
    uint query=block.x+lane,count=0u;
    if (lane<block.y && states[query]==accepted) count=metadata[layoutInfo.x+query*CandidateWords+1u];
    scan(lane,count);
    if (lane<block.y) commands[source.x+query]=prefix[lane]-count;
    if (lane==GroupWidth-1u) commands[source.y+job]=prefix[lane];
#elif EVANS_NANITE_BIN_PASS == 1
    uvec4 bin=recordAt(layoutInfo.z+job*BinWords);
    uint running=0u;
    for (uint first=0u;first<bin.w;first+=GroupWidth)
    {
        uint block=first+lane,value=0u;
        if (block<bin.w) value=commands[source.y+bin.z+block];
        scan(lane,value);
        if (block<bin.w) commands[source.z+bin.z+block]=running+prefix[lane]-value;
        uint total=prefix[GroupWidth-1u]; barrier(); running+=total;
    }
    if (lane==0u)
    {
        uint command=source.w+job*CommandWords;
        commands[command]=running; commands[command+1u]=running>0u?1u:0u;
        commands[command+2u]=target.x+metadata[layoutInfo.z+job*BinWords+4u];
        commands[command+3u]=0u; commands[command+4u]=0u;
    }
#else
    if (states[job]!=accepted) return;
    uvec4 candidate=recordAt(layoutInfo.x+job*CandidateWords);
    uint base=target.x+metadata[layoutInfo.z+candidate.w*BinWords+4u]+commands[source.x+job]+commands[source.z+candidate.z];
    for (uint corner=lane;corner<candidate.y;corner+=GroupWidth) commands[base+corner]=originalIndices[candidate.x+corner];
#endif
}
#endif
