// Nanite hardware-raster submission: stable cluster slots, GPU-only visibility.
// D3D11's indexed indirect ABI is five 32-bit words. Zero instances skip a
// rejected cluster without changing material order or reading a count on CPU.
cbuffer DrawParameters : register(b0)
{
    uint4 Counts; uint4 Layout; uint4 Source; uint4 Target;
};
ByteAddressBuffer Ranges : register(t0);
ByteAddressBuffer Decisions : register(t1);
RWByteAddressBuffer Arguments : register(u0);
static const uint WorkgroupWidth=64;
static const uint RangeBytes=8,ArgumentBytes=20,WordBytes=4;
static const uint MainVisible=2,PostVisible=4;
[numthreads(WorkgroupWidth,1,1)]
void CSBuildDraws(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    const uint cluster=(group.y*Target.w+group.x)*WorkgroupWidth+lane;
    [branch] if (cluster>=Counts.x) return;
    const uint2 range=Ranges.Load2(cluster*RangeBytes); // First index, index count.
    const uint accepted=Counts.w==0?MainVisible:PostVisible;
    const uint visible=Decisions.Load(cluster*WordBytes)==accepted?1u:0u;
    const uint address=cluster*ArgumentBytes;
    Arguments.Store4(address,uint4(range.y,visible,range.x,0));
    Arguments.Store(address+4*WordBytes,0); // First instance.
}

ByteAddressBuffer SourceIndices : register(t2);
static const uint CandidateWords=4,BlockWords=4,BinWords=8;
groupshared uint Prefix[WorkgroupWidth];
// Inclusive Hillis-Steele scan; every lane, including a partial block's padding,
// participates in every barrier. The scatter stage never uses atomic append,
// preserving original triangle order within the material bin.
void ScanBlock(uint lane,uint value)
{
    Prefix[lane]=value; GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint offset=1;offset<WorkgroupWidth;offset*=2)
    {
        uint previous=0; [branch] if (lane>=offset) previous=Prefix[lane-offset];
        GroupMemoryBarrierWithGroupSync(); Prefix[lane]+=previous; GroupMemoryBarrierWithGroupSync();
    }
}
[numthreads(WorkgroupWidth,1,1)]
void CSCountBins(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint block=group.y*Target.w+group.x; [branch] if (block>=Target.z) return;
    uint4 descriptor=Ranges.Load4((Layout.y+block*BlockWords)*WordBytes);
    uint query=descriptor.x+lane,count=0;
    [branch] if (lane<descriptor.y)
    {
        uint accepted=Counts.w==0?MainVisible:PostVisible;
        [branch] if (Decisions.Load(query*WordBytes)==accepted)
            count=Ranges.Load((Layout.x+query*CandidateWords+1)*WordBytes);
    }
    ScanBlock(lane,count);
    if (lane<descriptor.y) Arguments.Store((Source.x+query)*WordBytes,Prefix[lane]-count);
    if (lane==WorkgroupWidth-1) Arguments.Store((Source.y+block)*WordBytes,Prefix[lane]);
}
[numthreads(WorkgroupWidth,1,1)]
void CSScanBins(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint bin=group.y*Target.w+group.x; [branch] if (bin>=Target.z) return;
    uint4 descriptor=Ranges.Load4((Layout.z+bin*BinWords)*WordBytes);
    uint outputIndex=Target.x+Ranges.Load((Layout.z+bin*BinWords+4)*WordBytes),running=0;
    for (uint first=0;first<descriptor.w;first+=WorkgroupWidth)
    {
        uint localBlock=first+lane,count=0;
        [branch] if (localBlock<descriptor.w) count=Arguments.Load((Source.y+descriptor.z+localBlock)*WordBytes);
        ScanBlock(lane,count);
        if (localBlock<descriptor.w) Arguments.Store((Source.z+descriptor.z+localBlock)*WordBytes,running+Prefix[lane]-count);
        uint batch=Prefix[WorkgroupWidth-1]; GroupMemoryBarrierWithGroupSync(); running+=batch;
    }
    if (lane==0)
    {
        uint address=(Source.w+bin*ArgumentBytes/WordBytes)*WordBytes;
        Arguments.Store4(address,uint4(running,running==0?0u:1u,outputIndex,0)); Arguments.Store(address+4*WordBytes,0);
    }
}
[numthreads(WorkgroupWidth,1,1)]
void CSScatterBins(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint query=group.y*Target.w+group.x; [branch] if (query>=Target.z) return;
    uint accepted=Counts.w==0?MainVisible:PostVisible; [branch] if (Decisions.Load(query*WordBytes)!=accepted) return;
    uint4 candidate=Ranges.Load4((Layout.x+query*CandidateWords)*WordBytes);
    uint outputIndex=Ranges.Load((Layout.z+candidate.w*BinWords+4)*WordBytes);
    uint destination=Target.x+outputIndex+Arguments.Load((Source.x+query)*WordBytes)+Arguments.Load((Source.z+candidate.z)*WordBytes);
    for (uint index=lane;index<candidate.y;index+=WorkgroupWidth)
        Arguments.Store((destination+index)*WordBytes,SourceIndices.Load((candidate.x+index)*WordBytes));
}
