// Independent D3D12 implementation of the Nanite group DAG and page decoder.
// Explicit UAV barriers order height dispatches and scans. Prefix readback
// sizes the native output buffers before the decode submission.
// Karis, Stubbe, Wihlidal, SIGGRAPH 2021: https://www.wihlidal.com/projects/nanite-deepdive/
static const uint BatchSize=64,WordsPerVector=4,BytesPerWord=4,GroupStride=16,DirectoryStride=16,ViewStride=36;
static const uint GroupScratch=4,ClusterScratch=4,BoundsLow=8,BoundsHigh=12,MatrixUncertainty=16,Viewport=32;
static const uint ErrorField=0,ResidentField=1,RefinedField=2,RequestField=3;
static const uint PageStride=16384,PhysicalStride=PageStride+WordsPerVector,PageHeader=8,ClusterHeader=44,VertexStride=12;
static const uint ChannelBase=8,ChannelWidth=20,ChannelOffset=32,PayloadLength=6,IndexOffset=7;
static const uint PageUsed=2,PageFlags=4,PagePrecision=5,PageNumber=6,PageReserved=7;
static const uint WordBits=32,PositionAxes=3,CornerLimit=768,ExponentBias=127,MantissaBits=23,MaximumSigned=0x7fffffff;
static const uint NoGroup=0xffffffff,InfiniteBits=0x7f800000,PageSignature=0x47504e45,FormatVersion=1;
static const uint StaleIdentity=1,MalformedPayload=2,CapacityExceeded=4,MissingPayload=8;
static const float MachineEpsilon=1.1920928955078125e-7,SmallestNormal=1.1754943508222875e-38;
static const float OperationAllowance=16,RoundAllowance=OperationAllowance*MachineEpsilon,TinyAllowance=OperationAllowance*SmallestNormal;
cbuffer Frame:register(b0) { uint4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; float4 Quality; };
StructuredBuffer<uint> Heap:register(t0),Pool:register(t2);
StructuredBuffer<uint4> PageMap:register(t1);
RWStructuredBuffer<uint> Scratch:register(u0),IndexOutput:register(u2);
RWByteAddressBuffer VertexOutput:register(u1);
groupshared uint3 Prefix[BatchSize]; groupshared uint3 PrefixCarry; groupshared uint CanDecode;
uint4 Read4(uint word) { return uint4(Heap[word],Heap[word+1],Heap[word+2],Heap[word+3]); }
uint4 Scratch4(uint word) { return uint4(Scratch[word],Scratch[word+1],Scratch[word+2],Scratch[word+3]); }
void Write4(uint word,uint4 value) { Scratch[word]=value.x; Scratch[word+1]=value.y; Scratch[word+2]=value.z; Scratch[word+3]=value.w; }
uint GroupRecord(uint id) { return Offsets.x+id*GroupStride; }
uint ClusterRecord(uint id) { return Offsets.w+id*DirectoryStride; }
uint ClusterStart() { return Counts.x*GroupScratch; }
uint BatchStart() { return ClusterStart()+Tables.w*ClusterScratch; }
uint Summary() { return BatchStart()+Limits.z*WordsPerVector; }
uint Batch(uint3 id) { return id.y*Dispatch.z+id.x; }
void Fail(uint code) { InterlockedOr(Scratch[Summary()],code); }
bool Available(uint page)
{
    uint4 map=PageMap[page]; bool present=false;
    [branch] if (map.x!=NoGroup)
    {
        [branch] if (map.x<Counts.w && map.y!=0 && all(map.zw==Identity.xy))
        {
            uint base=map.x*PhysicalStride;
            present=Pool[base]==page && Pool[base+1]==map.y && Pool[base+2]==map.z && Pool[base+3]==map.w;
            if (!present) Fail(StaleIdentity);
        }
        else Fail(StaleIdentity);
    }
    return present;
}
float Widen(float value) { return isfinite(value) && value>=0?value*(1+RoundAllowance)+TinyAllowance:asfloat(InfiniteBits); }
float Narrow(float value) { return value>0?value*(1-RoundAllowance)-TinyAllowance:value; }
float ScreenError(uint group,uint view)
{
    uint record=GroupRecord(group),viewRecord=Tables.x+view*ViewStride;
    float4 low=asfloat(Read4(record+BoundsLow)),high=asfloat(Read4(record+BoundsHigh)); if (low.w==0) return 0;
    float closestW=asfloat(InfiniteBits); float2 largestNdc=0;
    const uint BoxCornerCount=8,RowCount=4;
    for (uint corner=0;corner<BoxCornerCount;++corner)
    {
        float4 cornerPosition=float4((corner&1)?high.x:low.x,(corner&2)?high.y:low.y,(corner&4)?high.z:low.z,1),clip=0,bound=0;
        [unroll] for (uint row=0;row<RowCount;++row)
        {
            float4 coefficients=asfloat(Read4(viewRecord+row*WordsPerVector)),uncertainty=asfloat(Read4(viewRecord+MatrixUncertainty+row*WordsPerVector));
            clip[row]=dot(coefficients,cornerPosition); bound[row]=Widen(dot(abs(coefficients),abs(cornerPosition))*RoundAllowance+dot(uncertainty,abs(cornerPosition)));
        }
        float w=Narrow(clip.w-bound.w),z=Narrow(clip.z-bound.z);
        if (!(w>0) || !(z>0) || !isfinite(w)) return asfloat(InfiniteBits);
        closestW=min(closestW,w); largestNdc=max(largestNdc,float2(Widen(Widen(abs(clip.x)+bound.x)/w),Widen(Widen(abs(clip.y)+bound.y)/w)));
    }
    float4 movement=0;
    [unroll] for (uint row=0;row<RowCount;++row)
    {
        float norm=0;
        for (uint axis=0;axis<PositionAxes;++axis) norm+=abs(asfloat(Heap[viewRecord+row*WordsPerVector+axis]))+asfloat(Heap[viewRecord+MatrixUncertainty+row*WordsPerVector+axis]);
        movement[row]=Widen(Widen(norm)*low.w);
    }
    float denominator=Narrow(closestW-movement.w); if (!(denominator>0)) return asfloat(InfiniteBits);
    float dx=Widen(Widen(movement.x+Widen(largestNdc.x*movement.w))/denominator),dy=Widen(Widen(movement.y+Widen(largestNdc.y*movement.w))/denominator);
    return Widen(Widen(dx*asfloat(Heap[viewRecord+Viewport]))+Widen(dy*asfloat(Heap[viewRecord+Viewport+1])));
}
[numthreads(BatchSize,1,1)] void CSEvaluate(uint3 batch:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint offset=Batch(batch)*BatchSize+lane; [branch] if (offset>=Dispatch.y) return;
    uint id=Heap[Tables.y+Dispatch.x+offset],record=GroupRecord(id); uint4 links=Read4(record),pages=Read4(record+WordsPerVector);
    float error=0;
    for (uint view=0;view<Counts.y;++view) error=max(error,ScreenError(id,view));
    for (uint child=0;child<links.y;++child)
    { uint dependency=Read4(ClusterRecord(Heap[Offsets.y+links.x+child])+WordsPerVector).z; [branch] if (dependency!=NoGroup) error=max(error,asfloat(Scratch[dependency*GroupScratch+ErrorField])); }
    bool complete=true; for (uint page=0;page<pages.y;++page) { bool loaded=Available(pages.x+page); complete=complete && loaded; }
    Write4(id*GroupScratch,uint4(asuint(error),complete?1:0,0,error>Quality.x?1:0));
}
[numthreads(BatchSize,1,1)] void CSRefine(uint3 batch:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint offset=Batch(batch)*BatchSize+lane; [branch] if (offset>=Dispatch.y) return;
    uint id=Heap[Tables.y+Dispatch.x+offset]; uint4 links=Read4(GroupRecord(id)); bool permitted=true;
    for (uint parent=0;parent<links.w;++parent)
    { uint consumer=Read4(ClusterRecord(Heap[Offsets.z+links.z+parent])+WordsPerVector).w; [branch] if (consumer!=NoGroup) { if (Scratch[consumer*GroupScratch+RefinedField]==0) permitted=false; } }
    uint state=id*GroupScratch; Scratch[state+RefinedField]=permitted && Scratch[state+ResidentField]!=0 && Scratch[state+RequestField]!=0?1:0;
}
void ScanInclusive(uint lane)
{
    for (uint distance=1;distance<BatchSize;distance*=2)
    { uint3 add=0; [branch] if (lane>=distance) add=Prefix[lane-distance]; GroupMemoryBarrierWithGroupSync(); Prefix[lane]+=add; GroupMemoryBarrierWithGroupSync(); }
}
[numthreads(BatchSize,1,1)] void CSClusters(uint3 batch:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint block=Batch(batch); if (block>=Limits.z) return;
    uint id=block*BatchSize+lane; bool selected=false; uint3 amount=0;
    // Root descriptors have no element-count bounds. Legacy HLSL evaluates
    // both sides of boolean/ternary expressions; sentinel and partial-batch
    // loads need explicit branches, not short-circuit assumptions.
    [branch] if (id<Tables.w)
    {
        uint4 geometry=Read4(ClusterRecord(id)),owners=Read4(ClusterRecord(id)+WordsPerVector);
        bool ancestorReady=true,producerRetained=true;
        [branch] if (owners.w!=NoGroup) ancestorReady=Scratch[owners.w*GroupScratch+RefinedField]!=0;
        [branch] if (owners.z!=NoGroup) producerRetained=Scratch[owners.z*GroupScratch+RefinedField]==0;
        selected=ancestorReady && producerRetained;
        if (selected) amount=uint3(geometry.zw,1);
    }
    Prefix[lane]=amount; GroupMemoryBarrierWithGroupSync(); ScanInclusive(lane);
    if (id<Tables.w) Write4(ClusterStart()+id*ClusterScratch,uint4(selected?1:0,Prefix[lane]-amount));
    if (lane==BatchSize-1) Write4(BatchStart()+block*WordsPerVector,uint4(Prefix[lane],0));
}
[numthreads(BatchSize,1,1)] void CSBlocks(uint lane:SV_GroupIndex)
{
    if (lane==0) PrefixCarry=0; GroupMemoryBarrierWithGroupSync();
    for (uint first=0;first<Limits.z;first+=BatchSize)
    {
        uint id=first+lane; uint3 amount=0;
        [branch] if (id<Limits.z) amount=Scratch4(BatchStart()+id*WordsPerVector).xyz;
        Prefix[lane]=amount; GroupMemoryBarrierWithGroupSync(); ScanInclusive(lane);
        uint3 start=PrefixCarry+Prefix[lane]-amount; if (id<Limits.z) Write4(BatchStart()+id*WordsPerVector,uint4(start,0));
        GroupMemoryBarrierWithGroupSync(); if (lane==0) PrefixCarry+=Prefix[BatchSize-1]; GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0)
    { if (PrefixCarry.x>Capacity.x || PrefixCarry.y>Capacity.y) Fail(CapacityExceeded); Scratch[Summary()+1]=PrefixCarry.x; Scratch[Summary()+2]=PrefixCarry.y; Scratch[Summary()+3]=PrefixCarry.z; }
}
uint Extract(uint payload,uint bit,uint width)
{
    if (width==0) return 0; uint index=bit/WordBits,shift=bit%WordBits,value=Pool[payload+index]>>shift;
    if (shift+width>WordBits) value|=Pool[payload+index+1]<<(WordBits-shift);
    if (width==WordBits) return value; return value&((1u<<width)-1);
}
bool CheckPayload(uint page,uint header,uint4 geometry,uint4 owners)
{
    uint used=Pool[page+PageUsed];
    if (Pool[page]!=PageSignature || Pool[page+1]!=FormatVersion || used>PageStride || used<PageHeader+ClusterHeader ||
        Pool[page+PageFlags]>1 || Pool[page+PagePrecision]!=Limits.w || Pool[page+PageNumber]!=geometry.x || Pool[page+PageReserved]!=0 ||
        geometry.y>used-ClusterHeader || geometry.z==0 || geometry.z>CornerLimit || geometry.w==0 || geometry.w>CornerLimit) return false;
    if (Pool[header]!=owners.y || Pool[header+1]!=geometry.z || Pool[header+2]!=geometry.w || Pool[header+3]!=owners.x ||
        Pool[header+WordsPerVector]!=owners.z || Pool[header+WordsPerVector+1]!=owners.w ||
        Pool[header+PayloadLength]<ClusterHeader || Pool[header+PayloadLength]>used-geometry.y) return false;
    uint bitCount=0;
    for (uint channel=0;channel<VertexStride;++channel)
    { uint width=Pool[header+ChannelWidth+channel]; if (width>WordBits || Pool[header+ChannelOffset+channel]!=bitCount) return false; bitCount+=geometry.z*width; }
    if (Pool[header+IndexOffset]!=bitCount) return false;
    uint indexBits=0; for (uint value=geometry.z-1;value!=0;value>>=1) ++indexBits;
    bitCount+=geometry.w*indexBits; return Pool[header+PayloadLength]==ClusterHeader+(bitCount+WordBits-1)/WordBits;
}
[numthreads(BatchSize,1,1)] void CSDecode(uint3 batch:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=Batch(batch); if (id>=Tables.w) return; uint4 selected=Scratch4(ClusterStart()+id*ClusterScratch);
    uint4 geometry=Read4(ClusterRecord(id)),owners=Read4(ClusterRecord(id)+WordsPerVector),prefix=Scratch4(BatchStart()+(id/BatchSize)*WordsPerVector);
    uint firstVertex=prefix.x+selected.y,firstIndex=prefix.y+selected.z;
    if (lane==0)
    {
        CanDecode=0;
        if (selected.x!=0)
        {
            if (!Available(geometry.x)) Fail(MissingPayload);
            else
            {
                uint page=PageMap[geometry.x].x*PhysicalStride+WordsPerVector;
                if (!CheckPayload(page,page+geometry.y,geometry,owners)) Fail(MalformedPayload);
                else if (firstVertex>Capacity.x || geometry.z>Capacity.x-firstVertex || firstIndex>Capacity.y || geometry.w>Capacity.y-firstIndex) Fail(CapacityExceeded);
                else CanDecode=1;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync(); if (CanDecode==0) return;
    uint header=PageMap[geometry.x].x*PhysicalStride+WordsPerVector+geometry.y,payload=header+ClusterHeader;
    float gridScale=asfloat(uint(asint(Limits.w)+ExponentBias)<<MantissaBits);
    for (uint vertex=lane;vertex<geometry.z;vertex+=BatchSize) for (uint channel=0;channel<VertexStride;++channel)
    {
        uint base=Pool[header+ChannelBase+channel],width=Pool[header+ChannelWidth+channel];
        uint delta=Extract(payload,Pool[header+ChannelOffset+channel]+vertex*width,width),bits=base^delta;
        if (channel<PositionAxes) { if (delta>MaximumSigned-base) Fail(MalformedPayload); bits=asuint(float(asint(base+delta))*gridScale); }
        if (!isfinite(asfloat(bits))) Fail(MalformedPayload);
        VertexOutput.Store(((firstVertex+vertex)*VertexStride+channel)*BytesPerWord,bits);
    }
    uint indexBits=0; for (uint value=geometry.z-1;value!=0;value>>=1) ++indexBits;
    for (uint index=lane;index<geometry.w;index+=BatchSize)
    { uint local=Extract(payload,Pool[header+IndexOffset]+index*indexBits,indexBits); if (local>=geometry.z) Fail(MalformedPayload); else IndexOutput[firstIndex+index]=firstVertex+local; }
}
