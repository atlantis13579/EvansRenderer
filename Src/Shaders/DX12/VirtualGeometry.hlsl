// D3D12 owns this complete virtual-geometry compute implementation. References:
// Hoppe, "View-dependent refinement of progressive meshes", SIGGRAPH 1997,
// https://hhoppe.com/proj/vdrpm/. Project-owned lossless pages and boundary locks.
// UAV barriers separate height propagation and prefix phases. A final UAV/read
// transition is required before native vertex/index or host consumption.
static const uint WaveBatch=64,NodeStride=20,ViewStride=36,DirectoryStride=8,NodeScratchStride=8,ClusterScratchStride=4;
static const uint GeometryWord=4,PreviousWord=10,MinimumWord=12,MaximumWord=16,ViewErrorWord=16,ViewportWord=32;
static const uint FirstClusterWord=4,ClusterCountWord=5,FirstPageWord=6,PageCountWord=7;
static const uint ErrorWord=0,OwnWord=1,AvailableWord=2,HistoryWord=3,SelectedWord=4,TransitionWord=5,PriorityWord=6;
static const uint VertexPrefixWord=1,IndexPrefixWord=2,ActivePrefixWord=3;
static const uint WordsPerPage=2048,PageHeaderWords=8,BlockHeaderWords=16,WordsPerVertex=12,BytesPerWord=4;
static const uint PageIdWord=5,PageGenerationWord=6,PageEpochWord=7,UsedWordsWord=2,MaterialWord=4,DirectoryIdWord=5;
static const uint NoPage=0xffffffff,InfinityEncoding=0x7f800000,MaximumFiniteEncoding=0x7f7fffff,PageSignature=0x50475645,FormatVersion=1;
static const uint MissingCoverage=1,StaleIdentity=2,MalformedPage=4,CapacityExceeded=8,PreviouslySelected=1;
static const float MachineEpsilon=1.1920928955078125e-7,NormalMinimum=1.1754943508222875e-38;
static const float ArithmeticOperations=16.0,ArithmeticAllowance=ArithmeticOperations*MachineEpsilon,UnderflowAllowance=ArithmeticOperations*NormalMinimum;
cbuffer Frame:register(b0) { uint4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; float4 Quality; };
StructuredBuffer<uint> Heap:register(t0),PageWords:register(t2);
StructuredBuffer<uint4> Mapping:register(t1);
RWStructuredBuffer<uint> State:register(u0),IndexStream:register(u2);
RWByteAddressBuffer VertexStream:register(u1);
groupshared uint3 Prefix[WaveBatch]; groupshared uint3 Carry;
uint4 Read4(uint p) { return uint4(Heap[p],Heap[p+1],Heap[p+2],Heap[p+3]); }
uint Node(uint i) { return Offsets.x+i*NodeStride; }
uint Directory(uint i) { return Tables.x+i*DirectoryStride; }
uint ClusterStart() { return Counts.x*NodeScratchStride; }
uint Batches() { return (Limits.x+WaveBatch-1)/WaveBatch; }
uint BatchStart() { return ClusterStart()+Limits.x*ClusterScratchStride; }
uint Summary() { return BatchStart()+Batches()*4; }
uint Group(uint3 g) { return g.y*Dispatch.z+g.x; }
void Fail(uint flag) { uint old; InterlockedOr(State[Summary()],flag,old); }
float UpperBound(float x) { return isfinite(x) && x>=0?x*(1+ArithmeticAllowance)+UnderflowAllowance:asfloat(InfinityEncoding); }
float LowerBound(float x) { return x>0?x*(1-ArithmeticAllowance)-UnderflowAllowance:x; }
bool PagePresent(uint virtualPage)
{
    uint4 m=Mapping[virtualPage]; if (m.x==NoPage) return false;
    if (m.x>=Counts.w || m.y==0 || any(m.zw!=Identity.xy)) { Fail(StaleIdentity); return false; }
    uint p=m.x*WordsPerPage;
    if (PageWords[p+PageIdWord]!=virtualPage || PageWords[p+PageGenerationWord]!=m.y || PageWords[p+PageEpochWord]!=m.z) { Fail(StaleIdentity); return false; }
    return true;
}
float PixelBound(uint node,uint view)
{
    float4 lo=asfloat(Read4(Node(node)+MinimumWord)),hi=asfloat(Read4(Node(node)+MaximumWord)); if (lo.w==0) return 0;
    uint v=Tables.y+view*ViewStride; float minW=asfloat(InfinityEncoding),maxX=0,maxY=0;
    for (uint corner=0;corner<8;++corner)
    {
        float4 position=float4((corner&1)?hi.x:lo.x,(corner&2)?hi.y:lo.y,(corner&4)?hi.z:lo.z,1),clip=0,uncertainty=0;
        [unroll] for (uint row=0;row<4;++row)
        { float4 coefficients=asfloat(Read4(v+row*4)),error=asfloat(Read4(v+ViewErrorWord+row*4)); clip[row]=dot(coefficients,position); uncertainty[row]=UpperBound(dot(abs(coefficients),abs(position))*ArithmeticAllowance+dot(error,abs(position))); }
        float w=LowerBound(clip.w-uncertainty.w),z=LowerBound(clip.z-uncertainty.z); if (!(w>0) || !(z>0) || !isfinite(w)) return asfloat(InfinityEncoding);
        minW=min(minW,w); maxX=max(maxX,UpperBound(UpperBound(abs(clip.x)+uncertainty.x)/w)); maxY=max(maxY,UpperBound(UpperBound(abs(clip.y)+uncertainty.y)/w));
    }
    float4 radius=0;
    [unroll] for (uint axis=0;axis<4;++axis)
    { float3 interval=abs(asfloat(Read4(v+axis*4)).xyz)+asfloat(Read4(v+ViewErrorWord+axis*4)).xyz; radius[axis]=UpperBound(UpperBound(interval.x+interval.y+interval.z)*lo.w); }
    float denominator=LowerBound(minW-radius.w); if (!(denominator>0)) return asfloat(InfinityEncoding);
    float x=UpperBound(UpperBound(radius.x+UpperBound(maxX*radius.w))/denominator),y=UpperBound(UpperBound(radius.y+UpperBound(maxY*radius.w))/denominator);
    float2 viewport=asfloat(Read4(v+ViewportWord)).xy; return UpperBound(UpperBound(x*viewport.x)+UpperBound(y*viewport.y));
}
bool AllChildren(uint id)
{
    uint4 n=Read4(Node(id)); if (n.z==0) return false;
    for (uint c=0;c<n.z;++c) if (State[Heap[Offsets.y+n.y+c]*NodeScratchStride+AvailableWord]==0) return false; return true;
}
bool Refine(uint id) { uint s=id*NodeScratchStride; return Read4(Node(id)).z!=0 && asfloat(State[s+ErrorWord])>(State[s+HistoryWord]!=0?Quality.y:Quality.x); }
bool Pick(uint id) { return State[id*NodeScratchStride+OwnWord]!=0 && (!Refine(id) || !AllChildren(id)); }
uint3 Scan(uint3 value,uint lane)
{
    Prefix[lane]=value; GroupMemoryBarrierWithGroupSync();
    for (uint step=1;step<WaveBatch;step*=2)
    {
        uint3 increment=0;
        [branch] if (lane>=step) increment=Prefix[lane-step];
        GroupMemoryBarrierWithGroupSync(); Prefix[lane]+=increment; GroupMemoryBarrierWithGroupSync();
    }
    return Prefix[lane];
}
[numthreads(WaveBatch,1,1)] void CSEvaluate(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=Group(group)*WaveBatch+lane; if (id>=Counts.x) return; uint n=Node(id),s=id*NodeScratchStride; bool present=true; float error=0;
    for (uint view=0;view<Counts.y;++view) error=max(error,PixelBound(id,view));
    for (uint page=0;page<Heap[n+PageCountWord];++page) present=PagePresent(Heap[Offsets.w+Heap[n+FirstPageWord]+page]) && present;
    State[s+ErrorWord]=asuint(error); State[s+OwnWord]=present?1:0; State[s+AvailableWord]=present?1:0; State[s+HistoryWord]=0;
}
[numthreads(WaveBatch,1,1)] void CSPropagate(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint ordinal=Group(group)*WaveBatch+lane; if (ordinal>=Dispatch.y) return; uint id=Heap[Tables.z+Dispatch.x+ordinal],s=id*NodeScratchStride; uint4 n=Read4(Node(id)); bool ready=true,history=false;
    for (uint c=0;c<n.z;++c) { uint child=Heap[Offsets.y+n.y+c],t=child*NodeScratchStride; ready=ready && State[t+AvailableWord]!=0; history=history || State[t+HistoryWord]!=0 || (Heap[Node(child)+PreviousWord]&PreviouslySelected)!=0; }
    State[s+AvailableWord]=(State[s+OwnWord]!=0 || ready)?1:0; State[s+HistoryWord]=history?1:0;
}
[numthreads(WaveBatch,1,1)] void CSSelect(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=Group(group)*WaveBatch+lane; if (id>=Counts.x) return; uint n=Node(id),s=id*NodeScratchStride; bool reachable=true,suppressed=false;
    for (uint parent=Heap[n];parent!=NoPage;parent=Heap[Node(parent)]) { reachable=reachable && !Pick(parent); suppressed=suppressed || (State[parent*NodeScratchStride+OwnWord]==0 && AllChildren(parent) && !Refine(parent)); }
    if (!reachable) return; float scale=max(Quality.x,NormalMinimum);
    if (Pick(id))
    {
        State[s+SelectedWord]=1; for (uint c=0;c<Heap[n+ClusterCountWord];++c) State[ClusterStart()+Heap[Offsets.z+Heap[n+FirstClusterWord]+c]*ClusterScratchStride]=1;
        if (Refine(id) && !suppressed) { State[s+TransitionWord]=1; State[s+PriorityWord]=asuint(min(asfloat(MaximumFiniteEncoding),1+asfloat(State[s+ErrorWord])/scale)); }
    }
    else if (State[s+OwnWord]==0 && AllChildren(id) && !Refine(id) && !suppressed) { State[s+TransitionWord]=2; State[s+PriorityWord]=asuint(1+1/(1+asfloat(State[s+ErrorWord])/scale)); }
    else if (State[s+OwnWord]==0 && !AllChildren(id)) Fail(MissingCoverage);
}
[numthreads(WaveBatch,1,1)] void CSClusters(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint block=Group(group); if (block>=Batches()) return; uint id=block*WaveBatch+lane,s=ClusterStart()+id*ClusterScratchStride; uint3 value=0;
    // Root descriptors have no element-count bounds. Do not rely on HLSL
    // short-circuit/ternary evaluation to protect partial-batch memory loads.
    // https://devblogs.microsoft.com/directx/hlsl-2021-migration-guide/
    [branch] if (id<Limits.x) { if (State[s]!=0) { uint4 d=Read4(Directory(id)); value=uint3(d.z,d.w,1); } }
    uint3 inclusive=Scan(value,lane),exclusive=inclusive-value;
    if (id<Limits.x) { State[s+VertexPrefixWord]=exclusive.x; State[s+IndexPrefixWord]=exclusive.y; State[s+ActivePrefixWord]=exclusive.z; }
    if (lane==WaveBatch-1) { uint b=BatchStart()+block*4; State[b]=inclusive.x; State[b+1]=inclusive.y; State[b+2]=inclusive.z; }
}
[numthreads(WaveBatch,1,1)] void CSBlocks(uint lane:SV_GroupIndex)
{
    if (lane==0) Carry=0; GroupMemoryBarrierWithGroupSync();
    for (uint first=0;first<Batches();first+=WaveBatch)
    {
        uint block=first+lane,b=BatchStart()+block*4; uint3 value=0;
        [branch] if (block<Batches()) value=uint3(State[b],State[b+1],State[b+2]);
        uint3 inclusive=Scan(value,lane),exclusive=Carry+inclusive-value;
        if (block<Batches()) { State[b]=exclusive.x; State[b+1]=exclusive.y; State[b+2]=exclusive.z; }
        GroupMemoryBarrierWithGroupSync(); if (lane==WaveBatch-1) Carry+=inclusive; GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0) { State[Summary()+1]=Carry.x; State[Summary()+2]=Carry.y; State[Summary()+3]=Carry.z; if (Carry.x==0 || Carry.y==0 || Carry.x>Capacity.x || Carry.y>Capacity.y) Fail(CapacityExceeded); }
}
[numthreads(WaveBatch,1,1)] void CSDecode(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=Group(group); if (id>=Limits.x || State[Summary()]!=0) return; uint s=ClusterStart()+id*ClusterScratchStride; if (State[s]==0) return;
    uint4 d=Read4(Directory(id)),identity=Read4(Directory(id)+4); if (!PagePresent(d.x)) return; uint p=Mapping[d.x].x*WordsPerPage,h=p+d.y,v=d.y+BlockHeaderWords,i=v+d.z*WordsPerVertex,used=PageWords[p+UsedWordsWord];
    bool valid=PageWords[p]==PageSignature && PageWords[p+1]==FormatVersion && used<=WordsPerPage && i+d.w<=used &&
        PageWords[h]==d.z && PageWords[h+1]==d.w && PageWords[h+2]==v && PageWords[h+3]==i && PageWords[h+MaterialWord]==identity.x && PageWords[h+DirectoryIdWord]==identity.z;
    if (!valid) { Fail(MalformedPage); return; }
    uint b=BatchStart()+(id/WaveBatch)*4,outV=State[b]+State[s+VertexPrefixWord],outI=State[b+1]+State[s+IndexPrefixWord];
    if (outV>Capacity.x || d.z>Capacity.x-outV || outI>Capacity.y || d.w>Capacity.y-outI) { Fail(CapacityExceeded); return; }
    for (uint word=lane;word<d.z*WordsPerVertex;word+=WaveBatch) VertexStream.Store((outV*WordsPerVertex+word)*BytesPerWord,PageWords[p+v+word]);
    for (uint at=lane;at<d.w;at+=WaveBatch) { uint local=PageWords[p+i+at]; if (local>=d.z) Fail(MalformedPage); else IndexStream[outI+at]=outV+local; }
}
