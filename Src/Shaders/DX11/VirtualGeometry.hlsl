// Independent D3D11 virtual geometry: interval projection, resident tree cuts,
// linear two-level prefix sums and lossless page decoding. Hoppe, SIGGRAPH 1997:
// https://hhoppe.com/proj/vdrpm/. The hierarchy/page format is project-owned.
// Dispatch Evaluate, each non-leaf height, Select, Clusters, Blocks, Decode.
// Uploads finish before use; UAVs are unbound before vertex/index consumption.
static const uint GroupWidth=64,WordBytes=4,NodeWords=20,ViewWords=36,ClusterWords=8;
static const uint NodeStateWords=8,ClusterStateWords=4,BlockWords=4;
static const uint NodeMinimum=12,NodeMaximum=16,ViewErrors=16,ViewViewport=32;
static const uint PageWords=2048,PageHeader=8,ClusterHeader=16,VertexWords=12;
static const uint FirstClusterWord=4,ClusterCountWord=5,FirstPageWord=6,PageCountWord=7,PreviousWord=10,ChildCountWord=2;
static const uint ErrorWord=0,OwnWord=1,AvailableWord=2,HistoryWord=3,SelectedWord=4,TransitionWord=5,PriorityWord=6;
static const uint VertexPrefixWord=1,IndexPrefixWord=2,ActivePrefixWord=3,PageIdWord=5,PageGenerationWord=6,PageEpochWord=7;
static const uint Invalid=0xffffffff,PositiveInfinityBits=0x7f800000,MaximumFloatBits=0x7f7fffff;
static const uint PageMagic=0x50475645,PageVersion=1; // ASCII EVGP, versioned page protocol
static const uint MissingPage=1,StalePage=2,BadPayload=4,OutputOverflow=8;
static const float FloatEpsilon=1.1920928955078125e-7,MinimumNormal=1.1754943508222875e-38;
static const float ArithmeticOperations=16.0,Roundoff=ArithmeticOperations*FloatEpsilon,Underflow=ArithmeticOperations*MinimumNormal;
cbuffer Parameters:register(b0) { uint4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; float4 Quality; };
ByteAddressBuffer Metadata:register(t0),Physical:register(t2);
StructuredBuffer<uint4> PageTable:register(t1);
RWByteAddressBuffer Scratch:register(u0),Vertices:register(u1);
RWBuffer<uint> Indices:register(u2); // typed R32_UINT UAV and native index stream
groupshared uint3 ScanValues[GroupWidth];
groupshared uint3 ScanCarry;
uint M(uint word) { return Metadata.Load(word*WordBytes); }
uint4 M4(uint word) { return Metadata.Load4(word*WordBytes); }
uint S(uint word) { return Scratch.Load(word*WordBytes); }
void Put(uint word,uint value) { Scratch.Store(word*WordBytes,value); }
uint N(uint id) { return Offsets.x+id*NodeWords; }
uint C(uint id) { return Tables.x+id*ClusterWords; }
uint ClusterState() { return Counts.x*NodeStateWords; }
uint BlockCount() { return (Limits.x+GroupWidth-1)/GroupWidth; }
uint BlockState() { return ClusterState()+Limits.x*ClusterStateWords; }
uint Status() { return BlockState()+BlockCount()*BlockWords; }
void Reject(uint reason) { uint previous; Scratch.InterlockedOr(Status()*WordBytes,reason,previous); }
uint FlatGroup(uint3 group) { return group.y*Dispatch.z+group.x; }
float Infinity() { return asfloat(PositiveInfinityBits); }
// Padding covers dot/division rounding and flush-to-zero underflow. Coefficient
// uncertainties come from the full double-precision host matrix product.
float Up(float v) { return isfinite(v) && v>=0?v*(1+Roundoff)+Underflow:Infinity(); }
float Down(float v) { return v>0?v*(1-Roundoff)-Underflow:v; }
bool Resident(uint page)
{
    uint4 map=PageTable[page]; if (map.x==Invalid) return false;
    if (map.x>=Counts.w || map.y==0 || any(map.zw!=Identity.xy)) { Reject(StalePage); return false; }
    uint base=map.x*PageWords;
    if (Physical.Load((base+PageIdWord)*WordBytes)!=page || Physical.Load((base+PageGenerationWord)*WordBytes)!=map.y || Physical.Load((base+PageEpochWord)*WordBytes)!=map.z)
    { Reject(StalePage); return false; } return true;
}
float Project(uint node,uint view)
{
    float4 minimum=asfloat(M4(N(node)+NodeMinimum)),maximum=asfloat(M4(N(node)+NodeMaximum));
    if (minimum.w==0) return 0; uint start=Tables.y+view*ViewWords;
    float wMinimum=Infinity(),xMaximum=0,yMaximum=0;
    for (uint corner=0;corner<8;++corner)
    {
        float4 p=float4((corner&1)?maximum.x:minimum.x,(corner&2)?maximum.y:minimum.y,(corner&4)?maximum.z:minimum.z,1),clip=0,error=0;
        [unroll] for (uint row=0;row<4;++row)
        { float4 a=asfloat(M4(start+row*4)),e=asfloat(M4(start+ViewErrors+row*4)); clip[row]=dot(a,p); error[row]=Up(dot(abs(a),abs(p))*Roundoff+dot(e,abs(p))); }
        float w=Down(clip.w-error.w),z=Down(clip.z-error.z); if (!(w>0) || !(z>0) || !isfinite(w)) return Infinity();
        wMinimum=min(wMinimum,w); xMaximum=max(xMaximum,Up(Up(abs(clip.x)+error.x)/w)); yMaximum=max(yMaximum,Up(Up(abs(clip.y)+error.y)/w));
    }
    float4 delta=0;
    [unroll] for (uint axis=0;axis<4;++axis)
    { float3 a=abs(asfloat(M4(start+axis*4)).xyz)+asfloat(M4(start+ViewErrors+axis*4)).xyz; delta[axis]=Up(Up(a.x+a.y+a.z)*minimum.w); }
    float denominator=Down(wMinimum-delta.w); if (!(denominator>0)) return Infinity();
    float x=Up(Up(delta.x+Up(xMaximum*delta.w))/denominator),y=Up(Up(delta.y+Up(yMaximum*delta.w))/denominator);
    float2 halfViewport=asfloat(M4(start+ViewViewport)).xy; return Up(Up(x*halfViewport.x)+Up(y*halfViewport.y));
}
bool ChildrenReady(uint node)
{
    uint4 link=M4(N(node)); if (link.z==0) return false;
    for (uint at=0;at<link.z;++at) if (S(M(Offsets.y+link.y+at)*NodeStateWords+AvailableWord)==0) return false; return true;
}
bool WantsFine(uint node) { uint s=node*NodeStateWords; return M(N(node)+ChildCountWord)!=0 && asfloat(S(s))>(S(s+HistoryWord)!=0?Quality.y:Quality.x); }
bool ChoosesSelf(uint node) { return S(node*NodeStateWords+OwnWord)!=0 && (!WantsFine(node) || !ChildrenReady(node)); }
uint3 InclusiveScan(uint3 value,uint lane)
{
    ScanValues[lane]=value; GroupMemoryBarrierWithGroupSync();
    for (uint stride=1;stride<GroupWidth;stride*=2)
    {
        uint3 add=0;
        [branch] if (lane>=stride) add=ScanValues[lane-stride];
        GroupMemoryBarrierWithGroupSync(); ScanValues[lane]+=add; GroupMemoryBarrierWithGroupSync();
    }
    return ScanValues[lane];
}
[numthreads(GroupWidth,1,1)] void CSEvaluate(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=FlatGroup(group)*GroupWidth+lane; if (id>=Counts.x) return; float pixels=0; bool own=true; uint n=N(id),s=id*NodeStateWords;
    for (uint view=0;view<Counts.y;++view) pixels=max(pixels,Project(id,view));
    for (uint at=0;at<M(n+PageCountWord);++at) own=Resident(M(Offsets.w+M(n+FirstPageWord)+at)) && own;
    Put(s,asuint(pixels)); Put(s+OwnWord,own?1:0); Put(s+AvailableWord,own?1:0); Put(s+HistoryWord,0);
}
[numthreads(GroupWidth,1,1)] void CSPropagate(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint ordinal=FlatGroup(group)*GroupWidth+lane; if (ordinal>=Dispatch.y) return;
    uint id=M(Tables.z+Dispatch.x+ordinal),s=id*NodeStateWords; uint4 link=M4(N(id)); bool ready=true,refined=false;
    for (uint at=0;at<link.z;++at)
    { uint child=M(Offsets.y+link.y+at),t=child*NodeStateWords; ready=ready && S(t+AvailableWord)!=0; refined=refined || S(t+HistoryWord)!=0 || (M(N(child)+PreviousWord)&1)!=0; }
    Put(s+AvailableWord,(S(s+OwnWord)!=0 || ready)?1:0); Put(s+HistoryWord,refined?1:0);
}
[numthreads(GroupWidth,1,1)] void CSSelect(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=FlatGroup(group)*GroupWidth+lane; if (id>=Counts.x) return; uint n=N(id),s=id*NodeStateWords; bool reached=true,suppressed=false;
    for (uint parent=M(n);parent!=Invalid;parent=M(N(parent)))
    { reached=reached && !ChoosesSelf(parent); suppressed=suppressed || (S(parent*NodeStateWords+OwnWord)==0 && ChildrenReady(parent) && !WantsFine(parent)); }
    if (!reached) return; float scale=max(Quality.x,MinimumNormal);
    if (ChoosesSelf(id))
    {
        Put(s+SelectedWord,1); for (uint at=0;at<M(n+ClusterCountWord);++at) Put(ClusterState()+M(Offsets.z+M(n+FirstClusterWord)+at)*ClusterStateWords,1);
        if (WantsFine(id) && !suppressed) { Put(s+TransitionWord,1); Put(s+PriorityWord,asuint(min(asfloat(MaximumFloatBits),1+asfloat(S(s))/scale))); }
    }
    else if (S(s+OwnWord)==0 && ChildrenReady(id) && !WantsFine(id) && !suppressed) { Put(s+TransitionWord,2); Put(s+PriorityWord,asuint(1+1/(1+asfloat(S(s))/scale))); }
    else if (S(s+OwnWord)==0 && !ChildrenReady(id)) Reject(MissingPage);
}
[numthreads(GroupWidth,1,1)] void CSClusters(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint block=FlatGroup(group); if (block>=BlockCount()) return; uint id=block*GroupWidth+lane,s=ClusterState()+id*ClusterStateWords; uint3 value=0;
    // Legacy HLSL evaluates both operands of logical/conditional expressions.
    // Explicit branches guard the final partial batch and shared scan lanes.
    [branch] if (id<Limits.x) { if (S(s)!=0) { uint4 record=M4(C(id)); value=uint3(record.z,record.w,1); } }
    uint3 inclusive=InclusiveScan(value,lane),exclusive=inclusive-value;
    if (id<Limits.x) { Put(s+VertexPrefixWord,exclusive.x); Put(s+IndexPrefixWord,exclusive.y); Put(s+ActivePrefixWord,exclusive.z); }
    if (lane==GroupWidth-1) { uint b=BlockState()+block*BlockWords; Put(b,inclusive.x); Put(b+1,inclusive.y); Put(b+2,inclusive.z); }
}
[numthreads(GroupWidth,1,1)] void CSBlocks(uint lane:SV_GroupIndex)
{
    if (lane==0) ScanCarry=0; GroupMemoryBarrierWithGroupSync();
    for (uint begin=0;begin<BlockCount();begin+=GroupWidth)
    {
        uint block=begin+lane,b=BlockState()+block*BlockWords; uint3 value=0;
        [branch] if (block<BlockCount()) value=uint3(S(b),S(b+1),S(b+2));
        uint3 inclusive=InclusiveScan(value,lane),exclusive=ScanCarry+inclusive-value;
        if (block<BlockCount()) { Put(b,exclusive.x); Put(b+1,exclusive.y); Put(b+2,exclusive.z); }
        GroupMemoryBarrierWithGroupSync(); if (lane==GroupWidth-1) ScanCarry+=inclusive; GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0)
    { Put(Status()+1,ScanCarry.x); Put(Status()+2,ScanCarry.y); Put(Status()+3,ScanCarry.z); if (ScanCarry.x==0 || ScanCarry.y==0 || ScanCarry.x>Capacity.x || ScanCarry.y>Capacity.y) Reject(OutputOverflow); }
}
[numthreads(GroupWidth,1,1)] void CSDecode(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=FlatGroup(group); if (id>=Limits.x || S(Status())!=0) return; uint s=ClusterState()+id*ClusterStateWords; if (S(s)==0) return;
    uint4 record=M4(C(id)),identity=M4(C(id)+4); if (!Resident(record.x)) return;
    uint base=PageTable[record.x].x*PageWords,h=base+record.y,firstV=record.y+ClusterHeader,firstI=firstV+record.z*VertexWords;
    uint4 header=Physical.Load4(h*WordBytes); uint used=Physical.Load((base+2)*WordBytes);
    bool valid=Physical.Load(base*WordBytes)==PageMagic && Physical.Load((base+1)*WordBytes)==PageVersion && used<=PageWords && firstI+record.w<=used &&
        all(header==uint4(record.z,record.w,firstV,firstI)) && Physical.Load((h+4)*WordBytes)==identity.x && Physical.Load((h+5)*WordBytes)==identity.z;
    if (!valid) { Reject(BadPayload); return; }
    uint b=BlockState()+(id/GroupWidth)*BlockWords,outV=S(b)+S(s+VertexPrefixWord),outI=S(b+1)+S(s+IndexPrefixWord);
    if (outV>Capacity.x || record.z>Capacity.x-outV || outI>Capacity.y || record.w>Capacity.y-outI) { Reject(OutputOverflow); return; }
    for (uint word=lane;word<record.z*VertexWords;word+=GroupWidth) Vertices.Store((outV*VertexWords+word)*WordBytes,Physical.Load((base+firstV+word)*WordBytes));
    for (uint index=lane;index<record.w;index+=GroupWidth)
    { uint local=Physical.Load((base+firstI+index)*WordBytes); if (local>=record.z) Reject(BadPayload); else Indices[outI+index]=outV+local; }
}
