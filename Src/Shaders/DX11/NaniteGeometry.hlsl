// D3D11 Nanite DAG consumer: upward shared error, downward whole-group
// refinement, material-ordered prefix scans and compressed cluster decoding.
// Karis et al., "A Deep Dive into Nanite Virtualized Geometry", SIGGRAPH 2021.
// Each level is a separate ordered dispatch; no cross-workgroup spin barriers.
static const uint WaveWidth=64,WordBytes=4,VectorWords=4,GroupWords=16,ClusterWords=16,ViewWords=36;
static const uint GroupStateWords=4,ClusterStateWords=4,MinimumWords=8,MaximumWords=12,ViewErrors=16,ViewViewport=32;
static const uint ErrorWord=0,CompleteWord=1,RefinedWord=2,RequestedWord=3;
static const uint PageWords=16384,PhysicalWords=PageWords+VectorWords,PageHeaderWords=8,ClusterHeaderWords=44,VertexWords=12;
static const uint HeaderBases=8,HeaderWidths=20,HeaderOffsets=32,HeaderWordCount=6,HeaderIndexOffset=7;
static const uint PageUsedWord=2,PageFlagsWord=4,PagePrecisionWord=5,PageIdWord=6,PageReservedWord=7;
static const uint BitsPerWord=32,PositionChannels=3,MaximumCorners=768,FloatExponentBias=127,FloatMantissaBits=23;
static const uint Invalid=0xffffffff,MaximumSigned=0x7fffffff,InfinityBits=0x7f800000,PageMagic=0x47504e45,PageVersion=1;
static const uint StalePage=1,BadPayload=2,OutputOverflow=4,MissingPage=8;
static const float FloatEpsilon=1.1920928955078125e-7,MinimumNormal=1.1754943508222875e-38;
static const float ArithmeticOperations=16,Roundoff=ArithmeticOperations*FloatEpsilon,Underflow=ArithmeticOperations*MinimumNormal;
cbuffer Parameters:register(b0) { uint4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; float4 Quality; };
ByteAddressBuffer Metadata:register(t0),Physical:register(t2);
StructuredBuffer<uint4> Mapping:register(t1);
RWByteAddressBuffer State:register(u0),Vertices:register(u1);
RWBuffer<uint> Indices:register(u2);
groupshared uint3 Scan[WaveWidth];
groupshared uint3 Carry;
groupshared uint DecodeReady;
uint M(uint word) { return Metadata.Load(word*WordBytes); }
uint4 M4(uint word) { return Metadata.Load4(word*WordBytes); }
uint S(uint word) { return State.Load(word*WordBytes); }
uint4 S4(uint word) { return State.Load4(word*WordBytes); }
void Put(uint word,uint value) { State.Store(word*WordBytes,value); }
uint G(uint id) { return Offsets.x+id*GroupWords; }
uint C(uint id) { return Offsets.w+id*ClusterWords; }
uint ClusterStates() { return Counts.x*GroupStateWords; }
uint BlockStates() { return ClusterStates()+Tables.w*ClusterStateWords; }
uint Status() { return BlockStates()+Limits.z*VectorWords; }
uint Block(uint3 group) { return group.y*Dispatch.z+group.x; }
void Error(uint mask) { uint previous; State.InterlockedOr(Status()*WordBytes,mask,previous); }
uint P(uint word) { return Physical.Load(word*WordBytes); }
bool Resident(uint page)
{
    uint4 entry=Mapping[page]; if (entry.x==Invalid) return false;
    if (entry.x>=Counts.w || entry.y==0 || any(entry.zw!=Identity.xy)) { Error(StalePage); return false; }
    uint4 stored=Physical.Load4(entry.x*PhysicalWords*WordBytes);
    if (any(stored!=uint4(page,entry.yzw))) { Error(StalePage); return false; }
    return true;
}
float Expand(float v) { return isfinite(v) && v>=0?v*(1+Roundoff)+Underflow:asfloat(InfinityBits); }
float Contract(float v) { return v>0?v*(1-Roundoff)-Underflow:v; }
float Project(uint id,uint view)
{
    uint g=G(id),v=Tables.x+view*ViewWords; float4 lo=asfloat(M4(g+MinimumWords)),hi=asfloat(M4(g+MaximumWords));
    if (lo.w==0) return 0;
    float minimumW=asfloat(InfinityBits),maxX=0,maxY=0;
    const uint BoxCorners=8,MatrixRows=4,SpatialAxes=3;
    for (uint corner=0;corner<BoxCorners;++corner)
    {
        float4 cornerPosition=float4((corner&1)?hi.x:lo.x,(corner&2)?hi.y:lo.y,(corner&4)?hi.z:lo.z,1),clip=0,uncertainty=0;
        [unroll] for (uint row=0;row<MatrixRows;++row)
        {
            float4 coefficients=asfloat(M4(v+row*VectorWords)),errors=asfloat(M4(v+ViewErrors+row*VectorWords));
            clip[row]=dot(coefficients,cornerPosition); uncertainty[row]=Expand(dot(abs(coefficients),abs(cornerPosition))*Roundoff+dot(errors,abs(cornerPosition)));
        }
        float w=Contract(clip.w-uncertainty.w),z=Contract(clip.z-uncertainty.z);
        if (!(w>0) || !(z>0) || !isfinite(w)) return asfloat(InfinityBits);
        minimumW=min(minimumW,w); maxX=max(maxX,Expand(Expand(abs(clip.x)+uncertainty.x)/w)); maxY=max(maxY,Expand(Expand(abs(clip.y)+uncertainty.y)/w));
    }
    float4 delta=0;
    [unroll] for (uint row=0;row<MatrixRows;++row)
    {
        float norm=0;
        for (uint axis=0;axis<SpatialAxes;++axis) norm+=abs(asfloat(M(v+row*VectorWords+axis)))+asfloat(M(v+ViewErrors+row*VectorWords+axis));
        delta[row]=Expand(Expand(norm)*lo.w);
    }
    float denominator=Contract(minimumW-delta.w); if (!(denominator>0)) return asfloat(InfinityBits);
    float x=Expand(Expand(delta.x+Expand(maxX*delta.w))/denominator),y=Expand(Expand(delta.y+Expand(maxY*delta.w))/denominator);
    return Expand(Expand(x*asfloat(M(v+ViewViewport)))+Expand(y*asfloat(M(v+ViewViewport+1))));
}
[numthreads(WaveWidth,1,1)] void CSEvaluate(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint at=Block(group)*WaveWidth+lane; if (at>=Dispatch.y) return;
    uint id=M(Tables.y+Dispatch.x+at),g=G(id); uint4 links=M4(g),range=M4(g+VectorWords);
    float pixels=0; for (uint view=0;view<Counts.y;++view) pixels=max(pixels,Project(id,view));
    for (uint child=0;child<links.y;++child)
    { uint producer=M4(C(M(Offsets.y+links.x+child))+VectorWords).z; if (producer!=Invalid) pixels=max(pixels,asfloat(S(producer*GroupStateWords+ErrorWord))); }
    bool complete=true; for (uint page=0;page<range.y;++page) { bool present=Resident(range.x+page); complete=complete && present; }
    State.Store4(id*GroupStateWords*WordBytes,uint4(asuint(pixels),complete?1:0,0,pixels>Quality.x?1:0));
}
[numthreads(WaveWidth,1,1)] void CSRefine(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint at=Block(group)*WaveWidth+lane; if (at>=Dispatch.y) return;
    uint id=M(Tables.y+Dispatch.x+at); uint4 links=M4(G(id)); bool ancestors=true;
    for (uint parent=0;parent<links.w;++parent)
    { uint consumer=M4(C(M(Offsets.z+links.z+parent))+VectorWords).w; if (consumer!=Invalid && S(consumer*GroupStateWords+RefinedWord)==0) ancestors=false; }
    uint s=id*GroupStateWords; Put(s+RefinedWord,ancestors && S(s+CompleteWord)!=0 && S(s+RequestedWord)!=0?1:0);
}
void Inclusive(uint lane)
{
    for (uint stride=1;stride<WaveWidth;stride*=2)
    { uint3 value=lane>=stride?Scan[lane-stride]:uint3(0,0,0); GroupMemoryBarrierWithGroupSync(); Scan[lane]+=value; GroupMemoryBarrierWithGroupSync(); }
}
[numthreads(WaveWidth,1,1)] void CSClusters(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint block=Block(group),id=block*WaveWidth+lane; bool active=false; uint3 values=0;
    if (block>=Limits.z) return;
    if (id<Tables.w)
    {
        uint4 geometry=M4(C(id)),owners=M4(C(id)+VectorWords);
        active=(owners.w==Invalid || S(owners.w*GroupStateWords+RefinedWord)!=0) && (owners.z==Invalid || S(owners.z*GroupStateWords+RefinedWord)==0);
        if (active) values=uint3(geometry.zw,1);
    }
    Scan[lane]=values; GroupMemoryBarrierWithGroupSync(); Inclusive(lane);
    if (id<Tables.w) State.Store4((ClusterStates()+id*ClusterStateWords)*WordBytes,uint4(active?1:0,Scan[lane]-values));
    if (lane==WaveWidth-1) State.Store4((BlockStates()+block*VectorWords)*WordBytes,uint4(Scan[lane],0));
}
[numthreads(WaveWidth,1,1)] void CSBlocks(uint lane:SV_GroupIndex)
{
    if (lane==0) Carry=0; GroupMemoryBarrierWithGroupSync();
    for (uint first=0;first<Limits.z;first+=WaveWidth)
    {
        uint id=first+lane; uint3 values=id<Limits.z?S4(BlockStates()+id*VectorWords).xyz:uint3(0,0,0);
        Scan[lane]=values; GroupMemoryBarrierWithGroupSync(); Inclusive(lane);
        uint3 prefix=Carry+Scan[lane]-values;
        if (id<Limits.z) State.Store4((BlockStates()+id*VectorWords)*WordBytes,uint4(prefix,0));
        GroupMemoryBarrierWithGroupSync(); if (lane==0) Carry+=Scan[WaveWidth-1]; GroupMemoryBarrierWithGroupSync();
    }
    if (lane==0)
    { if (Carry.x>Capacity.x || Carry.y>Capacity.y) Error(OutputOverflow); Put(Status()+1,Carry.x); Put(Status()+2,Carry.y); Put(Status()+3,Carry.z); }
}
uint ReadBits(uint payload,uint bit,uint width)
{
    if (width==0) return 0;
    uint word=bit/BitsPerWord,shift=bit%BitsPerWord,value=P(payload+word)>>shift;
    if (shift+width>BitsPerWord) value|=P(payload+word+1)<<(BitsPerWord-shift);
    if (width==BitsPerWord) return value;
    return value&((1u<<width)-1);
}
bool ValidHeader(uint page,uint header,uint4 geometry,uint4 owners)
{
    uint used=P(page+PageUsedWord);
    if (P(page)!=PageMagic || P(page+1)!=PageVersion || used>PageWords || used<PageHeaderWords+ClusterHeaderWords ||
        P(page+PageFlagsWord)>1 || P(page+PagePrecisionWord)!=Limits.w || P(page+PageIdWord)!=geometry.x || P(page+PageReservedWord)!=0 ||
        geometry.y>used-ClusterHeaderWords || geometry.z==0 || geometry.z>MaximumCorners || geometry.w==0 || geometry.w>MaximumCorners) return false;
    uint4 counts=Physical.Load4(header*WordBytes),identity=Physical.Load4((header+VectorWords)*WordBytes);
    if (any(counts!=uint4(owners.y,geometry.zw,owners.x)) || identity.x!=owners.z || identity.y!=owners.w || identity.z<ClusterHeaderWords || identity.z>used-geometry.y) return false;
    uint bits=0;
    for (uint channel=0;channel<VertexWords;++channel)
    { uint width=P(header+HeaderWidths+channel); if (width>BitsPerWord || P(header+HeaderOffsets+channel)!=bits) return false; bits+=geometry.z*width; }
    if (identity.w!=bits) return false;
    uint indexWidth=0; for (uint value=geometry.z-1;value!=0;value>>=1) ++indexWidth;
    bits+=geometry.w*indexWidth;
    return identity.z==ClusterHeaderWords+(bits+BitsPerWord-1)/BitsPerWord;
}
[numthreads(WaveWidth,1,1)] void CSDecode(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint id=Block(group); if (id>=Tables.w) return;
    uint4 selected=S4(ClusterStates()+id*ClusterStateWords);
    uint4 geometry=M4(C(id)),owners=M4(C(id)+VectorWords),prefix=S4(BlockStates()+(id/WaveWidth)*VectorWords);
    uint firstVertex=prefix.x+selected.y,firstIndex=prefix.y+selected.z;
    if (lane==0)
    {
        DecodeReady=0;
        if (selected.x!=0)
        {
            if (!Resident(geometry.x)) Error(MissingPage);
            else
            {
                uint page=Mapping[geometry.x].x*PhysicalWords+VectorWords;
                if (!ValidHeader(page,page+geometry.y,geometry,owners)) Error(BadPayload);
                else if (firstVertex>Capacity.x || geometry.z>Capacity.x-firstVertex || firstIndex>Capacity.y || geometry.w>Capacity.y-firstIndex) Error(OutputOverflow);
                else DecodeReady=1;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync(); if (DecodeReady==0) return;
    uint header=Mapping[geometry.x].x*PhysicalWords+VectorWords+geometry.y,payload=header+ClusterHeaderWords;
    float scale=asfloat(uint(asint(Limits.w)+FloatExponentBias)<<FloatMantissaBits);
    for (uint vertex=lane;vertex<geometry.z;vertex+=WaveWidth) for (uint channel=0;channel<VertexWords;++channel)
    {
        uint base=P(header+HeaderBases+channel),width=P(header+HeaderWidths+channel);
        uint value=ReadBits(payload,P(header+HeaderOffsets+channel)+vertex*width,width),decoded=base^value;
        if (channel<PositionChannels)
        { if (value>MaximumSigned-base) Error(BadPayload); decoded=asuint(float(asint(base+value))*scale); }
        if (!isfinite(asfloat(decoded))) Error(BadPayload);
        Vertices.Store(((firstVertex+vertex)*VertexWords+channel)*WordBytes,decoded);
    }
    uint indexWidth=0; for (uint value=geometry.z-1;value!=0;value>>=1) ++indexWidth;
    for (uint index=lane;index<geometry.w;index+=WaveWidth)
    {
        uint local=ReadBits(payload,P(header+HeaderIndexOffset)+index*indexWidth,indexWidth);
        if (local>=geometry.z) Error(BadPayload); else Indices[firstIndex+index]=firstVertex+local;
    }
}
