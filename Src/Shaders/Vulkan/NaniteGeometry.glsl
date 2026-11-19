#version 450
// Native Vulkan Nanite DAG selection and random-access page decoding.
// GLSL is compiled at runtime. Compute barriers separate level dependencies;
// vertex/index and host barriers precede publication of the decoded buffers.
// Karis et al., SIGGRAPH 2021, "A Deep Dive into Nanite Virtualized Geometry".
#define NANITE_WIDTH 64
#define NANITE_EVALUATE 0
#define NANITE_REFINE 1
#define NANITE_CLUSTERS 2
#define NANITE_BLOCKS 3
#define NANITE_DECODE 4
layout(local_size_x=NANITE_WIDTH) in;
layout(std140,set=0,binding=0) uniform Parameters { uvec4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; vec4 Quality; } Frame;
layout(std430,set=0,binding=1) readonly buffer MetadataBuffer { uint Words[]; } Metadata;
layout(std430,set=0,binding=2) readonly buffer PageTableBuffer { uvec4 Entries[]; } Mapping;
layout(std430,set=0,binding=3) readonly buffer PhysicalBuffer { uint Words[]; } Physical;
layout(std430,set=0,binding=4) buffer StateBuffer { uint Words[]; } State;
layout(std430,set=0,binding=5) writeonly buffer VertexBuffer { uint Words[]; } Vertices;
layout(std430,set=0,binding=6) writeonly buffer IndexBuffer { uint Values[]; } Indices;
const uint Width=64u,VectorWords=4u,GroupWords=16u,ClusterWords=16u,ViewWords=36u,StateWords=4u;
const uint MinimumWord=8u,MaximumWord=12u,ViewErrorWord=16u,ViewportWord=32u;
const uint ErrorWord=0u,CompleteWord=1u,RefinedWord=2u,RequestedWord=3u;
const uint PageWords=16384u,PhysicalWords=PageWords+VectorWords,PageHeader=8u,ClusterHeader=44u,VertexWords=12u;
const uint BasesWord=8u,WidthsWord=20u,OffsetsWord=32u,PayloadSizeWord=6u,IndexOffsetWord=7u;
const uint UsedWord=2u,FlagsWord=4u,PrecisionWord=5u,PageIdWord=6u,ReservedWord=7u;
const uint WordBits=32u,PositionAxes=3u,MaximumCorners=768u,ExponentBias=127u,MantissaBits=23u,MaximumSigned=0x7fffffffu;
const uint Invalid=0xffffffffu,InfinityBits=0x7f800000u,Magic=0x47504e45u,Version=1u;
const uint StalePage=1u,BadPayload=2u,OutputOverflow=4u,MissingPage=8u;
const float Epsilon=1.1920928955078125e-7,MinimumNormal=1.1754943508222875e-38;
const float OperationCount=16.0,Roundoff=OperationCount*Epsilon,Underflow=OperationCount*MinimumNormal;
shared uvec3 Scan[NANITE_WIDTH]; shared uvec3 Carry; shared uint DecodeReady;
uvec4 Load4(uint word) { return uvec4(Metadata.Words[word],Metadata.Words[word+1u],Metadata.Words[word+2u],Metadata.Words[word+3u]); }
uvec4 State4(uint word) { return uvec4(State.Words[word],State.Words[word+1u],State.Words[word+2u],State.Words[word+3u]); }
void Store4(uint word,uvec4 value) { State.Words[word]=value.x; State.Words[word+1u]=value.y; State.Words[word+2u]=value.z; State.Words[word+3u]=value.w; }
uint GroupRecord(uint id) { return Frame.Offsets.x+id*GroupWords; }
uint ClusterRecord(uint id) { return Frame.Offsets.w+id*ClusterWords; }
uint ClusterState() { return Frame.Counts.x*StateWords; }
uint BlockState() { return ClusterState()+Frame.Tables.w*StateWords; }
uint Status() { return BlockState()+Frame.Limits.z*VectorWords; }
uint Block() { return gl_WorkGroupID.y*Frame.Dispatch.z+gl_WorkGroupID.x; }
void Report(uint mask) { atomicOr(State.Words[Status()],mask); }
bool Finite(float value) { return !isnan(value) && !isinf(value); }
bool Resident(uint page)
{
    uvec4 entry=Mapping.Entries[page]; if (entry.x==Invalid) return false;
    if (entry.x>=Frame.Counts.w || entry.y==0u || any(notEqual(entry.zw,Frame.Identity.xy))) { Report(StalePage); return false; }
    uint word=entry.x*PhysicalWords;
    uvec4 stored=uvec4(Physical.Words[word],Physical.Words[word+1u],Physical.Words[word+2u],Physical.Words[word+3u]);
    if (any(notEqual(stored,uvec4(page,entry.yzw)))) { Report(StalePage); return false; } return true;
}
float Expand(float value) { return Finite(value) && value>=0.0?value*(1.0+Roundoff)+Underflow:uintBitsToFloat(InfinityBits); }
float Contract(float value) { return value>0.0?value*(1.0-Roundoff)-Underflow:value; }
float Project(uint id,uint view)
{
    uint g=GroupRecord(id),v=Frame.Tables.x+view*ViewWords;
    vec4 low=uintBitsToFloat(Load4(g+MinimumWord)),high=uintBitsToFloat(Load4(g+MaximumWord)); if (low.w==0.0) return 0.0;
    float minimumW=uintBitsToFloat(InfinityBits); vec2 ndc=vec2(0.0);
    const uint BoxCorners=8u,Rows=4u;
    for (uint corner=0u;corner<BoxCorners;++corner)
    {
        vec4 point=vec4((corner&1u)!=0u?high.x:low.x,(corner&2u)!=0u?high.y:low.y,(corner&4u)!=0u?high.z:low.z,1.0),clip=vec4(0.0),error=vec4(0.0);
        for (uint row=0u;row<Rows;++row)
        {
            vec4 coefficients=uintBitsToFloat(Load4(v+row*VectorWords)),uncertainty=uintBitsToFloat(Load4(v+ViewErrorWord+row*VectorWords));
            clip[row]=dot(coefficients,point); error[row]=Expand(dot(abs(coefficients),abs(point))*Roundoff+dot(uncertainty,abs(point)));
        }
        float w=Contract(clip.w-error.w),z=Contract(clip.z-error.z);
        if (!(w>0.0) || !(z>0.0) || !Finite(w)) return uintBitsToFloat(InfinityBits);
        minimumW=min(minimumW,w); ndc=max(ndc,vec2(Expand(Expand(abs(clip.x)+error.x)/w),Expand(Expand(abs(clip.y)+error.y)/w)));
    }
    vec4 delta=vec4(0.0);
    for (uint row=0u;row<Rows;++row)
    {
        float norm=0.0;
        for (uint axis=0u;axis<PositionAxes;++axis) norm+=abs(uintBitsToFloat(Metadata.Words[v+row*VectorWords+axis]))+uintBitsToFloat(Metadata.Words[v+ViewErrorWord+row*VectorWords+axis]);
        delta[row]=Expand(Expand(norm)*low.w);
    }
    float denominator=Contract(minimumW-delta.w); if (!(denominator>0.0)) return uintBitsToFloat(InfinityBits);
    float x=Expand(Expand(delta.x+Expand(ndc.x*delta.w))/denominator),y=Expand(Expand(delta.y+Expand(ndc.y*delta.w))/denominator);
    return Expand(Expand(x*uintBitsToFloat(Metadata.Words[v+ViewportWord]))+Expand(y*uintBitsToFloat(Metadata.Words[v+ViewportWord+1u])));
}
void Evaluate()
{
    uint at=Block()*Width+gl_LocalInvocationIndex; if (at>=Frame.Dispatch.y) return;
    uint id=Metadata.Words[Frame.Tables.y+Frame.Dispatch.x+at],g=GroupRecord(id); uvec4 links=Load4(g),pages=Load4(g+VectorWords);
    float pixels=0.0; for (uint view=0u;view<Frame.Counts.y;++view) pixels=max(pixels,Project(id,view));
    for (uint child=0u;child<links.y;++child)
    { uint producer=Load4(ClusterRecord(Metadata.Words[Frame.Offsets.y+links.x+child])+VectorWords).z; if (producer!=Invalid) pixels=max(pixels,uintBitsToFloat(State.Words[producer*StateWords+ErrorWord])); }
    bool complete=true; for (uint page=0u;page<pages.y;++page) { bool present=Resident(pages.x+page); complete=complete && present; }
    Store4(id*StateWords,uvec4(floatBitsToUint(pixels),complete?1u:0u,0u,pixels>Frame.Quality.x?1u:0u));
}
void Refine()
{
    uint at=Block()*Width+gl_LocalInvocationIndex; if (at>=Frame.Dispatch.y) return;
    uint id=Metadata.Words[Frame.Tables.y+Frame.Dispatch.x+at]; uvec4 links=Load4(GroupRecord(id)); bool ancestors=true;
    for (uint parent=0u;parent<links.w;++parent)
    { uint consumer=Load4(ClusterRecord(Metadata.Words[Frame.Offsets.z+links.z+parent])+VectorWords).w; if (consumer!=Invalid && State.Words[consumer*StateWords+RefinedWord]==0u) ancestors=false; }
    uint s=id*StateWords; State.Words[s+RefinedWord]=ancestors && State.Words[s+CompleteWord]!=0u && State.Words[s+RequestedWord]!=0u?1u:0u;
}
void Inclusive(uint lane)
{
    for (uint stride=1u;stride<Width;stride*=2u)
    { uvec3 add=lane>=stride?Scan[lane-stride]:uvec3(0u); barrier(); Scan[lane]+=add; barrier(); }
}
void Clusters()
{
    uint block=Block(); if (block>=Frame.Limits.z) return;
    uint lane=gl_LocalInvocationIndex,id=block*Width+lane; bool selected=false; uvec3 values=uvec3(0u);
    if (id<Frame.Tables.w)
    {
        uvec4 geometry=Load4(ClusterRecord(id)),owners=Load4(ClusterRecord(id)+VectorWords);
        selected=(owners.w==Invalid || State.Words[owners.w*StateWords+RefinedWord]!=0u) && (owners.z==Invalid || State.Words[owners.z*StateWords+RefinedWord]==0u);
        if (selected) values=uvec3(geometry.zw,1u);
    }
    Scan[lane]=values; barrier(); Inclusive(lane);
    if (id<Frame.Tables.w) Store4(ClusterState()+id*StateWords,uvec4(selected?1u:0u,Scan[lane]-values));
    if (lane==Width-1u) Store4(BlockState()+block*VectorWords,uvec4(Scan[lane],0u));
}
void Blocks()
{
    uint lane=gl_LocalInvocationIndex; if (lane==0u) Carry=uvec3(0u); barrier();
    for (uint first=0u;first<Frame.Limits.z;first+=Width)
    {
        uint id=first+lane; uvec3 values=id<Frame.Limits.z?State4(BlockState()+id*VectorWords).xyz:uvec3(0u);
        Scan[lane]=values; barrier(); Inclusive(lane); uvec3 prefix=Carry+Scan[lane]-values;
        if (id<Frame.Limits.z) Store4(BlockState()+id*VectorWords,uvec4(prefix,0u));
        barrier(); if (lane==0u) Carry+=Scan[Width-1u]; barrier();
    }
    if (lane==0u)
    { if (Carry.x>Frame.Capacity.x || Carry.y>Frame.Capacity.y) Report(OutputOverflow); State.Words[Status()+1u]=Carry.x; State.Words[Status()+2u]=Carry.y; State.Words[Status()+3u]=Carry.z; }
}
uint ReadBits(uint payload,uint bit,uint width)
{
    if (width==0u) return 0u; uint word=bit/WordBits,shift=bit%WordBits,value=Physical.Words[payload+word]>>shift;
    if (shift+width>WordBits) value|=Physical.Words[payload+word+1u]<<(WordBits-shift);
    if (width==WordBits) return value; return value&((1u<<width)-1u);
}
bool HeaderValid(uint page,uint header,uvec4 geometry,uvec4 owners)
{
    uint used=Physical.Words[page+UsedWord];
    if (Physical.Words[page]!=Magic || Physical.Words[page+1u]!=Version || used>PageWords || used<PageHeader+ClusterHeader ||
        Physical.Words[page+FlagsWord]>1u || Physical.Words[page+PrecisionWord]!=Frame.Limits.w || Physical.Words[page+PageIdWord]!=geometry.x || Physical.Words[page+ReservedWord]!=0u ||
        geometry.y>used-ClusterHeader || geometry.z==0u || geometry.z>MaximumCorners || geometry.w==0u || geometry.w>MaximumCorners) return false;
    if (Physical.Words[header]!=owners.y || Physical.Words[header+1u]!=geometry.z || Physical.Words[header+2u]!=geometry.w || Physical.Words[header+3u]!=owners.x ||
        Physical.Words[header+VectorWords]!=owners.z || Physical.Words[header+VectorWords+1u]!=owners.w ||
        Physical.Words[header+PayloadSizeWord]<ClusterHeader || Physical.Words[header+PayloadSizeWord]>used-geometry.y) return false;
    uint bits=0u;
    for (uint channel=0u;channel<VertexWords;++channel)
    { uint width=Physical.Words[header+WidthsWord+channel]; if (width>WordBits || Physical.Words[header+OffsetsWord+channel]!=bits) return false; bits+=geometry.z*width; }
    if (Physical.Words[header+IndexOffsetWord]!=bits) return false;
    uint indexWidth=0u; for (uint value=geometry.z-1u;value!=0u;value>>=1u) ++indexWidth;
    bits+=geometry.w*indexWidth; return Physical.Words[header+PayloadSizeWord]==ClusterHeader+(bits+WordBits-1u)/WordBits;
}
void Decode()
{
    uint id=Block(),lane=gl_LocalInvocationIndex; if (id>=Frame.Tables.w) return;
    uvec4 selected=State4(ClusterState()+id*StateWords); if (selected.x==0u) return;
    uvec4 geometry=Load4(ClusterRecord(id)),owners=Load4(ClusterRecord(id)+VectorWords),prefix=State4(BlockState()+(id/Width)*VectorWords);
    uint firstVertex=prefix.x+selected.y,firstIndex=prefix.y+selected.z;
    if (lane==0u)
    {
        DecodeReady=0u;
        if (!Resident(geometry.x)) Report(MissingPage);
        else
        {
            uint page=Mapping.Entries[geometry.x].x*PhysicalWords+VectorWords;
            if (!HeaderValid(page,page+geometry.y,geometry,owners)) Report(BadPayload);
            else if (firstVertex>Frame.Capacity.x || geometry.z>Frame.Capacity.x-firstVertex || firstIndex>Frame.Capacity.y || geometry.w>Frame.Capacity.y-firstIndex) Report(OutputOverflow);
            else DecodeReady=1u;
        }
    }
    barrier(); if (DecodeReady==0u) return;
    uint header=Mapping.Entries[geometry.x].x*PhysicalWords+VectorWords+geometry.y,payload=header+ClusterHeader;
    float scale=uintBitsToFloat(uint(int(Frame.Limits.w)+int(ExponentBias))<<MantissaBits);
    for (uint vertex=lane;vertex<geometry.z;vertex+=Width) for (uint channel=0u;channel<VertexWords;++channel)
    {
        uint base=Physical.Words[header+BasesWord+channel],width=Physical.Words[header+WidthsWord+channel];
        uint value=ReadBits(payload,Physical.Words[header+OffsetsWord+channel]+vertex*width,width),decoded=base^value;
        if (channel<PositionAxes) { if (value>MaximumSigned-base) Report(BadPayload); decoded=floatBitsToUint(float(int(base+value))*scale); }
        if (!Finite(uintBitsToFloat(decoded))) Report(BadPayload); Vertices.Words[(firstVertex+vertex)*VertexWords+channel]=decoded;
    }
    uint indexWidth=0u; for (uint value=geometry.z-1u;value!=0u;value>>=1u) ++indexWidth;
    for (uint index=lane;index<geometry.w;index+=Width)
    { uint local=ReadBits(payload,Physical.Words[header+IndexOffsetWord]+index*indexWidth,indexWidth); if (local>=geometry.z) Report(BadPayload); else Indices.Values[firstIndex+index]=firstVertex+local; }
}
void main()
{
#if EVANS_NANITE_GEOMETRY_PASS == NANITE_EVALUATE
    Evaluate();
#elif EVANS_NANITE_GEOMETRY_PASS == NANITE_REFINE
    Refine();
#elif EVANS_NANITE_GEOMETRY_PASS == NANITE_CLUSTERS
    Clusters();
#elif EVANS_NANITE_GEOMETRY_PASS == NANITE_BLOCKS
    Blocks();
#elif EVANS_NANITE_GEOMETRY_PASS == NANITE_DECODE
    Decode();
#endif
}
