#include <metal_stdlib>
using namespace metal;
// Independent Metal Nanite group-DAG evaluation and compressed page decoding.
// Encoder boundaries order height dependencies; command completion precedes
// buffer publication. Compile without fast math for interval error bounds.
// Karis et al., "A Deep Dive into Nanite Virtualized Geometry", SIGGRAPH 2021.
constant uint BatchWidth=64,VectorWords=4,GroupStride=16,ClusterStride=16,ViewStride=36,StateStride=4;
constant uint BoundsMinimum=8,BoundsMaximum=12,MatrixError=16,ViewportScale=32;
constant uint ErrorField=0,CompleteField=1,RefinedField=2,RequestField=3;
constant uint PageStride=16384,PhysicalStride=PageStride+VectorWords,PageHeader=8,ClusterHeader=44,VertexStride=12;
constant uint ChannelBases=8,ChannelWidths=20,ChannelOffsets=32,ClusterWordCount=6,IndexOffset=7;
constant uint UsedWord=2,FlagsWord=4,ExponentWord=5,PageIdWord=6,ReservedWord=7;
constant uint WordBits=32,PositionChannels=3,MaximumCorners=768,ExponentBias=127,MantissaBits=23,MaximumSigned=0x7fffffff;
constant uint Invalid=0xffffffff,InfinityBits=0x7f800000,PageMagic=0x47504e45,PageVersion=1;
constant uint StalePage=1,BadPayload=2,OutputOverflow=4,MissingPage=8;
constant float Epsilon=1.1920928955078125e-7,MinimumNormal=1.1754943508222875e-38;
constant float OperationCount=16.0,Roundoff=OperationCount*Epsilon,Underflow=OperationCount*MinimumNormal;
struct NaniteFrame { uint4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; float4 Quality; };
struct Memory
{
    device const uint* Heap; device const uint4* Map; device const uint* Pool;
    device atomic_uint* State; device uint* Vertices; device uint* Indices;
};
uint4 Read4(device const uint* data,uint word) { return uint4(data[word],data[word+1],data[word+2],data[word+3]); }
uint Read(thread const Memory& m,uint word) { return atomic_load_explicit(m.State+word,memory_order_relaxed); }
void Write(thread const Memory& m,uint word,uint value) { atomic_store_explicit(m.State+word,value,memory_order_relaxed); }
uint4 State4(thread const Memory& m,uint word) { return uint4(Read(m,word),Read(m,word+1),Read(m,word+2),Read(m,word+3)); }
void Write4(thread const Memory& m,uint word,uint4 value) { Write(m,word,value.x); Write(m,word+1,value.y); Write(m,word+2,value.z); Write(m,word+3,value.w); }
uint Group(constant NaniteFrame& f,uint id) { return f.Offsets.x+id*GroupStride; }
uint Cluster(constant NaniteFrame& f,uint id) { return f.Offsets.w+id*ClusterStride; }
uint ClusterState(constant NaniteFrame& f) { return f.Counts.x*StateStride; }
uint BlockState(constant NaniteFrame& f) { return ClusterState(f)+f.Tables.w*StateStride; }
uint Summary(constant NaniteFrame& f) { return BlockState(f)+f.Limits.z*VectorWords; }
uint Batch(constant NaniteFrame& f,uint3 group) { return group.y*f.Dispatch.z+group.x; }
void Report(thread const Memory& m,constant NaniteFrame& f,uint error) { atomic_fetch_or_explicit(m.State+Summary(f),error,memory_order_relaxed); }
bool Resident(thread const Memory& m,constant NaniteFrame& f,uint page)
{
    uint4 entry=m.Map[page]; if (entry.x==Invalid) return false;
    if (entry.x>=f.Counts.w || entry.y==0 || any(entry.zw!=f.Identity.xy)) { Report(m,f,StalePage); return false; }
    if (any(Read4(m.Pool,entry.x*PhysicalStride)!=uint4(page,entry.yzw))) { Report(m,f,StalePage); return false; }
    return true;
}
float Expand(float x) { return isfinite(x) && x>=0?x*(1+Roundoff)+Underflow:as_type<float>(InfinityBits); }
float Contract(float x) { return x>0?x*(1-Roundoff)-Underflow:x; }
float Project(thread const Memory& m,constant NaniteFrame& f,uint id,uint view)
{
    uint g=Group(f,id),v=f.Tables.x+view*ViewStride;
    float4 low=as_type<float4>(Read4(m.Heap,g+BoundsMinimum)),high=as_type<float4>(Read4(m.Heap,g+BoundsMaximum)); if (low.w==0) return 0;
    float minimumW=as_type<float>(InfinityBits); float2 ndc=0;
    constexpr uint BoxCorners=8,Rows=4;
    for (uint corner=0;corner<BoxCorners;++corner)
    {
        float4 p=float4((corner&1)?high.x:low.x,(corner&2)?high.y:low.y,(corner&4)?high.z:low.z,1),clip=0,uncertainty=0;
        for (uint row=0;row<Rows;++row)
        {
            float4 coefficients=as_type<float4>(Read4(m.Heap,v+row*VectorWords)),errors=as_type<float4>(Read4(m.Heap,v+MatrixError+row*VectorWords));
            clip[row]=dot(coefficients,p); uncertainty[row]=Expand(dot(abs(coefficients),abs(p))*Roundoff+dot(errors,abs(p)));
        }
        float w=Contract(clip.w-uncertainty.w),z=Contract(clip.z-uncertainty.z);
        if (!(w>0) || !(z>0) || !isfinite(w)) return as_type<float>(InfinityBits);
        minimumW=min(minimumW,w); ndc=max(ndc,float2(Expand(Expand(abs(clip.x)+uncertainty.x)/w),Expand(Expand(abs(clip.y)+uncertainty.y)/w)));
    }
    float4 movement=0;
    for (uint row=0;row<Rows;++row)
    {
        float norm=0;
        for (uint axis=0;axis<PositionChannels;++axis) norm+=abs(as_type<float>(m.Heap[v+row*VectorWords+axis]))+as_type<float>(m.Heap[v+MatrixError+row*VectorWords+axis]);
        movement[row]=Expand(Expand(norm)*low.w);
    }
    float denominator=Contract(minimumW-movement.w); if (!(denominator>0)) return as_type<float>(InfinityBits);
    float x=Expand(Expand(movement.x+Expand(ndc.x*movement.w))/denominator),y=Expand(Expand(movement.y+Expand(ndc.y*movement.w))/denominator);
    return Expand(Expand(x*as_type<float>(m.Heap[v+ViewportScale]))+Expand(y*as_type<float>(m.Heap[v+ViewportScale+1])));
}
kernel void EvansNaniteEvaluate(constant NaniteFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device const uint4* map [[buffer(2)]],
    device const uint* pool [[buffer(3)]],device atomic_uint* state [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint at=Batch(f,group)*BatchWidth+lane; if (at>=f.Dispatch.y) return;
    Memory m{heap,map,pool,state,nullptr,nullptr}; uint id=heap[f.Tables.y+f.Dispatch.x+at],g=Group(f,id);
    uint4 links=Read4(heap,g),pages=Read4(heap,g+VectorWords); float pixels=0;
    for (uint view=0;view<f.Counts.y;++view) pixels=max(pixels,Project(m,f,id,view));
    for (uint child=0;child<links.y;++child)
    { uint producer=Read4(heap,Cluster(f,heap[f.Offsets.y+links.x+child])+VectorWords).z; if (producer!=Invalid) pixels=max(pixels,as_type<float>(Read(m,producer*StateStride+ErrorField))); }
    bool complete=true; for (uint page=0;page<pages.y;++page) { bool available=Resident(m,f,pages.x+page); complete=complete && available; }
    Write4(m,id*StateStride,uint4(as_type<uint>(pixels),complete?1u:0u,0,pixels>f.Quality.x?1u:0u));
}
kernel void EvansNaniteRefine(constant NaniteFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device atomic_uint* state [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint at=Batch(f,group)*BatchWidth+lane; if (at>=f.Dispatch.y) return;
    Memory m{heap,nullptr,nullptr,state,nullptr,nullptr}; uint id=heap[f.Tables.y+f.Dispatch.x+at]; uint4 links=Read4(heap,Group(f,id)); bool ancestors=true;
    for (uint parent=0;parent<links.w;++parent)
    { uint consumer=Read4(heap,Cluster(f,heap[f.Offsets.z+links.z+parent])+VectorWords).w; if (consumer!=Invalid && Read(m,consumer*StateStride+RefinedField)==0) ancestors=false; }
    uint s=id*StateStride; Write(m,s+RefinedField,ancestors && Read(m,s+CompleteField)!=0 && Read(m,s+RequestField)!=0?1u:0u);
}
uint3 Scan(uint3 value,uint lane,threadgroup uint3* prefix)
{
    prefix[lane]=value; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=1;stride<BatchWidth;stride*=2)
    { uint3 add=lane>=stride?prefix[lane-stride]:uint3(0); threadgroup_barrier(mem_flags::mem_threadgroup); prefix[lane]+=add; threadgroup_barrier(mem_flags::mem_threadgroup); }
    return prefix[lane];
}
kernel void EvansNaniteClusters(constant NaniteFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device atomic_uint* state [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint block=Batch(f,group); if (block>=f.Limits.z) return;
    Memory m{heap,nullptr,nullptr,state,nullptr,nullptr}; uint id=block*BatchWidth+lane; bool selected=false; uint3 amount=0; threadgroup uint3 prefix[BatchWidth];
    if (id<f.Tables.w)
    {
        uint4 geometry=Read4(heap,Cluster(f,id)),owners=Read4(heap,Cluster(f,id)+VectorWords);
        selected=(owners.w==Invalid || Read(m,owners.w*StateStride+RefinedField)!=0) && (owners.z==Invalid || Read(m,owners.z*StateStride+RefinedField)==0);
        if (selected) amount=uint3(geometry.zw,1);
    }
    uint3 inclusive=Scan(amount,lane,prefix);
    if (id<f.Tables.w) Write4(m,ClusterState(f)+id*StateStride,uint4(selected?1u:0u,inclusive-amount));
    if (lane==BatchWidth-1) Write4(m,BlockState(f)+block*VectorWords,uint4(inclusive,0));
}
kernel void EvansNaniteBlocks(constant NaniteFrame& f [[buffer(0)]],device atomic_uint* state [[buffer(4)]],uint lane [[thread_index_in_threadgroup]])
{
    Memory m{nullptr,nullptr,nullptr,state,nullptr,nullptr}; threadgroup uint3 prefix[BatchWidth]; threadgroup uint3 carry;
    if (lane==0) carry=0; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint first=0;first<f.Limits.z;first+=BatchWidth)
    {
        uint id=first+lane; uint3 amount=id<f.Limits.z?State4(m,BlockState(f)+id*VectorWords).xyz:uint3(0);
        uint3 inclusive=Scan(amount,lane,prefix),exclusive=carry+inclusive-amount;
        if (id<f.Limits.z) Write4(m,BlockState(f)+id*VectorWords,uint4(exclusive,0));
        threadgroup_barrier(mem_flags::mem_threadgroup); if (lane==BatchWidth-1) carry+=inclusive; threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane==0)
    { if (carry.x>f.Capacity.x || carry.y>f.Capacity.y) Report(m,f,OutputOverflow); Write(m,Summary(f)+1,carry.x); Write(m,Summary(f)+2,carry.y); Write(m,Summary(f)+3,carry.z); }
}
uint Extract(device const uint* pool,uint payload,uint bit,uint width)
{
    if (width==0) return 0; uint word=bit/WordBits,shift=bit%WordBits,value=pool[payload+word]>>shift;
    if (shift+width>WordBits) value|=pool[payload+word+1]<<(WordBits-shift);
    if (width==WordBits) return value; return value&((1u<<width)-1);
}
bool CheckHeader(thread const Memory& m,constant NaniteFrame& f,uint page,uint header,uint4 geometry,uint4 owners)
{
    device const uint* p=m.Pool; uint used=p[page+UsedWord];
    if (p[page]!=PageMagic || p[page+1]!=PageVersion || used>PageStride || used<PageHeader+ClusterHeader ||
        p[page+FlagsWord]>1 || p[page+ExponentWord]!=f.Limits.w || p[page+PageIdWord]!=geometry.x || p[page+ReservedWord]!=0 ||
        geometry.y>used-ClusterHeader || geometry.z==0 || geometry.z>MaximumCorners || geometry.w==0 || geometry.w>MaximumCorners) return false;
    if (any(Read4(p,header)!=uint4(owners.y,geometry.zw,owners.x)) || p[header+VectorWords]!=owners.z || p[header+VectorWords+1]!=owners.w ||
        p[header+ClusterWordCount]<ClusterHeader || p[header+ClusterWordCount]>used-geometry.y) return false;
    uint bits=0;
    for (uint channel=0;channel<VertexStride;++channel)
    { uint width=p[header+ChannelWidths+channel]; if (width>WordBits || p[header+ChannelOffsets+channel]!=bits) return false; bits+=geometry.z*width; }
    if (p[header+IndexOffset]!=bits) return false;
    uint indexWidth=0; for (uint value=geometry.z-1;value!=0;value>>=1) ++indexWidth;
    bits+=geometry.w*indexWidth; return p[header+ClusterWordCount]==ClusterHeader+(bits+WordBits-1)/WordBits;
}
kernel void EvansNaniteDecode(constant NaniteFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device const uint4* map [[buffer(2)]],
    device const uint* pool [[buffer(3)]],device atomic_uint* state [[buffer(4)]],device uint* vertices [[buffer(5)]],device uint* indices [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    uint id=Batch(f,group); if (id>=f.Tables.w) return; Memory m{heap,map,pool,state,vertices,indices};
    uint4 selected=State4(m,ClusterState(f)+id*StateStride); if (selected.x==0) return;
    uint4 geometry=Read4(heap,Cluster(f,id)),owners=Read4(heap,Cluster(f,id)+VectorWords),prefix=State4(m,BlockState(f)+(id/BatchWidth)*VectorWords);
    uint firstVertex=prefix.x+selected.y,firstIndex=prefix.y+selected.z; threadgroup uint allowed;
    if (lane==0)
    {
        allowed=0;
        if (!Resident(m,f,geometry.x)) Report(m,f,MissingPage);
        else
        {
            uint page=map[geometry.x].x*PhysicalStride+VectorWords;
            if (!CheckHeader(m,f,page,page+geometry.y,geometry,owners)) Report(m,f,BadPayload);
            else if (firstVertex>f.Capacity.x || geometry.z>f.Capacity.x-firstVertex || firstIndex>f.Capacity.y || geometry.w>f.Capacity.y-firstIndex) Report(m,f,OutputOverflow);
            else allowed=1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); if (allowed==0) return;
    uint header=map[geometry.x].x*PhysicalStride+VectorWords+geometry.y,payload=header+ClusterHeader;
    float scale=as_type<float>(uint(as_type<int>(f.Limits.w)+int(ExponentBias))<<MantissaBits);
    for (uint vertexSlot=lane;vertexSlot<geometry.z;vertexSlot+=BatchWidth) for (uint channel=0;channel<VertexStride;++channel)
    {
        uint base=pool[header+ChannelBases+channel],width=pool[header+ChannelWidths+channel];
        uint delta=Extract(pool,payload,pool[header+ChannelOffsets+channel]+vertexSlot*width,width),decoded=base^delta;
        if (channel<PositionChannels) { if (delta>MaximumSigned-base) Report(m,f,BadPayload); decoded=as_type<uint>(float(as_type<int>(base+delta))*scale); }
        if (!isfinite(as_type<float>(decoded))) Report(m,f,BadPayload); vertices[(firstVertex+vertexSlot)*VertexStride+channel]=decoded;
    }
    uint indexWidth=0; for (uint value=geometry.z-1;value!=0;value>>=1) ++indexWidth;
    for (uint index=lane;index<geometry.w;index+=BatchWidth)
    { uint local=Extract(pool,payload,pool[header+IndexOffset]+index*indexWidth,indexWidth); if (local>=geometry.z) Report(m,f,BadPayload); else indices[firstIndex+index]=firstVertex+local; }
}
