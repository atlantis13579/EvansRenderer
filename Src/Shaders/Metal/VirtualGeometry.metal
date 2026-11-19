#include <metal_stdlib>
using namespace metal;
// Independent Metal virtual geometry. Hoppe, SIGGRAPH 1997:
// https://hhoppe.com/proj/vdrpm/. The region hierarchy and lossless page protocol
// belong to this project. Disable fast math for interval/NaN conservatism.
// Separate encoders order Evaluate, height propagation, selection, prefix sums
// and decode; publish a candidate only after GPU completion and status checks.
constant uint BatchWidth=64,NodeStride=20,ViewStride=36,DirectoryStride=8,NodeStateStride=8,ClusterStateStride=4,VectorWords=4;
constant uint FirstCluster=4,ClusterCount=5,FirstPage=6,PageCount=7,PreviousSelection=10,Minimum=12,Maximum=16,ViewErrors=16,Viewport=32;
constant uint ErrorWord=0,OwnWord=1,AvailableWord=2,HistoryWord=3,SelectedWord=4,TransitionWord=5,PriorityWord=6;
constant uint VertexPrefix=1,IndexPrefix=2,ActivePrefix=3;
constant uint PageStride=2048,ClusterHeader=16,VertexStride=12,IdWord=5,GenerationWord=6,EpochWord=7,UsedWord=2,MaterialWord=4,DirectoryIdWord=5;
constant uint Invalid=0xffffffff,InfinityBits=0x7f800000,MaximumFloatBits=0x7f7fffff,Magic=0x50475645,Version=1,PreviousFlag=1;
constant uint MissingCoverage=1,StalePage=2,BadPayload=4,OutputOverflow=8;
constant float Epsilon=1.1920928955078125e-7,MinimumNormal=1.1754943508222875e-38,ArithmeticOperations=16.0,Roundoff=ArithmeticOperations*Epsilon,Underflow=ArithmeticOperations*MinimumNormal;
struct VirtualFrame { uint4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; float4 Quality; };
struct Memory
{
    device const uint* Heap; device const uint4* Map; device const uint* Pool;
    device atomic_uint* State; device uint* Vertices; device uint* Indices;
};
uint4 Read4(device const uint* p,uint word) { return uint4(p[word],p[word+1],p[word+2],p[word+3]); }
uint Read(thread const Memory& m,uint word) { return atomic_load_explicit(m.State+word,memory_order_relaxed); }
void Write(thread const Memory& m,uint word,uint value) { atomic_store_explicit(m.State+word,value,memory_order_relaxed); }
uint Node(constant VirtualFrame& f,uint id) { return f.Offsets.x+id*NodeStride; }
uint Directory(constant VirtualFrame& f,uint id) { return f.Tables.x+id*DirectoryStride; }
uint ClusterState(constant VirtualFrame& f) { return f.Counts.x*NodeStateStride; }
uint Blocks(constant VirtualFrame& f) { return (f.Limits.x+BatchWidth-1)/BatchWidth; }
uint BlockState(constant VirtualFrame& f) { return ClusterState(f)+f.Limits.x*ClusterStateStride; }
uint Status(constant VirtualFrame& f) { return BlockState(f)+Blocks(f)*VectorWords; }
uint Group(constant VirtualFrame& f,uint3 coordinate) { return coordinate.y*f.Dispatch.z+coordinate.x; }
void Reject(thread const Memory& m,constant VirtualFrame& f,uint error) { atomic_fetch_or_explicit(m.State+Status(f),error,memory_order_relaxed); }
float Expand(float x) { return isfinite(x) && x>=0?x*(1+Roundoff)+Underflow:as_type<float>(InfinityBits); }
float Contract(float x) { return x>0?x*(1-Roundoff)-Underflow:x; }
bool Resident(thread const Memory& m,constant VirtualFrame& f,uint page)
{
    uint4 map=m.Map[page]; if (map.x==Invalid) return false;
    if (map.x>=f.Counts.w || map.y==0 || any(map.zw!=f.Identity.xy)) { Reject(m,f,StalePage); return false; }
    uint p=map.x*PageStride;
    if (m.Pool[p+IdWord]!=page || m.Pool[p+GenerationWord]!=map.y || m.Pool[p+EpochWord]!=map.z) { Reject(m,f,StalePage); return false; } return true;
}
float PixelError(thread const Memory& m,constant VirtualFrame& f,uint id,uint view)
{
    float4 lo=as_type<float4>(Read4(m.Heap,Node(f,id)+Minimum)),hi=as_type<float4>(Read4(m.Heap,Node(f,id)+Maximum)); if (lo.w==0) return 0;
    uint v=f.Tables.y+view*ViewStride; float minimumW=as_type<float>(InfinityBits),maximumX=0,maximumY=0;
    for (uint corner=0;corner<8;++corner)
    {
        float4 p=float4((corner&1)?hi.x:lo.x,(corner&2)?hi.y:lo.y,(corner&4)?hi.z:lo.z,1),clip=0,error=0;
        for (uint row=0;row<4;++row)
        { float4 a=as_type<float4>(Read4(m.Heap,v+row*VectorWords)),e=as_type<float4>(Read4(m.Heap,v+ViewErrors+row*VectorWords)); clip[row]=dot(a,p); error[row]=Expand(dot(abs(a),abs(p))*Roundoff+dot(e,abs(p))); }
        float w=Contract(clip.w-error.w),z=Contract(clip.z-error.z); if (!(w>0) || !(z>0) || !isfinite(w)) return as_type<float>(InfinityBits);
        minimumW=min(minimumW,w); maximumX=max(maximumX,Expand(Expand(abs(clip.x)+error.x)/w)); maximumY=max(maximumY,Expand(Expand(abs(clip.y)+error.y)/w));
    }
    float4 delta=0;
    for (uint row=0;row<4;++row) { float3 norm=abs(as_type<float4>(Read4(m.Heap,v+row*VectorWords)).xyz)+as_type<float4>(Read4(m.Heap,v+ViewErrors+row*VectorWords)).xyz; delta[row]=Expand(Expand(norm.x+norm.y+norm.z)*lo.w); }
    float denominator=Contract(minimumW-delta.w); if (!(denominator>0)) return as_type<float>(InfinityBits);
    float x=Expand(Expand(delta.x+Expand(maximumX*delta.w))/denominator),y=Expand(Expand(delta.y+Expand(maximumY*delta.w))/denominator);
    float2 viewport=as_type<float4>(Read4(m.Heap,v+Viewport)).xy; return Expand(Expand(x*viewport.x)+Expand(y*viewport.y));
}
bool Children(thread const Memory& m,constant VirtualFrame& f,uint id)
{
    uint4 n=Read4(m.Heap,Node(f,id)); if (n.z==0) return false;
    for (uint at=0;at<n.z;++at) if (Read(m,m.Heap[f.Offsets.y+n.y+at]*NodeStateStride+AvailableWord)==0) return false; return true;
}
bool Fine(thread const Memory& m,constant VirtualFrame& f,uint id)
{ uint s=id*NodeStateStride; return Read4(m.Heap,Node(f,id)).z!=0 && as_type<float>(Read(m,s+ErrorWord))>(Read(m,s+HistoryWord)!=0?f.Quality.y:f.Quality.x); }
bool ChoosesSelf(thread const Memory& m,constant VirtualFrame& f,uint id)
{ return Read(m,id*NodeStateStride+OwnWord)!=0 && (!Fine(m,f,id) || !Children(m,f,id)); }
uint3 Scan(uint3 value,uint lane,threadgroup uint3* prefix)
{
    prefix[lane]=value; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=1;stride<BatchWidth;stride*=2)
    { uint3 increment=lane>=stride?prefix[lane-stride]:uint3(0); threadgroup_barrier(mem_flags::mem_threadgroup); prefix[lane]+=increment; threadgroup_barrier(mem_flags::mem_threadgroup); }
    return prefix[lane];
}
kernel void EvansVirtualEvaluate(constant VirtualFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device const uint4* map [[buffer(2)]],device const uint* pool [[buffer(3)]],device atomic_uint* state [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    Memory m{heap,map,pool,state,nullptr,nullptr}; uint id=Group(f,group)*BatchWidth+lane; if (id>=f.Counts.x) return; uint n=Node(f,id),s=id*NodeStateStride; bool own=true; float error=0;
    for (uint view=0;view<f.Counts.y;++view) error=max(error,PixelError(m,f,id,view));
    for (uint at=0;at<heap[n+PageCount];++at) own=Resident(m,f,heap[f.Offsets.w+heap[n+FirstPage]+at]) && own;
    Write(m,s+ErrorWord,as_type<uint>(error)); Write(m,s+OwnWord,own?1:0); Write(m,s+AvailableWord,own?1:0); Write(m,s+HistoryWord,0);
}
kernel void EvansVirtualPropagate(constant VirtualFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device atomic_uint* state [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    Memory m{heap,nullptr,nullptr,state,nullptr,nullptr}; uint ordinal=Group(f,group)*BatchWidth+lane; if (ordinal>=f.Dispatch.y) return;
    uint id=heap[f.Tables.z+f.Dispatch.x+ordinal],s=id*NodeStateStride; uint4 n=Read4(heap,Node(f,id)); bool ready=true,history=false;
    for (uint at=0;at<n.z;++at) { uint child=heap[f.Offsets.y+n.y+at],t=child*NodeStateStride; ready=ready && Read(m,t+AvailableWord)!=0; history=history || Read(m,t+HistoryWord)!=0 || (heap[Node(f,child)+PreviousSelection]&PreviousFlag)!=0; }
    Write(m,s+AvailableWord,(Read(m,s+OwnWord)!=0 || ready)?1:0); Write(m,s+HistoryWord,history?1:0);
}
kernel void EvansVirtualSelect(constant VirtualFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device atomic_uint* state [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    Memory m{heap,nullptr,nullptr,state,nullptr,nullptr}; uint id=Group(f,group)*BatchWidth+lane; if (id>=f.Counts.x) return; uint n=Node(f,id),s=id*NodeStateStride; bool reached=true,suppressed=false;
    for (uint parent=heap[n];parent!=Invalid;parent=heap[Node(f,parent)]) { reached=reached && !ChoosesSelf(m,f,parent); suppressed=suppressed || (Read(m,parent*NodeStateStride+OwnWord)==0 && Children(m,f,parent) && !Fine(m,f,parent)); }
    if (!reached) return; float scale=max(f.Quality.x,MinimumNormal);
    if (ChoosesSelf(m,f,id)) { Write(m,s+SelectedWord,1); for (uint at=0;at<heap[n+ClusterCount];++at) Write(m,ClusterState(f)+heap[f.Offsets.z+heap[n+FirstCluster]+at]*ClusterStateStride,1);
        if (Fine(m,f,id) && !suppressed) { Write(m,s+TransitionWord,1); Write(m,s+PriorityWord,as_type<uint>(min(as_type<float>(MaximumFloatBits),1+as_type<float>(Read(m,s+ErrorWord))/scale))); } }
    else if (Read(m,s+OwnWord)==0 && Children(m,f,id) && !Fine(m,f,id) && !suppressed) { Write(m,s+TransitionWord,2); Write(m,s+PriorityWord,as_type<uint>(1+1/(1+as_type<float>(Read(m,s+ErrorWord))/scale))); }
    else if (Read(m,s+OwnWord)==0 && !Children(m,f,id)) Reject(m,f,MissingCoverage);
}
kernel void EvansVirtualClusters(constant VirtualFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device atomic_uint* state [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    Memory m{heap,nullptr,nullptr,state,nullptr,nullptr}; uint block=Group(f,group); if (block>=Blocks(f)) return; uint id=block*BatchWidth+lane,s=ClusterState(f)+id*ClusterStateStride; uint3 value=0; threadgroup uint3 prefix[BatchWidth];
    if (id<f.Limits.x && Read(m,s)!=0) { uint4 d=Read4(heap,Directory(f,id)); value=uint3(d.z,d.w,1); }
    uint3 inclusive=Scan(value,lane,prefix),exclusive=inclusive-value;
    if (id<f.Limits.x) { Write(m,s+VertexPrefix,exclusive.x); Write(m,s+IndexPrefix,exclusive.y); Write(m,s+ActivePrefix,exclusive.z); }
    if (lane==BatchWidth-1) { uint b=BlockState(f)+block*VectorWords; Write(m,b,inclusive.x); Write(m,b+1,inclusive.y); Write(m,b+2,inclusive.z); }
}
kernel void EvansVirtualBlocks(constant VirtualFrame& f [[buffer(0)]],device atomic_uint* state [[buffer(4)]],uint lane [[thread_index_in_threadgroup]])
{
    Memory m{nullptr,nullptr,nullptr,state,nullptr,nullptr}; threadgroup uint3 prefix[BatchWidth]; threadgroup uint3 carry;
    if (lane==0) carry=0; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint begin=0;begin<Blocks(f);begin+=BatchWidth)
    {
        uint block=begin+lane,b=BlockState(f)+block*VectorWords; uint3 value=0; if (block<Blocks(f)) value=uint3(Read(m,b),Read(m,b+1),Read(m,b+2));
        uint3 inclusive=Scan(value,lane,prefix),exclusive=carry+inclusive-value;
        if (block<Blocks(f)) { Write(m,b,exclusive.x); Write(m,b+1,exclusive.y); Write(m,b+2,exclusive.z); }
        threadgroup_barrier(mem_flags::mem_threadgroup); if (lane==BatchWidth-1) carry+=inclusive; threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane==0) { Write(m,Status(f)+1,carry.x); Write(m,Status(f)+2,carry.y); Write(m,Status(f)+3,carry.z); if (carry.x==0 || carry.y==0 || carry.x>f.Capacity.x || carry.y>f.Capacity.y) Reject(m,f,OutputOverflow); }
}
kernel void EvansVirtualDecode(constant VirtualFrame& f [[buffer(0)]],device const uint* heap [[buffer(1)]],device const uint4* map [[buffer(2)]],device const uint* pool [[buffer(3)]],device atomic_uint* state [[buffer(4)]],device uint* vertices [[buffer(5)]],device uint* indices [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]])
{
    Memory m{heap,map,pool,state,vertices,indices}; uint id=Group(f,group); if (id>=f.Limits.x || Read(m,Status(f))!=0) return; uint s=ClusterState(f)+id*ClusterStateStride; if (Read(m,s)==0) return;
    uint4 d=Read4(heap,Directory(f,id)),identity=Read4(heap,Directory(f,id)+VectorWords); if (!Resident(m,f,d.x)) return;
    uint base=map[d.x].x*PageStride,h=base+d.y,firstV=d.y+ClusterHeader,firstI=firstV+d.z*VertexStride,used=pool[base+UsedWord];
    bool valid=pool[base]==Magic && pool[base+1]==Version && used<=PageStride && firstI+d.w<=used && pool[h]==d.z && pool[h+1]==d.w && pool[h+2]==firstV && pool[h+3]==firstI && pool[h+MaterialWord]==identity.x && pool[h+DirectoryIdWord]==identity.z;
    if (!valid) { Reject(m,f,BadPayload); return; }
    uint b=BlockState(f)+(id/BatchWidth)*VectorWords,outV=Read(m,b)+Read(m,s+VertexPrefix),outI=Read(m,b+1)+Read(m,s+IndexPrefix);
    if (outV>f.Capacity.x || d.z>f.Capacity.x-outV || outI>f.Capacity.y || d.w>f.Capacity.y-outI) { Reject(m,f,OutputOverflow); return; }
    for (uint word=lane;word<d.z*VertexStride;word+=BatchWidth) vertices[outV*VertexStride+word]=pool[base+firstV+word];
    for (uint at=lane;at<d.w;at+=BatchWidth) { uint local=pool[base+firstI+at]; if (local>=d.z) Reject(m,f,BadPayload); else indices[outI+at]=outV+local; }
}
