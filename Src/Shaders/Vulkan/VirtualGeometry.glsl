#version 450
// Vulkan GLSL is compiled in memory at runtime, never embedded as SPIR-V arrays.
// Independent virtual geometry: conservative projection, resident tree selection,
// two-level scan and lossless decoding. Hoppe, SIGGRAPH 1997:
// https://hhoppe.com/proj/vdrpm/. This page format is independent of Nanite.
// Compute/compute barriers separate phases; decode needs vertex/index/host visibility.
#define VIRTUAL_GROUP_WIDTH 64
layout(local_size_x=VIRTUAL_GROUP_WIDTH) in;
layout(std140,set=0,binding=0) uniform Parameters { uvec4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; vec4 Quality; } Frame;
layout(std430,set=0,binding=1) readonly buffer MetadataBuffer { uint Words[]; } Heap;
layout(std430,set=0,binding=2) readonly buffer MappingBuffer { uvec4 Values[]; } Mapping;
layout(std430,set=0,binding=3) readonly buffer PhysicalBuffer { uint Words[]; } Physical;
layout(std430,set=0,binding=4) buffer StateBuffer { uint Words[]; } State;
layout(std430,set=0,binding=5) writeonly buffer VertexBuffer { uint Words[]; } Vertices;
layout(std430,set=0,binding=6) writeonly buffer IndexBuffer { uint Values[]; } Indices;
const uint GroupWidth=64u,NodeWords=20u,ViewWords=36u,DirectoryWords=8u,NodeStateWords=8u,ClusterStateWords=4u;
const uint FirstCluster=4u,ClusterCount=5u,FirstPage=6u,PageCount=7u,Previous=10u,Minimum=12u,Maximum=16u,ViewError=16u,Viewport=32u;
const uint ErrorWord=0u,OwnWord=1u,AvailableWord=2u,HistoryWord=3u,SelectedWord=4u,TransitionWord=5u,PriorityWord=6u;
const uint VertexPrefix=1u,IndexPrefix=2u,ActivePrefix=3u,VectorWords=4u;
const uint PageWords=2048u,PageHeaderWords=8u,ClusterHeaderWords=16u,VertexWords=12u,PageId=5u,PageGeneration=6u,PageEpoch=7u,PageUsed=2u,MaterialWord=4u,DirectoryId=5u;
const uint Invalid=0xffffffffu,InfinityBits=0x7f800000u,MaximumFloatBits=0x7f7fffffu,Magic=0x50475645u,Version=1u,PreviouslySelected=1u;
const uint MissingPage=1u,StalePage=2u,BadPayload=4u,OutputOverflow=8u;
const float Epsilon=1.1920928955078125e-7,MinimumNormal=1.1754943508222875e-38,ArithmeticOperations=16.0,Roundoff=ArithmeticOperations*Epsilon,Underflow=ArithmeticOperations*MinimumNormal;
shared uvec3 Prefix[VIRTUAL_GROUP_WIDTH]; shared uvec3 Carry;
uvec4 Read4(uint p) { return uvec4(Heap.Words[p],Heap.Words[p+1u],Heap.Words[p+2u],Heap.Words[p+3u]); }
uint Node(uint id) { return Frame.Offsets.x+id*NodeWords; }
uint Directory(uint id) { return Frame.Tables.x+id*DirectoryWords; }
uint ClusterState() { return Frame.Counts.x*NodeStateWords; }
uint Blocks() { return (Frame.Limits.x+GroupWidth-1u)/GroupWidth; }
uint BlockState() { return ClusterState()+Frame.Limits.x*ClusterStateWords; }
uint Status() { return BlockState()+Blocks()*VectorWords; }
uint Group() { return gl_WorkGroupID.y*Frame.Dispatch.z+gl_WorkGroupID.x; }
void Reject(uint reason) { atomicOr(State.Words[Status()],reason); }
bool Finite(float x) { return !isnan(x) && !isinf(x); }
float Up(float x) { return Finite(x) && x>=0?x*(1+Roundoff)+Underflow:uintBitsToFloat(InfinityBits); }
float Down(float x) { return x>0?x*(1-Roundoff)-Underflow:x; }
bool Resident(uint page)
{
    uvec4 entry=Mapping.Values[page]; if (entry.x==Invalid) return false;
    if (entry.x>=Frame.Counts.w || entry.y==0u || any(notEqual(entry.zw,Frame.Identity.xy))) { Reject(StalePage); return false; }
    uint base=entry.x*PageWords;
    if (Physical.Words[base+PageId]!=page || Physical.Words[base+PageGeneration]!=entry.y || Physical.Words[base+PageEpoch]!=entry.z) { Reject(StalePage); return false; }
    return true;
}
float PixelError(uint node,uint view)
{
    vec4 lo=uintBitsToFloat(Read4(Node(node)+Minimum)),hi=uintBitsToFloat(Read4(Node(node)+Maximum)); if (lo.w==0) return 0;
    uint v=Frame.Tables.y+view*ViewWords; float minimumW=uintBitsToFloat(InfinityBits),maximumX=0,maximumY=0;
    for (uint corner=0u;corner<8u;++corner)
    {
        vec4 p=vec4((corner&1u)!=0u?hi.x:lo.x,(corner&2u)!=0u?hi.y:lo.y,(corner&4u)!=0u?hi.z:lo.z,1),clip=vec4(0),error=vec4(0);
        for (uint row=0u;row<4u;++row)
        { vec4 a=uintBitsToFloat(Read4(v+row*VectorWords)),e=uintBitsToFloat(Read4(v+ViewError+row*VectorWords)); clip[row]=dot(a,p); error[row]=Up(dot(abs(a),abs(p))*Roundoff+dot(e,abs(p))); }
        float w=Down(clip.w-error.w),z=Down(clip.z-error.z); if (!(w>0) || !(z>0) || !Finite(w)) return uintBitsToFloat(InfinityBits);
        minimumW=min(minimumW,w); maximumX=max(maximumX,Up(Up(abs(clip.x)+error.x)/w)); maximumY=max(maximumY,Up(Up(abs(clip.y)+error.y)/w));
    }
    vec4 delta=vec4(0);
    for (uint row=0u;row<4u;++row)
    { vec3 norm=abs(uintBitsToFloat(Read4(v+row*VectorWords)).xyz)+uintBitsToFloat(Read4(v+ViewError+row*VectorWords)).xyz; delta[row]=Up(Up(norm.x+norm.y+norm.z)*lo.w); }
    float denominator=Down(minimumW-delta.w); if (!(denominator>0)) return uintBitsToFloat(InfinityBits);
    float x=Up(Up(delta.x+Up(maximumX*delta.w))/denominator),y=Up(Up(delta.y+Up(maximumY*delta.w))/denominator);
    vec2 halfViewport=uintBitsToFloat(Read4(v+Viewport)).xy; return Up(Up(x*halfViewport.x)+Up(y*halfViewport.y));
}
bool ChildrenAvailable(uint id)
{
    uvec4 link=Read4(Node(id)); if (link.z==0u) return false;
    for (uint at=0u;at<link.z;++at) if (State.Words[Heap.Words[Frame.Offsets.y+link.y+at]*NodeStateWords+AvailableWord]==0u) return false; return true;
}
bool Refine(uint id) { uint s=id*NodeStateWords; return Read4(Node(id)).z!=0u && uintBitsToFloat(State.Words[s+ErrorWord])>(State.Words[s+HistoryWord]!=0u?Frame.Quality.y:Frame.Quality.x); }
bool SelectsSelf(uint id) { return State.Words[id*NodeStateWords+OwnWord]!=0u && (!Refine(id) || !ChildrenAvailable(id)); }
uvec3 Scan(uvec3 value,uint lane)
{
    Prefix[lane]=value; barrier();
    for (uint stride=1u;stride<GroupWidth;stride*=2u)
    { uvec3 increment=lane>=stride?Prefix[lane-stride]:uvec3(0); barrier(); Prefix[lane]+=increment; barrier(); }
    return Prefix[lane];
}
void main()
{
    uint lane=gl_LocalInvocationIndex,group=Group(),id=group*GroupWidth+lane;
#if EVANS_VIRTUAL_GEOMETRY_PASS == 0
    if (id>=Frame.Counts.x) return; uint n=Node(id),s=id*NodeStateWords; float pixels=0; bool own=true;
    for (uint view=0u;view<Frame.Counts.y;++view) pixels=max(pixels,PixelError(id,view));
    for (uint at=0u;at<Heap.Words[n+PageCount];++at) own=Resident(Heap.Words[Frame.Offsets.w+Heap.Words[n+FirstPage]+at]) && own;
    State.Words[s+ErrorWord]=floatBitsToUint(pixels); State.Words[s+OwnWord]=own?1u:0u; State.Words[s+AvailableWord]=own?1u:0u; State.Words[s+HistoryWord]=0u;
#elif EVANS_VIRTUAL_GEOMETRY_PASS == 1
    if (id>=Frame.Dispatch.y) return; uint node=Heap.Words[Frame.Tables.z+Frame.Dispatch.x+id],s=node*NodeStateWords; uvec4 link=Read4(Node(node)); bool ready=true,refined=false;
    for (uint at=0u;at<link.z;++at) { uint child=Heap.Words[Frame.Offsets.y+link.y+at],t=child*NodeStateWords; ready=ready && State.Words[t+AvailableWord]!=0u; refined=refined || State.Words[t+HistoryWord]!=0u || (Heap.Words[Node(child)+Previous]&PreviouslySelected)!=0u; }
    State.Words[s+AvailableWord]=(State.Words[s+OwnWord]!=0u || ready)?1u:0u; State.Words[s+HistoryWord]=refined?1u:0u;
#elif EVANS_VIRTUAL_GEOMETRY_PASS == 2
    if (id>=Frame.Counts.x) return; uint n=Node(id),s=id*NodeStateWords; bool reached=true,suppressed=false;
    for (uint parent=Heap.Words[n];parent!=Invalid;parent=Heap.Words[Node(parent)]) { reached=reached && !SelectsSelf(parent); suppressed=suppressed || (State.Words[parent*NodeStateWords+OwnWord]==0u && ChildrenAvailable(parent) && !Refine(parent)); }
    if (!reached) return; float scale=max(Frame.Quality.x,MinimumNormal);
    if (SelectsSelf(id))
    { State.Words[s+SelectedWord]=1u; for (uint at=0u;at<Heap.Words[n+ClusterCount];++at) State.Words[ClusterState()+Heap.Words[Frame.Offsets.z+Heap.Words[n+FirstCluster]+at]*ClusterStateWords]=1u;
      if (Refine(id) && !suppressed) { State.Words[s+TransitionWord]=1u; State.Words[s+PriorityWord]=floatBitsToUint(min(uintBitsToFloat(MaximumFloatBits),1+uintBitsToFloat(State.Words[s+ErrorWord])/scale)); } }
    else if (State.Words[s+OwnWord]==0u && ChildrenAvailable(id) && !Refine(id) && !suppressed) { State.Words[s+TransitionWord]=2u; State.Words[s+PriorityWord]=floatBitsToUint(1+1/(1+uintBitsToFloat(State.Words[s+ErrorWord])/scale)); }
    else if (State.Words[s+OwnWord]==0u && !ChildrenAvailable(id)) Reject(MissingPage);
#elif EVANS_VIRTUAL_GEOMETRY_PASS == 3
    if (group>=Blocks()) return; uint s=ClusterState()+id*ClusterStateWords; uvec3 value=uvec3(0);
    if (id<Frame.Limits.x && State.Words[s]!=0u) { uvec4 record=Read4(Directory(id)); value=uvec3(record.zw,1u); }
    uvec3 inclusive=Scan(value,lane),exclusive=inclusive-value;
    if (id<Frame.Limits.x) { State.Words[s+VertexPrefix]=exclusive.x; State.Words[s+IndexPrefix]=exclusive.y; State.Words[s+ActivePrefix]=exclusive.z; }
    if (lane==GroupWidth-1u) { uint b=BlockState()+group*VectorWords; State.Words[b]=inclusive.x; State.Words[b+1u]=inclusive.y; State.Words[b+2u]=inclusive.z; }
#elif EVANS_VIRTUAL_GEOMETRY_PASS == 4
    if (lane==0u) Carry=uvec3(0); barrier();
    for (uint begin=0u;begin<Blocks();begin+=GroupWidth)
    {
        uint block=begin+lane,b=BlockState()+block*VectorWords; uvec3 value=uvec3(0); if (block<Blocks()) value=uvec3(State.Words[b],State.Words[b+1u],State.Words[b+2u]);
        uvec3 inclusive=Scan(value,lane),exclusive=Carry+inclusive-value;
        if (block<Blocks()) { State.Words[b]=exclusive.x; State.Words[b+1u]=exclusive.y; State.Words[b+2u]=exclusive.z; }
        barrier(); if (lane==GroupWidth-1u) Carry+=inclusive; barrier();
    }
    if (lane==0u) { State.Words[Status()+1u]=Carry.x; State.Words[Status()+2u]=Carry.y; State.Words[Status()+3u]=Carry.z; if (Carry.x==0u || Carry.y==0u || Carry.x>Frame.Capacity.x || Carry.y>Frame.Capacity.y) Reject(OutputOverflow); }
#else
    id=group; if (id>=Frame.Limits.x || State.Words[Status()]!=0u) return; uint s=ClusterState()+id*ClusterStateWords; if (State.Words[s]==0u) return;
    uvec4 record=Read4(Directory(id)),identity=Read4(Directory(id)+VectorWords); if (!Resident(record.x)) return;
    uint base=Mapping.Values[record.x].x*PageWords,h=base+record.y,firstV=record.y+ClusterHeaderWords,firstI=firstV+record.z*VertexWords,used=Physical.Words[base+PageUsed];
    bool valid=Physical.Words[base]==Magic && Physical.Words[base+1u]==Version && used<=PageWords && firstI+record.w<=used &&
        Physical.Words[h]==record.z && Physical.Words[h+1u]==record.w && Physical.Words[h+2u]==firstV && Physical.Words[h+3u]==firstI && Physical.Words[h+MaterialWord]==identity.x && Physical.Words[h+DirectoryId]==identity.z;
    if (!valid) { Reject(BadPayload); return; }
    uint b=BlockState()+(id/GroupWidth)*VectorWords,outV=State.Words[b]+State.Words[s+VertexPrefix],outI=State.Words[b+1u]+State.Words[s+IndexPrefix];
    if (outV>Frame.Capacity.x || record.z>Frame.Capacity.x-outV || outI>Frame.Capacity.y || record.w>Frame.Capacity.y-outI) { Reject(OutputOverflow); return; }
    for (uint word=lane;word<record.z*VertexWords;word+=GroupWidth) Vertices.Words[outV*VertexWords+word]=Physical.Words[base+firstV+word];
    for (uint at=lane;at<record.w;at+=GroupWidth) { uint local=Physical.Words[base+firstI+at]; if (local>=record.z) Reject(BadPayload); else Indices.Values[outI+at]=outV+local; }
#endif
}
