#version 310 es
// Independent ES 3.1 virtual geometry, using six SSBOs and no optional mesh stage.
// Hoppe, "View-dependent refinement of progressive meshes", SIGGRAPH 1997:
// https://hhoppe.com/proj/vdrpm/. Lossless pages and region boundaries are local.
// SSBO barriers separate phases; vertex/element barriers precede drawing.
precision highp float;
precision highp int;
#define VIRTUAL_BATCH_WIDTH 64
layout(local_size_x=VIRTUAL_BATCH_WIDTH) in;
layout(std140,binding=0) uniform Parameters { uvec4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; vec4 Quality; } P;
layout(std430,binding=0) readonly buffer Metadata { uint Data[]; } H;
layout(std430,binding=1) readonly buffer PageTable { uvec4 Data[]; } Map;
layout(std430,binding=2) readonly buffer PagePool { uint Data[]; } Pool;
layout(std430,binding=3) buffer Work { uint Data[]; } S;
layout(std430,binding=4) writeonly buffer DrawableVertices { uint Data[]; } V;
layout(std430,binding=5) writeonly buffer DrawableIndices { uint Data[]; } I;
const uint BatchWidth=64u,NodeStride=20u,ViewStride=36u,ClusterStride=8u,NodeWorkStride=8u,ClusterWorkStride=4u,VectorWords=4u;
const uint FirstClusterWord=4u,ClusterCountWord=5u,FirstPageWord=6u,PageCountWord=7u,PreviousWord=10u,MinimumWord=12u,MaximumWord=16u,ViewErrorWord=16u,ViewportWord=32u;
const uint PixelWord=0u,OwnWord=1u,AvailableWord=2u,HistoryWord=3u,SelectedWord=4u,TransitionWord=5u,PriorityWord=6u;
const uint VertexPrefixWord=1u,IndexPrefixWord=2u,ActivePrefixWord=3u;
const uint PageStride=2048u,ClusterHeaderWords=16u,VertexStride=12u,IdWord=5u,GenerationWord=6u,EpochWord=7u,UsedWord=2u,MaterialWord=4u,ClusterIdWord=5u;
const uint Invalid=0xffffffffu,InfinityBits=0x7f800000u,MaximumBits=0x7f7fffffu,Magic=0x50475645u,Version=1u,PreviousFlag=1u;
const uint Missing=1u,Stale=2u,Malformed=4u,Overflow=8u;
const float Epsilon=1.1920928955078125e-7,MinimumNormal=1.1754943508222875e-38,ArithmeticOperations=16.0,Allowance=ArithmeticOperations*Epsilon,AbsoluteAllowance=ArithmeticOperations*MinimumNormal;
shared uvec3 ScanValues[VIRTUAL_BATCH_WIDTH]; shared uvec3 ScanCarry;
uvec4 Fetch(uint p) { return uvec4(H.Data[p],H.Data[p+1u],H.Data[p+2u],H.Data[p+3u]); }
uint Node(uint id) { return P.Offsets.x+id*NodeStride; }
uint Cluster(uint id) { return P.Tables.x+id*ClusterStride; }
uint ClusterWork() { return P.Counts.x*NodeWorkStride; }
uint Batches() { return (P.Limits.x+BatchWidth-1u)/BatchWidth; }
uint BatchWork() { return ClusterWork()+P.Limits.x*ClusterWorkStride; }
uint Result() { return BatchWork()+Batches()*VectorWords; }
uint Group() { return gl_WorkGroupID.y*P.Dispatch.z+gl_WorkGroupID.x; }
void Error(uint flag) { atomicOr(S.Data[Result()],flag); }
bool Finite(float x) { return !isnan(x) && !isinf(x); }
float Expand(float x) { return Finite(x) && x>=0.0?x*(1.0+Allowance)+AbsoluteAllowance:uintBitsToFloat(InfinityBits); }
float Contract(float x) { return x>0.0?x*(1.0-Allowance)-AbsoluteAllowance:x; }
bool Present(uint page)
{
    uvec4 entry=Map.Data[page]; if (entry.x==Invalid) return false;
    if (entry.x>=P.Counts.w || entry.y==0u || any(notEqual(entry.zw,P.Identity.xy))) { Error(Stale); return false; }
    uint base=entry.x*PageStride;
    if (Pool.Data[base+IdWord]!=page || Pool.Data[base+GenerationWord]!=entry.y || Pool.Data[base+EpochWord]!=entry.z) { Error(Stale); return false; } return true;
}
float ScreenError(uint id,uint view)
{
    vec4 lo=uintBitsToFloat(Fetch(Node(id)+MinimumWord)),hi=uintBitsToFloat(Fetch(Node(id)+MaximumWord)); if (lo.w==0.0) return 0.0;
    uint v=P.Tables.y+view*ViewStride; float minimumW=uintBitsToFloat(InfinityBits),maxX=0.0,maxY=0.0;
    for (uint corner=0u;corner<8u;++corner)
    {
        vec4 p=vec4((corner&1u)!=0u?hi.x:lo.x,(corner&2u)!=0u?hi.y:lo.y,(corner&4u)!=0u?hi.z:lo.z,1.0),clip=vec4(0),error=vec4(0);
        for (uint row=0u;row<4u;++row)
        { vec4 coefficients=uintBitsToFloat(Fetch(v+row*VectorWords)),uncertainty=uintBitsToFloat(Fetch(v+ViewErrorWord+row*VectorWords)); clip[row]=dot(coefficients,p); error[row]=Expand(dot(abs(coefficients),abs(p))*Allowance+dot(uncertainty,abs(p))); }
        float w=Contract(clip.w-error.w),z=Contract(clip.z-error.z); if (!(w>0.0) || !(z>0.0) || !Finite(w)) return uintBitsToFloat(InfinityBits);
        minimumW=min(minimumW,w); maxX=max(maxX,Expand(Expand(abs(clip.x)+error.x)/w)); maxY=max(maxY,Expand(Expand(abs(clip.y)+error.y)/w));
    }
    vec4 radii=vec4(0);
    for (uint row=0u;row<4u;++row) { vec3 norm=abs(uintBitsToFloat(Fetch(v+row*VectorWords)).xyz)+uintBitsToFloat(Fetch(v+ViewErrorWord+row*VectorWords)).xyz; radii[row]=Expand(Expand(norm.x+norm.y+norm.z)*lo.w); }
    float denominator=Contract(minimumW-radii.w); if (!(denominator>0.0)) return uintBitsToFloat(InfinityBits);
    float x=Expand(Expand(radii.x+Expand(maxX*radii.w))/denominator),y=Expand(Expand(radii.y+Expand(maxY*radii.w))/denominator);
    vec2 viewport=uintBitsToFloat(Fetch(v+ViewportWord)).xy; return Expand(Expand(x*viewport.x)+Expand(y*viewport.y));
}
bool Children(uint id)
{
    uvec4 n=Fetch(Node(id)); if (n.z==0u) return false;
    for (uint at=0u;at<n.z;++at) if (S.Data[H.Data[P.Offsets.y+n.y+at]*NodeWorkStride+AvailableWord]==0u) return false; return true;
}
bool Fine(uint id) { uint s=id*NodeWorkStride; return Fetch(Node(id)).z!=0u && uintBitsToFloat(S.Data[s+PixelWord])>(S.Data[s+HistoryWord]!=0u?P.Quality.y:P.Quality.x); }
bool Pick(uint id) { return S.Data[id*NodeWorkStride+OwnWord]!=0u && (!Fine(id) || !Children(id)); }
uvec3 Inclusive(uvec3 value,uint lane)
{
    ScanValues[lane]=value; barrier();
    for (uint stride=1u;stride<BatchWidth;stride*=2u)
    { uvec3 increment=lane>=stride?ScanValues[lane-stride]:uvec3(0); barrier(); ScanValues[lane]+=increment; barrier(); }
    return ScanValues[lane];
}
void main()
{
    uint lane=gl_LocalInvocationIndex,group=Group(),id=group*BatchWidth+lane;
#if EVANS_VIRTUAL_GEOMETRY_PASS == 0
    if (id>=P.Counts.x) return; uint n=Node(id),s=id*NodeWorkStride; bool own=true; float error=0.0;
    for (uint view=0u;view<P.Counts.y;++view) error=max(error,ScreenError(id,view));
    for (uint at=0u;at<H.Data[n+PageCountWord];++at) own=Present(H.Data[P.Offsets.w+H.Data[n+FirstPageWord]+at]) && own;
    S.Data[s+PixelWord]=floatBitsToUint(error); S.Data[s+OwnWord]=own?1u:0u; S.Data[s+AvailableWord]=own?1u:0u; S.Data[s+HistoryWord]=0u;
#elif EVANS_VIRTUAL_GEOMETRY_PASS == 1
    if (id>=P.Dispatch.y) return; uint node=H.Data[P.Tables.z+P.Dispatch.x+id],s=node*NodeWorkStride; uvec4 n=Fetch(Node(node)); bool ready=true,refined=false;
    for (uint at=0u;at<n.z;++at) { uint child=H.Data[P.Offsets.y+n.y+at],t=child*NodeWorkStride; ready=ready && S.Data[t+AvailableWord]!=0u; refined=refined || S.Data[t+HistoryWord]!=0u || (H.Data[Node(child)+PreviousWord]&PreviousFlag)!=0u; }
    S.Data[s+AvailableWord]=(S.Data[s+OwnWord]!=0u || ready)?1u:0u; S.Data[s+HistoryWord]=refined?1u:0u;
#elif EVANS_VIRTUAL_GEOMETRY_PASS == 2
    if (id>=P.Counts.x) return; uint n=Node(id),s=id*NodeWorkStride; bool reached=true,suppressed=false;
    for (uint parent=H.Data[n];parent!=Invalid;parent=H.Data[Node(parent)]) { reached=reached && !Pick(parent); suppressed=suppressed || (S.Data[parent*NodeWorkStride+OwnWord]==0u && Children(parent) && !Fine(parent)); }
    if (!reached) return; float scale=max(P.Quality.x,MinimumNormal);
    if (Pick(id)) { S.Data[s+SelectedWord]=1u; for (uint at=0u;at<H.Data[n+ClusterCountWord];++at) S.Data[ClusterWork()+H.Data[P.Offsets.z+H.Data[n+FirstClusterWord]+at]*ClusterWorkStride]=1u;
        if (Fine(id) && !suppressed) { S.Data[s+TransitionWord]=1u; S.Data[s+PriorityWord]=floatBitsToUint(min(uintBitsToFloat(MaximumBits),1.0+uintBitsToFloat(S.Data[s+PixelWord])/scale)); } }
    else if (S.Data[s+OwnWord]==0u && Children(id) && !Fine(id) && !suppressed) { S.Data[s+TransitionWord]=2u; S.Data[s+PriorityWord]=floatBitsToUint(1.0+1.0/(1.0+uintBitsToFloat(S.Data[s+PixelWord])/scale)); }
    else if (S.Data[s+OwnWord]==0u && !Children(id)) Error(Missing);
#elif EVANS_VIRTUAL_GEOMETRY_PASS == 3
    if (group>=Batches()) return; uint s=ClusterWork()+id*ClusterWorkStride; uvec3 value=uvec3(0);
    if (id<P.Limits.x && S.Data[s]!=0u) { uvec4 record=Fetch(Cluster(id)); value=uvec3(record.zw,1u); }
    uvec3 total=Inclusive(value,lane),offset=total-value;
    if (id<P.Limits.x) { S.Data[s+VertexPrefixWord]=offset.x; S.Data[s+IndexPrefixWord]=offset.y; S.Data[s+ActivePrefixWord]=offset.z; }
    if (lane==BatchWidth-1u) { uint b=BatchWork()+group*VectorWords; S.Data[b]=total.x; S.Data[b+1u]=total.y; S.Data[b+2u]=total.z; }
#elif EVANS_VIRTUAL_GEOMETRY_PASS == 4
    if (lane==0u) ScanCarry=uvec3(0); barrier();
    for (uint begin=0u;begin<Batches();begin+=BatchWidth) { uint block=begin+lane,b=BatchWork()+block*VectorWords; uvec3 value=uvec3(0); if (block<Batches()) value=uvec3(S.Data[b],S.Data[b+1u],S.Data[b+2u]);
        uvec3 total=Inclusive(value,lane),offset=ScanCarry+total-value;
        if (block<Batches()) { S.Data[b]=offset.x; S.Data[b+1u]=offset.y; S.Data[b+2u]=offset.z; }
        barrier(); if (lane==BatchWidth-1u) ScanCarry+=total; barrier(); }
    if (lane==0u) { S.Data[Result()+1u]=ScanCarry.x; S.Data[Result()+2u]=ScanCarry.y; S.Data[Result()+3u]=ScanCarry.z; if (ScanCarry.x==0u || ScanCarry.y==0u || ScanCarry.x>P.Capacity.x || ScanCarry.y>P.Capacity.y) Error(Overflow); }
#else
    id=group; if (id>=P.Limits.x || S.Data[Result()]!=0u) return; uint s=ClusterWork()+id*ClusterWorkStride; if (S.Data[s]==0u) return;
    uvec4 d=Fetch(Cluster(id)),identity=Fetch(Cluster(id)+VectorWords); if (!Present(d.x)) return;
    uint base=Map.Data[d.x].x*PageStride,h=base+d.y,firstV=d.y+ClusterHeaderWords,firstI=firstV+d.z*VertexStride,used=Pool.Data[base+UsedWord];
    bool valid=Pool.Data[base]==Magic && Pool.Data[base+1u]==Version && used<=PageStride && firstI+d.w<=used && Pool.Data[h]==d.z && Pool.Data[h+1u]==d.w && Pool.Data[h+2u]==firstV && Pool.Data[h+3u]==firstI && Pool.Data[h+MaterialWord]==identity.x && Pool.Data[h+ClusterIdWord]==identity.z;
    if (!valid) { Error(Malformed); return; }
    uint b=BatchWork()+(id/BatchWidth)*VectorWords,outV=S.Data[b]+S.Data[s+VertexPrefixWord],outI=S.Data[b+1u]+S.Data[s+IndexPrefixWord];
    if (outV>P.Capacity.x || d.z>P.Capacity.x-outV || outI>P.Capacity.y || d.w>P.Capacity.y-outI) { Error(Overflow); return; }
    for (uint word=lane;word<d.z*VertexStride;word+=BatchWidth) V.Data[outV*VertexStride+word]=Pool.Data[base+firstV+word];
    for (uint at=lane;at<d.w;at+=BatchWidth) { uint local=Pool.Data[base+firstI+at]; if (local>=d.z) Error(Malformed); else I.Data[outI+at]=outV+local; }
#endif
}
