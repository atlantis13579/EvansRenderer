#version 310 es
precision highp float;
precision highp int;
#if defined(EVANS_NANITE_WIRE_EDGES)
// ES has no polygon line mode. Expand native triangle indices into line pairs
// on the GPU, preserving the selected cut without a host geometry mirror.
#define NANITE_EDGE_THREADS 64
layout(local_size_x=NANITE_EDGE_THREADS,local_size_y=1,local_size_z=1) in;
layout(std140,binding=0) uniform EdgeFrame { uvec4 EdgeDraw; }; // First index, count, dispatch row stride, reserved.
layout(std430,binding=0) readonly buffer EdgeInput { uint SourceIndex[]; };
layout(std430,binding=1) writeonly buffer EdgeOutput { uint LineIndex[]; };
const uint TriangleCorners=3u,LineCorners=2u,EdgesPerTriangle=3u;
void main()
{
    uint triangle=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*EdgeDraw.z;
    if (triangle>=EdgeDraw.y/TriangleCorners) return;
    uint source=EdgeDraw.x+triangle*TriangleCorners;
    for (uint edge=0u;edge<EdgesPerTriangle;++edge)
    {
        uint target=(triangle*EdgesPerTriangle+edge)*LineCorners;
        LineIndex[target]=SourceIndex[source+edge];
        LineIndex[target+1u]=SourceIndex[source+(edge+1u)%TriangleCorners];
    }
}
#else
// OpenGL ES 3.1 Nanite: six SSBO bindings and a UBO. Every stage owns its
// storage barrier; CPU readback follows buffer visibility, and drawable use
// follows vertex/index visibility. This source is independent of other APIs.
// Reference: Karis, Stubbe and Wihlidal, Nanite SIGGRAPH 2021 deep dive.
#define NANITE_THREADS 64
#define PASS_EVALUATION 0
#define PASS_REFINEMENT 1
#define PASS_CLUSTER_SCAN 2
#define PASS_BLOCK_SCAN 3
#define PASS_DECODING 4
layout(local_size_x=NANITE_THREADS) in;
layout(std140,binding=0) uniform FrameBlock { uvec4 Counts,Offsets,Tables,Dispatch,Limits,Capacity,Identity; vec4 Quality; } F;
layout(std430,binding=0) readonly buffer HeapBlock { uint Data[]; } Heap;
layout(std430,binding=1) readonly buffer MapBlock { uvec4 Data[]; } Map;
layout(std430,binding=2) readonly buffer PoolBlock { uint Data[]; } Pool;
layout(std430,binding=3) buffer ScratchBlock { uint Data[]; } Scratch;
layout(std430,binding=4) writeonly buffer VertexBlock { uint Data[]; } VertexStream;
layout(std430,binding=5) writeonly buffer IndexBlock { uint Data[]; } IndexStream;
const uint Threads=64u,VectorSize=4u,GroupSize=16u,ClusterSize=16u,ViewSize=36u,ScratchSize=4u;
const uint LowBounds=8u,HighBounds=12u,CoefficientErrors=16u,ScreenScale=32u;
const uint ErrorField=0u,CompleteField=1u,RefinementField=2u,RequestField=3u;
const uint PageSize=16384u,SlotSize=PageSize+VectorSize,PageHeader=8u,ClusterHeader=44u,VertexSize=12u;
const uint AttributeBases=8u,AttributeWidths=20u,AttributeOffsets=32u,ClusterLength=6u,IndexBitStart=7u;
const uint UsedSize=2u,PageFlags=4u,PageExponent=5u,PageIdentity=6u,Reserved=7u;
const uint BitCount=32u,PositionSize=3u,MaxCorners=768u,ExponentBias=127u,MantissaSize=23u,SignedMaximum=0x7fffffffu;
const uint Unavailable=0xffffffffu,InfiniteValue=0x7f800000u,PageMagic=0x47504e45u,PageVersion=1u;
const uint StaleMapping=1u,BadEncoding=2u,OutputCapacity=4u,MissingEncoding=8u;
const float FloatEpsilon=1.1920928955078125e-7,FloatMinimumNormal=1.1754943508222875e-38;
const float ArithmeticAllowance=16.0,RelativeError=ArithmeticAllowance*FloatEpsilon,AbsoluteError=ArithmeticAllowance*FloatMinimumNormal;
shared uvec3 Prefix[NANITE_THREADS]; shared uvec3 Running; shared uint DecodeAllowed;
uvec4 H4(uint p) { return uvec4(Heap.Data[p],Heap.Data[p+1u],Heap.Data[p+2u],Heap.Data[p+3u]); }
uvec4 S4(uint p) { return uvec4(Scratch.Data[p],Scratch.Data[p+1u],Scratch.Data[p+2u],Scratch.Data[p+3u]); }
void Save4(uint p,uvec4 value) { Scratch.Data[p]=value.x; Scratch.Data[p+1u]=value.y; Scratch.Data[p+2u]=value.z; Scratch.Data[p+3u]=value.w; }
uint GroupAt(uint id) { return F.Offsets.x+id*GroupSize; }
uint ClusterAt(uint id) { return F.Offsets.w+id*ClusterSize; }
uint ClusterScratch() { return F.Counts.x*ScratchSize; }
uint BlockScratch() { return ClusterScratch()+F.Tables.w*ScratchSize; }
uint Summary() { return BlockScratch()+F.Limits.z*VectorSize; }
uint Workgroup() { return gl_WorkGroupID.y*F.Dispatch.z+gl_WorkGroupID.x; }
void Fault(uint bit) { atomicOr(Scratch.Data[Summary()],bit); }
bool IsFinite(float value) { return !isnan(value) && !isinf(value); }
bool IsResident(uint page)
{
    uvec4 entry=Map.Data[page]; if (entry.x==Unavailable) return false;
    if (entry.x>=F.Counts.w || entry.y==0u || any(notEqual(entry.zw,F.Identity.xy))) { Fault(StaleMapping); return false; }
    uint p=entry.x*SlotSize;
    if (Pool.Data[p]!=page || Pool.Data[p+1u]!=entry.y || Pool.Data[p+2u]!=entry.z || Pool.Data[p+3u]!=entry.w) { Fault(StaleMapping); return false; }
    return true;
}
float Up(float x) { return IsFinite(x) && x>=0.0?x*(1.0+RelativeError)+AbsoluteError:uintBitsToFloat(InfiniteValue); }
float Down(float x) { return x>0.0?x*(1.0-RelativeError)-AbsoluteError:x; }
float PixelBound(uint group,uint view)
{
    uint g=GroupAt(group),v=F.Tables.x+view*ViewSize; vec4 low=uintBitsToFloat(H4(g+LowBounds)),high=uintBitsToFloat(H4(g+HighBounds));
    if (low.w==0.0) return 0.0;
    float nearestW=uintBitsToFloat(InfiniteValue),largestX=0.0,largestY=0.0;
    const uint Corners=8u,Rows=4u;
    for (uint corner=0u;corner<Corners;++corner)
    {
        vec4 p=vec4((corner&1u)!=0u?high.x:low.x,(corner&2u)!=0u?high.y:low.y,(corner&4u)!=0u?high.z:low.z,1.0),clip=vec4(0.0),error=vec4(0.0);
        for (uint row=0u;row<Rows;++row)
        {
            vec4 a=uintBitsToFloat(H4(v+row*VectorSize)),e=uintBitsToFloat(H4(v+CoefficientErrors+row*VectorSize));
            clip[row]=dot(a,p); error[row]=Up(dot(abs(a),abs(p))*RelativeError+dot(e,abs(p)));
        }
        float w=Down(clip.w-error.w),z=Down(clip.z-error.z);
        if (!(w>0.0) || !(z>0.0) || !IsFinite(w)) return uintBitsToFloat(InfiniteValue);
        nearestW=min(nearestW,w); largestX=max(largestX,Up(Up(abs(clip.x)+error.x)/w)); largestY=max(largestY,Up(Up(abs(clip.y)+error.y)/w));
    }
    vec4 displacement=vec4(0.0);
    for (uint row=0u;row<Rows;++row)
    {
        float norm=0.0;
        for (uint axis=0u;axis<PositionSize;++axis) norm+=abs(uintBitsToFloat(Heap.Data[v+row*VectorSize+axis]))+uintBitsToFloat(Heap.Data[v+CoefficientErrors+row*VectorSize+axis]);
        displacement[row]=Up(Up(norm)*low.w);
    }
    float denominator=Down(nearestW-displacement.w); if (!(denominator>0.0)) return uintBitsToFloat(InfiniteValue);
    float x=Up(Up(displacement.x+Up(largestX*displacement.w))/denominator),y=Up(Up(displacement.y+Up(largestY*displacement.w))/denominator);
    return Up(Up(x*uintBitsToFloat(Heap.Data[v+ScreenScale]))+Up(y*uintBitsToFloat(Heap.Data[v+ScreenScale+1u])));
}
void EvaluateGroups()
{
    uint offset=Workgroup()*Threads+gl_LocalInvocationIndex; if (offset>=F.Dispatch.y) return;
    uint id=Heap.Data[F.Tables.y+F.Dispatch.x+offset],g=GroupAt(id); uvec4 links=H4(g),pages=H4(g+VectorSize);
    float error=0.0; for (uint view=0u;view<F.Counts.y;++view) error=max(error,PixelBound(id,view));
    for (uint child=0u;child<links.y;++child)
    { uint producer=H4(ClusterAt(Heap.Data[F.Offsets.y+links.x+child])+VectorSize).z; if (producer!=Unavailable) error=max(error,uintBitsToFloat(Scratch.Data[producer*ScratchSize+ErrorField])); }
    bool complete=true; for (uint page=0u;page<pages.y;++page) { bool available=IsResident(pages.x+page); complete=complete && available; }
    Save4(id*ScratchSize,uvec4(floatBitsToUint(error),complete?1u:0u,0u,error>F.Quality.x?1u:0u));
}
void RefineGroups()
{
    uint offset=Workgroup()*Threads+gl_LocalInvocationIndex; if (offset>=F.Dispatch.y) return;
    uint id=Heap.Data[F.Tables.y+F.Dispatch.x+offset]; uvec4 links=H4(GroupAt(id)); bool ancestors=true;
    for (uint parent=0u;parent<links.w;++parent)
    { uint consumer=H4(ClusterAt(Heap.Data[F.Offsets.z+links.z+parent])+VectorSize).w; if (consumer!=Unavailable && Scratch.Data[consumer*ScratchSize+RefinementField]==0u) ancestors=false; }
    uint state=id*ScratchSize; Scratch.Data[state+RefinementField]=ancestors && Scratch.Data[state+CompleteField]!=0u && Scratch.Data[state+RequestField]!=0u?1u:0u;
}
void PrefixSum(uint lane)
{
    for (uint step=1u;step<Threads;step*=2u)
    { uvec3 add=lane>=step?Prefix[lane-step]:uvec3(0u); barrier(); Prefix[lane]+=add; barrier(); }
}
void ScanClusters()
{
    uint batch=Workgroup(); if (batch>=F.Limits.z) return;
    uint lane=gl_LocalInvocationIndex,id=batch*Threads+lane; bool chosen=false; uvec3 amount=uvec3(0u);
    if (id<F.Tables.w)
    {
        uvec4 sizes=H4(ClusterAt(id)),owners=H4(ClusterAt(id)+VectorSize);
        chosen=(owners.w==Unavailable || Scratch.Data[owners.w*ScratchSize+RefinementField]!=0u) && (owners.z==Unavailable || Scratch.Data[owners.z*ScratchSize+RefinementField]==0u);
        if (chosen) amount=uvec3(sizes.zw,1u);
    }
    Prefix[lane]=amount; barrier(); PrefixSum(lane);
    if (id<F.Tables.w) Save4(ClusterScratch()+id*ScratchSize,uvec4(chosen?1u:0u,Prefix[lane]-amount));
    if (lane==Threads-1u) Save4(BlockScratch()+batch*VectorSize,uvec4(Prefix[lane],0u));
}
void ScanBlocks()
{
    uint lane=gl_LocalInvocationIndex; if (lane==0u) Running=uvec3(0u); barrier();
    for (uint start=0u;start<F.Limits.z;start+=Threads)
    {
        uint id=start+lane; uvec3 amount=id<F.Limits.z?S4(BlockScratch()+id*VectorSize).xyz:uvec3(0u);
        Prefix[lane]=amount; barrier(); PrefixSum(lane); uvec3 prefix=Running+Prefix[lane]-amount;
        if (id<F.Limits.z) Save4(BlockScratch()+id*VectorSize,uvec4(prefix,0u));
        barrier(); if (lane==0u) Running+=Prefix[Threads-1u]; barrier();
    }
    if (lane==0u)
    { if (Running.x>F.Capacity.x || Running.y>F.Capacity.y) Fault(OutputCapacity); Scratch.Data[Summary()+1u]=Running.x; Scratch.Data[Summary()+2u]=Running.y; Scratch.Data[Summary()+3u]=Running.z; }
}
uint Bits(uint payload,uint position,uint width)
{
    if (width==0u) return 0u; uint word=position/BitCount,shift=position%BitCount,value=Pool.Data[payload+word]>>shift;
    if (shift+width>BitCount) value|=Pool.Data[payload+word+1u]<<(BitCount-shift);
    if (width==BitCount) return value; return value&((1u<<width)-1u);
}
bool CheckHeader(uint page,uint header,uvec4 sizes,uvec4 owners)
{
    uint used=Pool.Data[page+UsedSize];
    if (Pool.Data[page]!=PageMagic || Pool.Data[page+1u]!=PageVersion || used>PageSize || used<PageHeader+ClusterHeader ||
        Pool.Data[page+PageFlags]>1u || Pool.Data[page+PageExponent]!=F.Limits.w || Pool.Data[page+PageIdentity]!=sizes.x || Pool.Data[page+Reserved]!=0u ||
        sizes.y>used-ClusterHeader || sizes.z==0u || sizes.z>MaxCorners || sizes.w==0u || sizes.w>MaxCorners) return false;
    if (Pool.Data[header]!=owners.y || Pool.Data[header+1u]!=sizes.z || Pool.Data[header+2u]!=sizes.w || Pool.Data[header+3u]!=owners.x ||
        Pool.Data[header+VectorSize]!=owners.z || Pool.Data[header+VectorSize+1u]!=owners.w ||
        Pool.Data[header+ClusterLength]<ClusterHeader || Pool.Data[header+ClusterLength]>used-sizes.y) return false;
    uint count=0u;
    for (uint channel=0u;channel<VertexSize;++channel)
    { uint width=Pool.Data[header+AttributeWidths+channel]; if (width>BitCount || Pool.Data[header+AttributeOffsets+channel]!=count) return false; count+=sizes.z*width; }
    if (Pool.Data[header+IndexBitStart]!=count) return false;
    uint indexWidth=0u; for (uint value=sizes.z-1u;value!=0u;value>>=1u) ++indexWidth;
    count+=sizes.w*indexWidth; return Pool.Data[header+ClusterLength]==ClusterHeader+(count+BitCount-1u)/BitCount;
}
void DecodeClusters()
{
    uint id=Workgroup(),lane=gl_LocalInvocationIndex; if (id>=F.Tables.w) return;
    uvec4 chosen=S4(ClusterScratch()+id*ScratchSize);
    uvec4 sizes=H4(ClusterAt(id)),owners=H4(ClusterAt(id)+VectorSize),prefix=S4(BlockScratch()+(id/Threads)*VectorSize);
    uint firstVertex=prefix.x+chosen.y,firstIndex=prefix.y+chosen.z;
    if (lane==0u)
    {
        DecodeAllowed=0u;
        // Every lane reaches the barrier, including inactive clusters. Some
        // ES translators cannot prove SSBO-loaded decisions are group uniform.
        if (chosen.x!=0u)
        {
            if (!IsResident(sizes.x)) Fault(MissingEncoding);
            else
            {
                uint page=Map.Data[sizes.x].x*SlotSize+VectorSize;
                if (!CheckHeader(page,page+sizes.y,sizes,owners)) Fault(BadEncoding);
                else if (firstVertex>F.Capacity.x || sizes.z>F.Capacity.x-firstVertex || firstIndex>F.Capacity.y || sizes.w>F.Capacity.y-firstIndex) Fault(OutputCapacity);
                else DecodeAllowed=1u;
            }
        }
    }
    barrier(); if (DecodeAllowed==0u) return;
    uint header=Map.Data[sizes.x].x*SlotSize+VectorSize+sizes.y,payload=header+ClusterHeader;
    float scale=uintBitsToFloat(uint(int(F.Limits.w)+int(ExponentBias))<<MantissaSize);
    for (uint vertex=lane;vertex<sizes.z;vertex+=Threads) for (uint channel=0u;channel<VertexSize;++channel)
    {
        uint base=Pool.Data[header+AttributeBases+channel],width=Pool.Data[header+AttributeWidths+channel];
        uint delta=Bits(payload,Pool.Data[header+AttributeOffsets+channel]+vertex*width,width),decoded=base^delta;
        if (channel<PositionSize) { if (delta>SignedMaximum-base) Fault(BadEncoding); decoded=floatBitsToUint(float(int(base+delta))*scale); }
        if (!IsFinite(uintBitsToFloat(decoded))) Fault(BadEncoding); VertexStream.Data[(firstVertex+vertex)*VertexSize+channel]=decoded;
    }
    uint indexWidth=0u; for (uint value=sizes.z-1u;value!=0u;value>>=1u) ++indexWidth;
    for (uint index=lane;index<sizes.w;index+=Threads)
    { uint local=Bits(payload,Pool.Data[header+IndexBitStart]+index*indexWidth,indexWidth); if (local>=sizes.z) Fault(BadEncoding); else IndexStream.Data[firstIndex+index]=firstVertex+local; }
}
void main()
{
#if EVANS_NANITE_GEOMETRY_PASS == PASS_EVALUATION
    EvaluateGroups();
#elif EVANS_NANITE_GEOMETRY_PASS == PASS_REFINEMENT
    RefineGroups();
#elif EVANS_NANITE_GEOMETRY_PASS == PASS_CLUSTER_SCAN
    ScanClusters();
#elif EVANS_NANITE_GEOMETRY_PASS == PASS_BLOCK_SCAN
    ScanBlocks();
#elif EVANS_NANITE_GEOMETRY_PASS == PASS_DECODING
    DecodeClusters();
#endif
}
#endif
