// Two-stage Nanite occlusion (Karis, Stubbe, Wihlidal, SIGGRAPH 2021).
// Root SRV/UAV buffers hold depth and bounded queries; each mip has a UAV barrier.
cbuffer Parameters : register(b0) { uint4 Counts,Layout,Source,Target; };
RWStructuredBuffer<uint> ResultWords : register(u0);
static const uint WorkgroupSize=64,CandidateStride=44,CurrentTransform=8,PreviousTransform=24,Metadata=40;
static const uint OutsideFrustum=1,DrawMain=2,Retest=3,DrawPost=4,RejectOccluded=5,FootprintLimit=4;
static const float ErrorFactor=32.0*1.1920928955078125e-7,ErrorFloor=32.0*1.1754943508222875e-38;
#if defined(EVANS_NANITE_CAPTURE_DEPTH)
Texture2D<float> NativeDepth : register(t0);
[numthreads(WorkgroupSize,1,1)]
void CSCapture(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint pixel=(group.y*Target.w+group.x)*WorkgroupSize+lane; [branch] if (pixel>=Target.y*Target.z) return;
    ResultWords[pixel]=asuint(NativeDepth.Load(int3(pixel%Target.y,pixel/Target.y,0)));
}
#else
StructuredBuffer<uint> SourceWords : register(t0);
#if defined(EVANS_NANITE_FRAME_CULL)
StructuredBuffer<uint> NativeHierarchy : register(t1);
StructuredBuffer<uint> MainStates : register(t2);
#endif
float4 LoadFloat4(uint address)
{ return asfloat(uint4(SourceWords[address],SourceWords[address+1],SourceWords[address+2],SourceWords[address+3])); }
struct BoundsProjection { float2 Minimum,Maximum; float Depth; bool Outside,Usable; };
BoundsProjection ProjectBounds(uint base,uint transform)
{
    BoundsProjection result; result.Minimum=1; result.Maximum=0; result.Depth=1; result.Outside=false; result.Usable=true;
    uint excludedPlanes=63; bool finite=true;
    float3 a=LoadFloat4(base).xyz,b=LoadFloat4(base+4).xyz;
    [unroll] for (uint corner=0;corner<8;++corner)
    {
        float4 pointValue=float4((corner&1)?b.x:a.x,(corner&2)?b.y:a.y,(corner&4)?b.z:a.z,1);
        float4 lower=0,upper=0;
        [unroll] for (uint row=0;row<4;++row)
        {
            float4 coefficients=LoadFloat4(base+transform+row*4);
            float sum=dot(coefficients,pointValue),magnitude=dot(abs(coefficients),abs(pointValue));
            float error=magnitude*ErrorFactor+ErrorFloor; lower[row]=sum-error; upper[row]=sum+error;
            finite=finite && isfinite(magnitude) && isfinite(lower[row]) && isfinite(upper[row]);
        }
        uint planes=0;
        if (upper.x+upper.w<0) planes|=1; if (upper.w-lower.x<0) planes|=2;
        if (upper.y+upper.w<0) planes|=4; if (upper.w-lower.y<0) planes|=8;
        if (upper.z<0) planes|=16; if (upper.w-lower.z<0) planes|=32; excludedPlanes&=planes;
        if (lower.w<=0 || lower.z<=0) { result.Usable=false; continue; }
        float3 lo=min(lower.xyz/lower.w,lower.xyz/upper.w),hi=max(upper.xyz/lower.w,upper.xyz/upper.w);
        float3 error=(max(abs(lo),abs(hi))+1)*ErrorFactor; lo-=error; hi+=error;
        result.Minimum=min(result.Minimum,float2(lo.x+1,1-hi.y)*0.5);
        result.Maximum=max(result.Maximum,float2(hi.x+1,1-lo.y)*0.5); result.Depth=min(result.Depth,lo.z);
    }
    result.Outside=finite && excludedPlanes!=0;
    result.Usable=result.Usable && finite && all(result.Minimum>=0) && all(result.Maximum<=1) && result.Depth>0;
    return result;
}
bool IsOccluded(BoundsProjection box)
{
    if (!box.Usable) return false;
    uint2 size=Counts.yz,lo=(uint2)floor(box.Minimum*size),hi=min(size-1,(uint2)floor(box.Maximum*size)); uint start=0;
    while (any(hi-lo+1>FootprintLimit)) { start+=size.x*size.y; size=(size+1)/2; lo/=2; hi/=2; }
    for (uint y=lo.y;y<=hi.y;++y) for (uint x=lo.x;x<=hi.x;++x)
    {
#if defined(EVANS_NANITE_FRAME_CULL)
        float depth=asfloat(NativeHierarchy[start+y*size.x+x]);
#else
        float depth=asfloat(ResultWords[start+y*size.x+x]);
#endif
        if (!(depth<box.Depth)) return false;
    }
    return true;
}
[numthreads(WorkgroupSize,1,1)]
void CSReduce(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint pixel=(group.y*Target.w+group.x)*WorkgroupSize+lane; [branch] if (pixel>=Target.y*Target.z) return;
    float depth=0;
    [branch] if (Source.w==0) depth=asfloat(SourceWords[pixel]);
    else
    {
        uint2 origin=uint2(pixel%Target.y,pixel/Target.y)*2;
        [unroll] for (uint y=0;y<2;++y) [unroll] for (uint x=0;x<2;++x)
        {
            uint2 child=origin+uint2(x,y);
            // Root descriptors have no bounds checking; keep loads inside branches.
            [branch] if (child.x<Source.y && child.y<Source.z) depth=max(depth,asfloat(ResultWords[Source.x+child.y*Source.y+child.x]));
        }
    }
    ResultWords[Target.x+pixel]=asuint(depth);
}
[numthreads(WorkgroupSize,1,1)]
void CSCull(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    uint index=(group.y*Target.w+group.x)*WorkgroupSize+lane; [branch] if (index>=Counts.x) return;
    uint base=Layout.x+index*CandidateStride,state=DrawMain;
    [branch] if (Counts.w==0)
    {
        if (ProjectBounds(base,CurrentTransform).Outside) state=OutsideFrustum;
        else
        {
            bool history=SourceWords[base+Metadata]!=0;
#if defined(EVANS_NANITE_FRAME_CULL)
            history=history && Layout.w!=0;
#endif
            [branch] if (history) { if (IsOccluded(ProjectBounds(base,PreviousTransform))) state=Retest; }
        }
    }
    else
    {
#if defined(EVANS_NANITE_FRAME_CULL)
        state=MainStates[index];
#else
        state=SourceWords[base+Metadata+1];
#endif
        if (state==Retest) state=IsOccluded(ProjectBounds(base,CurrentTransform))?RejectOccluded:DrawPost;
    }
    ResultWords[Layout.y+index]=state;
}
#endif
