#if defined(EVANS_TSR_PACK_STAGE)
// Point-load resolved scene images into the canonical top-left packet. Absent
// motion/outline inputs are disabled explicitly, never read through null views.
cbuffer TextureFrame : register(b0) { uint4 ExtentFlags; };
Texture2D<float4> FinalHDR : register(t0);
Texture2D<float4> OpaqueHDR : register(t1);
Texture2D<float4> PhysicalMotion : register(t2);
Texture2D<float> HardwareDepth : register(t3);
Texture2D<float4> OutlineMask : register(t4);
Texture2D<float> AdditionalReactive : register(t5);
RWStructuredBuffer<float4> PackedInput : register(u0);
static const uint PacketStride=5,ColorField=0,OpaqueField=1,MotionField=2,DepthField=3,OutlineField=4;
static const uint HasMotion=1,HasOutline=2,HasReactive=4,PackGroupSide=8;
[numthreads(PackGroupSide,PackGroupSide,1)]
void CSPack(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy>=ExtentFlags.xy)) return;
    int3 at=int3(id.xy,0); uint packet=(id.y*ExtentFlags.x+id.x)*PacketStride;
    PackedInput[packet+ColorField]=FinalHDR.Load(at);
    PackedInput[packet+OpaqueField]=OpaqueHDR.Load(at);
    float4 motion=0,outline=0; float reactive=0;
    [branch] if ((ExtentFlags.z&HasMotion)!=0) motion=PhysicalMotion.Load(at);
    [branch] if ((ExtentFlags.z&HasOutline)!=0) outline=OutlineMask.Load(at);
    [branch] if ((ExtentFlags.z&HasReactive)!=0) reactive=AdditionalReactive.Load(at);
    PackedInput[packet+MotionField]=motion;
    PackedInput[packet+DepthField]=float4(HardwareDepth.Load(at),reactive,0,0);
    PackedInput[packet+OutlineField]=outline;
}
#elif defined(EVANS_TSR_EXPORT_STAGE)
cbuffer ExportFrame : register(b0) { uint4 ExtentFlags; };
StructuredBuffer<float4> ReconstructedHDR : register(t0);
static const uint HistoryStride=2;
float4 VSExport(uint id : SV_VertexID) : SV_Position
{
    const float FullscreenExtent=3;
    return float4(id==0?float2(-1,-1):id==1?float2(-1,FullscreenExtent):float2(FullscreenExtent,-1),0,1);
}
float4 PSExport(float4 position : SV_Position) : SV_Target0
{
    uint2 pixel=uint2(position.xy);
    return ReconstructedHDR[(pixel.y*ExtentFlags.x+pixel.x)*HistoryStride];
}
#else
// Independent HDR temporal reconstruction. Render-pixel jitter locates samples
// on the presentation grid; physical motion reprojects that unjittered history.
// Salvi, "An Excursion in Temporal Supersampling", GDC 2016 (variance clipping).
// AMD, "Temporal Upscaling: Past, Present, and Future", GDC 2023 (input contracts).
// This is neither Unreal TSR nor a vendor SDK implementation.
cbuffer ReconstructionFrame : register(b0)
{
    row_major float4x4 CurrentInverse, PreviousInverse;
    float4 CurrentDepthPlane, PreviousDepthPlane;
    uint4 Extent; // render XY, presentation ZW
    float4 JitterQuality; // render jitter XY, maximum history weight, relative depth tolerance
    float4 Controls; // variance gamma, reactive scale, sample-weight limit, reserved
    uint4 Options; // valid history, static background, reserved XY
};
StructuredBuffer<float4> Input : register(t0);
StructuredBuffer<float4> Previous : register(t1);
RWStructuredBuffer<float4> History : register(u0);
RWStructuredBuffer<float4> Auxiliary : register(u1);

static const uint InputStride=5, HistoryStride=2, WorkgroupSide=8;
static const int FilterRadius=2, DilationRadius=1;
static const float PixelCenter=0.5, ClearDepth=1, Pi=3.14159265358979323846;
static const float HomogeneousEpsilon=1e-7, AbsoluteDepthTolerance=1e-4;
static const float ColorExtentFloor=1e-4, NormalizationFloor=1e-6, MaximumRadiance=65504;
static const float UnsupportedObservation=0.001;
static const float FilterSamples=(FilterRadius*2)*(FilterRadius*2);
float4 ReadInput(int2 p,uint field)
{
    p=clamp(p,int2(0,0),int2(Extent.xy)-1);
    return Input[(uint(p.y)*Extent.x+uint(p.x))*InputStride+field];
}
float4 ReadPrevious(int2 p,uint field)
{
    p=clamp(p,int2(0,0),int2(Extent.zw)-1);
    return Previous[(uint(p.y)*Extent.z+uint(p.x))*HistoryStride+field];
}
float3 ToYCoCg(float3 c) { return float3((c.r+2*c.g+c.b)/4,(c.r-c.b)/2,(-c.r+2*c.g-c.b)/4); }
float3 FromYCoCg(float3 c) { return float3(c.x+c.y-c.z,c.x+c.z,c.x-c.y-c.z); }
float Largest(float3 c) { return max(c.x,max(c.y,c.z)); }
float Lanczos(float x)
{
    x=abs(x); if (x>=FilterRadius) return 0; if (x<NormalizationFloor) return 1;
    return sin(Pi*x)*sin(Pi*x/FilterRadius)/(Pi*Pi*x*x/FilterRadius);
}
float OpticalDistance(float2 uv,float depth,bool old)
{
    float4 p=float4(uv*float2(2,-2)+float2(-1,1),depth,1);
    float4 world=old?mul(PreviousInverse,p):mul(CurrentInverse,p);
    if (abs(world.w)<=HomogeneousEpsilon) return -1;
    return dot(old?PreviousDepthPlane:CurrentDepthPlane,float4(world.xyz/world.w,1));
}
// Divide the integer pixel-center numerator before converting its remainder.
// Local fractions avoid a reciprocal rounding an exact input boundary downward.
int2 StableGridFloor(float2 fraction)
{
    const float GridBoundaryTolerance=1e-6;
    float2 nearest=round(fraction);
    return int2(floor(float2(abs(fraction.x-nearest.x)<=GridBoundaryTolerance?nearest.x:fraction.x,
        abs(fraction.y-nearest.y)<=GridBoundaryTolerance?nearest.y:fraction.y)));
}
[numthreads(WorkgroupSide,WorkgroupSide,1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x>=Extent.z || id.y>=Extent.w) return;
    uint2 denominator=Extent.zw*2,numerator=(id.xy*2+1)*Extent.xy,whole=numerator/denominator;
    float2 fraction=float2(numerator%denominator)/float2(denominator)+JitterQuality.xy;
    float2 source=float2(whole)+fraction;
    int2 base=int2(whole)+StableGridFloor(fraction-PixelCenter);
    float2 footprint=float2(Extent.xy)/float2(Extent.zw);
    float3 sum=0,wideSum=0,mean=0,square=0,minimum=MaximumRadiance,maximum=0;
    float alpha=0,weight=0,wideWeight=0,reactive=0;
    [loop] for (int filterY=1-FilterRadius;filterY<=FilterRadius;++filterY) [loop] for (int filterX=1-FilterRadius;filterX<=FilterRadius;++filterX)
    {
        int2 at=base+int2(filterX,filterY); float4 raw=ReadInput(at,0),opaque=ReadInput(at,1);
        float3 rgb=clamp(raw.rgb,0,MaximumRadiance),color=ToYCoCg(rgb);
        float2 offset=float2(at)+PixelCenter-source;
        float w=Lanczos(offset.x/footprint.x)*Lanczos(offset.y/footprint.y),wide=Lanczos(offset.x)*Lanczos(offset.y);
        sum+=rgb*w; wideSum+=rgb*wide; alpha+=raw.a*wide; weight+=w; wideWeight+=wide; mean+=color; square+=color*color;
        minimum=min(minimum,rgb); maximum=max(maximum,rgb);
        float scale=max(max(Largest(rgb),Largest(opaque.rgb)),ColorExtentFloor);
        reactive=max(reactive,max(ReadInput(at,3).y,Largest(abs(raw.rgb-opaque.rgb))/scale*Controls.y));
    }
    bool supported=weight>NormalizationFloor;
    float3 current;
    [branch] if (supported) current=sum/weight; else current=wideSum/max(wideWeight,NormalizationFloor);
    current=clamp(current,minimum,maximum);
    alpha=saturate(alpha/max(wideWeight,NormalizationFloor)); reactive=saturate(reactive);
    int2 center=int2(whole)+StableGridFloor(fraction),selected=center; float depth=ReadInput(selected,3).x;
    [loop] for (int dilateY=-DilationRadius;dilateY<=DilationRadius;++dilateY) [loop] for (int dilateX=-DilationRadius;dilateX<=DilationRadius;++dilateX)
    {
        int2 at=center+int2(dilateX,dilateY); float candidate=ReadInput(at,3).x;
        if (candidate<depth) { depth=candidate; selected=at; }
    }
    selected=clamp(selected,int2(0,0),int2(Extent.xy)-1);
    float4 motion=ReadInput(selected,2); bool background=depth>=ClearDepth;
    float2 displacement=background?float2(0,0):motion.xy;
    float2 previousPixel=float2(id.xy)+PixelCenter-displacement*float2(Extent.zw)/float2(Extent.xy);
    float2 residual=(float2(center)+PixelCenter-source)/footprint;
    float confidence=supported?exp2(-dot(residual,residual)):UnsupportedObservation,count=confidence;
    float3 color=current;
    bool accepted=Options.x!=0 && reactive<1 && (background?Options.y!=0:motion.w>0) &&
        all(previousPixel>=PixelCenter) && all(previousPixel<=float2(Extent.zw)-PixelCenter);
    [branch] if (accepted)
    {
        float2 previousUV=(float2(selected)+PixelCenter-JitterQuality.xy-displacement)/float2(Extent.xy);
        float expected=OpticalDistance(previousUV,background?ClearDepth:1-motion.z,true);
        int2 historyBase=int2(floor(previousPixel-PixelCenter)); bool agrees=false;
        [loop] for (int depthY=0;depthY<2;++depthY) [loop] for (int depthX=0;depthX<2;++depthX)
        {
            float4 meta=ReadPrevious(historyBase+int2(depthX,depthY),1);
            agrees=agrees || (background?meta.x>=ClearDepth:meta.x<ClearDepth && expected>0 && meta.z>0 &&
                abs(meta.z-expected)<=max(AbsoluteDepthTolerance,abs(expected)*JitterQuality.w));
        }
        [branch] if (agrees)
        {
            float2 fraction=previousPixel-PixelCenter-float2(historyBase); float3 history=0; float samples=0;
            [loop] for (int historyY=0;historyY<2;++historyY) [loop] for (int historyX=0;historyX<2;++historyX)
            {
                float w=(historyX?fraction.x:1-fraction.x)*(historyY?fraction.y:1-fraction.y);
                history+=ReadPrevious(historyBase+int2(historyX,historyY),0).rgb*w; samples+=ReadPrevious(historyBase+int2(historyX,historyY),1).y*w;
            }
            mean/=FilterSamples; square/=FilterSamples;
            float3 extent=max(sqrt(max(square-mean*mean,0))*Controls.x,ColorExtentFloor);
            float3 delta=ToYCoCg(history)-mean; float clip=max(1,Largest(abs(delta/extent)));
            history=FromYCoCg(mean+delta/clip); samples=min(samples,Controls.z);
            float feedback=min(JitterQuality.z,samples/(samples+confidence))*(1-reactive);
            color=lerp(current,history,feedback); count=min(Controls.z,samples*(1-reactive)+confidence);
        }
    }
    uint output=(id.y*Extent.z+id.x)*HistoryStride;
    float distance=background?-1:OpticalDistance((float2(selected)+PixelCenter-JitterQuality.xy)/float2(Extent.xy),depth,false);
    History[output]=float4(clamp(color,0,MaximumRadiance),alpha);
    History[output+1]=float4(depth,count,distance,reactive);
    Auxiliary[output]=float4(motion.xy*float2(Extent.zw)/float2(Extent.xy),motion.w>0?1-motion.z:0,motion.w);
    Auxiliary[output+1]=ReadInput(center,4);
}
#endif
