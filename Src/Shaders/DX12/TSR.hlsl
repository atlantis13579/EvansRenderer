#if defined(EVANS_TSR_PACK_STAGE)
// The image bridge has its own descriptor-table root layout. It does not add
// descriptors to the reconstruction root block; both use bounded buffer views.
cbuffer PackFrame : register(b0) { uint4 ImageExtentFlags; };
Texture2D<float4> SceneRadiance : register(t0);
Texture2D<float4> OpaqueRadiance : register(t1);
Texture2D<float4> RigidVelocity : register(t2);
Texture2D<float> SceneDepth : register(t3);
Texture2D<float4> CurrentOutline : register(t4);
Texture2D<float> ExtraReactivity : register(t5);
RWStructuredBuffer<float4> RenderPacket : register(u0);
static const uint RecordWidth=5,FinalSlot=0,OpaqueSlot=1,VelocitySlot=2,DepthSlot=3,OverlaySlot=4;
static const uint MotionAvailable=1,OutlineAvailable=2,ReactiveAvailable=4,BridgeGroupWidth=8;
[numthreads(BridgeGroupWidth,BridgeGroupWidth,1)]
void CSPack(uint3 dispatch : SV_DispatchThreadID)
{
    if (any(dispatch.xy>=ImageExtentFlags.xy)) return;
    uint first=(dispatch.y*ImageExtentFlags.x+dispatch.x)*RecordWidth;
    int3 texel=int3(dispatch.xy,0); float4 velocity=0,overlay=0; float reactive=0;
    [branch] if ((ImageExtentFlags.z&MotionAvailable)!=0) velocity=RigidVelocity.Load(texel);
    [branch] if ((ImageExtentFlags.z&OutlineAvailable)!=0) overlay=CurrentOutline.Load(texel);
    [branch] if ((ImageExtentFlags.z&ReactiveAvailable)!=0) reactive=ExtraReactivity.Load(texel);
    RenderPacket[first+FinalSlot]=SceneRadiance.Load(texel);
    RenderPacket[first+OpaqueSlot]=OpaqueRadiance.Load(texel);
    RenderPacket[first+VelocitySlot]=velocity;
    RenderPacket[first+DepthSlot]=float4(SceneDepth.Load(texel),reactive,0,0);
    RenderPacket[first+OverlaySlot]=overlay;
}
#elif defined(EVANS_TSR_EXPORT_STAGE)
cbuffer PresentationFrame : register(b0) { uint4 ImageExtentFlags; };
StructuredBuffer<float4> UpscaledRecords : register(t0);
static const uint PresentationRecordWidth=2;
float4 VSExport(uint vertex : SV_VertexID) : SV_Position
{
    const float TriangleReach=3;
    float2 corner=vertex==0?float2(-1,-1):vertex==1?float2(-1,TriangleReach):float2(TriangleReach,-1);
    return float4(corner,0,1);
}
float4 PSExport(float4 position : SV_Position) : SV_Target0
{
    // Bound helper-lane reads at odd raster edges as well as the native SRV.
    uint2 at=min(uint2(position.xy),ImageExtentFlags.xy-1);
    return UpscaledRecords[(at.y*ImageExtentFlags.x+at.x)*PresentationRecordWidth];
}
#else
// Native D3D12 HDR temporal upscaling. Bounded structured-buffer descriptors
// complement explicit dispatch and history guards.
// Lanczos-2 spatial reconstruction, optical-depth rejection, YCoCg variance
// clipping and color-derived transparency reactivity are implemented here.
// Salvi, "An Excursion in Temporal Supersampling", GDC 2016.
// AMD, "Temporal Upscaling: Past, Present, and Future", GDC 2023.
cbuffer UpscaleFrame : register(b0)
{
    row_major float4x4 CurrentInverse,PreviousInverse;
    float4 CurrentDepthPlane,PreviousDepthPlane;
    uint4 Extent;
    float4 JitterQuality;
    float4 Controls;
    uint4 Options;
};
StructuredBuffer<float4> SourcePixels : register(t0);
StructuredBuffer<float4> PresentedHistory : register(t1);
RWStructuredBuffer<float4> CandidateHistory : register(u0);
RWStructuredBuffer<float4> CurrentAuxiliary : register(u1);
static const uint SourceStride=5,HistoryStride=2,GroupSide=8;
static const int ReconstructionRadius=2,MotionRadius=1;
static const float HalfPixel=0.5,FarDepth=1,Pi=3.14159265358979323846;
static const float MinimumW=1e-7,DepthFloor=1e-4,ColorFloor=1e-4,WeightFloor=1e-6,HalfFloatLimit=65504;
static const float UnsupportedObservation=0.001;
static const float NeighborhoodSamples=(ReconstructionRadius*2)*(ReconstructionRadius*2);
float4 Source(int2 p,uint member)
{
    p=clamp(p,int2(0,0),int2(Extent.xy)-1);
    return SourcePixels[(uint(p.y)*Extent.x+uint(p.x))*SourceStride+member];
}
float4 Past(int2 p,uint member)
{
    p=clamp(p,int2(0,0),int2(Extent.zw)-1);
    return PresentedHistory[(uint(p.y)*Extent.z+uint(p.x))*HistoryStride+member];
}
float MaxComponent(float3 c) { return max(c.x,max(c.y,c.z)); }
float3 ToLumaChroma(float3 c) { return float3((c.r+2*c.g+c.b)/4,(c.r-c.b)/2,(-c.r+2*c.g-c.b)/4); }
float3 ToRGB(float3 c) { return float3(c.x+c.y-c.z,c.x+c.z,c.x-c.y-c.z); }
float Kernel(float distance)
{
    float x=abs(distance); if (x>=ReconstructionRadius) return 0; if (x<WeightFloor) return 1;
    return sin(Pi*x)*sin(Pi*x/ReconstructionRadius)/(Pi*Pi*x*x/ReconstructionRadius);
}
float Optical(float2 uv,float depth,bool previous)
{
    float4 clip=float4(uv*float2(2,-2)+float2(-1,1),depth,1);
    float4 world=previous?mul(PreviousInverse,clip):mul(CurrentInverse,clip);
    if (abs(world.w)<=MinimumW) return -1;
    return dot(previous?PreviousDepthPlane:CurrentDepthPlane,float4(world.xyz/world.w,1));
}
int2 FloorLocalSample(float2 local)
{
    // Only subpixel fractions are snapped; large pixel coordinates never lose
    // their integer part. The threshold covers float division roundoff.
    const float BoundaryRoundoff=1e-6;
    float2 integral=round(local);
    float2 stable=float2(abs(local.x-integral.x)<=BoundaryRoundoff?integral.x:local.x,
        abs(local.y-integral.y)<=BoundaryRoundoff?integral.y:local.y);
    return int2(floor(stable));
}
[numthreads(GroupSide,GroupSide,1)]
void CSMain(uint3 invocation:SV_DispatchThreadID)
{
    [branch] if (invocation.x>=Extent.z || invocation.y>=Extent.w) return;
    uint2 divisor=Extent.zw*2;
    uint2 rationalCenter=(invocation.xy*2+1)*Extent.xy,cell=rationalCenter/divisor;
    float2 local=float2(rationalCenter%divisor)/float2(divisor)+JitterQuality.xy;
    float2 renderPoint=float2(cell)+local;
    int2 lower=int2(cell)+FloorLocalSample(local-HalfPixel);
    float2 footprint=float2(Extent.xy)/float2(Extent.zw);
    float3 total=0,wideTotal=0,mean=0,secondMoment=0,lo=HalfFloatLimit,hi=0; float alpha=0,totalWeight=0,wideWeight=0,reactivity=0;
    [loop] for (int sampleY=1-ReconstructionRadius;sampleY<=ReconstructionRadius;++sampleY) [loop] for (int sampleX=1-ReconstructionRadius;sampleX<=ReconstructionRadius;++sampleX)
    {
        int2 p=lower+int2(sampleX,sampleY); float4 finalColor=Source(p,0),opaque=Source(p,1);
        float3 rgb=clamp(finalColor.rgb,0,HalfFloatLimit),yc=ToLumaChroma(rgb);
        float2 offset=float2(p)+HalfPixel-renderPoint;
        float w=Kernel(offset.x/footprint.x)*Kernel(offset.y/footprint.y),wide=Kernel(offset.x)*Kernel(offset.y);
        total+=rgb*w; wideTotal+=rgb*wide; alpha+=finalColor.a*wide; totalWeight+=w; wideWeight+=wide; mean+=yc; secondMoment+=yc*yc; lo=min(lo,rgb); hi=max(hi,rgb);
        float normalization=max(max(MaxComponent(rgb),MaxComponent(opaque.rgb)),ColorFloor);
        reactivity=max(reactivity,max(Source(p,3).y,MaxComponent(abs(finalColor.rgb-opaque.rgb))/normalization*Controls.y));
    }
    bool supported=totalWeight>WeightFloor;
    float3 fresh;
    [branch] if (supported) fresh=total/totalWeight; else fresh=wideTotal/max(wideWeight,WeightFloor);
    fresh=clamp(fresh,lo,hi); float3 resolved=fresh;
    alpha=saturate(alpha/max(wideWeight,WeightFloor)); reactivity=saturate(reactivity);
    int2 nearest=int2(cell)+FloorLocalSample(local),chosen=nearest; float depth=Source(chosen,3).x;
    [loop] for (int motionY=-MotionRadius;motionY<=MotionRadius;++motionY) [loop] for (int motionX=-MotionRadius;motionX<=MotionRadius;++motionX)
    {
        int2 p=nearest+int2(motionX,motionY); float candidate=Source(p,3).x;
        if (candidate<depth) { depth=candidate; chosen=p; }
    }
    chosen=clamp(chosen,int2(0,0),int2(Extent.xy)-1);
    float4 velocity=Source(chosen,2); bool sky=depth>=FarDepth;
    float2 motion=sky?float2(0,0):velocity.xy;
    float2 oldPoint=float2(invocation.xy)+HalfPixel-motion*float2(Extent.zw)/float2(Extent.xy);
    float2 residual=(float2(nearest)+HalfPixel-renderPoint)/footprint; float observation=supported?exp2(-dot(residual,residual)):UnsupportedObservation,sampleCount=observation;
    bool useHistory=Options.x!=0 && reactivity<1 && (sky?Options.y!=0:velocity.w>0) && all(oldPoint>=HalfPixel) && all(oldPoint<=float2(Extent.zw)-HalfPixel);
    [branch] if (useHistory)
    {
        float expected=Optical((float2(chosen)+HalfPixel-JitterQuality.xy-motion)/float2(Extent.xy),sky?FarDepth:1-velocity.z,true);
        int2 first=int2(floor(oldPoint-HalfPixel)); bool matches=false;
        [loop] for (int depthY=0;depthY<2;++depthY) [loop] for (int depthX=0;depthX<2;++depthX)
        {
            float4 meta=Past(first+int2(depthX,depthY),1);
            matches=matches || (sky?meta.x>=FarDepth:meta.x<FarDepth && expected>0 && meta.z>0 && abs(expected-meta.z)<=max(DepthFloor,abs(expected)*JitterQuality.w));
        }
        [branch] if (matches)
        {
            float2 f=oldPoint-HalfPixel-float2(first); float3 oldColor=0; float oldCount=0;
            [loop] for (int historyY=0;historyY<2;++historyY) [loop] for (int historyX=0;historyX<2;++historyX)
            {
                float w=(historyX?f.x:1-f.x)*(historyY?f.y:1-f.y);
                oldColor+=Past(first+int2(historyX,historyY),0).rgb*w; oldCount+=Past(first+int2(historyX,historyY),1).y*w;
            }
            mean/=NeighborhoodSamples; secondMoment/=NeighborhoodSamples;
            float3 radius=max(sqrt(max(secondMoment-mean*mean,0))*Controls.x,ColorFloor);
            float3 offset=ToLumaChroma(oldColor)-mean;
            oldColor=ToRGB(mean+offset/max(1,MaxComponent(abs(offset/radius)))); oldCount=min(oldCount,Controls.z);
            float blend=min(JitterQuality.z,oldCount/(oldCount+observation))*(1-reactivity);
            resolved=lerp(fresh,oldColor,blend); sampleCount=min(Controls.z,oldCount*(1-reactivity)+observation);
        }
    }
    uint address=(invocation.y*Extent.z+invocation.x)*HistoryStride;
    float optical=sky?-1:Optical((float2(chosen)+HalfPixel-JitterQuality.xy)/float2(Extent.xy),depth,false);
    CandidateHistory[address]=float4(clamp(resolved,0,HalfFloatLimit),alpha);
    CandidateHistory[address+1]=float4(depth,sampleCount,optical,reactivity);
    CurrentAuxiliary[address]=float4(velocity.xy*float2(Extent.zw)/float2(Extent.xy),velocity.w>0?1-velocity.z:0,velocity.w);
    CurrentAuxiliary[address+1]=Source(nearest,4);
}
#endif
