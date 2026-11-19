#include <metal_stdlib>
using namespace metal;
// Independent MSL fiber scattering and oriented ribbons. Packed stream = 72
// bytes; row arrays retain the CPU matrix convention without shader translation.
// https://pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair
constant float Pi=3.141592653589793,FullTurn=2*Pi;
constant float DirectionTolerance=1e-12,ReflectanceMinimum=1e-4,TransmissionMinimum=1e-6;
constant float MaximumRadiance=65504,DistanceSquaredMinimum=1e-6,BesselThreshold=12;
constant float BesselFirst=1.0f/8,BesselSecond=9.0f/128,BesselThird=225.0f/3072;
constant float ParallelLimit=0.9,OrthographicThreshold=0.5;
constant uint BesselTerms=20,ScatteringOrders=4,DirectionalKind=0,RectangleKind=2;
struct FiberFrame
{
    float4 ClipFromWorld[4];
    float4 Eye,Forward,Variance,SinTilt,CosTilt,Optics;
    float4 RootTint,TipTint,Ambient,ClipPlane;
    uint4 Options;
    float4 ShadowDepth;
    float4 PreviousClipFromWorld[4];
    float4 PreviousEye,PreviousForward,MotionViewport;
    uint4 MotionOptions;
    float4 ShadowRows[4],ShadowCascadeRows[16];
    float4 ShadowFar,ShadowBlend,ShadowEyePlane,ShadowCascadeParams,ShadowFiltering;
    float4 ShadowSoftMetrics[5],ShadowEmitter;
    uint4 ShadowSoftControl;
    float4 ShadowVarianceEncoding[5],ShadowVarianceParams;
    uint4 ShadowVarianceControl;
    float4 ShadowExtents[5],ShadowBias;
    uint4 ShadowControl;
    float4 ShadowComposition;
    uint4 ContactControl;
    float4 Animation;
};
struct HairVertexData
{
    packed_float3 Center,Tangent;
    packed_float4 Reflectance,Emission,Ribbon;
};
struct FiberLight { float4 PositionKind,DirectionRange,Radiance; };
struct FiberFragment
{
    float4 Position [[position,invariant]];
    float3 World,Tangent,Reflectance,Emission;
    // Covered-sample interpolation keeps thin ribbons away from false grazing.
    float3 CrossHeight [[centroid_perspective]];
    float LinearShadowDepth;
    float4 PreviousClip;
};
float3 Unit(float3 value,float3 fallback)
{ float squared=dot(value,value); return squared>DirectionTolerance && isfinite(squared) ? value*rsqrt(squared):fallback; }
float3 TowardEye(float3 world,constant FiberFrame& frame)
{ return frame.Eye.w>OrthographicThreshold ? -frame.Forward.xyz:Unit(frame.Eye.xyz-world,-frame.Forward.xyz); }
float3 RibbonSide(float3 tangent,float3 view)
{
    float3 side=cross(tangent,view);
    if (dot(side,side)<=DirectionTolerance) side=cross(tangent,abs(tangent.y)<ParallelLimit ? float3(0,1,0):float3(1,0,0));
    return Unit(side,float3(1,0,0));
}
// Roots remain attached; bending increases quadratically toward each tip.
float3 WindOffset(float3 center,float height,float time,float4 wind)
{
    float phase=time*2.2+dot(center,float3(2.0,1.0,3.0));
    return float3(sin(phase),0.18*sin(phase*0.7),0.65*cos(phase*1.13))*wind.z*height*height;
}
float3 WindAxis(float3 center,float3 tangent,float height,float time,float4 wind)
{
    float stepLength=0.001;
    float nextHeight=min(height+stepLength,1.0);
    float delta=max(nextHeight-height,stepLength);
    return tangent+(WindOffset(center+tangent*delta/max(wind.w,0.0001),height+delta,time,wind)-WindOffset(center,height,time,wind))*wind.w/delta;
}
vertex FiberFragment VSMain(device const HairVertexData* vertices [[buffer(0)]],constant FiberFrame& frame [[buffer(1)]],uint index [[vertex_id]])
{
    const device HairVertexData& input=vertices[index]; FiberFragment result;
    const float4 ribbon=float4(input.Ribbon),reflectance=float4(input.Reflectance),emission=float4(input.Emission);
    float3 center=float3(input.Center)+WindOffset(float3(input.Center),ribbon.z,frame.Animation.x,frame.Animation);
    result.Tangent=Unit(WindAxis(float3(input.Center),float3(input.Tangent),ribbon.z,frame.Animation.x,frame.Animation),float3(0,1,0));
    result.World=center+RibbonSide(result.Tangent,TowardEye(center,frame))*ribbon.x;
    float4 world=float4(result.World,1);
    result.Position=float4(dot(frame.ClipFromWorld[0],world),dot(frame.ClipFromWorld[1],world),dot(frame.ClipFromWorld[2],world),dot(frame.ClipFromWorld[3],world));
    result.LinearShadowDepth=dot(frame.ShadowDepth.xyz,float3(result.Position.zw,1));
    float3 oldCenter=float3(input.Center)+WindOffset(float3(input.Center),ribbon.z,frame.Animation.y,frame.Animation);
    float3 oldAxis=Unit(WindAxis(float3(input.Center),float3(input.Tangent),ribbon.z,frame.Animation.y,frame.Animation),float3(0,1,0));
    float3 oldView=frame.PreviousEye.w>OrthographicThreshold ? -frame.PreviousForward.xyz:Unit(frame.PreviousEye.xyz-oldCenter,-frame.PreviousForward.xyz);
    float4 oldWorld=float4(oldCenter+RibbonSide(oldAxis,oldView)*ribbon.x,1);
    result.PreviousClip=float4(dot(frame.PreviousClipFromWorld[0],oldWorld),dot(frame.PreviousClipFromWorld[1],oldWorld),dot(frame.PreviousClipFromWorld[2],oldWorld),dot(frame.PreviousClipFromWorld[3],oldWorld));
    result.Reflectance=reflectance.rgb; result.Emission=emission.rgb; result.CrossHeight=ribbon.yzw; return result;
}
#if defined(EVANS_HAIR_VELOCITY)
fragment float4 PSMain(FiberFragment input [[stage_in]],constant FiberFrame& frame [[buffer(1)]])
{
    const float MinimumClipW=1e-7,PixelHalf=0.5;
    if (dot(float4(input.World,1),frame.ClipPlane)<0) discard_fragment();
    if (frame.MotionOptions.x==0 || input.PreviousClip.w<=MinimumClipW) return float4(0);
    float3 oldNdc=input.PreviousClip.xyz/input.PreviousClip.w;
    if (!all(isfinite(oldNdc)) || oldNdc.z<0 || oldNdc.z>1) return float4(0);
    float2 oldPixel=(oldNdc.xy*float2(PixelHalf,-PixelHalf)+PixelHalf)*frame.MotionViewport.xy;
    float2 velocity=input.Position.xy-oldPixel-frame.MotionViewport.zw;
    if (any(abs(velocity)>MaximumRadiance)) return float4(0);
    return float4(velocity,1-oldNdc.z,1);
}
#elif defined(EVANS_HAIR_VSM_SHADOW)
fragment float2 PSMain(FiberFragment input [[stage_in]],constant FiberFrame& frame [[buffer(1)]])
{
    const float HalfPixelFootprint=0.5;
    float2 gradient=float2(dfdx(input.LinearShadowDepth),dfdy(input.LinearShadowDepth))*HalfPixelFootprint;
    float variance=min(dot(gradient,gradient),frame.ShadowDepth.w);
    if (frame.Options.w!=0 && input.CrossHeight.z<0) discard_fragment();
    if (dot(float4(input.World,1),frame.ClipPlane)<0) discard_fragment();
    float linearDepth=clamp(input.LinearShadowDepth,0.0f,1.0f); return float2(linearDepth,linearDepth*linearDepth+variance);
}
#elif defined(EVANS_HAIR_DEPTH_ONLY)
fragment void PSMain(FiberFragment input [[stage_in]],constant FiberFrame& frame [[buffer(1)]])
{ if ((frame.Options.w!=0 && input.CrossHeight.z<0) || dot(float4(input.World,1),frame.ClipPlane)<0) discard_fragment(); }
#else
float SafeRoot(float value) { return sqrt(max(value,0.0f)); }
float PolarAngle(float y,float x) { return abs(y)+abs(x)>DirectionTolerance ? atan2(y,x):0; }
float BesselLog(float x)
{
    x=abs(x);
    if (x>BesselThreshold)
    {
        float inverse=1/x,series=1+inverse*(BesselFirst+inverse*(BesselSecond+inverse*BesselThird));
        return x-log(FullTurn*x)/2+log(series);
    }
    float term=1,sum=1,quarterSquare=x*x/4;
    for (uint order=1;order<BesselTerms;++order) { term*=quarterSquare/float(order*order); sum+=term; }
    return log(sum);
}
float Longitudinal(float incoming,float outgoing,float variance)
{
    float a=SafeRoot(1-incoming*incoming)*SafeRoot(1-outgoing*outgoing)/variance,b=incoming*outgoing/variance;
    float denominator=log(variance)+1/variance+log(1-exp(-2/variance));
    return exp(BesselLog(a)-b-denominator);
}
float Azimuthal(float difference,float scale)
{
    difference-=FullTurn*floor((difference+Pi)/FullTurn);
    float value=exp(-abs(difference)/scale),edge=exp(-Pi/scale),normalization=(1-edge)/(1+edge);
    return value/(scale*(1+value)*(1+value)*normalization);
}
float Fresnel(float cosine,float eta)
{
    float transmitted=SafeRoot(1-(1-cosine*cosine)/(eta*eta));
    float parallel=(eta*cosine-transmitted)/(eta*cosine+transmitted);
    float perpendicular=(cosine-eta*transmitted)/(cosine+eta*transmitted);
    return (parallel*parallel+perpendicular*perpendicular)/2;
}
struct FiberState
{
    float3 Attenuation[ScatteringOrders]; float4 OutgoingSine;
    float OutgoingPhi,Gamma,InsideGamma;
};
FiberState PrepareFiber(float3 color,float h,float3 outgoing,constant FiberFrame& frame)
{
    FiberState fiber; float sine=clamp(outgoing.x,-1.0f,1.0f),cosine=SafeRoot(1-sine*sine),eta=frame.Optics.x;
    h=clamp(h,-1.0f,1.0f); float inside=h*cosine/SafeRoot(eta*eta-sine*sine);
    float distance=2*SafeRoot(1-inside*inside)/SafeRoot(1-sine*sine/(eta*eta));
    float3 absorption=log(clamp(color,ReflectanceMinimum,1.0f))/frame.Optics.z;
    float3 transmittance=exp(-absorption*absorption*distance);
    float reflection=Fresnel(cosine*SafeRoot(1-h*h),eta); float3 recurrence=reflection*transmittance;
    fiber.Attenuation[0]=float3(reflection); fiber.Attenuation[1]=(1-reflection)*(1-reflection)*transmittance;
    fiber.Attenuation[2]=fiber.Attenuation[1]*recurrence;
    fiber.Attenuation[3]=fiber.Attenuation[2]*recurrence/max(1-recurrence,TransmissionMinimum);
    fiber.OutgoingSine=clamp(float4(sine*frame.CosTilt.y-cosine*frame.SinTilt.y,sine*frame.CosTilt.x+cosine*frame.SinTilt.x,
        sine*frame.CosTilt.z+cosine*frame.SinTilt.z,sine),-1.0f,1.0f);
    fiber.OutgoingPhi=PolarAngle(outgoing.z,outgoing.y); fiber.Gamma=asin(h); fiber.InsideGamma=asin(clamp(inside,-1.0f,1.0f)); return fiber;
}
float3 Scatter(thread const FiberState& fiber,float3 incoming,constant FiberFrame& frame)
{
    float difference=PolarAngle(incoming.z,incoming.y)-fiber.OutgoingPhi; float3 result=0;
    for (uint order=0;order<ScatteringOrders;++order)
    {
        float center=2*order*fiber.InsideGamma-2*fiber.Gamma+order*Pi;
        float azimuth=order==ScatteringOrders-1 ? 1/FullTurn:Azimuthal(difference-center,frame.Optics.y);
        result+=fiber.Attenuation[order]*(Longitudinal(clamp(incoming.x,-1.0f,1.0f),fiber.OutgoingSine[order],frame.Variance[order])*azimuth);
    }
    return result;
}
float3 Incident(FiberLight light,float3 world,thread float3& direction)
{
    direction=light.DirectionRange.xyz; if (uint(light.PositionKind.w)==DirectionalKind) return light.Radiance.rgb;
    float3 offset=light.PositionKind.xyz-world; float squared=dot(offset,offset);
    if (!(squared>0) || !isfinite(squared)) { direction=float3(0); return float3(0); }
    float distance=sqrt(squared),window=1; direction=offset/distance;
    if (light.DirectionRange.w>0)
    { float relative=distance/light.DirectionRange.w,relativeSquared=relative*relative; window=saturate(1-relativeSquared*relativeSquared); window*=window; }
    if (uint(light.PositionKind.w)==RectangleKind) window*=saturate(dot(light.DirectionRange.xyz,-direction));
    return light.Radiance.rgb*(window/max(squared,DistanceSquaredMinimum));
}
#if !defined(EVANS_HAIR_CAPTURE)
// Shadow references: Reeves et al. (1987), PCF; Fernando (2005), PCSS;
// Donnelly and Lauritzen (2006), Variance Shadow Maps. Filter visibility or
// linear moments, then apply the selected direct-light strength once.
// Depth and moment maps have distinct Metal resource types. Only the surface
// moment variant changes this alias; all geometry replay remains invariant.
#ifdef EVANS_HAIR_MOMENT_RECEIVER
typedef texture2d<float,access::read> FiberShadowImage;
#else
typedef depth2d<float,access::read> FiberShadowImage;
#endif
constant uint SoftShadowMode=2,VarianceShadowMode=3;
constant float MinimumShadowW=1e-7,FilterMapFraction=0.25;
float2 ShadowPixel(uint slot,int2 pixel,constant FiberFrame& f,FiberShadowImage a,FiberShadowImage b,FiberShadowImage c,FiberShadowImage d,FiberShadowImage e)
{
    if (any(pixel<0) || any(pixel>=int2(f.ShadowExtents[slot].xy))) return float2(1);
#ifdef EVANS_HAIR_MOMENT_RECEIVER
    if (slot==0) return a.read(uint2(pixel)).rg;
    if (slot==1) return b.read(uint2(pixel)).rg;
    if (slot==2) return c.read(uint2(pixel)).rg;
    if (slot==3) return d.read(uint2(pixel)).rg;
    return e.read(uint2(pixel)).rg;
#else
    if (slot==0) return float2(a.read(uint2(pixel)),0);
    if (slot==1) return float2(b.read(uint2(pixel)),0);
    if (slot==2) return float2(c.read(uint2(pixel)),0);
    if (slot==3) return float2(d.read(uint2(pixel)),0);
    return float2(e.read(uint2(pixel)),0);
#endif
}
float ShadowComparison(uint slot,float2 uv,float reference,constant FiberFrame& f,FiberShadowImage a,FiberShadowImage b,FiberShadowImage c,FiberShadowImage d,FiberShadowImage e)
{
    float2 pixel=uv*f.ShadowExtents[slot].xy-0.5,weight=fract(pixel); int2 p=int2(floor(pixel));
    float4 depths=float4(ShadowPixel(slot,p,f,a,b,c,d,e).x,ShadowPixel(slot,p+int2(1,0),f,a,b,c,d,e).x,
        ShadowPixel(slot,p+int2(0,1),f,a,b,c,d,e).x,ShadowPixel(slot,p+1,f,a,b,c,d,e).x);
    float4 visible=step(float4(reference),depths);
    return mix(mix(visible.x,visible.y,weight.x),mix(visible.z,visible.w,weight.x),weight.y);
}
float2 ShadowMoments(uint slot,float2 uv,constant FiberFrame& f,FiberShadowImage a,FiberShadowImage b,FiberShadowImage c,FiberShadowImage d,FiberShadowImage e)
{
    float2 pixel=uv*f.ShadowExtents[slot].xy-0.5,weight=fract(pixel); int2 p=int2(floor(pixel));
    return mix(mix(ShadowPixel(slot,p,f,a,b,c,d,e),ShadowPixel(slot,p+int2(1,0),f,a,b,c,d,e),weight.x),
        mix(ShadowPixel(slot,p+int2(0,1),f,a,b,c,d,e),ShadowPixel(slot,p+1,f,a,b,c,d,e),weight.x),weight.y);
}
float ShadowDistance(float depth,float4 metric)
{ return metric.x>0 ? 1/((1-depth)/metric.x+depth/metric.y):depth*metric.y; }
float2 ShadowDisk(uint index,uint count)
{
    float angle=index*Pi*(3-sqrt(5.0f));
    return sqrt((index+0.5f)/count)*float2(cos(angle),sin(angle));
}
float2 ShadowFootprint(uint slot,float radius,float4 metric,float budget,constant FiberFrame& f)
{
    float2 footprint=radius/metric.zw;
    float texels=max(footprint.x*f.ShadowExtents[slot].x,footprint.y*f.ShadowExtents[slot].y);
    float maximum=min(budget,min(f.ShadowExtents[slot].x,f.ShadowExtents[slot].y)*FilterMapFraction);
    return texels>maximum?footprint*(maximum/texels):footprint;
}
float4 ShadowClip(uint slot,float3 world,constant FiberFrame& f)
{
    float4 p=float4(world,1);
    if (slot==0) return float4(dot(f.ShadowRows[0],p),dot(f.ShadowRows[1],p),dot(f.ShadowRows[2],p),dot(f.ShadowRows[3],p));
    uint row=(slot-1)*4;
    return float4(dot(f.ShadowCascadeRows[row],p),dot(f.ShadowCascadeRows[row+1],p),dot(f.ShadowCascadeRows[row+2],p),dot(f.ShadowCascadeRows[row+3],p));
}
float ShadowMapVisibility(uint slot,float3 world,float bias,constant FiberFrame& f,FiberShadowImage a,FiberShadowImage b,FiberShadowImage c,FiberShadowImage d,FiberShadowImage e)
{
    float4 clip=ShadowClip(slot,world,f);
    if ((f.ShadowControl.y&(1u<<slot))==0 || clip.w<=MinimumShadowW) return 1;
    float3 ndc=clip.xyz/clip.w; float2 uv=ndc.xy*float2(0.5,-0.5)+0.5;
    if (any(uv<0) || any(uv>1) || ndc.z<0 || ndc.z>1) return 1;
    float reference=ndc.z-bias;
    if (f.ShadowControl.x==VarianceShadowMode)
    {
        float2 moments=ShadowMoments(slot,uv,f,a,b,c,d,e);
        float linearDepth=dot(f.ShadowVarianceEncoding[slot].xyz,float3(clip.zw,1));
        float delta=linearDepth-bias-moments.x; if (delta<=0) return 1;
        float variance=max(moments.y-moments.x*moments.x,f.ShadowVarianceParams.x);
        return saturate((variance/(variance+delta*delta)-f.ShadowVarianceParams.y)/(1-f.ShadowVarianceParams.y));
    }
    if (f.ShadowControl.x==SoftShadowMode)
    {
        float4 metric=f.ShadowSoftMetrics[slot]; float emitter=metric.x>0?f.ShadowEmitter.x:f.ShadowEmitter.y;
        if (emitter<=0) return step(reference,ShadowPixel(slot,int2(floor(uv*f.ShadowExtents[slot].xy)),f,a,b,c,d,e).x);
        float receiver=ShadowDistance(ndc.z,metric),sum=0; uint blockers=0;
        float searchWorld=metric.x>0?emitter*(receiver-metric.x)/receiver:emitter*receiver;
        float2 search=ShadowFootprint(slot,searchWorld,metric,f.ShadowEmitter.z,f);
        for (uint i=0;i<f.ShadowSoftControl.x;++i)
        {
            float2 at=uv+search*ShadowDisk(i,f.ShadowSoftControl.x);
            float stored=ShadowPixel(slot,int2(floor(at*f.ShadowExtents[slot].xy)),f,a,b,c,d,e).x;
            if (stored<reference) { sum+=ShadowDistance(stored,metric); ++blockers; }
        }
        if (blockers==0) return 1;
        float average=sum/blockers,gap=max(receiver-average,0.0f);
        float penumbra=metric.x>0?emitter*gap/average*metric.x/receiver:emitter*gap;
        float2 radius=ShadowFootprint(slot,penumbra,metric,f.ShadowEmitter.w,f); float result=0;
        for (uint i=0;i<f.ShadowSoftControl.y;++i) result+=ShadowComparison(slot,uv+radius*ShadowDisk(i,f.ShadowSoftControl.y),reference,f,a,b,c,d,e);
        return result/f.ShadowSoftControl.y;
    }
    if (f.ShadowFiltering.x==0) return step(reference,ShadowPixel(slot,int2(floor(uv*f.ShadowExtents[slot].xy)),f,a,b,c,d,e).x);
    int radius=int(f.ShadowFiltering.x); float visible=0;
    for (int y=-radius;y<=radius;++y) for (int x=-radius;x<=radius;++x)
        visible+=ShadowComparison(slot,uv+float2(x,y)*f.ShadowExtents[slot].zw,reference,f,a,b,c,d,e);
    return visible/f.ShadowFiltering.y;
}
float FiberShadow(float3 world,float3 cylinderNormal,float3 incoming,constant FiberFrame& f,FiberShadowImage a,FiberShadowImage b,FiberShadowImage c,FiberShadowImage d,FiberShadowImage e)
{
    if (f.ShadowControl.z==0) return 1;
    float cosine=saturate(abs(dot(cylinderNormal,incoming)));
    float bias=min(f.ShadowBias.z,f.ShadowBias.x+f.ShadowBias.y*SafeRoot(1-cosine*cosine)/max(cosine,f.ShadowBias.w));
    float visible=1; uint count=uint(f.ShadowCascadeParams.z);
    if (count==0) visible=ShadowMapVisibility(0,world,bias,f,a,b,c,d,e);
    else
    {
        float distance=dot(f.ShadowEyePlane,float4(world,1));
        if (distance>=f.ShadowCascadeParams.x && distance<=f.ShadowCascadeParams.y)
        {
            uint cascade=0; while (cascade+1<count && distance>f.ShadowFar[cascade]) ++cascade;
            visible=ShadowMapVisibility(cascade+1,world,bias,f,a,b,c,d,e);
            float interval=f.ShadowFar[cascade]-f.ShadowBlend[cascade];
            if (interval>0 && distance>=f.ShadowBlend[cascade])
            {
                float next=cascade+1<count?ShadowMapVisibility(cascade+2,world,bias,f,a,b,c,d,e):1;
                visible=mix(visible,next,saturate((distance-f.ShadowBlend[cascade])/interval));
            }
        }
    }
    return visible;
}
#endif
struct FiberOutput
{
    float4 Color [[color(0)]];
#ifndef EVANS_HAIR_CAPTURE
    float4 MaterialNormal [[color(1)]];
#endif
};
fragment FiberOutput PSMain(FiberFragment input [[stage_in]],constant FiberFrame& frame [[buffer(1)]],device const FiberLight* lights [[buffer(2)]]
#ifndef EVANS_HAIR_CAPTURE
    ,FiberShadowImage shadow0 [[texture(0)]],FiberShadowImage shadow1 [[texture(1)]],FiberShadowImage shadow2 [[texture(2)]],FiberShadowImage shadow3 [[texture(3)]],FiberShadowImage shadow4 [[texture(4)]]
#ifdef EVANS_HAIR_CONTACT_RECEIVER
    ,texture2d_array<float,access::read> contactLayers [[texture(5)]]
#ifdef EVANS_HAIR_CONTACT_SAMPLES
    ,uint contactSample [[sample_id]]
#endif
#endif
#endif
    )
{
    if (dot(float4(input.World,1),frame.ClipPlane)<0) discard_fragment();
    float3 tangent=Unit(input.Tangent,float3(0,1,0)),view=TowardEye(input.World,frame),side=RibbonSide(tangent,view);
    float3 normal=Unit(cross(side,tangent),-frame.Forward.xyz);
    float3 color=saturate(input.Reflectance*mix(frame.RootTint.rgb,frame.TipTint.rgb,saturate(input.CrossHeight.y)));
    FiberState fiber=PrepareFiber(color,input.CrossHeight.x,float3(dot(view,tangent),dot(view,side),dot(view,normal)),frame);
    float3 radiance=color*frame.Ambient.rgb*(frame.MotionOptions.y!=0?0.0f:frame.Ambient.w);
    float h=clamp(input.CrossHeight.x,-1.0f,1.0f); float3 cylinderNormal=side*h+normal*SafeRoot(1-h*h);
    for (uint index=0;index<frame.Options.x;++index)
    {
        float3 direction; float3 incident=Incident(lights[index],input.World,direction);
#ifndef EVANS_HAIR_CAPTURE
        if (index>=frame.Options.y && index<frame.Options.z)
        {
            float visibility=FiberShadow(input.World,cylinderNormal,direction,frame,shadow0,shadow1,shadow2,shadow3,shadow4);
#ifdef EVANS_HAIR_CONTACT_RECEIVER
            uint layer=0;
#ifdef EVANS_HAIR_CONTACT_SAMPLES
            layer=contactSample;
#endif
            if (frame.ContactControl.x!=0) visibility*=contactLayers.read(uint2(input.Position.xy),layer).r;
#endif
            incident*=1-frame.ShadowComposition.x*(1-visibility);
        }
#endif
        radiance+=incident*Scatter(fiber,float3(dot(direction,tangent),dot(direction,side),dot(direction,normal)),frame);
    }
    FiberOutput result; result.Color=float4(clamp(input.Emission+radiance*frame.Optics.w,0.0f,MaximumRadiance),1);
#ifndef EVANS_HAIR_CAPTURE
    result.MaterialNormal=float4(0,0,0,-1);
#endif
    return result;
}
#endif
