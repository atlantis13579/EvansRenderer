#version 450
// Native Vulkan GLSL, compiled in memory. This file owns ribbon placement,
// cylindrical fiber scattering, a depth variant and capture/surface outputs.
// https://pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair
const float Pi=3.141592653589793,TwoPi=2.0*Pi;
const float DirectionEpsilon=1e-12,ReflectanceMinimum=1e-4,DenominatorMinimum=1e-6;
const float HalfMaximum=65504.0,DistanceSquaredMinimum=1e-6,AsymptoticThreshold=12.0;
const float BesselFirst=1.0/8.0,BesselSecond=9.0/128.0,BesselThird=225.0/3072.0;
const float ParallelThreshold=0.9,OrthographicThreshold=0.5;
const int BesselSeriesTerms=20,ScatteringOrderCount=4;
const uint DirectionalSource=0u,RectangleSource=2u;
layout(set=0,binding=0,std140,row_major) uniform FiberFrame
{
    mat4 WorldToClip;
    vec4 CameraPosition,CameraForward,PathVariance,TiltSine,TiltCosine,Optical;
    vec4 RootColor,TipColor,AmbientLight,ClipPlane;
    uvec4 LightOptions;
    vec4 ShadowDepth;
    mat4 PreviousWorldToClip;
    vec4 PreviousPosition,PreviousForward,MotionViewport;
    uvec4 MotionOptions;
    mat4 ShadowProjection,ShadowCascadeProjection[4];
    vec4 ShadowFar,ShadowBlend,ShadowEyePlane,ShadowCascadeParams,ShadowFiltering;
    vec4 ShadowSoftMetrics[5],ShadowEmitter;
    uvec4 ShadowSoftControl;
    vec4 ShadowVarianceEncoding[5],ShadowVarianceParams;
    uvec4 ShadowVarianceControl;
    vec4 ShadowExtents[5],ShadowBias;
    uvec4 ShadowControl;
    vec4 ShadowComposition;
    uvec4 ContactControl;
    vec4 Animation;
} Frame;
vec3 SafeUnit(vec3 vector,vec3 fallback)
{
    float squared=dot(vector,vector);
    return squared>DirectionEpsilon && !isinf(squared) && !isnan(squared) ? vector*inversesqrt(squared):fallback;
}
vec3 ViewVector(vec3 position)
{ return Frame.CameraPosition.w>OrthographicThreshold ? -Frame.CameraForward.xyz:SafeUnit(Frame.CameraPosition.xyz-position,-Frame.CameraForward.xyz); }
vec3 WidthDirection(vec3 tangent,vec3 towardCamera)
{
    vec3 width=cross(tangent,towardCamera);
    if (dot(width,width)<=DirectionEpsilon)
        width=cross(tangent,abs(tangent.y)<ParallelThreshold ? vec3(0,1,0):vec3(1,0,0));
    return SafeUnit(width,vec3(1,0,0));
}
// Roots remain attached; bending increases quadratically toward each tip.
vec3 WindOffset(vec3 center,float height,float time,vec4 wind)
{
    float phase=time*2.2+dot(center,vec3(2.0,1.0,3.0));
    return vec3(sin(phase),0.18*sin(phase*0.7),0.65*cos(phase*1.13))*wind.z*height*height;
}
vec3 WindAxis(vec3 center,vec3 tangent,float height,float time,vec4 wind)
{
    float stepLength=0.001;
    float nextHeight=min(height+stepLength,1.0);
    float delta=max(nextHeight-height,stepLength);
    return tangent+(WindOffset(center+tangent*delta/max(wind.w,0.0001),height+delta,time,wind)-WindOffset(center,height,time,wind))*wind.w/delta;
}
#ifdef EVANS_VERTEX_STAGE
invariant gl_Position;
precise gl_Position;
layout(location=0) in vec3 InCenter;
layout(location=1) in vec3 InAxis;
layout(location=2) in vec4 InReflectance;
layout(location=3) in vec4 InEmission;
layout(location=4) in vec4 InRibbon;
layout(location=0) out vec3 WorldPosition;
layout(location=1) out vec3 FiberAxis;
layout(location=2) out vec3 Reflectance;
layout(location=3) out vec3 Emission;
layout(location=4) centroid out vec3 CrossHeight;
layout(location=5) out float LinearShadowDepth;
layout(location=6) out vec4 PreviousClip;
void main()
{
    vec3 center=InCenter+WindOffset(InCenter,InRibbon.z,Frame.Animation.x,Frame.Animation);
    FiberAxis=SafeUnit(WindAxis(InCenter,InAxis,InRibbon.z,Frame.Animation.x,Frame.Animation),vec3(0,1,0));
    WorldPosition=center+WidthDirection(FiberAxis,ViewVector(center))*InRibbon.x;
    gl_Position=Frame.WorldToClip*vec4(WorldPosition,1.0);
    LinearShadowDepth=dot(Frame.ShadowDepth.xyz,vec3(gl_Position.zw,1.0));
    vec3 oldCenter=InCenter+WindOffset(InCenter,InRibbon.z,Frame.Animation.y,Frame.Animation);
    vec3 oldAxis=SafeUnit(WindAxis(InCenter,InAxis,InRibbon.z,Frame.Animation.y,Frame.Animation),vec3(0,1,0));
    vec3 oldView=Frame.PreviousPosition.w>OrthographicThreshold ? -Frame.PreviousForward.xyz:SafeUnit(Frame.PreviousPosition.xyz-oldCenter,-Frame.PreviousForward.xyz);
    vec3 oldWorld=oldCenter+WidthDirection(oldAxis,oldView)*InRibbon.x;
    PreviousClip=Frame.PreviousWorldToClip*vec4(oldWorld,1.0);
    gl_Position.y=-gl_Position.y; // Positive-height viewport, matching opaque and shadow geometry.
    Reflectance=InReflectance.rgb; Emission=InEmission.rgb; CrossHeight=InRibbon.yzw;
}
#else
layout(location=0) in vec3 WorldPosition;
layout(location=1) in vec3 FiberAxis;
layout(location=2) in vec3 Reflectance;
layout(location=3) in vec3 Emission;
// Evaluate the fiber cross section at covered samples of thin MSAA ribbons.
layout(location=4) centroid in vec3 CrossHeight;
layout(location=5) in float LinearShadowDepth;
layout(location=6) in vec4 PreviousClip;
#if defined(EVANS_HAIR_VELOCITY)
layout(location=0) out vec4 OutVelocity;
void main()
{
    const float MinimumHomogeneousW=1e-7,PixelHalf=0.5;
    if (dot(vec4(WorldPosition,1.0),Frame.ClipPlane)<0.0) discard;
    OutVelocity=vec4(0.0);
    if (Frame.MotionOptions.x==0u || PreviousClip.w<=MinimumHomogeneousW) return;
    vec3 oldNdc=PreviousClip.xyz/PreviousClip.w;
    if (any(isnan(oldNdc)) || any(isinf(oldNdc)) || oldNdc.z<0.0 || oldNdc.z>1.0) return;
    vec2 oldPixel=(oldNdc.xy*vec2(PixelHalf,-PixelHalf)+PixelHalf)*Frame.MotionViewport.xy;
    vec2 velocity=gl_FragCoord.xy-oldPixel-Frame.MotionViewport.zw;
    if (any(greaterThan(abs(velocity),vec2(HalfMaximum)))) return;
    OutVelocity=vec4(velocity,1.0-oldNdc.z,1.0);
}
#elif defined(EVANS_HAIR_VSM_SHADOW)
layout(location=0) out vec2 OutMoments;
void main()
{
    const float FootprintHalfWidth=0.5;
    vec2 footprint=vec2(dFdx(LinearShadowDepth),dFdy(LinearShadowDepth))*FootprintHalfWidth;
    float variance=min(dot(footprint,footprint),Frame.ShadowDepth.w);
    if (Frame.LightOptions.w!=0u && CrossHeight.z<0.0) discard;
    if (dot(vec4(WorldPosition,1.0),Frame.ClipPlane)<0.0) discard;
    float linearDepth=clamp(LinearShadowDepth,0.0,1.0);
    OutMoments=vec2(linearDepth,linearDepth*linearDepth+variance);
}
#elif defined(EVANS_HAIR_DEPTH_ONLY)
void main() { if ((Frame.LightOptions.w!=0u && CrossHeight.z<0.0) || dot(vec4(WorldPosition,1.0),Frame.ClipPlane)<0.0) discard; }
#else
struct LightSample { vec4 PositionKind,DirectionRange,Radiance; };
layout(set=0,binding=1,std430) readonly buffer FiberLights { LightSample Samples[]; };
layout(location=0) out vec4 OutColor;
#ifndef EVANS_HAIR_CAPTURE
layout(location=1) out vec4 OutNormal;
#endif
float SafeRoot(float value) { return sqrt(max(value,0.0)); }
float PolarAngle(float y,float x) { return abs(y)+abs(x)>DirectionEpsilon ? atan(y,x):0.0; }
float BesselLog(float x)
{
    x=abs(x);
    if (x>AsymptoticThreshold)
    {
        float inverse=1.0/x,correction=1.0+inverse*(BesselFirst+inverse*(BesselSecond+inverse*BesselThird));
        return x-log(TwoPi*x)/2.0+log(correction);
    }
    float term=1.0,sum=1.0,quarterSquare=x*x/4.0;
    for (int order=1;order<BesselSeriesTerms;++order) { term*=quarterSquare/float(order*order); sum+=term; }
    return log(sum);
}
float LongitudinalDensity(float incomingSine,float outgoingSine,float variance)
{
    float a=SafeRoot(1.0-incomingSine*incomingSine)*SafeRoot(1.0-outgoingSine*outgoingSine)/variance;
    float b=incomingSine*outgoingSine/variance;
    float logMass=log(variance)+1.0/variance+log(1.0-exp(-2.0/variance));
    return exp(BesselLog(a)-b-logMass);
}
float AzimuthalDensity(float delta)
{
    delta-=TwoPi*floor((delta+Pi)/TwoPi);
    float scale=Frame.Optical.y,weight=exp(-abs(delta)/scale),boundary=exp(-Pi/scale);
    float mass=(1.0-boundary)/(1.0+boundary);
    return weight/(scale*(1.0+weight)*(1.0+weight)*mass);
}
float InterfaceFresnel(float cosine)
{
    float eta=Frame.Optical.x,inside=SafeRoot(1.0-(1.0-cosine*cosine)/(eta*eta));
    float parallel=(eta*cosine-inside)/(eta*cosine+inside);
    float perpendicular=(cosine-eta*inside)/(cosine+eta*inside);
    return (parallel*parallel+perpendicular*perpendicular)/2.0;
}
struct FiberContext
{
    vec4 OutgoingSine;
    vec3 Attenuation[ScatteringOrderCount];
    float OutgoingPhi,Gamma,InsideGamma;
};
FiberContext PrepareContext(vec3 color,float widthCoordinate,vec3 view)
{
    FiberContext result; float sine=clamp(view.x,-1.0,1.0),cosine=SafeRoot(1.0-sine*sine),eta=Frame.Optical.x;
    float h=clamp(widthCoordinate,-1.0,1.0),insideSine=h*cosine/SafeRoot(eta*eta-sine*sine);
    float distance=2.0*SafeRoot(1.0-insideSine*insideSine)/SafeRoot(1.0-sine*sine/(eta*eta));
    vec3 coefficient=log(clamp(color,vec3(ReflectanceMinimum),vec3(1.0)))/Frame.Optical.z;
    vec3 through=exp(-coefficient*coefficient*distance);
    float reflection=InterfaceFresnel(cosine*SafeRoot(1.0-h*h)); vec3 recurrence=reflection*through;
    result.Attenuation[0]=vec3(reflection); result.Attenuation[1]=(1.0-reflection)*(1.0-reflection)*through;
    result.Attenuation[2]=result.Attenuation[1]*recurrence;
    result.Attenuation[3]=result.Attenuation[2]*recurrence/max(vec3(1.0)-recurrence,vec3(DenominatorMinimum));
    result.OutgoingSine=clamp(vec4(sine*Frame.TiltCosine.y-cosine*Frame.TiltSine.y,sine*Frame.TiltCosine.x+cosine*Frame.TiltSine.x,
        sine*Frame.TiltCosine.z+cosine*Frame.TiltSine.z,sine),-1.0,1.0);
    result.OutgoingPhi=PolarAngle(view.z,view.y); result.Gamma=asin(h); result.InsideGamma=asin(clamp(insideSine,-1.0,1.0)); return result;
}
vec3 Scatter(FiberContext fiber,vec3 incoming)
{
    vec3 sum=vec3(0.0); float delta=PolarAngle(incoming.z,incoming.y)-fiber.OutgoingPhi;
    for (int order=0;order<ScatteringOrderCount;++order)
    {
        float center=2.0*float(order)*fiber.InsideGamma-2.0*fiber.Gamma+float(order)*Pi;
        float angular=order==ScatteringOrderCount-1 ? 1.0/TwoPi:AzimuthalDensity(delta-center);
        sum+=fiber.Attenuation[order]*(angular*LongitudinalDensity(clamp(incoming.x,-1.0,1.0),fiber.OutgoingSine[order],Frame.PathVariance[order]));
    }
    return sum;
}
vec3 IncidentRadiance(LightSample source,vec3 position,out vec3 direction)
{
    direction=source.DirectionRange.xyz; if (uint(source.PositionKind.w)==DirectionalSource) return source.Radiance.rgb;
    vec3 offset=source.PositionKind.xyz-position; float squared=dot(offset,offset);
    if (!(squared>0.0) || isinf(squared) || isnan(squared)) { direction=vec3(0.0); return vec3(0.0); }
    float distance=sqrt(squared),window=1.0; direction=offset/distance;
    if (source.DirectionRange.w>0.0)
    {
        float relative=distance/source.DirectionRange.w,relativeSquared=relative*relative;
        window=clamp(1.0-relativeSquared*relativeSquared,0.0,1.0); window*=window;
    }
    if (uint(source.PositionKind.w)==RectangleSource) window*=clamp(dot(source.DirectionRange.xyz,-direction),0.0,1.0);
    return source.Radiance.rgb*(window/max(squared,DistanceSquaredMinimum));
}
// Shadow references: Reeves et al. (1987), PCF; Fernando (2005), PCSS;
// Donnelly and Lauritzen (2006), Variance Shadow Maps. Filter visibility or
// linear moments, then apply the selected direct-light strength once.
layout(set=0,binding=2) uniform sampler2D ShadowMap0;
layout(set=0,binding=3) uniform sampler2D ShadowMap1;
layout(set=0,binding=4) uniform sampler2D ShadowMap2;
layout(set=0,binding=5) uniform sampler2D ShadowMap3;
layout(set=0,binding=6) uniform sampler2D ShadowMap4;
#ifdef EVANS_HAIR_CONTACT_RECEIVER
layout(set=0,binding=7) uniform sampler2DArray FiberContactLayers;
#endif
const uint PcssFilter=2u,VsmFilter=3u;
const float ShadowClipMinimum=1e-7,FilterExtentFraction=0.25;
// Literal resource branches work without descriptor-indexing extensions.
vec2 ReadShadow(uint slot,ivec2 pixel)
{
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(Frame.ShadowExtents[slot].xy)))) return vec2(1);
    if (slot==0u) return texelFetch(ShadowMap0,pixel,0).rg;
    if (slot==1u) return texelFetch(ShadowMap1,pixel,0).rg;
    if (slot==2u) return texelFetch(ShadowMap2,pixel,0).rg;
    if (slot==3u) return texelFetch(ShadowMap3,pixel,0).rg;
    return texelFetch(ShadowMap4,pixel,0).rg;
}
float BilinearVisibility(uint slot,vec2 uv,float reference)
{
    vec2 location=uv*Frame.ShadowExtents[slot].xy-0.5,f=fract(location); ivec2 p=ivec2(floor(location));
    vec4 depths=vec4(ReadShadow(slot,p).x,ReadShadow(slot,p+ivec2(1,0)).x,ReadShadow(slot,p+ivec2(0,1)).x,ReadShadow(slot,p+1).x);
    vec4 comparisons=step(vec4(reference),depths);
    return mix(mix(comparisons.x,comparisons.y,f.x),mix(comparisons.z,comparisons.w,f.x),f.y);
}
vec2 BilinearMoments(uint slot,vec2 uv)
{
    vec2 location=uv*Frame.ShadowExtents[slot].xy-0.5,f=fract(location); ivec2 p=ivec2(floor(location));
    return mix(mix(ReadShadow(slot,p),ReadShadow(slot,p+ivec2(1,0)),f.x),
        mix(ReadShadow(slot,p+ivec2(0,1)),ReadShadow(slot,p+1),f.x),f.y);
}
float AxialDistance(float z,vec4 metric)
{ return metric.x>0.0 ? 1.0/((1.0-z)/metric.x+z/metric.y):z*metric.y; }
vec2 SpiralPoint(uint index,uint count)
{
    const float GoldenAngle=Pi*(3.0-sqrt(5.0)); float angle=float(index)*GoldenAngle;
    return sqrt((float(index)+0.5)/float(count))*vec2(cos(angle),sin(angle));
}
vec2 FilterFootprint(uint slot,float worldRadius,vec4 metric,float cap)
{
    vec2 uv=worldRadius/metric.zw;
    float size=max(uv.x*Frame.ShadowExtents[slot].x,uv.y*Frame.ShadowExtents[slot].y);
    float maximum=min(cap,min(Frame.ShadowExtents[slot].x,Frame.ShadowExtents[slot].y)*FilterExtentFraction);
    return size>maximum?uv*(maximum/size):uv;
}
float ShadowLayer(uint slot,vec3 world,float bias)
{
    vec4 clip=slot==0u?Frame.ShadowProjection*vec4(world,1):Frame.ShadowCascadeProjection[slot-1u]*vec4(world,1);
    if ((Frame.ShadowControl.y&(1u<<slot))==0u || clip.w<=ShadowClipMinimum) return 1.0;
    vec3 projected=clip.xyz/clip.w; vec2 uv=projected.xy*vec2(0.5,-0.5)+0.5;
    if (any(lessThan(uv,vec2(0))) || any(greaterThan(uv,vec2(1))) || projected.z<0.0 || projected.z>1.0) return 1.0;
    float reference=projected.z-bias;
    if (Frame.ShadowControl.x==VsmFilter)
    {
        vec2 moments=BilinearMoments(slot,uv);
        float metricDepth=dot(Frame.ShadowVarianceEncoding[slot].xyz,vec3(clip.zw,1));
        float delta=metricDepth-bias-moments.x; if (delta<=0.0) return 1.0;
        float variance=max(moments.y-moments.x*moments.x,Frame.ShadowVarianceParams.x);
        float probability=variance/(variance+delta*delta);
        return clamp((probability-Frame.ShadowVarianceParams.y)/(1.0-Frame.ShadowVarianceParams.y),0.0,1.0);
    }
    if (Frame.ShadowControl.x==PcssFilter)
    {
        vec4 metric=Frame.ShadowSoftMetrics[slot]; float emitter=metric.x>0.0?Frame.ShadowEmitter.x:Frame.ShadowEmitter.y;
        if (emitter<=0.0) return step(reference,ReadShadow(slot,ivec2(floor(uv*Frame.ShadowExtents[slot].xy))).x);
        float receiver=AxialDistance(projected.z,metric),sum=0.0; uint blockers=0u;
        float worldSearch=metric.x>0.0?emitter*(receiver-metric.x)/receiver:emitter*receiver;
        vec2 search=FilterFootprint(slot,worldSearch,metric,Frame.ShadowEmitter.z);
        for (uint i=0u;i<Frame.ShadowSoftControl.x;++i)
        {
            vec2 location=uv+search*SpiralPoint(i,Frame.ShadowSoftControl.x);
            float z=ReadShadow(slot,ivec2(floor(location*Frame.ShadowExtents[slot].xy))).x;
            if (z<reference) { sum+=AxialDistance(z,metric); ++blockers; }
        }
        if (blockers==0u) return 1.0;
        float average=sum/float(blockers),gap=max(receiver-average,0.0);
        float worldFilter=metric.x>0.0?emitter*gap/average*metric.x/receiver:emitter*gap;
        vec2 radius=FilterFootprint(slot,worldFilter,metric,Frame.ShadowEmitter.w); float visible=0.0;
        for (uint i=0u;i<Frame.ShadowSoftControl.y;++i) visible+=BilinearVisibility(slot,uv+radius*SpiralPoint(i,Frame.ShadowSoftControl.y),reference);
        return visible/float(Frame.ShadowSoftControl.y);
    }
    if (Frame.ShadowFiltering.x==0.0) return step(reference,ReadShadow(slot,ivec2(floor(uv*Frame.ShadowExtents[slot].xy))).x);
    int radius=int(Frame.ShadowFiltering.x); float result=0.0;
    for (int y=-radius;y<=radius;++y) for (int x=-radius;x<=radius;++x)
        result+=BilinearVisibility(slot,uv+vec2(x,y)*Frame.ShadowExtents[slot].zw,reference);
    return result/Frame.ShadowFiltering.y;
}
float DirectVisibility(vec3 world,vec3 cylindricalNormal,vec3 incoming)
{
    if (Frame.ShadowControl.z==0u) return 1.0;
    float cosine=clamp(abs(dot(cylindricalNormal,incoming)),0.0,1.0);
    float slope=SafeRoot(1.0-cosine*cosine)/max(cosine,Frame.ShadowBias.w);
    float bias=min(Frame.ShadowBias.z,Frame.ShadowBias.x+Frame.ShadowBias.y*slope),visibility=1.0;
    uint count=uint(Frame.ShadowCascadeParams.z);
    if (count==0u) visibility=ShadowLayer(0u,world,bias);
    else
    {
        float distance=dot(Frame.ShadowEyePlane,vec4(world,1));
        if (distance>=Frame.ShadowCascadeParams.x && distance<=Frame.ShadowCascadeParams.y)
        {
            uint layer=0u; while (layer+1u<count && distance>Frame.ShadowFar[layer]) ++layer;
            visibility=ShadowLayer(layer+1u,world,bias); float overlap=Frame.ShadowFar[layer]-Frame.ShadowBlend[layer];
            if (overlap>0.0 && distance>=Frame.ShadowBlend[layer])
            {
                float next=layer+1u<count?ShadowLayer(layer+2u,world,bias):1.0;
                visibility=mix(visibility,next,clamp((distance-Frame.ShadowBlend[layer])/overlap,0.0,1.0));
            }
        }
    }
    return visibility;
}
void main()
{
    if (dot(vec4(WorldPosition,1.0),Frame.ClipPlane)<0.0) discard;
    vec3 tangent=SafeUnit(FiberAxis,vec3(0,1,0)),view=ViewVector(WorldPosition),side=WidthDirection(tangent,view);
    vec3 normal=SafeUnit(cross(side,tangent),-Frame.CameraForward.xyz);
    vec3 color=clamp(Reflectance*mix(Frame.RootColor.rgb,Frame.TipColor.rgb,clamp(CrossHeight.y,0.0,1.0)),0.0,1.0);
    FiberContext fiber=PrepareContext(color,CrossHeight.x,vec3(dot(view,tangent),dot(view,side),dot(view,normal)));
    vec3 radiance=color*Frame.AmbientLight.rgb*(Frame.MotionOptions.y!=0u?0.0:Frame.AmbientLight.w);
    float h=clamp(CrossHeight.x,-1.0,1.0); vec3 cylinderNormal=side*h+normal*SafeRoot(1.0-h*h);
    for (uint index=0u;index<Frame.LightOptions.x;++index)
    {
        vec3 direction; vec3 incident=IncidentRadiance(Samples[index],WorldPosition,direction);
#ifndef EVANS_HAIR_CAPTURE
        if (index>=Frame.LightOptions.y && index<Frame.LightOptions.z)
        {
            float visibility=DirectVisibility(WorldPosition,cylinderNormal,direction);
#ifdef EVANS_HAIR_CONTACT_RECEIVER
            int sampleIndex=0;
#ifdef EVANS_HAIR_CONTACT_SAMPLES
            sampleIndex=gl_SampleID;
#endif
            if (Frame.ContactControl.x!=0u) visibility*=texelFetch(FiberContactLayers,ivec3(ivec2(gl_FragCoord.xy),sampleIndex),0).r;
#endif
            incident*=1.0-Frame.ShadowComposition.x*(1.0-visibility);
        }
#endif
        radiance+=incident*Scatter(fiber,vec3(dot(direction,tangent),dot(direction,side),dot(direction,normal)));
    }
    OutColor=vec4(clamp(Emission+radiance*Frame.Optical.w,0.0,HalfMaximum),1.0);
#ifndef EVANS_HAIR_CAPTURE
    OutNormal=vec4(0.0,0.0,0.0,-1.0);
#endif
}
#endif
#endif
