#version 310 es
precision highp float;
precision highp int;
// Independent ES ribbon and fiber model. RGB source attributes are linear.
// R/TT/TRT plus a higher-order tail follow the normalized cylindrical model:
// https://pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair
const float Pi=3.141592653589793,FullTurn=2.0*Pi;
const float DirectionTolerance=1e-12,ReflectanceFloor=1e-4,TransmissionFloor=1e-6;
const float HalfMaximum=65504.0,PointDistanceFloor=1e-6,BesselThreshold=12.0;
const float BesselFirst=1.0/8.0,BesselSecond=9.0/128.0,BesselThird=225.0/3072.0;
const float ParallelLimit=0.9,OrthographicThreshold=0.5;
const int BesselTerms=20,PathCount=4;
const uint DirectionalKind=0u,RectangleKind=2u;
layout(std140,binding=0) uniform HairFrame
{
    vec4 ClipRows[4];
    vec4 Eye,CameraForward,Variance,SinTilt,CosTilt,Optics;
    vec4 RootTint,TipTint,Ambient,ClipPlane;
    uvec4 Options;
    vec4 ShadowDepth;
    vec4 PreviousClipRows[4];
    vec4 PreviousEye,PreviousForward,MotionViewport;
    uvec4 MotionOptions;
    vec4 ShadowRows[4],ShadowCascadeRows[16];
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
};
vec3 Unit(vec3 value,vec3 fallback)
{
    float squared=dot(value,value);
    return squared>DirectionTolerance && !isinf(squared) && !isnan(squared) ? value*inversesqrt(squared):fallback;
}
vec3 TowardEye(vec3 world)
{ return Eye.w>OrthographicThreshold ? -CameraForward.xyz:Unit(Eye.xyz-world,-CameraForward.xyz); }
vec3 RibbonSide(vec3 tangent,vec3 view)
{
    vec3 side=cross(tangent,view);
    if (dot(side,side)<=DirectionTolerance) side=cross(tangent,abs(tangent.y)<ParallelLimit ? vec3(0,1,0):vec3(1,0,0));
    return Unit(side,vec3(1,0,0));
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
layout(location=0) in vec3 InCenter;
layout(location=1) in vec3 InTangent;
layout(location=2) in vec4 InReflectance;
layout(location=3) in vec4 InEmission;
layout(location=4) in vec4 InRibbon;
out vec3 WorldPosition,FiberTangent,SourceReflectance,SourceEmission;
centroid out vec3 CrossHeight;
out float LinearShadowDepth;
out vec4 PreviousClip;
void main()
{
    vec3 center=InCenter+WindOffset(InCenter,InRibbon.z,Animation.x,Animation);
    FiberTangent=Unit(WindAxis(InCenter,InTangent,InRibbon.z,Animation.x,Animation),vec3(0,1,0));
    WorldPosition=center+RibbonSide(FiberTangent,TowardEye(center))*InRibbon.x;
    vec4 world=vec4(WorldPosition,1.0);
    vec4 clip=vec4(dot(ClipRows[0],world),dot(ClipRows[1],world),dot(ClipRows[2],world),dot(ClipRows[3],world));
    LinearShadowDepth=dot(ShadowDepth.xyz,vec3(clip.zw,1.0));
    vec3 oldCenter=InCenter+WindOffset(InCenter,InRibbon.z,Animation.y,Animation);
    vec3 oldAxis=Unit(WindAxis(InCenter,InTangent,InRibbon.z,Animation.y,Animation),vec3(0,1,0));
    vec3 oldView=PreviousEye.w>OrthographicThreshold ? -PreviousForward.xyz:Unit(PreviousEye.xyz-oldCenter,-PreviousForward.xyz);
    vec4 oldWorld=vec4(oldCenter+RibbonSide(oldAxis,oldView)*InRibbon.x,1.0);
    PreviousClip=vec4(dot(PreviousClipRows[0],oldWorld),dot(PreviousClipRows[1],oldWorld),dot(PreviousClipRows[2],oldWorld),dot(PreviousClipRows[3],oldWorld));
    clip.z=clip.z*2.0-clip.w; gl_Position=clip;
    SourceReflectance=InReflectance.rgb; SourceEmission=InEmission.rgb; CrossHeight=InRibbon.yzw;
}
#else
in vec3 WorldPosition,FiberTangent,SourceReflectance,SourceEmission;
// Avoid extrapolating the cylindrical coordinate outside subpixel coverage.
centroid in vec3 CrossHeight;
in float LinearShadowDepth;
in vec4 PreviousClip;
#if defined(EVANS_HAIR_VELOCITY)
layout(location=0) out vec4 OutVelocity;
void main()
{
    const float MinimumClipW=1e-7,PixelHalf=0.5;
    if (dot(vec4(WorldPosition,1.0),ClipPlane)<0.0) discard;
    OutVelocity=vec4(0.0);
    if (MotionOptions.x==0u || PreviousClip.w<=MinimumClipW) return;
    vec3 oldNdc=PreviousClip.xyz/PreviousClip.w;
    if (any(isnan(oldNdc)) || any(isinf(oldNdc)) || oldNdc.z<0.0 || oldNdc.z>1.0) return;
    vec2 oldPixel=(oldNdc.xy*vec2(PixelHalf,-PixelHalf)+PixelHalf)*MotionViewport.xy;
    vec2 currentPixel=vec2(gl_FragCoord.x,MotionViewport.y-gl_FragCoord.y);
    vec2 velocity=currentPixel-oldPixel-MotionViewport.zw;
    if (any(greaterThan(abs(velocity),vec2(HalfMaximum)))) return;
    OutVelocity=vec4(velocity,1.0-oldNdc.z,1.0);
}
#elif defined(EVANS_HAIR_VSM_SHADOW)
layout(location=0) out vec2 OutMoments;
void main()
{
    const float PixelHalfWidth=0.5;
    vec2 gradient=vec2(dFdx(LinearShadowDepth),dFdy(LinearShadowDepth))*PixelHalfWidth;
    float footprint=min(dot(gradient,gradient),ShadowDepth.w);
    if (Options.w!=0u && CrossHeight.z<0.0) discard;
    if (dot(vec4(WorldPosition,1.0),ClipPlane)<0.0) discard;
    float linearDepth=clamp(LinearShadowDepth,0.0,1.0);
    OutMoments=vec2(linearDepth,linearDepth*linearDepth+footprint);
}
#elif defined(EVANS_HAIR_DEPTH_ONLY)
void main() { if ((Options.w!=0u && CrossHeight.z<0.0) || dot(vec4(WorldPosition,1.0),ClipPlane)<0.0) discard; }
#else
struct LightSample { vec4 PositionKind,DirectionRange,Radiance; };
layout(std430,binding=1) readonly buffer HairLights { LightSample Lights[]; };
layout(location=0) out vec4 OutColor;
#ifndef EVANS_HAIR_CAPTURE
layout(location=1) out vec4 OutMaterialNormal;
#endif
float SafeRoot(float value) { return sqrt(max(value,0.0)); }
float PolarAngle(float y,float x) { return abs(y)+abs(x)>DirectionTolerance ? atan(y,x):0.0; }
float LogBessel(float value)
{
    value=abs(value);
    if (value>BesselThreshold)
    {
        float inverse=1.0/value,series=1.0+inverse*(BesselFirst+inverse*(BesselSecond+inverse*BesselThird));
        return value-log(FullTurn*value)/2.0+log(series);
    }
    float sum=1.0,term=1.0,argument=value*value/4.0;
    for (int order=1;order<BesselTerms;++order) { term*=argument/float(order*order); sum+=term; }
    return log(sum);
}
float Longitudinal(float incoming,float outgoing,float variance)
{
    float argument=SafeRoot(1.0-incoming*incoming)*SafeRoot(1.0-outgoing*outgoing)/variance;
    float exponent=incoming*outgoing/variance;
    float normalization=log(variance)+1.0/variance+log(1.0-exp(-2.0/variance));
    return exp(LogBessel(argument)-exponent-normalization);
}
float Azimuthal(float difference)
{
    difference-=FullTurn*floor((difference+Pi)/FullTurn);
    float value=exp(-abs(difference)/Optics.y),edge=exp(-Pi/Optics.y),mass=(1.0-edge)/(1.0+edge);
    return value/(Optics.y*(1.0+value)*(1.0+value)*mass);
}
float Fresnel(float cosine)
{
    float eta=Optics.x,transmitted=SafeRoot(1.0-(1.0-cosine*cosine)/(eta*eta));
    float parallel=(eta*cosine-transmitted)/(eta*cosine+transmitted);
    float perpendicular=(cosine-eta*transmitted)/(cosine+eta*transmitted);
    return (parallel*parallel+perpendicular*perpendicular)/2.0;
}
struct FiberState
{
    vec3 Attenuation[PathCount]; vec4 OutgoingSine;
    float OutgoingAzimuth,Gamma,TransmittedGamma;
};
FiberState PrepareFiber(vec3 color,float h,vec3 outgoing)
{
    FiberState result; float sine=clamp(outgoing.x,-1.0,1.0),cosine=SafeRoot(1.0-sine*sine),eta=Optics.x;
    h=clamp(h,-1.0,1.0); float inside=h*cosine/SafeRoot(eta*eta-sine*sine);
    float pathLength=2.0*SafeRoot(1.0-inside*inside)/SafeRoot(1.0-sine*sine/(eta*eta));
    vec3 absorption=log(clamp(color,vec3(ReflectanceFloor),vec3(1.0)))/Optics.z;
    vec3 transmittance=exp(-absorption*absorption*pathLength);
    float reflection=Fresnel(cosine*SafeRoot(1.0-h*h)); vec3 recurrence=transmittance*reflection;
    result.Attenuation[0]=vec3(reflection); result.Attenuation[1]=(1.0-reflection)*(1.0-reflection)*transmittance;
    result.Attenuation[2]=result.Attenuation[1]*recurrence;
    result.Attenuation[3]=result.Attenuation[2]*recurrence/max(vec3(1.0)-recurrence,vec3(TransmissionFloor));
    result.OutgoingSine=clamp(vec4(sine*CosTilt.y-cosine*SinTilt.y,sine*CosTilt.x+cosine*SinTilt.x,
        sine*CosTilt.z+cosine*SinTilt.z,sine),-1.0,1.0);
    result.OutgoingAzimuth=PolarAngle(outgoing.z,outgoing.y); result.Gamma=asin(h); result.TransmittedGamma=asin(clamp(inside,-1.0,1.0)); return result;
}
vec3 Scatter(FiberState fiber,vec3 incoming)
{
    vec3 result=vec3(0.0); float difference=PolarAngle(incoming.z,incoming.y)-fiber.OutgoingAzimuth;
    for (int order=0;order<PathCount;++order)
    {
        float center=2.0*float(order)*fiber.TransmittedGamma-2.0*fiber.Gamma+float(order)*Pi;
        float azimuth=order==PathCount-1 ? 1.0/FullTurn:Azimuthal(difference-center);
        result+=fiber.Attenuation[order]*(Longitudinal(clamp(incoming.x,-1.0,1.0),fiber.OutgoingSine[order],Variance[order])*azimuth);
    }
    return result;
}
// Field loads avoid a whole-structure SSBO copy in ES-to-D3D drivers.
vec3 Incident(uint index,vec3 position,out vec3 direction)
{
    vec4 positionKind=Lights[index].PositionKind,directionRange=Lights[index].DirectionRange,radiance=Lights[index].Radiance;
    direction=directionRange.xyz; if (uint(positionKind.w)==DirectionalKind) return radiance.rgb;
    vec3 offset=positionKind.xyz-position; float squared=dot(offset,offset);
    if (!(squared>0.0) || isinf(squared) || isnan(squared)) { direction=vec3(0.0); return vec3(0.0); }
    float distance=sqrt(squared),window=1.0; direction=offset/distance;
    if (directionRange.w>0.0)
    { float relative=distance/directionRange.w,relativeSquared=relative*relative; window=clamp(1.0-relativeSquared*relativeSquared,0.0,1.0); window*=window; }
    float emitter=uint(positionKind.w)==RectangleKind ? clamp(dot(directionRange.xyz,-direction),0.0,1.0):1.0;
    return radiance.rgb*(window*emitter/max(squared,PointDistanceFloor));
}
// Shadow references: Reeves et al. (1987), PCF; Fernando (2005), PCSS;
// Donnelly and Lauritzen (2006), Variance Shadow Maps. Filter visibility or
// linear moments, then apply the selected direct-light strength once.
// Canonical shadow coordinates have a top-left origin. ES storage is bottom-up.
const uint SoftShadowMode=2u,VarianceShadowMode=3u;
const float MinimumShadowW=1e-7,MaximumFilterFraction=0.25;
uniform highp sampler2D FiberMap0,FiberMap1,FiberMap2,FiberMap3,FiberMap4;
uniform highp sampler2DArray FiberContactLayers;
uniform highp int FiberContactSample;
vec2 ShadowTexel(uint map,ivec2 pixel)
{
    ivec2 extent=ivec2(ShadowExtents[map].xy);
    if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,extent))) return vec2(1.0);
    pixel.y=extent.y-1-pixel.y;
    if (map==0u) return texelFetch(FiberMap0,pixel,0).rg;
    if (map==1u) return texelFetch(FiberMap1,pixel,0).rg;
    if (map==2u) return texelFetch(FiberMap2,pixel,0).rg;
    if (map==3u) return texelFetch(FiberMap3,pixel,0).rg;
    return texelFetch(FiberMap4,pixel,0).rg;
}
float CompareShadow(uint map,vec2 uv,float depth)
{
    vec2 pixel=uv*ShadowExtents[map].xy-0.5,f=fract(pixel); ivec2 base=ivec2(floor(pixel));
    vec4 values=vec4(ShadowTexel(map,base).x,ShadowTexel(map,base+ivec2(1,0)).x,
        ShadowTexel(map,base+ivec2(0,1)).x,ShadowTexel(map,base+1).x);
    vec4 lit=step(vec4(depth),values);
    return mix(mix(lit.x,lit.y,f.x),mix(lit.z,lit.w,f.x),f.y);
}
vec2 FilterMoments(uint map,vec2 uv)
{
    vec2 pixel=uv*ShadowExtents[map].xy-0.5,f=fract(pixel); ivec2 base=ivec2(floor(pixel));
    return mix(mix(ShadowTexel(map,base),ShadowTexel(map,base+ivec2(1,0)),f.x),
        mix(ShadowTexel(map,base+ivec2(0,1)),ShadowTexel(map,base+1),f.x),f.y);
}
float LightAxisDistance(float deviceDepth,vec4 metric)
{ return metric.x>0.0 ? 1.0/((1.0-deviceDepth)/metric.x+deviceDepth/metric.y):deviceDepth*metric.y; }
vec2 DiskOffset(uint index,uint count)
{
    float angle=float(index)*Pi*(3.0-sqrt(5.0));
    return sqrt((float(index)+0.5)/float(count))*vec2(cos(angle),sin(angle));
}
vec2 LimitShadowRadius(uint map,float worldRadius,vec4 metric,float budget)
{
    vec2 radius=worldRadius/metric.zw;
    float footprint=max(radius.x*ShadowExtents[map].x,radius.y*ShadowExtents[map].y);
    float limit=min(budget,min(ShadowExtents[map].x,ShadowExtents[map].y)*MaximumFilterFraction);
    return footprint>limit ? radius*(limit/footprint):radius;
}
vec4 ShadowPosition(uint map,vec3 world)
{
    vec4 p=vec4(world,1.0);
    if (map==0u) return vec4(dot(ShadowRows[0],p),dot(ShadowRows[1],p),dot(ShadowRows[2],p),dot(ShadowRows[3],p));
    // Keep the unused cascade address valid when drivers specialize map zero.
    // Some linkers fold its unsigned subtraction before removing this branch.
    uint row=(max(map,1u)-1u)*4u;
    return vec4(dot(ShadowCascadeRows[row],p),dot(ShadowCascadeRows[row+1u],p),dot(ShadowCascadeRows[row+2u],p),dot(ShadowCascadeRows[row+3u],p));
}
float MapVisibility(uint map,vec3 world,float bias)
{
    vec4 clip=ShadowPosition(map,world);
    if ((ShadowControl.y&(1u<<map))==0u || clip.w<=MinimumShadowW) return 1.0;
    vec3 ndc=clip.xyz/clip.w; vec2 uv=ndc.xy*vec2(0.5,-0.5)+0.5;
    if (any(lessThan(uv,vec2(0))) || any(greaterThan(uv,vec2(1))) || ndc.z<0.0 || ndc.z>1.0) return 1.0;
    float receiverDepth=ndc.z-bias;
    if (ShadowControl.x==VarianceShadowMode)
    {
        float linearDepth=dot(ShadowVarianceEncoding[map].xyz,vec3(clip.zw,1)); vec2 moments=FilterMoments(map,uv);
        float delta=linearDepth-bias-moments.x; if (delta<=0.0) return 1.0;
        float variance=max(moments.y-moments.x*moments.x,ShadowVarianceParams.x);
        return clamp((variance/(variance+delta*delta)-ShadowVarianceParams.y)/(1.0-ShadowVarianceParams.y),0.0,1.0);
    }
    if (ShadowControl.x==SoftShadowMode)
    {
        vec4 metric=ShadowSoftMetrics[map]; float emitter=metric.x>0.0?ShadowEmitter.x:ShadowEmitter.y;
        if (emitter<=0.0) return step(receiverDepth,ShadowTexel(map,ivec2(floor(uv*ShadowExtents[map].xy))).x);
        float distance=LightAxisDistance(ndc.z,metric),total=0.0; uint blockers=0u;
        float searchWorld=metric.x>0.0?emitter*(distance-metric.x)/distance:emitter*distance;
        vec2 search=LimitShadowRadius(map,searchWorld,metric,ShadowEmitter.z);
        for (uint sampleIndex=0u;sampleIndex<ShadowSoftControl.x;++sampleIndex)
        {
            vec2 location=uv+search*DiskOffset(sampleIndex,ShadowSoftControl.x);
            float stored=ShadowTexel(map,ivec2(floor(location*ShadowExtents[map].xy))).x;
            if (stored<receiverDepth) { total+=LightAxisDistance(stored,metric); ++blockers; }
        }
        if (blockers==0u) return 1.0;
        float blocker=total/float(blockers),gap=max(distance-blocker,0.0);
        float width=metric.x>0.0?emitter*gap/blocker*metric.x/distance:emitter*gap;
        vec2 radius=LimitShadowRadius(map,width,metric,ShadowEmitter.w); float visible=0.0;
        for (uint sampleIndex=0u;sampleIndex<ShadowSoftControl.y;++sampleIndex)
            visible+=CompareShadow(map,uv+radius*DiskOffset(sampleIndex,ShadowSoftControl.y),receiverDepth);
        return visible/float(ShadowSoftControl.y);
    }
    if (ShadowFiltering.x==0.0) return step(receiverDepth,ShadowTexel(map,ivec2(floor(uv*ShadowExtents[map].xy))).x);
    int radius=int(ShadowFiltering.x); float visibility=0.0;
    for (int y=-radius;y<=radius;++y) for (int x=-radius;x<=radius;++x)
        visibility+=CompareShadow(map,uv+vec2(x,y)*ShadowExtents[map].zw,receiverDepth);
    return visibility/ShadowFiltering.y;
}
float FiberVisibility(vec3 world,vec3 normal,vec3 incoming)
{
    if (ShadowControl.z==0u) return 1.0;
    float cosine=clamp(abs(dot(normal,incoming)),0.0,1.0);
    float bias=min(ShadowBias.z,ShadowBias.x+ShadowBias.y*SafeRoot(1.0-cosine*cosine)/max(cosine,ShadowBias.w));
    uint count=uint(ShadowCascadeParams.z); float visibility=1.0;
    if (count==0u) visibility=MapVisibility(0u,world,bias);
    else
    {
        float distance=dot(ShadowEyePlane,vec4(world,1));
        if (distance>=ShadowCascadeParams.x && distance<=ShadowCascadeParams.y)
        {
            uint cascade=0u; while (cascade+1u<count && distance>ShadowFar[cascade]) ++cascade;
            visibility=MapVisibility(cascade+1u,world,bias); float interval=ShadowFar[cascade]-ShadowBlend[cascade];
            if (interval>0.0 && distance>=ShadowBlend[cascade])
            {
                float next=cascade+1u<count?MapVisibility(cascade+2u,world,bias):1.0;
                visibility=mix(visibility,next,clamp((distance-ShadowBlend[cascade])/interval,0.0,1.0));
            }
        }
    }
    return visibility;
}
void main()
{
    if (dot(vec4(WorldPosition,1.0),ClipPlane)<0.0) discard;
    vec3 tangent=Unit(FiberTangent,vec3(0,1,0)),view=TowardEye(WorldPosition),side=RibbonSide(tangent,view);
    vec3 normal=Unit(cross(side,tangent),-CameraForward.xyz);
    vec3 color=clamp(SourceReflectance*mix(RootTint.rgb,TipTint.rgb,clamp(CrossHeight.y,0.0,1.0)),0.0,1.0);
    FiberState fiber=PrepareFiber(color,CrossHeight.x,vec3(dot(view,tangent),dot(view,side),dot(view,normal)));
    vec3 radiance=color*Ambient.rgb*(MotionOptions.y!=0u?0.0:Ambient.w);
    float h=clamp(CrossHeight.x,-1.0,1.0); vec3 cylinderNormal=side*h+normal*SafeRoot(1.0-h*h);
    for (uint index=0u;index<Options.x;++index)
    {
        vec3 direction; vec3 incident=Incident(index,WorldPosition,direction);
#ifndef EVANS_HAIR_CAPTURE
        if (index>=Options.y && index<Options.z)
        {
            float visibility=FiberVisibility(WorldPosition,cylinderNormal,direction);
            if (ContactControl.x!=0u) visibility*=texelFetch(FiberContactLayers,ivec3(ivec2(gl_FragCoord.xy),FiberContactSample),0).r;
            incident*=1.0-ShadowComposition.x*(1.0-visibility);
        }
#endif
        radiance+=incident*Scatter(fiber,vec3(dot(direction,tangent),dot(direction,side),dot(direction,normal)));
    }
    OutColor=vec4(clamp(SourceEmission+radiance*Optics.w,0.0,HalfMaximum),1.0);
#ifndef EVANS_HAIR_CAPTURE
    OutMaterialNormal=vec4(0.0,0.0,0.0,-1.0);
#endif
}
#endif
#endif
