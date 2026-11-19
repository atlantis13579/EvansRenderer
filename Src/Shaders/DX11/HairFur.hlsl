// Independent DX11 cylindrical fiber scattering and camera-facing ribbons.
// d'Eon/Marschner longitudinal + azimuthal model, with R/TT/TRT and a tail:
// https://pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair
// S already includes the BSDF's cancelling surface cosine. Root data is linear.
static const float Pi=3.141592653589793,TwoPi=2*Pi;
static const float DirectionTolerance=1e-12,MinimumReflectance=1e-4,MinimumDenominator=1e-6;
static const float BesselThreshold=12,MaximumRadiance=65504,LocalDistanceSquaredFloor=1e-6;
static const float BesselFirst=1.0/8,BesselSecond=9.0/128,BesselThird=225.0/3072;
static const float ParallelLimit=0.9,OrthographicThreshold=0.5;
static const uint BesselTerms=20,DirectionalKind=0,RectangleKind=2,ScatteringOrders=4;
cbuffer HairFrame:register(b0)
{
    row_major float4x4 ViewProjection;
    float4 Eye,CameraForward,Variance,SinTilt,CosTilt,Optics;
    float4 RootTint,TipTint,Ambient,ClipPlane;
    uint4 Options;
    float4 ShadowDepth;
    row_major float4x4 PreviousViewProjection;
    float4 PreviousEye,PreviousForward,MotionViewport;
    uint4 MotionOptions;
    row_major float4x4 ShadowProjection,ShadowCascadeProjection[4];
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
struct LightSample { float4 PositionKind,DirectionRange,Radiance; };
StructuredBuffer<LightSample> Lights:register(t0);
struct HairVertex
{
    float3 Center:POSITION; float3 Tangent:TANGENT;
    float4 Reflectance:COLOR0; float4 Emission:COLOR1; float4 Ribbon:TEXCOORD0;
};
struct HairFragment
{
    precise float4 Position:SV_POSITION; float3 World:TEXCOORD0; float3 Tangent:TEXCOORD1;
    float3 Reflectance:TEXCOORD2; float3 Emission:TEXCOORD3;
    // Keep subpixel MSAA ribbons inside their cylindrical cross section.
    centroid float3 CrossHeight:TEXCOORD4;
    float LinearShadowDepth:TEXCOORD5;
    float4 PreviousClip:TEXCOORD6;
};
float3 Unit(float3 value,float3 fallback)
{ float squared=dot(value,value); return squared>DirectionTolerance && isfinite(squared) ? value*rsqrt(squared):fallback; }
float3 ViewDirection(float3 world)
{ return Eye.w>OrthographicThreshold ? -CameraForward.xyz:Unit(Eye.xyz-world,-CameraForward.xyz); }
float3 RibbonSide(float3 tangent,float3 view)
{
    float3 side=cross(tangent,view);
    if (dot(side,side)<=DirectionTolerance)
        side=cross(tangent,abs(tangent.y)<ParallelLimit ? float3(0,1,0):float3(1,0,0));
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
HairFragment VSMain(HairVertex vertex)
{
    HairFragment result; float3 center=vertex.Center+WindOffset(vertex.Center,vertex.Ribbon.z,Animation.x,Animation);
    float3 tangent=Unit(WindAxis(vertex.Center,vertex.Tangent,vertex.Ribbon.z,Animation.x,Animation),float3(0,1,0));
    result.World=center+RibbonSide(tangent,ViewDirection(center))*vertex.Ribbon.x;
    result.Position=mul(ViewProjection,float4(result.World,1)); result.Tangent=tangent;
    result.LinearShadowDepth=dot(ShadowDepth.xyz,float3(result.Position.zw,1));
    float3 oldCenter=vertex.Center+WindOffset(vertex.Center,vertex.Ribbon.z,Animation.y,Animation);
    float3 oldAxis=Unit(WindAxis(vertex.Center,vertex.Tangent,vertex.Ribbon.z,Animation.y,Animation),float3(0,1,0));
    float3 previousView=PreviousEye.w>OrthographicThreshold ? -PreviousForward.xyz:Unit(PreviousEye.xyz-oldCenter,-PreviousForward.xyz);
    float3 previousWorld=oldCenter+RibbonSide(oldAxis,previousView)*vertex.Ribbon.x;
    result.PreviousClip=mul(PreviousViewProjection,float4(previousWorld,1));
    result.Reflectance=vertex.Reflectance.rgb; result.Emission=vertex.Emission.rgb; result.CrossHeight=vertex.Ribbon.yzw; return result;
}
#if defined(EVANS_HAIR_VELOCITY)
// Previous homogeneous positions are interpolated before the perspective divide.
// The ribbon itself rotates with the camera, even when its centerline is static.
float4 PSMain(HairFragment fragment):SV_TARGET0
{
    const float ClipWEpsilon=1e-7,MaximumVelocity=65504,PixelHalf=0.5;
    if (dot(float4(fragment.World,1),ClipPlane)<0) discard;
    if (MotionOptions.x==0 || fragment.PreviousClip.w<=ClipWEpsilon) return 0;
    float3 previous=fragment.PreviousClip.xyz/fragment.PreviousClip.w;
    if (!all(isfinite(previous)) || previous.z<0 || previous.z>1) return 0;
    float2 previousPixel=(previous.xy*float2(PixelHalf,-PixelHalf)+PixelHalf)*MotionViewport.xy;
    float2 displacement=fragment.Position.xy-previousPixel-MotionViewport.zw;
    if (any(abs(displacement)>MaximumVelocity)) return 0;
    return float4(displacement,1-previous.z,1);
}
#elif defined(EVANS_HAIR_VSM_SHADOW)
// Primitive-local footprint variance follows Donnelly/Lauritzen, VSM (2006).
float2 PSMain(HairFragment fragment):SV_TARGET0
{
    const float FootprintHalfWidth=0.5;
    float2 derivative=float2(ddx(fragment.LinearShadowDepth),ddy(fragment.LinearShadowDepth))*FootprintHalfWidth;
    float footprint=min(dot(derivative,derivative),ShadowDepth.w);
    if (Options.w!=0 && fragment.CrossHeight.z<0) discard;
    if (dot(float4(fragment.World,1),ClipPlane)<0) discard;
    float depth=saturate(fragment.LinearShadowDepth); return float2(depth,depth*depth+footprint);
}
#elif defined(EVANS_HAIR_DEPTH_ONLY)
void PSMain(HairFragment fragment)
{ if ((Options.w!=0 && fragment.CrossHeight.z<0) || dot(float4(fragment.World,1),ClipPlane)<0) discard; }
#else
float SafeSqrt(float value) { return sqrt(max(value,0)); }
float Azimuth(float y,float x) { return abs(y)+abs(x)>DirectionTolerance ? atan2(y,x):0; }
float LogBessel(float argument)
{
    argument=abs(argument);
    if (argument>BesselThreshold)
    {
        float inverse=1/argument,series=1+inverse*(BesselFirst+inverse*(BesselSecond+inverse*BesselThird));
        return argument-log(TwoPi*argument)/2+log(series);
    }
    float sum=1,term=1,quarterSquare=argument*argument/4;
    [unroll] for (uint order=1;order<BesselTerms;++order) { term*=quarterSquare/(order*order); sum+=term; }
    return log(sum);
}
float Longitudinal(float incomingSine,float outgoingSine,float variance)
{
    float a=SafeSqrt(1-incomingSine*incomingSine)*SafeSqrt(1-outgoingSine*outgoingSine)/variance;
    float b=incomingSine*outgoingSine/variance;
    float denominator=log(variance)+1/variance+log(1-exp(-2/variance));
    return exp(LogBessel(a)-b-denominator);
}
float Azimuthal(float difference)
{
    difference-=TwoPi*floor((difference+Pi)/TwoPi);
    float value=exp(-abs(difference)/Optics.y),edge=exp(-Pi/Optics.y);
    float normalization=(1-edge)/(1+edge);
    return value/(Optics.y*(1+value)*(1+value)*normalization);
}
float Fresnel(float cosine)
{
    float eta=Optics.x,transmitted=SafeSqrt(1-(1-cosine*cosine)/(eta*eta));
    float parallel=(eta*cosine-transmitted)/(eta*cosine+transmitted);
    float perpendicular=(cosine-eta*transmitted)/(cosine+eta*transmitted);
    return (parallel*parallel+perpendicular*perpendicular)/2;
}
struct FiberState
{
    float3 Attenuation[ScatteringOrders]; float4 ViewSine;
    float Phi,Gamma,GammaTransmitted;
};
FiberState PrepareFiber(float3 reflectance,float h,float3 outgoing)
{
    FiberState fiber; float sine=clamp(outgoing.x,-1,1),cosine=SafeSqrt(1-sine*sine),eta=Optics.x;
    h=clamp(h,-1,1); float transmittedSine=h*cosine/SafeSqrt(eta*eta-sine*sine);
    float distance=2*SafeSqrt(1-transmittedSine*transmittedSine)/SafeSqrt(1-sine*sine/(eta*eta));
    float3 absorption=log(clamp(reflectance,MinimumReflectance,1))/Optics.z;
    float3 transmission=exp(-absorption*absorption*distance);
    float reflection=Fresnel(cosine*SafeSqrt(1-h*h)); float3 internal=reflection*transmission;
    fiber.Attenuation[0]=reflection;
    fiber.Attenuation[1]=(1-reflection)*(1-reflection)*transmission;
    fiber.Attenuation[2]=fiber.Attenuation[1]*internal;
    fiber.Attenuation[3]=fiber.Attenuation[2]*internal/max(1-internal,MinimumDenominator);
    fiber.ViewSine=float4(sine*CosTilt.y-cosine*SinTilt.y,sine*CosTilt.x+cosine*SinTilt.x,
        sine*CosTilt.z+cosine*SinTilt.z,sine);
    fiber.ViewSine=clamp(fiber.ViewSine,-1,1); fiber.Phi=Azimuth(outgoing.z,outgoing.y);
    fiber.Gamma=asin(h); fiber.GammaTransmitted=asin(clamp(transmittedSine,-1,1)); return fiber;
}
float3 Scatter(FiberState fiber,float3 incoming)
{
    float difference=Azimuth(incoming.z,incoming.y)-fiber.Phi; float3 value=0;
    [unroll] for (uint order=0;order<ScatteringOrders;++order)
    {
        float phase=2*order*fiber.GammaTransmitted-2*fiber.Gamma+order*Pi;
        float azimuth=order==ScatteringOrders-1 ? 1/TwoPi:Azimuthal(difference-phase);
        value+=fiber.Attenuation[order]*(Longitudinal(clamp(incoming.x,-1,1),fiber.ViewSine[order],Variance[order])*azimuth);
    }
    return value;
}
float3 IncidentLight(LightSample light,float3 world,out float3 direction)
{
    direction=light.DirectionRange.xyz;
    if (uint(light.PositionKind.w)==DirectionalKind) return light.Radiance.rgb;
    float3 offset=light.PositionKind.xyz-world; float squared=dot(offset,offset);
    if (!(squared>0) || !isfinite(squared)) { direction=0; return 0; }
    float distance=sqrt(squared); direction=offset/distance; float window=1;
    if (light.DirectionRange.w>0)
    { float relative=distance/light.DirectionRange.w,relativeSquared=relative*relative; window=saturate(1-relativeSquared*relativeSquared); window*=window; }
    float emitter=uint(light.PositionKind.w)==RectangleKind ? saturate(dot(light.DirectionRange.xyz,-direction)):1;
    return light.Radiance.rgb*(window*emitter/max(squared,LocalDistanceSquaredFloor));
}
// Standalone Hard/PCF, Fernando PCSS (2005), Donnelly/Lauritzen VSM (2006)
// and split/overlap selection. All inputs use canonical top-left shadow pixels.
Texture2D<float4> HairShadow0:register(t1),HairShadow1:register(t2),HairShadow2:register(t3),HairShadow3:register(t4),HairShadow4:register(t5);
// Shadow references: Reeves et al. (1987), PCF; Fernando (2005), PCSS;
// Donnelly and Lauritzen (2006), Variance Shadow Maps. Filter visibility or
// linear moments, then apply the selected direct-light strength once.
float2 HairMapAt(uint map,int2 pixel)
{
    int2 extent=int2(ShadowExtents[map].xy);
    if (any(pixel<0) || any(pixel>=extent)) return 1;
    if (map==0) return HairShadow0.Load(int3(pixel,0)).xy;
    if (map==1) return HairShadow1.Load(int3(pixel,0)).xy;
    if (map==2) return HairShadow2.Load(int3(pixel,0)).xy;
    if (map==3) return HairShadow3.Load(int3(pixel,0)).xy;
    return HairShadow4.Load(int3(pixel,0)).xy;
}
float HairCompare(uint map,float2 uv,float receiver)
{
    float2 texel=uv*ShadowExtents[map].xy-0.5,weight=frac(texel); int2 p=int2(floor(texel));
    float a=HairMapAt(map,p).x>=receiver,b=HairMapAt(map,p+int2(1,0)).x>=receiver;
    float c=HairMapAt(map,p+int2(0,1)).x>=receiver,d=HairMapAt(map,p+1).x>=receiver;
    return lerp(lerp(a,b,weight.x),lerp(c,d,weight.x),weight.y);
}
float2 HairMoments(uint map,float2 uv)
{
    float2 texel=uv*ShadowExtents[map].xy-0.5,weight=frac(texel); int2 p=int2(floor(texel));
    return lerp(lerp(HairMapAt(map,p),HairMapAt(map,p+int2(1,0)),weight.x),
        lerp(HairMapAt(map,p+int2(0,1)),HairMapAt(map,p+1),weight.x),weight.y);
}
float HairDistance(float depth,float4 metric)
{ return metric.x>0 ? rcp((1-depth)/metric.x+depth/metric.y):depth*metric.y; }
float2 HairDisk(uint index,uint count)
{
    const float GoldenAngle=Pi*(3-sqrt(5.0)); float angle=index*GoldenAngle;
    return sqrt((index+0.5)/count)*float2(cos(angle),sin(angle));
}
float2 HairRadius(uint map,float radius,float4 metric,float cap)
{
    const float MaximumMapFraction=0.25;
    float2 uv=radius/metric.zw; float texels=max(uv.x*ShadowExtents[map].x,uv.y*ShadowExtents[map].y);
    float limit=min(cap,min(ShadowExtents[map].x,ShadowExtents[map].y)*MaximumMapFraction);
    return texels>limit ? uv*(limit/texels):uv;
}
static const uint HairSoftShadowMode=2,HairVarianceShadowMode=3;
float HairMapVisibility(uint map,float4 clip,float bias)
{
    const float MinimumClipW=1e-7;
    if ((ShadowControl.y&(1u<<map))==0 || clip.w<=MinimumClipW) return 1;
    float3 projected=clip.xyz/clip.w; float2 uv=projected.xy*float2(0.5,-0.5)+0.5;
    if (any(uv<0) || any(uv>1) || projected.z<0 || projected.z>1) return 1;
    float comparison=projected.z-bias;
    if (ShadowControl.x==HairVarianceShadowMode)
    {
        float depth=dot(ShadowVarianceEncoding[map].xyz,float3(clip.zw,1)); float2 moments=HairMoments(map,uv);
        float difference=depth-bias-moments.x; if (difference<=0) return 1;
        float variance=max(moments.y-moments.x*moments.x,ShadowVarianceParams.x);
        float bound=variance/(variance+difference*difference);
        return saturate((bound-ShadowVarianceParams.y)/(1-ShadowVarianceParams.y));
    }
    if (ShadowControl.x==HairSoftShadowMode)
    {
        float4 metric=ShadowSoftMetrics[map]; float emitter=metric.x>0?ShadowEmitter.x:ShadowEmitter.y;
        if (emitter<=0) return HairMapAt(map,int2(floor(uv*ShadowExtents[map].xy))).x>=comparison;
        float receiver=HairDistance(projected.z,metric);
        float searchWorld=metric.x>0?emitter*(receiver-metric.x)/receiver:emitter*receiver;
        float2 search=HairRadius(map,searchWorld,metric,ShadowEmitter.z); float sum=0; uint blockers=0;
        [loop] for (uint index=0;index<ShadowSoftControl.x;++index)
        {
            float2 sampleUV=uv+HairDisk(index,ShadowSoftControl.x)*search;
            float stored=HairMapAt(map,int2(floor(sampleUV*ShadowExtents[map].xy))).x;
            if (stored<comparison) { sum+=HairDistance(stored,metric); ++blockers; }
        }
        if (blockers==0) return 1;
        float blocker=sum/blockers,gap=max(receiver-blocker,0);
        float penumbra=metric.x>0?emitter*gap/blocker*metric.x/receiver:emitter*gap;
        float2 radius=HairRadius(map,penumbra,metric,ShadowEmitter.w); float visible=0;
        [loop] for (uint filterIndex=0;filterIndex<ShadowSoftControl.y;++filterIndex) visible+=HairCompare(map,uv+HairDisk(filterIndex,ShadowSoftControl.y)*radius,comparison);
        return visible/ShadowSoftControl.y;
    }
    if (ShadowFiltering.x==0) return HairMapAt(map,int2(floor(uv*ShadowExtents[map].xy))).x>=comparison;
    float visible=0; int radius=int(ShadowFiltering.x);
    [loop] for (int y=-radius;y<=radius;++y) [loop] for (int x=-radius;x<=radius;++x)
        visible+=HairCompare(map,uv+float2(x,y)*ShadowExtents[map].zw,comparison);
    return visible/ShadowFiltering.y;
}
#ifdef EVANS_HAIR_CONTACT_RECEIVER
Texture2DArray<float> FiberContactMask:register(t6);
float HairContact(float4 pixel,uint layer)
{ return ContactControl.x!=0 && layer<ContactControl.y?FiberContactMask.Load(int4(int2(pixel.xy),layer,0)):1; }
#endif
float HairVisibility(float3 world,float3 cylinderNormal,float3 incoming)
{
    if (ShadowControl.z==0) return 1;
    float cosine=saturate(abs(dot(cylinderNormal,incoming))),slope=SafeSqrt(1-cosine*cosine)/max(cosine,ShadowBias.w);
    float bias=min(ShadowBias.z,ShadowBias.x+ShadowBias.y*slope),visibility=1;
    uint count=uint(ShadowCascadeParams.z);
    if (count==0) visibility=HairMapVisibility(0,mul(ShadowProjection,float4(world,1)),bias);
    else
    {
        float distance=dot(ShadowEyePlane,float4(world,1));
        if (distance>=ShadowCascadeParams.x && distance<=ShadowCascadeParams.y)
        {
            uint cascade=0; [loop] while (cascade+1<count && distance>ShadowFar[cascade]) ++cascade;
            visibility=HairMapVisibility(cascade+1,mul(ShadowCascadeProjection[cascade],float4(world,1)),bias);
            float interval=ShadowFar[cascade]-ShadowBlend[cascade];
            if (interval>0 && distance>=ShadowBlend[cascade])
            {
                float next=1; if (cascade+1<count) next=HairMapVisibility(cascade+2,mul(ShadowCascadeProjection[cascade+1],float4(world,1)),bias);
                visibility=lerp(visibility,next,saturate((distance-ShadowBlend[cascade])/interval));
            }
        }
    }
    return visibility;
}
struct HairOutput
{
    float4 Color:SV_TARGET0;
#if !defined(EVANS_HAIR_CAPTURE)
    float4 MaterialNormal:SV_TARGET1;
#endif
};
HairOutput PSMain(HairFragment fragment
#ifdef EVANS_HAIR_CONTACT_RECEIVER
    ,uint contactSample:SV_SampleIndex
#endif
    )
{
    if (dot(float4(fragment.World,1),ClipPlane)<0) discard;
    float3 tangent=Unit(fragment.Tangent,float3(0,1,0)),view=ViewDirection(fragment.World),side=RibbonSide(tangent,view);
    float3 normal=Unit(cross(side,tangent),-CameraForward.xyz),outgoing=float3(dot(view,tangent),dot(view,side),dot(view,normal));
    float3 reflectance=saturate(fragment.Reflectance*lerp(RootTint.rgb,TipTint.rgb,saturate(fragment.CrossHeight.y)));
    FiberState fiber=PrepareFiber(reflectance,fragment.CrossHeight.x,outgoing);
    float3 radiance=reflectance*Ambient.rgb*(MotionOptions.y!=0?0:Ambient.w);
    float h=clamp(fragment.CrossHeight.x,-1,1); float3 cylinderNormal=side*h+normal*SafeSqrt(1-h*h);
    [loop] for (uint index=0;index<Options.x;++index)
    {
        float3 direction; float3 incident=IncidentLight(Lights[index],fragment.World,direction);
#if !defined(EVANS_HAIR_CAPTURE)
        if (index>=Options.y && index<Options.z)
        {
            float visibility=HairVisibility(fragment.World,cylinderNormal,direction);
#ifdef EVANS_HAIR_CONTACT_RECEIVER
            visibility*=HairContact(fragment.Position,contactSample);
#endif
            incident*=1-ShadowComposition.x*(1-visibility);
        }
#endif
        float3 local=float3(dot(direction,tangent),dot(direction,side),dot(direction,normal));
        radiance+=incident*Scatter(fiber,local);
    }
    HairOutput result; result.Color=float4(clamp(fragment.Emission+radiance*Optics.w,0,MaximumRadiance),1);
#if !defined(EVANS_HAIR_CAPTURE)
    result.MaterialNormal=float4(0,0,0,-1); // Fiber lighting is not mesh diffuse.
#endif
    return result;
}
#endif
