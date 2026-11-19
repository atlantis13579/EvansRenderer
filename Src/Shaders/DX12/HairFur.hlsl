// DX12 owns its ribbon projection and cylindrical scattering implementation.
// R/TT/TRT and the higher-order tail use normalized angular distributions.
// Model and fitted profile parameters: https://pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair
static const float Pi=3.141592653589793,FullTurn=2*Pi;
static const float SmallDirectionSquared=1e-12,ReflectanceFloor=1e-4,TransmissionFloor=1e-6;
static const float HalfCeiling=65504,PointDistanceFloor=1e-6,LargeBesselArgument=12;
static const float ExpansionFirst=1.0/8,ExpansionSecond=9.0/128,ExpansionThird=225.0/3072;
static const float ParallelAxisLimit=0.9,OrthographicView=0.5;
static const uint BesselSeriesCount=20,PathCount=4,DirectionalSource=0,RectangleSource=2;
cbuffer FiberDraw:register(b0)
{
    row_major float4x4 ClipFromWorld;
    float4 ViewOrigin,ViewForward,PathVariance,TiltSine,TiltCosine,OpticalParameters;
    float4 RootColor,TipColor,AmbientLight,ClipEquation;
    uint4 LightControl;
    float4 ShadowMetric;
    row_major float4x4 PreviousClipFromWorld;
    float4 PreviousOrigin,PreviousForward,MotionExtent;
    uint4 MotionControl;
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
struct FiberLight { float4 PositionKind,DirectionRange,Radiance; };
StructuredBuffer<FiberLight> FiberLights:register(t0);
struct FiberVertex
{
    float3 Center:POSITION; float3 Axis:TANGENT;
    float4 Color:COLOR0; float4 Emitted:COLOR1; float4 Shape:TEXCOORD0;
};
struct RasterFiber
{
    precise float4 Position:SV_POSITION; float3 World:TEXCOORD0; float3 Axis:TEXCOORD1;
    float3 Color:TEXCOORD2; float3 Emitted:TEXCOORD3;
    // Pixel centers can lie outside thin MSAA ribbons; do not extrapolate h.
    centroid float3 Parameters:TEXCOORD4;
    float LinearDepth:TEXCOORD5;
    float4 OldClip:TEXCOORD6;
};
float3 NormalizeOr(float3 value,float3 fallback)
{
    float squared=dot(value,value); return squared>SmallDirectionSquared && isfinite(squared) ? value*rsqrt(squared):fallback;
}
float3 TowardEye(float3 position)
{ return ViewOrigin.w>OrthographicView ? -ViewForward.xyz:NormalizeOr(ViewOrigin.xyz-position,-ViewForward.xyz); }
float3 WidthAxis(float3 tangent,float3 towardEye)
{
    float3 axis=cross(tangent,towardEye);
    if (dot(axis,axis)<=SmallDirectionSquared)
    { float3 helper=abs(tangent.y)<ParallelAxisLimit ? float3(0,1,0):float3(1,0,0); axis=cross(tangent,helper); }
    return NormalizeOr(axis,float3(1,0,0));
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
RasterFiber VSMain(FiberVertex input)
{
    RasterFiber result; float3 center=input.Center+WindOffset(input.Center,input.Shape.z,Animation.x,Animation);
    result.Axis=NormalizeOr(WindAxis(input.Center,input.Axis,input.Shape.z,Animation.x,Animation),float3(0,1,0));
    result.World=center+WidthAxis(result.Axis,TowardEye(center))*input.Shape.x;
    result.Position=mul(ClipFromWorld,float4(result.World,1)); result.Color=input.Color.rgb;
    result.LinearDepth=dot(ShadowMetric.xyz,float3(result.Position.zw,1));
    float3 oldCenter=input.Center+WindOffset(input.Center,input.Shape.z,Animation.y,Animation);
    float3 oldAxis=NormalizeOr(WindAxis(input.Center,input.Axis,input.Shape.z,Animation.y,Animation),float3(0,1,0));
    float3 oldView=PreviousOrigin.w>OrthographicView ? -PreviousForward.xyz:NormalizeOr(PreviousOrigin.xyz-oldCenter,-PreviousForward.xyz);
    float3 oldWorld=oldCenter+WidthAxis(oldAxis,oldView)*input.Shape.x;
    result.OldClip=mul(PreviousClipFromWorld,float4(oldWorld,1));
    result.Emitted=input.Emitted.rgb; result.Parameters=input.Shape.yzw; return result;
}
#if defined(EVANS_HAIR_VELOCITY)
float4 PSMain(RasterFiber input):SV_TARGET0
{
    const float MinimumHomogeneousW=1e-7,VelocityCeiling=65504,PixelHalf=0.5;
    if (dot(float4(input.World,1),ClipEquation)<0) discard;
    if (MotionControl.x==0 || input.OldClip.w<=MinimumHomogeneousW) return 0;
    float3 oldNdc=input.OldClip.xyz/input.OldClip.w;
    if (!all(isfinite(oldNdc)) || oldNdc.z<0 || oldNdc.z>1) return 0;
    float2 oldPixel=(oldNdc.xy*float2(PixelHalf,-PixelHalf)+PixelHalf)*MotionExtent.xy;
    float2 velocity=input.Position.xy-oldPixel-MotionExtent.zw;
    if (any(abs(velocity)>VelocityCeiling)) return 0;
    return float4(velocity,1-oldNdc.z,1);
}
#elif defined(EVANS_HAIR_VSM_SHADOW)
// VSM caster moments include the local raster footprint, before clipping.
float2 PSMain(RasterFiber input):SV_TARGET0
{
    const float HalfFootprint=0.5;
    float2 gradient=float2(ddx(input.LinearDepth),ddy(input.LinearDepth))*HalfFootprint;
    float variance=min(dot(gradient,gradient),ShadowMetric.w);
    if (LightControl.w!=0 && input.Parameters.z<0) discard;
    if (dot(float4(input.World,1),ClipEquation)<0) discard;
    float linearDepth=saturate(input.LinearDepth); return float2(linearDepth,linearDepth*linearDepth+variance);
}
#elif defined(EVANS_HAIR_DEPTH_ONLY)
void PSMain(RasterFiber input) { if ((LightControl.w!=0 && input.Parameters.z<0) || dot(float4(input.World,1),ClipEquation)<0) discard; }
#else
float PositiveRoot(float value) { return sqrt(max(value,0)); }
float PolarAngle(float y,float x) { return abs(y)+abs(x)>SmallDirectionSquared ? atan2(y,x):0; }
float LogI0(float value)
{
    value=abs(value);
    if (value>LargeBesselArgument)
    {
        float inverse=1/value;
        float correction=1+inverse*(ExpansionFirst+inverse*(ExpansionSecond+inverse*ExpansionThird));
        return value-log(FullTurn*value)/2+log(correction);
    }
    float accumulator=1,coefficient=1,argument=value*value/4;
    [unroll] for (uint term=1;term<BesselSeriesCount;++term)
    { coefficient*=argument/(term*term); accumulator+=coefficient; }
    return log(accumulator);
}
float LongitudinalDensity(float sineIn,float sineOut,float variance)
{
    float correlation=PositiveRoot(1-sineIn*sineIn)*PositiveRoot(1-sineOut*sineOut)/variance;
    float exponent=sineIn*sineOut/variance;
    float logNormalization=log(variance)+1/variance+log(1-exp(-2/variance));
    return exp(LogI0(correlation)-exponent-logNormalization);
}
float WrappedLogistic(float delta)
{
    delta-=FullTurn*floor((delta+Pi)/FullTurn);
    float exponential=exp(-abs(delta)/OpticalParameters.y),endpoint=exp(-Pi/OpticalParameters.y);
    float mass=(1-endpoint)/(1+endpoint);
    return exponential/(OpticalParameters.y*(1+exponential)*(1+exponential)*mass);
}
float DielectricReflection(float incidenceCosine)
{
    float eta=OpticalParameters.x;
    float transmittedCosine=PositiveRoot(1-(1-incidenceCosine*incidenceCosine)/(eta*eta));
    float parallel=(eta*incidenceCosine-transmittedCosine)/(eta*incidenceCosine+transmittedCosine);
    float perpendicular=(incidenceCosine-eta*transmittedCosine)/(incidenceCosine+eta*transmittedCosine);
    return (parallel*parallel+perpendicular*perpendicular)/2;
}
struct FiberOptics
{
    float4 ShiftedSine;
    float3 Attenuation[PathCount];
    float AzimuthOut,BoundaryAngle,TransmittedAngle;
};
FiberOptics MakeOptics(float3 reflectance,float crossSection,float3 outgoing)
{
    FiberOptics result;
    float sine=clamp(outgoing.x,-1,1),cosine=PositiveRoot(1-sine*sine),eta=OpticalParameters.x;
    float h=clamp(crossSection,-1,1),inside=h*cosine/PositiveRoot(eta*eta-sine*sine);
    float pathLength=2*PositiveRoot(1-inside*inside)/PositiveRoot(1-sine*sine/(eta*eta));
    float3 coefficient=log(clamp(reflectance,ReflectanceFloor,1))/OpticalParameters.z;
    float3 transmittance=exp(-coefficient*coefficient*pathLength);
    float reflection=DielectricReflection(cosine*PositiveRoot(1-h*h));
    float3 recurrence=transmittance*reflection;
    result.Attenuation[0]=reflection; result.Attenuation[1]=(1-reflection)*(1-reflection)*transmittance;
    result.Attenuation[2]=result.Attenuation[1]*recurrence;
    result.Attenuation[3]=result.Attenuation[2]*recurrence/max(1-recurrence,TransmissionFloor);
    result.ShiftedSine=clamp(float4(sine*TiltCosine.y-cosine*TiltSine.y,sine*TiltCosine.x+cosine*TiltSine.x,
        sine*TiltCosine.z+cosine*TiltSine.z,sine),-1,1);
    result.AzimuthOut=PolarAngle(outgoing.z,outgoing.y); result.BoundaryAngle=asin(h);
    result.TransmittedAngle=asin(clamp(inside,-1,1)); return result;
}
float3 FiberResponse(FiberOptics optics,float3 incoming)
{
    float difference=PolarAngle(incoming.z,incoming.y)-optics.AzimuthOut; float3 spectrum=0;
    [unroll] for (uint path=0;path<PathCount;++path)
    {
        float center=2*path*optics.TransmittedAngle-2*optics.BoundaryAngle+path*Pi;
        float azimuthal=path==PathCount-1 ? 1/FullTurn:WrappedLogistic(difference-center);
        spectrum+=optics.Attenuation[path]*(azimuthal*LongitudinalDensity(clamp(incoming.x,-1,1),optics.ShiftedSine[path],PathVariance[path]));
    }
    return spectrum;
}
float3 EvaluateEmitter(FiberLight emitter,float3 position,out float3 toLight)
{
    toLight=emitter.DirectionRange.xyz; if (uint(emitter.PositionKind.w)==DirectionalSource) return emitter.Radiance.rgb;
    float3 offset=emitter.PositionKind.xyz-position; float squared=dot(offset,offset);
    if (!(squared>0) || !isfinite(squared)) { toLight=0; return 0; }
    float distance=sqrt(squared); toLight=offset/distance; float attenuation=1;
    if (emitter.DirectionRange.w>0)
    { float relative=distance/emitter.DirectionRange.w,relativeSquared=relative*relative; attenuation=saturate(1-relativeSquared*relativeSquared); attenuation*=attenuation; }
    if (uint(emitter.PositionKind.w)==RectangleSource) attenuation*=saturate(dot(emitter.DirectionRange.xyz,-toLight));
    return emitter.Radiance.rgb*(attenuation/max(squared,PointDistanceFloor));
}
Texture2D<float4> FiberShadow0:register(t1),FiberShadow1:register(t2),FiberShadow2:register(t3),FiberShadow3:register(t4),FiberShadow4:register(t5);
static const uint SoftShadowMode=2,VarianceShadowMode=3;
// Shadow references: Reeves et al. (1987), PCF; Fernando (2005), PCSS;
// Donnelly and Lauritzen (2006), Variance Shadow Maps. Filter visibility or
// linear moments, then apply the selected direct-light strength once.
float2 FetchFiberShadow(uint slot,int2 coordinate)
{
    if (any(coordinate<0) || any(coordinate>=int2(ShadowExtents[slot].xy))) return 1;
    if (slot==0) return FiberShadow0.Load(int3(coordinate,0)).rg;
    if (slot==1) return FiberShadow1.Load(int3(coordinate,0)).rg;
    if (slot==2) return FiberShadow2.Load(int3(coordinate,0)).rg;
    if (slot==3) return FiberShadow3.Load(int3(coordinate,0)).rg;
    return FiberShadow4.Load(int3(coordinate,0)).rg;
}
float FiberComparison(uint slot,float2 uv,float reference)
{
    float2 position=uv*ShadowExtents[slot].xy-0.5,fraction=frac(position); int2 base=int2(floor(position));
    float upper=lerp(float(FetchFiberShadow(slot,base).r>=reference),float(FetchFiberShadow(slot,base+int2(1,0)).r>=reference),fraction.x);
    float lower=lerp(float(FetchFiberShadow(slot,base+int2(0,1)).r>=reference),float(FetchFiberShadow(slot,base+1).r>=reference),fraction.x);
    return lerp(upper,lower,fraction.y);
}
float2 FiberMomentPair(uint slot,float2 uv)
{
    float2 position=uv*ShadowExtents[slot].xy-0.5,weight=frac(position); int2 base=int2(floor(position));
    return lerp(lerp(FetchFiberShadow(slot,base),FetchFiberShadow(slot,base+int2(1,0)),weight.x),
        lerp(FetchFiberShadow(slot,base+int2(0,1)),FetchFiberShadow(slot,base+1),weight.x),weight.y);
}
float FiberLightDistance(float z,float4 projection)
{ return projection.x>0 ? rcp((1-z)/projection.x+z/projection.y):z*projection.y; }
float2 FiberDiskSample(uint sampleIndex,uint sampleCount)
{
    const float PhyllotaxisAngle=Pi*(3-sqrt(5.0)); float angle=sampleIndex*PhyllotaxisAngle;
    return sqrt((sampleIndex+0.5)/sampleCount)*float2(cos(angle),sin(angle));
}
float2 FiberFootprint(uint slot,float worldRadius,float4 projection,float texelBudget)
{
    const float MapFractionLimit=0.25;
    float2 radius=worldRadius/projection.zw; float footprint=max(radius.x*ShadowExtents[slot].x,radius.y*ShadowExtents[slot].y);
    float limit=min(texelBudget,min(ShadowExtents[slot].x,ShadowExtents[slot].y)*MapFractionLimit);
    return footprint>limit ? radius*limit/footprint:radius;
}
float FiberMapShadow(uint slot,float4 position,float bias)
{
    const float HomogeneousTolerance=1e-7;
    if ((ShadowControl.y&(1u<<slot))==0 || position.w<=HomogeneousTolerance) return 1;
    float3 ndc=position.xyz/position.w; float2 uv=ndc.xy*float2(0.5,-0.5)+0.5;
    if (any(uv<0) || any(uv>1) || ndc.z<0 || ndc.z>1) return 1;
    float reference=ndc.z-bias;
    if (ShadowControl.x==VarianceShadowMode)
    {
        float linearDepth=dot(ShadowVarianceEncoding[slot].xyz,float3(position.zw,1)); float2 pair=FiberMomentPair(slot,uv);
        float delta=linearDepth-bias-pair.x; if (delta<=0) return 1;
        float variance=max(pair.y-pair.x*pair.x,ShadowVarianceParams.x),bound=variance/(variance+delta*delta);
        return saturate((bound-ShadowVarianceParams.y)/(1-ShadowVarianceParams.y));
    }
    if (ShadowControl.x==SoftShadowMode)
    {
        float4 projection=ShadowSoftMetrics[slot]; float emitter=projection.x>0?ShadowEmitter.x:ShadowEmitter.y;
        if (emitter<=0) return FetchFiberShadow(slot,int2(floor(uv*ShadowExtents[slot].xy))).r>=reference;
        float receiver=FiberLightDistance(ndc.z,projection);
        float search=projection.x>0?emitter*(receiver-projection.x)/receiver:emitter*receiver;
        float2 searchUV=FiberFootprint(slot,search,projection,ShadowEmitter.z); float accumulated=0; uint found=0;
        [loop] for (uint tap=0;tap<ShadowSoftControl.x;++tap)
        {
            float2 sampleUV=uv+FiberDiskSample(tap,ShadowSoftControl.x)*searchUV;
            float depth=FetchFiberShadow(slot,int2(floor(sampleUV*ShadowExtents[slot].xy))).r;
            if (depth<reference) { accumulated+=FiberLightDistance(depth,projection); ++found; }
        }
        if (found==0) return 1;
        float blocker=accumulated/found,gap=max(receiver-blocker,0);
        float penumbra=projection.x>0?emitter*gap/blocker*projection.x/receiver:emitter*gap;
        float2 radius=FiberFootprint(slot,penumbra,projection,ShadowEmitter.w); float visible=0;
        [loop] for (uint filterTap=0;filterTap<ShadowSoftControl.y;++filterTap) visible+=FiberComparison(slot,uv+FiberDiskSample(filterTap,ShadowSoftControl.y)*radius,reference);
        return visible/ShadowSoftControl.y;
    }
    if (ShadowFiltering.x==0) return FetchFiberShadow(slot,int2(floor(uv*ShadowExtents[slot].xy))).r>=reference;
    float total=0; int radius=int(ShadowFiltering.x);
    [loop] for (int y=-radius;y<=radius;++y) [loop] for (int x=-radius;x<=radius;++x)
        total+=FiberComparison(slot,uv+float2(x,y)*ShadowExtents[slot].zw,reference);
    return total/ShadowFiltering.y;
}
#ifdef EVANS_HAIR_CONTACT_RECEIVER
Texture2DArray<float> ContactLayers:register(t6);
float FiberContact(float4 position,uint layer)
{ return ContactControl.x!=0 && layer<ContactControl.y?ContactLayers.Load(int4(int2(position.xy),layer,0)):1; }
#endif
float FiberShadow(float3 world,float3 outward,float3 incoming)
{
    if (ShadowControl.z==0) return 1;
    float cosine=saturate(abs(dot(outward,incoming))),slope=PositiveRoot(1-cosine*cosine)/max(cosine,ShadowBias.w);
    float bias=min(ShadowBias.z,ShadowBias.x+ShadowBias.y*slope),visible=1; uint count=uint(ShadowCascadeParams.z);
    if (count==0) visible=FiberMapShadow(0,mul(ShadowProjection,float4(world,1)),bias);
    else
    {
        float eyeDistance=dot(ShadowEyePlane,float4(world,1));
        if (eyeDistance>=ShadowCascadeParams.x && eyeDistance<=ShadowCascadeParams.y)
        {
            uint slice=0; [loop] while (slice+1<count && eyeDistance>ShadowFar[slice]) ++slice;
            visible=FiberMapShadow(slice+1,mul(ShadowCascadeProjection[slice],float4(world,1)),bias);
            float overlap=ShadowFar[slice]-ShadowBlend[slice];
            if (overlap>0 && eyeDistance>=ShadowBlend[slice])
            {
                float adjacent=1; if (slice+1<count) adjacent=FiberMapShadow(slice+2,mul(ShadowCascadeProjection[slice+1],float4(world,1)),bias);
                visible=lerp(visible,adjacent,saturate((eyeDistance-ShadowBlend[slice])/overlap));
            }
        }
    }
    return visible;
}
struct FiberOutput
{
    float4 Color:SV_TARGET0;
#ifndef EVANS_HAIR_CAPTURE
    float4 InvalidMaterial:SV_TARGET1;
#endif
};
FiberOutput PSMain(RasterFiber input
#ifdef EVANS_HAIR_CONTACT_RECEIVER
    ,uint contactSample:SV_SampleIndex
#endif
    )
{
    if (dot(float4(input.World,1),ClipEquation)<0) discard;
    float3 tangent=NormalizeOr(input.Axis,float3(0,1,0)),view=TowardEye(input.World),side=WidthAxis(tangent,view);
    float3 normal=NormalizeOr(cross(side,tangent),-ViewForward.xyz);
    float3 color=saturate(input.Color*lerp(RootColor.rgb,TipColor.rgb,saturate(input.Parameters.y)));
    FiberOptics fiber=MakeOptics(color,input.Parameters.x,float3(dot(view,tangent),dot(view,side),dot(view,normal)));
    float3 lighting=color*AmbientLight.rgb*(MotionControl.y!=0?0:AmbientLight.w);
    float height=clamp(input.Parameters.x,-1,1); float3 outward=side*height+normal*PositiveRoot(1-height*height);
    [loop] for (uint sampleIndex=0;sampleIndex<LightControl.x;++sampleIndex)
    {
        float3 direction; float3 radiance=EvaluateEmitter(FiberLights[sampleIndex],input.World,direction);
#ifndef EVANS_HAIR_CAPTURE
        if (sampleIndex>=LightControl.y && sampleIndex<LightControl.z)
        {
            float visibility=FiberShadow(input.World,outward,direction);
#ifdef EVANS_HAIR_CONTACT_RECEIVER
            visibility*=FiberContact(input.Position,contactSample);
#endif
            radiance*=1-ShadowComposition.x*(1-visibility);
        }
#endif
        lighting+=radiance*FiberResponse(fiber,float3(dot(direction,tangent),dot(direction,side),dot(direction,normal)));
    }
    FiberOutput result; result.Color=float4(clamp(input.Emitted+lighting*OpticalParameters.w,0,HalfCeiling),1);
#ifndef EVANS_HAIR_CAPTURE
    result.InvalidMaterial=float4(0,0,0,-1);
#endif
    return result;
}
#endif
