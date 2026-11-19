// Material capture is view-independent input to Lumen surface-cache relighting.
// Each card owns a guarded rectangle; native depth testing selects its first layer.
struct ProbeOrigin { float4 PositionMinimum; uint4 Metadata; };
static const uint LumenScreenSurface=3,LumenWorldCache=4,LumenHairPointerShift=2;
float3 LumenEnvironmentRadiance(float3 baseSky,float4 axisPower,float3 peak,float3 direction)
{
    if (!any(peak>0)) return baseSky;
    float cosine=saturate(dot(axisPower.xyz,direction));
    return baseSky+peak*pow(cosine,axisPower.w);
}
static const uint LumenProbeNeighbors=4,LumenLinkWordBits=32;
uint probeLinkWords(uint slots) { return (LumenProbeNeighbors*slots+LumenLinkWordBits-1)/LumenLinkWordBits; }
// SM5 has no dynamically addressed vector-component stores.
uint3 probeLinkBit(uint candidate)
{
    uint word=candidate/LumenLinkWordBits,bit=1u<<(candidate%LumenLinkWordBits);
    return uint3(word==0?bit:0,word==1?bit:0,word==2?bit:0);
}
float3 ProbeSphereDirection(uint ray,uint count,uint id,uint sequence)
{
    static const float Pi=3.14159265358979323846,UnsignedPhaseScale=1.0/4294967296.0;
    float z=1-2*(float(ray)+0.5)/float(count);
    uint phase=reversebits(ray)+reversebits(id^sequence);
    float angle=2*Pi*(float(phase)*UnsignedPhaseScale),radius=sqrt(max(0,1-z*z));
    return float3(radius*cos(angle),radius*sin(angle),z);
}
// Clarberg (JGT 2008): square texels cover equal spherical area. The atlas
// keeps one angular layout at every world probe; no per-probe phase is applied.
float3 WorldCacheDirection(uint texel,uint side)
{
    static const float CachePi=3.14159265358979323846;
    side=max(side,1u); float2 square=(float2(texel%side,texel/side)+0.5)/float(side)*2-1;
    float hemisphere=1-abs(square.x)-abs(square.y),radius=1-abs(hemisphere);
    float angle=radius>0?(CachePi/4)*(1+(abs(square.y)-abs(square.x))/radius):0;
    float ring=radius*sqrt(max(0,2-radius*radius)),height=1-radius*radius;
    return float3(ring*abs(cos(angle))*(square.x<0?-1:1),ring*abs(sin(angle))*(square.y<0?-1:1),hemisphere<0?-height:height);
}
#if defined(EVANS_LUMEN_REFLECTION_RESOLVE_STAGE)
// Heitz (2018) VNDF estimator: Fresnel * height-correlated Smith G2 / G1(V).
// Spatial reuse stays on compatible material planes. Unknown transport remains
// a zero contribution in the denominator; partial estimates never reuse history.
cbuffer ReflectionResolveFrame:register(b0)
{
    row_major float4x4 ResolveInverse; uint4 ResolveViewport,ResolveWork;
    float4 ResolveTrace,ResolveNumeric,ResolveAcceptance,ResolveFilter,ResolveSky; uint4 ResolveDispatch; float4 EnvironmentAxis,EnvironmentPeak;
};
Texture2D<float> ResolveDepth:register(t0);
Texture2D<float4> ResolveNormal:register(t1),ResolveDiffuse:register(t2),ResolveAmbient:register(t3),ResolveSpecular:register(t4);
struct ResolveRay { float4 OriginMinimum,DirectionMaximum; };
struct ResolveHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct ResolveSample { ResolveHit Hit; float4 RadianceCoverage; uint4 Cache; };
StructuredBuffer<ResolveRay> ResolveRays:register(t5);
StructuredBuffer<ResolveSample> ResolveSamples:register(t6);
RWStructuredBuffer<float4> ResolveOutput:register(u0);
static const uint ResolveThreads=64,ResolvePBR=3,ResolveMiss=0,ResolveSurface=1,ResolveScreen=3,ResolveComplete=1,ResolvePartial=2;
static const float ResolveIncompleteConfidence=0.5;
bool ResolveWorld(uint2 pixel,float depth,out float3 world)
{
    float2 uv=(float2(pixel)+0.5)/float2(ResolveViewport.xy);
    float4 h=mul(ResolveInverse,float4(uv*float2(2,-2)+float2(-1,1),depth,1)); world=0;
    if (!all(isfinite(h)) || abs(h.w)<=ResolveNumeric.y) return false;
    world=h.xyz/h.w; return all(isfinite(world)) && all(abs(world)<=ResolveNumeric.x);
}
bool ResolveMaterial(uint2 pixel,out float3 world,out float3 normal,out float roughness)
{
    float depth=ResolveDepth.Load(int3(pixel,0)); float4 material=ResolveNormal.Load(int3(pixel,0));
    world=normal=0; roughness=material.w;
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(material)) || roughness<0 || roughness>=ResolveTrace.z || dot(material.xyz,material.xyz)<=ResolveNumeric.z) return false;
    normal=normalize(material.xyz); return ResolveWorld(pixel,depth,world);
}
float ResolveLambda(float cosine,float alpha)
{ float square=max(cosine*cosine,ResolveFilter.y*ResolveFilter.y); return (sqrt(1+alpha*alpha*max(0,1-square)/square)-1)*0.5; }
[numthreads(ResolveThreads,1,1)]
void CSReflectionResolve(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*ResolveDispatch.y; if (index>=ResolveDispatch.z) return; ResolveOutput[index]=0;
    uint2 pixel=uint2(index%ResolveViewport.x,index/ResolveViewport.x); float3 position,normal; float roughness;
    if (!ResolveMaterial(pixel,position,normal,roughness)) return;
    float4 diffuse=ResolveDiffuse.Load(int3(pixel,0)),ambient=ResolveAmbient.Load(int3(pixel,0)),specular=ResolveSpecular.Load(int3(pixel,0));
    if (!all(isfinite(diffuse)) || !all(isfinite(ambient)) || !all(isfinite(specular))) return;
    float3 f0=ResolveDispatch.x==ResolvePBR?lerp(ResolveFilter.w.xxx,saturate(diffuse.xyz),saturate(ambient.w)):saturate(specular.xyz);
    int2 center=int2(pixel/ResolveViewport.z); int radius=int(ResolveFilter.x); float3 sum=0; float denominator=0; bool complete=true;
    [loop] for (int y=-radius;y<=radius;++y) [loop] for (int x=-radius;x<=radius;++x)
    {
        int2 cell=center+int2(x,y); if (any(cell<0) || cell.x>=int(ResolveViewport.w) || cell.y>=int(ResolveWork.x)) continue;
        uint rayIndex=uint(cell.y)*ResolveViewport.w+uint(cell.x); ResolveRay ray=ResolveRays[rayIndex];
        if (!all(isfinite(ray.DirectionMaximum)) || dot(ray.DirectionMaximum.xyz,ray.DirectionMaximum.xyz)<=ResolveNumeric.z) continue;
        uint2 source=min(uint2(cell)*ResolveViewport.z+ResolveViewport.z/2,ResolveViewport.xy-1);
        float3 p,n; float r; if (!ResolveMaterial(source,p,n,r) || dot(normal,n)<ResolveAcceptance.x || abs(r-roughness)>ResolveAcceptance.w) continue;
        float3 delta=p-position; float plane=max(abs(dot(delta,n)),abs(dot(delta,normal)));
        if (!isfinite(plane) || plane>ResolveAcceptance.y+ResolveAcceptance.z*length(delta)) continue;
        float2 offset=(float2(source)-float2(pixel))/float(ResolveViewport.z); float weight=1/(1+dot(offset,offset)); denominator+=weight;
        // VNDF samples reflected below the horizon are valid zero contributions.
        if (ray.DirectionMaximum.w<=0) continue;
        float3 nearWorld,midWorld; if (!ResolveWorld(source,0,nearWorld) || !ResolveWorld(source,0.5,midWorld)) { complete=false; continue; }
        float3 view=normalize(nearWorld-midWorld),light=ray.DirectionMaximum.xyz;
        float noV=dot(n,view),noL=dot(n,light); if (!all(isfinite(view)) || noV<=0 || noL<=0) { complete=false; continue; }
        ResolveSample incoming=ResolveSamples[rayIndex]; float3 radiance=0;
        if (incoming.Hit.Metadata.z==ResolveMiss) radiance=LumenEnvironmentRadiance(ResolveSky.xyz,EnvironmentAxis,EnvironmentPeak.xyz,light);
        else if ((incoming.Hit.Metadata.z==ResolveSurface || incoming.Hit.Metadata.z==ResolveScreen) && (incoming.Cache.z==ResolveComplete || incoming.Cache.z==ResolvePartial))
        { radiance=incoming.RadianceCoverage.xyz; complete=complete && incoming.Cache.z==ResolveComplete; }
        else complete=false;
        radiance=radiance*incoming.Hit.Medium.w+incoming.Hit.Medium.xyz;
        if (!all(isfinite(radiance)) || any(radiance<0)) { complete=false; continue; }
        float alpha=max(r*r,ResolveTrace.w),lambdaV=ResolveLambda(noV,alpha),lambdaL=ResolveLambda(noL,alpha);
        float masking=r==0?1:(1+lambdaV)/(1+lambdaV+lambdaL);
        float fresnelBase=1-saturate(dot(view,normalize(view+light)));
        float fresnelPower=fresnelBase*fresnelBase*fresnelBase*fresnelBase*fresnelBase;
        sum+=radiance*(f0+(1-f0)*fresnelPower)*masking*weight;
    }
    if (denominator>0 && all(isfinite(sum))) ResolveOutput[index]=float4(clamp(sum/denominator,0,ResolveFilter.z),complete?1:ResolveIncompleteConfidence);
}
#elif defined(EVANS_LUMEN_REFLECTION_RAYS_STAGE)
// Heitz, Sampling the GGX Distribution of Visible Normals (JCGT 2018).
// Stretch the view, sample its projected disk, then unstretch the micro-normal.
// A below-horizon reflection keeps its direction but has zero trace length:
// reconstruction must count this valid zero contribution instead of resampling.
cbuffer ReflectionRayFrame:register(b0)
{ row_major float4x4 ReflectionInverse; uint4 ReflectionViewport,ReflectionWork; float4 ReflectionTrace,ReflectionNumeric; };
Texture2D<float> ReflectionDepth:register(t0);
Texture2D<float4> ReflectionNormal:register(t1);
struct ReflectionRay { float4 OriginMinimum,DirectionMaximum; };
RWStructuredBuffer<ReflectionRay> ReflectionRays:register(u0);
static const uint ReflectionThreads=64,ReflectionSamplePeriod=64;
static const float ReflectionPi=3.14159265358979323846,ReflectionUIntFraction=1.0/4294967296.0;
static const float ReflectionGoldenConjugate=0.6180339887498948482,ReflectionBasisLimit=0.9;
bool ReflectionPosition(uint2 pixel,float depth,out float3 world)
{
    float2 uv=(float2(pixel)+0.5)/float2(ReflectionViewport.xy);
    float4 h=mul(ReflectionInverse,float4(uv.x*2-1,1-uv.y*2,depth,1)); world=0;
    if (!all(isfinite(h)) || abs(h.w)<=ReflectionNumeric.y) return false;
    world=h.xyz/h.w; return all(isfinite(world)) && all(abs(world)<=ReflectionNumeric.x);
}
float3 ReflectionVisibleNormal(float3 view,float alpha,float2 xi)
{
    float3 stretched=normalize(float3(alpha*view.xy,view.z)); float squared=dot(stretched.xy,stretched.xy);
    float3 first=squared>0?float3(-stretched.y,stretched.x,0)*rsqrt(squared):float3(1,0,0),second=cross(stretched,first);
    float radius=sqrt(xi.x),angle=2*ReflectionPi*xi.y,x=radius*cos(angle),y=radius*sin(angle),blend=(1+stretched.z)*0.5;
    y=lerp(sqrt(max(0,1-x*x)),y,blend);
    float3 disk=first*x+second*y+stretched*sqrt(max(0,1-x*x-y*y));
    return normalize(float3(alpha*disk.xy,max(0,disk.z)));
}
[numthreads(ReflectionThreads,1,1)]
void CSReflectionRays(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*ReflectionWork.w; if (index>=ReflectionWork.y) return;
    ReflectionRay ray=(ReflectionRay)0; ReflectionRays[index]=ray;
    uint2 pixel=min(uint2(index%ReflectionViewport.w,index/ReflectionViewport.w)*ReflectionViewport.z+ReflectionViewport.z/2,ReflectionViewport.xy-1);
    float depth=ReflectionDepth.Load(int3(pixel,0)); float4 material=ReflectionNormal.Load(int3(pixel,0));
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(material)) || material.w<0 || material.w>=ReflectionTrace.z || dot(material.xyz,material.xyz)<=ReflectionNumeric.z) return;
    float3 world,nearWorld,middleWorld;
    if (!ReflectionPosition(pixel,depth,world) || !ReflectionPosition(pixel,0,nearWorld) || !ReflectionPosition(pixel,0.5,middleWorld)) return;
    float3 normal=normalize(material.xyz),view=normalize(nearWorld-middleWorld); if (!all(isfinite(view)) || dot(normal,view)<=0) return;
    float3 axis=abs(normal.z)<ReflectionBasisLimit?float3(0,0,1):float3(0,1,0),tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent);
    float rotation=float(reversebits(index))*ReflectionUIntFraction;
    float2 xi=frac(float2((float(ReflectionWork.z%ReflectionSamplePeriod)+0.5)/ReflectionSamplePeriod,
        float(reversebits(ReflectionWork.z))*ReflectionUIntFraction)+float2(rotation,frac(rotation*ReflectionGoldenConjugate)));
    float alpha=max(material.w*material.w,ReflectionTrace.w);
    float3 localHalf=material.w==0?float3(0,0,1):ReflectionVisibleNormal(float3(dot(view,tangent),dot(view,bitangent),dot(view,normal)),alpha,xi);
    float3 halfVector=tangent*localHalf.x+bitangent*localHalf.y+normal*localHalf.z,direction=reflect(-view,halfVector);
    ray.OriginMinimum=float4(world+normal*ReflectionTrace.y,0);
    ray.DirectionMaximum=float4(direction,dot(normal,direction)>0?ReflectionTrace.x:0); ReflectionRays[index]=ray;
}
#elif defined(EVANS_LUMEN_TEMPORAL_STAGE) || defined(EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE)
// Diffuse-only history follows physical motion plus raster jitter. Binary16
// radiance, octahedral SNORM16 normals and FP32 depth occupy one uint4 per pixel.
cbuffer LumenTemporalFrame:register(b0)
{
    row_major float4x4 TemporalInverse,TemporalPreviousInverse;
    float4 TemporalPreviousPlane; uint4 TemporalExtent;
    float4 TemporalRules,TemporalClip,TemporalJitter,TemporalNumeric;
};
Texture2D<float4> TemporalCurrent:register(t0);
StructuredBuffer<uint4> TemporalHistory:register(t4);
static const uint TemporalGroupWidth=64,TemporalHalfBits=16,TemporalHalfMask=0xffffu;
static const int TemporalFilterRadius=1;
static const float TemporalNormalScale=32767,TemporalNormalFloor=1e-12;
float4 TemporalColor(uint4 value)
{ return float4(f16tof32(value.x&TemporalHalfMask),f16tof32(value.x>>TemporalHalfBits),f16tof32(value.y&TemporalHalfMask),f16tof32(value.y>>TemporalHalfBits)); }
#if defined(EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE)
struct TemporalVertex { float4 Position:SV_POSITION; };
TemporalVertex VSTemporal(uint id:SV_VertexID)
{ TemporalVertex v; float2 corner=float2((id<<1)&2,id&2); v.Position=float4(corner*float2(2,-2)+float2(-1,1),0,1); return v; }
float4 PSTemporalComposite(TemporalVertex input):SV_TARGET
{
    uint2 pixel=uint2(input.Position.xy); float4 scene=TemporalCurrent.Load(int3(pixel,0));
    float4 indirect=TemporalColor(TemporalHistory[pixel.y*TemporalExtent.x+pixel.x]);
    float3 value=indirect.w>0?indirect.xyz*TemporalJitter.z:0;
    return float4(clamp(scene.xyz+value,0,TemporalNumeric.w),scene.w);
}
#else
Texture2D<float> TemporalDepth:register(t1);
Texture2D<float4> TemporalNormals:register(t2),TemporalMotion:register(t3);
RWStructuredBuffer<uint4> TemporalOutput:register(u0);
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
StructuredBuffer<float4> TemporalSpecular:register(t5);
#endif
float4 TemporalEstimate(uint2 pixel)
{
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
    return TemporalSpecular[pixel.y*TemporalExtent.x+pixel.x];
#else
    return TemporalCurrent.Load(int3(pixel,0));
#endif
}
float2 TemporalSign(float2 value) { return float2(value.x>=0?1:-1,value.y>=0?1:-1); }
float3 TemporalNormal(uint value)
{
    float2 oct=float2(int(value<<TemporalHalfBits)>>TemporalHalfBits,int(value)>>TemporalHalfBits)/TemporalNormalScale;
    float3 normal=float3(oct,1-abs(oct.x)-abs(oct.y));
    if (normal.z<0) normal.xy=(1-abs(normal.yx))*TemporalSign(normal.xy);
    return normalize(normal);
}
uint4 TemporalPack(float4 color,float3 normal,float depth)
{
    float2 oct=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));
    if (normal.z<0) oct=(1-abs(oct.yx))*TemporalSign(oct);
    int2 quantized=int2(round(clamp(oct,-1,1)*TemporalNormalScale));
    return uint4(f32tof16(color.x)|(f32tof16(color.y)<<TemporalHalfBits),f32tof16(color.z)|(f32tof16(color.w)<<TemporalHalfBits),
        (uint(quantized.x)&TemporalHalfMask)|(uint(quantized.y)<<TemporalHalfBits),asuint(depth));
}
bool TemporalSurface(uint2 pixel,out float3 position,out float3 normal,out float depth)
{
    depth=TemporalDepth.Load(int3(pixel,0)); float4 surface=TemporalNormals.Load(int3(pixel,0)); position=0; normal=float3(0,0,1);
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(surface)) || surface.w<0 || dot(surface.xyz,surface.xyz)<=TemporalNormalFloor) return false;
    float2 uv=(float2(pixel)+0.5)/float2(TemporalExtent.xy);
    float4 h=mul(TemporalInverse,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
    if (!all(isfinite(h)) || abs(h.w)<=TemporalNumeric.z) return false;
    position=h.xyz/h.w; normal=normalize(surface.xyz); return all(isfinite(position));
}
float TemporalPreviousDistance(float2 pixel,float depth)
{
    float2 uv=pixel/float2(TemporalExtent.xy);
    float4 h=mul(TemporalPreviousInverse,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
    if (!all(isfinite(h)) || abs(h.w)<=TemporalNumeric.z) return -1;
    return dot(TemporalPreviousPlane,float4(h.xyz/h.w,1));
}
[numthreads(TemporalGroupWidth,1,1)]
void CSTemporal(uint3 dispatch:SV_DispatchThreadID)
{
    uint index=dispatch.x+dispatch.y*TemporalExtent.w; if (index>=TemporalExtent.z) return;
    uint2 pixel=uint2(index%TemporalExtent.x,index/TemporalExtent.x); float3 position,normal; float depth;
    float4 current=TemporalEstimate(pixel);
    if (!TemporalSurface(pixel,position,normal,depth) || !all(isfinite(current)) || current.w<=0)
    { TemporalOutput[index]=TemporalPack(0,float3(0,0,1),1); return; }
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
    // Display known partial energy, but never admit it to the reusable history.
    if (current.w<1) { TemporalOutput[index]=TemporalPack(float4(clamp(current.xyz,0,TemporalNumeric.w),current.w),normal,depth); return; }
#endif
    current=float4(clamp(current.xyz,0,TemporalNumeric.w),1); TemporalOutput[index]=TemporalPack(current,normal,depth);
    if (TemporalJitter.w==0 || TemporalRules.x<=0) return;
    float4 motion=TemporalMotion.Load(int3(pixel,0)); if (!all(isfinite(motion)) || motion.w<=0) return;
    float2 previous=float2(pixel)+0.5-motion.xy-TemporalJitter.xy;
    if (any(previous<0.5) || any(previous>float2(TemporalExtent.xy)-0.5)) return;
    float expected=TemporalPreviousDistance(previous,1-motion.z); if (!isfinite(expected) || expected<0) return;
    int2 base=int2(floor(previous-0.5)); float2 fraction=frac(previous-0.5); float4 sum=0; float weights=0;
    [unroll] for (uint tap=0;tap<4;++tap)
    {
        int2 corner=int2(tap&1,tap>>1),candidate=base+corner;
        // Clamp the address before loading: a zero edge weight alone does not
        // prevent a compiler from speculating an out-of-range buffer access.
        uint2 bounded=uint2(clamp(candidate,0,int2(TemporalExtent.xy)-1));
        uint4 packed=TemporalHistory[bounded.y*TemporalExtent.x+bounded.x]; float4 old=TemporalColor(packed); float oldDepth=asfloat(packed.w);
        float weight=(corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y);
        if (weight<=0 || any(candidate<0) || any(candidate>=int2(TemporalExtent.xy)) || !all(isfinite(old)) || old.w<1 || old.w>TemporalClip.y ||
            !isfinite(oldDepth) || oldDepth<0 || oldDepth>=1 || dot(normal,TemporalNormal(packed.z))<TemporalRules.z) continue;
        float optical=TemporalPreviousDistance(float2(candidate)+0.5,oldDepth);
        if (!isfinite(optical) || abs(optical-expected)>max(TemporalNumeric.x,abs(expected)*TemporalRules.y)) continue;
        sum+=old*weight; weights+=weight;
    }
    if (weights<=TemporalNormalFloor) return;
    float4 history=sum/weights; float3 low=current.xyz,high=low;
    [unroll] for (int y=-TemporalFilterRadius;y<=TemporalFilterRadius;++y) [unroll] for (int x=-TemporalFilterRadius;x<=TemporalFilterRadius;++x)
    {
        uint2 neighbor=uint2(clamp(int2(pixel)+int2(x,y),0,int2(TemporalExtent.xy)-1));
        float3 p,n; float z; float4 value=TemporalEstimate(neighbor);
        if (value.w<=0 || !all(isfinite(value)) || !TemporalSurface(neighbor,p,n,z) || dot(normal,n)<TemporalRules.z) continue;
        float plane=abs(dot(p-position,normal)); if (!isfinite(plane) || plane>TemporalRules.w) continue;
        low=min(low,value.xyz); high=max(high,value.xyz);
    }
    float3 center=(low+high)*0.5,extent=max((high-low)*0.5+abs(center)*TemporalClip.x,TemporalNumeric.y);
    float3 delta=history.xyz-center,ratio=abs(delta/extent); history.xyz=center+delta/max(1,max(ratio.x,max(ratio.y,ratio.z)));
    float samples=min(history.w+1,TemporalClip.y),blend=min(TemporalRules.x,(samples-1)/samples);
    TemporalOutput[index]=TemporalPack(float4(clamp(lerp(current.xyz,history.xyz,blend),0,TemporalNumeric.w),samples),normal,depth);
}
#endif
#elif defined(EVANS_LUMEN_PROBE_GATHER_STAGE)
// Screen-cache reconstruction: compatible local planes share low-frequency
// irradiance. This geometric rejection does not replace ray visibility.
cbuffer GatherFrame:register(b0)
{
    uint4 GatherTraceCounts,GatherTraceOffsets,GatherTraceWork; float4 GatherTraceParameters,GatherHairSky;
    row_major float4x4 GatherInverse; uint4 GatherViewport,GatherGrid; float4 GatherOriginParameters,GatherAdaptive;
    float4 GatherAppearance,GatherNumeric; uint4 GatherModel; float4 GatherVisibility;
};
struct GatherProbe { float4 Coefficients[9]; float4 Coverage; uint4 TraceCounts,CacheCounts,Metadata; };
Texture2D<float4> GatherScene:register(t0);
Texture2D<float> GatherDepth:register(t1);
Texture2D<float4> GatherNormal:register(t2),GatherDiffuse:register(t3),GatherAmbient:register(t4);
StructuredBuffer<ProbeOrigin> GatherOrigins:register(t5);
StructuredBuffer<GatherProbe> GatherProbes:register(t6);
StructuredBuffer<uint> GatherLinks:register(t7);
static const uint GatherCoefficients=9,GatherPartial=2,GatherPBR=3;
static const float GatherPi=3.14159265358979323846,GatherCoordinateRoundoff=32*1.1920928955078125e-7;
struct GatherVertex { float4 Position:SV_POSITION; };
GatherVertex VSGather(uint vertex:SV_VertexID)
{ GatherVertex result; float2 corner=float2((vertex<<1)&2,vertex&2); result.Position=float4(corner*float2(2,-2)+float2(-1,1),0,1); return result; }
float GatherBasis(uint coefficient,float3 n)
{
    switch (coefficient)
    {
        case 0:return sqrt(1/(4*GatherPi));
        case 1:return sqrt(3/(4*GatherPi))*n.y;
        case 2:return sqrt(3/(4*GatherPi))*n.z;
        case 3:return sqrt(3/(4*GatherPi))*n.x;
        case 4:return sqrt(15/(4*GatherPi))*n.x*n.y;
        case 5:return sqrt(15/(4*GatherPi))*n.y*n.z;
        case 6:return sqrt(5/(16*GatherPi))*(3*n.z*n.z-1);
        case 7:return sqrt(15/(4*GatherPi))*n.x*n.z;
        case 8:return sqrt(15/(16*GatherPi))*(n.x*n.x-n.y*n.y);
    }
    return 0;
}
float GatherAxis(float pixel,uint extent,uint tiles)
{
    float side=float(GatherViewport.z),coordinate=pixel/side-0.5;
    if (tiles>1 && extent%GatherViewport.z!=0)
    {
        float previous=(float(tiles)-1.5)*side,last=(float(tiles-1)*side+extent)*0.5;
        if (pixel>previous) coordinate=float(tiles-2)+(pixel-previous)/(last-previous);
    }
    return clamp(coordinate,0,float(tiles-1));
}
float GatherMagnitude(float3 value) { return max(abs(value.x),max(abs(value.y),abs(value.z))); }
float4 PSGather(GatherVertex input):SV_TARGET
{
    int2 pixel=int2(input.Position.xy); float4 scene=GatherModel.z!=0?float4(0,0,0,0):GatherScene.Load(int3(pixel,0));
    float depth=GatherDepth.Load(int3(pixel,0)); float4 materialNormal=GatherNormal.Load(int3(pixel,0));
    if (GatherAppearance.x==0 || depth>=1 || depth<0 || !isfinite(depth) || materialNormal.w<0 || !all(isfinite(materialNormal)) || dot(materialNormal.xyz,materialNormal.xyz)<=GatherNumeric.x) return scene;
    float2 uv=input.Position.xy/float2(GatherViewport.xy);
    float4 h=mul(GatherInverse,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
    if (!all(isfinite(h)) || abs(h.w)<=GatherOriginParameters.w) return scene;
    float3 world=h.xyz/h.w,normal=normalize(materialNormal.xyz);
    if (!all(isfinite(world)) || any(abs(world)>GatherOriginParameters.z)) return scene;
    float2 grid=float2(GatherAxis(input.Position.x,GatherViewport.x,GatherViewport.w),GatherAxis(input.Position.y,GatherViewport.y,GatherGrid.x));
    uint2 cell=uint2(floor(grid)); float2 fraction=grid-float2(cell); float3 irradiance=0; float totalWeight=0;
    uint slots=GatherGrid.w+1,words=probeLinkWords(GatherGrid.w+1);
    [loop] for (uint tap=0;tap<4*slots;++tap)
    {
        uint cornerIndex=tap/slots;
        uint2 corner=uint2(cornerIndex&1,cornerIndex>>1),tile=min(cell+corner,uint2(GatherViewport.w,GatherGrid.x)-1);
        float weight=(corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y); if (weight<=0) continue;
        if (GatherModel.y!=0 && (GatherLinks[(uint(pixel.y)*GatherViewport.x+uint(pixel.x))*words+tap/LumenLinkWordBits]&(1u<<(tap%LumenLinkWordBits)))==0) continue;
        uint index=(tile.y*GatherViewport.w+tile.x)*slots+tap%slots; ProbeOrigin origin=GatherOrigins[index]; GatherProbe probe=GatherProbes[index];
        if (origin.Metadata.y!=0 || origin.Metadata.x!=index || any(origin.Metadata.zw>=GatherViewport.xy) || probe.Metadata.x!=index || probe.Metadata.y==0 || probe.Metadata.y>GatherPartial) continue;
        float4 anchorNormal=GatherNormal.Load(int3(origin.Metadata.zw,0));
        if (anchorNormal.w<0 || !all(isfinite(anchorNormal)) || dot(anchorNormal.xyz,anchorNormal.xyz)<=GatherNumeric.x) continue;
        float3 n=normalize(anchorNormal.xyz),anchor=origin.PositionMinimum.xyz-n*GatherOriginParameters.x;
        float agreement=dot(normal,n); if (agreement<=GatherAppearance.y) continue;
        float3 delta=world-anchor; float magnitude=GatherMagnitude(delta),distance=magnitude>0?magnitude*length(delta/magnitude):0;
        float error=max(abs(dot(delta,normal)),abs(dot(delta,n)));
        float tolerance=GatherAppearance.z+GatherAppearance.w*distance+GatherCoordinateRoundoff*max(GatherMagnitude(world),GatherMagnitude(anchor));
        if (!isfinite(error) || !isfinite(tolerance) || error>tolerance) continue;
        float normalWeight=(agreement-GatherAppearance.y)/(1-GatherAppearance.y),planeWeight=tolerance>0?saturate(1-error/tolerance):1;
        weight*=normalWeight*normalWeight*planeWeight;
        float3 value=0; [unroll] for (uint coefficient=0;coefficient<GatherCoefficients;++coefficient) value+=probe.Coefficients[coefficient].xyz*GatherBasis(coefficient,normal);
        if (!all(isfinite(value))) continue;
        irradiance+=max(value,0)*weight; totalWeight+=weight;
    }
    if (totalWeight<=GatherNumeric.y || !all(isfinite(irradiance))) return scene;
    float3 diffuse=saturate(GatherDiffuse.Load(int3(pixel,0)).xyz);
    if (GatherModel.x==GatherPBR) diffuse*=(1-GatherNumeric.w)*(1-saturate(GatherAmbient.Load(int3(pixel,0)).w));
    float3 indirect=(irradiance/totalWeight)*(diffuse/GatherPi);
    if (!all(isfinite(indirect))) return scene;
    if (GatherModel.z!=0) return float4(clamp(indirect,0,GatherNumeric.z),1);
    return float4(clamp(scene.xyz+indirect*GatherAppearance.x,0,GatherNumeric.z),scene.w);
}
#elif defined(EVANS_LUMEN_SCREEN_DEPTH_STAGE)
// Conservative forward-depth pyramid. A parent covers every child, including
// odd viewport edges; unknown reconstruction must never become empty space.
cbuffer DepthPyramidFrame:register(b0)
{ row_major float4x4 DepthInverse; float4 DepthAnchor,DepthForward; uint4 DepthSource,DepthDestination,DepthDispatch; float4 DepthNumeric; };
Texture2D<float> DepthInput:register(t0);
RWStructuredBuffer<uint> DepthPyramid:register(u0);
static const uint DepthThreads=64,DepthHeaderWords=4;
[numthreads(DepthThreads,1,1)]
void CSBuildScreenDepth(uint3 dispatch:SV_DispatchThreadID)
{
    uint index=dispatch.x+dispatch.y*DepthDispatch.x; if (index>=DepthDestination.x*DepthDestination.y) return;
    if (index==0) for (uint component=0;component<DepthHeaderWords;++component) DepthPyramid[DepthDestination.w*DepthHeaderWords+component]=DepthDestination[component];
    uint2 pixel=uint2(index%DepthDestination.x,index/DepthDestination.x); float value=DepthNumeric.y;
    if (DepthDestination.w==0)
    {
        float depth=DepthInput.Load(int3(pixel,0));
        if (isfinite(depth) && depth>=0 && depth<1)
        {
            float2 uv=(float2(pixel)+0.5)/DepthDestination.xy;
            float4 h=mul(DepthInverse,float4(uv*float2(2,-2)+float2(-1,1),depth,1)); value=DepthNumeric.z;
            if (all(isfinite(h)) && abs(h.w)>DepthNumeric.x)
            {
                float3 world=h.xyz/h.w;
                float margin=DepthAnchor.w*dot(abs(world)+abs(DepthAnchor.xyz),abs(DepthForward.xyz));
                float candidate=dot(world-DepthAnchor.xyz,DepthForward.xyz)-margin;
                if (isfinite(candidate)) value=candidate;
            }
        }
    }
    else for (uint child=0;child<4;++child)
    {
        // Repeating an odd edge leaves a minimum unchanged. Clamp before the
        // load because predicated SIMD lanes can issue speculative reads.
        uint2 p=min(pixel*2+uint2(child&1,child>>1),max(DepthSource.xy,1u)-1);
        value=min(value,asfloat(DepthPyramid[DepthSource.z+p.y*DepthSource.x+p.x]));
    }
    DepthPyramid[DepthDestination.z+index]=asuint(value);
}
#elif defined(EVANS_LUMEN_SCREEN_TRACE_STAGE)
// McGuire and Mara, "Efficient GPU Screen-Space Ray Tracing" (JCGT 2014).
// Perspective-correct DDA tests a finite depth slab. Unknown visibility falls
// back to the original mesh-SDF ray; this pass never samples GI feedback.
cbuffer HybridFrame:register(b0)
{ row_major float4x4 HybridProjection,HybridInverse; uint4 HybridCounts,HybridViewport; float4 HybridAnchor,HybridForward,HybridParameters; };
Texture2D<float> HybridDepth:register(t0);
Texture2D<float4> HybridNormal:register(t1),HybridScene:register(t2);
StructuredBuffer<ProbeOrigin> HybridOrigins:register(t3);
StructuredBuffer<uint> HybridPyramid:register(t4);
struct HybridHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct HybridRadiance { HybridHit Hit; float4 RadianceCoverage; uint4 Cache; };
RWStructuredBuffer<HybridHit> HybridHits:register(u0);
RWStructuredBuffer<HybridRadiance> HybridResults:register(u1);
static const uint HybridThreads=64,HybridScreenSurface=3,HybridComplete=1,HybridMissing=0xffffffffu;
static const float HybridNormalMinimum=1e-12,HybridParallelMinimum=1e-7,HybridPixelCenter=0.5,HybridClearDepth=1;
bool HybridClip(float a,float b,inout float first,inout float last)
{
    if (a<0 && b<0) return false;
    if (a<0) first=max(first,a/(a-b)); else if (b<0) last=min(last,a/(a-b));
    return first<=last;
}
bool HybridTrace(ProbeOrigin probe,float3 direction,out HybridHit hit,out float3 radiance)
{
    hit=(HybridHit)0; hit.Medium=float4(0,0,0,1); radiance=0; float3 origin=probe.PositionMinimum.xyz;
    float3 start=origin+direction*probe.PositionMinimum.w,end=origin+direction*HybridParameters.x;
    float4 a=mul(HybridProjection,float4(start,1)),b=mul(HybridProjection,float4(end,1));
    if (!all(isfinite(a)) || !all(isfinite(b))) return false;
    float first=0,last=1;
    if (!HybridClip(a.w+a.x,b.w+b.x,first,last) || !HybridClip(a.w-a.x,b.w-b.x,first,last) ||
        !HybridClip(a.w+a.y,b.w+b.y,first,last) || !HybridClip(a.w-a.y,b.w-b.y,first,last) ||
        !HybridClip(a.z,b.z,first,last) || !HybridClip(a.w-a.z,b.w-b.z,first,last) ||
        !HybridClip(a.w-HybridParameters.w,b.w-HybridParameters.w,first,last)) return false;
    float3 p0=lerp(start,end,first),p1=lerp(start,end,last); float4 h0=lerp(a,b,first),h1=lerp(a,b,last);
    float k0=1/h0.w,k1=1/h1.w;
    float2 screen0=(h0.xy*k0*float2(0.5,-0.5)+0.5)*HybridViewport.xy;
    float2 screen1=(h1.xy*k1*float2(0.5,-0.5)+0.5)*HybridViewport.xy;
    float span=max(abs(screen1.x-screen0.x),abs(screen1.y-screen0.y)); if (span<1) return false;
    float delta=HybridParameters.z/span,rate=dot(direction,HybridForward.xyz);
    uint nextStep=0,level=0;
    [loop] for (uint visit=0;visit<HybridViewport.z;++visit)
    {
        uint step=nextStep++;
        float u0=float(step)*delta; if (u0>=1) break; float u1=min(u0+delta,1),u=(u0+u1)*0.5;
        int2 pixel=int2(floor(lerp(screen0,screen1,u)));
        if (any(pixel<0) || any(pixel>=int2(HybridViewport.xy))) return false;
        if (all(uint2(pixel)==probe.Metadata.zw)) continue;
        if (HybridViewport.w!=0)
        {
            // Endpoint membership proves that every original DDA sample in the
            // batch belongs to this cell. Coarse levels only reject, never hit.
            uint2 cell=uint2(pixel)>>level; float2 motion=screen1-screen0;
            float2 low=float2(cell<<level),high=min(low+float(1u<<level),float2(HybridViewport.xy));
            float exit=1;
            if (motion.x!=0) exit=min(exit,((motion.x>0?high.x:low.x)-screen0.x)/motion.x);
            if (motion.y!=0) exit=min(exit,((motion.y>0?high.y:low.y)-screen0.y)/motion.y);
            uint stop=uint(clamp(ceil(exit/delta-HybridPixelCenter),float(step+1),ceil(1/delta)));
            float lastStart=float(stop-1)*delta,lastEnd=min(lastStart+delta,1);
            int2 lastPixel=int2(floor(lerp(screen0,screen1,(lastStart+lastEnd)*0.5)));
            // Roundoff at a cell exit can retain the adjacent sample. Restrict
            // that batch to the already checked current pixel before rejection.
            if (any(lastPixel<0) || any(lastPixel>=int2(HybridViewport.xy)) || any((uint2(lastPixel)>>level)!=cell))
            { stop=step+1; lastEnd=u1; }

            {
                static const uint HeaderWords=4;
                uint header=level*HeaderWords;
                float nearest=asfloat(HybridPyramid[HybridPyramid[header+2]+cell.y*HybridPyramid[header]+cell.x]);
                float3 beginPoint=lerp(p0*k0,p1*k1,u0)/lerp(k0,k1,u0),endPoint=lerp(p0*k0,p1*k1,lastEnd)/lerp(k0,k1,lastEnd);
                float upper=max(dot(beginPoint-origin,direction)*rate,dot(endPoint-origin,direction)*rate)+dot(origin-HybridAnchor.xyz,HybridForward.xyz);
                float margin=HybridAnchor.w*dot(abs(beginPoint)+abs(endPoint)+2*(abs(origin)+abs(HybridAnchor.xyz)),1.0.xxx);
                if (isfinite(upper+margin) && upper+margin<nearest)
                { nextStep=stop; level=min(level+1,HybridViewport.w-1); continue; }
            }
            if (level>0) { --level; nextStep=step; continue; }
        }
        float depth=HybridDepth.Load(int3(pixel,0)); if (!isfinite(depth) || depth<0 || depth>=HybridClearDepth) continue;
        float2 uv=(float2(pixel)+HybridPixelCenter)/HybridViewport.xy;
        float4 homogeneous=mul(HybridInverse,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
        if (!all(isfinite(homogeneous)) || abs(homogeneous.w)<=HybridParameters.w) return false;
        float3 surface=homogeneous.xyz/homogeneous.w;
        float3 q0=lerp(p0*k0,p1*k1,u0)/lerp(k0,k1,u0),q1=lerp(p0*k0,p1*k1,u1)/lerp(k0,k1,u1);
        float t0=dot(q0-origin,direction),t1=dot(q1-origin,direction),surfaceDepth=dot(surface-origin,HybridForward.xyz);
        float lower=min(t0*rate,t1*rate),upper=max(t0*rate,t1*rate);
        if (!all(isfinite(float4(t0,t1,surfaceDepth,upper)))) return false;
        if (upper<surfaceDepth) continue;
        if (lower>surfaceDepth+HybridParameters.y) return false;
        float3 normal=HybridNormal.Load(int3(pixel,0)).xyz;
        if (!all(isfinite(normal)) || dot(normal,normal)<=HybridNormalMinimum || dot(normal,direction)>=0) return false;
        float distance=abs(rate)>HybridParallelMinimum?clamp(surfaceDepth/rate,min(t0,t1),max(t0,t1)):(t0+t1)*0.5;
        if (distance<=probe.PositionMinimum.w || distance>HybridParameters.x) return false;
        radiance=HybridScene.Load(int3(pixel,0)).xyz; if (!all(isfinite(radiance))) return false;
        radiance=max(radiance,0); hit.PositionDistance=float4(surface,distance); hit.NormalResidual=float4(normalize(normal),0);
        hit.Metadata=uint4(uint2(pixel),HybridScreenSurface,visit+1); return true;
    }
    return false;
}
[numthreads(HybridThreads,1,1)]
void CSTraceScreen(uint3 dispatch:SV_DispatchThreadID)
{
    uint index=dispatch.x+dispatch.y*HybridCounts.w; if (index>=HybridCounts.x) return;
    ProbeOrigin probe=HybridOrigins[index/HybridCounts.y]; HybridRadiance result=(HybridRadiance)0; result.Hit.Medium=float4(0,0,0,1);
    result.Cache.x=HybridMissing; result.Hit.Metadata=uint4(HybridMissing,HybridMissing,0,0);
    HybridHit hit; hit.Medium=float4(0,0,0,1); float3 light;
    float3 direction=ProbeSphereDirection(index%HybridCounts.y,HybridCounts.y,probe.Metadata.x,HybridCounts.z);
    if (HybridForward.w!=0)
    { float4 reflected=asfloat(probe.Metadata); direction=reflected.xyz; probe.Metadata.y=reflected.w>0?0:1; }
    if (probe.Metadata.y==0 && HybridTrace(probe,direction,hit,light))
    { result.Hit=hit; result.RadianceCoverage=float4(light,1); result.Cache.z=HybridComplete; }
    HybridHits[index]=result.Hit; HybridResults[index]=result;
}
#elif defined(EVANS_LUMEN_SCREEN_PROBES_STAGE)
// Bounded adaptive screen caching: each tile retains its uniform probe and
// reserves breadth-first quadtree slots for surfaces absent from every ancestor.
cbuffer ScreenFrame:register(b0)
{ row_major float4x4 ScreenInverse; uint4 ScreenViewport,ScreenGrid; float4 ScreenParameters,ScreenAdaptive; };
Texture2D<float> ScreenDepth:register(t0);
Texture2D<float4> ScreenNormal:register(t1);
RWStructuredBuffer<ProbeOrigin> ScreenOrigins:register(u0);
static const uint ScreenThreads=64,ScreenBackground=1,ScreenInvalid=2,ScreenMissing=0xffffffffu;
static const uint ScreenMaximumLevels=2,ScreenChildren=4;
static const float ScreenClearDepth=1,ScreenMinimumNormalSquared=1e-12;
float ScreenMagnitude(float3 value) { return max(abs(value.x),max(abs(value.y),abs(value.z))); }
bool ScreenSurface(uint2 pixel,out float3 world,out float3 normal)
{
    world=normal=0; float2 uv=(float2(pixel)+0.5)/float2(ScreenViewport.xy);
    float4 h=mul(ScreenInverse,float4(uv.x*2-1,1-uv.y*2,ScreenDepth.Load(int3(pixel,0)),1));
    if (!all(isfinite(h)) || abs(h.w)<=ScreenParameters.w) return false;
    world=h.xyz/h.w; normal=normalize(ScreenNormal.Load(int3(pixel,0)).xyz);
    return all(isfinite(world)) && all(abs(world)+ScreenParameters.x<=ScreenParameters.z);
}
uint2 SelectScreenPixel(uint2 begin,uint2 end,uint ancestorCount,float3 anchors[ScreenMaximumLevels],float3 normals[ScreenMaximumLevels],out bool valid)
{
    uint best=ScreenMissing; uint2 chosen=0;
    [loop] for (uint row=begin.y;row<end.y;++row) [loop] for (uint column=begin.x;column<end.x;++column)
    {
        uint2 pixel=uint2(column,row); float d=ScreenDepth.Load(int3(pixel,0)); float4 material=ScreenNormal.Load(int3(pixel,0));
        if (!isfinite(d) || d<0 || d>=ScreenClearDepth || !all(isfinite(material)) || material.w<0 || dot(material.xyz,material.xyz)<=ScreenMinimumNormalSquared) continue;
        if (ancestorCount!=0)
        {
            float3 p,n; if (!ScreenSurface(pixel,p,n)) continue;
            bool represented=false;
            [unroll] for (uint parent=0;parent<ScreenMaximumLevels;++parent) if (parent<ancestorCount)
            {
                float3 difference=p-anchors[parent]; float magnitude=ScreenMagnitude(difference);
                float distance=magnitude>0?magnitude*length(difference/magnitude):0;
                float plane=max(abs(dot(difference,n)),abs(dot(difference,normals[parent])));
                float allowed=ScreenAdaptive.y+ScreenAdaptive.z*distance+ScreenAdaptive.w*max(ScreenMagnitude(p),ScreenMagnitude(anchors[parent]));
                represented=represented || (dot(n,normals[parent])>ScreenAdaptive.x && plane<=allowed);
            }
            if (represented) continue;
        }
        int2 delta=int2(pixel*2+1)-int2(begin+end); uint squared=uint(dot(delta,delta));
        if (squared<best) { chosen=pixel; best=squared; }
    }
    valid=best!=ScreenMissing; return chosen;
}
[numthreads(ScreenThreads,1,1)]
void CSPlaceScreenProbes(uint3 dispatch:SV_DispatchThreadID)
{
    uint index=dispatch.x; if (index>=ScreenGrid.y) return;
    uint count=ScreenGrid.w+1,tile=index/count,refinement=index%count;
    uint2 first=uint2(tile%ScreenViewport.w,tile/ScreenViewport.w)*ScreenViewport.z,end=min(first+ScreenViewport.z,ScreenViewport.xy);
    float3 anchors[ScreenMaximumLevels],normals[ScreenMaximumLevels];
    [unroll] for (uint zero=0;zero<ScreenMaximumLevels;++zero) { anchors[zero]=0; normals[zero]=0; }
    uint path[ScreenMaximumLevels],depth=0;
    for (uint node=refinement;node!=0;node=(node-1)/ScreenChildren) path[depth++]=(node-1)%ScreenChildren;
    bool valid; uint2 selected=SelectScreenPixel(first,end,0,anchors,normals,valid);
    ProbeOrigin result; result.PositionMinimum=0; result.Metadata=uint4(index,ScreenBackground,0,0);
    float3 world=0,normal=0;
    for (uint level=0;level<depth;++level)
    {
        if (!valid || !ScreenSurface(selected,world,normal)) { ScreenOrigins[index]=result; return; }
        anchors[level]=world; normals[level]=normal;
        uint quadrant=path[depth-1-level]; uint2 middle=(first+end)/2;
        first=uint2((quadrant&1)?middle.x:first.x,(quadrant&2)?middle.y:first.y);
        end=uint2((quadrant&1)?end.x:middle.x,(quadrant&2)?end.y:middle.y);
        selected=SelectScreenPixel(first,end,level+1,anchors,normals,valid);
    }
    if (valid)
    {
        result.Metadata=uint4(index,ScreenInvalid,selected);
        if (ScreenSurface(selected,world,normal))
        { result.PositionMinimum=float4(world+normal*ScreenParameters.x,ScreenParameters.y); result.Metadata.y=0; }
    }
    ScreenOrigins[index]=result;
}
#elif defined(EVANS_LUMEN_PROBE_INTEGRATE_STAGE)
// Equal-solid-angle projection followed by clamped-cosine convolution.
// SH order: 1, y, z, x, xy, yz, 3z*z-1, xz, x*x-y*y.
cbuffer ProbeFrame:register(b0) { uint4 ProbeCounts; float4 ProbeSkyMeasure,EnvironmentAxis,EnvironmentPeak; };
struct ProbeRayHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct ProbeRayLight { ProbeRayHit Hit; float4 RadianceCoverage; uint4 Cache; };
struct ProbeIrradiance { float4 Coefficients[9]; float4 Coverage; uint4 TraceCounts,CacheCounts,Metadata; };
StructuredBuffer<ProbeRayLight> ProbeRadiance:register(t0);
StructuredBuffer<ProbeOrigin> ProbePositions:register(t1);
RWStructuredBuffer<ProbeIrradiance> ProbeResults:register(u0);
static const uint ProbeBandCoefficients=9,ProbeThreadWidth=64;
static const uint ProbeComplete=1,ProbePartial=2,ProbeInvalid=3,ProbeSurface=1,ProbeMiss=0;
float ProbeIrradianceBasis(uint coefficient,float3 direction)
{
    static const float Pi=3.14159265358979323846;
    float dc=Pi*sqrt(1/(4*Pi)),linearBand=(2*Pi/3)*sqrt(3/(4*Pi));
    float mixed=(Pi/4)*sqrt(15/(4*Pi)),axial=(Pi/4)*sqrt(5/(16*Pi)),difference=(Pi/4)*sqrt(15/(16*Pi));
    float x=direction.x,y=direction.y,z=direction.z;
    switch (coefficient)
    {
        case 0: return dc;
        case 1: return linearBand*y;
        case 2: return linearBand*z;
        case 3: return linearBand*x;
        case 4: return mixed*x*y;
        case 5: return mixed*y*z;
        case 6: return axial*(3*z*z-1);
        case 7: return mixed*x*z;
        case 8: return difference*(x*x-y*y);
    }
    return 0;
}
[numthreads(ProbeThreadWidth,1,1)]
void CSIntegrateProbes(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*ProbeCounts.w; if (index>=ProbeCounts.x) return;
    ProbeIrradiance result; [unroll] for (uint coefficient=0;coefficient<ProbeBandCoefficients;++coefficient) result.Coefficients[coefficient]=0;
    result.Coverage=0; result.TraceCounts=0; result.CacheCounts=0;
    uint id=ProbePositions[index].Metadata.x; float covered=0,meanDistance=0; bool invalid=ProbePositions[index].Metadata.y!=0;
    [loop] for (uint sample=0;sample<ProbeCounts.y;++sample)
    {
        ProbeRayLight ray=ProbeRadiance[index*ProbeCounts.y+sample]; float3 incoming=0;
        float3 direction=ProbeSphereDirection(sample,ProbeCounts.y,id,ProbeCounts.z);
        if (ray.Hit.Metadata.z==ProbeSurface || ray.Hit.Metadata.z==LumenScreenSurface || ray.Hit.Metadata.z==LumenWorldCache)
        {
            if (ray.Hit.Metadata.z==LumenWorldCache) ++result.TraceCounts.w;
            else { ++result.TraceCounts.x; meanDistance+=ray.Hit.PositionDistance.w/float(ProbeCounts.y); }
            if (ray.Cache.z==ProbeComplete) { ++result.CacheCounts.x; incoming=ray.RadianceCoverage.xyz; covered+=ray.RadianceCoverage.w; }
            else if (ray.Cache.z==ProbePartial) { ++result.CacheCounts.y; incoming=ray.RadianceCoverage.xyz; covered+=ray.RadianceCoverage.w; }
            else if (ray.Cache.z==ProbeInvalid) { ++result.CacheCounts.w; invalid=true; }
            else ++result.CacheCounts.z;
        }
        else if (ray.Hit.Metadata.z==ProbeMiss) { ++result.TraceCounts.y; incoming=LumenEnvironmentRadiance(ProbeSkyMeasure.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction); }
        else ++result.TraceCounts.z;
        incoming=incoming*ray.Hit.Medium.w+ray.Hit.Medium.xyz;
        [unroll] for (uint term=0;term<ProbeBandCoefficients;++term)
            result.Coefficients[term]+=float4(incoming*(ProbeSkyMeasure.w*ProbeIrradianceBasis(term,direction)),0);
    }
    result.Coverage=float4(float(result.TraceCounts.y+result.CacheCounts.x)/ProbeCounts.y,
        (result.TraceCounts.x+result.TraceCounts.w)?covered/(result.TraceCounts.x+result.TraceCounts.w):0,float(result.TraceCounts.y)/ProbeCounts.y,
        result.TraceCounts.x?meanDistance*(float(ProbeCounts.y)/result.TraceCounts.x):0);
    [unroll] for (uint check=0;check<ProbeBandCoefficients;++check) invalid=invalid || !all(isfinite(result.Coefficients[check]));
    invalid=invalid || !all(isfinite(result.Coverage));
    // Rounded division can put an otherwise valid mean one ULP above unity.
    result.Coverage.y=saturate(result.Coverage.y); uint state=(result.TraceCounts.z || result.CacheCounts.y || result.CacheCounts.z)?ProbePartial:ProbeComplete;
    if (invalid) { state=ProbeInvalid; result.Coverage=0; [unroll] for (uint clear=0;clear<ProbeBandCoefficients;++clear) result.Coefficients[clear]=0; }
    result.Metadata=uint4(id,state,ProbeCounts.y,ProbeCounts.z); ProbeResults[index]=result;
}
#elif defined(EVANS_LUMEN_RADIANCE_RESOLVE_STAGE)
// Native SDF hits are resolved directly against their immutable lighting
// version. Geometry coverage and unresolved lighting remain separate signals.
cbuffer RadianceFrame:register(b0) { uint4 ResolveCounts,ResolveAtlas,ResolveMapping; float4 ResolveParameters,ResolvePrecision; };
struct RadianceHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct RadianceLight { float4 RadianceCoverage; uint4 Metadata; };
struct RadianceAnswer { RadianceHit Hit; float4 RadianceCoverage; uint4 Cache; };
Texture2D<float4> ResolveNormal:register(t0);
StructuredBuffer<RadianceHit> ResolveHits:register(t1);
StructuredBuffer<RadianceLight> ResolveLighting:register(t2);
StructuredBuffer<uint4> ResolveData:register(t3);
RWStructuredBuffer<RadianceAnswer> ResolveAnswers:register(u0);
static const uint ResolveThreads=64,ResolveDirections=6,ResolveCardVectors=11,ResolveInverse=4,ResolveOrigin=8,ResolveBackward=9,ResolveTile=10,ResolveBorder=1;
static const uint ResolveMissing=0xffffffffu,ResolveSurface=1,ResolveComplete=1,ResolvePartial=2,ResolveInvalid=3;
float4 ResolveVector(uint at) { return asfloat(ResolveData[at]); }
float ResolveMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float3 ResolveTransform(uint base,float4 v)
{ return float3(dot(ResolveVector(base),v),dot(ResolveVector(base+1),v),dot(ResolveVector(base+2),v)); }
bool ResolveCard(uint card,RadianceHit hit,float radius,out float4 radianceCoverage,out float error,out float facing,out uint state)
{
    radianceCoverage=0; error=0; state=0; uint base=ResolveMapping.x+card*ResolveCardVectors;
    uint4 tile=ResolveData[base+ResolveTile]; float4 backward=ResolveVector(base+ResolveBackward);
    facing=dot(hit.NormalResidual.xyz,backward.xyz); if (facing<ResolveParameters.x) return false;
    float3 relative=hit.PositionDistance.xyz-ResolveVector(base+ResolveOrigin).xyz;
    float3 projected=ResolveTransform(base,float4(relative,1)); if (!all(isfinite(projected)) || any(abs(projected.xy)>1)) return false;
    float2 uv=float2(projected.x*0.5+0.5,0.5-projected.y*0.5),location=float2(tile.xy)+ResolveBorder+uv*ResolveCounts.y-0.5;
    int2 lower=int2(floor(location)); float2 fraction=location-float2(lower); float coverage=0; state=ResolveComplete;
    [unroll] for (uint tap=0;tap<4;++tap)
    {
        uint2 corner=uint2(tap&1,(tap>>1)&1); int2 pixel=lower+int2(corner);
        float weight=(corner.x!=0?fraction.x:1-fraction.x)*(corner.y!=0?fraction.y:1-fraction.y);
        float4 normal=ResolveNormal.Load(int3(pixel,0)); if (weight<=0 || dot(normal.xyz,normal.xyz)==0 || dot(normal.xyz,hit.NormalResidual.xyz)<ResolveParameters.y) continue;
        float2 texelUv=(float2(pixel)-float2(tile.xy)-ResolveBorder+0.5)/ResolveCounts.y;
        float3 captured=ResolveTransform(base+ResolveInverse,float4(texelUv.x*2-1,1-texelUv.y*2,normal.w,1));
        float planeError=abs(dot(relative-captured,hit.NormalResidual.xyz));
        float roundoff=ResolvePrecision.x*max(backward.w,max(ResolveMagnitude(relative),ResolveMagnitude(captured)))+ResolvePrecision.y*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!isfinite(planeError) || planeError>radius+roundoff) continue;
        RadianceLight light=ResolveLighting[uint(pixel.y)*ResolveAtlas.x+uint(pixel.x)];
        state=max(state,light.Metadata.x==0?ResolveInvalid:light.Metadata.x); radianceCoverage.xyz+=light.RadianceCoverage.xyz*weight;
        error+=planeError*weight; coverage+=weight;
    }
    if (coverage<=0) { state=0; return false; }
    radianceCoverage=float4(radianceCoverage.xyz/coverage,saturate(coverage)); error/=coverage; return true;
}
[numthreads(ResolveThreads,1,1)]
void CSResolveRadiance(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*ResolveAtlas.w; if (index>=ResolveCounts.x) return;
    RadianceAnswer result; result.Hit.Medium=float4(0,0,0,1); result.Hit=ResolveHits[index]; result.RadianceCoverage=0; result.Cache=uint4(ResolveMissing,0,0,0);
    if (result.Hit.Metadata.z==LumenScreenSurface || result.Hit.Metadata.z==LumenWorldCache) { ResolveAnswers[index].Hit.Medium=result.Hit.Medium; return; } // Prior transport owns the base radiance.
    float radius=ResolveParameters.z+result.Hit.Uncertainty.z*ResolveParameters.w;
    if (ResolveMapping.z==0) result.Hit.Uncertainty.z=0;
    [branch] if (result.Hit.Metadata.z!=ResolveSurface) { ResolveAnswers[index]=result; return; }
    [branch] if (!isfinite(radius)) { result.Cache.z=ResolveInvalid; ResolveAnswers[index]=result; return; }
    uint4 range=ResolveData[result.Hit.Metadata.x];
    [branch] if (range.y==0) { ResolveAnswers[index]=result; return; }
    float4 values[ResolveDirections]; float errors[ResolveDirections],facings[ResolveDirections]; uint states[ResolveDirections],cards[ResolveDirections];
    [unroll] for (uint initialize=0;initialize<ResolveDirections;++initialize)
    { values[initialize]=0; errors[initialize]=0; facings[initialize]=0; states[initialize]=0; cards[initialize]=ResolveMissing; }
    [loop] for (uint candidate=0;candidate<range.y;++candidate)
    {
        uint word=range.x+candidate,card=ResolveData[word/4][word%4],group=ResolveData[ResolveMapping.x+card*ResolveCardVectors+ResolveTile].z;
        float4 value; float error,facing; uint state;
        if (!ResolveCard(card,result.Hit,radius,value,error,facing,state)) continue;
        if (states[group]==0 || error<errors[group] || (error==errors[group] && value.w>values[group].w))
        { values[group]=value; errors[group]=error; facings[group]=facing; states[group]=state; cards[group]=card; }
    }
    float total=0,available=0,dominant=0; uint status=ResolveComplete;
    [unroll] for (uint measure=0;measure<ResolveDirections;++measure) if (states[measure]!=0)
    { float orientation=facings[measure]*facings[measure]; total+=orientation*values[measure].w; available+=orientation; }
    if (total>0)
    {
        [unroll] for (uint combine=0;combine<ResolveDirections;++combine) if (states[combine]!=0)
        {
            float weight=facings[combine]*facings[combine]*values[combine].w/total;
            result.RadianceCoverage.xyz+=values[combine].xyz*weight; ++result.Cache.y; status=max(status,states[combine]);
            if (states[combine]!=ResolveComplete) ++result.Cache.w;
            if (weight>dominant || (weight==dominant && cards[combine]<result.Cache.x)) { dominant=weight; result.Cache.x=cards[combine]; }
        }
        result.Cache.z=status; result.RadianceCoverage.w=saturate(total/available);
        if (status==ResolveInvalid || !all(isfinite(result.RadianceCoverage))) { result.RadianceCoverage=0; result.Cache.z=ResolveInvalid; }
    }
    ResolveAnswers[index]=result;
}
#elif defined(EVANS_LUMEN_LOOKUP_STAGE)
// Reconstruct each tap's world position before depth acceptance. Comparing in
// the query's tangent plane preserves sloped surfaces without mixing layers.
cbuffer LookupFrame:register(b0) { uint4 LookupCounts,LookupAtlas; float4 LookupParameters; };
Texture2D<float4> LookupAlbedo:register(t0),LookupNormal:register(t1),LookupEmission:register(t2);
struct LookupCard { row_major float4x4 Projection,InverseProjection; float4 Origin,BackwardScale; uint4 Tile; };
struct LookupQuery { float4 PositionRadius,NormalMinimumCosine; uint4 Range; };
struct LookupSample { float4 AlbedoMetallic,NormalCoverage,EmissionRoughness; uint4 Metadata; };
StructuredBuffer<LookupCard> LookupCards:register(t3);
StructuredBuffer<LookupQuery> LookupQueries:register(t4);
RWStructuredBuffer<LookupSample> LookupSamples:register(u0);
static const uint LookupThreads=64,LookupDirections=6,LookupBorder=1,LookupInvalid=0xffffffffu;
float LookupMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
LookupSample EmptyLookup()
{ LookupSample value; value.AlbedoMetallic=0; value.NormalCoverage=0; value.EmissionRoughness=0; value.Metadata=uint4(LookupInvalid,0,0,0); return value; }
bool SampleLookupCard(LookupCard card,LookupQuery query,out LookupSample value,out float error,out float facing)
{
    value=EmptyLookup(); error=0; facing=dot(query.NormalMinimumCosine.xyz,card.BackwardScale.xyz);
    if (facing<LookupParameters.x) return false;
    float3 relative=query.PositionRadius.xyz-card.Origin.xyz;
    float3 projected=mul(card.Projection,float4(relative,1)).xyz;
    if (!all(isfinite(projected)) || any(abs(projected.xy)>1)) return false;
    float2 uv=float2(projected.x*0.5+0.5,0.5-projected.y*0.5);
    float2 location=float2(card.Tile.xy)+LookupBorder+uv*LookupCounts.y-0.5;
    int2 lower=int2(floor(location)); float2 fraction=location-float2(lower);
    float coverage=0; float3 normal=0;
    [unroll] for (uint tap=0;tap<4;++tap)
    {
        uint2 corner=uint2(tap&1,(tap>>1)&1); int2 pixel=lower+int2(corner);
        float weight=(corner.x!=0?fraction.x:1-fraction.x)*(corner.y!=0?fraction.y:1-fraction.y);
        float4 materialNormal=LookupNormal.Load(int3(pixel,0));
        if (weight<=0 || dot(materialNormal.xyz,materialNormal.xyz)==0 || dot(materialNormal.xyz,query.NormalMinimumCosine.xyz)<query.NormalMinimumCosine.w) continue;
        float2 texelUv=(float2(pixel)-float2(card.Tile.xy)-LookupBorder+0.5)/LookupCounts.y;
        float3 captured=mul(card.InverseProjection,float4(texelUv.x*2-1,1-texelUv.y*2,materialNormal.w,1)).xyz;
        float planeError=abs(dot(relative-captured,query.NormalMinimumCosine.xyz));
        float roundoff=LookupParameters.y*max(card.BackwardScale.w,max(LookupMagnitude(relative),LookupMagnitude(captured)))+
            LookupParameters.z*dot(abs(query.PositionRadius.xyz),abs(query.NormalMinimumCosine.xyz));
        if (!isfinite(planeError) || planeError>query.PositionRadius.w+roundoff) continue;
        value.AlbedoMetallic+=LookupAlbedo.Load(int3(pixel,0))*weight; value.EmissionRoughness+=LookupEmission.Load(int3(pixel,0))*weight;
        normal+=materialNormal.xyz*weight; error+=planeError*weight; coverage+=weight;
    }
    if (coverage<=0) return false;
    value.AlbedoMetallic/=coverage; value.EmissionRoughness/=coverage; normal/=coverage;
    value.NormalCoverage=float4(normal,saturate(coverage)); value.Metadata=uint4(card.Tile.w,1,1,0); error/=coverage; return true;
}
[numthreads(LookupThreads,1,1)]
void CSLookup(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*LookupAtlas.z; if (index>=LookupCounts.x) return;
    LookupQuery query=LookupQueries[index]; LookupSample groups[LookupDirections]; float errors[LookupDirections],facings[LookupDirections];
    [branch] if (query.Range.y==0) { LookupSamples[index]=EmptyLookup(); return; }
    [unroll] for (uint initialize=0;initialize<LookupDirections;++initialize) { groups[initialize]=EmptyLookup(); errors[initialize]=0; facings[initialize]=0; }
    [loop] for (uint candidate=0;candidate<query.Range.y;++candidate)
    {
        LookupCard card=LookupCards[query.Range.x+candidate]; LookupSample value; float error,facing;
        if (!SampleLookupCard(card,query,value,error,facing)) continue;
        uint group=card.Tile.z;
        if (groups[group].Metadata.z==0 || error<errors[group] || (error==errors[group] && value.NormalCoverage.w>groups[group].NormalCoverage.w))
        { groups[group]=value; errors[group]=error; facings[group]=facing; }
    }
    LookupSample result=EmptyLookup(); float total=0,available=0,dominant=0;
    [unroll] for (uint measure=0;measure<LookupDirections;++measure) if (groups[measure].Metadata.z!=0)
    { float orientation=facings[measure]*facings[measure]; total+=orientation*groups[measure].NormalCoverage.w; available+=orientation; }
    if (total>0)
    {
        float3 normal=0;
        [unroll] for (uint combine=0;combine<LookupDirections;++combine) if (groups[combine].Metadata.z!=0)
        {
            float weight=facings[combine]*facings[combine]*groups[combine].NormalCoverage.w/total;
            result.AlbedoMetallic+=groups[combine].AlbedoMetallic*weight; result.EmissionRoughness+=groups[combine].EmissionRoughness*weight;
            normal+=groups[combine].NormalCoverage.xyz*weight; ++result.Metadata.y;
            if (weight>dominant || (weight==dominant && groups[combine].Metadata.x<result.Metadata.x)) { dominant=weight; result.Metadata.x=groups[combine].Metadata.x; }
        }
        float squared=dot(normal,normal); result.NormalCoverage=float4(squared>0?normal*rsqrt(squared):query.NormalMinimumCosine.xyz,saturate(total/available)); result.Metadata.z=1;
    }
    result.AlbedoMetallic=saturate(result.AlbedoMetallic); result.EmissionRoughness.w=saturate(result.EmissionRoughness.w);
    LookupSamples[index]=result;
}
#elif defined(EVANS_LUMEN_DISTANCE_TRACE_STAGE) || defined(EVANS_LUMEN_LIGHTING_STAGE) || defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE) || defined(EVANS_LUMEN_PROBE_TRACE_STAGE) || defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE) || defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
// Packed word records match LumenDistanceScene.h. A single scene SRV retains
// integer addresses and two signed 16-bit samples per word without float IDs.
cbuffer DistanceTraceFrame:register(b0)
{
    uint4 Counts,Offsets,Traversal; float4 Parameters,HairSky;
#if defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
    row_major float4x4 LinkInverse; uint4 LinkViewport,LinkGrid; float4 LinkOrigin,LinkAdaptive;
    float4 LinkAppearance,LinkNumeric; uint4 LinkModel; float4 LinkVisibility;
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uint4 ProbeSampling; float4 ProbeRange;
#endif
#if defined(EVANS_LUMEN_LIGHTING_STAGE)
    uint4 LightingAtlas,LightingWork; float4 LightingParameters,LightingNumeric; uint4 LightingTiles,LightingRadiosity; float4 LightingSkyDistance,EnvironmentAxis,EnvironmentPeak;
#endif
};
#if defined(EVANS_LUMEN_LIGHTING_STAGE)
StructuredBuffer<uint> DistanceScene:register(t3);
Texture2D<float4> LightingAlbedo:register(t0),LightingNormal:register(t1),LightingEmission:register(t2);
StructuredBuffer<uint4> LightingData:register(t4);
struct LightingTexel { float4 RadianceCoverage; uint4 Metadata; };
RWStructuredBuffer<LightingTexel> LitSurface:register(u0);
#else
StructuredBuffer<uint> DistanceScene:register(t0);
#endif
struct DistanceRay { float4 OriginMinimum,DirectionMaximum; };
struct DistanceHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
#if defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
StructuredBuffer<ProbeOrigin> LinkOrigins:register(t1);
Texture2D<float> LinkDepth:register(t2);
Texture2D<float4> LinkNormals:register(t3);
RWStructuredBuffer<uint> LinkMask:register(u0);
#elif defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
StructuredBuffer<ProbeOrigin> VolumeOrigins:register(t1);
RWStructuredBuffer<float> VolumeSupport:register(u0);
#elif !defined(EVANS_LUMEN_LIGHTING_STAGE)
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
StructuredBuffer<ProbeOrigin> ProbeOrigins:register(t1);
#else
StructuredBuffer<DistanceRay> DistanceRays:register(t1);
#endif
RWStructuredBuffer<DistanceHit> DistanceHits:register(u0);
#endif
static const uint TraceThreads=64,FieldWords=20,InstanceWords=20,NodeWords=12,VectorWords=4;
static const uint FieldGrid=0,FieldBricks=4,FieldStorage=8,FieldMinimum=12,FieldDistances=16;
static const uint InstanceCenter=0,InstanceRows=4,InstanceMetadata=16,NodeMinimum=0,NodeMaximum=4,NodeLinks=8;
static const uint BrickInterior=6,BrickBorder=1,BrickSide=8,BrickSamples=512,FirstBrick=2,NegativeBrick=1,InvalidAddress=0xffffffffu;
static const uint TraceMiss=0,TraceSurface=1,TraceUnresolved=2;
static const float SignedSampleMaximum=32767;
uint4 DistanceWords(uint address)
{ return uint4(DistanceScene[address],DistanceScene[address+1],DistanceScene[address+2],DistanceScene[address+3]); }
float4 DistanceFloats(uint address) { return asfloat(DistanceWords(address)); }
float3 DistanceTransform(uint instance,float3 value)
{
    return float3(dot(DistanceFloats(instance+InstanceRows).xyz,value),dot(DistanceFloats(instance+InstanceRows+VectorWords).xyz,value),dot(DistanceFloats(instance+InstanceRows+VectorWords*2).xyz,value));
}
float DistanceLength(float3 value)
{ float scale=max(abs(value.x),max(abs(value.y),abs(value.z))); return scale>0?scale*length(value/scale):0; }
float3 DistanceUnit(float3 value,float3 fallback)
{
    if (!all(isfinite(value))) return fallback;
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z))); if (scale==0) return fallback;
    float3 scaled=value/scale; return scaled*rsqrt(dot(scaled,scaled));
}
bool DistanceSlab(float3 origin,float3 direction,float3 minimum,float3 maximum,inout float enter,inout float leave)
{
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (direction[axis]==0) { if (origin[axis]<minimum[axis] || origin[axis]>maximum[axis]) return false; }
        else
        {
            float a=(minimum[axis]-origin[axis])/direction[axis],b=(maximum[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(a,b)); leave=min(leave,max(a,b)); if (enter>leave) return false;
        }
    }
    return true;
}
float DistanceValue(uint field,float3 position)
{
    uint3 dimensions=DistanceWords(field+FieldGrid).xyz,brickDimensions=DistanceWords(field+FieldBricks).xyz;
    uint4 storage=DistanceWords(field+FieldStorage); float4 grid=DistanceFloats(field+FieldMinimum);
    float band=DistanceFloats(field+FieldDistances).x;
    float3 coordinate=clamp((position-grid.xyz)/grid.w-0.5,0,float3(dimensions)-1);
    uint3 cell=uint3(floor(coordinate)),brick=cell/BrickInterior,local=cell%BrickInterior+BrickBorder; float3 fraction=coordinate-float3(cell);
    uint entry=DistanceScene[storage.x+(brick.z*brickDimensions.y+brick.y)*brickDimensions.x+brick.x];
    if (entry<FirstBrick) return entry==NegativeBrick?-band:band;
    float value=0;
    [unroll] for (uint z=0;z<2;++z) [unroll] for (uint y=0;y<2;++y) [unroll] for (uint x=0;x<2;++x)
    {
        uint sampleIndex=(entry-FirstBrick)*BrickSamples+((local.z+z)*BrickSide+local.y+y)*BrickSide+local.x+x;
        uint packed=DistanceScene[storage.y+sampleIndex/2]; int encoded=(sampleIndex&1)?(int(packed)>>16):(int(packed<<16)>>16);
        float weight=(x!=0?fraction.x:1-fraction.x)*(y!=0?fraction.y:1-fraction.y)*(z!=0?fraction.z:1-fraction.z);
        value+=float(encoded)/SignedSampleMaximum*weight;
    }
    return value*band;
}
float3 DistanceNormal(uint instance,uint field,float3 position,float cell,float3 fallback)
{
    float3 g=float3(DistanceValue(field,position+float3(cell,0,0))-DistanceValue(field,position-float3(cell,0,0)),
        DistanceValue(field,position+float3(0,cell,0))-DistanceValue(field,position-float3(0,cell,0)),
        DistanceValue(field,position+float3(0,0,cell))-DistanceValue(field,position-float3(0,0,cell)));
    g=DistanceUnit(g,float3(0,0,0));
    float3 world=DistanceFloats(instance+InstanceRows).xyz*g.x+DistanceFloats(instance+InstanceRows+VectorWords).xyz*g.y+DistanceFloats(instance+InstanceRows+VectorWords*2).xyz*g.z;
    return DistanceUnit(world,fallback);
}
uint TraceDistanceInstance(DistanceRay ray,uint index,float anchor,float limit,out float at,out float4 normalResidual,out float2 uncertainty,out uint steps)
{
    at=anchor; normalResidual=0; uncertainty=0; steps=0;
    uint instance=Offsets.y+index*InstanceWords,field=Offsets.x+DistanceWords(instance+InstanceMetadata).x*FieldWords;
    float3 localOrigin=DistanceTransform(instance,ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*anchor-DistanceFloats(instance+InstanceCenter).xyz);
    float3 localDirection=DistanceTransform(instance,ray.DirectionMaximum.xyz); float speed=DistanceLength(localDirection);
    if (!all(isfinite(localOrigin)) || !all(isfinite(localDirection)) || !isfinite(speed) || speed<=0) return TraceUnresolved;
    float4 grid=DistanceFloats(field+FieldMinimum),metric=DistanceFloats(field+FieldDistances);
    float enter=0,leave=max(0,limit-anchor); float3 maximum=grid.xyz+float3(DistanceWords(field+FieldGrid).xyz)*grid.w;
    if (!DistanceSlab(localOrigin,localDirection,grid.xyz,maximum,enter,leave)) return TraceMiss;
    float travel=enter,threshold=metric.y+Parameters.x*grid.w;
    [loop] for (uint iteration=0;iteration<Traversal.x;++iteration)
    {
        at=min(limit,anchor+travel); float3 position=localOrigin+localDirection*travel; ++steps;
        if (!all(isfinite(position))) return TraceUnresolved;
        float distance=DistanceValue(field,position);
        if (abs(distance)<=threshold)
        {
            normalResidual=float4(DistanceNormal(instance,field,position,grid.w,-ray.DirectionMaximum.xyz),distance/speed);
            uncertainty=float2(metric.y,threshold)/speed; return TraceSurface;
        }
        // Convert the safe normalized radius to world-ray parameter distance.
        // This remains conservative under rotation, mirroring, shear and scale.
        float next=travel+(abs(distance)-metric.y)/speed*Parameters.y;
        if (!isfinite(next) || next<=travel) return TraceUnresolved;
        if (next>leave) return TraceMiss; travel=next;
    }
    at=min(limit,anchor+travel); return TraceUnresolved;
}
// With local = A * world, local plane distance per world-normal step is
// 1 / length(inverse(transpose(A)) * normal), not length(A * normal).
float DistanceNormalRate(uint instance,float3 normal)
{
    float3 x=DistanceFloats(instance+InstanceRows).xyz,y=DistanceFloats(instance+InstanceRows+VectorWords).xyz,z=DistanceFloats(instance+InstanceRows+VectorWords*2).xyz;
    float3 largest=max(abs(x),max(abs(y),abs(z))); float scale=max(largest.x,max(largest.y,largest.z)); if (scale==0) return 0;
    x/=scale; y/=scale; z/=scale; float3 cx=cross(y,z),cy=cross(z,x),cz=cross(x,y);
    float covector=DistanceLength(float3(dot(cx,normal),dot(cy,normal),dot(cz,normal)));
    float speed=covector>0?scale*(abs(dot(x,cx))/covector):0; return isfinite(speed)?speed:0;
}
void TraceLocalOpaqueRay(DistanceRay ray,out DistanceHit result,uint requiredFlags)
{
    result.Medium=float4(0,0,0,1);
    result.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
    result.NormalResidual=0; result.Uncertainty=0; result.Metadata=uint4(InvalidAddress,InvalidAddress,TraceMiss,0);
    [branch] if (Counts.y==0) return;
    float closest=ray.DirectionMaximum.w,unknown=ray.DirectionMaximum.w;
    uint nearest=InvalidAddress,pending=InvalidAddress,totalSteps=0,node=0; float4 chosenNormal=0; float2 chosenUncertainty=0;
    [loop] while (node<Counts.y)
    {
        uint address=Offsets.z+node*NodeWords; uint4 links=DistanceWords(address+NodeLinks);
        float enter=ray.OriginMinimum.w,leave=min(closest,unknown);
        if (!DistanceSlab(ray.OriginMinimum.xyz,ray.DirectionMaximum.xyz,DistanceFloats(address+NodeMinimum).xyz,DistanceFloats(address+NodeMaximum).xyz,enter,leave)) { node=links.w; continue; }
        if (links.z==0) { node=links.x; continue; }
        // Required flags select shadow casters without hiding GI surfaces.
        uint flags=DistanceWords(Offsets.y+links.y*InstanceWords+InstanceMetadata).z;
        if ((flags&requiredFlags)!=requiredFlags) { node=links.w; continue; }
        float at; float4 normalResidual; float2 uncertainty; uint steps;
        uint status=TraceDistanceInstance(ray,links.y,enter,leave,at,normalResidual,uncertainty,steps); totalSteps+=steps;
        if (status==TraceSurface && (at<closest || (at==closest && links.y<nearest)))
        { closest=at; nearest=links.y; chosenNormal=normalResidual; chosenUncertainty=uncertainty; }
        if (status==TraceUnresolved && (at<unknown || (at==unknown && links.y<pending))) { unknown=at; pending=links.y; }
        node=links.w;
    }
    result.Metadata.w=totalSteps;
    float distance=ray.DirectionMaximum.w;
    [branch] if (pending!=InvalidAddress && (nearest==InvalidAddress || unknown<=closest))
    { distance=unknown; result.Metadata.xyz=uint3(pending,DistanceWords(Offsets.y+pending*InstanceWords+InstanceMetadata).y,TraceUnresolved); }
    else [branch] if (nearest!=InvalidAddress)
    {
        distance=closest; result.NormalResidual=chosenNormal; result.Uncertainty.xy=chosenUncertainty;
        result.Metadata.xyz=uint3(nearest,DistanceWords(Offsets.y+nearest*InstanceWords+InstanceMetadata).y,TraceSurface);
    }
    result.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*distance,distance);
}

// Global clearance is only an empty-space certificate. Near surfaces, outside
// clipmap coverage and on budget exhaustion, mesh tracing supplies identity.
float GlobalClearance(float3 world)
{
    static const uint HeaderPrefix=8,GridWords=12,GridMetrics=4,GridStorage=8;
    uint4 layout=DistanceWords(Offsets.w);
    [loop] for (uint level=0;level<layout.x;++level)
    {
        uint grid=Offsets.w+HeaderPrefix+level*GridWords; float4 origin=DistanceFloats(grid);
        float3 p=(world-origin.xyz)/origin.w;
        if (!all(isfinite(p)) || any(p<0) || any(p>float(layout.y-1))) continue;
        uint3 lower=min(uint3(floor(p)),layout.y-2); float3 f=p-float3(lower); float sum=0;
        uint start=DistanceWords(grid+GridStorage).x;
        [unroll] for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),at=lower+bit;
            float4 sample=DistanceFloats(start+((at.z*layout.y+at.y)*layout.y+at.x)*VectorWords);
            float weight=(bit.x?f.x:1-f.x)*(bit.y?f.y:1-f.y)*(bit.z?f.z:1-f.z);
            sum+=(Traversal.z!=0?sample.w:sample.y)*weight;
        }
        return isfinite(sum)?max(0,sum-DistanceFloats(grid+GridMetrics).x):0;
    }
    return 0;
}
void TraceLocalOpaqueRay(DistanceRay ray,out DistanceHit result)
{ TraceLocalOpaqueRay(ray,result,Traversal.z); }
void TraceOpaqueRay(DistanceRay ray,out DistanceHit result)
{
    result=(DistanceHit)0; result.Medium=float4(0,0,0,1);
    bool global=Offsets.w!=0; uint stages=global && Parameters.z>0?2:1;
    float at=ray.OriginMinimum.w; uint nearSteps=0,globalSteps=0;
    // One mesh traversal call site avoids duplicating the large local marcher.
    [loop] for (uint stage=0;stage<stages;++stage)
    {
        bool near=stages==2 && stage==0;
        if (global && !near) [loop] for (uint iteration=0;iteration<uint(Parameters.w);++iteration)
        {
            ++globalSteps; float empty=GlobalClearance(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*at);
            float next=at+empty*Parameters.y; if (!isfinite(next) || next<=at) break;
            if (next>ray.DirectionMaximum.w)
            {
                result.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
                result.NormalResidual=0; result.Uncertainty=float4(0,0,0,float(globalSteps)); result.Metadata=uint4(InvalidAddress,InvalidAddress,TraceMiss,nearSteps); return;
            }
            at=next;
        }
        DistanceRay detail=ray; detail.OriginMinimum.w=at;
        if (near) detail.DirectionMaximum.w=min(ray.DirectionMaximum.w,at+Parameters.z);
        TraceLocalOpaqueRay(detail,result); result.Metadata.w+=nearSteps; result.Uncertainty.w=float(globalSteps);
        if (!near || result.Metadata.z!=TraceMiss || detail.DirectionMaximum.w==ray.DirectionMaximum.w) return;
        at=detail.DirectionMaximum.w; nearSteps=result.Metadata.w;
    }
}
// A piecewise constant fiber medium integrates Beer attenuation per actual cell
// interval. Orientation covariance approximates the projected width of crossing
// fibers. The low-frequency source is isotropic sky scattering plus emission.
float4 TraceHairMedium(DistanceRay ray,float maximum,uint flags)
{
    static const uint HairCellWords=24,HairVectorSize=4;
    float4 value=float4(0,0,0,1); uint address=Traversal.w>>LumenHairPointerShift; if (!address) return value;
    uint4 shape=DistanceWords(address); float4 lattice=DistanceFloats(address+HairVectorSize);
    float3 o=ray.OriginMinimum.xyz-lattice.xyz,d=ray.DirectionMaximum.xyz;
    float enter=ray.OriginMinimum.w,leave=min(maximum,ray.DirectionMaximum.w);
    if (leave<=enter || !DistanceSlab(o,d,0,float3(shape.xyz)*lattice.w,enter,leave) || leave<=enter) return value;
    int3 voxel=clamp(int3(floor((o+d*enter)/lattice.w)),int3(0,0,0),int3(shape.xyz)-1);
    int3 advance=int3(d.x>0?1:d.x<0?-1:0,d.y>0?1:d.y<0?-1:0,d.z>0?1:d.z<0?-1:0);
    bool caster=(flags&1u)!=0;
    [loop] for (uint iteration=0;iteration<shape.x+shape.y+shape.z+1 && enter<leave;++iteration)
    {
        float3 planes=leave.xxx;
        [unroll] for (uint axis=0;axis<3;++axis) if (advance[axis]!=0)
            planes[axis]=((float(voxel[axis])+(advance[axis]>0?1:0))*lattice.w-o[axis])/d[axis];
        float next=max(enter,min(leave,min(planes.x,min(planes.y,planes.z))));
        uint cell=shape.w+(uint(voxel.x)+shape.x*(uint(voxel.y)+shape.y*uint(voxel.z)))*HairCellWords;
        float4 material=DistanceFloats(cell),emitted=DistanceFloats(cell+HairVectorSize);
        float4 m=DistanceFloats(cell+(caster?4:2)*HairVectorSize),n=DistanceFloats(cell+(caster?5:3)*HairVectorSize);
        float alignment=dot(m.xyz,d*d)+2*(m.w*d.x*d.y+n.x*d.x*d.z+n.y*d.y*d.z);
        float sigma=(caster?emitted.w:material.x)*sqrt(saturate(1-alignment));
        float transmission=exp(-sigma*(next-enter));
        if (!caster) value.xyz+=value.w*(1-transmission)*(material.yzw*HairSky.xyz+emitted.xyz);
        value.w*=transmission; enter=next; if (enter>=leave) break;
        bool escaped=false;
        [unroll] for (uint nextAxis=0;nextAxis<3;++nextAxis) if (advance[nextAxis]!=0 && planes[nextAxis]<=next)
        { voxel[nextAxis]+=advance[nextAxis]; escaped=escaped || voxel[nextAxis]<0 || voxel[nextAxis]>=int(shape[nextAxis]); }
        if (escaped) break;
    }
    return value;
}
void TraceLocalDistanceRay(DistanceRay ray,out DistanceHit result,uint flags)
{
    TraceLocalOpaqueRay(ray,result,flags);
    result.Medium=TraceHairMedium(ray,result.Metadata.z==TraceMiss?ray.DirectionMaximum.w:result.PositionDistance.w,flags);
}
void TraceLocalDistanceRay(DistanceRay ray,out DistanceHit result) { TraceLocalDistanceRay(ray,result,Traversal.z); }
void TraceDistanceRay(DistanceRay ray,out DistanceHit result)
{
    TraceOpaqueRay(ray,result);
    result.Medium=TraceHairMedium(ray,result.Metadata.z==TraceMiss?ray.DirectionMaximum.w:result.PositionDistance.w,Traversal.z);
}
#if defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
float VolumeMagnitude(float3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
float VolumeInstanceSupport(uint index,float3 world)
{
    uint instance=Offsets.y+index*InstanceWords,field=Offsets.x+DistanceWords(instance+InstanceMetadata).x*FieldWords;
    float3 center=DistanceFloats(instance).xyz,delta=world-center;
    float3 a=DistanceFloats(instance+InstanceRows).xyz,b=DistanceFloats(instance+InstanceRows+VectorWords).xyz,c=DistanceFloats(instance+InstanceRows+VectorWords*2).xyz;
    float3 p=float3(dot(a,delta),dot(b,delta),dot(c,delta)); float scale=max(VolumeMagnitude(a),max(VolumeMagnitude(b),VolumeMagnitude(c)));
    if (!all(isfinite(p)) || !isfinite(scale) || scale<=0) return 0;
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    // ||A||2 <= sqrt(||A||1 ||A||inf). Dividing the mesh-space lower bound
    // by this upper norm keeps the support sphere valid under nonuniform scale.
    float row=max(dot(a,1.0.xxx),max(dot(b,1.0.xxx),dot(c,1.0.xxx))),norm=scale*sqrt(row*VolumeMagnitude(a+b+c));
    if (!isfinite(norm) || norm<=0) return 0;
    float4 grid=DistanceFloats(field+FieldMinimum),errors=DistanceFloats(field+FieldDistances);
    float3 end=grid.xyz+float3(DistanceWords(field+FieldGrid).xyz)*grid.w,sampled=clamp(p,grid.xyz+grid.w*0.5,end-grid.w*0.5);
    float outside=DistanceLength(p-clamp(p,grid.xyz,end)),extension=DistanceLength(p-sampled);
    // This point certificate has one interpolation error. Ray-step skipping
    // additionally budgets errors at a possible later hit; no such hit is used here.
    float lower=max(outside,abs(DistanceValue(field,sampled))-errors.y-Parameters.x*grid.w-extension);
    lower-=Parameters.y*max(VolumeMagnitude(p),max(VolumeMagnitude(grid.xyz),VolumeMagnitude(end)));
    float support=lower/norm-Parameters.y*max(VolumeMagnitude(world),VolumeMagnitude(center));
    return isfinite(support)?clamp(support,0,Parameters.z):0;
}
[numthreads(TraceThreads,1,1)]
void CSVolumeSupport(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*Traversal.y; if (index>=Counts.x) return;
    ProbeOrigin probe=VolumeOrigins[index]; float support=probe.Metadata.y==0?Parameters.z:0;
    [loop] for (uint instance=0;instance<Counts.z && support>0;++instance) support=min(support,VolumeInstanceSupport(instance,probe.PositionMinimum.xyz));
    VolumeSupport[index]=support;
}
#elif defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
// A clear mesh-SDF segment certifies one interpolation link. Neither a hit nor
// an exhausted trace may pass light through an occluder, including non-casters.
float LinkAxis(float center,uint extent,uint count)
{
    float side=LinkViewport.z,coordinate=center/side-0.5;
    if (count>1 && extent%LinkViewport.z!=0)
    {
        float before=(float(count)-1.5)*side,last=(float(count-1)*side+extent)*0.5;
        if (center>before) coordinate=float(count-2)+(center-before)/(last-before);
    }
    return clamp(coordinate,0,float(count-1));
}
float LinkMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
[numthreads(TraceThreads,1,1)]
void CSProbeVisibility(uint3 dispatch:SV_DispatchThreadID)
{
    uint index=dispatch.x+dispatch.y*Traversal.y; if (index>=Counts.x) return;
    uint words=probeLinkWords(LinkGrid.w+1);
    for (uint clearWord=0;clearWord<words;++clearWord) LinkMask[index*words+clearWord]=0; uint2 pixel=uint2(index%LinkViewport.x,index/LinkViewport.x);
    float depth=LinkDepth.Load(int3(pixel,0)); float4 surface=LinkNormals.Load(int3(pixel,0));
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(surface)) || surface.w<0 || dot(surface.xyz,surface.xyz)<=LinkNumeric.x) return;
    float2 center=float2(pixel)+0.5,uv=center/float2(LinkViewport.xy);
    float4 h=mul(LinkInverse,float4(uv*float2(2,-2)+float2(-1,1),depth,1));
    if (!all(isfinite(h)) || abs(h.w)<=LinkOrigin.w) return;
    float3 world=h.xyz/h.w,normal=normalize(surface.xyz);
    if (!all(isfinite(world)) || any(abs(world)>LinkOrigin.z)) return;
    float2 grid=float2(LinkAxis(center.x,LinkViewport.x,LinkViewport.w),LinkAxis(center.y,LinkViewport.y,LinkGrid.x));
    uint2 cell=uint2(floor(grid)); float2 fraction=grid-float2(cell); uint3 mask=0;
    uint slots=LinkGrid.w+1;
    [loop] for (uint tap=0;tap<4*slots;++tap)
    {
        uint cornerIndex=tap/slots;
        uint2 corner=uint2(cornerIndex&1,cornerIndex>>1),tile=min(cell+corner,uint2(LinkViewport.w,LinkGrid.x)-1);
        if ((corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y)<=0) continue;
        uint probeIndex=(tile.y*LinkViewport.w+tile.x)*slots+tap%slots; ProbeOrigin origin=LinkOrigins[probeIndex];
        if (origin.Metadata.x!=probeIndex || origin.Metadata.y!=0 || any(origin.Metadata.zw>=LinkViewport.xy)) continue;
        float4 anchorNormal=LinkNormals.Load(int3(origin.Metadata.zw,0));
        if (!all(isfinite(anchorNormal)) || anchorNormal.w<0 || dot(anchorNormal.xyz,anchorNormal.xyz)<=LinkNumeric.x) continue;
        float3 n=normalize(anchorNormal.xyz),anchor=origin.PositionMinimum.xyz-n*LinkOrigin.x;
        if (dot(normal,n)<=LinkAppearance.y || !all(isfinite(anchor))) continue;
        float bias=LinkVisibility.x+LinkVisibility.y*max(LinkMagnitude(world),LinkMagnitude(anchor));
        float3 start=world+normal*bias,end=anchor+n*bias,delta=end-start; float distance=DistanceLength(delta);
        if (!all(isfinite(start)) || !all(isfinite(end)) || !isfinite(distance)) continue;
        if (distance==0) { mask|=probeLinkBit(tap); continue; }
        DistanceRay ray; ray.OriginMinimum=float4(start,0); ray.DirectionMaximum=float4(DistanceUnit(delta,normal),distance);
        DistanceHit hit; hit.Medium=float4(0,0,0,1); TraceDistanceRay(ray,hit); if (hit.Metadata.z==TraceMiss) mask|=probeLinkBit(tap);
    }
    for (uint storedWord=0;storedWord<words;++storedWord) LinkMask[index*words+storedWord]=mask[storedWord];
}
#elif !defined(EVANS_LUMEN_LIGHTING_STAGE)
[numthreads(TraceThreads,1,1)]
void CSTraceDistance(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*Traversal.y; if (index>=Counts.x) return;
    DistanceRay ray;
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    ProbeOrigin probe=ProbeOrigins[index/ProbeSampling.x]; ray.OriginMinimum=probe.PositionMinimum;

    if (probe.Metadata.y!=0)
    { DistanceHit empty; empty.Medium=float4(0,0,0,1); empty.PositionDistance=0; empty.NormalResidual=0; empty.Uncertainty=0; empty.Metadata=uint4(InvalidAddress,InvalidAddress,TraceUnresolved,0); DistanceHits[index]=empty; return; }
    float3 direction=ProbeRange.y>0?WorldCacheDirection(index%ProbeSampling.x,uint(ProbeRange.y)):
        ProbeSphereDirection(index%ProbeSampling.x,ProbeSampling.x,probe.Metadata.x,ProbeSampling.z);
    bool cached=(ProbeSampling.w&2)!=0 && DistanceHits[index].Metadata.z==LumenWorldCache;
    bool cachedSurface=cached && DistanceHits[index].Metadata.x!=InvalidAddress;
    // A cache value without a surface endpoint cannot delimit fiber transport.
    if ((Traversal.w>>LumenHairPointerShift)!=0 && !cachedSurface) cached=false;
    ray.DirectionMaximum=float4(direction,cached && !cachedSurface?DistanceHits[index].PositionDistance.w:ProbeRange.x);
    if ((ProbeSampling.w&1)!=0 && DistanceHits[index].Metadata.z==LumenScreenSurface)
    { DistanceHits[index].Medium=TraceHairMedium(ray,DistanceHits[index].PositionDistance.w,Traversal.z); return; }
#else
    ray=DistanceRays[index];
#if defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE)
    static const uint GeneratedReflection=1,ReflectionScreenFirst=2;
    if ((Traversal.w&ReflectionScreenFirst)!=0 && DistanceHits[index].Metadata.z==LumenScreenSurface) { DistanceHits[index].Medium=TraceHairMedium(ray,DistanceHits[index].PositionDistance.w,Traversal.z); return; }
    if ((Traversal.w&GeneratedReflection)!=0 && ray.DirectionMaximum.w<=0)
    { DistanceHit empty=(DistanceHit)0; empty.Medium=float4(0,0,0,1); empty.Metadata=uint4(InvalidAddress,InvalidAddress,TraceUnresolved,0); DistanceHits[index]=empty; return; }
#endif
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uint layers=uint(ProbeRange.z); bool terminal=false; DistanceHit previous=(DistanceHit)0; previous.Medium=float4(0,0,0,1);
    static const float MinimumLayerNormalCosine=0.05,LayerNormalAgreement=0.95;
    // SDF layers retain ordered crossings. Unresolvable surface footprints,
    // grazing continuations and exhausted intervals remain explicitly unknown.
    [loop] for (uint layer=0;layer<layers;++layer)
    {
    DistanceHit hit=previous; if (!terminal) TraceDistanceRay(ray,hit);
#else
    DistanceHit hit; hit.Medium=float4(0,0,0,1); TraceDistanceRay(ray,hit);
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    if (cached && !cachedSurface && hit.Metadata.z==TraceMiss) { DistanceHits[index].Medium=hit.Medium; return; }
#endif
#if defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE) || defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    [branch] if (hit.Metadata.z==TraceSurface)
    {
        uint instance=Offsets.y+hit.Metadata.x*InstanceWords,field=Offsets.x+DistanceWords(instance+InstanceMetadata).x*FieldWords;
        float threshold=DistanceFloats(field+FieldDistances).y+Parameters.x*DistanceFloats(field+FieldMinimum).w;
        hit.Uncertainty.z=threshold/DistanceNormalRate(instance,hit.NormalResidual.xyz);
    }
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    float roundoff=ProbeRange.w*max(1,max(abs(ray.OriginMinimum.x),max(abs(ray.OriginMinimum.y),abs(ray.OriginMinimum.z)))+ray.DirectionMaximum.w);
    if (cachedSurface && hit.Metadata.z==TraceSurface)
    {
        DistanceHit expected=DistanceHits[index]; float tolerance=expected.Uncertainty.x+hit.Uncertainty.z+roundoff;
        if (all(hit.Metadata.xy==expected.Metadata.xy) && dot(hit.NormalResidual.xyz,expected.NormalResidual.xyz)>=LayerNormalAgreement &&
            isfinite(tolerance) && abs(dot(hit.PositionDistance.xyz-expected.PositionDistance.xyz,expected.NormalResidual.xyz))<=tolerance) { DistanceHits[index].Medium=hit.Medium; return; }
    }
    if (layer>0 && !terminal && hit.Metadata.z==TraceSurface && hit.Metadata.x==previous.Metadata.x &&
        abs(dot(hit.NormalResidual.xyz,previous.NormalResidual.xyz))>=LayerNormalAgreement &&
        abs(dot(hit.PositionDistance.xyz-previous.PositionDistance.xyz,previous.NormalResidual.xyz))<=hit.Uncertainty.z+previous.Uncertainty.z+roundoff)
    { hit.Metadata.z=TraceUnresolved; hit.NormalResidual=0; hit.Uncertainty.xyz=0; }
    uint destination=((index/ProbeSampling.x)*layers+layer)*ProbeSampling.x+index%ProbeSampling.x;
    DistanceHits[destination]=hit; if (layer+1==layers) break;
    previous=hit;
    if (!terminal && hit.Metadata.z==TraceSurface)
    {
        float cosine=abs(dot(hit.NormalResidual.xyz,ray.DirectionMaximum.xyz));
        float next=hit.PositionDistance.w+2*hit.Uncertainty.z/max(cosine,MinimumLayerNormalCosine)+roundoff;
        if (cosine>=MinimumLayerNormalCosine && isfinite(next) && next>hit.PositionDistance.w && next<ray.DirectionMaximum.w) ray.OriginMinimum.w=next;
        else { previous.Metadata.z=TraceUnresolved; previous.NormalResidual=0; previous.Uncertainty.xyz=0; terminal=true; }
    }
    else terminal=true;
    }
#else
    DistanceHits[index]=hit;
#endif
}
#else
// Cache lighting is camera independent. Depth derivatives supply a geometric
// offset normal; the captured material normal supplies the Lambert cosine.
static const uint LightingCardVectors=7,LightingEmitterVectors=4,LightingOriginVector=4,LightingBackwardVector=5,LightingTileVector=6;
static const uint LightingBorder=1,DirectionalEmitter=0,RectangleEmitter=2,LightingComplete=1,LightingUnknown=2,LightingInvalid=3;
static const uint LightingEmitterNormal=1,LightingEmitterEnergy=2,LightingEmitterFlags=3;
float4 LightingVector(uint address) { return asfloat(LightingData[address]); }
float3 LightingPosition(uint base,uint2 tile,int2 pixel,float depth)
{
    float2 uv=(float2(pixel)-float2(tile)-LightingBorder+0.5)/float(LightingAtlas.z);
    float4 clip=float4(uv.x*2-1,1-uv.y*2,depth,1);
    return float3(dot(LightingVector(base),clip),dot(LightingVector(base+1),clip),dot(LightingVector(base+2),clip));
}
bool LightingDelta(uint base,uint2 tile,int2 pixel,int2 step,float3 center,out float3 delta)
{
    int2 neighbor=pixel+step; delta=0;
    if (any(neighbor<int2(tile)) || any(neighbor>=int2(tile+LightingWork.w))) return false;
    float4 value=LightingNormal.Load(int3(neighbor,0)); if (dot(value.xyz,value.xyz)==0) return false;
    delta=LightingPosition(base,tile,neighbor,value.w)-center; return all(isfinite(delta));
}
float3 LightingGeometricNormal(uint base,uint2 tile,int2 pixel,float3 center,float3 backward)
{
    float3 right,left,down,up;
    bool r=LightingDelta(base,tile,pixel,int2(1,0),center,right),l=LightingDelta(base,tile,pixel,int2(-1,0),center,left);
    bool d=LightingDelta(base,tile,pixel,int2(0,1),center,down),u=LightingDelta(base,tile,pixel,int2(0,-1),center,up);
    if ((!r && !l) || (!d && !u)) return backward;
    float3 dx=r && (!l || dot(right,right)<=dot(left,left))?right:-left,dy=d && (!u || dot(down,down)<=dot(up,up))?down:-up;
    float3 normal=DistanceUnit(cross(dx,dy),backward); return dot(normal,backward)>=0?normal:-normal;
}
#if defined(EVANS_LUMEN_RADIOSITY)
// Jacobi transport reads only the previous completed atlas. Cosine-weighted
// hemisphere sampling cancels Lambert's cosine/pdf; each iteration recomputes
// emission/direct light instead of repeatedly adding them to the old result.
// Wright et al., "Lumen" (SIGGRAPH 2022), surface-cache radiosity.
StructuredBuffer<LightingTexel> PreviousLighting:register(t5);
static const uint RadiosityDirections=6,RadiosityCardVectors=11,RadiosityInverse=4,RadiosityOrigin=8,RadiosityBackward=9,RadiosityTile=10,RadiosityBorder=1;
static const float RadiosityFacing=0.1,RadiosityNormalAgreement=0.25,RadiosityVoxelRadius=2;
static const float RadiosityRelativeRoundoff=16*1.1920928955078125e-7,RadiosityWorldRoundoff=0.5*1.1920928955078125e-7;
static const float RadiosityPi=3.14159265358979323846,RadiosityPhaseScale=1.0/4294967296.0,RadiosityBasisParallel=0.9;
float RadiosityMagnitude(float3 value) { return max(abs(value.x),max(abs(value.y),abs(value.z))); }
float3 RadiosityTransform(uint base,float4 value)
{ return float3(dot(LightingVector(base),value),dot(LightingVector(base+1),value),dot(LightingVector(base+2),value)); }
bool RadiosityCard(uint card,DistanceHit hit,float radius,out float4 value,out float error,out float facing,out uint state)
{
    uint base=LightingRadiosity.z+LightingRadiosity.w+card*RadiosityCardVectors; uint4 tile=LightingData[base+RadiosityTile];
    float4 backward=LightingVector(base+RadiosityBackward); value=0; error=0; state=0;
    facing=dot(hit.NormalResidual.xyz,backward.xyz); if (facing<RadiosityFacing) return false;
    float3 relative=hit.PositionDistance.xyz-LightingVector(base+RadiosityOrigin).xyz;
    float3 projected=RadiosityTransform(base,float4(relative,1)); if (!all(isfinite(projected)) || any(abs(projected.xy)>1)) return false;
    float2 uv=float2(projected.x*0.5+0.5,0.5-projected.y*0.5),location=float2(tile.xy)+RadiosityBorder+uv*LightingAtlas.z-0.5;
    int2 lower=int2(floor(location)); float2 fraction=location-float2(lower); float coverage=0; state=LightingComplete;
    [unroll] for (uint tap=0;tap<4;++tap)
    {
        uint2 corner=uint2(tap&1,tap>>1); int2 pixel=lower+int2(corner);
        float weight=(corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y);
        float4 normal=LightingNormal.Load(int3(pixel,0));
        if (weight<=0 || dot(normal.xyz,normal.xyz)==0 || dot(normal.xyz,hit.NormalResidual.xyz)<RadiosityNormalAgreement) continue;
        float2 texelUv=(float2(pixel)-float2(tile.xy)-RadiosityBorder+0.5)/LightingAtlas.z;
        float3 captured=RadiosityTransform(base+RadiosityInverse,float4(texelUv.x*2-1,1-texelUv.y*2,normal.w,1));
        float planeError=abs(dot(relative-captured,hit.NormalResidual.xyz));
        float roundoff=RadiosityRelativeRoundoff*max(backward.w,max(RadiosityMagnitude(relative),RadiosityMagnitude(captured)))+
            RadiosityWorldRoundoff*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!isfinite(planeError) || planeError>radius+roundoff) continue;
        LightingTexel previous=PreviousLighting[uint(pixel.y)*LightingAtlas.x+uint(pixel.x)];
        state=max(state,previous.Metadata.x==0?LightingInvalid:previous.Metadata.x);
        value.xyz+=previous.RadianceCoverage.xyz*weight; error+=planeError*weight; coverage+=weight;
    }
    if (coverage<=0) { state=0; return false; }
    value=float4(value.xyz/coverage,saturate(coverage)); error/=coverage; return true;
}
uint RadiosityLookup(DistanceHit hit,out float3 incoming)
{
    incoming=0; uint4 range=LightingData[LightingRadiosity.z+hit.Metadata.x]; if (range.y==0) return 0;
    uint instance=Offsets.y+hit.Metadata.x*InstanceWords,field=Offsets.x+DistanceWords(instance+InstanceMetadata).x*FieldWords;
    float threshold=DistanceFloats(field+FieldDistances).y+Parameters.x*DistanceFloats(field+FieldMinimum).w;
    float radius=threshold/DistanceNormalRate(instance,hit.NormalResidual.xyz)*RadiosityVoxelRadius;
    if (!isfinite(radius)) return LightingInvalid;
    float4 values[RadiosityDirections]; float errors[RadiosityDirections],facings[RadiosityDirections]; uint states[RadiosityDirections];
    [unroll] for (uint group=0;group<RadiosityDirections;++group) { values[group]=0; errors[group]=0; facings[group]=0; states[group]=0; }
    [loop] for (uint candidate=0;candidate<range.y;++candidate)
    {
        uint word=LightingRadiosity.z*4+range.x+candidate,card=LightingData[word/4][word%4];
        uint group=LightingData[LightingRadiosity.z+LightingRadiosity.w+card*RadiosityCardVectors+RadiosityTile].z;
        float4 value; float error,facing; uint state;
        if (!RadiosityCard(card,hit,radius,value,error,facing,state)) continue;
        if (states[group]==0 || error<errors[group] || (error==errors[group] && value.w>values[group].w))
        { values[group]=value; errors[group]=error; facings[group]=facing; states[group]=state; }
    }
    float total=0; uint status=0;
    [unroll] for (uint resolvedDirection=0;resolvedDirection<RadiosityDirections;++resolvedDirection) if (states[resolvedDirection]!=0)
    {
        float weight=facings[resolvedDirection]*facings[resolvedDirection]*values[resolvedDirection].w;
        incoming+=values[resolvedDirection].xyz*weight; total+=weight; status=max(status,states[resolvedDirection]);
    }
    if (total>0) incoming/=total;
    if (status==LightingInvalid || !all(isfinite(incoming))) { incoming=0; return LightingInvalid; }
    return total>0?status:0;
}
float3 RadiosityDirection(uint sampleIndex,uint pixel,float3 normal)
{
    float radialSquared=(float(sampleIndex)+0.5)/LightingRadiosity.x;
    uint phase=reversebits(sampleIndex)+reversebits(pixel^LightingRadiosity.y); float angle=2*RadiosityPi*(float(phase)*RadiosityPhaseScale);
    float3 axis=abs(normal.z)<RadiosityBasisParallel?float3(0,0,1):float3(0,1,0),right=normalize(cross(axis,normal)),up=cross(normal,right);
    return right*(sqrt(radialSquared)*cos(angle))+up*(sqrt(radialSquared)*sin(angle))+normal*sqrt(1-radialSquared);
}
#endif
[numthreads(TraceThreads,1,1)]
void CSLightSurface(uint3 invocation:SV_DispatchThreadID)
{
    uint work=invocation.x+invocation.y*Traversal.y,area=LightingTiles.w*LightingTiles.w;
    if (work>=LightingTiles.y*area) return;
    uint block=LightingTiles.x+work/area,local=work%area;
    uint2 pixel=uint2(block%LightingTiles.z,block/LightingTiles.z)*LightingTiles.w+uint2(local%LightingTiles.w,local/LightingTiles.w);
    if (any(pixel>=LightingAtlas.xy)) return;
    uint index=pixel.y*LightingAtlas.x+pixel.x;
    LightingTexel result; result.RadianceCoverage=0; result.Metadata=0;
    uint card=(pixel.y/LightingWork.w)*LightingAtlas.w+pixel.x/LightingWork.w;
    [branch] if (card>=LightingWork.x) { LitSurface[index]=result; return; }
    float4 normalDepth=LightingNormal.Load(int3(pixel,0));
    [branch] if (dot(normalDepth.xyz,normalDepth.xyz)==0) { LitSurface[index]=result; return; }
    uint base=card*LightingCardVectors; uint4 tile=LightingData[base+LightingTileVector];
    float3 relative=LightingPosition(base,tile.xy,int2(pixel),normalDepth.w),position=relative+LightingVector(base+LightingOriginVector).xyz;
    float3 geometric=LightingGeometricNormal(base,tile.xy,int2(pixel),relative,LightingVector(base+LightingBackwardVector).xyz);
    float bias=max(LightingParameters.x,LightingNumeric.x*max(abs(position.x),max(abs(position.y),abs(position.z))));
    [branch] if (tile.z!=InvalidAddress)
    {
        uint instance=Offsets.y+tile.z*InstanceWords,field=Offsets.x+DistanceWords(instance+InstanceMetadata).x*FieldWords;
        float speed=DistanceNormalRate(instance,geometric);
        float threshold=DistanceFloats(field+FieldDistances).y+Parameters.x*DistanceFloats(field+FieldMinimum).w;
        bias=max(bias,threshold/speed*LightingParameters.y);
    }
    float3 origin=position+geometric*bias,irradiance=0; bool invalid=!all(isfinite(origin)) || !all(isfinite(normalDepth));
    [loop] for (uint lightSample=0;lightSample<LightingWork.y && !invalid;++lightSample)
    {
        uint emitter=LightingWork.z+lightSample*LightingEmitterVectors;
        float4 source=LightingVector(emitter),normalRange=LightingVector(emitter+LightingEmitterNormal),energy=LightingVector(emitter+LightingEmitterEnergy);
        if (!any(energy.xyz>0) || energy.w==0) continue;
        float3 direction=source.xyz; float distance=LightingParameters.z,attenuation=1;
        if (uint(source.w)!=DirectionalEmitter)
        {
            float3 offset=source.xyz-position; distance=DistanceLength(offset); if (distance==0) continue;
            direction=offset/distance; float window=1;
            if (normalRange.w>0)
            {
                if (max(abs(offset.x),max(abs(offset.y),abs(offset.z)))>=normalRange.w) continue;
                float3 scaled=offset/normalRange.w; float squared=dot(scaled,scaled); window=saturate(1-squared*squared); window*=window;
            }
            float emitterCosine=uint(source.w)==RectangleEmitter?saturate(dot(normalRange.xyz,-direction)):1;
            attenuation=energy.w*emitterCosine*window/max(distance*distance,LightingParameters.w);
        }
        invalid=!all(isfinite(direction)) || !isfinite(attenuation); if (invalid) continue;
        float cosine=saturate(dot(normalDepth.xyz,direction));
        if (cosine==0 || attenuation==0 || dot(geometric,direction)<=0) continue;
        [branch] if (LightingData[emitter+LightingEmitterFlags].x!=0)
        {
            DistanceRay ray; ray.OriginMinimum=float4(origin,0); ray.DirectionMaximum=float4(direction,LightingParameters.z);
            if (uint(source.w)!=DirectionalEmitter)
            {
                float3 offset=source.xyz-origin; float length=DistanceLength(offset); if (length>0) ray.DirectionMaximum=float4(offset/length,min(LightingParameters.z,max(0,length-LightingParameters.x)));
                else ray.DirectionMaximum.w=0;
            }
            if (ray.DirectionMaximum.w>0)
            {
                ++result.Metadata.y; DistanceHit visibility; visibility.Medium=float4(0,0,0,1); TraceDistanceRay(ray,visibility);
                if (visibility.Metadata.z==TraceSurface) { ++result.Metadata.z; continue; }
                if (visibility.Metadata.z==TraceUnresolved) { ++result.Metadata.w; continue; }
                attenuation*=visibility.Medium.w;
            }
        }
        irradiance+=energy.xyz*(attenuation*cosine);
    }
#if defined(EVANS_LUMEN_RADIOSITY)
    float3 indirect=0,shadingNormal=DistanceUnit(normalDepth.xyz,geometric);
    [loop] for (uint sampleIndex=0;sampleIndex<LightingRadiosity.x && !invalid;++sampleIndex)
    {
        ++result.Metadata.y; float3 direction=RadiosityDirection(sampleIndex,index,shadingNormal);
        if (dot(direction,geometric)<=0) continue;
        DistanceRay ray; ray.OriginMinimum=float4(origin,0); ray.DirectionMaximum=float4(direction,LightingSkyDistance.w);
        DistanceHit hit; hit.Medium=float4(0,0,0,1); TraceLocalDistanceRay(ray,hit,0); // Indirect transport includes non-shadow-casting surfaces.
        if (hit.Metadata.z==TraceMiss) { indirect+=LumenEnvironmentRadiance(LightingSkyDistance.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction)*hit.Medium.w+hit.Medium.xyz; continue; }
        if (hit.Metadata.z!=TraceSurface) { ++result.Metadata.w; continue; }
        float3 incoming; uint state=RadiosityLookup(hit,incoming);
        if (state==LightingInvalid) { invalid=true; continue; }
        if (state!=LightingComplete) ++result.Metadata.w;
        indirect+=incoming*hit.Medium.w+hit.Medium.xyz;
    }
    irradiance+=indirect*(RadiosityPi/LightingRadiosity.x);
#endif
    float4 albedo=LightingAlbedo.Load(int3(pixel,0)); float3 radiance=LightingEmission.Load(int3(pixel,0)).xyz+albedo.xyz*(1-albedo.w)*irradiance*LightingNumeric.y;
    invalid=invalid || !all(isfinite(radiance)); result.RadianceCoverage=float4(invalid?float3(0,0,0):max(radiance,0),1);
    result.Metadata.x=invalid?LightingInvalid:(result.Metadata.w!=0?LightingUnknown:LightingComplete); LitSurface[index]=result;
}
#endif
#elif defined(EVANS_LUMEN_READ_STAGE)
cbuffer ReadFrame:register(b0) { uint4 Extent; };
Texture2D<float4> AlbedoImage:register(t0),NormalImage:register(t1),EmissionImage:register(t2);
RWStructuredBuffer<float4> PackedSurface:register(u0);
static const uint ReadThreads=64,MaterialPlanes=3;
[numthreads(ReadThreads,1,1)]
void CSRead(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*Extent.z; if (index>=Extent.x*Extent.y) return;
    int3 pixel=int3(index%Extent.x,index/Extent.x,0);
    PackedSurface[index*MaterialPlanes]=AlbedoImage.Load(pixel);
    PackedSurface[index*MaterialPlanes+1]=NormalImage.Load(pixel);
    PackedSurface[index*MaterialPlanes+2]=EmissionImage.Load(pixel);
}
#else
cbuffer CaptureFrame:register(b0)
{
    row_major float4x4 RelativeWorld,NormalWorld,ViewProjection;
    float4 BaseColor,EmissionRoughness,Surface;
    uint4 Textures;
};
Texture2D<float4> DiffuseImage:register(t0),NormalImage:register(t1),OpacityImage:register(t2);
SamplerState MaterialSampler:register(s0);
static const float NormalLengthEpsilon=1.0e-10,ProjectionEpsilon=1.0e-7,TangentDeterminantEpsilon=1.0e-8;
static const float SrgbThreshold=0.04045,SrgbLinearScale=12.92,SrgbOffset=0.055,SrgbScale=1.055,SrgbGamma=2.4;
float3 LinearVertexColor(float3 value)
{
    value=saturate(value); float3 low=value/SrgbLinearScale,high=pow((value+SrgbOffset)/SrgbScale,SrgbGamma);
    return float3(value.x<=SrgbThreshold?low.x:high.x,value.y<=SrgbThreshold?low.y:high.y,value.z<=SrgbThreshold?low.z:high.z);
}
float3 UnitNormal(float3 value,float3 fallback)
{ float lengthSquared=dot(value,value); return lengthSquared>NormalLengthEpsilon?value*rsqrt(lengthSquared):fallback; }
struct MaterialVertex { float3 Position:POSITION; float3 Normal:NORMAL; float4 Color:COLOR0; float2 UV:TEXCOORD0; };
struct MaterialFragment { float4 Position:SV_POSITION; float3 World:TEXCOORD0; float3 Normal:TEXCOORD1; float4 Color:COLOR0; float2 UV:TEXCOORD2; };
MaterialFragment VSCapture(MaterialVertex vertex)
{
    MaterialFragment result; float4 world=mul(RelativeWorld,float4(vertex.Position,1));
    // Preserve origin subtraction before projection for thin, distant cards.
    result.Position=mul(ViewProjection,world); result.World=world.xyz/max(abs(world.w),ProjectionEpsilon);
    result.Normal=mul((float3x3)NormalWorld,vertex.Normal); result.Color=float4(LinearVertexColor(vertex.Color.rgb),vertex.Color.a);
    result.UV=float2(vertex.UV.x,1-vertex.UV.y); return result;
}
float3 MaterialNormal(MaterialFragment input,bool front)
{
    float3 normal=UnitNormal(input.Normal,float3(0,0,1)); if (!front) normal=-normal;
    float3 dx=ddx(input.World),dy=ddy(input.World); float2 du=ddx(input.UV),dv=ddy(input.UV);
    float determinant=du.x*dv.y-du.y*dv.x;
    if (Textures.y==0 || abs(determinant)<TangentDeterminantEpsilon) return normal;
    float3 tangent=(dx*dv.y-dy*du.y)/determinant,bitangent=(-dx*dv.x+dy*du.x)/determinant;
    tangent-=normal*dot(normal,tangent); if (dot(tangent,tangent)<=NormalLengthEpsilon) return normal;
    tangent=normalize(tangent); float3 oriented=cross(normal,tangent); oriented*=dot(oriented,bitangent)<0?-1:1;
    float3 mapped=NormalImage.Sample(MaterialSampler,input.UV).xyz*2-1;
    return UnitNormal(tangent*mapped.x+oriented*mapped.y+normal*mapped.z,normal);
}
struct CapturedMaterial { float4 AlbedoMetallic:SV_TARGET0; float4 NormalDepth:SV_TARGET1; float4 EmissionRoughness:SV_TARGET2; };
CapturedMaterial PSCapture(MaterialFragment input,bool front:SV_IsFrontFace)
{
    // Derivatives precede coverage rejection so cutout edges retain valid frames.
    float3 normal=MaterialNormal(input,front);
    float4 diffuse=Textures.x!=0?DiffuseImage.Sample(MaterialSampler,input.UV):float4(1,1,1,1);
    float opacity=Textures.z!=0?OpacityImage.Sample(MaterialSampler,input.UV).r:1;
    float coverage=saturate(input.Color.a*BaseColor.a*diffuse.a*opacity); if (coverage<Surface.y) discard;
    CapturedMaterial output; output.AlbedoMetallic=float4(saturate(input.Color.rgb*BaseColor.rgb*diffuse.rgb),Surface.x);
    output.NormalDepth=float4(normal,saturate(input.Position.z)); output.EmissionRoughness=EmissionRoughness; return output;
}
#endif
