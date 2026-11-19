// Independent D3D12 card capture: linear material data precedes any lighting.
// Readback is a diagnostic export; persistent atlas textures remain GPU-owned.
struct ProbePosition { float4 PositionMinimum; uint4 Metadata; };
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
float3 directionForProbe(uint sampleIndex,uint sampleCount,uint identifier,uint sequence)
{
    static const float SpherePi=3.14159265358979323846,UintFraction=1.0/4294967296.0;
    const uint phase=reversebits(sampleIndex)+reversebits(identifier^sequence);
    float height=1-2*(float(sampleIndex)+0.5)/sampleCount;
    float azimuth=2*SpherePi*(float(phase)*UintFraction),ring=sqrt(max(0,1-height*height));
    return float3(cos(azimuth)*ring,sin(azimuth)*ring,height);
}
// Fixed equal-area atlas directions for persistent world-space probes.
// Clarberg, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD" (2008).
float3 directionFromCacheTexel(uint index,uint resolution)
{
    static const float QuarterCircle=3.14159265358979323846/4;
    resolution=max(resolution,1u);
    float2 coordinate=2*(float2(index%resolution,index/resolution)+0.5)/float(resolution)-1;
    float signedHeight=1-abs(coordinate.x)-abs(coordinate.y),radial=1-abs(signedHeight);
    float azimuth=radial>0?QuarterCircle*(1+(abs(coordinate.y)-abs(coordinate.x))/radial):0;
    float2 signs=float2(coordinate.x<0?-1:1,coordinate.y<0?-1:1);
    float span=radial*sqrt(max(0,2-radial*radial));
    return float3(span*abs(float2(cos(azimuth),sin(azimuth)))*signs,(signedHeight<0?-1:1)*(1-radial*radial));
}
#if defined(EVANS_LUMEN_REFLECTION_RESOLVE_STAGE)
// Reflected-ray reconstruction evaluates the GGX visible-normal sampling weight
// before material-aware spatial reuse. Missing cache links retain zero energy;
// their denominator weight prevents normalization from inventing brightness.
cbuffer SpecularReconstruction:register(b0)
{
    row_major float4x4 InverseCamera; uint4 TraceExtent,TraceWork;
    float4 TraceRules,NumericRules,SurfaceRules,FilterRules,SkyRadiance; uint4 ReconstructionWork; float4 EnvironmentAxis,EnvironmentPeak;
};
Texture2D<float> SurfaceDepth:register(t0);
Texture2D<float4> SurfaceNormal:register(t1),BaseColor:register(t2),AmbientMetalness:register(t3),SpecularColor:register(t4);
struct SpecularPath { float4 Start,Direction; };
struct SpecularIntersection { float4 Position,Normal,Uncertainty; uint4 Kind; float4 Medium; };
struct SpecularIncident { SpecularIntersection Hit; float4 Light; uint4 Cache; };
StructuredBuffer<SpecularPath> Paths:register(t5);
StructuredBuffer<SpecularIncident> Incident:register(t6);
RWStructuredBuffer<float4> Reconstructed:register(u0);
static const uint ReconstructionThreads=64,PhysicallyBasedModel=3,FiniteMiss=0,MeshHit=1,ScreenHit=3,KnownLight=1,PartialLight=2;
static const float PartialConfidence=0.5;
bool reconstructionPosition(uint2 pixel,float z,out float3 p)
{
    float2 ndc=(float2(pixel)+0.5)/float2(TraceExtent.xy)*float2(2,-2)+float2(-1,1);
    float4 projected=mul(InverseCamera,float4(ndc,z,1)); p=0;
    if (!all(isfinite(projected)) || abs(projected.w)<=NumericRules.y) return false;
    p=projected.xyz/projected.w; return all(isfinite(p)) && all(abs(p)<=NumericRules.x);
}
bool reconstructionSurface(uint2 pixel,out float3 p,out float3 n,out float roughness)
{
    float z=SurfaceDepth.Load(int3(pixel,0)); float4 gbuffer=SurfaceNormal.Load(int3(pixel,0)); p=n=0; roughness=gbuffer.w;
    if (!isfinite(z) || z<0 || z>=1 || !all(isfinite(gbuffer)) || roughness<0 || roughness>=TraceRules.z || dot(gbuffer.xyz,gbuffer.xyz)<=NumericRules.z) return false;
    n=normalize(gbuffer.xyz); return reconstructionPosition(pixel,z,p);
}
float smithHeightLambda(float cosine,float alphaSquared)
{
    float denominator=max(cosine*cosine,FilterRules.y*FilterRules.y);
    return 0.5*(sqrt(1+alphaSquared*max(0,1-denominator)/denominator)-1);
}
float3 visibleNormalWeight(float3 eye,float3 outgoing,float3 normal,float roughness,float3 f0)
{
    float alpha=max(roughness*roughness,TraceRules.w);
    float viewLambda=smithHeightLambda(dot(normal,eye),alpha*alpha),lightLambda=smithHeightLambda(dot(normal,outgoing),alpha*alpha);
    float ratio=roughness==0?1:(1+viewLambda)/(1+viewLambda+lightLambda);
    float complement=1-saturate(dot(eye,normalize(eye+outgoing))),square=complement*complement;
    return (f0+(1-f0)*square*square*complement)*ratio;
}
[numthreads(ReconstructionThreads,1,1)]
void CSReflectionResolve(uint3 dispatch:SV_DispatchThreadID)
{
    uint destination=dispatch.x+dispatch.y*ReconstructionWork.y; if (destination>=ReconstructionWork.z) return;
    Reconstructed[destination]=0; uint2 pixel=uint2(destination%TraceExtent.x,destination/TraceExtent.x);
    float3 receiver,n; float roughness; if (!reconstructionSurface(pixel,receiver,n,roughness)) return;
    float4 base=BaseColor.Load(int3(pixel,0)),metal=AmbientMetalness.Load(int3(pixel,0)),specular=SpecularColor.Load(int3(pixel,0));
    if (!all(isfinite(base)) || !all(isfinite(metal)) || !all(isfinite(specular))) return;
    float3 reflectance=ReconstructionWork.x==PhysicallyBasedModel?lerp(FilterRules.w.xxx,saturate(base.xyz),saturate(metal.w)):saturate(specular.xyz);
    int2 center=int2(pixel/TraceExtent.z); int radius=int(FilterRules.x); float3 accumulated=0; float normalization=0; bool reliable=true;
    [loop] for (int row=-radius;row<=radius;++row) [loop] for (int column=-radius;column<=radius;++column)
    {
        int2 cell=center+int2(column,row); if (any(cell<0) || cell.x>=int(TraceExtent.w) || cell.y>=int(TraceWork.x)) continue;
        uint address=uint(cell.y)*TraceExtent.w+uint(cell.x); SpecularPath path=Paths[address];
        if (!all(isfinite(path.Direction)) || dot(path.Direction.xyz,path.Direction.xyz)<=NumericRules.z) continue;
        uint2 anchor=min(uint2(cell)*TraceExtent.z+TraceExtent.z/2,TraceExtent.xy-1); float3 p,normal; float r;
        if (!reconstructionSurface(anchor,p,normal,r) || dot(normal,n)<SurfaceRules.x || abs(r-roughness)>SurfaceRules.w) continue;
        float3 displacement=p-receiver; float separation=max(abs(dot(displacement,n)),abs(dot(displacement,normal)));
        if (!isfinite(separation) || separation>SurfaceRules.y+SurfaceRules.z*length(displacement)) continue;
        float2 distance=(float2(anchor)-float2(pixel))/float(TraceExtent.z); float weight=rcp(1+dot(distance,distance)); normalization+=weight;
        if (path.Direction.w<=0) continue; // Keep zero samples from the VNDF distribution.
        float3 nearPosition,midPosition;
        if (!reconstructionPosition(anchor,0,nearPosition) || !reconstructionPosition(anchor,0.5,midPosition)) { reliable=false; continue; }
        float3 eye=normalize(nearPosition-midPosition);
        if (!all(isfinite(eye)) || dot(eye,normal)<=0 || dot(path.Direction.xyz,normal)<=0) { reliable=false; continue; }
        SpecularIncident incoming=Incident[address]; float3 light=0;
        if (incoming.Hit.Kind.z==FiniteMiss) light=LumenEnvironmentRadiance(SkyRadiance.xyz,EnvironmentAxis,EnvironmentPeak.xyz,path.Direction.xyz);
        else if ((incoming.Hit.Kind.z==MeshHit || incoming.Hit.Kind.z==ScreenHit) && (incoming.Cache.z==KnownLight || incoming.Cache.z==PartialLight))
        { light=incoming.Light.xyz; reliable=reliable && incoming.Cache.z==KnownLight; }
        else reliable=false;
        light=light*incoming.Hit.Medium.w+incoming.Hit.Medium.xyz;
        if (!all(isfinite(light)) || any(light<0)) { reliable=false; continue; }
        accumulated+=light*visibleNormalWeight(eye,path.Direction.xyz,normal,r,reflectance)*weight;
    }
    if (normalization>0 && all(isfinite(accumulated))) Reconstructed[destination]=float4(clamp(accumulated/normalization,0,FilterRules.z),reliable?1:PartialConfidence);
}
#elif defined(EVANS_LUMEN_REFLECTION_RAYS_STAGE)
// Visible GGX normals, Heitz (JCGT 2018). The projected-disk distribution
// accounts for view masking; the subsequent estimator uses Smith G2 / G1.
cbuffer SpecularRayFrame:register(b0)
{ row_major float4x4 SpecularInverse; uint4 SpecularViewport,SpecularWork; float4 SpecularLimits,SpecularNumeric; };
Texture2D<float> SpecularDepth:register(t0);
Texture2D<float4> SpecularNormals:register(t1);
struct SpecularRay { float4 OriginMinimum,DirectionMaximum; };
RWStructuredBuffer<SpecularRay> SpecularRays:register(u0);
static const uint SpecularThreadCount=64,SpecularSequenceLength=64;
static const float SpecularPi=3.14159265358979323846,SpecularIntegerUnit=1.0/4294967296.0;
static const float SpecularGoldenRotation=0.6180339887498948482,SpecularAxisLimit=0.9;
bool specularWorld(uint2 pixel,float depth,out float3 position)
{
    float2 uv=(float2(pixel)+0.5)/float2(SpecularViewport.xy);
    float4 h=mul(SpecularInverse,float4(uv.x*2-1,1-uv.y*2,depth,1)); position=0;
    if (!all(isfinite(h)) || abs(h.w)<=SpecularNumeric.y) return false;
    position=h.xyz/h.w; return all(isfinite(position)) && all(abs(position)<=SpecularNumeric.x);
}
float3 specularGGXNormal(float3 view,float alpha,float2 uniformSample)
{
    float3 stretched=normalize(float3(view.xy*alpha,view.z)); float radial=dot(stretched.xy,stretched.xy);
    float3 tangent=radial>0?float3(-stretched.y,stretched.x,0)/sqrt(radial):float3(1,0,0),bitangent=cross(stretched,tangent);
    float azimuth=2*SpecularPi*uniformSample.y; float2 disk=sqrt(uniformSample.x)*float2(cos(azimuth),sin(azimuth));
    disk.y=lerp(sqrt(max(0,1-disk.x*disk.x)),disk.y,(1+stretched.z)*0.5);
    float3 normal=disk.x*tangent+disk.y*bitangent+sqrt(max(0,1-dot(disk,disk)))*stretched;
    return normalize(float3(alpha*normal.xy,max(0,normal.z)));
}
[numthreads(SpecularThreadCount,1,1)]
void CSReflectionRays(uint3 thread:SV_DispatchThreadID)
{
    uint index=thread.x+thread.y*SpecularWork.w; if (index>=SpecularWork.y) return;
    SpecularRay result=(SpecularRay)0; SpecularRays[index]=result;
    uint2 cell=uint2(index%SpecularViewport.w,index/SpecularViewport.w);
    uint2 pixel=min(cell*SpecularViewport.z+SpecularViewport.z/2,SpecularViewport.xy-1);
    float depth=SpecularDepth.Load(int3(pixel,0)); float4 material=SpecularNormals.Load(int3(pixel,0));
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(material)) || material.w<0 || material.w>=SpecularLimits.z || dot(material.xyz,material.xyz)<=SpecularNumeric.z) return;
    float3 position,nearPosition,midPosition;
    if (!specularWorld(pixel,depth,position) || !specularWorld(pixel,0,nearPosition) || !specularWorld(pixel,0.5,midPosition)) return;
    float3 eye=normalize(nearPosition-midPosition),normal=normalize(material.xyz);
    if (!all(isfinite(eye)) || dot(normal,eye)<=0) return;
    float3 axis=abs(normal.z)<SpecularAxisLimit?float3(0,0,1):float3(0,1,0),right=normalize(cross(axis,normal)),up=cross(normal,right);
    float phase=float(reversebits(index))*SpecularIntegerUnit;
    float2 random=frac(float2((float(SpecularWork.z%SpecularSequenceLength)+0.5)/SpecularSequenceLength,
        float(reversebits(SpecularWork.z))*SpecularIntegerUnit)+float2(phase,frac(phase*SpecularGoldenRotation)));
    float3 micro=material.w==0?float3(0,0,1):specularGGXNormal(float3(dot(eye,right),dot(eye,up),dot(eye,normal)),max(material.w*material.w,SpecularLimits.w),random);
    float3 halfVector=micro.x*right+micro.y*up+micro.z*normal,reflected=reflect(-eye,halfVector);
    result.OriginMinimum=float4(position+normal*SpecularLimits.y,0);
    result.DirectionMaximum=float4(reflected,dot(reflected,normal)>0?SpecularLimits.x:0); SpecularRays[index]=result;
}
#elif defined(EVANS_LUMEN_TEMPORAL_STAGE) || defined(EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE)
// The history stores indirect diffuse energy only. Surface correspondence and
// a bounded color neighborhood reject stale light before scene composition.
cbuffer AccumulationFrame:register(b0)
{
    row_major float4x4 CurrentInverse,PreviousInverse;
    float4 PreviousOpticalPlane; uint4 HistoryExtent;
    float4 Reprojection,Clipping,JitterIntensity,PrecisionRules;
};
Texture2D<float4> CurrentRadiance:register(t0);
StructuredBuffer<uint4> CommittedRadiance:register(t4);
static const uint AccumulationGroupWidth=64,ComponentBits=16,ComponentMask=0xffffu;
static const int AccumulationFilterRadius=1;
static const float SignedNormalRange=32767,NormalSquaredFloor=1e-12;
float4 unpackRadiance(uint4 bits)
{ return float4(f16tof32(bits.x&ComponentMask),f16tof32(bits.x>>ComponentBits),f16tof32(bits.y&ComponentMask),f16tof32(bits.y>>ComponentBits)); }
#if defined(EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE)
struct AccumulationVertex { float4 Position:SV_POSITION; };
AccumulationVertex VSTemporal(uint vertex:SV_VertexID)
{ AccumulationVertex result; float2 corner=float2((vertex<<1)&2,vertex&2); result.Position=float4(corner.x*2-1,1-corner.y*2,0,1); return result; }
float4 PSTemporalComposite(AccumulationVertex input):SV_TARGET
{
    uint2 pixel=uint2(input.Position.xy); float4 scene=CurrentRadiance.Load(int3(pixel,0));
    float4 diffuse=unpackRadiance(CommittedRadiance[pixel.y*HistoryExtent.x+pixel.x]);
    float3 contribution=diffuse.w>0?diffuse.xyz*JitterIntensity.z:0;
    return float4(clamp(scene.xyz+contribution,0,PrecisionRules.w),scene.w);
}
#else
Texture2D<float> ReceiverZ:register(t1);
Texture2D<float4> ReceiverOrientation:register(t2),PhysicalMotion:register(t3);
RWStructuredBuffer<uint4> PendingRadiance:register(u0);
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
StructuredBuffer<float4> SpecularEstimate:register(t5);
#endif
float4 currentEstimate(uint2 pixel)
{
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
    return SpecularEstimate[pixel.y*HistoryExtent.x+pixel.x];
#else
    return CurrentRadiance.Load(int3(pixel,0));
#endif
}
float2 octantSign(float2 v) { return float2(v.x<0?-1:1,v.y<0?-1:1); }
float3 unpackOrientation(uint bits)
{
    float2 coordinate=float2(int(bits<<ComponentBits)>>ComponentBits,int(bits)>>ComponentBits)/SignedNormalRange;
    float3 value=float3(coordinate,1-abs(coordinate.x)-abs(coordinate.y));
    if (value.z<0) value.xy=(1-abs(value.yx))*octantSign(value.xy);
    return normalize(value);
}
uint4 packReceiver(float4 radiance,float3 orientation,float depth)
{
    float2 coordinate=orientation.xy/(abs(orientation.x)+abs(orientation.y)+abs(orientation.z));
    if (orientation.z<0) coordinate=(1-abs(coordinate.yx))*octantSign(coordinate);
    int2 encoded=int2(round(clamp(coordinate,-1,1)*SignedNormalRange));
    return uint4(f32tof16(radiance.x)|(f32tof16(radiance.y)<<ComponentBits),f32tof16(radiance.z)|(f32tof16(radiance.w)<<ComponentBits),
        (uint(encoded.x)&ComponentMask)|(uint(encoded.y)<<ComponentBits),asuint(depth));
}
bool receiverAt(uint2 texel,out float3 world,out float3 orientation,out float depth)
{
    world=0; orientation=float3(0,0,1); depth=ReceiverZ.Load(int3(texel,0)); float4 material=ReceiverOrientation.Load(int3(texel,0));
    if (!isfinite(depth) || depth<0 || depth>=1 || !all(isfinite(material)) || material.w<0 || dot(material.xyz,material.xyz)<=NormalSquaredFloor) return false;
    float2 uv=(float2(texel)+0.5)/float2(HistoryExtent.xy);
    float4 h=mul(CurrentInverse,float4(uv.x*2-1,1-uv.y*2,depth,1));
    if (!all(isfinite(h)) || abs(h.w)<=PrecisionRules.z) return false;
    world=h.xyz/h.w; orientation=normalize(material.xyz); return all(isfinite(world));
}
float priorOpticalDepth(float2 center,float depth)
{
    float2 uv=center/float2(HistoryExtent.xy); float4 h=mul(PreviousInverse,float4(uv.x*2-1,1-uv.y*2,depth,1));
    if (!all(isfinite(h)) || abs(h.w)<=PrecisionRules.z) return -1;
    return dot(PreviousOpticalPlane,float4(h.xyz/h.w,1));
}
[numthreads(AccumulationGroupWidth,1,1)]
void CSTemporal(uint3 invocation:SV_DispatchThreadID)
{
    uint address=invocation.x+invocation.y*HistoryExtent.w; if (address>=HistoryExtent.z) return;
    uint2 texel=uint2(address%HistoryExtent.x,address/HistoryExtent.x); float3 world,orientation; float z;
    float4 estimate=currentEstimate(texel);
    if (!receiverAt(texel,world,orientation,z) || !all(isfinite(estimate)) || estimate.w<=0)
    { PendingRadiance[address]=packReceiver(0,float3(0,0,1),1); return; }
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
    if (estimate.w<1) { PendingRadiance[address]=packReceiver(float4(clamp(estimate.xyz,0,PrecisionRules.w),estimate.w),orientation,z); return; }
#endif
    estimate=float4(clamp(estimate.xyz,0,PrecisionRules.w),1); PendingRadiance[address]=packReceiver(estimate,orientation,z);
    if (JitterIntensity.w==0 || Reprojection.x<=0) return;
    float4 velocity=PhysicalMotion.Load(int3(texel,0)); if (!all(isfinite(velocity)) || velocity.w<=0) return;
    float2 priorCenter=float2(texel)+0.5-velocity.xy-JitterIntensity.xy;
    if (any(priorCenter<0.5) || any(priorCenter>float2(HistoryExtent.xy)-0.5)) return;
    float expected=priorOpticalDepth(priorCenter,1-velocity.z); if (!isfinite(expected) || expected<0) return;
    int2 lower=int2(floor(priorCenter-0.5)); float2 fraction=frac(priorCenter-0.5); float4 accumulated=0; float normalization=0;
    [unroll] for (uint cornerIndex=0;cornerIndex<4;++cornerIndex)
    {
        int2 corner=int2(cornerIndex&1,cornerIndex>>1),candidate=lower+corner;
        uint2 safeAddress=uint2(clamp(candidate,0,int2(HistoryExtent.xy)-1));
        uint4 bits=CommittedRadiance[safeAddress.y*HistoryExtent.x+safeAddress.x]; float4 old=unpackRadiance(bits); float oldZ=asfloat(bits.w);
        float weight=(corner.x?fraction.x:1-fraction.x)*(corner.y?fraction.y:1-fraction.y);
        if (weight<=0 || any(candidate<0) || any(candidate>=int2(HistoryExtent.xy)) || !all(isfinite(old)) || old.w<1 || old.w>Clipping.y ||
            !isfinite(oldZ) || oldZ<0 || oldZ>=1 || dot(orientation,unpackOrientation(bits.z))<Reprojection.z) continue;
        float actual=priorOpticalDepth(float2(candidate)+0.5,oldZ);
        if (!isfinite(actual) || abs(actual-expected)>max(PrecisionRules.x,abs(expected)*Reprojection.y)) continue;
        accumulated+=old*weight; normalization+=weight;
    }
    if (normalization<=NormalSquaredFloor) return;
    float4 history=accumulated/normalization; float3 minimum=estimate.xyz,maximum=minimum;
    [unroll] for (int row=-AccumulationFilterRadius;row<=AccumulationFilterRadius;++row) [unroll] for (int column=-AccumulationFilterRadius;column<=AccumulationFilterRadius;++column)
    {
        uint2 neighbor=uint2(clamp(int2(texel)+int2(column,row),0,int2(HistoryExtent.xy)-1));
        float4 value=currentEstimate(neighbor); float3 otherPosition,otherNormal; float otherZ;
        if (!all(isfinite(value)) || value.w<=0 || !receiverAt(neighbor,otherPosition,otherNormal,otherZ) || dot(orientation,otherNormal)<Reprojection.z) continue;
        float error=abs(dot(otherPosition-world,orientation)); if (!isfinite(error) || error>Reprojection.w) continue;
        minimum=min(minimum,value.xyz); maximum=max(maximum,value.xyz);
    }
    float3 middle=(minimum+maximum)*0.5,radius=max((maximum-minimum)*0.5+abs(middle)*Clipping.x,PrecisionRules.y);
    float3 displacement=history.xyz-middle,scale=abs(displacement/radius);
    history.xyz=middle+displacement/max(1,max(scale.x,max(scale.y,scale.z)));
    float count=min(history.w+1,Clipping.y),weight=min(Reprojection.x,(count-1)/count);
    PendingRadiance[address]=packReceiver(float4(clamp(lerp(estimate.xyz,history.xyz,weight),0,PrecisionRules.w),count),orientation,z);
}
#endif
#elif defined(EVANS_LUMEN_PROBE_GATHER_STAGE)
// Reconstruct cosine-convolved SH at the receiver normal. Normal and symmetric
// plane tests reject unrelated surfaces; incomplete ray energy stays incomplete.
cbuffer ReconstructionFrame:register(b0)
{
    uint4 ReconstructionTraceCount,ReconstructionTraceOffset,ReconstructionTraceWork; float4 ReconstructionTraceOptions,ReconstructionHairSky;
    row_major float4x4 ReconstructionInverse; uint4 ReconstructionViewport,ReconstructionGrid; float4 OriginParameters,PlacementAdaptive;
    float4 ReconstructionSettings,ReconstructionNumeric; uint4 ReceiverModel; float4 VisibilitySettings;
};
struct ReconstructionProbe { float4 Coefficients[9]; float4 Coverage; uint4 TraceCounts,CacheCounts,Metadata; };
Texture2D<float4> InputScene:register(t0);
Texture2D<float> InputDepth:register(t1);
Texture2D<float4> InputNormal:register(t2),InputDiffuse:register(t3),InputAmbient:register(t4);
StructuredBuffer<ProbePosition> ScreenPositions:register(t5);
StructuredBuffer<ReconstructionProbe> ScreenIrradiance:register(t6);
StructuredBuffer<uint> ReceiverLinks:register(t7);
static const uint ReconstructionTerms=9,UsablePartialProbe=2,PBRReceiver=3;
static const float ReconstructionPi=3.14159265358979323846,WorldRoundoff=32*1.1920928955078125e-7;
struct ReconstructionVertex { float4 Position:SV_POSITION; };
ReconstructionVertex VSGather(uint vertexId:SV_VertexID)
{ ReconstructionVertex result; float2 corner=float2((vertexId<<1)&2,vertexId&2); result.Position=float4(corner.x*2-1,1-corner.y*2,0,1); return result; }
float reconstructionHarmonic(uint term,float3 n)
{
    switch (term)
    {
        case 0:return sqrt(1/(4*ReconstructionPi));
        case 1:return n.y*sqrt(3/(4*ReconstructionPi));
        case 2:return n.z*sqrt(3/(4*ReconstructionPi));
        case 3:return n.x*sqrt(3/(4*ReconstructionPi));
        case 4:return n.x*n.y*sqrt(15/(4*ReconstructionPi));
        case 5:return n.y*n.z*sqrt(15/(4*ReconstructionPi));
        case 6:return (3*n.z*n.z-1)*sqrt(5/(16*ReconstructionPi));
        case 7:return n.x*n.z*sqrt(15/(4*ReconstructionPi));
        case 8:return (n.x*n.x-n.y*n.y)*sqrt(15/(16*ReconstructionPi));
    }
    return 0;
}
float reconstructionCoordinate(float pixel,uint extent,uint count)
{
    float side=ReconstructionViewport.z,coordinate=pixel/side-0.5;
    if (count>1 && extent%ReconstructionViewport.z!=0)
    {
        float penultimate=(float(count)-1.5)*side,last=(float(count-1)*side+extent)*0.5;
        if (pixel>penultimate) coordinate=float(count-2)+(pixel-penultimate)/(last-penultimate);
    }
    return clamp(coordinate,0,float(count-1));
}
float reconstructionMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float4 PSGather(ReconstructionVertex vertex):SV_TARGET
{
    int2 pixel=int2(vertex.Position.xy); float4 original=ReceiverModel.z!=0?float4(0,0,0,0):InputScene.Load(int3(pixel,0));
    float z=InputDepth.Load(int3(pixel,0)); float4 normalRoughness=InputNormal.Load(int3(pixel,0));
    if (ReconstructionSettings.x==0 || !isfinite(z) || z<0 || z>=1 || normalRoughness.w<0 || !all(isfinite(normalRoughness)) || dot(normalRoughness.xyz,normalRoughness.xyz)<=ReconstructionNumeric.x) return original;
    float2 uv=vertex.Position.xy/float2(ReconstructionViewport.xy);
    float4 homogeneous=mul(ReconstructionInverse,float4(uv.x*2-1,1-uv.y*2,z,1));
    if (!all(isfinite(homogeneous)) || abs(homogeneous.w)<=OriginParameters.w) return original;
    float3 position=homogeneous.xyz/homogeneous.w,receiverNormal=normalize(normalRoughness.xyz);
    if (!all(isfinite(position)) || any(abs(position)>OriginParameters.z)) return original;
    float2 coordinate=float2(reconstructionCoordinate(vertex.Position.x,ReconstructionViewport.x,ReconstructionViewport.w),reconstructionCoordinate(vertex.Position.y,ReconstructionViewport.y,ReconstructionGrid.x));
    uint2 lower=uint2(floor(coordinate)); float2 blend=coordinate-float2(lower); float3 sum=0; float normalization=0;
    uint slots=ReconstructionGrid.w+1,words=probeLinkWords(ReconstructionGrid.w+1);
    [loop] for (uint candidate=0;candidate<4*slots;++candidate)
    {
        uint cornerIndex=candidate/slots,slot=candidate%slots;
        uint2 corner=uint2(cornerIndex&1,cornerIndex>>1),address=min(lower+corner,uint2(ReconstructionViewport.w,ReconstructionGrid.x)-1);
        float weight=(corner.x?blend.x:1-blend.x)*(corner.y?blend.y:1-blend.y); if (weight<=0) continue;
        if (ReceiverModel.y!=0 && (ReceiverLinks[(uint(pixel.y)*ReconstructionViewport.x+uint(pixel.x))*words+candidate/LumenLinkWordBits]&(1u<<(candidate%LumenLinkWordBits)))==0) continue;
        uint probeIndex=(address.y*ReconstructionViewport.w+address.x)*slots+slot; ProbePosition origin=ScreenPositions[probeIndex]; ReconstructionProbe probe=ScreenIrradiance[probeIndex];
        if (origin.Metadata.x!=probeIndex || origin.Metadata.y!=0 || any(origin.Metadata.zw>=ReconstructionViewport.xy) || probe.Metadata.x!=probeIndex || probe.Metadata.y==0 || probe.Metadata.y>UsablePartialProbe) continue;
        float4 sampledNormal=InputNormal.Load(int3(origin.Metadata.zw,0));
        if (!all(isfinite(sampledNormal)) || sampledNormal.w<0 || dot(sampledNormal.xyz,sampledNormal.xyz)<=ReconstructionNumeric.x) continue;
        float3 probeNormal=normalize(sampledNormal.xyz),anchor=origin.PositionMinimum.xyz-probeNormal*OriginParameters.x;
        float cosine=dot(receiverNormal,probeNormal); if (cosine<=ReconstructionSettings.y) continue;
        float3 separation=position-anchor; float scale=reconstructionMagnitude(separation),distance=scale>0?scale*length(separation/scale):0;
        float planeError=max(abs(dot(separation,receiverNormal)),abs(dot(separation,probeNormal)));
        float allowed=ReconstructionSettings.z+ReconstructionSettings.w*distance+WorldRoundoff*max(reconstructionMagnitude(position),reconstructionMagnitude(anchor));
        if (!isfinite(planeError) || !isfinite(allowed) || planeError>allowed) continue;
        float normalWeight=(cosine-ReconstructionSettings.y)/(1-ReconstructionSettings.y),planeWeight=allowed>0?saturate(1-planeError/allowed):1;
        weight*=normalWeight*normalWeight*planeWeight;
        float3 irradiance=0; [unroll] for (uint term=0;term<ReconstructionTerms;++term) irradiance+=probe.Coefficients[term].xyz*reconstructionHarmonic(term,receiverNormal);
        if (!all(isfinite(irradiance))) continue;
        sum+=max(irradiance,0)*weight; normalization+=weight;
    }
    if (normalization<=ReconstructionNumeric.y || !all(isfinite(sum))) return original;
    float3 albedo=saturate(InputDiffuse.Load(int3(pixel,0)).xyz);
    if (ReceiverModel.x==PBRReceiver) albedo*=(1-ReconstructionNumeric.w)*(1-saturate(InputAmbient.Load(int3(pixel,0)).w));
    float3 incoming=(sum/normalization)*(albedo/ReconstructionPi);
    if (!all(isfinite(incoming))) return original;
    if (ReceiverModel.z!=0) return float4(clamp(incoming,0,ReconstructionNumeric.z),1);
    return float4(clamp(original.xyz+incoming*ReconstructionSettings.x,0,ReconstructionNumeric.z),original.w);
}
#elif defined(EVANS_LUMEN_SCREEN_DEPTH_STAGE)
// Min-reduction in camera-forward units. Empty depth is positive infinity's
// finite sentinel; failed unprojection uses the negative sentinel to descend.
cbuffer ScreenDepthParameters:register(b0)
{ row_major float4x4 PyramidInverse; float4 PyramidAnchor,PyramidForward; uint4 ParentInput,LevelOutput,PyramidWork; float4 PyramidLimits; };
Texture2D<float> OpaqueDepth:register(t0);
RWStructuredBuffer<uint> ScreenDepthWords:register(u0);
static const uint PyramidThreads=64,PyramidHeaderStride=4;
[numthreads(PyramidThreads,1,1)]
void CSBuildScreenDepth(uint3 thread:SV_DispatchThreadID)
{
    uint index=thread.x+thread.y*PyramidWork.x; if (index>=LevelOutput.x*LevelOutput.y) return;
    if (index==0) for (uint lane=0;lane<PyramidHeaderStride;++lane) ScreenDepthWords[LevelOutput.w*PyramidHeaderStride+lane]=LevelOutput[lane];
    uint2 texel=uint2(index%LevelOutput.x,index/LevelOutput.x); float minimum=PyramidLimits.y;
    if (LevelOutput.w!=0)
    {
        for (uint y=0;y<2;++y) for (uint x=0;x<2;++x)
        {
            // Edge duplication preserves min-reduction and bounds every load,
            // including reads the compiler speculates across a branch.
            uint2 child=min(texel*2+uint2(x,y),max(ParentInput.xy,1u)-1);
            minimum=min(minimum,asfloat(ScreenDepthWords[ParentInput.z+child.y*ParentInput.x+child.x]));
        }
    }
    else
    {
        float z=OpaqueDepth.Load(int3(texel,0));
        if (isfinite(z) && z>=0 && z<1)
        {
            float2 uv=(float2(texel)+0.5)/LevelOutput.xy;
            float4 projected=mul(PyramidInverse,float4(uv.x*2-1,1-uv.y*2,z,1)); minimum=PyramidLimits.z;
            if (all(isfinite(projected)) && abs(projected.w)>PyramidLimits.x)
            {
                float3 world=projected.xyz/projected.w;
                float allowance=PyramidAnchor.w*dot(abs(world)+abs(PyramidAnchor.xyz),abs(PyramidForward.xyz));
                float depth=dot(world-PyramidAnchor.xyz,PyramidForward.xyz)-allowance;
                if (isfinite(depth)) minimum=depth;
            }
        }
    }
    ScreenDepthWords[LevelOutput.z+index]=asuint(minimum);
}
#elif defined(EVANS_LUMEN_SCREEN_TRACE_STAGE)
// Screen-first Lumen transport. Projected intervals interpolate position/W and
// reciprocal W, following McGuire and Mara's perspective-correct DDA (2014).
// The bounded single-layer search leaves uncertain rays to mesh-SDF traversal.
cbuffer ScreenRayParameters:register(b0)
{ row_major float4x4 RayCamera,RayCameraInverse; uint4 RayWork,RayViewport; float4 CameraAnchor,CameraForward,RayLimits; };
Texture2D<float> RayDepth:register(t0);
Texture2D<float4> RayNormals:register(t1),RayColor:register(t2);
StructuredBuffer<ProbePosition> RaySeeds:register(t3);
StructuredBuffer<uint> RayPyramid:register(t4);
struct ScreenIntersection { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct ScreenLighting { ScreenIntersection Hit; float4 RadianceCoverage; uint4 Cache; };
RWStructuredBuffer<ScreenIntersection> ScreenIntersections:register(u0);
RWStructuredBuffer<ScreenLighting> ScreenLightingOutput:register(u1);
static const uint ScreenRayThreads=64,ScreenRaySurface=3,ScreenRayKnown=1,ScreenRayNoCard=0xffffffffu;
static const float ScreenRayNormalFloor=1e-12,ScreenRayParallelFloor=1e-7,ScreenRayClear=1,ScreenRayCenter=0.5;
bool intersectClipHalfspace(float begin,float finish,inout float enter,inout float leave)
{
    if (max(begin,finish)<0) return false;
    if (begin<0) enter=max(enter,begin/(begin-finish));
    if (finish<0) leave=min(leave,begin/(begin-finish));
    return enter<=leave;
}
bool findScreenIntersection(ProbePosition seed,float3 direction,out ScreenIntersection found,out float3 light)
{
    found=(ScreenIntersection)0; found.Medium=float4(0,0,0,1); light=0;
    float3 start=seed.PositionMinimum.xyz+direction*seed.PositionMinimum.w;
    float3 finish=seed.PositionMinimum.xyz+direction*RayLimits.x;
    float4 clipStart=mul(RayCamera,float4(start,1)),clipEnd=mul(RayCamera,float4(finish,1));
    if (!all(isfinite(clipStart)) || !all(isfinite(clipEnd))) return false;
    float enter=0,leave=1;
    if (!intersectClipHalfspace(clipStart.w+clipStart.x,clipEnd.w+clipEnd.x,enter,leave) ||
        !intersectClipHalfspace(clipStart.w-clipStart.x,clipEnd.w-clipEnd.x,enter,leave) ||
        !intersectClipHalfspace(clipStart.w+clipStart.y,clipEnd.w+clipEnd.y,enter,leave) ||
        !intersectClipHalfspace(clipStart.w-clipStart.y,clipEnd.w-clipEnd.y,enter,leave) ||
        !intersectClipHalfspace(clipStart.z,clipEnd.z,enter,leave) ||
        !intersectClipHalfspace(clipStart.w-clipStart.z,clipEnd.w-clipEnd.z,enter,leave) ||
        !intersectClipHalfspace(clipStart.w-RayLimits.w,clipEnd.w-RayLimits.w,enter,leave)) return false;
    float4 first=lerp(clipStart,clipEnd,enter),last=lerp(clipStart,clipEnd,leave);
    float reciprocalFirst=1/first.w,reciprocalLast=1/last.w;
    float3 weightedFirst=lerp(start,finish,enter)*reciprocalFirst,weightedLast=lerp(start,finish,leave)*reciprocalLast;
    float2 pixelFirst=(first.xy*reciprocalFirst*float2(0.5,-0.5)+0.5)*RayViewport.xy;
    float2 pixelLast=(last.xy*reciprocalLast*float2(0.5,-0.5)+0.5)*RayViewport.xy;
    float2 extent=abs(pixelLast-pixelFirst); float pixels=max(extent.x,extent.y); if (pixels<1) return false;
    float stride=RayLimits.z/pixels,depthRate=dot(direction,CameraForward.xyz);
    uint cursor=0,mip=0;
    [loop] for (uint visit=0;visit<RayViewport.z;++visit)
    {
        uint iteration=cursor++;
        float begin=float(iteration)*stride; if (begin>=1) break;
        float end=min(begin+stride,1); int2 pixel=int2(floor(lerp(pixelFirst,pixelLast,(begin+end)*0.5)));
        if (any(pixel<0) || any(pixel>=int2(RayViewport.xy))) return false;
        if (all(uint2(pixel)==seed.Metadata.zw)) continue;
        if (RayViewport.w!=0)
        {
            uint2 tile=uint2(pixel)>>mip; float2 motion=pixelLast-pixelFirst;
            float2 minimumPixel=float2(tile<<mip),maximumPixel=min(minimumPixel+float(1u<<mip),float2(RayViewport.xy));
            float boundary=1;
            if (motion.x!=0) boundary=min(boundary,((motion.x>0?maximumPixel.x:minimumPixel.x)-pixelFirst.x)/motion.x);
            if (motion.y!=0) boundary=min(boundary,((motion.y>0?maximumPixel.y:minimumPixel.y)-pixelFirst.y)/motion.y);
            uint limit=uint(clamp(ceil(boundary/stride-ScreenRayCenter),float(iteration+1),ceil(1/stride)));
            float tailBegin=float(limit-1)*stride,tailEnd=min(tailBegin+stride,1);
            int2 tail=int2(floor(lerp(pixelFirst,pixelLast,(tailBegin+tailEnd)*0.5)));
            if (any(tail<0) || any(tail>=int2(RayViewport.xy)) || any((uint2(tail)>>mip)!=tile)) { limit=iteration+1; tailEnd=end; }
            {
                static const uint HeaderStride=4; uint header=mip*HeaderStride;
                float minimum=asfloat(RayPyramid[RayPyramid[header+2]+tile.y*RayPyramid[header]+tile.x]);
                float3 from=lerp(weightedFirst,weightedLast,begin)/lerp(reciprocalFirst,reciprocalLast,begin);
                float3 to=lerp(weightedFirst,weightedLast,tailEnd)/lerp(reciprocalFirst,reciprocalLast,tailEnd),origin=seed.PositionMinimum.xyz;
                float bound=max(dot(from-origin,direction)*depthRate,dot(to-origin,direction)*depthRate)+dot(origin-CameraAnchor.xyz,CameraForward.xyz);
                float error=CameraAnchor.w*dot(abs(from)+abs(to)+2*(abs(origin)+abs(CameraAnchor.xyz)),float3(1,1,1));
                if (isfinite(bound+error) && bound+error<minimum) { cursor=limit; mip=min(mip+1,RayViewport.w-1); continue; }
            }
            if (mip>0) { --mip; cursor=iteration; continue; }
        }
        float z=RayDepth.Load(int3(pixel,0)); if (!isfinite(z) || z<0 || z>=ScreenRayClear) continue;
        float2 uv=(float2(pixel)+ScreenRayCenter)/RayViewport.xy;
        float4 position=mul(RayCameraInverse,float4(uv.x*2-1,1-uv.y*2,z,1));
        if (!all(isfinite(position)) || abs(position.w)<=RayLimits.w) return false;
        float3 surface=position.xyz/position.w;
        float3 startPoint=lerp(weightedFirst,weightedLast,begin)/lerp(reciprocalFirst,reciprocalLast,begin);
        float3 endPoint=lerp(weightedFirst,weightedLast,end)/lerp(reciprocalFirst,reciprocalLast,end);
        float from=dot(startPoint-seed.PositionMinimum.xyz,direction),to=dot(endPoint-seed.PositionMinimum.xyz,direction);
        float sceneDepth=dot(surface-seed.PositionMinimum.xyz,CameraForward.xyz);
        float nearDepth=min(from*depthRate,to*depthRate),farDepth=max(from*depthRate,to*depthRate);
        if (!all(isfinite(float4(from,to,sceneDepth,farDepth)))) return false;
        if (farDepth<sceneDepth) continue;
        if (nearDepth>sceneDepth+RayLimits.y) return false;
        float3 n=RayNormals.Load(int3(pixel,0)).xyz;
        if (!all(isfinite(n)) || dot(n,n)<=ScreenRayNormalFloor || dot(n,direction)>=0) return false;
        float travel=abs(depthRate)>ScreenRayParallelFloor?clamp(sceneDepth/depthRate,min(from,to),max(from,to)):(from+to)*0.5;
        if (travel<=seed.PositionMinimum.w || travel>RayLimits.x) return false;
        light=RayColor.Load(int3(pixel,0)).xyz; if (!all(isfinite(light))) return false;
        light=max(light,0); found.PositionDistance=float4(surface,travel); found.NormalResidual=float4(normalize(n),0);
        found.Metadata=uint4(uint2(pixel),ScreenRaySurface,visit+1); return true;
    }
    return false;
}
[numthreads(ScreenRayThreads,1,1)]
void CSTraceScreen(uint3 thread:SV_DispatchThreadID)
{
    uint ray=thread.x+thread.y*RayWork.w; if (ray>=RayWork.x) return;
    ProbePosition seed=RaySeeds[ray/RayWork.y]; ScreenLighting answer=(ScreenLighting)0; answer.Hit.Medium=float4(0,0,0,1);
    answer.Hit.Metadata=uint4(ScreenRayNoCard,ScreenRayNoCard,0,0); answer.Cache.x=ScreenRayNoCard;
    ScreenIntersection hit; hit.Medium=float4(0,0,0,1); float3 incoming;
    float3 direction=directionForProbe(ray%RayWork.y,RayWork.y,seed.Metadata.x,RayWork.z);
    if (CameraForward.w!=0)
    { float4 reflected=asfloat(seed.Metadata); direction=reflected.xyz; seed.Metadata.y=reflected.w>0?0:1; }
    if (seed.Metadata.y==0 && findScreenIntersection(seed,direction,hit,incoming))
    { answer.Hit=hit; answer.RadianceCoverage=float4(incoming,1); answer.Cache.z=ScreenRayKnown; }
    ScreenIntersections[ray]=answer.Hit; ScreenLightingOutput[ray]=answer;
}
#elif defined(EVANS_LUMEN_SCREEN_PROBES_STAGE)
// One uniform seed plus a bounded quadtree. A refinement exists
// only where every ancestor fails normal/plane interpolation. Stable reserved slots
// bound work without a readback counter; inactive slots never receive sky energy.
cbuffer PlacementFrame:register(b0)
{ row_major float4x4 PlacementInverse; uint4 PlacementViewport,PlacementGrid; float4 PlacementParameters,AdaptiveRules; };
Texture2D<float> PlacementDepth:register(t0);
Texture2D<float4> PlacementNormal:register(t1);
RWStructuredBuffer<ProbePosition> PlacementOrigins:register(u0);
static const uint PlacementThreads=64,PlacementBackground=1,PlacementInvalid=2,PlacementAbsent=0xffffffffu;
static const uint PlacementDepthLimit=2,PlacementBranching=4;
static const float PlacementFarDepth=1,PlacementNormalEpsilon=1e-12;
float placementMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
bool placementSurface(uint2 pixel,out float3 position,out float3 normal)
{
    position=normal=0; float2 uv=(float2(pixel)+0.5)/PlacementViewport.xy;
    float4 h=mul(PlacementInverse,float4(uv*float2(2,-2)+float2(-1,1),PlacementDepth.Load(int3(pixel,0)),1));
    if (!all(isfinite(h)) || abs(h.w)<=PlacementParameters.w) return false;
    normal=normalize(PlacementNormal.Load(int3(pixel,0)).xyz); position=h.xyz/h.w;
    return all(isfinite(position)) && all(abs(position)+PlacementParameters.x<=PlacementParameters.z);
}
uint2 placementPixel(uint2 lower,uint2 upper,uint ancestors,float3 parents[PlacementDepthLimit],float3 parentNormals[PlacementDepthLimit],out bool found)
{
    uint best=PlacementAbsent; uint2 selected=0;
    [loop] for (uint y=lower.y;y<upper.y;++y) [loop] for (uint x=lower.x;x<upper.x;++x)
    {
        uint2 pixel=uint2(x,y); float depth=PlacementDepth.Load(int3(pixel,0)); float4 material=PlacementNormal.Load(int3(pixel,0));
        if (!isfinite(depth) || depth<0 || depth>=PlacementFarDepth || !all(isfinite(material)) || material.w<0 || dot(material.xyz,material.xyz)<=PlacementNormalEpsilon) continue;
        if (ancestors!=0)
        {
            float3 worldPosition,normal; if (!placementSurface(pixel,worldPosition,normal)) continue;
            bool covered=false;
            [unroll] for (uint ancestor=0;ancestor<PlacementDepthLimit;++ancestor) if (ancestor<ancestors)
            {
                float3 delta=worldPosition-parents[ancestor]; float scale=placementMagnitude(delta),distance=scale>0?scale*length(delta/scale):0;
                float error=max(abs(dot(delta,normal)),abs(dot(delta,parentNormals[ancestor])));
                float tolerance=AdaptiveRules.y+AdaptiveRules.z*distance+AdaptiveRules.w*max(placementMagnitude(worldPosition),placementMagnitude(parents[ancestor]));
                covered=covered || (dot(normal,parentNormals[ancestor])>AdaptiveRules.x && error<=tolerance);
            }
            if (covered) continue;
        }
        int2 offset=int2(pixel*2+1)-int2(lower+upper); uint metric=uint(dot(offset,offset));
        if (metric<best) { best=metric; selected=pixel; }
    }
    found=best!=PlacementAbsent; return selected;
}
[numthreads(PlacementThreads,1,1)]
void CSPlaceScreenProbes(uint3 threadId:SV_DispatchThreadID)
{
    uint probe=threadId.x; if (probe>=PlacementGrid.y) return;
    uint slots=PlacementGrid.w+1,tile=probe/slots,slot=probe%slots;
    uint2 lower=uint2(tile%PlacementViewport.w,tile/PlacementViewport.w)*PlacementViewport.z;
    uint2 upper=min(lower+PlacementViewport.z,PlacementViewport.xy); bool found;
    float3 parents[PlacementDepthLimit],parentNormals[PlacementDepthLimit];
    [unroll] for (uint clear=0;clear<PlacementDepthLimit;++clear) { parents[clear]=0; parentNormals[clear]=0; }
    uint quadrants[PlacementDepthLimit],levels=0;
    for (uint node=slot;node!=0;node=(node-1)/PlacementBranching) quadrants[levels++]=(node-1)%PlacementBranching;
    uint2 pixel=placementPixel(lower,upper,0,parents,parentNormals,found);
    ProbePosition origin; origin.PositionMinimum=0; origin.Metadata=uint4(probe,PlacementBackground,0,0);
    float3 worldPosition=0,normal=0;
    for (uint level=0;level<levels;++level)
    {
        if (!found || !placementSurface(pixel,worldPosition,normal)) { PlacementOrigins[probe]=origin; return; }
        parents[level]=worldPosition; parentNormals[level]=normal;
        uint quadrant=quadrants[levels-1-level]; uint2 middle=(lower+upper)/2;
        lower=uint2((quadrant&1)?middle.x:lower.x,(quadrant&2)?middle.y:lower.y);
        upper=uint2((quadrant&1)?upper.x:middle.x,(quadrant&2)?upper.y:middle.y);
        pixel=placementPixel(lower,upper,level+1,parents,parentNormals,found);
    }
    if (found)
    {
        origin.Metadata=uint4(probe,PlacementInvalid,pixel);
        if (placementSurface(pixel,worldPosition,normal))
        { origin.PositionMinimum=float4(worldPosition+normal*PlacementParameters.x,PlacementParameters.y); origin.Metadata.y=0; }
    }
    PlacementOrigins[probe]=origin;
}
#elif defined(EVANS_LUMEN_PROBE_INTEGRATE_STAGE)
cbuffer IrradianceFrame:register(b0) { uint4 ProbeDispatch; float4 SkyWeight,EnvironmentAxis,EnvironmentPeak; };
struct ProbeIntersection { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct ProbeIncident { ProbeIntersection Hit; float4 RadianceCoverage; uint4 Cache; };
struct IrradianceProbe { float4 Coefficients[9]; float4 Coverage; uint4 TraceCounts,CacheCounts,Metadata; };
StructuredBuffer<ProbeIncident> Incident:register(t0);
StructuredBuffer<ProbePosition> Positions:register(t1);
RWStructuredBuffer<IrradianceProbe> Irradiance:register(u0);
static const uint ShCoefficientCount=9,ProbeGroupWidth=64,CompleteProbe=1,PartialProbe=2,InvalidProbe=3,SurfaceIntersection=1,MissIntersection=0;
float irradianceHarmonic(uint index,float3 w)
{
    // Real SH normalization and Lambertian band factors pi, 2*pi/3, pi/4.
    static const float Pi=3.14159265358979323846;
    float band0=Pi*sqrt(1/(4*Pi)),band1=(2*Pi/3)*sqrt(3/(4*Pi));
    float crossTerm=(Pi/4)*sqrt(15/(4*Pi)),zSquare=(Pi/4)*sqrt(5/(16*Pi)),xyDifference=(Pi/4)*sqrt(15/(16*Pi));
    switch (index)
    {
        case 0: return band0;
        case 1: return w.y*band1;
        case 2: return w.z*band1;
        case 3: return w.x*band1;
        case 4: return w.x*w.y*crossTerm;
        case 5: return w.y*w.z*crossTerm;
        case 6: return (3*w.z*w.z-1)*zSquare;
        case 7: return w.x*w.z*crossTerm;
        case 8: return (w.x*w.x-w.y*w.y)*xyDifference;
    }
    return 0;
}
[numthreads(ProbeGroupWidth,1,1)]
void CSIntegrateProbes(uint3 dispatch:SV_DispatchThreadID)
{
    uint probe=dispatch.x+dispatch.y*ProbeDispatch.w; if (probe>=ProbeDispatch.x) return;
    IrradianceProbe value; [unroll] for (uint initialize=0;initialize<ShCoefficientCount;++initialize) value.Coefficients[initialize]=0;
    value.Coverage=0; value.TraceCounts=0; value.CacheCounts=0;
    uint identifier=Positions[probe].Metadata.x; bool bad=Positions[probe].Metadata.y!=0; float coverage=0,scaledDistance=0;
    [loop] for (uint ray=0;ray<ProbeDispatch.y;++ray)
    {
        ProbeIncident incoming=Incident[probe*ProbeDispatch.y+ray]; float3 light=0;
        float3 direction=directionForProbe(ray,ProbeDispatch.y,identifier,ProbeDispatch.z);
        if (incoming.Hit.Metadata.z==SurfaceIntersection || incoming.Hit.Metadata.z==LumenScreenSurface || incoming.Hit.Metadata.z==LumenWorldCache)
        {
            if (incoming.Hit.Metadata.z==LumenWorldCache) ++value.TraceCounts.w;
            else { ++value.TraceCounts.x; scaledDistance+=incoming.Hit.PositionDistance.w/float(ProbeDispatch.y); }
            if (incoming.Cache.z==CompleteProbe) { ++value.CacheCounts.x; light=incoming.RadianceCoverage.xyz; coverage+=incoming.RadianceCoverage.w; }
            else if (incoming.Cache.z==PartialProbe) { ++value.CacheCounts.y; light=incoming.RadianceCoverage.xyz; coverage+=incoming.RadianceCoverage.w; }
            else if (incoming.Cache.z==InvalidProbe) { ++value.CacheCounts.w; bad=true; }
            else ++value.CacheCounts.z;
        }
        else if (incoming.Hit.Metadata.z==MissIntersection) { ++value.TraceCounts.y; light=LumenEnvironmentRadiance(SkyWeight.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction); }
        else ++value.TraceCounts.z;
        light=light*incoming.Hit.Medium.w+incoming.Hit.Medium.xyz;
        [unroll] for (uint harmonic=0;harmonic<ShCoefficientCount;++harmonic)
            value.Coefficients[harmonic]+=float4(light*(SkyWeight.w*irradianceHarmonic(harmonic,direction)),0);
    }
    value.Coverage=float4(float(value.TraceCounts.y+value.CacheCounts.x)/ProbeDispatch.y,(value.TraceCounts.x+value.TraceCounts.w)?coverage/(value.TraceCounts.x+value.TraceCounts.w):0,
        float(value.TraceCounts.y)/ProbeDispatch.y,value.TraceCounts.x?scaledDistance*(float(ProbeDispatch.y)/value.TraceCounts.x):0);
    [unroll] for (uint check=0;check<ShCoefficientCount;++check) bad=bad || !all(isfinite(value.Coefficients[check]));
    bad=bad || !all(isfinite(value.Coverage));
    // Coverage remains a probability after floating-point averaging.
    value.Coverage.y=saturate(value.Coverage.y); uint status=(value.TraceCounts.z || value.CacheCounts.y || value.CacheCounts.z)?PartialProbe:CompleteProbe;
    if (bad) { status=InvalidProbe; value.Coverage=0; [unroll] for (uint clear=0;clear<ShCoefficientCount;++clear) value.Coefficients[clear]=0; }
    value.Metadata=uint4(identifier,status,ProbeDispatch.y,ProbeDispatch.z); Irradiance[probe]=value;
}
#elif defined(EVANS_LUMEN_RADIANCE_RESOLVE_STAGE)
// Four source bindings retain depth, native hits, resident lighting and card
// membership. No CPU hit array is needed between the two compute dispatches.
cbuffer CacheResolve:register(b0) { uint4 QuerySize,ImageSize,CardMap; float4 Acceptance,Roundoff; };
struct CacheHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
struct CacheLight { float4 RadianceCoverage; uint4 Metadata; };
struct CacheAnswer { CacheHit Hit; float4 RadianceCoverage; uint4 Cache; };
Texture2D<float4> CacheNormal:register(t0);
StructuredBuffer<CacheHit> CacheHits:register(t1);
StructuredBuffer<CacheLight> CacheLightValues:register(t2);
StructuredBuffer<uint4> CacheWords:register(t3);
RWStructuredBuffer<CacheAnswer> CacheOutput:register(u0);
static const uint CacheThreads=64,CacheDirections=6,CardStride=11,CardInverse=4,CardOrigin=8,CardBackward=9,CardTile=10,Gutter=1;
static const uint InvalidCard=0xffffffffu,SurfaceHit=1,ResolvedLight=1,PartialLight=2,InvalidLight=3;
float4 cacheFloat(uint offset) { return asfloat(CacheWords[offset]); }
float maximumCoordinate(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float3 cacheProject(uint offset,float4 v)
{ return float3(dot(cacheFloat(offset),v),dot(cacheFloat(offset+1),v),dot(cacheFloat(offset+2),v)); }
bool gatherCard(uint card,CacheHit hit,float radius,out float4 light,out float distanceError,out float facing,out uint status)
{
    light=0; distanceError=0; status=0; uint first=CardMap.x+card*CardStride; uint4 tile=CacheWords[first+CardTile];
    float4 backward=cacheFloat(first+CardBackward); facing=dot(backward.xyz,hit.NormalResidual.xyz); if (facing<Acceptance.x) return false;
    float3 local=hit.PositionDistance.xyz-cacheFloat(first+CardOrigin).xyz,clip=cacheProject(first,float4(local,1));
    if (!all(isfinite(clip)) || any(abs(clip.xy)>1)) return false;
    float2 texel=float2(tile.xy)+Gutter+float2(clip.x*0.5+0.5,0.5-clip.y*0.5)*QuerySize.y-0.5;
    int2 base=int2(floor(texel)); float2 fraction=texel-float2(base); float coverage=0; status=ResolvedLight;
    [unroll] for (uint corner=0;corner<4;++corner)
    {
        uint2 offset=uint2(corner&1,corner>>1); int2 pixel=base+int2(offset);
        float weight=(offset.x!=0?fraction.x:1-fraction.x)*(offset.y!=0?fraction.y:1-fraction.y);
        float4 normalDepth=CacheNormal.Load(int3(pixel,0));
        if (weight<=0 || dot(normalDepth.xyz,normalDepth.xyz)==0 || dot(normalDepth.xyz,hit.NormalResidual.xyz)<Acceptance.y) continue;
        float2 uv=(float2(pixel)-float2(tile.xy)-Gutter+0.5)/QuerySize.y;
        float3 captured=cacheProject(first+CardInverse,float4(uv.x*2-1,1-uv.y*2,normalDepth.w,1));
        float error=abs(dot(local-captured,hit.NormalResidual.xyz));
        float allowance=Roundoff.x*max(backward.w,max(maximumCoordinate(local),maximumCoordinate(captured)))+Roundoff.y*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!isfinite(error) || error>radius+allowance) continue;
        CacheLight value=CacheLightValues[uint(pixel.y)*ImageSize.x+uint(pixel.x)];
        light.xyz+=value.RadianceCoverage.xyz*weight; status=max(status,value.Metadata.x!=0?value.Metadata.x:InvalidLight); distanceError+=error*weight; coverage+=weight;
    }
    if (coverage<=0) { status=0; return false; }
    light=float4(light.xyz/coverage,saturate(coverage)); distanceError/=coverage; return true;
}
[numthreads(CacheThreads,1,1)]
void CSResolveRadiance(uint3 thread:SV_DispatchThreadID)
{
    uint query=thread.x+thread.y*ImageSize.w; if (query>=QuerySize.x) return;
    CacheAnswer answer; answer.Hit.Medium=float4(0,0,0,1); answer.Hit=CacheHits[query]; answer.RadianceCoverage=0; answer.Cache=uint4(InvalidCard,0,0,0);
    if (answer.Hit.Metadata.z==LumenScreenSurface || answer.Hit.Metadata.z==LumenWorldCache) { CacheOutput[query].Hit.Medium=answer.Hit.Medium; return; }
    float radius=Acceptance.z+Acceptance.w*answer.Hit.Uncertainty.z;
    if (CardMap.z==0) answer.Hit.Uncertainty.z=0;
    [branch] if (answer.Hit.Metadata.z!=SurfaceHit) { CacheOutput[query]=answer; return; }
    [branch] if (!isfinite(radius)) { answer.Cache.z=InvalidLight; CacheOutput[query]=answer; return; }
    uint4 range=CacheWords[answer.Hit.Metadata.x]; [branch] if (range.y==0) { CacheOutput[query]=answer; return; }
    float4 directions[CacheDirections]; float errors[CacheDirections],cosines[CacheDirections]; uint validity[CacheDirections],selected[CacheDirections];
    [unroll] for (uint initialize=0;initialize<CacheDirections;++initialize)
    { directions[initialize]=0; errors[initialize]=0; cosines[initialize]=0; validity[initialize]=0; selected[initialize]=InvalidCard; }
    [loop] for (uint candidate=0;candidate<range.y;++candidate)
    {
        uint word=range.x+candidate,card=CacheWords[word/4][word%4],group=CacheWords[CardMap.x+card*CardStride+CardTile].z;
        float4 value; float error,cosine; uint status; if (!gatherCard(card,answer.Hit,radius,value,error,cosine,status)) continue;
        if (validity[group]==0 || error<errors[group] || (error==errors[group] && value.w>directions[group].w))
        { directions[group]=value; errors[group]=error; cosines[group]=cosine; validity[group]=status; selected[group]=card; }
    }
    float total=0,available=0,largest=0; uint state=ResolvedLight;
    [unroll] for (uint measure=0;measure<CacheDirections;++measure) if (validity[measure]!=0)
    { float orientation=cosines[measure]*cosines[measure]; total+=orientation*directions[measure].w; available+=orientation; }
    if (total>0)
    {
        [unroll] for (uint accumulate=0;accumulate<CacheDirections;++accumulate) if (validity[accumulate]!=0)
        {
            float weight=cosines[accumulate]*cosines[accumulate]*directions[accumulate].w/total;
            answer.RadianceCoverage.xyz+=directions[accumulate].xyz*weight; ++answer.Cache.y; state=max(state,validity[accumulate]);
            if (validity[accumulate]!=ResolvedLight) ++answer.Cache.w;
            if (weight>largest || (weight==largest && selected[accumulate]<answer.Cache.x)) { largest=weight; answer.Cache.x=selected[accumulate]; }
        }
        answer.RadianceCoverage.w=saturate(total/available); answer.Cache.z=state;
        if (state==InvalidLight || !all(isfinite(answer.RadianceCoverage))) { answer.RadianceCoverage=0; answer.Cache.z=InvalidLight; }
    }
    CacheOutput[query]=answer;
}
#elif defined(EVANS_LUMEN_LOOKUP_STAGE)
// A material lookup gathers four depth-tested taps per card. Layers compete
// within their original projection direction before angular interpolation.
cbuffer LookupConstants:register(b0) { uint4 Count,Atlas; float4 Settings; };
Texture2D<float4> CacheReflectance:register(t0),CacheOrientation:register(t1),CacheEmission:register(t2);
struct CardRecord { row_major float4x4 ToCard,FromCard; float4 Origin,BackwardScale; uint4 Tile; };
struct SurfaceQuery { float4 LocationRadius,NormalCosine; uint4 Range; };
struct SurfaceResult { float4 Reflectance,NormalCoverage,Emission; uint4 Metadata; };
StructuredBuffer<CardRecord> Cards:register(t3);
StructuredBuffer<SurfaceQuery> Requests:register(t4);
RWStructuredBuffer<SurfaceResult> Results:register(u0);
static const uint QueryGroupWidth=64,DirectionCount=6,Gutter=1,MissingCard=0xffffffffu;
SurfaceResult missingSurface()
{ SurfaceResult v; v.Reflectance=0; v.NormalCoverage=0; v.Emission=0; v.Metadata=uint4(MissingCard,0,0,0); return v; }
float largestCoordinate(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
bool evaluateCard(CardRecord card,SurfaceQuery query,out SurfaceResult value,out float discrepancy,out float orientation)
{
    value=missingSurface(); discrepancy=0; orientation=dot(query.NormalCosine.xyz,card.BackwardScale.xyz);
    if (orientation<Settings.x) return false;
    float3 local=query.LocationRadius.xyz-card.Origin.xyz,clip=mul(card.ToCard,float4(local,1)).xyz;
    if (!all(isfinite(clip)) || any(abs(clip.xy)>1)) return false;
    float2 coordinate=float2(card.Tile.xy)+Gutter+float2(clip.x+1,1-clip.y)*0.5*Count.y-0.5;
    int2 base=int2(floor(coordinate)); float2 blend=frac(coordinate); float weightSum=0; float3 normalSum=0;
    [unroll] for (uint y=0;y<2;++y) [unroll] for (uint x=0;x<2;++x)
    {
        int2 texel=base+int2(x,y); float weight=(x?blend.x:1-blend.x)*(y?blend.y:1-blend.y);
        float4 normalDepth=CacheOrientation.Load(int3(texel,0));
        if (weight<=0 || dot(normalDepth.xyz,normalDepth.xyz)==0 || dot(normalDepth.xyz,query.NormalCosine.xyz)<query.NormalCosine.w) continue;
        float2 uv=(float2(texel)-float2(card.Tile.xy)-Gutter+0.5)/Count.y;
        float3 position=mul(card.FromCard,float4(uv.x*2-1,1-uv.y*2,normalDepth.w,1)).xyz;
        float error=abs(dot(local-position,query.NormalCosine.xyz));
        float uncertainty=Settings.y*max(card.BackwardScale.w,max(largestCoordinate(local),largestCoordinate(position)))+
            Settings.z*dot(abs(query.LocationRadius.xyz),abs(query.NormalCosine.xyz));
        if (!isfinite(error) || error>query.LocationRadius.w+uncertainty) continue;
        weightSum+=weight; discrepancy+=error*weight; normalSum+=normalDepth.xyz*weight;
        value.Reflectance+=CacheReflectance.Load(int3(texel,0))*weight; value.Emission+=CacheEmission.Load(int3(texel,0))*weight;
    }
    if (weightSum<=0) return false;
    value.Reflectance/=weightSum; value.Emission/=weightSum; value.NormalCoverage=float4(normalSum/weightSum,saturate(weightSum));
    value.Metadata=uint4(card.Tile.w,1,1,0); discrepancy/=weightSum; return true;
}
[numthreads(QueryGroupWidth,1,1)]
void CSLookup(uint3 invocation:SV_DispatchThreadID)
{
    uint index=invocation.x+invocation.y*Atlas.z; if (index>=Count.x) return;
    SurfaceQuery query=Requests[index]; SurfaceResult selected[DirectionCount]; float layerError[DirectionCount],facing[DirectionCount];
    [branch] if (query.Range.y==0) { Results[index]=missingSurface(); return; }
    [unroll] for (uint initialize=0;initialize<DirectionCount;++initialize) { selected[initialize]=missingSurface(); layerError[initialize]=0; facing[initialize]=0; }
    [loop] for (uint candidate=0;candidate<query.Range.y;++candidate)
    {
        CardRecord card=Cards[query.Range.x+candidate]; SurfaceResult value; float error,cosine;
        if (!evaluateCard(card,query,value,error,cosine)) continue;
        uint axis=card.Tile.z;
        if (selected[axis].Metadata.z==0 || error<layerError[axis] || (error==layerError[axis] && value.NormalCoverage.w>selected[axis].NormalCoverage.w))
        { selected[axis]=value; layerError[axis]=error; facing[axis]=cosine; }
    }
    SurfaceResult result=missingSurface(); float normalization=0,orientationSum=0;
    [unroll] for (uint measure=0;measure<DirectionCount;++measure) if (selected[measure].Metadata.z!=0)
    { float weight=facing[measure]*facing[measure]; normalization+=weight*selected[measure].NormalCoverage.w; orientationSum+=weight; }
    if (normalization>0)
    {
        float3 normal=0; float largestWeight=0;
        [unroll] for (uint combine=0;combine<DirectionCount;++combine) if (selected[combine].Metadata.z!=0)
        {
            float weight=facing[combine]*facing[combine]*selected[combine].NormalCoverage.w/normalization;
            result.Reflectance+=selected[combine].Reflectance*weight; result.Emission+=selected[combine].Emission*weight; normal+=selected[combine].NormalCoverage.xyz*weight;
            ++result.Metadata.y;
            if (weight>largestWeight || (weight==largestWeight && selected[combine].Metadata.x<result.Metadata.x)) { largestWeight=weight; result.Metadata.x=selected[combine].Metadata.x; }
        }
        float squared=dot(normal,normal); result.NormalCoverage=float4(squared>0?normal*rsqrt(squared):query.NormalCosine.xyz,saturate(normalization/orientationSum)); result.Metadata.z=1;
    }
    result.Reflectance=saturate(result.Reflectance); result.Emission.w=saturate(result.Emission.w);
    Results[index]=result;
}
#elif defined(EVANS_LUMEN_DISTANCE_TRACE_STAGE) || defined(EVANS_LUMEN_LIGHTING_STAGE) || defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE) || defined(EVANS_LUMEN_PROBE_TRACE_STAGE) || defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE) || defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
// Software ray queries use sparse mesh SDFs and a stackless instance BVH.
// Wright, "Dynamic Occlusion with Signed Distance Fields" (SIGGRAPH 2015).
cbuffer TraceConstants:register(b0)
{
    uint4 Count,Offset,Work; float4 Options,HairSky;
#if defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
    row_major float4x4 VisibilityInverse; uint4 VisibilityViewport,VisibilityGrid; float4 PlacementRules,VisibilityAdaptive;
    float4 SurfaceRules,NumericRules; uint4 VisibilityModel; float4 LinkRules;
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uint4 ProbeLayout; float4 ProbeInterval;
#endif
#if defined(EVANS_LUMEN_LIGHTING_STAGE)
    uint4 CacheSize,LightWork; float4 Transport,Precision; uint4 TileDispatch,BounceWork; float4 BounceSky,EnvironmentAxis,EnvironmentPeak;
#endif
};
#if defined(EVANS_LUMEN_LIGHTING_STAGE)
StructuredBuffer<uint> SceneWords:register(t3);
Texture2D<float4> CacheReflectance:register(t0),CacheOrientation:register(t1),CacheEmission:register(t2);
StructuredBuffer<uint4> RelightVectors:register(t4);
struct RelightValue { float4 RadianceCoverage; uint4 Metadata; };
RWStructuredBuffer<RelightValue> Relit:register(u0);
#else
StructuredBuffer<uint> SceneWords:register(t0);
#endif
struct TraceRay { float4 OriginMinimum,DirectionMaximum; };
struct TraceHit { float4 PositionDistance,NormalResidual,Uncertainty; uint4 Metadata; float4 Medium; };
#if defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
StructuredBuffer<ProbePosition> CandidatePositions:register(t1);
Texture2D<float> ReceiverDepth:register(t2);
Texture2D<float4> ReceiverNormals:register(t3);
RWStructuredBuffer<uint> VisibleCandidates:register(u0);
#elif defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
StructuredBuffer<ProbePosition> VolumePositions:register(t1);
RWStructuredBuffer<float> VolumeSupport:register(u0);
#elif !defined(EVANS_LUMEN_LIGHTING_STAGE)
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
StructuredBuffer<ProbePosition> ProbeSources:register(t1);
#else
StructuredBuffer<TraceRay> Queries:register(t1);
#endif
RWStructuredBuffer<TraceHit> Answers:register(u0);
#endif
static const uint GroupWidth=64,FieldStride=20,InstanceStride=20,NodeStride=12,RowStride=4;
static const uint GridOffset=0,BrickGridOffset=4,StorageOffset=8,MetricOffset=12,ErrorOffset=16;
static const uint TransformOffset=4,InstanceTagOffset=16,BoundsMaxOffset=4,LinkOffset=8;
static const uint InteriorSide=6,StoredSide=8,Halo=1,SamplesPerBrick=512,FirstStoredBrick=2;
static const uint InvalidIndex=0xffffffffu,Miss=0,Surface=1,Unresolved=2;
static const float QuantizedMaximum=32767;
uint4 loadWords(uint at) { return uint4(SceneWords[at],SceneWords[at+1],SceneWords[at+2],SceneWords[at+3]); }
float4 loadFloats(uint at) { return asfloat(loadWords(at)); }
float3 mapVector(uint instance,float3 v)
{ return float3(dot(loadFloats(instance+TransformOffset).xyz,v),dot(loadFloats(instance+TransformOffset+RowStride).xyz,v),dot(loadFloats(instance+TransformOffset+RowStride*2).xyz,v)); }
float robustLength(float3 v)
{ float largest=max(abs(v.x),max(abs(v.y),abs(v.z))); return largest>0?length(v/largest)*largest:0; }
float3 robustUnit(float3 v,float3 fallback)
{
    if (!all(isfinite(v))) return fallback;
    float largest=max(abs(v.x),max(abs(v.y),abs(v.z))); if (largest==0) return fallback;
    v/=largest; return v*rsqrt(dot(v,v));
}
bool clipBox(float3 o,float3 d,float3 lo,float3 hi,inout float first,inout float last)
{
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (d[axis]==0) { if (o[axis]<lo[axis] || o[axis]>hi[axis]) return false; }
        else
        {
            float a=(lo[axis]-o[axis])/d[axis],b=(hi[axis]-o[axis])/d[axis];
            first=max(first,min(a,b)); last=min(last,max(a,b)); if (first>last) return false;
        }
    }
    return true;
}
float readField(uint field,float3 p)
{
    uint3 size=loadWords(field+GridOffset).xyz,bricks=loadWords(field+BrickGridOffset).xyz;
    uint4 storage=loadWords(field+StorageOffset); float4 metric=loadFloats(field+MetricOffset);
    float band=loadFloats(field+ErrorOffset).x;
    float3 grid=clamp((p-metric.xyz)/metric.w-0.5,0,float3(size)-1),blend=frac(grid);
    uint3 lower=uint3(floor(grid)),block=lower/InteriorSide,local=lower%InteriorSide+Halo;
    uint address=SceneWords[storage.x+(block.z*bricks.y+block.y)*bricks.x+block.x];
    if (address<FirstStoredBrick) return address==1?-band:band;
    float sum=0;
    [unroll] for (uint corner=0;corner<8;++corner)
    {
        uint3 delta=uint3(corner&1,(corner>>1)&1,(corner>>2)&1),at=local+delta;
        uint element=(address-FirstStoredBrick)*SamplesPerBrick+(at.z*StoredSide+at.y)*StoredSide+at.x;
        uint pair=SceneWords[storage.y+element/2]; int signedValue=(element&1)?(int(pair)>>16):(int(pair<<16)>>16);
        float3 weight=float3(delta.x?blend.x:1-blend.x,delta.y?blend.y:1-blend.y,delta.z?blend.z:1-blend.z);
        sum+=float(signedValue)*weight.x*weight.y*weight.z;
    }
    return sum*(band/QuantizedMaximum);
}
float3 fieldNormal(uint instance,uint field,float3 p,float cell,float3 fallback)
{
    float3 gradient=float3(readField(field,p+float3(cell,0,0))-readField(field,p-float3(cell,0,0)),
        readField(field,p+float3(0,cell,0))-readField(field,p-float3(0,cell,0)),
        readField(field,p+float3(0,0,cell))-readField(field,p-float3(0,0,cell)));
    gradient=robustUnit(gradient,0);
    return robustUnit(loadFloats(instance+TransformOffset).xyz*gradient.x+loadFloats(instance+TransformOffset+RowStride).xyz*gradient.y+
        loadFloats(instance+TransformOffset+RowStride*2).xyz*gradient.z,fallback);
}
uint marchMesh(TraceRay ray,uint index,float anchor,float end,out float distance,out float4 normal,out float2 error,out uint steps)
{
    distance=anchor; normal=0; error=0; steps=0;
    uint instance=Offset.y+index*InstanceStride,field=Offset.x+loadWords(instance+InstanceTagOffset).x*FieldStride;
    float3 o=mapVector(instance,ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*anchor-loadFloats(instance).xyz);
    float3 d=mapVector(instance,ray.DirectionMaximum.xyz); float speed=robustLength(d);
    if (!all(isfinite(o)) || !all(isfinite(d)) || !isfinite(speed) || speed<=0) return Unresolved;
    float4 metric=loadFloats(field+MetricOffset),bounds=loadFloats(field+ErrorOffset);
    float entry=0,exit=max(0,end-anchor);
    if (!clipBox(o,d,metric.xyz,metric.xyz+float3(loadWords(field+GridOffset).xyz)*metric.w,entry,exit)) return Miss;
    float t=entry,tolerance=bounds.y+Options.x*metric.w;
    [loop] for (uint iteration=0;iteration<Work.x;++iteration)
    {
        distance=min(end,anchor+t); float3 p=o+d*t; ++steps;
        if (!all(isfinite(p))) return Unresolved;
        float signedDistance=readField(field,p);
        if (abs(signedDistance)<=tolerance)
        {
            normal=float4(fieldNormal(instance,field,p,metric.w,-ray.DirectionMaximum.xyz),signedDistance/speed);
            error=float2(bounds.y,tolerance)/speed; return Surface;
        }
        float next=t+(abs(signedDistance)-bounds.y)/speed*Options.y;
        if (!isfinite(next) || next<=t) return Unresolved;
        if (next>exit) return Miss; t=next;
    }
    distance=min(end,anchor+t); return Unresolved;
}
// Plane-distance conversion uses the inverse transpose. A transformed vector's
// length would underestimate the required source offset under shear.
float normalMetric(uint instance,float3 worldNormal)
{
    float3 row0=loadFloats(instance+TransformOffset).xyz,row1=loadFloats(instance+TransformOffset+RowStride).xyz,row2=loadFloats(instance+TransformOffset+RowStride*2).xyz;
    float3 extent=max(abs(row0),max(abs(row1),abs(row2))); float magnitude=max(extent.x,max(extent.y,extent.z)); if (magnitude==0) return 0;
    row0/=magnitude; row1/=magnitude; row2/=magnitude;
    float3 cofactor0=cross(row1,row2),cofactor1=cross(row2,row0),cofactor2=cross(row0,row1);
    float denominator=robustLength(float3(dot(cofactor0,worldNormal),dot(cofactor1,worldNormal),dot(cofactor2,worldNormal)));
    float metric=denominator>0?magnitude*(abs(dot(row0,cofactor0))/denominator):0; return isfinite(metric)?metric:0;
}
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
// DXR 1.1 inline queries retain application-owned cutout filtering. Both maps
// are bilinearly filtered before multiplication, matching SceneRayQuery.
RaytracingAccelerationStructure HardwareScene:register(t8);
StructuredBuffer<uint> HardwareWords:register(t9);
static const uint HardwareInstanceWords=16,HardwareTriangleWords=24,HardwareMaterialWords=12;
static const uint HardwareGeometryMask=1,HardwareShadowMask=2;
// Word offsets mirror RaySceneTriangle/Material and LumenHardwareGPUInstance.
static const uint HardwareB=4,HardwareC=8,HardwareUVAB=12,HardwareUVC=16,HardwareTags=20;
static const uint HardwareOpacity=4,HardwareCoverage=8,HardwareNormal0=4,HardwareNormal1=8,HardwareNormal2=12;
uint4 hardwareWords(uint at) { return uint4(HardwareWords[at],HardwareWords[at+1],HardwareWords[at+2],HardwareWords[at+3]); }
float4 hardwareFloats(uint at) { return asfloat(hardwareWords(at)); }
float hardwareImage(uint base,uint4 image,float2 uv)
{
    if (image.w==0) return 1;
    float2 wrapped=frac(uv),pixel=float2(wrapped.x,1-wrapped.y)*float2(image.yz)-0.5;
    int2 first=int2(floor(pixel)),size=int2(image.yz); float2 f=frac(pixel); float value=0;
    [unroll] for (uint corner=0;corner<4;++corner)
    {
        int2 delta=int2(corner&1,corner>>1),at=((first+delta)%size+size)%size;
        float2 weight=float2(delta.x?f.x:1-f.x,delta.y?f.y:1-f.y);
        value+=asfloat(HardwareWords[base+image.x+uint(at.y)*image.y+uint(at.x)])*weight.x*weight.y;
    }
    return value;
}
bool hardwareCoverage(uint instance,uint primitive,float2 barycentric,uint flags)
{
    uint4 storage=hardwareWords(instance*HardwareInstanceWords);
    uint face=storage.x+primitive*HardwareTriangleWords,material=storage.y+hardwareWords(face+HardwareTags).x*HardwareMaterialWords;
    if ((hardwareWords(face+HardwareTags).w&flags)!=flags) return false;
    float4 coverage=hardwareFloats(material+HardwareCoverage); if (coverage.y<=0) return true;
    float3 bary=float3(1-barycentric.x-barycentric.y,barycentric);
    float4 a=hardwareFloats(face),b=hardwareFloats(face+HardwareB),c=hardwareFloats(face+HardwareC),uvab=hardwareFloats(face+HardwareUVAB);
    float2 uv=uvab.xy*bary.x+uvab.zw*bary.y+hardwareFloats(face+HardwareUVC).xy*bary.z;
    float alpha=saturate(dot(float3(a.w,b.w,c.w),bary)*coverage.x*hardwareImage(storage.z,hardwareWords(material),uv)*hardwareImage(storage.z,hardwareWords(material+HardwareOpacity),uv));
    return isfinite(alpha) && alpha>=coverage.y;
}
void traceHardwareOpaque(TraceRay ray,out TraceHit hit,uint flags)
{
    hit=(TraceHit)0; hit.Medium=float4(0,0,0,1); hit.Metadata=uint4(InvalidIndex,InvalidIndex,Miss,0);
    hit.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
    if (Count.z==0 || ray.DirectionMaximum.w<=ray.OriginMinimum.w) return;
    RayDesc segment; segment.Origin=ray.OriginMinimum.xyz; segment.TMin=ray.OriginMinimum.w;
    segment.Direction=ray.DirectionMaximum.xyz; segment.TMax=ray.DirectionMaximum.w;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(HardwareScene,RAY_FLAG_NONE,flags!=0?HardwareShadowMask:HardwareGeometryMask,segment);
    uint selected=InvalidIndex,primitive=InvalidIndex; float nearest=segment.TMax;
    [loop] while (query.Proceed())
    {
        if (query.CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
        uint candidate=query.CandidateInstanceID(),face=query.CandidatePrimitiveIndex(); float distance=query.CandidateTriangleRayT();
        if (!hardwareCoverage(candidate,face,query.CandidateTriangleBarycentrics(),flags)) continue;
        if (selected==InvalidIndex || distance<nearest || (distance==nearest && (candidate<selected || (candidate==selected && face<primitive))))
        { nearest=distance; selected=candidate; primitive=face; query.CommitNonOpaqueTriangleHit(); }
    }
    if (selected==InvalidIndex) return;
    uint instance=selected*HardwareInstanceWords,face=hardwareWords(instance).x+primitive*HardwareTriangleWords;
    float3 a=hardwareFloats(face).xyz,b=hardwareFloats(face+HardwareB).xyz,c=hardwareFloats(face+HardwareC).xyz;
    float3 e0=b-a,e1=c-a;
    if (!all(isfinite(e0)) || !all(isfinite(e1)))
    { float scale=max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(max(abs(b.x),max(abs(b.y),abs(b.z))),max(abs(c.x),max(abs(c.y),abs(c.z))))); e0=b/scale-a/scale; e1=c/scale-a/scale; }
    float3 local=cross(robustUnit(e0,0),robustUnit(e1,0));
    float3 normal=robustUnit(float3(dot(hardwareFloats(instance+HardwareNormal0).xyz,local),dot(hardwareFloats(instance+HardwareNormal1).xyz,local),dot(hardwareFloats(instance+HardwareNormal2).xyz,local)),0);
    hit.PositionDistance=float4(segment.Origin+segment.Direction*nearest,nearest);
    hit.NormalResidual=float4(normal,0);
    hit.Metadata=uint4(selected,loadWords(Offset.y+selected*InstanceStride+InstanceTagOffset).y,dot(normal,normal)>0?Surface:Unresolved,0);
}
#endif
void traceLocalOpaqueScene(TraceRay ray,out TraceHit hit,uint requiredFlags)
{
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
    traceHardwareOpaque(ray,hit,requiredFlags);
#else
    hit.Medium=float4(0,0,0,1);
    hit.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
    hit.NormalResidual=0; hit.Uncertainty=0; hit.Metadata=uint4(InvalidIndex,InvalidIndex,Miss,0);
    // Root SRVs have no descriptor bounds. Keep empty-scene work in an explicit
    // branch so the compiler cannot speculate a record load from the dummy word.
    [branch] if (Count.y==0) return;
    float best=ray.DirectionMaximum.w,unknown=best;
    uint bestInstance=InvalidIndex,unknownInstance=InvalidIndex,node=0,steps=0; float4 bestNormal=0; float2 bestError=0;
    [loop] while (node<Count.y)
    {
        uint address=Offset.z+node*NodeStride; uint4 links=loadWords(address+LinkOffset);
        float entry=ray.OriginMinimum.w,exit=min(best,unknown);
        if (!clipBox(ray.OriginMinimum.xyz,ray.DirectionMaximum.xyz,loadFloats(address).xyz,loadFloats(address+BoundsMaxOffset).xyz,entry,exit)) { node=links.w; continue; }
        if (links.z==0) { node=links.x; continue; }
        uint visibility=loadWords(Offset.y+links.y*InstanceStride+InstanceTagOffset).z;
        if ((visibility&requiredFlags)!=requiredFlags) { node=links.w; continue; }
        float t; float4 normal; float2 error; uint work;
        uint status=marchMesh(ray,links.y,entry,exit,t,normal,error,work); steps+=work;
        if (status==Surface && (t<best || (t==best && links.y<bestInstance))) { best=t; bestInstance=links.y; bestNormal=normal; bestError=error; }
        if (status==Unresolved && (t<unknown || (t==unknown && links.y<unknownInstance))) { unknown=t; unknownInstance=links.y; }
        node=links.w;
    }
    hit.Metadata.w=steps; float t=ray.DirectionMaximum.w;
    [branch] if (unknownInstance!=InvalidIndex && (bestInstance==InvalidIndex || unknown<=best))
    { t=unknown; hit.Metadata.xyz=uint3(unknownInstance,loadWords(Offset.y+unknownInstance*InstanceStride+InstanceTagOffset).y,Unresolved); }
    else [branch] if (bestInstance!=InvalidIndex)
    { t=best; hit.NormalResidual=bestNormal; hit.Uncertainty.xy=bestError; hit.Metadata.xyz=uint3(bestInstance,loadWords(Offset.y+bestInstance*InstanceStride+InstanceTagOffset).y,Surface); }
    hit.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*t,t);
#endif
}

float globalEmptyRadius(float3 position)
{
    static const uint HeaderWords=8,ClipWords=12,MetricOffset=4,StorageOffset=8,SampleWords=4;
    uint4 shape=loadWords(Offset.w);
    [loop] for (uint cascade=0;cascade<shape.x;++cascade)
    {
        uint header=Offset.w+HeaderWords+cascade*ClipWords; float4 transform=loadFloats(header);
        float3 coordinate=(position-transform.xyz)/transform.w;
        if (!all(isfinite(coordinate)) || any(coordinate<0) || any(coordinate>float(shape.y-1))) continue;
        uint3 cell=min(uint3(floor(coordinate)),shape.y-2); float3 f=coordinate-float3(cell);
        uint first=loadWords(header+StorageOffset).x; float radius=0;
        [unroll] for (uint z=0;z<2;++z) [unroll] for (uint y=0;y<2;++y) [unroll] for (uint x=0;x<2;++x)
        {
            uint3 q=cell+uint3(x,y,z); float4 sample=loadFloats(first+((q.z*shape.y+q.y)*shape.y+q.x)*SampleWords);
            radius+=(Work.z!=0?sample.w:sample.y)*(x?f.x:1-f.x)*(y?f.y:1-f.y)*(z?f.z:1-f.z);
        }
        return isfinite(radius)?max(0,radius-loadFloats(header+MetricOffset).x):0;
    }
    return 0;
}
void traceLocalOpaqueScene(TraceRay ray,out TraceHit hit)
{ traceLocalOpaqueScene(ray,hit,Work.z); }
void traceOpaqueScene(TraceRay ray,out TraceHit hit)
{
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
    traceHardwareOpaque(ray,hit,Work.z);
#else
    hit=(TraceHit)0; hit.Medium=float4(0,0,0,1);
    bool useGlobal=Offset.w!=0; uint intervals=useGlobal && Options.z>0?2:1;
    float distance=ray.OriginMinimum.w; uint localSteps=0,globalSteps=0;
    // Retain one compiled copy of the instance BVH and sphere-tracing loops.
    [loop] for (uint interval=0;interval<intervals;++interval)
    {
        bool detailNear=intervals==2 && interval==0;
        if (useGlobal && !detailNear) [loop] for (uint step=0;step<uint(Options.w);++step)
        {
            ++globalSteps; float empty=globalEmptyRadius(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*distance);
            float next=distance+empty*Options.y; if (!isfinite(next) || next<=distance) break;
            if (next>ray.DirectionMaximum.w)
            {
                hit.PositionDistance=float4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
                hit.NormalResidual=0; hit.Uncertainty=float4(0,0,0,float(globalSteps)); hit.Metadata=uint4(InvalidIndex,InvalidIndex,Miss,localSteps); return;
            }
            distance=next;
        }
        TraceRay detail=ray; detail.OriginMinimum.w=distance;
        if (detailNear) detail.DirectionMaximum.w=min(ray.DirectionMaximum.w,distance+Options.z);
        traceLocalOpaqueScene(detail,hit); hit.Metadata.w+=localSteps; hit.Uncertainty.w=float(globalSteps);
        if (!detailNear || hit.Metadata.z!=Miss || detail.DirectionMaximum.w==ray.DirectionMaximum.w) return;
        distance=detail.DirectionMaximum.w; localSteps=hit.Metadata.w;
    }
#endif
}
// Cell-averaged fiber projected area and orientation moments follow Kajiya/Kay.
// Integrate each crossed cell analytically: T=exp(-sigma*length). The source is
// isotropic low-frequency sky scattering plus emission, not a full fiber phase
// function. Caster-only queries carry extinction without an in-scattering source.
float4 hairTransport(TraceRay ray,float limit,uint requiredFlags)
{
    static const uint HairVoxelWords=24,HairVectorWords=4;
    float4 medium=float4(0,0,0,1); uint header=Work.w>>LumenHairPointerShift; if (header==0) return medium;
    uint4 grid=loadWords(header); float4 metric=loadFloats(header+HairVectorWords);
    float first=ray.OriginMinimum.w,last=min(limit,ray.DirectionMaximum.w);
    float3 direction=ray.DirectionMaximum.xyz,origin=ray.OriginMinimum.xyz-metric.xyz;
    if (last<=first || !clipBox(origin,direction,0,float3(grid.xyz)*metric.w,first,last) || last<=first) return medium;
    int3 cell=clamp(int3(floor((origin+direction*first)/metric.w)),int3(0,0,0),int3(grid.xyz)-1);
    int3 step=int3(direction.x>0?1:direction.x<0?-1:0,direction.y>0?1:direction.y<0?-1:0,direction.z>0?1:direction.z<0?-1:0);
    bool shadow=(requiredFlags&1u)!=0;
    [loop] for (uint visit=0;visit<grid.x+grid.y+grid.z+1 && first<last;++visit)
    {
        float3 crossing=last.xxx;
        [unroll] for (uint axis=0;axis<3;++axis) if (step[axis]!=0)
            crossing[axis]=((float(cell[axis])+(step[axis]>0?1:0))*metric.w-origin[axis])/direction[axis];
        float end=max(first,min(last,min(crossing.x,min(crossing.y,crossing.z))));
        uint at=grid.w+(uint(cell.x)+grid.x*(uint(cell.y)+grid.y*uint(cell.z)))*HairVoxelWords;
        float4 optical=loadFloats(at),emission=loadFloats(at+HairVectorWords);
        float4 moment=loadFloats(at+(shadow?4:2)*HairVectorWords),crossMoment=loadFloats(at+(shadow?5:3)*HairVectorWords);
        float quadratic=dot(moment.xyz,direction*direction)+2*(moment.w*direction.x*direction.y+crossMoment.x*direction.x*direction.z+crossMoment.y*direction.y*direction.z);
        float density=shadow?emission.w:optical.x,extinction=density*sqrt(saturate(1-quadratic));
        float transmitted=exp(-extinction*(end-first));
        if (!shadow) medium.xyz+=medium.w*(1-transmitted)*(optical.yzw*HairSky.xyz+emission.xyz);
        medium.w*=transmitted; first=end; if (first>=last) break;
        bool outside=false;
        [unroll] for (uint nextAxis=0;nextAxis<3;++nextAxis) if (step[nextAxis]!=0 && crossing[nextAxis]<=end)
        { cell[nextAxis]+=step[nextAxis]; outside=outside || cell[nextAxis]<0 || cell[nextAxis]>=int(grid[nextAxis]); }
        if (outside) break;
    }
    return medium;
}
void traceLocalMeshScene(TraceRay ray,out TraceHit hit,uint requiredFlags)
{
    traceLocalOpaqueScene(ray,hit,requiredFlags);
    hit.Medium=hairTransport(ray,hit.Metadata.z==Miss?ray.DirectionMaximum.w:hit.PositionDistance.w,requiredFlags);
}
void traceLocalMeshScene(TraceRay ray,out TraceHit hit) { traceLocalMeshScene(ray,hit,Work.z); }
void traceMeshScene(TraceRay ray,out TraceHit hit)
{
    traceOpaqueScene(ray,hit);
    hit.Medium=hairTransport(ray,hit.Metadata.z==Miss?ray.DirectionMaximum.w:hit.PositionDistance.w,Work.z);
}
#if defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
float volumeMagnitude(float3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
float volumeSupportForInstance(uint index,float3 world)
{
    uint instance=Offset.y+index*InstanceStride,field=Offset.x+loadWords(instance+InstanceTagOffset).x*FieldStride;
    float3 center=loadFloats(instance).xyz,relative=world-center;
    float3 a=loadFloats(instance+TransformOffset).xyz,b=loadFloats(instance+TransformOffset+RowStride).xyz,c=loadFloats(instance+TransformOffset+RowStride*2).xyz;
    float3 p=float3(dot(a,relative),dot(b,relative),dot(c,relative)); float scale=max(volumeMagnitude(a),max(volumeMagnitude(b),volumeMagnitude(c)));
    if (!all(isfinite(p)) || !isfinite(scale) || scale<=0) return 0;
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    // The induced 1/infinity norms bound the spectral norm. The resulting
    // world sphere remains a lower clearance bound for scaled instances.
    float rows=max(dot(a,1.0.xxx),max(dot(b,1.0.xxx),dot(c,1.0.xxx))),norm=scale*sqrt(rows*volumeMagnitude(a+b+c));
    if (!isfinite(norm) || norm<=0) return 0;
    float4 grid=loadFloats(field+MetricOffset),errors=loadFloats(field+ErrorOffset);
    float3 end=grid.xyz+float3(loadWords(field+GridOffset).xyz)*grid.w,sampled=clamp(p,grid.xyz+grid.w*0.5,end-grid.w*0.5);
    float outside=robustLength(p-clamp(p,grid.xyz,end)),extension=robustLength(p-sampled);
    // One sampled point needs one interpolation error budget; the extra future
    // hit errors used for global ray-step certificates do not apply to a sphere.
    float lower=max(outside,abs(readField(field,sampled))-errors.y-Options.x*grid.w-extension);
    lower-=Options.y*max(volumeMagnitude(p),max(volumeMagnitude(grid.xyz),volumeMagnitude(end)));
    float support=lower/norm-Options.y*max(volumeMagnitude(world),volumeMagnitude(center));
    return isfinite(support)?clamp(support,0,Options.z):0;
}
[numthreads(GroupWidth,1,1)]
void CSVolumeSupport(uint3 thread:SV_DispatchThreadID)
{
    uint index=thread.x+thread.y*Work.y; if (index>=Count.x) return;
    ProbePosition probe=VolumePositions[index]; float support=probe.Metadata.y==0?Options.z:0;
    [loop] for (uint instance=0;instance<Count.z && support>0;++instance) support=min(support,volumeSupportForInstance(instance,probe.PositionMinimum.xyz));
    VolumeSupport[index]=support;
}
#elif defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
// Visibility is evaluated against the probe producer's immutable GI scene.
// Only a completed miss is a clear segment; step exhaustion retains occlusion.
float visibilityCoordinate(float center,uint extent,uint tiles)
{
    float width=VisibilityViewport.z,coordinate=center/width-0.5;
    if (tiles>1 && extent%VisibilityViewport.z!=0)
    {
        float previous=(float(tiles)-1.5)*width,finalCenter=(float(tiles-1)*width+extent)*0.5;
        if (center>previous) coordinate=float(tiles-2)+(center-previous)/(finalCenter-previous);
    }
    return clamp(coordinate,0,float(tiles-1));
}
float visibilityScale(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
[numthreads(GroupWidth,1,1)]
void CSProbeVisibility(uint3 invocation:SV_DispatchThreadID)
{
    uint receiver=invocation.x+invocation.y*Work.y; if (receiver>=Count.x) return;
    uint words=probeLinkWords(VisibilityGrid.w+1);
    for (uint clearWord=0;clearWord<words;++clearWord) VisibleCandidates[receiver*words+clearWord]=0;
    uint2 texel=uint2(receiver%VisibilityViewport.x,receiver/VisibilityViewport.x);
    float z=ReceiverDepth.Load(int3(texel,0)); float4 material=ReceiverNormals.Load(int3(texel,0));
    if (!isfinite(z) || z<0 || z>=1 || !all(isfinite(material)) || material.w<0 || dot(material.xyz,material.xyz)<=NumericRules.x) return;
    float2 center=float2(texel)+0.5,uv=center/float2(VisibilityViewport.xy);
    float4 homogeneous=mul(VisibilityInverse,float4(uv.x*2-1,1-uv.y*2,z,1));
    if (!all(isfinite(homogeneous)) || abs(homogeneous.w)<=PlacementRules.w) return;
    float3 position=homogeneous.xyz/homogeneous.w,n=normalize(material.xyz);
    if (!all(isfinite(position)) || any(abs(position)>PlacementRules.z)) return;
    float2 grid=float2(visibilityCoordinate(center.x,VisibilityViewport.x,VisibilityViewport.w),visibilityCoordinate(center.y,VisibilityViewport.y,VisibilityGrid.x));
    uint2 base=uint2(floor(grid)); float2 blend=grid-float2(base); uint3 accepted=0;
    uint slots=VisibilityGrid.w+1;
    [loop] for (uint candidate=0;candidate<4*slots;++candidate)
    {
        uint cornerIndex=candidate/slots;
        uint2 corner=uint2(cornerIndex&1,cornerIndex>>1),tile=min(base+corner,uint2(VisibilityViewport.w,VisibilityGrid.x)-1);
        if ((corner.x?blend.x:1-blend.x)*(corner.y?blend.y:1-blend.y)<=0) continue;
        uint id=(tile.y*VisibilityViewport.w+tile.x)*slots+candidate%slots; ProbePosition probe=CandidatePositions[id];
        if (probe.Metadata.x!=id || probe.Metadata.y!=0 || any(probe.Metadata.zw>=VisibilityViewport.xy)) continue;
        float4 materialAtProbe=ReceiverNormals.Load(int3(probe.Metadata.zw,0));
        if (!all(isfinite(materialAtProbe)) || materialAtProbe.w<0 || dot(materialAtProbe.xyz,materialAtProbe.xyz)<=NumericRules.x) continue;
        float3 probeNormal=normalize(materialAtProbe.xyz),anchor=probe.PositionMinimum.xyz-probeNormal*PlacementRules.x;
        if (dot(n,probeNormal)<=SurfaceRules.y || !all(isfinite(anchor))) continue;
        float offset=LinkRules.x+LinkRules.y*max(visibilityScale(position),visibilityScale(anchor));
        float3 begin=position+n*offset,destination=anchor+probeNormal*offset,separation=destination-begin;
        float span=robustLength(separation);
        if (!all(isfinite(begin)) || !all(isfinite(destination)) || !isfinite(span)) continue;
        if (span==0) { accepted|=probeLinkBit(candidate); continue; }
        TraceRay segment; segment.OriginMinimum=float4(begin,0); segment.DirectionMaximum=float4(robustUnit(separation,n),span);
        TraceHit answer; answer.Medium=float4(0,0,0,1); traceMeshScene(segment,answer); if (answer.Metadata.z==Miss) accepted|=probeLinkBit(candidate);
    }
    for (uint storedWord=0;storedWord<words;++storedWord) VisibleCandidates[receiver*words+storedWord]=accepted[storedWord];
}
#elif !defined(EVANS_LUMEN_LIGHTING_STAGE)
[numthreads(GroupWidth,1,1)]
void CSTraceDistance(uint3 thread:SV_DispatchThreadID)
{
    uint query=thread.x+thread.y*Work.y; if (query>=Count.x) return;
    TraceRay ray;
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    ProbePosition source=ProbeSources[query/ProbeLayout.x]; ray.OriginMinimum=source.PositionMinimum;

    if (source.Metadata.y!=0)
    { TraceHit empty; empty.Medium=float4(0,0,0,1); empty.PositionDistance=0; empty.NormalResidual=0; empty.Uncertainty=0; empty.Metadata=uint4(InvalidIndex,InvalidIndex,Unresolved,0); Answers[query]=empty; return; }
    float3 direction=ProbeInterval.y>0?directionFromCacheTexel(query%ProbeLayout.x,uint(ProbeInterval.y)):
        directionForProbe(query%ProbeLayout.x,ProbeLayout.x,source.Metadata.x,ProbeLayout.z);
    bool cached=(ProbeLayout.w&2)!=0 && Answers[query].Metadata.z==LumenWorldCache;
    bool cachedSurface=cached && Answers[query].Metadata.x!=InvalidIndex;
    // Unlocated cache radiance cannot bound a medium segment; retain full tracing.
    if ((Work.w>>LumenHairPointerShift)!=0 && !cachedSurface) cached=false;
    ray.DirectionMaximum=float4(direction,cached && !cachedSurface?Answers[query].PositionDistance.w:ProbeInterval.x);
    if ((ProbeLayout.w&1)!=0 && Answers[query].Metadata.z==LumenScreenSurface)
    { Answers[query].Medium=hairTransport(ray,Answers[query].PositionDistance.w,Work.z); return; }
#else
    ray=Queries[query];
#if defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE)
    static const uint ReflectionGenerated=1,ReflectionScreenFirst=2;
    if ((Work.w&ReflectionScreenFirst)!=0 && Answers[query].Metadata.z==LumenScreenSurface) { Answers[query].Medium=hairTransport(ray,Answers[query].PositionDistance.w,Work.z); return; }
    if ((Work.w&ReflectionGenerated)!=0 && ray.DirectionMaximum.w<=0)
    { TraceHit empty=(TraceHit)0; empty.Medium=float4(0,0,0,1); empty.Metadata=uint4(InvalidIndex,InvalidIndex,Unresolved,0); Answers[query]=empty; return; }
#endif
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uint layers=uint(ProbeInterval.z); bool terminal=false; TraceHit previous=(TraceHit)0; previous.Medium=float4(0,0,0,1);
    static const float LayerCrossingCosine=0.05,LayerPlaneNormalCosine=0.95;
    // A depth stack stores SDF crossings, not repeated samples of one hit shell.
    // Sub-footprint and grazing continuation cannot certify another layer.
    [loop] for (uint layer=0;layer<layers;++layer)
    {
    TraceHit hit=previous; if (!terminal) traceMeshScene(ray,hit);
#else
    TraceHit hit; hit.Medium=float4(0,0,0,1); traceMeshScene(ray,hit);
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    if (cached && !cachedSurface && hit.Metadata.z==Miss) { Answers[query].Medium=hit.Medium; return; }
#endif
#if defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE) || defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    [branch] if (hit.Metadata.z==Surface)
    {
        uint instance=Offset.y+hit.Metadata.x*InstanceStride,field=Offset.x+loadWords(instance+InstanceTagOffset).x*FieldStride;
        float threshold=loadFloats(field+ErrorOffset).y+Options.x*loadFloats(field+MetricOffset).w;
#if !defined(EVANS_LUMEN_HARDWARE_RAYS)
        hit.Uncertainty.z=threshold/normalMetric(instance,hit.NormalResidual.xyz);
#endif
    }
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    float roundoff=ProbeInterval.w*max(1,max(abs(ray.OriginMinimum.x),max(abs(ray.OriginMinimum.y),abs(ray.OriginMinimum.z)))+ray.DirectionMaximum.w);
    if (cachedSurface && hit.Metadata.z==Surface)
    {
        TraceHit expected=Answers[query]; float tolerance=expected.Uncertainty.x+hit.Uncertainty.z+roundoff;
        if (all(hit.Metadata.xy==expected.Metadata.xy) && dot(hit.NormalResidual.xyz,expected.NormalResidual.xyz)>=LayerPlaneNormalCosine &&
            isfinite(tolerance) && abs(dot(hit.PositionDistance.xyz-expected.PositionDistance.xyz,expected.NormalResidual.xyz))<=tolerance) { Answers[query].Medium=hit.Medium; return; }
    }
    if (layer>0 && !terminal && hit.Metadata.z==Surface && hit.Metadata.x==previous.Metadata.x &&
        abs(dot(hit.NormalResidual.xyz,previous.NormalResidual.xyz))>=LayerPlaneNormalCosine &&
        abs(dot(hit.PositionDistance.xyz-previous.PositionDistance.xyz,previous.NormalResidual.xyz))<=hit.Uncertainty.z+previous.Uncertainty.z+roundoff)
    { hit.Metadata.z=Unresolved; hit.NormalResidual=0; hit.Uncertainty.xyz=0; }
    uint destination=((query/ProbeLayout.x)*layers+layer)*ProbeLayout.x+query%ProbeLayout.x;
    Answers[destination]=hit; if (layer+1==layers) break;
    previous=hit;
    if (!terminal && hit.Metadata.z==Surface)
    {
        float cosine=abs(dot(hit.NormalResidual.xyz,ray.DirectionMaximum.xyz));
        float next=hit.PositionDistance.w+2*hit.Uncertainty.z/max(cosine,LayerCrossingCosine)+roundoff;
        if (cosine>=LayerCrossingCosine && isfinite(next) && next>hit.PositionDistance.w && next<ray.DirectionMaximum.w) ray.OriginMinimum.w=next;
        else { previous.Metadata.z=Unresolved; previous.NormalResidual=0; previous.Uncertainty.xyz=0; terminal=true; }
    }
    else terminal=true;
    }
#else
    Answers[query]=hit;
#endif
}
#else
static const uint CardVectorStride=7,EmitterVectorStride=4,CardOrigin=4,CardNormal=5,CardTile=6,CardGutter=1;
static const uint DirectionalKind=0,AreaKind=2,CompleteLighting=1,UnresolvedLighting=2,InvalidLighting=3;
static const uint EmitterOrientation=1,EmitterRadiance=2,EmitterShadow=3;
float4 relightFloat(uint at) { return asfloat(RelightVectors[at]); }
float3 unprojectSurface(uint card,uint2 tile,int2 texel,float depth)
{
    float2 uv=(float2(texel)-float2(tile)-CardGutter+0.5)/float(CacheSize.z);
    float4 clip=float4(uv.x*2-1,1-uv.y*2,depth,1);
    return float3(dot(relightFloat(card),clip),dot(relightFloat(card+1),clip),dot(relightFloat(card+2),clip));
}
bool surfaceDifference(uint card,uint2 tile,int2 texel,int2 delta,float3 center,out float3 difference)
{
    int2 neighbor=texel+delta; difference=0;
    if (any(neighbor<int2(tile)) || any(neighbor>=int2(tile+LightWork.w))) return false;
    float4 depth=CacheOrientation.Load(int3(neighbor,0)); if (dot(depth.xyz,depth.xyz)==0) return false;
    difference=unprojectSurface(card,tile,neighbor,depth.w)-center; return all(isfinite(difference));
}
float3 offsetNormal(uint card,uint2 tile,int2 texel,float3 center,float3 backward)
{
    float3 xp,xn,yp,yn;
    bool hasXP=surfaceDifference(card,tile,texel,int2(1,0),center,xp),hasXN=surfaceDifference(card,tile,texel,int2(-1,0),center,xn);
    bool hasYP=surfaceDifference(card,tile,texel,int2(0,1),center,yp),hasYN=surfaceDifference(card,tile,texel,int2(0,-1),center,yn);
    if ((!hasXP && !hasXN) || (!hasYP && !hasYN)) return backward;
    float3 dx=hasXP && (!hasXN || dot(xp,xp)<=dot(xn,xn))?xp:-xn;
    float3 dy=hasYP && (!hasYN || dot(yp,yp)<=dot(yn,yn))?yp:-yn;
    float3 result=robustUnit(cross(dx,dy),backward); return dot(result,backward)<0?-result:result;
}
#if defined(EVANS_LUMEN_RADIOSITY)
// Bounded surface-cache radiosity (Wright et al., Lumen, SIGGRAPH 2022).
// The SRV belongs to a completed earlier iteration; this UAV is a new atlas.
StructuredBuffer<RelightValue> BounceSource:register(t5);
static const uint BounceDirectionCount=6,BounceCardStride=11,BounceInverse=4,BounceOrigin=8,BounceBackward=9,BounceTile=10,BounceBorder=1;
static const float BounceFacingThreshold=0.1,BounceNormalThreshold=0.25,BounceVoxelRadiusScale=2;
static const float BounceLocalPrecision=16*1.1920928955078125e-7,BounceWorldPrecision=0.5*1.1920928955078125e-7;
static const float BouncePi=3.14159265358979323846,BouncePhaseScale=1.0/4294967296.0,BounceAxisThreshold=0.9;
float bounceMagnitude(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float3 bounceTransform(uint address,float4 v)
{ return float3(dot(relightFloat(address),v),dot(relightFloat(address+1),v),dot(relightFloat(address+2),v)); }
bool bounceCardSample(uint cardIndex,TraceHit hit,float radius,out float4 value,out float error,out float facing,out uint status)
{
    uint address=BounceWork.z+BounceWork.w+cardIndex*BounceCardStride; uint4 tile=RelightVectors[address+BounceTile];
    float4 backward=relightFloat(address+BounceBackward); value=0; error=0; status=0;
    facing=dot(hit.NormalResidual.xyz,backward.xyz); if (facing<BounceFacingThreshold) return false;
    float3 relative=hit.PositionDistance.xyz-relightFloat(address+BounceOrigin).xyz;
    float3 projected=bounceTransform(address,float4(relative,1)); if (!all(isfinite(projected)) || any(abs(projected.xy)>1)) return false;
    float2 uv=float2(projected.x*0.5+0.5,0.5-projected.y*0.5),location=float2(tile.xy)+BounceBorder+uv*CacheSize.z-0.5;
    int2 first=int2(floor(location)); float2 blend=location-float2(first); float coverage=0; status=CompleteLighting;
    [unroll] for (uint tap=0;tap<4;++tap)
    {
        uint2 corner=uint2(tap&1,tap>>1); int2 texel=first+int2(corner);
        float weight=(corner.x?blend.x:1-blend.x)*(corner.y?blend.y:1-blend.y);
        float4 material=CacheOrientation.Load(int3(texel,0));
        if (weight<=0 || dot(material.xyz,material.xyz)==0 || dot(material.xyz,hit.NormalResidual.xyz)<BounceNormalThreshold) continue;
        float2 capturedUv=(float2(texel)-float2(tile.xy)-BounceBorder+0.5)/CacheSize.z;
        float3 captured=bounceTransform(address+BounceInverse,float4(capturedUv.x*2-1,1-capturedUv.y*2,material.w,1));
        float planeError=abs(dot(relative-captured,hit.NormalResidual.xyz));
        float tolerance=radius+BounceLocalPrecision*max(backward.w,max(bounceMagnitude(relative),bounceMagnitude(captured)))+
            BounceWorldPrecision*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!isfinite(planeError) || planeError>tolerance) continue;
        RelightValue previous=BounceSource[uint(texel.y)*CacheSize.x+uint(texel.x)];
        status=max(status,previous.Metadata.x==0?InvalidLighting:previous.Metadata.x);
        value.xyz+=previous.RadianceCoverage.xyz*weight; error+=planeError*weight; coverage+=weight;
    }
    if (coverage<=0) { status=0; return false; }
    value=float4(value.xyz/coverage,saturate(coverage)); error/=coverage; return true;
}
uint bounceLightingAt(TraceHit hit,out float3 radiance)
{
    radiance=0; uint4 range=RelightVectors[BounceWork.z+hit.Metadata.x]; if (range.y==0) return 0;
    uint instance=Offset.y+hit.Metadata.x*InstanceStride,field=Offset.x+loadWords(instance+InstanceTagOffset).x*FieldStride;
    float uncertainty=loadFloats(field+ErrorOffset).y+Options.x*loadFloats(field+MetricOffset).w;
    float radius=uncertainty/normalMetric(instance,hit.NormalResidual.xyz)*BounceVoxelRadiusScale;
    if (!isfinite(radius)) return InvalidLighting;
    float4 values[BounceDirectionCount]; float errors[BounceDirectionCount],facings[BounceDirectionCount]; uint states[BounceDirectionCount];
    [unroll] for (uint group=0;group<BounceDirectionCount;++group) { values[group]=0; errors[group]=0; facings[group]=0; states[group]=0; }
    [loop] for (uint candidate=0;candidate<range.y;++candidate)
    {
        uint word=BounceWork.z*4+range.x+candidate,card=RelightVectors[word/4][word%4];
        uint group=RelightVectors[BounceWork.z+BounceWork.w+card*BounceCardStride+BounceTile].z;
        float4 value; float error,facing; uint status;
        if (!bounceCardSample(card,hit,radius,value,error,facing,status)) continue;
        if (states[group]==0 || error<errors[group] || (error==errors[group] && value.w>values[group].w))
        { states[group]=status; values[group]=value; errors[group]=error; facings[group]=facing; }
    }
    float total=0; uint status=0;
    [unroll] for (uint resolvedDirection=0;resolvedDirection<BounceDirectionCount;++resolvedDirection) if (states[resolvedDirection]!=0)
    {
        float weight=facings[resolvedDirection]*facings[resolvedDirection]*values[resolvedDirection].w;
        radiance+=values[resolvedDirection].xyz*weight; total+=weight; status=max(status,states[resolvedDirection]);
    }
    if (total>0) radiance/=total;
    if (status==InvalidLighting || !all(isfinite(radiance))) { radiance=0; return InvalidLighting; }
    return total>0?status:0;
}
float3 bounceHemisphere(uint rayIndex,uint texel,float3 normal)
{
    float diskRadiusSquared=(float(rayIndex)+0.5)/BounceWork.x;
    uint rotation=reversebits(rayIndex)+reversebits(texel^BounceWork.y); float phi=2*BouncePi*(float(rotation)*BouncePhaseScale);
    float3 reference=abs(normal.z)<BounceAxisThreshold?float3(0,0,1):float3(0,1,0),tangent=normalize(cross(reference,normal)),bitangent=cross(normal,tangent);
    return tangent*(sqrt(diskRadiusSquared)*cos(phi))+bitangent*(sqrt(diskRadiusSquared)*sin(phi))+normal*sqrt(1-diskRadiusSquared);
}
#endif
[numthreads(GroupWidth,1,1)]
void CSLightSurface(uint3 thread:SV_DispatchThreadID)
{
    uint lane=thread.x+thread.y*Work.y,tileArea=TileDispatch.w*TileDispatch.w;
    if (lane>=TileDispatch.y*tileArea) return;
    uint tileIndex=TileDispatch.x+lane/tileArea,within=lane%tileArea;
    uint2 pixel=uint2(tileIndex%TileDispatch.z,tileIndex/TileDispatch.z)*TileDispatch.w+uint2(within%TileDispatch.w,within/TileDispatch.w);
    if (any(pixel>=CacheSize.xy)) return;
    uint index=pixel.y*CacheSize.x+pixel.x;
    RelightValue result; result.RadianceCoverage=0; result.Metadata=0;
    uint cardIndex=(pixel.y/LightWork.w)*CacheSize.w+pixel.x/LightWork.w;
    [branch] if (cardIndex>=LightWork.x) { Relit[index]=result; return; }
    float4 materialNormal=CacheOrientation.Load(int3(pixel,0));
    [branch] if (dot(materialNormal.xyz,materialNormal.xyz)==0) { Relit[index]=result; return; }
    uint card=cardIndex*CardVectorStride; uint4 tile=RelightVectors[card+CardTile];
    float3 local=unprojectSurface(card,tile.xy,int2(pixel),materialNormal.w),position=local+relightFloat(card+CardOrigin).xyz;
    float3 geometric=offsetNormal(card,tile.xy,int2(pixel),local,relightFloat(card+CardNormal).xyz);
    float bias=max(Transport.x,Precision.x*max(abs(position.x),max(abs(position.y),abs(position.z))));
    [branch] if (tile.z!=InvalidIndex)
    {
        uint instance=Offset.y+tile.z*InstanceStride,field=Offset.x+loadWords(instance+InstanceTagOffset).x*FieldStride;
        float speed=normalMetric(instance,geometric); float threshold=loadFloats(field+ErrorOffset).y+Options.x*loadFloats(field+MetricOffset).w;
        bias=max(bias,Transport.y*threshold/speed);
    }
    float3 origin=position+geometric*bias,irradiance=0; bool invalid=!all(isfinite(origin)) || !all(isfinite(materialNormal));
    [loop] for (uint emitterIndex=0;emitterIndex<LightWork.y && !invalid;++emitterIndex)
    {
        uint first=LightWork.z+emitterIndex*EmitterVectorStride;
        float4 source=relightFloat(first),normalRange=relightFloat(first+EmitterOrientation),energy=relightFloat(first+EmitterRadiance);
        if (!any(energy.xyz>0) || energy.w==0) continue;
        float3 wi=source.xyz; float attenuation=1;
        if (uint(source.w)!=DirectionalKind)
        {
            float3 delta=source.xyz-position; float distance=robustLength(delta); if (distance==0) continue;
            wi=delta/distance; float window=1;
            if (normalRange.w>0)
            {
                if (max(abs(delta.x),max(abs(delta.y),abs(delta.z)))>=normalRange.w) continue;
                float3 relative=delta/normalRange.w; float squared=dot(relative,relative); window=saturate(1-squared*squared); window*=window;
            }
            float emitterCosine=uint(source.w)==AreaKind?saturate(dot(normalRange.xyz,-wi)):1;
            attenuation=energy.w*window*emitterCosine/max(distance*distance,Transport.w);
        }
        invalid=!all(isfinite(wi)) || !isfinite(attenuation); if (invalid) continue;
        float cosine=saturate(dot(materialNormal.xyz,wi)); if (cosine==0 || attenuation==0 || dot(geometric,wi)<=0) continue;
        [branch] if (RelightVectors[first+EmitterShadow].x!=0)
        {
            TraceRay ray; ray.OriginMinimum=float4(origin,0); ray.DirectionMaximum=float4(wi,Transport.z);
            if (uint(source.w)!=DirectionalKind)
            {
                float3 target=source.xyz-origin; float distance=robustLength(target);
                if (distance>0) ray.DirectionMaximum=float4(target/distance,min(Transport.z,max(0,distance-Transport.x))); else ray.DirectionMaximum.w=0;
            }
            if (ray.DirectionMaximum.w>0)
            {
                ++result.Metadata.y; TraceHit visibility; visibility.Medium=float4(0,0,0,1); traceMeshScene(ray,visibility);
                if (visibility.Metadata.z==Surface) { ++result.Metadata.z; continue; }
                if (visibility.Metadata.z==Unresolved) { ++result.Metadata.w; continue; }
                attenuation*=visibility.Medium.w;
            }
        }
        irradiance+=energy.xyz*(cosine*attenuation);
    }
#if defined(EVANS_LUMEN_RADIOSITY)
    float3 bounced=0,shadingNormal=robustUnit(materialNormal.xyz,geometric);
    [loop] for (uint rayIndex=0;rayIndex<BounceWork.x && !invalid;++rayIndex)
    {
        ++result.Metadata.y; float3 direction=bounceHemisphere(rayIndex,index,shadingNormal);
        if (dot(direction,geometric)<=0) continue;
        TraceRay ray; ray.OriginMinimum=float4(origin,0); ray.DirectionMaximum=float4(direction,BounceSky.w);
        TraceHit hit; hit.Medium=float4(0,0,0,1); traceLocalMeshScene(ray,hit,0);
        if (hit.Metadata.z==Miss) { bounced+=LumenEnvironmentRadiance(BounceSky.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction)*hit.Medium.w+hit.Medium.xyz; continue; }
        if (hit.Metadata.z!=Surface) { ++result.Metadata.w; continue; }
        float3 incoming; uint status=bounceLightingAt(hit,incoming);
        if (status==InvalidLighting) { invalid=true; continue; }
        if (status!=CompleteLighting) ++result.Metadata.w;
        bounced+=incoming*hit.Medium.w+hit.Medium.xyz;
    }
    irradiance+=bounced*(BouncePi/BounceWork.x);
#endif
    float4 reflectance=CacheReflectance.Load(int3(pixel,0));
    float3 radiance=CacheEmission.Load(int3(pixel,0)).xyz+reflectance.xyz*(1-reflectance.w)*irradiance*Precision.y;
    invalid=invalid || !all(isfinite(radiance)); result.RadianceCoverage=float4(invalid?float3(0,0,0):max(radiance,0),1);
    result.Metadata.x=invalid?InvalidLighting:(result.Metadata.w!=0?UnresolvedLighting:CompleteLighting); Relit[index]=result;
}
#endif
#elif defined(EVANS_LUMEN_READ_STAGE)
cbuffer ExportParameters:register(b0) { uint4 Size; };
Texture2D<float4> Reflectance:register(t0),SurfaceNormal:register(t1),RadianceSource:register(t2);
RWStructuredBuffer<float4> Exported:register(u0);
static const uint ExportGroupWidth=64,SurfaceVectorCount=3;
[numthreads(ExportGroupWidth,1,1)]
void CSRead(uint3 thread:SV_DispatchThreadID)
{
    uint address=thread.x+thread.y*Size.z; if (address>=Size.x*Size.y) return;
    int3 coordinate=int3(address%Size.x,address/Size.x,0); uint first=address*SurfaceVectorCount;
    Exported[first]=Reflectance.Load(coordinate); Exported[first+1]=SurfaceNormal.Load(coordinate); Exported[first+2]=RadianceSource.Load(coordinate);
}
#else
cbuffer MaterialParameters:register(b0)
{
    row_major float4x4 ObjectToRelativeWorld,NormalToWorld,RelativeWorldToClip;
    float4 ReflectanceCoverage,EmissionAndRoughness,MaterialValues;
    uint4 Images;
};
Texture2D<float4> DiffuseMap:register(t0),NormalMap:register(t1),OpacityMap:register(t2);
SamplerState SurfaceSampler:register(s0);
static const float NormalEpsilon=1.0e-10,HomogeneousEpsilon=1.0e-7,UvEpsilon=1.0e-8;
static const float LinearBreakpoint=0.04045,LinearDivisor=12.92,TransferOffset=0.055,TransferScale=1.055,TransferExponent=2.4;
float3 DecodeVertex(float3 authored)
{
    authored=saturate(authored); float3 lowBranch=authored/LinearDivisor,curved=pow((authored+TransferOffset)/TransferScale,TransferExponent);
    return float3(authored.r<=LinearBreakpoint?lowBranch.r:curved.r,authored.g<=LinearBreakpoint?lowBranch.g:curved.g,authored.b<=LinearBreakpoint?lowBranch.b:curved.b);
}
float3 NormalizeMaterial(float3 value,float3 fallback)
{ float square=dot(value,value); return square>NormalEpsilon?value*rsqrt(square):fallback; }
struct CaptureInput { float3 Position:POSITION; float3 Normal:NORMAL; float4 Color:COLOR0; float2 UV:TEXCOORD0; };
struct CaptureVarying { float4 Raster:SV_POSITION; float3 Position:TEXCOORD0; float3 Normal:TEXCOORD1; float4 Tint:COLOR0; float2 UV:TEXCOORD2; };
CaptureVarying VSCapture(CaptureInput input)
{
    CaptureVarying output; float4 worldPoint=mul(ObjectToRelativeWorld,float4(input.Position,1));
    // Separate transforms keep the near-plane subtraction out of the depth scale.
    output.Raster=mul(RelativeWorldToClip,worldPoint); output.Position=worldPoint.xyz/max(abs(worldPoint.w),HomogeneousEpsilon);
    output.Normal=mul((float3x3)NormalToWorld,input.Normal); output.Tint=float4(DecodeVertex(input.Color.rgb),input.Color.a);
    output.UV=float2(input.UV.x,1-input.UV.y); return output;
}
float3 CaptureNormal(CaptureVarying surface,bool frontFacing)
{
    float3 n=NormalizeMaterial(surface.Normal,float3(0,0,1)); n=frontFacing?n:-n;
    float3 horizontal=ddx(surface.Position),vertical=ddy(surface.Position); float2 uvX=ddx(surface.UV),uvY=ddy(surface.UV);
    float area=uvX.x*uvY.y-uvX.y*uvY.x;
    if (Images.y==0 || abs(area)<UvEpsilon) return n;
    float3 t=(horizontal*uvY.y-vertical*uvX.y)/area,b=(vertical*uvX.x-horizontal*uvY.x)/area;
    t-=n*dot(n,t); if (dot(t,t)<=NormalEpsilon) return n; t=normalize(t);
    float3 perpendicular=cross(n,t); if (dot(perpendicular,b)<0) perpendicular=-perpendicular;
    float3 encoded=NormalMap.Sample(SurfaceSampler,surface.UV).xyz*2-1;
    return NormalizeMaterial(t*encoded.x+perpendicular*encoded.y+n*encoded.z,n);
}
struct MaterialTargets { float4 Base:SV_TARGET0; float4 Normal:SV_TARGET1; float4 Emission:SV_TARGET2; };
MaterialTargets PSCapture(CaptureVarying input,bool front:SV_IsFrontFace)
{
    float3 normal=CaptureNormal(input,front);
    float4 texel=Images.x!=0?DiffuseMap.Sample(SurfaceSampler,input.UV):float4(1,1,1,1);
    float mask=Images.z!=0?OpacityMap.Sample(SurfaceSampler,input.UV).r:1;
    float coverage=saturate(input.Tint.a*ReflectanceCoverage.a*texel.a*mask); if (coverage<MaterialValues.y) discard;
    MaterialTargets result; result.Base=float4(saturate(input.Tint.rgb*ReflectanceCoverage.rgb*texel.rgb),MaterialValues.x);
    result.Normal=float4(normal,saturate(input.Raster.z)); result.Emission=EmissionAndRoughness; return result;
}
#endif
