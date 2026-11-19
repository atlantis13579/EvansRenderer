#version 310 es
precision highp float;
precision highp int;
const uint LumenScreenSurface=3u,LumenWorldCache=4u,LumenHairPointerShift=2u;
vec3 lumenEnvironmentRadiance(vec3 baseSky,vec4 axisPower,vec3 peak,vec3 direction)
{
    if (!any(greaterThan(peak,vec3(0.0)))) return baseSky;
    float cosine=clamp(dot(axisPower.xyz,direction),0.0,1.0);
    return baseSky+peak*pow(cosine,axisPower.w);
}
const uint LumenProbeNeighbors=4u,LumenLinkWordBits=32u;
uint probeLinkWords(uint slots) { return (LumenProbeNeighbors*slots+LumenLinkWordBits-1u)/LumenLinkWordBits; }
vec3 probeSphereSample(uint index,uint count,uint sourceId,uint sequence)
{
    const float Pi=3.14159265358979323846,FixedPointScale=1.0/4294967296.0;
    uint phase=bitfieldReverse(index)+bitfieldReverse(sourceId^sequence);
    float z=1.0-2.0*(float(index)+0.5)/float(count),angle=2.0*Pi*(float(phase)*FixedPointScale);
    float radius=sqrt(max(0.0,1.0-z*z)); return vec3(cos(angle)*radius,sin(angle)*radius,z);
}
// ES stores framebuffer rows bottom-up. Compute readers convert addresses to
// the canonical top-left atlas convention; material UVs retain the OBJ conversion.
// A fixed square atlas uses Clarberg's equal-area sphere parameterization.
// Angular rows are canonical; they are not ES framebuffer texture addresses.
vec3 worldProbeAtlasDirection(uint ray,uint size)
{
    const float AtlasPi=3.14159265358979323846;
    size=max(size,1u);
    vec2 p=2.0*(vec2(ray%size,ray/size)+0.5)/float(size)-1.0;
    float diagonal=1.0-abs(p.x)-abs(p.y),radius=1.0-abs(diagonal);
    float angle=radius>0.0?(AtlasPi/4.0)*((abs(p.y)-abs(p.x))/radius+1.0):0.0;
    float radial=radius*sqrt(max(0.0,2.0-radius*radius)),z=1.0-radius*radius;
    return vec3(radial*abs(cos(angle))*(p.x<0.0?-1.0:1.0),radial*abs(sin(angle))*(p.y<0.0?-1.0:1.0),diagonal<0.0?-z:z);
}
#if defined(EVANS_LUMEN_REFLECTION_RESOLVE_STAGE)
// The ES reconstruction keeps storage-buffer rows canonical and flips only
// framebuffer texture addresses. Raw vector offsets follow the native ray ABI.
// VNDF normalization includes below-horizon and unavailable-transport samples.
layout(std140,row_major,binding=0) uniform SpecularResolveFrame
{
    mat4 ReconstructionInverse; uvec4 ReconstructionViewport,ReconstructionTraceWork;
    vec4 ReconstructionTrace,ReconstructionNumeric,ReconstructionAcceptance,ReconstructionFilter,ReconstructionSky; uvec4 ReconstructionDispatch; vec4 EnvironmentAxis,EnvironmentPeak;
};
uniform highp sampler2D ReconstructionDepth,ReconstructionNormal,ReconstructionDiffuse,ReconstructionMetal,ReconstructionSpecular;
layout(std430,binding=0) readonly buffer ReflectionPaths { uvec4 Data[]; } Paths;
layout(std430,binding=1) readonly buffer ReflectionLighting { uvec4 Data[]; } Lighting;
layout(std430,binding=2) writeonly buffer ReflectionEstimate { vec4 Data[]; } Estimate;
const uint ReconstructionRayVectors=2u,ReconstructionSampleVectors=7u,ReconstructionHitKind=3u,ReconstructionRadiance=5u,ReconstructionCache=6u;
const uint ReconstructionPBR=3u,ReconstructionMiss=0u,ReconstructionHit=1u,ReconstructionScreenHit=3u,ReconstructionKnown=1u,ReconstructionPartial=2u;
const float ReconstructionPartialConfidence=0.5;
bool reconstructionFinite(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
ivec2 reconstructionAddress(uvec2 pixel) { return ivec2(pixel.x,ReconstructionViewport.y-1u-pixel.y); }
bool reconstructionWorld(uvec2 pixel,float depth,out vec3 p)
{
    vec2 uv=(vec2(pixel)+0.5)/vec2(ReconstructionViewport.xy);
    vec4 h=ReconstructionInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1); p=vec3(0);
    if (!reconstructionFinite(h) || abs(h.w)<=ReconstructionNumeric.y) return false;
    p=h.xyz/h.w; return reconstructionFinite(vec4(p,0)) && all(lessThanEqual(abs(p),vec3(ReconstructionNumeric.x)));
}
bool reconstructionMaterial(uvec2 pixel,out vec3 p,out vec3 n,out float roughness)
{
    ivec2 address=reconstructionAddress(pixel); float depth=texelFetch(ReconstructionDepth,address,0).x;
    vec4 material=texelFetch(ReconstructionNormal,address,0); roughness=material.w; p=n=vec3(0);
    if (!reconstructionFinite(vec4(depth)) || depth<0.0 || depth>=1.0 || !reconstructionFinite(material) || roughness<0.0 || roughness>=ReconstructionTrace.z || dot(material.xyz,material.xyz)<=ReconstructionNumeric.z) return false;
    n=normalize(material.xyz); return reconstructionWorld(pixel,depth,p);
}
float reconstructionSmith(float cosine,float alpha)
{
    float squared=max(cosine*cosine,ReconstructionFilter.y*ReconstructionFilter.y);
    return 0.5*(sqrt(1.0+alpha*alpha*max(0.0,1.0-squared)/squared)-1.0);
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*ReconstructionDispatch.y; if (index>=ReconstructionDispatch.z) return;
    Estimate.Data[index]=vec4(0); uvec2 pixel=uvec2(index%ReconstructionViewport.x,index/ReconstructionViewport.x);
    vec3 receiver,normal; float roughness; if (!reconstructionMaterial(pixel,receiver,normal,roughness)) return;
    ivec2 address=reconstructionAddress(pixel); vec4 base=texelFetch(ReconstructionDiffuse,address,0),metal=texelFetch(ReconstructionMetal,address,0),specular=texelFetch(ReconstructionSpecular,address,0);
    if (!reconstructionFinite(base) || !reconstructionFinite(metal) || !reconstructionFinite(specular)) return;
    vec3 f0=ReconstructionDispatch.x==ReconstructionPBR?mix(vec3(ReconstructionFilter.w),clamp(base.xyz,0.0,1.0),clamp(metal.w,0.0,1.0)):clamp(specular.xyz,0.0,1.0);
    ivec2 center=ivec2(pixel/ReconstructionViewport.z); int radius=int(ReconstructionFilter.x); vec3 total=vec3(0); float divisor=0.0; bool known=true;
    for (int row=-radius;row<=radius;++row) for (int column=-radius;column<=radius;++column)
    {
        ivec2 cell=center+ivec2(column,row); if (any(lessThan(cell,ivec2(0))) || cell.x>=int(ReconstructionViewport.w) || cell.y>=int(ReconstructionTraceWork.x)) continue;
        uint ray=uint(cell.y)*ReconstructionViewport.w+uint(cell.x); vec4 direction=uintBitsToFloat(Paths.Data[ray*ReconstructionRayVectors+1u]);
        if (!reconstructionFinite(direction) || dot(direction.xyz,direction.xyz)<=ReconstructionNumeric.z) continue;
        uvec2 anchor=min(uvec2(cell)*ReconstructionViewport.z+ReconstructionViewport.z/2u,ReconstructionViewport.xy-1u);
        vec3 p,n; float r; if (!reconstructionMaterial(anchor,p,n,r) || dot(n,normal)<ReconstructionAcceptance.x || abs(r-roughness)>ReconstructionAcceptance.w) continue;
        vec3 displacement=p-receiver; float plane=max(abs(dot(displacement,n)),abs(dot(displacement,normal)));
        if (isnan(plane) || isinf(plane) || plane>ReconstructionAcceptance.y+ReconstructionAcceptance.z*length(displacement)) continue;
        vec2 offset=(vec2(anchor)-vec2(pixel))/float(ReconstructionViewport.z); float weight=1.0/(1.0+dot(offset,offset)); divisor+=weight;
        if (direction.w<=0.0) continue;
        vec3 nearPosition,midPosition;
        if (!reconstructionWorld(anchor,0.0,nearPosition) || !reconstructionWorld(anchor,0.5,midPosition)) { known=false; continue; }
        vec3 eye=normalize(nearPosition-midPosition); float noV=dot(n,eye),noL=dot(n,direction.xyz);
        if (!reconstructionFinite(vec4(eye,0)) || noV<=0.0 || noL<=0.0) { known=false; continue; }
        uint start=ray*ReconstructionSampleVectors,kind=Lighting.Data[start+ReconstructionHitKind].z,status=Lighting.Data[start+ReconstructionCache].z;
        vec3 incoming=vec3(0);
        if (kind==ReconstructionMiss) incoming=lumenEnvironmentRadiance(ReconstructionSky.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction.xyz);
        else if ((kind==ReconstructionHit || kind==ReconstructionScreenHit) && (status==ReconstructionKnown || status==ReconstructionPartial))
        { incoming=uintBitsToFloat(Lighting.Data[start+ReconstructionRadiance]).xyz; known=known && status==ReconstructionKnown; }
        else known=false;
        vec4 medium=uintBitsToFloat(Lighting.Data[start+4u]); incoming=incoming*medium.w+medium.xyz;
        if (!reconstructionFinite(vec4(incoming,0)) || any(lessThan(incoming,vec3(0)))) { known=false; continue; }
        float alpha=max(r*r,ReconstructionTrace.w),viewLambda=reconstructionSmith(noV,alpha),lightLambda=reconstructionSmith(noL,alpha);
        float smith=r==0.0?1.0:(1.0+viewLambda)/(1.0+viewLambda+lightLambda);
        float complement=1.0-clamp(dot(eye,normalize(eye+direction.xyz)),0.0,1.0),square=complement*complement;
        total+=incoming*(f0+(1.0-f0)*square*square*complement)*smith*weight;
    }
    if (divisor>0.0 && reconstructionFinite(vec4(total,0))) Estimate.Data[index]=vec4(clamp(total/divisor,0.0,ReconstructionFilter.z),known?1.0:ReconstructionPartialConfidence);
}
#elif defined(EVANS_LUMEN_REFLECTION_RAYS_STAGE)
// ES owns its GGX ray generation (Heitz 2018). Material fetches flip Y;
// reflected rays retain canonical row order in two raw vectors per pixel.
layout(std140,row_major,binding=0) uniform SpecularRayFrame
{ mat4 SpecularInverse; uvec4 SpecularViewport,SpecularWork; vec4 SpecularTrace,SpecularNumeric; };
uniform highp sampler2D SpecularDepth,SpecularNormal;
layout(std430,binding=0) writeonly buffer SpecularQueue { uvec4 Data[]; } Output;
const uint SpecularRayVectors=2u,SpecularSamplePeriod=64u;
const float SpecularPi=3.14159265358979323846,SpecularIntegerFraction=1.0/4294967296.0;
const float SpecularRotation=0.6180339887498948482,SpecularTangentLimit=0.9;
bool specularFinite(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
bool specularReconstruct(uvec2 pixel,float depth,out vec3 p)
{
    vec2 uv=(vec2(pixel)+0.5)/vec2(SpecularViewport.xy);
    vec4 h=SpecularInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1); p=vec3(0);
    if (!specularFinite(h) || abs(h.w)<=SpecularNumeric.y) return false;
    p=h.xyz/h.w; return specularFinite(vec4(p,0)) && all(lessThanEqual(abs(p),vec3(SpecularNumeric.x)));
}
vec3 specularMicroNormal(vec3 view,float alpha,vec2 xi)
{
    vec3 stretched=normalize(vec3(alpha*view.xy,view.z)); float lengthSquared=dot(stretched.xy,stretched.xy);
    vec3 right=lengthSquared>0.0?vec3(-stretched.y,stretched.x,0)*inversesqrt(lengthSquared):vec3(1,0,0),up=cross(stretched,right);
    float angle=2.0*SpecularPi*xi.y; vec2 disk=sqrt(xi.x)*vec2(cos(angle),sin(angle));
    float blend=(1.0+stretched.z)*0.5; disk.y=(1.0-blend)*sqrt(max(0.0,1.0-disk.x*disk.x))+blend*disk.y;
    vec3 normal=disk.x*right+disk.y*up+sqrt(max(0.0,1.0-dot(disk,disk)))*stretched;
    return normalize(vec3(alpha*normal.xy,max(0.0,normal.z)));
}
layout(local_size_x=64) in;
void main()
{
    uint ray=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*SpecularWork.w; if (ray>=SpecularWork.y) return;
    uint at=ray*SpecularRayVectors; Output.Data[at]=uvec4(0u); Output.Data[at+1u]=uvec4(0u);
    uvec2 pixel=min(uvec2(ray%SpecularViewport.w,ray/SpecularViewport.w)*SpecularViewport.z+SpecularViewport.z/2u,SpecularViewport.xy-1u);
    ivec2 native=ivec2(pixel.x,SpecularViewport.y-1u-pixel.y); float depth=texelFetch(SpecularDepth,native,0).x;
    vec4 material=texelFetch(SpecularNormal,native,0);
    if (!specularFinite(vec4(depth)) || depth<0.0 || depth>=1.0 || !specularFinite(material) || material.w<0.0 || material.w>=SpecularTrace.z || dot(material.xyz,material.xyz)<=SpecularNumeric.z) return;
    vec3 position,nearPosition,midPosition;
    if (!specularReconstruct(pixel,depth,position) || !specularReconstruct(pixel,0.0,nearPosition) || !specularReconstruct(pixel,0.5,midPosition)) return;
    vec3 normal=normalize(material.xyz),view=normalize(nearPosition-midPosition); if (!specularFinite(vec4(view,0)) || dot(view,normal)<=0.0) return;
    vec3 axis=abs(normal.z)<SpecularTangentLimit?vec3(0,0,1):vec3(0,1,0),tangent=normalize(cross(axis,normal)),bitangent=cross(normal,tangent);
    float phase=float(bitfieldReverse(ray))*SpecularIntegerFraction;
    vec2 xi=fract(vec2((float(SpecularWork.z%SpecularSamplePeriod)+0.5)/float(SpecularSamplePeriod),float(bitfieldReverse(SpecularWork.z))*SpecularIntegerFraction)+vec2(phase,fract(phase*SpecularRotation)));
    vec3 local=material.w==0.0?vec3(0,0,1):specularMicroNormal(vec3(dot(view,tangent),dot(view,bitangent),dot(view,normal)),max(material.w*material.w,SpecularTrace.w),xi);
    vec3 halfVector=local.x*tangent+local.y*bitangent+local.z*normal,reflected=reflect(-view,halfVector);
    Output.Data[at]=floatBitsToUint(vec4(position+normal*SpecularTrace.y,0));
    Output.Data[at+1u]=floatBitsToUint(vec4(reflected,dot(reflected,normal)>0.0?SpecularTrace.x:0.0));
}
#elif defined(EVANS_LUMEN_TEMPORAL_STAGE) || defined(EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE)
// Temporal buffers use canonical top-left rows. Only texture accesses convert
// to ES framebuffer coordinates; motion already describes top-left pixels.
layout(std140,row_major,binding=0) uniform AccumulationFrame
{
    mat4 AccumulationInverse,AccumulationPreviousInverse; vec4 AccumulationPlane; uvec4 AccumulationExtent;
    vec4 AccumulationRules,AccumulationClip,AccumulationJitter,AccumulationNumeric;
};
uniform highp sampler2D TemporalCurrent;
layout(std430,binding=0) readonly buffer CommittedHistory { uvec4 Pixels[]; } Committed;
const float AccumulationNormalFloor=1e-12;
const int AccumulationFilterRadius=1;
vec4 readIrradiance(uvec4 encoded) { return vec4(unpackHalf2x16(encoded.x),unpackHalf2x16(encoded.y)); }
ivec2 temporalAddress(uvec2 canonical) { return ivec2(canonical.x,AccumulationExtent.y-1u-canonical.y); }
#if defined(EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE)
#ifdef EVANS_VERTEX_STAGE
void main()
{ vec2 corner=vec2((gl_VertexID<<1)&2,gl_VertexID&2); gl_Position=vec4(corner*2.0-1.0,0.0,1.0); }
#else
layout(location=0) out vec4 Color;
void main()
{
    ivec2 nativePixel=ivec2(gl_FragCoord.xy); uvec2 pixel=uvec2(uint(nativePixel.x),AccumulationExtent.y-1u-uint(nativePixel.y));
    vec4 scene=texelFetch(TemporalCurrent,nativePixel,0),indirect=readIrradiance(Committed.Pixels[pixel.y*AccumulationExtent.x+pixel.x]);
    vec3 contribution=indirect.w>0.0?indirect.xyz*AccumulationJitter.z:vec3(0.0);
    Color=vec4(clamp(scene.xyz+contribution,0.0,AccumulationNumeric.w),scene.w);
}
#endif
#else
uniform highp sampler2D TemporalDepth,TemporalNormals,TemporalMotion;
layout(std430,binding=1) writeonly buffer PendingHistory { uvec4 Pixels[]; } Pending;
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
layout(std430,binding=2) readonly buffer CurrentSpecular { vec4 Pixels[]; } Specular;
#endif
vec4 accumulationEstimate(uvec2 pixel)
{
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
    return Specular.Pixels[pixel.y*AccumulationExtent.x+pixel.x];
#else
    return texelFetch(TemporalCurrent,temporalAddress(pixel),0);
#endif
}
bool accumulationFinite(vec4 v) { return !(any(isnan(v)) || any(isinf(v))); }
bool accumulationScalar(float v) { return !(isnan(v) || isinf(v)); }
vec2 octahedralSign(vec2 v) { return vec2(v.x>=0.0?1.0:-1.0,v.y>=0.0?1.0:-1.0); }
vec3 readOrientation(uint packed)
{
    vec2 p=unpackSnorm2x16(packed); vec3 n=vec3(p,1.0-abs(p.x)-abs(p.y));
    if (n.z<0.0) n.xy=(1.0-abs(n.yx))*octahedralSign(n.xy);
    return normalize(n);
}
uvec4 encodeReceiver(vec4 radiance,vec3 normal,float depth)
{
    vec2 p=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));
    if (normal.z<0.0) p=(1.0-abs(p.yx))*octahedralSign(p);
    return uvec4(packHalf2x16(radiance.xy),packHalf2x16(radiance.zw),packSnorm2x16(p),floatBitsToUint(depth));
}
bool currentSurface(uvec2 pixel,out vec3 world,out vec3 normal,out float depth)
{
    ivec2 address=temporalAddress(pixel); depth=texelFetch(TemporalDepth,address,0).r; vec4 material=texelFetch(TemporalNormals,address,0);
    world=vec3(0.0); normal=vec3(0,0,1);
    if (!accumulationScalar(depth) || depth<0.0 || depth>=1.0 || !accumulationFinite(material) || material.w<0.0 || dot(material.xyz,material.xyz)<=AccumulationNormalFloor) return false;
    vec2 uv=(vec2(pixel)+0.5)/vec2(AccumulationExtent.xy); vec4 h=AccumulationInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    if (!accumulationFinite(h) || abs(h.w)<=AccumulationNumeric.z) return false;
    world=h.xyz/h.w; normal=normalize(material.xyz); return accumulationFinite(vec4(world,0.0));
}
float opticalDistance(vec2 pixel,float depth)
{
    vec2 uv=pixel/vec2(AccumulationExtent.xy); vec4 h=AccumulationPreviousInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    if (!accumulationFinite(h) || abs(h.w)<=AccumulationNumeric.z) return -1.0;
    return dot(AccumulationPlane,vec4(h.xyz/h.w,1.0));
}
// LumenTemporalThreads is the host dispatch width.
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*AccumulationExtent.w; if (index>=AccumulationExtent.z) return;
    uvec2 pixel=uvec2(index%AccumulationExtent.x,index/AccumulationExtent.x); vec3 world,normal; float depth;
    vec4 estimate=accumulationEstimate(pixel);
    if (!currentSurface(pixel,world,normal,depth) || !accumulationFinite(estimate) || estimate.w<=0.0)
    { Pending.Pixels[index]=encodeReceiver(vec4(0.0),vec3(0,0,1),1.0); return; }
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
    if (estimate.w<1.0) { Pending.Pixels[index]=encodeReceiver(vec4(clamp(estimate.xyz,0.0,AccumulationNumeric.w),estimate.w),normal,depth); return; }
#endif
    estimate=vec4(clamp(estimate.xyz,0.0,AccumulationNumeric.w),1.0); Pending.Pixels[index]=encodeReceiver(estimate,normal,depth);
    if (AccumulationJitter.w==0.0 || AccumulationRules.x<=0.0) return;
    vec4 motion=texelFetch(TemporalMotion,temporalAddress(pixel),0); if (!accumulationFinite(motion) || motion.w<=0.0) return;
    vec2 previous=vec2(pixel)+0.5-motion.xy-AccumulationJitter.xy;
    if (any(lessThan(previous,vec2(0.5))) || any(greaterThan(previous,vec2(AccumulationExtent.xy)-0.5))) return;
    float expected=opticalDistance(previous,1.0-motion.z); if (!accumulationScalar(expected) || expected<0.0) return;
    ivec2 base=ivec2(floor(previous-0.5)); vec2 blend=fract(previous-0.5); vec4 sum=vec4(0.0); float weights=0.0;
    for (uint tap=0u;tap<4u;++tap)
    {
        ivec2 offset=ivec2(tap&1u,tap>>1u),candidate=base+offset;
        uvec2 safePixel=uvec2(clamp(candidate,ivec2(0),ivec2(AccumulationExtent.xy)-1));
        uvec4 packed=Committed.Pixels[safePixel.y*AccumulationExtent.x+safePixel.x]; vec4 old=readIrradiance(packed); float oldDepth=uintBitsToFloat(packed.w);
        float weight=(offset.x!=0?blend.x:1.0-blend.x)*(offset.y!=0?blend.y:1.0-blend.y);
        if (weight<=0.0 || any(lessThan(candidate,ivec2(0))) || any(greaterThanEqual(candidate,ivec2(AccumulationExtent.xy))) || !accumulationFinite(old) ||
            old.w<1.0 || old.w>AccumulationClip.y || !accumulationScalar(oldDepth) || oldDepth<0.0 || oldDepth>=1.0 || dot(normal,readOrientation(packed.z))<AccumulationRules.z) continue;
        float optical=opticalDistance(vec2(candidate)+0.5,oldDepth);
        if (!accumulationScalar(optical) || abs(optical-expected)>max(AccumulationNumeric.x,abs(expected)*AccumulationRules.y)) continue;
        sum+=old*weight; weights+=weight;
    }
    if (weights<=AccumulationNormalFloor) return;
    vec4 history=sum/weights; vec3 low=estimate.xyz,high=low;
    for (int y=-AccumulationFilterRadius;y<=AccumulationFilterRadius;++y) for (int x=-AccumulationFilterRadius;x<=AccumulationFilterRadius;++x)
    {
        uvec2 neighbor=uvec2(clamp(ivec2(pixel)+ivec2(x,y),ivec2(0),ivec2(AccumulationExtent.xy)-1));
        vec3 otherPosition,otherNormal; float otherDepth; vec4 value=accumulationEstimate(neighbor);
        if (!accumulationFinite(value) || value.w<=0.0 || !currentSurface(neighbor,otherPosition,otherNormal,otherDepth) || dot(normal,otherNormal)<AccumulationRules.z) continue;
        float planeError=abs(dot(otherPosition-world,normal)); if (!accumulationScalar(planeError) || planeError>AccumulationRules.w) continue;
        low=min(low,value.xyz); high=max(high,value.xyz);
    }
    vec3 middle=(low+high)*0.5,radius=max((high-low)*0.5+abs(middle)*AccumulationClip.x,vec3(AccumulationNumeric.y));
    vec3 difference=history.xyz-middle,ratio=abs(difference/radius); history.xyz=middle+difference/max(1.0,max(ratio.x,max(ratio.y,ratio.z)));
    float count=min(history.w+1.0,AccumulationClip.y),historyWeight=min(AccumulationRules.x,(count-1.0)/count);
    Pending.Pixels[index]=encodeReceiver(vec4(clamp(mix(estimate.xyz,history.xyz,historyWeight),0.0,AccumulationNumeric.w),count),normal,depth);
}
#endif
#elif defined(EVANS_LUMEN_PROBE_GATHER_STAGE)
#ifdef EVANS_VERTEX_STAGE
void main()
{ vec2 corner=vec2((gl_VertexID<<1)&2,gl_VertexID&2); gl_Position=vec4(corner*2.0-1.0,0.0,1.0); }
#else
layout(std140,row_major,binding=0) uniform GatherFrame
{ uvec4 GatherCounts,GatherOffsets,GatherWork; vec4 GatherTraceOptions,GatherHairSky; mat4 GatherInverse; uvec4 GatherViewport,GatherGrid; vec4 GatherOrigin,GatherAdaptive,GatherSettings,GatherNumeric; uvec4 GatherModel; vec4 GatherVisibility; };
uniform highp sampler2D GatherScene,GatherDepth,GatherNormal,GatherDiffuse,GatherAmbient;
layout(std430,binding=0) readonly buffer GatherOrigins { uvec4 Data[]; } Positions;
layout(std430,binding=1) readonly buffer GatherIrradiance { uvec4 Data[]; } Irradiance;
layout(std430,binding=2) readonly buffer GatherLinks { uint Masks[]; } Links;
layout(location=0) out vec4 Color;
// Raw vector loads preserve integer tags and avoid ANGLE whole-structure loads.
const uint OriginStride=2u,ProbeStride=13u,ProbeTag=12u,HarmonicCount=9u,PartialProbe=2u,PBRModel=3u;
const float Pi=3.14159265358979323846,CoordinatePrecision=32.0*1.1920928955078125e-7;
bool gatherFinite(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
float gatherMagnitude(vec3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
ivec2 gatherAddress(uvec2 canonical) { return ivec2(canonical.x,GatherViewport.y-1u-canonical.y); }
float irradianceBasis(uint term,vec3 normal)
{
    switch (term)
    {
        case 0u:return sqrt(1.0/(4.0*Pi));
        case 1u:return normal.y*sqrt(3.0/(4.0*Pi));
        case 2u:return normal.z*sqrt(3.0/(4.0*Pi));
        case 3u:return normal.x*sqrt(3.0/(4.0*Pi));
        case 4u:return normal.x*normal.y*sqrt(15.0/(4.0*Pi));
        case 5u:return normal.y*normal.z*sqrt(15.0/(4.0*Pi));
        case 6u:return (3.0*normal.z*normal.z-1.0)*sqrt(5.0/(16.0*Pi));
        case 7u:return normal.x*normal.z*sqrt(15.0/(4.0*Pi));
        case 8u:return (normal.x*normal.x-normal.y*normal.y)*sqrt(15.0/(16.0*Pi));
    }
    return 0.0;
}
float gatherCoordinate(float center,uint extent,uint count)
{
    float side=float(GatherViewport.z),coordinate=center/side-0.5;
    if (count>1u && extent%GatherViewport.z!=0u)
    {
        float before=(float(count)-1.5)*side,last=(float(count-1u)*side+float(extent))*0.5;
        if (center>before) coordinate=float(count-2u)+(center-before)/(last-before);
    }
    return clamp(coordinate,0.0,float(count-1u));
}
void main()
{
    ivec2 nativePixel=ivec2(gl_FragCoord.xy); vec2 center=vec2(gl_FragCoord.x,float(GatherViewport.y)-gl_FragCoord.y);
    vec4 scene=GatherModel.z!=0u?vec4(0.0):texelFetch(GatherScene,nativePixel,0); Color=scene;
    float depth=texelFetch(GatherDepth,nativePixel,0).r; vec4 surface=texelFetch(GatherNormal,nativePixel,0);
    if (GatherSettings.x==0.0 || isnan(depth) || isinf(depth) || depth<0.0 || depth>=1.0 || surface.w<0.0 || !gatherFinite(surface) || dot(surface.xyz,surface.xyz)<=GatherNumeric.x) return;
    vec2 uv=center/vec2(GatherViewport.xy); vec4 h=GatherInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    if (!gatherFinite(h) || abs(h.w)<=GatherOrigin.w) return;
    vec3 world=h.xyz/h.w,normal=normalize(surface.xyz);
    if (!gatherFinite(vec4(world,0.0)) || any(greaterThan(abs(world),vec3(GatherOrigin.z)))) return;
    vec2 grid=vec2(gatherCoordinate(center.x,GatherViewport.x,GatherViewport.w),gatherCoordinate(center.y,GatherViewport.y,GatherGrid.x));
    uvec2 base=uvec2(floor(grid)); vec2 fraction=grid-vec2(base); vec3 irradiance=vec3(0.0); float normalization=0.0;
    uint slots=GatherGrid.w+1u,words=probeLinkWords(GatherGrid.w+1u);
    for (uint tap=0u;tap<4u*slots;++tap)
    {
        uint cornerIndex=tap/slots;
        uvec2 corner=uvec2(cornerIndex&1u,cornerIndex>>1u),tile=min(base+corner,uvec2(GatherViewport.w,GatherGrid.x)-1u);
        float weight=(corner.x!=0u?fraction.x:1.0-fraction.x)*(corner.y!=0u?fraction.y:1.0-fraction.y); if (weight<=0.0) continue;
        if (GatherModel.y!=0u && (Links.Masks[(uint(center.y)*GatherViewport.x+uint(center.x))*words+tap/LumenLinkWordBits]&(1u<<(tap%LumenLinkWordBits)))==0u) continue;
        uint probe=(tile.y*GatherViewport.w+tile.x)*slots+tap%slots; uvec4 location=Positions.Data[probe*OriginStride+1u],tag=Irradiance.Data[probe*ProbeStride+ProbeTag];
        if (location.x!=probe || location.y!=0u || any(greaterThanEqual(location.zw,GatherViewport.xy)) || tag.x!=probe || tag.y==0u || tag.y>PartialProbe) continue;
        vec4 sampled=texelFetch(GatherNormal,gatherAddress(location.zw),0);
        if (sampled.w<0.0 || !gatherFinite(sampled) || dot(sampled.xyz,sampled.xyz)<=GatherNumeric.x) continue;
        vec3 n=normalize(sampled.xyz),anchor=uintBitsToFloat(Positions.Data[probe*OriginStride]).xyz-n*GatherOrigin.x;
        float agreement=dot(n,normal); if (agreement<=GatherSettings.y) continue;
        vec3 offset=world-anchor; float scale=gatherMagnitude(offset),distance=scale>0.0?scale*length(offset/scale):0.0;
        float planeError=max(abs(dot(offset,n)),abs(dot(offset,normal)));
        float tolerance=GatherSettings.z+GatherSettings.w*distance+CoordinatePrecision*max(gatherMagnitude(world),gatherMagnitude(anchor));
        if (isnan(planeError) || isinf(planeError) || isnan(tolerance) || isinf(tolerance) || planeError>tolerance) continue;
        float normalWeight=(agreement-GatherSettings.y)/(1.0-GatherSettings.y),planeWeight=tolerance>0.0?clamp(1.0-planeError/tolerance,0.0,1.0):1.0;
        weight*=normalWeight*normalWeight*planeWeight; vec3 value=vec3(0.0);
        for (uint coefficient=0u;coefficient<HarmonicCount;++coefficient) value+=uintBitsToFloat(Irradiance.Data[probe*ProbeStride+coefficient]).xyz*irradianceBasis(coefficient,normal);
        if (!gatherFinite(vec4(value,0.0))) continue;
        irradiance+=max(value,vec3(0.0))*weight; normalization+=weight;
    }
    if (normalization<=GatherNumeric.y || !gatherFinite(vec4(irradiance,0.0))) return;
    vec3 diffuse=clamp(texelFetch(GatherDiffuse,nativePixel,0).xyz,0.0,1.0);
    if (GatherModel.x==PBRModel) diffuse*=(1.0-GatherNumeric.w)*(1.0-clamp(texelFetch(GatherAmbient,nativePixel,0).w,0.0,1.0));
    vec3 indirect=(irradiance/normalization)*(diffuse/Pi);
    if (!gatherFinite(vec4(indirect,0.0))) return;
    if (GatherModel.z!=0u) { Color=vec4(clamp(indirect,0.0,GatherNumeric.z),1.0); return; }
    Color=vec4(clamp(scene.xyz+indirect*GatherSettings.x,0.0,GatherNumeric.z),scene.w);
}
#endif
#elif defined(EVANS_LUMEN_SCREEN_DEPTH_STAGE)
// The hierarchy uses top-left logical cells. Only the first depth fetch flips
// OpenGL's native Y; reductions then share the screen ray's pixel convention.
layout(std140,row_major,binding=0) uniform DepthReduceFrame
{ mat4 DepthInverse; vec4 DepthAnchor,DepthForward; uvec4 DepthSource,DepthTarget,DepthWork; vec4 DepthRules; };
uniform highp sampler2D PyramidDepth;
layout(std430,binding=0) buffer DepthHierarchy { uint Words[]; } Tree;
const uint DepthHeaderWords=4u;
layout(local_size_x=64) in;
void main()
{
    uint linear=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*DepthWork.x;
    if (linear>=DepthTarget.x*DepthTarget.y) return;
    if (linear==0u) for (uint k=0u;k<DepthHeaderWords;++k) Tree.Words[DepthTarget.w*DepthHeaderWords+k]=DepthTarget[k];
    uvec2 p=uvec2(linear%DepthTarget.x,linear/DepthTarget.x); float nearest=DepthRules.y;
    if (DepthTarget.w>0u)
    {
        for (uint row=0u;row<2u;++row) for (uint column=0u;column<2u;++column)
        {
            // Min-reduction tolerates duplicated border samples. Clamp the
            // address itself so backend predication cannot read past an edge.
            uvec2 child=min(p*2u+uvec2(column,row),max(DepthSource.xy,uvec2(1u))-1u);
            nearest=min(nearest,uintBitsToFloat(Tree.Words[DepthSource.z+child.y*DepthSource.x+child.x]));
        }
    }
    else
    {
        ivec2 address=ivec2(p.x,DepthTarget.y-1u-p.y); float depth=texelFetch(PyramidDepth,address,0).r;
        if (!isnan(depth) && !isinf(depth) && depth>=0.0 && depth<1.0)
        {
            vec2 uv=(vec2(p)+0.5)/vec2(DepthTarget.xy);
            vec4 h=DepthInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0); nearest=DepthRules.z;
            if (!any(isnan(h)) && !any(isinf(h)) && abs(h.w)>DepthRules.x)
            {
                vec3 world=h.xyz/h.w;
                float error=DepthAnchor.w*dot(abs(world)+abs(DepthAnchor.xyz),abs(DepthForward.xyz));
                float value=dot(world-DepthAnchor.xyz,DepthForward.xyz)-error;
                if (!isnan(value) && !isinf(value)) nearest=value;
            }
        }
    }
    Tree.Words[DepthTarget.z+linear]=floatBitsToUint(nearest);
}
#elif defined(EVANS_LUMEN_SCREEN_TRACE_STAGE)
// Perspective-correct pixel DDA (McGuire and Mara, JCGT 2014). Buffer pixels
// remain top-left addresses; only texture loads convert to native ES rows.
layout(std140,row_major,binding=0) uniform HybridTraceFrame
{ mat4 HybridCamera,HybridInverse; uvec4 HybridWork,HybridExtent; vec4 HybridAnchor,HybridForward,HybridLimits; };
uniform highp sampler2D HybridDepth,HybridNormals,HybridColor;
struct HybridOrigin { vec4 PositionMinimum; uvec4 Metadata; };
struct HybridHit { vec4 PositionDistance,NormalResidual,Uncertainty; uvec4 Metadata; vec4 Medium; };
struct HybridLight { HybridHit Hit; vec4 RadianceCoverage; uvec4 Cache; };
layout(std430,binding=0) readonly buffer HybridSeeds { uvec4 Values[]; } Seeds;
layout(std430,binding=1) writeonly buffer HybridIntersections { uvec4 Values[]; } Intersections;
layout(std430,binding=2) writeonly buffer HybridLighting { uvec4 Values[]; } Lighting;
layout(std430,binding=3) readonly buffer HybridHierarchy { uint Words[]; } Hierarchy;
const uint HybridSurface=3u,HybridKnown=1u,HybridNoCard=0xffffffffu;
const uint HybridOriginVectors=2u,HybridHitVectors=5u,HybridLightVectors=7u;
const float HybridNormalFloor=1e-12,HybridParallelFloor=1e-7,HybridClear=1.0,HybridPixelCenter=0.5;
bool hybridFinite(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
bool hybridClipPlane(float start,float finish,inout float lower,inout float upper)
{
    if (start<0.0 && finish<0.0) return false;
    if (start<0.0) lower=max(lower,start/(start-finish));
    else if (finish<0.0) upper=min(upper,start/(start-finish));
    return lower<=upper;
}
bool hybridScreenHit(HybridOrigin seed,vec3 direction,out HybridHit hit,out vec3 light)
{
    hit=HybridHit(vec4(0.0),vec4(0.0),vec4(0.0),uvec4(0u),vec4(0,0,0,1)); light=vec3(0.0);
    vec3 first=seed.PositionMinimum.xyz+direction*seed.PositionMinimum.w,last=seed.PositionMinimum.xyz+direction*HybridLimits.x;
    vec4 a=HybridCamera*vec4(first,1.0),b=HybridCamera*vec4(last,1.0); if (!hybridFinite(a) || !hybridFinite(b)) return false;
    float enter=0.0,leave=1.0;
    if (!hybridClipPlane(a.w+a.x,b.w+b.x,enter,leave) || !hybridClipPlane(a.w-a.x,b.w-b.x,enter,leave) ||
        !hybridClipPlane(a.w+a.y,b.w+b.y,enter,leave) || !hybridClipPlane(a.w-a.y,b.w-b.y,enter,leave) ||
        !hybridClipPlane(a.z,b.z,enter,leave) || !hybridClipPlane(a.w-a.z,b.w-b.z,enter,leave) ||
        !hybridClipPlane(a.w-HybridLimits.w,b.w-HybridLimits.w,enter,leave)) return false;
    vec4 h0=mix(a,b,enter),h1=mix(a,b,leave); float inv0=1.0/h0.w,inv1=1.0/h1.w;
    vec3 q0=mix(first,last,enter)*inv0,q1=mix(first,last,leave)*inv1;
    vec2 p0=(h0.xy*inv0*vec2(0.5,-0.5)+0.5)*vec2(HybridExtent.xy),p1=(h1.xy*inv1*vec2(0.5,-0.5)+0.5)*vec2(HybridExtent.xy);
    vec2 extent=abs(p1-p0); float span=max(extent.x,extent.y); if (span<1.0) return false;
    float increment=HybridLimits.z/span,rate=dot(direction,HybridForward.xyz);
    uint next=0u,level=0u;
    for (uint visit=0u;visit<HybridExtent.z;++visit)
    {
        uint iteration=next++;
        float lower=float(iteration)*increment; if (lower>=1.0) break; float upper=min(lower+increment,1.0);
        ivec2 pixel=ivec2(floor(mix(p0,p1,(lower+upper)*0.5)));
        if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(HybridExtent.xy)))) return false;
        if (all(equal(uvec2(pixel),seed.Metadata.zw))) continue;
        if (HybridExtent.w!=0u)
        {
            uvec2 cell=uvec2(pixel)>>level; vec2 motion=p1-p0;
            vec2 low=vec2(cell<<level),high=min(low+float(1u<<level),vec2(HybridExtent.xy));
            float boundary=1.0;
            if (motion.x!=0.0) boundary=min(boundary,((motion.x>0.0?high.x:low.x)-p0.x)/motion.x);
            if (motion.y!=0.0) boundary=min(boundary,((motion.y>0.0?high.y:low.y)-p0.y)/motion.y);
            uint stop=uint(clamp(ceil(boundary/increment-HybridPixelCenter),float(iteration+1u),ceil(1.0/increment)));
            float lastBegin=float(stop-1u)*increment,lastEnd=min(lastBegin+increment,1.0);
            ivec2 lastPixel=ivec2(floor(mix(p0,p1,(lastBegin+lastEnd)*0.5)));
            if (any(lessThan(lastPixel,ivec2(0))) || any(greaterThanEqual(lastPixel,ivec2(HybridExtent.xy))) || any(notEqual(uvec2(lastPixel)>>level,cell)))
            { stop=iteration+1u; lastEnd=upper; }

            {
                const uint HeaderStride=4u; uint header=level*HeaderStride;
                float nearest=uintBitsToFloat(Hierarchy.Words[Hierarchy.Words[header+2u]+cell.y*Hierarchy.Words[header]+cell.x]);
                vec3 aPoint=mix(q0,q1,lower)/mix(inv0,inv1,lower),bPoint=mix(q0,q1,lastEnd)/mix(inv0,inv1,lastEnd),origin=seed.PositionMinimum.xyz;
                float bound=max(dot(aPoint-origin,direction)*rate,dot(bPoint-origin,direction)*rate)+dot(origin-HybridAnchor.xyz,HybridForward.xyz);
                float margin=HybridAnchor.w*dot(abs(aPoint)+abs(bPoint)+2.0*(abs(origin)+abs(HybridAnchor.xyz)),vec3(1.0));
                if (!isnan(bound+margin) && !isinf(bound+margin) && bound+margin<nearest)
                { next=stop; level=min(level+1u,HybridExtent.w-1u); continue; }
            }
            if (level>0u) { --level; next=iteration; continue; }
        }
        ivec2 address=ivec2(pixel.x,int(HybridExtent.y)-1-pixel.y); float depth=texelFetch(HybridDepth,address,0).r;
        if (isnan(depth) || isinf(depth) || depth<0.0 || depth>=HybridClear) continue;
        vec2 uv=(vec2(pixel)+HybridPixelCenter)/vec2(HybridExtent.xy);
        vec4 h=HybridInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
        if (!hybridFinite(h) || abs(h.w)<=HybridLimits.w) return false;
        vec3 surface=h.xyz/h.w;
        vec3 point0=mix(q0,q1,lower)/mix(inv0,inv1,lower),point1=mix(q0,q1,upper)/mix(inv0,inv1,upper);
        float t0=dot(point0-seed.PositionMinimum.xyz,direction),t1=dot(point1-seed.PositionMinimum.xyz,direction);
        float sceneDepth=dot(surface-seed.PositionMinimum.xyz,HybridForward.xyz),farther=max(t0*rate,t1*rate);
        if (!hybridFinite(vec4(t0,t1,sceneDepth,farther))) return false;
        if (farther<sceneDepth) continue;
        if (min(t0*rate,t1*rate)>sceneDepth+HybridLimits.y) return false;
        vec3 normal=texelFetch(HybridNormals,address,0).xyz;
        if (!hybridFinite(vec4(normal,0.0)) || dot(normal,normal)<=HybridNormalFloor || dot(normal,direction)>=0.0) return false;
        float travel=abs(rate)>HybridParallelFloor?clamp(sceneDepth/rate,min(t0,t1),max(t0,t1)):(t0+t1)*0.5;
        if (travel<=seed.PositionMinimum.w || travel>HybridLimits.x) return false;
        light=texelFetch(HybridColor,address,0).xyz; if (!hybridFinite(vec4(light,0.0))) return false;
        light=max(light,vec3(0.0)); hit.PositionDistance=vec4(surface,travel); hit.NormalResidual=vec4(normalize(normal),0.0);
        hit.Metadata=uvec4(uvec2(pixel),HybridSurface,visit+1u); return true;
    }
    return false;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*HybridWork.w; if (index>=HybridWork.x) return;
    uint origin=(index/HybridWork.y)*HybridOriginVectors;
    HybridOrigin seed=HybridOrigin(uintBitsToFloat(Seeds.Values[origin]),Seeds.Values[origin+1u]); HybridLight answer; answer.Hit.Medium=vec4(0,0,0,1);
    answer.Hit=HybridHit(vec4(0.0),vec4(0.0),vec4(0.0),uvec4(HybridNoCard,HybridNoCard,0u,0u),vec4(0,0,0,1));
    answer.RadianceCoverage=vec4(0.0); answer.Cache=uvec4(HybridNoCard,0u,0u,0u);
    HybridHit hit; hit.Medium=vec4(0,0,0,1); vec3 light;
    vec3 direction=probeSphereSample(index%HybridWork.y,HybridWork.y,seed.Metadata.x,HybridWork.z);
    if (HybridForward.w!=0.0)
    { vec4 reflected=uintBitsToFloat(seed.Metadata); direction=reflected.xyz; seed.Metadata.y=reflected.w>0.0?0u:1u; }
    if (seed.Metadata.y==0u && hybridScreenHit(seed,direction,hit,light))
    { answer.Hit=hit; answer.RadianceCoverage=vec4(light,1.0); answer.Cache.z=HybridKnown; }
    uint hitBase=index*HybridHitVectors,lightBase=index*HybridLightVectors;
    Intersections.Values[hitBase]=floatBitsToUint(answer.Hit.PositionDistance); Intersections.Values[hitBase+1u]=floatBitsToUint(answer.Hit.NormalResidual);
    Intersections.Values[hitBase+2u]=floatBitsToUint(answer.Hit.Uncertainty); Intersections.Values[hitBase+3u]=answer.Hit.Metadata;
    Lighting.Values[lightBase]=floatBitsToUint(answer.Hit.PositionDistance); Lighting.Values[lightBase+1u]=floatBitsToUint(answer.Hit.NormalResidual);
    Lighting.Values[lightBase+2u]=floatBitsToUint(answer.Hit.Uncertainty); Lighting.Values[lightBase+3u]=answer.Hit.Metadata;
    Lighting.Values[lightBase+4u]=floatBitsToUint(answer.Hit.Medium);
    Intersections.Values[hitBase+4u]=floatBitsToUint(answer.Hit.Medium);
    Lighting.Values[lightBase+5u]=floatBitsToUint(answer.RadianceCoverage); Lighting.Values[lightBase+6u]=answer.Cache;
}
#elif defined(EVANS_LUMEN_SCREEN_PROBES_STAGE)
// Tile/subtile searches stay in top-left logical coordinates. ES framebuffer
// storage alone is flipped, including every adaptive normal/depth fetch.
layout(std140,row_major,binding=0) uniform ScreenPlacementFrame
{ mat4 PlacementInverse; uvec4 PlacementViewport,PlacementGrid; vec4 PlacementParameters,PlacementAdaptive; };
uniform highp sampler2D PlacementDepth;
uniform highp sampler2D PlacementNormal;
layout(std430,binding=0) writeonly buffer PlacementBuffer { uvec4 Data[]; } Output;
const uint OriginVectors=2u,MissingPixel=0xffffffffu,Background=1u,InvalidPosition=2u;
const uint PlacementLevels=2u,PlacementChildren=4u;
const float FarClearValue=1.0,NormalLengthThreshold=1e-12;
ivec2 placementAddress(uvec2 pixel) { return ivec2(pixel.x,PlacementViewport.y-1u-pixel.y); }
float placementMagnitude(vec3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
bool placementSurface(uvec2 pixel,out vec3 world,out vec3 normal)
{
    world=normal=vec3(0.0); ivec2 address=placementAddress(pixel); vec2 uv=(vec2(pixel)+0.5)/vec2(PlacementViewport.xy);
    vec4 h=PlacementInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,texelFetch(PlacementDepth,address,0).r,1.0);
    if (any(isnan(h)) || any(isinf(h)) || abs(h.w)<=PlacementParameters.w) return false;
    world=h.xyz/h.w; normal=normalize(texelFetch(PlacementNormal,address,0).xyz);
    return !any(isnan(world)) && !any(isinf(world)) && all(lessThanEqual(abs(world)+PlacementParameters.x,vec3(PlacementParameters.z)));
}
uvec2 placementPixel(uvec2 first,uvec2 last,uint count,vec3 anchors[PlacementLevels],vec3 anchorNormals[PlacementLevels],out bool found)
{
    uvec2 chosen=uvec2(0u); uint best=MissingPixel;
    for (uint row=first.y;row<last.y;++row) for (uint column=first.x;column<last.x;++column)
    {
        uvec2 pixel=uvec2(column,row); ivec2 address=placementAddress(pixel);
        float depth=texelFetch(PlacementDepth,address,0).r; vec4 material=texelFetch(PlacementNormal,address,0);
        if (isnan(depth) || isinf(depth) || depth<0.0 || depth>=FarClearValue || any(isnan(material)) || any(isinf(material)) || material.w<0.0 || dot(material.xyz,material.xyz)<=NormalLengthThreshold) continue;
        if (count!=0u)
        {
            vec3 world,normal; if (!placementSurface(pixel,world,normal)) continue;
            bool covered=false;
            for (uint ancestor=0u;ancestor<count;++ancestor)
            {
                vec3 delta=world-anchors[ancestor]; float scale=placementMagnitude(delta),distance=scale>0.0?scale*length(delta/scale):0.0;
                float plane=max(abs(dot(delta,normal)),abs(dot(delta,anchorNormals[ancestor])));
                float tolerance=PlacementAdaptive.y+PlacementAdaptive.z*distance+PlacementAdaptive.w*max(placementMagnitude(world),placementMagnitude(anchors[ancestor]));
                covered=covered || (dot(normal,anchorNormals[ancestor])>PlacementAdaptive.x && plane<=tolerance);
            }
            if (covered) continue;
        }
        ivec2 offset=ivec2(pixel*2u+1u)-ivec2(first+last); uint metric=uint(offset.x*offset.x+offset.y*offset.y);
        if (metric<best) { best=metric; chosen=pixel; }
    }
    found=best!=MissingPixel; return chosen;
}
void storePlacement(uint index,vec4 origin,uvec4 identity)
{ Output.Data[index*OriginVectors]=floatBitsToUint(origin); Output.Data[index*OriginVectors+1u]=identity; }
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x; if (index>=PlacementGrid.y) return;
    uint slots=PlacementGrid.w+1u,tile=index/slots,child=index%slots;
    uvec2 first=uvec2(tile%PlacementViewport.w,tile/PlacementViewport.w)*PlacementViewport.z;
    uvec2 last=min(first+PlacementViewport.z,PlacementViewport.xy); bool found;
    vec3 anchors[PlacementLevels],anchorNormals[PlacementLevels];
    for (uint clear=0u;clear<PlacementLevels;++clear) { anchors[clear]=vec3(0.0); anchorNormals[clear]=vec3(0.0); }
    uint path[PlacementLevels],depth=0u;
    for (uint node=child;node!=0u;node=(node-1u)/PlacementChildren) path[depth++]=(node-1u)%PlacementChildren;
    uvec2 chosen=placementPixel(first,last,0u,anchors,anchorNormals,found);
    vec4 origin=vec4(0.0); uvec4 identity=uvec4(index,Background,0u,0u); vec3 world=vec3(0.0),normal=vec3(0.0);
    for (uint level=0u;level<depth;++level)
    {
        if (!found || !placementSurface(chosen,world,normal)) { storePlacement(index,origin,identity); return; }
        anchors[level]=world; anchorNormals[level]=normal;
        uvec2 middle=(first+last)/2u; uint quadrant=path[depth-1u-level];
        first=uvec2((quadrant&1u)!=0u?middle.x:first.x,(quadrant&2u)!=0u?middle.y:first.y);
        last=uvec2((quadrant&1u)!=0u?last.x:middle.x,(quadrant&2u)!=0u?last.y:middle.y);
        chosen=placementPixel(first,last,level+1u,anchors,anchorNormals,found);
    }
    if (found)
    {
        identity=uvec4(index,InvalidPosition,chosen);
        if (placementSurface(chosen,world,normal))
        { origin=vec4(world+normal*PlacementParameters.x,PlacementParameters.y); identity.y=0u; }
    }
    storePlacement(index,origin,identity);
}
#elif defined(EVANS_LUMEN_PROBE_INTEGRATE_STAGE)
layout(std140,binding=0) uniform ProbeIntegrationFrame { uvec4 IntegrationCounts; vec4 SkyMeasure,EnvironmentAxis,EnvironmentPeak; };
layout(std430,binding=0) readonly buffer IncidentBuffer { uvec4 Data[]; } Incident;
layout(std430,binding=1) readonly buffer ProbeOriginsBuffer { uvec4 Data[]; } Origins;
layout(std430,binding=2) writeonly buffer IrradianceBuffer { uvec4 Data[]; } Irradiance;
const uint HarmonicCount=9u,ProbeStride=13u,OriginStride=2u,RayStride=7u;
const uint HitMetadataOffset=3u,RadianceOffset=5u,CacheOffset=6u,CoverageOffset=9u,TraceOffset=10u,CacheCountsOffset=11u,IdentityOffset=12u;
const uint Complete=1u,Partial=2u,Invalid=3u,Surface=1u,Miss=0u;
void projectProbe(inout vec3 coefficients[9],vec3 flux,vec3 direction)
{
    const float Pi=3.14159265358979323846;
    float low=Pi*sqrt(1.0/(4.0*Pi)),linear=(2.0*Pi/3.0)*sqrt(3.0/(4.0*Pi));
    float crossTerm=(Pi/4.0)*sqrt(15.0/(4.0*Pi)),polar=(Pi/4.0)*sqrt(5.0/(16.0*Pi)),difference=(Pi/4.0)*sqrt(15.0/(16.0*Pi));
    float x=direction.x,y=direction.y,z=direction.z;
    coefficients[0]+=flux*low; coefficients[1]+=flux*(linear*y); coefficients[2]+=flux*(linear*z); coefficients[3]+=flux*(linear*x);
    coefficients[4]+=flux*(crossTerm*x*y); coefficients[5]+=flux*(crossTerm*y*z); coefficients[6]+=flux*(polar*(3.0*z*z-1.0));
    coefficients[7]+=flux*(crossTerm*x*z); coefficients[8]+=flux*(difference*(x*x-y*y));
}
bool finiteCoefficient(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
layout(local_size_x=64) in;
void main()
{
    uint probe=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*IntegrationCounts.w; if (probe>=IntegrationCounts.x) return;
    uint identifier=Origins.Data[probe*OriginStride+1u].x; vec3 coefficients[9];
    for (uint i=0u;i<HarmonicCount;++i) coefficients[i]=vec3(0.0);
    uvec4 traces=uvec4(0u),caches=uvec4(0u); float coverage=0.0,scaledDistance=0.0; bool invalid=Origins.Data[probe*OriginStride+1u].y!=0u;
    for (uint ray=0u;ray<IntegrationCounts.y;++ray)
    {
        uint offset=(probe*IntegrationCounts.y+ray)*RayStride; uvec4 hit=Incident.Data[offset+HitMetadataOffset],cache=Incident.Data[offset+CacheOffset];
        vec4 material=uintBitsToFloat(Incident.Data[offset+RadianceOffset]); vec3 incoming=vec3(0.0);
        vec3 direction=probeSphereSample(ray,IntegrationCounts.y,identifier,IntegrationCounts.z);
        if (hit.z==Surface || hit.z==LumenScreenSurface || hit.z==LumenWorldCache)
        {
            if (hit.z==LumenWorldCache) ++traces.w;
            else { ++traces.x; scaledDistance+=uintBitsToFloat(Incident.Data[offset]).w/float(IntegrationCounts.y); }
            if (cache.z==Complete) { ++caches.x; incoming=material.xyz; coverage+=material.w; }
            else if (cache.z==Partial) { ++caches.y; incoming=material.xyz; coverage+=material.w; }
            else if (cache.z==Invalid) { ++caches.w; invalid=true; }
            else ++caches.z;
        }
        else if (hit.z==Miss) { ++traces.y; incoming=lumenEnvironmentRadiance(SkyMeasure.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction); }
        else ++traces.z;
        vec4 medium=uintBitsToFloat(Incident.Data[offset+4u]); incoming=incoming*medium.w+medium.xyz;
        projectProbe(coefficients,incoming*SkyMeasure.w,direction);
    }
    vec4 confidence=vec4(float(traces.y+caches.x)/float(IntegrationCounts.y),(traces.x+traces.w)!=0u?coverage/float(traces.x+traces.w):0.0,
        float(traces.y)/float(IntegrationCounts.y),traces.x!=0u?scaledDistance*(float(IntegrationCounts.y)/float(traces.x)):0.0);
    for (uint i=0u;i<HarmonicCount;++i) invalid=invalid || !finiteCoefficient(coefficients[i]);
    invalid=invalid || any(isnan(confidence)) || any(isinf(confidence));
    // Bound the finite average without hiding invalid arithmetic above.
    confidence.y=clamp(confidence.y,0.0,1.0);
    uint status=invalid?Invalid:((traces.z!=0u || caches.y!=0u || caches.z!=0u)?Partial:Complete),outputBase=probe*ProbeStride;
    for (uint i=0u;i<HarmonicCount;++i) Irradiance.Data[outputBase+i]=floatBitsToUint(vec4(invalid?vec3(0.0):coefficients[i],0.0));
    Irradiance.Data[outputBase+CoverageOffset]=floatBitsToUint(invalid?vec4(0.0):confidence);
    Irradiance.Data[outputBase+TraceOffset]=traces; Irradiance.Data[outputBase+CacheCountsOffset]=caches;
    Irradiance.Data[outputBase+IdentityOffset]=uvec4(identifier,status,IntegrationCounts.y,IntegrationCounts.z);
}
#elif defined(EVANS_LUMEN_RADIANCE_RESOLVE_STAGE)
layout(std140,binding=0) uniform RadianceFrame { uvec4 QuerySize,AtlasSize,MapOffsets; vec4 Acceptance,Roundoff; };
uniform highp sampler2D RadianceNormal;
// Raw vectors avoid ANGLE's whole-structure SSBO load issue. Four indexed
// storage buffers are enough for the complete GPU-to-GPU resolve stage.
layout(std430,binding=0) readonly buffer HitInput { uvec4 Vectors[]; } HitData;
layout(std430,binding=1) readonly buffer LightInput { uvec4 Vectors[]; } LightData;
layout(std430,binding=2) readonly buffer MapInput { uvec4 Vectors[]; } MapData;
layout(std430,binding=3) writeonly buffer ResolveOutput { uvec4 Vectors[]; } ResultData;
const uint DirectionCount=6u,HitStride=5u,LightStride=2u,AnswerStride=7u,CardStride=11u,InverseRows=4u,OriginRow=8u,BackwardRow=9u,TileRow=10u,Border=1u;
const uint Missing=0xffffffffu,SurfaceHit=1u,Complete=1u,Partial=2u,Invalid=3u;
bool finiteRadiance(float v) { return !(isnan(v) || isinf(v)); }
bool finiteRadiance3(vec3 v) { return !(any(isnan(v)) || any(isinf(v))); }
vec4 mapFloat(uint index) { return uintBitsToFloat(MapData.Vectors[index]); }
float largestCoordinate(vec3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
vec3 mapPoint(uint base,vec4 point)
{ return vec3(dot(mapFloat(base),point),dot(mapFloat(base+1u),point),dot(mapFloat(base+2u),point)); }
bool gatherRadiance(uint card,vec3 position,vec3 normal,float radius,out vec4 value,out float distanceError,out float facing,out uint status)
{
    value=vec4(0.0); distanceError=0.0; status=0u; uint base=MapOffsets.x+card*CardStride;
    uvec4 tile=MapData.Vectors[base+TileRow]; vec4 backward=mapFloat(base+BackwardRow); facing=dot(normal,backward.xyz);
    if (facing<Acceptance.x) return false;
    vec3 relative=position-mapFloat(base+OriginRow).xyz,projected=mapPoint(base,vec4(relative,1.0));
    if (!finiteRadiance3(projected) || any(greaterThan(abs(projected.xy),vec2(1.0)))) return false;
    vec2 location=vec2(tile.xy)+float(Border)+vec2(projected.x*0.5+0.5,0.5-projected.y*0.5)*float(QuerySize.y)-0.5;
    ivec2 lower=ivec2(floor(location)); vec2 fraction=location-vec2(lower); float coverage=0.0; status=Complete;
    for (uint corner=0u;corner<4u;++corner)
    {
        uvec2 offset=uvec2(corner&1u,corner>>1u); ivec2 pixel=lower+ivec2(offset);
        float weight=(offset.x!=0u?fraction.x:1.0-fraction.x)*(offset.y!=0u?fraction.y:1.0-fraction.y);
        ivec2 nativePixel=ivec2(pixel.x,int(AtlasSize.y)-1-pixel.y); vec4 capturedNormal=texelFetch(RadianceNormal,nativePixel,0);
        if (weight<=0.0 || dot(capturedNormal.xyz,capturedNormal.xyz)==0.0 || dot(capturedNormal.xyz,normal)<Acceptance.y) continue;
        vec2 uv=(vec2(pixel)-vec2(tile.xy)-float(Border)+0.5)/float(QuerySize.y);
        vec3 captured=mapPoint(base+InverseRows,vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,capturedNormal.w,1.0));
        float error=abs(dot(relative-captured,normal));
        float allowance=Roundoff.x*max(backward.w,max(largestCoordinate(relative),largestCoordinate(captured)))+Roundoff.y*dot(abs(position),abs(normal));
        if (!finiteRadiance(error) || error>radius+allowance) continue;
        uint texel=(uint(pixel.y)*AtlasSize.x+uint(pixel.x))*LightStride; vec3 energy=uintBitsToFloat(LightData.Vectors[texel]).xyz;
        uint sourceStatus=LightData.Vectors[texel+1u].x; status=max(status,sourceStatus==0u?Invalid:sourceStatus);
        value.xyz+=energy*weight; distanceError+=error*weight; coverage+=weight;
    }
    if (coverage<=0.0) { status=0u; return false; }
    value=vec4(value.xyz/coverage,clamp(coverage,0.0,1.0)); distanceError/=coverage; return true;
}
void storeRadiance(uint index,vec4 position,vec4 normal,vec4 uncertainty,uvec4 hit,vec4 energy,uvec4 cache)
{
    uint at=index*AnswerStride; ResultData.Vectors[at]=floatBitsToUint(position); ResultData.Vectors[at+1u]=floatBitsToUint(normal);
    ResultData.Vectors[at+2u]=floatBitsToUint(uncertainty); ResultData.Vectors[at+3u]=hit;
    ResultData.Vectors[at+4u]=HitData.Vectors[index*HitStride+4u];
    ResultData.Vectors[at+5u]=floatBitsToUint(energy); ResultData.Vectors[at+6u]=cache;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*AtlasSize.w; if (index>=QuerySize.x) return;
    uint inputBase=index*HitStride; vec4 position=uintBitsToFloat(HitData.Vectors[inputBase]),normal=uintBitsToFloat(HitData.Vectors[inputBase+1u]);
    vec4 uncertainty=uintBitsToFloat(HitData.Vectors[inputBase+2u]); uvec4 hit=HitData.Vectors[inputBase+3u];
    if (hit.z==LumenScreenSurface || hit.z==LumenWorldCache) { ResultData.Vectors[index*AnswerStride+4u]=HitData.Vectors[inputBase+4u]; return; }
    float radius=Acceptance.z+uncertainty.z*Acceptance.w; if (MapOffsets.z==0u) uncertainty.z=0.0;
    vec4 energy=vec4(0.0); uvec4 cache=uvec4(Missing,0u,0u,0u);
    if (hit.z!=SurfaceHit) { storeRadiance(index,position,normal,uncertainty,hit,energy,cache); return; }
    if (!finiteRadiance(radius)) { cache.z=Invalid; storeRadiance(index,position,normal,uncertainty,hit,energy,cache); return; }
    uvec4 range=MapData.Vectors[hit.x]; if (range.y==0u) { storeRadiance(index,position,normal,uncertainty,hit,energy,cache); return; }
    vec4 directions[DirectionCount]; float errors[DirectionCount],cosines[DirectionCount]; uint validity[DirectionCount],selected[DirectionCount];
    for (uint direction=0u;direction<DirectionCount;++direction)
    { directions[direction]=vec4(0.0); errors[direction]=0.0; cosines[direction]=0.0; validity[direction]=0u; selected[direction]=Missing; }
    for (uint candidate=0u;candidate<range.y;++candidate)
    {
        uint word=range.x+candidate,card=MapData.Vectors[word/4u][word%4u],group=MapData.Vectors[MapOffsets.x+card*CardStride+TileRow].z;
        vec4 value; float error,cosine; uint status; if (!gatherRadiance(card,position.xyz,normal.xyz,radius,value,error,cosine,status)) continue;
        if (validity[group]==0u || error<errors[group] || (error==errors[group] && value.w>directions[group].w))
        { directions[group]=value; errors[group]=error; cosines[group]=cosine; validity[group]=status; selected[group]=card; }
    }
    float total=0.0,available=0.0,dominant=0.0; uint state=Complete;
    for (uint direction=0u;direction<DirectionCount;++direction) if (validity[direction]!=0u)
    { float orientation=cosines[direction]*cosines[direction]; total+=orientation*directions[direction].w; available+=orientation; }
    if (total>0.0)
    {
        for (uint direction=0u;direction<DirectionCount;++direction) if (validity[direction]!=0u)
        {
            float weight=cosines[direction]*cosines[direction]*directions[direction].w/total; energy.xyz+=directions[direction].xyz*weight;
            ++cache.y; state=max(state,validity[direction]); if (validity[direction]!=Complete) ++cache.w;
            if (weight>dominant || (weight==dominant && selected[direction]<cache.x)) { dominant=weight; cache.x=selected[direction]; }
        }
        energy.w=clamp(total/available,0.0,1.0); cache.z=state;
        if (state==Invalid || !finiteRadiance3(energy.xyz)) { energy=vec4(0.0); cache.z=Invalid; }
    }
    storeRadiance(index,position,normal,uncertainty,hit,energy,cache);
}
#elif defined(EVANS_LUMEN_LOOKUP_STAGE)
layout(std140,binding=0) uniform LookupFrame { uvec4 Counts,Atlas; vec4 Parameters; };
uniform highp sampler2D LookupAlbedo,LookupNormal,LookupEmission;
// Vector storage keeps integer metadata intact and avoids whole-struct SSBO
// loads on ANGLE. Only image addresses are flipped to native framebuffer rows.
layout(std430,binding=0) readonly buffer LookupCards { uvec4 Values[]; } Cards;
layout(std430,binding=1) readonly buffer LookupQueries { uvec4 Values[]; } Queries;
layout(std430,binding=2) writeonly buffer LookupOutput { uvec4 Values[]; } Output;
const uint DirectionCount=6u,CardVectors=11u,QueryVectors=3u,ResultVectors=4u,Border=1u,MissingCard=0xffffffffu;
const uint InverseOffset=4u,OriginOffset=8u,BackwardOffset=9u,TileOffset=10u;
struct LookupCard { vec4 Projection[4],Inverse[4]; vec4 Origin,BackwardScale; uvec4 Tile; };
struct LookupQuery { vec4 PositionRadius,NormalCosine; uvec4 Range; };
struct LookupValue { vec4 AlbedoMetallic,NormalCoverage,EmissionRoughness; uvec4 Metadata; };
LookupValue emptyValue() { return LookupValue(vec4(0.0),vec4(0.0),vec4(0.0),uvec4(MissingCard,0u,0u,0u)); }
void readCard(uint index,out LookupCard card)
{
    uint base=index*CardVectors;
    for (uint row=0u;row<4u;++row) { card.Projection[row]=uintBitsToFloat(Cards.Values[base+row]); card.Inverse[row]=uintBitsToFloat(Cards.Values[base+InverseOffset+row]); }
    card.Origin=uintBitsToFloat(Cards.Values[base+OriginOffset]); card.BackwardScale=uintBitsToFloat(Cards.Values[base+BackwardOffset]); card.Tile=Cards.Values[base+TileOffset];
}
float maxCoordinate(vec3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
bool sampleCard(LookupCard card,LookupQuery query,out LookupValue value,out float depthError,out float facing)
{
    value=emptyValue(); depthError=0.0; facing=dot(query.NormalCosine.xyz,card.BackwardScale.xyz);
    if (facing<Parameters.x) return false;
    vec3 relative=query.PositionRadius.xyz-card.Origin.xyz; vec4 local=vec4(relative,1.0);
    vec3 clip=vec3(dot(card.Projection[0],local),dot(card.Projection[1],local),dot(card.Projection[2],local));
    if (any(isnan(clip)) || any(isinf(clip)) || any(greaterThan(abs(clip.xy),vec2(1.0)))) return false;
    vec2 location=vec2(card.Tile.xy)+float(Border)+vec2(clip.x+1.0,1.0-clip.y)*0.5*float(Counts.y)-0.5;
    ivec2 lower=ivec2(floor(location)); vec2 blend=fract(location); float coverage=0.0; vec3 normalSum=vec3(0.0);
    for (uint y=0u;y<2u;++y) for (uint x=0u;x<2u;++x)
    {
        ivec2 pixel=lower+ivec2(x,y),nativePixel=ivec2(pixel.x,int(Atlas.y)-1-pixel.y);
        float weight=(x!=0u?blend.x:1.0-blend.x)*(y!=0u?blend.y:1.0-blend.y); vec4 materialNormal=texelFetch(LookupNormal,nativePixel,0);
        if (weight<=0.0 || dot(materialNormal.xyz,materialNormal.xyz)==0.0 || dot(materialNormal.xyz,query.NormalCosine.xyz)<query.NormalCosine.w) continue;
        vec2 uv=(vec2(pixel)-vec2(card.Tile.xy)-float(Border)+0.5)/float(Counts.y);
        vec4 projected=vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,materialNormal.w,1.0);
        vec3 position=vec3(dot(card.Inverse[0],projected),dot(card.Inverse[1],projected),dot(card.Inverse[2],projected));
        float error=abs(dot(relative-position,query.NormalCosine.xyz));
        float roundoff=Parameters.y*max(card.BackwardScale.w,max(maxCoordinate(relative),maxCoordinate(position)))+
            Parameters.z*dot(abs(query.PositionRadius.xyz),abs(query.NormalCosine.xyz));
        if (isnan(error) || isinf(error) || error>query.PositionRadius.w+roundoff) continue;
        value.AlbedoMetallic+=texelFetch(LookupAlbedo,nativePixel,0)*weight; value.EmissionRoughness+=texelFetch(LookupEmission,nativePixel,0)*weight;
        normalSum+=materialNormal.xyz*weight; coverage+=weight; depthError+=error*weight;
    }
    if (coverage<=0.0) return false;
    value.AlbedoMetallic/=coverage; value.EmissionRoughness/=coverage; value.NormalCoverage=vec4(normalSum/coverage,clamp(coverage,0.0,1.0));
    value.Metadata=uvec4(card.Tile.w,1u,1u,0u); depthError/=coverage; return true;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Atlas.z; if (index>=Counts.x) return;
    uint address=index*QueryVectors; LookupQuery query;
    query.PositionRadius=uintBitsToFloat(Queries.Values[address]); query.NormalCosine=uintBitsToFloat(Queries.Values[address+1u]); query.Range=Queries.Values[address+2u];
    LookupValue selected[DirectionCount]; float errors[DirectionCount],facing[DirectionCount];
    for (uint group=0u;group<DirectionCount;++group) { selected[group]=emptyValue(); errors[group]=0.0; facing[group]=0.0; }
    for (uint candidate=0u;candidate<query.Range.y;++candidate)
    {
        LookupCard card; readCard(query.Range.x+candidate,card); LookupValue value; float error,cosine;
        if (!sampleCard(card,query,value,error,cosine)) continue;
        uint group=card.Tile.z;
        if (selected[group].Metadata.z==0u || error<errors[group] || (error==errors[group] && value.NormalCoverage.w>selected[group].NormalCoverage.w))
        { selected[group]=value; errors[group]=error; facing[group]=cosine; }
    }
    LookupValue result=emptyValue(); float sum=0.0,angularSum=0.0;
    for (uint group=0u;group<DirectionCount;++group) if (selected[group].Metadata.z!=0u)
    { float angle=facing[group]*facing[group]; sum+=angle*selected[group].NormalCoverage.w; angularSum+=angle; }
    if (sum>0.0)
    {
        vec3 normal=vec3(0.0); float dominant=0.0;
        for (uint group=0u;group<DirectionCount;++group) if (selected[group].Metadata.z!=0u)
        {
            float weight=facing[group]*facing[group]*selected[group].NormalCoverage.w/sum;
            result.AlbedoMetallic+=selected[group].AlbedoMetallic*weight; result.EmissionRoughness+=selected[group].EmissionRoughness*weight;
            normal+=selected[group].NormalCoverage.xyz*weight; ++result.Metadata.y;
            if (weight>dominant || (weight==dominant && selected[group].Metadata.x<result.Metadata.x)) { dominant=weight; result.Metadata.x=selected[group].Metadata.x; }
        }
        float squared=dot(normal,normal); result.NormalCoverage=vec4(squared>0.0?normal*inversesqrt(squared):query.NormalCosine.xyz,clamp(sum/angularSum,0.0,1.0)); result.Metadata.z=1u;
    }
    result.AlbedoMetallic=clamp(result.AlbedoMetallic,0.0,1.0); result.EmissionRoughness.w=clamp(result.EmissionRoughness.w,0.0,1.0);
    uint base=index*ResultVectors; Output.Values[base]=floatBitsToUint(result.AlbedoMetallic); Output.Values[base+1u]=floatBitsToUint(result.NormalCoverage);
    Output.Values[base+2u]=floatBitsToUint(result.EmissionRoughness); Output.Values[base+3u]=result.Metadata;
}
#elif defined(EVANS_LUMEN_DISTANCE_TRACE_STAGE) || defined(EVANS_LUMEN_LIGHTING_STAGE) || defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE) || defined(EVANS_LUMEN_PROBE_TRACE_STAGE) || defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE) || defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
// ES 3.1 needs only three storage bindings: immutable scene, rays and feedback.
// World rays are re-anchored at each instance box before local-space marching.
layout(std140,row_major,binding=0) uniform DistanceFrame
{
    uvec4 Counts,Offsets,Traversal; vec4 Parameters,HairSky;
#if defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
    mat4 LinkInverse; uvec4 LinkViewport,LinkGrid; vec4 LinkOrigin,LinkAdaptive,LinkSurface,LinkNumeric;
    uvec4 LinkModel; vec4 LinkRules;
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uvec4 ProbeSampling; vec4 ProbeRange;
#endif
#if defined(EVANS_LUMEN_LIGHTING_STAGE)
    uvec4 CacheExtent,LightingWork; vec4 LightingOptions,LightingPrecision; uvec4 LightingTiles,BounceWork; vec4 BounceSky,EnvironmentAxis,EnvironmentPeak;
#endif
};
layout(std430,binding=0) readonly buffer DistanceScene { uint Data[]; } Scene;
struct DistanceRay { vec4 OriginMinimum,DirectionMaximum; };
struct DistanceHit { vec4 PositionDistance,NormalResidual,Uncertainty; uvec4 Metadata; vec4 Medium; };
// Explicit vector loads also avoid ANGLE's whole-structure SSBO load helper.
// Feedback uses integer bit patterns, preserving caller identifiers exactly.
#if defined(EVANS_LUMEN_LIGHTING_STAGE)
uniform highp sampler2D LightingAlbedo,LightingNormal,LightingEmission;
layout(std430,binding=1) readonly buffer LightingBuffer { uvec4 Vectors[]; } LightData;
layout(std430,binding=2) writeonly buffer LitBuffer { uvec4 Vectors[]; } Lit;
#elif defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
uniform highp sampler2D LinkDepth,LinkNormals;
layout(std430,binding=1) readonly buffer LinkOrigins { uvec4 Vectors[]; } Candidates;
layout(std430,binding=2) writeonly buffer LinkOutput { uint Masks[]; } Visible;
#elif defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
layout(std430,binding=1) readonly buffer VolumeOrigins { uvec4 Vectors[]; } VolumePositions;
layout(std430,binding=2) writeonly buffer VolumeBounds { float Radii[]; } VolumeSupport;
#else
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
layout(std430,binding=1) readonly buffer ProbeBuffer { uvec4 Vectors[]; } ProbeOrigins;
#else
layout(std430,binding=1) readonly buffer RayBuffer { vec4 Vectors[]; } Rays;
#endif
layout(std430,binding=2) buffer HitBuffer { uvec4 Vectors[]; } Hits;
#endif
const uint RayVectors=2u,HitVectors=5u;
const uint FieldSize=20u,InstanceSize=20u,NodeSize=12u,VectorSize=4u;
const uint FieldBrickGrid=4u,FieldAddresses=8u,FieldMinimumCell=12u,FieldBandError=16u,InstanceInverse=4u,InstanceTag=16u,NodeMax=4u,NodeEscape=8u;
const uint Interior=6u,Halo=1u,Side=8u,BrickCount=512u,FirstActive=2u;
const uint Invalid=0xffffffffu,Miss=0u,Surface=1u,Unresolved=2u;
const float Quantization=32767.0;
uvec4 wordsAt(uint a) { return uvec4(Scene.Data[a],Scene.Data[a+1u],Scene.Data[a+2u],Scene.Data[a+3u]); }
vec4 floatsAt(uint a) { return uintBitsToFloat(wordsAt(a)); }
bool finite3(vec3 v) { return !(any(isnan(v)) || any(isinf(v))); }
bool finite1(float v) { return !(isnan(v) || isinf(v)); }
vec3 applyInverse(uint object,vec3 v)
{ return vec3(dot(floatsAt(object+InstanceInverse).xyz,v),dot(floatsAt(object+InstanceInverse+VectorSize).xyz,v),dot(floatsAt(object+InstanceInverse+VectorSize*2u).xyz,v)); }
float vectorLength(vec3 v)
{ float m=max(abs(v.x),max(abs(v.y),abs(v.z))); return m>0.0?m*length(v/m):0.0; }
vec3 unitVector(vec3 v,vec3 fallback)
{
    if (!finite3(v)) return fallback;
    float m=max(abs(v.x),max(abs(v.y),abs(v.z))); if (m==0.0) return fallback;
    v/=m; return v*inversesqrt(dot(v,v));
}
bool rayBounds(vec3 o,vec3 d,vec3 lo,vec3 hi,inout float begin,inout float end)
{
    for (int axis=0;axis<3;++axis)
    {
        if (d[axis]==0.0) { if (o[axis]<lo[axis] || o[axis]>hi[axis]) return false; }
        else
        {
            float a=(lo[axis]-o[axis])/d[axis],b=(hi[axis]-o[axis])/d[axis];
            begin=max(begin,min(a,b)); end=min(end,max(a,b)); if (begin>end) return false;
        }
    }
    return true;
}
float distanceAt(uint field,vec3 p)
{
    uvec3 dimensions=wordsAt(field).xyz,brickGrid=wordsAt(field+FieldBrickGrid).xyz;
    uvec4 storage=wordsAt(field+FieldAddresses); vec4 grid=floatsAt(field+FieldMinimumCell);
    float band=floatsAt(field+FieldBandError).x;
    vec3 c=clamp((p-grid.xyz)/grid.w-0.5,vec3(0.0),vec3(dimensions)-1.0),f=fract(c);
    uvec3 cell=uvec3(floor(c)),block=cell/Interior,inside=cell%Interior+Halo;
    uint page=Scene.Data[storage.x+(block.z*brickGrid.y+block.y)*brickGrid.x+block.x];
    if (page<FirstActive) return page==1u?-band:band;
    float value=0.0;
    for (uint corner=0u;corner<8u;++corner)
    {
        uvec3 bit=uvec3(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u),texel=inside+bit;
        uint address=(page-FirstActive)*BrickCount+(texel.z*Side+texel.y)*Side+texel.x;
        uint pair=Scene.Data[storage.y+address/2u]; int encoded=(address&1u)!=0u?int(pair)>>16:int(pair<<16)>>16;
        vec3 weights=vec3(bit.x!=0u?f.x:1.0-f.x,bit.y!=0u?f.y:1.0-f.y,bit.z!=0u?f.z:1.0-f.z);
        value+=float(encoded)*weights.x*weights.y*weights.z;
    }
    return value*(band/Quantization);
}
vec3 normalAt(uint object,uint field,vec3 p,float cell,vec3 fallback)
{
    vec3 gradient=vec3(distanceAt(field,p+vec3(cell,0,0))-distanceAt(field,p-vec3(cell,0,0)),
        distanceAt(field,p+vec3(0,cell,0))-distanceAt(field,p-vec3(0,cell,0)),distanceAt(field,p+vec3(0,0,cell))-distanceAt(field,p-vec3(0,0,cell)));
    gradient=unitVector(gradient,vec3(0.0));
    return unitVector(floatsAt(object+InstanceInverse).xyz*gradient.x+floatsAt(object+InstanceInverse+VectorSize).xyz*gradient.y+
        floatsAt(object+InstanceInverse+VectorSize*2u).xyz*gradient.z,fallback);
}
uint traceObject(DistanceRay ray,uint index,float anchor,float limit,out float at,out vec4 normal,out vec2 uncertainty,out uint steps)
{
    at=anchor; normal=vec4(0.0); uncertainty=vec2(0.0); steps=0u;
    uint object=Offsets.y+index*InstanceSize,field=Offsets.x+wordsAt(object+InstanceTag).x*FieldSize;
    vec3 o=applyInverse(object,ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*anchor-floatsAt(object).xyz),d=applyInverse(object,ray.DirectionMaximum.xyz);
    float speed=vectorLength(d); if (!finite3(o) || !finite3(d) || !finite1(speed) || speed<=0.0) return Unresolved;
    vec4 grid=floatsAt(field+FieldMinimumCell),bounds=floatsAt(field+FieldBandError); float begin=0.0,end=max(0.0,limit-anchor);
    if (!rayBounds(o,d,grid.xyz,grid.xyz+vec3(wordsAt(field).xyz)*grid.w,begin,end)) return Miss;
    float travel=begin,threshold=bounds.y+Parameters.x*grid.w;
    for (uint iteration=0u;iteration<Traversal.x;++iteration)
    {
        at=min(limit,anchor+travel); vec3 p=o+d*travel; ++steps;
        if (!finite3(p)) return Unresolved;
        float value=distanceAt(field,p);
        if (abs(value)<=threshold)
        {
            normal=vec4(normalAt(object,field,p,grid.w,-ray.DirectionMaximum.xyz),value/speed);
            uncertainty=vec2(bounds.y,threshold)/speed; return Surface;
        }
        float next=travel+(abs(value)-bounds.y)/speed*Parameters.y;
        if (!finite1(next) || next<=travel) return Unresolved;
        if (next>end) return Miss; travel=next;
    }
    at=min(limit,anchor+travel); return Unresolved;
}
// Convert voxel error through the local tangent plane. Using |A*n| here
// would leave sheared receivers inside their own conservative SDF shell.
float planeDistanceRate(uint object,vec3 normal)
{
    vec3 r0=floatsAt(object+InstanceInverse).xyz,r1=floatsAt(object+InstanceInverse+VectorSize).xyz,r2=floatsAt(object+InstanceInverse+VectorSize*2u).xyz;
    vec3 extent=max(abs(r0),max(abs(r1),abs(r2))); float largest=max(extent.x,max(extent.y,extent.z)); if (largest==0.0) return 0.0;
    r0/=largest; r1/=largest; r2/=largest; vec3 c0=cross(r1,r2),c1=cross(r2,r0),c2=cross(r0,r1);
    float normalLength=vectorLength(vec3(dot(c0,normal),dot(c1,normal),dot(c2,normal)));
    float rate=normalLength>0.0?largest*(abs(dot(r0,c0))/normalLength):0.0; return finite1(rate)?rate:0.0;
}
DistanceHit traceLocalOpaqueDistance(DistanceRay ray,uint requiredFlags)
{
    float closest=ray.DirectionMaximum.w,unknown=closest;
    uint selected=Invalid,pending=Invalid,node=0u,stepCount=0u; vec4 surfaceNormal=vec4(0.0); vec2 surfaceError=vec2(0.0);
    while (node<Counts.y)
    {
        uint base=Offsets.z+node*NodeSize; uvec4 link=wordsAt(base+NodeEscape); float first=ray.OriginMinimum.w,last=min(closest,unknown);
        if (!rayBounds(ray.OriginMinimum.xyz,ray.DirectionMaximum.xyz,floatsAt(base).xyz,floatsAt(base+NodeMax).xyz,first,last)) { node=link.w; continue; }
        if (link.z==0u) { node=link.x; continue; }
        uint flags=wordsAt(Offsets.y+link.y*InstanceSize+InstanceTag).z;
        if ((flags&requiredFlags)!=requiredFlags) { node=link.w; continue; }
        float at; vec4 normal; vec2 error; uint steps;
        uint status=traceObject(ray,link.y,first,last,at,normal,error,steps); stepCount+=steps;
        if (status==Surface && (at<closest || (at==closest && link.y<selected))) { closest=at; selected=link.y; surfaceNormal=normal; surfaceError=error; }
        if (status==Unresolved && (at<unknown || (at==unknown && link.y<pending))) { unknown=at; pending=link.y; }
        node=link.w;
    }
    DistanceHit result; result.Medium=vec4(0,0,0,1); result.NormalResidual=vec4(0.0); result.Uncertainty=vec4(0.0); result.Metadata=uvec4(Invalid,Invalid,Miss,stepCount);
    float at=ray.DirectionMaximum.w;
    if (pending!=Invalid && (selected==Invalid || unknown<=closest))
    { at=unknown; result.Metadata.xyz=uvec3(pending,wordsAt(Offsets.y+pending*InstanceSize+InstanceTag).y,Unresolved); }
    else if (selected!=Invalid)
    { at=closest; result.NormalResidual=surfaceNormal; result.Uncertainty.xy=surfaceError; result.Metadata.xyz=uvec3(selected,wordsAt(Offsets.y+selected*InstanceSize+InstanceTag).y,Surface); }
    result.PositionDistance=vec4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*at,at); return result;
}

DistanceHit traceLocalOpaqueDistance(DistanceRay ray) { return traceLocalOpaqueDistance(ray,Traversal.z); }

float clipmapEmptyRadius(vec3 world)
{
    const uint HeaderPrefix=8u,GridSize=12u,MetricOffset=4u,StorageOffset=8u,VoxelWords=4u;
    uvec4 shape=wordsAt(Offsets.w);
    for (uint cascade=0u;cascade<shape.x;++cascade)
    {
        uint address=Offsets.w+HeaderPrefix+cascade*GridSize; vec4 grid=floatsAt(address);
        vec3 p=(world-grid.xyz)/grid.w;
        if (!finite3(p) || any(lessThan(p,vec3(0))) || any(greaterThan(p,vec3(shape.y-1u)))) continue;
        uvec3 lower=min(uvec3(floor(p)),uvec3(shape.y-2u)); vec3 f=p-vec3(lower); float sum=0.0;
        uint base=wordsAt(address+StorageOffset).x;
        for (uint z=0u;z<2u;++z) for (uint y=0u;y<2u;++y) for (uint x=0u;x<2u;++x)
        {
            uvec3 cell=lower+uvec3(x,y,z); vec4 value=floatsAt(base+((cell.z*shape.y+cell.y)*shape.y+cell.x)*VoxelWords);
            float weight=(x!=0u?f.x:1.0-f.x)*(y!=0u?f.y:1.0-f.y)*(z!=0u?f.z:1.0-f.z);
            sum+=(Traversal.z!=0u?value.w:value.y)*weight;
        }
        return finite1(sum)?max(0.0,sum-floatsAt(address+MetricOffset).x):0.0;
    }
    return 0.0;
}
DistanceHit traceOpaqueDistance(DistanceRay ray)
{
    DistanceHit hit; hit.Medium=vec4(0,0,0,1); hit.PositionDistance=vec4(0); hit.NormalResidual=vec4(0); hit.Uncertainty=vec4(0); hit.Metadata=uvec4(Invalid,Invalid,Miss,0);
    bool global=Offsets.w!=0u; uint intervals=global && Parameters.z>0.0?2u:1u;
    float distance=ray.OriginMinimum.w; uint detailSteps=0u,globalSteps=0u;
    for (uint interval=0u;interval<intervals;++interval)
    {
        bool detailNear=intervals==2u && interval==0u;
        if (global && !detailNear) for (uint iteration=0u;iteration<uint(Parameters.w);++iteration)
        {
            ++globalSteps; float empty=clipmapEmptyRadius(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*distance);
            float next=distance+empty*Parameters.y; if (!finite1(next) || next<=distance) break;
            if (next>ray.DirectionMaximum.w)
            {
                hit.PositionDistance=vec4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
                hit.NormalResidual=vec4(0); hit.Uncertainty=vec4(0,0,0,float(globalSteps)); hit.Metadata=uvec4(Invalid,Invalid,Miss,detailSteps); return hit;
            }
            distance=next;
        }
        DistanceRay detail=ray; detail.OriginMinimum.w=distance;
        if (detailNear) detail.DirectionMaximum.w=min(ray.DirectionMaximum.w,distance+Parameters.z);
        hit=traceLocalOpaqueDistance(detail); hit.Metadata.w+=detailSteps; hit.Uncertainty.w=float(globalSteps);
        if (!detailNear || hit.Metadata.z!=Miss || detail.DirectionMaximum.w==ray.DirectionMaximum.w) return hit;
        distance=detail.DirectionMaximum.w; detailSteps=hit.Metadata.w;
    }
    return hit;
}
// ES keeps medium data in the existing immutable word binding. Each actual
// cell interval uses Beer attenuation; orientation moments average crossing
// strands. Sky scattering is isotropic and low frequency, with explicit emission.
vec4 hairMedium(DistanceRay ray,float maximum,uint requiredFlags)
{
    const uint HairWordsPerCell=24u,HairWordsPerVector=4u;
    vec4 medium=vec4(0,0,0,1); uint pointer=Traversal.w>>LumenHairPointerShift; if (pointer==0u) return medium;
    uvec4 extent=wordsAt(pointer); vec4 lattice=floatsAt(pointer+HairWordsPerVector);
    vec3 origin=ray.OriginMinimum.xyz-lattice.xyz,direction=ray.DirectionMaximum.xyz;
    float begin=ray.OriginMinimum.w,end=min(maximum,ray.DirectionMaximum.w);
    if (end<=begin || !rayBounds(origin,direction,vec3(0),vec3(extent.xyz)*lattice.w,begin,end) || end<=begin) return medium;
    ivec3 voxel=clamp(ivec3(floor((origin+begin*direction)/lattice.w)),ivec3(0),ivec3(extent.xyz)-1);
    ivec3 advance=ivec3(sign(direction)); bool caster=(requiredFlags&1u)!=0u;
    for (uint iteration=0u;iteration<extent.x+extent.y+extent.z+1u && begin<end;++iteration)
    {
        vec3 boundary=vec3(end);
        for (uint axis=0u;axis<3u;++axis) if (advance[axis]!=0)
            boundary[axis]=((float(voxel[axis])+(advance[axis]>0?1.0:0.0))*lattice.w-origin[axis])/direction[axis];
        float next=max(begin,min(end,min(boundary.x,min(boundary.y,boundary.z))));
        uint word=extent.w+(uint(voxel.x)+extent.x*(uint(voxel.y)+extent.y*uint(voxel.z)))*HairWordsPerCell;
        vec4 density=floatsAt(word),source=floatsAt(word+HairWordsPerVector);
        vec4 orientation=floatsAt(word+(caster?4u:2u)*HairWordsPerVector),crossing=floatsAt(word+(caster?5u:3u)*HairWordsPerVector);
        float parallel=dot(orientation.xyz,direction*direction)+2.0*(orientation.w*direction.x*direction.y+crossing.x*direction.x*direction.z+crossing.y*direction.y*direction.z);
        float extinction=(caster?source.w:density.x)*sqrt(clamp(1.0-parallel,0.0,1.0));
        float remaining=exp(-extinction*(next-begin));
        if (!caster) medium.xyz+=medium.w*(1.0-remaining)*(density.yzw*HairSky.xyz+source.xyz);
        medium.w*=remaining; begin=next; if (begin>=end) break;
        bool escaped=false;
        for (uint axis=0u;axis<3u;++axis) if (advance[axis]!=0 && boundary[axis]<=next)
        { voxel[axis]+=advance[axis]; escaped=escaped || voxel[axis]<0 || voxel[axis]>=int(extent[axis]); }
        if (escaped) break;
    }
    return medium;
}
DistanceHit traceLocalDistanceScene(DistanceRay ray,uint flags)
{
    DistanceHit hit=traceLocalOpaqueDistance(ray,flags);
    hit.Medium=hairMedium(ray,hit.Metadata.z==Miss?ray.DirectionMaximum.w:hit.PositionDistance.w,flags); return hit;
}
DistanceHit traceLocalDistanceScene(DistanceRay ray) { return traceLocalDistanceScene(ray,Traversal.z); }
DistanceHit traceDistanceScene(DistanceRay ray)
{
    DistanceHit hit=traceOpaqueDistance(ray);
    hit.Medium=hairMedium(ray,hit.Metadata.z==Miss?ray.DirectionMaximum.w:hit.PositionDistance.w,Traversal.z); return hit;
}
#if defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
float volumeMagnitude(vec3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
float volumeInstanceSupport(uint index,vec3 world)
{
    uint object=Offsets.y+index*InstanceSize,field=Offsets.x+wordsAt(object+InstanceTag).x*FieldSize;
    vec3 center=floatsAt(object).xyz,delta=world-center;
    vec3 a=floatsAt(object+InstanceInverse).xyz,b=floatsAt(object+InstanceInverse+VectorSize).xyz,c=floatsAt(object+InstanceInverse+VectorSize*2u).xyz;
    vec3 p=vec3(dot(a,delta),dot(b,delta),dot(c,delta)); float scale=max(volumeMagnitude(a),max(volumeMagnitude(b),volumeMagnitude(c)));
    if (!finite3(p) || !finite1(scale) || scale<=0.0) return 0.0;
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    // sqrt(||A||1 ||A||infinity) bounds the linear transform's spectral norm.
    float rows=max(dot(a,vec3(1.0)),max(dot(b,vec3(1.0)),dot(c,vec3(1.0)))),norm=scale*sqrt(rows*volumeMagnitude(a+b+c));
    if (!finite1(norm) || norm<=0.0) return 0.0;
    vec4 grid=floatsAt(field+FieldMinimumCell),errors=floatsAt(field+FieldBandError);
    vec3 end=grid.xyz+vec3(wordsAt(field).xyz)*grid.w,sampled=clamp(p,grid.xyz+grid.w*0.5,end-grid.w*0.5);
    float outside=vectorLength(p-clamp(p,grid.xyz,end)),extension=vectorLength(p-sampled);
    // A point sample has one interpolation error, without a later ray hit's
    // additional error bands. Keep the native hit shell outside the sphere.
    float lower=max(outside,abs(distanceAt(field,sampled))-errors.y-Parameters.x*grid.w-extension);
    lower-=Parameters.y*max(volumeMagnitude(p),max(volumeMagnitude(grid.xyz),volumeMagnitude(end)));
    float support=lower/norm-Parameters.y*max(volumeMagnitude(world),volumeMagnitude(center));
    return finite1(support)?clamp(support,0.0,Parameters.z):0.0;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Traversal.y; if (index>=Counts.x) return;
    uint base=index*RayVectors; vec3 origin=uintBitsToFloat(VolumePositions.Vectors[base]).xyz;
    float support=VolumePositions.Vectors[base+1u].y==0u?Parameters.z:0.0;
    for (uint object=0u;object<Counts.z && support>0.0;++object) support=min(support,volumeInstanceSupport(object,origin));
    VolumeSupport.Radii[index]=support;
}
#elif defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
// Canonical top-left probe indices remain independent of ES framebuffer rows.
// The SDF scene includes GI occluders even when direct-light shadows are off.
ivec2 visibilityAddress(uvec2 pixel) { return ivec2(pixel.x,LinkViewport.y-1u-pixel.y); }
float visibilityCoordinate(float center,uint extent,uint tiles)
{
    float side=float(LinkViewport.z),grid=center/side-0.5;
    if (tiles>1u && extent%LinkViewport.z!=0u)
    {
        float before=(float(tiles)-1.5)*side,last=(float(tiles-1u)*side+float(extent))*0.5;
        if (center>before) grid=float(tiles-2u)+(center-before)/(last-before);
    }
    return clamp(grid,0.0,float(tiles-1u));
}
float visibilityMagnitude(vec3 value) { return max(abs(value.x),max(abs(value.y),abs(value.z))); }
layout(local_size_x=64) in;
void main()
{
    uint receiver=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Traversal.y; if (receiver>=Counts.x) return;
    uint words=probeLinkWords(LinkGrid.w+1u);
    for (uint word=0u;word<words;++word) Visible.Masks[receiver*words+word]=0u; uvec2 pixel=uvec2(receiver%LinkViewport.x,receiver/LinkViewport.x);
    ivec2 address=visibilityAddress(pixel); float depth=texelFetch(LinkDepth,address,0).r; vec4 material=texelFetch(LinkNormals,address,0);
    if (!finite1(depth) || depth<0.0 || depth>=1.0 || !finite3(material.xyz) || !finite1(material.w) || material.w<0.0 || dot(material.xyz,material.xyz)<=LinkNumeric.x) return;
    vec2 center=vec2(pixel)+0.5,uv=center/vec2(LinkViewport.xy);
    vec4 h=LinkInverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    if (!finite3(h.xyz) || !finite1(h.w) || abs(h.w)<=LinkOrigin.w) return;
    vec3 position=h.xyz/h.w,n=normalize(material.xyz);
    if (!finite3(position) || any(greaterThan(abs(position),vec3(LinkOrigin.z)))) return;
    vec2 grid=vec2(visibilityCoordinate(center.x,LinkViewport.x,LinkViewport.w),visibilityCoordinate(center.y,LinkViewport.y,LinkGrid.x));
    uvec2 base=uvec2(floor(grid)); vec2 fraction=grid-vec2(base); uvec3 accepted=uvec3(0u);
    uint slots=LinkGrid.w+1u;
    for (uint tap=0u;tap<4u*slots;++tap)
    {
        uint cornerIndex=tap/slots;
        uvec2 corner=uvec2(cornerIndex&1u,cornerIndex>>1u),cell=min(base+corner,uvec2(LinkViewport.w,LinkGrid.x)-1u);
        if ((corner.x!=0u?fraction.x:1.0-fraction.x)*(corner.y!=0u?fraction.y:1.0-fraction.y)<=0.0) continue;
        uint probe=(cell.y*LinkViewport.w+cell.x)*slots+tap%slots; uvec4 tag=Candidates.Vectors[probe*RayVectors+1u];
        if (tag.x!=probe || tag.y!=0u || any(greaterThanEqual(tag.zw,LinkViewport.xy))) continue;
        vec4 probeSurface=texelFetch(LinkNormals,visibilityAddress(tag.zw),0);
        if (!finite3(probeSurface.xyz) || !finite1(probeSurface.w) || probeSurface.w<0.0 || dot(probeSurface.xyz,probeSurface.xyz)<=LinkNumeric.x) continue;
        vec3 normalAtProbe=normalize(probeSurface.xyz),anchor=uintBitsToFloat(Candidates.Vectors[probe*RayVectors]).xyz-normalAtProbe*LinkOrigin.x;
        if (dot(n,normalAtProbe)<=LinkSurface.y || !finite3(anchor)) continue;
        float offset=LinkRules.x+LinkRules.y*max(visibilityMagnitude(position),visibilityMagnitude(anchor));
        vec3 start=position+n*offset,end=anchor+normalAtProbe*offset,delta=end-start; float span=vectorLength(delta);
        if (!finite3(start) || !finite3(end) || !finite1(span)) continue;
        if (span==0.0) { accepted[tap/LumenLinkWordBits]|=1u<<(tap%LumenLinkWordBits); continue; }
        DistanceRay segment=DistanceRay(vec4(start,0.0),vec4(unitVector(delta,n),span));
        DistanceHit hit=traceDistanceScene(segment); if (hit.Metadata.z==Miss) accepted[tap/LumenLinkWordBits]|=1u<<(tap%LumenLinkWordBits);
    }
    for (uint word=0u;word<words;++word) Visible.Masks[receiver*words+word]=accepted[word];
}
#elif !defined(EVANS_LUMEN_LIGHTING_STAGE)
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Traversal.y; if (index>=Counts.x) return;
    DistanceRay ray;
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uint source=(index/ProbeSampling.x)*RayVectors; ray.OriginMinimum=uintBitsToFloat(ProbeOrigins.Vectors[source]);

    uint id=ProbeOrigins.Vectors[source+1u].x;
    if (ProbeOrigins.Vectors[source+1u].y!=0u)
    { uint base=index*HitVectors; Hits.Vectors[base]=uvec4(0u); Hits.Vectors[base+1u]=uvec4(0u); Hits.Vectors[base+2u]=uvec4(0u); Hits.Vectors[base+3u]=uvec4(Invalid,Invalid,Unresolved,0u); Hits.Vectors[base+4u]=floatBitsToUint(vec4(0,0,0,1)); return; }
    vec3 direction=ProbeRange.y>0.0?worldProbeAtlasDirection(index%ProbeSampling.x,uint(ProbeRange.y)):
        probeSphereSample(index%ProbeSampling.x,ProbeSampling.x,id,ProbeSampling.z);
    bool cached=(ProbeSampling.w&2u)!=0u && Hits.Vectors[index*HitVectors+3u].z==LumenWorldCache;
    bool cachedSurface=cached && Hits.Vectors[index*HitVectors+3u].x!=Invalid;
    // Full tracing preserves the medium interval when the cache has no endpoint.
    if ((Traversal.w>>LumenHairPointerShift)!=0u && !cachedSurface) cached=false;
    ray.DirectionMaximum=vec4(direction,cached && !cachedSurface?uintBitsToFloat(Hits.Vectors[index*HitVectors]).w:ProbeRange.x);
    if ((ProbeSampling.w&1u)!=0u && Hits.Vectors[index*HitVectors+3u].z==LumenScreenSurface)
    { Hits.Vectors[index*HitVectors+4u]=floatBitsToUint(hairMedium(ray,uintBitsToFloat(Hits.Vectors[index*HitVectors]).w,Traversal.z)); return; }
#else
    ray.OriginMinimum=Rays.Vectors[index*RayVectors]; ray.DirectionMaximum=Rays.Vectors[index*RayVectors+1u];
#if defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE)
    const uint ReflectionGenerated=1u,ReflectionScreenFirst=2u;
    if ((Traversal.w&ReflectionScreenFirst)!=0u && Hits.Vectors[index*HitVectors+3u].z==LumenScreenSurface) { Hits.Vectors[index*HitVectors+4u]=floatBitsToUint(hairMedium(ray,uintBitsToFloat(Hits.Vectors[index*HitVectors]).w,Traversal.z)); return; }
    if ((Traversal.w&ReflectionGenerated)!=0u && ray.DirectionMaximum.w<=0.0)
    {
        uint destination=index*HitVectors; Hits.Vectors[destination]=uvec4(0u); Hits.Vectors[destination+1u]=uvec4(0u);
        Hits.Vectors[destination+2u]=uvec4(0u); Hits.Vectors[destination+3u]=uvec4(Invalid,Invalid,Unresolved,0u); Hits.Vectors[destination+4u]=floatBitsToUint(vec4(0,0,0,1)); return;
    }
#endif
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uint layers=uint(ProbeRange.z); bool terminal=false;
    DistanceHit previous; previous.Medium=vec4(0,0,0,1); previous.PositionDistance=vec4(0); previous.NormalResidual=vec4(0); previous.Uncertainty=vec4(0); previous.Metadata=uvec4(0u);
    const float MinimumLayerCosine=0.05,SameLayerNormalCosine=0.95;
    // One invocation writes a contiguous probe's ordered directional stack.
    // Unknown tails stay unknown; a repeated SDF hit shell is not a new layer.
    for (uint layer=0u;layer<layers;++layer)
    {
    DistanceHit result=previous; if (!terminal) result=traceDistanceScene(ray);
#else
    DistanceHit result=traceDistanceScene(ray);
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    if (cached && !cachedSurface && result.Metadata.z==Miss) { Hits.Vectors[index*HitVectors+4u]=floatBitsToUint(result.Medium); return; }
#endif
#if defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE) || defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    if (result.Metadata.z==Surface)
    {
        uint object=Offsets.y+result.Metadata.x*InstanceSize,field=Offsets.x+wordsAt(object+InstanceTag).x*FieldSize;
        float threshold=floatsAt(field+FieldBandError).y+Parameters.x*floatsAt(field+FieldMinimumCell).w;
        result.Uncertainty.z=threshold/planeDistanceRate(object,result.NormalResidual.xyz);
    }
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    float roundoff=ProbeRange.w*max(1.0,max(abs(ray.OriginMinimum.x),max(abs(ray.OriginMinimum.y),abs(ray.OriginMinimum.z)))+ray.DirectionMaximum.w);
    if (cachedSurface && result.Metadata.z==Surface)
    {
        uint expected=index*HitVectors; uvec4 tag=Hits.Vectors[expected+3u];
        vec3 expectedNormal=uintBitsToFloat(Hits.Vectors[expected+1u]).xyz;
        float tolerance=uintBitsToFloat(Hits.Vectors[expected+2u]).x+result.Uncertainty.z+roundoff;
        if (all(equal(result.Metadata.xy,tag.xy)) && dot(result.NormalResidual.xyz,expectedNormal)>=SameLayerNormalCosine && finite1(tolerance) &&
            abs(dot(result.PositionDistance.xyz-uintBitsToFloat(Hits.Vectors[expected]).xyz,expectedNormal))<=tolerance) { Hits.Vectors[expected+4u]=floatBitsToUint(result.Medium); return; }
    }
    if (layer>0u && !terminal && result.Metadata.z==Surface && result.Metadata.x==previous.Metadata.x &&
        abs(dot(result.NormalResidual.xyz,previous.NormalResidual.xyz))>=SameLayerNormalCosine &&
        abs(dot(result.PositionDistance.xyz-previous.PositionDistance.xyz,previous.NormalResidual.xyz))<=result.Uncertainty.z+previous.Uncertainty.z+roundoff)
    { result.Metadata.z=Unresolved; result.NormalResidual=vec4(0); result.Uncertainty.xyz=vec3(0); }
    uint destination=(((index/ProbeSampling.x)*layers+layer)*ProbeSampling.x+index%ProbeSampling.x)*HitVectors;
#else
    uint destination=index*HitVectors;
#endif
    Hits.Vectors[destination]=floatBitsToUint(result.PositionDistance);
    Hits.Vectors[destination+1u]=floatBitsToUint(result.NormalResidual); Hits.Vectors[destination+2u]=floatBitsToUint(result.Uncertainty); Hits.Vectors[destination+3u]=result.Metadata; Hits.Vectors[destination+4u]=floatBitsToUint(result.Medium);
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    if (layer+1u==layers) break; previous=result;
    if (!terminal && result.Metadata.z==Surface)
    {
        float cosine=abs(dot(result.NormalResidual.xyz,ray.DirectionMaximum.xyz));
        float next=result.PositionDistance.w+2.0*result.Uncertainty.z/max(cosine,MinimumLayerCosine)+roundoff;
        if (cosine>=MinimumLayerCosine && finite1(next) && next>result.PositionDistance.w && next<ray.DirectionMaximum.w) ray.OriginMinimum.w=next;
        else { previous.Metadata.z=Unresolved; previous.NormalResidual=vec4(0); previous.Uncertainty.xyz=vec3(0); terminal=true; }
    }
    else terminal=true;
    }
#endif
}
#else
const uint LightingCardVectors=7u,LightingSampleVectors=4u,CardOriginOffset=4u,CardNormalOffset=5u,CardTileOffset=6u,LightingBorder=1u;
const uint DirectionalLight=0u,AreaLight=2u,CompleteLighting=1u,UnresolvedLighting=2u,InvalidLighting=3u,LightingResultVectors=2u;
const uint SampleOrientation=1u,SampleRadiance=2u,SampleShadow=3u;
vec4 lightingFloats(uint at) { return uintBitsToFloat(LightData.Vectors[at]); }
ivec2 lightingNativePixel(ivec2 canonical) { return ivec2(canonical.x,int(CacheExtent.y)-1-canonical.y); }
vec3 lightingPoint(uint card,uvec2 tile,ivec2 pixel,float depth)
{
    vec2 uv=(vec2(pixel)-vec2(tile)-float(LightingBorder)+0.5)/float(CacheExtent.z);
    vec4 clip=vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    return vec3(dot(lightingFloats(card),clip),dot(lightingFloats(card+1u),clip),dot(lightingFloats(card+2u),clip));
}
bool lightingDifference(uint card,uvec2 tile,ivec2 pixel,ivec2 offset,vec3 center,out vec3 delta)
{
    ivec2 neighbor=pixel+offset; delta=vec3(0.0);
    if (any(lessThan(neighbor,ivec2(tile))) || any(greaterThanEqual(neighbor,ivec2(tile+LightingWork.w)))) return false;
    vec4 depth=texelFetch(LightingNormal,lightingNativePixel(neighbor),0); if (dot(depth.xyz,depth.xyz)==0.0) return false;
    delta=lightingPoint(card,tile,neighbor,depth.w)-center; return finite3(delta);
}
vec3 lightingGeometric(uint card,uvec2 tile,ivec2 pixel,vec3 center,vec3 backward)
{
    vec3 xp,xn,yp,yn;
    bool px=lightingDifference(card,tile,pixel,ivec2(1,0),center,xp),nx=lightingDifference(card,tile,pixel,ivec2(-1,0),center,xn);
    bool py=lightingDifference(card,tile,pixel,ivec2(0,1),center,yp),ny=lightingDifference(card,tile,pixel,ivec2(0,-1),center,yn);
    if ((!px && !nx) || (!py && !ny)) return backward;
    vec3 dx=px && (!nx || dot(xp,xp)<=dot(xn,xn))?xp:-xn,dy=py && (!ny || dot(yp,yp)<=dot(yn,yn))?yp:-yn;
    vec3 n=unitVector(cross(dx,dy),backward); return dot(n,backward)<0.0?-n:n;
}
#if defined(EVANS_LUMEN_RADIOSITY)
// A fourth SSBO is the completed previous bounce. Raw vector loads avoid
// ANGLE whole-structure helpers; only material images use bottom-up Y.
layout(std430,binding=3) readonly buffer BounceSource { uvec4 Vectors[]; } Previous;
const uint BounceCardSize=11u,BounceInverseOffset=4u,BounceOriginOffset=8u,BounceNormalOffset=9u,BounceTileOffset=10u,CardDirections=6u;
const float BounceFacingLimit=0.1,BounceNormalLimit=0.25,BounceVoxelRadius=2.0;
const float BounceRelativeError=16.0*1.1920928955078125e-7,BounceAbsoluteError=0.5*1.1920928955078125e-7;
const float BouncePi=3.14159265358979323846,BouncePhaseUnit=1.0/4294967296.0,BounceAxisThreshold=0.9;
vec3 bounceProject(uint address,vec4 position)
{ return vec3(dot(lightingFloats(address),position),dot(lightingFloats(address+1u),position),dot(lightingFloats(address+2u),position)); }
float bounceMaxComponent(vec3 position) { return max(abs(position.x),max(abs(position.y),abs(position.z))); }
bool bounceReadCard(uint index,DistanceHit hit,float tolerance,out vec4 radiance,out vec2 quality,out uint status)
{
    uint address=BounceWork.z+BounceWork.w+index*BounceCardSize; uvec4 tile=LightData.Vectors[address+BounceTileOffset];
    vec4 backward=lightingFloats(address+BounceNormalOffset); radiance=vec4(0); quality=vec2(0); status=0u;
    quality.y=dot(backward.xyz,hit.NormalResidual.xyz); if (quality.y<BounceFacingLimit) return false;
    vec3 local=hit.PositionDistance.xyz-lightingFloats(address+BounceOriginOffset).xyz,projected=bounceProject(address,vec4(local,1));
    if (!finite3(projected) || any(greaterThan(abs(projected.xy),vec2(1)))) return false;
    vec2 uv=vec2(projected.x+1.0,1.0-projected.y)*0.5;
    vec2 coordinate=vec2(tile.xy)+float(LightingBorder)+uv*float(CacheExtent.z)-0.5;
    ivec2 lower=ivec2(floor(coordinate)); vec2 fraction=fract(coordinate); float accepted=0.0; status=CompleteLighting;
    for (int y=0;y<2;++y) for (int x=0;x<2;++x)
    {
        float weight=(x!=0?fraction.x:1.0-fraction.x)*(y!=0?fraction.y:1.0-fraction.y);
        ivec2 pixel=lower+ivec2(x,y); vec4 normal=texelFetch(LightingNormal,lightingNativePixel(pixel),0);
        if (weight<=0.0 || dot(normal.xyz,normal.xyz)==0.0 || dot(normal.xyz,hit.NormalResidual.xyz)<BounceNormalLimit) continue;
        vec2 texel=(vec2(pixel)-vec2(tile.xy)-float(LightingBorder)+0.5)/float(CacheExtent.z);
        vec3 captured=bounceProject(address+BounceInverseOffset,vec4(texel.x*2.0-1.0,1.0-texel.y*2.0,normal.w,1));
        float separation=abs(dot(local-captured,hit.NormalResidual.xyz));
        float roundoff=BounceRelativeError*max(backward.w,max(bounceMaxComponent(local),bounceMaxComponent(captured)))+
            BounceAbsoluteError*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!finite1(separation) || separation>tolerance+roundoff) continue;
        uint source=(uint(pixel.y)*CacheExtent.x+uint(pixel.x))*LightingResultVectors;
        uint state=Previous.Vectors[source+1u].x; status=max(status,state==0u?InvalidLighting:state);
        radiance.xyz+=uintBitsToFloat(Previous.Vectors[source]).xyz*weight; quality.x+=separation*weight; accepted+=weight;
    }
    if (accepted==0.0) { status=0u; return false; }
    radiance=vec4(radiance.xyz/accepted,clamp(accepted,0.0,1.0)); quality.x/=accepted; return true;
}
uint bounceResolve(DistanceHit hit,out vec3 value)
{
    value=vec3(0); uvec4 range=LightData.Vectors[BounceWork.z+hit.Metadata.x]; if (range.y==0u) return 0u;
    uint object=Offsets.y+hit.Metadata.x*InstanceSize,field=Offsets.x+wordsAt(object+InstanceTag).x*FieldSize;
    float error=floatsAt(field+FieldBandError).y+Parameters.x*floatsAt(field+FieldMinimumCell).w;
    float radius=error*BounceVoxelRadius/planeDistanceRate(object,hit.NormalResidual.xyz); if (!finite1(radius)) return InvalidLighting;
    vec4 candidates[CardDirections]; vec2 metrics[CardDirections]; uint validity[CardDirections];
    for (uint group=0u;group<CardDirections;++group) { candidates[group]=vec4(0); metrics[group]=vec2(0); validity[group]=0u; }
    for (uint candidate=0u;candidate<range.y;++candidate)
    {
        uint word=BounceWork.z*4u+range.x+candidate,card=LightData.Vectors[word/4u][word%4u];
        uint group=LightData.Vectors[BounceWork.z+BounceWork.w+card*BounceCardSize+BounceTileOffset].z;
        vec4 radiance; vec2 quality; uint status; if (!bounceReadCard(card,hit,radius,radiance,quality,status)) continue;
        if (validity[group]==0u || quality.x<metrics[group].x || (quality.x==metrics[group].x && radiance.w>candidates[group].w))
        { candidates[group]=radiance; metrics[group]=quality; validity[group]=status; }
    }
    float normalization=0.0; uint result=0u;
    for (uint group=0u;group<CardDirections;++group) if (validity[group]!=0u)
    {
        float weight=metrics[group].y*metrics[group].y*candidates[group].w;
        value+=candidates[group].xyz*weight; normalization+=weight; result=max(result,validity[group]);
    }
    if (normalization>0.0) value/=normalization;
    if (result==InvalidLighting || !finite3(value)) { value=vec3(0); return InvalidLighting; }
    return normalization>0.0?result:0u;
}
vec3 bounceCosineSample(uint ray,uint texel,vec3 normal)
{
    float squared=(float(ray)+0.5)/float(BounceWork.x);
    uint rotated=bitfieldReverse(ray)+bitfieldReverse(texel^BounceWork.y); float angle=2.0*BouncePi*float(rotated)*BouncePhaseUnit;
    vec3 axis=abs(normal.z)<BounceAxisThreshold?vec3(0,0,1):vec3(0,1,0),tangent=normalize(cross(axis,normal));
    return tangent*(sqrt(squared)*cos(angle))+cross(normal,tangent)*(sqrt(squared)*sin(angle))+normal*sqrt(1.0-squared);
}
#endif
void storeLighting(uint index,vec4 radiance,uvec4 metadata)
{ uint base=index*LightingResultVectors; Lit.Vectors[base]=floatBitsToUint(radiance); Lit.Vectors[base+1u]=metadata; }
layout(local_size_x=64) in;
void main()
{
    uint invocation=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Traversal.y,area=LightingTiles.w*LightingTiles.w;
    if (invocation>=LightingTiles.y*area) return;
    uint tileIndex=LightingTiles.x+invocation/area,within=invocation%area;
    uvec2 pixel=uvec2(tileIndex%LightingTiles.z,tileIndex/LightingTiles.z)*LightingTiles.w+uvec2(within%LightingTiles.w,within/LightingTiles.w);
    if (any(greaterThanEqual(pixel,CacheExtent.xy))) return;
    uint index=pixel.y*CacheExtent.x+pixel.x,cardIndex=(pixel.y/LightingWork.w)*CacheExtent.w+pixel.x/LightingWork.w;
    if (cardIndex>=LightingWork.x) { storeLighting(index,vec4(0.0),uvec4(0u)); return; }
    ivec2 native=lightingNativePixel(ivec2(pixel)); vec4 materialNormal=texelFetch(LightingNormal,native,0);
    if (dot(materialNormal.xyz,materialNormal.xyz)==0.0) { storeLighting(index,vec4(0.0),uvec4(0u)); return; }
    uint card=cardIndex*LightingCardVectors; uvec4 tile=LightData.Vectors[card+CardTileOffset];
    vec3 relative=lightingPoint(card,tile.xy,ivec2(pixel),materialNormal.w),position=relative+lightingFloats(card+CardOriginOffset).xyz;
    vec3 geometric=lightingGeometric(card,tile.xy,ivec2(pixel),relative,lightingFloats(card+CardNormalOffset).xyz);
    float bias=max(LightingOptions.x,LightingPrecision.x*max(abs(position.x),max(abs(position.y),abs(position.z))));
    if (tile.z!=Invalid)
    {
        uint object=Offsets.y+tile.z*InstanceSize,field=Offsets.x+wordsAt(object+InstanceTag).x*FieldSize;
        float speed=planeDistanceRate(object,geometric); float threshold=floatsAt(field+FieldBandError).y+Parameters.x*floatsAt(field+FieldMinimumCell).w;
        bias=max(bias,threshold*LightingOptions.y/speed);
    }
    vec3 origin=position+geometric*bias,irradiance=vec3(0.0); uvec4 metadata=uvec4(0u);
    bool invalid=!finite3(origin) || !finite3(materialNormal.xyz) || !finite1(materialNormal.w);
    for (uint lightIndex=0u;lightIndex<LightingWork.y && !invalid;++lightIndex)
    {
        uint base=LightingWork.z+lightIndex*LightingSampleVectors;
        vec4 source=lightingFloats(base),normalRange=lightingFloats(base+SampleOrientation),energy=lightingFloats(base+SampleRadiance);
        if (!any(greaterThan(energy.xyz,vec3(0.0))) || energy.w==0.0) continue;
        vec3 wi=source.xyz; float attenuation=1.0;
        if (uint(source.w)!=DirectionalLight)
        {
            vec3 delta=source.xyz-position; float distance=vectorLength(delta); if (distance==0.0) continue;
            wi=delta/distance; float window=1.0;
            if (normalRange.w>0.0)
            {
                if (max(abs(delta.x),max(abs(delta.y),abs(delta.z)))>=normalRange.w) continue;
                vec3 scaled=delta/normalRange.w; float squared=dot(scaled,scaled); window=max(1.0-squared*squared,0.0); window*=window;
            }
            float emitterCosine=uint(source.w)==AreaLight?clamp(dot(normalRange.xyz,-wi),0.0,1.0):1.0;
            attenuation=energy.w*emitterCosine*window/max(distance*distance,LightingOptions.w);
        }
        invalid=!finite3(wi) || !finite1(attenuation); if (invalid) continue;
        float cosine=clamp(dot(materialNormal.xyz,wi),0.0,1.0);
        if (cosine==0.0 || attenuation==0.0 || dot(geometric,wi)<=0.0) continue;
        if (LightData.Vectors[base+SampleShadow].x!=0u)
        {
            DistanceRay ray; ray.OriginMinimum=vec4(origin,0.0); ray.DirectionMaximum=vec4(wi,LightingOptions.z);
            if (uint(source.w)!=DirectionalLight)
            {
                vec3 delta=source.xyz-origin; float distance=vectorLength(delta);
                if (distance>0.0) ray.DirectionMaximum=vec4(delta/distance,min(LightingOptions.z,max(0.0,distance-LightingOptions.x))); else ray.DirectionMaximum.w=0.0;
            }
            if (ray.DirectionMaximum.w>0.0)
            {
                ++metadata.y; DistanceHit visibility=traceDistanceScene(ray);
                if (visibility.Metadata.z==Surface) { ++metadata.z; continue; }
                if (visibility.Metadata.z==Unresolved) { ++metadata.w; continue; }
                attenuation*=visibility.Medium.w;
            }
        }
        irradiance+=energy.xyz*(cosine*attenuation);
    }
#if defined(EVANS_LUMEN_RADIOSITY)
    vec3 indirect=vec3(0),normal=unitVector(materialNormal.xyz,geometric);
    for (uint rayIndex=0u;rayIndex<BounceWork.x && !invalid;++rayIndex)
    {
        ++metadata.y; vec3 direction=bounceCosineSample(rayIndex,index,normal); if (dot(direction,geometric)<=0.0) continue;
        DistanceRay ray; ray.OriginMinimum=vec4(origin,0); ray.DirectionMaximum=vec4(direction,BounceSky.w);
        DistanceHit hit=traceLocalDistanceScene(ray,0u);
        if (hit.Metadata.z==Miss) { indirect+=lumenEnvironmentRadiance(BounceSky.xyz,EnvironmentAxis,EnvironmentPeak.xyz,direction)*hit.Medium.w+hit.Medium.xyz; continue; }
        if (hit.Metadata.z!=Surface) { ++metadata.w; continue; }
        vec3 incident; uint status=bounceResolve(hit,incident);
        if (status==InvalidLighting) { invalid=true; continue; }
        if (status!=CompleteLighting) ++metadata.w;
        indirect+=incident*hit.Medium.w+hit.Medium.xyz;
    }
    irradiance+=indirect*(BouncePi/float(BounceWork.x));
#endif
    vec4 albedo=texelFetch(LightingAlbedo,native,0); vec3 radiance=texelFetch(LightingEmission,native,0).xyz+albedo.xyz*(1.0-albedo.w)*irradiance*LightingPrecision.y;
    invalid=invalid || !finite3(radiance); metadata.x=invalid?InvalidLighting:(metadata.w!=0u?UnresolvedLighting:CompleteLighting);
    storeLighting(index,vec4(invalid?vec3(0.0):max(radiance,vec3(0.0)),1.0),metadata);
}
#endif
#elif defined(EVANS_LUMEN_READ_STAGE)
layout(std140,binding=0) uniform ExportFrame { uvec4 Extent; };
uniform highp sampler2D AlbedoImage,NormalImage,EmissionImage;
layout(std430,binding=0) writeonly buffer MaterialExport { vec4 Values[]; } Export;
const uint MaterialPlanes=3u;
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Extent.z; if (index>=Extent.x*Extent.y) return;
    ivec2 pixel=ivec2(index%Extent.x,Extent.y-1u-index/Extent.x); uint first=index*MaterialPlanes;
    Export.Values[first]=texelFetch(AlbedoImage,pixel,0); Export.Values[first+1u]=texelFetch(NormalImage,pixel,0);
    Export.Values[first+2u]=texelFetch(EmissionImage,pixel,0);
}
#else
layout(std140,binding=0) uniform CaptureFrame
{
    vec4 WorldRows[4],NormalRows[4],ClipRows[4];
    vec4 BaseColor,EmissionRoughness,Surface;
    uvec4 Images;
};
const float NormalEpsilon=1.0e-10,ProjectionEpsilon=1.0e-7,UvDeterminantEpsilon=1.0e-8;
vec3 unitNormal(vec3 value,vec3 fallback)
{ float square=dot(value,value); return square>NormalEpsilon?value*inversesqrt(square):fallback; }
#if defined(EVANS_VERTEX_STAGE)
layout(location=0) in vec3 Position;
layout(location=1) in vec3 Normal;
layout(location=2) in vec4 Color;
layout(location=3) in vec2 TexCoord;
out vec3 WorldPosition,WorldNormal;
out vec4 VertexColor;
out vec2 UV;
const float SrgbBreakpoint=0.04045,SrgbLinearDivisor=12.92,SrgbOffset=0.055,SrgbScale=1.055,SrgbPower=2.4;
vec3 linearVertex(vec3 value)
{
    value=clamp(value,0.0,1.0); vec3 linear=value/SrgbLinearDivisor,curved=pow((value+SrgbOffset)/SrgbScale,vec3(SrgbPower));
    return mix(curved,linear,lessThanEqual(value,vec3(SrgbBreakpoint)));
}
void main()
{
    vec4 local=vec4(Position,1),world=vec4(dot(WorldRows[0],local),dot(WorldRows[1],local),dot(WorldRows[2],local),dot(WorldRows[3],local));
    // WorldRows already subtracts the capture origin; project only afterwards.
    gl_Position=vec4(dot(ClipRows[0],world),dot(ClipRows[1],world),dot(ClipRows[2],world),dot(ClipRows[3],world));
    gl_Position.z=gl_Position.z*2.0-gl_Position.w;
    WorldPosition=world.xyz/max(abs(world.w),ProjectionEpsilon);
    WorldNormal=vec3(dot(NormalRows[0].xyz,Normal),dot(NormalRows[1].xyz,Normal),dot(NormalRows[2].xyz,Normal));
    VertexColor=vec4(linearVertex(Color.rgb),Color.a); UV=vec2(TexCoord.x,1.0-TexCoord.y);
}
#else
in vec3 WorldPosition,WorldNormal;
in vec4 VertexColor;
in vec2 UV;
uniform highp sampler2D DiffuseImage,NormalImage,OpacityImage;
layout(location=0) out vec4 AlbedoMetallic;
layout(location=1) out vec4 NormalDepth;
layout(location=2) out vec4 EmissionMaterial;
vec3 captureNormal()
{
    vec3 normal=unitNormal(WorldNormal,vec3(0,0,1)); if (!gl_FrontFacing) normal=-normal;
    vec3 dx=dFdx(WorldPosition),dy=dFdy(WorldPosition); vec2 du=dFdx(UV),dv=dFdy(UV);
    float determinant=du.x*dv.y-du.y*dv.x;
    if (Images.y==0u || abs(determinant)<UvDeterminantEpsilon) return normal;
    vec3 tangent=(dx*dv.y-dy*du.y)/determinant,sourceBitangent=(dy*du.x-dx*dv.x)/determinant;
    tangent-=normal*dot(normal,tangent); if (dot(tangent,tangent)<=NormalEpsilon) return normal;
    tangent=normalize(tangent); vec3 bitangent=cross(normal,tangent); if (dot(bitangent,sourceBitangent)<0.0) bitangent=-bitangent;
    vec3 sampled=texture(NormalImage,UV).xyz*2.0-1.0;
    return unitNormal(tangent*sampled.x+bitangent*sampled.y+normal*sampled.z,normal);
}
void main()
{
    vec3 normal=captureNormal();
    vec4 diffuse=Images.x!=0u?texture(DiffuseImage,UV):vec4(1);
    float opacity=Images.z!=0u?texture(OpacityImage,UV).r:1.0;
    float coverage=clamp(VertexColor.a*BaseColor.a*diffuse.a*opacity,0.0,1.0); if (coverage<Surface.y) discard;
    AlbedoMetallic=vec4(clamp(VertexColor.rgb*BaseColor.rgb*diffuse.rgb,0.0,1.0),Surface.x);
    NormalDepth=vec4(normal,clamp(gl_FragCoord.z,0.0,1.0)); EmissionMaterial=EmissionRoughness;
}
#endif
#endif
