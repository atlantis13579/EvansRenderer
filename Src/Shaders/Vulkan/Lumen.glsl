#version 460
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
#extension GL_EXT_ray_query : require
#endif
struct ProbeOrigin { vec4 PositionMinimum; uvec4 Metadata; };
const uint LumenScreenSurface=3u,LumenWorldCache=4u,LumenHairPointerShift=2u;
vec3 lumenEnvironmentRadiance(vec3 baseSky,vec4 axisPower,vec3 peak,vec3 direction)
{
    if (!any(greaterThan(peak,vec3(0.0)))) return baseSky;
    float cosine=clamp(dot(axisPower.xyz,direction),0.0,1.0);
    return baseSky+peak*pow(cosine,axisPower.w);
}
const uint LumenProbeNeighbors=4u,LumenLinkWordBits=32u;
uint probeLinkWords(uint slots) { return (LumenProbeNeighbors*slots+LumenLinkWordBits-1u)/LumenLinkWordBits; }
vec3 probeDirection(uint sampleIndex,uint sampleCount,uint identifier,uint sequence)
{
    const float Pi=3.14159265358979323846,UnsignedFraction=1.0/4294967296.0;
    uint phase=bitfieldReverse(sampleIndex)+bitfieldReverse(identifier^sequence);
    float height=1.0-2.0*(float(sampleIndex)+0.5)/float(sampleCount);
    float azimuth=2.0*Pi*(float(phase)*UnsignedFraction),radius=sqrt(max(0.0,1.0-height*height));
    return vec3(radius*cos(azimuth),radius*sin(azimuth),height);
}
// Native Vulkan card capture. Row-major host matrices and the explicit Y flip
// preserve the renderer's canonical projection without translating another API.
// Clarberg's equal-area square map provides a stable angular address for each
// directional world-cache texel. It is independent of screen-probe jitter.
vec3 cacheTexelDirection(uint index,uint side)
{
    const float Pi=3.14159265358979323846;
    side=max(side,1u); vec2 xy=(vec2(index%side,index/side)+0.5)*2.0/float(side)-1.0;
    float folded=1.0-abs(xy.x)-abs(xy.y),r=1.0-abs(folded);
    float phi=r>0.0?(Pi/4.0)*(1.0+(abs(xy.y)-abs(xy.x))/r):0.0;
    float radial=r*sqrt(max(0.0,2.0-r*r));
    vec2 signs=vec2(xy.x<0.0?-1.0:1.0,xy.y<0.0?-1.0:1.0);
    return vec3(radial*abs(vec2(cos(phi),sin(phi)))*signs,(folded<0.0?-1.0:1.0)*(1.0-r*r));
}
#if defined(EVANS_LUMEN_REFLECTION_RESOLVE_STAGE)
// Vulkan owns its material-aware specular reconstruction. Smith's correlated
// visibility divides out the GGX visible-normal density (Heitz, JCGT 2018).
// Unknown links are dark samples, never removed from the normalization weight.
layout(std140,row_major,set=0,binding=0) uniform ReflectionResolveParameters
{
    mat4 Inverse; uvec4 Viewport,Work; vec4 Trace,Numeric,Acceptance,Filter,Sky; uvec4 Dispatch; vec4 EnvironmentAxis,EnvironmentPeak;
} Resolve;
layout(set=0,binding=1) uniform sampler2D ResolveDepth;
layout(set=0,binding=2) uniform sampler2D ResolveNormals;
layout(set=0,binding=3) uniform sampler2D ResolveBaseColor;
layout(set=0,binding=4) uniform sampler2D ResolveMetalness;
layout(set=0,binding=5) uniform sampler2D ResolveSpecular;
struct ResolvePath { vec4 Origin,Direction; };
struct ResolveIntersection { vec4 Position,Normal,Uncertainty; uvec4 Metadata; vec4 Medium; };
struct ResolveIncident { ResolveIntersection Hit; vec4 Radiance; uvec4 Cache; };
layout(std430,set=0,binding=6) readonly buffer RayStorage { ResolvePath Values[]; } Paths;
layout(std430,set=0,binding=7) readonly buffer IncidentStorage { ResolveIncident Values[]; } Lighting;
layout(std430,set=0,binding=8) writeonly buffer ReflectionStorage { vec4 Values[]; } Output;
const uint ResolvePBR=3u,ResolveFiniteMiss=0u,ResolveMeshHit=1u,ResolveScreenHit=3u,ResolveKnown=1u,ResolvePartial=2u;
const float ResolvePartialConfidence=0.5;
bool resolveFinite(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
bool resolvePosition(uvec2 pixel,float depth,out vec3 position)
{
    vec2 uv=(vec2(pixel)+0.5)/vec2(Resolve.Viewport.xy); vec4 h=Resolve.Inverse*vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1);
    position=vec3(0); if (!resolveFinite(h) || abs(h.w)<=Resolve.Numeric.y) return false;
    position=h.xyz/h.w; return resolveFinite(vec4(position,0)) && all(lessThanEqual(abs(position),vec3(Resolve.Numeric.x)));
}
bool resolveSurface(uvec2 pixel,out vec3 position,out vec3 normal,out float roughness)
{
    float z=texelFetch(ResolveDepth,ivec2(pixel),0).x; vec4 material=texelFetch(ResolveNormals,ivec2(pixel),0);
    position=normal=vec3(0); roughness=material.w;
    if (!resolveFinite(vec4(z)) || z<0.0 || z>=1.0 || !resolveFinite(material) || roughness<0.0 || roughness>=Resolve.Trace.z || dot(material.xyz,material.xyz)<=Resolve.Numeric.z) return false;
    normal=normalize(material.xyz); return resolvePosition(pixel,z,position);
}
float resolveLambda(float cosine,float alpha)
{
    float square=max(cosine*cosine,Resolve.Filter.y*Resolve.Filter.y);
    return (sqrt(1.0+alpha*alpha*max(0.0,1.0-square)/square)-1.0)*0.5;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Resolve.Dispatch.y; if (index>=Resolve.Dispatch.z) return;
    Output.Values[index]=vec4(0); uvec2 pixel=uvec2(index%Resolve.Viewport.x,index/Resolve.Viewport.x);
    vec3 position,normal; float roughness; if (!resolveSurface(pixel,position,normal,roughness)) return;
    vec4 base=texelFetch(ResolveBaseColor,ivec2(pixel),0),metal=texelFetch(ResolveMetalness,ivec2(pixel),0),specular=texelFetch(ResolveSpecular,ivec2(pixel),0);
    if (!resolveFinite(base) || !resolveFinite(metal) || !resolveFinite(specular)) return;
    vec3 f0=Resolve.Dispatch.x==ResolvePBR?mix(vec3(Resolve.Filter.w),clamp(base.xyz,0.0,1.0),clamp(metal.w,0.0,1.0)):clamp(specular.xyz,0.0,1.0);
    ivec2 center=ivec2(pixel/Resolve.Viewport.z); int radius=int(Resolve.Filter.x); vec3 sum=vec3(0); float weights=0.0; bool complete=true;
    for (int y=-radius;y<=radius;++y) for (int x=-radius;x<=radius;++x)
    {
        ivec2 cell=center+ivec2(x,y); if (any(lessThan(cell,ivec2(0))) || cell.x>=int(Resolve.Viewport.w) || cell.y>=int(Resolve.Work.x)) continue;
        uint rayIndex=uint(cell.y)*Resolve.Viewport.w+uint(cell.x); ResolvePath path=Paths.Values[rayIndex];
        if (!resolveFinite(path.Direction) || dot(path.Direction.xyz,path.Direction.xyz)<=Resolve.Numeric.z) continue;
        uvec2 anchor=min(uvec2(cell)*Resolve.Viewport.z+Resolve.Viewport.z/2u,Resolve.Viewport.xy-1u);
        vec3 source,n; float r; if (!resolveSurface(anchor,source,n,r) || dot(n,normal)<Resolve.Acceptance.x || abs(r-roughness)>Resolve.Acceptance.w) continue;
        vec3 delta=source-position; float plane=max(abs(dot(delta,normal)),abs(dot(delta,n)));
        if (isnan(plane) || isinf(plane) || plane>Resolve.Acceptance.y+Resolve.Acceptance.z*length(delta)) continue;
        vec2 distance=(vec2(anchor)-vec2(pixel))/float(Resolve.Viewport.z); float weight=1.0/(1.0+dot(distance,distance)); weights+=weight;
        if (path.Direction.w<=0.0) continue; // Valid zero samples still contribute to weights.
        vec3 nearPosition,midPosition;
        if (!resolvePosition(anchor,0.0,nearPosition) || !resolvePosition(anchor,0.5,midPosition)) { complete=false; continue; }
        vec3 view=normalize(nearPosition-midPosition),light=path.Direction.xyz; float noV=dot(view,n),noL=dot(light,n);
        if (!resolveFinite(vec4(view,0)) || noV<=0.0 || noL<=0.0) { complete=false; continue; }
        ResolveIncident incoming=Lighting.Values[rayIndex]; vec3 radiance=vec3(0);
        if (incoming.Hit.Metadata.z==ResolveFiniteMiss) radiance=lumenEnvironmentRadiance(Resolve.Sky.xyz,Resolve.EnvironmentAxis,Resolve.EnvironmentPeak.xyz,light);
        else if ((incoming.Hit.Metadata.z==ResolveMeshHit || incoming.Hit.Metadata.z==ResolveScreenHit) && (incoming.Cache.z==ResolveKnown || incoming.Cache.z==ResolvePartial))
        { radiance=incoming.Radiance.xyz; complete=complete && incoming.Cache.z==ResolveKnown; }
        else complete=false;
        radiance=radiance*incoming.Hit.Medium.w+incoming.Hit.Medium.xyz;
        if (!resolveFinite(vec4(radiance,0)) || any(lessThan(radiance,vec3(0)))) { complete=false; continue; }
        float alpha=max(r*r,Resolve.Trace.w),lambdaV=resolveLambda(noV,alpha),lambdaL=resolveLambda(noL,alpha);
        float masking=r==0.0?1.0:(1.0+lambdaV)/(1.0+lambdaV+lambdaL);
        float complement=1.0-clamp(dot(view,normalize(view+light)),0.0,1.0),square=complement*complement;
        vec3 fresnel=f0+(1.0-f0)*square*square*complement; sum+=radiance*fresnel*masking*weight;
    }
    if (weights>0.0 && resolveFinite(vec4(sum,0))) Output.Values[index]=vec4(clamp(sum/weights,0.0,Resolve.Filter.z),complete?1.0:ResolvePartialConfidence);
}
#elif defined(EVANS_LUMEN_REFLECTION_RAYS_STAGE)
// Native GGX visible-normal sampling; Heitz, JCGT 2018. Zero-length rays
// retain below-horizon directions so reconstruction preserves their zero weight.
layout(std140,row_major,set=0,binding=0) uniform ReflectionRayParameters
{ mat4 Inverse; uvec4 Viewport,Work; vec4 Trace,Numeric; } Reflection;
layout(set=0,binding=1) uniform sampler2D ReflectionDepth;
layout(set=0,binding=2) uniform sampler2D ReflectionNormals;
struct ReflectionRay { vec4 OriginMinimum,DirectionMaximum; };
layout(std430,set=0,binding=3) writeonly buffer ReflectionRayOutput { ReflectionRay Values[]; } Rays;
const uint ReflectionSequenceLength=64u;
const float ReflectionPi=3.14159265358979323846,ReflectionUIntUnit=1.0/4294967296.0;
const float ReflectionRotation=0.6180339887498948482,ReflectionAxisLimit=0.9;
bool reflectionFinite(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
bool reflectionPosition(uvec2 pixel,float depth,out vec3 p)
{
    vec2 uv=(vec2(pixel)+0.5)/vec2(Reflection.Viewport.xy); vec4 h=Reflection.Inverse*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1);
    p=vec3(0); if (!reflectionFinite(h) || abs(h.w)<=Reflection.Numeric.y) return false;
    p=h.xyz/h.w; return reflectionFinite(vec4(p,0)) && all(lessThanEqual(abs(p),vec3(Reflection.Numeric.x)));
}
vec3 reflectionVisibleNormal(vec3 eye,float alpha,vec2 xi)
{
    vec3 stretched=normalize(vec3(eye.xy*alpha,eye.z)); float radial=dot(stretched.xy,stretched.xy);
    vec3 first=radial>0.0?vec3(-stretched.y,stretched.x,0)/sqrt(radial):vec3(1,0,0),second=cross(stretched,first);
    float angle=2.0*ReflectionPi*xi.y; vec2 disk=sqrt(xi.x)*vec2(cos(angle),sin(angle));
    disk.y=mix(sqrt(max(0.0,1.0-disk.x*disk.x)),disk.y,(1.0+stretched.z)*0.5);
    vec3 h=disk.x*first+disk.y*second+sqrt(max(0.0,1.0-dot(disk,disk)))*stretched;
    return normalize(vec3(h.xy*alpha,max(0.0,h.z)));
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Reflection.Work.w; if (index>=Reflection.Work.y) return;
    ReflectionRay ray=ReflectionRay(vec4(0),vec4(0)); Rays.Values[index]=ray;
    uvec2 cell=uvec2(index%Reflection.Viewport.w,index/Reflection.Viewport.w);
    uvec2 pixel=min(cell*Reflection.Viewport.z+Reflection.Viewport.z/2u,Reflection.Viewport.xy-1u);
    float depth=texelFetch(ReflectionDepth,ivec2(pixel),0).x; vec4 material=texelFetch(ReflectionNormals,ivec2(pixel),0);
    if (!reflectionFinite(vec4(depth)) || depth<0.0 || depth>=1.0 || !reflectionFinite(material) || material.w<0.0 || material.w>=Reflection.Trace.z || dot(material.xyz,material.xyz)<=Reflection.Numeric.z) return;
    vec3 world,nearPosition,midPosition;
    if (!reflectionPosition(pixel,depth,world) || !reflectionPosition(pixel,0.0,nearPosition) || !reflectionPosition(pixel,0.5,midPosition)) return;
    vec3 normal=normalize(material.xyz),eye=normalize(nearPosition-midPosition);
    if (!reflectionFinite(vec4(eye,0)) || dot(eye,normal)<=0.0) return;
    vec3 axis=abs(normal.z)<ReflectionAxisLimit?vec3(0,0,1):vec3(0,1,0),right=normalize(cross(axis,normal)),up=cross(normal,right);
    float rotation=float(bitfieldReverse(index))*ReflectionUIntUnit;
    vec2 xi=fract(vec2((float(Reflection.Work.z%ReflectionSequenceLength)+0.5)/float(ReflectionSequenceLength),float(bitfieldReverse(Reflection.Work.z))*ReflectionUIntUnit)+vec2(rotation,fract(rotation*ReflectionRotation)));
    vec3 micro=material.w==0.0?vec3(0,0,1):reflectionVisibleNormal(vec3(dot(eye,right),dot(eye,up),dot(eye,normal)),max(material.w*material.w,Reflection.Trace.w),xi);
    vec3 halfVector=right*micro.x+up*micro.y+normal*micro.z,direction=reflect(-eye,halfVector);
    ray.OriginMinimum=vec4(world+normal*Reflection.Trace.y,0);
    ray.DirectionMaximum=vec4(direction,dot(direction,normal)>0.0?Reflection.Trace.x:0.0); Rays.Values[index]=ray;
}
#elif defined(EVANS_LUMEN_TEMPORAL_STAGE) || defined(EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE)
// Packed diffuse history is separate from scene HDR. Normal/depth rejection
// precedes bilinear history weighting; invalid current links never resurrect GI.
layout(std140,row_major,set=0,binding=0) uniform TemporalFrame
{
    mat4 Inverse,PreviousInverse; vec4 PreviousPlane; uvec4 Extent;
    vec4 Rules,Clip,Jitter,Numeric;
} HistoryFrame;
layout(set=0,binding=1) uniform sampler2D TemporalCurrent;
layout(std430,set=0,binding=5) readonly buffer TemporalInput { uvec4 Pixels[]; } OldHistory;
const float TemporalNormalFloor=1e-12;
const int TemporalFilterRadius=1;
vec4 temporalColor(uvec4 packed) { return vec4(unpackHalf2x16(packed.x),unpackHalf2x16(packed.y)); }
#if defined(EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE)
#ifdef EVANS_VERTEX_STAGE
void main()
{ vec2 corner=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2); gl_Position=vec4(corner*2.0-1.0,0.0,1.0); }
#else
layout(location=0) out vec4 Color;
void main()
{
    uvec2 pixel=uvec2(gl_FragCoord.xy); vec4 scene=texelFetch(TemporalCurrent,ivec2(pixel),0);
    vec4 indirect=temporalColor(OldHistory.Pixels[pixel.y*HistoryFrame.Extent.x+pixel.x]);
    vec3 contribution=indirect.w>0.0?indirect.xyz*HistoryFrame.Jitter.z:vec3(0.0);
    Color=vec4(clamp(scene.xyz+contribution,0.0,HistoryFrame.Numeric.w),scene.w);
}
#endif
#else
layout(set=0,binding=2) uniform sampler2D TemporalDepth;
layout(set=0,binding=3) uniform sampler2D TemporalNormals;
layout(set=0,binding=4) uniform sampler2D TemporalMotion;
layout(std430,set=0,binding=6) writeonly buffer TemporalOutput { uvec4 Pixels[]; } NewHistory;
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
layout(std430,set=0,binding=7) readonly buffer SpecularCurrent { vec4 Pixels[]; } Specular;
#endif
vec4 temporalEstimate(uvec2 pixel)
{
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
    return Specular.Pixels[pixel.y*HistoryFrame.Extent.x+pixel.x];
#else
    return texelFetch(TemporalCurrent,ivec2(pixel),0);
#endif
}
bool temporalFinite(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
bool temporalScalar(float value) { return !isnan(value) && !isinf(value); }
vec2 temporalSign(vec2 v) { return vec2(v.x<0.0?-1.0:1.0,v.y<0.0?-1.0:1.0); }
vec3 temporalNormal(uint packed)
{
    vec2 oct=unpackSnorm2x16(packed); vec3 normal=vec3(oct,1.0-abs(oct.x)-abs(oct.y));
    if (normal.z<0.0) normal.xy=(1.0-abs(normal.yx))*temporalSign(normal.xy);
    return normalize(normal);
}
uvec4 temporalPack(vec4 color,vec3 normal,float depth)
{
    vec2 oct=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));
    if (normal.z<0.0) oct=(1.0-abs(oct.yx))*temporalSign(oct);
    return uvec4(packHalf2x16(color.xy),packHalf2x16(color.zw),packSnorm2x16(oct),floatBitsToUint(depth));
}
bool temporalSurface(uvec2 pixel,out vec3 position,out vec3 normal,out float depth)
{
    position=vec3(0.0); normal=vec3(0,0,1); depth=texelFetch(TemporalDepth,ivec2(pixel),0).r;
    vec4 surface=texelFetch(TemporalNormals,ivec2(pixel),0);
    if (!temporalScalar(depth) || depth<0.0 || depth>=1.0 || !temporalFinite(surface) || surface.w<0.0 || dot(surface.xyz,surface.xyz)<=TemporalNormalFloor) return false;
    vec2 uv=(vec2(pixel)+0.5)/vec2(HistoryFrame.Extent.xy);
    vec4 h=HistoryFrame.Inverse*vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1.0);
    if (!temporalFinite(h) || abs(h.w)<=HistoryFrame.Numeric.z) return false;
    position=h.xyz/h.w; normal=normalize(surface.xyz); return temporalFinite(vec4(position,0.0));
}
float temporalDistance(vec2 center,float depth)
{
    vec2 uv=center/vec2(HistoryFrame.Extent.xy); vec4 h=HistoryFrame.PreviousInverse*vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1.0);
    if (!temporalFinite(h) || abs(h.w)<=HistoryFrame.Numeric.z) return -1.0;
    return dot(HistoryFrame.PreviousPlane,vec4(h.xyz/h.w,1.0));
}
// Matches LumenTemporalThreads in the dispatch contract.
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*HistoryFrame.Extent.w; if (index>=HistoryFrame.Extent.z) return;
    uvec2 pixel=uvec2(index%HistoryFrame.Extent.x,index/HistoryFrame.Extent.x); vec3 position,normal; float depth;
    vec4 current=temporalEstimate(pixel);
    if (!temporalSurface(pixel,position,normal,depth) || !temporalFinite(current) || current.w<=0.0)
    { NewHistory.Pixels[index]=temporalPack(vec4(0.0),vec3(0,0,1),1.0); return; }
#if defined(EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE)
    if (current.w<1.0) { NewHistory.Pixels[index]=temporalPack(vec4(clamp(current.xyz,0.0,HistoryFrame.Numeric.w),current.w),normal,depth); return; }
#endif
    current=vec4(clamp(current.xyz,0.0,HistoryFrame.Numeric.w),1.0); NewHistory.Pixels[index]=temporalPack(current,normal,depth);
    if (HistoryFrame.Jitter.w==0.0 || HistoryFrame.Rules.x<=0.0) return;
    vec4 motion=texelFetch(TemporalMotion,ivec2(pixel),0); if (!temporalFinite(motion) || motion.w<=0.0) return;
    vec2 previous=vec2(pixel)+0.5-motion.xy-HistoryFrame.Jitter.xy;
    if (any(lessThan(previous,vec2(0.5))) || any(greaterThan(previous,vec2(HistoryFrame.Extent.xy)-0.5))) return;
    float expected=temporalDistance(previous,1.0-motion.z); if (!temporalScalar(expected) || expected<0.0) return;
    ivec2 base=ivec2(floor(previous-0.5)); vec2 fraction=fract(previous-0.5); vec4 sum=vec4(0.0); float weights=0.0;
    for (uint tap=0u;tap<4u;++tap)
    {
        ivec2 corner=ivec2(tap&1u,tap>>1u),candidate=base+corner;
        uvec2 bounded=uvec2(clamp(candidate,ivec2(0),ivec2(HistoryFrame.Extent.xy)-1));
        uvec4 packed=OldHistory.Pixels[bounded.y*HistoryFrame.Extent.x+bounded.x]; vec4 old=temporalColor(packed); float oldDepth=uintBitsToFloat(packed.w);
        float weight=(corner.x!=0?fraction.x:1.0-fraction.x)*(corner.y!=0?fraction.y:1.0-fraction.y);
        if (weight<=0.0 || any(lessThan(candidate,ivec2(0))) || any(greaterThanEqual(candidate,ivec2(HistoryFrame.Extent.xy))) || !temporalFinite(old) ||
            old.w<1.0 || old.w>HistoryFrame.Clip.y || !temporalScalar(oldDepth) || oldDepth<0.0 || oldDepth>=1.0 || dot(normal,temporalNormal(packed.z))<HistoryFrame.Rules.z) continue;
        float optical=temporalDistance(vec2(candidate)+0.5,oldDepth);
        if (!temporalScalar(optical) || abs(optical-expected)>max(HistoryFrame.Numeric.x,abs(expected)*HistoryFrame.Rules.y)) continue;
        sum+=old*weight; weights+=weight;
    }
    if (weights<=TemporalNormalFloor) return;
    vec4 history=sum/weights; vec3 low=current.xyz,high=low;
    for (int y=-TemporalFilterRadius;y<=TemporalFilterRadius;++y) for (int x=-TemporalFilterRadius;x<=TemporalFilterRadius;++x)
    {
        uvec2 neighbor=uvec2(clamp(ivec2(pixel)+ivec2(x,y),ivec2(0),ivec2(HistoryFrame.Extent.xy)-1));
        vec3 p,n; float z; vec4 value=temporalEstimate(neighbor);
        if (value.w<=0.0 || !temporalFinite(value) || !temporalSurface(neighbor,p,n,z) || dot(normal,n)<HistoryFrame.Rules.z) continue;
        float plane=abs(dot(p-position,normal)); if (!temporalScalar(plane) || plane>HistoryFrame.Rules.w) continue;
        low=min(low,value.xyz); high=max(high,value.xyz);
    }
    vec3 center=(low+high)*0.5,extent=max((high-low)*0.5+abs(center)*HistoryFrame.Clip.x,vec3(HistoryFrame.Numeric.y));
    vec3 delta=history.xyz-center,ratio=abs(delta/extent); history.xyz=center+delta/max(1.0,max(ratio.x,max(ratio.y,ratio.z)));
    float count=min(history.w+1.0,HistoryFrame.Clip.y),blend=min(HistoryFrame.Rules.x,(count-1.0)/count);
    NewHistory.Pixels[index]=temporalPack(vec4(clamp(mix(current.xyz,history.xyz,blend),0.0,HistoryFrame.Numeric.w),count),normal,depth);
}
#endif
#elif defined(EVANS_LUMEN_PROBE_GATHER_STAGE)
#ifdef EVANS_VERTEX_STAGE
void main()
{ vec2 corner=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2); gl_Position=vec4(corner*2.0-1.0,0.0,1.0); }
#else
layout(std140,row_major,set=0,binding=0) uniform GatherFrame
{ uvec4 TraceCounts,TraceOffsets,TraceWork; vec4 TraceParameters,TraceHairSky; mat4 Inverse; uvec4 Viewport,Grid; vec4 Origin,Adaptive,Appearance,Numeric; uvec4 Model; vec4 Visibility; } Gather;
layout(set=0,binding=1) uniform sampler2D SceneInput;
layout(set=0,binding=2) uniform sampler2D DepthInput;
layout(set=0,binding=3) uniform sampler2D NormalInput;
layout(set=0,binding=4) uniform sampler2D DiffuseInput;
layout(set=0,binding=5) uniform sampler2D AmbientInput;
struct GatherProbe { vec4 Coefficients[9]; vec4 Coverage; uvec4 TraceCounts,CacheCounts,Metadata; };
layout(std430,set=0,binding=6) readonly buffer OriginInput { ProbeOrigin Values[]; } ProbePositions;
layout(std430,set=0,binding=7) readonly buffer IrradianceInput { GatherProbe Values[]; } ProbeIrradiance;
layout(std430,set=0,binding=8) readonly buffer GatherVisibility { uint Masks[]; } Links;
layout(location=0) out vec4 Color;
const uint HarmonicTerms=9u,PartialProbe=2u,PbrModel=3u;
const float GatherPi=3.14159265358979323846,CoordinateRoundoff=32.0*1.1920928955078125e-7;
bool gatherFinite(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
float gatherMagnitude(vec3 value) { return max(abs(value.x),max(abs(value.y),abs(value.z))); }
float gatherHarmonic(uint index,vec3 n)
{
    switch (index)
    {
        case 0u:return sqrt(1.0/(4.0*GatherPi));
        case 1u:return sqrt(3.0/(4.0*GatherPi))*n.y;
        case 2u:return sqrt(3.0/(4.0*GatherPi))*n.z;
        case 3u:return sqrt(3.0/(4.0*GatherPi))*n.x;
        case 4u:return sqrt(15.0/(4.0*GatherPi))*n.x*n.y;
        case 5u:return sqrt(15.0/(4.0*GatherPi))*n.y*n.z;
        case 6u:return sqrt(5.0/(16.0*GatherPi))*(3.0*n.z*n.z-1.0);
        case 7u:return sqrt(15.0/(4.0*GatherPi))*n.x*n.z;
        case 8u:return sqrt(15.0/(16.0*GatherPi))*(n.x*n.x-n.y*n.y);
    }
    return 0.0;
}
float gatherAxis(float pixel,uint extent,uint count)
{
    float side=float(Gather.Viewport.z),coordinate=pixel/side-0.5;
    if (count>1u && extent%Gather.Viewport.z!=0u)
    {
        float previous=(float(count)-1.5)*side,last=(float(count-1u)*side+float(extent))*0.5;
        if (pixel>previous) coordinate=float(count-2u)+(pixel-previous)/(last-previous);
    }
    return clamp(coordinate,0.0,float(count-1u));
}
void main()
{
    ivec2 pixel=ivec2(gl_FragCoord.xy); vec4 scene=Gather.Model.z!=0u?vec4(0.0):texelFetch(SceneInput,pixel,0); Color=scene;
    float depth=texelFetch(DepthInput,pixel,0).r; vec4 materialNormal=texelFetch(NormalInput,pixel,0);
    if (Gather.Appearance.x==0.0 || isnan(depth) || isinf(depth) || depth<0.0 || depth>=1.0 || materialNormal.w<0.0 || !gatherFinite(materialNormal) || dot(materialNormal.xyz,materialNormal.xyz)<=Gather.Numeric.x) return;
    vec2 uv=gl_FragCoord.xy/vec2(Gather.Viewport.xy); vec4 h=Gather.Inverse*vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1.0);
    if (!gatherFinite(h) || abs(h.w)<=Gather.Origin.w) return;
    vec3 world=h.xyz/h.w,normal=normalize(materialNormal.xyz);
    if (!gatherFinite(vec4(world,0.0)) || any(greaterThan(abs(world),vec3(Gather.Origin.z)))) return;
    vec2 grid=vec2(gatherAxis(gl_FragCoord.x,Gather.Viewport.x,Gather.Viewport.w),gatherAxis(gl_FragCoord.y,Gather.Viewport.y,Gather.Grid.x));
    uvec2 cell=uvec2(floor(grid)); vec2 fraction=grid-vec2(cell); vec3 sum=vec3(0.0); float weights=0.0;
    uint slots=Gather.Grid.w+1u,words=probeLinkWords(Gather.Grid.w+1u);
    for (uint tap=0u;tap<4u*slots;++tap)
    {
        uint cornerIndex=tap/slots;
        uvec2 corner=uvec2(cornerIndex&1u,cornerIndex>>1u),tile=min(cell+corner,uvec2(Gather.Viewport.w,Gather.Grid.x)-1u);
        float weight=(corner.x!=0u?fraction.x:1.0-fraction.x)*(corner.y!=0u?fraction.y:1.0-fraction.y); if (weight<=0.0) continue;
        if (Gather.Model.y!=0u && (Links.Masks[(uint(pixel.y)*Gather.Viewport.x+uint(pixel.x))*words+tap/LumenLinkWordBits]&(1u<<(tap%LumenLinkWordBits)))==0u) continue;
        uint index=(tile.y*Gather.Viewport.w+tile.x)*slots+tap%slots; ProbeOrigin origin=ProbePositions.Values[index]; GatherProbe probe=ProbeIrradiance.Values[index];
        if (origin.Metadata.y!=0u || origin.Metadata.x!=index || any(greaterThanEqual(origin.Metadata.zw,Gather.Viewport.xy)) || probe.Metadata.x!=index || probe.Metadata.y==0u || probe.Metadata.y>PartialProbe) continue;
        vec4 anchorNormal=texelFetch(NormalInput,ivec2(origin.Metadata.zw),0);
        if (!gatherFinite(anchorNormal) || anchorNormal.w<0.0 || dot(anchorNormal.xyz,anchorNormal.xyz)<=Gather.Numeric.x) continue;
        vec3 n=normalize(anchorNormal.xyz),anchor=origin.PositionMinimum.xyz-n*Gather.Origin.x; float agreement=dot(normal,n);
        if (agreement<=Gather.Appearance.y) continue;
        vec3 delta=world-anchor; float magnitude=gatherMagnitude(delta),distance=magnitude>0.0?magnitude*length(delta/magnitude):0.0;
        float error=max(abs(dot(delta,normal)),abs(dot(delta,n)));
        float tolerance=Gather.Appearance.z+Gather.Appearance.w*distance+CoordinateRoundoff*max(gatherMagnitude(world),gatherMagnitude(anchor));
        if (isnan(error) || isinf(error) || isnan(tolerance) || isinf(tolerance) || error>tolerance) continue;
        float facing=(agreement-Gather.Appearance.y)/(1.0-Gather.Appearance.y),plane=tolerance>0.0?clamp(1.0-error/tolerance,0.0,1.0):1.0;
        weight*=facing*facing*plane; vec3 value=vec3(0.0);
        for (uint term=0u;term<HarmonicTerms;++term) value+=probe.Coefficients[term].xyz*gatherHarmonic(term,normal);
        if (!gatherFinite(vec4(value,0.0))) continue;
        sum+=max(value,vec3(0.0))*weight; weights+=weight;
    }
    if (weights<=Gather.Numeric.y || !gatherFinite(vec4(sum,0.0))) return;
    vec3 albedo=clamp(texelFetch(DiffuseInput,pixel,0).xyz,0.0,1.0);
    if (Gather.Model.x==PbrModel) albedo*=(1.0-Gather.Numeric.w)*(1.0-clamp(texelFetch(AmbientInput,pixel,0).w,0.0,1.0));
    vec3 indirect=(sum/weights)*(albedo/GatherPi);
    if (!gatherFinite(vec4(indirect,0.0))) return;
    if (Gather.Model.z!=0u) { Color=vec4(clamp(indirect,0.0,Gather.Numeric.z),1.0); return; }
    Color=vec4(clamp(scene.xyz+indirect*Gather.Appearance.x,0.0,Gather.Numeric.z),scene.w);
}
#endif
#elif defined(EVANS_LUMEN_SCREEN_DEPTH_STAGE)
// Camera-relative forward-depth bounds, rather than nonlinear device Z, keep
// the rejection proof valid for general projective cameras and odd extents.
layout(std140,row_major,set=0,binding=0) uniform DepthFrame
{ mat4 Inverse; vec4 Anchor,Forward; uvec4 Source,Destination,Dispatch; vec4 Numeric; } Pyramid;
layout(set=0,binding=1) uniform sampler2D PyramidDepth;
layout(std430,set=0,binding=2) buffer PyramidBuffer { uint Words[]; } DepthTree;
const uint DepthHeaderStride=4u;
bool depthFinite(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Pyramid.Dispatch.x;
    if (index>=Pyramid.Destination.x*Pyramid.Destination.y) return;
    if (index==0u) for (uint k=0u;k<DepthHeaderStride;++k) DepthTree.Words[Pyramid.Destination.w*DepthHeaderStride+k]=Pyramid.Destination[k];
    uvec2 pixel=uvec2(index%Pyramid.Destination.x,index/Pyramid.Destination.x); float bound=Pyramid.Numeric.y;
    if (Pyramid.Destination.w==0u)
    {
        float z=texelFetch(PyramidDepth,ivec2(pixel),0).r;
        if (!isnan(z) && !isinf(z) && z>=0.0 && z<1.0)
        {
            vec2 uv=(vec2(pixel)+0.5)/vec2(Pyramid.Destination.xy);
            vec4 h=Pyramid.Inverse*vec4(uv*vec2(2.0,-2.0)+vec2(-1.0,1.0),z,1.0); bound=Pyramid.Numeric.z;
            if (depthFinite(h) && abs(h.w)>Pyramid.Numeric.x)
            {
                vec3 world=h.xyz/h.w;
                float roundoff=Pyramid.Anchor.w*dot(abs(world)+abs(Pyramid.Anchor.xyz),abs(Pyramid.Forward.xyz));
                float value=dot(world-Pyramid.Anchor.xyz,Pyramid.Forward.xyz)-roundoff;
                if (!isnan(value) && !isinf(value)) bound=value;
            }
        }
    }
    else for (uint tap=0u;tap<4u;++tap)
    {
        // Duplicate odd edges: their repeated minima are identical, while
        // every address remains legal even for speculatively evaluated lanes.
        uvec2 child=min(pixel*2u+uvec2(tap&1u,tap>>1u),max(Pyramid.Source.xy,uvec2(1u))-1u);
        bound=min(bound,uintBitsToFloat(DepthTree.Words[Pyramid.Source.z+child.y*Pyramid.Source.x+child.x]));
    }
    DepthTree.Words[Pyramid.Destination.z+index]=floatBitsToUint(bound);
}
#elif defined(EVANS_LUMEN_SCREEN_TRACE_STAGE)
// Screen-first radiance with clipped, perspective-correct DDA intervals.
// McGuire and Mara, JCGT 2014; Lumen's screen/world hierarchy: Wright et al., 2022.
layout(std140,row_major,set=0,binding=0) uniform ScreenRayFrame
{ mat4 Projection,Inverse; uvec4 Work,Viewport; vec4 Anchor,Forward,Limits; } RayFrame;
struct ScreenRayHit { vec4 PositionDistance,NormalResidual,Uncertainty; uvec4 Metadata; vec4 Medium; };
struct ScreenRayLight { ScreenRayHit Hit; vec4 RadianceCoverage; uvec4 Cache; };
layout(std430,set=0,binding=1) readonly buffer ScreenRaySeeds { ProbeOrigin Values[]; } Seeds;
layout(std430,set=0,binding=2) writeonly buffer ScreenRayHits { ScreenRayHit Values[]; } Hits;
layout(std430,set=0,binding=3) writeonly buffer ScreenRayResults { ScreenRayLight Values[]; } Results;
layout(set=0,binding=4) uniform sampler2D RayDepth;
layout(set=0,binding=5) uniform sampler2D RayNormals;
layout(set=0,binding=6) uniform sampler2D RayRadiance;
layout(std430,set=0,binding=7) readonly buffer RayDepthPyramid { uint Words[]; } RayTree;
const uint ScreenIntersection=3u,KnownRadiance=1u,NoSurfaceCard=0xffffffffu;
const float NormalMinimum=1e-12,ParallelMinimum=1e-7,PixelCenter=0.5,ClearRayDepth=1.0;
bool rayFinite(vec4 x) { return !any(isnan(x)) && !any(isinf(x)); }
bool clipScreenSegment(float a,float b,inout float enter,inout float exit)
{
    if (max(a,b)<0.0) return false;
    if (a<0.0) enter=max(enter,a/(a-b));
    if (b<0.0) exit=min(exit,a/(a-b));
    return enter<=exit;
}
bool traceScreenRay(ProbeOrigin seed,vec3 direction,out ScreenRayHit answer,out vec3 radiance)
{
    answer=ScreenRayHit(vec4(0.0),vec4(0.0),vec4(0.0),uvec4(0u),vec4(0,0,0,1)); radiance=vec3(0.0);
    vec3 origin=seed.PositionMinimum.xyz,from=origin+direction*seed.PositionMinimum.w,to=origin+direction*RayFrame.Limits.x;
    vec4 a=RayFrame.Projection*vec4(from,1.0),b=RayFrame.Projection*vec4(to,1.0); if (!rayFinite(a) || !rayFinite(b)) return false;
    float enter=0.0,exit=1.0;
    if (!clipScreenSegment(a.x+a.w,b.x+b.w,enter,exit) || !clipScreenSegment(a.w-a.x,b.w-b.x,enter,exit) ||
        !clipScreenSegment(a.y+a.w,b.y+b.w,enter,exit) || !clipScreenSegment(a.w-a.y,b.w-b.y,enter,exit) ||
        !clipScreenSegment(a.z,b.z,enter,exit) || !clipScreenSegment(a.w-a.z,b.w-b.z,enter,exit) ||
        !clipScreenSegment(a.w-RayFrame.Limits.w,b.w-RayFrame.Limits.w,enter,exit)) return false;
    vec4 h0=mix(a,b,enter),h1=mix(a,b,exit); float k0=1.0/h0.w,k1=1.0/h1.w;
    vec3 q0=mix(from,to,enter)*k0,q1=mix(from,to,exit)*k1;
    vec2 p0=(h0.xy*k0*vec2(0.5,-0.5)+0.5)*vec2(RayFrame.Viewport.xy);
    vec2 p1=(h1.xy*k1*vec2(0.5,-0.5)+0.5)*vec2(RayFrame.Viewport.xy);
    vec2 extent=abs(p1-p0); float majorLength=max(extent.x,extent.y); if (majorLength<1.0) return false;
    float advance=RayFrame.Limits.z/majorLength,forwardRate=dot(direction,RayFrame.Forward.xyz);
    uint cursor=0u,mip=0u;
    for (uint visit=0u;visit<RayFrame.Viewport.z;++visit)
    {
        uint step=cursor++;
        float begin=float(step)*advance; if (begin>=1.0) break; float end=min(begin+advance,1.0);
        ivec2 pixel=ivec2(floor(mix(p0,p1,(begin+end)*0.5)));
        if (any(lessThan(pixel,ivec2(0))) || any(greaterThanEqual(pixel,ivec2(RayFrame.Viewport.xy)))) return false;
        if (all(equal(uvec2(pixel),seed.Metadata.zw))) continue;
        if (RayFrame.Viewport.w!=0u)
        {
            uvec2 cell=uvec2(pixel)>>mip; vec2 motion=p1-p0;
            vec2 lowerPixel=vec2(cell<<mip),upperPixel=min(lowerPixel+float(1u<<mip),vec2(RayFrame.Viewport.xy));
            float boundary=1.0;
            if (motion.x!=0.0) boundary=min(boundary,((motion.x>0.0?upperPixel.x:lowerPixel.x)-p0.x)/motion.x);
            if (motion.y!=0.0) boundary=min(boundary,((motion.y>0.0?upperPixel.y:lowerPixel.y)-p0.y)/motion.y);
            uint stop=uint(clamp(ceil(boundary/advance-PixelCenter),float(step+1u),ceil(1.0/advance)));
            float tailStart=float(stop-1u)*advance,tailFinish=min(tailStart+advance,1.0);
            ivec2 tail=ivec2(floor(mix(p0,p1,(tailStart+tailFinish)*0.5)));
            if (any(lessThan(tail,ivec2(0))) || any(greaterThanEqual(tail,ivec2(RayFrame.Viewport.xy))) || any(notEqual(uvec2(tail)>>mip,cell)))
            { stop=step+1u; tailFinish=end; }

            {
                const uint HeaderWords=4u; uint header=mip*HeaderWords;
                float nearest=uintBitsToFloat(RayTree.Words[RayTree.Words[header+2u]+cell.y*RayTree.Words[header]+cell.x]);
                vec3 firstPoint=mix(q0,q1,begin)/mix(k0,k1,begin),lastPoint=mix(q0,q1,tailFinish)/mix(k0,k1,tailFinish);
                float upper=max(dot(firstPoint-origin,direction)*forwardRate,dot(lastPoint-origin,direction)*forwardRate)+dot(origin-RayFrame.Anchor.xyz,RayFrame.Forward.xyz);
                float allowance=RayFrame.Anchor.w*dot(abs(firstPoint)+abs(lastPoint)+2.0*(abs(origin)+abs(RayFrame.Anchor.xyz)),vec3(1.0));
                if (!isnan(upper+allowance) && !isinf(upper+allowance) && upper+allowance<nearest)
                { cursor=stop; mip=min(mip+1u,RayFrame.Viewport.w-1u); continue; }
            }
            if (mip>0u) { --mip; cursor=step; continue; }
        }
        float depth=texelFetch(RayDepth,pixel,0).r; if (isnan(depth) || isinf(depth) || depth<0.0 || depth>=ClearRayDepth) continue;
        vec2 uv=(vec2(pixel)+PixelCenter)/vec2(RayFrame.Viewport.xy);
        vec4 unprojected=RayFrame.Inverse*vec4(uv*vec2(2.0,-2.0)+vec2(-1.0,1.0),depth,1.0);
        if (!rayFinite(unprojected) || abs(unprojected.w)<=RayFrame.Limits.w) return false;
        vec3 surface=unprojected.xyz/unprojected.w;
        float first=dot(mix(q0,q1,begin)/mix(k0,k1,begin)-origin,direction);
        float last=dot(mix(q0,q1,end)/mix(k0,k1,end)-origin,direction);
        float target=dot(surface-origin,RayFrame.Forward.xyz),maximum=max(first*forwardRate,last*forwardRate);
        if (!rayFinite(vec4(first,last,target,maximum))) return false;
        if (maximum<target) continue;
        if (min(first*forwardRate,last*forwardRate)>target+RayFrame.Limits.y) return false;
        vec3 normal=texelFetch(RayNormals,pixel,0).xyz;
        if (!rayFinite(vec4(normal,0.0)) || dot(normal,normal)<=NormalMinimum || dot(normal,direction)>=0.0) return false;
        float travel=abs(forwardRate)>ParallelMinimum?clamp(target/forwardRate,min(first,last),max(first,last)):(first+last)*0.5;
        if (travel<=seed.PositionMinimum.w || travel>RayFrame.Limits.x) return false;
        radiance=texelFetch(RayRadiance,pixel,0).xyz; if (!rayFinite(vec4(radiance,0.0))) return false;
        radiance=max(radiance,vec3(0.0)); answer.PositionDistance=vec4(surface,travel); answer.NormalResidual=vec4(normalize(normal),0.0);
        answer.Metadata=uvec4(uvec2(pixel),ScreenIntersection,visit+1u); return true;
    }
    return false;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*RayFrame.Work.w; if (index>=RayFrame.Work.x) return;
    ProbeOrigin seed=Seeds.Values[index/RayFrame.Work.y]; ScreenRayLight result; result.Hit.Medium=vec4(0,0,0,1);
    result.Hit=ScreenRayHit(vec4(0.0),vec4(0.0),vec4(0.0),uvec4(NoSurfaceCard,NoSurfaceCard,0u,0u),vec4(0,0,0,1));
    result.RadianceCoverage=vec4(0.0); result.Cache=uvec4(NoSurfaceCard,0u,0u,0u);
    ScreenRayHit hit; hit.Medium=vec4(0,0,0,1); vec3 light;
    vec3 direction=probeDirection(index%RayFrame.Work.y,RayFrame.Work.y,seed.Metadata.x,RayFrame.Work.z);
    if (RayFrame.Forward.w!=0.0)
    { vec4 reflected=uintBitsToFloat(seed.Metadata); direction=reflected.xyz; seed.Metadata.y=reflected.w>0.0?0u:1u; }
    if (seed.Metadata.y==0u && traceScreenRay(seed,direction,hit,light))
    { result.Hit=hit; result.RadianceCoverage=vec4(light,1.0); result.Cache.z=KnownRadiance; }
    Hits.Values[index]=result.Hit; Results.Values[index]=result;
}
#elif defined(EVANS_LUMEN_SCREEN_PROBES_STAGE)
// Canonical top-left coordinates match Vulkan's corrected raster viewport.
// Stable quadtree slots represent discontinuities rejected by every ancestor.
layout(std140,row_major,set=0,binding=0) uniform ScreenFrame
{ mat4 Inverse; uvec4 Viewport,Grid; vec4 Parameters,Adaptive; } Screen;
layout(set=0,binding=1) uniform sampler2D ScreenDepth;
layout(set=0,binding=2) uniform sampler2D ScreenNormal;
layout(std430,set=0,binding=3) writeonly buffer ScreenOutput { ProbeOrigin Values[]; } Placement;
const uint NoPixel=0xffffffffu,BackgroundProbe=1u,InvalidProbe=2u;
const uint ScreenDepthLimit=2u,ScreenBranching=4u;
const float ClearDepth=1.0,MinimumNormalLengthSquared=1e-12;
float screenMagnitude(vec3 point) { return max(abs(point.x),max(abs(point.y),abs(point.z))); }
bool screenSurface(uvec2 pixel,out vec3 point,out vec3 normal)
{
    point=normal=vec3(0.0); vec2 uv=(vec2(pixel)+0.5)/vec2(Screen.Viewport.xy);
    vec4 h=Screen.Inverse*vec4(uv*vec2(2,-2)+vec2(-1,1),texelFetch(ScreenDepth,ivec2(pixel),0).r,1.0);
    if (any(isnan(h)) || any(isinf(h)) || abs(h.w)<=Screen.Parameters.w) return false;
    point=h.xyz/h.w; normal=normalize(texelFetch(ScreenNormal,ivec2(pixel),0).xyz);
    return !any(isnan(point)) && !any(isinf(point)) && all(lessThanEqual(abs(point)+Screen.Parameters.x,vec3(Screen.Parameters.z)));
}
uvec2 selectScreenPixel(uvec2 begin,uvec2 end,uint count,vec3 parents[ScreenDepthLimit],vec3 parentNormals[ScreenDepthLimit],out bool found)
{
    uint nearest=NoPixel; uvec2 chosen=uvec2(0u);
    for (uint y=begin.y;y<end.y;++y) for (uint x=begin.x;x<end.x;++x)
    {
        uvec2 pixel=uvec2(x,y); float depth=texelFetch(ScreenDepth,ivec2(pixel),0).r;
        vec4 material=texelFetch(ScreenNormal,ivec2(pixel),0);
        if (isnan(depth) || isinf(depth) || depth<0.0 || depth>=ClearDepth || any(isnan(material)) || any(isinf(material)) || material.w<0.0 || dot(material.xyz,material.xyz)<=MinimumNormalLengthSquared) continue;
        if (count!=0u)
        {
            vec3 point,normal; if (!screenSurface(pixel,point,normal)) continue;
            bool represented=false;
            for (uint ancestor=0u;ancestor<count;++ancestor)
            {
                vec3 separation=point-parents[ancestor]; float scale=screenMagnitude(separation),distance=scale>0.0?scale*length(separation/scale):0.0;
                float planeError=max(abs(dot(separation,normal)),abs(dot(separation,parentNormals[ancestor])));
                float tolerance=Screen.Adaptive.y+Screen.Adaptive.z*distance+Screen.Adaptive.w*max(screenMagnitude(point),screenMagnitude(parents[ancestor]));
                represented=represented || (dot(normal,parentNormals[ancestor])>Screen.Adaptive.x && planeError<=tolerance);
            }
            if (represented) continue;
        }
        ivec2 delta=ivec2(pixel*2u+1u)-ivec2(begin+end); uint metric=uint(delta.x*delta.x+delta.y*delta.y);
        if (metric<nearest) { nearest=metric; chosen=pixel; }
    }
    found=nearest!=NoPixel; return chosen;
}
layout(local_size_x=64) in;
void main()
{
    uint probe=gl_GlobalInvocationID.x; if (probe>=Screen.Grid.y) return;
    uint slots=Screen.Grid.w+1u,tile=probe/slots,slot=probe%slots;
    uvec2 begin=uvec2(tile%Screen.Viewport.w,tile/Screen.Viewport.w)*Screen.Viewport.z;
    uvec2 end=min(begin+Screen.Viewport.z,Screen.Viewport.xy); bool found;
    vec3 parents[ScreenDepthLimit],parentNormals[ScreenDepthLimit];
    for (uint clear=0u;clear<ScreenDepthLimit;++clear) { parents[clear]=vec3(0.0); parentNormals[clear]=vec3(0.0); }
    uint path[ScreenDepthLimit],depth=0u;
    for (uint node=slot;node!=0u;node=(node-1u)/ScreenBranching) path[depth++]=(node-1u)%ScreenBranching;
    uvec2 pixel=selectScreenPixel(begin,end,0u,parents,parentNormals,found);
    ProbeOrigin value; value.PositionMinimum=vec4(0.0); value.Metadata=uvec4(probe,BackgroundProbe,0u,0u);
    vec3 point=vec3(0.0),normal=vec3(0.0);
    for (uint level=0u;level<depth;++level)
    {
        if (!found || !screenSurface(pixel,point,normal)) { Placement.Values[probe]=value; return; }
        parents[level]=point; parentNormals[level]=normal;
        uvec2 middle=(begin+end)/2u; uint quadrant=path[depth-1u-level];
        begin=uvec2((quadrant&1u)!=0u?middle.x:begin.x,(quadrant&2u)!=0u?middle.y:begin.y);
        end=uvec2((quadrant&1u)!=0u?end.x:middle.x,(quadrant&2u)!=0u?end.y:middle.y);
        pixel=selectScreenPixel(begin,end,level+1u,parents,parentNormals,found);
    }
    if (found)
    {
        value.Metadata=uvec4(probe,InvalidProbe,pixel);
        if (screenSurface(pixel,point,normal))
        { value.PositionMinimum=vec4(point+normal*Screen.Parameters.x,Screen.Parameters.y); value.Metadata.y=0u; }
    }
    Placement.Values[probe]=value;
}
#elif defined(EVANS_LUMEN_PROBE_INTEGRATE_STAGE)
layout(std140,set=0,binding=0) uniform IntegrationFrame { uvec4 Counts; vec4 SkyMeasure,EnvironmentAxis,EnvironmentPeak; } Integrate;
struct ProbeHit { vec4 PositionDistance,NormalResidual,Uncertainty; uvec4 Metadata; vec4 Medium; };
struct ProbeIncoming { ProbeHit Hit; vec4 RadianceCoverage; uvec4 Cache; };
struct ProbeValue { vec4 Coefficients[9]; vec4 Coverage; uvec4 TraceCounts,CacheCounts,Metadata; };
layout(std430,set=0,binding=1) readonly buffer IncomingBuffer { ProbeIncoming Rays[]; } Incident;
layout(std430,set=0,binding=2) readonly buffer OriginBuffer { ProbeOrigin Probes[]; } Origins;
layout(std430,set=0,binding=3) writeonly buffer ProbeOutput { ProbeValue Probes[]; } Output;
const uint HarmonicCount=9u,Complete=1u,Partial=2u,Invalid=3u,SurfaceHit=1u,GeometryMiss=0u;
void integrateDirection(inout ProbeValue probe,vec3 flux,vec3 w)
{
    const float Pi=3.14159265358979323846;
    float constantBand=Pi*sqrt(1.0/(4.0*Pi)),linearBand=(2.0*Pi/3.0)*sqrt(3.0/(4.0*Pi));
    float crossBand=(Pi/4.0)*sqrt(15.0/(4.0*Pi)),polarBand=(Pi/4.0)*sqrt(5.0/(16.0*Pi)),differenceBand=(Pi/4.0)*sqrt(15.0/(16.0*Pi));
    probe.Coefficients[0].xyz+=flux*constantBand;
    probe.Coefficients[1].xyz+=flux*(linearBand*w.y); probe.Coefficients[2].xyz+=flux*(linearBand*w.z); probe.Coefficients[3].xyz+=flux*(linearBand*w.x);
    probe.Coefficients[4].xyz+=flux*(crossBand*w.x*w.y); probe.Coefficients[5].xyz+=flux*(crossBand*w.y*w.z);
    probe.Coefficients[6].xyz+=flux*(polarBand*(3.0*w.z*w.z-1.0)); probe.Coefficients[7].xyz+=flux*(crossBand*w.x*w.z);
    probe.Coefficients[8].xyz+=flux*(differenceBand*(w.x*w.x-w.y*w.y));
}
bool finiteProbeVector(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Integrate.Counts.w; if (index>=Integrate.Counts.x) return;
    ProbeValue result; for (uint coefficient=0u;coefficient<HarmonicCount;++coefficient) result.Coefficients[coefficient]=vec4(0.0);
    result.Coverage=vec4(0.0); result.TraceCounts=uvec4(0u); result.CacheCounts=uvec4(0u);
    uint identifier=Origins.Probes[index].Metadata.x; float covered=0.0,distanceSum=0.0; bool invalid=Origins.Probes[index].Metadata.y!=0u;
    for (uint sampleIndex=0u;sampleIndex<Integrate.Counts.y;++sampleIndex)
    {
        ProbeIncoming sampleValue=Incident.Rays[index*Integrate.Counts.y+sampleIndex]; vec3 radiance=vec3(0.0);
        vec3 direction=probeDirection(sampleIndex,Integrate.Counts.y,identifier,Integrate.Counts.z);
        if (sampleValue.Hit.Metadata.z==SurfaceHit || sampleValue.Hit.Metadata.z==LumenScreenSurface || sampleValue.Hit.Metadata.z==LumenWorldCache)
        {
            if (sampleValue.Hit.Metadata.z==LumenWorldCache) ++result.TraceCounts.w;
            else { ++result.TraceCounts.x; distanceSum+=sampleValue.Hit.PositionDistance.w/float(Integrate.Counts.y); }
            if (sampleValue.Cache.z==Complete) { ++result.CacheCounts.x; radiance=sampleValue.RadianceCoverage.xyz; covered+=sampleValue.RadianceCoverage.w; }
            else if (sampleValue.Cache.z==Partial) { ++result.CacheCounts.y; radiance=sampleValue.RadianceCoverage.xyz; covered+=sampleValue.RadianceCoverage.w; }
            else if (sampleValue.Cache.z==Invalid) { ++result.CacheCounts.w; invalid=true; }
            else ++result.CacheCounts.z;
        }
        else if (sampleValue.Hit.Metadata.z==GeometryMiss) { ++result.TraceCounts.y; radiance=lumenEnvironmentRadiance(Integrate.SkyMeasure.xyz,Integrate.EnvironmentAxis,Integrate.EnvironmentPeak.xyz,direction); }
        else ++result.TraceCounts.z;
        radiance=radiance*sampleValue.Hit.Medium.w+sampleValue.Hit.Medium.xyz;
        integrateDirection(result,radiance*Integrate.SkyMeasure.w,direction);
    }
    result.Coverage=vec4(float(result.TraceCounts.y+result.CacheCounts.x)/float(Integrate.Counts.y),
        (result.TraceCounts.x+result.TraceCounts.w)!=0u?covered/float(result.TraceCounts.x+result.TraceCounts.w):0.0,float(result.TraceCounts.y)/float(Integrate.Counts.y),
        result.TraceCounts.x!=0u?distanceSum*(float(Integrate.Counts.y)/float(result.TraceCounts.x)):0.0);
    for (uint coefficient=0u;coefficient<HarmonicCount;++coefficient) invalid=invalid || !finiteProbeVector(result.Coefficients[coefficient]);
    invalid=invalid || !finiteProbeVector(result.Coverage);
    // Reciprocal rounding must not publish confidence greater than one.
    result.Coverage.y=clamp(result.Coverage.y,0.0,1.0);
    uint status=(result.TraceCounts.z!=0u || result.CacheCounts.y!=0u || result.CacheCounts.z!=0u)?Partial:Complete;
    if (invalid) { status=Invalid; result.Coverage=vec4(0.0); for (uint coefficient=0u;coefficient<HarmonicCount;++coefficient) result.Coefficients[coefficient]=vec4(0.0); }
    result.Metadata=uvec4(identifier,status,Integrate.Counts.y,Integrate.Counts.z); Output.Probes[index]=result;
}
#elif defined(EVANS_LUMEN_RADIANCE_RESOLVE_STAGE)
layout(std140,binding=0) uniform RadianceFrame { uvec4 Counts,Atlas,Mapping; vec4 Parameters,Precision; } Resolve;
struct RadianceHit { vec4 PositionDistance,NormalResidual,Uncertainty; uvec4 Metadata; vec4 Medium; };
struct RadianceLight { vec4 RadianceCoverage; uvec4 Metadata; };
struct RadianceAnswer { RadianceHit Hit; vec4 RadianceCoverage; uvec4 Cache; };
layout(binding=1) uniform sampler2D RadianceNormal;
layout(std430,binding=2) readonly buffer RadianceHits { RadianceHit Values[]; } HitInput;
layout(std430,binding=3) readonly buffer RadianceLights { RadianceLight Values[]; } LightInput;
layout(std430,binding=4) readonly buffer RadianceMapping { uvec4 Vectors[]; } CardInput;
layout(std430,binding=5) writeonly buffer RadianceAnswers { RadianceAnswer Values[]; } Answer;
const uint RadianceDirections=6u,CardVectors=11u,InverseOffset=4u,OriginOffset=8u,BackwardOffset=9u,TileOffset=10u,Gutter=1u;
const uint MissingCard=0xffffffffu,SurfaceHit=1u,CompleteLight=1u,PartialLight=2u,InvalidLight=3u;
bool radianceFinite(float x) { return !(isnan(x) || isinf(x)); }
bool radianceFinite3(vec3 v) { return !(any(isnan(v)) || any(isinf(v))); }
vec4 radianceVector(uint at) { return uintBitsToFloat(CardInput.Vectors[at]); }
float coordinateMagnitude(vec3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
vec3 radianceProjection(uint base,vec4 value)
{ return vec3(dot(radianceVector(base),value),dot(radianceVector(base+1u),value),dot(radianceVector(base+2u),value)); }
// Depth acceptance chooses a geometric layer before lighting validity is used.
// An invalid nearest layer must not expose a bright layer behind it.
bool sampleRadianceCard(uint card,RadianceHit hit,float radius,out vec4 radiance,out float error,out float facing,out uint state)
{
    radiance=vec4(0.0); error=0.0; state=0u; uint base=Resolve.Mapping.x+card*CardVectors;
    uvec4 tile=CardInput.Vectors[base+TileOffset]; vec4 backward=radianceVector(base+BackwardOffset);
    facing=dot(hit.NormalResidual.xyz,backward.xyz); if (facing<Resolve.Parameters.x) return false;
    vec3 relative=hit.PositionDistance.xyz-radianceVector(base+OriginOffset).xyz;
    vec3 projected=radianceProjection(base,vec4(relative,1.0)); if (!radianceFinite3(projected) || any(greaterThan(abs(projected.xy),vec2(1.0)))) return false;
    vec2 uv=vec2(projected.x*0.5+0.5,0.5-projected.y*0.5),location=vec2(tile.xy)+float(Gutter)+uv*float(Resolve.Counts.y)-0.5;
    ivec2 lower=ivec2(floor(location)); vec2 fraction=location-vec2(lower); float coverage=0.0; state=CompleteLight;
    for (uint tap=0u;tap<4u;++tap)
    {
        uvec2 corner=uvec2(tap&1u,tap>>1u); ivec2 pixel=lower+ivec2(corner);
        float weight=(corner.x!=0u?fraction.x:1.0-fraction.x)*(corner.y!=0u?fraction.y:1.0-fraction.y);
        vec4 normal=texelFetch(RadianceNormal,pixel,0);
        if (weight<=0.0 || dot(normal.xyz,normal.xyz)==0.0 || dot(normal.xyz,hit.NormalResidual.xyz)<Resolve.Parameters.y) continue;
        vec2 texelUv=(vec2(pixel)-vec2(tile.xy)-float(Gutter)+0.5)/float(Resolve.Counts.y);
        vec3 captured=radianceProjection(base+InverseOffset,vec4(texelUv.x*2.0-1.0,1.0-texelUv.y*2.0,normal.w,1.0));
        float planeError=abs(dot(relative-captured,hit.NormalResidual.xyz));
        float allowance=Resolve.Precision.x*max(backward.w,max(coordinateMagnitude(relative),coordinateMagnitude(captured)))+Resolve.Precision.y*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!radianceFinite(planeError) || planeError>radius+allowance) continue;
        RadianceLight light=LightInput.Values[uint(pixel.y)*Resolve.Atlas.x+uint(pixel.x)];
        state=max(state,light.Metadata.x==0u?InvalidLight:light.Metadata.x); radiance.xyz+=light.RadianceCoverage.xyz*weight;
        error+=planeError*weight; coverage+=weight;
    }
    if (coverage<=0.0) { state=0u; return false; }
    radiance=vec4(radiance.xyz/coverage,clamp(coverage,0.0,1.0)); error/=coverage; return true;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Resolve.Atlas.w; if (index>=Resolve.Counts.x) return;
    RadianceAnswer result; result.Hit.Medium=vec4(0,0,0,1); result.Hit=HitInput.Values[index]; result.RadianceCoverage=vec4(0.0); result.Cache=uvec4(MissingCard,0u,0u,0u);
    if (result.Hit.Metadata.z==LumenScreenSurface || result.Hit.Metadata.z==LumenWorldCache) { Answer.Values[index].Hit.Medium=result.Hit.Medium; return; }
    float radius=Resolve.Parameters.z+result.Hit.Uncertainty.z*Resolve.Parameters.w;
    if (Resolve.Mapping.z==0u) result.Hit.Uncertainty.z=0.0;
    if (result.Hit.Metadata.z!=SurfaceHit) { Answer.Values[index]=result; return; }
    if (!radianceFinite(radius)) { result.Cache.z=InvalidLight; Answer.Values[index]=result; return; }
    uvec4 range=CardInput.Vectors[result.Hit.Metadata.x]; if (range.y==0u) { Answer.Values[index]=result; return; }
    vec4 groups[RadianceDirections]; float errors[RadianceDirections],facings[RadianceDirections]; uint states[RadianceDirections],cards[RadianceDirections];
    for (uint group=0u;group<RadianceDirections;++group) { groups[group]=vec4(0.0); errors[group]=0.0; facings[group]=0.0; states[group]=0u; cards[group]=MissingCard; }
    for (uint candidate=0u;candidate<range.y;++candidate)
    {
        uint word=range.x+candidate,card=CardInput.Vectors[word/4u][word%4u],group=CardInput.Vectors[Resolve.Mapping.x+card*CardVectors+TileOffset].z;
        vec4 value; float error,facing; uint state; if (!sampleRadianceCard(card,result.Hit,radius,value,error,facing,state)) continue;
        if (states[group]==0u || error<errors[group] || (error==errors[group] && value.w>groups[group].w))
        { groups[group]=value; errors[group]=error; facings[group]=facing; states[group]=state; cards[group]=card; }
    }
    float total=0.0,available=0.0,dominant=0.0; uint state=CompleteLight;
    for (uint group=0u;group<RadianceDirections;++group) if (states[group]!=0u)
    { float orientation=facings[group]*facings[group]; total+=orientation*groups[group].w; available+=orientation; }
    if (total>0.0)
    {
        for (uint group=0u;group<RadianceDirections;++group) if (states[group]!=0u)
        {
            float weight=facings[group]*facings[group]*groups[group].w/total; result.RadianceCoverage.xyz+=groups[group].xyz*weight;
            ++result.Cache.y; state=max(state,states[group]); if (states[group]!=CompleteLight) ++result.Cache.w;
            if (weight>dominant || (weight==dominant && cards[group]<result.Cache.x)) { dominant=weight; result.Cache.x=cards[group]; }
        }
        result.Cache.z=state; result.RadianceCoverage.w=clamp(total/available,0.0,1.0);
        if (state==InvalidLight || !radianceFinite3(result.RadianceCoverage.xyz)) { result.RadianceCoverage=vec4(0.0); result.Cache.z=InvalidLight; }
    }
    Answer.Values[index]=result;
}
#elif defined(EVANS_LUMEN_LOOKUP_STAGE)
// Explicit tap acceptance avoids interpolating a foreground depth with a
// background material. Projection groups retain the mesh-card layer relation.
layout(std140,set=0,binding=0) uniform LookupFrame { uvec4 Counts,Atlas; vec4 Parameters; } Lookup;
layout(set=0,binding=1) uniform sampler2D CacheAlbedo;
layout(set=0,binding=2) uniform sampler2D CacheNormal;
layout(set=0,binding=3) uniform sampler2D CacheEmission;
struct LookupCard { mat4 Projection,InverseProjection; vec4 Origin,BackwardScale; uvec4 Tile; };
struct LookupQuery { vec4 PositionRadius,NormalCosine; uvec4 Range; };
struct LookupSample { vec4 AlbedoMetallic,NormalCoverage,EmissionRoughness; uvec4 Metadata; };
layout(std430,row_major,set=0,binding=4) readonly buffer CardBuffer { LookupCard Values[]; } Cards;
layout(std430,set=0,binding=5) readonly buffer QueryBuffer { LookupQuery Values[]; } Queries;
layout(std430,set=0,binding=6) writeonly buffer SampleBuffer { LookupSample Values[]; } Samples;
const uint DirectionCount=6u,AtlasBorder=1u,MissingCard=0xffffffffu;
LookupSample missingSample()
{ return LookupSample(vec4(0.0),vec4(0.0),vec4(0.0),uvec4(MissingCard,0u,0u,0u)); }
float coordinateSize(vec3 value) { return max(abs(value.x),max(abs(value.y),abs(value.z))); }
bool lookupCard(LookupCard card,LookupQuery query,out LookupSample value,out float depthError,out float facing)
{
    value=missingSample(); depthError=0.0; facing=dot(query.NormalCosine.xyz,card.BackwardScale.xyz);
    if (facing<Lookup.Parameters.x) return false;
    vec3 relative=query.PositionRadius.xyz-card.Origin.xyz,clip=(card.Projection*vec4(relative,1.0)).xyz;
    if (any(isnan(clip)) || any(isinf(clip)) || any(greaterThan(abs(clip.xy),vec2(1.0)))) return false;
    vec2 location=vec2(card.Tile.xy)+float(AtlasBorder)+vec2(clip.x+1.0,1.0-clip.y)*0.5*float(Lookup.Counts.y)-0.5;
    ivec2 lower=ivec2(floor(location)); vec2 fraction=fract(location); float coverage=0.0; vec3 normal=vec3(0.0);
    for (uint corner=0u;corner<4u;++corner)
    {
        uvec2 delta=uvec2(corner&1u,(corner>>1u)&1u); ivec2 pixel=lower+ivec2(delta);
        float weight=(delta.x!=0u?fraction.x:1.0-fraction.x)*(delta.y!=0u?fraction.y:1.0-fraction.y);
        vec4 normalDepth=texelFetch(CacheNormal,pixel,0);
        if (weight<=0.0 || dot(normalDepth.xyz,normalDepth.xyz)==0.0 || dot(normalDepth.xyz,query.NormalCosine.xyz)<query.NormalCosine.w) continue;
        vec2 uv=(vec2(pixel)-vec2(card.Tile.xy)-float(AtlasBorder)+0.5)/float(Lookup.Counts.y);
        vec3 position=(card.InverseProjection*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,normalDepth.w,1.0)).xyz;
        float error=abs(dot(relative-position,query.NormalCosine.xyz));
        float roundoff=Lookup.Parameters.y*max(card.BackwardScale.w,max(coordinateSize(relative),coordinateSize(position)))+
            Lookup.Parameters.z*dot(abs(query.PositionRadius.xyz),abs(query.NormalCosine.xyz));
        if (isnan(error) || isinf(error) || error>query.PositionRadius.w+roundoff) continue;
        value.AlbedoMetallic+=texelFetch(CacheAlbedo,pixel,0)*weight; value.EmissionRoughness+=texelFetch(CacheEmission,pixel,0)*weight;
        normal+=normalDepth.xyz*weight; coverage+=weight; depthError+=error*weight;
    }
    if (coverage<=0.0) return false;
    value.AlbedoMetallic/=coverage; value.EmissionRoughness/=coverage; value.NormalCoverage=vec4(normal/coverage,clamp(coverage,0.0,1.0));
    value.Metadata=uvec4(card.Tile.w,1u,1u,0u); depthError/=coverage; return true;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Lookup.Atlas.z; if (index>=Lookup.Counts.x) return;
    LookupQuery query=Queries.Values[index]; LookupSample layers[DirectionCount]; float errors[DirectionCount],facing[DirectionCount];
    for (uint group=0u;group<DirectionCount;++group) { layers[group]=missingSample(); errors[group]=0.0; facing[group]=0.0; }
    for (uint candidate=0u;candidate<query.Range.y;++candidate)
    {
        LookupCard card=Cards.Values[query.Range.x+candidate]; LookupSample value; float error,cosine;
        if (!lookupCard(card,query,value,error,cosine)) continue;
        uint group=card.Tile.z;
        if (layers[group].Metadata.z==0u || error<errors[group] || (error==errors[group] && value.NormalCoverage.w>layers[group].NormalCoverage.w))
        { layers[group]=value; errors[group]=error; facing[group]=cosine; }
    }
    LookupSample result=missingSample(); float sum=0.0,orientationSum=0.0;
    for (uint group=0u;group<DirectionCount;++group) if (layers[group].Metadata.z!=0u)
    { float orientation=facing[group]*facing[group]; sum+=orientation*layers[group].NormalCoverage.w; orientationSum+=orientation; }
    if (sum>0.0)
    {
        vec3 normal=vec3(0.0); float dominant=0.0;
        for (uint group=0u;group<DirectionCount;++group) if (layers[group].Metadata.z!=0u)
        {
            float weight=facing[group]*facing[group]*layers[group].NormalCoverage.w/sum;
            result.AlbedoMetallic+=layers[group].AlbedoMetallic*weight; result.EmissionRoughness+=layers[group].EmissionRoughness*weight;
            normal+=layers[group].NormalCoverage.xyz*weight; ++result.Metadata.y;
            if (weight>dominant || (weight==dominant && layers[group].Metadata.x<result.Metadata.x)) { dominant=weight; result.Metadata.x=layers[group].Metadata.x; }
        }
        float squared=dot(normal,normal); result.NormalCoverage=vec4(squared>0.0?normal*inversesqrt(squared):query.NormalCosine.xyz,clamp(sum/orientationSum,0.0,1.0)); result.Metadata.z=1u;
    }
    result.AlbedoMetallic=clamp(result.AlbedoMetallic,0.0,1.0); result.EmissionRoughness.w=clamp(result.EmissionRoughness.w,0.0,1.0);
    Samples.Values[index]=result;
}
#elif defined(EVANS_LUMEN_DISTANCE_TRACE_STAGE) || defined(EVANS_LUMEN_LIGHTING_STAGE) || defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE) || defined(EVANS_LUMEN_PROBE_TRACE_STAGE) || defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE) || defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
// Native GLSL software tracing. Brick halos permit interpolation through one
// directory lookup; signed samples remain packed integers in the scene buffer.
layout(std140,row_major,set=0,binding=0) uniform TraceParameters
{
    uvec4 Counts,Offsets,Traversal; vec4 Parameters,HairSky;
#if defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
    mat4 Inverse; uvec4 Viewport,Grid; vec4 Origin,Adaptive,Appearance,GatherNumeric; uvec4 Model; vec4 Visibility;
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uvec4 ProbeSampling; vec4 ProbeRange;
#endif
#if defined(EVANS_LUMEN_LIGHTING_STAGE)
    uvec4 Atlas,LightingWork; vec4 Lighting,Numeric; uvec4 Tiles,Radiosity; vec4 SkyDistance,EnvironmentAxis,EnvironmentPeak;
#endif
} Trace;
#if defined(EVANS_LUMEN_LIGHTING_STAGE)
layout(std430,set=0,binding=4) readonly buffer MeshScene { uint Words[]; } Scene;
layout(set=0,binding=1) uniform sampler2D LightingAlbedo;
layout(set=0,binding=2) uniform sampler2D LightingNormal;
layout(set=0,binding=3) uniform sampler2D LightingEmission;
layout(std430,set=0,binding=5) readonly buffer LightingInput { uvec4 Vectors[]; } LightingData;
struct LightingTexel { vec4 RadianceCoverage; uvec4 Metadata; };
layout(std430,set=0,binding=6) writeonly buffer LightingOutput { LightingTexel Texels[]; } Lit;
#else
layout(std430,set=0,binding=1) readonly buffer MeshScene { uint Words[]; } Scene;
#endif
struct MeshRay { vec4 OriginMinimum,DirectionMaximum; };
struct MeshHit { vec4 PositionDistance,NormalResidual,Uncertainty; uvec4 Metadata; vec4 Medium; };
#if defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
layout(std430,set=0,binding=2) readonly buffer VisibilityOrigins { ProbeOrigin Values[]; } Candidates;
layout(std430,set=0,binding=3) writeonly buffer VisibilityOutput { uint Masks[]; } Visibility;
layout(set=0,binding=4) uniform sampler2D VisibilityDepth;
layout(set=0,binding=5) uniform sampler2D VisibilityNormal;
#elif defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
layout(std430,set=0,binding=2) readonly buffer VolumeProbeOrigins { ProbeOrigin Values[]; } VolumeOrigins;
layout(std430,set=0,binding=3) writeonly buffer VolumeProbeBounds { float Radii[]; } VolumeSupport;
#elif !defined(EVANS_LUMEN_LIGHTING_STAGE)
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
layout(std430,set=0,binding=2) readonly buffer ProbeQueue { ProbeOrigin Origins[]; } ProbeInput;
#else
layout(std430,set=0,binding=2) readonly buffer RayQueue { MeshRay Rays[]; } Input;
#endif
layout(std430,set=0,binding=3) buffer HitQueue { MeshHit Hits[]; } Output;
#endif
const uint FieldWords=20u,InstanceWords=20u,NodeWords=12u,RowWords=4u;
const uint FieldBricks=4u,FieldStorage=8u,FieldGrid=12u,FieldErrors=16u,InstanceRows=4u,InstanceData=16u,NodeMaximum=4u,NodeLinks=8u;
const uint BrickInterior=6u,BrickHalo=1u,BrickSide=8u,BrickSamples=512u,FirstBrick=2u;
const uint Invalid=0xffffffffu,Miss=0u,Surface=1u,Unresolved=2u;
const float SignedRange=32767.0;
uvec4 sceneWords(uint address) { return uvec4(Scene.Words[address],Scene.Words[address+1u],Scene.Words[address+2u],Scene.Words[address+3u]); }
vec4 sceneFloats(uint address) { return uintBitsToFloat(sceneWords(address)); }
bool finiteVector(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
bool finiteScalar(float value) { return !isnan(value) && !isinf(value); }
vec3 toField(uint instance,vec3 value)
{ return vec3(dot(sceneFloats(instance+InstanceRows).xyz,value),dot(sceneFloats(instance+InstanceRows+RowWords).xyz,value),dot(sceneFloats(instance+InstanceRows+RowWords*2u).xyz,value)); }
float scaledLength(vec3 value)
{ float scale=max(abs(value.x),max(abs(value.y),abs(value.z))); return scale>0.0?length(value/scale)*scale:0.0; }
vec3 safeUnit(vec3 value,vec3 fallback)
{
    if (!finiteVector(value)) return fallback;
    float scale=max(abs(value.x),max(abs(value.y),abs(value.z))); if (scale==0.0) return fallback;
    value/=scale; return value*inversesqrt(dot(value,value));
}
bool intersectBounds(vec3 origin,vec3 direction,vec3 minimum,vec3 maximum,inout float first,inout float last)
{
    for (uint axis=0u;axis<3u;++axis)
    {
        if (direction[axis]==0.0) { if (origin[axis]<minimum[axis] || origin[axis]>maximum[axis]) return false; }
        else
        {
            float a=(minimum[axis]-origin[axis])/direction[axis],b=(maximum[axis]-origin[axis])/direction[axis];
            first=max(first,min(a,b)); last=min(last,max(a,b)); if (first>last) return false;
        }
    }
    return true;
}
float sampleMesh(uint field,vec3 position)
{
    uvec3 dimensions=sceneWords(field).xyz,brickDimensions=sceneWords(field+FieldBricks).xyz;
    uvec4 storage=sceneWords(field+FieldStorage); vec4 grid=sceneFloats(field+FieldGrid);
    float band=sceneFloats(field+FieldErrors).x;
    vec3 coordinate=clamp((position-grid.xyz)/grid.w-0.5,vec3(0.0),vec3(dimensions)-1.0);
    uvec3 lower=uvec3(floor(coordinate)),brick=lower/BrickInterior,local=lower%BrickInterior+BrickHalo;
    vec3 fraction=coordinate-vec3(lower);
    uint page=Scene.Words[storage.x+(brick.z*brickDimensions.y+brick.y)*brickDimensions.x+brick.x];
    if (page<FirstBrick) return page==1u?-band:band;
    float sum=0.0;
    for (uint z=0u;z<2u;++z) for (uint y=0u;y<2u;++y) for (uint x=0u;x<2u;++x)
    {
        uint sampleIndex=(page-FirstBrick)*BrickSamples+((local.z+z)*BrickSide+local.y+y)*BrickSide+local.x+x;
        uint pair=Scene.Words[storage.y+sampleIndex/2u]; int encoded=(sampleIndex&1u)!=0u?int(pair)>>16:int(pair<<16)>>16;
        vec3 weight=mix(vec3(1.0)-fraction,fraction,bvec3(x!=0u,y!=0u,z!=0u));
        sum+=float(encoded)*weight.x*weight.y*weight.z;
    }
    return sum*band/SignedRange;
}
vec3 meshNormal(uint instance,uint field,vec3 p,float cell,vec3 fallback)
{
    vec3 gradient=vec3(sampleMesh(field,p+vec3(cell,0,0))-sampleMesh(field,p-vec3(cell,0,0)),
        sampleMesh(field,p+vec3(0,cell,0))-sampleMesh(field,p-vec3(0,cell,0)),sampleMesh(field,p+vec3(0,0,cell))-sampleMesh(field,p-vec3(0,0,cell)));
    gradient=safeUnit(gradient,vec3(0.0));
    vec3 normal=sceneFloats(instance+InstanceRows).xyz*gradient.x+sceneFloats(instance+InstanceRows+RowWords).xyz*gradient.y+sceneFloats(instance+InstanceRows+RowWords*2u).xyz*gradient.z;
    return safeUnit(normal,fallback);
}
uint marchInstance(MeshRay ray,uint index,float anchor,float limit,out float at,out vec4 normal,out vec2 uncertainty,out uint steps)
{
    at=anchor; normal=vec4(0.0); uncertainty=vec2(0.0); steps=0u;
    uint instance=Trace.Offsets.y+index*InstanceWords,field=Trace.Offsets.x+sceneWords(instance+InstanceData).x*FieldWords;
    vec3 origin=toField(instance,ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*anchor-sceneFloats(instance).xyz),direction=toField(instance,ray.DirectionMaximum.xyz);
    float speed=scaledLength(direction); if (!finiteVector(origin) || !finiteVector(direction) || !finiteScalar(speed) || speed<=0.0) return Unresolved;
    vec4 grid=sceneFloats(field+FieldGrid),errors=sceneFloats(field+FieldErrors); float first=0.0,last=max(0.0,limit-anchor);
    if (!intersectBounds(origin,direction,grid.xyz,grid.xyz+vec3(sceneWords(field).xyz)*grid.w,first,last)) return Miss;
    float travel=first,threshold=errors.y+Trace.Parameters.x*grid.w;
    for (uint iteration=0u;iteration<Trace.Traversal.x;++iteration)
    {
        at=min(limit,anchor+travel); vec3 position=origin+direction*travel; ++steps;
        if (!finiteVector(position)) return Unresolved;
        float sdf=sampleMesh(field,position);
        if (abs(sdf)<=threshold)
        {
            normal=vec4(meshNormal(instance,field,position,grid.w,-ray.DirectionMaximum.xyz),sdf/speed);
            uncertainty=vec2(errors.y,threshold)/speed; return Surface;
        }
        float next=travel+(abs(sdf)-errors.y)/speed*Trace.Parameters.y;
        if (!finiteScalar(next) || next<=travel) return Unresolved;
        if (next>last) return Miss; travel=next;
    }
    at=min(limit,anchor+travel); return Unresolved;
}
// A local tangent-plane distance changes by 1/|A^-T n| along the world
// normal. Cofactors evaluate this metric without assuming uniform scale.
float normalDistanceRate(uint instance,vec3 normal)
{
    vec3 a=sceneFloats(instance+InstanceRows).xyz,b=sceneFloats(instance+InstanceRows+RowWords).xyz,c=sceneFloats(instance+InstanceRows+RowWords*2u).xyz;
    vec3 extent=max(abs(a),max(abs(b),abs(c))); float scale=max(extent.x,max(extent.y,extent.z)); if (scale==0.0) return 0.0;
    a/=scale; b/=scale; c/=scale; vec3 ca=cross(b,c),cb=cross(c,a),cc=cross(a,b);
    float inverseNormal=scaledLength(vec3(dot(ca,normal),dot(cb,normal),dot(cc,normal)));
    float rate=inverseNormal>0.0?scale*(abs(dot(a,ca))/inverseNormal):0.0; return finiteScalar(rate)?rate:0.0;
}
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
// KHR inline traversal exposes every non-opaque candidate. Filtering uses the
// same immutable scalar images as software visibility, before hit confirmation.
layout(set=1,binding=0) uniform accelerationStructureEXT HardwareScene;
layout(std430,set=1,binding=1) readonly buffer HardwarePayload { uint Words[]; } Hardware;
const uint HardwareInstanceStride=16u,HardwareTriangleStride=24u,HardwareMaterialStride=12u;
// Integer word offsets follow the immutable host triangle/material/normal ABI.
const uint HardwareB=4u,HardwareC=8u,HardwareUVAB=12u,HardwareUVC=16u,HardwareTags=20u;
const uint HardwareOpacity=4u,HardwareCoverage=8u,HardwareNormal0=4u,HardwareNormal1=8u,HardwareNormal2=12u;
uvec4 hardwareWords(uint at) { return uvec4(Hardware.Words[at],Hardware.Words[at+1u],Hardware.Words[at+2u],Hardware.Words[at+3u]); }
vec4 hardwareFloats(uint at) { return uintBitsToFloat(hardwareWords(at)); }
float hardwareMap(uint start,uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0;
    vec2 repeated=fract(uv),p=vec2(repeated.x,1.0-repeated.y)*vec2(image.yz)-0.5;
    ivec2 first=ivec2(floor(p)),size=ivec2(image.yz); vec2 f=fract(p); float filtered=0.0;
    for (uint tap=0u;tap<4u;++tap)
    {
        ivec2 bit=ivec2(tap&1u,tap>>1u),cell=first+bit;
        // Repeated UVs put the four taps in [-1,size]. Avoid signed remainder
        // with negative operands, whose result GLSL leaves undefined.
        cell=ivec2(cell.x<0?cell.x+size.x:cell.x>=size.x?cell.x-size.x:cell.x,
            cell.y<0?cell.y+size.y:cell.y>=size.y?cell.y-size.y:cell.y);
        vec2 w=mix(vec2(1.0)-f,f,notEqual(bit,ivec2(0)));
        filtered+=uintBitsToFloat(Hardware.Words[start+image.x+uint(cell.y)*image.y+uint(cell.x)])*w.x*w.y;
    }
    return filtered;
}
bool hardwareCovered(uint instance,uint primitive,vec2 barycentric,uint flags)
{
    uvec4 range=hardwareWords(instance*HardwareInstanceStride); uint face=range.x+primitive*HardwareTriangleStride;
    uvec4 metadata=hardwareWords(face+HardwareTags); if ((metadata.w&flags)!=flags) return false;
    uint material=range.y+metadata.x*HardwareMaterialStride; vec4 rule=hardwareFloats(material+HardwareCoverage);
    if (rule.y<=0.0) return true;
    vec3 bary=vec3(1.0-barycentric.x-barycentric.y,barycentric);
    vec4 a=hardwareFloats(face),b=hardwareFloats(face+HardwareB),c=hardwareFloats(face+HardwareC),uvab=hardwareFloats(face+HardwareUVAB);
    vec2 uv=uvab.xy*bary.x+uvab.zw*bary.y+hardwareFloats(face+HardwareUVC).xy*bary.z;
    float alpha=clamp(dot(vec3(a.w,b.w,c.w),bary)*rule.x*hardwareMap(range.z,hardwareWords(material),uv)*hardwareMap(range.z,hardwareWords(material+HardwareOpacity),uv),0.0,1.0);
    return finiteScalar(alpha) && alpha>=rule.y;
}
MeshHit traceHardwareOpaque(MeshRay ray,uint flags)
{
    MeshHit hit; hit.Medium=vec4(0,0,0,1); hit.NormalResidual=vec4(0); hit.Uncertainty=vec4(0); hit.Metadata=uvec4(Invalid,Invalid,Miss,0);
    hit.PositionDistance=vec4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
    if (Trace.Counts.z==0u || ray.DirectionMaximum.w<=ray.OriginMinimum.w) return hit;
    const uint GeometryMask=1u,ShadowMask=2u;
    rayQueryEXT query;
    rayQueryInitializeEXT(query,HardwareScene,gl_RayFlagsNoOpaqueEXT,flags!=0u?ShadowMask:GeometryMask,
        ray.OriginMinimum.xyz,ray.OriginMinimum.w,ray.DirectionMaximum.xyz,ray.DirectionMaximum.w);
    uint selected=Invalid,primitive=Invalid; float distance=ray.DirectionMaximum.w;
    while (rayQueryProceedEXT(query))
    {
        if (rayQueryGetIntersectionTypeEXT(query,false)!=gl_RayQueryCandidateIntersectionTriangleEXT) continue;
        uint instance=rayQueryGetIntersectionInstanceCustomIndexEXT(query,false),face=rayQueryGetIntersectionPrimitiveIndexEXT(query,false);
        float at=rayQueryGetIntersectionTEXT(query,false);
        if (!hardwareCovered(instance,face,rayQueryGetIntersectionBarycentricsEXT(query,false),flags)) continue;
        if (selected==Invalid || at<distance || (at==distance && (instance<selected || (instance==selected && face<primitive))))
        { selected=instance; primitive=face; distance=at; rayQueryConfirmIntersectionEXT(query); }
    }
    if (selected==Invalid) return hit;
    uint object=selected*HardwareInstanceStride,face=hardwareWords(object).x+primitive*HardwareTriangleStride;
    vec3 a=hardwareFloats(face).xyz,b=hardwareFloats(face+HardwareB).xyz,c=hardwareFloats(face+HardwareC).xyz,e0=b-a,e1=c-a;
    if (!finiteVector(e0) || !finiteVector(e1))
    { float scale=max(max(abs(a.x),max(abs(a.y),abs(a.z))),max(max(abs(b.x),max(abs(b.y),abs(b.z))),max(abs(c.x),max(abs(c.y),abs(c.z)))));e0=b/scale-a/scale;e1=c/scale-a/scale; }
    vec3 local=cross(safeUnit(e0,vec3(0)),safeUnit(e1,vec3(0)));
    vec3 normal=safeUnit(vec3(dot(hardwareFloats(object+HardwareNormal0).xyz,local),dot(hardwareFloats(object+HardwareNormal1).xyz,local),dot(hardwareFloats(object+HardwareNormal2).xyz,local)),vec3(0));
    hit.PositionDistance=vec4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*distance,distance); hit.NormalResidual=vec4(normal,0);
    hit.Metadata=uvec4(selected,sceneWords(Trace.Offsets.y+selected*InstanceWords+InstanceData).y,dot(normal,normal)>0.0?Surface:Unresolved,0u); return hit;
}
#endif
MeshHit traceLocalOpaqueScene(MeshRay ray,uint requiredFlags)
{
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
    return traceHardwareOpaque(ray,requiredFlags);
#else
    float closest=ray.DirectionMaximum.w,unresolved=closest;
    uint selected=Invalid,unfinished=Invalid,node=0u,total=0u; vec4 selectedNormal=vec4(0.0); vec2 selectedError=vec2(0.0);
    while (node<Trace.Counts.y)
    {
        uint address=Trace.Offsets.z+node*NodeWords; uvec4 links=sceneWords(address+NodeLinks);
        float first=ray.OriginMinimum.w,last=min(closest,unresolved);
        if (!intersectBounds(ray.OriginMinimum.xyz,ray.DirectionMaximum.xyz,sceneFloats(address).xyz,sceneFloats(address+NodeMaximum).xyz,first,last)) { node=links.w; continue; }
        if (links.z==0u) { node=links.x; continue; }
        uint flags=sceneWords(Trace.Offsets.y+links.y*InstanceWords+InstanceData).z;
        if ((flags&requiredFlags)!=requiredFlags) { node=links.w; continue; }
        float at; vec4 normal; vec2 uncertainty; uint steps;
        uint status=marchInstance(ray,links.y,first,last,at,normal,uncertainty,steps); total+=steps;
        if (status==Surface && (at<closest || (at==closest && links.y<selected))) { closest=at; selected=links.y; selectedNormal=normal; selectedError=uncertainty; }
        if (status==Unresolved && (at<unresolved || (at==unresolved && links.y<unfinished))) { unresolved=at; unfinished=links.y; }
        node=links.w;
    }
    MeshHit hit; hit.Medium=vec4(0,0,0,1); hit.NormalResidual=vec4(0.0); hit.Uncertainty=vec4(0.0); hit.Metadata=uvec4(Invalid,Invalid,Miss,total); float at=ray.DirectionMaximum.w;
    if (unfinished!=Invalid && (selected==Invalid || unresolved<=closest))
    { at=unresolved; hit.Metadata.xyz=uvec3(unfinished,sceneWords(Trace.Offsets.y+unfinished*InstanceWords+InstanceData).y,Unresolved); }
    else if (selected!=Invalid)
    { at=closest; hit.NormalResidual=selectedNormal; hit.Uncertainty.xy=selectedError; hit.Metadata.xyz=uvec3(selected,sceneWords(Trace.Offsets.y+selected*InstanceWords+InstanceData).y,Surface); }
    hit.PositionDistance=vec4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*at,at); return hit;
#endif
}

MeshHit traceLocalOpaqueScene(MeshRay ray) { return traceLocalOpaqueScene(ray,Trace.Traversal.z); }

// A nonpositive radius means refinement, never a miss. The finest containing
// clipmap bounds trilinear error before the ray advances through empty space.
float globalClearance(vec3 position)
{
    const uint HeaderWords=8u,GridWords=12u,Metrics=4u,Storage=8u,SampleWords=4u;
    uvec4 shape=sceneWords(Trace.Offsets.w);
    for (uint level=0u;level<shape.x;++level)
    {
        uint header=Trace.Offsets.w+HeaderWords+level*GridWords; vec4 grid=sceneFloats(header);
        vec3 coordinate=(position-grid.xyz)/grid.w;
        if (!finiteVector(coordinate) || any(lessThan(coordinate,vec3(0))) || any(greaterThan(coordinate,vec3(shape.y-1u)))) continue;
        uvec3 lower=min(uvec3(floor(coordinate)),uvec3(shape.y-2u)); vec3 f=coordinate-vec3(lower); float radius=0.0;
        uint start=sceneWords(header+Storage).x;
        for (uint corner=0u;corner<8u;++corner)
        {
            uvec3 bit=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),at=lower+bit;
            vec4 sampleValue=sceneFloats(start+((at.z*shape.y+at.y)*shape.y+at.x)*SampleWords);
            vec3 weight=mix(vec3(1)-f,f,notEqual(bit,uvec3(0)));
            radius+=(Trace.Traversal.z!=0u?sampleValue.w:sampleValue.y)*weight.x*weight.y*weight.z;
        }
        return finiteScalar(radius)?max(0.0,radius-sceneFloats(header+Metrics).x):0.0;
    }
    return 0.0;
}
MeshHit traceOpaqueScene(MeshRay ray)
{
#if defined(EVANS_LUMEN_HARDWARE_RAYS)
    return traceHardwareOpaque(ray,Trace.Traversal.z);
#else
    MeshHit hit; hit.Medium=vec4(0,0,0,1); hit.PositionDistance=vec4(0); hit.NormalResidual=vec4(0); hit.Uncertainty=vec4(0); hit.Metadata=uvec4(Invalid,Invalid,Miss,0);
    bool global=Trace.Offsets.w!=0u; uint stages=global && Trace.Parameters.z>0.0?2u:1u;
    float at=ray.OriginMinimum.w; uint nearSteps=0u,globalSteps=0u;
    for (uint stage=0u;stage<stages;++stage)
    {
        bool near=stages==2u && stage==0u;
        if (global && !near) for (uint iteration=0u;iteration<uint(Trace.Parameters.w);++iteration)
        {
            ++globalSteps; float empty=globalClearance(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*at);
            float next=at+empty*Trace.Parameters.y; if (!finiteScalar(next) || next<=at) break;
            if (next>ray.DirectionMaximum.w)
            {
                hit.PositionDistance=vec4(ray.OriginMinimum.xyz+ray.DirectionMaximum.xyz*ray.DirectionMaximum.w,ray.DirectionMaximum.w);
                hit.NormalResidual=vec4(0); hit.Uncertainty=vec4(0,0,0,float(globalSteps)); hit.Metadata=uvec4(Invalid,Invalid,Miss,nearSteps); return hit;
            }
            at=next;
        }
        MeshRay detail=ray; detail.OriginMinimum.w=at;
        if (near) detail.DirectionMaximum.w=min(ray.DirectionMaximum.w,at+Trace.Parameters.z);
        hit=traceLocalOpaqueScene(detail); hit.Metadata.w+=nearSteps; hit.Uncertainty.w=float(globalSteps);
        if (!near || hit.Metadata.z!=Miss || detail.DirectionMaximum.w==ray.DirectionMaximum.w) return hit;
        at=detail.DirectionMaximum.w; nearSteps=hit.Metadata.w;
    }
    return hit;
#endif
}
// Fiber covariance supplies a cell-average projected width. Exact ray/cell
// intervals avoid a fixed raymarch step. The integrated source is low-frequency
// isotropic sky scattering plus emission; caster queries carry extinction only.
vec4 integrateHair(MeshRay ray,float limit,uint flags)
{
    const uint HairCellWords=24u,HairVectorWords=4u;
    vec4 result=vec4(0,0,0,1); uint header=Trace.Traversal.w>>LumenHairPointerShift;
    if (header==0u) return result;
    uvec4 grid=sceneWords(header); vec4 metric=sceneFloats(header+HairVectorWords);
    vec3 o=ray.OriginMinimum.xyz-metric.xyz,d=ray.DirectionMaximum.xyz;
    float first=ray.OriginMinimum.w,last=min(ray.DirectionMaximum.w,limit);
    if (last<=first || !intersectBounds(o,d,vec3(0),vec3(grid.xyz)*metric.w,first,last) || last<=first) return result;
    ivec3 cell=clamp(ivec3(floor((o+d*first)/metric.w)),ivec3(0),ivec3(grid.xyz)-1);
    ivec3 step=ivec3(sign(d)); bool shadow=(flags&1u)!=0u;
    for (uint visit=0u;visit<grid.x+grid.y+grid.z+1u && first<last;++visit)
    {
        vec3 crossing=vec3(last);
        for (uint axis=0u;axis<3u;++axis) if (step[axis]!=0)
            crossing[axis]=((float(cell[axis])+(step[axis]>0?1.0:0.0))*metric.w-o[axis])/d[axis];
        float end=max(first,min(last,min(crossing.x,min(crossing.y,crossing.z))));
        uint record=grid.w+(uint(cell.x)+grid.x*(uint(cell.y)+grid.y*uint(cell.z)))*HairCellWords;
        vec4 optical=sceneFloats(record),emission=sceneFloats(record+HairVectorWords);
        vec4 a=sceneFloats(record+(shadow?4u:2u)*HairVectorWords),b=sceneFloats(record+(shadow?5u:3u)*HairVectorWords);
        float alignment=dot(a.xyz,d*d)+2.0*(a.w*d.x*d.y+b.x*d.x*d.z+b.y*d.y*d.z);
        float sigma=(shadow?emission.w:optical.x)*sqrt(clamp(1.0-alignment,0.0,1.0));
        float transmission=exp(-sigma*(end-first));
        if (!shadow) result.xyz+=result.w*(1.0-transmission)*(optical.yzw*Trace.HairSky.xyz+emission.xyz);
        result.w*=transmission; first=end; if (first>=last) break;
        bool outside=false;
        for (uint axis=0u;axis<3u;++axis) if (step[axis]!=0 && crossing[axis]<=end)
        { cell[axis]+=step[axis]; outside=outside || cell[axis]<0 || cell[axis]>=int(grid[axis]); }
        if (outside) break;
    }
    return result;
}
MeshHit traceLocalMeshScene(MeshRay ray,uint flags)
{
    MeshHit hit=traceLocalOpaqueScene(ray,flags);
    hit.Medium=integrateHair(ray,hit.Metadata.z==Miss?ray.DirectionMaximum.w:hit.PositionDistance.w,flags); return hit;
}
MeshHit traceLocalMeshScene(MeshRay ray) { return traceLocalMeshScene(ray,Trace.Traversal.z); }
MeshHit traceMeshScene(MeshRay ray)
{
    MeshHit hit=traceOpaqueScene(ray);
    hit.Medium=integrateHair(ray,hit.Metadata.z==Miss?ray.DirectionMaximum.w:hit.PositionDistance.w,Trace.Traversal.z); return hit;
}
#if defined(EVANS_LUMEN_VOLUME_VISIBILITY_STAGE)
float volumeMagnitude(vec3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
float volumeInstanceSupport(uint index,vec3 world)
{
    uint instance=Trace.Offsets.y+index*InstanceWords,field=Trace.Offsets.x+sceneWords(instance+InstanceData).x*FieldWords;
    vec3 center=sceneFloats(instance).xyz,delta=world-center;
    vec3 a=sceneFloats(instance+InstanceRows).xyz,b=sceneFloats(instance+InstanceRows+RowWords).xyz,c=sceneFloats(instance+InstanceRows+RowWords*2u).xyz;
    vec3 p=vec3(dot(a,delta),dot(b,delta),dot(c,delta)); float scale=max(volumeMagnitude(a),max(volumeMagnitude(b),volumeMagnitude(c)));
    if (!finiteVector(p) || !finiteScalar(scale) || scale<=0.0) return 0.0;
    a=abs(a/scale); b=abs(b/scale); c=abs(c/scale);
    // A bound on the transform's spectral norm converts mesh-space clearance
    // into a conservative world sphere even for nonuniform instance scale.
    float rows=max(dot(a,vec3(1.0)),max(dot(b,vec3(1.0)),dot(c,vec3(1.0)))),norm=scale*sqrt(rows*volumeMagnitude(a+b+c));
    if (!finiteScalar(norm) || norm<=0.0) return 0.0;
    vec4 grid=sceneFloats(field+FieldGrid),errors=sceneFloats(field+FieldErrors);
    vec3 end=grid.xyz+vec3(sceneWords(field).xyz)*grid.w,sampled=clamp(p,grid.xyz+grid.w*0.5,end-grid.w*0.5);
    float outside=scaledLength(p-clamp(p,grid.xyz,end)),extension=scaledLength(p-sampled);
    // Budget the one point sample's interpolation error, not the extra hit
    // uncertainties needed when skipping a global ray-marching interval.
    float lower=max(outside,abs(sampleMesh(field,sampled))-errors.y-Trace.Parameters.x*grid.w-extension);
    lower-=Trace.Parameters.y*max(volumeMagnitude(p),max(volumeMagnitude(grid.xyz),volumeMagnitude(end)));
    float support=lower/norm-Trace.Parameters.y*max(volumeMagnitude(world),volumeMagnitude(center));
    return finiteScalar(support)?clamp(support,0.0,Trace.Parameters.z):0.0;
}
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Trace.Traversal.y; if (index>=Trace.Counts.x) return;
    ProbeOrigin probe=VolumeOrigins.Values[index]; float support=probe.Metadata.y==0u?Trace.Parameters.z:0.0;
    for (uint instance=0u;instance<Trace.Counts.z && support>0.0;++instance) support=min(support,volumeInstanceSupport(instance,probe.PositionMinimum.xyz));
    VolumeSupport.Radii[index]=support;
}
#elif defined(EVANS_LUMEN_GATHER_VISIBILITY_STAGE)
// Four visibility bits use the same clipped-edge bilinear lattice as gathering.
// Unknown mesh-SDF intervals reject a link; geometric weighting happens later.
float visibilityAxis(float center,uint extent,uint count)
{
    float side=float(Trace.Viewport.z),coordinate=center/side-0.5;
    if (count>1u && extent%Trace.Viewport.z!=0u)
    {
        float previous=(float(count)-1.5)*side,last=(float(count-1u)*side+float(extent))*0.5;
        if (center>previous) coordinate=float(count-2u)+(center-previous)/(last-previous);
    }
    return clamp(coordinate,0.0,float(count-1u));
}
float visibilityMagnitude(vec3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Trace.Traversal.y; if (index>=Trace.Counts.x) return;
    uint words=probeLinkWords(Trace.Grid.w+1u);
    for (uint word=0u;word<words;++word) Visibility.Masks[index*words+word]=0u; uvec2 pixel=uvec2(index%Trace.Viewport.x,index/Trace.Viewport.x);
    float depth=texelFetch(VisibilityDepth,ivec2(pixel),0).r; vec4 surface=texelFetch(VisibilityNormal,ivec2(pixel),0);
    if (!finiteScalar(depth) || depth<0.0 || depth>=1.0 || !finiteVector(surface.xyz) || !finiteScalar(surface.w) || surface.w<0.0 || dot(surface.xyz,surface.xyz)<=Trace.GatherNumeric.x) return;
    vec2 center=vec2(pixel)+0.5,uv=center/vec2(Trace.Viewport.xy);
    vec4 h=Trace.Inverse*vec4(uv*vec2(2,-2)+vec2(-1,1),depth,1.0);
    if (!finiteVector(h.xyz) || !finiteScalar(h.w) || abs(h.w)<=Trace.Origin.w) return;
    vec3 world=h.xyz/h.w,normal=normalize(surface.xyz);
    if (!finiteVector(world) || any(greaterThan(abs(world),vec3(Trace.Origin.z)))) return;
    vec2 grid=vec2(visibilityAxis(center.x,Trace.Viewport.x,Trace.Viewport.w),visibilityAxis(center.y,Trace.Viewport.y,Trace.Grid.x));
    uvec2 cell=uvec2(floor(grid)); vec2 fraction=grid-vec2(cell); uvec3 mask=uvec3(0u);
    uint slots=Trace.Grid.w+1u;
    for (uint tap=0u;tap<4u*slots;++tap)
    {
        uint cornerIndex=tap/slots;
        uvec2 corner=uvec2(cornerIndex&1u,cornerIndex>>1u),tile=min(cell+corner,uvec2(Trace.Viewport.w,Trace.Grid.x)-1u);
        if ((corner.x!=0u?fraction.x:1.0-fraction.x)*(corner.y!=0u?fraction.y:1.0-fraction.y)<=0.0) continue;
        uint probeIndex=(tile.y*Trace.Viewport.w+tile.x)*slots+tap%slots; ProbeOrigin origin=Candidates.Values[probeIndex];
        if (origin.Metadata.x!=probeIndex || origin.Metadata.y!=0u || any(greaterThanEqual(origin.Metadata.zw,Trace.Viewport.xy))) continue;
        vec4 surfaceAtProbe=texelFetch(VisibilityNormal,ivec2(origin.Metadata.zw),0);
        if (!finiteVector(surfaceAtProbe.xyz) || !finiteScalar(surfaceAtProbe.w) || surfaceAtProbe.w<0.0 || dot(surfaceAtProbe.xyz,surfaceAtProbe.xyz)<=Trace.GatherNumeric.x) continue;
        vec3 n=normalize(surfaceAtProbe.xyz),anchor=origin.PositionMinimum.xyz-n*Trace.Origin.x;
        if (dot(normal,n)<=Trace.Appearance.y || !finiteVector(anchor)) continue;
        float bias=Trace.Visibility.x+Trace.Visibility.y*max(visibilityMagnitude(world),visibilityMagnitude(anchor));
        vec3 start=world+normal*bias,end=anchor+n*bias,delta=end-start; float distance=scaledLength(delta);
        if (!finiteVector(start) || !finiteVector(end) || !finiteScalar(distance)) continue;
        if (distance==0.0) { mask[tap/LumenLinkWordBits]|=1u<<(tap%LumenLinkWordBits); continue; }
        MeshRay segment=MeshRay(vec4(start,0.0),vec4(safeUnit(delta,normal),distance));
        MeshHit hit=traceMeshScene(segment); if (hit.Metadata.z==Miss) mask[tap/LumenLinkWordBits]|=1u<<(tap%LumenLinkWordBits);
    }
    for (uint word=0u;word<words;++word) Visibility.Masks[index*words+word]=mask[word];
}
#elif !defined(EVANS_LUMEN_LIGHTING_STAGE)
layout(local_size_x=64) in;
void main()
{
    uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Trace.Traversal.y; if (index>=Trace.Counts.x) return;
    MeshRay ray;
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    ProbeOrigin probe=ProbeInput.Origins[index/Trace.ProbeSampling.x]; ray.OriginMinimum=probe.PositionMinimum;

    if (probe.Metadata.y!=0u)
    { MeshHit empty; empty.Medium=vec4(0,0,0,1); empty.PositionDistance=vec4(0.0); empty.NormalResidual=vec4(0.0); empty.Uncertainty=vec4(0.0); empty.Metadata=uvec4(Invalid,Invalid,Unresolved,0u); Output.Hits[index]=empty; return; }
    vec3 direction=Trace.ProbeRange.y>0.0?cacheTexelDirection(index%Trace.ProbeSampling.x,uint(Trace.ProbeRange.y)):
        probeDirection(index%Trace.ProbeSampling.x,Trace.ProbeSampling.x,probe.Metadata.x,Trace.ProbeSampling.z);
    bool cached=(Trace.ProbeSampling.w&2u)!=0u && Output.Hits[index].Metadata.z==LumenWorldCache;
    bool cachedSurface=cached && Output.Hits[index].Metadata.x!=Invalid;
    // Unlocated directional radiance has no segment endpoint for the medium.
    if ((Trace.Traversal.w>>LumenHairPointerShift)!=0u && !cachedSurface) cached=false;
    ray.DirectionMaximum=vec4(direction,cached && !cachedSurface?Output.Hits[index].PositionDistance.w:Trace.ProbeRange.x);
    if ((Trace.ProbeSampling.w&1u)!=0u && Output.Hits[index].Metadata.z==LumenScreenSurface)
    { Output.Hits[index].Medium=integrateHair(ray,Output.Hits[index].PositionDistance.w,Trace.Traversal.z); return; }
#else
    ray=Input.Rays[index];
#if defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE)
    const uint ReflectionGenerated=1u,ReflectionScreenFirst=2u;
    if ((Trace.Traversal.w&ReflectionScreenFirst)!=0u && Output.Hits[index].Metadata.z==LumenScreenSurface) { Output.Hits[index].Medium=integrateHair(ray,Output.Hits[index].PositionDistance.w,Trace.Traversal.z); return; }
    if ((Trace.Traversal.w&ReflectionGenerated)!=0u && ray.DirectionMaximum.w<=0.0)
    { Output.Hits[index]=MeshHit(vec4(0),vec4(0),vec4(0),uvec4(Invalid,Invalid,Unresolved,0u),vec4(0,0,0,1)); return; }
#endif
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    uint layers=uint(Trace.ProbeRange.z); bool terminal=false;
    MeshHit previous=MeshHit(vec4(0),vec4(0),vec4(0),uvec4(0),vec4(0,0,0,1));
    const float MinimumLayerCrossingCosine=0.05,LayerNormalAgreement=0.95;
    // Keep ordered SDF crossings per direction. Repeated surface shells and
    // grazing or sub-footprint intervals retain unknown provenance.
    for (uint layer=0u;layer<layers;++layer)
    {
    MeshHit hit=previous; if (!terminal) hit=traceMeshScene(ray);
#else
    MeshHit hit=traceMeshScene(ray);
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    if (cached && !cachedSurface && hit.Metadata.z==Miss) { Output.Hits[index].Medium=hit.Medium; return; }
#endif
#if defined(EVANS_LUMEN_RADIANCE_TRACE_STAGE) || defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    if (hit.Metadata.z==Surface)
    {
        uint instance=Trace.Offsets.y+hit.Metadata.x*InstanceWords,field=Trace.Offsets.x+sceneWords(instance+InstanceData).x*FieldWords;
        float threshold=sceneFloats(field+FieldErrors).y+Trace.Parameters.x*sceneFloats(field+FieldGrid).w;
#if !defined(EVANS_LUMEN_HARDWARE_RAYS)
        hit.Uncertainty.z=threshold/normalDistanceRate(instance,hit.NormalResidual.xyz);
#endif
    }
#endif
#if defined(EVANS_LUMEN_PROBE_TRACE_STAGE)
    float roundoff=Trace.ProbeRange.w*max(1.0,max(abs(ray.OriginMinimum.x),max(abs(ray.OriginMinimum.y),abs(ray.OriginMinimum.z)))+ray.DirectionMaximum.w);
    if (cachedSurface && hit.Metadata.z==Surface)
    {
        MeshHit expected=Output.Hits[index]; float tolerance=expected.Uncertainty.x+hit.Uncertainty.z+roundoff;
        if (all(equal(hit.Metadata.xy,expected.Metadata.xy)) && dot(hit.NormalResidual.xyz,expected.NormalResidual.xyz)>=LayerNormalAgreement &&
            finiteScalar(tolerance) && abs(dot(hit.PositionDistance.xyz-expected.PositionDistance.xyz,expected.NormalResidual.xyz))<=tolerance) { Output.Hits[index].Medium=hit.Medium; return; }
    }
    if (layer>0u && !terminal && hit.Metadata.z==Surface && hit.Metadata.x==previous.Metadata.x &&
        abs(dot(hit.NormalResidual.xyz,previous.NormalResidual.xyz))>=LayerNormalAgreement &&
        abs(dot(hit.PositionDistance.xyz-previous.PositionDistance.xyz,previous.NormalResidual.xyz))<=hit.Uncertainty.z+previous.Uncertainty.z+roundoff)
    { hit.Metadata.z=Unresolved; hit.NormalResidual=vec4(0); hit.Uncertainty.xyz=vec3(0); }
    uint destination=((index/Trace.ProbeSampling.x)*layers+layer)*Trace.ProbeSampling.x+index%Trace.ProbeSampling.x;
    Output.Hits[destination]=hit; if (layer+1u==layers) break;
    previous=hit;
    if (!terminal && hit.Metadata.z==Surface)
    {
        float cosine=abs(dot(hit.NormalResidual.xyz,ray.DirectionMaximum.xyz));
        float next=hit.PositionDistance.w+2.0*hit.Uncertainty.z/max(cosine,MinimumLayerCrossingCosine)+roundoff;
        if (cosine>=MinimumLayerCrossingCosine && finiteScalar(next) && next>hit.PositionDistance.w && next<ray.DirectionMaximum.w) ray.OriginMinimum.w=next;
        else { previous.Metadata.z=Unresolved; previous.NormalResidual=vec4(0); previous.Uncertainty.xyz=vec3(0); terminal=true; }
    }
    else terminal=true;
    }
#else
    Output.Hits[index]=hit;
#endif
}
#else
const uint LightingCardStride=7u,LightingEmitterStride=4u,CardOrigin=4u,CardBackward=5u,CardTile=6u,CardBorder=1u;
const uint DirectionalKind=0u,RectangleKind=2u,LightingComplete=1u,LightingUnresolved=2u,LightingInvalid=3u;
const uint EmitterNormal=1u,EmitterEnergy=2u,EmitterVisibility=3u;
vec4 lightingVector(uint at) { return uintBitsToFloat(LightingData.Vectors[at]); }
vec3 surfacePosition(uint card,uvec2 tile,ivec2 pixel,float depth)
{
    vec2 uv=(vec2(pixel)-vec2(tile)-float(CardBorder)+0.5)/float(Trace.Atlas.z);
    vec4 clip=vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    return vec3(dot(lightingVector(card),clip),dot(lightingVector(card+1u),clip),dot(lightingVector(card+2u),clip));
}
bool neighborDelta(uint card,uvec2 tile,ivec2 pixel,ivec2 step,vec3 center,out vec3 delta)
{
    ivec2 adjacent=pixel+step; delta=vec3(0.0);
    if (any(lessThan(adjacent,ivec2(tile))) || any(greaterThanEqual(adjacent,ivec2(tile+Trace.LightingWork.w)))) return false;
    vec4 depth=texelFetch(LightingNormal,adjacent,0); if (dot(depth.xyz,depth.xyz)==0.0) return false;
    delta=surfacePosition(card,tile,adjacent,depth.w)-center; return finiteVector(delta);
}
vec3 geometricNormal(uint card,uvec2 tile,ivec2 pixel,vec3 center,vec3 backward)
{
    vec3 right,left,down,up;
    bool r=neighborDelta(card,tile,pixel,ivec2(1,0),center,right),l=neighborDelta(card,tile,pixel,ivec2(-1,0),center,left);
    bool d=neighborDelta(card,tile,pixel,ivec2(0,1),center,down),u=neighborDelta(card,tile,pixel,ivec2(0,-1),center,up);
    if ((!r && !l) || (!d && !u)) return backward;
    vec3 dx=r && (!l || dot(right,right)<=dot(left,left))?right:-left,dy=d && (!u || dot(down,down)<=dot(up,up))?down:-up;
    vec3 normal=safeUnit(cross(dx,dy),backward); return dot(normal,backward)>=0.0?normal:-normal;
}
#if defined(EVANS_LUMEN_RADIOSITY)
layout(std430,set=0,binding=7) readonly buffer PreviousBounce { LightingTexel Texels[]; } Previous;
const uint BounceCardStride=11u,BounceInverse=4u,BounceOrigin=8u,BounceNormal=9u,BounceTile=10u,BounceDirections=6u;
const float BounceFacing=0.1,BounceNormalAgreement=0.25,BounceErrorRadius=2.0;
const float BounceLocalRoundoff=16.0*1.1920928955078125e-7,BounceWorldRoundoff=0.5*1.1920928955078125e-7;
const float BouncePi=3.14159265358979323846,BounceUIntFraction=1.0/4294967296.0,BounceBasisLimit=0.9;
float bounceMagnitude(vec3 p) { return max(abs(p.x),max(abs(p.y),abs(p.z))); }
vec3 bounceTransform(uint matrix,vec4 p)
{ return vec3(dot(lightingVector(matrix),p),dot(lightingVector(matrix+1u),p),dot(lightingVector(matrix+2u),p)); }
// Select depth-consistent material taps before reading the previous Jacobi
// iterate. An absent card remains unknown; it is never replaced by skylight.
bool bounceCard(uint card,MeshHit hit,float radius,out vec4 value,out float depthError,out float facing,out uint status)
{
    uint base=Trace.Radiosity.z+Trace.Radiosity.w+card*BounceCardStride; uvec4 tile=LightingData.Vectors[base+BounceTile];
    vec4 backward=lightingVector(base+BounceNormal); value=vec4(0); depthError=0.0; status=0u;
    facing=dot(backward.xyz,hit.NormalResidual.xyz); if (facing<BounceFacing) return false;
    vec3 relative=hit.PositionDistance.xyz-lightingVector(base+BounceOrigin).xyz,clip=bounceTransform(base,vec4(relative,1));
    if (!finiteVector(clip) || any(greaterThan(abs(clip.xy),vec2(1)))) return false;
    vec2 uv=vec2(clip.x*0.5+0.5,0.5-clip.y*0.5),pixel=vec2(tile.xy)+float(CardBorder)+uv*float(Trace.Atlas.z)-0.5;
    ivec2 lower=ivec2(floor(pixel)); vec2 fraction=fract(pixel); float coverage=0.0; status=LightingComplete;
    for (uint tap=0u;tap<4u;++tap)
    {
        uvec2 corner=uvec2(tap&1u,tap>>1u); ivec2 at=lower+ivec2(corner);
        vec2 weights=mix(vec2(1)-fraction,fraction,notEqual(corner,uvec2(0))); float weight=weights.x*weights.y;
        vec4 capturedNormal=texelFetch(LightingNormal,at,0);
        if (weight<=0.0 || dot(capturedNormal.xyz,capturedNormal.xyz)==0.0 || dot(capturedNormal.xyz,hit.NormalResidual.xyz)<BounceNormalAgreement) continue;
        vec2 sampleUv=(vec2(at)-vec2(tile.xy)-float(CardBorder)+0.5)/float(Trace.Atlas.z);
        vec3 captured=bounceTransform(base+BounceInverse,vec4(sampleUv.x*2.0-1.0,1.0-sampleUv.y*2.0,capturedNormal.w,1));
        float error=abs(dot(relative-captured,hit.NormalResidual.xyz));
        float roundoff=BounceLocalRoundoff*max(backward.w,max(bounceMagnitude(relative),bounceMagnitude(captured)))+
            BounceWorldRoundoff*dot(abs(hit.PositionDistance.xyz),abs(hit.NormalResidual.xyz));
        if (!finiteScalar(error) || error>radius+roundoff) continue;
        LightingTexel cached=Previous.Texels[uint(at.y)*Trace.Atlas.x+uint(at.x)];
        status=max(status,cached.Metadata.x==0u?LightingInvalid:cached.Metadata.x);
        value.xyz+=cached.RadianceCoverage.xyz*weight; depthError+=error*weight; coverage+=weight;
    }
    if (coverage<=0.0) { status=0u; return false; }
    value=vec4(value.xyz/coverage,clamp(coverage,0.0,1.0)); depthError/=coverage; return true;
}
uint bounceIncident(MeshHit hit,out vec3 incoming)
{
    incoming=vec3(0); uvec4 range=LightingData.Vectors[Trace.Radiosity.z+hit.Metadata.x]; if (range.y==0u) return 0u;
    uint object=Trace.Offsets.y+hit.Metadata.x*InstanceWords,field=Trace.Offsets.x+sceneWords(object+InstanceData).x*FieldWords;
    float tolerance=sceneFloats(field+FieldErrors).y+Trace.Parameters.x*sceneFloats(field+FieldGrid).w;
    float radius=BounceErrorRadius*tolerance/normalDistanceRate(object,hit.NormalResidual.xyz); if (!finiteScalar(radius)) return LightingInvalid;
    vec4 layers[BounceDirections]; vec2 quality[BounceDirections]; uint states[BounceDirections];
    for (uint direction=0u;direction<BounceDirections;++direction) { layers[direction]=vec4(0); quality[direction]=vec2(0); states[direction]=0u; }
    for (uint candidate=0u;candidate<range.y;++candidate)
    {
        uint address=Trace.Radiosity.z*4u+range.x+candidate,card=LightingData.Vectors[address/4u][address%4u];
        uint direction=LightingData.Vectors[Trace.Radiosity.z+Trace.Radiosity.w+card*BounceCardStride+BounceTile].z;
        vec4 value; float error,facing; uint state; if (!bounceCard(card,hit,radius,value,error,facing,state)) continue;
        if (states[direction]==0u || error<quality[direction].x || (error==quality[direction].x && value.w>layers[direction].w))
        { layers[direction]=value; quality[direction]=vec2(error,facing); states[direction]=state; }
    }
    float weightSum=0.0; uint status=0u;
    for (uint direction=0u;direction<BounceDirections;++direction) if (states[direction]!=0u)
    {
        float weight=quality[direction].y*quality[direction].y*layers[direction].w;
        incoming+=layers[direction].xyz*weight; weightSum+=weight; status=max(status,states[direction]);
    }
    if (weightSum>0.0) incoming/=weightSum;
    if (status==LightingInvalid || !finiteVector(incoming)) { incoming=vec3(0); return LightingInvalid; }
    return weightSum>0.0?status:0u;
}
vec3 bounceDirection(uint ray,uint pixel,vec3 normal)
{
    float radiusSquared=(float(ray)+0.5)/float(Trace.Radiosity.x);
    uint phase=bitfieldReverse(ray)+bitfieldReverse(pixel^Trace.Radiosity.y); float azimuth=2.0*BouncePi*float(phase)*BounceUIntFraction;
    vec3 axis=abs(normal.z)<BounceBasisLimit?vec3(0,0,1):vec3(0,1,0),right=normalize(cross(axis,normal));
    return sqrt(radiusSquared)*(cos(azimuth)*right+sin(azimuth)*cross(normal,right))+sqrt(1.0-radiusSquared)*normal;
}
#endif
layout(local_size_x=64) in;
void main()
{
    uint work=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Trace.Traversal.y,area=Trace.Tiles.w*Trace.Tiles.w;
    if (work>=Trace.Tiles.y*area) return;
    uint block=Trace.Tiles.x+work/area,within=work%area;
    uvec2 pixel=uvec2(block%Trace.Tiles.z,block/Trace.Tiles.z)*Trace.Tiles.w+uvec2(within%Trace.Tiles.w,within/Trace.Tiles.w);
    if (any(greaterThanEqual(pixel,Trace.Atlas.xy))) return;
    uint index=pixel.y*Trace.Atlas.x+pixel.x;
    LightingTexel result=LightingTexel(vec4(0.0),uvec4(0u));
    uint cardIndex=(pixel.y/Trace.LightingWork.w)*Trace.Atlas.w+pixel.x/Trace.LightingWork.w;
    if (cardIndex>=Trace.LightingWork.x) { Lit.Texels[index]=result; return; }
    vec4 normalDepth=texelFetch(LightingNormal,ivec2(pixel),0);
    if (dot(normalDepth.xyz,normalDepth.xyz)==0.0) { Lit.Texels[index]=result; return; }
    uint card=cardIndex*LightingCardStride; uvec4 tile=LightingData.Vectors[card+CardTile];
    vec3 relative=surfacePosition(card,tile.xy,ivec2(pixel),normalDepth.w),position=relative+lightingVector(card+CardOrigin).xyz;
    vec3 geometric=geometricNormal(card,tile.xy,ivec2(pixel),relative,lightingVector(card+CardBackward).xyz);
    float bias=max(Trace.Lighting.x,Trace.Numeric.x*max(abs(position.x),max(abs(position.y),abs(position.z))));
    if (tile.z!=Invalid)
    {
        uint instance=Trace.Offsets.y+tile.z*InstanceWords,field=Trace.Offsets.x+sceneWords(instance+InstanceData).x*FieldWords;
        float speed=normalDistanceRate(instance,geometric); float threshold=sceneFloats(field+FieldErrors).y+Trace.Parameters.x*sceneFloats(field+FieldGrid).w;
        bias=max(bias,threshold*Trace.Lighting.y/speed);
    }
    vec3 origin=position+geometric*bias,irradiance=vec3(0.0); bool invalid=!finiteVector(origin) || !finiteVector(normalDepth.xyz) || !finiteScalar(normalDepth.w);
    for (uint lightIndex=0u;lightIndex<Trace.LightingWork.y && !invalid;++lightIndex)
    {
        uint emitter=Trace.LightingWork.z+lightIndex*LightingEmitterStride;
        vec4 source=lightingVector(emitter),normalRange=lightingVector(emitter+EmitterNormal),energy=lightingVector(emitter+EmitterEnergy);
        if (!any(greaterThan(energy.xyz,vec3(0.0))) || energy.w==0.0) continue;
        vec3 direction=source.xyz; float attenuation=1.0;
        if (uint(source.w)!=DirectionalKind)
        {
            vec3 offset=source.xyz-position; float distance=scaledLength(offset); if (distance==0.0) continue;
            direction=offset/distance; float window=1.0;
            if (normalRange.w>0.0)
            {
                if (max(abs(offset.x),max(abs(offset.y),abs(offset.z)))>=normalRange.w) continue;
                vec3 scaled=offset/normalRange.w; float squared=dot(scaled,scaled); window=max(1.0-squared*squared,0.0); window*=window;
            }
            float cosine=uint(source.w)==RectangleKind?clamp(dot(normalRange.xyz,-direction),0.0,1.0):1.0;
            attenuation=energy.w*cosine*window/max(distance*distance,Trace.Lighting.w);
        }
        invalid=!finiteVector(direction) || !finiteScalar(attenuation); if (invalid) continue;
        float receiver=clamp(dot(normalDepth.xyz,direction),0.0,1.0);
        if (receiver==0.0 || attenuation==0.0 || dot(geometric,direction)<=0.0) continue;
        if (LightingData.Vectors[emitter+EmitterVisibility].x!=0u)
        {
            MeshRay ray=MeshRay(vec4(origin,0.0),vec4(direction,Trace.Lighting.z));
            if (uint(source.w)!=DirectionalKind)
            {
                vec3 target=source.xyz-origin; float distance=scaledLength(target);
                if (distance>0.0) ray.DirectionMaximum=vec4(target/distance,min(Trace.Lighting.z,max(0.0,distance-Trace.Lighting.x))); else ray.DirectionMaximum.w=0.0;
            }
            if (ray.DirectionMaximum.w>0.0)
            {
                ++result.Metadata.y; MeshHit visibility=traceMeshScene(ray);
                if (visibility.Metadata.z==Surface) { ++result.Metadata.z; continue; }
                if (visibility.Metadata.z==Unresolved) { ++result.Metadata.w; continue; }
                attenuation*=visibility.Medium.w;
            }
        }
        irradiance+=energy.xyz*(attenuation*receiver);
    }
#if defined(EVANS_LUMEN_RADIOSITY)
    vec3 bounce=vec3(0),normal=safeUnit(normalDepth.xyz,geometric);
    for (uint rayIndex=0u;rayIndex<Trace.Radiosity.x && !invalid;++rayIndex)
    {
        ++result.Metadata.y; vec3 direction=bounceDirection(rayIndex,index,normal); if (dot(direction,geometric)<=0.0) continue;
        MeshRay ray=MeshRay(vec4(origin,0),vec4(direction,Trace.SkyDistance.w)); MeshHit hit=traceLocalMeshScene(ray,0u);
        if (hit.Metadata.z==Miss) { bounce+=lumenEnvironmentRadiance(Trace.SkyDistance.xyz,Trace.EnvironmentAxis,Trace.EnvironmentPeak.xyz,direction)*hit.Medium.w+hit.Medium.xyz; continue; }
        if (hit.Metadata.z!=Surface) { ++result.Metadata.w; continue; }
        vec3 incident; uint state=bounceIncident(hit,incident);
        if (state==LightingInvalid) { invalid=true; continue; }
        if (state!=LightingComplete) ++result.Metadata.w;
        bounce+=incident*hit.Medium.w+hit.Medium.xyz;
    }
    irradiance+=bounce*(BouncePi/float(Trace.Radiosity.x));
#endif
    vec4 albedo=texelFetch(LightingAlbedo,ivec2(pixel),0);
    vec3 radiance=texelFetch(LightingEmission,ivec2(pixel),0).xyz+albedo.xyz*(1.0-albedo.w)*irradiance*Trace.Numeric.y;
    invalid=invalid || !finiteVector(radiance); result.RadianceCoverage=vec4(invalid?vec3(0.0):max(radiance,vec3(0.0)),1.0);
    result.Metadata.x=invalid?LightingInvalid:(result.Metadata.w!=0u?LightingUnresolved:LightingComplete); Lit.Texels[index]=result;
}
#endif
#elif defined(EVANS_LUMEN_READ_STAGE)
layout(std140,set=0,binding=0) uniform ExportParameters { uvec4 Size; } Frame;
layout(set=0,binding=1) uniform sampler2D ReflectanceImage;
layout(set=0,binding=2) uniform sampler2D OrientationImage;
layout(set=0,binding=3) uniform sampler2D EmissionImage;
layout(std430,set=0,binding=4) writeonly buffer ExportedMaterial { vec4 Elements[]; } Output;
const uint SurfaceVectors=3u;
layout(local_size_x=64) in;
void main()
{
    uint address=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*Frame.Size.z; if (address>=Frame.Size.x*Frame.Size.y) return;
    ivec2 texel=ivec2(address%Frame.Size.x,address/Frame.Size.x); uint base=address*SurfaceVectors;
    Output.Elements[base]=texelFetch(ReflectanceImage,texel,0); Output.Elements[base+1u]=texelFetch(OrientationImage,texel,0);
    Output.Elements[base+2u]=texelFetch(EmissionImage,texel,0);
}
#else
layout(std140,set=0,binding=0,row_major) uniform MaterialParameters
{
    mat4 ObjectToRelativeWorld,NormalToWorld,RelativeWorldToClip;
    vec4 ReflectanceCoverage,EmissionAndRoughness,MaterialValues;
    uvec4 TexturePresence;
} Material;
const float SafeNormalSquared=1.0e-10,SafeHomogeneous=1.0e-7,SafeUvDeterminant=1.0e-8;
vec3 normalizedMaterial(vec3 value,vec3 fallback)
{ float magnitude=dot(value,value); return magnitude>SafeNormalSquared?value*inversesqrt(magnitude):fallback; }
#if defined(EVANS_VERTEX_STAGE)
layout(location=0) in vec3 InPosition;
layout(location=1) in vec3 InNormal;
layout(location=2) in vec4 InColor;
layout(location=3) in vec2 InUV;
layout(location=0) out vec3 Position;
layout(location=1) out vec3 Normal;
layout(location=2) out vec4 Tint;
layout(location=3) out vec2 UV;
const float TransferBreakpoint=0.04045,TransferLinear=12.92,TransferOffset=0.055,TransferScale=1.055,TransferGamma=2.4;
vec3 decodeVertexColor(vec3 encoded)
{
    encoded=clamp(encoded,0.0,1.0); vec3 linear=encoded/TransferLinear,curved=pow((encoded+TransferOffset)/TransferScale,vec3(TransferGamma));
    return mix(curved,linear,lessThanEqual(encoded,vec3(TransferBreakpoint)));
}
void main()
{
    // Capture-relative coordinates avoid cancellation in thin card projections.
    vec4 point=Material.ObjectToRelativeWorld*vec4(InPosition,1);
    gl_Position=Material.RelativeWorldToClip*point; gl_Position.y=-gl_Position.y;
    Position=point.xyz/max(abs(point.w),SafeHomogeneous); Normal=mat3(Material.NormalToWorld)*InNormal;
    Tint=vec4(decodeVertexColor(InColor.rgb),InColor.a); UV=vec2(InUV.x,1.0-InUV.y);
}
#else
layout(location=0) in vec3 Position;
layout(location=1) in vec3 Normal;
layout(location=2) in vec4 Tint;
layout(location=3) in vec2 UV;
layout(set=0,binding=1) uniform sampler2D DiffuseTexture;
layout(set=0,binding=2) uniform sampler2D NormalTexture;
layout(set=0,binding=3) uniform sampler2D OpacityTexture;
layout(location=0) out vec4 Reflectance;
layout(location=1) out vec4 OrientationDepth;
layout(location=2) out vec4 Emission;
vec3 materialNormal()
{
    vec3 n=normalizedMaterial(Normal,vec3(0,0,1)); n=gl_FrontFacing?n:-n;
    vec3 horizontal=dFdx(Position),vertical=dFdy(Position); vec2 uvX=dFdx(UV),uvY=dFdy(UV);
    float determinant=uvX.x*uvY.y-uvX.y*uvY.x;
    if (Material.TexturePresence.y==0u || abs(determinant)<SafeUvDeterminant) return n;
    vec3 t=(horizontal*uvY.y-vertical*uvX.y)/determinant,b=(vertical*uvX.x-horizontal*uvY.x)/determinant;
    t-=n*dot(n,t); if (dot(t,t)<=SafeNormalSquared) return n; t=normalize(t);
    vec3 bitangent=cross(n,t); bitangent*=dot(bitangent,b)<0.0?-1.0:1.0;
    vec3 map=texture(NormalTexture,UV).xyz*2.0-1.0;
    return normalizedMaterial(t*map.x+bitangent*map.y+n*map.z,n);
}
void main()
{
    vec3 normal=materialNormal();
    vec4 sampled=Material.TexturePresence.x!=0u?texture(DiffuseTexture,UV):vec4(1);
    float opacity=Material.TexturePresence.z!=0u?texture(OpacityTexture,UV).r:1.0;
    float coverage=clamp(Tint.a*Material.ReflectanceCoverage.a*sampled.a*opacity,0.0,1.0); if (coverage<Material.MaterialValues.y) discard;
    Reflectance=vec4(clamp(Tint.rgb*Material.ReflectanceCoverage.rgb*sampled.rgb,0.0,1.0),Material.MaterialValues.x);
    OrientationDepth=vec4(normal,clamp(gl_FragCoord.z,0.0,1.0)); Emission=Material.EmissionAndRoughness;
}
#endif
#endif
