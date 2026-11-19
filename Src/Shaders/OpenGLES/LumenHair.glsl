#version 310 es
precision highp float;
precision highp int;
// ES owns its low-frequency fiber filter. The GI draw links the same invariant
// HairFur vertex source as the primary surface, retaining multisample coverage.
// Cylinder convolution after Lambert SH storage: band0/pi, band2/(-2*pi).
#if defined(EVANS_LUMEN_HAIR_COMPOSITE) && defined(EVANS_VERTEX_STAGE)
void main()
{ vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0,1); }
#else
uniform highp sampler2D FiberVolume;
layout(location=0) out vec4 OutColor;
const float SpherePi=3.14159265358979323846,HalfMaximum=65504.0;
#ifdef EVANS_LUMEN_HAIR_COMPOSITE
uniform highp sampler2D Scene;
void main()
{ ivec2 p=ivec2(gl_FragCoord.xy);vec4 c=texelFetch(Scene,p,0);OutColor=vec4(clamp(c.rgb+texelFetch(FiberVolume,p,0).rgb,0.0,HalfMaximum),c.a); }
#else
layout(std140,binding=1) uniform FiberVolumeFrame
{
    vec4 CameraRows[4],InverseRows[4];
    uvec4 Shape;
    vec4 Range,Atlas,Root,Tip,Fallback,ReceiverClipPlane;
};
in vec3 WorldPosition,FiberTangent,SourceReflectance;
in vec3 CrossHeight;
const uint SupportFlag=2u,Harmonics=9u;
vec4 projectFiber(vec4 v)
{ return vec4(dot(CameraRows[0],v),dot(CameraRows[1],v),dot(CameraRows[2],v),dot(CameraRows[3],v)); }
vec4 unprojectFiber(vec4 v)
{ return vec4(dot(InverseRows[0],v),dot(InverseRows[1],v),dot(InverseRows[2],v),dot(InverseRows[3],v)); }
bool fiberFinite(float v) { return !isnan(v)&&!isinf(v); }
ivec2 volumeTexel(uint x,uint y) { return ivec2(int(x),int(Atlas.y)-1-int(y)); }
float fiberHarmonic(uint term,vec3 t)
{
    if (term==0u) return sqrt(1.0/(4.0*SpherePi))/SpherePi;
    const float secondBand=-1.0/(2.0*SpherePi);
    if (term==4u) return secondBand*sqrt(15.0/(4.0*SpherePi))*t.x*t.y;
    if (term==5u) return secondBand*sqrt(15.0/(4.0*SpherePi))*t.y*t.z;
    if (term==6u) return secondBand*sqrt(5.0/(16.0*SpherePi))*(3.0*t.z*t.z-1.0);
    if (term==7u) return secondBand*sqrt(15.0/(4.0*SpherePi))*t.x*t.z;
    if (term==8u) return secondBand*sqrt(15.0/(16.0*SpherePi))*(t.x*t.x-t.y*t.y);
    return 0.0;
}
vec3 fiberIncoming(vec3 point,vec3 tangent)
{
    vec4 clip=projectFiber(vec4(point,1));
    if (any(isnan(clip)) || any(isinf(clip)) || clip.w<=Atlas.z) return Fallback.rgb;
    vec3 ndc=clip.xyz/clip.w;
    if (any(greaterThan(abs(ndc.xy),vec2(1))) || ndc.z<0.0 || ndc.z>1.0) return Fallback.rgb;
    vec4 a=unprojectFiber(vec4(ndc.xy,0,1)),b=unprojectFiber(vec4(ndc.xy,1,1));
    if (min(abs(a.w),abs(b.w))<=Atlas.z) return Fallback.rgb;
    vec3 start=a.xyz/a.w;float extent=min(length(b.xyz/b.w-start),Range.x),distance=length(point-start);
    if (!fiberFinite(extent) || extent<=Atlas.z || !fiberFinite(distance) || distance>extent) return Fallback.rgb;
    vec2 uv=ndc.xy*vec2(0.5,-0.5)+0.5;
    vec3 cell=clamp(vec3(uv*vec2(Shape.xy),pow(clamp(distance/extent,0.0,1.0),1.0/Range.y)*float(Shape.z))-0.5,vec3(0),vec3(Shape.xyz)-1.0);
    uvec3 base=uvec3(floor(cell));vec3 f=fract(cell),light=vec3(0);float mass=0.0;
    for (uint corner=0u;corner<8u;++corner)
    {
        uvec3 bit=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),p=min(base+bit,Shape.xyz-1u);
        vec3 w=mix(1.0-f,f,notEqual(bit,uvec3(0)));float weight=w.x*w.y*w.z;if (weight<=0.0) continue;
        uint row=p.z*Shape.y+p.y;
        if ((Shape.w&SupportFlag)!=0u)
        {
            float support=texelFetch(FiberVolume,volumeTexel(p.x,row),0).a;if (support<=0.0) continue;
            vec2 xy=(vec2(p.xy)+0.5)/vec2(Shape.xy)*vec2(2,-2)+vec2(-1,1);
            vec4 nearH=unprojectFiber(vec4(xy,0,1)),farH=unprojectFiber(vec4(xy,1,1));
            if (min(abs(nearH.w),abs(farH.w))<=Atlas.z) continue;
            vec3 origin=nearH.xyz/nearH.w,axis=farH.xyz/farH.w-origin;float span=length(axis);
            if (!fiberFinite(span) || span<=Atlas.z) continue;
            vec3 probe=origin+axis*(pow((float(p.z)+0.5)/float(Shape.z),Range.y)*min(span,Range.x)/span);
            float separation=length(point-probe);if (!fiberFinite(separation) || separation>support) continue;
        }
        mass+=weight;
        for (uint harmonic=0u;harmonic<Harmonics;++harmonic)
            light+=weight*fiberHarmonic(harmonic,tangent)*texelFetch(FiberVolume,volumeTexel(p.x+harmonic*Shape.x,row),0).rgb;
    }
    return max(light/Range.w,vec3(0))*Range.z+Fallback.rgb*clamp(1.0-mass,0.0,1.0);
}
void main()
{
    if (dot(vec4(WorldPosition,1),ReceiverClipPlane)<0.0) discard;
    float squared=dot(FiberTangent,FiberTangent);vec3 tangent=squared>Atlas.w?FiberTangent*inversesqrt(squared):vec3(0,1,0);
    vec3 color=clamp(SourceReflectance*mix(Root.rgb,Tip.rgb,clamp(CrossHeight.y,0.0,1.0)),0.0,1.0);
    OutColor=vec4(clamp(color*fiberIncoming(WorldPosition,tangent)*Root.w,0.0,HalfMaximum),0);
}
#endif
#endif
