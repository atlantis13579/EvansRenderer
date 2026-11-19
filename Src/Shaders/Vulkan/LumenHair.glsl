#version 450
// Vulkan fragment receiver for the existing invariant fiber vertex module.
// Projected-cylinder SH multipliers: {1,0,-1/8}. Deconvolve the atlas's Lambert
// factors, then apply the even fiber filter; direct scattering stays separate.
#if defined(EVANS_LUMEN_HAIR_COMPOSITE) && defined(EVANS_VERTEX_STAGE)
void main()
{ vec2 p=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);gl_Position=vec4(p*2.0-1.0,0,1); }
#else
layout(set=0,binding=2) uniform sampler2D FiberVolume;
layout(location=0) out vec4 OutputColor;
const float Pi=3.14159265358979323846,MaximumHalf=65504.0;
#ifdef EVANS_LUMEN_HAIR_COMPOSITE
layout(set=0,binding=3) uniform sampler2D Scene;
void main()
{ ivec2 p=ivec2(gl_FragCoord.xy);vec4 c=texelFetch(Scene,p,0);OutputColor=vec4(clamp(c.rgb+texelFetch(FiberVolume,p,0).rgb,0.0,MaximumHalf),c.a); }
#else
layout(std140,row_major,set=0,binding=1) uniform FiberIrradianceFrame
{
    mat4 Camera,Inverse;
    uvec4 Grid;
    vec4 Distribution,Atlas,RootTint,TipTint,Ambient,ClipPlane;
} Frame;
layout(location=0) in vec3 WorldPosition;
layout(location=1) in vec3 FiberAxis;
layout(location=2) in vec3 Reflectance;
layout(location=4) in vec3 CrossHeight;
const uint SupportBit=2u,CoefficientCount=9u;
bool finite3(vec3 v) { return !any(isnan(v))&&!any(isinf(v)); }
float cylinderBasis(uint term,vec3 t)
{
    if (term==0u) return sqrt(1.0/(4.0*Pi))/Pi;
    const float evenScale=-1.0/(2.0*Pi);
    if (term==4u) return evenScale*sqrt(15.0/(4.0*Pi))*t.x*t.y;
    if (term==5u) return evenScale*sqrt(15.0/(4.0*Pi))*t.y*t.z;
    if (term==6u) return evenScale*sqrt(5.0/(16.0*Pi))*(3.0*t.z*t.z-1.0);
    if (term==7u) return evenScale*sqrt(15.0/(4.0*Pi))*t.x*t.z;
    if (term==8u) return evenScale*sqrt(15.0/(16.0*Pi))*(t.x*t.x-t.y*t.y);
    return 0.0;
}
vec3 incidentFiber(vec3 world,vec3 tangent)
{
    vec4 clip=Frame.Camera*vec4(world,1);if (!finite3(clip.xyz) || isnan(clip.w) || isinf(clip.w) || clip.w<=Frame.Atlas.z) return Frame.Ambient.rgb;
    vec3 ndc=clip.xyz/clip.w;if (any(greaterThan(abs(ndc.xy),vec2(1))) || ndc.z<0.0 || ndc.z>1.0) return Frame.Ambient.rgb;
    vec4 a=Frame.Inverse*vec4(ndc.xy,0,1),b=Frame.Inverse*vec4(ndc.xy,1,1);
    if (min(abs(a.w),abs(b.w))<=Frame.Atlas.z) return Frame.Ambient.rgb;
    vec3 start=a.xyz/a.w;float span=min(length(b.xyz/b.w-start),Frame.Distribution.x),distance=length(world-start);
    if (isnan(span) || isinf(span) || span<=Frame.Atlas.z || isnan(distance) || isinf(distance) || distance>span) return Frame.Ambient.rgb;
    vec2 uv=ndc.xy*vec2(0.5,-0.5)+0.5;
    vec3 cell=clamp(vec3(uv*vec2(Frame.Grid.xy),pow(clamp(distance/span,0.0,1.0),1.0/Frame.Distribution.y)*float(Frame.Grid.z))-0.5,vec3(0),vec3(Frame.Grid.xyz)-1.0);
    uvec3 lower=uvec3(floor(cell));vec3 f=fract(cell),light=vec3(0);float accepted=0.0;
    for (uint tap=0u;tap<8u;++tap)
    {
        uvec3 bit=uvec3(tap&1u,(tap>>1u)&1u,tap>>2u),p=min(lower+bit,Frame.Grid.xyz-1u);
        vec3 weights=mix(1.0-f,f,notEqual(bit,uvec3(0)));float weight=weights.x*weights.y*weights.z;if (weight<=0.0) continue;
        uint row=p.z*Frame.Grid.y+p.y;
        if ((Frame.Grid.w&SupportBit)!=0u)
        {
            float radius=texelFetch(FiberVolume,ivec2(p.x,row),0).w;if (radius<=0.0) continue;
            vec2 xy=(vec2(p.xy)+0.5)/vec2(Frame.Grid.xy)*vec2(2,-2)+vec2(-1,1);
            vec4 nearH=Frame.Inverse*vec4(xy,0,1),farH=Frame.Inverse*vec4(xy,1,1);
            if (min(abs(nearH.w),abs(farH.w))<=Frame.Atlas.z) continue;
            vec3 nearPoint=nearH.xyz/nearH.w,axis=farH.xyz/farH.w-nearPoint;float extent=length(axis);
            if (isnan(extent) || isinf(extent) || extent<=Frame.Atlas.z) continue;
            vec3 probe=nearPoint+axis*(pow((float(p.z)+0.5)/float(Frame.Grid.z),Frame.Distribution.y)*min(extent,Frame.Distribution.x)/extent);
            float separation=length(world-probe);if (isnan(separation) || isinf(separation) || separation>radius) continue;
        }
        accepted+=weight;
        for (uint harmonic=0u;harmonic<CoefficientCount;++harmonic)
            light+=weight*cylinderBasis(harmonic,tangent)*texelFetch(FiberVolume,ivec2(p.x+harmonic*Frame.Grid.x,row),0).rgb;
    }
    return max(light/Frame.Distribution.w,vec3(0))*Frame.Distribution.z+Frame.Ambient.rgb*clamp(1.0-accepted,0.0,1.0);
}
void main()
{
    if (dot(vec4(WorldPosition,1),Frame.ClipPlane)<0.0) discard;
    float squared=dot(FiberAxis,FiberAxis);vec3 tangent=squared>Frame.Atlas.w?FiberAxis*inversesqrt(squared):vec3(0,1,0);
    vec3 color=clamp(Reflectance*mix(Frame.RootTint.rgb,Frame.TipTint.rgb,clamp(CrossHeight.y,0.0,1.0)),0.0,1.0);
    OutputColor=vec4(clamp(color*incidentFiber(WorldPosition,tangent)*Frame.RootTint.w,0.0,MaximumHalf),0);
}
#endif
#endif
