#include <metal_stdlib>
using namespace metal;
// Independent MSL cylinder receiver. Normalized projected area has even SH
// multipliers {1,0,-1/8}; remove the atlas's cosine convolution first. This is
// a low-frequency indirect approximation, separate from direct R/TT/TRT paths.
struct LumenFiberFrame
{
    float4 Camera[4],Inverse[4];
    uint4 Grid;
    float4 Distribution,Atlas,RootTint,TipTint,Ambient,ClipPlane;
};
struct FiberFragment
{
    float4 Position [[position,invariant]];
    float3 World,Tangent,Reflectance,Emission;
    float3 CrossHeight;
    float LinearShadowDepth;
    float4 PreviousClip;
};
constant float Pi=3.14159265358979323846,MaximumHalf=65504;
constant uint VisibilityBit=2,CoefficientCount=9;
float4 fiberTransform(constant float4* rows,float4 v)
{ return float4(dot(rows[0],v),dot(rows[1],v),dot(rows[2],v),dot(rows[3],v)); }
float projectedCylinder(uint term,float3 t)
{
    if (term==0) return sqrt(1/(4*Pi))/Pi;
    const float evenScale=-1/(2*Pi);
    switch (term)
    {
        case 4:return evenScale*sqrt(15/(4*Pi))*t.x*t.y;
        case 5:return evenScale*sqrt(15/(4*Pi))*t.y*t.z;
        case 6:return evenScale*sqrt(5/(16*Pi))*(3*t.z*t.z-1);
        case 7:return evenScale*sqrt(15/(4*Pi))*t.x*t.z;
        case 8:return evenScale*sqrt(15/(16*Pi))*(t.x*t.x-t.y*t.y);
    }
    return 0;
}
float3 fiberIllumination(float3 world,float3 tangent,constant LumenFiberFrame& frame,texture2d<float,access::read> atlas)
{
    float4 clip=fiberTransform(frame.Camera,float4(world,1));
    if (!all(isfinite(clip)) || clip.w<=frame.Atlas.z) return frame.Ambient.xyz;
    float3 ndc=clip.xyz/clip.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return frame.Ambient.xyz;
    float4 a=fiberTransform(frame.Inverse,float4(ndc.xy,0,1)),b=fiberTransform(frame.Inverse,float4(ndc.xy,1,1));
    if (min(abs(a.w),abs(b.w))<=frame.Atlas.z) return frame.Ambient.xyz;
    float3 start=a.xyz/a.w;float extent=min(length(b.xyz/b.w-start),frame.Distribution.x),distance=length(world-start);
    if (!isfinite(extent) || extent<=frame.Atlas.z || !isfinite(distance) || distance>extent) return frame.Ambient.xyz;
    float2 uv=ndc.xy*float2(0.5f,-0.5f)+0.5f;
    float3 cell=clamp(float3(uv*float2(frame.Grid.xy),pow(saturate(distance/extent),1/frame.Distribution.y)*frame.Grid.z)-0.5f,float3(0),float3(frame.Grid.xyz)-1);
    uint3 base=uint3(floor(cell));float3 f=fract(cell),incoming=0;float accepted=0;
    for (uint corner=0;corner<8;++corner)
    {
        uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,frame.Grid.xyz-1);
        float3 w=select(1-f,f,bit!=uint3(0));float weight=w.x*w.y*w.z;if (weight<=0) continue;
        uint row=p.z*frame.Grid.y+p.y;
        if ((frame.Grid.w&VisibilityBit)!=0)
        {
            float radius=atlas.read(uint2(p.x,row)).w;if (radius<=0) continue;
            float2 xy=(float2(p.xy)+0.5f)/float2(frame.Grid.xy)*float2(2,-2)+float2(-1,1);
            float4 nearH=fiberTransform(frame.Inverse,float4(xy,0,1)),farH=fiberTransform(frame.Inverse,float4(xy,1,1));
            if (min(abs(nearH.w),abs(farH.w))<=frame.Atlas.z) continue;
            float3 nearPoint=nearH.xyz/nearH.w,axis=farH.xyz/farH.w-nearPoint;float span=length(axis);
            if (!isfinite(span) || span<=frame.Atlas.z) continue;
            float3 probe=nearPoint+axis*(pow((float(p.z)+0.5f)/frame.Grid.z,frame.Distribution.y)*min(span,frame.Distribution.x)/span);
            float separation=length(world-probe);if (!isfinite(separation) || separation>radius) continue;
        }
        accepted+=weight;
        for (uint harmonic=0;harmonic<CoefficientCount;++harmonic)
            incoming+=weight*projectedCylinder(harmonic,tangent)*atlas.read(uint2(p.x+harmonic*frame.Grid.x,row)).xyz;
    }
    return max(incoming/frame.Distribution.w,float3(0))*frame.Distribution.z+frame.Ambient.xyz*saturate(1-accepted);
}
fragment float4 lumenFiberIndirect(FiberFragment input [[stage_in]],constant LumenFiberFrame& frame [[buffer(0)]],
    texture2d<float,access::read> volume [[texture(0)]])
{
    if (dot(float4(input.World,1),frame.ClipPlane)<0) discard_fragment();
    float squared=dot(input.Tangent,input.Tangent);float3 tangent=squared>frame.Atlas.w?input.Tangent*rsqrt(squared):float3(0,1,0);
    float3 albedo=saturate(input.Reflectance*mix(frame.RootTint.xyz,frame.TipTint.xyz,saturate(input.CrossHeight.y)));
    return float4(clamp(albedo*fiberIllumination(input.World,tangent,frame,volume)*frame.RootTint.w,0.0f,MaximumHalf),0);
}
struct FiberCompositeVertex { float4 Position [[position]]; };
vertex FiberCompositeVertex lumenFiberCompositeVertex(uint index [[vertex_id]])
{ float2 p=float2((index<<1)&2,index&2);return {float4(p*float2(2,-2)+float2(-1,1),0,1)}; }
fragment float4 lumenFiberComposite(FiberCompositeVertex input [[stage_in]],texture2d<float,access::read> indirect [[texture(0)]],
    texture2d<float,access::read> scene [[texture(1)]])
{ uint2 p=uint2(input.Position.xy);float4 original=scene.read(p);return float4(clamp(original.xyz+indirect.read(p).xyz,0.0f,MaximumHalf),original.w); }
