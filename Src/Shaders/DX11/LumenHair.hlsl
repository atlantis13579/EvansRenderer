// Low-frequency projected-cylinder receiver. Normalizing sqrt(1-mu^2) by pi^2
// gives SH eigenvalues {1,0,-1/8}. The atlas already contains cosine-convolved
// irradiance: undo its band scale before applying this cylinder filter.
// Direct R/TT/TRT remains in HairFur.hlsl; this is not multiple scattering.
cbuffer FiberVolume : register(b1)
{
    row_major float4x4 Camera,InverseCamera;
    uint4 Grid;
    float4 Distribution,Atlas,RootTint,TipTint,Ambient,ClipPlane;
};
Texture2D<float4> Volume : register(t0);
static const float Pi=3.14159265358979323846,MaximumHalf=65504;
static const uint VisibilityBit=2,Coefficients=9;
float CylinderBasis(uint i,float3 t)
{
    if (i==0) return sqrt(1/(4*Pi))/Pi;
    const float evenScale=-1/(2*Pi);
    if (i==4) return evenScale*sqrt(15/(4*Pi))*t.x*t.y;
    if (i==5) return evenScale*sqrt(15/(4*Pi))*t.y*t.z;
    if (i==6) return evenScale*sqrt(5/(16*Pi))*(3*t.z*t.z-1);
    if (i==7) return evenScale*sqrt(15/(4*Pi))*t.x*t.z;
    if (i==8) return evenScale*sqrt(15/(16*Pi))*(t.x*t.x-t.y*t.y);
    return 0;
}
float3 FilterVolume(float3 world,float3 tangent)
{
    float4 clip=mul(Camera,float4(world,1));
    if (!all(isfinite(clip)) || clip.w<=Atlas.z) return Ambient.rgb;
    float3 ndc=clip.xyz/clip.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return Ambient.rgb;
    float4 a=mul(InverseCamera,float4(ndc.xy,0,1)),b=mul(InverseCamera,float4(ndc.xy,1,1));
    if (min(abs(a.w),abs(b.w))<=Atlas.z) return Ambient.rgb;
    float3 start=a.xyz/a.w; float extent=min(length(b.xyz/b.w-start),Distribution.x),distance=length(world-start);
    if (!isfinite(extent) || extent<=Atlas.z || !isfinite(distance) || distance>extent) return Ambient.rgb;
    float2 uv=float2(ndc.x+1,1-ndc.y)*0.5;
    float3 cell=clamp(float3(uv*Grid.xy,pow(saturate(distance/extent),1/Distribution.y)*Grid.z)-0.5,0,float3(Grid.xyz)-1);
    uint3 lower=uint3(floor(cell)); float3 fraction=frac(cell),light=0; float mass=0;
    [loop] for (uint corner=0;corner<8;++corner)
    {
        uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(lower+bit,Grid.xyz-1);
        float3 w=float3(bit.x?fraction.x:1-fraction.x,bit.y?fraction.y:1-fraction.y,bit.z?fraction.z:1-fraction.z);
        float weight=w.x*w.y*w.z; if (weight<=0) continue; uint row=p.z*Grid.y+p.y;
        if ((Grid.w&VisibilityBit)!=0)
        {
            float support=Volume.Load(int3(p.x,row,0)).a; if (support<=0) continue;
            float2 xy=float2(2*(float(p.x)+0.5)/Grid.x-1,1-2*(float(p.y)+0.5)/Grid.y);
            float4 nearH=mul(InverseCamera,float4(xy,0,1)),farH=mul(InverseCamera,float4(xy,1,1));
            if (min(abs(nearH.w),abs(farH.w))<=Atlas.z) continue;
            float3 nearPoint=nearH.xyz/nearH.w,axis=farH.xyz/farH.w-nearPoint; float span=length(axis);
            if (!isfinite(span) || span<=Atlas.z) continue;
            float depth=pow((float(p.z)+0.5)/Grid.z,Distribution.y);
            float separation=length(world-nearPoint-axis*(depth*min(span,Distribution.x)/span));
            if (!isfinite(separation) || separation>support) continue;
        }
        mass+=weight;
        [unroll] for (uint coefficient=0;coefficient<Coefficients;++coefficient)
            light+=weight*CylinderBasis(coefficient,tangent)*Volume.Load(int3(p.x+coefficient*Grid.x,row,0)).rgb;
    }
    return max(light/Distribution.w,0)*Distribution.z+Ambient.rgb*saturate(1-mass);
}
struct FiberInput
{
    float4 Position:SV_POSITION; float3 World:TEXCOORD0,Tangent:TEXCOORD1,Reflectance:TEXCOORD2;
    float3 CrossHeight:TEXCOORD4;
};
float4 PSIndirect(FiberInput input):SV_TARGET
{
    if (dot(float4(input.World,1),ClipPlane)<0) discard;
    float squared=dot(input.Tangent,input.Tangent);
    float3 tangent=squared>Atlas.w?input.Tangent*rsqrt(squared):float3(0,1,0);
    float3 albedo=saturate(input.Reflectance*lerp(RootTint.rgb,TipTint.rgb,saturate(input.CrossHeight.y)));
    return float4(clamp(albedo*FilterVolume(input.World,tangent)*RootTint.w,0,MaximumHalf),0);
}
Texture2D<float4> Scene : register(t1);
float4 VSCompose(uint vertex:SV_VertexID):SV_POSITION
{ float2 p=float2((vertex<<1)&2,vertex&2); return float4(p*float2(2,-2)+float2(-1,1),0,1); }
float4 PSCompose(float4 position:SV_POSITION):SV_TARGET
{
    int3 pixel=int3(int2(position.xy),0); float4 scene=Scene.Load(pixel);
    return float4(clamp(scene.rgb+Volume.Load(pixel).rgb,0,MaximumHalf),scene.a);
}
