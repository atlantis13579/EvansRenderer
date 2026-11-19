// Independent D3D12 low-frequency fiber receiver. Projected cylinder area has
// a normalized even angular kernel; its first SH multipliers are 1,0,-1/8.
// The incident atlas stores Lambert-convolved SH. Undo that convolution before
// evaluating this filter, retaining the direct fiber BSDF in HairFur.hlsl.
cbuffer IndirectFiberParameters:register(b1)
{
    row_major float4x4 WorldToClip,ClipToWorld;
    uint4 Shape;
    float4 Range,Image,Root,Tip,Fallback,ClipEquation;
};
Texture2D<float4> Field:register(t0);
static const float SpherePi=3.14159265358979323846,HalfLimit=65504;
static const uint SupportFlag=2,HarmonicCount=9;
float projectedFiberHarmonic(uint band,float3 axis)
{
    if (band==0) return sqrt(1/(4*SpherePi))/SpherePi;
    const float secondBand=-1/(2*SpherePi);
    switch (band)
    {
        case 4:return secondBand*sqrt(15/(4*SpherePi))*axis.x*axis.y;
        case 5:return secondBand*sqrt(15/(4*SpherePi))*axis.y*axis.z;
        case 6:return secondBand*sqrt(5/(16*SpherePi))*(3*axis.z*axis.z-1);
        case 7:return secondBand*sqrt(15/(4*SpherePi))*axis.x*axis.z;
        case 8:return secondBand*sqrt(15/(16*SpherePi))*(axis.x*axis.x-axis.y*axis.y);
    }
    return 0;
}
float3 fiberIncident(float3 world,float3 axis)
{
    float4 projected=mul(WorldToClip,float4(world,1));
    if (!all(isfinite(projected)) || projected.w<=Image.z) return Fallback.xyz;
    float3 ndc=projected.xyz/projected.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return Fallback.xyz;
    float4 nearH=mul(ClipToWorld,float4(ndc.xy,0,1)),farH=mul(ClipToWorld,float4(ndc.xy,1,1));
    if (min(abs(nearH.w),abs(farH.w))<=Image.z) return Fallback.xyz;
    float3 nearPoint=nearH.xyz/nearH.w; float span=min(length(farH.xyz/farH.w-nearPoint),Range.x),distance=length(world-nearPoint);
    if (!isfinite(span) || !isfinite(distance) || span<=Image.z || distance>span) return Fallback.xyz;
    float3 coordinate=float3((ndc.xy*float2(0.5,-0.5)+0.5)*float2(Shape.xy),pow(saturate(distance/span),1/Range.y)*Shape.z)-0.5;
    coordinate=clamp(coordinate,0,float3(Shape.xyz)-1);
    uint3 first=uint3(floor(coordinate)); float3 f=frac(coordinate),result=0; float accepted=0;
    [loop] for (uint tap=0;tap<8;++tap)
    {
        uint3 corner=uint3(tap&1,(tap>>1)&1,tap>>2),p=min(first+corner,Shape.xyz-1);
        float3 weight=float3(corner.x?f.x:1-f.x,corner.y?f.y:1-f.y,corner.z?f.z:1-f.z);
        float mass=weight.x*weight.y*weight.z; if (mass<=0) continue;
        uint row=p.z*Shape.y+p.y;
        if ((Shape.w&SupportFlag)!=0)
        {
            float radius=Field.Load(int3(p.x,row,0)).w; if (radius<=0) continue;
            float2 xy=(float2(p.xy)+0.5)/float2(Shape.xy)*float2(2,-2)+float2(-1,1);
            float4 a=mul(ClipToWorld,float4(xy,0,1)),b=mul(ClipToWorld,float4(xy,1,1));
            if (min(abs(a.w),abs(b.w))<=Image.z) continue;
            float3 start=a.xyz/a.w,direction=b.xyz/b.w-start; float extent=length(direction);
            if (!isfinite(extent) || extent<=Image.z) continue;
            float3 probe=start+direction*(pow((float(p.z)+0.5)/Shape.z,Range.y)*min(extent,Range.x)/extent);
            float separation=length(world-probe); if (!isfinite(separation) || separation>radius) continue;
        }
        accepted+=mass;
        [unroll] for (uint harmonic=0;harmonic<HarmonicCount;++harmonic)
            result+=mass*projectedFiberHarmonic(harmonic,axis)*Field.Load(int3(p.x+harmonic*Shape.x,row,0)).xyz;
    }
    return max(result/Range.w,0)*Range.z+Fallback.xyz*saturate(1-accepted);
}
struct FiberSample
{
    // Keep the complete primary VS signature: DX12 links hardware registers,
    // including the scalar depth packed beside the preceding three-vector.
    float4 Pixel:SV_POSITION;
    float3 Point:TEXCOORD0,Axis:TEXCOORD1,Color:TEXCOORD2,Emitted:TEXCOORD3,Parameters:TEXCOORD4;
    float LinearDepth:TEXCOORD5;
    float4 OldClip:TEXCOORD6;
};
float4 PSIndirect(FiberSample fragment):SV_TARGET
{
    if (dot(float4(fragment.Point,1),ClipEquation)<0) discard;
    float square=dot(fragment.Axis,fragment.Axis); float3 axis=square>Image.w?fragment.Axis*rsqrt(square):float3(0,1,0);
    float3 color=saturate(fragment.Color*lerp(Root.xyz,Tip.xyz,saturate(fragment.Parameters.y)));
    return float4(clamp(color*fiberIncident(fragment.Point,axis)*Root.w,0,HalfLimit),0);
}
Texture2D<float4> Background:register(t1);
float4 VSCompose(uint id:SV_VertexID):SV_POSITION
{ float2 corner=float2((id<<1)&2,id&2);return float4(corner*float2(2,-2)+float2(-1,1),0,1); }
float4 PSCompose(float4 pixel:SV_POSITION):SV_TARGET
{
    int3 address=int3(int2(pixel.xy),0);float4 original=Background.Load(address);
    return float4(clamp(original.xyz+Field.Load(address).xyz,0,HalfLimit),original.w);
}
