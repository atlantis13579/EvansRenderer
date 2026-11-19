// DX11 front depth peeling. Everitt (2001), Interactive Order-Independent
// Transparency; Bavoil/Myers (2008), section 2.2, front-to-back compositing:
// https://developer.download.nvidia.com/SDK/10/opengl/src/dual_depth_peeling/doc/DualDepthPeeling.pdf
// Capture selects one distinct D32 depth. The selected material shader then
// shades that depth into a premultiplied layer using canonical source-over order.
// This file independently owns selection, accumulation and final composition.
cbuffer PeelingUniforms : register(b0)
{
    row_major float4x4 World, ViewProjection;
    float4 Material, Viewport;
    uint4 Options;
};
static const float ClearDepth = 1, MaximumHDRValue = 65504;
static const float3 OverflowColor = float3(1,0,1);
static const float OverflowOverlayOpacity = 0.75f;

#ifdef EVANS_OIT_CAPTURE
Texture2D<float> OpaqueDepth : register(t0), PreviousDepth : register(t1);
Texture2D<float4> DiffuseMap : register(t2), OpacityMap : register(t3);
SamplerState MaterialSampler : register(s0);
struct MeshVertex
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 UV : TEXCOORD0;
};
struct PeelVertex
{
    precise float4 Position : SV_POSITION;
    float Opacity : COLOR0;
    float2 UV : TEXCOORD0;
};
PeelVertex VSMain(MeshVertex vertex)
{
    PeelVertex output;
    float4 worldPosition = mul(World,float4(vertex.Position,1));
    output.Position = mul(ViewProjection,worldPosition);
    output.Opacity = vertex.Color.a;
    output.UV = float2(vertex.UV.x,1-vertex.UV.y);
    return output;
}
void PSMain(PeelVertex input)
{
    int3 pixel = int3(int2(input.Position.xy),0);
    float depth = input.Position.z;
    if (depth >= OpaqueDepth.Load(pixel)) discard;
    // No tolerance: adjacent representable depth values are separate layers.
    // The first layer accepts near-plane depth zero without consulting a bound.
    if (Options.x == 0 && depth <= PreviousDepth.Load(pixel)) discard;
    float alpha = input.Opacity*Material.x;
    if (Options.y != 0) alpha *= DiffuseMap.Sample(MaterialSampler,input.UV).a;
    if (Options.z != 0) alpha *= OpacityMap.Sample(MaterialSampler,input.UV).r;
    if (!isfinite(alpha)) discard;
    alpha = saturate(alpha);
    if (alpha <= 0 || alpha < Material.y) discard;
}
#else
Texture2D<float4> LayerColor : register(t4), PreviousAccumulation : register(t5), SceneColor : register(t6);
Texture2D<float> CapturedDepth : register(t7);
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint index : SV_VertexID)
{
    ScreenVertex output;
    float2 corner = float2((index<<1)&2,index&2);
    output.Position = float4(corner*2-1,0,1);
    return output;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    int3 pixel = int3(int2(input.Position.xy),0);
    float4 previous = PreviousAccumulation.Load(pixel);
#ifdef EVANS_OIT_ACCUMULATE
    // D32=1 is empty even if a depth-EQUAL lighting replay wrote far-plane
    // color there. Ignore that color, preserving exhausted pixels across layers.
    if (CapturedDepth.Load(pixel) >= ClearDepth) return previous;
    float4 layer = LayerColor.Load(pixel);
    float remaining = saturate(previous.a);
    return float4(previous.rgb+remaining*layer.rgb,remaining*(1-saturate(layer.a)));
#else
    float4 background = SceneColor.Load(pixel);
    float remaining = saturate(previous.a);
    float3 radiance = previous.rgb+remaining*background.rgb;
    if (Options.w != 0 && CapturedDepth.Load(pixel) < ClearDepth)
        radiance = lerp(radiance,OverflowColor,OverflowOverlayOpacity);
    float coverage = 1-remaining+remaining*saturate(background.a);
    return float4(clamp(radiance,0,MaximumHDRValue),coverage);
#endif
}
#endif
