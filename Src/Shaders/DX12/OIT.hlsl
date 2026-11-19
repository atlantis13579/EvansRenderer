// Independently authored DX12 depth-peeling stages. Opaque depth bounds the
// nearest surviving layer; strict comparison against the previous layer prevents
// duplicate composition. Capture and its lighting replay require identical
// raster positions, not an approximate depth match.
// Everitt, Interactive Order-Independent Transparency (2001).
// Bavoil and Myers, Order Independent Transparency with Dual Depth Peeling (2008),
// section 2.2: https://developer.download.nvidia.com/SDK/10/opengl/src/dual_depth_peeling/doc/DualDepthPeeling.pdf
cbuffer OITConstants : register(b0)
{
    row_major float4x4 World;
    row_major float4x4 ViewProjection;
    float4 Material;
    float4 Viewport;
    uint4 Options;
};
static const float FarDepth = 1, HalfFloatMaximum = 65504;
static const float DiagnosticOpacity = 0.75f;
static const float3 DiagnosticTint = float3(1,0,1);

#if defined(EVANS_OIT_CAPTURE)
Texture2D<float> OpaqueDepth : register(t0);
Texture2D<float> PreviousDepth : register(t1);
Texture2D<float4> DiffuseMap : register(t2);
Texture2D<float4> OpacityMap : register(t3);
SamplerState MaterialSampler : register(s0);
struct InputVertex
{
    float3 Position:POSITION;
    float3 Normal:NORMAL;
    float4 Color:COLOR0;
    float2 Coordinates:TEXCOORD0;
};
struct RasterVertex
{
    precise float4 Position:SV_POSITION;
    float2 Coordinates:TEXCOORD0;
    float Alpha:COLOR0;
};
RasterVertex VSMain(InputVertex input)
{
    RasterVertex output;
    float4 world = mul(World,float4(input.Position,1));
    output.Position = mul(ViewProjection,world);
    output.Coordinates = float2(input.Coordinates.x,1-input.Coordinates.y);
    output.Alpha = input.Color.a;
    return output;
}
void PSMain(RasterVertex input)
{
    int3 coordinate = int3(int2(input.Position.xy),0);
    float currentDepth = input.Position.z;
    if (currentDepth >= OpaqueDepth.Load(coordinate)) discard;
    if (Options.x == 0 && currentDepth <= PreviousDepth.Load(coordinate)) discard;
    float diffuseAlpha = Options.y != 0 ? DiffuseMap.Sample(MaterialSampler,input.Coordinates).a : 1;
    float mapAlpha = Options.z != 0 ? OpacityMap.Sample(MaterialSampler,input.Coordinates).r : 1;
    float coverage = input.Alpha*Material.x*diffuseAlpha*mapAlpha;
    if (!isfinite(coverage)) discard;
    coverage = saturate(coverage);
    if (coverage <= 0 || coverage < Material.y) discard;
}
#else
Texture2D<float4> LayerColor : register(t4);
Texture2D<float4> PreviousAccumulation : register(t5);
Texture2D<float4> SceneColor : register(t6);
Texture2D<float> CapturedDepth : register(t7);
struct FullscreenVertex { float4 Position:SV_POSITION; };
FullscreenVertex VSMain(uint vertexId:SV_VertexID)
{
    FullscreenVertex output;
    uint horizontal = (vertexId<<1)&2, vertical = vertexId&2;
    output.Position = float4(float2(horizontal,vertical)*2-1,0,1);
    return output;
}
float4 PSMain(FullscreenVertex input):SV_TARGET
{
    int3 coordinate = int3(int2(input.Position.xy),0);
    float4 accumulated = PreviousAccumulation.Load(coordinate);
#if defined(EVANS_OIT_ACCUMULATE)
    if (CapturedDepth.Load(coordinate) >= FarDepth) return accumulated;
    float4 surface = LayerColor.Load(coordinate);
    float transmittance = saturate(accumulated.a);
    float3 radiance = accumulated.rgb+transmittance*surface.rgb;
    return float4(radiance,transmittance*(1-saturate(surface.a)));
#else
    // RGB is premultiplied radiance; accumulation alpha stores T, not coverage.
    float4 opaque = SceneColor.Load(coordinate);
    float transmittance = saturate(accumulated.a);
    float3 radiance = accumulated.rgb+transmittance*opaque.rgb;
    if (Options.w != 0 && CapturedDepth.Load(coordinate) < FarDepth)
        radiance = lerp(radiance,DiagnosticTint,DiagnosticOpacity);
    return float4(clamp(radiance,0,HalfFloatMaximum),1-transmittance+transmittance*saturate(opaque.a));
#endif
}
#endif
