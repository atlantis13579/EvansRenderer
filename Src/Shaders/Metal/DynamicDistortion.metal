#include <metal_stdlib>
using namespace metal;
// Animate two orthogonal screen waves; amplitudes are measured in pixels.
struct PostConstants
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
    float4 NprEdge;
    float4 NprStyle;
    float4 NprHatch;
    float4 NprTone;
    float4 DynamicWarp;
};
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex PostVertexMain(uint id [[vertex_id]])
{
    float2 corner = id == 0 ? float2(-1.0, -1.0) :
        id == 1 ? float2(3.0, -1.0) : float2(-1.0, 3.0);
    return {float4(corner, 0.0, 1.0)};
}
fragment float4 PostFragmentMain(ScreenVertex input [[stage_in]],
    constant PostConstants& parameters [[buffer(0)]], texture2d<float> scene [[texture(0)]])
{
    constexpr sampler LinearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    constexpr float Tau = 6.28318530718;
    float2 uv = input.Position.xy * parameters.InverseExtentStrengths.xy;
    float t = parameters.DynamicWarp.w * parameters.DynamicWarp.z;
    float2 wave = float2(sin((uv.y * parameters.DynamicWarp.y + t) * Tau),
        sin((uv.x * parameters.DynamicWarp.y * 0.7 - t * 0.6) * Tau));
    float2 donor = uv + wave * parameters.DynamicWarp.x * parameters.InverseExtentStrengths.xy * float2(1.0, 0.5);
    return scene.sample(LinearClamp, donor, level(0.0));
}
