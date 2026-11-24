#include <metal_stdlib>
using namespace metal;
// Sample the center of each screen-aligned pixel block in linear light.
struct PostConstants
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
    float4 NprEdge;
    float4 NprStyle;
    float4 NprHatch;
    float4 NprTone;
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
    float2 extent = 1.0 / parameters.InverseExtentStrengths.xy;
    float block = max(1.0, floor(parameters.NprTone.y + 0.5));
    float2 pixel = floor(input.Position.xy / block) * block + block * 0.5;
    float2 uv = clamp(pixel, float2(0.5), extent - 0.5) * parameters.InverseExtentStrengths.xy;
    return scene.sample(LinearClamp, uv, level(0.0));
}
