#include <metal_stdlib>
using namespace metal;
// Radial lens falloff is a multiplicative exposure adjustment in linear RGB.
struct PostConstants { float4 InverseExtentStrengths; uint4 GrainSeedReserved; };
struct PostVertex { float4 Position [[position]]; };
vertex PostVertex PostVertexMain(uint vertexId [[vertex_id]])
{
    const float OversizedCorner = 3.0;
    float2 corner = vertexId == 0 ? float2(-1.0, -1.0) :
        vertexId == 1 ? float2(OversizedCorner, -1.0) : float2(-1.0, OversizedCorner);
    return {float4(corner, 0.0, 1.0)};
}
fragment float4 PostFragmentMain(PostVertex input [[stage_in]],
    constant PostConstants& constants [[buffer(0)]], texture2d<float> scene [[texture(0)]])
{
    constexpr sampler ColorSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float InnerRadiusSquared = 0.2;
    const float OuterRadiusSquared = 0.8;
    float2 uv = input.Position.xy * constants.InverseExtentStrengths.xy;
    float2 centered = uv * 2.0 - 1.0;
    float radiusSquared = dot(centered, centered) * 0.5;
    float attenuation = 1.0 - constants.InverseExtentStrengths.z * smoothstep(InnerRadiusSquared, OuterRadiusSquared, radiusSquared);
    float4 color = scene.sample(ColorSampler, uv);
    return float4(color.rgb * attenuation, color.a);
}
