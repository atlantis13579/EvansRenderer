#include <metal_stdlib>
using namespace metal;
// Seeded PCG output adds monochromatic noise in perceptual sRGB. The seed is
// explicit rather than frame-dependent, so parameter round trips are reproducible.
struct PostConstants { float4 InverseExtentStrengths; uint4 GrainSeedReserved; };
struct PostVertex { float4 Position [[position]]; };
vertex PostVertex PostVertexMain(uint vertexId [[vertex_id]])
{
    const float OversizedCorner = 3.0;
    float2 corner = vertexId == 0 ? float2(-1.0, -1.0) :
        vertexId == 1 ? float2(OversizedCorner, -1.0) : float2(-1.0, OversizedCorner);
    return {float4(corner, 0.0, 1.0)};
}
float3 DisplayEncode(float3 linear)
{
    const float LinearThreshold = 0.0031308;
    const float LinearScale = 12.92;
    const float PowerScale = 1.055;
    const float PowerOffset = 0.055;
    const float PowerExponent = 2.4;
    return select(PowerScale * pow(max(linear, 0.0), 1.0 / PowerExponent) - PowerOffset,
        linear * LinearScale, linear <= LinearThreshold);
}
float3 DisplayDecode(float3 display)
{
    const float EncodedThreshold = 0.04045;
    const float LinearScale = 12.92;
    const float PowerScale = 1.055;
    const float PowerOffset = 0.055;
    const float PowerExponent = 2.4;
    return select(pow((display + PowerOffset) / PowerScale, PowerExponent),
        display / LinearScale, display <= EncodedThreshold);
}
fragment float4 PostFragmentMain(PostVertex input [[stage_in]],
    constant PostConstants& constants [[buffer(0)]], texture2d<float> scene [[texture(0)]])
{
    constexpr sampler ColorSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const uint PixelMultiplierX = 1973u;
    const uint PixelMultiplierY = 9277u;
    const uint PcgMultiplier = 747796405u;
    const uint PcgIncrement = 2891336453u;
    const uint PcgOutputMultiplier = 277803737u;
    const uint PcgHighShift = 28u;
    const uint PcgShiftOffset = 4u;
    const uint PcgFinalShift = 22u;
    const uint NoiseMask = 0x00ffffffu;
    const float NoiseNormalization = 16777215.0;
    uint2 pixel = uint2(floor(input.Position.xy));
    uint inputValue = (pixel.x * PixelMultiplierX + pixel.y * PixelMultiplierY) ^ constants.GrainSeedReserved.x;
    uint state = inputValue * PcgMultiplier + PcgIncrement;
    uint word = ((state >> ((state >> PcgHighShift) + PcgShiftOffset)) ^ state) * PcgOutputMultiplier;
    uint hash = (word >> PcgFinalShift) ^ word;
    float noise = float(hash & NoiseMask) / NoiseNormalization - 0.5;
    float4 color = scene.sample(ColorSampler, input.Position.xy * constants.InverseExtentStrengths.xy);
    float3 display = saturate(DisplayEncode(color.rgb) + noise * constants.InverseExtentStrengths.w);
    return float4(DisplayDecode(display), color.a);
}
