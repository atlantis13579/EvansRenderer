#include <metal_stdlib>
using namespace metal;
// Lottes, FXAA (2009). Edge direction and luma use encoded sRGB; texture
// filtering and render-target writes retain Metal's linear/sRGB contract.
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
float3 ReadDisplay(texture2d<float> scene, float2 uv)
{
    constexpr sampler ColorSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    return DisplayEncode(scene.sample(ColorSampler, uv).rgb);
}
fragment float4 PostFragmentMain(PostVertex input [[stage_in]],
    constant PostConstants& constants [[buffer(0)]], texture2d<float> scene [[texture(0)]])
{
    constexpr sampler ColorSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float3 LumaWeights = float3(0.299, 0.587, 0.114);
    const float MinimumContrast = 1.0 / 16.0;
    const float RelativeContrast = 1.0 / 8.0;
    const float MinimumReduction = 1.0 / 128.0;
    const float ReductionScale = 1.0 / 8.0;
    const float QuarterWeight = 0.25;
    const float MaximumSpan = 8.0;
    const float OneThird = 1.0 / 3.0;
    float2 texel = constants.InverseExtentStrengths.xy;
    float2 uv = input.Position.xy * texel;
    float4 center = scene.sample(ColorSampler, uv);
    float centerLuma = dot(DisplayEncode(center.rgb), LumaWeights);
    float nw = dot(ReadDisplay(scene, uv + float2(-1.0, -1.0) * texel), LumaWeights);
    float ne = dot(ReadDisplay(scene, uv + float2(1.0, -1.0) * texel), LumaWeights);
    float sw = dot(ReadDisplay(scene, uv + float2(-1.0, 1.0) * texel), LumaWeights);
    float se = dot(ReadDisplay(scene, uv + float2(1.0, 1.0) * texel), LumaWeights);
    float minimumLuma = min(centerLuma, min(min(nw, ne), min(sw, se)));
    float maximumLuma = max(centerLuma, max(max(nw, ne), max(sw, se)));
    if (maximumLuma - minimumLuma < max(MinimumContrast, maximumLuma * RelativeContrast)) return center;
    float2 direction = float2(-(nw + ne - sw - se), nw + sw - ne - se);
    float reduction = max((nw + ne + sw + se) * QuarterWeight * ReductionScale, MinimumReduction);
    direction = clamp(direction / (min(abs(direction.x), abs(direction.y)) + reduction), -MaximumSpan, MaximumSpan) * texel;
    float3 inner = (ReadDisplay(scene, uv + direction * (OneThird - 0.5)) +
        ReadDisplay(scene, uv + direction * (2.0 * OneThird - 0.5))) * 0.5;
    float3 outer = inner * 0.5 + (ReadDisplay(scene, uv - direction * 0.5) + ReadDisplay(scene, uv + direction * 0.5)) * QuarterWeight;
    float lumaOuter = dot(outer, LumaWeights);
    float3 filtered = lumaOuter < minimumLuma || lumaOuter > maximumLuma ? inner : outer;
    return float4(DisplayDecode(filtered), center.a);
}
