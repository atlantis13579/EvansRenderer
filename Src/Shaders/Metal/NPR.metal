#include <metal_stdlib>
using namespace metal;
// Independent Metal XDoG, tonal crosshatching and posterization.
// Winnemoeller et al. (2012): https://www.kyprianidis.com/p/cag2012/
// Screen-space strokes do not track moving surfaces. No temporal history,
// flow-aligned filter, paper texture or extra material attribute is required.
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
struct PostVertex { float4 Position [[position]]; };
vertex PostVertex PostVertexMain(uint index [[vertex_id]])
{
    const float OversizedCorner = 3.0;
    float2 corner = index == 0 ? float2(-1, -1) : index == 1 ? float2(OversizedCorner, -1) : float2(-1, OversizedCorner);
    return {float4(corner, 0, 1)};
}
float3 encodeDisplay(float3 linear)
{
    const float Threshold = 0.0031308, Slope = 12.92;
    const float PowerScale = 1.055, PowerOffset = 0.055, Exponent = 2.4;
    return select(PowerScale * pow(saturate(linear), 1.0 / Exponent) - PowerOffset,
        linear * Slope, linear <= Threshold);
}
float3 decodeDisplay(float3 encoded)
{
    const float Threshold = 0.04045, Slope = 12.92;
    const float PowerScale = 1.055, PowerOffset = 0.055, Exponent = 2.4;
    return select(pow((encoded + PowerOffset) / PowerScale, Exponent), encoded / Slope, encoded <= Threshold);
}
float contour(texture2d<float> scene, int2 center, constant PostConstants& settings)
{
    const float ScaleRatio = 1.6, GaussianSupport = 3.0;
    const int MaximumSupport = 8; // Covers sigma <= 1.5 at the wide Gaussian scale.
    const float3 LumaWeights = float3(0.2126, 0.7152, 0.0722);
    float2 scales = settings.NprEdge.x * float2(1, ScaleRatio);
    int radius = min(MaximumSupport, int(ceil(GaussianSupport * scales.y)));
    int2 extent = int2(scene.get_width(), scene.get_height());
    float2 totals = 0, normalization = 0;
    for (int row = -radius; row <= radius; ++row)
        for (int column = -radius; column <= radius; ++column)
        {
            uint2 donor = uint2(clamp(center + int2(column, row), int2(0), extent - 1));
            float luma = dot(encodeDisplay(scene.read(donor).rgb), LumaWeights);
            float2 kernel = exp(-float(row * row + column * column) / (2 * scales * scales));
            totals += luma * kernel;
            normalization += kernel;
        }
    float2 mean = totals / normalization;
    float difference = mean.x - settings.NprEdge.y * mean.y;
    float e = exp(2 * min(0.0f, settings.NprEdge.w * (difference - settings.NprEdge.z)));
    return 2 * e / (1 + e); // Overflow-safe white/tanh XDoG response.
}
float lineCoverage(float2 pixel, float2 axis, float spacing, float width)
{
    float distance = abs(fract(dot(pixel, axis) / spacing + 0.5) - 0.5) * spacing;
    float halfWidth = width * 0.5;
    float footprint = (abs(axis.x) + abs(axis.y)) * 0.5;
    return 1 - smoothstep(max(0.0f, halfWidth - footprint), halfWidth + footprint, distance);
}
fragment float4 PostFragmentMain(PostVertex vertex [[stage_in]],
    constant PostConstants& settings [[buffer(0)]], texture2d<float> scene [[texture(0)]])
{
    float4 source = scene.read(uint2(vertex.Position.xy));
    float3 display = saturate(encodeDisplay(source.rgb));
    const float3 LumaWeights = float3(0.2126, 0.7152, 0.0722);
    const float LightTone = 0.25, MiddleTone = 0.5, DarkTone = 0.75, ToneFade = 0.1;
    const float DiagonalNormalization = 0.7071067811865475;
    float3 thresholds = float3(LightTone, MiddleTone, DarkTone);
    float darkness = 1 - dot(display, LumaWeights);
    float3 layers = smoothstep(thresholds - ToneFade, thresholds + ToneFade, float3(darkness));
    float2 axis = settings.NprHatch.zw;
    float spacing = settings.NprHatch.x, width = settings.NprHatch.y;
    float3 strokes = float3(lineCoverage(vertex.Position.xy, axis, spacing, width),
        lineCoverage(vertex.Position.xy, float2(-axis.y, axis.x), spacing, width),
        lineCoverage(vertex.Position.xy, float2(axis.x - axis.y, axis.x + axis.y) * DiagonalNormalization, spacing, width));
    float3 transmission = 1 - strokes * layers * settings.NprStyle.y;
    float edge = settings.NprStyle.x > 0 ? contour(scene, int2(vertex.Position.xy), settings) : 1;
    float intervals = settings.NprTone.x - 1;
    float3 tones = floor(display * intervals + 0.5) / intervals;
    float3 paper = mix(float3(1), tones, settings.NprStyle.z);
    float3 inked = paper * mix(1.0f, edge, settings.NprStyle.x) * transmission.x * transmission.y * transmission.z;
    return float4(decodeDisplay(mix(display, inked, settings.NprStyle.w)), source.a);
}
