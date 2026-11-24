#include <metal_stdlib>
using namespace metal;
// Bilateral pigment diffusion, tonal quantization and static paper granulation.
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
    float4 Painterly;
};
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex PostVertexMain(uint id [[vertex_id]])
{
    float2 corner = id == 0 ? float2(-1.0, -1.0) :
        id == 1 ? float2(3.0, -1.0) : float2(-1.0, 3.0);
    return {float4(corner, 0.0, 1.0)};
}
float3 at(texture2d<float> scene, int2 pixel, int2 extent)
{
    return scene.read(uint2(clamp(pixel, int2(0), extent - 1))).rgb;
}
fragment float4 PostFragmentMain(ScreenVertex input [[stage_in]],
    constant PostConstants& parameters [[buffer(0)]], texture2d<float> scene [[texture(0)]])
{
    int2 extent = int2(scene.get_width(), scene.get_height());
    int2 pixel = int2(input.Position.xy);
    float4 center = scene.read(uint2(pixel));
    int radius = clamp(int(parameters.Painterly.z + 0.5), 1, 4);
    float sigma = max(float(radius) * 0.7, 0.7);
    float3 sum = 0;
    float weightSum = 0;
    for (int y = -radius; y <= radius; ++y)
    for (int x = -radius; x <= radius; ++x)
    {
        float3 sampleColor = at(scene, pixel + int2(x, y), extent);
        float3 difference = sampleColor - center.rgb;
        float weight = exp(-float(x * x + y * y) / (2.0 * sigma * sigma) - dot(difference, difference) * 8.0);
        sum += sampleColor * weight;
        weightSum += weight;
    }
    float3 pigment = sum / max(weightSum, 1.0e-6);
    pigment = floor(pigment * 12.0 + 0.5) / 12.0;
    float paper = fract(sin(dot(float2(pixel), float2(12.9898, 78.233))) * 43758.5453) - 0.5;
    pigment = clamp(pigment + paper * 0.025, 0.0, 1.0);
    return float4(mix(center.rgb, pigment, parameters.Painterly.w), center.a);
}
