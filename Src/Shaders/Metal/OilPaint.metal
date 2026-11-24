#include <metal_stdlib>
using namespace metal;
// Four-quadrant Kuwahara filtering retains low-variance color regions.
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
    int radius = clamp(int(parameters.Painterly.x + 0.5), 1, 4);
    float bestVariance = 1.0e20;
    float3 bestColor = center.rgb;
    for (int quadrant = 0; quadrant < 4; ++quadrant)
    {
        int xStart = (quadrant & 1) != 0 ? 0 : -radius;
        int yStart = (quadrant & 2) != 0 ? 0 : -radius;
        float3 sum = 0, sumSquared = 0;
        for (int y = 0; y <= radius; ++y)
        for (int x = 0; x <= radius; ++x)
        {
            float3 value = at(scene, pixel + int2(xStart + x, yStart + y), extent);
            sum += value;
            sumSquared += value * value;
        }
        float count = float((radius + 1) * (radius + 1));
        float3 mean = sum / count;
        float variance = dot(max(sumSquared / count - mean * mean, float3(0)), float3(0.2126, 0.7152, 0.0722));
        if (variance < bestVariance) { bestVariance = variance; bestColor = mean; }
    }
    return float4(mix(center.rgb, bestColor, parameters.Painterly.y), center.a);
}
