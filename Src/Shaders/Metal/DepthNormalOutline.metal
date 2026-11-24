#include <metal_stdlib>
using namespace metal;
// Fullscreen depth and world-normal discontinuities form geometric contours.
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
    float4 DepthNormalEdge;
};
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex PostVertexMain(uint id [[vertex_id]])
{
    float2 corner = id == 0 ? float2(-1.0, -1.0) :
        id == 1 ? float2(3.0, -1.0) : float2(-1.0, 3.0);
    return {float4(corner, 0.0, 1.0)};
}
fragment float4 PostFragmentMain(ScreenVertex input [[stage_in]],
    constant PostConstants& parameters [[buffer(0)]],
    texture2d<float> scene [[texture(0)]], depth2d<float, access::read> depthImage [[texture(1)]],
    texture2d<float> normalImage [[texture(2)]])
{
    int2 colorExtent = int2(scene.get_width(), scene.get_height());
    int2 extent = int2(depthImage.get_width(), depthImage.get_height());
    int2 pixel = clamp(int2(input.Position.xy * float2(extent) / float2(colorExtent)), int2(0), extent - 1);
    float4 color = scene.read(uint2(input.Position.xy));
    float depth = depthImage.read(uint2(pixel));
    float3 normal = normalize(normalImage.read(uint2(pixel)).xyz + float3(1.0e-6));
    float depthEdge = 0, normalEdge = 0;
    const int2 offsets[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    for (int i = 0; i < 4; ++i)
    {
        int2 neighbor = clamp(pixel + offsets[i], int2(0), extent - 1);
        float nextDepth = depthImage.read(uint2(neighbor));
        depthEdge = max(depthEdge, abs(nextDepth - depth) / max(1.0 - max(nextDepth, depth), 0.001));
        float3 nextNormal = normalize(normalImage.read(uint2(neighbor)).xyz + float3(1.0e-6));
        if (depth < 0.999999 && nextDepth < 0.999999)
            normalEdge = max(normalEdge, 1.0 - dot(normal, nextNormal));
    }
    float edge = max(smoothstep(parameters.DepthNormalEdge.x, parameters.DepthNormalEdge.x * 2.0, depthEdge),
        smoothstep(parameters.DepthNormalEdge.y, parameters.DepthNormalEdge.y + 0.1, normalEdge));
    return float4(color.rgb * (1.0 - edge * parameters.DepthNormalEdge.z), color.a);
}
