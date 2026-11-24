#include <metal_stdlib>
using namespace metal;
// Sobel color gradients draw a screen-space contour while preserving alpha.
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
    float3 tl = at(scene, pixel + int2(-1, -1), extent), tc = at(scene, pixel + int2(0, -1), extent);
    float3 tr = at(scene, pixel + int2(1, -1), extent), ml = at(scene, pixel + int2(-1, 0), extent);
    float3 mr = at(scene, pixel + int2(1, 0), extent), bl = at(scene, pixel + int2(-1, 1), extent);
    float3 bc = at(scene, pixel + int2(0, 1), extent), br = at(scene, pixel + int2(1, 1), extent);
    float3 gx = tr + 2.0 * mr + br - tl - 2.0 * ml - bl;
    float3 gy = bl + 2.0 * bc + br - tl - 2.0 * tc - tr;
    float edge = length(float2(length(gx), length(gy))) * 0.25;
    float ink = smoothstep(parameters.NprTone.z, parameters.NprTone.z + 0.1, edge) * parameters.NprTone.w;
    return float4(center.rgb * (1.0 - ink), center.a);
}
