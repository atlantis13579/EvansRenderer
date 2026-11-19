#include <metal_stdlib>
using namespace metal;
// Metal centered-shutter reconstruction from independent motion vectors.
// McGuire et al., "A Reconstruction Filter for Plausible Motion Blur" (2012):
// https://casual-effects.com/research/McGuire2012Blur/index.html
// Tile/neighbor maxima select a gather axis. Source and destination line footprints
// with soft depth classification approximate a centered box shutter, rather than
// the paper's cone/cylinder kernels. Transparent surfaces are composited afterward.
struct MotionUniforms
{
    float4 World[4];
    float4 ViewProjection[4];
    float4 PreviousWorldViewProjection[4];
    float4 InverseViewProjection[4];
    float4 DepthPlane;
    float4 ViewportExposure; // dimensions, shutter / (2 * interval), maximum radius
    float4 Material; // object/material opacity, alpha cutoff
    uint4 Options; // filter pass, diffuse texture, opacity texture, valid history
};
constant float PixelCenter = 0.5;
constant float MinimumClipW = 1.0e-7;
constant float FarDepth = 1.0;
constant float MaximumHalfFloat = 65504.0;

float4 MultiplyRows(constant float4* rows, float4 position)
{
    return float4(dot(rows[0], position), dot(rows[1], position),
        dot(rows[2], position), dot(rows[3], position));
}
struct FilterVaryings { float4 Position [[position]]; };
vertex FilterVaryings MotionFilterVertex(uint vertexId [[vertex_id]])
{
    const float TriangleExtent = 3.0;
    float2 position = vertexId == 0 ? float2(-1.0) :
        vertexId == 1 ? float2(TriangleExtent, -1.0) : float2(-1.0, TriangleExtent);
    return {float4(position, 0.0, 1.0)};
}
constant uint TileReductionPass = 0;
constant uint NeighborhoodPass = 1;
constant uint CompositePass = 3;
constant int TileSize = 16;
constant int NeighborRadius = 1;
constant int MaximumGatherSamples = 33;
constant float StationaryRadius = 0.5;
constant float PixelWidth = 1.0;
constant float RelativeDepthSoftness = 0.02;
constant float MinimumDepthSoftness = 1.0e-4;
constant float BackgroundDistance = 1.0e20;

float2 HalfShutterTravel(uint2 pixel, constant MotionUniforms& uniforms, texture2d<float, access::read> velocities)
{
    float4 velocity = velocities.read(pixel);
    if (velocity.w <= 0.0 || uniforms.ViewportExposure.z <= 0.0) return float2(0.0);
    float2 travel = velocity.xy * uniforms.ViewportExposure.z;
    float distance = length(travel);
    return distance > uniforms.ViewportExposure.w ? travel * (uniforms.ViewportExposure.w / distance) : travel;
}
float OpticalDepth(uint2 pixel, constant MotionUniforms& uniforms, depth2d<float, access::read> depths)
{
    float depth = depths.read(pixel);
    if (depth >= FarDepth) return BackgroundDistance;
    float2 uv = (float2(pixel) + PixelCenter) / uniforms.ViewportExposure.xy;
    float4 world = MultiplyRows(uniforms.InverseViewProjection, float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), depth, 1.0));
    if (abs(world.w) <= MinimumClipW) return BackgroundDistance;
    return max(dot(uniforms.DepthPlane, float4(world.xyz / world.w, 1.0)), 0.0);
}
float ForegroundContribution(float sourceDepth, float targetDepth)
{
    float softness = max(min(sourceDepth, targetDepth) * RelativeDepthSoftness, MinimumDepthSoftness);
    return saturate(1.0 - (sourceDepth - targetDepth) / softness);
}
float SweptCoverage(float2 offset, float2 travel)
{
    float radius = length(travel);
    if (radius <= StationaryRadius) return 0.0;
    float2 direction = travel / radius;
    float along = abs(dot(offset, direction));
    float across = abs(offset.x * direction.y - offset.y * direction.x);
    return saturate(radius + PixelCenter - along) * saturate(PixelWidth - across) / max(2.0 * radius, PixelWidth);
}
fragment float4 MotionFilterFragment(FilterVaryings input [[stage_in]],
    constant MotionUniforms& uniforms [[buffer(0)]], texture2d<float, access::read> scene [[texture(0)]],
    depth2d<float, access::read> depths [[texture(1)]], texture2d<float, access::read> velocities [[texture(2)]],
    texture2d<float, access::read> maxima [[texture(3)]])
{
    int2 pixel = int2(input.Position.xy);
    int2 dimensions = int2(uniforms.ViewportExposure.xy);
    int2 tileCount = (dimensions + TileSize - 1) / TileSize;
    if (uniforms.Options.x == TileReductionPass)
    {
        float2 maximum = float2(0.0);
        for (int row = 0; row < TileSize; ++row)
        for (int column = 0; column < TileSize; ++column)
        {
            int2 source = pixel * TileSize + int2(column, row);
            if (any(source >= dimensions)) continue;
            float2 candidate = HalfShutterTravel(uint2(source), uniforms, velocities);
            if (dot(candidate, candidate) > dot(maximum, maximum)) maximum = candidate;
        }
        return float4(maximum, 0.0, 0.0);
    }
    if (uniforms.Options.x == NeighborhoodPass)
    {
        float2 maximum = float2(0.0);
        for (int row = -NeighborRadius; row <= NeighborRadius; ++row)
        for (int column = -NeighborRadius; column <= NeighborRadius; ++column)
        {
            uint2 tile = uint2(clamp(pixel + int2(column, row), int2(0), tileCount - 1));
            float2 candidate = maxima.read(tile).xy;
            if (dot(candidate, candidate) > dot(maximum, maximum)) maximum = candidate;
        }
        return float4(maximum, 0.0, 0.0);
    }
    float4 center = scene.read(uint2(pixel));
    if (uniforms.Options.x == CompositePass) return center;
    float2 dominant = maxima.read(uint2(pixel / TileSize)).xy;
    float radius = length(dominant);
    if (radius <= StationaryRadius) return center;
    float2 ownTravel = HalfShutterTravel(uint2(pixel), uniforms, velocities);
    float ownDepth = OpticalDepth(uint2(pixel), uniforms, depths);
    int halfSamples = min((MaximumGatherSamples - 1) / 2, max(1, int(ceil(radius))));
    float sampleSpacing = radius / float(halfSamples);
    float weightSum = 2.0 / max(2.0 * length(ownTravel), PixelWidth);
    float3 colorSum = center.rgb * weightSum;
    for (int sampleIndex = -halfSamples; sampleIndex <= halfSamples; ++sampleIndex)
    {
        if (sampleIndex == 0) continue;
        // rint uses nearest-even ties, matching the other native implementations.
        int2 source = int2(rint(float2(pixel) + dominant * (float(sampleIndex) / float(halfSamples))));
        if (any(source < int2(0)) || any(source >= dimensions) || all(source == pixel)) continue;
        float2 offset = float2(source - pixel);
        float depth = OpticalDepth(uint2(source), uniforms, depths);
        float weight = sampleSpacing * (ForegroundContribution(depth, ownDepth) *
            SweptCoverage(-offset, HalfShutterTravel(uint2(source), uniforms, velocities)) +
            ForegroundContribution(ownDepth, depth) * SweptCoverage(offset, ownTravel));
        colorSum += scene.read(uint2(source)).rgb * weight;
        weightSum += weight;
    }
    return float4(colorSum / weightSum, center.a);
}
