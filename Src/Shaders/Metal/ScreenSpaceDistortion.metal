#include <metal_stdlib>
using namespace metal;
// Artistic radial screen warp using the first camera-distortion polynomial
// term. This directly defines a pull-map; it is not calibrated lens correction.
// https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
struct PostConstants
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
};
struct ScreenVertex { float4 Position [[position]]; };
vertex ScreenVertex PostVertexMain(uint vertexId [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 corner = vertexId == 0 ? float2(-1.0, -1.0) :
        vertexId == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(corner, 0.0, 1.0)};
}
fragment float4 PostFragmentMain(ScreenVertex stageInput [[stage_in]],
    constant PostConstants& parameters [[buffer(0)]], texture2d<float> scene [[texture(0)]])
{
    constexpr sampler LinearEdge(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 LensCenter = float2(0.5);
    float2 uv = stageInput.Position.xy * parameters.InverseExtentStrengths.xy;
    float2 radial = (uv - LensCenter) * 2.0;
    float radiusSquared = dot(radial * radial, parameters.DistortionStrengthRadiusWeights.yz);
    float radialScale = 1.0 + parameters.DistortionStrengthRadiusWeights.x * radiusSquared;
    float2 sourceUV = LensCenter + (uv - LensCenter) * radialScale;
    // RGBA8Unorm_sRGB decodes before interpolation and re-encodes on output.
    // Resample alpha at the same donor coordinate; geometry depth is untouched.
    return scene.sample(LinearEdge, sourceUV, level(0.0));
}
