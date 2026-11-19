// Analytic radial resampling of the tone-resolved scene. We evaluate a backward
// map at destination pixel centers, so raster holes and iterative inversion
// are unnecessary. A zero coefficient is bypassed by the CPU pass planner.
// Polynomial reference: OpenCV camera calibration, radial distortion model.
// https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
};
Texture2D<float4> ResolvedScene : register(t0);
SamplerState ClampColorSampler : register(s0);

struct FullscreenVertex { float4 Position : SV_POSITION; };
FullscreenVertex VSMain(uint id : SV_VertexID)
{
    const float OversizedCorner = 3.0f;
    float2 position = id == 0u ? float2(-1.0f, -1.0f) :
        id == 1u ? float2(OversizedCorner, -1.0f) : float2(-1.0f, OversizedCorner);
    FullscreenVertex result;
    result.Position = float4(position, 0.0f, 1.0f);
    return result;
}
float4 PSMain(FullscreenVertex fragment) : SV_TARGET
{
    const float ScreenCenter = 0.5f;
    float2 coordinate = fragment.Position.xy * InverseExtentStrengths.xy;
    float2 displacement = coordinate - ScreenCenter;
    float2 diameterCoordinate = displacement * 2.0f;
    float radiusSquared = dot(diameterCoordinate * diameterCoordinate,
        DistortionStrengthRadiusWeights.yz);
    float magnification = 1.0f + radiusSquared * DistortionStrengthRadiusWeights.x;
    float2 donor = ScreenCenter + displacement * magnification;
    // The sRGB view supplies linear light for bilinear interpolation. Alpha
    // uses the same footprint; out-of-bounds coordinates extend the edge.
    return ResolvedScene.SampleLevel(ClampColorSampler, donor, 0);
}
