// Radial screen warp: destination -> source = center + delta * (1 + k * r^2).
// The camera-model polynomial is used as an artistic pull-map, not as its inverse.
// Reference: OpenCV, Camera Calibration and 3D Reconstruction (radial distortion).
// https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
};
Texture2D<float4> SceneColor : register(t0);
SamplerState LinearClamp : register(s0);

struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    ScreenVertex output;
    float2 corner = float2((vertexId << 1u) & 2u, vertexId & 2u);
    output.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    const float2 ImageCenter = float2(0.5f, 0.5f);
    float2 uv = input.Position.xy * InverseExtentStrengths.xy;
    float2 centered = (uv - ImageCenter) * 2.0f;
    // Squared viewport-axis weights measure radius in half-diagonal units,
    // preserving circular symmetry in pixels on non-square render targets.
    float radiusSquared = dot(centered * centered, DistortionStrengthRadiusWeights.yz);
    float radialScale = 1.0f + DistortionStrengthRadiusWeights.x * radiusSquared;
    float2 sourceUV = ImageCenter + (uv - ImageCenter) * radialScale;
    // The sRGB SRV decodes before interpolation. RGBA travel together; depth
    // stays in geometry coordinates. The clamp sampler extends edge texels.
    return SceneColor.SampleLevel(LinearClamp, sourceUV, 0);
}
