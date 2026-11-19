#include <metal_stdlib>
using namespace metal;
// Screen-space lens ghosts and halo from visible HDR highlights; five passes
// share this independent MSL source. Occlusion is already present in scene color.
// John Chapman, Screen Space Lens Flare (2017):
// https://john-chapman.github.io/2017/11/05/pseudo-lens-flare.html
struct HdrUniforms
{
    float4 InverseSourceExtentBloom;
    float4 GradeExposure;
    uint4 PassToneMapping;
    float4 FlareThresholdIntensitySpacingRadius;
    float4 FlareAspect;
};
struct HdrVertex { float4 Position [[position]]; float2 UV; };
vertex HdrVertex HdrVertexMain(uint vertexId [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 corner = vertexId == 0 ? float2(-1.0) : vertexId == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(corner, 0.0, 1.0), corner * float2(0.5, -0.5) + 0.5};
}
float4 ReadClamped(texture2d<float> image, float2 coordinate)
{
    constexpr sampler LinearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 margin = 0.5 / float2(image.get_width(), image.get_height());
    return image.sample(LinearClamp, clamp(coordinate, margin, 1.0 - margin), level(0.0));
}
float3 BrightRadiance(texture2d<float> scene, float2 uv, float threshold)
{
    const float3 Luminance = float3(0.2126, 0.7152, 0.0722);
    const float MinimumDivisor = 0.000001;
    float3 radiance = max(ReadClamped(scene, uv).rgb, 0.0);
    float light = dot(radiance, Luminance);
    return radiance * max(light - threshold, 0.0) / max(light, MinimumDivisor);
}
float3 FeatureSample(texture2d<float> scene, float2 uv)
{
    const float EdgeFade = 0.05;
    float2 edge = min(uv, 1.0 - uv);
    return ReadClamped(scene, uv).rgb * smoothstep(0.0, EdgeFade, min(edge.x, edge.y));
}
fragment float4 HdrFragmentMain(HdrVertex input [[stage_in]],
    constant HdrUniforms& uniforms [[buffer(0)]], texture2d<float> scene [[texture(0)]],
    texture2d<float> flare [[texture(1)]])
{
    const uint Extract = 6, Generate = 7, BlurX = 8, BlurY = 9;
    const float MaximumHalfRadiance = 65504.0;
    const float DirectionEpsilon = 0.000001;
    const int GhostCount = 4;
    // Artistic signed magnifications and spectral tints; no lens texture lookup.
    const float GhostScale[GhostCount] = {-1.0, -0.5, 0.5, 1.5};
    const float3 GhostTint[GhostCount] = {float3(1.0, 0.65, 0.35), float3(0.4, 0.75, 1.0), float3(0.65, 1.0, 0.5), float3(1.0, 0.4, 0.65)};
    const float HaloThickness = 0.08, HaloIntensity = 0.5;
    const int FilterRadius = 2;
    const float FilterWeights[3] = {0.375, 0.25, 0.0625}; // Normalized binomial convolution.
    uint operation = uniforms.PassToneMapping.x;
    if (operation == Extract)
    {
        float3 accumulated = 0.0;
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
                accumulated += BrightRadiance(scene, input.UV + (float2(x, y) - 0.5) * uniforms.InverseSourceExtentBloom.xy,
                    uniforms.FlareThresholdIntensitySpacingRadius.x);
        const float BoxWeight = 0.25;
        return float4(min(accumulated * BoxWeight, MaximumHalfRadiance), 0.0);
    }
    if (operation == Generate)
    {
        float2 centered = input.UV - 0.5;
        float3 features = 0.0;
        for (int ghost = 0; ghost < GhostCount; ++ghost)
            features += FeatureSample(scene, 0.5 + centered / (GhostScale[ghost] * uniforms.FlareThresholdIntensitySpacingRadius.z)) * GhostTint[ghost] / GhostCount;
        float2 radial = centered * float2(uniforms.FlareAspect.x, 1.0);
        float distanceToCenter = length(radial);
        float radius = uniforms.FlareThresholdIntensitySpacingRadius.w;
        if (radius > 0.0)
        {
            float2 direction = radial / max(distanceToCenter, DirectionEpsilon);
            float2 donor = 0.5 + direction * (distanceToCenter - radius) * float2(uniforms.FlareAspect.y, 1.0);
            float ring = 1.0 - smoothstep(0.0, HaloThickness, abs(distanceToCenter - radius));
            features += FeatureSample(scene, donor) * ring * HaloIntensity;
        }
        return float4(min(features, MaximumHalfRadiance), 0.0);
    }
    if (operation == BlurX || operation == BlurY)
    {
        float2 direction = operation == BlurX ? float2(uniforms.InverseSourceExtentBloom.x, 0.0) : float2(0.0, uniforms.InverseSourceExtentBloom.y);
        float3 filtered = ReadClamped(scene, input.UV).rgb * FilterWeights[0];
        for (int tap = 1; tap <= FilterRadius; ++tap)
            filtered += (ReadClamped(scene, input.UV + direction * tap).rgb + ReadClamped(scene, input.UV - direction * tap).rgb) * FilterWeights[tap];
        return float4(min(filtered, MaximumHalfRadiance), 0.0);
    }
    float4 original = ReadClamped(scene, input.UV);
    return float4(min(original.rgb + ReadClamped(flare, input.UV).rgb * uniforms.FlareThresholdIntensitySpacingRadius.y, MaximumHalfRadiance), original.a);
}
