// Screen-space pseudo lens flare: threshold visible HDR radiance, synthesize
// ghosts/halo, blur, then add before bloom and tone mapping. No lens/dirt asset.
// Inspired by John Chapman, "Screen Space Lens Flare" (2017):
// https://john-chapman.github.io/2017/11/05/pseudo-lens-flare.html
// Ghost scales/tints are artistic lens coefficients, not an optical simulation.
cbuffer HdrConstants : register(b0)
{
    float4 InverseSourceExtentBloom;
    float4 GradeExposure;
    uint4 PassToneMapping;
    float4 FlareThresholdIntensitySpacingRadius;
    float4 FlareAspect;
};
Texture2D<float4> SourceColor : register(t0);
Texture2D<float4> FlareColor : register(t1);
SamplerState LinearSampler : register(s0);
static const uint ExtractPass = 6u, FeaturesPass = 7u, BlurHorizontalPass = 8u, BlurVerticalPass = 9u;
static const float MaximumHalfRadiance = 65504.0f;
static const float RadianceEpsilon = 0.000001f;
static const float3 LuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);
static const int GhostCount = 4;
static const float GhostScale[GhostCount] = {-1.0f, -0.5f, 0.5f, 1.5f};
static const float3 GhostTint[GhostCount] = {
    float3(1.0f, 0.65f, 0.35f), float3(0.4f, 0.75f, 1.0f),
    float3(0.65f, 1.0f, 0.5f), float3(1.0f, 0.4f, 0.65f)};
static const float EdgeFadeWidth = 0.05f;
static const float HaloThickness = 0.08f; // Viewport-height units.
static const float HaloWeight = 0.5f;
static const int BlurRadius = 2;
static const float BlurWeight[3] = {0.375f, 0.25f, 0.0625f}; // Normalized binomial [1,4,6,4,1]/16.
struct FullscreenVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
FullscreenVertex VSMain(uint id : SV_VertexID)
{
    FullscreenVertex output;
    output.UV = float2((id << 1u) & 2u, id & 2u);
    output.Position = float4(output.UV * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}
float4 SampleClamped(Texture2D<float4> image, float2 uv)
{
    uint width, height;
    image.GetDimensions(width, height);
    float2 border = 0.5f / float2(width, height);
    return image.SampleLevel(LinearSampler, clamp(uv, border, 1.0f - border), 0);
}
float3 ExtractBright(float2 uv)
{
    float3 color = max(SampleClamped(SourceColor, uv).rgb, 0.0f);
    float luminance = dot(color, LuminanceWeights);
    return color * max(luminance - FlareThresholdIntensitySpacingRadius.x, 0.0f) / max(luminance, RadianceEpsilon);
}
float3 FeatureSample(float2 uv)
{
    float2 edge = min(uv, 1.0f - uv);
    float fade = smoothstep(0.0f, EdgeFadeWidth, min(edge.x, edge.y));
    return SampleClamped(SourceColor, uv).rgb * fade;
}
float4 PSMain(FullscreenVertex input) : SV_TARGET
{
    float2 uv = input.UV;
    if (PassToneMapping.x == ExtractPass)
    {
        float3 sum = 0.0f;
        [unroll] for (int y = 0; y < 2; ++y)
            [unroll] for (int x = 0; x < 2; ++x)
                sum += ExtractBright(uv + (float2(x, y) - 0.5f) * InverseSourceExtentBloom.xy);
        const float BoxNormalization = 0.25f;
        return float4(min(sum * BoxNormalization, MaximumHalfRadiance), 0.0f);
    }
    if (PassToneMapping.x == FeaturesPass)
    {
        float2 offset = uv - 0.5f;
        float3 features = 0.0f;
        [unroll] for (int ghost = 0; ghost < GhostCount; ++ghost)
            features += FeatureSample(0.5f + offset / (GhostScale[ghost] * FlareThresholdIntensitySpacingRadius.z)) * GhostTint[ghost] / GhostCount;
        float2 metric = offset * float2(FlareAspect.x, 1.0f);
        float radius = length(metric);
        float haloRadius = FlareThresholdIntensitySpacingRadius.w;
        if (haloRadius > 0.0f)
        {
            float2 direction = metric / max(radius, RadianceEpsilon);
            float2 donor = 0.5f + direction * (radius - haloRadius) * float2(FlareAspect.y, 1.0f);
            float window = 1.0f - smoothstep(0.0f, HaloThickness, abs(radius - haloRadius));
            features += FeatureSample(donor) * (window * HaloWeight);
        }
        return float4(min(features, MaximumHalfRadiance), 0.0f);
    }
    if (PassToneMapping.x == BlurHorizontalPass || PassToneMapping.x == BlurVerticalPass)
    {
        float2 axis = PassToneMapping.x == BlurHorizontalPass ? float2(InverseSourceExtentBloom.x, 0.0f) : float2(0.0f, InverseSourceExtentBloom.y);
        float3 sum = SampleClamped(SourceColor, uv).rgb * BlurWeight[0];
        [unroll] for (int tap = 1; tap <= BlurRadius; ++tap)
            sum += (SampleClamped(SourceColor, uv + tap * axis).rgb + SampleClamped(SourceColor, uv - tap * axis).rgb) * BlurWeight[tap];
        return float4(min(sum, MaximumHalfRadiance), 0.0f);
    }
    float4 scene = SampleClamped(SourceColor, uv);
    return float4(min(scene.rgb + SampleClamped(FlareColor, uv).rgb * FlareThresholdIntensitySpacingRadius.y, MaximumHalfRadiance), scene.a);
}
