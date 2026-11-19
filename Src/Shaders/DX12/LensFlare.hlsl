// Five native HDR passes implement a procedural screen-space lens flare.
// Bright visible pixels drive image-space ghost copies and a radial halo.
// John Chapman, "Screen Space Lens Flare" (2017), motivates this approximation:
// https://john-chapman.github.io/2017/11/05/pseudo-lens-flare.html
// It has no off-screen light query or calibrated lens geometry.
cbuffer HdrConstants : register(b0)
{
    float4 InverseSourceExtentBloom;
    float4 GradeExposure;
    uint4 PassToneMapping;
    float4 FlareThresholdIntensitySpacingRadius;
    float4 FlareAspect;
};
Texture2D<float4> Scene : register(t0);
Texture2D<float4> BlurredFeatures : register(t1);
SamplerState FilterSampler : register(s0);
static const uint BrightPass = 6u, GhostPass = 7u, HorizontalPass = 8u, VerticalPass = 9u;
static const float HalfFloatLimit = 65504.0f;
static const float NormalizationFloor = 0.000001f;
static const float3 Rec709 = float3(0.2126f, 0.7152f, 0.0722f);
static const int FeatureCount = 4;
// Signed scale places each ghost relative to the optical center; negative mirrors.
static const float FeatureScale[FeatureCount] = {-1.0f, -0.5f, 0.5f, 1.5f};
static const float3 FeatureTint[FeatureCount] = {
    float3(1.0f, 0.65f, 0.35f), float3(0.4f, 0.75f, 1.0f),
    float3(0.65f, 1.0f, 0.5f), float3(1.0f, 0.4f, 0.65f)};
static const float BorderFade = 0.05f, HaloWidth = 0.08f, HaloGain = 0.5f;
static const int ConvolutionRadius = 2;
static const float Binomial[3] = {0.375f, 0.25f, 0.0625f};
struct ScreenOutput { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
ScreenOutput VSMain(uint id : SV_VertexID)
{
    const float FarCorner = 3.0f;
    float2 corner = id == 0 ? float2(-1.0f, -1.0f) : id == 1 ? float2(FarCorner, -1.0f) : float2(-1.0f, FarCorner);
    ScreenOutput output;
    output.Position = float4(corner, 0.0f, 1.0f);
    output.UV = corner * float2(0.5f, -0.5f) + 0.5f;
    return output;
}
float4 ReadImage(Texture2D<float4> image, float2 coordinate)
{
    uint width, height;
    image.GetDimensions(width, height);
    float2 halfTexel = 0.5f / float2(width, height);
    return image.SampleLevel(FilterSampler, clamp(coordinate, halfTexel, 1.0f - halfTexel), 0);
}
float3 Highlight(float2 coordinate)
{
    float3 radiance = max(ReadImage(Scene, coordinate).rgb, 0.0f);
    float luminance = dot(radiance, Rec709);
    float excess = max(luminance - FlareThresholdIntensitySpacingRadius.x, 0.0f);
    return radiance * (excess / max(luminance, NormalizationFloor));
}
float3 ReadFeature(float2 coordinate)
{
    float2 nearestEdge = min(coordinate, 1.0f - coordinate);
    return ReadImage(Scene, coordinate).rgb * smoothstep(0.0f, BorderFade, min(nearestEdge.x, nearestEdge.y));
}
float4 PSMain(ScreenOutput fragment) : SV_TARGET
{
    float2 coordinate = fragment.UV;
    uint operation = PassToneMapping.x;
    if (operation == BrightPass)
    {
        float2 delta = InverseSourceExtentBloom.xy * 0.5f;
        float3 sum = Highlight(coordinate - delta) + Highlight(coordinate + delta) +
            Highlight(coordinate + delta * float2(-1.0f, 1.0f)) + Highlight(coordinate + delta * float2(1.0f, -1.0f));
        const float Quarter = 0.25f;
        return float4(min(sum * Quarter, HalfFloatLimit), 0.0f);
    }
    if (operation == GhostPass)
    {
        float2 displacement = coordinate - 0.5f;
        float3 generated = 0.0f;
        [unroll] for (int index = 0; index < FeatureCount; ++index)
            generated += ReadFeature(0.5f + displacement / (FeatureScale[index] * FlareThresholdIntensitySpacingRadius.z)) * FeatureTint[index] / FeatureCount;
        float2 radial = displacement * float2(FlareAspect.x, 1.0f);
        float distanceToCenter = length(radial);
        float radius = FlareThresholdIntensitySpacingRadius.w;
        if (radius > 0.0f)
        {
            float2 unitRadial = radial / max(distanceToCenter, NormalizationFloor);
            float2 donor = 0.5f + unitRadial * (distanceToCenter - radius) * float2(FlareAspect.y, 1.0f);
            float ring = 1.0f - smoothstep(0.0f, HaloWidth, abs(distanceToCenter - radius));
            generated += ReadFeature(donor) * (ring * HaloGain);
        }
        return float4(min(generated, HalfFloatLimit), 0.0f);
    }
    if (operation == HorizontalPass || operation == VerticalPass)
    {
        float2 stepUV = operation == HorizontalPass ? float2(InverseSourceExtentBloom.x, 0.0f) : float2(0.0f, InverseSourceExtentBloom.y);
        float3 filtered = ReadImage(Scene, coordinate).rgb * Binomial[0];
        [unroll] for (int offset = 1; offset <= ConvolutionRadius; ++offset)
            filtered += (ReadImage(Scene, coordinate + stepUV * offset).rgb + ReadImage(Scene, coordinate - stepUV * offset).rgb) * Binomial[offset];
        return float4(min(filtered, HalfFloatLimit), 0.0f);
    }
    float4 original = ReadImage(Scene, coordinate);
    float3 flare = ReadImage(BlurredFeatures, coordinate).rgb * FlareThresholdIntensitySpacingRadius.y;
    return float4(min(original.rgb + flare, HalfFloatLimit), original.a);
}
