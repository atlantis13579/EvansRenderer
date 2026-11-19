// Directional FXAA in perceptual RGB. Timothy Lottes, "FXAA" (2011):
// https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf
// Native sRGB texture filtering supplies linear samples; encode each sample
// for contrast detection, then decode the filtered result for the sRGB target.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
};
Texture2D<float4> SceneColor : register(t0);
SamplerState ClampSampler : register(s0);
static const float3 PerceptualLuma = float3(0.299f, 0.587f, 0.114f);
static const float MinimumContrast = 1.0f / 16.0f;
static const float RelativeContrast = 1.0f / 8.0f;
static const float ReductionMinimum = 1.0f / 128.0f;
static const float ReductionMultiplier = 1.0f / 8.0f;
static const float MaximumSpan = 8.0f;
static const float QuarterWeight = 0.25f;
static const float InnerOffset = 1.0f / 3.0f - 0.5f;
static const float SrgbLinearThreshold = 0.0031308f;
static const float SrgbEncodedThreshold = 0.04045f;
static const float SrgbSlope = 12.92f;
static const float SrgbScale = 1.055f;
static const float SrgbOffset = 0.055f;
static const float SrgbPower = 2.4f;
struct ScreenVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    ScreenVertex output;
    output.UV = float2((vertexId << 1u) & 2u, vertexId & 2u);
    output.Position = float4(output.UV.x * 2.0f - 1.0f, 1.0f - output.UV.y * 2.0f, 0.0f, 1.0f);
    return output;
}
float3 EncodeDisplay(float3 linearColor)
{
    float3 power = SrgbScale * pow(max(linearColor, 0.0f), 1.0f / SrgbPower) - SrgbOffset;
    return float3(linearColor.r <= SrgbLinearThreshold ? linearColor.r * SrgbSlope : power.r,
        linearColor.g <= SrgbLinearThreshold ? linearColor.g * SrgbSlope : power.g,
        linearColor.b <= SrgbLinearThreshold ? linearColor.b * SrgbSlope : power.b);
}
float3 DecodeDisplay(float3 displayColor)
{
    float3 power = pow((max(displayColor, 0.0f) + SrgbOffset) / SrgbScale, SrgbPower);
    return float3(displayColor.r <= SrgbEncodedThreshold ? displayColor.r / SrgbSlope : power.r,
        displayColor.g <= SrgbEncodedThreshold ? displayColor.g / SrgbSlope : power.g,
        displayColor.b <= SrgbEncodedThreshold ? displayColor.b / SrgbSlope : power.b);
}
float3 SampleDisplay(float2 uv) { return EncodeDisplay(SceneColor.SampleLevel(ClampSampler, uv, 0).rgb); }
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    float2 texel = InverseExtentStrengths.xy;
    float4 center = SceneColor.SampleLevel(ClampSampler, input.UV, 0);
    float centerLuma = dot(EncodeDisplay(center.rgb), PerceptualLuma);
    float northWest = dot(SampleDisplay(input.UV + float2(-texel.x, -texel.y)), PerceptualLuma);
    float northEast = dot(SampleDisplay(input.UV + float2(texel.x, -texel.y)), PerceptualLuma);
    float southWest = dot(SampleDisplay(input.UV + float2(-texel.x, texel.y)), PerceptualLuma);
    float southEast = dot(SampleDisplay(input.UV + float2(texel.x, texel.y)), PerceptualLuma);
    float minimumLuma = min(centerLuma, min(min(northWest, northEast), min(southWest, southEast)));
    float maximumLuma = max(centerLuma, max(max(northWest, northEast), max(southWest, southEast)));
    if (maximumLuma - minimumLuma < max(MinimumContrast, maximumLuma * RelativeContrast)) return center;
    float2 direction = float2(-(northWest + northEast - southWest - southEast),
        northWest + southWest - northEast - southEast);
    float reduction = max((northWest + northEast + southWest + southEast) * QuarterWeight * ReductionMultiplier, ReductionMinimum);
    direction = clamp(direction / (min(abs(direction.x), abs(direction.y)) + reduction), -MaximumSpan, MaximumSpan) * texel;
    float3 narrow = 0.5f * (SampleDisplay(input.UV + direction * InnerOffset) + SampleDisplay(input.UV - direction * InnerOffset));
    float3 wide = narrow * 0.5f + QuarterWeight * (SampleDisplay(input.UV - direction * 0.5f) + SampleDisplay(input.UV + direction * 0.5f));
    float wideLuma = dot(wide, PerceptualLuma);
    return float4(DecodeDisplay(wideLuma < minimumLuma || wideLuma > maximumLuma ? narrow : wide), center.a);
}
