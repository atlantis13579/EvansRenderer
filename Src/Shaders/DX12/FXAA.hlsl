// Independent DirectX 12 directional FXAA implementation. Lottes, "FXAA" (2011):
// https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf
// Contrast and reconstruction use perceptual RGB; native texture filtering and
// render-target conversion remain linear/sRGB respectively.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
};
Texture2D<float4> SceneColor : register(t0);
SamplerState LinearClamp : register(s0);
static const float3 LumaWeights = float3(0.299f, 0.587f, 0.114f);
static const float AbsoluteThreshold = 1.0f / 16.0f;
static const float RelativeThreshold = 1.0f / 8.0f;
static const float MinimumReduction = 1.0f / 128.0f;
static const float ReductionScale = 1.0f / 8.0f;
static const float EdgeSpanLimit = 8.0f;
static const float FourSampleWeight = 0.25f;
static const float InnerSamplePosition = 1.0f / 3.0f - 0.5f;
static const float LinearBreakpoint = 0.0031308f;
static const float EncodedBreakpoint = 0.04045f;
static const float TransferSlope = 12.92f;
static const float TransferScale = 1.055f;
static const float TransferOffset = 0.055f;
static const float TransferExponent = 2.4f;
struct FullscreenOutput { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
FullscreenOutput VSMain(uint id : SV_VertexID)
{
    FullscreenOutput result;
    float2 corner = float2((id << 1u) & 2u, id & 2u);
    result.UV = corner;
    result.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return result;
}
float3 ToPerceptual(float3 value)
{
    float3 powerPart = TransferScale * pow(max(value, 0.0f), 1.0f / TransferExponent) - TransferOffset;
    return float3(value.r <= LinearBreakpoint ? value.r * TransferSlope : powerPart.r,
        value.g <= LinearBreakpoint ? value.g * TransferSlope : powerPart.g,
        value.b <= LinearBreakpoint ? value.b * TransferSlope : powerPart.b);
}
float3 ToLinear(float3 value)
{
    float3 powerPart = pow((max(value, 0.0f) + TransferOffset) / TransferScale, TransferExponent);
    return float3(value.r <= EncodedBreakpoint ? value.r / TransferSlope : powerPart.r,
        value.g <= EncodedBreakpoint ? value.g / TransferSlope : powerPart.g,
        value.b <= EncodedBreakpoint ? value.b / TransferSlope : powerPart.b);
}
float3 ReadPerceptual(float2 uv) { return ToPerceptual(SceneColor.SampleLevel(LinearClamp, uv, 0).rgb); }
float4 PSMain(FullscreenOutput fragment) : SV_TARGET
{
    float2 pixelSize = InverseExtentStrengths.xy;
    float2 uv = fragment.UV;
    float4 center = SceneColor.SampleLevel(LinearClamp, uv, 0);
    float lumaCenter = dot(ToPerceptual(center.rgb), LumaWeights);
    float lumaNW = dot(ReadPerceptual(uv - pixelSize), LumaWeights);
    float lumaNE = dot(ReadPerceptual(uv + pixelSize * float2(1.0f, -1.0f)), LumaWeights);
    float lumaSW = dot(ReadPerceptual(uv + pixelSize * float2(-1.0f, 1.0f)), LumaWeights);
    float lumaSE = dot(ReadPerceptual(uv + pixelSize), LumaWeights);
    float low = min(lumaCenter, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float high = max(lumaCenter, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    if (high - low < max(AbsoluteThreshold, high * RelativeThreshold)) return center;
    float2 edge = float2(-(lumaNW + lumaNE - lumaSW - lumaSE), lumaNW + lumaSW - lumaNE - lumaSE);
    float reduction = max((lumaNW + lumaNE + lumaSW + lumaSE) * FourSampleWeight * ReductionScale, MinimumReduction);
    float inverseLength = rcp(min(abs(edge.x), abs(edge.y)) + reduction);
    float2 stepUV = clamp(edge * inverseLength, -EdgeSpanLimit, EdgeSpanLimit) * pixelSize;
    float3 narrow = (ReadPerceptual(uv + stepUV * InnerSamplePosition) + ReadPerceptual(uv - stepUV * InnerSamplePosition)) * 0.5f;
    float3 broad = narrow * 0.5f + (ReadPerceptual(uv - stepUV * 0.5f) + ReadPerceptual(uv + stepUV * 0.5f)) * FourSampleWeight;
    float reconstructedLuma = dot(broad, LumaWeights);
    float3 reconstructed = reconstructedLuma < low || reconstructedLuma > high ? narrow : broad;
    return float4(ToLinear(reconstructed), center.a);
}
