// Asset-free color grading in linear HDR. Luminance-preserving saturation
// precedes contrast about 18% gray; negative output is clipped before tone mapping.
// RGBA16F has a finite representable range; saturation is not tone mapping.
static const float MaximumHalfFloatRadiance = 65504.0f;
cbuffer HdrConstants : register(b0)
{
    float4 InverseSourceExtentBloom;
    float4 GradeExposure;
    uint4 PassToneMapping;
};
Texture2D<float4> SourceColor : register(t0);
SamplerState ClampSampler : register(s0);
static const float3 LuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);
static const float ContrastPivot = 0.18f;
struct FullscreenOutput { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
FullscreenOutput VSMain(uint id : SV_VertexID)
{
    FullscreenOutput result;
    result.UV = float2((id << 1u) & 2u, id & 2u);
    result.Position = float4(result.UV * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return result;
}
float4 PSMain(FullscreenOutput fragment) : SV_TARGET
{
    float4 color = SourceColor.SampleLevel(ClampSampler, fragment.UV, 0);
    float gray = dot(color.rgb, LuminanceWeights);
    float3 saturation = gray + (color.rgb - gray) * GradeExposure.y;
    float3 contrast = (saturation - ContrastPivot) * GradeExposure.x + ContrastPivot;
    return float4(clamp(contrast, 0.0f, MaximumHalfFloatRadiance), color.a);
}
