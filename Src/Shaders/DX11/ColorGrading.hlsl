// Analytic grading needs no LUT asset. Saturation changes the distance from
// Rec.709 luminance; contrast scales linear RGB about the middle-gray pivot.
// RGBA16F has a finite representable range; saturation is not tone mapping.
static const float MaximumHalfFloatRadiance = 65504.0f;
cbuffer HdrConstants : register(b0)
{
    float4 InverseSourceExtentBloom;
    float4 GradeExposure;
    uint4 PassToneMapping;
};
Texture2D<float4> SourceColor : register(t0);
SamplerState LinearClamp : register(s0);
static const float3 Rec709Luminance = float3(0.2126f, 0.7152f, 0.0722f);
static const float MiddleGray = 0.18f;
struct ScreenVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
ScreenVertex VSMain(uint id : SV_VertexID)
{
    ScreenVertex output;
    output.UV = float2((id << 1u) & 2u, id & 2u);
    output.Position = float4(output.UV.x * 2.0f - 1.0f, 1.0f - output.UV.y * 2.0f, 0.0f, 1.0f);
    return output;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    float4 source = SourceColor.SampleLevel(LinearClamp, input.UV, 0);
    float luminance = dot(source.rgb, Rec709Luminance);
    float3 saturated = lerp(luminance.xxx, source.rgb, GradeExposure.y);
    return float4(clamp((saturated - MiddleGray) * GradeExposure.x + MiddleGray, 0.0f, MaximumHalfFloatRadiance), source.a);
}
