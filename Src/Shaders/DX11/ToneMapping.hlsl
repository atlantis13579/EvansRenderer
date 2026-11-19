// Exposure precedes the display transform. Reinhard uses c/(1+c).
// The alternate curve is Narkowicz's ACES-inspired fit, not full ACES RRT/ODT:
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// Return linear display RGB; the sRGB attachment performs transfer encoding.
cbuffer HdrConstants : register(b0)
{
    float4 InverseSourceExtentBloom;
    float4 GradeExposure;
    uint4 PassToneMapping;
};
Texture2D<float4> SourceColor : register(t0);
SamplerState LinearClamp : register(s0);
static const uint ReinhardOperator = 1u;
static const float FitShoulderNumerator = 2.51f;
static const float FitToeNumerator = 0.03f;
static const float FitShoulderDenominator = 2.43f;
static const float FitLinearDenominator = 0.59f;
static const float FitToeDenominator = 0.14f;
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
    float3 exposed = max(source.rgb, 0.0f) * GradeExposure.z;
    float3 display = PassToneMapping.y == ReinhardOperator
        ? exposed / (1.0f + exposed)
        : (exposed * (FitShoulderNumerator * exposed + FitToeNumerator))
            / (exposed * (FitShoulderDenominator * exposed + FitLinearDenominator) + FitToeDenominator);
    return float4(saturate(display), source.a);
}
