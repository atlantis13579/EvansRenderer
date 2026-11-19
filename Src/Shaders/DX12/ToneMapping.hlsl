// Component-wise Reinhard or Narkowicz's 2016 filmic ACES approximation.
// This fit omits ACES color transforms and should not be called a full RRT/ODT.
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
cbuffer HdrConstants : register(b0)
{
    float4 InverseSourceExtentBloom;
    float4 GradeExposure;
    uint4 PassToneMapping;
};
Texture2D<float4> SourceColor : register(t0);
SamplerState ClampSampler : register(s0);
static const uint ReinhardMode = 1u;
static const float CurveA = 2.51f;
static const float CurveB = 0.03f;
static const float CurveC = 2.43f;
static const float CurveD = 0.59f;
static const float CurveE = 0.14f;
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
    float4 source = SourceColor.SampleLevel(ClampSampler, fragment.UV, 0);
    float3 radiance = max(source.rgb, 0.0f) * GradeExposure.z;
    float3 display;
    if (PassToneMapping.y == ReinhardMode) display = radiance / (radiance + 1.0f);
    else display = (radiance * (radiance * CurveA + CurveB)) / (radiance * (radiance * CurveC + CurveD) + CurveE);
    return float4(saturate(display), source.a);
}
