// Animate two orthogonal screen waves; amplitudes are measured in pixels.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
    float4 NprEdge;
    float4 NprStyle;
    float4 NprHatch;
    float4 NprTone;
    float4 DynamicWarp;
};
Texture2D<float4> SceneColor : register(t0);
SamplerState LinearClamp : register(s0);
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint id : SV_VertexID)
{
    float2 corner = float2((id << 1u) & 2u, id & 2u);
    ScreenVertex output;
    output.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    const float Tau = 6.28318530718f;
    float2 uv = input.Position.xy * InverseExtentStrengths.xy;
    float t = DynamicWarp.w * DynamicWarp.z;
    float2 wave = float2(sin((uv.y * DynamicWarp.y + t) * Tau),
        sin((uv.x * DynamicWarp.y * 0.7f - t * 0.6f) * Tau));
    float2 donor = uv + wave * DynamicWarp.x * InverseExtentStrengths.xy * float2(1.0f, 0.5f);
    return SceneColor.SampleLevel(LinearClamp, donor, 0);
}
