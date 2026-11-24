// Sample the center of each screen-aligned pixel block in linear light.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
    float4 NprEdge;
    float4 NprStyle;
    float4 NprHatch;
    float4 NprTone;
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
    float2 extent = 1.0f / InverseExtentStrengths.xy;
    float block = max(1.0f, floor(NprTone.y + 0.5f));
    float2 pixel = floor(input.Position.xy / block) * block + block * 0.5f;
    return SceneColor.SampleLevel(LinearClamp, clamp(pixel, 0.5f, extent - 0.5f) * InverseExtentStrengths.xy, 0);
}
