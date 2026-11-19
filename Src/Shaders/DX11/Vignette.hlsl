// A smooth radial exposure falloff in linear light. Screen radius is normalized
// to the viewport corners, so resizing preserves the artistic falloff.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
};
Texture2D<float4> SceneColor : register(t0);
SamplerState ClampSampler : register(s0);
static const float InnerRadiusSquared = 0.2f;
static const float OuterRadiusSquared = 0.8f;
struct ScreenVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    ScreenVertex output;
    output.UV = float2((vertexId << 1u) & 2u, vertexId & 2u);
    output.Position = float4(output.UV.x * 2.0f - 1.0f, 1.0f - output.UV.y * 2.0f, 0.0f, 1.0f);
    return output;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    float4 linearColor = SceneColor.SampleLevel(ClampSampler, input.UV, 0);
    float2 centered = input.UV * 2.0f - 1.0f;
    float normalizedRadiusSquared = dot(centered, centered) * 0.5f;
    float falloff = smoothstep(InnerRadiusSquared, OuterRadiusSquared, normalizedRadiusSquared);
    linearColor.rgb *= 1.0f - saturate(InverseExtentStrengths.z) * falloff;
    return linearColor;
}
