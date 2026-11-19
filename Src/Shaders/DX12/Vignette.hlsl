// Screen-normalized radial exposure. The input is decoded by its sRGB view;
// attenuation changes linear light before the sRGB target encodes the result.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
};
Texture2D<float4> SceneColor : register(t0);
SamplerState LinearClamp : register(s0);
static const float FalloffStart = 0.2f;
static const float FalloffEnd = 0.8f;
struct FullscreenOutput { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
FullscreenOutput VSMain(uint id : SV_VertexID)
{
    FullscreenOutput result;
    float2 corner = float2((id << 1u) & 2u, id & 2u);
    result.UV = corner;
    result.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return result;
}
float4 PSMain(FullscreenOutput fragment) : SV_TARGET
{
    float4 result = SceneColor.SampleLevel(LinearClamp, fragment.UV, 0);
    float2 distanceFromCenter = fragment.UV * 2.0f - 1.0f;
    float cornerNormalizedRadius = dot(distanceFromCenter, distanceFromCenter) * 0.5f;
    float vignette = smoothstep(FalloffStart, FalloffEnd, cornerNormalizedRadius);
    result.rgb *= 1.0f - saturate(InverseExtentStrengths.z) * vignette;
    return result;
}
