// DX11 cascaded shadow depth producer. The CPU submits one light projection per
// cascade with stable sphere fitting and a filter guard band. Alpha coverage is
// evaluated independently for every map; empty maps retain the clear depth.
// Zhang, Sun and Nyman, "Parallel-Split Shadow Maps on Programmable GPUs" (2007).
// Native lighting files own their sampling code, with no shader-to-shader include.
static const uint MaximumLights = 4u;
static const float CoverageNoiseScale = 52.9829189f;
static const float2 CoverageNoiseDirection = float2(0.06711056f, 0.00583715f);
struct LightConstants
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};
cbuffer DrawConstants : register(b0)
{
    row_major float4x4 World;
    row_major float4x4 NormalWorld;
    row_major float4x4 ViewProjection;
    row_major float4x4 LightViewProjection;
    float4 PipelineColor;
    float4 MaterialDiffuseColor;
    float4 MaterialAmbientColor;
    float4 MaterialSpecularColor;
    float4 MaterialEmissiveColor;
    float4 MaterialParameters;
    float4 AmbientColorStrength;
    float4 CameraLightCount;
    float4 RenderParameters;
    uint4 TextureFlags;
    LightConstants Lights[MaximumLights];
    float4 ShadingParameters;
};
Texture2D<float4> DiffuseMap : register(t1);
Texture2D<float4> OpacityMap : register(t3);
SamplerState SurfaceSampler : register(s0);
struct VertexInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};
struct DepthVertex
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    float Alpha : COLOR0;
};
DepthVertex VSMain(VertexInput input)
{
    DepthVertex output;
    output.Position = mul(ViewProjection, mul(World, float4(input.Position, 1.0f)));
    output.TexCoord = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    output.Alpha = input.Color.a;
    return output;
}
void PSMain(DepthVertex input)
{
    float diffuseAlpha = TextureFlags.x != 0u ? DiffuseMap.Sample(SurfaceSampler, input.TexCoord).a : 1.0f;
    float opacity = TextureFlags.z != 0u ? OpacityMap.Sample(SurfaceSampler, input.TexCoord).r : 1.0f;
    float alpha = saturate(input.Alpha * MaterialParameters.y * diffuseAlpha * opacity);
    if (MaterialParameters.z > 0.0f) clip(alpha - MaterialParameters.z);
    else if (RenderParameters.w > 0.5f)
    {
        // Stable interleaved gradient noise approximates partial transmission;
        // true alpha testing always uses the material cutoff instead.
        if (alpha <= 0.0f) discard;
        float threshold = frac(CoverageNoiseScale * frac(dot(floor(input.Position.xy), CoverageNoiseDirection)));
        clip(alpha - threshold);
    }
}
