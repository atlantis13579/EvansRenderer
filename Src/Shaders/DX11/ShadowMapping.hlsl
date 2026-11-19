// Independent DX11 single-map depth producer.
// Williams, "Casting curved shadows on curved surfaces" (SIGGRAPH 1978):
// https://doi.org/10.1145/800248.807402
// Rasterize nearest light-space depth; receiver lighting separately compares it.
// Alpha cutouts use material coverage, not a lighting model. Fractional opacity
// uses stable interleaved gradient coverage, an approximation to transmission.
static const float EnabledThreshold = 0.5f;
static const float CoverageNoiseScale = 52.9829189f;
static const float2 CoverageNoiseDirection = float2(0.06711056f, 0.00583715f);

// Prefix of the native DrawConstants ABI; unused fields preserve offsets.
cbuffer ShadowDraw : register(b0)
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
};
Texture2D<float4> DiffuseMap : register(t1);
Texture2D<float4> OpacityMap : register(t3);
SamplerState SurfaceSampler : register(s0);
struct ShadowVertex
{
    float3 Position : POSITION;
    float3 Normal : NORMAL; // Retain the shared DX11 input signature; depth never reads it.
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};
struct ShadowVaryings
{
    float4 Position : SV_POSITION;
    float2 UV : TEXCOORD0;
    float Opacity : COLOR0;
};
ShadowVaryings VSMain(ShadowVertex vertex)
{
    ShadowVaryings output;
    output.Position = mul(ViewProjection, mul(World, float4(vertex.Position, 1.0f)));
    output.UV = float2(vertex.TexCoord.x, 1.0f - vertex.TexCoord.y);
    output.Opacity = vertex.Color.a;
    return output;
}
void PSMain(ShadowVaryings input)
{
    float diffuseOpacity = TextureFlags.x != 0u ? DiffuseMap.Sample(SurfaceSampler, input.UV).a : 1.0f;
    float mappedOpacity = TextureFlags.z != 0u ? OpacityMap.Sample(SurfaceSampler, input.UV).r : 1.0f;
    float coverage = saturate(input.Opacity * MaterialParameters.y * diffuseOpacity * mappedOpacity);
    float cutoff = MaterialParameters.z;
    if (cutoff > 0.0f) clip(coverage - cutoff);
    else if (RenderParameters.w > EnabledThreshold)
    {
        if (coverage <= 0.0f) discard;
        float threshold = frac(CoverageNoiseScale * frac(dot(floor(input.Position.xy), CoverageNoiseDirection)));
        clip(coverage - threshold);
    }
}
