// DX12 owns this light-depth program independently of DX11 and surface shading.
// Shadow mapping stores the nearest light-space surface in D32, then compares
// receiver depth during lighting (Williams, SIGGRAPH 1978,
// "Casting curved shadows on curved surfaces", doi:10.1145/800248.807402).
// The root signature supplies draw constants and material textures. Depth bias
// is raster state; no normal map, BRDF, HDR target or screen history is required.
static const float FlagThreshold = 0.5f;
static const float InterleavedScale = 52.9829189f;
static const float2 InterleavedDirection = float2(0.06711056f, 0.00583715f);
cbuffer DepthDraw : register(b0)
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
Texture2D<float4> DiffuseImage : register(t1);
Texture2D<float4> OpacityImage : register(t3);
SamplerState MaterialSampler : register(s0);
struct DepthInput
{
    float3 Position : POSITION;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};
struct DepthOutput
{
    float4 ClipPosition : SV_POSITION;
    float2 MaterialUV : TEXCOORD0;
    float VertexAlpha : COLOR0;
};
DepthOutput VSMain(DepthInput input)
{
    float4 worldPosition = mul(World, float4(input.Position, 1.0f));
    DepthOutput output;
    output.ClipPosition = mul(ViewProjection, worldPosition);
    output.MaterialUV = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    output.VertexAlpha = input.Color.a;
    return output;
}
void PSMain(DepthOutput input)
{
    float opacity = input.VertexAlpha * MaterialParameters.y;
    if (TextureFlags.x != 0u) opacity *= DiffuseImage.Sample(MaterialSampler, input.MaterialUV).a;
    if (TextureFlags.z != 0u) opacity *= OpacityImage.Sample(MaterialSampler, input.MaterialUV).r;
    opacity = saturate(opacity);
    if (MaterialParameters.z > 0.0f)
        clip(opacity - MaterialParameters.z);
    else if (RenderParameters.w > FlagThreshold)
    {
        // Deterministic per-texel coverage approximates scalar transmission.
        // Zero opacity must reject even at a texel whose noise threshold is zero.
        float noise = frac(InterleavedScale * frac(dot(floor(input.ClipPosition.xy), InterleavedDirection)));
        if (opacity <= 0.0f || opacity < noise) discard;
    }
}
