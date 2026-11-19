#include <metal_stdlib>
using namespace metal;
// Independent MSL single-map depth generation. Williams, "Casting curved shadows
// on curved surfaces" (1978), doi:10.1145/800248.807402. Only geometry and material
// coverage affect depth; light response and filtering belong to receiver shading.
constant float EnabledThreshold = 0.5;
constant float CoverageNoiseScale = 52.9829189;
constant float2 CoverageNoiseDirection = float2(0.06711056, 0.00583715);
// Packed source vertex: position, normal, color, UV. Row arrays match CPU matrices.
struct ShadowVertexData { packed_float3 Position; packed_float3 Normal; packed_float4 Color; packed_float2 UV; };
struct ShadowDraw
{
    float4 World[4];
    float4 NormalWorld[4];
    float4 ViewProjection[4];
    float4 LightViewProjection[4];
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
struct ShadowVaryings
{
    float4 Position [[position]];
    float2 UV;
    float VertexAlpha;
};
float4 TransformShadow(constant float4* rows, float4 point)
{
    return float4(dot(rows[0], point), dot(rows[1], point), dot(rows[2], point), dot(rows[3], point));
}
vertex ShadowVaryings ShadowDepthVertex(device const ShadowVertexData* vertices [[buffer(0)]],
    constant ShadowDraw& draw [[buffer(1)]], uint index [[vertex_id]])
{
    const device ShadowVertexData& input = vertices[index];
    ShadowVaryings output;
    output.Position = TransformShadow(draw.ViewProjection, TransformShadow(draw.World, float4(float3(input.Position), 1.0)));
    output.UV = float2(input.UV.x, 1.0 - input.UV.y);
    output.VertexAlpha = input.Color.w;
    return output;
}
fragment void ShadowDepthFragment(ShadowVaryings input [[stage_in]], constant ShadowDraw& draw [[buffer(1)]],
    texture2d<float> diffuse [[texture(1)]], texture2d<float> opacity [[texture(3)]], sampler materialSampler [[sampler(0)]])
{
    float alpha = input.VertexAlpha * draw.MaterialParameters.y;
    if (draw.TextureFlags.x != 0u) alpha *= diffuse.sample(materialSampler, input.UV).a;
    if (draw.TextureFlags.z != 0u) alpha *= opacity.sample(materialSampler, input.UV).r;
    alpha = saturate(alpha);
    if (draw.MaterialParameters.z > 0.0)
    {
        if (alpha < draw.MaterialParameters.z) discard_fragment();
    }
    else if (draw.RenderParameters.w > EnabledThreshold)
    {
        float threshold = fract(CoverageNoiseScale * fract(dot(floor(input.Position.xy), CoverageNoiseDirection)));
        if (alpha <= 0.0 || alpha < threshold) discard_fragment();
    }
}
