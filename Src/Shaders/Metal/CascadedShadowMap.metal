#include <metal_stdlib>
using namespace metal;

// Independent Metal depth producer for stable parallel-split shadow maps.
// The CPU supplies a snapped light projection per interval. Material coverage
// is rasterized here; receiver shading selects and blends the resulting maps.
// Zhang, Sun and Nyman, "Parallel-Split Shadow Maps on Programmable GPUs" (2007).
// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus
constant uint MaximumLights = 4;
constant float BlendingThreshold = 0.5;
constant float CoverageNoiseScale = 52.9829189;
constant float2 CoverageNoiseFrequency = float2(0.06711056, 0.00583715);

struct MeshVertex { float Values[12]; };
struct SceneLight
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};
// Explicit rows preserve the CPU draw layout without MSL matrix transposition.
struct DepthConstants
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
    SceneLight Lights[MaximumLights];
    float4 ShadingParameters;
};
struct DepthVaryings
{
    float4 Position [[position]];
    float2 MaterialUV;
    float Opacity;
};
float4 TransformDepthRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value), dot(rows[2], value), dot(rows[3], value));
}
vertex DepthVaryings CascadeDepthVertex(device const MeshVertex* vertices [[buffer(0)]],
    constant DepthConstants& draw [[buffer(1)]], uint vertexId [[vertex_id]])
{
    const device MeshVertex& inputVertex = vertices[vertexId];
    float4 world = TransformDepthRows(draw.World, float4(inputVertex.Values[0], inputVertex.Values[1], inputVertex.Values[2], 1.0));
    DepthVaryings result;
    // Metal accepts the fitted [0,1] depth directly; its viewport establishes
    // top-left pixel coordinates for both the depth image and coverage noise.
    result.Position = TransformDepthRows(draw.ViewProjection, world);
    result.MaterialUV = float2(inputVertex.Values[10], 1.0 - inputVertex.Values[11]);
    result.Opacity = inputVertex.Values[9];
    return result;
}
fragment void CascadeDepthFragment(DepthVaryings input [[stage_in]],
    constant DepthConstants& draw [[buffer(1)]],
    texture2d<float> diffuseMap [[texture(1)]], texture2d<float> opacityMap [[texture(3)]],
    sampler materialSampler [[sampler(0)]])
{
    float diffuseAlpha = draw.TextureFlags.x != 0 ? diffuseMap.sample(materialSampler, input.MaterialUV).a : 1.0;
    float textureOpacity = draw.TextureFlags.z != 0 ? opacityMap.sample(materialSampler, input.MaterialUV).r : 1.0;
    float coverage = saturate(input.Opacity * draw.MaterialParameters.y * diffuseAlpha * textureOpacity);
    if (draw.MaterialParameters.z > 0.0)
    {
        if (coverage < draw.MaterialParameters.z) discard_fragment();
    }
    else if (draw.RenderParameters.w > BlendingThreshold)
    {
        float threshold = fract(CoverageNoiseScale * fract(dot(floor(input.Position.xy), CoverageNoiseFrequency)));
        if (coverage <= 0.0 || coverage < threshold) discard_fragment();
    }
}
