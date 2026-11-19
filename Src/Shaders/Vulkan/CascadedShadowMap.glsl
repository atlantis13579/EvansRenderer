#version 450
// Independent Vulkan depth producer for practical parallel-split shadow maps.
// Each draw receives one CPU-fitted light projection. Receivers select intervals
// and blend their visibility in the separate shading/deferred technique files.
// Zhang, Sun and Nyman, "Parallel-Split Shadow Maps on Programmable GPUs" (2007).
// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus
const int MaximumLights = 4;
struct SceneLight
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
// Match the native draw layout so material coverage is identical to color draws.
layout(set = 0, binding = 0, std140, row_major) uniform DepthDraw
{
    mat4 World;
    mat4 NormalWorld;
    mat4 ViewProjection;
    mat4 LightViewProjection;
    vec4 PipelineColor;
    vec4 MaterialDiffuseColor;
    vec4 MaterialAmbientColor;
    vec4 MaterialSpecularColor;
    vec4 MaterialEmissiveColor;
    vec4 MaterialParameters;
    vec4 AmbientColorStrength;
    vec4 CameraLightCount;
    vec4 RenderParameters;
    uvec4 TextureFlags;
    SceneLight Lights[MaximumLights];
    vec4 ShadingParameters;
} Draw;

#ifdef EVANS_VERTEX_STAGE
layout(location = 0) in vec3 InPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
layout(location = 0) out vec2 MaterialUV;
layout(location = 1) out float VertexOpacity;
void main()
{
    gl_Position = Draw.ViewProjection * Draw.World * vec4(InPosition, 1.0);
    // CPU light matrices already use Vulkan's [0,1] depth. Positive viewport
    // height requires a Y flip to keep the top-left image convention.
    gl_Position.y = -gl_Position.y;
    MaterialUV = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    VertexOpacity = InColor.a;
}
#else
layout(location = 0) in vec2 MaterialUV;
layout(location = 1) in float VertexOpacity;
layout(set = 0, binding = 2) uniform texture2D DiffuseImage;
layout(set = 0, binding = 4) uniform texture2D OpacityImage;
layout(set = 0, binding = 5) uniform sampler MaterialSampler;
const float BlendingEnabled = 0.5;
const float CoverageNoiseMultiplier = 52.9829189;
const vec2 CoverageNoiseDirection = vec2(0.06711056, 0.00583715);
void main()
{
    float diffuseAlpha = Draw.TextureFlags.x != 0u
        ? texture(sampler2D(DiffuseImage, MaterialSampler), MaterialUV).a : 1.0;
    float textureOpacity = Draw.TextureFlags.z != 0u
        ? texture(sampler2D(OpacityImage, MaterialSampler), MaterialUV).r : 1.0;
    float alpha = clamp(VertexOpacity * Draw.MaterialParameters.y * diffuseAlpha * textureOpacity, 0.0, 1.0);
    if (Draw.MaterialParameters.z > 0.0)
    {
        if (alpha < Draw.MaterialParameters.z) discard;
    }
    else if (Draw.RenderParameters.w > BlendingEnabled)
    {
        // Fixed interleaved gradient coverage avoids temporal shadow flicker.
        float threshold = fract(CoverageNoiseMultiplier * fract(dot(floor(gl_FragCoord.xy), CoverageNoiseDirection)));
        if (alpha <= 0.0 || alpha < threshold) discard;
    }
}
#endif
