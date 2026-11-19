#version 450
// Vulkan single-map depth generation, with no cross-API source translation.
// Williams, "Casting curved shadows on curved surfaces" (SIGGRAPH 1978),
// doi:10.1145/800248.807402. This produces visibility depth; shading files still
// own receiver comparisons and PCF. Alpha coverage is linear, not sRGB color.
layout(set = 0, binding = 0, std140, row_major) uniform ShadowDraw
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
} Draw;
#ifdef EVANS_VERTEX_STAGE
layout(location = 0) in vec3 InPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
layout(location = 0) out vec2 UV;
layout(location = 1) out float Opacity;
void main()
{
    vec4 world = Draw.World * vec4(InPosition, 1.0);
    gl_Position = Draw.ViewProjection * world;
    gl_Position.y = -gl_Position.y; // Positive-height viewport, top-left capture.
    UV = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    Opacity = InColor.a;
}
#else
layout(location = 0) in vec2 UV;
layout(location = 1) in float Opacity;
layout(set = 0, binding = 2) uniform texture2D DiffuseImage;
layout(set = 0, binding = 4) uniform texture2D OpacityImage;
layout(set = 0, binding = 5) uniform sampler MaterialSampler;
const float EnabledThreshold = 0.5;
const float InterleavedMultiplier = 52.9829189;
const vec2 InterleavedDirection = vec2(0.06711056, 0.00583715);
void main()
{
    float diffuseAlpha = Draw.TextureFlags.x != 0u
        ? texture(sampler2D(DiffuseImage, MaterialSampler), UV).a : 1.0;
    float opacityTexture = Draw.TextureFlags.z != 0u
        ? texture(sampler2D(OpacityImage, MaterialSampler), UV).r : 1.0;
    float coverage = clamp(Opacity * Draw.MaterialParameters.y * diffuseAlpha * opacityTexture, 0.0, 1.0);
    if (Draw.MaterialParameters.z > 0.0)
    {
        if (coverage < Draw.MaterialParameters.z) discard;
    }
    else if (Draw.RenderParameters.w > EnabledThreshold)
    {
        float threshold = fract(InterleavedMultiplier * fract(dot(floor(gl_FragCoord.xy), InterleavedDirection)));
        if (coverage <= 0.0 || coverage < threshold) discard;
    }
}
#endif
