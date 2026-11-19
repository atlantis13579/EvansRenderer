#version 300 es
// Visible HDR highlights produce tinted ghosts and a circular halo, followed
// by separable binomial blur and additive composition. No authored flare image.
// Reference: John Chapman, Screen Space Lens Flare (2017).
// https://john-chapman.github.io/2017/11/05/pseudo-lens-flare.html
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
out vec2 UV;
void main()
{
    const float OversizedCorner = 3.0;
    vec2 point = gl_VertexID == 0 ? vec2(-1.0) : gl_VertexID == 1 ? vec2(OversizedCorner, -1.0) : vec2(-1.0, OversizedCorner);
    gl_Position = vec4(point, 0.0, 1.0);
    UV = point * 0.5 + 0.5;
}
#else
in vec2 UV;
uniform sampler2D SceneColor;
uniform sampler2D BloomColor; // Shared HDR auxiliary slot holds the completed flare.
uniform vec4 InverseSourceExtentBloom;
uniform uvec4 PassToneMapping;
uniform vec4 FlareThresholdIntensitySpacingRadius;
uniform vec4 FlareAspect;
out vec4 OutColor;
const uint ExtractStage = 6u, FeaturesStage = 7u, BlurXStage = 8u, BlurYStage = 9u;
const float MaximumHalfRadiance = 65504.0;
const float SafeDivisor = 0.000001;
const vec3 Luminance = vec3(0.2126, 0.7152, 0.0722);
const int GhostCount = 4;
const float Scales[GhostCount] = float[GhostCount](-1.0, -0.5, 0.5, 1.5);
const vec3 Tints[GhostCount] = vec3[GhostCount](vec3(1.0, 0.65, 0.35), vec3(0.4, 0.75, 1.0), vec3(0.65, 1.0, 0.5), vec3(1.0, 0.4, 0.65));
const float EdgeWidth = 0.05, HaloWidth = 0.08, HaloWeight = 0.5;
const int KernelRadius = 2;
const float Kernel[3] = float[3](0.375, 0.25, 0.0625); // [1,4,6,4,1]/16.
vec4 ReadClamped(sampler2D image, vec2 uv)
{
    vec2 margin = 0.5 / vec2(textureSize(image, 0));
    return textureLod(image, clamp(uv, margin, 1.0 - margin), 0.0);
}
vec3 Bright(vec2 uv)
{
    vec3 color = max(ReadClamped(SceneColor, uv).rgb, 0.0);
    float light = dot(color, Luminance);
    return color * max(light - FlareThresholdIntensitySpacingRadius.x, 0.0) / max(light, SafeDivisor);
}
vec3 Feature(vec2 uv)
{
    // Discard out-of-frame donors smoothly instead of wrapping a highlight.
    vec2 edge = min(uv, 1.0 - uv);
    return ReadClamped(SceneColor, uv).rgb * smoothstep(0.0, EdgeWidth, min(edge.x, edge.y));
}
void main()
{
    if (PassToneMapping.x == ExtractStage)
    {
        vec3 bright = vec3(0.0);
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
                bright += Bright(UV + (vec2(x, y) - 0.5) * InverseSourceExtentBloom.xy);
        const float BoxWeight = 0.25;
        OutColor = vec4(min(bright * BoxWeight, MaximumHalfRadiance), 0.0);
    }
    else if (PassToneMapping.x == FeaturesStage)
    {
        vec2 offset = UV - 0.5;
        vec3 features = vec3(0.0);
        for (int ghost = 0; ghost < GhostCount; ++ghost)
            features += Feature(0.5 + offset / (Scales[ghost] * FlareThresholdIntensitySpacingRadius.z)) * Tints[ghost] / float(GhostCount);
        vec2 metric = offset * vec2(FlareAspect.x, 1.0);
        float distanceToCenter = length(metric);
        float radius = FlareThresholdIntensitySpacingRadius.w;
        if (radius > 0.0)
        {
            vec2 direction = metric / max(distanceToCenter, SafeDivisor);
            vec2 donor = 0.5 + direction * (distanceToCenter - radius) * vec2(FlareAspect.y, 1.0);
            float ring = 1.0 - smoothstep(0.0, HaloWidth, abs(distanceToCenter - radius));
            features += Feature(donor) * ring * HaloWeight;
        }
        OutColor = vec4(min(features, MaximumHalfRadiance), 0.0);
    }
    else if (PassToneMapping.x == BlurXStage || PassToneMapping.x == BlurYStage)
    {
        vec2 stepUV = PassToneMapping.x == BlurXStage ? vec2(InverseSourceExtentBloom.x, 0.0) : vec2(0.0, InverseSourceExtentBloom.y);
        vec3 filtered = ReadClamped(SceneColor, UV).rgb * Kernel[0];
        for (int offset = 1; offset <= KernelRadius; ++offset)
            filtered += (ReadClamped(SceneColor, UV + float(offset) * stepUV).rgb + ReadClamped(SceneColor, UV - float(offset) * stepUV).rgb) * Kernel[offset];
        OutColor = vec4(min(filtered, MaximumHalfRadiance), 0.0);
    }
    else
    {
        vec4 scene = ReadClamped(SceneColor, UV);
        OutColor = vec4(min(scene.rgb + ReadClamped(BloomColor, UV).rgb * FlareThresholdIntensitySpacingRadius.y, MaximumHalfRadiance), scene.a);
    }
}
#endif
