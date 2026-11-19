#version 450
// Native five-pass pseudo lens flare. Highlight visibility comes from resolved
// HDR scene color, so occluded sources contribute nothing and off-screen light
// is unavailable. Lens coefficients below are an artistic approximation.
// John Chapman, Screen Space Lens Flare (2017):
// https://john-chapman.github.io/2017/11/05/pseudo-lens-flare.html
#ifdef EVANS_VERTEX_STAGE
layout(location = 0) out vec2 UV;
void main()
{
    const float TriangleFarCorner = 3.0;
    vec2 corner = gl_VertexIndex == 0 ? vec2(-1.0) : gl_VertexIndex == 1 ? vec2(TriangleFarCorner, -1.0) : vec2(-1.0, TriangleFarCorner);
    UV = corner * 0.5 + 0.5;
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
layout(location = 0) in vec2 UV;
layout(set = 0, binding = 0, std140) uniform HdrConstants
{
    vec4 InverseSourceExtentBloom;
    vec4 GradeExposure;
    uvec4 PassToneMapping;
    vec4 FlareThresholdIntensitySpacingRadius;
    vec4 FlareAspect;
} Hdr;
layout(set = 0, binding = 2) uniform texture2D SceneImage;
layout(set = 0, binding = 3) uniform texture2D FeatureImage;
layout(set = 0, binding = 5) uniform sampler LinearSampler;
layout(location = 0) out vec4 Color;
const uint Extract = 6u, Generate = 7u, Horizontal = 8u, Vertical = 9u;
const float HalfFloatMaximum = 65504.0;
const float NormalizationEpsilon = 0.000001;
const vec3 Rec709Luminance = vec3(0.2126, 0.7152, 0.0722);
const int NumberOfGhosts = 4;
const float GhostMagnification[NumberOfGhosts] = float[NumberOfGhosts](-1.0, -0.5, 0.5, 1.5);
const vec3 GhostColor[NumberOfGhosts] = vec3[NumberOfGhosts](vec3(1.0, 0.65, 0.35), vec3(0.4, 0.75, 1.0), vec3(0.65, 1.0, 0.5), vec3(1.0, 0.4, 0.65));
const float BorderFeather = 0.05, RingThickness = 0.08, RingIntensity = 0.5;
const int BlurRadius = 2;
const float BinomialWeight[3] = float[3](0.375, 0.25, 0.0625);
vec4 ReadColor(texture2D image, vec2 coordinate)
{
    vec2 border = 0.5 / vec2(textureSize(sampler2D(image, LinearSampler), 0));
    return textureLod(sampler2D(image, LinearSampler), clamp(coordinate, border, 1.0 - border), 0.0);
}
vec3 Highlight(vec2 coordinate)
{
    vec3 color = max(ReadColor(SceneImage, coordinate).rgb, 0.0);
    float luminance = dot(color, Rec709Luminance);
    return color * max(luminance - Hdr.FlareThresholdIntensitySpacingRadius.x, 0.0) / max(luminance, NormalizationEpsilon);
}
vec3 Donor(vec2 coordinate)
{
    vec2 borderDistance = min(coordinate, 1.0 - coordinate);
    float edgeMask = smoothstep(0.0, BorderFeather, min(borderDistance.x, borderDistance.y));
    return ReadColor(SceneImage, coordinate).rgb * edgeMask;
}
void main()
{
    uint stage = Hdr.PassToneMapping.x;
    if (stage == Extract)
    {
        vec2 halfPixel = Hdr.InverseSourceExtentBloom.xy * 0.5;
        vec3 extracted = Highlight(UV - halfPixel) + Highlight(UV + halfPixel) +
            Highlight(UV + halfPixel * vec2(-1.0, 1.0)) + Highlight(UV + halfPixel * vec2(1.0, -1.0));
        const float FourSampleNormalization = 0.25;
        Color = vec4(min(extracted * FourSampleNormalization, HalfFloatMaximum), 0.0);
    }
    else if (stage == Generate)
    {
        vec2 centered = UV - 0.5;
        vec3 features = vec3(0.0);
        for (int index = 0; index < NumberOfGhosts; ++index)
            features += Donor(0.5 + centered / (GhostMagnification[index] * Hdr.FlareThresholdIntensitySpacingRadius.z)) * GhostColor[index] / float(NumberOfGhosts);
        vec2 radial = centered * vec2(Hdr.FlareAspect.x, 1.0);
        float distanceFromCenter = length(radial);
        float haloRadius = Hdr.FlareThresholdIntensitySpacingRadius.w;
        if (haloRadius > 0.0)
        {
            vec2 direction = radial / max(distanceFromCenter, NormalizationEpsilon);
            vec2 source = 0.5 + direction * (distanceFromCenter - haloRadius) * vec2(Hdr.FlareAspect.y, 1.0);
            float mask = 1.0 - smoothstep(0.0, RingThickness, abs(distanceFromCenter - haloRadius));
            features += Donor(source) * mask * RingIntensity;
        }
        Color = vec4(min(features, HalfFloatMaximum), 0.0);
    }
    else if (stage == Horizontal || stage == Vertical)
    {
        vec2 direction = stage == Horizontal ? vec2(Hdr.InverseSourceExtentBloom.x, 0.0) : vec2(0.0, Hdr.InverseSourceExtentBloom.y);
        vec3 blurred = ReadColor(SceneImage, UV).rgb * BinomialWeight[0];
        for (int tap = 1; tap <= BlurRadius; ++tap)
            blurred += (ReadColor(SceneImage, UV + direction * float(tap)).rgb + ReadColor(SceneImage, UV - direction * float(tap)).rgb) * BinomialWeight[tap];
        Color = vec4(min(blurred, HalfFloatMaximum), 0.0);
    }
    else
    {
        vec4 scene = ReadColor(SceneImage, UV);
        Color = vec4(min(scene.rgb + ReadColor(FeatureImage, UV).rgb * Hdr.FlareThresholdIntensitySpacingRadius.y, HalfFloatMaximum), scene.a);
    }
}
#endif
