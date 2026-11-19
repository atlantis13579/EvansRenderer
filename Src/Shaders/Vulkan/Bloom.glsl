#version 450
// Four HDR stages share this standalone technique: bright-pass downsampling,
// horizontal/vertical Gaussian convolution, and linear-radiance composition.
#ifdef EVANS_VERTEX_STAGE
layout(location = 0) out vec2 UV;
void main()
{
    const float FullscreenExtent = 3.0;
    vec2 point = gl_VertexIndex == 0 ? vec2(-1.0) :
        gl_VertexIndex == 1 ? vec2(FullscreenExtent, -1.0) : vec2(-1.0, FullscreenExtent);
    UV = point * 0.5 + 0.5;
    gl_Position = vec4(point, 0.0, 1.0);
}
#else
layout(location = 0) in vec2 UV;
layout(set = 0, binding = 0, std140) uniform HdrConstants
{ vec4 InverseSourceExtentBloom; vec4 GradeExposure; uvec4 PassToneMapping; } Hdr;
layout(set = 0, binding = 2) uniform texture2D SceneImage;
layout(set = 0, binding = 3) uniform texture2D BloomImage;
layout(set = 0, binding = 5) uniform sampler LinearSampler;
layout(location = 0) out vec4 Color;
const uint ExtractStage = 0u;
const uint HorizontalStage = 1u;
const uint VerticalStage = 2u;
const int FilterRadius = 4;
// exp(-x*x/(2*sigma*sigma)), normalized over [-4,4], sigma=2.
const float Weight[5] = float[5](0.20416369, 0.18017382, 0.12383154, 0.06628225, 0.02763055);
vec4 SampleRadiance(vec2 coordinate)
{
    vec2 border = 0.5 * Hdr.InverseSourceExtentBloom.xy;
    return texture(sampler2D(SceneImage, LinearSampler), clamp(coordinate, border, 1.0 - border));
}
void main()
{
    const float MaximumHalfFloatRadiance = 65504.0;
    vec4 scene = SampleRadiance(UV);
    if (Hdr.PassToneMapping.x == ExtractStage)
    {
        vec3 bright = vec3(0.0);
        for (int row = 0; row < 2; ++row)
            for (int column = 0; column < 2; ++column)
                bright += max(SampleRadiance(UV + (vec2(column, row) - 0.5) *
                    Hdr.InverseSourceExtentBloom.xy).rgb - Hdr.InverseSourceExtentBloom.z, vec3(0.0));
        const float FourTapNormalization = 0.25;
        Color = vec4(bright * FourTapNormalization, 1.0);
    }
    else if (Hdr.PassToneMapping.x == HorizontalStage || Hdr.PassToneMapping.x == VerticalStage)
    {
        vec2 delta = Hdr.PassToneMapping.x == HorizontalStage ?
            vec2(Hdr.InverseSourceExtentBloom.x, 0.0) : vec2(0.0, Hdr.InverseSourceExtentBloom.y);
        vec3 filtered = scene.rgb * Weight[0];
        for (int tap = 1; tap <= FilterRadius; ++tap)
            filtered += (SampleRadiance(UV - float(tap) * delta).rgb +
                SampleRadiance(UV + float(tap) * delta).rgb) * Weight[tap];
        Color = vec4(filtered, 1.0);
    }
    else
    {
        vec2 border = 0.5 / vec2(textureSize(sampler2D(BloomImage, LinearSampler), 0));
        vec3 blurred = texture(sampler2D(BloomImage, LinearSampler), clamp(UV, border, 1.0 - border)).rgb;
        Color = vec4(scene.rgb + blurred * Hdr.InverseSourceExtentBloom.w, scene.a);
    }
    Color.rgb = min(Color.rgb, vec3(MaximumHalfFloatRadiance));
}
#endif
