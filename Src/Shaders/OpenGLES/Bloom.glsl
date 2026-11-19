#version 300 es
// Scene-linear bloom: threshold/downsample, separable Gaussian, then additive
// reconstruction. The normalized radius-four kernel has sigma = two texels.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
out vec2 ScreenUV;
void main()
{
    const float FarCorner = 3.0;
    vec2 p = gl_VertexID == 0 ? vec2(-1.0) :
        gl_VertexID == 1 ? vec2(FarCorner, -1.0) : vec2(-1.0, FarCorner);
    ScreenUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
#else
in vec2 ScreenUV;
uniform sampler2D SceneColor;
uniform sampler2D BloomColor;
uniform vec4 InverseSourceExtentBloom;
uniform vec4 GradeExposure;
uniform uvec4 PassToneMapping;
out vec4 OutColor;
const uint ExtractPass = 0u;
const uint HorizontalPass = 1u;
const uint VerticalPass = 2u;
const int KernelRadius = 4;
const float MaximumHalfFloatRadiance = 65504.0;
const float GaussianWeights[5] = float[5](0.20416369, 0.18017382, 0.12383154, 0.06628225, 0.02763055);
vec4 ReadScene(vec2 uv)
{
    vec2 edge = InverseSourceExtentBloom.xy * 0.5;
    return texture(SceneColor, clamp(uv, edge, 1.0 - edge));
}
void main()
{
    vec4 center = ReadScene(ScreenUV);
    if (PassToneMapping.x == ExtractPass)
    {
        vec3 bright = vec3(0.0);
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
            {
                vec2 offset = (vec2(x, y) - 0.5) * InverseSourceExtentBloom.xy;
                bright += max(ReadScene(ScreenUV + offset).rgb - InverseSourceExtentBloom.z, vec3(0.0));
            }
        const float BoxSampleWeight = 0.25;
        OutColor = vec4(bright * BoxSampleWeight, 1.0);
    }
    else if (PassToneMapping.x == HorizontalPass || PassToneMapping.x == VerticalPass)
    {
        vec2 stepUV = PassToneMapping.x == HorizontalPass ?
            vec2(InverseSourceExtentBloom.x, 0.0) : vec2(0.0, InverseSourceExtentBloom.y);
        vec3 sum = center.rgb * GaussianWeights[0];
        for (int tap = 1; tap <= KernelRadius; ++tap)
            sum += (ReadScene(ScreenUV + stepUV * float(tap)).rgb +
                ReadScene(ScreenUV - stepUV * float(tap)).rgb) * GaussianWeights[tap];
        OutColor = vec4(sum, 1.0);
    }
    else
    {
        vec2 bloomEdge = 0.5 / vec2(textureSize(BloomColor, 0));
        vec3 glow = texture(BloomColor, clamp(ScreenUV, bloomEdge, 1.0 - bloomEdge)).rgb;
        OutColor = vec4(center.rgb + InverseSourceExtentBloom.w * glow, center.a);
    }
    OutColor.rgb = min(OutColor.rgb, vec3(MaximumHalfFloatRadiance));
}
#endif
