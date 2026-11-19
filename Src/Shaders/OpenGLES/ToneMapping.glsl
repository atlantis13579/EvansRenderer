#version 300 es
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// Reinhard et al., Photographic Tone Reproduction (2002), and Narkowicz,
// ACES Filmic Tone Mapping Curve (2016). The latter is a fitted approximation,
// not the complete ACES color pipeline. The sRGB render target encodes output.
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
uniform vec4 InverseSourceExtentBloom;
uniform vec4 GradeExposure;
uniform uvec4 PassToneMapping;
out vec4 OutColor;
void main()
{
    const uint AcesFittedOperator = 2u;
    const float ShoulderNumerator = 2.51;
    const float ToeNumerator = 0.03;
    const float ShoulderDenominator = 2.43;
    const float LinearDenominator = 0.59;
    const float ToeDenominator = 0.14;
    vec4 source = texture(SceneColor, ScreenUV);
    vec3 exposed = max(source.rgb, vec3(0.0)) * GradeExposure.z;
    vec3 mapped = exposed / (1.0 + exposed);
    if (PassToneMapping.y == AcesFittedOperator)
        mapped = (exposed * (ShoulderNumerator * exposed + ToeNumerator)) /
            (exposed * (ShoulderDenominator * exposed + LinearDenominator) + ToeDenominator);
    OutColor = vec4(clamp(mapped, 0.0, 1.0), source.a);
}
#endif
