#version 300 es
// Analytic scene-linear grading needs no LUT asset. Rec.709 luminance controls
// saturation; contrast pivots around photographic middle gray before tone mapping.
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
    const vec3 Rec709Luminance = vec3(0.2126, 0.7152, 0.0722);
    const float MiddleGray = 0.18;
    const float MaximumHalfFloatRadiance = 65504.0;
    vec4 source = texture(SceneColor, ScreenUV);
    float luminance = dot(source.rgb, Rec709Luminance);
    vec3 saturated = mix(vec3(luminance), source.rgb, GradeExposure.y);
    vec3 graded = (saturated - MiddleGray) * GradeExposure.x + MiddleGray;
    OutColor = vec4(clamp(graded, 0.0, MaximumHalfFloatRadiance), source.a);
}
#endif
