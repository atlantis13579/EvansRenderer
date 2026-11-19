#version 450
// Rec.709 luminance supplies the neutral axis for saturation. Linear contrast
// pivots at 18% middle gray; this analytic operator requires no external LUT.
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
layout(set = 0, binding = 5) uniform sampler LinearSampler;
layout(location = 0) out vec4 Color;
void main()
{
    const vec3 LuminanceCoefficients = vec3(0.2126, 0.7152, 0.0722);
    const float ContrastPivot = 0.18;
    const float MaximumHalfFloatRadiance = 65504.0;
    vec4 source = texture(sampler2D(SceneImage, LinearSampler), UV);
    vec3 neutral = vec3(dot(source.rgb, LuminanceCoefficients));
    vec3 saturated = mix(neutral, source.rgb, Hdr.GradeExposure.y);
    vec3 graded = (saturated - ContrastPivot) * Hdr.GradeExposure.x + ContrastPivot;
    Color = vec4(clamp(graded, 0.0, MaximumHalfFloatRadiance), source.a);
}
#endif
