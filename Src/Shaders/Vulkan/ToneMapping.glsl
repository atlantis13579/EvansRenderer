#version 450
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// Reinhard et al., Photographic Tone Reproduction (2002), component operator.
// Narkowicz, ACES Filmic Tone Mapping Curve (2016), rational approximation:
// this compact fit is not the complete ACES transform. The target is sRGB.
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
    const uint FilmicFit = 2u;
    const float NumeratorQuadratic = 2.51;
    const float NumeratorLinear = 0.03;
    const float DenominatorQuadratic = 2.43;
    const float DenominatorLinear = 0.59;
    const float DenominatorConstant = 0.14;
    vec4 source = texture(sampler2D(SceneImage, LinearSampler), UV);
    vec3 exposed = max(source.rgb, vec3(0.0)) * Hdr.GradeExposure.z;
    vec3 mapped = exposed / (1.0 + exposed);
    if (Hdr.PassToneMapping.y == FilmicFit)
        mapped = (exposed * (NumeratorQuadratic * exposed + NumeratorLinear)) /
            (exposed * (DenominatorQuadratic * exposed + DenominatorLinear) + DenominatorConstant);
    Color = vec4(clamp(mapped, 0.0, 1.0), source.a);
}
#endif
