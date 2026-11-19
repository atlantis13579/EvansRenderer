#version 450
// Native Vulkan implementation of Lottes' FXAA directional filter (2009).
// Scene images are sRGB views. Bilinear sampling yields linear light, which
// is encoded for perceptual luma decisions and decoded before the sRGB write.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float TriangleFarCorner = 3.0;
    vec2 position = gl_VertexIndex == 0 ? vec2(-1.0, -1.0) :
        gl_VertexIndex == 1 ? vec2(TriangleFarCorner, -1.0) : vec2(-1.0, TriangleFarCorner);
    gl_Position = vec4(position, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140) uniform PostConstants
{
    vec4 InverseExtentStrengths;
    uvec4 GrainSeedReserved;
} Post;
layout(set = 0, binding = 2) uniform texture2D SceneColor;
layout(set = 0, binding = 5) uniform sampler ColorSampler;
layout(location = 0) out vec4 OutputColor;
const vec3 PerceptualLuma = vec3(0.299, 0.587, 0.114);
const float ContrastFloor = 1.0 / 16.0;
const float ContrastScale = 1.0 / 8.0;
const float DirectionReductionFloor = 1.0 / 128.0;
const float DirectionReductionScale = 1.0 / 8.0;
const float MaximumEdgeSpan = 8.0;
const float Quarter = 0.25;
const float Third = 1.0 / 3.0;
const float SrgbToe = 0.0031308;
const float SrgbEncodedToe = 0.04045;
const float SrgbSlope = 12.92;
const float SrgbScale = 1.055;
const float SrgbOffset = 0.055;
const float SrgbGamma = 2.4;
vec3 ToDisplay(vec3 linear)
{
    return mix(SrgbScale * pow(max(linear, vec3(0.0)), vec3(1.0 / SrgbGamma)) - SrgbOffset,
        SrgbSlope * linear, lessThanEqual(linear, vec3(SrgbToe)));
}
vec3 ToLinear(vec3 display)
{
    return mix(pow((display + SrgbOffset) / SrgbScale, vec3(SrgbGamma)),
        display / SrgbSlope, lessThanEqual(display, vec3(SrgbEncodedToe)));
}
vec3 ReadPerceptual(vec2 coordinate)
{
    vec2 border = Post.InverseExtentStrengths.xy * 0.5;
    return ToDisplay(texture(sampler2D(SceneColor, ColorSampler), clamp(coordinate, border, 1.0 - border)).rgb);
}
void main()
{
    vec2 texel = Post.InverseExtentStrengths.xy;
    vec2 coordinate = gl_FragCoord.xy * texel;
    vec4 center = texture(sampler2D(SceneColor, ColorSampler), coordinate);
    float middle = dot(ToDisplay(center.rgb), PerceptualLuma);
    float northwest = dot(ReadPerceptual(coordinate - texel), PerceptualLuma);
    float northeast = dot(ReadPerceptual(coordinate + vec2(texel.x, -texel.y)), PerceptualLuma);
    float southwest = dot(ReadPerceptual(coordinate + vec2(-texel.x, texel.y)), PerceptualLuma);
    float southeast = dot(ReadPerceptual(coordinate + texel), PerceptualLuma);
    float darkest = min(middle, min(min(northwest, northeast), min(southwest, southeast)));
    float brightest = max(middle, max(max(northwest, northeast), max(southwest, southeast)));
    if (brightest - darkest < max(ContrastFloor, brightest * ContrastScale))
    {
        OutputColor = center;
        return;
    }
    vec2 direction = vec2(-(northwest + northeast - southwest - southeast),
        northwest + southwest - northeast - southeast);
    float reduction = max((northwest + northeast + southwest + southeast) * Quarter * DirectionReductionScale, DirectionReductionFloor);
    direction = clamp(direction / (min(abs(direction.x), abs(direction.y)) + reduction), -MaximumEdgeSpan, MaximumEdgeSpan) * texel;
    vec3 inner = (ReadPerceptual(coordinate + direction * (Third - 0.5)) +
        ReadPerceptual(coordinate + direction * (2.0 * Third - 0.5))) * 0.5;
    vec3 outer = inner * 0.5 + (ReadPerceptual(coordinate - direction * 0.5) + ReadPerceptual(coordinate + direction * 0.5)) * Quarter;
    float outerLuma = dot(outer, PerceptualLuma);
    vec3 filtered = outerLuma < darkest || outerLuma > brightest ? inner : outer;
    OutputColor = vec4(ToLinear(filtered), center.a);
}
#endif
