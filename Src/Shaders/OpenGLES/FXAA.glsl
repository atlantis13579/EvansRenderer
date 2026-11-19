#version 300 es
// Lottes, FXAA (2009): directional luminance edge filtering in display-sRGB.
// Hardware decodes the sRGB source and encodes the destination; only the
// perceptual filtering step explicitly converts between linear and sRGB.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float OversizedCorner = 3.0;
    vec2 position = gl_VertexID == 0 ? vec2(-1.0, -1.0) :
        gl_VertexID == 1 ? vec2(OversizedCorner, -1.0) : vec2(-1.0, OversizedCorner);
    gl_Position = vec4(position, 0.0, 1.0);
}
#else
uniform sampler2D SceneColor;
uniform vec4 InverseExtentStrengths;
out vec4 OutColor;
const vec3 LuminanceWeights = vec3(0.299, 0.587, 0.114);
const float EdgeThreshold = 1.0 / 8.0;
const float EdgeThresholdMinimum = 1.0 / 16.0;
const float ReductionMinimum = 1.0 / 128.0;
const float ReductionMultiplier = 1.0 / 8.0;
const float MaximumSpan = 8.0;
const float FourSampleWeight = 0.25;
const float OneThird = 1.0 / 3.0;
const float SrgbLinearThreshold = 0.0031308;
const float SrgbEncodedThreshold = 0.04045;
const float SrgbLinearScale = 12.92;
const float SrgbPowerOffset = 0.055;
const float SrgbPowerScale = 1.055;
const float SrgbExponent = 2.4;
vec3 Encode(vec3 linear)
{
    return mix(SrgbPowerScale * pow(max(linear, vec3(0.0)), vec3(1.0 / SrgbExponent)) - SrgbPowerOffset,
        linear * SrgbLinearScale, lessThanEqual(linear, vec3(SrgbLinearThreshold)));
}
vec3 Decode(vec3 display)
{
    return mix(pow((display + SrgbPowerOffset) / SrgbPowerScale, vec3(SrgbExponent)),
        display / SrgbLinearScale, lessThanEqual(display, vec3(SrgbEncodedThreshold)));
}
vec3 SampleDisplay(vec2 uv)
{
    vec2 halfTexel = InverseExtentStrengths.xy * 0.5;
    return Encode(texture(SceneColor, clamp(uv, halfTexel, 1.0 - halfTexel)).rgb);
}
void main()
{
    vec2 pixelSize = InverseExtentStrengths.xy;
    vec2 uv = gl_FragCoord.xy * pixelSize;
    vec4 center = texture(SceneColor, uv);
    float centerLuma = dot(Encode(center.rgb), LuminanceWeights);
    float northwest = dot(SampleDisplay(uv + vec2(-1.0, 1.0) * pixelSize), LuminanceWeights);
    float northeast = dot(SampleDisplay(uv + vec2(1.0, 1.0) * pixelSize), LuminanceWeights);
    float southwest = dot(SampleDisplay(uv + vec2(-1.0, -1.0) * pixelSize), LuminanceWeights);
    float southeast = dot(SampleDisplay(uv + vec2(1.0, -1.0) * pixelSize), LuminanceWeights);
    float minimumLuma = min(centerLuma, min(min(northwest, northeast), min(southwest, southeast)));
    float maximumLuma = max(centerLuma, max(max(northwest, northeast), max(southwest, southeast)));
    if (maximumLuma - minimumLuma < max(EdgeThresholdMinimum, maximumLuma * EdgeThreshold))
    {
        OutColor = center;
        return;
    }
    // GLES framebuffer Y increases upward; signs track that coordinate system.
    vec2 direction = vec2(-(northwest + northeast - southwest - southeast),
        northwest + southwest - northeast - southeast);
    direction.y = -direction.y;
    float reduction = max((northwest + northeast + southwest + southeast) * FourSampleWeight * ReductionMultiplier, ReductionMinimum);
    direction = clamp(direction / (min(abs(direction.x), abs(direction.y)) + reduction), -MaximumSpan, MaximumSpan) * pixelSize;
    vec3 inner = (SampleDisplay(uv + direction * (OneThird - 0.5)) + SampleDisplay(uv + direction * (2.0 * OneThird - 0.5))) * 0.5;
    vec3 outer = inner * 0.5 + (SampleDisplay(uv - direction * 0.5) + SampleDisplay(uv + direction * 0.5)) * FourSampleWeight;
    float outerLuma = dot(outer, LuminanceWeights);
    vec3 filtered = outerLuma < minimumLuma || outerLuma > maximumLuma ? inner : outer;
    OutColor = vec4(Decode(filtered), center.a);
}
#endif
