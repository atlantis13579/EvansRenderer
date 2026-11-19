#version 450
// PCG integer mixing provides deterministic, seed-controlled monochromatic
// grain. Noise is added in display-sRGB; source and output remain sRGB images.
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
void main()
{
    const uint PixelMultiplierX = 1973u;
    const uint PixelMultiplierY = 9277u;
    const uint PcgMultiplier = 747796405u;
    const uint PcgIncrement = 2891336453u;
    const uint PcgOutputMultiplier = 277803737u;
    const uint PcgHighShift = 28u;
    const uint PcgShiftOffset = 4u;
    const uint PcgFinalShift = 22u;
    const uint NoiseMask = 0x00ffffffu;
    const float NoiseRange = 16777215.0;
    uvec2 pixel = uvec2(floor(gl_FragCoord.xy));
    uint inputValue = (pixel.x * PixelMultiplierX + pixel.y * PixelMultiplierY) ^ Post.GrainSeedReserved.x;
    uint state = inputValue * PcgMultiplier + PcgIncrement;
    uint word = ((state >> ((state >> PcgHighShift) + PcgShiftOffset)) ^ state) * PcgOutputMultiplier;
    uint hash = (word >> PcgFinalShift) ^ word;
    float noise = float(hash & NoiseMask) / NoiseRange - 0.5;
    vec4 color = texture(sampler2D(SceneColor, ColorSampler), gl_FragCoord.xy * Post.InverseExtentStrengths.xy);
    vec3 display = clamp(ToDisplay(color.rgb) + noise * Post.InverseExtentStrengths.w, 0.0, 1.0);
    OutputColor = vec4(ToLinear(display), color.a);
}
#endif
