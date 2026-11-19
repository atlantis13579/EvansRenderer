#version 300 es
// Deterministic PCG-hash grain in display-sRGB. A fixed seed permits exact
// setting round trips; animated grain can later drive the seed explicitly.
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
uniform highp uvec4 GrainSeedReserved;
out vec4 OutColor;
const float SrgbLinearThreshold = 0.0031308;
const float SrgbEncodedThreshold = 0.04045;
const float SrgbLinearScale = 12.92;
const float SrgbPowerOffset = 0.055;
const float SrgbPowerScale = 1.055;
const float SrgbExponent = 2.4;
vec3 Encode(vec3 value)
{
    return mix(SrgbPowerScale * pow(max(value, vec3(0.0)), vec3(1.0 / SrgbExponent)) - SrgbPowerOffset,
        value * SrgbLinearScale, lessThanEqual(value, vec3(SrgbLinearThreshold)));
}
vec3 Decode(vec3 value)
{
    return mix(pow((value + SrgbPowerOffset) / SrgbPowerScale, vec3(SrgbExponent)),
        value / SrgbLinearScale, lessThanEqual(value, vec3(SrgbEncodedThreshold)));
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
    const float NoiseNormalization = 16777215.0;
    vec2 uv = gl_FragCoord.xy * InverseExtentStrengths.xy;
    uvec2 pixel = uvec2(floor(gl_FragCoord.x), floor(1.0 / InverseExtentStrengths.y - gl_FragCoord.y));
    uint inputValue = (pixel.x * PixelMultiplierX + pixel.y * PixelMultiplierY) ^ GrainSeedReserved.x;
    uint state = inputValue * PcgMultiplier + PcgIncrement;
    uint word = ((state >> ((state >> PcgHighShift) + PcgShiftOffset)) ^ state) * PcgOutputMultiplier;
    uint hash = (word >> PcgFinalShift) ^ word;
    float noise = float(hash & NoiseMask) / NoiseNormalization - 0.5;
    vec4 color = texture(SceneColor, uv);
    vec3 display = clamp(Encode(color.rgb) + noise * InverseExtentStrengths.w, 0.0, 1.0);
    OutColor = vec4(Decode(display), color.a);
}
#endif
