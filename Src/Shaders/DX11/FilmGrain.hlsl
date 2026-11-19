// Deterministic monochrome grain in display RGB, with the center alpha retained.
// PCG permutation follows the integer-hash family discussed by Jarzynski and
// Olano, "Hash Functions for GPU Rendering" (JCGT 2020). Seed is user controlled.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
};
Texture2D<float4> SceneColor : register(t0);
SamplerState ClampSampler : register(s0);
static const uint PixelXMultiplier = 1973u;
static const uint PixelYMultiplier = 9277u;
static const uint StateMultiplier = 747796405u;
static const uint StateIncrement = 2891336453u;
static const uint PermutationMultiplier = 277803737u;
static const uint DynamicShift = 28u;
static const uint MinimumShift = 4u;
static const uint OutputShift = 22u;
static const uint NoiseMantissaMask = 0x00ffffffu;
static const float NoiseMantissaMaximum = 16777215.0f;
static const float SrgbLinearThreshold = 0.0031308f;
static const float SrgbEncodedThreshold = 0.04045f;
static const float SrgbSlope = 12.92f;
static const float SrgbScale = 1.055f;
static const float SrgbOffset = 0.055f;
static const float SrgbPower = 2.4f;
struct ScreenVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    ScreenVertex output;
    output.UV = float2((vertexId << 1u) & 2u, vertexId & 2u);
    output.Position = float4(output.UV.x * 2.0f - 1.0f, 1.0f - output.UV.y * 2.0f, 0.0f, 1.0f);
    return output;
}
float3 EncodeDisplay(float3 linearColor)
{
    float3 power = SrgbScale * pow(max(linearColor, 0.0f), 1.0f / SrgbPower) - SrgbOffset;
    return float3(linearColor.r <= SrgbLinearThreshold ? linearColor.r * SrgbSlope : power.r,
        linearColor.g <= SrgbLinearThreshold ? linearColor.g * SrgbSlope : power.g,
        linearColor.b <= SrgbLinearThreshold ? linearColor.b * SrgbSlope : power.b);
}
float3 DecodeDisplay(float3 displayColor)
{
    float3 power = pow((displayColor + SrgbOffset) / SrgbScale, SrgbPower);
    return float3(displayColor.r <= SrgbEncodedThreshold ? displayColor.r / SrgbSlope : power.r,
        displayColor.g <= SrgbEncodedThreshold ? displayColor.g / SrgbSlope : power.g,
        displayColor.b <= SrgbEncodedThreshold ? displayColor.b / SrgbSlope : power.b);
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    uint2 pixel = uint2(floor(input.UV / InverseExtentStrengths.xy));
    uint value = (pixel.x * PixelXMultiplier + pixel.y * PixelYMultiplier) ^ GrainSeedReserved.x;
    uint state = value * StateMultiplier + StateIncrement;
    uint word = ((state >> ((state >> DynamicShift) + MinimumShift)) ^ state) * PermutationMultiplier;
    uint hash = (word >> OutputShift) ^ word;
    float noise = float(hash & NoiseMantissaMask) / NoiseMantissaMaximum - 0.5f;
    float4 center = SceneColor.SampleLevel(ClampSampler, input.UV, 0);
    float3 display = saturate(EncodeDisplay(center.rgb) + noise * InverseExtentStrengths.w);
    return float4(DecodeDisplay(display), center.a);
}
