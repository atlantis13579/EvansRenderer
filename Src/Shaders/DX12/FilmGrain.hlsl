// Static-seed monochrome display grain. PCG integer permutation gives stable
// pixel noise across APIs; see Jarzynski and Olano, JCGT "Hash Functions for
// GPU Rendering" (2020). Pixel coordinates use the top-left framebuffer origin.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
};
Texture2D<float4> SceneColor : register(t0);
SamplerState LinearClamp : register(s0);
static const uint HorizontalMultiplier = 1973u;
static const uint VerticalMultiplier = 9277u;
static const uint PcgMultiplier = 747796405u;
static const uint PcgIncrement = 2891336453u;
static const uint PcgOutputMultiplier = 277803737u;
static const uint PcgVariableShift = 28u;
static const uint PcgBaseShift = 4u;
static const uint PcgFinalShift = 22u;
static const uint UniformMask = 0x00ffffffu;
static const float UniformMaximum = 16777215.0f;
static const float LinearBreakpoint = 0.0031308f;
static const float EncodedBreakpoint = 0.04045f;
static const float TransferSlope = 12.92f;
static const float TransferScale = 1.055f;
static const float TransferOffset = 0.055f;
static const float TransferExponent = 2.4f;
struct FullscreenOutput { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
FullscreenOutput VSMain(uint id : SV_VertexID)
{
    FullscreenOutput result;
    float2 corner = float2((id << 1u) & 2u, id & 2u);
    result.UV = corner;
    result.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return result;
}
float3 ToPerceptual(float3 value)
{
    float3 nonlinear = TransferScale * pow(max(value, 0.0f), 1.0f / TransferExponent) - TransferOffset;
    return float3(value.r <= LinearBreakpoint ? value.r * TransferSlope : nonlinear.r,
        value.g <= LinearBreakpoint ? value.g * TransferSlope : nonlinear.g,
        value.b <= LinearBreakpoint ? value.b * TransferSlope : nonlinear.b);
}
float3 ToLinear(float3 value)
{
    float3 nonlinear = pow((value + TransferOffset) / TransferScale, TransferExponent);
    return float3(value.r <= EncodedBreakpoint ? value.r / TransferSlope : nonlinear.r,
        value.g <= EncodedBreakpoint ? value.g / TransferSlope : nonlinear.g,
        value.b <= EncodedBreakpoint ? value.b / TransferSlope : nonlinear.b);
}
float4 PSMain(FullscreenOutput fragment) : SV_TARGET
{
    uint2 pixel = uint2(floor(fragment.UV / InverseExtentStrengths.xy));
    uint inputSeed = (pixel.x * HorizontalMultiplier + pixel.y * VerticalMultiplier) ^ GrainSeedReserved.x;
    uint state = inputSeed * PcgMultiplier + PcgIncrement;
    uint permutation = ((state >> ((state >> PcgVariableShift) + PcgBaseShift)) ^ state) * PcgOutputMultiplier;
    uint randomBits = (permutation >> PcgFinalShift) ^ permutation;
    float signedNoise = float(randomBits & UniformMask) / UniformMaximum - 0.5f;
    float4 linearColor = SceneColor.SampleLevel(LinearClamp, fragment.UV, 0);
    float3 noisyDisplay = saturate(ToPerceptual(linearColor.rgb) + signedNoise * InverseExtentStrengths.w);
    return float4(ToLinear(noisyDisplay), linearColor.a);
}
