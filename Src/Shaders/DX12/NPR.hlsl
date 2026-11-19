// DX12 image-space ink: XDoG, Winnemoeller et al., Computers & Graphics 2012.
// https://www.kyprianidis.com/p/cag2012/
// This source owns contour filtering, procedural crosshatching and color
// quantization. It consumes the resolved, tone-mapped scene, including alpha.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
    float4 NprEdge;
    float4 NprStyle;
    float4 NprHatch;
    float4 NprTone;
};
Texture2D<float4> SceneColor : register(t0);
struct RasterVertex { float4 Position : SV_POSITION; };
RasterVertex VSMain(uint id : SV_VertexID)
{
    const float FarCorner = 3.0;
    float2 corner = id == 0 ? float2(-1, -1) : id == 1 ? float2(FarCorner, -1) : float2(-1, FarCorner);
    RasterVertex vertex; vertex.Position = float4(corner, 0, 1); return vertex;
}
float3 ToDisplay(float3 sceneLinear)
{
    const float LinearThreshold = 0.0031308, LinearSlope = 12.92;
    const float GammaScale = 1.055, GammaOffset = 0.055, GammaPower = 2.4;
    float3 curved = GammaScale * pow(saturate(sceneLinear), 1.0 / GammaPower) - GammaOffset;
    return float3(sceneLinear.x <= LinearThreshold ? LinearSlope * sceneLinear.x : curved.x,
        sceneLinear.y <= LinearThreshold ? LinearSlope * sceneLinear.y : curved.y,
        sceneLinear.z <= LinearThreshold ? LinearSlope * sceneLinear.z : curved.z);
}
float3 ToLinear(float3 display)
{
    const float DisplayThreshold = 0.04045, LinearSlope = 12.92;
    const float GammaScale = 1.055, GammaOffset = 0.055, GammaPower = 2.4;
    float3 curved = pow((display + GammaOffset) / GammaScale, GammaPower);
    return float3(display.x <= DisplayThreshold ? display.x / LinearSlope : curved.x,
        display.y <= DisplayThreshold ? display.y / LinearSlope : curved.y,
        display.z <= DisplayThreshold ? display.z / LinearSlope : curved.z);
}
float InkContour(int2 center, int2 dimensions)
{
    const float ScaleRatio = 1.6, SupportStandardDeviations = 3.0;
    const int KernelLimit = 8; // Covers maximum allowed sigma with three-sigma support.
    const float3 PerceptualLuma = float3(0.2126, 0.7152, 0.0722);
    float narrowVariance = NprEdge.x * NprEdge.x;
    float2 variance = float2(narrowVariance, narrowVariance * ScaleRatio * ScaleRatio);
    int support = min(KernelLimit, int(ceil(SupportStandardDeviations * NprEdge.x * ScaleRatio)));
    float2 sum = 0, weights = 0;
    [loop] for (int row = -support; row <= support; ++row)
        [loop] for (int column = -support; column <= support; ++column)
        {
            int2 donor = clamp(center + int2(column, row), 0, dimensions - 1);
            float intensity = dot(ToDisplay(SceneColor.Load(int3(donor, 0)).rgb), PerceptualLuma);
            float2 gaussian = exp(-float(row * row + column * column) / (2 * variance));
            sum += gaussian * intensity; weights += gaussian;
        }
    float2 means = sum / weights;
    float difference = means.x - NprEdge.y * means.y;
    // Stable one-sided XDoG soft threshold: white above epsilon, 1+tanh below.
    float e = exp(2 * min(0, (difference - NprEdge.z) * NprEdge.w));
    return 2 * e / (1 + e);
}
float StrokeCoverage(float2 position, float2 axis)
{
    float nearest = abs(frac(dot(position, axis) / NprHatch.x + 0.5) - 0.5) * NprHatch.x;
    float halfWidth = NprHatch.y * 0.5;
    float pixelFootprint = dot(abs(axis), float2(1, 1)) * 0.5;
    return 1 - smoothstep(max(0, halfWidth - pixelFootprint), halfWidth + pixelFootprint, nearest);
}
float4 PSMain(RasterVertex vertex) : SV_TARGET
{
    uint width, height; SceneColor.GetDimensions(width, height);
    int2 center = int2(vertex.Position.xy);
    float4 rgba = SceneColor.Load(int3(center, 0));
    float3 display = saturate(ToDisplay(rgba.rgb));
    const float3 PerceptualLuma = float3(0.2126, 0.7152, 0.0722);
    const float FirstTone = 0.25, SecondTone = 0.5, ThirdTone = 0.75, Transition = 0.1;
    const float InverseSqrtTwo = 0.7071067811865475;
    float3 thresholds = float3(FirstTone, SecondTone, ThirdTone);
    float darkness = 1 - dot(display, PerceptualLuma);
    float3 masks = smoothstep(thresholds - Transition, thresholds + Transition, darkness);
    float2 direction = NprHatch.zw;
    float a = StrokeCoverage(vertex.Position.xy, direction);
    float b = StrokeCoverage(vertex.Position.xy, float2(-direction.y, direction.x));
    float c = StrokeCoverage(vertex.Position.xy, float2(direction.x - direction.y, direction.x + direction.y) * InverseSqrtTwo);
    float3 transmittance = 1 - float3(a, b, c) * masks * NprStyle.y;
    float edge = NprStyle.x > 0 ? InkContour(center, int2(width, height)) : 1;
    float toneIntervals = NprTone.x - 1;
    float3 tones = floor(display * toneIntervals + 0.5) / toneIntervals;
    float3 paperColor = lerp(float3(1, 1, 1), tones, NprStyle.z);
    float3 styled = paperColor * lerp(1, edge, NprStyle.x) * transmittance.x * transmittance.y * transmittance.z;
    return float4(ToLinear(lerp(display, styled, NprStyle.w)), rgba.a);
}
