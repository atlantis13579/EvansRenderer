// Isotropic XDoG contours: Winnemoeller, Kyprianidis and Olsen (2012).
// https://www.kyprianidis.com/p/cag2012/
// Finite normalized Gaussians act on display-sRGB luma. Analytic tonal hatch
// layers and posterized color need no textures beyond the completed scene.
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
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    float2 corner = float2((vertexId << 1u) & 2u, vertexId & 2u);
    ScreenVertex result;
    result.Position = float4(corner * float2(2, -2) + float2(-1, 1), 0, 1);
    return result;
}
float3 DisplayEncode(float3 value)
{
    const float LinearThreshold = 0.0031308, LinearScale = 12.92;
    const float PowerScale = 1.055, PowerOffset = 0.055, Exponent = 2.4;
    float3 power = PowerScale * pow(saturate(value), 1.0 / Exponent) - PowerOffset;
    return float3(value.r <= LinearThreshold ? value.r * LinearScale : power.r,
        value.g <= LinearThreshold ? value.g * LinearScale : power.g,
        value.b <= LinearThreshold ? value.b * LinearScale : power.b);
}
float3 DisplayDecode(float3 value)
{
    const float EncodedThreshold = 0.04045, LinearScale = 12.92;
    const float PowerScale = 1.055, PowerOffset = 0.055, Exponent = 2.4;
    float3 power = pow((value + PowerOffset) / PowerScale, Exponent);
    return float3(value.r <= EncodedThreshold ? value.r / LinearScale : power.r,
        value.g <= EncodedThreshold ? value.g / LinearScale : power.g,
        value.b <= EncodedThreshold ? value.b / LinearScale : power.b);
}
float LumaAt(int2 pixel, int2 extent)
{
    const float3 Rec709Luma = float3(0.2126, 0.7152, 0.0722);
    return dot(DisplayEncode(SceneColor.Load(int3(clamp(pixel, 0, extent - 1), 0)).rgb), Rec709Luma);
}
float Contour(int2 pixel, int2 extent)
{
    const float GaussianScaleRatio = 1.6; // Conventional DoG approximation to a Laplacian.
    const float GaussianSupportSigma = 3.0;
    const int MaximumKernelRadius = 8; // ceil(3 * 1.6 * maximum GUI sigma 1.5).
    float2 sigma = NprEdge.x * float2(1, GaussianScaleRatio);
    int radius = min(MaximumKernelRadius, int(ceil(GaussianSupportSigma * sigma.y)));
    float2 weightedLuma = 0, normalization = 0;
    [loop] for (int y = -radius; y <= radius; ++y)
        [loop] for (int x = -radius; x <= radius; ++x)
        {
            float2 weight = exp(-float(x * x + y * y) / (2 * sigma * sigma));
            weightedLuma += weight * LumaAt(pixel + int2(x, y), extent);
            normalization += weight;
        }
    float2 blurred = weightedLuma / normalization;
    float response = blurred.x - NprEdge.y * blurred.y;
    // 1 + tanh(phi * (D - epsilon)), evaluated without overflowing exp.
    float exponential = exp(2 * min(0, NprEdge.w * (response - NprEdge.z)));
    return 2 * exponential / (1 + exponential);
}
float HatchLine(float2 pixel, float2 normal)
{
    float distance = abs(frac(dot(pixel, normal) / NprHatch.x + 0.5) - 0.5) * NprHatch.x;
    float halfWidth = NprHatch.y * 0.5;
    float footprint = (abs(normal.x) + abs(normal.y)) * 0.5;
    return 1 - smoothstep(max(0, halfWidth - footprint), halfWidth + footprint, distance);
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    uint width, height; SceneColor.GetDimensions(width, height);
    int2 pixel = int2(input.Position.xy), extent = int2(width, height);
    float4 source = SceneColor.Load(int3(pixel, 0));
    float3 display = saturate(DisplayEncode(source.rgb));
    float edge = NprStyle.x > 0 ? Contour(pixel, extent) : 1;
    const float3 Rec709Luma = float3(0.2126, 0.7152, 0.0722);
    const float ToneTransition = 0.1, LightHatchThreshold = 0.25;
    const float MiddleHatchThreshold = 0.5, DarkHatchThreshold = 0.75;
    const float DiagonalNormalization = 0.7071067811865475;
    float darkness = 1 - dot(display, Rec709Luma);
    float3 activation = smoothstep(float3(LightHatchThreshold, MiddleHatchThreshold, DarkHatchThreshold) - ToneTransition,
        float3(LightHatchThreshold, MiddleHatchThreshold, DarkHatchThreshold) + ToneTransition, darkness);
    float2 normal = NprHatch.zw;
    float3 strokes = float3(HatchLine(input.Position.xy, normal),
        HatchLine(input.Position.xy, float2(-normal.y, normal.x)),
        HatchLine(input.Position.xy, float2(normal.x - normal.y, normal.x + normal.y) * DiagonalNormalization));
    float3 transmission = 1 - strokes * activation * NprStyle.y;
    float levels = NprTone.x - 1;
    float3 posterized = floor(display * levels + 0.5) / levels;
    float3 paper = lerp(float3(1, 1, 1), posterized, NprStyle.z);
    float3 inked = paper * lerp(1, edge, NprStyle.x) * transmission.x * transmission.y * transmission.z;
    // Preserve coverage; native sRGB RTV encodes the decoded display result.
    return float4(DisplayDecode(lerp(display, inked, NprStyle.w)), source.a);
}
