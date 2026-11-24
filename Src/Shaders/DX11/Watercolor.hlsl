// Bilateral pigment diffusion, tonal quantization and static paper granulation.
cbuffer PostConstants : register(b0)
{
    float4 InverseExtentStrengths;
    uint4 GrainSeedReserved;
    float4 DistortionStrengthRadiusWeights;
    float4 NprEdge;
    float4 NprStyle;
    float4 NprHatch;
    float4 NprTone;
    float4 DynamicWarp;
    float4 Painterly;
};
Texture2D<float4> SceneColor : register(t0);
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint id : SV_VertexID)
{
    float2 corner = float2((id << 1u) & 2u, id & 2u);
    ScreenVertex output;
    output.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}
float3 At(int2 pixel, int2 extent)
{
    return SceneColor.Load(int3(clamp(pixel, int2(0, 0), extent - 1), 0)).rgb;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    uint width, height;
    SceneColor.GetDimensions(width, height);
    int2 pixel = int2(input.Position.xy), extent = int2(width, height);
    float4 center = SceneColor.Load(int3(pixel, 0));
    int radius = clamp(int(Painterly.z + 0.5f), 1, 4);
    float sigma = max(float(radius) * 0.7f, 0.7f);
    float3 sum = 0;
    float weightSum = 0;
    [loop] for (int y = -radius; y <= radius; ++y)
    [loop] for (int x = -radius; x <= radius; ++x)
    {
        float3 sampleColor = At(pixel + int2(x, y), extent);
        float3 difference = sampleColor - center.rgb;
        float weight = exp(-float(x * x + y * y) / (2.0f * sigma * sigma) - dot(difference, difference) * 8.0f);
        sum += sampleColor * weight;
        weightSum += weight;
    }
    float3 pigment = sum / max(weightSum, 1.0e-6f);
    pigment = floor(pigment * 12.0f + 0.5f) / 12.0f;
    float paper = frac(sin(dot(float2(pixel), float2(12.9898f, 78.233f))) * 43758.5453f) - 0.5f;
    pigment = saturate(pigment + paper * 0.025f);
    return float4(lerp(center.rgb, pigment, Painterly.w), center.a);
}
