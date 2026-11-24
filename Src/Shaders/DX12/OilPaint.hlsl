// Four-quadrant Kuwahara filtering retains low-variance color regions.
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
    int radius = clamp(int(Painterly.x + 0.5f), 1, 4);
    float bestVariance = 1.0e20f;
    float3 bestColor = center.rgb;
    [loop] for (int quadrant = 0; quadrant < 4; ++quadrant)
    {
        int xStart = (quadrant & 1) != 0 ? 0 : -radius;
        int yStart = (quadrant & 2) != 0 ? 0 : -radius;
        float3 sum = 0, sumSquared = 0;
        [loop] for (int y = 0; y <= radius; ++y)
        [loop] for (int x = 0; x <= radius; ++x)
        {
            float3 value = At(pixel + int2(xStart + x, yStart + y), extent);
            sum += value;
            sumSquared += value * value;
        }
        float count = float((radius + 1) * (radius + 1));
        float3 mean = sum / count;
        float variance = dot(max(sumSquared / count - mean * mean, 0), float3(0.2126f, 0.7152f, 0.0722f));
        if (variance < bestVariance) { bestVariance = variance; bestColor = mean; }
    }
    return float4(lerp(center.rgb, bestColor, Painterly.y), center.a);
}
