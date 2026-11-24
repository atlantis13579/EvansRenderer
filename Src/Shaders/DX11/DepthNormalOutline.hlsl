// Fullscreen depth and world-normal discontinuities form geometric contours.
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
    float4 DepthNormalEdge;
};
Texture2D<float4> SceneColor : register(t0);
Texture2D<float> SceneDepth : register(t1);
Texture2D<float4> SceneNormal : register(t2);
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint id : SV_VertexID)
{
    float2 corner = float2((id << 1u) & 2u, id & 2u);
    ScreenVertex output;
    output.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    uint colorWidth, colorHeight, width, height;
    SceneColor.GetDimensions(colorWidth, colorHeight);
    SceneDepth.GetDimensions(width, height);
    int2 extent = int2(width, height);
    int2 pixel = clamp(int2(input.Position.xy * float2(width, height) / float2(colorWidth, colorHeight)), int2(0, 0), extent - 1);
    float4 color = SceneColor.Load(int3(input.Position.xy, 0));
    float depth = SceneDepth.Load(int3(pixel, 0));
    float3 normal = normalize(SceneNormal.Load(int3(pixel, 0)).xyz + 1.0e-6f);
    float depthEdge = 0, normalEdge = 0;
    const int2 offsets[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    [unroll] for (int i = 0; i < 4; ++i)
    {
        int2 neighbor = clamp(pixel + offsets[i], int2(0, 0), extent - 1);
        float nextDepth = SceneDepth.Load(int3(neighbor, 0));
        depthEdge = max(depthEdge, abs(nextDepth - depth) / max(1.0f - max(nextDepth, depth), 0.001f));
        float3 nextNormal = normalize(SceneNormal.Load(int3(neighbor, 0)).xyz + 1.0e-6f);
        if (depth < 0.999999f && nextDepth < 0.999999f)
            normalEdge = max(normalEdge, 1.0f - dot(normal, nextNormal));
    }
    float edge = max(smoothstep(DepthNormalEdge.x, DepthNormalEdge.x * 2.0f, depthEdge),
        smoothstep(DepthNormalEdge.y, DepthNormalEdge.y + 0.1f, normalEdge));
    return float4(color.rgb * (1.0f - edge * DepthNormalEdge.z), color.a);
}
