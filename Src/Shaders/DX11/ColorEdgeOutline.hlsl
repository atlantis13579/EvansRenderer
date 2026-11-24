// Sobel color gradients draw a screen-space contour while preserving alpha.
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
    float3 tl = At(pixel + int2(-1, -1), extent), tc = At(pixel + int2(0, -1), extent);
    float3 tr = At(pixel + int2(1, -1), extent), ml = At(pixel + int2(-1, 0), extent);
    float3 mr = At(pixel + int2(1, 0), extent), bl = At(pixel + int2(-1, 1), extent);
    float3 bc = At(pixel + int2(0, 1), extent), br = At(pixel + int2(1, 1), extent);
    float3 gx = tr + 2.0f * mr + br - tl - 2.0f * ml - bl;
    float3 gy = bl + 2.0f * bc + br - tl - 2.0f * tc - tr;
    float edge = length(float2(length(gx), length(gy))) * 0.25f;
    float ink = smoothstep(NprTone.z, NprTone.z + 0.1f, edge) * NprTone.w;
    return float4(center.rgb * (1.0f - ink), center.a);
}
