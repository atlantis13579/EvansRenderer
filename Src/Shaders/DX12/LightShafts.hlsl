// Independently authored DX12 image-space scattering from an occluded light disk.
// Kenny Mitchell, GPU Gems 3, Chapter 13 (2007): radial light-scattering integration.
// The 64-byte root-constant block describes one selected light and one camera.
cbuffer LightShaftBlock : register(b0)
{
    float2 LightCenter;
    float LightDepth;
    float DiskRadius;
    float3 LightRadiance;
    float ScatteringScale;
    float OpticalDecay;
    float OcclusionDepthBias;
    float2 PixelExtent;
    uint RadialSamples;
    uint PassIndex;
    uint VisibleSource;
    uint Reserved;
};
Texture2D<float4> OpaqueRadiance : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> OccludedDisk : register(t2);
static const uint DiskPass = 0;
static const float MaximumStoredRadiance = 65504.0;
struct RasterVertex { float4 Position : SV_Position; };

RasterVertex VSMain(uint vertexId : SV_VertexID)
{
    const float OuterTriangleCoordinate = 3.0;
    float2 position = float2(-1.0, -1.0);
    if (vertexId == 1) position.x = OuterTriangleCoordinate;
    if (vertexId == 2) position.y = OuterTriangleCoordinate;
    RasterVertex result; result.Position = float4(position, 0.0, 1.0); return result;
}
float DiskAt(int2 texel)
{
    if (any(texel < int2(0, 0)) || any(texel >= int2(PixelExtent))) return 0.0;
    return OccludedDisk.Load(int3(texel, 0)).x;
}
float ReconstructDisk(float2 uv)
{
    // Explicit zero-border bilinear reconstruction keeps an off-image light
    // from being replicated along the edge of the viewport.
    float2 coordinate = uv * PixelExtent - 0.5;
    int2 base = int2(floor(coordinate));
    float2 fraction = frac(coordinate);
    float4 weights = float4((1.0 - fraction.x) * (1.0 - fraction.y), fraction.x * (1.0 - fraction.y),
        (1.0 - fraction.x) * fraction.y, fraction.x * fraction.y);
    return dot(weights, float4(DiskAt(base), DiskAt(base + int2(1, 0)), DiskAt(base + int2(0, 1)), DiskAt(base + int2(1, 1))));
}
float4 PSMain(RasterVertex input) : SV_Target0
{
    int2 texel = int2(input.Position.xy);
    float2 uv = input.Position.xy / PixelExtent;
    if (PassIndex == DiskPass)
    {
        float disk = 0.0;
        if (VisibleSource != 0 && OpaqueDepth.Load(int3(texel, 0)) + OcclusionDepthBias >= LightDepth)
        {
            float2 imageHeightOffset = uv - LightCenter;
            imageHeightOffset.x *= PixelExtent.x / PixelExtent.y;
            float profile = saturate(1.0 - length(imageHeightOffset) / DiskRadius);
            disk = profile * profile;
        }
        return float4(disk, 0.0, 0.0, 1.0);
    }
    float4 opaque = OpaqueRadiance.Load(int3(texel, 0));
    if (VisibleSource == 0) return opaque;
    float integral = 0.0;
    float interval = 1.0 / float(RadialSamples);
    float2 towardSource = LightCenter - uv;
    [loop] for (uint sampleIndex = 0; sampleIndex < RadialSamples; ++sampleIndex)
    {
        float fraction = (float(sampleIndex) + 0.5) * interval;
        integral += ReconstructDisk(uv + towardSource * fraction) * exp(-OpticalDecay * fraction);
    }
    // Integrate over a normalized interval, rather than increasing exposure
    // with sample count. Surface alpha and depth retain their opaque values.
    float3 scattered = LightRadiance * (ScatteringScale * integral * interval);
    return float4(min(opaque.rgb + scattered, MaximumStoredRadiance), opaque.a);
}
