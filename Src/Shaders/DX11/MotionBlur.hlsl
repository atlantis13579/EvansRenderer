// DX11 centered-shutter reconstruction from independent motion vectors.
// McGuire et al., "A Reconstruction Filter for Plausible Motion Blur" (I3D 2012)
// motivates tile/neighbor maxima and depth-aware foreground/background gathering.
// This spatial variant uses oriented line footprints for a box shutter, keeps
// alpha/depth unchanged, and has no color history. Hidden surfaces remain absent.
// https://casual-effects.com/research/McGuire2012Blur/index.html
cbuffer MotionConstants : register(b0)
{
    row_major float4x4 World;
    row_major float4x4 ViewProjection;
    row_major float4x4 PreviousWorldViewProjection;
    row_major float4x4 InverseViewProjection;
    float4 DepthPlane;
    float4 ViewportExposure;
    float4 Material;
    uint4 Options;
};
Texture2D<float4> SourceColor : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> MotionVectors : register(t2);
Texture2D<float4> Auxiliary : register(t3); // tile maxima
SamplerState SurfaceSampler : register(s0);

static const uint TileMaxPass = 0u;
static const uint NeighborMaxPass = 1u;
static const uint CompositePass = 3u;
static const int TileSize = 16;
static const int NeighborRadius = 1;
static const int MaximumSamples = 33;
static const float ClipWEpsilon = 1.0e-7f;
static const float MaximumHalfFloat = 65504.0f;
static const float PixelCenter = 0.5f;
static const float SharpRadius = 0.5f;
static const float MinimumFootprintWidth = 1.0f;
static const float DepthSoftnessFraction = 0.02f;
static const float MinimumDepthSoftness = 1.0e-4f;
static const float BackgroundDistance = 1.0e20f;
static const float ClearDepth = 1.0f;

struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint index : SV_VertexID)
{
    float2 corner = float2((index << 1u) & 2u, index & 2u);
    ScreenVertex output;
    output.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0, 1);
    return output;
}
float2 HalfMotion(int2 pixel)
{
    float4 motion = MotionVectors.Load(int3(pixel, 0));
    if (motion.w <= 0.0f || ViewportExposure.z <= 0.0f) return 0.0f;
    float2 value = motion.xy * ViewportExposure.z;
    float radius = length(value);
    return radius > ViewportExposure.w ? value * (ViewportExposure.w / radius) : value;
}
float LinearDepth(int2 pixel)
{
    float depth = OpaqueDepth.Load(int3(pixel, 0));
    if (depth >= ClearDepth) return BackgroundDistance;
    float2 uv = (float2(pixel) + PixelCenter) / ViewportExposure.xy;
    float4 world = mul(InverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    if (abs(world.w) <= ClipWEpsilon) return BackgroundDistance;
    return max(dot(DepthPlane, float4(world.xyz / world.w, 1)), 0.0f);
}
float SoftForeground(float nearDepth, float farDepth)
{
    float softness = max(min(nearDepth, farDepth) * DepthSoftnessFraction, MinimumDepthSoftness);
    return saturate(1.0f - (nearDepth - farDepth) / softness);
}
float Footprint(float2 offset, float2 halfMotion)
{
    float radius = length(halfMotion);
    if (radius <= SharpRadius) return 0.0f;
    float2 axis = halfMotion / radius;
    float along = abs(dot(offset, axis));
    float across = abs(offset.x * axis.y - offset.y * axis.x);
    // The transverse tent models a pixel footprint; the longitudinal interval
    // is the centered box shutter. Direction rejects unrelated crossing motion.
    float coverage = saturate(radius + PixelCenter - along) * saturate(MinimumFootprintWidth - across);
    return coverage / max(2.0f * radius, MinimumFootprintWidth);
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy);
    int2 extent = int2(ViewportExposure.xy);
    if (Options.x == TileMaxPass)
    {
        float2 longest = 0.0f;
        int2 origin = pixel * TileSize;
        for (int y = 0; y < TileSize; ++y)
        for (int x = 0; x < TileSize; ++x)
        {
            int2 location = origin + int2(x, y);
            if (any(location >= extent)) continue;
            float2 candidate = HalfMotion(location);
            if (dot(candidate, candidate) > dot(longest, longest)) longest = candidate;
        }
        return float4(longest, 0, 0);
    }
    if (Options.x == NeighborMaxPass)
    {
        int2 tiles = (extent + TileSize - 1) / TileSize;
        float2 longest = 0.0f;
        for (int y = -NeighborRadius; y <= NeighborRadius; ++y)
        for (int x = -NeighborRadius; x <= NeighborRadius; ++x)
        {
            int2 tile = clamp(pixel + int2(x, y), int2(0, 0), tiles - 1);
            float2 candidate = Auxiliary.Load(int3(tile, 0)).xy;
            if (dot(candidate, candidate) > dot(longest, longest)) longest = candidate;
        }
        return float4(longest, 0, 0);
    }
    float4 center = SourceColor.Load(int3(pixel, 0));
    if (Options.x == CompositePass) return center;
    float2 dominant = Auxiliary.Load(int3(pixel / TileSize, 0)).xy;
    float radius = length(dominant);
    if (radius <= SharpRadius) return center;
    float2 centerMotion = HalfMotion(pixel);
    float centerDepth = LinearDepth(pixel);
    // An odd sample count retains the exact center and bounds sample spacing.
    int halfSamples = min((MaximumSamples - 1) / 2, max(1, int(ceil(radius))));
    float spacing = radius / float(halfSamples);
    float totalWeight = 2.0f / max(2.0f * length(centerMotion), MinimumFootprintWidth);
    float3 radiance = center.rgb * totalWeight;
    for (int tap = -halfSamples; tap <= halfSamples; ++tap)
    {
        if (tap == 0) continue;
        int2 source = int2(round(float2(pixel) + dominant * (float(tap) / float(halfSamples))));
        if (any(source < 0) || any(source >= extent) || all(source == pixel)) continue;
        float2 offset = float2(source - pixel);
        float2 sourceMotion = HalfMotion(source);
        float sourceDepth = LinearDepth(source);
        float weight = spacing * (
            SoftForeground(sourceDepth, centerDepth) * Footprint(-offset, sourceMotion) +
            SoftForeground(centerDepth, sourceDepth) * Footprint(offset, centerMotion));
        radiance += SourceColor.Load(int3(source, 0)).rgb * weight;
        totalWeight += weight;
    }
    return float4(radiance / totalWeight, center.a);
}
