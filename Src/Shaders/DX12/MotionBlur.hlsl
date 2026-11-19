// DX12 centered-shutter reconstruction from independent motion vectors.
// Dominant tiles and soft depth classification: McGuire et al., "A Reconstruction
// Filter for Plausible Motion Blur" (I3D 2012).
// https://casual-effects.com/research/McGuire2012Blur/index.html
// This variant gathers oriented box-shutter footprints instead of the paper's
// cone/cylinder kernels. One depth layer cannot recover hidden moving surfaces.
cbuffer ShutterParameters : register(b0)
{
    row_major float4x4 ObjectToWorld;
    row_major float4x4 CameraTransform;
    row_major float4x4 PreviousObjectToClip;
    row_major float4x4 ClipToWorld;
    float4 OpticalDepthPlane;
    float4 ExtentExposureRadius;
    float4 OpacityCutoff;
    uint4 StageFlags;
};
Texture2D<float4> Radiance : register(t0);
Texture2D<float> VisibleDepth : register(t1);
Texture2D<float4> Displacement : register(t2);
Texture2D<float4> LocalMaximumOrDiffuse : register(t3);
SamplerState MaterialSampler : register(s0);

static const uint ReduceTileStage = 0u;
static const uint ExpandNeighborhoodStage = 1u;
static const uint CopyResultStage = 3u;
static const int TileExtent = 16;
static const int AdjacentTileRadius = 1;
static const int SampleLimit = 33;
static const float PixelHalfWidth = 0.5f;
static const float PixelWidth = 1.0f;
static const float MinimumBlurRadius = 0.5f;
static const float HomogeneousEpsilon = 1.0e-7f;
static const float HalfFloatMaximum = 65504.0f;
static const float FarDepth = 1.0f;
static const float UncoveredDistance = 1.0e20f;
static const float RelativeDepthTolerance = 0.02f;
static const float AbsoluteDepthTolerance = 1.0e-4f;

struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    ScreenVertex result;
    float2 corner = float2((vertexId << 1u) & 2u, vertexId & 2u);
    result.Position = float4(corner.x * 2 - 1, 1 - corner.y * 2, 0, 1);
    return result;
}
float2 ShutterTravel(int2 pixel)
{
    float4 sample = Displacement.Load(int3(pixel, 0));
    if (sample.w <= 0 || ExtentExposureRadius.z <= 0) return 0;
    float2 halfTravel = sample.xy * ExtentExposureRadius.z;
    float distance = length(halfTravel);
    if (distance > ExtentExposureRadius.w) halfTravel *= ExtentExposureRadius.w / distance;
    return halfTravel;
}
float EyeDistance(int2 pixel)
{
    float depth = VisibleDepth.Load(int3(pixel, 0));
    if (depth >= FarDepth) return UncoveredDistance;
    float2 uv = (float2(pixel) + PixelHalfWidth) / ExtentExposureRadius.xy;
    float4 position = mul(ClipToWorld, float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1));
    if (abs(position.w) <= HomogeneousEpsilon) return UncoveredDistance;
    return max(dot(OpticalDepthPlane, float4(position.xyz / position.w, 1)), 0);
}
float InFront(float candidate, float reference)
{
    float tolerance = max(min(candidate, reference) * RelativeDepthTolerance, AbsoluteDepthTolerance);
    return saturate(1 - (candidate - reference) / tolerance);
}
float LineCoverage(float2 offset, float2 travel)
{
    float radius = length(travel);
    if (radius <= MinimumBlurRadius) return 0;
    float2 direction = travel / radius;
    float parallelDistance = abs(dot(offset, direction));
    float perpendicularDistance = abs(offset.x * direction.y - offset.y * direction.x);
    // Tent coverage across the line prevents crossing motion from contributing
    // outside its own swept pixel. Along the line the shutter is a box interval.
    return saturate(radius + PixelHalfWidth - parallelDistance) *
        saturate(PixelWidth - perpendicularDistance) / max(2 * radius, PixelWidth);
}
float4 PSMain(ScreenVertex vertex) : SV_TARGET
{
    int2 pixel = int2(vertex.Position.xy);
    int2 dimensions = int2(ExtentExposureRadius.xy);
    if (StageFlags.x == ReduceTileStage)
    {
        float2 maximumTravel = 0;
        for (int row = 0; row < TileExtent; ++row)
        for (int column = 0; column < TileExtent; ++column)
        {
            int2 source = pixel * TileExtent + int2(column, row);
            if (any(source >= dimensions)) continue;
            float2 candidate = ShutterTravel(source);
            if (dot(candidate, candidate) > dot(maximumTravel, maximumTravel)) maximumTravel = candidate;
        }
        return float4(maximumTravel, 0, 0);
    }
    if (StageFlags.x == ExpandNeighborhoodStage)
    {
        int2 tileCount = (dimensions + TileExtent - 1) / TileExtent;
        float2 maximumTravel = 0;
        for (int row = -AdjacentTileRadius; row <= AdjacentTileRadius; ++row)
        for (int column = -AdjacentTileRadius; column <= AdjacentTileRadius; ++column)
        {
            int2 source = clamp(pixel + int2(column, row), int2(0, 0), tileCount - 1);
            float2 candidate = LocalMaximumOrDiffuse.Load(int3(source, 0)).xy;
            if (dot(candidate, candidate) > dot(maximumTravel, maximumTravel)) maximumTravel = candidate;
        }
        return float4(maximumTravel, 0, 0);
    }
    float4 original = Radiance.Load(int3(pixel, 0));
    if (StageFlags.x == CopyResultStage) return original;
    float2 axis = LocalMaximumOrDiffuse.Load(int3(pixel / TileExtent, 0)).xy;
    float support = length(axis);
    if (support <= MinimumBlurRadius) return original;
    float2 ownTravel = ShutterTravel(pixel);
    float ownDepth = EyeDistance(pixel);
    int steps = min((SampleLimit - 1) / 2, max(1, int(ceil(support))));
    float sampleWidth = support / float(steps);
    float sumWeight = 2 / max(2 * length(ownTravel), PixelWidth);
    float3 sumColor = original.rgb * sumWeight;
    for (int step = -steps; step <= steps; ++step)
    {
        if (step == 0) continue;
        int2 source = int2(round(float2(pixel) + axis * (float(step) / float(steps))));
        if (any(source < 0) || any(source >= dimensions) || all(source == pixel)) continue;
        float2 delta = float2(source - pixel);
        float sourceDepth = EyeDistance(source);
        float weight = sampleWidth * (InFront(sourceDepth, ownDepth) * LineCoverage(-delta, ShutterTravel(source)) +
            InFront(ownDepth, sourceDepth) * LineCoverage(delta, ownTravel));
        sumColor += Radiance.Load(int3(source, 0)).rgb * weight;
        sumWeight += weight;
    }
    return float4(sumColor / sumWeight, original.a);
}
