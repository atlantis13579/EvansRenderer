// Native DX12 SSGI. Visible-layer diffuse transport, spatial rejection,
// motion-reprojected private history, then full HDR scene composition.
// Ray intervals: McGuire/Mara, JCGT 2014, https://jcgt.org/published/0003/04/04/
cbuffer DiffuseTransport : register(b0)
{
    row_major float4x4 WorldToClip, ClipToWorld, PreviousClipToWorld;
    float4 PreviousOpticalPlane, ImageExtentInverse, RayDistanceThicknessBiasStride;
    float4 IntensityEdgeDepthNormal, HistoryWeightDepthNormal, RasterJitterChange;
    uint4 StageRaysStepsSequence, HistoryValidityModel;
};
Texture2D<float4> OpaqueRadiance : register(t0);
Texture2D<float> CameraDepth : register(t1);
Texture2D<float4> SurfaceNormalRoughness : register(t2);
Texture2D<float4> SurfaceDiffuse : register(t3);
Texture2D<float4> SurfaceAmbientMetallic : register(t4);
Texture2D<float4> GeometryMotion : register(t5);
Texture2D<float4> CurrentBounce : register(t6);
Texture2D<float4> PreviousBounce : register(t7);
Texture2D<float> PreviousDepth : register(t8);
Texture2D<float4> PreviousNormal : register(t9);
SamplerState RadianceSampler : register(s0);
static const uint TraceStage = 0, SpatialStage = 1, CompositeStage = 3, MetallicRoughnessShading = 3;
static const uint ClipPlaneCount = 7;
static const float TexelCenter = 0.5f, BackgroundDepth = 1, InteriorCameraDepth = 0.5f;
static const float MinimumHomogeneousW = 1e-7f, MinimumVectorLengthSquared = 1e-12f, DepthIntersectionTolerance = 1e-7f;
static const float ExcludedOriginPixels = 2, RayFadeStart = 0.8f, DielectricF0 = 0.04f, HalfFloatMaximum = 65504;
static const float Pi = 3.14159265358979323846f, BasisParallelLimit = 0.999f;
static const float AbsoluteHistoryDepthError = 0.001f, ColorBoxMinimumRadius = 0.001f, HistorySampleLimit = 32;
static const int SpatialRadius = 1;
// PCG permutation constants define RXS-M-XS, rather than artistic tuning.
// Jarzynski/Olano, Hash Functions for GPU Rendering, JCGT 2020.
// https://jcgt.org/published/0009/03/02/
static const uint GeneratorMultiplier = 747796405u, GeneratorOffset = 2891336453u, OutputMultiplier = 277803737u;
static const uint RotationBits = 28, MinimumRotation = 4, OutputRotation = 22, LowBitsToDiscard = 8;
static const float UniformScale = 1.0f / 16777216.0f; // 24 exactly representable random bits
uint Permute(uint seed)
{
    uint state = seed * GeneratorMultiplier + GeneratorOffset;
    uint permuted = ((state >> ((state >> RotationBits) + MinimumRotation)) ^ state) * OutputMultiplier;
    return (permuted >> OutputRotation) ^ permuted;
}
float NextUniform(inout uint seed)
{
    seed = Permute(seed);
    return float(seed >> LowBitsToDiscard) * UniformScale;
}
struct FullscreenVertex { float4 Position : SV_POSITION; };
FullscreenVertex VSMain(uint id : SV_VertexID)
{
    FullscreenVertex result;
    float2 corner = float2((id << 1u) & 2u, id & 2u);
    result.Position = float4(corner.x * 2.0f - 1.0f, 1.0f - corner.y * 2.0f, 0.0f, 1.0f);
    return result;
}

bool NormalizeDirection(inout float3 direction)
{
    float squaredLength = dot(direction, direction);
    if (!isfinite(squaredLength) || squaredLength <= MinimumVectorLengthSquared) return false;
    direction *= rsqrt(squaredLength);
    return true;
}
bool WorldPosition(float2 texel, float depth, out float3 position)
{
    float2 uv = texel * ImageExtentInverse.zw;
    float4 homogeneous = mul(ClipToWorld, float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f));
    position = 0.0f;
    if (abs(homogeneous.w) < MinimumHomogeneousW) return false;
    position = homogeneous.xyz / homogeneous.w;
    return all(isfinite(position));
}
struct VisibleSurface
{
    float3 Position;
    float3 Normal;
    float3 CameraRay;
    float Roughness;
};
bool LoadSurface(int2 texel, out VisibleSurface surface)
{
    surface.Position = surface.Normal = surface.CameraRay = 0.0f;
    surface.Roughness = 0.0f;
    if (any(texel < 0) || any(texel >= int2(ImageExtentInverse.xy))) return false;
    float depth = CameraDepth.Load(int3(texel, 0));
    float4 normalRoughness = SurfaceNormalRoughness.Load(int3(texel, 0));
    if (depth >= BackgroundDepth || normalRoughness.w < 0.0f) return false;
    surface.Normal = normalRoughness.xyz;
    surface.Roughness = normalRoughness.w;
    float2 center = float2(texel) + TexelCenter;
    float3 nearPosition, interiorPosition;
    if (!NormalizeDirection(surface.Normal) || !WorldPosition(center, depth, surface.Position) ||
        !WorldPosition(center, 0.0f, nearPosition) || !WorldPosition(center, InteriorCameraDepth, interiorPosition)) return false;
    surface.CameraRay = interiorPosition - nearPosition;
    return NormalizeDirection(surface.CameraRay);
}

struct ProjectedRay
{
    float2 PixelStart;
    float2 PixelDelta;
    float3 WeightedStart;
    float3 WeightedDelta;
    float ReciprocalStart;
    float ReciprocalDelta;
    float PixelLength;
};
bool ProjectSegment(float3 origin, float3 direction, out ProjectedRay ray)
{
    ray = (ProjectedRay)0;
    float3 endpoint = origin + direction * RayDistanceThicknessBiasStride.x;
    float4 startClip = mul(WorldToClip, float4(origin, 1.0f));
    float4 endClip = mul(WorldToClip, float4(endpoint, 1.0f));
    if (!all(isfinite(startClip)) || !all(isfinite(endClip))) return false;
    // Homogeneous half-space clipping retains a positive denominator and bounds
    // texture accesses even when the reflected ray points toward the near plane.
    float startPlanes[ClipPlaneCount] = {startClip.w - MinimumHomogeneousW,
        startClip.w + startClip.x, startClip.w - startClip.x, startClip.w + startClip.y,
        startClip.w - startClip.y, startClip.z, startClip.w - startClip.z};
    float endPlanes[ClipPlaneCount] = {endClip.w - MinimumHomogeneousW,
        endClip.w + endClip.x, endClip.w - endClip.x, endClip.w + endClip.y,
        endClip.w - endClip.y, endClip.z, endClip.w - endClip.z};
    float entry = 0.0f, exit = 1.0f;
    [unroll] for (uint plane = 0; plane < ClipPlaneCount; ++plane)
    {
        float a = startPlanes[plane], b = endPlanes[plane];
        if (a < 0.0f && b < 0.0f) return false;
        if ((a < 0.0f) != (b < 0.0f))
        {
            float intersection = a / (a - b);
            if (a < 0.0f) entry = max(entry, intersection);
            else exit = min(exit, intersection);
        }
    }
    if (entry >= exit) return false;
    float3 displacement = endpoint - origin;
    float3 clippedStart = origin + displacement * entry;
    float3 clippedEnd = origin + displacement * exit;
    startClip = mul(WorldToClip, float4(clippedStart, 1.0f));
    endClip = mul(WorldToClip, float4(clippedEnd, 1.0f));
    ray.ReciprocalStart = rcp(startClip.w);
    float reciprocalEnd = rcp(endClip.w);
    ray.ReciprocalDelta = reciprocalEnd - ray.ReciprocalStart;
    ray.WeightedStart = clippedStart * ray.ReciprocalStart;
    ray.WeightedDelta = clippedEnd * reciprocalEnd - ray.WeightedStart;
    ray.PixelStart = (startClip.xy * ray.ReciprocalStart * float2(0.5f, -0.5f) + TexelCenter) * ImageExtentInverse.xy;
    float2 pixelEnd = (endClip.xy * reciprocalEnd * float2(0.5f, -0.5f) + TexelCenter) * ImageExtentInverse.xy;
    ray.PixelDelta = pixelEnd - ray.PixelStart;
    ray.PixelLength = max(abs(ray.PixelDelta.x), abs(ray.PixelDelta.y));
    return ray.PixelLength > ExcludedOriginPixels;
}

bool IntersectInterval(ProjectedRay ray, float begin, float end, float3 direction,
    out float2 hitPixel, out float3 hitPosition)
{
    hitPixel = 0.0f; hitPosition = 0.0f;
    float middle = (begin + end) * 0.5f;
    VisibleSurface candidate;
    if (!LoadSurface(int2(floor(ray.PixelStart + ray.PixelDelta * middle)), candidate) ||
        dot(candidate.Normal, -direction) <= 0.0f) return false;
    float depthStartNumerator = dot(ray.WeightedStart - candidate.Position * ray.ReciprocalStart, candidate.CameraRay);
    float depthNumeratorDelta = dot(ray.WeightedDelta - candidate.Position * ray.ReciprocalDelta, candidate.CameraRay);
    float depthBegin = (depthStartNumerator + depthNumeratorDelta * begin) / (ray.ReciprocalStart + ray.ReciprocalDelta * begin);
    float depthEnd = (depthStartNumerator + depthNumeratorDelta * end) / (ray.ReciprocalStart + ray.ReciprocalDelta * end);
    if (max(depthBegin, depthEnd) < 0.0f || min(depthBegin, depthEnd) > RayDistanceThicknessBiasStride.y) return false;
    // Solve the slab front-plane crossing in projective parameter space. Check
    // its actual texel again to reject false intersections at depth discontinuities.
    float crossing = abs(depthNumeratorDelta) > DepthIntersectionTolerance ?
        -depthStartNumerator / depthNumeratorDelta : middle;
    crossing = clamp(crossing, begin, end);
    hitPixel = ray.PixelStart + ray.PixelDelta * crossing;
    VisibleSurface refined;
    if (!LoadSurface(int2(floor(hitPixel)), refined) || dot(refined.Normal, -direction) <= 0.0f) return false;
    hitPosition = (ray.WeightedStart + ray.WeightedDelta * crossing) / (ray.ReciprocalStart + ray.ReciprocalDelta * crossing);
    float depthSeparation = dot(hitPosition - refined.Position, refined.CameraRay);
    return depthSeparation >= -DepthIntersectionTolerance && depthSeparation <= RayDistanceThicknessBiasStride.y;
}


float3 IncomingRay(float3 origin, float3 direction)
{
    ProjectedRay ray;
    if (!ProjectSegment(origin, direction, ray)) return 0;
    float increment = RayDistanceThicknessBiasStride.w / ray.PixelLength;
    float initial = ExcludedOriginPixels / ray.PixelLength;
    [loop] for (uint step = 0; step < StageRaysStepsSequence.z; ++step)
    {
        float first = initial + step * increment;
        if (first >= 1) break;
        float2 hit; float3 position;
        if (!IntersectInterval(ray, first, min(first + increment, 1), direction, hit, position)) continue;
        float2 edgeDistance = min(hit, ImageExtentInverse.xy - hit);
        float edge = IntensityEdgeDepthNormal.y > 0 ? saturate(min(edgeDistance.x, edgeDistance.y) / IntensityEdgeDepthNormal.y) : 1;
        float fade = 1 - smoothstep(RayDistanceThicknessBiasStride.x * RayFadeStart,
            RayDistanceThicknessBiasStride.x, length(position - origin));
        int2 texel = clamp(int2(hit), 0, int2(ImageExtentInverse.xy) - 1);
        return max(OpaqueRadiance.Load(int3(texel, 0)).rgb, 0) * (edge * fade);
    }
    return 0;
}
float4 TraceBounce(int2 texel)
{
    VisibleSurface surface;
    if (!LoadSurface(texel, surface)) return 0;
    float3 albedo = saturate(SurfaceDiffuse.Load(int3(texel, 0)).rgb);
    if (HistoryValidityModel.y == MetallicRoughnessShading)
        albedo *= (1 - saturate(SurfaceAmbientMetallic.Load(int3(texel, 0)).w)) * (1 - DielectricF0);
    float3 guide = abs(surface.Normal.z) < BasisParallelLimit ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = normalize(cross(guide, surface.Normal)), binormal = cross(surface.Normal, tangent);
    uint seed = (uint(texel.y) * uint(ImageExtentInverse.x) + uint(texel.x)) ^ Permute(StageRaysStepsSequence.w);
    float3 sum = 0;
    [loop] for (uint rayIndex = 0; rayIndex < StageRaysStepsSequence.y; ++rayIndex)
    {
        float radial = NextUniform(seed), azimuth = 2 * Pi * NextUniform(seed);
        float radius = sqrt(radial);
        float3 direction = tangent * (radius * cos(azimuth)) + binormal * (radius * sin(azimuth)) + surface.Normal * sqrt(1 - radial);
        sum += IncomingRay(surface.Position + surface.Normal * RayDistanceThicknessBiasStride.z, direction);
    }
    // Lambertian albedo/pi times cosine divided by the cosine/pi PDF.
    return float4(clamp(albedo * sum / float(StageRaysStepsSequence.y), 0, HalfFloatMaximum), 1);
}
float NeighborWeight(int2 texel, int2 candidate, VisibleSurface center)
{
    VisibleSurface other;
    if (!LoadSurface(candidate, other) || dot(center.Normal, other.Normal) < IntensityEdgeDepthNormal.w) return 0;
    float separation = abs(dot(other.Position - center.Position, center.Normal));
    float gaussian = (texel.x == candidate.x ? 2 : 1) * (texel.y == candidate.y ? 2 : 1);
    return gaussian * exp(-separation / IntensityEdgeDepthNormal.z);
}
float4 SpatialBounce(int2 texel)
{
    VisibleSurface surface;
    if (!LoadSurface(texel, surface)) return 0;
    float total = 0; float3 sum = 0;
    [unroll] for (int y = -SpatialRadius; y <= SpatialRadius; ++y) [unroll] for (int x = -SpatialRadius; x <= SpatialRadius; ++x)
    {
        int2 candidate = texel + int2(x, y); float weight = NeighborWeight(texel, candidate, surface);
        if (weight <= 0) continue;
        sum += max(CurrentBounce.Load(int3(candidate, 0)).rgb, 0) * weight; total += weight;
    }
    return float4(sum / max(total, MinimumVectorLengthSquared), 1);
}
float HistoryDistance(float2 pixel, float depth)
{
    float2 uv = pixel * ImageExtentInverse.zw;
    float4 world = mul(PreviousClipToWorld, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    if (abs(world.w) <= MinimumHomogeneousW) return -1;
    return dot(PreviousOpticalPlane, float4(world.xyz / world.w, 1));
}
struct TransportHistory { float4 Bounce : SV_TARGET0; float Depth : SV_TARGET1; float4 Normal : SV_TARGET2; };
TransportHistory PSTemporal(FullscreenVertex input)
{
    int2 texel = int2(input.Position.xy);
    TransportHistory result;
    result.Bounce = CurrentBounce.Load(int3(texel, 0)); result.Depth = CameraDepth.Load(int3(texel, 0));
    result.Normal = SurfaceNormalRoughness.Load(int3(texel, 0));
    VisibleSurface surface;
    if (HistoryValidityModel.x == 0 || HistoryWeightDepthNormal.x <= 0 || !LoadSurface(texel, surface)) return result;
    float4 motion = GeometryMotion.Load(int3(texel, 0));
    if (motion.w <= 0) return result;
    float2 previousPixel = input.Position.xy - motion.xy - RasterJitterChange.xy;
    if (any(previousPixel < TexelCenter) || any(previousPixel > ImageExtentInverse.xy - TexelCenter)) return result;
    float expected = HistoryDistance(previousPixel, 1-motion.z);
    if (!isfinite(expected) || expected < 0) return result;
    float tolerance = max(AbsoluteHistoryDepthError, abs(expected) * HistoryWeightDepthNormal.y);
    int2 first = int2(floor(previousPixel - TexelCenter)); float2 blend = frac(previousPixel - TexelCenter);
    float4 sum = 0; float total = 0;
    [unroll] for (int y = 0; y < 2; ++y) [unroll] for (int x = 0; x < 2; ++x)
    {
        int2 candidate = first + int2(x, y);
        if (any(candidate < 0) || any(candidate >= int2(ImageExtentInverse.xy))) continue;
        float depth = PreviousDepth.Load(int3(candidate, 0)); float3 normal = PreviousNormal.Load(int3(candidate, 0)).xyz;
        float4 old = PreviousBounce.Load(int3(candidate, 0));
        float distance = HistoryDistance(float2(candidate) + TexelCenter, depth);
        if (depth >= BackgroundDepth || !NormalizeDirection(normal) || dot(normal, surface.Normal) < HistoryWeightDepthNormal.z ||
            old.a <= 0 || !isfinite(distance) || abs(distance - expected) > tolerance) continue;
        float weight = (x == 0 ? 1 - blend.x : blend.x) * (y == 0 ? 1 - blend.y : blend.y);
        sum += old * weight; total += weight;
    }
    if (total <= MinimumVectorLengthSquared) return result;
    float4 old = sum / total;
    float3 low = result.Bounce.rgb, high = low;
    [unroll] for (int y = -SpatialRadius; y <= SpatialRadius; ++y) [unroll] for (int x = -SpatialRadius; x <= SpatialRadius; ++x)
    {
        int2 candidate = texel + int2(x, y);
        if (NeighborWeight(texel, candidate, surface) <= 0) continue;
        float3 neighborValue = CurrentBounce.Load(int3(candidate, 0)).rgb;
        low = min(low, neighborValue); high = max(high, neighborValue);
    }
    float3 midpoint = (low + high) * 0.5f, radius = max((high - low) * 0.5f, ColorBoxMinimumRadius);
    float3 difference = old.rgb - midpoint, ratio = abs(difference / radius);
    old.rgb = midpoint + difference / max(1, max(ratio.x, max(ratio.y, ratio.z)));
    float feedback = min(HistoryWeightDepthNormal.x, old.a / (old.a + 1));
    result.Bounce = float4(clamp(lerp(result.Bounce.rgb, old.rgb, feedback), 0, HalfFloatMaximum), min(old.a + 1, HistorySampleLimit));
    return result;
}
float4 PSMain(FullscreenVertex input) : SV_TARGET
{
    int2 texel = int2(input.Position.xy);
    if (StageRaysStepsSequence.x == TraceStage) return TraceBounce(texel);
    if (StageRaysStepsSequence.x == SpatialStage) return SpatialBounce(texel);
    float4 scene = OpaqueRadiance.Load(int3(texel, 0));
    float3 indirect = max(CurrentBounce.Load(int3(texel, 0)).rgb, 0) * IntensityEdgeDepthNormal.x;
    return float4(clamp(scene.rgb + indirect, 0, HalfFloatMaximum), scene.a);
}
