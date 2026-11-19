// DX12 screen-space reflection: independent HLSL implementation of projected
// ray traversal and HDR composition. Root constants and seven SRV tables are
// bound per draw; the composite reads only the reflection image at t6.
// McGuire and Mara, "Efficient GPU Screen-Space Ray Tracing", JCGT (2014).
// https://jcgt.org/published/0003/04/04/paper.pdf
// A visible depth slab approximates geometry. Roughness fades a mirror ray;
// this spatial estimator has no lobe convolution or temporal color history.
cbuffer ScreenReflectionConstants : register(b0)
{
    row_major float4x4 WorldToClip;
    row_major float4x4 ClipToWorld;
    float4 ImageExtentInverse;
    float4 RayDistanceThicknessBiasStride;
    float4 StrengthRoughnessEdgeFade;
    uint4 StageBudgetShading;
};
Texture2D<float4> OpaqueRadiance : register(t0);
Texture2D<float> CameraDepth : register(t1);
Texture2D<float4> SurfaceNormalRoughness : register(t2);
Texture2D<float4> SurfaceSpecular : register(t3);
Texture2D<float4> SurfaceDiffuse : register(t4);
Texture2D<float4> SurfaceAmbientMetallic : register(t5);
Texture2D<float4> TracedRadiance : register(t6);
SamplerState RadianceSampler : register(s0);

static const uint CompositeStage = 1u;
static const uint MetallicRoughnessShading = 3u;
static const uint ClipPlaneCount = 7u;
static const float TexelCenter = 0.5f;
static const float BackgroundDepth = 1.0f;
static const float InteriorCameraDepth = 0.5f;
static const float MinimumHomogeneousW = 1.0e-7f;
static const float MinimumVectorLengthSquared = 1.0e-12f;
static const float DepthIntersectionTolerance = 1.0e-7f;
static const float ExcludedOriginPixels = 2.0f;
static const float RayFadeStart = 0.8f;
static const float DielectricF0 = 0.04f;
static const float FresnelPower = 5.0f;
static const float HalfFloatMaximum = 65504.0f;
static const float HalfFloatMaximumSpacing = 32.0f;

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

float3 ReflectionEstimate(int2 texel)
{
    VisibleSurface source;
    if (StrengthRoughnessEdgeFade.x <= 0.0f || !LoadSurface(texel, source) ||
        source.Roughness >= StrengthRoughnessEdgeFade.y) return 0.0f;
    float facing = dot(source.Normal, -source.CameraRay);
    if (facing <= 0.0f) return 0.0f;
    float3 direction = reflect(source.CameraRay, source.Normal);
    float3 origin = source.Position + source.Normal * RayDistanceThicknessBiasStride.z;
    ProjectedRay ray;
    if (!ProjectSegment(origin, direction, ray)) return 0.0f;
    float increment = RayDistanceThicknessBiasStride.w / ray.PixelLength;
    float initial = ExcludedOriginPixels / ray.PixelLength;
    [loop] for (uint sampleIndex = 0; sampleIndex < StageBudgetShading.y; ++sampleIndex)
    {
        float begin = initial + float(sampleIndex) * increment;
        if (begin >= 1.0f) break;
        float2 hitPixel; float3 hitPosition;
        if (!IntersectInterval(ray, begin, min(begin + increment, 1.0f), direction, hitPixel, hitPosition)) continue;
        float3 f0 = saturate(SurfaceSpecular.Load(int3(texel, 0)).rgb);
        if (StageBudgetShading.z == MetallicRoughnessShading)
            f0 = lerp(f0, saturate(SurfaceDiffuse.Load(int3(texel, 0)).rgb),
                saturate(SurfaceAmbientMetallic.Load(int3(texel, 0)).w));
        float3 fresnel = f0 + (1.0f - f0) * pow(1.0f - saturate(facing), FresnelPower);
        float2 edgeDistance = min(hitPixel, ImageExtentInverse.xy - hitPixel);
        float edgeWeight = StrengthRoughnessEdgeFade.z > 0.0f ?
            saturate(min(edgeDistance.x, edgeDistance.y) / StrengthRoughnessEdgeFade.z) : 1.0f;
        float distanceWeight = 1.0f - smoothstep(RayDistanceThicknessBiasStride.x * RayFadeStart,
            RayDistanceThicknessBiasStride.x, length(hitPosition - origin));
        float roughnessRatio = source.Roughness / StrengthRoughnessEdgeFade.y;
        float weight = StrengthRoughnessEdgeFade.x * edgeWeight * distanceWeight * saturate(1.0f - roughnessRatio * roughnessRatio);
        float3 reflected = max(OpaqueRadiance.SampleLevel(RadianceSampler, hitPixel * ImageExtentInverse.zw, 0).rgb, 0.0f) * fresnel * weight;
        // Reserve one binary16 ULP for rounding before the RGB additive blend.
        float3 available = max(HalfFloatMaximum - HalfFloatMaximumSpacing - OpaqueRadiance.Load(int3(texel, 0)).rgb, 0.0f);
        return min(reflected, available);
    }
    return 0.0f;
}

float4 PSMain(FullscreenVertex input) : SV_TARGET
{
    int2 texel = int2(input.Position.xy);
    if (StageBudgetShading.x == CompositeStage)
        return float4(TracedRadiance.Load(int3(texel, 0)).rgb, 0.0f);
    return float4(ReflectionEstimate(texel), 0.0f);
}
