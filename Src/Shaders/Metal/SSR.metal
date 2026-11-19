#include <metal_stdlib>
using namespace metal;

// Independent Metal implementation of perspective-correct screen-space rays.
// McGuire and Mara, Efficient GPU Screen-Space Ray Tracing, JCGT (2014):
// https://jcgt.org/published/0003/04/04/paper.pdf
// World/w and reciprocal W interpolate along the projected major axis. A finite
// world-depth slab approximates the single visible layer. Roughness attenuates
// mirror radiance; glossy integration and hidden-surface fallback are absent.
struct ReflectionUniforms
{
    float4 ViewProjection[4];
    float4 InverseViewProjection[4];
    float4 Viewport; // width, height, reciprocal width, reciprocal height
    float4 Trace; // world distance, thickness, bias, pixel stride
    float4 Appearance; // intensity, roughness cutoff, edge fade pixels, reserved
    uint4 Options; // phase, maximum steps, shading model, reserved
};
struct ReflectionVertex { float4 Position [[position]]; };
struct SurfaceSample
{
    float3 Position;
    float3 Normal;
    float3 CameraRay;
    float Roughness;
};
struct ProjectedSegment
{
    float2 FirstPixel;
    float2 LastPixel;
    float3 FirstWorldOverW;
    float3 LastWorldOverW;
    float2 ReciprocalW;
};
constant float ProjectionTolerance = 1e-7;
constant float BackgroundDepth = 1.0;
constant float CameraInteriorDepth = 0.5;
constant float RayOriginExclusionPixels = 2.0;
constant float DistanceFadeStartFraction = 0.8;
constant float DielectricF0 = 0.04;
constant float HalfFloatLimit = 65504.0;
constant float HalfFloatTopInterval = 32.0;
constant uint CompositePhase = 1;
constant uint PbrShadingModel = 3;

vertex ReflectionVertex SSRVertexMain(uint index [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 point = index == 0 ? float2(-1) :
        index == 1 ? float2(FullscreenExtent, -1) : float2(-1, FullscreenExtent);
    return {float4(point, 0, 1)};
}
float4 MultiplyRows(constant float4* matrix, float4 value)
{
    return float4(dot(matrix[0], value), dot(matrix[1], value), dot(matrix[2], value), dot(matrix[3], value));
}
float3 UnitDirection(float3 value)
{
    return value * rsqrt(max(dot(value, value), ProjectionTolerance));
}
bool ReconstructWorld(float2 uv, float depth, constant ReflectionUniforms& settings, thread float3& position)
{
    // Metal pixel/texture coordinates have a top-left origin. The CPU matrix
    // uses clip Y up and hardware depth [0,1]; explicit rows preserve its layout.
    float4 projected(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
    float4 world = MultiplyRows(settings.InverseViewProjection, projected);
    if (abs(world.w) < ProjectionTolerance) return false;
    position = world.xyz / world.w;
    return all(isfinite(position));
}
bool LoadSurface(int2 pixel, constant ReflectionUniforms& settings,
    depth2d<float, access::read> depth, texture2d<float, access::read> normals, thread SurfaceSample& surface)
{
    if (any(pixel < int2(0)) || any(pixel >= int2(settings.Viewport.xy))) return false;
    float z = depth.read(uint2(pixel));
    float4 encoded = normals.read(uint2(pixel));
    if (z >= BackgroundDepth || encoded.w < 0 || dot(encoded.xyz, encoded.xyz) < ProjectionTolerance) return false;
    float2 uv = (float2(pixel) + 0.5) * settings.Viewport.zw;
    float3 nearPosition, middlePosition;
    if (!ReconstructWorld(uv, z, settings, surface.Position) ||
        !ReconstructWorld(uv, 0, settings, nearPosition) ||
        !ReconstructWorld(uv, CameraInteriorDepth, settings, middlePosition)) return false;
    surface.Normal = UnitDirection(encoded.xyz);
    surface.CameraRay = UnitDirection(middlePosition - nearPosition);
    surface.Roughness = encoded.w;
    return all(isfinite(surface.Normal)) && all(isfinite(surface.CameraRay));
}
bool TrimHalfspace(float firstDistance, float lastDistance, thread float2& interval)
{
    if (firstDistance < 0 && lastDistance < 0) return false;
    if (firstDistance < 0) interval.x = max(interval.x, firstDistance / (firstDistance - lastDistance));
    if (lastDistance < 0) interval.y = min(interval.y, firstDistance / (firstDistance - lastDistance));
    return interval.x < interval.y;
}
bool ProjectRay(float3 origin, float3 direction, constant ReflectionUniforms& settings, thread ProjectedSegment& segment)
{
    float3 endpoint = origin + direction * settings.Trace.x;
    float4 a = MultiplyRows(settings.ViewProjection, float4(origin, 1));
    float4 b = MultiplyRows(settings.ViewProjection, float4(endpoint, 1));
    float2 interval(0, 1);
    if (!TrimHalfspace(a.w - ProjectionTolerance, b.w - ProjectionTolerance, interval) ||
        !TrimHalfspace(a.w + a.x, b.w + b.x, interval) || !TrimHalfspace(a.w - a.x, b.w - b.x, interval) ||
        !TrimHalfspace(a.w + a.y, b.w + b.y, interval) || !TrimHalfspace(a.w - a.y, b.w - b.y, interval) ||
        !TrimHalfspace(a.z, b.z, interval) || !TrimHalfspace(a.w - a.z, b.w - b.z, interval)) return false;
    float4 first = mix(a, b, interval.x), last = mix(a, b, interval.y);
    segment.ReciprocalW = 1.0 / float2(first.w, last.w);
    segment.FirstWorldOverW = mix(origin, endpoint, interval.x) * segment.ReciprocalW.x;
    segment.LastWorldOverW = mix(origin, endpoint, interval.y) * segment.ReciprocalW.y;
    segment.FirstPixel = (first.xy * segment.ReciprocalW.x * float2(0.5, -0.5) + 0.5) * settings.Viewport.xy;
    segment.LastPixel = (last.xy * segment.ReciprocalW.y * float2(0.5, -0.5) + 0.5) * settings.Viewport.xy;
    return all(isfinite(segment.FirstWorldOverW)) && all(isfinite(segment.LastWorldOverW));
}
bool MarchVisibleLayer(float3 origin, float3 direction, constant ReflectionUniforms& settings,
    depth2d<float, access::read> depth, texture2d<float, access::read> normals,
    thread float2& hitUV, thread float& hitDistance)
{
    ProjectedSegment segment;
    if (!ProjectRay(origin, direction, settings, segment)) return false;
    float2 displacement = abs(segment.LastPixel - segment.FirstPixel);
    float majorLength = max(displacement.x, displacement.y);
    if (majorLength <= RayOriginExclusionPixels) return false;
    float intervalSize = settings.Trace.w / majorLength;
    float firstInterval = RayOriginExclusionPixels / majorLength;
    for (uint index = 0; index < settings.Options.y; ++index)
    {
        float start = firstInterval + float(index) * intervalSize;
        if (start >= 1) break;
        float end = min(start + intervalSize, 1.0), center = (start + end) * 0.5;
        int2 candidate = int2(floor(mix(segment.FirstPixel, segment.LastPixel, center)));
        SurfaceSample surface;
        if (!LoadSurface(candidate, settings, depth, normals, surface) || dot(surface.Normal, -direction) <= 0) continue;
        // Project separation onto this pixel's camera ray in world units. Its
        // numerator is linear; dividing by interpolated reciprocal W bounds the
        // complete interval instead of testing isolated world-distance samples.
        float a = dot(segment.FirstWorldOverW - surface.Position * segment.ReciprocalW.x, surface.CameraRay);
        float b = dot(segment.LastWorldOverW - surface.Position * segment.ReciprocalW.y, surface.CameraRay);
        float nearSeparation = mix(a, b, start) / mix(segment.ReciprocalW.x, segment.ReciprocalW.y, start);
        float farSeparation = mix(a, b, end) / mix(segment.ReciprocalW.x, segment.ReciprocalW.y, end);
        if (max(nearSeparation, farSeparation) < 0 || min(nearSeparation, farSeparation) > settings.Trace.y) continue;
        float slope = b - a;
        float crossing = abs(slope) > ProjectionTolerance ? clamp(-a / slope, start, end) : center;
        float2 hitPixel = mix(segment.FirstPixel, segment.LastPixel, crossing);
        if (!LoadSurface(int2(floor(hitPixel)), settings, depth, normals, surface) || dot(surface.Normal, -direction) <= 0) continue;
        float3 hitPosition = mix(segment.FirstWorldOverW, segment.LastWorldOverW, crossing) /
            mix(segment.ReciprocalW.x, segment.ReciprocalW.y, crossing);
        float refinedSeparation = dot(hitPosition - surface.Position, surface.CameraRay);
        if (refinedSeparation < -ProjectionTolerance || refinedSeparation > settings.Trace.y) continue;
        hitUV = hitPixel * settings.Viewport.zw;
        hitDistance = length(hitPosition - origin);
        return true;
    }
    return false;
}

fragment float4 SSRFragmentMain(ReflectionVertex input [[stage_in]],
    constant ReflectionUniforms& settings [[buffer(0)]],
    texture2d<float, access::sample> scene [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::read> normals [[texture(2)]],
    texture2d<float, access::read> specular [[texture(3)]],
    texture2d<float, access::read> diffuse [[texture(4)]],
    texture2d<float, access::read> ambient [[texture(5)]],
    texture2d<float, access::read> reflection [[texture(6)]])
{
    uint2 pixel = uint2(input.Position.xy);
    if (settings.Options.x == CompositePhase) return float4(reflection.read(pixel).rgb, 0);
    SurfaceSample surface;
    if (!LoadSurface(int2(pixel), settings, depth, normals, surface) ||
        surface.Roughness >= settings.Appearance.y || settings.Appearance.x <= 0) return float4(0);
    float facing = saturate(dot(surface.Normal, -surface.CameraRay));
    if (facing <= 0) return float4(0);
    float3 direction = reflect(surface.CameraRay, surface.Normal);
    float2 hitUV;
    float hitDistance;
    if (!MarchVisibleLayer(surface.Position + surface.Normal * settings.Trace.z, direction,
        settings, depth, normals, hitUV, hitDistance)) return float4(0);
    float3 f0 = saturate(specular.read(pixel).rgb);
    if (settings.Options.z == PbrShadingModel)
        f0 = mix(f0, saturate(diffuse.read(pixel).rgb), saturate(ambient.read(pixel).a));
    float grazing = 1 - facing;
    float3 fresnel = f0 + (1 - f0) * grazing * grazing * grazing * grazing * grazing;
    float2 edgePixels = min(hitUV, 1 - hitUV) * settings.Viewport.xy;
    float edgeWeight = settings.Appearance.z > 0 ? saturate(min(edgePixels.x, edgePixels.y) / settings.Appearance.z) : 1;
    float distanceWeight = 1 - smoothstep(settings.Trace.x * DistanceFadeStartFraction, settings.Trace.x, hitDistance);
    float roughnessRatio = surface.Roughness / settings.Appearance.y;
    float roughnessWeight = saturate(1 - roughnessRatio * roughnessRatio);
    constexpr sampler RadianceSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float3 incoming = max(scene.sample(RadianceSampler, hitUV, level(0)).rgb, float3(0));
    float3 contribution = incoming * fresnel * (settings.Appearance.x * edgeWeight * distanceWeight * roughnessWeight);
    // Reserve one top-range binary16 interval before RGB-only additive blending.
    float3 available = max(float3(HalfFloatLimit - HalfFloatTopInterval) - scene.read(pixel).rgb, float3(0));
    return float4(min(contribution, available), 0);
}
