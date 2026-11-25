#include <metal_stdlib>
using namespace metal;
// Native Metal one-bounce diffuse transport. The visible scene is sampled before
// GI, reflections and volumes; only private validated GI history is accumulated.
// McGuire/Mara, JCGT 2014: https://jcgt.org/published/0003/04/04/
struct GIUniforms
{
    float4 ViewProjection[4], InverseViewProjection[4], PreviousInverseViewProjection[4];
    float4 PreviousDepthPlane, Viewport, Trace, Appearance, Temporal, JitterDelta;
    uint4 Options, History;
};
constant float Pi = 3.14159265358979323846, BasisParallelCosine = 0.999, PixelCenter = 0.5;
constant float AbsoluteHistoryTolerance = 0.001, MinimumHistoryExtent = 0.001, MaximumHistorySamples = 32.0;
constant int FilterRadius = 1;
// PCG RXS-M-XS has fixed arithmetic constants, unrelated to the scene scale.
// Jarzynski/Olano, JCGT 2020: https://jcgt.org/published/0009/03/02/
constant uint PcgMultiplier = 747796405u, PcgIncrement = 2891336453u, PcgOutputMultiplier = 277803737u;
constant uint PcgStateShift = 28u, PcgMinimumShift = 4u, PcgFinalShift = 22u, DiscardLowBits = 8u;
constant float RandomScale = 1.0 / 16777216.0;
uint PermutedSeed(uint input)
{
    uint state = input * PcgMultiplier + PcgIncrement;
    uint output = ((state >> ((state >> PcgStateShift) + PcgMinimumShift)) ^ state) * PcgOutputMultiplier;
    return (output >> PcgFinalShift) ^ output;
}
float UniformValue(thread uint& state)
{
    state = PermutedSeed(state);
    return float(state >> DiscardLowBits) * RandomScale;
}
struct GIVertex { float4 Position [[position]]; };
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
constant uint TracePhase = 0, SpatialPhase = 1, CompositePhase = 3;
constant uint PbrShadingModel = 3;

vertex GIVertex GIVertexMain(uint index [[vertex_id]])
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
bool ReconstructWorld(float2 uv, float depth, constant GIUniforms& settings, thread float3& position)
{
    // Metal pixel/texture coordinates have a top-left origin. The CPU matrix
    // uses clip Y up and hardware depth [0,1]; explicit rows preserve its layout.
    float4 projected(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
    float4 world = MultiplyRows(settings.InverseViewProjection, projected);
    if (abs(world.w) < ProjectionTolerance) return false;
    position = world.xyz / world.w;
    return all(isfinite(position));
}
bool LoadSurface(int2 pixel, constant GIUniforms& settings,
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
bool ProjectRay(float3 origin, float3 direction, constant GIUniforms& settings, thread ProjectedSegment& segment)
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
bool MarchVisibleLayer(float3 origin, float3 direction, constant GIUniforms& settings,
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
    for (uint index = 0; index < settings.Options.z; ++index)
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


float4 DiffuseTransport(uint2 pixel, constant GIUniforms& settings,
    texture2d<float, access::read> scene, depth2d<float, access::read> depth,
    texture2d<float, access::read> normals, texture2d<float, access::read> diffuse,
    texture2d<float, access::read> ambient)
{
    SurfaceSample receiver;
    if (!LoadSurface(int2(pixel), settings, depth, normals, receiver)) return float4(0);
    float3 reflectance = saturate(diffuse.read(pixel).rgb);
    if (settings.History.y == PbrShadingModel)
        reflectance *= (1 - saturate(ambient.read(pixel).w)) * (1 - DielectricF0);
    float3 helper = abs(receiver.Normal.z) < BasisParallelCosine ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = normalize(cross(helper, receiver.Normal)), bitangent = cross(receiver.Normal, tangent);
    uint state = (pixel.y * uint(settings.Viewport.x) + pixel.x) ^ PermutedSeed(settings.Options.w);
    float3 incoming = 0;
    for (uint ray = 0; ray < settings.Options.y; ++ray)
    {
        float radial = UniformValue(state), azimuth = 2 * Pi * UniformValue(state);
        float radius = sqrt(radial);
        float3 direction = tangent * (radius * cos(azimuth)) + bitangent * (radius * sin(azimuth)) + receiver.Normal * sqrt(1 - radial);
        float2 uv; float distance;
        if (!MarchVisibleLayer(receiver.Position + receiver.Normal * settings.Trace.z, direction, settings, depth, normals, uv, distance)) continue;
        float2 edge = min(uv, 1 - uv) * settings.Viewport.xy;
        float fade = settings.Appearance.y > 0 ? saturate(min(edge.x, edge.y) / settings.Appearance.y) : 1;
        fade *= 1 - smoothstep(settings.Trace.x * DistanceFadeStartFraction, settings.Trace.x, distance);
        uint2 hit = uint2(clamp(int2(uv * settings.Viewport.xy), int2(0), int2(settings.Viewport.xy) - 1));
        incoming += max(scene.read(hit).rgb, 0.0) * fade;
    }
    // f*cos/pdf = albedo for a Lambertian surface and a cosine-weighted PDF.
    return float4(clamp(reflectance * incoming / float(settings.Options.y), 0.0, HalfFloatLimit), 1);
}
float SurfaceWeight(int2 center, int2 candidate, SurfaceSample receiver,
    constant GIUniforms& settings, depth2d<float, access::read> depth, texture2d<float, access::read> normals)
{
    SurfaceSample other;
    if (!LoadSurface(candidate, settings, depth, normals, other) || dot(receiver.Normal, other.Normal) < settings.Appearance.w) return 0;
    float planeDistance = abs(dot(other.Position - receiver.Position, receiver.Normal));
    float tapWeight = (center.x == candidate.x ? 2.0 : 1.0) * (center.y == candidate.y ? 2.0 : 1.0);
    return tapWeight * exp(-planeDistance / settings.Appearance.z);
}
float4 FilterTransport(uint2 pixel, constant GIUniforms& settings, depth2d<float, access::read> depth,
    texture2d<float, access::read> normals, texture2d<float, access::read> estimate)
{
    SurfaceSample receiver;
    if (!LoadSurface(int2(pixel), settings, depth, normals, receiver)) return float4(0);
    float3 sum = 0; float total = 0;
    for (int y = -FilterRadius; y <= FilterRadius; ++y) for (int x = -FilterRadius; x <= FilterRadius; ++x)
    {
        int2 candidate = int2(pixel) + int2(x, y);
        float weight = SurfaceWeight(int2(pixel), candidate, receiver, settings, depth, normals);
        if (weight <= 0) continue;
        sum += max(estimate.read(uint2(candidate)).rgb, 0.0) * weight; total += weight;
    }
    return float4(sum / max(total, ProjectionTolerance), 1);
}
float PreviousDistance(float2 pixel, float depth, constant GIUniforms& settings)
{
    float2 uv = pixel * settings.Viewport.zw;
    float4 world = MultiplyRows(settings.PreviousInverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    if (abs(world.w) <= ProjectionTolerance) return -1;
    return dot(settings.PreviousDepthPlane, float4(world.xyz / world.w, 1));
}
fragment float4 GIFragmentMain(GIVertex input [[stage_in]], constant GIUniforms& settings [[buffer(0)]],
    texture2d<float, access::read> scene [[texture(0)]], depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::read> normals [[texture(2)]], texture2d<float, access::read> diffuse [[texture(3)]],
    texture2d<float, access::read> ambient [[texture(4)]], texture2d<float, access::read> estimate [[texture(6)]])
{
    uint2 pixel = uint2(input.Position.xy);
    if (settings.Options.x == TracePhase) return DiffuseTransport(pixel, settings, scene, depth, normals, diffuse, ambient);
    if (settings.Options.x == SpatialPhase) return FilterTransport(pixel, settings, depth, normals, estimate);
    float4 color = scene.read(pixel);
    float3 bounce = max(estimate.read(pixel).rgb, 0.0) * settings.Appearance.x;
    return float4(clamp(color.rgb + bounce, 0.0, HalfFloatLimit), color.a);
}
struct GIHistoryOutput { float4 Color [[color(0)]]; float Depth [[color(1)]]; float4 Normal [[color(2)]]; };
fragment GIHistoryOutput GIHistoryFragment(GIVertex input [[stage_in]], constant GIUniforms& settings [[buffer(0)]],
    depth2d<float, access::read> depth [[texture(1)]], texture2d<float, access::read> normals [[texture(2)]],
    texture2d<float, access::read> velocity [[texture(5)]], texture2d<float, access::read> estimate [[texture(6)]],
    texture2d<float, access::read> historyColor [[texture(7)]], texture2d<float, access::read> historyDepth [[texture(8)]],
    texture2d<float, access::read> historyNormal [[texture(9)]])
{
    uint2 pixel = uint2(input.Position.xy);
    GIHistoryOutput result {estimate.read(pixel), depth.read(pixel), normals.read(pixel)};
    SurfaceSample receiver;
    if (settings.History.x == 0 || settings.Temporal.x <= 0 || !LoadSurface(int2(pixel), settings, depth, normals, receiver)) return result;
    float4 motion = velocity.read(pixel);
    if (motion.w <= 0) return result;
    float2 previous = input.Position.xy - motion.xy - settings.JitterDelta.xy;
    if (any(previous < PixelCenter) || any(previous > settings.Viewport.xy - PixelCenter)) return result;
    float expected = PreviousDistance(previous, 1-motion.z, settings);
    if (!isfinite(expected) || expected < 0) return result;
    float tolerance = max(AbsoluteHistoryTolerance, abs(expected) * settings.Temporal.y);
    int2 base = int2(floor(previous - PixelCenter)); float2 fraction = fract(previous - PixelCenter);
    float4 historySum = 0; float total = 0;
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x)
    {
        int2 candidate = base + int2(x, y);
        if (any(candidate < int2(0)) || any(candidate >= int2(settings.Viewport.xy))) continue;
        uint2 address = uint2(candidate);
        float oldDepth = historyDepth.read(address).x; float3 normal = historyNormal.read(address).xyz;
        float4 old = historyColor.read(address);
        float distance = PreviousDistance(float2(candidate) + PixelCenter, oldDepth, settings);
        if (oldDepth >= BackgroundDepth || dot(normal, normal) < ProjectionTolerance ||
            dot(UnitDirection(normal), receiver.Normal) < settings.Temporal.z || old.a <= 0 ||
            !isfinite(distance) || abs(distance - expected) > tolerance) continue;
        float weight = (x == 0 ? 1 - fraction.x : fraction.x) * (y == 0 ? 1 - fraction.y : fraction.y);
        historySum += old * weight; total += weight;
    }
    if (total <= ProjectionTolerance) return result;
    float4 history = historySum / total;
    float3 low = result.Color.rgb, high = low;
    for (int y = -FilterRadius; y <= FilterRadius; ++y) for (int x = -FilterRadius; x <= FilterRadius; ++x)
    {
        int2 candidate = int2(pixel) + int2(x, y);
        if (SurfaceWeight(int2(pixel), candidate, receiver, settings, depth, normals) <= 0) continue;
        float3 value = estimate.read(uint2(candidate)).rgb;
        low = min(low, value); high = max(high, value);
    }
    float3 center = (low + high) * 0.5, extent = max((high - low) * 0.5, MinimumHistoryExtent);
    float3 delta = history.rgb - center, ratio = abs(delta / extent);
    history.rgb = center + delta / max(1.0, max(ratio.x, max(ratio.y, ratio.z)));
    float feedback = min(settings.Temporal.x, history.a / (history.a + 1));
    result.Color = float4(clamp(mix(result.Color.rgb, history.rgb, feedback), 0.0, HalfFloatLimit), min(history.a + 1, MaximumHistorySamples));
    return result;
}
