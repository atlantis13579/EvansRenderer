#version 450
// Vulkan-native screen-space diffuse transport and temporal reconstruction.
// Independent GLSL is compiled in memory at runtime; no embedded bytecode.
// Projected intervals: https://jcgt.org/published/0003/04/04/
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform DiffuseGI
{
    mat4 ViewProjection, InverseViewProjection, PreviousInverseViewProjection;
    vec4 PreviousDepthPlane, Viewport, Trace, Appearance, Temporal, JitterDelta;
    uvec4 Options, History;
} GI;
layout(set = 0, binding = 1) uniform texture2D OpaqueRadiance;
layout(set = 0, binding = 2) uniform texture2D OpaqueDepth;
layout(set = 0, binding = 3) uniform texture2D SurfaceNormal;
layout(set = 0, binding = 4) uniform texture2D SurfaceDiffuse;
layout(set = 0, binding = 5) uniform texture2D SurfaceAmbient;
layout(set = 0, binding = 6) uniform texture2D SurfaceMotion;
layout(set = 0, binding = 7) uniform texture2D CurrentEstimate;
layout(set = 0, binding = 8) uniform texture2D PreviousEstimate;
layout(set = 0, binding = 9) uniform texture2D PreviousDepth;
layout(set = 0, binding = 10) uniform texture2D PreviousNormal;
layout(set = 0, binding = 11) uniform sampler NearestSampler;
layout(location = 0) out vec4 OutRadiance;
#ifdef EVANS_SSGI_TEMPORAL
layout(location = 1) out float OutDepth;
layout(location = 2) out vec4 OutNormal;
#endif
const float DivisionEpsilon = 1e-7, ClearDepth = 1.0, InteriorDepth = 0.5;
const float ExcludedStartPixels = 2.0, FarFadeFraction = 0.8, NonmetalF0 = 0.04, MaximumBinary16 = 65504.0;
const uint MetallicPbrModel = 3u, TraceStage = 0u, SpatialStage = 1u;
const float Pi = 3.14159265358979323846, BasisParallelCosine = 0.999, PixelCenter = 0.5;
const float MinimumDepthTolerance = 0.001, MinimumColorBox = 0.001, MaximumHistoryCount = 32.0;
const int FilterRadius = 1;
// Published PCG RXS-M-XS parameters, not scene-dependent coefficients.
// Jarzynski/Olano, JCGT 2020: https://jcgt.org/published/0009/03/02/
const uint StateMultiplier = 747796405u, StateIncrement = 2891336453u, OutputMultiplier = 277803737u;
const uint StateRotationShift = 28u, MinimumRotationShift = 4u, FinalRotationShift = 22u, DiscardLowBits = 8u;
const float UniformDivisor = 16777216.0;
uint PermuteBits(uint seed)
{
    uint state = seed * StateMultiplier + StateIncrement;
    uint word = ((state >> ((state >> StateRotationShift) + MinimumRotationShift)) ^ state) * OutputMultiplier;
    return (word >> FinalRotationShift) ^ word;
}
float UniformSample(inout uint seed)
{
    seed = PermuteBits(seed);
    return float(seed >> DiscardLowBits) / UniformDivisor;
}
struct VisibleSurface
{
    vec3 position;
    vec3 normal;
    vec3 cameraDirection;
    float roughness;
};
struct ScreenRay
{
    vec2 pixelBegin;
    vec2 pixelEnd;
    vec3 worldOverWBegin;
    vec3 worldOverWEnd;
    vec2 inverseW;
};
bool IsFinite(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
vec3 SafeDirection(vec3 value) { return value * inversesqrt(max(dot(value, value), DivisionEpsilon)); }

bool Reconstruct(vec2 uv, float depth, out vec3 world)
{
    // CPU projection uses +Y up and [0,1] depth. Geometry flips clip Y for
    // Vulkan; fragment and texture coordinates therefore share a top-left origin.
    vec4 homogeneous = GI.InverseViewProjection * vec4(uv * vec2(2, -2) + vec2(-1, 1), depth, 1);
    if (abs(homogeneous.w) < DivisionEpsilon) return false;
    world = homogeneous.xyz / homogeneous.w;
    return IsFinite(world);
}

bool FetchSurface(ivec2 pixel, out VisibleSurface surface)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ivec2(GI.Viewport.xy)))) return false;
    float depth = texelFetch(sampler2D(OpaqueDepth, NearestSampler), pixel, 0).r;
    vec4 normalRoughness = texelFetch(sampler2D(SurfaceNormal, NearestSampler), pixel, 0);
    if (depth >= ClearDepth || normalRoughness.w < 0.0 || dot(normalRoughness.xyz, normalRoughness.xyz) < DivisionEpsilon) return false;
    vec2 uv = (vec2(pixel) + 0.5) * GI.Viewport.zw;
    vec3 nearPoint, interiorPoint;
    if (!Reconstruct(uv, depth, surface.position) || !Reconstruct(uv, 0.0, nearPoint) ||
        !Reconstruct(uv, InteriorDepth, interiorPoint)) return false;
    surface.normal = SafeDirection(normalRoughness.xyz);
    surface.cameraDirection = SafeDirection(interiorPoint - nearPoint);
    surface.roughness = normalRoughness.w;
    return IsFinite(surface.normal) && IsFinite(surface.cameraDirection);
}

bool RestrictSegment(vec2 signedDistances, inout float first, inout float last)
{
    if (all(lessThan(signedDistances, vec2(0)))) return false;
    if (signedDistances.x < 0.0) first = max(first, signedDistances.x / (signedDistances.x - signedDistances.y));
    if (signedDistances.y < 0.0) last = min(last, signedDistances.x / (signedDistances.x - signedDistances.y));
    return first < last;
}

bool MakeScreenRay(vec3 origin, vec3 direction, out ScreenRay ray)
{
    vec3 finish = origin + direction * GI.Trace.x;
    vec4 clipA = GI.ViewProjection * vec4(origin, 1);
    vec4 clipB = GI.ViewProjection * vec4(finish, 1);
    vec2 w = vec2(clipA.w, clipB.w), x = vec2(clipA.x, clipB.x);
    vec2 y = vec2(clipA.y, clipB.y), z = vec2(clipA.z, clipB.z);
    float first = 0.0, last = 1.0;
    if (!RestrictSegment(w - DivisionEpsilon, first, last) ||
        !RestrictSegment(w + x, first, last) || !RestrictSegment(w - x, first, last) ||
        !RestrictSegment(w + y, first, last) || !RestrictSegment(w - y, first, last) ||
        !RestrictSegment(z, first, last) || !RestrictSegment(w - z, first, last)) return false;
    vec4 a = mix(clipA, clipB, first), b = mix(clipA, clipB, last);
    ray.inverseW = 1.0 / vec2(a.w, b.w);
    ray.worldOverWBegin = mix(origin, finish, first) * ray.inverseW.x;
    ray.worldOverWEnd = mix(origin, finish, last) * ray.inverseW.y;
    ray.pixelBegin = (a.xy * ray.inverseW.x * vec2(0.5, -0.5) + 0.5) * GI.Viewport.xy;
    ray.pixelEnd = (b.xy * ray.inverseW.y * vec2(0.5, -0.5) + 0.5) * GI.Viewport.xy;
    return IsFinite(ray.worldOverWBegin) && IsFinite(ray.worldOverWEnd);
}

bool IntersectInterval(ScreenRay ray, vec2 interval, vec3 direction, out vec2 pixel, out vec3 world)
{
    float middle = (interval.x + interval.y) * 0.5;
    VisibleSurface surface;
    if (!FetchSurface(ivec2(floor(mix(ray.pixelBegin, ray.pixelEnd, middle))), surface) ||
        dot(surface.normal, -direction) <= 0.0) return false;
    vec2 separationNumerator = vec2(
        dot(ray.worldOverWBegin - surface.position * ray.inverseW.x, surface.cameraDirection),
        dot(ray.worldOverWEnd - surface.position * ray.inverseW.y, surface.cameraDirection));
    vec2 separation = mix(vec2(separationNumerator.x), vec2(separationNumerator.y), interval) /
        mix(vec2(ray.inverseW.x), vec2(ray.inverseW.y), interval);
    if (max(separation.x, separation.y) < 0.0 || min(separation.x, separation.y) > GI.Trace.y) return false;
    float slope = separationNumerator.y - separationNumerator.x;
    float crossing = abs(slope) > DivisionEpsilon ?
        clamp(-separationNumerator.x / slope, interval.x, interval.y) : middle;
    pixel = mix(ray.pixelBegin, ray.pixelEnd, crossing);
    // The plane crossing can move to a neighboring pixel. Revalidate that
    // pixel's normal and depth instead of accepting the original sample's slab.
    if (!FetchSurface(ivec2(floor(pixel)), surface) || dot(surface.normal, -direction) <= 0.0) return false;
    world = mix(ray.worldOverWBegin, ray.worldOverWEnd, crossing) / mix(ray.inverseW.x, ray.inverseW.y, crossing);
    float refinedSeparation = dot(world - surface.position, surface.cameraDirection);
    return refinedSeparation >= -DivisionEpsilon && refinedSeparation <= GI.Trace.y;
}

bool TraceVisibleRay(vec3 origin, vec3 direction, out vec2 uv, out float distance)
{
    ScreenRay ray;
    if (!MakeScreenRay(origin, direction, ray)) return false;
    vec2 extent = abs(ray.pixelEnd - ray.pixelBegin);
    float majorAxis = max(extent.x, extent.y);
    if (majorAxis <= ExcludedStartPixels) return false;
    float advance = GI.Trace.w / majorAxis;
    for (uint sampleIndex = 0u; sampleIndex < GI.Options.z; ++sampleIndex)
    {
        float begin = ExcludedStartPixels / majorAxis + float(sampleIndex) * advance;
        if (begin >= 1.0) break;
        vec2 pixel;
        vec3 world;
        if (!IntersectInterval(ray, vec2(begin, min(begin + advance, 1.0)), direction, pixel, world)) continue;
        uv = pixel * GI.Viewport.zw;
        distance = length(world - origin);
        return true;
    }
    return false;
}

vec4 TraceDiffuse(ivec2 pixel)
{
    VisibleSurface receiver;
    if (!FetchSurface(pixel, receiver)) return vec4(0.0);
    vec3 albedo = clamp(texelFetch(sampler2D(SurfaceDiffuse, NearestSampler), pixel, 0).rgb, 0.0, 1.0);
    if (GI.History.y == MetallicPbrModel)
        albedo *= (1.0 - clamp(texelFetch(sampler2D(SurfaceAmbient, NearestSampler), pixel, 0).w, 0.0, 1.0)) * (1.0 - NonmetalF0);
    vec3 axis = abs(receiver.normal.z) < BasisParallelCosine ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(axis, receiver.normal));
    vec3 bitangent = cross(receiver.normal, tangent);
    uint seed = (uint(pixel.y) * uint(GI.Viewport.x) + uint(pixel.x)) ^ PermuteBits(GI.Options.w);
    vec3 incoming = vec3(0.0);
    for (uint ray = 0u; ray < GI.Options.y; ++ray)
    {
        float radial = UniformSample(seed), phase = 2.0 * Pi * UniformSample(seed);
        float radius = sqrt(radial);
        vec3 direction = tangent * (radius * cos(phase)) + bitangent * (radius * sin(phase)) + receiver.normal * sqrt(1.0 - radial);
        vec2 uv; float distance;
        if (!TraceVisibleRay(receiver.position + receiver.normal * GI.Trace.z, direction, uv, distance)) continue;
        vec2 edge = min(uv, 1.0 - uv) * GI.Viewport.xy;
        float confidence = GI.Appearance.y > 0.0 ? clamp(min(edge.x, edge.y) / GI.Appearance.y, 0.0, 1.0) : 1.0;
        confidence *= 1.0 - smoothstep(GI.Trace.x * FarFadeFraction, GI.Trace.x, distance);
        ivec2 hit = clamp(ivec2(uv * GI.Viewport.xy), ivec2(0), ivec2(GI.Viewport.xy) - 1);
        incoming += max(texelFetch(sampler2D(OpaqueRadiance, NearestSampler), hit, 0).rgb, 0.0) * confidence;
    }
    // The cosine-weighted PDF cancels the Lambertian cosine/pi factor.
    return vec4(clamp(albedo * incoming / float(GI.Options.y), 0.0, MaximumBinary16), 1.0);
}
float SurfaceWeight(ivec2 pixel, ivec2 neighbor, VisibleSurface receiver)
{
    VisibleSurface other;
    if (!FetchSurface(neighbor, other) || dot(receiver.normal, other.normal) < GI.Appearance.w) return 0.0;
    float planeDistance = abs(dot(other.position - receiver.position, receiver.normal));
    float gaussian = (neighbor.x == pixel.x ? 2.0 : 1.0) * (neighbor.y == pixel.y ? 2.0 : 1.0);
    return gaussian * exp(-planeDistance / GI.Appearance.z);
}
vec4 FilterDiffuse(ivec2 pixel)
{
    VisibleSurface receiver;
    if (!FetchSurface(pixel, receiver)) return vec4(0.0);
    vec3 weighted = vec3(0.0); float total = 0.0;
    for (int y = -FilterRadius; y <= FilterRadius; ++y) for (int x = -FilterRadius; x <= FilterRadius; ++x)
    {
        ivec2 neighbor = pixel + ivec2(x, y); float weight = SurfaceWeight(pixel, neighbor, receiver);
        if (weight <= 0.0) continue;
        weighted += max(texelFetch(sampler2D(CurrentEstimate, NearestSampler), neighbor, 0).rgb, 0.0) * weight; total += weight;
    }
    return vec4(weighted / max(total, DivisionEpsilon), 1.0);
}
float ReprojectedDistance(vec2 pixel, float depth)
{
    vec2 uv = pixel * GI.Viewport.zw;
    vec4 world = GI.PreviousInverseViewProjection * vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth, 1.0);
    if (abs(world.w) <= DivisionEpsilon) return -1.0;
    return dot(GI.PreviousDepthPlane, vec4(world.xyz / world.w, 1.0));
}
vec4 TemporalDiffuse(ivec2 pixel)
{
    vec4 current = texelFetch(sampler2D(CurrentEstimate, NearestSampler), pixel, 0);
    VisibleSurface receiver;
    if (GI.History.x == 0u || GI.Temporal.x <= 0.0 || !FetchSurface(pixel, receiver)) return current;
    vec4 motion = texelFetch(sampler2D(SurfaceMotion, NearestSampler), pixel, 0);
    if (motion.w <= 0.0) return current;
    vec2 previousPixel = vec2(pixel) + PixelCenter - motion.xy - GI.JitterDelta.xy;
    if (any(lessThan(previousPixel, vec2(PixelCenter))) || any(greaterThan(previousPixel, GI.Viewport.xy - PixelCenter))) return current;
    float expected = ReprojectedDistance(previousPixel, 1.0-motion.z);
    if (isnan(expected) || isinf(expected) || expected < 0.0) return current;
    float tolerance = max(MinimumDepthTolerance, abs(expected) * GI.Temporal.y);
    ivec2 base = ivec2(floor(previousPixel - PixelCenter)); vec2 fraction = fract(previousPixel - PixelCenter);
    vec4 historySum = vec4(0.0); float total = 0.0;
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x)
    {
        ivec2 candidate = base + ivec2(x, y);
        if (any(lessThan(candidate, ivec2(0))) || any(greaterThanEqual(candidate, ivec2(GI.Viewport.xy)))) continue;
        float depth = texelFetch(sampler2D(PreviousDepth, NearestSampler), candidate, 0).r;
        vec3 normal = texelFetch(sampler2D(PreviousNormal, NearestSampler), candidate, 0).xyz;
        vec4 old = texelFetch(sampler2D(PreviousEstimate, NearestSampler), candidate, 0);
        float distance = ReprojectedDistance(vec2(candidate) + PixelCenter, depth);
        if (depth >= ClearDepth || dot(normal, normal) < DivisionEpsilon || dot(SafeDirection(normal), receiver.normal) < GI.Temporal.z ||
            old.a <= 0.0 || isnan(distance) || isinf(distance) || abs(distance - expected) > tolerance) continue;
        float weight = (x == 0 ? 1.0 - fraction.x : fraction.x) * (y == 0 ? 1.0 - fraction.y : fraction.y);
        historySum += old * weight; total += weight;
    }
    if (total <= DivisionEpsilon) return current;
    vec4 history = historySum / total;
    vec3 low = current.rgb, high = low;
    for (int y = -FilterRadius; y <= FilterRadius; ++y) for (int x = -FilterRadius; x <= FilterRadius; ++x)
    {
        ivec2 neighbor = pixel + ivec2(x, y);
        if (SurfaceWeight(pixel, neighbor, receiver) <= 0.0) continue;
        vec3 value = texelFetch(sampler2D(CurrentEstimate, NearestSampler), neighbor, 0).rgb;
        low = min(low, value); high = max(high, value);
    }
    vec3 center = (low + high) * 0.5, extent = max((high - low) * 0.5, MinimumColorBox);
    vec3 delta = history.rgb - center, ratio = abs(delta / extent);
    history.rgb = center + delta / max(1.0, max(ratio.x, max(ratio.y, ratio.z)));
    float feedback = min(GI.Temporal.x, history.a / (history.a + 1.0));
    return vec4(clamp(mix(current.rgb, history.rgb, feedback), 0.0, MaximumBinary16), min(history.a + 1.0, MaximumHistoryCount));
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#ifdef EVANS_SSGI_TEMPORAL
    OutRadiance = TemporalDiffuse(pixel);
    OutDepth = texelFetch(sampler2D(OpaqueDepth, NearestSampler), pixel, 0).r;
    OutNormal = texelFetch(sampler2D(SurfaceNormal, NearestSampler), pixel, 0);
#else
    if (GI.Options.x == TraceStage) OutRadiance = TraceDiffuse(pixel);
    else if (GI.Options.x == SpatialStage) OutRadiance = FilterDiffuse(pixel);
    else
    {
        vec4 scene = texelFetch(sampler2D(OpaqueRadiance, NearestSampler), pixel, 0);
        vec3 bounce = max(texelFetch(sampler2D(CurrentEstimate, NearestSampler), pixel, 0).rgb, 0.0) * GI.Appearance.x;
        OutRadiance = vec4(clamp(scene.rgb + bounce, 0.0, MaximumBinary16), scene.a);
    }
#endif
}
#endif
