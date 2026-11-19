#version 450
// Independent Vulkan spatial SSR. Perspective-correct world/w interpolation
// follows McGuire and Mara, Efficient GPU Screen-Space Ray Tracing (2014):
// https://jcgt.org/published/0003/04/04/paper.pdf
// A single visible depth layer supplies finite world-space hit slabs. Misses
// contribute zero; roughness fades a mirror ray without glossy convolution.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const vec2 Triangle[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(Triangle[gl_VertexIndex], 0, 1);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform ReflectionParameters
{
    mat4 ViewProjection;
    mat4 InverseViewProjection;
    vec4 Viewport; // extent and reciprocal extent
    vec4 Trace; // world distance, thickness, bias, pixel stride
    vec4 Appearance; // intensity, maximum roughness, edge fade pixels, reserved
    uvec4 Options; // phase, step budget, shading model, reserved
} Reflection;
layout(set = 0, binding = 1) uniform texture2D OpaqueRadiance;
layout(set = 0, binding = 2) uniform texture2D OpaqueDepth;
layout(set = 0, binding = 3) uniform texture2D SurfaceNormal;
layout(set = 0, binding = 4) uniform texture2D SurfaceSpecular;
layout(set = 0, binding = 5) uniform texture2D SurfaceDiffuse;
layout(set = 0, binding = 6) uniform texture2D SurfaceAmbient;
layout(set = 0, binding = 7) uniform texture2D TracedRadiance;
layout(set = 0, binding = 8) uniform sampler NearestSampler;
layout(set = 0, binding = 9) uniform sampler LinearSampler;
layout(location = 0) out vec4 OutRadiance;

const float DivisionEpsilon = 1e-7;
const float ClearDepth = 1.0;
const float InteriorDepth = 0.5;
const float ExcludedStartPixels = 2.0;
const float FarFadeFraction = 0.8;
const float NonmetalF0 = 0.04;
const float MaximumBinary16 = 65504.0;
const float TopBinary16Spacing = 32.0;
const uint CompositePass = 1u;
const uint MetallicPbrModel = 3u;

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
    vec4 homogeneous = Reflection.InverseViewProjection * vec4(uv * vec2(2, -2) + vec2(-1, 1), depth, 1);
    if (abs(homogeneous.w) < DivisionEpsilon) return false;
    world = homogeneous.xyz / homogeneous.w;
    return IsFinite(world);
}

bool FetchSurface(ivec2 pixel, out VisibleSurface surface)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ivec2(Reflection.Viewport.xy)))) return false;
    float depth = texelFetch(sampler2D(OpaqueDepth, NearestSampler), pixel, 0).r;
    vec4 normalRoughness = texelFetch(sampler2D(SurfaceNormal, NearestSampler), pixel, 0);
    if (depth >= ClearDepth || normalRoughness.w < 0.0 || dot(normalRoughness.xyz, normalRoughness.xyz) < DivisionEpsilon) return false;
    vec2 uv = (vec2(pixel) + 0.5) * Reflection.Viewport.zw;
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
    vec3 finish = origin + direction * Reflection.Trace.x;
    vec4 clipA = Reflection.ViewProjection * vec4(origin, 1);
    vec4 clipB = Reflection.ViewProjection * vec4(finish, 1);
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
    ray.pixelBegin = (a.xy * ray.inverseW.x * vec2(0.5, -0.5) + 0.5) * Reflection.Viewport.xy;
    ray.pixelEnd = (b.xy * ray.inverseW.y * vec2(0.5, -0.5) + 0.5) * Reflection.Viewport.xy;
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
    if (max(separation.x, separation.y) < 0.0 || min(separation.x, separation.y) > Reflection.Trace.y) return false;
    float slope = separationNumerator.y - separationNumerator.x;
    float crossing = abs(slope) > DivisionEpsilon ?
        clamp(-separationNumerator.x / slope, interval.x, interval.y) : middle;
    pixel = mix(ray.pixelBegin, ray.pixelEnd, crossing);
    // The plane crossing can move to a neighboring pixel. Revalidate that
    // pixel's normal and depth instead of accepting the original sample's slab.
    if (!FetchSurface(ivec2(floor(pixel)), surface) || dot(surface.normal, -direction) <= 0.0) return false;
    world = mix(ray.worldOverWBegin, ray.worldOverWEnd, crossing) / mix(ray.inverseW.x, ray.inverseW.y, crossing);
    float refinedSeparation = dot(world - surface.position, surface.cameraDirection);
    return refinedSeparation >= -DivisionEpsilon && refinedSeparation <= Reflection.Trace.y;
}

bool TraceVisibleRay(vec3 origin, vec3 direction, out vec2 uv, out float distance)
{
    ScreenRay ray;
    if (!MakeScreenRay(origin, direction, ray)) return false;
    vec2 extent = abs(ray.pixelEnd - ray.pixelBegin);
    float majorAxis = max(extent.x, extent.y);
    if (majorAxis <= ExcludedStartPixels) return false;
    float advance = Reflection.Trace.w / majorAxis;
    for (uint sampleIndex = 0u; sampleIndex < Reflection.Options.y; ++sampleIndex)
    {
        float begin = ExcludedStartPixels / majorAxis + float(sampleIndex) * advance;
        if (begin >= 1.0) break;
        vec2 pixel;
        vec3 world;
        if (!IntersectInterval(ray, vec2(begin, min(begin + advance, 1.0)), direction, pixel, world)) continue;
        uv = pixel * Reflection.Viewport.zw;
        distance = length(world - origin);
        return true;
    }
    return false;
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Reflection.Options.x == CompositePass)
    {
        OutRadiance = vec4(texelFetch(sampler2D(TracedRadiance, NearestSampler), pixel, 0).rgb, 0);
        return;
    }
    OutRadiance = vec4(0);
    VisibleSurface surface;
    if (!FetchSurface(pixel, surface) || surface.roughness >= Reflection.Appearance.y || Reflection.Appearance.x <= 0.0) return;
    float normalView = clamp(dot(surface.normal, -surface.cameraDirection), 0.0, 1.0);
    if (normalView <= 0.0) return;
    vec3 direction = reflect(surface.cameraDirection, surface.normal);
    vec2 hitUV;
    float hitDistance;
    if (!TraceVisibleRay(surface.position + surface.normal * Reflection.Trace.z, direction, hitUV, hitDistance)) return;
    vec3 f0 = clamp(texelFetch(sampler2D(SurfaceSpecular, NearestSampler), pixel, 0).rgb, 0.0, 1.0);
    if (Reflection.Options.z == MetallicPbrModel)
    {
        vec3 diffuse = clamp(texelFetch(sampler2D(SurfaceDiffuse, NearestSampler), pixel, 0).rgb, 0.0, 1.0);
        float metallic = clamp(texelFetch(sampler2D(SurfaceAmbient, NearestSampler), pixel, 0).a, 0.0, 1.0);
        f0 = mix(f0, diffuse, metallic);
    }
    float grazing = 1.0 - normalView;
    vec3 fresnel = f0 + (1.0 - f0) * grazing * grazing * grazing * grazing * grazing;
    vec2 edgeDistance = min(hitUV, 1.0 - hitUV) * Reflection.Viewport.xy;
    float edgeWeight = Reflection.Appearance.z > 0.0 ?
        clamp(min(edgeDistance.x, edgeDistance.y) / Reflection.Appearance.z, 0.0, 1.0) : 1.0;
    float farWeight = 1.0 - smoothstep(Reflection.Trace.x * FarFadeFraction, Reflection.Trace.x, hitDistance);
    float relativeRoughness = surface.roughness / Reflection.Appearance.y;
    float roughnessWeight = clamp(1.0 - relativeRoughness * relativeRoughness, 0.0, 1.0);
    vec3 radiance = max(textureLod(sampler2D(OpaqueRadiance, LinearSampler), hitUV, 0).rgb, vec3(0));
    vec3 contribution = radiance * fresnel * (Reflection.Appearance.x * edgeWeight * farWeight * roughnessWeight);
    vec3 existing = texelFetch(sampler2D(OpaqueRadiance, NearestSampler), pixel, 0).rgb;
    // One top-range binary16 spacing protects the later attachment addition
    // from overflow after the reflection image rounds its stored contribution.
    vec3 headroom = max(vec3(MaximumBinary16 - TopBinary16Spacing) - existing, vec3(0));
    OutRadiance = vec4(min(contribution, headroom), 0);
}
#endif
