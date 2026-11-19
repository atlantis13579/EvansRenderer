#version 310 es
precision highp float;
precision highp int;
precision highp sampler2D;
// Independent ES diffuse GI. Pixel coordinates remain top-left in all geometry
// and motion math; texture reads alone convert to native bottom-left storage.
// McGuire/Mara, JCGT 2014: https://jcgt.org/published/0003/04/04/
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
uniform mat4 ViewProjection, InverseViewProjection, PreviousInverseViewProjection;
uniform vec4 PreviousDepthPlane, Viewport, Trace, Appearance, Temporal, JitterDelta;
uniform uvec4 Options, History;
uniform sampler2D SceneColor, SceneDepth, NormalRoughness, DiffuseOpacity, AmbientMetallic;
uniform sampler2D MotionVectors, Estimate, HistoryColor, HistoryDepth, HistoryNormal;
layout(location = 0) out vec4 OutColor;
#ifdef EVANS_SSGI_TEMPORAL
layout(location = 1) out float OutDepth;
layout(location = 2) out vec4 OutNormal;
#endif
const float ProjectionEpsilon = 1e-7, BackgroundDepth = 1.0, CameraInteriorDepth = 0.5;
const float SelfExclusionPixels = 2.0, DistanceFadeStart = 0.8, DielectricReflectance = 0.04, HalfFloatMaximum = 65504.0;
const uint PbrShadingModel = 3u, TracePhase = 0u, SpatialPhase = 1u;
const float Pi = 3.14159265358979323846, BasisParallelLimit = 0.999, PixelCenter = 0.5;
const float HistoryDepthFloor = 0.001, HistoryBoxFloor = 0.001, HistoryCountLimit = 32.0;
const int Neighborhood = 1;
// PCG RXS-M-XS constants, defining the permutation rather than visual tuning.
// Jarzynski/Olano, JCGT 2020: https://jcgt.org/published/0009/03/02/
const uint PcgMultiplier = 747796405u, PcgOffset = 2891336453u, PcgFinalMultiplier = 277803737u;
const uint StateShift = 28u, RotationFloor = 4u, FinalShift = 22u, FloatDiscardBits = 8u;
const float FloatConversion = 1.0 / 16777216.0;
uint Pcg(uint value)
{
    uint state = value * PcgMultiplier + PcgOffset;
    uint mixed = ((state >> ((state >> StateShift) + RotationFloor)) ^ state) * PcgFinalMultiplier;
    return (mixed >> FinalShift) ^ mixed;
}
float DrawRandom(inout uint state)
{
    state = Pcg(state);
    return float(state >> FloatDiscardBits) * FloatConversion;
}
ivec2 TexturePixel(ivec2 pixel) { return ivec2(pixel.x, int(Viewport.y) - 1 - pixel.y); }
vec2 TextureUV(vec2 uv) { return vec2(uv.x, 1.0 - uv.y); }
bool Inside(ivec2 pixel) { return all(greaterThanEqual(pixel, ivec2(0))) && all(lessThan(pixel, ivec2(Viewport.xy))); }
bool Finite(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
vec3 Unit(vec3 value) { return value * inversesqrt(max(dot(value, value), ProjectionEpsilon)); }

bool Unproject(vec2 pixel, float depth, out vec3 position)
{
    vec2 uv = pixel * Viewport.zw;
    // CPU clip depth is [0, 1]. GL geometry converts clip Z, so the sampled
    // hardware depth is already in that range; applying another conversion fails.
    vec4 world = InverseViewProjection * vec4(uv * vec2(2, -2) + vec2(-1, 1), depth, 1);
    if (abs(world.w) < ProjectionEpsilon) return false;
    position = world.xyz / world.w;
    return Finite(position);
}

bool ReadSurface(ivec2 pixel, out vec3 position, out vec3 normal, out vec3 cameraRay, out float roughness)
{
    if (!Inside(pixel)) return false;
    ivec2 nativePixel = TexturePixel(pixel);
    float depth = texelFetch(SceneDepth, nativePixel, 0).r;
    vec4 material = texelFetch(NormalRoughness, nativePixel, 0);
    if (depth >= BackgroundDepth || material.w < 0.0 || dot(material.xyz, material.xyz) < ProjectionEpsilon) return false;
    vec2 center = vec2(pixel) + 0.5;
    vec3 nearPoint, interiorPoint;
    if (!Unproject(center, depth, position) || !Unproject(center, 0.0, nearPoint) ||
        !Unproject(center, CameraInteriorDepth, interiorPoint)) return false;
    normal = Unit(material.xyz);
    roughness = material.w;
    cameraRay = Unit(interiorPoint - nearPoint);
    return Finite(normal) && Finite(cameraRay);
}

bool ClipPlane(float start, float finish, inout vec2 range)
{
    if (start < 0.0 && finish < 0.0) return false;
    if (start < 0.0) range.x = max(range.x, start / (start - finish));
    if (finish < 0.0) range.y = min(range.y, start / (start - finish));
    return range.x < range.y;
}

bool ProjectSegment(vec3 origin, vec3 direction, out vec2 pixelsStart, out vec2 pixelsEnd,
    out vec3 weightedStart, out vec3 weightedEnd, out vec2 reciprocalW)
{
    vec3 endpoint = origin + direction * Trace.x;
    vec4 a = ViewProjection * vec4(origin, 1), b = ViewProjection * vec4(endpoint, 1);
    vec2 range = vec2(0, 1);
    if (!ClipPlane(a.w - ProjectionEpsilon, b.w - ProjectionEpsilon, range) ||
        !ClipPlane(a.w + a.x, b.w + b.x, range) || !ClipPlane(a.w - a.x, b.w - b.x, range) ||
        !ClipPlane(a.w + a.y, b.w + b.y, range) || !ClipPlane(a.w - a.y, b.w - b.y, range) ||
        !ClipPlane(a.z, b.z, range) || !ClipPlane(a.w - a.z, b.w - b.z, range)) return false;
    vec4 clipStart = mix(a, b, range.x), clipEnd = mix(a, b, range.y);
    reciprocalW = 1.0 / vec2(clipStart.w, clipEnd.w);
    weightedStart = mix(origin, endpoint, range.x) * reciprocalW.x;
    weightedEnd = mix(origin, endpoint, range.y) * reciprocalW.y;
    pixelsStart = (clipStart.xy * reciprocalW.x * vec2(0.5, -0.5) + 0.5) * Viewport.xy;
    pixelsEnd = (clipEnd.xy * reciprocalW.y * vec2(0.5, -0.5) + 0.5) * Viewport.xy;
    return Finite(weightedStart) && Finite(weightedEnd);
}

bool FindHit(vec3 origin, vec3 direction, out vec2 hitUV, out float hitDistance)
{
    vec2 firstPixel, lastPixel, reciprocalW;
    vec3 firstWeighted, lastWeighted;
    if (!ProjectSegment(origin, direction, firstPixel, lastPixel, firstWeighted, lastWeighted, reciprocalW)) return false;
    vec2 span = abs(lastPixel - firstPixel);
    float majorLength = max(span.x, span.y);
    if (majorLength <= SelfExclusionPixels) return false;
    float intervalLength = Trace.w / majorLength;
    float start = SelfExclusionPixels / majorLength;
    for (uint step = 0u; step < Options.z; ++step)
    {
        float begin = start + float(step) * intervalLength;
        if (begin >= 1.0) break;
        float end = min(begin + intervalLength, 1.0), midpoint = (begin + end) * 0.5;
        ivec2 pixel = ivec2(floor(mix(firstPixel, lastPixel, midpoint)));
        vec3 position, normal, viewDirection;
        float roughness;
        if (!ReadSurface(pixel, position, normal, viewDirection, roughness) || dot(normal, -direction) <= 0.0) continue;
        // Along a pixel's camera ray, separation is a linear numerator divided
        // by reciprocal W. Test the whole interval against a world-unit slab.
        vec2 numerator = vec2(dot(firstWeighted - position * reciprocalW.x, viewDirection),
            dot(lastWeighted - position * reciprocalW.y, viewDirection));
        float depthBegin = mix(numerator.x, numerator.y, begin) / mix(reciprocalW.x, reciprocalW.y, begin);
        float depthEnd = mix(numerator.x, numerator.y, end) / mix(reciprocalW.x, reciprocalW.y, end);
        if (max(depthBegin, depthEnd) < 0.0 || min(depthBegin, depthEnd) > Trace.y) continue;
        float slope = numerator.y - numerator.x;
        float crossing = abs(slope) > ProjectionEpsilon ? clamp(-numerator.x / slope, begin, end) : midpoint;
        vec2 refinedPixel = mix(firstPixel, lastPixel, crossing);
        if (!ReadSurface(ivec2(floor(refinedPixel)), position, normal, viewDirection, roughness) || dot(normal, -direction) <= 0.0) continue;
        vec3 point = mix(firstWeighted, lastWeighted, crossing) / mix(reciprocalW.x, reciprocalW.y, crossing);
        float separation = dot(point - position, viewDirection);
        if (separation < -ProjectionEpsilon || separation > Trace.y) continue;
        hitUV = refinedPixel * Viewport.zw;
        hitDistance = length(point - origin);
        return true;
    }
    return false;
}

vec4 TraceBounce(ivec2 pixel)
{
    vec3 position, normal, cameraRay; float roughness;
    if (!ReadSurface(pixel, position, normal, cameraRay, roughness)) return vec4(0.0);
    ivec2 native = TexturePixel(pixel);
    vec3 diffuse = clamp(texelFetch(DiffuseOpacity, native, 0).rgb, 0.0, 1.0);
    if (History.y == PbrShadingModel)
        diffuse *= (1.0 - clamp(texelFetch(AmbientMetallic, native, 0).w, 0.0, 1.0)) * (1.0 - DielectricReflectance);
    vec3 reference = abs(normal.z) < BasisParallelLimit ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(reference, normal)), bitangent = cross(normal, tangent);
    uint state = (uint(pixel.y) * uint(Viewport.x) + uint(pixel.x)) ^ Pcg(Options.w);
    vec3 sum = vec3(0.0);
    for (uint ray = 0u; ray < Options.y; ++ray)
    {
        float radial = DrawRandom(state), azimuth = 2.0 * Pi * DrawRandom(state);
        float radius = sqrt(radial);
        vec3 direction = tangent * (radius * cos(azimuth)) + bitangent * (radius * sin(azimuth)) + normal * sqrt(1.0 - radial);
        vec2 hitUV; float distance;
        if (!FindHit(position + normal * Trace.z, direction, hitUV, distance)) continue;
        vec2 edgePixels = min(hitUV, 1.0 - hitUV) * Viewport.xy;
        float fade = Appearance.y > 0.0 ? clamp(min(edgePixels.x, edgePixels.y) / Appearance.y, 0.0, 1.0) : 1.0;
        fade *= 1.0 - smoothstep(Trace.x * DistanceFadeStart, Trace.x, distance);
        ivec2 hit = clamp(ivec2(hitUV * Viewport.xy), ivec2(0), ivec2(Viewport.xy) - 1);
        sum += max(texelFetch(SceneColor, TexturePixel(hit), 0).rgb, 0.0) * fade;
    }
    // cosine/pi sampling cancels the Lambertian BRDF's cosine/pi factor.
    return vec4(clamp(diffuse * sum / float(Options.y), 0.0, HalfFloatMaximum), 1.0);
}
float NeighborWeight(ivec2 center, ivec2 neighbor, vec3 position, vec3 normal)
{
    vec3 other, direction, cameraRay; float roughness;
    if (!ReadSurface(neighbor, other, direction, cameraRay, roughness) || dot(normal, direction) < Appearance.w) return 0.0;
    float kernel = (center.x == neighbor.x ? 2.0 : 1.0) * (center.y == neighbor.y ? 2.0 : 1.0);
    return kernel * exp(-abs(dot(other - position, normal)) / Appearance.z);
}
vec4 FilterBounce(ivec2 pixel)
{
    vec3 position, normal, cameraRay; float roughness;
    if (!ReadSurface(pixel, position, normal, cameraRay, roughness)) return vec4(0.0);
    vec3 sum = vec3(0.0); float weightSum = 0.0;
    for (int y = -Neighborhood; y <= Neighborhood; ++y) for (int x = -Neighborhood; x <= Neighborhood; ++x)
    {
        ivec2 neighbor = pixel + ivec2(x, y); float weight = NeighborWeight(pixel, neighbor, position, normal);
        if (weight <= 0.0) continue;
        sum += max(texelFetch(Estimate, TexturePixel(neighbor), 0).rgb, 0.0) * weight; weightSum += weight;
    }
    return vec4(sum / max(weightSum, ProjectionEpsilon), 1.0);
}
float PreviousDistance(vec2 pixel, float depth)
{
    vec2 uv = pixel * Viewport.zw;
    vec4 world = PreviousInverseViewProjection * vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth, 1.0);
    if (abs(world.w) <= ProjectionEpsilon) return -1.0;
    return dot(PreviousDepthPlane, vec4(world.xyz / world.w, 1.0));
}
vec4 AccumulateBounce(ivec2 pixel)
{
    ivec2 native = TexturePixel(pixel);
    vec4 current = texelFetch(Estimate, native, 0);
    vec3 position, normal, cameraRay; float roughness;
    if (History.x == 0u || Temporal.x <= 0.0 || !ReadSurface(pixel, position, normal, cameraRay, roughness)) return current;
    vec4 motion = texelFetch(MotionVectors, native, 0);
    if (motion.w <= 0.0) return current;
    vec2 previous = vec2(pixel) + PixelCenter - motion.xy - JitterDelta.xy;
    if (any(lessThan(previous, vec2(PixelCenter))) || any(greaterThan(previous, Viewport.xy - PixelCenter))) return current;
    float expected = PreviousDistance(previous, 1.0-motion.z);
    if (isnan(expected) || isinf(expected) || expected < 0.0) return current;
    float tolerance = max(HistoryDepthFloor, abs(expected) * Temporal.y);
    ivec2 base = ivec2(floor(previous - PixelCenter)); vec2 fraction = fract(previous - PixelCenter);
    vec4 historySum = vec4(0.0); float weightSum = 0.0;
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x)
    {
        ivec2 samplePixel = base + ivec2(x, y);
        if (!Inside(samplePixel)) continue;
        ivec2 address = TexturePixel(samplePixel);
        float depth = texelFetch(HistoryDepth, address, 0).r;
        vec3 oldNormal = texelFetch(HistoryNormal, address, 0).xyz;
        vec4 old = texelFetch(HistoryColor, address, 0);
        float distance = PreviousDistance(vec2(samplePixel) + PixelCenter, depth);
        if (depth >= BackgroundDepth || dot(oldNormal, oldNormal) < ProjectionEpsilon ||
            dot(normal, Unit(oldNormal)) < Temporal.z || old.a <= 0.0 || isnan(distance) || isinf(distance) || abs(distance - expected) > tolerance) continue;
        float weight = (x == 0 ? 1.0 - fraction.x : fraction.x) * (y == 0 ? 1.0 - fraction.y : fraction.y);
        historySum += old * weight; weightSum += weight;
    }
    if (weightSum <= ProjectionEpsilon) return current;
    vec4 history = historySum / weightSum;
    vec3 minimumValue = current.rgb, maximumValue = current.rgb;
    for (int y = -Neighborhood; y <= Neighborhood; ++y) for (int x = -Neighborhood; x <= Neighborhood; ++x)
    {
        ivec2 neighbor = pixel + ivec2(x, y);
        if (NeighborWeight(pixel, neighbor, position, normal) <= 0.0) continue;
        vec3 value = texelFetch(Estimate, TexturePixel(neighbor), 0).rgb;
        minimumValue = min(minimumValue, value); maximumValue = max(maximumValue, value);
    }
    vec3 center = (minimumValue + maximumValue) * 0.5;
    vec3 extent = max((maximumValue - minimumValue) * 0.5, HistoryBoxFloor);
    vec3 delta = history.rgb - center, ratio = abs(delta / extent);
    history.rgb = center + delta / max(1.0, max(ratio.x, max(ratio.y, ratio.z)));
    float feedback = min(Temporal.x, history.a / (history.a + 1.0));
    return vec4(clamp(mix(current.rgb, history.rgb, feedback), 0.0, HalfFloatMaximum), min(history.a + 1.0, HistoryCountLimit));
}
void main()
{
    ivec2 pixel = TexturePixel(ivec2(gl_FragCoord.xy));
#ifdef EVANS_SSGI_TEMPORAL
    OutColor = AccumulateBounce(pixel);
    OutDepth = texelFetch(SceneDepth, ivec2(gl_FragCoord.xy), 0).r;
    OutNormal = texelFetch(NormalRoughness, ivec2(gl_FragCoord.xy), 0);
#else
    if (Options.x == TracePhase) OutColor = TraceBounce(pixel);
    else if (Options.x == SpatialPhase) OutColor = FilterBounce(pixel);
    else
    {
        ivec2 native = TexturePixel(pixel);
        vec4 scene = texelFetch(SceneColor, native, 0);
        vec3 bounce = max(texelFetch(Estimate, native, 0).rgb, 0.0) * Appearance.x;
        OutColor = vec4(clamp(scene.rgb + bounce, 0.0, HalfFloatMaximum), scene.a);
    }
#endif
}
#endif
