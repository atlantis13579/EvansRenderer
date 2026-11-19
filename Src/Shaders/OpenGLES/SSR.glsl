#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

// Spatial SSR using perspective-correct projected intervals. See McGuire and
// Mara, Efficient GPU Screen-Space Ray Tracing (JCGT 2014), equation 1:
// https://jcgt.org/published/0003/04/04/paper.pdf
// This independently authored ES implementation uses canonical top-left pixels
// for traversal and converts only texture accesses to the GL bottom-left origin.
// Roughness attenuates a mirror estimate; it does not integrate a glossy lobe.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const vec2 FullscreenTriangle[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(FullscreenTriangle[gl_VertexID], 0, 1);
}
#else
uniform mat4 ViewProjection;
uniform mat4 InverseViewProjection;
uniform vec4 Viewport; // extent, reciprocal extent
uniform vec4 Trace; // distance, thickness, normal bias, pixel stride
uniform vec4 Appearance; // intensity, roughness cutoff, edge fade pixels, reserved
uniform uvec4 Options; // phase, maximum steps, shading model, reserved
uniform sampler2D SceneColor;
uniform sampler2D SceneDepth;
uniform sampler2D NormalRoughness;
uniform sampler2D SpecularShininess;
uniform sampler2D DiffuseOpacity;
uniform sampler2D AmbientMetallic;
uniform sampler2D ReflectionRadiance;
out vec4 OutColor;

const float ProjectionEpsilon = 1e-7;
const float BackgroundDepth = 1.0;
const float CameraInteriorDepth = 0.5;
const float SelfExclusionPixels = 2.0;
const float DistanceFadeStart = 0.8;
const float DielectricReflectance = 0.04;
const float HalfFloatMaximum = 65504.0;
const float HalfFloatTopSpacing = 32.0;
const uint PbrShadingModel = 3u;
const uint CompositePhase = 1u;

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
    for (uint step = 0u; step < Options.y; ++step)
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

void main()
{
    ivec2 nativePixel = ivec2(gl_FragCoord.xy);
    if (Options.x == CompositePhase)
    {
        OutColor = vec4(texelFetch(ReflectionRadiance, nativePixel, 0).rgb, 0);
        return;
    }
    OutColor = vec4(0);
    ivec2 pixel = TexturePixel(nativePixel);
    vec3 position, normal, incoming;
    float roughness;
    if (!ReadSurface(pixel, position, normal, incoming, roughness)) return;
    float facing = clamp(dot(normal, -incoming), 0.0, 1.0);
    if (facing <= 0.0 || roughness >= Appearance.y || Appearance.x <= 0.0) return;
    vec3 direction = reflect(incoming, normal);
    vec3 origin = position + normal * Trace.z;
    vec2 hitUV;
    float hitDistance;
    if (!FindHit(origin, direction, hitUV, hitDistance)) return;
    vec3 reflectance = clamp(texelFetch(SpecularShininess, nativePixel, 0).rgb, 0.0, 1.0);
    if (Options.z == PbrShadingModel)
    {
        float metallic = clamp(texelFetch(AmbientMetallic, nativePixel, 0).a, 0.0, 1.0);
        reflectance = mix(reflectance, clamp(texelFetch(DiffuseOpacity, nativePixel, 0).rgb, 0.0, 1.0), metallic);
    }
    float grazing = 1.0 - facing;
    vec3 fresnel = reflectance + (1.0 - reflectance) * grazing * grazing * grazing * grazing * grazing;
    vec2 edgePixels = min(hitUV, 1.0 - hitUV) * Viewport.xy;
    float edgeFade = Appearance.z > 0.0 ? clamp(min(edgePixels.x, edgePixels.y) / Appearance.z, 0.0, 1.0) : 1.0;
    float distanceFade = 1.0 - smoothstep(Trace.x * DistanceFadeStart, Trace.x, hitDistance);
    float roughnessRatio = roughness / Appearance.y;
    float roughnessFade = clamp(1.0 - roughnessRatio * roughnessRatio, 0.0, 1.0);
    vec3 reflected = max(texture(SceneColor, TextureUV(hitUV)).rgb, vec3(0)) * fresnel *
        (Appearance.x * edgeFade * distanceFade * roughnessFade);
    // Leave one binary16 spacing of headroom before fixed-function addition.
    vec3 remaining = max(vec3(HalfFloatMaximum - HalfFloatTopSpacing) - texelFetch(SceneColor, nativePixel, 0).rgb, vec3(0));
    OutColor = vec4(min(reflected, remaining), 0);
}
#endif
