#version 450
// Independently authored Vulkan SSAO: hemisphere visibility, two bilateral
// filters, then ambient-only subtraction. No external noise texture is needed.
// Mittring, "Finding Next Gen - CryEngine 2" (SIGGRAPH 2007), section 8.5.4.3.
// https://advances.realtimerendering.com/s2007/
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform OcclusionConstants
{
    mat4 ViewProjection;
    mat4 InverseViewProjection;
    vec4 ViewportRadiusBias;
    vec4 ForwardIntensity;
    uvec4 Options;
} Occlusion;
layout(set = 0, binding = 1) uniform texture2D OpaqueDepth;
layout(set = 0, binding = 2) uniform texture2D NormalRoughness;
layout(set = 0, binding = 3) uniform texture2D AmbientMetallic;
layout(set = 0, binding = 4) uniform texture2D PreviousVisibility;
layout(set = 0, binding = 5) uniform sampler PointSampler;
layout(location = 0) out vec4 OutValue;

const uint HemispherePass = 0u;
const uint HorizontalPass = 1u;
const uint VerticalPass = 2u;
const uint CompositePass = 3u;
const int HemisphereSamples = 24;
const int BlurRadius = 2;
const float Pi = 3.141592653589793;
const float GoldenAngle = Pi * (3.0 - sqrt(5.0));
const float ClearDepth = 1.0;
const float PixelCenter = 0.5;
const float HomogeneousEpsilon = 1.0e-7;
const float NormalizeEpsilon = 1.0e-12;
const float ParallelAxisLimit = 0.999;
const float MinimumRadialFraction = 0.1;
const float RangeFadeStart = 0.5;
const float SpatialSigma = 1.5;
const float RelativeDepthSigma = 0.25;
const float NormalWeightExponent = 8.0;
const float WeightEpsilon = 1.0e-7;
const vec2 RotationHashDirection = vec2(0.06711056, 0.00583715);
const float RotationHashScale = 52.9829189;

vec3 UnitDirection(vec3 value)
{
    return value * inversesqrt(max(dot(value, value), NormalizeEpsilon));
}
float DepthAt(ivec2 pixel)
{
    return texelFetch(sampler2D(OpaqueDepth, PointSampler), pixel, 0).r;
}
vec4 NormalAt(ivec2 pixel)
{
    return texelFetch(sampler2D(NormalRoughness, PointSampler), pixel, 0);
}
bool HasLitSurface(float depth, vec4 normal)
{
    return depth < ClearDepth && normal.w >= 0.0 && dot(normal.xyz, normal.xyz) > NormalizeEpsilon;
}
bool PositionAt(ivec2 pixel, float depth, out vec3 position)
{
    vec2 uv = (vec2(pixel) + PixelCenter) * Occlusion.ViewportRadiusBias.xy;
    // Geometry flips clip Y for Vulkan's positive viewport height; invert that
    // flip before applying the CPU camera's inverse view-projection matrix.
    vec2 cameraNdc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec4 world = Occlusion.InverseViewProjection * vec4(cameraNdc, depth, 1.0);
    position = vec3(0.0);
    if (abs(world.w) < HomogeneousEpsilon) return false;
    position = world.xyz / world.w;
    return true;
}
float HemisphereVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    float radius = Occlusion.ViewportRadiusBias.z;
    float bias = Occlusion.ViewportRadiusBias.w;
    vec3 helper = abs(normal.y) < ParallelAxisLimit ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = UnitDirection(cross(helper, normal));
    vec3 bitangent = cross(normal, tangent);
    // Integer top-left pixel coordinates keep the sampling phase API-independent.
    float rotation = fract(RotationHashScale * fract(dot(vec2(pixel), RotationHashDirection))) * 2.0 * Pi;
    ivec2 extent = textureSize(sampler2D(OpaqueDepth, PointSampler), 0);
    float blocked = 0.0;
    for (int index = 0; index < HemisphereSamples; ++index)
    {
        float xi = (float(index) + PixelCenter) / float(HemisphereSamples);
        float angle = float(index) * GoldenAngle + rotation;
        float diskRadius = sqrt(xi);
        vec3 local = vec3(cos(angle) * diskRadius, sin(angle) * diskRadius, sqrt(1.0 - xi));
        float radialFraction = mix(MinimumRadialFraction, 1.0, xi * xi);
        vec3 samplePosition = position + radius * radialFraction *
            (tangent * local.x + bitangent * local.y + normal * local.z);
        vec4 projected = Occlusion.ViewProjection * vec4(samplePosition, 1.0);
        if (projected.w <= HomogeneousEpsilon) continue;
        vec3 ndc = projected.xyz / projected.w;
        vec2 uv = vec2(ndc.x * PixelCenter + PixelCenter, PixelCenter - ndc.y * PixelCenter);
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0))) || ndc.z < 0.0 || ndc.z > ClearDepth) continue;
        ivec2 neighbor = ivec2(uv * vec2(extent));
        float depth = DepthAt(neighbor);
        if (depth >= ClearDepth) continue;
        vec3 scenePosition;
        if (!PositionAt(neighbor, depth, scenePosition)) continue;
        float distanceToCenter = length(scenePosition - position);
        bool nearerThanSample = dot(samplePosition - scenePosition, Occlusion.ForwardIntensity.xyz) > bias;
        if (nearerThanSample && distanceToCenter < radius)
            blocked += 1.0 - smoothstep(RangeFadeStart * radius, radius, distanceToCenter);
    }
    return clamp(1.0 - Occlusion.ForwardIntensity.w * blocked / float(HemisphereSamples), 0.0, 1.0);
}
float FilterVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(sampler2D(OpaqueDepth, PointSampler), 0);
    ivec2 axis = Occlusion.Options.x == HorizontalPass ? ivec2(1, 0) : ivec2(0, 1);
    float depthSigma = Occlusion.ViewportRadiusBias.z * RelativeDepthSigma;
    float weightedVisibility = 0.0;
    float totalWeight = 0.0;
    for (int offset = -BlurRadius; offset <= BlurRadius; ++offset)
    {
        ivec2 neighbor = clamp(pixel + axis * offset, ivec2(0), extent - 1);
        float depth = DepthAt(neighbor);
        vec4 neighborNormal = NormalAt(neighbor);
        if (!HasLitSurface(depth, neighborNormal)) continue;
        vec3 neighborPosition;
        if (!PositionAt(neighbor, depth, neighborPosition)) continue;
        float viewDistance = dot(neighborPosition - position, Occlusion.ForwardIntensity.xyz);
        float spatialDistance = float(offset) / SpatialSigma;
        float normalizedDepth = viewDistance / depthSigma;
        float weight = exp(-PixelCenter * (spatialDistance * spatialDistance + normalizedDepth * normalizedDepth));
        weight *= pow(clamp(dot(normal, UnitDirection(neighborNormal.xyz)), 0.0, 1.0), NormalWeightExponent);
        weightedVisibility += texelFetch(sampler2D(PreviousVisibility, PointSampler), neighbor, 0).r * weight;
        totalWeight += weight;
    }
    return totalWeight > WeightEpsilon ? weightedVisibility / totalWeight : 1.0;
}
void main()
{
    // Zero strength is an exact identity, including after half-float storage.
    // Avoid approximate bilateral division perturbing a uniform visibility of one.
    if (Occlusion.ForwardIntensity.w <= 0.0)
    {
        OutValue = Occlusion.Options.x == CompositePass ? vec4(0.0) : vec4(1.0);
        return;
    }
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float depth = DepthAt(pixel);
    vec4 normal = NormalAt(pixel);
    vec3 position = vec3(0.0);
    bool lit = HasLitSurface(depth, normal) && PositionAt(pixel, depth, position);
    if (Occlusion.Options.x == CompositePass)
    {
        float visibility = lit ? clamp(texelFetch(sampler2D(PreviousVisibility, PointSampler), pixel, 0).r, 0.0, 1.0) : 1.0;
        vec3 ambient = texelFetch(sampler2D(AmbientMetallic, PointSampler), pixel, 0).rgb;
        // Reverse-subtract RGB blending removes only lost ambient. The target
        // is loaded, never sampled, and its alpha write mask is disabled.
        OutValue = vec4(ambient * (1.0 - visibility), 0.0);
        return;
    }
    float visibility = 1.0;
    if (lit)
    {
        vec3 direction = UnitDirection(normal.xyz);
        visibility = Occlusion.Options.x == HemispherePass
            ? HemisphereVisibility(pixel, position, direction) : FilterVisibility(pixel, position, direction);
    }
    OutValue = vec4(visibility, visibility, visibility, 1.0);
}
#endif
