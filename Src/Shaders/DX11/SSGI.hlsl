// DX11 screen-space diffuse GI: cosine hemisphere rays, bilateral filtering,
// validated motion reprojection and feedback-free HDR composition.
// McGuire/Mara, JCGT 2014: https://jcgt.org/published/0003/04/04/
// Cosine sampling: https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection
cbuffer GlobalIlluminationConstants : register(b0)
{
    row_major float4x4 ViewProjection, InverseViewProjection, PreviousInverseViewProjection;
    float4 PreviousDepthPlane, Viewport, Trace, Appearance, Temporal, JitterDelta;
    uint4 Options, History;
};
Texture2D<float4> SceneColor : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> NormalRoughness : register(t2);
Texture2D<float4> DiffuseOpacity : register(t3);
Texture2D<float4> AmbientMetallic : register(t4);
Texture2D<float4> MotionVectors : register(t5);
Texture2D<float4> Estimate : register(t6);
Texture2D<float4> HistoryColor : register(t7);
Texture2D<float> HistoryDepth : register(t8);
Texture2D<float4> HistoryNormal : register(t9);
SamplerState LinearClamp : register(s0);
static const uint TracePass = 0u, SpatialPass = 1u, CompositePass = 3u, PBRModel = 3u;
static const float Pi = 3.14159265358979323846f;
static const float ClearDepth = 1.0f, PixelCenter = 0.5f, InteriorClipDepth = 0.5f;
static const float HomogeneousEpsilon = 1.0e-7f, VectorSquaredEpsilon = 1.0e-12f, IntersectionEpsilon = 1.0e-7f;
static const float SelfHitExclusionPixels = 2.0f, DistanceFadeStartFraction = 0.8f;
static const float DielectricReflectance = 0.04f, MaximumHalfFloatRadiance = 65504.0f;
static const float AbsoluteDepthTolerance = 0.001f, MinimumClipExtent = 0.001f;
static const float MaximumHistorySamples = 32.0f, ParallelBasisThreshold = 0.999f;
static const int FilterRadius = 1;
// PCG RXS-M-XS permutation; these integer coefficients define the generator.
// Jarzynski/Olano, JCGT 2020: https://jcgt.org/published/0009/03/02/
static const uint PcgMultiplier = 747796405u, PcgIncrement = 2891336453u, PcgOutputMultiplier = 277803737u;
static const uint PcgRotationShift = 28u, PcgMinimumShift = 4u, PcgOutputShift = 22u;
static const uint UniformDiscardBits = 8u, UniformMantissaScale = 16777216u;
uint Scramble(uint value)
{
    uint state = value * PcgMultiplier + PcgIncrement;
    uint word = ((state >> ((state >> PcgRotationShift) + PcgMinimumShift)) ^ state) * PcgOutputMultiplier;
    return (word >> PcgOutputShift) ^ word;
}
float RandomUnit(inout uint state)
{
    state = Scramble(state);
    return float(state >> UniformDiscardBits) / float(UniformMantissaScale);
}
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    float2 corner = float2((vertexId << 1u) & 2u, vertexId & 2u);
    ScreenVertex output;
    output.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}

bool UnitVector(inout float3 value)
{
    float squaredLength = dot(value, value);
    if (!all(isfinite(value)) || squaredLength <= VectorSquaredEpsilon) return false;
    value *= rsqrt(squaredLength);
    return true;
}
bool Unproject(float2 uv, float depth, out float3 position)
{
    float4 homogeneous = mul(InverseViewProjection, float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), depth, 1.0f));
    position = 0.0f;
    if (abs(homogeneous.w) < HomogeneousEpsilon) return false;
    position = homogeneous.xyz / homogeneous.w;
    return all(isfinite(position));
}
bool ViewRay(float2 uv, out float3 direction)
{
    float3 nearPoint, interiorPoint;
    direction = 0.0f;
    if (!Unproject(uv, 0.0f, nearPoint) || !Unproject(uv, InteriorClipDepth, interiorPoint)) return false;
    direction = interiorPoint - nearPoint;
    return UnitVector(direction);
}
bool ReadSurface(int2 pixel, out float3 position, out float3 normal)
{
    position = normal = 0.0f;
    if (any(pixel < 0) || any(pixel >= int2(Viewport.xy))) return false;
    float depth = OpaqueDepth.Load(int3(pixel, 0));
    float4 materialNormal = NormalRoughness.Load(int3(pixel, 0));
    normal = materialNormal.xyz;
    return depth < ClearDepth && materialNormal.w >= 0.0f && UnitVector(normal) &&
        Unproject((float2(pixel) + PixelCenter) * Viewport.zw, depth, position);
}

bool ClipPlane(float startDistance, float endDistance, inout float enter, inout float leave)
{
    if (startDistance < 0.0f && endDistance < 0.0f) return false;
    if ((startDistance < 0.0f) != (endDistance < 0.0f))
    {
        float crossing = startDistance / (startDistance - endDistance);
        if (startDistance < 0.0f) enter = max(enter, crossing);
        else leave = min(leave, crossing);
    }
    return enter < leave;
}
bool ClipRay(inout float3 start, inout float3 end)
{
    float4 a = mul(ViewProjection, float4(start, 1.0f));
    float4 b = mul(ViewProjection, float4(end, 1.0f));
    if (!all(isfinite(a)) || !all(isfinite(b))) return false;
    float enter = 0.0f, leave = 1.0f;
    if (!ClipPlane(a.w - HomogeneousEpsilon, b.w - HomogeneousEpsilon, enter, leave) ||
        !ClipPlane(a.w + a.x, b.w + b.x, enter, leave) ||
        !ClipPlane(a.w - a.x, b.w - b.x, enter, leave) ||
        !ClipPlane(a.w + a.y, b.w + b.y, enter, leave) ||
        !ClipPlane(a.w - a.y, b.w - b.y, enter, leave) ||
        !ClipPlane(a.z, b.z, enter, leave) ||
        !ClipPlane(a.w - a.z, b.w - b.z, enter, leave)) return false;
    float3 segment = end - start;
    end = start + segment * leave;
    start += segment * enter;
    return true;
}

bool TraceDiffuseRay(float3 origin, float3 direction, out float2 hitUV, out float hitDistance)
{
    hitUV = 0.0f; hitDistance = 0.0f;
    float3 start = origin, end = origin + direction * Trace.x;
    if (!ClipRay(start, end)) return false;
    float4 h0 = mul(ViewProjection, float4(start, 1.0f));
    float4 h1 = mul(ViewProjection, float4(end, 1.0f));
    float k0 = rcp(h0.w), k1 = rcp(h1.w);
    float3 q0 = start * k0, q1 = end * k1;
    float2 p0 = (h0.xy * k0 * float2(0.5f, -0.5f) + PixelCenter) * Viewport.xy;
    float2 p1 = (h1.xy * k1 * float2(0.5f, -0.5f) + PixelCenter) * Viewport.xy;
    float majorLength = max(abs(p1.x - p0.x), abs(p1.y - p0.y));
    if (majorLength <= SelfHitExclusionPixels) return false;
    float step = Trace.w / majorLength;
    float first = SelfHitExclusionPixels / majorLength;
    [loop] for (uint index = 0; index < Options.z; ++index)
    {
        float begin = first + float(index) * step;
        if (begin >= 1.0f) break;
        float finish = min(begin + step, 1.0f);
        float middle = (begin + finish) * 0.5f;
        int2 pixel = int2(floor(lerp(p0, p1, middle)));
        float3 surface, normal, viewDirection;
        if (!ReadSurface(pixel, surface, normal) || dot(normal, -direction) <= 0.0f ||
            !ViewRay((float2(pixel) + PixelCenter) * Viewport.zw, viewDirection)) continue;
        // Signed world-unit depth against this texel's camera-ray slab. The
        // numerator of depth(t) is linear, so its front-plane root is analytic.
        float numerator0 = dot(q0 - surface * k0, viewDirection);
        float numerator1 = dot(q1 - surface * k1, viewDirection);
        float depthBegin = lerp(numerator0, numerator1, begin) / lerp(k0, k1, begin);
        float depthEnd = lerp(numerator0, numerator1, finish) / lerp(k0, k1, finish);
        if (max(depthBegin, depthEnd) < 0.0f || min(depthBegin, depthEnd) > Trace.y) continue;
        float slope = numerator1 - numerator0;
        float crossing = abs(slope) > IntersectionEpsilon ? -numerator0 / slope : middle;
        crossing = clamp(crossing, begin, finish);
        float2 screenPoint = lerp(p0, p1, crossing);
        int2 refinedPixel = int2(floor(screenPoint));
        float3 refinedSurface, refinedNormal, refinedView;
        if (!ReadSurface(refinedPixel, refinedSurface, refinedNormal) || dot(refinedNormal, -direction) <= 0.0f ||
            !ViewRay((float2(refinedPixel) + PixelCenter) * Viewport.zw, refinedView)) continue;
        float3 rayPoint = lerp(q0, q1, crossing) / lerp(k0, k1, crossing);
        float separation = dot(rayPoint - refinedSurface, refinedView);
        if (separation < -IntersectionEpsilon || separation > Trace.y) continue;
        hitUV = screenPoint * Viewport.zw;
        hitDistance = length(rayPoint - origin);
        return true;
    }
    return false;
}


float4 EstimateBounce(int2 pixel)
{
    float3 position, normal;
    if (!ReadSurface(pixel, position, normal)) return 0;
    float3 diffuse = saturate(DiffuseOpacity.Load(int3(pixel, 0)).rgb);
    if (History.y == PBRModel)
        diffuse *= (1 - saturate(AmbientMetallic.Load(int3(pixel, 0)).w)) * (1 - DielectricReflectance);
    float3 helper = abs(normal.z) < ParallelBasisThreshold ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
    uint state = uint(pixel.y) * uint(Viewport.x) + uint(pixel.x);
    state ^= Scramble(Options.w);
    float3 incident = 0;
    [loop] for (uint ray = 0; ray < Options.y; ++ray)
    {
        float radial = RandomUnit(state), angle = 2 * Pi * RandomUnit(state);
        float radius = sqrt(radial);
        float3 direction = tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle)) + normal * sqrt(1 - radial);
        float2 hitUV; float distance;
        if (!TraceDiffuseRay(position + normal * Trace.z, direction, hitUV, distance)) continue;
        float2 edge = min(hitUV, 1 - hitUV) * Viewport.xy;
        float fade = Appearance.y > 0 ? saturate(min(edge.x, edge.y) / Appearance.y) : 1;
        fade *= 1 - smoothstep(Trace.x * DistanceFadeStartFraction, Trace.x, distance);
        // Fetch one visible receiver: filtering radiance across a depth edge
        // would attribute an occluded bright surface to the ray's hit.
        int2 hitPixel = clamp(int2(hitUV * Viewport.xy), 0, int2(Viewport.xy) - 1);
        incident += max(SceneColor.Load(int3(hitPixel, 0)).rgb, 0) * fade;
    }
    return float4(clamp(diffuse * incident / float(Options.y), 0, MaximumHalfFloatRadiance), 1);
}
float SpatialWeight(int2 center, int2 candidate, float3 position, float3 normal)
{
    float3 otherPosition, otherNormal;
    if (!ReadSurface(candidate, otherPosition, otherNormal) || dot(normal, otherNormal) < Appearance.w) return 0;
    float distance = abs(dot(otherPosition - position, normal));
    float kernelX = candidate.x == center.x ? 2 : 1, kernelY = candidate.y == center.y ? 2 : 1;
    return kernelX * kernelY * exp(-distance / Appearance.z);
}
float4 FilterBounce(int2 pixel)
{
    float3 position, normal;
    if (!ReadSurface(pixel, position, normal)) return 0;
    float3 sum = 0; float weightSum = 0;
    [unroll] for (int y = -FilterRadius; y <= FilterRadius; ++y)
        [unroll] for (int x = -FilterRadius; x <= FilterRadius; ++x)
        {
            int2 neighbor = pixel + int2(x, y);
            float weight = SpatialWeight(pixel, neighbor, position, normal);
            if (weight <= 0) continue;
            sum += max(Estimate.Load(int3(neighbor, 0)).rgb, 0) * weight; weightSum += weight;
        }
    return float4(sum / max(weightSum, VectorSquaredEpsilon), 1);
}
float PreviousDistance(float2 pixel, float depth)
{
    float2 uv = pixel * Viewport.zw;
    float4 homogeneous = mul(PreviousInverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    if (abs(homogeneous.w) <= HomogeneousEpsilon) return -1;
    return dot(PreviousDepthPlane, float4(homogeneous.xyz / homogeneous.w, 1));
}
struct HistoryOutput { float4 Color : SV_TARGET0; float Depth : SV_TARGET1; float4 Normal : SV_TARGET2; };
HistoryOutput PSTemporal(ScreenVertex input)
{
    int2 pixel = int2(input.Position.xy);
    HistoryOutput result;
    result.Color = Estimate.Load(int3(pixel, 0)); result.Depth = OpaqueDepth.Load(int3(pixel, 0));
    result.Normal = NormalRoughness.Load(int3(pixel, 0));
    float3 position, normal;
    if (History.x == 0 || Temporal.x <= 0 || !ReadSurface(pixel, position, normal)) return result;
    float4 motion = MotionVectors.Load(int3(pixel, 0));
    if (motion.w <= 0) return result;
    float2 previous = input.Position.xy - motion.xy - JitterDelta.xy;
    if (any(previous < PixelCenter) || any(previous > Viewport.xy - PixelCenter)) return result;
    float expected = PreviousDistance(previous, 1-motion.z);
    if (!isfinite(expected) || expected < 0) return result;
    float tolerance = max(AbsoluteDepthTolerance, abs(expected) * Temporal.y);
    int2 base = int2(floor(previous - PixelCenter)); float2 fraction = frac(previous - PixelCenter);
    float4 historySum = 0; float historyWeight = 0;
    [unroll] for (int y = 0; y < 2; ++y) [unroll] for (int x = 0; x < 2; ++x)
    {
        int2 candidate = base + int2(x, y);
        if (any(candidate < 0) || any(candidate >= int2(Viewport.xy))) continue;
        float depth = HistoryDepth.Load(int3(candidate, 0));
        float3 oldNormal = HistoryNormal.Load(int3(candidate, 0)).xyz;
        float4 oldColor = HistoryColor.Load(int3(candidate, 0));
        float distance = PreviousDistance(float2(candidate) + PixelCenter, depth);
        if (depth >= ClearDepth || !UnitVector(oldNormal) || dot(normal, oldNormal) < Temporal.z || oldColor.a <= 0 ||
            !isfinite(distance) || abs(distance - expected) > tolerance) continue;
        float weight = (x == 0 ? 1 - fraction.x : fraction.x) * (y == 0 ? 1 - fraction.y : fraction.y);
        historySum += oldColor * weight; historyWeight += weight;
    }
    if (historyWeight <= VectorSquaredEpsilon) return result;
    float4 history = historySum / historyWeight;
    float3 minimumColor = result.Color.rgb, maximumColor = minimumColor;
    [unroll] for (int y = -FilterRadius; y <= FilterRadius; ++y) [unroll] for (int x = -FilterRadius; x <= FilterRadius; ++x)
    {
        int2 candidate = pixel + int2(x, y);
        if (SpatialWeight(pixel, candidate, position, normal) <= 0) continue;
        float3 value = Estimate.Load(int3(candidate, 0)).rgb;
        minimumColor = min(minimumColor, value); maximumColor = max(maximumColor, value);
    }
    float3 center = (minimumColor + maximumColor) * 0.5f;
    float3 extent = max((maximumColor - minimumColor) * 0.5f, MinimumClipExtent);
    float3 delta = history.rgb - center, ratio = abs(delta / extent);
    history.rgb = center + delta / max(1, max(ratio.x, max(ratio.y, ratio.z)));
    float blend = min(Temporal.x, history.a / (history.a + 1));
    result.Color.rgb = clamp(lerp(result.Color.rgb, history.rgb, blend), 0, MaximumHalfFloatRadiance);
    result.Color.a = min(history.a + 1, MaximumHistorySamples);
    return result;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy);
    if (Options.x == TracePass) return EstimateBounce(pixel);
    if (Options.x == SpatialPass) return FilterBounce(pixel);
    float4 scene = SceneColor.Load(int3(pixel, 0));
    float3 indirect = max(Estimate.Load(int3(pixel, 0)).rgb, 0) * Appearance.x;
    return float4(clamp(scene.rgb + indirect, 0, MaximumHalfFloatRadiance), scene.a);
}
