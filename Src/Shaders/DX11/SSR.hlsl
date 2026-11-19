// DX11 spatial screen-space reflections. This native file owns trace/composite.
// Perspective-correct DDA: McGuire and Mara, JCGT 2014, equation 1.
// https://jcgt.org/published/0003/04/04/paper.pdf
// World/w and reciprocal w interpolate along the projected major axis. Each
// interval intersects a finite depth slab, followed by a refined-texel check.
// Visible opaque radiance is a view-dependent proxy for incoming radiance.
// Roughness attenuates this mirror estimate; it does not integrate a GGX lobe.
cbuffer ReflectionConstants : register(b0)
{
    row_major float4x4 ViewProjection;
    row_major float4x4 InverseViewProjection;
    float4 Viewport;
    float4 Trace;
    float4 Appearance;
    uint4 Options;
};
Texture2D<float4> SceneColor : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> NormalRoughness : register(t2);
Texture2D<float4> SpecularShininess : register(t3);
Texture2D<float4> DiffuseOpacity : register(t4);
Texture2D<float4> AmbientMetallic : register(t5);
Texture2D<float4> ReflectionRadiance : register(t6);
SamplerState LinearClamp : register(s0);

static const uint CompositePass = 1u;
static const uint PBRModel = 3u;
static const float ClearDepth = 1.0f;
static const float PixelCenter = 0.5f;
static const float InteriorClipDepth = 0.5f;
static const float HomogeneousEpsilon = 1.0e-7f;
static const float VectorSquaredEpsilon = 1.0e-12f;
static const float IntersectionEpsilon = 1.0e-7f;
static const float SelfHitExclusionPixels = 2.0f;
static const float DistanceFadeStartFraction = 0.8f;
static const float DielectricReflectance = 0.04f;
static const float SchlickExponent = 5.0f;
static const float MaximumHalfFloatRadiance = 65504.0f;
static const float MaximumHalfFloatSpacing = 32.0f;

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

bool TraceReflection(float3 origin, float3 direction, out float2 hitUV, out float hitDistance)
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
    [loop] for (uint index = 0; index < Options.y; ++index)
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

float4 PSMain(ScreenVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy);
    if (Options.x == CompositePass) return float4(ReflectionRadiance.Load(int3(pixel, 0)).rgb, 0.0f);
    if (Appearance.x <= 0.0f) return 0.0f;
    float3 position, normal, incoming;
    if (!ReadSurface(pixel, position, normal) ||
        !ViewRay((float2(pixel) + PixelCenter) * Viewport.zw, incoming)) return 0.0f;
    float roughness = NormalRoughness.Load(int3(pixel, 0)).w;
    float facing = dot(normal, -incoming);
    if (roughness >= Appearance.y || facing <= 0.0f) return 0.0f;
    float3 direction = reflect(incoming, normal);
    float2 hitUV; float distance;
    if (!TraceReflection(position + normal * Trace.z, direction, hitUV, distance)) return 0.0f;
    float3 f0 = saturate(SpecularShininess.Load(int3(pixel, 0)).rgb);
    if (Options.z == PBRModel)
        f0 = lerp(f0, saturate(DiffuseOpacity.Load(int3(pixel, 0)).rgb),
            saturate(AmbientMetallic.Load(int3(pixel, 0)).w));
    float3 fresnel = f0 + (1.0f - f0) * pow(1.0f - saturate(facing), SchlickExponent);
    float2 edge = min(hitUV, 1.0f - hitUV) * Viewport.xy;
    float edgeFade = Appearance.z > 0.0f ? saturate(min(edge.x, edge.y) / Appearance.z) : 1.0f;
    float distanceFade = 1.0f - smoothstep(Trace.x * DistanceFadeStartFraction, Trace.x, distance);
    float relativeRoughness = roughness / Appearance.y;
    float weight = Appearance.x * edgeFade * distanceFade * saturate(1.0f - relativeRoughness * relativeRoughness);
    float3 radiance = max(SceneColor.SampleLevel(LinearClamp, hitUV, 0).rgb, 0.0f) * fresnel * weight;
    // Leave one binary16 spacing for rounding the intermediate contribution.
    // The later additive blend must remain finite even near HDR saturation.
    float3 headroom = max(MaximumHalfFloatRadiance - MaximumHalfFloatSpacing - SceneColor.Load(int3(pixel, 0)).rgb, 0.0f);
    return float4(min(radiance, headroom), 0.0f);
}
