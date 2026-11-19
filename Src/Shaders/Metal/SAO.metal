#include <metal_stdlib>
using namespace metal;
// McGuire, Mara and Luebke, Scalable Ambient Obscurance, HPG 2012.
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Scalable-Ambient-Obscurance/McGuire12SAO.pdf
// Updated author appendix: https://casual-effects.com/research/McGuire2012SAO/McGuire12SAO.pdf
// A near-relative R32F depth hierarchy amortizes reconstruction and keeps nine
// spiral gathers cache-friendly. The exact selected child supplies its own ray.
// One-sided depth differences are an edge-aware variant of the paper's normals.
constant uint EstimatePass = 0;
constant uint HorizontalPass = 1;
constant uint VerticalPass = 2;
constant uint CompositePass = 3;
constant uint DepthReducePass = 5;
constant int SpiralSamples = 9;
constant int SpiralTurns = 7;
constant int BlurRadius = 2;
constant float Pi = 3.141592653589793;
constant float BackgroundDepth = -1.0;
constant float MinimumHomogeneousW = 1.0e-7;
constant float MinimumRayDenominator = 1.0e-12;
constant float MinimumSquaredLength = 1.0e-12;
constant float InteriorClipDepth = 0.5;
constant float MipRadiusIncrement = 16.0;
constant float ObscuranceNormalization = 5.0;
constant float ObscuranceDistanceEpsilon = 0.01;
constant float ObscuranceSquaredEpsilon = ObscuranceDistanceEpsilon * ObscuranceDistanceEpsilon;
constant float SpatialStandardDeviation = 1.5;
constant float DepthStandardDeviationFraction = 0.25;
constant float MinimumFilterWeight = 1.0e-7;
constant float RotationScale = 52.9829189;
constant float2 RotationDirection(0.06711056, 0.00583715);

struct ScalableUniforms
{
    float4 ViewProjection[4];
    float4 InverseViewProjection[4];
    float4 ViewportRadiusBias;
    float4 ForwardIntensity;
    uint4 Options;
};
struct ScalableVertex { float4 Position [[position]]; };
vertex ScalableVertex SAOVertexMain(uint id [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 corner = id == 0 ? float2(-1.0) :
        id == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(corner, 0.0, 1.0)};
}
float4 ApplyRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value), dot(rows[2], value), dot(rows[3], value));
}
bool IsValidDepth(float depth) { return depth >= 0.0 && isfinite(depth); }
bool NormalizeVector(thread float3& value)
{
    if (!all(isfinite(value))) return false;
    float scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if (scale <= 0.0) return false;
    value /= scale;
    value *= rsqrt(dot(value, value));
    return true;
}
bool Unproject(float2 pixel, float clipDepth, constant ScalableUniforms& uniforms, thread float3& position)
{
    float2 uv = (pixel + 0.5) * uniforms.ViewportRadiusBias.xy;
    float4 world = ApplyRows(uniforms.InverseViewProjection,
        float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, clipDepth, 1.0));
    position = float3(0.0);
    if (abs(world.w) < MinimumHomogeneousW) return false;
    position = world.xyz / world.w;
    return all(isfinite(position));
}
bool Reconstruct(float2 pixel, float depth, constant ScalableUniforms& uniforms, thread float3& position)
{
    position = float3(0.0);
    if (!IsValidDepth(depth)) return false;
    float3 nearPoint, middlePoint;
    if (!Unproject(pixel, 0.0, uniforms, nearPoint) ||
        !Unproject(pixel, InteriorClipDepth, uniforms, middlePoint)) return false;
    float3 ray = middlePoint - nearPoint;
    float denominator = dot(ray, uniforms.ForwardIntensity.xyz);
    if (!isfinite(denominator) || abs(denominator) <= MinimumRayDenominator) return false;
    position = nearPoint + ray * (depth / denominator);
    return all(isfinite(position));
}
bool IsLit(int2 pixel, texture2d<float, access::read> depth, texture2d<float, access::read> material)
{
    int2 extent(depth.get_width(), depth.get_height());
    return all(pixel >= 0) && all(pixel < extent) && material.read(uint2(pixel)).w >= 0.0;
}
bool ReadDepthSurface(int2 pixel, constant ScalableUniforms& uniforms,
    texture2d<float, access::read> hierarchy,
    thread float& depth, thread float3& position)
{
    depth = BackgroundDepth;
    position = float3(0.0);
    int2 extent(hierarchy.get_width(), hierarchy.get_height());
    if (any(pixel < 0) || any(pixel >= extent)) return false;
    depth = hierarchy.read(uint2(pixel)).r;
    return Reconstruct(float2(pixel), depth, uniforms, position);
}
bool ReadSurface(int2 pixel, constant ScalableUniforms& uniforms,
    texture2d<float, access::read> hierarchy, texture2d<float, access::read> material,
    thread float& depth, thread float3& position)
{
    depth = BackgroundDepth;
    position = float3(0.0);
    return IsLit(pixel, hierarchy, material) && ReadDepthSurface(pixel, uniforms, hierarchy, depth, position);
}
int2 SelectedChild(int2 parent, uint sourceMip, texture2d<float, access::read> hierarchy)
{
    int2 parity((parent.y & 1) ^ 1, (parent.x & 1) ^ 1);
    int2 extent(hierarchy.get_width(sourceMip), hierarchy.get_height(sourceMip));
    return clamp(parent * 2 + parity, int2(0), extent - 1);
}
int2 RepresentativePixel(int2 pixel, int mip, texture2d<float, access::read> hierarchy)
{
    // Reverse every rotated-grid selection; use actual floor-sized mip extents.
    for (int sourceMip = mip - 1; sourceMip >= 0; --sourceMip)
        pixel = SelectedChild(pixel, uint(sourceMip), hierarchy);
    return pixel;
}
bool GeometricNormal(int2 pixel, float centerDepth, float3 center, constant ScalableUniforms& uniforms,
    texture2d<float, access::read> hierarchy, thread float3& normal)
{
    float leftDepth, rightDepth, upDepth, downDepth;
    float3 left, right, up, down;
    // Normal reconstruction depends on depth even when a neighbor is an unlit wire.
    bool leftValid = ReadDepthSurface(pixel + int2(-1, 0), uniforms, hierarchy, leftDepth, left);
    bool rightValid = ReadDepthSurface(pixel + int2(1, 0), uniforms, hierarchy, rightDepth, right);
    bool upValid = ReadDepthSurface(pixel + int2(0, -1), uniforms, hierarchy, upDepth, up);
    bool downValid = ReadDepthSurface(pixel + int2(0, 1), uniforms, hierarchy, downDepth, down);
    normal = float3(0.0);
    if ((!leftValid && !rightValid) || (!upValid && !downValid)) return false;
    bool useRight = rightValid && (!leftValid || abs(rightDepth - centerDepth) < abs(leftDepth - centerDepth));
    bool useDown = downValid && (!upValid || abs(downDepth - centerDepth) < abs(upDepth - centerDepth));
    float3 horizontal = useRight ? right - center : center - left;
    float3 vertical = useDown ? down - center : center - up;
    // Normalize first so degeneracy tests measure an angle, not a pixel's area.
    if (!NormalizeVector(horizontal) || !NormalizeVector(vertical)) return false;
    normal = cross(vertical, horizontal);
    float normalSquared = dot(normal, normal);
    if (normalSquared <= MinimumSquaredLength || !NormalizeVector(normal)) return false;
    if (dot(normal, -uniforms.ForwardIntensity.xyz) < 0.0) normal = -normal;
    return all(isfinite(normal));
}
float EstimateVisibility(int2 pixel, float depth, float3 center, float3 normal,
    constant ScalableUniforms& uniforms, texture2d<float, access::read> hierarchy)
{
    int2 extent(hierarchy.get_width(), hierarchy.get_height());
    float rotation = 2.0 * Pi * fract(RotationScale * fract(dot(float2(pixel), RotationDirection)));
    float radius = uniforms.ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    float obscurance = 0.0;
    for (int sampleIndex = 0; sampleIndex < SpiralSamples; ++sampleIndex)
    {
        float alpha = (float(sampleIndex) + 0.5) / float(SpiralSamples);
        float angle = 2.0 * Pi * float(SpiralTurns) * alpha + rotation;
        float3 adjacent;
        if (!Reconstruct(float2(pixel) + float2(cos(angle), sin(angle)), depth, uniforms, adjacent)) continue;
        float3 direction = adjacent - center;
        direction -= uniforms.ForwardIntensity.xyz * dot(direction, uniforms.ForwardIntensity.xyz);
        if (!NormalizeVector(direction)) continue;
        float4 endpoint = ApplyRows(uniforms.ViewProjection, float4(center + direction * radius, 1.0));
        if (!all(isfinite(endpoint)) || endpoint.w <= MinimumHomogeneousW) continue;
        float2 projected = endpoint.xy / endpoint.w;
        float2 uv(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
        float2 tapOffset = (uv * float2(extent) - float2(pixel) - 0.5) * alpha;
        float2 snapped = floor(float2(pixel) + tapOffset + 0.5);
        if (!all(isfinite(snapped)) || any(snapped < 0.0) || any(snapped >= float2(extent))) continue;
        int mip = int(clamp(floor(log2(max(length(tapOffset), 1.0) / MipRadiusIncrement)), 0.0, float(uniforms.Options.z)));
        int2 mipExtent(hierarchy.get_width(uint(mip)), hierarchy.get_height(uint(mip)));
        int2 mipPixel = clamp(int2(snapped) >> mip, int2(0), mipExtent - 1);
        float sampleDepth = hierarchy.read(uint2(mipPixel), uint(mip)).r;
        int2 originalPixel = RepresentativePixel(mipPixel, mip, hierarchy);
        float3 sampled;
        if (!Reconstruct(float2(originalPixel), sampleDepth, uniforms, sampled)) continue;
        float3 displacement = sampled - center;
        float distanceSquared = dot(displacement, displacement);
        if (distanceSquared >= radiusSquared) continue;
        float falloff = max(1.0 - distanceSquared / radiusSquared, 0.0);
        float facing = max(dot(displacement, normal) - uniforms.ViewportRadiusBias.w, 0.0);
        obscurance += falloff * falloff * falloff * facing / (distanceSquared + ObscuranceSquaredEpsilon);
    }
    return saturate(1.0 - ObscuranceNormalization * uniforms.ForwardIntensity.w * obscurance / float(SpiralSamples));
}
float FilterVisibility(int2 pixel, float depth, constant ScalableUniforms& uniforms,
    texture2d<float, access::read> hierarchy, texture2d<float, access::read> material,
    texture2d<float, access::read> previous)
{
    int2 extent(hierarchy.get_width(), hierarchy.get_height());
    int2 axis = uniforms.Options.x == HorizontalPass ? int2(1, 0) : int2(0, 1);
    float depthSigma = uniforms.ViewportRadiusBias.z * DepthStandardDeviationFraction;
    float weightedVisibility = 0.0;
    float weightSum = 0.0;
    for (int offset = -BlurRadius; offset <= BlurRadius; ++offset)
    {
        int2 neighbor = clamp(pixel + axis * offset, int2(0), extent - 1);
        if (!IsLit(neighbor, hierarchy, material)) continue;
        float neighborDepth = hierarchy.read(uint2(neighbor)).r;
        if (!IsValidDepth(neighborDepth)) continue;
        float delta = neighborDepth - depth;
        float spatialWeight = exp(-float(offset * offset) / (2.0 * SpatialStandardDeviation * SpatialStandardDeviation));
        float depthWeight = exp(-delta * delta / (2.0 * depthSigma * depthSigma));
        float weight = spatialWeight * depthWeight;
        weightedVisibility += previous.read(uint2(neighbor)).r * weight;
        weightSum += weight;
    }
    return weightSum > MinimumFilterWeight ? saturate(weightedVisibility / weightSum) : 1.0;
}

// A separate entry keeps Metal's depth2d binding distinct from R32F texture2d.
fragment float4 SAODepthFragmentMain(ScalableVertex input [[stage_in]],
    constant ScalableUniforms& uniforms [[buffer(0)]], depth2d<float, access::read> hardwareDepth [[texture(0)]])
{
    uint2 pixel = uint2(input.Position.xy);
    float depth = hardwareDepth.read(pixel);
    float3 world, nearPoint;
    if (depth >= 1.0 || !Unproject(float2(pixel), depth, uniforms, world) ||
        !Unproject(float2(pixel), 0.0, uniforms, nearPoint)) return float4(BackgroundDepth, 0.0, 0.0, 1.0);
    float linearDepth = dot(world - nearPoint, uniforms.ForwardIntensity.xyz);
    return float4(isfinite(linearDepth) ? max(linearDepth, 0.0) : BackgroundDepth, 0.0, 0.0, 1.0);
}
fragment float4 SAOFragmentMain(ScalableVertex input [[stage_in]],
    constant ScalableUniforms& uniforms [[buffer(0)]], texture2d<float, access::read> hierarchy [[texture(0)]],
    texture2d<float, access::read> materialNormal [[texture(1)]],
    texture2d<float, access::read> ambient [[texture(2)]], texture2d<float, access::read> previousAO [[texture(3)]])
{
    int2 pixel = int2(input.Position.xy);
    if (uniforms.Options.x == DepthReducePass)
    {
        uint sourceMip = uniforms.Options.y - 1;
        int2 selected = SelectedChild(pixel, sourceMip, hierarchy);
        return float4(hierarchy.read(uint2(selected), sourceMip).r, 0.0, 0.0, 1.0);
    }
    if (uniforms.ForwardIntensity.w <= 0.0)
        return uniforms.Options.x == CompositePass ? float4(0.0) : float4(1.0, 0.0, 0.0, 1.0);
    float depth;
    float3 center;
    bool valid = ReadSurface(pixel, uniforms, hierarchy, materialNormal, depth, center);
    if (uniforms.Options.x == EstimatePass)
    {
        float3 normal;
        float visibility = valid && GeometricNormal(pixel, depth, center, uniforms, hierarchy, normal)
            ? EstimateVisibility(pixel, depth, center, normal, uniforms, hierarchy) : 1.0;
        return float4(visibility, 0.0, 0.0, 1.0);
    }
    if (uniforms.Options.x == HorizontalPass || uniforms.Options.x == VerticalPass)
        return float4(valid ? FilterVisibility(pixel, depth, uniforms, hierarchy, materialNormal, previousAO) : 1.0, 0.0, 0.0, 1.0);
    if (!valid) return float4(0.0);
    float visibility = saturate(previousAO.read(uint2(pixel)).r);
    return float4(max(ambient.read(uint2(pixel)).rgb, 0.0) * (1.0 - visibility), 0.0);
}
