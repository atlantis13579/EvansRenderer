#include <metal_stdlib>
using namespace metal;
// Mittring, Finding Next Gen - CryEngine 2, SIGGRAPH 2007 (8.5.4.3).
// A rotated cosine hemisphere tests screen-space depth. Separable bilateral
// filtering rejects surface boundaries; composition subtracts lost ambient only.
// All depth/material/AO accesses are integer texel reads with no filtering.
constant uint RawVisibility = 0;
constant uint HorizontalFilter = 1;
constant uint VerticalFilter = 2;
constant uint AmbientComposite = 3;
constant int HemisphereSampleCount = 24;
constant int FilterRadius = 2;
constant float Pi = 3.141592653589793;
constant float NormalEpsilon = 1.0e-12;
constant float ProjectionEpsilon = 1.0e-7;
constant float ParallelThreshold = 0.999;
constant float InnerKernelFraction = 0.1;
constant float RangeFadeFraction = 0.5;
constant float SpatialStandardDeviation = 1.5;
constant float DepthStandardDeviationFraction = 0.25;
constant float NormalWeightPower = 8.0;
constant float MinimumFilterWeight = 1.0e-7;
constant float RotationScale = 52.9829189;
constant float2 RotationDirection(0.06711056, 0.00583715);
struct SsaoUniforms
{
    float4 ViewProjection[4];
    float4 InverseViewProjection[4];
    float4 ViewportRadiusBias;
    float4 ForwardIntensity;
    uint4 Options;
};
struct SsaoVertex { float4 Position [[position]]; };
vertex SsaoVertex SSAOVertexMain(uint vertexId [[vertex_id]])
{
    const float FarCorner = 3.0;
    float2 corner = vertexId == 0 ? float2(-1.0) :
        vertexId == 1 ? float2(FarCorner, -1.0) : float2(-1.0, FarCorner);
    return {float4(corner, 0.0, 1.0)};
}

float4 TransformRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value), dot(rows[2], value), dot(rows[3], value));
}

bool Reconstruct(uint2 pixel, float depth, constant SsaoUniforms& uniforms, thread float3& position)
{
    float2 uv = (float2(pixel) + 0.5) * uniforms.ViewportRadiusBias.xy;
    float4 homogeneous = TransformRows(uniforms.InverseViewProjection,
        float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0));
    position = float3(0.0);
    if (abs(homogeneous.w) < ProjectionEpsilon) return false;
    position = homogeneous.xyz / homogeneous.w;
    return true;
}

bool ReadSurface(uint2 pixel, constant SsaoUniforms& uniforms,
    depth2d<float, access::read> depthImage, texture2d<float, access::read> normalImage,
    thread float3& position, thread float3& normal)
{
    float depth = depthImage.read(pixel);
    float4 storedNormal = normalImage.read(pixel);
    float squaredLength = dot(storedNormal.xyz, storedNormal.xyz);
    position = float3(0.0);
    normal = float3(0.0, 1.0, 0.0);
    if (depth >= 1.0 || storedNormal.w < 0.0 || squaredLength <= NormalEpsilon) return false;
    normal = storedNormal.xyz * rsqrt(squaredLength);
    return Reconstruct(pixel, depth, uniforms, position);
}

float EstimateVisibility(uint2 pixel, float3 center, float3 normal,
    constant SsaoUniforms& uniforms, depth2d<float, access::read> depthImage)
{
    const float GoldenAzimuth = Pi * (3.0 - sqrt(5.0));
    float2 extent(depthImage.get_width(), depthImage.get_height());
    float rotation = 2.0 * Pi * fract(RotationScale * fract(dot(float2(pixel), RotationDirection)));
    float3 helper = abs(normal.y) < ParallelThreshold ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(helper, normal));
    float3 bitangent = cross(normal, tangent);
    float radius = uniforms.ViewportRadiusBias.z;
    float sum = 0.0;
    for (int sampleIndex = 0; sampleIndex < HemisphereSampleCount; ++sampleIndex)
    {
        float xi = (float(sampleIndex) + 0.5) / float(HemisphereSampleCount);
        float angle = rotation + float(sampleIndex) * GoldenAzimuth;
        float diskRadius = sqrt(xi);
        float3 direction = tangent * (diskRadius * cos(angle)) + bitangent * (diskRadius * sin(angle)) +
            normal * sqrt(1.0 - xi);
        float radialFraction = mix(InnerKernelFraction, 1.0, xi * xi);
        float3 samplePosition = center + direction * radius * radialFraction;
        float4 clip = TransformRows(uniforms.ViewProjection, float4(samplePosition, 1.0));
        if (clip.w <= ProjectionEpsilon) continue;
        float3 projected = clip.xyz / clip.w;
        float2 uv(projected.x * 0.5 + 0.5, -projected.y * 0.5 + 0.5);
        if (any(uv < 0.0) || any(uv >= 1.0) || projected.z < 0.0 || projected.z > 1.0) continue;
        uint2 neighbor = uint2(floor(uv * extent));
        float depth = depthImage.read(neighbor);
        if (depth >= 1.0) continue;
        float3 scenePosition;
        if (!Reconstruct(neighbor, depth, uniforms, scenePosition)) continue;
        float separation = dot(samplePosition - scenePosition, uniforms.ForwardIntensity.xyz);
        float weight = 1.0 - smoothstep(radius * RangeFadeFraction, radius, distance(scenePosition, center));
        if (separation > uniforms.ViewportRadiusBias.w) sum += weight;
    }
    // Offscreen samples count as open space, keeping silhouettes from darkening.
    return saturate(1.0 - uniforms.ForwardIntensity.w * sum / float(HemisphereSampleCount));
}

float FilterVisibility(uint2 pixel, float3 center, float3 normal, constant SsaoUniforms& uniforms,
    depth2d<float, access::read> depthImage, texture2d<float, access::read> normalImage,
    texture2d<float, access::read> previousVisibility)
{
    int2 extent(depthImage.get_width(), depthImage.get_height());
    int2 direction = uniforms.Options.x == HorizontalFilter ? int2(1, 0) : int2(0, 1);
    float depthSigma = uniforms.ViewportRadiusBias.z * DepthStandardDeviationFraction;
    float weightedVisibility = 0.0;
    float weightSum = 0.0;
    for (int offset = -FilterRadius; offset <= FilterRadius; ++offset)
    {
        uint2 neighbor = uint2(clamp(int2(pixel) + direction * offset, int2(0), extent - 1));
        float3 neighborPosition, neighborNormal;
        if (!ReadSurface(neighbor, uniforms, depthImage, normalImage, neighborPosition, neighborNormal)) continue;
        float depthDelta = dot(neighborPosition - center, uniforms.ForwardIntensity.xyz);
        float spatialWeight = exp(-float(offset * offset) / (2.0 * SpatialStandardDeviation * SpatialStandardDeviation));
        float depthWeight = exp(-depthDelta * depthDelta / (2.0 * depthSigma * depthSigma));
        float normalWeight = pow(saturate(dot(normal, neighborNormal)), NormalWeightPower);
        float weight = spatialWeight * depthWeight * normalWeight;
        weightedVisibility += previousVisibility.read(neighbor).r * weight;
        weightSum += weight;
    }
    return weightSum > MinimumFilterWeight ? saturate(weightedVisibility / weightSum) : 1.0;
}

fragment float4 SSAOFragmentMain(SsaoVertex input [[stage_in]],
    constant SsaoUniforms& uniforms [[buffer(0)]], depth2d<float, access::read> opaqueDepth [[texture(0)]],
    texture2d<float, access::read> normalRoughness [[texture(1)]],
    texture2d<float, access::read> ambientMetallic [[texture(2)]],
    texture2d<float, access::read> previousAO [[texture(3)]])
{
    // Zero strength must survive the half-float intermediates as an exact identity.
    if (uniforms.ForwardIntensity.w <= 0.0)
        return uniforms.Options.x == AmbientComposite ? float4(0.0) : float4(1.0, 0.0, 0.0, 1.0);
    uint2 pixel = uint2(input.Position.xy);
    float3 position, normal;
    bool valid = ReadSurface(pixel, uniforms, opaqueDepth, normalRoughness, position, normal);
    if (uniforms.Options.x == RawVisibility)
    {
        float visibility = valid ? EstimateVisibility(pixel, position, normal, uniforms, opaqueDepth) : 1.0;
        return float4(visibility, 0.0, 0.0, 1.0);
    }
    if (uniforms.Options.x == HorizontalFilter || uniforms.Options.x == VerticalFilter)
    {
        float visibility = valid ? FilterVisibility(pixel, position, normal, uniforms, opaqueDepth, normalRoughness, previousAO) : 1.0;
        return float4(visibility, 0.0, 0.0, 1.0);
    }
    if (!valid) return float4(0.0);
    float visibility = saturate(previousAO.read(pixel).r);
    float3 ambient = max(ambientMetallic.read(pixel).rgb, 0.0);
    return float4(ambient * (1.0 - visibility), 0.0);
}
