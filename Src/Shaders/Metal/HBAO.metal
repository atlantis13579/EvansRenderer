#include <metal_stdlib>
using namespace metal;
// Bavoil and Sainz, Image-Space Horizon-Based Ambient Occlusion, SIGGRAPH 2008.
// https://developer.download.nvidia.com/presentations/2008/SIGGRAPH/HBAO_SIG08b.pdf
// Quadratic falloff: https://developer.download.nvidia.com/SDK/10.5/direct3d/Source/ScreenSpaceAO/doc/ScreenSpaceAO.pdf
// Eight depth marches accumulate distance-weighted increases in the horizon.
// Correcting each snapped horizon against the local tangent avoids planar acne.
// Grazing/back-facing geometry remains a single-layer heightfield approximation.
constant uint EstimateVisibilityPass = 0;
constant uint HorizontalFilterPass = 1;
constant uint VerticalFilterPass = 2;
constant uint CompositePass = 3;
constant int DirectionCount = 8;
constant int StepsPerDirection = 8;
constant int FilterRadius = 2;
constant float Pi = 3.141592653589793;
constant float SquaredLengthEpsilon = 1.0e-12;
constant float HomogeneousEpsilon = 1.0e-7;
constant float TangentDenominatorEpsilon = 1.0e-6;
constant float SpatialStandardDeviation = 1.5;
constant float DepthStandardDeviationFraction = 0.25;
constant float NormalWeightExponent = 8.0;
constant float MinimumFilterWeight = 1.0e-7;
constant float RotationScale = 52.9829189;
constant float2 RotationDirection(0.06711056, 0.00583715);

struct HorizonUniforms
{
    float4 ViewProjection[4];
    float4 InverseViewProjection[4];
    float4 ViewportRadiusBias;
    float4 ForwardIntensity;
    uint4 Options;
};
struct HorizonVertex { float4 Position [[position]]; };
vertex HorizonVertex HBAOVertexMain(uint vertexId [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 corner = vertexId == 0 ? float2(-1.0) :
        vertexId == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(corner, 0.0, 1.0)};
}

float4 MultiplyRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value), dot(rows[2], value), dot(rows[3], value));
}

bool Reconstruct(float2 pixel, float depth, constant HorizonUniforms& uniforms, thread float3& position)
{
    float2 uv = (pixel + 0.5) * uniforms.ViewportRadiusBias.xy;
    float4 world = MultiplyRows(uniforms.InverseViewProjection,
        float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0));
    position = float3(0.0);
    if (abs(world.w) < HomogeneousEpsilon) return false;
    position = world.xyz / world.w;
    return all(isfinite(position));
}

bool ReadSurface(uint2 pixel, constant HorizonUniforms& uniforms,
    depth2d<float, access::read> depthImage, texture2d<float, access::read> normalImage,
    thread float3& position, thread float3& normal)
{
    float depth = depthImage.read(pixel);
    float4 materialNormal = normalImage.read(pixel);
    float squaredLength = dot(materialNormal.xyz, materialNormal.xyz);
    position = float3(0.0);
    normal = float3(0.0, 1.0, 0.0);
    if (depth >= 1.0 || materialNormal.w < 0.0 || squaredLength <= SquaredLengthEpsilon) return false;
    normal = materialNormal.xyz * rsqrt(squaredLength);
    return Reconstruct(float2(pixel), depth, uniforms, position);
}

float EstimateHorizonVisibility(uint2 pixel, float3 center, float3 normal,
    constant HorizonUniforms& uniforms, depth2d<float, access::read> depthImage)
{
    int2 extent(depthImage.get_width(), depthImage.get_height());
    // Metal raster coordinates already use the canonical top-left pixel origin.
    float rotation = 2.0 * Pi * fract(RotationScale * fract(dot(float2(pixel), RotationDirection)));
    float3 towardViewer = -uniforms.ForwardIntensity.xyz;
    float depth = depthImage.read(pixel);
    float radius = uniforms.ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    float blocked = 0.0;
    for (int directionIndex = 0; directionIndex < DirectionCount; ++directionIndex)
    {
        float angle = rotation + 2.0 * Pi * float(directionIndex) / float(DirectionCount);
        float2 imageDirection(cos(angle), sin(angle));
        float3 neighborOnDepthPlane;
        if (!Reconstruct(float2(pixel) + imageDirection, depth, uniforms, neighborOnDepthPlane)) continue;
        float3 lateralDirection = neighborOnDepthPlane - center;
        lateralDirection -= towardViewer * dot(lateralDirection, towardViewer);
        float lateralSquared = dot(lateralDirection, lateralDirection);
        if (lateralSquared <= SquaredLengthEpsilon) continue;
        lateralDirection *= rsqrt(lateralSquared);
        // Projecting a world-radius endpoint handles perspective and aspect ratio.
        float4 endpoint = MultiplyRows(uniforms.ViewProjection, float4(center + radius * lateralDirection, 1.0));
        if (!all(isfinite(endpoint)) || endpoint.w <= HomogeneousEpsilon) continue;
        float2 endpointNdc = endpoint.xy / endpoint.w;
        float2 endpointUV(endpointNdc.x * 0.5 + 0.5, 0.5 - endpointNdc.y * 0.5);
        float2 marchOffset = endpointUV * float2(extent) - (float2(pixel) + 0.5);
        if (!all(isfinite(marchOffset))) continue;
        float highestHorizon = 0.0;
        int2 previousPixel = int2(pixel);
        for (int stepIndex = 1; stepIndex <= StepsPerDirection; ++stepIndex)
        {
            float fraction = float(stepIndex) / float(StepsPerDirection);
            float2 snapped = floor(float2(pixel) + marchOffset * fraction + 0.5);
            // Reject offscreen floating coordinates before converting to integer texels.
            if (any(snapped < 0.0) || any(snapped >= float2(extent))) continue;
            int2 samplePixel = int2(snapped);
            if (all(samplePixel == previousPixel)) continue;
            previousPixel = samplePixel;
            if (all(samplePixel == int2(pixel))) continue;
            float sampleDepth = depthImage.read(uint2(samplePixel));
            if (sampleDepth >= 1.0) continue;
            float3 samplePosition;
            if (!Reconstruct(float2(samplePixel), sampleDepth, uniforms, samplePosition)) continue;
            float3 displacement = samplePosition - center;
            float distanceSquared = dot(displacement, displacement);
            if (distanceSquared <= SquaredLengthEpsilon || distanceSquared >= radiusSquared) continue;
            float height = dot(displacement, towardViewer);
            float3 snappedLateral = displacement - towardViewer * height;
            float snappedSquared = dot(snappedLateral, snappedLateral);
            if (snappedSquared <= SquaredLengthEpsilon) continue;
            snappedLateral *= rsqrt(snappedSquared);
            float tangentAngle = atan2(-dot(normal, snappedLateral),
                max(dot(normal, towardViewer), TangentDenominatorEpsilon));
            float tangentWithBias = clamp(tangentAngle + uniforms.ViewportRadiusBias.w, -Pi * 0.5, Pi * 0.5);
            float sineHorizon = clamp(height * rsqrt(distanceSquared), -1.0, 1.0);
            float correctedHorizon = max(sineHorizon - sin(tangentWithBias), 0.0);
            float newOcclusion = max(correctedHorizon - highestHorizon, 0.0);
            blocked += newOcclusion * (1.0 - distanceSquared / radiusSquared);
            highestHorizon = max(highestHorizon, correctedHorizon);
        }
    }
    return saturate(1.0 - uniforms.ForwardIntensity.w * blocked / float(DirectionCount));
}

float BilateralVisibility(uint2 pixel, float3 center, float3 normal, constant HorizonUniforms& uniforms,
    depth2d<float, access::read> depthImage, texture2d<float, access::read> normalImage,
    texture2d<float, access::read> previousVisibility)
{
    int2 extent(depthImage.get_width(), depthImage.get_height());
    int2 direction = uniforms.Options.x == HorizontalFilterPass ? int2(1, 0) : int2(0, 1);
    float depthSigma = uniforms.ViewportRadiusBias.z * DepthStandardDeviationFraction;
    float weightedVisibility = 0.0;
    float totalWeight = 0.0;
    for (int offset = -FilterRadius; offset <= FilterRadius; ++offset)
    {
        uint2 neighbor = uint2(clamp(int2(pixel) + direction * offset, int2(0), extent - 1));
        float3 neighborPosition, neighborNormal;
        if (!ReadSurface(neighbor, uniforms, depthImage, normalImage, neighborPosition, neighborNormal)) continue;
        float depthDelta = dot(neighborPosition - center, uniforms.ForwardIntensity.xyz);
        float spatialWeight = exp(-float(offset * offset) / (2.0 * SpatialStandardDeviation * SpatialStandardDeviation));
        float depthWeight = exp(-depthDelta * depthDelta / (2.0 * depthSigma * depthSigma));
        float normalWeight = pow(saturate(dot(normal, neighborNormal)), NormalWeightExponent);
        float weight = spatialWeight * depthWeight * normalWeight;
        weightedVisibility += previousVisibility.read(neighbor).r * weight;
        totalWeight += weight;
    }
    return totalWeight > MinimumFilterWeight ? saturate(weightedVisibility / totalWeight) : 1.0;
}

fragment float4 HBAOFragmentMain(HorizonVertex input [[stage_in]],
    constant HorizonUniforms& uniforms [[buffer(0)]], depth2d<float, access::read> opaqueDepth [[texture(0)]],
    texture2d<float, access::read> normalRoughness [[texture(1)]],
    texture2d<float, access::read> ambientMetallic [[texture(2)]],
    texture2d<float, access::read> previousAO [[texture(3)]])
{
    // Preserve zero intensity exactly through all half-float stages.
    if (uniforms.ForwardIntensity.w <= 0.0)
        return uniforms.Options.x == CompositePass ? float4(0.0) : float4(1.0, 0.0, 0.0, 1.0);
    uint2 pixel = uint2(input.Position.xy);
    float3 position, normal;
    bool valid = ReadSurface(pixel, uniforms, opaqueDepth, normalRoughness, position, normal);
    if (uniforms.Options.x == EstimateVisibilityPass)
    {
        float visibility = valid ? EstimateHorizonVisibility(pixel, position, normal, uniforms, opaqueDepth) : 1.0;
        return float4(visibility, 0.0, 0.0, 1.0);
    }
    if (uniforms.Options.x == HorizontalFilterPass || uniforms.Options.x == VerticalFilterPass)
    {
        float visibility = valid ? BilateralVisibility(pixel, position, normal, uniforms, opaqueDepth,
            normalRoughness, previousAO) : 1.0;
        return float4(visibility, 0.0, 0.0, 1.0);
    }
    if (!valid) return float4(0.0);
    float visibility = saturate(previousAO.read(pixel).r);
    float3 ambient = max(ambientMetallic.read(pixel).rgb, 0.0);
    // The native RGB-only reverse-subtract blend preserves direct light and alpha.
    return float4(ambient * (1.0 - visibility), 0.0);
}
