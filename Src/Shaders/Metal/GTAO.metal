#include <metal_stdlib>
using namespace metal;
// Jimenez et al., Practical Realtime Strategies for Accurate Indirect Occlusion,
// ATVI-TR-16-01 (2016), equations 5-8.
// https://www.activision.com/cdn/research/PracticalRealtimeStrategiesTRfinal.pdf
// Eight spatial slices integrate cosine-weighted visibility; temporal sampling
// is optional. Periodic hemisphere clipping preserves the authored normal.
// The unoccluded slice sum removes finite-quadrature darkening on tilted planes.
constant uint EstimatePass = 0;
constant uint HorizontalPass = 1;
constant uint VerticalPass = 2;
constant uint CompositePass = 3;
constant int SliceCount = 8;
constant int RadialStepsPerSide = 8;
constant int FilterRadius = 2;
constant int HemisphereWrapCount = 1;
constant float Pi = 3.141592653589793;
constant float HalfPi = Pi * 0.5;
constant float PrimitiveCoefficient = 0.25;
constant float TexelCenter = 0.5;
constant float InteriorClipDepth = 0.5;
constant float ClearDepth = 1.0;
constant float ProjectionEpsilon = 1.0e-7;
constant float MinimumBaseline = 1.0e-7;
constant float RangeFadeStartFraction = 0.5;
constant float OpenHorizonCosine = -1.0;
constant float SpatialDeviation = 1.5;
constant float DepthDeviationRadiusFraction = 0.25;
constant float NormalSimilarityPower = 8.0;
constant float MinimumFilterWeight = 1.0e-7;
constant float RotationScale = 52.9829189;
constant float2 RotationDirection(0.06711056, 0.00583715);
// One binary16 ULP covers truncating conversion; subnormal components need an absolute bound.
constant float HalfNormalRelativeError = 1.0 / 1024.0;
constant float HalfNormalSubnormalStep = 1.0 / 16777216.0;

struct GroundTruthUniforms
{
    float4 ViewProjection[4];
    float4 InverseViewProjection[4];
    float4 ViewportRadiusBias; // Inverse dimensions, radius and height bias in world units.
    float4 ForwardIntensity; // Optical forward for the depth filter, then intensity; view direction is local.
    uint4 Options;
};
struct GroundTruthVertex { float4 Position [[position]]; };
vertex GroundTruthVertex GTAOVertexMain(uint index [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 corner = index == 0 ? float2(-1.0) :
        index == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(corner, 0.0, 1.0)};
}
float4 TransformRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value), dot(rows[2], value), dot(rows[3], value));
}
bool MakeUnit(thread float3& value)
{
    if (!all(isfinite(value))) return false;
    float scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if (scale <= 0.0) return false;
    value /= scale;
    value *= rsqrt(dot(value, value));
    return true;
}
bool Unproject(float2 pixel, float depth, constant GroundTruthUniforms& uniforms, thread float3& world)
{
    float2 uv = (pixel + TexelCenter) * uniforms.ViewportRadiusBias.xy;
    float4 homogeneous = TransformRows(uniforms.InverseViewProjection,
        float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0));
    world = float3(0.0);
    if (!all(isfinite(homogeneous)) || abs(homogeneous.w) < ProjectionEpsilon) return false;
    world = homogeneous.xyz / homogeneous.w;
    return all(isfinite(world));
}
bool ReadSurface(uint2 pixel, constant GroundTruthUniforms& uniforms,
    depth2d<float, access::read> depth, texture2d<float, access::read> material,
    thread float3& world, thread float3& normal)
{
    float hardwareDepth = depth.read(pixel);
    float4 stored = material.read(pixel);
    world = float3(0.0);
    normal = stored.xyz;
    return hardwareDepth < ClearDepth && stored.w >= 0.0 && MakeUnit(normal) &&
        Unproject(float2(pixel), hardwareDepth, uniforms, world);
}
float SignedPrimitive(float angle, float viewCoefficient, float tangentCoefficient)
{
    float doubleAngle = 2.0 * angle;
    float integral = PrimitiveCoefficient * (viewCoefficient * (1.0 - cos(doubleAngle)) +
        tangentCoefficient * (doubleAngle - sin(doubleAngle)));
    // The signed primitive accounts for the spherical Jacobian |sin(theta)|.
    return sign(angle) * integral;
}
float IntegrateSlice(float lower, float upper, float viewCoefficient, float tangentCoefficient, float normalAngle)
{
    // The two normal coefficients already include the projected normal's length.
    // Wrapped positive-cosine intervals handle every authored normal orientation.
    float integrated = 0.0;
    for (int wrap = -HemisphereWrapCount; wrap <= HemisphereWrapCount; ++wrap)
    {
        float hemisphereCenter = normalAngle + float(wrap) * 2.0 * Pi;
        float start = max(lower, hemisphereCenter - HalfPi);
        float end = min(upper, hemisphereCenter + HalfPi);
        if (end > start)
            integrated += SignedPrimitive(end, viewCoefficient, tangentCoefficient) -
                SignedPrimitive(start, viewCoefficient, tangentCoefficient);
    }
    return integrated;
}
float SearchHorizon(uint2 pixel, float3 center, float3 normal, float3 view, float3 tangent, float side,
    constant GroundTruthUniforms& uniforms, depth2d<float, access::read> depthImage)
{
    float2 extent(depthImage.get_width(), depthImage.get_height());
    float radius = uniforms.ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    const float NormalEncodingVectorError = HalfNormalRelativeError + sqrt(3.0) * HalfNormalSubnormalStep;
    const float NormalEncodingSlopeBound = NormalEncodingVectorError / (1.0 - NormalEncodingVectorError);
    float2 normalInSlice(dot(normal, view), dot(normal, tangent));
    float normalProjectionSquared = dot(normalInSlice, normalInSlice);
    float horizon = OpenHorizonCosine;
    uint2 previousPixel = pixel;
    for (int stepIndex = 1; stepIndex <= RadialStepsPerSide; ++stepIndex)
    {
        float fraction = float(stepIndex) / float(RadialStepsPerSide);
        // Squared radial spacing gives nearby detail more samples.
        float sampleRadius = radius * fraction * fraction;
        float4 clip = TransformRows(uniforms.ViewProjection, float4(center + side * tangent * sampleRadius, 1.0));
        if (!all(isfinite(clip)) || clip.w <= ProjectionEpsilon) continue;
        float2 projected = clip.xy / clip.w;
        float2 uv(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
        if (!all(isfinite(uv)) || any(uv < 0.0) || any(uv >= 1.0)) continue;
        uint2 samplePixel = uint2(floor(uv * extent));
        if (all(samplePixel == previousPixel)) continue;
        previousPixel = samplePixel;
        if (all(samplePixel == pixel)) continue;
        float sampleDepth = depthImage.read(samplePixel);
        float3 samplePosition;
        if (sampleDepth >= ClearDepth || !Unproject(float2(samplePixel), sampleDepth, uniforms, samplePosition)) continue;
        float3 displacement = samplePosition - center;
        float distanceSquared = dot(displacement, displacement);
        float alongTangent = dot(displacement, tangent);
        if (!all(isfinite(displacement)) || distanceSquared >= radiusSquared || side * alongTangent <= 0.0) continue;
        float worldDistance = sqrt(distanceSquared);
        float worldHeight = dot(displacement, normal);
        // Bound the apparent plane slope introduced by storing a unit normal in RGBA16F.
        if (worldHeight <= uniforms.ViewportRadiusBias.w + worldDistance * NormalEncodingSlopeBound) continue;
        float2 components(dot(displacement, view), alongTangent);
        // Correct the snapped sample's off-slice displacement while preserving its normal height.
        // Multiplying by the projected normal length squared avoids a division in this direction.
        components = components * normalProjectionSquared + normalInSlice *
            (worldHeight - dot(normalInSlice, components));
        if (side * components.y <= 0.0) continue;
        float scale = max(abs(components.x), abs(components.y));
        if (!isfinite(scale) || scale <= 0.0) continue;
        components /= scale;
        float cosine = components.x * rsqrt(dot(components, components));
        float distanceWeight = 1.0 - smoothstep(radius * RangeFadeStartFraction, radius, worldDistance);
        float candidate = mix(OpenHorizonCosine, clamp(cosine, -1.0, 1.0), distanceWeight);
        horizon = max(horizon, candidate);
    }
    return horizon;
}
float EstimateVisibility(uint2 pixel, float3 center, float3 normal, constant GroundTruthUniforms& uniforms,
    depth2d<float, access::read> depthImage)
{
    float3 nearPoint, interiorPoint, adjacentPoint;
    if (!Unproject(float2(pixel), 0.0, uniforms, nearPoint) ||
        !Unproject(float2(pixel), InteriorClipDepth, uniforms, interiorPoint) ||
        !Unproject(float2(pixel) + float2(1.0, 0.0), depthImage.read(pixel), uniforms, adjacentPoint)) return 1.0;
    // A local near/interior ray supports perspective, orthographic and off-axis cameras.
    float3 view = nearPoint - interiorPoint;
    if (!MakeUnit(view)) return 1.0;
    float3 right = adjacentPoint - center;
    right -= view * dot(right, view);
    if (!MakeUnit(right)) return 1.0;
    float3 down = cross(right, view);
    // Deterministic interleaved-gradient rotation hashes integer top-left pixels.
    float rotation = fract(RotationScale * fract(dot(float2(pixel), RotationDirection)));
    float visibleIntegral = 0.0;
    float unoccludedIntegral = 0.0;
    for (int sliceIndex = 0; sliceIndex < SliceCount; ++sliceIndex)
    {
        float azimuth = (float(sliceIndex) / float(SliceCount) + rotation) * Pi;
        float3 tangent = cos(azimuth) * right + sin(azimuth) * down;
        float viewCoefficient = dot(normal, view);
        float tangentCoefficient = dot(normal, tangent);
        if (viewCoefficient == 0.0 && tangentCoefficient == 0.0) continue;
        float normalAngle = atan2(tangentCoefficient, viewCoefficient);
        float negativeCosine = SearchHorizon(pixel, center, normal, view, tangent, -1.0, uniforms, depthImage);
        float positiveCosine = SearchHorizon(pixel, center, normal, view, tangent, 1.0, uniforms, depthImage);
        float lower = -acos(clamp(negativeCosine, -1.0, 1.0));
        float upper = acos(clamp(positiveCosine, -1.0, 1.0));
        visibleIntegral += IntegrateSlice(lower, upper, viewCoefficient, tangentCoefficient, normalAngle);
        unoccludedIntegral += IntegrateSlice(-Pi, Pi, viewCoefficient, tangentCoefficient, normalAngle);
    }
    if (unoccludedIntegral <= MinimumBaseline) return 1.0;
    return saturate(1.0 - uniforms.ForwardIntensity.w * (1.0 - visibleIntegral / unoccludedIntegral));
}
float FilterVisibility(uint2 pixel, float3 center, float3 normal, constant GroundTruthUniforms& uniforms,
    depth2d<float, access::read> depthImage, texture2d<float, access::read> material,
    texture2d<float, access::read> previous)
{
    int2 extent(depthImage.get_width(), depthImage.get_height());
    int2 axis = uniforms.Options.x == HorizontalPass ? int2(1, 0) : int2(0, 1);
    float depthSigma = uniforms.ViewportRadiusBias.z * DepthDeviationRadiusFraction;
    float visibilitySum = 0.0;
    float weightSum = 0.0;
    for (int offset = -FilterRadius; offset <= FilterRadius; ++offset)
    {
        uint2 neighbor = uint2(clamp(int2(pixel) + axis * offset, int2(0), extent - 1));
        float3 position, neighborNormal;
        if (!ReadSurface(neighbor, uniforms, depthImage, material, position, neighborNormal)) continue;
        float depthDelta = dot(position - center, uniforms.ForwardIntensity.xyz);
        float spatialWeight = exp(-float(offset * offset) / (2.0 * SpatialDeviation * SpatialDeviation));
        float depthWeight = exp(-depthDelta * depthDelta / (2.0 * depthSigma * depthSigma));
        float normalWeight = pow(saturate(dot(normal, neighborNormal)), NormalSimilarityPower);
        float weight = spatialWeight * depthWeight * normalWeight;
        visibilitySum += previous.read(neighbor).r * weight;
        weightSum += weight;
    }
    return weightSum > MinimumFilterWeight ? saturate(visibilitySum / weightSum) : 1.0;
}
fragment float4 GTAOFragmentMain(GroundTruthVertex input [[stage_in]],
    constant GroundTruthUniforms& uniforms [[buffer(0)]], depth2d<float, access::read> opaqueDepth [[texture(0)]],
    texture2d<float, access::read> normalRoughness [[texture(1)]],
    texture2d<float, access::read> ambientMetallic [[texture(2)]],
    texture2d<float, access::read> previousAO [[texture(3)]])
{
    if (uniforms.ForwardIntensity.w <= 0.0)
        return uniforms.Options.x == CompositePass ? float4(0.0) : float4(1.0, 0.0, 0.0, 1.0);
    uint2 pixel = uint2(input.Position.xy);
    float3 center, normal;
    bool valid = ReadSurface(pixel, uniforms, opaqueDepth, normalRoughness, center, normal);
    if (uniforms.Options.x == EstimatePass)
        return float4(valid ? EstimateVisibility(pixel, center, normal, uniforms, opaqueDepth) : 1.0, 0.0, 0.0, 1.0);
    if (uniforms.Options.x == HorizontalPass || uniforms.Options.x == VerticalPass)
        return float4(valid ? FilterVisibility(pixel, center, normal, uniforms, opaqueDepth, normalRoughness, previousAO) : 1.0,
            0.0, 0.0, 1.0);
    if (!valid) return float4(0.0);
    float visibility = saturate(previousAO.read(pixel).r);
    return float4(max(ambientMetallic.read(pixel).rgb, 0.0) * (1.0 - visibility), 0.0);
}
