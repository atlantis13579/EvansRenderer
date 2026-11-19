// Independently authored DX12 horizon-based ambient occlusion.
// Bavoil, Sainz and Dimitrov (SIGGRAPH 2008): integrate sin(horizon)-sin(tangent).
// Each newly exposed horizon band receives quadratic distance attenuation, as in
// the September 2008 NVIDIA SDK. Sample-specific tangents correct projection and
// snapping drift; at fixed azimuth this is the original incremental integral.
// Grazing surfaces retain the limitations of a single depth heightfield.
// https://rdimitrov.twistedsanity.net/HBAO_SIGGRAPH08.pdf
// https://developer.download.nvidia.com/SDK/10.5/direct3d/Source/ScreenSpaceAO/doc/ScreenSpaceAO.pdf
cbuffer AmbientHorizonConstants : register(b0)
{
    row_major float4x4 CameraViewProjection;
    row_major float4x4 InverseCameraViewProjection;
    float4 InverseExtentRadiusAngle;
    float4 CameraForwardStrength;
    uint4 StageOptions;
};
Texture2D<float> DepthImage : register(t0);
Texture2D<float4> NormalImage : register(t1);
Texture2D<float4> AmbientImage : register(t2);
Texture2D<float4> VisibilityImage : register(t3);

static const uint EstimateVisibility = 0u;
static const uint FilterHorizontal = 1u;
static const uint FilterVertical = 2u;
static const int AzimuthCount = 8;
static const int RadialStepCount = 8;
static const int FilterRadius = 2;
static const float Pi = 3.141592653589793f;
static const float PixelCenter = 0.5f;
static const float MinimumHomogeneousW = 1.0e-7f;
static const float MinimumSquaredVectorLength = 1.0e-12f;
static const float MinimumTangentCosine = 1.0e-6f;
static const float SpatialStandardDeviation = 1.5f;
static const float DepthStandardDeviationFraction = 0.25f;
static const float NormalSimilarityPower = 8.0f;
static const float MinimumFilterWeight = 1.0e-7f;
static const float BackgroundDepth = 1.0f;
static const float2 NoiseDirection = float2(0.06711056f, 0.00583715f);
static const float NoiseMultiplier = 52.9829189f;

struct FullscreenVertex { float4 Position : SV_POSITION; };
FullscreenVertex VSMain(uint id : SV_VertexID)
{
    float2 uv = float2((id << 1u) & 2u, id & 2u);
    FullscreenVertex output;
    output.Position = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}

bool WorldFromPixel(float2 texel, float depth, out float3 world)
{
    float2 uv = (texel + PixelCenter) * InverseExtentRadiusAngle.xy;
    float4 homogeneous = mul(InverseCameraViewProjection,
        float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), depth, 1.0f));
    world = 0.0f;
    if (abs(homogeneous.w) < MinimumHomogeneousW) return false;
    world = homogeneous.xyz / homogeneous.w;
    return all(isfinite(world));
}

bool OpaqueSurface(int2 texel, out float3 position, out float3 normal)
{
    float4 encoded = NormalImage.Load(int3(texel, 0));
    float depth = DepthImage.Load(int3(texel, 0));
    float normalLengthSquared = dot(encoded.xyz, encoded.xyz);
    position = normal = 0.0f;
    if (encoded.w < 0.0f || depth >= BackgroundDepth || normalLengthSquared <= MinimumSquaredVectorLength) return false;
    normal = encoded.xyz * rsqrt(normalLengthSquared);
    return WorldFromPixel(float2(texel), depth, position);
}

float EstimateHorizons(int2 texel, int2 dimensions, float3 origin, float3 normal)
{
    float radius = InverseExtentRadiusAngle.z;
    float radiusSquared = radius * radius;
    float originDepth = DepthImage.Load(int3(texel, 0));
    float3 eyeAxis = -CameraForwardStrength.xyz;
    float angleOffset = 2.0f * Pi * frac(NoiseMultiplier * frac(dot(float2(texel), NoiseDirection)));
    float blocked = 0.0f;
    [loop] for (int azimuthIndex = 0; azimuthIndex != AzimuthCount; ++azimuthIndex)
    {
        float azimuth = angleOffset + 2.0f * Pi * float(azimuthIndex) / AzimuthCount;
        float3 offsetWorld;
        float2 directionPixel = float2(cos(azimuth), sin(azimuth));
        if (!WorldFromPixel(float2(texel) + directionPixel, originDepth, offsetWorld)) continue;
        float3 diskDirection = offsetWorld - origin;
        diskDirection -= eyeAxis * dot(diskDirection, eyeAxis);
        float diskLengthSquared = dot(diskDirection, diskDirection);
        if (diskLengthSquared <= MinimumSquaredVectorLength) continue;
        diskDirection *= rsqrt(diskLengthSquared);

        float4 radiusClip = mul(CameraViewProjection, float4(origin + radius * diskDirection, 1.0f));
        if (!all(isfinite(radiusClip)) || radiusClip.w <= MinimumHomogeneousW) continue;
        float2 radiusUV = radiusClip.xy / radiusClip.w * float2(0.5f, -0.5f) + 0.5f;
        float2 projectedRadius = radiusUV * float2(dimensions) - float2(texel) - PixelCenter;
        if (!all(isfinite(projectedRadius))) continue;
        float highestBand = 0.0f;
        int2 lastTexel = texel;
        [loop] for (int radialStep = 1; radialStep <= RadialStepCount; ++radialStep)
        {
            float radialFraction = float(radialStep) / RadialStepCount;
            float2 coordinate = floor(float2(texel) + projectedRadius * radialFraction + PixelCenter);
            if (any(coordinate < 0.0f) || any(coordinate >= float2(dimensions))) continue;
            int2 snapped = int2(coordinate);
            if (all(snapped == lastTexel)) continue;
            lastTexel = snapped;
            if (all(snapped == texel)) continue;
            float depth = DepthImage.Load(int3(snapped, 0));
            float3 neighborWorld;
            if (depth >= BackgroundDepth || !WorldFromPixel(float2(snapped), depth, neighborWorld)) continue;

            float3 delta = neighborWorld - origin;
            float squaredDistance = dot(delta, delta);
            if (squaredDistance <= MinimumSquaredVectorLength || squaredDistance >= radiusSquared) continue;
            float alongEyeAxis = dot(delta, eyeAxis);
            float3 actualAzimuth = delta - eyeAxis * alongEyeAxis;
            float azimuthLengthSquared = dot(actualAzimuth, actualAzimuth);
            if (azimuthLengthSquared <= MinimumSquaredVectorLength) continue;
            actualAzimuth *= rsqrt(azimuthLengthSquared);
            float horizonSine = alongEyeAxis * rsqrt(squaredDistance);
            float tangentAngle = atan2(-dot(normal, actualAzimuth), max(dot(normal, eyeAxis), MinimumTangentCosine));
            tangentAngle = clamp(tangentAngle + InverseExtentRadiusAngle.w, -Pi * 0.5f, Pi * 0.5f);
            float currentBand = max(horizonSine - sin(tangentAngle), 0.0f);
            // Measure the tangent at the snapped sample's azimuth. Comparing to
            // the first ray tangent would self-occlude a perspective-tilted plane.
            float increment = max(currentBand - highestBand, 0.0f);
            float distanceWeight = 1.0f - squaredDistance / radiusSquared;
            blocked += increment * distanceWeight;
            highestBand = max(highestBand, currentBand);
        }
    }
    return saturate(1.0f - CameraForwardStrength.w * blocked / AzimuthCount);
}

float FilterVisibility(int2 texel, int2 dimensions, float3 origin, float3 normal)
{
    int2 axis = StageOptions.x == FilterHorizontal ? int2(1, 0) : int2(0, 1);
    float depthSigma = InverseExtentRadiusAngle.z * DepthStandardDeviationFraction;
    float numerator = 0.0f;
    float denominator = 0.0f;
    [unroll] for (int offset = -FilterRadius; offset <= FilterRadius; ++offset)
    {
        int2 neighbor = clamp(texel + axis * offset, int2(0, 0), dimensions - 1);
        float3 position, neighborNormal;
        if (!OpaqueSurface(neighbor, position, neighborNormal)) continue;
        float depthDelta = dot(position - origin, CameraForwardStrength.xyz);
        float spatialWeight = exp(-float(offset * offset) / (2.0f * SpatialStandardDeviation * SpatialStandardDeviation));
        float depthWeight = exp(-depthDelta * depthDelta / (2.0f * depthSigma * depthSigma));
        float weight = spatialWeight * depthWeight * pow(saturate(dot(normal, neighborNormal)), NormalSimilarityPower);
        numerator += weight * VisibilityImage.Load(int3(neighbor, 0)).r;
        denominator += weight;
    }
    return denominator > MinimumFilterWeight ? numerator / denominator : 1.0f;
}

float4 PSMain(FullscreenVertex fragment) : SV_TARGET
{
    // Avoid arithmetic drift in an exact zero-strength four-pass identity.
    if (CameraForwardStrength.w <= 0.0f)
    {
        bool visibilityStage = StageOptions.x == EstimateVisibility || StageOptions.x == FilterHorizontal || StageOptions.x == FilterVertical;
        return float4(visibilityStage ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    }
    int2 texel = int2(fragment.Position.xy);
    uint width, height;
    DepthImage.GetDimensions(width, height);
    float3 origin, normal;
    bool valid = OpaqueSurface(texel, origin, normal);
    if (StageOptions.x == EstimateVisibility)
        return float4(valid ? EstimateHorizons(texel, int2(width, height), origin, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (StageOptions.x == FilterHorizontal || StageOptions.x == FilterVertical)
        return float4(valid ? FilterVisibility(texel, int2(width, height), origin, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (!valid) return 0.0f;
    float visibility = saturate(VisibilityImage.Load(int3(texel, 0)).r);
    return float4(AmbientImage.Load(int3(texel, 0)).rgb * (1.0f - visibility), 0.0f);
}
