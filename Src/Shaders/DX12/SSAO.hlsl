// Independently maintained DX12 screen-space ambient occlusion.
// Depth reconstructs opaque world positions. A cosine hemisphere estimates
// visibility, separable bilateral filtering respects depth/normal boundaries,
// and reverse-subtract compositing removes only the material's ambient radiance.
// Mittring, "Finding Next Gen - CryEngine 2", SIGGRAPH 2007, section 8.5.4.3.
// https://advances.realtimerendering.com/s2007/Mittring-Finding_NextGen_CryEngine2(Siggraph07%20Course%20Notes).pdf
cbuffer AmbientOcclusionConstants : register(b0)
{
    row_major float4x4 CameraViewProjection;
    row_major float4x4 InverseCameraViewProjection;
    float4 InverseExtentRadiusBias;
    float4 CameraForwardIntensity;
    uint4 PassOptions;
};
Texture2D<float> DepthImage : register(t0);
Texture2D<float4> NormalImage : register(t1);
Texture2D<float4> AmbientImage : register(t2);
Texture2D<float4> VisibilityImage : register(t3);

static const uint EstimateVisibility = 0u;
static const uint FilterHorizontal = 1u;
static const uint FilterVertical = 2u;
static const int KernelSize = 24;
static const int FilterRadius = 2;
static const float Pi = 3.141592653589793f;
static const float GoldenAngleRadians = Pi * (3.0f - sqrt(5.0f));
static const float PixelCenter = 0.5f;
static const float MinimumHomogeneousW = 1.0e-7f;
static const float MinimumNormalLengthSquared = 1.0e-12f;
static const float PoleThreshold = 0.999f;
static const float KernelStart = 0.1f;
static const float OccluderFadeStart = 0.5f;
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

bool WorldFromDepth(int2 texel, float depth, out float3 world)
{
    float2 uv = (float2(texel) + PixelCenter) * InverseExtentRadiusBias.xy;
    float4 homogeneous = mul(InverseCameraViewProjection, float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), depth, 1.0f));
    world = 0.0f;
    if (abs(homogeneous.w) < MinimumHomogeneousW) return false;
    world = homogeneous.xyz / homogeneous.w;
    return true;
}

bool OpaqueSurface(int2 texel, out float3 position, out float3 normal)
{
    float4 encoded = NormalImage.Load(int3(texel, 0));
    float depth = DepthImage.Load(int3(texel, 0));
    float normSquared = dot(encoded.xyz, encoded.xyz);
    position = normal = 0.0f;
    if (encoded.w < 0.0f || depth >= BackgroundDepth || normSquared <= MinimumNormalLengthSquared) return false;
    normal = encoded.xyz * rsqrt(normSquared);
    return WorldFromDepth(texel, depth, position);
}

float EstimateHemisphere(int2 texel, int2 dimensions, float3 origin, float3 normal)
{
    float3 up = abs(normal.y) < PoleThreshold ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    float angleOffset = 2.0f * Pi * frac(NoiseMultiplier * frac(dot(float2(texel), NoiseDirection)));
    float radius = InverseExtentRadiusBias.z;
    float blockedWeight = 0.0f;
    [loop] for (int sampleIndex = 0; sampleIndex != KernelSize; ++sampleIndex)
    {
        float xi = (float(sampleIndex) + PixelCenter) / KernelSize;
        float azimuth = angleOffset + float(sampleIndex) * GoldenAngleRadians;
        float diskDistance = sqrt(xi);
        float3 hemisphere = tangent * cos(azimuth) * diskDistance + bitangent * sin(azimuth) * diskDistance
            + normal * sqrt(1.0f - xi);
        float radialFraction = lerp(KernelStart, 1.0f, xi * xi);
        float3 candidate = origin + hemisphere * (radius * radialFraction);
        float4 projected = mul(CameraViewProjection, float4(candidate, 1.0f));
        if (projected.w <= MinimumHomogeneousW) continue;
        float3 ndc = projected.xyz / projected.w;
        float2 screenUV = ndc.xy * float2(0.5f, -0.5f) + 0.5f;
        if (any(screenUV < 0.0f) || any(screenUV >= 1.0f) || ndc.z < 0.0f || ndc.z > BackgroundDepth) continue;
        int2 nearestTexel = int2(floor(screenUV * float2(dimensions)));
        float sampledDepth = DepthImage.Load(int3(nearestTexel, 0));
        float3 occluder;
        if (sampledDepth >= BackgroundDepth || !WorldFromDepth(nearestTexel, sampledDepth, occluder)) continue;
        float separation = length(occluder - origin);
        float behindSurface = dot(candidate - occluder, CameraForwardIntensity.xyz);
        if (separation < radius && behindSurface > InverseExtentRadiusBias.w)
            blockedWeight += 1.0f - smoothstep(radius * OccluderFadeStart, radius, separation);
    }
    return saturate(1.0f - CameraForwardIntensity.w * blockedWeight / KernelSize);
}

float FilterVisibility(int2 texel, int2 dimensions, float3 origin, float3 normal)
{
    int2 axis = PassOptions.x == FilterHorizontal ? int2(1, 0) : int2(0, 1);
    float depthSigma = InverseExtentRadiusBias.z * DepthStandardDeviationFraction;
    float numerator = 0.0f;
    float denominator = 0.0f;
    [unroll] for (int step = -FilterRadius; step <= FilterRadius; ++step)
    {
        int2 neighbor = clamp(texel + axis * step, int2(0, 0), dimensions - 1);
        float3 position, neighborNormal;
        if (!OpaqueSurface(neighbor, position, neighborNormal)) continue;
        float delta = dot(position - origin, CameraForwardIntensity.xyz);
        float spatial = exp(-float(step * step) / (2.0f * SpatialStandardDeviation * SpatialStandardDeviation));
        float range = exp(-delta * delta / (2.0f * depthSigma * depthSigma));
        float similarity = pow(saturate(dot(normal, neighborNormal)), NormalSimilarityPower);
        float weight = spatial * range * similarity;
        numerator += weight * VisibilityImage.Load(int3(neighbor, 0)).r;
        denominator += weight;
    }
    return denominator > MinimumFilterWeight ? numerator / denominator : 1.0f;
}

float4 PSMain(FullscreenVertex fragment) : SV_TARGET
{
    // Preserve an exact zero-strength identity before any weighted arithmetic.
    // Estimation/filter targets receive visibility one; compositing removes zero.
    if (CameraForwardIntensity.w <= 0.0f)
    {
        bool visibilityPass = PassOptions.x == EstimateVisibility || PassOptions.x == FilterHorizontal || PassOptions.x == FilterVertical;
        return float4(visibilityPass ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    }
    int2 texel = int2(fragment.Position.xy);
    uint width, height;
    DepthImage.GetDimensions(width, height);
    float3 origin, normal;
    bool hasSurface = OpaqueSurface(texel, origin, normal);
    if (PassOptions.x == EstimateVisibility)
        return float4(hasSurface ? EstimateHemisphere(texel, int2(width, height), origin, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (PassOptions.x == FilterHorizontal || PassOptions.x == FilterVertical)
        return float4(hasSurface ? FilterVisibility(texel, int2(width, height), origin, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (!hasSurface) return 0.0f;
    float remainingAmbient = saturate(VisibilityImage.Load(int3(texel, 0)).r);
    return float4(AmbientImage.Load(int3(texel, 0)).rgb * (1.0f - remainingAmbient), 0.0f);
}
