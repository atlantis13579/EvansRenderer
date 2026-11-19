// Screen-space ambient visibility from opaque depth and material normals.
// Mittring, "Finding Next Gen - CryEngine 2", SIGGRAPH 2007, section 8.5.4.3.
// A rotated cosine hemisphere avoids an extra noise asset; two bilateral passes
// retain silhouettes. Composite subtracts only lost ambient, never direct light.
// https://advances.realtimerendering.com/s2007/Mittring-Finding_NextGen_CryEngine2(Siggraph07%20Course%20Notes).pdf
cbuffer SsaoConstants : register(b0)
{
    row_major float4x4 ViewProjection;
    row_major float4x4 InverseViewProjection;
    float4 ViewportRadiusBias;
    float4 ForwardIntensity;
    uint4 Options;
};
Texture2D<float> OpaqueDepth : register(t0);
Texture2D<float4> NormalRoughness : register(t1);
Texture2D<float4> AmbientMetallic : register(t2);
Texture2D<float4> PreviousVisibility : register(t3);

static const uint HemispherePass = 0u;
static const uint HorizontalPass = 1u;
static const uint VerticalPass = 2u;
static const int HemisphereSampleCount = 24;
static const int BilateralRadius = 2;
static const float Pi = 3.141592653589793f;
static const float GoldenAngle = Pi * (3.0f - sqrt(5.0f));
static const float TexelCenterOffset = 0.5f;
static const float HomogeneousEpsilon = 1.0e-7f;
static const float NormalLengthEpsilon = 1.0e-12f;
static const float ParallelAxisLimit = 0.999f;
static const float MinimumKernelRadiusFraction = 0.1f;
static const float RangeFadeStartFraction = 0.5f;
static const float SpatialSigma = 1.5f;
static const float DepthSigmaRadiusFraction = 0.25f;
static const float NormalWeightExponent = 8.0f;
static const float WeightEpsilon = 1.0e-7f;
static const float ClearDepth = 1.0f;
static const float2 RotationPixelDirection = float2(0.06711056f, 0.00583715f);
static const float RotationScale = 52.9829189f;

struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    float2 coordinate = float2((vertexId << 1u) & 2u, vertexId & 2u);
    ScreenVertex vertex;
    vertex.Position = float4(coordinate.x * 2.0f - 1.0f, 1.0f - coordinate.y * 2.0f, 0.0f, 1.0f);
    return vertex;
}

bool ReconstructPosition(int2 pixel, float depth, out float3 position)
{
    float2 uv = (float2(pixel) + TexelCenterOffset) * ViewportRadiusBias.xy;
    float4 homogeneous = mul(InverseViewProjection, float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f));
    position = 0.0f;
    if (abs(homogeneous.w) < HomogeneousEpsilon) return false;
    position = homogeneous.xyz / homogeneous.w;
    return true;
}

bool ReadSurface(int2 pixel, out float3 position, out float3 normal)
{
    float depth = OpaqueDepth.Load(int3(pixel, 0));
    float4 materialNormal = NormalRoughness.Load(int3(pixel, 0));
    float squaredLength = dot(materialNormal.xyz, materialNormal.xyz);
    position = normal = 0.0f;
    if (depth >= ClearDepth || materialNormal.w < 0.0f || squaredLength <= NormalLengthEpsilon) return false;
    normal = materialNormal.xyz * rsqrt(squaredLength);
    return ReconstructPosition(pixel, depth, position);
}

float HemisphereVisibility(int2 pixel, int2 extent, float3 center, float3 normal)
{
    float3 helper = abs(normal.y) < ParallelAxisLimit ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(helper, normal));
    float3 bitangent = cross(normal, tangent);
    float rotation = 2.0f * Pi * frac(RotationScale * frac(dot(float2(pixel), RotationPixelDirection)));
    float radius = ViewportRadiusBias.z;
    float occlusion = 0.0f;
    [loop] for (int index = 0; index < HemisphereSampleCount; ++index)
    {
        float xi = (float(index) + TexelCenterOffset) / HemisphereSampleCount;
        float angle = rotation + GoldenAngle * index;
        float diskRadius = sqrt(xi);
        float3 direction = tangent * (cos(angle) * diskRadius) + bitangent * (sin(angle) * diskRadius)
            + normal * sqrt(1.0f - xi);
        float3 samplePosition = center + direction * radius * lerp(MinimumKernelRadiusFraction, 1.0f, xi * xi);
        float4 clip = mul(ViewProjection, float4(samplePosition, 1.0f));
        if (clip.w <= HomogeneousEpsilon) continue;
        float3 projected = clip.xyz / clip.w;
        float2 uv = float2(projected.x * 0.5f + 0.5f, 0.5f - projected.y * 0.5f);
        if (any(uv < 0.0f) || any(uv >= 1.0f) || projected.z < 0.0f || projected.z > ClearDepth) continue;
        int2 samplePixel = int2(floor(uv * float2(extent)));
        float sceneDepth = OpaqueDepth.Load(int3(samplePixel, 0));
        float3 scenePosition;
        if (sceneDepth >= ClearDepth || !ReconstructPosition(samplePixel, sceneDepth, scenePosition)) continue;
        float distanceToCenter = length(scenePosition - center);
        if (distanceToCenter >= radius || dot(samplePosition - scenePosition, ForwardIntensity.xyz) <= ViewportRadiusBias.w) continue;
        occlusion += 1.0f - smoothstep(radius * RangeFadeStartFraction, radius, distanceToCenter);
    }
    // Missing and offscreen samples remain unoccluded in the fixed denominator.
    return saturate(1.0f - ForwardIntensity.w * occlusion / HemisphereSampleCount);
}

float BilateralVisibility(int2 pixel, int2 extent, float3 center, float3 normal)
{
    int2 axis = Options.x == HorizontalPass ? int2(1, 0) : int2(0, 1);
    float depthSigma = ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float weighted = 0.0f;
    float total = 0.0f;
    [unroll] for (int offset = -BilateralRadius; offset <= BilateralRadius; ++offset)
    {
        int2 neighbor = clamp(pixel + axis * offset, int2(0, 0), extent - 1);
        float3 position, neighborNormal;
        if (!ReadSurface(neighbor, position, neighborNormal)) continue;
        float depthDelta = dot(position - center, ForwardIntensity.xyz);
        float spatialWeight = exp(-float(offset * offset) / (2.0f * SpatialSigma * SpatialSigma));
        float depthWeight = exp(-depthDelta * depthDelta / (2.0f * depthSigma * depthSigma));
        float weight = spatialWeight * depthWeight * pow(saturate(dot(normal, neighborNormal)), NormalWeightExponent);
        weighted += PreviousVisibility.Load(int3(neighbor, 0)).r * weight;
        total += weight;
    }
    return total > WeightEpsilon ? weighted / total : 1.0f;
}

float4 PSMain(ScreenVertex input) : SV_TARGET
{
    // Zero strength is an exact identity, including on curved surfaces where
    // bilateral normalization can otherwise round unit visibility below one.
    if (ForwardIntensity.w <= 0.0f)
        return float4(Options.x == HemispherePass || Options.x == HorizontalPass || Options.x == VerticalPass
            ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    int2 pixel = int2(input.Position.xy);
    uint width, height;
    OpaqueDepth.GetDimensions(width, height);
    int2 extent = int2(width, height);
    float3 center, normal;
    bool valid = ReadSurface(pixel, center, normal);
    if (Options.x == HemispherePass)
        return float4(valid ? HemisphereVisibility(pixel, extent, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (Options.x == HorizontalPass || Options.x == VerticalPass)
        return float4(valid ? BilateralVisibility(pixel, extent, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (!valid) return 0.0f;
    float visibility = saturate(PreviousVisibility.Load(int3(pixel, 0)).r);
    return float4(AmbientMetallic.Load(int3(pixel, 0)).rgb * (1.0f - visibility), 0.0f);
}
