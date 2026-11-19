// Bavoil, Sainz and Dimitrov, "Image-Space Horizon-Based Ambient Occlusion", 2008.
// Integrate rising sine-horizon bands with the NVIDIA SDK's quadratic distance
// attenuation. Correct each band's tangent for perspective and texel snapping;
// this equals the original sine-horizon integral at a fixed azimuth. The depth
// heightfield remains an approximation at grazing angles and hidden surfaces.
// https://rdimitrov.twistedsanity.net/HBAO_SIGGRAPH08.pdf
// https://developer.download.nvidia.com/SDK/10.5/direct3d/Source/ScreenSpaceAO/doc/ScreenSpaceAO.pdf
cbuffer HorizonConstants : register(b0)
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

static const uint EstimatePass = 0u;
static const uint HorizontalPass = 1u;
static const uint VerticalPass = 2u;
static const int DirectionCount = 8;
static const int StepsPerDirection = 8;
static const int BilateralRadius = 2;
static const float Pi = 3.141592653589793f;
static const float TexelCenterOffset = 0.5f;
static const float HomogeneousEpsilon = 1.0e-7f;
static const float VectorLengthSquaredEpsilon = 1.0e-12f;
static const float TangentDenominatorEpsilon = 1.0e-6f;
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

bool ReconstructPosition(float2 pixel, float depth, out float3 position)
{
    float2 uv = (pixel + TexelCenterOffset) * ViewportRadiusBias.xy;
    float4 homogeneous = mul(InverseViewProjection, float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f));
    position = 0.0f;
    if (abs(homogeneous.w) < HomogeneousEpsilon) return false;
    position = homogeneous.xyz / homogeneous.w;
    return all(isfinite(position));
}

bool ReadSurface(int2 pixel, out float3 position, out float3 normal)
{
    float depth = OpaqueDepth.Load(int3(pixel, 0));
    float4 encodedNormal = NormalRoughness.Load(int3(pixel, 0));
    float normalSquared = dot(encodedNormal.xyz, encodedNormal.xyz);
    position = normal = 0.0f;
    if (depth >= ClearDepth || encodedNormal.w < 0.0f || normalSquared <= VectorLengthSquaredEpsilon) return false;
    normal = encodedNormal.xyz * rsqrt(normalSquared);
    return ReconstructPosition(float2(pixel), depth, position);
}

float HorizonVisibility(int2 pixel, int2 extent, float3 center, float3 normal)
{
    float radius = ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    float depth = OpaqueDepth.Load(int3(pixel, 0));
    float3 viewAxis = -ForwardIntensity.xyz;
    float rotation = 2.0f * Pi * frac(RotationScale * frac(dot(float2(pixel), RotationPixelDirection)));
    float occlusion = 0.0f;
    [loop] for (int directionIndex = 0; directionIndex < DirectionCount; ++directionIndex)
    {
        float angle = rotation + 2.0f * Pi * float(directionIndex) / DirectionCount;
        float3 neighboringPoint;
        if (!ReconstructPosition(float2(pixel) + float2(cos(angle), sin(angle)), depth, neighboringPoint)) continue;
        float3 screenDirection = neighboringPoint - center;
        screenDirection -= viewAxis * dot(screenDirection, viewAxis);
        float directionSquared = dot(screenDirection, screenDirection);
        if (directionSquared <= VectorLengthSquaredEpsilon) continue;
        screenDirection *= rsqrt(directionSquared);

        // Project a world-space disk, avoiding a hardcoded focal length or FOV.
        float4 endpoint = mul(ViewProjection, float4(center + screenDirection * radius, 1.0f));
        if (!all(isfinite(endpoint)) || endpoint.w <= HomogeneousEpsilon) continue;
        float2 endpointUV = float2(endpoint.x / endpoint.w, -endpoint.y / endpoint.w) * 0.5f + 0.5f;
        float2 pixelOffset = endpointUV * float2(extent) - (float2(pixel) + TexelCenterOffset);
        if (!all(isfinite(pixelOffset))) continue;
        int2 previousPixel = pixel;
        float largestHorizonBand = 0.0f;
        [loop] for (int step = 1; step <= StepsPerDirection; ++step)
        {
            float fraction = float(step) / StepsPerDirection;
            float2 snappedPixel = floor(float2(pixel) + pixelOffset * fraction + TexelCenterOffset);
            if (any(snappedPixel < 0.0f) || any(snappedPixel >= float2(extent))) continue;
            int2 samplePixel = int2(snappedPixel);
            if (all(samplePixel == previousPixel)) continue;
            previousPixel = samplePixel;
            if (all(samplePixel == pixel)) continue;
            float sampleDepth = OpaqueDepth.Load(int3(samplePixel, 0));
            float3 samplePosition;
            if (sampleDepth >= ClearDepth || !ReconstructPosition(float2(samplePixel), sampleDepth, samplePosition)) continue;
            float3 displacement = samplePosition - center;
            float distanceSquared = dot(displacement, displacement);
            if (distanceSquared <= VectorLengthSquaredEpsilon || distanceSquared >= radiusSquared) continue;
            float elevation = dot(displacement, viewAxis);
            float3 planarDirection = displacement - viewAxis * elevation;
            float planarSquared = dot(planarDirection, planarDirection);
            if (planarSquared <= VectorLengthSquaredEpsilon) continue;
            planarDirection *= rsqrt(planarSquared);
            float sineHorizon = elevation * rsqrt(distanceSquared);
            float tangent = atan2(-dot(normal, planarDirection), max(dot(normal, viewAxis), TangentDenominatorEpsilon));
            float biasedTangent = clamp(tangent + ViewportRadiusBias.w, -Pi * 0.5f, Pi * 0.5f);
            float horizonBand = max(sineHorizon - sin(biasedTangent), 0.0f);
            // A constant tangent is incorrect after perspective and grid snapping.
            // Track increasing tangent-corrected bands, so a tilted plane stays lit.
            float newBand = max(horizonBand - largestHorizonBand, 0.0f);
            occlusion += newBand * (1.0f - distanceSquared / radiusSquared);
            largestHorizonBand = max(largestHorizonBand, horizonBand);
        }
    }
    return saturate(1.0f - ForwardIntensity.w * occlusion / DirectionCount);
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
    // Zero intensity remains exact through FP16 filtering and ambient subtraction.
    if (ForwardIntensity.w <= 0.0f)
        return float4(Options.x == EstimatePass || Options.x == HorizontalPass || Options.x == VerticalPass
            ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    int2 pixel = int2(input.Position.xy);
    uint width, height;
    OpaqueDepth.GetDimensions(width, height);
    int2 extent = int2(width, height);
    float3 center, normal;
    bool valid = ReadSurface(pixel, center, normal);
    if (Options.x == EstimatePass)
        return float4(valid ? HorizonVisibility(pixel, extent, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (Options.x == HorizontalPass || Options.x == VerticalPass)
        return float4(valid ? BilateralVisibility(pixel, extent, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (!valid) return 0.0f;
    float visibility = saturate(PreviousVisibility.Load(int3(pixel, 0)).r);
    return float4(AmbientMetallic.Load(int3(pixel, 0)).rgb * (1.0f - visibility), 0.0f);
}
