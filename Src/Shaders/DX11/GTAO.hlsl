// Jimenez et al., "Practical Realtime Strategies for Accurate Indirect Occlusion",
// 2016, equations 5-8. Integrate the cosine-weighted visible arc analytically.
// Eight spatial slices replace temporal sample reuse. Periodic hemisphere clips
// retain back-facing material normals; projected-normal length is in the two
// integral coefficients. An unoccluded integral normalizes finite-slice variance.
// https://www.activision.com/cdn/research/PracticalRealtimeStrategiesTRfinal.pdf
cbuffer GroundTruthOcclusionConstants : register(b0)
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
static const uint CompositePass = 3u;
static const int SliceCount = 8;
static const int StepsPerSide = 8;
static const int BilateralRadius = 2;
static const int PeriodicWindowRadius = 1;
static const float Pi = 3.141592653589793f;
static const float HalfPi = Pi * 0.5f;
static const float FullTurn = Pi * 2.0f;
static const float IntegralScale = 0.25f;
static const float TexelCenter = 0.5f;
static const float InteriorClipDepth = 0.5f;
static const float ClearDepth = 1.0f;
static const float HomogeneousEpsilon = 1.0e-7f;
static const float MinimumBaselineIntegral = 1.0e-7f;
static const float HalfNormalRelativeError = 1.0f / 1024.0f;
static const float HalfNormalSubnormalStep = 1.0f / 16777216.0f;
static const float NormalEncodingError = HalfNormalRelativeError + sqrt(3.0f) * HalfNormalSubnormalStep;
static const float NormalEncodingSlopeBound = NormalEncodingError / (1.0f - NormalEncodingError);
static const float RadiusFadeStartFraction = 0.5f;
static const float SpatialSigma = 1.5f;
static const float DepthSigmaRadiusFraction = 0.25f;
static const float NormalWeightExponent = 8.0f;
static const float FilterWeightEpsilon = 1.0e-7f;
static const float2 RotationPixelDirection = float2(0.06711056f, 0.00583715f);
static const float RotationScale = 52.9829189f;

struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    float2 corner = float2((vertexId << 1u) & 2u, vertexId & 2u);
    ScreenVertex vertex;
    vertex.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return vertex;
}

bool NormalizeVector(inout float3 value)
{
    if (!all(isfinite(value))) return false;
    float scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (scale <= 0.0f) return false;
    value /= scale;
    value *= rsqrt(dot(value, value));
    return true;
}

bool ReconstructPosition(float2 pixel, float depth, out float3 position)
{
    float2 uv = (pixel + TexelCenter) * ViewportRadiusBias.xy;
    float4 homogeneous = mul(InverseViewProjection,
        float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f));
    position = 0.0f;
    if (!isfinite(homogeneous.w) || abs(homogeneous.w) < HomogeneousEpsilon) return false;
    position = homogeneous.xyz / homogeneous.w;
    return all(isfinite(position));
}

bool ReadSurface(int2 pixel, out float depth, out float3 position, out float3 normal)
{
    depth = OpaqueDepth.Load(int3(pixel, 0));
    float4 materialNormal = NormalRoughness.Load(int3(pixel, 0));
    position = 0.0f;
    normal = materialNormal.xyz;
    if (depth >= ClearDepth || materialNormal.w < 0.0f || !NormalizeVector(normal)) return false;
    return ReconstructPosition(float2(pixel), depth, position);
}

float SignedArcPrimitive(float angle, float normalView, float normalTangent)
{
    float twiceAngle = 2.0f * angle;
    float primitive = IntegralScale * (normalView * (1.0f - cos(twiceAngle))
        + normalTangent * (twiceAngle - sin(twiceAngle)));
    return sign(angle) * primitive;
}

float IntegrateSlice(float lower, float upper, float normalView, float normalTangent)
{
    if (normalView == 0.0f && normalTangent == 0.0f) return 0.0f;
    float normalAngle = atan2(normalTangent, normalView);
    float result = 0.0f;
    // Intersect with every positive-cosine interval that can overlap [-pi, pi].
    // This also handles a normal hemisphere crossing the angular branch cut.
    [unroll] for (int period = -PeriodicWindowRadius; period <= PeriodicWindowRadius; ++period)
    {
        float windowCenter = normalAngle + FullTurn * period;
        float clippedLower = max(lower, windowCenter - HalfPi);
        float clippedUpper = min(upper, windowCenter + HalfPi);
        if (clippedUpper > clippedLower)
            result += SignedArcPrimitive(clippedUpper, normalView, normalTangent)
                - SignedArcPrimitive(clippedLower, normalView, normalTangent);
    }
    return result;
}

float SearchHorizon(int2 pixel, int2 extent, float3 center, float3 normal,
    float3 viewDirection, float3 tangent, float side)
{
    float radius = ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    float horizonCosine = -1.0f;
    int2 previousPixel = pixel;
    [loop] for (int step = 1; step <= StepsPerSide; ++step)
    {
        float fraction = float(step) / StepsPerSide;
        float radialDistance = radius * fraction * fraction;
        float4 clip = mul(ViewProjection, float4(center + side * tangent * radialDistance, 1.0f));
        if (!all(isfinite(clip)) || clip.w <= HomogeneousEpsilon) continue;
        float2 uv = clip.xy / clip.w * float2(TexelCenter, -TexelCenter) + TexelCenter;
        if (any(uv < 0.0f) || any(uv >= 1.0f)) continue;
        int2 samplePixel = int2(floor(uv * float2(extent)));
        if (all(samplePixel == previousPixel)) continue;
        previousPixel = samplePixel;
        if (all(samplePixel == pixel)) continue;
        float depth = OpaqueDepth.Load(int3(samplePixel, 0));
        float3 samplePosition;
        if (depth >= ClearDepth || !ReconstructPosition(float2(samplePixel), depth, samplePosition)) continue;
        float3 delta = samplePosition - center;
        float distanceSquared = dot(delta, delta);
        if (!isfinite(distanceSquared) || distanceSquared >= radiusSquared) continue;
        float distance = sqrt(distanceSquared);
        float worldHeight = dot(normal, delta);
        // Allow one binary16 ULP, including truncating render-target conversion.
        // Normalization bounds the resulting tangent slope by q/(1-q).
        if (worldHeight <= ViewportRadiusBias.w + distance * NormalEncodingSlopeBound) continue;
        float alongTangent = dot(delta, tangent);
        if (side * alongTangent <= 0.0f) continue;
        float2 sliceDelta = float2(dot(delta, viewDirection), alongTangent);
        float2 projectedNormal = float2(dot(normal, viewDirection), dot(normal, tangent));
        // Preserve signed surface height when snapping moves the sample out of
        // this slice. The omitted common divisor does not change the angle.
        sliceDelta = sliceDelta * dot(projectedNormal, projectedNormal)
            + projectedNormal * (worldHeight - dot(projectedNormal, sliceDelta));
        if (side * sliceDelta.y <= 0.0f) continue;
        float scale = max(abs(sliceDelta.x), abs(sliceDelta.y));
        if (!isfinite(scale) || scale <= 0.0f) continue;
        sliceDelta /= scale;
        float cosine = sliceDelta.x * rsqrt(dot(sliceDelta, sliceDelta));
        float fade = 1.0f - smoothstep(radius * RadiusFadeStartFraction, radius, distance);
        float candidate = lerp(-1.0f, clamp(cosine, -1.0f, 1.0f), fade);
        horizonCosine = max(horizonCosine, candidate);
    }
    return horizonCosine;
}

float GroundTruthVisibility(int2 pixel, int2 extent, float depth, float3 center, float3 normal)
{
    float3 nearPoint, middlePoint, neighboringPoint;
    if (!ReconstructPosition(float2(pixel), 0.0f, nearPoint)
        || !ReconstructPosition(float2(pixel), InteriorClipDepth, middlePoint)
        || !ReconstructPosition(float2(pixel) + float2(1.0f, 0.0f), depth, neighboringPoint)) return 1.0f;
    float3 viewDirection = nearPoint - middlePoint;
    if (!NormalizeVector(viewDirection)) return 1.0f;
    float3 firstTangent = neighboringPoint - center;
    firstTangent -= viewDirection * dot(firstTangent, viewDirection);
    if (!NormalizeVector(firstTangent)) return 1.0f;
    float3 secondTangent = cross(firstTangent, viewDirection);
    float rotation = frac(RotationScale * frac(dot(float2(pixel), RotationPixelDirection)));
    float visibleIntegral = 0.0f, baselineIntegral = 0.0f;
    [loop] for (int slice = 0; slice < SliceCount; ++slice)
    {
        float azimuth = (float(slice) / SliceCount + rotation) * Pi;
        float3 tangent = cos(azimuth) * firstTangent + sin(azimuth) * secondTangent;
        float normalView = dot(normal, viewDirection);
        float normalTangent = dot(normal, tangent);
        float negativeHorizon = SearchHorizon(pixel, extent, center, normal, viewDirection, tangent, -1.0f);
        float positiveHorizon = SearchHorizon(pixel, extent, center, normal, viewDirection, tangent, 1.0f);
        float lower = -acos(clamp(negativeHorizon, -1.0f, 1.0f));
        float upper = acos(clamp(positiveHorizon, -1.0f, 1.0f));
        visibleIntegral += IntegrateSlice(lower, upper, normalView, normalTangent);
        baselineIntegral += IntegrateSlice(-Pi, Pi, normalView, normalTangent);
    }
    if (baselineIntegral <= MinimumBaselineIntegral) return 1.0f;
    return saturate(1.0f - ForwardIntensity.w * (1.0f - visibleIntegral / baselineIntegral));
}

float BilateralVisibility(int2 pixel, int2 extent, float3 center, float3 normal)
{
    int2 axis = Options.x == HorizontalPass ? int2(1, 0) : int2(0, 1);
    float depthSigma = ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float weighted = 0.0f, total = 0.0f;
    [unroll] for (int offset = -BilateralRadius; offset <= BilateralRadius; ++offset)
    {
        int2 neighbor = clamp(pixel + axis * offset, int2(0, 0), extent - 1);
        float depth;
        float3 position, neighborNormal;
        if (!ReadSurface(neighbor, depth, position, neighborNormal)) continue;
        float difference = dot(position - center, ForwardIntensity.xyz);
        float spatialWeight = exp(-float(offset * offset) / (2.0f * SpatialSigma * SpatialSigma));
        float depthWeight = exp(-difference * difference / (2.0f * depthSigma * depthSigma));
        float weight = spatialWeight * depthWeight * pow(saturate(dot(normal, neighborNormal)), NormalWeightExponent);
        weighted += weight * PreviousVisibility.Load(int3(neighbor, 0)).r;
        total += weight;
    }
    return total > FilterWeightEpsilon ? weighted / total : 1.0f;
}

float4 PSMain(ScreenVertex fragment) : SV_TARGET
{
    if (ForwardIntensity.w <= 0.0f)
        return float4(Options.x == CompositePass ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
    int2 pixel = int2(fragment.Position.xy);
    uint width, height;
    OpaqueDepth.GetDimensions(width, height);
    int2 extent = int2(width, height);
    float depth;
    float3 center, normal;
    bool valid = ReadSurface(pixel, depth, center, normal);
    if (Options.x == EstimatePass)
        return float4(valid ? GroundTruthVisibility(pixel, extent, depth, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (Options.x == HorizontalPass || Options.x == VerticalPass)
        return float4(valid ? BilateralVisibility(pixel, extent, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (!valid) return 0.0f;
    float visibility = saturate(PreviousVisibility.Load(int3(pixel, 0)).r);
    return float4(AmbientMetallic.Load(int3(pixel, 0)).rgb * (1.0f - visibility), 0.0f);
}
