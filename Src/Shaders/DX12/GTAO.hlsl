// Independently authored DX12 Ground-Truth Ambient Occlusion.
// Jimenez et al., "Practical Realtime Strategies for Accurate Indirect Occlusion",
// 2016, equations 5-8: integrate visible cosine-weighted arcs in view-aligned slices.
// This implementation uses eight spatial slices; temporal reuse is optional.
// Periodic normal-hemisphere clipping preserves material normal orientation, and
// division by the full arc quadrature removes finite-slice bias on clear planes.
// https://www.activision.com/cdn/research/PracticalRealtimeStrategiesTRfinal.pdf
cbuffer GroundTruthConstants : register(b0)
{
    row_major float4x4 CameraViewProjection;
    row_major float4x4 InverseCameraViewProjection;
    float4 InverseExtentRadiusBias;
    float4 OpticalForwardIntensity;
    uint4 StageOptions;
};
Texture2D<float> DepthImage : register(t0);
Texture2D<float4> MaterialNormals : register(t1);
Texture2D<float4> AmbientImage : register(t2);
Texture2D<float4> VisibilityImage : register(t3);

static const uint EstimateStage = 0u;
static const uint HorizontalStage = 1u;
static const uint VerticalStage = 2u;
static const uint CompositeStage = 3u;
static const int SpatialSliceCount = 8;
static const int RadialStepCount = 8;
static const int SliceSideCount = 2;
static const int AngularPeriodRadius = 1;
static const int FilterHalfWidth = 2;
static const float Pi = 3.141592653589793f;
static const float HalfPi = Pi * 0.5f;
static const float TwoPi = Pi * 2.0f;
static const float ArcIntegralFactor = 0.25f;
static const float PixelCenter = 0.5f;
static const float InteriorDepth = 0.5f;
static const float BackgroundDepth = 1.0f;
static const float MinimumHomogeneousW = 1.0e-7f;
static const float MinimumIntegralWeight = 1.0e-7f;
static const float HalfNormalRelativeError = 1.0f / 1024.0f;
static const float HalfNormalSubnormalStep = 1.0f / 16777216.0f;
static const float NormalEncodingError = HalfNormalRelativeError + sqrt(3.0f) * HalfNormalSubnormalStep;
static const float NormalEncodingSlopeBound = NormalEncodingError / (1.0f - NormalEncodingError);
static const float RangeFalloffStart = 0.5f;
static const float SpatialStandardDeviation = 1.5f;
static const float DepthStandardDeviationRatio = 0.25f;
static const float NormalSimilarityExponent = 8.0f;
static const float MinimumFilterWeight = 1.0e-7f;
static const float2 HashPixelVector = float2(0.06711056f, 0.00583715f);
static const float HashMultiplier = 52.9829189f;

struct FullscreenVertex { float4 Position : SV_POSITION; };
FullscreenVertex VSMain(uint id : SV_VertexID)
{
    float2 coordinate = float2((id << 1u) & 2u, id & 2u);
    FullscreenVertex result;
    result.Position = float4(coordinate.x * 2.0f - 1.0f, 1.0f - coordinate.y * 2.0f, 0.0f, 1.0f);
    return result;
}

bool MakeUnit(inout float3 direction)
{
    if (!all(isfinite(direction))) return false;
    float largestComponent = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    if (largestComponent <= 0.0f) return false;
    direction /= largestComponent;
    direction *= rsqrt(dot(direction, direction));
    return true;
}

bool WorldFromPixel(float2 texel, float depth, out float3 world)
{
    float2 uv = (texel + PixelCenter) * InverseExtentRadiusBias.xy;
    float4 homogeneous = mul(InverseCameraViewProjection,
        float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), depth, 1.0f));
    world = 0.0f;
    if (!isfinite(homogeneous.w) || abs(homogeneous.w) < MinimumHomogeneousW) return false;
    world = homogeneous.xyz / homogeneous.w;
    return all(isfinite(world));
}

bool LitSurface(int2 texel, out float depth, out float3 world, out float3 normal)
{
    depth = DepthImage.Load(int3(texel, 0));
    float4 encodedNormal = MaterialNormals.Load(int3(texel, 0));
    world = 0.0f;
    normal = encodedNormal.xyz;
    if (depth >= BackgroundDepth || encodedNormal.w < 0.0f || !MakeUnit(normal)) return false;
    return WorldFromPixel(float2(texel), depth, world);
}

float HemispherePrimitive(float theta, float viewCoefficient, float tangentCoefficient)
{
    float doubled = 2.0f * theta;
    return sign(theta) * ArcIntegralFactor * (viewCoefficient * (1.0f - cos(doubled))
        + tangentCoefficient * (doubled - sin(doubled)));
}

float VisibleArcIntegral(float lower, float upper, float viewCoefficient, float tangentCoefficient)
{
    if (viewCoefficient == 0.0f && tangentCoefficient == 0.0f) return 0.0f;
    float gamma = atan2(tangentCoefficient, viewCoefficient);
    float integral = 0.0f;
    [unroll] for (int turn = -AngularPeriodRadius; turn <= AngularPeriodRadius; ++turn)
    {
        float middle = gamma + TwoPi * turn;
        float start = max(lower, middle - HalfPi);
        float finish = min(upper, middle + HalfPi);
        if (finish > start)
            integral += HemispherePrimitive(finish, viewCoefficient, tangentCoefficient)
                - HemispherePrimitive(start, viewCoefficient, tangentCoefficient);
    }
    // Coefficients retain the projected normal's length. A single slice can
    // legitimately exceed unit visibility; normalization occurs after summation.
    return integral;
}

float2 HorizonCosines(int2 texel, int2 dimensions, float3 center, float3 normal, float3 viewDirection, float3 tangent)
{
    float radius = InverseExtentRadiusBias.z;
    float radiusSquared = radius * radius;
    float2 horizons = float2(-1.0f, -1.0f);
    [unroll] for (int sideIndex = 0; sideIndex < SliceSideCount; ++sideIndex)
    {
        float side = sideIndex == 0 ? -1.0f : 1.0f;
        int2 lastTexel = texel;
        [loop] for (int radialStep = 1; radialStep <= RadialStepCount; ++radialStep)
        {
            float fraction = float(radialStep) / RadialStepCount;
            float sampleRadius = radius * fraction * fraction;
            float3 probe = center + tangent * (side * sampleRadius);
            float4 clip = mul(CameraViewProjection, float4(probe, 1.0f));
            if (!all(isfinite(clip)) || clip.w <= MinimumHomogeneousW) continue;
            float2 uv = float2(clip.x / clip.w, -clip.y / clip.w) * PixelCenter + PixelCenter;
            if (any(uv < 0.0f) || any(uv >= 1.0f)) continue;
            int2 nearest = int2(floor(uv * float2(dimensions)));
            if (all(nearest == lastTexel)) continue;
            lastTexel = nearest;
            if (all(nearest == texel)) continue;
            float sampledDepth = DepthImage.Load(int3(nearest, 0));
            float3 sampledWorld;
            if (sampledDepth >= BackgroundDepth || !WorldFromPixel(float2(nearest), sampledDepth, sampledWorld)) continue;
            float3 delta = sampledWorld - center;
            float squaredDistance = dot(delta, delta);
            if (!isfinite(squaredDistance) || squaredDistance >= radiusSquared) continue;
            float distance = sqrt(squaredDistance);
            float signedHeight = dot(normal, delta);
            // One-ULP half conversion error covers truncation and subnormals.
            // The normalized tangent uncertainty q/(1-q) is scale independent.
            if (signedHeight <= InverseExtentRadiusBias.w + distance * NormalEncodingSlopeBound) continue;
            float tangentDistance = dot(delta, tangent);
            if (side * tangentDistance <= 0.0f) continue;
            float2 projected = float2(dot(delta, viewDirection), tangentDistance);
            float2 sliceNormal = float2(dot(normal, viewDirection), dot(normal, tangent));
            float projectedNormalLengthSquared = dot(sliceNormal, sliceNormal);
            float missingHeight = signedHeight - dot(sliceNormal, projected);
            // Recover the height lost with the out-of-slice component. Keeping
            // the common squared-normal factor avoids division near grazing slices.
            projected = projected * projectedNormalLengthSquared + sliceNormal * missingHeight;
            if (side * projected.y <= 0.0f) continue;
            float scale = max(abs(projected.x), abs(projected.y));
            if (!isfinite(scale) || scale <= 0.0f) continue;
            projected /= scale;
            float cosine = projected.x * rsqrt(dot(projected, projected));
            float weight = 1.0f - smoothstep(radius * RangeFalloffStart, radius, distance);
            float weightedCosine = lerp(-1.0f, clamp(cosine, -1.0f, 1.0f), weight);
            horizons[sideIndex] = max(horizons[sideIndex], weightedCosine);
        }
    }
    return horizons;
}

float EvaluateGroundTruth(int2 texel, int2 dimensions, float depth, float3 center, float3 normal)
{
    float3 nearWorld, middleWorld, rightWorld;
    if (!WorldFromPixel(float2(texel), 0.0f, nearWorld)
        || !WorldFromPixel(float2(texel), InteriorDepth, middleWorld)
        || !WorldFromPixel(float2(texel) + float2(1.0f, 0.0f), depth, rightWorld)) return 1.0f;
    float3 viewDirection = nearWorld - middleWorld;
    if (!MakeUnit(viewDirection)) return 1.0f;
    float3 right = rightWorld - center;
    right -= viewDirection * dot(right, viewDirection);
    if (!MakeUnit(right)) return 1.0f;
    float3 down = cross(right, viewDirection);
    float noise = frac(HashMultiplier * frac(dot(float2(texel), HashPixelVector)));
    float visible = 0.0f, unoccluded = 0.0f;
    [loop] for (int sliceIndex = 0; sliceIndex < SpatialSliceCount; ++sliceIndex)
    {
        float phi = (float(sliceIndex) / SpatialSliceCount + noise) * Pi;
        float3 sliceTangent = right * cos(phi) + down * sin(phi);
        float a = dot(normal, viewDirection);
        float b = dot(normal, sliceTangent);
        float2 horizons = HorizonCosines(texel, dimensions, center, normal, viewDirection, sliceTangent);
        float negativeAngle = -acos(clamp(horizons.x, -1.0f, 1.0f));
        float positiveAngle = acos(clamp(horizons.y, -1.0f, 1.0f));
        visible += VisibleArcIntegral(negativeAngle, positiveAngle, a, b);
        unoccluded += VisibleArcIntegral(-Pi, Pi, a, b);
    }
    if (unoccluded <= MinimumIntegralWeight) return 1.0f;
    return saturate(1.0f - OpticalForwardIntensity.w * (1.0f - visible / unoccluded));
}

float FilterVisibility(int2 texel, int2 dimensions, float3 center, float3 normal)
{
    int2 axis = StageOptions.x == HorizontalStage ? int2(1, 0) : int2(0, 1);
    float depthSigma = InverseExtentRadiusBias.z * DepthStandardDeviationRatio;
    float numerator = 0.0f, denominator = 0.0f;
    [unroll] for (int offset = -FilterHalfWidth; offset <= FilterHalfWidth; ++offset)
    {
        int2 neighbor = clamp(texel + axis * offset, int2(0, 0), dimensions - 1);
        float neighborDepth;
        float3 world, neighborNormal;
        if (!LitSurface(neighbor, neighborDepth, world, neighborNormal)) continue;
        float separation = dot(world - center, OpticalForwardIntensity.xyz);
        float spatial = exp(-float(offset * offset) / (2.0f * SpatialStandardDeviation * SpatialStandardDeviation));
        float depthWeight = exp(-separation * separation / (2.0f * depthSigma * depthSigma));
        float normalWeight = pow(saturate(dot(normal, neighborNormal)), NormalSimilarityExponent);
        float weight = spatial * depthWeight * normalWeight;
        numerator += VisibilityImage.Load(int3(neighbor, 0)).r * weight;
        denominator += weight;
    }
    return denominator > MinimumFilterWeight ? numerator / denominator : 1.0f;
}

float4 PSMain(FullscreenVertex input) : SV_TARGET
{
    if (OpticalForwardIntensity.w <= 0.0f)
        return float4(StageOptions.x == CompositeStage ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
    int2 texel = int2(input.Position.xy);
    uint width, height;
    DepthImage.GetDimensions(width, height);
    int2 dimensions = int2(width, height);
    float depth;
    float3 center, normal;
    bool valid = LitSurface(texel, depth, center, normal);
    if (StageOptions.x == EstimateStage)
        return float4(valid ? EvaluateGroundTruth(texel, dimensions, depth, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (StageOptions.x == HorizontalStage || StageOptions.x == VerticalStage)
        return float4(valid ? FilterVisibility(texel, dimensions, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (!valid) return 0.0f;
    float visibility = saturate(VisibilityImage.Load(int3(texel, 0)).r);
    return float4(AmbientImage.Load(int3(texel, 0)).rgb * (1.0f - visibility), 0.0f);
}
