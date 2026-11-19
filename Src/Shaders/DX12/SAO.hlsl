// Independently authored DX12 Scalable Ambient Obscurance.
// McGuire, Mara and Luebke, HPG 2012: a depth hierarchy bounds the working set of
// a fixed spiral estimator. The updated appendix supplies the cubic radial falloff.
// Depth is measured from the near plane. Edge-aware finite differences recover
// face normals, and each coarse depth is reconstructed at its selected base texel.
// https://casual-effects.com/research/McGuire2012SAO/McGuire12SAO.pdf
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Scalable-Ambient-Obscurance/McGuire12SAO.pdf
cbuffer ScalableAmbientConstants : register(b0)
{
    row_major float4x4 CameraViewProjection;
    row_major float4x4 InverseCameraViewProjection;
    float4 InverseExtentRadiusBias;
    float4 OpticalForwardStrength;
    uint4 StageOptions;
};
Texture2D<float> SceneDepth : register(t0);
Texture2D<float4> MaterialNormal : register(t1);
Texture2D<float4> AmbientRadiance : register(t2);
Texture2D<float4> InputVisibility : register(t3);

static const uint EvaluateStage = 0u;
static const uint HorizontalStage = 1u;
static const uint VerticalStage = 2u;
static const uint CompositeStage = 3u;
static const uint LinearizeStage = 4u;
static const uint DownsampleStage = 5u;
static const int SpiralSampleCount = 9;
static const float SpiralRevolutions = 7.0f;
static const float MipSelectionRadius = 16.0f;
static const float MinimumPixelFootprint = 1.0f;
static const float ObscuranceNormalization = 5.0f;
static const float SquaredDistanceEpsilon = 1.0e-4f;
static const float MinimumRayProjection = 1.0e-12f;
static const float MinimumNormalSineSquared = 1.0e-12f;
static const float MinimumHomogeneousW = 1.0e-7f;
static const float EmptyDepth = -1.0f;
static const float HardwareFarDepth = 1.0f;
static const float InteriorDepth = 0.5f;
static const float TexelCenter = 0.5f;
static const float Pi = 3.141592653589793f;
static const int FilterHalfWidth = 2;
static const float SpatialDeviation = 1.5f;
static const float DepthDeviationRadiusRatio = 0.25f;
static const float MinimumFilterWeight = 1.0e-7f;
static const float2 RotationCoefficients = float2(0.06711056f, 0.00583715f);
static const float RotationMultiplier = 52.9829189f;

struct FullscreenVertex { float4 Position : SV_POSITION; };
FullscreenVertex VSMain(uint index : SV_VertexID)
{
    float2 corner = float2((index << 1u) & 2u, index & 2u);
    FullscreenVertex output;
    output.Position = float4(corner.x * 2.0f - 1.0f, 1.0f - corner.y * 2.0f, 0.0f, 1.0f);
    return output;
}

bool WorldAtHardwareDepth(float2 pixel, float depth, out float3 world)
{
    float2 uv = (pixel + TexelCenter) * InverseExtentRadiusBias.xy;
    float4 homogeneous = mul(InverseCameraViewProjection,
        float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f));
    world = 0.0f;
    if (abs(homogeneous.w) < MinimumHomogeneousW) return false;
    world = homogeneous.xyz / homogeneous.w;
    return all(isfinite(world));
}

bool HasDepth(float value) { return value >= 0.0f && isfinite(value); }

bool WorldAtLinearDepth(float2 pixel, float depth, out float3 world)
{
    float3 nearWorld, interiorWorld;
    world = 0.0f;
    if (!HasDepth(depth) || !WorldAtHardwareDepth(pixel, 0.0f, nearWorld)
        || !WorldAtHardwareDepth(pixel, InteriorDepth, interiorWorld)) return false;
    float3 ray = interiorWorld - nearWorld;
    float projection = dot(ray, OpticalForwardStrength.xyz);
    if (!isfinite(projection) || abs(projection) <= MinimumRayProjection) return false;
    world = nearWorld + ray * (depth / projection);
    return all(isfinite(world));
}

bool MakeUnit(inout float3 direction)
{
    if (!all(isfinite(direction))) return false;
    float largest = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    if (largest <= 0.0f) return false;
    direction /= largest;
    direction *= rsqrt(dot(direction, direction));
    return true;
}

int2 SelectedChild(int2 cell, int2 previousSize)
{
    int horizontal = (cell.y & 1) ^ 1;
    int vertical = (cell.x & 1) ^ 1;
    return clamp(cell * 2 + int2(horizontal, vertical), int2(0, 0), previousSize - 1);
}

float BuildDepthLevel(int2 cell)
{
    if (StageOptions.x == LinearizeStage)
    {
        float hardware = SceneDepth.Load(int3(cell, 0));
        float3 surface, nearSurface;
        if (hardware >= HardwareFarDepth || !WorldAtHardwareDepth(float2(cell), hardware, surface)
            || !WorldAtHardwareDepth(float2(cell), 0.0f, nearSurface)) return EmptyDepth;
        float linearDepth = dot(surface - nearSurface, OpticalForwardStrength.xyz);
        return isfinite(linearDepth) ? max(linearDepth, 0.0f) : EmptyDepth;
    }
    uint width, height, levels;
    uint previousLevel = StageOptions.y - 1u;
    SceneDepth.GetDimensions(previousLevel, width, height, levels);
    return SceneDepth.Load(int3(SelectedChild(cell, int2(width, height)), previousLevel));
}

bool DepthPosition(int2 texel, int2 size, out float depth, out float3 world)
{
    depth = EmptyDepth;
    world = 0.0f;
    if (any(texel < 0) || any(texel >= size)) return false;
    depth = SceneDepth.Load(int3(texel, 0));
    return WorldAtLinearDepth(float2(texel), depth, world);
}

bool LitPosition(int2 texel, int2 size, out float depth, out float3 world)
{
    bool valid = DepthPosition(texel, size, depth, world);
    return valid && MaterialNormal.Load(int3(texel, 0)).w >= 0.0f;
}

bool DepthNormal(int2 texel, int2 size, float centerDepth, float3 center, out float3 normal)
{
    float zLeft, zRight, zUp, zDown;
    float3 pLeft, pRight, pUp, pDown;
    bool leftValid = DepthPosition(texel - int2(1, 0), size, zLeft, pLeft);
    bool rightValid = DepthPosition(texel + int2(1, 0), size, zRight, pRight);
    bool upValid = DepthPosition(texel - int2(0, 1), size, zUp, pUp);
    bool downValid = DepthPosition(texel + int2(0, 1), size, zDown, pDown);
    normal = 0.0f;
    if (!(leftValid || rightValid) || !(upValid || downValid)) return false;
    bool chooseLeft = leftValid && (!rightValid || abs(zLeft - centerDepth) <= abs(zRight - centerDepth));
    bool chooseUp = upValid && (!downValid || abs(zUp - centerDepth) <= abs(zDown - centerDepth));
    float3 rightDerivative = chooseLeft ? center - pLeft : pRight - center;
    float3 downDerivative = chooseUp ? center - pUp : pDown - center;
    // Normalize derivatives first so the validity threshold measures their angle,
    // independently of camera distance or the size of a pixel in world units.
    if (!MakeUnit(rightDerivative) || !MakeUnit(downDerivative)) return false;
    normal = cross(downDerivative, rightDerivative);
    if (dot(normal, normal) <= MinimumNormalSineSquared || !MakeUnit(normal)) return false;
    if (dot(normal, -OpticalForwardStrength.xyz) < 0.0f) normal = -normal;
    return true;
}

int2 OriginalSample(int2 cell, int mip, int2 size)
{
    [loop] for (int level = mip; level != 0; --level)
        cell = SelectedChild(cell, max(int2(1, 1), size >> (level - 1)));
    return cell;
}

float ScalableVisibility(int2 texel, int2 size, float centerDepth, float3 center, float3 normal)
{
    float radius = InverseExtentRadiusBias.z;
    float radiusSquared = radius * radius;
    float rotation = frac(RotationMultiplier * frac(dot(float2(texel), RotationCoefficients))) * 2.0f * Pi;
    float sum = 0.0f;
    [loop] for (int tap = 0; tap < SpiralSampleCount; ++tap)
    {
        float alpha = (float(tap) + TexelCenter) / SpiralSampleCount;
        float azimuth = 2.0f * Pi * SpiralRevolutions * alpha + rotation;
        float3 screenNeighbor;
        if (!WorldAtLinearDepth(float2(texel) + float2(cos(azimuth), sin(azimuth)), centerDepth, screenNeighbor)) continue;
        float3 radial = screenNeighbor - center;
        radial -= OpticalForwardStrength.xyz * dot(radial, OpticalForwardStrength.xyz);
        if (!MakeUnit(radial)) continue;
        float4 endpoint = mul(CameraViewProjection, float4(center + radial * radius, 1.0f));
        if (!all(isfinite(endpoint)) || endpoint.w <= MinimumHomogeneousW) continue;
        float2 endpointUV = float2(endpoint.x / endpoint.w, -endpoint.y / endpoint.w) * TexelCenter + TexelCenter;
        float2 offset = (endpointUV * float2(size) - (float2(texel) + TexelCenter)) * alpha;
        if (!all(isfinite(offset))) continue;
        float2 requested = floor(float2(texel) + offset + TexelCenter);
        if (any(requested < 0.0f) || any(requested >= float2(size))) continue;
        float footprint = max(length(offset), MinimumPixelFootprint);
        int mip = clamp(int(floor(log2(footprint / MipSelectionRadius))), 0, int(StageOptions.z));
        int2 levelSize = max(int2(1, 1), size >> mip);
        int2 cell = clamp(int2(requested) >> mip, int2(0, 0), levelSize - 1);
        float sampledDepth = SceneDepth.Load(int3(cell, mip));
        int2 original = OriginalSample(cell, mip, size);
        float3 occluder;
        if (!WorldAtLinearDepth(float2(original), sampledDepth, occluder)) continue;
        float3 delta = occluder - center;
        float squaredDistance = dot(delta, delta);
        if (!isfinite(squaredDistance) || squaredDistance >= radiusSquared) continue;
        float radialFalloff = max(1.0f - squaredDistance / radiusSquared, 0.0f);
        float aboveSurface = max(dot(delta, normal) - InverseExtentRadiusBias.w, 0.0f);
        sum += radialFalloff * radialFalloff * radialFalloff * aboveSurface / (squaredDistance + SquaredDistanceEpsilon);
    }
    return saturate(1.0f - ObscuranceNormalization * OpticalForwardStrength.w * sum / SpiralSampleCount);
}

float ReconstructVisibility(int2 texel, int2 size, float centerDepth)
{
    int2 axis = StageOptions.x == HorizontalStage ? int2(1, 0) : int2(0, 1);
    float depthSigma = InverseExtentRadiusBias.z * DepthDeviationRadiusRatio;
    float weightedSum = 0.0f, weightSum = 0.0f;
    [unroll] for (int tap = -FilterHalfWidth; tap <= FilterHalfWidth; ++tap)
    {
        int2 adjacent = clamp(texel + axis * tap, int2(0, 0), size - 1);
        float depth = SceneDepth.Load(int3(adjacent, 0));
        if (!HasDepth(depth) || MaterialNormal.Load(int3(adjacent, 0)).w < 0.0f) continue;
        float delta = depth - centerDepth;
        float spatial = exp(-float(tap * tap) / (2.0f * SpatialDeviation * SpatialDeviation));
        float range = exp(-delta * delta / (2.0f * depthSigma * depthSigma));
        float weight = spatial * range;
        weightedSum += weight * InputVisibility.Load(int3(adjacent, 0)).r;
        weightSum += weight;
    }
    return weightSum > MinimumFilterWeight ? weightedSum / weightSum : 1.0f;
}

float4 PSMain(FullscreenVertex fragment) : SV_TARGET
{
    int2 texel = int2(fragment.Position.xy);
    if (StageOptions.x == LinearizeStage || StageOptions.x == DownsampleStage)
        return float4(BuildDepthLevel(texel), 0.0f, 0.0f, 0.0f);
    // Hierarchy construction still executes at zero strength; the visible result
    // remains an exact identity through half-float filtering and RGB subtraction.
    if (OpticalForwardStrength.w <= 0.0f)
        return float4(StageOptions.x == CompositeStage ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
    uint width, height, mipCount;
    SceneDepth.GetDimensions(0, width, height, mipCount);
    int2 size = int2(width, height);
    float depth;
    float3 world;
    bool lit = LitPosition(texel, size, depth, world);
    if (StageOptions.x == EvaluateStage)
    {
        float3 normal = 0.0f;
        bool reconstructable = lit && DepthNormal(texel, size, depth, world, normal);
        return float4(reconstructable ? ScalableVisibility(texel, size, depth, world, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    }
    if (StageOptions.x == HorizontalStage || StageOptions.x == VerticalStage)
        return float4(lit ? ReconstructVisibility(texel, size, depth) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (!lit) return 0.0f;
    float visible = saturate(InputVisibility.Load(int3(texel, 0)).r);
    return float4(AmbientRadiance.Load(int3(texel, 0)).rgb * (1.0f - visible), 0.0f);
}
