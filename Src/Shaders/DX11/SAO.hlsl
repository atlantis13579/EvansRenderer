// McGuire, Mara and Luebke, "Scalable Ambient Obscurance", HPG 2012.
// A native R32 hierarchy supplies nine distributed taps at bounded mip footprints.
// Rotated-grid reduction preserves scene depths; recover the selected base pixel
// before reconstructing positions. Depth is optical distance from the near plane.
// This version uses bounded, edge-aware depth differences for geometric normals
// instead of quad derivatives, and retains separate bilateral reconstruction.
// https://casual-effects.com/research/McGuire2012SAO/McGuire12SAO.pdf
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Scalable-Ambient-Obscurance/McGuire12SAO.pdf
cbuffer ScalableObscuranceConstants : register(b0)
{
    row_major float4x4 ViewProjection;
    row_major float4x4 InverseViewProjection;
    float4 ViewportRadiusBias;
    float4 ForwardIntensity;
    uint4 Options;
};
Texture2D<float> DepthInput : register(t0);
Texture2D<float4> NormalRoughness : register(t1);
Texture2D<float4> AmbientMetallic : register(t2);
Texture2D<float4> PreviousVisibility : register(t3);

static const uint EstimatePass = 0u;
static const uint HorizontalPass = 1u;
static const uint VerticalPass = 2u;
static const uint CompositePass = 3u;
static const uint PrepareDepthPass = 4u;
static const uint ReduceDepthPass = 5u;
static const int SampleCount = 9;
static const float SpiralTurns = 7.0f;
static const float MipRadiusIncrement = 16.0f;
static const float MinimumSampleFootprint = 1.0f;
static const float ObscuranceNormalization = 5.0f;
static const float SquaredDistanceEpsilon = 1.0e-4f;
static const float RayDenominatorEpsilon = 1.0e-12f;
static const float NormalAngleSquaredEpsilon = 1.0e-12f;
static const float HomogeneousEpsilon = 1.0e-7f;
static const float InvalidLinearDepth = -1.0f;
static const float ClearHardwareDepth = 1.0f;
static const float InteriorClipDepth = 0.5f;
static const float PixelCenter = 0.5f;
static const float Pi = 3.141592653589793f;
static const int BilateralRadius = 2;
static const float SpatialSigma = 1.5f;
static const float DepthSigmaRadiusFraction = 0.25f;
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

bool Unproject(float2 pixel, float hardwareDepth, out float3 position)
{
    float2 uv = (pixel + PixelCenter) * ViewportRadiusBias.xy;
    float4 homogeneous = mul(InverseViewProjection,
        float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), hardwareDepth, 1.0f));
    position = 0.0f;
    if (abs(homogeneous.w) < HomogeneousEpsilon) return false;
    position = homogeneous.xyz / homogeneous.w;
    return all(isfinite(position));
}

bool ValidLinearDepth(float depth) { return isfinite(depth) && depth >= 0.0f; }

bool Reconstruct(float2 pixel, float linearDepth, out float3 position)
{
    float3 nearPoint, middlePoint;
    position = 0.0f;
    if (!ValidLinearDepth(linearDepth) || !Unproject(pixel, 0.0f, nearPoint)
        || !Unproject(pixel, InteriorClipDepth, middlePoint)) return false;
    float3 ray = middlePoint - nearPoint;
    float denominator = dot(ray, ForwardIntensity.xyz);
    if (!isfinite(denominator) || abs(denominator) <= RayDenominatorEpsilon) return false;
    position = nearPoint + ray * (linearDepth / denominator);
    return all(isfinite(position));
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

bool ReadDepthSurface(int2 pixel, int2 extent, out float depth, out float3 position)
{
    depth = InvalidLinearDepth;
    position = 0.0f;
    if (any(pixel < 0) || any(pixel >= extent)) return false;
    depth = DepthInput.Load(int3(pixel, 0));
    return Reconstruct(float2(pixel), depth, position);
}

bool ReadSurface(int2 pixel, int2 extent, out float depth, out float3 position)
{
    bool valid = ReadDepthSurface(pixel, extent, depth, position);
    return valid && NormalRoughness.Load(int3(pixel, 0)).w >= 0.0f;
}

bool GeometricNormal(int2 pixel, int2 extent, float centerDepth, float3 center, out float3 normal)
{
    float leftDepth, rightDepth, upDepth, downDepth;
    float3 left, right, up, down;
    bool hasLeft = ReadDepthSurface(pixel + int2(-1, 0), extent, leftDepth, left);
    bool hasRight = ReadDepthSurface(pixel + int2(1, 0), extent, rightDepth, right);
    bool hasUp = ReadDepthSurface(pixel + int2(0, -1), extent, upDepth, up);
    bool hasDown = ReadDepthSurface(pixel + int2(0, 1), extent, downDepth, down);
    normal = 0.0f;
    if ((!hasLeft && !hasRight) || (!hasUp && !hasDown)) return false;
    bool useLeft = hasLeft && (!hasRight || abs(leftDepth - centerDepth) <= abs(rightDepth - centerDepth));
    bool useUp = hasUp && (!hasDown || abs(upDepth - centerDepth) <= abs(downDepth - centerDepth));
    float3 across = useLeft ? center - left : right - center;
    float3 downward = useUp ? center - up : down - center;
    if (!NormalizeVector(across) || !NormalizeVector(downward)) return false;
    normal = cross(downward, across);
    if (dot(normal, normal) <= NormalAngleSquaredEpsilon || !NormalizeVector(normal)) return false;
    if (dot(normal, -ForwardIntensity.xyz) < 0.0f) normal = -normal;
    return true;
}

int2 RotatedChild(int2 parent, int2 sourceExtent)
{
    int2 childOffset = int2((parent.y & 1) ^ 1, (parent.x & 1) ^ 1);
    return clamp(parent * 2 + childOffset, int2(0, 0), sourceExtent - 1);
}

float HierarchicalDepth(int2 pixel)
{
    if (Options.x == PrepareDepthPass)
    {
        float hardwareDepth = DepthInput.Load(int3(pixel, 0));
        float3 position, nearPoint;
        if (hardwareDepth >= ClearHardwareDepth || !Unproject(float2(pixel), hardwareDepth, position)
            || !Unproject(float2(pixel), 0.0f, nearPoint)) return InvalidLinearDepth;
        float depth = dot(position - nearPoint, ForwardIntensity.xyz);
        return isfinite(depth) ? max(depth, 0.0f) : InvalidLinearDepth;
    }
    uint width, height, mipLevels;
    uint sourceMip = Options.y - 1u;
    DepthInput.GetDimensions(sourceMip, width, height, mipLevels);
    return DepthInput.Load(int3(RotatedChild(pixel, int2(width, height)), sourceMip));
}

int2 RepresentativePixel(int2 texel, int mip, int2 baseExtent)
{
    // Follow the same reduction decisions back to an actual original scene sample.
    [loop] for (int level = mip; level > 0; --level)
        texel = RotatedChild(texel, max(int2(1, 1), baseExtent >> (level - 1)));
    return texel;
}

float EstimateObscurance(int2 pixel, int2 extent, float depth, float3 center, float3 normal)
{
    float radius = ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    float rotation = 2.0f * Pi * frac(RotationScale * frac(dot(float2(pixel), RotationPixelDirection)));
    float obscurance = 0.0f;
    [loop] for (int index = 0; index < SampleCount; ++index)
    {
        float alpha = (float(index) + PixelCenter) / SampleCount;
        float angle = 2.0f * Pi * SpiralTurns * alpha + rotation;
        float3 adjacent;
        if (!Reconstruct(float2(pixel) + float2(cos(angle), sin(angle)), depth, adjacent)) continue;
        float3 direction = adjacent - center;
        direction -= ForwardIntensity.xyz * dot(direction, ForwardIntensity.xyz);
        if (!NormalizeVector(direction)) continue;
        float4 projected = mul(ViewProjection, float4(center + radius * direction, 1.0f));
        if (!all(isfinite(projected)) || projected.w <= HomogeneousEpsilon) continue;
        float2 endpointUV = projected.xy / projected.w * float2(PixelCenter, -PixelCenter) + PixelCenter;
        float2 tapOffset = (endpointUV * float2(extent) - (float2(pixel) + PixelCenter)) * alpha;
        if (!all(isfinite(tapOffset))) continue;
        float2 requestedPixel = floor(float2(pixel) + tapOffset + PixelCenter);
        if (any(requestedPixel < 0.0f) || any(requestedPixel >= float2(extent))) continue;
        int mip = clamp(int(floor(log2(max(length(tapOffset), MinimumSampleFootprint) / MipRadiusIncrement))), 0, int(Options.z));
        int2 mipExtent = max(int2(1, 1), extent >> mip);
        int2 mipPixel = clamp(int2(requestedPixel) >> mip, int2(0, 0), mipExtent - 1);
        float sampleDepth = DepthInput.Load(int3(mipPixel, mip));
        int2 selectedPixel = RepresentativePixel(mipPixel, mip, extent);
        float3 samplePosition;
        if (!Reconstruct(float2(selectedPixel), sampleDepth, samplePosition)) continue;
        float3 displacement = samplePosition - center;
        float distanceSquared = dot(displacement, displacement);
        if (!isfinite(distanceSquared) || distanceSquared >= radiusSquared) continue;
        float falloff = max(1.0f - distanceSquared / radiusSquared, 0.0f);
        obscurance += falloff * falloff * falloff * max(dot(displacement, normal) - ViewportRadiusBias.w, 0.0f)
            / (distanceSquared + SquaredDistanceEpsilon);
    }
    return saturate(1.0f - ObscuranceNormalization * ForwardIntensity.w * obscurance / SampleCount);
}

float BilateralVisibility(int2 pixel, int2 extent, float centerDepth)
{
    int2 axis = Options.x == HorizontalPass ? int2(1, 0) : int2(0, 1);
    float depthSigma = ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float weighted = 0.0f, total = 0.0f;
    [unroll] for (int offset = -BilateralRadius; offset <= BilateralRadius; ++offset)
    {
        int2 neighbor = clamp(pixel + axis * offset, int2(0, 0), extent - 1);
        float depth = DepthInput.Load(int3(neighbor, 0));
        if (!ValidLinearDepth(depth) || NormalRoughness.Load(int3(neighbor, 0)).w < 0.0f) continue;
        float difference = depth - centerDepth;
        float weight = exp(-float(offset * offset) / (2.0f * SpatialSigma * SpatialSigma))
            * exp(-difference * difference / (2.0f * depthSigma * depthSigma));
        weighted += PreviousVisibility.Load(int3(neighbor, 0)).r * weight;
        total += weight;
    }
    return total > FilterWeightEpsilon ? weighted / total : 1.0f;
}

float4 PSMain(ScreenVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy);
    if (Options.x == PrepareDepthPass || Options.x == ReduceDepthPass)
        return float4(HierarchicalDepth(pixel), 0.0f, 0.0f, 0.0f);
    if (ForwardIntensity.w <= 0.0f)
        return float4(Options.x == CompositePass ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
    uint width, height, mipLevels;
    DepthInput.GetDimensions(0, width, height, mipLevels);
    int2 extent = int2(width, height);
    float depth;
    float3 center;
    bool valid = ReadSurface(pixel, extent, depth, center);
    if (Options.x == EstimatePass)
    {
        float3 normal = 0.0f;
        bool hasNormal = valid && GeometricNormal(pixel, extent, depth, center, normal);
        return float4(hasNormal ? EstimateObscurance(pixel, extent, depth, center, normal) : 1.0f, 0.0f, 0.0f, 0.0f);
    }
    if (Options.x == HorizontalPass || Options.x == VerticalPass)
        return float4(valid ? BilateralVisibility(pixel, extent, depth) : 1.0f, 0.0f, 0.0f, 0.0f);
    if (!valid) return 0.0f;
    float visibility = saturate(PreviousVisibility.Load(int3(pixel, 0)).r);
    return float4(AmbientMetallic.Load(int3(pixel, 0)).rgb * (1.0f - visibility), 0.0f);
}
