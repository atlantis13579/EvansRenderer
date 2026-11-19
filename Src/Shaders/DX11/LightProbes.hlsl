// DX11 world-space probe gather. Eight regular-grid neighbors combine trilinear,
// normal-facing and moment visibility weights; invalid probes contribute nothing.
// Majercik et al., JCGT 2019: https://jcgt.org/published/0008/02/01/
// L2 irradiance SH: https://graphics.stanford.edu/papers/envmap/
// This opaque diffuse pass is independent of AO and specular reflections.
cbuffer ProbeConstants : register(b0)
{
    row_major float4x4 InverseViewProjection;
    float4 GridMinimum, GridMaximum, Viewport, Appearance, Visibility;
    uint4 Dimensions, Options;
};
Texture2D<float4> Coefficients : register(t1);
Texture2D<float4> DistanceMoments : register(t2);
Texture2D<float4> SceneColor : register(t3);
Texture2D<float> SceneDepth : register(t4);
Texture2D<float4> NormalRoughness : register(t5);
Texture2D<float4> DiffuseOpacity : register(t6);
Texture2D<float4> AmbientMetallic : register(t7);
static const float Pi = 3.14159265358979323846f, CoefficientDecodeScale = 4;
static const float ProjectionEpsilon = 1e-7f, VectorSquaredEpsilon = 1e-12f;
static const float MinimumFacingWeight = 0.05f, MinimumGatherWeight = 1e-6f;
static const float DielectricReflectance = 0.04f, HalfFloatMaximum = 65504, PixelCenter = 0.5f;
static const uint CoefficientCount = 9, PBRModel = 3;
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint id : SV_VertexID)
{
    ScreenVertex result; float2 uv = float2((id << 1) & 2, id & 2);
    result.Position = float4(uv * 2 - 1, 0, 1); return result;
}
float Basis(uint index, float3 n)
{
    if (index == 0) return sqrt(1 / (4 * Pi));
    if (index == 1) return sqrt(3 / (4 * Pi)) * n.y;
    if (index == 2) return sqrt(3 / (4 * Pi)) * n.z;
    if (index == 3) return sqrt(3 / (4 * Pi)) * n.x;
    if (index == 4) return sqrt(15 / (4 * Pi)) * n.x * n.y;
    if (index == 5) return sqrt(15 / (4 * Pi)) * n.y * n.z;
    if (index == 6) return sqrt(5 / (16 * Pi)) * (3 * n.z * n.z - 1);
    if (index == 7) return sqrt(15 / (4 * Pi)) * n.x * n.z;
    return sqrt(15 / (16 * Pi)) * (n.x * n.x - n.y * n.y);
}
float2 EncodeOct(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 p = n.xy;
    if (n.z < 0) p = (1 - abs(p.yx)) * float2(p.x >= 0 ? 1 : -1, p.y >= 0 ? 1 : -1);
    return p * 0.5f + 0.5f;
}
int2 FoldTexel(int2 cell)
{
    int size = int(Options.z);
    if (cell.x < 0) { cell.x = -cell.x - 1; cell.y = size - 1 - cell.y; }
    if (cell.x >= size) { cell.x = 2 * size - 1 - cell.x; cell.y = size - 1 - cell.y; }
    if (cell.y < 0) { cell.y = -cell.y - 1; cell.x = size - 1 - cell.x; }
    if (cell.y >= size) { cell.y = 2 * size - 1 - cell.y; cell.x = size - 1 - cell.x; }
    return cell;
}
float2 ReadMoments(uint probe, float3 direction)
{
    float2 raster = EncodeOct(direction) * Options.z - PixelCenter;
    int2 base = int2(floor(raster)); float2 fraction = frac(raster);
    uint columns = uint(Visibility.w);
    int2 tile = int2(probe % columns, probe / columns) * int(Options.z);
    float2 result = 0;
    [unroll] for (int y = 0; y < 2; ++y) [unroll] for (int x = 0; x < 2; ++x)
    {
        float weight = (x == 0 ? 1 - fraction.x : fraction.x) * (y == 0 ? 1 - fraction.y : fraction.y);
        result += DistanceMoments.Load(int3(tile + FoldTexel(base + int2(x, y)), 0)).xy * weight;
    }
    return result;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy); float4 scene = SceneColor.Load(int3(pixel, 0));
    float depth = SceneDepth.Load(int3(pixel, 0)); float4 surface = NormalRoughness.Load(int3(pixel, 0));
    if (depth >= 1 || surface.w < 0 || !all(isfinite(surface.xyz)) || dot(surface.xyz, surface.xyz) <= VectorSquaredEpsilon) return scene;
    float2 uv = input.Position.xy * Viewport.zw;
    float4 homogeneous = mul(InverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    if (abs(homogeneous.w) <= ProjectionEpsilon || !all(isfinite(homogeneous))) return scene;
    float3 world = homogeneous.xyz / homogeneous.w, normal = normalize(surface.xyz);
    float3 grid = (world - GridMinimum.xyz) / (GridMaximum.xyz - GridMinimum.xyz) * float3(Dimensions.xyz - 1);
    if (any(grid < 0) || any(grid > float3(Dimensions.xyz - 1))) return scene;
    uint3 cell = min(uint3(floor(grid)), Dimensions.xyz - 2); float3 fraction = grid - float3(cell);
    float3 biased = world + normal * Appearance.y;
    float3 irradiance = 0; float weightSum = 0;
    [unroll] for (uint z = 0; z < 2; ++z) [unroll] for (uint y = 0; y < 2; ++y) [unroll] for (uint x = 0; x < 2; ++x)
    {
        uint3 coordinate = cell + uint3(x, y, z);
        uint probe = coordinate.x + Dimensions.x * (coordinate.y + Dimensions.y * coordinate.z);
        if (Coefficients.Load(int3(0, probe, 0)).w <= 0) continue;
        float3 position = lerp(GridMinimum.xyz, GridMaximum.xyz, float3(coordinate) / float3(Dimensions.xyz - 1));
        float3 toProbe = position - world, fromProbe = biased - position;
        float rayLength = length(fromProbe);
        float3 direction = rayLength > ProjectionEpsilon ? fromProbe / rayLength : normal;
        float2 moments = ReadMoments(probe, direction);
        float delta = max((rayLength - Appearance.z) / Visibility.z - moments.x, 0);
        float variance = max(moments.y - moments.x * moments.x, Visibility.x);
        float visibility = pow(saturate(variance / max(variance + delta * delta, VectorSquaredEpsilon)), Appearance.w);
        float facing = dot(toProbe, toProbe) > VectorSquaredEpsilon ? dot(normal, normalize(toProbe)) : 1;
        float normalWeight = max((facing + 1) * 0.5f, MinimumFacingWeight);
        float3 trilinear = float3(x == 0 ? 1 - fraction.x : fraction.x, y == 0 ? 1 - fraction.y : fraction.y, z == 0 ? 1 - fraction.z : fraction.z);
        float weight = trilinear.x * trilinear.y * trilinear.z * normalWeight * normalWeight * visibility;
        float3 value = 0;
        [unroll] for (uint coefficient = 0; coefficient < CoefficientCount; ++coefficient)
            value += Coefficients.Load(int3(coefficient, probe, 0)).rgb * Basis(coefficient, normal);
        irradiance += max(value * CoefficientDecodeScale, 0) * weight; weightSum += weight;
    }
    if (weightSum <= MinimumGatherWeight) return scene;
    float3 diffuse = saturate(DiffuseOpacity.Load(int3(pixel, 0)).rgb);
    if (Options.y == PBRModel) diffuse *= (1 - DielectricReflectance) * (1 - saturate(AmbientMetallic.Load(int3(pixel, 0)).w));
    return float4(clamp(scene.rgb + irradiance / weightSum * diffuse * Appearance.x, 0, HalfFloatMaximum), scene.a);
}
