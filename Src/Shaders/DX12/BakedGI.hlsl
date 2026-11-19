// Native D3D12 diffuse bake projection. Input rows are immutable CPU ray results,
// not color captured from the main camera. Projection is rerun only for changed
// samples or angular moment filtering, and is published after command submission.
// Irradiance SH: https://graphics.stanford.edu/papers/envmap/
// Directional visibility moments: https://jcgt.org/published/0008/02/01/
cbuffer BakeUniforms : register(b0)
{
    row_major float4x4 InverseViewProjection;
    float4 GridMinimum, GridMaximum, Viewport, Appearance, Visibility;
    uint4 Dimensions, Options;
};
Texture2D<float4> Samples : register(t0);
static const float Pi = 3.14159265358979323846f, StorageScale = 0.25f;
static const float HalfFloatLimit = 65504, KernelWeightFloor = 1e-12f, TexelCenter = 0.5f;
static const uint SampleBorder = 1, SHProjection = 0;
struct FullscreenOutput { float4 Position : SV_Position; };
FullscreenOutput VSMain(uint vertex : SV_VertexID)
{
    FullscreenOutput result;
    float2 uv = float2((vertex << 1u) & 2u, vertex & 2u);
    result.Position = float4(uv * 2 - 1, 0, 1); return result;
}
float Basis(uint index, float3 omega)
{
    switch (index)
    {
    case 0: return sqrt(1.0f / (4 * Pi));
    case 1: return omega.y * sqrt(3.0f / (4 * Pi));
    case 2: return omega.z * sqrt(3.0f / (4 * Pi));
    case 3: return omega.x * sqrt(3.0f / (4 * Pi));
    case 4: return omega.x * omega.y * sqrt(15.0f / (4 * Pi));
    case 5: return omega.y * omega.z * sqrt(15.0f / (4 * Pi));
    case 6: return (3 * omega.z * omega.z - 1) * sqrt(5.0f / (16 * Pi));
    case 7: return omega.x * omega.z * sqrt(15.0f / (4 * Pi));
    default: return (omega.x * omega.x - omega.y * omega.y) * sqrt(15.0f / (16 * Pi));
    }
}
float BandIntegral(uint index) { return index == 0 ? 1 : index <= 3 ? 2.0f / 3.0f : 1.0f / 4.0f; }
float3 OctDirection(uint2 cell)
{
    float2 p = ((float2(cell) + TexelCenter) / Options.z) * 2 - 1;
    float3 n = float3(p, 1 - abs(p.x) - abs(p.y));
    float folded = max(-n.z, 0);
    n.xy += float2(n.x >= 0 ? -folded : folded, n.y >= 0 ? -folded : folded);
    return normalize(n);
}
float4 PSMain(FullscreenOutput raster) : SV_Target0
{
    uint2 location = uint2(raster.Position.xy);
    if (Options.x == SHProjection)
    {
        float3 coefficient = 0;
        [loop] for (uint sampleIndex = 0; sampleIndex < Dimensions.w; ++sampleIndex)
        {
            uint x = sampleIndex + SampleBorder;
            float4 ray = Samples.Load(int3(x, 0, 0));
            coefficient += Samples.Load(int3(x, location.y + SampleBorder, 0)).rgb * (ray.w * Basis(location.x, ray.xyz));
        }
        coefficient *= BandIntegral(location.x) * StorageScale;
        float valid = Samples.Load(int3(0, location.y + SampleBorder, 0)).a;
        return float4(clamp(coefficient, -HalfFloatLimit, HalfFloatLimit), valid);
    }
    uint2 tile = location / Options.z;
    uint probeIndex = tile.x + tile.y * uint(Visibility.w);
    if (probeIndex >= Options.w) return float4(1, 1, 0, 0);
    float3 axis = OctDirection(location - tile * Options.z);
    float weightSum = 0, distanceSum = 0, squareSum = 0;
    [loop] for (uint sampleIndex = 0; sampleIndex < Dimensions.w; ++sampleIndex)
    {
        uint x = sampleIndex + SampleBorder;
        float3 omega = Samples.Load(int3(x, 0, 0)).xyz;
        float d = saturate(Samples.Load(int3(x, probeIndex + SampleBorder, 0)).a / Visibility.z);
        float weight = pow(saturate(dot(omega, axis)), Visibility.y);
        weightSum += weight; distanceSum += weight * d; squareSum += weight * d * d;
    }
    if (weightSum <= KernelWeightFloor) return float4(1, 1, 0, 1);
    return float4(distanceSum / weightSum, squareSum / weightSum, 0, 1);
}
