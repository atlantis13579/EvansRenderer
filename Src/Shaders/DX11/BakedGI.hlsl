// DX11 probe projection: radiance samples -> L2 irradiance/pi and distance moments.
// Ramamoorthi/Hanrahan, SIGGRAPH 2001: https://graphics.stanford.edu/papers/envmap/
// Majercik et al., JCGT 2019: https://jcgt.org/published/0008/02/01/
// Quarter-scale SH storage preserves the source half-float HDR range. Distances
// are normalized before squaring. Invalid probes retain an explicit alpha flag.
cbuffer ProbeConstants : register(b0)
{
    row_major float4x4 InverseViewProjection;
    float4 GridMinimum, GridMaximum, Viewport, Appearance, Visibility;
    uint4 Dimensions, Options;
};
Texture2D<float4> RaySamples : register(t0);
static const float Pi = 3.14159265358979323846f;
static const float CoefficientStorageScale = 0.25f, MaximumHalfFloat = 65504;
static const float MinimumKernelWeight = 1e-12f, PixelCenter = 0.5f;
static const uint MetadataBorder = 1, IrradiancePhase = 0, BandOneEnd = 4;
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint id : SV_VertexID)
{
    ScreenVertex output;
    float2 corner = float2((id << 1) & 2, id & 2);
    output.Position = float4(corner * 2 - 1, 0, 1); return output;
}
float SH(uint coefficient, float3 direction)
{
    if (coefficient == 0) return sqrt(1 / (4 * Pi));
    if (coefficient == 1) return sqrt(3 / (4 * Pi)) * direction.y;
    if (coefficient == 2) return sqrt(3 / (4 * Pi)) * direction.z;
    if (coefficient == 3) return sqrt(3 / (4 * Pi)) * direction.x;
    if (coefficient == 4) return sqrt(15 / (4 * Pi)) * direction.x * direction.y;
    if (coefficient == 5) return sqrt(15 / (4 * Pi)) * direction.y * direction.z;
    if (coefficient == 6) return sqrt(5 / (16 * Pi)) * (3 * direction.z * direction.z - 1);
    if (coefficient == 7) return sqrt(15 / (4 * Pi)) * direction.x * direction.z;
    return sqrt(15 / (16 * Pi)) * (direction.x * direction.x - direction.y * direction.y);
}
float3 DecodeOctahedron(float2 uv)
{
    float2 square = uv * 2 - 1;
    float3 direction = float3(square, 1 - abs(square.x) - abs(square.y));
    float fold = saturate(-direction.z);
    direction.x += direction.x >= 0 ? -fold : fold;
    direction.y += direction.y >= 0 ? -fold : fold;
    return normalize(direction);
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    uint2 pixel = uint2(input.Position.xy);
    if (Options.x == IrradiancePhase)
    {
        uint probe = pixel.y, coefficient = pixel.x;
        float active = RaySamples.Load(int3(0, probe + MetadataBorder, 0)).w;
        float3 sum = 0;
        [loop] for (uint ray = 0; ray < Dimensions.w; ++ray)
        {
            float4 direction = RaySamples.Load(int3(ray + MetadataBorder, 0, 0));
            float3 radiance = RaySamples.Load(int3(ray + MetadataBorder, probe + MetadataBorder, 0)).rgb;
            sum += radiance * (SH(coefficient, direction.xyz) * direction.w);
        }
        // Clamped-cosine convolution divided by pi: bands 0, 1, 2.
        float convolution = coefficient == 0 ? 1 : coefficient < BandOneEnd ? 2.0f / 3.0f : 0.25f;
        return float4(clamp(sum * (convolution * CoefficientStorageScale), -MaximumHalfFloat, MaximumHalfFloat), active);
    }
    uint tileSize = Options.z, columns = uint(Visibility.w);
    uint probe = pixel.x / tileSize + (pixel.y / tileSize) * columns;
    if (probe >= Options.w) return float4(1, 1, 0, 0);
    float3 axis = DecodeOctahedron((float2(pixel % tileSize) + PixelCenter) / tileSize);
    float2 moments = 0; float weights = 0;
    [loop] for (uint ray = 0; ray < Dimensions.w; ++ray)
    {
        float3 direction = RaySamples.Load(int3(ray + MetadataBorder, 0, 0)).xyz;
        float distance = saturate(RaySamples.Load(int3(ray + MetadataBorder, probe + MetadataBorder, 0)).w / Visibility.z);
        float weight = pow(saturate(dot(axis, direction)), Visibility.y);
        moments += weight * float2(distance, distance * distance); weights += weight;
    }
    return float4(weights > MinimumKernelWeight ? moments / weights : float2(1, 1), 0, 1);
}
