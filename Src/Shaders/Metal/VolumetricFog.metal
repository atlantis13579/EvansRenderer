#include <metal_stdlib>
using namespace metal;
// Independent Metal integration of a procedural height medium. Each atlas texel
// owns a cumulative prefix with shared midpoint sample locations. For constant
// source radiance, each segment has analytic Beer-Lambert transport.
// Pharr, Jakob, Humphreys, Physically Based Rendering, "Transmittance".
// https://pbr-book.org/4ed/Volume_Scattering/Transmittance
struct FogParameters
{
    float4 CameraInverseRows[4];
    float4 ExtinctionHeightFalloffRange;
    float4 SourceRadiance;
    float4 ViewportCellCount;
    uint4 PassSlicesSamplesEnabled;
};
struct FogVertex { float4 Position [[position]]; };
constant uint FogIntegrationPass = 0;
constant float FogMidpoint = 0.5;
constant float FogMinimumRayExtent = 1.0e-6;
constant float FogOpaqueExponent = 80.0;
constant float FogRadianceCeiling = 65504.0;

vertex FogVertex FogVertexMain(uint vertexIndex [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 p = vertexIndex == 0 ? float2(-1.0) :
        vertexIndex == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(p, 0.0, 1.0)};
}
float3 FogUnproject(float2 pixelUV, float deviceDepth, constant FogParameters& parameters)
{
    // Metal fragment positions and texture rows start at the top-left. Explicit
    // rows consume the host's row-major camera without a matrix transpose.
    float4 clip = float4(pixelUV * float2(2.0, -2.0) + float2(-1.0, 1.0), deviceDepth, 1.0);
    float4 world = float4(dot(parameters.CameraInverseRows[0], clip), dot(parameters.CameraInverseRows[1], clip),
        dot(parameters.CameraInverseRows[2], clip), dot(parameters.CameraInverseRows[3], clip));
    return world.xyz / world.w;
}
float4 FogIntegrate(uint2 atlasPixel, constant FogParameters& parameters)
{
    uint cellColumns = uint(parameters.ViewportCellCount.z);
    uint prefixLength = atlasPixel.x / cellColumns + 1;
    float2 cellUV = (float2(atlasPixel.x % cellColumns, atlasPixel.y) + FogMidpoint) / parameters.ViewportCellCount.zw;
    float3 start = FogUnproject(cellUV, 0.0, parameters);
    float3 ray = FogUnproject(cellUV, 1.0, parameters) - start;
    float rayExtent = max(length(ray), FogMinimumRayExtent);
    float sampleDistance = min(rayExtent, parameters.ExtinctionHeightFalloffRange.w) /
        float(parameters.PassSlicesSamplesEnabled.y * parameters.PassSlicesSamplesEnabled.z);
    float verticalStep = ray.y * sampleDistance / rayExtent;
    float4 cumulative = float4(0.0, 0.0, 0.0, 1.0);
    uint intervalCount = prefixLength * parameters.PassSlicesSamplesEnabled.z;
    for (uint interval = 0; interval < intervalCount; ++interval)
    {
        float height = start.y + (float(interval) + FogMidpoint) * verticalStep;
        float relativeHeight = max(height - parameters.ExtinctionHeightFalloffRange.y, 0.0);
        float density = parameters.ExtinctionHeightFalloffRange.x *
            exp(-min(parameters.ExtinctionHeightFalloffRange.z * relativeHeight, FogOpaqueExponent));
        float transmission = exp(-min(density * sampleDistance, FogOpaqueExponent));
        cumulative.rgb += cumulative.a * (1.0 - transmission) * parameters.SourceRadiance.rgb;
        cumulative.a *= transmission;
    }
    return cumulative;
}
float4 FogBoundary(float2 pixelUV, uint boundary, constant FogParameters& parameters, texture2d<float> atlas)
{
    if (boundary == 0) return float4(0.0, 0.0, 0.0, 1.0);
    float2 halfCell = FogMidpoint / parameters.ViewportCellCount.zw;
    float2 packedUV = clamp(pixelUV, halfCell, 1.0 - halfCell);
    packedUV.x = (float(boundary - 1) + packedUV.x) / float(parameters.PassSlicesSamplesEnabled.y);
    constexpr sampler PrefixSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    return atlas.sample(PrefixSampler, packedUV, level(0.0));
}
fragment float4 FogFragmentMain(FogVertex input [[stage_in]],
    constant FogParameters& parameters [[buffer(0)]],
    texture2d<float, access::read> scene [[texture(0)]],
    depth2d<float, access::read> sceneDepth [[texture(1)]],
    texture2d<float> atlas [[texture(2)]])
{
    uint2 pixel = uint2(input.Position.xy);
    if (parameters.PassSlicesSamplesEnabled.x == FogIntegrationPass)
        return FogIntegrate(pixel, parameters);
    float2 pixelUV = input.Position.xy / parameters.ViewportCellCount.xy;
    float3 start = FogUnproject(pixelUV, 0.0, parameters);
    float rayExtent = min(length(FogUnproject(pixelUV, 1.0, parameters) - start), parameters.ExtinctionHeightFalloffRange.w);
    float surfaceDistance = length(FogUnproject(pixelUV, sceneDepth.read(pixel), parameters) - start);
    float endpoint = clamp(surfaceDistance / max(rayExtent, FogMinimumRayExtent), 0.0, 1.0) * float(parameters.PassSlicesSamplesEnabled.y);
    uint first = uint(endpoint), second = min(first + 1, parameters.PassSlicesSamplesEnabled.y);
    float4 medium = mix(FogBoundary(pixelUV, first, parameters, atlas),
        FogBoundary(pixelUV, second, parameters, atlas), endpoint - float(first));
    float4 surface = scene.read(pixel);
    return float4(clamp(surface.rgb * medium.a + medium.rgb, 0.0, FogRadianceCeiling), surface.a);
}
