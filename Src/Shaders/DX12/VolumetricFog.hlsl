// DirectX 12 height-medium transport. The 128-byte parameter block is recorded
// as root constants; descriptor tables name immutable scene/depth/atlas views.
// In the constant ambient-source model, cumulative radiance is L * (1 - T).
// Integrating optical depth first avoids repeated RGB accumulation. This identity
// does not extend to spatially varying illumination without a scattering integral.
// Reference: Physically Based Rendering, "Transmittance" (Beer-Lambert law).
// https://pbr-book.org/4ed/Volume_Scattering/Transmittance

cbuffer VolumeParameters : register(b4)
{
    row_major float4x4 InverseRasterCamera;
    float4 HeightMedium; // base extinction, base height, falloff, ray-distance cap
    float4 AmbientSource;
    float4 ViewAndGridSize;
    uint4 VolumeControl; // phase, endpoint count, samples per interval, enabled
};
Texture2D<float4> OpaqueRadiance : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> IntegratedVolume : register(t14);
SamplerState VolumeClampSampler : register(s1);

static const uint VolumeIntegrationPhase = 0;
static const float OpticalDepthLimit = 80.0f;
static const float RayLengthFloor = 1.0e-6f;
static const float HalfFloatRadianceLimit = 65504.0f;

float4 VSMain(uint vertexId : SV_VertexID) : SV_POSITION
{
    const float2 corners[3] = {float2(-1, 1), float2(3, 1), float2(-1, -3)};
    return float4(corners[vertexId], 0, 1);
}
float3 UnprojectPoint(float2 screenUV, float depth)
{
    float2 ndc = float2(screenUV.x * 2 - 1, 1 - screenUV.y * 2);
    float4 homogeneousPoint = mul(InverseRasterCamera, float4(ndc, depth, 1));
    return homogeneousPoint.xyz / homogeneousPoint.w;
}
struct VolumeRay
{
    float3 Origin;
    float3 Direction;
    float Distance;
};
VolumeRay CameraRay(float2 uv)
{
    VolumeRay ray;
    ray.Origin = UnprojectPoint(uv, 0);
    float3 delta = UnprojectPoint(uv, 1) - ray.Origin;
    float lengthToFar = max(length(delta), RayLengthFloor);
    ray.Direction = delta / lengthToFar;
    ray.Distance = min(lengthToFar, HeightMedium.w);
    return ray;
}
float4 IntegratePrefix(uint2 destination)
{
    uint columns = (uint)ViewAndGridSize.z;
    uint endpoint = destination.x / columns + 1;
    float2 rayUV = (float2(destination.x % columns, destination.y) + 0.5f) / ViewAndGridSize.zw;
    VolumeRay ray = CameraRay(rayUV);
    float intervalLength = ray.Distance / (VolumeControl.y * VolumeControl.z);
    float opticalDepth = 0;
    uint sampleCount = endpoint * VolumeControl.z;
    [loop] for (uint interval = 0; interval < sampleCount; ++interval)
    {
        float midpoint = (float(interval) + 0.5f) * intervalLength;
        float heightAboveBase = max(ray.Origin.y + ray.Direction.y * midpoint - HeightMedium.y, 0);
        float localExtinction = HeightMedium.x * exp(-min(heightAboveBase * HeightMedium.z, OpticalDepthLimit));
        opticalDepth += localExtinction * intervalLength;
    }
    float transmission = exp(-min(opticalDepth, OpticalDepthLimit));
    return float4(AmbientSource.rgb * (1 - transmission), transmission);
}
float4 ReadPrefix(float2 uv, uint boundary)
{
    if (boundary == 0) return float4(0, 0, 0, 1);
    float2 gridPosition = clamp(uv * ViewAndGridSize.zw, 0.5f, ViewAndGridSize.zw - 0.5f);
    gridPosition.x += float(boundary - 1) * ViewAndGridSize.z;
    float2 atlasSize = ViewAndGridSize.zw * float2(VolumeControl.y, 1);
    return IntegratedVolume.SampleLevel(VolumeClampSampler, gridPosition / atlasSize, 0);
}
float4 ResolveOpaque(uint2 pixel, float2 uv)
{
    VolumeRay ray = CameraRay(uv);
    float3 surfacePoint = UnprojectPoint(uv, OpaqueDepth.Load(int3(pixel, 0)));
    float distanceFraction = saturate(length(surfacePoint - ray.Origin) / max(ray.Distance, RayLengthFloor));
    float slicePosition = distanceFraction * VolumeControl.y;
    uint first = (uint)slicePosition;
    uint second = min(first + 1, VolumeControl.y);
    float4 transport = lerp(ReadPrefix(uv, first), ReadPrefix(uv, second), slicePosition - first);
    float4 surface = OpaqueRadiance.Load(int3(pixel, 0));
    return float4(clamp(surface.rgb * transport.w + transport.rgb, 0, HalfFloatRadianceLimit), surface.a);
}
float4 PSMain(float4 rasterPosition : SV_POSITION) : SV_TARGET
{
    uint2 pixel = (uint2)rasterPosition.xy;
    if (VolumeControl.x == VolumeIntegrationPhase) return IntegratePrefix(pixel);
    return ResolveOpaque(pixel, rasterPosition.xy / ViewAndGridSize.xy);
}
