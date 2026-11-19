// Integrated height fog, independent DirectX 11 implementation.
// Beer-Lambert transport: S += T * L * (1 - exp(-sigmaT * ds)); T *= exp(-sigmaT * ds).
// L is prescribed isotropic ambient equilibrium radiance, not shadowed lights.
// Reference: Pharr, Jakob, Humphreys, Physically Based Rendering, "Transmittance".
// https://pbr-book.org/4ed/Volume_Scattering/Transmittance
// The atlas represents cumulative camera-ray transport, including transparent
// fragment endpoints. No authored density texture or compute shader is needed.

cbuffer FogConstants : register(b3)
{
    row_major float4x4 FogInverseViewProjection;
    float4 FogMedium;
    float4 FogSourceRadiance;
    float4 FogViewportGrid;
    uint4 FogOptions;
};
Texture2D<float4> SceneColor : register(t0);
Texture2D<float> SceneDepth : register(t1);
Texture2D<float4> FogAtlas : register(t14);
SamplerState FogLinearClamp : register(s3);

static const uint IntegratePass = 0;
static const float MaximumOpticalDepth = 80.0f;
static const float MinimumRayLength = 1.0e-6f;
static const float MaximumHalfRadiance = 65504.0f;
static const float4 IdentityTransport = float4(0, 0, 0, 1);

float4 VSMain(uint index : SV_VertexID) : SV_POSITION
{
    float2 uv = float2((index << 1) & 2, index & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float3 WorldPosition(float2 uv, float depth)
{
    float4 world = mul(FogInverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    return world.xyz / world.w;
}
float4 EndpointTransport(float2 uv, uint endpoint)
{
    if (endpoint == 0) return IdentityTransport;
    float2 halfCell = 0.5f / FogViewportGrid.zw;
    float2 cellUV = clamp(uv, halfCell, 1 - halfCell);
    cellUV.x = (cellUV.x + endpoint - 1) / FogOptions.y;
    return FogAtlas.SampleLevel(FogLinearClamp, cellUV, 0);
}
float4 IntegratedTransport(float2 uv, float depth)
{
    float3 start = WorldPosition(uv, 0);
    float rayExtent = min(length(WorldPosition(uv, 1) - start), FogMedium.w);
    float endpoint = saturate(length(WorldPosition(uv, depth) - start) / max(rayExtent, MinimumRayLength)) * FogOptions.y;
    uint lower = (uint)floor(endpoint);
    uint upper = min(lower + 1, FogOptions.y);
    return lerp(EndpointTransport(uv, lower), EndpointTransport(uv, upper), frac(endpoint));
}
float4 Integrate(float4 position)
{
    uint gridWidth = (uint)FogViewportGrid.z;
    uint2 pixel = (uint2)position.xy;
    uint slice = pixel.x / gridWidth;
    float2 uv = (float2(pixel.x % gridWidth, pixel.y) + 0.5f) / FogViewportGrid.zw;
    float3 start = WorldPosition(uv, 0);
    float3 ray = WorldPosition(uv, 1) - start;
    float rayLength = max(length(ray), MinimumRayLength);
    float stepLength = min(rayLength, FogMedium.w) / (FogOptions.y * FogOptions.z);
    float3 direction = ray / rayLength;
    uint count = (slice + 1) * FogOptions.z;
    float4 transport = IdentityTransport;
    [loop] for (uint sampleIndex = 0; sampleIndex < count; ++sampleIndex)
    {
        float height = start.y + direction.y * (sampleIndex + 0.5f) * stepLength;
        float extinction = FogMedium.x * exp(-min(FogMedium.z * max(height - FogMedium.y, 0), MaximumOpticalDepth));
        float segmentT = exp(-min(extinction * stepLength, MaximumOpticalDepth));
        transport.rgb += transport.a * FogSourceRadiance.rgb * (1 - segmentT);
        transport.a *= segmentT;
    }
    return transport;
}
float4 PSMain(float4 position : SV_POSITION) : SV_TARGET
{
    if (FogOptions.x == IntegratePass) return Integrate(position);
    int2 pixel = int2(position.xy);
    float4 color = SceneColor.Load(int3(pixel, 0));
    float4 transport = IntegratedTransport(position.xy / FogViewportGrid.xy, SceneDepth.Load(int3(pixel, 0)));
    return float4(clamp(color.rgb * transport.a + transport.rgb, 0, MaximumHalfRadiance), color.a);
}
