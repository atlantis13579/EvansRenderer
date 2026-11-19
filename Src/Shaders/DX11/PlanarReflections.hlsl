// DX11 planar receiver lookup and HDR composition, independently authored.
// The CPU mirrors the camera and replaces its near plane with the receiver plane
// using oblique clipping (Lengyel, "Oblique View Frustum Depth Projection and
// Clipping", 2005). The capture excludes the receiver and the opposite half-space.
// Reconstruct receivers from main depth, but project them with the UNCLIPPED
// mirror matrix: capture bias deliberately puts them behind its near plane.
// Distance and normal tests restrict this single-plane method to planar surfaces.
// Ordinary image mips approximate roughness; they are not a GGX convolution.
cbuffer PlanarReflectionConstants : register(b0)
{
    row_major float4x4 InverseViewProjection;
    row_major float4x4 ReflectionViewProjection;
    float4 ReceiverPlane;
    float4 Viewport; // main extent, reciprocal extent
    float4 Sampling; // intensity, last mip, plane tolerance, minimum normal cosine
    uint4 Options; // phase, surface model, active capture, reserved
    float4 ReceiverMinimum; // world bounds, w: finite aperture enabled
    float4 ReceiverMaximum;
};
Texture2D<float4> SceneColor : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> NormalRoughness : register(t2);
Texture2D<float4> SpecularShininess : register(t3);
Texture2D<float4> DiffuseOpacity : register(t4);
Texture2D<float4> AmbientMetallic : register(t5);
Texture2D<float4> ReflectionRadiance : register(t6);
Texture2D<float4> PlanarRadiance : register(t9);
Texture2D<float4> MipSource : register(t10);
SamplerState TrilinearClamp : register(s0);

static const uint CompositePhase = 1u, PhysicalShading = 3u;
static const float BackgroundDepth = 1.0f;
static const float PixelCenter = 0.5f;
static const float InteriorDepth = 0.5f; // Infinite projections have a singular far endpoint.
static const float MinimumProjectiveW = 1e-7f;
static const float MinimumSquaredLength = 1e-12f;
static const float DielectricReflectance = 0.04f;
static const float FresnelPower = 5.0f;
static const float MaximumHalfRadiance = 65504.0f;
static const float HalfRoundingGuard = 32.0f;

struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    float2 corner = float2((vertexId << 1u) & 2u, vertexId & 2u);
    ScreenVertex output;
    output.Position = float4(corner * float2(2, -2) + float2(-1, 1), 0, 1);
    return output;
}
bool WorldPosition(float2 uv, float depth, out float3 position)
{
    float4 world = mul(InverseViewProjection, float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1));
    position = 0;
    if (!all(isfinite(world)) || abs(world.w) <= MinimumProjectiveW) return false;
    position = world.xyz / world.w;
    return all(isfinite(position));
}
bool UnitDirection(inout float3 direction)
{
    float squaredLength = dot(direction, direction);
    if (!all(isfinite(direction)) || squaredLength <= MinimumSquaredLength) return false;
    direction *= rsqrt(squaredLength);
    return true;
}
float4 PSMain(ScreenVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy);
    if (Options.x == CompositePhase) return float4(ReflectionRadiance.Load(int3(pixel, 0)).rgb, 0);
    float depth = OpaqueDepth.Load(int3(pixel, 0));
    if (Options.z == 0u || Sampling.x <= 0 || depth >= BackgroundDepth) return 0;
    float4 materialNormal = NormalRoughness.Load(int3(pixel, 0));
    float3 normal = materialNormal.xyz;
    if (materialNormal.w < 0 || !UnitDirection(normal) || abs(dot(normal, ReceiverPlane.xyz)) < Sampling.w) return 0;
    float2 uv = (float2(pixel) + PixelCenter) * Viewport.zw;
    float3 receiver, nearPoint, interiorPoint;
    if (!WorldPosition(uv, depth, receiver) || abs(dot(ReceiverPlane, float4(receiver, 1))) > Sampling.z) return 0;
    if (ReceiverMinimum.w != 0 && (any(receiver < ReceiverMinimum.xyz) || any(receiver > ReceiverMaximum.xyz))) return 0;
    float4 projected = mul(ReflectionViewProjection, float4(receiver, 1));
    if (!all(isfinite(projected)) || projected.w <= MinimumProjectiveW) return 0;
    float2 mirrorUV = float2(projected.x / projected.w + 1, 1 - projected.y / projected.w) / 2;
    if (any(mirrorUV < 0) || any(mirrorUV > 1)) return 0;
    if (!WorldPosition(uv, 0, nearPoint) || !WorldPosition(uv, InteriorDepth, interiorPoint)) return 0;
    float3 incident = interiorPoint - nearPoint;
    if (!UnitDirection(incident)) return 0;
    if (ReceiverMinimum.w != 0.0)
    {
        // Project the change in reflection direction from the flat mirror to the
        // local wave facet. Limit the footprint where the capture runs out.
        float3 bend = reflect(incident, normal) - reflect(incident, ReceiverPlane.xyz);
        float4 delta = mul(ReflectionViewProjection, float4(bend, 0));
        float2 offset = (delta.xy - projected.xy / projected.w * delta.w) * float2(0.14, -0.14);
        float2 margin = min(mirrorUV, 1.0 - mirrorUV);
        mirrorUV += clamp(offset, -margin, margin);
    }

    float roughness = saturate(materialNormal.w);
    float mip = roughness * Sampling.y;
    mip = max(mip, PlanarRadiance.CalculateLevelOfDetail(TrilinearClamp, mirrorUV));
    float3 radiance = max(PlanarRadiance.SampleLevel(TrilinearClamp, mirrorUV, mip).rgb, 0);
    float3 baseReflectance = saturate(SpecularShininess.Load(int3(pixel, 0)).rgb);
    if (Options.y == PhysicalShading)
    {
        float metallic = saturate(AmbientMetallic.Load(int3(pixel, 0)).a);
        float3 baseColor = saturate(DiffuseOpacity.Load(int3(pixel, 0)).rgb);
        baseReflectance = lerp(baseReflectance, baseColor, metallic);
    }
    // Plane orientation is chosen per camera; either side may act as a receiver.
    float cosine = saturate(abs(dot(normal, -incident)));
    float3 fresnel = baseReflectance + (1 - baseReflectance) * pow(1 - cosine, FresnelPower);
    float3 reflected = radiance * fresnel * Sampling.x;
    float3 headroom = max(MaximumHalfRadiance - HalfRoundingGuard - SceneColor.Load(int3(pixel, 0)).rgb, 0);
    return float4(min(reflected, headroom), 0); // RGB-only additive composition preserves main alpha/depth.
}

float4 PSMip(ScreenVertex input) : SV_TARGET
{
    // Odd viewport extents need area integration. A single bilinear sample can
    // discard whole rows in a 3-to-1 reduction and erase small reflected lights.
    // Each destination footprint overlaps at most three source texels per axis.
    static const int MaximumMipFootprint = 3;
    float2 footprintSize = Viewport.zw / Viewport.xy; // source extent / destination extent
    float2 begin = floor(input.Position.xy) * footprintSize;
    float2 end = begin + footprintSize;
    int2 first = int2(floor(begin));
    float4 integral = 0;
    [unroll] for (int y = 0; y < MaximumMipFootprint; ++y)
    [unroll] for (int x = 0; x < MaximumMipFootprint; ++x)
    {
        int2 texel = first + int2(x, y);
        float2 overlap = max(min(end, float2(texel + 1)) - max(begin, float2(texel)), 0);
        float weight = overlap.x * overlap.y;
        if (weight > 0 && all(texel < int2(Viewport.zw))) integral += MipSource.Load(int3(texel, 0)) * weight;
    }
    return integral / (footprintSize.x * footprintSize.y);
}
