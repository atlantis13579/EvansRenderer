// Independently authored DX12 planar reflection technique.
// A mirrored camera captures one half-space using Lengyel's oblique near plane.
// Main depth locates candidate receivers; their distance and normal must agree
// with the selected plane. The lookup projection remains unbiased so a receiver
// clipped out of capture can still sample its reflection. RGB-only composition
// runs after AO, before opaque blur and transparent forward shading.
// Mips integrate texel areas in a separate render/copy chain. This approximates
// rough reflection footprints, without a BRDF convolution or recursive capture.
cbuffer PlanarParameters : register(b0)
{
    row_major float4x4 MainClipToWorld;
    row_major float4x4 WorldToMirrorClip;
    float4 WorldReceiverPlane;
    float4 RasterSize; // main width/height and reciprocals; mip pass: destination/source extents
    float4 ReceiverSettings; // intensity, maximum mip, plane-distance tolerance, normal cosine
    uint4 Operation; // phase, shading model, valid capture, reserved
    float4 ReceiverMinimum; // world-space aperture; w enables bounds
    float4 ReceiverMaximum;
};
Texture2D<float4> OpaqueRadiance : register(t0);
Texture2D<float> MainDepth : register(t1);
Texture2D<float4> SurfaceNormal : register(t2);
Texture2D<float4> SurfaceSpecular : register(t3);
Texture2D<float4> SurfaceDiffuse : register(t4);
Texture2D<float4> SurfaceAmbient : register(t5);
Texture2D<float4> ReflectedRadiance : register(t6);
Texture2D<float4> CapturedMirror : register(t9);
Texture2D<float4> ParentLevel : register(t10);
SamplerState LinearMipClamp : register(s0);

static const uint CompositeOperation = 1u;
static const uint PhysicalMaterial = 3u;
static const float EmptyDepth = 1.0f;
static const float RayInteriorDepth = 0.5f;
static const float MinimumHomogeneousWeight = 1e-7f;
static const float MinimumDirectionSquared = 1e-12f;
static const float FresnelExponent = 5.0f;
static const float DielectricF0 = 0.04f;
static const float HalfFloatLimit = 65504.0f;
static const float HalfRoundingAllowance = 32.0f;
static const int MipFootprintSpan = 3; // An odd dimension can shrink by up to 3:1.

struct TriangleVertex { float4 RasterPosition : SV_POSITION; };
TriangleVertex VSMain(uint vertex : SV_VertexID)
{
    float2 corner = float2(vertex & 2u, (vertex << 1u) & 2u);
    TriangleVertex result;
    result.RasterPosition = float4(corner * 2 - 1, 0, 1);
    return result;
}
bool ReconstructWorld(float2 uv, float depth, out float3 position)
{
    float4 h = mul(MainClipToWorld, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    position = 0;
    if (!all(isfinite(h)) || abs(h.w) < MinimumHomogeneousWeight) return false;
    position = h.xyz / h.w;
    return all(isfinite(position));
}
bool NormalizeFinite(inout float3 direction)
{
    float squaredLength = dot(direction, direction);
    if (!all(isfinite(direction)) || squaredLength <= MinimumDirectionSquared) return false;
    direction /= sqrt(squaredLength);
    return true;
}
float4 PSMain(TriangleVertex input) : SV_TARGET
{
    int3 texel = int3(int2(input.RasterPosition.xy), 0);
    if (Operation.x == CompositeOperation) return float4(ReflectedRadiance.Load(texel).rgb, 0);
    float depth = MainDepth.Load(texel);
    if (Operation.z == 0u || ReceiverSettings.x <= 0 || depth >= EmptyDepth) return 0;
    float4 normalRoughness = SurfaceNormal.Load(texel);
    float3 normal = normalRoughness.xyz;
    if (normalRoughness.w < 0 || !NormalizeFinite(normal) || abs(dot(normal, WorldReceiverPlane.xyz)) < ReceiverSettings.w) return 0;
    float2 mainUV = input.RasterPosition.xy * RasterSize.zw;
    float3 receiver;
    if (!ReconstructWorld(mainUV, depth, receiver) || abs(dot(WorldReceiverPlane, float4(receiver, 1))) > ReceiverSettings.z) return 0;
    if (ReceiverMinimum.w != 0 && (any(receiver < ReceiverMinimum.xyz) || any(receiver > ReceiverMaximum.xyz))) return 0;
    float4 mirrorPosition = mul(WorldToMirrorClip, float4(receiver, 1));
    if (!all(isfinite(mirrorPosition)) || mirrorPosition.w <= MinimumHomogeneousWeight) return 0;
    float2 mirrorUV = mirrorPosition.xy / mirrorPosition.w * float2(0.5f, -0.5f) + 0.5f;
    if (any(mirrorUV < 0) || any(mirrorUV > 1)) return 0;
    float3 rayNear, rayInterior;
    if (!ReconstructWorld(mainUV, 0, rayNear) || !ReconstructWorld(mainUV, RayInteriorDepth, rayInterior)) return 0;
    float3 toViewer = rayNear - rayInterior;
    if (!NormalizeFinite(toViewer)) return 0;
    if (ReceiverMinimum.w != 0.0)
    {
        // Project the change in reflection direction from the flat mirror to the
        // local wave facet. Limit the footprint where the capture runs out.
        float3 bend = reflect(-toViewer, normal) - reflect(-toViewer, WorldReceiverPlane.xyz);
        float4 delta = mul(WorldToMirrorClip, float4(bend, 0));
        float2 offset = (delta.xy - mirrorPosition.xy / mirrorPosition.w * delta.w) * float2(0.14, -0.14);
        float2 margin = min(mirrorUV, 1.0 - mirrorUV);
        mirrorUV += clamp(offset, -margin, margin);
    }

    float3 f0 = saturate(SurfaceSpecular.Load(texel).rgb);
    if (Operation.y == PhysicalMaterial)
        f0 = lerp(f0, saturate(SurfaceDiffuse.Load(texel).rgb), saturate(SurfaceAmbient.Load(texel).a));
    float grazing = 1 - saturate(abs(dot(normal, toViewer)));
    float3 fresnel = f0 + (1 - f0) * pow(grazing, FresnelExponent);
    float mip = saturate(normalRoughness.w) * ReceiverSettings.y;
    mip = max(mip, CapturedMirror.CalculateLevelOfDetail(LinearMipClamp, mirrorUV));
    float3 reflected = max(CapturedMirror.SampleLevel(LinearMipClamp, mirrorUV, mip).rgb, 0) * fresnel * ReceiverSettings.x;
    float3 headroom = max(HalfFloatLimit - HalfRoundingAllowance - OpaqueRadiance.Load(texel).rgb, 0);
    return float4(min(reflected, headroom), 0);
}
float4 PSMip(TriangleVertex input) : SV_TARGET
{
    // Integrating each destination footprint keeps small lights and odd edge
    // rows through the 1x1 tail; a centered bilinear tap can miss them entirely.
    uint sourceWidth, sourceHeight;
    ParentLevel.GetDimensions(sourceWidth, sourceHeight);
    float2 sourceExtent = float2(sourceWidth, sourceHeight);
    float2 scale = sourceExtent / RasterSize.xy;
    float2 lower = floor(input.RasterPosition.xy) * scale;
    float2 upper = lower + scale;
    int2 first = int2(floor(lower));
    float4 sum = 0;
    [unroll] for (int row = 0; row < MipFootprintSpan; ++row)
    [unroll] for (int column = 0; column < MipFootprintSpan; ++column)
    {
        int2 pixel = first + int2(column, row);
        float2 overlap = max(0, min(upper, float2(pixel) + 1) - max(lower, float2(pixel)));
        if (all(pixel < int2(sourceExtent)) && all(overlap > 0))
            sum += ParentLevel.Load(int3(pixel, 0)) * overlap.x * overlap.y;
    }
    return sum / (scale.x * scale.y);
}
