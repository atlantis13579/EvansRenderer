#include <metal_stdlib>
using namespace metal;

// Independent Metal planar receiver lookup, HDR composition and mip integration.
// Lengyel, "Oblique View Frustum Depth Projection and Clipping" (2005):
// https://terathon.com/lengyel/Lengyel-Oblique.pdf
// Geometry uses the clipped mirrored camera; receiver lookup uses its original
// projection because the clipping bias deliberately excludes the receiver itself.
// A single plane cannot reflect curved receivers or recursively reflect mirrors.
struct PlanarSettings
{
    float4 WorldFromClipRows[4];
    float4 MirrorFromWorldRows[4];
    float4 ReceiverPlane;
    float4 Viewport; // main extent and reciprocal; mip reduction uses destination xy
    float4 Sampling; // intensity, last mip, plane tolerance, minimum normal cosine
    uint4 Options; // phase (or source mip during reduction), surface model, active capture
    float4 ReceiverMinimum; // world-space aperture; w enables bounds
    float4 ReceiverMaximum;
};
struct PlanarVertex { float4 Position [[position]]; };

constant uint CompositePhase = 1;
constant uint PhysicalShading = 3;
constant float BackgroundDepth = 1.0;
constant float InteriorRayDepth = 0.5;
constant float MinimumHomogeneousW = 1e-7;
constant float MinimumSquaredLength = 1e-12;
constant float DielectricReflectance = 0.04;
constant float SchlickExponent = 5.0;
constant float MaximumHalfRadiance = 65504.0;
constant float HalfRoundingGuard = 32.0;

vertex PlanarVertex PlanarVertexMain(uint index [[vertex_id]])
{
    const float OversizedExtent = 3.0;
    float2 corner = index == 0 ? float2(-1, -1) :
        index == 1 ? float2(OversizedExtent, -1) : float2(-1, OversizedExtent);
    return {float4(corner, 0, 1)};
}

bool PlanarUnitDirection(thread float3& direction)
{
    float squared = dot(direction, direction);
    if (!all(isfinite(direction)) || squared <= MinimumSquaredLength) return false;
    direction *= rsqrt(squared);
    return true;
}

bool PlanarWorldPoint(float2 uv, float depth, constant PlanarSettings& settings, thread float3& world)
{
    // Explicit rows preserve the CPU layout. Metal pixels/UVs start at the top;
    // canonical projection Y points upward and depth already occupies [0,1].
    float4 clip(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
    float4 h(dot(settings.WorldFromClipRows[0], clip), dot(settings.WorldFromClipRows[1], clip),
        dot(settings.WorldFromClipRows[2], clip), dot(settings.WorldFromClipRows[3], clip));
    world = float3(0);
    if (!all(isfinite(h)) || abs(h.w) <= MinimumHomogeneousW) return false;
    world = h.xyz / h.w;
    return all(isfinite(world));
}

fragment float4 PlanarFragmentMain(PlanarVertex input [[stage_in]],
    constant PlanarSettings& settings [[buffer(0)]],
    texture2d<float, access::read> scene [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::read> normalRoughness [[texture(2)]],
    texture2d<float, access::read> specularShininess [[texture(3)]],
    texture2d<float, access::read> diffuseOpacity [[texture(4)]],
    texture2d<float, access::read> ambientMetallic [[texture(5)]],
    texture2d<float, access::read> reflectedRadiance [[texture(6)]],
    texture2d<float, access::sample> mirror [[texture(7)]])
{
    uint2 pixel = uint2(input.Position.xy);
    if (settings.Options.x == CompositePhase) return float4(reflectedRadiance.read(pixel).rgb, 0);
    float opaqueDepth = depth.read(pixel);
    if (settings.Options.z == 0 || settings.Sampling.x <= 0 || opaqueDepth >= BackgroundDepth) return float4(0);
    float4 materialNormal = normalRoughness.read(pixel);
    float3 normal = materialNormal.xyz;
    if (materialNormal.a < 0 || !PlanarUnitDirection(normal) ||
        abs(dot(normal, settings.ReceiverPlane.xyz)) < settings.Sampling.w) return float4(0);
    float2 uv = input.Position.xy * settings.Viewport.zw;
    float3 receiver, nearPoint, interiorPoint;
    if (!PlanarWorldPoint(uv, opaqueDepth, settings, receiver) ||
        abs(dot(settings.ReceiverPlane, float4(receiver, 1))) > settings.Sampling.z) return float4(0);
    float4 world(receiver, 1);
    if (settings.ReceiverMinimum.w != 0 && (any(receiver < settings.ReceiverMinimum.xyz) ||
        any(receiver > settings.ReceiverMaximum.xyz))) return float4(0);
    float4 projected(dot(settings.MirrorFromWorldRows[0], world), dot(settings.MirrorFromWorldRows[1], world),
        dot(settings.MirrorFromWorldRows[2], world), dot(settings.MirrorFromWorldRows[3], world));
    if (!all(isfinite(projected)) || projected.w <= MinimumHomogeneousW) return float4(0);
    float2 mirrorUV(projected.x / projected.w + 1, 1 - projected.y / projected.w);
    mirrorUV /= 2;
    if (any(mirrorUV < 0) || any(mirrorUV > 1)) return float4(0);
    if (!PlanarWorldPoint(uv, 0, settings, nearPoint) ||
        !PlanarWorldPoint(uv, InteriorRayDepth, settings, interiorPoint)) return float4(0);
    float3 incident = interiorPoint - nearPoint;
    if (!PlanarUnitDirection(incident)) return float4(0);
    if (settings.ReceiverMinimum.w != 0.0)
    {
        // Project the change in reflection direction from the flat mirror to the
        // local wave facet. Limit the footprint where the capture runs out.
        float3 bend = reflect(incident, normal) - reflect(incident, settings.ReceiverPlane.xyz);
        float4 delta = float4(dot(settings.MirrorFromWorldRows[0].xyz, bend), dot(settings.MirrorFromWorldRows[1].xyz, bend),
            0, dot(settings.MirrorFromWorldRows[3].xyz, bend));
        float2 offset = (delta.xy - projected.xy / projected.w * delta.w) * float2(0.14, -0.14);
        float2 margin = min(mirrorUV, 1.0 - mirrorUV);
        mirrorUV += clamp(offset, -margin, margin);
    }

    // Area-filtered image mips approximate roughness, not a GGX convolution.
    constexpr sampler MirrorFilter(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
    float mip = saturate(materialNormal.a) * settings.Sampling.y;
    float2 texelUV = mirrorUV * float2(mirror.get_width(), mirror.get_height());
    float footprint = max(length(dfdx(texelUV)), length(dfdy(texelUV)));
    mip = max(mip, log2(max(footprint, 1.0f)));
    float3 radiance = max(mirror.sample(MirrorFilter, mirrorUV, level(mip)).rgb, float3(0));
    float3 baseReflectance = saturate(specularShininess.read(pixel).rgb);
    if (settings.Options.y == PhysicalShading)
        baseReflectance = mix(baseReflectance, saturate(diffuseOpacity.read(pixel).rgb),
            saturate(ambientMetallic.read(pixel).a));
    // The camera can be on either side of the receiver plane.
    float cosine = saturate(abs(dot(normal, -incident)));
    float3 fresnel = baseReflectance + (1 - baseReflectance) * pow(1 - cosine, SchlickExponent);
    float3 headroom = max(float3(MaximumHalfRadiance - HalfRoundingGuard) - scene.read(pixel).rgb, float3(0));
    return float4(min(radiance * fresnel * settings.Sampling.x, headroom), 0);
}

fragment float4 PlanarMipMain(PlanarVertex input [[stage_in]],
    constant PlanarSettings& settings [[buffer(0)]],
    texture2d<float, access::read> source [[texture(0)]])
{
    // Read the full image at an explicit source level, render to a distinct
    // scratch image, then blit into the destination mip. Odd extents need area
    // integration: a bilinear 3-to-1 reduction can discard small bright objects.
    const uint MaximumFootprintTexels = 3;
    uint mip = settings.Options.x;
    uint2 sourceExtent(source.get_width(mip), source.get_height(mip));
    float2 footprint = float2(sourceExtent) / settings.Viewport.xy;
    float2 begin = floor(input.Position.xy) * footprint;
    float2 end = begin + footprint;
    uint2 first = uint2(floor(begin));
    float4 integral(0);
    for (uint y = 0; y < MaximumFootprintTexels; ++y)
    for (uint x = 0; x < MaximumFootprintTexels; ++x)
    {
        uint2 texel = first + uint2(x, y);
        float2 overlap = max(min(end, float2(texel + 1)) - max(begin, float2(texel)), float2(0));
        float area = overlap.x * overlap.y;
        if (area > 0 && all(texel < sourceExtent)) integral += source.read(texel, mip) * area;
    }
    return integral / (footprint.x * footprint.y);
}
