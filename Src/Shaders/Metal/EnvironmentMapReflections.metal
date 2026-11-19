#include <metal_stdlib>
using namespace metal;

// Independent Metal latitude-longitude conversion and specular probe lookup.
// PBRT, 3rd ed., Infinite Area Lights describes this spherical representation:
// https://www.pbr-book.org/3ed-2018/Light_Sources/Infinite_Area_Lights
// The base image comes from six scene views. Native image mips approximate
// roughness without solid-angle weighting or GGX convolution. This is not
// diffuse GI, and the camera-centered probe retains local parallax error.
struct EnvironmentSettings
{
    float4 WorldFromClipRows[4];
    float4 Viewport; // extent, reciprocal extent
    float4 Reflection; // intensity, last image mip, reserved
    uint4 PhaseAndModel;
    float4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
};
struct EnvironmentVertex { float4 Position [[position]]; };

constant float Pi = 3.14159265358979323846;
constant float FullTurn = 2 * Pi;
constant float LongitudeCenter = 0.5;
constant uint CompositePhase = 1;
constant uint PbrModel = 3;
constant float BackgroundDepth = 1.0;
constant float InteriorRayDepth = 0.5;
constant float HomogeneousThreshold = 1e-7;
constant float DirectionSquaredThreshold = 1e-12;
constant float DielectricReflectance = 0.04;
constant float SchlickExponent = 5.0;
constant float HalfRadianceLimit = 65504.0;
constant float HalfRoundingGuard = 32.0;

vertex EnvironmentVertex EnvironmentVertexMain(uint index [[vertex_id]])
{
    const float OversizedExtent = 3.0;
    float2 corner = index == 0 ? float2(-1, -1) :
        index == 1 ? float2(OversizedExtent, -1) : float2(-1, OversizedExtent);
    return {float4(corner, 0, 1)};
}

fragment float4 EnvironmentConvertMain(EnvironmentVertex input [[stage_in]],
    constant EnvironmentSettings& settings [[buffer(0)]],
    texturecube<float, access::sample> capturedFaces [[texture(0)]])
{
    // Metal texture V=0 is north. The native cube differs from world axes only
    // in Z here; the 2D reflection lookup keeps the original world direction.
    float2 uv = input.Position.xy * settings.Viewport.zw;
    float longitude = (uv.x - LongitudeCenter) * FullTurn;
    float colatitude = uv.y * Pi;
    float radial = sin(colatitude);
    float3 cubeRay(radial * cos(longitude), cos(colatitude), -radial * sin(longitude));
    constexpr sampler CaptureFilter(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
    return float4(max(capturedFaces.sample(CaptureFilter, cubeRay, level(0.0)).rgb, float3(0)), 1);
}

bool EnvironmentUnitRay(thread float3& ray)
{
    float squared = dot(ray, ray);
    if (!all(isfinite(ray)) || squared <= DirectionSquaredThreshold) return false;
    ray *= rsqrt(squared);
    return true;
}
bool EnvironmentWorldPoint(float2 pixel, float depth, constant EnvironmentSettings& settings, thread float3& world)
{
    // Explicit rows match the CPU contract. Reverse top-left raster Y while
    // keeping canonical [0,1] clip depth, including orthographic projections.
    float2 uv = pixel * settings.Viewport.zw;
    float4 clip(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
    float4 h(dot(settings.WorldFromClipRows[0], clip), dot(settings.WorldFromClipRows[1], clip),
        dot(settings.WorldFromClipRows[2], clip), dot(settings.WorldFromClipRows[3], clip));
    world = float3(0);
    if (abs(h.w) < HomogeneousThreshold) return false;
    world = h.xyz / h.w;
    return all(isfinite(world));
}

fragment float4 EnvironmentFragmentMain(EnvironmentVertex input [[stage_in]],
    constant EnvironmentSettings& settings [[buffer(0)]],
    texture2d<float, access::read> scene [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::read> normalRoughness [[texture(2)]],
    texture2d<float, access::read> specularShininess [[texture(3)]],
    texture2d<float, access::read> diffuseOpacity [[texture(4)]],
    texture2d<float, access::read> ambientMetallic [[texture(5)]],
    texture2d<float, access::read> reflectedRadiance [[texture(6)]],
    texture2d<float, access::sample> environment [[texture(7)]])
{
    uint2 pixel(input.Position.xy);
    if (settings.PhaseAndModel.x == CompositePhase) return float4(reflectedRadiance.read(pixel).rgb, 0);
    float4 surface = normalRoughness.read(pixel);
    float3 normal = surface.xyz;
    if (settings.Reflection.x <= 0 || surface.a < 0 || depth.read(pixel) >= BackgroundDepth || !EnvironmentUnitRay(normal)) return float4(0);
    float3 nearPoint, interiorPoint;
    if (settings.PhaseAndModel.w != 0)
    {
        float3 receiver;
        bool planeMatch = EnvironmentWorldPoint(input.Position.xy, depth.read(pixel), settings, receiver) &&
            abs(dot(settings.ReceiverPlane.xyz, receiver) + settings.ReceiverPlane.w) <= settings.ReceiverMinimum.w &&
            abs(dot(normal, settings.ReceiverPlane.xyz)) >= settings.ReceiverMaximum.w;
        constexpr uint BoundedReceiver = 2;
        bool inside = settings.PhaseAndModel.w != BoundedReceiver ||
            (all(receiver >= settings.ReceiverMinimum.xyz) && all(receiver <= settings.ReceiverMaximum.xyz));
        if (planeMatch && inside) return float4(0);
    }
    if (!EnvironmentWorldPoint(input.Position.xy, 0, settings, nearPoint) ||
        !EnvironmentWorldPoint(input.Position.xy, InteriorRayDepth, settings, interiorPoint)) return float4(0);
    // Use two finite positions so an infinite far plane never yields w=0.
    float3 eyeRay = interiorPoint - nearPoint;
    if (!EnvironmentUnitRay(eyeRay)) return float4(0);
    float3 direction = reflect(eyeRay, normal);
    float longitude = dot(direction.xz, direction.xz) > DirectionSquaredThreshold ? atan2(direction.z, direction.x) : 0.0;
    float2 uv(longitude / FullTurn + LongitudeCenter, acos(clamp(direction.y, -1.0f, 1.0f)) / Pi);
    constexpr sampler EnvironmentFilter(coord::normalized, s_address::repeat, t_address::clamp_to_edge,
        filter::linear, mip_filter::linear);
    float mip = saturate(surface.a) * settings.Reflection.y;
    float3 radiance = max(environment.sample(EnvironmentFilter, uv, level(mip)).rgb, float3(0));

    float3 f0 = saturate(specularShininess.read(pixel).rgb);
    if (settings.PhaseAndModel.y == PbrModel)
        f0 = mix(f0, saturate(diffuseOpacity.read(pixel).rgb), saturate(ambientMetallic.read(pixel).a));
    float cosine = saturate(dot(normal, -eyeRay));
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    float4 brdfFit=clamp(surface.a,0.0,1.0)*float4(-1.0,-0.0275,-0.572,0.022)+float4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*cosine))*brdfFit.x+brdfFit.y;
    float2 integrated=float2(-1.04,1.04)*grazing+brdfFit.zw;
    float3 fresnel = settings.PhaseAndModel.y == PbrModel?clamp(f0*integrated.x+integrated.y,0.0,1.0):f0 + (1 - f0) * pow(1 - cosine, SchlickExponent);
    float3 remaining = max(float3(HalfRadianceLimit - HalfRoundingGuard) - scene.read(pixel).rgb, float3(0));
    return float4(min(settings.Reflection.x * fresnel * radiance, remaining), 0);
}
