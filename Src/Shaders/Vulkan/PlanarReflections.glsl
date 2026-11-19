#version 450
// Independently authored Vulkan planar reflection technique.
// Lengyel, "Oblique View Frustum Depth Projection and Clipping" (2005):
// https://terathon.com/lengyel/Lengyel-Oblique.pdf
// Capture clips geometry against the mirror, while lookup keeps the unbiased
// mirrored projection. Material-qualified receivers add Fresnel-weighted HDR
// radiance before opaque blur and transparency. Mips integrate image footprints;
// they approximate roughness without implementing a GGX convolution.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 vertex = vec2(gl_VertexIndex & 2, (gl_VertexIndex << 1) & 2);
    gl_Position = vec4(vertex * 2.0 - 1.0, 0, 1);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform PlanarParameters
{
    mat4 CameraFromClip;
    mat4 MirrorFromWorld;
    vec4 Plane;
    vec4 Extent; // main width/height and reciprocals; reduction destination extent
    vec4 Settings; // intensity, highest mip, receiver distance, normal alignment
    uvec4 Control; // phase, shading model, active capture, reserved
    vec4 ReceiverMinimum; // world-space aperture; w enables bounds
    vec4 ReceiverMaximum;
} Reflection;
layout(set = 0, binding = 8) uniform sampler PointSampler;
layout(location = 0) out vec4 Result;

#ifdef EVANS_PLANAR_MIP
layout(set = 0, binding = 14) uniform texture2D PreviousMip;
const int MaximumSourceSpan = 3;
void main()
{
    // Each source view exposes exactly one mip. An odd reduction can cover
    // three source texels per axis; overlap weights preserve the full extent.
    ivec2 sourceSize = textureSize(sampler2D(PreviousMip, PointSampler), 0);
    vec2 footprint = vec2(sourceSize) / Reflection.Extent.xy;
    vec2 firstEdge = floor(gl_FragCoord.xy) * footprint;
    vec2 lastEdge = firstEdge + footprint;
    ivec2 firstTexel = ivec2(floor(firstEdge));
    vec4 integral = vec4(0);
    for (int row = 0; row < MaximumSourceSpan; ++row)
    for (int column = 0; column < MaximumSourceSpan; ++column)
    {
        ivec2 texel = firstTexel + ivec2(column, row);
        vec2 coverage = max(vec2(0), min(lastEdge, vec2(texel) + 1.0) - max(firstEdge, vec2(texel)));
        if (all(lessThan(texel, sourceSize)) && all(greaterThan(coverage, vec2(0))))
            integral += texelFetch(sampler2D(PreviousMip, PointSampler), texel, 0) * coverage.x * coverage.y;
    }
    Result = integral / (footprint.x * footprint.y);
}
#else
layout(set = 0, binding = 1) uniform texture2D OpaqueLight;
layout(set = 0, binding = 2) uniform texture2D OpaqueDepth;
layout(set = 0, binding = 3) uniform texture2D NormalAndRoughness;
layout(set = 0, binding = 4) uniform texture2D SpecularAndGloss;
layout(set = 0, binding = 5) uniform texture2D DiffuseAndOpacity;
layout(set = 0, binding = 6) uniform texture2D AmbientAndMetallic;
layout(set = 0, binding = 7) uniform texture2D ReflectionLight;
layout(set = 0, binding = 9) uniform sampler MirrorSampler; // Clamp and trilinear.
layout(set = 0, binding = 13) uniform texture2D CapturedLight;

const uint ComposePhase = 1u;
const uint PhysicalShadingModel = 3u;
const float BackgroundDepth = 1.0;
const float RayInteriorDepth = 0.5;
const float HomogeneousTolerance = 1e-7;
const float SquaredDirectionTolerance = 1e-12;
const float DielectricFresnel = 0.04;
const float SchlickExponent = 5.0;
const float HalfRadianceMaximum = 65504.0;
const float HalfRoundingReserve = 32.0;

bool Finite(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
bool UnitVector(inout vec3 direction)
{
    float magnitudeSquared = dot(direction, direction);
    if (!Finite(vec4(direction, magnitudeSquared)) || magnitudeSquared <= SquaredDirectionTolerance) return false;
    direction *= inversesqrt(magnitudeSquared);
    return true;
}
bool WorldPosition(vec2 ndc, float depth, out vec3 world)
{
    vec4 h = Reflection.CameraFromClip * vec4(ndc, depth, 1);
    world = vec3(0);
    if (!Finite(h) || abs(h.w) < HomogeneousTolerance) return false;
    world = h.xyz / h.w;
    return Finite(vec4(world, 1));
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Reflection.Control.x == ComposePhase)
    {
        Result = vec4(texelFetch(sampler2D(ReflectionLight, PointSampler), pixel, 0).rgb, 0);
        return;
    }
    Result = vec4(0);
    float depth = texelFetch(sampler2D(OpaqueDepth, PointSampler), pixel, 0).r;
    if (Reflection.Control.z == 0u || Reflection.Settings.x <= 0.0 || depth >= BackgroundDepth) return;
    vec4 materialNormal = texelFetch(sampler2D(NormalAndRoughness, PointSampler), pixel, 0);
    vec3 normal = materialNormal.xyz;
    if (materialNormal.a < 0.0 || !UnitVector(normal) || abs(dot(normal, Reflection.Plane.xyz)) < Reflection.Settings.w) return;
    // Undo surface vertex shaders' Vulkan Y inversion. CPU clip Z remains [0,1].
    vec2 ndc = gl_FragCoord.xy * Reflection.Extent.zw * vec2(2, -2) + vec2(-1, 1);
    vec3 receiver;
    if (!WorldPosition(ndc, depth, receiver) || abs(dot(Reflection.Plane, vec4(receiver, 1))) > Reflection.Settings.z) return;
    if (Reflection.ReceiverMinimum.w != 0.0 && (any(lessThan(receiver, Reflection.ReceiverMinimum.xyz)) ||
        any(greaterThan(receiver, Reflection.ReceiverMaximum.xyz)))) return;
    vec4 mirrorClip = Reflection.MirrorFromWorld * vec4(receiver, 1);
    if (!Finite(mirrorClip) || mirrorClip.w <= HomogeneousTolerance) return;
    vec2 uv = mirrorClip.xy / mirrorClip.w * vec2(0.5, -0.5) + 0.5;
    if (any(lessThan(uv, vec2(0))) || any(greaterThan(uv, vec2(1)))) return;
    vec3 nearWorld, interiorWorld;
    if (!WorldPosition(ndc, 0.0, nearWorld) || !WorldPosition(ndc, RayInteriorDepth, interiorWorld)) return;
    vec3 towardCamera = nearWorld - interiorWorld;
    if (!UnitVector(towardCamera)) return;
    if (Reflection.ReceiverMinimum.w != 0.0)
    {
        // Project the change in reflection direction from the flat mirror to the
        // local wave facet. Limit the footprint where the capture runs out.
        vec3 bend = reflect(-towardCamera, normal) - reflect(-towardCamera, Reflection.Plane.xyz);
        vec4 delta = Reflection.MirrorFromWorld * vec4(bend, 0);
        vec2 offset = (delta.xy - mirrorClip.xy / mirrorClip.w * delta.w) * vec2(0.14, -0.14);
        vec2 margin = min(uv, 1.0 - uv);
        uv += clamp(offset, -margin, margin);
    }

    vec3 f0 = clamp(texelFetch(sampler2D(SpecularAndGloss, PointSampler), pixel, 0).rgb, 0.0, 1.0);
    if (Reflection.Control.y == PhysicalShadingModel)
    {
        vec3 albedo = clamp(texelFetch(sampler2D(DiffuseAndOpacity, PointSampler), pixel, 0).rgb, 0.0, 1.0);
        float metallic = clamp(texelFetch(sampler2D(AmbientAndMetallic, PointSampler), pixel, 0).a, 0.0, 1.0);
        f0 = mix(f0, albedo, metallic);
    }
    float grazing = 1.0 - clamp(abs(dot(normal, towardCamera)), 0.0, 1.0);
    vec3 fresnel = f0 + (1.0 - f0) * pow(grazing, SchlickExponent);
    float mip = clamp(materialNormal.a, 0.0, 1.0) * Reflection.Settings.y;
    mip = max(mip, textureQueryLod(sampler2D(CapturedLight, MirrorSampler), uv).x);
    vec3 reflected = max(textureLod(sampler2D(CapturedLight, MirrorSampler), uv, mip).rgb, vec3(0)) * fresnel * Reflection.Settings.x;
    vec3 remaining = max(vec3(HalfRadianceMaximum - HalfRoundingReserve) - texelFetch(sampler2D(OpaqueLight, PointSampler), pixel, 0).rgb, vec3(0));
    Result = vec4(min(reflected, remaining), 0);
}
#endif
#endif
