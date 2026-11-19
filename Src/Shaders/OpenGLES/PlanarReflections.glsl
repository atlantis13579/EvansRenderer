#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

// Independent ES planar reflection lookup, composition and mip reduction.
// Lengyel, "Oblique View Frustum Depth Projection and Clipping" (2005):
// https://terathon.com/lengyel/Lengyel-Oblique.pdf
// Geometry uses the biased capture projection; receivers use the original
// mirrored projection. GL images and gl_FragCoord both have a bottom-left origin.
// Area-weighted image mips approximate roughness, without a GGX convolution.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2(gl_VertexID & 2, (gl_VertexID << 1) & 2);
    gl_Position = vec4(corner * 2.0 - 1.0, 0, 1);
}
#else
uniform vec4 Viewport;
uniform uvec4 Options; // lookup: phase/model/active; reduction: source mip
uniform sampler2D PlanarImage;
out vec4 OutColor;

#ifdef EVANS_PLANAR_MIP
const int MaximumFootprintSpan = 3; // The last odd-sized reduction can be 3:1.
void main()
{
    int sourceMip = int(Options.x);
    ivec2 sourceExtent = textureSize(PlanarImage, sourceMip);
    vec2 footprint = vec2(sourceExtent) / Viewport.xy;
    vec2 begin = floor(gl_FragCoord.xy) * footprint;
    vec2 end = begin + footprint;
    ivec2 first = ivec2(floor(begin));
    vec4 radiance = vec4(0);
    for (int y = 0; y < MaximumFootprintSpan; ++y)
    for (int x = 0; x < MaximumFootprintSpan; ++x)
    {
        ivec2 pixel = first + ivec2(x, y);
        vec2 overlap = max(vec2(0), min(end, vec2(pixel) + 1.0) - max(begin, vec2(pixel)));
        if (all(lessThan(pixel, sourceExtent)) && all(greaterThan(overlap, vec2(0))))
            radiance += texelFetch(PlanarImage, pixel, sourceMip) * overlap.x * overlap.y;
    }
    OutColor = radiance / (footprint.x * footprint.y);
}
#else
uniform mat4 InverseViewProjection;
uniform mat4 ReflectionViewProjection;
uniform vec4 ReceiverPlane;
uniform vec4 ReceiverMinimum; // world-space aperture; w enables bounds
uniform vec4 ReceiverMaximum;
uniform vec4 Sampling; // intensity, last mip, world distance tolerance, normal cosine
uniform sampler2D SceneColor;
uniform sampler2D SceneDepth;
uniform sampler2D NormalRoughness;
uniform sampler2D SpecularShininess;
uniform sampler2D DiffuseOpacity;
uniform sampler2D AmbientMetallic;
uniform sampler2D ReflectionRadiance;

const uint CompositionPhase = 1u;
const uint PbrShading = 3u;
const float FarDepth = 1.0;
const float InteriorRayDepth = 0.5;
const float MinimumHomogeneousMagnitude = 1e-7;
const float MinimumSquaredDirection = 1e-12;
const float DielectricSpecular = 0.04;
const float SchlickPower = 5.0;
const float MaximumHalfRadiance = 65504.0;
const float HalfRoundingMargin = 32.0;

bool Finite(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }
bool NormalizeDirection(inout vec3 direction)
{
    float squared = dot(direction, direction);
    if (!Finite(vec4(direction, squared)) || squared <= MinimumSquaredDirection) return false;
    direction *= inversesqrt(squared);
    return true;
}
bool Reconstruct(vec2 ndc, float depth, out vec3 world)
{
    // Surface vertex shaders map canonical [0,1] clip depth to GL [-1,1].
    // Window depth reverses that mapping, so the CPU inverse applies directly.
    vec4 h = InverseViewProjection * vec4(ndc, depth, 1);
    world = vec3(0);
    if (!Finite(h) || abs(h.w) < MinimumHomogeneousMagnitude) return false;
    world = h.xyz / h.w;
    return Finite(vec4(world, 1));
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Options.x == CompositionPhase)
    {
        OutColor = vec4(texelFetch(ReflectionRadiance, pixel, 0).rgb, 0);
        return;
    }
    OutColor = vec4(0);
    float depth = texelFetch(SceneDepth, pixel, 0).r;
    if (Options.z == 0u || Sampling.x <= 0.0 || depth >= FarDepth) return;
    vec4 surface = texelFetch(NormalRoughness, pixel, 0);
    vec3 normal = surface.xyz;
    if (surface.a < 0.0 || !NormalizeDirection(normal) || abs(dot(normal, ReceiverPlane.xyz)) < Sampling.w) return;
    vec2 ndc = gl_FragCoord.xy * Viewport.zw * 2.0 - 1.0;
    vec3 world;
    if (!Reconstruct(ndc, depth, world) || abs(dot(ReceiverPlane, vec4(world, 1))) > Sampling.z) return;
    if (ReceiverMinimum.w != 0.0 && (any(lessThan(world, ReceiverMinimum.xyz)) || any(greaterThan(world, ReceiverMaximum.xyz)))) return;
    vec4 mirrorClip = ReflectionViewProjection * vec4(world, 1);
    if (!Finite(mirrorClip) || mirrorClip.w <= MinimumHomogeneousMagnitude) return;
    vec2 address = mirrorClip.xy / mirrorClip.w * 0.5 + 0.5;
    if (any(lessThan(address, vec2(0))) || any(greaterThan(address, vec2(1)))) return;
    vec3 nearPoint, interiorPoint;
    if (!Reconstruct(ndc, 0.0, nearPoint) || !Reconstruct(ndc, InteriorRayDepth, interiorPoint)) return;
    vec3 toEye = nearPoint - interiorPoint;
    if (!NormalizeDirection(toEye)) return;
    if (ReceiverMinimum.w != 0.0)
    {
        // Project the change in reflection direction from the flat mirror to the
        // local wave facet. Limit the footprint where the capture runs out.
        vec3 bend = reflect(-toEye, normal) - reflect(-toEye, ReceiverPlane.xyz);
        vec4 delta = ReflectionViewProjection * vec4(bend, 0);
        vec2 offset = (delta.xy - mirrorClip.xy / mirrorClip.w * delta.w) * vec2(0.14, 0.14);
        vec2 margin = min(address, 1.0 - address);
        address += clamp(offset, -margin, margin);
    }

    vec3 f0 = clamp(texelFetch(SpecularShininess, pixel, 0).rgb, 0.0, 1.0);
    if (Options.y == PbrShading)
    {
        vec3 albedo = clamp(texelFetch(DiffuseOpacity, pixel, 0).rgb, 0.0, 1.0);
        float metallic = clamp(texelFetch(AmbientMetallic, pixel, 0).a, 0.0, 1.0);
        f0 = mix(f0, albedo, metallic);
    }
    // Either side of the plane can be the receiver; the capture half-space is
    // chosen on the CPU from the eye. Use the corresponding two-sided Fresnel.
    float grazing = 1.0 - clamp(abs(dot(normal, toEye)), 0.0, 1.0);
    vec3 fresnel = f0 + (1.0 - f0) * pow(grazing, SchlickPower);
    float mip = clamp(surface.a, 0.0, 1.0) * Sampling.y;
    vec2 texelUV = address * vec2(textureSize(PlanarImage, 0));
    float footprint = max(length(dFdx(texelUV)), length(dFdy(texelUV)));
    mip = max(mip, log2(max(footprint, 1.0)));
    vec3 reflected = max(textureLod(PlanarImage, address, mip).rgb, vec3(0)) * Sampling.x * fresnel;
    vec3 headroom = max(vec3(MaximumHalfRadiance - HalfRoundingMargin) - texelFetch(SceneColor, pixel, 0).rgb, vec3(0));
    OutColor = vec4(min(reflected, headroom), 0);
}
#endif
#endif
