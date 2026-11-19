#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp samplerCube;

// Independent ES spherical environment conversion, lookup and HDR composition.
// PBRT, 3rd ed., Infinite Area Lights describes latitude-longitude coordinates:
// https://www.pbr-book.org/3ed-2018/Light_Sources/Infinite_Area_Lights
// Six scene views supply radiance; native image mips approximate roughness.
// This local specular probe is neither diffuse GI nor GGX prefiltered lighting.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const vec2 Triangle[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(Triangle[gl_VertexID], 0, 1);
}
#else
uniform vec4 Viewport;
out vec4 OutColor;
const float Pi = 3.14159265358979323846;
const float FullTurn = 2.0 * Pi;
const float LongitudeCenter = 0.5;

#ifdef EVANS_ENVIRONMENT_CONVERSION
uniform samplerCube CapturedFaces;
void main()
{
    // Texture V=0 stores north even though native framebuffer Y grows upward.
    // Conversion and lookup use the same storage coordinates, without a V flip.
    vec2 uv = gl_FragCoord.xy * Viewport.zw;
    float longitude = (uv.x - LongitudeCenter) * FullTurn;
    float colatitude = uv.y * Pi;
    float radial = sin(colatitude);
    vec3 cubeDirection = vec3(radial * cos(longitude), cos(colatitude), -radial * sin(longitude));
    OutColor = vec4(max(textureLod(CapturedFaces, cubeDirection, 0.0).rgb, vec3(0)), 1);
}
#else
uniform mat4 InverseViewProjection;
uniform vec4 Sampling; // intensity, last image mip, reserved
uniform uvec4 Options; // phase, surface model, reserved
uniform vec4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
uniform sampler2D SceneColor;
uniform sampler2D SceneDepth;
uniform sampler2D NormalRoughness;
uniform sampler2D SpecularShininess;
uniform sampler2D DiffuseOpacity;
uniform sampler2D AmbientMetallic;
uniform sampler2D ReflectionRadiance;
uniform sampler2D EnvironmentImage; // Repeat S, clamp T, trilinear.

const uint CompositePhase = 1u;
const uint PbrModel = 3u;
const float ClearDepth = 1.0;
const float InteriorDepth = 0.5;
const float MinimumHomogeneousW = 1e-7;
const float MinimumDirectionSquared = 1e-12;
const float DielectricReflectance = 0.04;
const float SchlickExponent = 5.0;
const float HalfFloatMaximum = 65504.0;
const float HalfFloatRoundingGuard = 32.0;

bool UnitDirection(inout vec3 direction)
{
    float squared = dot(direction, direction);
    if (any(isnan(direction)) || any(isinf(direction)) || squared <= MinimumDirectionSquared) return false;
    direction *= inversesqrt(squared);
    return true;
}
bool WorldPosition(vec2 ndc, float depth, out vec3 world)
{
    // The CPU projection uses [0,1] depth; GL window depth already matches it.
    vec4 homogeneous = InverseViewProjection * vec4(ndc, depth, 1);
    world = vec3(0);
    if (abs(homogeneous.w) < MinimumHomogeneousW) return false;
    world = homogeneous.xyz / homogeneous.w;
    return !any(isnan(world)) && !any(isinf(world));
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Options.x == CompositePhase)
    {
        OutColor = vec4(texelFetch(ReflectionRadiance, pixel, 0).rgb, 0);
        return;
    }
    OutColor = vec4(0);
    vec4 surface = texelFetch(NormalRoughness, pixel, 0);
    vec3 normal = surface.xyz;
    if (Sampling.x <= 0.0 || surface.a < 0.0 || texelFetch(SceneDepth, pixel, 0).r >= ClearDepth || !UnitDirection(normal)) return;
    vec2 ndc = gl_FragCoord.xy * Viewport.zw * 2.0 - 1.0;
    if (Options.w != 0u)
    {
        vec3 receiver;
        bool onMirror = WorldPosition(ndc, texelFetch(SceneDepth, pixel, 0).r, receiver) &&
            abs(dot(ReceiverPlane, vec4(receiver, 1.0))) <= ReceiverMinimum.w &&
            abs(dot(normal, ReceiverPlane.xyz)) >= ReceiverMaximum.w;
        const uint BoundedMirror = 2u;
        bool inside = Options.w != BoundedMirror ||
            (all(greaterThanEqual(receiver, ReceiverMinimum.xyz)) && all(lessThanEqual(receiver, ReceiverMaximum.xyz)));
        if (onMirror && inside) return;
    }
    vec3 nearWorld, interiorWorld;
    if (!WorldPosition(ndc, 0.0, nearWorld) || !WorldPosition(ndc, InteriorDepth, interiorWorld)) return;
    // Two finite positions support orthographic cameras and infinite far planes.
    vec3 eyeRay = interiorWorld - nearWorld;
    if (!UnitDirection(eyeRay)) return;
    vec3 direction = reflect(eyeRay, normal);
    float longitude = dot(direction.xz, direction.xz) > MinimumDirectionSquared ? atan(direction.z, direction.x) : 0.0;
    vec2 address = vec2(longitude / FullTurn + LongitudeCenter, acos(clamp(direction.y, -1.0, 1.0)) / Pi);
    float mip = clamp(surface.a, 0.0, 1.0) * Sampling.y;
    vec3 radiance = max(textureLod(EnvironmentImage, address, mip).rgb, vec3(0));

    vec3 f0 = clamp(texelFetch(SpecularShininess, pixel, 0).rgb, 0.0, 1.0);
    if (Options.y == PbrModel)
    {
        vec3 albedo = clamp(texelFetch(DiffuseOpacity, pixel, 0).rgb, 0.0, 1.0);
        float metallic = clamp(texelFetch(AmbientMetallic, pixel, 0).a, 0.0, 1.0);
        f0 = mix(f0, albedo, metallic);
    }
    float cosine = clamp(dot(normal, -eyeRay), 0.0, 1.0);
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    vec4 brdfFit=clamp(surface.a,0.0,1.0)*vec4(-1.0,-0.0275,-0.572,0.022)+vec4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*cosine))*brdfFit.x+brdfFit.y;
    vec2 integrated=vec2(-1.04,1.04)*grazing+brdfFit.zw;
    vec3 fresnel = Options.y == PbrModel?clamp(f0*integrated.x+integrated.y,0.0,1.0):f0 + (1.0 - f0) * pow(1.0 - cosine, SchlickExponent);
    vec3 headroom = max(vec3(HalfFloatMaximum - HalfFloatRoundingGuard) - texelFetch(SceneColor, pixel, 0).rgb, vec3(0));
    OutColor = vec4(min(Sampling.x * radiance * fresnel, headroom), 0);
}
#endif
#endif
