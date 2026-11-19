#version 450
// Independent Vulkan latitude-longitude environment technique. PBRT, 3rd ed.,
// Infinite Area Lights describes the Y-up spherical parameterization:
// https://www.pbr-book.org/3ed-2018/Light_Sources/Infinite_Area_Lights
// Convert completed scene faces, reduce the 2D image, then evaluate a local
// specular probe. Box mips approximate roughness; they are not GGX convolution
// or solid-angle-weighted integrals. Nearby geometry retains probe parallax.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const vec2 Triangle[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(Triangle[gl_VertexIndex], 0, 1);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform EnvironmentParameters
{
    mat4 InverseCamera;
    vec4 Viewport; // destination size, reciprocal size
    vec4 Reflection; // intensity, highest image mip, reserved
    uvec4 PhaseAndModel;
    vec4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
} Parameters;
layout(location = 0) out vec4 Color;

#if defined(EVANS_ENVIRONMENT_CONVERSION)
layout(set = 0, binding = 9) uniform sampler CubeFilter;
layout(set = 0, binding = 10) uniform textureCube SceneFaces;
void main()
{
    const float Pi = 3.14159265358979323846;
    const float FullTurn = 2.0 * Pi;
    const float LongitudeCenter = 0.5;
    vec2 uv = gl_FragCoord.xy * Parameters.Viewport.zw;
    float longitude = (uv.x - LongitudeCenter) * FullTurn;
    float colatitude = uv.y * Pi;
    float radial = sin(colatitude);
    // Vulkan's positive-height viewport puts north at image V=0. Convert world
    // Z to native cube Z only here; later 2D lookup uses world directions.
    vec3 cubeRay = vec3(radial * cos(longitude), cos(colatitude), -radial * sin(longitude));
    Color = vec4(max(textureLod(samplerCube(SceneFaces, CubeFilter), cubeRay, 0.0).rgb, vec3(0)), 1);
}
#elif defined(EVANS_ENVIRONMENT_MIP)
layout(set = 0, binding = 8) uniform sampler PointFilter;
layout(set = 0, binding = 14) uniform texture2D ParentLevel;
void main()
{
    const int ReductionSpan = 2;
    const float ReductionWeight = 1.0 / float(ReductionSpan * ReductionSpan);
    ivec2 origin = ivec2(gl_FragCoord.xy) * ReductionSpan;
    ivec2 lastTexel = textureSize(sampler2D(ParentLevel, PointFilter), 0) - 1;
    vec4 sum = vec4(0);
    for (int y = 0; y < ReductionSpan; ++y)
        for (int x = 0; x < ReductionSpan; ++x)
            sum += texelFetch(sampler2D(ParentLevel, PointFilter), min(origin + ivec2(x, y), lastTexel), 0);
    // The 2x1 tail repeats its only row. Sampling beyond that row loses energy.
    Color = sum * ReductionWeight;
}
#else
layout(set = 0, binding = 1) uniform texture2D SceneRadiance;
layout(set = 0, binding = 2) uniform texture2D MainDepth;
layout(set = 0, binding = 3) uniform texture2D NormalRoughness;
layout(set = 0, binding = 4) uniform texture2D SpecularShininess;
layout(set = 0, binding = 5) uniform texture2D DiffuseOpacity;
layout(set = 0, binding = 6) uniform texture2D AmbientMetallic;
layout(set = 0, binding = 7) uniform texture2D ReflectedRadiance;
layout(set = 0, binding = 8) uniform sampler PointFilter;
layout(set = 0, binding = 12) uniform sampler LongitudeFilter; // Wrap U, clamp V/W.
layout(set = 0, binding = 13) uniform texture2D SphericalRadiance;

const uint CompositePhase = 1u;
const uint PbrModel = 3u;
const float Pi = 3.14159265358979323846;
const float FullTurn = 2.0 * Pi;
const float LongitudeCenter = 0.5;
const float ClearedDepth = 1.0;
const float FiniteInteriorDepth = 0.5;
const float HomogeneousEpsilon = 1e-7;
const float DirectionSquaredEpsilon = 1e-12;
const float DielectricF0 = 0.04;
const float FresnelExponent = 5.0;
const float MaximumHalfRadiance = 65504.0;
const float HalfRoundingGuard = 32.0;

bool NormalizeRay(inout vec3 ray)
{
    float squared = dot(ray, ray);
    if (any(isnan(ray)) || any(isinf(ray)) || squared <= DirectionSquaredEpsilon) return false;
    ray *= inversesqrt(squared);
    return true;
}
bool Unproject(float depth, out vec3 world)
{
    // Undo Vulkan raster Y adaptation. CPU projection depth remains [0,1].
    vec2 ndc = gl_FragCoord.xy * Parameters.Viewport.zw * vec2(2, -2) + vec2(-1, 1);
    vec4 h = Parameters.InverseCamera * vec4(ndc, depth, 1);
    world = vec3(0);
    if (abs(h.w) < HomogeneousEpsilon) return false;
    world = h.xyz / h.w;
    return !any(isnan(world)) && !any(isinf(world));
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Parameters.PhaseAndModel.x == CompositePhase)
    {
        Color = vec4(texelFetch(sampler2D(ReflectedRadiance, PointFilter), pixel, 0).rgb, 0);
        return;
    }
    Color = vec4(0);
    vec4 material = texelFetch(sampler2D(NormalRoughness, PointFilter), pixel, 0);
    vec3 normal = material.xyz;
    if (Parameters.Reflection.x <= 0 || material.a < 0 || !NormalizeRay(normal) ||
        texelFetch(sampler2D(MainDepth, PointFilter), pixel, 0).r >= ClearedDepth) return;
    vec3 nearPoint, interiorPoint;
    if (Parameters.PhaseAndModel.w != 0u)
    {
        vec3 receiver;
        float depth = texelFetch(sampler2D(MainDepth, PointFilter), pixel, 0).r;
        bool onPlane = Unproject(depth, receiver) &&
            abs(dot(Parameters.ReceiverPlane.xyz, receiver) + Parameters.ReceiverPlane.w) <= Parameters.ReceiverMinimum.w &&
            abs(dot(normal, Parameters.ReceiverPlane.xyz)) >= Parameters.ReceiverMaximum.w;
        const uint FiniteReceiver = 2u;
        bool inside = Parameters.PhaseAndModel.w != FiniteReceiver ||
            (all(greaterThanEqual(receiver, Parameters.ReceiverMinimum.xyz)) &&
             all(lessThanEqual(receiver, Parameters.ReceiverMaximum.xyz)));
        if (onPlane && inside) return;
    }
    if (!Unproject(0.0, nearPoint) || !Unproject(FiniteInteriorDepth, interiorPoint)) return;
    vec3 viewRay = interiorPoint - nearPoint;
    if (!NormalizeRay(viewRay)) return;
    vec3 direction = reflect(viewRay, normal);
    float longitude = dot(direction.xz, direction.xz) > DirectionSquaredEpsilon ? atan(direction.z, direction.x) : 0.0;
    vec2 uv = vec2(longitude / FullTurn + LongitudeCenter, acos(clamp(direction.y, -1.0, 1.0)) / Pi);
    float lod = clamp(material.a, 0.0, 1.0) * Parameters.Reflection.y;
    vec3 radiance = max(textureLod(sampler2D(SphericalRadiance, LongitudeFilter), uv, lod).rgb, vec3(0));

    vec3 f0 = clamp(texelFetch(sampler2D(SpecularShininess, PointFilter), pixel, 0).rgb, 0.0, 1.0);
    if (Parameters.PhaseAndModel.y == PbrModel)
    {
        vec3 base = clamp(texelFetch(sampler2D(DiffuseOpacity, PointFilter), pixel, 0).rgb, 0.0, 1.0);
        float metal = clamp(texelFetch(sampler2D(AmbientMetallic, PointFilter), pixel, 0).a, 0.0, 1.0);
        f0 = mix(f0, base, metal);
    }
    float cosine = clamp(dot(normal, -viewRay), 0.0, 1.0);
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    vec4 brdfFit=clamp(material.a,0.0,1.0)*vec4(-1.0,-0.0275,-0.572,0.022)+vec4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*cosine))*brdfFit.x+brdfFit.y;
    vec2 integrated=vec2(-1.04,1.04)*grazing+brdfFit.zw;
    vec3 fresnel = Parameters.PhaseAndModel.y == PbrModel?clamp(f0*integrated.x+integrated.y,0.0,1.0):f0 + (1.0 - f0) * pow(1.0 - cosine, FresnelExponent);
    vec3 available = max(vec3(MaximumHalfRadiance - HalfRoundingGuard) - texelFetch(sampler2D(SceneRadiance, PointFilter), pixel, 0).rgb, vec3(0));
    Color = vec4(min(Parameters.Reflection.x * radiance * fresnel, available), 0);
}
#endif
#endif
