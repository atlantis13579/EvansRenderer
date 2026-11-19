#version 300 es
#ifdef EVANS_VERTEX_STAGE

// Native line primitives have no face classification. Reject all endpoints of a
// back-facing original triangle, retaining the native line coverage and clipping.
invariant gl_Position;
layout(location=15) in highp vec4 WireFacePlane;
uniform highp int WireFaceCull;
uniform highp vec4 WireFaceCamera;
void CullWireFace()
{
    if (WireFaceCull!=0 && dot(WireFacePlane,WireFaceCamera)<=0.0)
        gl_Position=vec4(0.0,0.0,2.0,1.0);
}
#endif
// Anime environment shading, with a low-detail palette and a cool/warm ramp.
// Artistic defaults inspired by Cai's GDC 2021 art-direction presentation,
// "Genshin Impact: Crafting an Anime Style Open World"; no authored ramp asset.
precision highp float;
precision highp int;
const float ToonShadingModel = 4.0;
const float ToonAlbedoExponent = 0.8;
const float ToonPaletteSteps = 16.0;
const float ToonColorSaturation = 1.12;
const float ToonNormalStrength = 0.12;
const float ToonLightScale = 0.45;
const float ToonBandSoftness = 0.4;
const float ToonShadowFloor = 0.18;
const float ToonHighlightThreshold = 0.8;
const float ToonHighlightStrength = 0.18;
const vec3 ToonLuma = vec3(0.2126, 0.7152, 0.0722);
const vec3 ToonCoolShadow = vec3(0.56, 0.68, 1.0);
const vec3 ToonWarmLight = vec3(1.0, 0.94, 0.82);
const vec3 ToonSkyFill = vec3(0.24, 0.29, 0.4);
vec3 ToonAlbedo(vec3 value, float model)
{
    if (model != ToonShadingModel) return value;
    vec3 paint = pow(clamp(value, 0.0, 1.0), vec3(ToonAlbedoExponent));
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return clamp(mix(vec3(dot(paint, ToonLuma)), paint, ToonColorSaturation), 0.0, 1.0);
}
vec3 ToonAmbient(vec3 albedo, vec3 ambient, float model)
{ return model == ToonShadingModel ? max(ambient, albedo * ToonSkyFill) : ambient; }
vec3 ToonResponse(float cosine, float bands)
{
    float angularBand = clamp(cosine, 0.0, 1.0) * bands;
    float transition = smoothstep(1.0 - ToonBandSoftness, 1.0, fract(angularBand));
    float level = min(floor(angularBand) + transition, bands - 1.0) / (bands - 1.0);
    return mix(ToonCoolShadow, ToonWarmLight, level) * mix(ToonShadowFloor, 1.0, level) * ToonLightScale;
}
float ToonHighlight(float value)
{ return step(ToonHighlightThreshold, value) * ToonHighlightStrength; }

// Average RGB over a screen footprint; center alpha keeps cutout coverage.
// Explicit gradients allow this helper to coexist with vertex entry points.
const float ToonTextureRadius = 1.25;
const int ToonTextureKernelRadius = 1;
const int ToonTextureKernelWidth = ToonTextureKernelRadius * 2 + 1;
const float ToonTextureSampleWeight = 1.0 / float(ToonTextureKernelWidth * ToonTextureKernelWidth);
vec3 ToonTextureColor(sampler2D image, vec2 uv, vec2 uvDx, vec2 uvDy)
{
    vec3 color = vec3(0.0);
    for (int row = -ToonTextureKernelRadius; row <= ToonTextureKernelRadius; ++row)
        for (int column = -ToonTextureKernelRadius; column <= ToonTextureKernelRadius; ++column)
            color += textureGrad(image, uv + ToonTextureRadius * (float(column) * uvDx + float(row) * uvDy), uvDx, uvDy).rgb;
    return color * ToonTextureSampleWeight;
}

// Karis' smooth finite-radius window; Filament, Punctual lights.
// Explicit highp also permits this helper before either stage's declarations.
highp float LightRangeWindow(highp vec3 towardEmitter, highp float range)
{
    if (range == 0.0) return 1.0;
    highp vec3 absoluteOffset = abs(towardEmitter);
    if (max(max(absoluteOffset.x, absoluteOffset.y), absoluteOffset.z) >= range) return 0.0;
    highp vec3 relativeOffset = towardEmitter / range;
    highp float squaredDistance = dot(relativeOffset, relativeOffset);
    highp float fade = max(1.0 - squaredDistance * squaredDistance, 0.0);
    return fade * fade;
}

// Standalone ES deferred shading. Compile geometry and lighting separately with
// EVANS_DEFERRED_LIGHTING, and each program's vertex stage with EVANS_VERTEX_STAGE.
// Five material MRTs preserve colored ambient separately from emission for later AO.
// Shishkovtsov, "Deferred Shading in S.T.A.L.K.E.R.", GPU Gems 2 (2005).
// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-9-deferred-shading-stalker
precision highp float;
precision highp int;

#if defined(EVANS_CONTACT_RECEIVER) && !defined(EVANS_VERTEX_STAGE)
uniform highp sampler2DArray ContactVisibility;
uniform int ContactEnabled, ContactSample;
// Combine raw contact visibility with the primary map before applying shadow
// strength once. ES image coordinates and the current sample layer stay native.
float ContactShadow(float primary, float strength)
{
    if (ContactEnabled == 0) return primary;
    float contact = texelFetch(ContactVisibility, ivec3(ivec2(gl_FragCoord.xy), ContactSample), 0).r;
    return mix(1.0 - strength, primary, contact);
}
#endif
const float NormalizeEpsilon = 1.0e-12;
const float HomogeneousEpsilon = 1.0e-7;
const float TangentDeterminantEpsilon = 1.0e-10;
const float ParallelAxisLimit = 0.999;
const float MinimumDistanceSquared = 1.0e-6;
const float HalfFloatMaximum = 65504.0;
const float MinimumRoughness = 0.045;
const uint FlatModel = 1u;
const uint PhysicalModel = 3u;
const uint ToonModel = 4u;
const int MaximumLights = 4;

vec3 UnitDirection(vec3 value, vec3 fallback)
{
    float squaredLength = dot(value, value);
    return squaredLength > NormalizeEpsilon ? value * inversesqrt(squaredLength) : fallback;
}
vec3 DecodeVertexColor(vec3 value)
{
    const float SrgbThreshold = 0.04045;
    const float SrgbLinearDivisor = 12.92;
    const float SrgbOffset = 0.055;
    const float SrgbDivisor = 1.055;
    const float SrgbExponent = 2.4;
    vec3 low = value / SrgbLinearDivisor;
    vec3 high = pow((value + SrgbOffset) / SrgbDivisor, vec3(SrgbExponent));
    return mix(high, low, lessThanEqual(value, vec3(SrgbThreshold)));
}

#if !defined(EVANS_DEFERRED_LIGHTING)

#ifdef EVANS_VERTEX_STAGE
// Match camera-depth producers and consumers across separately linked programs.
// GLSL ES 3.00, section 4.6: invariant raster positions prevent replay holes.

layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
uniform mat4 World;
uniform mat4 NormalWorld;
uniform mat4 ViewProjection;
out vec3 SurfacePosition;
out vec3 SurfaceNormal;
out vec4 VertexColor;
out vec2 UV;
void main()
{
    vec4 world = World * vec4(InPosition, 1.0);
    vec4 clipPosition = ViewProjection * world;
    // CPU cameras use [0,1] depth. ES clipping needs [-1,1], while the stored
    // window depth returns to [0,1] and can use the original inverse matrix.
    clipPosition.z = clipPosition.z * 2.0 - clipPosition.w;
    gl_Position = clipPosition;
    SurfacePosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    SurfaceNormal = UnitDirection(mat3(NormalWorld) * InNormal, vec3(0.0, 1.0, 0.0));
    VertexColor = vec4(DecodeVertexColor(clamp(InColor.rgb, 0.0, 1.0)), InColor.a);
    UV = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    CullWireFace();
}
#else
in vec3 SurfacePosition;
in vec3 SurfaceNormal;
in vec4 VertexColor;
in vec2 UV;
uniform highp sampler2D DiffuseMap;
uniform highp sampler2D NormalMap;
uniform highp sampler2D OpacityMap;
uniform vec4 PipelineColor;
uniform vec4 MaterialDiffuseColor;
uniform vec4 MaterialAmbientColor;
uniform vec4 MaterialSpecularColor;
uniform vec4 MaterialEmissiveColor;
uniform vec4 MaterialParameters;
uniform vec4 AmbientColorStrength;
uniform vec4 ShadingParameters;
uniform vec4 RenderParameters;
uniform vec4 TextureFlags;
layout(location = 0) out vec4 OutDiffuseOpacity;
layout(location = 1) out vec4 OutNormalRoughness;
layout(location = 2) out vec4 OutSpecularShininess;
layout(location = 3) out vec4 OutAmbientMetallic;
layout(location = 4) out vec4 OutEmission;
vec3 ResolveNormal(uint model)
{
    vec3 normal = UnitDirection(SurfaceNormal, vec3(0.0, 1.0, 0.0));
    vec3 positionDx = dFdx(SurfacePosition);
    vec3 positionDy = dFdy(SurfacePosition);
    if (model == FlatModel)
    {
        vec3 geometric = UnitDirection(cross(positionDx, positionDy), normal);
        normal = dot(geometric, normal) < 0.0 ? -geometric : geometric;
    }
    if (!gl_FrontFacing) normal = -normal;
    if (model == FlatModel || TextureFlags.y < 0.5) return normal;
    vec2 uvDx = dFdx(UV);
    vec2 uvDy = dFdy(UV);
    float determinant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    if (abs(determinant) < TangentDeterminantEpsilon) return normal;
    vec3 tangent = (positionDx * uvDy.y - positionDy * uvDx.y) / determinant;
    vec3 sourceBitangent = (positionDy * uvDx.x - positionDx * uvDy.x) / determinant;
    tangent -= normal * dot(normal, tangent);
    vec3 helper = abs(normal.y) < ParallelAxisLimit ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    tangent = UnitDirection(tangent, UnitDirection(cross(helper, normal), vec3(1.0, 0.0, 0.0)));
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    vec3 bitangent = UnitDirection(cross(normal, tangent) * handedness, vec3(0.0, 0.0, 1.0));
    vec3 mapped = texture(NormalMap, UV).xyz * 2.0 - 1.0;
    if (float(TextureFlags.w) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
    return UnitDirection(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
}
void main()
{
    vec4 albedo = TextureFlags.x > 0.5 ? texture(DiffuseMap, UV) : vec4(1.0);
    if (float(TextureFlags.w) == ToonShadingModel && float(TextureFlags.x) > 0.5)
        albedo.rgb = ToonTextureColor(DiffuseMap, UV, dFdx(UV), dFdy(UV));
    vec3 surfaceMap = TextureFlags.z > 0.5 ? texture(OpacityMap, UV).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = ShadingParameters.xy;
    if (TextureFlags.z > 1.5) roughnessMetallic *= surfaceMap.gb;
    float opacityMap = surfaceMap.r;
    float opacity = clamp(VertexColor.a * MaterialParameters.y * albedo.a * opacityMap, 0.0, 1.0);
    if (MaterialParameters.z > 0.0 && opacity < MaterialParameters.z) discard;
    uint model = uint(round(TextureFlags.w));
    vec3 surface = ToonAlbedo(VertexColor.rgb * max(PipelineColor.rgb, vec3(0.0)) * max(albedo.rgb, vec3(0.0)), float(TextureFlags.w));
    float metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    vec3 ambient = ToonAmbient(surface * max(MaterialDiffuseColor.rgb, vec3(0.0)), surface * max(MaterialAmbientColor.rgb, vec3(0.0))
        * max(AmbientColorStrength.rgb, vec3(0.0)) * max(AmbientColorStrength.w, 0.0), float(TextureFlags.w));
    if (model == PhysicalModel) ambient *= 1.0 - metallic;
    OutDiffuseOpacity = vec4(clamp(surface * max(MaterialDiffuseColor.rgb, vec3(0.0)), 0.0, HalfFloatMaximum), opacity);
    OutSpecularShininess = clamp(vec4((model == PhysicalModel ? MaterialSpecularColor.www : MaterialSpecularColor.rgb), max(MaterialParameters.x, 1.0)), 0.0, HalfFloatMaximum);
    OutAmbientMetallic = vec4(clamp(ambient, 0.0, HalfFloatMaximum), metallic);
    OutEmission = vec4(clamp(MaterialEmissiveColor.rgb, 0.0, HalfFloatMaximum), clamp(MaterialParameters.w, 0.0, 1.0));
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (RenderParameters.y > 1.0 && RenderParameters.y < 1.5)
    {
        vec3 wireBary = VertexColor.rgb;
        vec3 wireDx = dFdx(wireBary), wireDy = dFdy(wireBary);
        vec3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), vec3(1.0e-8));
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (RenderParameters.y > 0.5)
    {
        const float UnlitCoverageMarker = -1.0;
        OutNormalRoughness = vec4(0.0, 0.0, 0.0, UnlitCoverageMarker);
        OutEmission.rgb = DecodeVertexColor(clamp(PipelineColor.rgb, 0.0, 1.0));
    }
    else OutNormalRoughness = vec4(ResolveNormal(model), clamp(roughnessMetallic.x, MinimumRoughness, 1.0));
}
#endif

#else // Deferred lighting program

#ifdef EVANS_VERTEX_STAGE
out vec2 ScreenUV;
void main()
{
    ScreenUV = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(ScreenUV * 2.0 - 1.0, 0.0, 1.0);
    CullWireFace();
}
#else
in vec2 ScreenUV;
layout(location = 0) out vec4 OutRadiance;
#if defined(EVANS_MULTISAMPLE)
// ES 3.1 uses one fullscreen invocation per stored sample. The output sample
// mask selects its destination; querying sample positions avoids assumed layouts.
uniform int LightingSample;
uniform vec4 SampleLocation;
uniform highp sampler2DMS DiffuseOpacity;
uniform highp sampler2DMS NormalRoughness;
uniform highp sampler2DMS SpecularShininess;
uniform highp sampler2DMS AmbientMetallic;
uniform highp sampler2DMS EmissionColor;
uniform highp sampler2DMS SceneDepth;
#else
uniform highp sampler2D DiffuseOpacity;
uniform highp sampler2D NormalRoughness;
uniform highp sampler2D SpecularShininess;
uniform highp sampler2D AmbientMetallic;
uniform highp sampler2D EmissionColor;
uniform highp sampler2D SceneDepth;
#endif
uniform highp sampler2D ShadowMap;
uniform mat4 InverseViewProjection;
uniform mat4 LightViewProjection;
uniform vec4 CameraLightCount;
uniform vec4 ShadingParameters;
uniform uvec4 LightingOptions; // model, shadow enabled, shadow light index
uniform vec4 BackgroundColor;
struct LightConstants
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
uniform LightConstants Lights[MaximumLights];
struct SurfaceData
{
    vec3 VirtualReceiverData;
    vec3 Position;
    vec3 Normal;
    vec3 View;
    vec3 Diffuse;
    vec3 Specular;
    float Roughness;
    float Shininess;
    float Metallic;
};

// Native ES CSM sampling: positive camera depth selects a fitted interval.
// The CPU includes the previous interval's overlap in each receiver footprint.
// Separate sampler names keep texture selection legal in GLSL ES 3.00.
const int CascadeLimit = 4;
uniform mat4 CascadeLightMatrices[CascadeLimit];
uniform vec4 CascadeFarDistances;
uniform vec4 CascadeBlendStarts;
uniform vec4 CascadeEyeDepthPlane;
uniform vec4 CascadeParameters; // near, far, count, map resolution
uniform vec4 ShadowFiltering; // radius (zero means hard), comparison count, reserved
uniform highp sampler2D CascadeDepth0;
uniform highp sampler2D CascadeDepth1;
uniform highp sampler2D CascadeDepth2;
uniform highp sampler2D CascadeDepth3;
const float CascadeNormalizeEpsilon = 1.0e-12;
const float CascadeHomogeneousEpsilon = 1.0e-7;
const float CascadeIntervalEpsilon = 1.0e-6;
const float CascadeCosineFloor = 0.15;
const float CascadeConstantBias = 0.00035;
const float CascadeSlopeBias = 0.00060;
const float CascadeMaximumBias = 0.0045;
const float CascadeShadowStrength = 0.55;


float CascadeStoredDepth(int cascade, ivec2 pixel)
{
    if (cascade == 0) return texelFetch(CascadeDepth0, pixel, 0).r;
    if (cascade == 1) return texelFetch(CascadeDepth1, pixel, 0).r;
    if (cascade == 2) return texelFetch(CascadeDepth2, pixel, 0).r;
    return texelFetch(CascadeDepth3, pixel, 0).r;
}
float CascadeComparison(int cascade, ivec2 pixel, int resolution, float receiver)
{
    return receiver > CascadeStoredDepth(cascade, clamp(pixel, ivec2(0), ivec2(resolution - 1))) ? 1.0 : 0.0;
}
float CascadeFilteredVisibility(int cascade, vec3 position, float bias)
{
    vec4 projected = CascadeLightMatrices[cascade] * vec4(position, 1.0);
    if (abs(projected.w) < CascadeHomogeneousEpsilon) return 1.0;
    vec3 ndc = projected.xyz / projected.w;
    // ES textures start at the bottom left. Fitted light depth is already [0,1].
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        ndc.z < 0.0 || ndc.z > 1.0) return 1.0;
    int resolution = int(CascadeParameters.w);
    if (ShadowFiltering.x == 0.0)
        return 1.0 - CascadeShadowStrength * CascadeComparison(cascade, ivec2(floor(uv * float(resolution))), resolution, ndc.z - bias);
    vec2 texel = uv * float(resolution) - 0.5;
    ivec2 base = ivec2(floor(texel));
    vec2 fraction = fract(texel);
    float receiver = ndc.z - bias;
    float occlusion = 0.0;
    // Bilinear interpolation of comparisons, never interpolation of depth.
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            ivec2 p = base + ivec2(x, y);
            float lowerLeft = CascadeComparison(cascade, p, resolution, receiver);
            float lowerRight = CascadeComparison(cascade, p + ivec2(1, 0), resolution, receiver);
            float upperLeft = CascadeComparison(cascade, p + ivec2(0, 1), resolution, receiver);
            float upperRight = CascadeComparison(cascade, p + ivec2(1, 1), resolution, receiver);
            occlusion += mix(mix(lowerLeft, lowerRight, fraction.x),
                mix(upperLeft, upperRight, fraction.x), fraction.y);
        }
    return 1.0 - CascadeShadowStrength * occlusion / ShadowFiltering.y;
}
// Virtual receiver metadata occupies the exclusive cascade uniforms. Directory
// rows are uploaded verbatim; only rasterized atlas depth needs an ES Y flip.
// Filtering compares each logical corner after independent ancestor lookup.
const uint VirtualBorder = 1u;
const float VirtualCenter = 0.5;
vec2 VirtualDepth(vec2 uv, uint requestedMip)
{
    uvec4 extent = uvec4(CascadeFarDistances), physical = uvec4(CascadeBlendStarts);
    uint side = extent.x / extent.y, offset = 0u;
    for (uint mip = 0u; mip < extent.z; ++mip)
    {
        if (mip >= requestedMip)
        {
            uint resolution = side * extent.y;
            uvec2 pixel = min(uvec2(uv * float(resolution)), uvec2(resolution - 1u));
            uvec2 page = pixel / extent.y;
            uint address = offset + page.y * side + page.x;
            uint entry = uint(round(texelFetch(CascadeDepth1, ivec2(address % extent.w, address / extent.w), 0).r * float(physical.w)));
            if (entry != 0u)
            {
                uint slot = entry - 1u;
                uvec2 at = uvec2(slot % physical.z, slot / physical.z) * (extent.y + VirtualBorder * 2u) + VirtualBorder + pixel % extent.y;
                at.y = physical.y - 1u - at.y;
                return vec2(texelFetch(CascadeDepth0, ivec2(at), 0).r, float(mip));
            }
        }
        offset += side * side; side /= 2u;
    }
    return vec2(1.0, float(extent.z - 1u));
}
// Compare receiver and caster at the same resolved virtual texel center.
// The actual ancestor mip is essential when detail pages are not resident.
vec2 VirtualReceiverGradient(vec2 uv, float depth)
{
    vec2 x = dFdx(uv), y = dFdy(uv);
    float zx = dFdx(depth), zy = dFdy(depth);
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x, x) * dot(y, y), 1.0e-30);
    if (determinant * determinant <= scale * 1.0e-12) return vec2(0.0, 0.0);
    return vec2(y.y * zx - x.y * zy, x.x * zy - y.x * zx) / determinant;
}
float VirtualReceiverAt(vec2 sampleUv, uint actualMip, vec2 receiverUv, float receiver, vec2 gradient)
{
    float resolution = float(uint(CascadeFarDistances.x) >> actualMip);
    vec2 pixel = clamp(floor(sampleUv * resolution), vec2(0.0, 0.0), vec2(resolution - 1.0, resolution - 1.0));
    vec2 center = (pixel + vec2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualCompare(ivec2 pixel, uint mip, vec2 receiverUv, float receiver, vec2 gradient)
{
    uint resolution = uint(CascadeFarDistances.x) >> mip;
    vec2 uv = (vec2(clamp(pixel, ivec2(0, 0), ivec2(int(resolution) - 1, int(resolution) - 1))) + vec2(0.5, 0.5)) / float(resolution);
    vec2 stored = VirtualDepth(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
// Reconstruct one-sided depth slopes on the same geometric surface.
// Fragment-quad derivatives can cross a G-buffer silhouette or intersection.
// Camera depth is affine within a rasterized triangle; lower second difference
// chooses the side that best preserves that plane without using shading normals.
float DeferredReceiverDepth(ivec2 pixel, uint sampleIndex, uvec2 size)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ivec2(size)))) return 1.0;
    return texelFetch(SceneDepth, pixel, int(sampleIndex)).r;
}
float DeferredReceiverSlope(float center, float negative, float negativeTwo, float positive, float positiveTwo)
{
    bool leftValid = negative < 1.0 && negativeTwo < 1.0;
    bool rightValid = positive < 1.0 && positiveTwo < 1.0;
    if (!leftValid && !rightValid) return 0.0;
    float leftError = abs(center - 2.0 * negative + negativeTwo);
    float rightError = abs(center - 2.0 * positive + positiveTwo);
    return leftValid && (!rightValid || leftError <= rightError) ? center - negative : positive - center;
}
vec3 DeferredReceiverLightPoint(vec2 coordinate, float depth)
{
    vec4 world = InverseViewProjection * vec4((coordinate * 2.0 - 1.0), depth, 1.0);
    vec4 light = CascadeLightMatrices[0] * vec4(world.xyz / world.w, 1.0);
    vec3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    return vec3(projected.xy * vec2(0.5, -0.5) + 0.5, projected.z);
}
vec3 DeferredVirtualReceiverData(vec2 coordinate, float depth, uint sampleIndex)
{
    if (ShadowFiltering.z == 0.0) return vec3(0.0);
#if defined(EVANS_MULTISAMPLE)
    uvec2 size = uvec2(textureSize(SceneDepth));
#else
    uvec2 size = uvec2(textureSize(SceneDepth, 0));
#endif
    ivec2 pixel = ivec2(coordinate * vec2(size));
    float dx = DeferredReceiverSlope(depth,
        DeferredReceiverDepth(pixel + ivec2(-1, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(-2, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(1, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(2, 0), sampleIndex, size));
    float dy = DeferredReceiverSlope(depth,
        DeferredReceiverDepth(pixel + ivec2(0, -1), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(0, -2), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(0, 1), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(0, 2), sampleIndex, size));
    vec3 center = DeferredReceiverLightPoint(coordinate, depth);
    vec3 x = DeferredReceiverLightPoint(coordinate + vec2(1.0 / float(size.x), 0), depth + dx) - center;
    vec3 y = DeferredReceiverLightPoint(coordinate + vec2(0, 1.0 / float(size.y)), depth + dy) - center;
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x.xy, x.xy) * dot(y.xy, y.xy), 1.0e-30);
    vec2 gradient = determinant * determinant > scale * 1.0e-12
        ? vec2(y.y * x.z - x.y * y.z, x.x * y.z - y.x * x.z) / determinant : vec2(0, 0);
    float footprint = max(length(x.xy * CascadeEyeDepthPlane.y), length(y.xy * CascadeEyeDepthPlane.z)) * CascadeFarDistances.x;
    float mip = clamp(floor(log2(max(1.0, footprint)) + CascadeEyeDepthPlane.x), 0.0, CascadeFarDistances.z - 1.0);
    return vec3(gradient, mip);
}
float VirtualVisibility(vec3 position, float bias, vec3 receiverData)
{
    vec4 light = CascadeLightMatrices[0] * vec4(position, 1.0);
    vec3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    vec2 uv = vec2(projected.x, -projected.y) * 0.5 + vec2(0.5, 0.5);
    vec2 gradient = receiverData.xy;
    uint mip = uint(receiverData.z);
    if (light.w <= CascadeEyeDepthPlane.w || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec2 stored = VirtualDepth(uv, mip);
    float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    vec2 texel = uv * float(uint(CascadeFarDistances.x) >> mip) - vec2(0.5, 0.5);
    ivec2 first = ivec2(floor(texel)); vec2 weight = fract(texel); float visible = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            ivec2 p = first + ivec2(x, y);
            float a = VirtualCompare(p, mip, uv, receiver, gradient);
            float b = VirtualCompare(p + ivec2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualCompare(p + ivec2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualCompare(p + ivec2(1, 1), mip, uv, receiver, gradient);
            visible += mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
        }
    return visible / ShadowFiltering.y;
}


float CascadedVisibility(vec3 position, vec3 normal, vec3 lightDirection, vec3 receiverData)
{
    if (ShadowFiltering.z != 0.0)
    {
        float cosine = clamp(dot(normal, lightDirection * inversesqrt(max(dot(lightDirection, lightDirection), CascadeNormalizeEpsilon))), 0.0, 1.0);
        float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CascadeCosineFloor);
        float bias = min(CascadeMaximumBias, CascadeConstantBias + CascadeSlopeBias * slope);
        return 1.0 - CascadeShadowStrength * (1.0 - VirtualVisibility(position, bias, receiverData));
    }
    int count = int(CascadeParameters.z);
    float depth = dot(vec4(position, 1.0), CascadeEyeDepthPlane);
    if (count <= 0 || depth < CascadeParameters.x || depth > CascadeParameters.y) return 1.0;
    int cascade = 0;
    for (int index = 0; index < CascadeLimit - 1; ++index)
        if (index < count - 1 && depth > CascadeFarDistances[index]) cascade = index + 1;
    float lengthSquared = dot(lightDirection, lightDirection);
    vec3 direction = lengthSquared > CascadeNormalizeEpsilon ? lightDirection * inversesqrt(lengthSquared) : normal;
    float cosine = clamp(dot(normal, direction), 0.0, 1.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CascadeCosineFloor);
    float bias = min(CascadeConstantBias + CascadeSlopeBias * slope, CascadeMaximumBias);
    float visibility = CascadeFilteredVisibility(cascade, position, bias);
    float overlap = CascadeFarDistances[cascade] - CascadeBlendStarts[cascade];
    if (overlap > CascadeIntervalEpsilon && depth > CascadeBlendStarts[cascade])
    {
        float nextVisibility = cascade + 1 < count ? CascadeFilteredVisibility(cascade + 1, position, bias) : 1.0;
        float blend = clamp((depth - CascadeBlendStarts[cascade]) / overlap, 0.0, 1.0);
        visibility = mix(visibility, nextVisibility, blend);
    }
    return visibility;
}

float LightVisibility(SurfaceData surface, vec3 direction)
{
    if (LightingOptions.y == 0u) return 1.0;
    if (CascadeParameters.z > 0.0) return CascadedVisibility(surface.Position, surface.Normal, direction, surface.VirtualReceiverData);
    vec4 projected = LightViewProjection * vec4(surface.Position, 1.0);
    if (abs(projected.w) < HomogeneousEpsilon) return 1.0;
    vec3 lightNdc = projected.xyz / projected.w;
    // ES shadow textures and fragment coordinates both have a bottom-left origin.
    vec2 uv = lightNdc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || lightNdc.z < 0.0 || lightNdc.z > 1.0) return 1.0;
    const float CosineFloor = 0.15;
    const float ConstantBias = 0.00035;
    const float SlopeBias = 0.00060;
    const float MaximumBias = 0.0045;
    const float ShadowStrength = 0.55;

    float cosine = clamp(dot(surface.Normal, UnitDirection(direction, surface.Normal)), 0.0, 1.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CosineFloor);
    float receiverDepth = lightNdc.z - min(ConstantBias + SlopeBias * slope, MaximumBias);
    ivec2 size = textureSize(ShadowMap, 0);
    // Nearest hard comparison is separate from bilinear PCF.
    if (ShadowFiltering.x == 0.0)
    {
        ivec2 pixel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
        return 1.0 - ShadowStrength * (receiverDepth > texelFetch(ShadowMap, pixel, 0).r ? 1.0 : 0.0);
    }
    vec2 texel = uv * vec2(size) - 0.5;
    ivec2 base = ivec2(floor(texel));
    vec2 fraction = fract(texel);
    float occlusion = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            ivec2 p = clamp(base + ivec2(x, y), ivec2(0), size - 1);
            float a = receiverDepth > texelFetch(ShadowMap, p, 0).r ? 1.0 : 0.0;
            float b = receiverDepth > texelFetch(ShadowMap, min(p + ivec2(1, 0), size - 1), 0).r ? 1.0 : 0.0;
            float c = receiverDepth > texelFetch(ShadowMap, min(p + ivec2(0, 1), size - 1), 0).r ? 1.0 : 0.0;
            float d = receiverDepth > texelFetch(ShadowMap, min(p + 1, size - 1), 0).r ? 1.0 : 0.0;
            occlusion += mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
        }
    return 1.0 - ShadowStrength * occlusion / ShadowFiltering.y;
}
// Output mode is shared with DeferredOutputMode; no new shader permutation.
const uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly() { return LightingOptions.w == DirectDiffuseOutput; }
vec3 ReflectedSample(SurfaceData surface, vec3 direction, vec3 radiance)
{
    float directionLengthSquared = dot(direction, direction);
    if (directionLengthSquared <= NormalizeEpsilon) return vec3(0.0);
    direction *= inversesqrt(directionLengthSquared);
    float nl = max(dot(surface.Normal, direction), 0.0);
    if (nl <= 0.0 && LightingOptions.x != ToonModel) return vec3(0.0);
    vec3 halfVector = UnitDirection(direction + surface.View, surface.Normal);
    float nh = clamp(dot(surface.Normal, halfVector), 0.0, 1.0);
    if (LightingOptions.x == PhysicalModel)
    {
        // GGX distribution, separable exact Smith masking and Schlick Fresnel.
        // Walter et al., "Microfacet Models for Refraction through Rough Surfaces" (2007).
        const float Pi = 3.141592653589793;
        const float DielectricF0 = 0.04;
        const float FresnelExponent = 5.0;
        const float BrdfEpsilon = 1.0e-6;
        const float DistributionEpsilon = 1.0e-12;
        float nv = clamp(dot(surface.Normal, surface.View), 0.0, 1.0);
        if (nv <= 0.0) return vec3(0.0);
        float vh = clamp(dot(surface.View, halfVector), 0.0, 1.0);
        float alpha = surface.Roughness * surface.Roughness;
        float alphaSquared = alpha * alpha;
        float denominator = nh * nh * (alphaSquared - 1.0) + 1.0;
        float distribution = alphaSquared / max(Pi * denominator * denominator, DistributionEpsilon);
        float smithView = 2.0 * nv / max(nv + sqrt(alphaSquared + (1.0 - alphaSquared) * nv * nv), BrdfEpsilon);
        float smithLight = 2.0 * nl / max(nl + sqrt(alphaSquared + (1.0 - alphaSquared) * nl * nl), BrdfEpsilon);
        vec3 f0 = mix(clamp(surface.Specular,0.0,1.0), clamp(surface.Diffuse, 0.0, 1.0), surface.Metallic);
        vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, FresnelExponent);
        vec3 diffuse = surface.Diffuse * (1.0 - fresnel) * (1.0 - surface.Metallic) / Pi;
        vec3 specular = distribution * smithView * smithLight * fresnel / max(4.0 * nv * nl, BrdfEpsilon);
        if (DirectDiffuseOnly()) return radiance * nl * diffuse;
        return radiance * nl * (diffuse + specular);
    }
    float highlight = pow(nh, surface.Shininess);
    if (LightingOptions.x == ToonModel)
    {
        const float MinimumBands = 2.0;

        float bands = max(ShadingParameters.x, MinimumBands);
        return radiance * (surface.Diffuse * ToonResponse(nl, bands) + (DirectDiffuseOnly() ? vec3(0.0) : surface.Specular * ToonHighlight(highlight)));
    }
    if (DirectDiffuseOnly()) return radiance * surface.Diffuse * nl;
    return radiance * (surface.Diffuse * nl + surface.Specular * highlight);
}
vec3 EvaluateLight(SurfaceData surface, int index)
{
    const int DirectionalType = 0;
    const int PointType = 1;
    LightConstants light = Lights[index];
    int type = int(round(light.PositionType.w));
    vec3 direction = type == DirectionalType ? light.DirectionIntensity.xyz : light.PositionType.xyz - surface.Position;
    float visibility = uint(index) == LightingOptions.z && light.ColorCastShadows.w > 0.5
        ? LightVisibility(surface, direction) : 1.0;
#ifdef EVANS_CONTACT_RECEIVER
    if (uint(index) == LightingOptions.z && light.ColorCastShadows.w > 0.5)
        visibility = ContactShadow(visibility, CascadeShadowStrength);
#endif
    vec3 radiance = max(light.ColorCastShadows.rgb, vec3(0.0)) * max(light.DirectionIntensity.w, 0.0) * visibility;
    if (type == DirectionalType) return ReflectedSample(surface, direction, radiance);
    if (type == PointType) return ReflectedSample(surface, direction, radiance * LightRangeWindow(direction, light.AreaSizePadding.z) / max(dot(direction, direction), MinimumDistanceSquared));
    // Prepared emitters integrate physical area; legacy lights retain normalized energy.
    const int SamplesPerAxis = 2;
    const float SampleOffset = 0.25;
    const float SampleWeight = 0.25;
    vec3 emissionDirection = UnitDirection(light.DirectionIntensity.xyz, vec3(0.0, -1.0, 0.0));
    vec3 helper = abs(emissionDirection.y) < ParallelAxisLimit ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = UnitDirection(cross(helper, emissionDirection), vec3(1.0, 0.0, 0.0));
    vec3 bitangent = UnitDirection(cross(emissionDirection, tangent), vec3(0.0, 0.0, 1.0));
    vec3 result = vec3(0.0);
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSizePadding.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaSizePadding.w, MinimumEmitterGrid, MaximumEmitterGrid)) : SamplesPerAxis;
    float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSizePadding.x * light.AreaSizePadding.y) : 1.0) / float(emitterGrid * emitterGrid);
    for (int y = 0; y < emitterGrid; ++y)
        for (int x = 0; x < emitterGrid; ++x)
        {
            vec2 offset = vec2(((float(x) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter), ((float(y) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter));
            vec3 position = light.PositionType.xyz + tangent * offset.x * light.AreaSizePadding.x
                + bitangent * offset.y * light.AreaSizePadding.y;
            vec3 toSample = position - surface.Position;
            float distanceSquared = max(dot(toSample, toSample), MinimumDistanceSquared);
            float emitterCosine = max(dot(emissionDirection, -toSample * inversesqrt(distanceSquared)), 0.0);
            result += ReflectedSample(surface, toSample, radiance * emitterSampleWeight * emitterCosine * LightRangeWindow(toSample, light.AreaSizePadding.z) / distanceSquared);
        }
    return result;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#if defined(EVANS_MULTISAMPLE)
    int materialSample = LightingSample;
    vec2 coordinate = (vec2(pixel) + SampleLocation.xy) / vec2(textureSize(SceneDepth));
#else
    int materialSample = 0;
    vec2 coordinate = ScreenUV;
#endif
    vec4 normal = texelFetch(NormalRoughness, pixel, materialSample);
    vec4 diffuse = texelFetch(DiffuseOpacity, pixel, materialSample);
    vec4 emission = texelFetch(EmissionColor, pixel, materialSample);
    // Solid-wire overlays keep depth unchanged and can cover clear-depth pixels.
    // Their explicit coverage marker must therefore win over the background test.
    if (normal.w < 0.0) { OutRadiance = DirectDiffuseOnly() ? vec4(0.0) : vec4(emission.rgb, diffuse.a); return; }
    const float BackgroundDepth = 1.0;
    float depth = texelFetch(SceneDepth, pixel, materialSample).r;
    if (depth >= BackgroundDepth) { OutRadiance = DirectDiffuseOnly() ? vec4(0.0) : BackgroundColor; return; }
    vec4 specular = texelFetch(SpecularShininess, pixel, materialSample);
    vec4 ambient = texelFetch(AmbientMetallic, pixel, materialSample);
    vec4 world = InverseViewProjection * vec4(coordinate * 2.0 - 1.0, depth, 1.0);
    SurfaceData surface;
    surface.Position = world.xyz / world.w;
    surface.VirtualReceiverData = DeferredVirtualReceiverData(coordinate, depth, uint(materialSample));
    surface.Normal = UnitDirection(normal.xyz, vec3(0.0, 1.0, 0.0));
    surface.View = UnitDirection(CameraLightCount.xyz - surface.Position, surface.Normal);
    surface.Diffuse = diffuse.rgb;
    surface.Specular = specular.rgb;
    surface.Roughness = max(normal.w, MinimumRoughness);
    surface.Shininess = max(specular.w, 1.0);
    surface.Metallic = clamp(ambient.w, 0.0, 1.0);
    vec3 radiance = DirectDiffuseOnly() ? vec3(0.0) : ambient.rgb + emission.rgb;
    int count = min(int(round(CameraLightCount.w)), MaximumLights);
    for (int index = 0; index < MaximumLights; ++index)
    {
        if (index >= count) break;
        radiance += EvaluateLight(surface, index);
    }
    float coverage = DirectDiffuseOnly() ? (LightingOptions.x == PhysicalModel ? 1.0 - surface.Metallic : 1.0) : diffuse.a;
    if (DirectDiffuseOnly()) { radiance *= emission.a; coverage *= emission.a; }
    OutRadiance = vec4(clamp(radiance, 0.0, HalfFloatMaximum), coverage);
}
#endif
#endif
