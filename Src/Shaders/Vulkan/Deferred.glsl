#version 450
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
vec3 ToonTextureColor(texture2D image, sampler imageSampler, vec2 uv, vec2 uvDx, vec2 uvDy)
{
    vec3 color = vec3(0.0);
    for (int row = -ToonTextureKernelRadius; row <= ToonTextureKernelRadius; ++row)
        for (int column = -ToonTextureKernelRadius; column <= ToonTextureKernelRadius; ++column)
            color += textureGrad(sampler2D(image, imageSampler), uv + ToonTextureRadius * (float(column) * uvDx + float(row) * uvDy), uvDx, uvDy).rgb;
    return color * ToonTextureSampleWeight;
}

// Karis' smooth inverse-square window (Filament, Punctual lights).
// Range is measured at each emitter sample; zero keeps infinite support.
float LightRangeWindow(vec3 delta, float range)
{
    if (range <= 0.0) return 1.0;
    if (any(greaterThanEqual(abs(delta), vec3(range)))) return 0.0;
    vec3 ratio = delta / range;
    float radiusFractionSquared = dot(ratio, ratio);
    float window = clamp(1.0 - radiusFractionSquared * radiusFractionSquared, 0.0, 1.0);
    return window * window;
}

// Native Vulkan deferred geometry and lighting, independently authored for the
// explicit image/sampler descriptors and row-major CPU uniform contract.
// Select EVANS_DEFERRED_LIGHTING for the fullscreen program and
// EVANS_VERTEX_STAGE for either program's vertex stage. No shader includes.
// Shishkovtsov, "Deferred Shading in S.T.A.L.K.E.R.", GPU Gems 2 (2005).
// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-9-deferred-shading-stalker
const int MaximumLights = 4;
const uint FlatModel = 1u;
const uint PhysicalModel = 3u;
const uint ToonModel = 4u;
const float NormalizeEpsilon = 1.0e-12;
const float HomogeneousEpsilon = 1.0e-7;
const float TangentEpsilon = 1.0e-10;
const float ParallelAxisLimit = 0.999;
const float MinimumRoughness = 0.045;
const float HalfFloatMaximum = 65504.0;
struct SceneLight
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
vec3 UnitDirection(vec3 value, vec3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalizeEpsilon ? value * inversesqrt(lengthSquared) : fallback;
}

#ifndef EVANS_DEFERRED_LIGHTING
layout(set = 0, binding = 0, std140, row_major) uniform GeometryConstants
{
    mat4 World;
    mat4 NormalWorld;
    mat4 ViewProjection;
    mat4 LightViewProjection;
    vec4 PipelineColor;
    vec4 MaterialDiffuseColor;
    vec4 MaterialAmbientColor;
    vec4 MaterialSpecularColor;
    vec4 MaterialEmissiveColor;
    vec4 MaterialParameters;
    vec4 AmbientColorStrength;
    vec4 CameraLightCount;
    vec4 RenderParameters;
    uvec4 TextureFlags;
    SceneLight Lights[MaximumLights];
    vec4 ShadingParameters;
} Draw;
vec3 DecodeSrgb(vec3 color)
{
    const float LinearThreshold = 0.04045;
    const float LinearScale = 12.92;
    const float PowerOffset = 0.055;
    const float PowerScale = 1.055;
    const float PowerExponent = 2.4;
    vec3 linear = color / LinearScale;
    vec3 power = pow((color + PowerOffset) / PowerScale, vec3(PowerExponent));
    return mix(power, linear, lessThanEqual(color, vec3(LinearThreshold)));
}
#ifdef EVANS_VERTEX_STAGE
// Match opaque forward depth exactly when material capture is enabled.
invariant gl_Position;
precise gl_Position;
layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
layout(location = 0) out vec3 WorldPosition;
layout(location = 1) out vec3 WorldNormal;
layout(location = 2) out vec4 VertexColor;
layout(location = 3) out vec2 TexCoord;
void main()
{
    vec4 world = Draw.World * vec4(InPosition, 1.0);
    gl_Position = Draw.ViewProjection * world;
    // Positive Vulkan viewport height maps NDC -Y to the top. Keep CPU cameras
    // unchanged; the lighting reconstruction applies the matching inverse flip.
    gl_Position.y = -gl_Position.y;
    WorldPosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    WorldNormal = UnitDirection(mat3(Draw.NormalWorld) * InNormal, vec3(0.0, 1.0, 0.0));
    VertexColor = vec4(DecodeSrgb(clamp(InColor.rgb, 0.0, 1.0)), InColor.a);
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
}
#else
layout(location = 0) in vec3 WorldPosition;
layout(location = 1) in vec3 WorldNormal;
layout(location = 2) in vec4 VertexColor;
layout(location = 3) in vec2 TexCoord;
layout(set = 0, binding = 2) uniform texture2D DiffuseImage;
layout(set = 0, binding = 3) uniform texture2D NormalImage;
layout(set = 0, binding = 4) uniform texture2D OpacityImage;
layout(set = 0, binding = 5) uniform sampler MaterialSampler;
layout(location = 0) out vec4 DiffuseOpacity;
layout(location = 1) out vec4 NormalRoughness;
layout(location = 2) out vec4 SpecularShininess;
layout(location = 3) out vec4 AmbientMetallic;
layout(location = 4) out vec4 Emission;
vec3 MaterialNormal()
{
    vec3 normal = UnitDirection(WorldNormal, vec3(0.0, 1.0, 0.0));
    vec3 dx = dFdx(WorldPosition);
    vec3 dy = dFdy(WorldPosition);
    if (Draw.TextureFlags.w == FlatModel)
    {
        vec3 geometric = UnitDirection(cross(dx, dy), normal);
        normal = dot(geometric, normal) < 0.0 ? -geometric : geometric;
    }
    if (!gl_FrontFacing) normal = -normal;
    if (Draw.TextureFlags.w == FlatModel || Draw.TextureFlags.y == 0u) return normal;
    vec2 uvDx = dFdx(TexCoord);
    vec2 uvDy = dFdy(TexCoord);
    float determinant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    if (abs(determinant) < TangentEpsilon) return normal;
    vec3 tangent = (dx * uvDy.y - dy * uvDx.y) / determinant;
    vec3 sourceBitangent = (dy * uvDx.x - dx * uvDy.x) / determinant;
    tangent -= normal * dot(normal, tangent);
    vec3 helper = abs(normal.y) < ParallelAxisLimit ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    tangent = UnitDirection(tangent, UnitDirection(cross(helper, normal), vec3(1.0, 0.0, 0.0)));
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    vec3 bitangent = UnitDirection(cross(normal, tangent) * handedness, vec3(0.0, 0.0, 1.0));
    vec3 mapped = texture(sampler2D(NormalImage, MaterialSampler), TexCoord).xyz * 2.0 - 1.0;
    if (float(Draw.TextureFlags.w) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
    return UnitDirection(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
}
void main()
{
    vec4 albedo = Draw.TextureFlags.x != 0u ? texture(sampler2D(DiffuseImage, MaterialSampler), TexCoord) : vec4(1.0);
    if (float(Draw.TextureFlags.w) == ToonShadingModel && float(Draw.TextureFlags.x) > 0.5)
        albedo.rgb = ToonTextureColor(DiffuseImage, MaterialSampler, TexCoord, dFdx(TexCoord), dFdy(TexCoord));
    vec3 surfaceMap = Draw.TextureFlags.z != 0 ? texture(sampler2D(OpacityImage, MaterialSampler), TexCoord).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = Draw.ShadingParameters.xy;
    if (Draw.TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacityMap = surfaceMap.r;
    float opacity = clamp(VertexColor.a * Draw.MaterialParameters.y * albedo.a * opacityMap, 0.0, 1.0);
    if (Draw.MaterialParameters.z > 0.0 && opacity < Draw.MaterialParameters.z) discard;
    vec3 surface = ToonAlbedo(VertexColor.rgb * max(Draw.PipelineColor.rgb, vec3(0.0)) * max(albedo.rgb, vec3(0.0)), float(Draw.TextureFlags.w));
    float metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    vec3 ambient = ToonAmbient(surface * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), surface * max(Draw.MaterialAmbientColor.rgb, vec3(0.0))
        * max(Draw.AmbientColorStrength.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.w, 0.0), float(Draw.TextureFlags.w));
    if (Draw.TextureFlags.w == PhysicalModel) ambient *= 1.0 - metallic;
    DiffuseOpacity = vec4(clamp(surface * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), 0.0, HalfFloatMaximum), opacity);
    SpecularShininess = clamp(vec4((Draw.TextureFlags.w == 3u ? Draw.MaterialSpecularColor.www : Draw.MaterialSpecularColor.rgb), max(Draw.MaterialParameters.x, 1.0)), 0.0, HalfFloatMaximum);
    AmbientMetallic = vec4(clamp(ambient, 0.0, HalfFloatMaximum), metallic);
    Emission = vec4(clamp(Draw.MaterialEmissiveColor.rgb, 0.0, HalfFloatMaximum), clamp(Draw.MaterialParameters.w, 0.0, 1.0));
    const float OverlayMode = 0.5;
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (Draw.RenderParameters.y > 1.0 && Draw.RenderParameters.y < 1.5)
    {
        vec3 wireBary = VertexColor.rgb;
        vec3 wireDx = dFdx(wireBary), wireDy = dFdy(wireBary);
        vec3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), vec3(1.0e-8));
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (Draw.RenderParameters.y > OverlayMode)
    {
        const float UnlitCoverage = -1.0;
        NormalRoughness = vec4(0.0, 0.0, 0.0, UnlitCoverage);
        Emission.rgb = DecodeSrgb(clamp(Draw.PipelineColor.rgb, 0.0, 1.0));
    }
    else NormalRoughness = vec4(MaterialNormal(), clamp(roughnessMetallic.x, MinimumRoughness, 1.0));
}
#endif

#else // Fullscreen deferred lighting
#ifdef EVANS_VERTEX_STAGE
layout(location = 0) out vec2 ScreenUV;
void main()
{
    ScreenUV = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(ScreenUV * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(location = 0) in vec2 ScreenUV;
layout(location = 0) out vec4 OutRadiance;
layout(set = 0, binding = 0, std140, row_major) uniform LightingConstants
{
    mat4 InverseViewProjection;
    mat4 LightViewProjection;
    vec4 CameraLightCount;
    vec4 ShadingParameters;
    uvec4 Options;
    vec4 BackgroundColor;
    SceneLight Lights[MaximumLights];
} Lighting;
#if defined(EVANS_MULTISAMPLE)
layout(set = 0, binding = 1) uniform texture2DMS DiffuseOpacityImage;
layout(set = 0, binding = 2) uniform texture2DMS NormalRoughnessImage;
layout(set = 0, binding = 3) uniform texture2DMS SpecularShininessImage;
layout(set = 0, binding = 4) uniform texture2DMS AmbientMetallicImage;
layout(set = 0, binding = 5) uniform texture2DMS EmissionImage;
layout(set = 0, binding = 6) uniform texture2DMS SceneDepth;
#else
layout(set = 0, binding = 1) uniform texture2D DiffuseOpacityImage;
layout(set = 0, binding = 2) uniform texture2D NormalRoughnessImage;
layout(set = 0, binding = 3) uniform texture2D SpecularShininessImage;
layout(set = 0, binding = 4) uniform texture2D AmbientMetallicImage;
layout(set = 0, binding = 5) uniform texture2D EmissionImage;
layout(set = 0, binding = 6) uniform texture2D SceneDepth;
#endif
layout(set = 0, binding = 7) uniform texture2D ShadowDepth;
layout(set = 0, binding = 8) uniform sampler NearestSampler;

// Contact is a separate camera-depth producer. Apply artistic strength once:
// (1 - strength) + strength * rawMapVisibility * rawContactVisibility.
#if defined(EVANS_CONTACT_RECEIVER)
layout(set = 0, binding = 9) uniform texture2DArray ContactImage;
#endif
float ComposeContact(float primaryVisibility)
{
#if defined(EVANS_CONTACT_RECEIVER)
    const float ContactShadowStrength = 0.55;
#ifdef EVANS_MULTISAMPLE
    int sampleIndex = gl_SampleID;
#else
    int sampleIndex = 0;
#endif
    float contact = texelFetch(sampler2DArray(ContactImage, NearestSampler),
        ivec3(ivec2(gl_FragCoord.xy), sampleIndex), 0).r;
    return mix(1.0 - ContactShadowStrength, primaryVisibility, contact);
#else
    return primaryVisibility;
#endif
}

struct Surface
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
float CompareShadow(ivec2 pixel, float receiver)
{
    return receiver > texelFetch(sampler2D(ShadowDepth, NearestSampler), pixel, 0).r ? 1.0 : 0.0;
}
// Set 1 is shared by Vulkan lighting layouts; images are separate descriptors
// so this path requires neither descriptor indexing nor non-uniform indexing.
const int CsmMaximumSplits = 4;
layout(set = 1, binding = 0, std140, row_major) uniform CascadeConstants
{
    mat4 LightMatrices[CsmMaximumSplits];
    vec4 FarDepths;
    vec4 BlendDepths;
    vec4 EyeDepthPlane;
    vec4 Parameters; // near, far, count, resolution
    vec4 Filtering; // radius (zero means hard), comparison count, reserved
} Csm;
layout(set = 1, binding = 1) uniform texture2D CsmDepth0;
layout(set = 1, binding = 2) uniform texture2D CsmDepth1;
layout(set = 1, binding = 3) uniform texture2D CsmDepth2;
layout(set = 1, binding = 4) uniform texture2D CsmDepth3;
layout(set = 1, binding = 5) uniform sampler CsmNearest;
const float CsmDirectionEpsilon = 1.0e-12;
const float CsmProjectionEpsilon = 1.0e-7;
const float CsmBlendEpsilon = 1.0e-6;
const float CsmCosineFloor = 0.15;
const float CsmBaseBias = 0.00035;
const float CsmSlopeBias = 0.00060;
const float CsmBiasCeiling = 0.0045;
const float CsmOcclusionStrength = 0.55;


float CsmDepthAt(int split, ivec2 pixel)
{
    if (split == 0) return texelFetch(sampler2D(CsmDepth0, CsmNearest), pixel, 0).r;
    if (split == 1) return texelFetch(sampler2D(CsmDepth1, CsmNearest), pixel, 0).r;
    if (split == 2) return texelFetch(sampler2D(CsmDepth2, CsmNearest), pixel, 0).r;
    return texelFetch(sampler2D(CsmDepth3, CsmNearest), pixel, 0).r;
}
float CsmOcclusionAt(int split, ivec2 pixel, float receiver)
{
    ivec2 maximumPixel = ivec2(int(Csm.Parameters.w) - 1);
    return receiver > CsmDepthAt(split, clamp(pixel, ivec2(0), maximumPixel)) ? 1.0 : 0.0;
}
float CsmSampleInterval(int split, vec3 worldPosition, float bias)
{
    vec4 light = Csm.LightMatrices[split] * vec4(worldPosition, 1.0);
    if (abs(light.w) < CsmProjectionEpsilon) return 1.0;
    vec3 ndc = light.xyz / light.w;
    // Depth rendering flips clip Y for a positive-height viewport; reproduce
    // that convention when addressing the top-left Vulkan depth image.
    vec2 uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        ndc.z < 0.0 || ndc.z > 1.0) return 1.0;
    if (Csm.Filtering.x == 0.0)
        return 1.0 - CsmOcclusionStrength * CsmOcclusionAt(split, ivec2(floor(uv * Csm.Parameters.w)), ndc.z - bias);
    vec2 texel = uv * Csm.Parameters.w - 0.5;
    ivec2 origin = ivec2(floor(texel));
    vec2 weight = fract(texel);
    float receiver = ndc.z - bias;
    float occlusion = 0.0;
    for (int row = -int(Csm.Filtering.x); row <= int(Csm.Filtering.x); ++row)
        for (int column = -int(Csm.Filtering.x); column <= int(Csm.Filtering.x); ++column)
        {
            ivec2 pixel = origin + ivec2(column, row);
            float upperLeft = CsmOcclusionAt(split, pixel, receiver);
            float upperRight = CsmOcclusionAt(split, pixel + ivec2(1, 0), receiver);
            float lowerLeft = CsmOcclusionAt(split, pixel + ivec2(0, 1), receiver);
            float lowerRight = CsmOcclusionAt(split, pixel + ivec2(1, 1), receiver);
            occlusion += mix(mix(upperLeft, upperRight, weight.x),
                mix(lowerLeft, lowerRight, weight.x), weight.y);
        }
    return 1.0 - CsmOcclusionStrength * occlusion * (1.0 / Csm.Filtering.y);
}
// The virtual hierarchy shares set 1 with CSM. FarDepths packs logical extent;
// BlendDepths packs physical extent. Normalized directory entries decode exact
// integer slots, and every comparison corner performs its own coarse fallback.
const uint VirtualPageGutter = 1u;
const float VirtualTexelOrigin = 0.5;
vec2 VirtualDepthAt(vec2 coordinate, uint firstLevel)
{
    uvec4 hierarchy = uvec4(Csm.FarDepths), atlas = uvec4(Csm.BlendDepths);
    uint pages = hierarchy.x / hierarchy.y, start = 0u;
    for (uint level = 0u; level < hierarchy.z; ++level)
    {
        if (level >= firstLevel)
        {
            uint resolution = pages * hierarchy.y;
            uvec2 pixel = min(uvec2(coordinate * float(resolution)), uvec2(resolution - 1u));
            uvec2 page = pixel / hierarchy.y;
            uint index = start + page.y * pages + page.x;
            ivec2 directoryPixel = ivec2(index % hierarchy.w, index / hierarchy.w);
            uint resident = uint(round(texelFetch(sampler2D(CsmDepth1, CsmNearest), directoryPixel, 0).r * float(atlas.w)));
            if (resident > 0u)
            {
                uint slot = resident - 1u;
                uvec2 tile = uvec2(slot % atlas.z, slot / atlas.z) * (hierarchy.y + 2u * VirtualPageGutter);
                ivec2 source = ivec2(tile + VirtualPageGutter + pixel % hierarchy.y);
                return vec2(texelFetch(sampler2D(CsmDepth0, CsmNearest), source, 0).r, float(level));
            }
        }
        start += pages * pages; pages /= 2u;
    }
    return vec2(1.0, float(hierarchy.z - 1u));
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
    float resolution = float(uint(Csm.FarDepths.x) >> actualMip);
    vec2 pixel = clamp(floor(sampleUv * resolution), vec2(0.0, 0.0), vec2(resolution - 1.0, resolution - 1.0));
    vec2 center = (pixel + vec2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualCompareAt(ivec2 pixel, uint mip, vec2 receiverUv, float receiver, vec2 gradient)
{
    uint resolution = uint(Csm.FarDepths.x) >> mip;
    vec2 uv = (vec2(clamp(pixel, ivec2(0, 0), ivec2(int(resolution) - 1, int(resolution) - 1))) + vec2(0.5, 0.5)) / float(resolution);
    vec2 stored = VirtualDepthAt(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
// Reconstruct one-sided depth slopes on the same geometric surface.
// Fragment-quad derivatives can cross a G-buffer silhouette or intersection.
// Camera depth is affine within a rasterized triangle; lower second difference
// chooses the side that best preserves that plane without using shading normals.
float DeferredReceiverDepth(ivec2 pixel, uint sampleIndex, uvec2 size)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ivec2(size)))) return 1.0;
#if defined(EVANS_MULTISAMPLE)
    return texelFetch(sampler2DMS(SceneDepth, NearestSampler), pixel, int(sampleIndex)).r;
#else
    return texelFetch(sampler2D(SceneDepth, NearestSampler), pixel, 0).r;
#endif
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
    vec4 world = Lighting.InverseViewProjection * vec4(vec2(coordinate.x * 2.0 - 1.0, 1.0 - coordinate.y * 2.0), depth, 1.0);
    vec4 light = Csm.LightMatrices[0] * vec4(world.xyz / world.w, 1.0);
    vec3 projected = light.xyz / max(light.w, Csm.EyeDepthPlane.w);
    return vec3(projected.xy * vec2(0.5, -0.5) + 0.5, projected.z);
}
vec3 DeferredVirtualReceiverData(vec2 coordinate, float depth, uint sampleIndex)
{
    if (Csm.Filtering.z == 0.0) return vec3(0.0);
#if defined(EVANS_MULTISAMPLE)
    uvec2 size = uvec2(textureSize(sampler2DMS(SceneDepth, NearestSampler)));
#else
    uvec2 size = uvec2(textureSize(sampler2D(SceneDepth, NearestSampler), 0));
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
    float footprint = max(length(x.xy * Csm.EyeDepthPlane.y), length(y.xy * Csm.EyeDepthPlane.z)) * Csm.FarDepths.x;
    float mip = clamp(floor(log2(max(1.0, footprint)) + Csm.EyeDepthPlane.x), 0.0, Csm.FarDepths.z - 1.0);
    return vec3(gradient, mip);
}
float VirtualVisibility(vec3 position, float bias, vec3 receiverData)
{
    vec4 light = Csm.LightMatrices[0] * vec4(position, 1.0);
    vec3 projected = light.xyz / max(light.w, Csm.EyeDepthPlane.w);
    vec2 uv = vec2(projected.x, -projected.y) * 0.5 + vec2(0.5, 0.5);
    vec2 gradient = receiverData.xy;
    uint mip = uint(receiverData.z);
    if (light.w <= Csm.EyeDepthPlane.w || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec2 stored = VirtualDepthAt(uv, mip);
    float receiver = projected.z - bias;
    if (Csm.Filtering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    vec2 texel = uv * float(uint(Csm.FarDepths.x) >> mip) - vec2(0.5, 0.5);
    ivec2 first = ivec2(floor(texel)); vec2 weight = fract(texel); float visible = 0.0;
    for (int y = -int(Csm.Filtering.x); y <= int(Csm.Filtering.x); ++y)
        for (int x = -int(Csm.Filtering.x); x <= int(Csm.Filtering.x); ++x)
        {
            ivec2 p = first + ivec2(x, y);
            float a = VirtualCompareAt(p, mip, uv, receiver, gradient);
            float b = VirtualCompareAt(p + ivec2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualCompareAt(p + ivec2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualCompareAt(p + ivec2(1, 1), mip, uv, receiver, gradient);
            visible += mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
        }
    return visible / Csm.Filtering.y;
}


float CsmVisibility(vec3 position, vec3 normal, vec3 lightVector, vec3 receiverData)
{
    if (Csm.Filtering.z != 0.0)
    {
        float cosine = clamp(dot(normal, lightVector * inversesqrt(max(dot(lightVector, lightVector), CsmDirectionEpsilon))), 0.0, 1.0);
        float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CsmCosineFloor);
        float bias = min(CsmBiasCeiling, CsmBaseBias + CsmSlopeBias * slope);
        return 1.0 - CsmOcclusionStrength * (1.0 - VirtualVisibility(position, bias, receiverData));
    }
    float eyeDepth = dot(Csm.EyeDepthPlane, vec4(position, 1.0));
    int splitCount = int(Csm.Parameters.z);
    if (splitCount <= 0 || eyeDepth < Csm.Parameters.x || eyeDepth > Csm.Parameters.y) return 1.0;
    int selected = 0;
    for (int index = 0; index < CsmMaximumSplits - 1; ++index)
        if (index + 1 < splitCount && eyeDepth > Csm.FarDepths[index]) selected = index + 1;
    float squaredLength = dot(lightVector, lightVector);
    vec3 direction = squaredLength > CsmDirectionEpsilon ? lightVector * inversesqrt(squaredLength) : normal;
    float cosine = clamp(dot(normal, direction), 0.0, 1.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CsmCosineFloor);
    float bias = min(CsmBaseBias + slope * CsmSlopeBias, CsmBiasCeiling);
    float visibility = CsmSampleInterval(selected, position, bias);
    float overlap = Csm.FarDepths[selected] - Csm.BlendDepths[selected];
    if (overlap > CsmBlendEpsilon && eyeDepth > Csm.BlendDepths[selected])
    {
        // The final interval fades to the unshadowed region beyond its distance.
        float neighbor = selected + 1 < splitCount ? CsmSampleInterval(selected + 1, position, bias) : 1.0;
        float blend = clamp((eyeDepth - Csm.BlendDepths[selected]) / overlap, 0.0, 1.0);
        visibility = mix(visibility, neighbor, blend);
    }
    return visibility;
}

float ShadowVisibility(Surface surface, vec3 lightVector)
{
    if (Lighting.Options.y == 0u) return 1.0;
    if (Csm.Parameters.z > 0.0) return CsmVisibility(surface.Position, surface.Normal, lightVector, surface.VirtualReceiverData);
    vec4 lightPosition = Lighting.LightViewProjection * vec4(surface.Position, 1.0);
    if (abs(lightPosition.w) < HomogeneousEpsilon) return 1.0;
    vec3 projected = lightPosition.xyz / lightPosition.w;
    vec2 uv = vec2(projected.x * 0.5 + 0.5, -projected.y * 0.5 + 0.5);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    const float CosineFloor = 0.15;
    const float ConstantBias = 0.00035;
    const float SlopeBias = 0.00060;
    const float MaximumBias = 0.0045;
    const float ShadowStrength = 0.55;

    float cosine = clamp(dot(surface.Normal, UnitDirection(lightVector, surface.Normal)), 0.0, 1.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CosineFloor);
    float receiver = projected.z - min(ConstantBias + SlopeBias * slope, MaximumBias);
    ivec2 extent = textureSize(sampler2D(ShadowDepth, NearestSampler), 0);
    // Manual depth fetch keeps hard edges distinct from comparison filtering.
    if (Csm.Filtering.x == 0.0)
    {
        ivec2 pixel = clamp(ivec2(floor(uv * vec2(extent))), ivec2(0), extent - 1);
        return 1.0 - ShadowStrength * CompareShadow(pixel, receiver);
    }
    vec2 texel = uv * vec2(extent) - 0.5;
    ivec2 base = ivec2(floor(texel));
    vec2 fraction = fract(texel);
    float blocked = 0.0;
    for (int y = -int(Csm.Filtering.x); y <= int(Csm.Filtering.x); ++y)
        for (int x = -int(Csm.Filtering.x); x <= int(Csm.Filtering.x); ++x)
        {
            ivec2 p = clamp(base + ivec2(x, y), ivec2(0), extent - 1);
            float a = CompareShadow(p, receiver);
            float b = CompareShadow(min(p + ivec2(1, 0), extent - 1), receiver);
            float c = CompareShadow(min(p + ivec2(0, 1), extent - 1), receiver);
            float d = CompareShadow(min(p + 1, extent - 1), receiver);
            blocked += mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
        }
    return 1.0 - ShadowStrength * blocked / Csm.Filtering.y;
}
// Output mode is shared with DeferredOutputMode; no new shader permutation.
const uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly() { return Lighting.Options.w == DirectDiffuseOutput; }
vec3 ShadeSample(Surface surface, vec3 lightVector, vec3 radiance)
{
    float distanceSquared = dot(lightVector, lightVector);
    if (distanceSquared <= NormalizeEpsilon) return vec3(0.0);
    vec3 direction = lightVector * inversesqrt(distanceSquared);
    float nl = max(dot(surface.Normal, direction), 0.0);
    if (nl <= 0.0 && Lighting.Options.x != ToonModel) return vec3(0.0);
    vec3 halfway = UnitDirection(direction + surface.View, surface.Normal);
    float nh = clamp(dot(surface.Normal, halfway), 0.0, 1.0);
    if (Lighting.Options.x == PhysicalModel)
    {
        // GGX with separable exact Smith masking and Schlick Fresnel.
        // Walter et al., "Microfacet Models for Refraction through Rough Surfaces" (2007).
        const float Pi = 3.141592653589793;
        const float DielectricF0 = 0.04;
        const float FresnelExponent = 5.0;
        const float BrdfEpsilon = 1.0e-6;
        const float DistributionEpsilon = 1.0e-12;
        float nv = clamp(dot(surface.Normal, surface.View), 0.0, 1.0);
        if (nv <= 0.0) return vec3(0.0);
        float vh = clamp(dot(surface.View, halfway), 0.0, 1.0);
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
    if (Lighting.Options.x == ToonModel)
    {
        const float MinimumBands = 2.0;

        float bands = max(Lighting.ShadingParameters.x, MinimumBands);
        return radiance * (surface.Diffuse * ToonResponse(nl, bands) + (DirectDiffuseOnly() ? vec3(0.0) : surface.Specular * ToonHighlight(highlight)));
    }
    if (DirectDiffuseOnly()) return radiance * surface.Diffuse * nl;
    return radiance * (surface.Diffuse * nl + surface.Specular * highlight);
}
vec3 ShadeLight(Surface surface, int index)
{
    const int DirectionalLight = 0;
    const int PointLight = 1;
    const float MinimumDistanceSquared = 1.0e-6;
    const float EnabledThreshold = 0.5;
    SceneLight light = Lighting.Lights[index];
    int type = int(round(light.PositionType.w));
    vec3 toLight = type == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - surface.Position;
    float visibility = uint(index) == Lighting.Options.z && light.ColorCastShadows.w > EnabledThreshold
        ? ComposeContact(ShadowVisibility(surface, toLight)) : 1.0;
    vec3 radiance = max(light.ColorCastShadows.rgb, vec3(0.0)) * max(light.DirectionIntensity.w, 0.0) * visibility;
    if (type == DirectionalLight) return ShadeSample(surface, toLight, radiance);
    if (type == PointLight) return ShadeSample(surface, toLight, radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / max(dot(toLight, toLight), MinimumDistanceSquared));
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
            vec3 displacement = position - surface.Position;
            float distanceSquared = max(dot(displacement, displacement), MinimumDistanceSquared);
            float emitterCosine = max(dot(emissionDirection, -displacement * inversesqrt(distanceSquared)), 0.0);
            result += ShadeSample(surface, displacement, radiance * emitterSampleWeight * emitterCosine * LightRangeWindow(displacement, light.AreaSizePadding.z) / distanceSquared);
        }
    return result;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#if defined(EVANS_MULTISAMPLE)
    // Sample-rate lighting reads one surface and reconstructs at its own raster
    // location; no fixed pattern or averaged normal is substituted at silhouettes.
    int materialSample = gl_SampleID;
    vec2 coordinate = (vec2(pixel) + gl_SamplePosition) / vec2(textureSize(sampler2DMS(SceneDepth, NearestSampler)));
#define MaterialSampler sampler2DMS
#else
    int materialSample = 0;
    vec2 coordinate = ScreenUV;
#define MaterialSampler sampler2D
#endif
    vec4 normal = texelFetch(MaterialSampler(NormalRoughnessImage, NearestSampler), pixel, materialSample);
    vec4 diffuse = texelFetch(MaterialSampler(DiffuseOpacityImage, NearestSampler), pixel, materialSample);
    vec4 emission = texelFetch(MaterialSampler(EmissionImage, NearestSampler), pixel, materialSample);
    // An unlit overlay can cover a pixel whose depth is still clear because
    // solid-wire rendering preserves the opaque depth buffer.
    if (normal.w < 0.0) { OutRadiance = DirectDiffuseOnly() ? vec4(0.0) : vec4(emission.rgb, diffuse.a); return; }
    const float ClearDepth = 1.0;
    float depth = texelFetch(MaterialSampler(SceneDepth, NearestSampler), pixel, materialSample).r;
    if (depth >= ClearDepth) { OutRadiance = DirectDiffuseOnly() ? vec4(0.0) : Lighting.BackgroundColor; return; }
    vec4 specular = texelFetch(MaterialSampler(SpecularShininessImage, NearestSampler), pixel, materialSample);
    vec4 ambient = texelFetch(MaterialSampler(AmbientMetallicImage, NearestSampler), pixel, materialSample);
    vec2 cameraNdc = vec2(coordinate.x * 2.0 - 1.0, 1.0 - coordinate.y * 2.0);
    vec4 world = Lighting.InverseViewProjection * vec4(cameraNdc, depth, 1.0);
    Surface surface;
    surface.Position = world.xyz / world.w;
    surface.VirtualReceiverData = DeferredVirtualReceiverData(coordinate, depth, uint(materialSample));
    surface.Normal = UnitDirection(normal.xyz, vec3(0.0, 1.0, 0.0));
    surface.View = UnitDirection(Lighting.CameraLightCount.xyz - surface.Position, surface.Normal);
    surface.Diffuse = diffuse.rgb;
    surface.Specular = specular.rgb;
    surface.Roughness = max(normal.w, MinimumRoughness);
    surface.Shininess = max(specular.w, 1.0);
    surface.Metallic = clamp(ambient.w, 0.0, 1.0);
    vec3 radiance = DirectDiffuseOnly() ? vec3(0.0) : ambient.rgb + emission.rgb;
    int count = min(int(round(Lighting.CameraLightCount.w)), MaximumLights);
    for (int index = 0; index < MaximumLights; ++index)
    {
        if (index >= count) break;
        radiance += ShadeLight(surface, index);
    }
    float coverage = DirectDiffuseOnly() ? (Lighting.Options.x == PhysicalModel ? 1.0 - surface.Metallic : 1.0) : diffuse.a;
    if (DirectDiffuseOnly()) { radiance *= emission.a; coverage *= emission.a; }
    OutRadiance = vec4(clamp(radiance, 0.0, HalfFloatMaximum), coverage);
}
#endif
#endif
