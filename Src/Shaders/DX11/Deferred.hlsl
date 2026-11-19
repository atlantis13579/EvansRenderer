// Anime environment palette: lift photographic albedo, keep broad form lighting,
// tint the diffuse ramp and limit highlights. These are artistic defaults, not
// a reproduction of Genshin assets. Art direction: Cai, GDC 2021,
// "Genshin Impact: Crafting an Anime Style Open World".
static const float ToonShadingModel = 4.0;
static const float ToonAlbedoExponent = 0.8;
static const float ToonPaletteSteps = 16.0;
static const float ToonColorSaturation = 1.12;
static const float ToonNormalStrength = 0.12;
static const float ToonLightScale = 0.45;
static const float ToonBandSoftness = 0.4;
static const float ToonShadowFloor = 0.18;
static const float ToonHighlightThreshold = 0.8;
static const float ToonHighlightStrength = 0.18;
static const float3 ToonLuma = float3(0.2126, 0.7152, 0.0722);
static const float3 ToonCoolShadow = float3(0.56, 0.68, 1.0);
static const float3 ToonWarmLight = float3(1.0, 0.94, 0.82);
static const float3 ToonSkyFill = float3(0.24, 0.29, 0.4);
float3 ToonAlbedo(float3 color, float model)
{
    if (model != ToonShadingModel) return color;
    float3 paint = pow(saturate(color), ToonAlbedoExponent);
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return saturate(lerp(dot(paint, ToonLuma).xxx, paint, ToonColorSaturation));
}
float3 ToonAmbient(float3 albedo, float3 ambient, float model)
{ return model == ToonShadingModel ? max(ambient, albedo * ToonSkyFill) : ambient; }
float3 ToonResponse(float cosine, float bands)
{
    float angularBand = saturate(cosine) * bands;
    float transition = smoothstep(1.0 - ToonBandSoftness, 1.0, frac(angularBand));
    float level = min(floor(angularBand) + transition, bands - 1.0) / (bands - 1.0);
    return lerp(ToonCoolShadow, ToonWarmLight, level) * lerp(ToonShadowFloor, 1.0, level) * ToonLightScale;
}
float ToonHighlight(float value)
{ return step(ToonHighlightThreshold, value) * ToonHighlightStrength; }

// Integrate a small screen-space footprint in the existing diffuse texture.
// Alpha remains the center sample, preserving foliage and curtain coverage.
static const float ToonTextureRadius = 1.25;
static const int ToonTextureKernelRadius = 1;
static const int ToonTextureKernelWidth = ToonTextureKernelRadius * 2 + 1;
static const float ToonTextureSampleWeight = 1.0 / (ToonTextureKernelWidth * ToonTextureKernelWidth);
float3 ToonTextureColor(Texture2D<float4> image, SamplerState surfaceSampler,
    float2 uv, float2 uvDx, float2 uvDy)
{
    float3 color = 0;
    [unroll] for (int y = -ToonTextureKernelRadius; y <= ToonTextureKernelRadius; ++y)
        [unroll] for (int x = -ToonTextureKernelRadius; x <= ToonTextureKernelRadius; ++x)
            color += image.SampleGrad(surfaceSampler, uv + ToonTextureRadius * (x * uvDx + y * uvDy), uvDx, uvDy).rgb;
    return color * ToonTextureSampleWeight;
}

// Smooth finite support for inverse-square point and rectangle samples.
// Karis' attenuation window; Filament, Punctual lights / Attenuation function.
float LightRangeWindow(float3 toEmitter, float radius)
{
    if (radius <= 0.0f) return 1.0f;
    if (any(abs(toEmitter) >= radius)) return 0.0f;
    float3 relative = toEmitter / radius;
    float distanceRatioSquared = dot(relative, relative);
    float window = saturate(1.0f - distanceRatioSquared * distanceRatioSquared);
    return window * window;
}

// Contact visibility is an independent producer. Combine only the selected
// light's direct visibility; primary filters already include artistic strength.
// This algebra applies that strength once: (1-s) + s * mapRaw * contactRaw.
#if defined(EVANS_CONTACT_RECEIVER)
cbuffer ContactReceiverConstants : register(b6) { uint4 ContactControl; };
Texture2DArray<float> ContactMask : register(t15);
#endif
float ReadContact(float4 position, uint sampleIndex)
{
#if defined(EVANS_CONTACT_RECEIVER)
    if (ContactControl.x != 0) return ContactMask.Load(int4(int2(position.xy), sampleIndex, 0));
#endif
    return 1.0f;
}
float ComposeContact(float primary, float contact)
{
    const float ContactShadowStrength = 0.55f;
    return lerp(1.0f - ContactShadowStrength, primary, contact);
}

// DX11 deferred shading. Geometry evaluates material coverage and normals into
// five RGBA16F attachments. Lighting reconstructs positions from opaque depth;
// it never replays meshes. Transparent surfaces still use standalone forward shaders.
// Shishkovtsov, "Deferred Shading in S.T.A.L.K.E.R.", GPU Gems 2 (2005).
// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-9-deferred-shading-stalker
static const uint MaximumLights = 4u;
static const uint FlatModel = 1u;
static const uint PbrModel = 3u;
static const uint ToonModel = 4u;
static const uint DirectionalLight = 0u;
static const uint PointLight = 1u;
static const float MaximumHalfFloatRadiance = 65504.0f;
static const float NormalLengthEpsilon = 1.0e-12f;
static const float ProjectionEpsilon = 1.0e-7f;
static const float TangentDeterminantEpsilon = 1.0e-10f;
static const float ParallelAxisLimit = 0.999f;
static const float LightDistanceEpsilon = 1.0e-6f;
static const float SrgbThreshold = 0.04045f;
static const float SrgbLinearScale = 12.92f;
static const float SrgbScale = 1.055f;
static const float SrgbOffset = 0.055f;
static const float SrgbGamma = 2.4f;
static const float MinimumRoughness = 0.045f;
static const float UnlitRoughness = -1.0f;
static const float ClearDepth = 1.0f;

struct LightConstants
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};
cbuffer DrawConstants : register(b0)
{
    row_major float4x4 World;
    row_major float4x4 NormalWorld;
    row_major float4x4 ViewProjection;
    row_major float4x4 LightViewProjection;
    float4 PipelineColor;
    float4 MaterialDiffuseColor;
    float4 MaterialAmbientColor;
    float4 MaterialSpecularColor;
    float4 MaterialEmissiveColor;
    float4 MaterialParameters;
    float4 AmbientColorStrength;
    float4 CameraLightCount;
    float4 RenderParameters;
    uint4 TextureFlags;
    LightConstants Lights[MaximumLights];
    float4 ShadingParameters;
};
cbuffer DeferredConstants : register(b1)
{
    row_major float4x4 InverseViewProjection;
    row_major float4x4 DeferredLightViewProjection;
    float4 DeferredCameraLightCount;
    float4 DeferredShadingParameters;
    uint4 DeferredOptions;
    float4 BackgroundColor;
    LightConstants DeferredLights[MaximumLights];
};
Texture2D<float> ShadowMap : register(t0);
Texture2D<float4> DiffuseMap : register(t1);
Texture2D<float4> NormalMap : register(t2);
Texture2D<float4> OpacityMap : register(t3);
#if defined(EVANS_MULTISAMPLE)
Texture2DMS<float4> DiffuseOpacity : register(t4);
Texture2DMS<float4> NormalRoughness : register(t5);
Texture2DMS<float4> SpecularShininess : register(t6);
Texture2DMS<float4> AmbientMetallic : register(t7);
Texture2DMS<float4> EmissionColor : register(t8);
Texture2DMS<float> SceneDepth : register(t9);
#else
Texture2D<float4> DiffuseOpacity : register(t4);
Texture2D<float4> NormalRoughness : register(t5);
Texture2D<float4> SpecularShininess : register(t6);
Texture2D<float4> AmbientMetallic : register(t7);
Texture2D<float4> EmissionColor : register(t8);
Texture2D<float> SceneDepth : register(t9);
#endif
SamplerState SurfaceSampler : register(s0);

float3 SafeNormal(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalLengthEpsilon ? value * rsqrt(lengthSquared) : fallback;
}
float3 DecodeSRGB(float3 color)
{
    float3 low = color / SrgbLinearScale;
    float3 high = pow((color + SrgbOffset) / SrgbScale, SrgbGamma);
    return float3(color.r <= SrgbThreshold ? low.r : high.r,
        color.g <= SrgbThreshold ? low.g : high.g, color.b <= SrgbThreshold ? low.b : high.b);
}
struct MeshVertex
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};
struct SurfaceVertex
{
    float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float2 UV : TEXCOORD2;
};
SurfaceVertex VSMain(MeshVertex input)
{
    SurfaceVertex output;
    float4 world = mul(World, float4(input.Position, 1.0f));
    output.Position = mul(ViewProjection, world);
    output.WorldPosition = world.xyz / max(abs(world.w), ProjectionEpsilon);
    output.Normal = SafeNormal(mul((float3x3)NormalWorld, input.Normal), float3(0.0f, 1.0f, 0.0f));
    output.Color = float4(DecodeSRGB(saturate(input.Color.rgb)), input.Color.a);
    output.UV = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    return output;
}
float3 SurfaceNormal(SurfaceVertex input, bool front)
{
    float3 normal = SafeNormal(input.Normal, float3(0.0f, 1.0f, 0.0f));
    float3 dx = ddx(input.WorldPosition);
    float3 dy = ddy(input.WorldPosition);
    if (TextureFlags.w == FlatModel)
    {
        float3 face = SafeNormal(cross(dx, dy), normal);
        normal = dot(face, normal) < 0.0f ? -face : face;
    }
    if (!front) normal = -normal;
    if (TextureFlags.w == FlatModel || TextureFlags.y == 0u) return normal;
    float2 uvDx = ddx(input.UV);
    float2 uvDy = ddy(input.UV);
    float determinant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    if (abs(determinant) < TangentDeterminantEpsilon) return normal;
    float3 tangent = (dx * uvDy.y - dy * uvDx.y) / determinant;
    float3 sourceBitangent = (-dx * uvDy.x + dy * uvDx.x) / determinant;
    tangent -= normal * dot(normal, tangent);
    float3 helper = abs(normal.y) < ParallelAxisLimit ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = SafeNormal(tangent, SafeNormal(cross(helper, normal), float3(1.0f, 0.0f, 0.0f)));
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0f ? -1.0f : 1.0f;
    float3 bitangent = SafeNormal(cross(normal, tangent) * handedness, float3(0.0f, 0.0f, 1.0f));
    float3 mapped = NormalMap.Sample(SurfaceSampler, input.UV).xyz * 2.0f - 1.0f;
    if (float(TextureFlags.w) == ToonShadingModel) mapped = lerp(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
    return SafeNormal(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
}
struct MaterialTargets
{
    float4 Diffuse : SV_TARGET0;
    float4 Normal : SV_TARGET1;
    float4 Specular : SV_TARGET2;
    float4 Ambient : SV_TARGET3;
    float4 Emission : SV_TARGET4;
};
MaterialTargets PSMain(SurfaceVertex input, bool front : SV_IsFrontFace)
{
    float4 textureColor = TextureFlags.x != 0u ? DiffuseMap.Sample(SurfaceSampler, input.UV) : 1.0f;
    if (float(TextureFlags.w) == ToonShadingModel && TextureFlags.x != 0u)
        textureColor.rgb = ToonTextureColor(DiffuseMap, SurfaceSampler, input.UV, ddx(input.UV), ddy(input.UV));
    float3 surfaceMap = TextureFlags.z != 0 ? OpacityMap.Sample(SurfaceSampler, input.UV).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = ShadingParameters.xy;
    if (TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    float alpha = saturate(input.Color.a * MaterialParameters.y * textureColor.a * opacity);
    if (MaterialParameters.z > 0.0f) clip(alpha - MaterialParameters.z);
    float3 surface = ToonAlbedo(input.Color.rgb * max(PipelineColor.rgb, 0.0f) * max(textureColor.rgb, 0.0f), float(TextureFlags.w));
    float metallic = saturate(roughnessMetallic.y);
    float3 ambient = ToonAmbient(surface * max(MaterialDiffuseColor.rgb, 0.0), surface * max(MaterialAmbientColor.rgb, 0.0f) * max(AmbientColorStrength.rgb, 0.0f)
        * max(AmbientColorStrength.w, 0.0f), float(TextureFlags.w));
    if (TextureFlags.w == PbrModel) ambient *= 1.0f - metallic;
    MaterialTargets output;
    output.Diffuse = float4(clamp(surface * MaterialDiffuseColor.rgb, 0.0f, MaximumHalfFloatRadiance), alpha);
    output.Normal = float4(SurfaceNormal(input, front), clamp(roughnessMetallic.x, MinimumRoughness, 1.0f));
    output.Specular = clamp(float4((TextureFlags.w == 3u ? MaterialSpecularColor.www : MaterialSpecularColor.rgb), max(MaterialParameters.x, 1.0f)), 0.0f, MaximumHalfFloatRadiance);
    output.Ambient = float4(clamp(ambient, 0.0f, MaximumHalfFloatRadiance), metallic);
    // Emission alpha is the material's diffusion coverage, not emissive power.
    output.Emission = float4(clamp(MaterialEmissiveColor.rgb, 0.0f, MaximumHalfFloatRadiance), saturate(MaterialParameters.w));
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (RenderParameters.y > 1.0 && RenderParameters.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (RenderParameters.y > 0.5f)
    {
        output.Normal.w = UnlitRoughness;
        output.Emission.rgb = DecodeSRGB(saturate(PipelineColor.rgb));
    }
    return output;
}

// PCF uses the same depth-map convention and receiver bias as forward. The
// light projection is recomputed from reconstructed world position.

// CSM consumer ABI (b2, t10..t13). Selection uses positive camera eye depth,
// not light depth. Overlap fits both maps; the last interval fades to visibility 1.
static const uint CascadeCapacity = 4u;

static const float CascadeMinimumCosine = 0.15f;
static const float CascadeConstantBias = 0.00035f;
static const float CascadeSlopeBias = 0.00060f;
static const float CascadeMaximumBias = 0.0045f;
static const float CascadeShadowStrength = 0.55f;
static const float CascadeNormalEpsilon = 1.0e-12f;
cbuffer CascadedShadowConstants : register(b2)
{
    row_major float4x4 CascadeLightViewProjection[CascadeCapacity];
    float4 CascadeFarDistances;
    float4 CascadeBlendStarts;
    float4 CascadeEyeDepthPlane;
    float4 CascadeParameters; // near, far, count, map resolution
    float4 ShadowFiltering; // radius (zero means hard), comparison count, reserved
};
Texture2D<float> CascadeMaps[CascadeCapacity] : register(t10);
float CascadeDepth(uint cascade, int2 pixel)
{
    // Shader Model 5 resource arrays require literal resource indices.
    if (cascade == 0u) return CascadeMaps[0].Load(int3(pixel, 0));
    if (cascade == 1u) return CascadeMaps[1].Load(int3(pixel, 0));
    if (cascade == 2u) return CascadeMaps[2].Load(int3(pixel, 0));
    return CascadeMaps[3].Load(int3(pixel, 0));
}
float CascadeVisibility(uint cascade, float3 world, float3 normal, float3 toLight)
{
    float3 projected = mul(CascadeLightViewProjection[cascade], float4(world, 1.0f)).xyz;
    float2 uv = float2(projected.x * 0.5f + 0.5f, 0.5f - projected.y * 0.5f);
    if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f) return 1.0f;
    float cosine = saturate(dot(normal, toLight * rsqrt(max(dot(toLight, toLight), CascadeNormalEpsilon))));
    float slope = sqrt(max(1.0f - cosine * cosine, 0.0f)) / max(cosine, CascadeMinimumCosine);
    float compareDepth = projected.z - min(CascadeMaximumBias, CascadeConstantBias + CascadeSlopeBias * slope);
    if (ShadowFiltering.x == 0.0f)
    {
        int2 pixel = clamp(int2(floor(uv * CascadeParameters.w)), 0, int2(CascadeParameters.ww) - 1);
        return 1.0f - CascadeShadowStrength * (compareDepth > CascadeDepth(cascade, pixel) ? 1.0f : 0.0f);
    }
    float2 texel = uv * CascadeParameters.w - 0.5f;
    int2 basePixel = int2(floor(texel));
    int2 lastPixel = int2(CascadeParameters.ww) - 1;
    float2 fraction = frac(texel);
    float occlusion = 0.0f;
    // Keep the filter loops compact; expanding resource-selection branches at
    // every tap inflates Shader Model 5 bytecode and stalls first-use compilation.
    [loop] for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        [loop] for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = basePixel + int2(x, y);
            float a = compareDepth > CascadeDepth(cascade, clamp(p, 0, lastPixel)) ? 1.0f : 0.0f;
            float b = compareDepth > CascadeDepth(cascade, clamp(p + int2(1, 0), 0, lastPixel)) ? 1.0f : 0.0f;
            float c = compareDepth > CascadeDepth(cascade, clamp(p + int2(0, 1), 0, lastPixel)) ? 1.0f : 0.0f;
            float d = compareDepth > CascadeDepth(cascade, clamp(p + 1, 0, lastPixel)) ? 1.0f : 0.0f;
            occlusion += lerp(lerp(a, b, fraction.x), lerp(c, d, fraction.x), fraction.y);
        }
    return 1.0f - CascadeShadowStrength * occlusion / ShadowFiltering.y;
}
// Virtual-map receiver ABI reuses the mutually exclusive cascade block:
// Far = logical resolution/page side/mips/directory width; Blend = atlas extent/
// columns/directory denominator. Every PCF corner resolves its own ancestor.
static const uint VirtualPageBorder = 1;
static const float VirtualTexelCenter = 0.5f;
float2 VirtualDepth(float2 uv, uint requestedMip)
{
    uint4 logical = uint4(CascadeFarDistances), physical = uint4(CascadeBlendStarts);
    uint side = logical.x / logical.y, base = 0;
    [loop] for (uint mip = 0; mip < logical.z; ++mip)
    {
        if (mip >= requestedMip)
        {
            uint2 pixel = min(uint2(uv * (side * logical.y)), side * logical.y - 1);
            uint2 page = pixel / logical.y;
            uint address = base + page.y * side + page.x;
            uint entry = uint(round(CascadeMaps[1].Load(int3(address % logical.w, address / logical.w, 0)).x * physical.w));
            if (entry != 0)
            {
                uint slot = entry - 1, stride = logical.y + VirtualPageBorder * 2;
                uint2 tile = uint2(slot % physical.z, slot / physical.z) * stride;
                return float2(CascadeMaps[0].Load(int3(tile + VirtualPageBorder + pixel % logical.y, 0)).x, mip);
            }
        }
        base += side * side; side /= 2;
    }
    return float2(1, logical.z - 1);
}
// Compare receiver and caster at the same resolved virtual texel center.
// The actual ancestor mip is essential when detail pages are not resident.
float2 VirtualReceiverGradient(float2 uv, float depth)
{
    float2 x = ddx(uv), y = ddy(uv);
    float zx = ddx(depth), zy = ddy(depth);
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x, x) * dot(y, y), 1.0e-30);
    if (determinant * determinant <= scale * 1.0e-12) return float2(0.0, 0.0);
    return float2(y.y * zx - x.y * zy, x.x * zy - y.x * zx) / determinant;
}
float VirtualReceiverAt(float2 sampleUv, uint actualMip, float2 receiverUv, float receiver, float2 gradient)
{
    float resolution = float(uint(CascadeFarDistances.x) >> actualMip);
    float2 pixel = clamp(floor(sampleUv * resolution), float2(0.0, 0.0), float2(resolution - 1.0, resolution - 1.0));
    float2 center = (pixel + float2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualCompare(int2 pixel, uint mip, float2 receiverUv, float receiver, float2 gradient)
{
    uint resolution = uint(CascadeFarDistances.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0, 0), int2(int(resolution) - 1, int(resolution) - 1))) + float2(0.5, 0.5)) / float(resolution);
    float2 stored = VirtualDepth(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
// Reconstruct one-sided depth slopes on the same geometric surface.
// Fragment-quad derivatives can cross a G-buffer silhouette or intersection.
// Camera depth is affine within a rasterized triangle; lower second difference
// chooses the side that best preserves that plane without using shading normals.
float DeferredReceiverDepth(int2 pixel, uint sampleIndex, uint2 size)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return 1.0;
#if defined(EVANS_MULTISAMPLE)
    return SceneDepth.Load(pixel, sampleIndex);
#else
    return SceneDepth.Load(int3(pixel, 0));
#endif
}
float DeferredReceiverSlope(float center, float negative, float negativeTwo, float positive, float positiveTwo)
{
    bool leftValid = negative < ClearDepth && negativeTwo < ClearDepth;
    bool rightValid = positive < ClearDepth && positiveTwo < ClearDepth;
    if (!leftValid && !rightValid) return 0.0;
    float leftError = abs(center - 2.0 * negative + negativeTwo);
    float rightError = abs(center - 2.0 * positive + positiveTwo);
    return leftValid && (!rightValid || leftError <= rightError) ? center - negative : positive - center;
}
float3 DeferredReceiverLightPoint(float2 coordinate, float depth)
{
    float4 world = mul(InverseViewProjection, float4(coordinate.x * 2.0 - 1.0, 1.0 - coordinate.y * 2.0, depth, 1.0));
    float4 light = mul(CascadeLightViewProjection[0], float4(world.xyz / world.w, 1.0));
    float3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    return float3(projected.xy * float2(0.5, -0.5) + 0.5, projected.z);
}
float3 DeferredVirtualReceiverData(float2 coordinate, float depth, uint sampleIndex)
{
    if (ShadowFiltering.z == 0) return 0.0;
    uint width, height;
#if defined(EVANS_MULTISAMPLE)
    uint sampleCount; SceneDepth.GetDimensions(width, height, sampleCount);
#else
    SceneDepth.GetDimensions(width, height);
#endif
    uint2 size = uint2(width, height); int2 pixel = int2(coordinate * float2(size));
    float dx = DeferredReceiverSlope(depth,
        DeferredReceiverDepth(pixel + int2(-1, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(-2, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(1, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(2, 0), sampleIndex, size));
    float dy = DeferredReceiverSlope(depth,
        DeferredReceiverDepth(pixel + int2(0, -1), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(0, -2), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(0, 1), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(0, 2), sampleIndex, size));
    float3 center = DeferredReceiverLightPoint(coordinate, depth);
    float3 x = DeferredReceiverLightPoint(coordinate + float2(1.0 / width, 0), depth + dx) - center;
    float3 y = DeferredReceiverLightPoint(coordinate + float2(0, 1.0 / height), depth + dy) - center;
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x.xy, x.xy) * dot(y.xy, y.xy), 1.0e-30);
    float2 gradient = determinant * determinant > scale * 1.0e-12
        ? float2(y.y * x.z - x.y * y.z, x.x * y.z - y.x * x.z) / determinant : float2(0, 0);
    float footprint = max(length(x.xy * CascadeEyeDepthPlane.y), length(y.xy * CascadeEyeDepthPlane.z)) * CascadeFarDistances.x;
    float mip = clamp(floor(log2(max(1.0, footprint)) + CascadeEyeDepthPlane.x), 0.0, CascadeFarDistances.z - 1.0);
    return float3(gradient, mip);
}
float VirtualVisibility(float3 position, float bias, float3 receiverData)
{
    float4 light = mul(CascadeLightViewProjection[0], float4(position, 1.0));
    float3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    float2 uv = float2(projected.x, -projected.y) * 0.5 + float2(0.5, 0.5);
    float2 gradient = receiverData.xy;
    uint mip = uint(receiverData.z);
    if (light.w <= CascadeEyeDepthPlane.w || any(uv < 0.0) || any(uv > 1.0) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float2 stored = VirtualDepth(uv, mip);
    float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(CascadeFarDistances.x) >> mip) - float2(0.5, 0.5);
    int2 first = int2(floor(texel)); float2 weight = frac(texel); float visible = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = first + int2(x, y);
            float a = VirtualCompare(p, mip, uv, receiver, gradient);
            float b = VirtualCompare(p + int2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualCompare(p + int2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualCompare(p + int2(1, 1), mip, uv, receiver, gradient);
            visible += lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        }
    return visible / ShadowFiltering.y;
}


float CascadedVisibility(float3 world, float3 normal, float3 toLight, float3 receiverData)
{
    if (ShadowFiltering.z != 0)
    {
        float cosine = saturate(dot(normal, toLight * rsqrt(max(dot(toLight, toLight), CascadeNormalEpsilon))));
        float slope = sqrt(max(1 - cosine * cosine, 0)) / max(cosine, CascadeMinimumCosine);
        float bias = min(CascadeMaximumBias, CascadeConstantBias + CascadeSlopeBias * slope);
        return 1 - CascadeShadowStrength * (1 - VirtualVisibility(world, bias, receiverData));
    }
    float distance = dot(CascadeEyeDepthPlane, float4(world, 1.0f));
    if (distance < CascadeParameters.x || distance > CascadeParameters.y) return 1.0f;
    uint count = uint(CascadeParameters.z);
    [loop] for (uint cascade = 0u; cascade < count; ++cascade)
    {
        if (distance > CascadeFarDistances[cascade]) continue;
        float visibility = CascadeVisibility(cascade, world, normal, toLight);
        float width = CascadeFarDistances[cascade] - CascadeBlendStarts[cascade];
        if (width > 0.0f && distance >= CascadeBlendStarts[cascade])
        {
            float next = cascade + 1u < count ? CascadeVisibility(cascade + 1u, world, normal, toLight) : 1.0f;
            visibility = lerp(visibility, next, saturate((distance - CascadeBlendStarts[cascade]) / width));
        }
        return visibility;
    }
    return 1.0f;
}

float ShadowVisibility(float3 world, float3 normal, float3 toLight, float3 receiverData)
{
    if (DeferredOptions.y == 0u) return 1.0f;
    if (CascadeParameters.z > 0.0f) return CascadedVisibility(world, normal, toLight, receiverData);
    float4 clipPosition = mul(DeferredLightViewProjection, float4(world, 1.0f));
    if (abs(clipPosition.w) < ProjectionEpsilon) return 1.0f;
    float3 projected = clipPosition.xyz / clipPosition.w;
    float2 uv = float2(projected.x * 0.5f + 0.5f, 0.5f - projected.y * 0.5f);
    if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f) return 1.0f;
    const float MinimumCosine = 0.15f;
    const float MaximumBias = 0.0045f;
    const float ConstantBias = 0.00035f;
    const float SlopeBias = 0.00060f;
    const float ShadowStrength = 0.55f;

    float cosine = saturate(dot(normal, SafeNormal(toLight, normal)));
    float slope = sqrt(max(1.0f - cosine * cosine, 0.0f)) / max(cosine, MinimumCosine);
    float compareDepth = projected.z - min(MaximumBias, ConstantBias + SlopeBias * slope);
    uint width, height;
    ShadowMap.GetDimensions(width, height);
    // Hard mode selects one nearest depth texel; PCF averages comparisons.
    if (ShadowFiltering.x == 0.0f)
    {
        int2 pixel = clamp(int2(floor(uv * float2(width, height))), 0, int2(width, height) - 1);
        return 1.0f - ShadowStrength * (compareDepth > ShadowMap.Load(int3(pixel, 0)) ? 1.0f : 0.0f);
    }
    float2 texel = uv * float2(width, height) - 0.5f;
    int2 basePixel = int2(floor(texel));
    float2 fraction = frac(texel);
    float occlusion = 0.0f;
    [loop] for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        [loop] for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = clamp(basePixel + int2(x, y), int2(0, 0), int2(width - 1, height - 1));
            int2 right = min(p + int2(1, 0), int2(width - 1, height - 1));
            int2 below = min(p + int2(0, 1), int2(width - 1, height - 1));
            int2 diagonal = min(p + 1, int2(width - 1, height - 1));
            float a = compareDepth > ShadowMap.Load(int3(p, 0)) ? 1.0f : 0.0f;
            float b = compareDepth > ShadowMap.Load(int3(right, 0)) ? 1.0f : 0.0f;
            float c = compareDepth > ShadowMap.Load(int3(below, 0)) ? 1.0f : 0.0f;
            float d = compareDepth > ShadowMap.Load(int3(diagonal, 0)) ? 1.0f : 0.0f;
            occlusion += lerp(lerp(a, b, fraction.x), lerp(c, d, fraction.x), fraction.y);
        }
    return 1.0f - ShadowStrength * occlusion / ShadowFiltering.y;
}
struct Surface
{
    float3 VirtualReceiverData;
    float3 Position;
    float3 Normal;
    float3 View;
    float3 Diffuse;
    float3 Specular;
    float Roughness;
    float Metallic;
    float Shininess;
};
// Output mode is shared with DeferredOutputMode; no new shader permutation.
static const uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly() { return DeferredOptions.w == DirectDiffuseOutput; }
float3 LightSample(Surface surface, float3 toLight, float3 radiance)
{
    float3 direction = SafeNormal(toLight, surface.Normal);
    float noL = saturate(dot(surface.Normal, direction));
    float3 halfway = SafeNormal(direction + surface.View, surface.Normal);
    if (DeferredOptions.x == PbrModel)
    {
        // GGX, exact separable Smith masking, Schlick Fresnel. See Walter et al.,
        // "Microfacet Models for Refraction through Rough Surfaces" (2007).
        const float Pi = 3.14159265359f;
        const float DielectricReflectance = 0.04f;
        const float BrdfEpsilon = 1.0e-6f;
        const float DistributionEpsilon = 1.0e-12f;
        const float FresnelPower = 5.0f;
        float noV = saturate(dot(surface.Normal, surface.View));
        if (noL <= 0.0f || noV <= 0.0f) return 0.0f;
        float noH = saturate(dot(surface.Normal, halfway));
        float voH = saturate(dot(surface.View, halfway));
        float alpha = surface.Roughness * surface.Roughness;
        float alphaSquared = alpha * alpha;
        float denominator = noH * noH * (alphaSquared - 1.0f) + 1.0f;
        float distribution = alphaSquared / max(Pi * denominator * denominator, DistributionEpsilon);
        float smithV = 2.0f * noV / max(noV + sqrt(alphaSquared + (1.0f - alphaSquared) * noV * noV), BrdfEpsilon);
        float smithL = 2.0f * noL / max(noL + sqrt(alphaSquared + (1.0f - alphaSquared) * noL * noL), BrdfEpsilon);
        float3 f0 = lerp(saturate(surface.Specular), saturate(surface.Diffuse), surface.Metallic);
        float3 fresnel = f0 + (1.0f - f0) * pow(1.0f - voH, FresnelPower);
        float3 diffuse = surface.Diffuse * (1.0f - fresnel) * (1.0f - surface.Metallic) / Pi;
        float3 specular = distribution * smithV * smithL * fresnel / max(4.0f * noV * noL, BrdfEpsilon);
        return radiance * noL * (diffuse + (DirectDiffuseOnly() ? 0.0f : specular));
    }
    float specular = noL > 0.0f ? pow(saturate(dot(surface.Normal, halfway)), surface.Shininess) : 0.0f;
    if (DeferredOptions.x == ToonModel)
    {
        const float MinimumBands = 2.0f;
        const float MaximumBands = 12.0f;

        float bands = clamp(round(DeferredShadingParameters.x), MinimumBands, MaximumBands);
        return radiance * (surface.Diffuse * ToonResponse(noL, bands) + (DirectDiffuseOnly() ? 0.0 : surface.Specular * ToonHighlight(specular)));
    }
    return radiance * (surface.Diffuse * noL + (DirectDiffuseOnly() ? 0.0f : surface.Specular * specular));
}
float3 SceneLight(Surface surface, uint lightIndex, float contactVisibility)
{
    LightConstants light = DeferredLights[lightIndex];
    uint type = (uint)round(light.PositionType.w);
    float3 toLight = type == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - surface.Position;
    float visibility = lightIndex == DeferredOptions.z && light.ColorCastShadows.w > 0.5f
        ? ShadowVisibility(surface.Position, surface.Normal, toLight, surface.VirtualReceiverData) : 1.0f;
    if (lightIndex == DeferredOptions.z && light.ColorCastShadows.w > 0.5f)
        visibility = ComposeContact(visibility, contactVisibility);
    float3 radiance = max(light.ColorCastShadows.rgb, 0.0f) * max(light.DirectionIntensity.w, 0.0f) * visibility;
    if (type == DirectionalLight) return LightSample(surface, toLight, radiance);
    if (type == PointLight) return LightSample(surface, toLight, radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / max(dot(toLight, toLight), LightDistanceEpsilon));
    // Prepared emitters integrate area-weighted radiance; legacy lights retain normalized energy.
    const int SamplesPerAxis = 2;
    const float SampleOffset = 0.25f;
    const float SampleWeight = 0.25f;
    float3 emissionDirection = SafeNormal(light.DirectionIntensity.xyz, float3(0.0f, -1.0f, 0.0f));
    float3 helper = abs(emissionDirection.y) < ParallelAxisLimit ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = SafeNormal(cross(helper, emissionDirection), float3(1.0f, 0.0f, 0.0f));
    float3 bitangent = SafeNormal(cross(emissionDirection, tangent), float3(0.0f, 0.0f, 1.0f));
    float3 result = 0.0f;
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSizePadding.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaSizePadding.w, MinimumEmitterGrid, MaximumEmitterGrid)) : SamplesPerAxis;
    float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSizePadding.x * light.AreaSizePadding.y) : 1.0) / float(emitterGrid * emitterGrid);
    [loop] for (int y = 0; y < emitterGrid; ++y)
        [loop] for (int x = 0; x < emitterGrid; ++x)
        {
            float2 offset = float2(((float(x) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter), ((float(y) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter));
            float3 samplePosition = light.PositionType.xyz + tangent * offset.x * light.AreaSizePadding.x
                + bitangent * offset.y * light.AreaSizePadding.y;
            float3 sampleDirection = samplePosition - surface.Position;
            float distanceSquared = max(dot(sampleDirection, sampleDirection), LightDistanceEpsilon);
            float emitterCosine = max(dot(emissionDirection, -sampleDirection * rsqrt(distanceSquared)), 0.0f);
            result += LightSample(surface, sampleDirection, radiance * emitterSampleWeight * emitterCosine * LightRangeWindow(sampleDirection, light.AreaSizePadding.z) / distanceSquared);
        }
    return result;
}
struct ScreenVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
ScreenVertex VSLighting(uint vertexId : SV_VertexID)
{
    ScreenVertex output;
    output.UV = float2((vertexId << 1u) & 2u, vertexId & 2u);
    output.Position = float4(output.UV.x * 2.0f - 1.0f, 1.0f - output.UV.y * 2.0f, 0.0f, 1.0f);
    return output;
}
#if defined(EVANS_MULTISAMPLE)
// SV_SampleIndex forces sample-frequency lighting, including uncovered samples
// which retain the background. Depth and inverse projection use the same sample
// position; center reconstruction would bend silhouettes on sloped surfaces.
// https://learn.microsoft.com/windows/win32/direct3dhlsl/sm5-object-texture2dms-getsampleposition
float4 PSLighting(ScreenVertex input, uint sampleIndex : SV_SampleIndex) : SV_TARGET
#else
float4 PSLighting(ScreenVertex input) : SV_TARGET
#endif
{
#if defined(EVANS_MULTISAMPLE)
    int2 pixel = int2(input.Position.xy);
    float depth = SceneDepth.Load(pixel, sampleIndex);
    float4 diffuse = DiffuseOpacity.Load(pixel, sampleIndex);
    float4 normal = NormalRoughness.Load(pixel, sampleIndex);
    float4 emission = EmissionColor.Load(pixel, sampleIndex);
    float4 specular = SpecularShininess.Load(pixel, sampleIndex);
    float4 ambient = AmbientMetallic.Load(pixel, sampleIndex);
    uint width, height, count;
    SceneDepth.GetDimensions(width, height, count);
    const float PixelCenterOffset = 0.5f;
    float2 uv = (float2(pixel) + PixelCenterOffset + SceneDepth.GetSamplePosition(sampleIndex)) / float2(width, height);
#else
    int3 texel = int3(int2(input.Position.xy), 0);
    float depth = SceneDepth.Load(texel);
    float4 diffuse = DiffuseOpacity.Load(texel);
    float4 normal = NormalRoughness.Load(texel);
    float4 emission = EmissionColor.Load(texel);
    float4 specular = SpecularShininess.Load(texel);
    float4 ambient = AmbientMetallic.Load(texel);
    float2 uv = input.UV;
#endif
    // A wire overlay intentionally keeps opaque depth. Its edge can cover a
    // background pixel whose depth is still clear, so honor unlit coverage first.
    if (normal.w < 0.0f) return DirectDiffuseOnly() ? 0.0f : float4(emission.rgb, diffuse.a);
    if (depth >= ClearDepth) return DirectDiffuseOnly() ? 0.0f : BackgroundColor;
    float4 world = mul(InverseViewProjection, float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f));
    Surface surface;
    surface.Position = world.xyz / world.w;
    surface.Normal = SafeNormal(normal.xyz, float3(0.0f, 1.0f, 0.0f));
    surface.View = SafeNormal(DeferredCameraLightCount.xyz - surface.Position, surface.Normal);
    surface.Diffuse = diffuse.rgb;
    surface.Specular = specular.rgb;
    surface.Roughness = max(normal.w, MinimumRoughness);
    surface.Shininess = max(specular.w, 1.0f);
    surface.Metallic = saturate(ambient.w);
#if defined(EVANS_MULTISAMPLE)
    const uint contactSample = sampleIndex;
#else
    const uint contactSample = 0u;
#endif
    surface.VirtualReceiverData = DeferredVirtualReceiverData(uv, depth, contactSample);
    float3 result = DirectDiffuseOnly() ? 0.0f : ambient.rgb + emission.rgb;
    uint lightCount = min((uint)round(DeferredCameraLightCount.w), MaximumLights);
    [loop] for (uint light = 0u; light < lightCount; ++light) result += SceneLight(surface, light, ReadContact(input.Position, contactSample));
    float coverage = DirectDiffuseOnly() ? (DeferredOptions.x == PbrModel ? 1.0f - surface.Metallic : 1.0f) : diffuse.a;
    if (DirectDiffuseOnly()) { result *= emission.a; coverage *= emission.a; }
    return float4(clamp(result, 0.0f, MaximumHalfFloatRadiance), coverage);
}
