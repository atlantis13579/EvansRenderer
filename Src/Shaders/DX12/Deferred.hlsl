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

// Finite support is part of the light model, shared by every emitter sample.
// Karis' smooth attenuation window; Filament, Punctual lights.
float LightRangeWindow(float3 displacement, float range)
{
    if (range == 0.0f) return 1.0f;
    float3 magnitude = abs(displacement);
    if (max(magnitude.x, max(magnitude.y, magnitude.z)) >= range) return 0.0f;
    float3 scaled = displacement / range;
    float squaredRatio = dot(scaled, scaled);
    float fade = max(1.0f - squaredRatio * squaredRatio, 0.0f);
    return fade * fade;
}

// The independent contact producer supplies raw visibility for this raster sample.
// Only the selected direct light consumes it. Primary map filters already apply
// artistic strength, so the composition below applies that strength just once.
#if defined(EVANS_CONTACT_RECEIVER)
cbuffer ContactReceiver : register(b7) { uint ContactEnabled; };
Texture2DArray<float> ContactImage : register(t15);
#endif
float ReadContactVisibility(float4 pixel, uint sampleIndex)
{
#if defined(EVANS_CONTACT_RECEIVER)
    if (ContactEnabled != 0) return ContactImage.Load(int4(int2(pixel.xy), sampleIndex, 0));
#endif
    return 1.0f;
}
float CombineContactVisibility(float primary, float contact, float strength)
{
    return lerp(1.0f - strength, primary, contact);
}

// Native DX12 deferred pipeline: material MRTs followed by a vertexless light pass.
// Five wide RGBA16F targets keep diffuse, normal, specular, ambient and emission
// inspectable. Depth is D32/R32 with D3D's [0,1] clip range. Sorted transparency
// uses the forward pipeline after this resolve. No other shader is included.
// Shishkovtsov, "Deferred Shading in S.T.A.L.K.E.R.", GPU Gems 2 (2005).
// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-9-deferred-shading-stalker
static const uint LightCapacity = 4u;
static const uint FlatShading = 1u;
static const uint PhysicalShading = 3u;
static const uint CelShading = 4u;
static const uint DirectionalKind = 0u;
static const uint PointKind = 1u;
static const float HalfFloatLimit = 65504.0f;
static const float NormalEpsilonSquared = 1.0e-12f;
static const float HomogeneousEpsilon = 1.0e-7f;
static const float UvDeterminantEpsilon = 1.0e-10f;
static const float DistanceEpsilonSquared = 1.0e-6f;
static const float ParallelAxisCosine = 0.999f;
static const float RoughnessFloor = 0.045f;
static const float WireCoverageMarker = -1.0f;
static const float EmptyDepth = 1.0f;
static const float SrgbBreakpoint = 0.04045f;
static const float SrgbLinearDivisor = 12.92f;
static const float SrgbOffset = 0.055f;
static const float SrgbDivisor = 1.055f;
static const float SrgbExponent = 2.4f;

struct PackedLight
{
    float4 PositionKind;
    float4 DirectionIntensity;
    float4 ColorShadow;
    float4 RectangleSize;
};
// b0 has the renderer's 688-byte draw ABI; b1 has the 448-byte lighting ABI.
// The geometry and lighting root signatures expose only their respective CBV.
cbuffer GeometryUniforms : register(b0)
{
    row_major float4x4 ObjectToWorld;
    row_major float4x4 NormalToWorld;
    row_major float4x4 CameraProjection;
    row_major float4x4 ShadowProjectionUnused;
    float4 DrawColor;
    float4 DiffuseReflectance;
    float4 AmbientReflectance;
    float4 SpecularReflectance;
    float4 EmissionRadiance;
    float4 MaterialControls; // shininess, opacity, cutout threshold, diffusion coverage / outline width
    float4 AmbientIllumination;
    float4 DrawCameraUnused;
    float4 DrawControls; // shadow, wire/outline mode, shadow index, pass
    uint4 MaterialTextures; // diffuse, normal, opacity, shading model
    PackedLight DrawLightsUnused[LightCapacity];
    float4 SurfaceControls; // roughness, metallic, cel bands, HDR attachment
};
cbuffer LightingUniforms : register(b1)
{
    row_major float4x4 ClipToWorld;
    row_major float4x4 WorldToShadow;
    float4 EyeAndLightCount;
    float4 LightingControls; // cel bands
    uint4 LightingOptions; // model, shadow enabled, shadow light index
    float4 ClearRadiance;
    PackedLight SceneLights[LightCapacity];
};
Texture2D<float> ShadowDepth : register(t0);
Texture2D<float4> AlbedoTexture : register(t1);
Texture2D<float4> TangentNormalTexture : register(t2);
Texture2D<float4> OpacityTexture : register(t3);
#if defined(EVANS_MULTISAMPLE)
Texture2DMS<float4> MaterialDiffuse : register(t4);
Texture2DMS<float4> MaterialNormal : register(t5);
Texture2DMS<float4> MaterialSpecular : register(t6);
Texture2DMS<float4> MaterialAmbient : register(t7);
Texture2DMS<float4> MaterialEmission : register(t8);
Texture2DMS<float> OpaqueDepth : register(t9);
#else
Texture2D<float4> MaterialDiffuse : register(t4);
Texture2D<float4> MaterialNormal : register(t5);
Texture2D<float4> MaterialSpecular : register(t6);
Texture2D<float4> MaterialAmbient : register(t7);
Texture2D<float4> MaterialEmission : register(t8);
Texture2D<float> OpaqueDepth : register(t9);
#endif
SamplerState MaterialSampler : register(s0);

float3 UnitVector(float3 direction, float3 fallback)
{
    float squaredLength = dot(direction, direction);
    return squaredLength > NormalEpsilonSquared ? direction * rsqrt(squaredLength) : fallback;
}
float3 LinearVertexColor(float3 encoded)
{
    float3 linearBranch = encoded / SrgbLinearDivisor;
    float3 powerBranch = pow((encoded + SrgbOffset) / SrgbDivisor, SrgbExponent);
    return float3(encoded.r <= SrgbBreakpoint ? linearBranch.r : powerBranch.r,
        encoded.g <= SrgbBreakpoint ? linearBranch.g : powerBranch.g,
        encoded.b <= SrgbBreakpoint ? linearBranch.b : powerBranch.b);
}
struct VertexInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 UV : TEXCOORD0;
};
struct GeometryVaryings
{
    float4 RasterPosition : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    float2 TextureCoordinate : TEXCOORD2;
    float4 LinearColor : COLOR0;
};
GeometryVaryings VSMain(VertexInput vertex)
{
    GeometryVaryings result;
    float4 world = mul(ObjectToWorld, float4(vertex.Position, 1.0f));
    result.RasterPosition = mul(CameraProjection, world);
    result.WorldPosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    result.WorldNormal = UnitVector(mul((float3x3)NormalToWorld, vertex.Normal), float3(0.0f, 1.0f, 0.0f));
    result.TextureCoordinate = float2(vertex.UV.x, 1.0f - vertex.UV.y);
    result.LinearColor = float4(LinearVertexColor(saturate(vertex.Color.rgb)), vertex.Color.a);
    return result;
}
float3 EvaluateMaterialNormal(GeometryVaryings fragment, bool frontFacing)
{
    float3 normal = UnitVector(fragment.WorldNormal, float3(0.0f, 1.0f, 0.0f));
    float3 positionDx = ddx(fragment.WorldPosition);
    float3 positionDy = ddy(fragment.WorldPosition);
    if (MaterialTextures.w == FlatShading)
    {
        float3 geometric = UnitVector(cross(positionDx, positionDy), normal);
        normal = dot(geometric, normal) < 0.0f ? -geometric : geometric;
    }
    normal *= frontFacing ? 1.0f : -1.0f;
    if (MaterialTextures.y == 0u || MaterialTextures.w == FlatShading) return normal;
    float2 uvDx = ddx(fragment.TextureCoordinate);
    float2 uvDy = ddy(fragment.TextureCoordinate);
    float determinant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    if (abs(determinant) < UvDeterminantEpsilon) return normal;
    float3 tangent = (positionDx * uvDy.y - positionDy * uvDx.y) / determinant;
    float3 sourceBitangent = (positionDy * uvDx.x - positionDx * uvDy.x) / determinant;
    float3 helper = abs(normal.y) < ParallelAxisCosine ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = UnitVector(tangent - normal * dot(normal, tangent),
        UnitVector(cross(helper, normal), float3(1.0f, 0.0f, 0.0f)));
    float orientation = dot(cross(normal, tangent), sourceBitangent) < 0.0f ? -1.0f : 1.0f;
    float3 bitangent = UnitVector(cross(normal, tangent) * orientation, float3(0.0f, 0.0f, 1.0f));
    float3 mapped = TangentNormalTexture.Sample(MaterialSampler, fragment.TextureCoordinate).xyz * 2.0f - 1.0f;
    if (float(MaterialTextures.w) == ToonShadingModel) mapped = lerp(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
    return UnitVector(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
}
struct GBufferOutput
{
    float4 DiffuseOpacity : SV_TARGET0;
    float4 NormalRoughness : SV_TARGET1;
    float4 SpecularShininess : SV_TARGET2;
    float4 AmbientMetallic : SV_TARGET3;
    float4 Emission : SV_TARGET4;
};
GBufferOutput PSMain(GeometryVaryings fragment, bool frontFacing : SV_IsFrontFace)
{
    float4 albedo = MaterialTextures.x != 0u ? AlbedoTexture.Sample(MaterialSampler, fragment.TextureCoordinate) : 1.0f;
    if (float(MaterialTextures.w) == ToonShadingModel && MaterialTextures.x != 0u)
        albedo.rgb = ToonTextureColor(AlbedoTexture, MaterialSampler, fragment.TextureCoordinate, ddx(fragment.TextureCoordinate), ddy(fragment.TextureCoordinate));
    float3 surfaceMap = MaterialTextures.z != 0 ? OpacityTexture.Sample(MaterialSampler, fragment.TextureCoordinate).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = SurfaceControls.xy;
    if (MaterialTextures.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacityMap = surfaceMap.r;
    float opacity = saturate(fragment.LinearColor.a * MaterialControls.y * albedo.a * opacityMap);
    if (MaterialControls.z > 0.0f) clip(opacity - MaterialControls.z);
    float3 base = ToonAlbedo(fragment.LinearColor.rgb * max(DrawColor.rgb, 0.0f) * max(albedo.rgb, 0.0f), float(MaterialTextures.w));
    float metallic = saturate(roughnessMetallic.y);
    float3 ambient = ToonAmbient(base * max(DiffuseReflectance.rgb, 0.0), base * max(AmbientReflectance.rgb, 0.0f)
        * max(AmbientIllumination.rgb, 0.0f) * max(AmbientIllumination.w, 0.0f), float(MaterialTextures.w));
    if (MaterialTextures.w == PhysicalShading) ambient *= 1.0f - metallic;
    GBufferOutput result;
    result.DiffuseOpacity = float4(clamp(base * DiffuseReflectance.rgb, 0.0f, HalfFloatLimit), opacity);
    result.NormalRoughness = float4(EvaluateMaterialNormal(fragment, frontFacing), clamp(roughnessMetallic.x, RoughnessFloor, 1.0f));
    result.SpecularShininess = clamp(float4((MaterialTextures.w == 3u ? SpecularReflectance.www : SpecularReflectance.rgb), max(MaterialControls.x, 1.0f)), 0.0f, HalfFloatLimit);
    result.AmbientMetallic = float4(clamp(ambient, 0.0f, HalfFloatLimit), metallic);
    result.Emission = float4(clamp(EmissionRadiance.rgb, 0.0f, HalfFloatLimit), saturate(MaterialControls.w));
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (DrawControls.y > 1.0 && DrawControls.y < 1.5)
    {
        float3 wireBary = fragment.LinearColor.rgb;
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (DrawControls.y > 0.5f)
    {
        result.NormalRoughness.w = WireCoverageMarker;
        result.Emission.rgb = LinearVertexColor(saturate(DrawColor.rgb));
    }
    return result;
}

// Manual bilinear PCF matches the forward receiver bias and 3x3 footprint.
// Geometry writes no shadows into the G-buffer: each selected light evaluates it.

// CSM: positive eye-depth intervals choose one map, or two in their overlap.
// The terminal interval blends toward an unshadowed receiver. Separate resource
// declarations match independent DX12 descriptor tables at t10 through t13.
static const uint SplitCapacity = 4u;

static const float ShadowOpacity = 0.55f;
static const float SplitDarkening = ShadowOpacity;
static const float SplitBiasBase = 0.00035f;
static const float SplitBiasSlope = 0.00060f;
static const float SplitBiasLimit = 0.0045f;
static const float SplitCosineFloor = 0.15f;
static const float SplitVectorEpsilon = 1.0e-12f;
cbuffer CascadeUniforms : register(b2)
{
    row_major float4x4 SplitToLight[SplitCapacity];
    float4 SplitEndDistances;
    float4 SplitFadeDistances;
    float4 CameraDistancePlane;
    float4 SplitRange; // near, far, count, map resolution
    float4 ShadowFiltering; // radius (zero means hard), comparison count, reserved
};
Texture2D<float> SplitDepth0 : register(t10);
Texture2D<float> SplitDepth1 : register(t11);
Texture2D<float> SplitDepth2 : register(t12);
Texture2D<float> SplitDepth3 : register(t13);
float SplitDepthAt(uint index, int2 coordinate)
{
    if (index == 0u) return SplitDepth0.Load(int3(coordinate, 0));
    if (index == 1u) return SplitDepth1.Load(int3(coordinate, 0));
    if (index == 2u) return SplitDepth2.Load(int3(coordinate, 0));
    return SplitDepth3.Load(int3(coordinate, 0));
}
float SplitComparison(uint index, int2 coordinate, float receiver)
{
    int2 limit = int2(SplitRange.ww) - 1;
    return receiver > SplitDepthAt(index, clamp(coordinate, 0, limit)) ? 1.0f : 0.0f;
}
float SplitVisibility(uint index, float3 position, float bias)
{
    // Fitting produces affine light matrices with homogeneous W equal to one.
    float3 light = mul(SplitToLight[index], float4(position, 1.0f)).xyz;
    float2 texcoord = float2(light.x + 1.0f, 1.0f - light.y) * 0.5f;
    if (light.z < 0.0f || light.z > 1.0f || any(texcoord < 0.0f) || any(texcoord > 1.0f)) return 1.0f;
    if (ShadowFiltering.x == 0.0f)
        return 1.0f - SplitDarkening * SplitComparison(index, int2(floor(texcoord * SplitRange.w)), light.z - bias);
    float2 texel = texcoord * SplitRange.w - 0.5f;
    int2 origin = int2(floor(texel));
    float2 interpolation = frac(texel);
    float receiver = light.z - bias;
    float occluded = 0.0f;
    // Compact loops avoid expanding four resource-selection branches per tap.
    [loop] for (int row = -int(ShadowFiltering.x); row <= int(ShadowFiltering.x); ++row)
        [loop] for (int column = -int(ShadowFiltering.x); column <= int(ShadowFiltering.x); ++column)
        {
            int2 pixel = origin + int2(column, row);
            float top = lerp(SplitComparison(index, pixel, receiver),
                SplitComparison(index, pixel + int2(1, 0), receiver), interpolation.x);
            float bottom = lerp(SplitComparison(index, pixel + int2(0, 1), receiver),
                SplitComparison(index, pixel + 1, receiver), interpolation.x);
            occluded += lerp(top, bottom, interpolation.y);
        }
    return 1.0f - occluded * (1.0f / ShadowFiltering.y) * SplitDarkening;
}
// Virtual shadows share the cascade receiver slots. End distances describe the
// logical hierarchy; fade distances describe the physical atlas and directory.
// Point reads plus per-corner indirection keep unrelated physical tiles apart.
static const uint VirtualGutter = 1;
static const float VirtualPixelCenter = 0.5f;
float2 VirtualStoredDepth(float2 coordinate, uint minimumLevel)
{
    uint4 hierarchy = uint4(SplitEndDistances), atlas = uint4(SplitFadeDistances);
    uint rowPages = hierarchy.x / hierarchy.y, levelOffset = 0;
    [loop] for (uint level = 0; level < hierarchy.z; ++level)
    {
        if (level >= minimumLevel)
        {
            uint resolution = rowPages * hierarchy.y;
            uint2 texel = min(uint2(coordinate * resolution), resolution - 1);
            uint2 page = texel / hierarchy.y;
            uint entryIndex = levelOffset + page.x + page.y * rowPages;
            int2 tablePixel = int2(entryIndex % hierarchy.w, entryIndex / hierarchy.w);
            uint encodedSlot = uint(round(SplitDepth1.Load(int3(tablePixel, 0)).x * atlas.w));
            if (encodedSlot > 0)
            {
                uint slot = encodedSlot - 1;
                uint2 origin = uint2(slot % atlas.z, slot / atlas.z) * (hierarchy.y + 2 * VirtualGutter);
                uint2 location = origin + VirtualGutter + texel % hierarchy.y;
                return float2(SplitDepth0.Load(int3(location, 0)).x, level);
            }
        }
        levelOffset += rowPages * rowPages; rowPages /= 2;
    }
    return float2(1, hierarchy.z - 1);
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
    float resolution = float(uint(SplitEndDistances.x) >> actualMip);
    float2 pixel = clamp(floor(sampleUv * resolution), float2(0.0, 0.0), float2(resolution - 1.0, resolution - 1.0));
    float2 center = (pixel + float2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualComparison(int2 pixel, uint mip, float2 receiverUv, float receiver, float2 gradient)
{
    uint resolution = uint(SplitEndDistances.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0, 0), int2(int(resolution) - 1, int(resolution) - 1))) + float2(0.5, 0.5)) / float(resolution);
    float2 stored = VirtualStoredDepth(uv, mip);
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
    return OpaqueDepth.Load(pixel, sampleIndex);
#else
    return OpaqueDepth.Load(int3(pixel, 0));
#endif
}
float DeferredReceiverSlope(float center, float negative, float negativeTwo, float positive, float positiveTwo)
{
    bool leftValid = negative < EmptyDepth && negativeTwo < EmptyDepth;
    bool rightValid = positive < EmptyDepth && positiveTwo < EmptyDepth;
    if (!leftValid && !rightValid) return 0.0;
    float leftError = abs(center - 2.0 * negative + negativeTwo);
    float rightError = abs(center - 2.0 * positive + positiveTwo);
    return leftValid && (!rightValid || leftError <= rightError) ? center - negative : positive - center;
}
float3 DeferredReceiverLightPoint(float2 coordinate, float depth)
{
    float4 world = mul(ClipToWorld, float4(coordinate.x * 2.0 - 1.0, 1.0 - coordinate.y * 2.0, depth, 1.0));
    float4 light = mul(SplitToLight[0], float4(world.xyz / world.w, 1.0));
    float3 projected = light.xyz / max(light.w, CameraDistancePlane.w);
    return float3(projected.xy * float2(0.5, -0.5) + 0.5, projected.z);
}
float3 DeferredVirtualReceiverData(float2 coordinate, float depth, uint sampleIndex)
{
    if (ShadowFiltering.z == 0) return 0.0;
    uint width, height;
#if defined(EVANS_MULTISAMPLE)
    uint sampleCount; OpaqueDepth.GetDimensions(width, height, sampleCount);
#else
    OpaqueDepth.GetDimensions(width, height);
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
    float footprint = max(length(x.xy * CameraDistancePlane.y), length(y.xy * CameraDistancePlane.z)) * SplitEndDistances.x;
    float mip = clamp(floor(log2(max(1.0, footprint)) + CameraDistancePlane.x), 0.0, SplitEndDistances.z - 1.0);
    return float3(gradient, mip);
}
float VirtualVisibility(float3 position, float bias, float3 receiverData)
{
    float4 light = mul(SplitToLight[0], float4(position, 1.0));
    float3 projected = light.xyz / max(light.w, CameraDistancePlane.w);
    float2 uv = float2(projected.x, -projected.y) * 0.5 + float2(0.5, 0.5);
    float2 gradient = receiverData.xy;
    uint mip = uint(receiverData.z);
    if (light.w <= CameraDistancePlane.w || any(uv < 0.0) || any(uv > 1.0) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float2 stored = VirtualStoredDepth(uv, mip);
    float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(SplitEndDistances.x) >> mip) - float2(0.5, 0.5);
    int2 first = int2(floor(texel)); float2 weight = frac(texel); float visible = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = first + int2(x, y);
            float a = VirtualComparison(p, mip, uv, receiver, gradient);
            float b = VirtualComparison(p + int2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualComparison(p + int2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualComparison(p + int2(1, 1), mip, uv, receiver, gradient);
            visible += lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        }
    return visible / ShadowFiltering.y;
}


float CascadedReceiverVisibility(float3 position, float3 normal, float3 towardLight, float3 receiverData)
{
    if (ShadowFiltering.z != 0)
    {
        float cosine = saturate(dot(normal, towardLight * rsqrt(max(dot(towardLight, towardLight), SplitVectorEpsilon))));
        float slope = sqrt(max(1 - cosine * cosine, 0)) / max(cosine, SplitCosineFloor);
        float bias = min(SplitBiasLimit, SplitBiasBase + SplitBiasSlope * slope);
        return 1 - SplitDarkening * (1 - VirtualVisibility(position, bias, receiverData));
    }
    float eyeDistance = dot(CameraDistancePlane, float4(position, 1.0f));
    if (eyeDistance < SplitRange.x || eyeDistance > SplitRange.y) return 1.0f;
    float cosine = saturate(dot(normal, towardLight * rsqrt(max(dot(towardLight, towardLight), SplitVectorEpsilon))));
    float slope = sqrt(max(1.0f - cosine * cosine, 0.0f)) / max(cosine, SplitCosineFloor);
    float bias = min(SplitBiasLimit, SplitBiasBase + SplitBiasSlope * slope);
    uint count = uint(SplitRange.z);
    [loop] for (uint index = 0u; index < count; ++index)
    {
        if (eyeDistance > SplitEndDistances[index]) continue;
        float visibility = SplitVisibility(index, position, bias);
        float overlap = SplitEndDistances[index] - SplitFadeDistances[index];
        if (overlap > 0.0f && eyeDistance >= SplitFadeDistances[index])
        {
            float nextVisibility = 1.0f;
            if (index + 1u < count) nextVisibility = SplitVisibility(index + 1u, position, bias);
            float weight = saturate((eyeDistance - SplitFadeDistances[index]) / overlap);
            visibility = lerp(visibility, nextVisibility, weight);
        }
        return visibility;
    }
    return 1.0f;
}

float ReceiverVisibility(float3 position, float3 normal, float3 lightDirection, float3 receiverData)
{
    if (LightingOptions.y == 0u) return 1.0f;
    if (SplitRange.z > 0.0f) return CascadedReceiverVisibility(position, normal, lightDirection, receiverData);
    float4 projected = mul(WorldToShadow, float4(position, 1.0f));
    if (abs(projected.w) < HomogeneousEpsilon) return 1.0f;
    float3 ndc = projected.xyz / projected.w;
    float2 uv = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    if (any(uv < 0.0f) || any(uv > 1.0f) || ndc.z < 0.0f || ndc.z > 1.0f) return 1.0f;
    const float BiasFloor = 0.00035f;
    const float BiasSlope = 0.00060f;
    const float BiasCeiling = 0.0045f;
    const float SlopeCosineFloor = 0.15f;

    float cosine = saturate(dot(normal, UnitVector(lightDirection, normal)));
    float tangent = sqrt(max(1.0f - cosine * cosine, 0.0f)) / max(cosine, SlopeCosineFloor);
    float receiver = ndc.z - min(BiasFloor + BiasSlope * tangent, BiasCeiling);
    uint width, height;
    ShadowDepth.GetDimensions(width, height);
    int2 maximumPixel = int2(width - 1, height - 1);
    // Hard shadows bypass the bilinear comparison kernel.
    if (ShadowFiltering.x == 0.0f)
    {
        int2 pixel = clamp(int2(floor(uv * float2(width, height))), 0, int2(width, height) - 1);
        return 1.0f - ShadowOpacity * (receiver > ShadowDepth.Load(int3(pixel, 0)) ? 1.0f : 0.0f);
    }
    float2 samplePosition = uv * float2(width, height) - 0.5f;
    int2 origin = int2(floor(samplePosition));
    float2 blend = frac(samplePosition);
    float occluded = 0.0f;
    [loop] for (int row = -int(ShadowFiltering.x); row <= int(ShadowFiltering.x); ++row)
        [loop] for (int column = -int(ShadowFiltering.x); column <= int(ShadowFiltering.x); ++column)
        {
            int2 pixel = clamp(origin + int2(column, row), int2(0, 0), maximumPixel);
            float topLeft = receiver > ShadowDepth.Load(int3(pixel, 0)) ? 1.0f : 0.0f;
            float topRight = receiver > ShadowDepth.Load(int3(min(pixel + int2(1, 0), maximumPixel), 0)) ? 1.0f : 0.0f;
            float bottomLeft = receiver > ShadowDepth.Load(int3(min(pixel + int2(0, 1), maximumPixel), 0)) ? 1.0f : 0.0f;
            float bottomRight = receiver > ShadowDepth.Load(int3(min(pixel + 1, maximumPixel), 0)) ? 1.0f : 0.0f;
            occluded += lerp(lerp(topLeft, topRight, blend.x), lerp(bottomLeft, bottomRight, blend.x), blend.y);
        }
    return 1.0f - occluded * ShadowOpacity / ShadowFiltering.y;
}
struct LitMaterial
{
    float3 VirtualReceiverData;
    float3 Position;
    float3 Normal;
    float3 ViewDirection;
    float3 Diffuse;
    float3 Specular;
    float Roughness;
    float Shininess;
    float Metallic;
};
// Output mode is shared with DeferredOutputMode; no new shader permutation.
static const uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly() { return LightingOptions.w == DirectDiffuseOutput; }
float3 EvaluateReflectance(LitMaterial material, float3 toLight, float3 irradiance)
{
    float3 light = UnitVector(toLight, material.Normal);
    float3 halfDirection = UnitVector(light + material.ViewDirection, material.Normal);
    float nl = saturate(dot(material.Normal, light));
    float nh = saturate(dot(material.Normal, halfDirection));
    if (LightingOptions.x == PhysicalShading)
    {
        // GGX with separable exact Smith G1 and Schlick Fresnel. Walter et al.,
        // "Microfacet Models for Refraction through Rough Surfaces" (2007).
        const float Pi = 3.14159265359f;
        const float DielectricF0 = 0.04f;
        const float FresnelExponent = 5.0f;
        const float BrdfDenominatorFloor = 1.0e-6f;
        const float DistributionDenominatorFloor = 1.0e-12f;
        float nv = saturate(dot(material.Normal, material.ViewDirection));
        if (nl <= 0.0f || nv <= 0.0f) return 0.0f;
        float vh = saturate(dot(material.ViewDirection, halfDirection));
        float alpha = material.Roughness * material.Roughness;
        float alphaSquared = alpha * alpha;
        float denominator = nh * nh * (alphaSquared - 1.0f) + 1.0f;
        float distribution = alphaSquared / max(Pi * denominator * denominator, DistributionDenominatorFloor);
        float maskView = 2.0f * nv / max(nv + sqrt(alphaSquared + (1.0f - alphaSquared) * nv * nv), BrdfDenominatorFloor);
        float maskLight = 2.0f * nl / max(nl + sqrt(alphaSquared + (1.0f - alphaSquared) * nl * nl), BrdfDenominatorFloor);
        float3 f0 = lerp(saturate(material.Specular), saturate(material.Diffuse), material.Metallic);
        float3 fresnel = f0 + (1.0f - f0) * pow(1.0f - vh, FresnelExponent);
        float3 diffuse = material.Diffuse * (1.0f - fresnel) * (1.0f - material.Metallic) / Pi;
        float3 specular = distribution * maskView * maskLight * fresnel / max(4.0f * nv * nl, BrdfDenominatorFloor);
        return irradiance * nl * (diffuse + (DirectDiffuseOnly() ? 0.0f : specular));
    }
    float highlight = nl > 0.0f ? pow(nh, material.Shininess) : 0.0f;
    if (LightingOptions.x == CelShading)
    {
        const float BandCountMinimum = 2.0f;
        const float BandCountMaximum = 12.0f;

        float bands = clamp(round(LightingControls.x), BandCountMinimum, BandCountMaximum);
        return irradiance * (material.Diffuse * ToonResponse(nl, bands) + (DirectDiffuseOnly() ? 0.0 : material.Specular * ToonHighlight(highlight)));
    }
    return irradiance * (material.Diffuse * nl + (DirectDiffuseOnly() ? 0.0f : material.Specular * highlight));
}
float3 EvaluateLight(LitMaterial material, uint index, float contactVisibility)
{
    PackedLight source = SceneLights[index];
    uint kind = (uint)round(source.PositionKind.w);
    float3 direction = kind == DirectionalKind ? source.DirectionIntensity.xyz : source.PositionKind.xyz - material.Position;
    float visibility = index == LightingOptions.z && source.ColorShadow.w > 0.5f
        ? ReceiverVisibility(material.Position, material.Normal, direction, material.VirtualReceiverData) : 1.0f;
    if (index == LightingOptions.z && source.ColorShadow.w > 0.5f)
        visibility = CombineContactVisibility(visibility, contactVisibility, ShadowOpacity);
    float3 radiance = max(source.ColorShadow.rgb, 0.0f) * max(source.DirectionIntensity.w, 0.0f) * visibility;
    if (kind == DirectionalKind) return EvaluateReflectance(material, direction, radiance);
    if (kind == PointKind) return EvaluateReflectance(material, direction, radiance * LightRangeWindow(direction, source.RectangleSize.z) / max(dot(direction, direction), DistanceEpsilonSquared));
    // Prepared rectangle samples integrate emitted radiance over their physical area.
    const int QuadratureWidth = 2;
    const float QuadratureOffset = 0.25f;
    const float QuadratureWeight = 0.25f;
    float3 axis = UnitVector(source.DirectionIntensity.xyz, float3(0.0f, -1.0f, 0.0f));
    float3 helper = abs(axis.y) < ParallelAxisCosine ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 horizontal = UnitVector(cross(helper, axis), float3(1.0f, 0.0f, 0.0f));
    float3 vertical = UnitVector(cross(axis, horizontal), float3(0.0f, 0.0f, 1.0f));
    float3 lighting = 0.0f;
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = source.RectangleSize.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(source.RectangleSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : QuadratureWidth;
    float emitterSampleWeight = (physicalEmitter ? abs(source.RectangleSize.x * source.RectangleSize.y) : 1.0) / float(emitterGrid * emitterGrid);
    [loop] for (int y = 0; y < emitterGrid; ++y)
        [loop] for (int x = 0; x < emitterGrid; ++x)
        {
            float2 offset = float2(((float(x) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter), ((float(y) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter));
            float3 samplePosition = source.PositionKind.xyz + horizontal * offset.x * source.RectangleSize.x
                + vertical * offset.y * source.RectangleSize.y;
            float3 toSample = samplePosition - material.Position;
            float distanceSquared = max(dot(toSample, toSample), DistanceEpsilonSquared);
            float emissionCosine = max(dot(axis, -toSample * rsqrt(distanceSquared)), 0.0f);
            lighting += EvaluateReflectance(material, toSample, radiance * emitterSampleWeight * emissionCosine * LightRangeWindow(toSample, source.RectangleSize.z) / distanceSquared);
        }
    return lighting;
}
struct LightingVaryings { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
LightingVaryings VSLighting(uint corner : SV_VertexID)
{
    LightingVaryings result;
    result.UV = float2((corner << 1u) & 2u, corner & 2u);
    result.Position = float4(result.UV.x * 2.0f - 1.0f, 1.0f - result.UV.y * 2.0f, 0.0f, 1.0f);
    return result;
}
#if defined(EVANS_MULTISAMPLE)
// Deferred lighting evaluates every stored sample, including uncovered background.
// Use the hardware sample offset for position reconstruction; the G-buffer and
// depth must describe the same subpixel ray.
float4 PSLighting(LightingVaryings fragment, uint coverageSample : SV_SampleIndex) : SV_TARGET
#else
float4 PSLighting(LightingVaryings fragment) : SV_TARGET
#endif
{
#if defined(EVANS_MULTISAMPLE)
    int2 location = int2(fragment.Position.xy);
    float4 normalRoughness = MaterialNormal.Load(location, coverageSample);
    float4 diffuseOpacity = MaterialDiffuse.Load(location, coverageSample);
    float4 emission = MaterialEmission.Load(location, coverageSample);
    float depth = OpaqueDepth.Load(location, coverageSample);
    float4 specularShininess = MaterialSpecular.Load(location, coverageSample);
    float4 ambientMetallic = MaterialAmbient.Load(location, coverageSample);
    uint width, height, storedSamples;
    OpaqueDepth.GetDimensions(width, height, storedSamples);
    const float PixelCenter = 0.5f;
    float2 coordinate = (float2(location) + PixelCenter + OpaqueDepth.GetSamplePosition(coverageSample)) / float2(width, height);
#else
    const uint coverageSample = 0;
    int3 pixel = int3(int2(fragment.Position.xy), 0);
    float4 normalRoughness = MaterialNormal.Load(pixel);
    float4 diffuseOpacity = MaterialDiffuse.Load(pixel);
    float4 emission = MaterialEmission.Load(pixel);
    float depth = OpaqueDepth.Load(pixel);
    float4 specularShininess = MaterialSpecular.Load(pixel);
    float4 ambientMetallic = MaterialAmbient.Load(pixel);
    float2 coordinate = fragment.UV;
#endif
    // Wire overlays do not write depth; their coverage may extend onto clear
    // depth pixels. Test their material marker before rejecting empty depth.
    if (normalRoughness.w < 0.0f) return DirectDiffuseOnly() ? 0.0f : float4(emission.rgb, diffuseOpacity.a);
    if (depth >= EmptyDepth) return DirectDiffuseOnly() ? 0.0f : ClearRadiance;
    float4 world = mul(ClipToWorld, float4(coordinate.x * 2.0f - 1.0f, 1.0f - coordinate.y * 2.0f, depth, 1.0f));
    LitMaterial material;
    material.Position = world.xyz / world.w;
    material.VirtualReceiverData = DeferredVirtualReceiverData(coordinate, depth, coverageSample);
    material.Normal = UnitVector(normalRoughness.xyz, float3(0.0f, 1.0f, 0.0f));
    material.ViewDirection = UnitVector(EyeAndLightCount.xyz - material.Position, material.Normal);
    material.Diffuse = diffuseOpacity.rgb;
    material.Specular = specularShininess.rgb;
    material.Roughness = max(normalRoughness.w, RoughnessFloor);
    material.Shininess = max(specularShininess.w, 1.0f);
    material.Metallic = saturate(ambientMetallic.w);
    float3 radiance = DirectDiffuseOnly() ? 0.0f : ambientMetallic.rgb + emission.rgb;
    uint count = min((uint)round(EyeAndLightCount.w), LightCapacity);
    [loop] for (uint lightIndex = 0u; lightIndex < count; ++lightIndex) radiance += EvaluateLight(material, lightIndex, ReadContactVisibility(fragment.Position, coverageSample));
    float coverage = DirectDiffuseOnly() ? (LightingOptions.x == PhysicalShading ? 1.0f - material.Metallic : 1.0f) : diffuseOpacity.a;
    if (DirectDiffuseOnly()) { radiance *= emission.a; coverage *= emission.a; }
    return float4(clamp(radiance, 0.0f, HalfFloatLimit), coverage);
}
