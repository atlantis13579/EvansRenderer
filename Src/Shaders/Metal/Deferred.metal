#include <metal_stdlib>
using namespace metal;
// Anime environment material palette and diffuse ramp. Native MSL evaluation
// keeps radiance, visibility and alpha separate. Cai, GDC 2021,
// "Genshin Impact: Crafting an Anime Style Open World" motivates the art target.
constant float ToonShadingModel = 4.0;
constant float ToonAlbedoExponent = 0.8;
constant float ToonPaletteSteps = 16.0;
constant float ToonColorSaturation = 1.12;
constant float ToonNormalStrength = 0.12;
constant float ToonLightScale = 0.45;
constant float ToonBandSoftness = 0.4;
constant float ToonShadowFloor = 0.18;
constant float ToonHighlightThreshold = 0.8;
constant float ToonHighlightStrength = 0.18;
constant float3 ToonLuma = float3(0.2126, 0.7152, 0.0722);
constant float3 ToonCoolShadow = float3(0.56, 0.68, 1.0);
constant float3 ToonWarmLight = float3(1.0, 0.94, 0.82);
constant float3 ToonSkyFill = float3(0.24, 0.29, 0.4);
float3 ToonAlbedo(float3 value, float model)
{
    if (model != ToonShadingModel) return value;
    float3 paint = pow(saturate(value), float3(ToonAlbedoExponent));
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return saturate(mix(float3(dot(paint, ToonLuma)), paint, ToonColorSaturation));
}
float3 ToonAmbient(float3 albedo, float3 ambient, float model)
{ return model == ToonShadingModel ? max(ambient, albedo * ToonSkyFill) : ambient; }
float3 ToonResponse(float cosine, float bands)
{
    float angularBand = saturate(cosine) * bands;
    float transition = smoothstep(1.0 - ToonBandSoftness, 1.0, fract(angularBand));
    float level = min(floor(angularBand) + transition, bands - 1.0) / (bands - 1.0);
    return mix(ToonCoolShadow, ToonWarmLight, level) * mix(ToonShadowFloor, 1.0, level) * ToonLightScale;
}
float ToonHighlight(float value)
{ return step(ToonHighlightThreshold, value) * ToonHighlightStrength; }

// Smooth only the texture color. Explicit gradients and the center alpha
// preserve the original material's mip selection and silhouette coverage.
constant float ToonTextureRadius = 1.25;
constant int ToonTextureKernelRadius = 1;
constant int ToonTextureKernelWidth = ToonTextureKernelRadius * 2 + 1;
constant float ToonTextureSampleWeight = 1.0 / (ToonTextureKernelWidth * ToonTextureKernelWidth);
float3 ToonTextureColor(texture2d<float> image, sampler surfaceSampler,
    float2 uv, float2 uvDx, float2 uvDy)
{
    float3 color = float3(0.0);
    for (int row = -ToonTextureKernelRadius; row <= ToonTextureKernelRadius; ++row)
        for (int column = -ToonTextureKernelRadius; column <= ToonTextureKernelRadius; ++column)
            color += image.sample(surfaceSampler, uv + ToonTextureRadius * (column * uvDx + row * uvDy), gradient2d(uvDx, uvDy)).rgb;
    return color * ToonTextureSampleWeight;
}


// Karis' smooth inverse-square window; Filament, Punctual lights.
// The host canonicalizes range: zero is unbounded, positive is finite support.
float LightRangeWindow(float3 offset, float radius)
{
    if (radius <= 0.0f) return 1.0f;
    if (any(abs(offset) >= float3(radius))) return 0.0f;
    float3 normalizedOffset = offset / radius;
    float squaredFraction = dot(normalizedOffset, normalizedOffset);
    float window = max(1.0f - squaredFraction * squaredFraction, 0.0f);
    return window * window;
}

// Contact variants read the matching opaque sample; inactive draws retain their
// ordinary entry points. MSL keeps material varyings at pixel-centre interpolation.
#ifndef EVANS_CONTACT_SHADOWS
#define EVANS_CONTACT_SHADOWS 0
#endif
#ifndef EVANS_CONTACT_MULTISAMPLE
#define EVANS_CONTACT_MULTISAMPLE 0
#endif


// Deferred shading stores material attributes before evaluating scene lights.
// Shishkovtsov, "Deferred Shading in S.T.A.L.K.E.R.", GPU Gems 2 (2005).
// Five RGBA16F images favor readable semantics over packing efficiency. Opaque
// depth reconstructs position; sorted transparency remains a forward pass.
// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-9-deferred-shading-stalker
constant uint MaximumLights = 4;
constant uint FlatModel = 1;
constant uint PbrModel = 3;
constant uint ToonModel = 4;
constant uint DirectionalLight = 0;
constant uint PointLight = 1;
constant float MaximumHalfFloatRadiance = 65504.0;
constant float NormalizeEpsilon = 1.0e-12;
constant float HomogeneousEpsilon = 1.0e-7;
constant float UvDeterminantEpsilon = 1.0e-10;
constant float DistanceEpsilon = 1.0e-6;
constant float DistributionEpsilon = 1.0e-12;
constant float BrdfEpsilon = 1.0e-6;
constant float ParallelAxisThreshold = 0.999;
constant float MinimumRoughness = 0.045;
constant float Pi = 3.141592653589793;
constant float DielectricReflectance = 0.04;
constant float UnlitRoughness = -1.0;
constant float OverlayModeThreshold = 0.5;
constant float OutlineModeThreshold = 1.5;
constant float OutlineWorldScale = 0.01;
constant float ReceiverBias = 0.00035;
constant float ReceiverSlopeBias = 0.00060;
constant float MaximumReceiverBias = 0.0045;
constant float MinimumShadowCosine = 0.15;
constant float ShadowStrength = 0.55;

constant int AreaGridSize = 2;
constant float AreaSampleWeight = 0.25;

struct SceneLight
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};

// These rows match RealtimeDrawConstants, including its unused geometry lights.
struct GeometryConstants
{
    float4 World[4];
    float4 NormalWorld[4];
    float4 ViewProjection[4];
    float4 LightViewProjection[4];
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
    SceneLight Lights[MaximumLights];
    float4 ShadingParameters;
};

struct LightingConstants
{
    float4 InverseViewProjection[4];
    float4 LightViewProjection[4];
    float4 CameraLightCount;
    float4 ShadingParameters;
    uint4 Options;
    float4 BackgroundColor;
    SceneLight Lights[MaximumLights];
};

struct MeshVertex { float Values[12]; };
struct GeometryVaryings
{
    float4 Position [[position]];
    float3 WorldPosition;
    float3 Normal;
    float4 Color;
    float2 TexCoord;
};
struct MaterialOutputs
{
    float4 DiffuseOpacity [[color(0)]];
    float4 NormalRoughness [[color(1)]];
    float4 SpecularShininess [[color(2)]];
    float4 AmbientMetallic [[color(3)]];
    float4 Emission [[color(4)]];
};
struct LightingVaryings { float4 Position [[position]]; };

float4 TransformRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value), dot(rows[2], value), dot(rows[3], value));
}

float3 UnitOr(float3 value, float3 fallback)
{
    float magnitudeSquared = dot(value, value);
    return magnitudeSquared > NormalizeEpsilon ? value * rsqrt(magnitudeSquared) : fallback;
}

float3 DecodeSrgb(float3 value)
{
    const float EncodedThreshold = 0.04045;
    const float LinearSlope = 12.92;
    const float PowerOffset = 0.055;
    const float PowerScale = 1.055;
    const float Exponent = 2.4;
    return select(pow((value + PowerOffset) / PowerScale, Exponent), value / LinearSlope,
        value <= EncodedThreshold);
}

vertex GeometryVaryings DeferredGeometryVertex(device const MeshVertex* vertices [[buffer(0)]],
    constant GeometryConstants& draw [[buffer(1)]], uint vertexId [[vertex_id]])
{
    const device MeshVertex& meshVertex = vertices[vertexId];
    float4 world = TransformRows(draw.World, float4(meshVertex.Values[0], meshVertex.Values[1], meshVertex.Values[2], 1.0));
    float3 normal = UnitOr(TransformRows(draw.NormalWorld,
        float4(meshVertex.Values[3], meshVertex.Values[4], meshVertex.Values[5], 0.0)).xyz, float3(0.0, 1.0, 0.0));
    if (draw.RenderParameters.y > OutlineModeThreshold)
        world.xyz += normal * draw.MaterialParameters.w * OutlineWorldScale;
    GeometryVaryings output;
    output.Position = TransformRows(draw.ViewProjection, world);
    output.WorldPosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    output.Normal = normal;
    output.Color = float4(DecodeSrgb(saturate(float3(meshVertex.Values[6], meshVertex.Values[7], meshVertex.Values[8]))), meshVertex.Values[9]);
    output.TexCoord = float2(meshVertex.Values[10], 1.0 - meshVertex.Values[11]);
    return output;
}

float3 MaterialNormal(GeometryVaryings input, constant GeometryConstants& draw,
    texture2d<float> normalMap, sampler surfaceSampler, bool frontFacing)
{
    float3 reference = UnitOr(input.Normal, float3(0.0, 1.0, 0.0));
    float3 positionDx = dfdx(input.WorldPosition);
    float3 positionDy = dfdy(input.WorldPosition);
    if (draw.TextureFlags.w == FlatModel)
    {
        float3 plane = UnitOr(cross(positionDx, positionDy), reference);
        if (dot(plane, reference) < 0.0) plane = -plane;
        return frontFacing ? plane : -plane;
    }
    float3 normal = frontFacing ? reference : -reference;
    if (draw.TextureFlags.y == 0) return normal;
    float2 uvDx = dfdx(input.TexCoord);
    float2 uvDy = dfdy(input.TexCoord);
    float determinant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    if (abs(determinant) < UvDeterminantEpsilon) return normal;
    float3 tangent = (positionDx * uvDy.y - positionDy * uvDx.y) / determinant;
    float3 sourceBitangent = (-positionDx * uvDy.x + positionDy * uvDx.x) / determinant;
    tangent -= normal * dot(normal, tangent);
    float3 helper = abs(normal.y) < ParallelAxisThreshold ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    tangent = UnitOr(tangent, UnitOr(cross(helper, normal), float3(1.0, 0.0, 0.0)));
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    float3 bitangent = UnitOr(cross(normal, tangent) * handedness, float3(0.0, 0.0, 1.0));
    float3 mapped = normalMap.sample(surfaceSampler, input.TexCoord).xyz * 2.0 - 1.0;
    if (draw.TextureFlags.w == ToonShadingModel) mapped = mix(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
    return UnitOr(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
}

fragment MaterialOutputs DeferredGeometryFragment(GeometryVaryings input [[stage_in]],
    constant GeometryConstants& draw [[buffer(1)]], texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]], texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]], bool frontFacing [[front_facing]])
{
    float4 diffuse = draw.TextureFlags.x != 0 ? diffuseMap.sample(surfaceSampler, input.TexCoord) : float4(1.0);
    if (draw.TextureFlags.w == ToonShadingModel && draw.TextureFlags.x != 0u)
        diffuse.rgb = ToonTextureColor(diffuseMap, surfaceSampler, input.TexCoord, dfdx(input.TexCoord), dfdy(input.TexCoord));
    float3 surfaceMap = draw.TextureFlags.z != 0 ? opacityMap.sample(surfaceSampler, input.TexCoord).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = draw.ShadingParameters.xy;
    if (draw.TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    float alpha = saturate(input.Color.a * draw.MaterialParameters.y * diffuse.a * opacity);
    if (draw.MaterialParameters.z > 0.0 && alpha < draw.MaterialParameters.z) discard_fragment();
    MaterialOutputs output;
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (draw.RenderParameters.y > 1.0 && draw.RenderParameters.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = dfdx(wireBary), wireDy = dfdy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard_fragment();
    }
    if (draw.RenderParameters.y > OverlayModeThreshold)
    {
        output.DiffuseOpacity = float4(0.0, 0.0, 0.0, alpha);
        output.NormalRoughness = float4(0.0, 0.0, 1.0, UnlitRoughness);
        output.SpecularShininess = float4(0.0);
        output.AmbientMetallic = float4(0.0);
        output.Emission = float4(DecodeSrgb(saturate(draw.PipelineColor.rgb)), 0.0);
        return output;
    }
    float3 base = ToonAlbedo(input.Color.rgb * max(draw.PipelineColor.rgb, 0.0) * max(diffuse.rgb, 0.0), draw.TextureFlags.w);
    float metallic = saturate(roughnessMetallic.y);
    float3 ambient = ToonAmbient(base * max(draw.MaterialDiffuseColor.rgb, float3(0.0)), base * max(draw.MaterialAmbientColor.rgb, 0.0) *
        max(draw.AmbientColorStrength.rgb, 0.0) * max(draw.AmbientColorStrength.w, 0.0), draw.TextureFlags.w);
    if (draw.TextureFlags.w == PbrModel) ambient *= 1.0 - metallic;
    output.DiffuseOpacity = float4(clamp(base * max(draw.MaterialDiffuseColor.rgb, 0.0), 0.0, MaximumHalfFloatRadiance), alpha);
    output.NormalRoughness = float4(MaterialNormal(input, draw, normalMap, surfaceSampler, frontFacing),
        clamp(roughnessMetallic.x, MinimumRoughness, 1.0));
    output.SpecularShininess = float4(clamp((draw.TextureFlags.w == 3u ? draw.MaterialSpecularColor.www : draw.MaterialSpecularColor.rgb), 0.0, MaximumHalfFloatRadiance),
        clamp(draw.MaterialParameters.x, 1.0, MaximumHalfFloatRadiance));
    output.AmbientMetallic = float4(clamp(ambient, 0.0, MaximumHalfFloatRadiance), metallic);
    output.Emission = float4(clamp(draw.MaterialEmissiveColor.rgb, 0.0, MaximumHalfFloatRadiance), saturate(draw.MaterialParameters.w));
    return output;
}

vertex LightingVaryings DeferredLightingVertex(uint vertexId [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 point = vertexId == 0 ? float2(-1.0) :
        vertexId == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(point, 0.0, 1.0)};
}

float ShadowVisibility(float3 position, float3 normal, float3 towardLight,
    constant LightingConstants& lighting, depth2d<float, access::read> shadow, float4 filtering)
{
    if (lighting.Options.y == 0) return 1.0;
    float4 lightPosition = TransformRows(lighting.LightViewProjection, float4(position, 1.0));
    if (abs(lightPosition.w) < HomogeneousEpsilon) return 1.0;
    float3 projected = lightPosition.xyz / lightPosition.w;
    float2 uv(projected.x * 0.5 + 0.5, -projected.y * 0.5 + 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float cosine = saturate(dot(normal, UnitOr(towardLight, normal)));
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumShadowCosine);
    float receiver = projected.z - min(MaximumReceiverBias, ReceiverBias + ReceiverSlopeBias * slope);
    int2 extent(shadow.get_width(), shadow.get_height());
    if (filtering.x == 0.0)
    {
        int2 pixel = clamp(int2(floor(uv * float2(extent))), int2(0), extent - 1);
        return 1.0 - ShadowStrength * (receiver > shadow.read(uint2(pixel)) ? 1.0 : 0.0);
    }
    float2 samplePosition = uv * float2(extent) - 0.5;
    int2 base = int2(floor(samplePosition));
    float2 fraction = fract(samplePosition);
    float occlusion = 0.0;
    for (int y = -int(filtering.x); y <= int(filtering.x); ++y)
        for (int x = -int(filtering.x); x <= int(filtering.x); ++x)
        {
            int2 p00 = clamp(base + int2(x, y), int2(0), extent - 1);
            int2 p10 = clamp(p00 + int2(1, 0), int2(0), extent - 1);
            int2 p01 = clamp(p00 + int2(0, 1), int2(0), extent - 1);
            int2 p11 = clamp(p00 + int2(1, 1), int2(0), extent - 1);
            float a = receiver > shadow.read(uint2(p00)) ? 1.0 : 0.0;
            float b = receiver > shadow.read(uint2(p10)) ? 1.0 : 0.0;
            float c = receiver > shadow.read(uint2(p01)) ? 1.0 : 0.0;
            float d = receiver > shadow.read(uint2(p11)) ? 1.0 : 0.0;
            occlusion += mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
        }
    return 1.0 - ShadowStrength * occlusion / filtering.y;
}

// GGX distribution, exact separable Smith masking, and Schlick Fresnel for PBR;
// the other modes preserve the forward path's Blinn-Phong or quantized response.
// This output selects only direct diffuse, with the same shadowed light integral.
constant uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly(constant LightingConstants& lighting)
{ return lighting.Options.w == DirectDiffuseOutput; }
float3 EvaluateLight(float3 towardLight, float3 radiance, float3 normal, float3 view,
    float3 albedo, float4 specularShininess, float roughness, float metallic,
    constant LightingConstants& lighting)
{
    float lengthSquared = dot(towardLight, towardLight);
    if (lengthSquared <= NormalizeEpsilon) return float3(0.0);
    float3 direction = towardLight * rsqrt(lengthSquared);
    float nDotL = saturate(dot(normal, direction));
    if (nDotL <= 0.0 && lighting.Options.x != ToonModel) return float3(0.0);
    float3 halfway = UnitOr(direction + view, normal);
    float nDotH = saturate(dot(normal, halfway));
    if (lighting.Options.x == PbrModel)
    {
        float nDotV = saturate(dot(normal, view));
        if (nDotV <= 0.0) return float3(0.0);
        float vDotH = saturate(dot(view, halfway));
        float alpha = roughness * roughness;
        float alphaSquared = alpha * alpha;
        float denominator = nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
        float distribution = alphaSquared / max(Pi * denominator * denominator, DistributionEpsilon);
        float masking = 2.0 * nDotV / (nDotV + sqrt(alphaSquared + (1.0 - alphaSquared) * nDotV * nDotV));
        float shadowing = 2.0 * nDotL / (nDotL + sqrt(alphaSquared + (1.0 - alphaSquared) * nDotL * nDotL));
        float3 f0 = mix(saturate(specularShininess.rgb), saturate(albedo), metallic);
        const float SchlickExponent = 5.0;
        const float MicrofacetNormalization = 4.0;
        float3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vDotH, SchlickExponent);
        float3 specular = fresnel * distribution * masking * shadowing / max(MicrofacetNormalization * nDotV * nDotL, BrdfEpsilon);
        if (DirectDiffuseOnly(lighting)) return radiance * nDotL * albedo * (1.0 - fresnel) * (1.0 - metallic) / Pi;
        return radiance * nDotL * (albedo * (1.0 - fresnel) * (1.0 - metallic) / Pi + specular);
    }
    float diffuse = nDotL;
    float specular = pow(nDotH, max(specularShininess.w, 1.0));
    if (lighting.Options.x == ToonModel)
    {
        const float MinimumToonBands = 2.0;

        float bands = max(lighting.ShadingParameters.x, MinimumToonBands);
        return radiance * (albedo * ToonResponse(nDotL, bands) + (DirectDiffuseOnly(lighting) ? float3(0.0) : specularShininess.rgb * ToonHighlight(specular)));
    }
    if (DirectDiffuseOnly(lighting)) return radiance * albedo * diffuse;
    return radiance * (albedo * diffuse + specularShininess.rgb * specular);
}

// Each light matrix is four CPU row vectors; the full block matches the
// renderer's 336-byte shadow receiver contract. Depth textures use top-left coordinates.
constant uint CsmCascadeLimit = 4;
constant float CsmProjectionEpsilon = 1.0e-7;
constant float CsmDirectionEpsilon = 1.0e-12;
constant float CsmOverlapEpsilon = 1.0e-6;
constant float CsmMinimumCosine = 0.15;
constant float CsmConstantBias = 0.00035;
constant float CsmSlopeBias = 0.00060;
constant float CsmMaximumBias = 0.0045;
constant float CsmShadowStrength = 0.55;

struct CascadeConstants
{
    float4 LightRows[CsmCascadeLimit][4];
    float4 FarDepths;
    float4 BlendDepths;
    float4 EyeDepthPlane;
    float4 Parameters; // near, far, count, map resolution
    float4 Filtering; // radius (zero means hard), comparison count, reserved
};

float CsmCompare(depth2d<float, access::read> map, int2 pixel, int2 maximumPixel, float receiver)
{
    return receiver > map.read(uint2(clamp(pixel, int2(0), maximumPixel))) ? 1.0 : 0.0;
}
float CsmFilter(depth2d<float, access::read> map, constant float4* rows, float3 position, float bias, float4 filtering)
{
    float4 world(position, 1.0);
    float4 light(dot(rows[0], world), dot(rows[1], world), dot(rows[2], world), dot(rows[3], world));
    if (abs(light.w) < CsmProjectionEpsilon) return 1.0;
    float3 ndc = light.xyz / light.w;
    float2 uv(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (!all(uv >= 0.0) || !all(uv <= 1.0) || ndc.z < 0.0 || ndc.z > 1.0) return 1.0;
    uint2 extent(map.get_width(), map.get_height());
    if (filtering.x == 0.0)
        return 1.0 - CsmShadowStrength * CsmCompare(map, int2(floor(uv * float2(extent))), int2(extent) - 1, ndc.z - bias);
    float2 texel = uv * float2(extent) - 0.5;
    int2 base = int2(floor(texel));
    int2 maximumPixel = int2(extent) - 1;
    float2 fraction = fract(texel);
    float receiver = ndc.z - bias;
    float blocked = 0.0;
    // Read and compare D32 texels before bilinear interpolation; a filtered
    // depth value would create false occluders across discontinuities.
    for (int row = -int(filtering.x); row <= int(filtering.x); ++row)
        for (int column = -int(filtering.x); column <= int(filtering.x); ++column)
        {
            int2 pixel = base + int2(column, row);
            float upperLeft = CsmCompare(map, pixel, maximumPixel, receiver);
            float upperRight = CsmCompare(map, pixel + int2(1, 0), maximumPixel, receiver);
            float lowerLeft = CsmCompare(map, pixel + int2(0, 1), maximumPixel, receiver);
            float lowerRight = CsmCompare(map, pixel + int2(1, 1), maximumPixel, receiver);
            blocked += mix(mix(upperLeft, upperRight, fraction.x), mix(lowerLeft, lowerRight, fraction.x), fraction.y);
        }
    return 1.0 - CsmShadowStrength * blocked * (1.0 / filtering.y);
}
float CsmSample(uint index, float3 position, float bias, constant CascadeConstants& cascades,
    depth2d<float, access::read> map0, depth2d<float, access::read> map1,
    depth2d<float, access::read> map2, depth2d<float, access::read> map3)
{
    // Ordinary resource arguments keep this independent of argument-buffer tiers.
    if (index == 0) return CsmFilter(map0, cascades.LightRows[0], position, bias, cascades.Filtering);
    if (index == 1) return CsmFilter(map1, cascades.LightRows[1], position, bias, cascades.Filtering);
    if (index == 2) return CsmFilter(map2, cascades.LightRows[2], position, bias, cascades.Filtering);
    return CsmFilter(map3, cascades.LightRows[3], position, bias, cascades.Filtering);
}
// The cascade argument block also describes the exclusive virtual hierarchy.
// FarDepths = logical extent; BlendDepths = atlas extent and directory scale.
// Both images are point-read D32 resources, including the normalized directory.
constant uint VirtualTileBorder = 1;
constant float VirtualSampleCenter = 0.5;
float2 VirtualDepth(float2 uv, uint firstMip, constant CascadeConstants& frame,
    depth2d<float, access::read> atlas, depth2d<float, access::read> directory)
{
    uint4 layout(frame.FarDepths), physical(frame.BlendDepths);
    uint side = layout.x / layout.y, offset = 0;
    for (uint mip = 0; mip < layout.z; ++mip)
    {
        if (mip >= firstMip)
        {
            uint resolution = side * layout.y;
            uint2 pixel = min(uint2(uv * resolution), uint2(resolution - 1)), page = pixel / layout.y;
            uint address = offset + page.y * side + page.x;
            uint entry = uint(round(directory.read(uint2(address % layout.w, address / layout.w)) * physical.w));
            if (entry != 0)
            {
                uint slot = entry - 1;
                uint2 tile(slot % physical.z, slot / physical.z);
                uint2 at = tile * (layout.y + 2 * VirtualTileBorder) + VirtualTileBorder + pixel % layout.y;
                return float2(atlas.read(at), float(mip));
            }
        }
        offset += side * side; side /= 2;
    }
    return float2(1, layout.z - 1);
}
float VirtualCompare(int2 pixel, uint mip, float receiver, constant CascadeConstants& frame,
    depth2d<float, access::read> atlas, depth2d<float, access::read> directory)
{
    uint resolution = uint(frame.FarDepths.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0), int2(resolution - 1))) + VirtualSampleCenter) / resolution;
    return receiver <= VirtualDepth(uv, mip, frame, atlas, directory).x ? 1.0 : 0.0;
}
float VirtualVisibility(float3 position, float bias, constant CascadeConstants& frame,
    depth2d<float, access::read> atlas, depth2d<float, access::read> directory)
{
    float4 world(position, 1);
    constant float4* rows = frame.LightRows[0];
    float4 clip(dot(rows[0], world), dot(rows[1], world), dot(rows[2], world), dot(rows[3], world));
    float3 projected = clip.xyz / max(clip.w, frame.EyeDepthPlane.w);
    float2 uv(projected.x * VirtualSampleCenter + VirtualSampleCenter, VirtualSampleCenter - projected.y * VirtualSampleCenter);
    float footprint = max(length(dfdx(uv)) * frame.EyeDepthPlane.y, length(dfdy(uv)) * frame.EyeDepthPlane.z) * frame.FarDepths.x;
    uint mip = uint(clamp(floor(log2(max(1.0, footprint)) + frame.EyeDepthPlane.x), 0.0, frame.FarDepths.z - 1.0));
    if (clip.w <= frame.EyeDepthPlane.w || !all(uv >= 0.0) || !all(uv <= 1.0) || projected.z < 0 || projected.z > 1) return 1;
    float2 stored = VirtualDepth(uv, mip, frame, atlas, directory); float receiver = projected.z - bias;
    if (frame.Filtering.x == 0) return receiver <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(frame.FarDepths.x) >> mip) - VirtualSampleCenter;
    int2 base(floor(texel)); float2 fraction = fract(texel); float visible = 0;
    for (int y = -int(frame.Filtering.x); y <= int(frame.Filtering.x); ++y)
        for (int x = -int(frame.Filtering.x); x <= int(frame.Filtering.x); ++x)
        {
            int2 p = base + int2(x, y);
            float a = VirtualCompare(p, mip, receiver, frame, atlas, directory);
            float b = VirtualCompare(p + int2(1, 0), mip, receiver, frame, atlas, directory);
            float c = VirtualCompare(p + int2(0, 1), mip, receiver, frame, atlas, directory);
            float d = VirtualCompare(p + 1, mip, receiver, frame, atlas, directory);
            visible += mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
        }
    return visible / frame.Filtering.y;
}


float CsmVisibility(float3 position, float3 normal, float3 directionToLight, constant CascadeConstants& cascades,
    depth2d<float, access::read> map0, depth2d<float, access::read> map1,
    depth2d<float, access::read> map2, depth2d<float, access::read> map3)
{
    if (cascades.Filtering.z != 0)
    {
        float cosine = saturate(dot(normal, directionToLight * rsqrt(max(dot(directionToLight, directionToLight), CsmDirectionEpsilon))));
        float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CsmMinimumCosine);
        float bias = min(CsmMaximumBias, CsmConstantBias + CsmSlopeBias * slope);
        return 1.0 - CsmShadowStrength * (1.0 - VirtualVisibility(position, bias, cascades, map0, map1));
    }
    float depth = dot(cascades.EyeDepthPlane, float4(position, 1.0));
    uint count = uint(cascades.Parameters.z);
    if (count == 0 || depth < cascades.Parameters.x || depth > cascades.Parameters.y) return 1.0;
    uint selected = 0;
    for (uint index = 0; index + 1 < CsmCascadeLimit; ++index)
        if (index + 1 < count && depth > cascades.FarDepths[index]) selected = index + 1;
    float lengthSquared = dot(directionToLight, directionToLight);
    float3 direction = lengthSquared > CsmDirectionEpsilon ? directionToLight * rsqrt(lengthSquared) : normal;
    float cosine = saturate(dot(normal, direction));
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CsmMinimumCosine);
    float bias = min(CsmConstantBias + CsmSlopeBias * slope, CsmMaximumBias);
    float visibility = CsmSample(selected, position, bias, cascades, map0, map1, map2, map3);
    float overlap = cascades.FarDepths[selected] - cascades.BlendDepths[selected];
    if (overlap > CsmOverlapEpsilon && depth > cascades.BlendDepths[selected])
    {
        float nextVisibility = selected + 1 < count
            ? CsmSample(selected + 1, position, bias, cascades, map0, map1, map2, map3) : 1.0;
        visibility = mix(visibility, nextVisibility, saturate((depth - cascades.BlendDepths[selected]) / overlap));
    }
    return visibility;
}

float4 ShadeDeferredSurface(float2 rasterPosition, float2 extent, float4 diffuse,
    float4 materialNormal, float depth, float4 specular, float4 ambient, float4 emission,
    constant LightingConstants& lighting, constant CascadeConstants& cascades,
    depth2d<float, access::read> shadowDepth, depth2d<float, access::read> cascade0,
    depth2d<float, access::read> cascade1, depth2d<float, access::read> cascade2,
    depth2d<float, access::read> cascade3, float contactVisibility = 1.0)
{
    // Wire overlays can intentionally leave opaque depth untouched.
    if (materialNormal.w < 0.0) return DirectDiffuseOnly(lighting) ? float4(0.0) : float4(emission.rgb, diffuse.a);
    if (depth >= 1.0) return DirectDiffuseOnly(lighting) ? float4(0.0) : lighting.BackgroundColor;
    float2 uv = rasterPosition / extent;
    float4 homogeneous = TransformRows(lighting.InverseViewProjection,
        float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0));
    if (abs(homogeneous.w) < HomogeneousEpsilon) return DirectDiffuseOnly(lighting) ? float4(0.0) : lighting.BackgroundColor;
    float3 position = homogeneous.xyz / homogeneous.w;
    float3 normal = UnitOr(materialNormal.xyz, float3(0.0, 1.0, 0.0));
    float3 view = UnitOr(lighting.CameraLightCount.xyz - position, normal);
    float roughness = clamp(materialNormal.w, MinimumRoughness, 1.0);
    float metallic = saturate(ambient.w);
    float3 result = DirectDiffuseOnly(lighting) ? float3(0.0) : ambient.rgb + emission.rgb;
    uint count = min(uint(lighting.CameraLightCount.w + 0.5), MaximumLights);
    for (uint index = 0; index < count; ++index)
    {
        SceneLight light = lighting.Lights[index];
        uint type = uint(light.PositionType.w + 0.5);
        float3 towardLight = type == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - position;
        float visibility = 1.0;
        if (index == lighting.Options.z && light.ColorCastShadows.w > OverlayModeThreshold &&
            lighting.Options.y != 0)
            visibility = cascades.Parameters.z > 0.0
                ? CsmVisibility(position, normal, towardLight, cascades, cascade0, cascade1, cascade2, cascade3)
                : ShadowVisibility(position, normal, towardLight, lighting, shadowDepth, cascades.Filtering);
        if (index == lighting.Options.z && light.ColorCastShadows.w > OverlayModeThreshold && contactVisibility < 1.0)
            visibility = mix(1.0 - ShadowStrength, visibility, contactVisibility);
        float3 radiance = max(light.ColorCastShadows.rgb, 0.0) * max(light.DirectionIntensity.w, 0.0) * visibility;
        if (type == DirectionalLight || type == PointLight)
        {
            if (type == PointLight) radiance *= LightRangeWindow(towardLight, light.AreaSizePadding.z) / max(dot(towardLight, towardLight), DistanceEpsilon);
            result += EvaluateLight(towardLight, radiance, normal, view, diffuse.rgb, specular, roughness, metallic, lighting);
            continue;
        }
        float3 emitterDirection = UnitOr(light.DirectionIntensity.xyz, float3(0.0, -1.0, 0.0));
        float3 helper = abs(emitterDirection.y) < ParallelAxisThreshold ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        float3 tangent = UnitOr(cross(helper, emitterDirection), float3(1.0, 0.0, 0.0));
        float3 bitangent = UnitOr(cross(emitterDirection, tangent), float3(0.0, 0.0, 1.0));
        // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
        const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
        const float EmitterStratumCenter = 0.5;
        bool physicalEmitter = light.AreaSizePadding.w > 0.0;
        int emitterGrid = physicalEmitter ? int(clamp(light.AreaSizePadding.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AreaGridSize;
        float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSizePadding.x * light.AreaSizePadding.y) : 1.0) / float(emitterGrid * emitterGrid);
        for (int row = 0; row < emitterGrid; ++row)
            for (int column = 0; column < emitterGrid; ++column)
            {
                float offsetX = ((float(column) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter);
                float offsetY = ((float(row) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter);
                float3 samplePosition = light.PositionType.xyz + tangent * offsetX * light.AreaSizePadding.x +
                    bitangent * offsetY * light.AreaSizePadding.y;
                float3 delta = samplePosition - position;
                float distanceSquared = max(dot(delta, delta), DistanceEpsilon);
                float3 direction = delta * rsqrt(distanceSquared);
                float emitterCosine = max(dot(emitterDirection, -direction), 0.0);
                result += EvaluateLight(direction, radiance * (emitterSampleWeight * emitterCosine * LightRangeWindow(delta, light.AreaSizePadding.z) / distanceSquared),
                    normal, view, diffuse.rgb, specular, roughness, metallic, lighting);
            }
    }
    float coverage = DirectDiffuseOnly(lighting) ? (lighting.Options.x == PbrModel ? 1.0 - metallic : 1.0) : diffuse.a;
    if (DirectDiffuseOnly(lighting)) { result *= emission.a; coverage *= emission.a; }
    return float4(clamp(result, 0.0, MaximumHalfFloatRadiance), coverage);
}

fragment float4 DeferredLightingFragment(LightingVaryings input [[stage_in]],
#if EVANS_CONTACT_SHADOWS
    texture2d_array<float, access::read> contactMask [[texture(15)]],
#endif
    constant LightingConstants& lighting [[buffer(0)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    depth2d<float, access::read> cascade0 [[texture(10)]],
    depth2d<float, access::read> cascade1 [[texture(11)]],
    depth2d<float, access::read> cascade2 [[texture(12)]],
    depth2d<float, access::read> cascade3 [[texture(13)]],
    texture2d<float, access::read> diffuseOpacity [[texture(0)]],
    texture2d<float, access::read> normalRoughness [[texture(1)]],
    texture2d<float, access::read> specularShininess [[texture(2)]],
    texture2d<float, access::read> ambientMetallic [[texture(3)]],
    texture2d<float, access::read> emission [[texture(4)]],
    depth2d<float, access::read> sceneDepth [[texture(5)]],
    depth2d<float, access::read> shadowDepth [[texture(6)]])
{
    // Material texels and opaque depth describe the same visible surface.
    uint2 pixel = uint2(input.Position.xy);
    float contactVisibility = 1.0;
#if EVANS_CONTACT_SHADOWS
    contactVisibility = contactMask.read(uint2(input.Position.xy), 0).x;
#endif
    return ShadeDeferredSurface(input.Position.xy, float2(sceneDepth.get_width(), sceneDepth.get_height()),
        diffuseOpacity.read(pixel), normalRoughness.read(pixel), sceneDepth.read(pixel),
        specularShininess.read(pixel), ambientMetallic.read(pixel), emission.read(pixel),
        lighting, cascades, shadowDepth, cascade0, cascade1, cascade2, cascade3, contactVisibility);
}

fragment float4 DeferredLightingSampleFragment(LightingVaryings input [[stage_in]], uint sampleIndex [[sample_id]],
#if EVANS_CONTACT_SHADOWS
    texture2d_array<float, access::read> contactMask [[texture(15)]],
#endif
    constant LightingConstants& lighting [[buffer(0)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    depth2d<float, access::read> cascade0 [[texture(10)]],
    depth2d<float, access::read> cascade1 [[texture(11)]],
    depth2d<float, access::read> cascade2 [[texture(12)]],
    depth2d<float, access::read> cascade3 [[texture(13)]],
    texture2d_ms<float, access::read> diffuseOpacity [[texture(0)]],
    texture2d_ms<float, access::read> normalRoughness [[texture(1)]],
    texture2d_ms<float, access::read> specularShininess [[texture(2)]],
    texture2d_ms<float, access::read> ambientMetallic [[texture(3)]],
    texture2d_ms<float, access::read> emission [[texture(4)]],
    depth2d_ms<float, access::read> sceneDepth [[texture(5)]],
    depth2d<float, access::read> shadowDepth [[texture(6)]])
{
    uint2 pixel = uint2(input.Position.xy);
    // sample_id selects the stored material and the output sample. Ordinary
    // stage inputs stay at pixel center, so reconstruct with the real offset.
    float2 rasterPosition = float2(pixel) + get_sample_position(sampleIndex);
    float contactVisibility = 1.0;
#if EVANS_CONTACT_SHADOWS
    contactVisibility = contactMask.read(uint2(input.Position.xy), sampleIndex).x;
#endif
    return ShadeDeferredSurface(rasterPosition, float2(sceneDepth.get_width(), sceneDepth.get_height()),
        diffuseOpacity.read(pixel, sampleIndex), normalRoughness.read(pixel, sampleIndex), sceneDepth.read(pixel, sampleIndex),
        specularShininess.read(pixel, sampleIndex), ambientMetallic.read(pixel, sampleIndex), emission.read(pixel, sampleIndex),
        lighting, cascades, shadowDepth, cascade0, cascade1, cascade2, cascade3, contactVisibility);
}
