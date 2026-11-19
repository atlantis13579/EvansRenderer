#version 450
#ifndef EVANS_VERTEX_STAGE
// Low-frequency Lumen translucency. This native GLSL consumer reconstructs the
// nine real SH terms after clamped trilinear interpolation of the froxel grid.
layout(set=3,binding=4,std140,row_major) uniform LumenTranslucency
{
    mat4 LumenVolumeCamera,LumenVolumeInverse;
    uvec4 LumenVolumeCells;
    vec4 LumenVolumeDepth,LumenVolumeExtent;
};
layout(set=3,binding=5) uniform sampler2D LumenVolumeSH;
const float LumenDielectricReflectance=0.04;
vec3 LumenTranslucentIrradiance(vec3 position,vec3 normal)
{
    if (LumenVolumeCells.w==0u) return vec3(0);
    vec4 projected=LumenVolumeCamera*vec4(position,1);
    if (projected.w<=LumenVolumeExtent.z) return vec3(0);
    vec3 clip=projected.xyz/projected.w;
    if (any(greaterThan(abs(clip.xy),vec2(1))) || clip.z<0 || clip.z>1) return vec3(0);
    vec4 startH=LumenVolumeInverse*vec4(clip.xy,0,1),stopH=LumenVolumeInverse*vec4(clip.xy,1,1);
    if (min(abs(startH.w),abs(stopH.w))<=LumenVolumeExtent.z) return vec3(0);
    vec3 start=startH.xyz/startH.w;
    float span=min(length(stopH.xyz/stopH.w-start),LumenVolumeDepth.x),distanceToStart=length(position-start);
    if (span<=LumenVolumeExtent.z || distanceToStart>span) return vec3(0);
    vec2 uv=vec2(clip.x+1,1-clip.y)*0.5;
    vec3 grid=vec3(uv*vec2(LumenVolumeCells.xy),pow(clamp(distanceToStart/span,0,1),1/LumenVolumeDepth.y)*float(LumenVolumeCells.z))-0.5;
    grid=clamp(grid,vec3(0),vec3(LumenVolumeCells.xyz)-1);
    float first=floor(grid.z),second=min(first+1,float(LumenVolumeCells.z-1u));
    vec3 n=normal*inversesqrt(max(dot(normal,normal),LumenVolumeExtent.w));
    const float ShPi=3.14159265358979323846;
    const int Coefficients=9;
    float basis[Coefficients]=float[](sqrt(1/(4*ShPi)),sqrt(3/(4*ShPi))*n.y,sqrt(3/(4*ShPi))*n.z,sqrt(3/(4*ShPi))*n.x,
        sqrt(15/(4*ShPi))*n.x*n.y,sqrt(15/(4*ShPi))*n.y*n.z,sqrt(5/(16*ShPi))*(3*n.z*n.z-1),
        sqrt(15/(4*ShPi))*n.x*n.z,sqrt(15/(16*ShPi))*(n.x*n.x-n.y*n.y));
    vec3 incoming=vec3(0);

    const uint VolumeVisibilityBit=2u;
    if ((LumenVolumeCells.w&VolumeVisibilityBit)!=0u)
    {
        uvec3 base=uvec3(floor(grid)); vec3 fraction=fract(grid);
        // Evaluate certificates before interpolating SH. Rejected corners keep
        // their original kernel mass at zero, preventing light across surfaces.
        for (uint corner=0u;corner<8u;++corner)
        {
            uvec3 bit=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),p=min(base+bit,LumenVolumeCells.xyz-1u);
            vec3 weights=mix(1.0-fraction,fraction,notEqual(bit,uvec3(0u))); float weight=weights.x*weights.y*weights.z; if (weight<=0.0) continue;
            uint row=p.z*LumenVolumeCells.y+p.y; float support=texelFetch(LumenVolumeSH,ivec2(p.x,row),0).a; if (support<=0.0) continue;
            vec2 clipXY=vec2(2.0*(float(p.x)+0.5)/float(LumenVolumeCells.x)-1.0,1.0-2.0*(float(p.y)+0.5)/float(LumenVolumeCells.y));
            vec4 a=LumenVolumeInverse*vec4(clipXY,0,1),b=LumenVolumeInverse*vec4(clipXY,1,1);
            if (min(abs(a.w),abs(b.w))<=LumenVolumeExtent.z) continue;
            vec3 begin=a.xyz/a.w,axis=b.xyz/b.w-begin; float span=length(axis);
            if (isnan(span) || isinf(span) || span<=LumenVolumeExtent.z) continue;
            float depth=pow((float(p.z)+0.5)/float(LumenVolumeCells.z),LumenVolumeDepth.y);
            vec3 probe=begin+axis*(depth*min(span,LumenVolumeDepth.x)/span); float separation=length(position-probe);
            if (isnan(separation) || isinf(separation) || separation>support) continue;
            for (int coefficient=0;coefficient<Coefficients;++coefficient)
                incoming+=weight*basis[coefficient]*texelFetch(LumenVolumeSH,ivec2(p.x+uint(coefficient)*LumenVolumeCells.x,row),0).xyz;
        }
        return max(incoming,vec3(0))*LumenVolumeDepth.z/(ShPi*LumenVolumeDepth.w);
    }
    for (int coefficient=0;coefficient<Coefficients;++coefficient)
    {
        float x=float(coefficient*int(LumenVolumeCells.x))+grid.x+0.5;
        vec2 a=vec2(x,first*float(LumenVolumeCells.y)+grid.y+0.5)/LumenVolumeExtent.xy;
        vec2 b=vec2(x,second*float(LumenVolumeCells.y)+grid.y+0.5)/LumenVolumeExtent.xy;
        incoming+=basis[coefficient]*mix(textureLod(LumenVolumeSH,a,0).xyz,textureLod(LumenVolumeSH,b,0).xyz,fract(grid.z));
    }
    return max(incoming,vec3(0))*LumenVolumeDepth.z/(ShPi*LumenVolumeDepth.w);
}

#endif

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

// Cook-Torrance microfacet reflection: GGX, separable Smith masking, Schlick Fresnel.
// Walter et al. (2007), Microfacet Models for Refraction through Rough Surfaces.
// Vulkan uses independent GLSL, explicit descriptor bindings, and row-major host matrices.
const int MaximumLights = 4;
const float Pi = 3.141592653589793;
const float MinimumRoughness = 0.045;
const float DistributionEpsilon = 1.0e-12;
const float BrdfEpsilon = 1.0e-6;
const float DielectricFresnel = 0.04;
const float SchlickExponent = 5.0;
const float MicrofacetNormalization = 4.0;
const int DirectionalLight = 0;
const int PointLight = 1;
const int AreaLight = 2;
const int AreaGridSize = 2;
const float AreaSampleWeight = 0.25;
const float AreaSampleOffset = 0.25;
const float NormalizeEpsilon = 1.0e-12;
const float HomogeneousEpsilon = 1.0e-7;
const float DistanceEpsilon = 1.0e-6;
const float UvDeterminantEpsilon = 1.0e-10;
const float ParallelAxisThreshold = 0.999;
const float OutlineWorldScale = 0.01;
const float OverlayMode = 0.5;
const float OutlineMode = 1.5;

const float ShadowStrength = 0.55;
const float MinimumShadowCosine = 0.15;
const float MaximumReceiverBias = 0.0045;
const float ReceiverBias = 0.00035;
const float ReceiverSlopeBias = 0.00060;
const float SrgbLinearThreshold = 0.04045;
const float SrgbLinearScale = 12.92;
const float SrgbPowerOffset = 0.055;
const float SrgbPowerScale = 1.055;
const float SrgbExponent = 2.4;

struct SceneLight
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
layout(set = 0, binding = 0, std140, row_major) uniform DrawConstants
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

vec3 NormalizeOr(vec3 value, vec3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalizeEpsilon ? value * inversesqrt(lengthSquared) : fallback;
}
vec3 DecodeSrgb(vec3 color)
{
    vec3 linear = color / SrgbLinearScale;
    vec3 power = pow((color + SrgbPowerOffset) / SrgbPowerScale, vec3(SrgbExponent));
    return mix(power, linear, lessThanEqual(color, vec3(SrgbLinearThreshold)));
}

void EvaluateSample(vec2 roughnessMetallic, vec3 lightVector, vec3 radiance, float visibility,
    vec3 normal, vec3 viewDirection, vec3 albedo, inout vec3 diffuse, inout vec3 specular)
{
    float distanceSquared = dot(lightVector, lightVector);
    if (distanceSquared <= NormalizeEpsilon) return;
    vec3 direction = lightVector * inversesqrt(distanceSquared);
    float cosine = clamp(dot(normal, direction), 0.0, 1.0);
    if (cosine <= 0.0) return;
    vec3 visibleRadiance = radiance * visibility;
    vec3 halfway = NormalizeOr(direction + viewDirection, normal);
    float nDotV = clamp(dot(normal, viewDirection), 0.0, 1.0);
    if (nDotV <= 0.0) return;
    float nDotH = clamp(dot(normal, halfway), 0.0, 1.0);
    float vDotH = clamp(dot(viewDirection, halfway), 0.0, 1.0);
    float roughness = clamp(roughnessMetallic.x, MinimumRoughness, 1.0);
    float metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float ggxBase = nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
    float distribution = alphaSquared / max(Pi * ggxBase * ggxBase, DistributionEpsilon);
    float masking = 2.0 * nDotV / (nDotV + sqrt(alphaSquared + (1.0 - alphaSquared) * nDotV * nDotV));
    float shadowing = 2.0 * cosine / (cosine + sqrt(alphaSquared + (1.0 - alphaSquared) * cosine * cosine));
    vec3 f0 = mix(Draw.MaterialSpecularColor.www, clamp(albedo, 0.0, 1.0), metallic);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vDotH, SchlickExponent);
    vec3 microfacet = fresnel * distribution * masking * shadowing / max(MicrofacetNormalization * nDotV * cosine, BrdfEpsilon);
    diffuse += visibleRadiance * (1.0 - fresnel) * (1.0 - metallic) * cosine / Pi;
    specular += visibleRadiance * microfacet * cosine;
}

void EvaluateSceneLight(vec2 roughnessMetallic, int index, vec3 position, vec3 normal, vec3 viewDirection,
    vec3 albedo, float visibility, inout vec3 diffuse, inout vec3 specular)
{
    SceneLight light = Draw.Lights[index];
    int type = int(light.PositionType.w + 0.5);
    vec3 radiance = max(light.ColorCastShadows.rgb, vec3(0.0)) * max(light.DirectionIntensity.w, 0.0);
    if (type == DirectionalLight)
    {
        EvaluateSample(roughnessMetallic, light.DirectionIntensity.xyz, radiance, visibility, normal, viewDirection, albedo, diffuse, specular);
        return;
    }
    if (type == PointLight)
    {
        vec3 displacement = light.PositionType.xyz - position;
        float distanceSquared = max(dot(displacement, displacement), DistanceEpsilon);
        EvaluateSample(roughnessMetallic, displacement, radiance * LightRangeWindow(displacement, light.AreaSizePadding.z) / distanceSquared, visibility, normal, viewDirection, albedo, diffuse, specular);
        return;
    }
    vec3 emission = NormalizeOr(light.DirectionIntensity.xyz, vec3(0.0, -1.0, 0.0));
    vec3 axis = abs(emission.y) < ParallelAxisThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = NormalizeOr(cross(axis, emission), vec3(1.0, 0.0, 0.0));
    vec3 bitangent = NormalizeOr(cross(emission, tangent), vec3(0.0, 0.0, 1.0));
    for (int y = 0; y < AreaGridSize; ++y)
    {
        for (int x = 0; x < AreaGridSize; ++x)
        {
            vec2 offset = vec2(x == 0 ? -AreaSampleOffset : AreaSampleOffset, y == 0 ? -AreaSampleOffset : AreaSampleOffset);
            vec3 samplePosition = light.PositionType.xyz + tangent * offset.x * light.AreaSizePadding.x + bitangent * offset.y * light.AreaSizePadding.y;
            vec3 displacement = samplePosition - position;
            float distanceSquared = max(dot(displacement, displacement), DistanceEpsilon);
            vec3 direction = displacement * inversesqrt(distanceSquared);
            float emitterCosine = max(dot(emission, -direction), 0.0);
            EvaluateSample(roughnessMetallic, direction, radiance * (AreaSampleWeight * emitterCosine * LightRangeWindow(displacement, light.AreaSizePadding.z) / distanceSquared), visibility, normal, viewDirection, albedo, diffuse, specular);
        }
    }
}

#ifdef EVANS_VERTEX_STAGE
// Depth capture and material replay require identical raster positions.
invariant gl_Position;
precise gl_Position;
layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
layout(location = 0) out vec3 WorldPosition;
layout(location = 1) out vec3 WorldNormal;
layout(location = 2) out vec4 VertexColor;
layout(location = 3) out vec4 LightPosition;
layout(location = 4) out vec2 TexCoord;
void main()
{
    vec4 position = Draw.World * vec4(InPosition, 1.0);
    vec3 normal = NormalizeOr(mat3(Draw.NormalWorld) * InNormal, vec3(0.0, 1.0, 0.0));
    if (Draw.RenderParameters.y > OutlineMode)
        position.xyz += normal * Draw.MaterialParameters.w * OutlineWorldScale;
    gl_Position = Draw.ViewProjection * position;
    gl_Position.y = -gl_Position.y;
    WorldPosition = position.xyz / max(abs(position.w), HomogeneousEpsilon);
    WorldNormal = normal;
    VertexColor = vec4(DecodeSrgb(clamp(InColor.rgb, 0.0, 1.0)), InColor.a);
    LightPosition = Draw.LightViewProjection * position;
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
}
#else
layout(location = 0) in vec3 WorldPosition;
layout(location = 1) in vec3 WorldNormal;
layout(location = 2) in vec4 VertexColor;
layout(location = 3) in vec4 LightPosition;
layout(location = 4) in vec2 TexCoord;
layout(set = 0, binding = 1) uniform texture2D ShadowDepth;
layout(set = 0, binding = 2) uniform texture2D DiffuseImage;
layout(set = 0, binding = 3) uniform texture2D NormalImage;
layout(set = 0, binding = 4) uniform texture2D OpacityImage;
layout(set = 0, binding = 5) uniform sampler SurfaceSampler;

// Sample-frequency contact lookup must not move the existing pixel-centred
// material/alpha evaluation. In particular, cutouts must match the depth pass.
#if defined(EVANS_CONTACT_RECEIVER) && defined(EVANS_MULTISAMPLE)
#define WorldPosition interpolateAtOffset(WorldPosition, vec2(0.0))
#define WorldNormal interpolateAtOffset(WorldNormal, vec2(0.0))
#define VertexColor interpolateAtOffset(VertexColor, vec2(0.0))
#define LightPosition interpolateAtOffset(LightPosition, vec2(0.0))
#define TexCoord interpolateAtOffset(TexCoord, vec2(0.0))
#endif

// Contact is a separate camera-depth producer. Apply artistic strength once:
// (1 - strength) + strength * rawMapVisibility * rawContactVisibility.
#if defined(EVANS_CONTACT_RECEIVER)
layout(set = 0, binding = 6) uniform texture2DArray ContactImage;
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
    float contact = texelFetch(sampler2DArray(ContactImage, SurfaceSampler),
        ivec3(ivec2(gl_FragCoord.xy), sampleIndex), 0).r;
    return mix(1.0 - ContactShadowStrength, primaryVisibility, contact);
#else
    return primaryVisibility;
#endif
}

layout(location = 0) out vec4 OutColor;
#ifdef EVANS_WEIGHTED_TRANSPARENCY
layout(location = 1) out float RevealageOpacity;
layout(push_constant) uniform WeightedCoverageParameters
{
    vec4 Weights; // opacity exponent, depth exponent, depth scale, minimum weight
} Transparency;
#endif

float CompareShadow(ivec2 pixel, float receiverDepth)
{
    return receiverDepth > texelFetch(sampler2D(ShadowDepth, SurfaceSampler), pixel, 0).r ? 1.0 : 0.0;
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
float VirtualVisibility(vec3 position, float bias)
{
    vec4 light = Csm.LightMatrices[0] * vec4(position, 1.0);
    vec3 projected = light.xyz / max(light.w, Csm.EyeDepthPlane.w);
    vec2 uv = vec2(projected.x, -projected.y) * 0.5 + vec2(0.5, 0.5);
    vec2 gradient = VirtualReceiverGradient(uv, projected.z);
    float footprint = max(length(dFdx(uv) * Csm.EyeDepthPlane.y), length(dFdy(uv) * Csm.EyeDepthPlane.z)) * Csm.FarDepths.x;
    uint mip = uint(clamp(floor(log2(max(1.0, footprint)) + Csm.EyeDepthPlane.x), 0.0, Csm.FarDepths.z - 1.0));
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


float CsmVisibility(vec3 position, vec3 normal, vec3 lightVector)
{
    if (Csm.Filtering.z != 0.0)
    {
        float cosine = clamp(dot(normal, lightVector * inversesqrt(max(dot(lightVector, lightVector), CsmDirectionEpsilon))), 0.0, 1.0);
        float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CsmCosineFloor);
        float bias = min(CsmBiasCeiling, CsmBaseBias + CsmSlopeBias * slope);
        return 1.0 - CsmOcclusionStrength * (1.0 - VirtualVisibility(position, bias));
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

float ShadowVisibility(vec3 normal, vec3 lightVector)
{
    if (Draw.RenderParameters.x >= OverlayMode && Csm.Parameters.z > 0.0)
        return CsmVisibility(WorldPosition, normal, lightVector);
    if (Draw.RenderParameters.x < OverlayMode || abs(LightPosition.w) < HomogeneousEpsilon) return 1.0;
    vec3 projected = LightPosition.xyz / LightPosition.w;
    vec2 uv = vec2(projected.x * 0.5 + 0.5, -projected.y * 0.5 + 0.5);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    ivec2 extent = textureSize(sampler2D(ShadowDepth, SurfaceSampler), 0);
    float cosine = clamp(dot(normal, NormalizeOr(lightVector, normal)), 0.0, 1.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumShadowCosine);
    float bias = min(MaximumReceiverBias, ReceiverBias + ReceiverSlopeBias * slope);
    // Manual depth fetch keeps hard edges distinct from comparison filtering.
    if (Csm.Filtering.x == 0.0)
    {
        ivec2 pixel = clamp(ivec2(floor(uv * vec2(extent))), ivec2(0), extent - 1);
        return 1.0 - ShadowStrength * CompareShadow(pixel, (projected.z - bias));
    }
    vec2 position = uv * vec2(extent) - 0.5;
    ivec2 base = ivec2(floor(position));
    vec2 fraction = fract(position);
    float occlusion = 0.0;
    for (int y = -int(Csm.Filtering.x); y <= int(Csm.Filtering.x); ++y)
    {
        for (int x = -int(Csm.Filtering.x); x <= int(Csm.Filtering.x); ++x)
        {
            ivec2 p00 = clamp(base + ivec2(x, y), ivec2(0), extent - 1);
            ivec2 p10 = clamp(p00 + ivec2(1, 0), ivec2(0), extent - 1);
            ivec2 p01 = clamp(p00 + ivec2(0, 1), ivec2(0), extent - 1);
            ivec2 p11 = clamp(p00 + ivec2(1, 1), ivec2(0), extent - 1);
            float depth = projected.z - bias;
            occlusion += mix(mix(CompareShadow(p00, depth), CompareShadow(p10, depth), fraction.x),
                mix(CompareShadow(p01, depth), CompareShadow(p11, depth), fraction.x), fraction.y);
        }
    }
    return 1.0 - ShadowStrength * occlusion / Csm.Filtering.y;
}

vec3 SurfaceNormal()
{
    vec3 normal = NormalizeOr(WorldNormal, vec3(0.0, 1.0, 0.0));
    if (!gl_FrontFacing) normal = -normal;
    if (Draw.TextureFlags.y == 0) return normal;
    vec3 dx = dFdx(WorldPosition);
    vec3 dy = dFdy(WorldPosition);
    vec2 uvDx = dFdx(TexCoord);
    vec2 uvDy = dFdy(TexCoord);
    float determinant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    if (abs(determinant) < UvDeterminantEpsilon) return normal;
    vec3 tangent = (dx * uvDy.y - dy * uvDx.y) / determinant;
    vec3 sourceBitangent = (-dx * uvDy.x + dy * uvDx.x) / determinant;
    tangent -= normal * dot(normal, tangent);
    vec3 axis = abs(normal.y) < ParallelAxisThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    tangent = NormalizeOr(tangent, NormalizeOr(cross(axis, normal), vec3(1.0, 0.0, 0.0)));
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    vec3 bitangent = NormalizeOr(cross(normal, tangent) * handedness, vec3(0.0, 0.0, 1.0));
    vec3 mapped = texture(sampler2D(NormalImage, SurfaceSampler), TexCoord).xyz * 2.0 - 1.0;
    return NormalizeOr(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
}

// Transparent fragments read their own cumulative medium before source-over.
// Density integration lives in the independent VolumetricFog technique source.
layout(set = 3, binding = 0, std140, row_major) uniform SurfaceFogParameters
{
    mat4 InverseCamera;
    vec4 DensityHeightRange;
    vec4 AmbientSource;
    vec4 ViewAndGrid;
    uvec4 StageSlicesSamplesEnabled;
} SurfaceFog;
layout(set = 3, binding = 3) uniform sampler2D SurfaceFogAtlas;
const float FogCellCenter = 0.5;
const float FogDistanceEpsilon = 1.0e-6;

vec3 FogWorldPosition(vec2 pixelUV, float depth)
{
    vec2 ndc = pixelUV * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    vec4 h = SurfaceFog.InverseCamera * vec4(ndc, depth, 1.0);
    return h.xyz / h.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
const uint RgbTransportMode = 2;
vec4 FogAtBoundaryTransmission(vec2 pixelUV, uint boundary)
{
    if (boundary == 0u) return vec4(1, 1, 1, 1);
    vec2 margin = FogCellCenter / SurfaceFog.ViewAndGrid.zw;
    vec2 atlasUV = clamp(pixelUV, margin, 1.0 - margin);
    atlasUV.x = (float(boundary - 1u) + atlasUV.x) / float(SurfaceFog.StageSlicesSamplesEnabled.y);
    atlasUV.y = (atlasUV.y + 1.0) * 0.5;
    return textureLod(SurfaceFogAtlas, atlasUV, 0.0);
}
vec4 FogAtBoundary(vec2 pixelUV, uint boundary)
{
    if (boundary == 0u) return vec4(0.0, 0.0, 0.0, 1.0);
    vec2 margin = FogCellCenter / SurfaceFog.ViewAndGrid.zw;
    vec2 atlasUV = clamp(pixelUV, margin, 1.0 - margin);
    atlasUV.x = (float(boundary - 1u) + atlasUV.x) / float(SurfaceFog.StageSlicesSamplesEnabled.y);
    if (SurfaceFog.StageSlicesSamplesEnabled.w == RgbTransportMode) atlasUV.y *= 0.5;
    return textureLod(SurfaceFogAtlas, atlasUV, 0.0);
}
vec3 ApplyVolumetricMedium(vec3 radiance)
{
    if (SurfaceFog.StageSlicesSamplesEnabled.w == 0u) return radiance;
    vec2 pixelUV = gl_FragCoord.xy / SurfaceFog.ViewAndGrid.xy;
    vec3 start = FogWorldPosition(pixelUV, 0.0);
    float extent = min(length(FogWorldPosition(pixelUV, 1.0) - start), SurfaceFog.DensityHeightRange.w);
    float distance = length(FogWorldPosition(pixelUV, gl_FragCoord.z) - start);
    float boundary = clamp(distance / max(extent, FogDistanceEpsilon), 0.0, 1.0) * float(SurfaceFog.StageSlicesSamplesEnabled.y);
    uint first = uint(boundary), second = min(first + 1u, SurfaceFog.StageSlicesSamplesEnabled.y);
    vec4 transport = mix(FogAtBoundary(pixelUV, first), FogAtBoundary(pixelUV, second), fract(boundary));
    vec3 mediumTransmission = SurfaceFog.StageSlicesSamplesEnabled.w == RgbTransportMode ? (mix(FogAtBoundaryTransmission(pixelUV, first), FogAtBoundaryTransmission(pixelUV, second), fract(boundary))).rgb : transport.aaa;
    return radiance * mediumTransmission + transport.rgb;
}
#ifdef EVANS_WEIGHTED_TRANSPARENCY
void EvaluateMaterialSurface()
#else
void main()
#endif
{
    vec4 texel = Draw.TextureFlags.x != 0 ? texture(sampler2D(DiffuseImage, SurfaceSampler), TexCoord) : vec4(1.0);
    vec3 surfaceMap = Draw.TextureFlags.z != 0 ? texture(sampler2D(OpacityImage, SurfaceSampler), TexCoord).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = Draw.ShadingParameters.xy;
    if (Draw.TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    float alpha = clamp(VertexColor.a * Draw.MaterialParameters.y * texel.a * opacity, 0.0, 1.0);
    if (Draw.MaterialParameters.z > 0.0 && alpha < Draw.MaterialParameters.z) discard;
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
        OutColor = vec4(ApplyVolumetricMedium(DecodeSrgb(clamp(Draw.PipelineColor.rgb, 0.0, 1.0))), alpha);
        return;
    }
    vec3 base = VertexColor.rgb * max(Draw.PipelineColor.rgb, vec3(0.0)) * max(texel.rgb, vec3(0.0));
    vec3 albedo = base * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0));
    vec3 normal = SurfaceNormal();
    vec3 viewDirection = NormalizeOr(Draw.CameraLightCount.xyz - WorldPosition, normal);
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    for (int index = 0; index < min(int(Draw.CameraLightCount.w + 0.5), MaximumLights); ++index)
    {
        SceneLight light = Draw.Lights[index];
        vec3 shadowVector = int(light.PositionType.w + 0.5) == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - WorldPosition;
        float visibility = index == int(Draw.RenderParameters.z) && light.ColorCastShadows.w > OverlayMode ? ComposeContact(ShadowVisibility(normal, shadowVector)) : 1.0;
        EvaluateSceneLight(roughnessMetallic, index, WorldPosition, normal, viewDirection, albedo, visibility, diffuse, specular);
    }
    vec3 color = base * max(Draw.MaterialAmbientColor.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.w, 0.0) * (1.0 - clamp(roughnessMetallic.y, 0.0, 1.0))
        + albedo * diffuse + specular + max(Draw.MaterialEmissiveColor.rgb, vec3(0.0));
    color = max(color, vec3(0.0));
    color += albedo * (1-LumenDielectricReflectance) * (1.0-clamp(roughnessMetallic.y,0.0,1.0)) * LumenTranslucentIrradiance(WorldPosition,normal);
    color = ApplyVolumetricMedium(color);
    // HDR attachments retain radiance; the legacy display path remains exact.
    if (Draw.ShadingParameters.w <= 0.5)
    {
        color /= max(1.0, max(color.r, max(color.g, color.b)));
        color = clamp(color, 0.0, 1.0);
    }
    const float MaximumHalfFloatRadiance = 65504.0;
    OutColor = vec4(min(color, vec3(MaximumHalfFloatRadiance)), alpha);
}
#ifdef EVANS_WEIGHTED_TRANSPARENCY
void main()
{
    EvaluateMaterialSurface();
    // Preserve this shader's material lighting and fragment-depth fog, then
    // emit the two statistics consumed by the independent Vulkan resolve.
    float coverage = clamp(OutColor.a, 0.0, 1.0);
    RevealageOpacity = coverage;
    if (coverage == 0.0) { OutColor = vec4(0.0); return; }
    float depthFactor = 1.0 - clamp(gl_FragCoord.z, 0.0, 1.0) * Transparency.Weights.z;
    float weightedCoverage = coverage * max(pow(coverage, Transparency.Weights.x) *
        pow(depthFactor, Transparency.Weights.y), Transparency.Weights.w);
    OutColor = vec4(OutColor.rgb * weightedCoverage, weightedCoverage);
}
#endif
#endif
