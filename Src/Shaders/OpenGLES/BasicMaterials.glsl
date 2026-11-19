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

// Basic material evaluation: linear diffuse/ambient/specular/emissive terms,
// scalar opacity and the selected shading model. Raster coverage is separate
// from incident lighting; opaque material tuples retain that separation in MRTs.
// Standalone OpenGLES source. G-buffer and forward entry points share this
// file's material contract; specialized lighting pipelines own their consumers.
#if defined(EVANS_MATERIAL_GBUFFER)
// Material capture writes attributes; the independent Deferred/Tiled stage lights them.
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

// ES material capture selects its vertex stage with EVANS_VERTEX_STAGE.
// Five material MRTs preserve colored ambient separately from emission for later AO.
// Shishkovtsov, "Deferred Shading in S.T.A.L.K.E.R.", GPU Gems 2 (2005).
// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-9-deferred-shading-stalker
precision highp float;
precision highp int;

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
    OutSpecularShininess = clamp(vec4(MaterialSpecularColor.rgb, max(MaterialParameters.x, 1.0)), 0.0, HalfFloatMaximum);
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


#else
// Percentage-closer filtering: average visibility comparisons, not depths.
// Reeves, Salesin and Cook, Rendering Antialiased Shadows with Depth Maps (1987).
// Radius, reciprocal tap count and activation arrive in immutable frame controls.
precision highp float;
precision highp int;
uniform vec4 PcfKernel;
// GLES material shaders consume the GPU lighting volume as one filtered texture;
// storage buffers are needed only by the separate probe integration producer.
uniform highp mat4 LumenVolumeCamera,LumenVolumeInverse;
uniform highp uvec4 LumenVolumeCells;
uniform highp vec4 LumenVolumeDepth,LumenVolumeExtent;
uniform highp sampler2D LumenVolumeSH;
const highp float LumenDielectricReflectance=0.04;
highp vec3 LumenTranslucentIrradiance(highp vec3 position,highp vec3 normal)
{
    if (LumenVolumeCells.w==0u) return vec3(0.0);
    highp vec4 clip=LumenVolumeCamera*vec4(position,1.0);
    if (clip.w<=LumenVolumeExtent.z) return vec3(0.0);
    highp vec3 ndc=clip.xyz/clip.w;
    if (any(greaterThan(abs(ndc.xy),vec2(1.0))) || ndc.z<0.0 || ndc.z>1.0) return vec3(0.0);
    highp vec4 nearH=LumenVolumeInverse*vec4(ndc.xy,0.0,1.0),farH=LumenVolumeInverse*vec4(ndc.xy,1.0,1.0);
    if (min(abs(nearH.w),abs(farH.w))<=LumenVolumeExtent.z) return vec3(0.0);
    highp vec3 start=nearH.xyz/nearH.w;
    highp float extent=min(length(farH.xyz/farH.w-start),LumenVolumeDepth.x),distanceToStart=length(position-start);
    if (extent<=LumenVolumeExtent.z || distanceToStart>extent) return vec3(0.0);
    highp vec2 uv=vec2(ndc.x+1.0,1.0-ndc.y)*0.5;
    highp vec3 cell=vec3(uv*vec2(LumenVolumeCells.xy),pow(clamp(distanceToStart/extent,0.0,1.0),1.0/LumenVolumeDepth.y)*float(LumenVolumeCells.z))-0.5;
    cell=clamp(cell,vec3(0.0),vec3(LumenVolumeCells.xyz)-1.0);
    highp float z0=floor(cell.z),z1=min(z0+1.0,float(LumenVolumeCells.z-1u));
    highp vec3 n=normal*inversesqrt(max(dot(normal,normal),LumenVolumeExtent.w));
    const highp float IrradiancePi=3.14159265358979323846;
    const int TermCount=9;
    highp float terms[TermCount]=float[](sqrt(1.0/(4.0*IrradiancePi)),sqrt(3.0/(4.0*IrradiancePi))*n.y,
        sqrt(3.0/(4.0*IrradiancePi))*n.z,sqrt(3.0/(4.0*IrradiancePi))*n.x,sqrt(15.0/(4.0*IrradiancePi))*n.x*n.y,
        sqrt(15.0/(4.0*IrradiancePi))*n.y*n.z,sqrt(5.0/(16.0*IrradiancePi))*(3.0*n.z*n.z-1.0),
        sqrt(15.0/(4.0*IrradiancePi))*n.x*n.z,sqrt(15.0/(16.0*IrradiancePi))*(n.x*n.x-n.y*n.y));
    highp vec3 sum=vec3(0.0);

    const uint VolumeVisibilityBit=2u;
    if ((LumenVolumeCells.w&VolumeVisibilityBit)!=0u)
    {
        uvec3 base=uvec3(floor(cell)); highp vec3 fraction=fract(cell);
        // Explicit corners retain each probe's visibility certificate. Missing
        // corners stay missing instead of renormalizing SH from another side.
        for (uint corner=0u;corner<8u;++corner)
        {
            uvec3 bit=uvec3(corner&1u,(corner>>1u)&1u,corner>>2u),p=min(base+bit,LumenVolumeCells.xyz-1u);
            highp vec3 weights=mix(1.0-fraction,fraction,notEqual(bit,uvec3(0u))); highp float weight=weights.x*weights.y*weights.z; if (weight<=0.0) continue;
            uint row=p.z*LumenVolumeCells.y+p.y; highp float support=texelFetch(LumenVolumeSH,ivec2(p.x,row),0).a; if (support<=0.0) continue;
            highp vec2 clipXY=vec2(2.0*(float(p.x)+0.5)/float(LumenVolumeCells.x)-1.0,1.0-2.0*(float(p.y)+0.5)/float(LumenVolumeCells.y));
            highp vec4 a=LumenVolumeInverse*vec4(clipXY,0.0,1.0),b=LumenVolumeInverse*vec4(clipXY,1.0,1.0);
            if (min(abs(a.w),abs(b.w))<=LumenVolumeExtent.z) continue;
            highp vec3 begin=a.xyz/a.w,axis=b.xyz/b.w-begin; highp float span=length(axis);
            if (isnan(span) || isinf(span) || span<=LumenVolumeExtent.z) continue;
            highp float depth=pow((float(p.z)+0.5)/float(LumenVolumeCells.z),LumenVolumeDepth.y);
            highp vec3 probe=begin+axis*(depth*min(span,LumenVolumeDepth.x)/span); highp float separation=length(position-probe);
            if (isnan(separation) || isinf(separation) || separation>support) continue;
            for (int band=0;band<TermCount;++band) sum+=weight*terms[band]*texelFetch(LumenVolumeSH,ivec2(p.x+uint(band)*LumenVolumeCells.x,row),0).rgb;
        }
        return max(sum,vec3(0.0))*LumenVolumeDepth.z/(IrradiancePi*LumenVolumeDepth.w);
    }
    for (int band=0;band<TermCount;++band)
    {
        highp float column=float(band)*float(LumenVolumeCells.x)+cell.x+0.5;
        highp vec2 first=vec2(column,z0*float(LumenVolumeCells.y)+cell.y+0.5)/LumenVolumeExtent.xy;
        highp vec2 second=vec2(column,z1*float(LumenVolumeCells.y)+cell.y+0.5)/LumenVolumeExtent.xy;
        sum+=terms[band]*mix(textureLod(LumenVolumeSH,first,0.0).rgb,textureLod(LumenVolumeSH,second,0.0).rgb,cell.z-z0);
    }
    return max(sum,vec3(0.0))*LumenVolumeDepth.z/(IrradiancePi*LumenVolumeDepth.w);
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

// Independent OpenGLES percentage-closer filtering receivers.
// Reeves, Salesin and Cook, Rendering Antialiased Shadows with Depth Maps (1987).
// Average visibility comparisons, never depths. Every square-kernel tap uses
// bilinear comparisons; radius, normalization and activation are frame controls.
#define MATERIAL_FLAT 1
#define MATERIAL_GOURAUD 2
#define MATERIAL_PBR 3
#define MATERIAL_TOON 4
precision highp float;
precision highp int;

const int MaximumLights = 4, MaximumCascades = 4, MaximumProjections = 5;
const int DirectionalLight = 0, PointLight = 1;
const float Pi = 3.14159265358979323846;
const float NormalEpsilon = 1.0e-12, HomogeneousEpsilon = 1.0e-7;
const float HalfRadianceLimit = 65504.0, TexelCenter = 0.5;
const float MaximumMapFraction = 0.25, ShadowOpacity = 0.55;
const float ConstantBias = 0.00035, SlopeBias = 0.00060, MaximumBias = 0.0045, MinimumBiasCosine = 0.15;

struct LightConstants
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
uniform LightConstants Lights[MaximumLights];
uniform vec4 CameraLightCount, ShadingParameters;
uniform mat4 LightViewProjection;
uniform mat4 CascadeLightMatrices[MaximumCascades];
uniform vec4 CascadeFarDistances, CascadeBlendStarts, CascadeEyeDepthPlane, CascadeParameters;
uniform highp sampler2D ShadowMap;
in vec3 WorldPosition, WorldNormal;
in vec4 VertexColor;
in vec2 TexCoord;
#if EVANS_MATERIAL_PROGRAM == MATERIAL_GOURAUD
in vec3 VertexDiffuseFront, VertexSpecularFront, VertexDiffuseBack, VertexSpecularBack;
in vec3 ShadowDiffuseFront, ShadowSpecularFront, ShadowDiffuseBack, ShadowSpecularBack;
#endif
uniform highp sampler2D DiffuseMap, NormalMap, OpacityMap;
uniform vec4 PipelineColor, MaterialDiffuseColor, MaterialAmbientColor, MaterialSpecularColor, MaterialEmissiveColor;
uniform vec4 MaterialParameters, AmbientColorStrength, RenderParameters, TextureFlags;
uniform mat4 FogInverseCamera;
uniform vec4 FogMedium, FogViewGrid;
uniform uvec4 FogControl;
uniform highp sampler2D FogTexture;
layout(location = 0) out vec4 OutColor;
#ifdef EVANS_WEIGHTED_TRANSPARENCY
uniform vec4 TransparencyWeights;
layout(location = 1) out float OutRevealage;
#endif
uniform highp sampler2D CascadeDepth0, CascadeDepth1, CascadeDepth2, CascadeDepth3;

vec3 UnitVector(vec3 value, vec3 fallback)
{
    float squared = dot(value, value);
    return squared > NormalEpsilon ? value * inversesqrt(squared) : fallback;
}
float TopLeftDepth(highp sampler2D image, ivec2 location, bool clampEdges)
{
    ivec2 extent = textureSize(image, 0);
    if (clampEdges) location=clamp(location,ivec2(0),extent-1);
    if (any(lessThan(location, ivec2(0))) || any(greaterThanEqual(location, extent))) return 1.0;
    return texelFetch(image, ivec2(location.x, extent.y - 1 - location.y), 0).r;
}
float Comparison(highp sampler2D image, ivec2 location, float receiver, bool clampEdges)
{
    return receiver <= TopLeftDepth(image, location, clampEdges) ? 1.0 : 0.0;
}
float BilinearVisibility(highp sampler2D image, vec2 uv, float receiver, bool clampEdges)
{
    vec2 footprint = uv * vec2(textureSize(image, 0)) - TexelCenter;
    ivec2 first = ivec2(floor(footprint));
    vec2 weight = fract(footprint);
    float upper = mix(Comparison(image, first, receiver, clampEdges), Comparison(image, first + ivec2(1, 0), receiver, clampEdges), weight.x);
    float lower = mix(Comparison(image, first + ivec2(0, 1), receiver, clampEdges), Comparison(image, first + ivec2(1), receiver, clampEdges), weight.x);
    return mix(upper, lower, weight.y);
}

float EvaluatePCF(highp sampler2D image, int projectionIndex, vec4 clip, float bias)
{
    if (PcfKernel.z<0.5 || clip.w<=HomogeneousEpsilon) return 1.0;
    vec3 projected=clip.xyz/clip.w;
    vec2 center=vec2(projected.x+1.0,1.0-projected.y)*0.5;
    if (any(lessThan(center,vec2(0.0))) || any(greaterThan(center,vec2(1.0))) || projected.z<0.0 || projected.z>1.0) return 1.0;
    vec2 resolution=vec2(textureSize(image,0));
    if (PcfKernel.x==0.0) return Comparison(image,min(ivec2(floor(center*resolution)),ivec2(resolution)-1),projected.z-bias,projectionIndex==0);
    int radius=int(PcfKernel.x);
    float visibility=0.0;
    for (int row=-radius;row<=radius;++row)
        for (int column=-radius;column<=radius;++column)
            visibility+=BilinearVisibility(image,center+vec2(column,row)/resolution,projected.z-bias,projectionIndex==0);
    return visibility*PcfKernel.y;
}
float CascadeShadow(int cascade, vec3 world, float bias)
{
    if (cascade >= int(CascadeParameters.z)) return 1.0;
    vec4 clip = CascadeLightMatrices[cascade] * vec4(world, 1.0);
    if (cascade == 0) return EvaluatePCF(CascadeDepth0, 1, clip, bias);
    if (cascade == 1) return EvaluatePCF(CascadeDepth1, 2, clip, bias);
    if (cascade == 2) return EvaluatePCF(CascadeDepth2, 3, clip, bias);
    return EvaluatePCF(CascadeDepth3, 4, clip, bias);
}
float LightVisibility(vec3 world, float bias, bool cascaded)
{
    if (!cascaded) return EvaluatePCF(ShadowMap, 0, LightViewProjection * vec4(world, 1.0), bias);
    float eyeDistance = dot(CascadeEyeDepthPlane, vec4(world, 1.0));
    if (eyeDistance < CascadeParameters.x || eyeDistance > CascadeParameters.y) return 1.0;
    int cascade = 0;
    while (cascade + 1 < int(CascadeParameters.z) && eyeDistance > CascadeFarDistances[cascade]) ++cascade;
    float visibility = CascadeShadow(cascade, world, bias);
    float overlap = CascadeFarDistances[cascade] - CascadeBlendStarts[cascade];
    if (overlap > 0.0 && eyeDistance >= CascadeBlendStarts[cascade])
        visibility = mix(visibility, CascadeShadow(cascade + 1, world, bias),
            clamp((eyeDistance - CascadeBlendStarts[cascade]) / overlap, 0.0, 1.0));
    return visibility;
}

float SurfaceShadow(vec3 position, vec3 normal, vec3 towardLight)
{
    float cosine = max(dot(UnitVector(normal, vec3(0.0, 1.0, 0.0)), UnitVector(towardLight, vec3(0.0, 1.0, 0.0))), 0.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumBiasCosine);
    float bias = min(ConstantBias + slope * SlopeBias, MaximumBias);
    return 1.0 - ShadowOpacity * (1.0 - LightVisibility(position, bias, CascadeParameters.z > 0.0));
}
struct Material
{
    vec3 Position, Normal, EyeDirection, Diffuse, Specular;
    float Roughness, Metallic, Shininess, Bands;
    int Model;
};
// Separate direct diffuse before screen-space diffusion; ordinary lighting is unchanged.
const uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly()
{
    return false;
}
vec3 ReflectSample(Material material, vec3 towardLight, vec3 radiance)
{
    const float RoughnessFloor = 0.045, DistributionFloor = 1.0e-12, DenominatorFloor = 1.0e-6;
    const float DielectricF0 = 0.04, SchlickExponent = 5.0, HighlightThreshold = 0.5;
    const float MaximumBands = 16.0;
    if (dot(towardLight, towardLight) <= NormalEpsilon) return vec3(0.0);
    vec3 light = UnitVector(towardLight, material.Normal);
    vec3 halfway = UnitVector(light + material.EyeDirection, material.Normal);
    float nl = clamp(dot(material.Normal, light), 0.0, 1.0), nh = clamp(dot(material.Normal, halfway), 0.0, 1.0);
    if (material.Model == MATERIAL_PBR)
    {
        float nv = clamp(dot(material.Normal, material.EyeDirection), 0.0, 1.0);
        if (nl == 0.0 || nv == 0.0) return vec3(0.0);
        float vh = clamp(dot(material.EyeDirection, halfway), 0.0, 1.0);
        float roughness = clamp(material.Roughness, RoughnessFloor, 1.0);
        float alpha = roughness * roughness, variance = alpha * alpha;
        float denominator = 1.0 + (variance - 1.0) * nh * nh;
        float distribution = variance / max(Pi * denominator * denominator, DistributionFloor);
        float smithEye = 2.0 * nv / max(nv + sqrt(variance + (1.0 - variance) * nv * nv), DenominatorFloor);
        float smithLight = 2.0 * nl / max(nl + sqrt(variance + (1.0 - variance) * nl * nl), DenominatorFloor);
        vec3 f0 = mix(vec3(DielectricF0), clamp(material.Diffuse, 0.0, 1.0), material.Metallic);
        vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, SchlickExponent);
        vec3 diffuse = material.Diffuse * (1.0 - fresnel) * (1.0 - material.Metallic) / Pi;
        vec3 specular = fresnel * distribution * smithEye * smithLight / max(4.0 * nl * nv, DenominatorFloor);
        if (DirectDiffuseOnly()) return radiance * nl * diffuse;
        return radiance * nl * (diffuse + specular);
    }
    float specular = nl > 0.0 ? pow(nh, max(material.Shininess, 1.0)) : 0.0;
    if (material.Model == MATERIAL_TOON)
    {
        float bands = clamp(round(material.Bands), 2.0, MaximumBands);
        return radiance * (material.Diffuse * ToonResponse(nl, bands) + (DirectDiffuseOnly() ? vec3(0.0) : material.Specular * ToonHighlight(specular)));
    }
    if (DirectDiffuseOnly()) return radiance * material.Diffuse * nl;
    return radiance * (material.Diffuse * nl + material.Specular * specular);
}
vec3 ShadeLight(Material material, int index, bool shadows, int shadowIndex)
{
    const float DistanceFloor = 1.0e-6, ParallelAxis = 0.999;
    const int RectangleAxisSamples = 2;
    const float RectangleWeight = 1.0 / float(RectangleAxisSamples * RectangleAxisSamples);
    LightConstants light = Lights[index];
    int type = int(light.PositionType.w + 0.5);
    vec3 direction = type == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - material.Position;
    float visibility = shadows && index == shadowIndex && light.ColorCastShadows.w > 0.5 ?
        SurfaceShadow(material.Position, material.Normal, direction) : 1.0;
#ifdef EVANS_CONTACT_RECEIVER
    if (index == shadowIndex && light.ColorCastShadows.w > 0.5)
        visibility = ContactShadow(visibility, ShadowOpacity);
#endif
    vec3 radiance = max(light.ColorCastShadows.rgb, vec3(0.0)) * max(light.DirectionIntensity.w, 0.0) * visibility;
    if (type == DirectionalLight) return ReflectSample(material, direction, radiance);
    if (type == PointLight) return ReflectSample(material, direction, radiance * LightRangeWindow(direction, light.AreaSizePadding.z) / max(dot(direction, direction), DistanceFloor));
    vec3 emission = UnitVector(light.DirectionIntensity.xyz, vec3(0.0, -1.0, 0.0));
    vec3 helper = abs(emission.y) < ParallelAxis ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = UnitVector(cross(helper, emission), vec3(1.0, 0.0, 0.0));
    vec3 bitangent = UnitVector(cross(emission, tangent), vec3(0.0, 0.0, 1.0));
    vec3 result = vec3(0.0);
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSizePadding.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaSizePadding.w, MinimumEmitterGrid, MaximumEmitterGrid)) : RectangleAxisSamples;
    float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSizePadding.x * light.AreaSizePadding.y) : 1.0) / float(emitterGrid * emitterGrid);
    for (int row = 0; row < emitterGrid; ++row)
        for (int column = 0; column < emitterGrid; ++column)
        {
            vec2 offset = ((vec2(column, row) + TexelCenter) / float(emitterGrid) - TexelCenter) * light.AreaSizePadding.xy;
            vec3 delta = light.PositionType.xyz + tangent * offset.x + bitangent * offset.y - material.Position;
            float squared = max(dot(delta, delta), DistanceFloor);
            float emitterCosine = max(dot(emission, -delta * inversesqrt(squared)), 0.0);
            result += ReflectSample(material, delta, radiance * (emitterSampleWeight * emitterCosine * LightRangeWindow(delta, light.AreaSizePadding.z) / squared));
        }
    return result;
}

vec3 SurfaceNormal()
{
    const float UvDeterminantFloor = 1.0e-10, ParallelAxis = 0.999;
    vec3 normal = UnitVector(WorldNormal, vec3(0.0, 1.0, 0.0));
#if EVANS_MATERIAL_PROGRAM == MATERIAL_FLAT
    normal = UnitVector(cross(dFdx(WorldPosition), dFdy(WorldPosition)), normal);
    if (dot(normal, WorldNormal) < 0.0) normal = -normal;
#endif
    if (!gl_FrontFacing) normal = -normal;
#if EVANS_MATERIAL_PROGRAM != MATERIAL_FLAT && EVANS_MATERIAL_PROGRAM != MATERIAL_GOURAUD
    if (TextureFlags.y > 0.5)
    {
        vec3 dx = dFdx(WorldPosition), dy = dFdy(WorldPosition);
        vec2 tx = dFdx(TexCoord), ty = dFdy(TexCoord);
        float determinant = tx.x * ty.y - tx.y * ty.x;
        if (abs(determinant) >= UvDeterminantFloor)
        {
            vec3 tangent = (dx * ty.y - dy * tx.y) / determinant;
            vec3 sourceBitangent = (dy * tx.x - dx * ty.x) / determinant;
            vec3 helper = abs(normal.y) < ParallelAxis ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            tangent = UnitVector(tangent - normal * dot(normal, tangent), UnitVector(cross(helper, normal), vec3(1.0, 0.0, 0.0)));
            float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
            vec3 bitangent = UnitVector(cross(normal, tangent) * handedness, vec3(0.0, 0.0, 1.0));
            vec3 mapped = texture(NormalMap, TexCoord).xyz * 2.0 - 1.0;
            if (float(TextureFlags.w) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
            normal = UnitVector(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
        }
    }
#endif
    return normal;
}
vec3 LinearColor(vec3 encoded)
{
    const float LinearThreshold = 0.04045, LinearDivisor = 12.92, Offset = 0.055, PowerScale = 1.055, Power = 2.4;
    vec3 low = encoded / LinearDivisor, high = pow((encoded + Offset) / PowerScale, vec3(Power));
    return mix(high, low, lessThanEqual(encoded, vec3(LinearThreshold)));
}
vec3 FogWorldPoint(vec2 uv, float depth)
{
    vec4 world = FogInverseCamera * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return world.xyz / world.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
const uint RgbTransportMode = 2u;
vec4 PrefixTransportTransmission(vec2 uv, uint boundary)
{
    if (boundary == 0u) return vec4(1, 1, 1, 1);
    vec2 inset = TexelCenter / FogViewGrid.zw;
    vec2 lookup = clamp(uv, inset, 1.0 - inset);
    lookup.x = (lookup.x + float(boundary - 1u)) / float(FogControl.y);
    lookup.y = (lookup.y + 1.0) * 0.5;
    return textureLod(FogTexture, lookup, 0.0);
}
vec4 PrefixTransport(vec2 uv, uint boundary)
{
    if (boundary == 0u) return vec4(0.0, 0.0, 0.0, 1.0);
    vec2 inset = TexelCenter / FogViewGrid.zw;
    vec2 lookup = clamp(uv, inset, 1.0 - inset);
    lookup.x = (lookup.x + float(boundary - 1u)) / float(FogControl.y);
    if (FogControl.w == RgbTransportMode) lookup.y *= 0.5;
    return textureLod(FogTexture, lookup, 0.0);
}
vec3 FoggedSurface(vec3 radiance)
{
    const float RayDistanceFloor = 1.0e-6;
    if (FogControl.w == 0u) return radiance;
    vec2 uv = gl_FragCoord.xy / FogViewGrid.xy;
    vec3 nearPoint = FogWorldPoint(uv, 0.0);
    float farDistance = min(length(FogWorldPoint(uv, 1.0) - nearPoint), FogMedium.w);
    float normalizedDistance = clamp(length(FogWorldPoint(uv, gl_FragCoord.z) - nearPoint) / max(farDistance, RayDistanceFloor), 0.0, 1.0);
    float endpoint = normalizedDistance * float(FogControl.y);
    uint first = uint(floor(endpoint));
    vec4 transport = mix(PrefixTransport(uv, first), PrefixTransport(uv, min(first + 1u, FogControl.y)), fract(endpoint));
    vec3 mediumTransmission = FogControl.w == RgbTransportMode ? (mix(PrefixTransportTransmission(uv, first), PrefixTransportTransmission(uv, min(first + 1u, FogControl.y)), fract(endpoint))).rgb : transport.aaa;
    return clamp(radiance * mediumTransmission + transport.rgb, 0.0, HalfRadianceLimit);
}
vec4 ShadeSurface()
{
    vec4 diffuseSample = TextureFlags.x > 0.5 ? texture(DiffuseMap, TexCoord) : vec4(1.0);
    if (float(TextureFlags.w) == ToonShadingModel && float(TextureFlags.x) > 0.5)
        diffuseSample.rgb = ToonTextureColor(DiffuseMap, TexCoord, dFdx(TexCoord), dFdy(TexCoord));
    vec3 surfaceMap = TextureFlags.z > 0.5 ? texture(OpacityMap, TexCoord).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = ShadingParameters.xy;
    if (TextureFlags.z > 1.5) roughnessMetallic *= surfaceMap.gb;
    float opacitySample = surfaceMap.r;
    float opacity = clamp(VertexColor.a * MaterialParameters.y * diffuseSample.a * opacitySample, 0.0, 1.0);
    if (MaterialParameters.z > 0.0 && opacity < MaterialParameters.z) discard;
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (RenderParameters.y > 1.0 && RenderParameters.y < 1.5)
    {
        vec3 wireBary = VertexColor.rgb;
        vec3 wireDx = dFdx(wireBary), wireDy = dFdy(wireBary);
        vec3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), vec3(1.0e-8));
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (RenderParameters.y > 0.5) return vec4(FoggedSurface(LinearColor(clamp(PipelineColor.rgb, 0.0, 1.0))), opacity);
    vec3 base = ToonAlbedo(VertexColor.rgb * max(PipelineColor.rgb, vec3(0.0)) * max(diffuseSample.rgb, vec3(0.0)), float(TextureFlags.w));
    Material material;
    material.Position = WorldPosition; material.Normal = SurfaceNormal();
    material.EyeDirection = UnitVector(CameraLightCount.xyz - WorldPosition, material.Normal);
    material.Diffuse = base * max(MaterialDiffuseColor.rgb, vec3(0.0));
    material.Specular = max(MaterialSpecularColor.rgb, vec3(0.0));
    material.Roughness = roughnessMetallic.x; material.Metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    material.Shininess = MaterialParameters.x; material.Bands = ShadingParameters.z; material.Model = EVANS_MATERIAL_PROGRAM;
    vec3 color = ToonAmbient(base * max(MaterialDiffuseColor.rgb, vec3(0.0)), base * max(MaterialAmbientColor.rgb, vec3(0.0)) * max(AmbientColorStrength.rgb, vec3(0.0)) * max(AmbientColorStrength.w, 0.0), float(TextureFlags.w));
#if EVANS_MATERIAL_PROGRAM == MATERIAL_PBR
    color *= 1.0 - material.Metallic;
#endif
    color += max(MaterialEmissiveColor.rgb, vec3(0.0));
#if EVANS_MATERIAL_PROGRAM == MATERIAL_GOURAUD
    int shadowIndex = int(RenderParameters.z);
    float visibility = 1.0;
    if (RenderParameters.x > 0.5 && shadowIndex >= 0 && shadowIndex < MaximumLights)
    {
        LightConstants light = Lights[shadowIndex];
        vec3 direction = int(light.PositionType.w + 0.5) == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - WorldPosition;
        visibility = SurfaceShadow(WorldPosition, material.Normal, direction);
    }
#ifdef EVANS_CONTACT_RECEIVER
    if (shadowIndex >= 0 && shadowIndex < MaximumLights && Lights[shadowIndex].ColorCastShadows.w > 0.5)
        visibility = ContactShadow(visibility, ShadowOpacity);
#endif
    // ES Gouraud vertices carry total irradiance plus the primary light's share.
    // Remove only its occluded contribution, including the instanced interface.
    vec3 diffuse = gl_FrontFacing ? VertexDiffuseFront - (1.0 - visibility) * ShadowDiffuseFront : VertexDiffuseBack - (1.0 - visibility) * ShadowDiffuseBack;
    vec3 specular = gl_FrontFacing ? VertexSpecularFront - (1.0 - visibility) * ShadowSpecularFront : VertexSpecularBack - (1.0 - visibility) * ShadowSpecularBack;
    color += material.Diffuse * diffuse + material.Specular * specular;
#else
    int lightCount = min(int(CameraLightCount.w + 0.5), MaximumLights);
    for (int index = 0; index < lightCount; ++index)
        color += ShadeLight(material, index, RenderParameters.x > 0.5, int(RenderParameters.z));
#endif
    color += material.Diffuse * (material.Model==MATERIAL_PBR ? (1.0-LumenDielectricReflectance)*(1.0-material.Metallic) : 1.0) * LumenTranslucentIrradiance(material.Position,material.Normal);
    color = max(FoggedSurface(color), vec3(0.0));
    color = ShadingParameters.w > 0.5 ? min(color, vec3(HalfRadianceLimit)) :
        clamp(color / max(1.0, max(color.r, max(color.g, color.b))), 0.0, 1.0);
    return vec4(color, opacity);
}
void main()
{
    OutColor = ShadeSurface();
#ifdef EVANS_WEIGHTED_TRANSPARENCY
    float opacity = clamp(OutColor.a, 0.0, 1.0);
    float preference = max(1.0 - clamp(gl_FragCoord.z, 0.0, 1.0) * TransparencyWeights.z, 0.0);
    float weight = max(pow(opacity, TransparencyWeights.x) * pow(preference, TransparencyWeights.y), TransparencyWeights.w);
    OutRevealage = opacity;
    OutColor = vec4(OutColor.rgb * (opacity * weight), opacity * weight);
#endif
}

#endif
