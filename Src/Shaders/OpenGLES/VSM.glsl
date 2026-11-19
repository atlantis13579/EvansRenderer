#version 300 es
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

#ifndef EVANS_VERTEX_STAGE
precision highp float;
precision highp int;
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

#endif

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

// Independent OpenGL ES variance shadow maps. Donnelly and Lauritzen,
// "Variance Shadow Maps" (2006); Lauritzen, GPU Gems 3, chapter 8:
// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps
// RG32F stores normalized linear-depth moments, optionally augmented by local
// raster variance. Positive Gaussian weights preserve a filterable distribution.
// This file owns caster, filter and receiver programs without shader includes.
#define PHONG_RECEIVER 0
#define FLAT_RECEIVER 1
#define GOURAUD_RECEIVER 2
#define PBR_RECEIVER 3
#define TOON_RECEIVER 4
#define DEFERRED_RECEIVER 5
#define MULTISAMPLE_RECEIVER 6
#define VOLUME_RECEIVER 7
#define MOMENT_CASTER 8
#define MOMENT_FILTER 9
#ifndef EVANS_VSM_PROGRAM
#define EVANS_VSM_PROGRAM PHONG_RECEIVER
#endif
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

const int MomentProjectionCount = 5;
uniform vec4 MomentDepthEncoding[MomentProjectionCount];
uniform vec4 MomentSettings; // minimum variance, bleeding reduction, slope cap, Gaussian sigma
uniform uvec4 MomentOptions; // valid map bits, filter radius, reserved, filter axis

#if EVANS_VSM_PROGRAM == MOMENT_CASTER
#ifdef EVANS_VERTEX_STAGE
layout(location = 0) in vec3 InPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
uniform mat4 World, ViewProjection;
#ifdef EVANS_VSM_INSTANCED
const int InstanceCapacity = 64;
struct RigidObject { mat4 World; mat4 Normal; mat4 PreviousClip; vec4 Wind, Gust, VegetationHistory; };
layout(std140, row_major) uniform InstanceBlock
{
    mat4 BatchProjection;
    vec4 BatchMaterial;
    RigidObject Objects[InstanceCapacity];
};
// Spatially coherent gusts travel across patches; a finer wave flutters tips.
// The quadratic profile keeps roots fixed in color, depth, reflection and motion passes.
vec3 VegetationOffset(vec3 world, float height, vec4 wind, vec4 gust, float referenceHeight, float time)
{
    float phase=dot(world.xz,wind.xy)*6.2831853/max(gust.x,1.0)-time*1.7;
    float wave=0.55+gust.y*(0.35*sin(phase)+0.22*sin(phase*0.47+1.3));
    float flutter=gust.z*sin(time*7.1+world.x*1.83+world.z*2.17);
    float bend=clamp(wind.w*(wave+flutter)*height*height/max(referenceHeight,0.1),-height*0.6,height*0.6);
    return vec3(wind.x*bend,-abs(bend)*0.22,wind.y*bend);
}
vec4 VegetationPoint(vec3 position,vec2 uv,RigidObject data)
{
    if (data.Gust.w<0.5) return vec4(position,1);
    vec3 world=(data.World*vec4(position,1)).xyz;
    float height=max(0.0,data.Gust.w<1.5?uv.y:position.y);
    vec3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z);
    return vec4(position+transpose(mat3(data.Normal))*offset,1);
}
#endif
#ifdef EVANS_VSM_INSTANCED
layout(location=15) in highp vec4 WireFacePlane;
uniform highp int WireFaceCull;
uniform highp int WireFaceSign;
void CullWireFace()
{
    if (WireFaceCull==0) return;
    highp mat4 clip=BatchProjection*Objects[gl_InstanceID].World;
    highp vec3 a=clip[0].xyw,b=clip[1].xyw,c=clip[2].xyw,d=clip[3].xyw;
    highp vec4 camera=vec4(dot(b,cross(c,d)),-dot(a,cross(c,d)),dot(a,cross(b,d)),-dot(a,cross(b,c)));
    if (dot(WireFacePlane,camera)*float(WireFaceSign)<=0.0)
        gl_Position=vec4(0.0,0.0,2.0,1.0);
}

#else
layout(location=15) in highp vec4 WireFacePlane;
uniform highp int WireFaceCull;
uniform highp vec4 WireFaceCamera;
void CullWireFace()
{
    if (WireFaceCull!=0 && dot(WireFacePlane,WireFaceCamera)<=0.0)
        gl_Position=vec4(0.0,0.0,2.0,1.0);
}
#endif
out vec2 MaterialUV;
out float LinearLightDepth, VertexAlpha;
void main()
{
#ifdef EVANS_VSM_INSTANCED
    vec4 clip = BatchProjection * Objects[gl_InstanceID].World * VegetationPoint(InPosition,InTexCoord,Objects[gl_InstanceID]);
#else
    vec4 clip = ViewProjection * World * vec4(InPosition, 1.0);
#endif
    // Encode canonical clip coordinates before changing the raster depth range.
    LinearLightDepth = dot(MomentDepthEncoding[0].xyz, vec3(clip.zw, 1.0));
    clip.z = 2.0 * clip.z - clip.w;
    gl_Position = clip;
    MaterialUV = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    VertexAlpha = InColor.a;
    CullWireFace();
}
#else
in vec2 MaterialUV;
in float LinearLightDepth, VertexAlpha;
uniform highp sampler2D DiffuseMap, OpacityMap;
uniform vec4 MaterialParameters, TextureFlags, RenderParameters, ShadowDepthSize;
layout(location = 0) out vec2 OutMoments;
void main()
{
    const float FootprintStandardDeviation = 0.5;
    const float FlagThreshold = 0.5;
    const float CoverageNoiseScale = 52.9829189;
    const vec2 CoverageNoiseDirection = vec2(0.06711056, 0.00583715);
    // Raster gradients belong to this primitive. Evaluate before discard,
    // without differentiating a stored depth discontinuity across silhouettes.
    vec2 slope = vec2(dFdx(LinearLightDepth), dFdy(LinearLightDepth)) * FootprintStandardDeviation;
    float footprint = min(dot(slope, slope), MomentSettings.z);
    float opacity = VertexAlpha * MaterialParameters.y;
    if (TextureFlags.x > FlagThreshold) opacity *= texture(DiffuseMap, MaterialUV).a;
    if (TextureFlags.z > FlagThreshold) opacity *= texture(OpacityMap, MaterialUV).r;
    opacity = clamp(opacity, 0.0, 1.0);
    if (MaterialParameters.z > 0.0)
    {
        if (opacity < MaterialParameters.z) discard;
    }
    else if (RenderParameters.w > FlagThreshold)
    {
        vec2 pixel = floor(gl_FragCoord.xy);
        pixel.y = ShadowDepthSize.y - pixel.y - 1.0;
        float threshold = fract(CoverageNoiseScale * fract(dot(pixel, CoverageNoiseDirection)));
        if (opacity <= 0.0 || opacity < threshold) discard;
    }
    float depth = clamp(LinearLightDepth, 0.0, 1.0);
    OutMoments = vec2(depth, depth * depth + footprint);
}
#endif
#elif EVANS_VSM_PROGRAM == MOMENT_FILTER
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float TriangleExtent = 3.0;
    vec2 position = gl_VertexID == 1 ? vec2(TriangleExtent, -1.0) :
        gl_VertexID == 2 ? vec2(-1.0, TriangleExtent) : vec2(-1.0);
    gl_Position = vec4(position, 0.0, 1.0);
}
#else
uniform highp sampler2D MomentInput;
layout(location = 0) out vec2 OutMoments;
void main()
{
    const float GaussianExponent = -0.5;
    ivec2 size = textureSize(MomentInput, 0), center = ivec2(gl_FragCoord.xy);
    ivec2 axis = MomentOptions.w == 0u ? ivec2(1, 0) : ivec2(0, 1);
    vec2 sum = vec2(0.0); float weightSum = 0.0;
    int radius = int(MomentOptions.y);
    for (int tap = -radius; tap <= radius; ++tap)
    {
        float scaled = float(tap) / MomentSettings.w;
        float weight = exp(GaussianExponent * scaled * scaled);
        ivec2 pixel = center + tap * axis;
        vec2 moments = vec2(1.0);
        if (all(greaterThanEqual(pixel, ivec2(0))) && all(lessThan(pixel, size)))
            moments = texelFetch(MomentInput, pixel, 0).rg;
        sum += moments * weight; weightSum += weight;
    }
    OutMoments = sum / weightSum;
}
#endif
#else
#ifdef EVANS_VERTEX_STAGE
out vec2 ScreenUV;
void main()
{
    const float FullscreenExtent = 3.0;
    vec2 position = gl_VertexID == 1 ? vec2(FullscreenExtent, -1.0) :
        gl_VertexID == 2 ? vec2(-1.0, FullscreenExtent) : vec2(-1.0);
    ScreenUV = (position + 1.0) * 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
#else
const int MaximumLights = 4, MaximumCascades = 4, MaximumProjections = 5;
const int DirectionalLight = 0, PointLight = 1;
const float Pi = 3.14159265358979323846;
const float NormalEpsilon = 1.0e-12, HomogeneousEpsilon = 1.0e-7;
const float HalfRadianceLimit = 65504.0, TexelCenter = 0.5;
const float ShadowOpacity = 0.55;
const float ConstantBias = 0.00035, SlopeBias = 0.00060, MaximumBias = 0.0045, MinimumBiasCosine = 0.15;
#if EVANS_VSM_PROGRAM == VOLUME_RECEIVER
uniform mat4 FogInverseCamera;
uniform vec4 FogMedium, FogRadiance, FogViewGrid;
uniform uvec4 FogControl;
uniform vec4 VolumeScattering, VolumeParameters;
uniform uvec4 VolumeControl;
uniform mat4 VolumeShadowMatrix;
uniform vec4 VolumeLightPosition[MaximumLights], VolumeLightDirection[MaximumLights];
uniform vec4 VolumeLightColor[MaximumLights], VolumeLightArea[MaximumLights];
struct VolumeEmitter
{
    vec4 PositionKind, DirectionPower, ColorShadow, AreaExtent;
};
#ifdef EVANS_VOLUME_FULL_LIGHTS
// Every scattering point reads the complete scene's incident emitters.
layout(std430, binding = 0) readonly buffer VolumeSceneStorage
{ vec4 VolumeSceneWords[]; };
#endif
VolumeEmitter ReadVolumeEmitter(int index)
{
#ifdef EVANS_VOLUME_FULL_LIGHTS
    // Scalar-vector loads avoid aggregate SSBO-return bugs in ANGLE's HLSL path.
    const int WordsPerEmitter = 4;
    const int PositionWord = 0, DirectionWord = 1, ColorWord = 2, AreaWord = 3;
    int first = index * WordsPerEmitter;
    return VolumeEmitter(VolumeSceneWords[first + PositionWord], VolumeSceneWords[first + DirectionWord],
        VolumeSceneWords[first + ColorWord], VolumeSceneWords[first + AreaWord]);
#else
    return VolumeEmitter(VolumeLightPosition[index], VolumeLightDirection[index],
        VolumeLightColor[index], VolumeLightArea[index]);
#endif
}

uniform mat4 VolumeCascadeMatrix[MaximumCascades];
uniform vec4 VolumeCascadeEnds, VolumeBlendBegin, VolumeEyeDepth, VolumeCascadeRange;
uniform highp sampler2D PrimaryDepth;
// These aliases describe this API's two receiver uniform interfaces; no shader
// source is included or generated from another technique or another API.
#define ShadowMap PrimaryDepth
#define LightViewProjection VolumeShadowMatrix
#define CascadeLightMatrices VolumeCascadeMatrix
#define CascadeFarDistances VolumeCascadeEnds
#define CascadeBlendStarts VolumeBlendBegin
#define CascadeEyeDepthPlane VolumeEyeDepth
#define CascadeParameters VolumeCascadeRange
layout(location = 0) out vec4 OutTransport;
#else
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
#if EVANS_VSM_PROGRAM < DEFERRED_RECEIVER
in vec3 WorldPosition, WorldNormal;
in vec4 VertexColor;
in vec2 TexCoord;
#if EVANS_VSM_PROGRAM == GOURAUD_RECEIVER
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
#else
in vec2 ScreenUV;
uniform mat4 InverseViewProjection;
uniform vec4 BackgroundColor;
uniform uvec4 LightingOptions;
#if EVANS_VSM_PROGRAM == MULTISAMPLE_RECEIVER
uniform highp sampler2DMS DiffuseOpacity, NormalRoughness, SpecularShininess, AmbientMetallic, EmissionColor, SceneDepth;
uniform int LightingSample;
uniform vec4 SampleLocation;
#else
uniform highp sampler2D DiffuseOpacity, NormalRoughness, SpecularShininess, AmbientMetallic, EmissionColor, SceneDepth;
#endif
layout(location = 0) out vec4 OutRadiance;
#endif
#endif
uniform highp sampler2D CascadeDepth0, CascadeDepth1, CascadeDepth2, CascadeDepth3;

vec3 UnitVector(vec3 value, vec3 fallback)
{
    float squared = dot(value, value);
    return squared > NormalEpsilon ? value * inversesqrt(squared) : fallback;
}
// Receivers use top-left shadow coordinates. The textures themselves retain
// native ES row order through both Gaussian passes.
vec2 MomentTexel(highp sampler2D image, ivec2 pixel)
{
    ivec2 size = textureSize(image, 0);
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, size))) return vec2(1.0);
    return texelFetch(image, ivec2(pixel.x, size.y - 1 - pixel.y), 0).rg;
}
vec2 FilteredMoment(highp sampler2D image, vec2 uv)
{
    vec2 pixel = uv * vec2(textureSize(image, 0)) - TexelCenter;
    ivec2 lower = ivec2(floor(pixel)); vec2 weight = fract(pixel);
    vec2 top = mix(MomentTexel(image, lower), MomentTexel(image, lower + ivec2(1, 0)), weight.x);
    vec2 bottom = mix(MomentTexel(image, lower + ivec2(0, 1)), MomentTexel(image, lower + ivec2(1)), weight.x);
    return mix(top, bottom, weight.y);
}
float EvaluateMoments(highp sampler2D image, int projectionIndex, vec4 clip, float bias)
{
    if (clip.w <= HomogeneousEpsilon || (MomentOptions.x & (1u << uint(projectionIndex))) == 0u) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = vec2(projected.x + 1.0, 1.0 - projected.y) * 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float receiver = dot(MomentDepthEncoding[projectionIndex].xyz, vec3(clip.zw, 1.0));
    vec2 moments = FilteredMoment(image, uv);
    float separation = receiver - bias - moments.x;
    if (separation <= 0.0) return 1.0;
    float variance = max(moments.y - moments.x * moments.x, MomentSettings.x);
    float bound = variance / (variance + separation * separation);
    // Artistic bleeding suppression forfeits the mathematical upper bound.
    return clamp((bound - MomentSettings.y) / (1.0 - MomentSettings.y), 0.0, 1.0);
}
float CascadeShadow(int cascade, vec3 world, float bias)
{
    if (cascade >= int(CascadeParameters.z)) return 1.0;
    vec4 clip = CascadeLightMatrices[cascade] * vec4(world, 1.0);
    if (cascade == 0) return EvaluateMoments(CascadeDepth0, 1, clip, bias);
    if (cascade == 1) return EvaluateMoments(CascadeDepth1, 2, clip, bias);
    if (cascade == 2) return EvaluateMoments(CascadeDepth2, 3, clip, bias);
    return EvaluateMoments(CascadeDepth3, 4, clip, bias);
}
float LightVisibility(vec3 world, float bias, bool cascaded)
{
    if (!cascaded) return EvaluateMoments(ShadowMap, 0, LightViewProjection * vec4(world, 1.0), bias);
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

#if EVANS_VSM_PROGRAM != VOLUME_RECEIVER
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
#if EVANS_VSM_PROGRAM >= DEFERRED_RECEIVER && EVANS_VSM_PROGRAM < VOLUME_RECEIVER
    return LightingOptions.w == DirectDiffuseOutput;
#else
    return false;
#endif
}
vec3 ReflectSample(Material material, vec3 towardLight, vec3 radiance)
{
    const float RoughnessFloor = 0.045, DistributionFloor = 1.0e-12, DenominatorFloor = 1.0e-6;
    const float DielectricF0 = 0.04, SchlickExponent = 5.0, HighlightThreshold = 0.5;
#if EVANS_VSM_PROGRAM < DEFERRED_RECEIVER
    const float MaximumBands = 16.0;
    if (dot(towardLight, towardLight) <= NormalEpsilon) return vec3(0.0);
#else
    const float MaximumBands = 12.0;
#endif
    vec3 light = UnitVector(towardLight, material.Normal);
    vec3 halfway = UnitVector(light + material.EyeDirection, material.Normal);
    float nl = clamp(dot(material.Normal, light), 0.0, 1.0), nh = clamp(dot(material.Normal, halfway), 0.0, 1.0);
    if (material.Model == PBR_RECEIVER)
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
    if (material.Model == TOON_RECEIVER)
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
#endif

#if EVANS_VSM_PROGRAM < DEFERRED_RECEIVER
vec3 SurfaceNormal()
{
    const float UvDeterminantFloor = 1.0e-10, ParallelAxis = 0.999;
    vec3 normal = UnitVector(WorldNormal, vec3(0.0, 1.0, 0.0));
#if EVANS_VSM_PROGRAM == FLAT_RECEIVER
    normal = UnitVector(cross(dFdx(WorldPosition), dFdy(WorldPosition)), normal);
    if (dot(normal, WorldNormal) < 0.0) normal = -normal;
#endif
    if (!gl_FrontFacing) normal = -normal;
#if EVANS_VSM_PROGRAM != FLAT_RECEIVER && EVANS_VSM_PROGRAM != GOURAUD_RECEIVER
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
    material.Shininess = MaterialParameters.x; material.Bands = ShadingParameters.z; material.Model = EVANS_VSM_PROGRAM;
    vec3 color = ToonAmbient(base * max(MaterialDiffuseColor.rgb, vec3(0.0)), base * max(MaterialAmbientColor.rgb, vec3(0.0)) * max(AmbientColorStrength.rgb, vec3(0.0)) * max(AmbientColorStrength.w, 0.0), float(TextureFlags.w));
#if EVANS_VSM_PROGRAM == PBR_RECEIVER
    color *= 1.0 - material.Metallic;
#endif
    color += max(MaterialEmissiveColor.rgb, vec3(0.0));
#if EVANS_VSM_PROGRAM == GOURAUD_RECEIVER
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
    color += material.Diffuse * (material.Model==PBR_RECEIVER ? (1.0-LumenDielectricReflectance)*(1.0-material.Metallic) : 1.0) * LumenTranslucentIrradiance(material.Position,material.Normal);
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
#elif EVANS_VSM_PROGRAM != VOLUME_RECEIVER
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#if EVANS_VSM_PROGRAM == MULTISAMPLE_RECEIVER
    int selectedSample = LightingSample;
    vec2 uv = (vec2(pixel) + SampleLocation.xy) / vec2(textureSize(SceneDepth));
#else
    int selectedSample = 0;
    vec2 uv = ScreenUV;
#endif
    vec4 normal = texelFetch(NormalRoughness, pixel, selectedSample);
    vec4 diffuse = texelFetch(DiffuseOpacity, pixel, selectedSample);
    vec4 emission = texelFetch(EmissionColor, pixel, selectedSample);
    if (normal.w < 0.0) { OutRadiance = DirectDiffuseOnly() ? vec4(0.0) : vec4(emission.rgb, diffuse.a); return; }
    float depth = texelFetch(SceneDepth, pixel, selectedSample).r;
    if (depth >= 1.0) { OutRadiance = DirectDiffuseOnly() ? vec4(0.0) : BackgroundColor; return; }
    vec4 specular = texelFetch(SpecularShininess, pixel, selectedSample);
    vec4 ambient = texelFetch(AmbientMetallic, pixel, selectedSample);
    vec4 world = InverseViewProjection * vec4(uv * 2.0 - 1.0, depth, 1.0);
    Material material;
    material.Position = world.xyz / world.w; material.Normal = UnitVector(normal.xyz, vec3(0.0, 1.0, 0.0));
    material.EyeDirection = UnitVector(CameraLightCount.xyz - material.Position, material.Normal);
    material.Diffuse = diffuse.rgb; material.Specular = specular.rgb;
    material.Roughness = normal.w; material.Metallic = clamp(ambient.w, 0.0, 1.0);
    material.Shininess = specular.w; material.Bands = ShadingParameters.x; material.Model = int(LightingOptions.x);
    vec3 radiance = DirectDiffuseOnly() ? vec3(0.0) : ambient.rgb + emission.rgb;
    int count = min(int(CameraLightCount.w + 0.5), MaximumLights);
    for (int index = 0; index < count; ++index)
        radiance += ShadeLight(material, index, LightingOptions.y != 0u, int(LightingOptions.z));
    float coverage = DirectDiffuseOnly() ? (LightingOptions.x == uint(PBR_RECEIVER) ? 1.0 - material.Metallic : 1.0) : diffuse.a;
    if (DirectDiffuseOnly()) { radiance *= emission.a; coverage *= emission.a; }
    OutRadiance = vec4(clamp(radiance, 0.0, HalfRadianceLimit), coverage);
}
#else
// Inject local single-scattering transport. The separate volume-light program
// still owns the associative prefix scan and its floating intermediate targets.
const float MediumDistanceFloor = 1.0e-6, LightDistanceSquaredFloor = 1.0e-4;
const float OpticalDepthLimit = 80.0, ParallelThreshold = 0.99;
vec3 MediumDirection(vec3 value, vec3 fallback)
{
    float squared = dot(value, value);
    return squared > LightDistanceSquaredFloor ? value * inversesqrt(squared) : fallback;
}
vec3 MediumPoint(vec2 uv, float depth)
{
    vec4 world = FogInverseCamera * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return world.xyz / world.w;
}
float AngularDensity(vec3 towardLight, vec3 cameraDirection)
{
    float cosine = clamp(dot(towardLight, cameraDirection), -1.0, 1.0), g = VolumeScattering.w;
    float denominator = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / (4.0 * Pi * denominator * sqrt(denominator));
}
float MediumLightGeometry(VolumeEmitter light, vec3 world, vec3 cameraDirection)
{
    const int AxisSamples = 2;
    const float SampleWeight = 1.0 / float(AxisSamples * AxisSamples);
    int type = int(light.PositionKind.w);
    vec3 direction = MediumDirection(light.DirectionPower.xyz, vec3(0.0, 1.0, 0.0));
    if (type == DirectionalLight) return AngularDensity(direction, cameraDirection);
    if (type == PointLight)
    {
        vec3 delta = light.PositionKind.xyz - world;
        return AngularDensity(MediumDirection(delta, direction), cameraDirection) * LightRangeWindow(delta, light.AreaExtent.z) / max(dot(delta, delta), LightDistanceSquaredFloor);
    }
    vec3 helper = abs(direction.y) < ParallelThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, direction)), bitangent = cross(direction, tangent);
    float integral = 0.0;
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaExtent.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaExtent.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AxisSamples;
    float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
    for (int row = 0; row < emitterGrid; ++row)
        for (int column = 0; column < emitterGrid; ++column)
        {
            vec2 offset = ((vec2(column, row) + TexelCenter) / float(emitterGrid) - TexelCenter) * light.AreaExtent.xy;
            vec3 delta = light.PositionKind.xyz + tangent * offset.x + bitangent * offset.y - world;
            vec3 towardLight = MediumDirection(delta, -direction);
            integral += emitterSampleWeight * max(dot(direction, -towardLight), 0.0) * AngularDensity(towardLight, cameraDirection) * LightRangeWindow(delta, light.AreaExtent.z) / max(dot(delta, delta), LightDistanceSquaredFloor);
        }
    return integral * light.AreaExtent.x * light.AreaExtent.y;
}
vec3 MediumSource(vec3 world, vec3 cameraDirection)
{
    float visibility = VolumeControl.y != 0u ? LightVisibility(world, VolumeParameters.y, VolumeControl.y == 2u) : 1.0;
    vec3 direct = vec3(0.0);
    for (int index = 0; index < int(VolumeControl.x); ++index)
    {
        float visible = VolumeControl.y != 0u && uint(index) == VolumeControl.z ? visibility : 1.0;
        VolumeEmitter light = ReadVolumeEmitter(index);
        vec3 radiance = min(light.ColorShadow.rgb, vec3(HalfRadianceLimit)) * min(light.DirectionPower.w, HalfRadianceLimit);
        direct += radiance * (visible * MediumLightGeometry(light, world, cameraDirection));
    }
    return FogRadiance.rgb + VolumeScattering.rgb * VolumeParameters.x * direct;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    int columns = int(FogViewGrid.z), slice = pixel.x / columns;
    vec2 uv = (vec2(pixel.x % columns, pixel.y) + TexelCenter) / FogViewGrid.zw;
    vec3 origin = MediumPoint(uv, 0.0), ray = MediumPoint(uv, 1.0) - origin;
    float rayLength = max(length(ray), MediumDistanceFloor);
    vec3 direction = ray / rayLength;
    float stepDistance = min(rayLength, FogMedium.w) / float(FogControl.y * FogControl.z);
    vec4 transport = vec4(0.0, 0.0, 0.0, 1.0);
    for (uint index = 0u; index < FogControl.z; ++index)
    {
        float distance = (float(uint(slice) * FogControl.z + index) + TexelCenter) * stepDistance;
        vec3 world = origin + direction * distance;
        float sigma = FogMedium.x * exp(-min(FogMedium.z * max(world.y - FogMedium.y, 0.0), OpticalDepthLimit));
        float transmission = exp(-min(sigma * stepDistance, OpticalDepthLimit));
        transport.rgb += transport.a * (1.0 - transmission) * MediumSource(world, direction);
        transport.a *= transmission;
    }
    OutTransport = transport;
}
#endif
#endif

#endif // independent moment producers / lighting receivers
