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

// Native D3D12 translucent Lumen receiver. The atlas contains signed, cosine-
// convolved SH; division by pi below applies the diffuse BRDF exactly once.
cbuffer LumenTranslucencyParameters : register(b9)
{
    row_major float4x4 TranslucentCamera,TranslucentInverse;
    uint4 TranslucentCells;
    float4 TranslucentDepthScale,TranslucentImageSize;
};
Texture2D<float4> TranslucentSH : register(t19);
SamplerState TranslucentFilter : register(s2);
static const float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 position,float3 n)
{
    if (TranslucentCells.w==0) return 0;
    float4 homogeneous=mul(TranslucentCamera,float4(position,1));
    if (homogeneous.w<=TranslucentImageSize.z) return 0;
    float3 screen=homogeneous.xyz/homogeneous.w;
    if (screen.z<0 || screen.z>1 || any(abs(screen.xy)>1)) return 0;
    float4 originH=mul(TranslucentInverse,float4(screen.xy,0,1)),endH=mul(TranslucentInverse,float4(screen.xy,1,1));
    if (min(abs(originH.w),abs(endH.w))<=TranslucentImageSize.z) return 0;
    float3 origin=originH.xyz/originH.w;
    float span=min(length(endH.xyz/endH.w-origin),TranslucentDepthScale.x),travel=length(position-origin);
    if (span<=TranslucentImageSize.z || travel>span) return 0;
    float2 screenUV=float2(screen.x*0.5+0.5,0.5-screen.y*0.5);
    float3 location=float3(screenUV*TranslucentCells.xy,pow(saturate(travel/span),rcp(TranslucentDepthScale.y))*TranslucentCells.z)-0.5;
    location=clamp(location,0,float3(TranslucentCells.xyz)-1);
    float z0=floor(location.z),z1=min(z0+1,float(TranslucentCells.z-1));
    n*=rsqrt(max(dot(n,n),TranslucentImageSize.w));
    static const float SHPi=3.14159265358979323846;
    static const uint SHCount=9;
    float sh[SHCount]={sqrt(1/(4*SHPi)),sqrt(3/(4*SHPi))*n.y,sqrt(3/(4*SHPi))*n.z,sqrt(3/(4*SHPi))*n.x,
        sqrt(15/(4*SHPi))*n.x*n.y,sqrt(15/(4*SHPi))*n.y*n.z,sqrt(5/(16*SHPi))*(3*n.z*n.z-1),
        sqrt(15/(4*SHPi))*n.x*n.z,sqrt(15/(16*SHPi))*(n.x*n.x-n.y*n.y)};
    float3 result=0;

    static const uint VolumeVisibilityBit=2;
    if ((TranslucentCells.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(location)); float3 fraction=frac(location);
        // Unsupported corners keep zero weight in the original kernel; a
        // surviving bright corner must not acquire the missing corner's mass.
        [loop] for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,TranslucentCells.xyz-1);
            float weight=(bit.x?fraction.x:1-fraction.x)*(bit.y?fraction.y:1-fraction.y)*(bit.z?fraction.z:1-fraction.z); if (weight<=0) continue;
            uint row=p.z*TranslucentCells.y+p.y; float radius=TranslucentSH.Load(int3(p.x,row,0)).w; if (radius<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5)/TranslucentCells.x-1,1-2*(float(p.y)+0.5)/TranslucentCells.y);
            float4 a=mul(TranslucentInverse,float4(clipXY,0,1)),b=mul(TranslucentInverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=TranslucentImageSize.z) continue;
            float3 start=a.xyz/a.w,axis=b.xyz/b.w-start; float spanToFar=length(axis);
            if (!isfinite(spanToFar) || spanToFar<=TranslucentImageSize.z) continue;
            float depth=pow((float(p.z)+0.5)/TranslucentCells.z,TranslucentDepthScale.y);
            float3 probe=start+axis*(depth*min(spanToFar,TranslucentDepthScale.x)/spanToFar);
            float separation=length(position-probe); if (!isfinite(separation) || separation>radius) continue;
            [unroll] for (uint band=0;band<SHCount;++band)
                result+=weight*sh[band]*TranslucentSH.Load(int3(p.x+band*TranslucentCells.x,row,0)).xyz;
        }
        return max(result,0)*TranslucentDepthScale.z/(SHPi*TranslucentDepthScale.w);
    }
    [unroll] for (uint band=0;band<SHCount;++band)
    {
        float column=band*TranslucentCells.x+location.x+0.5;
        float2 low=float2(column,z0*TranslucentCells.y+location.y+0.5)/TranslucentImageSize.xy;
        float2 high=float2(column,z1*TranslucentCells.y+location.y+0.5)/TranslucentImageSize.xy;
        result+=sh[band]*lerp(TranslucentSH.SampleLevel(TranslucentFilter,low,0).xyz,
            TranslucentSH.SampleLevel(TranslucentFilter,high,0).xyz,location.z-z0);
    }
    return max(result,0)*TranslucentDepthScale.z/(SHPi*TranslucentDepthScale.w);
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

// DX12 variance shadow mapping: independent moment rasterization, separable
// filtering and surface/volume visibility. No cross-technique shader includes.
// Donnelly and Lauritzen, "Variance Shadow Maps" (2006); GPU Gems 3, chapter 8:
// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps
// Linear moments are generated before device-depth quantization. An immutable
// b6 root CBV supplies each draw's projection encoding and variance settings.
#define RECEIVER_PHONG 0
#define RECEIVER_FLAT 1
#define RECEIVER_GOURAUD 2
#define RECEIVER_PBR 3
#define RECEIVER_TOON 4
#define RECEIVER_DEFERRED 5
#define RECEIVER_MULTISAMPLE 6
#define RECEIVER_VOLUME 7
#define MOMENT_CASTER 8
#define MOMENT_FILTER 9
#ifndef EVANS_VSM_PROGRAM
#define EVANS_VSM_PROGRAM RECEIVER_PHONG
#endif

static const uint LightCountLimit = 4, CascadeCountLimit = 4, ProjectionCount = 5;
static const uint DirectionalKind = 0, PointKind = 1;
static const float Pi = 3.14159265358979323846;
static const float PixelCenterOffset = 0.5;
static const float ClipWMinimum = 1.0e-7, VectorLengthSquaredMinimum = 1.0e-12;
static const float HalfFloatMaximum = 65504;
static const float ShadowStrength = 0.55;
static const float BiasConstant = 0.00035, BiasSlope = 0.00060, BiasMaximum = 0.0045, BiasCosineFloor = 0.15;
struct Emitter
{
    float4 PositionKind;
    float4 DirectionPower;
    float4 ColorShadow;
    float4 Rectangle;
};
cbuffer VarianceUpload : register(b6)
{
    float4 LinearDepthEncoding[ProjectionCount]; // raw clip Z/W coefficients, offset, reserved
    float4 MomentSettings; // variance floor, bleeding suppression, raster variance cap, Gaussian sigma
    uint4 MomentOptions; // valid projection bits, filter radius, reserved, filter axis
};

#if EVANS_VSM_PROGRAM >= MOMENT_CASTER
#if EVANS_VSM_PROGRAM == MOMENT_CASTER
cbuffer CasterUpload : register(b0)
{
    row_major float4x4 ObjectToWorld, ObjectNormal, CameraProjection, ShadowProjection;
    float4 ObjectTint, DiffuseReflectance, AmbientReflectance, SpecularReflectance, Emission;
    float4 MaterialShape, AmbientIllumination, EyeAndLightCount, RenderOptions;
    uint4 MaterialTextures;
};
static const uint ObjectCapacity = 64;
struct ObjectRecord
{
    row_major float4x4 LocalToWorld, NormalToWorld, PreviousProjection;
    float4 Wind, Gust, VegetationHistory;
};
cbuffer InstanceUpload : register(b3)
{
    row_major float4x4 InstanceCamera;
    float4 InstanceOptions;
    ObjectRecord Objects[ObjectCapacity];
};
// Spatially coherent gusts travel across patches; a finer wave flutters tips.
// The quadratic profile keeps roots fixed in color, depth, reflection and motion passes.
float3 VegetationOffset(float3 world, float height, float4 wind, float4 gust, float referenceHeight, float time)
{
    float phase=dot(world.xz,wind.xy)*6.2831853/max(gust.x,1.0)-time*1.7;
    float wave=0.55+gust.y*(0.35*sin(phase)+0.22*sin(phase*0.47+1.3));
    float flutter=gust.z*sin(time*7.1+world.x*1.83+world.z*2.17);
    float bend=clamp(wind.w*(wave+flutter)*height*height/max(referenceHeight,0.1),-height*0.6,height*0.6);
    return float3(wind.x*bend,-abs(bend)*0.22,wind.y*bend);
}
float4 VegetationPoint(float3 position,float2 uv,ObjectRecord data)
{
    if (data.Gust.w<0.5) return float4(position,1);
    float3 world=mul(data.LocalToWorld,float4(position,1)).xyz;
    float height=max(0.0,data.Gust.w<1.5?uv.y:position.y);
    float3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z);
    return float4(position+mul(transpose((float3x3)data.NormalToWorld),offset),1);
}
Texture2D<float4> DiffuseImage : register(t1);
Texture2D<float4> OpacityImage : register(t3);
SamplerState MaterialSampler : register(s0);
struct CasterInput
{
    float3 Position : POSITION;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};
struct CasterOutput
{
    float4 ClipPosition : SV_POSITION;
    float2 MaterialUV : TEXCOORD0;
    float LinearDepth : TEXCOORD1;
    float VertexAlpha : COLOR0;
};
CasterOutput ProjectCaster(CasterInput input, float4 clip)
{
    CasterOutput output;
    output.ClipPosition = clip;
    output.MaterialUV = float2(input.TexCoord.x, 1 - input.TexCoord.y);
    output.VertexAlpha = input.Color.a;
    output.LinearDepth = dot(LinearDepthEncoding[0].xyz, float3(clip.zw, 1));
    return output;
}
CasterOutput VSMain(CasterInput input)
{
    float4 world = mul(ObjectToWorld, float4(input.Position, 1));
    return ProjectCaster(input, mul(CameraProjection, world));
}
CasterOutput VSInstanced(CasterInput input, uint objectIndex : SV_InstanceID)
{
    float4 world = mul(Objects[objectIndex].LocalToWorld, VegetationPoint(input.Position,input.TexCoord,Objects[objectIndex]));
    return ProjectCaster(input, mul(InstanceCamera, world));
}
float2 PSMain(CasterOutput input) : SV_TARGET
{
    const float RasterFootprintSigma = 0.5;
    const float InterleavedScale = 52.9829189;
    const float2 InterleavedDirection = float2(0.06711056, 0.00583715);
    // Derivatives must be evaluated before alpha discard, within the current
    // primitive's interpolation plane. Clamp their variance near grazing slopes.
    float derivativeX = ddx(input.LinearDepth) * RasterFootprintSigma;
    float derivativeY = ddy(input.LinearDepth) * RasterFootprintSigma;
    float footprint = min(derivativeX * derivativeX + derivativeY * derivativeY, MomentSettings.z);
    float opacity = input.VertexAlpha * MaterialShape.y;
    if (MaterialTextures.x != 0) opacity *= DiffuseImage.Sample(MaterialSampler, input.MaterialUV).a;
    if (MaterialTextures.z != 0) opacity *= OpacityImage.Sample(MaterialSampler, input.MaterialUV).r;
    opacity = saturate(opacity);
    if (MaterialShape.z > 0) clip(opacity - MaterialShape.z);
    else if (RenderOptions.w > 0.5)
    {
        float threshold = frac(InterleavedScale * frac(dot(floor(input.ClipPosition.xy), InterleavedDirection)));
        if (opacity <= 0 || opacity < threshold) discard;
    }
    float first = saturate(input.LinearDepth);
    return float2(first, first * first + footprint);
}
#else
Texture2D<float2> UnfilteredMoments : register(t0);
float4 VSMain(uint vertexIndex : SV_VertexID) : SV_POSITION
{
    const float2 FullscreenTriangle[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(FullscreenTriangle[vertexIndex], 0, 1);
}
float2 PSMain(float4 position : SV_POSITION) : SV_TARGET
{
    const float GaussianPower = -0.5;
    uint width, height; UnfilteredMoments.GetDimensions(width, height);
    int2 extent = int2(width, height);
    int2 center = int2(position.xy), axis = MomentOptions.w == 0 ? int2(1, 0) : int2(0, 1);
    float2 accumulated = 0; float normalization = 0;
    int radius = int(MomentOptions.y);
    [loop] for (int tap = -radius; tap <= radius; ++tap)
    {
        float standardizedDistance = tap / MomentSettings.w;
        float weight = exp(GaussianPower * standardizedDistance * standardizedDistance);
        int2 pixel = center + tap * axis;
        float2 value = float2(1, 1);
        if (all(pixel >= 0) && all(pixel < extent)) value = UnfilteredMoments.Load(int3(pixel, 0));
        accumulated += value * weight;
        normalization += weight;
    }
    return accumulated / normalization;
}
#endif
#else
#if EVANS_VSM_PROGRAM == RECEIVER_VOLUME
// Volume samples use complete scene storage, never opaque surface lists.
StructuredBuffer<Emitter> VolumeSceneLights : register(t16);
cbuffer MediumRoot : register(b0)
{
    row_major float4x4 CameraInverse;
    float4 MediumDensity, AmbientSource, GridDimensions;
    uint4 IntegrationSteps;
};
cbuffer ScatteringUpload : register(b1)
{
    float4 ScatteringAlbedo, ScatteringScale;
    uint4 LightOptions;
    row_major float4x4 ShadowProjection;
    Emitter SceneLights[LightCountLimit];
    row_major float4x4 CascadeProjection[CascadeCountLimit];
    float4 CascadeEnds, CascadeOverlaps, EyeDepthPlane, CascadeSettings, UnusedFixedFilter;
};
Texture2D<float2> SingleMoments : register(t1);
Texture2D<float2> CascadeMoments0 : register(t2);
Texture2D<float2> CascadeMoments1 : register(t3);
Texture2D<float2> CascadeMoments2 : register(t4);
Texture2D<float2> CascadeMoments3 : register(t5);
#else
#if EVANS_VSM_PROGRAM < RECEIVER_DEFERRED
cbuffer SurfaceDraw : register(b0)
{
    row_major float4x4 ObjectTransform, ObjectNormalTransform, CameraProjection, ShadowProjection;
    float4 DrawColor, DiffuseReflectance, AmbientReflectance, SpecularReflectance, EmissionRadiance;
    float4 MaterialParameters, AmbientLighting, EyeLightCount, DrawOptions;
    uint4 MaterialImages;
    Emitter SceneLights[LightCountLimit];
    float4 SurfaceParameters; // roughness, metallic, toon bands, HDR output
};
Texture2D<float4> DiffuseImage : register(t1);
Texture2D<float4> NormalImage : register(t2);
Texture2D<float4> OpacityImage : register(t3);
SamplerState MaterialSampler : register(s0);
cbuffer SurfaceFogRoot : register(b4)
{
    row_major float4x4 FogCameraInverse;
    float4 FogDensity, FogAmbient, FogDimensions;
    uint4 FogIntegration;
};
Texture2D<float4> FogTransport : register(t14);
SamplerState FogSampler : register(s1);
cbuffer TransparentRoot : register(b5)
{
    float4 TransparencyWeights;
    uint4 TransparencyOptions;
};
#else
cbuffer DeferredUpload : register(b1)
{
    row_major float4x4 CameraInverse, ShadowProjection;
    float4 EyeLightCount, SurfaceParameters;
    uint4 DeferredOptions;
    float4 EmptyColor;
    Emitter SceneLights[LightCountLimit];
};
#if EVANS_VSM_PROGRAM == RECEIVER_MULTISAMPLE
Texture2DMS<float4> MaterialDiffuse : register(t4);
Texture2DMS<float4> MaterialNormal : register(t5);
Texture2DMS<float4> MaterialSpecular : register(t6);
Texture2DMS<float4> MaterialAmbient : register(t7);
Texture2DMS<float4> MaterialEmission : register(t8);
Texture2DMS<float> CameraDepth : register(t9);
#else
Texture2D<float4> MaterialDiffuse : register(t4);
Texture2D<float4> MaterialNormal : register(t5);
Texture2D<float4> MaterialSpecular : register(t6);
Texture2D<float4> MaterialAmbient : register(t7);
Texture2D<float4> MaterialEmission : register(t8);
Texture2D<float> CameraDepth : register(t9);
#endif
#endif
cbuffer CascadeUpload : register(b2)
{
    row_major float4x4 CascadeProjection[CascadeCountLimit];
    float4 CascadeEnds, CascadeOverlaps, EyeDepthPlane, CascadeSettings, UnusedFixedFilter;
};
Texture2D<float2> SingleMoments : register(t0);
Texture2D<float2> CascadeMoments0 : register(t10);
Texture2D<float2> CascadeMoments1 : register(t11);
Texture2D<float2> CascadeMoments2 : register(t12);
Texture2D<float2> CascadeMoments3 : register(t13);
#endif

float3 Unit(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > VectorLengthSquaredMinimum ? value * rsqrt(squared) : fallback;
}
float2 ReadMoments(Texture2D<float2> image, int2 pixel, int2 extent)
{
    // The far clear plane also represents samples outside a map.
    if (any(pixel < 0) || any(pixel >= extent)) return float2(1, 1);
    return image.Load(int3(pixel, 0));
}
float2 FilteredMoments(Texture2D<float2> image, float2 uv)
{
    uint width, height; image.GetDimensions(width, height);
    int2 extent = int2(width, height);
    float2 position = uv * extent - PixelCenterOffset;
    int2 lower = int2(floor(position)); float2 blend = frac(position);
    float2 upperRow = lerp(ReadMoments(image, lower, extent), ReadMoments(image, lower + int2(1, 0), extent), blend.x);
    float2 lowerRow = lerp(ReadMoments(image, lower + int2(0, 1), extent), ReadMoments(image, lower + 1, extent), blend.x);
    return lerp(upperRow, lowerRow, blend.y);
}
float MomentVisibility(Texture2D<float2> image, uint projection, float4 clip, float linearBias)
{
    if (clip.w <= ClipWMinimum || (MomentOptions.x & (1u << projection)) == 0) return 1;
    float3 device = clip.xyz / clip.w;
    float2 uv = float2(device.x + 1, 1 - device.y) / 2;
    if (any(uv < 0) || any(uv > 1) || device.z < 0 || device.z > 1) return 1;
    float receiver = dot(LinearDepthEncoding[projection].xyz, float3(clip.zw, 1));
    float2 moments = FilteredMoments(image, uv);
    float separation = receiver - linearBias - moments.x;
    if (separation <= 0) return 1;
    float variance = max(moments.y - moments.x * moments.x, MomentSettings.x);
    float probability = variance / (variance + separation * separation);
    // Bleeding suppression is an artistic remap, not a tighter probability bound.
    return saturate((probability - MomentSettings.y) / (1 - MomentSettings.y));
}
float CascadeVisibility(uint cascade, float3 world, float bias)
{
    float result = 1;
    if (cascade < uint(CascadeSettings.z))
    {
        float4 clip = mul(CascadeProjection[cascade], float4(world, 1));
        if (cascade == 0) result = MomentVisibility(CascadeMoments0, 1, clip, bias);
        else if (cascade == 1) result = MomentVisibility(CascadeMoments1, 2, clip, bias);
        else if (cascade == 2) result = MomentVisibility(CascadeMoments2, 3, clip, bias);
        else result = MomentVisibility(CascadeMoments3, 4, clip, bias);
    }
    return result;
}
float PrimaryVisibility(float3 world, float bias, bool cascaded)
{
    float visibility = 1;
    if (!cascaded) visibility = MomentVisibility(SingleMoments, 0, mul(ShadowProjection, float4(world, 1)), bias);
    else
    {
        float eyeDepth = dot(EyeDepthPlane, float4(world, 1));
        if (eyeDepth >= CascadeSettings.x && eyeDepth <= CascadeSettings.y)
        {
            uint index = 0;
            [loop] while (index + 1 < uint(CascadeSettings.z) && eyeDepth > CascadeEnds[index]) ++index;
            visibility = CascadeVisibility(index, world, bias);
            float overlap = CascadeEnds[index] - CascadeOverlaps[index];
            if (overlap > 0 && eyeDepth >= CascadeOverlaps[index])
            {
                float following = CascadeVisibility(index + 1, world, bias);
                visibility = lerp(visibility, following, saturate((eyeDepth - CascadeOverlaps[index]) / overlap));
            }
        }
    }
    return visibility;
}

#if EVANS_VSM_PROGRAM != RECEIVER_VOLUME
float SurfaceVisibility(float3 world, float3 normal, float3 toLight)
{
    float cosine = saturate(dot(Unit(normal, float3(0, 1, 0)), Unit(toLight, float3(0, 1, 0))));
    float tangent = sqrt(max(1 - cosine * cosine, 0)) / max(cosine, BiasCosineFloor);
    float bias = min(BiasConstant + BiasSlope * tangent, BiasMaximum);
    return 1 - ShadowStrength * (1 - PrimaryVisibility(world, bias, CascadeSettings.z > 0));
}
struct Surface
{
    float3 World, Normal, View, Diffuse, Specular;
    float Roughness, Metallic, Shininess, Bands;
    uint Model;
};
// Separate direct diffuse before screen-space diffusion; ordinary lighting is unchanged.
static const uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly()
{
#if EVANS_VSM_PROGRAM >= RECEIVER_DEFERRED && EVANS_VSM_PROGRAM < RECEIVER_VOLUME
    return DeferredOptions.w == DirectDiffuseOutput;
#else
    return false;
#endif
}
float3 SampleReflectance(Surface material, float3 toLight, float3 radiance)
{
    const float RoughnessFloor = 0.045, DielectricF0 = 0.04;
    const float DenominatorFloor = 1.0e-6, DistributionFloor = 1.0e-12, FresnelPower = 5;
    const float SpecularThreshold = 0.5, MinimumBands = 2;
#if EVANS_VSM_PROGRAM < RECEIVER_DEFERRED
    const float MaximumBands = 16;
#else
    const float MaximumBands = 12;
#endif
    float3 direction = Unit(toLight, material.Normal), halfway = Unit(direction + material.View, material.Normal);
    float nl = saturate(dot(material.Normal, direction)), nh = saturate(dot(material.Normal, halfway));
    float3 reflected = 0;
    if (material.Model == RECEIVER_PBR)
    {
        float nv = saturate(dot(material.Normal, material.View));
        if (nl > 0 && nv > 0)
        {
            float vh = saturate(dot(material.View, halfway));
            float roughness = clamp(material.Roughness, RoughnessFloor, 1);
            float alpha = roughness * roughness, variance = alpha * alpha;
            float ggxTerm = 1 + (variance - 1) * nh * nh;
            float distribution = variance / max(Pi * ggxTerm * ggxTerm, DistributionFloor);
            float maskingView = 2 * nv / max(nv + sqrt(variance + (1 - variance) * nv * nv), DenominatorFloor);
            float maskingLight = 2 * nl / max(nl + sqrt(variance + (1 - variance) * nl * nl), DenominatorFloor);
            float3 f0 = lerp(DielectricF0.xxx, saturate(material.Diffuse), material.Metallic);
            float3 fresnel = f0 + (1 - f0) * pow(1 - vh, FresnelPower);
            float3 diffuse = material.Diffuse * (1 - fresnel) * (1 - material.Metallic) / Pi;
            float3 specular = fresnel * distribution * maskingView * maskingLight / max(4 * nl * nv, DenominatorFloor);
            reflected = radiance * nl * (diffuse + (DirectDiffuseOnly() ? 0.0 : specular));
        }
    }
    else
    {
        float specular = nl > 0 ? pow(nh, max(material.Shininess, 1)) : 0;
        if (material.Model == RECEIVER_TOON)
        {
            float bands = clamp(round(material.Bands), MinimumBands, MaximumBands);
            reflected = radiance * (material.Diffuse * ToonResponse(nl, bands) + (DirectDiffuseOnly() ? 0.0 : material.Specular * ToonHighlight(specular)));
        }
        else reflected = radiance * (material.Diffuse * nl + (DirectDiffuseOnly() ? 0.0 : material.Specular * specular));
#if EVANS_VSM_PROGRAM < RECEIVER_DEFERRED
        if (material.Model <= RECEIVER_FLAT && dot(toLight, toLight) <= VectorLengthSquaredMinimum) reflected = 0;
#endif
    }
    return reflected;
}
float3 SurfaceLight(Surface material, uint index, bool shadowEnabled, int shadowLight, float contactVisibility)
{
    const float DistanceSquaredFloor = 1.0e-6, ParallelThreshold = 0.999;
    const uint RectangleSamplesPerAxis = 2;
    const float SampleWeight = 1.0 / (RectangleSamplesPerAxis * RectangleSamplesPerAxis);
    Emitter light = SceneLights[index];
    uint kind = uint(light.PositionKind.w + 0.5);
    float3 centerDirection = kind == DirectionalKind ? light.DirectionPower.xyz : light.PositionKind.xyz - material.World;
    float visibility = 1;
    if (int(index) == shadowLight && light.ColorShadow.w > 0.5)
    {
        if (shadowEnabled) visibility = SurfaceVisibility(material.World, material.Normal, centerDirection);
        visibility = CombineContactVisibility(visibility, contactVisibility, ShadowStrength);
    }
    float3 power = max(light.ColorShadow.rgb, 0) * max(light.DirectionPower.w, 0) * visibility;
    float3 result = 0;
    if (kind == DirectionalKind) result = SampleReflectance(material, centerDirection, power);
    else if (kind == PointKind)
        result = SampleReflectance(material, centerDirection, power * LightRangeWindow(centerDirection, light.Rectangle.z) / max(dot(centerDirection, centerDirection), DistanceSquaredFloor));
    else
    {
        float3 normal = Unit(light.DirectionPower.xyz, float3(0, -1, 0));
        float3 axis = abs(normal.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 tangent = Unit(cross(axis, normal), float3(1, 0, 0));
        float3 bitangent = Unit(cross(normal, tangent), float3(0, 0, 1));
        // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
        const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
        const float EmitterStratumCenter = 0.5;
        bool physicalEmitter = light.Rectangle.w > 0.0;
        uint emitterGrid = physicalEmitter ? uint(clamp(light.Rectangle.w, MinimumEmitterGrid, MaximumEmitterGrid)) : RectangleSamplesPerAxis;
        float emitterSampleWeight = (physicalEmitter ? abs(light.Rectangle.x * light.Rectangle.y) : 1.0) / float(emitterGrid * emitterGrid);
        [loop] for (uint row = 0; row < emitterGrid; ++row)
            [loop] for (uint column = 0; column < emitterGrid; ++column)
            {
                float2 samplePoint = ((float2(column, row) + PixelCenterOffset) / emitterGrid - PixelCenterOffset) * light.Rectangle.xy;
                float3 delta = light.PositionKind.xyz + samplePoint.x * tangent + samplePoint.y * bitangent - material.World;
                float squared = max(dot(delta, delta), DistanceSquaredFloor);
                float cosine = max(dot(normal, -delta * rsqrt(squared)), 0);
                result += SampleReflectance(material, delta, power * (emitterSampleWeight * cosine * LightRangeWindow(delta, light.Rectangle.z) / squared));
            }
    }
    return result;
}
#endif

#if EVANS_VSM_PROGRAM < RECEIVER_DEFERRED
struct SurfaceFragment
{
    float4 Pixel : SV_POSITION;
    float3 World : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 UnusedShadowPosition : TEXCOORD2;
    float2 UV : TEXCOORD3;
#if EVANS_VSM_PROGRAM == RECEIVER_GOURAUD
    float3 DiffuseFront : TEXCOORD4;
    float3 SpecularFront : TEXCOORD5;
    float3 ShadowDiffuseFront : TEXCOORD6;
    float3 ShadowSpecularFront : TEXCOORD7;
    float3 DiffuseBack : TEXCOORD8;
    float3 SpecularBack : TEXCOORD9;
    float3 ShadowDiffuseBack : TEXCOORD10;
    float3 ShadowSpecularBack : TEXCOORD11;
#endif
};
float3 ShadingNormal(SurfaceFragment fragment, bool front)
{
    const float UvDeterminantFloor = 1.0e-10, ParallelThreshold = 0.999;
    float3 normal = Unit(fragment.Normal, float3(0, 1, 0));
#if EVANS_VSM_PROGRAM == RECEIVER_FLAT
    normal = Unit(cross(ddx(fragment.World), ddy(fragment.World)), normal);
    if (dot(normal, fragment.Normal) < 0) normal = -normal;
#endif
    if (!front) normal = -normal;
#if EVANS_VSM_PROGRAM != RECEIVER_FLAT && EVANS_VSM_PROGRAM != RECEIVER_GOURAUD
    if (MaterialImages.y != 0)
    {
        float3 dx = ddx(fragment.World), dy = ddy(fragment.World);
        float2 du = ddx(fragment.UV), dv = ddy(fragment.UV);
        float determinant = du.x * dv.y - du.y * dv.x;
        if (abs(determinant) >= UvDeterminantFloor)
        {
            float3 tangent = (dx * dv.y - dy * du.y) / determinant;
            float3 referenceBitangent = (dy * du.x - dx * dv.x) / determinant;
            float3 helper = abs(normal.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
            tangent = Unit(tangent - normal * dot(normal, tangent), Unit(cross(helper, normal), float3(1, 0, 0)));
            float handedness = dot(cross(normal, tangent), referenceBitangent) < 0 ? -1 : 1;
            float3 bitangent = Unit(cross(normal, tangent) * handedness, float3(0, 0, 1));
            float3 encoded = NormalImage.Sample(MaterialSampler, fragment.UV).xyz * 2 - 1;
            if (float(MaterialImages.w) == ToonShadingModel) encoded = lerp(float3(0.0, 0.0, 1.0), encoded, ToonNormalStrength);
            normal = Unit(tangent * encoded.x + bitangent * encoded.y + normal * encoded.z, normal);
        }
    }
#endif
    return normal;
}
float3 DecodeColor(float3 color)
{
    const float Threshold = 0.04045, LinearDivisor = 12.92, Offset = 0.055, Scale = 1.055, Gamma = 2.4;
    float3 low = color / LinearDivisor, high = pow((color + Offset) / Scale, Gamma);
    return float3(color.r <= Threshold ? low.r : high.r, color.g <= Threshold ? low.g : high.g, color.b <= Threshold ? low.b : high.b);
}
float3 FogWorld(float2 uv, float depth)
{
    float4 homogeneous = mul(FogCameraInverse, float4(2 * uv.x - 1, 1 - 2 * uv.y, depth, 1));
    return homogeneous.xyz / homogeneous.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
static const uint RgbTransportMode = 2;
float4 IntegratedBoundaryTransmission(float2 uv, uint boundary)
{
    float4 transport = float4(1, 1, 1, 1);
    if (boundary > 0)
    {
        float2 inset = PixelCenterOffset / FogDimensions.zw;
        float2 lookup = clamp(uv, inset, 1 - inset);
        lookup.x = (lookup.x + boundary - 1) / FogIntegration.y;
        lookup.y = (lookup.y + 1.0) * 0.5;
        transport = FogTransport.SampleLevel(FogSampler, lookup, 0);
    }
    return transport;
}
float4 IntegratedBoundary(float2 uv, uint boundary)
{
    float4 transport = float4(0, 0, 0, 1);
    if (boundary > 0)
    {
        float2 inset = PixelCenterOffset / FogDimensions.zw;
        float2 lookup = clamp(uv, inset, 1 - inset);
        lookup.x = (lookup.x + boundary - 1) / FogIntegration.y;
        if (FogIntegration.w == RgbTransportMode) lookup.y *= 0.5;
        transport = FogTransport.SampleLevel(FogSampler, lookup, 0);
    }
    return transport;
}
float3 FogSurface(float3 radiance, float4 pixel)
{
    const float DistanceFloor = 1.0e-6;
    if (FogIntegration.w != 0)
    {
        float2 uv = pixel.xy / FogDimensions.xy;
        float3 nearPoint = FogWorld(uv, 0);
        float distance = min(length(FogWorld(uv, 1) - nearPoint), FogDensity.w);
        float endpoint = FogIntegration.y * saturate(length(FogWorld(uv, pixel.z) - nearPoint) / max(distance, DistanceFloor));
        uint lower = uint(floor(endpoint)), upper = min(lower + 1, FogIntegration.y);
        float4 transport = lerp(IntegratedBoundary(uv, lower), IntegratedBoundary(uv, upper), frac(endpoint));
        float3 mediumTransmission = FogIntegration.w == RgbTransportMode ? (lerp(IntegratedBoundaryTransmission(uv, lower), IntegratedBoundaryTransmission(uv, upper), frac(endpoint))).rgb : transport.aaa;
        radiance = clamp(radiance * mediumTransmission + transport.rgb, 0, HalfFloatMaximum);
    }
    return radiance;
}
#if defined(EVANS_CONTACT_RECEIVER)
float4 PSContact(SurfaceFragment fragment, bool front : SV_IsFrontFace, uint contactSample : SV_SampleIndex) : SV_TARGET
#else
float4 PSMain(SurfaceFragment fragment, bool front : SV_IsFrontFace) : SV_TARGET
#endif
{
#if !defined(EVANS_CONTACT_RECEIVER)
    const uint contactSample = 0;
#endif
    float4 diffuseSample = MaterialImages.x != 0 ? DiffuseImage.Sample(MaterialSampler, fragment.UV) : 1;
    if (float(MaterialImages.w) == ToonShadingModel && MaterialImages.x != 0u)
        diffuseSample.rgb = ToonTextureColor(DiffuseImage, MaterialSampler, fragment.UV, ddx(fragment.UV), ddy(fragment.UV));
    float3 surfaceMap = MaterialImages.z != 0 ? OpacityImage.Sample(MaterialSampler, fragment.UV).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = SurfaceParameters.xy;
    if (MaterialImages.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacitySample = surfaceMap.r;
    float opacity = saturate(fragment.Color.a * MaterialParameters.y * diffuseSample.a * opacitySample);
    if (MaterialParameters.z > 0) clip(opacity - MaterialParameters.z);
    float3 color = 0;
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (DrawOptions.y > 1.0 && DrawOptions.y < 1.5)
    {
        float3 wireBary = fragment.Color.rgb;
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (DrawOptions.y > 0.5) color = FogSurface(DecodeColor(saturate(DrawColor.rgb)), fragment.Pixel);
    else
    {
        float3 base = ToonAlbedo(fragment.Color.rgb * max(DrawColor.rgb, 0) * max(diffuseSample.rgb, 0), float(MaterialImages.w));
        Surface material;
        material.World = fragment.World; material.Normal = ShadingNormal(fragment, front);
        material.View = Unit(EyeLightCount.xyz - material.World, material.Normal);
        material.Diffuse = base * max(DiffuseReflectance.rgb, 0);
        material.Specular = max(SpecularReflectance.rgb, 0);
        material.Roughness = roughnessMetallic.x; material.Metallic = saturate(roughnessMetallic.y);
        material.Shininess = MaterialParameters.x; material.Bands = SurfaceParameters.z;
        material.Model = EVANS_VSM_PROGRAM;
        color = ToonAmbient(base * max(DiffuseReflectance.rgb, 0.0), base * max(AmbientReflectance.rgb, 0) * max(AmbientLighting.rgb, 0) * max(AmbientLighting.w, 0), float(MaterialImages.w));
#if EVANS_VSM_PROGRAM == RECEIVER_PBR
        color *= 1 - material.Metallic;
#endif
        color += max(EmissionRadiance.rgb, 0);
#if EVANS_VSM_PROGRAM == RECEIVER_GOURAUD
        float visibility = 1;
        int shadowLight = int(DrawOptions.z);
        if (shadowLight >= 0 && shadowLight < int(LightCountLimit))
        {
            Emitter light = SceneLights[shadowLight];
            float3 direction = uint(light.PositionKind.w + 0.5) == DirectionalKind ? light.DirectionPower.xyz : light.PositionKind.xyz - material.World;
            if (DrawOptions.x > 0.5) visibility = SurfaceVisibility(material.World, material.Normal, direction);
            if (light.ColorShadow.w > 0.5)
                visibility = CombineContactVisibility(visibility, ReadContactVisibility(fragment.Pixel, contactSample), ShadowStrength);
        }
        float3 diffuse = front ? fragment.DiffuseFront + visibility * fragment.ShadowDiffuseFront : fragment.DiffuseBack + visibility * fragment.ShadowDiffuseBack;
        float3 specular = front ? fragment.SpecularFront + visibility * fragment.ShadowSpecularFront : fragment.SpecularBack + visibility * fragment.ShadowSpecularBack;
        color += material.Diffuse * diffuse + material.Specular * specular;
#else
        [loop] for (uint index = 0; index < min(uint(EyeLightCount.w + 0.5), LightCountLimit); ++index)
            color += SurfaceLight(material, index, DrawOptions.x > 0.5, int(DrawOptions.z), ReadContactVisibility(fragment.Pixel, contactSample));
#endif
        color += material.Diffuse * (material.Model==RECEIVER_PBR ? (1-LumenDielectricReflectance)*(1-material.Metallic) : 1) * LumenTranslucentIrradiance(material.World,material.Normal);
        color = max(FogSurface(color, fragment.Pixel), 0);
        if (SurfaceParameters.w > 0.5) color = min(color, HalfFloatMaximum);
        else color = saturate(color / max(1, max(color.r, max(color.g, color.b))));
    }
    return float4(color, opacity);
}
struct TransparentStatistics
{
    float4 Accumulation : SV_Target0;
    float Revealage : SV_Target1;
};
#if defined(EVANS_CONTACT_RECEIVER)
float4 PSMain(SurfaceFragment fragment, bool front : SV_IsFrontFace) : SV_TARGET
{ return PSContact(fragment, front, 0); }
#endif
TransparentStatistics PSWeighted(SurfaceFragment fragment, bool front : SV_IsFrontFace)
{
    float4 shaded = PSMain(fragment, front);
    float opacity = saturate(shaded.a);
    float depthWeight = pow(max(1 - saturate(fragment.Pixel.z) * TransparencyWeights.z, 0), TransparencyWeights.y);
    float weight = max(pow(opacity, TransparencyWeights.x) * depthWeight, TransparencyWeights.w);
    TransparentStatistics statistics;
    statistics.Accumulation = float4(shaded.rgb * opacity * weight, opacity * weight);
    statistics.Revealage = opacity;
    return statistics;
}
#elif EVANS_VSM_PROGRAM < RECEIVER_VOLUME
struct DeferredFragment { float4 Pixel : SV_POSITION; float2 UV : TEXCOORD0; };
#if EVANS_VSM_PROGRAM == RECEIVER_MULTISAMPLE
float4 PSMain(DeferredFragment fragment, uint sampleIndex : SV_SampleIndex) : SV_TARGET
#else
float4 PSMain(DeferredFragment fragment) : SV_TARGET
#endif
{
#if EVANS_VSM_PROGRAM == RECEIVER_MULTISAMPLE
    int2 pixel = int2(fragment.Pixel.xy);
    float depth = CameraDepth.Load(pixel, sampleIndex);
    float4 diffuse = MaterialDiffuse.Load(pixel, sampleIndex), normal = MaterialNormal.Load(pixel, sampleIndex);
    float4 specular = MaterialSpecular.Load(pixel, sampleIndex), ambient = MaterialAmbient.Load(pixel, sampleIndex);
    float4 emission = MaterialEmission.Load(pixel, sampleIndex);
    uint width, height, samples;
    CameraDepth.GetDimensions(width, height, samples);
    float2 uv = (pixel + PixelCenterOffset + CameraDepth.GetSamplePosition(sampleIndex)) / float2(width, height);
#else
    const uint sampleIndex = 0;
    int3 pixel = int3(fragment.Pixel.xy, 0);
    float depth = CameraDepth.Load(pixel);
    float4 diffuse = MaterialDiffuse.Load(pixel), normal = MaterialNormal.Load(pixel);
    float4 specular = MaterialSpecular.Load(pixel), ambient = MaterialAmbient.Load(pixel);
    float4 emission = MaterialEmission.Load(pixel);
    float2 uv = fragment.UV;
#endif
    float4 result = DirectDiffuseOnly() ? 0.0 : EmptyColor;
    if (normal.w < 0) result = DirectDiffuseOnly() ? 0.0 : float4(emission.rgb, diffuse.a);
    else if (depth < 1)
    {
        float4 world = mul(CameraInverse, float4(2 * uv.x - 1, 1 - 2 * uv.y, depth, 1));
        Surface material;
        material.World = world.xyz / world.w; material.Normal = Unit(normal.xyz, float3(0, 1, 0));
        material.View = Unit(EyeLightCount.xyz - material.World, material.Normal);
        material.Diffuse = diffuse.rgb; material.Specular = specular.rgb;
        material.Roughness = normal.w; material.Metallic = saturate(ambient.w);
        material.Shininess = specular.w; material.Bands = SurfaceParameters.x; material.Model = DeferredOptions.x;
        float3 radiance = DirectDiffuseOnly() ? 0.0 : ambient.rgb + emission.rgb;
        [loop] for (uint index = 0; index < min(uint(EyeLightCount.w + 0.5), LightCountLimit); ++index)
            radiance += SurfaceLight(material, index, DeferredOptions.y != 0, int(DeferredOptions.z), ReadContactVisibility(fragment.Pixel, sampleIndex));
        float coverage = DirectDiffuseOnly() ? (DeferredOptions.x == RECEIVER_PBR ? 1 - material.Metallic : 1) : diffuse.a;
        if (DirectDiffuseOnly()) { radiance *= emission.a; coverage *= emission.a; }
        result = float4(clamp(radiance, 0, HalfFloatMaximum), coverage);
    }
    return result;
}
#else
// The prefix-scan program remains in VolumetricLighting. This receiver replaces
// only injection, evaluating visibility at every medium integration point.
static const float OpticalDepthMaximum = 80, MediumDistanceFloor = 1.0e-6;
static const float LightDistanceSquaredFloor = 1.0e-4, VolumeParallelThreshold = 0.99;
float3 MediumUnit(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > LightDistanceSquaredFloor ? value * rsqrt(squared) : fallback;
}
float Phase(float3 towardLight, float3 cameraRay)
{
    float cosine = clamp(dot(towardLight, cameraRay), -1, 1), asymmetry = ScatteringAlbedo.w;
    float denominator = 1 + asymmetry * asymmetry - 2 * asymmetry * cosine;
    return (1 - asymmetry * asymmetry) / (4 * Pi * denominator * sqrt(denominator));
}
float MediumLightIntegral(Emitter light, float3 world, float3 cameraRay)
{
    const uint AxisSamples = 2;
    const float SampleWeight = 1.0 / (AxisSamples * AxisSamples);
    uint kind = uint(light.PositionKind.w);
    float3 normal = MediumUnit(light.DirectionPower.xyz, float3(0, 1, 0));
    float integral = 0;
    if (kind == DirectionalKind) integral = Phase(normal, cameraRay);
    else if (kind == PointKind)
    {
        float3 delta = light.PositionKind.xyz - world;
        integral = Phase(MediumUnit(delta, normal), cameraRay) * LightRangeWindow(delta, light.Rectangle.z) / max(dot(delta, delta), LightDistanceSquaredFloor);
    }
    else
    {
        float3 helper = abs(normal.y) < VolumeParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
        // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
        const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
        const float EmitterStratumCenter = 0.5;
        bool physicalEmitter = light.Rectangle.w > 0.0;
        uint emitterGrid = physicalEmitter ? uint(clamp(light.Rectangle.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AxisSamples;
        float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
        [loop] for (uint row = 0; row < emitterGrid; ++row)
            [loop] for (uint column = 0; column < emitterGrid; ++column)
            {
                float2 position = ((float2(column, row) + PixelCenterOffset) / emitterGrid - PixelCenterOffset) * light.Rectangle.xy;
                float3 delta = light.PositionKind.xyz + tangent * position.x + bitangent * position.y - world;
                float3 direction = MediumUnit(delta, -normal);
                integral += emitterSampleWeight * saturate(dot(normal, -direction)) * Phase(direction, cameraRay) * LightRangeWindow(delta, light.Rectangle.z) / max(dot(delta, delta), LightDistanceSquaredFloor);
            }
        integral *= light.Rectangle.x * light.Rectangle.y;
    }
    return integral;
}
float3 MediumRadiance(float3 world, float3 cameraRay)
{
    float visibility = 1;
    if (LightOptions.y != 0) visibility = PrimaryVisibility(world, ScatteringScale.y, LightOptions.y == 2);
    float3 direct = 0;
    [loop] for (uint index = 0; index < LightOptions.x; ++index)
    {
        Emitter light;
        if (ScatteringScale.z != 0.0) light = VolumeSceneLights[index];
        else light = SceneLights[index];
        float3 power = min(light.ColorShadow.rgb, HalfFloatMaximum) * min(light.DirectionPower.w, HalfFloatMaximum);
        float visible = LightOptions.y != 0 && index == LightOptions.z ? visibility : 1;
        direct += power * (visible * MediumLightIntegral(light, world, cameraRay));
    }
    return AmbientSource.rgb + ScatteringAlbedo.rgb * ScatteringScale.x * direct;
}
float3 MediumWorld(float2 uv, float depth)
{
    float4 world = mul(CameraInverse, float4(2 * uv.x - 1, 1 - 2 * uv.y, depth, 1));
    return world.xyz / world.w;
}
float4 PSMain(float4 fragment : SV_POSITION) : SV_TARGET
{
    uint2 pixel = uint2(fragment.xy);
    uint columns = uint(GridDimensions.z), slice = pixel.x / columns;
    float2 uv = (float2(pixel.x % columns, pixel.y) + PixelCenterOffset) / GridDimensions.zw;
    float3 origin = MediumWorld(uv, 0), ray = MediumWorld(uv, 1) - origin;
    float rayLength = max(length(ray), MediumDistanceFloor);
    float3 direction = ray / rayLength;
    float stepDistance = min(rayLength, MediumDensity.w) / (IntegrationSteps.y * IntegrationSteps.z);
    float4 transport = float4(0, 0, 0, 1);
    [loop] for (uint index = 0; index < IntegrationSteps.z; ++index)
    {
        float distance = (slice * IntegrationSteps.z + index + PixelCenterOffset) * stepDistance;
        float3 world = origin + direction * distance;
        float extinction = MediumDensity.x * exp(-min(MediumDensity.z * max(world.y - MediumDensity.y, 0), OpticalDepthMaximum));
        float transmission = exp(-min(extinction * stepDistance, OpticalDepthMaximum));
        transport.rgb += transport.a * (1 - transmission) * MediumRadiance(world, direction);
        transport.a *= transmission;
    }
    return transport;
}
#endif

#endif // independent moment stages / receivers
