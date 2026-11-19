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

// Lumen froxel irradiance for transparent receivers. Real SH through band two;
// XYZ interpolation is clamped within coefficient bands, never across them.
cbuffer LumenTranslucentReceiver : register(b8)
{
    row_major float4x4 LumenVolumeCamera,LumenVolumeInverse;
    uint4 LumenVolumeGrid;
    float4 LumenVolumeDistribution,LumenVolumeAtlasSize;
};
Texture2D<float4> LumenVolumeImage : register(t19);
SamplerState LumenVolumeSampler : register(s4);
static const float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 worldPosition,float3 normal)
{
    if (LumenVolumeGrid.w==0) return 0;
    float4 clip=mul(LumenVolumeCamera,float4(worldPosition,1));
    if (clip.w<=LumenVolumeAtlasSize.z) return 0;
    float3 ndc=clip.xyz/clip.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return 0;
    float4 nearH=mul(LumenVolumeInverse,float4(ndc.xy,0,1)),farH=mul(LumenVolumeInverse,float4(ndc.xy,1,1));
    if (min(abs(nearH.w),abs(farH.w))<=LumenVolumeAtlasSize.z) return 0;
    float3 nearPoint=nearH.xyz/nearH.w;
    float extent=min(length(farH.xyz/farH.w-nearPoint),LumenVolumeDistribution.x);
    float distance=length(worldPosition-nearPoint);
    if (extent<=LumenVolumeAtlasSize.z || distance>extent) return 0;
    float2 uv=float2(ndc.x+1,1-ndc.y)*0.5;
    float3 cell=float3(uv*float2(LumenVolumeGrid.xy),pow(saturate(distance/extent),1/LumenVolumeDistribution.y)*LumenVolumeGrid.z)-0.5;
    cell=clamp(cell,0,float3(LumenVolumeGrid.xyz)-1);
    float lower=floor(cell.z),upper=min(lower+1,float(LumenVolumeGrid.z-1));
    normal*=rsqrt(max(dot(normal,normal),LumenVolumeAtlasSize.w));
    static const float SpherePi=3.14159265358979323846;
    static const uint CoefficientCount=9;
    float basis[CoefficientCount]={sqrt(1/(4*SpherePi)),sqrt(3/(4*SpherePi))*normal.y,sqrt(3/(4*SpherePi))*normal.z,sqrt(3/(4*SpherePi))*normal.x,
        sqrt(15/(4*SpherePi))*normal.x*normal.y,sqrt(15/(4*SpherePi))*normal.y*normal.z,
        sqrt(5/(16*SpherePi))*(3*normal.z*normal.z-1),sqrt(15/(4*SpherePi))*normal.x*normal.z,
        sqrt(15/(16*SpherePi))*(normal.x*normal.x-normal.y*normal.y)};
    float3 irradiance=0;

    static const uint VolumeVisibilityBit=2;
    if ((LumenVolumeGrid.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(cell)); float3 fraction=frac(cell);
        // Keep the original trilinear mass. Rejected support spheres contribute
        // no light, rather than renormalizing bright probes across an occluder.
        [loop] for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,LumenVolumeGrid.xyz-1);
            float3 weights=float3(bit.x?fraction.x:1-fraction.x,bit.y?fraction.y:1-fraction.y,bit.z?fraction.z:1-fraction.z);
            float weight=weights.x*weights.y*weights.z; if (weight<=0) continue;
            uint row=p.z*LumenVolumeGrid.y+p.y; float support=LumenVolumeImage.Load(int3(p.x,row,0)).a; if (support<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5)/LumenVolumeGrid.x-1,1-2*(float(p.y)+0.5)/LumenVolumeGrid.y);
            float4 a=mul(LumenVolumeInverse,float4(clipXY,0,1)),b=mul(LumenVolumeInverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=LumenVolumeAtlasSize.z) continue;
            float3 start=a.xyz/a.w,axis=b.xyz/b.w-start; float span=length(axis);
            if (!isfinite(span) || span<=LumenVolumeAtlasSize.z) continue;
            float depth=pow((float(p.z)+0.5)/LumenVolumeGrid.z,LumenVolumeDistribution.y);
            float3 probe=start+axis*(depth*min(span,LumenVolumeDistribution.x)/span);
            float separation=length(worldPosition-probe); if (!isfinite(separation) || separation>support) continue;
            [unroll] for (uint coefficient=0;coefficient<CoefficientCount;++coefficient)
                irradiance+=weight*basis[coefficient]*LumenVolumeImage.Load(int3(p.x+coefficient*LumenVolumeGrid.x,row,0)).rgb;
        }
        return max(irradiance,0)*(LumenVolumeDistribution.z/(SpherePi*LumenVolumeDistribution.w));
    }
    [unroll] for (uint coefficient=0;coefficient<CoefficientCount;++coefficient)
    {
        float x=coefficient*LumenVolumeGrid.x+cell.x+0.5;
        float2 a=float2(x,lower*LumenVolumeGrid.y+cell.y+0.5)/LumenVolumeAtlasSize.xy;
        float2 b=float2(x,upper*LumenVolumeGrid.y+cell.y+0.5)/LumenVolumeAtlasSize.xy;
        irradiance+=lerp(LumenVolumeImage.SampleLevel(LumenVolumeSampler,a,0).rgb,
            LumenVolumeImage.SampleLevel(LumenVolumeSampler,b,0).rgb,frac(cell.z))*basis[coefficient];
    }
    return max(irradiance,0)*(LumenVolumeDistribution.z/(SpherePi*LumenVolumeDistribution.w));
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

// Independent DX11 variance shadow maps: caster moments, separable Gaussian
// filtering and Forward/Deferred/volume receivers. No cross-technique includes.
// Donnelly and Lauritzen, "Variance Shadow Maps" (2006); GPU Gems 3, chapter 8.
// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps
// Moments use normalized linear light-axis depth. Each cascade owns a separate
// distribution; evaluate its visibility before interpolating cascade overlap.
#define VSM_PHONG 0
#define VSM_FLAT 1
#define VSM_GOURAUD 2
#define VSM_PBR 3
#define VSM_TOON 4
#define VSM_DEFERRED 5
#define VSM_DEFERRED_MSAA 6
#define VSM_VOLUME 7
#define VSM_CASTER 8
#define VSM_FILTER 9
#ifndef EVANS_VSM_PROGRAM
#define EVANS_VSM_PROGRAM VSM_PHONG
#endif
static const uint LightCapacity = 4, CascadeCapacity = 4, MapCapacity = 5;
static const uint DirectionalLight = 0, PointLight = 1;
static const float Pi = 3.14159265358979323846;
static const float NormalEpsilon = 1.0e-12, ProjectionEpsilon = 1.0e-7;
static const float HalfRadianceLimit = 65504.0, PixelCenter = 0.5;
static const float SurfaceShadowStrength = 0.55;
static const float MinimumCosine = 0.15, ConstantBias = 0.00035, SlopeBias = 0.00060, MaximumBias = 0.0045;
struct Light { float4 PositionType; float4 DirectionIntensity; float4 ColorCastShadows; float4 AreaSize; };

cbuffer VSMConstants : register(b5)
{
    float4 DepthEncoding[MapCapacity]; // clip Z coefficient, clip W coefficient, offset, reserved
    float4 MomentParameters; // variance floor, bleeding reduction, derivative variance cap, Gaussian sigma
    uint4 MomentControl; // valid map bits, blur radius, reserved, filter axis
};

#if EVANS_VSM_PROGRAM >= VSM_CASTER
#if EVANS_VSM_PROGRAM == VSM_CASTER
cbuffer CasterDraw : register(b0)
{
    row_major float4x4 World, NormalWorld, ViewProjection, LightViewProjection;
    float4 PipelineColor, MaterialDiffuseColor, MaterialAmbientColor, MaterialSpecularColor, MaterialEmissiveColor;
    float4 MaterialParameters, AmbientColorStrength, CameraLightCount, RenderParameters;
    uint4 TextureFlags;
};
static const uint MaximumInstances = 64;
struct InstanceData
{
    row_major float4x4 World, NormalWorld, PreviousWorldViewProjection;
    float4 Wind, Gust, VegetationHistory;
};
cbuffer InstanceConstants : register(b3)
{
    row_major float4x4 InstanceViewProjection;
    float4 InstanceMaterial;
    InstanceData Instances[MaximumInstances];
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
float4 VegetationPoint(float3 position,float2 uv,InstanceData data)
{
    if (data.Gust.w<0.5) return float4(position,1);
    float3 world=mul(data.World,float4(position,1)).xyz;
    float height=max(0.0,data.Gust.w<1.5?uv.y:position.y);
    float3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z);
    return float4(position+mul(transpose((float3x3)data.NormalWorld),offset),1);
}
Texture2D<float4> DiffuseMap : register(t1);
Texture2D<float4> OpacityMap : register(t3);
SamplerState SurfaceSampler : register(s0);
struct MeshVertex
{
    float3 Position : POSITION; float3 Normal : NORMAL;
    float4 Color : COLOR0; float2 TexCoord : TEXCOORD0;
};
struct CasterVertex
{
    float4 Position : SV_POSITION;
    float2 UV : TEXCOORD0;
    float LinearDepth : TEXCOORD1;
    float Opacity : COLOR0;
};
CasterVertex Caster(MeshVertex vertex, float4 clip)
{
    CasterVertex output;
    output.Position = clip;
    output.UV = float2(vertex.TexCoord.x, 1 - vertex.TexCoord.y);
    output.LinearDepth = dot(float3(clip.zw, 1), DepthEncoding[0].xyz);
    output.Opacity = vertex.Color.a;
    return output;
}
CasterVertex VSMain(MeshVertex vertex)
{ return Caster(vertex, mul(ViewProjection, mul(World, float4(vertex.Position, 1)))); }
CasterVertex VSInstanced(MeshVertex vertex, uint instance : SV_InstanceID)
{ return Caster(vertex, mul(InstanceViewProjection, mul(Instances[instance].World, VegetationPoint(vertex.Position,vertex.TexCoord,Instances[instance])))); }
float2 PSMain(CasterVertex input) : SV_TARGET
{
    const float RasterStandardDeviation = 0.5; // half a shadow texel
    const float CoverageNoiseScale = 52.9829189;
    const float2 CoverageNoiseDirection = float2(0.06711056, 0.00583715);
    // Derivatives precede coverage discard and represent the current primitive,
    // never finite differences across silhouettes in a stored depth image.
    float depth = saturate(input.LinearDepth);
    float2 slope = float2(ddx(input.LinearDepth), ddy(input.LinearDepth)) * RasterStandardDeviation;
    float variance = min(dot(slope, slope), MomentParameters.z);
    float diffuseAlpha = TextureFlags.x != 0 ? DiffuseMap.Sample(SurfaceSampler, input.UV).a : 1;
    float opacity = TextureFlags.z != 0 ? OpacityMap.Sample(SurfaceSampler, input.UV).r : 1;
    float coverage = saturate(input.Opacity * MaterialParameters.y * diffuseAlpha * opacity);
    if (MaterialParameters.z > 0) clip(coverage - MaterialParameters.z);
    else if (RenderParameters.w > 0.5)
    {
        if (coverage <= 0) discard;
        float threshold = frac(CoverageNoiseScale * frac(dot(floor(input.Position.xy), CoverageNoiseDirection)));
        clip(coverage - threshold);
    }
    return float2(depth, depth * depth + variance);
}
#else
Texture2D<float2> InputMoments : register(t0);
float4 VSMain(uint vertex : SV_VertexID) : SV_POSITION
{
    const float2 Vertices[3] = {float2(-1, -1), float2(-1, 3), float2(3, -1)};
    return float4(Vertices[vertex], 0, 1);
}
float2 PSMain(float4 position : SV_POSITION) : SV_TARGET
{
    const float GaussianExponent = -0.5;
    uint width, height; InputMoments.GetDimensions(width, height);
    int2 size = int2(width, height), center = int2(position.xy);
    int2 direction = MomentControl.w == 0 ? int2(1, 0) : int2(0, 1);
    float2 sum = 0; float totalWeight = 0;
    int radius = int(MomentControl.y);
    [loop] for (int offset = -radius; offset <= radius; ++offset)
    {
        float distanceInSigma = offset / MomentParameters.w;
        float weight = exp(GaussianExponent * distanceInSigma * distanceInSigma);
        int2 pixel = center + offset * direction;
        float2 moments = float2(1, 1);
        if (all(pixel >= 0) && all(pixel < size)) moments = InputMoments.Load(int3(pixel, 0));
        sum += weight * moments; totalWeight += weight;
    }
    return sum / totalWeight;
}
#endif
#else
#if EVANS_VSM_PROGRAM == VSM_VOLUME
// Volume samples use complete scene storage, never opaque surface lists.
StructuredBuffer<Light> VolumeSceneLights : register(t16);
cbuffer MediumConstants : register(b3)
{
    row_major float4x4 InverseViewProjection;
    float4 Medium, AmbientSource, ViewportGrid;
    uint4 GridOptions;
};
cbuffer VolumeLightingConstants : register(b4)
{
    float4 Scattering, LightParameters;
    uint4 Control;
    row_major float4x4 LightViewProjection;
    Light Lights[LightCapacity];
    row_major float4x4 CascadeLightViewProjection[CascadeCapacity];
    float4 CascadeFarDistances, CascadeBlendStarts, CascadeEyeDepthPlane, CascadeParameters, ShadowFiltering;
};
Texture2D<float2> ShadowMap : register(t1);
Texture2D<float2> CascadeMaps[CascadeCapacity] : register(t2);
#else
Texture2D<float2> ShadowMap : register(t0);
Texture2D<float2> CascadeMaps[CascadeCapacity] : register(t10);
cbuffer CascadedShadowConstants : register(b2)
{
    row_major float4x4 CascadeLightViewProjection[CascadeCapacity];
    float4 CascadeFarDistances, CascadeBlendStarts, CascadeEyeDepthPlane, CascadeParameters, ShadowFiltering;
};
#if EVANS_VSM_PROGRAM >= VSM_DEFERRED
cbuffer DeferredConstants : register(b1)
{
    row_major float4x4 InverseViewProjection, LightViewProjection;
    float4 CameraLightCount, ShadingParameters;
    uint4 DeferredOptions;
    float4 BackgroundColor;
    Light Lights[LightCapacity];
};
#if EVANS_VSM_PROGRAM == VSM_DEFERRED_MSAA
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
#else
cbuffer DrawConstants : register(b0)
{
    row_major float4x4 World, NormalWorld, ViewProjection, LightViewProjection;
    float4 PipelineColor, MaterialDiffuseColor, MaterialAmbientColor, MaterialSpecularColor, MaterialEmissiveColor;
    float4 MaterialParameters, AmbientColorStrength, CameraLightCount, RenderParameters;
    uint4 TextureFlags;
    Light Lights[LightCapacity];
    float4 ShadingParameters;
};
Texture2D<float4> DiffuseMap : register(t1);
Texture2D<float4> NormalMap : register(t2);
Texture2D<float4> OpacityMap : register(t3);
SamplerState SurfaceSampler : register(s0);
cbuffer SurfaceFogConstants : register(b3)
{
    row_major float4x4 FogInverseViewProjection;
    float4 FogMedium, FogSourceRadiance, FogViewportGrid;
    uint4 FogOptions;
};
Texture2D<float4> FogAtlas : register(t14);
SamplerState FogLinearClamp : register(s3);
cbuffer WeightedTransparencyConstants : register(b4)
{ float4 TransparencyWeights; uint4 TransparencyOptions; };
#endif
#endif

float3 SafeNormal(float3 value, float3 fallback)
{ float squared = dot(value, value); return squared > NormalEpsilon ? value * rsqrt(squared) : fallback; }
uint2 MapSize(uint map)
{
    uint width = 1, height = 1;
    if (map == 0) ShadowMap.GetDimensions(width, height);
    else CascadeMaps[0].GetDimensions(width, height);
    return uint2(width, height);
}
float2 MomentsAt(uint map, int2 pixel, uint2 size)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return float2(1, 1);
    // Literal resource indices preserve the SM5 texture-array contract.
    if (map == 0) return ShadowMap.Load(int3(pixel, 0));
    if (map == 1) return CascadeMaps[0].Load(int3(pixel, 0));
    if (map == 2) return CascadeMaps[1].Load(int3(pixel, 0));
    if (map == 3) return CascadeMaps[2].Load(int3(pixel, 0));
    return CascadeMaps[3].Load(int3(pixel, 0));
}
float2 BilinearMoments(uint map, float2 uv)
{
    uint2 size = MapSize(map);
    float2 texel = uv * size - PixelCenter;
    int2 base = int2(floor(texel)); float2 weight = frac(texel);
    float2 top = lerp(MomentsAt(map, base, size), MomentsAt(map, base + int2(1, 0), size), weight.x);
    float2 bottom = lerp(MomentsAt(map, base + int2(0, 1), size), MomentsAt(map, base + int2(1, 1), size), weight.x);
    return lerp(top, bottom, weight.y);
}
float MapVisibility(uint map, float4 clip, float bias)
{
    if ((MomentControl.x & (1u << map)) == 0 || clip.w <= ProjectionEpsilon) return 1;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1) return 1;
    float linearDepth = dot(float3(clip.zw, 1), DepthEncoding[map].xyz);
    float2 moments = BilinearMoments(map, uv);
    float distanceBeyondMean = linearDepth - bias - moments.x;
    if (distanceBeyondMean <= 0) return 1;
    float variance = max(moments.y - moments.x * moments.x, MomentParameters.x);
    float upperBound = variance / (variance + distanceBeyondMean * distanceBeyondMean);
    // This artistic remap suppresses bleeding but forfeits the upper-bound guarantee.
    return saturate((upperBound - MomentParameters.y) / (1 - MomentParameters.y));
}
float PrimaryVisibility(float3 world, float bias, bool cascades)
{
    float result = 1;
    if (!cascades) result = MapVisibility(0, mul(LightViewProjection, float4(world, 1)), bias);
    else
    {
        float eyeDistance = dot(CascadeEyeDepthPlane, float4(world, 1));
        uint count = uint(CascadeParameters.z);
        if (eyeDistance >= CascadeParameters.x && eyeDistance <= CascadeParameters.y)
        {
            uint cascade = 0;
            [loop] while (cascade + 1 < count && eyeDistance > CascadeFarDistances[cascade]) ++cascade;
            result = MapVisibility(cascade + 1, mul(CascadeLightViewProjection[cascade], float4(world, 1)), bias);
            float interval = CascadeFarDistances[cascade] - CascadeBlendStarts[cascade];
            if (interval > 0 && eyeDistance >= CascadeBlendStarts[cascade])
            {
                float next = 1;
                if (cascade + 1 < count) next = MapVisibility(cascade + 2,
                    mul(CascadeLightViewProjection[cascade + 1], float4(world, 1)), bias);
                result = lerp(result, next, saturate((eyeDistance - CascadeBlendStarts[cascade]) / interval));
            }
        }
    }
    return result;
}

#if EVANS_VSM_PROGRAM != VSM_VOLUME
float SurfaceVisibility(float3 world, float3 normal, float3 toLight)
{
    float cosine = max(dot(SafeNormal(normal, float3(0, 1, 0)), SafeNormal(toLight, float3(0, 1, 0))), 0);
    float slope = sqrt(max(1 - cosine * cosine, 0)) / max(cosine, MinimumCosine);
    float bias = min(ConstantBias + SlopeBias * slope, MaximumBias);
    float visibility = PrimaryVisibility(world, bias, CascadeParameters.z > 0);
    return 1 - SurfaceShadowStrength * (1 - visibility);
}
struct MaterialSurface
{
    float3 Position, Normal, View, Diffuse, Specular;
    float Roughness, Metallic, Shininess, Bands;
    uint Model;
};
// Separate direct diffuse before screen-space diffusion; ordinary lighting is unchanged.
static const uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly()
{
#if EVANS_VSM_PROGRAM >= VSM_DEFERRED && EVANS_VSM_PROGRAM < VSM_VOLUME
    return DeferredOptions.w == DirectDiffuseOutput;
#else
    return false;
#endif
}
float3 LightSample(MaterialSurface surface, float3 toLight, float3 radiance)
{
    const float MinimumRoughness = 0.045, DielectricReflectance = 0.04;
    const float BrdfEpsilon = 1.0e-6, DistributionEpsilon = 1.0e-12, FresnelExponent = 5;
    const float MinimumBands = 2;
#if EVANS_VSM_PROGRAM < VSM_DEFERRED
    const float MaximumBands = 16;
    if (surface.Model <= VSM_FLAT && dot(toLight, toLight) <= NormalEpsilon) return 0;
#else
    const float MaximumBands = 12;
#endif
    float3 direction = SafeNormal(toLight, surface.Normal);
    float noL = saturate(dot(surface.Normal, direction));
    float3 halfway = SafeNormal(direction + surface.View, surface.Normal);
    float3 result = 0;
    if (surface.Model == VSM_PBR)
    {
        float noV = saturate(dot(surface.Normal, surface.View));
        if (noL > 0 && noV > 0)
        {
            float noH = saturate(dot(surface.Normal, halfway)), voH = saturate(dot(surface.View, halfway));
            float roughness = clamp(surface.Roughness, MinimumRoughness, 1);
            float alpha = roughness * roughness, alphaSquared = alpha * alpha;
            float denominator = noH * noH * (alphaSquared - 1) + 1;
            float distribution = alphaSquared / max(Pi * denominator * denominator, DistributionEpsilon);
            float smithV = 2 * noV / max(noV + sqrt(alphaSquared + (1 - alphaSquared) * noV * noV), BrdfEpsilon);
            float smithL = 2 * noL / max(noL + sqrt(alphaSquared + (1 - alphaSquared) * noL * noL), BrdfEpsilon);
            float3 f0 = lerp(DielectricReflectance.xxx, saturate(surface.Diffuse), surface.Metallic);
            float3 fresnel = f0 + (1 - f0) * pow(1 - voH, FresnelExponent);
            float3 diffuse = surface.Diffuse * (1 - fresnel) * (1 - surface.Metallic) / Pi;
            float3 specular = distribution * smithV * smithL * fresnel / max(4 * noV * noL, BrdfEpsilon);
            result = radiance * noL * (diffuse + (DirectDiffuseOnly() ? 0.0 : specular));
        }
    }
    else
    {
        float specular = noL > 0 ? pow(saturate(dot(surface.Normal, halfway)), max(surface.Shininess, 1)) : 0;
        if (surface.Model == VSM_TOON)
        {
            float bands = clamp(round(surface.Bands), MinimumBands, MaximumBands);
            result = radiance * (surface.Diffuse * ToonResponse(noL, bands) + (DirectDiffuseOnly() ? 0.0 : surface.Specular * ToonHighlight(specular)));
        }
        else result = radiance * (surface.Diffuse * noL + (DirectDiffuseOnly() ? 0.0 : surface.Specular * specular));
    }
    return result;
}
float3 SceneLight(MaterialSurface surface, uint index, bool shadows, int shadowIndex, float contactVisibility)
{
    const float DistanceEpsilon = 1.0e-6, ParallelThreshold = 0.999;
    const int AreaSamplesPerAxis = 2;
    const float AreaSampleOffset = 0.25, AreaSampleWeight = 0.25;
    Light light = Lights[index]; uint type = uint(light.PositionType.w + 0.5);
    float3 shadowDirection = type == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - surface.Position;
    float visibility = shadows && int(index) == shadowIndex && light.ColorCastShadows.w > 0.5 ?
        SurfaceVisibility(surface.Position, surface.Normal, shadowDirection) : 1;
    if (int(index) == shadowIndex && light.ColorCastShadows.w > 0.5)
        visibility = ComposeContact(visibility, contactVisibility);
    float3 radiance = max(light.ColorCastShadows.rgb, 0) * max(light.DirectionIntensity.w, 0) * visibility;
    float3 contribution = 0;
    if (type == DirectionalLight) contribution = LightSample(surface, light.DirectionIntensity.xyz, radiance);
    else if (type == PointLight)
    {
        float3 delta = light.PositionType.xyz - surface.Position;
        contribution = LightSample(surface, delta, radiance * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), DistanceEpsilon));
    }
    else
    {
        float3 emission = SafeNormal(light.DirectionIntensity.xyz, float3(0, -1, 0));
        float3 helper = abs(emission.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 tangent = SafeNormal(cross(helper, emission), float3(1, 0, 0));
        float3 bitangent = SafeNormal(cross(emission, tangent), float3(0, 0, 1));
        // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
        const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
        const float EmitterStratumCenter = 0.5;
        bool physicalEmitter = light.AreaSize.w > 0.0;
        int emitterGrid = physicalEmitter ? int(clamp(light.AreaSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AreaSamplesPerAxis;
        float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSize.x * light.AreaSize.y) : 1.0) / float(emitterGrid * emitterGrid);
        [loop] for (int y = 0; y < emitterGrid; ++y)
            [loop] for (int x = 0; x < emitterGrid; ++x)
            {
                float2 offset = float2(((float(x) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter), ((float(y) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter));
                float3 delta = light.PositionType.xyz + tangent * offset.x * light.AreaSize.x + bitangent * offset.y * light.AreaSize.y - surface.Position;
                float distanceSquared = max(dot(delta, delta), DistanceEpsilon);
                float emitterCosine = max(dot(emission, -delta * rsqrt(distanceSquared)), 0);
                contribution += LightSample(surface, delta, radiance * emitterCosine * emitterSampleWeight * LightRangeWindow(delta, light.AreaSize.z) / distanceSquared);
            }
    }
    return contribution;
}
#endif

#if EVANS_VSM_PROGRAM < VSM_DEFERRED
struct SurfaceVertex
{
    float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0; float3 Normal : TEXCOORD1;
    float4 Color : COLOR0; float4 LightPosition : TEXCOORD2; float2 TexCoord : TEXCOORD3;
#if EVANS_VSM_PROGRAM == VSM_GOURAUD
    float3 DiffuseFront : TEXCOORD4; float3 SpecularFront : TEXCOORD5;
    float3 ShadowDiffuseFront : TEXCOORD6; float3 ShadowSpecularFront : TEXCOORD7;
    float3 DiffuseBack : TEXCOORD8; float3 SpecularBack : TEXCOORD9;
    float3 ShadowDiffuseBack : TEXCOORD10; float3 ShadowSpecularBack : TEXCOORD11;
#endif
};
float3 SurfaceNormal(SurfaceVertex input, bool front)
{
    const float TangentEpsilon = 1.0e-10, ParallelThreshold = 0.999;
#if EVANS_VSM_PROGRAM == VSM_FLAT
    float3 normal = SafeNormal(cross(ddx(input.WorldPosition), ddy(input.WorldPosition)), input.Normal);
    if (dot(normal, input.Normal) < 0) normal = -normal;
#else
    float3 normal = SafeNormal(input.Normal, float3(0, 1, 0));
#endif
    if (!front) normal = -normal;
#if EVANS_VSM_PROGRAM == VSM_FLAT || EVANS_VSM_PROGRAM == VSM_GOURAUD
    return normal;
#else
    if (TextureFlags.y == 0) return normal;
    float3 dx = ddx(input.WorldPosition), dy = ddy(input.WorldPosition);
    float2 tx = ddx(input.TexCoord), ty = ddy(input.TexCoord);
    float determinant = tx.x * ty.y - tx.y * ty.x;
    if (abs(determinant) < TangentEpsilon) return normal;
    float3 tangent = (dx * ty.y - dy * tx.y) / determinant;
    float3 originalBitangent = (-dx * ty.x + dy * tx.x) / determinant;
    float3 helper = abs(normal.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    tangent = SafeNormal(tangent - normal * dot(normal, tangent), SafeNormal(cross(helper, normal), float3(1, 0, 0)));
    float handedness = dot(cross(normal, tangent), originalBitangent) < 0 ? -1 : 1;
    float3 bitangent = SafeNormal(cross(normal, tangent) * handedness, float3(0, 0, 1));
    float3 bump = NormalMap.Sample(SurfaceSampler, input.TexCoord).xyz * 2 - 1;
    if (float(TextureFlags.w) == ToonShadingModel) bump = lerp(float3(0.0, 0.0, 1.0), bump, ToonNormalStrength);
    return SafeNormal(tangent * bump.x + bitangent * bump.y + normal * bump.z, normal);
#endif
}
float3 DecodeSRGB(float3 color)
{
    const float LinearScale = 12.92, Offset = 0.055, Scale = 1.055, Gamma = 2.4, Threshold = 0.04045;
    float3 low = color / LinearScale, high = pow((color + Offset) / Scale, Gamma);
    return float3(color.x <= Threshold ? low.x : high.x, color.y <= Threshold ? low.y : high.y, color.z <= Threshold ? low.z : high.z);
}
float3 FogPosition(float2 uv, float depth)
{
    float4 world = mul(FogInverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    return world.xyz / world.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
static const uint RgbTransportMode = 2;
float4 FogBoundaryTransmission(float2 uv, uint boundary)
{
    if (boundary == 0) return float4(1, 1, 1, 1);
    float2 halfCell = PixelCenter / FogViewportGrid.zw;
    float2 lookup = clamp(uv, halfCell, 1 - halfCell);
    lookup.x = (lookup.x + boundary - 1) / FogOptions.y;
    lookup.y = (lookup.y + 1.0) * 0.5;
    return FogAtlas.SampleLevel(FogLinearClamp, lookup, 0);
}
float4 FogBoundary(float2 uv, uint boundary)
{
    if (boundary == 0) return float4(0, 0, 0, 1);
    float2 halfCell = PixelCenter / FogViewportGrid.zw;
    float2 lookup = clamp(uv, halfCell, 1 - halfCell);
    lookup.x = (lookup.x + boundary - 1) / FogOptions.y;
    if (FogOptions.w == RgbTransportMode) lookup.y *= 0.5;
    return FogAtlas.SampleLevel(FogLinearClamp, lookup, 0);
}
float3 SurfaceFog(float3 color, float4 position)
{
    const float MinimumRayLength = 1.0e-6;
    if (FogOptions.w == 0) return color;
    float2 uv = position.xy / FogViewportGrid.xy;
    float3 start = FogPosition(uv, 0);
    float extent = min(length(FogPosition(uv, 1) - start), FogMedium.w);
    float endpoint = saturate(length(FogPosition(uv, position.z) - start) / max(extent, MinimumRayLength)) * FogOptions.y;
    uint lower = uint(floor(endpoint)), upper = min(lower + 1, FogOptions.y);
    float4 transport = lerp(FogBoundary(uv, lower), FogBoundary(uv, upper), frac(endpoint));
    float3 mediumTransmission = FogOptions.w == RgbTransportMode ? (lerp(FogBoundaryTransmission(uv, lower), FogBoundaryTransmission(uv, upper), frac(endpoint))).rgb : transport.aaa;
    return clamp(color * mediumTransmission + transport.rgb, 0, HalfRadianceLimit);
}
#if defined(EVANS_CONTACT_RECEIVER)
float4 PSContact(SurfaceVertex input, bool front : SV_IsFrontFace, uint contactSample : SV_SampleIndex) : SV_TARGET
#else
float4 PSMain(SurfaceVertex input, bool front : SV_IsFrontFace) : SV_TARGET
#endif
{
#if !defined(EVANS_CONTACT_RECEIVER)
    const uint contactSample = 0u;
#endif
    float4 textureColor = TextureFlags.x != 0 ? DiffuseMap.Sample(SurfaceSampler, input.TexCoord) : 1;
    if (float(TextureFlags.w) == ToonShadingModel && TextureFlags.x != 0u)
        textureColor.rgb = ToonTextureColor(DiffuseMap, SurfaceSampler, input.TexCoord, ddx(input.TexCoord), ddy(input.TexCoord));
    float3 surfaceMap = TextureFlags.z != 0 ? OpacityMap.Sample(SurfaceSampler, input.TexCoord).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = ShadingParameters.xy;
    if (TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    float alpha = saturate(input.Color.a * MaterialParameters.y * textureColor.a * opacity);
    if (MaterialParameters.z > 0) clip(alpha - MaterialParameters.z);
    float3 color = 0;
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (RenderParameters.y > 1.0 && RenderParameters.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (RenderParameters.y > 0.5) color = SurfaceFog(DecodeSRGB(saturate(PipelineColor.rgb)), input.Position);
    else
    {
        float3 base = ToonAlbedo(input.Color.rgb * max(PipelineColor.rgb, 0) * max(textureColor.rgb, 0), float(TextureFlags.w));
        MaterialSurface surface;
        surface.Position = input.WorldPosition; surface.Normal = SurfaceNormal(input, front);
        surface.View = SafeNormal(CameraLightCount.xyz - surface.Position, surface.Normal);
        surface.Diffuse = base * max(MaterialDiffuseColor.rgb, 0); surface.Specular = max(MaterialSpecularColor.rgb, 0);
        surface.Roughness = roughnessMetallic.x; surface.Metallic = saturate(roughnessMetallic.y);
        surface.Shininess = MaterialParameters.x; surface.Bands = ShadingParameters.z; surface.Model = EVANS_VSM_PROGRAM;
        color = ToonAmbient(base * max(MaterialDiffuseColor.rgb, 0.0), base * max(MaterialAmbientColor.rgb, 0) * max(AmbientColorStrength.rgb, 0) * max(AmbientColorStrength.w, 0), float(TextureFlags.w));
#if EVANS_VSM_PROGRAM == VSM_PBR
        color *= 1 - surface.Metallic;
#endif
        color += max(MaterialEmissiveColor.rgb, 0);
#if EVANS_VSM_PROGRAM == VSM_GOURAUD
        float visibility = 1; int shadowIndex = int(RenderParameters.z);
        if (RenderParameters.x > 0.5 && shadowIndex >= 0 && shadowIndex < int(LightCapacity))
        {
            Light light = Lights[shadowIndex];
            float3 direction = uint(light.PositionType.w + 0.5) == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - surface.Position;
            visibility = SurfaceVisibility(surface.Position, surface.Normal, direction);
        }
        visibility = ComposeContact(visibility, ReadContact(input.Position, contactSample));
        float3 diffuse = front ? input.DiffuseFront + visibility * input.ShadowDiffuseFront : input.DiffuseBack + visibility * input.ShadowDiffuseBack;
        float3 specular = front ? input.SpecularFront + visibility * input.ShadowSpecularFront : input.SpecularBack + visibility * input.ShadowSpecularBack;
        color += surface.Diffuse * diffuse + surface.Specular * specular;
#else
        [loop] for (uint index = 0; index < min(uint(CameraLightCount.w + 0.5), LightCapacity); ++index)
            color += SceneLight(surface, index, RenderParameters.x > 0.5, int(RenderParameters.z), ReadContact(input.Position, contactSample));
#endif
        color += surface.Diffuse * (surface.Model==VSM_PBR ? (1-LumenDielectricReflectance)*(1-surface.Metallic) : 1) * LumenTranslucentIrradiance(surface.Position,surface.Normal);
        color = max(SurfaceFog(color, input.Position), 0);
        color = ShadingParameters.w > 0.5 ? min(color, HalfRadianceLimit) : saturate(color / max(1, max(color.r, max(color.g, color.b))));
    }
    return float4(color, alpha);
}
#if defined(EVANS_CONTACT_RECEIVER)
float4 PSMain(SurfaceVertex input, bool front : SV_IsFrontFace) : SV_TARGET
{ return PSContact(input, front, 0u); }
#endif
struct WeightedOutput { float4 Accumulation : SV_Target0; float Revealage : SV_Target1; };
WeightedOutput PSWeighted(SurfaceVertex input, bool front : SV_IsFrontFace)
{
    float4 color = PSMain(input, front); WeightedOutput output = (WeightedOutput)0;
    float alpha = saturate(color.a); if (alpha == 0) return output;
    float weight = max(pow(alpha, TransparencyWeights.x) * pow(max(1 - saturate(input.Position.z) * TransparencyWeights.z, 0), TransparencyWeights.y), TransparencyWeights.w);
    output.Accumulation = float4(color.rgb * alpha * weight, alpha * weight); output.Revealage = alpha;
    return output;
}
#elif EVANS_VSM_PROGRAM != VSM_VOLUME
struct ScreenVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
#if EVANS_VSM_PROGRAM == VSM_DEFERRED_MSAA
float4 PSMain(ScreenVertex input, uint sampleIndex : SV_SampleIndex) : SV_TARGET
#else
float4 PSMain(ScreenVertex input) : SV_TARGET
#endif
{
#if EVANS_VSM_PROGRAM == VSM_DEFERRED_MSAA
    int2 pixel = int2(input.Position.xy);
    float depth = SceneDepth.Load(pixel, sampleIndex);
    float4 diffuse = DiffuseOpacity.Load(pixel, sampleIndex), normal = NormalRoughness.Load(pixel, sampleIndex);
    float4 specular = SpecularShininess.Load(pixel, sampleIndex), ambient = AmbientMetallic.Load(pixel, sampleIndex), emission = EmissionColor.Load(pixel, sampleIndex);
    uint width, height, samples; SceneDepth.GetDimensions(width, height, samples);
    float2 uv = (float2(pixel) + PixelCenter + SceneDepth.GetSamplePosition(sampleIndex)) / float2(width, height);
#else
    int3 pixel = int3(input.Position.xy, 0);
    float depth = SceneDepth.Load(pixel);
    float4 diffuse = DiffuseOpacity.Load(pixel), normal = NormalRoughness.Load(pixel);
    float4 specular = SpecularShininess.Load(pixel), ambient = AmbientMetallic.Load(pixel), emission = EmissionColor.Load(pixel);
    float2 uv = input.UV;
#endif
    if (normal.w < 0) return DirectDiffuseOnly() ? 0.0 : float4(emission.rgb, diffuse.a);
    if (depth >= 1) return DirectDiffuseOnly() ? 0.0 : BackgroundColor;
    float4 world = mul(InverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    MaterialSurface surface;
    surface.Position = world.xyz / world.w; surface.Normal = SafeNormal(normal.xyz, float3(0, 1, 0));
    surface.View = SafeNormal(CameraLightCount.xyz - surface.Position, surface.Normal);
    surface.Diffuse = diffuse.rgb; surface.Specular = specular.rgb;
    surface.Roughness = normal.w; surface.Metallic = saturate(ambient.w); surface.Shininess = specular.w;
    surface.Bands = ShadingParameters.x; surface.Model = DeferredOptions.x;
#if EVANS_VSM_PROGRAM == VSM_DEFERRED_MSAA
    const uint contactSample = sampleIndex;
#else
    const uint contactSample = 0u;
#endif
    float3 color = DirectDiffuseOnly() ? 0.0 : ambient.rgb + emission.rgb;
    [loop] for (uint index = 0; index < min(uint(CameraLightCount.w + 0.5), LightCapacity); ++index)
        color += SceneLight(surface, index, DeferredOptions.y != 0, int(DeferredOptions.z), ReadContact(input.Position, contactSample));
    float coverage = DirectDiffuseOnly() ? (DeferredOptions.x == VSM_PBR ? 1 - surface.Metallic : 1) : diffuse.a;
    if (DirectDiffuseOnly()) { color *= emission.a; coverage *= emission.a; }
    return float4(clamp(color, 0, HalfRadianceLimit), coverage);
}
#else
static const float MaximumOpticalDepth = 80, MinimumRayLength = 1.0e-6;
static const float MinimumLightDistanceSquared = 1.0e-4, VolumeBasisThreshold = 0.99;
float3 VolumePosition(float2 uv, float depth)
{
    float4 world = mul(InverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    return world.xyz / world.w;
}
float3 VolumeDirection(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > MinimumLightDistanceSquared ? value * rsqrt(squared) : fallback;
}
float Phase(float3 toLight, float3 ray)
{
    float cosine = clamp(dot(toLight, ray), -1, 1), g = Scattering.w;
    float denominator = 1 + g * g - 2 * g * cosine;
    return (1 - g * g) / (4 * Pi * denominator * sqrt(denominator));
}
float3 VolumeSource(float3 position, float3 ray)
{
    const uint SingleShadow = 1, CascadedShadow = 2;
    const uint AreaSamplesPerAxis = 2;
    const float AreaSampleOffset = 0.25, AreaSampleWeight = 0.25;
    float visibility = Control.y == SingleShadow || Control.y == CascadedShadow ?
        PrimaryVisibility(position, LightParameters.y, Control.y == CascadedShadow) : 1;
    float3 source = 0;
    [loop] for (uint index = 0; index < Control.x; ++index)
    {
        Light light;
        if (LightParameters.z != 0.0) light = VolumeSceneLights[index];
        else light = Lights[index]; uint type = uint(light.PositionType.w);
        float3 radiance = min(light.ColorCastShadows.rgb, HalfRadianceLimit) * min(light.DirectionIntensity.w, HalfRadianceLimit);
        float shadow = index == Control.z && Control.y != 0 ? visibility : 1;
        float3 direction = VolumeDirection(light.DirectionIntensity.xyz, float3(0, 1, 0));
        float contribution = 0;
        if (type == DirectionalLight) contribution = Phase(direction, ray);
        else if (type == PointLight)
        {
            float3 delta = light.PositionType.xyz - position;
            contribution = Phase(VolumeDirection(delta, direction), ray) * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), MinimumLightDistanceSquared);
        }
        else
        {
            float3 helper = abs(direction.y) < VolumeBasisThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
            float3 tangent = normalize(cross(helper, direction)), bitangent = cross(direction, tangent);
            // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
            const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
            const float EmitterStratumCenter = 0.5;
            bool physicalEmitter = light.AreaSize.w > 0.0;
            uint emitterGrid = physicalEmitter ? uint(clamp(light.AreaSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AreaSamplesPerAxis;
            float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
            [loop] for (uint y = 0; y < emitterGrid; ++y)
                [loop] for (uint x = 0; x < emitterGrid; ++x)
                {
                    float2 offset = float2(((float(x) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter), ((float(y) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter));
                    float3 delta = light.PositionType.xyz + tangent * offset.x * light.AreaSize.x + bitangent * offset.y * light.AreaSize.y - position;
                    float3 toLight = VolumeDirection(delta, direction);
                    contribution += emitterSampleWeight * saturate(dot(direction, -toLight)) * Phase(toLight, ray) * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), MinimumLightDistanceSquared);
                }
            contribution *= light.AreaSize.x * light.AreaSize.y;
        }
        source += radiance * (shadow * contribution);
    }
    return Scattering.rgb * LightParameters.x * source;
}
float4 PSMain(float4 position : SV_POSITION) : SV_TARGET
{
    uint2 pixel = uint2(position.xy);
    uint gridWidth = uint(ViewportGrid.z), slice = pixel.x / gridWidth;
    float2 uv = (float2(pixel.x % gridWidth, pixel.y) + PixelCenter) / ViewportGrid.zw;
    float3 start = VolumePosition(uv, 0), ray = VolumePosition(uv, 1) - start;
    float rayLength = max(length(ray), MinimumRayLength);
    float3 direction = ray / rayLength;
    float stepLength = min(rayLength, Medium.w) / (GridOptions.y * GridOptions.z);
    float4 transport = float4(0, 0, 0, 1);
    [loop] for (uint sampleIndex = 0; sampleIndex < GridOptions.z; ++sampleIndex)
    {
        float distance = (slice * GridOptions.z + sampleIndex + PixelCenter) * stepLength;
        float3 samplePosition = start + direction * distance;
        float extinction = Medium.x * exp(-min(Medium.z * max(samplePosition.y - Medium.y, 0), MaximumOpticalDepth));
        float transmission = exp(-min(extinction * stepLength, MaximumOpticalDepth));
        float3 source = AmbientSource.rgb + VolumeSource(samplePosition, direction);
        transport.rgb += transport.a * source * (1 - transmission);
        transport.a *= transmission;
    }
    return transport;
}
#endif

#endif // receiver / caster / filter
