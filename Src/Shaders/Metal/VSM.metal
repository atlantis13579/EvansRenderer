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

// Independent Metal receiver for the Lumen froxel SH atlas. Fixed angular
// samples make the low-frequency volume stable without borrowing opaque history.
struct LumenVolumeParameters { float4 Camera[4],Inverse[4]; uint4 Grid; float4 Distribution,Atlas; };
float4 lumenVolumeProject(constant float4* rows,float4 value)
{ return float4(dot(rows[0],value),dot(rows[1],value),dot(rows[2],value),dot(rows[3],value)); }
constant float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 point,float3 direction,
    constant LumenVolumeParameters& volume,texture2d<float> image)
{
    if (volume.Grid.w==0) return 0;
    float4 projected=lumenVolumeProject(volume.Camera,float4(point,1));
    if (projected.w<=volume.Atlas.z) return 0;
    float3 ndc=projected.xyz/projected.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return 0;
    float4 nearH=lumenVolumeProject(volume.Inverse,float4(ndc.xy,0,1)),farH=lumenVolumeProject(volume.Inverse,float4(ndc.xy,1,1));
    if (min(abs(nearH.w),abs(farH.w))<=volume.Atlas.z) return 0;
    float3 origin=nearH.xyz/nearH.w;
    float extent=min(length(farH.xyz/farH.w-origin),volume.Distribution.x),travel=length(point-origin);
    if (extent<=volume.Atlas.z || travel>extent) return 0;
    float2 uv=float2(ndc.x+1,1-ndc.y)*0.5;
    float3 cell=float3(uv*float2(volume.Grid.xy),pow(saturate(travel/extent),1/volume.Distribution.y)*volume.Grid.z)-0.5;
    cell=clamp(cell,float3(0),float3(volume.Grid.xyz)-1);
    float lower=floor(cell.z),upper=min(lower+1,float(volume.Grid.z-1));
    float3 n=direction*rsqrt(max(dot(direction,direction),volume.Atlas.w));
    constexpr float SHPi=3.14159265358979323846;
    constexpr uint SHCount=9;
    float basis[SHCount]={sqrt(1/(4*SHPi)),sqrt(3/(4*SHPi))*n.y,sqrt(3/(4*SHPi))*n.z,sqrt(3/(4*SHPi))*n.x,
        sqrt(15/(4*SHPi))*n.x*n.y,sqrt(15/(4*SHPi))*n.y*n.z,sqrt(5/(16*SHPi))*(3*n.z*n.z-1),
        sqrt(15/(4*SHPi))*n.x*n.z,sqrt(15/(16*SHPi))*(n.x*n.x-n.y*n.y)};
    constexpr sampler volumeFilter(coord::normalized,address::clamp_to_edge,filter::linear);
    float3 sum=0;

    constexpr uint VolumeVisibilityBit=2;
    if ((volume.Grid.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(cell)); float3 fraction=fract(cell);
        // A corner may contribute only inside its surface-free support sphere.
        // Preserve the original weights so rejected corners cannot add energy.
        for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,volume.Grid.xyz-1);
            float weight=(bit.x?fraction.x:1-fraction.x)*(bit.y?fraction.y:1-fraction.y)*(bit.z?fraction.z:1-fraction.z); if (weight<=0) continue;
            uint row=p.z*volume.Grid.y+p.y; float support=image.read(uint2(p.x,row)).w; if (support<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5f)/volume.Grid.x-1,1-2*(float(p.y)+0.5f)/volume.Grid.y);
            float4 a=lumenVolumeProject(volume.Inverse,float4(clipXY,0,1)),b=lumenVolumeProject(volume.Inverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=volume.Atlas.z) continue;
            float3 begin=a.xyz/a.w,axis=b.xyz/b.w-begin; float span=length(axis);
            if (!isfinite(span) || span<=volume.Atlas.z) continue;
            float depth=pow((float(p.z)+0.5f)/volume.Grid.z,volume.Distribution.y);
            float3 probe=begin+axis*(depth*min(span,volume.Distribution.x)/span); float separation=length(point-probe);
            if (!isfinite(separation) || separation>support) continue;
            for (uint coefficient=0;coefficient<SHCount;++coefficient)
                sum+=weight*basis[coefficient]*image.read(uint2(p.x+coefficient*volume.Grid.x,row)).xyz;
        }
        return max(sum,0.0f)*volume.Distribution.z/(SHPi*volume.Distribution.w);
    }
    for (uint coefficient=0;coefficient<SHCount;++coefficient)
    {
        float x=coefficient*volume.Grid.x+cell.x+0.5;
        float2 a=float2(x,lower*volume.Grid.y+cell.y+0.5)/volume.Atlas.xy;
        float2 b=float2(x,upper*volume.Grid.y+cell.y+0.5)/volume.Atlas.xy;
        sum+=basis[coefficient]*mix(image.sample(volumeFilter,a,level(0)).xyz,image.sample(volumeFilter,b,level(0)).xyz,cell.z-lower);
    }
    return max(sum,0.0f)*volume.Distribution.z/(SHPi*volume.Distribution.w);
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

// Independent Metal VSM: linear caster moments, Gaussian prefiltering and
// variance visibility. Native row arrays preserve the host upload ABI.
// Donnelly and Lauritzen, "Variance Shadow Maps" (2006); Lauritzen, GPU Gems 3:
// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps
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
constant uint MomentProjectionCount = 5;
struct VSMConstants
{
    float4 DepthEncoding[MomentProjectionCount];
    float4 Parameters; // variance floor, bleeding reduction, derivative cap, sigma
    uint4 Control; // valid projection mask, radius, reserved, filter axis
};
static_assert(sizeof(VSMConstants) == 112, "VSM upload layout mismatch");

#if EVANS_VSM_PROGRAM == MOMENT_CASTER
struct CasterVertex { packed_float3 Position; packed_float3 Normal; packed_float4 Color; packed_float2 UV; };
struct CasterDraw
{
    float4 World[4], NormalWorld[4], ViewProjection[4], LightViewProjection[4];
    float4 PipelineColor, MaterialDiffuseColor, MaterialAmbientColor, MaterialSpecularColor, MaterialEmissiveColor;
    float4 MaterialParameters, AmbientColorStrength, CameraLightCount, RenderParameters;
    uint4 TextureFlags;
};
constant uint CasterInstanceCount = 64;
struct CasterObject { float4 World[4], NormalWorld[4], PreviousClip[4]; float4 Wind, Gust, VegetationHistory; };
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
float4 VegetationPoint(float3 position,float2 uv,constant CasterObject& data)
{
    if (data.Gust.w<0.5) return float4(position,1);
    float4 local=float4(position,1);
    float3 world=float3(dot(data.World[0],local),dot(data.World[1],local),dot(data.World[2],local));
    float height=max(0.0f,data.Gust.w<1.5?uv.y:position.y);
    float3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z);
    return float4(position+data.NormalWorld[0].xyz*offset.x+data.NormalWorld[1].xyz*offset.y+data.NormalWorld[2].xyz*offset.z,1);
}
struct CasterBatch { float4 Camera[4]; float4 Material; CasterObject Objects[CasterInstanceCount]; };
struct MomentVaryings
{
    float4 Position [[position]];
    float2 UV;
    float VertexAlpha;
    float LinearDepth;
};
float4 CasterTransform(constant float4* rows, float4 point)
{
    return float4(dot(rows[0], point), dot(rows[1], point), dot(rows[2], point), dot(rows[3], point));
}
MomentVaryings MakeMomentVertex(device const CasterVertex& input, float4 clip, constant VSMConstants& moments)
{
    MomentVaryings output;
    output.Position = clip;
    output.UV = float2(input.UV.x, 1.0 - input.UV.y);
    output.VertexAlpha = input.Color.w;
    // Clip depth is encoded before perspective division; the varying then
    // interpolates linear light-axis distance across the rasterized primitive.
    output.LinearDepth = dot(moments.DepthEncoding[0].xyz, float3(clip.zw, 1.0));
    return output;
}
vertex MomentVaryings MomentCasterVertex(device const CasterVertex* vertices [[buffer(0)]],
    constant CasterDraw& draw [[buffer(1)]], constant VSMConstants& moments [[buffer(6)]], uint index [[vertex_id]])
{
    float4 world = CasterTransform(draw.World, float4(float3(vertices[index].Position), 1.0));
    return MakeMomentVertex(vertices[index], CasterTransform(draw.ViewProjection, world), moments);
}
vertex MomentVaryings MomentInstanceVertex(device const CasterVertex* vertices [[buffer(0)]],
    constant CasterBatch& batch [[buffer(3)]], constant VSMConstants& moments [[buffer(6)]],
    uint index [[vertex_id]], uint instance [[instance_id]])
{
    float4 world = CasterTransform(batch.Objects[instance].World, VegetationPoint(float3(vertices[index].Position),float2(vertices[index].UV),batch.Objects[instance]));
    return MakeMomentVertex(vertices[index], CasterTransform(batch.Camera, world), moments);
}
fragment float2 MomentCasterFragment(MomentVaryings input [[stage_in]],
    constant CasterDraw& draw [[buffer(1)]], constant VSMConstants& moments [[buffer(6)]],
    texture2d<float> diffuse [[texture(1)]], texture2d<float> opacity [[texture(3)]], sampler materialSampler [[sampler(0)]])
{
    const float FootprintDeviation = 0.5, EnabledThreshold = 0.5;
    const float CoverageNoiseScale = 52.9829189;
    const float2 CoverageNoiseDirection = float2(0.06711056, 0.00583715);
    // Derivatives precede coverage discard and never cross stored silhouettes.
    float2 gradient = float2(dfdx(input.LinearDepth), dfdy(input.LinearDepth)) * FootprintDeviation;
    float rasterVariance = min(dot(gradient, gradient), moments.Parameters.z);
    float alpha = input.VertexAlpha * draw.MaterialParameters.y;
    if (draw.TextureFlags.x != 0u) alpha *= diffuse.sample(materialSampler, input.UV).a;
    if (draw.TextureFlags.z != 0u) alpha *= opacity.sample(materialSampler, input.UV).r;
    alpha = saturate(alpha);
    if (draw.MaterialParameters.z > 0.0)
    {
        if (alpha < draw.MaterialParameters.z) discard_fragment();
    }
    else if (draw.RenderParameters.w > EnabledThreshold)
    {
        float threshold = fract(CoverageNoiseScale * fract(dot(floor(input.Position.xy), CoverageNoiseDirection)));
        if (alpha <= 0.0 || alpha < threshold) discard_fragment();
    }
    float depth = saturate(input.LinearDepth);
    return float2(depth, depth * depth + rasterVariance);
}
#elif EVANS_VSM_PROGRAM == MOMENT_FILTER
struct FilterVaryings { float4 Position [[position]]; };
vertex FilterVaryings MomentFilterVertex(uint index [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 corner = index == 1 ? float2(FullscreenExtent, -1.0) :
        index == 2 ? float2(-1.0, FullscreenExtent) : float2(-1.0);
    return {float4(corner, 0.0, 1.0)};
}
fragment float2 MomentFilterFragment(FilterVaryings input [[stage_in]], constant VSMConstants& settings [[buffer(0)]],
    texture2d<float, access::read> source [[texture(0)]])
{
    const float GaussianExponent = -0.5;
    int2 center = int2(input.Position.xy), extent = int2(source.get_width(), source.get_height());
    int2 direction = settings.Control.w == 0u ? int2(1, 0) : int2(0, 1);
    float2 sum = float2(0.0); float totalWeight = 0.0;
    int radius = int(settings.Control.y);
    for (int offset = -radius; offset <= radius; ++offset)
    {
        float normalizedOffset = float(offset) / settings.Parameters.w;
        float weight = exp(GaussianExponent * normalizedOffset * normalizedOffset);
        int2 pixel = center + offset * direction;
        float2 value = float2(1.0);
        if (all(pixel >= int2(0)) && all(pixel < extent)) value = source.read(uint2(pixel)).rg;
        sum += value * weight; totalWeight += weight;
    }
    return sum / totalWeight;
}
#else
float2 ReadMoment(texture2d<float, access::read> image, int2 pixel)
{
    if (any(pixel < int2(0)) || any(pixel >= int2(image.get_width(), image.get_height()))) return float2(1.0);
    return image.read(uint2(pixel)).rg;
}
float2 BilinearMoments(texture2d<float, access::read> image, float2 uv)
{
    const float TexelCenter = 0.5;
    float2 center = uv * float2(image.get_width(), image.get_height()) - TexelCenter;
    int2 corner = int2(floor(center)); float2 weight = fract(center);
    float2 upper = mix(ReadMoment(image, corner), ReadMoment(image, corner + int2(1, 0)), weight.x);
    float2 lower = mix(ReadMoment(image, corner + int2(0, 1)), ReadMoment(image, corner + int2(1)), weight.x);
    return mix(upper, lower, weight.y);
}
float VarianceMap(texture2d<float, access::read> image, uint projectionIndex, float4 clip,
    float comparisonBias, constant VSMConstants& vsm)
{
    const float ProjectionTolerance = 1.0e-7;
    if ((vsm.Control.x & (1u << projectionIndex)) == 0u || clip.w <= ProjectionTolerance) return 1.0;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 || any(uv < 0.0) || any(uv > 1.0)) return 1.0;
    float depth = dot(vsm.DepthEncoding[projectionIndex].xyz, float3(clip.zw, 1.0));
    float2 moments = BilinearMoments(image, uv);
    float difference = depth - comparisonBias - moments.x;
    if (difference <= 0.0) return 1.0;
    float variance = max(moments.y - moments.x * moments.x, vsm.Parameters.x);
    float bound = variance / (variance + difference * difference);
    // This artistic remap suppresses bleeding but sacrifices the upper bound.
    return saturate((bound - vsm.Parameters.y) / (1.0 - vsm.Parameters.y));
}

#if EVANS_VSM_PROGRAM < DEFERRED_RECEIVER
constant uint MaximumLights = 4;

constant float ShadowStrength = 0.55;
constant float Pi = 3.141592653589793;
constant float MinimumRoughness = 0.045;
constant float MinimumCosine = 0.0001;
constant float DielectricReflectance = 0.04;
constant float SrgbLinearScale = 12.92;
constant float SrgbPowerOffset = 0.055;
constant float SrgbPowerScale = 1.055;
constant float SrgbExponent = 2.4;
constant float SrgbLinearThreshold = 0.04045;
constant float NormalizeEpsilon = 1.0e-12;
constant float HomogeneousEpsilon = 1.0e-7;
constant float UvDeterminantEpsilon = 1.0e-10;
constant float DistanceEpsilon = 1.0e-6;
constant float ParallelAxisThreshold = 0.999;
constant float OutlineWorldScale = 0.01;
constant float MinimumShadowCosine = 0.15;
constant float MaximumReceiverBias = 0.0045;
constant float ReceiverBias = 0.00035;
constant float ReceiverSlopeBias = 0.00060;
constant float AreaSampleWeight = 0.25;
constant float OutlineMode = 1.5;
constant int DirectionalLight = 0;
constant int PointLight = 1;
constant int RectangularLight = 2;
constant int AreaGridSize = 2;


constant float DistributionEpsilon = 1.0e-12;
constant float BrdfEpsilon = 1.0e-6;
struct LightConstants
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};

struct DrawConstants
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
    LightConstants Lights[MaximumLights];
    float4 ShadingParameters;
};

#if EVANS_VSM_PROGRAM == GOURAUD_RECEIVER
struct VertexOutput
{
    float4 Position [[position]];
    float3 WorldPosition;
    float3 Normal;
    float4 Color;
    float4 LightPosition;
    float2 TexCoord;
    float3 DiffuseFront;
    float3 DiffuseBack;
    float3 SpecularFront;
    float3 SpecularBack;
    float3 ShadowDiffuseFront;
    float3 ShadowSpecularFront;
    float3 ShadowDiffuseBack;
    float3 ShadowSpecularBack;
};

#else
struct VertexOutput
{
    float4 Position [[position]];
    float3 WorldPosition;
    float3 Normal;
    float4 Color;
    float4 LightPosition;
    float2 TexCoord;
};

#endif
float4 TransformRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value),
        dot(rows[2], value), dot(rows[3], value));
}

float3 SRGBToLinear(float3 color)
{
    float3 low = color / SrgbLinearScale;
    float3 high = pow((color + SrgbPowerOffset) / SrgbPowerScale, SrgbExponent);
    return float3(
        color.r <= SrgbLinearThreshold ? low.r : high.r,
        color.g <= SrgbLinearThreshold ? low.g : high.g,
        color.b <= SrgbLinearThreshold ? low.b : high.b);
}

float3 PrepareDisplayLinear(float3 color)
{
    color = max(color, 0.0);
    color /= max(1.0, max(color.r, max(color.g, color.b)));
    return saturate(color);
}

float3 NormalizeSafe(float3 value, float3 fallbackValue)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalizeEpsilon
        ? value * rsqrt(lengthSquared)
        : fallbackValue;
}

float4 SampleDiffuse(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> diffuseMap,
    sampler surfaceSampler)
{
    return constants.TextureFlags.x != 0
        ? diffuseMap.sample(surfaceSampler, input.TexCoord)
        : float4(1.0);
}

float SampleOpacity(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> opacityMap,
    sampler surfaceSampler)
{
    return constants.TextureFlags.z != 0
        ? opacityMap.sample(surfaceSampler, input.TexCoord).r
        : 1.0;
}

float SurfaceAlpha(
    VertexOutput input,
    constant DrawConstants& constants,
    float4 diffuseSample,
    texture2d<float> opacityMap,
    sampler surfaceSampler)
{
    return saturate(
        input.Color.a *
        constants.MaterialParameters.y *
        diffuseSample.a *
        SampleOpacity(input, constants, opacityMap, surfaceSampler));
}

void ApplyAlphaClip(float alpha, constant DrawConstants& constants)
{
    if (constants.MaterialParameters.z > 0.0 &&
        alpha < constants.MaterialParameters.z)
        discard_fragment();
}

// Each light matrix is four CPU row vectors; the full block matches the
// renderer's 336-byte shadow receiver contract. Depth textures use top-left coordinates.
constant uint CsmCascadeLimit = 4;
struct CascadeConstants
{
    float4 LightRows[CsmCascadeLimit][4];
    float4 FarDepths;
    float4 BlendDepths;
    float4 EyeDepthPlane;
    float4 Parameters;
    float4 Filtering; // Preserves the shared cascade ABI; VSM owns its sampling controls.
};
float SurfaceComparisonBias(float3 normal, float3 towardLight)
{
    float squaredLength = dot(towardLight, towardLight);
    float3 direction = squaredLength > NormalizeEpsilon ? towardLight * rsqrt(squaredLength) : normal;
    float cosine = saturate(dot(normal, direction));
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumShadowCosine);
    return min(ReceiverBias + ReceiverSlopeBias * slope, MaximumReceiverBias);
}
float CsmSample(uint index, float3 position, float bias, constant CascadeConstants& cascades,
    texture2d<float, access::read> first, texture2d<float, access::read> second,
    texture2d<float, access::read> third, texture2d<float, access::read> fourth,
    constant VSMConstants& vsm)
{
    float4 clip = TransformRows(cascades.LightRows[index], float4(position, 1.0));
    if (index == 0) return VarianceMap(first, 1, clip, bias, vsm);
    if (index == 1) return VarianceMap(second, 2, clip, bias, vsm);
    if (index == 2) return VarianceMap(third, 3, clip, bias, vsm);
    return VarianceMap(fourth, 4, clip, bias, vsm);
}
float CsmVisibility(float3 position, float3 normal, float3 directionToLight, constant CascadeConstants& cascades,
    texture2d<float, access::read> first, texture2d<float, access::read> second,
    texture2d<float, access::read> third, texture2d<float, access::read> fourth,
    constant VSMConstants& vsm)
{
    float eyeDepth = dot(cascades.EyeDepthPlane, float4(position, 1.0));
    uint count = uint(cascades.Parameters.z);
    if (count == 0 || eyeDepth < cascades.Parameters.x || eyeDepth > cascades.Parameters.y) return 1.0;
    uint interval = 0;
    while (interval + 1 < count && eyeDepth > cascades.FarDepths[interval]) ++interval;
    float bias = SurfaceComparisonBias(normal, directionToLight);
    float visibility = CsmSample(interval, position, bias, cascades, first, second, third, fourth, vsm);
    float overlap = cascades.FarDepths[interval] - cascades.BlendDepths[interval];
    if (overlap > DistanceEpsilon && eyeDepth > cascades.BlendDepths[interval])
    {
        float neighbor = interval + 1 < count ? CsmSample(interval + 1, position, bias, cascades, first, second, third, fourth, vsm) : 1.0;
        visibility = mix(visibility, neighbor, saturate((eyeDepth - cascades.BlendDepths[interval]) / overlap));
    }
    return 1.0 - ShadowStrength * (1.0 - visibility);
}
float SampleShadow(VertexOutput input, constant DrawConstants& constants,
    texture2d<float, access::read> shadowMap, float3 normal,
    float3 directionToLight, constant VSMConstants& vsm)
{
    if (constants.RenderParameters.x < 0.5) return 1.0;
    float visible = VarianceMap(shadowMap, 0, input.LightPosition, SurfaceComparisonBias(normal, directionToLight), vsm);
    return 1.0 - ShadowStrength * (1.0 - visible);
}
#if EVANS_VSM_PROGRAM == FLAT_RECEIVER
float3 ResolveSurfaceNormal(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> normalMap,
    sampler surfaceSampler,
    bool frontFacing)
{
    float3 referenceNormal = NormalizeSafe(input.Normal, float3(0.0, 1.0, 0.0));
    float3 planeNormal = NormalizeSafe(cross(dfdx(input.WorldPosition), dfdy(input.WorldPosition)), referenceNormal);
    if (dot(planeNormal, referenceNormal) < 0.0) planeNormal = -planeNormal;
    return frontFacing ? planeNormal : -planeNormal;
}

#else
float3 ResolveSurfaceNormal(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> normalMap,
    sampler surfaceSampler,
    bool frontFacing)
{
    float3 normal = NormalizeSafe(input.Normal, float3(0.0, 1.0, 0.0));
    if (!frontFacing) normal = -normal;
    if (constants.TextureFlags.y == 0) return normal;
    float3 positionDx = dfdx(input.WorldPosition);
    float3 positionDy = dfdy(input.WorldPosition);
    float2 texCoordDx = dfdx(input.TexCoord);
    float2 texCoordDy = dfdy(input.TexCoord);
    float determinant =
        texCoordDx.x * texCoordDy.y -
        texCoordDx.y * texCoordDy.x;
    if (abs(determinant) < UvDeterminantEpsilon) return normal;
    float inverseDeterminant = 1.0 / determinant;
    float3 tangent =
        (positionDx * texCoordDy.y - positionDy * texCoordDx.y) *
        inverseDeterminant;
    float3 sourceBitangent =
        (-positionDx * texCoordDy.x + positionDy * texCoordDx.x) *
        inverseDeterminant;
    tangent -= normal * dot(normal, tangent);
    float3 helper = abs(normal.y) < ParallelAxisThreshold
        ? float3(0.0, 1.0, 0.0)
        : float3(1.0, 0.0, 0.0);
    tangent = NormalizeSafe(
        tangent,
        NormalizeSafe(cross(helper, normal), float3(1.0, 0.0, 0.0)));
    float handedness =
        dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    float3 bitangent = NormalizeSafe(
        cross(normal, tangent) * handedness,
        float3(0.0, 0.0, 1.0));
    float3 tangentNormal =
        normalMap.sample(surfaceSampler, input.TexCoord).xyz * 2.0 - 1.0;
    if (constants.TextureFlags.w == ToonShadingModel) tangentNormal = mix(float3(0.0, 0.0, 1.0), tangentNormal, ToonNormalStrength);
    return NormalizeSafe(
        tangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

#endif
#if EVANS_VSM_PROGRAM == PBR_RECEIVER
void AccumulateLightSample(float2 roughnessMetallic,
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= NormalizeEpsilon) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = saturate(dot(normal, direction));
    if (normalToLight <= 0.0) return;
    float3 visibleRadiance = radiance * visibility;
    float3 halfway = NormalizeSafe(direction + viewDirection, normal);
    float nDotV = saturate(dot(normal, viewDirection));
    if (nDotV <= 0.0) return;
    float nDotH = saturate(dot(normal, halfway));
    float vDotH = saturate(dot(viewDirection, halfway));
    float roughness = clamp(roughnessMetallic.x, MinimumRoughness, 1.0);
    float metallic = saturate(roughnessMetallic.y);
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float ggxDenominator = nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
    float distribution = alphaSquared / max(Pi * ggxDenominator * ggxDenominator, DistributionEpsilon);
    float masking = 2.0 * nDotV / (nDotV + sqrt(alphaSquared + (1.0 - alphaSquared) * nDotV * nDotV));
    float shadowing = 2.0 * normalToLight / (normalToLight + sqrt(alphaSquared + (1.0 - alphaSquared) * normalToLight * normalToLight));
    float3 baseReflectance = mix(float3(DielectricReflectance), saturate(albedo), metallic);
    const float SchlickExponent = 5.0;
    float3 fresnel = baseReflectance + (1.0 - baseReflectance) * pow(1.0 - vDotH, SchlickExponent);
    const float MicrofacetNormalization = 4.0;
    float3 specular = fresnel * distribution * masking * shadowing /
        max(MicrofacetNormalization * nDotV * normalToLight, BrdfEpsilon);
    diffuseLight += visibleRadiance * (1.0 - fresnel) * (1.0 - metallic) * normalToLight / Pi;
    specularLight += visibleRadiance * specular * normalToLight;
}

#elif EVANS_VSM_PROGRAM == TOON_RECEIVER
void AccumulateLightSample(float2 roughnessMetallic,
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= NormalizeEpsilon) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = max(dot(normal, direction), 0.0);
    float3 visibleRadiance = radiance * visibility;
    float bands = max(constants.ShadingParameters.z, 2.0);
    float3 toonResponse = ToonResponse(normalToLight, bands);
    diffuseLight += visibleRadiance * toonResponse;
    float3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0),
        max(constants.MaterialParameters.x, 1.0));

    specularLight += visibleRadiance * ToonHighlight(specularAmount);
}

#else
void AccumulateLightSample(float2 roughnessMetallic,
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= NormalizeEpsilon) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = max(dot(normal, direction), 0.0);
    if (normalToLight <= 0.0) return;
    float3 visibleRadiance = radiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    float3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0),
        max(constants.MaterialParameters.x, 1.0));
    specularLight += visibleRadiance * specularAmount;
}

#endif
void AccumulateSceneLight(float2 roughnessMetallic,
    uint lightIndex,
    VertexOutput input,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    texture2d<float, access::read> shadowMap,
    bool useCascades, float cascadeVisibility, float contactVisibility,
    constant VSMConstants& vsm,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    LightConstants light = constants.Lights[lightIndex];
    uint type = uint(light.PositionType.w + 0.5);
    float3 shadowDirection = type == DirectionalLight
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - input.WorldPosition;
    bool isShadowLight =
        int(lightIndex) == int(constants.RenderParameters.z) &&
        light.ColorCastShadows.w > 0.5;
    float visibility = isShadowLight
        ? (useCascades ? cascadeVisibility : SampleShadow(
            input, constants, shadowMap, normal, shadowDirection, vsm))
        : 1.0;
    if (isShadowLight && contactVisibility < 1.0)
        visibility = mix(1.0 - ShadowStrength, visibility, contactVisibility);
    float3 radiance =
        max(light.ColorCastShadows.rgb, 0.0) *
        max(light.DirectionIntensity.w, 0.0);
    if (type == DirectionalLight)
    {
        AccumulateLightSample(roughnessMetallic,
            light.DirectionIntensity.xyz, radiance, visibility,
            normal, viewDirection, albedo, constants, diffuseLight, specularLight);
        return;
    }
    if (type == PointLight)
    {
        float3 toLight = light.PositionType.xyz - input.WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), DistanceEpsilon);
        AccumulateLightSample(roughnessMetallic,
            toLight, radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared, visibility,
            normal, viewDirection, albedo, constants, diffuseLight, specularLight);
        return;
    }
    float3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz, float3(0.0, -1.0, 0.0));
    float3 helper = abs(emissionDirection.y) < ParallelAxisThreshold
        ? float3(0.0, 1.0, 0.0)
        : float3(1.0, 0.0, 0.0);
    float3 tangent = NormalizeSafe(
        cross(helper, emissionDirection), float3(1.0, 0.0, 0.0));
    float3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent), float3(0.0, 0.0, 1.0));
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSizePadding.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaSizePadding.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AreaGridSize;
    float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSizePadding.x * light.AreaSizePadding.y) : 1.0) / float(emitterGrid * emitterGrid);
    for (int sampleY = 0; sampleY < emitterGrid; ++sampleY)
    {
        for (int sampleX = 0; sampleX < emitterGrid; ++sampleX)
        {
            float offsetX = ((float(sampleX) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter);
            float offsetY = ((float(sampleY) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter);
            float3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            float3 toLight = samplePosition - input.WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), DistanceEpsilon);
            float3 direction = toLight * rsqrt(distanceSquared);
            float emitterCosine =
                max(dot(emissionDirection, -direction), 0.0);
            AccumulateLightSample(roughnessMetallic,
                direction,
                radiance * (emitterSampleWeight * emitterCosine * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared),
                visibility, normal, viewDirection, albedo, constants,
                diffuseLight, specularLight);
        }
    }
}

// Transparent fragments use their own depth, then ordinary source-over blends
// the fogged radiance. The standalone VolumetricFog source produces the atlas.
struct SurfaceFogParameters
{
    float4 InverseCameraRows[4];
    float4 Medium;
    float4 SourceRadiance;
    float4 ViewAndCells;
    uint4 Control;
};
constant float FogLookupCenter = 0.5;
constant float FogLookupRayEpsilon = 1.0e-6;
float3 SurfaceFogPoint(float2 uv, float depth, constant SurfaceFogParameters& fog)
{
    float4 ndc = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), depth, 1.0);
    float4 world = float4(dot(fog.InverseCameraRows[0], ndc), dot(fog.InverseCameraRows[1], ndc),
        dot(fog.InverseCameraRows[2], ndc), dot(fog.InverseCameraRows[3], ndc));
    return world.xyz / world.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
constant uint RgbTransportMode = 2;
float4 SurfaceFogPrefixTransmission(float2 uv, uint boundary, constant SurfaceFogParameters& fog, texture2d<float> atlas)
{
    if (boundary == 0) return float4(1, 1, 1, 1);
    float2 margin = FogLookupCenter / fog.ViewAndCells.zw;
    float2 packed = clamp(uv, margin, 1.0 - margin);
    packed.x = (float(boundary - 1) + packed.x) / float(fog.Control.y);
    constexpr sampler AtlasSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    packed.y = (packed.y + 1.0) * 0.5;
    return atlas.sample(AtlasSampler, packed, level(0.0));
}
float4 SurfaceFogPrefix(float2 uv, uint boundary, constant SurfaceFogParameters& fog, texture2d<float> atlas)
{
    if (boundary == 0) return float4(0.0, 0.0, 0.0, 1.0);
    float2 margin = FogLookupCenter / fog.ViewAndCells.zw;
    float2 packed = clamp(uv, margin, 1.0 - margin);
    packed.x = (float(boundary - 1) + packed.x) / float(fog.Control.y);
    constexpr sampler AtlasSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    if (fog.Control.w == RgbTransportMode) packed.y *= 0.5;
    return atlas.sample(AtlasSampler, packed, level(0.0));
}
float3 ApplySurfaceFog(float3 radiance, float3 fragmentPosition,
    constant SurfaceFogParameters& fog, texture2d<float> atlas)
{
    if (fog.Control.w == 0) return radiance;
    float2 uv = fragmentPosition.xy / fog.ViewAndCells.xy;
    float3 start = SurfaceFogPoint(uv, 0.0, fog);
    float extent = min(length(SurfaceFogPoint(uv, 1.0, fog) - start), fog.Medium.w);
    float distance = length(SurfaceFogPoint(uv, fragmentPosition.z, fog) - start);
    float boundary = clamp(distance / max(extent, FogLookupRayEpsilon), 0.0, 1.0) * float(fog.Control.y);
    uint low = uint(boundary), high = min(low + 1, fog.Control.y);
    float4 medium = mix(SurfaceFogPrefix(uv, low, fog, atlas), SurfaceFogPrefix(uv, high, fog, atlas), boundary - float(low));
    float3 mediumTransmission = fog.Control.w == RgbTransportMode ? (mix(SurfaceFogPrefixTransmission(uv, low, fog, atlas), SurfaceFogPrefixTransmission(uv, high, fog, atlas), boundary - float(low))).rgb : medium.aaa;
    return radiance * mediumTransmission + medium.rgb;
}
float4 EvaluateMaterialSurface(
    VertexOutput input,
    constant DrawConstants& constants,
    constant VSMConstants& vsm,
    constant CascadeConstants& cascades,
    constant SurfaceFogParameters& fog,
    texture2d<float> fogAtlas,
    constant LumenVolumeParameters& lumenVolume, texture2d<float> lumenAtlas,
    texture2d<float, access::read> cascade0,
    texture2d<float, access::read> cascade1,
    texture2d<float, access::read> cascade2,
    texture2d<float, access::read> cascade3,
    texture2d<float, access::read> shadowMap,
    texture2d<float> diffuseMap,
    texture2d<float> normalMap,
    texture2d<float> opacityMap,
    sampler surfaceSampler,
    bool frontFacing, float contactVisibility = 1.0)
{
    float2 roughnessMetallic = constants.ShadingParameters.xy;
    if (constants.TextureFlags.z > 1) roughnessMetallic *= opacityMap.sample(surfaceSampler, input.TexCoord).gb;
    float4 diffuseSample =
        SampleDiffuse(input, constants, diffuseMap, surfaceSampler);
    if (constants.TextureFlags.w == ToonShadingModel && constants.TextureFlags.x != 0u)
        diffuseSample.rgb = ToonTextureColor(diffuseMap, surfaceSampler, input.TexCoord, dfdx(input.TexCoord), dfdy(input.TexCoord));
    float alpha = SurfaceAlpha(
        input, constants, diffuseSample, opacityMap, surfaceSampler);
    ApplyAlphaClip(alpha, constants);
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (constants.RenderParameters.y > 1.0 && constants.RenderParameters.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = dfdx(wireBary), wireDy = dfdy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard_fragment();
    }
    if (constants.RenderParameters.y > 0.5)
    {
        return float4(
            ApplySurfaceFog(SRGBToLinear(saturate(constants.PipelineColor.rgb)), input.Position.xyz, fog, fogAtlas),
            alpha);
    }
    float3 surfaceBase = ToonAlbedo(input.Color.rgb *
        max(constants.PipelineColor.rgb, 0.0) *
        max(diffuseSample.rgb, 0.0), constants.TextureFlags.w);
    float3 normal = ResolveSurfaceNormal(
        input, constants, normalMap, surfaceSampler, frontFacing);
    float3 viewDirection = NormalizeSafe(
        constants.CameraLightCount.xyz - input.WorldPosition,
        normal);
#if EVANS_VSM_PROGRAM == GOURAUD_RECEIVER
    float3 diffuseLight = frontFacing ? input.DiffuseFront : input.DiffuseBack;
    float3 specularLight = frontFacing ? input.SpecularFront : input.SpecularBack;
    int shadowIndex = int(constants.RenderParameters.z);
    if (shadowIndex >= 0 && shadowIndex < int(MaximumLights) && constants.Lights[shadowIndex].ColorCastShadows.w > 0.5)
    {
        LightConstants light = constants.Lights[shadowIndex];
        float3 direction = uint(light.PositionType.w + 0.5) == 0 ? light.DirectionIntensity.xyz : light.PositionType.xyz - input.WorldPosition;
        float visibility = cascades.Parameters.z > 0.0 && constants.RenderParameters.x >= 0.5
            ? CsmVisibility(input.WorldPosition, normal, direction, cascades, cascade0, cascade1, cascade2, cascade3, vsm)
            : SampleShadow(input, constants, shadowMap, normal, direction, vsm);
        // Compose raw primary/contact visibility before applying shadow strength once.
        if (contactVisibility < 1.0) visibility = mix(1.0 - ShadowStrength, visibility, contactVisibility);
        diffuseLight -= (1.0 - visibility) * (frontFacing ? input.ShadowDiffuseFront : input.ShadowDiffuseBack);
        specularLight -= (1.0 - visibility) * (frontFacing ? input.ShadowSpecularFront : input.ShadowSpecularBack);
    }
#else
    float3 diffuseLight = 0.0;
    float3 specularLight = 0.0;
    uint lightCount = min(uint(constants.CameraLightCount.w + 0.5), MaximumLights);
    bool useCascades = cascades.Parameters.z > 0.0 && constants.RenderParameters.x >= 0.5;
    float cascadeVisibility = 1.0;
    int shadowIndex = int(constants.RenderParameters.z);
    if (useCascades && shadowIndex >= 0 && shadowIndex < int(lightCount))
        cascadeVisibility = CsmVisibility(input.WorldPosition, normal,
            constants.Lights[shadowIndex].DirectionIntensity.xyz, cascades, cascade0, cascade1, cascade2, cascade3, vsm);
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        AccumulateSceneLight(roughnessMetallic,
            lightIndex, input, normal, viewDirection,
            surfaceBase * max(constants.MaterialDiffuseColor.rgb, 0.0), constants, shadowMap,
            useCascades, cascadeVisibility, contactVisibility, vsm, diffuseLight, specularLight);
    }
#endif
#if EVANS_VSM_PROGRAM == PBR_RECEIVER
    float3 color =
        ToonAmbient(surfaceBase * max(constants.MaterialDiffuseColor.rgb, float3(0.0)), surfaceBase *
            max(constants.MaterialAmbientColor.rgb, 0.0) *
            max(constants.AmbientColorStrength.rgb, 0.0) *
            max(constants.AmbientColorStrength.w, 0.0), constants.TextureFlags.w) * (1.0 - saturate(roughnessMetallic.y)) +
        surfaceBase *
            max(constants.MaterialDiffuseColor.rgb, 0.0) *
            diffuseLight +
        specularLight +
        max(constants.MaterialEmissiveColor.rgb, 0.0);
#else
    float3 color =
        ToonAmbient(surfaceBase * max(constants.MaterialDiffuseColor.rgb, float3(0.0)), surfaceBase *
            max(constants.MaterialAmbientColor.rgb, 0.0) *
            max(constants.AmbientColorStrength.rgb, 0.0) *
            max(constants.AmbientColorStrength.w, 0.0), constants.TextureFlags.w) +
        surfaceBase *
            max(constants.MaterialDiffuseColor.rgb, 0.0) *
            diffuseLight +
        max(constants.MaterialSpecularColor.rgb, 0.0) * specularLight +
        max(constants.MaterialEmissiveColor.rgb, 0.0);
#endif
    color += surfaceBase * max(constants.MaterialDiffuseColor.rgb,0.0) * (EVANS_VSM_PROGRAM==3 ? (1-LumenDielectricReflectance)*(1.0-saturate(roughnessMetallic.y)) : 1.0) * LumenTranslucentIrradiance(input.WorldPosition,normal,lumenVolume,lumenAtlas);
    color = ApplySurfaceFog(color, input.Position.xyz, fog, fogAtlas);
    // Floating-point targets preserve radiance until display tone mapping.
    const float MaximumHalfFloatRadiance = 65504.0;
    return float4(constants.ShadingParameters.w > 0.5 ? clamp(color, 0.0, MaximumHalfFloatRadiance) : PrepareDisplayLinear(color), alpha);
}

fragment float4 EvansFragmentMain(
    VertexOutput input [[stage_in]],
#if EVANS_CONTACT_SHADOWS
    texture2d_array<float, access::read> contactMask [[texture(15)]],
#if EVANS_CONTACT_MULTISAMPLE
    uint contactSample [[sample_id]],
#endif
#endif
    constant DrawConstants& constants [[buffer(1)]],
    constant VSMConstants& vsm [[buffer(6)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    constant SurfaceFogParameters& fog [[buffer(4)]],
    texture2d<float> fogAtlas [[texture(14)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(9)]],
    texture2d<float> lumenAtlas [[texture(19)]],
    texture2d<float, access::read> cascade0 [[texture(10)]],
    texture2d<float, access::read> cascade1 [[texture(11)]],
    texture2d<float, access::read> cascade2 [[texture(12)]],
    texture2d<float, access::read> cascade3 [[texture(13)]],
    texture2d<float, access::read> shadowMap [[texture(0)]],
    texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]],
    bool frontFacing [[front_facing]])
{
    float contactVisibility = 1.0;
#if EVANS_CONTACT_SHADOWS
#if EVANS_CONTACT_MULTISAMPLE
    contactVisibility = contactMask.read(uint2(input.Position.xy), contactSample).x;
#else
    contactVisibility = contactMask.read(uint2(input.Position.xy), 0).x;
#endif
#endif
    return EvaluateMaterialSurface(input, constants, vsm, cascades, fog, fogAtlas, lumenVolume, lumenAtlas,
        cascade0, cascade1, cascade2, cascade3, shadowMap,
        diffuseMap, normalMap, opacityMap, surfaceSampler, frontFacing, contactVisibility);
}

struct WeightedCoverageParameters
{
    float4 Weights; // Opacity/depth exponents, depth scale, positive weight floor.
    uint4 Options;
};
struct WeightedSurfaceOutput
{
    float4 RadianceSum [[color(0)]];
    float Opacity [[color(1)]];
};
fragment WeightedSurfaceOutput EvansWeightedFragment(
    VertexOutput input [[stage_in]],
    constant DrawConstants& constants [[buffer(1)]],
    constant VSMConstants& vsm [[buffer(6)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    constant SurfaceFogParameters& fog [[buffer(4)]],
    texture2d<float> fogAtlas [[texture(14)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(9)]],
    texture2d<float> lumenAtlas [[texture(19)]],
    texture2d<float, access::read> cascade0 [[texture(10)]],
    texture2d<float, access::read> cascade1 [[texture(11)]],
    texture2d<float, access::read> cascade2 [[texture(12)]],
    texture2d<float, access::read> cascade3 [[texture(13)]],
    texture2d<float, access::read> shadowMap [[texture(0)]],
    texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]],
    constant WeightedCoverageParameters& transparency [[buffer(5)]],
    bool frontFacing [[front_facing]])
{
    // Preserve this model's lighting and fragment-depth fog before accumulation.
    float4 surface = EvaluateMaterialSurface(input, constants, vsm, cascades, fog, fogAtlas, lumenVolume, lumenAtlas,
        cascade0, cascade1, cascade2, cascade3, shadowMap,
        diffuseMap, normalMap, opacityMap, surfaceSampler, frontFacing);
    float coverage = clamp(surface.a, 0.0, 1.0);
    WeightedSurfaceOutput output {float4(0.0), coverage};
    if (coverage == 0.0) return output;
    float depthFactor = 1.0 - clamp(input.Position.z, 0.0, 1.0) * transparency.Weights.z;
    float contribution = coverage * max(pow(coverage, transparency.Weights.x) *
        pow(depthFactor, transparency.Weights.y), transparency.Weights.w);
    output.RadianceSum = float4(surface.rgb * contribution, contribution);
    return output;
}

#elif EVANS_VSM_PROGRAM < VOLUME_RECEIVER
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

vertex LightingVaryings DeferredLightingVertex(uint vertexId [[vertex_id]])
{
    const float FullscreenExtent = 3.0;
    float2 point = vertexId == 0 ? float2(-1.0) :
        vertexId == 1 ? float2(FullscreenExtent, -1.0) : float2(-1.0, FullscreenExtent);
    return {float4(point, 0.0, 1.0)};
}

constant uint CsmCascadeLimit = 4;
struct CascadeConstants
{
    float4 LightRows[CsmCascadeLimit][4];
    float4 FarDepths;
    float4 BlendDepths;
    float4 EyeDepthPlane;
    float4 Parameters;
    float4 Filtering; // Preserves the shared cascade ABI; VSM owns its sampling controls.
};
float SurfaceComparisonBias(float3 normal, float3 towardLight)
{
    float squaredLength = dot(towardLight, towardLight);
    float3 direction = squaredLength > NormalizeEpsilon ? towardLight * rsqrt(squaredLength) : normal;
    float cosine = saturate(dot(normal, direction));
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumShadowCosine);
    return min(ReceiverBias + ReceiverSlopeBias * slope, MaximumReceiverBias);
}
float CsmSample(uint index, float3 position, float bias, constant CascadeConstants& cascades,
    texture2d<float, access::read> first, texture2d<float, access::read> second,
    texture2d<float, access::read> third, texture2d<float, access::read> fourth,
    constant VSMConstants& vsm)
{
    float4 clip = TransformRows(cascades.LightRows[index], float4(position, 1.0));
    if (index == 0) return VarianceMap(first, 1, clip, bias, vsm);
    if (index == 1) return VarianceMap(second, 2, clip, bias, vsm);
    if (index == 2) return VarianceMap(third, 3, clip, bias, vsm);
    return VarianceMap(fourth, 4, clip, bias, vsm);
}
float CsmVisibility(float3 position, float3 normal, float3 directionToLight, constant CascadeConstants& cascades,
    texture2d<float, access::read> first, texture2d<float, access::read> second,
    texture2d<float, access::read> third, texture2d<float, access::read> fourth,
    constant VSMConstants& vsm)
{
    float eyeDepth = dot(cascades.EyeDepthPlane, float4(position, 1.0));
    uint count = uint(cascades.Parameters.z);
    if (count == 0 || eyeDepth < cascades.Parameters.x || eyeDepth > cascades.Parameters.y) return 1.0;
    uint interval = 0;
    while (interval + 1 < count && eyeDepth > cascades.FarDepths[interval]) ++interval;
    float bias = SurfaceComparisonBias(normal, directionToLight);
    float visibility = CsmSample(interval, position, bias, cascades, first, second, third, fourth, vsm);
    float overlap = cascades.FarDepths[interval] - cascades.BlendDepths[interval];
    if (overlap > DistanceEpsilon && eyeDepth > cascades.BlendDepths[interval])
    {
        float neighbor = interval + 1 < count ? CsmSample(interval + 1, position, bias, cascades, first, second, third, fourth, vsm) : 1.0;
        visibility = mix(visibility, neighbor, saturate((eyeDepth - cascades.BlendDepths[interval]) / overlap));
    }
    return 1.0 - ShadowStrength * (1.0 - visibility);
}
float ShadowVisibility(float3 position, float3 normal, float3 towardLight,
    constant LightingConstants& lighting, texture2d<float, access::read> shadow,
    constant VSMConstants& vsm)
{
    if (lighting.Options.y == 0) return 1.0;
    float visible = VarianceMap(shadow, 0, TransformRows(lighting.LightViewProjection, float4(position, 1.0)),
        SurfaceComparisonBias(normal, towardLight), vsm);
    return 1.0 - ShadowStrength * (1.0 - visible);
}
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
        float3 f0 = mix(float3(DielectricReflectance), saturate(albedo), metallic);
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
float4 ShadeDeferredSurface(float2 rasterPosition, float2 extent, float4 diffuse,
    float4 materialNormal, float depth, float4 specular, float4 ambient, float4 emission,
    constant VSMConstants& vsm,
    constant LightingConstants& lighting, constant CascadeConstants& cascades,
    texture2d<float, access::read> shadowDepth, texture2d<float, access::read> cascade0,
    texture2d<float, access::read> cascade1, texture2d<float, access::read> cascade2,
    texture2d<float, access::read> cascade3, float contactVisibility = 1.0)
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
                ? CsmVisibility(position, normal, towardLight, cascades, cascade0, cascade1, cascade2, cascade3, vsm)
                : ShadowVisibility(position, normal, towardLight, lighting, shadowDepth, vsm);
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

#if EVANS_VSM_PROGRAM == DEFERRED_RECEIVER
fragment float4 DeferredLightingFragment(LightingVaryings input [[stage_in]],
    constant VSMConstants& vsm [[buffer(6)]],
#if EVANS_CONTACT_SHADOWS
    texture2d_array<float, access::read> contactMask [[texture(15)]],
#endif
    constant LightingConstants& lighting [[buffer(0)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    texture2d<float, access::read> cascade0 [[texture(10)]],
    texture2d<float, access::read> cascade1 [[texture(11)]],
    texture2d<float, access::read> cascade2 [[texture(12)]],
    texture2d<float, access::read> cascade3 [[texture(13)]],
    texture2d<float, access::read> diffuseOpacity [[texture(0)]],
    texture2d<float, access::read> normalRoughness [[texture(1)]],
    texture2d<float, access::read> specularShininess [[texture(2)]],
    texture2d<float, access::read> ambientMetallic [[texture(3)]],
    texture2d<float, access::read> emission [[texture(4)]],
    depth2d<float, access::read> sceneDepth [[texture(5)]],
    texture2d<float, access::read> shadowDepth [[texture(6)]])
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
        vsm, lighting, cascades, shadowDepth, cascade0, cascade1, cascade2, cascade3, contactVisibility);
}

#else
fragment float4 DeferredLightingSampleFragment(LightingVaryings input [[stage_in]], uint sampleIndex [[sample_id]],
    constant VSMConstants& vsm [[buffer(6)]],
#if EVANS_CONTACT_SHADOWS
    texture2d_array<float, access::read> contactMask [[texture(15)]],
#endif
    constant LightingConstants& lighting [[buffer(0)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    texture2d<float, access::read> cascade0 [[texture(10)]],
    texture2d<float, access::read> cascade1 [[texture(11)]],
    texture2d<float, access::read> cascade2 [[texture(12)]],
    texture2d<float, access::read> cascade3 [[texture(13)]],
    texture2d_ms<float, access::read> diffuseOpacity [[texture(0)]],
    texture2d_ms<float, access::read> normalRoughness [[texture(1)]],
    texture2d_ms<float, access::read> specularShininess [[texture(2)]],
    texture2d_ms<float, access::read> ambientMetallic [[texture(3)]],
    texture2d_ms<float, access::read> emission [[texture(4)]],
    depth2d_ms<float, access::read> sceneDepth [[texture(5)]],
    texture2d<float, access::read> shadowDepth [[texture(6)]])
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
        vsm, lighting, cascades, shadowDepth, cascade0, cascade1, cascade2, cascade3, contactVisibility);
}
#endif
#else
constant uint VolumeMaximumLights = 4;
constant uint VolumeMaximumCascades = 4;
struct VolumeMatrix { float4 Rows[4]; };
struct VolumeMedium
{
    VolumeMatrix InverseCamera;
    float4 ExtinctionHeightFalloffRange;
    float4 AmbientSource;
    float4 ViewportGrid;
    uint4 StageSlicesSamplesEnabled;
};
struct VolumeLight
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorShadow;
    float4 AreaSize;
};
struct VolumeLighting
{
    float4 AlbedoAsymmetry;
    float4 IntensityShadowBias;
    uint4 CountShadowModeIndexStride;
    VolumeMatrix PrimaryCamera;
    VolumeLight Lights[VolumeMaximumLights];
    VolumeMatrix CascadeCameras[VolumeMaximumCascades];
    float4 CascadeEnds;
    float4 CascadeBlendStarts;
    float4 EyeDepthPlane;
    float4 CascadeRangeCount;
    float4 ShadowFilter;
};
static_assert(sizeof(VolumeMedium) == 128, "Medium upload layout mismatch");
static_assert(sizeof(VolumeLighting) == 704, "Lighting upload layout mismatch");
struct VolumeVertex { float4 Position [[position]]; };

constant float VolumePi = 3.14159265358979323846;
constant float VolumeOpticalDepthCeiling = 80.0;
constant float VolumeRayEpsilon = 1.0e-6;
constant float VolumeSquaredLightEpsilon = 1.0e-4;
constant float VolumeProjectionEpsilon = 1.0e-7;
constant float VolumeHalfMaximum = 65504.0;
constant float VolumeParallelThreshold = 0.99;
constant uint VolumeDirectionalType = 0, VolumePointType = 1;
constant uint VolumeSingleShadow = 1, VolumeCascadedShadow = 2;
constant uint VolumeAreaSteps = 2;
constant float VolumeAreaWeight = 1.0 / float(VolumeAreaSteps * VolumeAreaSteps);

vertex VolumeVertex VolumeVertexMain(uint vertexIndex [[vertex_id]])
{
    const float FullscreenEdge = 3.0;
    float2 corner = vertexIndex == 1 ? float2(FullscreenEdge, -1.0) :
        vertexIndex == 2 ? float2(-1.0, FullscreenEdge) : float2(-1.0);
    return {float4(corner, 0.0, 1.0)};
}
float4 VolumeTransform(constant VolumeMatrix& matrix, float4 value)
{
    return float4(dot(matrix.Rows[0], value), dot(matrix.Rows[1], value),
        dot(matrix.Rows[2], value), dot(matrix.Rows[3], value));
}
float3 VolumeDirection(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > VolumeSquaredLightEpsilon ? value * rsqrt(lengthSquared) : fallback;
}
float3 VolumeWorld(float2 uv, float depth, constant VolumeMedium& medium)
{
    // Fragment positions and texture rows have a top-left origin. Explicit row
    // vectors preserve the host matrix layout and canonical [0,1] clip depth.
    float4 clip = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), depth, 1.0);
    float4 world = VolumeTransform(medium.InverseCamera, clip);
    return world.xyz / world.w;
}
float VolumePhase(float3 towardLight, float3 awayFromEye, float asymmetry)
{
    // Physical incoming and outgoing travel are the negatives of both inputs.
    // Their cosine thus peaks at +1 for forward scattering with positive g.
    float cosine = clamp(dot(towardLight, awayFromEye), -1.0, 1.0);
    float denominator = 1.0 + asymmetry * asymmetry - 2.0 * asymmetry * cosine;
    return (1.0 - asymmetry * asymmetry) / (4.0 * VolumePi * denominator * sqrt(denominator));
}
float VolumeShadowMap(texture2d<float, access::read> image, uint projectionIndex, float4 clip,
    constant VolumeLighting& lighting, constant VSMConstants& vsm)
{
    return VarianceMap(image, projectionIndex, clip, lighting.IntensityShadowBias.y, vsm);
}
float VolumeCascade(uint index, float3 world, constant VolumeLighting& lighting, constant VSMConstants& vsm,
    texture2d<float, access::read> first, texture2d<float, access::read> second,
    texture2d<float, access::read> third, texture2d<float, access::read> fourth)
{
    float4 clip = VolumeTransform(lighting.CascadeCameras[index], float4(world, 1.0));
    if (index == 0) return VolumeShadowMap(first, 1, clip, lighting, vsm);
    if (index == 1) return VolumeShadowMap(second, 2, clip, lighting, vsm);
    if (index == 2) return VolumeShadowMap(third, 3, clip, lighting, vsm);
    return VolumeShadowMap(fourth, 4, clip, lighting, vsm);
}
float VolumeVisibility(float3 world, constant VolumeLighting& lighting, constant VSMConstants& vsm, texture2d<float, access::read> primary,
    texture2d<float, access::read> first, texture2d<float, access::read> second,
    texture2d<float, access::read> third, texture2d<float, access::read> fourth)
{
    uint mode = lighting.CountShadowModeIndexStride.y;
    if (mode == VolumeSingleShadow)
        return VolumeShadowMap(primary, 0, VolumeTransform(lighting.PrimaryCamera, float4(world, 1.0)), lighting, vsm);
    if (mode != VolumeCascadedShadow) return 1.0;
    float eyeDepth = dot(lighting.EyeDepthPlane, float4(world, 1.0));
    if (eyeDepth < lighting.CascadeRangeCount.x || eyeDepth > lighting.CascadeRangeCount.y) return 1.0;
    uint cascade = 0, count = uint(lighting.CascadeRangeCount.z);
    while (cascade + 1 < count && eyeDepth > lighting.CascadeEnds[cascade]) ++cascade;
    float visibility = VolumeCascade(cascade, world, lighting, vsm, first, second, third, fourth);
    float overlap = lighting.CascadeEnds[cascade] - lighting.CascadeBlendStarts[cascade];
    if (overlap > VolumeRayEpsilon && eyeDepth > lighting.CascadeBlendStarts[cascade])
    {
        float next = cascade + 1 < count ? VolumeCascade(cascade + 1, world, lighting, vsm, first, second, third, fourth) : 1.0;
        visibility = mix(visibility, next, saturate((eyeDepth - lighting.CascadeBlendStarts[cascade]) / overlap));
    }
    return visibility;
}
float VolumeEmitter(VolumeLight light, float3 world, float3 cameraRay, float asymmetry)
{
    uint type = uint(light.PositionType.w);
    float3 normal = VolumeDirection(light.DirectionIntensity.xyz, float3(0.0, 1.0, 0.0));
    if (type == VolumeDirectionalType) return VolumePhase(normal, cameraRay, asymmetry);
    if (type == VolumePointType)
    {
        float3 delta = light.PositionType.xyz - world;
        return VolumePhase(VolumeDirection(delta, normal), cameraRay, asymmetry) * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), VolumeSquaredLightEpsilon);
    }
    float3 helper = abs(normal.y) < VolumeParallelThreshold ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
    float integral = 0.0;
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSize.w > 0.0;
    uint emitterGrid = physicalEmitter ? uint(clamp(light.AreaSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : VolumeAreaSteps;
    float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
    for (uint row = 0; row < emitterGrid; ++row)
        for (uint column = 0; column < emitterGrid; ++column)
        {
            float2 offset = ((float2(column, row) + 0.5) / float(emitterGrid) - 0.5) * light.AreaSize.xy;
            float3 delta = light.PositionType.xyz + tangent * offset.x + bitangent * offset.y - world;
            float3 towardLight = VolumeDirection(delta, -normal);
            integral += emitterSampleWeight * saturate(dot(normal, -towardLight)) * VolumePhase(towardLight, cameraRay, asymmetry) * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), VolumeSquaredLightEpsilon);
        }
    return integral * light.AreaSize.x * light.AreaSize.y;
}
float3 VolumeSource(float3 world, float3 cameraRay, constant VolumeMedium& medium, constant VolumeLighting& lighting, device const VolumeLight* sceneLights, constant VSMConstants& vsm,
    texture2d<float, access::read> primary, texture2d<float, access::read> first,
    texture2d<float, access::read> second, texture2d<float, access::read> third, texture2d<float, access::read> fourth)
{
    float visibility = VolumeVisibility(world, lighting, vsm, primary, first, second, third, fourth);
    float3 direct = float3(0.0);
    for (uint index = 0; index < lighting.CountShadowModeIndexStride.x; ++index)
    {
        VolumeLight light;
        if (lighting.IntensityShadowBias.z != 0.0) light = sceneLights[index];
        else light = lighting.Lights[index];
        float3 radiance = min(light.ColorShadow.rgb, float3(VolumeHalfMaximum)) * min(light.DirectionIntensity.w, VolumeHalfMaximum);
        float shadow = lighting.CountShadowModeIndexStride.y != 0 && index == lighting.CountShadowModeIndexStride.z ? visibility : 1.0;
        direct += radiance * (VolumeEmitter(light, world, cameraRay, lighting.AlbedoAsymmetry.w) * shadow);
    }
    return medium.AmbientSource.rgb + lighting.AlbedoAsymmetry.rgb * lighting.IntensityShadowBias.x * direct;
}
float4 VolumeSegment(uint2 pixel, constant VolumeMedium& medium, constant VolumeLighting& lighting, device const VolumeLight* sceneLights, constant VSMConstants& vsm,
    texture2d<float, access::read> primary, texture2d<float, access::read> first,
    texture2d<float, access::read> second, texture2d<float, access::read> third, texture2d<float, access::read> fourth)
{
    uint columns = uint(medium.ViewportGrid.z), slice = pixel.x / columns;
    float2 uv = (float2(pixel.x % columns, pixel.y) + 0.5) / medium.ViewportGrid.zw;
    float3 origin = VolumeWorld(uv, 0.0, medium), ray = VolumeWorld(uv, 1.0, medium) - origin;
    float lengthToFar = max(length(ray), VolumeRayEpsilon);
    float3 direction = ray / lengthToFar;
    float stepLength = min(lengthToFar, medium.ExtinctionHeightFalloffRange.w) /
        float(medium.StageSlicesSamplesEnabled.y * medium.StageSlicesSamplesEnabled.z);
    float4 segment = float4(0.0, 0.0, 0.0, 1.0);
    for (uint stepIndex = 0; stepIndex < medium.StageSlicesSamplesEnabled.z; ++stepIndex)
    {
        float distance = (float(slice * medium.StageSlicesSamplesEnabled.z + stepIndex) + 0.5) * stepLength;
        float3 world = origin + direction * distance;
        float heightExponent = min(medium.ExtinctionHeightFalloffRange.z * max(world.y - medium.ExtinctionHeightFalloffRange.y, 0.0), VolumeOpticalDepthCeiling);
        float sigmaT = medium.ExtinctionHeightFalloffRange.x * exp(-heightExponent);
        float transmission = exp(-min(sigmaT * stepLength, VolumeOpticalDepthCeiling));
        segment.rgb += segment.a * (1.0 - transmission) * VolumeSource(world, direction, medium, lighting, sceneLights, vsm, primary, first, second, third, fourth);
        segment.a *= transmission;
    }
    return segment;
}
fragment float4 VolumeFragmentMain(VolumeVertex input [[stage_in]],
    constant VSMConstants& vsm [[buffer(6)]],
    constant VolumeMedium& medium [[buffer(0)]], constant VolumeLighting& lighting [[buffer(1)]],
    device const VolumeLight* sceneLights [[buffer(8)]], // Complete scene; no opaque-depth list.
    texture2d<float, access::read> primary [[texture(1)]],
    texture2d<float, access::read> first [[texture(2)]], texture2d<float, access::read> second [[texture(3)]],
    texture2d<float, access::read> third [[texture(4)]], texture2d<float, access::read> fourth [[texture(5)]])
{
    uint2 pixel = uint2(input.Position.xy);
    return VolumeSegment(pixel, medium, lighting, sceneLights, vsm, primary, first, second, third, fourth);
}

#endif

#endif
