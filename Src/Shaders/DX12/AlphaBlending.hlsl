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

// Alpha blending: evaluate straight linear RGB and independent opacity before
// native source-alpha composition. Weighted passes export transport planes for
// the independent OIT resolve. Ordered transparency retains opaque depth.
// Standalone DX12 source. G-buffer and forward entry points share this
// file's material contract; specialized lighting pipelines own their consumers.
#if defined(EVANS_MATERIAL_GBUFFER)
// Material capture writes attributes; the independent Deferred/Tiled stage lights them.
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
Texture2D<float4> AlbedoTexture : register(t1);
Texture2D<float4> TangentNormalTexture : register(t2);
Texture2D<float4> OpacityTexture : register(t3);
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
    result.SpecularShininess = clamp(float4(SpecularReflectance.rgb, max(MaterialControls.x, 1.0f)), 0.0f, HalfFloatLimit);
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

#else
// Percentage-closer filtering: average visibility comparisons, not depths.
// Reeves, Salesin and Cook, Rendering Antialiased Shadows with Depth Maps (1987).
// Radius, reciprocal tap count and activation arrive in immutable frame controls.
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
float ReadContactVisibility(float4 pixel, uint sampleIndex)
{
    return 1.0f;
}
float CombineContactVisibility(float primary, float contact, float strength)
{
    return lerp(1.0f - strength, primary, contact);
}

// Forward material stage with hard-map visibility and independent native bindings.
// b6 is an immutable root CBV, separate from surface/fog/transparent constants.
#define MATERIAL_FLAT 1
#define MATERIAL_GOURAUD 2
#define MATERIAL_PBR 3
#define MATERIAL_TOON 4

static const uint LightCountLimit = 4, CascadeCountLimit = 4, ProjectionCount = 5;
static const uint DirectionalKind = 0, PointKind = 1;
static const float Pi = 3.14159265358979323846;
static const float PixelCenterOffset = 0.5;
static const float ClipWMinimum = 1.0e-7, VectorLengthSquaredMinimum = 1.0e-12;
static const float HalfFloatMaximum = 65504;
static const float MapRadiusFraction = 0.25;
static const float ShadowStrength = 0.55;
static const float BiasConstant = 0.00035, BiasSlope = 0.00060, BiasMaximum = 0.0045, BiasCosineFloor = 0.15;
struct Emitter
{
    float4 PositionKind;
    float4 DirectionPower;
    float4 ColorShadow;
    float4 Rectangle;
};
cbuffer PCFReceiverConstants : register(b6)
{ float4 PcfKernel; };

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
cbuffer CascadeUpload : register(b2)
{
    row_major float4x4 CascadeProjection[CascadeCountLimit];
    float4 CascadeEnds, CascadeOverlaps, EyeDepthPlane, CascadeSettings, UnusedFixedFilter;
};
Texture2D<float> SingleDepth : register(t0);
Texture2D<float> CascadeDepth0 : register(t10);
Texture2D<float> CascadeDepth1 : register(t11);
Texture2D<float> CascadeDepth2 : register(t12);
Texture2D<float> CascadeDepth3 : register(t13);

float3 Unit(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > VectorLengthSquaredMinimum ? value * rsqrt(squared) : fallback;
}
float ReadShadowDepth(Texture2D<float> depth, int2 location, int2 extent, bool clampEdges)
{
    float result = 1;
    if (clampEdges) location=clamp(location,0,extent-1);
    if (all(location >= 0) && all(location < extent)) result = depth.Load(int3(location, 0));
    return result;
}
float CompareBilinear(Texture2D<float> depth, float2 uv, int2 extent, float receiver, bool clampEdges)
{
    float2 location = uv * extent - PixelCenterOffset;
    int2 cell = int2(floor(location));
    float2 fraction = frac(location);
    float upperLeft = receiver <= ReadShadowDepth(depth, cell, extent, clampEdges);
    float upperRight = receiver <= ReadShadowDepth(depth, cell + int2(1, 0), extent, clampEdges);
    float lowerLeft = receiver <= ReadShadowDepth(depth, cell + int2(0, 1), extent, clampEdges);
    float lowerRight = receiver <= ReadShadowDepth(depth, cell + 1, extent, clampEdges);
    return lerp(lerp(upperLeft, upperRight, fraction.x), lerp(lowerLeft, lowerRight, fraction.x), fraction.y);
}

float SoftVisibility(Texture2D<float> depth, uint mapIndex, float4 clip, float bias)
{
    float visibility=1;
    if (PcfKernel.z>=0.5 && clip.w>ClipWMinimum)
    {
        float3 projected=clip.xyz/clip.w;
        float2 uv=float2(projected.x+1,1-projected.y)*0.5;
        if (all(uv>=0) && all(uv<=1) && projected.z>=0 && projected.z<=1)
        {
            uint width,height; depth.GetDimensions(width,height);
            int2 extent=int2(width,height);
            // Keep one initialized return value: FXC's strict resource inliner
            // otherwise diagnoses an early comparison return as undefined.
            if (PcfKernel.x==0)
                visibility=projected.z-bias<=ReadShadowDepth(depth,min(int2(floor(uv*extent)),extent-1),extent,mapIndex==0)?1.0:0.0;
            else
            {
                int radius=int(PcfKernel.x);
                float total=0;
                [loop] for (int y=-radius;y<=radius;++y)
                    [loop] for (int x=-radius;x<=radius;++x)
                        total+=CompareBilinear(depth,uv+float2(x,y)/float2(extent),extent,projected.z-bias,mapIndex==0);
                visibility=total*PcfKernel.y;
            }
        }
    }
    return visibility;
}
float CascadeVisibility(uint cascade, float3 world, float bias)
{
    float result = 1;
    if (cascade < uint(CascadeSettings.z))
    {
        float4 clip = mul(CascadeProjection[cascade], float4(world, 1));
        if (cascade == 0) result = SoftVisibility(CascadeDepth0, 1, clip, bias);
        else if (cascade == 1) result = SoftVisibility(CascadeDepth1, 2, clip, bias);
        else if (cascade == 2) result = SoftVisibility(CascadeDepth2, 3, clip, bias);
        else result = SoftVisibility(CascadeDepth3, 4, clip, bias);
    }
    return result;
}
float PrimaryVisibility(float3 world, float bias, bool cascaded)
{
    float visibility = 1;
    if (!cascaded) visibility = SoftVisibility(SingleDepth, 0, mul(ShadowProjection, float4(world, 1)), bias);
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
    return false;
}
float3 SampleReflectance(Surface material, float3 toLight, float3 radiance)
{
    const float RoughnessFloor = 0.045, DielectricF0 = 0.04;
    const float DenominatorFloor = 1.0e-6, DistributionFloor = 1.0e-12, FresnelPower = 5;
    const float SpecularThreshold = 0.5, MinimumBands = 2;
    const float MaximumBands = 16;
    float3 direction = Unit(toLight, material.Normal), halfway = Unit(direction + material.View, material.Normal);
    float nl = saturate(dot(material.Normal, direction)), nh = saturate(dot(material.Normal, halfway));
    float3 reflected = 0;
    if (material.Model == MATERIAL_PBR)
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
        if (material.Model == MATERIAL_TOON)
        {
            float bands = clamp(round(material.Bands), MinimumBands, MaximumBands);
            reflected = radiance * (material.Diffuse * ToonResponse(nl, bands) + (DirectDiffuseOnly() ? 0.0 : material.Specular * ToonHighlight(specular)));
        }
        else reflected = radiance * (material.Diffuse * nl + (DirectDiffuseOnly() ? 0.0 : material.Specular * specular));
        if (material.Model <= MATERIAL_FLAT && dot(toLight, toLight) <= VectorLengthSquaredMinimum) reflected = 0;
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

struct SurfaceFragment
{
    float4 Pixel : SV_POSITION;
    float3 World : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 UnusedShadowPosition : TEXCOORD2;
    float2 UV : TEXCOORD3;
#if EVANS_MATERIAL_PROGRAM == MATERIAL_GOURAUD
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
#if EVANS_MATERIAL_PROGRAM == MATERIAL_FLAT
    normal = Unit(cross(ddx(fragment.World), ddy(fragment.World)), normal);
    if (dot(normal, fragment.Normal) < 0) normal = -normal;
#endif
    if (!front) normal = -normal;
#if EVANS_MATERIAL_PROGRAM != MATERIAL_FLAT && EVANS_MATERIAL_PROGRAM != MATERIAL_GOURAUD
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
float4 PSMain(SurfaceFragment fragment, bool front : SV_IsFrontFace) : SV_TARGET
{
    const uint contactSample = 0;
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
        material.Model = EVANS_MATERIAL_PROGRAM;
        color = ToonAmbient(base * max(DiffuseReflectance.rgb, 0.0), base * max(AmbientReflectance.rgb, 0) * max(AmbientLighting.rgb, 0) * max(AmbientLighting.w, 0), float(MaterialImages.w));
#if EVANS_MATERIAL_PROGRAM == MATERIAL_PBR
        color *= 1 - material.Metallic;
#endif
        color += max(EmissionRadiance.rgb, 0);
#if EVANS_MATERIAL_PROGRAM == MATERIAL_GOURAUD
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
        color += material.Diffuse * (material.Model==MATERIAL_PBR ? (1-LumenDielectricReflectance)*(1-material.Metallic) : 1) * LumenTranslucentIrradiance(material.World,material.Normal);
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

#endif
