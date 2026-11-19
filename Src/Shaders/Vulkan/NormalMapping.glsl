#version 450
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
vec3 ToonTextureColor(texture2D image, sampler imageSampler, vec2 uv, vec2 uvDx, vec2 uvDy)
{
    vec3 color = vec3(0.0);
    for (int row = -ToonTextureKernelRadius; row <= ToonTextureKernelRadius; ++row)
        for (int column = -ToonTextureKernelRadius; column <= ToonTextureKernelRadius; ++column)
            color += textureGrad(sampler2D(image, imageSampler), uv + ToonTextureRadius * (float(column) * uvDx + float(row) * uvDy), uvDx, uvDy).rgb;
    return color * ToonTextureSampleWeight;
}

// Normal mapping: screen derivatives reconstruct a tangent frame from world
// position and UVs. Degenerate UVs retain the surface normal. Decode the data
// texture into tangent space, then normalize in world space before lighting.
// Standalone Vulkan source. G-buffer and forward entry points share this
// file's material contract; specialized lighting pipelines own their consumers.
#define MATERIAL_PHONG 0
#define MATERIAL_FLAT 1
#define MATERIAL_GOURAUD 2
#define MATERIAL_PBR 3
#define MATERIAL_TOON 4

#if defined(EVANS_MATERIAL_GBUFFER)
// Material capture writes attributes; the independent Deferred/Tiled stage lights them.
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

// Native material capture preserves explicit descriptors and row-major uniforms.
// EVANS_VERTEX_STAGE selects the geometry vertex entry point.
// Shishkovtsov, "Deferred Shading in S.T.A.L.K.E.R.", GPU Gems 2 (2005).
// https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-9-deferred-shading-stalker
const int MaximumLights = 4;
const uint FlatModel = 1u;
const uint PhysicalModel = 3u;
const uint ToonModel = 4u;
const float NormalizeEpsilon = 1.0e-12;
const float HomogeneousEpsilon = 1.0e-7;
const float TangentEpsilon = 1.0e-10;
const float ParallelAxisLimit = 0.999;
const float MinimumRoughness = 0.045;
const float HalfFloatMaximum = 65504.0;
struct SceneLight
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
vec3 UnitDirection(vec3 value, vec3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalizeEpsilon ? value * inversesqrt(lengthSquared) : fallback;
}

layout(set = 0, binding = 0, std140, row_major) uniform GeometryConstants
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
vec3 DecodeSrgb(vec3 color)
{
    const float LinearThreshold = 0.04045;
    const float LinearScale = 12.92;
    const float PowerOffset = 0.055;
    const float PowerScale = 1.055;
    const float PowerExponent = 2.4;
    vec3 linear = color / LinearScale;
    vec3 power = pow((color + PowerOffset) / PowerScale, vec3(PowerExponent));
    return mix(power, linear, lessThanEqual(color, vec3(LinearThreshold)));
}
#ifdef EVANS_VERTEX_STAGE
// Match opaque forward depth exactly when material capture is enabled.
invariant gl_Position;
precise gl_Position;
layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
layout(location = 0) out vec3 WorldPosition;
layout(location = 1) out vec3 WorldNormal;
layout(location = 2) out vec4 VertexColor;
layout(location = 3) out vec2 TexCoord;
void main()
{
    vec4 world = Draw.World * vec4(InPosition, 1.0);
    gl_Position = Draw.ViewProjection * world;
    // Positive Vulkan viewport height maps NDC -Y to the top. Keep CPU cameras
    // unchanged; the lighting reconstruction applies the matching inverse flip.
    gl_Position.y = -gl_Position.y;
    WorldPosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    WorldNormal = UnitDirection(mat3(Draw.NormalWorld) * InNormal, vec3(0.0, 1.0, 0.0));
    VertexColor = vec4(DecodeSrgb(clamp(InColor.rgb, 0.0, 1.0)), InColor.a);
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
}
#else
layout(location = 0) in vec3 WorldPosition;
layout(location = 1) in vec3 WorldNormal;
layout(location = 2) in vec4 VertexColor;
layout(location = 3) in vec2 TexCoord;
layout(set = 0, binding = 2) uniform texture2D DiffuseImage;
layout(set = 0, binding = 3) uniform texture2D NormalImage;
layout(set = 0, binding = 4) uniform texture2D OpacityImage;
layout(set = 0, binding = 5) uniform sampler MaterialSampler;
layout(location = 0) out vec4 DiffuseOpacity;
layout(location = 1) out vec4 NormalRoughness;
layout(location = 2) out vec4 SpecularShininess;
layout(location = 3) out vec4 AmbientMetallic;
layout(location = 4) out vec4 Emission;
vec3 MaterialNormal()
{
    vec3 normal = UnitDirection(WorldNormal, vec3(0.0, 1.0, 0.0));
    vec3 dx = dFdx(WorldPosition);
    vec3 dy = dFdy(WorldPosition);
    if (Draw.TextureFlags.w == FlatModel)
    {
        vec3 geometric = UnitDirection(cross(dx, dy), normal);
        normal = dot(geometric, normal) < 0.0 ? -geometric : geometric;
    }
    if (!gl_FrontFacing) normal = -normal;
    if (Draw.TextureFlags.w == FlatModel || Draw.TextureFlags.y == 0u) return normal;
    vec2 uvDx = dFdx(TexCoord);
    vec2 uvDy = dFdy(TexCoord);
    float determinant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    if (abs(determinant) < TangentEpsilon) return normal;
    vec3 tangent = (dx * uvDy.y - dy * uvDx.y) / determinant;
    vec3 sourceBitangent = (dy * uvDx.x - dx * uvDy.x) / determinant;
    tangent -= normal * dot(normal, tangent);
    vec3 helper = abs(normal.y) < ParallelAxisLimit ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    tangent = UnitDirection(tangent, UnitDirection(cross(helper, normal), vec3(1.0, 0.0, 0.0)));
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    vec3 bitangent = UnitDirection(cross(normal, tangent) * handedness, vec3(0.0, 0.0, 1.0));
    vec3 mapped = texture(sampler2D(NormalImage, MaterialSampler), TexCoord).xyz * 2.0 - 1.0;
    if (float(Draw.TextureFlags.w) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
    return UnitDirection(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
}
void main()
{
    vec4 albedo = Draw.TextureFlags.x != 0u ? texture(sampler2D(DiffuseImage, MaterialSampler), TexCoord) : vec4(1.0);
    if (float(Draw.TextureFlags.w) == ToonShadingModel && float(Draw.TextureFlags.x) > 0.5)
        albedo.rgb = ToonTextureColor(DiffuseImage, MaterialSampler, TexCoord, dFdx(TexCoord), dFdy(TexCoord));
    vec3 surfaceMap = Draw.TextureFlags.z != 0 ? texture(sampler2D(OpacityImage, MaterialSampler), TexCoord).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = Draw.ShadingParameters.xy;
    if (Draw.TextureFlags.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacityMap = surfaceMap.r;
    float opacity = clamp(VertexColor.a * Draw.MaterialParameters.y * albedo.a * opacityMap, 0.0, 1.0);
    if (Draw.MaterialParameters.z > 0.0 && opacity < Draw.MaterialParameters.z) discard;
    vec3 surface = ToonAlbedo(VertexColor.rgb * max(Draw.PipelineColor.rgb, vec3(0.0)) * max(albedo.rgb, vec3(0.0)), float(Draw.TextureFlags.w));
    float metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    vec3 ambient = ToonAmbient(surface * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), surface * max(Draw.MaterialAmbientColor.rgb, vec3(0.0))
        * max(Draw.AmbientColorStrength.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.w, 0.0), float(Draw.TextureFlags.w));
    if (Draw.TextureFlags.w == PhysicalModel) ambient *= 1.0 - metallic;
    DiffuseOpacity = vec4(clamp(surface * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), 0.0, HalfFloatMaximum), opacity);
    SpecularShininess = clamp(vec4(Draw.MaterialSpecularColor.rgb, max(Draw.MaterialParameters.x, 1.0)), 0.0, HalfFloatMaximum);
    AmbientMetallic = vec4(clamp(ambient, 0.0, HalfFloatMaximum), metallic);
    Emission = vec4(clamp(Draw.MaterialEmissiveColor.rgb, 0.0, HalfFloatMaximum), clamp(Draw.MaterialParameters.w, 0.0, 1.0));
    const float OverlayMode = 0.5;
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
        const float UnlitCoverage = -1.0;
        NormalRoughness = vec4(0.0, 0.0, 0.0, UnlitCoverage);
        Emission.rgb = DecodeSrgb(clamp(Draw.PipelineColor.rgb, 0.0, 1.0));
    }
    else NormalRoughness = vec4(MaterialNormal(), clamp(roughnessMetallic.x, MinimumRoughness, 1.0));
}
#endif


#else
// Percentage-closer filtering: average visibility comparisons, not depths.
// Reeves, Salesin and Cook, Rendering Antialiased Shadows with Depth Maps (1987).
// Radius, reciprocal tap count and activation arrive in immutable frame controls.
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

// Forward material stage: mapping, coverage, selected surface model and hard-map visibility.
const int PCFMapCount = 5; // One single map and four cascade intervals.
// The UBO suffix retains its established 112-byte upload stride.
struct PCFData { vec4 Kernel; vec4 Reserved[6]; };
const int MaximumLights = 4;
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
    PCFData PCF;
} Draw;

#define Pcf Draw.PCF
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
layout(location = 0) out vec4 OutColor;
#ifdef EVANS_WEIGHTED_TRANSPARENCY
layout(location = 1) out float RevealageOpacity;
layout(push_constant) uniform WeightedCoverageParameters
{
    vec4 Weights; // opacity exponent, depth exponent, depth scale, minimum weight
} Transparency;
#endif

#if EVANS_MATERIAL_PROGRAM == MATERIAL_GOURAUD
layout(location = 5) in vec3 DiffuseFront;
layout(location = 6) in vec3 SpecularFront;
layout(location = 7) in vec3 DiffuseBack;
layout(location = 8) in vec3 SpecularBack;
layout(location = 9) in vec3 ShadowDiffuseFront;
layout(location = 10) in vec3 ShadowSpecularFront;
layout(location = 11) in vec3 ShadowDiffuseBack;
layout(location = 12) in vec3 ShadowSpecularBack;
#endif
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
#define PrimaryImage sampler2D(ShadowDepth, SurfaceSampler)
#define CascadeImage0 sampler2D(CsmDepth0, CsmNearest)
#define CascadeImage1 sampler2D(CsmDepth1, CsmNearest)
#define CascadeImage2 sampler2D(CsmDepth2, CsmNearest)
#define CascadeImage3 sampler2D(CsmDepth3, CsmNearest)

// Sample-frequency contact lookup must not move the existing pixel-centred
// material/alpha evaluation. In particular, cutouts must match the depth pass.
#if defined(EVANS_CONTACT_RECEIVER) && defined(EVANS_MULTISAMPLE) && EVANS_MATERIAL_PROGRAM < 5
#define WorldPosition interpolateAtOffset(WorldPosition, vec2(0.0))
#define WorldNormal interpolateAtOffset(WorldNormal, vec2(0.0))
#define VertexColor interpolateAtOffset(VertexColor, vec2(0.0))
#define LightPosition interpolateAtOffset(LightPosition, vec2(0.0))
#define TexCoord interpolateAtOffset(TexCoord, vec2(0.0))
#define DiffuseFront interpolateAtOffset(DiffuseFront, vec2(0.0))
#define SpecularFront interpolateAtOffset(SpecularFront, vec2(0.0))
#define DiffuseBack interpolateAtOffset(DiffuseBack, vec2(0.0))
#define SpecularBack interpolateAtOffset(SpecularBack, vec2(0.0))
#define ShadowDiffuseFront interpolateAtOffset(ShadowDiffuseFront, vec2(0.0))
#define ShadowSpecularFront interpolateAtOffset(ShadowSpecularFront, vec2(0.0))
#define ShadowDiffuseBack interpolateAtOffset(ShadowDiffuseBack, vec2(0.0))
#define ShadowSpecularBack interpolateAtOffset(ShadowSpecularBack, vec2(0.0))
#endif

// Contact is a separate camera-depth producer. Apply artistic strength once:
// (1 - strength) + strength * rawMapVisibility * rawContactVisibility.
float ComposeContact(float primaryVisibility)
{
    return primaryVisibility;
}

// Constant descriptor choices avoid descriptor-indexing extensions. Vulkan GLSL
// combines separate images and samplers only at the texture instruction itself.
ivec2 ShadowExtent(uint map)
{
    if (map == 0u) return textureSize(PrimaryImage, 0);
    if (map == 1u) return textureSize(CascadeImage0, 0);
    if (map == 2u) return textureSize(CascadeImage1, 0);
    if (map == 3u) return textureSize(CascadeImage2, 0);
    return textureSize(CascadeImage3, 0);
}
float ShadowDepthAt(uint map, ivec2 pixel)
{
    if (map == 0u) return texelFetch(PrimaryImage, pixel, 0).r;
    if (map == 1u) return texelFetch(CascadeImage0, pixel, 0).r;
    if (map == 2u) return texelFetch(CascadeImage1, pixel, 0).r;
    if (map == 3u) return texelFetch(CascadeImage2, pixel, 0).r;
    return texelFetch(CascadeImage3, pixel, 0).r;
}
// Blockers are averaged in linear light distance, not projected depth. Bias
// classifies blockers/comparisons but does not change the penumbra geometry.
// The disk quadrature is deterministic; both loops have independent GUI budgets.
const float PcfPi = 3.14159265358979323846;
const float GoldenAngle = PcfPi * (3.0 - sqrt(5.0));
const float PcfProjectionEpsilon = 1.0e-7;
const float MaximumMapFraction = 0.25;

float VisibilityTexel(uint map, ivec2 pixel, float receiver)
{
    if (map==0u) pixel=clamp(pixel,ivec2(0),ShadowExtent(map)-1);
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ShadowExtent(map)))) return 1.0;
    return receiver <= ShadowDepthAt(map, pixel) ? 1.0 : 0.0;
}
float ComparisonBilinear(uint map, vec2 uv, float receiver)
{
    vec2 texel = uv * vec2(ShadowExtent(map)) - 0.5;
    ivec2 origin = ivec2(floor(texel));
    vec2 fraction = fract(texel);
    float upper = mix(VisibilityTexel(map, origin, receiver),
        VisibilityTexel(map, origin + ivec2(1, 0), receiver), fraction.x);
    float lower = mix(VisibilityTexel(map, origin + ivec2(0, 1), receiver),
        VisibilityTexel(map, origin + ivec2(1, 1), receiver), fraction.x);
    return mix(upper, lower, fraction.y);
}
float PcfVisibility(uint map, vec4 clip, float bias)
{
    if (Pcf.Kernel.z<0.5 || clip.w<=PcfProjectionEpsilon) return 1.0;
    vec3 ndc=clip.xyz/clip.w;
    vec2 uv=ndc.xy*vec2(0.5,-0.5)+0.5;
    if (ndc.z<0.0 || ndc.z>1.0 || any(lessThan(uv,vec2(0.0))) || any(greaterThan(uv,vec2(1.0)))) return 1.0;
    ivec2 extent=ShadowExtent(map);
    if (Pcf.Kernel.x==0.0) return VisibilityTexel(map,min(ivec2(floor(uv*vec2(extent))),extent-1),ndc.z-bias);
    int radius=int(Pcf.Kernel.x);
    float visibility=0.0;
    for (int row=-radius;row<=radius;++row)
        for (int column=-radius;column<=radius;++column)
            visibility+=ComparisonBilinear(map,uv+vec2(column,row)/vec2(extent),ndc.z-bias);
    return visibility*Pcf.Kernel.y;
}

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

float CascadePCF(int split, vec3 world, float bias)
{
    vec4 clip = Csm.LightMatrices[split] * vec4(world, 1.0);
    uint map = uint(split + 1);
    return PcfVisibility(map, clip, bias);
}
float SurfaceVisibility(vec3 world, vec3 normal, vec3 towardLight, vec4 singleClip, bool enabled)
{
    if (!enabled) return 1.0;
    float cosine = clamp(dot(normal, NormalizeOr(towardLight, normal)), 0.0, 1.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumShadowCosine);
    float bias = min(ReceiverBias + ReceiverSlopeBias * slope, MaximumReceiverBias);
    float visibility;
    int count = int(Csm.Parameters.z);
    if (count > 0)
    {
        float eyeDepth = dot(Csm.EyeDepthPlane, vec4(world, 1.0));
        if (eyeDepth < Csm.Parameters.x || eyeDepth > Csm.Parameters.y) return 1.0;
        int split = 0;
        while (split + 1 < count && eyeDepth > Csm.FarDepths[split]) ++split;
        visibility = CascadePCF(split, world, bias);
        float overlap = Csm.FarDepths[split] - Csm.BlendDepths[split];
        if (overlap > DistanceEpsilon && eyeDepth > Csm.BlendDepths[split])
        {
            float neighbor = split + 1 < count ? CascadePCF(split + 1, world, bias) : 1.0;
            visibility = mix(visibility, neighbor, clamp((eyeDepth - Csm.BlendDepths[split]) / overlap, 0.0, 1.0));
        }
    }
    else
    {
        visibility = PcfVisibility(0u, singleClip, bias);
    }
    return 1.0 - ShadowStrength * (1.0 - visibility);
}
float ShadowVisibility(vec3 normal, vec3 lightVector)
{
    return SurfaceVisibility(WorldPosition, normal, lightVector, LightPosition, Draw.RenderParameters.x >= OverlayMode);
}
#if EVANS_MATERIAL_PROGRAM == MATERIAL_PBR
const float Pi = 3.141592653589793;
const float MinimumRoughness = 0.045;
const float DistributionEpsilon = 1.0e-12;
const float BrdfEpsilon = 1.0e-6;
const float DielectricFresnel = 0.04;
const float SchlickExponent = 5.0;
const float MicrofacetNormalization = 4.0;
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
    vec3 f0 = mix(vec3(DielectricFresnel), clamp(albedo, 0.0, 1.0), metallic);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vDotH, SchlickExponent);
    vec3 microfacet = fresnel * distribution * masking * shadowing / max(MicrofacetNormalization * nDotV * cosine, BrdfEpsilon);
    diffuse += visibleRadiance * (1.0 - fresnel) * (1.0 - metallic) * cosine / Pi;
    specular += visibleRadiance * microfacet * cosine;
}

#elif EVANS_MATERIAL_PROGRAM == MATERIAL_TOON
void EvaluateSample(vec2 roughnessMetallic, vec3 lightVector, vec3 radiance, float visibility,
    vec3 normal, vec3 viewDirection, vec3 albedo, inout vec3 diffuse, inout vec3 specular)
{
    float distanceSquared = dot(lightVector, lightVector);
    if (distanceSquared <= NormalizeEpsilon) return;
    vec3 direction = lightVector * inversesqrt(distanceSquared);
    float cosine = max(dot(normal, direction), 0.0);
    vec3 visibleRadiance = radiance * visibility;
    float bands = max(Draw.ShadingParameters.z, 2.0);
    vec3 toonResponse = ToonResponse(cosine, bands);
    diffuse += visibleRadiance * toonResponse;
    vec3 halfway = NormalizeOr(direction + viewDirection, normal);
    float highlight = pow(max(dot(normal, halfway), 0.0), max(Draw.MaterialParameters.x, 1.0));

    specular += visibleRadiance * ToonHighlight(highlight);
}

#else
void EvaluateSample(vec2 roughnessMetallic, vec3 lightVector, vec3 radiance, float visibility,
    vec3 normal, vec3 viewDirection, vec3 albedo, inout vec3 diffuse, inout vec3 specular)
{
    float distanceSquared = dot(lightVector, lightVector);
    if (distanceSquared <= NormalizeEpsilon) return;
    vec3 direction = lightVector * inversesqrt(distanceSquared);
    float cosine = max(dot(normal, direction), 0.0);
    if (cosine <= 0.0) return;
    vec3 visibleRadiance = radiance * visibility;
    diffuse += visibleRadiance * cosine;
    vec3 halfway = NormalizeOr(direction + viewDirection, normal);
    float highlight = pow(max(dot(normal, halfway), 0.0), max(Draw.MaterialParameters.x, 1.0));
    specular += visibleRadiance * highlight;
}

#endif
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
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSizePadding.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaSizePadding.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AreaGridSize;
    float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSizePadding.x * light.AreaSizePadding.y) : 1.0) / float(emitterGrid * emitterGrid);
    for (int y = 0; y < emitterGrid; ++y)
    {
        for (int x = 0; x < emitterGrid; ++x)
        {
            vec2 offset = vec2(((float(x) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter), ((float(y) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter));
            vec3 samplePosition = light.PositionType.xyz + tangent * offset.x * light.AreaSizePadding.x + bitangent * offset.y * light.AreaSizePadding.y;
            vec3 displacement = samplePosition - position;
            float distanceSquared = max(dot(displacement, displacement), DistanceEpsilon);
            vec3 direction = displacement * inversesqrt(distanceSquared);
            float emitterCosine = max(dot(emission, -direction), 0.0);
            EvaluateSample(roughnessMetallic, direction, radiance * (emitterSampleWeight * emitterCosine * LightRangeWindow(displacement, light.AreaSizePadding.z) / distanceSquared), visibility, normal, viewDirection, albedo, diffuse, specular);
        }
    }
}

#if EVANS_MATERIAL_PROGRAM == MATERIAL_FLAT
vec3 SurfaceNormal()
{
    vec3 reference = NormalizeOr(WorldNormal, vec3(0.0, 1.0, 0.0));
    vec3 plane = NormalizeOr(cross(dFdx(WorldPosition), dFdy(WorldPosition)), reference);
    if (dot(plane, reference) < 0.0) plane = -plane;
    return gl_FrontFacing ? plane : -plane;
}

#else
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
    if (float(Draw.TextureFlags.w) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
    return NormalizeOr(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
}

#endif
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
    if (float(Draw.TextureFlags.w) == ToonShadingModel && float(Draw.TextureFlags.x) > 0.5)
        texel.rgb = ToonTextureColor(DiffuseImage, SurfaceSampler, TexCoord, dFdx(TexCoord), dFdy(TexCoord));
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
    vec3 base = ToonAlbedo(VertexColor.rgb * max(Draw.PipelineColor.rgb, vec3(0.0)) * max(texel.rgb, vec3(0.0)), float(Draw.TextureFlags.w));
    vec3 albedo = base * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0));
    vec3 normal = SurfaceNormal();
    vec3 viewDirection = NormalizeOr(Draw.CameraLightCount.xyz - WorldPosition, normal);
#if EVANS_MATERIAL_PROGRAM == MATERIAL_GOURAUD
    vec3 diffuse = gl_FrontFacing ? DiffuseFront : DiffuseBack;
    vec3 specular = gl_FrontFacing ? SpecularFront : SpecularBack;
    int shadowIndex = int(Draw.RenderParameters.z);
    if (shadowIndex >= 0 && shadowIndex < MaximumLights && Draw.Lights[shadowIndex].ColorCastShadows.w > OverlayMode)
    {
        SceneLight light = Draw.Lights[shadowIndex];
        vec3 direction = int(light.PositionType.w + 0.5) == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - WorldPosition;
        float occlusion = 1.0 - ComposeContact(ShadowVisibility(normal, direction));
        diffuse -= occlusion * (gl_FrontFacing ? ShadowDiffuseFront : ShadowDiffuseBack);
        specular -= occlusion * (gl_FrontFacing ? ShadowSpecularFront : ShadowSpecularBack);
    }
#else
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    for (int index = 0; index < min(int(Draw.CameraLightCount.w + 0.5), MaximumLights); ++index)
    {
        SceneLight light = Draw.Lights[index];
        vec3 shadowVector = int(light.PositionType.w + 0.5) == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - WorldPosition;
        float visibility = index == int(Draw.RenderParameters.z) && light.ColorCastShadows.w > OverlayMode ? ComposeContact(ShadowVisibility(normal, shadowVector)) : 1.0;
        EvaluateSceneLight(roughnessMetallic, index, WorldPosition, normal, viewDirection, albedo, visibility, diffuse, specular);
    }
#endif
#if EVANS_MATERIAL_PROGRAM == MATERIAL_PBR
    vec3 color = ToonAmbient(base * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), base * max(Draw.MaterialAmbientColor.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.w, 0.0), float(Draw.TextureFlags.w)) * (1.0 - clamp(roughnessMetallic.y, 0.0, 1.0))
        + albedo * diffuse + specular + max(Draw.MaterialEmissiveColor.rgb, vec3(0.0));
#else
    vec3 color = ToonAmbient(base * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), base * max(Draw.MaterialAmbientColor.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.w, 0.0), float(Draw.TextureFlags.w))
        + albedo * diffuse + max(Draw.MaterialSpecularColor.rgb, vec3(0.0)) * specular + max(Draw.MaterialEmissiveColor.rgb, vec3(0.0));
#endif
    color = max(color, vec3(0.0));
    color += albedo * (EVANS_MATERIAL_PROGRAM==MATERIAL_PBR ? (1-LumenDielectricReflectance)*(1.0-clamp(roughnessMetallic.y,0.0,1.0)) : 1.0) * LumenTranslucentIrradiance(WorldPosition,normal);
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
