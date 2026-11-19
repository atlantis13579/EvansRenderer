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

// Normal mapping: screen derivatives reconstruct a tangent frame from world
// position and UVs. Degenerate UVs retain the surface normal. Decode the data
// texture into tangent space, then normalize in world space before lighting.
// Standalone Metal source. G-buffer and forward entry points share this
// file's material contract; specialized lighting pipelines own their consumers.
#define MATERIAL_PHONG 0
#define MATERIAL_FLAT 1
#define MATERIAL_GOURAUD 2
#define MATERIAL_PBR 3
#define MATERIAL_TOON 4

#if defined(EVANS_MATERIAL_GBUFFER)
// Material capture writes attributes; the independent Deferred/Tiled stage lights them.
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
    output.SpecularShininess = float4(clamp(draw.MaterialSpecularColor.rgb, 0.0, MaximumHalfFloatRadiance),
        clamp(draw.MaterialParameters.x, 1.0, MaximumHalfFloatRadiance));
    output.AmbientMetallic = float4(clamp(ambient, 0.0, MaximumHalfFloatRadiance), metallic);
    output.Emission = float4(clamp(draw.MaterialEmissiveColor.rgb, 0.0, MaximumHalfFloatRadiance), saturate(draw.MaterialParameters.w));
    return output;
}

#else
// Percentage-closer filtering: average visibility comparisons, not depths.
// Reeves, Salesin and Cook, Rendering Antialiased Shadows with Depth Maps (1987).
// Radius, reciprocal tap count and activation arrive in immutable frame controls.
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

// Forward material stage: mapping, coverage, selected surface model and hard-map visibility.
// Integer depth reads form four comparisons for each bilinear kernel tap.
constant uint PCFProjectionCount = 5;
constant float PCFPi = 3.14159265358979323846;
constant float PCFProjectionTolerance = 1.0e-7;
constant float PCFMapFractionLimit = 0.25;
struct PCFConstants { float4 Kernel; };
static_assert(sizeof(PCFConstants) == 16, "PCF upload layout mismatch");

float ReadComparison(depth2d<float, access::read> image, int2 pixel, float receiver, bool clampEdges)
{
    if (clampEdges) pixel=clamp(pixel,int2(0),int2(image.get_width(),image.get_height())-1);
    if (any(pixel < int2(0)) || any(pixel >= int2(image.get_width(), image.get_height()))) return 1.0;
    return receiver <= image.read(uint2(pixel)) ? 1.0 : 0.0;
}
float FilterComparison(depth2d<float, access::read> image, float2 uv, float receiver, bool clampEdges)
{
    float2 center = uv * float2(image.get_width(), image.get_height()) - 0.5;
    int2 corner = int2(floor(center));
    float2 weight = fract(center);
    float firstRow = mix(ReadComparison(image, corner, receiver, clampEdges), ReadComparison(image, corner + int2(1, 0), receiver, clampEdges), weight.x);
    float secondRow = mix(ReadComparison(image, corner + int2(0, 1), receiver, clampEdges), ReadComparison(image, corner + int2(1, 1), receiver, clampEdges), weight.x);
    return mix(firstRow, secondRow, weight.y);
}
float PcfMap(depth2d<float, access::read> image, uint projectionIndex, float4 clip,
    float comparisonBias, constant PCFConstants& pcf)
{
    if (pcf.Kernel.z<0.5 || clip.w<=PCFProjectionTolerance) return 1.0;
    float3 projected=clip.xyz/clip.w;
    float2 uv=projected.xy*float2(0.5,-0.5)+0.5;
    if (projected.z<0.0 || projected.z>1.0 || any(uv<0.0) || any(uv>1.0)) return 1.0;
    float2 extent(image.get_width(),image.get_height());
    if (pcf.Kernel.x==0.0) return ReadComparison(image,min(int2(floor(uv*extent)),int2(extent)-1),projected.z-comparisonBias,projectionIndex==0u);
    int radius=int(pcf.Kernel.x);
    float visibility=0.0;
    for (int y=-radius;y<=radius;++y)
        for (int x=-radius;x<=radius;++x)
            visibility+=FilterComparison(image,uv+float2(x,y)/extent,projected.z-comparisonBias,projectionIndex==0u);
    return visibility*pcf.Kernel.y;
}

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

#if EVANS_MATERIAL_PROGRAM == MATERIAL_GOURAUD
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
    float4 Filtering; // Preserves the shared cascade ABI; PCF owns its sampling controls.
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
    depth2d<float, access::read> first, depth2d<float, access::read> second,
    depth2d<float, access::read> third, depth2d<float, access::read> fourth,
    constant PCFConstants& pcf)
{
    float4 clip = TransformRows(cascades.LightRows[index], float4(position, 1.0));
    if (index == 0) return PcfMap(first, 1, clip, bias, pcf);
    if (index == 1) return PcfMap(second, 2, clip, bias, pcf);
    if (index == 2) return PcfMap(third, 3, clip, bias, pcf);
    return PcfMap(fourth, 4, clip, bias, pcf);
}
float CsmVisibility(float3 position, float3 normal, float3 directionToLight, constant CascadeConstants& cascades,
    depth2d<float, access::read> first, depth2d<float, access::read> second,
    depth2d<float, access::read> third, depth2d<float, access::read> fourth,
    constant PCFConstants& pcf)
{
    float eyeDepth = dot(cascades.EyeDepthPlane, float4(position, 1.0));
    uint count = uint(cascades.Parameters.z);
    if (count == 0 || eyeDepth < cascades.Parameters.x || eyeDepth > cascades.Parameters.y) return 1.0;
    uint interval = 0;
    while (interval + 1 < count && eyeDepth > cascades.FarDepths[interval]) ++interval;
    float bias = SurfaceComparisonBias(normal, directionToLight);
    float visibility = CsmSample(interval, position, bias, cascades, first, second, third, fourth, pcf);
    float overlap = cascades.FarDepths[interval] - cascades.BlendDepths[interval];
    if (overlap > DistanceEpsilon && eyeDepth > cascades.BlendDepths[interval])
    {
        float neighbor = interval + 1 < count ? CsmSample(interval + 1, position, bias, cascades, first, second, third, fourth, pcf) : 1.0;
        visibility = mix(visibility, neighbor, saturate((eyeDepth - cascades.BlendDepths[interval]) / overlap));
    }
    return 1.0 - ShadowStrength * (1.0 - visibility);
}
float SampleShadow(VertexOutput input, constant DrawConstants& constants,
    depth2d<float, access::read> shadowMap, float3 normal,
    float3 directionToLight, constant PCFConstants& pcf)
{
    if (constants.RenderParameters.x < 0.5) return 1.0;
    float visible = PcfMap(shadowMap, 0, input.LightPosition, SurfaceComparisonBias(normal, directionToLight), pcf);
    return 1.0 - ShadowStrength * (1.0 - visible);
}
#if EVANS_MATERIAL_PROGRAM == MATERIAL_FLAT
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
#if EVANS_MATERIAL_PROGRAM == MATERIAL_PBR
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

#elif EVANS_MATERIAL_PROGRAM == MATERIAL_TOON
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
    depth2d<float, access::read> shadowMap,
    bool useCascades, float cascadeVisibility, float contactVisibility,
    constant PCFConstants& pcf,
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
            input, constants, shadowMap, normal, shadowDirection, pcf))
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
    constant PCFConstants& pcf,
    constant CascadeConstants& cascades,
    constant SurfaceFogParameters& fog,
    texture2d<float> fogAtlas,
    constant LumenVolumeParameters& lumenVolume, texture2d<float> lumenAtlas,
    depth2d<float, access::read> cascade0,
    depth2d<float, access::read> cascade1,
    depth2d<float, access::read> cascade2,
    depth2d<float, access::read> cascade3,
    depth2d<float, access::read> shadowMap,
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
#if EVANS_MATERIAL_PROGRAM == MATERIAL_GOURAUD
    float3 diffuseLight = frontFacing ? input.DiffuseFront : input.DiffuseBack;
    float3 specularLight = frontFacing ? input.SpecularFront : input.SpecularBack;
    int shadowIndex = int(constants.RenderParameters.z);
    if (shadowIndex >= 0 && shadowIndex < int(MaximumLights) && constants.Lights[shadowIndex].ColorCastShadows.w > 0.5)
    {
        LightConstants light = constants.Lights[shadowIndex];
        float3 direction = uint(light.PositionType.w + 0.5) == 0 ? light.DirectionIntensity.xyz : light.PositionType.xyz - input.WorldPosition;
        float visibility = cascades.Parameters.z > 0.0 && constants.RenderParameters.x >= 0.5
            ? CsmVisibility(input.WorldPosition, normal, direction, cascades, cascade0, cascade1, cascade2, cascade3, pcf)
            : SampleShadow(input, constants, shadowMap, normal, direction, pcf);
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
            constants.Lights[shadowIndex].DirectionIntensity.xyz, cascades, cascade0, cascade1, cascade2, cascade3, pcf);
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        AccumulateSceneLight(roughnessMetallic,
            lightIndex, input, normal, viewDirection,
            surfaceBase * max(constants.MaterialDiffuseColor.rgb, 0.0), constants, shadowMap,
            useCascades, cascadeVisibility, contactVisibility, pcf, diffuseLight, specularLight);
    }
#endif
#if EVANS_MATERIAL_PROGRAM == MATERIAL_PBR
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
    color += surfaceBase * max(constants.MaterialDiffuseColor.rgb,0.0) * (EVANS_MATERIAL_PROGRAM==MATERIAL_PBR ? (1-LumenDielectricReflectance)*(1.0-saturate(roughnessMetallic.y)) : 1.0) * LumenTranslucentIrradiance(input.WorldPosition,normal,lumenVolume,lumenAtlas);
    color = ApplySurfaceFog(color, input.Position.xyz, fog, fogAtlas);
    // Floating-point targets preserve radiance until display tone mapping.
    const float MaximumHalfFloatRadiance = 65504.0;
    return float4(constants.ShadingParameters.w > 0.5 ? clamp(color, 0.0, MaximumHalfFloatRadiance) : PrepareDisplayLinear(color), alpha);
}

fragment float4 EvansFragmentMain(
    VertexOutput input [[stage_in]],
    constant DrawConstants& constants [[buffer(1)]],
    constant PCFConstants& pcf [[buffer(6)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    constant SurfaceFogParameters& fog [[buffer(4)]],
    texture2d<float> fogAtlas [[texture(14)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(9)]],
    texture2d<float> lumenAtlas [[texture(19)]],
    depth2d<float, access::read> cascade0 [[texture(10)]],
    depth2d<float, access::read> cascade1 [[texture(11)]],
    depth2d<float, access::read> cascade2 [[texture(12)]],
    depth2d<float, access::read> cascade3 [[texture(13)]],
    depth2d<float, access::read> shadowMap [[texture(0)]],
    texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]],
    bool frontFacing [[front_facing]])
{
    float contactVisibility = 1.0;
    return EvaluateMaterialSurface(input, constants, pcf, cascades, fog, fogAtlas, lumenVolume, lumenAtlas,
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
    constant PCFConstants& pcf [[buffer(6)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    constant SurfaceFogParameters& fog [[buffer(4)]],
    texture2d<float> fogAtlas [[texture(14)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(9)]],
    texture2d<float> lumenAtlas [[texture(19)]],
    depth2d<float, access::read> cascade0 [[texture(10)]],
    depth2d<float, access::read> cascade1 [[texture(11)]],
    depth2d<float, access::read> cascade2 [[texture(12)]],
    depth2d<float, access::read> cascade3 [[texture(13)]],
    depth2d<float, access::read> shadowMap [[texture(0)]],
    texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]],
    constant WeightedCoverageParameters& transparency [[buffer(5)]],
    bool frontFacing [[front_facing]])
{
    // Preserve this model's lighting and fragment-depth fog before accumulation.
    float4 surface = EvaluateMaterialSurface(input, constants, pcf, cascades, fog, fogAtlas, lumenVolume, lumenAtlas,
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


#endif
