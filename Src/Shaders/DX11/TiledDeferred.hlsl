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

#if defined(EVANS_TILED_DEFERRED_SURFACE) && !defined(EVANS_VERTEX_STAGE)
// DX11 world-space shadow queries. Woop, Benthin and Wald, JCGT 2013.
// https://jcgt.org/published/0002/01/05/ . Forward escape links bound traversal;
// independent alpha planes preserve bilinear cutout coverage. The RGBA32UI ABI
// is described in RayTracedShadowScene.h. No screen-space visibility is assumed.
Texture2D<uint4> ShadowRayScene : register(t20);
static const uint ShadowRayCasterFlag=1u,ShadowRayImageABI=1u;
static const uint ShadowRayCountsHeader=0u,ShadowRayOffsetsHeader=1u,ShadowRayTraceHeader=2u,ShadowRayLightHeader=3u;
static const uint ShadowRayNodeSize=3u,ShadowRayTriangleSize=6u,ShadowRayMaterialSize=3u,ShadowRayScalarLanes=4u;
static const uint ShadowRayNodeMaximum=1u,ShadowRayNodeLinks=2u;
static const uint ShadowRayTriangleUV01=3u,ShadowRayTriangleUV2=4u,ShadowRayTriangleMetadata=5u;
static const uint ShadowRayMaterialOpacity=1u,ShadowRayMaterialCoverageWord=2u;
uint4 ShadowRayLoad(uint address)
{ uint width,height; ShadowRayScene.GetDimensions(width,height); width=max(width,1u); return ShadowRayScene.Load(int3(address%width,address/width,0)); }
struct ShadowRayNode { float4 Minimum,Maximum; uint4 Links; };
struct ShadowRayTriangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct ShadowRayMaterial { uint4 DiffuseAlpha,OpacityRed; float4 Coverage; };
static const float ShadowRayPixelCenter=0.5;
static const float ShadowRayFloatEpsilon=1.1920928955078125e-7;
static const float ShadowRayEdgeError=(3*ShadowRayFloatEpsilon)/(1-3*ShadowRayFloatEpsilon);
ShadowRayNode ShadowRayReadNode(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).x+index*ShadowRayNodeSize;
    ShadowRayNode result;
    result.Minimum=asfloat(ShadowRayLoad(base));
    result.Maximum=asfloat(ShadowRayLoad(base+1u));
    result.Links=ShadowRayLoad(base+2u);
    return result;
}
ShadowRayTriangle ShadowRayReadTriangle(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).y+index*ShadowRayTriangleSize;
    ShadowRayTriangle result;
    result.A=asfloat(ShadowRayLoad(base));
    result.B=asfloat(ShadowRayLoad(base+1u));
    result.C=asfloat(ShadowRayLoad(base+2u));
    result.UVAB=asfloat(ShadowRayLoad(base+3u));
    result.UVC=asfloat(ShadowRayLoad(base+4u));
    result.Metadata=ShadowRayLoad(base+5u);
    return result;
}
ShadowRayMaterial ShadowRayReadMaterial(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).z+index*ShadowRayMaterialSize;
    ShadowRayMaterial result;
    result.DiffuseAlpha=ShadowRayLoad(base);
    result.OpacityRed=ShadowRayLoad(base+1u);
    result.Coverage=asfloat(ShadowRayLoad(base+2u));
    return result;
}
float ShadowRayScalar(uint index)
{ uint4 value=ShadowRayLoad(ShadowRayLoad(ShadowRayOffsetsHeader).w+index/ShadowRayScalarLanes); return asfloat(value)[index%ShadowRayScalarLanes]; }
float ShadowRayCoverage(uint4 image,float2 uv)
{
    if (image.w==0) return 1;
    float2 texel=frac(float2(uv.x,-uv.y))*float2(image.yz)-ShadowRayPixelCenter;
    uint2 a=uint2(texel.x<0 ? image.y-1 : uint(floor(texel.x)),texel.y<0 ? image.z-1 : uint(floor(texel.y)));
    uint2 b=uint2(a.x+1==image.y ? 0 : a.x+1,a.y+1==image.z ? 0 : a.y+1);
    float2 weight=frac(texel);
    float top=lerp(ShadowRayScalar(image.x+a.y*image.y+a.x),ShadowRayScalar(image.x+a.y*image.y+b.x),weight.x);
    float bottom=lerp(ShadowRayScalar(image.x+b.y*image.y+a.x),ShadowRayScalar(image.x+b.y*image.y+b.x),weight.x);
    return lerp(top,bottom,weight.y);
}
bool ShadowRayBoxHit(ShadowRayNode node,float3 origin,float3 direction,float radius)
{
    float enter=0,leave=radius;
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (direction[axis]==0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float a=(node.Minimum[axis]-origin[axis])/direction[axis],b=(node.Maximum[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(a,b)); leave=min(leave,max(a,b)); if (enter>leave) return false;
        }
    }
    return true;
}
float ShadowRayEdge(float2 a,float2 b)
{
    precise float forward=a.x*b.y,backward=a.y*b.x;
    precise float value=forward-backward;
    return abs(value)<=ShadowRayEdgeError*(abs(forward)+abs(backward)) ? 0 : value;
}
bool ShadowRayTriangleHit(ShadowRayTriangle primitive,float3 origin,float3 direction,float radius,out float3 weights)
{
    weights=0;
    uint z=abs(direction.x)>abs(direction.y) ? 0 : 1; if (abs(direction.z)>abs(direction[z])) z=2;
    uint x=(z+1)%3,y=(x+1)%3;
    float3 a=primitive.A.xyz-origin,b=primitive.B.xyz-origin,c=primitive.C.xyz-origin;
    float2 shear=float2(direction[x],direction[y])/direction[z];
    precise float2 pa=float2(a[x],a[y])-shear*a[z],pb=float2(b[x],b[y])-shear*b[z],pc=float2(c[x],c[y])-shear*c[z];
    float3 edges=float3(ShadowRayEdge(pb,pc),ShadowRayEdge(pc,pa),ShadowRayEdge(pa,pb));
    if (any(edges<0) && any(edges>0)) return false;
    float area=edges.x+edges.y+edges.z; if (area==0) return false;
    float distance=dot(edges,float3(a[z],b[z],c[z]))/(area*direction[z]);
    if (!(distance>0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool ShadowRayOccluded(float3 origin,float3 direction,float radius)
{
    bool blocked=false;
    uint nodeIndex=0;
    [loop] while (nodeIndex<ShadowRayLoad(0u).y && !blocked)
    {
        ShadowRayNode node=ShadowRayReadNode(nodeIndex);
        if (!ShadowRayBoxHit(node,origin,direction,radius)) { nodeIndex=node.Links.w; continue; }
        if (node.Links.z==0) { nodeIndex=node.Links.x; continue; }
        [loop] for (uint index=0;index<node.Links.z && !blocked;++index)
        {
            ShadowRayTriangle primitive=ShadowRayReadTriangle(node.Links.y+index);
            if ((primitive.Metadata.w & ShadowRayCasterFlag)==0u) continue; float3 weights;
            if (!ShadowRayTriangleHit(primitive,origin,direction,radius,weights)) continue;
            ShadowRayMaterial material=ShadowRayReadMaterial(primitive.Metadata.x); if (material.Coverage.y<=0) { blocked=true; break; }
            float2 uv=primitive.UVAB.xy*weights.x+primitive.UVAB.zw*weights.y+primitive.UVC.xy*weights.z;
            float alpha=saturate(dot(weights,float3(primitive.A.w,primitive.B.w,primitive.C.w))*material.Coverage.x*
                ShadowRayCoverage(material.DiffuseAlpha,uv)*ShadowRayCoverage(material.OpacityRed,uv));
            if (alpha>=material.Coverage.y) { blocked=true; break; }
        }
        nodeIndex=node.Links.w;
    }
    return blocked;
}
static const float ShadowRayNormalFloor=1e-12,ShadowRayOriginRoundoff=8.0*1.1920928955078125e-7;
bool ShadowRayEnabled() { return ShadowRayLoad(ShadowRayCountsHeader).x!=0u && ShadowRayLoad(ShadowRayLightHeader).y==ShadowRayImageABI; }
float ShadowRayVisibility(float3 position,float3 normal,float3 towardLight)
{
    if (!ShadowRayEnabled()) return 1.0;
    float4 parameters=asfloat(ShadowRayLoad(ShadowRayTraceHeader));
    float scale=max(abs(towardLight.x),max(abs(towardLight.y),abs(towardLight.z)));
    if (!(scale>0.0) || !(all(isfinite(float4(position,scale))))) return 1.0;
    float3 scaled=towardLight/scale; float magnitude=length(scaled);
    float3 direction=scaled/magnitude;
    bool finiteEmitter=ShadowRayLoad(ShadowRayLightHeader).x!=0u;
    float limit=finiteEmitter?min(scale*magnitude,parameters.z):parameters.z;
    if (!(limit>parameters.y)) return 1.0;
    float normalSquared=dot(normal,normal);
    float3 unitNormal=normalSquared>ShadowRayNormalFloor?normal*rsqrt(normalSquared):0.0;
    float roundoff=ShadowRayOriginRoundoff*max(abs(position.x),max(abs(position.y),abs(position.z)));
    float3 normalOffset=unitNormal*max(parameters.x,roundoff);
    if (finiteEmitter)
    {
        // Bias the origin while preserving the actual emitter endpoint.
        float3 endpointVector=towardLight-normalOffset;
        float endpointScale=max(abs(endpointVector.x),max(abs(endpointVector.y),abs(endpointVector.z)));
        if (!(endpointScale>0.0)) return 1.0;
        float3 endpointUnit=endpointVector/endpointScale;
        float endpointMagnitude=length(endpointUnit);
        direction=endpointUnit/endpointMagnitude;
        limit=min(endpointScale*endpointMagnitude,parameters.z);
    }
    float3 origin=position+normalOffset+direction*parameters.y;
    limit-=parameters.y;
    if (!(limit>0.0)) return 1.0;
    if (!(all(isfinite(origin)))) return 1.0;
    float visibility=1.0;
    if (ShadowRayOccluded(origin,direction,limit)) visibility=0.0;
    return visibility;
}

#endif

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

// Tiled deferred shading: G-buffer depth -> per-tile lists -> reconstructed lighting.
// Captures and transparent geometry retain independent complete-light paths.
// Lauritzen, "Deferred Rendering for Current and Future Rendering Pipelines" (SIGGRAPH 2010).
// https://www.intel.com/content/www/us/en/developer/articles/technical/deferred-rendering-for-current-and-future-rendering-pipelines.html
// DX11 Tiled Deferred material consumer and alpha-tested depth prerequisite.
// Tile indices and structured light records share immutable scene order.
// Overflow, capture cameras and explicit reference draws use the full list.
// No light is dropped merely because the tile's fixed storage is exhausted.
#if defined(EVANS_TILED_DEFERRED_SURFACE)
#ifndef EVANS_TILED_DEFERRED_FILTER
#define EVANS_TILED_DEFERRED_FILTER 0
#endif
#define FILTER_PCSS 2
#define FILTER_VSM 3
static const uint ShadingPhong = 0, ShadingFlat = 1, ShadingPBR = 3, ShadingToon = 4;
static const uint InlineLightCapacity = 4, CascadeCapacity = 4, MapCapacity = 5;
static const uint DirectionalLight = 0, PointLight = 1;
static const uint FullSceneList = 0, OpaqueList = 1, TransparentList = 2;
static const float Pi = 3.14159265358979323846;
static const float NormalEpsilon = 1.0e-12, ProjectionEpsilon = 1.0e-7;
static const float HalfRadianceLimit = 65504.0, PixelCenter = 0.5;
static const float MaximumMapFraction = 0.25;
static const float SurfaceShadowStrength = 0.55;
static const float MinimumCosine = 0.15, ConstantBias = 0.00035, SlopeBias = 0.00060, MaximumBias = 0.0045;
struct Light { float4 PositionType; float4 DirectionIntensity; float4 ColorCastShadows; float4 AreaSize; };
cbuffer DrawConstants : register(b0)
{
    row_major float4x4 World, NormalWorld, ViewProjection, LightViewProjection;
    float4 PipelineColor, MaterialDiffuseColor, MaterialAmbientColor, MaterialSpecularColor, MaterialEmissiveColor;
    float4 MaterialParameters, AmbientColorStrength, CameraLightCount, RenderParameters;
    uint4 TextureFlags;
    Light InlineLights[InlineLightCapacity];
    float4 ShadingParameters;
};
cbuffer ForwardLightConstants : register(b7)
{
    uint4 LightLayout; // full count, full shadow index (UINT_MAX = none), list mode, capacity
    uint4 SurfaceTiles; // tile size, tiles X, viewport width, viewport height
};
StructuredBuffer<Light> SceneLights : register(t16);
StructuredBuffer<uint4> SurfaceTileHeaders : register(t17);
StructuredBuffer<uint> SurfaceLightIndices : register(t18);
cbuffer CascadedShadowConstants : register(b2)
{
    row_major float4x4 CascadeLightViewProjection[CascadeCapacity];
    float4 CascadeFarDistances, CascadeBlendStarts, CascadeEyeDepthPlane, CascadeParameters, ShadowFiltering;
};
#if EVANS_TILED_DEFERRED_FILTER == FILTER_VSM
Texture2D<float2> ShadowMap : register(t0);
Texture2D<float2> CascadeMaps[CascadeCapacity] : register(t10);
cbuffer ShadowReceiverConstants : register(b5)
{
    float4 DepthEncoding[MapCapacity];
    float4 MomentParameters;
    uint4 MomentControl;
};
#else
Texture2D<float> ShadowMap : register(t0);
Texture2D<float> CascadeMaps[CascadeCapacity] : register(t10);
#if EVANS_TILED_DEFERRED_FILTER == FILTER_PCSS
cbuffer ShadowReceiverConstants : register(b5)
{
    float4 ProjectionMetrics[MapCapacity];
    float4 EmitterAndLimits;
    uint4 SamplesAndMaps;
};
#endif
#endif
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
cbuffer ContactReceiverConstants : register(b6) { uint4 ContactControl; };
Texture2DArray<float> ContactMask : register(t15);
float ReadContact(float4 position, uint sampleIndex)
{
    return ContactControl.x != 0 ? ContactMask.Load(int4(int2(position.xy), sampleIndex, 0)) : 1;
}
float ComposeContact(float primary, float contact)
{ return lerp(1 - SurfaceShadowStrength, primary, contact); }
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

float3 SafeNormal(float3 value, float3 fallback)
{ float squared = dot(value, value); return squared > NormalEpsilon ? value * rsqrt(squared) : fallback; }
uint2 MapSize(uint map)
{
    uint width = 1, height = 1;
    if (map == 0) ShadowMap.GetDimensions(width, height);
    else CascadeMaps[0].GetDimensions(width, height);
    return uint2(width, height);
}
#if EVANS_TILED_DEFERRED_FILTER == FILTER_VSM
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
#elif EVANS_TILED_DEFERRED_FILTER == FILTER_PCSS
float DepthAt(uint map, int2 pixel, uint2 size)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return 1;
    // SM5 resource arrays require literal indices. Map zero is the single map.
    if (map == 0) return ShadowMap.Load(int3(pixel, 0));
    if (map == 1) return CascadeMaps[0].Load(int3(pixel, 0));
    if (map == 2) return CascadeMaps[1].Load(int3(pixel, 0));
    if (map == 3) return CascadeMaps[2].Load(int3(pixel, 0));
    return CascadeMaps[3].Load(int3(pixel, 0));
}
float LinearDistance(float depth, float4 metric)
{ return metric.x > 0 ? rcp((1 - depth) / metric.x + depth / metric.y) : depth * metric.y; }
float2 DiskPoint(uint index, uint count)
{
    const float GoldenAngle = Pi * (3 - sqrt(5.0));
    float angle = index * GoldenAngle;
    return sqrt((index + PixelCenter) / count) * float2(cos(angle), sin(angle));
}
float2 BoundedRadius(float worldRadius, float4 metric, uint2 size, float budget)
{
    float2 radius = worldRadius / metric.zw;
    float extent = max(radius.x * size.x, radius.y * size.y);
    float limit = min(budget, min(size.x, size.y) * MaximumMapFraction);
    return extent > limit ? radius * (limit / extent) : radius;
}
float FilterComparison(uint map, float2 uv, uint2 size, float receiver)
{
    float2 texel = uv * size - PixelCenter;
    int2 base = int2(floor(texel)); float2 weight = frac(texel);
    float a = DepthAt(map, base, size) >= receiver;
    float b = DepthAt(map, base + int2(1, 0), size) >= receiver;
    float c = DepthAt(map, base + int2(0, 1), size) >= receiver;
    float d = DepthAt(map, base + int2(1, 1), size) >= receiver;
    return lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
}
float MapVisibility(uint map, float4 clip, float bias)
{
    if ((SamplesAndMaps.z & (1u << map)) == 0 || clip.w <= ProjectionEpsilon) return 1;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1) return 1;
    uint2 size = MapSize(map);
    float4 metric = ProjectionMetrics[map];
    float comparison = projected.z - bias;
    float emitter = metric.x > 0 ? EmitterAndLimits.x : EmitterAndLimits.y;
    if (emitter == 0) return DepthAt(map, int2(floor(uv * size)), size) >= comparison;
    float receiver = LinearDistance(projected.z, metric);
    float searchWorld = metric.x > 0 ? emitter * (receiver - metric.x) / receiver : emitter * receiver;
    float2 search = BoundedRadius(searchWorld, metric, size, EmitterAndLimits.z);
    float blockerSum = 0; uint blockers = 0;
    [loop] for (uint sampleIndex = 0; sampleIndex < SamplesAndMaps.x; ++sampleIndex)
    {
        float2 sampleUV = uv + DiskPoint(sampleIndex, SamplesAndMaps.x) * search;
        float depth = DepthAt(map, int2(floor(sampleUV * size)), size);
        if (depth < comparison) { blockerSum += LinearDistance(depth, metric); ++blockers; }
    }
    if (blockers == 0) return 1;
    float meanBlocker = blockerSum / blockers;
    float gap = max(receiver - meanBlocker, 0);
    float penumbra = metric.x > 0 ? emitter * gap / meanBlocker * metric.x / receiver : emitter * gap;
    float2 radius = BoundedRadius(penumbra, metric, size, EmitterAndLimits.w);
    float visibility = 0;
    [loop] for (uint filterIndex = 0; filterIndex < SamplesAndMaps.y; ++filterIndex)
        visibility += FilterComparison(map, uv + DiskPoint(filterIndex, SamplesAndMaps.y) * radius, size, comparison);
    return visibility / SamplesAndMaps.y;
}
#else
float DepthAt(uint map, int2 pixel, uint2 size)
{
    const int2 clampedPixel = clamp(pixel, 0, int2(size) - 1);
    float depth = 1;
    if (map == 0) depth = ShadowMap.Load(int3(clampedPixel, 0));
    else if (map == 1) depth = CascadeMaps[0].Load(int3(clampedPixel, 0));
    else if (map == 2) depth = CascadeMaps[1].Load(int3(clampedPixel, 0));
    else if (map == 3) depth = CascadeMaps[2].Load(int3(clampedPixel, 0));
    else depth = CascadeMaps[3].Load(int3(clampedPixel, 0));
    return depth;
}
float MapVisibility(uint map, float4 clip, float bias)
{
    if (clip.w <= ProjectionEpsilon) return 1;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1) return 1;
    uint2 size = MapSize(map); float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0) return DepthAt(map, int2(floor(uv * size)), size) >= receiver;
    float2 texel = uv * size - PixelCenter, weight = frac(texel);
    int2 base = int2(floor(texel)); float visibility = 0;
    [loop] for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        [loop] for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = base + int2(x, y);
            float a = DepthAt(map, p, size) >= receiver;
            float b = DepthAt(map, p + int2(1, 0), size) >= receiver;
            float c = DepthAt(map, p + int2(0, 1), size) >= receiver;
            float d = DepthAt(map, p + 1, size) >= receiver;
            visibility += lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        }
    return visibility / ShadowFiltering.y;
}
#endif
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

// Virtual-map receiver ABI reuses the mutually exclusive cascade block:
// Far = logical resolution/page side/mips/directory width; Blend = atlas extent/
// columns/directory denominator. Every PCF corner resolves its own ancestor.
static const uint VirtualPageBorder = 1;
static const float VirtualTexelCenter = 0.5f;
float2 VirtualDepth(float2 uv, uint requestedMip)
{
    uint4 logical = uint4(CascadeFarDistances), physical = uint4(CascadeBlendStarts);
    uint side = logical.x / logical.y, base = 0;
    [loop] for (uint mip = 0; mip < logical.z; ++mip)
    {
        if (mip >= requestedMip)
        {
            uint2 pixel = min(uint2(uv * (side * logical.y)), side * logical.y - 1);
            uint2 page = pixel / logical.y;
            uint address = base + page.y * side + page.x;
            uint entry = uint(round(CascadeMaps[1].Load(int3(address % logical.w, address / logical.w, 0)).x * physical.w));
            if (entry != 0)
            {
                uint slot = entry - 1, stride = logical.y + VirtualPageBorder * 2;
                uint2 tile = uint2(slot % physical.z, slot / physical.z) * stride;
                return float2(CascadeMaps[0].Load(int3(tile + VirtualPageBorder + pixel % logical.y, 0)).x, mip);
            }
        }
        base += side * side; side /= 2;
    }
    return float2(1, logical.z - 1);
}
// Compare receiver and caster at the same resolved virtual texel center.
// The actual ancestor mip is essential when detail pages are not resident.
float2 VirtualReceiverGradient(float2 uv, float depth)
{
    float2 x = ddx(uv), y = ddy(uv);
    float zx = ddx(depth), zy = ddy(depth);
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x, x) * dot(y, y), 1.0e-30);
    if (determinant * determinant <= scale * 1.0e-12) return float2(0.0, 0.0);
    return float2(y.y * zx - x.y * zy, x.x * zy - y.x * zx) / determinant;
}
float VirtualReceiverAt(float2 sampleUv, uint actualMip, float2 receiverUv, float receiver, float2 gradient)
{
    float resolution = float(uint(CascadeFarDistances.x) >> actualMip);
    float2 pixel = clamp(floor(sampleUv * resolution), float2(0.0, 0.0), float2(resolution - 1.0, resolution - 1.0));
    float2 center = (pixel + float2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualCompare(int2 pixel, uint mip, float2 receiverUv, float receiver, float2 gradient)
{
    uint resolution = uint(CascadeFarDistances.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0, 0), int2(int(resolution) - 1, int(resolution) - 1))) + float2(0.5, 0.5)) / float(resolution);
    float2 stored = VirtualDepth(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
float VirtualVisibility(float3 position, float bias, float3 receiverData)
{
    float4 light = mul(CascadeLightViewProjection[0], float4(position, 1.0));
    float3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    float2 uv = float2(projected.x, -projected.y) * 0.5 + float2(0.5, 0.5);
    float2 gradient = receiverData.xy;
    uint mip = uint(receiverData.z);
    if (light.w <= CascadeEyeDepthPlane.w || any(uv < 0.0) || any(uv > 1.0) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float2 stored = VirtualDepth(uv, mip);
    float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(CascadeFarDistances.x) >> mip) - float2(0.5, 0.5);
    int2 first = int2(floor(texel)); float2 weight = frac(texel); float visible = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = first + int2(x, y);
            float a = VirtualCompare(p, mip, uv, receiver, gradient);
            float b = VirtualCompare(p + int2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualCompare(p + int2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualCompare(p + int2(1, 1), mip, uv, receiver, gradient);
            visible += lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        }
    return visible / ShadowFiltering.y;
}



// Resolve the receiver plane before entering a varying light-list loop.
float3 VirtualReceiverData(float3 world)
{
    if (ShadowFiltering.z == 0) return float3(0.0, 0.0, 0.0);
    float4 light = mul(CascadeLightViewProjection[0], float4(world, 1));
    float2 uv = light.xy / max(light.w, CascadeEyeDepthPlane.w) * float2(0.5, -0.5);
    float footprint = max(length(ddx(uv) * CascadeEyeDepthPlane.y), length(ddy(uv) * CascadeEyeDepthPlane.z)) * CascadeFarDistances.x;
    return float3(VirtualReceiverGradient(uv, light.z / max(light.w, CascadeEyeDepthPlane.w)), clamp(floor(log2(max(1, footprint)) + CascadeEyeDepthPlane.x), 0, CascadeFarDistances.z - 1));
}

float SurfaceVisibility(float3 world, float3 normal, float3 toLight, float3 receiverData)
{
    float result=1.0;
    if (ShadowRayEnabled()) result=1.0-SurfaceShadowStrength*(1.0-ShadowRayVisibility(world,normal,toLight));
    else
    {
    float cosine = max(dot(SafeNormal(normal, float3(0, 1, 0)), SafeNormal(toLight, float3(0, 1, 0))), 0);
    float slope = sqrt(max(1 - cosine * cosine, 0)) / max(cosine, MinimumCosine);
    float bias = min(ConstantBias + SlopeBias * slope, MaximumBias);
    float visibility = ShadowFiltering.z != 0 ? VirtualVisibility(world, bias, receiverData) : PrimaryVisibility(world, bias, CascadeParameters.z > 0);
    result = 1 - SurfaceShadowStrength * (1 - visibility);

    }
    return result;
}
struct MaterialSurface
{
    float3 Position, Normal, View, Diffuse, Specular;
    float Roughness, Metallic, Shininess, Bands;
    uint Model;
    float3 VirtualReceiver;
};
float3 LightSample(MaterialSurface surface, float3 toLight, float3 radiance)
{
    const float MinimumRoughness = 0.045, DielectricReflectance = 0.04;
    const float BrdfEpsilon = 1.0e-6, DistributionEpsilon = 1.0e-12, FresnelExponent = 5;
    const float MinimumBands = 2;
    const float MaximumBands = 16;
    if (surface.Model <= ShadingFlat && dot(toLight, toLight) <= NormalEpsilon) return 0;
    float3 direction = SafeNormal(toLight, surface.Normal);
    float noL = saturate(dot(surface.Normal, direction));
    float3 halfway = SafeNormal(direction + surface.View, surface.Normal);
    float3 result = 0;
    if (surface.Model == ShadingPBR)
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
            result = radiance * noL * (surface.Diffuse * (1 - fresnel) * (1 - surface.Metallic) / Pi +
                distribution * smithV * smithL * fresnel / max(4 * noV * noL, BrdfEpsilon));
        }
    }
    else
    {
        float specular = noL > 0 ? pow(saturate(dot(surface.Normal, halfway)), max(surface.Shininess, 1)) : 0;
        if (surface.Model == ShadingToon)
        {
            float bands = clamp(round(surface.Bands), MinimumBands, MaximumBands);
            result = radiance * (surface.Diffuse * ToonResponse(noL, bands) + surface.Specular * ToonHighlight(specular));
        }
        else result = radiance * (surface.Diffuse * noL + surface.Specular * specular);
    }
    return result;
}
float3 SceneLight(MaterialSurface surface, uint index, bool shadows, int shadowIndex, float contactVisibility)
{
    const float DistanceEpsilon = 1.0e-6, ParallelThreshold = 0.999;
    const int AreaSamplesPerAxis = 2;
    const float AreaSampleOffset = 0.25, AreaSampleWeight = 0.25;
    Light light = SceneLights[index]; uint type = uint(light.PositionType.w + 0.5);
    float3 shadowDirection = type == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - surface.Position;
    float visibility = shadows && int(index) == shadowIndex && light.ColorCastShadows.w > 0.5 ?
        SurfaceVisibility(surface.Position, surface.Normal, shadowDirection, surface.VirtualReceiver) : 1;
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
struct MeshVertex { float3 Position : POSITION; float3 Normal : NORMAL; float4 Color : COLOR0; float2 TexCoord : TEXCOORD0; };
struct SurfaceVertex
{
    precise float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0; float3 Normal : TEXCOORD1;
    float4 Color : COLOR0; float4 LightPosition : TEXCOORD2; float2 TexCoord : TEXCOORD3;
};
float3 SurfaceNormal(SurfaceVertex input, bool front)
{
    const float TangentEpsilon = 1.0e-10, ParallelThreshold = 0.999;
    float3 normal = SafeNormal(input.Normal, float3(0, 1, 0));
    if (TextureFlags.w == ShadingFlat)
    {
        normal = SafeNormal(cross(ddx(input.WorldPosition), ddy(input.WorldPosition)), normal);
        if (dot(normal, input.Normal) < 0) normal = -normal;
    }

    if (!front) normal = -normal;
    if (TextureFlags.w == ShadingFlat) return normal;

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
SurfaceVertex VSMain(MeshVertex input)
{
    SurfaceVertex output;
    float4 world = mul(World, float4(input.Position, 1));
    output.Position = mul(ViewProjection, world);
    output.WorldPosition = world.xyz / max(abs(world.w), ProjectionEpsilon);
    output.Normal = SafeNormal(mul((float3x3)NormalWorld, input.Normal), float3(0, 1, 0));
    output.Color = float4(DecodeSRGB(saturate(input.Color.rgb)), input.Color.a);
    output.LightPosition = mul(LightViewProjection, world);
    output.TexCoord = float2(input.TexCoord.x, 1 - input.TexCoord.y);
    return output;
}
float SurfaceAlpha(SurfaceVertex input, float4 textureColor)
{
    float opacity = TextureFlags.z != 0 ? OpacityMap.Sample(SurfaceSampler, input.TexCoord).r : 1;
    float alpha = saturate(input.Color.a * MaterialParameters.y * textureColor.a * opacity);
    if (MaterialParameters.z > 0) clip(alpha - MaterialParameters.z);
    return alpha;
}
void PSDepth(SurfaceVertex input)
{
    float4 textureColor = TextureFlags.x != 0 ? DiffuseMap.Sample(SurfaceSampler, input.TexCoord) : 1;
    if (float(TextureFlags.w) == ToonShadingModel && TextureFlags.x != 0u)
        textureColor.rgb = ToonTextureColor(DiffuseMap, SurfaceSampler, input.TexCoord, ddx(input.TexCoord), ddy(input.TexCoord));
    SurfaceAlpha(input, textureColor);
}
// A header stores total accepted lights, including overflow. Select an ordered
// tile slice only when every accepted index fits; otherwise use the full scene.
uint2 SurfaceSelection(float4 position, out bool indexed)
{
    indexed = false;
    uint2 selection = uint2(0, LightLayout.x);
    if (LightLayout.z != FullSceneList)
    {
        uint2 pixel = min(uint2(position.xy), SurfaceTiles.zw - 1);
        uint2 tile = pixel / SurfaceTiles.x;
        uint4 header = SurfaceTileHeaders[tile.y * SurfaceTiles.y + tile.x];
        uint2 candidate = LightLayout.z == TransparentList ? header.zw : header.xy;
        if (candidate.y <= LightLayout.w) { selection = candidate; indexed = true; }
    }
    return selection;
}
float4 ShadeSurface(SurfaceVertex input, bool front, uint contactSample)
{
    float2 roughnessMetallic = ShadingParameters.xy;
    if (TextureFlags.z > 1) roughnessMetallic *= OpacityMap.Sample(SurfaceSampler, input.TexCoord).gb;
    float4 textureColor = TextureFlags.x != 0 ? DiffuseMap.Sample(SurfaceSampler, input.TexCoord) : 1;
    if (float(TextureFlags.w) == ToonShadingModel && TextureFlags.x != 0u)
        textureColor.rgb = ToonTextureColor(DiffuseMap, SurfaceSampler, input.TexCoord, ddx(input.TexCoord), ddy(input.TexCoord));
    float alpha = SurfaceAlpha(input, textureColor);
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
        MaterialSurface surface = (MaterialSurface)0;
        surface.Position = input.WorldPosition; surface.VirtualReceiver = VirtualReceiverData(surface.Position); surface.Normal = SurfaceNormal(input, front);
        surface.View = SafeNormal(CameraLightCount.xyz - surface.Position, surface.Normal);
        surface.Diffuse = base * max(MaterialDiffuseColor.rgb, 0); surface.Specular = max(MaterialSpecularColor.rgb, 0);
        surface.Roughness = roughnessMetallic.x; surface.Metallic = saturate(roughnessMetallic.y);
        surface.Shininess = MaterialParameters.x; surface.Bands = ShadingParameters.z; surface.Model = TextureFlags.w;
        color = ToonAmbient(base * max(MaterialDiffuseColor.rgb, 0.0), base * max(MaterialAmbientColor.rgb, 0) * max(AmbientColorStrength.rgb, 0) * max(AmbientColorStrength.w, 0), float(TextureFlags.w));
        if (surface.Model == ShadingPBR) color *= 1 - surface.Metallic;
        color += max(MaterialEmissiveColor.rgb, 0);
        bool indexed = false; uint2 selection = SurfaceSelection(input.Position, indexed);
        float contact = ReadContact(input.Position, contactSample);
        [loop] for (uint entry = 0; entry < selection.y; ++entry)
        {
            uint index = indexed ? SurfaceLightIndices[selection.x + entry] : entry;
            color += SceneLight(surface, index, RenderParameters.x > 0.5, int(LightLayout.y), contact);
        }
        color += surface.Diffuse * (surface.Model==ShadingPBR ? (1-LumenDielectricReflectance)*(1-surface.Metallic) : 1) * LumenTranslucentIrradiance(surface.Position,surface.Normal);
        color = max(SurfaceFog(color, input.Position), 0);
        color = ShadingParameters.w > 0.5 ? min(color, HalfRadianceLimit) : saturate(color / max(1, max(color.r, max(color.g, color.b))));
    }
    return float4(color, alpha);
}
float4 PSMain(SurfaceVertex input, bool front : SV_IsFrontFace) : SV_TARGET
{ return ShadeSurface(input, front, 0); }
float4 PSContact(SurfaceVertex input, bool front : SV_IsFrontFace, uint sampleIndex : SV_SampleIndex) : SV_TARGET
{ return ShadeSurface(input, front, sampleIndex); }
struct WeightedOutput { float4 Accumulation : SV_Target0; float Revealage : SV_Target1; };
WeightedOutput PSWeighted(SurfaceVertex input, bool front : SV_IsFrontFace)
{
    float4 color = ShadeSurface(input, front, 0); WeightedOutput output = (WeightedOutput)0;
    float alpha = saturate(color.a); if (alpha == 0) return output;
    float weight = max(pow(alpha, TransparencyWeights.x) * pow(max(1 - saturate(input.Position.z) * TransparencyWeights.z, 0), TransparencyWeights.y), TransparencyWeights.w);
    output.Accumulation = float4(color.rgb * alpha * weight, alpha * weight); output.Revealage = alpha;
    return output;
}
#if defined(EVANS_TILED_DEFERRED_LIGHTING)
cbuffer TiledReconstruction : register(b1)
{
    row_major float4x4 InverseViewProjection;
    float4 BackgroundColor;
};
#if defined(EVANS_MULTISAMPLE)
Texture2DMS<float4> DiffuseOpacity : register(t4);
Texture2DMS<float4> NormalRoughness : register(t5);
Texture2DMS<float4> SpecularShininess : register(t6);
Texture2DMS<float4> AmbientMetallic : register(t7);
Texture2DMS<float4> EmissionColor : register(t8);
Texture2DMS<float> OpaqueDepth : register(t9);
#else
Texture2D<float4> DiffuseOpacity : register(t4);
Texture2D<float4> NormalRoughness : register(t5);
Texture2D<float4> SpecularShininess : register(t6);
Texture2D<float4> AmbientMetallic : register(t7);
Texture2D<float4> EmissionColor : register(t8);
Texture2D<float> OpaqueDepth : register(t9);
#endif
// Reconstruct one-sided depth slopes on the same geometric surface.
// Fragment-quad derivatives can cross a G-buffer silhouette or intersection.
// Camera depth is affine within a rasterized triangle; lower second difference
// chooses the side that best preserves that plane without using shading normals.
float DeferredReceiverDepth(int2 pixel, uint sampleIndex, uint2 size)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return 1.0;
#if defined(EVANS_MULTISAMPLE)
    return OpaqueDepth.Load(pixel, sampleIndex);
#else
    return OpaqueDepth.Load(int3(pixel, 0));
#endif
}
float DeferredReceiverSlope(float center, float negative, float negativeTwo, float positive, float positiveTwo)
{
    bool leftValid = negative < 1.0 && negativeTwo < 1.0;
    bool rightValid = positive < 1.0 && positiveTwo < 1.0;
    if (!leftValid && !rightValid) return 0.0;
    float leftError = abs(center - 2.0 * negative + negativeTwo);
    float rightError = abs(center - 2.0 * positive + positiveTwo);
    return leftValid && (!rightValid || leftError <= rightError) ? center - negative : positive - center;
}
float3 DeferredReceiverLightPoint(float2 coordinate, float depth)
{
    float4 world = mul(InverseViewProjection, float4(coordinate.x * 2.0 - 1.0, 1.0 - coordinate.y * 2.0, depth, 1.0));
    float4 light = mul(CascadeLightViewProjection[0], float4(world.xyz / world.w, 1.0));
    float3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    return float3(projected.xy * float2(0.5, -0.5) + 0.5, projected.z);
}
float3 DeferredVirtualReceiverData(float2 coordinate, float depth, uint sampleIndex)
{
    if (ShadowFiltering.z == 0) return 0.0;
    uint width, height;
#if defined(EVANS_MULTISAMPLE)
    uint sampleCount; OpaqueDepth.GetDimensions(width, height, sampleCount);
#else
    OpaqueDepth.GetDimensions(width, height);
#endif
    uint2 size = uint2(width, height); int2 pixel = int2(coordinate * float2(size));
    float dx = DeferredReceiverSlope(depth,
        DeferredReceiverDepth(pixel + int2(-1, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(-2, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(1, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(2, 0), sampleIndex, size));
    float dy = DeferredReceiverSlope(depth,
        DeferredReceiverDepth(pixel + int2(0, -1), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(0, -2), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(0, 1), sampleIndex, size),
        DeferredReceiverDepth(pixel + int2(0, 2), sampleIndex, size));
    float3 center = DeferredReceiverLightPoint(coordinate, depth);
    float3 x = DeferredReceiverLightPoint(coordinate + float2(1.0 / width, 0), depth + dx) - center;
    float3 y = DeferredReceiverLightPoint(coordinate + float2(0, 1.0 / height), depth + dy) - center;
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x.xy, x.xy) * dot(y.xy, y.xy), 1.0e-30);
    float2 gradient = determinant * determinant > scale * 1.0e-12
        ? float2(y.y * x.z - x.y * y.z, x.x * y.z - y.x * x.z) / determinant : float2(0, 0);
    float footprint = max(length(x.xy * CascadeEyeDepthPlane.y), length(y.xy * CascadeEyeDepthPlane.z)) * CascadeFarDistances.x;
    float mip = clamp(floor(log2(max(1.0, footprint)) + CascadeEyeDepthPlane.x), 0.0, CascadeFarDistances.z - 1.0);
    return float3(gradient, mip);
}

struct LightingVertex { float4 Position : SV_POSITION; float2 UV : TEXCOORD0; };
LightingVertex VSLighting(uint vertex : SV_VertexID)
{
    LightingVertex result;
    result.UV = float2((vertex << 1u) & 2u, vertex & 2u);
    result.Position = float4(result.UV * float2(2, -2) + float2(-1, 1), 0, 1);
    return result;
}
#if defined(EVANS_MULTISAMPLE)
float4 PSLighting(LightingVertex input, uint sampleIndex : SV_SampleIndex) : SV_TARGET
#else
float4 PSLighting(LightingVertex input) : SV_TARGET
#endif
{
    const float ClearDepth = 1.0, PixelCenter = 0.5;
#if defined(EVANS_MULTISAMPLE)
    int2 pixel = int2(input.Position.xy);
    float depth = OpaqueDepth.Load(pixel, sampleIndex);
    float4 normal = NormalRoughness.Load(pixel, sampleIndex);
    float4 diffuse = DiffuseOpacity.Load(pixel, sampleIndex);
    float4 specular = SpecularShininess.Load(pixel, sampleIndex);
    float4 ambient = AmbientMetallic.Load(pixel, sampleIndex);
    float3 emission = EmissionColor.Load(pixel, sampleIndex).rgb;
    uint width, height, samples; OpaqueDepth.GetDimensions(width, height, samples);
    float2 uv = (float2(pixel) + PixelCenter + OpaqueDepth.GetSamplePosition(sampleIndex)) / float2(width, height);
#else
    const uint sampleIndex = 0;
    int3 pixel = int3(int2(input.Position.xy), 0);
    float depth = OpaqueDepth.Load(pixel);
    float4 normal = NormalRoughness.Load(pixel);
    float4 diffuse = DiffuseOpacity.Load(pixel);
    float4 specular = SpecularShininess.Load(pixel);
    float4 ambient = AmbientMetallic.Load(pixel);
    float3 emission = EmissionColor.Load(pixel).rgb;
    float2 uv = input.UV;
#endif
    // Unlit overlays may cover clear depth; their material marker takes priority.
    if (normal.w < 0) return float4(emission, diffuse.a);
    if (depth >= ClearDepth) return BackgroundColor;
    float4 world = mul(InverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    MaterialSurface surface = (MaterialSurface)0;
    surface.Position = world.xyz / world.w; surface.VirtualReceiver = DeferredVirtualReceiverData(uv, depth, sampleIndex);
    surface.Normal = SafeNormal(normal.xyz, float3(0, 1, 0));
    surface.View = SafeNormal(CameraLightCount.xyz - surface.Position, surface.Normal);
    surface.Diffuse = diffuse.rgb; surface.Specular = specular.rgb;
    surface.Roughness = normal.w; surface.Metallic = saturate(ambient.w);
    surface.Shininess = specular.w; surface.Bands = ShadingParameters.z; surface.Model = TextureFlags.w;
    float3 radiance = ambient.rgb + emission;
    bool indexed; uint2 span = SurfaceSelection(input.Position, indexed);
    float contact = ReadContact(input.Position, sampleIndex);
    [loop] for (uint entry = 0; entry < span.y; ++entry)
    {
        uint index = indexed ? SurfaceLightIndices[span.x + entry] : entry;
        radiance += SceneLight(surface, index, RenderParameters.x > 0.5, int(LightLayout.y), contact);
    }
    return float4(clamp(radiance, 0, HalfRadianceLimit), diffuse.a);
}
#endif
#else
// Independently authored DX11 Tiled Deferred light assignment.
// Harada, McKee and Yang, "Forward+: Bringing Deferred Lighting to the Next
// Level" (2012): https://takahiroharada.github.io/forward%2B/
// Original camera-depth samples -> opaque tile depth interval -> light lists.
// Transparency uses a second list spanning the full camera depth interval.
// A light bound encloses its exact support; an unbounded light always survives.
// Material consumers must fall back to every light if the total count exceeds
// the stored capacity. The producer never represents overflow as a shorter list.
#ifndef EVANS_TILED_DEFERRED_MULTISAMPLE
#define EVANS_TILED_DEFERRED_MULTISAMPLE 0
#endif

static const uint ThreadsPerAxis = 8u;
static const uint ThreadsPerTile = ThreadsPerAxis * ThreadsPerAxis;
static const uint BitsPerWord = 32u;
static const uint MaximumLights = 4096u;
static const uint MembershipWords = MaximumLights / BitsPerWord;
static const uint SidePlaneCount = 4u;
static const uint OpaqueNearPlane = SidePlaneCount;
static const uint TransparentNearPlane = OpaqueNearPlane + 2u;
static const uint StoredPlanes = TransparentNearPlane + 2u;
static const uint ListsPerTile = 2u;
static const float FloatPrecision = 1.1920928955078125e-7f;
static const float PlaneRoundoffOperations = 16.0f;
static const float PlaneRoundoff = PlaneRoundoffOperations * FloatPrecision;
static const float SmallestNormalFloat = 1.1754943508222875e-38f;
// Covers flushed subnormal intermediate terms after plane/receiver scaling.
static const float UnderflowOperations = 64.0f;
static const float UnderflowAllowance = UnderflowOperations * SmallestNormalFloat;

cbuffer AssignmentConstants : register(b0)
{
    row_major float4x4 ViewProjection;
    uint4 ExtentAndLights; // width, height, sample count, light count
    uint4 TileLayout; // tile size, list capacity, tile counts X/Y
};
struct LightBounds { float4 CenterRadius; uint4 Flags; };
#if EVANS_TILED_DEFERRED_MULTISAMPLE
Texture2DMS<float> SceneDepth : register(t0);
#else
Texture2D<float> SceneDepth : register(t0);
#endif
StructuredBuffer<LightBounds> Bounds : register(t1);
RWStructuredBuffer<uint4> TileHeaders : register(u0);
RWStructuredBuffer<uint> LightIndices : register(u1);

groupshared uint MinimumDepthBits, MaximumDepthBits;
groupshared float4 PlaneEquation[StoredPlanes];
groupshared float4 PlaneError[StoredPlanes];
groupshared uint OpaqueMembership[MembershipWords];
groupshared uint TransparentMembership[MembershipWords];

float ReadDepth(uint2 pixel, uint sampleIndex)
{
#if EVANS_TILED_DEFERRED_MULTISAMPLE
    return SceneDepth.Load(int2(pixel), sampleIndex);
#else
    return SceneDepth.Load(int3(pixel, 0));
#endif
}
float MaximumComponent(float4 value)
{ return max(max(value.x, value.y), max(value.z, value.w)); }

void StorePlane(uint index, uint axis, float boundary, float inward)
{
    // Plane = inward * (clip-axis row - boundary * clip-W row). Scale before
    // subtraction, so opposite large coefficients cannot overflow. Keeping both
    // original row magnitudes in the error bound protects near cancellation.
    float scale = max(MaximumComponent(abs(ViewProjection[axis])), MaximumComponent(abs(ViewProjection[3])));
    scale = max(scale, SmallestNormalFloat);
    precise float4 axisRow = ViewProjection[axis] / scale;
    precise float4 wRow = ViewProjection[3] / scale;
    precise float4 boundaryRow = boundary * wRow;
    PlaneEquation[index] = inward * (axisRow - boundaryRow);
    // NDC boundary conversion has absolute error near zero, hence abs(wRow).
    PlaneError[index] = PlaneRoundoff * (abs(axisRow) + abs(boundaryRow) + abs(wRow));
}
bool InsidePlane(uint index, float4 center, float radius)
{
    float4 plane = PlaneEquation[index], error = PlaneError[index];
    precise float4 terms = plane * center;
    precise float distance = (terms.x + terms.y) + (terms.z + terms.w);
    precise float support = radius * length(plane.xyz);
    precise float uncertainty = dot(error, abs(center)) + radius * length(error.xyz);
    uncertainty += PlaneRoundoff * (dot(abs(terms), float4(1, 1, 1, 1)) + support) + UnderflowAllowance;
    return distance + support >= -uncertainty;
}
void AssignLight(uint lightIndex)
{
    LightBounds light = Bounds[lightIndex];
    bool opaque = true, transparent = true;
    if (light.Flags.x == 0u)
    {
        // Normalize the homogeneous center and sphere radius together. This
        // bounds dot products even for lights far outside the camera volume.
        float scale = max(1.0f, MaximumComponent(abs(light.CenterRadius)));
        float4 center = float4(light.CenterRadius.xyz, 1.0f) / scale;
        float radius = light.CenterRadius.w / scale;
        [unroll] for (uint plane = 0; plane < SidePlaneCount; ++plane)
            if (!InsidePlane(plane, center, radius)) return;
        opaque = InsidePlane(OpaqueNearPlane, center, radius) && InsidePlane(OpaqueNearPlane + 1u, center, radius);
        transparent = InsidePlane(TransparentNearPlane, center, radius) && InsidePlane(TransparentNearPlane + 1u, center, radius);
    }
    uint word = lightIndex / BitsPerWord, bit = 1u << (lightIndex % BitsPerWord);
    if (opaque) InterlockedOr(OpaqueMembership[word], bit);
    if (transparent) InterlockedOr(TransparentMembership[word], bit);
}
void EmitLists(uint tileIndex)
{
    uint base = tileIndex * ListsPerTile * TileLayout.y;
    uint opaqueCount = 0u, transparentCount = 0u;
    uint words = (ExtentAndLights.w + BitsPerWord - 1u) / BitsPerWord;
    // Atomic membership, followed by an ordered scan, preserves scene-light
    // accumulation order without depending on wave size or append scheduling.
    for (uint word = 0u; word < words; ++word)
    {
        uint bits = OpaqueMembership[word];
        while (bits != 0u)
        {
            uint light = word * BitsPerWord + firstbitlow(bits);
            if (opaqueCount < TileLayout.y) LightIndices[base + opaqueCount] = light;
            ++opaqueCount; bits &= bits - 1u;
        }
        bits = TransparentMembership[word];
        while (bits != 0u)
        {
            uint light = word * BitsPerWord + firstbitlow(bits);
            if (transparentCount < TileLayout.y) LightIndices[base + TileLayout.y + transparentCount] = light;
            ++transparentCount; bits &= bits - 1u;
        }
    }
    TileHeaders[tileIndex] = uint4(base, opaqueCount, base + TileLayout.y, transparentCount);
}

[numthreads(ThreadsPerAxis, ThreadsPerAxis, 1)]
void CSAssign(uint3 tile : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (lane == 0u) { MinimumDepthBits = asuint(1.0f); MaximumDepthBits = asuint(0.0f); }
    for (uint word = lane; word < MembershipWords; word += ThreadsPerTile)
    { OpaqueMembership[word] = 0u; TransparentMembership[word] = 0u; }
    GroupMemoryBarrierWithGroupSync();
    uint tileSize = TileLayout.x;
    for (uint offset = lane; offset < tileSize * tileSize; offset += ThreadsPerTile)
    {
        uint2 pixel = tile.xy * tileSize + uint2(offset % tileSize, offset / tileSize);
        if (any(pixel >= ExtentAndLights.xy)) continue;
        for (uint sampleIndex = 0u; sampleIndex < ExtentAndLights.z; ++sampleIndex)
        {
            float value = ReadDepth(pixel, sampleIndex);
            bool valid = isfinite(value) && value >= 0.0f && value <= 1.0f;
            // Positive IEEE depth is ordered as unsigned bits. Invalid depth
            // widens the interval conservatively; clear depth remains included.
            // Canonicalize signed zero explicitly before unsigned comparison.
            float ordered = value > 0.0f ? value : 0.0f;
            InterlockedMin(MinimumDepthBits, asuint(valid ? ordered : 0.0f));
            InterlockedMax(MaximumDepthBits, asuint(valid ? ordered : 1.0f));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float2 first = float2(tile.xy * tileSize);
    float2 last = float2(min((tile.xy + 1u) * tileSize, ExtentAndLights.xy));
    float2 leftTop = first * 2.0f / float2(ExtentAndLights.xy) - 1.0f;
    float2 rightBottom = last * 2.0f / float2(ExtentAndLights.xy) - 1.0f;
    if (lane == 0u) StorePlane(0u, 0u, leftTop.x, 1.0f);
    if (lane == 1u) StorePlane(1u, 0u, rightBottom.x, -1.0f);
    if (lane == 2u) StorePlane(2u, 1u, -leftTop.y, -1.0f);
    if (lane == 3u) StorePlane(3u, 1u, -rightBottom.y, 1.0f);
    if (lane == OpaqueNearPlane) StorePlane(lane, 2u, asfloat(MinimumDepthBits), 1.0f);
    if (lane == OpaqueNearPlane + 1u) StorePlane(lane, 2u, asfloat(MaximumDepthBits), -1.0f);
    if (lane == TransparentNearPlane) StorePlane(lane, 2u, 0.0f, 1.0f);
    if (lane == TransparentNearPlane + 1u) StorePlane(lane, 2u, 1.0f, -1.0f);
    GroupMemoryBarrierWithGroupSync();
    for (uint light = lane; light < ExtentAndLights.w; light += ThreadsPerTile) AssignLight(light);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) EmitLists(tile.y * TileLayout.z + tile.x);
}

#endif
