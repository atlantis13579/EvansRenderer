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
// DX12 world-space shadow queries. Woop, Benthin and Wald, JCGT 2013.
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
struct ShadowRayNode { float4 Lower,Upper; uint4 Link; };
struct ShadowRayTriangle { float4 P0,P1,P2,UV01,UV2; uint4 Identity; };
struct ShadowRayMaterial { uint4 Diffuse,Opacity; float4 Alpha; };
static const float ShadowRayHalfPixel=0.5;
static const float ShadowRayBinary32Epsilon=1.1920928955078125e-7;
static const float ShadowRayEdgeRoundoff=3*ShadowRayBinary32Epsilon/(1-3*ShadowRayBinary32Epsilon);
ShadowRayNode ShadowRayReadNode(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).x+index*ShadowRayNodeSize;
    ShadowRayNode result;
    result.Lower=asfloat(ShadowRayLoad(base));
    result.Upper=asfloat(ShadowRayLoad(base+1u));
    result.Link=ShadowRayLoad(base+2u);
    return result;
}
ShadowRayTriangle ShadowRayReadTriangle(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).y+index*ShadowRayTriangleSize;
    ShadowRayTriangle result;
    result.P0=asfloat(ShadowRayLoad(base));
    result.P1=asfloat(ShadowRayLoad(base+1u));
    result.P2=asfloat(ShadowRayLoad(base+2u));
    result.UV01=asfloat(ShadowRayLoad(base+3u));
    result.UV2=asfloat(ShadowRayLoad(base+4u));
    result.Identity=ShadowRayLoad(base+5u);
    return result;
}
ShadowRayMaterial ShadowRayReadMaterial(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).z+index*ShadowRayMaterialSize;
    ShadowRayMaterial result;
    result.Diffuse=ShadowRayLoad(base);
    result.Opacity=ShadowRayLoad(base+1u);
    result.Alpha=asfloat(ShadowRayLoad(base+2u));
    return result;
}
float ShadowRayScalar(uint index)
{ uint4 value=ShadowRayLoad(ShadowRayLoad(ShadowRayOffsetsHeader).w+index/ShadowRayScalarLanes); return asfloat(value)[index%ShadowRayScalarLanes]; }
float ShadowRayReadCoverage(uint4 map,float2 uv)
{
    if (!map.w) return 1;
    float2 p=frac(float2(uv.x,-uv.y))*float2(map.yz)-ShadowRayHalfPixel;
    uint x=p.x<0 ? map.y-1 : uint(floor(p.x)),y=p.y<0 ? map.z-1 : uint(floor(p.y));
    uint nextX=x+1==map.y ? 0 : x+1,nextY=y+1==map.z ? 0 : y+1;
    float upper=lerp(ShadowRayScalar(map.x+y*map.y+x),ShadowRayScalar(map.x+y*map.y+nextX),frac(p.x));
    float lower=lerp(ShadowRayScalar(map.x+nextY*map.y+x),ShadowRayScalar(map.x+nextY*map.y+nextX),frac(p.x));
    return lerp(upper,lower,frac(p.y));
}
bool ShadowRayInBounds(ShadowRayNode node,float3 origin,float3 ray,float limit)
{
    float nearDistance=0,farDistance=limit;
    [unroll] for (uint dimension=0;dimension<3;++dimension)
    {
        if (ray[dimension]==0)
        {
            if (origin[dimension]<node.Lower[dimension] || origin[dimension]>node.Upper[dimension]) return false;
            continue;
        }
        float first=(node.Lower[dimension]-origin[dimension])/ray[dimension];
        float last=(node.Upper[dimension]-origin[dimension])/ray[dimension];
        nearDistance=max(nearDistance,min(first,last)); farDistance=min(farDistance,max(first,last));
        if (nearDistance>farDistance) return false;
    }
    return true;
}
float ShadowRayEdgeFunction(float2 first,float2 second)
{
    precise float positive=first.x*second.y,negative=first.y*second.x;
    precise float signedArea=positive-negative;
    return abs(signedArea)<=ShadowRayEdgeRoundoff*(abs(positive)+abs(negative)) ? 0 : signedArea;
}
bool ShadowRayHitTriangle(ShadowRayTriangle primitive,float3 origin,float3 ray,float limit,out float3 barycentric)
{
    barycentric=0;
    uint major=abs(ray.x)>abs(ray.y) ? 0 : 1; if (abs(ray.z)>abs(ray[major])) major=2;
    uint horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    float3 first=primitive.P0.xyz-origin,second=primitive.P1.xyz-origin,third=primitive.P2.xyz-origin;
    float2 slope=float2(ray[horizontal],ray[vertical])/ray[major];
    precise float2 a=float2(first[horizontal],first[vertical])-slope*first[major];
    precise float2 b=float2(second[horizontal],second[vertical])-slope*second[major];
    precise float2 c=float2(third[horizontal],third[vertical])-slope*third[major];
    float3 area=float3(ShadowRayEdgeFunction(b,c),ShadowRayEdgeFunction(c,a),ShadowRayEdgeFunction(a,b));
    if (any(area<0) && any(area>0)) return false;
    float determinant=area.x+area.y+area.z; if (determinant==0) return false;
    float distance=dot(area,float3(first[major],second[major],third[major]))/(determinant*ray[major]);
    if (!(distance>0 && distance<=limit)) return false;
    barycentric=area/determinant; return true;
}
bool ShadowRayAnyAcceptedHit(float3 origin,float3 ray,float limit)
{
    bool blocked=false;
    uint address=0;
    [loop] while (address<ShadowRayLoad(0u).y && !blocked)
    {
        ShadowRayNode node=ShadowRayReadNode(address);
        if (!ShadowRayInBounds(node,origin,ray,limit)) { address=node.Link.w; continue; }
        if (!node.Link.z) { address=node.Link.x; continue; }
        [loop] for (uint local=0;local<node.Link.z && !blocked;++local)
        {
            ShadowRayTriangle primitive=ShadowRayReadTriangle(node.Link.y+local);
            if ((primitive.Identity.w & ShadowRayCasterFlag)==0u) continue; float3 barycentric;
            if (!ShadowRayHitTriangle(primitive,origin,ray,limit,barycentric)) continue;
            ShadowRayMaterial material=ShadowRayReadMaterial(primitive.Identity.x); if (material.Alpha.y<=0) { blocked=true; break; }
            float2 uv=barycentric.x*primitive.UV01.xy+barycentric.y*primitive.UV01.zw+barycentric.z*primitive.UV2.xy;
            float alpha=dot(barycentric,float3(primitive.P0.w,primitive.P1.w,primitive.P2.w))*material.Alpha.x;
            alpha*=ShadowRayReadCoverage(material.Diffuse,uv)*ShadowRayReadCoverage(material.Opacity,uv);
            if (saturate(alpha)>=material.Alpha.y) { blocked=true; break; }
        }
        address=node.Link.w;
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
    if (ShadowRayAnyAcceptedHit(origin,direction,limit)) visibility=0.0;
    return visibility;
}

#endif

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

// Tiled deferred shading: G-buffer depth -> per-tile lists -> reconstructed lighting.
// Captures and transparent geometry retain independent complete-light paths.
// Lauritzen, "Deferred Rendering for Current and Future Rendering Pipelines" (SIGGRAPH 2010).
// https://www.intel.com/content/www/us/en/developer/articles/technical/deferred-rendering-for-current-and-future-rendering-pipelines.html
// Native DX12 Tiled Deferred material/depth stages. Full scene-order emitter storage
// is independent of the inline compatibility constants. Transparent tiles span
// the entire camera interval; overflow and captures traverse every emitter.
#if defined(EVANS_TILED_DEFERRED_SURFACE)
#ifndef EVANS_TILED_DEFERRED_FILTER
#define EVANS_TILED_DEFERRED_FILTER 0
#endif
#define FILTER_PCSS 2
#define FILTER_VSM 3
static const uint RECEIVER_PHONG = 0, RECEIVER_FLAT = 1, RECEIVER_PBR = 3, RECEIVER_TOON = 4;
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
cbuffer ContactReceiver : register(b7) { uint ContactEnabled; };
Texture2DArray<float> ContactImage : register(t15);
float ReadContactVisibility(float4 pixel, uint sampleIndex)
{
    float visibility = 1.0f;
    if (ContactEnabled != 0) visibility = ContactImage.Load(int4(int2(pixel.xy), sampleIndex, 0));
    return visibility;
}
float CombineContactVisibility(float primary, float contact, float strength)
{
    return lerp(1.0f - strength, primary, contact);
}

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
cbuffer SurfaceDraw : register(b0)
{
    row_major float4x4 ObjectTransform, ObjectNormalTransform, CameraProjection, ShadowProjection;
    float4 DrawColor, DiffuseReflectance, AmbientReflectance, SpecularReflectance, EmissionRadiance;
    float4 MaterialParameters, AmbientLighting, EyeLightCount, DrawOptions;
    uint4 MaterialImages;
    Emitter InlineLights[LightCountLimit];
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
cbuffer ForwardLightRoot : register(b8)
{
    uint4 SceneLightLayout; // complete count, complete shadow index, list mode, capacity
    uint4 ScreenTileLayout; // tile side, column count, viewport extent
};
StructuredBuffer<Emitter> SceneLights : register(t16);
StructuredBuffer<uint4> LightingTiles : register(t17);
StructuredBuffer<uint> LightingIndices : register(t18);
cbuffer CascadeUpload : register(b2)
{
    row_major float4x4 CascadeProjection[CascadeCountLimit];
    float4 CascadeEnds, CascadeOverlaps, EyeDepthPlane, CascadeSettings, UnusedFixedFilter;
};
#if EVANS_TILED_DEFERRED_FILTER == FILTER_VSM
cbuffer VarianceUpload : register(b6)
{
    float4 LinearDepthEncoding[ProjectionCount]; // raw clip Z/W coefficients, offset, reserved
    float4 MomentSettings; // variance floor, bleeding suppression, raster variance cap, Gaussian sigma
    uint4 MomentOptions; // valid projection bits, filter radius, reserved, filter axis
};

Texture2D<float2> SingleMoments : register(t0);
Texture2D<float2> CascadeMoments0 : register(t10);
Texture2D<float2> CascadeMoments1 : register(t11);
Texture2D<float2> CascadeMoments2 : register(t12);
Texture2D<float2> CascadeMoments3 : register(t13);
#else
Texture2D<float> SingleDepth : register(t0);
Texture2D<float> CascadeDepth0 : register(t10);
Texture2D<float> CascadeDepth1 : register(t11);
Texture2D<float> CascadeDepth2 : register(t12);
Texture2D<float> CascadeDepth3 : register(t13);
#if EVANS_TILED_DEFERRED_FILTER == FILTER_PCSS
cbuffer SoftShadowUpload : register(b6)
{
    float4 MapProjection[ProjectionCount]; // near, far, width, height; near zero denotes orthographic
    float4 EmitterFootprint; // finite radius, tan(directional half-angle), search/filter texel caps
    uint4 SamplingBudget; // blocker samples, comparison samples, valid projection bits, reserved
};

#endif
#endif
float3 Unit(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > VectorLengthSquaredMinimum ? value * rsqrt(squared) : fallback;
}
#if EVANS_TILED_DEFERRED_FILTER == FILTER_VSM
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

#else
#if EVANS_TILED_DEFERRED_FILTER == FILTER_PCSS
float ReadShadowDepth(Texture2D<float> depth, int2 location, int2 extent)
{
    float result = 1;
    if (all(location >= 0) && all(location < extent)) result = depth.Load(int3(location, 0));
    return result;
}
float CompareBilinear(Texture2D<float> depth, float2 uv, int2 extent, float receiver)
{
    float2 location = uv * extent - PixelCenterOffset;
    int2 cell = int2(floor(location));
    float2 fraction = frac(location);
    float upperLeft = receiver <= ReadShadowDepth(depth, cell, extent);
    float upperRight = receiver <= ReadShadowDepth(depth, cell + int2(1, 0), extent);
    float lowerLeft = receiver <= ReadShadowDepth(depth, cell + int2(0, 1), extent);
    float lowerRight = receiver <= ReadShadowDepth(depth, cell + 1, extent);
    return lerp(lerp(upperLeft, upperRight, fraction.x), lerp(lowerLeft, lowerRight, fraction.x), fraction.y);
}
float AxisDistance(float depth, float4 projection)
{
    return projection.x == 0 ? depth * projection.y :
        projection.x * projection.y / lerp(projection.y, projection.x, depth);
}
float2 UnitDisk(uint index, uint count)
{
    const float GoldenAngle = Pi * (3 - sqrt(5.0));
    float radius = sqrt((index + PixelCenterOffset) / count);
    float sine, cosine;
    sincos(GoldenAngle * index, sine, cosine);
    return radius * float2(cosine, sine);
}
float2 ProjectRadius(float planeRadius, float2 planeSize, int2 extent, float texelBudget)
{
    float2 footprint = planeRadius / planeSize;
    float longestAxis = max(footprint.x * extent.x, footprint.y * extent.y);
    float cap = min(texelBudget, min(extent.x, extent.y) * MapRadiusFraction);
    if (longestAxis > cap) footprint *= cap / longestAxis;
    return footprint;
}
float SoftVisibility(Texture2D<float> depth, uint mapIndex, float4 clip, float bias)
{
    float result = 1;
    if (clip.w > ClipWMinimum && (SamplingBudget.z & (1u << mapIndex)) != 0)
    {
        float3 projected = clip.xyz / clip.w;
        float2 uv = float2(projected.x + 1, 1 - projected.y) / 2;
        if (all(uv >= 0) && all(uv <= 1) && projected.z >= 0 && projected.z <= 1)
        {
            uint width, height;
            depth.GetDimensions(width, height);
            int2 extent = int2(width, height);
            float4 projection = MapProjection[mapIndex];
            bool finiteLight = projection.x > 0;
            float emitterSize = finiteLight ? EmitterFootprint.x : EmitterFootprint.y;
            float comparisonDepth = projected.z - bias;
            if (emitterSize == 0)
                result = comparisonDepth <= ReadShadowDepth(depth, int2(floor(uv * extent)), extent);
            else
            {
                float receiverDistance = AxisDistance(projected.z, projection);
                float searchSize = finiteLight ? emitterSize * (receiverDistance - projection.x) / receiverDistance :
                    emitterSize * receiverDistance;
                float2 search = ProjectRadius(searchSize, projection.zw, extent, EmitterFootprint.z);
                float linearBlockerSum = 0;
                uint blockerCount = 0;
                [loop] for (uint blockerIndex = 0; blockerIndex < SamplingBudget.x; ++blockerIndex)
                {
                    int2 location = int2(floor((uv + search * UnitDisk(blockerIndex, SamplingBudget.x)) * extent));
                    float candidate = ReadShadowDepth(depth, location, extent);
                    if (candidate < comparisonDepth)
                    {
                        linearBlockerSum += AxisDistance(candidate, projection);
                        ++blockerCount;
                    }
                }
                if (blockerCount != 0)
                {
                    float blockerDistance = linearBlockerSum / blockerCount;
                    float separation = max(receiverDistance - blockerDistance, 0);
                    float penumbraSize = emitterSize * separation;
                    if (finiteLight) penumbraSize *= projection.x / (blockerDistance * receiverDistance);
                    float2 penumbra = ProjectRadius(penumbraSize, projection.zw, extent, EmitterFootprint.w);
                    float comparisonSum = 0;
                    [loop] for (uint tapIndex = 0; tapIndex < SamplingBudget.y; ++tapIndex)
                        comparisonSum += CompareBilinear(depth, uv + penumbra * UnitDisk(tapIndex, SamplingBudget.y), extent, comparisonDepth);
                    result = comparisonSum / SamplingBudget.y;
                }
            }
        }
    }
    return result;
}
#else
float FixedVisibility(Texture2D<float> image, float4 clip, float bias)
{
    if (clip.w <= ClipWMinimum) return 1;
    float3 projected = clip.xyz / clip.w;
    float2 uv = float2(projected.x + 1, 1 - projected.y) / 2;
    if (any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1) return 1;
    uint width, height; image.GetDimensions(width, height);
    int2 last = int2(width, height) - 1;
    float receiver = projected.z - bias;
    if (UnusedFixedFilter.x == 0)
        return receiver <= image.Load(int3(clamp(int2(uv * float2(width, height)), 0, last), 0));
    float2 texel = uv * float2(width, height) - PixelCenterOffset;
    int2 cell = int2(floor(texel)); float2 fraction = frac(texel);
    float sum = 0;
    [loop] for (int row = -int(UnusedFixedFilter.x); row <= int(UnusedFixedFilter.x); ++row)
        [loop] for (int column = -int(UnusedFixedFilter.x); column <= int(UnusedFixedFilter.x); ++column)
        {
            int2 p = cell + int2(column, row);
            float topLeft = receiver <= image.Load(int3(clamp(p, 0, last), 0));
            float topRight = receiver <= image.Load(int3(clamp(p + int2(1, 0), 0, last), 0));
            float bottomLeft = receiver <= image.Load(int3(clamp(p + int2(0, 1), 0, last), 0));
            float bottomRight = receiver <= image.Load(int3(clamp(p + 1, 0, last), 0));
            sum += lerp(lerp(topLeft, topRight, fraction.x), lerp(bottomLeft, bottomRight, fraction.x), fraction.y);
        }
    return sum / UnusedFixedFilter.y;
}
float SoftVisibility(Texture2D<float> depth, uint mapIndex, float4 clip, float bias)
{ return FixedVisibility(depth, clip, bias); }
#endif
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

#endif
#if EVANS_TILED_DEFERRED_FILTER != 3
// Virtual shadows share the cascade receiver slots. End distances describe the
// logical hierarchy; fade distances describe the physical atlas and directory.
// Point reads plus per-corner indirection keep unrelated physical tiles apart.
static const uint VirtualGutter = 1;
static const float VirtualPixelCenter = 0.5f;
float2 VirtualStoredDepth(float2 coordinate, uint minimumLevel)
{
    uint4 hierarchy = uint4(CascadeEnds), atlas = uint4(CascadeOverlaps);
    uint rowPages = hierarchy.x / hierarchy.y, levelOffset = 0;
    [loop] for (uint level = 0; level < hierarchy.z; ++level)
    {
        if (level >= minimumLevel)
        {
            uint resolution = rowPages * hierarchy.y;
            uint2 texel = min(uint2(coordinate * resolution), resolution - 1);
            uint2 page = texel / hierarchy.y;
            uint entryIndex = levelOffset + page.x + page.y * rowPages;
            int2 tablePixel = int2(entryIndex % hierarchy.w, entryIndex / hierarchy.w);
            uint encodedSlot = uint(round(CascadeDepth1.Load(int3(tablePixel, 0)).x * atlas.w));
            if (encodedSlot > 0)
            {
                uint slot = encodedSlot - 1;
                uint2 origin = uint2(slot % atlas.z, slot / atlas.z) * (hierarchy.y + 2 * VirtualGutter);
                uint2 location = origin + VirtualGutter + texel % hierarchy.y;
                return float2(CascadeDepth0.Load(int3(location, 0)).x, level);
            }
        }
        levelOffset += rowPages * rowPages; rowPages /= 2;
    }
    return float2(1, hierarchy.z - 1);
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
    float resolution = float(uint(CascadeEnds.x) >> actualMip);
    float2 pixel = clamp(floor(sampleUv * resolution), float2(0.0, 0.0), float2(resolution - 1.0, resolution - 1.0));
    float2 center = (pixel + float2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualComparison(int2 pixel, uint mip, float2 receiverUv, float receiver, float2 gradient)
{
    uint resolution = uint(CascadeEnds.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0, 0), int2(int(resolution) - 1, int(resolution) - 1))) + float2(0.5, 0.5)) / float(resolution);
    float2 stored = VirtualStoredDepth(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
float VirtualVisibility(float3 position, float bias, float3 receiverData)
{
    float4 light = mul(CascadeProjection[0], float4(position, 1.0));
    float3 projected = light.xyz / max(light.w, EyeDepthPlane.w);
    float2 uv = float2(projected.x, -projected.y) * 0.5 + float2(0.5, 0.5);
    float2 gradient = receiverData.xy;
    uint mip = uint(receiverData.z);
    if (light.w <= EyeDepthPlane.w || any(uv < 0.0) || any(uv > 1.0) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float2 stored = VirtualStoredDepth(uv, mip);
    float receiver = projected.z - bias;
    if (UnusedFixedFilter.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(CascadeEnds.x) >> mip) - float2(0.5, 0.5);
    int2 first = int2(floor(texel)); float2 weight = frac(texel); float visible = 0.0;
    for (int y = -int(UnusedFixedFilter.x); y <= int(UnusedFixedFilter.x); ++y)
        for (int x = -int(UnusedFixedFilter.x); x <= int(UnusedFixedFilter.x); ++x)
        {
            int2 p = first + int2(x, y);
            float a = VirtualComparison(p, mip, uv, receiver, gradient);
            float b = VirtualComparison(p + int2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualComparison(p + int2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualComparison(p + int2(1, 1), mip, uv, receiver, gradient);
            visible += lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        }
    return visible / UnusedFixedFilter.y;
}



#endif

// Resolve the receiver plane before entering a varying light-list loop.
float3 VirtualReceiverData(float3 position)
{
#if EVANS_TILED_DEFERRED_FILTER != 3
    if (UnusedFixedFilter.z == 0) return float3(0.0, 0.0, 0.0);
    float4 projected = mul(CascadeProjection[0], float4(position, 1));
    float2 uv = projected.xy / max(projected.w, EyeDepthPlane.w) * float2(0.5, -0.5);
    float footprint = max(length(ddx(uv)) * EyeDepthPlane.y, length(ddy(uv)) * EyeDepthPlane.z) * CascadeEnds.x;
    return float3(VirtualReceiverGradient(uv, projected.z / max(projected.w, EyeDepthPlane.w)), clamp(floor(log2(max(1, footprint)) + EyeDepthPlane.x), 0, CascadeEnds.z - 1));
#else
    return float3(0.0, 0.0, 0.0);
#endif
}

float SurfaceVisibility(float3 world, float3 normal, float3 toLight, float3 receiverData)
{
    float result=1.0;
    if (ShadowRayEnabled()) result=1.0-ShadowStrength*(1.0-ShadowRayVisibility(world,normal,toLight));
    else
    {
    float cosine = saturate(dot(Unit(normal, float3(0, 1, 0)), Unit(toLight, float3(0, 1, 0))));
    float tangent = sqrt(max(1 - cosine * cosine, 0)) / max(cosine, BiasCosineFloor);
    float bias = min(BiasConstant + BiasSlope * tangent, BiasMaximum);
    float visibility = 1;
#if EVANS_TILED_DEFERRED_FILTER != 3
    if (UnusedFixedFilter.z != 0) visibility = VirtualVisibility(world, bias, receiverData);
    else
#endif
        visibility = PrimaryVisibility(world, bias, CascadeSettings.z > 0);
    result = 1 - ShadowStrength * (1 - visibility);

    }
    return result;
}
struct Surface
{
    float3 World, Normal, View, Diffuse, Specular;
    float Roughness, Metallic, Shininess, Bands;
    uint Model;
    float3 VirtualReceiver;
};
float3 SampleReflectance(Surface material, float3 toLight, float3 radiance)
{
    const float RoughnessFloor = 0.045, DielectricF0 = 0.04;
    const float DenominatorFloor = 1.0e-6, DistributionFloor = 1.0e-12, FresnelPower = 5;
    const float SpecularThreshold = 0.5, MinimumBands = 2;
    const float MaximumBands = 16;
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
            reflected = radiance * nl * (diffuse + specular);
        }
    }
    else
    {
        float specular = nl > 0 ? pow(nh, max(material.Shininess, 1)) : 0;
        if (material.Model == RECEIVER_TOON)
        {
            float bands = clamp(round(material.Bands), MinimumBands, MaximumBands);
            reflected = radiance * (material.Diffuse * ToonResponse(nl, bands) + material.Specular * ToonHighlight(specular));
        }
        else reflected = radiance * (material.Diffuse * nl + material.Specular * specular);
        if (material.Model <= RECEIVER_FLAT && dot(toLight, toLight) <= VectorLengthSquaredMinimum) reflected = 0;
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
        if (shadowEnabled) visibility = SurfaceVisibility(material.World, material.Normal, centerDirection, material.VirtualReceiver);
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
    precise float4 Pixel : SV_POSITION;
    float3 World : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 UnusedShadowPosition : TEXCOORD2;
    float2 UV : TEXCOORD3;

};
float3 ShadingNormal(SurfaceFragment fragment, bool front)
{
    const float UvDeterminantFloor = 1.0e-10, ParallelThreshold = 0.999;
    float3 normal = Unit(fragment.Normal, float3(0, 1, 0));
    if (MaterialImages.w == RECEIVER_FLAT)
    {
    normal = Unit(cross(ddx(fragment.World), ddy(fragment.World)), normal);
    if (dot(normal, fragment.Normal) < 0) normal = -normal;
    }
    if (!front) normal = -normal;
    if (MaterialImages.w != RECEIVER_FLAT)
    {
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
    }
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
struct MeshInput { float3 Position : POSITION; float3 Normal : NORMAL; float4 Color : COLOR0; float2 UV : TEXCOORD0; };
SurfaceFragment VSMain(MeshInput input)
{
    SurfaceFragment fragment;
    float4 world = mul(ObjectTransform, float4(input.Position, 1));
    fragment.Pixel = mul(CameraProjection, world);
    fragment.World = world.xyz / max(abs(world.w), ClipWMinimum);
    fragment.Normal = Unit(mul((float3x3)ObjectNormalTransform, input.Normal), float3(0, 1, 0));
    fragment.Color = float4(DecodeColor(saturate(input.Color.rgb)), input.Color.a);
    fragment.UnusedShadowPosition = mul(ShadowProjection, world);
    fragment.UV = float2(input.UV.x, 1 - input.UV.y);
    return fragment;
}
void PSDepth(SurfaceFragment fragment)
{
    float alpha = fragment.Color.a * MaterialParameters.y;
    if (MaterialImages.x != 0) alpha *= DiffuseImage.Sample(MaterialSampler, fragment.UV).a;
    if (MaterialImages.z != 0) alpha *= OpacityImage.Sample(MaterialSampler, fragment.UV).r;
    if (MaterialParameters.z > 0) clip(saturate(alpha) - MaterialParameters.z);
}
// mode 0 = complete scene, 1 = opaque tile, 2 = transparent full-depth tile.
uint2 LightSpan(float2 pixelPosition, out bool indirect)
{
    uint2 span = uint2(0, SceneLightLayout.x); indirect = false;
    if (SceneLightLayout.z != 0)
    {
        uint2 pixel = min(uint2(pixelPosition), ScreenTileLayout.zw - 1);
        uint2 tile = pixel / ScreenTileLayout.x;
        uint4 record = LightingTiles[tile.x + tile.y * ScreenTileLayout.y];
        uint2 tileSpan = SceneLightLayout.z == 1 ? record.xy : record.zw;
        if (tileSpan.y <= SceneLightLayout.w) { indirect = true; span = tileSpan; }
    }
    return span;
}
float4 ShadeFragment(SurfaceFragment fragment, bool front, uint contactSample)
{
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
        Surface material = (Surface)0;
        material.World = fragment.World; material.VirtualReceiver = VirtualReceiverData(material.World); material.Normal = ShadingNormal(fragment, front);
        material.View = Unit(EyeLightCount.xyz - material.World, material.Normal);
        material.Diffuse = base * max(DiffuseReflectance.rgb, 0);
        material.Specular = max(SpecularReflectance.rgb, 0);
        material.Roughness = roughnessMetallic.x; material.Metallic = saturate(roughnessMetallic.y);
        material.Shininess = MaterialParameters.x; material.Bands = SurfaceParameters.z;
        material.Model = MaterialImages.w;
        color = ToonAmbient(base * max(DiffuseReflectance.rgb, 0.0), base * max(AmbientReflectance.rgb, 0) * max(AmbientLighting.rgb, 0) * max(AmbientLighting.w, 0), float(MaterialImages.w));
        if (material.Model == RECEIVER_PBR) color *= 1 - material.Metallic;
        color += max(EmissionRadiance.rgb, 0);
        bool indirect = false; uint2 span = LightSpan(fragment.Pixel.xy, indirect);
        float contact = ReadContactVisibility(fragment.Pixel, contactSample);
        [loop] for (uint offset = 0; offset < span.y; ++offset)
        {
            uint index = indirect ? LightingIndices[span.x + offset] : offset;
            color += SurfaceLight(material, index, DrawOptions.x > 0.5, int(SceneLightLayout.y), contact);
        }

        color += material.Diffuse * (material.Model==RECEIVER_PBR ? (1-LumenDielectricReflectance)*(1-material.Metallic) : 1) * LumenTranslucentIrradiance(material.World,material.Normal);
        color = max(FogSurface(color, fragment.Pixel), 0);
        if (SurfaceParameters.w > 0.5) color = min(color, HalfFloatMaximum);
        else color = saturate(color / max(1, max(color.r, max(color.g, color.b))));
    }
    return float4(color, opacity);
}
float4 PSMain(SurfaceFragment fragment, bool front : SV_IsFrontFace) : SV_TARGET
{ return ShadeFragment(fragment, front, 0); }
float4 PSContact(SurfaceFragment fragment, bool front : SV_IsFrontFace, uint sampleIndex : SV_SampleIndex) : SV_TARGET
{ return ShadeFragment(fragment, front, sampleIndex); }
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
#if defined(EVANS_TILED_DEFERRED_LIGHTING)
cbuffer LightingCamera : register(b1)
{
    row_major float4x4 InverseCamera;
    float4 ClearRadiance;
};
#ifdef EVANS_MULTISAMPLE
Texture2DMS<float4> MaterialDiffuse : register(t4);
Texture2DMS<float4> MaterialNormal : register(t5);
Texture2DMS<float4> MaterialSpecular : register(t6);
Texture2DMS<float4> MaterialAmbient : register(t7);
Texture2DMS<float4> MaterialEmission : register(t8);
Texture2DMS<float> MaterialDepth : register(t9);
#else
Texture2D<float4> MaterialDiffuse : register(t4);
Texture2D<float4> MaterialNormal : register(t5);
Texture2D<float4> MaterialSpecular : register(t6);
Texture2D<float4> MaterialAmbient : register(t7);
Texture2D<float4> MaterialEmission : register(t8);
Texture2D<float> MaterialDepth : register(t9);
#endif
// Reconstruct one-sided depth slopes on the same geometric surface.
// Fragment-quad derivatives can cross a G-buffer silhouette or intersection.
// Camera depth is affine within a rasterized triangle; lower second difference
// chooses the side that best preserves that plane without using shading normals.
float DeferredReceiverDepth(int2 pixel, uint sampleIndex, uint2 size)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return 1.0;
#if defined(EVANS_MULTISAMPLE)
    return MaterialDepth.Load(pixel, sampleIndex);
#else
    return MaterialDepth.Load(int3(pixel, 0));
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
    float4 world = mul(InverseCamera, float4(coordinate.x * 2.0 - 1.0, 1.0 - coordinate.y * 2.0, depth, 1.0));
    float4 light = mul(CascadeProjection[0], float4(world.xyz / world.w, 1.0));
    float3 projected = light.xyz / max(light.w, EyeDepthPlane.w);
    return float3(projected.xy * float2(0.5, -0.5) + 0.5, projected.z);
}
float3 DeferredVirtualReceiverData(float2 coordinate, float depth, uint sampleIndex)
{
    if (UnusedFixedFilter.z == 0) return 0.0;
    uint width, height;
#if defined(EVANS_MULTISAMPLE)
    uint sampleCount; MaterialDepth.GetDimensions(width, height, sampleCount);
#else
    MaterialDepth.GetDimensions(width, height);
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
    float footprint = max(length(x.xy * EyeDepthPlane.y), length(y.xy * EyeDepthPlane.z)) * CascadeEnds.x;
    float mip = clamp(floor(log2(max(1.0, footprint)) + EyeDepthPlane.x), 0.0, CascadeEnds.z - 1.0);
    return float3(gradient, mip);
}

struct LightingRaster { float4 Pixel : SV_POSITION; float2 Coordinate : TEXCOORD0; };
LightingRaster VSLighting(uint vertex : SV_VertexID)
{
    LightingRaster result;
    result.Coordinate = float2((vertex << 1) & 2, vertex & 2);
    result.Pixel = float4(2 * result.Coordinate.x - 1, 1 - 2 * result.Coordinate.y, 0, 1);
    return result;
}
#ifdef EVANS_MULTISAMPLE
float4 PSLighting(LightingRaster raster, uint sampleIndex : SV_SampleIndex) : SV_TARGET
#else
float4 PSLighting(LightingRaster raster) : SV_TARGET
#endif
{
#ifdef EVANS_MULTISAMPLE
    int2 address = int2(raster.Pixel.xy);
    float4 diffuse = MaterialDiffuse.Load(address, sampleIndex), normal = MaterialNormal.Load(address, sampleIndex);
    float4 specular = MaterialSpecular.Load(address, sampleIndex), ambient = MaterialAmbient.Load(address, sampleIndex);
    float3 emission = MaterialEmission.Load(address, sampleIndex).rgb;
    float depth = MaterialDepth.Load(address, sampleIndex);
    uint width, height, samples; MaterialDepth.GetDimensions(width, height, samples);
    float2 uv = (float2(address) + PixelCenterOffset + MaterialDepth.GetSamplePosition(sampleIndex)) / float2(width, height);
#else
    const uint sampleIndex = 0;
    int3 address = int3(int2(raster.Pixel.xy), 0);
    float4 diffuse = MaterialDiffuse.Load(address), normal = MaterialNormal.Load(address);
    float4 specular = MaterialSpecular.Load(address), ambient = MaterialAmbient.Load(address);
    float3 emission = MaterialEmission.Load(address).rgb;
    float depth = MaterialDepth.Load(address);
    float2 uv = raster.Coordinate;
#endif
    const float ClearedDepth = 1.0;
    if (normal.w < 0) return float4(emission, diffuse.a);
    if (depth >= ClearedDepth) return ClearRadiance;
    float4 projected = mul(InverseCamera, float4(2 * uv.x - 1, 1 - 2 * uv.y, depth, 1));
    Surface material = (Surface)0;
    material.World = projected.xyz / projected.w; material.VirtualReceiver = DeferredVirtualReceiverData(uv, depth, sampleIndex);
    material.Normal = Unit(normal.xyz, float3(0, 1, 0));
    material.View = Unit(EyeLightCount.xyz - material.World, material.Normal);
    material.Diffuse = diffuse.rgb; material.Specular = specular.rgb;
    material.Roughness = normal.w; material.Metallic = saturate(ambient.w);
    material.Shininess = specular.w; material.Bands = SurfaceParameters.z; material.Model = MaterialImages.w;
    bool indirect; uint2 span = LightSpan(raster.Pixel.xy, indirect);
    float3 radiance = ambient.rgb + emission;
    float visibility = ReadContactVisibility(raster.Pixel, sampleIndex);
    [loop] for (uint offset = 0; offset < span.y; ++offset)
    {
        uint light = indirect ? LightingIndices[span.x + offset] : offset;
        radiance += SurfaceLight(material, light, DrawOptions.x > 0.5, int(SceneLightLayout.y), visibility);
    }
    return float4(clamp(radiance, 0, HalfFloatMaximum), diffuse.a);
}
#endif
#else
// Native DX12 Tiled Deferred assignment, authored independently for this API.
// Harada, McKee and Yang, "Forward+: Bringing Deferred Lighting to the Next
// Level" (2012): https://takahiroharada.github.io/forward%2B/
// A tile group reduces original depth samples, tests conservative light support,
// then compacts two membership bitsets using parallel population-count scans.
// Scan background: Harris, Sengupta and Owens, GPU Gems 3, Chapter 39:
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// Opaque lists use observed depth; transparent lists span the camera interval.
// Overflow headers retain the full count: consumers must evaluate every scene
// light instead of treating the stored prefix as a complete list.
#ifndef EVANS_TILED_DEFERRED_MULTISAMPLE
#define EVANS_TILED_DEFERRED_MULTISAMPLE 0
#endif

static const uint GroupSize = 64u;
static const uint WordBits = 32u;
static const uint LightLimit = 4096u;
static const uint WordLimit = LightLimit / WordBits;
static const uint SideCount = 4u;
static const uint OpaqueDepthStart = SideCount;
static const uint TransparentDepthStart = OpaqueDepthStart + 2u;
static const uint PlaneCount = TransparentDepthStart + 2u;
static const uint ListsInTile = 2u;
static const uint OpaqueMember = 1u, TransparentMember = 2u;
static const float UnitRoundoff = 1.1920928955078125e-7f;
static const float ErrorOperations = 16.0f;
static const float ErrorScale = ErrorOperations * UnitRoundoff;
static const float NormalFloatMinimum = 1.1754943508222875e-38f;
static const float FlushOperations = 64.0f;
static const float FlushAllowance = FlushOperations * NormalFloatMinimum;

// Explicit rows avoid dynamic indexing into DX12 root constants. The host's
// 96-byte ForwardPlusConstants layout is identical to these four rows and uint4s.
cbuffer TileCamera : register(b0)
{
    float4 ClipX, ClipY, ClipZ, ClipW;
    uint4 Image; // width, height, depth sample count, number of light bounds
    uint4 Tiles; // pixels per tile, stored capacity, tile counts X/Y
};
struct SupportSphere { float4 CenterRadius; uint4 Flags; };
StructuredBuffer<SupportSphere> LightSupport : register(t1);
#if EVANS_TILED_DEFERRED_MULTISAMPLE
Texture2DMS<float> CameraDepth : register(t0);
#else
Texture2D<float> CameraDepth : register(t0);
#endif
RWStructuredBuffer<uint4> Headers : register(u0);
RWStructuredBuffer<uint> Indices : register(u1);

groupshared uint NearBits, FarBits;
groupshared float4 RegionPlanes[PlaneCount], RegionErrors[PlaneCount];
groupshared uint OpaqueBits[WordLimit], TransparentBits[WordLimit];
groupshared uint OpaquePrefix[WordLimit], TransparentPrefix[WordLimit];

float GreatestAbsolute(float4 value)
{
    float4 magnitude = abs(value);
    return max(max(magnitude.x, magnitude.y), max(magnitude.z, magnitude.w));
}
void MakePlane(uint slot, float4 axis, float coordinate, float facing)
{
    // Homogeneous half-space: facing * (clip-axis - coordinate * clip-W).
    // Common scaling precedes subtraction and prevents finite rows overflowing.
    float divisor = max(NormalFloatMinimum, max(GreatestAbsolute(axis), GreatestAbsolute(ClipW)));
    precise float4 a = axis / divisor, w = ClipW / divisor;
    precise float4 b = coordinate * w;
    RegionPlanes[slot] = facing * (a - b);
    RegionErrors[slot] = ErrorScale * (abs(a) + abs(b) + abs(w));
}
bool SphereInHalfSpace(uint slot, float4 center, float radius)
{
    float4 plane = RegionPlanes[slot];
    precise float4 product = plane * center;
    precise float distance = (product.x + product.y) + (product.z + product.w);
    precise float reach = radius * length(plane.xyz);
    precise float uncertainty = dot(RegionErrors[slot], abs(center)) + radius * length(RegionErrors[slot].xyz);
    uncertainty += ErrorScale * (dot(abs(product), float4(1, 1, 1, 1)) + reach) + FlushAllowance;
    return distance + reach >= -uncertainty;
}
uint Membership(SupportSphere light)
{
    if (light.Flags.x != 0u) return OpaqueMember | TransparentMember;
    float divisor = max(1.0f, GreatestAbsolute(light.CenterRadius));
    float4 center = float4(light.CenterRadius.xyz, 1.0f) / divisor;
    float radius = light.CenterRadius.w / divisor;
    [unroll] for (uint side = 0u; side < SideCount; ++side)
        if (!SphereInHalfSpace(side, center, radius)) return 0u;
    bool opaque = SphereInHalfSpace(OpaqueDepthStart, center, radius) &&
        SphereInHalfSpace(OpaqueDepthStart + 1u, center, radius);
    bool transparent = SphereInHalfSpace(TransparentDepthStart, center, radius) &&
        SphereInHalfSpace(TransparentDepthStart + 1u, center, radius);
    return (opaque ? OpaqueMember : 0u) | (transparent ? TransparentMember : 0u);
}
void ScatterWord(uint word, bool transparent, uint tileBase)
{
    uint members = transparent ? TransparentBits[word] : OpaqueBits[word];
    uint inclusive = transparent ? TransparentPrefix[word] : OpaquePrefix[word];
    uint ordinal = inclusive - countbits(members);
    uint start = tileBase + (transparent ? Tiles.y : 0u);
    while (members != 0u)
    {
        uint light = word * WordBits + firstbitlow(members);
        if (ordinal < Tiles.y) Indices[start + ordinal] = light;
        ++ordinal; members &= members - 1u;
    }
}

[numthreads(GroupSize, 1, 1)]
void CSAssign(uint3 tile : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (lane == 0u) { NearBits = asuint(1.0f); FarBits = asuint(0.0f); }
    for (uint clearWord = lane; clearWord < WordLimit; clearWord += GroupSize)
    { OpaqueBits[clearWord] = TransparentBits[clearWord] = OpaquePrefix[clearWord] = TransparentPrefix[clearWord] = 0u; }
    GroupMemoryBarrierWithGroupSync();
    for (uint localPixel = lane; localPixel < Tiles.x * Tiles.x; localPixel += GroupSize)
    {
        uint2 pixel = tile.xy * Tiles.x + uint2(localPixel % Tiles.x, localPixel / Tiles.x);
        if (pixel.x >= Image.x || pixel.y >= Image.y) continue;
        for (uint sample = 0u; sample < Image.z; ++sample)
        {
#if EVANS_TILED_DEFERRED_MULTISAMPLE
            float value = CameraDepth.Load(int2(pixel), sample);
#else
            float value = CameraDepth.Load(int3(pixel, 0));
#endif
            bool finiteDepth = isfinite(value) && value >= 0.0f && value <= 1.0f;
            // Unsigned bit ordering agrees with nonnegative IEEE depth only
            // after signed zero is canonicalized. Invalid samples widen bounds.
            float canonical = value > 0.0f ? value : 0.0f;
            InterlockedMin(NearBits, asuint(finiteDepth ? canonical : 0.0f));
            InterlockedMax(FarBits, asuint(finiteDepth ? canonical : 1.0f));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float2 first = float2(tile.xy * Tiles.x) * 2.0f / float2(Image.xy) - 1.0f;
    float2 last = float2(min((tile.xy + 1u) * Tiles.x, Image.xy)) * 2.0f / float2(Image.xy) - 1.0f;
    if (lane == 0u) MakePlane(lane, ClipX, first.x, 1.0f);
    if (lane == 1u) MakePlane(lane, ClipX, last.x, -1.0f);
    if (lane == 2u) MakePlane(lane, ClipY, -first.y, -1.0f);
    if (lane == 3u) MakePlane(lane, ClipY, -last.y, 1.0f);
    if (lane == OpaqueDepthStart) MakePlane(lane, ClipZ, asfloat(NearBits), 1.0f);
    if (lane == OpaqueDepthStart + 1u) MakePlane(lane, ClipZ, asfloat(FarBits), -1.0f);
    if (lane == TransparentDepthStart) MakePlane(lane, ClipZ, 0.0f, 1.0f);
    if (lane == TransparentDepthStart + 1u) MakePlane(lane, ClipZ, 1.0f, -1.0f);
    GroupMemoryBarrierWithGroupSync();
    for (uint light = lane; light < Image.w; light += GroupSize)
    {
        uint mask = Membership(LightSupport[light]);
        uint lightWord = light / WordBits, flag = 1u << (light % WordBits);
        if ((mask & OpaqueMember) != 0u) InterlockedOr(OpaqueBits[lightWord], flag);
        if ((mask & TransparentMember) != 0u) InterlockedOr(TransparentBits[lightWord], flag);
    }
    GroupMemoryBarrierWithGroupSync();
    uint usedWords = (Image.w + WordBits - 1u) / WordBits;
    for (uint prefixWord = lane; prefixWord < usedWords; prefixWord += GroupSize)
    { OpaquePrefix[prefixWord] = countbits(OpaqueBits[prefixWord]); TransparentPrefix[prefixWord] = countbits(TransparentBits[prefixWord]); }
    GroupMemoryBarrierWithGroupSync();
    // Inclusive Hillis-Steele scan over at most two words per thread. Every
    // predecessor is read before any thread writes that iteration's new sums.
    // No wave instructions or assumed hardware subgroup width are needed.
    for (uint distance = 1u; distance < usedWords; distance *= 2u)
    {
        uint upper = lane + GroupSize;
        // Predicated expressions can evaluate both operands. Keep every load
        // in bounds, including inactive lanes and the first scan elements.
        uint lowerSource = min(max(lane, distance) - distance, usedWords - 1u);
        uint upperSource = min(max(upper, distance) - distance, usedWords - 1u);
        uint2 opaque = uint2(lane >= distance && lane < usedWords ? OpaquePrefix[lowerSource] : 0u,
            upper >= distance && upper < usedWords ? OpaquePrefix[upperSource] : 0u);
        uint2 transparent = uint2(lane >= distance && lane < usedWords ? TransparentPrefix[lowerSource] : 0u,
            upper >= distance && upper < usedWords ? TransparentPrefix[upperSource] : 0u);
        GroupMemoryBarrierWithGroupSync();
        if (lane < usedWords) { OpaquePrefix[lane] += opaque.x; TransparentPrefix[lane] += transparent.x; }
        if (upper < usedWords) { OpaquePrefix[upper] += opaque.y; TransparentPrefix[upper] += transparent.y; }
        GroupMemoryBarrierWithGroupSync();
    }
    uint tileIndex = tile.y * Tiles.z + tile.x;
    uint base = tileIndex * ListsInTile * Tiles.y;
    for (uint outputWord = lane; outputWord < usedWords; outputWord += GroupSize)
    { ScatterWord(outputWord, false, base); ScatterWord(outputWord, true, base); }
    if (lane == 0u)
    {
        uint lastWord = max(usedWords, 1u) - 1u;
        uint opaqueCount = usedWords > 0u ? OpaquePrefix[lastWord] : 0u;
        uint transparentCount = usedWords > 0u ? TransparentPrefix[lastWord] : 0u;
        Headers[tileIndex] = uint4(base, opaqueCount, base + Tiles.y, transparentCount);
    }
}

#endif
