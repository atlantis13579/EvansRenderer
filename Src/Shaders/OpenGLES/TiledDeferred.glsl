#version 310 es
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

#if defined(EVANS_TILED_DEFERRED_SURFACE) && !defined(EVANS_VERTEX_STAGE)
// OpenGLES world-space shadow queries. Woop, Benthin and Wald, JCGT 2013.
// https://jcgt.org/published/0002/01/05/ . Forward escape links bound traversal;
// independent alpha planes preserve bilinear cutout coverage. The RGBA32UI ABI
// is described in RayTracedShadowScene.h. No screen-space visibility is assumed.
precision highp float;
precision highp int;
uniform highp usampler2D ShadowRayScene;
const uint ShadowRayCasterFlag=1u,ShadowRayImageABI=1u;
const uint ShadowRayCountsHeader=0u,ShadowRayOffsetsHeader=1u,ShadowRayTraceHeader=2u,ShadowRayLightHeader=3u;
const uint ShadowRayNodeSize=3u,ShadowRayTriangleSize=6u,ShadowRayMaterialSize=3u,ShadowRayScalarLanes=4u;
const uint ShadowRayNodeMaximum=1u,ShadowRayNodeLinks=2u;
const uint ShadowRayTriangleUV01=3u,ShadowRayTriangleUV2=4u,ShadowRayTriangleMetadata=5u;
const uint ShadowRayMaterialOpacity=1u,ShadowRayMaterialCoverageWord=2u;
uvec4 ShadowRayLoad(uint address)
{ uint width=uint(max(textureSize(ShadowRayScene,0).x,1)); return texelFetch(ShadowRayScene,ivec2(address%width,address/width),0); }
struct ShadowRayNode { vec4 Lower,Upper; uvec4 Link; };
struct ShadowRayTriangle { vec4 A,B,C,UVAB,UVC; uvec4 Identity; };
struct ShadowRayMaterial { uvec4 Diffuse,Opacity; vec4 Alpha; };
const float ShadowRayPixelCenter=0.5;
const float ShadowRayFloatEpsilon=1.1920928955078125e-7;
const float ShadowRayEdgeUncertainty=3.0*ShadowRayFloatEpsilon/(1.0-3.0*ShadowRayFloatEpsilon);
ShadowRayNode ShadowRayReadNode(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).x+index*ShadowRayNodeSize;
    ShadowRayNode result;
    result.Lower=uintBitsToFloat(ShadowRayLoad(base));
    result.Upper=uintBitsToFloat(ShadowRayLoad(base+1u));
    result.Link=ShadowRayLoad(base+2u);
    return result;
}
ShadowRayTriangle ShadowRayReadTriangle(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).y+index*ShadowRayTriangleSize;
    ShadowRayTriangle result;
    result.A=uintBitsToFloat(ShadowRayLoad(base));
    result.B=uintBitsToFloat(ShadowRayLoad(base+1u));
    result.C=uintBitsToFloat(ShadowRayLoad(base+2u));
    result.UVAB=uintBitsToFloat(ShadowRayLoad(base+3u));
    result.UVC=uintBitsToFloat(ShadowRayLoad(base+4u));
    result.Identity=ShadowRayLoad(base+5u);
    return result;
}
ShadowRayMaterial ShadowRayReadMaterial(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).z+index*ShadowRayMaterialSize;
    ShadowRayMaterial result;
    result.Diffuse=ShadowRayLoad(base);
    result.Opacity=ShadowRayLoad(base+1u);
    result.Alpha=uintBitsToFloat(ShadowRayLoad(base+2u));
    return result;
}
float ShadowRayScalar(uint index)
{ uvec4 value=ShadowRayLoad(ShadowRayLoad(ShadowRayOffsetsHeader).w+index/ShadowRayScalarLanes); return uintBitsToFloat(value)[index%ShadowRayScalarLanes]; }
float ShadowRayMaterialCoverage(uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0;
    vec2 point=fract(vec2(uv.x,-uv.y))*vec2(image.yz)-ShadowRayPixelCenter;
    uint x=point.x<0.0 ? image.y-1u : uint(floor(point.x));
    uint y=point.y<0.0 ? image.z-1u : uint(floor(point.y));
    uint right=x+1u==image.y ? 0u : x+1u,bottom=y+1u==image.z ? 0u : y+1u;
    float upper=mix(ShadowRayScalar(image.x+y*image.y+x),ShadowRayScalar(image.x+y*image.y+right),fract(point.x));
    float lower=mix(ShadowRayScalar(image.x+bottom*image.y+x),ShadowRayScalar(image.x+bottom*image.y+right),fract(point.x));
    return mix(upper,lower,fract(point.y));
}
bool ShadowRayBoundsHit(ShadowRayNode node,vec3 origin,vec3 direction,float radius)
{
    float enter=0.0,leave=radius;
    for (int axis=0;axis<3;++axis)
    {
        if (direction[axis]==0.0)
        {
            if (origin[axis]<node.Lower[axis] || origin[axis]>node.Upper[axis]) return false;
        }
        else
        {
            float first=(node.Lower[axis]-origin[axis])/direction[axis],last=(node.Upper[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(first,last)); leave=min(leave,max(first,last));
            if (enter>leave) return false;
        }
    }
    return true;
}
float ShadowRayEdgeValue(vec2 a,vec2 b)
{
    float positive=a.x*b.y,negative=a.y*b.x,value=positive-negative;
    return abs(value)<=ShadowRayEdgeUncertainty*(abs(positive)+abs(negative)) ? 0.0 : value;
}
bool ShadowRayTriangleHit(ShadowRayTriangle triangle,vec3 origin,vec3 direction,float radius,out vec3 weights)
{
    weights=vec3(0.0); int major=abs(direction.x)>abs(direction.y) ? 0 : 1;
    if (abs(direction.z)>abs(direction[major])) major=2;
    int horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    vec3 first=triangle.A.xyz-origin,second=triangle.B.xyz-origin,third=triangle.C.xyz-origin;
    vec2 shear=vec2(direction[horizontal],direction[vertical])/direction[major];
    vec2 a=vec2(first[horizontal],first[vertical])-shear*first[major];
    vec2 b=vec2(second[horizontal],second[vertical])-shear*second[major];
    vec2 c=vec2(third[horizontal],third[vertical])-shear*third[major];
    vec3 edges=vec3(ShadowRayEdgeValue(b,c),ShadowRayEdgeValue(c,a),ShadowRayEdgeValue(a,b));
    if (any(lessThan(edges,vec3(0.0))) && any(greaterThan(edges,vec3(0.0)))) return false;
    float area=edges.x+edges.y+edges.z; if (area==0.0) return false;
    float distance=dot(edges,vec3(first[major],second[major],third[major]))/(area*direction[major]);
    if (!(distance>0.0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool ShadowRayAnyHit(vec3 origin,vec3 direction,float radius)
{
    uint index=0u;
    while (index<ShadowRayLoad(0u).y)
    {
        ShadowRayNode node=ShadowRayReadNode(index);
        if (!ShadowRayBoundsHit(node,origin,direction,radius)) { index=node.Link.w; continue; }
        if (node.Link.z==0u) { index=node.Link.x; continue; }
        for (uint primitive=0u;primitive<node.Link.z;++primitive)
        {
            uint triangleIndex=node.Link.y+primitive;
            ShadowRayTriangle triangle=ShadowRayReadTriangle(triangleIndex);
            if ((triangle.Identity.w & ShadowRayCasterFlag)==0u) continue;
            vec3 weights;
            if (!ShadowRayTriangleHit(triangle,origin,direction,radius,weights)) continue;
            ShadowRayMaterial material=ShadowRayReadMaterial(triangle.Identity.x);
            if (material.Alpha.y<=0.0) return true;
            vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
            float coverage=dot(weights,vec3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Alpha.x;
            coverage*=ShadowRayMaterialCoverage(material.Diffuse,uv)*ShadowRayMaterialCoverage(material.Opacity,uv);
            if (clamp(coverage,0.0,1.0)>=material.Alpha.y) return true;
        }
        index=node.Link.w;
    }
    return false;
}
const float ShadowRayNormalFloor=1e-12,ShadowRayOriginRoundoff=8.0*1.1920928955078125e-7;
bool ShadowRayEnabled() { return ShadowRayLoad(ShadowRayCountsHeader).x!=0u && ShadowRayLoad(ShadowRayLightHeader).y==ShadowRayImageABI; }
float ShadowRayVisibility(vec3 position,vec3 normal,vec3 towardLight)
{
    if (!ShadowRayEnabled()) return 1.0;
    vec4 parameters=uintBitsToFloat(ShadowRayLoad(ShadowRayTraceHeader));
    float scale=max(abs(towardLight.x),max(abs(towardLight.y),abs(towardLight.z)));
    if (!(scale>0.0) || !(!any(isnan(vec4(position,scale))) && !any(isinf(vec4(position,scale))))) return 1.0;
    vec3 scaled=towardLight/scale; float magnitude=length(scaled);
    vec3 direction=scaled/magnitude;
    bool finiteEmitter=ShadowRayLoad(ShadowRayLightHeader).x!=0u;
    float limit=finiteEmitter?min(scale*magnitude,parameters.z):parameters.z;
    if (!(limit>parameters.y)) return 1.0;
    float normalSquared=dot(normal,normal);
    vec3 unitNormal=normalSquared>ShadowRayNormalFloor?normal*inversesqrt(normalSquared):vec3(0.0);
    float roundoff=ShadowRayOriginRoundoff*max(abs(position.x),max(abs(position.y),abs(position.z)));
    vec3 normalOffset=unitNormal*max(parameters.x,roundoff);
    if (finiteEmitter)
    {
        // Bias the origin while preserving the actual emitter endpoint.
        vec3 endpointVector=towardLight-normalOffset;
        float endpointScale=max(abs(endpointVector.x),max(abs(endpointVector.y),abs(endpointVector.z)));
        if (!(endpointScale>0.0)) return 1.0;
        vec3 endpointUnit=endpointVector/endpointScale;
        float endpointMagnitude=length(endpointUnit);
        direction=endpointUnit/endpointMagnitude;
        limit=min(endpointScale*endpointMagnitude,parameters.z);
    }
    vec3 origin=position+normalOffset+direction*parameters.y;
    limit-=parameters.y;
    if (!(limit>0.0)) return 1.0;
    if (!(!any(isnan(origin)) && !any(isinf(origin)))) return 1.0;
    return ShadowRayAnyHit(origin,direction,limit)?0.0:1.0;
}

#endif

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

// Tiled deferred shading: G-buffer depth -> per-tile lists -> reconstructed lighting.
// Captures and transparent geometry retain independent complete-light paths.
// Lauritzen, "Deferred Rendering for Current and Future Rendering Pipelines" (SIGGRAPH 2010).
// https://www.intel.com/content/www/us/en/developer/articles/technical/deferred-rendering-for-current-and-future-rendering-pipelines.html
// Independent ES 3.1 Tiled Deferred surfaces. Storage uses top-left tile order while
// gl_FragCoord is bottom-left. Only the list lookup flips its pixel Y; material,
// fog and contact sampling retain their established ES conventions.
#if defined(EVANS_TILED_DEFERRED_SURFACE)
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
// Depth capture and material replay require identical raster positions.

layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
uniform mat4 World, NormalWorld, ViewProjection;
out vec3 WorldPosition, WorldNormal;
out vec4 VertexColor;
out vec2 TexCoord;
void main()
{
#ifdef EVANS_TILED_DEFERRED_LIGHTING
    TexCoord = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(TexCoord * 2.0 - 1.0, 0.0, 1.0);
    WorldPosition = WorldNormal = vec3(0.0); VertexColor = vec4(1.0);
#else
    const float NormalFloor = 1.0e-12, HomogeneousFloor = 1.0e-7;
    const float SrgbThreshold = 0.04045, SrgbLinearDivisor = 12.92;
    const float SrgbOffset = 0.055, SrgbScale = 1.055, SrgbExponent = 2.4;
    vec4 world = World * vec4(InPosition, 1.0);
    vec4 clip = ViewProjection * world;
    clip.z = clip.z * 2.0 - clip.w;
    gl_Position = clip;
    WorldPosition = world.xyz / max(abs(world.w), HomogeneousFloor);
    vec3 normal = mat3(NormalWorld) * InNormal;
    float squared = dot(normal, normal);
    WorldNormal = squared > NormalFloor ? normal * inversesqrt(squared) : vec3(0.0, 1.0, 0.0);
    vec3 encoded = clamp(InColor.rgb, 0.0, 1.0);
    vec3 low = encoded / SrgbLinearDivisor, high = pow((encoded + SrgbOffset) / SrgbScale, vec3(SrgbExponent));
    VertexColor = vec4(mix(high, low, lessThanEqual(encoded, vec3(SrgbThreshold))), InColor.a);
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
#endif
    CullWireFace();
}
#else
#ifndef EVANS_TILED_DEFERRED_FILTER
#define EVANS_TILED_DEFERRED_FILTER 0
#endif
const int PHONG_RECEIVER = 0, FLAT_RECEIVER = 1, PBR_RECEIVER = 3, TOON_RECEIVER = 4;
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

const int MaximumLights = 4, MaximumCascades = 4, MaximumProjections = 5;
const int DirectionalLight = 0, PointLight = 1;
const float Pi = 3.14159265358979323846;
const float NormalEpsilon = 1.0e-12, HomogeneousEpsilon = 1.0e-7;
const float HalfRadianceLimit = 65504.0, TexelCenter = 0.5;
const float MaximumMapFraction = 0.25, ShadowOpacity = 0.55;
const float ConstantBias = 0.00035, SlopeBias = 0.00060, MaximumBias = 0.0045, MinimumBiasCosine = 0.15;
uniform vec4 ShadowFiltering;
#if EVANS_TILED_DEFERRED_FILTER == 2
uniform vec4 PCSSProjection[MaximumProjections], PCSSEmitterLimits;
uniform uvec4 PCSSSamplesMaps;
#elif EVANS_TILED_DEFERRED_FILTER == 3
uniform vec4 MomentDepthEncoding[MaximumProjections], MomentSettings;
uniform uvec4 MomentOptions;
#endif
struct LightConstants
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
layout(std430, binding = 0) readonly buffer CompleteLights { LightConstants Lights[]; };
layout(std430, binding = 1) readonly buffer TileRecords { uvec4 Headers[]; };
layout(std430, binding = 2) readonly buffer TileEntries { uint Entries[]; };
uniform uvec4 ForwardLightLayout; // count, scene shadow index, mode, per-list capacity
uniform uvec4 ForwardSurfaceTiles; // tile side, columns, viewport width, viewport height
uniform vec4 CameraLightCount, ShadingParameters;
uniform mat4 LightViewProjection;
uniform mat4 CascadeLightMatrices[MaximumCascades];
uniform vec4 CascadeFarDistances, CascadeBlendStarts, CascadeEyeDepthPlane, CascadeParameters;
uniform highp sampler2D ShadowMap;
in vec3 WorldPosition, WorldNormal;
in vec4 VertexColor;
in vec2 TexCoord;

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
vec3 UnitVector(vec3 value, vec3 fallback)
{
    float squared = dot(value, value);
    return squared > NormalEpsilon ? value * inversesqrt(squared) : fallback;
}
#if EVANS_TILED_DEFERRED_FILTER == 3
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

#else
#if EVANS_TILED_DEFERRED_FILTER == 2
float TopLeftDepth(highp sampler2D image, ivec2 location)
{
    ivec2 extent = textureSize(image, 0);
    if (any(lessThan(location, ivec2(0))) || any(greaterThanEqual(location, extent))) return 1.0;
    return texelFetch(image, ivec2(location.x, extent.y - 1 - location.y), 0).r;
}
float Comparison(highp sampler2D image, ivec2 location, float receiver)
{
    return receiver <= TopLeftDepth(image, location) ? 1.0 : 0.0;
}
float BilinearVisibility(highp sampler2D image, vec2 uv, float receiver)
{
    vec2 footprint = uv * vec2(textureSize(image, 0)) - TexelCenter;
    ivec2 first = ivec2(floor(footprint));
    vec2 weight = fract(footprint);
    float upper = mix(Comparison(image, first, receiver), Comparison(image, first + ivec2(1, 0), receiver), weight.x);
    float lower = mix(Comparison(image, first + ivec2(0, 1), receiver), Comparison(image, first + ivec2(1), receiver), weight.x);
    return mix(upper, lower, weight.y);
}
float LightAxisDistance(float depth, vec4 metric)
{
    if (metric.x == 0.0) return depth * metric.y;
    return 1.0 / mix(1.0 / metric.x, 1.0 / metric.y, depth);
}
vec2 DiskLocation(uint index, uint count)
{
    const float GoldenAngle = Pi * (3.0 - sqrt(5.0));
    float angle = GoldenAngle * float(index);
    float radius = sqrt((float(index) + TexelCenter) / float(count));
    return radius * vec2(cos(angle), sin(angle));
}
vec2 ShadowFootprint(float radius, vec2 plane, vec2 resolution, float requestedLimit)
{
    vec2 projected = radius / plane;
    float span = max(projected.x * resolution.x, projected.y * resolution.y);
    float limit = min(requestedLimit, min(resolution.x, resolution.y) * MaximumMapFraction);
    return span > limit ? projected * (limit / span) : projected;
}
float EvaluatePCSS(highp sampler2D image, int projectionIndex, vec4 clip, float bias)
{
    if (clip.w <= HomogeneousEpsilon || (PCSSSamplesMaps.z & (1u << uint(projectionIndex))) == 0u) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 center = vec2(projected.x + 1.0, 1.0 - projected.y) * 0.5;
    if (any(lessThan(center, vec2(0.0))) || any(greaterThan(center, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec4 metric = PCSSProjection[projectionIndex];
    vec2 resolution = vec2(textureSize(image, 0));
    bool perspective = metric.x > 0.0;
    float lightSize = perspective ? PCSSEmitterLimits.x : PCSSEmitterLimits.y;
    float comparisonDepth = projected.z - bias;
    if (lightSize == 0.0) return Comparison(image, ivec2(floor(center * resolution)), comparisonDepth);
    float receiverDistance = LightAxisDistance(projected.z, metric);
    float searchRadius = perspective ? lightSize * (receiverDistance - metric.x) / receiverDistance : lightSize * receiverDistance;
    vec2 searchFootprint = ShadowFootprint(searchRadius, metric.zw, resolution, PCSSEmitterLimits.z);
    float distanceSum = 0.0;
    uint blockerCount = 0u;
    for (uint index = 0u; index < PCSSSamplesMaps.x; ++index)
    {
        vec2 uv = center + DiskLocation(index, PCSSSamplesMaps.x) * searchFootprint;
        float blocker = TopLeftDepth(image, ivec2(floor(uv * resolution)));
        if (blocker < comparisonDepth)
        {
            distanceSum += LightAxisDistance(blocker, metric);
            ++blockerCount;
        }
    }
    if (blockerCount == 0u) return 1.0;
    float blockerDistance = distanceSum / float(blockerCount);
    float separation = max(receiverDistance - blockerDistance, 0.0);
    float filterRadius = lightSize * separation;
    if (perspective) filterRadius *= metric.x / (blockerDistance * receiverDistance);
    vec2 filterFootprint = ShadowFootprint(filterRadius, metric.zw, resolution, PCSSEmitterLimits.w);
    float visibility = 0.0;
    for (uint index = 0u; index < PCSSSamplesMaps.y; ++index)
        visibility += BilinearVisibility(image, center + DiskLocation(index, PCSSSamplesMaps.y) * filterFootprint, comparisonDepth);
    return visibility / float(PCSSSamplesMaps.y);
}
#else
float FixedCompare(highp sampler2D image, ivec2 pixel, float receiver)
{
    ivec2 size = textureSize(image, 0);
    pixel = clamp(pixel, ivec2(0), size - 1);
    return receiver <= texelFetch(image, ivec2(pixel.x, size.y - 1 - pixel.y), 0).r ? 1.0 : 0.0;
}
float EvaluatePCSS(highp sampler2D image, int projectionIndex, vec4 clip, float bias)
{
    if (clip.w <= HomogeneousEpsilon) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = vec2(projected.x + 1.0, 1.0 - projected.y) * 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec2 dimensions = vec2(textureSize(image, 0)); float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0.0) return FixedCompare(image, ivec2(floor(uv * dimensions)), receiver);
    vec2 texel = uv * dimensions - TexelCenter, fraction = fract(texel);
    ivec2 base = ivec2(floor(texel)); float sum = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            ivec2 p = base + ivec2(x, y);
            float top = mix(FixedCompare(image, p, receiver), FixedCompare(image, p + ivec2(1, 0), receiver), fraction.x);
            float bottom = mix(FixedCompare(image, p + ivec2(0, 1), receiver), FixedCompare(image, p + 1, receiver), fraction.x);
            sum += mix(top, bottom, fraction.y);
        }
    return sum / ShadowFiltering.y;
}
#endif
float CascadeShadow(int cascade, vec3 world, float bias)
{
    if (cascade >= int(CascadeParameters.z)) return 1.0;
    vec4 clip = CascadeLightMatrices[cascade] * vec4(world, 1.0);
    if (cascade == 0) return EvaluatePCSS(CascadeDepth0, 1, clip, bias);
    if (cascade == 1) return EvaluatePCSS(CascadeDepth1, 2, clip, bias);
    if (cascade == 2) return EvaluatePCSS(CascadeDepth2, 3, clip, bias);
    return EvaluatePCSS(CascadeDepth3, 4, clip, bias);
}
float LightVisibility(vec3 world, float bias, bool cascaded)
{
    if (!cascaded) return EvaluatePCSS(ShadowMap, 0, LightViewProjection * vec4(world, 1.0), bias);
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

#endif
// Virtual receiver metadata occupies the exclusive cascade uniforms. Directory
// rows are uploaded verbatim; only rasterized atlas depth needs an ES Y flip.
// Filtering compares each logical corner after independent ancestor lookup.
const uint VirtualBorder = 1u;
const float VirtualCenter = 0.5;
vec2 VirtualDepth(vec2 uv, uint requestedMip)
{
    uvec4 extent = uvec4(CascadeFarDistances), physical = uvec4(CascadeBlendStarts);
    uint side = extent.x / extent.y, offset = 0u;
    for (uint mip = 0u; mip < extent.z; ++mip)
    {
        if (mip >= requestedMip)
        {
            uint resolution = side * extent.y;
            uvec2 pixel = min(uvec2(uv * float(resolution)), uvec2(resolution - 1u));
            uvec2 page = pixel / extent.y;
            uint address = offset + page.y * side + page.x;
            uint entry = uint(round(texelFetch(CascadeDepth1, ivec2(address % extent.w, address / extent.w), 0).r * float(physical.w)));
            if (entry != 0u)
            {
                uint slot = entry - 1u;
                uvec2 at = uvec2(slot % physical.z, slot / physical.z) * (extent.y + VirtualBorder * 2u) + VirtualBorder + pixel % extent.y;
                at.y = physical.y - 1u - at.y;
                return vec2(texelFetch(CascadeDepth0, ivec2(at), 0).r, float(mip));
            }
        }
        offset += side * side; side /= 2u;
    }
    return vec2(1.0, float(extent.z - 1u));
}
// Compare receiver and caster at the same resolved virtual texel center.
// The actual ancestor mip is essential when detail pages are not resident.
vec2 VirtualReceiverGradient(vec2 uv, float depth)
{
    vec2 x = dFdx(uv), y = dFdy(uv);
    float zx = dFdx(depth), zy = dFdy(depth);
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x, x) * dot(y, y), 1.0e-30);
    if (determinant * determinant <= scale * 1.0e-12) return vec2(0.0, 0.0);
    return vec2(y.y * zx - x.y * zy, x.x * zy - y.x * zx) / determinant;
}
float VirtualReceiverAt(vec2 sampleUv, uint actualMip, vec2 receiverUv, float receiver, vec2 gradient)
{
    float resolution = float(uint(CascadeFarDistances.x) >> actualMip);
    vec2 pixel = clamp(floor(sampleUv * resolution), vec2(0.0, 0.0), vec2(resolution - 1.0, resolution - 1.0));
    vec2 center = (pixel + vec2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualCompare(ivec2 pixel, uint mip, vec2 receiverUv, float receiver, vec2 gradient)
{
    uint resolution = uint(CascadeFarDistances.x) >> mip;
    vec2 uv = (vec2(clamp(pixel, ivec2(0, 0), ivec2(int(resolution) - 1, int(resolution) - 1))) + vec2(0.5, 0.5)) / float(resolution);
    vec2 stored = VirtualDepth(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
float VirtualVisibility(vec3 position, float bias, vec3 receiverData)
{
    vec4 light = CascadeLightMatrices[0] * vec4(position, 1.0);
    vec3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    vec2 uv = vec2(projected.x, -projected.y) * 0.5 + vec2(0.5, 0.5);
    vec2 gradient = receiverData.xy;
    uint mip = uint(receiverData.z);
    if (light.w <= CascadeEyeDepthPlane.w || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec2 stored = VirtualDepth(uv, mip);
    float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    vec2 texel = uv * float(uint(CascadeFarDistances.x) >> mip) - vec2(0.5, 0.5);
    ivec2 first = ivec2(floor(texel)); vec2 weight = fract(texel); float visible = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            ivec2 p = first + ivec2(x, y);
            float a = VirtualCompare(p, mip, uv, receiver, gradient);
            float b = VirtualCompare(p + ivec2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualCompare(p + ivec2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualCompare(p + ivec2(1, 1), mip, uv, receiver, gradient);
            visible += mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
        }
    return visible / ShadowFiltering.y;
}



// Resolve the receiver plane before entering a varying light-list loop.
vec3 VirtualReceiverData(vec3 position)
{
    if (ShadowFiltering.z == 0.0) return vec3(0.0, 0.0, 0.0);
    vec4 light = CascadeLightMatrices[0] * vec4(position, 1.0);
    vec2 uv = light.xy / max(light.w, CascadeEyeDepthPlane.w) * vec2(0.5, -0.5);
    float footprint = max(length(dFdx(uv)) * CascadeEyeDepthPlane.y, length(dFdy(uv)) * CascadeEyeDepthPlane.z) * CascadeFarDistances.x;
    return vec3(VirtualReceiverGradient(uv, light.z / max(light.w, CascadeEyeDepthPlane.w)), clamp(floor(log2(max(1.0, footprint)) + CascadeEyeDepthPlane.x), 0.0, CascadeFarDistances.z - 1.0));
}

float SurfaceShadow(vec3 position, vec3 normal, vec3 towardLight, vec3 receiverData)
{
    if (ShadowRayEnabled()) return 1.0-ShadowOpacity*(1.0-ShadowRayVisibility(position,normal,towardLight));
    float cosine = max(dot(UnitVector(normal, vec3(0.0, 1.0, 0.0)), UnitVector(towardLight, vec3(0.0, 1.0, 0.0))), 0.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumBiasCosine);
    float bias = min(ConstantBias + slope * SlopeBias, MaximumBias);
    if (ShadowFiltering.z != 0.0) return 1.0 - ShadowOpacity * (1.0 - VirtualVisibility(position, bias, receiverData));
    return 1.0 - ShadowOpacity * (1.0 - LightVisibility(position, bias, CascadeParameters.z > 0.0));
}
struct Material
{
    vec3 Position, Normal, EyeDirection, Diffuse, Specular;
    float Roughness, Metallic, Shininess, Bands;
    int Model;
    vec3 VirtualReceiver;
};
vec3 ReflectSample(Material material, vec3 towardLight, vec3 radiance)
{
    const float RoughnessFloor = 0.045, DistributionFloor = 1.0e-12, DenominatorFloor = 1.0e-6;
    const float DielectricF0 = 0.04, SchlickExponent = 5.0, HighlightThreshold = 0.5;
    const float MaximumBands = 16.0;
    if (dot(towardLight, towardLight) <= NormalEpsilon) return vec3(0.0);
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
        return radiance * nl * (diffuse + specular);
    }
    float specular = nl > 0.0 ? pow(nh, max(material.Shininess, 1.0)) : 0.0;
    if (material.Model == TOON_RECEIVER)
    {
        float bands = clamp(round(material.Bands), 2.0, MaximumBands);
        return radiance * (material.Diffuse * ToonResponse(nl, bands) + material.Specular * ToonHighlight(specular));
    }
    return radiance * (material.Diffuse * nl + material.Specular * specular);
}
vec3 ShadeLight(Material material, int index, bool shadows, int shadowIndex)
{
    const float DistanceFloor = 1.0e-6, ParallelAxis = 0.999;
    const int RectangleAxisSamples = 2;
    const float RectangleWeight = 1.0 / float(RectangleAxisSamples * RectangleAxisSamples);
    // Member loads avoid ANGLE's whole-struct SSBO lowering issue.
    LightConstants light;
    light.PositionType = Lights[index].PositionType;
    light.DirectionIntensity = Lights[index].DirectionIntensity;
    light.ColorCastShadows = Lights[index].ColorCastShadows;
    light.AreaSizePadding = Lights[index].AreaSizePadding;
    int type = int(light.PositionType.w + 0.5);
    vec3 direction = type == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - material.Position;
    float visibility = shadows && index == shadowIndex && light.ColorCastShadows.w > 0.5 ?
        SurfaceShadow(material.Position, material.Normal, direction, material.VirtualReceiver) : 1.0;
    if (index == shadowIndex && light.ColorCastShadows.w > 0.5)
        visibility = ContactShadow(visibility, ShadowOpacity);
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
    if (int(TextureFlags.w) == FLAT_RECEIVER)
    {
        normal = UnitVector(cross(dFdx(WorldPosition), dFdy(WorldPosition)), normal);
        if (dot(normal, WorldNormal) < 0.0) normal = -normal;
    }
    if (!gl_FrontFacing) normal = -normal;
    if (int(TextureFlags.w) != FLAT_RECEIVER && TextureFlags.y > 0.5)
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
// Overflow restores the full loop; total counts never become truncated light counts.
uvec2 SelectLightSpan(out bool indexed)
{
    uvec2 span = uvec2(0u, ForwardLightLayout.x); indexed = false;
    if (ForwardLightLayout.z != 0u)
    {
        uvec2 pixel = min(uvec2(gl_FragCoord.xy), ForwardSurfaceTiles.zw - 1u);
        pixel.y = ForwardSurfaceTiles.w - 1u - pixel.y;
        uvec2 tile = pixel / ForwardSurfaceTiles.x;
        uvec4 header = Headers[tile.y * ForwardSurfaceTiles.y + tile.x];
        uvec2 candidate = ForwardLightLayout.z == 1u ? header.xy : header.zw;
        if (candidate.y <= ForwardLightLayout.w) { indexed = true; span = candidate; }
    }
    return span;
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
    material.Position = WorldPosition; material.VirtualReceiver = VirtualReceiverData(material.Position); material.Normal = SurfaceNormal();
    material.EyeDirection = UnitVector(CameraLightCount.xyz - WorldPosition, material.Normal);
    material.Diffuse = base * max(MaterialDiffuseColor.rgb, vec3(0.0));
    material.Specular = max(MaterialSpecularColor.rgb, vec3(0.0));
    material.Roughness = roughnessMetallic.x; material.Metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    material.Shininess = MaterialParameters.x; material.Bands = ShadingParameters.z; material.Model = int(TextureFlags.w);
    vec3 color = ToonAmbient(base * max(MaterialDiffuseColor.rgb, vec3(0.0)), base * max(MaterialAmbientColor.rgb, vec3(0.0)) * max(AmbientColorStrength.rgb, vec3(0.0)) * max(AmbientColorStrength.w, 0.0), float(TextureFlags.w));
    if (material.Model == PBR_RECEIVER) color *= 1.0 - material.Metallic;
    color += max(MaterialEmissiveColor.rgb, vec3(0.0));
    bool indexed; uvec2 span = SelectLightSpan(indexed);
    for (uint offset = 0u; offset < span.y; ++offset)
    {
        uint index = indexed ? Entries[span.x + offset] : offset;
        color += ShadeLight(material, int(index), RenderParameters.x > 0.5, int(ForwardLightLayout.y));
    }

    color += material.Diffuse * (material.Model==PBR_RECEIVER ? (1.0-LumenDielectricReflectance)*(1.0-material.Metallic) : 1.0) * LumenTranslucentIrradiance(material.Position,material.Normal);
    color = max(FoggedSurface(color), vec3(0.0));
    color = ShadingParameters.w > 0.5 ? min(color, vec3(HalfRadianceLimit)) :
        clamp(color / max(1.0, max(color.r, max(color.g, color.b))), 0.0, 1.0);
    return vec4(color, opacity);
}
#if defined(EVANS_TILED_DEFERRED_LIGHTING)
uniform mat4 InverseViewProjection;
uniform vec4 BackgroundColor;
uniform uvec4 LightingOptions;
#ifdef EVANS_MULTISAMPLE
uniform int LightingSample;
uniform vec4 SampleLocation;
uniform highp sampler2DMS DiffuseOpacity, NormalRoughness, SpecularShininess, AmbientMetallic, EmissionColor, SceneDepth;
#else
uniform highp sampler2D DiffuseOpacity, NormalRoughness, SpecularShininess, AmbientMetallic, EmissionColor, SceneDepth;
#endif
// Reconstruct one-sided depth slopes on the same geometric surface.
// Fragment-quad derivatives can cross a G-buffer silhouette or intersection.
// Camera depth is affine within a rasterized triangle; lower second difference
// chooses the side that best preserves that plane without using shading normals.
float DeferredReceiverDepth(ivec2 pixel, uint sampleIndex, uvec2 size)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ivec2(size)))) return 1.0;
    return texelFetch(SceneDepth, pixel, int(sampleIndex)).r;
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
vec3 DeferredReceiverLightPoint(vec2 coordinate, float depth)
{
    vec4 world = InverseViewProjection * vec4((coordinate * 2.0 - 1.0), depth, 1.0);
    vec4 light = CascadeLightMatrices[0] * vec4(world.xyz / world.w, 1.0);
    vec3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    return vec3(projected.xy * vec2(0.5, -0.5) + 0.5, projected.z);
}
vec3 DeferredVirtualReceiverData(vec2 coordinate, float depth, uint sampleIndex)
{
    if (ShadowFiltering.z == 0.0) return vec3(0.0);
#if defined(EVANS_MULTISAMPLE)
    uvec2 size = uvec2(textureSize(SceneDepth));
#else
    uvec2 size = uvec2(textureSize(SceneDepth, 0));
#endif
    ivec2 pixel = ivec2(coordinate * vec2(size));
    float dx = DeferredReceiverSlope(depth,
        DeferredReceiverDepth(pixel + ivec2(-1, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(-2, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(1, 0), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(2, 0), sampleIndex, size));
    float dy = DeferredReceiverSlope(depth,
        DeferredReceiverDepth(pixel + ivec2(0, -1), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(0, -2), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(0, 1), sampleIndex, size),
        DeferredReceiverDepth(pixel + ivec2(0, 2), sampleIndex, size));
    vec3 center = DeferredReceiverLightPoint(coordinate, depth);
    vec3 x = DeferredReceiverLightPoint(coordinate + vec2(1.0 / float(size.x), 0), depth + dx) - center;
    vec3 y = DeferredReceiverLightPoint(coordinate + vec2(0, 1.0 / float(size.y)), depth + dy) - center;
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x.xy, x.xy) * dot(y.xy, y.xy), 1.0e-30);
    vec2 gradient = determinant * determinant > scale * 1.0e-12
        ? vec2(y.y * x.z - x.y * y.z, x.x * y.z - y.x * x.z) / determinant : vec2(0, 0);
    float footprint = max(length(x.xy * CascadeEyeDepthPlane.y), length(y.xy * CascadeEyeDepthPlane.z)) * CascadeFarDistances.x;
    float mip = clamp(floor(log2(max(1.0, footprint)) + CascadeEyeDepthPlane.x), 0.0, CascadeFarDistances.z - 1.0);
    return vec3(gradient, mip);
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#ifdef EVANS_MULTISAMPLE
    int sampleIndex = LightingSample;
    vec2 uv = (vec2(pixel) + SampleLocation.xy) / vec2(textureSize(SceneDepth));
#else
    int sampleIndex = 0;
    vec2 uv = TexCoord;
#endif
    vec4 normal = texelFetch(NormalRoughness, pixel, sampleIndex);
    vec4 diffuse = texelFetch(DiffuseOpacity, pixel, sampleIndex);
    vec3 emission = texelFetch(EmissionColor, pixel, sampleIndex).rgb;
    if (normal.w < 0.0) { OutColor = vec4(emission, diffuse.a); return; }
    const float ClearDepth = 1.0;
    float depth = texelFetch(SceneDepth, pixel, sampleIndex).r;
    if (depth >= ClearDepth) { OutColor = BackgroundColor; return; }
    vec4 specular = texelFetch(SpecularShininess, pixel, sampleIndex);
    vec4 ambient = texelFetch(AmbientMetallic, pixel, sampleIndex);
    vec4 world = InverseViewProjection * vec4(uv * 2.0 - 1.0, depth, 1.0);
    Material material;
    material.Position = world.xyz / world.w; material.VirtualReceiver = DeferredVirtualReceiverData(uv, depth, uint(sampleIndex));
    material.Normal = UnitVector(normal.xyz, vec3(0.0, 1.0, 0.0));
    material.EyeDirection = UnitVector(CameraLightCount.xyz - material.Position, material.Normal);
    material.Diffuse = diffuse.rgb; material.Specular = specular.rgb;
    material.Roughness = normal.w; material.Metallic = clamp(ambient.w, 0.0, 1.0);
    material.Shininess = specular.w; material.Bands = ShadingParameters.x; material.Model = int(LightingOptions.x);
    vec3 radiance = ambient.rgb + emission;
    bool indexed; uvec2 span = SelectLightSpan(indexed);
    for (uint ordinal = 0u; ordinal < span.y; ++ordinal)
    {
        uint light = indexed ? Entries[span.x + ordinal] : ordinal;
        radiance += ShadeLight(material, int(light), LightingOptions.y != 0u, int(ForwardLightLayout.y));
    }
    OutColor = vec4(clamp(radiance, 0.0, HalfRadianceLimit), diffuse.a);
}
#elif defined(EVANS_TILED_DEFERRED_DEPTH)
void main()
{
    float alpha = VertexColor.a * MaterialParameters.y;
    if (TextureFlags.x > 0.5) alpha *= texture(DiffuseMap, TexCoord).a;
    if (TextureFlags.z > 0.5) alpha *= texture(OpacityMap, TexCoord).r;
    if (MaterialParameters.z > 0.0 && clamp(alpha, 0.0, 1.0) < MaterialParameters.z) discard;
}
#else
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
#endif
#else
// Independent ES 3.1 Tiled Deferred assignment; all list offsets use top-left tiles.
// Harada, McKee and Yang, "Forward+: Bringing Deferred Lighting to the Next
// Level" (2012): https://takahiroharada.github.io/forward%2B/
// Two membership bitsets feed a Blelloch exclusive scan and ordered scatter.
// Scan background: Harris, Sengupta and Owens, GPU Gems 3, Chapter 39:
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// Depth is stored bottom-left but encodes the renderer's increasing [0,1] Z.
// The supplied camera rows belong to that canonical projection, before the
// raster vertex shader converts clip Z to the ES [-W,+W] convention.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DMS;

#define FORWARD_PLUS_LANES 64
layout(local_size_x = FORWARD_PLUS_LANES, local_size_y = 1, local_size_z = 1) in;
const uint LaneCount = uint(FORWARD_PLUS_LANES);
const uint WordWidth = 32u;
const uint MaximumLights = 4096u;
const uint WordCount = MaximumLights / WordWidth; // power-of-two scan tree
const uint TileSides = 4u;
const uint OpaqueDepthPlanes = TileSides;
const uint TransparentDepthPlanes = OpaqueDepthPlanes + 2u;
const uint FrustumPlanes = TransparentDepthPlanes + 2u;
const uint ListsPerTile = 2u;
const uint NegativeZeroBits = 0x80000000u; // IEEE single-precision sign bit
const float FloatEpsilon = 1.1920928955078125e-7;
const float PlaneErrorOperations = 16.0;
const float PlaneTolerance = PlaneErrorOperations * FloatEpsilon;
const float SmallestNormal = 1.1754943508222875e-38;
const float FlushOperations = 64.0;
const float FlushTolerance = FlushOperations * SmallestNormal;

struct Bound { vec4 CenterRadius; uvec4 Flags; };
layout(std430, binding = 0) readonly buffer LightSupport { Bound Bounds[]; };
layout(std430, binding = 1) writeonly buffer TileRecords { uvec4 Headers[]; };
layout(std430, binding = 2) writeonly buffer TileIndices { uint Indices[]; };
uniform vec4 CameraRows[4];
uniform uvec4 ExtentAndLights; // width, height, depth samples, light count
uniform uvec4 TileLayout; // tile side, list capacity, grid width/height
#ifdef EVANS_TILED_DEFERRED_MULTISAMPLE
uniform sampler2DMS CameraDepth;
#else
uniform sampler2D CameraDepth;
#endif

shared uint MinimumDepth, MaximumDepth;
shared vec4 Planes[FrustumPlanes], Uncertainty[FrustumPlanes];
shared uvec2 Members[WordCount], Prefix[WordCount];
shared uvec2 Totals;

float Largest(vec4 value)
{ return max(max(value.x, value.y), max(value.z, value.w)); }
void SetBoundary(uint destination, vec4 row, float edge, float signIntoTile)
{
    float scale = max(SmallestNormal, max(Largest(abs(row)), Largest(abs(CameraRows[3]))));
    vec4 a = row / scale, w = CameraRows[3] / scale, b = edge * w;
    Planes[destination] = signIntoTile * (a - b);
    // Include original rows before subtraction and absolute NDC conversion
    // error. Relative error of the resulting plane alone misses cancellation.
    Uncertainty[destination] = PlaneTolerance * (abs(a) + abs(b) + abs(w));
}
bool Inside(uint planeIndex, vec4 center, float radius)
{
    vec4 plane = Planes[planeIndex], error = Uncertainty[planeIndex];
    vec4 products = plane * center;
    float distance = (products.x + products.y) + (products.z + products.w);
    float support = radius * length(plane.xyz);
    float allowance = dot(error, abs(center)) + radius * length(error.xyz);
    allowance += PlaneTolerance * (dot(abs(products), vec4(1.0)) + support) + FlushTolerance;
    return distance + support >= -allowance;
}
bvec2 Accept(vec4 centerRadius, uint unbounded)
{
    if (unbounded != 0u) return bvec2(true);
    float divisor = max(1.0, Largest(abs(centerRadius)));
    vec4 center = vec4(centerRadius.xyz, 1.0) / divisor;
    float radius = centerRadius.w / divisor;
    for (uint side = 0u; side < TileSides; ++side)
        if (!Inside(side, center, radius)) return bvec2(false);
    return bvec2(Inside(OpaqueDepthPlanes, center, radius) && Inside(OpaqueDepthPlanes + 1u, center, radius),
        Inside(TransparentDepthPlanes, center, radius) && Inside(TransparentDepthPlanes + 1u, center, radius));
}
void WriteMembers(uint bits, uint firstOrdinal, uint destination, uint word)
{
    uint ordinal = firstOrdinal;
    while (bits != 0u)
    {
        uint light = word * WordWidth + uint(findLSB(bits));
        if (ordinal < TileLayout.y) Indices[destination + ordinal] = light;
        ++ordinal; bits &= bits - 1u;
    }
}
void main()
{
    uint lane = gl_LocalInvocationIndex;
    uvec2 tile = gl_WorkGroupID.xy;
    if (lane == 0u) { MinimumDepth = floatBitsToUint(1.0); MaximumDepth = floatBitsToUint(0.0); }
    for (uint word = lane; word < WordCount; word += LaneCount) Members[word] = uvec2(0u);
    barrier();
    for (uint cell = lane; cell < TileLayout.x * TileLayout.x; cell += LaneCount)
    {
        uvec2 pixel = tile * TileLayout.x + uvec2(cell % TileLayout.x, cell / TileLayout.x);
        if (any(greaterThanEqual(pixel, ExtentAndLights.xy))) continue;
        ivec2 nativePixel = ivec2(pixel.x, ExtentAndLights.y - 1u - pixel.y);
        for (uint depthSample = 0u; depthSample < ExtentAndLights.z; ++depthSample)
        {
#ifdef EVANS_TILED_DEFERRED_MULTISAMPLE
            float depth = texelFetch(CameraDepth, nativePixel, int(depthSample)).r;
#else
            float depth = texelFetch(CameraDepth, nativePixel, 0).r;
#endif
            uint encoded = floatBitsToUint(depth);
            if (encoded == NegativeZeroBits) encoded = 0u;
            // Nonnegative normalized float encodings are ordered integers.
            // NaNs, infinities and negative values all exceed the encoding of one.
            bool valid = encoded <= floatBitsToUint(1.0);
            atomicMin(MinimumDepth, valid ? encoded : 0u);
            atomicMax(MaximumDepth, valid ? encoded : floatBitsToUint(1.0));
        }
    }
    barrier();
    vec2 first = vec2(tile * TileLayout.x) * 2.0 / vec2(ExtentAndLights.xy) - 1.0;
    vec2 last = vec2(min((tile + 1u) * TileLayout.x, ExtentAndLights.xy)) * 2.0 / vec2(ExtentAndLights.xy) - 1.0;
    if (lane == 0u) SetBoundary(lane, CameraRows[0], first.x, 1.0);
    if (lane == 1u) SetBoundary(lane, CameraRows[0], last.x, -1.0);
    if (lane == 2u) SetBoundary(lane, CameraRows[1], -first.y, -1.0);
    if (lane == 3u) SetBoundary(lane, CameraRows[1], -last.y, 1.0);
    if (lane == OpaqueDepthPlanes) SetBoundary(lane, CameraRows[2], uintBitsToFloat(MinimumDepth), 1.0);
    if (lane == OpaqueDepthPlanes + 1u) SetBoundary(lane, CameraRows[2], uintBitsToFloat(MaximumDepth), -1.0);
    if (lane == TransparentDepthPlanes) SetBoundary(lane, CameraRows[2], 0.0, 1.0);
    if (lane == TransparentDepthPlanes + 1u) SetBoundary(lane, CameraRows[2], 1.0, -1.0);
    barrier();
    for (uint light = lane; light < ExtentAndLights.w; light += LaneCount)
    {
        // Member loads preserve the std430 ABI without requiring a compiler to
        // materialize an aggregate value from a runtime-sized storage array.
        bvec2 accepted = Accept(Bounds[light].CenterRadius, Bounds[light].Flags.x);
        uint word = light / WordWidth, bit = 1u << (light % WordWidth);
        if (accepted.x) atomicOr(Members[word].x, bit);
        if (accepted.y) atomicOr(Members[word].y, bit);
    }
    barrier();
    for (uint word = lane; word < WordCount; word += LaneCount) Prefix[word] = uvec2(bitCount(Members[word]));
    barrier();
    // Blelloch upsweep combines disjoint pairs; the downsweep distributes each
    // subtree's exclusive start. All invocations reach every shared-memory
    // barrier, including tiles containing no accepted lights.
    for (uint span = 2u; span <= WordCount; span *= 2u)
    {
        uint right = (lane + 1u) * span - 1u;
        if (right < WordCount) Prefix[right] += Prefix[right - span / 2u];
        barrier();
    }
    if (lane == 0u) { Totals = Prefix[WordCount - 1u]; Prefix[WordCount - 1u] = uvec2(0u); }
    barrier();
    for (uint span = WordCount; span >= 2u; span /= 2u)
    {
        uint right = (lane + 1u) * span - 1u;
        if (right < WordCount)
        {
            uint left = right - span / 2u;
            uvec2 subtotal = Prefix[left]; Prefix[left] = Prefix[right]; Prefix[right] += subtotal;
        }
        barrier();
    }
    uint tileIndex = tile.y * TileLayout.z + tile.x;
    uint base = tileIndex * ListsPerTile * TileLayout.y;
    for (uint word = lane; word < WordCount; word += LaneCount)
    {
        WriteMembers(Members[word].x, Prefix[word].x, base, word);
        WriteMembers(Members[word].y, Prefix[word].y, base + TileLayout.y, word);
    }
    // Header counts deliberately exceed capacity on overflow. Pixel lighting
    // must then use every scene light; it must not shade with only this prefix.
    if (lane == 0u) Headers[tileIndex] = uvec4(base, Totals.x, base + TileLayout.y, Totals.y);
}

#endif
