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

#if defined(EVANS_FORWARD_PLUS_SURFACE) && !defined(EVANS_VERTEX_STAGE)
// Vulkan world-space shadow queries. Woop, Benthin and Wald, JCGT 2013.
// https://jcgt.org/published/0002/01/05/ . Forward escape links bound traversal;
// independent alpha planes preserve bilinear cutout coverage. The RGBA32UI ABI
// is described in RayTracedShadowScene.h. No screen-space visibility is assumed.
layout(set=3,binding=6) uniform usampler2D ShadowRayScene;
const uint ShadowRayCasterFlag=1u,ShadowRayImageABI=1u;
const uint ShadowRayCountsHeader=0u,ShadowRayOffsetsHeader=1u,ShadowRayTraceHeader=2u,ShadowRayLightHeader=3u;
const uint ShadowRayNodeSize=3u,ShadowRayTriangleSize=6u,ShadowRayMaterialSize=3u,ShadowRayScalarLanes=4u;
const uint ShadowRayNodeMaximum=1u,ShadowRayNodeLinks=2u;
const uint ShadowRayTriangleUV01=3u,ShadowRayTriangleUV2=4u,ShadowRayTriangleMetadata=5u;
const uint ShadowRayMaterialOpacity=1u,ShadowRayMaterialCoverageWord=2u;
uvec4 ShadowRayLoad(uint address)
{ uint width=uint(max(textureSize(ShadowRayScene,0).x,1)); return texelFetch(ShadowRayScene,ivec2(address%width,address/width),0); }
struct ShadowRayNode { vec4 Minimum,Maximum; uvec4 Links; };
struct ShadowRayTriangle { vec4 A,B,C,UVAB,UVC; uvec4 Metadata; };
struct ShadowRayMaterial { uvec4 DiffuseAlpha,OpacityRed; vec4 Coverage; };
const float ShadowRayPixelCenter=0.5;
const float ShadowRayBinary32Epsilon=1.1920928955078125e-7;
const float ShadowRayEdgeError=3.0*ShadowRayBinary32Epsilon/(1.0-3.0*ShadowRayBinary32Epsilon);
ShadowRayNode ShadowRayReadNode(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).x+index*ShadowRayNodeSize;
    ShadowRayNode result;
    result.Minimum=uintBitsToFloat(ShadowRayLoad(base));
    result.Maximum=uintBitsToFloat(ShadowRayLoad(base+1u));
    result.Links=ShadowRayLoad(base+2u);
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
    result.Metadata=ShadowRayLoad(base+5u);
    return result;
}
ShadowRayMaterial ShadowRayReadMaterial(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).z+index*ShadowRayMaterialSize;
    ShadowRayMaterial result;
    result.DiffuseAlpha=ShadowRayLoad(base);
    result.OpacityRed=ShadowRayLoad(base+1u);
    result.Coverage=uintBitsToFloat(ShadowRayLoad(base+2u));
    return result;
}
float ShadowRayScalar(uint index)
{ uvec4 value=ShadowRayLoad(ShadowRayLoad(ShadowRayOffsetsHeader).w+index/ShadowRayScalarLanes); return uintBitsToFloat(value)[index%ShadowRayScalarLanes]; }
float ShadowRaySampleCoverage(uvec4 image,vec2 uv)
{
    if (image.w==0u) return 1.0;
    vec2 texel=fract(vec2(uv.x,-uv.y))*vec2(image.yz)-ShadowRayPixelCenter;
    uvec2 a=uvec2(texel.x<0.0 ? image.y-1u : uint(floor(texel.x)),texel.y<0.0 ? image.z-1u : uint(floor(texel.y)));
    uvec2 b=uvec2(a.x+1u==image.y ? 0u : a.x+1u,a.y+1u==image.z ? 0u : a.y+1u);
    float top=mix(ShadowRayScalar(image.x+a.y*image.y+a.x),ShadowRayScalar(image.x+a.y*image.y+b.x),fract(texel.x));
    float bottom=mix(ShadowRayScalar(image.x+b.y*image.y+a.x),ShadowRayScalar(image.x+b.y*image.y+b.x),fract(texel.x));
    return mix(top,bottom,fract(texel.y));
}
bool ShadowRayIntersectsBounds(ShadowRayNode node,vec3 origin,vec3 ray,float radius)
{
    float nearDistance=0.0,farDistance=radius;
    for (int axis=0;axis<3;++axis)
    {
        if (ray[axis]==0.0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float a=(node.Minimum[axis]-origin[axis])/ray[axis],b=(node.Maximum[axis]-origin[axis])/ray[axis];
            nearDistance=max(nearDistance,min(a,b)); farDistance=min(farDistance,max(a,b));
            if (nearDistance>farDistance) return false;
        }
    }
    return true;
}
float ShadowRayProjectedEdge(vec2 first,vec2 second)
{
    precise float positive=first.x*second.y,negative=first.y*second.x;
    precise float area=positive-negative;
    return abs(area)<=ShadowRayEdgeError*(abs(positive)+abs(negative)) ? 0.0 : area;
}
bool ShadowRayIntersectsTriangle(ShadowRayTriangle triangle,vec3 origin,vec3 ray,float radius,out vec3 weights)
{
    weights=vec3(0.0); int major=abs(ray.x)>abs(ray.y) ? 0 : 1;
    if (abs(ray.z)>abs(ray[major])) major=2;
    int horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    vec3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    vec2 shear=vec2(ray[horizontal],ray[vertical])/ray[major];
    precise vec2 first=vec2(a[horizontal],a[vertical])-shear*a[major];
    precise vec2 second=vec2(b[horizontal],b[vertical])-shear*b[major];
    precise vec2 third=vec2(c[horizontal],c[vertical])-shear*c[major];
    vec3 edges=vec3(ShadowRayProjectedEdge(second,third),ShadowRayProjectedEdge(third,first),ShadowRayProjectedEdge(first,second));
    if (any(lessThan(edges,vec3(0.0))) && any(greaterThan(edges,vec3(0.0)))) return false;
    float area=edges.x+edges.y+edges.z; if (area==0.0) return false;
    float distance=dot(edges,vec3(a[major],b[major],c[major]))/(area*ray[major]);
    if (!(distance>0.0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool ShadowRayTraceOcclusion(vec3 origin,vec3 ray,float radius)
{
    uint current=0u;
    while (current<ShadowRayLoad(0u).y)
    {
        ShadowRayNode node=ShadowRayReadNode(current);
        if (!ShadowRayIntersectsBounds(node,origin,ray,radius)) { current=node.Links.w; continue; }
        if (node.Links.z==0u) { current=node.Links.x; continue; }
        for (uint index=0u;index<node.Links.z;++index)
        {
            ShadowRayTriangle triangle=ShadowRayReadTriangle(node.Links.y+index);
            if ((triangle.Metadata.w & ShadowRayCasterFlag)==0u) continue; vec3 weights;
            if (!ShadowRayIntersectsTriangle(triangle,origin,ray,radius,weights)) continue;
            ShadowRayMaterial material=ShadowRayReadMaterial(triangle.Metadata.x); if (material.Coverage.y<=0.0) return true;
            vec2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
            float alpha=dot(weights,vec3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Coverage.x;
            alpha*=ShadowRaySampleCoverage(material.DiffuseAlpha,uv)*ShadowRaySampleCoverage(material.OpacityRed,uv);
            if (clamp(alpha,0.0,1.0)>=material.Coverage.y) return true;
        }
        current=node.Links.w;
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
    return ShadowRayTraceOcclusion(origin,direction,limit)?0.0:1.0;
}

#endif

#ifndef EVANS_VERTEX_STAGE
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

#endif

// Native Vulkan Forward+ surface/depth programs. The material set adds three
// storage buffers; the existing instance and fog sets remain independent.
// Scene-order indices survive tile overflow and capture-camera full traversal.
#if defined(EVANS_FORWARD_PLUS_SURFACE)
#ifndef EVANS_FORWARD_PLUS_FILTER
#define EVANS_FORWARD_PLUS_FILTER 0
#endif
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

const int PCSSMapCount = 5; // One single map and four cascade intervals.
struct ForwardShadowData
{
    vec4 ProjectionMetrics[PCSSMapCount];
    vec4 EmitterAndLimits;
    uvec4 SamplesAndMaps;
};
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
    ForwardShadowData Filter;
    uvec4 LightLayout; // count, scene shadow index, list mode, capacity
    uvec4 SurfaceTiles; // tile side, columns, viewport extent
} Draw;

#define Pcss Draw.Filter
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

#ifdef EVANS_VERTEX_STAGE
// Depth capture and material replay require identical raster positions.
invariant gl_Position;
precise gl_Position;
layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
layout(location = 0) out vec3 WorldPosition;
layout(location = 1) out vec3 WorldNormal;
layout(location = 2) out vec4 VertexColor;
layout(location = 3) out vec4 LightPosition;
layout(location = 4) out vec2 TexCoord;
void main()
{
    vec4 position = Draw.World * vec4(InPosition, 1.0);
    vec3 normal = NormalizeOr(mat3(Draw.NormalWorld) * InNormal, vec3(0.0, 1.0, 0.0));
    if (Draw.RenderParameters.y > OutlineMode)
        position.xyz += normal * Draw.MaterialParameters.w * OutlineWorldScale;
    gl_Position = Draw.ViewProjection * position;
    gl_Position.y = -gl_Position.y;
    WorldPosition = position.xyz / max(abs(position.w), HomogeneousEpsilon);
    WorldNormal = normal;
    VertexColor = vec4(DecodeSrgb(clamp(InColor.rgb, 0.0, 1.0)), InColor.a);
    LightPosition = Draw.LightViewProjection * position;
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
}
#else
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

layout(std430, set = 0, binding = 7) readonly buffer SceneEmitterBuffer { SceneLight SceneEmitters[]; };
layout(std430, set = 0, binding = 8) readonly buffer MaterialTileBuffer { uvec4 MaterialTiles[]; };
layout(std430, set = 0, binding = 9) readonly buffer MaterialIndexBuffer { uint MaterialIndices[]; };
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
#define CONTACT_BINDING 6
#define CONTACT_SAMPLER SurfaceSampler
// Sample-frequency contact lookup must not move the existing pixel-centred
// material/alpha evaluation. In particular, cutouts must match the depth pass.
#if defined(EVANS_CONTACT_RECEIVER) && defined(EVANS_MULTISAMPLE)
#define WorldPosition interpolateAtOffset(WorldPosition, vec2(0.0))
#define WorldNormal interpolateAtOffset(WorldNormal, vec2(0.0))
#define VertexColor interpolateAtOffset(VertexColor, vec2(0.0))
#define LightPosition interpolateAtOffset(LightPosition, vec2(0.0))
#define TexCoord interpolateAtOffset(TexCoord, vec2(0.0))
#endif

// Contact is a separate camera-depth producer. Apply artistic strength once:
// (1 - strength) + strength * rawMapVisibility * rawContactVisibility.
#if defined(EVANS_CONTACT_RECEIVER)
layout(set = 0, binding = CONTACT_BINDING) uniform texture2DArray ContactImage;
#endif
float ComposeContact(float primaryVisibility)
{
#if defined(EVANS_CONTACT_RECEIVER)
    const float ContactShadowStrength = 0.55;
#ifdef EVANS_MULTISAMPLE
    int sampleIndex = gl_SampleID;
#else
    int sampleIndex = 0;
#endif
    float contact = texelFetch(sampler2DArray(ContactImage, CONTACT_SAMPLER),
        ivec3(ivec2(gl_FragCoord.xy), sampleIndex), 0).r;
    return mix(1.0 - ContactShadowStrength, primaryVisibility, contact);
#else
    return primaryVisibility;
#endif
}

ivec2 ShadowExtent(uint map)
{
    if (map == 0u) return textureSize(PrimaryImage, 0);
    if (map == 1u) return textureSize(CascadeImage0, 0);
    if (map == 2u) return textureSize(CascadeImage1, 0);
    if (map == 3u) return textureSize(CascadeImage2, 0);
    return textureSize(CascadeImage3, 0);
}
#if EVANS_FORWARD_PLUS_FILTER == 3
vec2 MomentTexel(uint map, ivec2 pixel)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ShadowExtent(map)))) return vec2(1.0);
    if (map == 0u) return texelFetch(PrimaryImage, pixel, 0).rg;
    if (map == 1u) return texelFetch(CascadeImage0, pixel, 0).rg;
    if (map == 2u) return texelFetch(CascadeImage1, pixel, 0).rg;
    if (map == 3u) return texelFetch(CascadeImage2, pixel, 0).rg;
    return texelFetch(CascadeImage3, pixel, 0).rg;
}
vec2 FilteredMoments(uint map, vec2 uv)
{
    const float TexelCenter = 0.5;
    vec2 pixel = uv * vec2(ShadowExtent(map)) - TexelCenter;
    ivec2 lower = ivec2(floor(pixel)); vec2 weight = fract(pixel);
    vec2 upper = mix(MomentTexel(map, lower), MomentTexel(map, lower + ivec2(1, 0)), weight.x);
    vec2 bottom = mix(MomentTexel(map, lower + ivec2(0, 1)), MomentTexel(map, lower + ivec2(1)), weight.x);
    return mix(upper, bottom, weight.y);
}
float PcssVisibility(uint map, vec4 clip, float linearBias)
{
    if ((Draw.Filter.SamplesAndMaps.x & (1u << map)) == 0u || clip.w <= HomogeneousEpsilon) return 1.0;
    vec3 device = clip.xyz / clip.w;
    vec2 uv = vec2(device.x + 1.0, 1.0 - device.y) * 0.5;
    if (device.z < 0.0 || device.z > 1.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    float depth = dot(Draw.Filter.ProjectionMetrics[map].xyz, vec3(clip.zw, 1.0));
    vec2 distribution = FilteredMoments(map, uv);
    float difference = depth - linearBias - distribution.x;
    if (difference <= 0.0) return 1.0;
    float variance = max(distribution.y - distribution.x * distribution.x, Draw.Filter.EmitterAndLimits.x);
    float probability = variance / (variance + difference * difference);
    // Bleeding reduction is an artistic remap, not a stronger probability bound.
    return clamp((probability - Draw.Filter.EmitterAndLimits.y) / (1.0 - Draw.Filter.EmitterAndLimits.y), 0.0, 1.0);
}

#elif EVANS_FORWARD_PLUS_FILTER == 2
float ShadowDepthAt(uint map, ivec2 pixel)
{
    if (map == 0u) return texelFetch(PrimaryImage, pixel, 0).r;
    if (map == 1u) return texelFetch(CascadeImage0, pixel, 0).r;
    if (map == 2u) return texelFetch(CascadeImage1, pixel, 0).r;
    if (map == 3u) return texelFetch(CascadeImage2, pixel, 0).r;
    return texelFetch(CascadeImage3, pixel, 0).r;
}
// Fernando, "Percentage-Closer Soft Shadows" (SIGGRAPH 2005 sketch).
// https://developer.download.nvidia.com/shaderlibrary/docs/shadow_PCSS.pdf
// Blockers are averaged in linear light distance, not projected depth. Bias
// classifies blockers/comparisons but does not change the penumbra geometry.
// The disk quadrature is deterministic; both loops have independent GUI budgets.
const float PcssPi = 3.14159265358979323846;
const float GoldenAngle = PcssPi * (3.0 - sqrt(5.0));
const float PcssProjectionEpsilon = 1.0e-7;
const float MaximumMapFraction = 0.25;

vec2 DiskSample(uint index, uint count)
{
    float angle = float(index) * GoldenAngle;
    float radius = sqrt((float(index) + 0.5) / float(count));
    return radius * vec2(cos(angle), sin(angle));
}
float LinearLightDistance(float depth, vec4 projection)
{
    if (projection.x == 0.0) return depth * projection.y;
    return projection.x * projection.y / (projection.y - depth * (projection.y - projection.x));
}
vec2 CappedFootprint(vec2 radius, ivec2 extent, float requestedLimit)
{
    vec2 texels = radius * vec2(extent);
    float largest = max(texels.x, texels.y);
    float limit = min(requestedLimit, float(min(extent.x, extent.y)) * MaximumMapFraction);
    // A uniform scale preserves non-square projection/texture ellipses.
    return largest > limit ? radius * (limit / largest) : radius;
}
float VisibilityTexel(uint map, ivec2 pixel, float receiver)
{
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
float PcssVisibility(uint map, vec4 clip, float bias)
{
    if ((Pcss.SamplesAndMaps.z & (1u << map)) == 0u || clip.w <= PcssProjectionEpsilon) return 1.0;
    vec3 ndc = clip.xyz / clip.w;
    // Native Vulkan depth producers flip clip Y for a positive-height viewport.
    // The resulting image and this kernel both use top-left coordinates.
    vec2 uv = ndc.xy * vec2(0.5, -0.5) + 0.5;
    if (ndc.z < 0.0 || ndc.z > 1.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    ivec2 extent = ShadowExtent(map);
    vec4 projection = Pcss.ProjectionMetrics[map];
    bool perspective = projection.x > 0.0;
    float emitter = perspective ? Pcss.EmitterAndLimits.x : Pcss.EmitterAndLimits.y;
    float receiver = ndc.z - bias;
    if (emitter == 0.0)
        return VisibilityTexel(map, min(ivec2(uv * vec2(extent)), extent - 1), receiver);
    float receiverDistance = LinearLightDistance(ndc.z, projection);
    if (receiverDistance <= 0.0) return 1.0;
    float searchWidth = perspective ? emitter * max(receiverDistance - projection.x, 0.0) / receiverDistance
        : emitter * receiverDistance;
    vec2 searchRadius = CappedFootprint(searchWidth / projection.zw, extent, Pcss.EmitterAndLimits.z);
    float blockerDistance = 0.0;
    uint blockers = 0u;
    for (uint index = 0u; index < Pcss.SamplesAndMaps.x; ++index)
    {
        ivec2 pixel = ivec2(floor((uv + DiskSample(index, Pcss.SamplesAndMaps.x) * searchRadius) * vec2(extent)));
        if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, extent))) continue;
        float depth = ShadowDepthAt(map, pixel);
        if (depth < receiver)
        {
            blockerDistance += LinearLightDistance(depth, projection);
            ++blockers;
        }
    }
    if (blockers == 0u) return 1.0;
    blockerDistance /= float(blockers);
    float separation = max(receiverDistance - blockerDistance, 0.0);
    // Perspective widths are measured at the near plane. An orthographic
    // directional emitter uses tan(angular half-angle) times the separation.
    float filterWidth = emitter * separation;
    if (perspective) filterWidth *= projection.x / (blockerDistance * receiverDistance);
    vec2 filterRadius = CappedFootprint(filterWidth / projection.zw, extent, Pcss.EmitterAndLimits.w);
    float visible = 0.0;
    for (uint index = 0u; index < Pcss.SamplesAndMaps.y; ++index)
        visible += ComparisonBilinear(map, uv + DiskSample(index, Pcss.SamplesAndMaps.y) * filterRadius, receiver);
    return visible / float(Pcss.SamplesAndMaps.y);
}

#else
float FixedDepth(uint map, ivec2 pixel)
{
    pixel = clamp(pixel, ivec2(0), ShadowExtent(map) - 1);
    if (map == 0u) return texelFetch(PrimaryImage, pixel, 0).r;
    if (map == 1u) return texelFetch(CascadeImage0, pixel, 0).r;
    if (map == 2u) return texelFetch(CascadeImage1, pixel, 0).r;
    if (map == 3u) return texelFetch(CascadeImage2, pixel, 0).r;
    return texelFetch(CascadeImage3, pixel, 0).r;
}
float FixedCompare(uint map, ivec2 pixel, float receiver)
{ return receiver <= FixedDepth(map, pixel) ? 1.0 : 0.0; }
float PcssVisibility(uint map, vec4 clip, float bias)
{
    const float PixelCenter = 0.5;
    if (clip.w <= HomogeneousEpsilon) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = vec2(projected.x + 1.0, 1.0 - projected.y) * 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec2 extent = vec2(ShadowExtent(map)); float receiver = projected.z - bias;
    if (Csm.Filtering.x == 0.0) return FixedCompare(map, ivec2(uv * extent), receiver);
    vec2 texel = uv * extent - PixelCenter, fraction = fract(texel);
    ivec2 base = ivec2(floor(texel)); float sum = 0.0;
    for (int row = -int(Csm.Filtering.x); row <= int(Csm.Filtering.x); ++row)
        for (int column = -int(Csm.Filtering.x); column <= int(Csm.Filtering.x); ++column)
        {
            ivec2 cell = base + ivec2(column, row);
            float top = mix(FixedCompare(map, cell, receiver), FixedCompare(map, cell + ivec2(1, 0), receiver), fraction.x);
            float bottom = mix(FixedCompare(map, cell + ivec2(0, 1), receiver), FixedCompare(map, cell + 1, receiver), fraction.x);
            sum += mix(top, bottom, fraction.y);
        }
    return sum / Csm.Filtering.y;
}
#endif
float CascadePCSS(int split, vec3 world, float bias)
{
    vec4 clip = Csm.LightMatrices[split] * vec4(world, 1.0);
    uint map = uint(split + 1);
    return PcssVisibility(map, clip, bias);
}
// The virtual hierarchy shares set 1 with CSM. FarDepths packs logical extent;
// BlendDepths packs physical extent. Normalized directory entries decode exact
// integer slots, and every comparison corner performs its own coarse fallback.
const uint VirtualPageGutter = 1u;
const float VirtualTexelOrigin = 0.5;
vec2 VirtualDepthAt(vec2 coordinate, uint firstLevel)
{
    uvec4 hierarchy = uvec4(Csm.FarDepths), atlas = uvec4(Csm.BlendDepths);
    uint pages = hierarchy.x / hierarchy.y, start = 0u;
    for (uint level = 0u; level < hierarchy.z; ++level)
    {
        if (level >= firstLevel)
        {
            uint resolution = pages * hierarchy.y;
            uvec2 pixel = min(uvec2(coordinate * float(resolution)), uvec2(resolution - 1u));
            uvec2 page = pixel / hierarchy.y;
            uint index = start + page.y * pages + page.x;
            ivec2 directoryPixel = ivec2(index % hierarchy.w, index / hierarchy.w);
            uint resident = uint(round(texelFetch(sampler2D(CsmDepth1, CsmNearest), directoryPixel, 0).r * float(atlas.w)));
            if (resident > 0u)
            {
                uint slot = resident - 1u;
                uvec2 tile = uvec2(slot % atlas.z, slot / atlas.z) * (hierarchy.y + 2u * VirtualPageGutter);
                ivec2 source = ivec2(tile + VirtualPageGutter + pixel % hierarchy.y);
                return vec2(texelFetch(sampler2D(CsmDepth0, CsmNearest), source, 0).r, float(level));
            }
        }
        start += pages * pages; pages /= 2u;
    }
    return vec2(1.0, float(hierarchy.z - 1u));
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
    float resolution = float(uint(Csm.FarDepths.x) >> actualMip);
    vec2 pixel = clamp(floor(sampleUv * resolution), vec2(0.0, 0.0), vec2(resolution - 1.0, resolution - 1.0));
    vec2 center = (pixel + vec2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualCompareAt(ivec2 pixel, uint mip, vec2 receiverUv, float receiver, vec2 gradient)
{
    uint resolution = uint(Csm.FarDepths.x) >> mip;
    vec2 uv = (vec2(clamp(pixel, ivec2(0, 0), ivec2(int(resolution) - 1, int(resolution) - 1))) + vec2(0.5, 0.5)) / float(resolution);
    vec2 stored = VirtualDepthAt(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
float VirtualVisibility(vec3 position, float bias, vec3 receiverData)
{
    vec4 light = Csm.LightMatrices[0] * vec4(position, 1.0);
    vec3 projected = light.xyz / max(light.w, Csm.EyeDepthPlane.w);
    vec2 uv = vec2(projected.x, -projected.y) * 0.5 + vec2(0.5, 0.5);
    vec2 gradient = receiverData.xy;
    uint mip = uint(receiverData.z);
    if (light.w <= Csm.EyeDepthPlane.w || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec2 stored = VirtualDepthAt(uv, mip);
    float receiver = projected.z - bias;
    if (Csm.Filtering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    vec2 texel = uv * float(uint(Csm.FarDepths.x) >> mip) - vec2(0.5, 0.5);
    ivec2 first = ivec2(floor(texel)); vec2 weight = fract(texel); float visible = 0.0;
    for (int y = -int(Csm.Filtering.x); y <= int(Csm.Filtering.x); ++y)
        for (int x = -int(Csm.Filtering.x); x <= int(Csm.Filtering.x); ++x)
        {
            ivec2 p = first + ivec2(x, y);
            float a = VirtualCompareAt(p, mip, uv, receiver, gradient);
            float b = VirtualCompareAt(p + ivec2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualCompareAt(p + ivec2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualCompareAt(p + ivec2(1, 1), mip, uv, receiver, gradient);
            visible += mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
        }
    return visible / Csm.Filtering.y;
}



// Resolve the receiver plane before entering a varying light-list loop.
vec3 VirtualReceiverData(vec3 world)
{
    if (Csm.Filtering.z == 0.0) return vec3(0.0, 0.0, 0.0);
    vec4 clip = Csm.LightMatrices[0] * vec4(world, 1.0);
    vec2 uv = clip.xy / max(clip.w, Csm.EyeDepthPlane.w) * vec2(0.5, -0.5);
    float size = max(length(dFdx(uv)) * Csm.EyeDepthPlane.y, length(dFdy(uv)) * Csm.EyeDepthPlane.z) * Csm.FarDepths.x;
    return vec3(VirtualReceiverGradient(uv, clip.z / max(clip.w, Csm.EyeDepthPlane.w)), clamp(floor(log2(max(1.0, size)) + Csm.EyeDepthPlane.x), 0.0, Csm.FarDepths.z - 1.0));
}

float SurfaceVisibility(vec3 world, vec3 normal, vec3 towardLight, vec4 singleClip, bool enabled, vec3 receiverData)
{
    if (enabled && ShadowRayEnabled()) return 1.0-ShadowStrength*(1.0-ShadowRayVisibility(world,normal,towardLight));
    if (!enabled) return 1.0;
    float cosine = clamp(dot(normal, NormalizeOr(towardLight, normal)), 0.0, 1.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumShadowCosine);
    float bias = min(ReceiverBias + ReceiverSlopeBias * slope, MaximumReceiverBias);
    if (Csm.Filtering.z != 0.0) return 1.0 - ShadowStrength * (1.0 - VirtualVisibility(world, bias, receiverData));
    float visibility;
    int count = int(Csm.Parameters.z);
    if (count > 0)
    {
        float eyeDepth = dot(Csm.EyeDepthPlane, vec4(world, 1.0));
        if (eyeDepth < Csm.Parameters.x || eyeDepth > Csm.Parameters.y) return 1.0;
        int split = 0;
        while (split + 1 < count && eyeDepth > Csm.FarDepths[split]) ++split;
        visibility = CascadePCSS(split, world, bias);
        float overlap = Csm.FarDepths[split] - Csm.BlendDepths[split];
        if (overlap > DistanceEpsilon && eyeDepth > Csm.BlendDepths[split])
        {
            float neighbor = split + 1 < count ? CascadePCSS(split + 1, world, bias) : 1.0;
            visibility = mix(visibility, neighbor, clamp((eyeDepth - Csm.BlendDepths[split]) / overlap, 0.0, 1.0));
        }
    }
    else
    {
        visibility = PcssVisibility(0u, singleClip, bias);
    }
    return 1.0 - ShadowStrength * (1.0 - visibility);
}
float ShadowVisibility(vec3 normal, vec3 lightVector, vec3 receiverData)
{
    return SurfaceVisibility(WorldPosition, normal, lightVector, LightPosition, Draw.RenderParameters.x >= OverlayMode, receiverData);
}
const float Pi = 3.141592653589793;
const float MinimumRoughness = 0.045;
const float DistributionEpsilon = 1.0e-12;
const float BrdfEpsilon = 1.0e-6;
const float DielectricFresnel = 0.04;
const float SchlickExponent = 5.0;
const float MicrofacetNormalization = 4.0;
void EvaluatePBR(vec2 roughnessMetallic, vec3 lightVector, vec3 radiance, float visibility,
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

void EvaluateToon(vec3 lightVector, vec3 radiance, float visibility,
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
void EvaluatePhong(vec3 lightVector, vec3 radiance, float visibility,
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
void EvaluateSample(vec2 roughnessMetallic, vec3 lightVector, vec3 radiance, float visibility,
    vec3 normal, vec3 viewDirection, vec3 albedo, inout vec3 diffuse, inout vec3 specular)
{
    const uint PhysicalModel = 3u, ToonModel = 4u;
    if (Draw.TextureFlags.w == PhysicalModel) EvaluatePBR(roughnessMetallic, lightVector, radiance, visibility, normal, viewDirection, albedo, diffuse, specular);
    else if (Draw.TextureFlags.w == ToonModel) EvaluateToon(lightVector, radiance, visibility, normal, viewDirection, albedo, diffuse, specular);
    else EvaluatePhong(lightVector, radiance, visibility, normal, viewDirection, albedo, diffuse, specular);
}
void EvaluateSceneLight(vec2 roughnessMetallic, int index, vec3 position, vec3 normal, vec3 viewDirection,
    vec3 albedo, float visibility, inout vec3 diffuse, inout vec3 specular)
{
    SceneLight light = SceneEmitters[index];
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

vec3 SurfaceNormal()
{
    vec3 normal = NormalizeOr(WorldNormal, vec3(0.0, 1.0, 0.0));
    const uint FlatModel = 1u;
    if (Draw.TextureFlags.w == FlatModel)
    {
        vec3 plane = NormalizeOr(cross(dFdx(WorldPosition), dFdy(WorldPosition)), normal);
        if (dot(plane, normal) < 0.0) plane = -plane;
        return gl_FrontFacing ? plane : -plane;
    }
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
uvec2 LightSelection(out bool indexed)
{
    uvec2 span = uvec2(0u, Draw.LightLayout.x); indexed = false;
    if (Draw.LightLayout.z != 0u)
    {
        uvec2 pixel = min(uvec2(gl_FragCoord.xy), Draw.SurfaceTiles.zw - 1u);
        uvec2 tile = pixel / Draw.SurfaceTiles.x;
        uvec4 header = MaterialTiles[tile.y * Draw.SurfaceTiles.y + tile.x];
        uvec2 selected = Draw.LightLayout.z == 1u ? header.xy : header.zw;
        if (selected.y <= Draw.LightLayout.w) { span = selected; indexed = true; }
    }
    return span;
}
#ifdef EVANS_FORWARD_PLUS_DEPTH
void main()
{
    float alpha = VertexColor.a * Draw.MaterialParameters.y;
    if (Draw.TextureFlags.x != 0u) alpha *= texture(sampler2D(DiffuseImage, SurfaceSampler), TexCoord).a;
    if (Draw.TextureFlags.z != 0u) alpha *= texture(sampler2D(OpacityImage, SurfaceSampler), TexCoord).r;
    if (Draw.MaterialParameters.z > 0.0 && clamp(alpha, 0.0, 1.0) < Draw.MaterialParameters.z) discard;
}
#else
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
    vec3 diffuse = vec3(0.0), specular = vec3(0.0);
    bool indexed; uvec2 span = LightSelection(indexed);
    vec3 virtualReceiver = VirtualReceiverData(WorldPosition);
    for (uint entry = 0u; entry < span.y; ++entry)
    {
        uint index = indexed ? MaterialIndices[span.x + entry] : entry;
        SceneLight light = SceneEmitters[index];
        vec3 shadowVector = int(light.PositionType.w + 0.5) == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - WorldPosition;
        float visibility = index == Draw.LightLayout.y && light.ColorCastShadows.w > OverlayMode ? ComposeContact(ShadowVisibility(normal, shadowVector, virtualReceiver)) : 1.0;
        EvaluateSceneLight(roughnessMetallic, int(index), WorldPosition, normal, viewDirection, albedo, visibility, diffuse, specular);
    }
    const uint PhysicalModel = 3u;
    vec3 ambient = ToonAmbient(base * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), base * max(Draw.MaterialAmbientColor.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.w, 0.0), float(Draw.TextureFlags.w));
    if (Draw.TextureFlags.w == PhysicalModel) ambient *= 1.0 - clamp(roughnessMetallic.y, 0.0, 1.0);
    else specular *= max(Draw.MaterialSpecularColor.rgb, vec3(0.0));
    vec3 color = ambient + albedo * diffuse + specular + max(Draw.MaterialEmissiveColor.rgb, vec3(0.0));
    color = max(color, vec3(0.0));
    color += albedo * (Draw.TextureFlags.w==PhysicalModel ? (1-LumenDielectricReflectance)*(1.0-clamp(roughnessMetallic.y,0.0,1.0)) : 1.0) * LumenTranslucentIrradiance(WorldPosition,normal);
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
#endif
#else
// Independently authored Vulkan 1.0 Forward+ light assignment.
// Harada, McKee and Yang, "Forward+: Bringing Deferred Lighting to the Next
// Level" (2012): https://takahiroharada.github.io/forward%2B/
// A two-level blocked prefix sum compacts membership in scene-light order.
// This uses workgroup memory/barriers, without assuming a subgroup width.
// The depth image and tile grid both use top-left pixels; CameraRows describe
// the canonical increasing [0,1] projection before raster clip-Y inversion.
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
const uint GroupSize = 64u;
const uint BitsPerWord = 32u;
const uint LightLimit = 4096u;
const uint MembershipWords = LightLimit / BitsPerWord;
const uint WordsPerLane = MembershipWords / GroupSize;
const uint ScanBlockSize = 8u;
const uint ScanBlocks = GroupSize / ScanBlockSize;
const uint SidePlanes = 4u;
const uint NearOpaque = SidePlanes, FarOpaque = NearOpaque + 1u;
const uint NearTransparent = FarOpaque + 1u, FarTransparent = NearTransparent + 1u;
const uint PlaneCount = FarTransparent + 1u;
const float MachineEpsilon = 1.1920928955078125e-7;
const float RoundingOperations = 16.0;
const float RelativeAllowance = RoundingOperations * MachineEpsilon;
const float MinimumNormal = 1.1754943508222875e-38;
const float UnderflowOperations = 64.0;
const float UnderflowAllowance = UnderflowOperations * MinimumNormal;
const uint SignBit = 0x80000000u;

layout(push_constant) uniform GridSnapshot
{
    vec4 CameraRows[4];
    uvec4 ExtentAndLights; // width, height, original samples, light count
    uvec4 TileLayout; // tile side, list capacity, columns, rows
};
struct LightBound { vec4 CenterRadius; uvec4 Flags; };
layout(std430, set = 0, binding = 0) readonly buffer SupportBounds { LightBound Lights[]; };
layout(std430, set = 0, binding = 2) writeonly buffer TileHeaders { uvec4 Headers[]; };
layout(std430, set = 0, binding = 3) writeonly buffer TileIndices { uint Indices[]; };
#ifdef EVANS_MULTISAMPLE
layout(set = 0, binding = 1) uniform sampler2DMS Depth;
#else
layout(set = 0, binding = 1) uniform sampler2D Depth;
#endif
shared uint DepthMinimum, DepthMaximum;
shared vec4 Planes[PlaneCount], Errors[PlaneCount];
shared uvec2 Membership[MembershipWords], LaneOffsets[GroupSize], BlockOffsets[ScanBlocks];
shared uvec2 TotalCount;

float MaxComponent(vec4 value) { return max(max(value.x, value.y), max(value.z, value.w)); }
void Plane(uint index, uint axis, float edge, float orientation)
{
    float scale = max(MinimumNormal, max(MaxComponent(abs(CameraRows[axis])), MaxComponent(abs(CameraRows[3]))));
    vec4 axisRow = CameraRows[axis] / scale, wRow = CameraRows[3] / scale;
    vec4 edgeRow = edge * wRow;
    Planes[index] = orientation * (axisRow - edgeRow);
    // Preserve the magnitudes BEFORE cancellation, including NDC edge error.
    Errors[index] = RelativeAllowance * (abs(axisRow) + abs(edgeRow) + abs(wRow));
}
bool Overlaps(uint index, vec4 center, float radius)
{
    vec4 product = Planes[index] * center;
    float distance = (product.x + product.y) + (product.z + product.w);
    float support = radius * length(Planes[index].xyz);
    float uncertainty = dot(Errors[index], abs(center)) + radius * length(Errors[index].xyz);
    uncertainty += RelativeAllowance * (dot(abs(product), vec4(1.0)) + support) + UnderflowAllowance;
    return distance + support >= -uncertainty;
}
bvec2 Classify(uint light)
{
    if (Lights[light].Flags.x != 0u) return bvec2(true);
    vec4 sphere = Lights[light].CenterRadius;
    float scale = max(1.0, MaxComponent(abs(sphere)));
    vec4 center = vec4(sphere.xyz, 1.0) / scale;
    float radius = sphere.w / scale;
    for (uint side = 0u; side < SidePlanes; ++side)
        if (!Overlaps(side, center, radius)) return bvec2(false);
    return bvec2(Overlaps(NearOpaque, center, radius) && Overlaps(FarOpaque, center, radius),
        Overlaps(NearTransparent, center, radius) && Overlaps(FarTransparent, center, radius));
}
void Scatter(uint bits, uint word, uint destination, uint ordinal)
{
    while (bits != 0u)
    {
        uint first = uint(findLSB(bits));
        if (ordinal < TileLayout.y) Indices[destination + ordinal] = word * BitsPerWord + first;
        bits &= bits - 1u; ++ordinal;
    }
}
void main()
{
    uint lane = gl_LocalInvocationIndex;
    uvec2 tile = gl_WorkGroupID.xy;
    if (lane == 0u) { DepthMinimum = floatBitsToUint(1.0); DepthMaximum = 0u; }
    for (uint word = lane; word < MembershipWords; word += GroupSize) Membership[word] = uvec2(0u);
    barrier();
    for (uint pixelIndex = lane; pixelIndex < TileLayout.x * TileLayout.x; pixelIndex += GroupSize)
    {
        uvec2 pixel = tile * TileLayout.x + uvec2(pixelIndex % TileLayout.x, pixelIndex / TileLayout.x);
        if (any(greaterThanEqual(pixel, ExtentAndLights.xy))) continue;
        for (uint sampleIndex = 0u; sampleIndex < ExtentAndLights.z; ++sampleIndex)
        {
#ifdef EVANS_MULTISAMPLE
            float z = texelFetch(Depth, ivec2(pixel), int(sampleIndex)).r;
#else
            float z = texelFetch(Depth, ivec2(pixel), 0).r;
#endif
            uint bits = floatBitsToUint(z); if (bits == SignBit) bits = 0u;
            // Positive [0,1] IEEE encodings preserve order. Invalid values widen
            // the interval, while clear depth is retained as a possible receiver.
            bool valid = bits <= floatBitsToUint(1.0);
            atomicMin(DepthMinimum, valid ? bits : 0u);
            atomicMax(DepthMaximum, valid ? bits : floatBitsToUint(1.0));
        }
    }
    barrier();
    vec2 lower = vec2(tile * TileLayout.x) * 2.0 / vec2(ExtentAndLights.xy) - 1.0;
    vec2 upper = vec2(min((tile + 1u) * TileLayout.x, ExtentAndLights.xy)) * 2.0 / vec2(ExtentAndLights.xy) - 1.0;
    if (lane == 0u) Plane(lane, 0u, lower.x, 1.0);
    if (lane == 1u) Plane(lane, 0u, upper.x, -1.0);
    if (lane == 2u) Plane(lane, 1u, -lower.y, -1.0);
    if (lane == 3u) Plane(lane, 1u, -upper.y, 1.0);
    if (lane == NearOpaque) Plane(lane, 2u, uintBitsToFloat(DepthMinimum), 1.0);
    if (lane == FarOpaque) Plane(lane, 2u, uintBitsToFloat(DepthMaximum), -1.0);
    if (lane == NearTransparent) Plane(lane, 2u, 0.0, 1.0);
    if (lane == FarTransparent) Plane(lane, 2u, 1.0, -1.0);
    barrier();
    for (uint light = lane; light < ExtentAndLights.w; light += GroupSize)
    {
        bvec2 keep = Classify(light);
        uint word = light / BitsPerWord, bit = 1u << (light % BitsPerWord);
        if (keep.x) atomicOr(Membership[word].x, bit);
        if (keep.y) atomicOr(Membership[word].y, bit);
    }
    barrier();
    uvec2 firstCount = uvec2(bitCount(Membership[lane * WordsPerLane]));
    LaneOffsets[lane] = firstCount + uvec2(bitCount(Membership[lane * WordsPerLane + 1u]));
    barrier();
    // Eight leaders independently scan eight lane counts each. A final scan of
    // the eight block totals supplies the common offset for each block: O(W)
    // work and four barriers, with no atomic append order or subgroup contract.
    if (lane < ScanBlocks)
    {
        uvec2 total = uvec2(0u);
        for (uint element = 0u; element < ScanBlockSize; ++element)
        {
            uint index = lane * ScanBlockSize + element;
            uvec2 count = LaneOffsets[index]; LaneOffsets[index] = total; total += count;
        }
        BlockOffsets[lane] = total;
    }
    barrier();
    if (lane == 0u)
    {
        uvec2 total = uvec2(0u);
        for (uint block = 0u; block < ScanBlocks; ++block)
        { uvec2 count = BlockOffsets[block]; BlockOffsets[block] = total; total += count; }
        TotalCount = total;
    }
    barrier();
    uvec2 ordinal = LaneOffsets[lane] + BlockOffsets[lane / ScanBlockSize];
    uint tileIndex = tile.y * TileLayout.z + tile.x;
    uint opaqueBase = tileIndex * 2u * TileLayout.y, transparentBase = opaqueBase + TileLayout.y;
    for (uint offset = 0u; offset < WordsPerLane; ++offset)
    {
        uint word = lane * WordsPerLane + offset;
        Scatter(Membership[word].x, word, opaqueBase, ordinal.x);
        Scatter(Membership[word].y, word, transparentBase, ordinal.y);
        ordinal += uvec2(bitCount(Membership[word]));
    }
    // Never clamp these totals: overflow consumers must evaluate ALL lights.
    if (lane == 0u) Headers[tileIndex] = uvec4(opaqueBase, TotalCount.x, transparentBase, TotalCount.y);
}

#endif
