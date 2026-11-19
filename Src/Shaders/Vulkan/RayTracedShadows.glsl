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
const float ShadowRaySurfaceStrength=0.55;

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

// Native ray-shadow receivers preserve the established material/lighting ABI.
// Visibility comes exclusively from world-space triangle queries.
#ifndef EVANS_RAY_SHADOW_PROGRAM
#define EVANS_RAY_SHADOW_PROGRAM 0
#endif
#if EVANS_RAY_SHADOW_PROGRAM == 6
#define EVANS_MULTISAMPLE 1
#endif
#ifdef EVANS_VERTEX_STAGE
layout(location = 0) out vec2 ScreenUV;
void main()
{
    const float FullscreenEdge = 3.0;
    vec2 corner = gl_VertexIndex == 1 ? vec2(FullscreenEdge, -1.0) :
        gl_VertexIndex == 2 ? vec2(-1.0, FullscreenEdge) : vec2(-1.0);
    ScreenUV = corner * 0.5 + 0.5;
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
const int PCSSMapCount = 5; // One single map and four cascade intervals.
struct ReservedReceiverData
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
#if EVANS_RAY_SHADOW_PROGRAM < 5
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
    ReservedReceiverData PCSS;
} Draw;

#define Pcss Draw.PCSS
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

#if EVANS_RAY_SHADOW_PROGRAM == 2
layout(location = 5) in vec3 DiffuseFront;
layout(location = 6) in vec3 SpecularFront;
layout(location = 7) in vec3 DiffuseBack;
layout(location = 8) in vec3 SpecularBack;
layout(location = 9) in vec3 ShadowDiffuseFront;
layout(location = 10) in vec3 ShadowSpecularFront;
layout(location = 11) in vec3 ShadowDiffuseBack;
layout(location = 12) in vec3 ShadowSpecularBack;
#endif
#elif EVANS_RAY_SHADOW_PROGRAM < 7
layout(location = 0) in vec2 ScreenUV;
layout(location = 0) out vec4 OutRadiance;
layout(set = 0, binding = 0, std140, row_major) uniform LightingConstants
{
    mat4 InverseViewProjection;
    mat4 LightViewProjection;
    vec4 CameraLightCount;
    vec4 ShadingParameters;
    uvec4 Options;
    vec4 BackgroundColor;
    SceneLight Lights[MaximumLights];
    ReservedReceiverData PCSS;
} Lighting;
#if defined(EVANS_MULTISAMPLE)
layout(set = 0, binding = 1) uniform texture2DMS DiffuseOpacityImage;
layout(set = 0, binding = 2) uniform texture2DMS NormalRoughnessImage;
layout(set = 0, binding = 3) uniform texture2DMS SpecularShininessImage;
layout(set = 0, binding = 4) uniform texture2DMS AmbientMetallicImage;
layout(set = 0, binding = 5) uniform texture2DMS EmissionImage;
layout(set = 0, binding = 6) uniform texture2DMS SceneDepth;
#else
layout(set = 0, binding = 1) uniform texture2D DiffuseOpacityImage;
layout(set = 0, binding = 2) uniform texture2D NormalRoughnessImage;
layout(set = 0, binding = 3) uniform texture2D SpecularShininessImage;
layout(set = 0, binding = 4) uniform texture2D AmbientMetallicImage;
layout(set = 0, binding = 5) uniform texture2D EmissionImage;
layout(set = 0, binding = 6) uniform texture2D SceneDepth;
#endif
layout(set = 0, binding = 7) uniform texture2D ShadowDepth;
layout(set = 0, binding = 8) uniform sampler NearestSampler;
#define Pcss Lighting.PCSS
#else
const uint SceneLightLimit = 4u;
const uint ShadowCascadeLimit = 4u;
struct LightRecord
{
    vec4 PositionKind;
    vec4 DirectionStrength;
    vec4 ColorCastsShadow;
    vec4 RectangleExtent;
};
// The host packs the 128-byte medium followed by the 704-byte lighting block.
// Each pass owns a distinct dynamic UBO range and immutable descriptor set.
layout(set = 0, binding = 0, std140, row_major) uniform TransportParameters
{
    mat4 InverseCamera;
    vec4 DensityHeightRange;
    vec4 AmbientRadiance;
    vec4 ViewportCells;
    uvec4 MediumControl;
    vec4 AlbedoPhase;
    vec4 IntensityBias;
    uvec4 LightControl;
    mat4 PrimaryShadowMatrix;
    LightRecord SceneLights[SceneLightLimit];
    mat4 CascadeMatrices[ShadowCascadeLimit];
    vec4 CascadeFarPlanes;
    vec4 CascadeBlendPlanes;
    vec4 EyeDepthPlane;
    vec4 CascadeParameters;
    vec4 ShadowFilter;
    ReservedReceiverData PCSS;
} Parameters;
layout(set = 0, binding = 1) uniform sampler2D PreviousTransport;
// Full-scene storage preserves lights before the opaque depth interval.
layout(set = 0, binding = 7, std430) readonly buffer VolumeSceneStorage
{ LightRecord VolumeSceneLights[]; };
layout(set = 0, binding = 2) uniform sampler2D PrimaryShadowDepth;
layout(set = 0, binding = 3) uniform sampler2D FirstCascadeDepth;
layout(set = 0, binding = 4) uniform sampler2D SecondCascadeDepth;
layout(set = 0, binding = 5) uniform sampler2D ThirdCascadeDepth;
layout(set = 0, binding = 6) uniform sampler2D FourthCascadeDepth;
layout(location = 0) out vec4 IntegratedTransport;

#define Pcss Parameters.PCSS
#endif
#if EVANS_RAY_SHADOW_PROGRAM < 7
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
#if EVANS_RAY_SHADOW_PROGRAM < 5
#define PrimaryImage sampler2D(ShadowDepth, SurfaceSampler)
#else
#define PrimaryImage sampler2D(ShadowDepth, NearestSampler)
#endif
#define CascadeImage0 sampler2D(CsmDepth0, CsmNearest)
#define CascadeImage1 sampler2D(CsmDepth1, CsmNearest)
#define CascadeImage2 sampler2D(CsmDepth2, CsmNearest)
#define CascadeImage3 sampler2D(CsmDepth3, CsmNearest)
#else
#define PrimaryImage PrimaryShadowDepth
#define CascadeImage0 FirstCascadeDepth
#define CascadeImage1 SecondCascadeDepth
#define CascadeImage2 ThirdCascadeDepth
#define CascadeImage3 FourthCascadeDepth
#endif
#if defined(EVANS_CONTACT_RECEIVER)
#if EVANS_RAY_SHADOW_PROGRAM < 5
#define CONTACT_BINDING 6
#define CONTACT_SAMPLER SurfaceSampler
#else
#define CONTACT_BINDING 9
#define CONTACT_SAMPLER NearestSampler
#endif
#endif

// Sample-frequency contact lookup must not move the existing pixel-centred
// material/alpha evaluation. In particular, cutouts must match the depth pass.
#if defined(EVANS_CONTACT_RECEIVER) && defined(EVANS_MULTISAMPLE) && EVANS_RAY_SHADOW_PROGRAM < 5
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

// Blockers are averaged in linear light distance, not projected depth. Bias
// classifies blockers/comparisons but does not change the penumbra geometry.
// The disk quadrature is deterministic; both loops have independent GUI budgets.
const float PcssPi = 3.14159265358979323846;
const float GoldenAngle = PcssPi * (3.0 - sqrt(5.0));
const float PcssProjectionEpsilon = 1.0e-7;
const float MaximumMapFraction = 0.25;

#if EVANS_RAY_SHADOW_PROGRAM < 7
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


float SurfaceVisibility(vec3 world, vec3 normal, vec3 towardLight, vec4 singleClip, bool enabled)
{
    if (!enabled) return 1.0;
    return 1.0-ShadowRaySurfaceStrength*(1.0-ShadowRayVisibility(world,normal,towardLight));
}
#endif
#if EVANS_RAY_SHADOW_PROGRAM < 5
float ShadowVisibility(vec3 normal, vec3 lightVector)
{
    return SurfaceVisibility(WorldPosition, normal, lightVector, LightPosition, Draw.RenderParameters.x >= OverlayMode);
}
#if EVANS_RAY_SHADOW_PROGRAM == 3
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

#elif EVANS_RAY_SHADOW_PROGRAM == 4
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

#if EVANS_RAY_SHADOW_PROGRAM == 1
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
#if EVANS_RAY_SHADOW_PROGRAM == 2
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
#if EVANS_RAY_SHADOW_PROGRAM == 3
    vec3 color = ToonAmbient(base * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), base * max(Draw.MaterialAmbientColor.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.w, 0.0), float(Draw.TextureFlags.w)) * (1.0 - clamp(roughnessMetallic.y, 0.0, 1.0))
        + albedo * diffuse + specular + max(Draw.MaterialEmissiveColor.rgb, vec3(0.0));
#else
    vec3 color = ToonAmbient(base * max(Draw.MaterialDiffuseColor.rgb, vec3(0.0)), base * max(Draw.MaterialAmbientColor.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.rgb, vec3(0.0)) * max(Draw.AmbientColorStrength.w, 0.0), float(Draw.TextureFlags.w))
        + albedo * diffuse + max(Draw.MaterialSpecularColor.rgb, vec3(0.0)) * specular + max(Draw.MaterialEmissiveColor.rgb, vec3(0.0));
#endif
    color = max(color, vec3(0.0));
    color += albedo * (EVANS_RAY_SHADOW_PROGRAM==3 ? (1-LumenDielectricReflectance)*(1.0-clamp(roughnessMetallic.y,0.0,1.0)) : 1.0) * LumenTranslucentIrradiance(WorldPosition,normal);
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
#elif EVANS_RAY_SHADOW_PROGRAM < 7
const uint PhysicalModel = 3u, ToonModel = 4u;
const float MinimumRoughness = 0.045;
const float HalfFloatMaximum = 65504.0;
const float ParallelAxisLimit = 0.999;
vec3 UnitDirection(vec3 value, vec3 fallback) { return NormalizeOr(value, fallback); }
struct Surface
{
    vec3 Position;
    vec3 Normal;
    vec3 View;
    vec3 Diffuse;
    vec3 Specular;
    float Roughness;
    float Shininess;
    float Metallic;
};
float ShadowVisibility(Surface surface, vec3 towardLight)
{
    return SurfaceVisibility(surface.Position, surface.Normal, towardLight,
        Lighting.LightViewProjection * vec4(surface.Position, 1.0), Lighting.Options.y != 0u);
}
// Separate direct diffuse before screen-space diffusion; ordinary lighting is unchanged.
const uint DirectDiffuseOutput = 1u;
bool DirectDiffuseOnly()
{
    return Lighting.Options.w == DirectDiffuseOutput;
}
vec3 ShadeSample(Surface surface, vec3 lightVector, vec3 radiance)
{
    float distanceSquared = dot(lightVector, lightVector);
    if (distanceSquared <= NormalizeEpsilon) return vec3(0.0);
    vec3 direction = lightVector * inversesqrt(distanceSquared);
    float nl = max(dot(surface.Normal, direction), 0.0);
    if (nl <= 0.0 && Lighting.Options.x != ToonModel) return vec3(0.0);
    vec3 halfway = UnitDirection(direction + surface.View, surface.Normal);
    float nh = clamp(dot(surface.Normal, halfway), 0.0, 1.0);
    if (Lighting.Options.x == PhysicalModel)
    {
        // GGX with separable exact Smith masking and Schlick Fresnel.
        // Walter et al., "Microfacet Models for Refraction through Rough Surfaces" (2007).
        const float Pi = 3.141592653589793;
        const float DielectricF0 = 0.04;
        const float FresnelExponent = 5.0;
        const float BrdfEpsilon = 1.0e-6;
        const float DistributionEpsilon = 1.0e-12;
        float nv = clamp(dot(surface.Normal, surface.View), 0.0, 1.0);
        if (nv <= 0.0) return vec3(0.0);
        float vh = clamp(dot(surface.View, halfway), 0.0, 1.0);
        float alpha = surface.Roughness * surface.Roughness;
        float alphaSquared = alpha * alpha;
        float denominator = nh * nh * (alphaSquared - 1.0) + 1.0;
        float distribution = alphaSquared / max(Pi * denominator * denominator, DistributionEpsilon);
        float smithView = 2.0 * nv / max(nv + sqrt(alphaSquared + (1.0 - alphaSquared) * nv * nv), BrdfEpsilon);
        float smithLight = 2.0 * nl / max(nl + sqrt(alphaSquared + (1.0 - alphaSquared) * nl * nl), BrdfEpsilon);
        vec3 f0 = mix(vec3(DielectricF0), clamp(surface.Diffuse, 0.0, 1.0), surface.Metallic);
        vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, FresnelExponent);
        vec3 diffuse = surface.Diffuse * (1.0 - fresnel) * (1.0 - surface.Metallic) / Pi;
        vec3 specular = distribution * smithView * smithLight * fresnel / max(4.0 * nv * nl, BrdfEpsilon);
        if (DirectDiffuseOnly()) return radiance * nl * diffuse;
        return radiance * nl * (diffuse + specular);
    }
    float highlight = pow(nh, surface.Shininess);
    if (Lighting.Options.x == ToonModel)
    {
        const float MinimumBands = 2.0;

        float bands = max(Lighting.ShadingParameters.x, MinimumBands);
        return radiance * (surface.Diffuse * ToonResponse(nl, bands) + (DirectDiffuseOnly() ? vec3(0.0) : surface.Specular * ToonHighlight(highlight)));
    }
    if (DirectDiffuseOnly()) return radiance * surface.Diffuse * nl;
    return radiance * (surface.Diffuse * nl + surface.Specular * highlight);
}
vec3 ShadeLight(Surface surface, int index)
{
    const int DirectionalLight = 0;
    const int PointLight = 1;
    const float MinimumDistanceSquared = 1.0e-6;
    const float EnabledThreshold = 0.5;
    SceneLight light = Lighting.Lights[index];
    int type = int(round(light.PositionType.w));
    vec3 toLight = type == DirectionalLight ? light.DirectionIntensity.xyz : light.PositionType.xyz - surface.Position;
    float visibility = uint(index) == Lighting.Options.z && light.ColorCastShadows.w > EnabledThreshold
        ? ComposeContact(ShadowVisibility(surface, toLight)) : 1.0;
    vec3 radiance = max(light.ColorCastShadows.rgb, vec3(0.0)) * max(light.DirectionIntensity.w, 0.0) * visibility;
    if (type == DirectionalLight) return ShadeSample(surface, toLight, radiance);
    if (type == PointLight) return ShadeSample(surface, toLight, radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / max(dot(toLight, toLight), MinimumDistanceSquared));
    // Prepared emitters integrate physical area; legacy lights retain normalized energy.
    const int SamplesPerAxis = 2;
    const float SampleOffset = 0.25;
    const float SampleWeight = 0.25;
    vec3 emissionDirection = UnitDirection(light.DirectionIntensity.xyz, vec3(0.0, -1.0, 0.0));
    vec3 helper = abs(emissionDirection.y) < ParallelAxisLimit ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = UnitDirection(cross(helper, emissionDirection), vec3(1.0, 0.0, 0.0));
    vec3 bitangent = UnitDirection(cross(emissionDirection, tangent), vec3(0.0, 0.0, 1.0));
    vec3 result = vec3(0.0);
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSizePadding.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaSizePadding.w, MinimumEmitterGrid, MaximumEmitterGrid)) : SamplesPerAxis;
    float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSizePadding.x * light.AreaSizePadding.y) : 1.0) / float(emitterGrid * emitterGrid);
    for (int y = 0; y < emitterGrid; ++y)
        for (int x = 0; x < emitterGrid; ++x)
        {
            vec2 offset = vec2(((float(x) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter), ((float(y) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter));
            vec3 position = light.PositionType.xyz + tangent * offset.x * light.AreaSizePadding.x
                + bitangent * offset.y * light.AreaSizePadding.y;
            vec3 displacement = position - surface.Position;
            float distanceSquared = max(dot(displacement, displacement), MinimumDistanceSquared);
            float emitterCosine = max(dot(emissionDirection, -displacement * inversesqrt(distanceSquared)), 0.0);
            result += ShadeSample(surface, displacement, radiance * emitterSampleWeight * emitterCosine * LightRangeWindow(displacement, light.AreaSizePadding.z) / distanceSquared);
        }
    return result;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#if defined(EVANS_MULTISAMPLE)
    // Sample-rate lighting reads one surface and reconstructs at its own raster
    // location; no fixed pattern or averaged normal is substituted at silhouettes.
    int materialSample = gl_SampleID;
    vec2 coordinate = (vec2(pixel) + gl_SamplePosition) / vec2(textureSize(sampler2DMS(SceneDepth, NearestSampler)));
#define MaterialSampler sampler2DMS
#else
    int materialSample = 0;
    vec2 coordinate = ScreenUV;
#define MaterialSampler sampler2D
#endif
    vec4 normal = texelFetch(MaterialSampler(NormalRoughnessImage, NearestSampler), pixel, materialSample);
    vec4 diffuse = texelFetch(MaterialSampler(DiffuseOpacityImage, NearestSampler), pixel, materialSample);
    vec4 emission = texelFetch(MaterialSampler(EmissionImage, NearestSampler), pixel, materialSample);
    // An unlit overlay can cover a pixel whose depth is still clear because
    // solid-wire rendering preserves the opaque depth buffer.
    if (normal.w < 0.0) { OutRadiance = DirectDiffuseOnly() ? vec4(0.0) : vec4(emission.rgb, diffuse.a); return; }
    const float ClearDepth = 1.0;
    float depth = texelFetch(MaterialSampler(SceneDepth, NearestSampler), pixel, materialSample).r;
    if (depth >= ClearDepth) { OutRadiance = DirectDiffuseOnly() ? vec4(0.0) : Lighting.BackgroundColor; return; }
    vec4 specular = texelFetch(MaterialSampler(SpecularShininessImage, NearestSampler), pixel, materialSample);
    vec4 ambient = texelFetch(MaterialSampler(AmbientMetallicImage, NearestSampler), pixel, materialSample);
    vec2 cameraNdc = vec2(coordinate.x * 2.0 - 1.0, 1.0 - coordinate.y * 2.0);
    vec4 world = Lighting.InverseViewProjection * vec4(cameraNdc, depth, 1.0);
    Surface surface;
    surface.Position = world.xyz / world.w;
    surface.Normal = UnitDirection(normal.xyz, vec3(0.0, 1.0, 0.0));
    surface.View = UnitDirection(Lighting.CameraLightCount.xyz - surface.Position, surface.Normal);
    surface.Diffuse = diffuse.rgb;
    surface.Specular = specular.rgb;
    surface.Roughness = max(normal.w, MinimumRoughness);
    surface.Shininess = max(specular.w, 1.0);
    surface.Metallic = clamp(ambient.w, 0.0, 1.0);
    vec3 radiance = DirectDiffuseOnly() ? vec3(0.0) : ambient.rgb + emission.rgb;
    int count = min(int(round(Lighting.CameraLightCount.w)), MaximumLights);
    for (int index = 0; index < MaximumLights; ++index)
    {
        if (index >= count) break;
        radiance += ShadeLight(surface, index);
    }
    float coverage = DirectDiffuseOnly() ? (Lighting.Options.x == PhysicalModel ? 1.0 - surface.Metallic : 1.0) : diffuse.a;
    if (DirectDiffuseOnly()) { radiance *= emission.a; coverage *= emission.a; }
    OutRadiance = vec4(clamp(radiance, 0.0, HalfFloatMaximum), coverage);
}
#else
const float Pi = 3.14159265358979323846;
const float OpticalDepthCeiling = 80.0;
const float SmallRayDistance = 1.0e-6;
const float SmallSquaredLightDistance = 1.0e-4;
const float SmallProjectiveW = 1.0e-7;
const float HalfFloatCeiling = 65504.0;
const float ParallelNormalThreshold = 0.99;
const uint DirectionalKind = 0u, PointKind = 1u;
const uint PrimaryShadowMode = 1u, CascadeShadowMode = 2u;
const int RectangleSamplesPerAxis = 2;
const float RectangleSampleWeight = 1.0 / float(RectangleSamplesPerAxis * RectangleSamplesPerAxis);

vec3 DirectionOr(vec3 value, vec3 fallback)
{
    float squaredLength = dot(value, value);
    return squaredLength > SmallSquaredLightDistance ? value * inversesqrt(squaredLength) : fallback;
}
vec3 CameraRayPosition(vec2 uv, float depth)
{
    // Positive-height viewports address images from the top left. Undo the
    // surface vertex shader's clip-Y flip before applying the canonical inverse.
    vec2 clipXY = vec2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
    vec4 homogeneous = Parameters.InverseCamera * vec4(clipXY, depth, 1.0);
    return homogeneous.xyz / homogeneous.w;
}
float PhaseProbability(vec3 towardEmitter, vec3 outwardCameraRay)
{
    float cosine = clamp(dot(towardEmitter, outwardCameraRay), -1.0, 1.0);
    float g = Parameters.AlbedoPhase.w;
    // Actual incoming and outgoing travel vectors are both negated here.
    // Positive g peaks at equal travel directions (cosine = +1).
    float denominator = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / (4.0 * Pi * denominator * sqrt(denominator));
}

float EmitterIntegral(LightRecord light, vec3 world, vec3 cameraRay)
{
    vec3 normal = DirectionOr(light.DirectionStrength.xyz, vec3(0.0, 1.0, 0.0));
    uint kind = uint(light.PositionKind.w);
    if (kind == DirectionalKind) return PhaseProbability(normal, cameraRay);
    if (kind == PointKind)
    {
        vec3 delta = light.PositionKind.xyz - world;
        return PhaseProbability(DirectionOr(delta, normal), cameraRay) * LightRangeWindow(delta, light.RectangleExtent.z) / max(dot(delta, delta), SmallSquaredLightDistance);
    }
    vec3 helper = abs(normal.y) < ParallelNormalThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
    float integral = 0.0;
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.RectangleExtent.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.RectangleExtent.w, MinimumEmitterGrid, MaximumEmitterGrid)) : RectangleSamplesPerAxis;
    float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
    for (int y = 0; y < emitterGrid; ++y)
        for (int x = 0; x < emitterGrid; ++x)
        {
            vec2 offset = ((vec2(x, y) + 0.5) / float(emitterGrid) - 0.5) * light.RectangleExtent.xy;
            vec3 delta = light.PositionKind.xyz + tangent * offset.x + bitangent * offset.y - world;
            vec3 towardEmitter = DirectionOr(delta, -normal);
            integral += emitterSampleWeight * clamp(dot(normal, -towardEmitter), 0.0, 1.0) *
                PhaseProbability(towardEmitter, cameraRay) * LightRangeWindow(delta, light.RectangleExtent.z) / max(dot(delta, delta), SmallSquaredLightDistance);
        }
    return integral * light.RectangleExtent.x * light.RectangleExtent.y;
}
vec3 ScatteringSource(vec3 world, vec3 cameraRay)
{

    vec3 irradiance = vec3(0.0);
    for (uint index = 0u; index < Parameters.LightControl.x; ++index)
    {
        LightRecord light;
        if (Parameters.IntensityBias.z != 0.0) light = VolumeSceneLights[index];
        else light = Parameters.SceneLights[index];
        vec3 radiance = min(light.ColorCastsShadow.rgb, vec3(HalfFloatCeiling)) * min(light.DirectionStrength.w, HalfFloatCeiling);
        vec3 shadowVector=uint(light.PositionKind.w)==0u?light.DirectionStrength.xyz:light.PositionKind.xyz-world;
        float visibility=Parameters.LightControl.y!=0u && index==Parameters.LightControl.z?ShadowRayVisibility(world,vec3(0.0),shadowVector):1.0;
        irradiance += radiance * (EmitterIntegral(light, world, cameraRay) * visibility);
    }
    return Parameters.AmbientRadiance.rgb + Parameters.AlbedoPhase.rgb * Parameters.IntensityBias.x * irradiance;
}
vec4 LocalSegment(ivec2 pixel)
{
    int columns = int(Parameters.ViewportCells.z), slice = pixel.x / columns;
    vec2 uv = (vec2(pixel.x % columns, pixel.y) + 0.5) / Parameters.ViewportCells.zw;
    vec3 nearPosition = CameraRayPosition(uv, 0.0);
    vec3 ray = CameraRayPosition(uv, 1.0) - nearPosition;
    float rayLength = max(length(ray), SmallRayDistance);
    vec3 direction = ray / rayLength;
    float interval = min(rayLength, Parameters.DensityHeightRange.w) /
        float(Parameters.MediumControl.y * Parameters.MediumControl.z);
    vec4 segment = vec4(0.0, 0.0, 0.0, 1.0);
    for (uint stepIndex = 0u; stepIndex < Parameters.MediumControl.z; ++stepIndex)
    {
        float distance = (float(uint(slice) * Parameters.MediumControl.z + stepIndex) + 0.5) * interval;
        vec3 world = nearPosition + direction * distance;
        float heightExponent = min(Parameters.DensityHeightRange.z * max(world.y - Parameters.DensityHeightRange.y, 0.0), OpticalDepthCeiling);
        float sigmaT = Parameters.DensityHeightRange.x * exp(-heightExponent);
        float transmission = exp(-min(sigmaT * interval, OpticalDepthCeiling));
        segment.rgb += segment.a * (1.0 - transmission) * ScatteringSource(world, direction);
        segment.a *= transmission;
    }
    return segment;
}
void main() { IntegratedTransport = LocalSegment(ivec2(gl_FragCoord.xy)); }
#endif
#endif
