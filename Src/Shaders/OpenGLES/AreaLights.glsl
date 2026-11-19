#version 300 es
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

#ifndef EVANS_VERTEX_STAGE
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

const float ShadowRaySurfaceStrength=0.55;

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

// OpenGLES physically based rectangle emitters: midpoint surface quadrature of the
// full GGX/Smith/Schlick BRDF. CPU preparation converts one-sided total power
// to radiance (Phi = pi * area * Le); W selects the N x N integration grid.
// Visibility comes from the selected raster shadow method, shared by emitter
// samples. This does not claim exact per-sample visibility or analytic GGX.
// Reference: PBRT 4e, Area Lights, https://pbr-book.org/4ed/Light_Sources/Area_Lights
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

// Cook-Torrance microfacet BRDF with GGX distribution, Smith masking and Schlick Fresnel. Walter et al., Microfacet Models for Refraction through Rough Surfaces (2007).
#ifdef EVANS_VERTEX_STAGE
// Depth capture and material replay require identical raster positions.

precision highp float;
precision highp int;
const int MaximumLights = 4;

const float ShadowStrength = 0.55;
const float Pi = 3.141592653589793;
const float MinimumRoughness = 0.045;
const float BrdfEpsilon = 1.0e-6;
const float DistributionEpsilon = 1.0e-12;
const float DielectricReflectance = 0.04;
const float SrgbLinearScale = 12.92;
const float SrgbPowerOffset = 0.055;
const float SrgbPowerScale = 1.055;
const float SrgbExponent = 2.4;
const float SrgbLinearThreshold = 0.04045;
const float NormalizeEpsilon = 1.0e-12;
const float HomogeneousEpsilon = 1.0e-7;
const float UvDeterminantEpsilon = 1.0e-10;
const float DistanceEpsilon = 1.0e-6;
const float ParallelAxisThreshold = 0.999;
const float OutlineWorldScale = 0.01;
const float MinimumShadowCosine = 0.15;
const float MaximumReceiverBias = 0.0045;
const float ReceiverBias = 0.00035;
const float ReceiverSlopeBias = 0.00060;
const float AreaSampleWeight = 0.25;
const float OutlineMode = 1.5;
const int DirectionalLight = 0;
const int PointLight = 1;
const int RectangularLight = 2;
const int AreaGridSize = 2;


layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
uniform mat4 World;
uniform mat4 NormalWorld;
uniform mat4 ViewProjection;
uniform mat4 LightViewProjection;
uniform vec4 MaterialParameters;
uniform vec4 RenderParameters;
out vec3 WorldPosition;
out vec3 WorldNormal;
out vec4 VertexColor;
out vec4 LightPosition;
out vec2 TexCoord;
vec3 SRGBToLinear(vec3 color)
{
    vec3 low = color / SrgbLinearScale;
    vec3 high = pow((color + SrgbPowerOffset) / SrgbPowerScale, vec3(SrgbExponent));
    return mix(high, low, lessThanEqual(color, vec3(SrgbLinearThreshold)));
}
vec3 NormalizeSafe(vec3 value, vec3 fallbackValue)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalizeEpsilon ? value * inversesqrt(lengthSquared) : fallbackValue;
}
void main()
{
    vec4 world = World * vec4(InPosition, 1.0);
    vec3 normal = NormalizeSafe(
        mat3(NormalWorld) * InNormal,
        vec3(0.0, 1.0, 0.0));
    if (RenderParameters.y > OutlineMode)
    {
        world.xyz += normal * (MaterialParameters.w * OutlineWorldScale);
    }
    vec4 clip = ViewProjection * world;
    clip.z = clip.z * 2.0 - clip.w;
    gl_Position = clip;
    WorldPosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    WorldNormal = normal;
    VertexColor = vec4(SRGBToLinear(clamp(InColor.rgb, 0.0, 1.0)), InColor.a);
    LightPosition = LightViewProjection * world;
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    CullWireFace();
}

#else
precision highp float;
precision highp int;

#if defined(EVANS_CONTACT_RECEIVER) && !defined(EVANS_VERTEX_STAGE)
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
#endif
const int MaximumLights = 4;

const float ShadowStrength = 0.55;
const float Pi = 3.141592653589793;
const float MinimumRoughness = 0.045;
const float BrdfEpsilon = 1.0e-6;
const float DistributionEpsilon = 1.0e-12;
const float DielectricReflectance = 0.04;
const float SrgbLinearScale = 12.92;
const float SrgbPowerOffset = 0.055;
const float SrgbPowerScale = 1.055;
const float SrgbExponent = 2.4;
const float SrgbLinearThreshold = 0.04045;
const float NormalizeEpsilon = 1.0e-12;
const float HomogeneousEpsilon = 1.0e-7;
const float UvDeterminantEpsilon = 1.0e-10;
const float DistanceEpsilon = 1.0e-6;
const float ParallelAxisThreshold = 0.999;
const float OutlineWorldScale = 0.01;
const float MinimumShadowCosine = 0.15;
const float MaximumReceiverBias = 0.0045;
const float ReceiverBias = 0.00035;
const float ReceiverSlopeBias = 0.00060;
const float AreaSampleWeight = 0.25;
const float OutlineMode = 1.5;
const int DirectionalLight = 0;
const int PointLight = 1;
const int RectangularLight = 2;
const int AreaGridSize = 2;


struct LightConstants
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
in vec3 WorldPosition;
in vec3 WorldNormal;
in vec4 VertexColor;
in vec4 LightPosition;
in vec2 TexCoord;
uniform sampler2D ShadowMap;
uniform sampler2D DiffuseMap;
uniform sampler2D NormalMap;
uniform sampler2D OpacityMap;
uniform vec4 PipelineColor;
uniform vec4 MaterialDiffuseColor;
uniform vec4 MaterialAmbientColor;
uniform vec4 MaterialSpecularColor;
uniform vec4 MaterialEmissiveColor;
uniform vec4 MaterialParameters;
uniform vec4 AmbientColorStrength;
uniform vec4 CameraLightCount;
uniform vec4 RenderParameters;
uniform vec4 TextureFlags;
uniform vec4 ShadingParameters;
uniform LightConstants Lights[MaximumLights];
layout(location = 0) out vec4 OutColor;
#ifdef EVANS_WEIGHTED_TRANSPARENCY
layout(location = 1) out float OutRevealage;
uniform vec4 TransparencyWeights; // opacity/depth exponents, depth scale, weight floor
#endif

vec3 SRGBToLinear(vec3 color)
{
    vec3 low = color / SrgbLinearScale;
    vec3 high = pow((color + SrgbPowerOffset) / SrgbPowerScale, vec3(SrgbExponent));
    return mix(high, low, lessThanEqual(color, vec3(SrgbLinearThreshold)));
}
vec3 PrepareDisplayLinear(vec3 color)
{
    color = max(color, vec3(0.0));
    color /= max(1.0, max(color.r, max(color.g, color.b)));
    return clamp(color, 0.0, 1.0);
}
vec3 NormalizeSafe(vec3 value, vec3 fallbackValue)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalizeEpsilon ? value * inversesqrt(lengthSquared) : fallbackValue;
}
vec4 SampleDiffuse()
{
    return TextureFlags.x > 0.5
        ? texture(DiffuseMap, TexCoord)
        : vec4(1.0);
}
float SampleOpacity()
{
    return TextureFlags.z > 0.5
        ? texture(OpacityMap, TexCoord).r
        : 1.0;
}
float SurfaceAlpha(vec4 diffuseSample)
{
    return clamp(
        VertexColor.a *
        MaterialParameters.y *
        diffuseSample.a *
        SampleOpacity(),
        0.0,
        1.0);
}
void ApplyAlphaClip(float alpha)
{
    if (MaterialParameters.z > 0.0 && alpha < MaterialParameters.z) discard;
}
// Native ES CSM sampling: positive camera depth selects a fitted interval.
// The CPU includes the previous interval's overlap in each receiver footprint.
// Separate sampler names keep texture selection legal in GLSL ES 3.00.
const int CascadeLimit = 4;
uniform mat4 CascadeLightMatrices[CascadeLimit];
uniform vec4 CascadeFarDistances;
uniform vec4 CascadeBlendStarts;
uniform vec4 CascadeEyeDepthPlane;
uniform vec4 CascadeParameters; // near, far, count, map resolution
uniform vec4 ShadowFiltering; // radius (zero means hard), comparison count, reserved
uniform highp sampler2D CascadeDepth0;
uniform highp sampler2D CascadeDepth1;
uniform highp sampler2D CascadeDepth2;
uniform highp sampler2D CascadeDepth3;
const float CascadeNormalizeEpsilon = 1.0e-12;
const float CascadeHomogeneousEpsilon = 1.0e-7;
const float CascadeIntervalEpsilon = 1.0e-6;
const float CascadeCosineFloor = 0.15;
const float CascadeConstantBias = 0.00035;
const float CascadeSlopeBias = 0.00060;
const float CascadeMaximumBias = 0.0045;
const float CascadeShadowStrength = 0.55;


float CascadeStoredDepth(int cascade, ivec2 pixel)
{
    if (cascade == 0) return texelFetch(CascadeDepth0, pixel, 0).r;
    if (cascade == 1) return texelFetch(CascadeDepth1, pixel, 0).r;
    if (cascade == 2) return texelFetch(CascadeDepth2, pixel, 0).r;
    return texelFetch(CascadeDepth3, pixel, 0).r;
}
float CascadeComparison(int cascade, ivec2 pixel, int resolution, float receiver)
{
    return receiver > CascadeStoredDepth(cascade, clamp(pixel, ivec2(0), ivec2(resolution - 1))) ? 1.0 : 0.0;
}
float CascadeFilteredVisibility(int cascade, vec3 position, float bias)
{
    vec4 projected = CascadeLightMatrices[cascade] * vec4(position, 1.0);
    if (abs(projected.w) < CascadeHomogeneousEpsilon) return 1.0;
    vec3 ndc = projected.xyz / projected.w;
    // ES textures start at the bottom left. Fitted light depth is already [0,1].
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        ndc.z < 0.0 || ndc.z > 1.0) return 1.0;
    int resolution = int(CascadeParameters.w);
    if (ShadowFiltering.x == 0.0)
        return 1.0 - CascadeShadowStrength * CascadeComparison(cascade, ivec2(floor(uv * float(resolution))), resolution, ndc.z - bias);
    vec2 texel = uv * float(resolution) - 0.5;
    ivec2 base = ivec2(floor(texel));
    vec2 fraction = fract(texel);
    float receiver = ndc.z - bias;
    float occlusion = 0.0;
    // Bilinear interpolation of comparisons, never interpolation of depth.
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            ivec2 p = base + ivec2(x, y);
            float lowerLeft = CascadeComparison(cascade, p, resolution, receiver);
            float lowerRight = CascadeComparison(cascade, p + ivec2(1, 0), resolution, receiver);
            float upperLeft = CascadeComparison(cascade, p + ivec2(0, 1), resolution, receiver);
            float upperRight = CascadeComparison(cascade, p + ivec2(1, 1), resolution, receiver);
            occlusion += mix(mix(lowerLeft, lowerRight, fraction.x),
                mix(upperLeft, upperRight, fraction.x), fraction.y);
        }
    return 1.0 - CascadeShadowStrength * occlusion / ShadowFiltering.y;
}
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
float VirtualVisibility(vec3 position, float bias)
{
    vec4 light = CascadeLightMatrices[0] * vec4(position, 1.0);
    vec3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    vec2 uv = vec2(projected.x, -projected.y) * 0.5 + vec2(0.5, 0.5);
    vec2 gradient = VirtualReceiverGradient(uv, projected.z);
    float footprint = max(length(dFdx(uv) * CascadeEyeDepthPlane.y), length(dFdy(uv) * CascadeEyeDepthPlane.z)) * CascadeFarDistances.x;
    uint mip = uint(clamp(floor(log2(max(1.0, footprint)) + CascadeEyeDepthPlane.x), 0.0, CascadeFarDistances.z - 1.0));
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


float CascadedVisibility(vec3 position, vec3 normal, vec3 lightDirection)
{
    if (ShadowFiltering.z != 0.0)
    {
        float cosine = clamp(dot(normal, lightDirection * inversesqrt(max(dot(lightDirection, lightDirection), CascadeNormalizeEpsilon))), 0.0, 1.0);
        float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CascadeCosineFloor);
        float bias = min(CascadeMaximumBias, CascadeConstantBias + CascadeSlopeBias * slope);
        return 1.0 - CascadeShadowStrength * (1.0 - VirtualVisibility(position, bias));
    }
    int count = int(CascadeParameters.z);
    float depth = dot(vec4(position, 1.0), CascadeEyeDepthPlane);
    if (count <= 0 || depth < CascadeParameters.x || depth > CascadeParameters.y) return 1.0;
    int cascade = 0;
    for (int index = 0; index < CascadeLimit - 1; ++index)
        if (index < count - 1 && depth > CascadeFarDistances[index]) cascade = index + 1;
    float lengthSquared = dot(lightDirection, lightDirection);
    vec3 direction = lengthSquared > CascadeNormalizeEpsilon ? lightDirection * inversesqrt(lengthSquared) : normal;
    float cosine = clamp(dot(normal, direction), 0.0, 1.0);
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CascadeCosineFloor);
    float bias = min(CascadeConstantBias + CascadeSlopeBias * slope, CascadeMaximumBias);
    float visibility = CascadeFilteredVisibility(cascade, position, bias);
    float overlap = CascadeFarDistances[cascade] - CascadeBlendStarts[cascade];
    if (overlap > CascadeIntervalEpsilon && depth > CascadeBlendStarts[cascade])
    {
        float nextVisibility = cascade + 1 < count ? CascadeFilteredVisibility(cascade + 1, position, bias) : 1.0;
        float blend = clamp((depth - CascadeBlendStarts[cascade]) / overlap, 0.0, 1.0);
        visibility = mix(visibility, nextVisibility, blend);
    }
    return visibility;
}

float SampleShadow(vec3 surfaceNormal, vec3 directionToLight)
{
    if (RenderParameters.x>=0.5 && ShadowRayEnabled()) return 1.0-ShadowRaySurfaceStrength*(1.0-ShadowRayVisibility(WorldPosition,surfaceNormal,directionToLight));
    if (RenderParameters.x >= 0.5 && CascadeParameters.z > 0.0)
        return CascadedVisibility(WorldPosition, surfaceNormal, directionToLight);
    if (RenderParameters.x < 0.5 || abs(LightPosition.w) < HomogeneousEpsilon)
    {
        return 1.0;
    }
    vec3 projected = LightPosition.xyz / LightPosition.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        projected.z < 0.0 || projected.z > 1.0)
    {
        return 1.0;
    }
    ivec2 size = textureSize(ShadowMap, 0);
    if (size.x <= 0 || size.y <= 0) return 1.0;
    float directionLengthSquared = dot(directionToLight, directionToLight);
    vec3 direction = directionLengthSquared > NormalizeEpsilon
        ? directionToLight * inversesqrt(directionLengthSquared)
        : surfaceNormal;
    float normalToLight = clamp(dot(surfaceNormal, direction), 0.0, 1.0);
    float sineToLight = sqrt(max(1.0 - normalToLight * normalToLight, 0.0));
    float slope = sineToLight / max(normalToLight, MinimumShadowCosine);
    float receiverBias = min(MaximumReceiverBias, ReceiverBias + ReceiverSlopeBias * slope);
    // Nearest hard comparison is separate from bilinear PCF.
    if (ShadowFiltering.x == 0.0)
    {
        ivec2 pixel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
        return 1.0 - ShadowStrength * ((projected.z - receiverBias) > texelFetch(ShadowMap, pixel, 0).r ? 1.0 : 0.0);
    }
    vec2 texelPosition = uv * vec2(size) - 0.5;
    vec2 interpolation = fract(texelPosition);
    ivec2 basePixel = ivec2(floor(texelPosition));
    float occluded = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
    {
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            ivec2 pixel00 = clamp(basePixel + ivec2(x, y), ivec2(0), size - ivec2(1));
            ivec2 pixel10 = clamp(pixel00 + ivec2(1, 0), ivec2(0), size - ivec2(1));
            ivec2 pixel01 = clamp(pixel00 + ivec2(0, 1), ivec2(0), size - ivec2(1));
            ivec2 pixel11 = clamp(pixel00 + ivec2(1, 1), ivec2(0), size - ivec2(1));
            float comparisonDepth = projected.z - receiverBias;
            float comparison00 = comparisonDepth > texelFetch(ShadowMap, pixel00, 0).r ? 1.0 : 0.0;
            float comparison10 = comparisonDepth > texelFetch(ShadowMap, pixel10, 0).r ? 1.0 : 0.0;
            float comparison01 = comparisonDepth > texelFetch(ShadowMap, pixel01, 0).r ? 1.0 : 0.0;
            float comparison11 = comparisonDepth > texelFetch(ShadowMap, pixel11, 0).r ? 1.0 : 0.0;
            float comparisonTop = mix(comparison00, comparison10, interpolation.x);
            float comparisonBottom = mix(comparison01, comparison11, interpolation.x);
            occluded += mix(comparisonTop, comparisonBottom, interpolation.y);
        }
    }
    return 1.0 - ShadowStrength * occluded / ShadowFiltering.y;
}
vec3 ResolveSurfaceNormal()
{
    vec3 normal = NormalizeSafe(WorldNormal, vec3(0.0, 1.0, 0.0));
    if (!gl_FrontFacing) normal = -normal;
    if (TextureFlags.y < 0.5) return normal;
    vec3 positionDx = dFdx(WorldPosition);
    vec3 positionDy = dFdy(WorldPosition);
    vec2 texCoordDx = dFdx(TexCoord);
    vec2 texCoordDy = dFdy(TexCoord);
    float determinant =
        texCoordDx.x * texCoordDy.y -
        texCoordDx.y * texCoordDy.x;
    if (abs(determinant) < UvDeterminantEpsilon) return normal;
    float inverseDeterminant = 1.0 / determinant;
    vec3 tangent =
        (positionDx * texCoordDy.y - positionDy * texCoordDx.y) *
        inverseDeterminant;
    vec3 sourceBitangent =
        (-positionDx * texCoordDy.x + positionDy * texCoordDx.x) *
        inverseDeterminant;
    tangent -= normal * dot(normal, tangent);
    vec3 helper = abs(normal.y) < ParallelAxisThreshold
        ? vec3(0.0, 1.0, 0.0)
        : vec3(1.0, 0.0, 0.0);
    tangent = NormalizeSafe(
        tangent,
        NormalizeSafe(cross(helper, normal), vec3(1.0, 0.0, 0.0)));
    float handedness =
        dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    vec3 bitangent = NormalizeSafe(
        cross(normal, tangent) * handedness,
        vec3(0.0, 0.0, 1.0));
    vec3 tangentNormal = texture(NormalMap, TexCoord).xyz * 2.0 - 1.0;
    return NormalizeSafe(
        tangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}
vec3 SurfaceAlbedo;
void AccumulateLightSample(vec2 roughnessMetallic,
    vec3 lightDirection,
    vec3 radiance,
    float visibility,
    vec3 normal,
    vec3 viewDirection,
    inout vec3 diffuseLight,
    inout vec3 specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= NormalizeEpsilon) return;
    vec3 direction = lightDirection * inversesqrt(lengthSquared);
    float normalToLight = clamp(dot(normal, direction), 0.0, 1.0);
    if (normalToLight <= 0.0) return;
    vec3 visibleRadiance = radiance * visibility;
    vec3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float normalToView = clamp(dot(normal, viewDirection), 0.0, 1.0);
    if (normalToView <= 0.0) return;
    float normalToHalf = clamp(dot(normal, halfVector), 0.0, 1.0);
    float viewToHalf = clamp(dot(viewDirection, halfVector), 0.0, 1.0);
    float roughness = clamp(roughnessMetallic.x, MinimumRoughness, 1.0);
    float metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float denominator = normalToHalf * normalToHalf * (alphaSquared - 1.0) + 1.0;
    float distribution = alphaSquared / max(Pi * denominator * denominator, DistributionEpsilon);
    float smithView = 2.0 * normalToView / (normalToView + sqrt(alphaSquared + (1.0 - alphaSquared) * normalToView * normalToView));
    float smithLight = 2.0 * normalToLight / (normalToLight + sqrt(alphaSquared + (1.0 - alphaSquared) * normalToLight * normalToLight));
    vec3 reflectance = mix(vec3(DielectricReflectance), clamp(SurfaceAlbedo, 0.0, 1.0), metallic);
    const float SchlickExponent = 5.0;
    vec3 fresnel = reflectance + (1.0 - reflectance) * pow(1.0 - viewToHalf, SchlickExponent);
    const float MicrofacetNormalization = 4.0;
    vec3 specular = fresnel * distribution * smithView * smithLight /
        max(MicrofacetNormalization * normalToView * normalToLight, BrdfEpsilon);
    diffuseLight += visibleRadiance * (1.0 - fresnel) * (1.0 - metallic) * normalToLight / Pi;
    specularLight += visibleRadiance * specular * normalToLight;
}
void AccumulateSceneLight(vec2 roughnessMetallic,
    int lightIndex,
    vec3 normal,
    vec3 viewDirection,
    inout vec3 diffuseLight,
    inout vec3 specularLight)
{
    LightConstants light = Lights[lightIndex];
    int type = int(light.PositionType.w + 0.5);
    vec3 shadowDirection = type == DirectionalLight
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - WorldPosition;
    bool isShadowLight =
        lightIndex == int(RenderParameters.z) &&
        light.ColorCastShadows.w > 0.5;
    float visibility = isShadowLight
        ? SampleShadow(normal, shadowDirection)
        : 1.0;
#ifdef EVANS_CONTACT_RECEIVER
    if (isShadowLight) visibility = ContactShadow(visibility, ShadowStrength);
#endif
    vec3 radiance =
        max(light.ColorCastShadows.rgb, vec3(0.0)) *
        max(light.DirectionIntensity.w, 0.0);
    if (type == DirectionalLight)
    {
        AccumulateLightSample(roughnessMetallic,
            light.DirectionIntensity.xyz, radiance, visibility,
            normal, viewDirection, diffuseLight, specularLight);
        return;
    }
    if (type == PointLight)
    {
        vec3 toLight = light.PositionType.xyz - WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), DistanceEpsilon);
        AccumulateLightSample(roughnessMetallic,
            toLight, radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared, visibility,
            normal, viewDirection, diffuseLight, specularLight);
        return;
    }
    vec3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz, vec3(0.0, -1.0, 0.0));
    vec3 helper = abs(emissionDirection.y) < ParallelAxisThreshold
        ? vec3(0.0, 1.0, 0.0)
        : vec3(1.0, 0.0, 0.0);
    vec3 tangent = NormalizeSafe(
        cross(helper, emissionDirection), vec3(1.0, 0.0, 0.0));
    vec3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent), vec3(0.0, 0.0, 1.0));
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
            vec3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            vec3 toLight = samplePosition - WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), DistanceEpsilon);
            vec3 direction = toLight * inversesqrt(distanceSquared);
            float emitterCosine = max(dot(emissionDirection, -direction), 0.0);
            AccumulateLightSample(roughnessMetallic,
                direction,
                radiance * (emitterSampleWeight * emitterCosine * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared),
                visibility, normal, viewDirection, diffuseLight, specularLight);
        }
    }
}
// Transparent coverage samples cumulative fog at this fragment's depth.
// The standalone VolumetricFog program owns density integration. GL window
// coordinates and atlas rows both run upward from the bottom-left corner.
uniform mat4 FogInverseCamera;
uniform vec4 FogMedium;
uniform vec4 FogViewGrid;
uniform uvec4 FogControl;
uniform highp sampler2D FogTexture;
const float FogRayEpsilon = 1.0e-6;
const float FogHalfFloatMaximum = 65504.0;

vec3 FogCameraPoint(vec2 uv, float depth)
{
    vec4 h = FogInverseCamera * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return h.xyz / h.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
const uint RgbTransportMode = 2u;
vec4 FogPrefixTransmission(vec2 uv, uint boundary)
{
    if (boundary == 0u) return vec4(1, 1, 1, 1);
    vec2 halfCell = vec2(0.5) / FogViewGrid.zw;
    vec2 insideCell = clamp(uv, halfCell, vec2(1.0) - halfCell);
    insideCell.x = (insideCell.x + float(boundary - 1u)) / float(FogControl.y);
    insideCell.y = (insideCell.y + 1.0) * 0.5;
    return textureLod(FogTexture, insideCell, 0.0);
}
vec4 FogPrefix(vec2 uv, uint boundary)
{
    if (boundary == 0u) return vec4(0.0, 0.0, 0.0, 1.0);
    vec2 halfCell = vec2(0.5) / FogViewGrid.zw;
    vec2 insideCell = clamp(uv, halfCell, vec2(1.0) - halfCell);
    insideCell.x = (insideCell.x + float(boundary - 1u)) / float(FogControl.y);
    if (FogControl.w == RgbTransportMode) insideCell.y *= 0.5;
    return textureLod(FogTexture, insideCell, 0.0);
}
vec3 ApplyHeightMedium(vec3 surfaceRadiance)
{
    if (FogControl.w == 0u) return surfaceRadiance;
    vec2 uv = gl_FragCoord.xy / FogViewGrid.xy;
    vec3 rayStart = FogCameraPoint(uv, 0.0);
    float farDistance = min(length(FogCameraPoint(uv, 1.0) - rayStart), FogMedium.w);
    float surfaceDistance = length(FogCameraPoint(uv, gl_FragCoord.z) - rayStart);
    float slice = clamp(surfaceDistance / max(farDistance, FogRayEpsilon), 0.0, 1.0) * float(FogControl.y);
    uint first = uint(slice);
    vec4 transport = mix(FogPrefix(uv, first), FogPrefix(uv, min(first + 1u, FogControl.y)), slice - float(first));
    vec3 mediumTransmission = FogControl.w == RgbTransportMode ? (mix(FogPrefixTransmission(uv, first), FogPrefixTransmission(uv, min(first + 1u, FogControl.y)), slice - float(first))).rgb : transport.aaa;
    return clamp(surfaceRadiance * mediumTransmission + transport.rgb, 0.0, FogHalfFloatMaximum);
}

void ShadeSurface()
{
    vec2 roughnessMetallic = ShadingParameters.xy;
    if (TextureFlags.z > 1.5) roughnessMetallic *= texture(OpacityMap, TexCoord).gb;
    vec4 diffuseSample = SampleDiffuse();
    float alpha = SurfaceAlpha(diffuseSample);
    ApplyAlphaClip(alpha);
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (RenderParameters.y > 1.0 && RenderParameters.y < 1.5)
    {
        vec3 wireBary = VertexColor.rgb;
        vec3 wireDx = dFdx(wireBary), wireDy = dFdy(wireBary);
        vec3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), vec3(1.0e-8));
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (RenderParameters.y > 0.5)
    {
        OutColor = vec4(
            ApplyHeightMedium(SRGBToLinear(clamp(PipelineColor.rgb, 0.0, 1.0))),
            alpha);
        return;
    }
    vec3 surfaceBase =
        VertexColor.rgb *
        max(PipelineColor.rgb, vec3(0.0)) *
        max(diffuseSample.rgb, vec3(0.0));
    SurfaceAlbedo = surfaceBase * max(MaterialDiffuseColor.rgb, vec3(0.0));
    vec3 normal = ResolveSurfaceNormal();
    vec3 viewDirection = NormalizeSafe(
        CameraLightCount.xyz - WorldPosition,
        normal);
    vec3 diffuseLight = vec3(0.0);
    vec3 specularLight = vec3(0.0);
    int lightCount = min(int(CameraLightCount.w + 0.5), MaximumLights);
    for (int lightIndex = 0; lightIndex < MaximumLights; ++lightIndex)
    {
        if (lightIndex >= lightCount) break;
        AccumulateSceneLight(roughnessMetallic,
            lightIndex, normal, viewDirection,
            diffuseLight, specularLight);
    }
    vec3 color =
        surfaceBase *
            max(MaterialAmbientColor.rgb, vec3(0.0)) *
            max(AmbientColorStrength.rgb, vec3(0.0)) *
            max(AmbientColorStrength.w, 0.0) * (1.0 - clamp(roughnessMetallic.y, 0.0, 1.0)) +
        surfaceBase *
            max(MaterialDiffuseColor.rgb, vec3(0.0)) *
            diffuseLight +
        specularLight +
        max(MaterialEmissiveColor.rgb, vec3(0.0));
    color += surfaceBase * max(MaterialDiffuseColor.rgb,vec3(0.0)) * (1.0-LumenDielectricReflectance) * (1.0-clamp(roughnessMetallic.y,0.0,1.0)) * LumenTranslucentIrradiance(WorldPosition,normal);
    color = ApplyHeightMedium(color);
    // HDR keeps scene radiance above one until the separate tone-map pass.
    const float MaximumHalfFloatRadiance = 65504.0;
    OutColor = vec4(ShadingParameters.w > 0.5 ? clamp(color, 0.0, MaximumHalfFloatRadiance) : PrepareDisplayLinear(color), alpha);
}

void main()
{
    ShadeSurface();
#ifdef EVANS_WEIGHTED_TRANSPARENCY
    // Lighting and fragment-depth fog precede order-independent statistics.
    // Independent MRT blending adds the first output and multiplies revealage
    // by one minus the opacity carried by the second output.
    float opacity = clamp(OutColor.a, 0.0, 1.0);
    OutRevealage = opacity;
    if (opacity <= 0.0) { OutColor = vec4(0.0); return; }
    float depthPreference = 1.0 - clamp(gl_FragCoord.z, 0.0, 1.0) * TransparencyWeights.z;
    float weight = max(pow(opacity, TransparencyWeights.x) *
        pow(depthPreference, TransparencyWeights.y), TransparencyWeights.w);
    float weightedOpacity = opacity * weight;
    OutColor = vec4(OutColor.rgb * weightedOpacity, weightedOpacity);
#endif
}

#endif
