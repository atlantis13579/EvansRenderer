#version 300 es
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

// Native ES 3.0 height-medium single scattering and ordered transport scan.
// GLSL uses bottom-left texture coordinates. Geometry remaps canonical clip Z
// into GL clip space, so sampled depth already lies in canonical [0,1].
// HG: https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
// Hillis and Steele, "Data Parallel Algorithms" (1986): associative prefix scan.
// Primary opaque shadows are included; light-path extinction and multiple
// scattering are omitted. Area lights use four-point rectangular quadrature.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float TriangleReach = 3.0;
    vec2 corner = gl_VertexID == 1 ? vec2(TriangleReach, -1.0) :
        gl_VertexID == 2 ? vec2(-1.0, TriangleReach) : vec2(-1.0);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
const int LightCountLimit = 4;
const int CascadeCountLimit = 4;
uniform mat4 FogInverseCamera;
uniform vec4 FogMedium;
uniform vec4 FogRadiance;
uniform vec4 FogViewGrid;
uniform uvec4 FogControl;
uniform vec4 VolumeScattering; // RGB scattering albedo, HG asymmetry.
uniform vec4 VolumeParameters; // Direct-light scale, normalized shadow bias.
uniform uvec4 VolumeControl; // Light count, shadow mode, primary index, scan stride.
uniform mat4 VolumeShadowMatrix;
uniform vec4 VolumeLightPosition[LightCountLimit];
uniform vec4 VolumeLightDirection[LightCountLimit];
uniform vec4 VolumeLightColor[LightCountLimit];
uniform vec4 VolumeLightArea[LightCountLimit];
struct VolumeEmitter
{
    vec4 PositionKind, DirectionPower, ColorShadow, AreaExtent;
};
#ifdef EVANS_VOLUME_FULL_LIGHTS
// ES 3.1 variant: scattering positions are not bounded by opaque tile depths.
layout(std430, binding = 0) readonly buffer VolumeSceneStorage
{ vec4 VolumeSceneWords[]; };
#endif
VolumeEmitter ReadVolumeEmitter(int index)
{
#ifdef EVANS_VOLUME_FULL_LIGHTS
    // Scalar-vector loads avoid aggregate SSBO-return bugs in ANGLE's HLSL path.
    const int WordsPerEmitter = 4;
    const int PositionWord = 0, DirectionWord = 1, ColorWord = 2, AreaWord = 3;
    int first = index * WordsPerEmitter;
    return VolumeEmitter(VolumeSceneWords[first + PositionWord], VolumeSceneWords[first + DirectionWord],
        VolumeSceneWords[first + ColorWord], VolumeSceneWords[first + AreaWord]);
#else
    return VolumeEmitter(VolumeLightPosition[index], VolumeLightDirection[index],
        VolumeLightColor[index], VolumeLightArea[index]);
#endif
}

uniform mat4 VolumeCascadeMatrix[CascadeCountLimit];
uniform vec4 VolumeCascadeEnds;
uniform vec4 VolumeBlendBegin;
uniform vec4 VolumeEyeDepth;
uniform vec4 VolumeCascadeRange;
uniform vec4 VolumeFilter;
uniform highp sampler2D PreviousSegments;
uniform highp sampler2D PrimaryDepth;
uniform highp sampler2D CascadeDepth0;
uniform highp sampler2D CascadeDepth1;
uniform highp sampler2D CascadeDepth2;
uniform highp sampler2D CascadeDepth3;
layout(location = 0) out vec4 OutTransport;

const float Pi = 3.14159265358979323846;
const float MaximumOpticalDepth = 80.0;
const float RayLengthFloor = 1.0e-6;
const float LightDistanceSquaredFloor = 1.0e-4;
const float ProjectionFloor = 1.0e-7;
const float HalfRadianceLimit = 65504.0;
const float BasisParallelThreshold = 0.99;
const uint DirectionalLight = 0u, PointLight = 1u;
const uint SingleShadow = 1u, CascadedShadow = 2u;
const int RectangleAxisSteps = 2;
const float RectangleWeight = 1.0 / float(RectangleAxisSteps * RectangleAxisSteps);

vec3 NormalizeOr(vec3 value, vec3 fallback)
{
    float squareLength = dot(value, value);
    return squareLength > LightDistanceSquaredFloor ? value * inversesqrt(squareLength) : fallback;
}
vec3 WorldFromImage(vec2 uv, float depth)
{
    vec4 homogeneous = FogInverseCamera * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return homogeneous.xyz / homogeneous.w;
}
float PhaseDensity(vec3 towardLight, vec3 awayFromCamera)
{
    // Incoming light travel and outgoing eye travel are both negatives of these
    // vectors. Their cosine gives forward scattering for positive g near +1.
    float cosine = clamp(dot(towardLight, awayFromCamera), -1.0, 1.0);
    float g = VolumeScattering.w;
    float denominator = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / (4.0 * Pi * denominator * sqrt(denominator));
}
float CompareShadow(highp sampler2D image, ivec2 pixel, float receiver)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, textureSize(image, 0)))) return 1.0;
    return receiver <= texelFetch(image, pixel, 0).r ? 1.0 : 0.0;
}
float LinearComparison(highp sampler2D image, vec2 uv, float receiver)
{
    vec2 location = uv * vec2(textureSize(image, 0)) - 0.5;
    ivec2 origin = ivec2(floor(location));
    vec2 weight = fract(location);
    vec2 rows = vec2(mix(CompareShadow(image, origin, receiver),
        CompareShadow(image, origin + ivec2(1, 0), receiver), weight.x),
        mix(CompareShadow(image, origin + ivec2(0, 1), receiver),
        CompareShadow(image, origin + ivec2(1, 1), receiver), weight.x));
    return mix(rows.x, rows.y, weight.y);
}
float VisibilityInMap(highp sampler2D image, vec4 clip)
{
    if (clip.w <= ProjectionFloor) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 || any(lessThan(uv, vec2(0.0))) ||
        any(greaterThan(uv, vec2(1.0)))) return 1.0;
    ivec2 extent = textureSize(image, 0);
    float receiver = projected.z - VolumeParameters.y;
    int radius = int(VolumeFilter.x);
    if (radius == 0) return CompareShadow(image, min(ivec2(uv * vec2(extent)), extent - 1), receiver);
    float visibility = 0.0;
    for (int row = -radius; row <= radius; ++row)
        for (int column = -radius; column <= radius; ++column)
            visibility += LinearComparison(image, uv + vec2(column, row) / vec2(extent), receiver);
    return visibility / VolumeFilter.y;
}
float VisibilityInCascade(int cascade, vec3 world)
{
    vec4 clip = VolumeCascadeMatrix[cascade] * vec4(world, 1.0);
    // ES 3.0 sampler arrays require constant indices. Named samplers keep each
    // cascade independently bound without relying on a newer language version.
    if (cascade == 0) return VisibilityInMap(CascadeDepth0, clip);
    if (cascade == 1) return VisibilityInMap(CascadeDepth1, clip);
    if (cascade == 2) return VisibilityInMap(CascadeDepth2, clip);
    return VisibilityInMap(CascadeDepth3, clip);
}
// Virtual depth and page directory share the exclusive cascade slots.
// Each PCF corner resolves its own page or pinned coarse ancestor.
const uint VirtualBorder = 1u;
const float VirtualCenter = 0.5;
vec2 VirtualDepth(vec2 uv, uint requestedMip)
{
    uvec4 extent = uvec4(VolumeCascadeEnds), physical = uvec4(VolumeBlendBegin);
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
float VirtualCompare(ivec2 pixel, uint mip, float receiver)
{
    uint resolution = uint(VolumeCascadeEnds.x) >> mip;
    vec2 uv = (vec2(clamp(pixel, ivec2(0), ivec2(int(resolution) - 1))) + VirtualCenter) / float(resolution);
    return step(receiver, VirtualDepth(uv, mip).x);
}
uniform vec4 VirtualCameraRows[4];
// Reconstruct one horizontal and vertical cell neighbor at equal camera depth.
// The resulting footprint is independent of fragment-quad control flow.
uint VirtualVolumeMip(vec3 position, vec2 centerUV)
{
    vec4 world = vec4(position, 1.0);
    vec4 camera = vec4(dot(VirtualCameraRows[0], world), dot(VirtualCameraRows[1], world), dot(VirtualCameraRows[2], world), dot(VirtualCameraRows[3], world));
    if (camera.w <= VolumeEyeDepth.w) return 0u;
    vec3 ndc = camera.xyz / camera.w;
    vec4 right = FogInverseCamera * vec4(ndc + vec3(2.0 * VolumeEyeDepth.y, 0.0, 0.0), 1.0);
    vec4 down = FogInverseCamera * vec4(ndc - vec3(0.0, 2.0 * VolumeEyeDepth.z, 0.0), 1.0);
    vec4 lightX = VolumeCascadeMatrix[0] * vec4(right.xyz / right.w, 1.0);
    vec4 lightY = VolumeCascadeMatrix[0] * vec4(down.xyz / down.w, 1.0);
    if (min(lightX.w, lightY.w) <= VolumeEyeDepth.w) return uint(VolumeCascadeEnds.z - 1.0);
    vec2 uvX = lightX.xy / lightX.w * vec2(VirtualCenter, -VirtualCenter) + VirtualCenter;
    vec2 uvY = lightY.xy / lightY.w * vec2(VirtualCenter, -VirtualCenter) + VirtualCenter;
    float width = max(length(uvX - centerUV), length(uvY - centerUV)) * VolumeCascadeEnds.x;
    return uint(clamp(floor(log2(max(1.0, width)) + VolumeEyeDepth.x), 0.0, VolumeCascadeEnds.z - 1.0));
}

float VirtualVisibility(vec3 position, float bias)
{
    vec4 light = VolumeCascadeMatrix[0] * vec4(position, 1.0);
    vec3 projected = light.xyz / max(light.w, VolumeEyeDepth.w);
    vec2 uv = vec2(projected.x, -projected.y) * VirtualCenter + VirtualCenter;
    uint mip = VirtualVolumeMip(position, uv);
    if (light.w <= VolumeEyeDepth.w || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec2 depth = VirtualDepth(uv, mip); float receiver = projected.z - bias;
    if (VolumeFilter.x == 0.0) return step(receiver, depth.x);
    mip = uint(depth.y);
    vec2 texel = uv * float(uint(VolumeCascadeEnds.x) >> mip) - VirtualCenter;
    ivec2 base = ivec2(floor(texel)); vec2 fraction = fract(texel); float visible = 0.0;
    for (int y = -int(VolumeFilter.x); y <= int(VolumeFilter.x); ++y)
        for (int x = -int(VolumeFilter.x); x <= int(VolumeFilter.x); ++x)
        {
            ivec2 p = base + ivec2(x, y);
            float a = VirtualCompare(p, mip, receiver), b = VirtualCompare(p + ivec2(1, 0), mip, receiver);
            float c = VirtualCompare(p + ivec2(0, 1), mip, receiver), d = VirtualCompare(p + ivec2(1), mip, receiver);
            visible += mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
        }
    return visible / VolumeFilter.y;
}


float PrimaryVisibility(vec3 world)
{
    if (ShadowRayEnabled()) return 1.0;
    if (VolumeFilter.z != 0.0 && VolumeControl.y != 0u) return VirtualVisibility(world, VolumeParameters.y);
    if (VolumeControl.y == SingleShadow) return VisibilityInMap(PrimaryDepth, VolumeShadowMatrix * vec4(world, 1.0));
    if (VolumeControl.y != CascadedShadow) return 1.0;
    float depth = dot(VolumeEyeDepth, vec4(world, 1.0));
    if (depth < VolumeCascadeRange.x || depth > VolumeCascadeRange.y) return 1.0;
    int cascade = 0, count = int(VolumeCascadeRange.z);
    while (cascade + 1 < count && depth > VolumeCascadeEnds[cascade]) ++cascade;
    float visibility = VisibilityInCascade(cascade, world);
    if (cascade + 1 < count && depth > VolumeBlendBegin[cascade])
    {
        float span = max(VolumeCascadeEnds[cascade] - VolumeBlendBegin[cascade], RayLengthFloor);
        float blend = clamp((depth - VolumeBlendBegin[cascade]) / span, 0.0, 1.0);
        visibility = mix(visibility, VisibilityInCascade(cascade + 1, world), blend);
    }
    return visibility;
}
float LightGeometry(VolumeEmitter light, vec3 world, vec3 cameraRay)
{
    uint kind = uint(light.PositionKind.w);
    vec3 normal = NormalizeOr(light.DirectionPower.xyz, vec3(0.0, 1.0, 0.0));
    if (kind == DirectionalLight) return PhaseDensity(normal, cameraRay);
    if (kind == PointLight)
    {
        vec3 delta = light.PositionKind.xyz - world;
        return PhaseDensity(NormalizeOr(delta, normal), cameraRay) * LightRangeWindow(delta, light.AreaExtent.z) / max(dot(delta, delta), LightDistanceSquaredFloor);
    }
    vec3 helper = abs(normal.y) < BasisParallelThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
    float integral = 0.0;
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaExtent.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaExtent.w, MinimumEmitterGrid, MaximumEmitterGrid)) : RectangleAxisSteps;
    float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
    for (int row = 0; row < emitterGrid; ++row)
        for (int column = 0; column < emitterGrid; ++column)
        {
            vec2 offset = ((vec2(column, row) + 0.5) / float(emitterGrid) - 0.5) * light.AreaExtent.xy;
            vec3 delta = light.PositionKind.xyz + tangent * offset.x + bitangent * offset.y - world;
            vec3 towardLight = NormalizeOr(delta, -normal);
            integral += emitterSampleWeight * max(dot(normal, -towardLight), 0.0) * PhaseDensity(towardLight, cameraRay) * LightRangeWindow(delta, light.AreaExtent.z) / max(dot(delta, delta), LightDistanceSquaredFloor);
        }
    return integral * light.AreaExtent.x * light.AreaExtent.y;
}
vec3 MediumSource(vec3 world, vec3 cameraRay)
{
    vec3 direct = vec3(0.0);
    float primaryVisibility = VolumeControl.y == 0u ? 1.0 : PrimaryVisibility(world);
    for (int index = 0; index < int(VolumeControl.x); ++index)
    {
        VolumeEmitter light = ReadVolumeEmitter(index);
        vec3 radiance = min(light.ColorShadow.rgb, vec3(HalfRadianceLimit)) * min(light.DirectionPower.w, HalfRadianceLimit);
        vec3 rayToward=uint(light.PositionKind.w)==0u?light.DirectionPower.xyz:light.PositionKind.xyz-world;
            float visibility=VolumeControl.y!=0u && uint(index)==VolumeControl.z?(ShadowRayEnabled()?ShadowRayVisibility(world,vec3(0.0),rayToward):primaryVisibility):1.0;
        direct += radiance * (LightGeometry(light, world, cameraRay) * visibility);
    }
    return FogRadiance.rgb + VolumeScattering.rgb * VolumeParameters.x * direct;
}
vec4 IntegrateLocalSegment(ivec2 pixel)
{
    int width = int(FogViewGrid.z), slice = pixel.x / width;
    vec2 uv = (vec2(pixel.x % width, pixel.y) + 0.5) / FogViewGrid.zw;
    vec3 origin = WorldFromImage(uv, 0.0), ray = WorldFromImage(uv, 1.0) - origin;
    float lengthToFar = max(length(ray), RayLengthFloor);
    vec3 direction = ray / lengthToFar;
    float stepLength = min(lengthToFar, FogMedium.w) / float(FogControl.y * FogControl.z);
    vec4 segment = vec4(0.0, 0.0, 0.0, 1.0);
    for (uint sampleIndex = 0u; sampleIndex < FogControl.z; ++sampleIndex)
    {
        float distance = (float(uint(slice) * FogControl.z + sampleIndex) + 0.5) * stepLength;
        vec3 world = origin + direction * distance;
        float sigmaT = FogMedium.x * exp(-min(FogMedium.z * max(world.y - FogMedium.y, 0.0), MaximumOpticalDepth));
        float transmission = exp(-min(sigmaT * stepLength, MaximumOpticalDepth));
        segment.rgb += segment.a * (1.0 - transmission) * MediumSource(world, direction);
        segment.a *= transmission;
    }
    return segment;
}
vec4 ScanSegmentPrefix(ivec2 pixel)
{
    uint width = uint(FogViewGrid.z), slice = uint(pixel.x) / width;
    vec4 current = texelFetch(PreviousSegments, pixel, 0);
    if (slice >= VolumeControl.w)
    {
        vec4 left = texelFetch(PreviousSegments, pixel - ivec2(VolumeControl.w * width, 0u), 0);
        current = vec4(left.rgb + left.a * current.rgb, left.a * current.a);
    }
    // FP32 intermediates preserve small extinction. Clamp all terminal slices
    // only at the final FP16 atlas store, after nearer transport attenuates them.
    if (VolumeControl.w * 2u >= FogControl.y) current.rgb = min(current.rgb, vec3(HalfRadianceLimit));
    return current;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    OutTransport = VolumeControl.w == 0u ? IntegrateLocalSegment(pixel) : ScanSegmentPrefix(pixel);
}
#endif
