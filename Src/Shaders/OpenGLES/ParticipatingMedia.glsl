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

// Independent OpenGLES participating-medium transport and opaque composition.
// Pharr, Jakob and Humphreys, PBRT v4, Volume Scattering and Equation of Transfer.
// https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance
// EVANS_MEDIA_COMPOSITE selects the resolved-scene consumer of paired S.rgb/T.rgb planes.
#if defined(EVANS_MEDIA_COMPOSITE)
// Paired-plane opaque composition uses the native GL bottom-left image origin.
// Geometry converts canonical clip Z to GL clip Z; sampled hardware depth is
// already the canonical [0,1] value.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FullscreenReach = 3.0;
    vec2 location = gl_VertexID == 0 ? vec2(-1.0, -1.0) :
        gl_VertexID == 1 ? vec2(FullscreenReach, -1.0) : vec2(-1.0, FullscreenReach);
    gl_Position = vec4(location, 0.0, 1.0);
}
#else
uniform mat4 FogInverseCamera;
uniform vec4 FogMedium; // extinction, base height, height falloff, distance limit
uniform vec4 FogRadiance;
uniform vec4 FogViewGrid; // viewport XY and logical cell grid XY
uniform uvec4 FogControl; // integration/composition, depth slices, samples, enabled
uniform highp sampler2D FogSceneColor;
uniform highp sampler2D FogSceneDepth;
uniform highp sampler2D FogTexture;
layout(location = 0) out vec4 OutTransport;

const float MidpointOffset = 0.5;
const float MinimumSegmentExtent = 1.0e-6;
const float HalfFloatCeiling = 65504.0;
const vec4 VacuumTransport = vec4(0.0, 0.0, 0.0, 1.0);

vec3 ReconstructMediumPoint(vec2 imageUV, float hardwareDepth)
{
    vec4 position = FogInverseCamera * vec4(imageUV * 2.0 - 1.0, hardwareDepth, 1.0);
    return position.xyz / position.w;
}

vec4 SampleBoundaryTransmission(vec2 imageUV, uint endpoint)
{
    if (endpoint == 0u) return vec4(1, 1, 1, 1);
    vec2 inset = MidpointOffset / FogViewGrid.zw;
    vec2 boundedUV = clamp(imageUV, inset, 1.0 - inset);
    vec2 atlasUV = vec2((float(endpoint - 1u) + boundedUV.x) / float(FogControl.y), boundedUV.y);
    atlasUV.y = (atlasUV.y + 1.0) * 0.5;
    return textureLod(FogTexture, atlasUV, 0.0);
}
vec4 SampleBoundary(vec2 imageUV, uint endpoint)
{
    if (endpoint == 0u) return VacuumTransport;
    vec2 inset = MidpointOffset / FogViewGrid.zw;
    vec2 boundedUV = clamp(imageUV, inset, 1.0 - inset);
    vec2 atlasUV = vec2((float(endpoint - 1u) + boundedUV.x) / float(FogControl.y), boundedUV.y);
    atlasUV.y *= 0.5;
    return textureLod(FogTexture, atlasUV, 0.0);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec2 imageUV = gl_FragCoord.xy / FogViewGrid.xy;
    vec3 nearPosition = ReconstructMediumPoint(imageUV, 0.0);
    float extent = min(length(ReconstructMediumPoint(imageUV, 1.0) - nearPosition), FogMedium.w);
    float depth = texelFetch(FogSceneDepth, pixel, 0).r;
    float distanceToSurface = length(ReconstructMediumPoint(imageUV, depth) - nearPosition);
    float endpoint = clamp(distanceToSurface / max(extent, MinimumSegmentExtent), 0.0, 1.0) * float(FogControl.y);
    uint before = uint(endpoint);
    uint after = min(before + 1u, FogControl.y);
    vec4 medium = mix(SampleBoundary(imageUV, before), SampleBoundary(imageUV, after), endpoint - float(before));
    vec3 transmittance = (mix(SampleBoundaryTransmission(imageUV, before), SampleBoundaryTransmission(imageUV, after), endpoint - float(before))).rgb;
    vec4 surface = texelFetch(FogSceneColor, pixel, 0);
    OutTransport = vec4(clamp(surface.rgb * transmittance + medium.rgb, 0.0, HalfFloatCeiling), surface.a);
}
#endif

#else
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

// Native ES 3.0 bounded RGB single scattering and ordered transport scan.
// GLSL uses bottom-left texture coordinates. Geometry remaps canonical clip Z
// into GL clip space, so sampled depth already lies in canonical [0,1].
// HG: https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
// Hillis and Steele, "Data Parallel Algorithms" (1986): associative prefix scan.
// Camera and incident-light paths share RGB extinction. Multiple scattering
// is omitted; area lights use four-point rectangular quadrature.
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
uniform vec4 MediaShadowEncoding[CascadeCountLimit + 1];
uniform vec4 MediaShadowParameters;
uniform uvec4 MediaShadowOptions;
uniform vec4 MediaAbsorptionDensity;
uniform vec4 MediaScatteringModulation;
uniform vec4 MediaEmissionFrequency;
uniform vec4 MediaBoundsMinimum;
uniform vec4 MediaBoundsMaximum;
uniform uvec4 MediaOptions;
uniform vec4 MediaDensityGrid;
uniform highp sampler2D DensityImage;
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
const float TexelCenter = 0.5, HomogeneousEpsilon = 1.0e-7, MaximumMapFraction = 0.25;
const uint MediaSoftFilter = 2u, MediaMomentFilter = 3u;
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
    if (clip.w <= HomogeneousEpsilon || (MediaShadowOptions.z & (1u << uint(projectionIndex))) == 0u) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 center = vec2(projected.x + 1.0, 1.0 - projected.y) * 0.5;
    if (any(lessThan(center, vec2(0.0))) || any(greaterThan(center, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    vec4 metric = MediaShadowEncoding[projectionIndex];
    vec2 resolution = vec2(textureSize(image, 0));
    bool perspective = metric.x > 0.0;
    float lightSize = perspective ? MediaShadowParameters.x : MediaShadowParameters.y;
    float comparisonDepth = projected.z - bias;
    if (lightSize == 0.0) return Comparison(image, ivec2(floor(center * resolution)), comparisonDepth);
    float receiverDistance = LightAxisDistance(projected.z, metric);
    float searchRadius = perspective ? lightSize * (receiverDistance - metric.x) / receiverDistance : lightSize * receiverDistance;
    vec2 searchFootprint = ShadowFootprint(searchRadius, metric.zw, resolution, MediaShadowParameters.z);
    float distanceSum = 0.0;
    uint blockerCount = 0u;
    for (uint index = 0u; index < MediaShadowOptions.x; ++index)
    {
        vec2 uv = center + DiskLocation(index, MediaShadowOptions.x) * searchFootprint;
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
    vec2 filterFootprint = ShadowFootprint(filterRadius, metric.zw, resolution, MediaShadowParameters.w);
    float visibility = 0.0;
    for (uint index = 0u; index < MediaShadowOptions.y; ++index)
        visibility += BilinearVisibility(image, center + DiskLocation(index, MediaShadowOptions.y) * filterFootprint, comparisonDepth);
    return visibility / float(MediaShadowOptions.y);
}
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
    if (clip.w <= HomogeneousEpsilon || (MediaShadowOptions.x & (1u << uint(projectionIndex))) == 0u) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = vec2(projected.x + 1.0, 1.0 - projected.y) * 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float receiver = dot(MediaShadowEncoding[projectionIndex].xyz, vec3(clip.zw, 1.0));
    vec2 moments = FilteredMoment(image, uv);
    float separation = receiver - bias - moments.x;
    if (separation <= 0.0) return 1.0;
    float variance = max(moments.y - moments.x * moments.x, MediaShadowParameters.x);
    float bound = variance / (variance + separation * separation);
    // Artistic bleeding suppression forfeits the mathematical upper bound.
    return clamp((bound - MediaShadowParameters.y) / (1.0 - MediaShadowParameters.y), 0.0, 1.0);
}

float VisibilityInMap(highp sampler2D image, vec4 clip, int projection)
{
    if (MediaOptions.y == MediaSoftFilter) return EvaluatePCSS(image, projection, clip, VolumeParameters.y);
    if (MediaOptions.y == MediaMomentFilter) return EvaluateMoments(image, projection, clip, VolumeParameters.y);
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
    if (cascade == 0) return VisibilityInMap(CascadeDepth0, clip, 1);
    if (cascade == 1) return VisibilityInMap(CascadeDepth1, clip, 2);
    if (cascade == 2) return VisibilityInMap(CascadeDepth2, clip, 3);
    return VisibilityInMap(CascadeDepth3, clip, 4);
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
    if (VolumeControl.y == SingleShadow) return VisibilityInMap(PrimaryDepth, VolumeShadowMatrix * vec4(world, 1.0), 0);
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
vec2 mediumRange(vec3 origin, vec3 direction, float maximumDistance)
{
    vec2 range = vec2(0.0, maximumDistance);
    for (int axis = 0; axis < 3; ++axis)
    {
        if (abs(direction[axis]) <= RayLengthFloor)
        {
            if (origin[axis] < MediaBoundsMinimum[axis] || origin[axis] > MediaBoundsMaximum[axis]) return vec2(0.0);
        }
        else
        {
            vec2 intersections = (vec2(MediaBoundsMinimum[axis], MediaBoundsMaximum[axis]) - origin[axis]) / direction[axis];
            range.x = max(range.x, min(intersections.x, intersections.y));
            range.y = min(range.y, max(intersections.x, intersections.y));
        }
    }
    range.y = max(range.y, range.x); return range;
}
float densityCell(ivec3 cell)
{
    ivec3 size = ivec3(MediaDensityGrid.xyz);
    if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, size))) return 0.0;
    int columns = int(MediaDensityGrid.w);
    ivec2 tile = ivec2(cell.z % columns, cell.z / columns);
    const float UnormByteMaximum = 255.0;
    const uint ByteBitCount = 8u;
    uvec4 bytes = uvec4(round(texelFetch(DensityImage, tile * size.xy + cell.xy, 0) * UnormByteMaximum));
    uint bits = bytes.x | (bytes.y << ByteBitCount) | (bytes.z << (2u * ByteBitCount)) | (bytes.w << (3u * ByteBitCount));
    return uintBitsToFloat(bits);
}
float densityFromGrid(vec3 position)
{
    // Point sampling avoids filtering the RGBA8 representation of a float.
    vec3 gridPoint = (position - MediaBoundsMinimum.xyz) / (MediaBoundsMaximum.xyz - MediaBoundsMinimum.xyz) * MediaDensityGrid.xyz - 0.5;
    ivec3 lower = ivec3(floor(gridPoint));
    vec3 blend = gridPoint - vec3(lower);
    float result = 0.0;
    for (int corner = 0; corner < 8; ++corner)
    {
        int x = corner & 1, y = (corner >> 1) & 1, z = (corner >> 2) & 1;
        vec3 weight = vec3(x == 0 ? 1.0 - blend.x : blend.x, y == 0 ? 1.0 - blend.y : blend.y, z == 0 ? 1.0 - blend.z : blend.z);
        result += densityCell(lower + ivec3(x, y, z)) * weight.x * weight.y * weight.z;
    }
    return MediaAbsorptionDensity.w * result;
}
float densityAt(vec3 world)
{
    if (MediaDensityGrid.x > 0.0) return densityFromGrid(world);
    vec3 phase = (world - MediaBoundsMinimum.xyz) * MediaEmissionFrequency.w;
    float wave = sin(phase.x) * sin(phase.y) * sin(phase.z);
    return MediaAbsorptionDensity.w * max(0.0, 1.0 + MediaScatteringModulation.w * wave);
}
vec3 transmittanceToEmitter(vec3 world, vec3 towardLight, float lightDistance)
{
    vec2 bounds = mediumRange(world, towardLight, lightDistance);
    float densityIntegral = MediaAbsorptionDensity.w * (bounds.y - bounds.x);
    if (MediaDensityGrid.x > 0.0 || (MediaScatteringModulation.w > 0.0 && MediaEmissionFrequency.w > 0.0))
    {
        densityIntegral = 0.0;
        float interval = (bounds.y - bounds.x) / float(MediaOptions.x);
        for (uint index = 0u; index < MediaOptions.x; ++index)
            densityIntegral += interval * densityAt(world + towardLight * (bounds.x + (float(index) + 0.5) * interval));
    }
    return exp(-min((MediaAbsorptionDensity.rgb + MediaScatteringModulation.rgb) * densityIntegral, vec3(MaximumOpticalDepth)));
}
vec3 LightGeometry(VolumeEmitter light, vec3 world, vec3 cameraRay)
{
    uint type = uint(light.PositionKind.w);
    bool directional = type == DirectionalLight, rectangle = type != DirectionalLight && type != PointLight;
    vec3 normal = NormalizeOr(light.DirectionPower.xyz, vec3(0.0, 1.0, 0.0));
    vec3 tangent = vec3(0.0), bitangent = vec3(0.0);
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0, StratumCenter = 0.5;
    int steps = 1;
    if (rectangle)
    {
        vec3 helper = abs(normal.y) < BasisParallelThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        tangent = normalize(cross(helper, normal)); bitangent = cross(normal, tangent);
        steps = light.AreaExtent.w > 0.0 ? int(clamp(light.AreaExtent.w, MinimumEmitterGrid, MaximumEmitterGrid)) : RectangleAxisSteps;
    }
    vec3 integral = vec3(0.0);
    // Keep one native density march instead of one per light branch.
    // Point/directional lights have a single quadrature sample.
    for (int sampleIndex = 0; sampleIndex < steps * steps; ++sampleIndex)
    {
        vec3 direction = normal;
        float distanceToLight = length(MediaBoundsMaximum.xyz - MediaBoundsMinimum.xyz), weight = 1.0;
        if (!directional)
        {
            vec2 offset = ((vec2(sampleIndex % steps, sampleIndex / steps) + StratumCenter) / float(steps) - StratumCenter) * light.AreaExtent.xy;
            vec3 delta = light.PositionKind.xyz + tangent * offset.x + bitangent * offset.y - world;
            direction = NormalizeOr(delta, rectangle ? -normal : normal); distanceToLight = length(delta);
            weight = LightRangeWindow(delta, light.AreaExtent.z) / max(dot(delta, delta), LightDistanceSquaredFloor);
            if (rectangle) weight *= light.AreaExtent.x * light.AreaExtent.y * max(dot(normal, -direction), 0.0) / float(steps * steps);
        }
        integral += weight * PhaseDensity(direction, cameraRay) * transmittanceToEmitter(world, direction, distanceToLight);
    }
    return integral;
}
vec3 MediumSource(vec3 world, vec3 cameraRay)
{
    vec3 direct = vec3(0.0);
    if (VolumeParameters.x > 0.0)
    {
        float primaryVisibility = VolumeControl.y == 0u ? 1.0 : PrimaryVisibility(world);
        for (int index = 0; index < int(VolumeControl.x); ++index)
        {
            VolumeEmitter light = ReadVolumeEmitter(index);
            vec3 radiance = min(light.ColorShadow.rgb, vec3(HalfRadianceLimit)) * min(light.DirectionPower.w, HalfRadianceLimit);
            vec3 rayToward=uint(light.PositionKind.w)==0u?light.DirectionPower.xyz:light.PositionKind.xyz-world;
            float visibility=VolumeControl.y!=0u && uint(index)==VolumeControl.z?(ShadowRayEnabled()?ShadowRayVisibility(world,vec3(0.0),rayToward):primaryVisibility):1.0;
            direct += radiance * visibility * LightGeometry(light, world, cameraRay);
        }
    }
    return MediaEmissionFrequency.rgb + MediaScatteringModulation.rgb * (FogRadiance.rgb + VolumeParameters.x * direct);
}
vec3 integratedDistance(vec3 extinction, float distance)
{
    const float SmallOpticalDepth = 0.001, DivisionFloor = 1.0e-20;
    const float QuadraticDenominator = 6.0;
    vec3 opticalDepth = min(extinction * distance, vec3(MaximumOpticalDepth));
    vec3 weight = (1.0 - exp(-opticalDepth)) / max(extinction, vec3(DivisionFloor));
    for (int channel = 0; channel < 3; ++channel)
        if (opticalDepth[channel] < SmallOpticalDepth)
            weight[channel] = distance * (1.0 - opticalDepth[channel] * 0.5 + opticalDepth[channel] * opticalDepth[channel] / QuadraticDenominator);
    return weight;
}
vec4 IntegrateLocalSegment(ivec2 pixel)
{
    ivec2 dimensions = ivec2(FogViewGrid.zw);
    int slice = pixel.x / dimensions.x;
    bool transmissionPlane = pixel.y >= dimensions.y;
    vec2 uv = (vec2(pixel.x % dimensions.x, pixel.y % dimensions.y) + 0.5) / FogViewGrid.zw;
    vec3 origin = WorldFromImage(uv, 0.0), ray = WorldFromImage(uv, 1.0) - origin;
    float rayLength = max(length(ray), RayLengthFloor), extent = min(rayLength, FogMedium.w);
    vec3 direction = ray / rayLength;
    vec2 interval = mediumRange(origin, direction, extent);
    float sliceLength = extent / float(FogControl.y);
    interval.x = max(interval.x, float(slice) * sliceLength);
    interval.y = min(interval.y, float(slice + 1) * sliceLength);
    float stepLength = max(interval.y - interval.x, 0.0) / float(FogControl.z);
    vec3 source = vec3(0.0), transmission = vec3(1.0);
    if (stepLength > 0.0)
        for (uint index = 0u; index < FogControl.z; ++index)
        {
            vec3 world = origin + direction * (interval.x + (float(index) + 0.5) * stepLength);
            float density = densityAt(world);
            vec3 sigma = density * (MediaAbsorptionDensity.rgb + MediaScatteringModulation.rgb);
            if (!transmissionPlane) source += transmission * density * MediumSource(world, direction) * integratedDistance(sigma, stepLength);
            transmission *= exp(-min(sigma * stepLength, vec3(MaximumOpticalDepth)));
        }
    return vec4(transmissionPlane ? transmission : source, 1.0);
}
vec4 ScanSegmentPrefix(ivec2 pixel)
{
    ivec2 dimensions = ivec2(FogViewGrid.zw);
    uint slice = uint(pixel.x / dimensions.x);
    bool transmissionPlane = pixel.y >= dimensions.y;
    vec3 result = texelFetch(PreviousSegments, pixel, 0).rgb;
    if (slice >= VolumeControl.w)
    {
        ivec2 left = pixel - ivec2(int(VolumeControl.w) * dimensions.x, 0);
        ivec2 leftT = ivec2(left.x, left.y % dimensions.y + dimensions.y);
        result *= texelFetch(PreviousSegments, leftT, 0).rgb;
        if (!transmissionPlane) result += texelFetch(PreviousSegments, left, 0).rgb;
    }
    if (VolumeControl.w * 2u >= FogControl.y) result = min(result, vec3(HalfRadianceLimit));
    return vec4(result, 1.0);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    OutTransport = VolumeControl.w == 0u ? IntegrateLocalSegment(pixel) : ScanSegmentPrefix(pixel);
}
#endif

#endif
