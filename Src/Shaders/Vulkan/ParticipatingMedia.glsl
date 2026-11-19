#version 450
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

// Independent Vulkan participating-medium transport and opaque composition.
// Pharr, Jakob and Humphreys, PBRT v4, Volume Scattering and Equation of Transfer.
// https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance
// EVANS_MEDIA_COMPOSITE selects the resolved-scene consumer of paired S.rgb/T.rgb planes.
#if defined(EVANS_MEDIA_COMPOSITE)
// Resolved opaque composition consumes paired RGB transport planes.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FullscreenTriangleExtent = 3.0;
    vec2 corner = gl_VertexIndex == 0 ? vec2(-1.0) : gl_VertexIndex == 1 ?
        vec2(FullscreenTriangleExtent, -1.0) : vec2(-1.0, FullscreenTriangleExtent);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform FogParameters
{
    mat4 InverseCamera;
    vec4 DensityHeightRange;
    vec4 AmbientSource;
    vec4 ViewAndGrid;
    uvec4 StageSlicesSamplesEnabled;
} Fog;
layout(set = 0, binding = 1) uniform sampler2D SceneRadiance;
layout(set = 0, binding = 2) uniform sampler2D SceneDepth;
layout(set = 0, binding = 3) uniform sampler2D CumulativeMedium;
layout(location = 0) out vec4 Result;

const float SampleMidpoint = 0.5;
const float MinimumRayLength = 1.0e-6;
const float MaximumStoredRadiance = 65504.0;
const vec4 IdentityMedium = vec4(0.0, 0.0, 0.0, 1.0);

vec3 PositionOnRay(vec2 uv, float depth)
{
    // The surface vertex shaders flip clip Y. Positive-height Vulkan viewports
    // give top-left image coordinates, so undo that flip before unprojection.
    vec2 ndc = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    vec4 point = Fog.InverseCamera * vec4(ndc, depth, 1.0);
    return point.xyz / point.w;
}

vec4 ReadPrefixTransmission(vec2 uv, uint boundary)
{
    if (boundary == 0u) return vec4(1, 1, 1, 1);
    vec2 border = SampleMidpoint / Fog.ViewAndGrid.zw;
    vec2 localUV = clamp(uv, border, 1.0 - border);
    localUV.x = (float(boundary - 1u) + localUV.x) / float(Fog.StageSlicesSamplesEnabled.y);
    localUV.y = (localUV.y + 1.0) * 0.5;
    return textureLod(CumulativeMedium, localUV, 0.0);
}
vec4 ReadPrefix(vec2 uv, uint boundary)
{
    if (boundary == 0u) return IdentityMedium;
    vec2 border = SampleMidpoint / Fog.ViewAndGrid.zw;
    vec2 localUV = clamp(uv, border, 1.0 - border);
    localUV.x = (float(boundary - 1u) + localUV.x) / float(Fog.StageSlicesSamplesEnabled.y);
    localUV.y *= 0.5;
    return textureLod(CumulativeMedium, localUV, 0.0);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec2 uv = gl_FragCoord.xy / Fog.ViewAndGrid.xy;
    vec3 origin = PositionOnRay(uv, 0.0);
    float range = min(length(PositionOnRay(uv, 1.0) - origin), Fog.DensityHeightRange.w);
    float distance = length(PositionOnRay(uv, texelFetch(SceneDepth, pixel, 0).r) - origin);
    float boundary = clamp(distance / max(range, MinimumRayLength), 0.0, 1.0) * float(Fog.StageSlicesSamplesEnabled.y);
    uint lower = uint(boundary), upper = min(lower + 1u, Fog.StageSlicesSamplesEnabled.y);
    vec4 medium = mix(ReadPrefix(uv, lower), ReadPrefix(uv, upper), fract(boundary));
    vec3 transmittance = (mix(ReadPrefixTransmission(uv, lower), ReadPrefixTransmission(uv, upper), fract(boundary))).rgb;
    vec4 scene = texelFetch(SceneRadiance, pixel, 0);
    Result = vec4(clamp(scene.rgb * transmittance + medium.rgb, 0.0, MaximumStoredRadiance), scene.a);
}
#endif

#else
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

// Independent Vulkan single-scattering producer. Each invocation integrates a
// local depth segment; ordered prefix composition produces the cumulative atlas.
// Hillis and Steele, "Data Parallel Algorithms" (1986).
// Phase convention: https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
// Single scattering includes camera and incident-light extinction. Opaque
// visibility uses the selected primary shadow filter; rectangles use quadrature.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float TriangleEdge = 3.0;
    vec2 corner = gl_VertexIndex == 1 ? vec2(TriangleEdge, -1.0) :
        gl_VertexIndex == 2 ? vec2(-1.0, TriangleEdge) : vec2(-1.0);
    gl_Position = vec4(corner, 0.0, 1.0);
}
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
// The host packs 128-byte camera data, 704-byte lights, 112-byte shadow
// descriptors and 96-byte bounded-medium coefficients in the dynamic UBO.
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
    vec4 ShadowEncoding[ShadowCascadeLimit + 1u];
    vec4 ShadowParameters;
    uvec4 ShadowOptions;
    vec4 AbsorptionDensity;
    vec4 ScatteringModulation;
    vec4 EmissionFrequency;
    vec4 BoundsMinimum;
    vec4 BoundsMaximum;
    uvec4 MaterialOptions;
    vec4 DensityGrid;
} Parameters;
// Full-scene storage preserves lights before the opaque depth interval.
layout(set = 0, binding = 7, std430) readonly buffer VolumeSceneStorage
{ LightRecord VolumeSceneLights[]; };
layout(set = 0, binding = 1) uniform sampler2D PreviousTransport;
layout(set = 0, binding = 2) uniform sampler2D PrimaryShadowDepth;
layout(set = 0, binding = 3) uniform sampler2D FirstCascadeDepth;
layout(set = 0, binding = 4) uniform sampler2D SecondCascadeDepth;
layout(set = 0, binding = 5) uniform sampler2D ThirdCascadeDepth;
layout(set = 0, binding = 6) uniform sampler2D FourthCascadeDepth;
layout(set = 0, binding = 8) uniform sampler2D DensityAtlas;
layout(location = 0) out vec4 IntegratedTransport;

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
float CompareDepth(sampler2D image, ivec2 location, float receiver)
{
    if (any(lessThan(location, ivec2(0))) || any(greaterThanEqual(location, textureSize(image, 0)))) return 1.0;
    return receiver <= texelFetch(image, location, 0).r ? 1.0 : 0.0;
}
float BilinearVisibility(sampler2D image, vec2 uv, float receiver)
{
    vec2 pixel = uv * vec2(textureSize(image, 0)) - 0.5;
    ivec2 corner = ivec2(floor(pixel));
    vec2 fraction = fract(pixel);
    float top = mix(CompareDepth(image, corner, receiver), CompareDepth(image, corner + ivec2(1, 0), receiver), fraction.x);
    float bottom = mix(CompareDepth(image, corner + ivec2(0, 1), receiver), CompareDepth(image, corner + ivec2(1, 1), receiver), fraction.x);
    return mix(top, bottom, fraction.y);
}
ivec2 ShadowExtent(uint map)
{
    if (map == 0u) return textureSize(PrimaryShadowDepth, 0);
    if (map == 1u) return textureSize(FirstCascadeDepth, 0);
    if (map == 2u) return textureSize(SecondCascadeDepth, 0);
    if (map == 3u) return textureSize(ThirdCascadeDepth, 0);
    return textureSize(FourthCascadeDepth, 0);
}
float ShadowDepthAt(uint map, ivec2 pixel)
{
    if (map == 0u) return texelFetch(PrimaryShadowDepth, pixel, 0).r;
    if (map == 1u) return texelFetch(FirstCascadeDepth, pixel, 0).r;
    if (map == 2u) return texelFetch(SecondCascadeDepth, pixel, 0).r;
    if (map == 3u) return texelFetch(ThirdCascadeDepth, pixel, 0).r;
    return texelFetch(FourthCascadeDepth, pixel, 0).r;
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
    if ((Parameters.ShadowOptions.z & (1u << map)) == 0u || clip.w <= PcssProjectionEpsilon) return 1.0;
    vec3 ndc = clip.xyz / clip.w;
    // Native Vulkan depth producers flip clip Y for a positive-height viewport.
    // The resulting image and this kernel both use top-left coordinates.
    vec2 uv = ndc.xy * vec2(0.5, -0.5) + 0.5;
    if (ndc.z < 0.0 || ndc.z > 1.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    ivec2 extent = ShadowExtent(map);
    vec4 projection = Parameters.ShadowEncoding[map];
    bool perspective = projection.x > 0.0;
    float emitter = perspective ? Parameters.ShadowParameters.x : Parameters.ShadowParameters.y;
    float receiver = ndc.z - bias;
    if (emitter == 0.0)
        return VisibilityTexel(map, min(ivec2(uv * vec2(extent)), extent - 1), receiver);
    float receiverDistance = LinearLightDistance(ndc.z, projection);
    if (receiverDistance <= 0.0) return 1.0;
    float searchWidth = perspective ? emitter * max(receiverDistance - projection.x, 0.0) / receiverDistance
        : emitter * receiverDistance;
    vec2 searchRadius = CappedFootprint(searchWidth / projection.zw, extent, Parameters.ShadowParameters.z);
    float blockerDistance = 0.0;
    uint blockers = 0u;
    for (uint index = 0u; index < Parameters.ShadowOptions.x; ++index)
    {
        ivec2 pixel = ivec2(floor((uv + DiskSample(index, Parameters.ShadowOptions.x) * searchRadius) * vec2(extent)));
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
    vec2 filterRadius = CappedFootprint(filterWidth / projection.zw, extent, Parameters.ShadowParameters.w);
    float visible = 0.0;
    for (uint index = 0u; index < Parameters.ShadowOptions.y; ++index)
        visible += ComparisonBilinear(map, uv + DiskSample(index, Parameters.ShadowOptions.y) * filterRadius, receiver);
    return visible / float(Parameters.ShadowOptions.y);
}

vec2 MomentTexel(uint map, ivec2 pixel)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ShadowExtent(map)))) return vec2(1.0);
    if (map == 0u) return texelFetch(PrimaryShadowDepth, pixel, 0).rg;
    if (map == 1u) return texelFetch(FirstCascadeDepth, pixel, 0).rg;
    if (map == 2u) return texelFetch(SecondCascadeDepth, pixel, 0).rg;
    if (map == 3u) return texelFetch(ThirdCascadeDepth, pixel, 0).rg;
    return texelFetch(FourthCascadeDepth, pixel, 0).rg;
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
float VarianceVisibility(uint map, vec4 clip, float linearBias)
{
    if ((Parameters.ShadowOptions.x & (1u << map)) == 0u || clip.w <= SmallProjectiveW) return 1.0;
    vec3 device = clip.xyz / clip.w;
    vec2 uv = vec2(device.x + 1.0, 1.0 - device.y) * 0.5;
    if (device.z < 0.0 || device.z > 1.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    float depth = dot(Parameters.ShadowEncoding[map].xyz, vec3(clip.zw, 1.0));
    vec2 distribution = FilteredMoments(map, uv);
    float difference = depth - linearBias - distribution.x;
    if (difference <= 0.0) return 1.0;
    float variance = max(distribution.y - distribution.x * distribution.x, Parameters.ShadowParameters.x);
    float probability = variance / (variance + difference * difference);
    // Bleeding reduction is an artistic remap, not a stronger probability bound.
    return clamp((probability - Parameters.ShadowParameters.y) / (1.0 - Parameters.ShadowParameters.y), 0.0, 1.0);
}

const uint SoftShadowFilter = 2u, MomentShadowFilter = 3u;
float MapVisibility(sampler2D image, vec4 clip, uint map)
{
    if (Parameters.MaterialOptions.y == SoftShadowFilter) return PcssVisibility(map, clip, Parameters.IntensityBias.y);
    if (Parameters.MaterialOptions.y == MomentShadowFilter) return VarianceVisibility(map, clip, Parameters.IntensityBias.y);
    if (clip.w <= SmallProjectiveW) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = projected.xy * vec2(0.5, -0.5) + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 || any(lessThan(uv, vec2(0.0))) ||
        any(greaterThan(uv, vec2(1.0)))) return 1.0;
    float receiver = projected.z - Parameters.IntensityBias.y;
    ivec2 extent = textureSize(image, 0);
    int radius = int(Parameters.ShadowFilter.x);
    if (radius == 0) return CompareDepth(image, min(ivec2(uv * vec2(extent)), extent - 1), receiver);
    float sum = 0.0;
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x)
            sum += BilinearVisibility(image, uv + vec2(x, y) / vec2(extent), receiver);
    return sum / Parameters.ShadowFilter.y;
}
float CascadeVisibility(uint cascade, vec3 world)
{
    vec4 clip = Parameters.CascadeMatrices[cascade] * vec4(world, 1.0);
    if (cascade == 0u) return MapVisibility(FirstCascadeDepth, clip, cascade + 1u);
    if (cascade == 1u) return MapVisibility(SecondCascadeDepth, clip, cascade + 1u);
    if (cascade == 2u) return MapVisibility(ThirdCascadeDepth, clip, cascade + 1u);
    return MapVisibility(FourthCascadeDepth, clip, cascade + 1u);
}
// Virtual depth and page directory share the exclusive cascade slots.
// Each PCF corner resolves its own page or pinned coarse ancestor.
const uint VirtualPageGutter = 1u;
const float VirtualTexelOrigin = 0.5;
vec2 VirtualDepthAt(vec2 coordinate, uint firstLevel)
{
    uvec4 hierarchy = uvec4(Parameters.CascadeFarPlanes), atlas = uvec4(Parameters.CascadeBlendPlanes);
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
            uint resident = uint(round(texelFetch(SecondCascadeDepth, directoryPixel, 0).r * float(atlas.w)));
            if (resident > 0u)
            {
                uint slot = resident - 1u;
                uvec2 tile = uvec2(slot % atlas.z, slot / atlas.z) * (hierarchy.y + 2u * VirtualPageGutter);
                ivec2 source = ivec2(tile + VirtualPageGutter + pixel % hierarchy.y);
                return vec2(texelFetch(FirstCascadeDepth, source, 0).r, float(level));
            }
        }
        start += pages * pages; pages /= 2u;
    }
    return vec2(1.0, float(hierarchy.z - 1u));
}
float VirtualCompareAt(ivec2 pixel, uint level, float reference)
{
    uint size = uint(Parameters.CascadeFarPlanes.x) >> level;
    vec2 coordinate = (vec2(clamp(pixel, ivec2(0), ivec2(int(size) - 1))) + VirtualTexelOrigin) / float(size);
    return reference <= VirtualDepthAt(coordinate, level).x ? 1.0 : 0.0;
}
// The explicit camera-depth-plane footprint remains valid in ray marching.
uint VirtualVolumeMip(vec3 worldPosition, vec2 coordinate)
{
    vec4 world = vec4(worldPosition, 1.0);
    vec4 camera = vec4(dot(Parameters.ShadowEncoding[0], world), dot(Parameters.ShadowEncoding[1], world), dot(Parameters.ShadowEncoding[2], world), dot(Parameters.ShadowEncoding[3], world));
    if (camera.w <= Parameters.EyeDepthPlane.w) return 0u;
    vec3 depthPlane = camera.xyz / camera.w;
    vec4 neighborX = Parameters.InverseCamera * vec4(depthPlane + vec3(Parameters.EyeDepthPlane.y * 2.0, 0.0, 0.0), 1.0);
    vec4 neighborY = Parameters.InverseCamera * vec4(depthPlane - vec3(0.0, Parameters.EyeDepthPlane.z * 2.0, 0.0), 1.0);
    vec4 projectedX = Parameters.CascadeMatrices[0] * vec4(neighborX.xyz / neighborX.w, 1.0);
    vec4 projectedY = Parameters.CascadeMatrices[0] * vec4(neighborY.xyz / neighborY.w, 1.0);
    if (min(projectedX.w, projectedY.w) <= Parameters.EyeDepthPlane.w) return uint(Parameters.CascadeFarPlanes.z - 1.0);
    vec2 uvX = vec2(projectedX.x / projectedX.w + 1.0, 1.0 - projectedX.y / projectedX.w) * VirtualTexelOrigin;
    vec2 uvY = vec2(projectedY.x / projectedY.w + 1.0, 1.0 - projectedY.y / projectedY.w) * VirtualTexelOrigin;
    float texels = max(length(uvX - coordinate), length(uvY - coordinate)) * Parameters.CascadeFarPlanes.x;
    return uint(clamp(floor(log2(max(1.0, texels)) + Parameters.EyeDepthPlane.x), 0.0, Parameters.CascadeFarPlanes.z - 1.0));
}

float VirtualVisibility(vec3 worldPosition, float bias)
{
    vec4 clip = Parameters.CascadeMatrices[0] * vec4(worldPosition, 1.0);
    vec3 ndc = clip.xyz / max(clip.w, Parameters.EyeDepthPlane.w);
    vec2 coordinate = vec2(ndc.x + 1.0, 1.0 - ndc.y) * VirtualTexelOrigin;
    uint level = VirtualVolumeMip(worldPosition, coordinate);
    if (clip.w <= Parameters.EyeDepthPlane.w || ndc.z < 0.0 || ndc.z > 1.0 || any(lessThan(coordinate, vec2(0.0))) || any(greaterThan(coordinate, vec2(1.0)))) return 1.0;
    vec2 stored = VirtualDepthAt(coordinate, level); float reference = ndc.z - bias;
    if (Parameters.ShadowFilter.x == 0.0) return reference <= stored.x ? 1.0 : 0.0;
    level = uint(stored.y);
    vec2 pixel = coordinate * float(uint(Parameters.CascadeFarPlanes.x) >> level) - VirtualTexelOrigin;
    ivec2 first = ivec2(floor(pixel)); vec2 weight = fract(pixel); float sum = 0.0;
    for (int row = -int(Parameters.ShadowFilter.x); row <= int(Parameters.ShadowFilter.x); ++row)
        for (int column = -int(Parameters.ShadowFilter.x); column <= int(Parameters.ShadowFilter.x); ++column)
        {
            ivec2 at = first + ivec2(column, row);
            float top = mix(VirtualCompareAt(at, level, reference), VirtualCompareAt(at + ivec2(1, 0), level, reference), weight.x);
            float bottom = mix(VirtualCompareAt(at + ivec2(0, 1), level, reference), VirtualCompareAt(at + ivec2(1), level, reference), weight.x);
            sum += mix(top, bottom, weight.y);
        }
    return sum / Parameters.ShadowFilter.y;
}


float OpaqueVisibility(vec3 world)
{
    if (ShadowRayEnabled()) return 1.0;
    if (Parameters.ShadowFilter.z != 0.0 && Parameters.LightControl.y != 0u) return VirtualVisibility(world, Parameters.IntensityBias.y);
    if (Parameters.LightControl.y == PrimaryShadowMode)
        return MapVisibility(PrimaryShadowDepth, Parameters.PrimaryShadowMatrix * vec4(world, 1.0), 0u);
    if (Parameters.LightControl.y != CascadeShadowMode) return 1.0;
    float eyeDepth = dot(Parameters.EyeDepthPlane, vec4(world, 1.0));
    if (eyeDepth < Parameters.CascadeParameters.x || eyeDepth > Parameters.CascadeParameters.y) return 1.0;
    uint count = uint(Parameters.CascadeParameters.z), cascade = 0u;
    while (cascade + 1u < count && eyeDepth > Parameters.CascadeFarPlanes[cascade]) ++cascade;
    float visibility = CascadeVisibility(cascade, world);
    if (cascade + 1u < count && eyeDepth > Parameters.CascadeBlendPlanes[cascade])
    {
        float span = max(Parameters.CascadeFarPlanes[cascade] - Parameters.CascadeBlendPlanes[cascade], SmallRayDistance);
        float blend = clamp((eyeDepth - Parameters.CascadeBlendPlanes[cascade]) / span, 0.0, 1.0);
        visibility = mix(visibility, CascadeVisibility(cascade + 1u, world), blend);
    }
    return visibility;
}
// Clip before quadrature so a thin bounded medium contributes even if its
// support lies between the original uniform camera-ray sample positions.
vec2 MediumInterval(vec3 origin, vec3 direction, float limit)
{
    const float ParallelRayTolerance = 1.0e-8;
    float nearDistance = 0.0, farDistance = limit;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (abs(direction[axis]) < ParallelRayTolerance)
        {
            if (origin[axis] < Parameters.BoundsMinimum[axis] || origin[axis] > Parameters.BoundsMaximum[axis]) return vec2(0.0);
        }
        else
        {
            float first = (Parameters.BoundsMinimum[axis] - origin[axis]) / direction[axis];
            float second = (Parameters.BoundsMaximum[axis] - origin[axis]) / direction[axis];
            nearDistance = max(nearDistance, min(first, second));
            farDistance = min(farDistance, max(first, second));
        }
    }
    return farDistance > nearDistance ? vec2(nearDistance, farDistance) : vec2(0.0);
}
float GridCell(ivec3 cell)
{
    ivec3 dimensions = ivec3(Parameters.DensityGrid.xyz);
    if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, dimensions))) return 0.0;
    int columns = int(Parameters.DensityGrid.w);
    ivec2 address = ivec2(cell.z % columns, cell.z / columns) * dimensions.xy + cell.xy;
    const float ByteRange = 255.0;
    const uint ByteWidth = 8u;
    uvec4 bytes = uvec4(round(texelFetch(DensityAtlas, address, 0) * ByteRange));
    return uintBitsToFloat(bytes.x | (bytes.y << ByteWidth) | (bytes.z << (2u * ByteWidth)) | (bytes.w << (3u * ByteWidth)));
}
float GridDensity(vec3 world)
{
    // Decode lossless float bytes at cell centers, then interpolate eight values.
    vec3 coordinate = (world - Parameters.BoundsMinimum.xyz) /
        (Parameters.BoundsMaximum.xyz - Parameters.BoundsMinimum.xyz) * Parameters.DensityGrid.xyz - 0.5;
    ivec3 base = ivec3(floor(coordinate));
    vec3 fraction = coordinate - vec3(base);
    float result = 0.0;
    for (int corner = 0; corner < 8; ++corner)
    {
        ivec3 offset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        vec3 weight = mix(vec3(1.0) - fraction, fraction, vec3(offset));
        result += GridCell(base + offset) * weight.x * weight.y * weight.z;
    }
    return Parameters.AbsorptionDensity.w * result;
}
float MediumDensity(vec3 world)
{
    if (Parameters.DensityGrid.x > 0.0) return GridDensity(world);
    vec3 wave = sin((world - Parameters.BoundsMinimum.xyz) * Parameters.EmissionFrequency.w);
    return Parameters.AbsorptionDensity.w * max(1.0 + Parameters.ScatteringModulation.w * wave.x * wave.y * wave.z, 0.0);
}
vec3 IncidentTransmission(vec3 world, vec3 direction, float distance)
{
    vec2 segment = MediumInterval(world, direction, distance);
    float integratedDensity = Parameters.AbsorptionDensity.w * (segment.y - segment.x);
    if (Parameters.DensityGrid.x > 0.0 || Parameters.ScatteringModulation.w != 0.0)
    {
        integratedDensity = 0.0;
        float spacing = (segment.y - segment.x) / float(Parameters.MaterialOptions.x);
        for (uint index = 0u; index < Parameters.MaterialOptions.x; ++index)
            integratedDensity += spacing * MediumDensity(world + direction * (segment.x + (float(index) + 0.5) * spacing));
    }
    vec3 opticalDepth = (Parameters.AbsorptionDensity.rgb + Parameters.ScatteringModulation.rgb) * integratedDensity;
    return exp(-min(opticalDepth, vec3(OpticalDepthCeiling)));
}
vec3 EmitterIntegral(LightRecord light, vec3 world, vec3 cameraRay)
{
    uint kind = uint(light.PositionKind.w);
    bool sun = kind == DirectionalKind, area = kind != DirectionalKind && kind != PointKind;
    vec3 normal = DirectionOr(light.DirectionStrength.xyz, vec3(0.0, 1.0, 0.0));
    vec3 axisU = vec3(0.0), axisV = vec3(0.0);
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0, StratumCenter = 0.5;
    int axisSamples = 1;
    if (area)
    {
        vec3 helper = abs(normal.y) < ParallelNormalThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        axisU = normalize(cross(helper, normal)); axisV = cross(normal, axisU);
        axisSamples = light.RectangleExtent.w > 0.0 ? int(clamp(light.RectangleExtent.w, MinimumEmitterGrid, MaximumEmitterGrid)) : RectangleSamplesPerAxis;
    }
    vec3 result = vec3(0.0);
    // A shared sample body bounds SPIR-V/driver inlining of the density trace.
    // The area-to-solid-angle weight remains dA * emitterCosine / distance^2.
    for (int sampleId = 0; sampleId < axisSamples * axisSamples; ++sampleId)
    {
        vec3 ray = normal;
        float extent = length(Parameters.BoundsMaximum.xyz - Parameters.BoundsMinimum.xyz), measure = 1.0;
        if (!sun)
        {
            vec2 uv = ((vec2(sampleId % axisSamples, sampleId / axisSamples) + StratumCenter) / float(axisSamples) - StratumCenter) * light.RectangleExtent.xy;
            vec3 delta = light.PositionKind.xyz + axisU * uv.x + axisV * uv.y - world;
            ray = DirectionOr(delta, area ? -normal : normal); extent = length(delta);
            measure = LightRangeWindow(delta, light.RectangleExtent.z) / max(dot(delta, delta), SmallSquaredLightDistance);
            if (area) measure *= light.RectangleExtent.x * light.RectangleExtent.y * clamp(dot(normal, -ray), 0.0, 1.0) / float(axisSamples * axisSamples);
        }
        result += measure * PhaseProbability(ray, cameraRay) * IncidentTransmission(world, ray, extent);
    }
    return result;
}
vec3 ScatteringSource(vec3 world, vec3 cameraRay)
{
    vec3 irradiance = Parameters.AmbientRadiance.rgb;
    if (Parameters.IntensityBias.x > 0.0)
    {
        float primaryVisibility = Parameters.LightControl.y == 0u ? 1.0 : OpaqueVisibility(world);
        for (uint index = 0u; index < Parameters.LightControl.x; ++index)
        {
            LightRecord light;
            if (Parameters.IntensityBias.z != 0.0) light = VolumeSceneLights[index];
            else light = Parameters.SceneLights[index];
            vec3 radiance = min(light.ColorCastsShadow.rgb, vec3(HalfFloatCeiling)) * min(light.DirectionStrength.w, HalfFloatCeiling);
            vec3 rayToward=uint(light.PositionKind.w)==0u?light.DirectionStrength.xyz:light.PositionKind.xyz-world;
            float visibility=Parameters.LightControl.y!=0u && index==Parameters.LightControl.z?(ShadowRayEnabled()?ShadowRayVisibility(world,vec3(0.0),rayToward):primaryVisibility):1.0;
            irradiance += Parameters.IntensityBias.x * radiance * EmitterIntegral(light, world, cameraRay) * visibility;
        }
    }
    return Parameters.EmissionFrequency.rgb + Parameters.ScatteringModulation.rgb * irradiance;
}
vec3 SourceIntegral(vec3 extinction, float distance)
{
    const float SeriesThreshold = 1.0e-3, SafeDivision = 1.0e-20;
    const float QuadraticDenominator = 6.0;
    vec3 tau = extinction * distance;
    vec3 series = distance * (1.0 - tau * 0.5 + tau * tau / QuadraticDenominator);
    vec3 analytic = (1.0 - exp(-min(tau, vec3(OpticalDepthCeiling)))) / max(extinction, vec3(SafeDivision));
    return mix(analytic, series, lessThan(tau, vec3(SeriesThreshold)));
}
vec4 LocalSegment(ivec2 pixel)
{
    int columns = int(Parameters.ViewportCells.z), rows = int(Parameters.ViewportCells.w), slice = pixel.x / columns;
    bool transmissionPlane = pixel.y >= rows;
    vec2 uv = (vec2(pixel.x % columns, pixel.y % rows) + 0.5) / Parameters.ViewportCells.zw;
    vec3 origin = CameraRayPosition(uv, 0.0), ray = CameraRayPosition(uv, 1.0) - origin;
    float rayLength = max(length(ray), SmallRayDistance), extent = min(rayLength, Parameters.DensityHeightRange.w);
    vec3 direction = ray / rayLength;
    vec2 segment = MediumInterval(origin, direction, extent);
    float sliceExtent = extent / float(Parameters.MediumControl.y);
    segment.x = max(segment.x, float(slice) * sliceExtent);
    segment.y = min(segment.y, float(slice + 1) * sliceExtent);
    vec3 scattering = vec3(0.0), transmission = vec3(1.0);
    if (segment.y > segment.x)
    {
        float spacing = (segment.y - segment.x) / float(Parameters.MediumControl.z);
        for (uint index = 0u; index < Parameters.MediumControl.z; ++index)
        {
            vec3 world = origin + direction * (segment.x + (float(index) + 0.5) * spacing);
            float density = MediumDensity(world);
            vec3 sigmaT = density * (Parameters.AbsorptionDensity.rgb + Parameters.ScatteringModulation.rgb);
            if (!transmissionPlane) scattering += transmission * density * ScatteringSource(world, direction) * SourceIntegral(sigmaT, spacing);
            transmission *= exp(-min(sigmaT * spacing, vec3(OpticalDepthCeiling)));
        }
    }
    return vec4(transmissionPlane ? transmission : scattering, 1.0);
}
vec4 PrefixTransport(ivec2 pixel)
{
    int columns = int(Parameters.ViewportCells.z), rows = int(Parameters.ViewportCells.w);
    uint slice = uint(pixel.x / columns), stride = Parameters.LightControl.w;
    vec3 result = texelFetch(PreviousTransport, pixel, 0).rgb;
    if (slice >= stride)
    {
        ivec2 leftPixel = pixel - ivec2(int(stride) * columns, 0);
        vec3 left = texelFetch(PreviousTransport, leftPixel, 0).rgb;
        if (pixel.y >= rows) result *= left;
        else result = left + texelFetch(PreviousTransport, leftPixel + ivec2(0, rows), 0).rgb * result;
    }
    if (stride * 2u >= Parameters.MediumControl.y) result = min(result, vec3(HalfFloatCeiling));
    return vec4(result, 1.0);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    IntegratedTransport = Parameters.LightControl.w == 0u ? LocalSegment(pixel) : PrefixTransport(pixel);
}
#endif

#endif
