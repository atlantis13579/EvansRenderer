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
// Scene light visibility uses primary opaque shadow maps. Light-path medium
// extinction and multiple scattering are not modeled; rectangles use quadrature.
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
    vec4 ShadowEncoding[ShadowCascadeLimit + 1u];
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
float MapVisibility(sampler2D image, vec4 clip)
{
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
    if (cascade == 0u) return MapVisibility(FirstCascadeDepth, clip);
    if (cascade == 1u) return MapVisibility(SecondCascadeDepth, clip);
    if (cascade == 2u) return MapVisibility(ThirdCascadeDepth, clip);
    return MapVisibility(FourthCascadeDepth, clip);
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
        return MapVisibility(PrimaryShadowDepth, Parameters.PrimaryShadowMatrix * vec4(world, 1.0));
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
    float primaryVisibility = Parameters.LightControl.y == 0u ? 1.0 : OpaqueVisibility(world);
    vec3 irradiance = vec3(0.0);
    for (uint index = 0u; index < Parameters.LightControl.x; ++index)
    {
        LightRecord light;
        if (Parameters.IntensityBias.z != 0.0) light = VolumeSceneLights[index];
        else light = Parameters.SceneLights[index];
        vec3 radiance = min(light.ColorCastsShadow.rgb, vec3(HalfFloatCeiling)) * min(light.DirectionStrength.w, HalfFloatCeiling);
        vec3 rayToward=uint(light.PositionKind.w)==0u?light.DirectionStrength.xyz:light.PositionKind.xyz-world;
            float visibility=Parameters.LightControl.y!=0u && index==Parameters.LightControl.z?(ShadowRayEnabled()?ShadowRayVisibility(world,vec3(0.0),rayToward):primaryVisibility):1.0;
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
vec4 PrefixTransport(ivec2 pixel)
{
    uint columns = uint(Parameters.ViewportCells.z), slice = uint(pixel.x) / columns;
    vec4 result = texelFetch(PreviousTransport, pixel, 0);
    if (slice >= Parameters.LightControl.w)
    {
        vec4 left = texelFetch(PreviousTransport, pixel - ivec2(Parameters.LightControl.w * columns, 0u), 0);
        result = vec4(left.rgb + left.a * result.rgb, left.a * result.a);
    }
    // Only the terminal scan stores FP16. Earlier stages retain FP32 radiance
    // and near-unit transmission, including low-density and bright intervals.
    if (Parameters.LightControl.w * 2u >= Parameters.MediumControl.y)
        result.rgb = min(result.rgb, vec3(HalfFloatCeiling));
    return result;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    IntegratedTransport = Parameters.LightControl.w == 0u ? LocalSegment(pixel) : PrefixTransport(pixel);
}
#endif
