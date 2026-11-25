#include <metal_stdlib>
using namespace metal;
// Metal world-space shadow queries. Woop, Benthin and Wald, JCGT 2013.
// https://jcgt.org/published/0002/01/05/ . Forward escape links bound traversal;
// independent alpha planes preserve bilinear cutout coverage. The RGBA32UI ABI
// is described in RayTracedShadowScene.h. No screen-space visibility is assumed.
constant uint ShadowRayCasterFlag=1u,ShadowRayImageABI=1u;
constant uint ShadowRayCountsHeader=0u,ShadowRayOffsetsHeader=1u,ShadowRayTraceHeader=2u,ShadowRayLightHeader=3u;
constant uint ShadowRayNodeSize=3u,ShadowRayTriangleSize=6u,ShadowRayMaterialSize=3u,ShadowRayScalarLanes=4u;
constant uint ShadowRayNodeMaximum=1u,ShadowRayNodeLinks=2u;
constant uint ShadowRayTriangleUV01=3u,ShadowRayTriangleUV2=4u,ShadowRayTriangleMetadata=5u;
constant uint ShadowRayMaterialOpacity=1u,ShadowRayMaterialCoverageWord=2u;
uint4 ShadowRayLoad(texture2d<uint,access::read> storage,uint address)
{ uint width=max(storage.get_width(),1u); return storage.read(uint2(address%width,address/width)); }
struct ShadowRayNode { float4 Minimum,Maximum; uint4 Links; };
struct ShadowRayTriangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct ShadowRayMaterial { uint4 DiffuseAlpha,OpacityRed; float4 Coverage; };
constant float ShadowRayPixelCenter=0.5;
constant float ShadowRayBinary32Epsilon=1.1920928955078125e-7;
constant float ShadowRayEdgeError=3*ShadowRayBinary32Epsilon/(1-3*ShadowRayBinary32Epsilon);
ShadowRayNode ShadowRayReadNode(texture2d<uint,access::read> storage,uint index)
{
    uint base=ShadowRayLoad(storage,ShadowRayOffsetsHeader).x+index*ShadowRayNodeSize;
    ShadowRayNode result;
    result.Minimum=as_type<float4>(ShadowRayLoad(storage,base));
    result.Maximum=as_type<float4>(ShadowRayLoad(storage,base+1u));
    result.Links=ShadowRayLoad(storage,base+2u);
    return result;
}
ShadowRayTriangle ShadowRayReadTriangle(texture2d<uint,access::read> storage,uint index)
{
    uint base=ShadowRayLoad(storage,ShadowRayOffsetsHeader).y+index*ShadowRayTriangleSize;
    ShadowRayTriangle result;
    result.A=as_type<float4>(ShadowRayLoad(storage,base));
    result.B=as_type<float4>(ShadowRayLoad(storage,base+1u));
    result.C=as_type<float4>(ShadowRayLoad(storage,base+2u));
    result.UVAB=as_type<float4>(ShadowRayLoad(storage,base+3u));
    result.UVC=as_type<float4>(ShadowRayLoad(storage,base+4u));
    result.Metadata=ShadowRayLoad(storage,base+5u);
    return result;
}
ShadowRayMaterial ShadowRayReadMaterial(texture2d<uint,access::read> storage,uint index)
{
    uint base=ShadowRayLoad(storage,ShadowRayOffsetsHeader).z+index*ShadowRayMaterialSize;
    ShadowRayMaterial result;
    result.DiffuseAlpha=ShadowRayLoad(storage,base);
    result.OpacityRed=ShadowRayLoad(storage,base+1u);
    result.Coverage=as_type<float4>(ShadowRayLoad(storage,base+2u));
    return result;
}
float ShadowRayScalar(texture2d<uint,access::read> storage,uint index)
{ uint4 value=ShadowRayLoad(storage,ShadowRayLoad(storage,ShadowRayOffsetsHeader).w+index/ShadowRayScalarLanes); return as_type<float4>(value)[index%ShadowRayScalarLanes]; }
float ShadowRayCoverage(uint4 image,float2 uv,texture2d<uint,access::read> storage)
{
    if (!image.w) return 1;
    float2 texel=fract(float2(uv.x,-uv.y))*float2(image.yz)-ShadowRayPixelCenter;
    uint x=texel.x<0 ? image.y-1 : uint(floor(texel.x)),y=texel.y<0 ? image.z-1 : uint(floor(texel.y));
    uint nextX=x+1==image.y ? 0 : x+1,nextY=y+1==image.z ? 0 : y+1;
    float upper=mix(ShadowRayScalar(storage,image.x+y*image.y+x),ShadowRayScalar(storage,image.x+y*image.y+nextX),fract(texel.x));
    float lower=mix(ShadowRayScalar(storage,image.x+nextY*image.y+x),ShadowRayScalar(storage,image.x+nextY*image.y+nextX),fract(texel.x));
    return mix(upper,lower,fract(texel.y));
}
bool ShadowRayBoundsHit(ShadowRayNode node,float3 origin,float3 ray,float radius)
{
    float nearDistance=0,farDistance=radius;
    for (uint axis=0;axis<3;++axis)
    {
        if (ray[axis]==0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float first=(node.Minimum[axis]-origin[axis])/ray[axis],last=(node.Maximum[axis]-origin[axis])/ray[axis];
            nearDistance=max(nearDistance,min(first,last)); farDistance=min(farDistance,max(first,last));
            if (nearDistance>farDistance) return false;
        }
    }
    return true;
}
float ShadowRayEdge(float2 first,float2 second)
{
    float positive=first.x*second.y,negative=first.y*second.x,area=positive-negative;
    return abs(area)<=ShadowRayEdgeError*(abs(positive)+abs(negative)) ? 0 : area;
}
bool ShadowRayTriangleHit(ShadowRayTriangle triangle,float3 origin,float3 ray,float radius,thread float3& weights)
{
    weights=0; uint major=abs(ray.x)>abs(ray.y) ? 0 : 1; if (abs(ray.z)>abs(ray[major])) major=2;
    uint horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    float3 a=triangle.A.xyz-origin,b=triangle.B.xyz-origin,c=triangle.C.xyz-origin;
    float2 shear=float2(ray[horizontal],ray[vertical])/ray[major];
    float2 first=float2(a[horizontal],a[vertical])-shear*a[major];
    float2 second=float2(b[horizontal],b[vertical])-shear*b[major];
    float2 third=float2(c[horizontal],c[vertical])-shear*c[major];
    float3 edges=float3(ShadowRayEdge(second,third),ShadowRayEdge(third,first),ShadowRayEdge(first,second));
    if (any(edges<0) && any(edges>0)) return false;
    float area=edges.x+edges.y+edges.z; if (area==0) return false;
    float distance=dot(edges,float3(a[major],b[major],c[major]))/(area*ray[major]);
    if (!(distance>0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool ShadowRayAnyHit(float3 origin,float3 ray,float radius,uint nodeCount,texture2d<uint,access::read> storage)
{
    uint address=0;
    while (address<nodeCount)
    {
        ShadowRayNode node=ShadowRayReadNode(storage,address);
        if (!ShadowRayBoundsHit(node,origin,ray,radius)) { address=node.Links.w; continue; }
        if (!node.Links.z) { address=node.Links.x; continue; }
        for (uint local=0;local<node.Links.z;++local)
        {
            ShadowRayTriangle triangle=ShadowRayReadTriangle(storage,node.Links.y+local);
            if ((triangle.Metadata.w & ShadowRayCasterFlag)==0u) continue; float3 weights;
            if (!ShadowRayTriangleHit(triangle,origin,ray,radius,weights)) continue;
            ShadowRayMaterial material=ShadowRayReadMaterial(storage,triangle.Metadata.x); if (material.Coverage.y<=0) return true;
            float2 uv=triangle.UVAB.xy*weights.x+triangle.UVAB.zw*weights.y+triangle.UVC.xy*weights.z;
            float alpha=dot(weights,float3(triangle.A.w,triangle.B.w,triangle.C.w))*material.Coverage.x;
            alpha*=ShadowRayCoverage(material.DiffuseAlpha,uv,storage)*ShadowRayCoverage(material.OpacityRed,uv,storage);
            if (saturate(alpha)>=material.Coverage.y) return true;
        }
        address=node.Links.w;
    }
    return false;
}
constant float ShadowRayNormalFloor=1e-12,ShadowRayOriginRoundoff=8.0*1.1920928955078125e-7;
bool ShadowRayEnabled(texture2d<uint,access::read> storage) { return ShadowRayLoad(storage,ShadowRayCountsHeader).x!=0u && ShadowRayLoad(storage,ShadowRayLightHeader).y==ShadowRayImageABI; }
float ShadowRayVisibility(float3 position,float3 normal,float3 towardLight,texture2d<uint,access::read> image)
{
    if (!ShadowRayEnabled(image)) return 1.0;
    float4 parameters=as_type<float4>(ShadowRayLoad(image,ShadowRayTraceHeader));
    float scale=max(abs(towardLight.x),max(abs(towardLight.y),abs(towardLight.z)));
    if (!(scale>0.0) || !(all(isfinite(float4(position,scale))))) return 1.0;
    float3 scaled=towardLight/scale; float magnitude=length(scaled);
    float3 direction=scaled/magnitude;
    bool finiteEmitter=ShadowRayLoad(image,ShadowRayLightHeader).x!=0u;
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
    return ShadowRayAnyHit(origin,direction,limit,ShadowRayLoad(image,ShadowRayCountsHeader).y,image)?0.0:1.0;
}


// Karis' smooth inverse-square window; Filament, Punctual lights.
// The host canonicalizes range: zero is unbounded, positive is finite support.
float LightRangeWindow(float3 offset, float radius)
{
    if (radius <= 0.0f) return 1.0f;
    if (any(abs(offset) >= float3(radius))) return 0.0f;
    float3 normalizedOffset = offset / radius;
    float squaredFraction = dot(normalizedOffset, normalizedOffset);
    float window = max(1.0f - squaredFraction * squaredFraction, 0.0f);
    return window * window;
}


// Native Metal direct single scattering and associative transport prefix scan.
// Hillis and Steele, "Data Parallel Algorithms" (1986).
// HG phase convention: https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
// Primary opaque shadows are included. Light-path medium attenuation and
// multiple scattering are omitted; rectangle emitters use four-point quadrature.
constant uint VolumeMaximumLights = 4;
constant uint VolumeMaximumCascades = 4;
struct VolumeMatrix { float4 Rows[4]; };
struct VolumeMedium
{
    VolumeMatrix InverseCamera;
    float4 ExtinctionHeightFalloffRange;
    float4 AmbientSource;
    float4 ViewportGrid;
    uint4 StageSlicesSamplesEnabled;
};
struct VolumeLight
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorShadow;
    float4 AreaSize;
};
struct VolumeLighting
{
    float4 AlbedoAsymmetry;
    float4 IntensityShadowBias;
    uint4 CountShadowModeIndexStride;
    VolumeMatrix PrimaryCamera;
    VolumeLight Lights[VolumeMaximumLights];
    VolumeMatrix CascadeCameras[VolumeMaximumCascades];
    float4 CascadeEnds;
    float4 CascadeBlendStarts;
    float4 EyeDepthPlane;
    float4 CascadeRangeCount;
    float4 ShadowFilter;
    float4 ShadowEncoding[VolumeMaximumCascades + 1];
};
static_assert(sizeof(VolumeMedium) == 128, "Medium upload layout mismatch");
static_assert(sizeof(VolumeLighting) == 784, "Lighting upload layout mismatch");
struct VolumeVertex { float4 Position [[position]]; };

constant float VolumePi = 3.14159265358979323846;
constant float VolumeOpticalDepthCeiling = 80.0;
constant float VolumeRayEpsilon = 1.0e-6;
constant float VolumeSquaredLightEpsilon = 1.0e-4;
constant float VolumeProjectionEpsilon = 1.0e-7;
constant float VolumeHalfMaximum = 65504.0;
constant float VolumeParallelThreshold = 0.99;
constant uint VolumeDirectionalType = 0, VolumePointType = 1;
constant uint VolumeSingleShadow = 1, VolumeCascadedShadow = 2;
constant uint VolumeAreaSteps = 2;
constant float VolumeAreaWeight = 1.0 / float(VolumeAreaSteps * VolumeAreaSteps);

vertex VolumeVertex VolumeVertexMain(uint vertexIndex [[vertex_id]])
{
    const float FullscreenEdge = 3.0;
    float2 corner = vertexIndex == 1 ? float2(FullscreenEdge, -1.0) :
        vertexIndex == 2 ? float2(-1.0, FullscreenEdge) : float2(-1.0);
    return {float4(corner, 0.0, 1.0)};
}
float4 VolumeTransform(constant VolumeMatrix& matrix, float4 value)
{
    return float4(dot(matrix.Rows[0], value), dot(matrix.Rows[1], value),
        dot(matrix.Rows[2], value), dot(matrix.Rows[3], value));
}
float3 VolumeDirection(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > VolumeSquaredLightEpsilon ? value * rsqrt(lengthSquared) : fallback;
}
float3 VolumeWorld(float2 uv, float depth, constant VolumeMedium& medium)
{
    // Fragment positions and texture rows have a top-left origin. Explicit row
    // vectors preserve the host matrix layout and canonical [0,1] clip depth.
    float4 clip = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), depth, 1.0);
    float4 world = VolumeTransform(medium.InverseCamera, clip);
    return world.xyz / world.w;
}
float VolumePhase(float3 towardLight, float3 awayFromEye, float asymmetry)
{
    // Physical incoming and outgoing travel are the negatives of both inputs.
    // Their cosine thus peaks at +1 for forward scattering with positive g.
    float cosine = clamp(dot(towardLight, awayFromEye), -1.0, 1.0);
    float denominator = 1.0 + asymmetry * asymmetry - 2.0 * asymmetry * cosine;
    return (1.0 - asymmetry * asymmetry) / (4.0 * VolumePi * denominator * sqrt(denominator));
}
#define MediumShadowTexture depth2d<float, access::read>
float ReadMediumDepth(MediumShadowTexture image, uint2 pixel) { return image.read(pixel); }
float VolumeCompare(depth2d<float, access::read> image, int2 pixel, float receiver)
{
    int2 extent = int2(image.get_width(), image.get_height());
    if (any(pixel < int2(0)) || any(pixel >= extent)) return 1.0;
    return receiver <= image.read(uint2(pixel)) ? 1.0 : 0.0;
}
float VolumeBilinearComparison(depth2d<float, access::read> image, float2 uv, float receiver)
{
    float2 location = uv * float2(image.get_width(), image.get_height()) - 0.5;
    int2 corner = int2(floor(location));
    float2 weight = fract(location);
    float firstRow = mix(VolumeCompare(image, corner, receiver), VolumeCompare(image, corner + int2(1, 0), receiver), weight.x);
    float secondRow = mix(VolumeCompare(image, corner + int2(0, 1), receiver), VolumeCompare(image, corner + int2(1, 1), receiver), weight.x);
    return mix(firstRow, secondRow, weight.y);
}
float VolumeShadowMap(depth2d<float, access::read> image, float4 clip, constant VolumeLighting& lighting)
{
    if (clip.w <= VolumeProjectionEpsilon) return 1.0;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 || any(uv < float2(0.0)) || any(uv > float2(1.0))) return 1.0;
    int2 extent = int2(image.get_width(), image.get_height());
    float receiver = projected.z - lighting.IntensityShadowBias.y;
    int radius = int(lighting.ShadowFilter.x);
    if (radius == 0) return VolumeCompare(image, min(int2(uv * float2(extent)), extent - 1), receiver);
    float visibility = 0.0;
    for (int row = -radius; row <= radius; ++row)
        for (int column = -radius; column <= radius; ++column)
            visibility += VolumeBilinearComparison(image, uv + float2(column, row) / float2(extent), receiver);
    return visibility / lighting.ShadowFilter.y;
}
float VolumeCascade(uint index, float3 world, constant VolumeLighting& lighting,
    depth2d<float, access::read> first, depth2d<float, access::read> second,
    depth2d<float, access::read> third, depth2d<float, access::read> fourth)
{
    float4 clip = VolumeTransform(lighting.CascadeCameras[index], float4(world, 1.0));
    if (index == 0) return VolumeShadowMap(first, clip, lighting);
    if (index == 1) return VolumeShadowMap(second, clip, lighting);
    if (index == 2) return VolumeShadowMap(third, clip, lighting);
    return VolumeShadowMap(fourth, clip, lighting);
}
// Virtual depth and page directory share the exclusive cascade slots.
// Each PCF corner resolves its own page or pinned coarse ancestor.
constant uint VirtualTileBorder = 1;
constant float VirtualSampleCenter = 0.5;
float2 VirtualDepth(float2 uv, uint firstMip, constant VolumeLighting& frame,
    MediumShadowTexture atlas, MediumShadowTexture directory)
{
    uint4 layout = uint4(frame.CascadeEnds);
    uint4 physical = uint4(frame.CascadeBlendStarts);
    uint side = layout.x / layout.y, offset = 0;
    for (uint mip = 0; mip < layout.z; ++mip)
    {
        if (mip >= firstMip)
        {
            uint resolution = side * layout.y;
            uint2 pixel = min(uint2(uv * resolution), uint2(resolution - 1)), page = pixel / layout.y;
            uint address = offset + page.y * side + page.x;
            uint entry = uint(round(ReadMediumDepth(directory, uint2(address % layout.w, address / layout.w)) * physical.w));
            if (entry != 0)
            {
                uint slot = entry - 1;
                uint2 tile(slot % physical.z, slot / physical.z);
                uint2 at = tile * (layout.y + 2 * VirtualTileBorder) + VirtualTileBorder + pixel % layout.y;
                return float2(ReadMediumDepth(atlas, at), float(mip));
            }
        }
        offset += side * side; side /= 2;
    }
    return float2(1, layout.z - 1);
}
float VirtualCompare(int2 pixel, uint mip, float receiver, constant VolumeLighting& frame,
    MediumShadowTexture atlas, MediumShadowTexture directory)
{
    uint resolution = uint(frame.CascadeEnds.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0), int2(resolution - 1))) + VirtualSampleCenter) / resolution;
    return receiver <= VirtualDepth(uv, mip, frame, atlas, directory).x ? 1.0 : 0.0;
}
// Camera rows in the unused filter suffix define an explicit ray footprint.
uint VirtualVolumeMip(float3 position, float2 uv, constant VolumeLighting& frame, constant VolumeMedium& medium)
{
    float4 world(position, 1);
    float4 camera(dot(frame.ShadowEncoding[0], world), dot(frame.ShadowEncoding[1], world), dot(frame.ShadowEncoding[2], world), dot(frame.ShadowEncoding[3], world));
    if (camera.w <= frame.EyeDepthPlane.w) return 0;
    float3 ndc = camera.xyz / camera.w;
    float4 x = VolumeTransform(medium.InverseCamera, float4(ndc + float3(frame.EyeDepthPlane.y * 2, 0, 0), 1));
    float4 y = VolumeTransform(medium.InverseCamera, float4(ndc - float3(0, frame.EyeDepthPlane.z * 2, 0), 1));
    float4 lightX = VolumeTransform(frame.CascadeCameras[0], float4(x.xyz / x.w, 1));
    float4 lightY = VolumeTransform(frame.CascadeCameras[0], float4(y.xyz / y.w, 1));
    if (min(lightX.w, lightY.w) <= frame.EyeDepthPlane.w) return uint(frame.CascadeEnds.z - 1);
    float2 uvX = float2(lightX.x / lightX.w + 1, 1 - lightX.y / lightX.w) * VirtualSampleCenter;
    float2 uvY = float2(lightY.x / lightY.w + 1, 1 - lightY.y / lightY.w) * VirtualSampleCenter;
    float footprint = max(length(uvX - uv), length(uvY - uv)) * frame.CascadeEnds.x;
    return uint(clamp(floor(log2(max(1.0, footprint)) + frame.EyeDepthPlane.x), 0.0, frame.CascadeEnds.z - 1.0));
}

float VirtualVisibility(float3 position, float bias, constant VolumeLighting& frame, constant VolumeMedium& medium,
    MediumShadowTexture atlas, MediumShadowTexture directory)
{
    float4 world(position, 1);
    float4 clip = VolumeTransform(frame.CascadeCameras[0], world);
    float3 projected = clip.xyz / max(clip.w, frame.EyeDepthPlane.w);
    float2 uv(projected.x * VirtualSampleCenter + VirtualSampleCenter, VirtualSampleCenter - projected.y * VirtualSampleCenter);
    uint mip = VirtualVolumeMip(position, uv, frame, medium);
    if (clip.w <= frame.EyeDepthPlane.w || !all(uv >= 0.0) || !all(uv <= 1.0) || projected.z < 0 || projected.z > 1) return 1;
    float2 stored = VirtualDepth(uv, mip, frame, atlas, directory); float receiver = projected.z - bias;
    if (frame.ShadowFilter.x == 0) return receiver <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(frame.CascadeEnds.x) >> mip) - VirtualSampleCenter;
    int2 base = int2(floor(texel)); float2 fraction = fract(texel); float visible = 0;
    for (int y = -int(frame.ShadowFilter.x); y <= int(frame.ShadowFilter.x); ++y)
        for (int x = -int(frame.ShadowFilter.x); x <= int(frame.ShadowFilter.x); ++x)
        {
            int2 p = base + int2(x, y);
            float a = VirtualCompare(p, mip, receiver, frame, atlas, directory);
            float b = VirtualCompare(p + int2(1, 0), mip, receiver, frame, atlas, directory);
            float c = VirtualCompare(p + int2(0, 1), mip, receiver, frame, atlas, directory);
            float d = VirtualCompare(p + 1, mip, receiver, frame, atlas, directory);
            visible += mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
        }
    return visible / frame.ShadowFilter.y;
}


float VolumeVisibility(float3 world, constant VolumeLighting& lighting, constant VolumeMedium& medium, depth2d<float, access::read> primary,
    depth2d<float, access::read> first, depth2d<float, access::read> second,
    depth2d<float, access::read> third, depth2d<float, access::read> fourth, texture2d<uint,access::read> shadowRayImage)
{
    if (ShadowRayEnabled(shadowRayImage)) return 1.0;
    if (lighting.ShadowFilter.z != 0 && lighting.CountShadowModeIndexStride.y != 0) return VirtualVisibility(world, lighting.IntensityShadowBias.y, lighting, medium, first, second);
    uint mode = lighting.CountShadowModeIndexStride.y;
    if (mode == VolumeSingleShadow)
        return VolumeShadowMap(primary, VolumeTransform(lighting.PrimaryCamera, float4(world, 1.0)), lighting);
    if (mode != VolumeCascadedShadow) return 1.0;
    float eyeDepth = dot(lighting.EyeDepthPlane, float4(world, 1.0));
    if (eyeDepth < lighting.CascadeRangeCount.x || eyeDepth > lighting.CascadeRangeCount.y) return 1.0;
    uint cascade = 0, count = uint(lighting.CascadeRangeCount.z);
    while (cascade + 1 < count && eyeDepth > lighting.CascadeEnds[cascade]) ++cascade;
    float visibility = VolumeCascade(cascade, world, lighting, first, second, third, fourth);
    if (cascade + 1 < count && eyeDepth > lighting.CascadeBlendStarts[cascade])
    {
        float overlap = max(lighting.CascadeEnds[cascade] - lighting.CascadeBlendStarts[cascade], VolumeRayEpsilon);
        float blend = saturate((eyeDepth - lighting.CascadeBlendStarts[cascade]) / overlap);
        visibility = mix(visibility, VolumeCascade(cascade + 1, world, lighting, first, second, third, fourth), blend);
    }
    return visibility;
}
float VolumeEmitter(VolumeLight light, float3 world, float3 cameraRay, float asymmetry)
{
    uint type = uint(light.PositionType.w);
    float3 normal = VolumeDirection(light.DirectionIntensity.xyz, float3(0.0, 1.0, 0.0));
    if (type == VolumeDirectionalType) return VolumePhase(normal, cameraRay, asymmetry);
    if (type == VolumePointType)
    {
        float3 delta = light.PositionType.xyz - world;
        return VolumePhase(VolumeDirection(delta, normal), cameraRay, asymmetry) * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), VolumeSquaredLightEpsilon);
    }
    float3 helper = abs(normal.y) < VolumeParallelThreshold ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
    float integral = 0.0;
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSize.w > 0.0;
    uint emitterGrid = physicalEmitter ? uint(clamp(light.AreaSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : VolumeAreaSteps;
    float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
    for (uint row = 0; row < emitterGrid; ++row)
        for (uint column = 0; column < emitterGrid; ++column)
        {
            float2 offset = ((float2(column, row) + 0.5) / float(emitterGrid) - 0.5) * light.AreaSize.xy;
            float3 delta = light.PositionType.xyz + tangent * offset.x + bitangent * offset.y - world;
            float3 towardLight = VolumeDirection(delta, -normal);
            integral += emitterSampleWeight * saturate(dot(normal, -towardLight)) * VolumePhase(towardLight, cameraRay, asymmetry) * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), VolumeSquaredLightEpsilon);
        }
    return integral * light.AreaSize.x * light.AreaSize.y;
}
float3 VolumeSource(float3 world, float3 cameraRay, constant VolumeMedium& medium, constant VolumeLighting& lighting, device const VolumeLight* sceneLights,
    depth2d<float, access::read> primary, depth2d<float, access::read> first,
    depth2d<float, access::read> second, depth2d<float, access::read> third, depth2d<float, access::read> fourth, texture2d<uint,access::read> shadowRayImage)
{
    float visibility = VolumeVisibility(world, lighting, medium, primary, first, second, third, fourth, shadowRayImage);
    float3 direct = float3(0.0);
    for (uint index = 0; index < lighting.CountShadowModeIndexStride.x; ++index)
    {
        VolumeLight light;
        if (lighting.IntensityShadowBias.z != 0.0) light = sceneLights[index];
        else light = lighting.Lights[index];
        float3 radiance = min(light.ColorShadow.rgb, float3(VolumeHalfMaximum)) * min(light.DirectionIntensity.w, VolumeHalfMaximum);
        float3 rayToward=uint(light.PositionType.w)==0u?light.DirectionIntensity.xyz:light.PositionType.xyz-world;
        float shadow=lighting.CountShadowModeIndexStride.y!=0 && index==lighting.CountShadowModeIndexStride.z?(ShadowRayEnabled(shadowRayImage)?ShadowRayVisibility(world,float3(0),rayToward,shadowRayImage):visibility):1.0;
        direct += radiance * (VolumeEmitter(light, world, cameraRay, lighting.AlbedoAsymmetry.w) * shadow);
    }
    return medium.AmbientSource.rgb + lighting.AlbedoAsymmetry.rgb * lighting.IntensityShadowBias.x * direct;
}
float4 VolumeSegment(uint2 pixel, constant VolumeMedium& medium, constant VolumeLighting& lighting, device const VolumeLight* sceneLights,
    depth2d<float, access::read> primary, depth2d<float, access::read> first,
    depth2d<float, access::read> second, depth2d<float, access::read> third, depth2d<float, access::read> fourth, texture2d<uint,access::read> shadowRayImage)
{
    uint columns = uint(medium.ViewportGrid.z), slice = pixel.x / columns;
    float2 uv = (float2(pixel.x % columns, pixel.y) + 0.5) / medium.ViewportGrid.zw;
    float3 origin = VolumeWorld(uv, 0.0, medium), ray = VolumeWorld(uv, 1.0, medium) - origin;
    float lengthToFar = max(length(ray), VolumeRayEpsilon);
    float3 direction = ray / lengthToFar;
    float stepLength = min(lengthToFar, medium.ExtinctionHeightFalloffRange.w) /
        float(medium.StageSlicesSamplesEnabled.y * medium.StageSlicesSamplesEnabled.z);
    float4 segment = float4(0.0, 0.0, 0.0, 1.0);
    for (uint stepIndex = 0; stepIndex < medium.StageSlicesSamplesEnabled.z; ++stepIndex)
    {
        float distance = (float(slice * medium.StageSlicesSamplesEnabled.z + stepIndex) + 0.5) * stepLength;
        float3 world = origin + direction * distance;
        float heightExponent = min(medium.ExtinctionHeightFalloffRange.z * max(world.y - medium.ExtinctionHeightFalloffRange.y, 0.0), VolumeOpticalDepthCeiling);
        float sigmaT = medium.ExtinctionHeightFalloffRange.x * exp(-heightExponent);
        float transmission = exp(-min(sigmaT * stepLength, VolumeOpticalDepthCeiling));
        segment.rgb += segment.a * (1.0 - transmission) * VolumeSource(world, direction, medium, lighting, sceneLights, primary, first, second, third, fourth, shadowRayImage);
        segment.a *= transmission;
    }
    return segment;
}
float4 VolumePrefix(uint2 pixel, constant VolumeMedium& medium, constant VolumeLighting& lighting,
    texture2d<float, access::read> previous)
{
    uint columns = uint(medium.ViewportGrid.z), slice = pixel.x / columns;
    float4 prefix = previous.read(pixel);
    uint stride = lighting.CountShadowModeIndexStride.w;
    if (slice >= stride)
    {
        float4 left = previous.read(pixel - uint2(stride * columns, 0));
        prefix = float4(left.rgb + left.a * prefix.rgb, left.a * prefix.a);
    }
    // Intermediate attachments preserve FP32 transport. Clamp all slices only
    // when the terminal scan stores the shared cumulative RGBA16F atlas.
    if (stride * 2 >= medium.StageSlicesSamplesEnabled.y) prefix.rgb = min(prefix.rgb, float3(VolumeHalfMaximum));
    return prefix;
}
fragment float4 VolumeFragmentMain(VolumeVertex input [[stage_in]],
    constant VolumeMedium& medium [[buffer(0)]], constant VolumeLighting& lighting [[buffer(1)]],
    device const VolumeLight* sceneLights [[buffer(8)]], // Complete scene; no opaque-depth list.
    texture2d<float, access::read> previous [[texture(0)]], depth2d<float, access::read> primary [[texture(1)]],
    depth2d<float, access::read> first [[texture(2)]], depth2d<float, access::read> second [[texture(3)]],
    depth2d<float, access::read> third [[texture(4)]], depth2d<float, access::read> fourth [[texture(5)]], texture2d<uint,access::read> shadowRayImage [[texture(20)]])
{
    uint2 pixel = uint2(input.Position.xy);
    if (lighting.CountShadowModeIndexStride.w == 0)
        return VolumeSegment(pixel, medium, lighting, sceneLights, primary, first, second, third, fourth, shadowRayImage);
    return VolumePrefix(pixel, medium, lighting, previous);
}
