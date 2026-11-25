#include <metal_stdlib>
using namespace metal;
// Anime environment material palette and diffuse ramp. Native MSL evaluation
// keeps radiance, visibility and alpha separate. Cai, GDC 2021,
// "Genshin Impact: Crafting an Anime Style Open World" motivates the art target.
constant float ToonShadingModel = 4.0;
constant float ToonAlbedoExponent = 0.8;
constant float ToonPaletteSteps = 16.0;
constant float ToonColorSaturation = 1.12;
constant float ToonNormalStrength = 0.12;
constant float ToonLightScale = 0.45;
constant float ToonBandSoftness = 0.4;
constant float ToonShadowFloor = 0.18;
constant float ToonHighlightThreshold = 0.8;
constant float ToonHighlightStrength = 0.18;
constant float3 ToonLuma = float3(0.2126, 0.7152, 0.0722);
constant float3 ToonCoolShadow = float3(0.56, 0.68, 1.0);
constant float3 ToonWarmLight = float3(1.0, 0.94, 0.82);
constant float3 ToonSkyFill = float3(0.24, 0.29, 0.4);
float3 ToonAlbedo(float3 value, float model)
{
    if (model != ToonShadingModel) return value;
    float3 paint = pow(saturate(value), float3(ToonAlbedoExponent));
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return saturate(mix(float3(dot(paint, ToonLuma)), paint, ToonColorSaturation));
}
float3 ToonAmbient(float3 albedo, float3 ambient, float model)
{ return model == ToonShadingModel ? max(ambient, albedo * ToonSkyFill) : ambient; }
float3 ToonResponse(float cosine, float bands)
{
    float angularBand = saturate(cosine) * bands;
    float transition = smoothstep(1.0 - ToonBandSoftness, 1.0, fract(angularBand));
    float level = min(floor(angularBand) + transition, bands - 1.0) / (bands - 1.0);
    return mix(ToonCoolShadow, ToonWarmLight, level) * mix(ToonShadowFloor, 1.0, level) * ToonLightScale;
}
float ToonHighlight(float value)
{ return step(ToonHighlightThreshold, value) * ToonHighlightStrength; }

// Smooth only the texture color. Explicit gradients and the center alpha
// preserve the original material's mip selection and silhouette coverage.
constant float ToonTextureRadius = 1.25;
constant int ToonTextureKernelRadius = 1;
constant int ToonTextureKernelWidth = ToonTextureKernelRadius * 2 + 1;
constant float ToonTextureSampleWeight = 1.0 / (ToonTextureKernelWidth * ToonTextureKernelWidth);
float3 ToonTextureColor(texture2d<float> image, sampler surfaceSampler,
    float2 uv, float2 uvDx, float2 uvDy)
{
    float3 color = float3(0.0);
    for (int row = -ToonTextureKernelRadius; row <= ToonTextureKernelRadius; ++row)
        for (int column = -ToonTextureKernelRadius; column <= ToonTextureKernelRadius; ++column)
            color += image.sample(surfaceSampler, uv + ToonTextureRadius * (column * uvDx + row * uvDy), gradient2d(uvDx, uvDy)).rgb;
    return color * ToonTextureSampleWeight;
}

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

// Independent Metal receiver for the Lumen froxel SH atlas. Fixed angular
// samples make the low-frequency volume stable without borrowing opaque history.
struct LumenVolumeParameters { float4 Camera[4],Inverse[4]; uint4 Grid; float4 Distribution,Atlas; };
float4 lumenVolumeProject(constant float4* rows,float4 value)
{ return float4(dot(rows[0],value),dot(rows[1],value),dot(rows[2],value),dot(rows[3],value)); }
constant float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 point,float3 direction,
    constant LumenVolumeParameters& volume,texture2d<float> image)
{
    if (volume.Grid.w==0) return 0;
    float4 projected=lumenVolumeProject(volume.Camera,float4(point,1));
    if (projected.w<=volume.Atlas.z) return 0;
    float3 ndc=projected.xyz/projected.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return 0;
    float4 nearH=lumenVolumeProject(volume.Inverse,float4(ndc.xy,0,1)),farH=lumenVolumeProject(volume.Inverse,float4(ndc.xy,1,1));
    if (min(abs(nearH.w),abs(farH.w))<=volume.Atlas.z) return 0;
    float3 origin=nearH.xyz/nearH.w;
    float extent=min(length(farH.xyz/farH.w-origin),volume.Distribution.x),travel=length(point-origin);
    if (extent<=volume.Atlas.z || travel>extent) return 0;
    float2 uv=float2(ndc.x+1,1-ndc.y)*0.5;
    float3 cell=float3(uv*float2(volume.Grid.xy),pow(saturate(travel/extent),1/volume.Distribution.y)*volume.Grid.z)-0.5;
    cell=clamp(cell,float3(0),float3(volume.Grid.xyz)-1);
    float lower=floor(cell.z),upper=min(lower+1,float(volume.Grid.z-1));
    float3 n=direction*rsqrt(max(dot(direction,direction),volume.Atlas.w));
    constexpr float SHPi=3.14159265358979323846;
    constexpr uint SHCount=9;
    float basis[SHCount]={sqrt(1/(4*SHPi)),sqrt(3/(4*SHPi))*n.y,sqrt(3/(4*SHPi))*n.z,sqrt(3/(4*SHPi))*n.x,
        sqrt(15/(4*SHPi))*n.x*n.y,sqrt(15/(4*SHPi))*n.y*n.z,sqrt(5/(16*SHPi))*(3*n.z*n.z-1),
        sqrt(15/(4*SHPi))*n.x*n.z,sqrt(15/(16*SHPi))*(n.x*n.x-n.y*n.y)};
    constexpr sampler volumeFilter(coord::normalized,address::clamp_to_edge,filter::linear);
    float3 sum=0;

    constexpr uint VolumeVisibilityBit=2;
    if ((volume.Grid.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(cell)); float3 fraction=fract(cell);
        // A corner may contribute only inside its surface-free support sphere.
        // Preserve the original weights so rejected corners cannot add energy.
        for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,volume.Grid.xyz-1);
            float weight=(bit.x?fraction.x:1-fraction.x)*(bit.y?fraction.y:1-fraction.y)*(bit.z?fraction.z:1-fraction.z); if (weight<=0) continue;
            uint row=p.z*volume.Grid.y+p.y; float support=image.read(uint2(p.x,row)).w; if (support<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5f)/volume.Grid.x-1,1-2*(float(p.y)+0.5f)/volume.Grid.y);
            float4 a=lumenVolumeProject(volume.Inverse,float4(clipXY,0,1)),b=lumenVolumeProject(volume.Inverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=volume.Atlas.z) continue;
            float3 begin=a.xyz/a.w,axis=b.xyz/b.w-begin; float span=length(axis);
            if (!isfinite(span) || span<=volume.Atlas.z) continue;
            float depth=pow((float(p.z)+0.5f)/volume.Grid.z,volume.Distribution.y);
            float3 probe=begin+axis*(depth*min(span,volume.Distribution.x)/span); float separation=length(point-probe);
            if (!isfinite(separation) || separation>support) continue;
            for (uint coefficient=0;coefficient<SHCount;++coefficient)
                sum+=weight*basis[coefficient]*image.read(uint2(p.x+coefficient*volume.Grid.x,row)).xyz;
        }
        return max(sum,0.0f)*volume.Distribution.z/(SHPi*volume.Distribution.w);
    }
    for (uint coefficient=0;coefficient<SHCount;++coefficient)
    {
        float x=coefficient*volume.Grid.x+cell.x+0.5;
        float2 a=float2(x,lower*volume.Grid.y+cell.y+0.5)/volume.Atlas.xy;
        float2 b=float2(x,upper*volume.Grid.y+cell.y+0.5)/volume.Atlas.xy;
        sum+=basis[coefficient]*mix(image.sample(volumeFilter,a,level(0)).xyz,image.sample(volumeFilter,b,level(0)).xyz,cell.z-lower);
    }
    return max(sum,0.0f)*volume.Distribution.z/(SHPi*volume.Distribution.w);
}


// Metal 2.1 adds the attribute; guaranteed cross-pass invariance also needs
// the runtime preserveInvariance option (macOS 11 / iOS 14 or newer).
#if __METAL_VERSION__ >= 210
#define EVANS_POSITION [[position, invariant]]
#else
#define EVANS_POSITION [[position]]
#endif

// Native Metal Forward+ surfaces and alpha-tested depth. Device buffers retain
// full scene order; a constant selection block chooses opaque, transparent or
// complete-scene traversal. Overflow restores the complete loop.
#if defined(EVANS_FORWARD_PLUS_SURFACE)
#ifndef EVANS_FORWARD_PLUS_FILTER
#define EVANS_FORWARD_PLUS_FILTER 0
#endif
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

// Contact variants read the matching opaque sample; inactive draws retain their
// ordinary entry points. MSL keeps material varyings at pixel-centre interpolation.
#ifndef EVANS_CONTACT_SHADOWS
#define EVANS_CONTACT_SHADOWS 0
#endif
#ifndef EVANS_CONTACT_MULTISAMPLE
#define EVANS_CONTACT_MULTISAMPLE 0
#endif

#if EVANS_FORWARD_PLUS_FILTER == 3
constant uint MomentProjectionCount = 5;
struct VSMConstants
{
    float4 DepthEncoding[MomentProjectionCount];
    float4 Parameters; // variance floor, bleeding reduction, derivative cap, sigma
    uint4 Control; // valid projection mask, radius, reserved, filter axis
};
static_assert(sizeof(VSMConstants) == 112, "VSM upload layout mismatch");

float2 ReadMoment(texture2d<float, access::read> image, int2 pixel)
{
    if (any(pixel < int2(0)) || any(pixel >= int2(image.get_width(), image.get_height()))) return float2(1.0);
    return image.read(uint2(pixel)).rg;
}
float2 BilinearMoments(texture2d<float, access::read> image, float2 uv)
{
    const float TexelCenter = 0.5;
    float2 center = uv * float2(image.get_width(), image.get_height()) - TexelCenter;
    int2 corner = int2(floor(center)); float2 weight = fract(center);
    float2 upper = mix(ReadMoment(image, corner), ReadMoment(image, corner + int2(1, 0)), weight.x);
    float2 lower = mix(ReadMoment(image, corner + int2(0, 1)), ReadMoment(image, corner + int2(1)), weight.x);
    return mix(upper, lower, weight.y);
}
float PcssMap(texture2d<float, access::read> image, uint projectionIndex, float4 clip,
    float comparisonBias, constant VSMConstants& vsm)
{
    const float ProjectionTolerance = 1.0e-7;
    if ((vsm.Control.x & (1u << projectionIndex)) == 0u || clip.w <= ProjectionTolerance) return 1.0;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 || any(uv < 0.0) || any(uv > 1.0)) return 1.0;
    float depth = dot(vsm.DepthEncoding[projectionIndex].xyz, float3(clip.zw, 1.0));
    float2 moments = BilinearMoments(image, uv);
    float difference = depth - comparisonBias - moments.x;
    if (difference <= 0.0) return 1.0;
    float variance = max(moments.y - moments.x * moments.x, vsm.Parameters.x);
    float bound = variance / (variance + difference * difference);
    // This artistic remap suppresses bleeding but sacrifices the upper bound.
    return saturate((bound - vsm.Parameters.y) / (1.0 - vsm.Parameters.y));
}

using ForwardFilterConstants = VSMConstants;
using ForwardShadowImage = texture2d<float, access::read>;
#else
constant uint PCSSProjectionCount = 5;
constant float PCSSPi = 3.14159265358979323846;
constant float PCSSProjectionTolerance = 1.0e-7;
constant float PCSSMapFractionLimit = 0.25;
struct PCSSConstants
{
    float4 ProjectionMetrics[PCSSProjectionCount]; // near, far, width, height
    float4 EmitterAndLimits; // finite radius, tan(angular radius), search/filter caps
    uint4 SamplesAndMaps; // blocker count, filter count, valid projection mask
};
static_assert(sizeof(PCSSConstants) == 112, "PCSS upload layout mismatch");

#if EVANS_FORWARD_PLUS_FILTER == 2
float2 SearchDiskPoint(uint sampleIndex, uint count)
{
    const float GoldenAngle = PCSSPi * (3.0 - sqrt(5.0));
    float angle = GoldenAngle * float(sampleIndex);
    float radialDistance = sqrt((float(sampleIndex) + 0.5) / float(count));
    return float2(cos(angle), sin(angle)) * radialDistance;
}
float ProjectorDistance(float normalizedDepth, float4 metric)
{
    if (metric.x == 0.0) return normalizedDepth * metric.y;
    return metric.x * metric.y / (metric.y - normalizedDepth * (metric.y - metric.x));
}
float2 CapShadowFootprint(float2 uvRadius, uint2 dimensions, float requestedTexels)
{
    float longestAxis = max(uvRadius.x * dimensions.x, uvRadius.y * dimensions.y);
    float texelLimit = min(requestedTexels, float(min(dimensions.x, dimensions.y)) * PCSSMapFractionLimit);
    return longestAxis > texelLimit ? uvRadius * (texelLimit / longestAxis) : uvRadius;
}
float ReadComparison(depth2d<float, access::read> image, int2 pixel, float receiver)
{
    if (any(pixel < int2(0)) || any(pixel >= int2(image.get_width(), image.get_height()))) return 1.0;
    return receiver <= image.read(uint2(pixel)) ? 1.0 : 0.0;
}
float FilterComparison(depth2d<float, access::read> image, float2 uv, float receiver)
{
    float2 center = uv * float2(image.get_width(), image.get_height()) - 0.5;
    int2 corner = int2(floor(center));
    float2 weight = fract(center);
    float firstRow = mix(ReadComparison(image, corner, receiver), ReadComparison(image, corner + int2(1, 0), receiver), weight.x);
    float secondRow = mix(ReadComparison(image, corner + int2(0, 1), receiver), ReadComparison(image, corner + int2(1, 1), receiver), weight.x);
    return mix(firstRow, secondRow, weight.y);
}
float PcssMap(depth2d<float, access::read> image, uint projectionIndex, float4 clip,
    float comparisonBias, constant PCSSConstants& pcss)
{
    if ((pcss.SamplesAndMaps.z & (1u << projectionIndex)) == 0u || clip.w <= PCSSProjectionTolerance) return 1.0;
    float3 projected = clip.xyz / clip.w;
    float2 coordinate = projected.xy * float2(0.5, -0.5) + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 || any(coordinate < 0.0) || any(coordinate > 1.0)) return 1.0;
    uint2 dimensions(image.get_width(), image.get_height());
    float4 metric = pcss.ProjectionMetrics[projectionIndex];
    bool finiteProjector = metric.x > 0.0;
    float emitter = finiteProjector ? pcss.EmitterAndLimits.x : pcss.EmitterAndLimits.y;
    float comparisonDepth = projected.z - comparisonBias;
    if (emitter == 0.0)
        return ReadComparison(image, min(int2(coordinate * float2(dimensions)), int2(dimensions) - 1), comparisonDepth);
    // Only the classifier uses bias. Unbiased linear distances determine the
    // search footprint and contact-hardening geometry, even near the emitter.
    float receiverDistance = ProjectorDistance(projected.z, metric);
    if (receiverDistance <= 0.0) return 1.0;
    float searchWidth = finiteProjector ? emitter * max(receiverDistance - metric.x, 0.0) / receiverDistance
        : emitter * receiverDistance;
    float2 searchRadius = CapShadowFootprint(searchWidth / metric.zw, dimensions, pcss.EmitterAndLimits.z);
    float distanceSum = 0.0;
    uint blockerCount = 0;
    for (uint sampleIndex = 0; sampleIndex < pcss.SamplesAndMaps.x; ++sampleIndex)
    {
        float2 uv = coordinate + SearchDiskPoint(sampleIndex, pcss.SamplesAndMaps.x) * searchRadius;
        int2 pixel = int2(floor(uv * float2(dimensions)));
        if (any(pixel < int2(0)) || any(pixel >= int2(dimensions))) continue;
        float blockerDepth = image.read(uint2(pixel));
        if (blockerDepth < comparisonDepth)
        {
            distanceSum += ProjectorDistance(blockerDepth, metric);
            ++blockerCount;
        }
    }
    if (blockerCount == 0) return 1.0;
    float blockerDistance = distanceSum / float(blockerCount);
    float width = emitter * max(receiverDistance - blockerDistance, 0.0);
    if (finiteProjector) width *= metric.x / (blockerDistance * receiverDistance);
    // Uniform scaling preserves the ellipse of asymmetric projector dimensions.
    float2 filterRadius = CapShadowFootprint(width / metric.zw, dimensions, pcss.EmitterAndLimits.w);
    float sum = 0.0;
    for (uint sampleIndex = 0; sampleIndex < pcss.SamplesAndMaps.y; ++sampleIndex)
        sum += FilterComparison(image, coordinate + SearchDiskPoint(sampleIndex, pcss.SamplesAndMaps.y) * filterRadius, comparisonDepth);
    return sum / float(pcss.SamplesAndMaps.y);
}

#else
// Fixed filters encode radius/count in EmitterAndLimits.xy in this variant.
float FixedComparison(depth2d<float, access::read> image, int2 pixel, float receiver)
{
    int2 last = int2(image.get_width(), image.get_height()) - 1;
    return receiver <= image.read(uint2(clamp(pixel, int2(0), last))) ? 1.0 : 0.0;
}
float PcssMap(depth2d<float, access::read> image, uint projectionIndex, float4 clip,
    float bias, constant PCSSConstants& filtering)
{
    const float PixelCenter = 0.5;
    if (clip.w <= PCSSProjectionTolerance) return 1.0;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 || any(uv < 0.0) || any(uv > 1.0)) return 1.0;
    float2 extent = float2(image.get_width(), image.get_height());
    float receiver = projected.z - bias;
    int radius = int(filtering.EmitterAndLimits.x);
    if (radius == 0) return FixedComparison(image, int2(uv * extent), receiver);
    float2 texel = uv * extent - PixelCenter, fraction = fract(texel);
    int2 cell = int2(floor(texel)); float visible = 0.0;
    for (int row = -radius; row <= radius; ++row)
        for (int column = -radius; column <= radius; ++column)
        {
            int2 p = cell + int2(column, row);
            float top = mix(FixedComparison(image, p, receiver), FixedComparison(image, p + int2(1, 0), receiver), fraction.x);
            float bottom = mix(FixedComparison(image, p + int2(0, 1), receiver), FixedComparison(image, p + 1, receiver), fraction.x);
            visible += mix(top, bottom, fraction.y);
        }
    return visible / filtering.EmitterAndLimits.y;
}
#endif
using ForwardFilterConstants = PCSSConstants;
using ForwardShadowImage = depth2d<float, access::read>;
#endif
constant uint MaximumLights = 4;

constant float ShadowStrength = 0.55;
constant float Pi = 3.141592653589793;
constant float MinimumRoughness = 0.045;
constant float MinimumCosine = 0.0001;
constant float DielectricReflectance = 0.04;
constant float SrgbLinearScale = 12.92;
constant float SrgbPowerOffset = 0.055;
constant float SrgbPowerScale = 1.055;
constant float SrgbExponent = 2.4;
constant float SrgbLinearThreshold = 0.04045;
constant float NormalizeEpsilon = 1.0e-12;
constant float HomogeneousEpsilon = 1.0e-7;
constant float UvDeterminantEpsilon = 1.0e-10;
constant float DistanceEpsilon = 1.0e-6;
constant float ParallelAxisThreshold = 0.999;
constant float OutlineWorldScale = 0.01;
constant float MinimumShadowCosine = 0.15;
constant float MaximumReceiverBias = 0.0045;
constant float ReceiverBias = 0.00035;
constant float ReceiverSlopeBias = 0.00060;
constant float AreaSampleWeight = 0.25;
constant float OutlineMode = 1.5;
constant int DirectionalLight = 0;
constant int PointLight = 1;
constant int RectangularLight = 2;
constant int AreaGridSize = 2;


constant float DistributionEpsilon = 1.0e-12;
constant float BrdfEpsilon = 1.0e-6;
struct LightConstants
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};

struct DrawConstants
{
    float4 World[4];
    float4 NormalWorld[4];
    float4 ViewProjection[4];
    float4 LightViewProjection[4];
    float4 PipelineColor;
    float4 MaterialDiffuseColor;
    float4 MaterialAmbientColor;
    float4 MaterialSpecularColor;
    float4 MaterialEmissiveColor;
    float4 MaterialParameters;
    float4 AmbientColorStrength;
    float4 CameraLightCount;
    float4 RenderParameters;
    uint4 TextureFlags;
    LightConstants Lights[MaximumLights];
    float4 ShadingParameters;
};

struct VertexOutput
{
    float4 Position EVANS_POSITION;
    float3 WorldPosition, Normal;
    float4 Color, LightPosition;
    float2 TexCoord;
};
struct ForwardLightControl
{
    uint4 LightLayout; // complete count, complete shadow index, mode, tile capacity
    uint4 SurfaceTiles; // tile side, columns, viewport dimensions
};
static_assert(sizeof(ForwardLightControl) == 32, "Forward+ material selection ABI changed");
float4 TransformRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value),
        dot(rows[2], value), dot(rows[3], value));
}

float3 SRGBToLinear(float3 color)
{
    float3 low = color / SrgbLinearScale;
    float3 high = pow((color + SrgbPowerOffset) / SrgbPowerScale, SrgbExponent);
    return float3(
        color.r <= SrgbLinearThreshold ? low.r : high.r,
        color.g <= SrgbLinearThreshold ? low.g : high.g,
        color.b <= SrgbLinearThreshold ? low.b : high.b);
}

float3 PrepareDisplayLinear(float3 color)
{
    color = max(color, 0.0);
    color /= max(1.0, max(color.r, max(color.g, color.b)));
    return saturate(color);
}

float3 NormalizeSafe(float3 value, float3 fallbackValue)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalizeEpsilon
        ? value * rsqrt(lengthSquared)
        : fallbackValue;
}

struct MeshVertex { packed_float3 Position, Normal; packed_float4 Color; packed_float2 UV; };
static_assert(sizeof(MeshVertex) == 48, "Forward+ mesh vertex ABI changed");
vertex VertexOutput EvansVertexMain(device const MeshVertex* vertices [[buffer(0)]],
    constant DrawConstants& constants [[buffer(1)]], uint index [[vertex_id]])
{
    const device MeshVertex& meshVertex = vertices[index];
    float4 world = TransformRows(constants.World, float4(float3(meshVertex.Position), 1.0));
    VertexOutput result;
    result.Position = TransformRows(constants.ViewProjection, world);
    result.WorldPosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    result.Normal = NormalizeSafe(TransformRows(constants.NormalWorld, float4(float3(meshVertex.Normal), 0.0)).xyz, float3(0.0, 1.0, 0.0));
    result.Color = float4(SRGBToLinear(saturate(float3(meshVertex.Color.xyz))), meshVertex.Color.w);
    result.LightPosition = TransformRows(constants.LightViewProjection, world);
    result.TexCoord = float2(meshVertex.UV.x, 1.0 - meshVertex.UV.y);
    return result;
}
float4 SampleDiffuse(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> diffuseMap,
    sampler surfaceSampler)
{
    return constants.TextureFlags.x != 0
        ? diffuseMap.sample(surfaceSampler, input.TexCoord)
        : float4(1.0);
}

float SampleOpacity(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> opacityMap,
    sampler surfaceSampler)
{
    return constants.TextureFlags.z != 0
        ? opacityMap.sample(surfaceSampler, input.TexCoord).r
        : 1.0;
}

float SurfaceAlpha(
    VertexOutput input,
    constant DrawConstants& constants,
    float4 diffuseSample,
    texture2d<float> opacityMap,
    sampler surfaceSampler)
{
    return saturate(
        input.Color.a *
        constants.MaterialParameters.y *
        diffuseSample.a *
        SampleOpacity(input, constants, opacityMap, surfaceSampler));
}

void ApplyAlphaClip(float alpha, constant DrawConstants& constants)
{
    if (constants.MaterialParameters.z > 0.0 &&
        alpha < constants.MaterialParameters.z)
        discard_fragment();
}

// Each light matrix is four CPU row vectors; the full block matches the
// renderer's 336-byte shadow receiver contract. Depth textures use top-left coordinates.
constant uint CsmCascadeLimit = 4;
struct CascadeConstants
{
    float4 LightRows[CsmCascadeLimit][4];
    float4 FarDepths;
    float4 BlendDepths;
    float4 EyeDepthPlane;
    float4 Parameters;
    float4 Filtering; // Preserves the shared cascade ABI; PCSS owns its sampling controls.
};
float SurfaceComparisonBias(float3 normal, float3 towardLight)
{
    float squaredLength = dot(towardLight, towardLight);
    float3 direction = squaredLength > NormalizeEpsilon ? towardLight * rsqrt(squaredLength) : normal;
    float cosine = saturate(dot(normal, direction));
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, MinimumShadowCosine);
    return min(ReceiverBias + ReceiverSlopeBias * slope, MaximumReceiverBias);
}
float CsmSample(uint index, float3 position, float bias, constant CascadeConstants& cascades,
    ForwardShadowImage first, ForwardShadowImage second,
    ForwardShadowImage third, ForwardShadowImage fourth,
    constant ForwardFilterConstants& pcss)
{
    float4 clip = TransformRows(cascades.LightRows[index], float4(position, 1.0));
    if (index == 0) return PcssMap(first, 1, clip, bias, pcss);
    if (index == 1) return PcssMap(second, 2, clip, bias, pcss);
    if (index == 2) return PcssMap(third, 3, clip, bias, pcss);
    return PcssMap(fourth, 4, clip, bias, pcss);
}
#if EVANS_FORWARD_PLUS_FILTER != 3
// The cascade argument block also describes the exclusive virtual hierarchy.
// FarDepths = logical extent; BlendDepths = atlas extent and directory scale.
// Both images are point-read D32 resources, including the normalized directory.
constant uint VirtualTileBorder = 1;
constant float VirtualSampleCenter = 0.5;
float2 VirtualDepth(float2 uv, uint firstMip, constant CascadeConstants& frame,
    depth2d<float, access::read> atlas, depth2d<float, access::read> directory)
{
    uint4 layout = uint4(frame.FarDepths);
    uint4 physical = uint4(frame.BlendDepths);
    uint side = layout.x / layout.y, offset = 0;
    for (uint mip = 0; mip < layout.z; ++mip)
    {
        if (mip >= firstMip)
        {
            uint resolution = side * layout.y;
            uint2 pixel = min(uint2(uv * resolution), uint2(resolution - 1)), page = pixel / layout.y;
            uint address = offset + page.y * side + page.x;
            uint entry = uint(round(directory.read(uint2(address % layout.w, address / layout.w)) * physical.w));
            if (entry != 0)
            {
                uint slot = entry - 1;
                uint2 tile(slot % physical.z, slot / physical.z);
                uint2 at = tile * (layout.y + 2 * VirtualTileBorder) + VirtualTileBorder + pixel % layout.y;
                return float2(atlas.read(at), float(mip));
            }
        }
        offset += side * side; side /= 2;
    }
    return float2(1, layout.z - 1);
}
float VirtualCompare(int2 pixel, uint mip, float receiver, constant CascadeConstants& frame,
    depth2d<float, access::read> atlas, depth2d<float, access::read> directory)
{
    uint resolution = uint(frame.FarDepths.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0), int2(resolution - 1))) + VirtualSampleCenter) / resolution;
    return receiver <= VirtualDepth(uv, mip, frame, atlas, directory).x ? 1.0 : 0.0;
}
float VirtualVisibility(float3 position, float bias, constant CascadeConstants& frame,
    depth2d<float, access::read> atlas, depth2d<float, access::read> directory)
{
    float4 world(position, 1);
    constant float4* rows = frame.LightRows[0];
    float4 clip(dot(rows[0], world), dot(rows[1], world), dot(rows[2], world), dot(rows[3], world));
    float3 projected = clip.xyz / max(clip.w, frame.EyeDepthPlane.w);
    float2 uv(projected.x * VirtualSampleCenter + VirtualSampleCenter, VirtualSampleCenter - projected.y * VirtualSampleCenter);
    float footprint = max(length(dfdx(uv)) * frame.EyeDepthPlane.y, length(dfdy(uv)) * frame.EyeDepthPlane.z) * frame.FarDepths.x;
    uint mip = uint(clamp(floor(log2(max(1.0, footprint)) + frame.EyeDepthPlane.x), 0.0, frame.FarDepths.z - 1.0));
    if (clip.w <= frame.EyeDepthPlane.w || !all(uv >= 0.0) || !all(uv <= 1.0) || projected.z < 0 || projected.z > 1) return 1;
    float2 stored = VirtualDepth(uv, mip, frame, atlas, directory); float receiver = projected.z - bias;
    if (frame.Filtering.x == 0) return receiver <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(frame.FarDepths.x) >> mip) - VirtualSampleCenter;
    int2 base = int2(int(floor(texel.x)), int(floor(texel.y))); float2 fraction = fract(texel); float visible = 0;
    for (int y = -int(frame.Filtering.x); y <= int(frame.Filtering.x); ++y)
        for (int x = -int(frame.Filtering.x); x <= int(frame.Filtering.x); ++x)
        {
            int2 p = base + int2(x, y);
            float a = VirtualCompare(p, mip, receiver, frame, atlas, directory);
            float b = VirtualCompare(p + int2(1, 0), mip, receiver, frame, atlas, directory);
            float c = VirtualCompare(p + int2(0, 1), mip, receiver, frame, atlas, directory);
            float d = VirtualCompare(p + 1, mip, receiver, frame, atlas, directory);
            visible += mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
        }
    return visible / frame.Filtering.y;
}

#endif

float CsmVisibility(float3 position, float3 normal, float3 directionToLight, constant CascadeConstants& cascades,
    ForwardShadowImage first, ForwardShadowImage second,
    ForwardShadowImage third, ForwardShadowImage fourth,
    constant ForwardFilterConstants& pcss)
{
#if EVANS_FORWARD_PLUS_FILTER != 3
    if (cascades.Filtering.z != 0) return 1.0 - ShadowStrength * (1.0 - VirtualVisibility(position, SurfaceComparisonBias(normal, directionToLight), cascades, first, second));
#endif
    float eyeDepth = dot(cascades.EyeDepthPlane, float4(position, 1.0));
    uint count = uint(cascades.Parameters.z);
    if (count == 0 || eyeDepth < cascades.Parameters.x || eyeDepth > cascades.Parameters.y) return 1.0;
    uint interval = 0;
    while (interval + 1 < count && eyeDepth > cascades.FarDepths[interval]) ++interval;
    float bias = SurfaceComparisonBias(normal, directionToLight);
    float visibility = CsmSample(interval, position, bias, cascades, first, second, third, fourth, pcss);
    float overlap = cascades.FarDepths[interval] - cascades.BlendDepths[interval];
    if (overlap > DistanceEpsilon && eyeDepth > cascades.BlendDepths[interval])
    {
        float neighbor = interval + 1 < count ? CsmSample(interval + 1, position, bias, cascades, first, second, third, fourth, pcss) : 1.0;
        visibility = mix(visibility, neighbor, saturate((eyeDepth - cascades.BlendDepths[interval]) / overlap));
    }
    return 1.0 - ShadowStrength * (1.0 - visibility);
}
float SampleShadow(VertexOutput input, constant DrawConstants& constants,
    ForwardShadowImage shadowMap, float3 normal,
    float3 directionToLight, constant ForwardFilterConstants& pcss, texture2d<uint,access::read> shadowRayImage)
{
    if (constants.RenderParameters.x>=0.5 && ShadowRayEnabled(shadowRayImage)) return 1.0-ShadowStrength*(1.0-ShadowRayVisibility(input.WorldPosition,normal,directionToLight,shadowRayImage));
    if (constants.RenderParameters.x < 0.5) return 1.0;
    float visible = PcssMap(shadowMap, 0, input.LightPosition, SurfaceComparisonBias(normal, directionToLight), pcss);
    return 1.0 - ShadowStrength * (1.0 - visible);
}
float3 ResolveSurfaceNormal(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> normalMap,
    sampler surfaceSampler,
    bool frontFacing)
{
    float3 normal = NormalizeSafe(input.Normal, float3(0.0, 1.0, 0.0));
    const uint FlatModel = 1;
    if (constants.TextureFlags.w == FlatModel)
    {
        float3 plane = NormalizeSafe(cross(dfdx(input.WorldPosition), dfdy(input.WorldPosition)), normal);
        if (dot(plane, normal) < 0.0) plane = -plane;
        return frontFacing ? plane : -plane;
    }
    if (!frontFacing) normal = -normal;
    if (constants.TextureFlags.y == 0) return normal;
    float3 positionDx = dfdx(input.WorldPosition);
    float3 positionDy = dfdy(input.WorldPosition);
    float2 texCoordDx = dfdx(input.TexCoord);
    float2 texCoordDy = dfdy(input.TexCoord);
    float determinant =
        texCoordDx.x * texCoordDy.y -
        texCoordDx.y * texCoordDy.x;
    if (abs(determinant) < UvDeterminantEpsilon) return normal;
    float inverseDeterminant = 1.0 / determinant;
    float3 tangent =
        (positionDx * texCoordDy.y - positionDy * texCoordDx.y) *
        inverseDeterminant;
    float3 sourceBitangent =
        (-positionDx * texCoordDy.x + positionDy * texCoordDx.x) *
        inverseDeterminant;
    tangent -= normal * dot(normal, tangent);
    float3 helper = abs(normal.y) < ParallelAxisThreshold
        ? float3(0.0, 1.0, 0.0)
        : float3(1.0, 0.0, 0.0);
    tangent = NormalizeSafe(
        tangent,
        NormalizeSafe(cross(helper, normal), float3(1.0, 0.0, 0.0)));
    float handedness =
        dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    float3 bitangent = NormalizeSafe(
        cross(normal, tangent) * handedness,
        float3(0.0, 0.0, 1.0));
    float3 tangentNormal =
        normalMap.sample(surfaceSampler, input.TexCoord).xyz * 2.0 - 1.0;
    if (constants.TextureFlags.w == ToonShadingModel) tangentNormal = mix(float3(0.0, 0.0, 1.0), tangentNormal, ToonNormalStrength);
    return NormalizeSafe(
        tangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

void AccumulatePhysicalSample(float2 roughnessMetallic,
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= NormalizeEpsilon) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = saturate(dot(normal, direction));
    if (normalToLight <= 0.0) return;
    float3 visibleRadiance = radiance * visibility;
    float3 halfway = NormalizeSafe(direction + viewDirection, normal);
    float nDotV = saturate(dot(normal, viewDirection));
    if (nDotV <= 0.0) return;
    float nDotH = saturate(dot(normal, halfway));
    float vDotH = saturate(dot(viewDirection, halfway));
    float roughness = clamp(roughnessMetallic.x, MinimumRoughness, 1.0);
    float metallic = saturate(roughnessMetallic.y);
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float ggxDenominator = nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
    float distribution = alphaSquared / max(Pi * ggxDenominator * ggxDenominator, DistributionEpsilon);
    float masking = 2.0 * nDotV / (nDotV + sqrt(alphaSquared + (1.0 - alphaSquared) * nDotV * nDotV));
    float shadowing = 2.0 * normalToLight / (normalToLight + sqrt(alphaSquared + (1.0 - alphaSquared) * normalToLight * normalToLight));
    float3 baseReflectance = mix(float3(DielectricReflectance), saturate(albedo), metallic);
    const float SchlickExponent = 5.0;
    float3 fresnel = baseReflectance + (1.0 - baseReflectance) * pow(1.0 - vDotH, SchlickExponent);
    const float MicrofacetNormalization = 4.0;
    float3 specular = fresnel * distribution * masking * shadowing /
        max(MicrofacetNormalization * nDotV * normalToLight, BrdfEpsilon);
    diffuseLight += visibleRadiance * (1.0 - fresnel) * (1.0 - metallic) * normalToLight / Pi;
    specularLight += visibleRadiance * specular * normalToLight;
}

void AccumulateToonSample(
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= NormalizeEpsilon) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = max(dot(normal, direction), 0.0);
    float3 visibleRadiance = radiance * visibility;
    float bands = max(constants.ShadingParameters.z, 2.0);
    float3 toonResponse = ToonResponse(normalToLight, bands);
    diffuseLight += visibleRadiance * toonResponse;
    float3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0),
        max(constants.MaterialParameters.x, 1.0));

    specularLight += visibleRadiance * ToonHighlight(specularAmount);
}
void AccumulatePhongSample(
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= NormalizeEpsilon) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = max(dot(normal, direction), 0.0);
    if (normalToLight <= 0.0) return;
    float3 visibleRadiance = radiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    float3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0),
        max(constants.MaterialParameters.x, 1.0));
    specularLight += visibleRadiance * specularAmount;
}
void AccumulateLightSample(float2 roughnessMetallic, float3 lightDirection, float3 radiance, float visibility,
    float3 normal, float3 viewDirection, float3 albedo, constant DrawConstants& constants,
    thread float3& diffuseLight, thread float3& specularLight)
{
    const uint PhysicalModel = 3, ToonModel = 4;
    if (constants.TextureFlags.w == PhysicalModel)
        AccumulatePhysicalSample(roughnessMetallic, lightDirection, radiance, visibility, normal, viewDirection, albedo, constants, diffuseLight, specularLight);
    else if (constants.TextureFlags.w == ToonModel)
        AccumulateToonSample(lightDirection, radiance, visibility, normal, viewDirection, albedo, constants, diffuseLight, specularLight);
    else AccumulatePhongSample(lightDirection, radiance, visibility, normal, viewDirection, albedo, constants, diffuseLight, specularLight);
}
void AccumulateSceneLight(float2 roughnessMetallic,
    uint lightIndex, device const LightConstants* sceneLights, constant ForwardLightControl& selection,
    VertexOutput input,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    ForwardShadowImage shadowMap,
    bool useCascades, float cascadeVisibility, float contactVisibility,
    constant ForwardFilterConstants& pcss,
    thread float3& diffuseLight,
    thread float3& specularLight, texture2d<uint,access::read> shadowRayImage)
{
    LightConstants light = sceneLights[lightIndex];
    uint type = uint(light.PositionType.w + 0.5);
    float3 shadowDirection = type == DirectionalLight
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - input.WorldPosition;
    bool isShadowLight =
        int(lightIndex) == int(selection.LightLayout.y) &&
        light.ColorCastShadows.w > 0.5;
    float visibility = isShadowLight
        ? (useCascades ? cascadeVisibility : SampleShadow(
            input, constants, shadowMap, normal, shadowDirection, pcss, shadowRayImage))
        : 1.0;
    if (isShadowLight && contactVisibility < 1.0)
        visibility = mix(1.0 - ShadowStrength, visibility, contactVisibility);
    float3 radiance =
        max(light.ColorCastShadows.rgb, 0.0) *
        max(light.DirectionIntensity.w, 0.0);
    if (type == DirectionalLight)
    {
        AccumulateLightSample(roughnessMetallic,
            light.DirectionIntensity.xyz, radiance, visibility,
            normal, viewDirection, albedo, constants, diffuseLight, specularLight);
        return;
    }
    if (type == PointLight)
    {
        float3 toLight = light.PositionType.xyz - input.WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), DistanceEpsilon);
        AccumulateLightSample(roughnessMetallic,
            toLight, radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared, visibility,
            normal, viewDirection, albedo, constants, diffuseLight, specularLight);
        return;
    }
    float3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz, float3(0.0, -1.0, 0.0));
    float3 helper = abs(emissionDirection.y) < ParallelAxisThreshold
        ? float3(0.0, 1.0, 0.0)
        : float3(1.0, 0.0, 0.0);
    float3 tangent = NormalizeSafe(
        cross(helper, emissionDirection), float3(1.0, 0.0, 0.0));
    float3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent), float3(0.0, 0.0, 1.0));
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
            float3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            float3 toLight = samplePosition - input.WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), DistanceEpsilon);
            float3 direction = toLight * rsqrt(distanceSquared);
            float emitterCosine =
                max(dot(emissionDirection, -direction), 0.0);
            AccumulateLightSample(roughnessMetallic,
                direction,
                radiance * (emitterSampleWeight * emitterCosine * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared),
                visibility, normal, viewDirection, albedo, constants,
                diffuseLight, specularLight);
        }
    }
}

// Transparent fragments use their own depth, then ordinary source-over blends
// the fogged radiance. The standalone VolumetricFog source produces the atlas.
struct SurfaceFogParameters
{
    float4 InverseCameraRows[4];
    float4 Medium;
    float4 SourceRadiance;
    float4 ViewAndCells;
    uint4 Control;
};
constant float FogLookupCenter = 0.5;
constant float FogLookupRayEpsilon = 1.0e-6;
float3 SurfaceFogPoint(float2 uv, float depth, constant SurfaceFogParameters& fog)
{
    float4 ndc = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), depth, 1.0);
    float4 world = float4(dot(fog.InverseCameraRows[0], ndc), dot(fog.InverseCameraRows[1], ndc),
        dot(fog.InverseCameraRows[2], ndc), dot(fog.InverseCameraRows[3], ndc));
    return world.xyz / world.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
constant uint RgbTransportMode = 2;
float4 SurfaceFogPrefixTransmission(float2 uv, uint boundary, constant SurfaceFogParameters& fog, texture2d<float> atlas)
{
    if (boundary == 0) return float4(1, 1, 1, 1);
    float2 margin = FogLookupCenter / fog.ViewAndCells.zw;
    float2 packed = clamp(uv, margin, 1.0 - margin);
    packed.x = (float(boundary - 1) + packed.x) / float(fog.Control.y);
    constexpr sampler AtlasSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    packed.y = (packed.y + 1.0) * 0.5;
    return atlas.sample(AtlasSampler, packed, level(0.0));
}
float4 SurfaceFogPrefix(float2 uv, uint boundary, constant SurfaceFogParameters& fog, texture2d<float> atlas)
{
    if (boundary == 0) return float4(0.0, 0.0, 0.0, 1.0);
    float2 margin = FogLookupCenter / fog.ViewAndCells.zw;
    float2 packed = clamp(uv, margin, 1.0 - margin);
    packed.x = (float(boundary - 1) + packed.x) / float(fog.Control.y);
    constexpr sampler AtlasSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    if (fog.Control.w == RgbTransportMode) packed.y *= 0.5;
    return atlas.sample(AtlasSampler, packed, level(0.0));
}
float3 ApplySurfaceFog(float3 radiance, float3 fragmentPosition,
    constant SurfaceFogParameters& fog, texture2d<float> atlas)
{
    if (fog.Control.w == 0) return radiance;
    float2 uv = fragmentPosition.xy / fog.ViewAndCells.xy;
    float3 start = SurfaceFogPoint(uv, 0.0, fog);
    float extent = min(length(SurfaceFogPoint(uv, 1.0, fog) - start), fog.Medium.w);
    float distance = length(SurfaceFogPoint(uv, fragmentPosition.z, fog) - start);
    float boundary = clamp(distance / max(extent, FogLookupRayEpsilon), 0.0, 1.0) * float(fog.Control.y);
    uint low = uint(boundary), high = min(low + 1, fog.Control.y);
    float4 medium = mix(SurfaceFogPrefix(uv, low, fog, atlas), SurfaceFogPrefix(uv, high, fog, atlas), boundary - float(low));
    float3 mediumTransmission = fog.Control.w == RgbTransportMode ? (mix(SurfaceFogPrefixTransmission(uv, low, fog, atlas), SurfaceFogPrefixTransmission(uv, high, fog, atlas), boundary - float(low))).rgb : medium.aaa;
    return radiance * mediumTransmission + medium.rgb;
}
uint2 SelectSurfaceLights(float2 position, constant ForwardLightControl& selection,
    device const uint4* headers, thread bool& indexed)
{
    indexed = false; uint2 span(0, selection.LightLayout.x);
    if (selection.LightLayout.z != 0)
    {
        uint2 pixel = min(uint2(position), selection.SurfaceTiles.zw - 1);
        uint2 tile = pixel / selection.SurfaceTiles.x;
        uint4 record = headers[tile.y * selection.SurfaceTiles.y + tile.x];
        uint2 candidate = selection.LightLayout.z == 1 ? record.xy : record.zw;
        if (candidate.y <= selection.LightLayout.w) { span = candidate; indexed = true; }
    }
    return span;
}
float4 EvaluateMaterialSurface(
    VertexOutput input, device const LightConstants* sceneLights, constant ForwardLightControl& selection,
    device const uint4* tileHeaders, device const uint* tileIndices,
    constant DrawConstants& constants,
    constant ForwardFilterConstants& pcss,
    constant CascadeConstants& cascades,
    constant SurfaceFogParameters& fog,
    texture2d<float> fogAtlas,
    constant LumenVolumeParameters& lumenVolume, texture2d<float> lumenAtlas,
    ForwardShadowImage cascade0,
    ForwardShadowImage cascade1,
    ForwardShadowImage cascade2,
    ForwardShadowImage cascade3,
    ForwardShadowImage shadowMap,
    texture2d<float> diffuseMap,
    texture2d<float> normalMap,
    texture2d<float> opacityMap,
    sampler surfaceSampler,
    bool frontFacing, texture2d<uint,access::read> shadowRayImage, float contactVisibility = 1.0)
{
    float2 roughnessMetallic = constants.ShadingParameters.xy;
    if (constants.TextureFlags.z > 1) roughnessMetallic *= opacityMap.sample(surfaceSampler, input.TexCoord).gb;
    float4 diffuseSample =
        SampleDiffuse(input, constants, diffuseMap, surfaceSampler);
    if (constants.TextureFlags.w == ToonShadingModel && constants.TextureFlags.x != 0u)
        diffuseSample.rgb = ToonTextureColor(diffuseMap, surfaceSampler, input.TexCoord, dfdx(input.TexCoord), dfdy(input.TexCoord));
    float alpha = SurfaceAlpha(
        input, constants, diffuseSample, opacityMap, surfaceSampler);
    ApplyAlphaClip(alpha, constants);
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (constants.RenderParameters.y > 1.0 && constants.RenderParameters.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = dfdx(wireBary), wireDy = dfdy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard_fragment();
    }
    if (constants.RenderParameters.y > 0.5)
    {
        return float4(
            ApplySurfaceFog(SRGBToLinear(saturate(constants.PipelineColor.rgb)), input.Position.xyz, fog, fogAtlas),
            alpha);
    }
    float3 surfaceBase = ToonAlbedo(input.Color.rgb *
        max(constants.PipelineColor.rgb, 0.0) *
        max(diffuseSample.rgb, 0.0), constants.TextureFlags.w);
    float3 normal = ResolveSurfaceNormal(
        input, constants, normalMap, surfaceSampler, frontFacing);
    float3 viewDirection = NormalizeSafe(
        constants.CameraLightCount.xyz - input.WorldPosition,
        normal);
    float3 diffuseLight = 0.0, specularLight = 0.0;
    bool useCascades = cascades.Parameters.z > 0.0 && constants.RenderParameters.x >= 0.5;
    float cascadeVisibility = 1.0;
    uint shadowIndex = selection.LightLayout.y;
    if (useCascades && shadowIndex < selection.LightLayout.x)
        cascadeVisibility = CsmVisibility(input.WorldPosition, normal,
            sceneLights[shadowIndex].DirectionIntensity.xyz, cascades, cascade0, cascade1, cascade2, cascade3, pcss);
    bool indexed; uint2 span = SelectSurfaceLights(input.Position.xy, selection, tileHeaders, indexed);
    for (uint entry = 0; entry < span.y; ++entry)
    {
        uint lightIndex = indexed ? tileIndices[span.x + entry] : entry;
        AccumulateSceneLight(roughnessMetallic, lightIndex, sceneLights, selection, input, normal, viewDirection,
            surfaceBase * max(constants.MaterialDiffuseColor.rgb, 0.0), constants, shadowMap,
            useCascades, cascadeVisibility, contactVisibility, pcss, diffuseLight, specularLight, shadowRayImage);
    }
    const uint PhysicalModel = 3;
    float3 ambient = ToonAmbient(surfaceBase * max(constants.MaterialDiffuseColor.rgb, float3(0.0)), surfaceBase * max(constants.MaterialAmbientColor.rgb, 0.0) *
        max(constants.AmbientColorStrength.rgb, 0.0) * max(constants.AmbientColorStrength.w, 0.0), constants.TextureFlags.w);
    if (constants.TextureFlags.w == PhysicalModel) ambient *= 1.0 - saturate(roughnessMetallic.y);
    else specularLight *= max(constants.MaterialSpecularColor.rgb, 0.0);
    float3 color = ambient + surfaceBase * max(constants.MaterialDiffuseColor.rgb, 0.0) * diffuseLight +
        specularLight + max(constants.MaterialEmissiveColor.rgb, 0.0);
    color += surfaceBase * max(constants.MaterialDiffuseColor.rgb,0.0) * (constants.TextureFlags.w==PhysicalModel ? (1-LumenDielectricReflectance)*(1.0-saturate(roughnessMetallic.y)) : 1.0) * LumenTranslucentIrradiance(input.WorldPosition,normal,lumenVolume,lumenAtlas);
    color = ApplySurfaceFog(color, input.Position.xyz, fog, fogAtlas);
    // Floating-point targets preserve radiance until display tone mapping.
    const float MaximumHalfFloatRadiance = 65504.0;
    return float4(constants.ShadingParameters.w > 0.5 ? clamp(color, 0.0, MaximumHalfFloatRadiance) : PrepareDisplayLinear(color), alpha);
}

fragment float4 EvansFragmentMain(
    VertexOutput input [[stage_in]],
#if EVANS_CONTACT_SHADOWS
    texture2d_array<float, access::read> contactMask [[texture(15)]],
#if EVANS_CONTACT_MULTISAMPLE
    uint contactSample [[sample_id]],
#endif
#endif
    constant DrawConstants& constants [[buffer(1)]],
    constant ForwardLightControl& selection [[buffer(7)]],
    device const LightConstants* sceneLights [[buffer(8)]],
    device const uint4* tileHeaders [[buffer(9)]],
    device const uint* tileIndices [[buffer(10)]],
    constant ForwardFilterConstants& pcss [[buffer(6)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    constant SurfaceFogParameters& fog [[buffer(4)]],
    texture2d<float> fogAtlas [[texture(14)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(11)]],
    texture2d<float> lumenAtlas [[texture(19)]],
    ForwardShadowImage cascade0 [[texture(10)]],
    ForwardShadowImage cascade1 [[texture(11)]],
    ForwardShadowImage cascade2 [[texture(12)]],
    ForwardShadowImage cascade3 [[texture(13)]],
    ForwardShadowImage shadowMap [[texture(0)]],
    texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]],
    bool frontFacing [[front_facing]], texture2d<uint,access::read> shadowRayImage [[texture(20)]])
{
    float contactVisibility = 1.0;
#if EVANS_CONTACT_SHADOWS
#if EVANS_CONTACT_MULTISAMPLE
    contactVisibility = contactMask.read(uint2(input.Position.xy), contactSample).x;
#else
    contactVisibility = contactMask.read(uint2(input.Position.xy), 0).x;
#endif
#endif
    return EvaluateMaterialSurface(input, sceneLights, selection, tileHeaders, tileIndices, constants, pcss, cascades, fog, fogAtlas, lumenVolume, lumenAtlas,
        cascade0, cascade1, cascade2, cascade3, shadowMap,
        diffuseMap, normalMap, opacityMap, surfaceSampler, frontFacing, shadowRayImage, contactVisibility);
}

struct WeightedCoverageParameters
{
    float4 Weights; // Opacity/depth exponents, depth scale, positive weight floor.
    uint4 Options;
};
struct WeightedSurfaceOutput
{
    float4 RadianceSum [[color(0)]];
    float Opacity [[color(1)]];
};
fragment WeightedSurfaceOutput EvansWeightedFragment(
    VertexOutput input [[stage_in]],
    constant DrawConstants& constants [[buffer(1)]],
    constant ForwardLightControl& selection [[buffer(7)]],
    device const LightConstants* sceneLights [[buffer(8)]],
    device const uint4* tileHeaders [[buffer(9)]],
    device const uint* tileIndices [[buffer(10)]],
    constant ForwardFilterConstants& pcss [[buffer(6)]],
    constant CascadeConstants& cascades [[buffer(2)]],
    constant SurfaceFogParameters& fog [[buffer(4)]],
    texture2d<float> fogAtlas [[texture(14)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(11)]],
    texture2d<float> lumenAtlas [[texture(19)]],
    ForwardShadowImage cascade0 [[texture(10)]],
    ForwardShadowImage cascade1 [[texture(11)]],
    ForwardShadowImage cascade2 [[texture(12)]],
    ForwardShadowImage cascade3 [[texture(13)]],
    ForwardShadowImage shadowMap [[texture(0)]],
    texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]],
    constant WeightedCoverageParameters& transparency [[buffer(5)]],
    bool frontFacing [[front_facing]], texture2d<uint,access::read> shadowRayImage [[texture(20)]])
{
    // Preserve this model's lighting and fragment-depth fog before accumulation.
    float4 surface = EvaluateMaterialSurface(input, sceneLights, selection, tileHeaders, tileIndices, constants, pcss, cascades, fog, fogAtlas, lumenVolume, lumenAtlas,
        cascade0, cascade1, cascade2, cascade3, shadowMap,
        diffuseMap, normalMap, opacityMap, surfaceSampler, frontFacing, shadowRayImage);
    float coverage = clamp(surface.a, 0.0, 1.0);
    WeightedSurfaceOutput output {float4(0.0), coverage};
    if (coverage == 0.0) return output;
    float depthFactor = 1.0 - clamp(input.Position.z, 0.0, 1.0) * transparency.Weights.z;
    float contribution = coverage * max(pow(coverage, transparency.Weights.x) *
        pow(depthFactor, transparency.Weights.y), transparency.Weights.w);
    output.RadianceSum = float4(surface.rgb * contribution, contribution);
    return output;
}

fragment void ForwardDepthFragment(VertexOutput input [[stage_in]],
    constant DrawConstants& constants [[buffer(1)]], texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> opacityMap [[texture(3)]], sampler surfaceSampler [[sampler(0)]])
{
    float4 diffuse = SampleDiffuse(input, constants, diffuseMap, surfaceSampler);
    if (constants.TextureFlags.w == ToonShadingModel && constants.TextureFlags.x != 0u)
        diffuse.rgb = ToonTextureColor(diffuseMap, surfaceSampler, input.TexCoord, dfdx(input.TexCoord), dfdy(input.TexCoord));
    ApplyAlphaClip(SurfaceAlpha(input, constants, diffuse, opacityMap, surfaceSampler), constants);
}
#else
// Independent Metal Forward+ assignment. Harada, McKee and Yang,
// "Forward+: Bringing Deferred Lighting to the Next Level" (2012):
// https://takahiroharada.github.io/forward%2B/
// A Brent-Kung inclusive scan compacts membership in ascending light order.
// Brent and Kung, "A Regular Layout for Parallel Adders" (1982), describes
// the reduction/distribution prefix tree. No SIMD-group width is assumed.
// https://maths-people.anu.edu.au/~brent/pub/pub060.html
// Compile without fast math; conservative plane tolerances include cancellation
// and flushed subnormals. Depth samples and list tiles both have top-left origin.
constexpr constant uint ThreadsPerTile = 64;
constexpr constant uint BitsPerMask = 32;
constexpr constant uint MaximumLights = 4096;
constexpr constant uint MaskCount = MaximumLights / BitsPerMask;
constexpr constant uint LateralPlanes = 4;
constexpr constant uint OpaqueNear = LateralPlanes, OpaqueFar = OpaqueNear + 1;
constexpr constant uint TransparentNear = OpaqueFar + 1, TransparentFar = TransparentNear + 1;
constexpr constant uint PlaneCount = TransparentFar + 1;
constexpr constant uint NegativeZero = 0x80000000u;
constant float Binary32Epsilon = 1.1920928955078125e-7;
constant float RelativeErrorOperations = 16.0;
constant float RelativeTolerance = RelativeErrorOperations * Binary32Epsilon;
constant float MinimumNormal = 1.1754943508222875e-38;
constant float UnderflowOperations = 64.0;
constant float AbsoluteTolerance = UnderflowOperations * MinimumNormal;

struct CameraGrid
{
    float4 CameraRows[4];
    uint4 ExtentAndLights; // width, height, original sample count, light count
    uint4 TileLayout; // tile side, list capacity, columns, rows
};
struct SupportBound { float4 CenterRadius; uint4 Flags; };
static_assert(sizeof(CameraGrid) == 96 && sizeof(SupportBound) == 32, "Forward+ Metal buffer ABI changed");
struct TileScratch
{
    atomic_uint MinimumDepth, MaximumDepth;
    float4 Planes[PlaneCount], Errors[PlaneCount];
    atomic_uint OpaqueMasks[MaskCount], TransparentMasks[MaskCount];
    uint2 Prefix[MaskCount];
};
float MaximumComponent(float4 value)
{ return max(max(value.x, value.y), max(value.z, value.w)); }
void MakePlane(uint slot, float4 row, float edge, float orientation,
    constant CameraGrid& camera, threadgroup TileScratch& tile)
{
    float scale = max(MinimumNormal, max(MaximumComponent(abs(row)), MaximumComponent(abs(camera.CameraRows[3]))));
    float4 axis = row / scale, clipW = camera.CameraRows[3] / scale, edgeW = edge * clipW;
    tile.Planes[slot] = orientation * (axis - edgeW);
    // Bound both terms BEFORE subtraction: the resulting plane may cancel.
    tile.Errors[slot] = RelativeTolerance * (abs(axis) + abs(edgeW) + abs(clipW));
}
bool Intersects(uint plane, float4 center, float radius, threadgroup TileScratch& tile)
{
    float4 products = tile.Planes[plane] * center;
    float signedDistance = (products.x + products.y) + (products.z + products.w);
    float support = radius * length(tile.Planes[plane].xyz);
    float tolerance = dot(tile.Errors[plane], abs(center)) + radius * length(tile.Errors[plane].xyz);
    tolerance += RelativeTolerance * (dot(abs(products), float4(1.0)) + support) + AbsoluteTolerance;
    return signedDistance + support >= -tolerance;
}
bool2 Membership(device const SupportBound& light, threadgroup TileScratch& tile)
{
    if (light.Flags.x != 0) return bool2(true);
    float scale = max(1.0, MaximumComponent(abs(light.CenterRadius)));
    float4 center = float4(light.CenterRadius.xyz, 1.0) / scale;
    float radius = light.CenterRadius.w / scale;
    for (uint side = 0; side < LateralPlanes; ++side)
        if (!Intersects(side, center, radius, tile)) return bool2(false);
    return bool2(Intersects(OpaqueNear, center, radius, tile) && Intersects(OpaqueFar, center, radius, tile),
        Intersects(TransparentNear, center, radius, tile) && Intersects(TransparentFar, center, radius, tile));
}
float FetchDepth(depth2d<float, access::read> depth, uint2 pixel, uint sampleIndex)
{ (void)sampleIndex; return depth.read(pixel); }
float FetchDepth(depth2d_ms<float, access::read> depth, uint2 pixel, uint sampleIndex)
{ return depth.read(pixel, sampleIndex); }
void StoreIndices(uint mask, uint word, uint firstOrdinal, uint destination, uint capacity, device uint* indices)
{
    uint ordinal = firstOrdinal;
    while (mask != 0)
    {
        if (ordinal < capacity) indices[destination + ordinal] = word * BitsPerMask + ctz(mask);
        mask &= mask - 1; ++ordinal;
    }
}
template<typename DepthTexture>
void AssignTile(DepthTexture depth, constant CameraGrid& camera, device const SupportBound* lights,
    device uint4* headers, device uint* indices, uint lane, uint2 coordinate, threadgroup TileScratch& tile)
{
    if (lane == 0)
    {
        atomic_store_explicit(&tile.MinimumDepth, as_type<uint>(1.0f), memory_order_relaxed);
        atomic_store_explicit(&tile.MaximumDepth, 0u, memory_order_relaxed);
    }
    for (uint word = lane; word < MaskCount; word += ThreadsPerTile)
    {
        atomic_store_explicit(&tile.OpaqueMasks[word], 0u, memory_order_relaxed);
        atomic_store_explicit(&tile.TransparentMasks[word], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pixelIndex = lane; pixelIndex < camera.TileLayout.x * camera.TileLayout.x; pixelIndex += ThreadsPerTile)
    {
        uint2 pixel = coordinate * camera.TileLayout.x + uint2(pixelIndex % camera.TileLayout.x, pixelIndex / camera.TileLayout.x);
        if (any(pixel >= camera.ExtentAndLights.xy)) continue;
        for (uint sampleIndex = 0; sampleIndex < camera.ExtentAndLights.z; ++sampleIndex)
        {
            uint bits = as_type<uint>(FetchDepth(depth, pixel, sampleIndex));
            if (bits == NegativeZero) bits = 0;
            // [0,1] float encodings preserve integer order. Clear remains a
            // possible far-plane receiver; invalid values widen the interval.
            bool valid = bits <= as_type<uint>(1.0f);
            atomic_fetch_min_explicit(&tile.MinimumDepth, valid ? bits : 0u, memory_order_relaxed);
            atomic_fetch_max_explicit(&tile.MaximumDepth, valid ? bits : as_type<uint>(1.0f), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 low = float2(coordinate * camera.TileLayout.x) * 2.0 / float2(camera.ExtentAndLights.xy) - 1.0;
    float2 high = float2(min((coordinate + 1u) * camera.TileLayout.x, camera.ExtentAndLights.xy)) * 2.0 / float2(camera.ExtentAndLights.xy) - 1.0;
    if (lane == 0) MakePlane(lane, camera.CameraRows[0], low.x, 1.0, camera, tile);
    if (lane == 1) MakePlane(lane, camera.CameraRows[0], high.x, -1.0, camera, tile);
    if (lane == 2) MakePlane(lane, camera.CameraRows[1], -low.y, -1.0, camera, tile);
    if (lane == 3) MakePlane(lane, camera.CameraRows[1], -high.y, 1.0, camera, tile);
    if (lane == OpaqueNear) MakePlane(lane, camera.CameraRows[2], as_type<float>(atomic_load_explicit(&tile.MinimumDepth, memory_order_relaxed)), 1.0, camera, tile);
    if (lane == OpaqueFar) MakePlane(lane, camera.CameraRows[2], as_type<float>(atomic_load_explicit(&tile.MaximumDepth, memory_order_relaxed)), -1.0, camera, tile);
    if (lane == TransparentNear) MakePlane(lane, camera.CameraRows[2], 0.0, 1.0, camera, tile);
    if (lane == TransparentFar) MakePlane(lane, camera.CameraRows[2], 1.0, -1.0, camera, tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint light = lane; light < camera.ExtentAndLights.w; light += ThreadsPerTile)
    {
        bool2 accepted = Membership(lights[light], tile);
        uint word = light / BitsPerMask, bit = 1u << (light % BitsPerMask);
        if (accepted.x) atomic_fetch_or_explicit(&tile.OpaqueMasks[word], bit, memory_order_relaxed);
        if (accepted.y) atomic_fetch_or_explicit(&tile.TransparentMasks[word], bit, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint word = lane; word < MaskCount; word += ThreadsPerTile)
        tile.Prefix[word] = popcount(uint2(atomic_load_explicit(&tile.OpaqueMasks[word], memory_order_relaxed),
            atomic_load_explicit(&tile.TransparentMasks[word], memory_order_relaxed)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Reduction computes prefixes at the right edge of each binary subtree.
    for (uint stride = 1; stride < MaskCount; stride *= 2)
    {
        uint right = (lane + 1) * stride * 2 - 1;
        if (right < MaskCount) tile.Prefix[right] += tile.Prefix[right - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Distribution fills the intervening prefixes, keeping the inclusive root.
    for (uint stride = MaskCount / 4; stride > 0; stride /= 2)
    {
        uint right = (lane + 1) * stride * 2 - 1;
        if (right + stride < MaskCount) tile.Prefix[right + stride] += tile.Prefix[right];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint tileIndex = coordinate.y * camera.TileLayout.z + coordinate.x;
    uint opaqueStart = tileIndex * 2 * camera.TileLayout.y, transparentStart = opaqueStart + camera.TileLayout.y;
    for (uint word = lane; word < MaskCount; word += ThreadsPerTile)
    {
        uint2 masks(atomic_load_explicit(&tile.OpaqueMasks[word], memory_order_relaxed),
            atomic_load_explicit(&tile.TransparentMasks[word], memory_order_relaxed));
        uint2 start = tile.Prefix[word] - popcount(masks);
        StoreIndices(masks.x, word, start.x, opaqueStart, camera.TileLayout.y, indices);
        StoreIndices(masks.y, word, start.y, transparentStart, camera.TileLayout.y, indices);
    }
    // Store TOTAL candidates, not just the stored prefix. Overflow requires a
    // full scene-light loop in the consumer; truncation would lose illumination.
    if (lane == 0) headers[tileIndex] = uint4(opaqueStart, tile.Prefix[MaskCount - 1].x,
        transparentStart, tile.Prefix[MaskCount - 1].y);
}
kernel void ForwardPlusSingle(depth2d<float, access::read> depth [[texture(0)]],
    constant CameraGrid& camera [[buffer(0)]], device const SupportBound* lights [[buffer(1)]],
    device uint4* headers [[buffer(2)]], device uint* indices [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]], uint3 coordinate [[threadgroup_position_in_grid]])
{
    threadgroup TileScratch scratch;
    AssignTile(depth, camera, lights, headers, indices, lane, coordinate.xy, scratch);
}
kernel void ForwardPlusMultisample(depth2d_ms<float, access::read> depth [[texture(0)]],
    constant CameraGrid& camera [[buffer(0)]], device const SupportBound* lights [[buffer(1)]],
    device uint4* headers [[buffer(2)]], device uint* indices [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]], uint3 coordinate [[threadgroup_position_in_grid]])
{
    threadgroup TileScratch scratch;
    AssignTile(depth, camera, lights, headers, indices, lane, coordinate.xy, scratch);
}

#endif
