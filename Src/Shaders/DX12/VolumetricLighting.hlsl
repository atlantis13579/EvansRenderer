// DX12 world-space shadow queries. Woop, Benthin and Wald, JCGT 2013.
// https://jcgt.org/published/0002/01/05/ . Forward escape links bound traversal;
// independent alpha planes preserve bilinear cutout coverage. The RGBA32UI ABI
// is described in RayTracedShadowScene.h. No screen-space visibility is assumed.
Texture2D<uint4> ShadowRayScene : register(t20);
static const uint ShadowRayCasterFlag=1u,ShadowRayImageABI=1u;
static const uint ShadowRayCountsHeader=0u,ShadowRayOffsetsHeader=1u,ShadowRayTraceHeader=2u,ShadowRayLightHeader=3u;
static const uint ShadowRayNodeSize=3u,ShadowRayTriangleSize=6u,ShadowRayMaterialSize=3u,ShadowRayScalarLanes=4u;
static const uint ShadowRayNodeMaximum=1u,ShadowRayNodeLinks=2u;
static const uint ShadowRayTriangleUV01=3u,ShadowRayTriangleUV2=4u,ShadowRayTriangleMetadata=5u;
static const uint ShadowRayMaterialOpacity=1u,ShadowRayMaterialCoverageWord=2u;
uint4 ShadowRayLoad(uint address)
{ uint width,height; ShadowRayScene.GetDimensions(width,height); width=max(width,1u); return ShadowRayScene.Load(int3(address%width,address/width,0)); }
struct ShadowRayNode { float4 Lower,Upper; uint4 Link; };
struct ShadowRayTriangle { float4 P0,P1,P2,UV01,UV2; uint4 Identity; };
struct ShadowRayMaterial { uint4 Diffuse,Opacity; float4 Alpha; };
static const float ShadowRayHalfPixel=0.5;
static const float ShadowRayBinary32Epsilon=1.1920928955078125e-7;
static const float ShadowRayEdgeRoundoff=3*ShadowRayBinary32Epsilon/(1-3*ShadowRayBinary32Epsilon);
ShadowRayNode ShadowRayReadNode(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).x+index*ShadowRayNodeSize;
    ShadowRayNode result;
    result.Lower=asfloat(ShadowRayLoad(base));
    result.Upper=asfloat(ShadowRayLoad(base+1u));
    result.Link=ShadowRayLoad(base+2u);
    return result;
}
ShadowRayTriangle ShadowRayReadTriangle(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).y+index*ShadowRayTriangleSize;
    ShadowRayTriangle result;
    result.P0=asfloat(ShadowRayLoad(base));
    result.P1=asfloat(ShadowRayLoad(base+1u));
    result.P2=asfloat(ShadowRayLoad(base+2u));
    result.UV01=asfloat(ShadowRayLoad(base+3u));
    result.UV2=asfloat(ShadowRayLoad(base+4u));
    result.Identity=ShadowRayLoad(base+5u);
    return result;
}
ShadowRayMaterial ShadowRayReadMaterial(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).z+index*ShadowRayMaterialSize;
    ShadowRayMaterial result;
    result.Diffuse=ShadowRayLoad(base);
    result.Opacity=ShadowRayLoad(base+1u);
    result.Alpha=asfloat(ShadowRayLoad(base+2u));
    return result;
}
float ShadowRayScalar(uint index)
{ uint4 value=ShadowRayLoad(ShadowRayLoad(ShadowRayOffsetsHeader).w+index/ShadowRayScalarLanes); return asfloat(value)[index%ShadowRayScalarLanes]; }
float ShadowRayReadCoverage(uint4 map,float2 uv)
{
    if (!map.w) return 1;
    float2 p=frac(float2(uv.x,-uv.y))*float2(map.yz)-ShadowRayHalfPixel;
    uint x=p.x<0 ? map.y-1 : uint(floor(p.x)),y=p.y<0 ? map.z-1 : uint(floor(p.y));
    uint nextX=x+1==map.y ? 0 : x+1,nextY=y+1==map.z ? 0 : y+1;
    float upper=lerp(ShadowRayScalar(map.x+y*map.y+x),ShadowRayScalar(map.x+y*map.y+nextX),frac(p.x));
    float lower=lerp(ShadowRayScalar(map.x+nextY*map.y+x),ShadowRayScalar(map.x+nextY*map.y+nextX),frac(p.x));
    return lerp(upper,lower,frac(p.y));
}
bool ShadowRayInBounds(ShadowRayNode node,float3 origin,float3 ray,float limit)
{
    float nearDistance=0,farDistance=limit;
    [unroll] for (uint dimension=0;dimension<3;++dimension)
    {
        if (ray[dimension]==0)
        {
            if (origin[dimension]<node.Lower[dimension] || origin[dimension]>node.Upper[dimension]) return false;
            continue;
        }
        float first=(node.Lower[dimension]-origin[dimension])/ray[dimension];
        float last=(node.Upper[dimension]-origin[dimension])/ray[dimension];
        nearDistance=max(nearDistance,min(first,last)); farDistance=min(farDistance,max(first,last));
        if (nearDistance>farDistance) return false;
    }
    return true;
}
float ShadowRayEdgeFunction(float2 first,float2 second)
{
    precise float positive=first.x*second.y,negative=first.y*second.x;
    precise float signedArea=positive-negative;
    return abs(signedArea)<=ShadowRayEdgeRoundoff*(abs(positive)+abs(negative)) ? 0 : signedArea;
}
bool ShadowRayHitTriangle(ShadowRayTriangle primitive,float3 origin,float3 ray,float limit,out float3 barycentric)
{
    barycentric=0;
    uint major=abs(ray.x)>abs(ray.y) ? 0 : 1; if (abs(ray.z)>abs(ray[major])) major=2;
    uint horizontal=(major+1)%3,vertical=(horizontal+1)%3;
    float3 first=primitive.P0.xyz-origin,second=primitive.P1.xyz-origin,third=primitive.P2.xyz-origin;
    float2 slope=float2(ray[horizontal],ray[vertical])/ray[major];
    precise float2 a=float2(first[horizontal],first[vertical])-slope*first[major];
    precise float2 b=float2(second[horizontal],second[vertical])-slope*second[major];
    precise float2 c=float2(third[horizontal],third[vertical])-slope*third[major];
    float3 area=float3(ShadowRayEdgeFunction(b,c),ShadowRayEdgeFunction(c,a),ShadowRayEdgeFunction(a,b));
    if (any(area<0) && any(area>0)) return false;
    float determinant=area.x+area.y+area.z; if (determinant==0) return false;
    float distance=dot(area,float3(first[major],second[major],third[major]))/(determinant*ray[major]);
    if (!(distance>0 && distance<=limit)) return false;
    barycentric=area/determinant; return true;
}
bool ShadowRayAnyAcceptedHit(float3 origin,float3 ray,float limit)
{
    bool blocked=false;
    uint address=0;
    [loop] while (address<ShadowRayLoad(0u).y && !blocked)
    {
        ShadowRayNode node=ShadowRayReadNode(address);
        if (!ShadowRayInBounds(node,origin,ray,limit)) { address=node.Link.w; continue; }
        if (!node.Link.z) { address=node.Link.x; continue; }
        [loop] for (uint local=0;local<node.Link.z && !blocked;++local)
        {
            ShadowRayTriangle primitive=ShadowRayReadTriangle(node.Link.y+local);
            if ((primitive.Identity.w & ShadowRayCasterFlag)==0u) continue; float3 barycentric;
            if (!ShadowRayHitTriangle(primitive,origin,ray,limit,barycentric)) continue;
            ShadowRayMaterial material=ShadowRayReadMaterial(primitive.Identity.x); if (material.Alpha.y<=0) { blocked=true; break; }
            float2 uv=barycentric.x*primitive.UV01.xy+barycentric.y*primitive.UV01.zw+barycentric.z*primitive.UV2.xy;
            float alpha=dot(barycentric,float3(primitive.P0.w,primitive.P1.w,primitive.P2.w))*material.Alpha.x;
            alpha*=ShadowRayReadCoverage(material.Diffuse,uv)*ShadowRayReadCoverage(material.Opacity,uv);
            if (saturate(alpha)>=material.Alpha.y) { blocked=true; break; }
        }
        address=node.Link.w;
    }
    return blocked;
}
static const float ShadowRayNormalFloor=1e-12,ShadowRayOriginRoundoff=8.0*1.1920928955078125e-7;
bool ShadowRayEnabled() { return ShadowRayLoad(ShadowRayCountsHeader).x!=0u && ShadowRayLoad(ShadowRayLightHeader).y==ShadowRayImageABI; }
float ShadowRayVisibility(float3 position,float3 normal,float3 towardLight)
{
    if (!ShadowRayEnabled()) return 1.0;
    float4 parameters=asfloat(ShadowRayLoad(ShadowRayTraceHeader));
    float scale=max(abs(towardLight.x),max(abs(towardLight.y),abs(towardLight.z)));
    if (!(scale>0.0) || !(all(isfinite(float4(position,scale))))) return 1.0;
    float3 scaled=towardLight/scale; float magnitude=length(scaled);
    float3 direction=scaled/magnitude;
    bool finiteEmitter=ShadowRayLoad(ShadowRayLightHeader).x!=0u;
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
    float visibility=1.0;
    if (ShadowRayAnyAcceptedHit(origin,direction,limit)) visibility=0.0;
    return visibility;
}

// Finite support is part of the light model, shared by every emitter sample.
// Karis' smooth attenuation window; Filament, Punctual lights.
float LightRangeWindow(float3 displacement, float range)
{
    if (range == 0.0f) return 1.0f;
    float3 magnitude = abs(displacement);
    if (max(magnitude.x, max(magnitude.y, magnitude.z)) >= range) return 0.0f;
    float3 scaled = displacement / range;
    float squaredRatio = dot(scaled, scaled);
    float fade = max(1.0f - squaredRatio * squaredRatio, 0.0f);
    return fade * fade;
}

// Native DX12 single scattering. Injection integrates one local depth interval;
// an ordered inclusive scan composes affine radiance/transmittance operators.
// Hillis and Steele, "Data Parallel Algorithms" (1986).
// HG phase conventions: https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
// Primary opaque shadows are sampled explicitly; light-path extinction and
// multiple scattering are omitted. Rectangle visibility uses its primary map.
cbuffer MediumRoot : register(b0)
{
    row_major float4x4 InverseCamera;
    float4 Density; // Extinction, base height, height falloff, maximum distance.
    float4 AmbientRadiance;
    float4 ViewAndGrid; // Viewport width/height, grid width/height.
    uint4 Integration; // Consumer pass, depth slices, samples per slice, enabled.
};
struct SceneLight
{
    float4 PositionKind;
    float4 DirectionPower;
    float4 ColorShadow;
    float4 RectangleSize;
};
// Full scene lights include emitters affecting points before the opaque surface.
StructuredBuffer<SceneLight> VolumeSceneLights : register(t16);
static const uint LightLimit = 4;
static const uint CascadeLimit = 4;
cbuffer LightUpload : register(b1)
{
    float4 AlbedoAsymmetry;
    float4 LightScaleBias;
    uint4 Dispatch; // Light count, shadow mode, shadow light index, scan stride.
    row_major float4x4 ShadowCamera;
    SceneLight SceneLights[LightLimit];
    row_major float4x4 CascadeCameras[CascadeLimit];
    float4 CascadeEnds;
    float4 CascadeBlendBegin;
    float4 CameraDepthPlane;
    float4 CascadeRangeCount;
    float4 Filter; // Radius, tap count, hierarchy tag, reserved.
    float4 MediumShadowEncoding[CascadeLimit + 1];
};
Texture2D<float4> ScanInput : register(t0);
Texture2D<float> ShadowDepth : register(t1);
Texture2D<float> CascadeDepth0 : register(t2);
Texture2D<float> CascadeDepth1 : register(t3);
Texture2D<float> CascadeDepth2 : register(t4);
Texture2D<float> CascadeDepth3 : register(t5);

static const float Pi = 3.14159265358979323846;
static const float OpticalDepthLimit = 80.0;
static const float RayEpsilon = 1.0e-6;
static const float SquaredDistanceFloor = 1.0e-4;
static const float ClipEpsilon = 1.0e-7;
static const float HalfFloatMaximum = 65504.0;
static const float ParallelAxisThreshold = 0.99;
static const uint DirectionalKind = 0, PointKind = 1;
static const uint ShadowSingle = 1, ShadowCascaded = 2;
static const uint RectangleAxisSamples = 2;
static const float RectangleSampleWeight = 1.0 / (RectangleAxisSamples * RectangleAxisSamples);

float4 VSMain(uint vertex : SV_VertexID) : SV_POSITION
{
    const float TriangleExtent = 3.0;
    return float4(vertex == 1 ? TriangleExtent : -1,
        vertex == 2 ? TriangleExtent : -1, 0, 1);
}
float3 UnitDirection(float3 value, float3 fallback)
{
    float magnitudeSquared = dot(value, value);
    return magnitudeSquared > SquaredDistanceFloor ? value * rsqrt(magnitudeSquared) : fallback;
}
float3 Unproject(float2 uv, float depth)
{
    float4 homogeneous = mul(InverseCamera, float4(2 * uv.x - 1, 1 - 2 * uv.y, depth, 1));
    return homogeneous.xyz / homogeneous.w;
}
float AngularScattering(float3 towardLight, float3 awayFromEye)
{
    // Both physical travel directions are negatives of these vectors. Positive
    // asymmetry therefore peaks when their dot product approaches one.
    float cosine = clamp(dot(towardLight, awayFromEye), -1.0, 1.0);
    float g = AlbedoAsymmetry.w;
    float denominator = 1 + g * g - 2 * g * cosine;
    return (1 - g * g) / (4 * Pi * denominator * sqrt(denominator));
}
float CompareDepth(Texture2D<float> image, int2 coordinate, int2 extent, float receiver)
{
    if (any(coordinate < 0) || any(coordinate >= extent)) return 1;
    return receiver <= image.Load(int3(coordinate, 0)) ? 1.0 : 0.0;
}
float FilteredDepth(Texture2D<float> image, float2 uv, int2 extent, float receiver)
{
    float2 location = uv * extent - 0.5;
    int2 base = int2(floor(location));
    float2 weight = frac(location);
    float top = lerp(CompareDepth(image, base, extent, receiver),
        CompareDepth(image, base + int2(1, 0), extent, receiver), weight.x);
    float bottom = lerp(CompareDepth(image, base + int2(0, 1), extent, receiver),
        CompareDepth(image, base + int2(1, 1), extent, receiver), weight.x);
    return lerp(top, bottom, weight.y);
}
float ShadowAt(Texture2D<float> image, float4 clip)
{
    if (clip.w <= ClipEpsilon) return 1;
    float3 ndc = clip.xyz / clip.w;
    float2 uv = float2(ndc.x, -ndc.y) * 0.5 + 0.5;
    if (ndc.z < 0 || ndc.z > 1 || any(uv < 0) || any(uv > 1)) return 1;
    uint width, height; image.GetDimensions(width, height);
    int2 extent = int2(width, height);
    float receiver = ndc.z - LightScaleBias.y;
    int radius = int(Filter.x);
    if (radius == 0) return CompareDepth(image, min(int2(uv * extent), extent - 1), extent, receiver);
    float total = 0;
    [loop] for (int row = -radius; row <= radius; ++row)
        [loop] for (int column = -radius; column <= radius; ++column)
            total += FilteredDepth(image, uv + float2(column, row) / extent, extent, receiver);
    return total / Filter.y;
}
float CascadeAt(uint cascade, float3 world)
{
    float4 clip = mul(CascadeCameras[cascade], float4(world, 1));
    if (cascade == 0) return ShadowAt(CascadeDepth0, clip);
    if (cascade == 1) return ShadowAt(CascadeDepth1, clip);
    if (cascade == 2) return ShadowAt(CascadeDepth2, clip);
    return ShadowAt(CascadeDepth3, clip);
}
// Virtual depth and page directory share the exclusive cascade slots.
// Each PCF corner resolves its own page or pinned coarse ancestor.
static const uint VirtualGutter = 1;
static const float VirtualPixelCenter = 0.5f;
float2 VirtualStoredDepth(float2 coordinate, uint minimumLevel)
{
    uint4 hierarchy = uint4(CascadeEnds), atlas = uint4(CascadeBlendBegin);
    uint rowPages = hierarchy.x / hierarchy.y, levelOffset = 0;
    [loop] for (uint level = 0; level < hierarchy.z; ++level)
    {
        if (level >= minimumLevel)
        {
            uint resolution = rowPages * hierarchy.y;
            uint2 texel = min(uint2(coordinate * resolution), resolution - 1);
            uint2 page = texel / hierarchy.y;
            uint entryIndex = levelOffset + page.x + page.y * rowPages;
            int2 tablePixel = int2(entryIndex % hierarchy.w, entryIndex / hierarchy.w);
            uint encodedSlot = uint(round(CascadeDepth1.Load(int3(tablePixel, 0)).x * atlas.w));
            if (encodedSlot > 0)
            {
                uint slot = encodedSlot - 1;
                uint2 origin = uint2(slot % atlas.z, slot / atlas.z) * (hierarchy.y + 2 * VirtualGutter);
                uint2 location = origin + VirtualGutter + texel % hierarchy.y;
                return float2(CascadeDepth0.Load(int3(location, 0)).x, level);
            }
        }
        levelOffset += rowPages * rowPages; rowPages /= 2;
    }
    return float2(1, hierarchy.z - 1);
}
float VirtualComparison(int2 location, uint level, float reference)
{
    uint size = uint(CascadeEnds.x) >> level;
    float2 coordinate = (float2(clamp(location, 0, int(size) - 1)) + VirtualPixelCenter) / size;
    return reference <= VirtualStoredDepth(coordinate, level).x ? 1 : 0;
}
// The hard/PCF metric suffix stores camera rows for an explicit cell footprint.
// Neighbor reconstruction avoids gradients in a nonuniform medium integration.
uint VirtualVolumeMip(float3 position, float2 coordinate)
{
    float4 world = float4(position, 1);
    float4 camera = float4(dot(MediumShadowEncoding[0], world), dot(MediumShadowEncoding[1], world), dot(MediumShadowEncoding[2], world), dot(MediumShadowEncoding[3], world));
    if (camera.w <= CameraDepthPlane.w) return 0;
    float3 center = camera.xyz / camera.w;
    float4 right = mul(InverseCamera, float4(center + float3(CameraDepthPlane.y * 2, 0, 0), 1));
    float4 lower = mul(InverseCamera, float4(center - float3(0, CameraDepthPlane.z * 2, 0), 1));
    float4 a = mul(CascadeCameras[0], float4(right.xyz / right.w, 1));
    float4 b = mul(CascadeCameras[0], float4(lower.xyz / lower.w, 1));
    if (min(a.w, b.w) <= CameraDepthPlane.w) return uint(CascadeEnds.z - 1);
    float2 uvA = float2(a.x / a.w + 1, 1 - a.y / a.w) * VirtualPixelCenter;
    float2 uvB = float2(b.x / b.w + 1, 1 - b.y / b.w) * VirtualPixelCenter;
    float size = max(length(uvA - coordinate), length(uvB - coordinate)) * CascadeEnds.x;
    return uint(clamp(floor(log2(max(1, size)) + CameraDepthPlane.x), 0, CascadeEnds.z - 1));
}

float VirtualVisibility(float3 position, float bias)
{
    float4 clip = mul(CascadeCameras[0], float4(position, 1));
    float3 projected = clip.xyz / max(clip.w, CameraDepthPlane.w);
    float2 coordinate = float2(projected.x + 1, 1 - projected.y) * VirtualPixelCenter;
    uint level = VirtualVolumeMip(position, coordinate);
    if (clip.w <= CameraDepthPlane.w || projected.z < 0 || projected.z > 1 || any(coordinate < 0) || any(coordinate > 1)) return 1;
    float2 depthAndLevel = VirtualStoredDepth(coordinate, level);
    float reference = projected.z - bias;
    if (Filter.x == 0) return reference <= depthAndLevel.x ? 1 : 0;
    level = uint(depthAndLevel.y);
    float2 samplePosition = coordinate * (uint(CascadeEnds.x) >> level) - VirtualPixelCenter;
    int2 first = int2(floor(samplePosition)); float2 weights = frac(samplePosition); float sum = 0;
    [loop] for (int row = -int(Filter.x); row <= int(Filter.x); ++row)
        [loop] for (int column = -int(Filter.x); column <= int(Filter.x); ++column)
        {
            int2 location = first + int2(column, row);
            float upper = lerp(VirtualComparison(location, level, reference), VirtualComparison(location + int2(1, 0), level, reference), weights.x);
            float lower = lerp(VirtualComparison(location + int2(0, 1), level, reference), VirtualComparison(location + 1, level, reference), weights.x);
            sum += lerp(upper, lower, weights.y);
        }
    return sum / Filter.y;
}


float PrimaryShadow(float3 world)
{
    if (ShadowRayEnabled()) return 1.0;
    if (Filter.z != 0 && Dispatch.y != 0) return VirtualVisibility(world, LightScaleBias.y);
    if (Dispatch.y == ShadowSingle) return ShadowAt(ShadowDepth, mul(ShadowCamera, float4(world, 1)));
    if (Dispatch.y != ShadowCascaded) return 1;
    float depth = dot(CameraDepthPlane, float4(world, 1));
    if (depth < CascadeRangeCount.x || depth > CascadeRangeCount.y) return 1;
    uint cascade = 0, count = uint(CascadeRangeCount.z);
    while (cascade + 1 < count && depth > CascadeEnds[cascade]) ++cascade;
    float visibility = CascadeAt(cascade, world);
    if (cascade + 1 < count && depth > CascadeBlendBegin[cascade])
    {
        float overlap = max(CascadeEnds[cascade] - CascadeBlendBegin[cascade], RayEpsilon);
        float blend = saturate((depth - CascadeBlendBegin[cascade]) / overlap);
        visibility = lerp(visibility, CascadeAt(cascade + 1, world), blend);
    }
    return visibility;
}
float LightIntegral(SceneLight light, float3 world, float3 cameraRay)
{
    uint kind = uint(light.PositionKind.w);
    float3 normal = UnitDirection(light.DirectionPower.xyz, float3(0, 1, 0));
    if (kind == DirectionalKind) return AngularScattering(normal, cameraRay);
    if (kind == PointKind)
    {
        float3 delta = light.PositionKind.xyz - world;
        return AngularScattering(UnitDirection(delta, normal), cameraRay) * LightRangeWindow(delta, light.RectangleSize.z) / max(dot(delta, delta), SquaredDistanceFloor);
    }
    float3 helper = abs(normal.y) < ParallelAxisThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 axisX = normalize(cross(helper, normal)), axisY = cross(normal, axisX);
    float integral = 0;
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.RectangleSize.w > 0.0;
    uint emitterGrid = physicalEmitter ? uint(clamp(light.RectangleSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : RectangleAxisSamples;
    float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
    [loop] for (uint row = 0; row < emitterGrid; ++row)
        [loop] for (uint column = 0; column < emitterGrid; ++column)
        {
            float2 location = ((float2(column, row) + 0.5) / emitterGrid - 0.5) * light.RectangleSize.xy;
            float3 delta = light.PositionKind.xyz + axisX * location.x + axisY * location.y - world;
            float3 towardLight = UnitDirection(delta, -normal);
            integral += emitterSampleWeight * saturate(dot(normal, -towardLight)) *
                AngularScattering(towardLight, cameraRay) * LightRangeWindow(delta, light.RectangleSize.z) / max(dot(delta, delta), SquaredDistanceFloor);
        }
    return integral * light.RectangleSize.x * light.RectangleSize.y;
}
float3 SourceRadiance(float3 world, float3 cameraRay)
{
    float primaryVisibility = Dispatch.y == 0 ? 1 : PrimaryShadow(world);
    float3 direct = 0;
    [loop] for (uint index = 0; index < Dispatch.x; ++index)
    {
        SceneLight light;
        if (LightScaleBias.z != 0.0) light = VolumeSceneLights[index];
        else light = SceneLights[index];
        float3 radiance = min(light.ColorShadow.rgb, HalfFloatMaximum) * min(light.DirectionPower.w, HalfFloatMaximum);
        float3 rayToward=uint(light.PositionKind.w)==0u?light.DirectionPower.xyz:light.PositionKind.xyz-world;
            float visibility=Dispatch.y!=0 && index==Dispatch.z?(ShadowRayEnabled()?ShadowRayVisibility(world,float3(0,0,0),rayToward):primaryVisibility):1.0;
        direct += radiance * (LightIntegral(light, world, cameraRay) * visibility);
    }
    return AmbientRadiance.rgb + AlbedoAsymmetry.rgb * LightScaleBias.x * direct;
}
float4 PSInject(float4 pixel : SV_POSITION) : SV_TARGET
{
    uint2 coordinate = uint2(pixel.xy);
    uint tileWidth = uint(ViewAndGrid.z), slice = coordinate.x / tileWidth;
    float2 uv = (float2(coordinate.x % tileWidth, coordinate.y) + 0.5) / ViewAndGrid.zw;
    float3 origin = Unproject(uv, 0), ray = Unproject(uv, 1) - origin;
    float lengthToFar = max(length(ray), RayEpsilon);
    float3 direction = ray / lengthToFar;
    float stepSize = min(lengthToFar, Density.w) / (Integration.y * Integration.z);
    float4 segment = float4(0, 0, 0, 1);
    [loop] for (uint stepIndex = 0; stepIndex < Integration.z; ++stepIndex)
    {
        float distance = (slice * Integration.z + stepIndex + 0.5) * stepSize;
        float3 world = origin + direction * distance;
        float heightExponent = min(Density.z * max(world.y - Density.y, 0), OpticalDepthLimit);
        float sigmaT = Density.x * exp(-heightExponent);
        float transmission = exp(-min(sigmaT * stepSize, OpticalDepthLimit));
        segment.rgb += segment.a * (1 - transmission) * SourceRadiance(world, direction);
        segment.a *= transmission;
    }
    return segment;
}
float4 PSPrefix(float4 pixel : SV_POSITION) : SV_TARGET
{
    int2 coordinate = int2(pixel.xy);
    uint tileWidth = uint(ViewAndGrid.z), slice = uint(coordinate.x) / tileWidth;
    float4 result = ScanInput.Load(int3(coordinate, 0));
    if (slice >= Dispatch.w)
    {
        float4 preceding = ScanInput.Load(int3(coordinate - int2(Dispatch.w * tileWidth, 0), 0));
        result = float4(preceding.rgb + preceding.a * result.rgb, preceding.a * result.a);
    }
    // FP32 ping-pong stages retain bright scattering and near-unit transmission.
    // Clamp every slice only when storing the final shared FP16 atlas.
    if (Dispatch.w * 2 >= Integration.y) result.rgb = min(result.rgb, HalfFloatMaximum);
    return result;
}
