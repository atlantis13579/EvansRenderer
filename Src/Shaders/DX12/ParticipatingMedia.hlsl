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

// Independent DX12 participating-medium transport and opaque composition.
// Pharr, Jakob and Humphreys, PBRT v4, Volume Scattering and Equation of Transfer.
// https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance
// EVANS_MEDIA_COMPOSITE selects the resolved-scene consumer of paired S.rgb/T.rgb planes.
#if defined(EVANS_MEDIA_COMPOSITE)
cbuffer VolumeParameters : register(b4)
{
    row_major float4x4 InverseRasterCamera;
    float4 HeightMedium; // base extinction, base height, falloff, ray-distance cap
    float4 AmbientSource;
    float4 ViewAndGridSize;
    uint4 VolumeControl; // phase, endpoint count, samples per interval, enabled
};
Texture2D<float4> OpaqueRadiance : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> IntegratedVolume : register(t14);
SamplerState VolumeClampSampler : register(s1);

static const uint VolumeIntegrationPhase = 0;
static const float RayLengthFloor = 1.0e-6f;
static const float HalfFloatRadianceLimit = 65504.0f;

float4 VSMain(uint vertexId : SV_VertexID) : SV_POSITION
{
    const float2 corners[3] = {float2(-1, 1), float2(3, 1), float2(-1, -3)};
    return float4(corners[vertexId], 0, 1);
}
float3 UnprojectPoint(float2 screenUV, float depth)
{
    float2 ndc = float2(screenUV.x * 2 - 1, 1 - screenUV.y * 2);
    float4 homogeneousPoint = mul(InverseRasterCamera, float4(ndc, depth, 1));
    return homogeneousPoint.xyz / homogeneousPoint.w;
}
struct VolumeRay
{
    float3 Origin;
    float3 Direction;
    float Distance;
};
VolumeRay CameraRay(float2 uv)
{
    VolumeRay ray;
    ray.Origin = UnprojectPoint(uv, 0);
    float3 delta = UnprojectPoint(uv, 1) - ray.Origin;
    float lengthToFar = max(length(delta), RayLengthFloor);
    ray.Direction = delta / lengthToFar;
    ray.Distance = min(lengthToFar, HeightMedium.w);
    return ray;
}

float4 ReadPrefixTransmission(float2 uv, uint boundary)
{
    if (boundary == 0) return float4(1, 1, 1, 1);
    float2 gridPosition = clamp(uv * ViewAndGridSize.zw, 0.5f, ViewAndGridSize.zw - 0.5f);
    gridPosition.x += float(boundary - 1) * ViewAndGridSize.z;
    float2 atlasSize = ViewAndGridSize.zw * float2(VolumeControl.y, 1);
    gridPosition.y = (gridPosition.y + atlasSize.y) * 0.5;
    return IntegratedVolume.SampleLevel(VolumeClampSampler, gridPosition / atlasSize, 0);
}
float4 ReadPrefix(float2 uv, uint boundary)
{
    if (boundary == 0) return float4(0, 0, 0, 1);
    float2 gridPosition = clamp(uv * ViewAndGridSize.zw, 0.5f, ViewAndGridSize.zw - 0.5f);
    gridPosition.x += float(boundary - 1) * ViewAndGridSize.z;
    float2 atlasSize = ViewAndGridSize.zw * float2(VolumeControl.y, 1);
    gridPosition.y *= 0.5;
    return IntegratedVolume.SampleLevel(VolumeClampSampler, gridPosition / atlasSize, 0);
}
float4 ResolveOpaque(uint2 pixel, float2 uv)
{
    VolumeRay ray = CameraRay(uv);
    float3 surfacePoint = UnprojectPoint(uv, OpaqueDepth.Load(int3(pixel, 0)));
    float distanceFraction = saturate(length(surfacePoint - ray.Origin) / max(ray.Distance, RayLengthFloor));
    float slicePosition = distanceFraction * VolumeControl.y;
    uint first = (uint)slicePosition;
    uint second = min(first + 1, VolumeControl.y);
    float4 transport = lerp(ReadPrefix(uv, first), ReadPrefix(uv, second), slicePosition - first);
    float3 transmittance = (lerp(ReadPrefixTransmission(uv, first), ReadPrefixTransmission(uv, second), slicePosition - first)).rgb;
    float4 surface = OpaqueRadiance.Load(int3(pixel, 0));
    return float4(clamp(surface.rgb * transmittance + transport.rgb, 0, HalfFloatRadianceLimit), surface.a);
}
float4 PSMain(float4 rasterPosition : SV_POSITION) : SV_TARGET
{
    uint2 pixel = (uint2)rasterPosition.xy;
    return ResolveOpaque(pixel, rasterPosition.xy / ViewAndGridSize.xy);
}

#else
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

// Native DX12 RGB participating-medium transport in a finite density field;
// an ordered inclusive scan composes affine radiance/transmittance operators.
// Hillis and Steele, "Data Parallel Algorithms" (1986).
// HG phase conventions: https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
// Primary opaque shadows and incident-light extinction are included.
// Multiple scattering is omitted. Rectangle emitters use finite quadrature.
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
    float4 Filter; // Radius, tap count, receiver bias, normal bias.
    float4 MediumShadowEncoding[CascadeLimit + 1];
    float4 MediumShadowParameters;
    uint4 MediumShadowOptions;
    float4 AbsorptionScale;
    float4 ScatteringVariation;
    float4 EmissiveFrequency;
    float4 MediumLow;
    float4 MediumHigh;
    uint4 MediumControl;
    float4 DensityLayout;
};
Texture2D<float4> ScanInput : register(t0);
Texture2D<float4> ShadowDepth : register(t1);
Texture2D<float4> CascadeDepth0 : register(t2);
Texture2D<float4> CascadeDepth1 : register(t3);
Texture2D<float4> CascadeDepth2 : register(t4);
Texture2D<float4> CascadeDepth3 : register(t5);
Texture2D<float4> DensityImage : register(t6);

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
float CompareDepth(Texture2D<float4> image, int2 coordinate, int2 extent, float receiver)
{
    if (any(coordinate < 0) || any(coordinate >= extent)) return 1;
    return receiver <= image.Load(int3(coordinate, 0)).r ? 1.0 : 0.0;
}
float FilteredDepth(Texture2D<float4> image, float2 uv, int2 extent, float receiver)
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
static const float PixelCenterOffset = 0.5, ClipWMinimum = 1.0e-7, MapRadiusFraction = 0.25;
static const uint MediumSoftFilter = 2, MediumMomentFilter = 3;
float ReadShadowDepth(Texture2D<float4> depth, int2 location, int2 extent)
{
    float result = 1;
    if (all(location >= 0) && all(location < extent)) result = depth.Load(int3(location, 0)).r;
    return result;
}
float CompareBilinear(Texture2D<float4> depth, float2 uv, int2 extent, float receiver)
{
    float2 location = uv * extent - PixelCenterOffset;
    int2 cell = int2(floor(location));
    float2 fraction = frac(location);
    float upperLeft = receiver <= ReadShadowDepth(depth, cell, extent);
    float upperRight = receiver <= ReadShadowDepth(depth, cell + int2(1, 0), extent);
    float lowerLeft = receiver <= ReadShadowDepth(depth, cell + int2(0, 1), extent);
    float lowerRight = receiver <= ReadShadowDepth(depth, cell + 1, extent);
    return lerp(lerp(upperLeft, upperRight, fraction.x), lerp(lowerLeft, lowerRight, fraction.x), fraction.y);
}
float AxisDistance(float depth, float4 projection)
{
    return projection.x == 0 ? depth * projection.y :
        projection.x * projection.y / lerp(projection.y, projection.x, depth);
}
float2 UnitDisk(uint index, uint count)
{
    const float GoldenAngle = Pi * (3 - sqrt(5.0));
    float radius = sqrt((index + PixelCenterOffset) / count);
    float sine = 0, cosine = 1;
    sincos(GoldenAngle * index, sine, cosine);
    return radius * float2(cosine, sine);
}
float2 ProjectRadius(float planeRadius, float2 planeSize, int2 extent, float texelBudget)
{
    float2 footprint = planeRadius / planeSize;
    float longestAxis = max(footprint.x * extent.x, footprint.y * extent.y);
    float cap = min(texelBudget, min(extent.x, extent.y) * MapRadiusFraction);
    if (longestAxis > cap) footprint *= cap / longestAxis;
    return footprint;
}
float SoftVisibility(Texture2D<float4> depth, uint mapIndex, float4 clip, float bias)
{
    float result = 1;
    if (clip.w > ClipWMinimum && (MediumShadowOptions.z & (1u << mapIndex)) != 0)
    {
        float3 projected = clip.xyz / clip.w;
        float2 uv = float2(projected.x + 1, 1 - projected.y) / 2;
        if (all(uv >= 0) && all(uv <= 1) && projected.z >= 0 && projected.z <= 1)
        {
            uint width = 0, height = 0;
            depth.GetDimensions(width, height);
            int2 extent = int2(width, height);
            float4 projection = MediumShadowEncoding[mapIndex];
            bool finiteLight = projection.x > 0;
            float emitterSize = finiteLight ? MediumShadowParameters.x : MediumShadowParameters.y;
            float comparisonDepth = projected.z - bias;
            if (emitterSize == 0)
                result = comparisonDepth <= ReadShadowDepth(depth, int2(floor(uv * extent)), extent);
            else
            {
                float receiverDistance = AxisDistance(projected.z, projection);
                float searchSize = finiteLight ? emitterSize * (receiverDistance - projection.x) / receiverDistance :
                    emitterSize * receiverDistance;
                float2 search = ProjectRadius(searchSize, projection.zw, extent, MediumShadowParameters.z);
                float linearBlockerSum = 0;
                uint blockerCount = 0;
                [loop] for (uint blockerIndex = 0; blockerIndex < MediumShadowOptions.x; ++blockerIndex)
                {
                    int2 location = int2(floor((uv + search * UnitDisk(blockerIndex, MediumShadowOptions.x)) * extent));
                    float candidate = ReadShadowDepth(depth, location, extent);
                    if (candidate < comparisonDepth)
                    {
                        linearBlockerSum += AxisDistance(candidate, projection);
                        ++blockerCount;
                    }
                }
                if (blockerCount != 0)
                {
                    float blockerDistance = linearBlockerSum / blockerCount;
                    float separation = max(receiverDistance - blockerDistance, 0);
                    float penumbraSize = emitterSize * separation;
                    if (finiteLight) penumbraSize *= projection.x / (blockerDistance * receiverDistance);
                    float2 penumbra = ProjectRadius(penumbraSize, projection.zw, extent, MediumShadowParameters.w);
                    float comparisonSum = 0;
                    [loop] for (uint tapIndex = 0; tapIndex < MediumShadowOptions.y; ++tapIndex)
                        comparisonSum += CompareBilinear(depth, uv + penumbra * UnitDisk(tapIndex, MediumShadowOptions.y), extent, comparisonDepth);
                    result = comparisonSum / MediumShadowOptions.y;
                }
            }
        }
    }
    return result;
}
float2 ReadMoments(Texture2D<float4> image, int2 pixel, int2 extent)
{
    // The far clear plane also represents samples outside a map.
    if (any(pixel < 0) || any(pixel >= extent)) return float2(1, 1);
    return image.Load(int3(pixel, 0)).rg;
}
float2 FilteredMoments(Texture2D<float4> image, float2 uv)
{
    uint width = 0, height = 0; image.GetDimensions(width, height);
    int2 extent = int2(width, height);
    float2 position = uv * extent - PixelCenterOffset;
    int2 lower = int2(floor(position)); float2 blend = frac(position);
    float2 upperRow = lerp(ReadMoments(image, lower, extent), ReadMoments(image, lower + int2(1, 0), extent), blend.x);
    float2 lowerRow = lerp(ReadMoments(image, lower + int2(0, 1), extent), ReadMoments(image, lower + 1, extent), blend.x);
    return lerp(upperRow, lowerRow, blend.y);
}
float MomentVisibility(Texture2D<float4> image, uint projection, float4 clip, float linearBias)
{
    if (clip.w <= ClipWMinimum || (MediumShadowOptions.x & (1u << projection)) == 0) return 1;
    float3 device = clip.xyz / clip.w;
    float2 uv = float2(device.x + 1, 1 - device.y) / 2;
    if (any(uv < 0) || any(uv > 1) || device.z < 0 || device.z > 1) return 1;
    float receiver = dot(MediumShadowEncoding[projection].xyz, float3(clip.zw, 1));
    float2 moments = FilteredMoments(image, uv);
    float separation = receiver - linearBias - moments.x;
    if (separation <= 0) return 1;
    float variance = max(moments.y - moments.x * moments.x, MediumShadowParameters.x);
    float probability = variance / (variance + separation * separation);
    // Bleeding suppression is an artistic remap, not a tighter probability bound.
    return saturate((probability - MediumShadowParameters.y) / (1 - MediumShadowParameters.y));
}

// Keep a single result path around nested sampling loops. This also makes
// the SM5 compiler's definite-assignment analysis explicit after inlining.
float ShadowAt(Texture2D<float4> image, float4 clip, uint projection)
{
    float visibility = 1;
    if (MediumControl.y == MediumSoftFilter)
        visibility = SoftVisibility(image, projection, clip, LightScaleBias.y);
    else if (MediumControl.y == MediumMomentFilter)
        visibility = MomentVisibility(image, projection, clip, LightScaleBias.y);
    else if (clip.w > ClipEpsilon)
    {
        float3 ndc = clip.xyz / clip.w;
        float2 uv = float2(ndc.x, -ndc.y) * 0.5 + 0.5;
        if (ndc.z >= 0 && ndc.z <= 1 && all(uv >= 0) && all(uv <= 1))
        {
            uint width = 0, height = 0; image.GetDimensions(width, height);
            int2 extent = int2(width, height);
            float receiver = ndc.z - LightScaleBias.y;
            int radius = int(Filter.x);
            if (radius == 0) visibility = CompareDepth(image, min(int2(uv * extent), extent - 1), extent, receiver);
            else
            {
                float total = 0;
                [loop] for (int row = -radius; row <= radius; ++row)
                    [loop] for (int column = -radius; column <= radius; ++column)
                        total += FilteredDepth(image, uv + float2(column, row) / extent, extent, receiver);
                visibility = total / Filter.y;
            }
        }
    }
    return visibility;
}
float CascadeAt(uint cascade, float3 world)
{
    float4 clip = mul(CascadeCameras[cascade], float4(world, 1));
    float visibility = 1;
    if (cascade == 0) visibility = ShadowAt(CascadeDepth0, clip, 1);
    else if (cascade == 1) visibility = ShadowAt(CascadeDepth1, clip, 2);
    else if (cascade == 2) visibility = ShadowAt(CascadeDepth2, clip, 3);
    else visibility = ShadowAt(CascadeDepth3, clip, 4);
    return visibility;
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
    float visibility = 1;
    if (Filter.z != 0 && Dispatch.y != 0) visibility = VirtualVisibility(world, LightScaleBias.y);
    else if (Dispatch.y == ShadowSingle)
        visibility = ShadowAt(ShadowDepth, mul(ShadowCamera, float4(world, 1)), 0);
    else if (Dispatch.y == ShadowCascaded)
    {
        float depth = dot(CameraDepthPlane, float4(world, 1));
        if (depth >= CascadeRangeCount.x && depth <= CascadeRangeCount.y)
        {
            uint cascade = 0, count = uint(CascadeRangeCount.z);
            [loop] while (cascade + 1 < count && depth > CascadeEnds[cascade]) ++cascade;
            visibility = CascadeAt(cascade, world);
            if (cascade + 1 < count && depth > CascadeBlendBegin[cascade])
            {
                float overlap = max(CascadeEnds[cascade] - CascadeBlendBegin[cascade], RayEpsilon);
                float blend = saturate((depth - CascadeBlendBegin[cascade]) / overlap);
                visibility = lerp(visibility, CascadeAt(cascade + 1, world), blend);
            }
        }
    }
    return visibility;
}
// Finite slabs bound the density field on both visibility and incident rays.
float2 ClipMedium(float3 start, float3 direction, float maximumLength)
{
    float2 range = float2(0, maximumLength);
    bool intersects = true;
    [unroll] for (uint axis = 0; axis < 3; ++axis)
    {
        if (abs(direction[axis]) <= RayEpsilon)
        {
            intersects = intersects && start[axis] >= MediumLow[axis] && start[axis] <= MediumHigh[axis];
        }
        else
        {
            float2 intersections = (float2(MediumLow[axis], MediumHigh[axis]) - start[axis]) / direction[axis];
            range.x = max(range.x, min(intersections.x, intersections.y));
            range.y = min(range.y, max(intersections.x, intersections.y));
        }
    }
    return intersects && range.y > range.x ? range : float2(0, 0);
}
float ReadVolumeCell(int3 coordinate)
{
    int3 dimensions = int3(DensityLayout.xyz);
    if (any(coordinate < 0) || any(coordinate >= dimensions)) return 0;
    int tileColumns = int(DensityLayout.w);
    int2 address = int2(coordinate.z % tileColumns, coordinate.z / tileColumns) * dimensions.xy + coordinate.xy;
    const float NormalizedByteRange = 255.0;
    const uint BitsPerByte = 8u;
    uint4 encoded = uint4(round(DensityImage.Load(int3(address, 0)) * NormalizedByteRange));
    uint word = encoded.x | (encoded.y << BitsPerByte) | (encoded.z << (2u * BitsPerByte)) | (encoded.w << (3u * BitsPerByte));
    return asfloat(word);
}
float GridDensity(float3 position)
{
    // Linear RGBA8 stores float bytes; interpolation must follow decoding.
    float3 coordinate = (position - MediumLow.xyz) / (MediumHigh.xyz - MediumLow.xyz) * DensityLayout.xyz - 0.5;
    int3 lower = int3(floor(coordinate));
    float3 fraction = coordinate - float3(lower);
    float density = 0;
    [loop] for (int corner = 0; corner < 8; ++corner)
    {
        int3 offset = int3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        float3 weight = lerp(1 - fraction, fraction, float3(offset));
        density += ReadVolumeCell(lower + offset) * weight.x * weight.y * weight.z;
    }
    return density * AbsorptionScale.w;
}
float LocalDensity(float3 world)
{
    if (DensityLayout.x > 0) return GridDensity(world);
    float3 argument = (world - MediumLow.xyz) * EmissiveFrequency.w;
    float modulation = sin(argument.x) * sin(argument.y) * sin(argument.z);
    return AbsorptionScale.w * max(1 + ScatteringVariation.w * modulation, 0);
}
float3 LightTransmission(float3 world, float3 direction, float distance)
{
    float2 range = ClipMedium(world, direction, distance);
    float integral = AbsorptionScale.w * (range.y - range.x);
    if (DensityLayout.x > 0 || (ScatteringVariation.w > 0 && EmissiveFrequency.w > 0))
    {
        float step = (range.y - range.x) / MediumControl.x; integral = 0;
        [loop] for (uint index = 0; index < MediumControl.x; ++index)
            integral += step * LocalDensity(world + direction * (range.x + (index + 0.5) * step));
    }
    return exp(-min((AbsorptionScale.rgb + ScatteringVariation.rgb) * integral, OpticalDepthLimit));
}
float3 LightIntegral(SceneLight light, float3 world, float3 cameraRay)
{
    uint kind = uint(light.PositionKind.w);
    bool sun = kind == DirectionalKind, area = kind != DirectionalKind && kind != PointKind;
    float3 normal = UnitDirection(light.DirectionPower.xyz, float3(0, 1, 0));
    float3 axisX = 0, axisY = 0;
    const float MinimumEmitterGrid = 2, MaximumEmitterGrid = 16, StratumCenter = 0.5;
    uint axisSamples = 1;
    if (area)
    {
        float3 helper = abs(normal.y) < ParallelAxisThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        axisX = normalize(cross(helper, normal)); axisY = cross(normal, axisX);
        axisSamples = light.RectangleSize.w > 0 ? uint(clamp(light.RectangleSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : RectangleAxisSamples;
    }
    float3 integral = 0;
    // One transport call prevents FXC from expanding the nested density march
    // separately for directional, point and rectangle emitter branches.
    [loop] for (uint sampleId = 0; sampleId < axisSamples * axisSamples; ++sampleId)
    {
        float3 ray = normal;
        float rayLength = length(MediumHigh.xyz - MediumLow.xyz), factor = 1;
        if (!sun)
        {
            float2 local = ((float2(sampleId % axisSamples, sampleId / axisSamples) + StratumCenter) / axisSamples - StratumCenter) * light.RectangleSize.xy;
            float3 delta = light.PositionKind.xyz + axisX * local.x + axisY * local.y - world;
            ray = UnitDirection(delta, area ? -normal : normal); rayLength = length(delta);
            factor = LightRangeWindow(delta, light.RectangleSize.z) / max(dot(delta, delta), SquaredDistanceFloor);
            if (area) factor *= light.RectangleSize.x * light.RectangleSize.y * saturate(dot(normal, -ray)) / float(axisSamples * axisSamples);
        }
        integral += factor * AngularScattering(ray, cameraRay) * LightTransmission(world, ray, rayLength);
    }
    return integral;
}
float3 SourceRadiance(float3 world, float3 cameraRay)
{
    float3 direct = 0;
    if (LightScaleBias.x > 0)
    {
        float primaryVisibility = Dispatch.y == 0 ? 1 : PrimaryShadow(world);
        [loop] for (uint index = 0; index < Dispatch.x; ++index)
        {
            SceneLight light;
            if (LightScaleBias.z != 0) light = VolumeSceneLights[index]; else light = SceneLights[index];
            float3 power = min(light.ColorShadow.rgb, HalfFloatMaximum) * min(light.DirectionPower.w, HalfFloatMaximum);
            float3 rayToward=uint(light.PositionKind.w)==0u?light.DirectionPower.xyz:light.PositionKind.xyz-world;
            float visibility=Dispatch.y!=0 && index==Dispatch.z?(ShadowRayEnabled()?ShadowRayVisibility(world,float3(0,0,0),rayToward):primaryVisibility):1.0;
            direct += power * visibility * LightIntegral(light, world, cameraRay);
        }
    }
    return EmissiveFrequency.rgb + ScatteringVariation.rgb * (AmbientRadiance.rgb + LightScaleBias.x * direct);
}
float3 IntegratedLength(float3 sigma, float distance)
{
    const float SmallTau = 0.001, DenominatorFloor = 1.0e-20;
    const float QuadraticDenominator = 6.0;
    float3 tau = min(sigma * distance, OpticalDepthLimit);
    float3 weightedLength = (1 - exp(-tau)) / max(sigma, DenominatorFloor);
    [unroll] for (uint channel = 0; channel < 3; ++channel)
        if (tau[channel] < SmallTau)
            weightedLength[channel] = distance * (1 - tau[channel] / 2 + tau[channel] * tau[channel] / QuadraticDenominator);
    return weightedLength;
}
float4 PSInject(float4 pixel : SV_POSITION) : SV_TARGET
{
    uint2 coordinate = uint2(pixel.xy);
    uint columns = uint(ViewAndGrid.z), rows = uint(ViewAndGrid.w);
    uint slice = coordinate.x / columns;
    bool attenuationOutput = coordinate.y >= rows;
    float2 uv = (float2(coordinate.x % columns, coordinate.y % rows) + 0.5) / ViewAndGrid.zw;
    float3 origin = Unproject(uv, 0), ray = Unproject(uv, 1) - origin;
    float lengthToFar = max(length(ray), RayEpsilon), range = min(lengthToFar, Density.w);
    float3 direction = ray / lengthToFar;
    float2 bounds = ClipMedium(origin, direction, range);
    float sliceLength = range / Integration.y;
    bounds.x = max(bounds.x, sliceLength * slice);
    bounds.y = min(bounds.y, sliceLength * (slice + 1));
    float stepDistance = max(bounds.y - bounds.x, 0) / Integration.z;
    float3 scattered = 0, transmission = 1;
    if (stepDistance > 0)
        [loop] for (uint index = 0; index < Integration.z; ++index)
        {
            float3 world = origin + direction * (bounds.x + (index + 0.5) * stepDistance);
            float density = LocalDensity(world);
            float3 extinction = density * (AbsorptionScale.rgb + ScatteringVariation.rgb);
            if (!attenuationOutput) scattered += transmission * density * SourceRadiance(world, direction) * IntegratedLength(extinction, stepDistance);
            transmission *= exp(-min(extinction * stepDistance, OpticalDepthLimit));
        }
    return float4(attenuationOutput ? transmission : scattered, 1);
}
float4 PSPrefix(float4 pixel : SV_POSITION) : SV_TARGET
{
    uint2 coordinate = uint2(pixel.xy);
    uint columns = uint(ViewAndGrid.z), rows = uint(ViewAndGrid.w);
    uint slice = coordinate.x / columns;
    bool attenuationOutput = coordinate.y >= rows;
    float3 result = ScanInput.Load(int3(coordinate, 0)).rgb;
    if (slice >= Dispatch.w)
    {
        uint2 left = coordinate - uint2(Dispatch.w * columns, 0);
        float3 leftT = ScanInput.Load(int3(left.x, left.y % rows + rows, 0)).rgb;
        result *= leftT;
        if (!attenuationOutput) result += ScanInput.Load(int3(left, 0)).rgb;
    }
    if (Dispatch.w * 2 >= Integration.y) result = min(result, HalfFloatMaximum);
    return float4(result, 1);
}

#endif
