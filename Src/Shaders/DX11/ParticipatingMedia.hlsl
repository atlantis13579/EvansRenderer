// DX11 world-space shadow queries. Woop, Benthin and Wald, JCGT 2013.
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
struct ShadowRayNode { float4 Minimum,Maximum; uint4 Links; };
struct ShadowRayTriangle { float4 A,B,C,UVAB,UVC; uint4 Metadata; };
struct ShadowRayMaterial { uint4 DiffuseAlpha,OpacityRed; float4 Coverage; };
static const float ShadowRayPixelCenter=0.5;
static const float ShadowRayFloatEpsilon=1.1920928955078125e-7;
static const float ShadowRayEdgeError=(3*ShadowRayFloatEpsilon)/(1-3*ShadowRayFloatEpsilon);
ShadowRayNode ShadowRayReadNode(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).x+index*ShadowRayNodeSize;
    ShadowRayNode result;
    result.Minimum=asfloat(ShadowRayLoad(base));
    result.Maximum=asfloat(ShadowRayLoad(base+1u));
    result.Links=ShadowRayLoad(base+2u);
    return result;
}
ShadowRayTriangle ShadowRayReadTriangle(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).y+index*ShadowRayTriangleSize;
    ShadowRayTriangle result;
    result.A=asfloat(ShadowRayLoad(base));
    result.B=asfloat(ShadowRayLoad(base+1u));
    result.C=asfloat(ShadowRayLoad(base+2u));
    result.UVAB=asfloat(ShadowRayLoad(base+3u));
    result.UVC=asfloat(ShadowRayLoad(base+4u));
    result.Metadata=ShadowRayLoad(base+5u);
    return result;
}
ShadowRayMaterial ShadowRayReadMaterial(uint index)
{
    uint base=ShadowRayLoad(ShadowRayOffsetsHeader).z+index*ShadowRayMaterialSize;
    ShadowRayMaterial result;
    result.DiffuseAlpha=ShadowRayLoad(base);
    result.OpacityRed=ShadowRayLoad(base+1u);
    result.Coverage=asfloat(ShadowRayLoad(base+2u));
    return result;
}
float ShadowRayScalar(uint index)
{ uint4 value=ShadowRayLoad(ShadowRayLoad(ShadowRayOffsetsHeader).w+index/ShadowRayScalarLanes); return asfloat(value)[index%ShadowRayScalarLanes]; }
float ShadowRayCoverage(uint4 image,float2 uv)
{
    if (image.w==0) return 1;
    float2 texel=frac(float2(uv.x,-uv.y))*float2(image.yz)-ShadowRayPixelCenter;
    uint2 a=uint2(texel.x<0 ? image.y-1 : uint(floor(texel.x)),texel.y<0 ? image.z-1 : uint(floor(texel.y)));
    uint2 b=uint2(a.x+1==image.y ? 0 : a.x+1,a.y+1==image.z ? 0 : a.y+1);
    float2 weight=frac(texel);
    float top=lerp(ShadowRayScalar(image.x+a.y*image.y+a.x),ShadowRayScalar(image.x+a.y*image.y+b.x),weight.x);
    float bottom=lerp(ShadowRayScalar(image.x+b.y*image.y+a.x),ShadowRayScalar(image.x+b.y*image.y+b.x),weight.x);
    return lerp(top,bottom,weight.y);
}
bool ShadowRayBoxHit(ShadowRayNode node,float3 origin,float3 direction,float radius)
{
    float enter=0,leave=radius;
    [unroll] for (uint axis=0;axis<3;++axis)
    {
        if (direction[axis]==0)
        { if (origin[axis]<node.Minimum[axis] || origin[axis]>node.Maximum[axis]) return false; }
        else
        {
            float a=(node.Minimum[axis]-origin[axis])/direction[axis],b=(node.Maximum[axis]-origin[axis])/direction[axis];
            enter=max(enter,min(a,b)); leave=min(leave,max(a,b)); if (enter>leave) return false;
        }
    }
    return true;
}
float ShadowRayEdge(float2 a,float2 b)
{
    precise float forward=a.x*b.y,backward=a.y*b.x;
    precise float value=forward-backward;
    return abs(value)<=ShadowRayEdgeError*(abs(forward)+abs(backward)) ? 0 : value;
}
bool ShadowRayTriangleHit(ShadowRayTriangle primitive,float3 origin,float3 direction,float radius,out float3 weights)
{
    weights=0;
    uint z=abs(direction.x)>abs(direction.y) ? 0 : 1; if (abs(direction.z)>abs(direction[z])) z=2;
    uint x=(z+1)%3,y=(x+1)%3;
    float3 a=primitive.A.xyz-origin,b=primitive.B.xyz-origin,c=primitive.C.xyz-origin;
    float2 shear=float2(direction[x],direction[y])/direction[z];
    precise float2 pa=float2(a[x],a[y])-shear*a[z],pb=float2(b[x],b[y])-shear*b[z],pc=float2(c[x],c[y])-shear*c[z];
    float3 edges=float3(ShadowRayEdge(pb,pc),ShadowRayEdge(pc,pa),ShadowRayEdge(pa,pb));
    if (any(edges<0) && any(edges>0)) return false;
    float area=edges.x+edges.y+edges.z; if (area==0) return false;
    float distance=dot(edges,float3(a[z],b[z],c[z]))/(area*direction[z]);
    if (!(distance>0 && distance<=radius)) return false;
    weights=edges/area; return true;
}
bool ShadowRayOccluded(float3 origin,float3 direction,float radius)
{
    bool blocked=false;
    uint nodeIndex=0;
    [loop] while (nodeIndex<ShadowRayLoad(0u).y && !blocked)
    {
        ShadowRayNode node=ShadowRayReadNode(nodeIndex);
        if (!ShadowRayBoxHit(node,origin,direction,radius)) { nodeIndex=node.Links.w; continue; }
        if (node.Links.z==0) { nodeIndex=node.Links.x; continue; }
        [loop] for (uint index=0;index<node.Links.z && !blocked;++index)
        {
            ShadowRayTriangle primitive=ShadowRayReadTriangle(node.Links.y+index);
            if ((primitive.Metadata.w & ShadowRayCasterFlag)==0u) continue; float3 weights;
            if (!ShadowRayTriangleHit(primitive,origin,direction,radius,weights)) continue;
            ShadowRayMaterial material=ShadowRayReadMaterial(primitive.Metadata.x); if (material.Coverage.y<=0) { blocked=true; break; }
            float2 uv=primitive.UVAB.xy*weights.x+primitive.UVAB.zw*weights.y+primitive.UVC.xy*weights.z;
            float alpha=saturate(dot(weights,float3(primitive.A.w,primitive.B.w,primitive.C.w))*material.Coverage.x*
                ShadowRayCoverage(material.DiffuseAlpha,uv)*ShadowRayCoverage(material.OpacityRed,uv));
            if (alpha>=material.Coverage.y) { blocked=true; break; }
        }
        nodeIndex=node.Links.w;
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
    if (ShadowRayOccluded(origin,direction,limit)) visibility=0.0;
    return visibility;
}

// Independent DX11 participating-medium transport and opaque composition.
// Pharr, Jakob and Humphreys, PBRT v4, Volume Scattering and Equation of Transfer.
// https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance
// EVANS_MEDIA_COMPOSITE selects the resolved-scene consumer of paired S.rgb/T.rgb planes.
#if defined(EVANS_MEDIA_COMPOSITE)
cbuffer FogConstants : register(b3)
{
    row_major float4x4 FogInverseViewProjection;
    float4 FogMedium;
    float4 FogSourceRadiance;
    float4 FogViewportGrid;
    uint4 FogOptions;
};
Texture2D<float4> SceneColor : register(t0);
Texture2D<float> SceneDepth : register(t1);
Texture2D<float4> FogAtlas : register(t14);
SamplerState FogLinearClamp : register(s3);

static const float MinimumRayLength = 1.0e-6f;
static const float MaximumHalfRadiance = 65504.0f;
static const float4 IdentityTransport = float4(0, 0, 0, 1);

float4 VSMain(uint index : SV_VertexID) : SV_POSITION
{
    float2 uv = float2((index << 1) & 2, index & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float3 WorldPosition(float2 uv, float depth)
{
    float4 world = mul(FogInverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    return world.xyz / world.w;
}
float4 EndpointTransportTransmission(float2 uv, uint endpoint)
{
    if (endpoint == 0) return float4(1, 1, 1, 1);
    float2 halfCell = 0.5f / FogViewportGrid.zw;
    float2 cellUV = clamp(uv, halfCell, 1 - halfCell);
    cellUV.x = (cellUV.x + endpoint - 1) / FogOptions.y;
    cellUV.y = (cellUV.y + 1.0) * 0.5;
    return FogAtlas.SampleLevel(FogLinearClamp, cellUV, 0);
}
float4 EndpointTransport(float2 uv, uint endpoint)
{
    if (endpoint == 0) return IdentityTransport;
    float2 halfCell = 0.5f / FogViewportGrid.zw;
    float2 cellUV = clamp(uv, halfCell, 1 - halfCell);
    cellUV.x = (cellUV.x + endpoint - 1) / FogOptions.y;
    cellUV.y *= 0.5;
    return FogAtlas.SampleLevel(FogLinearClamp, cellUV, 0);
}
float4 IntegratedTransmission(float2 uv, float depth)
{
    float3 start = WorldPosition(uv, 0);
    float rayExtent = min(length(WorldPosition(uv, 1) - start), FogMedium.w);
    float endpoint = saturate(length(WorldPosition(uv, depth) - start) / max(rayExtent, MinimumRayLength)) * FogOptions.y;
    uint lower = (uint)floor(endpoint);
    uint upper = min(lower + 1, FogOptions.y);
    return lerp(EndpointTransportTransmission(uv, lower), EndpointTransportTransmission(uv, upper), frac(endpoint));
}
float4 IntegratedTransport(float2 uv, float depth)
{
    float3 start = WorldPosition(uv, 0);
    float rayExtent = min(length(WorldPosition(uv, 1) - start), FogMedium.w);
    float endpoint = saturate(length(WorldPosition(uv, depth) - start) / max(rayExtent, MinimumRayLength)) * FogOptions.y;
    uint lower = (uint)floor(endpoint);
    uint upper = min(lower + 1, FogOptions.y);
    return lerp(EndpointTransport(uv, lower), EndpointTransport(uv, upper), frac(endpoint));
}

float4 PSMain(float4 position : SV_POSITION) : SV_TARGET
{
    int2 pixel = int2(position.xy);
    float4 color = SceneColor.Load(int3(pixel, 0));
    float4 transport = IntegratedTransport(position.xy / FogViewportGrid.xy, SceneDepth.Load(int3(pixel, 0)));
    float3 transmittance = IntegratedTransmission(position.xy / FogViewportGrid.xy, SceneDepth.Load(int3(pixel, 0))).rgb;
    return float4(clamp(color.rgb * transmittance + transport.rgb, 0, MaximumHalfRadiance), color.a);
}

#else
// Smooth finite support for inverse-square point and rectangle samples.
// Karis' attenuation window; Filament, Punctual lights / Attenuation function.
float LightRangeWindow(float3 toEmitter, float radius)
{
    if (radius <= 0.0f) return 1.0f;
    if (any(abs(toEmitter) >= radius)) return 0.0f;
    float3 relative = toEmitter / radius;
    float distanceRatioSquared = dot(relative, relative);
    float window = saturate(1.0f - distanceRatioSquared * distanceRatioSquared);
    return window * window;
}

// RGB single scattering in a bounded homogeneous or smoothly varying medium, native DX11.
// Henyey-Greenstein: https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
// Each invocation injects one depth segment. An inclusive transport scan then
// composes (S,T) operators; light evaluation is not repeated for every endpoint.
// Primary shadow-map/CSM visibility and incident-light extinction are included.
// Multiple scattering is omitted; rectangle emitters retain finite quadrature.
cbuffer MediumConstants : register(b3)
{
    row_major float4x4 InverseViewProjection;
    float4 Medium;
    float4 AmbientSource;
    float4 ViewportGrid;
    uint4 GridOptions;
};
struct VolumeLight
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSize;
};
// Full scene lights include emitters affecting points before the opaque surface.
StructuredBuffer<VolumeLight> VolumeSceneLights : register(t16);
static const uint MaximumLights = 4;
static const uint MaximumCascades = 4;
cbuffer LightingConstants : register(b4)
{
    float4 Scattering;
    float4 LightParameters;
    uint4 Control;
    row_major float4x4 SingleLightViewProjection;
    VolumeLight Lights[MaximumLights];
    row_major float4x4 CascadeLightViewProjection[MaximumCascades];
    float4 CascadeFarDistances;
    float4 CascadeBlendStarts;
    float4 EyeDepthPlane;
    float4 CascadeParameters;
    float4 ShadowFiltering;
    float4 ShadowEncoding[MaximumCascades + 1];
    float4 ShadowParameters;
    uint4 ShadowOptions;
    float4 AbsorptionDensity;
    float4 ScatteringModulation;
    float4 EmissionFrequency;
    float4 BoundsMinimum;
    float4 BoundsMaximum;
    uint4 MediumOptions;
    float4 DensityGrid;
};
Texture2D<float4> PreviousTransport : register(t0);
Texture2D<float4> SingleShadow : register(t1);
Texture2D<float4> Cascade0 : register(t2);
Texture2D<float4> Cascade1 : register(t3);
Texture2D<float4> Cascade2 : register(t4);
Texture2D<float4> Cascade3 : register(t5);
Texture2D<float4> DensityAtlas : register(t6);

static const float Pi = 3.14159265358979323846;
static const float MaximumOpticalDepth = 80.0;
static const float MinimumRayLength = 1.0e-6;
static const float MinimumSquaredLightDistance = 1.0e-4;
static const float ProjectionEpsilon = 1.0e-7;
static const float MaximumHalfRadiance = 65504.0;
static const uint DirectionalLight = 0, PointLight = 1;
static const uint SingleShadowMode = 1, CascadedShadowMode = 2;
static const uint AreaSamplesPerAxis = 2;
static const float AreaSampleWeight = 0.25;
static const float BasisParallelThreshold = 0.99;
static const float4 IdentityTransport = float4(0, 0, 0, 1);

float4 VSMain(uint vertexIndex : SV_VertexID) : SV_POSITION
{
    const float FullscreenCorner = 3.0;
    return float4(vertexIndex == 1 ? FullscreenCorner : -1.0,
        vertexIndex == 2 ? FullscreenCorner : -1.0, 0, 1);
}
float3 SafeDirection(float3 direction, float3 fallback)
{
    float squaredLength = dot(direction, direction);
    return squaredLength > MinimumSquaredLightDistance ? direction * rsqrt(squaredLength) : fallback;
}
float3 WorldPosition(float2 uv, float depth)
{
    float4 world = mul(InverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    return world.xyz / world.w;
}
float Phase(float3 toLight, float3 cameraRay)
{
    // Incoming travel is -toLight; outgoing travel to the eye is -cameraRay.
    // This convention uses a minus sign in HG, unlike PBRT's two-away vectors.
    float cosine = clamp(dot(toLight, cameraRay), -1.0, 1.0);
    float g = Scattering.w;
    float denominator = 1 + g * g - 2 * g * cosine;
    return (1 - g * g) / (4 * Pi * denominator * sqrt(denominator));
}
float DepthCompare(Texture2D<float4> map, int2 pixel, uint2 size, float receiver)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return 1;
    return receiver <= map.Load(int3(pixel, 0)).r ? 1.0 : 0.0;
}
float BilinearComparison(Texture2D<float4> map, float2 uv, uint2 size, float receiver)
{
    float2 texel = uv * size - 0.5;
    int2 corner = int2(floor(texel));
    float2 fraction = frac(texel);
    return lerp(lerp(DepthCompare(map, corner, size, receiver), DepthCompare(map, corner + int2(1, 0), size, receiver), fraction.x),
        lerp(DepthCompare(map, corner + int2(0, 1), size, receiver), DepthCompare(map, corner + 1, size, receiver), fraction.x), fraction.y);
}
static const float ShadowPixelCenter = 0.5;
static const float MaximumMapFraction = 0.25;
static const uint SoftFilterMode = 2, MomentFilterMode = 3;
uint2 MapSize(uint map)
{
    uint width = 1, height = 1;
    if (map == 0) SingleShadow.GetDimensions(width, height);
    else Cascade0.GetDimensions(width, height);
    return uint2(width, height);
}
float DepthAt(uint map, int2 pixel, uint2 size)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return 1;
    // SM5 resource arrays require literal indices. Map zero is the single map.
    if (map == 0) return SingleShadow.Load(int3(pixel, 0)).r;
    if (map == 1) return Cascade0.Load(int3(pixel, 0)).r;
    if (map == 2) return Cascade1.Load(int3(pixel, 0)).r;
    if (map == 3) return Cascade2.Load(int3(pixel, 0)).r;
    return Cascade3.Load(int3(pixel, 0)).r;
}
float LinearDistance(float depth, float4 metric)
{ return metric.x > 0 ? rcp((1 - depth) / metric.x + depth / metric.y) : depth * metric.y; }
float2 DiskPoint(uint index, uint count)
{
    const float GoldenAngle = Pi * (3 - sqrt(5.0));
    float angle = index * GoldenAngle;
    return sqrt((index + ShadowPixelCenter) / count) * float2(cos(angle), sin(angle));
}
float2 BoundedRadius(float worldRadius, float4 metric, uint2 size, float budget)
{
    float2 radius = worldRadius / metric.zw;
    float extent = max(radius.x * size.x, radius.y * size.y);
    float limit = min(budget, min(size.x, size.y) * MaximumMapFraction);
    return extent > limit ? radius * (limit / extent) : radius;
}
float FilterComparison(uint map, float2 uv, uint2 size, float receiver)
{
    float2 texel = uv * size - ShadowPixelCenter;
    int2 base = int2(floor(texel)); float2 weight = frac(texel);
    float a = DepthAt(map, base, size) >= receiver;
    float b = DepthAt(map, base + int2(1, 0), size) >= receiver;
    float c = DepthAt(map, base + int2(0, 1), size) >= receiver;
    float d = DepthAt(map, base + int2(1, 1), size) >= receiver;
    return lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
}
float SoftMapVisibility(uint map, float4 clip, float bias)
{
    if ((ShadowOptions.z & (1u << map)) == 0 || clip.w <= ProjectionEpsilon) return 1;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1) return 1;
    uint2 size = MapSize(map);
    float4 metric = ShadowEncoding[map];
    float comparison = projected.z - bias;
    float emitter = metric.x > 0 ? ShadowParameters.x : ShadowParameters.y;
    if (emitter == 0) return DepthAt(map, int2(floor(uv * size)), size) >= comparison;
    float receiver = LinearDistance(projected.z, metric);
    float searchWorld = metric.x > 0 ? emitter * (receiver - metric.x) / receiver : emitter * receiver;
    float2 search = BoundedRadius(searchWorld, metric, size, ShadowParameters.z);
    float blockerSum = 0; uint blockers = 0;
    [loop] for (uint sampleIndex = 0; sampleIndex < ShadowOptions.x; ++sampleIndex)
    {
        float2 sampleUV = uv + DiskPoint(sampleIndex, ShadowOptions.x) * search;
        float depth = DepthAt(map, int2(floor(sampleUV * size)), size);
        if (depth < comparison) { blockerSum += LinearDistance(depth, metric); ++blockers; }
    }
    if (blockers == 0) return 1;
    float meanBlocker = blockerSum / blockers;
    float gap = max(receiver - meanBlocker, 0);
    float penumbra = metric.x > 0 ? emitter * gap / meanBlocker * metric.x / receiver : emitter * gap;
    float2 radius = BoundedRadius(penumbra, metric, size, ShadowParameters.w);
    float visibility = 0;
    [loop] for (uint filterIndex = 0; filterIndex < ShadowOptions.y; ++filterIndex)
        visibility += FilterComparison(map, uv + DiskPoint(filterIndex, ShadowOptions.y) * radius, size, comparison);
    return visibility / ShadowOptions.y;
}
float2 MomentsAt(uint map, int2 pixel, uint2 size)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return float2(1, 1);
    // Literal resource indices preserve the SM5 texture-array contract.
    if (map == 0) return SingleShadow.Load(int3(pixel, 0)).rg;
    if (map == 1) return Cascade0.Load(int3(pixel, 0)).rg;
    if (map == 2) return Cascade1.Load(int3(pixel, 0)).rg;
    if (map == 3) return Cascade2.Load(int3(pixel, 0)).rg;
    return Cascade3.Load(int3(pixel, 0)).rg;
}
float2 BilinearMoments(uint map, float2 uv)
{
    uint2 size = MapSize(map);
    float2 texel = uv * size - ShadowPixelCenter;
    int2 base = int2(floor(texel)); float2 weight = frac(texel);
    float2 top = lerp(MomentsAt(map, base, size), MomentsAt(map, base + int2(1, 0), size), weight.x);
    float2 bottom = lerp(MomentsAt(map, base + int2(0, 1), size), MomentsAt(map, base + int2(1, 1), size), weight.x);
    return lerp(top, bottom, weight.y);
}
float MomentMapVisibility(uint map, float4 clip, float bias)
{
    if ((ShadowOptions.x & (1u << map)) == 0 || clip.w <= ProjectionEpsilon) return 1;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1) return 1;
    float linearDepth = dot(float3(clip.zw, 1), ShadowEncoding[map].xyz);
    float2 moments = BilinearMoments(map, uv);
    float distanceBeyondMean = linearDepth - bias - moments.x;
    if (distanceBeyondMean <= 0) return 1;
    float variance = max(moments.y - moments.x * moments.x, ShadowParameters.x);
    float upperBound = variance / (variance + distanceBeyondMean * distanceBeyondMean);
    // This artistic remap suppresses bleeding but forfeits the upper-bound guarantee.
    return saturate((upperBound - ShadowParameters.y) / (1 - ShadowParameters.y));
}

float MapVisibility(Texture2D<float4> map, float4 clip, uint mapIndex)
{
    // An explicit result keeps assignment visible to FXC after nested loops are inlined.
    float visibility = 1;
    if (MediumOptions.y == SoftFilterMode)
        visibility = SoftMapVisibility(mapIndex, clip, LightParameters.y);
    else if (MediumOptions.y == MomentFilterMode)
        visibility = MomentMapVisibility(mapIndex, clip, LightParameters.y);
    else if (clip.w > ProjectionEpsilon)
    {
        float3 projected = clip.xyz / clip.w;
        float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
        if (all(uv >= 0) && all(uv <= 1) && projected.z >= 0 && projected.z <= 1)
        {
            uint width = 0, height = 0; map.GetDimensions(width, height);
            uint2 size = uint2(width, height);
            float receiver = projected.z - LightParameters.y;
            int radius = int(ShadowFiltering.x);
            if (radius == 0)
                visibility = DepthCompare(map, min(int2(uv * size), int2(size) - 1), size, receiver);
            else
            {
                float sum = 0;
                [loop] for (int y = -radius; y <= radius; ++y)
                    [loop] for (int x = -radius; x <= radius; ++x)
                        sum += BilinearComparison(map, uv + float2(x, y) / size, size, receiver);
                visibility = sum / ShadowFiltering.y;
            }
        }
    }
    return visibility;
}
float CascadeVisibility(uint index, float3 position)
{
    float4 clip = mul(CascadeLightViewProjection[index], float4(position, 1));
    float visibility = 1;
    if (index == 0) visibility = MapVisibility(Cascade0, clip, 1);
    else if (index == 1) visibility = MapVisibility(Cascade1, clip, 2);
    else if (index == 2) visibility = MapVisibility(Cascade2, clip, 3);
    else visibility = MapVisibility(Cascade3, clip, 4);
    return visibility;
}
// Virtual depth and page directory share the exclusive cascade slots.
// Each PCF corner resolves its own page or pinned coarse ancestor.
static const uint VirtualPageBorder = 1;
static const float VirtualTexelCenter = 0.5f;
float2 VirtualDepth(float2 uv, uint requestedMip)
{
    uint4 logical = uint4(CascadeFarDistances), physical = uint4(CascadeBlendStarts);
    uint side = logical.x / logical.y, base = 0;
    [loop] for (uint mip = 0; mip < logical.z; ++mip)
    {
        if (mip >= requestedMip)
        {
            uint2 pixel = min(uint2(uv * (side * logical.y)), side * logical.y - 1);
            uint2 page = pixel / logical.y;
            uint address = base + page.y * side + page.x;
            uint entry = uint(round(Cascade1.Load(int3(address % logical.w, address / logical.w, 0)).x * physical.w));
            if (entry != 0)
            {
                uint slot = entry - 1, stride = logical.y + VirtualPageBorder * 2;
                uint2 tile = uint2(slot % physical.z, slot / physical.z) * stride;
                return float2(Cascade0.Load(int3(tile + VirtualPageBorder + pixel % logical.y, 0)).x, mip);
            }
        }
        base += side * side; side /= 2;
    }
    return float2(1, logical.z - 1);
}
float VirtualCompare(int2 pixel, uint mip, float receiver)
{
    uint resolution = uint(CascadeFarDistances.x) >> mip;
    float2 uv = (float2(clamp(pixel, 0, int(resolution) - 1)) + VirtualTexelCenter) / resolution;
    return receiver <= VirtualDepth(uv, mip).x ? 1 : 0;
}
// A volume cell uses neighboring camera rays at equal camera depth. Unlike
// screen derivatives, this construction is defined inside integration loops.
uint VirtualVolumeMip(float3 world, float2 centerUV)
{
    float4 p = float4(world, 1);
    float4 camera = float4(dot(ShadowEncoding[0], p), dot(ShadowEncoding[1], p), dot(ShadowEncoding[2], p), dot(ShadowEncoding[3], p));
    if (camera.w <= EyeDepthPlane.w) return 0;
    float3 ndc = camera.xyz / camera.w;
    float4 x = mul(InverseViewProjection, float4(ndc + float3(2 * EyeDepthPlane.y, 0, 0), 1));
    float4 y = mul(InverseViewProjection, float4(ndc - float3(0, 2 * EyeDepthPlane.z, 0), 1));
    float4 lightX = mul(CascadeLightViewProjection[0], float4(x.xyz / x.w, 1));
    float4 lightY = mul(CascadeLightViewProjection[0], float4(y.xyz / y.w, 1));
    if (lightX.w <= EyeDepthPlane.w || lightY.w <= EyeDepthPlane.w) return uint(CascadeFarDistances.z - 1);
    float2 uvX = lightX.xy / lightX.w * float2(VirtualTexelCenter, -VirtualTexelCenter) + VirtualTexelCenter;
    float2 uvY = lightY.xy / lightY.w * float2(VirtualTexelCenter, -VirtualTexelCenter) + VirtualTexelCenter;
    float footprint = max(length(uvX - centerUV), length(uvY - centerUV)) * CascadeFarDistances.x;
    return uint(clamp(floor(log2(max(1, footprint)) + EyeDepthPlane.x), 0, CascadeFarDistances.z - 1));
}

float VirtualVisibility(float3 world, float bias)
{
    float4 light = mul(CascadeLightViewProjection[0], float4(world, 1));
    float3 ndc = light.xyz / max(light.w, EyeDepthPlane.w);
    float2 uv = float2(ndc.x, -ndc.y) * VirtualTexelCenter + VirtualTexelCenter;
    uint mip = VirtualVolumeMip(world, uv);
    if (light.w <= EyeDepthPlane.w || any(uv < 0) || any(uv > 1) || ndc.z < 0 || ndc.z > 1) return 1;
    float2 center = VirtualDepth(uv, mip);
    float receiver = ndc.z - bias;
    if (ShadowFiltering.x == 0) return receiver <= center.x ? 1 : 0;
    mip = uint(center.y);
    float2 texel = uv * (uint(CascadeFarDistances.x) >> mip) - VirtualTexelCenter;
    int2 origin = int2(floor(texel)); float2 weight = frac(texel); float visible = 0;
    [loop] for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        [loop] for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = origin + int2(x, y);
            float a = VirtualCompare(p, mip, receiver), b = VirtualCompare(p + int2(1, 0), mip, receiver);
            float c = VirtualCompare(p + int2(0, 1), mip, receiver), d = VirtualCompare(p + 1, mip, receiver);
            visible += lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        }
    return visible / ShadowFiltering.y;
}


float PrimaryVisibility(float3 position)
{
    if (ShadowRayEnabled()) return 1.0;
    float visibility = 1;
    if (ShadowFiltering.z != 0 && Control.y != 0) visibility = VirtualVisibility(position, LightParameters.y);
    else if (Control.y == SingleShadowMode)
        visibility = MapVisibility(SingleShadow, mul(SingleLightViewProjection, float4(position, 1)), 0);
    else if (Control.y == CascadedShadowMode)
    {
        float eyeDepth = dot(EyeDepthPlane, float4(position, 1));
        if (eyeDepth >= CascadeParameters.x && eyeDepth <= CascadeParameters.y)
        {
            uint count = uint(CascadeParameters.z), index = 0;
            while (index + 1 < count && eyeDepth > CascadeFarDistances[index]) ++index;
            visibility = CascadeVisibility(index, position);
            if (index + 1 < count && eyeDepth > CascadeBlendStarts[index])
            {
                float span = max(CascadeFarDistances[index] - CascadeBlendStarts[index], MinimumRayLength);
                visibility = lerp(visibility, CascadeVisibility(index + 1, position), saturate((eyeDepth - CascadeBlendStarts[index]) / span));
            }
        }
    }
    return visibility;
}
// Slab clipping prevents thin volumes from falling between integration samples.
float2 MediumInterval(float3 origin, float3 direction, float limit)
{
    float enter = 0, leave = limit;
    [unroll] for (uint axis = 0; axis < 3; ++axis)
    {
        if (abs(direction[axis]) <= MinimumRayLength)
        {
            if (origin[axis] < BoundsMinimum[axis] || origin[axis] > BoundsMaximum[axis]) return float2(0, 0);
        }
        else
        {
            float a = (BoundsMinimum[axis] - origin[axis]) / direction[axis];
            float b = (BoundsMaximum[axis] - origin[axis]) / direction[axis];
            enter = max(enter, min(a, b)); leave = min(leave, max(a, b));
        }
    }
    return float2(enter, max(enter, leave));
}
// Decode point-fetched IEEE 754 bytes before trilinear density filtering.
float DensityVoxel(int3 cell)
{
    int3 size = int3(DensityGrid.xyz);
    if (any(cell < 0) || any(cell >= size)) return 0;
    int columns = int(DensityGrid.w);
    int2 texel = int2(cell.z % columns, cell.z / columns) * size.xy + cell.xy;
    const float ByteMaximum = 255.0;
    const uint ByteBits = 8u;
    uint4 bytes = uint4(round(DensityAtlas.Load(int3(texel, 0)) * ByteMaximum));
    return asfloat(bytes.x | (bytes.y << ByteBits) | (bytes.z << (2u * ByteBits)) | (bytes.w << (3u * ByteBits)));
}
float SampleDensityGrid(float3 world)
{
    float3 samplePoint = (world - BoundsMinimum.xyz) / (BoundsMaximum.xyz - BoundsMinimum.xyz) * DensityGrid.xyz - 0.5;
    int3 base = int3(floor(samplePoint));
    float3 blend = samplePoint - float3(base);
    float result = 0;
    [loop] for (int corner = 0; corner < 8; ++corner)
    {
        int x = corner & 1, y = (corner >> 1) & 1, z = (corner >> 2) & 1;
        float3 weight = float3(x ? blend.x : 1 - blend.x, y ? blend.y : 1 - blend.y, z ? blend.z : 1 - blend.z);
        result += DensityVoxel(base + int3(x, y, z)) * weight.x * weight.y * weight.z;
    }
    return AbsorptionDensity.w * result;
}
float MediumDensity(float3 position)
{
    if (DensityGrid.x > 0) return SampleDensityGrid(position);
    float3 phase = (position - BoundsMinimum.xyz) * EmissionFrequency.w;
    float variation = sin(phase.x) * sin(phase.y) * sin(phase.z);
    return AbsorptionDensity.w * max(0, 1 + ScatteringModulation.w * variation);
}
float3 IncidentTransmittance(float3 position, float3 direction, float distance)
{
    float2 interval = MediumInterval(position, direction, distance);
    float span = interval.y - interval.x;
    float integratedDensity = AbsorptionDensity.w * span;
    if (DensityGrid.x > 0 || (ScatteringModulation.w > 0 && EmissionFrequency.w > 0))
    {
        float stepLength = span / MediumOptions.x; integratedDensity = 0;
        [loop] for (uint index = 0; index < MediumOptions.x; ++index)
            integratedDensity += MediumDensity(position + direction * (interval.x + (index + 0.5) * stepLength)) * stepLength;
    }
    return exp(-min((AbsorptionDensity.rgb + ScatteringModulation.rgb) * integratedDensity, MaximumOpticalDepth));
}
float3 DirectSource(float3 position, float3 cameraRay)
{
    if (LightParameters.x == 0) return 0;
    float3 source = 0;
    float visibility = Control.y != 0 ? PrimaryVisibility(position) : 1;
    float directionalExtent = length(BoundsMaximum.xyz - BoundsMinimum.xyz);
    [loop] for (uint index = 0; index < Control.x; ++index)
    {
        VolumeLight light;
        if (LightParameters.z != 0) light = VolumeSceneLights[index]; else light = Lights[index];
        uint type = uint(light.PositionType.w);
        bool directional = type == DirectionalLight, rectangle = type != DirectionalLight && type != PointLight;
        float3 power = min(light.ColorCastShadows.rgb, MaximumHalfRadiance) * min(light.DirectionIntensity.w, MaximumHalfRadiance);
        float3 rayToward = directional ? light.DirectionIntensity.xyz : light.PositionType.xyz - position;
        float shadow = index == Control.z && Control.y != 0 ?
            (ShadowRayEnabled() ? ShadowRayVisibility(position, float3(0, 0, 0), rayToward) : visibility) : 1;
        float3 direction = SafeDirection(light.DirectionIntensity.xyz, float3(0, 1, 0));
        float3 tangent = 0, bitangent = 0;
        const float MinimumEmitterGrid = 2, MaximumEmitterGrid = 16, EmitterStratumCenter = 0.5;
        uint grid = 1;
        if (rectangle)
        {
            float3 helper = abs(direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
            tangent = normalize(cross(helper, direction)); bitangent = cross(direction, tangent);
            grid = light.AreaSize.w > 0 ? uint(clamp(light.AreaSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AreaSamplesPerAxis;
        }
        float3 contribution = 0;
        // Share one density march across emitter types. Duplicating this call
        // in each branch multiplies the inlined density/noise and BVH program.
        [loop] for (uint sampleIndex = 0; sampleIndex < grid * grid; ++sampleIndex)
        {
            float3 towardLight = direction;
            float distance = directionalExtent, weight = 1;
            if (!directional)
            {
                float2 offset = (float2(sampleIndex % grid, sampleIndex / grid) + EmitterStratumCenter) / grid - EmitterStratumCenter;
                float3 emitter = light.PositionType.xyz + tangent * offset.x * light.AreaSize.x + bitangent * offset.y * light.AreaSize.y;
                float3 delta = emitter - position;
                towardLight = SafeDirection(delta, rectangle ? -direction : direction);
                distance = length(delta);
                weight = LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), MinimumSquaredLightDistance);
                if (rectangle) weight *= light.AreaSize.x * light.AreaSize.y * saturate(dot(direction, -towardLight)) / float(grid * grid);
            }
            contribution += weight * Phase(towardLight, cameraRay) * IncidentTransmittance(position, towardLight, distance);
        }
        source += power * shadow * contribution;
    }
    return LightParameters.x * source;
}
float3 SegmentIntegral(float3 extinction, float distance)
{
    const float SmallOpticalDepth = 0.001, DivisionFloor = 1.0e-20;
    const float QuadraticDenominator = 6.0;
    float3 opticalDepth = min(extinction * distance, MaximumOpticalDepth);
    float3 integral = (1 - exp(-opticalDepth)) / max(extinction, DivisionFloor);
    [unroll] for (uint channel = 0; channel < 3; ++channel)
        if (opticalDepth[channel] < SmallOpticalDepth)
            integral[channel] = distance * (1 - opticalDepth[channel] * 0.5 + opticalDepth[channel] * opticalDepth[channel] / QuadraticDenominator);
    return integral;
}
float4 PSInject(float4 position : SV_POSITION) : SV_TARGET
{
    uint2 pixel = uint2(position.xy);
    uint gridWidth = uint(ViewportGrid.z), gridHeight = uint(ViewportGrid.w);
    uint slice = pixel.x / gridWidth;
    bool transmissionPlane = pixel.y >= gridHeight;
    float2 uv = (float2(pixel.x % gridWidth, pixel.y % gridHeight) + 0.5) / ViewportGrid.zw;
    float3 origin = WorldPosition(uv, 0), ray = WorldPosition(uv, 1) - origin;
    float rayLength = max(length(ray), MinimumRayLength), extent = min(rayLength, Medium.w);
    float3 direction = ray / rayLength;
    float2 interval = MediumInterval(origin, direction, extent);
    float sliceLength = extent / GridOptions.y;
    interval = float2(max(interval.x, slice * sliceLength), min(interval.y, (slice + 1) * sliceLength));
    float stepLength = max(interval.y - interval.x, 0) / GridOptions.z;
    float3 radiance = 0, transmittance = 1;
    if (stepLength > 0)
        [loop] for (uint index = 0; index < GridOptions.z; ++index)
        {
            float3 world = origin + direction * (interval.x + (index + 0.5) * stepLength);
            float density = MediumDensity(world);
            float3 sigmaT = density * (AbsorptionDensity.rgb + ScatteringModulation.rgb);
            if (!transmissionPlane)
            {
                float3 source = density * (EmissionFrequency.rgb + ScatteringModulation.rgb * (AmbientSource.rgb + DirectSource(world, direction)));
                radiance += transmittance * source * SegmentIntegral(sigmaT, stepLength);
            }
            transmittance *= exp(-min(sigmaT * stepLength, MaximumOpticalDepth));
        }
    return float4(transmissionPlane ? transmittance : radiance, 1);
}
float4 PSPrefix(float4 position : SV_POSITION) : SV_TARGET
{
    uint2 pixel = uint2(position.xy);
    uint gridWidth = uint(ViewportGrid.z), gridHeight = uint(ViewportGrid.w);
    uint slice = pixel.x / gridWidth;
    bool transmissionPlane = pixel.y >= gridHeight;
    float3 result = PreviousTransport.Load(int3(pixel, 0)).rgb;
    if (slice >= Control.w)
    {
        uint2 left = pixel - uint2(Control.w * gridWidth, 0);
        uint2 leftTransmission = uint2(left.x, left.y % gridHeight + gridHeight);
        float3 transmission = PreviousTransport.Load(int3(leftTransmission, 0)).rgb;
        result *= transmission;
        if (!transmissionPlane) result += PreviousTransport.Load(int3(left, 0)).rgb;
    }
    if (Control.w * 2 >= GridOptions.y) result = min(result, MaximumHalfRadiance);
    return float4(result, 1);
}

#endif
