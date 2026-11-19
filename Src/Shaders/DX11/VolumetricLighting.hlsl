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

// Direct single scattering in an exponential height medium, native DirectX 11.
// Henyey-Greenstein: https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
// Each invocation injects one depth segment. An inclusive transport scan then
// composes (S,T) operators; light evaluation is not repeated for every endpoint.
// Opaque visibility uses the primary shadow map or CSM. Light-path medium
// extinction and multiple scattering are not modeled. Rectangles use quadrature.
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
};
Texture2D<float4> PreviousTransport : register(t0);
Texture2D<float> SingleShadow : register(t1);
Texture2D<float> Cascade0 : register(t2);
Texture2D<float> Cascade1 : register(t3);
Texture2D<float> Cascade2 : register(t4);
Texture2D<float> Cascade3 : register(t5);

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
float DepthCompare(Texture2D<float> map, int2 pixel, uint2 size, float receiver)
{
    if (any(pixel < 0) || any(pixel >= int2(size))) return 1;
    return receiver <= map.Load(int3(pixel, 0)) ? 1.0 : 0.0;
}
float BilinearComparison(Texture2D<float> map, float2 uv, uint2 size, float receiver)
{
    float2 texel = uv * size - 0.5;
    int2 corner = int2(floor(texel));
    float2 fraction = frac(texel);
    return lerp(lerp(DepthCompare(map, corner, size, receiver), DepthCompare(map, corner + int2(1, 0), size, receiver), fraction.x),
        lerp(DepthCompare(map, corner + int2(0, 1), size, receiver), DepthCompare(map, corner + 1, size, receiver), fraction.x), fraction.y);
}
float MapVisibility(Texture2D<float> map, float4 clip)
{
    if (clip.w <= ProjectionEpsilon) return 1;
    float3 projected = clip.xyz / clip.w;
    float2 uv = projected.xy * float2(0.5, -0.5) + 0.5;
    if (any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1) return 1;
    uint width, height; map.GetDimensions(width, height);
    uint2 size = uint2(width, height);
    float receiver = projected.z - LightParameters.y;
    int radius = int(ShadowFiltering.x);
    if (radius == 0) return DepthCompare(map, min(int2(uv * size), int2(size) - 1), size, receiver);
    float sum = 0;
    [loop] for (int y = -radius; y <= radius; ++y)
        [loop] for (int x = -radius; x <= radius; ++x)
            sum += BilinearComparison(map, uv + float2(x, y) / size, size, receiver);
    return sum / ShadowFiltering.y;
}
float CascadeVisibility(uint index, float3 position)
{
    float4 clip = mul(CascadeLightViewProjection[index], float4(position, 1));
    if (index == 0) return MapVisibility(Cascade0, clip);
    if (index == 1) return MapVisibility(Cascade1, clip);
    if (index == 2) return MapVisibility(Cascade2, clip);
    return MapVisibility(Cascade3, clip);
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
    if (ShadowFiltering.z != 0 && Control.y != 0) return VirtualVisibility(position, LightParameters.y);
    if (Control.y == SingleShadowMode) return MapVisibility(SingleShadow, mul(SingleLightViewProjection, float4(position, 1)));
    if (Control.y != CascadedShadowMode) return 1;
    float eyeDepth = dot(EyeDepthPlane, float4(position, 1));
    if (eyeDepth < CascadeParameters.x || eyeDepth > CascadeParameters.y) return 1;
    uint count = uint(CascadeParameters.z), index = 0;
    while (index + 1 < count && eyeDepth > CascadeFarDistances[index]) ++index;
    float visibility = CascadeVisibility(index, position);
    if (index + 1 < count && eyeDepth > CascadeBlendStarts[index])
    {
        float span = max(CascadeFarDistances[index] - CascadeBlendStarts[index], MinimumRayLength);
        visibility = lerp(visibility, CascadeVisibility(index + 1, position), saturate((eyeDepth - CascadeBlendStarts[index]) / span));
    }
    return visibility;
}
float3 DirectSource(float3 position, float3 cameraRay)
{
    float3 source = 0;
    float visibility = Control.y != 0 ? PrimaryVisibility(position) : 1;
    [loop] for (uint index = 0; index < Control.x; ++index)
    {
        VolumeLight light;
        if (LightParameters.z != 0.0) light = VolumeSceneLights[index];
        else light = Lights[index];
        uint type = uint(light.PositionType.w);
        float3 radiance = min(light.ColorCastShadows.rgb, MaximumHalfRadiance) * min(light.DirectionIntensity.w, MaximumHalfRadiance);
        float3 rayToward=uint(light.PositionType.w)==0u?light.DirectionIntensity.xyz:light.PositionType.xyz-position;
        float shadow=index==Control.z && Control.y!=0?(ShadowRayEnabled()?ShadowRayVisibility(position,float3(0,0,0),rayToward):visibility):1.0;
        float3 direction = SafeDirection(light.DirectionIntensity.xyz, float3(0, 1, 0));
        float contribution = 0;
        if (type == DirectionalLight) contribution = Phase(direction, cameraRay);
        else if (type == PointLight)
        {
            float3 delta = light.PositionType.xyz - position;
            contribution = Phase(SafeDirection(delta, direction), cameraRay) * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), MinimumSquaredLightDistance);
        }
        else
        {
            float3 helper = abs(direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
            float3 tangent = normalize(cross(helper, direction));
            float3 bitangent = cross(direction, tangent);
            // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
            const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
            const float EmitterStratumCenter = 0.5;
            bool physicalEmitter = light.AreaSize.w > 0.0;
            uint emitterGrid = physicalEmitter ? uint(clamp(light.AreaSize.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AreaSamplesPerAxis;
            float emitterSampleWeight = 1.0 / float(emitterGrid * emitterGrid);
            [loop] for (uint y = 0; y < emitterGrid; ++y)
                [loop] for (uint x = 0; x < emitterGrid; ++x)
                {
                    float2 offset = (float2(x, y) + 0.5) / emitterGrid - 0.5;
                    float3 emitter = light.PositionType.xyz + tangent * offset.x * light.AreaSize.x + bitangent * offset.y * light.AreaSize.y;
                    float3 delta = emitter - position;
                    float3 toLight = SafeDirection(delta, -direction);
                    contribution += emitterSampleWeight * saturate(dot(direction, -toLight)) * Phase(toLight, cameraRay) * LightRangeWindow(delta, light.AreaSize.z) / max(dot(delta, delta), MinimumSquaredLightDistance);
                }
            contribution *= light.AreaSize.x * light.AreaSize.y;
        }
        source += radiance * (shadow * contribution);
    }
    return Scattering.rgb * LightParameters.x * source;
}
float4 PSInject(float4 position : SV_POSITION) : SV_TARGET
{
    uint2 pixel = uint2(position.xy);
    uint gridWidth = uint(ViewportGrid.z), slice = pixel.x / gridWidth;
    float2 uv = (float2(pixel.x % gridWidth, pixel.y) + 0.5) / ViewportGrid.zw;
    float3 start = WorldPosition(uv, 0);
    float3 ray = WorldPosition(uv, 1) - start;
    float rayLength = max(length(ray), MinimumRayLength);
    float3 direction = ray / rayLength;
    float stepLength = min(rayLength, Medium.w) / (GridOptions.y * GridOptions.z);
    float4 transport = IdentityTransport;
    [loop] for (uint sampleIndex = 0; sampleIndex < GridOptions.z; ++sampleIndex)
    {
        float distance = (slice * GridOptions.z + sampleIndex + 0.5) * stepLength;
        float3 samplePosition = start + direction * distance;
        float extinction = Medium.x * exp(-min(Medium.z * max(samplePosition.y - Medium.y, 0), MaximumOpticalDepth));
        float segmentT = exp(-min(extinction * stepLength, MaximumOpticalDepth));
        float3 source = AmbientSource.rgb + DirectSource(samplePosition, direction);
        transport.rgb += transport.a * source * (1 - segmentT);
        transport.a *= segmentT;
    }
    return transport;
}
float4 PSPrefix(float4 position : SV_POSITION) : SV_TARGET
{
    int2 pixel = int2(position.xy);
    uint gridWidth = uint(ViewportGrid.z), slice = uint(pixel.x) / gridWidth;
    float4 right = PreviousTransport.Load(int3(pixel, 0));
    float4 combined = right;
    if (slice >= Control.w)
    {
        float4 left = PreviousTransport.Load(int3(pixel - int2(Control.w * gridWidth, 0), 0));
        combined = float4(left.rgb + left.a * right.rgb, left.a * right.a);
    }
    // Only the final prefix writes the shared half-float atlas. Earlier clamping
    // would lose bright distant scattering before nearer extinction attenuates it.
    if (Control.w * 2 >= GridOptions.y) combined.rgb = min(combined.rgb, MaximumHalfRadiance);
    return combined;
}
