
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

static const float ShadowRaySurfaceStrength=0.55;
// Lumen froxel irradiance for transparent receivers. Real SH through band two;
// XYZ interpolation is clamped within coefficient bands, never across them.
cbuffer LumenTranslucentReceiver : register(b8)
{
    row_major float4x4 LumenVolumeCamera,LumenVolumeInverse;
    uint4 LumenVolumeGrid;
    float4 LumenVolumeDistribution,LumenVolumeAtlasSize;
};
Texture2D<float4> LumenVolumeImage : register(t19);
SamplerState LumenVolumeSampler : register(s4);
static const float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 worldPosition,float3 normal)
{
    if (LumenVolumeGrid.w==0) return 0;
    float4 clip=mul(LumenVolumeCamera,float4(worldPosition,1));
    if (clip.w<=LumenVolumeAtlasSize.z) return 0;
    float3 ndc=clip.xyz/clip.w;
    if (any(abs(ndc.xy)>1) || ndc.z<0 || ndc.z>1) return 0;
    float4 nearH=mul(LumenVolumeInverse,float4(ndc.xy,0,1)),farH=mul(LumenVolumeInverse,float4(ndc.xy,1,1));
    if (min(abs(nearH.w),abs(farH.w))<=LumenVolumeAtlasSize.z) return 0;
    float3 nearPoint=nearH.xyz/nearH.w;
    float extent=min(length(farH.xyz/farH.w-nearPoint),LumenVolumeDistribution.x);
    float distance=length(worldPosition-nearPoint);
    if (extent<=LumenVolumeAtlasSize.z || distance>extent) return 0;
    float2 uv=float2(ndc.x+1,1-ndc.y)*0.5;
    float3 cell=float3(uv*float2(LumenVolumeGrid.xy),pow(saturate(distance/extent),1/LumenVolumeDistribution.y)*LumenVolumeGrid.z)-0.5;
    cell=clamp(cell,0,float3(LumenVolumeGrid.xyz)-1);
    float lower=floor(cell.z),upper=min(lower+1,float(LumenVolumeGrid.z-1));
    normal*=rsqrt(max(dot(normal,normal),LumenVolumeAtlasSize.w));
    static const float SpherePi=3.14159265358979323846;
    static const uint CoefficientCount=9;
    float basis[CoefficientCount]={sqrt(1/(4*SpherePi)),sqrt(3/(4*SpherePi))*normal.y,sqrt(3/(4*SpherePi))*normal.z,sqrt(3/(4*SpherePi))*normal.x,
        sqrt(15/(4*SpherePi))*normal.x*normal.y,sqrt(15/(4*SpherePi))*normal.y*normal.z,
        sqrt(5/(16*SpherePi))*(3*normal.z*normal.z-1),sqrt(15/(4*SpherePi))*normal.x*normal.z,
        sqrt(15/(16*SpherePi))*(normal.x*normal.x-normal.y*normal.y)};
    float3 irradiance=0;

    static const uint VolumeVisibilityBit=2;
    if ((LumenVolumeGrid.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(cell)); float3 fraction=frac(cell);
        // Keep the original trilinear mass. Rejected support spheres contribute
        // no light, rather than renormalizing bright probes across an occluder.
        [loop] for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,LumenVolumeGrid.xyz-1);
            float3 weights=float3(bit.x?fraction.x:1-fraction.x,bit.y?fraction.y:1-fraction.y,bit.z?fraction.z:1-fraction.z);
            float weight=weights.x*weights.y*weights.z; if (weight<=0) continue;
            uint row=p.z*LumenVolumeGrid.y+p.y; float support=LumenVolumeImage.Load(int3(p.x,row,0)).a; if (support<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5)/LumenVolumeGrid.x-1,1-2*(float(p.y)+0.5)/LumenVolumeGrid.y);
            float4 a=mul(LumenVolumeInverse,float4(clipXY,0,1)),b=mul(LumenVolumeInverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=LumenVolumeAtlasSize.z) continue;
            float3 start=a.xyz/a.w,axis=b.xyz/b.w-start; float span=length(axis);
            if (!isfinite(span) || span<=LumenVolumeAtlasSize.z) continue;
            float depth=pow((float(p.z)+0.5)/LumenVolumeGrid.z,LumenVolumeDistribution.y);
            float3 probe=start+axis*(depth*min(span,LumenVolumeDistribution.x)/span);
            float separation=length(worldPosition-probe); if (!isfinite(separation) || separation>support) continue;
            [unroll] for (uint coefficient=0;coefficient<CoefficientCount;++coefficient)
                irradiance+=weight*basis[coefficient]*LumenVolumeImage.Load(int3(p.x+coefficient*LumenVolumeGrid.x,row,0)).rgb;
        }
        return max(irradiance,0)*(LumenVolumeDistribution.z/(SpherePi*LumenVolumeDistribution.w));
    }
    [unroll] for (uint coefficient=0;coefficient<CoefficientCount;++coefficient)
    {
        float x=coefficient*LumenVolumeGrid.x+cell.x+0.5;
        float2 a=float2(x,lower*LumenVolumeGrid.y+cell.y+0.5)/LumenVolumeAtlasSize.xy;
        float2 b=float2(x,upper*LumenVolumeGrid.y+cell.y+0.5)/LumenVolumeAtlasSize.xy;
        irradiance+=lerp(LumenVolumeImage.SampleLevel(LumenVolumeSampler,a,0).rgb,
            LumenVolumeImage.SampleLevel(LumenVolumeSampler,b,0).rgb,frac(cell.z))*basis[coefficient];
    }
    return max(irradiance,0)*(LumenVolumeDistribution.z/(SpherePi*LumenVolumeDistribution.w));
}

// DX11 physically based rectangle emitters: midpoint surface quadrature of the
// full GGX/Smith/Schlick BRDF. CPU preparation converts one-sided total power
// to radiance (Phi = pi * area * Le); W selects the N x N integration grid.
// Visibility comes from the selected raster shadow method, shared by emitter
// samples. This does not claim exact per-sample visibility or analytic GGX.
// Reference: PBRT 4e, Area Lights, https://pbr-book.org/4ed/Light_Sources/Area_Lights
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

// Contact visibility is an independent producer. Combine only the selected
// light's direct visibility; primary filters already include artistic strength.
// This algebra applies that strength once: (1-s) + s * mapRaw * contactRaw.
#if defined(EVANS_CONTACT_RECEIVER)
cbuffer ContactReceiverConstants : register(b6) { uint4 ContactControl; };
Texture2DArray<float> ContactMask : register(t15);
#endif
float ReadContact(float4 position, uint sampleIndex)
{
#if defined(EVANS_CONTACT_RECEIVER)
    if (ContactControl.x != 0) return ContactMask.Load(int4(int2(position.xy), sampleIndex, 0));
#endif
    return 1.0f;
}
float ComposeContact(float primary, float contact)
{
    const float ContactShadowStrength = 0.55f;
    return lerp(1.0f - ContactShadowStrength, primary, contact);
}

// DX11 PBR shading: standalone native HLSL, with forward material prerequisites.
// Numeric policy is explicit so these teaching shaders can be tuned without
// searching expressions. Registers and vertex semantics belong to the API ABI.
static const uint MaximumLights = 4u;
static const float MaximumHalfFloatRadiance = 65504.0f;
static const uint DirectionalLight = 0u;
static const uint PointLight = 1u;
static const float SrgbLinearScale = 12.92f;
static const float SrgbOffset = 0.055f;
static const float SrgbScale = 1.055f;
static const float SrgbGamma = 2.4f;
static const float SrgbThreshold = 0.04045f;
static const float NormalLengthEpsilon = 1.0e-12f;
static const float ProjectionEpsilon = 1.0e-7f;
static const float TangentDeterminantEpsilon = 1.0e-10f;
static const float LightDistanceEpsilon = 1.0e-6f;
static const float ParallelAxisLimit = 0.999f;
static const float ShadowMinimumCosine = 0.15f;
static const float ShadowMaximumBias = 0.0045f;
static const float ShadowConstantBias = 0.00035f;
static const float ShadowSlopeBias = 0.00060f;

static const float ShadowStrength = 0.55f;
static const float CoverageNoiseScale = 52.9829189f;
static const float2 CoverageNoiseDirection = float2(0.06711056f, 0.00583715f);
static const int AreaSamplesPerAxis = 2;
static const float AreaSampleOffset = 0.25f;
static const float AreaSampleWeight = 0.25f;
static const float OutlineWorldScale = 0.01f;
static const float OutlineModeThreshold = 1.5f;
static const float ByteMaximum = 255.0f;
static const float PackedOutlineMaximum = 65535.0f;
static const uint OutlineRadiusShift = 8u;

struct LightConstants
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};

cbuffer DrawConstants : register(b0)
{
    row_major float4x4 World;
    row_major float4x4 NormalWorld;
    row_major float4x4 ViewProjection;
    row_major float4x4 LightViewProjection;
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
    float4 ShadingParameters; // roughness, metallic, toon bands, HDR output
};

Texture2D<float> ShadowMap : register(t0);
Texture2D<float4> DiffuseMap : register(t1);
Texture2D<float4> NormalMap : register(t2);
Texture2D<float4> OpacityMap : register(t3);
SamplerState SurfaceSampler : register(s0);

float3 SRGBToLinear(float3 color)
{
    float3 low = color / SrgbLinearScale;
    float3 high = pow((color + SrgbOffset) / SrgbScale, SrgbGamma);
    return float3(
        color.r <= SrgbThreshold ? low.r : high.r,
        color.g <= SrgbThreshold ? low.g : high.g,
        color.b <= SrgbThreshold ? low.b : high.b);
}

float3 PrepareDisplayLinear(float3 color)
{
    color = max(color, 0.0f);
    // Float scene attachments retain radiance for the later HDR chain.
    if (ShadingParameters.w > 0.5f) return min(color, MaximumHalfFloatRadiance);
    color /= max(1.0f, max(color.r, max(color.g, color.b)));
    return saturate(color);
}

float3 NormalizeSafe(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalLengthEpsilon ? value * rsqrt(lengthSquared) : fallback;
}

struct VSInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};

struct VSOutput
{
    precise float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 LightPosition : TEXCOORD2;
    float2 TexCoord : TEXCOORD3;
};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float4 worldPosition = mul(World, float4(input.Position, 1.0f));
    float3 worldNormal = NormalizeSafe(
        mul((float3x3)NormalWorld, input.Normal),
        float3(0.0f, 1.0f, 0.0f));
    output.Position = mul(ViewProjection, worldPosition);
    output.WorldPosition = worldPosition.xyz / max(abs(worldPosition.w), ProjectionEpsilon);
    output.Normal = worldNormal;
    output.Color = float4(SRGBToLinear(saturate(input.Color.rgb)), input.Color.a);
    output.LightPosition = mul(LightViewProjection, worldPosition);
    output.TexCoord = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    return output;
}

float4 SampleDiffuse(float2 texCoord)
{
    return TextureFlags.x != 0
        ? DiffuseMap.Sample(SurfaceSampler, texCoord)
        : float4(1.0f, 1.0f, 1.0f, 1.0f);
}

float SampleOpacity(float2 texCoord)
{
    return TextureFlags.z != 0
        ? OpacityMap.Sample(SurfaceSampler, texCoord).r
        : 1.0f;
}

float SurfaceAlpha(VSOutput input, float4 diffuseSample)
{
    return saturate(
        input.Color.a *
        MaterialParameters.y *
        diffuseSample.a *
        SampleOpacity(input.TexCoord));
}

void ApplyAlphaClip(float alpha)
{
    if (MaterialParameters.z > 0.0f)
    {
        clip(alpha - MaterialParameters.z);
    }
}

// CSM consumer ABI (b2, t10..t13). Selection uses positive camera eye depth,
// not light depth. Overlap fits both maps; the last interval fades to visibility 1.
static const uint CascadeCapacity = 4u;

static const float CascadeMinimumCosine = 0.15f;
static const float CascadeConstantBias = 0.00035f;
static const float CascadeSlopeBias = 0.00060f;
static const float CascadeMaximumBias = 0.0045f;
static const float CascadeShadowStrength = 0.55f;
static const float CascadeNormalEpsilon = 1.0e-12f;
cbuffer CascadedShadowConstants : register(b2)
{
    row_major float4x4 CascadeLightViewProjection[CascadeCapacity];
    float4 CascadeFarDistances;
    float4 CascadeBlendStarts;
    float4 CascadeEyeDepthPlane;
    float4 CascadeParameters; // near, far, count, map resolution
    float4 ShadowFiltering; // radius (zero means hard), comparison count, reserved
};
Texture2D<float> CascadeMaps[CascadeCapacity] : register(t10);
float CascadeDepth(uint cascade, int2 pixel)
{
    // Shader Model 5 resource arrays require literal resource indices.
    if (cascade == 0u) return CascadeMaps[0].Load(int3(pixel, 0));
    if (cascade == 1u) return CascadeMaps[1].Load(int3(pixel, 0));
    if (cascade == 2u) return CascadeMaps[2].Load(int3(pixel, 0));
    return CascadeMaps[3].Load(int3(pixel, 0));
}
float CascadeVisibility(uint cascade, float3 world, float3 normal, float3 toLight)
{
    float3 projected = mul(CascadeLightViewProjection[cascade], float4(world, 1.0f)).xyz;
    float2 uv = float2(projected.x * 0.5f + 0.5f, 0.5f - projected.y * 0.5f);
    if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f) return 1.0f;
    float cosine = saturate(dot(normal, toLight * rsqrt(max(dot(toLight, toLight), CascadeNormalEpsilon))));
    float slope = sqrt(max(1.0f - cosine * cosine, 0.0f)) / max(cosine, CascadeMinimumCosine);
    float compareDepth = projected.z - min(CascadeMaximumBias, CascadeConstantBias + CascadeSlopeBias * slope);
    if (ShadowFiltering.x == 0.0f)
    {
        int2 pixel = clamp(int2(floor(uv * CascadeParameters.w)), 0, int2(CascadeParameters.ww) - 1);
        return 1.0f - CascadeShadowStrength * (compareDepth > CascadeDepth(cascade, pixel) ? 1.0f : 0.0f);
    }
    float2 texel = uv * CascadeParameters.w - 0.5f;
    int2 basePixel = int2(floor(texel));
    int2 lastPixel = int2(CascadeParameters.ww) - 1;
    float2 fraction = frac(texel);
    float occlusion = 0.0f;
    // Keep the filter loops compact; expanding resource-selection branches at
    // every tap inflates Shader Model 5 bytecode and stalls first-use compilation.
    [loop] for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        [loop] for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = basePixel + int2(x, y);
            float a = compareDepth > CascadeDepth(cascade, clamp(p, 0, lastPixel)) ? 1.0f : 0.0f;
            float b = compareDepth > CascadeDepth(cascade, clamp(p + int2(1, 0), 0, lastPixel)) ? 1.0f : 0.0f;
            float c = compareDepth > CascadeDepth(cascade, clamp(p + int2(0, 1), 0, lastPixel)) ? 1.0f : 0.0f;
            float d = compareDepth > CascadeDepth(cascade, clamp(p + 1, 0, lastPixel)) ? 1.0f : 0.0f;
            occlusion += lerp(lerp(a, b, fraction.x), lerp(c, d, fraction.x), fraction.y);
        }
    return 1.0f - CascadeShadowStrength * occlusion / ShadowFiltering.y;
}
// Virtual-map receiver ABI reuses the mutually exclusive cascade block:
// Far = logical resolution/page side/mips/directory width; Blend = atlas extent/
// columns/directory denominator. Every PCF corner resolves its own ancestor.
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
            uint entry = uint(round(CascadeMaps[1].Load(int3(address % logical.w, address / logical.w, 0)).x * physical.w));
            if (entry != 0)
            {
                uint slot = entry - 1, stride = logical.y + VirtualPageBorder * 2;
                uint2 tile = uint2(slot % physical.z, slot / physical.z) * stride;
                return float2(CascadeMaps[0].Load(int3(tile + VirtualPageBorder + pixel % logical.y, 0)).x, mip);
            }
        }
        base += side * side; side /= 2;
    }
    return float2(1, logical.z - 1);
}
// Compare receiver and caster at the same resolved virtual texel center.
// The actual ancestor mip is essential when detail pages are not resident.
float2 VirtualReceiverGradient(float2 uv, float depth)
{
    float2 x = ddx(uv), y = ddy(uv);
    float zx = ddx(depth), zy = ddy(depth);
    float determinant = x.x * y.y - x.y * y.x;
    float scale = max(dot(x, x) * dot(y, y), 1.0e-30);
    if (determinant * determinant <= scale * 1.0e-12) return float2(0.0, 0.0);
    return float2(y.y * zx - x.y * zy, x.x * zy - y.x * zx) / determinant;
}
float VirtualReceiverAt(float2 sampleUv, uint actualMip, float2 receiverUv, float receiver, float2 gradient)
{
    float resolution = float(uint(CascadeFarDistances.x) >> actualMip);
    float2 pixel = clamp(floor(sampleUv * resolution), float2(0.0, 0.0), float2(resolution - 1.0, resolution - 1.0));
    float2 center = (pixel + float2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualCompare(int2 pixel, uint mip, float2 receiverUv, float receiver, float2 gradient)
{
    uint resolution = uint(CascadeFarDistances.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0, 0), int2(int(resolution) - 1, int(resolution) - 1))) + float2(0.5, 0.5)) / float(resolution);
    float2 stored = VirtualDepth(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
float VirtualVisibility(float3 position, float bias)
{
    float4 light = mul(CascadeLightViewProjection[0], float4(position, 1.0));
    float3 projected = light.xyz / max(light.w, CascadeEyeDepthPlane.w);
    float2 uv = float2(projected.x, -projected.y) * 0.5 + float2(0.5, 0.5);
    float2 gradient = VirtualReceiverGradient(uv, projected.z);
    float footprint = max(length(ddx(uv) * CascadeEyeDepthPlane.y), length(ddy(uv) * CascadeEyeDepthPlane.z)) * CascadeFarDistances.x;
    uint mip = uint(clamp(floor(log2(max(1.0, footprint)) + CascadeEyeDepthPlane.x), 0.0, CascadeFarDistances.z - 1.0));
    if (light.w <= CascadeEyeDepthPlane.w || any(uv < 0.0) || any(uv > 1.0) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float2 stored = VirtualDepth(uv, mip);
    float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(CascadeFarDistances.x) >> mip) - float2(0.5, 0.5);
    int2 first = int2(floor(texel)); float2 weight = frac(texel); float visible = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = first + int2(x, y);
            float a = VirtualCompare(p, mip, uv, receiver, gradient);
            float b = VirtualCompare(p + int2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualCompare(p + int2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualCompare(p + int2(1, 1), mip, uv, receiver, gradient);
            visible += lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        }
    return visible / ShadowFiltering.y;
}


float CascadedVisibility(float3 world, float3 normal, float3 toLight)
{
    if (ShadowFiltering.z != 0)
    {
        float cosine = saturate(dot(normal, toLight * rsqrt(max(dot(toLight, toLight), CascadeNormalEpsilon))));
        float slope = sqrt(max(1 - cosine * cosine, 0)) / max(cosine, CascadeMinimumCosine);
        float bias = min(CascadeMaximumBias, CascadeConstantBias + CascadeSlopeBias * slope);
        return 1 - CascadeShadowStrength * (1 - VirtualVisibility(world, bias));
    }
    float distance = dot(CascadeEyeDepthPlane, float4(world, 1.0f));
    if (distance < CascadeParameters.x || distance > CascadeParameters.y) return 1.0f;
    uint count = uint(CascadeParameters.z);
    [loop] for (uint cascade = 0u; cascade < count; ++cascade)
    {
        if (distance > CascadeFarDistances[cascade]) continue;
        float visibility = CascadeVisibility(cascade, world, normal, toLight);
        float width = CascadeFarDistances[cascade] - CascadeBlendStarts[cascade];
        if (width > 0.0f && distance >= CascadeBlendStarts[cascade])
        {
            float next = cascade + 1u < count ? CascadeVisibility(cascade + 1u, world, normal, toLight) : 1.0f;
            visibility = lerp(visibility, next, saturate((distance - CascadeBlendStarts[cascade]) / width));
        }
        return visibility;
    }
    return 1.0f;
}

float SampleShadow(
    float3 worldPosition,
    float4 lightPosition,
    float3 surfaceNormal,
    float3 directionToLight)
{
    if (RenderParameters.x>=0.5f && ShadowRayEnabled()) return 1.0-ShadowRaySurfaceStrength*(1.0-ShadowRayVisibility(worldPosition,surfaceNormal,directionToLight));
    if (RenderParameters.x >= 0.5f && CascadeParameters.z > 0.0f)
        return CascadedVisibility(worldPosition, surfaceNormal, directionToLight);
    if (RenderParameters.x < 0.5f || abs(lightPosition.w) < ProjectionEpsilon)
    {
        return 1.0f;
    }
    float3 projected = lightPosition.xyz / lightPosition.w;
    float2 uv = float2(projected.x * 0.5f + 0.5f, 0.5f - projected.y * 0.5f);
    if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f)
    {
        return 1.0f;
    }
    uint width;
    uint height;
    ShadowMap.GetDimensions(width, height);
    if (width == 0 || height == 0)
    {
        return 1.0f;
    }
    float lightDirectionLengthSquared =
        dot(directionToLight, directionToLight);
    float3 normalizedLightDirection =
        lightDirectionLengthSquared > NormalLengthEpsilon
        ? directionToLight * rsqrt(lightDirectionLengthSquared)
        : surfaceNormal;
    float normalToLight = saturate(
        dot(surfaceNormal, normalizedLightDirection));
    float sineToLight = sqrt(max(
        1.0f - normalToLight * normalToLight,
        0.0f));
    float slope = sineToLight / max(normalToLight, ShadowMinimumCosine);
    float receiverBias = min(
        ShadowMaximumBias,
        ShadowConstantBias + ShadowSlopeBias * slope);

    // Hard mode selects one nearest depth texel; PCF averages comparisons.
    if (ShadowFiltering.x == 0.0f)
    {
        int2 pixel = clamp(int2(floor(uv * float2(width, height))), 0, int2(width, height) - 1);
        return 1.0f - ShadowStrength * ((projected.z - receiverBias) > ShadowMap.Load(int3(pixel, 0)) ? 1.0f : 0.0f);
    }
    float2 texelPosition =
        uv * float2(width, height) - 0.5f;
    float2 interpolation = frac(texelPosition);
    int2 basePixel = int2(floor(texelPosition));
    float occlusion = 0.0f;
    [loop]
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
    {
        [loop]
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 pixel00 = clamp(
                basePixel + int2(x, y),
                int2(0, 0),
                int2(width - 1, height - 1));
            int2 pixel10 = clamp(
                pixel00 + int2(1, 0),
                int2(0, 0),
                int2(width - 1, height - 1));
            int2 pixel01 = clamp(
                pixel00 + int2(0, 1),
                int2(0, 0),
                int2(width - 1, height - 1));
            int2 pixel11 = clamp(
                pixel00 + int2(1, 1),
                int2(0, 0),
                int2(width - 1, height - 1));
            float comparisonDepth = projected.z - receiverBias;
            float comparison00 = comparisonDepth >
                ShadowMap.Load(int3(pixel00, 0)) ? 1.0f : 0.0f;
            float comparison10 = comparisonDepth >
                ShadowMap.Load(int3(pixel10, 0)) ? 1.0f : 0.0f;
            float comparison01 = comparisonDepth >
                ShadowMap.Load(int3(pixel01, 0)) ? 1.0f : 0.0f;
            float comparison11 = comparisonDepth >
                ShadowMap.Load(int3(pixel11, 0)) ? 1.0f : 0.0f;
            float comparisonTop = lerp(
                comparison00,
                comparison10,
                interpolation.x);
            float comparisonBottom = lerp(
                comparison01,
                comparison11,
                interpolation.x);
            occlusion += lerp(
                comparisonTop,
                comparisonBottom,
                interpolation.y);
        }
    }
    return 1.0f - ShadowStrength * occlusion / ShadowFiltering.y;
}

float3 ResolveSurfaceNormal(VSOutput input, bool isFrontFace)
{
    float3 normal = NormalizeSafe(input.Normal, float3(0.0f, 1.0f, 0.0f));
    if (!isFrontFace)
    {
        normal = -normal;
    }
    if (TextureFlags.y == 0)
    {
        return normal;
    }

    float3 positionDx = ddx(input.WorldPosition);
    float3 positionDy = ddy(input.WorldPosition);
    float2 texCoordDx = ddx(input.TexCoord);
    float2 texCoordDy = ddy(input.TexCoord);
    float determinant =
        texCoordDx.x * texCoordDy.y -
        texCoordDx.y * texCoordDy.x;
    if (abs(determinant) < TangentDeterminantEpsilon)
    {
        return normal;
    }

    float inverseDeterminant = rcp(determinant);
    float3 tangent = (positionDx * texCoordDy.y -
        positionDy * texCoordDx.y) * inverseDeterminant;
    float3 sourceBitangent = (-positionDx * texCoordDy.x +
        positionDy * texCoordDx.x) * inverseDeterminant;
    tangent = tangent - normal * dot(normal, tangent);
    float3 tangentHelper = abs(normal.y) < ParallelAxisLimit
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    float3 tangentFallback = NormalizeSafe(
        cross(tangentHelper, normal),
        float3(1.0f, 0.0f, 0.0f));
    tangent = NormalizeSafe(tangent, tangentFallback);
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0f
        ? -1.0f
        : 1.0f;
    float3 bitangent = NormalizeSafe(
        cross(normal, tangent) * handedness,
        float3(0.0f, 0.0f, 1.0f));
    float3 tangentNormal =
        NormalMap.Sample(SurfaceSampler, input.TexCoord).xyz * 2.0f - 1.0f;
    return NormalizeSafe(
        tangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

// Cook and Torrance (1982); GGX distribution and Smith masking from
// Walter et al., "Microfacet Models for Refraction through Rough Surfaces" (2007).
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
void AccumulateLightSample(float2 roughnessMetallic,
    float3 lightDirection, float3 radiance, float visibility,
    float3 normal, float3 viewDirection,
    inout float3 diffuseLight, inout float3 specularLight, float3 baseColor)
{
    const float Pi = 3.14159265359f;
    const float MinimumRoughness = 0.045f;
    const float DielectricReflectance = 0.04f;
    const float BrdfDenominatorEpsilon = 1.0e-6f;
    const float DistributionDenominatorEpsilon = 1.0e-12f;
    const float SchlickExponent = 5.0f;
    float3 direction = NormalizeSafe(lightDirection, normal);
    float noL = saturate(dot(normal, direction));
    float noV = saturate(dot(normal, viewDirection));
    if (noL <= 0.0f || noV <= 0.0f) return;
    float3 halfway = NormalizeSafe(direction + viewDirection, normal);
    float noH = saturate(dot(normal, halfway));
    float voH = saturate(dot(viewDirection, halfway));
    float roughness = clamp(roughnessMetallic.x, MinimumRoughness, 1.0f);
    float metallic = saturate(roughnessMetallic.y);
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float denominator = noH * noH * (alphaSquared - 1.0f) + 1.0f;
    float distribution = alphaSquared / max(Pi * denominator * denominator, DistributionDenominatorEpsilon);
    float smithView = 2.0f * noV / max(noV + sqrt(alphaSquared + (1.0f - alphaSquared) * noV * noV), BrdfDenominatorEpsilon);
    float smithLight = 2.0f * noL / max(noL + sqrt(alphaSquared + (1.0f - alphaSquared) * noL * noL), BrdfDenominatorEpsilon);
    float3 normalReflectance = lerp(DielectricReflectance.xxx, saturate(baseColor), metallic);
    float3 fresnel = normalReflectance + (1.0f - normalReflectance) * pow(1.0f - voH, SchlickExponent);
    float3 incident = radiance * visibility * noL;
    diffuseLight += incident * (1.0f - fresnel) * (1.0f - metallic) / Pi;
    specularLight += incident * distribution * smithView * smithLight * fresnel
        / max(4.0f * noV * noL, BrdfDenominatorEpsilon);
}

void AccumulateSceneLight(float2 roughnessMetallic,
    uint lightIndex,
    float contactVisibility,
    VSOutput input,
    float3 normal,
    float3 viewDirection,
    inout float3 diffuseLight,
    inout float3 specularLight, float3 baseColor)
{
    LightConstants light = Lights[lightIndex];
    uint type = (uint)(light.PositionType.w + 0.5f);
    float3 shadowDirection = type == DirectionalLight
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - input.WorldPosition;
    bool isShadowLight =
        int(lightIndex) == int(RenderParameters.z) &&
        light.ColorCastShadows.w > 0.5f;
    float visibility = isShadowLight
        ? SampleShadow(
            input.WorldPosition, input.LightPosition,
            normal,
            shadowDirection)
        : 1.0f;
    if (isShadowLight) visibility = ComposeContact(visibility, contactVisibility);
    float3 radiance =
        max(light.ColorCastShadows.rgb, 0.0f) *
        max(light.DirectionIntensity.w, 0.0f);

    if (type == DirectionalLight)
    {
        AccumulateLightSample(roughnessMetallic,
            light.DirectionIntensity.xyz,
            radiance,
            visibility,
            normal,
            viewDirection,
            diffuseLight,
            specularLight, baseColor);
        return;
    }

    if (type == PointLight)
    {
        float3 toLight = light.PositionType.xyz - input.WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), LightDistanceEpsilon);
        AccumulateLightSample(roughnessMetallic,
            toLight,
            radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared,
            visibility,
            normal,
            viewDirection,
            diffuseLight,
            specularLight, baseColor);
        return;
    }

    float3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz,
        float3(0.0f, -1.0f, 0.0f));
    float3 helper = abs(emissionDirection.y) < ParallelAxisLimit
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = NormalizeSafe(
        cross(helper, emissionDirection),
        float3(1.0f, 0.0f, 0.0f));
    float3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent),
        float3(0.0f, 0.0f, 1.0f));
    // Area-to-solid-angle Jacobian: dA * emitterCosine / distanceSquared.
    const float MinimumEmitterGrid = 2.0, MaximumEmitterGrid = 16.0;
    const float EmitterStratumCenter = 0.5;
    bool physicalEmitter = light.AreaSizePadding.w > 0.0;
    int emitterGrid = physicalEmitter ? int(clamp(light.AreaSizePadding.w, MinimumEmitterGrid, MaximumEmitterGrid)) : AreaSamplesPerAxis;
    float emitterSampleWeight = (physicalEmitter ? abs(light.AreaSizePadding.x * light.AreaSizePadding.y) : 1.0) / float(emitterGrid * emitterGrid);
    [loop]
    for (int sampleY = 0; sampleY < emitterGrid; ++sampleY)
    {
        [loop]
        for (int sampleX = 0; sampleX < emitterGrid; ++sampleX)
        {
            float offsetX = ((float(sampleX) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter);
            float offsetY = ((float(sampleY) + EmitterStratumCenter) / float(emitterGrid) - EmitterStratumCenter);
            float3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            float3 toLight = samplePosition - input.WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), LightDistanceEpsilon);
            float3 direction = toLight * rsqrt(distanceSquared);
            float emitterCosine = max(
                dot(emissionDirection, -direction),
                0.0f);
            AccumulateLightSample(roughnessMetallic,
                direction,
                radiance * (emitterSampleWeight * emitterCosine * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared),
                visibility,
                normal,
                viewDirection,
                diffuseLight,
                specularLight, baseColor);
        }
    }
}

// Consumer of the independently integrated fog atlas. Sampling is repeated here
// to keep this shading technique self-contained; density integration belongs to
// VolumetricFog.hlsl. The transparent surface keeps its original coverage alpha.
cbuffer SurfaceFogConstants : register(b3)
{
    row_major float4x4 FogInverseViewProjection;
    float4 FogMedium;
    float4 FogSourceRadiance;
    float4 FogViewportGrid;
    uint4 FogOptions;
};
Texture2D<float4> FogAtlas : register(t14);
SamplerState FogLinearClamp : register(s3);
static const float FogMinimumRayLength = 1.0e-6f;
static const float FogMaximumHalfRadiance = 65504.0f;

float3 FogWorldPosition(float2 uv, float depth)
{
    float4 world = mul(FogInverseViewProjection, float4(uv * float2(2, -2) + float2(-1, 1), depth, 1));
    return world.xyz / world.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
static const uint RgbTransportMode = 2;
float4 FogEndpointTransportTransmission(float2 uv, uint endpoint)
{
    if (endpoint == 0) return float4(1, 1, 1, 1);
    float2 halfCell = 0.5f / FogViewportGrid.zw;
    float2 atlasUV = clamp(uv, halfCell, 1 - halfCell);
    atlasUV.x = (atlasUV.x + endpoint - 1) / FogOptions.y;
    atlasUV.y = (atlasUV.y + 1.0) * 0.5;
    return FogAtlas.SampleLevel(FogLinearClamp, atlasUV, 0);
}
float4 FogEndpointTransport(float2 uv, uint endpoint)
{
    if (endpoint == 0) return float4(0, 0, 0, 1);
    float2 halfCell = 0.5f / FogViewportGrid.zw;
    float2 atlasUV = clamp(uv, halfCell, 1 - halfCell);
    atlasUV.x = (atlasUV.x + endpoint - 1) / FogOptions.y;
    if (FogOptions.w == RgbTransportMode) atlasUV.y *= 0.5;
    return FogAtlas.SampleLevel(FogLinearClamp, atlasUV, 0);
}
float3 ApplySurfaceFog(float3 radiance, float4 position)
{
    if (FogOptions.w == 0) return radiance;
    float2 uv = position.xy / FogViewportGrid.xy;
    float3 start = FogWorldPosition(uv, 0);
    float extent = min(length(FogWorldPosition(uv, 1) - start), FogMedium.w);
    float endpoint = saturate(length(FogWorldPosition(uv, position.z) - start) /
        max(extent, FogMinimumRayLength)) * FogOptions.y;
    uint lower = (uint)floor(endpoint);
    uint upper = min(lower + 1, FogOptions.y);
    float4 transport = lerp(FogEndpointTransport(uv, lower), FogEndpointTransport(uv, upper), frac(endpoint));
    float3 mediumTransmission = FogOptions.w == RgbTransportMode ? (lerp(FogEndpointTransportTransmission(uv, lower), FogEndpointTransportTransmission(uv, upper), frac(endpoint))).rgb : transport.aaa;
    return clamp(radiance * mediumTransmission + transport.rgb, 0, FogMaximumHalfRadiance);
}

#if defined(EVANS_CONTACT_RECEIVER)
float4 PSContact(VSOutput input, bool isFrontFace : SV_IsFrontFace, uint contactSample : SV_SampleIndex) : SV_TARGET
#else
float4 PSMain(VSOutput input, bool isFrontFace : SV_IsFrontFace) : SV_TARGET
#endif
{
    float2 roughnessMetallic = ShadingParameters.xy;
    if (TextureFlags.z > 1) roughnessMetallic *= OpacityMap.Sample(SurfaceSampler, input.TexCoord).gb;
#if !defined(EVANS_CONTACT_RECEIVER)
    const uint contactSample = 0u;
#endif
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    float alpha = SurfaceAlpha(input, diffuseSample);
    ApplyAlphaClip(alpha);
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (RenderParameters.y > 1.0 && RenderParameters.y < 1.5)
    {
        float3 wireBary = input.Color.rgb;
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (RenderParameters.y > 0.5f)
    {
        return float4(
            ApplySurfaceFog(SRGBToLinear(saturate(PipelineColor.rgb)), input.Position),
            alpha);
    }

    float3 surfaceBase =
        input.Color.rgb *
        max(PipelineColor.rgb, 0.0f) *
        max(diffuseSample.rgb, 0.0f);
    float3 baseColor = saturate(surfaceBase * MaterialDiffuseColor.rgb);
    float3 normal = ResolveSurfaceNormal(input, isFrontFace);
    float3 viewDirection = NormalizeSafe(
        CameraLightCount.xyz - input.WorldPosition,
        normal);
    float3 diffuseLight = 0.0f;
    float3 specularLight = 0.0f;
    uint lightCount = min((uint)(CameraLightCount.w + 0.5f), MaximumLights);
    [loop]
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        AccumulateSceneLight(roughnessMetallic,
            lightIndex,
            ReadContact(input.Position, contactSample),
            input,
            normal,
            viewDirection,
            diffuseLight,
            specularLight, baseColor);
    }

    float3 color =
        surfaceBase *
            max(MaterialAmbientColor.rgb, 0.0f) *
                max(AmbientColorStrength.rgb, 0.0f) *
                max(AmbientColorStrength.w, 0.0f) * (1.0f - saturate(roughnessMetallic.y)) +
        surfaceBase *
            max(MaterialDiffuseColor.rgb, 0.0f) *
            diffuseLight +
        specularLight +
        max(MaterialEmissiveColor.rgb, 0.0f);
    color += surfaceBase * max(MaterialDiffuseColor.rgb,0) * (1-LumenDielectricReflectance) * (1-saturate(roughnessMetallic.y)) * LumenTranslucentIrradiance(input.WorldPosition,normal);
    return float4(PrepareDisplayLinear(ApplySurfaceFog(color, input.Position)), alpha);
}

// Weighted OIT consumes the same lit, fogged radiance as ordinary blending.
// Its separate resolve source normalizes the additive statistics and applies
// product revealage. Bounded weights keep HDR accumulation numerically stable.
cbuffer WeightedTransparencyConstants : register(b4)
{
    float4 TransparencyWeights; // opacity exponent, depth exponent, depth scale, minimum weight
    uint4 TransparencyOptions;
};
#if defined(EVANS_CONTACT_RECEIVER)
float4 PSMain(VSOutput input, bool isFrontFace : SV_IsFrontFace) : SV_TARGET
{ return PSContact(input, isFrontFace, 0u); }
#endif

struct WeightedSurfaceOutput
{
    float4 Accumulation : SV_Target0;
    float Revealage : SV_Target1;
};
WeightedSurfaceOutput PSWeighted(VSOutput input, bool isFrontFace : SV_IsFrontFace)
{
    float4 surface = PSMain(input, isFrontFace);
    WeightedSurfaceOutput output = (WeightedSurfaceOutput)0;
    float coverage = saturate(surface.a);
    if (coverage == 0.0) return output;
    float weight = max(pow(coverage, TransparencyWeights.x) *
        pow(1.0 - saturate(input.Position.z) * TransparencyWeights.z, TransparencyWeights.y), TransparencyWeights.w);
    float weightedCoverage = coverage * weight;
    output.Accumulation = float4(surface.rgb * weightedCoverage, weightedCoverage);
    output.Revealage = coverage;
    return output;
}

float4 PSOutline(VSOutput input) : SV_TARGET
{
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    ApplyAlphaClip(SurfaceAlpha(input, diffuseSample));
    uint radius = (uint)round(clamp(
        MaterialParameters.w,
        0.0f,
        ByteMaximum));
    uint opacity = (uint)round(
        saturate(PipelineColor.a) * ByteMaximum);
    uint packedOutline = (radius << OutlineRadiusShift) | opacity;
    return float4(
        saturate(PipelineColor.rgb),
        (float)packedOutline / PackedOutlineMaximum);
}
