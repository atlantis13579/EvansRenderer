// Cook-Torrance BRDF: GGX distribution, exact separable Smith masking, Schlick Fresnel, energy-conserving Lambert diffuse. Walter et al. (2007), Microfacet Models for Refraction through Rough Surfaces.
#include <metal_stdlib>
using namespace metal;
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

constant uint MaximumLights = 4;

constant float ShadowStrength = 0.55;
constant float Pi = 3.141592653589793;
constant float MinimumRoughness = 0.045;
constant float BrdfEpsilon = 1.0e-6;
constant float DistributionEpsilon = 1.0e-12;
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


struct EvansVertex
{
    float Values[12];
};

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
    float3 WorldPosition;
    float3 Normal;
    float4 Color;
    float4 LightPosition;
    float2 TexCoord;
};

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

vertex VertexOutput EvansVertexMain(device const EvansVertex* vertices [[buffer(0)]],
    constant DrawConstants& constants [[buffer(1)]], uint vertexId [[vertex_id]])
{
    const device EvansVertex& input = vertices[vertexId];
    const float3 position = float3(input.Values[0], input.Values[1], input.Values[2]);
    const float3 inputNormal = float3(input.Values[3], input.Values[4], input.Values[5]);
    float4 worldPosition = TransformRows(constants.World, float4(position, 1.0));
    float3 normal = NormalizeSafe(
        TransformRows(constants.NormalWorld, float4(inputNormal, 0.0)).xyz,
        float3(0.0, 1.0, 0.0));
    if (constants.RenderParameters.y > OutlineMode)
        worldPosition.xyz += normal * (constants.MaterialParameters.w * OutlineWorldScale);

    VertexOutput output;
    output.Position = TransformRows(constants.ViewProjection, worldPosition);
    output.WorldPosition = worldPosition.xyz / max(abs(worldPosition.w), HomogeneousEpsilon);
    output.Normal = normal;
    output.Color = float4(
        SRGBToLinear(saturate(float3(input.Values[6], input.Values[7], input.Values[8]))),
        input.Values[9]);
    output.LightPosition = TransformRows(constants.LightViewProjection, worldPosition);
    output.TexCoord = float2(input.Values[10], 1.0 - input.Values[11]);
    return output;
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
constant float CsmProjectionEpsilon = 1.0e-7;
constant float CsmDirectionEpsilon = 1.0e-12;
constant float CsmOverlapEpsilon = 1.0e-6;
constant float CsmMinimumCosine = 0.15;
constant float CsmConstantBias = 0.00035;
constant float CsmSlopeBias = 0.00060;
constant float CsmMaximumBias = 0.0045;
constant float CsmShadowStrength = 0.55;

struct CascadeConstants
{
    float4 LightRows[CsmCascadeLimit][4];
    float4 FarDepths;
    float4 BlendDepths;
    float4 EyeDepthPlane;
    float4 Parameters; // near, far, count, map resolution
    float4 Filtering; // radius (zero means hard), comparison count, reserved
};

float CsmCompare(depth2d<float, access::read> map, int2 pixel, int2 maximumPixel, float receiver)
{
    return receiver > map.read(uint2(clamp(pixel, int2(0), maximumPixel))) ? 1.0 : 0.0;
}
float CsmFilter(depth2d<float, access::read> map, constant float4* rows, float3 position, float bias, float4 filtering)
{
    float4 world(position, 1.0);
    float4 light(dot(rows[0], world), dot(rows[1], world), dot(rows[2], world), dot(rows[3], world));
    if (abs(light.w) < CsmProjectionEpsilon) return 1.0;
    float3 ndc = light.xyz / light.w;
    float2 uv(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (!all(uv >= 0.0) || !all(uv <= 1.0) || ndc.z < 0.0 || ndc.z > 1.0) return 1.0;
    uint2 extent(map.get_width(), map.get_height());
    if (filtering.x == 0.0)
        return 1.0 - CsmShadowStrength * CsmCompare(map, int2(floor(uv * float2(extent))), int2(extent) - 1, ndc.z - bias);
    float2 texel = uv * float2(extent) - 0.5;
    int2 base = int2(floor(texel));
    int2 maximumPixel = int2(extent) - 1;
    float2 fraction = fract(texel);
    float receiver = ndc.z - bias;
    float blocked = 0.0;
    // Read and compare D32 texels before bilinear interpolation; a filtered
    // depth value would create false occluders across discontinuities.
    for (int row = -int(filtering.x); row <= int(filtering.x); ++row)
        for (int column = -int(filtering.x); column <= int(filtering.x); ++column)
        {
            int2 pixel = base + int2(column, row);
            float upperLeft = CsmCompare(map, pixel, maximumPixel, receiver);
            float upperRight = CsmCompare(map, pixel + int2(1, 0), maximumPixel, receiver);
            float lowerLeft = CsmCompare(map, pixel + int2(0, 1), maximumPixel, receiver);
            float lowerRight = CsmCompare(map, pixel + int2(1, 1), maximumPixel, receiver);
            blocked += mix(mix(upperLeft, upperRight, fraction.x), mix(lowerLeft, lowerRight, fraction.x), fraction.y);
        }
    return 1.0 - CsmShadowStrength * blocked * (1.0 / filtering.y);
}
float CsmSample(uint index, float3 position, float bias, constant CascadeConstants& cascades,
    depth2d<float, access::read> map0, depth2d<float, access::read> map1,
    depth2d<float, access::read> map2, depth2d<float, access::read> map3)
{
    // Ordinary resource arguments keep this independent of argument-buffer tiers.
    if (index == 0) return CsmFilter(map0, cascades.LightRows[0], position, bias, cascades.Filtering);
    if (index == 1) return CsmFilter(map1, cascades.LightRows[1], position, bias, cascades.Filtering);
    if (index == 2) return CsmFilter(map2, cascades.LightRows[2], position, bias, cascades.Filtering);
    return CsmFilter(map3, cascades.LightRows[3], position, bias, cascades.Filtering);
}
// The cascade argument block also describes the exclusive virtual hierarchy.
// FarDepths = logical extent; BlendDepths = atlas extent and directory scale.
// Both images are point-read D32 resources, including the normalized directory.
constant uint VirtualTileBorder = 1;
constant float VirtualSampleCenter = 0.5;
float2 VirtualDepth(float2 uv, uint firstMip, constant CascadeConstants& frame,
    depth2d<float, access::read> atlas, depth2d<float, access::read> directory)
{
    uint4 layout(frame.FarDepths), physical(frame.BlendDepths);
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
    int2 base(floor(texel)); float2 fraction = fract(texel); float visible = 0;
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


float CsmVisibility(float3 position, float3 normal, float3 directionToLight, constant CascadeConstants& cascades,
    depth2d<float, access::read> map0, depth2d<float, access::read> map1,
    depth2d<float, access::read> map2, depth2d<float, access::read> map3)
{
    if (cascades.Filtering.z != 0)
    {
        float cosine = saturate(dot(normal, directionToLight * rsqrt(max(dot(directionToLight, directionToLight), CsmDirectionEpsilon))));
        float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CsmMinimumCosine);
        float bias = min(CsmMaximumBias, CsmConstantBias + CsmSlopeBias * slope);
        return 1.0 - CsmShadowStrength * (1.0 - VirtualVisibility(position, bias, cascades, map0, map1));
    }
    float depth = dot(cascades.EyeDepthPlane, float4(position, 1.0));
    uint count = uint(cascades.Parameters.z);
    if (count == 0 || depth < cascades.Parameters.x || depth > cascades.Parameters.y) return 1.0;
    uint selected = 0;
    for (uint index = 0; index + 1 < CsmCascadeLimit; ++index)
        if (index + 1 < count && depth > cascades.FarDepths[index]) selected = index + 1;
    float lengthSquared = dot(directionToLight, directionToLight);
    float3 direction = lengthSquared > CsmDirectionEpsilon ? directionToLight * rsqrt(lengthSquared) : normal;
    float cosine = saturate(dot(normal, direction));
    float slope = sqrt(max(1.0 - cosine * cosine, 0.0)) / max(cosine, CsmMinimumCosine);
    float bias = min(CsmConstantBias + CsmSlopeBias * slope, CsmMaximumBias);
    float visibility = CsmSample(selected, position, bias, cascades, map0, map1, map2, map3);
    float overlap = cascades.FarDepths[selected] - cascades.BlendDepths[selected];
    if (overlap > CsmOverlapEpsilon && depth > cascades.BlendDepths[selected])
    {
        float nextVisibility = selected + 1 < count
            ? CsmSample(selected + 1, position, bias, cascades, map0, map1, map2, map3) : 1.0;
        visibility = mix(visibility, nextVisibility, saturate((depth - cascades.BlendDepths[selected]) / overlap));
    }
    return visibility;
}

float SampleShadow(
    VertexOutput input,
    constant DrawConstants& constants,
    depth2d<float, access::read> shadowMap,
    float4 filtering,
    float3 surfaceNormal,
    float3 directionToLight)
{
    if (constants.RenderParameters.x < 0.5 ||
        abs(input.LightPosition.w) < HomogeneousEpsilon)
        return 1.0;

    const float3 projected =
        input.LightPosition.xyz / input.LightPosition.w;
    const float2 uv =
        float2(projected.x * 0.5 + 0.5, -projected.y * 0.5 + 0.5);
    if (!all(uv >= 0.0) || !all(uv <= 1.0) ||
        projected.z < 0.0 || projected.z > 1.0)
        return 1.0;

    const uint width = shadowMap.get_width();
    const uint height = shadowMap.get_height();
    if (width == 0 || height == 0) return 1.0;
    float directionLengthSquared = dot(directionToLight, directionToLight);
    float3 direction = directionLengthSquared > NormalizeEpsilon
        ? directionToLight * rsqrt(directionLengthSquared)
        : surfaceNormal;
    float normalToLight = saturate(dot(surfaceNormal, direction));
    float sineToLight = sqrt(max(1.0 - normalToLight * normalToLight, 0.0));
    float slope = sineToLight / max(normalToLight, MinimumShadowCosine);
    float receiverBias = min(MaximumReceiverBias, ReceiverBias + ReceiverSlopeBias * slope);
    // Hard mode compares a single nearest texel without bilinear mixing.
    if (filtering.x == 0.0)
    {
        int2 pixel = clamp(int2(floor(uv * float2(width, height))), int2(0), int2(width, height) - 1);
        return 1.0 - ShadowStrength * (projected.z - receiverBias > shadowMap.read(uint2(pixel)) ? 1.0 : 0.0);
    }
    float2 texelPosition = uv * float2(width, height) - 0.5;
    float2 interpolation = fract(texelPosition);
    int2 basePixel = int2(floor(texelPosition));
    float occlusion = 0.0;
    for (int y = -int(filtering.x); y <= int(filtering.x); ++y)
    {
        for (int x = -int(filtering.x); x <= int(filtering.x); ++x)
        {
            int2 pixel00 = clamp(
                basePixel + int2(x, y),
                int2(0),
                int2(width - 1, height - 1));
            int2 pixel10 = clamp(
                pixel00 + int2(1, 0),
                int2(0),
                int2(width - 1, height - 1));
            int2 pixel01 = clamp(
                pixel00 + int2(0, 1),
                int2(0),
                int2(width - 1, height - 1));
            int2 pixel11 = clamp(
                pixel00 + int2(1, 1),
                int2(0),
                int2(width - 1, height - 1));
            float comparisonDepth = projected.z - receiverBias;
            float comparison00 = comparisonDepth >
                shadowMap.read(uint2(pixel00)) ? 1.0 : 0.0;
            float comparison10 = comparisonDepth >
                shadowMap.read(uint2(pixel10)) ? 1.0 : 0.0;
            float comparison01 = comparisonDepth >
                shadowMap.read(uint2(pixel01)) ? 1.0 : 0.0;
            float comparison11 = comparisonDepth >
                shadowMap.read(uint2(pixel11)) ? 1.0 : 0.0;
            float comparisonTop =
                mix(comparison00, comparison10, interpolation.x);
            float comparisonBottom =
                mix(comparison01, comparison11, interpolation.x);
            occlusion +=
                mix(comparisonTop, comparisonBottom, interpolation.y);
        }
    }
    return 1.0 - ShadowStrength * occlusion / filtering.y;
}

float3 ResolveSurfaceNormal(
    VertexOutput input,
    constant DrawConstants& constants,
    texture2d<float> normalMap,
    sampler surfaceSampler,
    bool frontFacing)
{
    float3 normal = NormalizeSafe(input.Normal, float3(0.0, 1.0, 0.0));
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
    return NormalizeSafe(
        tangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

void AccumulateLightSample(float2 roughnessMetallic,
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
    float3 baseReflectance = mix(constants.MaterialSpecularColor.www, saturate(albedo), metallic);
    const float SchlickExponent = 5.0;
    float3 fresnel = baseReflectance + (1.0 - baseReflectance) * pow(1.0 - vDotH, SchlickExponent);
    const float MicrofacetNormalization = 4.0;
    float3 specular = fresnel * distribution * masking * shadowing /
        max(MicrofacetNormalization * nDotV * normalToLight, BrdfEpsilon);
    diffuseLight += visibleRadiance * (1.0 - fresnel) * (1.0 - metallic) * normalToLight / Pi;
    specularLight += visibleRadiance * specular * normalToLight;
}

void AccumulateSceneLight(float2 roughnessMetallic,
    uint lightIndex,
    VertexOutput input,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    constant DrawConstants& constants,
    depth2d<float, access::read> shadowMap,
    float4 filtering,
    bool useCascades, float cascadeVisibility, float contactVisibility,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    LightConstants light = constants.Lights[lightIndex];
    uint type = uint(light.PositionType.w + 0.5);
    float3 shadowDirection = type == DirectionalLight
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - input.WorldPosition;
    bool isShadowLight =
        int(lightIndex) == int(constants.RenderParameters.z) &&
        light.ColorCastShadows.w > 0.5;
    float visibility = isShadowLight
        ? (useCascades ? cascadeVisibility : SampleShadow(
            input, constants, shadowMap, filtering, normal, shadowDirection))
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
    for (int sampleY = 0; sampleY < AreaGridSize; ++sampleY)
    {
        for (int sampleX = 0; sampleX < AreaGridSize; ++sampleX)
        {
            float offsetX = sampleX == 0 ? -AreaSampleWeight : AreaSampleWeight;
            float offsetY = sampleY == 0 ? -AreaSampleWeight : AreaSampleWeight;
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
                radiance * (AreaSampleWeight * emitterCosine * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared),
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
float4 EvaluateMaterialSurface(
    VertexOutput input,
    constant DrawConstants& constants,
    constant CascadeConstants& cascades,
    constant SurfaceFogParameters& fog,
    texture2d<float> fogAtlas,
    constant LumenVolumeParameters& lumenVolume, texture2d<float> lumenAtlas,
    depth2d<float, access::read> cascade0,
    depth2d<float, access::read> cascade1,
    depth2d<float, access::read> cascade2,
    depth2d<float, access::read> cascade3,
    depth2d<float, access::read> shadowMap,
    texture2d<float> diffuseMap,
    texture2d<float> normalMap,
    texture2d<float> opacityMap,
    sampler surfaceSampler,
    bool frontFacing, float contactVisibility = 1.0)
{
    float2 roughnessMetallic = constants.ShadingParameters.xy;
    if (constants.TextureFlags.z > 1) roughnessMetallic *= opacityMap.sample(surfaceSampler, input.TexCoord).gb;
    float4 diffuseSample =
        SampleDiffuse(input, constants, diffuseMap, surfaceSampler);
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
    float3 surfaceBase =
        input.Color.rgb *
        max(constants.PipelineColor.rgb, 0.0) *
        max(diffuseSample.rgb, 0.0);
    float3 normal = ResolveSurfaceNormal(
        input, constants, normalMap, surfaceSampler, frontFacing);
    float3 viewDirection = NormalizeSafe(
        constants.CameraLightCount.xyz - input.WorldPosition,
        normal);
    float3 diffuseLight = 0.0;
    float3 specularLight = 0.0;
    uint lightCount = min(uint(constants.CameraLightCount.w + 0.5), MaximumLights);
    bool useCascades = cascades.Parameters.z > 0.0 && constants.RenderParameters.x >= 0.5;
    float cascadeVisibility = 1.0;
    int shadowIndex = int(constants.RenderParameters.z);
    if (useCascades && shadowIndex >= 0 && shadowIndex < int(lightCount))
        cascadeVisibility = CsmVisibility(input.WorldPosition, normal,
            constants.Lights[shadowIndex].DirectionIntensity.xyz, cascades, cascade0, cascade1, cascade2, cascade3);
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        AccumulateSceneLight(roughnessMetallic,
            lightIndex, input, normal, viewDirection,
            surfaceBase * max(constants.MaterialDiffuseColor.rgb, 0.0), constants, shadowMap, cascades.Filtering,
            useCascades, cascadeVisibility, contactVisibility, diffuseLight, specularLight);
    }
    float3 color =
        surfaceBase *
            max(constants.MaterialAmbientColor.rgb, 0.0) *
            max(constants.AmbientColorStrength.rgb, 0.0) *
            max(constants.AmbientColorStrength.w, 0.0) * (1.0 - saturate(roughnessMetallic.y)) +
        surfaceBase *
            max(constants.MaterialDiffuseColor.rgb, 0.0) *
            diffuseLight +
        specularLight +
        max(constants.MaterialEmissiveColor.rgb, 0.0);
    color += surfaceBase * max(constants.MaterialDiffuseColor.rgb,0.0) * (1-LumenDielectricReflectance) * (1.0-saturate(roughnessMetallic.y)) * LumenTranslucentIrradiance(input.WorldPosition,normal,lumenVolume,lumenAtlas);
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
    constant CascadeConstants& cascades [[buffer(2)]],
    constant SurfaceFogParameters& fog [[buffer(4)]],
    texture2d<float> fogAtlas [[texture(14)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(9)]],
    texture2d<float> lumenAtlas [[texture(19)]],
    depth2d<float, access::read> cascade0 [[texture(10)]],
    depth2d<float, access::read> cascade1 [[texture(11)]],
    depth2d<float, access::read> cascade2 [[texture(12)]],
    depth2d<float, access::read> cascade3 [[texture(13)]],
    depth2d<float, access::read> shadowMap [[texture(0)]],
    texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]],
    bool frontFacing [[front_facing]])
{
    float contactVisibility = 1.0;
#if EVANS_CONTACT_SHADOWS
#if EVANS_CONTACT_MULTISAMPLE
    contactVisibility = contactMask.read(uint2(input.Position.xy), contactSample).x;
#else
    contactVisibility = contactMask.read(uint2(input.Position.xy), 0).x;
#endif
#endif
    return EvaluateMaterialSurface(input, constants, cascades, fog, fogAtlas, lumenVolume, lumenAtlas,
        cascade0, cascade1, cascade2, cascade3, shadowMap,
        diffuseMap, normalMap, opacityMap, surfaceSampler, frontFacing, contactVisibility);
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
    constant CascadeConstants& cascades [[buffer(2)]],
    constant SurfaceFogParameters& fog [[buffer(4)]],
    texture2d<float> fogAtlas [[texture(14)]],
    constant LumenVolumeParameters& lumenVolume [[buffer(9)]],
    texture2d<float> lumenAtlas [[texture(19)]],
    depth2d<float, access::read> cascade0 [[texture(10)]],
    depth2d<float, access::read> cascade1 [[texture(11)]],
    depth2d<float, access::read> cascade2 [[texture(12)]],
    depth2d<float, access::read> cascade3 [[texture(13)]],
    depth2d<float, access::read> shadowMap [[texture(0)]],
    texture2d<float> diffuseMap [[texture(1)]],
    texture2d<float> normalMap [[texture(2)]],
    texture2d<float> opacityMap [[texture(3)]],
    sampler surfaceSampler [[sampler(0)]],
    constant WeightedCoverageParameters& transparency [[buffer(5)]],
    bool frontFacing [[front_facing]])
{
    // Preserve this model's lighting and fragment-depth fog before accumulation.
    float4 surface = EvaluateMaterialSurface(input, constants, cascades, fog, fogAtlas, lumenVolume, lumenAtlas,
        cascade0, cascade1, cascade2, cascade3, shadowMap,
        diffuseMap, normalMap, opacityMap, surfaceSampler, frontFacing);
    float coverage = clamp(surface.a, 0.0, 1.0);
    WeightedSurfaceOutput output {float4(0.0), coverage};
    if (coverage == 0.0) return output;
    float depthFactor = 1.0 - clamp(input.Position.z, 0.0, 1.0) * transparency.Weights.z;
    float contribution = coverage * max(pow(coverage, transparency.Weights.x) *
        pow(depthFactor, transparency.Weights.y), transparency.Weights.w);
    output.RadianceSum = float4(surface.rgb * contribution, contribution);
    return output;
}
