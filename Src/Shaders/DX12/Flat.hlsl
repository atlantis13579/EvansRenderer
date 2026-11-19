// Native D3D12 translucent Lumen receiver. The atlas contains signed, cosine-
// convolved SH; division by pi below applies the diffuse BRDF exactly once.
cbuffer LumenTranslucencyParameters : register(b9)
{
    row_major float4x4 TranslucentCamera,TranslucentInverse;
    uint4 TranslucentCells;
    float4 TranslucentDepthScale,TranslucentImageSize;
};
Texture2D<float4> TranslucentSH : register(t19);
SamplerState TranslucentFilter : register(s2);
static const float LumenDielectricReflectance=0.04;
float3 LumenTranslucentIrradiance(float3 position,float3 n)
{
    if (TranslucentCells.w==0) return 0;
    float4 homogeneous=mul(TranslucentCamera,float4(position,1));
    if (homogeneous.w<=TranslucentImageSize.z) return 0;
    float3 screen=homogeneous.xyz/homogeneous.w;
    if (screen.z<0 || screen.z>1 || any(abs(screen.xy)>1)) return 0;
    float4 originH=mul(TranslucentInverse,float4(screen.xy,0,1)),endH=mul(TranslucentInverse,float4(screen.xy,1,1));
    if (min(abs(originH.w),abs(endH.w))<=TranslucentImageSize.z) return 0;
    float3 origin=originH.xyz/originH.w;
    float span=min(length(endH.xyz/endH.w-origin),TranslucentDepthScale.x),travel=length(position-origin);
    if (span<=TranslucentImageSize.z || travel>span) return 0;
    float2 screenUV=float2(screen.x*0.5+0.5,0.5-screen.y*0.5);
    float3 location=float3(screenUV*TranslucentCells.xy,pow(saturate(travel/span),rcp(TranslucentDepthScale.y))*TranslucentCells.z)-0.5;
    location=clamp(location,0,float3(TranslucentCells.xyz)-1);
    float z0=floor(location.z),z1=min(z0+1,float(TranslucentCells.z-1));
    n*=rsqrt(max(dot(n,n),TranslucentImageSize.w));
    static const float SHPi=3.14159265358979323846;
    static const uint SHCount=9;
    float sh[SHCount]={sqrt(1/(4*SHPi)),sqrt(3/(4*SHPi))*n.y,sqrt(3/(4*SHPi))*n.z,sqrt(3/(4*SHPi))*n.x,
        sqrt(15/(4*SHPi))*n.x*n.y,sqrt(15/(4*SHPi))*n.y*n.z,sqrt(5/(16*SHPi))*(3*n.z*n.z-1),
        sqrt(15/(4*SHPi))*n.x*n.z,sqrt(15/(16*SHPi))*(n.x*n.x-n.y*n.y)};
    float3 result=0;

    static const uint VolumeVisibilityBit=2;
    if ((TranslucentCells.w&VolumeVisibilityBit)!=0)
    {
        uint3 base=uint3(floor(location)); float3 fraction=frac(location);
        // Unsupported corners keep zero weight in the original kernel; a
        // surviving bright corner must not acquire the missing corner's mass.
        [loop] for (uint corner=0;corner<8;++corner)
        {
            uint3 bit=uint3(corner&1,(corner>>1)&1,corner>>2),p=min(base+bit,TranslucentCells.xyz-1);
            float weight=(bit.x?fraction.x:1-fraction.x)*(bit.y?fraction.y:1-fraction.y)*(bit.z?fraction.z:1-fraction.z); if (weight<=0) continue;
            uint row=p.z*TranslucentCells.y+p.y; float radius=TranslucentSH.Load(int3(p.x,row,0)).w; if (radius<=0) continue;
            float2 clipXY=float2(2*(float(p.x)+0.5)/TranslucentCells.x-1,1-2*(float(p.y)+0.5)/TranslucentCells.y);
            float4 a=mul(TranslucentInverse,float4(clipXY,0,1)),b=mul(TranslucentInverse,float4(clipXY,1,1));
            if (min(abs(a.w),abs(b.w))<=TranslucentImageSize.z) continue;
            float3 start=a.xyz/a.w,axis=b.xyz/b.w-start; float spanToFar=length(axis);
            if (!isfinite(spanToFar) || spanToFar<=TranslucentImageSize.z) continue;
            float depth=pow((float(p.z)+0.5)/TranslucentCells.z,TranslucentDepthScale.y);
            float3 probe=start+axis*(depth*min(spanToFar,TranslucentDepthScale.x)/spanToFar);
            float separation=length(position-probe); if (!isfinite(separation) || separation>radius) continue;
            [unroll] for (uint band=0;band<SHCount;++band)
                result+=weight*sh[band]*TranslucentSH.Load(int3(p.x+band*TranslucentCells.x,row,0)).xyz;
        }
        return max(result,0)*TranslucentDepthScale.z/(SHPi*TranslucentDepthScale.w);
    }
    [unroll] for (uint band=0;band<SHCount;++band)
    {
        float column=band*TranslucentCells.x+location.x+0.5;
        float2 low=float2(column,z0*TranslucentCells.y+location.y+0.5)/TranslucentImageSize.xy;
        float2 high=float2(column,z1*TranslucentCells.y+location.y+0.5)/TranslucentImageSize.xy;
        result+=sh[band]*lerp(TranslucentSH.SampleLevel(TranslucentFilter,low,0).xyz,
            TranslucentSH.SampleLevel(TranslucentFilter,high,0).xyz,location.z-z0);
    }
    return max(result,0)*TranslucentDepthScale.z/(SHPi*TranslucentDepthScale.w);
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

// The independent contact producer supplies raw visibility for this raster sample.
// Only the selected direct light consumes it. Primary map filters already apply
// artistic strength, so the composition below applies that strength just once.
#if defined(EVANS_CONTACT_RECEIVER)
cbuffer ContactReceiver : register(b7) { uint ContactEnabled; };
Texture2DArray<float> ContactImage : register(t15);
#endif
float ReadContactVisibility(float4 pixel, uint sampleIndex)
{
#if defined(EVANS_CONTACT_RECEIVER)
    if (ContactEnabled != 0) return ContactImage.Load(int4(int2(pixel.xy), sampleIndex, 0));
#endif
    return 1.0f;
}
float CombineContactVisibility(float primary, float contact, float strength)
{
    return lerp(1.0f - strength, primary, contact);
}

// DX12 Flat shading: standalone native HLSL, with forward material prerequisites.
// Geometric face normals expose the mesh facets; tangent-space normal mapping is inactive.
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
    if (RenderParameters.y > OutlineModeThreshold)
    {
        worldPosition.xyz += worldNormal * (MaterialParameters.w * OutlineWorldScale);
    }
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

// CSM: positive eye-depth intervals choose one map, or two in their overlap.
// The terminal interval blends toward an unshadowed receiver. Separate resource
// declarations match independent DX12 descriptor tables at t10 through t13.
static const uint SplitCapacity = 4u;

static const float SplitDarkening = 0.55f;
static const float SplitBiasBase = 0.00035f;
static const float SplitBiasSlope = 0.00060f;
static const float SplitBiasLimit = 0.0045f;
static const float SplitCosineFloor = 0.15f;
static const float SplitVectorEpsilon = 1.0e-12f;
cbuffer CascadeUniforms : register(b2)
{
    row_major float4x4 SplitToLight[SplitCapacity];
    float4 SplitEndDistances;
    float4 SplitFadeDistances;
    float4 CameraDistancePlane;
    float4 SplitRange; // near, far, count, map resolution
    float4 ShadowFiltering; // radius (zero means hard), comparison count, reserved
};
Texture2D<float> SplitDepth0 : register(t10);
Texture2D<float> SplitDepth1 : register(t11);
Texture2D<float> SplitDepth2 : register(t12);
Texture2D<float> SplitDepth3 : register(t13);
float SplitDepthAt(uint index, int2 coordinate)
{
    if (index == 0u) return SplitDepth0.Load(int3(coordinate, 0));
    if (index == 1u) return SplitDepth1.Load(int3(coordinate, 0));
    if (index == 2u) return SplitDepth2.Load(int3(coordinate, 0));
    return SplitDepth3.Load(int3(coordinate, 0));
}
float SplitComparison(uint index, int2 coordinate, float receiver)
{
    int2 limit = int2(SplitRange.ww) - 1;
    return receiver > SplitDepthAt(index, clamp(coordinate, 0, limit)) ? 1.0f : 0.0f;
}
float SplitVisibility(uint index, float3 position, float bias)
{
    // Fitting produces affine light matrices with homogeneous W equal to one.
    float3 light = mul(SplitToLight[index], float4(position, 1.0f)).xyz;
    float2 texcoord = float2(light.x + 1.0f, 1.0f - light.y) * 0.5f;
    if (light.z < 0.0f || light.z > 1.0f || any(texcoord < 0.0f) || any(texcoord > 1.0f)) return 1.0f;
    if (ShadowFiltering.x == 0.0f)
        return 1.0f - SplitDarkening * SplitComparison(index, int2(floor(texcoord * SplitRange.w)), light.z - bias);
    float2 texel = texcoord * SplitRange.w - 0.5f;
    int2 origin = int2(floor(texel));
    float2 interpolation = frac(texel);
    float receiver = light.z - bias;
    float occluded = 0.0f;
    // Compact loops avoid expanding four resource-selection branches per tap.
    [loop] for (int row = -int(ShadowFiltering.x); row <= int(ShadowFiltering.x); ++row)
        [loop] for (int column = -int(ShadowFiltering.x); column <= int(ShadowFiltering.x); ++column)
        {
            int2 pixel = origin + int2(column, row);
            float top = lerp(SplitComparison(index, pixel, receiver),
                SplitComparison(index, pixel + int2(1, 0), receiver), interpolation.x);
            float bottom = lerp(SplitComparison(index, pixel + int2(0, 1), receiver),
                SplitComparison(index, pixel + 1, receiver), interpolation.x);
            occluded += lerp(top, bottom, interpolation.y);
        }
    return 1.0f - occluded * (1.0f / ShadowFiltering.y) * SplitDarkening;
}
// Virtual shadows share the cascade receiver slots. End distances describe the
// logical hierarchy; fade distances describe the physical atlas and directory.
// Point reads plus per-corner indirection keep unrelated physical tiles apart.
static const uint VirtualGutter = 1;
static const float VirtualPixelCenter = 0.5f;
float2 VirtualStoredDepth(float2 coordinate, uint minimumLevel)
{
    uint4 hierarchy = uint4(SplitEndDistances), atlas = uint4(SplitFadeDistances);
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
            uint encodedSlot = uint(round(SplitDepth1.Load(int3(tablePixel, 0)).x * atlas.w));
            if (encodedSlot > 0)
            {
                uint slot = encodedSlot - 1;
                uint2 origin = uint2(slot % atlas.z, slot / atlas.z) * (hierarchy.y + 2 * VirtualGutter);
                uint2 location = origin + VirtualGutter + texel % hierarchy.y;
                return float2(SplitDepth0.Load(int3(location, 0)).x, level);
            }
        }
        levelOffset += rowPages * rowPages; rowPages /= 2;
    }
    return float2(1, hierarchy.z - 1);
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
    float resolution = float(uint(SplitEndDistances.x) >> actualMip);
    float2 pixel = clamp(floor(sampleUv * resolution), float2(0.0, 0.0), float2(resolution - 1.0, resolution - 1.0));
    float2 center = (pixel + float2(0.5, 0.5)) / resolution;
    // Depth one is also the empty atlas value; extrapolation beyond the far plane must remain lit.
    return min(1.0, receiver + dot(center - receiverUv, gradient));
}
float VirtualComparison(int2 pixel, uint mip, float2 receiverUv, float receiver, float2 gradient)
{
    uint resolution = uint(SplitEndDistances.x) >> mip;
    float2 uv = (float2(clamp(pixel, int2(0, 0), int2(int(resolution) - 1, int(resolution) - 1))) + float2(0.5, 0.5)) / float(resolution);
    float2 stored = VirtualStoredDepth(uv, mip);
    return VirtualReceiverAt(uv, uint(stored.y), receiverUv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
}
float VirtualVisibility(float3 position, float bias)
{
    float4 light = mul(SplitToLight[0], float4(position, 1.0));
    float3 projected = light.xyz / max(light.w, CameraDistancePlane.w);
    float2 uv = float2(projected.x, -projected.y) * 0.5 + float2(0.5, 0.5);
    float2 gradient = VirtualReceiverGradient(uv, projected.z);
    float footprint = max(length(ddx(uv) * CameraDistancePlane.y), length(ddy(uv) * CameraDistancePlane.z)) * SplitEndDistances.x;
    uint mip = uint(clamp(floor(log2(max(1.0, footprint)) + CameraDistancePlane.x), 0.0, SplitEndDistances.z - 1.0));
    if (light.w <= CameraDistancePlane.w || any(uv < 0.0) || any(uv > 1.0) || projected.z < 0.0 || projected.z > 1.0) return 1.0;
    float2 stored = VirtualStoredDepth(uv, mip);
    float receiver = projected.z - bias;
    if (ShadowFiltering.x == 0.0)
        return VirtualReceiverAt(uv, uint(stored.y), uv, receiver, gradient) <= stored.x ? 1.0 : 0.0;
    mip = uint(stored.y);
    float2 texel = uv * float(uint(SplitEndDistances.x) >> mip) - float2(0.5, 0.5);
    int2 first = int2(floor(texel)); float2 weight = frac(texel); float visible = 0.0;
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 p = first + int2(x, y);
            float a = VirtualComparison(p, mip, uv, receiver, gradient);
            float b = VirtualComparison(p + int2(1, 0), mip, uv, receiver, gradient);
            float c = VirtualComparison(p + int2(0, 1), mip, uv, receiver, gradient);
            float d = VirtualComparison(p + int2(1, 1), mip, uv, receiver, gradient);
            visible += lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
        }
    return visible / ShadowFiltering.y;
}


float CascadedReceiverVisibility(float3 position, float3 normal, float3 towardLight)
{
    if (ShadowFiltering.z != 0)
    {
        float cosine = saturate(dot(normal, towardLight * rsqrt(max(dot(towardLight, towardLight), SplitVectorEpsilon))));
        float slope = sqrt(max(1 - cosine * cosine, 0)) / max(cosine, SplitCosineFloor);
        float bias = min(SplitBiasLimit, SplitBiasBase + SplitBiasSlope * slope);
        return 1 - SplitDarkening * (1 - VirtualVisibility(position, bias));
    }
    float eyeDistance = dot(CameraDistancePlane, float4(position, 1.0f));
    if (eyeDistance < SplitRange.x || eyeDistance > SplitRange.y) return 1.0f;
    float cosine = saturate(dot(normal, towardLight * rsqrt(max(dot(towardLight, towardLight), SplitVectorEpsilon))));
    float slope = sqrt(max(1.0f - cosine * cosine, 0.0f)) / max(cosine, SplitCosineFloor);
    float bias = min(SplitBiasLimit, SplitBiasBase + SplitBiasSlope * slope);
    uint count = uint(SplitRange.z);
    [loop] for (uint index = 0u; index < count; ++index)
    {
        if (eyeDistance > SplitEndDistances[index]) continue;
        float visibility = SplitVisibility(index, position, bias);
        float overlap = SplitEndDistances[index] - SplitFadeDistances[index];
        if (overlap > 0.0f && eyeDistance >= SplitFadeDistances[index])
        {
            float nextVisibility = 1.0f;
            if (index + 1u < count) nextVisibility = SplitVisibility(index + 1u, position, bias);
            float weight = saturate((eyeDistance - SplitFadeDistances[index]) / overlap);
            visibility = lerp(visibility, nextVisibility, weight);
        }
        return visibility;
    }
    return 1.0f;
}

float SampleShadow(
    float3 receiverWorld,
    float4 lightPosition,
    float3 surfaceNormal,
    float3 directionToLight)
{
    if (RenderParameters.x >= 0.5f && SplitRange.z > 0.0f)
        return CascadedReceiverVisibility(receiverWorld, surfaceNormal, directionToLight);
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
    if (width == 0 || height == 0) return 1.0f;

    float directionLengthSquared = dot(directionToLight, directionToLight);
    float3 direction = directionLengthSquared > NormalLengthEpsilon
        ? directionToLight * rsqrt(directionLengthSquared)
        : surfaceNormal;
    float normalToLight = saturate(dot(surfaceNormal, direction));
    float sineToLight = sqrt(max(1.0f - normalToLight * normalToLight, 0.0f));
    float slope = sineToLight / max(normalToLight, ShadowMinimumCosine);
    float receiverBias = min(ShadowMaximumBias, ShadowConstantBias + ShadowSlopeBias * slope);
    // Hard shadows bypass the bilinear comparison kernel.
    if (ShadowFiltering.x == 0.0f)
    {
        int2 pixel = clamp(int2(floor(uv * float2(width, height))), 0, int2(width, height) - 1);
        return 1.0f - ShadowStrength * ((projected.z - receiverBias) > ShadowMap.Load(int3(pixel, 0)) ? 1.0f : 0.0f);
    }
    float2 texelPosition = uv * float2(width, height) - 0.5f;
    float2 interpolation = frac(texelPosition);
    int2 basePixel = int2(floor(texelPosition));
    float occlusion = 0.0f;
    [loop]
    for (int y = -int(ShadowFiltering.x); y <= int(ShadowFiltering.x); ++y)
    {
        [loop]
        for (int x = -int(ShadowFiltering.x); x <= int(ShadowFiltering.x); ++x)
        {
            int2 pixel00 = clamp(basePixel + int2(x, y), int2(0, 0), int2(width - 1, height - 1));
            int2 pixel10 = clamp(pixel00 + int2(1, 0), int2(0, 0), int2(width - 1, height - 1));
            int2 pixel01 = clamp(pixel00 + int2(0, 1), int2(0, 0), int2(width - 1, height - 1));
            int2 pixel11 = clamp(pixel00 + int2(1, 1), int2(0, 0), int2(width - 1, height - 1));
            float comparisonDepth = projected.z - receiverBias;
            float comparison00 = comparisonDepth > ShadowMap.Load(int3(pixel00, 0)) ? 1.0f : 0.0f;
            float comparison10 = comparisonDepth > ShadowMap.Load(int3(pixel10, 0)) ? 1.0f : 0.0f;
            float comparison01 = comparisonDepth > ShadowMap.Load(int3(pixel01, 0)) ? 1.0f : 0.0f;
            float comparison11 = comparisonDepth > ShadowMap.Load(int3(pixel11, 0)) ? 1.0f : 0.0f;
            float comparisonTop = lerp(comparison00, comparison10, interpolation.x);
            float comparisonBottom = lerp(comparison01, comparison11, interpolation.x);
            occlusion += lerp(comparisonTop, comparisonBottom, interpolation.y);
        }
    }
    return 1.0f - ShadowStrength * occlusion / ShadowFiltering.y;
}

float3 ResolveSurfaceNormal(VSOutput input, bool isFrontFace)
{
    // Derivatives recover one geometric normal for each triangle. Align it
    // with the authored normal before applying the two-sided material rule.
    float3 geometricNormal = NormalizeSafe(
        cross(ddx(input.WorldPosition), ddy(input.WorldPosition)), input.Normal);
    if (dot(geometricNormal, input.Normal) < 0.0f) geometricNormal = -geometricNormal;
    return isFrontFace ? geometricNormal : -geometricNormal;
}

void AccumulateLightSample(
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    inout float3 diffuseLight,
    inout float3 specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= NormalLengthEpsilon) return;
    float3 direction = lightDirection * rsqrt(lengthSquared);
    float normalToLight = max(dot(normal, direction), 0.0f);
    if (normalToLight <= 0.0f) return;
    float3 visibleRadiance = radiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    float3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0f),
        max(MaterialParameters.x, 1.0f));
    specularLight += visibleRadiance * specularAmount;
}

void AccumulateSceneLight(
    uint lightIndex,
    VSOutput input,
    float3 normal,
    float3 viewDirection,
    float contactVisibility,
    inout float3 diffuseLight,
    inout float3 specularLight)
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
        ? SampleShadow(input.WorldPosition, input.LightPosition, normal, shadowDirection)
        : 1.0f;
    if (isShadowLight) visibility = CombineContactVisibility(visibility, contactVisibility, ShadowStrength);
    float3 radiance =
        max(light.ColorCastShadows.rgb, 0.0f) *
        max(light.DirectionIntensity.w, 0.0f);
    if (type == DirectionalLight)
    {
        AccumulateLightSample(
            light.DirectionIntensity.xyz, radiance, visibility,
            normal, viewDirection, diffuseLight, specularLight);
        return;
    }
    if (type == PointLight)
    {
        float3 toLight = light.PositionType.xyz - input.WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), LightDistanceEpsilon);
        AccumulateLightSample(
            toLight, radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared, visibility,
            normal, viewDirection, diffuseLight, specularLight);
        return;
    }

    float3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz, float3(0.0f, -1.0f, 0.0f));
    float3 helper = abs(emissionDirection.y) < ParallelAxisLimit
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = NormalizeSafe(
        cross(helper, emissionDirection), float3(1.0f, 0.0f, 0.0f));
    float3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent), float3(0.0f, 0.0f, 1.0f));
    [unroll]
    for (int sampleY = 0; sampleY < AreaSamplesPerAxis; ++sampleY)
    {
        [unroll]
        for (int sampleX = 0; sampleX < AreaSamplesPerAxis; ++sampleX)
        {
            float offsetX = sampleX == 0 ? -AreaSampleOffset : AreaSampleOffset;
            float offsetY = sampleY == 0 ? -AreaSampleOffset : AreaSampleOffset;
            float3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            float3 toLight = samplePosition - input.WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), LightDistanceEpsilon);
            float3 direction = toLight * rsqrt(distanceSquared);
            float emitterCosine = max(dot(emissionDirection, -direction), 0.0f);
            AccumulateLightSample(
                direction,
                radiance * (AreaSampleWeight * emitterCosine * LightRangeWindow(toLight, light.AreaSizePadding.z) / distanceSquared),
                visibility, normal, viewDirection, diffuseLight, specularLight);
        }
    }
}

// The surface reads a generated cumulative transport volume at its own depth.
// Integration belongs to the standalone VolumetricFog technique. Keep this
// small consumer local so shading sources remain independently compilable.
cbuffer SurfaceVolumeParameters : register(b4)
{
    row_major float4x4 VolumeInverseCamera;
    float4 VolumeMedium;
    float4 VolumeAmbientSource;
    float4 VolumeDimensions;
    uint4 VolumeSettings;
};
Texture2D<float4> SurfaceVolume : register(t14);
SamplerState SurfaceVolumeSampler : register(s1);
static const float VolumeDistanceFloor = 1.0e-6f;
static const float VolumeHalfFloatLimit = 65504.0f;

float3 VolumePoint(float2 ndcXY, float depth)
{
    float4 homogeneous = mul(VolumeInverseCamera, float4(ndcXY, depth, 1));
    return homogeneous.xyz / homogeneous.w;
}
// Paired transport planes keep colored extinction separate from source radiance.
static const uint RgbTransportMode = 2;
float4 VolumeBoundaryTransmission(float2 rayUV, uint endpoint)
{
    if (endpoint == 0) return float4(1, 1, 1, 1);
    float2 cell = clamp(rayUV * VolumeDimensions.zw, 0.5f, VolumeDimensions.zw - 0.5f);
    cell.x += (endpoint - 1) * VolumeDimensions.z;
    float2 extent = float2(VolumeDimensions.z * VolumeSettings.y, VolumeDimensions.w);
    cell.y = (cell.y + extent.y) * 0.5;
    return SurfaceVolume.SampleLevel(SurfaceVolumeSampler, cell / extent, 0);
}
float4 VolumeBoundary(float2 rayUV, uint endpoint)
{
    if (endpoint == 0) return float4(0, 0, 0, 1);
    float2 cell = clamp(rayUV * VolumeDimensions.zw, 0.5f, VolumeDimensions.zw - 0.5f);
    cell.x += (endpoint - 1) * VolumeDimensions.z;
    float2 extent = float2(VolumeDimensions.z * VolumeSettings.y, VolumeDimensions.w);
    if (VolumeSettings.w == RgbTransportMode) cell.y *= 0.5;
    return SurfaceVolume.SampleLevel(SurfaceVolumeSampler, cell / extent, 0);
}
float3 TransportSurfaceRadiance(float3 incoming, float4 pixelPosition)
{
    if (VolumeSettings.w == 0) return incoming;
    float2 rayUV = pixelPosition.xy / VolumeDimensions.xy;
    float2 ndcXY = float2(rayUV.x * 2 - 1, 1 - rayUV.y * 2);
    float3 start = VolumePoint(ndcXY, 0);
    float fullDistance = min(length(VolumePoint(ndcXY, 1) - start), VolumeMedium.w);
    float fraction = saturate(length(VolumePoint(ndcXY, pixelPosition.z) - start) /
        max(fullDistance, VolumeDistanceFloor));
    float continuousBoundary = fraction * VolumeSettings.y;
    uint boundary = (uint)continuousBoundary;
    float4 integrated = lerp(VolumeBoundary(rayUV, boundary),
        VolumeBoundary(rayUV, min(boundary + 1, VolumeSettings.y)), continuousBoundary - boundary);
    float3 mediumTransmission = VolumeSettings.w == RgbTransportMode ? (lerp(VolumeBoundaryTransmission(rayUV, boundary),
        VolumeBoundaryTransmission(rayUV, min(boundary + 1, VolumeSettings.y)), continuousBoundary - boundary)).rgb : integrated.aaa;
    return clamp(incoming * mediumTransmission + integrated.rgb, 0, VolumeHalfFloatLimit);
}

#if defined(EVANS_CONTACT_RECEIVER)
float4 PSContact(VSOutput input, bool isFrontFace : SV_IsFrontFace, uint contactSample : SV_SampleIndex) : SV_TARGET
#else
float4 PSMain(VSOutput input, bool isFrontFace : SV_IsFrontFace) : SV_TARGET
#endif
{
#if !defined(EVANS_CONTACT_RECEIVER)
    const uint contactSample = 0;
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
        return float4(TransportSurfaceRadiance(SRGBToLinear(saturate(PipelineColor.rgb)), input.Position), alpha);
    }

    float3 surfaceBase =
        input.Color.rgb *
        max(PipelineColor.rgb, 0.0f) *
        max(diffuseSample.rgb, 0.0f);
    float3 normal = ResolveSurfaceNormal(input, isFrontFace);
    float3 viewDirection = NormalizeSafe(
        CameraLightCount.xyz - input.WorldPosition, normal);
    float3 diffuseLight = 0.0f;
    float3 specularLight = 0.0f;
    uint lightCount = min((uint)(CameraLightCount.w + 0.5f), MaximumLights);
    [loop]
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        AccumulateSceneLight(
            lightIndex, input, normal, viewDirection, ReadContactVisibility(input.Position, contactSample),
            diffuseLight, specularLight);
    }

    float3 color =
        surfaceBase *
            max(MaterialAmbientColor.rgb, 0.0f) *
            max(AmbientColorStrength.rgb, 0.0f) *
            max(AmbientColorStrength.w, 0.0f) +
        surfaceBase * max(MaterialDiffuseColor.rgb, 0.0f) * diffuseLight +
        max(MaterialSpecularColor.rgb, 0.0f) * specularLight +
        max(MaterialEmissiveColor.rgb, 0.0f);
    color += surfaceBase * max(MaterialDiffuseColor.rgb,0) * LumenTranslucentIrradiance(input.WorldPosition,normal);
    return float4(PrepareDisplayLinear(TransportSurfaceRadiance(color, input.Position)), alpha);
}

#if defined(EVANS_CONTACT_RECEIVER)
float4 PSMain(VSOutput fragment, bool frontFacing : SV_IsFrontFace) : SV_TARGET
{ return PSContact(fragment, frontFacing, 0); }
#endif

// Native DX12 weighted accumulation consumes this material's lit, fogged HDR
// result. Additive radiance/opacity and product revealage have distinct MRT
// blend states; the independent WeightedBlendedOIT source resolves them.
cbuffer TransparencyParameters : register(b5)
{
    float4 OpacityAndDepthWeights; // alpha exponent, depth exponent, depth scale, weight floor
    uint4 TransparencyFlags;
};
struct TransparencyFragment
{
    float4 WeightedRadiance : SV_Target0;
    float OpacityForRevealage : SV_Target1;
};
TransparencyFragment PSWeighted(VSOutput fragment, bool frontFacing : SV_IsFrontFace)
{
    float4 shaded = PSMain(fragment, frontFacing);
    float opacity = saturate(shaded.a);
    TransparencyFragment result = (TransparencyFragment)0;
    if (opacity <= 0.0) return result;
    float nearPreference = 1.0 - saturate(fragment.Position.z) * OpacityAndDepthWeights.z;
    float weight = max(pow(opacity, OpacityAndDepthWeights.x) *
        pow(nearPreference, OpacityAndDepthWeights.y), OpacityAndDepthWeights.w);
    float weightedOpacity = weight * opacity;
    result.WeightedRadiance = float4(shaded.rgb * weightedOpacity, weightedOpacity);
    result.OpacityForRevealage = opacity;
    return result;
}

float4 PSOutline(VSOutput input) : SV_TARGET
{
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    ApplyAlphaClip(SurfaceAlpha(input, diffuseSample));
    return float4(SRGBToLinear(saturate(PipelineColor.rgb)), PipelineColor.a);
}
