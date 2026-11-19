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

// DX11 rigid instancing: one indexed draw, an independent transform per object.
// Dudash, "Animated Crowd Rendering", GPU Gems 3, chapter 2 (2007).
// SV_InstanceID addresses a constant array; the base vertex/index stream stays
// resident. These vertex entry points match the existing native pixel-stage
// contracts for surface, G-buffer, shadow and velocity passes. No shader include.
static const uint MaximumInstances = 64u;
static const uint MaximumLights = 4u;
static const uint DirectionalLight = 0u;
static const uint PointLight = 1u;
static const float NormalLengthEpsilon = 1.0e-12f;
static const float ProjectionEpsilon = 1.0e-7f;
static const float LightDistanceEpsilon = 1.0e-6f;
static const float ParallelAxisLimit = 0.999f;
static const int AreaSamplesPerAxis = 2;
static const float AreaSampleOffset = 0.25f;
static const float AreaSampleWeight = 0.25f;
static const float EnabledThreshold = 0.5f;
static const float SrgbLinearScale = 12.92f;
static const float SrgbOffset = 0.055f;
static const float SrgbScale = 1.055f;
static const float SrgbGamma = 2.4f;
static const float SrgbThreshold = 0.04045f;

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
    float4 ShadingParameters;
};
struct InstanceData
{
    row_major float4x4 World;
    row_major float4x4 NormalWorld;
    row_major float4x4 PreviousWorldViewProjection;
    float4 Wind, Gust, VegetationHistory;
};
cbuffer InstanceConstants : register(b3)
{
    row_major float4x4 InstanceViewProjection;
    float4 InstanceMaterial;
    InstanceData Instances[MaximumInstances];
};
struct MeshVertex
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};
float3 SafeNormal(float3 value, float3 fallback)
{
    float squareLength = dot(value, value);
    return squareLength > NormalLengthEpsilon ? value * rsqrt(squareLength) : fallback;
}
float3 DecodeSRGB(float3 color)
{
    float3 low = color / SrgbLinearScale;
    float3 high = pow((color + SrgbOffset) / SrgbScale, SrgbGamma);
    return float3(color.r <= SrgbThreshold ? low.r : high.r,
        color.g <= SrgbThreshold ? low.g : high.g, color.b <= SrgbThreshold ? low.b : high.b);
}
// Spatially coherent gusts travel across patches; a finer wave flutters tips.
// The quadratic profile keeps roots fixed in color, depth, reflection and motion passes.
float3 VegetationOffset(float3 world, float height, float4 wind, float4 gust, float referenceHeight, float time)
{
    float phase=dot(world.xz,wind.xy)*6.2831853/max(gust.x,1.0)-time*1.7;
    float wave=0.55+gust.y*(0.35*sin(phase)+0.22*sin(phase*0.47+1.3));
    float flutter=gust.z*sin(time*7.1+world.x*1.83+world.z*2.17);
    float bend=clamp(wind.w*(wave+flutter)*height*height/max(referenceHeight,0.1),-height*0.6,height*0.6);
    return float3(wind.x*bend,-abs(bend)*0.22,wind.y*bend);
}
float4 VegetationPosition(MeshVertex input, uint instance, bool previous)
{
    InstanceData data=Instances[instance];
    if (data.Gust.w<0.5) return float4(input.Position,1);
    float3 world=mul(data.World,float4(input.Position,1)).xyz;
    float height=max(0.0,data.Gust.w<1.5?input.TexCoord.y:input.Position.y);
    float3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,
        previous?data.VegetationHistory.x:data.Wind.z);
    return float4(input.Position+mul(transpose((float3x3)data.NormalWorld),offset),1);
}
float3 VegetationNormal(MeshVertex input,uint instance,float3 normal)
{
    InstanceData data=Instances[instance];
    if (data.Gust.w<0.5) return normal;
    float height=max(0.0,data.Gust.w<1.5?input.TexCoord.y:input.Position.y);
    float3 world=mul(data.World,float4(input.Position,1)).xyz;
    float3 derivative=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z)*2/max(height,0.01);
    normal.y-=dot(normal.xz,derivative.xz);
    return normalize(normal);
}
struct SurfaceVertex
{
    float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 LightPosition : TEXCOORD2;
    float2 TexCoord : TEXCOORD3;
};
SurfaceVertex VSForward(MeshVertex input, uint instance : SV_InstanceID)
{
    SurfaceVertex output;
    float4 world = mul(Instances[instance].World, VegetationPosition(input,instance,false));
    output.Position = mul(InstanceViewProjection, world);
    output.WorldPosition = world.xyz / max(abs(world.w), ProjectionEpsilon);
    output.Normal = SafeNormal(mul((float3x3)Instances[instance].NormalWorld, input.Normal), float3(0, 1, 0));
    output.Normal = VegetationNormal(input,instance,output.Normal);
    output.Color = float4(DecodeSRGB(saturate(input.Color.rgb)), input.Color.a);
    output.LightPosition = mul(LightViewProjection, world);
    output.TexCoord = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    return output;
}
struct DeferredVertex
{
    float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD2;
};
DeferredVertex VSDeferred(MeshVertex input, uint instance : SV_InstanceID)
{
    SurfaceVertex surface = VSForward(input, instance);
    DeferredVertex output;
    output.Position = surface.Position;
    output.WorldPosition = surface.WorldPosition;
    output.Normal = surface.Normal;
    output.Color = surface.Color;
    output.TexCoord = surface.TexCoord;
    return output;
}
struct ShadowVertex
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    float Alpha : COLOR0;
};
ShadowVertex VSShadow(MeshVertex input, uint instance : SV_InstanceID)
{
    ShadowVertex output;
    output.Position = mul(InstanceViewProjection, mul(Instances[instance].World, VegetationPosition(input,instance,false)));
    output.TexCoord = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    output.Alpha = input.Color.a;
    return output;
}
struct VelocityVertex
{
    float4 Position : SV_POSITION;
    float4 PreviousClip : TEXCOORD0;
    float2 TexCoord : TEXCOORD1;
    float Alpha : TEXCOORD2;
};
VelocityVertex VSVelocity(MeshVertex input, uint instance : SV_InstanceID)
{
    VelocityVertex output;
    // Match the opaque pass arithmetic for read-only depth replay. Invalid
    // histories have a zero previous matrix and are rejected by clip W in PS.
    output.Position = mul(InstanceViewProjection, mul(Instances[instance].World, VegetationPosition(input,instance,false)));
    output.PreviousClip = mul(Instances[instance].PreviousWorldViewProjection, VegetationPosition(input,instance,true));
    output.TexCoord = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    output.Alpha = input.Color.a * InstanceMaterial.x;
    return output;
}

// Gouraud still evaluates illumination at each transformed vertex. Splitting
// front/back and shadowed/unshadowed light preserves the surface pixel contract.
struct GouraudVertex
{
    float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 LightPosition : TEXCOORD2;
    float2 TexCoord : TEXCOORD3;
    float3 DiffuseFront : TEXCOORD4;
    float3 SpecularFront : TEXCOORD5;
    float3 ShadowDiffuseFront : TEXCOORD6;
    float3 ShadowSpecularFront : TEXCOORD7;
    float3 DiffuseBack : TEXCOORD8;
    float3 SpecularBack : TEXCOORD9;
    float3 ShadowDiffuseBack : TEXCOORD10;
    float3 ShadowSpecularBack : TEXCOORD11;
};
void LightSample(float3 toLight, float3 radiance, float3 normal, float3 view,
    inout float3 diffuse, inout float3 specular)
{
    float squareDistance = dot(toLight, toLight);
    if (squareDistance <= LightDistanceEpsilon) return;
    float3 direction = toLight * rsqrt(squareDistance);
    float cosine = max(dot(normal, direction), 0.0f);
    if (cosine <= 0.0f) return;
    diffuse += radiance * cosine;
    float3 halfVector = SafeNormal(direction + view, normal);
    specular += radiance * pow(max(dot(normal, halfVector), 0.0f), max(MaterialParameters.x, 1.0f));
}
void SceneLight(uint index, float3 position, float3 normal, float3 view,
    inout float3 diffuse, inout float3 specular)
{
    LightConstants light = Lights[index];
    uint type = uint(light.PositionType.w + EnabledThreshold);
    float3 radiance = max(light.ColorCastShadows.rgb, 0.0f) * max(light.DirectionIntensity.w, 0.0f);
    if (type == DirectionalLight)
    {
        LightSample(light.DirectionIntensity.xyz, radiance, normal, view, diffuse, specular);
        return;
    }
    if (type == PointLight)
    {
        float3 toLight = light.PositionType.xyz - position;
        float squareDistance = max(dot(toLight, toLight), LightDistanceEpsilon);
        LightSample(toLight, radiance * LightRangeWindow(toLight, light.AreaSizePadding.z) / squareDistance, normal, view, diffuse, specular);
        return;
    }
    float3 direction = SafeNormal(light.DirectionIntensity.xyz, float3(0, -1, 0));
    float3 helper = abs(direction.y) < ParallelAxisLimit ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = SafeNormal(cross(helper, direction), float3(1, 0, 0));
    float3 bitangent = SafeNormal(cross(direction, tangent), float3(0, 0, 1));
    [unroll] for (int y = 0; y < AreaSamplesPerAxis; ++y)
    [unroll] for (int x = 0; x < AreaSamplesPerAxis; ++x)
    {
        float offsetX = x == 0 ? -AreaSampleOffset : AreaSampleOffset;
        float offsetY = y == 0 ? -AreaSampleOffset : AreaSampleOffset;
        float3 samplePosition = light.PositionType.xyz + tangent * (offsetX * light.AreaSizePadding.x) +
            bitangent * (offsetY * light.AreaSizePadding.y);
        float3 toLight = samplePosition - position;
        float squareDistance = max(dot(toLight, toLight), LightDistanceEpsilon);
        float3 sampleDirection = toLight * rsqrt(squareDistance);
        float emitterCosine = max(dot(direction, -sampleDirection), 0.0f);
        LightSample(sampleDirection, radiance * (AreaSampleWeight * emitterCosine * LightRangeWindow(toLight, light.AreaSizePadding.z) / squareDistance),
            normal, view, diffuse, specular);
    }
}
void VertexLighting(float3 position, float3 normal, out float3 diffuse, out float3 specular,
    out float3 shadowDiffuse, out float3 shadowSpecular)
{
    diffuse = specular = shadowDiffuse = shadowSpecular = 0.0f;
    float3 view = SafeNormal(CameraLightCount.xyz - position, normal);
    uint count = min(uint(CameraLightCount.w + EnabledThreshold), MaximumLights);
    for (uint index = 0; index < count; ++index)
    {
        float3 lightDiffuse = 0.0f, lightSpecular = 0.0f;
        SceneLight(index, position, normal, view, lightDiffuse, lightSpecular);
        if (int(index) == int(RenderParameters.z) && Lights[index].ColorCastShadows.w > EnabledThreshold)
        {
            shadowDiffuse += lightDiffuse; shadowSpecular += lightSpecular;
        }
        else { diffuse += lightDiffuse; specular += lightSpecular; }
    }
}
GouraudVertex VSGouraud(MeshVertex input, uint instance : SV_InstanceID)
{
    SurfaceVertex surface = VSForward(input, instance);
    GouraudVertex output;
    output.Position = surface.Position; output.WorldPosition = surface.WorldPosition;
    output.Normal = surface.Normal; output.Color = surface.Color;
    output.LightPosition = surface.LightPosition; output.TexCoord = surface.TexCoord;
    VertexLighting(output.WorldPosition, output.Normal, output.DiffuseFront, output.SpecularFront,
        output.ShadowDiffuseFront, output.ShadowSpecularFront);
    VertexLighting(output.WorldPosition, -output.Normal, output.DiffuseBack, output.SpecularBack,
        output.ShadowDiffuseBack, output.ShadowSpecularBack);
    return output;
}
