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

// DX12 indexed rigid instances, independently maintained native HLSL.
// Dudash, "Animated Crowd Rendering", GPU Gems 3 (2007), chapter 2, motivates
// constant-array lookup by SV_InstanceID. Here the CPU supplies rigid current,
// inverse-transpose normal and previous clip transforms, without skinning.
// b3 is a root CBV into a fence-protected upload slice. These vertex entry points
// preserve DX12's existing material, shadow and velocity pixel-stage interfaces.
static const uint InstanceCapacity = 64u;
static const uint SceneLightCapacity = 4u;
static const uint SunLightKind = 0u;
static const uint PointLightKind = 1u;
static const float UnitVectorTolerance = 1.0e-12f;
static const float HomogeneousTolerance = 1.0e-7f;
static const float MinimumSquaredLightDistance = 1.0e-6f;
static const float BasisParallelThreshold = 0.999f;
static const float FlagThreshold = 0.5f;
static const float SrgbBreakpoint = 0.04045f;
static const float SrgbLinearDivisor = 12.92f;
static const float SrgbOffset = 0.055f;
static const float SrgbDivisor = 1.055f;
static const float SrgbExponent = 2.4f;
static const int RectangleQuadratureWidth = 2;
static const float RectangleQuadratureOffset = 0.25f;
static const float RectangleQuadratureWeight = 0.25f;

struct SceneEmitter
{
    float4 PositionKind;
    float4 DirectionPower;
    float4 ColorShadow;
    float4 Extent;
};
// Identical packing to the renderer's draw-constant ABI. Vertex lighting reads
// material shininess and emitters; fragment material/texture work stays in PS.
cbuffer DrawParameters : register(b0)
{
    row_major float4x4 ObjectToWorld;
    row_major float4x4 NormalToWorld;
    row_major float4x4 CameraProjection;
    row_major float4x4 ShadowProjection;
    float4 ObjectTint;
    float4 DiffuseReflectance;
    float4 AmbientReflectance;
    float4 SpecularReflectance;
    float4 Emission;
    float4 MaterialShape;
    float4 AmbientIllumination;
    float4 EyeAndLightCount;
    float4 RenderOptions;
    uint4 MaterialTextures;
    SceneEmitter Emitters[SceneLightCapacity];
    float4 SurfaceShape;
};
struct ObjectRecord
{
    row_major float4x4 LocalToWorld;
    row_major float4x4 NormalTransform;
    row_major float4x4 PreviousLocalToClip;
    float4 Wind, Gust, VegetationHistory;
};
cbuffer InstanceParameters : register(b3)
{
    row_major float4x4 BatchCamera;
    float4 BatchCoverage;
    ObjectRecord Objects[InstanceCapacity];
};
struct VertexInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 UV : TEXCOORD0;
};
float3 UnitVector(float3 direction, float3 fallback)
{
    float magnitudeSquared = dot(direction, direction);
    return magnitudeSquared > UnitVectorTolerance ? direction * rsqrt(magnitudeSquared) : fallback;
}
float3 DecodeVertexColor(float3 encoded)
{
    float3 linearBranch = encoded / SrgbLinearDivisor;
    float3 powerBranch = pow((encoded + SrgbOffset) / SrgbDivisor, SrgbExponent);
    return float3(encoded.r <= SrgbBreakpoint ? linearBranch.r : powerBranch.r,
        encoded.g <= SrgbBreakpoint ? linearBranch.g : powerBranch.g,
        encoded.b <= SrgbBreakpoint ? linearBranch.b : powerBranch.b);
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
float4 VegetationPosition(VertexInput input, uint instance, bool previous)
{
    ObjectRecord data=Objects[instance];
    if (data.Gust.w<0.5) return float4(input.Position,1);
    float3 world=mul(data.LocalToWorld,float4(input.Position,1)).xyz;
    float height=max(0.0,data.Gust.w<1.5?input.UV.y:input.Position.y);
    float3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,
        previous?data.VegetationHistory.x:data.Wind.z);
    return float4(input.Position+mul(transpose((float3x3)data.NormalTransform),offset),1);
}
float3 VegetationNormal(VertexInput input,uint instance,float3 normal)
{
    ObjectRecord data=Objects[instance];
    if (data.Gust.w<0.5) return normal;
    float height=max(0.0,data.Gust.w<1.5?input.UV.y:input.Position.y);
    float3 world=mul(data.LocalToWorld,float4(input.Position,1)).xyz;
    float3 derivative=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z)*2/max(height,0.01);
    normal.y-=dot(normal.xz,derivative.xz);
    return normalize(normal);
}
struct SurfaceVaryings
{
    float4 RasterPosition : SV_POSITION;
    float3 PositionInWorld : TEXCOORD0;
    float3 NormalInWorld : TEXCOORD1;
    float4 LinearColor : COLOR0;
    float4 ShadowPosition : TEXCOORD2;
    float2 TextureCoordinate : TEXCOORD3;
};
SurfaceVaryings VSForward(VertexInput vertex, uint objectIndex : SV_InstanceID)
{
    SurfaceVaryings result;
    float4 worldPosition = mul(Objects[objectIndex].LocalToWorld, VegetationPosition(vertex,objectIndex,false));
    result.RasterPosition = mul(BatchCamera, worldPosition);
    result.PositionInWorld = worldPosition.xyz / max(abs(worldPosition.w), HomogeneousTolerance);
    result.NormalInWorld = UnitVector(mul((float3x3)Objects[objectIndex].NormalTransform, vertex.Normal), float3(0, 1, 0));
    result.NormalInWorld = VegetationNormal(vertex,objectIndex,result.NormalInWorld);
    result.LinearColor = float4(DecodeVertexColor(saturate(vertex.Color.rgb)), vertex.Color.a);
    result.ShadowPosition = mul(ShadowProjection, worldPosition);
    result.TextureCoordinate = float2(vertex.UV.x, 1 - vertex.UV.y);
    return result;
}
struct MaterialVaryings
{
    float4 RasterPosition : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    float2 TextureCoordinate : TEXCOORD2;
    float4 LinearColor : COLOR0;
};
MaterialVaryings VSDeferred(VertexInput vertex, uint objectIndex : SV_InstanceID)
{
    MaterialVaryings result;
    float4 position = mul(Objects[objectIndex].LocalToWorld, VegetationPosition(vertex,objectIndex,false));
    result.RasterPosition = mul(BatchCamera, position);
    result.WorldPosition = position.xyz / max(abs(position.w), HomogeneousTolerance);
    result.WorldNormal = UnitVector(mul((float3x3)Objects[objectIndex].NormalTransform, vertex.Normal), float3(0, 1, 0));
    result.WorldNormal = VegetationNormal(vertex,objectIndex,result.WorldNormal);
    result.TextureCoordinate = float2(vertex.UV.x, 1 - vertex.UV.y);
    result.LinearColor = float4(DecodeVertexColor(saturate(vertex.Color.rgb)), vertex.Color.a);
    return result;
}
struct DepthVaryings
{
    float4 Position : SV_POSITION;
    float2 UV : TEXCOORD0;
    float Alpha : COLOR0;
};
DepthVaryings VSShadow(VertexInput vertex, uint objectIndex : SV_InstanceID)
{
    DepthVaryings result;
    result.Position = mul(BatchCamera, mul(Objects[objectIndex].LocalToWorld, VegetationPosition(vertex,objectIndex,false)));
    result.UV = float2(vertex.UV.x, 1 - vertex.UV.y);
    result.Alpha = vertex.Color.a;
    return result;
}
struct CascadeVaryings
{
    float4 Clip : SV_POSITION;
    float2 UV : TEXCOORD0;
    float VertexOpacity : TEXCOORD1;
};
CascadeVaryings VSCascade(VertexInput vertex, uint objectIndex : SV_InstanceID)
{
    // CSM's independent pixel stage carries coverage in TEXCOORD1, whereas
    // single-map shadows carry it in COLOR0. Preserve both native contracts.
    CascadeVaryings result;
    result.Clip = mul(BatchCamera, mul(Objects[objectIndex].LocalToWorld, VegetationPosition(vertex,objectIndex,false)));
    result.UV = float2(vertex.UV.x, 1 - vertex.UV.y);
    result.VertexOpacity = vertex.Color.a;
    return result;
}
struct MotionVaryings
{
    float4 RasterPosition : SV_POSITION;
    float4 PreviousPosition : TEXCOORD0;
    float2 UV : TEXCOORD1;
    float Coverage : TEXCOORD2;
};
MotionVaryings VSVelocity(VertexInput vertex, uint objectIndex : SV_InstanceID)
{
    MotionVaryings result;
    // Keep the opaque pass's operation order for its read-only depth replay.
    // A zero previous matrix carries per-instance invalid history through the
    // existing velocity PS clip-W rejection. Other instances remain valid.
    result.RasterPosition = mul(BatchCamera, mul(Objects[objectIndex].LocalToWorld, VegetationPosition(vertex,objectIndex,false)));
    result.PreviousPosition = mul(Objects[objectIndex].PreviousLocalToClip, VegetationPosition(vertex,objectIndex,true));
    result.UV = float2(vertex.UV.x, 1 - vertex.UV.y);
    result.Coverage = vertex.Color.a * BatchCoverage.x;
    return result;
}

struct LightPair { float3 Diffuse; float3 Specular; };
void AddLightSample(float3 towardLight, float3 radiance, float3 normal, float3 towardEye, inout LightPair sum)
{
    float lengthSquared = dot(towardLight, towardLight);
    if (lengthSquared <= UnitVectorTolerance) return;
    float3 lightDirection = towardLight * rsqrt(lengthSquared);
    float cosine = max(dot(normal, lightDirection), 0.0f);
    if (cosine <= 0) return;
    sum.Diffuse += radiance * cosine;
    float3 halfDirection = UnitVector(lightDirection + towardEye, normal);
    float highlight = pow(max(dot(normal, halfDirection), 0.0f), max(MaterialShape.x, 1.0f));
    sum.Specular += radiance * highlight;
}
LightPair VertexEmitter(SceneEmitter emitter, float3 worldPosition, float3 normal, float3 towardEye)
{
    LightPair result; result.Diffuse = result.Specular = 0;
    float3 radiance = max(emitter.ColorShadow.rgb, 0.0f) * max(emitter.DirectionPower.w, 0.0f);
    uint kind = uint(emitter.PositionKind.w + FlagThreshold);
    if (kind == SunLightKind)
        AddLightSample(emitter.DirectionPower.xyz, radiance, normal, towardEye, result);
    else if (kind == PointLightKind)
    {
        float3 offset = emitter.PositionKind.xyz - worldPosition;
        float squaredDistance = max(dot(offset, offset), MinimumSquaredLightDistance);
        AddLightSample(offset, radiance * LightRangeWindow(offset, emitter.Extent.z) / squaredDistance, normal, towardEye, result);
    }
    else
    {
        float3 axis = UnitVector(emitter.DirectionPower.xyz, float3(0, -1, 0));
        float3 helper = abs(axis.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 tangent = UnitVector(cross(helper, axis), float3(1, 0, 0));
        float3 bitangent = UnitVector(cross(axis, tangent), float3(0, 0, 1));
        [unroll] for (int row = 0; row < RectangleQuadratureWidth; ++row)
        [unroll] for (int column = 0; column < RectangleQuadratureWidth; ++column)
        {
            float horizontal = column == 0 ? -RectangleQuadratureOffset : RectangleQuadratureOffset;
            float vertical = row == 0 ? -RectangleQuadratureOffset : RectangleQuadratureOffset;
            float3 location = emitter.PositionKind.xyz + tangent * (horizontal * emitter.Extent.x) +
                bitangent * (vertical * emitter.Extent.y);
            float3 offset = location - worldPosition;
            float squaredDistance = max(dot(offset, offset), MinimumSquaredLightDistance);
            float3 direction = offset * rsqrt(squaredDistance);
            float emitterCosine = max(dot(axis, -direction), 0.0f);
            AddLightSample(direction, radiance * (RectangleQuadratureWeight * emitterCosine * LightRangeWindow(offset, emitter.Extent.z) / squaredDistance),
                normal, towardEye, result);
        }
    }
    return result;
}
void VertexIllumination(float3 position, float3 normal, out LightPair unshadowed, out LightPair shadowed)
{
    unshadowed.Diffuse = unshadowed.Specular = shadowed.Diffuse = shadowed.Specular = 0;
    float3 towardEye = UnitVector(EyeAndLightCount.xyz - position, normal);
    uint count = min(uint(EyeAndLightCount.w + FlagThreshold), SceneLightCapacity);
    for (uint index = 0; index < count; ++index)
    {
        LightPair contribution = VertexEmitter(Emitters[index], position, normal, towardEye);
        if (int(index) == int(RenderOptions.z) && Emitters[index].ColorShadow.w > FlagThreshold)
        {
            shadowed.Diffuse += contribution.Diffuse;
            shadowed.Specular += contribution.Specular;
        }
        else
        {
            unshadowed.Diffuse += contribution.Diffuse;
            unshadowed.Specular += contribution.Specular;
        }
    }
}
struct LitVaryings
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
LitVaryings VSGouraud(VertexInput vertex, uint objectIndex : SV_InstanceID)
{
    SurfaceVaryings surface = VSForward(vertex, objectIndex);
    LitVaryings result;
    result.Position = surface.RasterPosition; result.WorldPosition = surface.PositionInWorld;
    result.Normal = surface.NormalInWorld; result.Color = surface.LinearColor;
    result.LightPosition = surface.ShadowPosition; result.TexCoord = surface.TextureCoordinate;
    LightPair front, frontShadow, back, backShadow;
    VertexIllumination(result.WorldPosition, result.Normal, front, frontShadow);
    VertexIllumination(result.WorldPosition, -result.Normal, back, backShadow);
    result.DiffuseFront = front.Diffuse; result.SpecularFront = front.Specular;
    result.ShadowDiffuseFront = frontShadow.Diffuse; result.ShadowSpecularFront = frontShadow.Specular;
    result.DiffuseBack = back.Diffuse; result.SpecularBack = back.Specular;
    result.ShadowDiffuseBack = backShadow.Diffuse; result.ShadowSpecularBack = backShadow.Specular;
    return result;
}
