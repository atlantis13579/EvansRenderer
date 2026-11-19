#include <metal_stdlib>
using namespace metal;

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

// Independent Metal instancing: vertex_id addresses shared geometry, while
// instance_id selects the object's current, normal and previous transforms.
// Apple, Metal Shading Language Specification, "Vertex Function Arguments":
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
// Matrix rows preserve the CPU ABI without relying on MSL matrix storage order.
// Each output matches its consumer's stage-in names, types and interpolation.
constant uint MaximumInstances = 64;
constant uint MaximumLights = 4;
constant uint VertexFloatCount = 12;
constant uint PositionOffset = 0;
constant uint NormalOffset = 3;
constant uint ColorOffset = 6;
constant uint AlphaOffset = 9;
constant uint TexCoordOffset = 10;
constant uint DirectionalLight = 0;
constant uint PointLight = 1;
constant uint RectangularLight = 2;
constant uint AreaSampleCount = 4;
constant float AreaQuarterExtent = 0.25;
constant float LightTypeRounding = 0.5;
constant float NormalizeEpsilon = 1.0e-12;
constant float HomogeneousEpsilon = 1.0e-7;
constant float DistanceEpsilon = 1.0e-6;
constant float ParallelAxisThreshold = 0.999;
constant float SrgbLinearThreshold = 0.04045;
constant float SrgbLinearScale = 12.92;
constant float SrgbPowerOffset = 0.055;
constant float SrgbPowerScale = 1.055;
constant float SrgbExponent = 2.4;

struct MeshVertex { float Values[VertexFloatCount]; };
struct ObjectTransforms
{
    float4 World[4];
    float4 NormalWorld[4];
    float4 PreviousWorldViewProjection[4];
    float4 Wind, Gust, VegetationHistory;
};
// Buffer 3 is a real MTLBuffer: this 15,440-byte block exceeds setVertexBytes.
struct InstanceBlock
{
    float4 ViewProjection[4];
    float4 Material;
    ObjectTransforms Instances[MaximumInstances];
};
struct SceneLight
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
    SceneLight Lights[MaximumLights];
    float4 ShadingParameters;
};

float4 TransformRows(constant float4* rows, float4 value)
{
    return float4(dot(rows[0], value), dot(rows[1], value),
        dot(rows[2], value), dot(rows[3], value));
}
float4 LocalPosition(device const MeshVertex& input)
{
    return float4(input.Values[PositionOffset], input.Values[PositionOffset + 1],
        input.Values[PositionOffset + 2], 1.0);
}
float2 MaterialUV(device const MeshVertex& input)
{
    return float2(input.Values[TexCoordOffset], 1.0 - input.Values[TexCoordOffset + 1]);
}
float3 NormalizeSafe(float3 value, float3 fallbackValue)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > NormalizeEpsilon ? value * rsqrt(lengthSquared) : fallbackValue;
}
float3 ObjectNormal(device const MeshVertex& input, constant ObjectTransforms& object)
{
    float3 normal(input.Values[NormalOffset], input.Values[NormalOffset + 1], input.Values[NormalOffset + 2]);
    return NormalizeSafe(TransformRows(object.NormalWorld, float4(normal, 0.0)).xyz, float3(0.0, 1.0, 0.0));
}
float4 LinearVertexColor(device const MeshVertex& input)
{
    float3 color = saturate(float3(input.Values[ColorOffset], input.Values[ColorOffset + 1], input.Values[ColorOffset + 2]));
    float3 low = color / SrgbLinearScale;
    float3 high = pow((color + SrgbPowerOffset) / SrgbPowerScale, SrgbExponent);
    return float4(color.r <= SrgbLinearThreshold ? low.r : high.r,
        color.g <= SrgbLinearThreshold ? low.g : high.g,
        color.b <= SrgbLinearThreshold ? low.b : high.b, input.Values[AlphaOffset]);
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
float4 VegetationPosition(device const MeshVertex& input, constant ObjectTransforms& object, bool previous)
{
    float4 local=LocalPosition(input);
    if (object.Gust.w<0.5) return local;
    float3 world=TransformRows(object.World,local).xyz;
    float height=max(0.0f,object.Gust.w<1.5?input.Values[TexCoordOffset+1]:local.y);
    float3 offset=VegetationOffset(world,height,object.Wind,object.Gust,object.VegetationHistory.y,
        previous?object.VegetationHistory.x:object.Wind.z);
    local.xyz+=object.NormalWorld[0].xyz*offset.x+object.NormalWorld[1].xyz*offset.y+object.NormalWorld[2].xyz*offset.z;
    return local;
}
float3 VegetationNormal(device const MeshVertex& input, constant ObjectTransforms& object,float3 normal)
{
    if (object.Gust.w<0.5) return normal;
    float height=max(0.0f,object.Gust.w<1.5?input.Values[TexCoordOffset+1]:LocalPosition(input).y);
    float3 world=TransformRows(object.World,LocalPosition(input)).xyz;
    float3 derivative=VegetationOffset(world,height,object.Wind,object.Gust,object.VegetationHistory.y,object.Wind.z)*2.0/max(height,0.01f);
    normal.y-=dot(normal.xz,derivative.xz);
    return normalize(normal);
}
struct ForwardVaryings
{
    float4 Position [[position]];
    float3 WorldPosition;
    float3 Normal;
    float4 Color;
    float4 LightPosition;
    float2 TexCoord;
};
ForwardVaryings TransformForward(device const MeshVertex& input, constant ObjectTransforms& object,
    constant InstanceBlock& batch, constant DrawConstants& draw)
{
    float4 world = TransformRows(object.World, VegetationPosition(input,object,false));
    ForwardVaryings output;
    // Keep world-then-camera arithmetic identical to ordinary mesh draws.
    output.Position = TransformRows(batch.ViewProjection, world);
    output.WorldPosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    output.Normal = VegetationNormal(input,object,ObjectNormal(input, object));
    output.Color = LinearVertexColor(input);
    output.LightPosition = TransformRows(draw.LightViewProjection, world);
    output.TexCoord = MaterialUV(input);
    return output;
}
vertex ForwardVaryings InstanceForwardVertex(device const MeshVertex* vertices [[buffer(0)]],
    constant DrawConstants& draw [[buffer(1)]], constant InstanceBlock& batch [[buffer(3)]],
    uint vertexId [[vertex_id]], uint instanceId [[instance_id]])
{
    return TransformForward(vertices[vertexId], batch.Instances[instanceId], batch, draw);
}

struct GouraudVaryings
{
    float4 Position [[position]];
    float3 WorldPosition;
    float3 Normal;
    float4 Color;
    float4 LightPosition;
    float2 TexCoord;
    float3 DiffuseFront;
    float3 DiffuseBack;
    float3 SpecularFront;
    float3 SpecularBack;
    float3 ShadowDiffuseFront;
    float3 ShadowSpecularFront;
    float3 ShadowDiffuseBack;
    float3 ShadowSpecularBack;
};
// Gouraud evaluates both sides before interpolation. Its fragment stage expects
// total lighting plus the selected shadow light's contribution to subtract.
void EvaluateVertexLight(float3 position, float3 normal, constant DrawConstants& draw,
    thread float3& diffuse, thread float3& specular, thread float3& shadowDiffuse, thread float3& shadowSpecular)
{
    diffuse = specular = shadowDiffuse = shadowSpecular = float3(0.0);
    float3 view = NormalizeSafe(draw.CameraLightCount.xyz - position, normal);
    for (uint lightIndex = 0; lightIndex < min(uint(draw.CameraLightCount.w), MaximumLights); ++lightIndex)
    {
        SceneLight light = draw.Lights[lightIndex];
        uint type = uint(light.PositionType.w + LightTypeRounding);
        float3 emission = NormalizeSafe(light.DirectionIntensity.xyz, float3(0.0, -1.0, 0.0));
        float3 axis = abs(emission.y) < ParallelAxisThreshold ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        float3 tangent = NormalizeSafe(cross(axis, emission), float3(1.0, 0.0, 0.0));
        float3 bitangent = cross(emission, tangent);
        uint count = type == RectangularLight ? AreaSampleCount : 1;
        for (uint sampleIndex = 0; sampleIndex < count; ++sampleIndex)
        {
            float2 offset = float2(sampleIndex % 2 == 0 ? -1.0 : 1.0,
                sampleIndex / 2 == 0 ? -1.0 : 1.0) * AreaQuarterExtent;
            float3 samplePosition = light.PositionType.xyz + tangent * offset.x * light.AreaSizePadding.x +
                bitangent * offset.y * light.AreaSizePadding.y;
            float3 toLight = type == DirectionalLight ? light.DirectionIntensity.xyz :
                (type == PointLight ? light.PositionType.xyz : samplePosition) - position;
            float3 direction = NormalizeSafe(toLight, normal);
            float attenuation = type == DirectionalLight ? 1.0 : LightRangeWindow(toLight, light.AreaSizePadding.z) / max(dot(toLight, toLight), DistanceEpsilon);
            if (type == RectangularLight) attenuation *= max(dot(emission, -direction), 0.0) / float(AreaSampleCount);
            float3 radiance = max(light.ColorCastShadows.rgb, 0.0) * max(light.DirectionIntensity.w, 0.0) * attenuation;
            float cosine = max(dot(normal, direction), 0.0);
            float3 diffuseTerm = radiance * cosine;
            float3 specularTerm = cosine > 0.0 ? radiance * pow(max(dot(normal,
                NormalizeSafe(direction + view, normal)), 0.0), max(draw.MaterialParameters.x, 1.0)) : float3(0.0);
            diffuse += diffuseTerm;
            specular += specularTerm;
            if (int(lightIndex) == int(draw.RenderParameters.z))
            {
                shadowDiffuse += diffuseTerm;
                shadowSpecular += specularTerm;
            }
        }
    }
}
vertex GouraudVaryings InstanceGouraudVertex(device const MeshVertex* vertices [[buffer(0)]],
    constant DrawConstants& draw [[buffer(1)]], constant InstanceBlock& batch [[buffer(3)]],
    uint vertexId [[vertex_id]], uint instanceId [[instance_id]])
{
    ForwardVaryings surface = TransformForward(vertices[vertexId], batch.Instances[instanceId], batch, draw);
    GouraudVaryings output;
    output.Position = surface.Position;
    output.WorldPosition = surface.WorldPosition;
    output.Normal = surface.Normal;
    output.Color = surface.Color;
    output.LightPosition = surface.LightPosition;
    output.TexCoord = surface.TexCoord;
    EvaluateVertexLight(surface.WorldPosition, surface.Normal, draw, output.DiffuseFront,
        output.SpecularFront, output.ShadowDiffuseFront, output.ShadowSpecularFront);
    EvaluateVertexLight(surface.WorldPosition, -surface.Normal, draw, output.DiffuseBack,
        output.SpecularBack, output.ShadowDiffuseBack, output.ShadowSpecularBack);
    return output;
}

struct GeometryVaryings
{
    float4 Position [[position]];
    float3 WorldPosition;
    float3 Normal;
    float4 Color;
    float2 TexCoord;
};
vertex GeometryVaryings InstanceDeferredVertex(device const MeshVertex* vertices [[buffer(0)]],
    constant InstanceBlock& batch [[buffer(3)]], uint vertexId [[vertex_id]], uint instanceId [[instance_id]])
{
    device const MeshVertex& input = vertices[vertexId];
    constant ObjectTransforms& object = batch.Instances[instanceId];
    float4 world = TransformRows(object.World, VegetationPosition(input,object,false));
    GeometryVaryings output;
    output.Position = TransformRows(batch.ViewProjection, world);
    output.WorldPosition = world.xyz / max(abs(world.w), HomogeneousEpsilon);
    output.Normal = VegetationNormal(input,object,ObjectNormal(input, object));
    output.Color = LinearVertexColor(input);
    output.TexCoord = MaterialUV(input);
    return output;
}

struct ShadowVaryings { float4 Position [[position]]; float2 UV; float VertexAlpha; };
struct CascadeVaryings { float4 Position [[position]]; float2 MaterialUV; float Opacity; };
vertex ShadowVaryings InstanceShadowVertex(device const MeshVertex* vertices [[buffer(0)]],
    constant InstanceBlock& batch [[buffer(3)]], uint vertexId [[vertex_id]], uint instanceId [[instance_id]])
{
    device const MeshVertex& input = vertices[vertexId];
    ShadowVaryings output;
    output.Position = TransformRows(batch.ViewProjection, TransformRows(batch.Instances[instanceId].World, VegetationPosition(input,batch.Instances[instanceId],false)));
    output.UV = MaterialUV(input);
    output.VertexAlpha = input.Values[AlphaOffset];
    return output;
}
vertex CascadeVaryings InstanceCascadeVertex(device const MeshVertex* vertices [[buffer(0)]],
    constant InstanceBlock& batch [[buffer(3)]], uint vertexId [[vertex_id]], uint instanceId [[instance_id]])
{
    device const MeshVertex& input = vertices[vertexId];
    CascadeVaryings output;
    output.Position = TransformRows(batch.ViewProjection, TransformRows(batch.Instances[instanceId].World, VegetationPosition(input,batch.Instances[instanceId],false)));
    output.MaterialUV = MaterialUV(input);
    output.Opacity = input.Values[AlphaOffset];
    return output;
}

struct VelocityVaryings { float4 Position [[position]]; float4 PreviousClip; float2 UV; float Alpha; };
vertex VelocityVaryings InstanceVelocityVertex(device const MeshVertex* vertices [[buffer(0)]],
    constant InstanceBlock& batch [[buffer(3)]], uint vertexId [[vertex_id]], uint instanceId [[instance_id]])
{
    device const MeshVertex& input = vertices[vertexId];
    constant ObjectTransforms& object = batch.Instances[instanceId];
    float4 local = VegetationPosition(input,object,false);
    VelocityVaryings output;
    output.Position = TransformRows(batch.ViewProjection, TransformRows(object.World, local));
    // A zero previous matrix marks invalid history independently for each object.
    output.PreviousClip = TransformRows(object.PreviousWorldViewProjection, VegetationPosition(input,object,true));
    output.UV = MaterialUV(input);
    output.Alpha = input.Values[AlphaOffset] * batch.Material.x;
    return output;
}
