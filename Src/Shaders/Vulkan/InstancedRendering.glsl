#version 450
#if !defined(EVANS_INSTANCE_SHADOW) && !defined(EVANS_INSTANCE_CASCADE)
// Match the ordinary material/deferred vertex paths: contraction differences
// otherwise alter edge derivatives and texture LOD when a one-object draw
// switches to GPU-driven instance input.
// Velocity replay must retain the same depth as these opaque submissions.
invariant gl_Position;
precise gl_Position;
#endif
// Karis' smooth inverse-square window (Filament, Punctual lights).
// Range is measured at each emitter sample; zero keeps infinite support.
float LightRangeWindow(vec3 delta, float range)
{
    if (range <= 0.0) return 1.0;
    if (any(greaterThanEqual(abs(delta), vec3(range)))) return 0.0;
    vec3 ratio = delta / range;
    float radiusFractionSquared = dot(ratio, ratio);
    float window = clamp(1.0 - radiusFractionSquared * radiusFractionSquared, 0.0, 1.0);
    return window * window;
}

// Native Vulkan rigid instancing, independently authored for explicit bindings.
// Dudash, "Animated Crowd Rendering", GPU Gems 3 (2007), chapter 2: transform
// arrays indexed by the system instance ID avoid duplicating mesh vertices.
// Set 2 holds one immutable dynamic-UBO slice. Set 0 retains each consuming
// technique's material contract; set 1 remains available for cascade receivers.
// No descriptor indexing extension, skinning, or additional authored asset.
const int InstanceCapacity = 64;
const int EmitterCapacity = 4;
const float VectorLengthTolerance = 1.0e-12;
const float HomogeneousTolerance = 1.0e-7;
const float SrgbThreshold = 0.04045;
const float SrgbLinearScale = 12.92;
const float SrgbPowerOffset = 0.055;
const float SrgbPowerScale = 1.055;
const float SrgbPowerExponent = 2.4;

struct RigidTransform
{
    mat4 World;
    mat4 NormalWorld;
    mat4 PreviousClip;
    vec4 Wind, Gust, VegetationHistory;
};
layout(set = 2, binding = 0, std140, row_major) uniform InstanceTransforms
{
    mat4 Camera;
    vec4 Coverage;
    RigidTransform Objects[InstanceCapacity];
} Batch;
struct Emitter
{
    vec4 PositionKind;
    vec4 DirectionIntensity;
    vec4 ColorShadow;
    vec4 RectangleExtent;
};
layout(set = 0, binding = 0, std140, row_major) uniform MaterialDraw
{
    mat4 World;
    mat4 NormalWorld;
    mat4 ViewProjection;
    mat4 LightProjection;
    vec4 Tint;
    vec4 Diffuse;
    vec4 Ambient;
    vec4 Specular;
    vec4 Emission;
    vec4 MaterialShape;
    vec4 AmbientLight;
    vec4 EyeAndLightCount;
    vec4 RenderOptions;
    uvec4 TextureOptions;
    Emitter Lights[EmitterCapacity];
    vec4 SurfaceShape;
} Material;

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec4 Color;
layout(location = 3) in vec2 UV;
vec3 UnitDirection(vec3 value, vec3 fallback)
{
    float squaredLength = dot(value, value);
    return squaredLength > VectorLengthTolerance ? value * inversesqrt(squaredLength) : fallback;
}
vec4 DecodeColor()
{
    vec3 encoded = clamp(Color.rgb, 0.0, 1.0);
    vec3 linearPart = encoded / SrgbLinearScale;
    vec3 powerPart = pow((encoded + SrgbPowerOffset) / SrgbPowerScale, vec3(SrgbPowerExponent));
    return vec4(mix(powerPart, linearPart, lessThanEqual(encoded, vec3(SrgbThreshold))), Color.a);
}
vec4 VulkanPosition(vec4 clip)
{
    // Positive-height Vulkan viewports need this Y flip; previous clip positions
    // stay in the renderer's Y-up convention for the velocity fragment stage.
    clip.y = -clip.y;
    return clip;
}

// Spatially coherent gusts travel across patches; a finer wave flutters tips.
// The quadratic profile keeps roots fixed in color, depth, reflection and motion passes.
vec3 VegetationOffset(vec3 world, float height, vec4 wind, vec4 gust, float referenceHeight, float time)
{
    float phase=dot(world.xz,wind.xy)*6.2831853/max(gust.x,1.0)-time*1.7;
    float wave=0.55+gust.y*(0.35*sin(phase)+0.22*sin(phase*0.47+1.3));
    float flutter=gust.z*sin(time*7.1+world.x*1.83+world.z*2.17);
    float bend=clamp(wind.w*(wave+flutter)*height*height/max(referenceHeight,0.1),-height*0.6,height*0.6);
    return vec3(wind.x*bend,-abs(bend)*0.22,wind.y*bend);
}
vec4 VegetationPosition(bool previous)
{
    RigidTransform data=Batch.Objects[gl_InstanceIndex];
    if (data.Gust.w<0.5) return vec4(Position,1.0);
    vec3 world=(data.World*vec4(Position,1.0)).xyz;
    float height=max(0.0,data.Gust.w<1.5?UV.y:Position.y);
    vec3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,
        previous?data.VegetationHistory.x:data.Wind.z);
    return vec4(Position+transpose(mat3(data.NormalWorld))*offset,1.0);
}
vec3 VegetationNormal(vec3 normal)
{
    RigidTransform data=Batch.Objects[gl_InstanceIndex];
    if (data.Gust.w<0.5) return normal;
    float height=max(0.0,data.Gust.w<1.5?UV.y:Position.y);
    vec3 world=(data.World*vec4(Position,1.0)).xyz;
    vec3 derivative=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z)*2.0/max(height,0.01);
    normal.y-=dot(normal.xz,derivative.xz);
    return normalize(normal);
}
#if defined(EVANS_INSTANCE_SHADOW) || defined(EVANS_INSTANCE_CASCADE)
layout(location = 0) out vec2 MaterialUV;
layout(location = 1) out float VertexOpacity;
void main()
{
#ifdef EVANS_INSTANCE_CASCADE
    // Preserve CSM's matrix association separately from single-map depth.
    gl_Position = VulkanPosition(Batch.Camera * Batch.Objects[gl_InstanceIndex].World * VegetationPosition(false));
#else
    vec4 world = Batch.Objects[gl_InstanceIndex].World * VegetationPosition(false);
    gl_Position = VulkanPosition(Batch.Camera * world);
#endif
    MaterialUV = vec2(UV.x, 1.0 - UV.y);
    VertexOpacity = Color.a;
}
#elif defined(EVANS_INSTANCE_VELOCITY)
layout(location = 0) out vec4 PreviousClip;
layout(location = 1) out vec2 MaterialUV;
layout(location = 2) out float Opacity;
void main()
{
    vec4 world = Batch.Objects[gl_InstanceIndex].World * VegetationPosition(false);
    gl_Position = VulkanPosition(Batch.Camera * world);
    // Invalid history is a zero previous matrix. Its clip W rejects just this
    // object's correspondence, while other objects keep their valid velocity.
    PreviousClip = Batch.Objects[gl_InstanceIndex].PreviousClip * VegetationPosition(true);
    MaterialUV = vec2(UV.x, 1.0 - UV.y);
    Opacity = Color.a * Batch.Coverage.x;
}
#else
layout(location = 0) out vec3 WorldPosition;
layout(location = 1) out vec3 WorldNormal;
layout(location = 2) out vec4 VertexColor;
#ifdef EVANS_INSTANCE_DEFERRED
layout(location = 3) out vec2 MaterialUV;
#else
layout(location = 3) out vec4 LightPosition;
layout(location = 4) out vec2 MaterialUV;
#endif

#ifdef EVANS_INSTANCE_GOURAUD
const int DirectionalEmitter = 0;
const int PointEmitter = 1;
const int AreaGridWidth = 2;
const float AreaSampleOffset = 0.25;
const float AreaSampleWeight = 0.25;
const float MinimumLightDistanceSquared = 1.0e-6;
const float ParallelAxisThreshold = 0.999;
const float IntegerRoundingOffset = 0.5;
layout(location = 5) out vec3 DiffuseFront;
layout(location = 6) out vec3 SpecularFront;
layout(location = 7) out vec3 DiffuseBack;
layout(location = 8) out vec3 SpecularBack;
layout(location = 9) out vec3 ShadowDiffuseFront;
layout(location = 10) out vec3 ShadowSpecularFront;
layout(location = 11) out vec3 ShadowDiffuseBack;
layout(location = 12) out vec3 ShadowSpecularBack;
struct Irradiance { vec3 Diffuse; vec3 Specular; };
void AccumulateSample(vec3 towardLight, vec3 radiance, vec3 normal, vec3 view, inout Irradiance sum)
{
    float lengthSquared = dot(towardLight, towardLight);
    if (lengthSquared <= VectorLengthTolerance) return;
    vec3 direction = towardLight * inversesqrt(lengthSquared);
    float cosine = max(dot(normal, direction), 0.0);
    if (cosine <= 0.0) return;
    sum.Diffuse += radiance * cosine;
    vec3 halfDirection = UnitDirection(direction + view, normal);
    sum.Specular += radiance * pow(max(dot(normal, halfDirection), 0.0), max(Material.MaterialShape.x, 1.0));
}
Irradiance EvaluateEmitter(Emitter light, vec3 position, vec3 normal, vec3 view)
{
    Irradiance result; result.Diffuse = result.Specular = vec3(0.0);
    int kind = int(light.PositionKind.w + IntegerRoundingOffset);
    vec3 radiance = max(light.ColorShadow.rgb, vec3(0.0)) * max(light.DirectionIntensity.w, 0.0);
    if (kind == DirectionalEmitter)
        AccumulateSample(light.DirectionIntensity.xyz, radiance, normal, view, result);
    else if (kind == PointEmitter)
    {
        vec3 displacement = light.PositionKind.xyz - position;
        float squaredDistance = max(dot(displacement, displacement), MinimumLightDistanceSquared);
        AccumulateSample(displacement, radiance * LightRangeWindow(displacement, light.RectangleExtent.z) / squaredDistance, normal, view, result);
    }
    else
    {
        vec3 emission = UnitDirection(light.DirectionIntensity.xyz, vec3(0.0, -1.0, 0.0));
        vec3 helper = abs(emission.y) < ParallelAxisThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent = UnitDirection(cross(helper, emission), vec3(1.0, 0.0, 0.0));
        vec3 bitangent = UnitDirection(cross(emission, tangent), vec3(0.0, 0.0, 1.0));
        for (int row = 0; row < AreaGridWidth; ++row) for (int column = 0; column < AreaGridWidth; ++column)
        {
            vec2 offset = vec2(column == 0 ? -AreaSampleOffset : AreaSampleOffset,
                row == 0 ? -AreaSampleOffset : AreaSampleOffset);
            vec3 samplePosition = light.PositionKind.xyz + tangent * offset.x * light.RectangleExtent.x +
                bitangent * offset.y * light.RectangleExtent.y;
            vec3 displacement = samplePosition - position;
            float squaredDistance = max(dot(displacement, displacement), MinimumLightDistanceSquared);
            vec3 direction = displacement * inversesqrt(squaredDistance);
            float emitterCosine = max(dot(emission, -direction), 0.0);
            AccumulateSample(direction, radiance * (AreaSampleWeight * emitterCosine * LightRangeWindow(displacement, light.RectangleExtent.z) / squaredDistance), normal, view, result);
        }
    }
    return result;
}
void Illuminate()
{
    DiffuseFront = SpecularFront = DiffuseBack = SpecularBack = vec3(0.0);
    ShadowDiffuseFront = ShadowSpecularFront = ShadowDiffuseBack = ShadowSpecularBack = vec3(0.0);
    vec3 view = UnitDirection(Material.EyeAndLightCount.xyz - WorldPosition, WorldNormal);
    int count = min(int(Material.EyeAndLightCount.w + IntegerRoundingOffset), EmitterCapacity);
    for (int index = 0; index < count; ++index)
    {
        Irradiance front = EvaluateEmitter(Material.Lights[index], WorldPosition, WorldNormal, view);
        Irradiance back = EvaluateEmitter(Material.Lights[index], WorldPosition, -WorldNormal, view);
        DiffuseFront += front.Diffuse; SpecularFront += front.Specular;
        DiffuseBack += back.Diffuse; SpecularBack += back.Specular;
        // Vulkan Gouraud's fragment stage subtracts occluded light from total
        // irradiance. Keep the shadowed emitter's contribution separately.
        if (index == int(Material.RenderOptions.z))
        {
            ShadowDiffuseFront = front.Diffuse; ShadowSpecularFront = front.Specular;
            ShadowDiffuseBack = back.Diffuse; ShadowSpecularBack = back.Specular;
        }
    }
}
#endif
void main()
{
    vec4 world = Batch.Objects[gl_InstanceIndex].World * VegetationPosition(false);
    gl_Position = Batch.Camera * world;
    gl_Position.y = -gl_Position.y;
    WorldPosition = world.xyz / max(abs(world.w), HomogeneousTolerance);
    WorldNormal = VegetationNormal(UnitDirection(mat3(Batch.Objects[gl_InstanceIndex].NormalWorld) * Normal, vec3(0.0, 1.0, 0.0)));
    VertexColor = DecodeColor();
    MaterialUV = vec2(UV.x, 1.0 - UV.y);
#ifndef EVANS_INSTANCE_DEFERRED
    LightPosition = Material.LightProjection * world;
#endif
#ifdef EVANS_INSTANCE_GOURAUD
    Illuminate();
#endif
}
#endif
