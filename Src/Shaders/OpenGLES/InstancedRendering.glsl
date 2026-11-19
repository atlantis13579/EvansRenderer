#version 300 es
// Karis' smooth finite-radius window; Filament, Punctual lights.
// Explicit highp also permits this helper before either stage's declarations.
highp float LightRangeWindow(highp vec3 towardEmitter, highp float range)
{
    if (range == 0.0) return 1.0;
    highp vec3 absoluteOffset = abs(towardEmitter);
    if (max(max(absoluteOffset.x, absoluteOffset.y), absoluteOffset.z) >= range) return 0.0;
    highp vec3 relativeOffset = towardEmitter / range;
    highp float squaredDistance = dot(relativeOffset, relativeOffset);
    highp float fade = max(1.0 - squaredDistance * squaredDistance, 0.0);
    return fade * fade;
}

// Independent ES vertex implementation of rigid constant-array instancing.
// Dudash, "Animated Crowd Rendering", GPU Gems 3 (2007), chapter 2.
// gl_InstanceID selects one object's current, normal and previous transforms.
// std140 row-major storage matches the CPU ABI, including nested matrices.
// GLSL ES 3.00, section 4.3.8.3; no vertex stream duplication or skinning.
precision highp float;
precision highp int;
// Preserve camera-depth replay when geometry is submitted through instancing.
invariant gl_Position;

const int BatchCapacity = 64;
const float DirectionTolerance = 1.0e-12;
const float ClipTolerance = 1.0e-7;
const float SrgbBreak = 0.04045;
const float SrgbLinearDivisor = 12.92;
const float SrgbOffset = 0.055;
const float SrgbDivisor = 1.055;
const float SrgbPower = 2.4;
struct RigidObject
{
    mat4 World;
    mat4 Normal;
    mat4 PreviousClip;
    vec4 Wind, Gust, VegetationHistory;
};
layout(std140, row_major) uniform InstanceBlock
{
    mat4 BatchProjection;
    vec4 BatchMaterial;
    RigidObject Objects[BatchCapacity];
};
layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec4 Color;
layout(location = 3) in vec2 TextureCoordinate;


// Homogeneous screen area from each instance's transform and the original plane.
layout(location=15) in highp vec4 WireFacePlane;
uniform highp int WireFaceCull;
uniform highp int WireFaceSign;
void CullWireFace()
{
    if (WireFaceCull==0) return;
    highp mat4 clip=BatchProjection*Objects[gl_InstanceID].World;
    highp vec3 a=clip[0].xyw,b=clip[1].xyw,c=clip[2].xyw,d=clip[3].xyw;
    highp vec4 camera=vec4(dot(b,cross(c,d)),-dot(a,cross(c,d)),dot(a,cross(b,d)),-dot(a,cross(b,c)));
    if (dot(WireFacePlane,camera)*float(WireFaceSign)<=0.0)
        gl_Position=vec4(0.0,0.0,2.0,1.0);
}

vec3 UnitVector(vec3 value, vec3 fallback)
{
    float squaredLength = dot(value, value);
    return squaredLength > DirectionTolerance ? value * inversesqrt(squaredLength) : fallback;
}
vec4 NativeClip(vec4 clip)
{
    // Transform only the raster depth from [0,w] to ES's [-w,w]. Previous
    // positions retain the renderer's convention for velocity reconstruction.
    clip.z = clip.z * 2.0 - clip.w;
    return clip;
}
vec4 LinearVertexColor()
{
    vec3 encoded = clamp(Color.rgb, 0.0, 1.0);
    vec3 linearBranch = encoded / SrgbLinearDivisor;
    vec3 curvedBranch = pow((encoded + SrgbOffset) / SrgbDivisor, vec3(SrgbPower));
    return vec4(mix(curvedBranch, linearBranch, lessThanEqual(encoded, vec3(SrgbBreak))), Color.a);
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
    RigidObject data=Objects[gl_InstanceID];
    if (data.Gust.w<0.5) return vec4(Position,1.0);
    vec3 world=(data.World*vec4(Position,1.0)).xyz;
    float height=max(0.0,data.Gust.w<1.5?TextureCoordinate.y:Position.y);
    vec3 offset=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,
        previous?data.VegetationHistory.x:data.Wind.z);
    return vec4(Position+transpose(mat3(data.Normal))*offset,1.0);
}
vec3 VegetationNormal(vec3 normal)
{
    RigidObject data=Objects[gl_InstanceID];
    if (data.Gust.w<0.5) return normal;
    float height=max(0.0,data.Gust.w<1.5?TextureCoordinate.y:Position.y);
    vec3 world=(data.World*vec4(Position,1.0)).xyz;
    vec3 derivative=VegetationOffset(world,height,data.Wind,data.Gust,data.VegetationHistory.y,data.Wind.z)*2.0/max(height,0.01);
    normal.y-=dot(normal.xz,derivative.xz);
    return normalize(normal);
}
#if defined(EVANS_INSTANCE_SHADOW) || defined(EVANS_INSTANCE_CASCADE)
out vec2 MaterialUV;
#ifdef EVANS_INSTANCE_CASCADE
out float VertexOpacity;
#else
out float VertexAlpha;
#endif
void main()
{
    // Match the native depth producer's matrix association for identical depth.
    gl_Position = NativeClip(BatchProjection * Objects[gl_InstanceID].World * VegetationPosition(false));
    MaterialUV = vec2(TextureCoordinate.x, 1.0 - TextureCoordinate.y);
#ifdef EVANS_INSTANCE_CASCADE
    VertexOpacity = Color.a;
#else
    VertexAlpha = Color.a;
#endif
    CullWireFace();
}
#elif defined(EVANS_INSTANCE_VELOCITY)
out vec4 PreviousClip;
out vec2 SurfaceUV;
out float SurfaceAlpha;
void main()
{
    vec4 world = Objects[gl_InstanceID].World * VegetationPosition(false);
    gl_Position = NativeClip(BatchProjection * world);
    // An invalid object history has a zero matrix, so the existing velocity
    // fragment stage rejects its clip W without disabling neighboring objects.
    PreviousClip = Objects[gl_InstanceID].PreviousClip * VegetationPosition(true);
    SurfaceUV = vec2(TextureCoordinate.x, 1.0 - TextureCoordinate.y);
    SurfaceAlpha = Color.a * BatchMaterial.x;
    CullWireFace();
}
#elif defined(EVANS_INSTANCE_DEFERRED)
out vec3 SurfacePosition;
out vec3 SurfaceNormal;
out vec4 VertexColor;
out vec2 UV;
void main()
{
    vec4 world = Objects[gl_InstanceID].World * VegetationPosition(false);
    gl_Position = NativeClip(BatchProjection * world);
    SurfacePosition = world.xyz / max(abs(world.w), ClipTolerance);
    SurfaceNormal = VegetationNormal(UnitVector(mat3(Objects[gl_InstanceID].Normal) * Normal, vec3(0.0, 1.0, 0.0)));
    VertexColor = LinearVertexColor();
    UV = vec2(TextureCoordinate.x, 1.0 - TextureCoordinate.y);
    CullWireFace();
}
#else
uniform mat4 LightViewProjection;
out vec3 WorldPosition;
out vec3 WorldNormal;
out vec4 VertexColor;
out vec4 LightPosition;
out vec2 TexCoord;

#ifdef EVANS_INSTANCE_GOURAUD
const int LightCapacity = 4;
const int DirectionalEmitter = 0;
const int PointEmitter = 1;
const int RectangleEmitter = 2;
const int RectangleSamples = 4;
const float RectangleOffset = 0.25;
const float MinimumLightDistanceSquared = 1.0e-6;
const float ParallelAxisLimit = 0.999;
const float IntegerRoundOffset = 0.5;
// Keep this uniform type identical to the ES Gouraud fragment interface.
struct LightConstants
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
uniform LightConstants Lights[LightCapacity];
uniform vec4 CameraLightCount;
uniform vec4 MaterialParameters;
uniform vec4 RenderParameters;
out vec3 VertexDiffuseFront;
out vec3 VertexSpecularFront;
out vec3 VertexDiffuseBack;
out vec3 VertexSpecularBack;
out vec3 ShadowDiffuseFront;
out vec3 ShadowSpecularFront;
out vec3 ShadowDiffuseBack;
out vec3 ShadowSpecularBack;

void IlluminateVertex(vec3 position, vec3 normal, out vec3 diffuse, out vec3 specular,
    out vec3 shadowDiffuse, out vec3 shadowSpecular)
{
    diffuse = specular = shadowDiffuse = shadowSpecular = vec3(0.0);
    vec3 eyeDirection = UnitVector(CameraLightCount.xyz - position, normal);
    for (int lightIndex = 0; lightIndex < LightCapacity; ++lightIndex)
    {
        if (lightIndex >= int(CameraLightCount.w)) break;
        LightConstants light = Lights[lightIndex];
        int kind = int(light.PositionType.w + IntegerRoundOffset);
        vec3 emission = UnitVector(light.DirectionIntensity.xyz, vec3(0.0, -1.0, 0.0));
        vec3 helper = abs(emission.y) < ParallelAxisLimit ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent = UnitVector(cross(helper, emission), vec3(1.0, 0.0, 0.0));
        vec3 bitangent = cross(emission, tangent);
        int count = kind == RectangleEmitter ? RectangleSamples : 1;
        for (int sampleIndex = 0; sampleIndex < RectangleSamples; ++sampleIndex)
        {
            if (sampleIndex >= count) break;
            vec2 quadrant = vec2(sampleIndex % 2 == 0 ? -1.0 : 1.0,
                sampleIndex / 2 == 0 ? -1.0 : 1.0) * RectangleOffset;
            vec3 samplePosition = light.PositionType.xyz + tangent * quadrant.x * light.AreaSizePadding.x +
                bitangent * quadrant.y * light.AreaSizePadding.y;
            vec3 towardLight = kind == DirectionalEmitter ? light.DirectionIntensity.xyz :
                (kind == PointEmitter ? light.PositionType.xyz : samplePosition) - position;
            float squaredDistance = max(dot(towardLight, towardLight), MinimumLightDistanceSquared);
            vec3 direction = UnitVector(towardLight, normal);
            float attenuation = kind == DirectionalEmitter ? 1.0 : LightRangeWindow(towardLight, light.AreaSizePadding.z) / squaredDistance;
            if (kind == RectangleEmitter) attenuation *= max(dot(emission, -direction), 0.0) / float(RectangleSamples);
            vec3 radiance = max(light.ColorCastShadows.rgb, vec3(0.0)) * max(light.DirectionIntensity.w, 0.0) * attenuation;
            float cosine = max(dot(normal, direction), 0.0);
            vec3 diffuseSample = radiance * cosine;
            vec3 specularSample = cosine > 0.0 ? radiance * pow(max(dot(normal,
                UnitVector(direction + eyeDirection, normal)), 0.0), max(MaterialParameters.x, 1.0)) : vec3(0.0);
            diffuse += diffuseSample;
            specular += specularSample;
            // The ES pixel stage subtracts only this light's occluded portion
            // from total irradiance; it does not expect a separate unlit sum.
            if (lightIndex == int(RenderParameters.z))
            {
                shadowDiffuse += diffuseSample;
                shadowSpecular += specularSample;
            }
        }
    }
}
#endif
void main()
{
    vec4 world = Objects[gl_InstanceID].World * VegetationPosition(false);
    gl_Position = NativeClip(BatchProjection * world);
    WorldPosition = world.xyz / max(abs(world.w), ClipTolerance);
    WorldNormal = VegetationNormal(UnitVector(mat3(Objects[gl_InstanceID].Normal) * Normal, vec3(0.0, 1.0, 0.0)));
    VertexColor = LinearVertexColor();
    LightPosition = LightViewProjection * world;
    TexCoord = vec2(TextureCoordinate.x, 1.0 - TextureCoordinate.y);
#ifdef EVANS_INSTANCE_GOURAUD
    IlluminateVertex(WorldPosition, WorldNormal, VertexDiffuseFront, VertexSpecularFront, ShadowDiffuseFront, ShadowSpecularFront);
    IlluminateVertex(WorldPosition, -WorldNormal, VertexDiffuseBack, VertexSpecularBack, ShadowDiffuseBack, ShadowSpecularBack);
#endif
    CullWireFace();
}
#endif
