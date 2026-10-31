#include "Renderer/OpenGLESRHI.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/RealtimeShader.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace Evans::Renderer
{
#if defined(_WIN32)
namespace
{
#define EVANS_EGL_CALL __stdcall
#define EVANS_GL_CALL __stdcall

using EGLBoolean = unsigned int;
using EGLenum = unsigned int;
using EGLint = int;
using EGLDisplay = void*;
using EGLConfig = void*;
using EGLSurface = void*;
using EGLContext = void*;
using EGLNativeDisplayType = void*;

constexpr EGLBoolean EGL_FALSE_VALUE = 0;
constexpr EGLint EGL_NONE_VALUE = 0x3038;
constexpr EGLint EGL_RED_SIZE = 0x3024;
constexpr EGLint EGL_GREEN_SIZE = 0x3023;
constexpr EGLint EGL_BLUE_SIZE = 0x3022;
constexpr EGLint EGL_ALPHA_SIZE = 0x3021;
constexpr EGLint EGL_SURFACE_TYPE = 0x3033;
constexpr EGLint EGL_PBUFFER_BIT = 0x0001;
constexpr EGLint EGL_RENDERABLE_TYPE = 0x3040;
constexpr EGLint EGL_OPENGL_ES3_BIT = 0x0040;
constexpr EGLint EGL_WIDTH = 0x3057;
constexpr EGLint EGL_HEIGHT = 0x3056;
constexpr EGLint EGL_CONTEXT_CLIENT_VERSION = 0x3098;
constexpr EGLenum EGL_OPENGL_ES_API = 0x30A0;

using GLenum = unsigned int;
using GLboolean = unsigned char;
using GLbitfield = unsigned int;
using GLbyte = signed char;
using GLsizei = int;
using GLint = int;
using GLuint = unsigned int;
using GLfloat = float;
using GLchar = char;
using GLintptr = std::ptrdiff_t;
using GLsizeiptr = std::ptrdiff_t;

constexpr GLboolean GL_FALSE_VALUE = 0;
constexpr GLboolean GL_TRUE_VALUE = 1;
constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_VERSION = 0x1F02;
constexpr GLenum GL_MAX_TEXTURE_SIZE = 0x0D33;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_ELEMENT_ARRAY_BUFFER = 0x8893;
constexpr GLenum GL_STATIC_DRAW = 0x88E4;
constexpr GLenum GL_DYNAMIC_DRAW = 0x88E8;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_UNSIGNED_INT = 0x1405;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_LINES = 0x0001;
constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
constexpr GLenum GL_TEXTURE0 = 0x84C0;
constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
constexpr GLenum GL_NEAREST = 0x2600;
constexpr GLenum GL_LINEAR = 0x2601;
constexpr GLenum GL_REPEAT = 0x2901;
constexpr GLenum GL_CLAMP_TO_EDGE = 0x812F;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_RGBA8 = 0x8058;
constexpr GLenum GL_SRGB8_ALPHA8 = 0x8C43;
constexpr GLenum GL_DEPTH_COMPONENT = 0x1902;
constexpr GLenum GL_DEPTH_COMPONENT32F = 0x8CAC;
constexpr GLenum GL_FRAMEBUFFER = 0x8D40;
constexpr GLenum GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
constexpr GLenum GL_COLOR_ATTACHMENT0 = 0x8CE0;
constexpr GLenum GL_DEPTH_ATTACHMENT = 0x8D00;
constexpr GLenum GL_NONE_VALUE = 0;
constexpr GLbitfield GL_COLOR_BUFFER_BIT = 0x00004000;
constexpr GLbitfield GL_DEPTH_BUFFER_BIT = 0x00000100;
constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_CULL_FACE = 0x0B44;
constexpr GLenum GL_DITHER = 0x0BD0;
constexpr GLenum GL_BLEND = 0x0BE2;
constexpr GLenum GL_POLYGON_OFFSET_FILL = 0x8037;
constexpr GLenum GL_FRONT = 0x0404;
constexpr GLenum GL_CCW = 0x0901;
constexpr GLenum GL_LESS = 0x0201;
constexpr GLenum GL_LEQUAL = 0x0203;
constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
constexpr GLenum GL_ONE = 1;
constexpr GLenum GL_SRC_ALPHA = 0x0302;
constexpr GLenum GL_ONE_MINUS_SRC_ALPHA = 0x0303;

using PFNEGLGETDISPLAY = EGLDisplay(EVANS_EGL_CALL*)(EGLNativeDisplayType);
using PFNEGLINITIALIZE = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLint*, EGLint*);
using PFNEGLCHOOSECONFIG = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*);
using PFNEGLBINDAPI = EGLBoolean(EVANS_EGL_CALL*)(EGLenum);
using PFNEGLCREATEPBUFFERSURFACE = EGLSurface(EVANS_EGL_CALL*)(EGLDisplay, EGLConfig, const EGLint*);
using PFNEGLCREATECONTEXT = EGLContext(EVANS_EGL_CALL*)(EGLDisplay, EGLConfig, EGLContext, const EGLint*);
using PFNEGLMAKECURRENT = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
using PFNEGLDESTROYCONTEXT = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLContext);
using PFNEGLDESTROYSURFACE = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLSurface);
using PFNEGLTERMINATE = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay);
using PFNEGLGETERROR = EGLint(EVANS_EGL_CALL*)();
using PFNEGLGETPROCADDRESS = void*(EVANS_EGL_CALL*)(const char*);

using PFNGLGETERROR = GLenum(EVANS_GL_CALL*)();
using PFNGLGETSTRING = const unsigned char*(EVANS_GL_CALL*)(GLenum);
using PFNGLGETINTEGERV = void(EVANS_GL_CALL*)(GLenum, GLint*);
using PFNGLCREATESHADER = GLuint(EVANS_GL_CALL*)(GLenum);
using PFNGLSHADERSOURCE = void(EVANS_GL_CALL*)(GLuint, GLsizei, const GLchar* const*, const GLint*);
using PFNGLCOMPILESHADER = void(EVANS_GL_CALL*)(GLuint);
using PFNGLGETSHADERIV = void(EVANS_GL_CALL*)(GLuint, GLenum, GLint*);
using PFNGLGETSHADERINFOLOG = void(EVANS_GL_CALL*)(GLuint, GLsizei, GLsizei*, GLchar*);
using PFNGLDELETESHADER = void(EVANS_GL_CALL*)(GLuint);
using PFNGLCREATEPROGRAM = GLuint(EVANS_GL_CALL*)();
using PFNGLATTACHSHADER = void(EVANS_GL_CALL*)(GLuint, GLuint);
using PFNGLLINKPROGRAM = void(EVANS_GL_CALL*)(GLuint);
using PFNGLGETPROGRAMIV = void(EVANS_GL_CALL*)(GLuint, GLenum, GLint*);
using PFNGLGETPROGRAMINFOLOG = void(EVANS_GL_CALL*)(GLuint, GLsizei, GLsizei*, GLchar*);
using PFNGLDELETEPROGRAM = void(EVANS_GL_CALL*)(GLuint);
using PFNGLUSEPROGRAM = void(EVANS_GL_CALL*)(GLuint);
using PFNGLGETUNIFORMLOCATION = GLint(EVANS_GL_CALL*)(GLuint, const GLchar*);
using PFNGLUNIFORMMATRIX4FV = void(EVANS_GL_CALL*)(GLint, GLsizei, GLboolean, const GLfloat*);
using PFNGLUNIFORM4F = void(EVANS_GL_CALL*)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
using PFNGLUNIFORM1I = void(EVANS_GL_CALL*)(GLint, GLint);
using PFNGLGENBUFFERS = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDBUFFER = void(EVANS_GL_CALL*)(GLenum, GLuint);
using PFNGLBUFFERDATA = void(EVANS_GL_CALL*)(GLenum, GLsizeiptr, const void*, GLenum);
using PFNGLBUFFERSUBDATA = void(EVANS_GL_CALL*)(GLenum, GLintptr, GLsizeiptr, const void*);
using PFNGLDELETEBUFFERS = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLGENVERTEXARRAYS = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDVERTEXARRAY = void(EVANS_GL_CALL*)(GLuint);
using PFNGLDELETEVERTEXARRAYS = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLENABLEVERTEXATTRIBARRAY = void(EVANS_GL_CALL*)(GLuint);
using PFNGLVERTEXATTRIBPOINTER = void(EVANS_GL_CALL*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
using PFNGLGENTEXTURES = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDTEXTURE = void(EVANS_GL_CALL*)(GLenum, GLuint);
using PFNGLTEXIMAGE2D = void(EVANS_GL_CALL*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
using PFNGLTEXPARAMETERI = void(EVANS_GL_CALL*)(GLenum, GLenum, GLint);
using PFNGLDELETETEXTURES = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLACTIVETEXTURE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLGENFRAMEBUFFERS = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDFRAMEBUFFER = void(EVANS_GL_CALL*)(GLenum, GLuint);
using PFNGLFRAMEBUFFERTEXTURE2D = void(EVANS_GL_CALL*)(GLenum, GLenum, GLenum, GLuint, GLint);
using PFNGLCHECKFRAMEBUFFERSTATUS = GLenum(EVANS_GL_CALL*)(GLenum);
using PFNGLDELETEFRAMEBUFFERS = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLDRAWBUFFERS = void(EVANS_GL_CALL*)(GLsizei, const GLenum*);
using PFNGLREADBUFFER = void(EVANS_GL_CALL*)(GLenum);
using PFNGLVIEWPORT = void(EVANS_GL_CALL*)(GLint, GLint, GLsizei, GLsizei);
using PFNGLCLEARCOLOR = void(EVANS_GL_CALL*)(GLfloat, GLfloat, GLfloat, GLfloat);
using PFNGLCLEARDEPTHF = void(EVANS_GL_CALL*)(GLfloat);
using PFNGLCLEAR = void(EVANS_GL_CALL*)(GLbitfield);
using PFNGLENABLE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLDISABLE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLCULLFACE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLFRONTFACE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLDEPTHMASK = void(EVANS_GL_CALL*)(GLboolean);
using PFNGLDEPTHFUNC = void(EVANS_GL_CALL*)(GLenum);
using PFNGLCOLORMASK = void(EVANS_GL_CALL*)(GLboolean, GLboolean, GLboolean, GLboolean);
using PFNGLBLENDFUNCSEPARATE = void(EVANS_GL_CALL*)(GLenum, GLenum, GLenum, GLenum);
using PFNGLLINEWIDTH = void(EVANS_GL_CALL*)(GLfloat);
using PFNGLPOLYGONOFFSET = void(EVANS_GL_CALL*)(GLfloat, GLfloat);
using PFNGLDRAWELEMENTS = void(EVANS_GL_CALL*)(GLenum, GLsizei, GLenum, const void*);
using PFNGLDRAWARRAYS = void(EVANS_GL_CALL*)(GLenum, GLint, GLsizei);
using PFNGLPIXELSTOREI = void(EVANS_GL_CALL*)(GLenum, GLint);
using PFNGLREADPIXELS = void(EVANS_GL_CALL*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);

constexpr const char* VertexShaderSourceTemplate = R"GLSL(#version 300 es
precision highp float;
layout(location = 0) in vec3 InPosition;
layout(location = 1) in vec3 InNormal;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
uniform mat4 World;
uniform mat4 NormalWorld;
uniform mat4 ViewProjection;
uniform mat4 LightViewProjection;
uniform vec4 MaterialParameters;
uniform vec4 RenderParameters;
out vec3 WorldPosition;
out vec3 WorldNormal;
out vec4 VertexColor;
out vec4 LightPosition;
out vec2 TexCoord;
vec3 SRGBToLinear(vec3 color)
{
    vec3 low = color / 12.92;
    vec3 high = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, lessThanEqual(color, vec3(0.04045)));
}
vec3 NormalizeSafe(vec3 value, vec3 fallbackValue)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12 ? value * inversesqrt(lengthSquared) : fallbackValue;
}
void {{VERTEX_ENTRY}}()
{
    vec4 world = World * vec4(InPosition, 1.0);
    vec3 normal = NormalizeSafe(
        mat3(NormalWorld) * InNormal,
        vec3(0.0, 1.0, 0.0));
    if (RenderParameters.y > 1.5)
    {
        world.xyz += normal * (MaterialParameters.w * 0.01);
    }
    vec4 clip = ViewProjection * world;
    clip.z = clip.z * 2.0 - clip.w;
    gl_Position = clip;
    WorldPosition = world.xyz / max(abs(world.w), 1.0e-7);
    WorldNormal = normal;
    VertexColor = vec4(SRGBToLinear(clamp(InColor.rgb, 0.0, 1.0)), InColor.a);
    LightPosition = LightViewProjection * world;
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
}
)GLSL";

constexpr const char* FragmentShaderSourceTemplate = R"GLSL(#version 300 es
precision highp float;
precision highp int;
struct LightConstants
{
    vec4 PositionType;
    vec4 DirectionIntensity;
    vec4 ColorCastShadows;
    vec4 AreaSizePadding;
};
in vec3 WorldPosition;
in vec3 WorldNormal;
in vec4 VertexColor;
in vec4 LightPosition;
in vec2 TexCoord;
uniform sampler2D ShadowMap;
uniform sampler2D DiffuseMap;
uniform sampler2D NormalMap;
uniform sampler2D OpacityMap;
uniform vec4 PipelineColor;
uniform vec4 MaterialDiffuseColor;
uniform vec4 MaterialAmbientColor;
uniform vec4 MaterialSpecularColor;
uniform vec4 MaterialEmissiveColor;
uniform vec4 MaterialParameters;
uniform vec4 AmbientColorStrength;
uniform vec4 CameraLightCount;
uniform vec4 RenderParameters;
uniform vec4 TextureFlags;
uniform vec4 PassParameters;
uniform LightConstants Lights[{{MAX_LIGHTS}}];
out vec4 OutColor;

vec3 SRGBToLinear(vec3 color)
{
    vec3 low = color / 12.92;
    vec3 high = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, lessThanEqual(color, vec3(0.04045)));
}
vec3 PrepareDisplayLinear(vec3 color)
{
    color = max(color, vec3(0.0));
    color /= max(1.0, max(color.r, max(color.g, color.b)));
    return clamp(color, 0.0, 1.0);
}
vec3 NormalizeSafe(vec3 value, vec3 fallbackValue)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12 ? value * inversesqrt(lengthSquared) : fallbackValue;
}
vec4 SampleDiffuse()
{
    return TextureFlags.x > 0.5
        ? texture(DiffuseMap, TexCoord)
        : vec4(1.0);
}
float SampleOpacity()
{
    return TextureFlags.z > 0.5
        ? texture(OpacityMap, TexCoord).r
        : 1.0;
}
float SurfaceAlpha(vec4 diffuseSample)
{
    return clamp(
        VertexColor.a *
        MaterialParameters.y *
        diffuseSample.a *
        SampleOpacity(),
        0.0,
        1.0);
}
void ApplyAlphaClip(float alpha)
{
    if (MaterialParameters.z > 0.0 && alpha < MaterialParameters.z) discard;
}
void ApplyShadowAlpha(float alpha)
{
    if (MaterialParameters.z > 0.0)
    {
        if (alpha < MaterialParameters.z) discard;
        return;
    }
    if (RenderParameters.w > 0.5)
    {
        if (alpha <= 0.0) discard;
        if (alpha >= 1.0) return;
        float threshold = fract(
            52.9829189 *
            fract(dot(floor(gl_FragCoord.xy), vec2(0.06711056, 0.00583715))));
        if (alpha < threshold) discard;
    }
}
float SampleShadow(vec3 surfaceNormal, vec3 directionToLight)
{
    if (RenderParameters.x < 0.5 || abs(LightPosition.w) < 1.0e-7)
    {
        return 1.0;
    }
    vec3 projected = LightPosition.xyz / LightPosition.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        projected.z < 0.0 || projected.z > 1.0)
    {
        return 1.0;
    }
    ivec2 size = textureSize(ShadowMap, 0);
    if (size.x <= 0 || size.y <= 0) return 1.0;
    float directionLengthSquared = dot(directionToLight, directionToLight);
    vec3 direction = directionLengthSquared > 1.0e-12
        ? directionToLight * inversesqrt(directionLengthSquared)
        : surfaceNormal;
    float normalToLight = clamp(dot(surfaceNormal, direction), 0.0, 1.0);
    float sineToLight = sqrt(max(1.0 - normalToLight * normalToLight, 0.0));
    float slope = sineToLight / max(normalToLight, 0.15);
    float receiverBias = min(0.0045, 0.00035 + 0.00060 * slope);
    vec2 texelPosition = uv * vec2(size) - 0.5;
    vec2 interpolation = fract(texelPosition);
    ivec2 basePixel = ivec2(floor(texelPosition));
    float occluded = 0.0;
    for (int y = {{SHADOW_MIN_OFFSET}}; y <= {{SHADOW_MAX_OFFSET}}; ++y)
    {
        for (int x = {{SHADOW_MIN_OFFSET}}; x <= {{SHADOW_MAX_OFFSET}}; ++x)
        {
            ivec2 pixel00 = clamp(basePixel + ivec2(x, y), ivec2(0), size - ivec2(1));
            ivec2 pixel10 = clamp(pixel00 + ivec2(1, 0), ivec2(0), size - ivec2(1));
            ivec2 pixel01 = clamp(pixel00 + ivec2(0, 1), ivec2(0), size - ivec2(1));
            ivec2 pixel11 = clamp(pixel00 + ivec2(1, 1), ivec2(0), size - ivec2(1));
            float comparisonDepth = projected.z - receiverBias;
            float comparison00 = comparisonDepth > texelFetch(ShadowMap, pixel00, 0).r ? 1.0 : 0.0;
            float comparison10 = comparisonDepth > texelFetch(ShadowMap, pixel10, 0).r ? 1.0 : 0.0;
            float comparison01 = comparisonDepth > texelFetch(ShadowMap, pixel01, 0).r ? 1.0 : 0.0;
            float comparison11 = comparisonDepth > texelFetch(ShadowMap, pixel11, 0).r ? 1.0 : 0.0;
            float comparisonTop = mix(comparison00, comparison10, interpolation.x);
            float comparisonBottom = mix(comparison01, comparison11, interpolation.x);
            occluded += mix(comparisonTop, comparisonBottom, interpolation.y);
        }
    }
    return 1.0 - {{SHADOW_STRENGTH}} * occluded / {{SHADOW_SAMPLE_COUNT}}.0;
}
vec3 ResolveSurfaceNormal()
{
    vec3 normal = NormalizeSafe(WorldNormal, vec3(0.0, 1.0, 0.0));
    if (!gl_FrontFacing) normal = -normal;
    if (TextureFlags.y < 0.5) return normal;
    vec3 positionDx = dFdx(WorldPosition);
    vec3 positionDy = dFdy(WorldPosition);
    vec2 texCoordDx = dFdx(TexCoord);
    vec2 texCoordDy = dFdy(TexCoord);
    float determinant =
        texCoordDx.x * texCoordDy.y -
        texCoordDx.y * texCoordDy.x;
    if (abs(determinant) < 1.0e-10) return normal;
    float inverseDeterminant = 1.0 / determinant;
    vec3 tangent =
        (positionDx * texCoordDy.y - positionDy * texCoordDx.y) *
        inverseDeterminant;
    vec3 sourceBitangent =
        (-positionDx * texCoordDy.x + positionDy * texCoordDx.x) *
        inverseDeterminant;
    tangent -= normal * dot(normal, tangent);
    vec3 helper = abs(normal.y) < 0.999
        ? vec3(0.0, 1.0, 0.0)
        : vec3(1.0, 0.0, 0.0);
    tangent = NormalizeSafe(
        tangent,
        NormalizeSafe(cross(helper, normal), vec3(1.0, 0.0, 0.0)));
    float handedness =
        dot(cross(normal, tangent), sourceBitangent) < 0.0 ? -1.0 : 1.0;
    vec3 bitangent = NormalizeSafe(
        cross(normal, tangent) * handedness,
        vec3(0.0, 0.0, 1.0));
    vec3 tangentNormal = texture(NormalMap, TexCoord).xyz * 2.0 - 1.0;
    return NormalizeSafe(
        tangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}
void AccumulateLightSample(
    vec3 lightDirection,
    vec3 radiance,
    float visibility,
    vec3 normal,
    vec3 viewDirection,
    inout vec3 diffuseLight,
    inout vec3 specularLight)
{
    float lengthSquared = dot(lightDirection, lightDirection);
    if (lengthSquared <= 1.0e-12) return;
    vec3 direction = lightDirection * inversesqrt(lengthSquared);
    float normalToLight = max(dot(normal, direction), 0.0);
    if (normalToLight <= 0.0) return;
    vec3 visibleRadiance = radiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    vec3 halfVector = NormalizeSafe(direction + viewDirection, normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0),
        max(MaterialParameters.x, 1.0));
    specularLight += visibleRadiance * specularAmount;
}
void AccumulateSceneLight(
    int lightIndex,
    vec3 normal,
    vec3 viewDirection,
    inout vec3 diffuseLight,
    inout vec3 specularLight)
{
    LightConstants light = Lights[lightIndex];
    int type = int(light.PositionType.w + 0.5);
    vec3 shadowDirection = type == 0
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - WorldPosition;
    bool isShadowLight =
        lightIndex == int(RenderParameters.z) &&
        light.ColorCastShadows.w > 0.5;
    float visibility = isShadowLight
        ? SampleShadow(normal, shadowDirection)
        : 1.0;
    vec3 radiance =
        max(light.ColorCastShadows.rgb, vec3(0.0)) *
        max(light.DirectionIntensity.w, 0.0);
    if (type == 0)
    {
        AccumulateLightSample(
            light.DirectionIntensity.xyz, radiance, visibility,
            normal, viewDirection, diffuseLight, specularLight);
        return;
    }
    if (type == 1)
    {
        vec3 toLight = light.PositionType.xyz - WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), 1.0e-6);
        AccumulateLightSample(
            toLight, radiance / distanceSquared, visibility,
            normal, viewDirection, diffuseLight, specularLight);
        return;
    }
    vec3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz, vec3(0.0, -1.0, 0.0));
    vec3 helper = abs(emissionDirection.y) < 0.999
        ? vec3(0.0, 1.0, 0.0)
        : vec3(1.0, 0.0, 0.0);
    vec3 tangent = NormalizeSafe(
        cross(helper, emissionDirection), vec3(1.0, 0.0, 0.0));
    vec3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent), vec3(0.0, 0.0, 1.0));
    for (int sampleY = 0; sampleY < 2; ++sampleY)
    {
        for (int sampleX = 0; sampleX < 2; ++sampleX)
        {
            float offsetX = sampleX == 0 ? -0.25 : 0.25;
            float offsetY = sampleY == 0 ? -0.25 : 0.25;
            vec3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            vec3 toLight = samplePosition - WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), 1.0e-6);
            vec3 direction = toLight * inversesqrt(distanceSquared);
            float emitterCosine = max(dot(emissionDirection, -direction), 0.0);
            AccumulateLightSample(
                direction,
                radiance * (0.25 * emitterCosine / distanceSquared),
                visibility, normal, viewDirection, diffuseLight, specularLight);
        }
    }
}
void {{FRAGMENT_ENTRY}}()
{
    vec4 diffuseSample = SampleDiffuse();
    float alpha = SurfaceAlpha(diffuseSample);
    if (PassParameters.x < 0.5)
    {
        ApplyShadowAlpha(alpha);
        OutColor = vec4(0.0);
        return;
    }
    ApplyAlphaClip(alpha);
    if (RenderParameters.y > 0.5)
    {
        OutColor = vec4(
            SRGBToLinear(clamp(PipelineColor.rgb, 0.0, 1.0)),
            alpha);
        return;
    }
    vec3 surfaceBase =
        VertexColor.rgb *
        max(PipelineColor.rgb, vec3(0.0)) *
        max(diffuseSample.rgb, vec3(0.0));
    vec3 normal = ResolveSurfaceNormal();
    vec3 viewDirection = NormalizeSafe(
        CameraLightCount.xyz - WorldPosition,
        normal);
    vec3 diffuseLight = vec3(0.0);
    vec3 specularLight = vec3(0.0);
    int lightCount = min(int(CameraLightCount.w + 0.5), 4);
    for (int lightIndex = 0; lightIndex < {{MAX_LIGHTS}}; ++lightIndex)
    {
        if (lightIndex >= lightCount) break;
        AccumulateSceneLight(
            lightIndex, normal, viewDirection,
            diffuseLight, specularLight);
    }
    vec3 color =
        surfaceBase *
            max(MaterialAmbientColor.rgb, vec3(0.0)) *
            max(AmbientColorStrength.rgb, vec3(0.0)) *
            max(AmbientColorStrength.w, 0.0) +
        surfaceBase *
            max(MaterialDiffuseColor.rgb, vec3(0.0)) *
            diffuseLight +
        max(MaterialSpecularColor.rgb, vec3(0.0)) * specularLight +
        max(MaterialEmissiveColor.rgb, vec3(0.0));
    OutColor = vec4(PrepareDisplayLinear(color), alpha);
}
)GLSL";

constexpr const char* DepthReadVertexShaderSource = R"GLSL(#version 300 es
precision highp float;
void main()
{
    vec2 position = gl_VertexID == 0 ? vec2(-1.0, -1.0) :
                    gl_VertexID == 1 ? vec2( 3.0, -1.0) : vec2(-1.0, 3.0);
    gl_Position = vec4(position, 0.0, 1.0);
}
)GLSL";

constexpr const char* DepthReadFragmentShaderSource = R"GLSL(#version 300 es
precision highp float;
precision highp int;
uniform sampler2D DepthSource;
out vec4 OutColor;
void main()
{
    float depth = texelFetch(DepthSource, ivec2(gl_FragCoord.xy), 0).r;
    uint encoded = uint(round(clamp(depth, 0.0, 1.0) * 16777215.0));
    uvec3 bytes = uvec3(encoded & 255u, (encoded >> 8u) & 255u, (encoded >> 16u) & 255u);
    OutColor = vec4(vec3(bytes) / 255.0, 1.0);
}
)GLSL";

[[nodiscard]] std::string HexError(const char* api, unsigned long value)
{
    std::ostringstream stream;
    stream << api << " failed with error 0x" << std::hex << std::uppercase << value;
    return stream.str();
}

[[nodiscard]] std::wstring ModuleDirectory()
{
    std::array<wchar_t, 32768> path{};
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) return {};
    std::wstring result(path.data(), length);
    const std::size_t separator = result.find_last_of(L"\\/");
    return separator == std::wstring::npos ? std::wstring{} : result.substr(0, separator);
}

[[nodiscard]] std::wstring SystemAngleDirectory()
{
    std::array<wchar_t, MAX_PATH> system{};
    const UINT length = GetSystemDirectoryW(system.data(), static_cast<UINT>(system.size()));
    if (length == 0 || length >= system.size()) return {};
    return std::wstring(system.data(), length) + L"\\Microsoft-Edge-WebView";
}

[[nodiscard]] std::array<float, 16> ToColumnMajor(const Matrix4& matrix)
{
    std::array<float, 16> result{};
    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t column = 0; column < 4; ++column)
        {
            result[column * 4 + row] = matrix.Values[row * 4 + column];
        }
    }
    return result;
}

static_assert(sizeof(Vertex) == 48, "OpenGL ES vertex layout must match Renderer::Vertex");
} // namespace

struct OpenGLESRHI::Impl
{
    struct BufferResource
    {
        BufferDesc Description;
        std::vector<std::uint8_t> Bytes;
        GLuint Buffer = 0;
    };

    struct TextureResource
    {
        TextureDesc Description;
        GLuint Texture = 0;
        GLuint Framebuffer = 0;
        GLuint PrivateDepth = 0;
    };

    template <typename Function>
    bool LoadEGL(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetProcAddress(EGLModule, name));
        if (destination != nullptr) return true;
        return Fail(std::string("OpenGL ES EGL entry point is missing: ") + name);
    }

    template <typename Function>
    bool LoadGL(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetProcAddress(GLESModule, name));
        if (destination == nullptr && EglGetProcAddress != nullptr)
        {
            destination = reinterpret_cast<Function>(EglGetProcAddress(name));
        }
        if (destination != nullptr) return true;
        return Fail(std::string("OpenGL ES 3 entry point is missing: ") + name);
    }

    bool Fail(std::string error)
    {
        LastError = std::move(error);
        return false;
    }

    bool CheckThread(const char* operation)
    {
        if (!Initialized) return Fail("OpenGL ES RHI is not initialized");
        if (OwnerThread != std::this_thread::get_id())
        {
            return Fail(std::string(operation) + " must execute on the owning OpenGL ES RHI thread");
        }
        return true;
    }

    bool CheckGL(const char* operation)
    {
        const GLenum error = GlGetError();
        return error == GL_NO_ERROR || Fail(HexError(operation, error));
    }

    bool LoadLibrariesFrom(const std::wstring& directory)
    {
        if (directory.empty()) return false;
        const std::wstring glesPath = directory + L"\\libGLESv2.dll";
        const std::wstring eglPath = directory + L"\\libEGL.dll";
        constexpr DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
        GLESModule = LoadLibraryExW(glesPath.c_str(), nullptr, flags);
        if (GLESModule == nullptr) return false;
        EGLModule = LoadLibraryExW(eglPath.c_str(), nullptr, flags);
        if (EGLModule != nullptr) return true;
        FreeLibrary(GLESModule);
        GLESModule = nullptr;
        return false;
    }

    bool LoadLibraries()
    {
        if (LoadLibrariesFrom(ModuleDirectory()) || LoadLibrariesFrom(SystemAngleDirectory()))
        {
            return true;
        }
        LastError = "OpenGL ES is unavailable: no self-contained EGL/OpenGL ES 3 runtime was found";
        return false;
    }

    bool LoadEGLFunctions()
    {
        return LoadEGL(EglGetDisplay, "eglGetDisplay") &&
               LoadEGL(EglInitialize, "eglInitialize") &&
               LoadEGL(EglChooseConfig, "eglChooseConfig") &&
               LoadEGL(EglBindAPI, "eglBindAPI") &&
               LoadEGL(EglCreatePbufferSurface, "eglCreatePbufferSurface") &&
               LoadEGL(EglCreateContext, "eglCreateContext") &&
               LoadEGL(EglMakeCurrent, "eglMakeCurrent") &&
               LoadEGL(EglDestroyContext, "eglDestroyContext") &&
               LoadEGL(EglDestroySurface, "eglDestroySurface") &&
               LoadEGL(EglTerminate, "eglTerminate") &&
               LoadEGL(EglGetError, "eglGetError") &&
               LoadEGL(EglGetProcAddress, "eglGetProcAddress");
    }

    bool CreateContext()
    {
        Display = EglGetDisplay(nullptr);
        EGLint major = 0;
        EGLint minor = 0;
        if (Display == nullptr || EglInitialize(Display, &major, &minor) == EGL_FALSE_VALUE)
        {
            return Fail(HexError("eglInitialize", static_cast<unsigned long>(EglGetError())));
        }
        const EGLint configAttributes[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_NONE_VALUE};
        EGLint configCount = 0;
        if (EglChooseConfig(Display, configAttributes, &Config, 1, &configCount) == EGL_FALSE_VALUE ||
            configCount == 0)
        {
            return Fail(HexError("eglChooseConfig", static_cast<unsigned long>(EglGetError())));
        }
        if (EglBindAPI(EGL_OPENGL_ES_API) == EGL_FALSE_VALUE)
        {
            return Fail(HexError("eglBindAPI", static_cast<unsigned long>(EglGetError())));
        }
        const EGLint surfaceAttributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE_VALUE};
        Surface = EglCreatePbufferSurface(Display, Config, surfaceAttributes);
        if (Surface == nullptr)
        {
            return Fail(HexError("eglCreatePbufferSurface", static_cast<unsigned long>(EglGetError())));
        }
        const EGLint contextAttributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE_VALUE};
        Context = EglCreateContext(Display, Config, nullptr, contextAttributes);
        if (Context == nullptr)
        {
            return Fail(HexError("eglCreateContext(OpenGL ES 3)", static_cast<unsigned long>(EglGetError())));
        }
        if (EglMakeCurrent(Display, Surface, Surface, Context) == EGL_FALSE_VALUE)
        {
            return Fail(HexError("eglMakeCurrent", static_cast<unsigned long>(EglGetError())));
        }
        return true;
    }

    bool LoadGLFunctions()
    {
        return LoadGL(GlGetError, "glGetError") &&
               LoadGL(GlGetString, "glGetString") &&
               LoadGL(GlGetIntegerv, "glGetIntegerv") &&
               LoadGL(GlCreateShader, "glCreateShader") &&
               LoadGL(GlShaderSource, "glShaderSource") &&
               LoadGL(GlCompileShader, "glCompileShader") &&
               LoadGL(GlGetShaderiv, "glGetShaderiv") &&
               LoadGL(GlGetShaderInfoLog, "glGetShaderInfoLog") &&
               LoadGL(GlDeleteShader, "glDeleteShader") &&
               LoadGL(GlCreateProgram, "glCreateProgram") &&
               LoadGL(GlAttachShader, "glAttachShader") &&
               LoadGL(GlLinkProgram, "glLinkProgram") &&
               LoadGL(GlGetProgramiv, "glGetProgramiv") &&
               LoadGL(GlGetProgramInfoLog, "glGetProgramInfoLog") &&
               LoadGL(GlDeleteProgram, "glDeleteProgram") &&
               LoadGL(GlUseProgram, "glUseProgram") &&
               LoadGL(GlGetUniformLocation, "glGetUniformLocation") &&
               LoadGL(GlUniformMatrix4fv, "glUniformMatrix4fv") &&
               LoadGL(GlUniform4f, "glUniform4f") &&
               LoadGL(GlUniform1i, "glUniform1i") &&
               LoadGL(GlGenBuffers, "glGenBuffers") &&
               LoadGL(GlBindBuffer, "glBindBuffer") &&
               LoadGL(GlBufferData, "glBufferData") &&
               LoadGL(GlBufferSubData, "glBufferSubData") &&
               LoadGL(GlDeleteBuffers, "glDeleteBuffers") &&
               LoadGL(GlGenVertexArrays, "glGenVertexArrays") &&
               LoadGL(GlBindVertexArray, "glBindVertexArray") &&
               LoadGL(GlDeleteVertexArrays, "glDeleteVertexArrays") &&
               LoadGL(GlEnableVertexAttribArray, "glEnableVertexAttribArray") &&
               LoadGL(GlVertexAttribPointer, "glVertexAttribPointer") &&
               LoadGL(GlGenTextures, "glGenTextures") &&
               LoadGL(GlBindTexture, "glBindTexture") &&
               LoadGL(GlTexImage2D, "glTexImage2D") &&
               LoadGL(GlTexParameteri, "glTexParameteri") &&
               LoadGL(GlDeleteTextures, "glDeleteTextures") &&
               LoadGL(GlActiveTexture, "glActiveTexture") &&
               LoadGL(GlGenFramebuffers, "glGenFramebuffers") &&
               LoadGL(GlBindFramebuffer, "glBindFramebuffer") &&
               LoadGL(GlFramebufferTexture2D, "glFramebufferTexture2D") &&
               LoadGL(GlCheckFramebufferStatus, "glCheckFramebufferStatus") &&
               LoadGL(GlDeleteFramebuffers, "glDeleteFramebuffers") &&
               LoadGL(GlDrawBuffers, "glDrawBuffers") &&
               LoadGL(GlReadBuffer, "glReadBuffer") &&
               LoadGL(GlViewport, "glViewport") &&
               LoadGL(GlClearColor, "glClearColor") &&
               LoadGL(GlClearDepthf, "glClearDepthf") &&
               LoadGL(GlClear, "glClear") &&
               LoadGL(GlEnable, "glEnable") &&
               LoadGL(GlDisable, "glDisable") &&
               LoadGL(GlCullFace, "glCullFace") &&
               LoadGL(GlFrontFace, "glFrontFace") &&
               LoadGL(GlDepthMask, "glDepthMask") &&
               LoadGL(GlDepthFunc, "glDepthFunc") &&
               LoadGL(GlColorMask, "glColorMask") &&
               LoadGL(GlBlendFuncSeparate, "glBlendFuncSeparate") &&
               LoadGL(GlLineWidth, "glLineWidth") &&
               LoadGL(GlPolygonOffset, "glPolygonOffset") &&
               LoadGL(GlDrawElements, "glDrawElements") &&
               LoadGL(GlDrawArrays, "glDrawArrays") &&
               LoadGL(GlPixelStorei, "glPixelStorei") &&
               LoadGL(GlReadPixels, "glReadPixels");
    }

    GLuint Compile(GLenum type, const char* source)
    {
        const GLuint shader = GlCreateShader(type);
        if (shader == 0)
        {
            Fail("glCreateShader returned zero");
            return 0;
        }
        GlShaderSource(shader, 1, &source, nullptr);
        GlCompileShader(shader);
        GLint compiled = 0;
        GlGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (compiled != 0) return shader;
        GLint logLength = 0;
        GlGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
        GlGetShaderInfoLog(shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
        LastError = std::string("OpenGL ES shader compilation failed: ") + log.data();
        GlDeleteShader(shader);
        return 0;
    }

    bool CreateProgram()
    {
        if (!PrepareBuiltinRealtimeShader(
                RealtimeShaderLanguage::GLSLES,
                VertexShaderSourceTemplate,
                {RealtimeShaderStage::Vertex},
                sizeof(RealtimeDrawConstants),
                MaximumSceneLights,
                VertexShaderCode,
                LastError) ||
            !PrepareBuiltinRealtimeShader(
                RealtimeShaderLanguage::GLSLES,
                FragmentShaderSourceTemplate,
                {
                    RealtimeShaderStage::Fragment,
                    RealtimeShaderStage::Shadow,
                    RealtimeShaderStage::Outline},
                sizeof(RealtimeDrawConstants),
                MaximumSceneLights,
                FragmentShaderCode,
                LastError))
        {
            return false;
        }
        const GLuint vertex = Compile(
            GL_VERTEX_SHADER, VertexShaderCode.c_str());
        if (vertex == 0) return false;
        const GLuint fragment = Compile(
            GL_FRAGMENT_SHADER, FragmentShaderCode.c_str());
        if (fragment == 0)
        {
            GlDeleteShader(vertex);
            return false;
        }
        Program = GlCreateProgram();
        GlAttachShader(Program, vertex);
        GlAttachShader(Program, fragment);
        GlLinkProgram(Program);
        GlDeleteShader(vertex);
        GlDeleteShader(fragment);
        GLint linked = 0;
        GlGetProgramiv(Program, GL_LINK_STATUS, &linked);
        if (linked == 0)
        {
            GLint logLength = 0;
            GlGetProgramiv(Program, GL_INFO_LOG_LENGTH, &logLength);
            std::vector<char> log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
            GlGetProgramInfoLog(Program, static_cast<GLsizei>(log.size()), nullptr, log.data());
            LastError = std::string("OpenGL ES program link failed: ") + log.data();
            GlDeleteProgram(Program);
            Program = 0;
            return false;
        }
        WorldLocation = GlGetUniformLocation(Program, "World");
        NormalWorldLocation = GlGetUniformLocation(Program, "NormalWorld");
        ViewProjectionLocation = GlGetUniformLocation(Program, "ViewProjection");
        LightViewProjectionLocation = GlGetUniformLocation(Program, "LightViewProjection");
        PipelineColorLocation = GlGetUniformLocation(Program, "PipelineColor");
        MaterialDiffuseColorLocation = GlGetUniformLocation(Program, "MaterialDiffuseColor");
        MaterialAmbientColorLocation = GlGetUniformLocation(Program, "MaterialAmbientColor");
        MaterialSpecularColorLocation = GlGetUniformLocation(Program, "MaterialSpecularColor");
        MaterialEmissiveColorLocation = GlGetUniformLocation(Program, "MaterialEmissiveColor");
        MaterialParametersLocation = GlGetUniformLocation(Program, "MaterialParameters");
        AmbientColorStrengthLocation = GlGetUniformLocation(Program, "AmbientColorStrength");
        CameraLightCountLocation = GlGetUniformLocation(Program, "CameraLightCount");
        RenderParametersLocation = GlGetUniformLocation(Program, "RenderParameters");
        TextureFlagsLocation = GlGetUniformLocation(Program, "TextureFlags");
        PassParametersLocation = GlGetUniformLocation(Program, "PassParameters");
        ShadowMapLocation = GlGetUniformLocation(Program, "ShadowMap");
        DiffuseMapLocation = GlGetUniformLocation(Program, "DiffuseMap");
        NormalMapLocation = GlGetUniformLocation(Program, "NormalMap");
        OpacityMapLocation = GlGetUniformLocation(Program, "OpacityMap");
        const char* lightFields[] = {
            "PositionType",
            "DirectionIntensity",
            "ColorCastShadows",
            "AreaSizePadding"};
        for (std::size_t lightIndex = 0; lightIndex < LightLocations.size(); ++lightIndex)
        {
            for (std::size_t fieldIndex = 0; fieldIndex < LightLocations[lightIndex].size(); ++fieldIndex)
            {
                const std::string name = "Lights[" + std::to_string(lightIndex) +
                    "]." + lightFields[fieldIndex];
                LightLocations[lightIndex][fieldIndex] =
                    GlGetUniformLocation(Program, name.c_str());
            }
        }
        if (WorldLocation < 0 || NormalWorldLocation < 0 ||
            ViewProjectionLocation < 0 || LightViewProjectionLocation < 0 ||
            PipelineColorLocation < 0 || MaterialDiffuseColorLocation < 0 ||
            MaterialAmbientColorLocation < 0 || MaterialSpecularColorLocation < 0 ||
            MaterialEmissiveColorLocation < 0 || MaterialParametersLocation < 0 ||
            AmbientColorStrengthLocation < 0 || CameraLightCountLocation < 0 ||
            RenderParametersLocation < 0 || TextureFlagsLocation < 0 ||
            PassParametersLocation < 0 || ShadowMapLocation < 0 ||
            DiffuseMapLocation < 0 || NormalMapLocation < 0 ||
            OpacityMapLocation < 0)
        {
            return Fail("OpenGL ES program is missing a required uniform");
        }
        for (const auto& light : LightLocations)
        {
            for (GLint location : light)
            {
                if (location < 0) return Fail(
                    "OpenGL ES program is missing a light uniform");
            }
        }
        GlUseProgram(Program);
        const RealtimeShaderProgram& shader =
            BuiltinRealtimeShader().Program;
        GlUniform1i(
            ShadowMapLocation,
            static_cast<GLint>(
                shader.FindResourceByName("ShadowMap")->Binding));
        GlUniform1i(
            DiffuseMapLocation,
            static_cast<GLint>(
                shader.FindResourceByName("DiffuseMap")->Binding));
        GlUniform1i(
            NormalMapLocation,
            static_cast<GLint>(
                shader.FindResourceByName("NormalMap")->Binding));
        GlUniform1i(
            OpacityMapLocation,
            static_cast<GLint>(
                shader.FindResourceByName("OpacityMap")->Binding));
        return CheckGL("CreateProgram");
    }

    bool CreateDepthReadProgram()
    {
        const GLuint vertex = Compile(GL_VERTEX_SHADER, DepthReadVertexShaderSource);
        if (vertex == 0) return false;
        const GLuint fragment = Compile(GL_FRAGMENT_SHADER, DepthReadFragmentShaderSource);
        if (fragment == 0)
        {
            GlDeleteShader(vertex);
            return false;
        }
        DepthReadProgram = GlCreateProgram();
        GlAttachShader(DepthReadProgram, vertex);
        GlAttachShader(DepthReadProgram, fragment);
        GlLinkProgram(DepthReadProgram);
        GlDeleteShader(vertex);
        GlDeleteShader(fragment);
        GLint linked = 0;
        GlGetProgramiv(DepthReadProgram, GL_LINK_STATUS, &linked);
        if (linked == 0)
        {
            GLint logLength = 0;
            GlGetProgramiv(DepthReadProgram, GL_INFO_LOG_LENGTH, &logLength);
            std::vector<char> log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
            GlGetProgramInfoLog(
                DepthReadProgram, static_cast<GLsizei>(log.size()), nullptr, log.data());
            LastError = std::string("OpenGL ES depth readback program link failed: ") + log.data();
            GlDeleteProgram(DepthReadProgram);
            DepthReadProgram = 0;
            return false;
        }
        const GLint sourceLocation = GlGetUniformLocation(DepthReadProgram, "DepthSource");
        if (sourceLocation < 0) return Fail("OpenGL ES depth readback uniform is missing");
        GlUseProgram(DepthReadProgram);
        GlUniform1i(sourceLocation, 0);
        return CheckGL("CreateDepthReadProgram");
    }

    void ConfigureTexture()
    {
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }

    void ConfigureDepthTexture()
    {
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    bool CreateColorDepthTarget(int width, int height)
    {
        DestroyColorDepthTarget();
        GlGenTextures(1, &ColorTexture);
        GlBindTexture(GL_TEXTURE_2D, ColorTexture);
        ConfigureTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        GlGenTextures(1, &DepthTexture);
        GlBindTexture(GL_TEXTURE_2D, DepthTexture);
        ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        GlGenFramebuffers(1, &MainFramebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, MainFramebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ColorTexture, 0);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, DepthTexture, 0);
        const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &colorAttachment);
        GlReadBuffer(GL_COLOR_ATTACHMENT0);
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            DestroyColorDepthTarget();
            return Fail("OpenGL ES could not create the offscreen color/depth framebuffer");
        }
        return CheckGL("CreateColorDepthTarget");
    }

    void DestroyColorDepthTarget()
    {
        if (MainFramebuffer != 0 && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &MainFramebuffer);
        if (DepthTexture != 0 && GlDeleteTextures) GlDeleteTextures(1, &DepthTexture);
        if (ColorTexture != 0 && GlDeleteTextures) GlDeleteTextures(1, &ColorTexture);
        MainFramebuffer = 0;
        DepthTexture = 0;
        ColorTexture = 0;
    }

    bool EnsureDepthReadTarget(int width, int height)
    {
        if (DepthReadFramebuffer != 0 && DepthReadWidth == width && DepthReadHeight == height)
            return true;
        if (DepthReadFramebuffer != 0) GlDeleteFramebuffers(1, &DepthReadFramebuffer);
        if (DepthReadTexture != 0) GlDeleteTextures(1, &DepthReadTexture);
        DepthReadFramebuffer = 0;
        DepthReadTexture = 0;
        DepthReadWidth = 0;
        DepthReadHeight = 0;
        GlGenTextures(1, &DepthReadTexture);
        GlBindTexture(GL_TEXTURE_2D, DepthReadTexture);
        ConfigureTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        GlGenFramebuffers(1, &DepthReadFramebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, DepthReadFramebuffer);
        GlFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, DepthReadTexture, 0);
        const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &colorAttachment);
        GlReadBuffer(GL_COLOR_ATTACHMENT0);
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            return Fail("OpenGL ES depth readback framebuffer is incomplete");
        DepthReadWidth = width;
        DepthReadHeight = height;
        return CheckGL("EnsureDepthReadTarget");
    }

    bool CreateTextureResource(TextureResource& resource, const void* initialData)
    {
        const TextureDesc& desc = resource.Description;
        GlGenTextures(1, &resource.Texture);
        GlBindTexture(GL_TEXTURE_2D, resource.Texture);
        ConfigureTexture();
        if (desc.Format == TextureFormat::Depth32)
        {
            ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, desc.Width, desc.Height, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, initialData);
        }
        else
        {
            const GLint internalFormat = desc.ColorSpace == TextureColorSpace::SRGB
                ? static_cast<GLint>(GL_SRGB8_ALPHA8)
                : static_cast<GLint>(GL_RGBA8);
            GlTexImage2D(GL_TEXTURE_2D, 0, internalFormat, desc.Width, desc.Height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, initialData);
        }
        if (!desc.RenderTarget) return CheckGL("CreateTexture");

        GlGenFramebuffers(1, &resource.Framebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, resource.Framebuffer);
        if (desc.Format == TextureFormat::Depth32)
        {
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, resource.Texture, 0);
            const GLenum noColor = GL_NONE_VALUE;
            GlDrawBuffers(1, &noColor);
            GlReadBuffer(GL_NONE_VALUE);
        }
        else
        {
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, resource.Texture, 0);
            GlGenTextures(1, &resource.PrivateDepth);
            GlBindTexture(GL_TEXTURE_2D, resource.PrivateDepth);
            ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, desc.Width, desc.Height, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                                   resource.PrivateDepth, 0);
            const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
            GlDrawBuffers(1, &colorAttachment);
            GlReadBuffer(GL_COLOR_ATTACHMENT0);
        }
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            return Fail("OpenGL ES texture framebuffer is incomplete");
        }
        return CheckGL("CreateTexture(render target)");
    }

    void DestroyTextureResource(TextureResource& resource)
    {
        if (resource.Framebuffer != 0) GlDeleteFramebuffers(1, &resource.Framebuffer);
        if (resource.PrivateDepth != 0) GlDeleteTextures(1, &resource.PrivateDepth);
        if (resource.Texture != 0) GlDeleteTextures(1, &resource.Texture);
        resource = {};
    }

    void UploadMatrices()
    {
        const std::array<float, 16> world = ToColumnMajor(BoundWorld);
        const std::array<float, 16> normal =
            ToColumnMajor(BuildRealtimeNormalMatrix(BoundWorld));
        const std::array<float, 16> viewProjection = ToColumnMajor(BoundViewProjection);
        const std::array<float, 16> light = ToColumnMajor(BoundLightViewProjection);
        GlUniformMatrix4fv(WorldLocation, 1, GL_FALSE_VALUE, world.data());
        GlUniformMatrix4fv(NormalWorldLocation, 1, GL_FALSE_VALUE, normal.data());
        GlUniformMatrix4fv(ViewProjectionLocation, 1, GL_FALSE_VALUE, viewProjection.data());
        GlUniformMatrix4fv(LightViewProjectionLocation, 1, GL_FALSE_VALUE, light.data());
    }

    void UploadFloat4(GLint location, const Float4& value)
    {
        GlUniform4f(location, value.X, value.Y, value.Z, value.W);
    }

    bool ReadColor(std::vector<std::uint8_t>& output)
    {
        const std::size_t rowBytes = static_cast<std::size_t>(CreateInfo.Width) * 4;
        std::vector<std::uint8_t> bottomUp(rowBytes * static_cast<std::size_t>(CreateInfo.Height));
        GlBindFramebuffer(GL_FRAMEBUFFER, MainFramebuffer);
        GlReadBuffer(GL_COLOR_ATTACHMENT0);
        GlPixelStorei(GL_PACK_ALIGNMENT, 1);
        GlReadPixels(0, 0, CreateInfo.Width, CreateInfo.Height, GL_RGBA, GL_UNSIGNED_BYTE, bottomUp.data());
        if (!CheckGL("glReadPixels(color)")) return false;
        output.resize(bottomUp.size());
        for (int row = 0; row < CreateInfo.Height; ++row)
        {
            std::memcpy(output.data() + static_cast<std::size_t>(row) * rowBytes,
                        bottomUp.data() + static_cast<std::size_t>(CreateInfo.Height - 1 - row) * rowBytes,
                        rowBytes);
        }
        return true;
    }

    bool ReadDepth(GLuint sourceTexture, int width, int height, std::vector<float>& output)
    {
        if (sourceTexture == 0 || !EnsureDepthReadTarget(width, height)) return false;
        GlBindFramebuffer(GL_FRAMEBUFFER, DepthReadFramebuffer);
        GlViewport(0, 0, width, height);
        GlDisable(GL_DEPTH_TEST);
        GlDisable(GL_DITHER);
        GlDepthMask(GL_FALSE_VALUE);
        GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        GlUseProgram(DepthReadProgram);
        GlActiveTexture(GL_TEXTURE0);
        GlBindTexture(GL_TEXTURE_2D, sourceTexture);
        GlBindVertexArray(VertexArray);
        GlDrawArrays(GL_TRIANGLES, 0, 3);
        if (!CheckGL("DepthReadProgram")) return false;
        std::vector<std::uint8_t> bottomUp(
            static_cast<std::size_t>(width) * height * 4U);
        GlReadBuffer(GL_COLOR_ATTACHMENT0);
        GlPixelStorei(GL_PACK_ALIGNMENT, 1);
        GlReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, bottomUp.data());
        if (!CheckGL("glReadPixels(encoded depth)")) return false;
        output.resize(static_cast<std::size_t>(width) * height);
        for (int row = 0; row < height; ++row)
        {
            for (int column = 0; column < width; ++column)
            {
                const std::size_t pixel =
                    (static_cast<std::size_t>(height - 1 - row) * width + column) * 4U;
                const std::uint32_t encoded =
                    static_cast<std::uint32_t>(bottomUp[pixel]) |
                    (static_cast<std::uint32_t>(bottomUp[pixel + 1]) << 8U) |
                    (static_cast<std::uint32_t>(bottomUp[pixel + 2]) << 16U);
                output[static_cast<std::size_t>(row) * width + column] =
                    static_cast<float>(static_cast<double>(encoded) / 16777215.0);
            }
        }
        return true;
    }

    void PresentToWindow(const FrameCapture& capture) const
    {
        if (CreateInfo.NativeWindow == nullptr || !capture.HasColorOutput()) return;
        HWND window = static_cast<HWND>(CreateInfo.NativeWindow);
        if (!IsWindow(window)) return;
        std::vector<std::uint8_t> bgra(capture.ColorRGBA8.size());
        for (std::size_t pixel = 0; pixel < capture.ColorRGBA8.size() / 4; ++pixel)
        {
            bgra[pixel * 4] = capture.ColorRGBA8[pixel * 4 + 2];
            bgra[pixel * 4 + 1] = capture.ColorRGBA8[pixel * 4 + 1];
            bgra[pixel * 4 + 2] = capture.ColorRGBA8[pixel * 4];
            bgra[pixel * 4 + 3] = 255;
        }
        BITMAPINFO bitmap{};
        bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmap.bmiHeader.biWidth = static_cast<LONG>(capture.Width);
        bitmap.bmiHeader.biHeight = -static_cast<LONG>(capture.Height);
        bitmap.bmiHeader.biPlanes = 1;
        bitmap.bmiHeader.biBitCount = 32;
        bitmap.bmiHeader.biCompression = BI_RGB;
        RECT client{};
        GetClientRect(window, &client);
        HDC dc = GetDC(window);
        if (dc != nullptr)
        {
            StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0,
                          static_cast<int>(capture.Width), static_cast<int>(capture.Height),
                          bgra.data(), &bitmap, DIB_RGB_COLORS, SRCCOPY);
            ReleaseDC(window, dc);
        }
    }

    HMODULE EGLModule = nullptr;
    HMODULE GLESModule = nullptr;
    EGLDisplay Display = nullptr;
    EGLConfig Config = nullptr;
    EGLSurface Surface = nullptr;
    EGLContext Context = nullptr;
    PFNEGLGETDISPLAY EglGetDisplay = nullptr;
    PFNEGLINITIALIZE EglInitialize = nullptr;
    PFNEGLCHOOSECONFIG EglChooseConfig = nullptr;
    PFNEGLBINDAPI EglBindAPI = nullptr;
    PFNEGLCREATEPBUFFERSURFACE EglCreatePbufferSurface = nullptr;
    PFNEGLCREATECONTEXT EglCreateContext = nullptr;
    PFNEGLMAKECURRENT EglMakeCurrent = nullptr;
    PFNEGLDESTROYCONTEXT EglDestroyContext = nullptr;
    PFNEGLDESTROYSURFACE EglDestroySurface = nullptr;
    PFNEGLTERMINATE EglTerminate = nullptr;
    PFNEGLGETERROR EglGetError = nullptr;
    PFNEGLGETPROCADDRESS EglGetProcAddress = nullptr;
    PFNGLGETERROR GlGetError = nullptr;
    PFNGLGETSTRING GlGetString = nullptr;
    PFNGLGETINTEGERV GlGetIntegerv = nullptr;
    PFNGLCREATESHADER GlCreateShader = nullptr;
    PFNGLSHADERSOURCE GlShaderSource = nullptr;
    PFNGLCOMPILESHADER GlCompileShader = nullptr;
    PFNGLGETSHADERIV GlGetShaderiv = nullptr;
    PFNGLGETSHADERINFOLOG GlGetShaderInfoLog = nullptr;
    PFNGLDELETESHADER GlDeleteShader = nullptr;
    PFNGLCREATEPROGRAM GlCreateProgram = nullptr;
    PFNGLATTACHSHADER GlAttachShader = nullptr;
    PFNGLLINKPROGRAM GlLinkProgram = nullptr;
    PFNGLGETPROGRAMIV GlGetProgramiv = nullptr;
    PFNGLGETPROGRAMINFOLOG GlGetProgramInfoLog = nullptr;
    PFNGLDELETEPROGRAM GlDeleteProgram = nullptr;
    PFNGLUSEPROGRAM GlUseProgram = nullptr;
    PFNGLGETUNIFORMLOCATION GlGetUniformLocation = nullptr;
    PFNGLUNIFORMMATRIX4FV GlUniformMatrix4fv = nullptr;
    PFNGLUNIFORM4F GlUniform4f = nullptr;
    PFNGLUNIFORM1I GlUniform1i = nullptr;
    PFNGLGENBUFFERS GlGenBuffers = nullptr;
    PFNGLBINDBUFFER GlBindBuffer = nullptr;
    PFNGLBUFFERDATA GlBufferData = nullptr;
    PFNGLBUFFERSUBDATA GlBufferSubData = nullptr;
    PFNGLDELETEBUFFERS GlDeleteBuffers = nullptr;
    PFNGLGENVERTEXARRAYS GlGenVertexArrays = nullptr;
    PFNGLBINDVERTEXARRAY GlBindVertexArray = nullptr;
    PFNGLDELETEVERTEXARRAYS GlDeleteVertexArrays = nullptr;
    PFNGLENABLEVERTEXATTRIBARRAY GlEnableVertexAttribArray = nullptr;
    PFNGLVERTEXATTRIBPOINTER GlVertexAttribPointer = nullptr;
    PFNGLGENTEXTURES GlGenTextures = nullptr;
    PFNGLBINDTEXTURE GlBindTexture = nullptr;
    PFNGLTEXIMAGE2D GlTexImage2D = nullptr;
    PFNGLTEXPARAMETERI GlTexParameteri = nullptr;
    PFNGLDELETETEXTURES GlDeleteTextures = nullptr;
    PFNGLACTIVETEXTURE GlActiveTexture = nullptr;
    PFNGLGENFRAMEBUFFERS GlGenFramebuffers = nullptr;
    PFNGLBINDFRAMEBUFFER GlBindFramebuffer = nullptr;
    PFNGLFRAMEBUFFERTEXTURE2D GlFramebufferTexture2D = nullptr;
    PFNGLCHECKFRAMEBUFFERSTATUS GlCheckFramebufferStatus = nullptr;
    PFNGLDELETEFRAMEBUFFERS GlDeleteFramebuffers = nullptr;
    PFNGLDRAWBUFFERS GlDrawBuffers = nullptr;
    PFNGLREADBUFFER GlReadBuffer = nullptr;
    PFNGLVIEWPORT GlViewport = nullptr;
    PFNGLCLEARCOLOR GlClearColor = nullptr;
    PFNGLCLEARDEPTHF GlClearDepthf = nullptr;
    PFNGLCLEAR GlClear = nullptr;
    PFNGLENABLE GlEnable = nullptr;
    PFNGLDISABLE GlDisable = nullptr;
    PFNGLCULLFACE GlCullFace = nullptr;
    PFNGLFRONTFACE GlFrontFace = nullptr;
    PFNGLDEPTHMASK GlDepthMask = nullptr;
    PFNGLDEPTHFUNC GlDepthFunc = nullptr;
    PFNGLCOLORMASK GlColorMask = nullptr;
    PFNGLBLENDFUNCSEPARATE GlBlendFuncSeparate = nullptr;
    PFNGLLINEWIDTH GlLineWidth = nullptr;
    PFNGLPOLYGONOFFSET GlPolygonOffset = nullptr;
    PFNGLDRAWELEMENTS GlDrawElements = nullptr;
    PFNGLDRAWARRAYS GlDrawArrays = nullptr;
    PFNGLPIXELSTOREI GlPixelStorei = nullptr;
    PFNGLREADPIXELS GlReadPixels = nullptr;
    RHICreateInfo CreateInfo;
    GLuint Program = 0;
    GLuint DepthReadProgram = 0;
    GLuint VertexArray = 0;
    GLuint ScratchIndexBuffer = 0;
    GLuint MainFramebuffer = 0;
    GLuint ColorTexture = 0;
    GLuint DepthTexture = 0;
    GLuint DepthReadFramebuffer = 0;
    GLuint DepthReadTexture = 0;
    int DepthReadWidth = 0;
    int DepthReadHeight = 0;
    GLint WorldLocation = -1;
    GLint NormalWorldLocation = -1;
    GLint ViewProjectionLocation = -1;
    GLint LightViewProjectionLocation = -1;
    GLint PipelineColorLocation = -1;
    GLint MaterialDiffuseColorLocation = -1;
    GLint MaterialAmbientColorLocation = -1;
    GLint MaterialSpecularColorLocation = -1;
    GLint MaterialEmissiveColorLocation = -1;
    GLint MaterialParametersLocation = -1;
    GLint AmbientColorStrengthLocation = -1;
    GLint CameraLightCountLocation = -1;
    GLint RenderParametersLocation = -1;
    GLint TextureFlagsLocation = -1;
    GLint PassParametersLocation = -1;
    GLint ShadowMapLocation = -1;
    GLint DiffuseMapLocation = -1;
    GLint NormalMapLocation = -1;
    GLint OpacityMapLocation = -1;
    std::array<std::array<GLint, 4>, MaximumSceneLights> LightLocations {};
    GLint MaximumTextureSize = 0;
    std::unordered_map<std::uint32_t, BufferResource> Buffers;
    std::unordered_map<std::uint32_t, TextureResource> Textures;
    FrameCapture PendingFrame;
    FrameCapture LastFrame;
    PassDesc CurrentPass;
    PipelineState BoundPipeline;
    Matrix4 BoundWorld;
    Matrix4 BoundViewProjection;
    Matrix4 BoundLightViewProjection;
    BufferHandle BoundVertexBuffer;
    BufferHandle BoundIndexBuffer;
    TextureHandle LastShadowTexture;
    std::string VertexShaderCode;
    std::string FragmentShaderCode;
    std::string LastError;
    std::thread::id OwnerThread;
    std::uint64_t FrameNumber = 0;
    std::uint32_t NextBuffer = 0x47000001u;
    std::uint32_t NextTexture = 0x47800001u;
    bool Initialized = false;
    bool FrameOpen = false;
    bool PassOpen = false;
    bool PipelineBound = false;
    bool WorldBound = false;
    bool ViewProjectionBound = false;
    bool LightViewProjectionBound = false;
};

OpenGLESRHI::OpenGLESRHI() : m_Impl(std::make_unique<Impl>()) {}
OpenGLESRHI::~OpenGLESRHI() = default;

bool OpenGLESRHI::Initialize(const RHICreateInfo& createInfo)
{
    if (m_Impl->Initialized) return m_Impl->Fail("OpenGL ES RHI is already initialized");
    if (createInfo.Width <= 0 || createInfo.Height <= 0)
        return m_Impl->Fail("OpenGL ES RHI dimensions must be positive");
    if (!m_Impl->LoadLibraries() || !m_Impl->LoadEGLFunctions() || !m_Impl->CreateContext() ||
        !m_Impl->LoadGLFunctions())
    {
        Shutdown();
        return false;
    }
    const unsigned char* version = m_Impl->GlGetString(GL_VERSION);
    if (version == nullptr || std::strstr(reinterpret_cast<const char*>(version), "OpenGL ES 3") == nullptr)
    {
        m_Impl->Fail("The detected EGL runtime did not create an OpenGL ES 3 context");
        Shutdown();
        return false;
    }
    m_Impl->GlGetIntegerv(GL_MAX_TEXTURE_SIZE, &m_Impl->MaximumTextureSize);
    if (!m_Impl->CreateProgram() || !m_Impl->CreateDepthReadProgram())
    {
        Shutdown();
        return false;
    }
    m_Impl->GlGenVertexArrays(1, &m_Impl->VertexArray);
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    m_Impl->GlGenBuffers(1, &m_Impl->ScratchIndexBuffer);
    if (!m_Impl->CreateColorDepthTarget(createInfo.Width, createInfo.Height))
    {
        Shutdown();
        return false;
    }
    m_Impl->CreateInfo = createInfo;
    m_Impl->OwnerThread = std::this_thread::get_id();
    m_Impl->Initialized = true;
    m_Impl->LastError.clear();
    return true;
}

void OpenGLESRHI::Shutdown()
{
    if (!m_Impl) return;
    if (m_Impl->Display != nullptr && m_Impl->Context != nullptr && m_Impl->EglMakeCurrent)
    {
        m_Impl->EglMakeCurrent(m_Impl->Display, m_Impl->Surface, m_Impl->Surface, m_Impl->Context);
    }
    if (m_Impl->GlDeleteBuffers)
    {
        for (auto& [handle, resource] : m_Impl->Buffers)
        {
            (void)handle;
            if (resource.Buffer != 0) m_Impl->GlDeleteBuffers(1, &resource.Buffer);
        }
        if (m_Impl->ScratchIndexBuffer != 0)
            m_Impl->GlDeleteBuffers(1, &m_Impl->ScratchIndexBuffer);
    }
    if (m_Impl->GlDeleteTextures && m_Impl->GlDeleteFramebuffers)
    {
        for (auto& [handle, resource] : m_Impl->Textures)
        {
            (void)handle;
            m_Impl->DestroyTextureResource(resource);
        }
        m_Impl->DestroyColorDepthTarget();
        if (m_Impl->DepthReadFramebuffer != 0)
            m_Impl->GlDeleteFramebuffers(1, &m_Impl->DepthReadFramebuffer);
        if (m_Impl->DepthReadTexture != 0)
            m_Impl->GlDeleteTextures(1, &m_Impl->DepthReadTexture);
    }
    if (m_Impl->VertexArray != 0 && m_Impl->GlDeleteVertexArrays)
        m_Impl->GlDeleteVertexArrays(1, &m_Impl->VertexArray);
    if (m_Impl->Program != 0 && m_Impl->GlDeleteProgram) m_Impl->GlDeleteProgram(m_Impl->Program);
    if (m_Impl->DepthReadProgram != 0 && m_Impl->GlDeleteProgram)
        m_Impl->GlDeleteProgram(m_Impl->DepthReadProgram);
    m_Impl->Buffers.clear();
    m_Impl->Textures.clear();
    m_Impl->ScratchIndexBuffer = 0;
    m_Impl->VertexArray = 0;
    m_Impl->Program = 0;
    m_Impl->DepthReadProgram = 0;
    m_Impl->DepthReadFramebuffer = 0;
    m_Impl->DepthReadTexture = 0;
    m_Impl->DepthReadWidth = 0;
    m_Impl->DepthReadHeight = 0;
    if (m_Impl->Display != nullptr && m_Impl->EglMakeCurrent)
        m_Impl->EglMakeCurrent(m_Impl->Display, nullptr, nullptr, nullptr);
    if (m_Impl->Context != nullptr && m_Impl->EglDestroyContext)
        m_Impl->EglDestroyContext(m_Impl->Display, m_Impl->Context);
    if (m_Impl->Surface != nullptr && m_Impl->EglDestroySurface)
        m_Impl->EglDestroySurface(m_Impl->Display, m_Impl->Surface);
    if (m_Impl->Display != nullptr && m_Impl->EglTerminate) m_Impl->EglTerminate(m_Impl->Display);
    m_Impl->Context = nullptr;
    m_Impl->Surface = nullptr;
    m_Impl->Display = nullptr;
    if (m_Impl->EGLModule != nullptr) FreeLibrary(m_Impl->EGLModule);
    if (m_Impl->GLESModule != nullptr) FreeLibrary(m_Impl->GLESModule);
    m_Impl->EGLModule = nullptr;
    m_Impl->GLESModule = nullptr;
    m_Impl->Initialized = false;
    m_Impl->FrameOpen = false;
    m_Impl->PassOpen = false;
    m_Impl->OwnerThread = {};
}

bool OpenGLESRHI::IsInitialized() const { return m_Impl->Initialized; }
RHIBackend OpenGLESRHI::GetBackend() const { return RHIBackend::OpenGLES; }
const char* OpenGLESRHI::GetName() const { return "OpenGL ES 3 Native RHI"; }
RHICapabilities OpenGLESRHI::GetCapabilities() const
{
    RHICapabilities result;
    result.NativeAPI = m_Impl->Initialized;
    result.Wireframe = true;
    result.ShadowMaps = true;
    result.Outline = true;
    result.AsyncCommandRecording = false;
    result.MaximumTextureSize = m_Impl->MaximumTextureSize > 0
        ? static_cast<std::uint32_t>(m_Impl->MaximumTextureSize)
        : 0U;
    return result;
}

bool OpenGLESRHI::Resize(int width, int height)
{
    if (!m_Impl->CheckThread("Resize") || m_Impl->FrameOpen || width <= 0 || height <= 0)
        return m_Impl->Fail("OpenGL ES cannot resize in its current state");
    if (!m_Impl->CreateColorDepthTarget(width, height)) return false;
    m_Impl->CreateInfo.Width = width;
    m_Impl->CreateInfo.Height = height;
    return true;
}

BufferHandle OpenGLESRHI::CreateBuffer(const BufferDesc& desc, const void* initialData)
{
    if (!m_Impl->CheckThread("CreateBuffer") || desc.SizeInBytes == 0 ||
        desc.SizeInBytes > static_cast<std::size_t>(std::numeric_limits<GLsizeiptr>::max()))
    {
        m_Impl->Fail("OpenGL ES buffer description is invalid");
        return {};
    }
    Impl::BufferResource resource;
    resource.Description = desc;
    resource.Bytes.resize(desc.SizeInBytes);
    if (initialData != nullptr) std::memcpy(resource.Bytes.data(), initialData, desc.SizeInBytes);
    const GLenum target = desc.Usage == BufferUsage::Index ? GL_ELEMENT_ARRAY_BUFFER : GL_ARRAY_BUFFER;
    m_Impl->GlGenBuffers(1, &resource.Buffer);
    m_Impl->GlBindBuffer(target, resource.Buffer);
    m_Impl->GlBufferData(target, static_cast<GLsizeiptr>(desc.SizeInBytes), initialData,
                         desc.Dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
    if (!m_Impl->CheckGL("glBufferData"))
    {
        if (resource.Buffer != 0) m_Impl->GlDeleteBuffers(1, &resource.Buffer);
        return {};
    }
    const BufferHandle handle{m_Impl->NextBuffer++};
    m_Impl->Buffers.emplace(handle.Value, std::move(resource));
    return handle;
}

bool OpenGLESRHI::UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size)
{
    if (!m_Impl->CheckThread("UpdateBuffer")) return false;
    auto found = m_Impl->Buffers.find(buffer.Value);
    if (found == m_Impl->Buffers.end() || data == nullptr || size == 0 ||
        offset > found->second.Bytes.size() || size > found->second.Bytes.size() - offset)
        return m_Impl->Fail("OpenGL ES buffer update is invalid");
    std::memcpy(found->second.Bytes.data() + offset, data, size);
    const GLenum target = found->second.Description.Usage == BufferUsage::Index
        ? GL_ELEMENT_ARRAY_BUFFER : GL_ARRAY_BUFFER;
    m_Impl->GlBindBuffer(target, found->second.Buffer);
    m_Impl->GlBufferSubData(target, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), data);
    return m_Impl->CheckGL("glBufferSubData");
}

bool OpenGLESRHI::DestroyBuffer(BufferHandle buffer)
{
    if (!m_Impl->CheckThread("DestroyBuffer")) return false;
    auto found = m_Impl->Buffers.find(buffer.Value);
    if (!buffer || found == m_Impl->Buffers.end()) return m_Impl->Fail("OpenGL ES buffer handle is invalid");
    m_Impl->GlDeleteBuffers(1, &found->second.Buffer);
    m_Impl->Buffers.erase(found);
    return m_Impl->CheckGL("glDeleteBuffers");
}

TextureHandle OpenGLESRHI::CreateTexture(const TextureDesc& desc, const void* initialData)
{
    if (!m_Impl->CheckThread("CreateTexture") || desc.Width <= 0 || desc.Height <= 0 ||
        desc.Width > m_Impl->MaximumTextureSize || desc.Height > m_Impl->MaximumTextureSize)
    {
        m_Impl->Fail("OpenGL ES texture description is invalid");
        return {};
    }
    Impl::TextureResource resource;
    resource.Description = desc;
    if (!m_Impl->CreateTextureResource(resource, initialData))
    {
        m_Impl->DestroyTextureResource(resource);
        return {};
    }
    const TextureHandle handle{m_Impl->NextTexture++};
    m_Impl->Textures.emplace(handle.Value, std::move(resource));
    return handle;
}

bool OpenGLESRHI::DestroyTexture(TextureHandle texture)
{
    if (!m_Impl->CheckThread("DestroyTexture")) return false;
    auto found = m_Impl->Textures.find(texture.Value);
    if (!texture || found == m_Impl->Textures.end()) return m_Impl->Fail("OpenGL ES texture handle is invalid");
    m_Impl->DestroyTextureResource(found->second);
    m_Impl->Textures.erase(found);
    if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
    return m_Impl->CheckGL("DestroyTexture");
}

bool OpenGLESRHI::BeginFrame()
{
    if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
        return m_Impl->Fail("OpenGL ES frame cannot begin in the current state");
    m_Impl->PendingFrame = {};
    m_Impl->PendingFrame.Backend = RHIBackend::OpenGLES;
    m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
    m_Impl->LastShadowTexture = {};
    m_Impl->FrameOpen = true;
    m_Impl->LastError.clear();
    return true;
}

void OpenGLESRHI::AbortFrame()
{
    if (!m_Impl || !m_Impl->Initialized)
    {
        return;
    }
    if (m_Impl->OwnerThread != std::this_thread::get_id())
    {
        m_Impl->LastError =
            "AbortFrame must execute on the owning OpenGL ES RHI thread";
        return;
    }
    m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
    m_Impl->GlColorMask(
        GL_TRUE_VALUE,
        GL_TRUE_VALUE,
        GL_TRUE_VALUE,
        GL_TRUE_VALUE);
    m_Impl->GlDepthMask(GL_TRUE_VALUE);
    m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDisable(GL_BLEND);
    m_Impl->PendingFrame = {};
    m_Impl->CurrentPass = {};
    m_Impl->BoundPipeline = {};
    m_Impl->BoundVertexBuffer = {};
    m_Impl->BoundIndexBuffer = {};
    m_Impl->LastShadowTexture = {};
    m_Impl->FrameOpen = false;
    m_Impl->PassOpen = false;
    m_Impl->PipelineBound = false;
    m_Impl->WorldBound = false;
    m_Impl->ViewProjectionBound = false;
    m_Impl->LightViewProjectionBound = false;
}

bool OpenGLESRHI::BeginPass(const PassDesc& desc)
{
    if (!m_Impl->CheckThread("BeginPass") || !m_Impl->FrameOpen || m_Impl->PassOpen)
        return m_Impl->Fail("OpenGL ES pass cannot begin in the current state");
    GLuint framebuffer = m_Impl->MainFramebuffer;
    int width = m_Impl->CreateInfo.Width;
    int height = m_Impl->CreateInfo.Height;
    if (desc.Target)
    {
        auto target = m_Impl->Textures.find(desc.Target.Value);
        if (target == m_Impl->Textures.end() || target->second.Framebuffer == 0)
            return m_Impl->Fail("OpenGL ES pass target is invalid or not renderable");
        if (desc.Type == PassType::Shadow && target->second.Description.Format != TextureFormat::Depth32)
            return m_Impl->Fail("OpenGL ES shadow pass requires a depth texture");
        framebuffer = target->second.Framebuffer;
        width = target->second.Description.Width;
        height = target->second.Description.Height;
        if (desc.Type == PassType::Shadow) m_Impl->LastShadowTexture = desc.Target;
    }
    else if (desc.Type == PassType::Shadow)
    {
        return m_Impl->Fail("OpenGL ES shadow pass requires a depth texture");
    }
    m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    m_Impl->GlViewport(0, 0, width, height);
    GLbitfield clearMask = 0;
    if (desc.ClearColorEnabled && desc.Type != PassType::Shadow)
    {
        m_Impl->GlClearColor(desc.ClearColor.X, desc.ClearColor.Y, desc.ClearColor.Z, desc.ClearColor.W);
        clearMask |= GL_COLOR_BUFFER_BIT;
    }
    if (desc.ClearDepthEnabled)
    {
        m_Impl->GlClearDepthf(1.0f);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
        clearMask |= GL_DEPTH_BUFFER_BIT;
    }
    if (clearMask != 0) m_Impl->GlClear(clearMask);
    if (!m_Impl->CheckGL("BeginPass")) return false;
    m_Impl->CurrentPass = desc;
    m_Impl->PendingFrame.Passes.push_back({desc, {}});
    m_Impl->PassOpen = true;
    m_Impl->PipelineBound = false;
    m_Impl->WorldBound = false;
    m_Impl->ViewProjectionBound = false;
    m_Impl->LightViewProjectionBound = false;
    m_Impl->BoundVertexBuffer = {};
    m_Impl->BoundIndexBuffer = {};
    return true;
}

bool OpenGLESRHI::BindPipeline(const PipelineState& pipeline)
{
    if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
        return m_Impl->Fail("OpenGL ES pipeline must be bound inside a pass");
    if (pipeline.OutlineThickness < 0.0f) return m_Impl->Fail("OpenGL ES outline thickness is invalid");
    m_Impl->BoundPipeline = pipeline;
    m_Impl->PipelineBound = true;
    return true;
}

bool OpenGLESRHI::BindWorldTransform(const Matrix4& value)
{
    if (!m_Impl->CheckThread("BindWorldTransform") || !m_Impl->PassOpen) return false;
    m_Impl->BoundWorld = value;
    m_Impl->WorldBound = true;
    return true;
}

bool OpenGLESRHI::BindViewProjection(const Matrix4& value)
{
    if (!m_Impl->CheckThread("BindViewProjection") || !m_Impl->PassOpen) return false;
    m_Impl->BoundViewProjection = value;
    m_Impl->ViewProjectionBound = true;
    return true;
}

bool OpenGLESRHI::BindLightViewProjection(const Matrix4& value)
{
    if (!m_Impl->CheckThread("BindLightViewProjection") || !m_Impl->PassOpen) return false;
    m_Impl->BoundLightViewProjection = value;
    m_Impl->LightViewProjectionBound = true;
    return true;
}

bool OpenGLESRHI::BindVertexBuffer(BufferHandle buffer)
{
    if (!m_Impl->CheckThread("BindVertexBuffer") || !m_Impl->PassOpen) return false;
    const auto found = m_Impl->Buffers.find(buffer.Value);
    if (found == m_Impl->Buffers.end() || found->second.Description.Usage != BufferUsage::Vertex)
        return m_Impl->Fail("OpenGL ES vertex buffer is invalid");
    m_Impl->BoundVertexBuffer = buffer;
    return true;
}

bool OpenGLESRHI::BindIndexBuffer(BufferHandle buffer)
{
    if (!m_Impl->CheckThread("BindIndexBuffer") || !m_Impl->PassOpen) return false;
    const auto found = m_Impl->Buffers.find(buffer.Value);
    if (found == m_Impl->Buffers.end() || found->second.Description.Usage != BufferUsage::Index)
        return m_Impl->Fail("OpenGL ES index buffer is invalid");
    m_Impl->BoundIndexBuffer = buffer;
    return true;
}

bool OpenGLESRHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex)
{
    if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound ||
        !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound ||
        !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
        return m_Impl->Fail("OpenGL ES DrawIndexed is missing required state");
    auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
    auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
    if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end() ||
        vertex->second.Bytes.size() % sizeof(Vertex) != 0 ||
        index->second.Bytes.size() % sizeof(std::uint32_t) != 0 || indexCount % 3 != 0)
        return m_Impl->Fail("OpenGL ES DrawIndexed requires complete triangle-list buffers");
    const std::size_t availableIndices = index->second.Bytes.size() / sizeof(std::uint32_t);
    if (firstIndex > availableIndices || indexCount > availableIndices - firstIndex)
        return m_Impl->Fail("OpenGL ES DrawIndexed exceeds the index buffer");
    const std::size_t availableVertices = vertex->second.Bytes.size() / sizeof(Vertex);
    const auto* sourceIndices = reinterpret_cast<const std::uint32_t*>(index->second.Bytes.data());
    for (std::size_t offset = firstIndex; offset < static_cast<std::size_t>(firstIndex) + indexCount; ++offset)
    {
        const std::int64_t resolved = static_cast<std::int64_t>(sourceIndices[offset]) + baseVertex;
        if (resolved < 0 || static_cast<std::uint64_t>(resolved) >= availableVertices)
            return m_Impl->Fail("OpenGL ES DrawIndexed references a vertex outside the vertex buffer");
    }

    m_Impl->GlUseProgram(m_Impl->Program);
    m_Impl->UploadMatrices();
    const PipelineState& pipeline = m_Impl->BoundPipeline;

    GLuint shadowTexture = 0;
    bool shadowAvailable = false;
    if (m_Impl->CurrentPass.Type != PassType::Shadow &&
        pipeline.SamplesShadowMap && m_Impl->LastShadowTexture)
    {
        const auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
        if (shadow == m_Impl->Textures.end() ||
            shadow->second.Description.Format != TextureFormat::Depth32)
            return m_Impl->Fail("OpenGL ES shadow map resource is unavailable");
        shadowTexture = shadow->second.Texture;
        shadowAvailable = true;
    }
    std::array<GLuint, 3> surfaceTextures {};
    auto resolveSurfaceTexture = [&](TextureHandle handle, const char* usage,
                                     GLuint& nativeTexture)
    {
        if (!handle) return true;
        const auto texture = m_Impl->Textures.find(handle.Value);
        if (texture == m_Impl->Textures.end() ||
            texture->second.Description.Format != TextureFormat::RGBA8 ||
            texture->second.Texture == 0)
        {
            return m_Impl->Fail(
                std::string("OpenGL ES ") + usage + " texture is invalid");
        }
        nativeTexture = texture->second.Texture;
        return true;
    };
    if (!resolveSurfaceTexture(pipeline.DiffuseTexture, "diffuse", surfaceTextures[0]) ||
        !resolveSurfaceTexture(pipeline.NormalTexture, "normal", surfaceTextures[1]) ||
        !resolveSurfaceTexture(pipeline.OpacityTexture, "opacity", surfaceTextures[2]))
        return false;

    const RealtimeDrawConstants constants = BuildRealtimeDrawConstants(
        pipeline,
        m_Impl->BoundWorld,
        m_Impl->BoundViewProjection,
        m_Impl->BoundLightViewProjection,
        m_Impl->CurrentPass.Type,
        shadowAvailable,
        surfaceTextures[0] != 0,
        surfaceTextures[1] != 0,
        surfaceTextures[2] != 0);
    m_Impl->UploadFloat4(m_Impl->PipelineColorLocation, constants.PipelineColor);
    m_Impl->UploadFloat4(
        m_Impl->MaterialDiffuseColorLocation,
        constants.MaterialDiffuseColor);
    m_Impl->UploadFloat4(
        m_Impl->MaterialAmbientColorLocation,
        constants.MaterialAmbientColor);
    m_Impl->UploadFloat4(
        m_Impl->MaterialSpecularColorLocation,
        constants.MaterialSpecularColor);
    m_Impl->UploadFloat4(
        m_Impl->MaterialEmissiveColorLocation,
        constants.MaterialEmissiveColor);
    m_Impl->UploadFloat4(
        m_Impl->MaterialParametersLocation,
        constants.MaterialParameters);
    m_Impl->UploadFloat4(
        m_Impl->AmbientColorStrengthLocation,
        constants.AmbientColorStrength);
    m_Impl->UploadFloat4(
        m_Impl->CameraLightCountLocation,
        constants.CameraLightCount);
    m_Impl->UploadFloat4(
        m_Impl->RenderParametersLocation,
        constants.RenderParameters);
    m_Impl->GlUniform4f(
        m_Impl->TextureFlagsLocation,
        static_cast<float>(constants.TextureFlags.X),
        static_cast<float>(constants.TextureFlags.Y),
        static_cast<float>(constants.TextureFlags.Z),
        0.0f);
    m_Impl->GlUniform4f(
        m_Impl->PassParametersLocation,
        static_cast<float>(m_Impl->CurrentPass.Type),
        0.0f, 0.0f, 0.0f);
    for (std::size_t lightIndex = 0; lightIndex < constants.Lights.size(); ++lightIndex)
    {
        const ShaderLight& light = constants.Lights[lightIndex];
        m_Impl->UploadFloat4(
            m_Impl->LightLocations[lightIndex][0],
            light.PositionType);
        m_Impl->UploadFloat4(
            m_Impl->LightLocations[lightIndex][1],
            light.DirectionIntensity);
        m_Impl->UploadFloat4(
            m_Impl->LightLocations[lightIndex][2],
            light.ColorCastShadows);
        m_Impl->UploadFloat4(
            m_Impl->LightLocations[lightIndex][3],
            light.AreaSizePadding);
    }
    m_Impl->GlActiveTexture(GL_TEXTURE0);
    m_Impl->GlBindTexture(GL_TEXTURE_2D, shadowTexture);
    for (std::size_t textureIndex = 0; textureIndex < surfaceTextures.size(); ++textureIndex)
    {
        m_Impl->GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(textureIndex + 1));
        m_Impl->GlBindTexture(GL_TEXTURE_2D, surfaceTextures[textureIndex]);
    }

    if (pipeline.DepthTest)
    {
        m_Impl->GlEnable(GL_DEPTH_TEST);
        m_Impl->GlDepthFunc(pipeline.DepthWrite ? GL_LESS : GL_LEQUAL);
    }
    else
    {
        m_Impl->GlDisable(GL_DEPTH_TEST);
    }
    m_Impl->GlDepthMask(pipeline.DepthWrite ? GL_TRUE_VALUE : GL_FALSE_VALUE);
    const GLboolean colorWrite = pipeline.ColorWrite ? GL_TRUE_VALUE : GL_FALSE_VALUE;
    m_Impl->GlColorMask(colorWrite, colorWrite, colorWrite, colorWrite);
    if (m_Impl->CurrentPass.Type == PassType::Color && pipeline.BlendEnabled)
    {
        m_Impl->GlEnable(GL_BLEND);
        m_Impl->GlBlendFuncSeparate(
            GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_ONE,
            GL_ONE_MINUS_SRC_ALPHA);
    }
    else
    {
        m_Impl->GlDisable(GL_BLEND);
    }
    if (m_Impl->CurrentPass.Type == PassType::Shadow)
    {
        m_Impl->GlEnable(GL_POLYGON_OFFSET_FILL);
        m_Impl->GlPolygonOffset(ShadowSlopeScaledDepthBias, static_cast<float>(ShadowDepthBiasUnits));
    }
    else
    {
        m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    }
    m_Impl->GlFrontFace(GL_CCW);
    if (pipeline.OutlineExtrusion)
    {
        m_Impl->GlEnable(GL_CULL_FACE);
        m_Impl->GlCullFace(GL_FRONT);
    }
    else
    {
        m_Impl->GlDisable(GL_CULL_FACE);
    }

    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    m_Impl->GlBindBuffer(GL_ARRAY_BUFFER, vertex->second.Buffer);
    m_Impl->GlEnableVertexAttribArray(0);
    m_Impl->GlEnableVertexAttribArray(1);
    m_Impl->GlEnableVertexAttribArray(2);
    m_Impl->GlEnableVertexAttribArray(3);
    m_Impl->GlVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE_VALUE, sizeof(Vertex),
                                  reinterpret_cast<const void*>(offsetof(Vertex, Position)));
    m_Impl->GlVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE_VALUE, sizeof(Vertex),
                                  reinterpret_cast<const void*>(offsetof(Vertex, Normal)));
    m_Impl->GlVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE_VALUE, sizeof(Vertex),
                                  reinterpret_cast<const void*>(offsetof(Vertex, Color)));
    m_Impl->GlVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE_VALUE, sizeof(Vertex),
                                  reinterpret_cast<const void*>(offsetof(Vertex, TexCoord)));

    const bool drawLines = pipeline.Fill == FillMode::Wireframe;
    if (drawLines || baseVertex != 0)
    {
        std::vector<std::uint32_t> scratch;
        scratch.reserve(drawLines ? static_cast<std::size_t>(indexCount) * 2 : indexCount);
        for (std::size_t triangle = firstIndex;
             triangle < static_cast<std::size_t>(firstIndex) + indexCount; triangle += 3)
        {
            const std::uint32_t a = static_cast<std::uint32_t>(
                static_cast<std::int64_t>(sourceIndices[triangle]) + baseVertex);
            const std::uint32_t b = static_cast<std::uint32_t>(
                static_cast<std::int64_t>(sourceIndices[triangle + 1]) + baseVertex);
            const std::uint32_t c = static_cast<std::uint32_t>(
                static_cast<std::int64_t>(sourceIndices[triangle + 2]) + baseVertex);
            if (drawLines)
            {
                scratch.insert(scratch.end(), {a, b, b, c, c, a});
            }
            else
            {
                scratch.insert(scratch.end(), {a, b, c});
            }
        }
        m_Impl->GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Impl->ScratchIndexBuffer);
        m_Impl->GlBufferData(GL_ELEMENT_ARRAY_BUFFER,
                             static_cast<GLsizeiptr>(scratch.size() * sizeof(std::uint32_t)),
                             scratch.data(), GL_DYNAMIC_DRAW);
        if (drawLines) m_Impl->GlLineWidth(1.0f);
        m_Impl->GlDrawElements(drawLines ? GL_LINES : GL_TRIANGLES,
                               static_cast<GLsizei>(scratch.size()), GL_UNSIGNED_INT, nullptr);
    }
    else
    {
        m_Impl->GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index->second.Buffer);
        m_Impl->GlDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount), GL_UNSIGNED_INT,
                               reinterpret_cast<const void*>(static_cast<std::size_t>(firstIndex) *
                                                              sizeof(std::uint32_t)));
    }
    if (!m_Impl->CheckGL("glDrawElements")) return false;

    DrawCallRecord record;
    record.VertexBuffer = m_Impl->BoundVertexBuffer;
    record.IndexBuffer = m_Impl->BoundIndexBuffer;
    record.Pipeline = pipeline;
    record.WorldTransform = m_Impl->BoundWorld;
    record.FirstIndex = firstIndex;
    record.IndexCount = indexCount;
    record.BaseVertex = baseVertex;
    m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(record);
    FrameStatistics& statistics = m_Impl->PendingFrame.Statistics;
    ++statistics.DrawCallCount;
    statistics.SubmittedTriangleCount += indexCount / 3;
    statistics.ShadowDrawCallCount += m_Impl->CurrentPass.Type == PassType::Shadow ? 1U : 0U;
    statistics.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1U : 0U;
    return true;
}

bool OpenGLESRHI::EndPass()
{
    if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
        return m_Impl->Fail("OpenGL ES has no open pass");
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    m_Impl->GlDepthMask(GL_TRUE_VALUE);
    m_Impl->PassOpen = false;
    return m_Impl->CheckGL("EndPass");
}

bool OpenGLESRHI::Present()
{
    if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
        return m_Impl->Fail("OpenGL ES frame cannot be presented in the current state");
    m_Impl->PendingFrame.Statistics.PassCount =
        static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size());
    m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
    m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
    const bool captureDepth = IsDepthCaptureRequired() || m_Impl->CreateInfo.CaptureDiagnosticBuffers ||
        m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0;
    if (!m_Impl->ReadColor(m_Impl->PendingFrame.ColorRGBA8)) return false;
    if (captureDepth &&
        !m_Impl->ReadDepth(m_Impl->DepthTexture, m_Impl->CreateInfo.Width,
                           m_Impl->CreateInfo.Height, m_Impl->PendingFrame.Depth32))
        return false;
    if (m_Impl->CreateInfo.CaptureDiagnosticBuffers && m_Impl->LastShadowTexture)
    {
        const auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
        if (shadow != m_Impl->Textures.end())
        {
            m_Impl->PendingFrame.ShadowWidth = static_cast<std::uint32_t>(shadow->second.Description.Width);
            m_Impl->PendingFrame.ShadowHeight = static_cast<std::uint32_t>(shadow->second.Description.Height);
            if (!m_Impl->ReadDepth(shadow->second.Texture, shadow->second.Description.Width,
                                   shadow->second.Description.Height,
                                   m_Impl->PendingFrame.ShadowDepth32))
                return false;
        }
    }
    m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
    m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
    m_Impl->FrameOpen = false;
    m_Impl->PresentToWindow(m_Impl->LastFrame);
    return true;
}

FrameCapture OpenGLESRHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
std::string OpenGLESRHI::GetLastError() const { return m_Impl->LastError; }

#else
struct OpenGLESRHI::Impl
{
    std::string LastError = "OpenGL ES 3 requires an EGL runtime; this platform build has none";
};

OpenGLESRHI::OpenGLESRHI() : m_Impl(std::make_unique<Impl>()) {}
OpenGLESRHI::~OpenGLESRHI() = default;
bool OpenGLESRHI::Initialize(const RHICreateInfo&) { return false; }
void OpenGLESRHI::Shutdown() {}
bool OpenGLESRHI::IsInitialized() const { return false; }
RHIBackend OpenGLESRHI::GetBackend() const { return RHIBackend::OpenGLES; }
const char* OpenGLESRHI::GetName() const { return "OpenGL ES 3 Unavailable"; }
RHICapabilities OpenGLESRHI::GetCapabilities() const { return {}; }
bool OpenGLESRHI::Resize(int, int) { return false; }
BufferHandle OpenGLESRHI::CreateBuffer(const BufferDesc&, const void*) { return {}; }
bool OpenGLESRHI::UpdateBuffer(BufferHandle, std::size_t, const void*, std::size_t) { return false; }
bool OpenGLESRHI::DestroyBuffer(BufferHandle) { return false; }
TextureHandle OpenGLESRHI::CreateTexture(const TextureDesc&, const void*) { return {}; }
bool OpenGLESRHI::DestroyTexture(TextureHandle) { return false; }
bool OpenGLESRHI::BeginFrame() { return false; }
void OpenGLESRHI::AbortFrame() {}
bool OpenGLESRHI::BeginPass(const PassDesc&) { return false; }
bool OpenGLESRHI::BindPipeline(const PipelineState&) { return false; }
bool OpenGLESRHI::BindWorldTransform(const Matrix4&) { return false; }
bool OpenGLESRHI::BindViewProjection(const Matrix4&) { return false; }
bool OpenGLESRHI::BindLightViewProjection(const Matrix4&) { return false; }
bool OpenGLESRHI::BindVertexBuffer(BufferHandle) { return false; }
bool OpenGLESRHI::BindIndexBuffer(BufferHandle) { return false; }
bool OpenGLESRHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
bool OpenGLESRHI::EndPass() { return false; }
bool OpenGLESRHI::Present() { return false; }
FrameCapture OpenGLESRHI::GetLastFrameCapture() const { return {}; }
std::string OpenGLESRHI::GetLastError() const { return m_Impl->LastError; }
#endif
} // namespace Evans::Renderer
