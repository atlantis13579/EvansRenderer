#include "Renderer/DiagnosticBuffer.h"
#include "Renderer/OpenGLES/OpenGLESProgramCache.h"
#include "Renderer/Lumen/LumenHairReceiver.h"
#include "Renderer/Geometry/MeshGeometry.h"
#include "Renderer/Nanite/NaniteAsync.h"
#include "Renderer/MaterialTechnique.h"
#include "Renderer/Shadows/PCFContract.h"
#include "Renderer/Lumen/LumenGlobalDistanceAsync.h"
#include "Renderer/Lumen/LumenLightingAsync.h"
#include "Renderer/Lumen/LumenCaptureAsync.h"
#include "Renderer/Nanite/NaniteMaterials.h"
#include "Renderer/Nanite/NaniteInstances.h"
#include "Renderer/Nanite/NaniteHardware.h"
#include "Renderer/OpenGLES/OpenGLESRHI.h"
#include "Renderer/MaterialTexture.h"
#include "Renderer/Geometry/ParallaxOcclusionMapping.h"
#include "Renderer/Geometry/Instancing.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/RealtimeShader.h"
#include "Renderer/TechniqueShader.h"
#include "Renderer/PostProcessing/PostProcessContract.h"
#include "Renderer/PostProcessing/HdrContract.h"
#include "Renderer/Pipelines/DeferredContract.h"
#include "Renderer/RayTracing/RayTracedAOContract.h"
#include "Renderer/Reflections/ReflectionContract.h"
#include "Renderer/Reflections/CubemapReflectionContract.h"
#include "Renderer/Reflections/EnvironmentRadiance.h"
#include "Renderer/Reflections/EnvironmentMapReflectionContract.h"
#include "Renderer/Reflections/PlanarReflectionContract.h"
#include "Renderer/PostProcessing/DepthOfFieldContract.h"
#include "Renderer/PostProcessing/MotionBlurContract.h"
#include "Renderer/PostProcessing/TemporalAAContract.h"
#include "Renderer/Lighting/GlobalIlluminationContract.h"
#include "Renderer/BakedLighting/LightProbeContract.h"
#include "Renderer/Volumetrics/VolumetricFogContract.h"
#include "Renderer/Volumetrics/VolumetricLightingContract.h"
#include "Renderer/Volumetrics/LightShaftContract.h"
#include "Renderer/Transparency/WeightedTransparencyContract.h"
#include "Renderer/Shadows/CascadedShadowContract.h"
#include "Renderer/Shadows/PCSSContract.h"
#include "Renderer/Shadows/VSMContract.h"
#include "Renderer/OpenGLES/OpenGLESContactShadows.h"
#include "Renderer/OpenGLES/OpenGLESForwardPlus.h"
#include "Renderer/OpenGLES/OpenGLESClusteredForward.h"
#include "Renderer/Lighting/LightListFrame.h"
#include "Renderer/Pipelines/TiledDeferredContract.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <mutex>
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
constexpr EGLint EGL_CONTEXT_MINOR_VERSION = 0x30FB;
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
constexpr GLenum GL_EXTENSIONS = 0x1F03;
constexpr GLenum GL_MAX_TEXTURE_SIZE = 0x0D33;
constexpr GLenum GL_MAX_TEXTURE_IMAGE_UNITS = 0x8872;
constexpr GLenum GL_MAX_DRAW_BUFFERS = 0x8824;
constexpr GLenum GL_MAX_COLOR_ATTACHMENTS = 0x8CDF;
constexpr GLenum GL_MAX_UNIFORM_BLOCK_SIZE = 0x8A30;
constexpr GLenum GL_UNIFORM_BUFFER = 0x8A11;
constexpr GLenum GL_SHADER_STORAGE_BUFFER = 0x90D2;
constexpr GLenum GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS = 0x90DA;
constexpr GLenum GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS = 0x90DD;
constexpr GLenum GL_MAX_SHADER_STORAGE_BLOCK_SIZE = 0x90DE;
constexpr GLenum GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES = 0x8F39;
constexpr GLuint ForwardPlusLightBinding = 0, ForwardPlusHeaderBinding = 1, ForwardPlusIndexBinding = 2;
constexpr GLint ForwardPlusStorageCount = 3;
constexpr std::size_t ForwardPlusFilterCount = static_cast<std::size_t>(Techniques::ShadowFilterMode::VSM) + 1;
enum ForwardPlusProgram : std::size_t { ForwardPlusSurface, ForwardPlusDepth, ForwardPlusWeighted, ForwardPlusProgramCount };
constexpr GLuint GL_INVALID_INDEX_VALUE = 0xFFFFFFFFu;
constexpr GLuint InstanceUniformBinding = 0;
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
constexpr GLenum GL_TEXTURE_2D_ARRAY = 0x8C1A;
constexpr GLenum GL_TEXTURE_CUBE_MAP = 0x8513;
constexpr GLenum GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515;
constexpr GLenum GL_TEXTURE_WRAP_R = 0x8072;
constexpr GLenum GL_LINEAR_MIPMAP_LINEAR = 0x2703;
constexpr GLenum GL_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FE;
constexpr GLenum GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FF;
constexpr GLenum GL_TEXTURE_2D_MULTISAMPLE = 0x9100;
constexpr GLenum GL_NUM_SAMPLE_COUNTS = 0x9380;
constexpr GLenum GL_SAMPLES = 0x80A9;
constexpr GLenum GL_SAMPLE_POSITION = 0x8E50;
constexpr GLenum GL_SAMPLE_MASK = 0x8E51;
constexpr GLenum GL_ALWAYS = 0x0207;
constexpr GLenum GL_TEXTURE0 = 0x84C0;
constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr GLenum GL_TEXTURE_MAX_LEVEL = 0x813D;
constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
constexpr GLenum GL_NEAREST = 0x2600;
constexpr GLenum GL_NEAREST_MIPMAP_NEAREST = 0x2700;
constexpr GLenum GL_LINEAR = 0x2601;
constexpr GLenum GL_REPEAT = 0x2901;
constexpr GLenum GL_CLAMP_TO_EDGE = 0x812F;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_RGBA8 = 0x8058;
constexpr GLenum GL_RGBA16F = 0x881A;
constexpr GLenum GL_RGBA32F = 0x8814;
constexpr GLenum GL_RG32F = 0x8230;
constexpr GLenum GL_RG = 0x8227;
constexpr GLenum GL_COLOR = 0x1800;
constexpr GLenum GL_ZERO = 0;
constexpr GLenum GL_ONE_MINUS_SRC_COLOR = 0x0301;
constexpr GLenum GL_R32F = 0x822E;
constexpr GLenum GL_RED = 0x1903;
constexpr GLenum GL_HALF_FLOAT = 0x140B;
constexpr GLenum GL_SRGB8_ALPHA8 = 0x8C43;
constexpr GLenum GL_DEPTH_COMPONENT = 0x1902;
constexpr GLenum GL_DEPTH_COMPONENT32F = 0x8CAC;
constexpr GLenum GL_FRAMEBUFFER = 0x8D40;
constexpr GLenum GL_READ_FRAMEBUFFER = 0x8CA8;
constexpr GLenum GL_DRAW_FRAMEBUFFER = 0x8CA9;
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
constexpr GLenum GL_BACK = 0x0405;
constexpr GLenum GL_CCW = 0x0901;
constexpr GLenum GL_CW = 0x0900;
constexpr GLenum GL_LESS = 0x0201;
constexpr GLenum GL_EQUAL = 0x0202;
constexpr GLenum GL_LEQUAL = 0x0203;
constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
constexpr GLenum GL_ONE = 1;
constexpr GLenum GL_SRC_ALPHA = 0x0302;
constexpr GLenum GL_ONE_MINUS_SRC_ALPHA = 0x0303;
constexpr GLenum GL_ONE_MINUS_DST_ALPHA = 0x0305;
constexpr GLenum GL_FUNC_ADD = 0x8006;
constexpr GLenum GL_FUNC_REVERSE_SUBTRACT = 0x800B;
constexpr float MaximumHalfFloatRadiance = 65504.0f;

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
using PFNGLUNIFORM4UI = void(EVANS_GL_CALL*)(GLint, GLuint, GLuint, GLuint, GLuint);
using PFNGLUNIFORM1I = void(EVANS_GL_CALL*)(GLint, GLint);
using PFNGLGETUNIFORMBLOCKINDEX = GLuint(EVANS_GL_CALL*)(GLuint, const GLchar*);
using PFNGLUNIFORMBLOCKBINDING = void(EVANS_GL_CALL*)(GLuint, GLuint, GLuint);
using PFNGLBINDBUFFERBASE = void(EVANS_GL_CALL*)(GLenum, GLuint, GLuint);
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
using PFNGLTEXSUBIMAGE2D = void(EVANS_GL_CALL*)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*);
using PFNGLTEXIMAGE2D = void(EVANS_GL_CALL*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
using PFNGLCOPYTEXSUBIMAGE2D = void(EVANS_GL_CALL*)(GLenum, GLint, GLint, GLint, GLint, GLint, GLsizei, GLsizei);
using PFNGLGENERATEMIPMAP = void(EVANS_GL_CALL*)(GLenum);
using PFNGLTEXPARAMETERI = void(EVANS_GL_CALL*)(GLenum, GLenum, GLint);
using PFNGLDELETETEXTURES = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLACTIVETEXTURE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLGENFRAMEBUFFERS = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDFRAMEBUFFER = void(EVANS_GL_CALL*)(GLenum, GLuint);
using PFNGLBLITFRAMEBUFFER = void(EVANS_GL_CALL*)(GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum);
using PFNGLFRAMEBUFFERTEXTURE2D = void(EVANS_GL_CALL*)(GLenum, GLenum, GLenum, GLuint, GLint);
using PFNGLCHECKFRAMEBUFFERSTATUS = GLenum(EVANS_GL_CALL*)(GLenum);
using PFNGLDELETEFRAMEBUFFERS = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLDRAWBUFFERS = void(EVANS_GL_CALL*)(GLsizei, const GLenum*);
using PFNGLREADBUFFER = void(EVANS_GL_CALL*)(GLenum);
using PFNGLVIEWPORT = void(EVANS_GL_CALL*)(GLint, GLint, GLsizei, GLsizei);
using PFNGLCLEARCOLOR = void(EVANS_GL_CALL*)(GLfloat, GLfloat, GLfloat, GLfloat);
using PFNGLCLEARDEPTHF = void(EVANS_GL_CALL*)(GLfloat);
using PFNGLCLEAR = void(EVANS_GL_CALL*)(GLbitfield);
using PFNGLCLEARBUFFERFV = void(EVANS_GL_CALL*)(GLenum, GLint, const GLfloat*);
using PFNGLENABLE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLDISABLE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLCULLFACE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLFRONTFACE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLDEPTHMASK = void(EVANS_GL_CALL*)(GLboolean);
using PFNGLDEPTHFUNC = void(EVANS_GL_CALL*)(GLenum);
using PFNGLCOLORMASK = void(EVANS_GL_CALL*)(GLboolean, GLboolean, GLboolean, GLboolean);
using PFNGLBLENDFUNCSEPARATE = void(EVANS_GL_CALL*)(GLenum, GLenum, GLenum, GLenum);
using PFNGLBLENDFUNCSEPARATEI = void(EVANS_GL_CALL*)(GLuint, GLenum, GLenum, GLenum, GLenum);
using PFNGLBLENDEQUATION = void(EVANS_GL_CALL*)(GLenum);
using PFNGLLINEWIDTH = void(EVANS_GL_CALL*)(GLfloat);
using PFNGLPOLYGONOFFSET = void(EVANS_GL_CALL*)(GLfloat, GLfloat);
using PFNGLDRAWELEMENTS = void(EVANS_GL_CALL*)(GLenum, GLsizei, GLenum, const void*);
using PFNGLDRAWELEMENTSINSTANCED = void(EVANS_GL_CALL*)(GLenum, GLsizei, GLenum, const void*, GLsizei);
using PFNGLDRAWARRAYS = void(EVANS_GL_CALL*)(GLenum, GLint, GLsizei);
using PFNGLPIXELSTOREI = void(EVANS_GL_CALL*)(GLenum, GLint);
using PFNGLTEXSTORAGE2DMULTISAMPLE = void(EVANS_GL_CALL*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLboolean);
using PFNGLGETINTERNALFORMATIV = void(EVANS_GL_CALL*)(GLenum, GLenum, GLenum, GLsizei, GLint*);
using PFNGLGETMULTISAMPLEFV = void(EVANS_GL_CALL*)(GLenum, GLuint, GLfloat*);
using PFNGLSAMPLEMASKI = void(EVANS_GL_CALL*)(GLuint, GLbitfield);
using PFNGLREADPIXELS = void(EVANS_GL_CALL*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);

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
    enum class SampleProgram : std::size_t { Color, Depth, Material, Velocity, Restore, Count };
    std::array<GLuint, static_cast<std::size_t>(SampleProgram::Count)> SamplePrograms {};
    static constexpr std::size_t SampleColor = 0, SampleVelocity = 1, SampleMaterialFirst = 2;
    static constexpr std::size_t SampleColorFBO = 0, SampleLightingFBO = 1, SampleMaterialFBO = 2, SampleVelocityFBO = 3;
    struct MultisampleTargets
    {
        std::array<GLuint, SampleMaterialFirst + GBufferAttachmentCount> Textures {};
        std::array<GLuint, 4> Framebuffers {};
        GLuint Depth = 0;
        int Width = 0, Height = 0;
        std::uint32_t Samples = 1;
        GLenum ColorFormat = GL_RGBA16F;
        std::array<std::array<GLfloat, 2>, 8> Locations {};
    } Multisample;

    struct WeightedTargets
    {
        std::array<GLuint, 2> Textures {};
        GLuint Framebuffer = 0;
        int Width = 0, Height = 0;
        std::uint32_t Samples = 1;
    } Weighted;

    struct BufferResource
    {
        BufferDesc Description;
        bool NaniteGeometry=false;
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

    static constexpr std::size_t VSMRaw = 0, VSMHorizontal = 1, VSMFiltered = 2;
    static constexpr std::size_t VSMCasterStage = 3, VSMFilterStage = 4;
    struct VSMMomentMap
    {
        std::array<GLuint, 3> Textures {}, Framebuffers {};
        int Width = 0, Height = 0;
        bool Ready = false, RawWritten = false;
    };

    // Keep cascades above the deferred material/depth inputs. Four individually
    // named samplers avoid ES 3.0's constant-index restriction on sampler arrays.
    static constexpr GLuint FirstCascadeTextureUnit = 10;
    static constexpr GLuint FogTextureUnit = FirstCascadeTextureUnit + MaximumShadowCascades;
    static constexpr GLuint ContactTextureUnit = FogTextureUnit + 1;
    struct FogUniformLocations
    {
        GLint Camera = -1, Medium = -1, Radiance = -1, Grid = -1, Control = -1, Atlas = -1;
        GLint WaterBounds=-1, WaterSun=-1, WaterIrradiance=-1, WaterWaveA=-1, WaterWaveB=-1;
    };
    struct CascadeUniformLocations
    {
        std::array<GLint, MaximumShadowCascades> Matrices {}, Maps {};
        GLint FarDistances = -1, BlendStarts = -1, EyeDepthPlane = -1, Parameters = -1, Filtering = -1;
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

    bool ContextCurrent=true;
    bool NaniteDeviceAvailable=false,NaniteHardwareAvailable=false;
    bool CheckThread(const char* operation)
    {
        if (!Initialized) return Fail("OpenGL ES RHI is not initialized");
        if (OwnerThread != std::this_thread::get_id())
        {
            return Fail(std::string(operation) + " must execute on the owning OpenGL ES RHI thread");
        }
        if(!ContextCurrent)
        {
            if(EglMakeCurrent(Display,Surface,Surface,Context)==EGL_FALSE_VALUE)return Fail("Cannot acquire the GL render context");
            ContextCurrent=true;
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
        EGLint major = 0;
        EGLint minor = 0;
        // ANGLE's default D3D11 path can stop at ES 3.0. Prefer its native GL
        // path for ES 3.1 compute/storage features, retaining the default fallback.
        // EGL_ANGLE_platform_angle_opengl defines this optional display selection.
        using QueryString = const char*(EVANS_EGL_CALL*)(EGLDisplay, EGLint);
        using GetPlatformDisplay = EGLDisplay(EVANS_EGL_CALL*)(EGLenum, void*, const EGLint*);
        const auto queryString = reinterpret_cast<QueryString>(EglGetProcAddress("eglQueryString"));
        const auto getPlatformDisplay = reinterpret_cast<GetPlatformDisplay>(EglGetProcAddress("eglGetPlatformDisplayEXT"));
        constexpr EGLint Extensions = 0x3055, PlatformType = 0x3203, OpenGLPlatform = 0x320D;
        constexpr EGLenum AnglePlatform = 0x3202;
        const char* extensions = queryString ? queryString(nullptr, Extensions) : nullptr;
        if (GetEnvironmentVariableW(L"ANGLE_DEFAULT_PLATFORM", nullptr, 0) == 0 &&
            extensions && getPlatformDisplay && std::strstr(extensions, "EGL_ANGLE_platform_angle_opengl"))
        {
            const EGLint attributes[] = {PlatformType, OpenGLPlatform, EGL_NONE_VALUE};
            Display = getPlatformDisplay(AnglePlatform, nullptr, attributes);
            if (Display && EglInitialize(Display, &major, &minor) == EGL_FALSE_VALUE) Display = nullptr;
        }
        if (!Display)
        {
            (void)EglGetError();
            Display = EglGetDisplay(nullptr);
            if (Display && EglInitialize(Display, &major, &minor) == EGL_FALSE_VALUE) Display = nullptr;
        }
        if (Display == nullptr)
        {
            return Fail(HexError("eglInitialize", static_cast<unsigned long>(EglGetError())));
        }
        GpuDisplayOwner=RetainDisplay(Display,EglTerminate);
        const EGLint configAttributes[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT | 0x0004,
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
        // Negotiate ES 3.1 for sampled multisample attachments; retain the
        // established ES 3.0 path when the runtime cannot provide that version.
        const EGLint sampledAttributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE_VALUE};
        Context = EglCreateContext(Display, Config, nullptr, sampledAttributes);
        MultisampleAPI = Context != nullptr;
        if (!Context)
        {
            (void)EglGetError();
            const EGLint contextAttributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE_VALUE};
            Context = EglCreateContext(Display, Config, nullptr, contextAttributes);
        }
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
        if (MultisampleAPI && !(LoadGL(GlTexStorage2DMultisample, "glTexStorage2DMultisample") &&
            LoadGL(GlGetInternalformativ, "glGetInternalformativ") && LoadGL(GlGetMultisamplefv, "glGetMultisamplefv") &&
            LoadGL(GlSampleMaski, "glSampleMaski"))) return false;
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
               LoadGL(GlUniform4ui, "glUniform4ui") &&
               LoadGL(GlUniform1i, "glUniform1i") &&
               LoadGL(GlGetUniformBlockIndex, "glGetUniformBlockIndex") &&
               LoadGL(GlUniformBlockBinding, "glUniformBlockBinding") &&
               LoadGL(GlBindBufferBase, "glBindBufferBase") &&
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
               LoadGL(GlTexSubImage2D, "glTexSubImage2D") &&
               LoadGL(GlGenerateMipmap, "glGenerateMipmap") &&
               LoadGL(GlCopyTexSubImage2D, "glCopyTexSubImage2D") &&
               LoadGL(GlTexParameteri, "glTexParameteri") &&
               LoadGL(GlDeleteTextures, "glDeleteTextures") &&
               LoadGL(GlActiveTexture, "glActiveTexture") &&
               LoadGL(GlGenFramebuffers, "glGenFramebuffers") &&
               LoadGL(GlBindFramebuffer, "glBindFramebuffer") &&
               LoadGL(GlBlitFramebuffer, "glBlitFramebuffer") &&
               LoadGL(GlFramebufferTexture2D, "glFramebufferTexture2D") &&
               LoadGL(GlCheckFramebufferStatus, "glCheckFramebufferStatus") &&
               LoadGL(GlDeleteFramebuffers, "glDeleteFramebuffers") &&
               LoadGL(GlDrawBuffers, "glDrawBuffers") &&
               LoadGL(GlReadBuffer, "glReadBuffer") &&
               LoadGL(GlViewport, "glViewport") &&
               LoadGL(GlClearColor, "glClearColor") &&
               LoadGL(GlClearDepthf, "glClearDepthf") &&
               LoadGL(GlClear, "glClear") &&
               LoadGL(GlClearBufferfv, "glClearBufferfv") &&
               LoadGL(GlEnable, "glEnable") &&
               LoadGL(GlDisable, "glDisable") &&
               LoadGL(GlCullFace, "glCullFace") &&
               LoadGL(GlFrontFace, "glFrontFace") &&
               LoadGL(GlDepthMask, "glDepthMask") &&
               LoadGL(GlDepthFunc, "glDepthFunc") &&
               LoadGL(GlColorMask, "glColorMask") &&
               LoadGL(GlBlendFuncSeparate, "glBlendFuncSeparate") &&
               LoadGL(GlBlendEquation, "glBlendEquation") &&
               LoadGL(GlLineWidth, "glLineWidth") &&
               LoadGL(GlPolygonOffset, "glPolygonOffset") &&
               LoadGL(GlDrawElements, "glDrawElements") &&
               LoadGL(GlDrawElementsInstanced, "glDrawElementsInstanced") &&
               LoadGL(GlDrawArrays, "glDrawArrays") &&
               LoadGL(GlPixelStorei, "glPixelStorei") &&
               LoadGL(GlReadPixels, "glReadPixels");
    }

    #include "OpenGLESWireframe.inl"

    GLuint CompileProgramPair(std::string_view vertex,std::string_view fragment)
    {
        return CompileCachedGLESProgram([this](const char* name) {
            auto* address=reinterpret_cast<void*>(GetProcAddress(GLESModule,name));
            return address?address:EglGetProcAddress(name);
        },{{GL_VERTEX_SHADER,vertex},{GL_FRAGMENT_SHADER,fragment}},LastError);
    }
    GLuint CompileComputeProgram(std::string_view source)
    {
        return CompileCachedGLESProgram([this](const char* name) {
            auto* address=reinterpret_cast<void*>(GetProcAddress(GLESModule,name));
            return address?address:EglGetProcAddress(name);
        },{{0x91b9,source}},LastError);
    }

    bool UsesPCSS() const { return FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS; }
    bool UsesVSM() const { return FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM; }
    bool UsesSurfaceVSM() const
    { return UsesVSM() && CurrentPass.Type == PassType::Color && BoundPipeline.SamplesShadowMap; }
    bool UsesSurfacePCSS() const
    { return UsesPCSS() && CurrentPass.Type == PassType::Color && BoundPipeline.SamplesShadowMap; }

    bool UploadPCSS(GLuint program, const Matrix4& lightProjection, const LightingState& lighting, bool enabled)
    {
        PCSSConstants constants;
        if (!BuildPCSSConstants(lightProjection, lighting, FrameCascades, FrameShadowFilter.PCSS,
            enabled, constants, LastError)) return false;
        auto [entry, inserted] = PCSSLocations.try_emplace(program);
        auto& locations = entry->second;
        if (inserted)
        {
            for (std::size_t index = 0; index < PCSSMapCount; ++index)
                locations[index] = GlGetUniformLocation(program, ("PCSSProjection[" + std::to_string(index) + "]").c_str());
            locations[PCSSMapCount] = GlGetUniformLocation(program, "PCSSEmitterLimits");
            locations[PCSSMapCount + 1] = GlGetUniformLocation(program, "PCSSSamplesMaps");
        }
        // Uniform storage belongs to the linked program. Upload after selecting
        // ordinary, instanced, deferred or volume receivers; no other program's
        // cached location or previous-frame projection may be reused here.
        for (std::size_t index = 0; index < PCSSMapCount; ++index)
            UploadFloat4(locations[index], constants.ProjectionMetrics[index]);
        UploadFloat4(locations[PCSSMapCount], constants.EmitterAndLimits);
        const auto& samples = constants.SamplesAndMaps;
        GlUniform4ui(locations[PCSSMapCount + 1], samples.X, samples.Y, samples.Z, samples.W);
        return CheckGL("Upload PCSS receiver metrics");
    }

    bool UsesContactReceiver() const
    { return FrameContact.Enabled && CurrentPass.Type == PassType::Color && !CurrentPass.Target && !BoundPipeline.BlendEnabled; }

    bool UsesForwardPlusMaterial() const
    {
        return LightListsActive && (CurrentPass.Type == PassType::Color ||
            CurrentPass.Type == PassType::EnvironmentCapture || CurrentPass.Type == PassType::PlanarCapture);
    }

    std::size_t ForwardPlusFilter() const
    {
        return CurrentPass.Type == PassType::Color && BoundPipeline.SamplesShadowMap ?
            static_cast<std::size_t>(FrameShadowFilter.Mode) : static_cast<std::size_t>(Techniques::ShadowFilterMode::Hard);
    }

    ForwardPlusProgram ForwardPlusVariant() const
    { return CurrentPass.Type == PassType::ForwardPlusDepth ? ForwardPlusDepth : WeightedPass ? ForwardPlusWeighted : ForwardPlusSurface; }

    std::string ForwardPlusDefinitions() const
    {
        const std::string prefix = ClusteredActive ? "EVANS_CLUSTERED_FORWARD" : TiledActive ? "EVANS_TILED_DEFERRED" : "EVANS_FORWARD_PLUS";
        return "#define " + prefix + "_SURFACE 1\n#define " + prefix + "_FILTER " + std::to_string(ForwardPlusFilter()) + "\n" +
            (ForwardPlusVariant() == ForwardPlusDepth ? "#define " + prefix + "_DEPTH 1\n" :
                ForwardPlusVariant() == ForwardPlusWeighted ? "#define EVANS_WEIGHTED_TRANSPARENCY 1\n" : "");
    }

    bool SelectForwardPlusProgram()
    {
        if (BoundPipeline.Shading == Techniques::ShadingModel::Gouraud)
            return Fail("OpenGL ES Forward+ pixel lists cannot supply Gouraud vertex lighting");
        auto& program = (ClusteredActive ? ClusteredPrograms : TiledActive ? TiledPrograms : ForwardPlusPrograms)[ForwardPlusFilter()][ForwardPlusVariant()];
        if (!program) program = CompileStageProgram(ClusteredActive ? "ClusteredForward.glsl" : TiledActive ? "TiledDeferred.glsl" : "ForwardPlus.glsl", ForwardPlusDefinitions().c_str());
        if (!program) return false;
        Program = program; QueryProgramUniforms(); GlUseProgram(Program);
        GlUniform1i(ShadowMapLocation, 0); GlUniform1i(DiffuseMapLocation, 1);
        GlUniform1i(NormalMapLocation, 2); GlUniform1i(OpacityMapLocation, 3);
        return CheckGL("Select Forward+ material or depth program");
    }

    bool UploadForwardPlusScene()
    {
        const auto& lights = LightListFrame.Description().Lights->GetShaderLights();
        if (!ForwardPlusSceneBuffer) GlGenBuffers(1, &ForwardPlusSceneBuffer);
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER, ForwardPlusSceneBuffer);
        const ShaderLight emptyLight {};
        // Re-specification preserves earlier commands through driver-managed
        // storage renaming. The buffer has one harmless record for an empty scene.
        GlBufferData(GL_SHADER_STORAGE_BUFFER, std::max(lights.size(), std::size_t{1}) * sizeof(ShaderLight),
            lights.empty() ? &emptyLight : lights.data(), GL_DYNAMIC_DRAW);
        if (!ForwardPlusEmptyBuffer)
        {
            GlGenBuffers(1, &ForwardPlusEmptyBuffer);
            GlBindBuffer(GL_SHADER_STORAGE_BUFFER, ForwardPlusEmptyBuffer);
            const ForwardPlusTile emptyTile {};
            GlBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(emptyTile), &emptyTile, GL_STATIC_DRAW);
        }
        if (!ForwardPlusEmptyIndexBuffer)
        {
            // ANGLE maps even readonly SSBOs to UAVs and requires distinct
            // resources for the two active blocks during capture draws.
            GlGenBuffers(1, &ForwardPlusEmptyIndexBuffer);
            GlBindBuffer(GL_SHADER_STORAGE_BUFFER, ForwardPlusEmptyIndexBuffer);
            const std::uint32_t emptyIndex = 0;
            GlBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(emptyIndex), &emptyIndex, GL_STATIC_DRAW);
        }
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return CheckGL("Upload complete Forward+ scene lights");
    }

    GLuint TiledLightingProgram(bool multisample)
    {
        const auto filter = static_cast<std::size_t>(FrameShadowFilter.Mode);
        auto& program = TiledLightingPrograms[filter][multisample ? 1 : 0];
        if (!program)
        {
            std::string definitions = "#define EVANS_TILED_DEFERRED_SURFACE 1\n#define EVANS_TILED_DEFERRED_LIGHTING 1\n";
            definitions += "#define EVANS_TILED_DEFERRED_FILTER " + std::to_string(filter) + "\n";
            if (multisample) definitions += "#define EVANS_MULTISAMPLE 1\n";
            program = CompileStageProgram("TiledDeferred.glsl", definitions.c_str(), multisample);
        }
        return program;
    }

    bool BindForwardPlusSurface()
    {
        const bool mainCamera = CurrentPass.Type == PassType::Color && !CurrentPass.Target;
        const auto selection = !mainCamera ? LightListSelection::FullScene :
            BoundPipeline.BlendEnabled ? LightListSelection::Transparent : LightListSelection::Opaque;
        LightListSurfaceConstants constants;
        if (!LightListFrame.Surface(BoundPipeline.Lighting, BoundViewProjection, selection, constants, LastError)) return false;
        const auto& layout = constants.LightLayout; const auto& tiles = constants.SurfaceTiles;
        GlUniform4ui(GlGetUniformLocation(Program, "ForwardLightLayout"), layout.X, layout.Y, layout.Z, layout.W);
        GlUniform4ui(GlGetUniformLocation(Program, "ForwardSurfaceTiles"), tiles.X, tiles.Y, tiles.Z, tiles.W);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, ForwardPlusLightBinding, ForwardPlusSceneBuffer);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, ForwardPlusHeaderBinding, mainCamera ? (ClusteredActive ? ClusteredForward.Headers() : TileProducer().Headers()) : ForwardPlusEmptyBuffer);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, ForwardPlusIndexBinding, mainCamera ? (ClusteredActive ? ClusteredForward.Indices() : TileProducer().Indices()) : ForwardPlusEmptyIndexBuffer);
        // Even a disabled sampler has a type and texture-unit assignment. Keep
        // the array sampler away from the shadow sampler's 2D texture unit.
        if (UsesContactReceiver()) return BindContactReceiver(Program);
        GlUniform1i(GlGetUniformLocation(Program, "ContactEnabled"), 0);
        GlUniform1i(GlGetUniformLocation(Program, "ContactSample"), 0);
        GlUniform1i(GlGetUniformLocation(Program, "ContactVisibility"), static_cast<GLint>(ContactTextureUnit));
        GlActiveTexture(GL_TEXTURE0 + ContactTextureUnit); GlBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        return CheckGL("Bind Forward+ light selection");
    }

    bool BindContactReceiver(GLuint program)
    {
        if (!ContactGenerated || (ContactReceiverActive && !ContactShadows.Result()))
            return Fail("OpenGL ES contact receiver requires current visibility");
        GlUniform1i(GlGetUniformLocation(program, "ContactEnabled"), ContactReceiverActive ? 1 : 0);
        GlUniform1i(GlGetUniformLocation(program, "ContactSample"), 0);
        GlUniform1i(GlGetUniformLocation(program, "ContactVisibility"), static_cast<GLint>(ContactTextureUnit));
        GlActiveTexture(GL_TEXTURE0 + ContactTextureUnit);
        GlBindTexture(GL_TEXTURE_2D_ARRAY, ContactReceiverActive ? ContactShadows.Result() : 0);
        return CheckGL("Bind contact receiver");
    }

    GLuint ContactLightingProgram(bool multisample)
    {
        const unsigned family = UsesVSM() ? 2 : UsesPCSS() ? 1 : 0;
        auto& program = ContactScreenPrograms[family][multisample ? 1 : 0];
        if (!program)
        {
            std::string definitions = "#define EVANS_CONTACT_RECEIVER 1\n";
            if (family)
                definitions += std::string(family == 2 ? "#define EVANS_VSM_PROGRAM " : "#define EVANS_PCSS_PROGRAM ") +
                    (multisample ? "6\n" : "5\n");
            else definitions += std::string("#define EVANS_DEFERRED_LIGHTING 1\n") + (multisample ? "#define EVANS_MULTISAMPLE 1\n" : "");
            program = CompileStageProgram(family == 2 ? "VSM.glsl" : family == 1 ? "PCSS.glsl" : "Deferred.glsl",
                definitions.c_str(), multisample);
        }
        return program;
    }

    bool CreateProgram(Techniques::ShadingModel model = Techniques::ShadingModel::Phong, bool weighted = false, bool pcss = false, bool vsm = false, bool contact = false)
    {
        auto& programs = contact ? ContactSurfacePrograms[vsm ? 2 : pcss ? 1 : 0] :
            vsm ? VSMSurfacePrograms[weighted ? 1 : 0] : pcss ? PCSSSurfacePrograms[weighted ? 1 : 0] : weighted ? WeightedTechniquePrograms : TechniquePrograms;
        const auto modelIndex = ShadingTechniqueIndex(model, FrameAreaLights && !pcss && !vsm && !contact);
        if (modelIndex >= programs.size()) return Fail("Invalid shading model");
        if (programs[modelIndex] != 0)
        {
            if (Program != programs[modelIndex])
            {
                Program = programs[modelIndex];
                QueryProgramUniforms();
            }
            return true;
        }
        std::string source;
        const std::string filename = std::string(ShadingTechniqueFile(model, FrameAreaLights || contact)) + ".glsl";
        if (!LoadTechniqueShaderSource("OpenGLES", filename.c_str(), source, LastError)) return false;
        const std::size_t versionEnd = source.find('\n') + 1;
        if (weighted) source.insert(versionEnd, "#define EVANS_WEIGHTED_TRANSPARENCY 1\n");
        VertexShaderCode = source;
        VertexShaderCode.insert(versionEnd, "#define EVANS_VERTEX_STAGE 1\n");
        FragmentShaderCode = source;
        if (pcss || vsm)
        {
            if (!LoadTechniqueShaderSource("OpenGLES", vsm ? "VSM.glsl" : "PCSS.glsl", FragmentShaderCode, LastError)) return false;
            const auto definitions = std::string(vsm ? "#define EVANS_VSM_PROGRAM " : "#define EVANS_PCSS_PROGRAM ") + std::to_string(modelIndex) + "\n" +
                (weighted ? "#define EVANS_WEIGHTED_TRANSPARENCY 1\n" : "");
            FragmentShaderCode.insert(FragmentShaderCode.find('\n') + 1, definitions);
        }
        if (contact) FragmentShaderCode.insert(FragmentShaderCode.find('\n') + 1, "#define EVANS_CONTACT_RECEIVER 1\n");

        Program=CompileProgramPair(VertexShaderCode,FragmentShaderCode);
        if(!Program)return false;
        programs[modelIndex] = Program;
        QueryProgramUniforms();
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

    bool SelectInstanceProgram()
    {
        // Program is the ordinary consumer's linked program, selected for this
        // draw before entering here. Its ID distinguishes all fragment variants.
        const GLuint ordinary = Program;
        auto cached = InstancePrograms.find(ordinary);
        if (cached == InstancePrograms.end())
        {
            const char* vertexPhase = "";
            const char* vertexFile = "InstancedRendering.glsl";
            std::string fragmentPhase = WeightedPass ? "#define EVANS_WEIGHTED_TRANSPARENCY 1\n" : "";
            std::string fragmentFile = std::string(ShadingTechniqueFile(BoundPipeline.Shading, FrameAreaLights || UsesContactReceiver())) + ".glsl";
            if (CurrentPass.Type == PassType::Velocity)
            {
                fragmentFile = "MotionVectors.glsl";
                vertexPhase = "#define EVANS_INSTANCE_VELOCITY 1\n";
            }
            else if (CurrentPass.Type == PassType::Shadow)
            {
                const bool cascaded = FrameCascades.Frame.CascadeCount != 0;
                fragmentFile = cascaded ? "CascadedShadowMap.glsl" : "ShadowMapping.glsl";
                vertexPhase = cascaded ? "#define EVANS_INSTANCE_CASCADE 1\n" : "#define EVANS_INSTANCE_SHADOW 1\n";
            }
            else if (CurrentPass.Type == PassType::GBuffer)
            {
                fragmentFile = "Deferred.glsl";
                vertexPhase = "#define EVANS_INSTANCE_DEFERRED 1\n";
            }
            else if (CurrentPass.Type == PassType::ContactDepth)
            {
                fragmentFile = "ContactShadows.glsl";
                fragmentPhase = "#define EVANS_CONTACT_DEPTH 1\n";
            }
            else if (BoundPipeline.Shading == Techniques::ShadingModel::Gouraud)
                vertexPhase = "#define EVANS_INSTANCE_GOURAUD 1\n";
            if (FrameRayShadowImage && !FrameAreaLights && CurrentPass.Type==PassType::Color && BoundPipeline.SamplesShadowMap)
            {
                fragmentFile="RayTracedShadows.glsl";
                fragmentPhase+="#define EVANS_RAY_SHADOW_PROGRAM "+std::to_string(static_cast<unsigned>(BoundPipeline.Shading))+"\n";
            }
            if (UsesSurfacePCF())
            {
                fragmentFile="PCF.glsl";
                fragmentPhase+="#define EVANS_PCF_PROGRAM "+std::to_string(unsigned(BoundPipeline.Shading))+"\n";
            }
            if (UsesSurfacePCSS())
            {
                fragmentFile = "PCSS.glsl";
                fragmentPhase += "#define EVANS_PCSS_PROGRAM " + std::to_string(static_cast<unsigned>(BoundPipeline.Shading)) + "\n";
            }
            if (UsesSurfaceVSM())
            {
                fragmentFile = "VSM.glsl";
                fragmentPhase += "#define EVANS_VSM_PROGRAM " + std::to_string(static_cast<unsigned>(BoundPipeline.Shading)) + "\n";
            }
            if (UsesVSM() && CurrentPass.Type == PassType::Shadow)
            {
                vertexFile = "VSM.glsl"; fragmentFile = vertexFile;
                vertexPhase = "#define EVANS_VERTEX_STAGE 1\n#define EVANS_VSM_PROGRAM 8\n#define EVANS_VSM_INSTANCED 1\n";
                fragmentPhase = "#define EVANS_VSM_PROGRAM 8\n";
            }
            if (UsesMaterialStage(BoundPipeline,CurrentPass.Type))
            {
                fragmentFile=std::string(MaterialTechniqueFile(SelectMaterialTechnique(BoundPipeline,WeightedPass)))+".glsl";
                fragmentPhase+="#define EVANS_MATERIAL_PROGRAM "+std::to_string(unsigned(BoundPipeline.Shading))+"\n";
                if (CurrentPass.Type==PassType::GBuffer) fragmentPhase+="#define EVANS_MATERIAL_GBUFFER 1\n";
            }
            if (UsesContactReceiver()) fragmentPhase += "#define EVANS_CONTACT_RECEIVER 1\n";

            const bool forwardPlus = UsesForwardPlusMaterial() || CurrentPass.Type == PassType::ForwardPlusDepth;
            if (forwardPlus)
            {
                vertexPhase = "";
                fragmentFile = ClusteredActive ? "ClusteredForward.glsl" : TiledActive ? "TiledDeferred.glsl" : "ForwardPlus.glsl";
                fragmentPhase = ForwardPlusDefinitions();
            }

            std::string vertexSource, fragmentSource;
            if (!LoadTechniqueShaderSource("OpenGLES", vertexFile, vertexSource, LastError) ||
                !LoadTechniqueShaderSource("OpenGLES", fragmentFile.c_str(), fragmentSource, LastError)) return false;
            if (CurrentPass.Type == PassType::ContactDepth || forwardPlus)
                vertexSource.replace(0, vertexSource.find('\n'), "#version 310 es");
            vertexSource.insert(vertexSource.find('\n') + 1, vertexPhase);
            fragmentSource.insert(fragmentSource.find('\n') + 1, fragmentPhase);
            const GLuint linkedProgram=CompileProgramPair(vertexSource,fragmentSource);
            if(!linkedProgram)return false;
            const GLuint block = GlGetUniformBlockIndex(linkedProgram, "InstanceBlock");
            if (block == GL_INVALID_INDEX_VALUE)
            {
                GlDeleteProgram(linkedProgram);
                return Fail("OpenGL ES instance program has no transform block");
            }
            GlUniformBlockBinding(linkedProgram, block, InstanceUniformBinding);
            if (!CheckGL("Bind instance uniform block")) { GlDeleteProgram(linkedProgram); return false; }
            cached = InstancePrograms.emplace(ordinary, linkedProgram).first;
        }
        Program = cached->second;
        QueryProgramUniforms();
        GlUseProgram(Program);
        GlUniform1i(ShadowMapLocation, 0); GlUniform1i(DiffuseMapLocation, 1);
        GlUniform1i(NormalMapLocation, 2); GlUniform1i(OpacityMapLocation, 3);
        if (CurrentPass.Type == PassType::Shadow)
        {
            const auto& size = Textures.at(CurrentPass.Target.Value).Description;
            const char* name = !UsesVSM() && FrameCascades.Frame.CascadeCount ? "CascadeDepthSize" : "ShadowDepthSize";
            GlUniform4f(GlGetUniformLocation(Program, name), float(size.Width), float(size.Height), 0, 0);
        }
        return CheckGL("Select instance program");
    }

    bool UploadInstances(const InstanceConstantBlock& constants)
    {
        if (ActiveGpuBatch)
        {
            GlBindBufferBase(GL_UNIFORM_BUFFER, InstanceUniformBinding, ActiveGpuBatch->Output);
            return CheckGL("Bind GPU-compacted instance constants");
        }
        if (!InstanceBuffer) GlGenBuffers(1, &InstanceBuffer);
        GlBindBuffer(GL_UNIFORM_BUFFER, InstanceBuffer);
        // Re-specify the data store for each command. ES preserves earlier
        // draws' contents; the driver owns renaming/synchronization. No mapped
        // memory or manual fence can accidentally overwrite an in-flight batch.
        // OpenGL ES 3.0.6, sections 2.9.3, 2.10.2 and 2.12.6:
        // https://registry.khronos.org/OpenGL/specs/es/3.0/es_spec_3.0.pdf
        GlBufferData(GL_UNIFORM_BUFFER, sizeof(constants), &constants, GL_DYNAMIC_DRAW);
        GlBindBufferBase(GL_UNIFORM_BUFFER, InstanceUniformBinding, InstanceBuffer);
        return CheckGL("Upload instance constants");
    }

    void QueryProgramUniforms()
    {
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
        ShadingParametersLocation = GlGetUniformLocation(Program, "ShadingParameters");
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
    }

    bool CreateDepthReadProgram()
    {
        DepthReadProgram=CompileProgramPair(DepthReadVertexShaderSource,DepthReadFragmentShaderSource);
        if(!DepthReadProgram)return false;
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
        // Scene-color reconstruction extends edge texels; material UV wrapping
        // must not leak into fullscreen effects that sample beyond the image.
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
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
        // Visibility retains the primary depth attachment while other pipelines
        // render. A reused texture name must not preserve the deleted attachment.
        DestroyVisibilityTargets();
        DestroyMultisampleTargets(Multisample);
        DestroyAmbientOcclusionTargets();
        DestroyDeferredTargets();
        DestroyHdrTargets();
        DestroyPostTargets();
        if (MainFramebuffer != 0 && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &MainFramebuffer);
        if (DepthTexture != 0 && GlDeleteTextures) GlDeleteTextures(1, &DepthTexture);
        if (ColorTexture != 0 && GlDeleteTextures) GlDeleteTextures(1, &ColorTexture);
        MainFramebuffer = 0;
        DepthTexture = 0;
        ColorTexture = 0;
    }

    void DestroyPostTargets()
    {
        for (GLuint& framebuffer : PostFramebuffers)
        {
            if (framebuffer != 0 && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &framebuffer);
            framebuffer = 0;
        }
        for (GLuint& texture : PostTextures)
        {
            if (texture != 0 && GlDeleteTextures) GlDeleteTextures(1, &texture);
            texture = 0;
        }
    }

    bool EnsurePostTargets()
    {
        if (PostTextures[0] != 0 && PostTextures[1] != 0) return true;
        DestroyPostTargets();
        for (std::size_t index = 0; index < PostTextures.size(); ++index)
        {
            GlGenTextures(1, &PostTextures[index]);
            GlBindTexture(GL_TEXTURE_2D, PostTextures[index]);
            ConfigureTexture();
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, CreateInfo.Width, CreateInfo.Height, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            GlGenFramebuffers(1, &PostFramebuffers[index]);
            GlBindFramebuffer(GL_FRAMEBUFFER, PostFramebuffers[index]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, PostTextures[index], 0);
            const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
            GlDrawBuffers(1, &colorAttachment);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            {
                DestroyPostTargets();
                return Fail("OpenGL ES post-process framebuffer is incomplete");
            }
        }
        return CheckGL("Create post-process targets");
    }

    bool SelectShadowProgram()
    {
        GLuint depthProgram = CascadeProgram;
        if (UsesVSM())
        {
            depthProgram = VSMStageProgram(VSMCasterStage);
            if (!depthProgram) return false;
        }
        else if (!FrameCascades.Frame.CascadeCount)
        {
            if (!ShadowProgram) ShadowProgram = CompileStageProgram("ShadowMapping.glsl");
            if (!ShadowProgram) return false;
            depthProgram = ShadowProgram;
        }
        if (UsesVSM() || !FrameCascades.Frame.CascadeCount)
        {
            const auto& target = Textures.at(CurrentPass.Target.Value).Description;
            GlUseProgram(depthProgram);
            GlUniform1i(GlGetUniformLocation(depthProgram, "DiffuseMap"), 1);
            GlUniform1i(GlGetUniformLocation(depthProgram, "OpacityMap"), 3);
            GlUniform4f(GlGetUniformLocation(depthProgram, "ShadowDepthSize"),
                static_cast<float>(target.Width), static_cast<float>(target.Height), 0, 0);
        }
        if (Program != depthProgram) { Program = depthProgram; QueryProgramUniforms(); }
        return CheckGL("Select shadow depth program");
    }

    GLuint CompileStageProgram(const char* filename, const char* phaseDefinition = "", bool multisample = false)
    {
        std::string source;
        if (!LoadTechniqueShaderSource("OpenGLES", filename, source, LastError)) return 0;
        // A language-version variant of this API's own source, without cross-API translation.
        if (multisample) source.replace(0, source.find('\n'), "#version 310 es");
        source.insert(source.find('\n') + 1, phaseDefinition);
        std::string vertexSource = source;
        vertexSource.insert(vertexSource.find('\n') + 1, "#define EVANS_VERTEX_STAGE 1\n");
        return CompileProgramPair(vertexSource,source);
    }

    GLuint PCSSScreenProgram(std::size_t index)
    {
        // Surface receivers use existing native vertex stages. Screen receivers
        // own their fullscreen stage; ES 3.1 supplies multisample texture reads.
        if (index >= PCSSScreenPrograms.size()) { Fail("Invalid OpenGL ES PCSS screen receiver"); return 0; }
        auto& program = PCSSScreenPrograms[index];
        if (!program)
        {
            constexpr std::size_t FirstScreenProgram = 5;
            const auto definitions = "#define EVANS_PCSS_PROGRAM " + std::to_string(index + FirstScreenProgram) + "\n";
            program = CompileStageProgram("PCSS.glsl", definitions.c_str(), index == 1);
        }
        return program;
    }

    GLuint VSMStageProgram(std::size_t index)
    {
        if (index >= VSMStagePrograms.size()) { Fail("Invalid OpenGL ES VSM stage"); return 0; }
        auto& program = VSMStagePrograms[index];
        if (!program)
        {
            constexpr std::size_t FirstScreenProgram = 5, MultisampleProgram = 6;
            const auto stage = FirstScreenProgram + index;
            const auto define = "#define EVANS_VSM_PROGRAM " + std::to_string(stage) + "\n";
            program = CompileStageProgram("VSM.glsl", define.c_str(), stage == MultisampleProgram);
        }
        return program;
    }

    bool UploadVSM(GLuint program, const VSMConstants& constants)
    {
        auto [entry, inserted] = VSMLocations.try_emplace(program);
        auto& locations = entry->second;
        if (inserted)
        {
            for (std::size_t index = 0; index < VSMMapCount; ++index)
                locations[index] = GlGetUniformLocation(program, ("MomentDepthEncoding[" + std::to_string(index) + "]").c_str());
            locations[VSMMapCount] = GlGetUniformLocation(program, "MomentSettings");
            locations[VSMMapCount + 1] = GlGetUniformLocation(program, "MomentOptions");
        }
        for (std::size_t index = 0; index < VSMMapCount; ++index) UploadFloat4(locations[index], constants.DepthEncoding[index]);
        UploadFloat4(locations[VSMMapCount], constants.Parameters);
        const auto& control = constants.Control;
        GlUniform4ui(locations[VSMMapCount + 1], control.X, control.Y, control.Z, control.W);
        return CheckGL("Upload VSM program uniforms");
    }

    bool MomentTexture(TextureHandle handle, GLuint& texture)
    {
        const auto found = VSMMoments.find(handle.Value);
        if (found == VSMMoments.end() || !found->second.Ready)
            return Fail("OpenGL ES VSM receivers require filtered moments from this frame");
        texture = found->second.Textures[VSMFiltered];
        return true;
    }

    bool UploadVSMReceiver(GLuint program, const Matrix4& projection, const LightingState& lighting, bool shadows)
    {
        if (shadows)
        {
            GLuint texture;
            const auto count = FrameCascades.Frame.CascadeCount;
            if (!count) { if (!MomentTexture(LastShadowTexture, texture)) return false; }
            else for (std::uint32_t index = 0; index < count; ++index)
                if (!MomentTexture(FrameCascades.Maps[index], texture)) return false;
        }
        VSMConstants constants;
        return BuildVSMConstants(projection, lighting, FrameCascades, FrameShadowFilter.VSM,
            shadows, constants, LastError) && UploadVSM(program, constants);
    }

    bool CreateMomentTarget(GLuint& texture, GLuint& framebuffer, int width, int height)
    {
        GlActiveTexture(GL_TEXTURE0);
        GlGenTextures(1, &texture); GlBindTexture(GL_TEXTURE_2D, texture);
        ConfigureDepthTexture(); // Nearest sampling; the receiver owns bilinear moment reconstruction.
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, nullptr);
        GlGenFramebuffers(1, &framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        const GLenum color = GL_COLOR_ATTACHMENT0; GlDrawBuffers(1, &color);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        const bool clean = CheckGL("Allocate RG32F moment target");
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlBindTexture(GL_TEXTURE_2D, 0);
        if (complete && clean) return true;
        GlDeleteFramebuffers(1, &framebuffer); GlDeleteTextures(1, &texture);
        framebuffer = texture = 0;
        return clean ? Fail("OpenGL ES RG32F moment framebuffer is incomplete") : false;
    }

    bool ProbeVSMSupport()
    {
        // EXT_color_buffer_float adds RG32F rendering and RGBA/FLOAT readback.
        // No float blending or linear-filter extension is required by VSM.
        // https://registry.khronos.org/OpenGL/extensions/EXT/EXT_color_buffer_float.txt
        if (!FloatDepthTargetSupported) return false;
        const auto previousError = LastError;
        GLuint texture = 0, framebuffer = 0;
        bool supported = CreateMomentTarget(texture, framebuffer, 1, 1);
        if (supported)
        {
            GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            GlClearColor(1, 1, 0, 0); GlClear(GL_COLOR_BUFFER_BIT);
            Float4 result;
            GlReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, &result);
            supported = CheckGL("Probe RG32F moment rendering and readback") && result.X == 1 && result.Y == 1;
        }
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlBindTexture(GL_TEXTURE_2D, 0);
        if (framebuffer) GlDeleteFramebuffers(1, &framebuffer);
        if (texture) GlDeleteTextures(1, &texture);
        LastError = previousError;
        return supported;
    }

    void ReleaseMomentMap(VSMMomentMap& map)
    {
        if (GlDeleteFramebuffers) GlDeleteFramebuffers(static_cast<GLsizei>(map.Framebuffers.size()), map.Framebuffers.data());
        if (GlDeleteTextures) GlDeleteTextures(static_cast<GLsizei>(map.Textures.size()), map.Textures.data());
        map = {};
    }

    bool BeginMoments(TextureHandle handle, bool clear, GLuint& framebuffer)
    {
        if (!VSMTargetSupported) return Fail("OpenGL ES VSM requires RG32F render targets");
        if (!VSMStageProgram(VSMFilterStage)) return false;
        const auto& depth = Textures.at(handle.Value);
        auto& map = VSMMoments[handle.Value];
        if (!clear && !map.RawWritten) return Fail("OpenGL ES VSM requires an initial caster clear each frame");
        if (!map.Textures[VSMRaw])
        {
            for (std::size_t index = 0; index < map.Textures.size(); ++index)
                if (!CreateMomentTarget(map.Textures[index], map.Framebuffers[index], depth.Description.Width, depth.Description.Height))
                { ReleaseMomentMap(map); return false; }
            map.Width = depth.Description.Width; map.Height = depth.Description.Height;
            GlBindFramebuffer(GL_FRAMEBUFFER, map.Framebuffers[VSMRaw]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth.Texture, 0);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            { GlBindFramebuffer(GL_FRAMEBUFFER, 0); ReleaseMomentMap(map); return Fail("OpenGL ES moment/depth attachments are incompatible"); }
        }
        map.Ready = false;
        framebuffer = map.Framebuffers[VSMRaw];
        return CheckGL("Prepare VSM caster target");
    }

    bool FilterMoments(TextureHandle handle)
    {
        auto found = VSMMoments.find(handle.Value);
        if (found == VSMMoments.end()) return Fail("OpenGL ES VSM caster target is missing");
        auto& map = found->second;
        const GLuint program = VSMStageProgram(VSMFilterStage);
        if (!program) return false;
        GlUseProgram(program); GlUniform1i(GlGetUniformLocation(program, "MomentInput"), 0);
        GlDisable(GL_DEPTH_TEST); GlDisable(GL_BLEND); GlDisable(GL_CULL_FACE);
        GlDisable(GL_POLYGON_OFFSET_FILL); GlDisable(GL_DITHER);
        GlDepthMask(GL_FALSE_VALUE); GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        GlViewport(0, 0, map.Width, map.Height); GlBindVertexArray(VertexArray);
        auto constants = BuildVSMParameters(FrameShadowFilter.VSM);
        for (std::uint32_t axis = 0; axis < 2; ++axis)
        {
            // Preserve raw moments for load-preserving caster passes. Each
            // sampled texture belongs to a different FBO from the destination.
            UnbindSampledTextures();
            GlBindFramebuffer(GL_FRAMEBUFFER, map.Framebuffers[axis + 1]);
            GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D, map.Textures[axis]);
            constants.Control.W = axis;
            if (!UploadVSM(program, constants)) return false;
            constexpr GLsizei TriangleVertices = 3; GlDrawArrays(GL_TRIANGLES, 0, TriangleVertices);
        }
        UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlDepthMask(GL_TRUE_VALUE);
        if (!CheckGL("Filter variance shadow moments")) return false;
        map.Ready = map.RawWritten = true;
        PendingFrame.Statistics.ShadowMomentFilterPassCount += 2;
        return true;
    }

    bool ReadMoments(TextureHandle handle)
    {
        GLuint texture;
        if (!MomentTexture(handle, texture)) return false;
        const auto& map = VSMMoments.at(handle.Value);
        std::vector<Float4> bottomUp(static_cast<std::size_t>(map.Width) * map.Height);
        UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER, map.Framebuffers[VSMFiltered]);
        GlPixelStorei(GL_PACK_ALIGNMENT, 1);
        GlReadPixels(0, 0, map.Width, map.Height, GL_RGBA, GL_FLOAT, bottomUp.data());
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (!CheckGL("Read filtered variance moments")) return false;
        auto& output = PendingFrame.ShadowMoments; output.resize(bottomUp.size());
        for (int row = 0; row < map.Height; ++row)
            for (int column = 0; column < map.Width; ++column)
            {
                const auto value = bottomUp[static_cast<std::size_t>(map.Height - row - 1) * map.Width + column];
                output[static_cast<std::size_t>(row) * map.Width + column] = {value.X, value.Y};
            }
        return true;
    }

    bool PrepareCascadedShadows(const CascadedShadowState& state)
    {
        FrameCascades = {};
        CascadeTextures.fill(0);
        const auto count = state.Frame.CascadeCount;
        if (count > MaximumShadowCascades || (count && state.MapResolution <= 0))
            return Fail("OpenGL ES cascade count or map resolution is invalid");
        for (std::uint32_t index = 0; index < count; ++index)
        {
            const auto map = Textures.find(state.Maps[index].Value);
            if (map == Textures.end() || map->second.Description.Format != TextureFormat::Depth32 ||
                map->second.Description.Width != state.MapResolution ||
                map->second.Description.Height != state.MapResolution || !map->second.Framebuffer)
                return Fail("OpenGL ES cascade depth target is invalid");
            CascadeTextures[index] = map->second.Texture;
        }
        if (count)
        {
            if (!CascadeProgram) CascadeProgram = CompileStageProgram("CascadedShadowMap.glsl");
            if (!CascadeProgram) return false;
            GlUseProgram(CascadeProgram);
            GlUniform1i(GlGetUniformLocation(CascadeProgram, "DiffuseMap"), 1);
            GlUniform1i(GlGetUniformLocation(CascadeProgram, "OpacityMap"), 3);
            GlUniform4f(GlGetUniformLocation(CascadeProgram, "CascadeDepthSize"),
                float(state.MapResolution), float(state.MapResolution), 0, 0);
        }
        FrameCascades = state;
        return CheckGL("Prepare cascaded shadows");
    }

    bool BindCascades(GLuint program, bool enabled)
    {
        const bool active = enabled && (FrameCascades.Frame.CascadeCount > 0 || FrameVirtualShadowAtlas);
        auto [entry, inserted] = CascadeLocations.try_emplace(program);
        auto& locations = entry->second;
        if (inserted)
        {
            for (std::uint32_t index = 0; index < MaximumShadowCascades; ++index)
            {
                const std::string number = std::to_string(index);
                locations.Matrices[index] = GlGetUniformLocation(program, ("CascadeLightMatrices[" + number + "]").c_str());
                locations.Maps[index] = GlGetUniformLocation(program, ("CascadeDepth" + number).c_str());
            }
            locations.FarDistances = GlGetUniformLocation(program, "CascadeFarDistances");
            locations.BlendStarts = GlGetUniformLocation(program, "CascadeBlendStarts");
            locations.EyeDepthPlane = GlGetUniformLocation(program, "CascadeEyeDepthPlane");
            locations.Parameters = GlGetUniformLocation(program, "CascadeParameters");
            locations.Filtering = GlGetUniformLocation(program, "ShadowFiltering");
        }
        for (std::uint32_t index = 0; index < MaximumShadowCascades; ++index)
        {
            const GLuint unit = FirstCascadeTextureUnit + index;
            GlActiveTexture(GL_TEXTURE0 + unit);
            GLuint texture = active && FrameVirtualShadowAtlas ? (index==0?FrameVirtualShadowAtlas->Depth:index==1?FrameVirtualShadowAtlas->Directory:0) : active?CascadeTextures[index]:0;
            if (active && UsesVSM() && index < FrameCascades.Frame.CascadeCount &&
                !MomentTexture(FrameCascades.Maps[index], texture)) return false;
            GlBindTexture(GL_TEXTURE_2D, texture);
            GlUniform1i(locations.Maps[index], static_cast<GLint>(unit));
        }
        // Programs retain uniforms across frames. Explicitly disable selection
        // on single-map, depth, G-buffer and velocity draws after any CSM frame.
        if (locations.Parameters >= 0)
        {
            const auto constants = active && FrameVirtualShadowAtlas ? BuildVirtualShadowSamplingConstants(FrameVirtualShadowAtlas->Info,FrameVirtualShadowSampling,FrameShadowFilter) :
                BuildCascadedShadowConstants(active ? FrameCascades : CascadedShadowState{}, FrameShadowFilter);
            UploadFloat4(locations.Parameters, constants.Parameters);
            UploadFloat4(locations.Filtering, constants.Filtering);
            if (active)
            {
                UploadFloat4(locations.EyeDepthPlane, constants.EyeDepthPlane);
                GlUniform4f(locations.FarDistances, constants.FarDistances[0], constants.FarDistances[1],
                    constants.FarDistances[2], constants.FarDistances[3]);
                GlUniform4f(locations.BlendStarts, constants.BlendStarts[0], constants.BlendStarts[1],
                    constants.BlendStarts[2], constants.BlendStarts[3]);
                for (std::uint32_t index = 0; index < (FrameVirtualShadowAtlas?1U:FrameCascades.Frame.CascadeCount); ++index)
                {
                    const auto matrix = ToColumnMajor(constants.LightViewProjection[index]);
                    GlUniformMatrix4fv(locations.Matrices[index], 1, GL_FALSE_VALUE, matrix.data());
                }
            }
        }
        GlActiveTexture(GL_TEXTURE0);
        return true;
    }

    GLuint GetPostProgram(Techniques::PostProcessPassKind pass)
    {
        const std::size_t index = static_cast<std::size_t>(pass);
        if (index >= PostPrograms.size()) { Fail("Invalid post-process pass"); return 0; }
        constexpr const char* filenames[] = {"FXAA.glsl", "Vignette.glsl", "FilmGrain.glsl", "ScreenSpaceDistortion.glsl", "NPR.glsl"};
        static_assert(std::size(filenames) == PostProcessTechniqueCount);
        if (PostPrograms[index] == 0) PostPrograms[index] = CompileStageProgram(filenames[index]);
        return PostPrograms[index];
    }

    void DestroyHdrTargets()
    {
        DestroySubsurfaceTargets();
        DestroyExposureTargets();
        DestroyPathTargets();
        DestroyWeightedTargets(Weighted);
        DestroyFogTargets();
        DestroyLightShaftTargets();
        DestroyTemporalTargets();
        DestroyGITargets(); DestroyProbeTargets(); DestroyLightmapTargets(); DestroyRefractionTargets();
        DestroyMotionTargets();
        DestroyReflectionTargets();
        if (DepthOfFieldFramebuffer && GlDeleteFramebuffers)
            GlDeleteFramebuffers(1, &DepthOfFieldFramebuffer);
        DepthOfFieldFramebuffer = 0;
        for (auto& framebuffer : HdrFramebuffers)
        {
            if (framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &framebuffer);
            framebuffer = 0;
        }
        for (auto& texture : HdrTextures)
        {
            if (texture && GlDeleteTextures) GlDeleteTextures(1, &texture);
            texture = 0;
        }
    }

    void DestroyWeightedTargets(WeightedTargets& targets)
    {
        if (targets.Framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &targets.Framebuffer);
        if (GlDeleteTextures) for (auto texture : targets.Textures) if (texture) GlDeleteTextures(1, &texture);
        targets = {};
    }

    bool CreateWeightedTargets(WeightedTargets& targets, int width, int height, std::uint32_t samples)
    {
        WeightedTargets candidate;
        candidate.Width = width; candidate.Height = height; candidate.Samples = samples;
        const GLenum textureTarget = samples > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
        const GLenum formats[] {GL_RGBA32F, GL_R32F};
        const GLenum attachments[] {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0 + 1};
        GlActiveTexture(GL_TEXTURE0);
        GlGenTextures(2, candidate.Textures.data());
        GlGenFramebuffers(1, &candidate.Framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER, candidate.Framebuffer);
        for (std::size_t index = 0; index < candidate.Textures.size(); ++index)
        {
            GlBindTexture(textureTarget, candidate.Textures[index]);
            if (samples > 1) GlTexStorage2DMultisample(textureTarget, samples, formats[index], width, height, GL_TRUE_VALUE);
            else
            {
                ConfigureDepthTexture(); // Nearest, clamped loads need no float filtering extension.
                GlTexImage2D(textureTarget, 0, formats[index], width, height, 0, index == 0 ? GL_RGBA : GL_RED, GL_FLOAT, nullptr);
            }
            GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[index], textureTarget, candidate.Textures[index], 0);
        }
        GlDrawBuffers(2, attachments);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlBindTexture(textureTarget, 0);
        const bool clean = CheckGL("Create weighted transparency targets");
        if (!complete || !clean)
        {
            DestroyWeightedTargets(candidate);
            return clean ? Fail("OpenGL ES weighted transparency framebuffer is incomplete") : false;
        }
        DestroyWeightedTargets(targets); targets = candidate;
        return true;
    }

    std::uint32_t ProbeWeightedSupport()
    {
        if (!HalfFloatTargetsSupported || !FloatDepthTargetSupported || MaximumDrawBuffers < 2 || MaximumColorAttachments < 2) return 0;
        const auto* bytes = GlGetString(GL_EXTENSIONS);
        const auto* versionBytes = GlGetString(GL_VERSION);
        const std::string extensions = " " + std::string(bytes ? reinterpret_cast<const char*>(bytes) : "") + " ";
        const std::string version = versionBytes ? reinterpret_cast<const char*>(versionBytes) : "";
        // Color renderability alone does not permit blending into float32.
        // https://registry.khronos.org/OpenGL/extensions/EXT/EXT_float_blend.txt
        // Independent indexed blend functions are core in ES 3.2 or supplied
        // by EXT/OES_draw_buffers_indexed on earlier contexts.
        if (extensions.find(" GL_EXT_float_blend ") == std::string::npos) return 0;
        const char* function = extensions.find(" GL_EXT_draw_buffers_indexed ") != std::string::npos ? "glBlendFuncSeparateiEXT" :
            extensions.find(" GL_OES_draw_buffers_indexed ") != std::string::npos ? "glBlendFuncSeparateiOES" :
            version.find("OpenGL ES 3.2") != std::string::npos ? "glBlendFuncSeparatei" : nullptr;
        const auto previousError = LastError;
        if (!function || !LoadGL(GlBlendFuncSeparatei, function)) { LastError = previousError; return 0; }
        std::uint32_t counts = MultisampleAPI ? SupportedSampleCounts : 1;
        if (MultisampleAPI) for (GLenum format : {GL_RGBA32F, GL_R32F})
        {
            GLint count = 0;
            GlGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, format, GL_NUM_SAMPLE_COUNTS, 1, &count);
            if (GlGetError() != GL_NO_ERROR || count <= 0) { counts = 1; break; }
            std::vector<GLint> supported(static_cast<std::size_t>(count));
            GlGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, format, GL_SAMPLES, count, supported.data());
            if (GlGetError() != GL_NO_ERROR) { counts = 1; break; }
            std::uint32_t formatCounts = 1;
            for (auto value : supported) if (value == 2 || value == 4 || value == 8) formatCounts |= value;
            counts &= formatCounts;
        }
        for (std::uint32_t samples : {1u, 2u, 4u, 8u})
        {
            if (!(counts & samples)) continue;
            WeightedTargets probe;
            if (!CreateWeightedTargets(probe, 1, 1, samples)) counts &= ~samples;
            DestroyWeightedTargets(probe);
        }
        LastError = previousError;
        return (counts & 1u) ? counts : 0;
    }

    bool EnsureWeightedResources()
    {
        if (!HdrActive || !(WeightedSampleCounts & FrameSampleCount))
            return Fail("OpenGL ES weighted transparency requires supported blendable float targets and HDR");
        if (!Weighted.Framebuffer || Weighted.Width != CreateInfo.Width || Weighted.Height != CreateInfo.Height || Weighted.Samples != FrameSampleCount)
            if (!CreateWeightedTargets(Weighted, CreateInfo.Width, CreateInfo.Height, FrameSampleCount)) return false;
        // The original opaque depth remains attached and read-only during draws.
        // Reattach it after sample-count changes or scene-target replacement.
        GlBindFramebuffer(GL_FRAMEBUFFER, Weighted.Framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            FrameSampleCount > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D,
            FrameSampleCount > 1 ? Multisample.Depth : DepthTexture, 0);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (!complete) return Fail("OpenGL ES transparency targets do not match opaque depth");
        auto& program = WeightedResolvePrograms[FrameSampleCount > 1 ? 1 : 0];
        if (!program) program = CompileStageProgram("WeightedBlendedOIT.glsl",
            FrameSampleCount > 1 ? "#define EVANS_WEIGHTED_SAMPLES 1\n" : "", FrameSampleCount > 1);
        return program != 0 && CheckGL("Prepare weighted transparency resources");
    }

    std::uint32_t ProbeSampleCounts()
    {
        if (!MultisampleAPI || !HalfFloatTargetsSupported) return 1;
        std::uint32_t supported = Techniques::AllMultisampleCounts;
        // Query the intersection for the actual texture formats, not the global
        // renderbuffer maximum. Geometry, depth and material attachments must agree.
        for (GLenum format : {GL_SRGB8_ALPHA8, GL_RGBA16F, GL_DEPTH_COMPONENT32F})
        {
            GLint count = 0;
            GlGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, format, GL_NUM_SAMPLE_COUNTS, 1, &count);
            if (count <= 0 || GlGetError() != GL_NO_ERROR) return 1;
            std::vector<GLint> samples(static_cast<std::size_t>(count));
            GlGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, format, GL_SAMPLES, count, samples.data());
            if (GlGetError() != GL_NO_ERROR) return 1;
            std::uint32_t formatCounts = 1;
            for (auto value : samples) if (value == 2 || value == 4 || value == 8) formatCounts |= static_cast<std::uint32_t>(value);
            supported &= formatCounts;
        }
        return supported | 1u;
    }

    void DestroyMultisampleTargets(MultisampleTargets& targets)
    {
        for (auto framebuffer : targets.Framebuffers)
            if (framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &framebuffer);
        for (auto texture : targets.Textures)
            if (texture && GlDeleteTextures) GlDeleteTextures(1, &texture);
        if (targets.Depth && GlDeleteTextures) GlDeleteTextures(1, &targets.Depth);
        targets = {};
    }

    bool EnsureMultisampleResources()
    {
        const bool material = Hair.Active || GIActive || DeferredActive || AmbientOcclusionActive || ReflectionActive;
        const GLenum format = HdrActive ? GL_RGBA16F : GL_SRGB8_ALPHA8;
        if (!Multisample.Depth || Multisample.Width != CreateInfo.Width || Multisample.Height != CreateInfo.Height ||
            Multisample.Samples != FrameSampleCount || Multisample.ColorFormat != format ||
            (material && !Multisample.Framebuffers[SampleMaterialFBO]) || (VelocityActive && !Multisample.Framebuffers[SampleVelocityFBO]))
        {
            UnbindSampledTextures();
            MultisampleTargets candidate;
            candidate.Width = CreateInfo.Width; candidate.Height = CreateInfo.Height;
            candidate.Samples = FrameSampleCount; candidate.ColorFormat = format;
            const auto makeTexture = [&](GLuint& texture, GLenum internalFormat)
            {
                GlGenTextures(1, &texture);
                GlBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
                GlTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, static_cast<GLsizei>(FrameSampleCount),
                    internalFormat, CreateInfo.Width, CreateInfo.Height, GL_TRUE_VALUE);
            };
            makeTexture(candidate.Depth, GL_DEPTH_COMPONENT32F);
            makeTexture(candidate.Textures[SampleColor], format);
            if (VelocityActive) makeTexture(candidate.Textures[SampleVelocity], GL_RGBA16F);
            if (material) for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
                makeTexture(candidate.Textures[SampleMaterialFirst + index], GL_RGBA16F);
            bool complete = true;
            for (std::size_t index = 0; index < candidate.Framebuffers.size(); ++index)
            {
                if ((index == SampleMaterialFBO && !material) || (index == SampleVelocityFBO && !VelocityActive)) continue;
                GlGenFramebuffers(1, &candidate.Framebuffers[index]);
                GlBindFramebuffer(GL_FRAMEBUFFER, candidate.Framebuffers[index]);
                std::array<GLenum, GBufferAttachmentCount> attachments {};
                const auto count = index == SampleMaterialFBO ? GBufferAttachmentCount : 1;
                for (std::size_t target = 0; target < count; ++target)
                {
                    attachments[target] = GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(target);
                    const auto textureIndex = index == SampleMaterialFBO ? SampleMaterialFirst + target :
                        (index == SampleVelocityFBO ? SampleVelocity : SampleColor);
                    GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[target], GL_TEXTURE_2D_MULTISAMPLE, candidate.Textures[textureIndex], 0);
                }
                if (index != SampleLightingFBO)
                    GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE, candidate.Depth, 0);
                GlDrawBuffers(static_cast<GLsizei>(count), attachments.data());
                complete &= GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
            }
            // Fixed sample locations are shared by every attachment of a given
            // count. Query the bound FBO instead of embedding a vendor pattern.
            GlBindFramebuffer(GL_FRAMEBUFFER, candidate.Framebuffers[SampleColorFBO]);
            for (GLuint index = 0; index < FrameSampleCount; ++index)
                GlGetMultisamplefv(GL_SAMPLE_POSITION, index, candidate.Locations[index].data());
            GlBindFramebuffer(GL_FRAMEBUFFER, 0);
            GlBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
            const bool valid = CheckGL("Create multisample attachments");
            if (!valid || !complete)
            {
                DestroyMultisampleTargets(candidate);
                return valid ? Fail("OpenGL ES multisample attachments are incomplete or unsupported") : false;
            }
            DestroyMultisampleTargets(Multisample);
            Multisample = candidate;
        }
        const char* definitions[] = {"", "#define EVANS_RESOLVE_DEPTH 1\n", "#define EVANS_RESOLVE_MATERIAL 1\n",
            "#define EVANS_RESOLVE_VELOCITY 1\n", "#define EVANS_RESTORE_COLOR 1\n"};
        const char* inputs[] = {"ColorSamples", "DepthSamples", "DiffuseSamples", "NormalSamples", "SpecularSamples",
            "AmbientSamples", "EmissionSamples", "VelocitySamples", "FilteredColor"};
        for (std::size_t index = 0; index < SamplePrograms.size(); ++index)
        {
            if (SamplePrograms[index]) continue;
            SamplePrograms[index] = CompileStageProgram("MSAA.glsl", definitions[index]);
            if (!SamplePrograms[index]) return false;
            GlUseProgram(SamplePrograms[index]);
            for (std::size_t unit = 0; unit < std::size(inputs); ++unit)
                GlUniform1i(GlGetUniformLocation(SamplePrograms[index], inputs[unit]), static_cast<GLint>(unit));
        }
        return CheckGL("Prepare multisample programs");
    }

    bool DrawSampleResolve(SampleProgram phase, GLuint framebuffer)
    {
        UnbindSampledTextures();
        GlDisable(GL_SAMPLE_MASK); GlSampleMaski(0, std::numeric_limits<GLuint>::max());
        GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GlViewport(0, 0, CreateInfo.Width, CreateInfo.Height);
        GlDisable(GL_BLEND); GlDisable(GL_CULL_FACE); GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL);
        const bool depth = phase == SampleProgram::Depth || phase == SampleProgram::Material;
        if (depth) { GlEnable(GL_DEPTH_TEST); GlDepthFunc(GL_ALWAYS); }
        else GlDisable(GL_DEPTH_TEST);
        GlDepthMask(depth ? GL_TRUE_VALUE : GL_FALSE_VALUE);
        const auto color = phase == SampleProgram::Depth ? GL_FALSE_VALUE : GL_TRUE_VALUE;
        GlColorMask(color, color, color, color);
        const auto program = SamplePrograms[static_cast<std::size_t>(phase)];
        GlUseProgram(program);
        GlUniform1i(GlGetUniformLocation(program, "SampleCount"), static_cast<GLint>(FrameSampleCount));
        const auto bind = [&](GLuint unit, GLenum target, GLuint texture)
        {
            GlActiveTexture(GL_TEXTURE0 + unit); GlBindTexture(target, texture);
        };
        if (phase == SampleProgram::Color) bind(0, GL_TEXTURE_2D_MULTISAMPLE, Multisample.Textures[SampleColor]);
        if (depth || phase == SampleProgram::Velocity) bind(1, GL_TEXTURE_2D_MULTISAMPLE, Multisample.Depth);
        if (phase == SampleProgram::Material) for (GLuint index = 0; index < GBufferAttachmentCount; ++index)
            bind(2 + index, GL_TEXTURE_2D_MULTISAMPLE, Multisample.Textures[SampleMaterialFirst + index]);
        if (phase == SampleProgram::Velocity) bind(7, GL_TEXTURE_2D_MULTISAMPLE, Multisample.Textures[SampleVelocity]);
        if (phase == SampleProgram::Restore) bind(8, GL_TEXTURE_2D, HdrTextures[0]);
        GlBindVertexArray(VertexArray);
        constexpr GLsizei FullscreenTriangleVertices = 3;
        GlDrawArrays(GL_TRIANGLES, 0, FullscreenTriangleVertices);
        UnbindSampledTextures();
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        GlDepthMask(GL_TRUE_VALUE); GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        return CheckGL("Resolve multisample inputs");
    }

    bool NeedsSampleColorRestore() const
    {
        return SubsurfaceRequested || SubsurfaceResolved || GIActive || AmbientOcclusionActive || ReflectionActive || FogActive || LightShaftActive || TemporalActive || MotionBlurActive || DepthOfFieldActive;
    }
    bool OpaqueEffectsComplete() const
    {
        return SubsurfaceComplete() && ((!GIActive || GI.Resolved) && (!AmbientOcclusionActive || AmbientOcclusionResolved) && (!ReflectionActive || ReflectionResolved) &&
            (!TemporalActive || TemporalResolved) && (!FogActive || FogResolved) && (!LightShaftActive || LightShaftResolved) &&
            (!MotionBlurActive || MotionBlurResolved) &&
            (!DepthOfFieldActive || DepthOfFieldResolved));
    }

    bool SupportsHalfFloatTargets() const
    {
        // ES 3.0 supports half-float sampling; renderability needs an extension
        // (or ES 3.2), and the concrete FBO still has to be complete.
        const auto* extensionBytes = GlGetString(GL_EXTENSIONS);
        const auto* versionBytes = GlGetString(GL_VERSION);
        const std::string extensions = " " + std::string(extensionBytes ? reinterpret_cast<const char*>(extensionBytes) : "") + " ";
        const std::string version = versionBytes ? reinterpret_cast<const char*>(versionBytes) : "";
        return extensions.find(" GL_EXT_color_buffer_float ") != std::string::npos ||
            extensions.find(" GL_EXT_color_buffer_half_float ") != std::string::npos ||
            version.find("OpenGL ES 3.2") != std::string::npos;
    }

    bool ProbeFloatDepthSupport()
    {
        // SAO depth and TAA depth history need R32F; half-float support is insufficient.
        // EXT_color_buffer_float defines single-channel float render/copy support.
        const auto* bytes = GlGetString(GL_EXTENSIONS);
        const auto* versionBytes = GlGetString(GL_VERSION);
        const std::string extensions = " " + std::string(bytes ? reinterpret_cast<const char*>(bytes) : "") + " ";
        const std::string version = versionBytes ? reinterpret_cast<const char*>(versionBytes) : "";
        if (extensions.find(" GL_EXT_color_buffer_float ") == std::string::npos &&
            version.find("OpenGL ES 3.2") == std::string::npos) return false;
        GLuint texture = 0, framebuffer = 0;
        GlGenTextures(1, &texture);
        GlBindTexture(GL_TEXTURE_2D, texture);
        ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT, nullptr);
        GlGenFramebuffers(1, &framebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &attachment);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        const bool clean = GlGetError() == GL_NO_ERROR;
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        GlBindTexture(GL_TEXTURE_2D, 0);
        if (framebuffer) GlDeleteFramebuffers(1, &framebuffer);
        if (texture) GlDeleteTextures(1, &texture);
        return complete && clean;
    }

    bool EnsureHdrTargets()
    {
        if (HdrTextures[0]) return true;
        if (!HalfFloatTargetsSupported)
            return Fail("HDR requires OpenGL ES half-float color attachments (EXT_color_buffer_float or EXT_color_buffer_half_float)");
        for (std::size_t index = 0; index < HdrTextures.size(); ++index)
        {
            const bool halfResolution = index >= 2;
            const int width = halfResolution ? (CreateInfo.Width + 1) / 2 : CreateInfo.Width;
            const int height = halfResolution ? (CreateInfo.Height + 1) / 2 : CreateInfo.Height;
            GlGenTextures(1, &HdrTextures[index]);
            GlBindTexture(GL_TEXTURE_2D, HdrTextures[index]);
            ConfigureTexture();
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            GlGenFramebuffers(1, &HdrFramebuffers[index]);
            GlBindFramebuffer(GL_FRAMEBUFFER, HdrFramebuffers[index]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, HdrTextures[index], 0);
            if (index == 0) GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, DepthTexture, 0);
            const GLenum attachment = GL_COLOR_ATTACHMENT0;
            GlDrawBuffers(1, &attachment);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            {
                DestroyHdrTargets();
                return Fail("OpenGL ES RGBA16F HDR framebuffer is not supported");
            }
        }
        return CheckGL("Create HDR targets");
    }

    void DestroyDeferredTargets()
    {
        ResetVisibilityState();
        if (GBufferFramebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &GBufferFramebuffer);
        if (LightingFramebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &LightingFramebuffer);
        GBufferFramebuffer = LightingFramebuffer = 0;
        for (auto& texture : GBufferTextures)
        {
            if (texture && GlDeleteTextures) GlDeleteTextures(1, &texture);
            texture = 0;
        }
    }

    bool EnsureDeferredResources()
    {
        if (!HdrActive) return Fail("OpenGL ES deferred lighting requires HDR scene color");
        // ES 3.0 guarantees four outputs, while this explicit teaching layout
        // needs five. Query both limits and verify the concrete floating FBO.
        // OpenGL ES 3.0.6, sections 4.2.1 and 4.4.4.
        if (MaximumDrawBuffers < static_cast<GLint>(GBufferAttachmentCount) ||
            MaximumColorAttachments < static_cast<GLint>(GBufferAttachmentCount))
            return Fail("OpenGL ES deferred shading requires five draw buffers and color attachments");
        if (!GBufferFramebuffer)
        {
            GlGenFramebuffers(1, &GBufferFramebuffer);
            GlBindFramebuffer(GL_FRAMEBUFFER, GBufferFramebuffer);
            std::array<GLenum, GBufferAttachmentCount> attachments {};
            for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
            {
                GlGenTextures(1, &GBufferTextures[index]);
                GlBindTexture(GL_TEXTURE_2D, GBufferTextures[index]);
                ConfigureDepthTexture(); // Nearest sampling avoids filtering material identity.
                GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, CreateInfo.Width, CreateInfo.Height,
                    0, GL_RGBA, GL_HALF_FLOAT, nullptr);
                attachments[index] = GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(index);
                GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[index], GL_TEXTURE_2D, GBufferTextures[index], 0);
            }
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, DepthTexture, 0);
            GlDrawBuffers(static_cast<GLsizei>(attachments.size()), attachments.data());
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            {
                DestroyDeferredTargets();
                return Fail("OpenGL ES five-target floating G-buffer is incomplete");
            }
            // A separate color-only FBO prevents a depth feedback loop during
            // reconstruction. Forward transparency keeps the original HDR FBO
            // with opaque depth attached, including after aborted light passes.
            // https://registry.khronos.org/OpenGL/specs/es/3.0/es_spec_3.0.pdf (4.4.3)
            GlGenFramebuffers(1, &LightingFramebuffer);
            GlBindFramebuffer(GL_FRAMEBUFFER, LightingFramebuffer);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, HdrTextures[0], 0);
            const GLenum attachment = GL_COLOR_ATTACHMENT0;
            GlDrawBuffers(1, &attachment);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create deferred targets"))
            {
                DestroyDeferredTargets();
                return Fail("OpenGL ES deferred lighting framebuffer creation failed");
            }
        }
        if (!DeferredGeometryProgram)
        {
            DeferredGeometryProgram = CompileStageProgram("Deferred.glsl");
            if (!DeferredGeometryProgram) return false;
            GlUseProgram(DeferredGeometryProgram);
            GlUniform1i(GlGetUniformLocation(DeferredGeometryProgram, "DiffuseMap"), 1);
            GlUniform1i(GlGetUniformLocation(DeferredGeometryProgram, "NormalMap"), 2);
            GlUniform1i(GlGetUniformLocation(DeferredGeometryProgram, "OpacityMap"), 3);
        }
        if (!DeferredLightingProgram)
            DeferredLightingProgram = CompileStageProgram("Deferred.glsl", "#define EVANS_DEFERRED_LIGHTING 1\n");
        if (FrameSampleCount > 1 && !MultisampleLightingProgram)
            MultisampleLightingProgram = CompileStageProgram("Deferred.glsl",
                "#define EVANS_DEFERRED_LIGHTING 1\n#define EVANS_MULTISAMPLE 1\n", true);
        return DeferredLightingProgram != 0 && (FrameSampleCount == 1 || MultisampleLightingProgram != 0) && CheckGL("Create deferred programs");
    }

    void ResetVisibilityState()
    {
        ForwardPlus.AbortFrame(); TiledDeferred.AbortFrame(); ClusteredForward.AbortFrame(); LightListFrame.Reset();
        LightListsActive = ForwardPlusDepthWritten = false;
        if (ForwardPlusSupported)
        {
            for (GLuint slot = 0; slot < static_cast<GLuint>(ForwardPlusStorageCount); ++slot)
                GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, 0);
            GlBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
        ContactShadows.AbortFrame(); FrameContact = {};
        ContactDepthWritten = ContactGenerated = ContactReceiverActive = false;
        SamplesOpaqueResolved = SamplesVelocityResolved = false;
        SamplesColorRestored = SamplesColorResolved = SamplesTransparentWritten = false;
        DeferredActive = GBufferWritten = DeferredResolved = false;
        AmbientOcclusionActive = AmbientOcclusionResolved = ForwardColorWritten = false;
        ReflectionActive = ReflectionResolved = false;
        FrameReflections = {};
        FrameEnvironment = {};
        EnvironmentFacesWritten = 0;
        EnvironmentActive = EnvironmentReady = false;
        FramePlanar = {};
        PlanarActive = PlanarWritten = PlanarReady = false;
        FrameReflectionShading = Techniques::ShadingModel::Phong;
        DepthOfFieldActive = DepthOfFieldResolved = false;
        MotionBlurActive = VelocityCaptured = MotionBlurResolved = false;
        VelocityActive = TemporalActive = TemporalResolved = false;
        GI.Begin(); Probes.Begin(); GIActive = EarlyTemporalActive = ProbeActive = false;
        FogActive = FogResolved = false;
        LightShaftActive = LightShaftResolved = false;
        FrameLightShafts = {};
        WeightedActive = WeightedPass = WeightedPrepared = WeightedAccumulated = WeightedResolved = false;
        FrameTransparency = {};
        FrameFog = {};
        FrameAmbientOcclusionSettings = {};
        FramePostProcessSettings = {};
        DeferredBackground = {};
    }

    void DestroyAmbientOcclusionTargets()
    {
        DestroySaoDepthTargets();
        for (auto& framebuffer : AmbientOcclusionFramebuffers)
        {
            if (framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &framebuffer);
            framebuffer = 0;
        }
        for (auto& texture : AmbientOcclusionTextures)
        {
            if (texture && GlDeleteTextures) GlDeleteTextures(1, &texture);
            texture = 0;
        }
    }

    void DestroySaoDepthTargets()
    {
        if (SaoDepthFramebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &SaoDepthFramebuffer);
        if (SaoDepthScratch && GlDeleteTextures) GlDeleteTextures(1, &SaoDepthScratch);
        if (SaoDepthHierarchy && GlDeleteTextures) GlDeleteTextures(1, &SaoDepthHierarchy);
        SaoDepthFramebuffer = SaoDepthScratch = SaoDepthHierarchy = 0;
    }

    bool EnsureSaoDepthTargets()
    {
        if (SaoDepthHierarchy && SaoDepthScratch && SaoDepthFramebuffer) return true;
        if (!FloatDepthTargetSupported) return Fail("OpenGL ES SAO requires an R32F render target");
        DestroySaoDepthTargets();
        const auto levels = AmbientOcclusionDepthMipCount(CreateInfo.Width, CreateInfo.Height);
        GlGenTextures(1, &SaoDepthHierarchy);
        GlBindTexture(GL_TEXTURE_2D, SaoDepthHierarchy);
        ConfigureDepthTexture();
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(levels - 1));
        for (std::uint32_t level = 0; level < levels; ++level)
            GlTexImage2D(GL_TEXTURE_2D, static_cast<GLint>(level), GL_R32F,
                std::max(1, CreateInfo.Width >> level), std::max(1, CreateInfo.Height >> level),
                0, GL_RED, GL_FLOAT, nullptr);
        GlGenTextures(1, &SaoDepthScratch);
        GlBindTexture(GL_TEXTURE_2D, SaoDepthScratch);
        ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, CreateInfo.Width, CreateInfo.Height, 0, GL_RED, GL_FLOAT, nullptr);
        GlGenFramebuffers(1, &SaoDepthFramebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, SaoDepthFramebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, SaoDepthScratch, 0);
        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &attachment);
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create SAO depth hierarchy"))
        {
            DestroySaoDepthTargets();
            return Fail("OpenGL ES SAO depth hierarchy allocation failed");
        }
        return true;
    }

    #include "Renderer/OpenGLES/OpenGLESRayScene.inl"
    #include "Renderer/OpenGLES/OpenGLESRayRadiance.inl"
    #include "Renderer/OpenGLES/OpenGLESRayReflections.inl"
    #include "Renderer/OpenGLES/OpenGLESHybridRendering.inl"
    #include "Renderer/OpenGLES/OpenGLESSubsurfaceScattering.inl"
    #include "Renderer/OpenGLES/OpenGLESHairFur.inl"
    #include "Renderer/OpenGLES/OpenGLESTemporalEffects.inl"
    #include "Renderer/OpenGLES/OpenGLESParallaxOcclusionMapping.inl"
    #include "Renderer/OpenGLES/OpenGLESDisplacementMapping.inl"
    #include "Renderer/OpenGLES/OpenGLESTessellation.inl"
    #include "Renderer/VirtualGeometry/OpenGLES/OpenGLESVirtualGeometryCache.inl"
    #include "Renderer/VirtualGeometry/OpenGLES/OpenGLESVirtualGeometry.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteCache.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteGeometry.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteAsync.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteCacheAsync.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteFrameVisibility.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteDraws.inl"
#include "Renderer/OpenGLES/OpenGLESTemporalSuperResolution.inl"
#include "Renderer/VirtualShadowMaps/OpenGLES/OpenGLESVirtualShadowMaps.inl"
#include "Renderer/VirtualShadowMaps/OpenGLES/OpenGLESVirtualShadowAtlas.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenSurface.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenCaptureAsync.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenSurfaceLookup.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenDistanceTrace.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenGlobalDistance.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenGlobalDistanceAsync.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenSurfaceLighting.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenLightingAsync.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenRadianceTrace.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenProbeIntegration.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenProbeGather.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenTranslucency.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenHairReceiver.inl"
#include "Renderer/OpenGLES/OpenGLESRayTracedShadows.inl"
#include "Renderer/OpenGLES/OpenGLESPCF.inl"
#include "Renderer/OpenGLES/OpenGLESMaterials.inl"

    bool UsesMaterialSurface(const PipelineState& pipeline,PassType pass) const
    {
        return pass==PassType::Color && !pipeline.OutlineExtrusion &&
            FrameShadowFilter.Mode==Techniques::ShadowFilterMode::Hard &&
            !FrameVirtualShadowAtlas && !FrameContact.Enabled && !FrameAreaLights && !LightListsActive && !FrameRayShadowImage;
    }
    bool UsesMaterialStage(const PipelineState& pipeline,PassType pass) const
    { return (pass==PassType::GBuffer && !UsesParallaxMaterial(pipeline,pass)) || UsesMaterialSurface(pipeline,pass); }

    bool UsesDedicatedPCF() const
    { return FrameShadowFilter.Mode==Techniques::ShadowFilterMode::PCF && !FrameVirtualShadowAtlas && !FrameContact.Enabled && !FrameAreaLights && !LightListsActive; }
    bool UsesSurfacePCF() const
    { return UsesDedicatedPCF() && !FrameAreaLights && CurrentPass.Type==PassType::Color && BoundPipeline.SamplesShadowMap; }

#include "Renderer/Lumen/OpenGLES/OpenGLESLumenTemporal.inl"
#include "Renderer/Lumen/OpenGLES/OpenGLESLumenReflections.inl"
#include "Renderer/VirtualShadowMaps/OpenGLES/OpenGLESVirtualShadowReceivers.inl"
#include "Renderer/VirtualShadowMaps/OpenGLES/OpenGLESVirtualShadowLookup.inl"
    #include "Renderer/OpenGLES/OpenGLESRayGI.inl"
    #include "Renderer/OpenGLES/OpenGLESVoxelGI.inl"
    #include "Renderer/OpenGLES/OpenGLESPathTracing.inl"
    #include "Renderer/OpenGLES/OpenGLESGPUDriven.inl"
    #include "Renderer/OpenGLES/OpenGLESVisibilityBuffer.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteMaterials.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteFrameRaster.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteVisibilityGeometry.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteInstances.inl"
    #include "Renderer/Nanite/OpenGLES/OpenGLESNaniteHardware.inl"

    bool EnsureAmbientOcclusionResources()
    {
        const bool rayTraced = FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::RayTraced;
        if (rayTraced && !EnsureRayScene()) return false;
        const bool horizonBased = FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::HBAO;
        const bool scalable = FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::SAO;
        const bool groundTruth = FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::GTAO;
        GLuint& program = rayTraced ? RayAOProgram : groundTruth ? GtaoProgram : scalable ? SaoProgram : horizonBased ? HbaoProgram : SsaoProgram;
        if (!program) program = CompileStageProgram(rayTraced ? "RayTracedAO.glsl" : groundTruth ? "GTAO.glsl" : scalable ? "SAO.glsl" : horizonBased ? "HBAO.glsl" : "SSAO.glsl");
        if (!program) return false;
        if (AmbientOcclusionFramebuffers[0] && AmbientOcclusionFramebuffers[1])
            return !scalable || EnsureSaoDepthTargets();
        DestroyAmbientOcclusionTargets();
        for (std::size_t index = 0; index < AmbientOcclusionTextures.size(); ++index)
        {
            GlGenTextures(1, &AmbientOcclusionTextures[index]);
            GlBindTexture(GL_TEXTURE_2D, AmbientOcclusionTextures[index]);
            ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, CreateInfo.Width, CreateInfo.Height,
                0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            GlGenFramebuffers(1, &AmbientOcclusionFramebuffers[index]);
            GlBindFramebuffer(GL_FRAMEBUFFER, AmbientOcclusionFramebuffers[index]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, AmbientOcclusionTextures[index], 0);
            const GLenum attachment = GL_COLOR_ATTACHMENT0;
            GlDrawBuffers(1, &attachment);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create AO targets"))
            {
                DestroyAmbientOcclusionTargets();
                return Fail("OpenGL ES ambient occlusion requires complete RGBA16F color attachments");
            }
        }
        return !scalable || EnsureSaoDepthTargets();
    }

    void UnbindSampledTextures()
    {
        // Clear material, deferred and cascade inputs before binding attachments
        // so a previous color draw cannot create a depth-texture feedback loop.
        constexpr std::size_t SampledTextureCount = ContactTextureUnit + 1;
        for (std::size_t unit = 0; unit < SampledTextureCount; ++unit)
        {
            GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
            GlBindTexture(GL_TEXTURE_2D, 0);
            if (MultisampleAPI) GlBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
            GlBindTexture(GL_TEXTURE_CUBE_MAP, 0);
            GlBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        }
        GlActiveTexture(GL_TEXTURE0);
    }

    struct EnvironmentTargets
    {
        GLuint Cube = 0, Depth = 0;
        std::array<GLuint, EnvironmentFaceCount> Framebuffers {};
        std::uint32_t Resolution = 0, MipCount = 0;
    } Environment;

    void DestroyEnvironmentTargets(EnvironmentTargets& targets)
    {
        if (GlDeleteFramebuffers)
            for (auto framebuffer : targets.Framebuffers)
                if (framebuffer) GlDeleteFramebuffers(1, &framebuffer);
        if (GlDeleteTextures)
        {
            if (targets.Cube) GlDeleteTextures(1, &targets.Cube);
            if (targets.Depth) GlDeleteTextures(1, &targets.Depth);
        }
        targets = {};
    }

    bool PrepareEnvironmentCapture(const EnvironmentCaptureFrame& frame)
    {
        EnvironmentActive = Techniques::UsesEnvironmentTexture(FrameReflections);
        if (!EnvironmentActive) return true;
        if (!ValidEnvironmentSource(frame) || (!frame.Source && frame.Resolution != FrameReflections.CaptureResolution) || frame.MipCount == 0)
            return Fail("OpenGL ES environment reflections need a matching capture frame before BeginFrame");
        std::uint32_t mipCount = 0;
        for (auto extent = frame.Resolution; extent; extent >>= 1) ++mipCount;
        if (frame.MipCount != mipCount) return Fail("OpenGL ES capture requires the complete cube mip chain");
        for (const auto& face : frame.Faces)
            for (float value : face.ViewProjection.Values)
                if (!std::isfinite(value)) return Fail("OpenGL ES capture projections must be finite");
        if (Environment.Resolution != frame.Resolution)
        {
            // Allocate a replacement before releasing the old capture. Every face
            // has its own color attachment and shares private single-sample depth.
            EnvironmentTargets candidate;
            UnbindSampledTextures();
            const auto discard = [&]()
            {
                GlBindFramebuffer(GL_FRAMEBUFFER, 0);
                UnbindSampledTextures();
                DestroyEnvironmentTargets(candidate);
                return false;
            };
            GlGenTextures(1, &candidate.Cube);
            GlBindTexture(GL_TEXTURE_CUBE_MAP, candidate.Cube);
            GlTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            GlTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            for (auto coordinate : {GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R})
                GlTexParameteri(GL_TEXTURE_CUBE_MAP, coordinate, GL_CLAMP_TO_EDGE);
            GlTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(frame.MipCount - 1));
            for (std::uint32_t face = 0; face < EnvironmentFaceCount; ++face)
                for (std::uint32_t mip = 0; mip < frame.MipCount; ++mip)
                    GlTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, static_cast<GLint>(mip), GL_RGBA16F,
                        static_cast<GLsizei>(frame.Resolution >> mip), static_cast<GLsizei>(frame.Resolution >> mip),
                        0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            GlBindTexture(GL_TEXTURE_CUBE_MAP, 0);
            GlGenTextures(1, &candidate.Depth);
            GlBindTexture(GL_TEXTURE_2D, candidate.Depth);
            ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, frame.Resolution, frame.Resolution,
                0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            GlBindTexture(GL_TEXTURE_2D, 0);
            GlGenFramebuffers(static_cast<GLsizei>(EnvironmentFaceCount), candidate.Framebuffers.data());
            const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
            for (std::uint32_t face = 0; face < EnvironmentFaceCount; ++face)
            {
                GlBindFramebuffer(GL_FRAMEBUFFER, candidate.Framebuffers[face]);
                GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, candidate.Cube, 0);
                GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, candidate.Depth, 0);
                GlDrawBuffers(1, &colorAttachment);
                if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                {
                    Fail("OpenGL ES cubemap needs complete RGBA16F face attachments"); return discard();
                }
            }
            GlBindFramebuffer(GL_FRAMEBUFFER, 0);
            if (!CheckGL("Allocate six-face environment capture")) return discard();
            candidate.Resolution = frame.Resolution; candidate.MipCount = frame.MipCount;
            DestroyEnvironmentTargets(Environment);
            Environment = candidate;
        }
        FrameEnvironment = frame;
        return true;
    }

    struct EnvironmentMapTargets
    {
        GLuint Image = 0, Framebuffer = 0;
        EnvironmentMapLayout Layout;
    } EnvironmentMap;

    void DestroyEnvironmentMapTargets(EnvironmentMapTargets& targets)
    {
        if (targets.Framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &targets.Framebuffer);
        if (targets.Image && GlDeleteTextures) GlDeleteTextures(1, &targets.Image);
        targets = {};
    }

    bool EnsureEnvironmentMapResources()
    {
        const auto layout = BuildEnvironmentMapLayout(FrameEnvironment.Resolution);
        if (!layout.MipCount || layout.Width > static_cast<std::uint32_t>(MaximumTextureSize))
            return Fail("OpenGL ES latitude-longitude dimensions exceed the device texture limit");
        if (EnvironmentMap.Image && EnvironmentMap.Layout.Width == layout.Width) return true;
        EnvironmentMapTargets candidate;
        UnbindSampledTextures();
        GlGenTextures(1, &candidate.Image);
        GlBindTexture(GL_TEXTURE_2D, candidate.Image);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(layout.MipCount - 1));
        for (std::uint32_t mip = 0; mip < layout.MipCount; ++mip)
            GlTexImage2D(GL_TEXTURE_2D, static_cast<GLint>(mip), GL_RGBA16F,
                std::max(1u, layout.Width >> mip), std::max(1u, layout.Height >> mip),
                0, GL_RGBA, GL_HALF_FLOAT, nullptr);
        GlBindTexture(GL_TEXTURE_2D, 0);
        GlGenFramebuffers(1, &candidate.Framebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, candidate.Framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, candidate.Image, 0);
        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &attachment);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        const bool valid = CheckGL("Allocate latitude-longitude image");
        if (!complete || !valid)
        {
            DestroyEnvironmentMapTargets(candidate);
            return Fail("OpenGL ES latitude-longitude RGBA16F framebuffer is incomplete");
        }
        candidate.Layout = layout;
        DestroyEnvironmentMapTargets(EnvironmentMap);
        EnvironmentMap = candidate;
        return true;
    }

    bool ConvertEnvironmentMap()
    {
        // Capture and conversion have separate attachments. Unbind the destination
        // before filtering it; the next geometry pass establishes its own state.
        UnbindSampledTextures();
        GlBindFramebuffer(GL_FRAMEBUFFER, EnvironmentMap.Framebuffer);
        GlViewport(0, 0, EnvironmentMap.Layout.Width, EnvironmentMap.Layout.Height);
        GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE); GlDisable(GL_BLEND);
        GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL);
        GlDepthMask(GL_FALSE_VALUE);
        GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        GlUseProgram(EnvironmentConversionProgram);
        GlUniform4f(GlGetUniformLocation(EnvironmentConversionProgram, "Viewport"),
            float(EnvironmentMap.Layout.Width), float(EnvironmentMap.Layout.Height),
            1.0f / EnvironmentMap.Layout.Width, 1.0f / EnvironmentMap.Layout.Height);
        GlUniform1i(GlGetUniformLocation(EnvironmentConversionProgram, "CapturedFaces"), 0);
        GlBindTexture(GL_TEXTURE_CUBE_MAP, Environment.Cube);
        GlBindVertexArray(VertexArray);
        GlDrawArrays(GL_TRIANGLES, 0, 3);
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        UnbindSampledTextures();
        GlDepthMask(GL_TRUE_VALUE);
        if (!CheckGL("Convert capture faces to latitude-longitude")) return false;
        ++PendingFrame.Statistics.EnvironmentConversionPassCount;
        GlBindTexture(GL_TEXTURE_2D, EnvironmentMap.Image);
        // ES mip generation requires a filterable, color-renderable base format;
        // the HDR device gate and framebuffer check establish that for RGBA16F.
        // https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glGenerateMipmap.xhtml
        GlGenerateMipmap(GL_TEXTURE_2D);
        GlBindTexture(GL_TEXTURE_2D, 0);
        return CheckGL("Generate latitude-longitude image mips");
    }

    void DestroyReflectionTargets()
    {
        if (ReflectionFramebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &ReflectionFramebuffer);
        if (ReflectionTexture && GlDeleteTextures) GlDeleteTextures(1, &ReflectionTexture);
        ReflectionFramebuffer = ReflectionTexture = 0;
    }

    struct PlanarTargets
    {
        GLuint Image = 0, Depth = 0, Scratch = 0;
        GLuint CaptureFramebuffer = 0, ReductionFramebuffer = 0;
        std::uint32_t Width = 0, Height = 0;
    } Planar;

    void DestroyPlanarTargets(PlanarTargets& targets)
    {
        if (GlDeleteFramebuffers)
            for (auto framebuffer : {targets.CaptureFramebuffer, targets.ReductionFramebuffer})
                if (framebuffer) GlDeleteFramebuffers(1, &framebuffer);
        if (GlDeleteTextures)
            for (auto texture : {targets.Image, targets.Depth, targets.Scratch})
                if (texture) GlDeleteTextures(1, &texture);
        targets = {};
    }

    bool PreparePlanarCapture(const PlanarCaptureFrame& frame)
    {
        PlanarActive = Techniques::UsesPlanarCapture(FrameReflections);
        if (!PlanarActive) return true;
        ReflectionDesc desc; desc.ViewProjection = frame.MainViewProjection;
        PlanarReflectionConstants constants;
        if (!BuildPlanarReflectionConstants(desc, frame, FrameReflections, FrameReflectionShading, CreateInfo.Width, CreateInfo.Height, constants))
            return Fail("OpenGL ES planar capture requires a finite camera and matching main viewport before BeginFrame");
        std::uint32_t expectedMips = 0;
        for (auto extent = std::max(frame.Width, frame.Height); extent; extent >>= 1) ++expectedMips;
        const auto divisor = FrameReflections.Planar.ResolutionDivisor;
        if (frame.MipCount != expectedMips || frame.Width != (frame.MainWidth + divisor - 1) / divisor ||
            frame.Height != (frame.MainHeight + divisor - 1) / divisor ||
            std::max(frame.Width, frame.Height) > static_cast<std::uint32_t>(MaximumTextureSize))
            return Fail("OpenGL ES planar capture extent or mip chain is inconsistent or exceeds the device limit");
        for (float value : frame.ClippedViewProjection.Values)
            if (!std::isfinite(value)) return Fail("OpenGL ES oblique capture projection must be finite");
        if (Planar.Width != frame.Width || Planar.Height != frame.Height)
        {
            PlanarTargets candidate;
            UnbindSampledTextures();
            const auto discard = [&]()
            {
                GlBindFramebuffer(GL_FRAMEBUFFER, 0); UnbindSampledTextures();
                DestroyPlanarTargets(candidate); return false;
            };
            GlGenTextures(1, &candidate.Image);
            GlBindTexture(GL_TEXTURE_2D, candidate.Image);
            ConfigureDepthTexture();
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(frame.MipCount - 1));
            for (std::uint32_t mip = 0; mip < frame.MipCount; ++mip)
                GlTexImage2D(GL_TEXTURE_2D, static_cast<GLint>(mip), GL_RGBA16F,
                    std::max(1u, frame.Width >> mip), std::max(1u, frame.Height >> mip), 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            GlGenTextures(1, &candidate.Scratch);
            GlBindTexture(GL_TEXTURE_2D, candidate.Scratch); ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, std::max(1u, frame.Width >> 1), std::max(1u, frame.Height >> 1),
                0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            GlGenTextures(1, &candidate.Depth);
            GlBindTexture(GL_TEXTURE_2D, candidate.Depth); ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, frame.Width, frame.Height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            GlBindTexture(GL_TEXTURE_2D, 0);
            GlGenFramebuffers(1, &candidate.CaptureFramebuffer);
            GlGenFramebuffers(1, &candidate.ReductionFramebuffer);
            const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
            for (bool capture : {true, false})
            {
                GlBindFramebuffer(GL_FRAMEBUFFER, capture ? candidate.CaptureFramebuffer : candidate.ReductionFramebuffer);
                GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, capture ? candidate.Image : candidate.Scratch, 0);
                if (capture) GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, candidate.Depth, 0);
                GlDrawBuffers(1, &colorAttachment);
                if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                {
                    Fail("OpenGL ES planar capture and reduction require complete RGBA16F targets"); return discard();
                }
            }
            GlBindFramebuffer(GL_FRAMEBUFFER, 0);
            if (!CheckGL("Allocate planar capture and mip targets")) return discard();
            candidate.Width = frame.Width; candidate.Height = frame.Height;
            DestroyPlanarTargets(Planar); Planar = candidate;
        }
        FramePlanar = frame;
        return true;
    }

    bool EnsureReflectionResources()
    {
            // Sky lookup and authored cube upload also need the cube mip
            // pipeline when the selected surface-reflection method is None.
            if (FrameReflections.Skybox)
            {
                const auto mode=FrameReflections.Mode;
                FrameReflections.Mode=Techniques::ReflectionMode::Cubemap;
                const bool ready=EnsureReflectionResources(false);
                FrameReflections.Mode=mode;
                if (!ready) return false;
            }

        if (PlanarActive && !EnsureReflectionResources(true)) return false;
        return FrameReflections.Mode==Techniques::ReflectionMode::None || EnsureReflectionResources(false);
    }

    bool EnsureReflectionResources(bool planarPass)
    {
        if (Techniques::UsesLumenReflections(FrameReflections)) return true; // Native resources are sized after the bounded query.
            if (FrameReflections.Mode==Techniques::ReflectionMode::RayTraced)
            return Techniques::UsesHybridRendering(FrameRayRadianceSettings.Pipeline) ? EnsureHybridRenderingResources() : EnsureRayReflectionResources();
        if (!HdrActive || !LightingFramebuffer) return Fail("OpenGL ES reflections require HDR and material targets");
        const bool latitudeLongitude = !planarPass && FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude;
        GLuint& program = planarPass ? PlanarProgram : latitudeLongitude ? EnvironmentMapProgram : FrameReflections.Mode==Techniques::ReflectionMode::Cubemap ? CubeReflectionProgram : ReflectionProgram;
        if (!program) program = CompileStageProgram(planarPass ? "PlanarReflections.glsl" : latitudeLongitude ? "EnvironmentMapReflections.glsl" :
            FrameReflections.Mode==Techniques::ReflectionMode::Cubemap ? "CubemapReflections.glsl" : "SSR.glsl");
        if (!program) return false;
        if (planarPass)
        {
            if (!PlanarMipProgram) PlanarMipProgram = CompileStageProgram("PlanarReflections.glsl", "#define EVANS_PLANAR_MIP 1\n");
            if (!PlanarMipProgram) return false;
        }
        if (latitudeLongitude)
        {
            if (!EnvironmentConversionProgram) EnvironmentConversionProgram = CompileStageProgram(
                "EnvironmentMapReflections.glsl", "#define EVANS_ENVIRONMENT_CONVERSION 1\n");
            if (!EnvironmentConversionProgram || !EnsureEnvironmentMapResources()) return false;
        }
        if (ReflectionFramebuffer) return true;
        UnbindSampledTextures();
        GlGenTextures(1, &ReflectionTexture);
        GlBindTexture(GL_TEXTURE_2D, ReflectionTexture);
        ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, CreateInfo.Width, CreateInfo.Height, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
        GlBindTexture(GL_TEXTURE_2D, 0);
        GlGenFramebuffers(1, &ReflectionFramebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, ReflectionFramebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ReflectionTexture, 0);
        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &attachment);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (!complete || !CheckGL("Create SSR radiance target"))
        {
            DestroyReflectionTargets();
            return Fail("OpenGL ES SSR floating framebuffer is incomplete");
        }
        return true;
    }

    bool EnsureDepthOfFieldResources()
    {
        if (!HdrActive || !HdrTextures[0] || !HdrTextures[1])
            return Fail("OpenGL ES depth of field requires HDR scene targets");
        if (!DepthOfFieldProgram) DepthOfFieldProgram = CompileStageProgram("DepthOfField.glsl");
        if (!DepthOfFieldProgram) return false;
        if (DepthOfFieldFramebuffer) return true;
        GLuint framebuffer = 0;
        GlGenFramebuffers(1, &framebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        // HDR0's geometry FBO has depth attached. Sampling that depth requires
        // this independent color-only FBO, without allocating a G-buffer.
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, HdrTextures[0], 0);
        const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &colorAttachment);
        if (!framebuffer || GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
            !CheckGL("Create depth-of-field framebuffer"))
        {
            if (framebuffer) GlDeleteFramebuffers(1, &framebuffer);
            return Fail("OpenGL ES depth-of-field framebuffer is incomplete");
        }
        DepthOfFieldFramebuffer = framebuffer;
        return true;
    }

    void DestroyMotionTargets()
    {
        for (auto& framebuffer : MotionFramebuffers)
        {
            if (framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &framebuffer);
            framebuffer = 0;
        }
        for (auto& texture : MotionTextures)
        {
            if (texture && GlDeleteTextures) GlDeleteTextures(1, &texture);
            texture = 0;
        }
        if (MotionCompositeFramebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &MotionCompositeFramebuffer);
        MotionCompositeFramebuffer = 0;
    }

    bool EnsureVelocityResources()
    {
        if (!HdrActive || !HdrTextures[0] || !DepthTexture)
            return Fail("OpenGL ES velocity requires HDR color and opaque depth");
        if (!MotionGeometryProgram)
            MotionGeometryProgram = CompileStageProgram("MotionVectors.glsl");
        if (!MotionGeometryProgram) return false;
        if (MotionFramebuffers[MotionVelocity]) return true;
        UnbindSampledTextures();
        GlGenTextures(1, &MotionTextures[MotionVelocity]);
        GlBindTexture(GL_TEXTURE_2D, MotionTextures[MotionVelocity]);
        ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, CreateInfo.Width, CreateInfo.Height, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
        GlBindTexture(GL_TEXTURE_2D, 0);
        GlGenFramebuffers(1, &MotionFramebuffers[MotionVelocity]);
        GlBindFramebuffer(GL_FRAMEBUFFER, MotionFramebuffers[MotionVelocity]);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, MotionTextures[MotionVelocity], 0);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, DepthTexture, 0);
        const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &colorAttachment);
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create velocity framebuffer"))
        {
            DestroyMotionTargets();
            return Fail("OpenGL ES velocity framebuffer is incomplete");
        }
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        return true;
    }

    bool EnsureMotionResources()
    {
        if (!EnsureVelocityResources()) return false;
        if (!MotionScreenProgram) MotionScreenProgram = CompileStageProgram("MotionBlur.glsl");
        if (!MotionScreenProgram) return false;
        if (MotionCompositeFramebuffer) return true;
        UnbindSampledTextures();
        const GLenum colorAttachment = GL_COLOR_ATTACHMENT0;
        const int tile = Techniques::MotionBlurTileSize;
        for (std::size_t index = MotionTileMax; index < MotionTextures.size(); ++index)
        {
            const int width = (CreateInfo.Width + tile - 1) / tile;
            const int height = (CreateInfo.Height + tile - 1) / tile;
            GlGenTextures(1, &MotionTextures[index]);
            GlBindTexture(GL_TEXTURE_2D, MotionTextures[index]);
            ConfigureDepthTexture(); // nearest, clamped, one mip; used for exact texel fetches
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            GlGenFramebuffers(1, &MotionFramebuffers[index]);
            GlBindFramebuffer(GL_FRAMEBUFFER, MotionFramebuffers[index]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, MotionTextures[index], 0);
            GlDrawBuffers(1, &colorAttachment);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create motion framebuffer"))
            {
                DestroyMotionTargets();
                return Fail("OpenGL ES motion framebuffer is incomplete");
            }
        }
        // Sampling an attachment of the current draw FBO is a feedback loop even
        // if depth testing/writes are disabled. The composite uses color only.
        // https://registry.khronos.org/OpenGL/specs/es/3.0/es_spec_3.0.pdf (4.4.3)
        GlGenFramebuffers(1, &MotionCompositeFramebuffer);
        GlBindFramebuffer(GL_FRAMEBUFFER, MotionCompositeFramebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, HdrTextures[0], 0);
        GlDrawBuffers(1, &colorAttachment);
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create motion composite framebuffer"))
        {
            DestroyMotionTargets();
            return Fail("OpenGL ES motion composite framebuffer is incomplete");
        }
        return true;
    }

    bool ProbeHalfFloatColorBlit()
    {
        // Test format renderability and same-format color blits once. Fog and
        // shafts impose their own atlas/texture-slot limits independently.
        if (!HalfFloatTargetsSupported || MaximumTextureSize < 1) return false;
        GlActiveTexture(GL_TEXTURE0);
        std::array<GLuint, 2> textures{}, framebuffers{};
        GlGenTextures(static_cast<GLsizei>(textures.size()), textures.data());
        GlGenFramebuffers(static_cast<GLsizei>(framebuffers.size()), framebuffers.data());
        bool complete = true;
        for (std::size_t i = 0; i < textures.size(); ++i)
        {
            GlBindTexture(GL_TEXTURE_2D, textures[i]);
            ConfigureDepthTexture(); // nearest texel reads, no mip chain
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 1, 1, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            GlBindFramebuffer(GL_FRAMEBUFFER, framebuffers[i]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[i], 0);
            const GLenum attachment = GL_COLOR_ATTACHMENT0;
            GlDrawBuffers(1, &attachment);
            complete &= GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        }
        if (complete)
        {
            GlBindTexture(GL_TEXTURE_2D, 0);
            GlBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffers[0]); GlReadBuffer(GL_COLOR_ATTACHMENT0);
            GlBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffers[1]);
            GlBlitFramebuffer(0, 0, 1, 1, 0, 0, 1, 1, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        }
        const bool clean = CheckGL("Probe half-float color render and blit targets");
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlBindTexture(GL_TEXTURE_2D, 0);
        GlDeleteFramebuffers(static_cast<GLsizei>(framebuffers.size()), framebuffers.data());
        GlDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
        return complete && clean;
    }

    bool CreateVolumeLightTarget(GLuint& texture, GLuint& framebuffer, int width, int height)
    {
        GlActiveTexture(GL_TEXTURE0);
        GlGenTextures(1, &texture); GlBindTexture(GL_TEXTURE_2D, texture);
        ConfigureDepthTexture(); // FP32 texelFetch needs neither filtering nor float blending.
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        GlGenFramebuffers(1, &framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &attachment);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        const bool clean = CheckGL("Allocate FP32 volume transport target");
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlBindTexture(GL_TEXTURE_2D, 0);
        if (complete && clean) return true;
        GlDeleteFramebuffers(1, &framebuffer); GlDeleteTextures(1, &texture);
        framebuffer = texture = 0;
        return clean ? Fail("OpenGL ES FP32 volume transport framebuffer is incomplete") : false;
    }

    bool ProbeVolumeLightSupport()
    {
        // RGBA32F color rendering is an additional ES capability, distinct from
        // RGBA16F ambient fog. Probe the target rather than assuming an API version.
        // https://registry.khronos.org/OpenGL/extensions/EXT/EXT_color_buffer_float.txt
        if (!FogTargetSupported) return false;
        GLuint texture = 0, framebuffer = 0;
        if (!CreateVolumeLightTarget(texture, framebuffer, 1, 1)) return false;
        GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GlClearColor(0, 0, 0, 1); GlClear(GL_COLOR_BUFFER_BIT);
        const bool rendered = CheckGL("Probe FP32 volume transport rendering");
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        GlDeleteFramebuffers(1, &framebuffer); GlDeleteTextures(1, &texture);
        return rendered;
    }

    void DestroyVolumeLightTargets()
    {
        if (GlDeleteFramebuffers) GlDeleteFramebuffers(static_cast<GLsizei>(VolumeLightFramebuffers.size()), VolumeLightFramebuffers.data());
        if (GlDeleteTextures) GlDeleteTextures(static_cast<GLsizei>(VolumeLightTextures.size()), VolumeLightTextures.data());
        VolumeLightFramebuffers = {}; VolumeLightTextures = {};
    }

    bool EnsureVolumeLightResources()
    {
        if (!VolumeLightTargetSupported || !FogFramebuffer) return Fail("OpenGL ES volume lighting requires FP32 transport and the cumulative fog atlas");
        if (Techniques::UsesBoundedMedium(FrameFog))
        {
            const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
            auto& program = cloud ? CloudVolumePrograms[LightListsActive] : MediaVolumePrograms[LightListsActive];
            if (!program) program = CompileStageProgram(cloud ? "VolumetricClouds.glsl" : "ParticipatingMedia.glsl",
                LightListsActive ? "#define EVANS_VOLUME_FULL_LIGHTS 1\n" : "", LightListsActive);
            if (!program) return false;
        }
        else
        {
            if (!VolumeLightProgram) VolumeLightProgram = CompileStageProgram("VolumetricLighting.glsl");
            if (!VolumeLightProgram) return false;
        }
        if (VolumeLightFramebuffers[0] && VolumeLightFramebuffers[1]) return true;
        UnbindSampledTextures();
        DestroyVolumeLightTargets();
        for (std::size_t index = 0; index < VolumeLightTextures.size(); ++index)
            if (!CreateVolumeLightTarget(VolumeLightTextures[index], VolumeLightFramebuffers[index], FogDimensions.AtlasWidth(), FogDimensions.AtlasHeight()))
            { DestroyVolumeLightTargets(); return false; }
        return true;
    }

    GLint VolumeLightLocation(GLuint program, const std::string& name)
    {
        auto [entry, inserted] = VolumeLightLocations[program].try_emplace(name, -1);
        if (inserted) entry->second = GlGetUniformLocation(program, name.c_str());
        return entry->second;
    }

    GLuint CompleteVolumeProgram()
    {
        const std::size_t variant = UsesVSM() ? 2 : UsesPCSS() ? 1 : 0;
        auto& program = ForwardPlusVolumePrograms[variant];
        if (!program)
        {
            const char* file = variant == 2 ? "VSM.glsl" : variant == 1 ? "PCSS.glsl" : "VolumetricLighting.glsl";
            std::string definitions = "#define EVANS_VOLUME_FULL_LIGHTS 1\n";
            if (variant) definitions += std::string("#define EVANS_") + (variant == 2 ? "VSM" : "PCSS") + "_PROGRAM 7\n";
            // Only Forward+ needs the ES 3.1 storage-buffer variant. Legacy
            // volume rendering continues to compile its own ES 3.0 source path.
            program = CompileStageProgram(file, definitions.c_str(), true);
        }
        return program;
    }
    bool GenerateVolumeLighting(const VolumetricFogDesc& requested)
    {
        auto desc = requested;
        if (LightListsActive && !LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, LastError)) return false;
            if (LightListsActive) desc.Lighting.SceneLights = LightListFrame.Description().Lights;
        const auto shadow = Textures.find(LastShadowTexture.Value);
        if (!FrameRayShadowImage && !FrameVirtualShadowAtlas && !FrameCascades.Frame.CascadeCount)
            desc.SamplesShadowMap = desc.SamplesShadowMap && shadow != Textures.end() && shadow->second.Description.Format == TextureFormat::Depth32;
        VolumetricLightingConstants lights;
        if (!BuildVolumetricLightingConstants(desc, FrameVolumeLighting, FrameCascades, FrameShadowFilter, lights, LightListsActive, &FrameFog,
                FrameVirtualShadowAtlas ? &FrameVirtualShadowAtlas->Info : nullptr, &FrameVirtualShadowSampling,
                {FogConstants.ViewportGrid.Z, FogConstants.ViewportGrid.W}))
            return Fail("OpenGL ES volume lighting requires finite lights and shadow transforms");
        const auto passes = BuildVolumetricLightingPasses(FogDimensions.Slices);
        if (passes.empty()) return Fail("OpenGL ES volume scan has no depth segments");
        const auto restore = [&]()
        {
            if (LightListsActive) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, ForwardPlusLightBinding, 0);
            UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlDepthMask(GL_TRUE_VALUE);
        };
        const bool media = Techniques::UsesBoundedMedium(FrameFog);
        const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
        const bool customInjection = !media && (LightListsActive || UsesPCSS() || UsesVSM());
        const std::array<GLuint, 2> programs {cloud ? CloudVolumePrograms[LightListsActive] : media ? MediaVolumePrograms[LightListsActive] : VolumeLightProgram,
            media ? 0 : LightListsActive ? CompleteVolumeProgram() : UsesVSM() ? VSMStageProgram(2) : UsesPCSS() ? PCSSScreenProgram(2) : 0};
        if (customInjection && !programs[1]) return false;
        for (GLuint program : programs)
        {
            if (!program) continue;
            GlUseProgram(program);
            BindRayShadowImage(program);
            UploadFogConstants(program, FogConstants);
            const auto matrix = [&](const std::string& name, const Matrix4& value)
            {
                const auto columnMajor = ToColumnMajor(value);
                GlUniformMatrix4fv(VolumeLightLocation(program, name), 1, GL_FALSE_VALUE, columnMajor.data());
            };
            if (media)
            {
                const auto& material = lights.Participating;
                const auto upload = [&](const char* name, const Float4& value) { UploadFloat4(VolumeLightLocation(program, name), value); };
                const auto uploadControl = [&](const char* name, const ShaderUInt4& value)
                { GlUniform4ui(VolumeLightLocation(program, name), value.X, value.Y, value.Z, value.W); };
                upload("MediaAbsorptionDensity", material.AbsorptionDensity);
                upload("MediaScatteringModulation", material.ScatteringModulation);
                upload("MediaEmissionFrequency", material.EmissionFrequency);
                upload("MediaBoundsMinimum", material.BoundsMinimum); upload("MediaBoundsMaximum", material.BoundsMaximum);
                uploadControl("MediaOptions", material.Options);
                upload("MediaDensityGrid",material.DensityGrid);
                if (cloud)
                {
                    upload("CloudCoverageWeather", lights.Clouds.CoverageWeather);
                    upload("CloudDetailProfile", lights.Clouds.DetailProfile);
                    upload("CloudOffsetSeed", lights.Clouds.OffsetSeed);
                    upload("CloudShape", lights.Clouds.Shape);
                }
                for (std::size_t index = 0; index < lights.Shadows.Encoding.size(); ++index)
                    UploadFloat4(VolumeLightLocation(program, "MediaShadowEncoding[" + std::to_string(index) + "]"), lights.Shadows.Encoding[index]);
                upload("MediaShadowParameters", lights.Shadows.Parameters); uploadControl("MediaShadowOptions", lights.Shadows.Options);
            }
            if (FrameVirtualShadowAtlas && lights.Control.Y)
                for (unsigned row=0;row<4;++row)
                    UploadFloat4(VolumeLightLocation(program,"VirtualCameraRows["+std::to_string(row)+"]"),lights.Shadows.Encoding[row]);
            UploadFloat4(VolumeLightLocation(program, "VolumeScattering"), lights.Scattering);
            UploadFloat4(VolumeLightLocation(program, "VolumeParameters"), lights.Parameters);
            matrix("VolumeShadowMatrix", lights.LightViewProjection);
            for (std::uint32_t index = 0; index < std::min(lights.Control.X, MaximumSceneLights); ++index)
            {
                const auto suffix = "[" + std::to_string(index) + "]";
                UploadFloat4(VolumeLightLocation(program, "VolumeLightPosition" + suffix), lights.Lights[index].PositionType);
                UploadFloat4(VolumeLightLocation(program, "VolumeLightDirection" + suffix), lights.Lights[index].DirectionIntensity);
                UploadFloat4(VolumeLightLocation(program, "VolumeLightColor" + suffix), lights.Lights[index].ColorCastShadows);
                UploadFloat4(VolumeLightLocation(program, "VolumeLightArea" + suffix), lights.Lights[index].AreaSizePadding);
            }
            const auto& cascades = lights.Cascades;
            UploadFloat4(VolumeLightLocation(program, "VolumeCascadeRange"), cascades.Parameters);
            UploadFloat4(VolumeLightLocation(program, "VolumeFilter"), cascades.Filtering);
            UploadFloat4(VolumeLightLocation(program, "VolumeEyeDepth"), cascades.EyeDepthPlane);
            GlUniform4f(VolumeLightLocation(program, "VolumeCascadeEnds"), cascades.FarDistances[0], cascades.FarDistances[1], cascades.FarDistances[2], cascades.FarDistances[3]);
            GlUniform4f(VolumeLightLocation(program, "VolumeBlendBegin"), cascades.BlendStarts[0], cascades.BlendStarts[1], cascades.BlendStarts[2], cascades.BlendStarts[3]);
            for (std::uint32_t index = 0; index < (FrameVirtualShadowAtlas ? 1u : FrameCascades.Frame.CascadeCount); ++index)
                matrix("VolumeCascadeMatrix[" + std::to_string(index) + "]", cascades.LightViewProjection[index]);
            const std::array<const char*, 6> samplerNames {"PreviousSegments", "PrimaryDepth", "CascadeDepth0", "CascadeDepth1", "CascadeDepth2", "CascadeDepth3"};
            GlUniform1i(VolumeLightLocation(program,"DensityImage"),6);
            for (std::size_t unit = 0; unit < samplerNames.size(); ++unit)
                GlUniform1i(VolumeLightLocation(program, samplerNames[unit]), static_cast<GLint>(unit));
            if (program == programs[1] && UsesPCSS() && !UploadPCSS(program, desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap)) return false;
            if (program == programs[1] && UsesVSM() && !UploadVSMReceiver(program, desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap)) return false;
        }
GLuint density=0;
        if (desc.Density)
        {
            const auto found=Textures.find(desc.Density->Texture.Value);
            if (found==Textures.end() || !ValidDensityBinding(*desc.Density,found->second.Description))
                return Fail("OpenGL ES volume density requires a matching linear RGBA8 atlas");
            density=found->second.Texture;
        }
        std::array<GLuint, 6> inputs {};
        if (FrameRayShadowImage) {}
        else if (lights.Control.Y && FrameVirtualShadowAtlas)
        { inputs[2]=FrameVirtualShadowAtlas->Depth; inputs[3]=FrameVirtualShadowAtlas->Directory; }
        else if (lights.Control.Y == 1) inputs[1] = shadow->second.Texture;
        else if (lights.Control.Y == 2)
            for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index) inputs[index + 2] = CascadeTextures[index];
        if (UsesVSM() && lights.Control.Y == 1 && !MomentTexture(LastShadowTexture, inputs[1])) return false;
        if (UsesVSM() && lights.Control.Y == 2)
            for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index)
                if (!MomentTexture(FrameCascades.Maps[index], inputs[index + 2])) return false;
        GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE); GlDisable(GL_BLEND);
        GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL);
        GlDepthMask(GL_FALSE_VALUE); GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        GlBindVertexArray(VertexArray);
        if (LightListsActive) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, ForwardPlusLightBinding, ForwardPlusSceneBuffer);
        GlViewport(0, 0, FogDimensions.AtlasWidth(), FogDimensions.AtlasHeight());
        for (std::size_t index = 0; index < passes.size(); ++index)
        {
            UnbindSampledTextures();
            GlActiveTexture(GL_TEXTURE0+6); GlBindTexture(GL_TEXTURE_2D,density);
            const bool inject = index == 0, final = index + 1 == passes.size();
            if (!inject) { inputs = {}; inputs[0] = VolumeLightTextures[(index - 1) % VolumeLightTextures.size()]; }
            GlBindFramebuffer(GL_FRAMEBUFFER, final ? FogFramebuffer : VolumeLightFramebuffers[index % VolumeLightFramebuffers.size()]);
            const GLuint program = inject && customInjection ? programs[1] : programs[0];
            GlUseProgram(program);
            BindRayShadowImage(program);
            GlUniform4ui(VolumeLightLocation(program, "VolumeControl"), lights.Control.X, lights.Control.Y, lights.Control.Z, passes[index].PrefixStride);
            for (std::size_t unit = 0; unit < inputs.size(); ++unit)
            {
                GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit)); GlBindTexture(GL_TEXTURE_2D, inputs[unit]);
            }
            constexpr GLsizei FullscreenVertices = 3;
            GlDrawArrays(GL_TRIANGLES, 0, FullscreenVertices);
            if (!CheckGL("Draw volume-light injection or prefix scan")) { restore(); return false; }
        }
        restore();
        (Techniques::UsesVolumetricClouds(FrameFog) ? PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(FrameFog) ? PendingFrame.Statistics.ParticipatingMediaPassCount : PendingFrame.Statistics.VolumetricLightingPassCount) += static_cast<std::uint32_t>(passes.size());
        return true;
    }

    void DestroyLightShaftTargets()
    {
        if (LightShaftFramebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &LightShaftFramebuffer);
        if (LightShaftTexture && GlDeleteTextures) GlDeleteTextures(1, &LightShaftTexture);
        LightShaftFramebuffer = LightShaftTexture = 0;
        LightShaftResolved = false;
    }

    bool EnsureLightShaftResources()
    {
        if (!LightShaftTargetSupported || !HdrActive || !HdrTextures[0] || !DepthTexture)
            return Fail("OpenGL ES light shafts require sampled depth and floating color render/blit targets");
        if (!LightShaftProgram) LightShaftProgram = CompileStageProgram("LightShafts.glsl");
        if (!LightShaftProgram) return false;
        if (LightShaftFramebuffer) return true;
        UnbindSampledTextures();
        GLuint texture = 0, framebuffer = 0;
        GlActiveTexture(GL_TEXTURE0);
        GlGenTextures(1, &texture); GlBindTexture(GL_TEXTURE_2D, texture);
        ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, CreateInfo.Width, CreateInfo.Height, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
        GlGenFramebuffers(1, &framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &attachment);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        const bool clean = CheckGL("Allocate light-shaft occlusion target");
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlBindTexture(GL_TEXTURE_2D, 0);
        if (!complete || !clean)
        {
            if (framebuffer) GlDeleteFramebuffers(1, &framebuffer);
            if (texture) GlDeleteTextures(1, &texture);
            return Fail("OpenGL ES light-shaft occlusion framebuffer is incomplete");
        }
        DestroyLightShaftTargets();
        LightShaftFramebuffer = framebuffer; LightShaftTexture = texture;
        return true;
    }

    void DestroyFogTargets()
    {
        DestroyVolumeLightTargets();
        if (FogFramebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &FogFramebuffer);
        if (FogTexture && GlDeleteTextures) GlDeleteTextures(1, &FogTexture);
        FogFramebuffer = FogTexture = 0; FogDimensions = {};
        FogResolved = false;
    }

    bool EnsureFogResources()
    {
        if (!FogTargetSupported || !HdrActive || !HdrTextures[0] || !DepthTexture)
            return Fail("OpenGL ES fog requires sampled depth, RGBA16F render/blit targets and sufficient texture bindings");
        const bool media = Techniques::UsesBoundedMedium(FrameFog);
        const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
        auto& program = cloud ? CloudFogProgram : media ? MediaFogProgram : FogProgram;
        if (!program) program = CompileStageProgram(cloud ? "VolumetricClouds.glsl" : media ? "ParticipatingMedia.glsl" : "VolumetricFog.glsl",
            cloud ? "#define EVANS_CLOUD_COMPOSITE 1\n" : media ? "#define EVANS_MEDIA_COMPOSITE 1\n" : "");
        if (!program) return false;
        const auto grid = BuildFogGrid(CreateInfo.Width, CreateInfo.Height, FrameFog, MaximumTextureSize);
        if (grid.Width == 0) return Fail("Invalid OpenGL ES fog atlas dimensions");
        if (FogFramebuffer && FogDimensions.Width == grid.Width && FogDimensions.Height == grid.Height &&
            FogDimensions.Slices == grid.Slices && FogDimensions.Planes == grid.Planes) return true;
        UnbindSampledTextures();
        GLuint texture = 0, framebuffer = 0;
        GlGenTextures(1, &texture); GlBindTexture(GL_TEXTURE_2D, texture);
        ConfigureTexture();
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, grid.AtlasWidth(), grid.AtlasHeight(), 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
        GlGenFramebuffers(1, &framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        GlDrawBuffers(1, &attachment);
        const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        const bool clean = CheckGL("Allocate cumulative fog atlas");
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlBindTexture(GL_TEXTURE_2D, 0);
        if (!complete || !clean)
        {
            GlDeleteFramebuffers(1, &framebuffer); GlDeleteTextures(1, &texture);
            return clean ? Fail("OpenGL ES fog atlas framebuffer is incomplete") : false;
        }
        DestroyFogTargets();
        FogTexture = texture; FogFramebuffer = framebuffer; FogDimensions = grid;
        return true;
    }

    void UploadFogConstants(GLuint program, const VolumetricFogConstants& constants)
    {
        auto found = FogLocations.find(program);
        if (found == FogLocations.end())
        {
            FogUniformLocations locations;
            locations.Camera = GlGetUniformLocation(program, "FogInverseCamera");
            locations.Medium = GlGetUniformLocation(program, "FogMedium");
            locations.Radiance = GlGetUniformLocation(program, "FogRadiance");
            locations.Grid = GlGetUniformLocation(program, "FogViewGrid");
            locations.Control = GlGetUniformLocation(program, "FogControl");
            locations.Atlas = GlGetUniformLocation(program, "FogTexture");
            locations.WaterBounds=GlGetUniformLocation(program,"WaterBounds");
            locations.WaterSun=GlGetUniformLocation(program,"WaterSun");
            locations.WaterIrradiance=GlGetUniformLocation(program,"WaterIrradiance");
            locations.WaterWaveA=GlGetUniformLocation(program,"WaterWaveA");
            locations.WaterWaveB=GlGetUniformLocation(program,"WaterWaveB");
            found = FogLocations.emplace(program, locations).first;
        }
        const auto& locations = found->second;
        GlUniform4ui(locations.Control, constants.Options.X, constants.Options.Y, constants.Options.Z, constants.Options.W);
        GlUniform1i(locations.Atlas, static_cast<GLint>(FogTextureUnit));
        if (constants.Options.W != 0)
        {
            const auto matrix = ToColumnMajor(constants.InverseViewProjection);
            GlUniformMatrix4fv(locations.Camera, 1, GL_FALSE_VALUE, matrix.data());
            UploadFloat4(locations.Medium, constants.Medium);
            UploadFloat4(locations.Radiance, constants.SourceRadiance);
            UploadFloat4(locations.Grid, constants.ViewportGrid);
            if (locations.WaterBounds>=0)
            {
                UploadFloat4(locations.WaterBounds,constants.WaterBounds);
                UploadFloat4(locations.WaterSun,constants.WaterSun);
                UploadFloat4(locations.WaterIrradiance,constants.WaterIrradiance);
                UploadFloat4(locations.WaterWaveA,constants.WaterWaveA);
                UploadFloat4(locations.WaterWaveB,constants.WaterWaveB);
            }
        }
    }

    void DestroyTemporalTargets()
    {
        for (auto& framebuffer : TemporalFramebuffers)
        {
            if (framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1, &framebuffer);
            framebuffer = 0;
        }
        for (auto* textures : {&TemporalColors, &TemporalDepths})
            for (auto& texture : *textures)
            {
                if (texture && GlDeleteTextures) GlDeleteTextures(1, &texture);
                texture = 0;
            }
        TemporalHistoryValid = false;
        TemporalHistoryReadIndex = 0;
    }

    #include "Renderer/OpenGLES/OpenGLESLightmaps.inl"
    #include "Renderer/OpenGLES/OpenGLESRefraction.inl"
    #include "Renderer/OpenGLES/OpenGLESDepthPeeling.inl"

    void DestroyProbeTargets()
    {
        for (auto& framebuffer : ProbeFramebuffers) { if (framebuffer) GlDeleteFramebuffers(1,&framebuffer); framebuffer=0; }
        for (auto& texture : ProbeTextures) { if (texture) GlDeleteTextures(1,&texture); texture=0; }
        ProbeCount=0; ProbeWidth=ProbeHeight=0; UploadedProbeSamples.reset(); Probes.Invalidate();
    }
    bool EnsureProbeResources()
    {
        if (!HdrActive || !FrameProbeSamples || FrameProbeSamples->Texels.empty()) return Fail("OpenGL ES probes require HDR and baked ray samples");
        if (!ProbePrograms[0]) ProbePrograms[0]=CompileStageProgram("BakedGI.glsl");
        if (!ProbePrograms[1]) ProbePrograms[1]=CompileStageProgram("LightProbes.glsl");
        if (!ProbePrograms[0] || !ProbePrograms[1]) return false;
        const auto count=FrameProbeSamples->ProbeCount();
        if (ProbeCount!=count || ProbeWidth!=CreateInfo.Width || ProbeHeight!=CreateInfo.Height) DestroyProbeTargets();
        UnbindSampledTextures(); GlActiveTexture(GL_TEXTURE0);
        if (!ProbeTextures[ProbeRaw]) GlGenTextures(1,&ProbeTextures[ProbeRaw]);
        if (UploadedProbeSamples!=FrameProbeSamples)
        {
            GlBindTexture(GL_TEXTURE_2D,ProbeTextures[ProbeRaw]); ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,FrameProbeSamples->Width(),FrameProbeSamples->Height(),0,
                GL_RGBA,GL_FLOAT,FrameProbeSamples->Texels.data());
            GlBindTexture(GL_TEXTURE_2D,0);
            if (!CheckGL("Upload baked probe ray samples")) return false;
            UploadedProbeSamples=FrameProbeSamples; Probes.Invalidate();
        }
        const int widths[] {0,int(ProbeSHCoefficientCount),int(ProbeMomentTileSize*ProbeMomentAtlasColumns),CreateInfo.Width};
        const int heights[] {0,int(count),int(((count+ProbeMomentAtlasColumns-1)/ProbeMomentAtlasColumns)*ProbeMomentTileSize),CreateInfo.Height};
        for (std::size_t index=ProbeSH;index<ProbeTextures.size();++index)
        {
            if (ProbeFramebuffers[index]) continue;
            GlGenTextures(1,&ProbeTextures[index]); GlBindTexture(GL_TEXTURE_2D,ProbeTextures[index]); ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,widths[index],heights[index],0,GL_RGBA,GL_HALF_FLOAT,nullptr);
            GlBindTexture(GL_TEXTURE_2D,0);
            GlGenFramebuffers(1,&ProbeFramebuffers[index]); GlBindFramebuffer(GL_FRAMEBUFFER,ProbeFramebuffers[index]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,ProbeTextures[index],0);
            const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create light probe targets"))
            { GlBindFramebuffer(GL_FRAMEBUFFER,0); DestroyProbeTargets(); return Fail("OpenGL ES light probe framebuffer is incomplete"); }
        }
        GlBindFramebuffer(GL_FRAMEBUFFER,0); ProbeCount=count; ProbeWidth=CreateInfo.Width; ProbeHeight=CreateInfo.Height; return true;
    }
    bool ResolveProbes(const GlobalIlluminationDesc& desc)
    {
        LightProbeConstants constants;
        if (!FrameProbeSamples || !BuildLightProbeConstants(desc,*FrameProbeSamples,FrameGI,FrameReflectionShading,
            CreateInfo.Width,CreateInfo.Height,constants)) return Fail("OpenGL ES probes require finite geometry and sample inputs");
        auto restore=[&]() { UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0); GlDepthMask(GL_TRUE_VALUE); };
        UnbindSampledTextures(); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE); GlDisable(GL_BLEND);
        GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDepthMask(GL_FALSE_VALUE);
        GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlBindVertexArray(VertexArray);
        auto draw=[&](ProbePass phase,std::size_t target,int width,int height) {
            const bool gather=phase==ProbePass::Gather; const auto program=ProbePrograms[gather?1:0];
            GlUseProgram(program); GlBindFramebuffer(GL_FRAMEBUFFER,ProbeFramebuffers[target]); GlViewport(0,0,width,height);
            const auto inverse=ToColumnMajor(constants.InverseViewProjection);
            GlUniformMatrix4fv(GlGetUniformLocation(program,"InverseViewProjection"),1,GL_FALSE_VALUE,inverse.data());
            const char* names[] {"GridMinimum","GridMaximum","Viewport","Appearance","Visibility"};
            const Float4 values[] {constants.GridMinimum,constants.GridMaximum,constants.Viewport,constants.Appearance,constants.Visibility};
            for (std::size_t index=0;index<std::size(values);++index) UploadFloat4(GlGetUniformLocation(program,names[index]),values[index]);
            GlUniform4ui(GlGetUniformLocation(program,"Dimensions"),constants.Dimensions.X,constants.Dimensions.Y,constants.Dimensions.Z,constants.Dimensions.W);
            GlUniform4ui(GlGetUniformLocation(program,"Options"),static_cast<unsigned>(phase),constants.Options.Y,constants.Options.Z,constants.Options.W);
            const GLuint inputs[] {ProbeTextures[ProbeRaw],gather?ProbeTextures[ProbeSH]:0,gather?ProbeTextures[ProbeMoments]:0,
                gather?HdrTextures[0]:0,gather?DepthTexture:0,gather?GBufferTextures[1]:0,gather?GBufferTextures[0]:0,gather?GBufferTextures[3]:0};
            const char* samplers[] {"RaySamples","Coefficients","DistanceMoments","SceneColor","SceneDepth","NormalRoughness","DiffuseOpacity","AmbientMetallic"};
            for (std::size_t slot=0;slot<std::size(inputs);++slot)
            {
                GlActiveTexture(GL_TEXTURE0+static_cast<GLenum>(slot)); GlBindTexture(GL_TEXTURE_2D,inputs[slot]);
                GlUniform1i(GlGetUniformLocation(program,samplers[slot]),static_cast<GLint>(slot));
            }
            GlDrawArrays(GL_TRIANGLES,0,3);
            const bool result=CheckGL("Project/gather light probes"); UnbindSampledTextures(); return result;
        };
        const bool projection=Probes.NeedsProjection(FrameProbeSamples,constants.Visibility.Y);
        if (projection)
        {
            Probes.Invalidate();
            if (!draw(ProbePass::ProjectIrradiance,ProbeSH,ProbeSHCoefficientCount,ProbeCount) ||
                !draw(ProbePass::ProjectDistance,ProbeMoments,ProbeMomentAtlasColumns*ProbeMomentTileSize,
                    ((ProbeCount+ProbeMomentAtlasColumns-1)/ProbeMomentAtlasColumns)*ProbeMomentTileSize)) { restore(); return false; }
            Probes.Projected(FrameProbeSamples,constants.Visibility.Y);
        }
        if (!draw(ProbePass::Gather,ProbeComposite,CreateInfo.Width,CreateInfo.Height)) { restore(); return false; }
        GlReadBuffer(GL_COLOR_ATTACHMENT0); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]);
        GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,CreateInfo.Width,CreateInfo.Height);
        const bool result=CheckGL("Copy probe HDR composite"); restore(); if (!result) return false;
        Probes.Resolved=GI.Resolved=true; PendingFrame.Statistics.GlobalIlluminationPassCount=projection?3:1; return true;
    }

    void DestroyGITargets()
    {
        for (auto& framebuffer : GIFramebuffers)
        { if (framebuffer) GlDeleteFramebuffers(1, &framebuffer); framebuffer = 0; }
        for (auto& texture : GITextures)
        { if (texture) GlDeleteTextures(1, &texture); texture = 0; }
        GI.Invalidate();
    }
    bool EnsureGIResources()
    {
        if (!HdrActive || !FloatDepthTargetSupported || MaximumDrawBuffers < 3 || MaximumColorAttachments < 3)
            return Fail("OpenGL ES SSGI requires HDR and three floating history attachments");
        if (!GIProgram) GIProgram = CompileStageProgram("SSGI.glsl");
        if (!GITemporalProgram) GITemporalProgram = CompileStageProgram("SSGI.glsl", "#define EVANS_SSGI_TEMPORAL 1\n");
        if (!GIProgram || !GITemporalProgram) return false;
        if (GIFramebuffers[0]) return true;
        UnbindSampledTextures();
        for (std::size_t i = 0; i < GITextures.size(); ++i)
        {
            const bool depth = i >= GIDepthFirst && i < GINormalFirst;
            GlGenTextures(1, &GITextures[i]); GlBindTexture(GL_TEXTURE_2D, GITextures[i]);
            ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, depth ? GL_R32F : GL_RGBA16F, CreateInfo.Width, CreateInfo.Height,
                0, depth ? GL_RED : GL_RGBA, depth ? GL_FLOAT : GL_HALF_FLOAT, nullptr);
        }
        GlBindTexture(GL_TEXTURE_2D, 0);
        const GLenum attachments[] {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0 + 1, GL_COLOR_ATTACHMENT0 + 2};
        for (std::size_t i = 0; i < GIFramebuffers.size(); ++i)
        {
            GlGenFramebuffers(1, &GIFramebuffers[i]); GlBindFramebuffer(GL_FRAMEBUFFER, GIFramebuffers[i]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[0], GL_TEXTURE_2D, GITextures[i], 0);
            const bool history = i >= GIColorFirst;
            if (history)
            {
                GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[1], GL_TEXTURE_2D, GITextures[GIDepthFirst + i - GIColorFirst], 0);
                GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[2], GL_TEXTURE_2D, GITextures[GINormalFirst + i - GIColorFirst], 0);
            }
            GlDrawBuffers(history ? 3 : 1, attachments);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create SSGI targets"))
            {
                GlBindFramebuffer(GL_FRAMEBUFFER, 0); DestroyGITargets();
                return Fail("OpenGL ES SSGI framebuffer is incomplete");
            }
        }
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GI.Invalidate(); return true;
    }

    bool EnsureTemporalResources()
    {
        if (!HdrActive || !HdrTextures[0] || !FloatDepthTargetSupported || MaximumDrawBuffers < 2 || MaximumColorAttachments < 2)
            return Fail("OpenGL ES TAA requires HDR and two renderable color/depth history attachments");
        if (!TemporalProgram) TemporalProgram = CompileStageProgram("TAA.glsl");
        if (!TemporalProgram) return false;
        if (TemporalFramebuffers[0]) return true;
        UnbindSampledTextures();
        const GLenum attachments[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0 + 1};
        for (std::size_t index = 0; index < TemporalFramebuffers.size(); ++index)
        {
            GlGenTextures(1, &TemporalColors[index]);
            GlBindTexture(GL_TEXTURE_2D, TemporalColors[index]);
            ConfigureDepthTexture();
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, CreateInfo.Width, CreateInfo.Height, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            GlGenTextures(1, &TemporalDepths[index]);
            GlBindTexture(GL_TEXTURE_2D, TemporalDepths[index]);
            ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, CreateInfo.Width, CreateInfo.Height, 0, GL_RED, GL_FLOAT, nullptr);
            GlBindTexture(GL_TEXTURE_2D, 0);
            GlGenFramebuffers(1, &TemporalFramebuffers[index]);
            GlBindFramebuffer(GL_FRAMEBUFFER, TemporalFramebuffers[index]);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[0], GL_TEXTURE_2D, TemporalColors[index], 0);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[1], GL_TEXTURE_2D, TemporalDepths[index], 0);
            GlDrawBuffers(static_cast<GLsizei>(std::size(attachments)), attachments);
            if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create temporal history MRT"))
            {
                GlBindFramebuffer(GL_FRAMEBUFFER, 0);
                DestroyTemporalTargets();
                return Fail("OpenGL ES temporal history framebuffer is incomplete");
            }
        }
        GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        return true;
    }

    void UploadMotionConstants(GLuint program, const MotionBlurConstants& constants)
    {
        const auto matrix = [&](const char* name, const Matrix4& value)
        {
            const auto columnMajor = ToColumnMajor(value);
            GlUniformMatrix4fv(GlGetUniformLocation(program, name), 1, GL_FALSE_VALUE, columnMajor.data());
        };
        matrix("ObjectMatrix", constants.World);
        matrix("CameraMatrix", constants.ViewProjection);
        matrix("PreviousClipMatrix", constants.PreviousWorldViewProjection);
        matrix("InverseCamera", constants.InverseViewProjection);
        UploadFloat4(GlGetUniformLocation(program, "OpticalDepthPlane"), constants.DepthPlane);
        UploadFloat4(GlGetUniformLocation(program, "ExtentExposure"), constants.ViewportExposure);
        UploadFloat4(GlGetUniformLocation(program, "AlphaParameters"), constants.Material);
        GlUniform4ui(GlGetUniformLocation(program, "Options"), constants.Options.X, constants.Options.Y,
            constants.Options.Z, constants.Options.W);
    }

    bool ReadMotionVectors()
    {
        // EXT_color_buffer_float / half_float permit RGBA/FLOAT reads without
        // clamping signed pixel displacement to the normalized color interval.
        // https://registry.khronos.org/OpenGL/extensions/EXT/EXT_color_buffer_float.txt
        std::vector<Float4> bottomUp(static_cast<std::size_t>(CreateInfo.Width) * CreateInfo.Height);
        UnbindSampledTextures();
        GlBindFramebuffer(GL_FRAMEBUFFER, MotionFramebuffers[MotionVelocity]);
        GlReadBuffer(GL_COLOR_ATTACHMENT0);
        GlPixelStorei(GL_PACK_ALIGNMENT, 1);
        GlReadPixels(0, 0, CreateInfo.Width, CreateInfo.Height, GL_RGBA, GL_FLOAT, bottomUp.data());
        if (!CheckGL("Read floating-point motion vectors")) return false;
        PendingFrame.MotionVectors.resize(bottomUp.size());
        for (int y = 0; y < CreateInfo.Height; ++y)
            std::copy_n(bottomUp.data() + static_cast<std::size_t>(CreateInfo.Height - 1 - y) * CreateInfo.Width,
                CreateInfo.Width, PendingFrame.MotionVectors.data() + static_cast<std::size_t>(y) * CreateInfo.Width);
        for (auto& motion:PendingFrame.MotionVectors) motion=DecodeMotionVector(motion);
        return true;
    }

    bool ApplyHdr(const Techniques::HdrSettings& settings)
    {
        const auto plan = Techniques::BuildHdrPlan(settings);
        if (plan.empty()) return true;
        GLuint scene = HdrTextures[0];
        GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE); GlDisable(GL_BLEND);
        GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL);
        GlDepthMask(GL_FALSE_VALUE);
        GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        GlBindVertexArray(VertexArray);
        for (const auto pass : plan)
        {
            const auto programIndex = HdrShaderIndex(pass);
            if (!HdrPrograms[programIndex])
            {
                const std::string filename = std::string(HdrShaderFilename(pass)) + ".glsl";
                HdrPrograms[programIndex] = CompileStageProgram(filename.c_str());
            }
            const GLuint program = HdrPrograms[programIndex];
            if (!program) return false;
            GLuint source = scene;
            GLuint destinationTexture = scene == HdrTextures[0] ? HdrTextures[1] : HdrTextures[0];
            GLuint destinationFramebuffer = scene == HdrTextures[0] ? HdrFramebuffers[1] : HdrFramebuffers[0];
            int sourceWidth = CreateInfo.Width, sourceHeight = CreateInfo.Height;
            int width = CreateInfo.Width, height = CreateInfo.Height;
            using Pass = Techniques::HdrPassKind;
            const auto resources = DescribeHdrPass(pass);
            constexpr std::size_t FirstHalfTarget = 2;
            if (resources.DestinationHalf >= 0)
            {
                width = (width + 1) / 2; height = (height + 1) / 2;
                const std::size_t destination = FirstHalfTarget + resources.DestinationHalf;
                destinationTexture = HdrTextures[destination]; destinationFramebuffer = HdrFramebuffers[destination];
                if (resources.SourceHalf >= 0)
                {
                    source = HdrTextures[FirstHalfTarget + resources.SourceHalf];
                    sourceWidth = width; sourceHeight = height;
                }
            }
            else if (pass == Pass::ToneMapping)
            {
                destinationTexture = ColorTexture; destinationFramebuffer = MainFramebuffer;
            }
            const HdrConstants constants = BuildHdrConstants(settings, sourceWidth, sourceHeight, pass, CreateInfo.Width, CreateInfo.Height);
            GlBindFramebuffer(GL_FRAMEBUFFER, destinationFramebuffer);
            GlViewport(0, 0, width, height);
            GlUseProgram(program);
            UploadFloat4(GlGetUniformLocation(program, "InverseSourceExtentBloom"), constants.InverseSourceExtentBloom);
            UploadFloat4(GlGetUniformLocation(program, "GradeExposure"), constants.GradeExposure);
            UploadFloat4(GlGetUniformLocation(program, "FlareThresholdIntensitySpacingRadius"), constants.FlareThresholdIntensitySpacingRadius);
            UploadFloat4(GlGetUniformLocation(program, "FlareAspect"), constants.FlareAspect);
            GlUniform4ui(GlGetUniformLocation(program, "PassToneMapping"), constants.PassToneMapping.X, constants.PassToneMapping.Y, 0, 0);
            GlUniform1i(GlGetUniformLocation(program, "SceneColor"), 0);
            GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D, source);
            GlUniform1i(GlGetUniformLocation(program, "BloomColor"), 1);
            GlActiveTexture(GL_TEXTURE0 + 1); GlBindTexture(GL_TEXTURE_2D,
                resources.CompositeHalf >= 0 ? HdrTextures[FirstHalfTarget + resources.CompositeHalf] : source);
            GlDrawArrays(GL_TRIANGLES, 0, 3);
            if (!CheckGL("Draw HDR pass")) return false;
            if (resources.UpdatesScene) scene = destinationTexture;
            PendingFrame.HDRPasses.push_back(pass);
        }
        PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
        PendingFrame.Statistics.PassCount += static_cast<std::uint32_t>(plan.size());
        return true;
    }

    bool ApplyPostProcess(const Techniques::PostProcessSettings& settings, GLuint& resultFramebuffer)
    {
        resultFramebuffer = MainFramebuffer;
        const auto plan = Techniques::BuildPostProcessPlan(settings);
        if (plan.empty()) return true;
        if (!EnsurePostTargets()) return false;
        const PostProcessConstants constants = BuildPostProcessConstants(settings, CreateInfo.Width, CreateInfo.Height);
        GLuint source = ColorTexture;
        std::size_t destination = 0;
        GlViewport(0, 0, CreateInfo.Width, CreateInfo.Height);
        GlDisable(GL_DEPTH_TEST);
        GlDisable(GL_CULL_FACE);
        GlDisable(GL_BLEND);
        GlDisable(GL_DITHER);
        GlDisable(GL_POLYGON_OFFSET_FILL);
        GlDepthMask(GL_FALSE_VALUE);
        GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        GlBindVertexArray(VertexArray);
        for (const auto pass : plan)
        {
            const GLuint program = GetPostProgram(pass);
            if (program == 0) return false;
            GlBindFramebuffer(GL_FRAMEBUFFER, PostFramebuffers[destination]);
            GlUseProgram(program);
            UploadFloat4(GlGetUniformLocation(program, "InverseExtentStrengths"), constants.InverseExtentStrengths);
            GlUniform4ui(GlGetUniformLocation(program, "GrainSeedReserved"), constants.GrainSeedReserved.X, 0, 0, 0);
            UploadFloat4(GlGetUniformLocation(program, "DistortionStrengthRadiusWeights"), constants.DistortionStrengthRadiusWeights);
            UploadFloat4(GlGetUniformLocation(program, "NprEdge"), constants.NprEdge);
            UploadFloat4(GlGetUniformLocation(program, "NprStyle"), constants.NprStyle);
            UploadFloat4(GlGetUniformLocation(program, "NprHatch"), constants.NprHatch);
            UploadFloat4(GlGetUniformLocation(program, "NprTone"), constants.NprTone);
            GlUniform1i(GlGetUniformLocation(program, "SceneColor"), 0);
            GlActiveTexture(GL_TEXTURE0);
            GlBindTexture(GL_TEXTURE_2D, source);
            GlDrawArrays(GL_TRIANGLES, 0, 3);
            if (!CheckGL("Draw fullscreen post-process pass")) return false;
            source = PostTextures[destination];
            resultFramebuffer = PostFramebuffers[destination];
            destination ^= 1U;
            PendingFrame.PostProcessingPasses.push_back(pass);
        }
        PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
        PendingFrame.Statistics.PassCount += static_cast<std::uint32_t>(plan.size());
        return true;
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

    GLint MaximumAnisotropy = 1;
    bool AnisotropicFiltering = false;

    void ConfigureMaterialFiltering()
    {
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
            AnisotropicFiltering ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
        if (MaximumAnisotropy > 1)
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                AnisotropicFiltering ? std::min<GLint>(MaximumAnisotropy, MaterialAnisotropy) : 1);
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
            VisitTextureMips(desc, [&](int level, int width, int height, std::size_t offset) {
                GlTexImage2D(GL_TEXTURE_2D, level, internalFormat, width, height, 0, GL_RGBA,
                    GL_UNSIGNED_BYTE, initialData ? static_cast<const std::uint8_t*>(initialData) + offset : nullptr);
            });
        }
        if (desc.MipLevels > 1)
        {
            GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, desc.MipLevels - 1);
            ConfigureMaterialFiltering();
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

    std::shared_ptr<void> GpuContextOwner,GpuDisplayOwner;
    static std::shared_ptr<void> RetainDisplay(EGLDisplay display,PFNEGLTERMINATE terminate)
    {
        static std::mutex mutex;
        static std::unordered_map<EGLDisplay,std::weak_ptr<void>> displays;
        std::lock_guard<std::mutex> lock(mutex);
        auto& existing=displays[display];if(auto lease=existing.lock())return lease;
        auto lease=std::shared_ptr<void>(display,[terminate](void* handle){terminate(handle);});existing=lease;return lease;
    }
    struct GpuSurface { GLuint Texture=0; int Width=0,Height=0; std::shared_ptr<GpuImage> Published; };
    std::vector<GpuSurface> GpuSurfaces;
    void PublishGpuContext()
    {
        if(!GpuContextOwner)
        {
            const auto display=Display,context=Context,surface=Surface;
            const auto destroyContext=EglDestroyContext,destroySurface=EglDestroySurface;
            const auto egl=EGLModule,gles=GLESModule;auto displayOwner=GpuDisplayOwner;
            GpuContextOwner=std::shared_ptr<void>(context,[=](void*) mutable {
                destroyContext(display,context);destroySurface(display,surface);displayOwner.reset();
                FreeLibrary(egl);FreeLibrary(gles);
            });
        }
        auto context=std::make_shared<GpuImage>();context->Backend=RHIBackend::OpenGLES;
        context->NativeDisplay=Display;context->NativeContext=Context;context->ContextOwner=GpuContextOwner;
        PendingFrame.GpuContext=std::move(context);
    }
    bool PublishGpuColor(GLuint framebuffer)
    {
        GpuSurface* target=nullptr;
        for(auto& item:GpuSurfaces)if(item.Published.use_count()==1 && item.Width==CreateInfo.Width && item.Height==CreateInfo.Height){target=&item;break;}
        if(!target)
        {
            GpuSurfaces.erase(std::remove_if(GpuSurfaces.begin(),GpuSurfaces.end(),[this](auto& item){
                if(item.Published.use_count()!=1)return false;GlDeleteTextures(1,&item.Texture);return true;
            }),GpuSurfaces.end());
            GpuSurface item;item.Width=CreateInfo.Width;item.Height=CreateInfo.Height;
            GlGenTextures(1,&item.Texture);GlBindTexture(GL_TEXTURE_2D,item.Texture);
            GlTexImage2D(GL_TEXTURE_2D,0,GL_SRGB8_ALPHA8,item.Width,item.Height,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
            GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
            GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            item.Published=std::make_shared<GpuImage>();item.Published->Backend=RHIBackend::OpenGLES;
            item.Published->NativeTexture=reinterpret_cast<void*>(std::uintptr_t(item.Texture));
            item.Published->NativeDisplay=Display;item.Published->NativeContext=Context;item.Published->ContextOwner=GpuContextOwner;
            GpuSurfaces.push_back(item);target=&GpuSurfaces.back();
        }
        GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);GlReadBuffer(GL_COLOR_ATTACHMENT0);
        GlBindTexture(GL_TEXTURE_2D,target->Texture);
        GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,target->Width,target->Height);
        // Renderer and presenter serialize the same EGL context. Its command
        // stream orders the snapshot copy and subsequent sampling without a CPU wait.
        if(!CheckGL("Publish shared GL viewport"))return false;
        PendingFrame.GpuColor=target->Published;return true;
    }

    bool ReadColor(GLuint framebuffer, std::vector<std::uint8_t>& output)
    {
        const std::size_t rowBytes = static_cast<std::size_t>(CreateInfo.Width) * 4;
        std::vector<std::uint8_t> bottomUp(rowBytes * static_cast<std::size_t>(CreateInfo.Height));
        GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
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
        GlDisable(GL_CULL_FACE);
        GlDisable(GL_BLEND);
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
    PFNGLUNIFORM4UI GlUniform4ui = nullptr;
    PFNGLUNIFORM1I GlUniform1i = nullptr;
    PFNGLGETUNIFORMBLOCKINDEX GlGetUniformBlockIndex = nullptr;
    PFNGLUNIFORMBLOCKBINDING GlUniformBlockBinding = nullptr;
    PFNGLBINDBUFFERBASE GlBindBufferBase = nullptr;
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
    PFNGLTEXSUBIMAGE2D GlTexSubImage2D = nullptr;
    PFNGLGENERATEMIPMAP GlGenerateMipmap = nullptr;
    PFNGLCOPYTEXSUBIMAGE2D GlCopyTexSubImage2D = nullptr;
    PFNGLTEXPARAMETERI GlTexParameteri = nullptr;
    PFNGLDELETETEXTURES GlDeleteTextures = nullptr;
    PFNGLACTIVETEXTURE GlActiveTexture = nullptr;
    PFNGLGENFRAMEBUFFERS GlGenFramebuffers = nullptr;
    PFNGLBINDFRAMEBUFFER GlBindFramebuffer = nullptr;
    PFNGLBLITFRAMEBUFFER GlBlitFramebuffer = nullptr;
    PFNGLFRAMEBUFFERTEXTURE2D GlFramebufferTexture2D = nullptr;
    PFNGLCHECKFRAMEBUFFERSTATUS GlCheckFramebufferStatus = nullptr;
    PFNGLDELETEFRAMEBUFFERS GlDeleteFramebuffers = nullptr;
    PFNGLDRAWBUFFERS GlDrawBuffers = nullptr;
    PFNGLREADBUFFER GlReadBuffer = nullptr;
    PFNGLVIEWPORT GlViewport = nullptr;
    PFNGLCLEARCOLOR GlClearColor = nullptr;
    PFNGLCLEARDEPTHF GlClearDepthf = nullptr;
    PFNGLCLEAR GlClear = nullptr;
    PFNGLCLEARBUFFERFV GlClearBufferfv = nullptr;
    PFNGLENABLE GlEnable = nullptr;
    PFNGLDISABLE GlDisable = nullptr;
    PFNGLCULLFACE GlCullFace = nullptr;
    PFNGLFRONTFACE GlFrontFace = nullptr;
    PFNGLDEPTHMASK GlDepthMask = nullptr;
    PFNGLDEPTHFUNC GlDepthFunc = nullptr;
    PFNGLCOLORMASK GlColorMask = nullptr;
    PFNGLBLENDFUNCSEPARATE GlBlendFuncSeparate = nullptr;
    PFNGLBLENDFUNCSEPARATEI GlBlendFuncSeparatei = nullptr;
    PFNGLBLENDEQUATION GlBlendEquation = nullptr;
    PFNGLLINEWIDTH GlLineWidth = nullptr;
    PFNGLPOLYGONOFFSET GlPolygonOffset = nullptr;
    PFNGLDRAWELEMENTS GlDrawElements = nullptr;
    PFNGLDRAWELEMENTSINSTANCED GlDrawElementsInstanced = nullptr;
    PFNGLDRAWARRAYS GlDrawArrays = nullptr;
    PFNGLPIXELSTOREI GlPixelStorei = nullptr;
    PFNGLREADPIXELS GlReadPixels = nullptr;
    PFNGLTEXSTORAGE2DMULTISAMPLE GlTexStorage2DMultisample = nullptr;
    PFNGLGETINTERNALFORMATIV GlGetInternalformativ = nullptr;
    PFNGLGETMULTISAMPLEFV GlGetMultisamplefv = nullptr;
    PFNGLSAMPLEMASKI GlSampleMaski = nullptr;
    RHICreateInfo CreateInfo;
    std::array<GLuint, StandaloneShadingTechniqueCount> TechniquePrograms{};
    OpenGLESContactShadows ContactShadows;
    OpenGLESForwardPlus ForwardPlus;
    OpenGLESForwardPlus TiledDeferred {TiledLightShader::TiledDeferred};
    bool TiledActive = false;
    OpenGLESForwardPlus& TileProducer() { return TiledActive ? TiledDeferred : ForwardPlus; }
    OpenGLESClusteredForward ClusteredForward;
    bool ClusteredActive = false;
    LightListFrameState LightListFrame;
    bool TiledSupported = false, ClusteredSupported = false, ForwardPlusSupported = false, LightListsActive = false, ForwardPlusDepthWritten = false;
    GLuint ForwardPlusSceneBuffer = 0, ForwardPlusEmptyBuffer = 0, ForwardPlusEmptyIndexBuffer = 0;
    std::array<std::array<GLuint, ForwardPlusProgramCount>, ForwardPlusFilterCount> ForwardPlusPrograms {}, ClusteredPrograms {}, TiledPrograms {};
    Techniques::ContactShadowSettings FrameContact;
    bool ContactSupported = false, ContactDepthWritten = false, ContactGenerated = false, ContactReceiverActive = false;
    GLuint ContactDepthProgram = 0, ContactDepthFramebuffer = 0;
    std::array<std::array<GLuint, StandaloneShadingTechniqueCount>, 3> ContactSurfacePrograms {};
    std::array<std::array<GLuint, 2>, 3> ContactScreenPrograms {};
    std::array<GLuint, StandaloneShadingTechniqueCount> WeightedTechniquePrograms {};
    std::array<std::array<GLuint, StandaloneShadingTechniqueCount>, 2> PCSSSurfacePrograms {};
    std::array<GLuint, 3> PCSSScreenPrograms {};
    std::unordered_map<GLuint, std::array<GLint, PCSSMapCount + 2>> PCSSLocations;
    std::array<std::array<GLuint, StandaloneShadingTechniqueCount>, 2> VSMSurfacePrograms {};
    std::array<GLuint, 5> VSMStagePrograms {};
    std::unordered_map<GLuint, std::array<GLint, VSMMapCount + 2>> VSMLocations;
    std::unordered_map<std::uint32_t, VSMMomentMap> VSMMoments;
    bool VSMTargetSupported = false;
    std::array<GLuint, 2> WeightedResolvePrograms {};
    std::uint32_t WeightedSampleCounts = 0;
    bool WeightedActive = false, WeightedPass = false, WeightedPrepared = false, WeightedAccumulated = false, WeightedResolved = false;
    Techniques::TransparencySettings FrameTransparency;
    std::unordered_map<GLuint, GLuint> InstancePrograms;
    GLuint InstanceBuffer = 0;
    GLint MaximumUniformBlockSize = 0;
    GLuint Program = 0;
    GLuint CascadeProgram = 0;
    GLuint ShadowProgram = 0;
    CascadedShadowState FrameCascades {};
    Techniques::ShadowFilterSettings FrameShadowFilter;
    std::array<GLuint, MaximumShadowCascades> CascadeTextures {};
    std::unordered_map<GLuint, CascadeUniformLocations> CascadeLocations;
    GLuint DepthReadProgram = 0;
    GLuint VertexArray = 0;
    GLuint ScratchIndexBuffer = 0;
    GLuint MainFramebuffer = 0;
    GLuint ColorTexture = 0;
    std::array<GLuint, GBufferAttachmentCount> GBufferTextures {};
    GLuint GBufferFramebuffer = 0;
    GLuint LightingFramebuffer = 0;
    GLuint DeferredGeometryProgram = 0;
    std::array<std::array<GLuint, 2>, ForwardPlusFilterCount> TiledLightingPrograms {};
    GLuint DeferredLightingProgram = 0;
    GLuint MultisampleLightingProgram = 0;
    bool MultisampleAPI = false;
    std::uint32_t SupportedSampleCounts = 1, FrameSampleCount = 1;
    bool SamplesOpaqueResolved = false, SamplesVelocityResolved = false;
    bool SamplesColorRestored = false, SamplesColorResolved = false, SamplesTransparentWritten = false;

    GLint MaximumDrawBuffers = 0;
    GLint MaximumColorAttachments = 0;
    bool HalfFloatTargetsSupported = false;
    bool FogTargetSupported = false;
    bool VolumeLightTargetSupported = false;
    GLint MaximumFragmentTextures = 0;
    bool DeferredActive = false;
    std::array<GLuint, 2> AmbientOcclusionTextures{};
    std::array<GLuint, 2> AmbientOcclusionFramebuffers{};
    GLuint SsaoProgram = 0;
    GLuint HbaoProgram = 0;
    GLuint SaoProgram = 0;
    GLuint GtaoProgram = 0;
    GLuint SaoDepthHierarchy = 0;
    GLuint SaoDepthScratch = 0;
    GLuint SaoDepthFramebuffer = 0;
    bool FloatDepthTargetSupported = false;
    Techniques::AmbientOcclusionSettings FrameAmbientOcclusionSettings{};
    bool AmbientOcclusionActive = false;
    bool AmbientOcclusionResolved = false;
    GLuint ReflectionTexture = 0, ReflectionFramebuffer = 0, ReflectionProgram = 0;
    GLuint CubeReflectionProgram = 0;
    GLuint EnvironmentMapProgram = 0, EnvironmentConversionProgram = 0;
    GLuint PlanarProgram = 0, PlanarMipProgram = 0;
    PlanarCaptureFrame FramePlanar {};
    bool PlanarActive = false, PlanarWritten = false, PlanarReady = false;
    EnvironmentCaptureFrame FrameEnvironment {};
    std::uint32_t EnvironmentFacesWritten = 0;
    bool EnvironmentActive = false, EnvironmentReady = false;
    bool ReflectionActive = false, ReflectionResolved = false;
    Techniques::ReflectionSettings FrameReflections{};
    Techniques::ShadingModel FrameReflectionShading = Techniques::ShadingModel::Phong;
    bool ForwardColorWritten = false;
    bool GBufferWritten = false;
    bool DeferredResolved = false;
    Float4 DeferredBackground;
    std::array<GLuint, 4> HdrTextures{};
    std::array<GLuint, 4> HdrFramebuffers{};
    std::array<GLuint, HdrTechniqueCount> HdrPrograms{};
    bool HdrActive = false;
    GLuint DepthOfFieldFramebuffer = 0;
    GLuint DepthOfFieldProgram = 0;
    bool DepthOfFieldActive = false;
    bool DepthOfFieldResolved = false;
    static constexpr std::size_t MotionVelocity = 0, MotionTileMax = 1, MotionNeighborMax = 2;
    std::array<GLuint, 3> MotionTextures{};
    std::array<GLuint, 3> MotionFramebuffers{};
    GLuint MotionCompositeFramebuffer = 0;
    GLuint MotionGeometryProgram = 0, MotionScreenProgram = 0;
    bool MotionBlurActive = false, VelocityCaptured = false, MotionBlurResolved = false;
    bool VelocityActive = false, TemporalActive = false, TemporalResolved = false;
    bool FogActive = false, FogResolved = false;
    bool LightShaftActive = false, LightShaftResolved = false, LightShaftTargetSupported = false;
    GLuint LightShaftTexture = 0, LightShaftFramebuffer = 0, LightShaftProgram = 0;
    Techniques::LightShaftSettings FrameLightShafts;
    GLuint FogTexture = 0, FogFramebuffer = 0, FogProgram = 0, MediaFogProgram = 0, CloudFogProgram = 0;
    std::array<GLuint, 2> VolumeLightTextures {}, VolumeLightFramebuffers {};
    GLuint VolumeLightProgram = 0;
    std::array<GLuint, 2> MediaVolumePrograms {}, CloudVolumePrograms {};
    std::array<GLuint, 3> ForwardPlusVolumePrograms {};
    std::unordered_map<GLuint, std::unordered_map<std::string, GLint>> VolumeLightLocations;
    Techniques::VolumetricLightingSettings FrameVolumeLighting;
    FogGrid FogDimensions;
    VolumetricFogConstants FogConstants;
    Techniques::VolumetricFogSettings FrameFog;
    std::unordered_map<GLuint, FogUniformLocations> FogLocations;
    std::array<GLuint, 2> TemporalColors{}, TemporalDepths{}, TemporalFramebuffers{};
    GLuint TemporalProgram = 0;
    static constexpr std::size_t GIRaw = 0, GIFiltered = 1, GIComposite = 2, GIColorFirst = 3,
        GIDepthFirst = 5, GINormalFirst = 7, GITargetCount = 9;
    std::array<GLuint, GITargetCount> GITextures {};
    std::array<GLuint, GIDepthFirst> GIFramebuffers {};
    GLuint GIProgram = 0, GITemporalProgram = 0;
    static constexpr std::size_t ProbeRaw=0,ProbeSH=1,ProbeMoments=2,ProbeComposite=3;
    std::array<GLuint,4> ProbeTextures {},ProbeFramebuffers {};
    std::array<GLuint,2> ProbePrograms {};
    ProbeProjectionState Probes;
    std::shared_ptr<const BakedProbeSamples> FrameProbeSamples,UploadedProbeSamples;
    std::uint32_t ProbeCount=0;
    int ProbeWidth=0,ProbeHeight=0;
    GlobalIlluminationHistory GI;
    Techniques::GlobalIlluminationSettings FrameGI;
    bool GIActive = false, EarlyTemporalActive = false, ProbeActive = false;
    std::size_t TemporalHistoryReadIndex = 0;
    bool TemporalHistoryValid = false;
    Techniques::PostProcessSettings FramePostProcessSettings{};
    std::array<GLuint, 2> PostTextures{};
    std::array<GLuint, 2> PostFramebuffers{};
    std::array<GLuint, PostProcessTechniqueCount> PostPrograms{};
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
    GLint ShadingParametersLocation = -1;
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
    bool FrameAreaLights = false;
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
    m_Impl->ParallaxShaderSupported=std::strstr(reinterpret_cast<const char*>(version),"OpenGL ES 3.1") ||
        std::strstr(reinterpret_cast<const char*>(version),"OpenGL ES 3.2");
    m_Impl->GlGetIntegerv(GL_MAX_TEXTURE_SIZE, &m_Impl->MaximumTextureSize);
    m_Impl->MaximumAnisotropy = 1;
    m_Impl->AnisotropicFiltering = false;
    const auto* extensions = m_Impl->GlGetString(GL_EXTENSIONS);
    if (extensions && std::string_view(reinterpret_cast<const char*>(extensions)).find("GL_EXT_texture_filter_anisotropic") != std::string_view::npos)
        m_Impl->GlGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &m_Impl->MaximumAnisotropy);
    m_Impl->GlGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &m_Impl->MaximumFragmentTextures);
    m_Impl->GlGetIntegerv(GL_MAX_DRAW_BUFFERS, &m_Impl->MaximumDrawBuffers);
    m_Impl->GlGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &m_Impl->MaximumColorAttachments);
    m_Impl->GlGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &m_Impl->MaximumUniformBlockSize);
    m_Impl->GpuDrivenSupported = m_Impl->ProbeGpuDriven();
    m_Impl->DisplacementSupported=m_Impl->ProbeDisplacement();
    m_Impl->TessellationSupported=m_Impl->ProbeTessellation();
    m_Impl->VirtualGeometrySupported=m_Impl->ProbeVirtualGeometry();
    m_Impl->NaniteGeometrySupported=m_Impl->ProbeNaniteGeometry();
    m_Impl->VirtualShadowSupported=m_Impl->ProbeVirtualShadows();
    m_Impl->HalfFloatTargetsSupported = m_Impl->SupportsHalfFloatTargets();
    m_Impl->ExposureSupported=m_Impl->ProbeTemporalExposure();
    const auto beforeColorProbe = m_Impl->LastError;
    const bool colorBlitSupported = m_Impl->ProbeHalfFloatColorBlit();
    m_Impl->LastError = beforeColorProbe;
    constexpr GLint LightShaftTextureInputs = 3, MinimumFogAtlasWidth = 64;
    m_Impl->LightShaftTargetSupported = colorBlitSupported && m_Impl->MaximumFragmentTextures >= LightShaftTextureInputs;
    m_Impl->FogTargetSupported = colorBlitSupported && m_Impl->MaximumTextureSize >= MinimumFogAtlasWidth &&
        m_Impl->MaximumFragmentTextures > static_cast<GLint>(Impl::FogTextureUnit);
    m_Impl->VolumeLightTargetSupported = m_Impl->ProbeVolumeLightSupport();
    m_Impl->FloatDepthTargetSupported = m_Impl->ProbeFloatDepthSupport();
    m_Impl->VSMTargetSupported = m_Impl->ProbeVSMSupport();
    m_Impl->SupportedSampleCounts = m_Impl->ProbeSampleCounts();
    m_Impl->WeightedSampleCounts = m_Impl->ProbeWeightedSupport();
    m_Impl->PathTargetsSupported = m_Impl->ProbePathTracingSupport();
    std::string contactError;
    m_Impl->ContactSupported = m_Impl->MultisampleAPI && m_Impl->MaximumFragmentTextures > static_cast<GLint>(Impl::ContactTextureUnit) &&
        m_Impl->ContactShadows.Initialize([&](const char* name) -> void*
        {
            auto* address = reinterpret_cast<void*>(GetProcAddress(m_Impl->GLESModule, name));
            return address ? address : m_Impl->EglGetProcAddress(name);
        }, contactError);
    if (m_Impl->MultisampleAPI)
    {
        GLint fragmentBlocks=0, blockBytes=0, combinedOutputs=0,storageBindings=0;
        m_Impl->GlGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&storageBindings);
        m_Impl->GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,&fragmentBlocks);
        m_Impl->GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&blockBytes);
        m_Impl->GlGetIntegerv(GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES,&combinedOutputs);
        m_Impl->RaySceneSupported=m_Impl->GlGetError()==GL_NO_ERROR &&
            fragmentBlocks>=static_cast<GLint>(RaySceneBufferCount) && blockBytes>=sizeof(RaySceneTriangle) &&
            combinedOutputs>=static_cast<GLint>(RaySceneBufferCount+1);
        constexpr auto reflectionBlocks=RaySceneBufferCount+RayRadianceBufferCount;
        m_Impl->RayRadianceSupported=m_Impl->RaySceneSupported &&
            fragmentBlocks>=static_cast<GLint>(reflectionBlocks-1) && storageBindings>=static_cast<GLint>(reflectionBlocks) &&
            combinedOutputs>=static_cast<GLint>(reflectionBlocks) && m_Impl->MaximumFragmentTextures>=8;
        m_Impl->PathTargetsSupported = m_Impl->PathTargetsSupported && combinedOutputs>=static_cast<GLint>(reflectionBlocks+1);
    }
    std::string forwardPlusError;
    m_Impl->ForwardPlusSupported = m_Impl->MultisampleAPI &&
        m_Impl->MaximumFragmentTextures > static_cast<GLint>(Impl::ContactTextureUnit) &&
        m_Impl->ForwardPlus.Initialize([&](const char* name) -> void*
        {
            auto* address = reinterpret_cast<void*>(GetProcAddress(m_Impl->GLESModule, name));
            return address ? address : m_Impl->EglGetProcAddress(name);
        }, forwardPlusError);
    if (m_Impl->ForwardPlusSupported)
    {
        // ES 3.1 compute support alone does not guarantee fragment SSBOs.
        // Khronos OpenGL ES 3.1: Shader Storage Buffer Objects and required limits.
        GLint fragmentBlocks = 0, blockBytes = 0;
        m_Impl->GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &fragmentBlocks);
        m_Impl->GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &blockBytes);
        m_Impl->ForwardPlusSupported = m_Impl->GlGetError() == GL_NO_ERROR &&
            fragmentBlocks >= ForwardPlusStorageCount &&
            blockBytes >= static_cast<GLint>(MaximumForwardPlusLights * sizeof(ShaderLight));
        if (!m_Impl->ForwardPlusSupported) { m_Impl->ForwardPlus.Shutdown(); m_Impl->TiledDeferred.Shutdown(); }
    }
    m_Impl->TiledSupported = m_Impl->ForwardPlusSupported &&
        m_Impl->TiledDeferred.Initialize([&](const char* name) -> void*
        {
            auto* address = reinterpret_cast<void*>(GetProcAddress(m_Impl->GLESModule, name));
            return address ? address : m_Impl->EglGetProcAddress(name);
        }, forwardPlusError);
    m_Impl->ClusteredSupported = m_Impl->ForwardPlusSupported &&
        m_Impl->ClusteredForward.Initialize([&](const char* name) -> void*
        {
            auto* address = reinterpret_cast<void*>(GetProcAddress(m_Impl->GLESModule, name));
            return address ? address : m_Impl->EglGetProcAddress(name);
        }, forwardPlusError);
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
    m_Impl->VisibilitySupported = m_Impl->ProbeVisibilityBuffer();
    const auto voxelProbeError=m_Impl->LastError;
    m_Impl->VoxelSamplingSupported=m_Impl->ProbeVoxelSamplingSupport();
    m_Impl->LumenGatherSupported=false;
    if (m_Impl->GpuDrivenSupported)
    {
        GLint count=0;
        m_Impl->GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,&count);
        m_Impl->LumenGatherSupported=count>=3;
    }
    m_Impl->LastError=voxelProbeError;
    m_Impl->Initialized = true;
    m_Impl->ProbeTSRSupport();
    if(m_Impl->VisibilitySupported && m_Impl->NaniteGeometrySupported)
    {
        GLint vertexBlocks=0,fragmentBlocks=0;
        m_Impl->GlGetIntegerv(Impl::VisibilityMaxVertexBlocks,&vertexBlocks);
        m_Impl->GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,&fragmentBlocks);
        m_Impl->NaniteDeviceAvailable=vertexBlocks>=3 && fragmentBlocks>=2;
        m_Impl->NaniteHardwareAvailable=m_Impl->NaniteDeviceAvailable && m_Impl->SupportsNaniteHardwareBins();
    }
    m_Impl->LastError.clear();
    return true;
}

void OpenGLESRHI::Shutdown()
{
    SetCascadedShadowState({});
    SetEnvironmentCaptureFrame({});
    SetPlanarCaptureFrame({});
    if (!m_Impl || (!m_Impl->EGLModule && !m_Impl->GLESModule)) return;
    if (m_Impl->Display != nullptr && m_Impl->Context != nullptr && m_Impl->EglMakeCurrent)
    {
        m_Impl->EglMakeCurrent(m_Impl->Display, m_Impl->Surface, m_Impl->Surface, m_Impl->Context);
    }
    m_Impl->ContactShadows.Shutdown(); m_Impl->ContactSupported = false;
    m_Impl->ForwardPlus.Shutdown(); m_Impl->TiledDeferred.Shutdown(); m_Impl->ClusteredForward.Shutdown(); m_Impl->LightListFrame.Reset();
    m_Impl->ClusteredSupported = m_Impl->TiledSupported = false;
    m_Impl->ForwardPlusSupported = m_Impl->LightListsActive = m_Impl->ForwardPlusDepthWritten = false;
    if (m_Impl->GlDeleteProgram)
    {
        for (const auto& family : m_Impl->ForwardPlusPrograms)
            for (auto program : family) if (program) m_Impl->GlDeleteProgram(program);
        for (const auto& family : m_Impl->ClusteredPrograms)
            for (auto program : family) if (program) m_Impl->GlDeleteProgram(program);
        for (const auto& family : m_Impl->TiledPrograms)
            for (auto program : family) if (program) m_Impl->GlDeleteProgram(program);
        for (auto program : m_Impl->ForwardPlusVolumePrograms) if (program) m_Impl->GlDeleteProgram(program);
        if (m_Impl->ContactDepthProgram) m_Impl->GlDeleteProgram(m_Impl->ContactDepthProgram);
        for (const auto& family : m_Impl->ContactSurfacePrograms)
            for (auto program : family) if (program) m_Impl->GlDeleteProgram(program);
        for (const auto& family : m_Impl->ContactScreenPrograms)
            for (auto program : family) if (program) m_Impl->GlDeleteProgram(program);
    }
    if (m_Impl->ContactDepthFramebuffer) m_Impl->GlDeleteFramebuffers(1, &m_Impl->ContactDepthFramebuffer);
    m_Impl->ContactDepthProgram = m_Impl->ContactDepthFramebuffer = 0;
    m_Impl->ContactSurfacePrograms = {}; m_Impl->ContactScreenPrograms = {};
    if (m_Impl->GlDeleteProgram)
    {
        for (const auto& filter : m_Impl->TiledLightingPrograms)
            for (GLuint program : filter) if (program) m_Impl->GlDeleteProgram(program);
    }
    m_Impl->TiledLightingPrograms = {};
    m_Impl->ForwardPlusPrograms = {}; m_Impl->ClusteredPrograms = {}; m_Impl->TiledPrograms = {};
    m_Impl->ForwardPlusVolumePrograms = {};
    if (m_Impl->GlDeleteBuffers)
    {
        if (m_Impl->ForwardPlusSceneBuffer) m_Impl->GlDeleteBuffers(1, &m_Impl->ForwardPlusSceneBuffer);
        if (m_Impl->ForwardPlusEmptyBuffer) m_Impl->GlDeleteBuffers(1, &m_Impl->ForwardPlusEmptyBuffer);
        if (m_Impl->ForwardPlusEmptyIndexBuffer) m_Impl->GlDeleteBuffers(1, &m_Impl->ForwardPlusEmptyIndexBuffer);
        for (auto& [handle, resource] : m_Impl->Buffers)
        {
            (void)handle;
            if (resource.Buffer != 0) m_Impl->GlDeleteBuffers(1, &resource.Buffer);
        }
        if (m_Impl->ScratchIndexBuffer != 0)
            m_Impl->GlDeleteBuffers(1, &m_Impl->ScratchIndexBuffer);
        if (m_Impl->InstanceBuffer) m_Impl->GlDeleteBuffers(1, &m_Impl->InstanceBuffer);
    }
    m_Impl->InstanceBuffer = 0;
    m_Impl->DestroyWireframe();
    m_Impl->DestroyGpuDriven(); m_Impl->GpuDrivenSupported = false;
    m_Impl->DestroyVisibilityBuffer();
    m_Impl->ForwardPlusSceneBuffer = m_Impl->ForwardPlusEmptyBuffer = m_Impl->ForwardPlusEmptyIndexBuffer = 0;
    m_Impl->MaximumUniformBlockSize = 0;
    if (m_Impl->GlDeleteProgram)
        for (const auto& entry : m_Impl->InstancePrograms) m_Impl->GlDeleteProgram(entry.second);
    m_Impl->InstancePrograms.clear();
    if (m_Impl->GlDeleteTextures && m_Impl->GlDeleteFramebuffers)
    {
        for (auto& [handle, resource] : m_Impl->Textures)
        {
            (void)handle;
            m_Impl->DestroyTextureResource(resource);
        }
        m_Impl->DestroyColorDepthTarget();
        m_Impl->DestroyEnvironmentTargets(m_Impl->Environment);
        m_Impl->DestroyEnvironmentMapTargets(m_Impl->EnvironmentMap);
        m_Impl->DestroyPlanarTargets(m_Impl->Planar);
        if (m_Impl->DepthReadFramebuffer != 0)
            m_Impl->GlDeleteFramebuffers(1, &m_Impl->DepthReadFramebuffer);
        if (m_Impl->DepthReadTexture != 0)
            m_Impl->GlDeleteTextures(1, &m_Impl->DepthReadTexture);
    }
    if (m_Impl->VertexArray != 0 && m_Impl->GlDeleteVertexArrays)
        m_Impl->GlDeleteVertexArrays(1, &m_Impl->VertexArray);
    if (m_Impl->GlDeleteProgram)
        for (GLuint program : m_Impl->TechniquePrograms)
            if (program != 0) m_Impl->GlDeleteProgram(program);
    m_Impl->TechniquePrograms.fill(0);
    if (m_Impl->GlDeleteProgram)
    {
        for (auto program : m_Impl->WeightedTechniquePrograms) if (program) m_Impl->GlDeleteProgram(program);
        for (auto program : m_Impl->WeightedResolvePrograms) if (program) m_Impl->GlDeleteProgram(program);
    }
    m_Impl->WeightedTechniquePrograms.fill(0); m_Impl->WeightedResolvePrograms.fill(0);
    if (m_Impl->GlDeleteProgram)
    {
        for (const auto& variants : m_Impl->PCSSSurfacePrograms)
            for (auto program : variants) if (program) m_Impl->GlDeleteProgram(program);
        for (auto program : m_Impl->PCSSScreenPrograms) if (program) m_Impl->GlDeleteProgram(program);
        for (const auto& variants : m_Impl->VSMSurfacePrograms)
            for (auto program : variants) if (program) m_Impl->GlDeleteProgram(program);
        for (auto program : m_Impl->VSMStagePrograms) if (program) m_Impl->GlDeleteProgram(program);
    }
    m_Impl->PCSSSurfacePrograms = {}; m_Impl->PCSSScreenPrograms = {}; m_Impl->PCSSLocations.clear();
    m_Impl->VSMSurfacePrograms = {}; m_Impl->VSMStagePrograms = {}; m_Impl->VSMLocations.clear();
    for (auto& entry : m_Impl->VSMMoments) m_Impl->ReleaseMomentMap(entry.second);
    m_Impl->VSMMoments.clear(); m_Impl->VSMTargetSupported = false;
    m_Impl->WeightedSampleCounts = 0;
    if (m_Impl->CascadeProgram && m_Impl->GlDeleteProgram) m_Impl->GlDeleteProgram(m_Impl->CascadeProgram);
    m_Impl->CascadeProgram = 0;
    if (m_Impl->ShadowProgram && m_Impl->GlDeleteProgram) m_Impl->GlDeleteProgram(m_Impl->ShadowProgram);
    m_Impl->ShadowProgram = 0;
    m_Impl->CascadeLocations.clear();
    m_Impl->FrameCascades = {};
    m_Impl->CascadeTextures.fill(0);
    if (m_Impl->GlDeleteProgram)
        for (GLuint program : m_Impl->PostPrograms)
            if (program != 0) m_Impl->GlDeleteProgram(program);
    m_Impl->PostPrograms.fill(0);
    if (m_Impl->GlDeleteProgram)
        for (GLuint program : m_Impl->HdrPrograms)
            if (program) m_Impl->GlDeleteProgram(program);
    m_Impl->HdrPrograms.fill(0);
    if (m_Impl->DepthOfFieldProgram && m_Impl->GlDeleteProgram)
        m_Impl->GlDeleteProgram(m_Impl->DepthOfFieldProgram);
    m_Impl->DepthOfFieldProgram = 0;
    if (m_Impl->ReflectionProgram && m_Impl->GlDeleteProgram)
        m_Impl->GlDeleteProgram(m_Impl->ReflectionProgram);
    m_Impl->ReflectionProgram = 0;
    if (m_Impl->CubeReflectionProgram && m_Impl->GlDeleteProgram)
        m_Impl->GlDeleteProgram(m_Impl->CubeReflectionProgram);
    m_Impl->CubeReflectionProgram = 0;
    if (m_Impl->GlDeleteProgram)
    {
        if (m_Impl->EnvironmentMapProgram) m_Impl->GlDeleteProgram(m_Impl->EnvironmentMapProgram);
        if (m_Impl->EnvironmentConversionProgram) m_Impl->GlDeleteProgram(m_Impl->EnvironmentConversionProgram);
    }
    m_Impl->EnvironmentMapProgram = m_Impl->EnvironmentConversionProgram = 0;
    if (m_Impl->GlDeleteProgram)
    {
        if (m_Impl->PlanarProgram) m_Impl->GlDeleteProgram(m_Impl->PlanarProgram);
        if (m_Impl->PlanarMipProgram) m_Impl->GlDeleteProgram(m_Impl->PlanarMipProgram);
    }
    m_Impl->PlanarProgram = m_Impl->PlanarMipProgram = 0;
    if (m_Impl->GlDeleteProgram)
    {
        if (m_Impl->MotionGeometryProgram) m_Impl->GlDeleteProgram(m_Impl->MotionGeometryProgram);
        if (m_Impl->MotionScreenProgram) m_Impl->GlDeleteProgram(m_Impl->MotionScreenProgram);
        if (m_Impl->TemporalProgram) m_Impl->GlDeleteProgram(m_Impl->TemporalProgram);
        if (m_Impl->MediaFogProgram) m_Impl->GlDeleteProgram(m_Impl->MediaFogProgram);
        if (m_Impl->CloudFogProgram) m_Impl->GlDeleteProgram(m_Impl->CloudFogProgram);
        for (auto program : m_Impl->CloudVolumePrograms) if (program) m_Impl->GlDeleteProgram(program);
        for (auto program : m_Impl->MediaVolumePrograms) if (program) m_Impl->GlDeleteProgram(program);
        if (m_Impl->FogProgram) m_Impl->GlDeleteProgram(m_Impl->FogProgram);
        if (m_Impl->VolumeLightProgram) m_Impl->GlDeleteProgram(m_Impl->VolumeLightProgram);
    }
    m_Impl->MotionGeometryProgram = m_Impl->MotionScreenProgram = 0;
    m_Impl->TemporalProgram = 0;
    if (m_Impl->GIProgram) m_Impl->GlDeleteProgram(m_Impl->GIProgram);
    if (m_Impl->GITemporalProgram) m_Impl->GlDeleteProgram(m_Impl->GITemporalProgram);
    m_Impl->GIProgram = m_Impl->GITemporalProgram = 0;
    for (auto program : m_Impl->ProbePrograms) if (program) m_Impl->GlDeleteProgram(program);
    m_Impl->ProbePrograms = {}; m_Impl->FrameProbeSamples.reset();
    m_Impl->DestroyLightmapTargets(true); m_Impl->DestroyRefractionTargets(true); m_Impl->DestroyPeelingTargets(true);
    m_Impl->FogProgram = m_Impl->MediaFogProgram = m_Impl->CloudFogProgram = 0; m_Impl->MediaVolumePrograms = {}; m_Impl->CloudVolumePrograms = {}; m_Impl->FogLocations.clear(); m_Impl->FogTargetSupported = false;
    if (m_Impl->LightShaftProgram && m_Impl->GlDeleteProgram) m_Impl->GlDeleteProgram(m_Impl->LightShaftProgram);
    m_Impl->LightShaftProgram = 0; m_Impl->LightShaftTargetSupported = false;
    m_Impl->VolumeLightProgram = 0; m_Impl->VolumeLightLocations.clear(); m_Impl->VolumeLightTargetSupported = false;
    m_Impl->FrameVolumeLighting = {};
    if (m_Impl->SsaoProgram && m_Impl->GlDeleteProgram) m_Impl->GlDeleteProgram(m_Impl->SsaoProgram);
    m_Impl->SsaoProgram = 0;
    m_Impl->DestroyPathTracing();
        m_Impl->DestroyRayGI(); m_Impl->DestroyVoxelGI();
    m_Impl->DestroyRayReflections(); m_Impl->DestroyHybridRendering(); m_Impl->DestroySubsurface(); m_Impl->DestroyHairFur(); m_Impl->DestroyExposure(); m_Impl->DestroyParallax(); m_Impl->DestroyDisplacement(); m_Impl->DestroyTessellation(); m_Impl->DestroyVirtualGeometry(); m_Impl->DestroyNaniteFrameVisibility(); m_Impl->DestroyNaniteGeometry(); m_Impl->DestroyTSR(); m_Impl->DestroyVirtualShadowRequests(); m_Impl->DestroyVirtualShadowAtlases(); m_Impl->DestroyLumenProbes(); m_Impl->DestroyLumenRadianceTraces(); m_Impl->DestroyLumenLightingCaches(); m_Impl->DestroyLumenSurfaces(); m_Impl->DestroyLumenSurfaceLookup(); m_Impl->DestroyLumenDistanceScenes();
    m_Impl->DestroyRayScene(); m_Impl->DestroyRayShadowImage(); m_Impl->DestroyPCF(); m_Impl->DestroyMaterials();
    if (m_Impl->HbaoProgram && m_Impl->GlDeleteProgram) m_Impl->GlDeleteProgram(m_Impl->HbaoProgram);
    m_Impl->HbaoProgram = 0;
    if (m_Impl->SaoProgram && m_Impl->GlDeleteProgram) m_Impl->GlDeleteProgram(m_Impl->SaoProgram);
    m_Impl->SaoProgram = 0;
    if (m_Impl->GtaoProgram && m_Impl->GlDeleteProgram) m_Impl->GlDeleteProgram(m_Impl->GtaoProgram);
    m_Impl->GtaoProgram = 0;
    if (m_Impl->GlDeleteProgram)
    {
        if (m_Impl->DeferredGeometryProgram) m_Impl->GlDeleteProgram(m_Impl->DeferredGeometryProgram);
        if (m_Impl->DeferredLightingProgram) m_Impl->GlDeleteProgram(m_Impl->DeferredLightingProgram);
        if (m_Impl->MultisampleLightingProgram) m_Impl->GlDeleteProgram(m_Impl->MultisampleLightingProgram);
        for (auto program : m_Impl->SamplePrograms) if (program) m_Impl->GlDeleteProgram(program);
    }
    m_Impl->DeferredGeometryProgram = m_Impl->DeferredLightingProgram = m_Impl->MultisampleLightingProgram = 0;
    m_Impl->SamplePrograms.fill(0);
    m_Impl->SupportedSampleCounts = m_Impl->FrameSampleCount = 1;
    m_Impl->DeferredActive = m_Impl->GBufferWritten = m_Impl->DeferredResolved = false;
    m_Impl->HdrActive = false;
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
    if(m_Impl->GpuContextOwner)
    {
        m_Impl->GpuSurfaces.clear();m_Impl->GpuDisplayOwner.reset();m_Impl->GpuContextOwner.reset();
    }
    else
    {
    if (m_Impl->Context != nullptr && m_Impl->EglDestroyContext)
        m_Impl->EglDestroyContext(m_Impl->Display, m_Impl->Context);
    if (m_Impl->Surface != nullptr && m_Impl->EglDestroySurface)
        m_Impl->EglDestroySurface(m_Impl->Display, m_Impl->Surface);
    if (!m_Impl->GpuDisplayOwner && m_Impl->Display != nullptr && m_Impl->EglTerminate) m_Impl->EglTerminate(m_Impl->Display);
    m_Impl->GpuDisplayOwner.reset();
    m_Impl->Context = nullptr;
    m_Impl->Surface = nullptr;
    m_Impl->Display = nullptr;
    if (m_Impl->EGLModule != nullptr) FreeLibrary(m_Impl->EGLModule);
    if (m_Impl->GLESModule != nullptr) FreeLibrary(m_Impl->GLESModule);
    }
    m_Impl->Context=nullptr;m_Impl->Surface=nullptr;m_Impl->Display=nullptr;
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
    result.MaterialAnisotropicFiltering = m_Impl->Initialized && m_Impl->MaximumAnisotropy > 1;
    result.TemporalEffects=m_Impl->Initialized && m_Impl->HalfFloatTargetsSupported && m_Impl->ExposureSupported;
    result.InstancedRendering = m_Impl->Initialized &&
        m_Impl->MaximumUniformBlockSize >= static_cast<GLint>(sizeof(InstanceConstantBlock));
    result.GPUDrivenRendering = result.InstancedRendering && m_Impl->GpuDrivenSupported;
    result.VisibilityBufferRendering = m_Impl->Initialized && m_Impl->VisibilitySupported;
    result.SceneStorageBlockBytes = m_Impl->VisibilityMaximumBytes;
    result.Wireframe = true;
    result.ShadowMaps = true;
    result.VarianceShadowMaps = m_Impl->Initialized && m_Impl->VSMTargetSupported;
    result.ContactShadows = m_Impl->Initialized && m_Impl->ContactSupported;
    result.ForwardPlusRendering = m_Impl->Initialized && m_Impl->ForwardPlusSupported;
    result.ClusteredForwardRendering = m_Impl->Initialized && m_Impl->ClusteredSupported;
    result.DeferredShading = m_Impl->Initialized && m_Impl->HalfFloatTargetsSupported &&
        m_Impl->MaximumDrawBuffers >= static_cast<GLint>(GBufferAttachmentCount) &&
        m_Impl->MaximumColorAttachments >= static_cast<GLint>(GBufferAttachmentCount);
    result.ParallaxOcclusion=m_Impl->Initialized && result.DeferredShading && m_Impl->ParallaxShaderSupported;
    result.GeometryDisplacement=m_Impl->Initialized && m_Impl->DisplacementSupported;
        result.GeometryTessellation=m_Impl->Initialized && m_Impl->TessellationSupported;
        result.GeometryVirtualization=m_Impl->Initialized && m_Impl->VirtualGeometrySupported;
    if (result.VisibilityBufferRendering && m_Impl->NaniteGeometrySupported)
    {
        result.NaniteRendering=m_Impl->NaniteDeviceAvailable;
        result.NaniteHardwareScheduling=m_Impl->NaniteHardwareAvailable;
    }
        result.VirtualShadowMaps=m_Impl->Initialized && m_Impl->VirtualShadowSupported;
    result.TiledDeferredRendering = result.DeferredShading && m_Impl->TiledSupported;
    result.ScalableAmbientObscurance = result.DeferredShading && m_Impl->FloatDepthTargetSupported;
    result.TemporalAntiAliasing = m_Impl->Initialized && m_Impl->HalfFloatTargetsSupported &&
        m_Impl->FloatDepthTargetSupported && m_Impl->MaximumDrawBuffers >= 2 && m_Impl->MaximumColorAttachments >= 2;
    result.TemporalSuperResolution=m_Impl->Initialized && m_Impl->TSRSupported;
    result.ScreenSpaceGlobalIllumination = result.DeferredShading && result.TemporalAntiAliasing;
    result.BakedLightProbes = result.DeferredShading;
        result.BakedLightmaps = result.DeferredShading;
        result.VoxelGI = result.DeferredShading && m_Impl->Initialized && m_Impl->VoxelSamplingSupported;
        result.LumenGI=result.DeferredShading && m_Impl->Initialized && m_Impl->LumenGatherSupported;
    result.Refraction = result.DeferredShading;
        result.RayTracedAmbientOcclusion = result.DeferredShading && m_Impl->RaySceneSupported;
        result.RayTracedReflections = result.DeferredShading && m_Impl->RayRadianceSupported;
        result.HybridRendering = result.DeferredShading && result.RayTracedReflections;
        result.RayTracedGI = result.RayTracedReflections;
        result.PathTracing = result.RayTracedReflections && m_Impl->PathTargetsSupported;
    result.VolumetricFog = m_Impl->Initialized && m_Impl->FogTargetSupported;
    result.VolumetricLighting = m_Impl->Initialized && m_Impl->VolumeLightTargetSupported;
        result.DepthPeeling = result.Refraction && result.VolumetricLighting;
    result.LightShafts = m_Impl->Initialized && m_Impl->LightShaftTargetSupported;
    result.WeightedTransparencySampleCounts = m_Impl->Initialized ? m_Impl->WeightedSampleCounts : 0;
    result.Outline = true;
    result.SupportedSampleCounts = m_Impl->SupportedSampleCounts;
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
    if (m_Impl->ContactDepthFramebuffer) m_Impl->GlDeleteFramebuffers(1, &m_Impl->ContactDepthFramebuffer);
    m_Impl->ContactDepthFramebuffer = 0;
    m_Impl->Hair.ResetHistory();
    m_Impl->ExposureHistory.Invalidate();
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

bool OpenGLESRHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& candidate,VirtualGeometryCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache updates require a closed frame");
    return m_Impl->PrepareVirtualGeometryCache(upload,previous,candidate,statistics);
}
bool OpenGLESRHI::SubmitNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheTask& task,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Submit Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->SubmitNaniteCache(upload,previous,task,statistics);
}
bool OpenGLESRHI::PollNaniteCache(NaniteCacheTask task,bool& ready,NaniteCacheHandle& output)
{
    if (!m_Impl->CheckThread("Poll Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->PollNaniteCache(task,ready,output);
}
bool OpenGLESRHI::CancelNaniteCache(NaniteCacheTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->CancelNaniteCache(task);
}
bool OpenGLESRHI::PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& candidate,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache updates require a closed frame");
    return m_Impl->PrepareNaniteCache(upload,previous,candidate,statistics);
}
bool OpenGLESRHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache release requires a closed frame");
    return m_Impl->DestroyVirtualGeometryCache(handle);
}
bool OpenGLESRHI::DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache release requires a closed frame");
    return m_Impl->DestroyNaniteCache(handle);
}
bool OpenGLESRHI::SnapshotTSROpaque()
{
    if (!m_Impl->CheckThread("Snapshot TSR opaque")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || m_Impl->PassOpen) return m_Impl->Fail("TSR opaque snapshot requires a closed pass in an initialized frame");
    return m_Impl->SnapshotTSROpaque();
}
bool OpenGLESRHI::RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& settings,FrameCapture& output)
{
    if (!m_Impl->CheckThread("Render TSR display")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR display work requires an initialized RHI and closed frame");
    return m_Impl->RenderTSRDisplay(handle,settings,output);
}
bool OpenGLESRHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    if (!m_Impl->CheckThread("BuildVirtualShadowReceiverRequests")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow receivers require an initialized RHI and closed frame");
    return m_Impl->BuildVirtualShadowReceiverRequests(upload,output);
}
bool OpenGLESRHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    if (!m_Impl->CheckThread("SampleVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow lookup requires an initialized RHI and closed frame");
    return m_Impl->SampleVirtualShadowAtlas(handle,upload,output);
}
bool OpenGLESRHI::CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!m_Impl->CheckThread("CreateLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->CreateLumenDistanceScene(data,output);
}
bool OpenGLESRHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    if (!m_Impl->CheckThread("CreateLumenGlobalDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CreateLumenGlobalDistanceScene(source,settings,output,previous);
}

bool OpenGLESRHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenGlobalDistanceUpdate(source,settings,previous,task);
}

bool OpenGLESRHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenGlobalDistanceUpdate(task,budget,ready,output,statistics);
}

bool OpenGLESRHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{
    if (!m_Impl->CheckThread("CancelLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenGlobalDistanceUpdate(task);
}
bool OpenGLESRHI::TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->TraceLumenDistanceScene(handle,upload,output);
}
bool OpenGLESRHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenDistanceScene(handle);
}
bool OpenGLESRHI::RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    if (!m_Impl->CheckThread("RelightLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->RelightLumenSurface(surface,distance,upload,output);
}

bool OpenGLESRHI::BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenLightingUpdate(surface,distance,upload,task);
}

bool OpenGLESRHI::AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenLightingUpdate(task,budget,ready,output,statistics);
}

bool OpenGLESRHI::CancelLumenLightingUpdate(LumenLightingTask task)
{
    if (!m_Impl->CheckThread("CancelLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenLightingUpdate(task);
}
bool OpenGLESRHI::ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenLighting(handle,output);
}
bool OpenGLESRHI::DestroyLumenLighting(LumenLightingHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenLighting(handle);
}
bool OpenGLESRHI::TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenRadiance")) return false;
    if (upload.Probes.DirectionalSide && m_Impl->FrameOpen) return m_Impl->Fail("Directional world cache generation requires a closed frame");
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->TraceLumenRadiance(lighting,upload,output);
}
bool OpenGLESRHI::AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("AssembleLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("World radiance assembly requires an initialized RHI and closed frame");
    if (!m_Impl->ValidateLumenRecording(false)) return false;
    return m_Impl->AssembleLumenRadiance(assembly,output);
}
bool OpenGLESRHI::ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbeOrigins")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbeOrigins(handle,output);
}
bool OpenGLESRHI::ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    if (!m_Impl->CheckThread("ReadLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenRadiance(handle,output);
}
bool OpenGLESRHI::DestroyLumenRadiance(LumenRadianceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenRadiance(handle);
}
bool OpenGLESRHI::IntegrateLumenProbes(LumenRadianceHandle source,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    if (!m_Impl->CheckThread("IntegrateLumenProbes")) return false;
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->IntegrateLumenProbes(source,upload,output);
}
bool OpenGLESRHI::ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if (!m_Impl->CheckThread("ResolveLumenProbes")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || !m_Impl->ValidateLumenRecording(false)) return m_Impl->Fail("Lumen gather requires an open frame outside a graphics pass");
    return m_Impl->ResolveLumenProbes(handle,settings);
}
bool OpenGLESRHI::ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbes(handle,output);
}
bool OpenGLESRHI::DestroyLumenProbes(LumenProbeHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenProbes(handle);
}
bool OpenGLESRHI::LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    if (!m_Impl->CheckThread("LookupLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface lookup requires an initialized RHI and closed frame");
    return m_Impl->LookupLumenSurface(handle,upload,output);
}
bool OpenGLESRHI::CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    if (!m_Impl->CheckThread("CaptureLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->CaptureLumenSurface(upload,output);
}
bool OpenGLESRHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenCaptureUpdate(upload,task);
}
bool OpenGLESRHI::AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenCaptureUpdate(task,budget,ready,output,statistics);
}
bool OpenGLESRHI::CancelLumenCaptureUpdate(LumenCaptureTask task)
{
    if (!m_Impl->CheckThread("CancelLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenCaptureUpdate(task);
}
bool OpenGLESRHI::ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->ReadLumenSurface(handle,output);
}
bool OpenGLESRHI::DestroyLumenSurface(LumenSurfaceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenSurface(handle);
}
bool OpenGLESRHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    if (!m_Impl->CheckThread("RenderVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->RenderVirtualShadowAtlas(upload,previous,output);
}
bool OpenGLESRHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->ReadVirtualShadowAtlas(handle,output);
}
bool OpenGLESRHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{
    if (!m_Impl->CheckThread("DestroyVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->DestroyVirtualShadowAtlas(handle);
}
bool OpenGLESRHI::DestroyTSRHistory(TSRHistoryHandle handle)
{
    if (!m_Impl->CheckThread("Destroy TSR history")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR history release requires an initialized RHI and closed frame");
    return m_Impl->DestroyTSRHistory(handle);
}
bool OpenGLESRHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle handle,const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached virtual geometry")) return false;
    if (m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateVirtualGeometry(upload,output,handle);
}
bool OpenGLESRHI::SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& task)
{
    if (!m_Impl->CheckThread("Submit Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->SubmitNaniteGeometry(cache,upload,task);
}
bool OpenGLESRHI::PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Poll Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->PollNaniteGeometry(task,ready,output);
}
bool OpenGLESRHI::CancelNaniteGeometry(NaniteGeometryTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->CancelNaniteGeometry(task);
}
bool OpenGLESRHI::CreateCachedNaniteGeometry(NaniteCacheHandle handle,const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateNaniteGeometry(upload,output,handle);
}

bool OpenGLESRHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create virtual geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual geometry must be produced between frames");
    return m_Impl->CreateVirtualGeometry(upload,output);
}
bool OpenGLESRHI::PrepareNaniteDraws(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDraws") && m_Impl->PrepareNaniteDraws(visibility,draws,output);
}
bool OpenGLESRHI::PrepareNaniteDrawBins(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDrawBins") && m_Impl->PrepareNaniteDraws(visibility,draws,output,true);
}
bool OpenGLESRHI::DrawNaniteBins(NaniteDrawHandle handle)
{
    if (!m_Impl->CheckThread("DrawNaniteBins") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->NaniteCommandsEnded() ||
        m_Impl->CurrentPass.Target || m_Impl->CurrentPass.DepthPeelingLayer ||
        (m_Impl->CurrentPass.Type!=PassType::Color && m_Impl->CurrentPass.Type!=PassType::GBuffer)) return m_Impl->Fail("Nanite bins require an open primary opaque graphics pass");
    const auto found=m_Impl->NaniteDrawBatches.find(handle.Value);
    if (found==m_Impl->NaniteDrawBatches.end() || !found->second->Binning ||
        std::any_of(found->second->Submitted.begin(),found->second->Submitted.end(),[](bool submitted){return submitted;}))
        return m_Impl->Fail("Nanite bin handle is invalid or already consumed");
    auto& record=*found->second;
    if (record.Phase==NaniteVisibilityPhase::Post && m_Impl->CurrentPass.ClearDepthEnabled) return m_Impl->Fail("Nanite post bins must preserve main depth");
    for (unsigned id=0;id<record.Binning->Bins.size();++id)
    {
        const auto& bin=record.Binning->Bins[id]; const auto& draw=(*record.Draws)[bin.FirstDraw];
        if (!BindPipeline(draw.Pipeline) || !BindWorldTransform(draw.World) || !BindViewProjection(draw.ViewProjection) ||
            !BindLightViewProjection(draw.LightViewProjection) || !BindVertexBuffer(draw.VertexBuffer) || !BindIndexBuffer(draw.IndexBuffer)) return false;
        m_Impl->ActiveNaniteDraw=&record; m_Impl->ActiveNaniteQuery=bin.FirstDraw; m_Impl->ActiveNaniteBin=id;
        struct Reset { Impl& Owner; ~Reset() { Owner.ActiveNaniteDraw=nullptr; } } reset{*m_Impl};
        if (!DrawIndexedGeometry(bin.FirstIndex,bin.IndexCount,0,nullptr)) return false;
        std::fill(record.Submitted.begin()+bin.FirstDraw,record.Submitted.begin()+bin.FirstDraw+bin.DrawCount,true);
    }
    return true;
}
bool OpenGLESRHI::DrawNaniteCluster(NaniteDrawHandle handle,unsigned query)
{
    if (!m_Impl->CheckThread("DrawNaniteCluster") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->NaniteCommandsEnded() ||
        m_Impl->CurrentPass.Target || m_Impl->CurrentPass.DepthPeelingLayer ||
        (m_Impl->CurrentPass.Type!=PassType::Color && m_Impl->CurrentPass.Type!=PassType::GBuffer)) return m_Impl->Fail("Nanite draw requires an open primary opaque graphics pass");
    const auto found=m_Impl->NaniteDrawBatches.find(handle.Value);
    if (found==m_Impl->NaniteDrawBatches.end() || found->second->Binning || query>=found->second->Draws->size() || found->second->Submitted[query])
        return m_Impl->Fail("Nanite draw handle, query index or duplicate submission is invalid");
    auto& record=*found->second;
    if (record.Phase==NaniteVisibilityPhase::Post && m_Impl->CurrentPass.ClearDepthEnabled)
        return m_Impl->Fail("Nanite post pass must preserve main depth");
    const auto& draw=(*record.Draws)[query];
    if (!BindPipeline(draw.Pipeline) || !BindWorldTransform(draw.World) || !BindViewProjection(draw.ViewProjection) ||
        !BindLightViewProjection(draw.LightViewProjection) || !BindVertexBuffer(draw.VertexBuffer) || !BindIndexBuffer(draw.IndexBuffer)) return false;
    m_Impl->ActiveNaniteDraw=&record; m_Impl->ActiveNaniteQuery=query;
    struct Reset { Impl& Owner; ~Reset() { Owner.ActiveNaniteDraw=nullptr; } } reset{*m_Impl};
    if (!DrawIndexedGeometry(draw.FirstIndex,draw.IndexCount,0,nullptr)) return false;
    record.Submitted[query]=true; return true;
}
bool OpenGLESRHI::CaptureNaniteDepth(NaniteDepthHandle& output)
{
    if (!m_Impl->CheckThread("CaptureNaniteDepth") || !IsInitialized()) return false;
    return m_Impl->CaptureNaniteDepth(output);
}
bool OpenGLESRHI::BeginNaniteVisibility(NaniteDepthHandle history,const std::vector<NaniteVisibilityQuery>& queries,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("BeginNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->BeginNaniteVisibility(history,queries,output);
}
bool OpenGLESRHI::RetestNaniteVisibility(NaniteVisibilityHandle main,NaniteDepthHandle depth,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("RetestNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->RetestNaniteVisibility(main,depth,output);
}
bool OpenGLESRHI::ReadNaniteDepth(NaniteDepthHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(*found->second->Data,words) && ReadNaniteDepthWords(found->second->Info,words,output,m_Impl->LastError);
}
bool OpenGLESRHI::ReadNaniteVisibility(NaniteVisibilityHandle handle,std::vector<NaniteVisibilityState>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(*found->second->Data,words) && ReadNaniteDecisionWords(found->second->Info,words,output,m_Impl->LastError);
}
bool OpenGLESRHI::DestroyNaniteDepth(NaniteDepthHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDepths.erase(found); return true;
}
bool OpenGLESRHI::DestroyNaniteVisibility(NaniteVisibilityHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDecisions.erase(found); return true;
}
bool OpenGLESRHI::CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite geometry must be produced between frames");
    return m_Impl->CreateNaniteGeometry(upload,output);
}

bool OpenGLESRHI::CreateTessellatedGeometry(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create tessellated geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Tessellated geometry must be prepared between frames");
    return m_Impl->CreateTessellated(geometry,settings,output);
}

bool OpenGLESRHI::CreateDisplacedVertexBuffer(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    if (!m_Impl->CheckThread("Create displaced vertex buffer")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Displaced geometry must be prepared between frames");
    return m_Impl->CreateDisplaced(geometry,settings,output);
}

bool OpenGLESRHI::UpdateBuffer(BufferHandle buffer, std::size_t offset, const void* data, std::size_t size)
{
    if (!m_Impl->CheckThread("UpdateBuffer")) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
    auto found = m_Impl->Buffers.find(buffer.Value);
    if (found == m_Impl->Buffers.end() || data == nullptr || size == 0 ||
        offset > found->second.Bytes.size() || size > found->second.Bytes.size() - offset)
        return m_Impl->Fail("OpenGL ES buffer update is invalid");
    if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
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
    if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
    auto found = m_Impl->Buffers.find(buffer.Value);
    if (!buffer || found == m_Impl->Buffers.end()) return m_Impl->Fail("OpenGL ES buffer handle is invalid");
    m_Impl->GlDeleteBuffers(1, &found->second.Buffer);
    m_Impl->Buffers.erase(found);
    return m_Impl->CheckGL("glDeleteBuffers");
}

TextureHandle OpenGLESRHI::CreateTexture(const TextureDesc& desc, const void* initialData)
{
    if (!m_Impl->CheckThread("CreateTexture") || desc.Width <= 0 || desc.Height <= 0 ||
        desc.Width > m_Impl->MaximumTextureSize || desc.Height > m_Impl->MaximumTextureSize ||
        !ValidTextureMipLevels(desc, initialData))
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
    if (!m_Impl->RetireLumenCaptureInput(texture)) return false;
    auto found = m_Impl->Textures.find(texture.Value);
    if (!texture || found == m_Impl->Textures.end()) return m_Impl->Fail("OpenGL ES texture handle is invalid");
    const auto moments = m_Impl->VSMMoments.find(texture.Value);
    if (moments != m_Impl->VSMMoments.end())
    { m_Impl->ReleaseMomentMap(moments->second); m_Impl->VSMMoments.erase(moments); }
    m_Impl->DestroyTextureResource(found->second);
    m_Impl->Textures.erase(found);
    if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
    return m_Impl->CheckGL("DestroyTexture");
}

bool OpenGLESRHI::BeginFrame()
{
    if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
        return m_Impl->Fail("OpenGL ES frame cannot begin in the current state");
    if (!m_Impl->PrepareRayShadowImage(GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced?GetRayShadowImage():nullptr)) return false;
    if (!m_Impl->CollectLumenGlobalDistance() || !m_Impl->CollectLumenLighting() || !m_Impl->CollectLumenCapture()) return false;
    const bool anisotropic = GetTechniqueSettings().AnisotropicFiltering && m_Impl->MaximumAnisotropy > 1;
    if (m_Impl->AnisotropicFiltering != anisotropic)
    {
        m_Impl->AnisotropicFiltering = anisotropic;
        for (const auto& [handle, texture] : m_Impl->Textures)
        {
            if (texture.Description.MipLevels <= 1) continue;
            m_Impl->GlBindTexture(GL_TEXTURE_2D, texture.Texture);
            m_Impl->ConfigureMaterialFiltering();
        }
        if (!m_Impl->CheckGL("Set material anisotropic filtering")) return false;
    }
    ResetMotionTransform();
    m_Impl->ExposureHistory.Begin();
    m_Impl->GpuCursor = 0; m_Impl->ActiveGpuBatch = nullptr; ++m_Impl->GpuEpoch;
    m_Impl->ResetVisibilityState();
    m_Impl->FrameShadowFilter = GetTechniqueSettings().ShadowFilter;
    if (GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced) m_Impl->FrameShadowFilter.Mode=Techniques::ShadowFilterMode::Hard;
    m_Impl->TiledActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::TiledDeferred;
        m_Impl->ClusteredActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::ClusteredForward;
    m_Impl->LightListsActive = Techniques::UsesLightLists(GetTechniqueSettings().Pipeline);
    if (m_Impl->LightListsActive && !(m_Impl->ClusteredActive ? m_Impl->ClusteredSupported : m_Impl->TiledActive ? m_Impl->TiledSupported : m_Impl->ForwardPlusSupported))
        return m_Impl->Fail("OpenGL ES Forward+ requires ES 3.1 compute, fragment storage blocks and receiver texture slots");
    m_Impl->FrameContact = GetTechniqueSettings().ContactShadows;
    if (m_Impl->FrameContact.Enabled && !m_Impl->ContactSupported)
        return m_Impl->Fail("OpenGL ES contact shadows require ES 3.1 compute and a receiver texture slot");
    m_Impl->ContactShadows.BeginFrame();
    for (auto& entry : m_Impl->VSMMoments) entry.second.Ready = entry.second.RawWritten = false;
    if (!m_Impl->PrepareVirtualShadowSampling(GetVirtualShadowSamplingFrame(),GetCascadedShadowState()) ||
            !m_Impl->PrepareCascadedShadows(GetCascadedShadowState())) return false;
    m_Impl->SubsurfaceProducing=m_Impl->SubsurfaceResolved=m_Impl->SubsurfaceHasScene=false;
    m_Impl->FrameSubsurface=GetTechniqueSettings().Subsurface;
    m_Impl->SubsurfaceRequested=Techniques::UsesSubsurfaceScattering(m_Impl->FrameSubsurface);
    m_Impl->FramePostProcessSettings = GetTechniqueSettings().PostProcessing;
    if (!m_Impl->PrepareTSRScene(GetTSRSceneFrame(),m_Impl->FramePostProcessSettings)) return false;
    m_Impl->FrameGI = GetTechniqueSettings().GlobalIllumination;
    m_Impl->FrameLumenLighting=GetLumenLighting(); m_Impl->FrameLumenWorldCache=GetLumenWorldCache(); m_Impl->FrameLumenGlobalDistance=GetLumenGlobalDistance();
    m_Impl->FrameVoxelGI=GetVoxelGIFrame();
    if (!m_Impl->Hair.Freeze(GetHairFurFrame(),GetTechniqueSettings(),m_Impl->LastError)) return false;
        if (Techniques::UsesVoxelGI(m_Impl->FrameGI))
        {
            if (!m_Impl->FrameVoxelGI.Volume) return m_Impl->Fail("Voxel GI requires a current immutable radiance publication");
            m_Impl->FrameVoxelGI.Settings=m_Impl->FrameGI.Voxels; m_Impl->FrameVoxelGI.Intensity=m_Impl->FrameGI.Intensity;
            m_Impl->FrameVoxelGI.MaximumDistance=m_Impl->FrameGI.MaximumDistance; m_Impl->FrameVoxelGI.Shading=GetTechniqueSettings().Shading;
        }
        if (m_Impl->FrameVoxelGI.Volume && ((Techniques::UsesGlobalIllumination(m_Impl->FrameGI) && !Techniques::UsesVoxelGI(m_Impl->FrameGI)) ||
            Techniques::UsesPathTracing(GetTechniqueSettings().Pipeline) ||
            !Techniques::RequiresHDR(m_Impl->FramePostProcessSettings.HDR) || m_Impl->FrameVoxelGI.Shading!=GetTechniqueSettings().Shading))
            return m_Impl->Fail("Voxel GI requires an exclusive HDR raster frame and matching receiver shading");
        m_Impl->GIActive = Techniques::UsesGlobalIllumination(m_Impl->FrameGI) || bool(m_Impl->FrameVoxelGI.Volume);
    m_Impl->EarlyTemporalActive = Techniques::UsesTemporalGI(m_Impl->FrameGI) || Techniques::UsesTemporalReflections(GetTechniqueSettings().Reflections);
    m_Impl->ProbeActive = Techniques::UsesBakedProbes(m_Impl->FrameGI);
    m_Impl->LightmapActive = Techniques::UsesLightmaps(m_Impl->FrameGI);
    m_Impl->FrameLightmaps = GetLightmapFrame();
    m_Impl->FramePeeling = GetDepthPeelingFrame(); m_Impl->PeelingSequence.Abort();
    if (!ValidateDepthPeelingFrame(m_Impl->FramePeeling,m_Impl->LastError)) return false;
    m_Impl->FrameRefraction = GetRefractionFrame();
    m_Impl->FrameRefractionSettings = GetTechniqueSettings().Refraction;
    m_Impl->RefractionActive = Techniques::UsesRefraction(GetTechniqueSettings());
    m_Impl->RefractionResolved = false;
    m_Impl->FrameLightmapSettings = GetTechniqueSettings();
    m_Impl->FrameProbeSamples = GetBakedProbeSamples();
    m_Impl->FrameFog = GetTechniqueSettings().VolumetricFog;
    m_Impl->FrameVolumeLighting = GetTechniqueSettings().VolumetricLighting;
    m_Impl->FrameLightShafts = GetTechniqueSettings().LightShafts;
    m_Impl->LightShaftActive = Techniques::UsesLightShafts(m_Impl->FrameLightShafts);
    m_Impl->LightShaftResolved = false;
    m_Impl->FogActive = Techniques::UsesVolumetricFog(m_Impl->FrameFog);
    m_Impl->FogResolved = false;
    m_Impl->FrameAreaLights = GetTechniqueSettings().AreaLights.Enabled;
        m_Impl->FrameSampleCount = GetTechniqueSettings().SampleCount;
    m_Impl->FrameTransparency = GetTechniqueSettings().Transparency;
    m_Impl->WeightedActive = Techniques::UsesWeightedTransparency(GetTechniqueSettings());
    m_Impl->HdrActive = Techniques::RequiresHDR(m_Impl->FramePostProcessSettings.HDR);
    m_Impl->DepthOfFieldActive = m_Impl->FramePostProcessSettings.DepthOfField.Enabled;
    m_Impl->MotionBlurActive = m_Impl->FramePostProcessSettings.MotionBlur.Enabled;
    m_Impl->TemporalActive = Techniques::UsesTemporalAA(m_Impl->FramePostProcessSettings);
    m_Impl->VelocityActive = (Techniques::RequiresMotionVectors(m_Impl->FramePostProcessSettings) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive;
    if (m_Impl->HdrActive && !m_Impl->EnsureHdrTargets()) return false;
    if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI) && !m_Impl->EnsureGIResources()) return false;
    if (m_Impl->ProbeActive && !m_Impl->EnsureProbeResources()) return false;
    if (m_Impl->LightShaftActive && !m_Impl->EnsureLightShaftResources()) return false;
    if (m_Impl->FogActive && !m_Impl->EnsureFogResources()) return false;
    if (m_Impl->FogActive && (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog)) && !m_Impl->EnsureVolumeLightResources()) return false;
    m_Impl->DeferredActive = Techniques::UsesDeferredLighting(GetTechniqueSettings().Pipeline);
    m_Impl->AmbientOcclusionActive = Techniques::UsesAmbientOcclusion(GetTechniqueSettings().AmbientOcclusion);
    m_Impl->FrameAmbientOcclusionSettings = GetTechniqueSettings().AmbientOcclusion;
    m_Impl->FrameRayScene = GetRayScene();
    m_Impl->FrameRayRadiance = GetRayRadianceFrame();
    m_Impl->FrameRayRadianceSettings = GetTechniqueSettings();
        m_Impl->PathFrame.BeginFrame(Techniques::UsesPathTracing(GetTechniqueSettings().Pipeline));
    m_Impl->FrameReflections = GetTechniqueSettings().Reflections;
    m_Impl->FrameReflectionShading = GetTechniqueSettings().Shading;
    if (!m_Impl->PrepareEnvironmentCapture(GetEnvironmentCaptureFrame())) return false;
    if (!m_Impl->PreparePlanarCapture(GetPlanarCaptureFrame())) return false;
    m_Impl->ReflectionActive = Techniques::UsesReflections(m_Impl->FrameReflections);
    m_Impl->GBufferWritten = m_Impl->DeferredResolved = false;
    m_Impl->DeferredBackground = {};
    if ((m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive) && !m_Impl->EnsureDeferredResources()) return false;
    if (m_Impl->AmbientOcclusionActive && !m_Impl->EnsureAmbientOcclusionResources()) return false;
    if (m_Impl->ReflectionActive && !m_Impl->EnsureReflectionResources()) return false;
    if (Techniques::UsesRayTracedGI(m_Impl->FrameGI) && !m_Impl->EnsureRayGIResources()) return false;
    if (m_Impl->FrameVoxelGI.Volume && !m_Impl->EnsureVoxelGIResources()) return false;
    if (m_Impl->PathFrame.Active && !m_Impl->EnsurePathTracing()) return false;
    if (m_Impl->DepthOfFieldActive && !m_Impl->EnsureDepthOfFieldResources())
    {
        m_Impl->ResetVisibilityState();
        return false;
    }
    if ((m_Impl->VelocityActive && !m_Impl->EnsureVelocityResources()) ||
        (m_Impl->TemporalActive && !m_Impl->EnsureTemporalResources()) ||
        (m_Impl->MotionBlurActive && !m_Impl->EnsureMotionResources()))
    {
        m_Impl->ResetVisibilityState();
        return false;
    }
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->EnsureMultisampleResources()) return false;
    if (m_Impl->LightmapActive && !m_Impl->EnsureLightmapResources()) return false;
    if (m_Impl->FramePeeling.Enabled)
    {
        if (!m_Impl->HdrActive || m_Impl->FrameSampleCount!=1 || m_Impl->WeightedActive)
            return m_Impl->Fail("Depth peeling requires single-sample HDR without weighted accumulation");
        if (!m_Impl->EnsurePeelingResources() || !m_Impl->PeelingSequence.Begin(m_Impl->FramePeeling.Settings.LayerCount,m_Impl->LastError)) return false;
    }
    if (m_Impl->RefractionActive && !m_Impl->EnsureRefractionResources()) return false;
    if (m_Impl->WeightedActive && !m_Impl->EnsureWeightedResources()) return false;
    m_Impl->UnbindSampledTextures();
    if (m_Impl->Hair.Active && !m_Impl->EnsureHairFur()) return false;
    m_Impl->PendingFrame = {};
    m_Impl->PendingFrame.Backend = RHIBackend::OpenGLES;
    m_Impl->PendingFrame.Statistics.RasterSampleCount = m_Impl->FrameSampleCount;
    m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
    m_Impl->LastShadowTexture = {};
    m_Impl->FrameOpen = true;
    m_Impl->LastError.clear();
    if (m_Impl->FramePostProcessSettings.TemporalEffects.Enabled && !m_Impl->EnsureExposure()) return false;
    return true;
}

void OpenGLESRHI::AbortFrame()
{
    m_Impl->PeelingSequence.Abort(); m_Impl->FramePeeling={};
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
    if(!m_Impl->CheckThread("AbortFrame"))return;
        m_Impl->PathFrame.Abort();
    m_Impl->Probes.Invalidate();
    m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
    m_Impl->UnbindSampledTextures();
    for (auto& entry : m_Impl->VSMMoments) entry.second.Ready = entry.second.RawWritten = false;
    m_Impl->GlColorMask(
        GL_TRUE_VALUE,
        GL_TRUE_VALUE,
        GL_TRUE_VALUE,
        GL_TRUE_VALUE);
    m_Impl->GlDepthMask(GL_TRUE_VALUE);
    m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlBlendEquation(GL_FUNC_ADD);
    if (m_Impl->MultisampleAPI)
    {
        m_Impl->GlDisable(GL_SAMPLE_MASK);
        m_Impl->GlSampleMaski(0, std::numeric_limits<GLuint>::max());
    }
    m_Impl->PendingFrame = {};
    m_Impl->CurrentPass = {};
    m_Impl->BoundPipeline = {};
    m_Impl->BoundVertexBuffer = {};
    m_Impl->BoundIndexBuffer = {};
    m_Impl->LastShadowTexture = {};
    m_Impl->AbortTSRScene();
    m_Impl->FrameOpen = false;
    m_Impl->FinishLumenFrame(false); m_Impl->FinishNaniteFrame(false);
    m_Impl->FrameCascades = {};
    m_Impl->CascadeTextures.fill(0);
    SetCascadedShadowState({});
    SetEnvironmentCaptureFrame({});
    m_Impl->ResetVisibilityState();
    SetPlanarCaptureFrame({});
    m_Impl->PassOpen = false;
    m_Impl->PipelineBound = false;
    m_Impl->WorldBound = false;
    m_Impl->ViewProjectionBound = false;
    m_Impl->LightViewProjectionBound = false;
}

bool OpenGLESRHI::BeginPass(const PassDesc& desc)
{
    if (!m_Impl->CheckThread("TSR scene boundary")) return false;
    if (m_Impl->TSRScene.Active && m_Impl->TSRScene.OpaqueReady && !desc.Target && (desc.Type==PassType::GBuffer || (desc.Type==PassType::Color && !desc.DepthPeelingLayer && desc.ClearColorEnabled))) return m_Impl->Fail("Opaque scene writes cannot restart after the TSR snapshot");

    if (m_Impl->FrameOpen && m_Impl->ExposureHistory.Resolved && desc.Type!=PassType::Outline)
        return m_Impl->Fail("Temporal exposure is the last scene-HDR stage before display resolve");
    if (m_Impl->FrameOpen && m_Impl->Hair.Resolved &&
        ((desc.Type==PassType::Color && !desc.Target && desc.ClearColorEnabled) ||
         (desc.ClearDepthEnabled && desc.Type!=PassType::Shadow && desc.Type!=PassType::EnvironmentCapture && desc.Type!=PassType::PlanarCapture)))
        return m_Impl->Fail("Completed hair color and primary depth cannot be cleared by a later pass");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete() && !desc.Target && (desc.Type==PassType::Velocity || desc.Type==PassType::Outline))
        return m_Impl->Fail("Complete hair rasterization before velocity or outline draws");
    if (m_Impl->FrameOpen && m_Impl->Hair.Resolved && !desc.Target && desc.Type==PassType::GBuffer)
        return m_Impl->Fail("Opaque geometry cannot replace completed hair depth");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete() && !desc.Target &&
        (desc.Type==PassType::Color || desc.Type==PassType::Outline))
        return m_Impl->Fail("Complete subsurface diffusion before transparent or outline composition");
    if (!m_Impl->CheckThread("BeginPass") || !m_Impl->FrameOpen || m_Impl->PassOpen)
        return m_Impl->Fail("OpenGL ES pass cannot begin in the current state");
        if (m_Impl->PathFrame.Active) return m_Impl->Fail("Path tracing owns primary visibility; raster passes are inactive");
    if (!ValidateDepthPeelingPass(desc,m_Impl->FramePeeling,m_Impl->PeelingSequence,m_Impl->LastError)) return false;
    const bool weighted = m_Impl->WeightedActive && m_Impl->WeightedPrepared && !m_Impl->WeightedResolved;
    if (weighted && (desc.Type != PassType::Color || desc.Target || desc.ClearColorEnabled || desc.ClearDepthEnabled || m_Impl->WeightedAccumulated))
        return m_Impl->Fail("OpenGL ES weighted transparency accepts one depth-preserving color pass before resolve");
    if (m_Impl->WeightedActive && (m_Impl->ForwardColorWritten || m_Impl->DeferredResolved) &&
        desc.Type == PassType::Color && !desc.Target && !weighted)
        return m_Impl->Fail("OpenGL ES prepare weighted transparency before transparent geometry; resolve closes that phase");
    if (desc.Type == PassType::DeferredLighting)
        return m_Impl->Fail("Use DrawDeferredLighting for OpenGL ES fullscreen lighting");
    const bool environmentCapture = desc.Type == PassType::EnvironmentCapture;
    const bool planarCapture = desc.Type == PassType::PlanarCapture;
    const bool capture = environmentCapture || planarCapture;
    if (environmentCapture && (!m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || desc.Target ||
        desc.EnvironmentFaceIndex >= EnvironmentFaceCount || !desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
        m_Impl->ForwardColorWritten || m_Impl->GBufferWritten || (m_Impl->EnvironmentFacesWritten & (1u << desc.EnvironmentFaceIndex))))
        return m_Impl->Fail("OpenGL ES capture requires six distinct cleared faces before main geometry");
    if (!environmentCapture && m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady && desc.Type != PassType::Shadow)
        return m_Impl->Fail("OpenGL ES complete all capture faces and mips before main geometry");
    if (planarCapture && (!m_Impl->PlanarActive || m_Impl->PlanarWritten || m_Impl->PlanarReady || desc.Target ||
        !desc.ClearColorEnabled || !desc.ClearDepthEnabled || m_Impl->ForwardColorWritten || m_Impl->GBufferWritten))
        return m_Impl->Fail("OpenGL ES capture the planar view once with cleared private color/depth before main geometry");
    if (!planarCapture && m_Impl->PlanarActive && !m_Impl->PlanarReady && desc.Type != PassType::Shadow && desc.Type != PassType::EnvironmentCapture)
        return m_Impl->Fail("OpenGL ES complete planar capture and mips before main geometry");
    const bool velocity = desc.Type == PassType::Velocity;
    if (m_Impl->LightShaftActive && velocity && !m_Impl->EarlyTemporalActive && !m_Impl->LightShaftResolved)
        return m_Impl->Fail("OpenGL ES light shafts must precede opaque velocity and temporal effects");
    if (m_Impl->LightShaftResolved && desc.Type != PassType::Shadow &&
        (desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
        return m_Impl->Fail("OpenGL ES scattered color and opaque depth cannot be cleared");
    if (m_Impl->FogActive && velocity && !m_Impl->EarlyTemporalActive && !m_Impl->FogResolved)
        return m_Impl->Fail("OpenGL ES fog must complete before opaque velocity and temporal effects");
    if (m_Impl->FogResolved && desc.Type != PassType::Shadow &&
        (desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
        return m_Impl->Fail("OpenGL ES fogged scene color and depth cannot be cleared");
    const bool multisample = m_Impl->FrameSampleCount > 1 && !capture;
    if (multisample && desc.Target && desc.Type != PassType::Shadow)
        return m_Impl->Fail("OpenGL ES MSAA requires internal sample-count-matched attachments");
    if (multisample && (velocity || desc.Type == PassType::Outline) && !m_Impl->SamplesOpaqueResolved)
        return m_Impl->Fail("OpenGL ES resolve sample depth before velocity or outline draws");
    if (multisample && m_Impl->SamplesOpaqueResolved && desc.Type == PassType::Color &&
        (desc.ClearColorEnabled || desc.ClearDepthEnabled || (m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored)))
        return m_Impl->Fail("OpenGL ES transparency must load processed sample color and preserve opaque depth");
    const bool opaqueComplete = m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten;
    if (velocity && (!m_Impl->VelocityActive || desc.Target || desc.ClearDepthEnabled || !desc.ClearColorEnabled ||
        !opaqueComplete || m_Impl->VelocityCaptured || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
        (!m_Impl->EarlyTemporalActive && m_Impl->ReflectionActive && !m_Impl->ReflectionResolved)))
        return m_Impl->Fail("OpenGL ES velocity requires opaque lighting/AO and a fresh internal target, once before blur");
    if (m_Impl->VelocityCaptured && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
        return m_Impl->Fail("OpenGL ES opaque depth cannot be cleared after velocity capture");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->TemporalActive)
    {
        if (opaqueComplete && !m_Impl->TemporalResolved)
            return m_Impl->Fail("OpenGL ES transparent color must follow temporal reconstruction");
        if (m_Impl->TemporalResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("OpenGL ES transparent color must load reconstructed scene color and depth");
    }
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->MotionBlurActive)
    {
        if (opaqueComplete && !m_Impl->MotionBlurResolved)
            return m_Impl->Fail("OpenGL ES transparent color must follow motion blur");
        if (m_Impl->MotionBlurResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("OpenGL ES transparent color must load completed scene color and depth");
    }
    if (m_Impl->ReflectionResolved && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
        return m_Impl->Fail("OpenGL ES opaque depth cannot be cleared after SSR");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->GIActive)
    {
        if (!m_Impl->GBufferWritten) return m_Impl->Fail("SSGI requires material capture before lighting");
        if (opaqueComplete && !m_Impl->GI.Resolved) return m_Impl->Fail("Transparency must follow SSGI");
        if (m_Impl->GI.Resolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("Scene attachments cannot be cleared after SSGI");
    }
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->ReflectionActive)
    {
        if (!m_Impl->GBufferWritten) return m_Impl->Fail("OpenGL ES SSR requires material capture before opaque lighting");
        if (opaqueComplete && !m_Impl->ReflectionResolved) return m_Impl->Fail("OpenGL ES transparency must follow SSR");
        if (m_Impl->ReflectionResolved && desc.ClearColorEnabled) return m_Impl->Fail("OpenGL ES scene color cannot be cleared after SSR");
    }
    const bool geometry = desc.Type == PassType::GBuffer;
    const bool contactDepth = desc.Type == PassType::ContactDepth;
    const bool forwardPlusDepth = desc.Type == PassType::ForwardPlusDepth;
    if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Prepared())
        return m_Impl->Fail("OpenGL ES Forward+ requires complete scene lights before any frame pass");
    if (forwardPlusDepth && (m_Impl->ClusteredActive || m_Impl->TiledActive || !m_Impl->LightListsActive || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
        m_Impl->ForwardPlusDepthWritten || m_Impl->GBufferWritten || m_Impl->LightListFrame.Assigned() ||
        m_Impl->ForwardColorWritten || m_Impl->ContactGenerated))
        return m_Impl->Fail("OpenGL ES Forward+ depth must clear opaque camera depth once before assignment and lighting");
    if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && !desc.Target && !capture && desc.Type != PassType::Shadow &&
        (geometry || desc.ClearDepthEnabled))
        return m_Impl->Fail("OpenGL ES Forward+ light lists require preserved camera depth after assignment");
    if (m_Impl->LightListsActive && desc.Type == PassType::Color && !desc.Target &&
        (!m_Impl->LightListFrame.Assigned() || (!m_Impl->ClusteredActive && desc.ClearDepthEnabled)))
        return m_Impl->Fail("OpenGL ES Forward+ lighting requires completed assignment and preserved camera depth");
    if (m_Impl->ContactGenerated && !desc.Target && !capture && desc.Type != PassType::Shadow &&
        (geometry || desc.ClearDepthEnabled))
        return m_Impl->Fail("OpenGL ES contact visibility requires immutable camera depth after generation");
    if (contactDepth && (!m_Impl->FrameContact.Enabled || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
        m_Impl->ContactDepthWritten || m_Impl->GBufferWritten || m_Impl->ContactGenerated || m_Impl->ForwardColorWritten || m_Impl->DeferredResolved))
        return m_Impl->Fail("OpenGL ES contact depth must clear opaque camera depth once before visibility and lighting");
    if (m_Impl->FrameContact.Enabled && desc.Type == PassType::Color && !desc.Target &&
        (!m_Impl->ContactGenerated || desc.ClearDepthEnabled))
        return m_Impl->Fail("OpenGL ES contact lighting requires completed visibility and preserved camera depth");
    if (m_Impl->DepthOfFieldResolved && (geometry ||
        (desc.Type == PassType::Color && !desc.Target && (desc.ClearColorEnabled || desc.ClearDepthEnabled))))
        return m_Impl->Fail("OpenGL ES post-DoF transparency must load the completed scene and opaque depth");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->DepthOfFieldActive &&
        !m_Impl->DepthOfFieldResolved && (m_Impl->DeferredActive || m_Impl->ForwardColorWritten))
        return m_Impl->Fail("OpenGL ES depth of field must precede transparent forward draws");
    if (geometry && (!(m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive) || desc.Target ||
        m_Impl->DeferredResolved || m_Impl->AmbientOcclusionResolved || m_Impl->ForwardColorWritten))
        return m_Impl->Fail("OpenGL ES material capture requires an unresolved deferred/AO/SSR frame and the main target");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->DeferredActive && !m_Impl->DeferredResolved)
        return m_Impl->Fail("OpenGL ES forward transparency must follow deferred lighting");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved &&
        (!m_Impl->GBufferWritten || m_Impl->DeferredActive || m_Impl->ForwardColorWritten))
        return m_Impl->Fail("OpenGL ES ambient occlusion must follow captured opaque color and precede transparency");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->AmbientOcclusionResolved &&
        (desc.ClearColorEnabled || desc.ClearDepthEnabled))
        return m_Impl->Fail("OpenGL ES post-AO transparency must load opaque color and depth");
    GLuint framebuffer = m_Impl->HdrActive ? m_Impl->HdrFramebuffers[0] : m_Impl->MainFramebuffer;
    if (geometry) framebuffer = m_Impl->GBufferFramebuffer;
    if (velocity) framebuffer = m_Impl->MotionFramebuffers[Impl::MotionVelocity];
    if (multisample && !desc.Target && desc.Type != PassType::Outline)
        framebuffer = m_Impl->Multisample.Framebuffers[geometry ? Impl::SampleMaterialFBO :
            (velocity ? Impl::SampleVelocityFBO : Impl::SampleColorFBO)];
    if (weighted) framebuffer = m_Impl->Weighted.Framebuffer;
    if (m_Impl->MultisampleAPI) m_Impl->GlDisable(GL_SAMPLE_MASK);
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
    m_Impl->UnbindSampledTextures();
    if (environmentCapture)
    {
        framebuffer = m_Impl->Environment.Framebuffers[desc.EnvironmentFaceIndex];
        width = height = static_cast<int>(m_Impl->FrameEnvironment.Resolution);
    }
    else if (planarCapture)
    {
        framebuffer = m_Impl->Planar.CaptureFramebuffer;
        width = static_cast<int>(m_Impl->FramePlanar.Width); height = static_cast<int>(m_Impl->FramePlanar.Height);
    }
    if (desc.Type == PassType::Shadow && m_Impl->UsesVSM() &&
        !m_Impl->BeginMoments(desc.Target, desc.ClearDepthEnabled, framebuffer)) return false;
    if (contactDepth || forwardPlusDepth)
    {
        if (!m_Impl->ContactDepthFramebuffer) m_Impl->GlGenFramebuffers(1, &m_Impl->ContactDepthFramebuffer);
        framebuffer = m_Impl->ContactDepthFramebuffer;
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        m_Impl->GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            multisample ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D,
            multisample ? m_Impl->Multisample.Depth : m_Impl->DepthTexture, 0);
        const GLenum noColor = GL_NONE_VALUE;
        m_Impl->GlDrawBuffers(1, &noColor); m_Impl->GlReadBuffer(noColor);
        if (m_Impl->GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            return m_Impl->Fail("OpenGL ES camera-depth framebuffer is incomplete");
    }
    if (desc.DepthPeelingLayer && !m_Impl->BeginPeelingLayer(framebuffer)) return false;
    m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    m_Impl->GlViewport(0, 0, width, height);
    GLbitfield clearMask = 0;
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    if (desc.Type == PassType::Shadow && m_Impl->UsesVSM() && desc.ClearDepthEnabled)
    {
        m_Impl->GlClearColor(1, 1, 0, 0);
        clearMask |= GL_COLOR_BUFFER_BIT;
    }
    if (desc.ClearColorEnabled && desc.Type != PassType::Shadow)
    {
        const bool hdr = m_Impl->HdrActive && !desc.Target;
        m_Impl->GlClearColor(hdr ? std::clamp(desc.ClearColor.X, 0.0f, MaximumHalfFloatRadiance) : desc.ClearColor.X,
            hdr ? std::clamp(desc.ClearColor.Y, 0.0f, MaximumHalfFloatRadiance) : desc.ClearColor.Y,
            hdr ? std::clamp(desc.ClearColor.Z, 0.0f, MaximumHalfFloatRadiance) : desc.ClearColor.Z, desc.ClearColor.W);
        if (geometry)
        {
            m_Impl->DeferredBackground = {std::clamp(desc.ClearColor.X, 0.0f, MaximumHalfFloatRadiance),
                std::clamp(desc.ClearColor.Y, 0.0f, MaximumHalfFloatRadiance),
                std::clamp(desc.ClearColor.Z, 0.0f, MaximumHalfFloatRadiance), desc.ClearColor.W};
            m_Impl->GlClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (velocity) m_Impl->GlClearColor(0.0f, 0.0f, 0.0f, 0.0f);
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
    m_Impl->WeightedPass = weighted;
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
    if (!m_Impl->CheckThread("TSR scene boundary")) return false;
    if (m_Impl->TSRScene.Active && m_Impl->TSRScene.OpaqueReady && !m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.DepthPeelingLayer && !pipeline.BlendEnabled) return m_Impl->Fail("Only transparent color draws follow the TSR opaque snapshot");

    if (m_Impl->FrameOpen && m_Impl->Hair.Active && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target &&
        ((!m_Impl->Hair.Resolved && pipeline.BlendEnabled) || (m_Impl->Hair.Resolved && pipeline.DepthWrite)))
        return m_Impl->Fail("Opaque draws precede hair; transparent draws follow it without replacing depth");
    if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
        return m_Impl->Fail("OpenGL ES pipeline must be bound inside a pass");
    if (m_Impl->CurrentPass.Type == PassType::ContactDepth &&
        (pipeline.BlendEnabled || !pipeline.DepthTest || !pipeline.DepthWrite || pipeline.ColorWrite || pipeline.OutlineExtrusion))
        return m_Impl->Fail("OpenGL ES contact depth requires opaque depth writes without color");
    if (m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth &&
        (pipeline.BlendEnabled || !pipeline.DepthTest || !pipeline.DepthWrite || pipeline.ColorWrite || pipeline.OutlineExtrusion))
        return m_Impl->Fail("OpenGL ES Forward+ depth requires opaque depth writes without color");
    if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && !m_Impl->CurrentPass.Target && pipeline.DepthWrite &&
        m_Impl->CurrentPass.Type != PassType::Shadow && m_Impl->CurrentPass.Type != PassType::EnvironmentCapture &&
        m_Impl->CurrentPass.Type != PassType::PlanarCapture)
        return m_Impl->Fail("OpenGL ES Forward+ receivers must preserve assigned camera depth");
    if (m_Impl->LightListsActive && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && !pipeline.DepthTest)
        return m_Impl->Fail("OpenGL ES Forward+ tile receivers require camera depth testing");
    if (m_Impl->ContactGenerated && !m_Impl->CurrentPass.Target && pipeline.DepthWrite &&
        m_Impl->CurrentPass.Type != PassType::Shadow && m_Impl->CurrentPass.Type != PassType::EnvironmentCapture &&
        m_Impl->CurrentPass.Type != PassType::PlanarCapture)
        return m_Impl->Fail("OpenGL ES contact receivers must preserve traced depth");
    if (m_Impl->ReflectionResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
        return m_Impl->Fail("OpenGL ES opaque depth is read-only after SSR");
    if (m_Impl->FogResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
        return m_Impl->Fail("OpenGL ES opaque depth is read-only after fog composition");
    if (m_Impl->LightShaftResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
        return m_Impl->Fail("OpenGL ES opaque depth is read-only after light shafts");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->LightShaftActive &&
        pipeline.BlendEnabled != m_Impl->LightShaftResolved)
        return m_Impl->Fail("OpenGL ES opaque draws precede light shafts; transparent draws follow them");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->FogActive &&
        pipeline.BlendEnabled != m_Impl->FogResolved)
        return m_Impl->Fail("OpenGL ES opaque draws precede fog; transparent draws follow it");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->GIActive &&
        pipeline.BlendEnabled != m_Impl->GI.Resolved) return m_Impl->Fail("SSGI separates opaque and transparent material draws");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->ReflectionActive &&
        pipeline.BlendEnabled != m_Impl->ReflectionResolved)
        return m_Impl->Fail("OpenGL ES opaque draws precede SSR; transparent draws follow it");
    if (m_Impl->WeightedPass && (!pipeline.BlendEnabled || pipeline.DepthWrite || !pipeline.DepthTest))
        return m_Impl->Fail("OpenGL ES weighted transparency requires blended coverage and read-only opaque depth testing");
    if (m_Impl->WeightedActive && !m_Impl->WeightedPass && m_Impl->CurrentPass.Type == PassType::Color &&
        !m_Impl->CurrentPass.Target && pipeline.BlendEnabled) return m_Impl->Fail("OpenGL ES prepare weighted transparency before blended geometry");
    if (m_Impl->VelocityCaptured && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
        return m_Impl->Fail("OpenGL ES shared opaque depth is read-only after velocity capture");
    if (m_Impl->CurrentPass.Type == PassType::Velocity &&
        (!pipeline.DepthTest || pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite || pipeline.OutlineExtrusion))
        return m_Impl->Fail("OpenGL ES velocity requires opaque coverage and read-only depth testing");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->MotionBlurActive &&
        pipeline.BlendEnabled != m_Impl->MotionBlurResolved)
        return m_Impl->Fail("OpenGL ES opaque draws precede motion blur; transparent draws follow it");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->TemporalActive &&
        pipeline.BlendEnabled != m_Impl->TemporalResolved)
        return m_Impl->Fail("OpenGL ES opaque draws precede TAA; transparent draws follow it");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->DepthOfFieldActive)
    {
        if (m_Impl->DepthOfFieldResolved && (!pipeline.BlendEnabled || pipeline.DepthWrite))
            return m_Impl->Fail("OpenGL ES only transparent, depth-read-only draws may follow depth of field");
        if (!m_Impl->DepthOfFieldResolved && pipeline.BlendEnabled)
            return m_Impl->Fail("OpenGL ES transparent draws require depth of field first");
    }
    if ((m_Impl->CurrentPass.Type == PassType::EnvironmentCapture || m_Impl->CurrentPass.Type == PassType::PlanarCapture) &&
        (!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite ||
         pipeline.SamplesShadowMap || pipeline.OutlineExtrusion || pipeline.Shading != m_Impl->FrameReflectionShading))
        return m_Impl->Fail("OpenGL ES capture requires opaque depth and the frame shading model without shadow sampling");
    if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && !m_Impl->FramePlanar.Active())
        return m_Impl->Fail("OpenGL ES inactive planar capture must clear without drawing geometry");
    if (pipeline.OutlineThickness < 0.0f) return m_Impl->Fail("OpenGL ES outline thickness is invalid");
    if (m_Impl->CurrentPass.Type == PassType::GBuffer &&
        (pipeline.BlendEnabled || pipeline.Shading != m_Impl->FrameReflectionShading))
        return m_Impl->Fail("OpenGL ES G-buffer requires opaque materials and the frame shading model");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
        m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved && pipeline.BlendEnabled)
        return m_Impl->Fail("OpenGL ES transparency must follow ambient occlusion");
    if (m_Impl->CurrentPass.DepthPeelingLayer && !ValidateDepthPeelingPipeline(pipeline,m_Impl->LastError)) return false;
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
    if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && value.Values != m_Impl->FramePlanar.ClippedViewProjection.Values)
        return m_Impl->Fail("OpenGL ES planar geometry requires the frozen oblique capture projection");
    if (m_Impl->CurrentPass.DepthPeelingLayer && value.Values!=m_Impl->FramePeeling.ViewProjection.Values)
        return m_Impl->Fail("OIT capture and replay require the same raster projection");
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

bool OpenGLESRHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc& desc, GpuDrivenBatchHandle& handle)
{
    handle = {};
    if (!m_Impl->CheckThread("PrepareGpuDrivenBatch") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        desc.Instances.empty() || !desc.IndexCount)
        return m_Impl->Fail("OpenGLES GPU batch preparation requires an open frame outside a graphics pass");
    std::uint32_t index = 0;
    if (!m_Impl->PrepareGpuBatch(desc, index)) return false;
    handle = {m_Impl->GpuEpoch, index};
    ++m_Impl->PendingFrame.Statistics.GpuCullDispatchCount;
    return true;
}

bool OpenGLESRHI::RenderVisibilityBuffer(const VisibilityBufferDesc& desc)
{
    if (desc.MeshShaders) return m_Impl->Fail("OpenGLES does not expose native mesh shader stages");
    if (!m_Impl->CheckThread("RenderVisibilityBuffer") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->DeferredActive || m_Impl->GBufferWritten || m_Impl->FrameSampleCount != 1)
        return m_Impl->Fail("OpenGLES visibility geometry requires an unresolved single-sample deferred frame");
    if (!ValidateVisibilityFrame(desc, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->LastError) ||
        !m_Impl->PrepareVisibilityBuffer(desc.Scene)) return false;
        if (desc.NaniteRaster && !m_Impl->PrepareNaniteFrameRaster(desc)) return false;
        if (!m_Impl->PrepareNaniteHardware(desc)) return false;
        if (!m_Impl->PrepareNaniteInstances(desc)) return false;
        if (!m_Impl->PrepareNaniteMaterials(desc)) return false;
    PassDesc pass; pass.Type = PassType::GBuffer; pass.ClearColor = desc.Background;
    if (!BeginPass(pass) || !m_Impl->DrawVisibilityBuffer(desc)) return false;
    RecordVisibilityDraws(m_Impl->PendingFrame, desc);
    return EndPass();
}

bool OpenGLESRHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle handle)
{
    if (!m_Impl->CheckThread("DrawGpuDrivenBatch") || !m_Impl->PassOpen || !handle.Frame ||
        handle.Frame != m_Impl->GpuEpoch || handle.Index >= m_Impl->GpuCursor || m_Impl->ActiveGpuBatch)
        return m_Impl->Fail("OpenGLES indirect draw has an invalid frame token or pass state");
    auto& batch = m_Impl->GpuBatches[handle.Index];
    const auto& desc = batch.Description;
    if (!GpuDrivenDrawMatches(desc, m_Impl->BoundWorld, m_Impl->BoundViewProjection, m_Impl->BoundPipeline, m_Impl->CurrentPass))
        return m_Impl->Fail("OpenGLES indirect draw does not match its prepared instance data");
    m_Impl->ActiveGpuBatch = &batch;
    struct Reset { Impl* State; ~Reset() { State->ActiveGpuBatch = nullptr; } } reset {m_Impl.get()};
    return DrawIndexedGeometry(desc.FirstIndex, desc.IndexCount, desc.BaseVertex, &desc.Instances);
}

bool OpenGLESRHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex)
{
    return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, nullptr);
}

bool OpenGLESRHI::DrawIndexedInstanced(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
    const std::vector<DrawInstance>& instances)
{
    return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, &instances);
}

bool OpenGLESRHI::DrawIndexedGeometry(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
    const std::vector<DrawInstance>* instances)
{
    if (m_Impl->PassOpen && m_Impl->Hair.Active && HairFurReplayPass(m_Impl->CurrentPass.Type) &&
        m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Hair replay ends the pass geometry stream");
    if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound ||
        !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound ||
        !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
        return m_Impl->Fail("OpenGL ES DrawIndexed is missing required state");
    if (instances && m_Impl->CurrentPass.DepthPeelingLayer) return m_Impl->Fail("OIT replay requires ordinary invariant draws");
    if (instances && UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        return m_Impl->Fail("POM material capture currently requires individual mesh draws");
    std::optional<InstanceConstantBlock> instanceConstants;
    if (instances)
    {
        instanceConstants.emplace();
        if (!GetCapabilities().InstancedRendering || m_Impl->CurrentPass.Type == PassType::Outline ||
            m_Impl->BoundPipeline.OutlineExtrusion || !BuildInstanceConstants(*instances, m_Impl->BoundWorld, *instanceConstants))
            return m_Impl->Fail("OpenGL ES instancing requires a supported uniform block and a bounded batch with finite transforms and common winding");
        instanceConstants->ViewProjection = m_Impl->BoundViewProjection;
        instanceConstants->Material.X = m_Impl->BoundPipeline.Color.W * m_Impl->BoundPipeline.MaterialOpacity;
    }
    auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
    auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
    if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end())
        return m_Impl->Fail("OpenGL ES DrawIndexed requires complete triangle-list buffers");
    if (!ValidateIndexedGeometryDraw(vertex->second.Description,index->second.Description,index->second.Bytes,
        vertex->second.NaniteGeometry && index->second.NaniteGeometry,firstIndex,indexCount,baseVertex,m_Impl->LastError)) return false;

    const PipelineState& pipeline = m_Impl->BoundPipeline;
    if (m_Impl->LightListsActive && (m_Impl->CurrentPass.Type == PassType::ContactDepth || m_Impl->CurrentPass.Type == PassType::GBuffer ||
        m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) &&
        !m_Impl->LightListFrame.ValidateDraw(pipeline.Lighting, m_Impl->BoundViewProjection, true, m_Impl->LastError)) return false;
    if (m_Impl->CurrentPass.Type == PassType::Shadow)
    {
        if (!m_Impl->SelectShadowProgram()) return false;
    }
    else if (UsesParallaxMaterial(pipeline,m_Impl->CurrentPass.Type))
    {
        if (!m_Impl->EnsureParallax()) return false;
        m_Impl->Program=m_Impl->ParallaxProgram; m_Impl->QueryProgramUniforms();
    }
    else if (m_Impl->CurrentPass.Type == PassType::GBuffer)
    {
        const auto material=SelectMaterialTechnique(pipeline);
        const auto program=m_Impl->MaterialProgram(material,unsigned(pipeline.Shading),false,true);
        if (!program) return false;
        if (m_Impl->Program!=program) { m_Impl->Program=program; m_Impl->QueryProgramUniforms(); }
    }
    else if (m_Impl->CurrentPass.Type == PassType::Velocity)
    {
        if (m_Impl->Program != m_Impl->MotionGeometryProgram)
        {
            m_Impl->Program = m_Impl->MotionGeometryProgram;
            m_Impl->QueryProgramUniforms();
        }
    }
    else if (m_Impl->CurrentPass.Type == PassType::ContactDepth)
    {
        if (!m_Impl->ContactDepthProgram)
            m_Impl->ContactDepthProgram = m_Impl->CompileStageProgram("ContactShadows.glsl", "#define EVANS_CONTACT_DEPTH 1\n");
        if (!m_Impl->ContactDepthProgram) return false;
        m_Impl->Program = m_Impl->ContactDepthProgram; m_Impl->QueryProgramUniforms();
        m_Impl->GlUseProgram(m_Impl->Program);
        m_Impl->GlUniform1i(m_Impl->DiffuseMapLocation, 1); m_Impl->GlUniform1i(m_Impl->OpacityMapLocation, 3);
    }
    else if (m_Impl->UsesForwardPlusMaterial() || m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth)
    {
        if (!m_Impl->SelectForwardPlusProgram()) return false;
    }
    else if (m_Impl->UsesMaterialSurface(pipeline,m_Impl->CurrentPass.Type))
    {
        const auto material=SelectMaterialTechnique(pipeline,m_Impl->WeightedPass);
        const auto program=m_Impl->MaterialProgram(material,unsigned(pipeline.Shading),m_Impl->WeightedPass,false);
        if (!program) return false;
        if (m_Impl->Program!=program) { m_Impl->Program=program; m_Impl->QueryProgramUniforms(); }
    }
    else if (m_Impl->UsesSurfacePCF())
    {
        const auto program=m_Impl->PCFProgram(unsigned(pipeline.Shading),m_Impl->WeightedPass);
        if (!program) return false;
        if (m_Impl->Program!=program) { m_Impl->Program=program; m_Impl->QueryProgramUniforms(); }
    }
    else if (m_Impl->FrameRayShadowImage && !m_Impl->FrameAreaLights && m_Impl->CurrentPass.Type==PassType::Color && pipeline.SamplesShadowMap)
    {
        const auto program=m_Impl->RayShadowProgram(static_cast<unsigned>(pipeline.Shading),m_Impl->WeightedPass);
        if (!program) return false;
        if (m_Impl->Program!=program) { m_Impl->Program=program; m_Impl->QueryProgramUniforms(); }
    }
    else if (!m_Impl->CreateProgram(pipeline.Shading, m_Impl->WeightedPass, m_Impl->UsesSurfacePCSS(), m_Impl->UsesSurfaceVSM(), m_Impl->UsesContactReceiver())) return false;
    if (instances && !m_Impl->SelectInstanceProgram()) return false;
    m_Impl->GlUseProgram(m_Impl->Program);
    // Every selected consumer owns separate sampler uniforms. Rebind the common
    // material contract after program selection, including lazy PCF/ray stages.
    m_Impl->GlUniform1i(m_Impl->ShadowMapLocation,0); m_Impl->GlUniform1i(m_Impl->DiffuseMapLocation,1);
    m_Impl->GlUniform1i(m_Impl->NormalMapLocation,2); m_Impl->GlUniform1i(m_Impl->OpacityMapLocation,3);
    m_Impl->UploadMatrices();

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
        (bool(m_Impl->FrameRayShadowImage) || shadowAvailable || (m_Impl->FrameVirtualShadowAtlas && m_Impl->BoundPipeline.SamplesShadowMap && m_Impl->CurrentPass.Type==PassType::Color)),
        surfaceTextures[0] != 0,
        surfaceTextures[1] != 0,
        surfaceTextures[2] != 0,
        m_Impl->HdrActive && !m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type != PassType::Shadow);
    m_Impl->UploadFloat4(m_Impl->PipelineColorLocation, constants.PipelineColor);
    m_Impl->UploadFloat4(m_Impl->ShadingParametersLocation, constants.ShadingParameters);
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
        static_cast<float>(constants.TextureFlags.W));
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
    if (m_Impl->UsesSurfaceVSM())
    {
        if (!m_Impl->UploadVSMReceiver(m_Impl->Program, m_Impl->BoundLightViewProjection, pipeline.Lighting, shadowAvailable)) return false;
        if (shadowAvailable && !m_Impl->MomentTexture(m_Impl->LastShadowTexture, shadowTexture)) return false;
    }
    m_Impl->GlActiveTexture(GL_TEXTURE0);
    m_Impl->GlBindTexture(GL_TEXTURE_2D, shadowTexture);
    for (std::size_t textureIndex = 0; textureIndex < surfaceTextures.size(); ++textureIndex)
    {
        m_Impl->GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(textureIndex + 1));
        m_Impl->GlBindTexture(GL_TEXTURE_2D, surfaceTextures[textureIndex]);
    }

    if (!m_Impl->BindCascades(m_Impl->Program,
        m_Impl->CurrentPass.Type == PassType::Color && pipeline.SamplesShadowMap)) return false;
    if (m_Impl->UsesSurfacePCSS() && !m_Impl->UploadPCSS(m_Impl->Program, m_Impl->BoundLightViewProjection,
        pipeline.Lighting, shadowAvailable)) return false;
    if (m_Impl->UsesVSM() && m_Impl->CurrentPass.Type == PassType::Shadow)
    {
        auto moments = BuildVSMParameters(m_Impl->FrameShadowFilter.VSM);
        if (!BuildVSMEncoding(m_Impl->BoundViewProjection, moments.DepthEncoding[0], m_Impl->LastError) ||
            !m_Impl->UploadVSM(m_Impl->Program, moments)) return false;
    }
    const bool surfaceFog = m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
        pipeline.BlendEnabled && m_Impl->FogActive && m_Impl->FogResolved;
    const VolumetricFogConstants disabledFog {};
    m_Impl->UploadFogConstants(m_Impl->Program, surfaceFog ? m_Impl->FogConstants : disabledFog);
    if (m_Impl->UsesSurfacePCF() || m_Impl->UsesMaterialSurface(pipeline,m_Impl->CurrentPass.Type))
        m_Impl->BindPCF(m_Impl->Program,shadowAvailable);
    m_Impl->BindRayShadowImage(m_Impl->Program);
    m_Impl->BindLumenTranslucency(m_Impl->Program,m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target && pipeline.BlendEnabled);
    if (m_Impl->WeightedPass)
        m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(m_Impl->Program, "TransparencyWeights"),
            BuildWeightedTransparencyConstants(m_Impl->FrameTransparency).Weights);
    m_Impl->GlActiveTexture(GL_TEXTURE0 + Impl::FogTextureUnit);
    m_Impl->GlBindTexture(GL_TEXTURE_2D, surfaceFog ? m_Impl->FogTexture : 0);
    if (m_Impl->UsesForwardPlusMaterial())
    {
        if (!m_Impl->BindForwardPlusSurface()) return false;
    }
    else if (m_Impl->UsesContactReceiver() && !m_Impl->BindContactReceiver(m_Impl->Program)) return false;
    m_Impl->GlActiveTexture(GL_TEXTURE0);

    if (m_Impl->CurrentPass.Type == PassType::Velocity)
    {
        MotionBlurConstants motion{};
        if (!BuildMotionGeometryConstants(pipeline, m_Impl->BoundWorld, m_Impl->BoundViewProjection,
            GetMotionTransform(), m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
            surfaceTextures[0] != 0, surfaceTextures[2] != 0, motion))
            return m_Impl->Fail("OpenGL ES velocity requires finite current and previous transforms");
        if (instances) motion.Options.W = 1; // Each invalid object has a zero previous clip matrix.
        const GLuint program = m_Impl->Program;
        m_Impl->UploadMotionConstants(program, motion);
        m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "DiffuseAlpha"), 1);
        m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "OpacityMap"), 3);
    }

    if (pipeline.DepthTest)
    {
        m_Impl->GlEnable(GL_DEPTH_TEST);
        m_Impl->GlDepthFunc(m_Impl->CurrentPass.DepthPeelingLayer ? GL_EQUAL : pipeline.DepthWrite ? GL_LESS : GL_LEQUAL);
    }
    else
    {
        m_Impl->GlDisable(GL_DEPTH_TEST);
    }
    m_Impl->GlDepthMask(pipeline.DepthWrite ? GL_TRUE_VALUE : GL_FALSE_VALUE);
    const GLboolean colorWrite = pipeline.ColorWrite || (m_Impl->UsesVSM() && m_Impl->CurrentPass.Type == PassType::Shadow) ? GL_TRUE_VALUE : GL_FALSE_VALUE;
    m_Impl->GlColorMask(colorWrite, colorWrite, colorWrite, colorWrite);
    if (m_Impl->CurrentPass.Type == PassType::Color && pipeline.BlendEnabled)
    {
        m_Impl->GlEnable(GL_BLEND);
        m_Impl->GlBlendFuncSeparate(
            GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_ONE,
            GL_ONE_MINUS_SRC_ALPHA);
        if (m_Impl->WeightedPass)
        {
            m_Impl->GlBlendEquation(GL_FUNC_ADD);
            m_Impl->GlBlendFuncSeparatei(0, GL_ONE, GL_ONE, GL_ONE, GL_ONE);
            m_Impl->GlBlendFuncSeparatei(1, GL_ZERO, GL_ONE_MINUS_SRC_COLOR, GL_ZERO, GL_ONE);
        }
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
    // Cube capture flips Y for native cube T; planar capture reflects world
    // space instead. Both reversals combine with the object's determinant.
    const bool captureFlipsWinding = m_Impl->CurrentPass.Type == PassType::PlanarCapture ||
        (m_Impl->CurrentPass.Type == PassType::EnvironmentCapture && EnvironmentReversesWinding(RHIBackend::OpenGLES));
    m_Impl->GlFrontFace(RealtimeWorldIsMirrored(m_Impl->BoundWorld) != captureFlipsWinding ? GL_CW : GL_CCW);
    if (pipeline.OutlineExtrusion || pipeline.BackFaceCulling)
    {
        m_Impl->GlEnable(GL_CULL_FACE);
        m_Impl->GlCullFace(pipeline.OutlineExtrusion ? GL_FRONT : GL_BACK);
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

    if (instances && !m_Impl->UploadInstances(*instanceConstants)) return false;
    if (UsesParallaxMaterial(pipeline,m_Impl->CurrentPass.Type) && !m_Impl->BindParallaxDraw(constants)) return false;
    const auto drawElements = [&](GLenum primitive, GLsizei count, const void* offset)
    {
        const bool perSample = m_Impl->UsesContactReceiver() && m_Impl->ContactReceiverActive && m_Impl->FrameSampleCount > 1;
        const GLuint repetitions = perSample ? m_Impl->FrameSampleCount : 1;
        const GLint sampleLocation = perSample ? m_Impl->GlGetUniformLocation(m_Impl->Program, "ContactSample") : -1;
        if (perSample) m_Impl->GlEnable(GL_SAMPLE_MASK);
        // ES 3.1 core: shade each visibility layer into its matching coverage bit.
        // No sample-shading extension or premature mask resolve is required.
        for (GLuint sample = 0; sample < repetitions; ++sample)
        {
            if (perSample) { m_Impl->GlSampleMaski(0, 1u << sample); m_Impl->GlUniform1i(sampleLocation, static_cast<GLint>(sample)); }
            if (m_Impl->ActiveGpuBatch)
            {
                m_Impl->GlBindBuffer(Impl::GpuDrawIndirectBuffer, m_Impl->ActiveGpuBatch->Arguments);
                m_Impl->GpuDrawElementsIndirect(primitive, GL_UNSIGNED_INT, nullptr);
                m_Impl->GlBindBuffer(Impl::GpuDrawIndirectBuffer, 0);
            }
            else if (m_Impl->ActiveNaniteDraw)
            {
                constexpr GLenum DrawIndirectBinding=0x8F43;
                GLint previous=0; m_Impl->GlGetIntegerv(DrawIndirectBinding,&previous);
                m_Impl->GlBindBuffer(Impl::GpuDrawIndirectBuffer,m_Impl->ActiveNaniteDraw->Arguments->Buffer);
                const auto address=std::uintptr_t(m_Impl->NaniteArgumentOffset());
                m_Impl->GpuDrawElementsIndirect(primitive,GL_UNSIGNED_INT,reinterpret_cast<const void*>(address));
                m_Impl->GlBindBuffer(Impl::GpuDrawIndirectBuffer,GLuint(previous));
            }
            else if (instances) m_Impl->GlDrawElementsInstanced(primitive, count, GL_UNSIGNED_INT, offset,
                static_cast<GLsizei>(instances->size()));
            else m_Impl->GlDrawElements(primitive, count, GL_UNSIGNED_INT, offset);
        }
        if (perSample) { m_Impl->GlDisable(GL_SAMPLE_MASK); m_Impl->GlSampleMaski(0, std::numeric_limits<GLuint>::max()); }
    };
    const bool drawLines = pipeline.Fill == FillMode::Wireframe;
    if (drawLines && (pipeline.BackFaceCulling || pipeline.OutlineExtrusion))
    {
        if (m_Impl->ActiveNaniteDraw) return m_Impl->Fail("Nanite indirect wireframe is outside the solid-only draw contract");
        if (!m_Impl->PrepareWireframe(vertex->second,index->second,firstIndex,indexCount,baseVertex)) return false;
        if (!m_Impl->BindWireFace(RealtimeWorldIsMirrored(m_Impl->BoundWorld)!=captureFlipsWinding,pipeline.OutlineExtrusion)) return false;
        m_Impl->GlLineWidth(1.0f);
        drawElements(GL_LINES,static_cast<GLsizei>(indexCount*2),nullptr);
        m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(m_Impl->Program,"WireFaceCull"),0);
        m_Impl->WireDisableAttribute(15);
    }
    else if (drawLines && index->second.NaniteGeometry)
    {
        if (!m_Impl->PrepareNaniteWireIndices(index->second.Buffer,firstIndex,indexCount)) return false;
        m_Impl->GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_Impl->ScratchIndexBuffer); m_Impl->GlLineWidth(1.0f);
        drawElements(GL_LINES,static_cast<GLsizei>(indexCount*2),nullptr);
    }
    else if (drawLines || baseVertex != 0)
    {
        const auto* sourceIndices=reinterpret_cast<const std::uint32_t*>(index->second.Bytes.data());
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
        drawElements(drawLines ? GL_LINES : GL_TRIANGLES, static_cast<GLsizei>(scratch.size()), nullptr);
    }
    else
    {
        m_Impl->GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_Impl->ActiveNaniteDraw && m_Impl->ActiveNaniteDraw->Binning?m_Impl->ActiveNaniteDraw->Arguments->Buffer:index->second.Buffer);
        drawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount),
            reinterpret_cast<const void*>(static_cast<std::size_t>(firstIndex) * sizeof(std::uint32_t)));
    }
    if (!m_Impl->CheckGL(instances ? "glDrawElementsInstanced" : "glDrawElements")) return false;

    if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type)) ++m_Impl->PendingFrame.Statistics.ParallaxMaterialDrawCount;
    DrawCallRecord record;
    record.VertexBuffer = m_Impl->BoundVertexBuffer;
    record.IndexBuffer = m_Impl->BoundIndexBuffer;
    record.Pipeline = pipeline;
    record.WorldTransform = m_Impl->BoundWorld;
    record.Motion = GetMotionTransform();
    record.FirstIndex = firstIndex;
    record.IndexCount = indexCount;
    record.BaseVertex = baseVertex;
    if (instances) record.Instances = *instances;
    record.GPUDriven = m_Impl->ActiveGpuBatch != nullptr;
        record.NaniteIndirect=m_Impl->ActiveNaniteDraw!=nullptr; record.NaniteQueryIndex=record.NaniteIndirect?m_Impl->ActiveNaniteQuery:0;
    m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(std::move(record));
    FrameStatistics& statistics = m_Impl->PendingFrame.Statistics;
    ++statistics.DrawCallCount;
    statistics.InstancedDrawCallCount += instances ? 1u : 0u;
    if (m_Impl->ActiveGpuBatch)
    {
        ++statistics.IndirectDrawCallCount;
        statistics.GpuCandidateInstanceCount += instances->size();
        statistics.GpuCandidateTriangleCount += std::uint64_t(indexCount / 3) * instances->size();
    }
    else if (m_Impl->ActiveNaniteDraw)
        {
            ++statistics.IndirectDrawCallCount; ++statistics.NaniteIndirectDrawCount;
            statistics.NaniteCandidateTriangles+=indexCount/3;
        }
        else statistics.SubmittedTriangleCount += std::uint64_t(indexCount / 3) * (instances ? instances->size() : 1);
    statistics.ShadowDrawCallCount += m_Impl->CurrentPass.Type == PassType::Shadow ? 1U : 0U;
    statistics.ContactDepthDrawCallCount += m_Impl->CurrentPass.Type == PassType::ContactDepth ? 1U : 0U;
    statistics.ForwardPlusDepthDrawCallCount += m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth ? 1U : 0U;
    statistics.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1U : 0U;
    return true;
}

bool OpenGLESRHI::EndPass()
{
    if (m_Impl->FrameOpen && m_Impl->PassOpen && m_Impl->Hair.Active && !m_Impl->Hair.Frame.Geometry->Indices.empty() &&
        HairFurReplayPass(m_Impl->CurrentPass.Type) && (m_Impl->CurrentPass.Type!=PassType::PlanarCapture || m_Impl->FramePlanar.Active()) &&
        m_Impl->Hair.ReplayedPass!=m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Complete requested hair replay before ending its native geometry pass");
    if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
        return m_Impl->Fail("OpenGL ES has no open pass");
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    m_Impl->GlDepthMask(GL_TRUE_VALUE);
    if (m_Impl->CurrentPass.Type == PassType::Shadow && m_Impl->UsesVSM() &&
        !m_Impl->FilterMoments(m_Impl->CurrentPass.Target)) return false;
    if (m_Impl->CurrentPass.Type == PassType::EnvironmentCapture)
    {
        m_Impl->EnvironmentFacesWritten |= 1u << m_Impl->CurrentPass.EnvironmentFaceIndex;
        ++m_Impl->PendingFrame.Statistics.EnvironmentCaptureFaceCount;
    }
    if (m_Impl->CurrentPass.Type == PassType::GBuffer) m_Impl->GBufferWritten = true;
    if (m_Impl->CurrentPass.Type == PassType::ContactDepth) m_Impl->ContactDepthWritten = true;
    if (m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) m_Impl->ForwardPlusDepthWritten = true;
    if (m_Impl->WeightedPass) m_Impl->WeightedAccumulated = true;
    if (m_Impl->CurrentPass.Type == PassType::PlanarCapture)
    {
        m_Impl->PlanarWritten = true; ++m_Impl->PendingFrame.Statistics.PlanarCapturePassCount;
    }
    if (m_Impl->CurrentPass.Type == PassType::Velocity) m_Impl->VelocityCaptured = true;
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target)
    {
        if (!m_Impl->DeferredActive) m_Impl->ForwardColorWritten = true;
        if (m_Impl->FrameSampleCount > 1 && m_Impl->SamplesOpaqueResolved)
        {
            m_Impl->SamplesTransparentWritten = true;
            m_Impl->SamplesColorResolved = false;
        }
    }
    if (m_Impl->CurrentPass.DepthPeelingLayer && !m_Impl->PeelingSequence.Advance(DepthPeelingEvent::EndShading,m_Impl->LastError)) return false;
    m_Impl->PassOpen = false;
    return m_Impl->CheckGL("EndPass");
}

bool OpenGLESRHI::PrepareLightListFrame(const LightListFrameDesc& desc)
{
    if (!m_Impl->CheckThread("PrepareLightListFrame") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        desc.Layout != (m_Impl->ClusteredActive ? LightListLayout::Clusters : LightListLayout::Tiles) || !m_Impl->LightListsActive || !m_Impl->PendingFrame.Passes.empty())
        return m_Impl->Fail("OpenGL ES Forward+ light storage must precede every frame pass");
    if (!m_Impl->LightListFrame.Prepare(desc, m_Impl->LastError)) return false;
    if (m_Impl->ClusteredActive) m_Impl->ClusteredForward.BeginFrame(); else m_Impl->TileProducer().BeginFrame();
    return m_Impl->UploadForwardPlusScene();
}

bool OpenGLESRHI::GenerateLightLists()
{
    if (!m_Impl->CheckThread("GenerateLightLists") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->LightListsActive || !m_Impl->LightListFrame.Prepared() || m_Impl->LightListFrame.Assigned() ||
        (!m_Impl->ClusteredActive && !m_Impl->ForwardPlusDepthWritten && !m_Impl->GBufferWritten) || m_Impl->ForwardColorWritten || m_Impl->ContactGenerated)
        return m_Impl->Fail("OpenGL ES Forward+ assignment requires completed opaque depth, once before visibility and lighting");
    m_Impl->UnbindSampledTextures(); m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
    const auto& frame = m_Impl->LightListFrame.Description();
    const bool multisample = m_Impl->FrameSampleCount > 1;
    if (m_Impl->ClusteredActive)
    {
        if (!m_Impl->ClusteredForward.Dispatch(m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
            frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError) ||
            !m_Impl->LightListFrame.CompleteAssignment(m_Impl->ClusteredForward.Grid(), m_Impl->LastError)) return false;
        ++m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount;
    }
    else
    {
        if (!m_Impl->TileProducer().Dispatch(multisample ? m_Impl->Multisample.Depth : m_Impl->DepthTexture,
            multisample, frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError) ||
            !m_Impl->LightListFrame.CompleteAssignment(m_Impl->TileProducer().Grid(), m_Impl->LastError)) return false;
        ++(m_Impl->TiledActive ? m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount : m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount);
    }
    return m_Impl->CheckGL("Generate Forward+ light lists");
}

bool OpenGLESRHI::GenerateContactShadows(const ContactShadowDesc& desc)
{
    if (!m_Impl->CheckThread("GenerateContactShadows") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->FrameContact.Enabled || m_Impl->ContactGenerated || m_Impl->ForwardColorWritten || m_Impl->DeferredResolved ||
        (!m_Impl->ContactDepthWritten && !m_Impl->GBufferWritten && !m_Impl->ForwardPlusDepthWritten))
        return m_Impl->Fail("OpenGL ES contact visibility requires completed opaque depth, once before lighting");
    if (m_Impl->LightListsActive)
    {
        if (!m_Impl->LightListFrame.Assigned())
            return m_Impl->Fail("OpenGLES Forward+ contact visibility must follow light assignment");
        if (!m_Impl->LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, m_Impl->LastError)) return false;
    }

    m_Impl->UnbindSampledTextures(); m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
    auto lighting = desc.Lighting;
    if (lighting.SceneLights) lighting.SceneLights->WriteLegacySubset(lighting);
    if (!m_Impl->Hair.ValidateContactProducer(desc.ViewProjection,lighting,m_Impl->LastError)) return false;
    const auto light = lighting.ShadowLightIndex;
    m_Impl->ContactReceiverActive = light >= 0 && light < int(std::min(lighting.LightCount, MaximumSceneLights)) &&
        lighting.Lights[light].CastShadows;
    if (m_Impl->ContactReceiverActive)
    {
        ContactShadowDispatch trace; trace.ViewProjection = desc.ViewProjection; trace.Light = lighting.Lights[light];
        trace.MaximumDistance = m_Impl->FrameContact.MaximumDistance;
        trace.Parameters.StartBias = m_Impl->FrameContact.StartBias; trace.Parameters.Thickness = m_Impl->FrameContact.Thickness;
        trace.Parameters.MaximumSteps = m_Impl->FrameContact.MaximumSteps;
        const bool multisample = m_Impl->FrameSampleCount > 1;
        if (!m_Impl->ContactShadows.Dispatch(multisample ? m_Impl->Multisample.Depth : m_Impl->DepthTexture,
            multisample, trace, m_Impl->LastError)) return false;
        ++m_Impl->PendingFrame.Statistics.ContactShadowPassCount;
    }
    m_Impl->Hair.RecordContact(desc.ViewProjection,lighting,m_Impl->ContactReceiverActive);
    m_Impl->ContactGenerated = true;
    return m_Impl->CheckGL("Generate contact visibility");
}

bool OpenGLESRHI::DrawHairFurVelocity(const HairFurVelocityDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFurVelocity") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->CurrentPass.Type!=PassType::Velocity ||
        !m_Impl->Hair.Active || !m_Impl->Hair.Resolved || !m_Impl->Hair.VelocityRequested || m_Impl->Hair.VelocityResolved ||
        m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size() || !m_Impl->Hair.MatchesPrimaryView(desc.View))
        return m_Impl->Fail("Fiber velocity requires matching completed primary geometry, once at the read-only velocity pass tail");
    if (m_Impl->ViewProjectionBound && desc.View.ViewProjection.Values!=m_Impl->BoundViewProjection.Values)
        return m_Impl->Fail("Fiber velocity projection must match the current mesh velocity pass");
    HairFurConstants constants;
    if (!BuildHairFurVelocityConstants(desc,m_Impl->Hair,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,constants,m_Impl->LastError) ||
        !m_Impl->ReplayHairFur(constants,HairFurRasterPhase::Velocity)) return false;
    m_Impl->Hair.ReplayedPass=m_Impl->PendingFrame.Passes.size(); m_Impl->Hair.VelocityResolved=true;
    ++m_Impl->PendingFrame.Statistics.HairFurVelocityDrawCount; return true;
}

bool OpenGLESRHI::DrawHairFurInPass(const HairFurDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFurInPass") || !m_Impl->FrameOpen || !m_Impl->PassOpen || !m_Impl->Hair.Active || m_Impl->Hair.Resolved ||
        !HairFurReplayPass(m_Impl->CurrentPass.Type) || m_Impl->CurrentPass.Type==PassType::Velocity || m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size())
        return m_Impl->Fail("Hair replay requires an open depth, shadow or capture pass, once at its geometry tail");
    const auto type=m_Impl->CurrentPass.Type;
    if (type==PassType::PlanarCapture && !m_Impl->FramePlanar.Active()) return m_Impl->Fail("Inactive planar captures must remain empty");
    if (HairFurCameraDepthPass(type) && !m_Impl->Hair.MatchesPrimaryView(desc)) return m_Impl->Fail("Hair camera depth views must match");
    if ((HairFurCameraDepthPass(type) || type==PassType::Shadow) && m_Impl->ViewProjectionBound && desc.ViewProjection.Values!=m_Impl->BoundViewProjection.Values)
        return m_Impl->Fail("Hair replay projection must match the current mesh pass");
    if (type==PassType::EnvironmentCapture || type==PassType::PlanarCapture)
    {
        auto expected=desc;
        if (type==PassType::EnvironmentCapture)
        {
            const auto& face=m_Impl->FrameEnvironment.Faces[m_Impl->CurrentPass.EnvironmentFaceIndex];
            expected.ViewProjection=EnvironmentRasterProjection(face,GetBackend()); expected.Lighting.CameraPosition=m_Impl->FrameEnvironment.Position;
            expected.Forward=face.Forward; expected.Orthographic=false; expected.ClipPlane={0,0,0,0};
        }
        else
        {
            expected.ViewProjection=m_Impl->FramePlanar.ClippedViewProjection; expected.Lighting.CameraPosition=m_Impl->FramePlanar.Position;
            expected.ClipPlane=m_Impl->FramePlanar.ClipPlane;
        }
        if (!SameHairFurRasterView(desc,expected)) return m_Impl->Fail("Hair captures require the frozen reflection camera and clip plane");
    }
    const auto phase=HairFurReplayPhase(type,m_Impl->FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM); HairFurConstants constants;
    if (!BuildHairFurReplayConstants(desc,m_Impl->Hair.Frame,phase,m_Impl->FrameShadowFilter.VSM,constants,m_Impl->LastError) || !m_Impl->ReplayHairFur(constants,phase)) return false;
    m_Impl->Hair.ReplayedPass=m_Impl->PendingFrame.Passes.size();
    if (type==PassType::Shadow) m_Impl->Hair.RecordShadowView(m_Impl->CurrentPass.Target,desc.ViewProjection);
    if (HairFurCameraDepthPass(type)) m_Impl->Hair.RecordPrimaryView(desc);
    if (!m_Impl->Hair.Frame.Geometry->Indices.empty())
    {
        if (type==PassType::Shadow) ++m_Impl->PendingFrame.Statistics.HairFurShadowDrawCount;
        else if (HairFurCameraDepthPass(type)) ++m_Impl->PendingFrame.Statistics.HairFurDepthDrawCount;
        else ++m_Impl->PendingFrame.Statistics.HairFurCaptureDrawCount;
    }
    return true;
}

bool OpenGLESRHI::DrawHairFur(const HairFurDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFur") || !m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->Hair.Active || m_Impl->Hair.Resolved || !m_Impl->GBufferWritten ||
        !(m_Impl->DeferredActive?m_Impl->DeferredResolved:m_Impl->ForwardColorWritten) || m_Impl->SamplesOpaqueResolved || m_Impl->VelocityCaptured ||
        !HairFurPrecedesLaterPasses(m_Impl->PendingFrame))
        return m_Impl->Fail("Hair requires completed opaque lighting, once before MSAA resolution and screen-space effects");
    if (!m_Impl->Hair.MatchesPrimaryView(desc)) return m_Impl->Fail("Hair camera depth and primary surface require identical frozen raster inputs");
    HairFurConstants constants;
    if (!BuildHairFurConstants(desc.ViewProjection,desc.Lighting,m_Impl->Hair.Frame.Settings,*m_Impl->Hair.Frame.Lights,
        desc.Orthographic,desc.Forward,desc.ClipPlane,constants)) return m_Impl->Fail("Hair camera and lighting constants must be finite");
    if (!m_Impl->PrepareHairShadowReceiver(desc,constants) || !m_Impl->RenderHairFur(constants)) return false;
    m_Impl->Hair.RecordPrimaryView(desc); m_Impl->Hair.Resolved=true;
    if (!m_Impl->Hair.Frame.Geometry->Indices.empty()) ++m_Impl->PendingFrame.Statistics.HairFurPassCount;
    return true;
}

bool OpenGLESRHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc& desc)
{
    if (!m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before subsurface diffusion");
    if (!m_Impl->CheckThread("ApplySubsurfaceScattering") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->DeferredActive || !m_Impl->DeferredResolved || !m_Impl->SubsurfaceHasScene || m_Impl->SubsurfaceResolved || m_Impl->TiledActive ||
        m_Impl->PendingFrame.Statistics.SubsurfacePassCount ||
        (m_Impl->FrameSampleCount>1 && (!m_Impl->SamplesOpaqueResolved || m_Impl->SamplesColorRestored || m_Impl->SamplesTransparentWritten)))
        return m_Impl->Fail("Subsurface diffusion requires resolved opaque scene lighting, once before transparency");
    if (!SubsurfacePrecedesLaterPasses(m_Impl->PendingFrame))
        return m_Impl->Fail("Subsurface diffusion must precede AO, GI, reflections and lens/temporal effects");
    if (m_Impl->SubsurfaceRequested && !(Techniques::SanitizeSubsurfaceScatteringSettings(desc.Settings)==m_Impl->FrameSubsurface))
        return m_Impl->Fail("Subsurface controls must match the immutable frame settings");
    SubsurfaceScatteringConstants constants;
    if (!Techniques::UsesSubsurfaceScattering(desc.Settings) ||
        !BuildSubsurfaceScatteringConstants(m_Impl->SubsurfaceLighting.ViewProjection,desc.Settings,
            m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,Techniques::SubsurfacePass::Horizontal,constants))
        return m_Impl->Fail("Subsurface diffusion requires active finite profile and camera parameters");
    if (!m_Impl->SnapshotSubsurfaceScene()) return false;
    auto source=m_Impl->SubsurfaceLighting; source.Output=DeferredOutputMode::DirectDiffuse;
    m_Impl->SubsurfaceProducing=true;
    struct Reset { bool& Value; ~Reset() { Value=false; } } reset{m_Impl->SubsurfaceProducing};
    const bool produced=DrawDeferredLighting(source); m_Impl->SubsurfaceProducing=false;
    return produced && m_Impl->CaptureSubsurfaceSource() && m_Impl->FilterSubsurface(constants);
}

bool OpenGLESRHI::DrawDeferredLighting(const DeferredLightingDesc& desc)
{
    if (desc.Output == DeferredOutputMode::DirectDiffuse && m_Impl->TiledActive)
        return m_Impl->Fail("Direct-diffuse output requires ordinary deferred lighting");
    if (!m_Impl->CheckThread("DrawDeferredLighting")) return false;
    if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->DeferredActive ||
        !m_Impl->GBufferWritten || (m_Impl->DeferredResolved && !m_Impl->SubsurfaceProducing) || desc.Shading != GetTechniqueSettings().Shading)
        return m_Impl->Fail("OpenGL ES lighting requires one completed G-buffer with the frame shading model");
    if (m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
        return m_Impl->Fail("OpenGL ES deferred lighting requires completed contact visibility");
    const auto shadow = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
    const bool shadowAvailable = shadow != m_Impl->Textures.end() &&
        shadow->second.Description.Format == TextureFormat::Depth32;
    DeferredConstants constants;
    if (!BuildDeferredConstants(desc, (bool(m_Impl->FrameRayShadowImage) || shadowAvailable || bool(m_Impl->FrameVirtualShadowAtlas)), m_Impl->DeferredBackground, constants))
        return m_Impl->Fail("OpenGL ES deferred lighting requires a finite, invertible camera projection");
    const bool multisample = m_Impl->FrameSampleCount > 1;
    m_Impl->UnbindSampledTextures();
    m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, multisample ? m_Impl->Multisample.Framebuffers[Impl::SampleLightingFBO] : m_Impl->LightingFramebuffer);
    m_Impl->GlViewport(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    m_Impl->GlDisable(GL_DEPTH_TEST);
    m_Impl->GlDisable(GL_CULL_FACE);
    m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlDisable(GL_DITHER);
    m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    const GLuint program = m_Impl->TiledActive ? m_Impl->TiledLightingProgram(multisample) : m_Impl->UsesDedicatedPCF()?m_Impl->PCFProgram(multisample?6:5): m_Impl->FrameRayShadowImage ? m_Impl->RayShadowProgram(multisample?6:5) : m_Impl->FrameContact.Enabled ? m_Impl->ContactLightingProgram(multisample) :
        m_Impl->UsesVSM() ? m_Impl->VSMStageProgram(multisample ? 1 : 0) :
        m_Impl->UsesPCSS() ? m_Impl->PCSSScreenProgram(multisample ? 1 : 0) :
        multisample ? m_Impl->MultisampleLightingProgram : m_Impl->DeferredLightingProgram;
    if (!program) return false;
    m_Impl->GlUseProgram(program);
    m_Impl->BindRayShadowImage(program);
    if (m_Impl->UsesDedicatedPCF()) m_Impl->BindPCF(program,shadowAvailable && desc.SamplesShadowMap);
    if (m_Impl->FrameContact.Enabled && !m_Impl->BindContactReceiver(program)) return false;
    if (m_Impl->UsesPCSS() && !m_Impl->UploadPCSS(program, desc.LightViewProjection,
        desc.Lighting, shadowAvailable && desc.SamplesShadowMap)) return false;
    if (m_Impl->UsesVSM() && !m_Impl->UploadVSMReceiver(program, desc.LightViewProjection,
        desc.Lighting, shadowAvailable && desc.SamplesShadowMap)) return false;
    auto location = [&](const char* name) { return m_Impl->GlGetUniformLocation(program, name); };
    if (m_Impl->TiledActive)
    {
        LightListSurfaceConstants selection;
        if (!m_Impl->LightListFrame.Surface(desc.Lighting, desc.ViewProjection, LightListSelection::Opaque, selection, m_Impl->LastError)) return false;
        const auto& layout = selection.LightLayout; const auto& tile = selection.SurfaceTiles;
        m_Impl->GlUniform4ui(location("ForwardLightLayout"), layout.X, layout.Y, layout.Z, layout.W);
        m_Impl->GlUniform4ui(location("ForwardSurfaceTiles"), tile.X, tile.Y, tile.Z, tile.W);
        m_Impl->GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, ForwardPlusLightBinding, m_Impl->ForwardPlusSceneBuffer);
        m_Impl->GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, ForwardPlusHeaderBinding, m_Impl->TiledDeferred.Headers());
        m_Impl->GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, ForwardPlusIndexBinding, m_Impl->TiledDeferred.Indices());
        if (!m_Impl->FrameContact.Enabled)
        {
            m_Impl->GlUniform1i(location("ContactEnabled"), 0);
            m_Impl->GlUniform1i(location("ContactSample"), 0);
            m_Impl->GlUniform1i(location("ContactVisibility"), static_cast<GLint>(Impl::ContactTextureUnit));
        }
    }
    const auto inverse = ToColumnMajor(constants.InverseViewProjection);
    const auto lightProjection = ToColumnMajor(constants.LightViewProjection);
    m_Impl->GlUniformMatrix4fv(location("InverseViewProjection"), 1, GL_FALSE_VALUE, inverse.data());
    m_Impl->GlUniformMatrix4fv(location("LightViewProjection"), 1, GL_FALSE_VALUE, lightProjection.data());
    m_Impl->UploadFloat4(location("CameraLightCount"), constants.CameraLightCount);
    m_Impl->UploadFloat4(location("ShadingParameters"), constants.ShadingParameters);
    m_Impl->UploadFloat4(location("BackgroundColor"), constants.BackgroundColor);
    m_Impl->GlUniform4ui(location("LightingOptions"), constants.Options.X, constants.Options.Y,
        constants.Options.Z, constants.Options.W);
    for (std::size_t index = 0; index < constants.Lights.size(); ++index)
    {
        const auto& light = constants.Lights[index];
        const std::string prefix = "Lights[" + std::to_string(index) + "].";
        m_Impl->UploadFloat4(location((prefix + "PositionType").c_str()), light.PositionType);
        m_Impl->UploadFloat4(location((prefix + "DirectionIntensity").c_str()), light.DirectionIntensity);
        m_Impl->UploadFloat4(location((prefix + "ColorCastShadows").c_str()), light.ColorCastShadows);
        m_Impl->UploadFloat4(location((prefix + "AreaSizePadding").c_str()), light.AreaSizePadding);
    }
    const std::array<const char*, GBufferAttachmentCount> names {
        "DiffuseOpacity", "NormalRoughness", "SpecularShininess", "AmbientMetallic", "EmissionColor"};
    auto bindTexture = [&](std::size_t unit, const char* name, GLuint texture, GLenum target = GL_TEXTURE_2D)
    {
        m_Impl->GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
        m_Impl->GlBindTexture(target, texture);
        m_Impl->GlUniform1i(location(name), static_cast<GLint>(unit));
    };
    for (std::size_t index = 0; index < names.size(); ++index)
        bindTexture(index, names[index], multisample ? m_Impl->Multisample.Textures[Impl::SampleMaterialFirst + index] : m_Impl->GBufferTextures[index],
            multisample ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D);
    bindTexture(GBufferAttachmentCount, "SceneDepth", multisample ? m_Impl->Multisample.Depth : m_Impl->DepthTexture,
        multisample ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D);
    GLuint shadowTexture = shadowAvailable && desc.SamplesShadowMap ? shadow->second.Texture : 0;
    if (m_Impl->UsesVSM() && shadowAvailable && desc.SamplesShadowMap &&
        !m_Impl->MomentTexture(m_Impl->LastShadowTexture, shadowTexture)) return false;
    bindTexture(GBufferAttachmentCount + 1, "ShadowMap", shadowTexture);
    if (!m_Impl->BindCascades(program, desc.SamplesShadowMap)) return false;
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    if (multisample)
    {
        // ES 3.1 needs no sample-shading extension: evaluate each material sample
        // in its own fullscreen draw, writing only the corresponding output bit.
        m_Impl->GlEnable(GL_SAMPLE_MASK);
        for (GLuint sample = 0; sample < m_Impl->FrameSampleCount; ++sample)
        {
            const auto& position = m_Impl->Multisample.Locations[sample];
            m_Impl->GlSampleMaski(0, 1u << sample);
            m_Impl->GlUniform1i(location("LightingSample"), static_cast<GLint>(sample));
            if (m_Impl->FrameContact.Enabled) m_Impl->GlUniform1i(location("ContactSample"), static_cast<GLint>(sample));
            m_Impl->GlUniform4f(location("SampleLocation"), position[0], position[1], 0, 0);
            m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
        }
        m_Impl->GlDisable(GL_SAMPLE_MASK);
        m_Impl->GlSampleMaski(0, std::numeric_limits<GLuint>::max());
    }
    else m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
    m_Impl->UnbindSampledTextures();
    if (!m_Impl->CheckGL("Deferred lighting")) return false;
        if (m_Impl->SubsurfaceProducing)
        { ++m_Impl->PendingFrame.Statistics.SubsurfacePassCount; return true; }
        m_Impl->SubsurfaceLighting=desc;
        m_Impl->SubsurfaceHasScene=desc.Output==DeferredOutputMode::Scene;
    PassDesc pass;
    pass.Type = PassType::DeferredLighting;
    pass.ClearColorEnabled = pass.ClearDepthEnabled = false;
    m_Impl->PendingFrame.Passes.push_back({pass, {}});
    ++m_Impl->PendingFrame.Statistics.DeferredLightingPassCount;
    m_Impl->DeferredResolved = true;
    return true;
}

bool OpenGLESRHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ApplyAmbientOcclusion")) return false;
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
        return m_Impl->Fail("OpenGL ES AO requires resolved sample depth and materials");
    if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->AmbientOcclusionActive || m_Impl->AmbientOcclusionResolved ||
        m_Impl->ReflectionResolved || m_Impl->DepthOfFieldResolved || m_Impl->VelocityCaptured ||
        !m_Impl->GBufferWritten || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten))
        return m_Impl->Fail("OpenGL ES ambient occlusion requires completed opaque color and material inputs once per frame");
    RayTracedAOConstants packet{};
    auto& constants = packet.Surface;
    const bool rayTraced = m_Impl->FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::RayTraced;
    if (!BuildAmbientOcclusionFrameConstants(desc, m_Impl->FrameAmbientOcclusionSettings, m_Impl->FrameRayScene.get(), m_Impl->CreateInfo.Width,
        m_Impl->CreateInfo.Height, Techniques::AmbientOcclusionPassKind::Estimate, packet))
        return m_Impl->Fail("OpenGL ES ambient occlusion requires a finite, invertible camera projection");
    const auto restore = [&]()
    {
        m_Impl->UnbindSampledTextures();
        m_Impl->GlDisable(GL_BLEND);
        m_Impl->GlBlendEquation(GL_FUNC_ADD);
        m_Impl->GlBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
        if (rayTraced) for (GLuint slot=0;slot<RaySceneBufferCount;++slot)
            m_Impl->GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,0);
    };
    const bool scalable = m_Impl->FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::SAO;
    const bool groundTruth = m_Impl->FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::GTAO;
    const GLuint program = rayTraced ? m_Impl->RayAOProgram : groundTruth ? m_Impl->GtaoProgram : scalable ? m_Impl->SaoProgram :
        m_Impl->FrameAmbientOcclusionSettings.Mode == Techniques::AmbientOcclusionMode::HBAO
            ? m_Impl->HbaoProgram : m_Impl->SsaoProgram;
    const auto plan = Techniques::BuildAmbientOcclusionPlan(m_Impl->FrameAmbientOcclusionSettings,
        m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    const auto maxMip = scalable ? AmbientOcclusionDepthMipCount(m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height) - 1 : 0;
    m_Impl->GlUseProgram(program);
    if (rayTraced) for (GLuint slot=0;slot<RaySceneBufferCount;++slot)
        m_Impl->GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,m_Impl->RaySceneBuffers[slot]);
    const auto location = [&](const char* name) { return m_Impl->GlGetUniformLocation(program, name); };
    const auto projection = ToColumnMajor(constants.ViewProjection);
    const auto inverse = ToColumnMajor(constants.InverseViewProjection);
    m_Impl->GlUniformMatrix4fv(location("ViewProjection"), 1, GL_FALSE_VALUE, projection.data());
    m_Impl->GlUniformMatrix4fv(location("InverseViewProjection"), 1, GL_FALSE_VALUE, inverse.data());
    m_Impl->UploadFloat4(location("ViewportRadiusBias"), constants.ViewportRadiusBias);
    m_Impl->UploadFloat4(location("ForwardIntensity"), constants.ForwardIntensity);
    if (rayTraced)
    {
        const auto& counts=packet.SceneCounts;
        m_Impl->GlUniform4ui(location("SceneCounts"),counts.X,counts.Y,counts.Z,counts.W);
        m_Impl->GlUniform4ui(location("Sampling"),packet.Sampling.X,0,0,0);
    }
    const std::array<const char*, 4> names{"SceneDepth", "NormalRoughness", "AmbientMetallic", rayTraced ? "PreviousVisibility" : "PreviousAO"};
    for (std::size_t index = 0; index < names.size(); ++index)
        m_Impl->GlUniform1i(location(names[index]), static_cast<GLint>(index));
    m_Impl->GlViewport(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_DITHER);
    m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    std::uint32_t destinationMip = 0;
    for (const auto pass : plan)
    {
        const auto stage = static_cast<std::uint32_t>(pass);
        const bool prepare = pass == Techniques::AmbientOcclusionPassKind::DepthPrepare;
        const bool reduce = pass == Techniques::AmbientOcclusionPassKind::DepthReduce;
        const bool hierarchy = prepare || reduce;
        if (reduce) ++destinationMip;
        const int width = hierarchy ? std::max(1, m_Impl->CreateInfo.Width >> destinationMip) : m_Impl->CreateInfo.Width;
        const int height = hierarchy ? std::max(1, m_Impl->CreateInfo.Height >> destinationMip) : m_Impl->CreateInfo.Height;
        const bool composite = pass == Techniques::AmbientOcclusionPassKind::Composite;
        const std::size_t destination = stage == 1 ? 1 : 0;
        const GLuint previous = hierarchy || stage == 0 ? m_Impl->GBufferTextures[3] : m_Impl->AmbientOcclusionTextures[stage == 2 ? 1 : 0];
        m_Impl->UnbindSampledTextures();
        // Composite uses the color-only lighting FBO; opaque depth is sampled.
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, hierarchy ? m_Impl->SaoDepthFramebuffer :
            composite ? m_Impl->LightingFramebuffer : m_Impl->AmbientOcclusionFramebuffers[destination]);
        m_Impl->GlViewport(0, 0, width, height);
        const GLuint depthInput = scalable && !prepare ? m_Impl->SaoDepthHierarchy : m_Impl->DepthTexture;
        const std::array<GLuint, 4> inputs{depthInput, m_Impl->GBufferTextures[1], m_Impl->GBufferTextures[3], previous};
        for (std::size_t unit = 0; unit < inputs.size(); ++unit)
        {
            m_Impl->GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
            m_Impl->GlBindTexture(GL_TEXTURE_2D, inputs[unit]);
        }
        if (composite)
        {
            m_Impl->GlEnable(GL_BLEND);
            m_Impl->GlBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
            m_Impl->GlBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
            m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_FALSE_VALUE);
        }
        else
        {
            m_Impl->GlDisable(GL_BLEND);
            m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        }
        m_Impl->GlUniform4ui(location("Options"), stage, hierarchy ? destinationMip : 0, maxMip, 0);
        m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
        if (!m_Impl->CheckGL("Draw AO pass")) { restore(); return false; }
        if (hierarchy)
        {
            // Scratch is a separate object: rendering never samples an attached
            // texture, even when the requested mip differs from the sampled mip.
            m_Impl->UnbindSampledTextures();
            m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->SaoDepthHierarchy);
            m_Impl->GlReadBuffer(GL_COLOR_ATTACHMENT0);
            m_Impl->GlCopyTexSubImage2D(GL_TEXTURE_2D, static_cast<GLint>(destinationMip), 0, 0, 0, 0, width, height);
            m_Impl->GlBindTexture(GL_TEXTURE_2D, 0);
            if (!m_Impl->CheckGL("Copy SAO depth mip")) { restore(); return false; }
        }
        m_Impl->PendingFrame.AmbientOcclusionPasses.push_back(pass);
    }
    restore();
    if (!m_Impl->CheckGL("Restore AO state")) return false;
    m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount = static_cast<std::uint32_t>(plan.size());
    m_Impl->AmbientOcclusionResolved = true;
    return true;
}

bool OpenGLESRHI::CompleteEnvironmentCapture()
{
    constexpr std::uint32_t CompleteFaces = (1u << EnvironmentFaceCount) - 1u;
    if (!m_Impl->CheckThread("CompleteEnvironmentCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || (!m_Impl->FrameEnvironment.Source && m_Impl->EnvironmentFacesWritten != CompleteFaces))
        return m_Impl->Fail("OpenGL ES mip generation requires all six completed capture faces exactly once");
    if (const auto& source=m_Impl->FrameEnvironment.Source)
    {
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER,0); m_Impl->UnbindSampledTextures();
        m_Impl->GlBindTexture(GL_TEXTURE_CUBE_MAP,m_Impl->Environment.Cube);
        for (std::uint32_t face=0;face<EnvironmentFaceCount;++face)
            m_Impl->GlTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+face,0,GL_RGBA16F,source->Resolution,source->Resolution,
                0,GL_RGBA,GL_HALF_FLOAT,source->FacesRGBA16[face].data());
        m_Impl->GlBindTexture(GL_TEXTURE_CUBE_MAP,0);
        if (!m_Impl->CheckGL("Upload HDR environment faces")) return false;
    }
    if (m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude)
    {
        if (!m_Impl->ConvertEnvironmentMap()) return false;
        m_Impl->EnvironmentReady = true;
        return true;
    }
    m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
    m_Impl->UnbindSampledTextures();
    m_Impl->GlBindTexture(GL_TEXTURE_CUBE_MAP, m_Impl->Environment.Cube);
    // ES requires a cube-complete, filterable and color-renderable base format.
    // RGBA16F renderability is already gated by the native HDR/material capability.
    // https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glGenerateMipmap.xhtml
    m_Impl->GlGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    m_Impl->GlBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    if (!m_Impl->CheckGL("Generate environment cube mips")) return false;
    m_Impl->EnvironmentReady = true;
    return true;
}

bool OpenGLESRHI::CompletePlanarCapture()
{
    if (!m_Impl->CheckThread("CompletePlanarCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->PlanarActive || !m_Impl->PlanarWritten || m_Impl->PlanarReady)
        return m_Impl->Fail("OpenGL ES planar mip generation requires one completed capture exactly once per frame");
    const auto restore = [&]()
    {
        m_Impl->UnbindSampledTextures(); m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
    };
    m_Impl->UnbindSampledTextures();
    m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, m_Impl->Planar.ReductionFramebuffer);
    m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlDisable(GL_DITHER); m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    const auto program = m_Impl->PlanarMipProgram;
    m_Impl->GlUseProgram(program);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "PlanarImage"), 0);
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    for (std::uint32_t mip = 1; mip < m_Impl->FramePlanar.MipCount; ++mip)
    {
        const auto width = std::max(1u, m_Impl->FramePlanar.Width >> mip), height = std::max(1u, m_Impl->FramePlanar.Height >> mip);
        m_Impl->GlViewport(0, 0, width, height);
        m_Impl->GlUniform4f(m_Impl->GlGetUniformLocation(program, "Viewport"), float(width), float(height), 0, 0);
        m_Impl->GlUniform4ui(m_Impl->GlGetUniformLocation(program, "Options"), mip - 1, 0, 0, 0);
        m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->Planar.Image);
        m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
        if (!m_Impl->CheckGL("Filter planar capture mip")) { restore(); return false; }
        // Sampling and attachment writes use distinct texture objects. Copy the
        // scratch rectangle into its allocated level without changing GL's Y origin.
        // https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glCopyTexSubImage2D.xhtml
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->Planar.Image);
        m_Impl->GlReadBuffer(GL_COLOR_ATTACHMENT0);
        m_Impl->GlCopyTexSubImage2D(GL_TEXTURE_2D, static_cast<GLint>(mip), 0, 0, 0, 0, width, height);
        m_Impl->GlBindTexture(GL_TEXTURE_2D, 0);
        if (!m_Impl->CheckGL("Copy filtered planar mip")) { restore(); return false; }
        ++m_Impl->PendingFrame.Statistics.PlanarMipPassCount;
    }
    restore();
    if (!m_Impl->CheckGL("Restore planar mip state")) return false;
    m_Impl->PlanarReady = true;
    return true;
}

bool OpenGLESRHI::ApplyPathTracing(const PathTracingDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyPathTracing") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->PathFrame.Active || m_Impl->PathFrame.Resolved || !m_Impl->HdrActive || m_Impl->ForwardColorWritten)
        return m_Impl->Fail("Path tracing requires an untouched HDR frame with primary-ray visibility");
    return m_Impl->TracePaths(desc);
}

bool OpenGLESRHI::ApplyGlobalIllumination(const GlobalIlluminationDesc& desc)
{
    if (m_Impl->FrameOpen && m_Impl->EarlyTemporalActive && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ApplyGlobalIllumination") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->GIActive || m_Impl->GI.Resolved || !m_Impl->HdrActive || !m_Impl->GBufferWritten || (m_Impl->EarlyTemporalActive && !m_Impl->VelocityCaptured) ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
        (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || (m_Impl->EarlyTemporalActive && !m_Impl->SamplesVelocityResolved))) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) || m_Impl->ReflectionResolved ||
        m_Impl->FogResolved || m_Impl->LightShaftResolved || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
        return m_Impl->Fail("OpenGL ES SSGI requires opaque materials/depth/velocity before reflections and volumes");
    if (Techniques::UsesLumen(m_Impl->FrameGI)) return m_Impl->ApplyLumenGI(desc);
    if (m_Impl->FrameVoxelGI.Volume) return m_Impl->ApplyVoxelGI(desc);
        if (Techniques::UsesRayTracedGI(m_Impl->FrameGI)) return m_Impl->ApplyRayGI(desc);
        if (m_Impl->LightmapActive) return m_Impl->ResolveLightmaps(desc);
    if (m_Impl->ProbeActive) return m_Impl->ResolveProbes(desc);
    GlobalIlluminationConstants constants;
    if (!BuildGlobalIlluminationConstants(desc, m_Impl->FrameGI, m_Impl->FrameReflectionShading,
        m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->GI.Valid, constants))
        return m_Impl->Fail("OpenGL ES SSGI requires invertible current and previous cameras");
    const auto restore = [&]()
    {
        m_Impl->UnbindSampledTextures(); m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
    };
    m_Impl->UnbindSampledTextures();
    m_Impl->GlViewport(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlDisable(GL_DITHER); m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL); m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    const unsigned read = m_Impl->GI.ReadIndex, write = m_Impl->GI.WriteIndex();
    using Phase = Techniques::GlobalIlluminationPassKind;
    for (auto phase : {Phase::Trace, Phase::Spatial, Phase::Temporal, Phase::Composite})
    {
        const auto program = phase == Phase::Temporal ? m_Impl->GITemporalProgram : m_Impl->GIProgram;
        const auto target = phase == Phase::Trace ? Impl::GIRaw : phase == Phase::Spatial ? Impl::GIFiltered :
            phase == Phase::Temporal ? Impl::GIColorFirst + write : Impl::GIComposite;
        m_Impl->GlUseProgram(program);
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, m_Impl->GIFramebuffers[target]);
        const auto uploadMatrix = [&](const char* name, const Matrix4& value)
        {
            const auto matrix = ToColumnMajor(value);
            m_Impl->GlUniformMatrix4fv(m_Impl->GlGetUniformLocation(program, name), 1, GL_FALSE_VALUE, matrix.data());
        };
        uploadMatrix("ViewProjection", constants.ViewProjection);
        uploadMatrix("InverseViewProjection", constants.InverseViewProjection);
        uploadMatrix("PreviousInverseViewProjection", constants.PreviousInverseViewProjection);
        const char* vectorNames[] {"PreviousDepthPlane", "Viewport", "Trace", "Appearance", "Temporal", "JitterDelta"};
        const Float4 vectors[] {constants.PreviousDepthPlane, constants.Viewport, constants.Trace, constants.Appearance, constants.Temporal, constants.JitterDelta};
        for (std::size_t i = 0; i < std::size(vectors); ++i)
            m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, vectorNames[i]), vectors[i]);
        m_Impl->GlUniform4ui(m_Impl->GlGetUniformLocation(program, "Options"), static_cast<unsigned>(phase),
            constants.Options.Y, constants.Options.Z, constants.Options.W);
        m_Impl->GlUniform4ui(m_Impl->GlGetUniformLocation(program, "History"), constants.History.X, constants.History.Y, 0, 0);
        const auto estimate = phase == Phase::Spatial ? m_Impl->GITextures[Impl::GIRaw] : phase == Phase::Temporal ?
            m_Impl->GITextures[Impl::GIFiltered] : phase == Phase::Composite ? m_Impl->GITextures[Impl::GIColorFirst + write] : m_Impl->HdrTextures[0];
        const GLuint inputs[] {m_Impl->HdrTextures[0], m_Impl->DepthTexture, m_Impl->GBufferTextures[1],
            m_Impl->GBufferTextures[0], m_Impl->GBufferTextures[3], m_Impl->MotionTextures[Impl::MotionVelocity], estimate,
            m_Impl->GITextures[Impl::GIColorFirst + read], m_Impl->GITextures[Impl::GIDepthFirst + read], m_Impl->GITextures[Impl::GINormalFirst + read]};
        const char* names[] {"SceneColor", "SceneDepth", "NormalRoughness", "DiffuseOpacity", "AmbientMetallic", "MotionVectors",
            "Estimate", "HistoryColor", "HistoryDepth", "HistoryNormal"};
        for (std::size_t i = 0; i < std::size(inputs); ++i)
        {
            m_Impl->GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i)); m_Impl->GlBindTexture(GL_TEXTURE_2D, inputs[i]);
            m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, names[i]), static_cast<GLint>(i));
        }
        m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
        if (!m_Impl->CheckGL("Execute screen-space diffuse transport")) { restore(); return false; }
        m_Impl->UnbindSampledTextures();
    }
    // Copy only the composed scene; the three temporal outputs stay private.
    m_Impl->GlReadBuffer(GL_COLOR_ATTACHMENT0); m_Impl->GlActiveTexture(GL_TEXTURE0);
    m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->HdrTextures[0]);
    m_Impl->GlCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    const bool copied = m_Impl->CheckGL("Copy SSGI scene composite"); restore(); if (!copied) return false;
    m_Impl->GI.Resolved = true;
    m_Impl->PendingFrame.Temporal = desc.Sampling;
    m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount = 4;
    return true;
}

bool OpenGLESRHI::ApplyReflections(const ReflectionDesc& desc)
{
        if (desc.SkyboxOnly)
        {
            const auto saved=m_Impl->FrameReflections;
            const bool active=m_Impl->ReflectionActive,done=m_Impl->ReflectionResolved,planar=m_Impl->PlanarActive;
            m_Impl->FrameReflections.Mode=Techniques::ReflectionMode::Cubemap;
            m_Impl->FrameReflections.EnvironmentIntensity=0;
            m_Impl->FrameReflections.Planar.Enabled=false;
            m_Impl->ReflectionActive=true; m_Impl->ReflectionResolved=false; m_Impl->PlanarActive=false;
            auto request=desc; request.SkyboxOnly=false; request.DrawSkybox=true;
            const bool success=m_Impl->EnsureReflectionResources(false) && ApplyReflections(request);
            m_Impl->FrameReflections=saved; m_Impl->ReflectionActive=active; m_Impl->ReflectionResolved=done; m_Impl->PlanarActive=planar;
            if (success) m_Impl->PendingFrame.Statistics.SkyboxPassCount+=2;
            return success;
        }

    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ApplyReflections")) return false;
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
        return m_Impl->Fail("OpenGL ES SSR requires resolved sample color, depth and materials");
    if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->ReflectionActive || m_Impl->ReflectionResolved ||
        (m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady) ||
        (m_Impl->PlanarActive && !m_Impl->PlanarReady) ||
        !m_Impl->HdrActive || !m_Impl->GBufferWritten || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
        (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->DepthOfFieldResolved ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved))
        return m_Impl->Fail("OpenGL ES SSR requires opaque lighting/materials and AO, once before velocity and transparency");
    if (Techniques::UsesLumenReflections(m_Impl->FrameReflections)) return m_Impl->ApplyLumenReflections(desc);
        if (m_Impl->FrameReflections.Mode==Techniques::ReflectionMode::RayTraced)
        return Techniques::UsesHybridRendering(m_Impl->FrameRayRadianceSettings.Pipeline) ? m_Impl->ApplyHybridRendering(desc) : m_Impl->ApplyRayReflections(desc);
    for (const bool planarPass : {false,true})
    {
        if (planarPass ? !m_Impl->PlanarActive : m_Impl->FrameReflections.Mode==Techniques::ReflectionMode::None) continue;
        ReflectionConstants constants{};
        CubemapReflectionConstants cubeConstants{};
        EnvironmentMapReflectionConstants mapConstants{};
        PlanarReflectionConstants planarConstants{};
        const bool cube = !planarPass && m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::Cubemap;
        const bool latitudeLongitude = !planarPass && m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude;
        const bool planar = planarPass;
        if (!m_Impl->EnvironmentActive && !planar && !BuildReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
            return m_Impl->Fail("OpenGL ES SSR requires a finite invertible projection");
        if (cube && !BuildCubemapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->FrameEnvironment.MipCount, cubeConstants))
            return m_Impl->Fail("OpenGL ES cube reflections require a finite invertible projection");
        if (latitudeLongitude && !BuildEnvironmentMapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->EnvironmentMap.Layout.MipCount, mapConstants))
            return m_Impl->Fail("OpenGL ES environment map reflections require a finite invertible projection");
        if (planar && !BuildPlanarReflectionConstants(desc, m_Impl->FramePlanar, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, planarConstants))
            return m_Impl->Fail("OpenGL ES planar lookup requires the frozen capture camera and main viewport");
        const auto restore = [&]()
        {
            m_Impl->UnbindSampledTextures();
            m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
            m_Impl->GlDisable(GL_BLEND);
            m_Impl->GlBlendEquation(GL_FUNC_ADD);
            m_Impl->GlBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
            m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
            m_Impl->GlDepthMask(GL_TRUE_VALUE);
        };
        const GLuint program = planar ? m_Impl->PlanarProgram : latitudeLongitude ? m_Impl->EnvironmentMapProgram : cube ? m_Impl->CubeReflectionProgram : m_Impl->ReflectionProgram;
        m_Impl->UnbindSampledTextures();
        m_Impl->GlUseProgram(program);
        const auto location = [&](const char* name) { return m_Impl->GlGetUniformLocation(program, name); };
        const auto inverse = ToColumnMajor(planar ? planarConstants.InverseViewProjection : latitudeLongitude ? mapConstants.InverseViewProjection :
            cube ? cubeConstants.InverseViewProjection : constants.InverseViewProjection);
        m_Impl->GlUniformMatrix4fv(location("InverseViewProjection"), 1, GL_FALSE_VALUE, inverse.data());
        m_Impl->UploadFloat4(location("Viewport"), planar ? planarConstants.Viewport : latitudeLongitude ? mapConstants.Viewport : cube ? cubeConstants.Viewport : constants.Viewport);
        if (planar)
        {
            const auto mirror = ToColumnMajor(planarConstants.ReflectionViewProjection);
            m_Impl->GlUniformMatrix4fv(location("ReflectionViewProjection"), 1, GL_FALSE_VALUE, mirror.data());
            m_Impl->UploadFloat4(location("ReceiverPlane"), planarConstants.ReceiverPlane);
            m_Impl->UploadFloat4(location("ReceiverMinimum"), planarConstants.ReceiverMinimum);
            m_Impl->UploadFloat4(location("ReceiverMaximum"), planarConstants.ReceiverMaximum);
            m_Impl->UploadFloat4(location("Sampling"), planarConstants.Sampling);
            m_Impl->GlUniform1i(location("PlanarImage"), 7);
        }
        else if (cube || latitudeLongitude)
        {
            m_Impl->UploadFloat4(location("ReceiverPlane"),latitudeLongitude ? mapConstants.ReceiverPlane : cubeConstants.ReceiverPlane);
            m_Impl->UploadFloat4(location("ReceiverMinimum"),latitudeLongitude ? mapConstants.ReceiverMinimum : cubeConstants.ReceiverMinimum);
            m_Impl->UploadFloat4(location("ReceiverMaximum"),latitudeLongitude ? mapConstants.ReceiverMaximum : cubeConstants.ReceiverMaximum);
            m_Impl->UploadFloat4(location("Sampling"), latitudeLongitude ? mapConstants.Sampling : cubeConstants.Sampling);
            m_Impl->GlUniform1i(location(latitudeLongitude ? "EnvironmentImage" : "EnvironmentRadiance"), 7);
            if (cube)
            {
                m_Impl->UploadFloat4(location("AtmospherePlanet"),cubeConstants.AtmospherePlanet);
                m_Impl->UploadFloat4(location("AtmosphereRayleigh"),cubeConstants.AtmosphereRayleigh);
                m_Impl->UploadFloat4(location("AtmosphereMie"),cubeConstants.AtmosphereMie);
                m_Impl->UploadFloat4(location("AtmosphereSun"),cubeConstants.AtmosphereSun);
                m_Impl->UploadFloat4(location("AtmosphereSolar"),cubeConstants.AtmosphereSolar);
                m_Impl->UploadFloat4(location("AtmosphereQuality"),cubeConstants.AtmosphereQuality);
            }
        }
        else
        {
            const auto projection = ToColumnMajor(constants.ViewProjection);
            m_Impl->GlUniformMatrix4fv(location("ViewProjection"), 1, GL_FALSE_VALUE, projection.data());
            m_Impl->UploadFloat4(location("Trace"), constants.Trace);
            m_Impl->UploadFloat4(location("Appearance"), constants.Appearance);
        }
        constexpr std::array<const char*, 7> samplerNames{
            "SceneColor", "SceneDepth", "NormalRoughness", "SpecularShininess", "DiffuseOpacity", "AmbientMetallic", "ReflectionRadiance"};
        for (std::size_t unit = 0; unit < samplerNames.size(); ++unit)
            m_Impl->GlUniform1i(location(samplerNames[unit]), static_cast<GLint>(unit));
        m_Impl->GlViewport(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
        m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE);
        m_Impl->GlDisable(GL_DITHER); m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
        m_Impl->GlDepthMask(GL_FALSE_VALUE);
        m_Impl->GlBindVertexArray(m_Impl->VertexArray);
        const auto options = planar ? planarConstants.Options : latitudeLongitude ? mapConstants.Options : cube ? cubeConstants.Options : constants.Options;
        for (const auto pass : {planar ? Techniques::ReflectionPassKind::PlanarLookup :
            Techniques::UsesEnvironmentCapture(m_Impl->FrameReflections) ? Techniques::ReflectionPassKind::EnvironmentLookup : Techniques::ReflectionPassKind::Trace,
            Techniques::ReflectionPassKind::Composite})
        {
            const bool composite = pass == Techniques::ReflectionPassKind::Composite;
            m_Impl->UnbindSampledTextures();
            // Both destinations are color-only. The geometry FBO retains opaque
            // depth, which SSR samples during trace and later transparency tests.
            m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, composite ? m_Impl->LightingFramebuffer : m_Impl->ReflectionFramebuffer);
            m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, composite ? GL_FALSE_VALUE : GL_TRUE_VALUE);
            if (composite)
            {
                m_Impl->GlEnable(GL_BLEND);
                m_Impl->GlBlendEquation(GL_FUNC_ADD);
                m_Impl->GlBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
            }
            else m_Impl->GlDisable(GL_BLEND);
            m_Impl->GlUniform4ui(location("Options"), static_cast<GLuint>(pass),
                options.Y, options.Z, options.W);
            const std::array<GLuint, 7> sources = composite ?
                std::array<GLuint, 7>{0, 0, 0, 0, 0, 0, m_Impl->ReflectionTexture} :
                std::array<GLuint, 7>{m_Impl->HdrTextures[0], m_Impl->DepthTexture, m_Impl->GBufferTextures[1],
                    m_Impl->GBufferTextures[2], m_Impl->GBufferTextures[0], m_Impl->GBufferTextures[3], 0};
            for (std::size_t unit = 0; unit < sources.size(); ++unit)
            {
                m_Impl->GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
                m_Impl->GlBindTexture(GL_TEXTURE_2D, sources[unit]);
            }
            if (m_Impl->EnvironmentActive && !planar)
            {
                m_Impl->GlActiveTexture(GL_TEXTURE0 + 7);
                m_Impl->GlBindTexture(latitudeLongitude ? GL_TEXTURE_2D : GL_TEXTURE_CUBE_MAP,
                    latitudeLongitude ? m_Impl->EnvironmentMap.Image : m_Impl->Environment.Cube);
            }
            else if (planar)
            {
                m_Impl->GlActiveTexture(GL_TEXTURE0 + 7);
                m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->Planar.Image);
            }
            m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
            if (!m_Impl->CheckGL("Draw SSR phase")) { restore(); return false; }
            m_Impl->PendingFrame.ReflectionPasses.push_back(pass);
        }
        restore();
    }
    m_Impl->ReflectionResolved = true;
    m_Impl->PendingFrame.Statistics.ReflectionPassCount = static_cast<std::uint32_t>(m_Impl->PendingFrame.ReflectionPasses.size());
    return true;
}

bool OpenGLESRHI::ApplyDepthOfField(const DepthOfFieldDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
    if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved)
        return m_Impl->Fail("OpenGL ES light shafts must complete before depth of field");
    if (m_Impl->FogActive && !m_Impl->FogResolved)
        return m_Impl->Fail("OpenGL ES fog must complete before depth of field");
    if (!m_Impl->CheckThread("ApplyDepthOfField")) return false;
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
        return m_Impl->Fail("OpenGL ES depth of field requires resolved sample color and depth");
    if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->HdrActive || !m_Impl->DepthOfFieldActive ||
        m_Impl->DepthOfFieldResolved || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
        (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) ||
        (m_Impl->MotionBlurActive && !m_Impl->MotionBlurResolved) ||
        (m_Impl->TemporalActive && !m_Impl->TemporalResolved))
        return m_Impl->Fail("OpenGL ES depth of field requires completed opaque lighting and AO once per frame");
    DepthOfFieldConstants constants{};
    if (!BuildDepthOfFieldConstants(desc, m_Impl->FramePostProcessSettings.DepthOfField,
        m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, Techniques::DepthOfFieldPassKind::Gather, constants))
        return m_Impl->Fail("OpenGL ES depth of field requires a finite conventional perspective or affine camera");
    const auto restore = [&]()
    {
        m_Impl->UnbindSampledTextures();
        m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
    };
    const GLuint program = m_Impl->DepthOfFieldProgram;
    m_Impl->UnbindSampledTextures();
    m_Impl->GlUseProgram(program);
    m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, "ViewportRadius"), constants.ViewportRadius);
    m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, "CoCScaleBias"), constants.CoCScaleBias);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "SourceColor"), 0);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "OpaqueDepth"), 1);
    m_Impl->GlViewport(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlDisable(GL_DITHER); m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    const auto plan = Techniques::BuildDepthOfFieldPlan(m_Impl->FramePostProcessSettings.DepthOfField);
    for (const auto pass : plan)
    {
        const bool composite = pass == Techniques::DepthOfFieldPassKind::Composite;
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, composite ? m_Impl->DepthOfFieldFramebuffer : m_Impl->HdrFramebuffers[1]);
        m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, composite ? GL_FALSE_VALUE : GL_TRUE_VALUE);
        m_Impl->GlUniform4ui(m_Impl->GlGetUniformLocation(program, "Options"), static_cast<GLuint>(pass), 0, 0, 0);
        m_Impl->GlActiveTexture(GL_TEXTURE0);
        m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->HdrTextures[composite ? 1 : 0]);
        m_Impl->GlActiveTexture(GL_TEXTURE0 + 1);
        m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->DepthTexture);
        m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
        if (!m_Impl->CheckGL("Draw depth-of-field pass")) { restore(); return false; }
        m_Impl->PendingFrame.DepthOfFieldPasses.push_back(pass);
    }
    restore();
    m_Impl->DepthOfFieldResolved = true;
    m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount = static_cast<std::uint32_t>(plan.size());
    return true;
}

bool OpenGLESRHI::ApplyLightShafts(const LightShaftDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ApplyLightShafts") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->HdrActive || !m_Impl->LightShaftActive || m_Impl->LightShaftResolved || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
        (m_Impl->GIActive && !m_Impl->GI.Resolved) ||
        m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
        (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) || (m_Impl->FogActive && !m_Impl->FogResolved) ||
        (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
        return m_Impl->Fail("OpenGL ES light shafts require resolved opaque lighting, AO, reflections and fog, once before temporal effects and transparency");
    LightShaftConstants constants;
    if (!BuildLightShaftConstants(desc, m_Impl->FrameLightShafts, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
        return m_Impl->Fail("OpenGL ES light shafts require finite camera/light inputs and valid dimensions");
    const auto restore = [&]()
    {
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
    };
    const auto program = m_Impl->LightShaftProgram;
    m_Impl->GlUseProgram(program);
    m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, "ShaftProjection"), constants.Source);
    m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, "ShaftColorScale"), constants.Radiance);
    m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, "ShaftFalloffViewport"), constants.Integration);
    const GLint configuration = m_Impl->GlGetUniformLocation(program, "ShaftConfiguration");
    constexpr std::array<const char*, 3> samplers {"ShaftOpaqueColor", "ShaftOpaqueDepth", "ShaftOcclusion"};
    for (std::size_t index = 0; index < samplers.size(); ++index)
        m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, samplers[index]), static_cast<GLint>(index));
    m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlDisable(GL_DITHER); m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    m_Impl->GlViewport(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    for (GLuint phase = 0; phase < 2; ++phase)
    {
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, phase == 0 ? m_Impl->LightShaftFramebuffer : m_Impl->HdrFramebuffers[1]);
        m_Impl->GlUniform4ui(configuration, constants.Control.X, phase, constants.Control.Z, constants.Control.W);
        const std::array<GLuint, 3> textures {m_Impl->HdrTextures[0], m_Impl->DepthTexture,
            phase == 0 ? m_Impl->HdrTextures[0] : m_Impl->LightShaftTexture};
        for (GLuint index = 0; index < textures.size(); ++index)
        {
            m_Impl->GlActiveTexture(GL_TEXTURE0 + index);
            m_Impl->GlBindTexture(GL_TEXTURE_2D, textures[index]);
        }
        // Both draw FBOs are color-only, so sampled depth is never attached.
        // The unused mask input in phase zero also names a non-attached texture.
        constexpr GLsizei FullscreenVertices = 3;
        m_Impl->GlDrawArrays(GL_TRIANGLES, 0, FullscreenVertices);
        if (!m_Impl->CheckGL("Draw light-shaft phase")) { restore(); return false; }
    }
    m_Impl->UnbindSampledTextures();
    m_Impl->GlBindFramebuffer(GL_READ_FRAMEBUFFER, m_Impl->HdrFramebuffers[1]);
    m_Impl->GlReadBuffer(GL_COLOR_ATTACHMENT0);
    m_Impl->GlBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_Impl->HdrFramebuffers[0]);
    m_Impl->GlBlitFramebuffer(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
        0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    const bool copied = m_Impl->CheckGL("Blit light-shaft HDR composition");
    restore();
    if (!copied) return false;
    m_Impl->LightShaftResolved = true;
    m_Impl->PendingFrame.Statistics.LightShaftPassCount += 2;
    return true;
}

bool OpenGLESRHI::ApplyVolumetricFog(const VolumetricFogDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ApplyVolumetricFog") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->HdrActive || !m_Impl->FogActive || m_Impl->FogResolved || m_Impl->LightShaftResolved || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
        (m_Impl->GIActive && !m_Impl->GI.Resolved) ||
        m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
        (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) ||
        (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
        return m_Impl->Fail("OpenGL ES fog requires completed opaque lighting, AO and reflections, once before temporal effects and transparency");
    if (!BuildVolumetricFogConstants(desc, m_Impl->FrameFog, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
        m_Impl->MaximumTextureSize, m_Impl->FogConstants))
        return m_Impl->Fail("OpenGL ES fog requires a finite invertible camera frustum");
    const bool lighting = (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog));
    if (lighting && !m_Impl->GenerateVolumeLighting(desc)) return false;
    const auto restore = [&]()
    {
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
    };
    const GLuint program = Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->CloudFogProgram : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->MediaFogProgram : m_Impl->FogProgram;
    m_Impl->GlUseProgram(program);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "FogSceneColor"), 0);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "FogSceneDepth"), 1);
    m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlDisable(GL_DITHER); m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    for (std::uint32_t phase = lighting ? 1U : 0U; phase < 2; ++phase)
    {
        m_Impl->UnbindSampledTextures();
        const bool integrate = phase == 0;
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, integrate ? m_Impl->FogFramebuffer : m_Impl->HdrFramebuffers[1]);
        m_Impl->GlViewport(0, 0, integrate ? m_Impl->FogDimensions.AtlasWidth() : m_Impl->CreateInfo.Width,
            integrate ? m_Impl->FogDimensions.Height : m_Impl->CreateInfo.Height);
        auto constants = m_Impl->FogConstants; constants.Options.X = phase;
        m_Impl->UploadFogConstants(program, constants);
        if (!integrate)
        {
            m_Impl->GlActiveTexture(GL_TEXTURE0); m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->HdrTextures[0]);
            m_Impl->GlActiveTexture(GL_TEXTURE0 + 1); m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->DepthTexture);
            m_Impl->GlActiveTexture(GL_TEXTURE0 + Impl::FogTextureUnit); m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->FogTexture);
        }
        m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
        if (!m_Impl->CheckGL("Draw volumetric fog phase")) { restore(); return false; }
    }
    // Scratch has no depth attachment, preventing feedback while reconstructing
    // opaque positions. The color-only blit preserves HDR/alpha and scene depth;
    // it does not require the R32F/CopyTexImage support used by TAA history.
    m_Impl->UnbindSampledTextures();
    m_Impl->GlBindFramebuffer(GL_READ_FRAMEBUFFER, m_Impl->HdrFramebuffers[1]);
    m_Impl->GlReadBuffer(GL_COLOR_ATTACHMENT0);
    m_Impl->GlBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_Impl->HdrFramebuffers[0]);
    m_Impl->GlBlitFramebuffer(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
        0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    const bool copied = m_Impl->CheckGL("Blit fogged HDR scene");
    restore();
    if (!copied) return false;
    m_Impl->FogResolved = true;
    (Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount : m_Impl->PendingFrame.Statistics.VolumetricFogPassCount) += lighting ? 1 : 2;
    return true;
}

bool OpenGLESRHI::ApplyTemporalEffects(const TemporalEffectsDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyTemporalEffects") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->FramePostProcessSettings.TemporalEffects.Enabled || !m_Impl->HdrActive || m_Impl->ExposureHistory.Resolved)
        return m_Impl->Fail("Temporal exposure requires completed scene HDR, once before presentation");
    if (!m_Impl->OpaqueEffectsComplete() || !m_Impl->Hair.Complete() || !m_Impl->Hair.VelocityComplete() ||
        !(m_Impl->PathFrame.Active?m_Impl->PathFrame.Resolved:m_Impl->DeferredActive?m_Impl->DeferredResolved:m_Impl->ForwardColorWritten) ||
        (m_Impl->FrameSampleCount>1 && !m_Impl->SamplesColorResolved))
        return m_Impl->Fail("Complete opaque effects and final multisample color before exposure metering");
    return m_Impl->ResolveExposure(desc);
}

bool OpenGLESRHI::ApplyTemporalAA(const TemporalAADesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
    if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved)
        return m_Impl->Fail("OpenGL ES light shafts must complete before TAA");
    if (m_Impl->FogActive && !m_Impl->FogResolved)
        return m_Impl->Fail("OpenGL ES fog must complete before TAA");
    if (!m_Impl->CheckThread("ApplyTemporalAA")) return false;
    if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->HdrActive || !m_Impl->TemporalActive ||
        !m_Impl->VelocityCaptured || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved))
        return m_Impl->Fail("OpenGL ES TAA requires completed opaque velocity once before blur and transparency");
    TemporalAAConstants constants{};
    if (!BuildTemporalAAConstants(desc, m_Impl->FramePostProcessSettings.TemporalAA, m_Impl->CreateInfo.Width,
        m_Impl->CreateInfo.Height, m_Impl->TemporalHistoryValid, constants))
        return m_Impl->Fail("OpenGL ES TAA requires a finite previous camera and jitter");
    const auto restore = [&]()
    {
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
    };
    const auto read = m_Impl->TemporalHistoryReadIndex;
    const auto write = 1 - read;
    const GLuint program = m_Impl->TemporalProgram;
    m_Impl->UnbindSampledTextures();
    m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, m_Impl->TemporalFramebuffers[write]);
    m_Impl->GlViewport(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlDisable(GL_DITHER); m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    m_Impl->GlUseProgram(program);
    const auto matrix = ToColumnMajor(constants.PreviousInverseViewProjection);
    m_Impl->GlUniformMatrix4fv(m_Impl->GlGetUniformLocation(program, "PreviousInverseCamera"), 1, GL_FALSE_VALUE, matrix.data());
    m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, "PreviousOpticalPlane"), constants.PreviousDepthPlane);
    m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, "ExtentFeedbackTolerance"), constants.ExtentWeightTolerance);
    m_Impl->UploadFloat4(m_Impl->GlGetUniformLocation(program, "JitterDifference"), constants.JitterDelta);
    m_Impl->GlUniform4ui(m_Impl->GlGetUniformLocation(program, "HistoryOptions"), constants.Options.X, 0, 0, 0);
    const std::array<GLuint, 5> inputs {m_Impl->HdrTextures[0], m_Impl->DepthTexture,
        m_Impl->MotionTextures[Impl::MotionVelocity], m_Impl->TemporalColors[read], m_Impl->TemporalDepths[read]};
    constexpr const char* samplers[] = {"CurrentColor", "CurrentDepth", "PhysicalMotion", "HistoryColor", "HistoryDepth"};
    for (std::size_t unit = 0; unit < inputs.size(); ++unit)
    {
        m_Impl->GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
        m_Impl->GlBindTexture(GL_TEXTURE_2D, inputs[unit]);
        m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, samplers[unit]), static_cast<GLint>(unit));
    }
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
    if (!m_Impl->CheckGL("Resolve temporal color and depth history")) { restore(); return false; }
    // Copy into the mutable scene, leaving the pending history untouched by
    // subsequent blur/transparency/HDR effects. Float-to-float copying preserves
    // HDR radiance under EXT_color_buffer_float (the R32F capability gate).
    m_Impl->UnbindSampledTextures();
    m_Impl->GlReadBuffer(GL_COLOR_ATTACHMENT0);
    m_Impl->GlActiveTexture(GL_TEXTURE0);
    m_Impl->GlBindTexture(GL_TEXTURE_2D, m_Impl->HdrTextures[0]);
    m_Impl->GlCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    const bool copied = m_Impl->CheckGL("Copy reconstructed HDR scene");
    restore();
    if (!copied) return false;
    m_Impl->TemporalResolved = true;
    m_Impl->PendingFrame.Temporal = desc.Sampling;
    ++m_Impl->PendingFrame.Statistics.TemporalAAPassCount;
    return true;
}

bool OpenGLESRHI::ApplyMotionBlur(const MotionBlurDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
    if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved)
        return m_Impl->Fail("OpenGL ES light shafts must complete before motion blur");
    if (m_Impl->FogActive && !m_Impl->FogResolved)
        return m_Impl->Fail("OpenGL ES fog must complete before motion blur");
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved)
        return m_Impl->Fail("OpenGL ES motion blur requires resolved sample velocity");
    if (!m_Impl->CheckThread("ApplyMotionBlur")) return false;
    if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->HdrActive || !m_Impl->MotionBlurActive ||
        !m_Impl->VelocityCaptured || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        (m_Impl->TemporalActive && !m_Impl->TemporalResolved) ||
        (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved))
        return m_Impl->Fail("OpenGL ES motion blur requires completed velocity, once before lens blur/transparency");
    MotionBlurConstants constants{};
    if (!BuildMotionBlurConstants(desc, m_Impl->FramePostProcessSettings.MotionBlur,
        m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
        return m_Impl->Fail("OpenGL ES motion blur requires a finite camera and valid sample interval");
    const auto restore = [&]()
    {
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        m_Impl->GlDepthMask(GL_TRUE_VALUE);
    };
    const GLuint program = m_Impl->MotionScreenProgram;
    m_Impl->UnbindSampledTextures();
    m_Impl->GlUseProgram(program);
    m_Impl->UploadMotionConstants(program, constants);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "OpaqueRadiance"), 0);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "OpaqueDepth"), 1);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "VelocityMap"), 2);
    m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, "MaximumMap"), 3);
    m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_BLEND);
    m_Impl->GlDisable(GL_DITHER); m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL);
    m_Impl->GlDepthMask(GL_FALSE_VALUE);
    m_Impl->GlBindVertexArray(m_Impl->VertexArray);
    using Pass = Techniques::MotionBlurPassKind;
    const auto passes = Techniques::BuildMotionBlurPlan(m_Impl->FramePostProcessSettings.MotionBlur);
    const int tileSize = Techniques::MotionBlurTileSize;
    for (const auto pass : passes)
    {
        const bool composite = pass == Pass::Composite;
        const bool tiles = pass == Pass::TileMax || pass == Pass::NeighborMax;
        const GLuint destination = tiles ? m_Impl->MotionFramebuffers[pass == Pass::TileMax ? Impl::MotionTileMax : Impl::MotionNeighborMax] :
            (composite ? m_Impl->MotionCompositeFramebuffer : m_Impl->HdrFramebuffers[1]);
        const int width = tiles ? (m_Impl->CreateInfo.Width + tileSize - 1) / tileSize : m_Impl->CreateInfo.Width;
        const int height = tiles ? (m_Impl->CreateInfo.Height + tileSize - 1) / tileSize : m_Impl->CreateInfo.Height;
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, destination);
        m_Impl->GlViewport(0, 0, width, height);
        m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, composite ? GL_FALSE_VALUE : GL_TRUE_VALUE);
        m_Impl->GlUniform4ui(m_Impl->GlGetUniformLocation(program, "Options"), static_cast<GLuint>(pass), 0, 0, 0);
        // Keep all active samplers complete on ES, including branches that do
        // not fetch a given input. No bound input aliases the destination FBO.
        const std::array<GLuint, 4> inputs {m_Impl->HdrTextures[composite ? 1 : 0], m_Impl->DepthTexture,
            m_Impl->MotionTextures[Impl::MotionVelocity],
            m_Impl->MotionTextures[pass == Pass::NeighborMax ? Impl::MotionTileMax :
                (pass == Pass::Gather ? Impl::MotionNeighborMax : Impl::MotionVelocity)]};
        for (std::size_t unit = 0; unit < inputs.size(); ++unit)
        {
            m_Impl->GlActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(unit));
            m_Impl->GlBindTexture(GL_TEXTURE_2D, inputs[unit]);
        }
        m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
        if (!m_Impl->CheckGL("Draw motion reconstruction pass")) { restore(); return false; }
        m_Impl->PendingFrame.MotionBlurPasses.push_back(pass);
    }
    restore();
    m_Impl->MotionBlurResolved = true;
    m_Impl->PendingFrame.Statistics.MotionBlurPassCount = static_cast<std::uint32_t>(passes.size());
    return true;
}

bool OpenGLESRHI::ProcessDepthPeeling(DepthPeelingPass pass)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ProcessDepthPeeling") || !m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->FramePeeling.Enabled ||
        !m_Impl->OpaqueEffectsComplete() || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten))
        return m_Impl->Fail("OIT requires completed opaque HDR processing outside a raster pass");
    if (!m_Impl->ProcessPeeling(pass)) return false;
    RecordDepthPeelingPass(m_Impl->FramePeeling,pass,m_Impl->PendingFrame.Statistics); return true;
}

bool OpenGLESRHI::ProcessRefraction(const RefractionDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ProcessRefraction") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->RefractionActive || m_Impl->RefractionResolved || m_Impl->FrameSampleCount!=1 ||
        !m_Impl->HdrActive || !m_Impl->OpaqueEffectsComplete() || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten))
        return m_Impl->Fail("Refraction requires completed single-sample opaque HDR processing and executes once per frame");
    return m_Impl->ResolveRefraction(desc);
}

bool OpenGLESRHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind pass)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    using Pass = Techniques::WeightedTransparencyPassKind;
    if (!m_Impl->CheckThread("ProcessWeightedTransparency") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->WeightedActive || !m_Impl->HdrActive || !m_Impl->OpaqueEffectsComplete() ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten) ||
        (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved ||
            (m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored))))
        return m_Impl->Fail("OpenGL ES weighted transparency requires completed opaque processing and restored sample color");
    if (pass == Pass::Prepare)
    {
        if (m_Impl->WeightedPrepared || m_Impl->SamplesTransparentWritten)
            return m_Impl->Fail("OpenGL ES prepare weighted transparency once before transparent geometry");
        m_Impl->UnbindSampledTextures();
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, m_Impl->Weighted.Framebuffer);
        m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        const float zero[4] {}, one[4] {1, 1, 1, 1};
        m_Impl->GlClearBufferfv(GL_COLOR, 0, zero);
        m_Impl->GlClearBufferfv(GL_COLOR, 1, one);
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (!m_Impl->CheckGL("Clear weighted transparency statistics")) return false;
        m_Impl->WeightedPrepared = true;
    }
    else if (pass == Pass::Resolve)
    {
        if (!m_Impl->WeightedPrepared || !m_Impl->WeightedAccumulated || m_Impl->WeightedResolved)
            return m_Impl->Fail("OpenGL ES resolve weighted transparency once after its geometry pass");
        const bool multisample = m_Impl->FrameSampleCount > 1;
        const GLenum textureTarget = multisample ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
        const GLuint program = m_Impl->WeightedResolvePrograms[multisample ? 1 : 0];
        m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, multisample ?
            m_Impl->Multisample.Framebuffers[Impl::SampleColorFBO] : m_Impl->HdrFramebuffers[0]);
        m_Impl->GlViewport(0, 0, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
        m_Impl->GlUseProgram(program);
        const char* inputs[] {"AccumulatedRadiance", "Revealage"};
        for (GLuint index = 0; index < m_Impl->Weighted.Textures.size(); ++index)
        {
            m_Impl->GlActiveTexture(GL_TEXTURE0 + index);
            m_Impl->GlBindTexture(textureTarget, m_Impl->Weighted.Textures[index]);
            m_Impl->GlUniform1i(m_Impl->GlGetUniformLocation(program, inputs[index]), index);
        }
        m_Impl->GlDisable(GL_DEPTH_TEST); m_Impl->GlDisable(GL_CULL_FACE); m_Impl->GlDisable(GL_DITHER);
        m_Impl->GlDisable(GL_POLYGON_OFFSET_FILL); m_Impl->GlDepthMask(GL_FALSE_VALUE);
        m_Impl->GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
        m_Impl->GlEnable(GL_BLEND); m_Impl->GlBlendEquation(GL_FUNC_ADD);
        // The non-indexed function restores every target's blend factors,
        // including the revealage attachment's earlier multiplicative state.
        m_Impl->GlBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        m_Impl->GlBindVertexArray(m_Impl->VertexArray);
        if (multisample)
        {
            const GLint location = m_Impl->GlGetUniformLocation(program, "ResolveSample");
            m_Impl->GlEnable(GL_SAMPLE_MASK);
            for (GLuint sample = 0; sample < m_Impl->FrameSampleCount; ++sample)
            {
                m_Impl->GlSampleMaski(0, 1u << sample);
                m_Impl->GlUniform1i(location, sample);
                m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
            }
            m_Impl->GlDisable(GL_SAMPLE_MASK);
            m_Impl->GlSampleMaski(0, std::numeric_limits<GLuint>::max());
        }
        else m_Impl->GlDrawArrays(GL_TRIANGLES, 0, 3);
        m_Impl->GlDisable(GL_BLEND);
        m_Impl->UnbindSampledTextures(); m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (!m_Impl->CheckGL("Resolve weighted transparency")) return false;
        m_Impl->WeightedResolved = true;
    }
    else return m_Impl->Fail("Unknown OpenGL ES weighted transparency stage");
    ++m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount;
    return true;
}

bool OpenGLESRHI::ProcessMultisampling(Techniques::MultisamplePassKind pass)
{
    if (pass==Techniques::MultisamplePassKind::OpaqueResolve && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before opaque MSAA resolution");
    using Pass = Techniques::MultisamplePassKind;
    using Program = Impl::SampleProgram;
    if (!m_Impl->CheckThread("ProcessMultisampling") || !m_Impl->FrameOpen || m_Impl->PassOpen || m_Impl->FrameSampleCount <= 1)
        return m_Impl->Fail("OpenGL ES sample processing requires a multisample frame outside a raster pass");
    const bool material = m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive;
    const GLuint colorTarget = m_Impl->HdrActive ? m_Impl->HdrFramebuffers[0] : m_Impl->MainFramebuffer;
    switch (pass)
    {
    case Pass::OpaqueResolve:
        if (m_Impl->SamplesOpaqueResolved || (material && !m_Impl->GBufferWritten) ||
            !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardColorWritten))
            return m_Impl->Fail("OpenGL ES resolve opaque samples once after opaque lighting");
        if (!m_Impl->DrawSampleResolve(Program::Color, colorTarget) ||
            !m_Impl->DrawSampleResolve(material ? Program::Material : Program::Depth,
                material ? m_Impl->GBufferFramebuffer : m_Impl->MainFramebuffer)) return false;
        m_Impl->SamplesOpaqueResolved = m_Impl->SamplesColorResolved = true;
        break;
    case Pass::VelocityResolve:
        if (!m_Impl->SamplesOpaqueResolved || !m_Impl->VelocityCaptured || m_Impl->SamplesVelocityResolved ||
            m_Impl->TemporalResolved || m_Impl->MotionBlurResolved)
            return m_Impl->Fail("OpenGL ES resolve sample velocity once before motion blur");
        if (!m_Impl->DrawSampleResolve(Program::Velocity, m_Impl->MotionFramebuffers[Impl::MotionVelocity])) return false;
        m_Impl->SamplesVelocityResolved = true;
        break;
    case Pass::ColorRestore:
        if (!m_Impl->SamplesOpaqueResolved || !m_Impl->NeedsSampleColorRestore() || !m_Impl->OpaqueEffectsComplete() ||
            m_Impl->SamplesColorRestored || m_Impl->SamplesTransparentWritten)
            return m_Impl->Fail("OpenGL ES restore processed opaque radiance once before sample transparency");
        if (!m_Impl->DrawSampleResolve(Program::Restore, m_Impl->Multisample.Framebuffers[Impl::SampleColorFBO])) return false;
        m_Impl->SamplesColorRestored = true;
        break;
    case Pass::ColorResolve:
        if (m_Impl->WeightedActive && !m_Impl->WeightedResolved)
            return m_Impl->Fail("OpenGL ES resolve weighted transparency before the final MSAA color resolve");
        if (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesTransparentWritten || m_Impl->SamplesColorResolved)
            return m_Impl->Fail("OpenGL ES final sample resolve requires transparent composition");
        if (!m_Impl->DrawSampleResolve(Program::Color, colorTarget)) return false;
        m_Impl->SamplesColorResolved = true;
        break;
    default: return m_Impl->Fail("Unknown OpenGL ES sample resolve phase");
    }
    ++m_Impl->PendingFrame.Statistics.MultisamplePassCount;
    return true;
}

void OpenGLESRHI::ReleasePresentationContext() noexcept
{
    if(m_Impl->Display && m_Impl->Context && m_Impl->EglMakeCurrent && m_Impl->OwnerThread==std::this_thread::get_id())
    {
        m_Impl->EglMakeCurrent(m_Impl->Display,nullptr,nullptr,nullptr);m_Impl->ContextCurrent=false;
    }
}
bool OpenGLESRHI::Present()
{
    if (!m_Impl->CheckThread("Lumen frame publication")) return false;
    Impl::LumenPresentPublication lumenPublication{*m_Impl};
    Impl::NanitePresentPublication nanitePublication{*m_Impl};
    if (!m_Impl->CheckThread("TSR present boundary")) return false;
    if (m_Impl->FrameOpen && m_Impl->TSRScene.Active && (!m_Impl->TSRScene.OpaqueReady || !m_Impl->VelocityCaptured || (m_Impl->FrameSampleCount>1 && !m_Impl->SamplesVelocityResolved))) return m_Impl->Fail("TSR presentation requires completed opaque snapshot and physical motion");

    if (m_Impl->FrameOpen && m_Impl->FramePostProcessSettings.TemporalEffects.Enabled && !m_Impl->ExposureHistory.Resolved)
        return m_Impl->Fail("Complete temporal exposure before presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->FrameOpen && m_Impl->PendingFrame.Statistics.SubsurfacePassCount && !m_Impl->SubsurfaceResolved)
        return m_Impl->Fail("Incomplete subsurface diffusion must be aborted before presentation");
    if (m_Impl->PathFrame.Active && !m_Impl->PathFrame.Resolved) return m_Impl->Fail("Path tracing must resolve before presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
    if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
        return m_Impl->Fail("OpenGL ES frame cannot be presented in the current state");
    if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Assigned())
        return m_Impl->Fail("OpenGL ES Forward+ frame requires completed light assignment before presentation");
    if (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesColorResolved))
        return m_Impl->Fail("OpenGL ES MSAA must resolve opaque inputs and final color before presentation");
    if (m_Impl->DeferredActive && !m_Impl->DeferredResolved)
        return m_Impl->Fail("OpenGL ES deferred frame requires completed lighting before presentation");
    if (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved)
        return m_Impl->Fail("OpenGL ES ambient occlusion must complete before presentation");
    if (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved)
        return m_Impl->Fail("OpenGL ES SSR must complete before presentation");
    if (m_Impl->DepthOfFieldActive && !m_Impl->DepthOfFieldResolved)
        return m_Impl->Fail("OpenGL ES depth of field must complete before presentation");
    if (m_Impl->MotionBlurActive && !m_Impl->MotionBlurResolved)
        return m_Impl->Fail("OpenGL ES motion blur must complete before presentation");
    if (m_Impl->TemporalActive && !m_Impl->TemporalResolved)
        return m_Impl->Fail("OpenGL ES temporal reconstruction must complete before presentation");
    if (m_Impl->FogActive && !m_Impl->FogResolved)
        return m_Impl->Fail("OpenGL ES fog must resolve before presentation");
    if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved)
        return m_Impl->Fail("OpenGL ES light shafts must resolve before presentation");
    if (m_Impl->FramePeeling.Enabled && m_Impl->PeelingSequence.Stage()!=DepthPeelingStage::Complete)
        return m_Impl->Fail("Present requires completed OIT composition");
    if (m_Impl->RefractionActive && !m_Impl->RefractionResolved)
        return m_Impl->Fail("Present requires completed refraction composition");
    if (m_Impl->WeightedActive && !m_Impl->WeightedResolved)
        return m_Impl->Fail("OpenGL ES weighted transparency must resolve before presentation");
    if (m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
        return m_Impl->Fail("OpenGL ES contact visibility must complete before presentation");
    m_Impl->PendingFrame.Statistics.PassCount =
        static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size()) + m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount +
        m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount + m_Impl->PendingFrame.Statistics.MotionBlurPassCount
        + m_Impl->PendingFrame.Statistics.TemporalAAPassCount + m_Impl->PendingFrame.Statistics.TemporalEffectsPassCount + m_Impl->PendingFrame.Statistics.VolumetricFogPassCount
        + m_Impl->PendingFrame.Statistics.VolumetricLightingPassCount + m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount + m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount
        + m_Impl->PendingFrame.Statistics.LightShaftPassCount
        + m_Impl->PendingFrame.Statistics.ShadowMomentFilterPassCount
        + m_Impl->PendingFrame.Statistics.ContactShadowPassCount
        + m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount
            + m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount
            + m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount
        + m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount + m_Impl->PendingFrame.Statistics.DepthPeelingPassCount
        + m_Impl->PendingFrame.Statistics.MultisamplePassCount + m_Impl->PendingFrame.Statistics.HairFurPassCount + m_Impl->PendingFrame.Statistics.SubsurfacePassCount + m_Impl->PendingFrame.Statistics.PathTracingPassCount + m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount + m_Impl->PendingFrame.Statistics.ReflectionPassCount
        + m_Impl->PendingFrame.Statistics.EnvironmentConversionPassCount + m_Impl->PendingFrame.Statistics.PlanarMipPassCount;
    m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
    m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
    const bool captureDepth = IsDepthCaptureRequired() || (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) ||
        (!IsGpuPresentationEnabled() && m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0);
    GLuint resultFramebuffer = m_Impl->MainFramebuffer;
    m_Impl->PublishGpuContext();
    const bool gpuPresentation=IsGpuPresentationEnabled() && !captureDepth && !m_Impl->TSRScene.Active && GetDiagnosticBuffer()==Techniques::RenderBuffer::Color;
    if(!m_Impl->TSRScene.Active)
    {
        if(!m_Impl->ApplyHdr(m_Impl->FramePostProcessSettings.HDR) || !m_Impl->ApplyPostProcess(m_Impl->FramePostProcessSettings,resultFramebuffer))return false;
        if(gpuPresentation){if(!m_Impl->PublishGpuColor(resultFramebuffer))return false;}
        else if(!m_Impl->ReadColor(resultFramebuffer,m_Impl->PendingFrame.ColorRGBA8))return false;
    }
    if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) && m_Impl->VelocityActive && !m_Impl->ReadMotionVectors()) return false;
    if (captureDepth &&
        !m_Impl->ReadDepth(m_Impl->DepthTexture, m_Impl->CreateInfo.Width,
                           m_Impl->CreateInfo.Height, m_Impl->PendingFrame.Depth32))
        return false;
    if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) && m_Impl->LastShadowTexture)
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
            if (m_Impl->UsesVSM() && !m_Impl->ReadMoments(m_Impl->LastShadowTexture)) return false;
        }
    }
    if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
    {
        m_Impl->PendingFrame.CascadeDepth32.resize(m_Impl->FrameCascades.Frame.CascadeCount);
        for (std::size_t index = 0; index < m_Impl->PendingFrame.CascadeDepth32.size(); ++index)
            if (!m_Impl->ReadDepth(m_Impl->CascadeTextures[index], m_Impl->FrameCascades.MapResolution,
                m_Impl->FrameCascades.MapResolution, m_Impl->PendingFrame.CascadeDepth32[index])) return false;
    }
    if (!m_Impl->FinishTSRScene()) return false;
    m_Impl->CommitTSRScene();
    m_Impl->Hair.CommitPresent(m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height);
    m_Impl->ExposureHistory.Commit();
    const int diagnosticIndex=Techniques::MaterialBufferIndex(GetDiagnosticBuffer());
    const GLuint diagnostic=diagnosticIndex>=0?m_Impl->GBufferTextures[diagnosticIndex]:GetDiagnosticBuffer()==Techniques::RenderBuffer::AmbientOcclusion?m_Impl->AmbientOcclusionTextures[0]:0;
    if (diagnostic)
    {
        GLuint framebuffer=0;m_Impl->GlGenFramebuffers(1,&framebuffer);m_Impl->GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
        m_Impl->GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,diagnostic,0);m_Impl->GlReadBuffer(GL_COLOR_ATTACHMENT0);
        std::vector<Float4> pixels(std::size_t(m_Impl->CreateInfo.Width)*m_Impl->CreateInfo.Height);
        m_Impl->GlReadPixels(0,0,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,GL_RGBA,GL_FLOAT,pixels.data());
        m_Impl->GlDeleteFramebuffers(1,&framebuffer);
        if (!m_Impl->CheckGL("Read diagnostic buffer")) return false;
        auto& output=m_Impl->PendingFrame.DiagnosticPixels;output.resize(pixels.size());
        for (int y=0;y<m_Impl->CreateInfo.Height;++y) std::copy_n(pixels.data()+std::size_t(m_Impl->CreateInfo.Height-1-y)*m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Width,output.data()+std::size_t(y)*m_Impl->CreateInfo.Width);
    }
    ApplyDiagnosticCapture(m_Impl->PendingFrame,GetDiagnosticBuffer());
    m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
    m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
    m_Impl->FrameOpen = false;
    m_Impl->PresentToWindow(m_Impl->LastFrame);
    // An aborted resolve may have overwritten the pending slot, but never the
    // committed slot. Publish only after successful readback/presentation.
    if (m_Impl->TemporalResolved)
    {
        m_Impl->TemporalHistoryReadIndex = 1 - m_Impl->TemporalHistoryReadIndex;
        m_Impl->TemporalHistoryValid = true;
    }
    else m_Impl->TemporalHistoryValid = false;
    m_Impl->PathFrame.Commit();
        if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI)) m_Impl->GI.Commit(); else m_Impl->GI.Invalidate();
    if (m_Impl->ProbeActive) m_Impl->Probes.Commit();
    lumenPublication.Published=true; nanitePublication.Published=true;
    if(m_Impl->EglMakeCurrent(m_Impl->Display,nullptr,nullptr,nullptr)==EGL_FALSE_VALUE)return m_Impl->Fail("Cannot release the GL render context");
    m_Impl->ContextCurrent=false;
    return true;
}

FrameCapture OpenGLESRHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
FrameCapture OpenGLESRHI::TakeLastFrameCapture() { return std::move(m_Impl->LastFrame); }
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
    bool OpenGLESRHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload&,VirtualGeometryCacheHandle,VirtualGeometryCacheHandle&,VirtualGeometryCacheStatistics&) { return false; }
bool OpenGLESRHI::PrepareNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheHandle&,NaniteCacheStatistics&) { return false; }
bool OpenGLESRHI::SubmitNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheTask&,NaniteCacheStatistics&) { return false; }
bool OpenGLESRHI::PollNaniteCache(NaniteCacheTask,bool&,NaniteCacheHandle&) { return false; }
bool OpenGLESRHI::CancelNaniteCache(NaniteCacheTask) { return false; }
    bool OpenGLESRHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle) { return false; }
bool OpenGLESRHI::DestroyNaniteCache(NaniteCacheHandle) { return false; }
    bool OpenGLESRHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload&,VirtualShadowRequestResult&) { return false; }
    bool OpenGLESRHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle,const VirtualShadowLookupUpload&,VirtualShadowLookupResult&) { return false; }
    bool OpenGLESRHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload&,VirtualShadowAtlasHandle,VirtualShadowAtlasResult&) { return false; }
    bool OpenGLESRHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle,std::vector<float>&) { return false; }
    bool OpenGLESRHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle) { return false; }
    bool OpenGLESRHI::CreateLumenDistanceScene(const LumenDistanceSceneData&,LumenDistanceSceneHandle&) { return false; }
    bool OpenGLESRHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle&,LumenDistanceSceneHandle) { return false; }
    bool OpenGLESRHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle,LumenGlobalDistanceTask&) { return false; }
    bool OpenGLESRHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask,unsigned,bool& ready,LumenDistanceSceneHandle&,LumenGlobalDistanceStatistics&) { ready=false; return false; }
    bool OpenGLESRHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask) { return false; }
    bool OpenGLESRHI::TraceLumenDistanceScene(LumenDistanceSceneHandle,const LumenDistanceTraceUpload&,LumenDistanceTraceResult&) { return false; }
    bool OpenGLESRHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle) { return false; }
    bool OpenGLESRHI::RelightLumenSurface(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenSurfaceLightingResult&) { return false; }
    bool OpenGLESRHI::BeginLumenLightingUpdate(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenLightingTask&) { return false; }
    bool OpenGLESRHI::AdvanceLumenLightingUpdate(LumenLightingTask,unsigned,bool& ready,LumenSurfaceLightingResult&,LumenLightingStatistics&) { ready=false; return false; }
    bool OpenGLESRHI::CancelLumenLightingUpdate(LumenLightingTask) { return false; }
    bool OpenGLESRHI::ReadLumenLighting(LumenLightingHandle,std::vector<LumenLightingTexel>&) { return false; }
    bool OpenGLESRHI::DestroyLumenLighting(LumenLightingHandle) { return false; }
    bool OpenGLESRHI::TraceLumenRadiance(LumenLightingHandle,const LumenRadianceTraceUpload&,LumenRadianceTraceResult&) { return false; }
    bool OpenGLESRHI::ReadLumenRadiance(LumenRadianceHandle,std::vector<LumenRadianceSample>&) { return false; }
    bool OpenGLESRHI::AssembleLumenRadiance(const LumenRadianceAssembly&,LumenRadianceTraceResult&) { return false; }
    bool OpenGLESRHI::ReadLumenProbeOrigins(LumenRadianceHandle,std::vector<LumenProbeOrigin>&) { return false; }
    bool OpenGLESRHI::DestroyLumenRadiance(LumenRadianceHandle) { return false; }
    bool OpenGLESRHI::IntegrateLumenProbes(LumenRadianceHandle,const LumenProbeIntegrationUpload&,LumenProbeIntegrationResult&) { return false; }
    bool OpenGLESRHI::ResolveLumenProbes(LumenProbeHandle,const LumenProbeGatherSettings&) { return false; }
    bool OpenGLESRHI::ReadLumenProbes(LumenProbeHandle,std::vector<LumenProbeIrradiance>&) { return false; }
    bool OpenGLESRHI::DestroyLumenProbes(LumenProbeHandle) { return false; }
    bool OpenGLESRHI::LookupLumenSurface(LumenSurfaceHandle,const LumenSurfaceLookupUpload&,LumenSurfaceLookupResult&) { return false; }
    bool OpenGLESRHI::CaptureLumenSurface(const LumenSurfaceUpload&,LumenSurfaceResult&) { return false; }
    bool OpenGLESRHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload&,LumenCaptureTask&) { return false; }
    bool OpenGLESRHI::AdvanceLumenCaptureUpdate(LumenCaptureTask,unsigned,bool& ready,LumenSurfaceResult&,LumenCaptureStatistics&) { ready=false; return false; }
    bool OpenGLESRHI::CancelLumenCaptureUpdate(LumenCaptureTask) { return false; }
    bool OpenGLESRHI::ReadLumenSurface(LumenSurfaceHandle,std::vector<LumenSurfaceTexel>&) { return false; }
    bool OpenGLESRHI::DestroyLumenSurface(LumenSurfaceHandle) { return false; }
    bool OpenGLESRHI::DestroyTSRHistory(TSRHistoryHandle) { return false; }
    bool OpenGLESRHI::RenderTSRDisplay(TSRHistoryHandle,const Techniques::PostProcessSettings&,FrameCapture&) { return false; }
    bool OpenGLESRHI::SnapshotTSROpaque() { return false; }
    bool OpenGLESRHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle,const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool OpenGLESRHI::CreateCachedNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
bool OpenGLESRHI::SubmitNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryTask&) { return false; }
bool OpenGLESRHI::PollNaniteGeometry(NaniteGeometryTask,bool&,NaniteGeometryBuffers&) { return false; }
bool OpenGLESRHI::CancelNaniteGeometry(NaniteGeometryTask) { return false; }
    bool OpenGLESRHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool OpenGLESRHI::PrepareNaniteDraws(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool OpenGLESRHI::PrepareNaniteDrawBins(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool OpenGLESRHI::DrawNaniteBins(NaniteDrawHandle) { return false; }
bool OpenGLESRHI::DrawNaniteCluster(NaniteDrawHandle,unsigned) { return false; }
bool OpenGLESRHI::CaptureNaniteDepth(NaniteDepthHandle&) { return false; }
bool OpenGLESRHI::BeginNaniteVisibility(NaniteDepthHandle,const std::vector<NaniteVisibilityQuery>&,NaniteVisibilityHandle&) { return false; }
bool OpenGLESRHI::RetestNaniteVisibility(NaniteVisibilityHandle,NaniteDepthHandle,NaniteVisibilityHandle&) { return false; }
bool OpenGLESRHI::ReadNaniteDepth(NaniteDepthHandle,std::vector<float>&) { return false; }
bool OpenGLESRHI::ReadNaniteVisibility(NaniteVisibilityHandle,std::vector<NaniteVisibilityState>&) { return false; }
bool OpenGLESRHI::DestroyNaniteDepth(NaniteDepthHandle) { return false; }
bool OpenGLESRHI::DestroyNaniteVisibility(NaniteVisibilityHandle) { return false; }
bool OpenGLESRHI::CreateNaniteGeometry(const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
    bool OpenGLESRHI::CreateTessellatedGeometry(const PNTessellationGeometry&,const Techniques::TessellationSettings&,TessellatedGeometryBuffers&) { return false; }
    bool OpenGLESRHI::CreateDisplacedVertexBuffer(const DisplacementGeometry&,const Techniques::DisplacementMappingSettings&,DisplacedVertexBuffer&) { return false; }
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
bool OpenGLESRHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc&, GpuDrivenBatchHandle&) { return false; }
bool OpenGLESRHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle) { return false; }
bool OpenGLESRHI::RenderVisibilityBuffer(const VisibilityBufferDesc&) { return false; }
bool OpenGLESRHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
bool OpenGLESRHI::DrawIndexedInstanced(std::uint32_t, std::uint32_t, std::int32_t, const std::vector<DrawInstance>&) { return false; }
bool OpenGLESRHI::EndPass() { return false; }
bool OpenGLESRHI::PrepareLightListFrame(const LightListFrameDesc&) { return false; }
bool OpenGLESRHI::GenerateLightLists() { return false; }
bool OpenGLESRHI::DrawDeferredLighting(const DeferredLightingDesc&) { return false; }
    bool OpenGLESRHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc&) { return false; }
    bool OpenGLESRHI::DrawHairFur(const HairFurDesc&) { return false; }
    bool OpenGLESRHI::DrawHairFurInPass(const HairFurDesc&) { return false; }
    bool OpenGLESRHI::DrawHairFurVelocity(const HairFurVelocityDesc&) { return false; }
bool OpenGLESRHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc&) { return false; }
bool OpenGLESRHI::CompleteEnvironmentCapture() { return false; }
bool OpenGLESRHI::CompletePlanarCapture() { return false; }
bool OpenGLESRHI::ApplyPathTracing(const PathTracingDesc&) { return false; }
    bool OpenGLESRHI::ApplyGlobalIllumination(const GlobalIlluminationDesc&) { return false; }
bool OpenGLESRHI::ApplyReflections(const ReflectionDesc&) { return false; }
bool OpenGLESRHI::ApplyDepthOfField(const DepthOfFieldDesc&) { return false; }
bool OpenGLESRHI::ApplyMotionBlur(const MotionBlurDesc&) { return false; }
bool OpenGLESRHI::ApplyTemporalAA(const TemporalAADesc&) { return false; }
    bool OpenGLESRHI::ApplyTemporalEffects(const TemporalEffectsDesc&) { return false; }
bool OpenGLESRHI::ApplyVolumetricFog(const VolumetricFogDesc&) { return false; }
bool OpenGLESRHI::ApplyLightShafts(const LightShaftDesc&) { return false; }
bool OpenGLESRHI::ProcessMultisampling(Techniques::MultisamplePassKind) { return false; }
bool OpenGLESRHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind) { return false; }
bool OpenGLESRHI::ProcessDepthPeeling(DepthPeelingPass) { return false; }
bool OpenGLESRHI::ProcessRefraction(const RefractionDesc&) { return false; }
void OpenGLESRHI::ReleasePresentationContext() noexcept {}
bool OpenGLESRHI::Present() { return false; }
bool OpenGLESRHI::GenerateContactShadows(const ContactShadowDesc&) { return false; }
FrameCapture OpenGLESRHI::GetLastFrameCapture() const { return {}; }
FrameCapture OpenGLESRHI::TakeLastFrameCapture() { return {}; }
std::string OpenGLESRHI::GetLastError() const { return m_Impl->LastError; }
#endif
} // namespace Evans::Renderer
