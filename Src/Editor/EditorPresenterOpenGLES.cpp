#include "Renderer/Types.h"
#include "Renderer/OpenGLES/OpenGLESProgramCache.h"
#include "Editor/EditorPresenterOpenGLES.h"

#if defined(_WIN32)

#include "Editor/ImguiRenderOpenGLES.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace Evans::Editor
{
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
using EGLNativeWindowType = void*;

constexpr EGLBoolean EGL_FALSE_VALUE = 0;
constexpr EGLint EGL_NONE_VALUE = 0x3038;
constexpr EGLint EGL_RED_SIZE = 0x3024;
constexpr EGLint EGL_GREEN_SIZE = 0x3023;
constexpr EGLint EGL_BLUE_SIZE = 0x3022;
constexpr EGLint EGL_ALPHA_SIZE = 0x3021;
constexpr EGLint EGL_SURFACE_TYPE = 0x3033;
constexpr EGLint EGL_WINDOW_BIT = 0x0004;
constexpr EGLint EGL_RENDERABLE_TYPE = 0x3040;
constexpr EGLint EGL_OPENGL_ES3_BIT = 0x0040;
constexpr EGLint EGL_CONTEXT_CLIENT_VERSION = 0x3098;
constexpr EGLenum EGL_OPENGL_ES_API = 0x30A0;

using PFNEGLGETDISPLAY = EGLDisplay(EVANS_EGL_CALL*)(EGLNativeDisplayType);
using PFNEGLINITIALIZE = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLint*, EGLint*);
using PFNEGLCHOOSECONFIG = EGLBoolean(EVANS_EGL_CALL*)(
    EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*);
using PFNEGLBINDAPI = EGLBoolean(EVANS_EGL_CALL*)(EGLenum);
using PFNEGLCREATEWINDOWSURFACE = EGLSurface(EVANS_EGL_CALL*)(
    EGLDisplay, EGLConfig, EGLNativeWindowType, const EGLint*);
using PFNEGLCREATECONTEXT = EGLContext(EVANS_EGL_CALL*)(
    EGLDisplay, EGLConfig, EGLContext, const EGLint*);
using PFNEGLMAKECURRENT = EGLBoolean(EVANS_EGL_CALL*)(
    EGLDisplay, EGLSurface, EGLSurface, EGLContext);
using PFNEGLSWAPBUFFERS = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLSurface);
using PFNEGLSWAPINTERVAL = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLint);
using PFNEGLDESTROYCONTEXT = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLContext);
using PFNEGLDESTROYSURFACE = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay, EGLSurface);
using PFNEGLTERMINATE = EGLBoolean(EVANS_EGL_CALL*)(EGLDisplay);
using PFNEGLGETERROR = EGLint(EVANS_EGL_CALL*)();
using PFNEGLGETPROCADDRESS = void*(EVANS_EGL_CALL*)(const char*);

using GLenum = unsigned int;
using GLboolean = unsigned char;
using GLbitfield = unsigned int;
using GLsizei = int;
using GLint = int;
using GLuint = unsigned int;
using GLfloat = float;
using GLchar = char;
using GLsizeiptr = std::ptrdiff_t;

constexpr GLboolean GL_FALSE_VALUE = 0;
constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_VERSION = 0x1F02;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_DYNAMIC_DRAW = 0x88E8;
constexpr GLenum GL_FLOAT_VALUE = 0x1406;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
constexpr GLenum GL_TEXTURE0 = 0x84C0;
constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
constexpr GLenum GL_LINEAR = 0x2601;
constexpr GLenum GL_CLAMP_TO_EDGE = 0x812F;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_RGBA8 = 0x8058;
constexpr GLbitfield GL_COLOR_BUFFER_BIT = 0x00004000;
constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_CULL_FACE = 0x0B44;
constexpr GLenum GL_BLEND = 0x0BE2;
constexpr GLenum GL_SCISSOR_TEST = 0x0C11;
constexpr GLenum GL_UNPACK_ALIGNMENT = 0x0CF5;

using PFNGLGETERROR = GLenum(EVANS_GL_CALL*)();
using PFNGLGETSTRING = const unsigned char*(EVANS_GL_CALL*)(GLenum);
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
using PFNGLUNIFORM1I = void(EVANS_GL_CALL*)(GLint, GLint);
using PFNGLGENBUFFERS = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDBUFFER = void(EVANS_GL_CALL*)(GLenum, GLuint);
using PFNGLBUFFERDATA = void(EVANS_GL_CALL*)(GLenum, GLsizeiptr, const void*, GLenum);
using PFNGLDELETEBUFFERS = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLGENVERTEXARRAYS = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDVERTEXARRAY = void(EVANS_GL_CALL*)(GLuint);
using PFNGLDELETEVERTEXARRAYS = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLENABLEVERTEXATTRIBARRAY = void(EVANS_GL_CALL*)(GLuint);
using PFNGLVERTEXATTRIBPOINTER = void(EVANS_GL_CALL*)(
    GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
using PFNGLGENTEXTURES = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDTEXTURE = void(EVANS_GL_CALL*)(GLenum, GLuint);
using PFNGLTEXIMAGE2D = void(EVANS_GL_CALL*)(
    GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
using PFNGLTEXPARAMETERI = void(EVANS_GL_CALL*)(GLenum, GLenum, GLint);
using PFNGLDELETETEXTURES = void(EVANS_GL_CALL*)(GLsizei, const GLuint*);
using PFNGLACTIVETEXTURE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLVIEWPORT = void(EVANS_GL_CALL*)(GLint, GLint, GLsizei, GLsizei);
using PFNGLCLEARCOLOR = void(EVANS_GL_CALL*)(GLfloat, GLfloat, GLfloat, GLfloat);
using PFNGLCLEAR = void(EVANS_GL_CALL*)(GLbitfield);
using PFNGLDISABLE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLDRAWARRAYS = void(EVANS_GL_CALL*)(GLenum, GLint, GLsizei);
using PFNGLPIXELSTOREI = void(EVANS_GL_CALL*)(GLenum, GLint);

struct ViewportVertex
{
    float Position[2];
    float TexCoord[2];
};

std::string HexError(const char* operation, unsigned long value)
{
    std::ostringstream stream;
    stream << operation << " failed with error 0x" << std::hex << std::uppercase << value;
    return stream.str();
}

std::wstring ParentDirectory(std::wstring path)
{
    const std::size_t separator = path.find_last_of(L"\\/");
    if (separator == std::wstring::npos) return {};
    path.resize(separator);
    return path;
}

std::wstring ModuleDirectory()
{
    std::array<wchar_t, 32768> path{};
    const DWORD length = GetModuleFileNameW(
        nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) return {};
    return ParentDirectory(std::wstring(path.data(), length));
}

void AppendVersionDirectories(
    const std::wstring& root, std::vector<std::wstring>& directories)
{
    if (root.empty()) return;
    WIN32_FIND_DATAW data{};
    const std::wstring pattern = root + L"\\*";
    HANDLE search = FindFirstFileW(pattern.c_str(), &data);
    if (search == INVALID_HANDLE_VALUE) return;
    do
    {
        if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
            std::wcscmp(data.cFileName, L".") == 0 ||
            std::wcscmp(data.cFileName, L"..") == 0)
        {
            continue;
        }
        directories.push_back(root + L"\\" + data.cFileName);
    } while (FindNextFileW(search, &data) != FALSE);
    FindClose(search);
}

std::vector<std::wstring> AngleDirectories()
{
    std::vector<std::wstring> result;
    result.push_back(ModuleDirectory());

    std::vector<std::wstring> installedRuntimes;

    std::array<wchar_t, 32768> path{};
    DWORD length = GetEnvironmentVariableW(
        L"ProgramFiles(x86)", path.data(), static_cast<DWORD>(path.size()));
    if (length > 0 && length < path.size())
    {
        const std::wstring programFiles(path.data(), length);
        AppendVersionDirectories(
            programFiles + L"\\Microsoft\\EdgeWebView\\Application", installedRuntimes);
        AppendVersionDirectories(
            programFiles + L"\\Microsoft\\Edge\\Application", installedRuntimes);
    }
    std::sort(installedRuntimes.rbegin(), installedRuntimes.rend());
    result.insert(result.end(), installedRuntimes.begin(), installedRuntimes.end());
    const UINT systemLength = GetSystemDirectoryW(
        path.data(), static_cast<UINT>(path.size()));
    if (systemLength > 0 && systemLength < path.size())
        result.push_back(
            std::wstring(path.data(), systemLength) + L"\\Microsoft-Edge-WebView");
    return result;
}

} // namespace

struct EditorPresenterOpenGLES::Impl
{
    ~Impl()
    {
        Shutdown();
    }

    bool Fail(std::string message)
    {
        LastError = std::move(message);
        return false;
    }

    template <typename Function>
    bool LoadEGL(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetProcAddress(EGLModule, name));
        if (destination != nullptr) return true;
        return Fail(std::string("ANGLE EGL entry point is missing: ") + name);
    }

    template <typename Function>
    bool LoadGL(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(ResolveGL(name));
        if (destination != nullptr) return true;
        return Fail(std::string("OpenGL ES 3 entry point is missing: ") + name);
    }

    static void* ResolveForImgui(void* userData, const char* name)
    {
        return static_cast<Impl*>(userData)->ResolveGL(name);
    }

    void* ResolveGL(const char* name) const
    {
        void* address = reinterpret_cast<void*>(GetProcAddress(GLESModule, name));
        if (address == nullptr && EglGetProcAddress != nullptr)
            address = EglGetProcAddress(name);
        return address;
    }

    bool LoadLibrariesFrom(const std::wstring& directory)
    {
        if (directory.empty()) return false;
        constexpr DWORD flags =
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
        const std::wstring glesPath = directory + L"\\libGLESv2.dll";
        const std::wstring eglPath = directory + L"\\libEGL.dll";
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
        for (const std::wstring& directory : AngleDirectories())
        {
            if (LoadLibrariesFrom(directory)) return true;
        }
        return Fail(
            "OpenGL ES presentation is unavailable: libEGL.dll and "
            "libGLESv2.dll (ANGLE) were not found beside EvansEditor or in "
            "the installed Microsoft Edge/WebView runtime.");
    }

    bool LoadEGLFunctions()
    {
        return LoadEGL(EglGetDisplay, "eglGetDisplay") &&
               LoadEGL(EglInitialize, "eglInitialize") &&
               LoadEGL(EglChooseConfig, "eglChooseConfig") &&
               LoadEGL(EglBindAPI, "eglBindAPI") &&
               LoadEGL(EglCreateWindowSurface, "eglCreateWindowSurface") &&
               LoadEGL(EglCreateContext, "eglCreateContext") &&
               LoadEGL(EglMakeCurrent, "eglMakeCurrent") &&
               LoadEGL(EglSwapBuffers, "eglSwapBuffers") &&
               LoadEGL(EglSwapInterval, "eglSwapInterval") &&
               LoadEGL(EglDestroyContext, "eglDestroyContext") &&
               LoadEGL(EglDestroySurface, "eglDestroySurface") &&
               LoadEGL(EglTerminate, "eglTerminate") &&
               LoadEGL(EglGetError, "eglGetError") &&
               LoadEGL(EglGetProcAddress, "eglGetProcAddress");
    }

    bool CreateWindowContext(HWND window)
    {
        Display = SharedDisplay ? SharedDisplay : EglGetDisplay(nullptr);
        EGLint major = 0;
        EGLint minor = 0;
        if (Display == nullptr ||
            EglInitialize(Display, &major, &minor) == EGL_FALSE_VALUE)
        {
            return Fail(HexError(
                "eglInitialize(ANGLE)", static_cast<unsigned long>(EglGetError())));
        }
        const EGLint configAttributes[] = {
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_NONE_VALUE};
        EGLint count = 0;
        if (EglChooseConfig(
                Display, configAttributes, &Config, 1, &count) == EGL_FALSE_VALUE ||
            count == 0)
        {
            return Fail(HexError(
                "eglChooseConfig(window, ES3)",
                static_cast<unsigned long>(EglGetError())));
        }
        if (EglBindAPI(EGL_OPENGL_ES_API) == EGL_FALSE_VALUE)
            return Fail(HexError(
                "eglBindAPI(OpenGL ES)", static_cast<unsigned long>(EglGetError())));
        const EGLint surfaceAttributes[] = {EGL_NONE_VALUE};
        // WGL pixel formats are fixed for the lifetime of an HWND. Give ANGLE
        // its own surface window even when Vulkan previously used the parent.
        RECT client{};
        GetClientRect(window, &client);
        WNDCLASSW surfaceClass{};
        surfaceClass.style = CS_OWNDC;
        surfaceClass.lpfnWndProc = DefWindowProcW;
        surfaceClass.hInstance = GetModuleHandleW(nullptr);
        surfaceClass.lpszClassName = L"EvansOpenGLESSurface";
        if (!RegisterClassW(&surfaceClass) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
            return Fail("Cannot register the OpenGL ES presentation window");
        SurfaceWindow = CreateWindowExW(0, surfaceClass.lpszClassName, L"",
            WS_CHILD | WS_VISIBLE | WS_DISABLED | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
            0, 0, client.right, client.bottom, window, nullptr, surfaceClass.hInstance, nullptr);
        if (!SurfaceWindow) return Fail("Cannot create the OpenGL ES presentation window");
        Surface = EglCreateWindowSurface(
            Display, Config, reinterpret_cast<EGLNativeWindowType>(SurfaceWindow),
            surfaceAttributes);
        if (Surface == nullptr)
            return Fail(HexError(
                "eglCreateWindowSurface", static_cast<unsigned long>(EglGetError())));
        const EGLint contextAttributes[] = {
            EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE_VALUE};
        Context = SharedContext ? SharedContext : EglCreateContext(Display, Config, nullptr, contextAttributes);
        if (Context == nullptr)
            return Fail(HexError(
                "eglCreateContext(OpenGL ES 3)",
                static_cast<unsigned long>(EglGetError())));
        if (EglMakeCurrent(Display, Surface, Surface, Context) == EGL_FALSE_VALUE)
            return Fail(HexError(
                "eglMakeCurrent", static_cast<unsigned long>(EglGetError())));
        EglSwapInterval(Display, 0);
        return true;
    }

    bool LoadGLFunctions()
    {
#define EVANS_LOAD_GL(Member, Name) LoadGL(Member, Name)
        return EVANS_LOAD_GL(GlGetError, "glGetError") &&
               EVANS_LOAD_GL(GlGetString, "glGetString") &&
               EVANS_LOAD_GL(GlCreateShader, "glCreateShader") &&
               EVANS_LOAD_GL(GlShaderSource, "glShaderSource") &&
               EVANS_LOAD_GL(GlCompileShader, "glCompileShader") &&
               EVANS_LOAD_GL(GlGetShaderiv, "glGetShaderiv") &&
               EVANS_LOAD_GL(GlGetShaderInfoLog, "glGetShaderInfoLog") &&
               EVANS_LOAD_GL(GlDeleteShader, "glDeleteShader") &&
               EVANS_LOAD_GL(GlCreateProgram, "glCreateProgram") &&
               EVANS_LOAD_GL(GlAttachShader, "glAttachShader") &&
               EVANS_LOAD_GL(GlLinkProgram, "glLinkProgram") &&
               EVANS_LOAD_GL(GlGetProgramiv, "glGetProgramiv") &&
               EVANS_LOAD_GL(GlGetProgramInfoLog, "glGetProgramInfoLog") &&
               EVANS_LOAD_GL(GlDeleteProgram, "glDeleteProgram") &&
               EVANS_LOAD_GL(GlUseProgram, "glUseProgram") &&
               EVANS_LOAD_GL(GlGetUniformLocation, "glGetUniformLocation") &&
               EVANS_LOAD_GL(GlUniform1i, "glUniform1i") &&
               EVANS_LOAD_GL(GlGenBuffers, "glGenBuffers") &&
               EVANS_LOAD_GL(GlBindBuffer, "glBindBuffer") &&
               EVANS_LOAD_GL(GlBufferData, "glBufferData") &&
               EVANS_LOAD_GL(GlDeleteBuffers, "glDeleteBuffers") &&
               EVANS_LOAD_GL(GlGenVertexArrays, "glGenVertexArrays") &&
               EVANS_LOAD_GL(GlBindVertexArray, "glBindVertexArray") &&
               EVANS_LOAD_GL(GlDeleteVertexArrays, "glDeleteVertexArrays") &&
               EVANS_LOAD_GL(GlEnableVertexAttribArray, "glEnableVertexAttribArray") &&
               EVANS_LOAD_GL(GlVertexAttribPointer, "glVertexAttribPointer") &&
               EVANS_LOAD_GL(GlGenTextures, "glGenTextures") &&
               EVANS_LOAD_GL(GlBindTexture, "glBindTexture") &&
               EVANS_LOAD_GL(GlTexImage2D, "glTexImage2D") &&
               EVANS_LOAD_GL(GlTexParameteri, "glTexParameteri") &&
               EVANS_LOAD_GL(GlDeleteTextures, "glDeleteTextures") &&
               EVANS_LOAD_GL(GlActiveTexture, "glActiveTexture") &&
               EVANS_LOAD_GL(GlViewport, "glViewport") &&
               EVANS_LOAD_GL(GlClearColor, "glClearColor") &&
               EVANS_LOAD_GL(GlClear, "glClear") &&
               EVANS_LOAD_GL(GlDisable, "glDisable") &&
               EVANS_LOAD_GL(GlDrawArrays, "glDrawArrays") &&
               EVANS_LOAD_GL(GlPixelStorei, "glPixelStorei");
#undef EVANS_LOAD_GL
    }

    bool CheckGL(const char* operation)
    {
        const GLenum error = GlGetError();
        return error == GL_NO_ERROR || Fail(HexError(operation, error));
    }

    GLuint Compile(GLenum type, const char* source)
    {
        const GLuint shader = GlCreateShader(type);
        if (shader == 0)
        {
            Fail("glCreateShader returned zero while creating the editor viewport shader.");
            return 0;
        }
        GlShaderSource(shader, 1, &source, nullptr);
        GlCompileShader(shader);
        GLint compiled = 0;
        GlGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (compiled != 0) return shader;
        GLint logLength = 0;
        GlGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::vector<char> log(
            static_cast<std::size_t>(std::max(logLength, 1)), '\0');
        GlGetShaderInfoLog(
            shader, static_cast<GLsizei>(log.size()), nullptr, log.data());
        LastError = std::string("OpenGL ES viewport shader compilation failed: ") + log.data();
        GlDeleteShader(shader);
        return 0;
    }

    bool CreateViewportPipeline()
    {
        static constexpr const char* vertexSource = R"GLSL(#version 300 es
precision highp float;
layout(location=0) in vec2 InPosition;
layout(location=1) in vec2 InTexCoord;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(InPosition, 0.0, 1.0);
    TexCoord = InTexCoord;
}
)GLSL";
        static constexpr const char* fragmentSource = R"GLSL(#version 300 es
precision highp float;
in vec2 TexCoord;
uniform sampler2D ViewportTexture;
uniform int NativeGpuImage;
out vec4 OutColor;
void main()
{
    OutColor = texture(ViewportTexture, TexCoord);
    if(NativeGpuImage!=0)
        OutColor.rgb=mix(12.92*OutColor.rgb,1.055*pow(max(OutColor.rgb,vec3(0)),vec3(1.0/2.4))-0.055,step(vec3(0.0031308),OutColor.rgb));
}
)GLSL";
        ViewportProgram=Renderer::CompileCachedGLESProgram([this](const char* name) {
            auto* address=reinterpret_cast<void*>(GetProcAddress(GLESModule,name));
            return address?address:EglGetProcAddress(name);
        },{{GL_VERTEX_SHADER,vertexSource},{GL_FRAGMENT_SHADER,fragmentSource}},LastError);
        if(!ViewportProgram)return false;
        const GLint textureLocation =
            GlGetUniformLocation(ViewportProgram, "ViewportTexture");
        if (textureLocation < 0)
            return Fail("OpenGL ES viewport texture uniform is missing.");
        GlUseProgram(ViewportProgram);
        GlUniform1i(textureLocation, 0);

        GlGenVertexArrays(1, &ViewportVertexArray);
        GlGenBuffers(1, &ViewportVertexBuffer);
        GlBindVertexArray(ViewportVertexArray);
        GlBindBuffer(GL_ARRAY_BUFFER, ViewportVertexBuffer);
        GlEnableVertexAttribArray(0);
        GlVertexAttribPointer(
            0, 2, GL_FLOAT_VALUE, GL_FALSE_VALUE, sizeof(ViewportVertex),
            reinterpret_cast<const void*>(offsetof(ViewportVertex, Position)));
        GlEnableVertexAttribArray(1);
        GlVertexAttribPointer(
            1, 2, GL_FLOAT_VALUE, GL_FALSE_VALUE, sizeof(ViewportVertex),
            reinterpret_cast<const void*>(offsetof(ViewportVertex, TexCoord)));
        return ViewportVertexArray != 0 && ViewportVertexBuffer != 0 &&
               CheckGL("CreateViewportPipeline");
    }

    void ConfigureTexture()
    {
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    std::shared_ptr<void> SharedGroup;
    std::shared_ptr<std::mutex> SharedAccess;
    std::unique_lock<std::mutex> FrameAccess;
    EGLDisplay SharedDisplay=nullptr;
    EGLContext SharedContext=nullptr;
    bool GpuViewport=false,SkipTerminate=false;
    std::shared_ptr<const Renderer::GpuImage> GpuFrameLease;
    bool AdoptGpuGroup(const Image& image)
    {
        const auto gpu=image.GpuColor?image.GpuColor:image.GpuContext;
        if(!gpu || SharedGroup==gpu->ContextOwner)
            return Context ? true : InitializeNow(Window,Width,Height);
        if(gpu->Backend!=Renderer::RHIBackend::OpenGLES)return Fail("Incompatible GPU viewport backend");
        const auto window=Window;const auto width=Width,height=Height;
        SkipTerminate=Display==gpu->NativeDisplay;
        Shutdown();
        SharedDisplay=gpu->NativeDisplay;SharedContext=gpu->NativeContext;SharedGroup=gpu->ContextOwner;
        return InitializeNow(window,width,height);
    }

    bool EnsureViewportTexture(const Image& image, std::uint64_t revision)
    {
        if (!image.IsValid()) return false;
        if(image.GpuColor)
        {
            if(!GpuViewport && ViewportTexture)GlDeleteTextures(1,&ViewportTexture);
            ViewportTexture=GLuint(reinterpret_cast<std::uintptr_t>(image.GpuColor->NativeTexture));
            GpuViewport=true;GpuFrameLease=image.GpuColor;
            ImageWidth=image.Width;ImageHeight=image.Height;UploadedRevision=revision;return true;
        }
        if(GpuViewport){ViewportTexture=0;GpuViewport=false;GpuFrameLease.reset();}
        if (ViewportTexture == 0)
        {
            GlGenTextures(1, &ViewportTexture);
            if (ViewportTexture == 0)
                return Fail("glGenTextures returned zero for the editor viewport.");
        }
        if (UploadedRevision == revision &&
            ImageWidth == image.Width && ImageHeight == image.Height)
        {
            return true;
        }
        GlActiveTexture(GL_TEXTURE0);
        GlBindTexture(GL_TEXTURE_2D, ViewportTexture);
        ConfigureTexture();
        GlPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        GlTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8,
            static_cast<GLsizei>(image.Width), static_cast<GLsizei>(image.Height),
            0, GL_RGBA, GL_UNSIGNED_BYTE, image.PixelsRGBA8.data());
        if (!CheckGL("Upload editor viewport texture")) return false;
        ImageWidth = image.Width;
        ImageHeight = image.Height;
        UploadedRevision = revision;
        return true;
    }

    void DrawViewport(int x, int y, int width, int height)
    {
        if (ViewportTexture == 0 || width <= 0 || height <= 0) return;
        const float left = static_cast<float>(x) * 2.0f / Width - 1.0f;
        const float right = static_cast<float>(x + width) * 2.0f / Width - 1.0f;
        const float top = 1.0f - static_cast<float>(y) * 2.0f / Height;
        const float bottom = 1.0f - static_cast<float>(y + height) * 2.0f / Height;
        ViewportVertex vertices[] = {
            {{left, top}, {0.0f, 0.0f}},
            {{right, top}, {1.0f, 0.0f}},
            {{right, bottom}, {1.0f, 1.0f}},
            {{left, top}, {0.0f, 0.0f}},
            {{right, bottom}, {1.0f, 1.0f}},
            {{left, bottom}, {0.0f, 1.0f}}};
        if(GpuViewport)for(auto& vertex:vertices)vertex.TexCoord[1]=1.0f-vertex.TexCoord[1];
        GlUseProgram(ViewportProgram);
        GlUniform1i(GlGetUniformLocation(ViewportProgram,"NativeGpuImage"),GpuViewport?1:0);
        GlBindVertexArray(ViewportVertexArray);
        GlBindBuffer(GL_ARRAY_BUFFER, ViewportVertexBuffer);
        GlBufferData(
            GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
        GlActiveTexture(GL_TEXTURE0);
        GlBindTexture(GL_TEXTURE_2D, ViewportTexture);
        GlDisable(GL_DEPTH_TEST);
        GlDisable(GL_CULL_FACE);
        GlDisable(GL_BLEND);
        GlDisable(GL_SCISSOR_TEST);
        GlDrawArrays(GL_TRIANGLES, 0, 6);
    }

    void AbortFrameAccess() noexcept
    {
        if(Display && EglMakeCurrent)EglMakeCurrent(Display,nullptr,nullptr,nullptr);
        if(FrameAccess.owns_lock())FrameAccess.unlock();
    }
    bool MakeCurrent()
    {
        if (Display == nullptr || Surface == nullptr || Context == nullptr)
            return Fail("The OpenGL ES editor presenter is not initialized.");
        if (EglMakeCurrent(Display, Surface, Surface, Context) != EGL_FALSE_VALUE)
            return true;
        return Fail(HexError(
            "eglMakeCurrent", static_cast<unsigned long>(EglGetError())));
    }

    bool Initialize(HWND window,int width,int height)
    {
        Shutdown();Window=window;Width=width;Height=height;Initialized=true;return true;
    }
    bool InitializeNow(HWND window, int width, int height)
    {
        const auto shared=SharedGroup;const auto display=SharedDisplay,context=SharedContext;
        Shutdown();SharedGroup=shared;SharedDisplay=display;SharedContext=context;
        if (window == nullptr || IsWindow(window) == FALSE || width <= 0 || height <= 0)
            return Fail(
                "The OpenGL ES editor presenter requires a valid Win32 window and dimensions.");
        if (!LoadLibraries() || !LoadEGLFunctions() || !CreateWindowContext(window) ||
            !LoadGLFunctions())
        {
            return false;
        }
        const unsigned char* version = GlGetString(GL_VERSION);
        if (version == nullptr || std::strstr(
                reinterpret_cast<const char*>(version), "OpenGL ES 3") == nullptr)
        {
            return Fail(
                "ANGLE created a context that is not OpenGL ES 3; Evans requires ES 3.0 or newer.");
        }
        Window = window;
        Width = width;
        Height = height;
        if (!CreateViewportPipeline() ||
            !ImguiRenderer.Initialize(this, &Impl::ResolveForImgui))
        {
            if (LastError.empty())
                LastError = "The OpenGL ES Riemann-style ImGui renderer could not initialize.";
            return false;
        }
        Initialized = true;
        LastError.clear();
        return true;
    }

    void Shutdown() noexcept
    {
        std::unique_lock<std::mutex> cleanup;
        if(SharedAccess && !FrameAccess.owns_lock())cleanup=std::unique_lock<std::mutex>(*SharedAccess);
        if (Display != nullptr && Surface != nullptr && Context != nullptr &&
            EglMakeCurrent != nullptr)
        {
            EglMakeCurrent(Display, Surface, Surface, Context);
            ImguiRenderer.Shutdown();
            if (!GpuViewport && GlDeleteTextures != nullptr && ViewportTexture != 0)
                GlDeleteTextures(1, &ViewportTexture);
            if (GlDeleteBuffers != nullptr && ViewportVertexBuffer != 0)
                GlDeleteBuffers(1, &ViewportVertexBuffer);
            if (GlDeleteVertexArrays != nullptr && ViewportVertexArray != 0)
                GlDeleteVertexArrays(1, &ViewportVertexArray);
            if (GlDeleteProgram != nullptr && ViewportProgram != 0)
                GlDeleteProgram(ViewportProgram);
            EglMakeCurrent(Display, nullptr, nullptr, nullptr);
        }
        if (!SharedGroup && Display != nullptr && Context != nullptr && EglDestroyContext != nullptr)
            EglDestroyContext(Display, Context);
        if (Display != nullptr && Surface != nullptr && EglDestroySurface != nullptr)
            EglDestroySurface(Display, Surface);
        if (SurfaceWindow) DestroyWindow(SurfaceWindow);
        SurfaceWindow = nullptr;
        if (!SharedGroup && !SkipTerminate && Display != nullptr && EglTerminate != nullptr) EglTerminate(Display);
        GpuFrameLease.reset();SharedGroup.reset();SharedDisplay=nullptr;SharedContext=nullptr;GpuViewport=false;SkipTerminate=false;
        if (EGLModule != nullptr) FreeLibrary(EGLModule);
        if (GLESModule != nullptr) FreeLibrary(GLESModule);

        Window = nullptr;
        Width = 0;
        Height = 0;
        ImageWidth = 0;
        ImageHeight = 0;
        UploadedRevision = std::numeric_limits<std::uint64_t>::max();
        ViewportTexture = 0;
        ViewportVertexBuffer = 0;
        ViewportVertexArray = 0;
        ViewportProgram = 0;
        Context = nullptr;
        Surface = nullptr;
        Config = nullptr;
        Display = nullptr;
        EGLModule = nullptr;
        GLESModule = nullptr;
        Initialized = false;
    }

    bool Resize(int width, int height)
    {
        if (!Initialized || width <= 0 || height <= 0)
            return Fail("OpenGL ES presenter resize requires positive dimensions.");
        if (SurfaceWindow && !SetWindowPos(SurfaceWindow, nullptr, 0, 0, width, height,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE)) return Fail("Cannot resize the OpenGL ES presentation window");
        Width = width;
        Height = height;
        return true;
    }

    bool BeginFrame(
        const Image& image,
        std::uint64_t revision,
        int viewportX,
        int viewportY,
        int viewportWidth,
        int viewportHeight)
    {
        const auto context=image.GpuContext;
        if(context && context->Owner)SharedAccess=std::static_pointer_cast<std::mutex>(context->Owner);
        if(SharedAccess && !FrameAccess.owns_lock())FrameAccess=std::unique_lock<std::mutex>(*SharedAccess);
        if (!AdoptGpuGroup(image) || !Initialized || !MakeCurrent()) return false;
        const auto bindFramebuffer=reinterpret_cast<void(EVANS_GL_CALL*)(GLenum,GLuint)>(ResolveGL("glBindFramebuffer"));
        const auto bindSampler=reinterpret_cast<void(EVANS_GL_CALL*)(GLuint,GLuint)>(ResolveGL("glBindSampler"));
        if(bindFramebuffer)bindFramebuffer(0x8D40,0);
        if(bindSampler)bindSampler(0,0);
        GlViewport(0, 0, Width, Height);
        GlDisable(GL_SCISSOR_TEST);
        GlClearColor(0.035f, 0.043f, 0.059f, 1.0f);
        GlClear(GL_COLOR_BUFFER_BIT);
        if (image.IsValid())
        {
            if (!EnsureViewportTexture(image, revision)) return false;
            DrawViewport(
                viewportX, viewportY, viewportWidth, viewportHeight);
        }
        return CheckGL("Begin OpenGL ES editor frame");
    }

    void DrawImgui()
    {
        if (!Initialized || !MakeCurrent()) return;
        ImguiRenderer.Draw(Width, Height);
    }

    bool EndFrame()
    {
        if (!Initialized || !MakeCurrent()) return false;
        if (EglSwapBuffers(Display, Surface) != EGL_FALSE_VALUE)
        {
            EglMakeCurrent(Display,nullptr,nullptr,nullptr);
            if(FrameAccess.owns_lock())FrameAccess.unlock();
            return true;
        }
        return Fail(HexError(
            "eglSwapBuffers", static_cast<unsigned long>(EglGetError())));
    }

    HWND Window = nullptr;
    HWND SurfaceWindow = nullptr;
    int Width = 0;
    int Height = 0;
    bool Initialized = false;
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
    PFNEGLCREATEWINDOWSURFACE EglCreateWindowSurface = nullptr;
    PFNEGLCREATECONTEXT EglCreateContext = nullptr;
    PFNEGLMAKECURRENT EglMakeCurrent = nullptr;
    PFNEGLSWAPBUFFERS EglSwapBuffers = nullptr;
    PFNEGLSWAPINTERVAL EglSwapInterval = nullptr;
    PFNEGLDESTROYCONTEXT EglDestroyContext = nullptr;
    PFNEGLDESTROYSURFACE EglDestroySurface = nullptr;
    PFNEGLTERMINATE EglTerminate = nullptr;
    PFNEGLGETERROR EglGetError = nullptr;
    PFNEGLGETPROCADDRESS EglGetProcAddress = nullptr;
    PFNGLGETERROR GlGetError = nullptr;
    PFNGLGETSTRING GlGetString = nullptr;
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
    PFNGLUNIFORM1I GlUniform1i = nullptr;
    PFNGLGENBUFFERS GlGenBuffers = nullptr;
    PFNGLBINDBUFFER GlBindBuffer = nullptr;
    PFNGLBUFFERDATA GlBufferData = nullptr;
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
    PFNGLVIEWPORT GlViewport = nullptr;
    PFNGLCLEARCOLOR GlClearColor = nullptr;
    PFNGLCLEAR GlClear = nullptr;
    PFNGLDISABLE GlDisable = nullptr;
    PFNGLDRAWARRAYS GlDrawArrays = nullptr;
    PFNGLPIXELSTOREI GlPixelStorei = nullptr;
    GLuint ViewportProgram = 0;
    GLuint ViewportVertexArray = 0;
    GLuint ViewportVertexBuffer = 0;
    GLuint ViewportTexture = 0;
    std::uint32_t ImageWidth = 0;
    std::uint32_t ImageHeight = 0;
    std::uint64_t UploadedRevision = std::numeric_limits<std::uint64_t>::max();
    ImguiRenderOpenGLES ImguiRenderer;
    std::string LastError;
};

EditorPresenterOpenGLES::EditorPresenterOpenGLES()
    : Impl_(std::make_unique<Impl>())
{
}

EditorPresenterOpenGLES::~EditorPresenterOpenGLES() = default;

bool EditorPresenterOpenGLES::Initialize(void* nativeWindow, int width, int height)
{
    return Impl_->Initialize(static_cast<HWND>(nativeWindow), width, height);
}

void EditorPresenterOpenGLES::Shutdown() noexcept
{
    Impl_->Shutdown();Impl_->AbortFrameAccess();
}

bool EditorPresenterOpenGLES::Resize(int width, int height)
{
    return Impl_->Resize(width, height);
}

bool EditorPresenterOpenGLES::BeginFrame(
    const Image& viewportImage,
    std::uint64_t viewportRevision,
    int viewportX,
    int viewportY,
    int viewportWidth,
    int viewportHeight)
{
    const bool success=Impl_->BeginFrame(
        viewportImage, viewportRevision, viewportX, viewportY,
        viewportWidth, viewportHeight);
    if(!success)Impl_->AbortFrameAccess();
    return success;
}

void EditorPresenterOpenGLES::DrawImgui()
{
    Impl_->DrawImgui();
}

bool EditorPresenterOpenGLES::EndFrame()
{
    const bool success=Impl_->EndFrame();if(!success)Impl_->AbortFrameAccess();return success;
}

bool EditorPresenterOpenGLES::IsInitialized() const noexcept
{
    return Impl_->Initialized;
}

const std::string& EditorPresenterOpenGLES::LastError() const noexcept
{
    return Impl_->LastError;
}

} // namespace Evans::Editor

#endif
