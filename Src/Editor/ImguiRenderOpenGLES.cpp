#include "Editor/ImguiRenderOpenGLES.h"

#if defined(_WIN32)

#include "Editor/Imgui.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace Evans::Editor
{
namespace
{

#define EVANS_GL_CALL __stdcall

using GLenum = unsigned int;
using GLboolean = unsigned char;
using GLsizei = int;
using GLint = int;
using GLuint = unsigned int;
using GLfloat = float;
using GLchar = char;
using GLsizeiptr = std::ptrdiff_t;

constexpr GLboolean GL_FALSE_VALUE = 0;
constexpr GLboolean GL_TRUE_VALUE = 1;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
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
constexpr GLenum GL_BLEND = 0x0BE2;
constexpr GLenum GL_CULL_FACE = 0x0B44;
constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_SCISSOR_TEST = 0x0C11;
constexpr GLenum GL_SRC_ALPHA = 0x0302;
constexpr GLenum GL_ONE_MINUS_SRC_ALPHA = 0x0303;
constexpr GLenum GL_UNPACK_ALIGNMENT = 0x0CF5;

using PFNGLCREATESHADER = GLuint(EVANS_GL_CALL*)(GLenum);
using PFNGLSHADERSOURCE = void(EVANS_GL_CALL*)(GLuint, GLsizei, const GLchar* const*, const GLint*);
using PFNGLCOMPILESHADER = void(EVANS_GL_CALL*)(GLuint);
using PFNGLGETSHADERIV = void(EVANS_GL_CALL*)(GLuint, GLenum, GLint*);
using PFNGLDELETESHADER = void(EVANS_GL_CALL*)(GLuint);
using PFNGLCREATEPROGRAM = GLuint(EVANS_GL_CALL*)();
using PFNGLATTACHSHADER = void(EVANS_GL_CALL*)(GLuint, GLuint);
using PFNGLLINKPROGRAM = void(EVANS_GL_CALL*)(GLuint);
using PFNGLGETPROGRAMIV = void(EVANS_GL_CALL*)(GLuint, GLenum, GLint*);
using PFNGLDELETEPROGRAM = void(EVANS_GL_CALL*)(GLuint);
using PFNGLUSEPROGRAM = void(EVANS_GL_CALL*)(GLuint);
using PFNGLGETUNIFORMLOCATION = GLint(EVANS_GL_CALL*)(GLuint, const GLchar*);
using PFNGLUNIFORM2F = void(EVANS_GL_CALL*)(GLint, GLfloat, GLfloat);
using PFNGLUNIFORM1I = void(EVANS_GL_CALL*)(GLint, GLint);
using PFNGLGENBUFFERS = void(EVANS_GL_CALL*)(GLsizei, GLuint*);
using PFNGLBINDBUFFER = void(EVANS_GL_CALL*)(GLenum, GLuint);
using PFNGLBUFFERDATA = void(EVANS_GL_CALL*)(GLenum, GLsizeiptr, const void*, GLenum);
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
using PFNGLVIEWPORT = void(EVANS_GL_CALL*)(GLint, GLint, GLsizei, GLsizei);
using PFNGLENABLE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLDISABLE = void(EVANS_GL_CALL*)(GLenum);
using PFNGLBLENDFUNC = void(EVANS_GL_CALL*)(GLenum, GLenum);
using PFNGLSCISSOR = void(EVANS_GL_CALL*)(GLint, GLint, GLsizei, GLsizei);
using PFNGLDRAWARRAYS = void(EVANS_GL_CALL*)(GLenum, GLint, GLsizei);
using PFNGLPIXELSTOREI = void(EVANS_GL_CALL*)(GLenum, GLint);

constexpr float Pi = 3.14159265358979323846f;
constexpr float PullQueueScale = 1.0f / 8.0f;
constexpr float TabStops[] = {150.0f, 210.0f, 270.0f, 330.0f};

template <typename Function>
bool Resolve(Function& function, void* userData, OpenGLESProcResolver resolver, const char* name)
{
    function = reinterpret_cast<Function>(resolver(userData, name));
    return function != nullptr;
}

} // namespace

struct ImguiRenderOpenGLES::Impl
{
    struct Glyph
    {
        float U0 = 0.0f;
        float V0 = 0.0f;
        float U1 = 0.0f;
        float V1 = 0.0f;
        float Width = 0.0f;
        float Height = 0.0f;
        float Advance = 0.0f;
    };

    struct Vertex
    {
        float Position[2];
        float TexCoord[2];
        std::uint32_t Color = 0;
    };

    bool LoadFunctions(void* userData, OpenGLESProcResolver resolver)
    {
#define EVANS_RESOLVE_GL(Member, Name) Resolve(Member, userData, resolver, Name)
        return EVANS_RESOLVE_GL(GlCreateShader, "glCreateShader") &&
               EVANS_RESOLVE_GL(GlShaderSource, "glShaderSource") &&
               EVANS_RESOLVE_GL(GlCompileShader, "glCompileShader") &&
               EVANS_RESOLVE_GL(GlGetShaderiv, "glGetShaderiv") &&
               EVANS_RESOLVE_GL(GlDeleteShader, "glDeleteShader") &&
               EVANS_RESOLVE_GL(GlCreateProgram, "glCreateProgram") &&
               EVANS_RESOLVE_GL(GlAttachShader, "glAttachShader") &&
               EVANS_RESOLVE_GL(GlLinkProgram, "glLinkProgram") &&
               EVANS_RESOLVE_GL(GlGetProgramiv, "glGetProgramiv") &&
               EVANS_RESOLVE_GL(GlDeleteProgram, "glDeleteProgram") &&
               EVANS_RESOLVE_GL(GlUseProgram, "glUseProgram") &&
               EVANS_RESOLVE_GL(GlGetUniformLocation, "glGetUniformLocation") &&
               EVANS_RESOLVE_GL(GlUniform2f, "glUniform2f") &&
               EVANS_RESOLVE_GL(GlUniform1i, "glUniform1i") &&
               EVANS_RESOLVE_GL(GlGenBuffers, "glGenBuffers") &&
               EVANS_RESOLVE_GL(GlBindBuffer, "glBindBuffer") &&
               EVANS_RESOLVE_GL(GlBufferData, "glBufferData") &&
               EVANS_RESOLVE_GL(GlDeleteBuffers, "glDeleteBuffers") &&
               EVANS_RESOLVE_GL(GlGenVertexArrays, "glGenVertexArrays") &&
               EVANS_RESOLVE_GL(GlBindVertexArray, "glBindVertexArray") &&
               EVANS_RESOLVE_GL(GlDeleteVertexArrays, "glDeleteVertexArrays") &&
               EVANS_RESOLVE_GL(GlEnableVertexAttribArray, "glEnableVertexAttribArray") &&
               EVANS_RESOLVE_GL(GlVertexAttribPointer, "glVertexAttribPointer") &&
               EVANS_RESOLVE_GL(GlGenTextures, "glGenTextures") &&
               EVANS_RESOLVE_GL(GlBindTexture, "glBindTexture") &&
               EVANS_RESOLVE_GL(GlTexImage2D, "glTexImage2D") &&
               EVANS_RESOLVE_GL(GlTexParameteri, "glTexParameteri") &&
               EVANS_RESOLVE_GL(GlDeleteTextures, "glDeleteTextures") &&
               EVANS_RESOLVE_GL(GlActiveTexture, "glActiveTexture") &&
               EVANS_RESOLVE_GL(GlViewport, "glViewport") &&
               EVANS_RESOLVE_GL(GlEnable, "glEnable") &&
               EVANS_RESOLVE_GL(GlDisable, "glDisable") &&
               EVANS_RESOLVE_GL(GlBlendFunc, "glBlendFunc") &&
               EVANS_RESOLVE_GL(GlScissor, "glScissor") &&
               EVANS_RESOLVE_GL(GlDrawArrays, "glDrawArrays") &&
               EVANS_RESOLVE_GL(GlPixelStorei, "glPixelStorei");
#undef EVANS_RESOLVE_GL
    }

    GLuint Compile(GLenum type, const char* source)
    {
        const GLuint shader = GlCreateShader(type);
        if (shader == 0) return 0;
        GlShaderSource(shader, 1, &source, nullptr);
        GlCompileShader(shader);
        GLint compiled = 0;
        GlGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (compiled != 0) return shader;
        GlDeleteShader(shader);
        return 0;
    }

    bool CreateProgram()
    {
        static constexpr const char* vertexSource = R"GLSL(#version 300 es
precision highp float;
layout(location=0) in vec2 InPosition;
layout(location=1) in vec2 InTexCoord;
layout(location=2) in vec4 InColor;
uniform vec2 ScreenSize;
out vec2 TexCoord;
out vec4 Color;
void main()
{
    vec2 p = InPosition / ScreenSize;
    gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    TexCoord = InTexCoord;
    Color = InColor;
}
)GLSL";
        static constexpr const char* fragmentSource = R"GLSL(#version 300 es
precision highp float;
in vec2 TexCoord;
in vec4 Color;
uniform sampler2D SourceTexture;
out vec4 OutColor;
void main()
{
    OutColor = Color * texture(SourceTexture, TexCoord);
}
)GLSL";
        const GLuint vertex = Compile(GL_VERTEX_SHADER, vertexSource);
        if (vertex == 0) return false;
        const GLuint fragment = Compile(GL_FRAGMENT_SHADER, fragmentSource);
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
        if (linked == 0) return false;
        ScreenSizeLocation = GlGetUniformLocation(Program, "ScreenSize");
        const GLint textureLocation = GlGetUniformLocation(Program, "SourceTexture");
        if (ScreenSizeLocation < 0 || textureLocation < 0) return false;
        GlUseProgram(Program);
        GlUniform1i(textureLocation, 0);
        return true;
    }

    void ConfigureTexture()
    {
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        GlTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    bool CreateWhiteTexture()
    {
        constexpr std::uint32_t white = 0xffffffffu;
        GlGenTextures(1, &WhiteTexture);
        GlBindTexture(GL_TEXTURE_2D, WhiteTexture);
        ConfigureTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, &white);
        return WhiteTexture != 0;
    }

    bool CreateFontTexture()
    {
        constexpr int atlasWidth = 512;
        constexpr int atlasHeight = 512;
        BITMAPINFO bitmapInfo{};
        bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmapInfo.bmiHeader.biWidth = atlasWidth;
        bitmapInfo.bmiHeader.biHeight = -atlasHeight;
        bitmapInfo.bmiHeader.biPlanes = 1;
        bitmapInfo.bmiHeader.biBitCount = 32;
        bitmapInfo.bmiHeader.biCompression = BI_RGB;

        void* dibPixels = nullptr;
        HDC dc = CreateCompatibleDC(nullptr);
        if (dc == nullptr) return false;
        HBITMAP bitmap = CreateDIBSection(
            dc, &bitmapInfo, DIB_RGB_COLORS, &dibPixels, nullptr, 0);
        if (bitmap == nullptr || dibPixels == nullptr)
        {
            if (bitmap != nullptr) DeleteObject(bitmap);
            DeleteDC(dc);
            return false;
        }
        const HGDIOBJ oldBitmap = SelectObject(dc, bitmap);
        std::memset(dibPixels, 0, atlasWidth * atlasHeight * 4);
        HFONT font = CreateFontA(
            -15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, ANSI_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
            DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
        HGDIOBJ oldFont = nullptr;
        if (font != nullptr) oldFont = SelectObject(dc, font);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(255, 255, 255));
        TEXTMETRICA metrics{};
        GetTextMetricsA(dc, &metrics);
        FontHeight = std::max(8L, metrics.tmHeight);
        int penX = 1;
        int penY = 1;
        for (int character = 32; character < 128; ++character)
        {
            const char value = static_cast<char>(character);
            SIZE size{};
            GetTextExtentPoint32A(dc, &value, 1, &size);
            const int glyphWidth = std::max(1L, size.cx);
            if (penX + glyphWidth + 2 >= atlasWidth)
            {
                penX = 1;
                penY += FontHeight + 2;
            }
            if (penY + FontHeight + 2 >= atlasHeight) break;
            Glyph& glyph = Glyphs[static_cast<std::size_t>(character - 32)];
            glyph.U0 = static_cast<float>(penX) / atlasWidth;
            glyph.V0 = static_cast<float>(penY) / atlasHeight;
            glyph.U1 = static_cast<float>(penX + glyphWidth) / atlasWidth;
            glyph.V1 = static_cast<float>(penY + FontHeight) / atlasHeight;
            glyph.Width = static_cast<float>(glyphWidth);
            glyph.Height = static_cast<float>(FontHeight);
            glyph.Advance = static_cast<float>(glyphWidth);
            TextOutA(dc, penX, penY, &value, 1);
            penX += glyphWidth + 1;
        }

        std::vector<std::uint8_t> rgba(
            static_cast<std::size_t>(atlasWidth) * atlasHeight * 4U);
        const auto* source = static_cast<const std::uint8_t*>(dibPixels);
        for (std::size_t pixel = 0; pixel < rgba.size() / 4U; ++pixel)
        {
            const std::uint8_t alpha = std::max(
                source[pixel * 4U],
                std::max(source[pixel * 4U + 1U], source[pixel * 4U + 2U]));
            rgba[pixel * 4U] = 255;
            rgba[pixel * 4U + 1U] = 255;
            rgba[pixel * 4U + 2U] = 255;
            rgba[pixel * 4U + 3U] = alpha;
        }
        if (oldFont != nullptr) SelectObject(dc, oldFont);
        if (font != nullptr) DeleteObject(font);
        SelectObject(dc, oldBitmap);
        DeleteObject(bitmap);
        DeleteDC(dc);

        GlGenTextures(1, &FontTexture);
        GlBindTexture(GL_TEXTURE_2D, FontTexture);
        ConfigureTexture();
        GlPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        GlTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8, atlasWidth, atlasHeight, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        return FontTexture != 0;
    }

    bool Initialize(void* userData, OpenGLESProcResolver resolver)
    {
        Shutdown();
        if (resolver == nullptr || !LoadFunctions(userData, resolver) || !CreateProgram())
            return false;
        GlGenVertexArrays(1, &VertexArray);
        GlGenBuffers(1, &VertexBuffer);
        GlBindVertexArray(VertexArray);
        GlBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
        GlEnableVertexAttribArray(0);
        GlVertexAttribPointer(
            0, 2, GL_FLOAT_VALUE, GL_FALSE_VALUE, sizeof(Vertex),
            reinterpret_cast<const void*>(offsetof(Vertex, Position)));
        GlEnableVertexAttribArray(1);
        GlVertexAttribPointer(
            1, 2, GL_FLOAT_VALUE, GL_FALSE_VALUE, sizeof(Vertex),
            reinterpret_cast<const void*>(offsetof(Vertex, TexCoord)));
        GlEnableVertexAttribArray(2);
        GlVertexAttribPointer(
            2, 4, GL_UNSIGNED_BYTE, GL_TRUE_VALUE, sizeof(Vertex),
            reinterpret_cast<const void*>(offsetof(Vertex, Color)));
        return VertexArray != 0 && VertexBuffer != 0 &&
               CreateWhiteTexture() && CreateFontTexture();
    }

    void Shutdown() noexcept
    {
        if (GlDeleteTextures != nullptr)
        {
            if (FontTexture != 0) GlDeleteTextures(1, &FontTexture);
            if (WhiteTexture != 0) GlDeleteTextures(1, &WhiteTexture);
        }
        if (GlDeleteBuffers != nullptr && VertexBuffer != 0)
            GlDeleteBuffers(1, &VertexBuffer);
        if (GlDeleteVertexArrays != nullptr && VertexArray != 0)
            GlDeleteVertexArrays(1, &VertexArray);
        if (GlDeleteProgram != nullptr && Program != 0) GlDeleteProgram(Program);
        FontTexture = 0;
        WhiteTexture = 0;
        VertexBuffer = 0;
        VertexArray = 0;
        Program = 0;
        ScreenSizeLocation = -1;
    }

    float ToScreenY(float y) const
    {
        return static_cast<float>(Height) - y;
    }

    void DrawGeometry(const Vertex* vertices, std::size_t vertexCount, GLuint texture)
    {
        if (vertices == nullptr || vertexCount == 0 || texture == 0) return;
        GlBindTexture(GL_TEXTURE_2D, texture);
        GlBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(vertexCount * sizeof(Vertex)),
            vertices,
            GL_DYNAMIC_DRAW);
        GlDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertexCount));
    }

    void SetScissor(int x, int y, int width, int height)
    {
        const int left = std::clamp(x, 0, Width);
        const int bottom = std::clamp(y, 0, Height);
        const int right = std::clamp(x + width, left, Width);
        const int top = std::clamp(y + height, bottom, Height);
        GlScissor(left, bottom, right - left, top - bottom);
    }

    void AddRect(float x, float y, float width, float height, std::uint32_t color)
    {
        const float left = x;
        const float right = x + width;
        const float top = ToScreenY(y + height);
        const float bottom = ToScreenY(y);
        const Vertex vertices[] = {
            {{left, top}, {0.0f, 0.0f}, color},
            {{right, top}, {0.0f, 0.0f}, color},
            {{right, bottom}, {0.0f, 0.0f}, color},
            {{left, top}, {0.0f, 0.0f}, color},
            {{right, bottom}, {0.0f, 0.0f}, color},
            {{left, bottom}, {0.0f, 0.0f}, color}};
        DrawGeometry(vertices, std::size(vertices), WhiteTexture);
    }

    void AddRoundedRect(
        float x, float y, float width, float height, float radius,
        std::uint32_t color)
    {
        radius = std::max(0.0f, std::min(radius, std::min(width, height) * 0.5f));
        if (radius <= 0.5f)
        {
            AddRect(x, y, width, height, color);
            return;
        }
        const float left = x;
        const float right = x + width;
        const float top = ToScreenY(y + height);
        const float bottom = ToScreenY(y);
        const float centers[4][2] = {
            {right - radius, top + radius},
            {right - radius, bottom - radius},
            {left + radius, bottom - radius},
            {left + radius, top + radius}};
        const float starts[] = {-Pi * 0.5f, 0.0f, Pi * 0.5f, Pi};
        std::vector<Vertex> points;
        constexpr int segments = 6;
        for (int corner = 0; corner < 4; ++corner)
        {
            for (int segment = 0; segment <= segments; ++segment)
            {
                const float angle = starts[corner] + Pi * 0.5f *
                    static_cast<float>(segment) / segments;
                points.push_back({{
                    centers[corner][0] + std::cos(angle) * radius,
                    centers[corner][1] + std::sin(angle) * radius}, {}, color});
            }
        }
        const Vertex center = {
            {(left + right) * 0.5f, (top + bottom) * 0.5f}, {}, color};
        std::vector<Vertex> fan;
        fan.reserve(points.size() * 3U);
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            fan.push_back(center);
            fan.push_back(points[index]);
            fan.push_back(points[(index + 1U) % points.size()]);
        }
        DrawGeometry(fan.data(), fan.size(), WhiteTexture);
    }

    void AddLine(
        float x0, float y0, float x1, float y1, float radius,
        std::uint32_t color)
    {
        y0 = ToScreenY(y0);
        y1 = ToScreenY(y1);
        float dx = x1 - x0;
        float dy = y1 - y0;
        const float length = std::sqrt(dx * dx + dy * dy);
        if (length <= 0.0001f) return;
        dx /= length;
        dy /= length;
        const float halfWidth = std::max(1.0f, radius) * 0.5f;
        const float nx = -dy * halfWidth;
        const float ny = dx * halfWidth;
        const Vertex vertices[] = {
            {{x0 - nx, y0 - ny}, {}, color}, {{x0 + nx, y0 + ny}, {}, color},
            {{x1 + nx, y1 + ny}, {}, color}, {{x0 - nx, y0 - ny}, {}, color},
            {{x1 + nx, y1 + ny}, {}, color}, {{x1 - nx, y1 - ny}, {}, color}};
        DrawGeometry(vertices, std::size(vertices), WhiteTexture);
    }

    void AddTriangle(
        float x, float y, float width, float height, int flags,
        std::uint32_t color)
    {
        Vertex vertices[3]{};
        if (flags == 1)
        {
            vertices[0].Position[0] = x;
            vertices[0].Position[1] = ToScreenY(y);
            vertices[1].Position[0] = x + width;
            vertices[1].Position[1] = ToScreenY(y + height * 0.5f);
            vertices[2].Position[0] = x;
            vertices[2].Position[1] = ToScreenY(y + height);
        }
        else
        {
            vertices[0].Position[0] = x;
            vertices[0].Position[1] = ToScreenY(y + height);
            vertices[1].Position[0] = x + width * 0.5f;
            vertices[1].Position[1] = ToScreenY(y);
            vertices[2].Position[0] = x + width;
            vertices[2].Position[1] = ToScreenY(y + height);
        }
        for (Vertex& vertex : vertices) vertex.Color = color;
        DrawGeometry(vertices, std::size(vertices), WhiteTexture);
    }

    float MeasureText(const char* text) const
    {
        float x = 0.0f;
        float length = 0.0f;
        if (text == nullptr) return length;
        while (*text != '\0')
        {
            const unsigned char character = static_cast<unsigned char>(*text++);
            if (character == '\t')
            {
                for (float tabStop : TabStops)
                {
                    if (x < tabStop)
                    {
                        x = tabStop;
                        break;
                    }
                }
                length = std::max(length, x);
            }
            else if (character >= 32 && character < 128)
            {
                const Glyph& glyph = Glyphs[character - 32U];
                length = std::max(length, x + glyph.Width);
                x += glyph.Advance;
            }
        }
        return length;
    }

    void AddText(
        float x, float y, int align, const char* text, std::uint32_t color)
    {
        if (text == nullptr || FontTexture == 0) return;
        if (align == IMGUI_ALIGN_CENTER) x -= MeasureText(text) * 0.5f;
        else if (align == IMGUI_ALIGN_RIGHT) x -= MeasureText(text);
        const float originX = x;
        const float top = ToScreenY(y + static_cast<float>(FontHeight));
        std::vector<Vertex> vertices;
        vertices.reserve(std::strlen(text) * 6U);
        while (*text != '\0')
        {
            const unsigned char character = static_cast<unsigned char>(*text++);
            if (character == '\t')
            {
                for (float tabStop : TabStops)
                {
                    if (x < originX + tabStop)
                    {
                        x = originX + tabStop;
                        break;
                    }
                }
            }
            else if (character >= 32 && character < 128)
            {
                const Glyph& glyph = Glyphs[character - 32U];
                const float right = x + glyph.Width;
                const float bottom = top + glyph.Height;
                vertices.push_back({{x, top}, {glyph.U0, glyph.V0}, color});
                vertices.push_back({{right, top}, {glyph.U1, glyph.V0}, color});
                vertices.push_back({{right, bottom}, {glyph.U1, glyph.V1}, color});
                vertices.push_back({{x, top}, {glyph.U0, glyph.V0}, color});
                vertices.push_back({{right, bottom}, {glyph.U1, glyph.V1}, color});
                vertices.push_back({{x, bottom}, {glyph.U0, glyph.V1}, color});
                x += glyph.Advance;
            }
        }
        DrawGeometry(vertices.data(), vertices.size(), FontTexture);
    }

    void DrawQueue()
    {
        const imguiGfxCmd* commands = imguiGetRenderQueue();
        const int commandCount = imguiGetRenderQueueSize();
        for (int index = 0; index < commandCount; ++index)
        {
            const imguiGfxCmd& command = commands[index];
            switch (command.type)
            {
            case IMGUI_GFXCMD_RECT:
            {
                const float x = command.rect.x * PullQueueScale + 0.5f;
                const float y = command.rect.y * PullQueueScale + 0.5f;
                const float width = command.rect.w * PullQueueScale - 1.0f;
                const float height = command.rect.h * PullQueueScale - 1.0f;
                const float radius = command.rect.r * PullQueueScale;
                if (command.rect.r == 0) AddRect(x, y, width, height, command.col);
                else AddRoundedRect(x, y, width, height, radius, command.col);
                break;
            }
            case IMGUI_GFXCMD_LINE:
                AddLine(
                    command.line.x0 * PullQueueScale,
                    command.line.y0 * PullQueueScale,
                    command.line.x1 * PullQueueScale,
                    command.line.y1 * PullQueueScale,
                    command.line.r * PullQueueScale,
                    command.col);
                break;
            case IMGUI_GFXCMD_TRIANGLE:
                AddTriangle(
                    command.rect.x * PullQueueScale + 0.5f,
                    command.rect.y * PullQueueScale + 0.5f,
                    command.rect.w * PullQueueScale - 1.0f,
                    command.rect.h * PullQueueScale - 1.0f,
                    command.flags,
                    command.col);
                break;
            case IMGUI_GFXCMD_TEXT:
                AddText(
                    static_cast<float>(command.text.x),
                    static_cast<float>(command.text.y),
                    command.text.align, command.text.text, command.col);
                break;
            case IMGUI_GFXCMD_SCISSOR:
                if (command.flags != 0)
                    SetScissor(command.rect.x, command.rect.y, command.rect.w, command.rect.h);
                else SetScissor(0, 0, Width, Height);
                break;
            default:
                break;
            }
        }
    }

    void Draw(int width, int height)
    {
        if (Program == 0 || VertexBuffer == 0) return;
        Width = std::max(width, 1);
        Height = std::max(height, 1);
        GlViewport(0, 0, Width, Height);
        GlUseProgram(Program);
        GlUniform2f(ScreenSizeLocation, static_cast<float>(Width), static_cast<float>(Height));
        GlActiveTexture(GL_TEXTURE0);
        GlBindVertexArray(VertexArray);
        GlBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
        GlDisable(GL_DEPTH_TEST);
        GlDisable(GL_CULL_FACE);
        GlEnable(GL_BLEND);
        GlBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        GlEnable(GL_SCISSOR_TEST);
        SetScissor(0, 0, Width, Height);
        DrawQueue();
        SetScissor(0, 0, Width, Height);
    }

    PFNGLCREATESHADER GlCreateShader = nullptr;
    PFNGLSHADERSOURCE GlShaderSource = nullptr;
    PFNGLCOMPILESHADER GlCompileShader = nullptr;
    PFNGLGETSHADERIV GlGetShaderiv = nullptr;
    PFNGLDELETESHADER GlDeleteShader = nullptr;
    PFNGLCREATEPROGRAM GlCreateProgram = nullptr;
    PFNGLATTACHSHADER GlAttachShader = nullptr;
    PFNGLLINKPROGRAM GlLinkProgram = nullptr;
    PFNGLGETPROGRAMIV GlGetProgramiv = nullptr;
    PFNGLDELETEPROGRAM GlDeleteProgram = nullptr;
    PFNGLUSEPROGRAM GlUseProgram = nullptr;
    PFNGLGETUNIFORMLOCATION GlGetUniformLocation = nullptr;
    PFNGLUNIFORM2F GlUniform2f = nullptr;
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
    PFNGLENABLE GlEnable = nullptr;
    PFNGLDISABLE GlDisable = nullptr;
    PFNGLBLENDFUNC GlBlendFunc = nullptr;
    PFNGLSCISSOR GlScissor = nullptr;
    PFNGLDRAWARRAYS GlDrawArrays = nullptr;
    PFNGLPIXELSTOREI GlPixelStorei = nullptr;
    GLuint Program = 0;
    GLuint VertexArray = 0;
    GLuint VertexBuffer = 0;
    GLuint WhiteTexture = 0;
    GLuint FontTexture = 0;
    GLint ScreenSizeLocation = -1;
    int Width = 1;
    int Height = 1;
    int FontHeight = 16;
    Glyph Glyphs[96]{};
};

ImguiRenderOpenGLES::ImguiRenderOpenGLES()
    : Impl_(std::make_unique<Impl>())
{
}

ImguiRenderOpenGLES::~ImguiRenderOpenGLES() = default;

bool ImguiRenderOpenGLES::Initialize(
    void* resolverUserData, OpenGLESProcResolver resolver)
{
    return Impl_->Initialize(resolverUserData, resolver);
}

void ImguiRenderOpenGLES::Shutdown() noexcept
{
    Impl_->Shutdown();
}

void ImguiRenderOpenGLES::Draw(int width, int height)
{
    Impl_->Draw(width, height);
}

} // namespace Evans::Editor

#endif
