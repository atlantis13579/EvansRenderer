#include "Editor/ImguiRenderVulkan.h"

#if defined(_WIN32)

#include "Editor/Imgui.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>

namespace Evans::Editor
{
namespace
{
constexpr float Pi = 3.14159265358979323846f;
constexpr float QueueScale = 1.0f / 8.0f;
constexpr float TabStops[] = {150.0f, 210.0f, 270.0f, 330.0f};
}

bool ImguiRenderVulkan::Initialize()
{
    FontPixels_.clear();
    std::fill(std::begin(Glyphs_), std::end(Glyphs_), Glyph{});
    BITMAPINFO bitmapInfo{};
    bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfo.bmiHeader.biWidth = FontWidth_;
    bitmapInfo.bmiHeader.biHeight = -FontAtlasHeight_;
    bitmapInfo.bmiHeader.biPlanes = 1;
    bitmapInfo.bmiHeader.biBitCount = 32;
    bitmapInfo.bmiHeader.biCompression = BI_RGB;

    void* dibPixels = nullptr;
    HDC dc = CreateCompatibleDC(nullptr);
    if (dc == nullptr)
    {
        return false;
    }
    HBITMAP bitmap = CreateDIBSection(dc, &bitmapInfo, DIB_RGB_COLORS, &dibPixels, nullptr, 0);
    if (bitmap == nullptr || dibPixels == nullptr)
    {
        if (bitmap != nullptr)
        {
            DeleteObject(bitmap);
        }
        DeleteDC(dc);
        return false;
    }

    const HGDIOBJ oldBitmap = SelectObject(dc, bitmap);
    std::memset(dibPixels, 0, static_cast<std::size_t>(FontWidth_) * FontAtlasHeight_ * 4U);
    HFONT font = CreateFontA(
        -15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    HGDIOBJ oldFont = nullptr;
    if (font != nullptr)
    {
        oldFont = SelectObject(dc, font);
    }
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(255, 255, 255));

    TEXTMETRICA metrics{};
    GetTextMetricsA(dc, &metrics);
    FontLineHeight_ = std::max(8L, metrics.tmHeight);
    imguiSetFontHeight(FontLineHeight_);
    int penX = 1;
    int penY = 1;
    const int rowHeight = FontLineHeight_ + 2;
    for (int codepoint = 32; codepoint < 128; ++codepoint)
    {
        const char character = static_cast<char>(codepoint);
        SIZE size{};
        GetTextExtentPoint32A(dc, &character, 1, &size);
        const int glyphWidth = std::max(1L, size.cx);
        if (penX + glyphWidth + 2 >= FontWidth_)
        {
            penX = 1;
            penY += rowHeight;
        }
        if (penY + FontLineHeight_ + 2 >= FontAtlasHeight_)
        {
            break;
        }

        Glyph& glyph = Glyphs_[codepoint - 32];
        glyph.U0 = static_cast<float>(penX) / static_cast<float>(FontWidth_);
        glyph.V0 = static_cast<float>(penY) / static_cast<float>(FontAtlasHeight_);
        glyph.U1 = static_cast<float>(penX + glyphWidth) / static_cast<float>(FontWidth_);
        glyph.V1 = static_cast<float>(penY + FontLineHeight_) / static_cast<float>(FontAtlasHeight_);
        glyph.Width = static_cast<float>(glyphWidth);
        glyph.Height = static_cast<float>(FontLineHeight_);
        glyph.Advance = static_cast<float>(glyphWidth);
        TextOutA(dc, penX, penY, &character, 1);
        penX += glyphWidth + 1;
    }

    FontPixels_.resize(static_cast<std::size_t>(FontWidth_) * FontAtlasHeight_ * 4U);
    const auto* source = static_cast<const std::uint8_t*>(dibPixels);
    for (std::size_t pixel = 0; pixel < FontPixels_.size() / 4U; ++pixel)
    {
        const std::uint8_t alpha = std::max(source[pixel * 4U],
            std::max(source[pixel * 4U + 1U], source[pixel * 4U + 2U]));
        FontPixels_[pixel * 4U] = 255U;
        FontPixels_[pixel * 4U + 1U] = 255U;
        FontPixels_[pixel * 4U + 2U] = 255U;
        FontPixels_[pixel * 4U + 3U] = alpha;
    }

    if (oldFont != nullptr)
    {
        SelectObject(dc, oldFont);
    }
    if (font != nullptr)
    {
        DeleteObject(font);
    }
    SelectObject(dc, oldBitmap);
    DeleteObject(bitmap);
    DeleteDC(dc);
    return true;
}

void ImguiRenderVulkan::Build(int width, int height)
{
    Width_ = std::max(width, 1);
    Height_ = std::max(height, 1);
    Vertices_.clear();
    Batches_.clear();
    SetScissor(0, 0, Width_, Height_);

    const imguiGfxCmd* commands = imguiGetRenderQueue();
    const int commandCount = imguiGetRenderQueueSize();
    if (commands == nullptr || commandCount <= 0)
    {
        return;
    }
    for (int commandIndex = 0; commandIndex < commandCount; ++commandIndex)
    {
        const imguiGfxCmd& command = commands[commandIndex];
        switch (command.type)
        {
        case IMGUI_GFXCMD_RECT:
        {
            const float x = static_cast<float>(command.rect.x) * QueueScale + 0.5f;
            const float y = static_cast<float>(command.rect.y) * QueueScale + 0.5f;
            const float w = static_cast<float>(command.rect.w) * QueueScale - 1.0f;
            const float h = static_cast<float>(command.rect.h) * QueueScale - 1.0f;
            const float radius = static_cast<float>(command.rect.r) * QueueScale;
            if (command.rect.r == 0)
            {
                AddRect(x, y, w, h, command.col);
            }
            else
            {
                AddRoundedRect(x, y, w, h, radius, command.col);
            }
            break;
        }
        case IMGUI_GFXCMD_LINE:
            AddLine(
                static_cast<float>(command.line.x0) * QueueScale,
                static_cast<float>(command.line.y0) * QueueScale,
                static_cast<float>(command.line.x1) * QueueScale,
                static_cast<float>(command.line.y1) * QueueScale,
                static_cast<float>(command.line.r) * QueueScale,
                command.col);
            break;
        case IMGUI_GFXCMD_TRIANGLE:
            AddTriangle(
                static_cast<float>(command.rect.x) * QueueScale + 0.5f,
                static_cast<float>(command.rect.y) * QueueScale + 0.5f,
                static_cast<float>(command.rect.w) * QueueScale - 1.0f,
                static_cast<float>(command.rect.h) * QueueScale - 1.0f,
                command.flags,
                command.col);
            break;
        case IMGUI_GFXCMD_TEXT:
            AddText(static_cast<float>(command.text.x), static_cast<float>(command.text.y),
                command.text.align, command.text.text, command.col);
            break;
        case IMGUI_GFXCMD_SCISSOR:
            if (command.flags != 0)
            {
                SetScissor(command.rect.x, command.rect.y, command.rect.w, command.rect.h);
            }
            else
            {
                SetScissor(0, 0, Width_, Height_);
            }
            break;
        default:
            break;
        }
    }
}

void ImguiRenderVulkan::AddGeometry(const Vertex* vertices, std::uint32_t count, Texture texture)
{
    if (vertices == nullptr || count == 0 || ScissorWidth_ <= 0 || ScissorHeight_ <= 0)
    {
        return;
    }
    if (Vertices_.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() - count))
    {
        return;
    }
    const std::uint32_t firstVertex = static_cast<std::uint32_t>(Vertices_.size());
    if (!Batches_.empty())
    {
        Batch& previous = Batches_.back();
        if (previous.TextureSlot == texture &&
            previous.FirstVertex + previous.VertexCount == firstVertex &&
            previous.ScissorX == ScissorX_ && previous.ScissorY == ScissorY_ &&
            previous.ScissorWidth == ScissorWidth_ && previous.ScissorHeight == ScissorHeight_)
        {
            previous.VertexCount += count;
            Vertices_.insert(Vertices_.end(), vertices, vertices + count);
            return;
        }
    }
    Batch batch{};
    batch.FirstVertex = firstVertex;
    batch.VertexCount = count;
    batch.ScissorX = ScissorX_;
    batch.ScissorY = ScissorY_;
    batch.ScissorWidth = ScissorWidth_;
    batch.ScissorHeight = ScissorHeight_;
    batch.TextureSlot = texture;
    Vertices_.insert(Vertices_.end(), vertices, vertices + count);
    Batches_.push_back(batch);
}

void ImguiRenderVulkan::SetScissor(int x, int y, int width, int height)
{
    const int left = std::clamp(x, 0, Width_);
    const int top = std::clamp(Height_ - (y + height), 0, Height_);
    const int right = std::clamp(x + width, left, Width_);
    const int bottom = std::clamp(Height_ - y, top, Height_);
    ScissorX_ = left;
    ScissorY_ = top;
    ScissorWidth_ = right - left;
    ScissorHeight_ = bottom - top;
}

float ImguiRenderVulkan::ToScreenY(float y) const noexcept
{
    return static_cast<float>(Height_) - y;
}

void ImguiRenderVulkan::AddRect(float x, float y, float width, float height, std::uint32_t color)
{
    const float left = x;
    const float right = x + width;
    const float top = ToScreenY(y + height);
    const float bottom = ToScreenY(y);
    const Vertex vertices[6] = {
        {{left, top}, {0.0f, 0.0f}, color}, {{right, top}, {0.0f, 0.0f}, color},
        {{right, bottom}, {0.0f, 0.0f}, color}, {{left, top}, {0.0f, 0.0f}, color},
        {{right, bottom}, {0.0f, 0.0f}, color}, {{left, bottom}, {0.0f, 0.0f}, color}};
    AddGeometry(vertices, 6, Texture::White);
}

void ImguiRenderVulkan::AddRoundedRect(
    float x, float y, float width, float height, float radius, std::uint32_t color)
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
        {right - radius, top + radius}, {right - radius, bottom - radius},
        {left + radius, bottom - radius}, {left + radius, top + radius}};
    constexpr float startAngles[4] = {-Pi * 0.5f, 0.0f, Pi * 0.5f, Pi};
    constexpr int segments = 6;
    std::vector<Vertex> outline;
    outline.reserve(28);
    for (int corner = 0; corner < 4; ++corner)
    {
        for (int segment = 0; segment <= segments; ++segment)
        {
            const float angle = startAngles[corner] + Pi * 0.5f *
                (static_cast<float>(segment) / static_cast<float>(segments));
            outline.push_back({{
                centers[corner][0] + std::cos(angle) * radius,
                centers[corner][1] + std::sin(angle) * radius}, {0.0f, 0.0f}, color});
        }
    }
    const Vertex center{{(left + right) * 0.5f, (top + bottom) * 0.5f}, {0.0f, 0.0f}, color};
    std::vector<Vertex> triangles;
    triangles.reserve(outline.size() * 3U);
    for (std::size_t index = 0; index < outline.size(); ++index)
    {
        triangles.push_back(center);
        triangles.push_back(outline[index]);
        triangles.push_back(outline[(index + 1U) % outline.size()]);
    }
    AddGeometry(triangles.data(), static_cast<std::uint32_t>(triangles.size()), Texture::White);
}

void ImguiRenderVulkan::AddLine(
    float x0, float y0, float x1, float y1, float radius, std::uint32_t color)
{
    y0 = ToScreenY(y0);
    y1 = ToScreenY(y1);
    float dx = x1 - x0;
    float dy = y1 - y0;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length <= 0.0001f)
    {
        return;
    }
    dx /= length;
    dy /= length;
    const float halfWidth = std::max(1.0f, radius) * 0.5f;
    const float nx = -dy * halfWidth;
    const float ny = dx * halfWidth;
    const Vertex vertices[6] = {
        {{x0 - nx, y0 - ny}, {0.0f, 0.0f}, color}, {{x0 + nx, y0 + ny}, {0.0f, 0.0f}, color},
        {{x1 + nx, y1 + ny}, {0.0f, 0.0f}, color}, {{x0 - nx, y0 - ny}, {0.0f, 0.0f}, color},
        {{x1 + nx, y1 + ny}, {0.0f, 0.0f}, color}, {{x1 - nx, y1 - ny}, {0.0f, 0.0f}, color}};
    AddGeometry(vertices, 6, Texture::White);
}

void ImguiRenderVulkan::AddTriangle(
    float x, float y, float width, float height, int flags, std::uint32_t color)
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
    for (Vertex& vertex : vertices)
    {
        vertex.Color = color;
    }
    AddGeometry(vertices, 3, Texture::White);
}

float ImguiRenderVulkan::MeasureText(const char* text) const
{
    float x = 0.0f;
    float length = 0.0f;
    while (text != nullptr && *text != '\0')
    {
        const auto character = static_cast<unsigned char>(*text++);
        if (character == '\t')
        {
            for (float stop : TabStops)
            {
                if (x < stop)
                {
                    x = stop;
                    break;
                }
            }
            length = std::max(length, x);
        }
        else if (character >= 32U && character < 128U)
        {
            const Glyph& glyph = Glyphs_[character - 32U];
            length = std::max(length, x + glyph.Width);
            x += glyph.Advance;
        }
    }
    return length;
}

void ImguiRenderVulkan::AddText(
    float x, float y, int align, const char* text, std::uint32_t color)
{
    if (text == nullptr)
    {
        return;
    }
    if (align == IMGUI_ALIGN_CENTER)
    {
        x -= MeasureText(text) * 0.5f;
    }
    else if (align == IMGUI_ALIGN_RIGHT)
    {
        x -= MeasureText(text);
    }
    const float originX = x;
    const float top = ToScreenY(y + static_cast<float>(FontLineHeight_));
    std::vector<Vertex> vertices;
    vertices.reserve(std::strlen(text) * 6U);
    while (*text != '\0')
    {
        const auto character = static_cast<unsigned char>(*text++);
        if (character == '\t')
        {
            for (float stop : TabStops)
            {
                if (x < originX + stop)
                {
                    x = originX + stop;
                    break;
                }
            }
        }
        else if (character >= 32U && character < 128U)
        {
            const Glyph& glyph = Glyphs_[character - 32U];
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
    AddGeometry(vertices.data(), static_cast<std::uint32_t>(vertices.size()), Texture::Font);
}

} // namespace Evans::Editor

#endif
