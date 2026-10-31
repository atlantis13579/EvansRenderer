#pragma once

#if defined(_WIN32)

#include <cstdint>
#include <vector>

namespace Evans::Editor
{

// Converts Riemann's immediate-mode pull queue into Vulkan-ready triangles.
// GPU ownership stays in EditorPresenterVulkan so this class remains ABI-free.
class ImguiRenderVulkan final
{
public:
    enum class Texture : std::uint8_t
    {
        White,
        Font
    };

    struct Vertex
    {
        float Position[2];
        float TexCoord[2];
        std::uint32_t Color;
    };

    struct Batch
    {
        std::uint32_t FirstVertex = 0;
        std::uint32_t VertexCount = 0;
        int ScissorX = 0;
        int ScissorY = 0;
        int ScissorWidth = 0;
        int ScissorHeight = 0;
        Texture TextureSlot = Texture::White;
    };

    [[nodiscard]] bool Initialize();
    void Build(int width, int height);

    [[nodiscard]] const std::vector<Vertex>& Vertices() const noexcept { return Vertices_; }
    [[nodiscard]] const std::vector<Batch>& Batches() const noexcept { return Batches_; }
    [[nodiscard]] const std::vector<std::uint8_t>& FontPixels() const noexcept { return FontPixels_; }
    [[nodiscard]] int FontWidth() const noexcept { return FontWidth_; }
    [[nodiscard]] int FontHeight() const noexcept { return FontAtlasHeight_; }

private:
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

    void AddGeometry(const Vertex* vertices, std::uint32_t count, Texture texture);
    void SetScissor(int x, int y, int width, int height);
    void AddRect(float x, float y, float width, float height, std::uint32_t color);
    void AddRoundedRect(float x, float y, float width, float height, float radius, std::uint32_t color);
    void AddLine(float x0, float y0, float x1, float y1, float radius, std::uint32_t color);
    void AddTriangle(float x, float y, float width, float height, int flags, std::uint32_t color);
    void AddText(float x, float y, int align, const char* text, std::uint32_t color);
    [[nodiscard]] float MeasureText(const char* text) const;
    [[nodiscard]] float ToScreenY(float y) const noexcept;

    int Width_ = 1;
    int Height_ = 1;
    int FontWidth_ = 512;
    int FontAtlasHeight_ = 512;
    int FontLineHeight_ = 16;
    int ScissorX_ = 0;
    int ScissorY_ = 0;
    int ScissorWidth_ = 1;
    int ScissorHeight_ = 1;
    Glyph Glyphs_[96]{};
    std::vector<std::uint8_t> FontPixels_;
    std::vector<Vertex> Vertices_;
    std::vector<Batch> Batches_;
};

} // namespace Evans::Editor

#endif
