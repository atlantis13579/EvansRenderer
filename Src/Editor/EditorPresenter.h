#pragma once

#include "Editor/Types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace Evans::Editor
{

enum class EditorPresentationAPI : std::uint8_t
{
    DX11,
    DX12,
    Metal,
    OpenGLES,
    Vulkan
};

class IEditorPresenter
{
public:
    virtual ~IEditorPresenter() = default;

    virtual bool Initialize(void* nativeWindow, int width, int height) = 0;
    virtual void Shutdown() noexcept = 0;
    virtual bool Resize(int width, int height) = 0;

    virtual bool BeginFrame(
        const Image& viewportImage,
        std::uint64_t viewportRevision,
        int viewportX,
        int viewportY,
        int viewportWidth,
        int viewportHeight) = 0;
    virtual void DrawImgui() = 0;
    virtual bool EndFrame() = 0;
    virtual bool CaptureBackBuffer(Image& output) = 0;

    [[nodiscard]] virtual bool IsInitialized() const noexcept = 0;
    [[nodiscard]] virtual EditorPresentationAPI API() const noexcept = 0;
    [[nodiscard]] virtual const std::string& LastError() const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<IEditorPresenter> CreateEditorPresenter(EditorPresentationAPI api);
[[nodiscard]] const char* ToString(EditorPresentationAPI api) noexcept;

// Native realtime selections and DXR choose a presentation API. Offscreen
// OpenGL ES, software rasterizer, and ray-tracing selections retain the most
// recent window API because their CPU Image output is presentation-independent.
[[nodiscard]] EditorPresentationAPI ResolvePresentationAPI(
    std::string_view backendId,
    EditorPresentationAPI lastRealtimeAPI) noexcept;

} // namespace Evans::Editor
