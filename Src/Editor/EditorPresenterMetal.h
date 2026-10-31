#pragma once

#if defined(__APPLE__)

#include "Editor/EditorPresenter.h"

#include <memory>

namespace Evans::Editor
{

class EditorPresenterMetal final : public IEditorPresenter
{
public:
    EditorPresenterMetal();
    ~EditorPresenterMetal() override;

    EditorPresenterMetal(const EditorPresenterMetal&) = delete;
    EditorPresenterMetal& operator=(const EditorPresenterMetal&) = delete;

    bool Initialize(void* nativeWindow, int width, int height) override;
    void Shutdown() noexcept override;
    bool Resize(int width, int height) override;
    bool BeginFrame(
        const Image& viewportImage,
        std::uint64_t viewportRevision,
        int viewportX,
        int viewportY,
        int viewportWidth,
        int viewportHeight) override;
    void DrawImgui() override;
    bool EndFrame() override;
    bool CaptureBackBuffer(Image& output) override;

    [[nodiscard]] bool IsInitialized() const noexcept override;
    [[nodiscard]] EditorPresentationAPI API() const noexcept override
    {
        return EditorPresentationAPI::Metal;
    }
    [[nodiscard]] const std::string& LastError() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> Impl_;
};

} // namespace Evans::Editor

#endif
