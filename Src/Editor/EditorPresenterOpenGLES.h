#pragma once

#if defined(_WIN32)

#include "Editor/EditorPresenter.h"

#include <cstdint>
#include <memory>
#include <string>

namespace Evans::Editor
{

class EditorPresenterOpenGLES final : public IEditorPresenter
{
public:
    EditorPresenterOpenGLES();
    ~EditorPresenterOpenGLES();

    EditorPresenterOpenGLES(const EditorPresenterOpenGLES&) = delete;
    EditorPresenterOpenGLES& operator=(const EditorPresenterOpenGLES&) = delete;

    [[nodiscard]] bool Initialize(void* nativeWindow, int width, int height) override;
    void Shutdown() noexcept override;
    [[nodiscard]] bool Resize(int width, int height) override;

    [[nodiscard]] bool BeginFrame(
        const Image& viewportImage,
        std::uint64_t viewportRevision,
        int viewportX,
        int viewportY,
        int viewportWidth,
        int viewportHeight) override;
    void DrawImgui() override;
    [[nodiscard]] bool EndFrame() override;

    [[nodiscard]] bool IsInitialized() const noexcept override;
    [[nodiscard]] EditorPresentationAPI API() const noexcept override
    {
        return EditorPresentationAPI::OpenGLES;
    }
    [[nodiscard]] const std::string& LastError() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> Impl_;
};

} // namespace Evans::Editor

#endif
