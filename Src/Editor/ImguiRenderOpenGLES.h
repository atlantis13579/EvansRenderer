#pragma once

#if defined(_WIN32)

#include <memory>

namespace Evans::Editor
{

using OpenGLESProcResolver = void* (*)(void* userData, const char* name);

class ImguiRenderOpenGLES final
{
public:
    ImguiRenderOpenGLES();
    ~ImguiRenderOpenGLES();

    ImguiRenderOpenGLES(const ImguiRenderOpenGLES&) = delete;
    ImguiRenderOpenGLES& operator=(const ImguiRenderOpenGLES&) = delete;

    [[nodiscard]] bool Initialize(
        void* resolverUserData,
        OpenGLESProcResolver resolver);
    void Shutdown() noexcept;
    void Draw(int width, int height);

private:
    struct Impl;
    std::unique_ptr<Impl> Impl_;
};

} // namespace Evans::Editor

#endif
