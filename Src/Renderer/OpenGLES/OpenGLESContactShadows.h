#pragma once

#include "Renderer/Shadows/ContactShadowContract.h"
#include <functional>
#include <memory>

namespace Evans::Renderer
{
    // The caller owns the current ES 3.1 context and keeps it alive through
    // Shutdown/destruction. No GL objects may be used on another context/thread.
    // Dispatch runs outside a raster pass and consumes an actual D32F texture.
    // It preserves the program, active texture unit and read/draw framebuffers,
    // but clears texture/sampler unit zero and image unit zero. Consumers rebind
    // these slots. Output is bottom-left R32F with one array layer per sample.
    class OpenGLESContactShadows
    {
    public:
        using ProcLoader = std::function<void*(const char*)>;
        OpenGLESContactShadows();
        ~OpenGLESContactShadows();
        OpenGLESContactShadows(const OpenGLESContactShadows&) = delete;
        OpenGLESContactShadows& operator=(const OpenGLESContactShadows&) = delete;
        bool Initialize(const ProcLoader& load, std::string& error);
        void Shutdown();
        void BeginFrame();
        void AbortFrame();
        bool Dispatch(std::uint32_t depth, bool multisample, const ContactShadowDispatch& desc, std::string& error);
        std::uint32_t Result() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
