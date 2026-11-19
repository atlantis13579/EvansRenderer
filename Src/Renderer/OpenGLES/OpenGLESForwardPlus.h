#pragma once

#include "Renderer/Lighting/ForwardPlusContract.h"
#include <functional>
#include <memory>

namespace Evans::Renderer
{
    // The caller keeps the owning ES 3.1 context current on this thread through
    // destruction. Dispatch runs outside raster work and consumes original D32F
    // samples. It preserves the current program, active texture and framebuffer
    // routing, but clears texture/sampler unit zero, SSBO slots 0..2 and the
    // generic SSBO binding. Consumers rebind those slots and obey overflow policy.
    // Buffer tiles retain top-left ordering, independent of native image origin.
    class OpenGLESForwardPlus
    {
    public:
        using ProcLoader = std::function<void*(const char*)>;
        explicit OpenGLESForwardPlus(TiledLightShader shader = TiledLightShader::ForwardPlus);
        ~OpenGLESForwardPlus();
        OpenGLESForwardPlus(const OpenGLESForwardPlus&) = delete;
        OpenGLESForwardPlus& operator=(const OpenGLESForwardPlus&) = delete;
        bool Initialize(const ProcLoader& loader, std::string& error);
        void Shutdown();
        void BeginFrame();
        void AbortFrame(); // Hides results; issued GL work remains ordered.
        bool Dispatch(std::uint32_t depth, bool multisample, const Matrix4& camera,
            const std::vector<ForwardPlusLightBounds>& lights, const ForwardPlusParameters& parameters, std::string& error);
        std::uint32_t Headers() const;
        std::uint32_t Indices() const;
        ForwardPlusGrid Grid() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
