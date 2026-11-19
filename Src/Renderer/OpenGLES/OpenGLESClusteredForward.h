#pragma once

#include "Renderer/Lighting/ClusteredForwardContract.h"
#include <functional>
#include <memory>

namespace Evans::Renderer
{
    // Keep the owning ES 3.1 context current through destruction. Dispatch
    // preserves the current program and clears UBO 0 / SSBO 0..2. No texture or
    // framebuffer state changes; subsequent consumers rebind buffer slots.
    class OpenGLESClusteredForward
    {
    public:
        using ProcLoader = std::function<void*(const char*)>;
        OpenGLESClusteredForward();
        ~OpenGLESClusteredForward();
        OpenGLESClusteredForward(const OpenGLESClusteredForward&) = delete;
        OpenGLESClusteredForward& operator=(const OpenGLESClusteredForward&) = delete;
        bool Initialize(const ProcLoader& loader, std::string& error);
        void Shutdown();
        void BeginFrame();
        void AbortFrame(); // Hides results; issued GL work remains ordered.
        bool Dispatch(std::uint32_t width, std::uint32_t height, const Matrix4& camera,
            const std::vector<ForwardPlusLightBounds>& lights, const ClusteredForwardParameters& parameters, std::string& error);
        std::uint32_t Headers() const;
        std::uint32_t Indices() const;
        ClusteredForwardGrid Grid() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
