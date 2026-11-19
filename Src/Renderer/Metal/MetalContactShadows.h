#pragma once

#include "Renderer/Shadows/ContactShadowContract.h"
#include <memory>

namespace Evans::Renderer
{
    // Apple-only component. Native types stay behind the PIMPL so the C++ API does not need
    // Objective-C headers. Handles are borrowed id<MTLDevice/CommandBuffer/Texture>.
    // Record ends its compute encoder; no other encoder may be open. Default
    // tracked resources on one command buffer order depth -> compute -> lighting.
    // Present must wait for completion before BeginFrame. AbortFrame is only for
    // a discarded, unsubmitted buffer. A result is one R32Float array layer per
    // depth sample, never a resolved visibility image. The caller owns the depth
    // texture and must retain it through completion, with default sample positions.
    class MetalContactShadows
    {
    public:
        MetalContactShadows();
        ~MetalContactShadows();
        MetalContactShadows(const MetalContactShadows&) = delete;
        MetalContactShadows& operator=(const MetalContactShadows&) = delete;
        bool Initialize(void* device, std::string& error);
        void Shutdown();
        bool BeginFrame(std::string& error);
        bool AbortFrame(std::string& error);
        bool Record(void* commandBuffer, void* depthTexture, const ContactShadowDispatch& trace, std::string& error);
        void* Result() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
