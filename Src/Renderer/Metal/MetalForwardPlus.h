#pragma once
#include "Renderer/Lighting/ForwardPlusContract.h"
#include <memory>

namespace Evans::Renderer
{
    // Apple-only component used on its initializing thread, with borrowed native
    // handles exposed as void*.
    // Record consumes base-mip tracked D32 depth, ends its own serial compute
    // encoder and retains the command buffer and depth through completion.
    // All depth producers and list consumers use that same MTLCommandBuffer;
    // no other encoder may be open during Record. The buffer is not enqueued
    // before Record, and the caller commits it after encoding its consumers.
    // BeginFrame/Shutdown reject unsubmitted or pending work. AbortFrame only
    // follows discarding an unsubmitted command buffer; it cannot cancel work.
    // Device-created tracked resources order GPU accesses. CPU uploads cannot
    // be reused until the retained command buffer completes successfully.
    class MetalForwardPlus
    {
    public:
        explicit MetalForwardPlus(TiledLightShader shader = TiledLightShader::ForwardPlus);
        ~MetalForwardPlus(); // Waits for submitted work; discard unsubmitted work first.
        MetalForwardPlus(const MetalForwardPlus&) = delete;
        MetalForwardPlus& operator=(const MetalForwardPlus&) = delete;
        bool Initialize(void* device, std::string& error);
        bool BeginFrame(std::string& error);
        bool AbortFrame(std::string& error);
        bool Shutdown(std::string& error);
        bool Record(void* commandBuffer, void* depthTexture, const Matrix4& camera,
            const std::vector<ForwardPlusLightBounds>& lights, const ForwardPlusParameters& parameters, std::string& error);
        void* Headers() const; // Borrowed id<MTLBuffer>, ForwardPlusTile records.
        void* Indices() const; // Borrowed id<MTLBuffer>, 32-bit scene-light indices.
        ForwardPlusGrid Grid() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
