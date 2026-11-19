#pragma once
#include "Renderer/Lighting/ClusteredForwardContract.h"
#include <memory>

namespace Evans::Renderer
{
    // Apple-only component used on its initializing thread, with borrowed native
    // handles exposed as void*.
    // Record needs no depth image. It ends its own serial compute encoder and
    // retains the command buffer through completion. All list consumers use that same MTLCommandBuffer;
    // no other encoder may be open during Record. The buffer is not enqueued
    // before Record, and the caller commits it after encoding its consumers.
    // BeginFrame/Shutdown reject unsubmitted or pending work. AbortFrame only
    // follows discarding an unsubmitted command buffer; it cannot cancel work.
    // Device-created tracked resources order GPU accesses. CPU uploads cannot
    // be reused until the retained command buffer completes successfully.
    class MetalClusteredForward
    {
    public:
        MetalClusteredForward();
        ~MetalClusteredForward(); // Waits for submitted work; discard unsubmitted work first.
        MetalClusteredForward(const MetalClusteredForward&) = delete;
        MetalClusteredForward& operator=(const MetalClusteredForward&) = delete;
        bool Initialize(void* device, std::string& error);
        bool BeginFrame(std::string& error);
        bool AbortFrame(std::string& error);
        bool Shutdown(std::string& error);
        bool Record(void* commandBuffer, std::uint32_t width, std::uint32_t height, const Matrix4& camera,
            const std::vector<ForwardPlusLightBounds>& lights, const ClusteredForwardParameters& parameters, std::string& error);
        void* Headers() const; // Borrowed id<MTLBuffer>, boundary prefix followed by ClusteredForwardHeader records.
        void* Indices() const; // Borrowed id<MTLBuffer>, 32-bit scene-light indices.
        ClusteredForwardGrid Grid() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
