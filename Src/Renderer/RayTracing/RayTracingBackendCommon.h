#pragma once

#include "Renderer/RayTracing/RayTracing.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Evans
{
namespace RayTracing
{
namespace Detail
{
    using CancellationProbe = std::function<bool()>;

    struct alignas(16) PackedRayTextureDescriptor
    {
        uint32_t OffsetTexels = 0;
        uint32_t Width = 0;
        uint32_t Height = 0;
        uint32_t Padding = 0;
    };

    struct PackedRayTextureAtlas
    {
        std::vector<PackedRayTextureDescriptor> Descriptors;
        // Each uint stores RGBA8 with red in the least-significant byte.
        std::vector<uint32_t> Texels;
    };

    static_assert(
        sizeof(PackedRayTextureDescriptor) == 16 &&
            alignof(PackedRayTextureDescriptor) == 16,
        "Ray texture descriptor ABI mismatch.");

    [[nodiscard]] bool BuildRayTextureAtlas(
        const Scene& scene,
        PackedRayTextureAtlas& atlas,
        std::string& error);

    RenderResult MakeFailedResult(std::string error);
    RenderResult MakeCancelledResult();
    uint32_t GetBoundedWorkerCount(uint32_t maximum);
    CancellationProbe CombineCancellation(
        const CancellationToken& queued,
        const CancellationToken& external);

    class PersistentRenderQueue
    {
    public:
        using Task = std::function<RenderResult(const CancellationToken&)>;

        PersistentRenderQueue();
        ~PersistentRenderQueue();

        PersistentRenderQueue(const PersistentRenderQueue&) = delete;
        PersistentRenderQueue& operator=(const PersistentRenderQueue&) = delete;

        RenderJob Submit(Task task) const;
        void WaitIdle() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
}
}
