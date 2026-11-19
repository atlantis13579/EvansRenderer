#include "Renderer/RayTracing/RayTracingBackendCommon.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace Evans
{
namespace RayTracing
{
namespace Detail
{
    bool BuildRayTextureAtlas(
        const Scene& scene,
        PackedRayTextureAtlas& atlas,
        std::string& error)
    {
        if (scene.Textures.size() >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        {
            error = "Ray texture table is too large.";
            return false;
        }

        uint64_t texelCount = 0;
        for (const RayTexture& texture : scene.Textures)
        {
            if (!texture.IsValid())
            {
                continue;
            }
            texelCount += static_cast<uint64_t>(texture.Width) * texture.Height;
            if (texelCount > std::numeric_limits<uint32_t>::max())
            {
                error = "Ray texture atlas is too large.";
                return false;
            }
        }

        PackedRayTextureAtlas result;
        result.Descriptors.resize(scene.Textures.size());
        result.Texels.reserve(static_cast<size_t>(texelCount));
        for (size_t textureIndex = 0;
             textureIndex < scene.Textures.size();
             ++textureIndex)
        {
            const RayTexture& texture = scene.Textures[textureIndex];
            if (!texture.IsValid())
            {
                continue;
            }

            PackedRayTextureDescriptor& descriptor =
                result.Descriptors[textureIndex];
            descriptor.OffsetTexels =
                static_cast<uint32_t>(result.Texels.size());
            descriptor.Width = texture.Width;
            descriptor.Height = texture.Height;

            const std::vector<uint8_t>& source = *texture.PixelsRGBA8;
            const size_t sourceTexelCount =
                static_cast<size_t>(texture.Width) * texture.Height;
            for (size_t texel = 0; texel < sourceTexelCount; ++texel)
            {
                const size_t byte = texel * 4U;
                result.Texels.push_back(
                    static_cast<uint32_t>(source[byte]) |
                    (static_cast<uint32_t>(source[byte + 1U]) << 8U) |
                    (static_cast<uint32_t>(source[byte + 2U]) << 16U) |
                    (static_cast<uint32_t>(source[byte + 3U]) << 24U));
            }
        }
        atlas = std::move(result);
        error.clear();
        return true;
    }

    RenderResult MakeFailedResult(std::string error)
    {
        RenderResult result;
        result.Status = RenderStatus::Failed;
        result.Error = std::move(error);
        return result;
    }

    RenderResult MakeCancelledResult()
    {
        RenderResult result;
        result.Status = RenderStatus::Cancelled;
        return result;
    }

    uint32_t GetBoundedWorkerCount(uint32_t maximum)
    {
        const uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
        return std::max(1u, std::min(hardware, maximum));
    }

    CancellationProbe CombineCancellation(
        const CancellationToken& queued,
        const CancellationToken& external)
    {
        return [queued, external]() {
            return queued.IsCancellationRequested() || external.IsCancellationRequested();
        };
    }

    struct PersistentRenderQueue::Impl
    {
        struct QueueItem
        {
            Task TaskFunction;
            CancellationSource Cancellation;
            std::shared_ptr<std::promise<RenderResult>> Promise;
        };

        Impl()
            : Thread(&Impl::ThreadMain, this)
        {
        }

        ~Impl()
        {
            Shutdown();
        }

        RenderJob Submit(Task task) const
        {
            CancellationSource cancellation;
            auto promise = std::make_shared<std::promise<RenderResult>>();
            std::future<RenderResult> future = promise->get_future();
            if (!task)
            {
                promise->set_value(MakeFailedResult("A ray tracing queue task must not be empty."));
                return RenderJob(cancellation, std::move(future));
            }

            {
                std::lock_guard<std::mutex> lock(Mutex);
                if (StopRequested)
                {
                    cancellation.Cancel();
                    promise->set_value(MakeCancelledResult());
                    return RenderJob(cancellation, std::move(future));
                }
                QueueItem item;
                item.TaskFunction = std::move(task);
                item.Cancellation = cancellation;
                item.Promise = std::move(promise);
                Queue.push_back(std::move(item));
            }
            WorkCondition.notify_one();
            return RenderJob(cancellation, std::move(future));
        }

        void WaitIdle() const
        {
            std::unique_lock<std::mutex> lock(Mutex);
            IdleCondition.wait(lock, [this]() { return Queue.empty() && !Active; });
        }

        void Shutdown()
        {
            {
                std::lock_guard<std::mutex> lock(Mutex);
                if (!StopRequested)
                {
                    StopRequested = true;
                    if (CurrentCancellation) CurrentCancellation->Cancel();
                    for (const QueueItem& item : Queue) item.Cancellation.Cancel();
                }
            }
            WorkCondition.notify_all();
            if (Thread.joinable()) Thread.join();
        }

        void ThreadMain()
        {
            while (true)
            {
                QueueItem item;
                {
                    std::unique_lock<std::mutex> lock(Mutex);
                    WorkCondition.wait(lock, [this]() {
                        return StopRequested || !Queue.empty();
                    });
                    if (StopRequested && Queue.empty()) break;
                    item = std::move(Queue.front());
                    Queue.pop_front();
                    CurrentCancellation = item.Cancellation;
                    Active = true;
                }

                RenderResult result;
                try
                {
                    result = item.TaskFunction(item.Cancellation.GetToken());
                }
                catch (const std::exception& exception)
                {
                    result = MakeFailedResult(
                        std::string("Ray tracing queue failure: ") + exception.what());
                }
                catch (...)
                {
                    result = MakeFailedResult(
                        "Ray tracing queue failed with an unknown exception.");
                }
                try
                {
                    item.Promise->set_value(std::move(result));
                }
                catch (...)
                {
                    // An abandoned promise must not terminate the persistent backend thread.
                }

                {
                    std::lock_guard<std::mutex> lock(Mutex);
                    CurrentCancellation.reset();
                    Active = false;
                }
                IdleCondition.notify_all();
            }

            {
                std::lock_guard<std::mutex> lock(Mutex);
                Active = false;
                CurrentCancellation.reset();
            }
            IdleCondition.notify_all();
        }

        mutable std::mutex Mutex;
        mutable std::condition_variable WorkCondition;
        mutable std::condition_variable IdleCondition;
        mutable std::deque<QueueItem> Queue;
        mutable std::optional<CancellationSource> CurrentCancellation;
        mutable bool Active = false;
        bool StopRequested = false;
        std::thread Thread;
    };

    PersistentRenderQueue::PersistentRenderQueue()
        : m_Impl(std::make_unique<Impl>())
    {
    }

    PersistentRenderQueue::~PersistentRenderQueue() = default;

    RenderJob PersistentRenderQueue::Submit(Task task) const
    {
        return m_Impl->Submit(std::move(task));
    }

    void PersistentRenderQueue::WaitIdle() const
    {
        m_Impl->WaitIdle();
    }
}
}
}
