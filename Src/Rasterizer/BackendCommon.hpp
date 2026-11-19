#pragma once

#include "Rasterizer/Types.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace Evans::Rasterizer::Detail
{
class ParallelExecutor;

class ExecutionGate
{
public:
    bool Initialize(std::uint32_t requestedWorkerThreads);
    void Shutdown() noexcept;
    [[nodiscard]] bool IsInitialized() const noexcept;
    [[nodiscard]] std::uint32_t WorkerThreads() const noexcept;

    template <typename Function>
    [[nodiscard]] RasterStatus Execute(Function&& function)
    {
        auto cancellation = std::make_shared<std::atomic<bool>>(false);
        {
            std::lock_guard<std::mutex> lock(StateMutex_);
            if (!Initialized_ || ShuttingDown_)
            {
                throw std::runtime_error("Raster RHI is not initialized");
            }
            ++PendingSubmissions_;
        }

        std::unique_lock<std::mutex> executionLock(ExecutionMutex_);
        {
            std::lock_guard<std::mutex> lock(StateMutex_);
            ActiveCancellation_ = cancellation;
            if (ShuttingDown_)
            {
                cancellation->store(true, std::memory_order_relaxed);
            }
        }

        RasterStatus status = RasterStatus::Cancelled;
        try
        {
            status = function(*cancellation);
        }
        catch (...)
        {
            FinishSubmission(cancellation);
            throw;
        }
        FinishSubmission(cancellation);
        return status;
    }

    void Cancel() noexcept;
    void WaitIdle() noexcept;

private:
    void FinishSubmission(const std::shared_ptr<std::atomic<bool>>& cancellation) noexcept;

    mutable std::mutex StateMutex_;
    std::mutex ExecutionMutex_;
    std::condition_variable IdleCondition_;
    std::shared_ptr<std::atomic<bool>> ActiveCancellation_;
    std::uint32_t WorkerThreads_ = 1;
    std::uint32_t PendingSubmissions_ = 0;
    bool Initialized_ = false;
    bool ShuttingDown_ = false;
};

class StatisticsStore
{
public:
    void Set(RasterStatistics statistics);
    [[nodiscard]] RasterStatistics Get() const;

private:
    mutable std::mutex Mutex_;
    RasterStatistics Statistics_;
};

void ValidateRenderTarget(const RenderTarget& target);

// Composite an occlusion-aware silhouette over a native GPU color framebuffer.
void ApplyGpuSelectionOutline(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation);

struct PreparedGpuRasterScenes
{
    RasterScene Main;
    RasterScene Shadow;
    RasterSettings Settings;
    bool Cancelled = false;
};

// Clips local or clip-space geometry before it is uploaded to GPU raster
// backends. The GPU kernels can then consume only finite, in-frustum triangles
// while retaining world-space attributes for lighting and shadow lookup.
[[nodiscard]] PreparedGpuRasterScenes PrepareGpuRasterScenes(
    const RasterScene& scene,
    const RasterSettings& settings,
    bool prepareShadowScene,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation, ParallelExecutor* workers = nullptr);

} // namespace Evans::Rasterizer::Detail
