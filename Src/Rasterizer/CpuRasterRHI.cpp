#include "Rasterizer/CpuRasterRHI.h"

#include "BackendCommon.hpp"
#include "SoftwareRasterCore.hpp"

#include <atomic>
#include <utility>

namespace Evans::Rasterizer
{

class CpuRasterRHI::Context
{
public:
    Detail::ExecutionGate Execution;
    CpuRasterResources Resources;
    Detail::StatisticsStore Statistics;
};

CpuRasterRHI::CpuRasterRHI()
    : Context_(std::make_unique<Context>())
{
    (void)Initialize({});
}

CpuRasterRHI::~CpuRasterRHI()
{
    Shutdown();
}

bool CpuRasterRHI::Initialize(const RasterRHICreateInfo& createInfo)
{
    return Context_->Execution.Initialize(createInfo.WorkerThreads);
}

void CpuRasterRHI::Shutdown() noexcept
{
    Context_->Execution.Shutdown();
}

bool CpuRasterRHI::IsInitialized() const noexcept
{
    return Context_->Execution.IsInitialized();
}

RasterBackend CpuRasterRHI::Backend() const noexcept
{
    return RasterBackend::CPU;
}

std::string_view CpuRasterRHI::Name() const noexcept
{
    return "CPU Parallel Tiled";
}

RasterBackendCapabilities CpuRasterRHI::Capabilities() const
{
    return {
        RasterBackend::CPU,
        true,
        true,
        16,
        16,
        "Native C++ worker dispatch over binned 16x16 tiles"};
}

RasterStatus CpuRasterRHI::Render(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& cancellation)
{
    Detail::ValidateRenderTarget(target);
    RasterStatistics statistics;
    const RasterStatus status = Context_->Execution.Execute(
        [&](const std::atomic<bool>& backendCancellation)
        {
            return SoftwareRasterCore::RenderCpuTiled(
                scene,
                settings,
                target,
                cancellation,
                backendCancellation,
                Context_->Execution.WorkerThreads(),
                Context_->Resources,
                statistics);
        });
    Context_->Statistics.Set(std::move(statistics));
    return status;
}

void CpuRasterRHI::Cancel() noexcept
{
    Context_->Execution.Cancel();
}

void CpuRasterRHI::WaitIdle() noexcept
{
    Context_->Execution.WaitIdle();
}

RasterStatistics CpuRasterRHI::LastStatistics() const
{
    return Context_->Statistics.Get();
}

} // namespace Evans::Rasterizer
