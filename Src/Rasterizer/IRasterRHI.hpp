#pragma once

#include "Rasterizer/Types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace Evans::Rasterizer
{

enum class RasterBackend
{
    CPU,
    CUDA,
    ComputeShader,
    Metal
};

struct RasterRHICreateInfo
{
    // Zero chooses hardware_concurrency. Backend-specific tile dimensions remain fixed.
    std::uint32_t WorkerThreads = 0;
};

struct RasterBackendCapabilities
{
    RasterBackend Backend = RasterBackend::CPU;
    bool NativeExecutionAvailable = false;
    bool ParallelExecution = true;
    std::uint32_t TileWidth = 0;
    std::uint32_t TileHeight = 0;
    std::string ExecutionDescription;
};

class IRasterRHI
{
public:
    virtual ~IRasterRHI() = default;

    [[nodiscard]] virtual bool Initialize(const RasterRHICreateInfo& createInfo) = 0;
    virtual void Shutdown() noexcept = 0;
    [[nodiscard]] virtual bool IsInitialized() const noexcept = 0;
    [[nodiscard]] virtual RasterBackend Backend() const noexcept = 0;
    [[nodiscard]] virtual std::string_view Name() const noexcept = 0;
    [[nodiscard]] virtual RasterBackendCapabilities Capabilities() const = 0;

    [[nodiscard]] virtual RasterStatus Render(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& cancellation = {}) = 0;

    virtual void Cancel() noexcept = 0;
    virtual void WaitIdle() noexcept = 0;
    [[nodiscard]] virtual RasterStatistics LastStatistics() const = 0;
};

[[nodiscard]] std::unique_ptr<IRasterRHI> CreateRasterRHI(
    RasterBackend backend,
    const RasterRHICreateInfo& createInfo = {});

} // namespace Evans::Rasterizer
