#pragma once

#include "Rasterizer/Backends.hpp"

#include <memory>
#include <mutex>

namespace Evans::Rasterizer
{

class SoftwareRasterizer
{
public:
    explicit SoftwareRasterizer(
        RasterBackend backend = RasterBackend::CPU,
        const RasterRHICreateInfo& createInfo = {});
    ~SoftwareRasterizer();

    SoftwareRasterizer(const SoftwareRasterizer&) = delete;
    SoftwareRasterizer& operator=(const SoftwareRasterizer&) = delete;
    SoftwareRasterizer(SoftwareRasterizer&&) = delete;
    SoftwareRasterizer& operator=(SoftwareRasterizer&&) = delete;

    void SelectBackend(RasterBackend backend, const RasterRHICreateInfo& createInfo = {});
    [[nodiscard]] RasterBackend SelectedBackend() const noexcept;
    [[nodiscard]] std::string_view BackendName() const noexcept;
    [[nodiscard]] RasterBackendCapabilities Capabilities() const;
    [[nodiscard]] RasterStatus Render(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& cancellation = {});
    void Cancel() noexcept;
    void WaitIdle() noexcept;
    [[nodiscard]] RasterStatistics LastStatistics() const;

private:
    mutable std::mutex Mutex_;
    std::shared_ptr<IRasterRHI> RHI_;
};

} // namespace Evans::Rasterizer
