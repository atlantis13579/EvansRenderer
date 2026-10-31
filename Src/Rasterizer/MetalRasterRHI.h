#pragma once

#include "Rasterizer/IRasterRHI.hpp"

#include <memory>

namespace Evans::Rasterizer
{

class MetalRasterRHI final : public IRasterRHI
{
public:
    MetalRasterRHI();
    ~MetalRasterRHI() override;

    MetalRasterRHI(const MetalRasterRHI&) = delete;
    MetalRasterRHI& operator=(const MetalRasterRHI&) = delete;

    bool Initialize(const RasterRHICreateInfo& createInfo) override;
    void Shutdown() noexcept override;
    [[nodiscard]] bool IsInitialized() const noexcept override;
    [[nodiscard]] RasterBackend Backend() const noexcept override;
    [[nodiscard]] std::string_view Name() const noexcept override;
    [[nodiscard]] RasterBackendCapabilities Capabilities() const override;
    [[nodiscard]] RasterStatus Render(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& cancellation = {}) override;
    void Cancel() noexcept override;
    void WaitIdle() noexcept override;
    [[nodiscard]] RasterStatistics LastStatistics() const override;

private:
    class Context;
    std::unique_ptr<Context> Context_;
};

} // namespace Evans::Rasterizer
