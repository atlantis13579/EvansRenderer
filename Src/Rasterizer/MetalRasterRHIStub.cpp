#include "Rasterizer/MetalRasterRHI.h"

#if defined(__APPLE__)
#error MetalRasterRHIStub.cpp must not be compiled on Apple platforms
#endif

#include <stdexcept>
#include <utility>

namespace Evans::Rasterizer
{

class MetalRasterRHI::Context
{
public:
    RasterStatistics Statistics;
};

MetalRasterRHI::MetalRasterRHI()
    : Context_(std::make_unique<Context>())
{
}

MetalRasterRHI::~MetalRasterRHI() = default;

bool MetalRasterRHI::Initialize(const RasterRHICreateInfo&)
{
    return false;
}

void MetalRasterRHI::Shutdown() noexcept
{
}

bool MetalRasterRHI::IsInitialized() const noexcept
{
    return false;
}

RasterBackend MetalRasterRHI::Backend() const noexcept
{
    return RasterBackend::Metal;
}

std::string_view MetalRasterRHI::Name() const noexcept
{
    return "Metal Compute Raster Unavailable";
}

RasterBackendCapabilities MetalRasterRHI::Capabilities() const
{
    return {
        RasterBackend::Metal,
        false,
        false,
        8,
        8,
        "Unsupported: Metal compute raster is available only on Apple platforms"};
}

RasterStatus MetalRasterRHI::Render(
    const RasterScene&,
    const RasterSettings&,
    RenderTarget&,
    const RasterCancellationToken&)
{
    throw std::runtime_error(
        "Metal compute raster is available only on Apple platforms");
}

void MetalRasterRHI::Cancel() noexcept
{
}

void MetalRasterRHI::WaitIdle() noexcept
{
}

RasterStatistics MetalRasterRHI::LastStatistics() const
{
    return Context_->Statistics;
}

} // namespace Evans::Rasterizer
