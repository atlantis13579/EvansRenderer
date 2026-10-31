#include "Rasterizer/Backends.hpp"

#include <stdexcept>

namespace Evans::Rasterizer
{

std::unique_ptr<IRasterRHI> CreateRasterRHI(
    RasterBackend backend,
    const RasterRHICreateInfo& createInfo)
{
    std::unique_ptr<IRasterRHI> result;
    switch (backend)
    {
    case RasterBackend::CPU:
        result = std::make_unique<CpuRasterRHI>();
        break;
    case RasterBackend::CUDA:
        result = std::make_unique<CudaRasterRHI>();
        break;
    case RasterBackend::ComputeShader:
        result = std::make_unique<ComputeShaderRasterRHI>();
        break;
    case RasterBackend::Metal:
        result = std::make_unique<MetalRasterRHI>();
        break;
    }
    if (!result)
    {
        throw std::invalid_argument("Unknown software raster backend");
    }
    // Concrete backends initialize their default context in the constructor.
    // Reinitialize only when the caller requests a non-default worker policy.
    if (createInfo.WorkerThreads != 0)
    {
        (void)result->Initialize(createInfo);
    }
    return result;
}

} // namespace Evans::Rasterizer
