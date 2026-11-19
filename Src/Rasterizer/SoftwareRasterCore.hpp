#pragma once

#include "Rasterizer/Types.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace Evans::Rasterizer
{
namespace Detail { class ParallelExecutor; }

struct ScreenTriangle;

struct CpuRasterResources
{
    std::shared_ptr<Detail::ParallelExecutor> Workers;
    std::vector<std::vector<const ScreenTriangle*>> TileBins, ShadowTileBins;
    std::vector<std::uint8_t> SelectionMask;
};

enum class CpuRasterPass
{
    Full,
    SelectionOutline,
    PresentationOverlay
};

class SoftwareRasterCore
{
public:
    [[nodiscard]] static RasterStatus RenderCpuTiled(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& externalCancellation,
        const std::atomic<bool>& backendCancellation,
        std::uint32_t workerThreads,
        CpuRasterResources& resources,
        RasterStatistics& statistics,
        CpuRasterPass pass = CpuRasterPass::Full);

};

} // namespace Evans::Rasterizer
