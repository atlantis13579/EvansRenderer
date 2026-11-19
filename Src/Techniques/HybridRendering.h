#pragma once

#include <cstdint>

namespace Evans::Techniques
{
    // Raster primary visibility, stochastic screen-space intersections and scene
    // rays for missing/unreliable samples. Both estimators share the same GGX ray;
    // incoming radiance is mixed before applying its single BRDF/PDF weight.
    // https://gpuopen.com/fidelityfx-hybrid-reflections/
    // This implementation uses the project's shader BVH, without a vendor SDK.
    struct HybridRenderingSettings
    {
        bool ScreenSpace = true;
        bool RayFallback = true;
    };
    inline bool operator==(HybridRenderingSettings a, HybridRenderingSettings b) noexcept
    { return a.ScreenSpace == b.ScreenSpace && a.RayFallback == b.RayFallback; }
    enum class HybridRenderingPass : std::uint32_t { Trace, Horizontal, Vertical, Composite };
}
