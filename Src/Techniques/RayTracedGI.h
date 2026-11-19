#pragma once

#include <algorithm>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr std::uint32_t MinimumGIRays=1,MaximumGIRays=64;
    inline constexpr std::uint32_t MinimumDiffuseBounces=1,MaximumDiffuseBounces=4;
    struct RayTracedGISettings
    {
        std::uint32_t RayCount=4;
        std::uint32_t BounceCount=2;
        bool Shadows=true;
    };
    inline RayTracedGISettings SanitizeRayTracedGISettings(RayTracedGISettings settings) noexcept
    {
        settings.RayCount=std::clamp(settings.RayCount,MinimumGIRays,MaximumGIRays);
        settings.BounceCount=std::clamp(settings.BounceCount,MinimumDiffuseBounces,MaximumDiffuseBounces);
        return settings;
    }
    inline bool operator==(RayTracedGISettings a,RayTracedGISettings b) noexcept
    { return a.RayCount==b.RayCount && a.BounceCount==b.BounceCount && a.Shadows==b.Shadows; }
    enum class RayGIPass : std::uint32_t { Trace,Horizontal,Vertical,Composite };
}
