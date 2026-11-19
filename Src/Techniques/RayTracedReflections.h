#pragma once

#include <algorithm>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr std::uint32_t MinimumReflectionRays=1,MaximumReflectionRays=64;
    struct RayTracedReflectionSettings
    {
        std::uint32_t RayCount=4;
        bool Shadows=true;
    };
    inline RayTracedReflectionSettings SanitizeRayTracedReflectionSettings(RayTracedReflectionSettings settings) noexcept
    { settings.RayCount=std::clamp(settings.RayCount,MinimumReflectionRays,MaximumReflectionRays); return settings; }
    inline bool operator==(RayTracedReflectionSettings a,RayTracedReflectionSettings b) noexcept
    { return a.RayCount==b.RayCount && a.Shadows==b.Shadows; }
    enum class RayReflectionPass : std::uint32_t { Trace,Horizontal,Vertical,Composite };
}
