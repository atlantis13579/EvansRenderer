#pragma once

#include <algorithm>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr std::uint32_t MinimumAORays=4,MaximumAORays=128;
    struct RayTracedAOSettings { std::uint32_t RayCount=16; };
    inline RayTracedAOSettings SanitizeRayTracedAOSettings(RayTracedAOSettings settings) noexcept
    { settings.RayCount=std::clamp(settings.RayCount,MinimumAORays,MaximumAORays); return settings; }
    inline bool operator==(RayTracedAOSettings a,RayTracedAOSettings b) noexcept { return a.RayCount==b.RayCount; }
}
