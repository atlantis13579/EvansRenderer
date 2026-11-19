#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr std::uint32_t MaximumTessellationSegments=64;
    struct TessellationSettings
    {
        bool Enabled=false;
        std::uint32_t Segments=2;
        float Curvature=1; // blend between the linear triangle and its cubic PN position patch
    };
    inline bool operator==(const TessellationSettings& a,const TessellationSettings& b) noexcept
    { return a.Enabled==b.Enabled && a.Segments==b.Segments && a.Curvature==b.Curvature; }
    inline bool operator!=(const TessellationSettings& a,const TessellationSettings& b) noexcept { return !(a==b); }
    inline TessellationSettings SanitizeTessellationSettings(TessellationSettings settings) noexcept
    {
        settings.Segments=std::clamp(settings.Segments,1u,MaximumTessellationSegments);
        settings.Curvature=std::isfinite(settings.Curvature)?std::clamp(settings.Curvature,0.0f,1.0f):1;
        return settings;
    }
    inline bool UsesTessellation(const TessellationSettings& settings) noexcept { return settings.Enabled && settings.Segments>1; }
}
