#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    enum class DisplacementProjection : std::uint32_t { Automatic,XY,XZ,YZ };
    inline constexpr float MaximumDisplacementFraction=0.25f;
    inline constexpr float MinimumDisplacementTiling=0.25f,MaximumDisplacementTiling=32;
    inline constexpr std::uint32_t MaximumDisplacementSegments=64;
    struct DisplacementMappingSettings
    {
        bool Enabled=false;
        DisplacementProjection Projection=DisplacementProjection::Automatic;
        float HeightFraction=0.05f; // full height range relative to the longest local object extent
        float Midlevel=0.5f; // scalar height that leaves the source surface unchanged
        float Tiling=1.3f;
        std::uint32_t Segments=2; // uniform sampling per original triangle edge, independent of native tessellation stages
    };
    inline bool operator==(const DisplacementMappingSettings& a,const DisplacementMappingSettings& b) noexcept
    { return a.Enabled==b.Enabled && a.Projection==b.Projection && a.HeightFraction==b.HeightFraction && a.Midlevel==b.Midlevel && a.Tiling==b.Tiling && a.Segments==b.Segments; }
    inline bool operator!=(const DisplacementMappingSettings& a,const DisplacementMappingSettings& b) noexcept { return !(a==b); }
    inline DisplacementMappingSettings SanitizeDisplacementMappingSettings(DisplacementMappingSettings settings) noexcept
    {
        if (settings.Projection>DisplacementProjection::YZ) settings.Projection=DisplacementProjection::Automatic;
        const DisplacementMappingSettings defaults;
        if (!std::isfinite(settings.HeightFraction)) settings.HeightFraction=defaults.HeightFraction;
        if (!std::isfinite(settings.Midlevel)) settings.Midlevel=defaults.Midlevel;
        if (!std::isfinite(settings.Tiling)) settings.Tiling=defaults.Tiling;
        settings.HeightFraction=std::clamp(settings.HeightFraction,0.0f,MaximumDisplacementFraction);
        settings.Midlevel=std::clamp(settings.Midlevel,0.0f,1.0f);
        settings.Tiling=std::clamp(settings.Tiling,MinimumDisplacementTiling,MaximumDisplacementTiling);
        settings.Segments=std::clamp(settings.Segments,1u,MaximumDisplacementSegments);
        return settings;
    }
    inline bool UsesDisplacementMapping(const DisplacementMappingSettings& settings) noexcept
    { return settings.Enabled && settings.HeightFraction>0; }
}
