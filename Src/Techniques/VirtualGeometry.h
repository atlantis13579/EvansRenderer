#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr std::uint32_t MaximumVirtualGeometryPages=16*1024,MaximumVirtualGeometryUploads=1024;
    inline constexpr float MinimumVirtualGeometryError=0.25f,MaximumVirtualGeometryError=256,MaximumVirtualGeometryHysteresis=0.5f;
    struct VirtualGeometrySettings
    {
        bool Enabled=false;
        float MaximumErrorPixels=1,HysteresisFraction=0.15f;
        std::uint32_t PhysicalPages=0,UploadsPerFrame=64; // zero capacity chooses the dataset size; zero uploads pauses detail
    };
    inline bool operator==(const VirtualGeometrySettings& a,const VirtualGeometrySettings& b) noexcept
    { return a.Enabled==b.Enabled && a.MaximumErrorPixels==b.MaximumErrorPixels && a.HysteresisFraction==b.HysteresisFraction && a.PhysicalPages==b.PhysicalPages && a.UploadsPerFrame==b.UploadsPerFrame; }
    inline VirtualGeometrySettings SanitizeVirtualGeometrySettings(VirtualGeometrySettings value) noexcept
    {
        value.MaximumErrorPixels=std::isfinite(value.MaximumErrorPixels)?std::clamp(value.MaximumErrorPixels,MinimumVirtualGeometryError,MaximumVirtualGeometryError):1;
        value.HysteresisFraction=std::isfinite(value.HysteresisFraction)?std::clamp(value.HysteresisFraction,0.0f,MaximumVirtualGeometryHysteresis):0.15f;
        value.PhysicalPages=std::min(value.PhysicalPages,MaximumVirtualGeometryPages); value.UploadsPerFrame=std::min(value.UploadsPerFrame,MaximumVirtualGeometryUploads);
        return value;
    }
}
