#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    // A paged depth representation, distinct from variance shadow filtering.
    // These limits bound GPU address arithmetic and the physical atlas budget.
    inline constexpr std::uint32_t VirtualShadowMaximumResolution=16384;
    inline constexpr std::uint32_t VirtualShadowMinimumPageSide=64,VirtualShadowMaximumPageSide=256;
    inline constexpr std::uint32_t VirtualShadowMaximumPhysicalPages=1024;
    struct VirtualShadowSettings
    {
        std::uint32_t Resolution=8192,PageSide=128,PhysicalPages=144;
        std::uint32_t MaximumUnusedFrames=60;
        float MipBias=-1;
    };
    inline bool operator==(const VirtualShadowSettings& a,const VirtualShadowSettings& b) noexcept
    { return a.Resolution==b.Resolution && a.PageSide==b.PageSide && a.PhysicalPages==b.PhysicalPages && a.MaximumUnusedFrames==b.MaximumUnusedFrames && a.MipBias==b.MipBias; }
    inline VirtualShadowSettings SanitizeVirtualShadowSettings(VirtualShadowSettings value) noexcept
    {
        const VirtualShadowSettings defaults;
        const auto powerOfTwo=[](std::uint32_t x,std::uint32_t low,std::uint32_t high)
        {
            x=std::clamp(x,low,high); std::uint32_t result=low;
            while (result<high && result*2<=x) result*=2;
            return result;
        };
        value.PageSide=powerOfTwo(value.PageSide,VirtualShadowMinimumPageSide,VirtualShadowMaximumPageSide);
        value.Resolution=powerOfTwo(value.Resolution,value.PageSide,VirtualShadowMaximumResolution);
        value.PhysicalPages=std::clamp(value.PhysicalPages,1u,VirtualShadowMaximumPhysicalPages);
        value.MaximumUnusedFrames=std::min(value.MaximumUnusedFrames,3600u);
        value.MipBias=std::isfinite(value.MipBias)?std::clamp(value.MipBias,-4.0f,4.0f):defaults.MipBias;
        return value;
    }
}
