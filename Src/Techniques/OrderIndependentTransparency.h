#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr std::uint32_t MinimumPeelingLayers = 1, MaximumPeelingLayers = 32;

    // A layer is one distinct raster depth, not one mesh or fragment. Coincident
    // fragments share a layer and retain a canonical source-over draw order.
    struct OrderIndependentTransparencySettings
    {
        std::uint32_t LayerCount = 8;
        bool ShowOverflow = false;
    };

    inline OrderIndependentTransparencySettings SanitizeOrderIndependentTransparencySettings(
        OrderIndependentTransparencySettings value) noexcept
    {
        value.LayerCount = std::clamp(value.LayerCount,MinimumPeelingLayers,MaximumPeelingLayers);
        return value;
    }

    inline bool operator==(const OrderIndependentTransparencySettings& a,
        const OrderIndependentTransparencySettings& b) noexcept
    {
        return a.LayerCount == b.LayerCount && a.ShowOverflow == b.ShowOverflow;
    }
}
