#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr float MinimumRefractionDistance = 0.0001f;
    inline constexpr std::uint32_t MinimumRefractionSteps = 8, MaximumRefractionSteps = 512;

    // Settings for the nearest-layer, local parallel-slab approximation.
    // Native integration requires sampled opaque HDR/depth and single-sample
    // surface capture. Refractive materials are independent of alpha/OIT surfaces.
    struct RefractionSettings
    {
        bool Enabled = false;
        float ExteriorIOR = 1, MaterialIOR = 1.5f;
        float Thickness = 0.1f;
        std::array<float,3> Absorption = {0,0,0}; // Inverse world units.
        bool MaterialTint = false;
        float TintDistance = 1; // Material RGB is transmittance over this distance.
        float MaximumDistance = 100, DepthThickness = 0.1f, SurfaceBias = 0.002f;
        float StridePixels = 1, EdgeFadePixels = 8;
        std::uint32_t MaximumSteps = 128;
    };

    inline RefractionSettings SanitizeRefractionSettings(RefractionSettings value) noexcept
    {
        constexpr float MaximumIOR = 4, MaximumWorldMeasure = 1000000;
        constexpr float MinimumDistance = MinimumRefractionDistance, MaximumStride = 8, MaximumEdgeFade = 128;
        constexpr std::uint32_t MinimumSteps = MinimumRefractionSteps, MaximumSteps = MaximumRefractionSteps;
        const RefractionSettings defaults;
        const auto bounded = [](float v,float fallback,float minimum,float maximum)
        { return std::isfinite(v) ? std::clamp(v,minimum,maximum) : fallback; };
        value.ExteriorIOR = bounded(value.ExteriorIOR,defaults.ExteriorIOR,1,MaximumIOR);
        value.MaterialIOR = bounded(value.MaterialIOR,defaults.MaterialIOR,1,MaximumIOR);
        value.Thickness = bounded(value.Thickness,defaults.Thickness,0,MaximumWorldMeasure);
        for (auto& coefficient:value.Absorption) coefficient = bounded(coefficient,0,0,MaximumWorldMeasure);
        value.TintDistance = bounded(value.TintDistance,defaults.TintDistance,MinimumDistance,MaximumWorldMeasure);
        value.MaximumDistance = bounded(value.MaximumDistance,defaults.MaximumDistance,MinimumDistance,MaximumWorldMeasure);
        value.DepthThickness = bounded(value.DepthThickness,defaults.DepthThickness,MinimumDistance,MaximumWorldMeasure);
        value.SurfaceBias = bounded(value.SurfaceBias,defaults.SurfaceBias,0,MaximumWorldMeasure);
        value.StridePixels = bounded(value.StridePixels,defaults.StridePixels,1,MaximumStride);
        value.EdgeFadePixels = bounded(value.EdgeFadePixels,defaults.EdgeFadePixels,0,MaximumEdgeFade);
        value.MaximumSteps = std::clamp(value.MaximumSteps,MinimumSteps,MaximumSteps);
        return value;
    }

    inline bool operator==(const RefractionSettings& a,const RefractionSettings& b) noexcept
    {
        return a.Enabled == b.Enabled && a.ExteriorIOR == b.ExteriorIOR && a.MaterialIOR == b.MaterialIOR && a.Thickness == b.Thickness &&
            a.Absorption == b.Absorption && a.MaterialTint == b.MaterialTint &&
            a.TintDistance == b.TintDistance && a.MaximumDistance == b.MaximumDistance && a.DepthThickness == b.DepthThickness &&
            a.SurfaceBias == b.SurfaceBias && a.StridePixels == b.StridePixels && a.EdgeFadePixels == b.EdgeFadePixels &&
            a.MaximumSteps == b.MaximumSteps;
    }
}
