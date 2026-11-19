#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    enum class HeightCoordinates : std::uint32_t { Automatic=0,MeshUV=1,ObjectProjection=2 };
    inline constexpr std::uint32_t MinimumParallaxLayers=8,MaximumParallaxLayers=256,MaximumParallaxRefinements=8;
    inline constexpr float MaximumParallaxDepth=0.25f,MinimumHeightTiling=0.25f,MaximumHeightTiling=32.0f;
    inline constexpr float MinimumParallaxCosine=0.025f,MaximumParallaxCosine=0.25f;
    struct ParallaxOcclusionSettings
    {
        bool Enabled=false;
        HeightCoordinates Coordinates=HeightCoordinates::Automatic;
        float DepthFraction=0.05f; // fraction of the smaller world-space height-coordinate basis
        float Tiling=4.0f,NormalStrength=1.0f,GrazingCosine=0.05f;
        std::uint32_t MinimumLayers=16,MaximumLayers=64,RefinementSteps=4;
    };
    inline bool operator==(const ParallaxOcclusionSettings& a,const ParallaxOcclusionSettings& b) noexcept
    {
        return a.Enabled==b.Enabled && a.Coordinates==b.Coordinates && a.DepthFraction==b.DepthFraction && a.Tiling==b.Tiling &&
            a.NormalStrength==b.NormalStrength && a.GrazingCosine==b.GrazingCosine && a.MinimumLayers==b.MinimumLayers &&
            a.MaximumLayers==b.MaximumLayers && a.RefinementSteps==b.RefinementSteps;
    }
    inline ParallaxOcclusionSettings SanitizeParallaxOcclusionSettings(ParallaxOcclusionSettings settings) noexcept
    {
        const ParallaxOcclusionSettings defaults;
        const auto finite=[](float value,float fallback){return std::isfinite(value)?value:fallback;};
        if (settings.Coordinates>HeightCoordinates::ObjectProjection) settings.Coordinates=defaults.Coordinates;
        settings.DepthFraction=std::clamp(finite(settings.DepthFraction,defaults.DepthFraction),0.0f,MaximumParallaxDepth);
        settings.Tiling=std::clamp(finite(settings.Tiling,defaults.Tiling),MinimumHeightTiling,MaximumHeightTiling);
        settings.NormalStrength=std::clamp(finite(settings.NormalStrength,defaults.NormalStrength),0.0f,2.0f);
        settings.GrazingCosine=std::clamp(finite(settings.GrazingCosine,defaults.GrazingCosine),MinimumParallaxCosine,MaximumParallaxCosine);
        settings.MinimumLayers=std::clamp(settings.MinimumLayers,MinimumParallaxLayers,MaximumParallaxLayers);
        settings.MaximumLayers=std::clamp(settings.MaximumLayers,settings.MinimumLayers,MaximumParallaxLayers);
        settings.RefinementSteps=std::min(settings.RefinementSteps,MaximumParallaxRefinements);
        return settings;
    }
    inline bool UsesParallaxOcclusion(const ParallaxOcclusionSettings& settings) noexcept
    { return settings.Enabled && settings.DepthFraction>0; }
}
