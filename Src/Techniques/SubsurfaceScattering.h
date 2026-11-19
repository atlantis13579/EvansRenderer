#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    // Screen-space direct-diffuse diffusion. Material values remain editable
    // without adding authored textures. The distance is a profile scale in
    // scene units, not a claim about a model's physical unit convention.
    struct SubsurfaceScatteringSettings
    {
        bool Enabled = false;
        float Strength = 0.5f;
        float Distance = 0.02f;
        std::array<float, 3> ChannelRadius {1.0f, 0.35f, 0.2f};
        float DepthScale = 1.0f;
        float NormalPower = 8.0f;
        float MaximumRadiusPixels = 64.0f;
    };
    inline SubsurfaceScatteringSettings SanitizeSubsurfaceScatteringSettings(SubsurfaceScatteringSettings value) noexcept
    {
        const SubsurfaceScatteringSettings defaults;
        constexpr float MaximumDistance = 1000000, MinimumChannelRadius = 0.001f;
        constexpr float MinimumDepthScale = 0.01f, MaximumDepthScale = 16;
        constexpr float MaximumNormalPower = 64, MaximumPixelRadius = 256;
        const auto finite = [](float x, float fallback) { return std::isfinite(x) ? x : fallback; };
        value.Strength = std::clamp(finite(value.Strength, defaults.Strength), 0.0f, 1.0f);
        value.Distance = std::clamp(finite(value.Distance, defaults.Distance), 0.0f, MaximumDistance);
        for (std::size_t channel = 0; channel < value.ChannelRadius.size(); ++channel)
            value.ChannelRadius[channel] = std::clamp(finite(value.ChannelRadius[channel], defaults.ChannelRadius[channel]), MinimumChannelRadius, 1.0f);
        value.DepthScale = std::clamp(finite(value.DepthScale, defaults.DepthScale), MinimumDepthScale, MaximumDepthScale);
        value.NormalPower = std::clamp(finite(value.NormalPower, defaults.NormalPower), 0.0f, MaximumNormalPower);
        value.MaximumRadiusPixels = std::clamp(finite(value.MaximumRadiusPixels, defaults.MaximumRadiusPixels), 0.0f, MaximumPixelRadius);
        return value;
    }
    inline bool operator==(const SubsurfaceScatteringSettings& a, const SubsurfaceScatteringSettings& b) noexcept
    {
        return a.Enabled == b.Enabled && a.Strength == b.Strength && a.Distance == b.Distance &&
            a.ChannelRadius == b.ChannelRadius && a.DepthScale == b.DepthScale && a.NormalPower == b.NormalPower &&
            a.MaximumRadiusPixels == b.MaximumRadiusPixels;
    }
    inline bool UsesSubsurfaceScattering(const SubsurfaceScatteringSettings& requested) noexcept
    {
        const auto value = SanitizeSubsurfaceScatteringSettings(requested);
        return value.Enabled && value.Strength > 0 && value.Distance > 0 && value.MaximumRadiusPixels > 0;
    }
    enum class SubsurfacePass : std::uint32_t { DirectDiffuse, Horizontal, Vertical, Composite };
}
