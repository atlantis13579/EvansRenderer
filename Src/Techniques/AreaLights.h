#pragma once

#include <algorithm>
#include <cstdint>

namespace Evans::Techniques
{
    // A one-sided Lambertian emitter has total power Phi = pi * area * Le.
    // These are linear RGB radiometric scales, not lumens or exposure values.
    enum class AreaLightEnergyUnit : std::uint8_t { Radiance, Power };
    inline constexpr std::uint32_t MinimumAreaLightSamplesPerAxis = 2;
    inline constexpr std::uint32_t MaximumAreaLightSamplesPerAxis = 16;

    struct AreaLightSettings
    {
        // Integration dependency: PBR surface shading and scene-linear HDR.
        // This modifies emitter integration, not the selected rendering pipeline;
        // shadow filtering, AO, reflections and transparency remain independent.
        // Requested preferences survive temporarily incompatible shading choices.
        bool Enabled = false;
        AreaLightEnergyUnit EnergyUnit = AreaLightEnergyUnit::Radiance;
        std::uint32_t SamplesPerAxis = 4;
    };

    [[nodiscard]] inline AreaLightSettings SanitizeAreaLightSettings(AreaLightSettings settings) noexcept
    {
        if (settings.EnergyUnit != AreaLightEnergyUnit::Power) settings.EnergyUnit = AreaLightEnergyUnit::Radiance;
        settings.SamplesPerAxis = std::clamp(settings.SamplesPerAxis,
            MinimumAreaLightSamplesPerAxis, MaximumAreaLightSamplesPerAxis);
        return settings;
    }

    [[nodiscard]] inline bool operator==(const AreaLightSettings& a, const AreaLightSettings& b) noexcept
    { return a.Enabled == b.Enabled && a.EnergyUnit == b.EnergyUnit && a.SamplesPerAxis == b.SamplesPerAxis; }
}
