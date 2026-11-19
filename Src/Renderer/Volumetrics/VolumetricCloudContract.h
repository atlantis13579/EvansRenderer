#pragma once
#include "Renderer/Volumetrics/ParticipatingMediaContract.h"

namespace Evans::Renderer
{
    // Cloud density specializes the bounded-medium transport contract. The
    // shader evaluates coverage weather, value-noise octaves, cellular erosion
    // and a height envelope on both camera and light paths. No texture asset is
    // required. This is a finite single-scattering layer, not a weather simulator.
    // Schneider and Vos, "The Real-Time Volumetric Cloudscapes of Horizon Zero Dawn" (2015),
    // motivates separating weather, height, shape and erosion controls.
    // https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn
    struct VolumetricCloudConstants
    {
        Float4 CoverageWeather; // coverage, weather variation, weather frequency, base frequency
        Float4 DetailProfile; // detail frequency, erosion strength, bottom fade end, top fade start
        Float4 OffsetSeed; // world-space field offset XYZ, integer seed
        Float4 Shape; // horizontal edge fade, density contrast, reserved
    };
    static_assert(sizeof(VolumetricCloudConstants) == 64, "Cloud shape layout changed");
    inline VolumetricCloudConstants BuildVolumetricCloudConstants(const Techniques::VolumetricCloudSettings& requested)
    {
        const auto settings = Techniques::SanitizeVolumetricCloudSettings(requested);
        return {{settings.Coverage, settings.WeatherVariation, settings.WeatherFrequency, settings.BaseFrequency},
            {settings.DetailFrequency, settings.Erosion, settings.BaseFade, settings.TopFade},
            {settings.Offset[0], settings.Offset[1], settings.Offset[2], static_cast<float>(settings.Seed)},
            {settings.EdgeFade, settings.DensityContrast, 0, 0}};
    }
}
