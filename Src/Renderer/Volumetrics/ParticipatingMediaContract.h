#pragma once
#include "Renderer/Volumetrics/VolumetricFogContract.h"
#include "Renderer/Volumetrics/DensityVolume.h"
#include "Renderer/Shadows/PCSSContract.h"
#include "Renderer/Shadows/VSMContract.h"

namespace Evans::Renderer
{
    // Camera and light rays share sigma_t = density * (sigma_a + sigma_s).
    // RGB transport is an affine operator (S, T): Lout = S + T * Lin, with
    // componentwise products. Emission is radiance per unit path length.
    // Pharr, Jakob and Humphreys, PBRT v4, Volume Scattering:
    // https://pbr-book.org/4ed/Volume_Scattering/Transmittance
    // The native estimators model single scattering, including extinction on
    // the incident light path. Density comes from an authored grid or a bounded
    // procedural field; source samples remain float-accurate across native APIs.
    struct ParticipatingMediaConstants
    {
        Float4 AbsorptionDensity;
        Float4 ScatteringModulation;
        Float4 EmissionFrequency;
        Float4 BoundsMinimum;
        Float4 BoundsMaximum;
        ShaderUInt4 Options; // light-ray samples, shadow filter kind, reserved
        Float4 DensityGrid; // voxel dimensions XYZ, tile columns W; zero disables the atlas
    };
    static_assert(sizeof(ParticipatingMediaConstants) == 112, "Medium coefficient layout changed");

    // PCSS and VSM use equal-size descriptors. The mode selects the meaning of
    // these fields; preserving this suffix also preserves legacy Vulkan UBOs.
    struct VolumeShadowFilterConstants
    {
        std::array<Float4, MaximumShadowCascades + 1> Encoding;
        Float4 Parameters;
        ShaderUInt4 Options;
    };
    static_assert(sizeof(VolumeShadowFilterConstants) == sizeof(PCSSConstants) &&
        sizeof(VolumeShadowFilterConstants) == sizeof(VSMConstants), "Volume filter suffix changed");

    inline bool BuildParticipatingMediaConstants(const VolumetricFogDesc& desc,
        const Techniques::VolumetricFogSettings& requested, const CascadedShadowState& cascades,
        const Techniques::ShadowFilterSettings& filter, bool directLighting,
        ParticipatingMediaConstants& output, VolumeShadowFilterConstants& shadows)
    {
        const auto settings = Techniques::SanitizeParticipatingMediaSettings(
            requested.MediumMode == Techniques::VolumeMediumMode::Clouds ? requested.Clouds.Medium : requested.Participating);
        output.AbsorptionDensity = {settings.Absorption[0], settings.Absorption[1], settings.Absorption[2], settings.Density};
        output.ScatteringModulation = {settings.Scattering[0], settings.Scattering[1], settings.Scattering[2], settings.ModulationStrength};
        output.EmissionFrequency = {settings.Emission[0], settings.Emission[1], settings.Emission[2], settings.ModulationFrequency};
        output.BoundsMinimum = {settings.BoundsMinimum[0], settings.BoundsMinimum[1], settings.BoundsMinimum[2], 0};
        output.BoundsMaximum = {settings.BoundsMaximum[0], settings.BoundsMaximum[1], settings.BoundsMaximum[2], 0};
        output.Options = {settings.LightSamples, static_cast<std::uint32_t>(filter.Mode), 0, 0};
        output.DensityGrid=desc.Density ? desc.Density->Grid : Float4{};
        shadows = {};
        const bool sampleShadows = directLighting && desc.SamplesShadowMap;
        std::string error;
        if (filter.Mode == Techniques::ShadowFilterMode::PCSS)
        {
            PCSSConstants pcss;
            if (!BuildPCSSConstants(desc.LightViewProjection, desc.Lighting, cascades, filter.PCSS, sampleShadows, pcss, error)) return false;
            shadows.Encoding = pcss.ProjectionMetrics; shadows.Parameters = pcss.EmitterAndLimits; shadows.Options = pcss.SamplesAndMaps;
        }
        if (filter.Mode == Techniques::ShadowFilterMode::VSM)
        {
            VSMConstants vsm;
            if (!BuildVSMConstants(desc.LightViewProjection, desc.Lighting, cascades, filter.VSM, sampleShadows, vsm, error)) return false;
            shadows.Encoding = vsm.DepthEncoding; shadows.Parameters = vsm.Parameters; shadows.Options = vsm.Control;
        }
        return true;
    }
}
