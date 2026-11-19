#pragma once

#include "Renderer/Volumetrics/VolumetricFogContract.h"
#include "Renderer/Shadows/CascadedShadowContract.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include "Renderer/Volumetrics/ParticipatingMediaContract.h"
#include "Renderer/Volumetrics/VolumetricCloudContract.h"
#include "Renderer/VirtualShadowMaps/VirtualShadowSampling.h"

namespace Evans::Renderer
{
    // A segment is an affine transport operator (S, T): Lout = S + T * Lin.
    // Front-to-back composition is associative but not commutative. A bounded
    // inclusive scan therefore avoids re-evaluating lights for every endpoint.
    // Hillis and Steele, "Data Parallel Algorithms" (1986); phase conventions:
    // https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions
    struct VolumetricLightingPass
    {
        std::uint32_t PrefixStride = 0; // Zero injects local segment transport.
    };
    inline std::vector<VolumetricLightingPass> BuildVolumetricLightingPasses(std::uint32_t slices)
    {
        if (slices == 0 || slices > 64) return {};
        std::vector<VolumetricLightingPass> result {{0}};
        for (std::uint32_t stride = 1; stride < slices; stride *= 2) result.push_back({stride});
        return result;
    }

    struct VolumetricLightingConstants
    {
        Float4 Scattering; // RGB scattering albedo, Henyey-Greenstein asymmetry.
        Float4 Parameters; // Direct-light multiplier, shadow bias, complete-light buffer flag, reserved.
        ShaderUInt4 Control; // Light count, shadow mode (none/single/hierarchical), light index, scan stride.
        Matrix4 LightViewProjection;
        std::array<ShaderLight, MaximumSceneLights> Lights;
        CascadedShadowConstants Cascades;
        VolumeShadowFilterConstants Shadows;
        ParticipatingMediaConstants Participating;
        VolumetricCloudConstants Clouds;
    };
    static_assert(offsetof(VolumetricLightingConstants, Lights) == 112, "Volume light layout changed");
    static_assert(offsetof(VolumetricLightingConstants, Cascades) == 368 && offsetof(VolumetricLightingConstants, Shadows) == 704 &&
        offsetof(VolumetricLightingConstants, Participating) == 816 &&
        offsetof(VolumetricLightingConstants, Clouds) == 928 && sizeof(VolumetricLightingConstants) == 992,
        "Volume shadow layout changed");

    inline bool BuildVolumetricLightingConstants(const VolumetricFogDesc& desc,
        const Techniques::VolumetricLightingSettings& requested, const CascadedShadowState& cascades,
        const Techniques::ShadowFilterSettings& filter, VolumetricLightingConstants& output, bool completeLights = false,
        const Techniques::VolumetricFogSettings* medium = nullptr,
        const VirtualShadowSnapshot* virtualAtlas = nullptr, const VirtualShadowSamplingFrame* virtualFrame = nullptr,
        Float2 virtualGrid = {})
    {
        const auto settings = Techniques::SanitizeVolumetricLightingSettings(requested);
        output = {};
        output.Scattering = {settings.ScatteringAlbedo[0], settings.ScatteringAlbedo[1], settings.ScatteringAlbedo[2], settings.Anisotropy};
        output.Parameters = {settings.Intensity, settings.ShadowDepthBias, 0, 0};
        output.Control.X = std::min(desc.Lighting.LightCount, MaximumSceneLights);
        for (std::uint32_t index = 0; index < output.Control.X; ++index)
        {
            const auto& light = desc.Lighting.Lights[index];
            const std::array<float, 12> values {light.Position.X, light.Position.Y, light.Position.Z,
                light.Direction.X, light.Direction.Y, light.Direction.Z, light.Color.X, light.Color.Y, light.Color.Z,
                light.Intensity, light.AreaSize.X, light.AreaSize.Y};
            if (std::any_of(values.begin(), values.end(), [](float value) { return !std::isfinite(value); }) ||
                light.Type > LightType::RectArea) return false;
            auto& packed = output.Lights[index];
            packed.PositionType = {light.Position.X, light.Position.Y, light.Position.Z, static_cast<float>(light.Type)};
            packed.DirectionIntensity = {light.Direction.X, light.Direction.Y, light.Direction.Z, std::max(light.Intensity, 0.0f)};
            packed.ColorCastShadows = {std::max(light.Color.X, 0.0f), std::max(light.Color.Y, 0.0f),
                std::max(light.Color.Z, 0.0f), light.CastShadows ? 1.0f : 0.0f};
            packed.AreaSizePadding = {std::max(light.AreaSize.X, 0.0f), std::max(light.AreaSize.Y, 0.0f),
                Techniques::CanonicalLightRange(light.Range), static_cast<float>(light.AreaSampleAxisCount)};
        }
        const auto shadow = desc.Lighting.ShadowLightIndex;
        if (desc.SamplesShadowMap && shadow >= 0 && static_cast<std::uint32_t>(shadow) < output.Control.X &&
            desc.Lighting.Lights[shadow].CastShadows)
        {
            if (!FrameHistory::IsFinite(desc.LightViewProjection) || cascades.Frame.CascadeCount > MaximumShadowCascades) return false;
            output.Control.Y = cascades.Frame.CascadeCount ? 2 : 1;
            output.Control.Z = static_cast<std::uint32_t>(shadow);
            output.LightViewProjection = desc.LightViewProjection;
            output.Cascades = BuildCascadedShadowConstants(cascades, filter);
            for (std::uint32_t index = 0; index < cascades.Frame.CascadeCount; ++index)
                if (!FrameHistory::IsFinite(output.Cascades.LightViewProjection[index])) return false;
        }
        if (completeLights)
        {
            if (!desc.Lighting.SceneLights) return false;
            // Scattering points occupy the entire view ray. Opaque tile depth
            // intervals cannot conservatively reject their incident lights.
            const auto& scene = *desc.Lighting.SceneLights;
            output.Parameters.Z = 1.0f;
            output.Control.X = static_cast<std::uint32_t>(scene.GetShaderLights().size());
            output.Control.Z = static_cast<std::uint32_t>(scene.GetShadowLightIndex());
        }
        if (medium && Techniques::UsesBoundedMedium(*medium))
        {
            const bool directLighting = Techniques::UsesVolumetricLighting(settings);
            if (!directLighting) { output.Parameters.X = 0; output.Control.Y = 0; }
            if (!BuildParticipatingMediaConstants(desc, *medium, cascades, filter, directLighting,
                output.Participating, output.Shadows)) return false;
            if (Techniques::UsesVolumetricClouds(*medium)) output.Clouds = BuildVolumetricCloudConstants(medium->Clouds);
        }
        if (virtualAtlas && output.Control.Y)
        {
            if (!virtualFrame || cascades.Frame.CascadeCount || !FrameHistory::IsFinite(desc.ViewProjection) ||
                (filter.Mode != Techniques::ShadowFilterMode::Hard && filter.Mode != Techniques::ShadowFilterMode::PCF) ||
                !std::isfinite(virtualGrid.X) || !std::isfinite(virtualGrid.Y) || virtualGrid.X <= 0 || virtualGrid.Y <= 0)
                return false;
            // The camera's forward rows occupy the unused hard/PCF metric suffix.
            // Reconstruct neighboring rays at the receiver's camera depth, then
            // project their footprint into the virtual light hierarchy. This
            // remains defined inside bounded-medium and cloud integration loops.
            output.Cascades = BuildVirtualShadowSamplingConstants(*virtualAtlas, *virtualFrame, filter);
            output.Cascades.EyeDepthPlane.Y = 1.0f / virtualGrid.X;
            output.Cascades.EyeDepthPlane.Z = 1.0f / virtualGrid.Y;
            for (unsigned row = 0; row < 4; ++row)
            {
                const auto* values = desc.ViewProjection.Values.data() + row * 4;
                output.Shadows.Encoding[row] = {values[0], values[1], values[2], values[3]};
            }
            output.Control.Y = 2; // The exclusive hierarchical receiver bindings.
        }
        return true;
    }
}
