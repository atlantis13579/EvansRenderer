#pragma once
#include "Renderer/Shadows/ShadowProjection.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/Shadows/CascadedShadowContract.h"

namespace Evans::Renderer
{
    inline constexpr float PCSSRadiansPerDegree = 3.14159265358979323846f / 180.0f;
    inline constexpr std::uint32_t PCSSMapCount = MaximumShadowCascades + 1;
    inline constexpr float PCSSMaximumMapFraction = 0.25f;
    inline float PCSSFootprintLimit(float requested, int resolution)
    { return std::min(requested, float(std::max(resolution, 1)) * PCSSMaximumMapFraction); }

    // Slot zero describes the single map; slots one through four describe CSM.
    // Near == 0 identifies an orthographic metric. Width/height describe either
    // the orthographic rectangle or the perspective near-plane rectangle.
    struct PCSSConstants
    {
        std::array<Float4, PCSSMapCount> ProjectionMetrics {};
        Float4 EmitterAndLimits; // radius, tan(angular half-angle), search/filter texel caps
        ShaderUInt4 SamplesAndMaps; // blocker/filter samples, valid map bits, reserved
    };
    static_assert(sizeof(PCSSConstants) == 112, "Native PCSS receiver layout changed");

    inline bool BuildPCSSConstants(const Matrix4& singleProjection, const LightingState& lighting,
        const CascadedShadowState& cascades, const Techniques::PCSSSettings& requested,
        bool sampleShadows, PCSSConstants& result, std::string& error)
    {
        result = {}; error.clear();
        const auto settings = Techniques::SanitizePCSSSettings(requested);
        result.EmitterAndLimits = {settings.EmitterRadius, std::tan(settings.AngularRadiusDegrees * PCSSRadiansPerDegree),
            float(settings.MaximumSearchRadius), float(settings.MaximumFilterRadius)};
        result.SamplesAndMaps = {settings.BlockerSamples, settings.FilterSamples, 0, 0};
        const auto lightIndex = lighting.ShadowLightIndex;
        if (!sampleShadows || lightIndex < 0 || std::uint32_t(lightIndex) >= std::min(lighting.LightCount, MaximumSceneLights) ||
            !lighting.Lights[lightIndex].CastShadows) return true;
        if (cascades.Frame.CascadeCount > MaximumShadowCascades)
        { error = "PCSS cascade count exceeds the receiver layout"; return false; }
        const bool directional = lighting.Lights[lightIndex].Type == LightType::Directional;
        const auto pack = [&](std::uint32_t index, const Matrix4& matrix)
        {
            ShadowProjection projection;
            if (!BuildShadowProjection(matrix, projection, error)) return false;
            if (projection.Perspective == directional)
            { error = "PCSS requires orthographic directional maps or perspective finite-light maps"; return false; }
            result.ProjectionMetrics[index] = {float(projection.NearDistance), float(projection.FarDistance),
                float(projection.Width), float(projection.Height)};
            result.SamplesAndMaps.Z |= 1u << index;
            return true;
        };
        if (!cascades.Frame.CascadeCount) return pack(0, singleProjection);
        for (std::uint32_t cascade = 0; cascade < cascades.Frame.CascadeCount; ++cascade)
            if (!pack(cascade + 1, cascades.Frame.Cascades[cascade].LightViewProjection)) return false;
        return true;
    }
}
