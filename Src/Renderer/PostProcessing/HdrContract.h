#pragma once

#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // HDR is a scene-linear data contract, not a display filter. RGBA16F keeps
    // radiance above one through flare, bloom and analytic grading. Tone mapping is
    // the sole conversion into the existing display-linear/sRGB output path.
    // Manual EV is applied at resolve. Optional temporal exposure has already
    // adjusted scene radiance before this chain and its highlight thresholds.
    struct HdrConstants
    {
        Float4 InverseSourceExtentBloom;
        Float4 GradeExposure;
        ShaderUInt4 PassToneMapping;
        Float4 FlareThresholdIntensitySpacingRadius;
        Float4 FlareAspect;
    };
    static_assert(sizeof(HdrConstants) == 80,
        "Native HDR shaders use five aligned four-component uniform vectors");
    static_assert(offsetof(HdrConstants, PassToneMapping) == 32,
        "HDR operation identifiers must remain an aligned integer vector");
    static_assert(offsetof(HdrConstants, FlareThresholdIntensitySpacingRadius) == 48,
        "Flare constants extend the original HDR prefix");

    inline HdrConstants BuildHdrConstants(const Techniques::HdrSettings& requested,
        int sourceWidth, int sourceHeight, Techniques::HdrPassKind pass,
        int viewportWidth = 0, int viewportHeight = 0)
    {
        const auto settings = Techniques::SanitizeHdrSettings(requested);
        const float aspect = static_cast<float>(std::max(viewportWidth > 0 ? viewportWidth : sourceWidth, 1)) /
            static_cast<float>(std::max(viewportHeight > 0 ? viewportHeight : sourceHeight, 1));
        return {
            {1.0f / static_cast<float>(std::max(sourceWidth, 1)),
             1.0f / static_cast<float>(std::max(sourceHeight, 1)),
             settings.BloomThreshold, settings.BloomIntensity},
            {settings.Contrast, settings.Saturation, std::exp2(settings.ExposureEV), 0.0f},
            {static_cast<std::uint32_t>(pass),
             static_cast<std::uint32_t>(Techniques::EffectiveToneMapping(settings)), 0u, 0u},
            {settings.FlareThreshold, settings.FlareIntensity, settings.FlareGhostSpacing, settings.FlareHaloRadius},
            {aspect, 1.0f / aspect, 0.0f, 0.0f}};
    }

    // Half indices name two shared temporary images, never a shader source.
    // A negative index reads/writes the current full-resolution scene pair.
    struct HdrPassResources
    {
        int SourceHalf = -1;
        int DestinationHalf = -1;
        int CompositeHalf = -1;
        bool UpdatesScene = false;
    };
    inline HdrPassResources DescribeHdrPass(Techniques::HdrPassKind pass)
    {
        using Pass = Techniques::HdrPassKind;
        switch (pass)
        {
        case Pass::BloomExtract: case Pass::FlareExtract: return {-1, 0, -1, false};
        case Pass::BloomBlurHorizontal: case Pass::FlareFeatures: case Pass::FlareBlurVertical: return {0, 1, -1, false};
        case Pass::BloomBlurVertical: case Pass::FlareBlurHorizontal: return {1, 0, -1, false};
        case Pass::BloomComposite: return {-1, -1, 0, true};
        case Pass::FlareComposite: return {-1, -1, 1, true};
        case Pass::ColorGrading: return {-1, -1, -1, true};
        case Pass::ToneMapping: return {};
        }
        return {};
    }
}
