#pragma once

#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/FrameHistory.h"

namespace Evans::Renderer
{
    // Mitchell, GPU Gems 3, Chapter 13: radial image-space light scattering.
    // Our depth-tested analytic disk replaces authored source/occlusion images.
    // Midpoint weights integrate over [0,1], keeping brightness independent of
    // sample count. This is an image-space approximation, not medium transport.
    struct LightShaftConstants
    {
        Float4 Source; // top-left UV, canonical depth, radius in image heights
        Float4 Radiance; // source RGB, scattering intensity
        Float4 Integration; // decay, depth bias, image width, image height
        ShaderUInt4 Control; // sample count, mask/composite pass, source visible
    };
    static_assert(sizeof(LightShaftConstants) == 64 && offsetof(LightShaftConstants, Control) == 48,
        "Light shaft constant layout changed");

    inline bool BuildLightShaftConstants(const LightShaftDesc& desc, const Techniques::LightShaftSettings& requested,
        int width, int height, LightShaftConstants& output)
    {
        if (width <= 0 || height <= 0 || !FrameHistory::IsFinite(desc.ViewProjection)) return false;
        const auto settings = Techniques::SanitizeLightShaftSettings(requested);
        output = {};
        output.Source.W = settings.SourceRadius;
        output.Radiance = {0, 0, 0, settings.Intensity};
        output.Integration = {settings.Decay, settings.DepthBias, static_cast<float>(width), static_cast<float>(height)};
        output.Control = {settings.SampleCount, 0, 0, 0};
        if (settings.LightIndex >= std::min(desc.Lighting.LightCount, MaximumSceneLights)) return true;
        const auto& light = desc.Lighting.Lights[settings.LightIndex];
        const bool directional = light.Type == LightType::Directional;
        const auto location = directional ? light.Direction : light.Position;
        const std::array<float, 7> values {location.X, location.Y, location.Z, light.Color.X, light.Color.Y, light.Color.Z, light.Intensity};
        if (light.Type > LightType::RectArea || std::any_of(values.begin(), values.end(),
                [](float value) { return !std::isfinite(value); })) return false;
        const auto& matrix = desc.ViewProjection.Values;
        std::array<double, 4> clip {};
        for (std::size_t row = 0; row < clip.size(); ++row)
            clip[row] = double(matrix[row * 4]) * location.X + double(matrix[row * 4 + 1]) * location.Y +
                double(matrix[row * 4 + 2]) * location.Z + (directional ? 0.0 : matrix[row * 4 + 3]);
        constexpr double ProjectionEpsilon = 1.0e-6;
        // A direction under orthographic projection has no finite vanishing
        // point. Behind-eye/near-plane sources likewise contribute nothing.
        if (clip[3] <= ProjectionEpsilon || (!directional && clip[2] < 0.0)) return true;
        const double u = clip[0] / clip[3] * 0.5 + 0.5, v = 0.5 - clip[1] / clip[3] * 0.5;
        // Entirely off-image source disks have no observable radiance. Do not
        // clamp their center onto the screen border and manufacture a streak.
        const double horizontalRadius = settings.SourceRadius * double(height) / width;
        if (u + horizontalRadius <= 0 || u - horizontalRadius >= 1 || v + settings.SourceRadius <= 0 || v - settings.SourceRadius >= 1)
            return true;
        output.Source = {static_cast<float>(u), static_cast<float>(v), directional ? 1.0f :
            static_cast<float>(std::clamp(clip[2] / clip[3], 0.0, 1.0)), settings.SourceRadius};
        constexpr double MaximumHalf = 65504.0;
        const auto radiance = [&](float color) { return static_cast<float>(std::min(MaximumHalf,
            double(std::max(color, 0.0f)) * std::max(light.Intensity, 0.0f))); };
        output.Radiance = {radiance(light.Color.X), radiance(light.Color.Y), radiance(light.Color.Z), settings.Intensity};
        output.Control.Z = Techniques::UsesLightShafts(settings) ? 1 : 0;
        return true;
    }
}
