#pragma once
#include "Renderer/Shadows/ShadowProjection.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/Shadows/CascadedShadowContract.h"

namespace Evans::Renderer
{
    inline constexpr std::uint32_t VSMMapCount = MaximumShadowCascades + 1;
    struct VSMConstants
    {
        std::array<Float4, VSMMapCount> DepthEncoding {};
        Float4 Parameters; // variance floor, bleeding reduction, derivative cap, sigma
        ShaderUInt4 Control; // valid map bits, blur radius, reserved, filter axis
    };
    static_assert(sizeof(VSMConstants) == 112, "Native VSM layout changed");
    inline VSMConstants BuildVSMParameters(const Techniques::VSMSettings& requested)
    {
        const auto settings = Techniques::SanitizeVSMSettings(requested);
        VSMConstants result;
        result.Parameters = {settings.MinimumVariance, settings.LightBleedingReduction,
            settings.MaximumDerivativeVariance, settings.BlurSigma};
        result.Control = {0, settings.BlurRadius, 0, 0};
        return result;
    }
    inline bool BuildVSMEncoding(const Matrix4& matrix, Float4& encoding, std::string& error)
    {
        ShadowProjection projection;
        return BuildShadowProjection(matrix, projection, error) && BuildShadowDepthEncoding(projection, encoding, error);
    }
    inline bool BuildVSMConstants(const Matrix4& singleProjection, const LightingState& lighting,
        const CascadedShadowState& cascades, const Techniques::VSMSettings& requested,
        bool sampleShadows, VSMConstants& result, std::string& error)
    {
        result = BuildVSMParameters(requested); error.clear();
        const auto light = lighting.ShadowLightIndex;
        if (!sampleShadows || light < 0 || std::uint32_t(light) >= std::min(lighting.LightCount, MaximumSceneLights) ||
            !lighting.Lights[light].CastShadows) return true;
        if (cascades.Frame.CascadeCount > MaximumShadowCascades)
        { error = "VSM cascade count exceeds the receiver layout"; return false; }
        const auto pack = [&](std::uint32_t index, const Matrix4& matrix)
        {
            if (!BuildVSMEncoding(matrix, result.DepthEncoding[index], error)) return false;
            result.Control.X |= 1u << index; return true;
        };
        if (!cascades.Frame.CascadeCount) return pack(0, singleProjection);
        for (std::uint32_t cascade = 0; cascade < cascades.Frame.CascadeCount; ++cascade)
            if (!pack(cascade + 1, cascades.Frame.Cascades[cascade].LightViewProjection)) return false;
        return true;
    }
}
