#pragma once

#include "Renderer/Shadows/CascadedShadowMap.h"

namespace Evans::Renderer
{
    // Native consumers reconstruct each light position from the receiver's world
    // position. The single-map draw layout stays unchanged; this separate block
    // also serves deferred and transparent forward lighting.
    struct CascadedShadowConstants
    {
        std::array<Matrix4, MaximumShadowCascades> LightViewProjection;
        std::array<float, MaximumShadowCascades> FarDistances {};
        std::array<float, MaximumShadowCascades> BlendStarts {};
        Float4 EyeDepthPlane;
        Float4 Parameters; // near, far, count, map resolution
        Float4 Filtering; // PCF radius (zero means hard), kernel sample count, reserved
    };
    static_assert(offsetof(CascadedShadowConstants, Filtering) == 320 && sizeof(CascadedShadowConstants) == 336,
        "Native shadow receiver block layout changed");

    inline CascadedShadowConstants BuildCascadedShadowConstants(const CascadedShadowState& state,
        const Techniques::ShadowFilterSettings& requested = {})
    {
        CascadedShadowConstants result {};
        const auto filter = Techniques::SanitizeShadowFilterSettings(requested);
        const auto radius = filter.Mode == Techniques::ShadowFilterMode::Hard ? 0u : filter.Radius;
        const auto width = radius * 2 + 1;
        // Reeves et al., "Rendering Antialiased Shadows with Depth Maps" (1987).
        // Compare depths first, then interpolate/average visibility. Hard mode
        // fetches one nearest texel; it is not a radius-zero bilinear PCF kernel.
        // Filtering remains valid when CascadeCount is zero (single-map receivers).
        result.Filtering = {float(radius), float(width * width), 0, 0};
        result.EyeDepthPlane = state.Frame.EyeDepthPlane;
        result.Parameters = {state.Frame.NearDistance, state.Frame.FarDistance,
            float(state.Frame.CascadeCount), float(state.MapResolution)};
        for (std::uint32_t index = 0; index < state.Frame.CascadeCount; ++index)
        {
            result.LightViewProjection[index] = state.Frame.Cascades[index].LightViewProjection;
            result.FarDistances[index] = state.Frame.Cascades[index].FarDistance;
            result.BlendStarts[index] = state.Frame.Cascades[index].BlendStartDistance;
        }
        return result;
    }
}
