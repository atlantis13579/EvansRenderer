#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    inline constexpr auto MaximumShadowCascades = Techniques::MaximumShadowCascades;
    using CascadedShadowSettings = Techniques::CascadedShadowSettings;
    using Techniques::SanitizeCascadedShadowSettings;

    // Separate view/projection inputs keep the split distances and sphere size
    // independent of camera motion. View must be rigid, right-handed; projection
    // uses zero-to-one depth. Orthographic and asymmetric cameras are supported.
    struct CascadedShadowCamera
    {
        Matrix4 View;
        Matrix4 Projection;
    };

    struct ShadowCasterBounds
    {
        Float3 Minimum;
        Float3 Maximum;
    };

    struct ShadowCascade
    {
        Matrix4 LightViewProjection;
        std::array<Float3, 8> ReceiverCorners {};
        float FitNearDistance = 0.0f;
        float FarDistance = 0.0f;
        float BlendStartDistance = 0.0f;
        float WorldUnitsPerTexel = 0.0f;
        float LightDepthMinimum = 0.0f;
        float LightDepthMaximum = 0.0f;
    };

    struct CascadedShadowFrame
    {
        std::array<ShadowCascade, MaximumShadowCascades> Cascades {};
        Float4 EyeDepthPlane; // dot(plane, worldPosition) gives positive eye distance
        std::uint32_t CascadeCount = 0;
        float NearDistance = 0.0f;
        float FarDistance = 0.0f;
        bool Perspective = false;
    };

    struct ShadowCascadeSelection
    {
        // MaximumShadowCascades denotes unshadowed visibility. The final cascade
        // blends toward that sentinel, avoiding an abrupt maximum-distance cutoff.
        std::uint32_t Primary = MaximumShadowCascades;
        std::uint32_t Secondary = MaximumShadowCascades;
        float Blend = 0.0f;
    };

    struct CascadedShadowState
    {
        CascadedShadowFrame Frame;
        std::array<TextureHandle, MaximumShadowCascades> Maps {};
        int MapResolution = 0;
    };

    [[nodiscard]] bool BuildCascadedShadowFrame(const CascadedShadowCamera& camera, const SceneLight& light,
        const std::vector<ShadowCasterBounds>& casters, const CascadedShadowSettings& settings,
        int mapResolution, float filterRadiusTexels, CascadedShadowFrame& output, std::string& error);
    [[nodiscard]] ShadowCascadeSelection SelectShadowCascade(const CascadedShadowFrame& frame, float eyeDistance);
}
