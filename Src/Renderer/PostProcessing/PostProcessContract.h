#pragma once

#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // Post effects retain display-referred RGBA8 color. FXAA and grain operate
    // in encoded sRGB, as does NPR ink/tonal abstraction; vignette attenuates
    // and distortion samples linear light.
    // Each API owns its source and accounts for its native texture/view format.
    // Depth and selection outlines remain separate from this color-only chain.
    struct PostProcessConstants
    {
        Float4 InverseExtentStrengths;
        ShaderUInt4 GrainSeedReserved;
        Float4 DistortionStrengthRadiusWeights;
        Float4 NprEdge; // sigma, wide gain, threshold, tanh sharpness
        Float4 NprStyle; // contour strength, hatch strength, color retention, blend
        Float4 NprHatch; // spacing, width, cosine, sine of line normal
        Float4 NprTone; // posterization levels, reserved
    };
    static_assert(sizeof(PostProcessConstants) == 112,
        "Native post shaders use seven aligned four-component uniform vectors");
    static_assert(offsetof(PostProcessConstants, GrainSeedReserved) == 16,
        "The integer seed must retain its own aligned vector without float conversion");
    static_assert(offsetof(PostProcessConstants, DistortionStrengthRadiusWeights) == 32,
        "Distortion extends the existing post constant prefix");

    static_assert(offsetof(PostProcessConstants, NprEdge) == 48 &&
        offsetof(PostProcessConstants, NprTone) == 96, "NPR preserves the existing post prefix");

    // XDoG: Winnemoeller, Kyprianidis and Olsen, Computers & Graphics (2012).
    // https://www.kyprianidis.com/p/cag2012/
    // Independent API shaders evaluate normalized finite Gaussian kernels in
    // display sRGB, then a soft threshold. Analytic hatch layers replace paper
    // textures; this is isotropic image filtering, without flow-field tracking.
    inline PostProcessConstants BuildPostProcessConstants(
        const Techniques::PostProcessSettings& settings, int width, int height)
    {
        const auto sanitized = Techniques::SanitizePostProcessSettings(settings);
        const float viewportWidth = static_cast<float>(std::max(width, 1));
        const float viewportHeight = static_cast<float>(std::max(height, 1));
        const float widthSquared = viewportWidth * viewportWidth;
        const float heightSquared = viewportHeight * viewportHeight;
        const float diagonalSquared = widthSquared + heightSquared;
        constexpr float Pi = 3.14159265358979323846f;
        const auto& npr = sanitized.NPR;
        const float hatchAngle = npr.HatchAngleDegrees * Pi / 180.0f;
        return {
            {1.0f / viewportWidth, 1.0f / viewportHeight,
             sanitized.VignetteStrength, sanitized.FilmGrainStrength},
            {sanitized.FilmGrainSeed, 0u, 0u, 0u},
            {sanitized.DistortionStrength, widthSquared / diagonalSquared,
             heightSquared / diagonalSquared, 0.0f},
            {npr.EdgeSigma, npr.EdgeWeight, npr.EdgeThreshold, npr.EdgeSharpness},
            {npr.EdgeStrength, npr.HatchStrength, npr.ColorStrength, npr.Blend},
            {npr.HatchSpacing, npr.HatchWidth, std::cos(hatchAngle), std::sin(hatchAngle)},
            {static_cast<float>(npr.ToneLevels), 0, 0, 0}};
    }
}
