#pragma once
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // McGuire and Bavoil, "Weighted Blended Order-Independent Transparency",
    // JCGT 2(2), 2013: https://jcgt.org/published/0002/02/09/
    // Accumulate (radiance * alpha * weight, alpha * weight) additively and
    // revealage multiplicatively as product(1 - alpha). Resolve the normalized
    // radiance over the opaque scene with coverage = 1 - revealage. Background
    // coverage is exact; inter-layer color/occlusion remains an approximation.
    // The bounded heuristic is max(alpha^opacityExponent *
    // (1 - hardwareDepth * depthScale)^depthExponent, minimumWeight).
    // Float32 targets avoid half-float overflow for HDR and deep layer stacks.
    // MSAA retains separate accumulations/revealage and resolves per sample.
    struct WeightedTransparencyConstants
    {
        Float4 Weights; // opacity exponent, depth exponent, depth scale, minimum weight
        ShaderUInt4 Options; // enabled, reserved
    };
    static_assert(sizeof(WeightedTransparencyConstants) == 32, "Weighted transparency uniform layout changed");
    inline WeightedTransparencyConstants BuildWeightedTransparencyConstants(const Techniques::TransparencySettings& settings)
    {
        const auto sanitized = Techniques::SanitizeTransparencySettings(settings);
        constexpr float DepthScale = 0.9f;
        constexpr float MinimumWeight = 1.0e-4f;
        return {{sanitized.OpacityExponent, sanitized.DepthExponent, DepthScale, MinimumWeight}, {1, 0, 0, 0}};
    }
}
