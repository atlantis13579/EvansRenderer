#pragma once

#include "Renderer/Types.h"
#include "Techniques/SubsurfaceScattering.h"

namespace Evans::Renderer
{
    inline constexpr std::size_t SubsurfaceKernelSize = 25;
    struct SubsurfaceKernel
    {
        // RGB are independently normalized bin masses. W is a signed offset
        // measured in multiples of the settings' world-space profile distance.
        std::array<Float4, SubsurfaceKernelSize> Taps;
        float Support = 0;
    };
    // Project a radial normalized-diffusion distribution onto one image axis.
    // The product of the two marginals is a separable approximation, not the
    // original radially symmetric BSSRDF. The CPU integrates bins instead of
    // evaluating its singular center point. No fixed skin-color kernel is baked.
    [[nodiscard]] SubsurfaceKernel BuildSubsurfaceKernel(const Techniques::SubsurfaceScatteringSettings& settings);
}
