#pragma once

#include "Renderer/Types.h"
#include "Techniques/HairFur.h"

namespace Evans::Renderer
{
    struct FiberScatteringProfile
    {
        Float4 Variance; // R, TT, TRT, aggregate higher orders.
        Float4 SinTilt,CosTilt; // alpha, 2 alpha, 4 alpha, reserved.
        Float4 Optics; // IOR, azimuthal logistic scale, reflectance/absorption fit, intensity.
    };
    static_assert(sizeof(FiberScatteringProfile)==64,"Fiber scattering native ABI changed");
    FiberScatteringProfile BuildFiberScatteringProfile(const Techniques::HairFurSettings& settings) noexcept;
}
