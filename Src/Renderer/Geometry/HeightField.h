#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    struct HeightFieldSample
    {
        float Height=0; // normalized crest height: zero is the valley, one is the reference plane
        Float2 Gradient; // derivatives with respect to the two height coordinates
    };
    // A periodic analytic field is a runtime teaching input, not a new authored
    // asset. Gaussian footprint filtering damps the two diagonal Fourier modes;
    // the returned derivatives belong to exactly that filtered height function.
    // TODO: import explicitly authored scalar height textures with linear data
    // semantics. Albedo brightness and a normal map do not uniquely define height.
    [[nodiscard]] HeightFieldSample EvaluatePeriodicHeight(Float2 coordinate,Float2 dx={},Float2 dy={}) noexcept;
}
