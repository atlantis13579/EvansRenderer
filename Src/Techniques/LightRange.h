#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Techniques
{
    // Zero preserves unbounded inverse-square lighting. Invalid and subnormal
    // radii use that same mode, avoiding CPU/GPU flush-to-zero disagreement.
    inline float CanonicalLightRange(float range) noexcept
    {
        return std::isfinite(range) && range >= std::numeric_limits<float>::min() ? range : 0.0f;
    }

    // Karis' smooth inverse-square window, also described in Filament, section
    // "Punctual lights / Attenuation function": (max(1 - (d/r)^4, 0))^2.
    // https://google.github.io/filament/main/filament.html#lighting/directlighting/punctuallights
    // The caller retains its own inverse-square distance floor. Rectangle lights
    // apply the window at EACH emitter sample; directional lights bypass it.
    // Gouraud applies this at vertices; its interpolated illumination is not
    // confined to the analytic sphere and cannot consume per-pixel light lists.
    inline float LightRangeWindow(float x, float y, float z, float requestedRange) noexcept
    {
        const float range = CanonicalLightRange(requestedRange);
        if (range == 0.0f) return 1.0f;
        if (std::max({std::abs(x), std::abs(y), std::abs(z)}) >= range) return 0.0f;
        const float nx = x / range, ny = y / range, nz = z / range;
        const float relativeSquaredDistance = nx * nx + ny * ny + nz * nz;
        const float window = std::max(1.0f - relativeSquaredDistance * relativeSquaredDistance, 0.0f);
        return window * window;
    }
}
