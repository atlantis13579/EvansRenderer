#pragma once

#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    enum class LightListSelection : std::uint32_t
    {
        FullScene = 0, Opaque = 1, Transparent = 2
    };

    // Scene-order indices are shared by assignment, materials and shadow lookup.
    // Captures select FullScene because main-camera lists do not bound their rays.
    // The control word is zero for FullScene, a selection for Forward+, or the
    // positive depth-slice count for Clustered Forward. Each pipeline interprets
    // its own layout; an overflow always falls back to the complete scene loop.
    struct LightListSurfaceConstants
    {
        ShaderUInt4 LightLayout {0, UINT32_MAX, 0, 0}; // count, shadow index, control, capacity
        ShaderUInt4 SurfaceTiles; // tile side, columns, viewport width/height
    };
    static_assert(sizeof(LightListSurfaceConstants) == 32 &&
        offsetof(LightListSurfaceConstants, SurfaceTiles) == 16, "Light-list surface ABI changed");
}
