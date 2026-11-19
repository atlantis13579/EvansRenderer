#pragma once

#include "Renderer/VirtualShadowMaps/VirtualShadowMaps.h"

namespace Evans::Renderer
{
    struct VirtualShadowSceneCaster
    {
        std::string Key;
        VirtualShadowCaster Draw;
        Float3 BoundsMinimum,BoundsMaximum; // Local bounds of the selected geometry.
        std::uint64_t GeometryRevision=0,PreparationRevision=0;
    };
    struct VirtualShadowSceneState
    {
        VirtualShadowIdentity Identity;
        std::vector<VirtualShadowSceneCaster> Casters;
    };
    struct VirtualShadowSceneUpdate
    {
        VirtualShadowSceneState Scene;
        std::vector<VirtualShadowRectangle> Invalidations;
    };
    // Build a candidate without changing the last presented scene. The renderer
    // commits this state together with its native atlas only after Present.
    // Geometry, coverage and transforms invalidate both old and new footprints;
    // camera-only changes request pages without discarding cached caster depth.
    bool BuildVirtualShadowSceneUpdate(const Matrix4& light,std::vector<VirtualShadowSceneCaster> casters,
        const VirtualShadowSceneState* presented,VirtualShadowSceneUpdate& output,std::string& error);
}
