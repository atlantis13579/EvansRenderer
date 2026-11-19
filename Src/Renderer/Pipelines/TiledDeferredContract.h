#pragma once

#include "Renderer/Pipelines/DeferredContract.h"
#include "Renderer/Lighting/LightListContract.h"

namespace Evans::Renderer
{
    // Deferred material texels and the light lists must describe the same camera
    // and original depth samples. Tile construction follows G-buffer completion;
    // transparency reads the separate full-frustum lists after opaque lighting.
    // Lauritzen, "Deferred Rendering for Current and Future Rendering Pipelines"
    // (SIGGRAPH 2010). Geometry uses the shared five-target material producer.
    struct TiledDeferredReconstruction
    {
        Matrix4 InverseViewProjection;
        Float4 BackgroundColor;
    };
    static_assert(sizeof(TiledDeferredReconstruction) == 80, "Tiled reconstruction layout changed");

    inline bool BuildTiledDeferredConstants(const DeferredLightingDesc& desc,
        bool shadowAvailable, const Float4& background, RealtimeDrawConstants& surface,
        TiledDeferredReconstruction& reconstruction)
    {
        DeferredConstants camera;
        if (!BuildDeferredConstants(desc, shadowAvailable, background, camera)) return false;
        PipelineState pipeline;
        pipeline.Shading = desc.Shading; pipeline.ToonBands = desc.ToonBands;
        pipeline.Lighting = desc.Lighting;
        surface = BuildRealtimeDrawConstants(pipeline, Matrix4::Identity(), desc.ViewProjection,
            desc.LightViewProjection, PassType::Color, desc.SamplesShadowMap && shadowAvailable,
            false, false, false, true);
        reconstruction = {camera.InverseViewProjection, background};
        return true;
    }
}
