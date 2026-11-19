#pragma once

#include "Renderer/Lumen/LumenProbeIntegration.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenTranslucencyMaximumProbes=2048;
    inline constexpr unsigned LumenTranslucencyActive=1,LumenTranslucencyVisibilitySupport=2;
    // Wright et al., "Lumen", SIGGRAPH 2022: low-frequency lighting for receivers
    // absent from the opaque G-buffer. A view-frustum grid traces the same opaque
    // distance fields and surface cache; an optional world cache shortens rays.
    // XY coefficient bands and stacked Z slices permit filtered SH interpolation
    // using one ordinary texture, including GLES forward material shaders.
    struct LumenTranslucencyFrame
    {
        Matrix4 ViewProjection,Inverse;
        ShaderUInt4 Grid; // X/Y/Z cells, availability bits: active=1, visibility support=2.
        Float4 Distribution; // World horizon, depth exponent, intensity, SH packing scale.
        Float4 Atlas; // Width, height, homogeneous epsilon, normal squared epsilon.
    };
    static_assert(sizeof(LumenTranslucencyFrame)==176,"Lumen translucency ABI changed");
    bool BuildLumenTranslucencyFrame(const Matrix4& camera,unsigned width,unsigned height,
        const Techniques::GlobalIlluminationSettings& settings,LumenTranslucencyFrame& output,
        LumenRadianceTraceUpload& trace,std::string& error);
    // One GPU point query per world probe. Its surface-free support sphere uses
    // the same immutable SDF snapshot as transport, not sampled ray distances.
    bool BuildLumenTranslucencySupportFrame(const LumenDistanceSceneInfo&,const LumenRadianceTraceUpload&,
        LumenDistanceTraceFrame&,std::string& error);
}
