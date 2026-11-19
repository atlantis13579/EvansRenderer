#pragma once

#include "Renderer/RealtimeShaderContract.h"
#include "Techniques/ParallaxOcclusionMapping.h"

namespace Evans::Renderer
{
    inline bool UsesParallaxMaterial(const PipelineState& pipeline,PassType pass) noexcept
    {
        return pass==PassType::GBuffer && Techniques::UsesParallaxOcclusion(pipeline.Parallax) &&
            pipeline.Fill==FillMode::Solid && !pipeline.BlendEnabled && !pipeline.OutlineExtrusion &&
            pipeline.AlphaCutoff<=0 && !pipeline.OpacityTexture && pipeline.MaterialOpacity>=1 &&
            pipeline.Shading!=Techniques::ShadingModel::Flat && pipeline.Shading!=Techniques::ShadingModel::Gouraud;
    }
    struct ParallaxConstants
    {
        Float4 Shape; // depth fraction, coordinate tiling, normal strength, grazing cosine
        ShaderUInt4 Steps; // minimum/maximum layers, refinement count, coordinate mode
        Float4 ObjectBounds; // local bounds minimum XYZ, reciprocal maximum local extent
    };
    static_assert(sizeof(ParallaxConstants)==48 && offsetof(ParallaxConstants,ObjectBounds)==32,
        "Parallax shader constants require three aligned vectors");
    [[nodiscard]] bool BuildParallaxConstants(const Techniques::ParallaxOcclusionSettings& settings,
        Float3 boundsMinimum,Float3 boundsMaximum,ParallaxConstants& output) noexcept;
}
