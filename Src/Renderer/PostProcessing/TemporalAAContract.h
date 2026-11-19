#pragma once

#include "Renderer/PostProcessing/MotionBlurContract.h"

namespace Evans::Renderer
{
    // History color is accumulated at the unjittered output grid. History depth
    // records the jittered geometry grid, so depth lookup additionally subtracts
    // current-minus-previous jitter. Depth comparisons use previous optical units.
    struct TemporalAAConstants
    {
        Matrix4 PreviousInverseViewProjection;
        Float4 PreviousDepthPlane;
        Float4 ExtentWeightTolerance;
        Float4 JitterDelta;
        ShaderUInt4 Options;
    };
    static_assert(sizeof(TemporalAAConstants) == 128, "Temporal AA uniform layout changed");
    inline bool BuildTemporalAAConstants(const TemporalAADesc& desc, const Techniques::TemporalAASettings& requested,
        int width, int height, bool nativeHistoryValid, TemporalAAConstants& output)
    {
        MotionBlurDesc camera;
        camera.ViewProjection = desc.PreviousViewProjection;
        MotionBlurConstants geometry;
        if (!BuildMotionBlurConstants(camera, {}, width, height, geometry)) return false;
        const auto settings = Techniques::SanitizeTemporalAASettings(requested);
        output = {};
        output.PreviousInverseViewProjection = geometry.InverseViewProjection;
        output.PreviousDepthPlane = geometry.DepthPlane;
        output.ExtentWeightTolerance = {static_cast<float>(width), static_cast<float>(height),
            settings.HistoryWeight, settings.DepthTolerance};
        output.JitterDelta = {desc.Sampling.JitterPixels.X - desc.Sampling.PreviousJitterPixels.X,
            desc.Sampling.JitterPixels.Y - desc.Sampling.PreviousJitterPixels.Y, 0, 0};
        output.Options.X = nativeHistoryValid && desc.Sampling.HistoryValid ? 1u : 0u;
        return std::isfinite(output.JitterDelta.X) && std::isfinite(output.JitterDelta.Y);
    }
}
