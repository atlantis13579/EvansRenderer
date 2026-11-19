#pragma once

#include "Renderer/Reflections/PlanarReflection.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // CPU layout for independently authored native planar lookup/composition
    // shaders. Position and normal qualification isolate the chosen receiver;
    // capture projection, plane and material inputs must belong to the same frame.
    struct PlanarReflectionConstants
    {
        Matrix4 InverseViewProjection;
        Matrix4 ReflectionViewProjection;
        Float4 ReceiverPlane;
        Float4 Viewport;
        Float4 Sampling; // intensity, last mip, plane tolerance, minimum normal cosine
        ShaderUInt4 Options; // phase, surface model, active capture, reserved
        Float4 ReceiverMinimum; // xyz: world bounds, w: bounded receiver enabled
        Float4 ReceiverMaximum;
    };
    static_assert(sizeof(PlanarReflectionConstants) == 224, "Planar reflection uniform layout changed");

    inline bool BuildPlanarReflectionConstants(const ReflectionDesc& desc, const PlanarCaptureFrame& frame,
        const Techniques::ReflectionSettings& settings, Techniques::ShadingModel shading,
        int width, int height, PlanarReflectionConstants& output)
    {
        if (width <= 0 || height <= 0 || !frame.Width || !frame.Height || !frame.MipCount ||
            static_cast<std::uint32_t>(width) != frame.MainWidth || static_cast<std::uint32_t>(height) != frame.MainHeight ||
            desc.ViewProjection.Values != frame.MainViewProjection.Values) return false;
        for (const auto* matrix : {&frame.MainViewProjection, &frame.InverseMainViewProjection, &frame.ViewProjection})
            for (float value : matrix->Values) if (!std::isfinite(value)) return false;
        for (float value : {frame.ReceiverPlane.X, frame.ReceiverPlane.Y, frame.ReceiverPlane.Z, frame.ReceiverPlane.W})
            if (!std::isfinite(value)) return false;
        const auto sanitized = Techniques::SanitizeReflectionSettings(settings);
        PlanarReflectionConstants candidate {};
        candidate.InverseViewProjection = frame.InverseMainViewProjection;
        candidate.ReflectionViewProjection = frame.ViewProjection;
        candidate.ReceiverPlane = frame.ReceiverPlane;
        candidate.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
        candidate.Sampling = {sanitized.Planar.Intensity, float(frame.MipCount - 1), sanitized.Planar.ReceiverTolerance,
            sanitized.Planar.MinimumNormalCosine};
        candidate.Options = {0, static_cast<std::uint32_t>(shading), frame.Active() ? 1u : 0u, 0};
        const auto& minimum = sanitized.Planar.ReceiverMinimum;
        const auto& maximum = sanitized.Planar.ReceiverMaximum;
        candidate.ReceiverMinimum = {minimum[0], minimum[1], minimum[2], sanitized.Planar.BoundedReceiver ? 1.0f : 0.0f};
        candidate.ReceiverMaximum = {maximum[0], maximum[1], maximum[2], 0};
        output = candidate;
        return true;
    }
}
