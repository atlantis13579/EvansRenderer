#pragma once

#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/FrameHistory.h"
#include "Maths/Matrix4.h"

namespace Evans::Renderer
{
    // Velocity stores current-minus-previous pixels before shutter scaling, plus
    // one minus previous hardware depth and validity. Reversing depth before
    // RGBA16F storage preserves perspective depth precision near one; temporal
    // consumers decode it before optical-depth rejection. Diagnostic readbacks
    // retain conventional hardware depth. It belongs to opaque, alpha-tested
    // geometry and is generated with read-only opaque depth. A centered shutter
    // uses halfVelocity = velocity * shutterSeconds / (2 * sampleInterval).
    // Tile/neighbor maxima bound reconstruction to one tile radius. Transparent
    // surfaces are composed later; one visible layer cannot recover hidden color.
    // Independent MotionVectors and MotionBlur sources retain one explicit native
    // uniform layout; producer geometry does not read shutter or filter fields.
    inline Float4 DecodeMotionVector(Float4 encoded) noexcept
    {
        encoded.Z=encoded.W>0 ? 1-encoded.Z : 0;
        return encoded;
    }

    struct MotionBlurConstants
    {
        Matrix4 World;
        Matrix4 ViewProjection;
        Matrix4 PreviousWorldViewProjection;
        Matrix4 InverseViewProjection;
        Float4 DepthPlane;
        Float4 ViewportExposure; // width, height, half shutter/interval, maximum radius
        Float4 Material; // opacity, alpha cutoff, current-minus-previous jitter in pixels
        ShaderUInt4 Options; // pass, diffuse texture, opacity texture, valid history
    };
    static_assert(sizeof(MotionBlurConstants) == 320, "Motion-blur native uniform size changed");
    static_assert(offsetof(MotionBlurConstants, Options) == 304, "Motion-blur options offset changed");

    inline bool BuildMotionGeometryConstants(const PipelineState& pipeline, const Matrix4& world,
        const Matrix4& camera, const MotionTransform& motion, int width, int height,
        bool diffuse, bool opacity, MotionBlurConstants& output)
    {
        if (width <= 0 || height <= 0 || !FrameHistory::IsFinite(world) || !FrameHistory::IsFinite(camera) ||
            (motion.Valid && !FrameHistory::IsFinite(motion.PreviousWorldViewProjection))) return false;
        const auto current = Maths::Matrix4(camera.Values) * Maths::Matrix4(world.Values);
        output = {};
        output.World = world;
        output.ViewProjection = camera;
        std::copy(current.Data(), current.Data() + 16, output.PreviousWorldViewProjection.Values.begin());
        if (motion.Valid) output.PreviousWorldViewProjection = motion.PreviousWorldViewProjection;
        output.ViewportExposure = {static_cast<float>(width), static_cast<float>(height), 0, 0};
        if (!std::isfinite(motion.JitterDeltaPixels.X) || !std::isfinite(motion.JitterDeltaPixels.Y)) return false;
        output.Material = {pipeline.Color.W * pipeline.MaterialOpacity, pipeline.AlphaCutoff,
            motion.JitterDeltaPixels.X, motion.JitterDeltaPixels.Y};
        output.Options = {0, diffuse ? 1u : 0u, opacity ? 1u : 0u, motion.Valid ? 1u : 0u};
        return FrameHistory::IsFinite(output.PreviousWorldViewProjection);
    }

    inline bool BuildMotionBlurConstants(const MotionBlurDesc& desc,
        const Techniques::MotionBlurSettings& requested, int width, int height, MotionBlurConstants& output)
    {
        if (width <= 0 || height <= 0 || !FrameHistory::IsFinite(desc.ViewProjection)) return false;
        if (desc.Motion.Valid && (!std::isfinite(desc.Motion.DeltaSeconds) ||
            desc.Motion.DeltaSeconds < FrameHistory::MinimumIntervalSeconds ||
            desc.Motion.DeltaSeconds > FrameHistory::MaximumIntervalSeconds)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        output = {};
        std::copy(inverse.Data(), inverse.Data() + 16, output.InverseViewProjection.Values.begin());
        if (!FrameHistory::IsFinite(output.InverseViewProjection)) return false;
        const auto& projection = desc.ViewProjection.Values;
        // Clip W is optical eye distance for perspective cameras, including
        // asymmetric frusta. Affine cameras use their linear clip-depth plane.
        auto planeLength = [&](int row)
        {
            const double x = projection[row * 4], y = projection[row * 4 + 1], z = projection[row * 4 + 2];
            return std::sqrt(x * x + y * y + z * z);
        };
        constexpr double MinimumPlaneLength = 1.0e-12;
        const int row = planeLength(3) > MinimumPlaneLength ? 3 : 2;
        const double length = planeLength(row);
        if (length <= MinimumPlaneLength || !std::isfinite(length)) return false;
        output.DepthPlane = {static_cast<float>(projection[row * 4] / length),
            static_cast<float>(projection[row * 4 + 1] / length), static_cast<float>(projection[row * 4 + 2] / length),
            static_cast<float>(projection[row * 4 + 3] / length)};
        if (!std::isfinite(output.DepthPlane.X) || !std::isfinite(output.DepthPlane.Y) ||
            !std::isfinite(output.DepthPlane.Z) || !std::isfinite(output.DepthPlane.W)) return false;
        const auto settings = Techniques::SanitizeMotionBlurSettings(requested);
        const float halfExposure = desc.Motion.Valid ? 0.5f * settings.ShutterSeconds / desc.Motion.DeltaSeconds : 0.0f;
        output.ViewportExposure = {static_cast<float>(width), static_cast<float>(height), halfExposure, settings.MaxRadiusPixels};
        return true;
    }
}
