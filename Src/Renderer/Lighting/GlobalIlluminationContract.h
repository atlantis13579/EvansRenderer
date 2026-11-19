#pragma once

#include "Renderer/PostProcessing/MotionBlurContract.h"

namespace Evans::Renderer
{
    // A visible-layer, one-bounce diffuse estimator. Cosine-weighted hemisphere
    // sampling cancels the Lambertian cosine/pdf factor. Hit radiance comes from
    // this frame's pre-GI HDR image, never accumulated GI, fog or transparency.
    // The camera-visible outgoing radiance approximates the radiance toward the
    // receiver; hidden surfaces and off-screen emitters cannot be reconstructed.
    // Missing rays add zero, preserving the scene's existing ambient fallback.
    // McGuire and Mara, "Efficient GPU Screen-Space Ray Tracing", JCGT 2014:
    // https://jcgt.org/published/0003/04/04/
    // Ritschel et al., "Approximating Dynamic Global Illumination in Image Space",
    // I3D 2009: https://doi.org/10.1145/1507149.1507161
    //
    // Frame order: material/depth -> opaque HDR -> AO -> velocity -> SSGI ->
    // specular reflections -> volumes -> TAA/blur -> sharp transparency.
    // MSAA resolves nearest-depth material/velocity inputs before SSGI. Diffuse
    // transport is independent of the selected specular reflection and AO modes.
    //
    // Four fullscreen phases share one independent source per API: hemisphere
    // tracing, depth/normal-aware spatial filtering, validated temporal resolve,
    // and scene composition. Temporal writes color, R32 depth and world normals
    // to a private pending history set. Only successful presentation publishes
    // it. Invalid motion, disocclusion, a camera cut or changed inputs reject it.
    // Composition writes a distinct HDR image and copies back, avoiding both
    // attachment feedback and half-float overflow in additive hardware blending.
    //
    // Image slots: scene HDR, depth, normal/roughness, diffuse/opacity,
    // ambient/metallic, velocity, current estimate, history color/depth/normal.
    struct GlobalIlluminationConstants
    {
        Matrix4 ViewProjection;
        Matrix4 InverseViewProjection;
        Matrix4 PreviousInverseViewProjection;
        Float4 PreviousDepthPlane;
        Float4 Viewport; // extent XY, reciprocal extent ZW
        Float4 Trace; // world distance, slab thickness, normal bias, pixel stride
        Float4 Appearance; // intensity, edge fade pixels, spatial depth scale, normal threshold
        Float4 Temporal; // history weight, relative depth tolerance, normal threshold, reserved
        Float4 JitterDelta; // current-minus-previous raster displacement in top-left pixels
        ShaderUInt4 Options; // phase, rays per pixel, maximum ray steps, sequence index
        ShaderUInt4 History; // valid history, shading model, reserved, reserved
    };
    static_assert(sizeof(GlobalIlluminationConstants) == 320 &&
        offsetof(GlobalIlluminationConstants, Options) == 288 &&
        offsetof(GlobalIlluminationConstants, History) == 304, "SSGI uniform ABI changed");

    inline bool BuildGlobalIlluminationConstants(const GlobalIlluminationDesc& desc,
        const Techniques::GlobalIlluminationSettings& requested, Techniques::ShadingModel shading,
        int width, int height, bool nativeHistoryValid, GlobalIlluminationConstants& output)
    {
        if (width <= 0 || height <= 0 || !FrameHistory::IsFinite(desc.ViewProjection) ||
            !FrameHistory::IsFinite(desc.PreviousViewProjection)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        MotionBlurDesc previous; previous.ViewProjection = desc.PreviousViewProjection;
        MotionBlurConstants geometry;
        if (!BuildMotionBlurConstants(previous, {}, width, height, geometry)) return false;
        GlobalIlluminationConstants candidate {};
        candidate.ViewProjection = desc.ViewProjection;
        std::copy(inverse.Data(), inverse.Data() + 16, candidate.InverseViewProjection.Values.begin());
        if (!FrameHistory::IsFinite(candidate.InverseViewProjection)) return false;
        candidate.PreviousInverseViewProjection = geometry.InverseViewProjection;
        candidate.PreviousDepthPlane = geometry.DepthPlane;
        const auto settings = Techniques::SanitizeGlobalIlluminationSettings(requested);
        constexpr float PixelStride = 1.0f, MinimumSpatialDepthScale = 0.0001f;
        const auto spatialDepthScale = std::max(MinimumSpatialDepthScale, settings.Thickness * 2.0f);
        candidate.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
        candidate.Trace = {settings.MaximumDistance, settings.Thickness, settings.NormalBias, PixelStride};
        candidate.Appearance = {settings.Intensity, settings.EdgeFadePixels, spatialDepthScale, settings.NormalThreshold};
        candidate.Temporal = {settings.HistoryWeight, settings.DepthTolerance, settings.NormalThreshold, 0};
        candidate.JitterDelta = {desc.Sampling.JitterPixels.X - desc.Sampling.PreviousJitterPixels.X,
            desc.Sampling.JitterPixels.Y - desc.Sampling.PreviousJitterPixels.Y, 0, 0};
        if (!std::isfinite(candidate.JitterDelta.X) || !std::isfinite(candidate.JitterDelta.Y)) return false;
        candidate.Options = {0, settings.RayCount, settings.MaximumSteps, desc.Sampling.SampleIndex};
        candidate.History = {nativeHistoryValid && desc.Sampling.HistoryValid ? 1u : 0u,
            static_cast<std::uint32_t>(shading), 0, 0};
        output = candidate;
        return true;
    }

    // GPU targets retain their last committed slot across an abandoned frame.
    // Allocation/resize and disabling the effect invalidate correspondence.
    struct GlobalIlluminationHistory
    {
        unsigned ReadIndex = 0;
        bool Valid = false;
        bool Resolved = false;
        void Begin() noexcept { Resolved = false; }
        void Invalidate() noexcept { Valid = Resolved = false; }
        unsigned WriteIndex() const noexcept { return 1u - ReadIndex; }
        void Commit() noexcept
        {
            if (Resolved) { ReadIndex = WriteIndex(); Valid = true; }
        }
    };
}
