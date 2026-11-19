#pragma once

#include "Renderer/Lighting/AmbientOcclusionContract.h"
#include "Renderer/Lighting/SubsurfaceDiffusion.h"

namespace Evans::Renderer
{
    // Execute after opaque MSAA resolve and before AO/GI/reflections. The same
    // material samples and light selection produce both scene and diffuse light.
    struct SubsurfaceScatteringDesc
    {
        Techniques::SubsurfaceScatteringSettings Settings;
    };
    inline bool SubsurfacePrecedesLaterPasses(const FrameCapture& frame)
    {
        const auto& stats=frame.Statistics;
        if (stats.AmbientOcclusionPassCount || stats.GlobalIlluminationPassCount || stats.ReflectionPassCount ||
            stats.DepthOfFieldPassCount || stats.MotionBlurPassCount || stats.TemporalAAPassCount ||
            stats.VolumetricFogPassCount || stats.VolumetricLightingPassCount || stats.ParticipatingMediaPassCount ||
            stats.VolumetricCloudPassCount || stats.LightShaftPassCount) return false;
        return std::none_of(frame.Passes.begin(),frame.Passes.end(),[](const PassRecord& pass)
        { return pass.Description.Type==PassType::Color || pass.Description.Type==PassType::Outline; });
    }
    struct SubsurfaceScatteringConstants
    {
        Matrix4 InverseViewProjection;
        Float4 Viewport;
        Float4 Diffusion; // World profile distance, strength, pixel support limit, kernel support.
        Float4 Rejection; // Optical depth scale, normal exponent, reserved, reserved.
        Float4 CameraForward;
        ShaderUInt4 Options; // Pass, kernel taps, reserved, reserved.
        std::array<Float4, SubsurfaceKernelSize> Kernel;
    };
    static_assert(sizeof(SubsurfaceScatteringConstants) == 544 && offsetof(SubsurfaceScatteringConstants, Options) == 128 &&
        offsetof(SubsurfaceScatteringConstants, Kernel) == 144, "Subsurface diffusion uniform ABI changed");

    // Only direct diffuse radiance is transported. Specular reflection, emission
    // and ambient illumination remain in the opaque HDR image. Diffusion runs
    // before AO/GI/reflections, which retain their own lighting semantics.
    // Jimenez et al., "Separable Subsurface Scattering" (2015):
    // https://www.iryoku.com/separable-sss/
    // TODO: Generate mesh thickness before adding backlit transmission. Normal
    // maps and the editable diffusion profile do not supply this information.
    inline bool BuildSubsurfaceScatteringConstants(const Matrix4& camera,
        const Techniques::SubsurfaceScatteringSettings& requested, int width, int height,
        Techniques::SubsurfacePass pass, SubsurfaceScatteringConstants& output)
    {
        output = {};
        if (pass > Techniques::SubsurfacePass::Composite) return false;
        Techniques::AmbientOcclusionSettings optical; optical.Mode = Techniques::AmbientOcclusionMode::GTAO;
        AmbientOcclusionConstants projection;
        if (!BuildAmbientOcclusionConstants({camera}, optical, width, height,
            Techniques::AmbientOcclusionPassKind::Estimate, projection)) return false;
        const auto settings = Techniques::SanitizeSubsurfaceScatteringSettings(requested);
        const auto kernel = BuildSubsurfaceKernel(settings);
        output.InverseViewProjection = projection.InverseViewProjection;
        output.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
        output.Diffusion = {settings.Distance, settings.Strength, settings.MaximumRadiusPixels, kernel.Support};
        output.Rejection = {settings.DepthScale, settings.NormalPower, 0, 0};
        output.CameraForward = projection.ForwardIntensity;
        output.Options = {static_cast<std::uint32_t>(pass), static_cast<std::uint32_t>(SubsurfaceKernelSize), 0, 0};
        output.Kernel = kernel.Taps;
        return true;
    }
}
