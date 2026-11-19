#pragma once

#include "Renderer/BakedLighting/BakedLightProbes.h"
#include "Renderer/FrameHistory.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Maths/Matrix4.h"

namespace Evans::Renderer
{
    inline constexpr std::uint32_t ProbeSHCoefficientCount = 9;
    inline constexpr std::uint32_t ProbeMomentTileSize = 8;
    inline constexpr std::uint32_t ProbeMomentAtlasColumns = 8;
    enum class ProbePass : std::uint32_t { ProjectIrradiance, ProjectDistance, Gather };

    // Raw CPU samples -> native BakedGI SH/moments -> native LightProbes gather.
    // Both shader units duplicate their own mathematical functions and ABI;
    // neither source includes another technique or another API's source.
    //
    // L2 coefficients encode irradiance/pi, with a quarter storage scale to fit
    // the source half-float HDR range. Gather restores this scale. Distance and
    // distance-squared are normalized by MaximumDistance before half storage.
    // Each probe has an 8x8 octahedral moment tile. Manual seam-aware bilinear
    // reads stay in the selected tile and never blend adjacent probes.
    struct LightProbeConstants
    {
        Matrix4 InverseViewProjection;
        Float4 GridMinimum;
        Float4 GridMaximum;
        Float4 Viewport;
        Float4 Appearance; // intensity, world normal bias, world visibility bias, visibility exponent
        Float4 Visibility; // normalized variance floor, angular moment exponent, max distance, atlas columns
        ShaderUInt4 Dimensions; // probe grid XYZ, rays per probe
        ShaderUInt4 Options; // phase, shading model, moment tile size, probe count
    };
    static_assert(sizeof(LightProbeConstants) == 176 && offsetof(LightProbeConstants, Options) == 160,
        "Light probe native uniform ABI changed");

    inline bool BuildLightProbeConstants(const GlobalIlluminationDesc& desc, const BakedProbeSamples& samples,
        const Techniques::GlobalIlluminationSettings& requested, Techniques::ShadingModel shading,
        int width, int height, LightProbeConstants& output)
    {
        if (width <= 0 || height <= 0 || !FrameHistory::IsFinite(desc.ViewProjection) || samples.Texels.empty()) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        LightProbeConstants constants;
        std::copy(inverse.Data(), inverse.Data()+16, constants.InverseViewProjection.Values.begin());
        if (!FrameHistory::IsFinite(constants.InverseViewProjection)) return false;
        const auto settings = Techniques::SanitizeGlobalIlluminationSettings(requested);
        const auto& probe = settings.Probes;
        const auto& source = samples.Settings;
        const float spacing = std::min({(source.Maximum.X-source.Minimum.X)/(source.Counts[0]-1),
            (source.Maximum.Y-source.Minimum.Y)/(source.Counts[1]-1), (source.Maximum.Z-source.Minimum.Z)/(source.Counts[2]-1)});
        if (!(spacing > 0) || !std::isfinite(spacing) || !(source.MaximumDistance > 0)) return false;
        const float relativeSpacing = spacing / source.MaximumDistance;
        constants.GridMinimum = {source.Minimum.X,source.Minimum.Y,source.Minimum.Z,0};
        constants.GridMaximum = {source.Maximum.X,source.Maximum.Y,source.Maximum.Z,0};
        constants.Viewport = {float(width),float(height),1.0f/width,1.0f/height};
        constants.Appearance = {settings.Intensity,probe.NormalBias*spacing,probe.VisibilityBias*spacing,probe.VisibilitySharpness};
        constants.Visibility = {probe.MinimumVariance*relativeSpacing*relativeSpacing,probe.MomentSharpness,
            source.MaximumDistance,float(ProbeMomentAtlasColumns)};
        constants.Dimensions = {source.Counts[0],source.Counts[1],source.Counts[2],source.RayCount};
        constants.Options = {static_cast<unsigned>(ProbePass::Gather),static_cast<unsigned>(shading),ProbeMomentTileSize,samples.ProbeCount()};
        output = constants; return true;
    }

    // Explicit APIs may abort commands after projection. Do not publish an
    // unsubmitted texture version; abort/resize invalidates it. Camera cuts
    // only reset screen histories and do not invalidate world-space probes.
    struct ProbeProjectionState
    {
        std::shared_ptr<const BakedProbeSamples> Published, Pending;
        float PublishedSharpness = 0, PendingSharpness = 0;
        bool Resolved = false;
        void Begin() noexcept { Pending.reset(); Resolved = false; }
        void Invalidate() noexcept { Published.reset(); Pending.reset(); Resolved = false; }
        bool NeedsProjection(const std::shared_ptr<const BakedProbeSamples>& source, float sharpness) const noexcept
        { return source != Published || sharpness != PublishedSharpness; }
        void Projected(const std::shared_ptr<const BakedProbeSamples>& source, float sharpness)
        { Published.reset(); Pending = source; PendingSharpness = sharpness; }
        void Commit() noexcept
        { if (Pending && Resolved) { Published = std::move(Pending); PublishedSharpness = PendingSharpness; } }
    };
}
