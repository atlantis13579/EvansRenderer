#pragma once

#include "Renderer/RayTracing/SceneRayQuery.h"

namespace Evans::Renderer
{
    inline constexpr std::uint32_t MinimumProbeGridAxis = 2;
    inline constexpr std::uint32_t MaximumProbeGridAxis = 8;
    inline constexpr std::uint32_t MinimumProbeRayCount = 16;
    inline constexpr std::uint32_t MaximumProbeRayCount = 1024;
    inline constexpr std::uint32_t ProbeMetadataBorder = 1;

    struct BakedProbeSettings
    {
        Float3 Minimum = {-1, -1, -1};
        Float3 Maximum = {1, 1, 1};
        std::array<std::uint32_t, 3> Counts = {3, 3, 3};
        std::uint32_t RayCount = 256;
        float MaximumDistance = 100.0f;
        float SurfaceBias = 0.001f;
        float MaximumBackfaceFraction = 0.25f;
        bool Shadows = true;
    };

    struct BakedProbeSamples
    {
        BakedProbeSettings Settings;
        std::uint32_t ActiveProbeCount = 0;
        std::uint64_t SurfaceHitCount = 0;
        std::uint64_t ShadowQueryCount = 0;
        // Row-major RGBA32F upload, with no API-dependent orientation:
        // (0,0): reserved zero; (ray+1,0): world direction.xyz, solid angle;
        // (0,probe+1): position.xyz, active flag;
        // (ray+1,probe+1): linear outgoing radiance.rgb, hit distance.
        // Miss radiance is black and its distance is MaximumDistance. Backfaces
        // remain distance samples but emit black. Probe index is x + Nx*(y+Ny*z).
        // Raw samples are inputs to each API's own SH/moment projection shader.
        std::vector<Float4> Texels;

        [[nodiscard]] std::uint32_t ProbeCount() const noexcept
        { return Settings.Counts[0] * Settings.Counts[1] * Settings.Counts[2]; }
        [[nodiscard]] std::uint32_t Width() const noexcept { return Settings.RayCount + ProbeMetadataBorder; }
        [[nodiscard]] std::uint32_t Height() const noexcept { return ProbeCount() + ProbeMetadataBorder; }
    };

    // One diffuse bounce from existing opaque/cutout geometry and scene lights.
    // Immutable query + settings + lights are the entire input: no main-camera,
    // temporal history, previous GI, post-processing or authored bake assets.
    // Failure preserves the caller's previous complete output.
    [[nodiscard]] bool BakeLightProbeSamples(const SceneRayQuery& scene, const LightingState& lighting,
        const BakedProbeSettings& settings, BakedProbeSamples& output, std::string& error);
}
