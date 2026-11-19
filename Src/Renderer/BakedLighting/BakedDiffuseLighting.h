#pragma once

#include "Renderer/RayTracing/SceneRayQuery.h"

namespace Evans::Renderer
{
    inline constexpr float MaximumBakedRadiance = 65504.0f; // Largest finite binary16 value.

    struct BakedDiffuseLightingSettings
    {
        float SurfaceBias = 0.001f;
        bool Shadows = true;
    };

    // Outgoing emission + direct diffuse reflection at a known ray hit. This
    // CPU transport kernel is shared by probe and surface-atlas producers;
    // each native API still owns its independent projection/gather shaders.
    // The caller validates inputs and rejects nonfinite radiance before storage.
    [[nodiscard]] Maths::Vector3 EvaluateBakedDiffuse(const SceneRayQuery& scene, const SceneRayHit& hit,
        const std::vector<SceneLight>& lights, const BakedDiffuseLightingSettings& settings,
        std::uint64_t& shadowQueries);
}
