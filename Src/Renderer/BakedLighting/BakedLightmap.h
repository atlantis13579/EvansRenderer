#pragma once

#include "Renderer/BakedLighting/LightmapAtlas.h"
#include <functional>

namespace Evans::Renderer
{
    inline constexpr std::uint32_t MinimumLightmapRayCount = 4;
    inline constexpr std::uint32_t MaximumLightmapRayCount = 1024;

    struct BakedLightmapSettings
    {
        std::uint32_t RayCount = 32;
        float MaximumDistance = 100;
        float SurfaceBias = 0.001f;
        bool Shadows = true;
        bool BakeBackFaces = true;
    };

    enum class LightmapTexelKind : std::uint8_t { Empty, Surface, Gutter };

    struct BakedLightmap
    {
        std::shared_ptr<const LightmapAtlas> Atlas;
        BakedLightmapSettings Settings;
        // Top-left, row-major RGBA32F. RGB is incident diffuse irradiance / pi,
        // independent of receiver albedo. W is one for valid surface/gutter
        // texels, zero elsewhere. BackIrradiance is empty for front-only bakes.
        // A consumer multiplies by receiver diffuse reflectance, with its PBR
        // dielectric/metallic factor. Only the base level supports filtering;
        // ordinary mip generation would mix unrelated charts.
        std::vector<Float4> FrontIrradiance, BackIrradiance;
        std::vector<LightmapTexelKind> TexelKinds;
        std::uint32_t CoveredTexelCount = 0, GutterTexelCount = 0;
        std::uint64_t SurfaceSampleCount = 0, SurfaceHitCount = 0, ShadowQueryCount = 0;
    };

    // Atlas must come from BuildLightmapAtlas. Transport geometry is separate
    // from receiver geometry so LOD receivers can later share one full-detail
    // lighting scene. Both inputs are immutable snapshots for the entire bake.
    // A true cancellation callback aborts; failure never replaces old output.
    [[nodiscard]] bool BakeLightmap(std::shared_ptr<const LightmapAtlas> atlas,
        const SceneRayQuery& transportScene, const LightingState& lighting,
        const BakedLightmapSettings& settings, BakedLightmap& output, std::string& error,
        const std::function<bool()>& cancelled = {});
}
