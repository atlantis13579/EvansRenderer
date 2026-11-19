#pragma once

#include "Renderer/RayTracing/SceneRayQuery.h"
#include <limits>

namespace Evans::Renderer
{
    inline constexpr std::uint32_t InvalidLightmapChart = std::numeric_limits<std::uint32_t>::max();

    struct LightmapAtlasSettings
    {
        std::uint32_t Resolution = 512;
        std::uint32_t Padding = 2;
        float MaximumNormalAngleDegrees = 45;
        // Bounds overlap-check work on large connected planar meshes. Splits
        // add charts but do not discard triangles or change the source mesh.
        std::uint32_t MaximumChartTriangles = 512;
    };

    struct LightmapAtlasTriangle
    {
        std::array<Float2, 3> UV;
        std::uint32_t Chart = InvalidLightmapChart;
    };

    struct LightmapAtlasChart
    {
        std::vector<std::uint32_t> Triangles;
        // Pixel rectangle includes the complete gutter. UVs address texel
        // centers inside it; X/Y use a top-left texture convention.
        std::uint32_t X = 0, Y = 0, Width = 0, Height = 0;
        bool Rotated = false;
    };

    struct LightmapAtlas
    {
        std::shared_ptr<const SceneRayQuery> Scene;
        LightmapAtlasSettings Settings;
        std::vector<LightmapAtlasTriangle> Triangles;
        std::vector<LightmapAtlasChart> Charts;
        double TexelsPerWorldUnit = 0;
        std::uint32_t DegenerateTriangleCount = 0;
    };

    // Generates independent UV2 without modifying material UVs or resource
    // files. Each output triangle addresses SceneRayQuery's immutable corner
    // order; chart seams can later be represented in a separate vertex stream.
    // Failure leaves the previous output intact.
    [[nodiscard]] bool BuildLightmapAtlas(std::shared_ptr<const SceneRayQuery> scene,
        const LightmapAtlasSettings& settings, LightmapAtlas& output, std::string& error);
}
