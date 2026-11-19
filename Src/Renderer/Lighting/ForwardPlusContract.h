#pragma once

#include "Renderer/Lighting/ForwardPlus.h"
#include "Renderer/Lighting/LightListContract.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include "Renderer/Shadows/ShadowProjection.h"

#include <limits>

namespace Evans::Renderer
{
    struct ForwardPlusConstants
    {
        Matrix4 ViewProjection; // Explicit matrix rows in every native shader.
        ShaderUInt4 ExtentAndLights; // width, height, sample count, light count
        ShaderUInt4 TileLayout; // tile size, per-list capacity, tile counts X/Y
    };
    struct ForwardPlusGpuBounds
    {
        Float4 CenterRadius;
        ShaderUInt4 Flags; // x: unbounded; other components reserved
    };
    static_assert(sizeof(ForwardPlusConstants) == 96 && offsetof(ForwardPlusConstants, TileLayout) == 80,
        "Forward+ constant layout changed");
    static_assert(sizeof(ForwardPlusGpuBounds) == 32 && offsetof(ForwardPlusGpuBounds, Flags) == 16,
        "Forward+ light bounds layout changed");

    inline bool ValidateForwardPlusGrid(const ForwardPlusGrid& grid, std::string& error)
    {
        ForwardPlusGrid expected;
        if (!BuildForwardPlusGrid(grid.Width, grid.Height, grid.Samples, grid.LightCount,
                {grid.TileSize, grid.TileCapacity}, expected, error)) return false;
        if (expected.TilesX != grid.TilesX || expected.TilesY != grid.TilesY ||
            expected.TileCount != grid.TileCount || expected.IndexCount != grid.IndexCount)
        { error = "Forward+ tile grid does not match its dimensions"; return false; }
        return true;
    }

    // The frame owner retains this exact light snapshot and the assignment
    // camera through every tiled draw. Matching counts alone cannot establish
    // that two different snapshots have the same scene-order index space.
    inline bool BuildForwardPlusSurfaceConstants(const SceneLightCollection& lights,
        LightListSelection selection, const ForwardPlusGrid* grid,
        LightListSurfaceConstants& result, std::string& error)
    {
        result = {}; error.clear();
        if (selection > LightListSelection::Transparent)
        { error = "Forward+ material selection is invalid"; return false; }
        const auto lightCount = static_cast<std::uint32_t>(lights.GetShaderLights().size());
        LightListSurfaceConstants candidate;
        candidate.LightLayout = {lightCount, static_cast<std::uint32_t>(lights.GetShadowLightIndex()),
            static_cast<std::uint32_t>(selection), 0};
        if (selection != LightListSelection::FullScene)
        {
            if (!grid)
            { error = "Forward+ tile shading requires completed light assignment"; return false; }
            if (!ValidateForwardPlusGrid(*grid, error)) return false;
            if (grid->LightCount != lightCount)
            { error = "Forward+ tile and material light counts disagree"; return false; }
            candidate.LightLayout.W = grid->TileCapacity;
            candidate.SurfaceTiles = {grid->TileSize, grid->TilesX, grid->Width, grid->Height};
        }
        result = candidate;
        return true;
    }

    inline bool BuildForwardPlusConstants(const Matrix4& camera, const ForwardPlusGrid& grid,
        ForwardPlusConstants& result, std::string& error)
    {
        result = {}; error.clear();
        if (!ValidateForwardPlusGrid(grid, error)) return false;
        ShadowProjection metric;
        if (!BuildShadowProjection(camera, metric, error)) return false;
        ForwardPlusConstants candidate;
        for (std::size_t index = 0; index < camera.Values.size(); ++index)
        {
            const double normalized = camera.Values[index] / metric.ClipWScale;
            if (!std::isfinite(normalized) || std::abs(normalized) > std::numeric_limits<float>::max())
            { error = "Forward+ normalized camera is not representable"; return false; }
            candidate.ViewProjection.Values[index] = float(normalized);
        }
        candidate.ExtentAndLights = {grid.Width, grid.Height, grid.Samples, grid.LightCount};
        candidate.TileLayout = {grid.TileSize, grid.TileCapacity, grid.TilesX, grid.TilesY};
        result = candidate;
        return true;
    }
    inline bool BuildForwardPlusGpuBounds(const std::vector<ForwardPlusLightBounds>& lights,
        std::vector<ForwardPlusGpuBounds>& result, std::string& error)
    {
        result.clear(); error.clear();
        if (lights.size() > MaximumForwardPlusLights)
        { error = "Forward+ light count exceeds its native storage contract"; return false; }
        for (const auto& light : lights)
            if (!std::isfinite(light.Center.X) || !std::isfinite(light.Center.Y) || !std::isfinite(light.Center.Z) ||
                !std::isfinite(light.Radius) || light.Radius < 0)
            { error = "Forward+ bounds require a finite centre and nonnegative finite radius"; return false; }
        result.reserve(lights.size());
        for (const auto& light : lights)
            result.push_back({{light.Center.X, light.Center.Y, light.Center.Z, light.Radius}, {light.Unbounded ? 1u : 0u, 0, 0, 0}});
        return true;
    }
}
