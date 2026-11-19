#pragma once

#include "Renderer/Lighting/ClusteredForward.h"
#include "Renderer/Lighting/ForwardPlusContract.h"

namespace Evans::Renderer
{
    inline bool BuildClusteredForwardSurfaceConstants(const SceneLightCollection& lights,
        LightListSelection selection, const ClusteredForwardGrid* grid,
        LightListSurfaceConstants& result, std::string& error)
    {
        result = {}; error.clear();
        if (selection > LightListSelection::Transparent)
        { error = "Clustered material selection is invalid"; return false; }
        const auto count = static_cast<std::uint32_t>(lights.GetShaderLights().size());
        LightListSurfaceConstants candidate;
        candidate.LightLayout = {count, static_cast<std::uint32_t>(lights.GetShadowLightIndex()), 0, 0};
        if (selection != LightListSelection::FullScene)
        {
            if (!grid)
            { error = "Clustered shading requires completed light assignment"; return false; }
            ClusteredForwardGrid expected;
            if (!BuildClusteredForwardGrid(grid->Width, grid->Height, count,
                {grid->TileSize, grid->DepthSlices, grid->ListCapacity}, expected, error)) return false;
            if (grid->LightCount != count || grid->TilesX != expected.TilesX || grid->TilesY != expected.TilesY ||
                grid->ClusterCount != expected.ClusterCount || grid->IndexCount != expected.IndexCount)
            { error = "Clustered material grid disagrees with its scene or dimensions"; return false; }
            candidate.LightLayout.Z = grid->DepthSlices;
            candidate.LightLayout.W = grid->ListCapacity;
            candidate.SurfaceTiles = {grid->TileSize, grid->TilesX, grid->Width, grid->Height};
        }
        result = candidate;
        return true;
    }

    struct ClusteredForwardConstants
    {
        Matrix4 ViewProjection;
        ShaderUInt4 ExtentAndLights; // width, height, complete light count, depth slices
        ShaderUInt4 GridLayout; // tile side, columns, rows, per-cluster capacity
        std::array<Float4, ClusteredBoundarySlots> DepthBoundaries;
    };
    static_assert(sizeof(ClusteredForwardConstants) == 1136 &&
        offsetof(ClusteredForwardConstants, DepthBoundaries) == 96, "Clustered assignment ABI changed");

    // Native header storage begins with 65 uint4 records containing the float
    // boundary bits in X. Assignment writes cluster headers after this prefix.
    // Fragment lookup can therefore use the existing three storage-buffer slots
    // (scene, headers, indices) without growing DX12's root-constant budget.
    [[nodiscard]] inline bool BuildClusteredForwardConstants(const Matrix4& camera,
        const ClusteredForwardGrid& grid, ClusteredForwardConstants& result, std::string& error)
    {
        result = {}; error.clear();
        ClusteredForwardGrid expected;
        if (!BuildClusteredForwardGrid(grid.Width, grid.Height, grid.LightCount,
            {grid.TileSize, grid.DepthSlices, grid.ListCapacity}, expected, error)) return false;
        if (grid.TilesX != expected.TilesX || grid.TilesY != expected.TilesY ||
            grid.ClusterCount != expected.ClusterCount || grid.IndexCount != expected.IndexCount)
        { error = "Clustered light grid does not match its dimensions"; return false; }
        ShadowProjection metric;
        std::array<float, ClusteredBoundarySlots> boundaries;
        if (!BuildShadowProjection(camera, metric, error) ||
            !BuildClusteredForwardDepthBoundaries(camera, grid.DepthSlices, boundaries, error)) return false;
        ClusteredForwardConstants candidate;
        for (std::size_t index = 0; index < camera.Values.size(); ++index)
        {
            const double value = camera.Values[index] / metric.ClipWScale;
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
            { error = "Clustered normalized camera is not representable"; return false; }
            candidate.ViewProjection.Values[index] = float(value);
        }
        candidate.ExtentAndLights = {grid.Width, grid.Height, grid.LightCount, grid.DepthSlices};
        candidate.GridLayout = {grid.TileSize, grid.TilesX, grid.TilesY, grid.ListCapacity};
        for (std::size_t index = 0; index < boundaries.size(); ++index)
            candidate.DepthBoundaries[index] = {boundaries[index], 0, 0, 0};
        result = candidate;
        return true;
    }
}
