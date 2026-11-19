#pragma once

#include "Renderer/Lighting/ForwardPlus.h"

namespace Evans::Renderer
{
    inline constexpr std::uint32_t MaximumClusteredDepthSlices = 64;
    inline constexpr std::uint32_t ClusteredBoundarySlots = MaximumClusteredDepthSlices + 1;
    inline constexpr std::uint32_t MaximumClusteredTileSize = 64;
    inline constexpr std::uint32_t MaximumClusteredListCapacity = 256;
    struct ClusteredForwardParameters
    {
        std::uint32_t TileSize = 32;
        std::uint32_t DepthSlices = 16;
        std::uint32_t ListCapacity = 64;
    };
    struct ClusteredForwardGrid
    {
        std::uint32_t Width = 0, Height = 0, LightCount = 0, TileSize = 0;
        std::uint32_t TilesX = 0, TilesY = 0, DepthSlices = 0, ListCapacity = 0;
        std::uint32_t ClusterCount = 0, IndexCount = 0;
    };
    struct ClusteredForwardHeader
    {
        std::uint32_t Offset = 0, Count = 0, Reserved0 = 0, Reserved1 = 0;
    };
    static_assert(sizeof(ClusteredForwardHeader) == 16, "Clustered light header ABI changed");
    // The same finite-support spheres used by Forward+ remain valid here. Only
    // the spatial partition changes: full-frustum 3D clusters need no depth image,
    // no opaque occupancy mask, and no separate transparent list.
    [[nodiscard]] bool BuildClusteredForwardGrid(std::uint32_t width, std::uint32_t height,
        std::size_t lightCount, const ClusteredForwardParameters& parameters,
        ClusteredForwardGrid& result, std::string& error);
    [[nodiscard]] bool BuildClusteredForwardDepthBoundaries(const Matrix4& camera, std::uint32_t slices,
        std::array<float, ClusteredBoundarySlots>& result, std::string& error);
}
