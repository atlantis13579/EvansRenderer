#pragma once

#include "Renderer/BakedLighting/BakedLightmap.h"
#include <cstddef>

namespace Evans::Renderer
{
    inline constexpr std::uint32_t LightmapUnsegmented=std::numeric_limits<std::uint32_t>::max();
    struct LightmapReceiverSegment
    {
        std::uint32_t FirstTriangle=0,TriangleCount=0,Tag=LightmapUnsegmented;
    };
    struct LightmapSourceGeometry
    {
        std::vector<Vertex> Vertices;
        // All endpoint-collapse LOD index lists, including the original at zero.
        // Instances share this immutable snapshot; no per-instance vertex copy.
        std::vector<std::vector<std::uint32_t>> Levels;
        // Optional complete partitions within each receiver level. Virtual
        // clusters share one immutable atlas, then select tagged native ranges.
        // Level zero remains the complete transport surface.
        std::vector<std::vector<LightmapReceiverSegment>> Segments;
    };

    struct LightmapSourceInstance
    {
        std::shared_ptr<const LightmapSourceGeometry> Geometry;
        Matrix4 World;
        Float4 Color = {1,1,1,1};
        MaterialData Material;
        bool Visible = true, CastShadow = true;
    };

    struct LightmapReceiverVertex
    {
        Float3 Position;
        Float2 MaterialUV;
        Float2 AtlasUV;
        Float4 Color;
    };
    static_assert(sizeof(LightmapReceiverVertex) == 44 && offsetof(LightmapReceiverVertex,Color) == 28,
        "Lightmap receiver native vertex ABI changed");

    struct LightmapReceiverRange
    {
        std::uint32_t InstanceIndex = 0, LodLevel = 0;
        std::uint32_t FirstVertex = 0, VertexCount = 0;
        std::uint32_t Tag=LightmapUnsegmented;
    };

    struct LightmapReceivers
    {
        std::shared_ptr<const SceneRayQuery> TransportScene;
        std::shared_ptr<const LightmapAtlas> Atlas;
        std::vector<LightmapReceiverVertex> Vertices;
        std::vector<LightmapReceiverRange> Ranges;
    };

    // Each instance/LOD owns its atlas coordinates. Lighting transport uses
    // original geometry only, so alternate LOD receivers cannot occlude each
    // other. Streams retain original local positions and winding: GPU transforms
    // can match the main raster pass exactly, including mirrored instances.
    // The snapshots must remain immutable until this operation returns.
    [[nodiscard]] bool BuildLightmapReceivers(const std::vector<LightmapSourceInstance>& instances,
        const Techniques::Settings& settings, const LightmapAtlasSettings& atlasSettings,
        LightmapReceivers& output, std::string& error, const std::function<bool()>& cancelled = {});
}
