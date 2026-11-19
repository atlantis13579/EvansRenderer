#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    struct RefractionSourceGeometry
    {
        std::vector<Vertex> Vertices;
        // Original indices at zero, followed by endpoint-collapse LODs.
        // Immutable after publication, shared by repeated scene instances.
        std::vector<std::vector<std::uint32_t>> Levels;
    };

    struct RefractionGeometryRange
    {
        std::uint32_t FirstIndex = 0, IndexCount = 0;
    };

    struct RefractionGeometry
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
        // One entry per input source/instance. Shared sources reuse identical
        // ranges; each range addresses absolute indices, with base vertex zero.
        std::vector<std::vector<RefractionGeometryRange>> Sources;
    };

    // Preserve local positions, normal/UV seams, colors and winding exactly.
    // A nearest-layer capture replays selected LODs with each instance's current
    // transform/material. No lightmap UVs, bake results or new asset are needed.
    // Empty scenes/LOD ranges are valid; malformed input clears output.
    [[nodiscard]] bool BuildRefractionGeometry(
        const std::vector<std::shared_ptr<const RefractionSourceGeometry>>& sources,
        RefractionGeometry& output, std::string& error);

    class PreparedRefractionGeometry
    {
    public:
        // Pointer identity is sufficient only because published source snapshots
        // are immutable. Transform/material/visibility edits do not rebuild them.
        [[nodiscard]] bool Prepare(const std::vector<std::shared_ptr<const RefractionSourceGeometry>>& sources, std::string& error);
        [[nodiscard]] const std::shared_ptr<const RefractionGeometry>& GetGeometry() const noexcept { return Geometry; }
        void Reset() noexcept;

    private:
        std::vector<std::shared_ptr<const RefractionSourceGeometry>> Sources;
        std::shared_ptr<const RefractionGeometry> Geometry;
    };
}
