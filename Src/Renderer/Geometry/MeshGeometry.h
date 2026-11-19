#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    // Stored on the index buffer and cleared on every write to its host mirror.
    struct IndexedDrawRange
    {
        std::uint32_t FirstIndex = 0;
        std::uint32_t IndexCount = 0;
        int BaseVertex = 0;
        std::size_t VertexBytes = 0;
    };

    // Native decoder-owned streams are admitted through validated GPU prefix
    // state. They need no host index mirror and use base vertex zero. Ordinary
    // buffers retain full per-index bounds checks, including negative offsets.
    [[nodiscard]] bool ValidateIndexedGeometryDraw(const BufferDesc& vertex,const BufferDesc& index,
        const std::vector<std::uint8_t>& indices,bool nativeDecoded,std::uint32_t firstIndex,
        std::uint32_t indexCount,int baseVertex,std::string& error, IndexedDrawRange* lastValidRange = nullptr);

    // Authored blend classification is shared by raster draws and scene queries.
    // A cutout stays a cutout even when alpha testing is disabled.
    [[nodiscard]] bool IsMeshTranslucent(const std::vector<Vertex>& vertices,
        Float4 color, const MaterialData& material) noexcept;
    inline bool IsMeshTranslucent(const MeshData& mesh) noexcept
    { return IsMeshTranslucent(mesh.Vertices, mesh.Color, mesh.Material); }

    // Content identity for immutable geometry, independent of scene-node IDs,
    // transforms, materials and visibility. A hash match always needs equality.
    [[nodiscard]] std::uint64_t HashMeshGeometry(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices) noexcept;
    [[nodiscard]] bool EqualMeshGeometry(const std::vector<Vertex>& leftVertices,
        const std::vector<std::uint32_t>& leftIndices, const std::vector<Vertex>& rightVertices,
        const std::vector<std::uint32_t>& rightIndices) noexcept;
}
