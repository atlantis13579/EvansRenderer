#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    // Geometry producers must preflight their complete result before allocating
    // or publishing it. These are budgets, not implicit quality reductions.
    struct GeometryPreparationLimits
    {
        std::size_t MaximumVertices=2'000'000;
        std::size_t MaximumIndices=6'000'000;
    };
    [[nodiscard]] bool ValidateTriangleGeometry(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices,std::string& error);
}
