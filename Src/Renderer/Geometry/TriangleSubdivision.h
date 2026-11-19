#pragma once

#include "Renderer/Geometry/GeometryPreparation.h"

namespace Evans::Renderer
{
    inline constexpr unsigned MaximumTriangleSegments=64;
    struct SubdividedTriangles
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
    };
    // Uniform integer barycentric lattice: n segments produce n*n triangles
    // per input patch. Indexed shared edges own a single canonically oriented
    // vertex sequence, so adjacent patches cannot create T-junctions. Authored
    // UV/normal splits remain separate vertices. This CPU topology/reference
    // producer is not a claim of native hull/domain or mesh-stage execution.
    // Failure leaves output unchanged; a budget never silently lowers n.
    [[nodiscard]] bool SubdivideTriangles(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices,unsigned segments,
        const GeometryPreparationLimits& limits,SubdividedTriangles& output,std::string& error);
}
