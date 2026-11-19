#pragma once

#include "Renderer/Geometry/GeometryPreparation.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/Geometry/TriangleSubdivision.h"
#include "Techniques/Tessellation.h"

namespace Evans::Renderer
{
    inline constexpr unsigned TessellationThreadCount=64;
    inline constexpr std::uint32_t PNLinearEdge=1,PNReverseEdge=1,PNDegeneratePatch=1;
    struct alignas(16) TessellationConstants
    {
        ShaderUInt4 Inputs; // source vertices, patches, attribute edges, geometric edges
        ShaderUInt4 Outputs; // output vertices, output indices, segments, interior samples per patch
        Float4 Shape; // curvature blend, reserved
        Float4 ObjectBounds; // minimum XYZ and reciprocal maximum source extent
    };
    static_assert(sizeof(TessellationConstants)==64 && offsetof(TessellationConstants,ObjectBounds)==48,"PN tessellation ABI requires four vectors");
    struct PNTessellationGeometry
    {
        std::vector<Vertex> Vertices;
        std::vector<ShaderUInt4> Patches; // source corner IDs XYZ, degenerate flag W
        std::vector<ShaderUInt4> PatchEdges; // attribute edge IDs AB, BC, CA
        std::vector<ShaderUInt4> AttributeEdges; // source low/high IDs, geometric edge ID, reversed geometric orientation
        std::vector<ShaderUInt4> GeometricEdges; // canonical source low/high representatives, linear-crease flag, reserved
        Float3 BoundsMinimum,BoundsMaximum;
    };
    struct TessellatedGeometryBuffers
    {
        BufferHandle VertexBuffer,IndexBuffer;
        std::vector<Vertex> Vertices; // authoritative GPU readback of the drawable allocations
        std::vector<std::uint32_t> Indices;
        unsigned DispatchCount=0;
    };
    [[nodiscard]] bool ValidateTessellationUpload(const PNTessellationGeometry& geometry,
        const Techniques::TessellationSettings& settings,TessellationConstants& constants,std::string& error);
    // Vlachos, Peters, Boyd and Mitchell, "Curved PN Triangles", I3D 2001.
    // Cubic position and independent quadratic normal patches use existing
    // positions/normals. Shared geometric edges use canonical controls; an
    // incompatible tangent-plane projection falls back to a straight crease.
    // Attribute seams retain separate vertices and normal/UV interpolation.
    // https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Vlachos01.pdf
    [[nodiscard]] bool PreparePNTessellation(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices,const GeometryPreparationLimits& limits,
        PNTessellationGeometry& output,std::string& error);
    [[nodiscard]] bool BuildTessellationConstants(const PNTessellationGeometry& geometry,
        const Techniques::TessellationSettings& settings,const GeometryPreparationLimits& limits,
        TessellationConstants& output) noexcept;
    [[nodiscard]] bool ValidatePNTessellationGeometry(const PNTessellationGeometry& geometry,std::string& error);
}
