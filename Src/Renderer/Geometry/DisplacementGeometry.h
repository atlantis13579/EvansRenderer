#pragma once

#include "Renderer/Geometry/GeometryPreparation.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Techniques/DisplacementMapping.h"

namespace Evans::Renderer
{
    inline constexpr unsigned DisplacementThreadCount=64;
    inline constexpr unsigned DisplacementVertexWords=sizeof(Vertex)/sizeof(float);
    static_assert(sizeof(Vertex)==48 && offsetof(Vertex,Normal)==12 && offsetof(Vertex,Color)==24 && offsetof(Vertex,TexCoord)==40,
        "Displacement raw storage must match the canonical packed vertex layout");
    struct DisplacementConstants
    {
        Float4 ObjectBounds; // minimum XYZ, reciprocal maximum local extent
        Float4 Shape; // local full height range, tiling, neutral scalar height, reserved
        ShaderUInt4 Counts; // vertices, triangles, resolved projection, reserved
    };
    static_assert(sizeof(DisplacementConstants)==48 && offsetof(DisplacementConstants,Counts)==32,
        "Displacement compute constants require three aligned vectors");
    struct DisplacementGeometry
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
        std::vector<Float4> Directions;
        std::vector<ShaderUInt4> Neighborhoods; // adjacency start/count per source vertex
        std::vector<std::uint32_t> AdjacentTriangles;
        Float3 BoundsMinimum,BoundsMaximum;
    };
    struct DisplacedVertexBuffer
    {
        BufferHandle Buffer;
        std::vector<Vertex> Vertices; // authoritative GPU readback, matching the drawable buffer
        unsigned DispatchCount=0;
    };
    // Exact position welding shares the displacement direction across authored
    // UV/hard-normal splits. Shading adjacency additionally keys the authored
    // normal, so UV seams share normals while hard shading edges remain separate.
    // Coincident disconnected surfaces are also welded; cross-mesh welding and
    // imported scalar height images need a later scene/asset producer.
    // The immutable upload supports two ordered compute dispatches: positions,
    // then area-weighted deformed normals. CSR gathering needs no float atomics.
    [[nodiscard]] bool PrepareDisplacementGeometry(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices,const GeometryPreparationLimits& limits,
        DisplacementGeometry& output,std::string& error);
    [[nodiscard]] bool BuildDisplacementConstants(const DisplacementGeometry& geometry,
        const Techniques::DisplacementMappingSettings& settings,DisplacementConstants& output) noexcept;
    [[nodiscard]] bool ValidateDisplacementUpload(const DisplacementGeometry& geometry,
        const Techniques::DisplacementMappingSettings& settings,DisplacementConstants& constants,std::string& error);
}
