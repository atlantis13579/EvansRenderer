#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    // CPU geometry preprocessing; every native API consumes the same index lists
    // with its own existing surface shader. No cross-API shader translation is used.
    struct LodBuildSettings
    {
        std::uint32_t MaximumLevels = 4; // Includes the unchanged source level.
        std::uint32_t MinimumTriangles = 32;
        double TriangleRatio = 0.5;
        double MaximumNormalAngleDegrees = 30.0; // Bounds normal attributes and new face-to-corner deviation.
        double MaximumTexCoordDisplacement = 0.05;
        double MaximumColorDisplacement = 1.0 / 255.0; // Includes vertex opacity.
    };

    struct MeshLodLevel
    {
        std::vector<std::uint32_t> Indices;
        // Conservative local-space vertex-correspondence displacement. QEM cost
        // is only a collapse priority and must never be treated as this bound.
        double MaximumDisplacement = 0.0;
    };

    struct MeshLodChain
    {
        std::vector<MeshLodLevel> Levels;
        std::uint32_t LockedVertexCount = 0;
        std::uint32_t CollapseCount = 0;
    };

    // Endpoint-only edge contractions retain source positions and attributes.
    // Boundaries, coincident seam vertices, sharp folds and non-manifold regions
    // are fixed. Unreducible input succeeds with just its original level.
    // Additional fixed vertices preserve seams shared with geometry outside this
    // local region. Invalid input leaves output unchanged. No GPU resource is created.
    [[nodiscard]] bool BuildMeshLods(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices, const LodBuildSettings& settings,
        MeshLodChain& output, std::string& error,
        const std::vector<std::uint32_t>& lockedVertices = {});

    // A preprocessing weld may already approximate a fan of corner normals.
    // These normalized-vector chord radii consume the same angular budget as
    // later contractions; they cannot silently reset the attribute error to zero.
    [[nodiscard]] bool BuildMeshLods(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices,const LodBuildSettings& settings,
        MeshLodChain& output,std::string& error,const std::vector<std::uint32_t>& lockedVertices,
        const std::vector<double>& initialNormalRadii);

    struct LodView
    {
        Matrix4 World;
        Matrix4 ViewProjection;
        Float3 BoundsMinimum;
        Float3 BoundsMaximum;
        std::uint32_t Width = 0;
        std::uint32_t Height = 0;
    };

    // Project a local displacement ball through the full homogeneous transform.
    // Handles nonuniform scale/shear and asymmetric/orthographic projections.
    // A box crossing the eye/near plane or invalid input returns infinity.
    [[nodiscard]] double ProjectLodError(const LodView& view, double localDisplacement);

    // Coarsening uses a stricter threshold; refinement never exceeds the requested
    // pixel budget. Invalid chains/settings/views select the unchanged level.
    [[nodiscard]] std::uint32_t SelectMeshLod(const MeshLodChain& chain, const LodView& view,
        double maximumErrorPixels, double hysteresisFraction, std::uint32_t previousLevel = 0);
}
