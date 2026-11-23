#pragma once

#include "Renderer/Types.h"
#include "Maths/TriMesh.h"

namespace Evans::Renderer
{
    struct RaySceneUpload;
    struct RaySceneShadingUpload;
    // Borrowed only during Create. The published query owns transformed geometry
    // and material values; immutable images retain shared ownership.
    struct SceneRayMeshView
    {
        const std::vector<Vertex>* Vertices = nullptr;
        const std::vector<std::uint32_t>* Indices = nullptr;
        Matrix4 World;
        Float4 Color = {1, 1, 1, 1};
        MaterialData Material;
        bool Visible = true;
        bool CastShadow = true;

        SceneRayMeshView() = default;
        explicit SceneRayMeshView(const MeshData& mesh)
            : Vertices(&mesh.Vertices), Indices(&mesh.Indices), World(mesh.WorldTransform),
              Color(mesh.Color), Material(mesh.Material), Visible(mesh.Visible), CastShadow(mesh.CastShadow) {}
    };

    struct SceneRayHit
    {
        Float3 Position;
        Float3 Normal;
        Float3 GeometricNormal; // Outward, including mirrored-transform correction.
        Float3 Diffuse;
        Float3 Emission;
        Float2 TexCoord;
        float Distance = 0;
        float Opacity = 1;
        std::uint32_t MeshIndex = 0;
        std::uint32_t TriangleIndex = 0;
        bool FrontFace = true;
        float Metallic = 0;
    };

    struct SceneRayTriangle
    {
        // The same world-space corner order used by Trace, including mirrored
        // instance correction. Source material UVs remain in Corners.TexCoord.
        std::array<Vertex, 3> Corners;
        std::uint32_t MeshIndex = 0;
        std::uint32_t SourceTriangleIndex = 0;
        bool CastShadow = true;
        // Query corners can swap under a mirrored instance. These mappings
        // recover the original local-space vertex and triangle-corner order
        // without inverting a world transform or guessing at coincident seams.
        std::array<std::uint32_t,3> SourceVertexIndices{};
        std::array<std::uint32_t,3> SourceCorners{};
    };

    // Material-aware CPU geometry queries for precomputation. This is not a new
    // rendering backend or a replacement for the independent native shaders.
    class SceneRayQuery final
    {
    public:
        [[nodiscard]] static std::shared_ptr<const SceneRayQuery> Create(
            const std::vector<SceneRayMeshView>& meshes, const Techniques::Settings& settings, std::string& error);
        [[nodiscard]] bool Trace(Float3 origin, Float3 direction, float minimumDistance,
            float maximumDistance, SceneRayHit& hit, bool shadowCastersOnly = false) const;
        [[nodiscard]] bool GetTriangle(std::uint32_t triangle, SceneRayTriangle& output) const noexcept;
        // Export the immutable material snapshot for an external offline baker.
        [[nodiscard]] bool GetTriangleMaterial(std::uint32_t triangle, MaterialData& material, Float4& color) const;
        // Lightmap texels address a known triangle directly, avoiding a second
        // closest-hit query that could select a coincident layer. Returns false
        // for invalid barycentrics, degenerate geometry or rejected alpha.
        [[nodiscard]] bool SampleSurface(std::uint32_t triangle, Float3 barycentric,
            bool frontFace, SceneRayHit& output) const;
        [[nodiscard]] const Maths::AABB& GetBounds() const noexcept { return Geometry.GetBounds(); }
        [[nodiscard]] std::size_t GetTriangleCount() const noexcept { return Geometry.GetTriangleCount(); }
        [[nodiscard]] std::size_t GetNodeCount() const noexcept { return Geometry.GetAcceleration().GetNodeCount(); }
        [[nodiscard]] bool UsesPBR() const noexcept { return PBR; }
        [[nodiscard]] float GetMetallic() const noexcept { return Metallic; }
        // Reuse the accepted scene's transformed geometry, cutout rules and
        // acceleration for native shader traversal. Failure preserves output.
        [[nodiscard]] bool Export(RaySceneUpload& output,std::string& error) const;
        // Closest-hit effects need actual scene materials, including offscreen
        // surfaces. Publish matched visibility/shading arrays atomically; failure
        // leaves both outputs intact. No camera radiance or authored asset added.
        [[nodiscard]] bool ExportRadiance(RaySceneUpload& visibility,RaySceneShadingUpload& shading,std::string& error) const;

    private:
        struct Mesh
        {
            Float4 Color;
            MaterialData Material;
            std::uint32_t SourceIndex = 0;
            bool CastShadow = true;
            std::uint32_t VertexBase = 0, FirstTriangle = 0;
            bool Mirrored = false;
        };
        SceneRayQuery() = default;
        SceneRayHit ReadHit(std::uint32_t triangle, const Maths::RayTriangleHit& hit, bool shading) const;

        Maths::TriMesh Geometry;
        std::vector<Vertex> Vertices;
        std::vector<Float3> LinearVertexColors;
        std::vector<std::uint32_t> TriangleMeshes;
        std::vector<Mesh> Meshes;
        bool TextureMapping = true, NormalMapping = true, AlphaTesting = true, FlatShading = false, PBR = false;
        float Metallic = 0;
    };
}
