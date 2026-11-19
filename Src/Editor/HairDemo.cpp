#include "Editor/Demos.hpp"
#include <array>
#include <algorithm>
#include <cmath>

namespace Evans::Editor
{
namespace
{
constexpr float CubeHalfExtent = 0.9f;
constexpr float EdgeRadius = 0.12f;
constexpr unsigned FaceSegments = 16;
constexpr const char* CubeMaterial = "Fur substrate";

std::shared_ptr<const MeshAsset> CreateCube()
{
    // A small bevel joins the growth normals across edges while retaining flat
    // cube faces. Project a face grid onto an inner box offset by EdgeRadius;
    // the offset direction is also the smooth normal of the rounded edge.
    // Each (right, up, normal) frame has right x up = normal.
    struct Face { Float3 Right, Up, Normal; };
    constexpr std::array<Face, 6> faces{{
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}},
        {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},
        {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}},
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},
        {{1, 0, 0}, {0, 0, 1}, {0, -1, 0}}}};
    MeshAsset mesh; mesh.Name = "Fur Cube";
    constexpr float InnerExtent = CubeHalfExtent - EdgeRadius;
    constexpr unsigned RowStride = FaceSegments + 1;
    for (const auto& face : faces)
    {
        const auto first = static_cast<std::uint32_t>(mesh.Vertices.size());
        for (unsigned row = 0; row <= FaceSegments; ++row)
        for (unsigned column = 0; column <= FaceSegments; ++column)
        {
            const float u = float(column) / FaceSegments, v = float(row) / FaceSegments;
            const float right = 2 * u - 1, up = 2 * v - 1;
            Vertex vertex;
            const Float3 point{
                CubeHalfExtent * (face.Normal.X + right * face.Right.X + up * face.Up.X),
                CubeHalfExtent * (face.Normal.Y + right * face.Right.Y + up * face.Up.Y),
                CubeHalfExtent * (face.Normal.Z + right * face.Right.Z + up * face.Up.Z)};
            const Float3 inner{std::clamp(point.X, -InnerExtent, InnerExtent),
                std::clamp(point.Y, -InnerExtent, InnerExtent), std::clamp(point.Z, -InnerExtent, InnerExtent)};
            const Float3 offset{point.X - inner.X, point.Y - inner.Y, point.Z - inner.Z};
            const float inverseLength = 1 / std::sqrt(offset.X * offset.X + offset.Y * offset.Y + offset.Z * offset.Z);
            vertex.Normal = {offset.X * inverseLength, offset.Y * inverseLength, offset.Z * inverseLength};
            vertex.Position = {inner.X + vertex.Normal.X * EdgeRadius,
                inner.Y + vertex.Normal.Y * EdgeRadius, inner.Z + vertex.Normal.Z * EdgeRadius};
            vertex.TexCoord = {u, v};
            mesh.Vertices.push_back(vertex);
        }
        for (unsigned row = 0; row < FaceSegments; ++row)
        for (unsigned column = 0; column < FaceSegments; ++column)
        {
            const auto a = first + row * RowStride + column, b = a + RowStride;
            mesh.Indices.insert(mesh.Indices.end(), {a, a + 1, b + 1, a, b + 1, b});
        }
    }
    mesh.Sections.push_back({CubeMaterial, 0, static_cast<std::uint32_t>(mesh.Indices.size())});
    return std::make_shared<MeshAsset>(std::move(mesh));
}
}

SceneSnapshot DemoCatalog::LoadHair() const
{
    SceneSnapshot scene; scene.Demo = DemoId::FurCube; scene.Revision = 1;
    Material material; material.Name = CubeMaterial;
    // A matte neutral substrate keeps root/tip absorption colors independent
    // of hue, without bright gaps between the roots.
    material.Diffuse = {0.5f, 0.5f, 0.5f, 1};
    scene.Materials.push_back(material);
    scene.Nodes.push_back({1, "Fur Cube", {}, CreateCube()});
    scene.SceneCamera.Position = {3.1f, 2.3f, 4.2f};
    scene.SceneCamera.Target = {0, 0, 0};
    scene.SceneCamera.VerticalFieldOfViewDegrees = 38;
    constexpr float KeyIntensity = 1.8f, FillIntensity = 0.65f, RimIntensity = 1.1f;
    scene.Lights = {
        {LightType::Directional, {}, {-0.5f, -0.85f, -0.65f}, {1, 0.91f, 0.78f, 1}, KeyIntensity, {}, true},
        {LightType::Directional, {}, {0.8f, -0.25f, -0.4f}, {0.65f, 0.8f, 1, 1}, FillIntensity, {}, false},
        {LightType::Directional, {}, {-0.3f, -0.2f, 1}, {1, 0.8f, 0.55f, 1}, RimIntensity, {}, false}};
    return scene;
}

void DemoCatalog::ApplyHairAppearance(const SceneSnapshot& scene, Techniques::Settings& settings)
{
    if (scene.Demo != DemoId::FurCube) return;
    settings = {};
    settings.Pipeline = Techniques::RenderingPipeline::Forward;
    auto& fur = settings.Hair;
    fur.Enabled = true; fur.Animate = true; fur.UseSelectedObject = false;
    // Short unguided fur needs variation and a little bend rather than long,
    // parallel spikes. Eight segments keep the quadratic silhouette smooth.
    // SideFX: https://www.sidefx.com/docs/houdini/fur/workflow.html
    // Babylon's warm short-fur example: https://www.babylonjs.com/Demos/Fur/source/
    // Shell density there is not comparable to this explicit strand count.
    fur.StrandCount = 49152; fur.Segments = 8;
    fur.LengthFraction = 0.12f; fur.RadiusRatio = 0.016f;
    fur.Bend = 0.65f; fur.LengthVariation = 0.35f; fur.RootOffset = 0;
    fur.LongitudinalRoughness = 0.6f; fur.AzimuthalRoughness = 0.65f;
    fur.RootTint = {0.025f, 0.12f, 0.8f}; fur.TipTint = {0.08f, 0.85f, 0.18f};
    settings.NormalMapping = false; settings.Roughness = 0.8f;
    settings.SampleCount = 4;
    // MSAA integrates thin-ribbon coverage; FXAA softens subpixel highlights.
    settings.PostProcessing.AntiAliasing = Techniques::AntiAliasingMode::FXAA;
    settings.PostProcessing.HDR.ToneMapping = Techniques::ToneMappingMode::ACESFitted;
    settings = Techniques::SanitizeSettings(settings);
}
}
