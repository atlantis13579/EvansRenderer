#include "Editor/Demos.hpp"
#include "Renderer/Reflections/EnvironmentRadiance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Evans::Editor
{
namespace
{

[[nodiscard]] MeshAsset CreateTriangleMesh()
{
    MeshAsset mesh;
    mesh.Name = "Triangle";
    mesh.Vertices = {
        {{-0.9f, -0.65f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f},
         {0.10f, 0.80f, 0.20f, 1.0f}},
        {{0.9f, -0.65f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f},
         {0.10f, 0.35f, 1.0f, 1.0f}},
        {{0.0f, 0.9f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.5f, 0.0f},
         {1.0f, 0.15f, 0.10f, 1.0f}}};
    mesh.Indices = {0, 1, 2};
    mesh.Sections = {{"TriangleMaterial", 0, 3}};
    return mesh;
}

struct Bounds
{
    Float3 Minimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    Float3 Maximum{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    bool Valid = false;
};

[[nodiscard]] Bounds ComputeBounds(const std::vector<MeshAsset>& meshes)
{
    Bounds bounds;
    for (const MeshAsset& mesh : meshes)
    {
        for (const Vertex& vertex : mesh.Vertices)
        {
            bounds.Valid = true;
            bounds.Minimum.X = std::min(bounds.Minimum.X, vertex.Position.X);
            bounds.Minimum.Y = std::min(bounds.Minimum.Y, vertex.Position.Y);
            bounds.Minimum.Z = std::min(bounds.Minimum.Z, vertex.Position.Z);
            bounds.Maximum.X = std::max(bounds.Maximum.X, vertex.Position.X);
            bounds.Maximum.Y = std::max(bounds.Maximum.Y, vertex.Position.Y);
            bounds.Maximum.Z = std::max(bounds.Maximum.Z, vertex.Position.Z);
        }
    }
    return bounds;
}

[[nodiscard]] Transform FrameTransform(const Bounds& bounds, float targetExtent)
{
    if (!bounds.Valid)
    {
        return {};
    }
    const Float3 center{
        (bounds.Minimum.X + bounds.Maximum.X) * 0.5f,
        (bounds.Minimum.Y + bounds.Maximum.Y) * 0.5f,
        (bounds.Minimum.Z + bounds.Maximum.Z) * 0.5f};
    const float extent = std::max({
        bounds.Maximum.X - bounds.Minimum.X,
        bounds.Maximum.Y - bounds.Minimum.Y,
        bounds.Maximum.Z - bounds.Minimum.Z});
    const float scale = extent > 1.0e-6f ? targetExtent / extent : 1.0f;

    Transform transform;
    transform.Scale = {scale, scale, scale};
    transform.Translation = {-center.X * scale, -center.Y * scale, -center.Z * scale};
    return transform;
}

[[nodiscard]] std::filesystem::path ModelPath(DemoId demo)
{
    switch (demo)
    {
    case DemoId::UtahTeapot:
        return "Models/UtahTeapot/teapot.obj";
    case DemoId::SponzaAtrium:
        return "Scenes/Sponza/sponza.obj";
    case DemoId::StanfordDragon:
        return "Models/StanfordDragon/dragon.obj";
    case DemoId::Triangle:
    case DemoId::FurCube:
        break;
    }
    throw std::invalid_argument("The procedural demo does not have an external model path");
}

} // namespace

DemoCatalog::DemoCatalog(std::filesystem::path contentRoot)
    : ContentRoot_(std::move(contentRoot))
{
    constexpr unsigned EnvironmentFaceResolution=128;
    std::string error;
    SkyEnvironment_=Renderer::LoadEnvironmentRadiance(ContentRoot_/"Environments/KloppenheimSky/kloppenheim_06_puresky_1k.hdr",EnvironmentFaceResolution,error);
    if (!SkyEnvironment_) throw std::runtime_error(error);
}

const std::vector<DemoDescriptor>& DemoCatalog::Descriptors()
{
    static const std::vector<DemoDescriptor> descriptors = {
        {DemoId::Triangle, "Triangle", "A colored triangle used to validate the selected RHI."},
        {DemoId::UtahTeapot, "Rotating Utah Teapot", "The classic Utah teapot with continuous rotation."},
        {DemoId::SponzaAtrium, "Sponza Atrium", "A detailed architectural stress scene."},
        {DemoId::StanfordDragon, "Stanford Dragon", "Four fixed exhibits for diffusion, glass, transparency and planar reflection."},
        {DemoId::FurCube, "Fur Cube", "A procedural cube with short fibers and editable fur parameters."},
        {DemoId::BigWorld, "Big World", "Procedural mountains, lakes, forests, weather and real-time volumetric clouds."}};
    return descriptors;
}

bool DemoCatalog::SupportsBackend(DemoId demo,Techniques::Backend backend)
{
    using Techniques::Technique;
    switch (demo)
    {
    case DemoId::FurCube: return Techniques::QuerySupport(Technique::HairFur,backend).Available();
    case DemoId::StanfordDragon: return Techniques::IsSoftwareRaster(backend) || Techniques::QuerySupport(Technique::Refraction,backend).Available();
    case DemoId::BigWorld: return Techniques::QuerySupport(Technique::VolumetricClouds,backend).Available();
    default: return true;
    }
}

SceneSnapshot DemoCatalog::Load(DemoId demo) const
{
    auto scene=demo==DemoId::Triangle ? LoadTriangle() :
        demo==DemoId::StanfordDragon ? LoadDragon() :
        demo==DemoId::FurCube ? LoadHair() :
        IsVolumeDemo(demo) ? LoadVolume(demo) : LoadModelDemo(demo,ModelPath(demo));
    if (demo!=DemoId::BigWorld) scene.Environment=SkyEnvironment_;
    for (const auto& node : scene.Nodes)
        if (node.Visible && node.Mesh) scene.TriangleCountAtLoad += node.Mesh->Indices.size() / 3;
    return scene;
}

void DemoCatalog::Update(
    SceneSnapshot& scene,
    float deltaSeconds,
    float teapotSpeedDegrees) const
{
    if (scene.Demo != DemoId::UtahTeapot || deltaSeconds <= 0.0f)
    {
        return;
    }
    const float delta = deltaSeconds * teapotSpeedDegrees;
    // Paused animation must not invalidate cached frames or schedule new work.
    if (delta == 0.0f) return;
    for (SceneNode& node : scene.Nodes)
    {
        node.LocalTransform.RotationDegrees.Y =
            std::fmod(node.LocalTransform.RotationDegrees.Y + delta, 360.0f);
    }
    ++scene.TransformRevision;
}

SceneSnapshot DemoCatalog::LoadTriangle() const
{
    SceneSnapshot scene;
    scene.Demo = DemoId::Triangle;
    scene.Revision = 1;
    scene.SceneCamera.Position = {0.0f, 0.0f, 3.0f};
    scene.SceneCamera.Target = {0.0f, 0.0f, 0.0f};
    scene.Materials.push_back(
        {"TriangleMaterial", {0.05f, 0.05f, 0.05f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}});
    scene.Nodes.push_back({1, "Triangle", {}, std::make_shared<MeshAsset>(CreateTriangleMesh())});
    scene.Lights.push_back(
        {LightType::Directional, {}, {-0.4f, -1.0f, -0.5f}, {}, 2.0f, {}, true});
    return scene;
}

SceneSnapshot DemoCatalog::LoadModelDemo(
    DemoId demo,
    const std::filesystem::path& relativePath) const
{
    LoadedModel model = FileResourceLoader::LoadObj(ContentRoot_ / relativePath);
    if (model.Geometry.Meshes.empty())
    {
        throw std::runtime_error("The demo resource contains no triangles: " + relativePath.string());
    }

    SceneSnapshot scene;
    scene.Demo = demo;
    scene.Revision = 1;
    scene.Materials = std::move(model.Materials);

    const float targetExtent = demo == DemoId::SponzaAtrium ? 5.0f : 3.0f;
    const Transform frameTransform = FrameTransform(ComputeBounds(model.Geometry.Meshes), targetExtent);
    SceneNodeId nodeId = 1;
    for (MeshAsset& mesh : model.Geometry.Meshes)
    {
        // Bake the common framing transform so rotating multi-object models uses
        // the normalized model center as its pivot in every render adapter.
        for (Vertex& vertex : mesh.Vertices)
        {
            vertex.Position.X = vertex.Position.X * frameTransform.Scale.X +
                                frameTransform.Translation.X;
            vertex.Position.Y = vertex.Position.Y * frameTransform.Scale.Y +
                                frameTransform.Translation.Y;
            vertex.Position.Z = vertex.Position.Z * frameTransform.Scale.Z +
                                frameTransform.Translation.Z;
        }
        SceneNode node;
        node.Id = nodeId++;
        node.Name = mesh.Name.empty() ? "Mesh " + std::to_string(node.Id) : mesh.Name;
        node.Mesh = std::make_shared<MeshAsset>(std::move(mesh));
        scene.Nodes.push_back(std::move(node));
    }

    if (demo == DemoId::SponzaAtrium)
    {
        scene.AlternateMaterials = FileResourceLoader::LoadMaterialLibrary(ContentRoot_ / "Scenes/Sponza/PBR/Sponza.mtl");
        scene.UsePbrMaterials(true);
        ConfigureSponzaLighting(scene);
        scene.BakedLightingDirectory=ContentRoot_/"Scenes/Sponza/Baked";
    }
    else
    {
        // Preserve highlight headroom so the shading-model comparison does not
        // collapse to saturated white on the teapot's white diffuse material.
        constexpr float TeapotDirectionalIntensity = 0.8f;
        scene.SceneCamera.Position = {0.0f, 1.4f, 4.5f};
        scene.SceneCamera.Target = {0.0f, 0.3f, 0.0f};
        scene.Lights.push_back(
            {LightType::Directional, {}, {-0.35f, -1.0f, -0.25f}, {}, TeapotDirectionalIntensity, {}, true});
    }
    return scene;
}

} // namespace Evans::Editor
