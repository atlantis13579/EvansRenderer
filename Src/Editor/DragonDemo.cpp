#include "Editor/Demos.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Evans::Editor
{
namespace
{
constexpr const char* DragonMaterials[]{"Jade diffusion", "Glass refraction", "Alpha transparency", "Mirror subject"};
constexpr const char* MirrorMaterial = "Mirror receiver";
constexpr const char* StageMaterialName = "DragonStage";
constexpr float DragonSpacing = 3.4f;
constexpr float FloorHeight = -1.1f;
constexpr float StageMargin = 1.5f;
constexpr float MirrorHeight = FloorHeight + 0.18f;
constexpr unsigned DragonCount = 4;
constexpr SceneNodeId StageNode = DragonCount + 1;
constexpr float MirrorCenter = (DragonCount - 1) * 0.5f * DragonSpacing, HalfMirrorWidth = 1.6f;
constexpr float MirrorBack = -1.5f, MirrorFront = 3.0f, MirrorTolerance = 0.01f;

void Quad(MeshAsset& mesh, Float3 a, Float3 b, Float3 c, Float3 d, Float3 normal, Color color)
{
    const auto first = static_cast<std::uint32_t>(mesh.Vertices.size());
    mesh.Vertices.insert(mesh.Vertices.end(), {
        {a, normal, {0, 0}, color}, {b, normal, {1, 0}, color},
        {c, normal, {1, 1}, color}, {d, normal, {0, 1}, color}});
    mesh.Indices.insert(mesh.Indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
}

std::shared_ptr<const MeshAsset> CreateStage(float halfWidth, float halfDepth)
{
    // World-space checks and colored stripes make transmitted displacement and
    // absorption visible without introducing a texture or a view-dependent grid.
    MeshAsset mesh; mesh.Name = "Optical reference stage";
    constexpr unsigned ChecksPerAxis = 16;
    constexpr Color LightCheck{0.78f, 0.79f, 0.82f, 1};
    constexpr Color DarkCheck{0.16f, 0.19f, 0.24f, 1};
    for (unsigned row = 0; row < ChecksPerAxis; ++row)
    for (unsigned column = 0; column < ChecksPerAxis; ++column)
    {
        const float left = -halfWidth + 2 * halfWidth * column / ChecksPerAxis;
        const float right = -halfWidth + 2 * halfWidth * (column + 1) / ChecksPerAxis;
        const float back = -halfDepth + 2 * halfDepth * row / ChecksPerAxis;
        const float front = -halfDepth + 2 * halfDepth * (row + 1) / ChecksPerAxis;
        Quad(mesh, {left, FloorHeight, front}, {right, FloorHeight, front},
            {right, FloorHeight, back}, {left, FloorHeight, back}, {0, 1, 0},
            (row + column) % 2 ? DarkCheck : LightCheck);
    }
    constexpr Color Stripes[] = {{0.75f, 0.24f, 0.15f, 1}, {0.82f, 0.79f, 0.63f, 1},
        {0.13f, 0.42f, 0.72f, 1}, {0.82f, 0.79f, 0.63f, 1}};
    constexpr unsigned StripeCount = 12;
    constexpr float WallTop = 3.2f;
    for (unsigned stripe = 0; stripe < StripeCount; ++stripe)
    {
        const float left = -halfWidth + 2 * halfWidth * stripe / StripeCount;
        const float right = -halfWidth + 2 * halfWidth * (stripe + 1) / StripeCount;
        Quad(mesh, {left, FloorHeight, -halfDepth}, {right, FloorHeight, -halfDepth},
            {right, WallTop, -halfDepth}, {left, WallTop, -halfDepth}, {0, 0, 1}, Stripes[stripe % 4]);
    }
    mesh.Sections.push_back({StageMaterialName, 0, static_cast<std::uint32_t>(mesh.Indices.size())});
    return std::make_shared<MeshAsset>(std::move(mesh));
}
}

SceneSnapshot DemoCatalog::LoadDragon() const
{
    // The four fixed exhibits share the Stanford dragon geometry. Material
    // overrides route each object to one technique without changing selection.
    // Source: Sebastian Lague's Ray-Tracing, Dragon_80K.obj (87,130 triangles).
    SceneSnapshot scene = LoadModelDemo(DemoId::StanfordDragon, "Models/StanfordDragon/dragon.obj");
    if (scene.Nodes.empty() || !scene.Nodes.front().Mesh) throw std::runtime_error("Dragon source mesh is missing");
    const auto mesh=scene.Nodes.front().Mesh;
    const auto minimum=std::min_element(mesh->Vertices.begin(),mesh->Vertices.end(),
        [](const Vertex& a,const Vertex& b) { return a.Position.Y<b.Position.Y; });
    if (minimum==mesh->Vertices.end()) throw std::runtime_error("Dragon source mesh is empty");
    scene.Nodes.clear(); scene.Materials.clear();
    constexpr Color colors[]{{0.4f,0.82f,0.53f,1},{0.84f,0.95f,1,1},{0.92f,0.32f,0.2f,1},{0.95f,0.7f,0.25f,1}};
    for (unsigned index=0;index<DragonCount;++index)
    {
        Material material; material.Name=DragonMaterials[index]; material.Diffuse=colors[index];
        material.Roughness=index==1?0.12f:0.38f; material.Metallic=0.0f;
        material.SubsurfaceWeight=index==0?1.0f:0.0f;
        material.Refractive=index==1; material.Opacity=index==2?0.35f:1.0f;
        scene.Materials.push_back(material);
        SceneNode node; node.Id=index+1; node.Name=std::to_string(index+1)+". "+DragonMaterials[index];
        node.Mesh=mesh; node.MaterialOverride=material.Name;
        node.LocalTransform.RotationDegrees.Y=90;
        node.LocalTransform.Translation={(float(index)-(DragonCount-1)*0.5f)*DragonSpacing,
            (index==DragonCount-1?MirrorHeight:FloorHeight)-minimum->Position.Y,0};
        scene.Nodes.push_back(std::move(node));
    }
    Material stage; stage.Name=StageMaterialName; stage.Roughness=0.85f; stage.Metallic=0.0f;
    scene.Materials.push_back(stage);
    constexpr float HalfWidth=DragonCount*DragonSpacing*0.5f+StageMargin,HalfDepth=3.5f;
    scene.Nodes.push_back({StageNode,"Reference floor and backdrop",{},CreateStage(HalfWidth,HalfDepth),true,false});

    // Planar reflection belongs to a flat receiver, not the curved dragon.
    // A finite aperture prevents the other exhibits from receiving reflections
    // where their feet intersect this height. The fourth dragon is the subject.
    Material mirror; mirror.Name=MirrorMaterial; mirror.Diffuse={0.85f,0.88f,0.92f,1};
    mirror.Roughness=0.045f; mirror.Metallic=1.0f; scene.Materials.push_back(mirror);
    MeshAsset receiver; receiver.Name="Planar mirror";
    Quad(receiver,{MirrorCenter-HalfMirrorWidth,MirrorHeight,MirrorFront},{MirrorCenter+HalfMirrorWidth,MirrorHeight,MirrorFront},
        {MirrorCenter+HalfMirrorWidth,MirrorHeight,MirrorBack},{MirrorCenter-HalfMirrorWidth,MirrorHeight,MirrorBack},{0,1,0},{});
    receiver.Sections.push_back({MirrorMaterial,0,static_cast<unsigned>(receiver.Indices.size())});
    scene.Nodes.push_back({StageNode+1,"Mirror below dragon 4",{},std::make_shared<MeshAsset>(std::move(receiver)),true,false});
    scene.SceneCamera.Position={0,3.8f,16.5f}; scene.SceneCamera.Target={0,0.2f,0};
    scene.SceneCamera.VerticalFieldOfViewDegrees=35;
    constexpr float KeyIntensity=2.0f,FillIntensity=0.7f;
    scene.Lights={
        {LightType::Directional,{}, {-0.45f,-1,-0.3f},{1,0.95f,0.87f,1},KeyIntensity,{},true},
        {LightType::Directional,{}, {0.7f,-0.2f,0.5f},{0.55f,0.7f,1,1},FillIntensity,{},false}};
    ++scene.Revision; ++scene.CameraRevision; ++scene.LightingRevision;
    return scene;
}

void DemoCatalog::ApplyDragonAppearance(Techniques::Settings& settings)
{
    using namespace Techniques;
    settings={};
    settings.Metallic=0; settings.Roughness=0.4f;
    settings.AlphaBlending=true; settings.Transparency.Mode=TransparencyMode::Sorted;
    // Keep the jade profile ready for the optional Deferred diffusion pass.
    settings.Subsurface.Strength=0.8f; settings.Subsurface.Distance=0.12f;
    settings.Subsurface.ChannelRadius={0.55f,1,0.65f};
    settings.Refraction.Enabled=true; settings.Refraction.MaterialIOR=1.5f; settings.Refraction.Thickness=0.45f;
    settings.Refraction.MaterialTint=true; settings.Refraction.TintDistance=0.5f;
    settings.Refraction.Absorption={0.15f,0.02f,0.08f};
    settings.Reflections.Planar.Enabled=true; settings.Reflections.Planar.Intensity=1;
    settings.Reflections.Planar.Plane={0,1,0,-MirrorHeight};
    settings.Reflections.Planar.ReceiverTolerance=MirrorTolerance;
    settings.Reflections.Planar.BoundedReceiver=true;
    settings.Reflections.Planar.ReceiverMinimum={MirrorCenter-HalfMirrorWidth,MirrorHeight-MirrorTolerance,MirrorBack};
    settings.Reflections.Planar.ReceiverMaximum={MirrorCenter+HalfMirrorWidth,MirrorHeight+MirrorTolerance,MirrorFront};
    settings.Reflections.Planar.ResolutionDivisor=1;
    settings.SampleCount=1; settings.BackFaceCulling=false;
    settings.PostProcessing.HDR.ToneMapping=ToneMappingMode::ACESFitted;
    settings=SanitizeSettings(settings);
}
}
