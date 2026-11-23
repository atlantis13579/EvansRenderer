#pragma once
#include "Techniques/RenderBuffer.h"

#include "Techniques/TechniqueCatalog.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Evans::Renderer { struct GpuImage; struct EnvironmentRadiance; struct DensityVolume; }

namespace Evans::Editor
{
struct BigWorldState;

struct Float2
{
    float X = 0.0f;
    float Y = 0.0f;
};

struct Rect
{
    float X = 0.0f;
    float Y = 0.0f;
    float Width = 0.0f;
    float Height = 0.0f;

    [[nodiscard]] bool Contains(float x, float y) const noexcept
    {
        return x >= X && y >= Y && x < X + Width && y < Y + Height;
    }
};

struct Float3
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
};

struct Color
{
    float R = 1.0f;
    float G = 1.0f;
    float B = 1.0f;
    float A = 1.0f;
};

struct Transform
{
    Float3 Translation;
    Float3 RotationDegrees;
    Float3 Scale{1.0f, 1.0f, 1.0f};
};

struct Vertex
{
    Float3 Position;
    Float3 Normal{0.0f, 1.0f, 0.0f};
    Float2 TexCoord;
    // RGB is authored in display-sRGB; alpha remains linear.
    Color VertexColor;
};

struct MeshSection
{
    std::string MaterialName;
    std::uint32_t FirstIndex = 0;
    std::uint32_t IndexCount = 0;
};

struct MeshAsset
{
    std::string Name;
    std::vector<Vertex> Vertices;
    std::vector<std::uint32_t> Indices;
    std::vector<MeshSection> Sections;
};

struct ImageAsset
{
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    std::vector<std::uint8_t> PixelsRGBA8;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return Width > 0 && Height > 0 &&
               PixelsRGBA8.size() == static_cast<std::size_t>(Width) * Height * 4U;
    }
};

struct Material
{
    std::string Name;
    Color Ambient{0.0f, 0.0f, 0.0f, 1.0f};
    Color Diffuse;
    Color Specular{0.0f, 0.0f, 0.0f, 1.0f};
    Color Emissive{0.0f, 0.0f, 0.0f, 1.0f};
    float Shininess = 0.0f;
    float Opacity = 1.0f;
    std::string DiffuseTexture;
    std::string NormalTexture;
    std::string OpacityTexture;
    std::shared_ptr<const ImageAsset> DiffuseImage;
    std::shared_ptr<const ImageAsset> NormalImage;
    std::shared_ptr<const ImageAsset> OpacityImage;
    // Per-material diffusion coverage; zero leaves ordinary surfaces unchanged.
    float SubsurfaceWeight = 0;
    bool Refractive = false;
    std::optional<float> Roughness, Metallic;
    float DielectricReflectance=0.04f;
    std::string MetallicRoughnessTexture;
    std::shared_ptr<const ImageAsset> MetallicRoughnessImage;
};

using SceneNodeId = std::uint64_t;

struct SceneNode
{
    SceneNodeId Id = 0;
    std::string Name;
    Transform LocalTransform;
    std::shared_ptr<const MeshAsset> Mesh;
    bool Visible = true;
    bool Selectable = true;
    // A node can reuse immutable geometry with a distinct material.
    std::string MaterialOverride;
};

struct Camera
{
    Float3 Position{0.0f, 1.5f, 5.0f};
    Float3 Target{0.0f, 0.5f, 0.0f};
    Float3 Up{0.0f, 1.0f, 0.0f};
    float VerticalFieldOfViewDegrees = 60.0f;
};

enum class LightType
{
    Directional,
    Point,
    Area
};

struct Light
{
    LightType Type = LightType::Directional;
    Float3 Position;
    Float3 Direction{-0.4f, -1.0f, -0.3f};
    Color LightColor;
    float Intensity = 1.0f;
    Float2 AreaSize{1.0f, 1.0f};
    bool CastShadows = true;
    // World-space radius per emitter sample; zero means unbounded.
    float Range = 0.0f;
    bool Enabled = true;
    bool EditorOverride = false;
};

enum class DemoId
{
    Triangle,
    UtahTeapot,
    SponzaAtrium,
    StanfordDragon,
    FurCube,
    BigWorld,
    DisneyCloud=BigWorld, // Legacy serialized demo identifier.
};

struct SceneSnapshot
{
    DemoId Demo = DemoId::Triangle;
    // Geometry/material changes invalidate resident scene resources.
    std::uint64_t Revision = 0;
    // Rigid animation changes transforms without rebuilding local-space meshes.
    std::uint64_t TransformRevision = 0;
    std::uint64_t DynamicRevision = 0;
    // Light edits invalidate illumination while retaining resident geometry.
    std::uint64_t LightingRevision = 0;
    // Camera changes invalidate only rendered pixels, not scene resources.
    std::uint64_t CameraRevision = 0;
    // Scene sample time, independent of render-worker latency and frame rate.
    double SampleTimeSeconds = 0.0;
    Camera SceneCamera;
    std::vector<Light> Lights;
    std::vector<Material> Materials;
    // Swap the complete material library when the editor changes shading models.
    std::vector<Material> AlternateMaterials;
    bool UsingPbrMaterials = false;
    bool UsingDefaultPbrReflection = false;
    std::vector<SceneNode> Nodes;
    std::filesystem::path BakedLightingDirectory;
    std::filesystem::path LightmapDirectory;
    [[nodiscard]] std::filesystem::path ActiveLightmapDirectory() const
    { return LightmapDirectory.empty() ? ActiveBakedLightingDirectory() : LightmapDirectory; }
    std::shared_ptr<const Renderer::EnvironmentRadiance> Environment;
    void UsePbrMaterials(bool enabled)
    {
        if (enabled == UsingPbrMaterials || AlternateMaterials.empty()) return;
        Materials.swap(AlternateMaterials);
        UsingPbrMaterials = enabled;
        ++Revision;
    }
    [[nodiscard]] std::filesystem::path ActiveBakedLightingDirectory() const
    {
        return BakedLightingDirectory;
    }
    std::shared_ptr<const Renderer::DensityVolume> Volume;
    std::shared_ptr<const BigWorldState> World;
    // Source geometry at scene entry, shared by all subsequent frame snapshots.
    std::uint64_t TriangleCountAtLoad = 0;
};

enum class FillMode
{
    Solid,
    Wireframe,
    SolidWireframe
};

struct ViewportSettings
{
    std::uint32_t Width = 1280;
    std::uint32_t Height = 720;
    FillMode Mode = FillMode::Solid;
    Techniques::RenderBuffer Buffer=Techniques::RenderBuffer::Color;
    bool ShadowsEnabled = true;
    bool OutlineEnabled = true;
    bool ShowLights = false;
    bool ContinuousRendering = false;
    bool GpuPresentation = false;
    SceneNodeId SelectedNode = 0;
    float TeapotRotationDegreesPerSecond = 30.0f;
    Evans::Techniques::Settings Techniques;
};

struct Image
{
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    // Top-left RGBA8 with display-sRGB RGB and straight linear alpha.
    std::vector<std::uint8_t> PixelsRGBA8;
    std::shared_ptr<const Image> Overlay;
    std::shared_ptr<const Renderer::GpuImage> GpuColor;
    std::shared_ptr<const Renderer::GpuImage> GpuContext;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return Width > 0 && Height > 0 &&
               (GpuColor || PixelsRGBA8.size() == static_cast<std::size_t>(Width) * Height * 4U);
    }
};

} // namespace Evans::Editor
