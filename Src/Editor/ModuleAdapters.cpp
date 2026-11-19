#include "Editor/ModuleAdapters.h"
#include "Rasterizer/ParallelExecutor.h"
#include "Maths/ColorSpace.h"
#include "Editor/BigWorld.hpp"
#include "Renderer/BakedLighting/BakedLightingAssets.h"
#include "Renderer/MaterialTexture.h"
#include "Renderer/Reflections/EnvironmentRadiance.h"
#include <iostream>

#include "Maths/EvansMaths.h"
#include "Rasterizer/Rasterizer.hpp"
#include "Rasterizer/SoftwareRasterCore.hpp"
#include "Renderer/DX12/DxrRayTracingRHI.h"
#include "Renderer/EvansRenderer.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include "Renderer/Geometry/MeshGeometry.h"
#include "Renderer/Pipelines/SceneStorageBudget.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Evans::Editor
{
namespace
{

constexpr float Pi = 3.14159265358979323846f;

[[nodiscard]] RenderResult Failed(std::string error)
{
    RenderResult result;
    result.Status = RenderStatus::Failed;
    result.Error = std::move(error);
    return result;
}

[[nodiscard]] RenderResult Cancelled()
{
    RenderResult result;
    result.Status = RenderStatus::Cancelled;
    return result;
}

[[nodiscard]] Maths::Vector3 ToMath(const Float3& value)
{
    return {value.X, value.Y, value.Z};
}

[[nodiscard]] Maths::Transform ToMath(const Transform& value)
{
    const Maths::Vector3 radians(
        value.RotationDegrees.X * Pi / 180.0f,
        value.RotationDegrees.Y * Pi / 180.0f,
        value.RotationDegrees.Z * Pi / 180.0f);
    return {
        ToMath(value.Translation),
        Maths::Quaternion::FromEulerXYZ(radians),
        ToMath(value.Scale)};
}

[[nodiscard]] Maths::Vector3 TransformNormal(
    const Maths::Transform& transform,
    const Float3& source)
{
    const auto safeDivide = [](float value, float scale)
    {
        return std::fabs(scale) > 1.0e-8f ? value / scale : 0.0f;
    };
    const Maths::Vector3 inverseScaled(
        safeDivide(source.X, transform.Scale.x),
        safeDivide(source.Y, transform.Scale.y),
        safeDivide(source.Z, transform.Scale.z));
    const Maths::Vector3 transformed = transform.Rotation.Rotate(inverseScaled).Normalized();
    return transformed.LengthSquared() > 0.0f ? transformed : Maths::Vector3::UnitY();
}

[[nodiscard]] Renderer::Matrix4 ToRendererMatrix(const Maths::Matrix4& matrix)
{
    Renderer::Matrix4 result;
    std::copy(matrix.Data(), matrix.Data() + 16, result.Values.begin());
    return result;
}

[[nodiscard]] Rasterizer::Matrix4 ToRasterMatrix(const Maths::Matrix4& matrix)
{
    Rasterizer::Matrix4 result;
    std::copy(matrix.Data(), matrix.Data() + 16, result.Values.begin());
    return result;
}

[[nodiscard]] Image ToEditorImage(Renderer::FrameCapture& capture)
{
    Image result;
    if (!capture.HasColorOutput() && !capture.GpuColor) return result;
    result.Width = capture.Width;
    result.Height = capture.Height;
    result.PixelsRGBA8 = std::move(capture.ColorRGBA8);
    result.GpuColor = std::move(capture.GpuColor);
    result.GpuContext = std::move(capture.GpuContext);
    return result;
}

[[nodiscard]] const Material* FindMaterial(
    const SceneSnapshot& scene,
    const std::string& name)
{
    const auto found = std::find_if(
        scene.Materials.begin(),
        scene.Materials.end(),
        [&name](const Material& material) { return material.Name == name; });
    return found == scene.Materials.end() ? nullptr : &*found;
}

[[nodiscard]] const Material* MaterialForTriangle(
    const SceneSnapshot& scene,
    const SceneNode& node,
    std::size_t firstIndex)
{
    if (!node.MaterialOverride.empty()) return FindMaterial(scene,node.MaterialOverride);
    const auto& mesh=*node.Mesh;
    for (const MeshSection& section : mesh.Sections)
    {
        const std::size_t begin = section.FirstIndex;
        const std::size_t end = begin + section.IndexCount;
        if (firstIndex >= begin && firstIndex + 2 < end)
        {
            return FindMaterial(scene, section.MaterialName);
        }
    }
    return scene.Materials.empty() ? nullptr : &scene.Materials.front();
}

enum class SampleConversion
{
    Preserve,
    DecodeSRGB
};

[[nodiscard]] Color SampleImage(
    const ImageAsset& image,
    const Float2& texCoord,
    SampleConversion conversion)
{
    if (!image.IsValid())
    {
        return {};
    }
    const auto wrap = [](float value)
    {
        const float wrapped = value - std::floor(value);
        return wrapped < 0.0f ? wrapped + 1.0f : wrapped;
    };
    const float u = wrap(texCoord.X);
    const float v = wrap(texCoord.Y);
    const float x = u * static_cast<float>(image.Width - 1U);
    const float y = (1.0f - v) * static_cast<float>(image.Height - 1U);
    const std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(x));
    const std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(y));
    const std::uint32_t x1 = std::min(x0 + 1U, image.Width - 1U);
    const std::uint32_t y1 = std::min(y0 + 1U, image.Height - 1U);
    const float blendX = x - static_cast<float>(x0);
    const float blendY = y - static_cast<float>(y0);
    const auto channel = [&image, conversion](
        std::uint32_t pixelX,
        std::uint32_t pixelY,
        std::size_t component)
    {
        const std::size_t index =
            (static_cast<std::size_t>(pixelY) * image.Width + pixelX) * 4U + component;
        const float value = static_cast<float>(image.PixelsRGBA8[index]) / 255.0f;
        return component < 3U && conversion == SampleConversion::DecodeSRGB
            ? Maths::SRGBToLinear(value)
            : value;
    };
    const auto bilinear = [&](std::size_t component)
    {
        const float top = channel(x0, y0, component) * (1.0f - blendX) +
                          channel(x1, y0, component) * blendX;
        const float bottom = channel(x0, y1, component) * (1.0f - blendX) +
                             channel(x1, y1, component) * blendX;
        return top * (1.0f - blendY) + bottom * blendY;
    };
    return {bilinear(0), bilinear(1), bilinear(2), bilinear(3)};
}

[[nodiscard]] Color VertexSurfaceLinearColor(
    const Material* material,
    const Vertex& vertex)
{
    Color result{
        Maths::SRGBToLinear(vertex.VertexColor.R),
        Maths::SRGBToLinear(vertex.VertexColor.G),
        Maths::SRGBToLinear(vertex.VertexColor.B),
        vertex.VertexColor.A};
    if (material != nullptr && material->DiffuseImage && material->DiffuseImage->IsValid())
    {
        const Color texture = SampleImage(
            *material->DiffuseImage,
            vertex.TexCoord,
            SampleConversion::DecodeSRGB);
        result.R *= texture.R;
        result.G *= texture.G;
        result.B *= texture.B;
        result.A *= texture.A;
    }
    if (material != nullptr && material->OpacityImage && material->OpacityImage->IsValid())
    {
        const Color opacity = SampleImage(
            *material->OpacityImage,
            vertex.TexCoord,
            SampleConversion::Preserve);
        result.A *= opacity.R;
    }
    return result;
}

[[nodiscard]] Color VertexSurfaceColor(
    const Material* material,
    const Vertex& vertex)
{
    const Color linear = VertexSurfaceLinearColor(material, vertex);
    // Renderer vertex input is display-sRGB and is decoded again by the RHI.
    return {
        Maths::LinearToSRGB(linear.R),
        Maths::LinearToSRGB(linear.G),
        Maths::LinearToSRGB(linear.B),
        linear.A};
}

using RendererImageCache =
    std::unordered_map<std::shared_ptr<const ImageAsset>, std::shared_ptr<const Renderer::ImageData>>;

[[nodiscard]] std::shared_ptr<const Renderer::ImageData> ToRendererImage(
    const std::shared_ptr<const ImageAsset>& source,
    RendererImageCache& cache,
    const CancellationToken& cancellation)
{
    if (!source || !source->IsValid() || cancellation.IsCancellationRequested())
    {
        return {};
    }
    const auto found = cache.find(source);
    if (found != cache.end())
    {
        return found->second;
    }

    auto result = std::make_shared<Renderer::ImageData>();
    result->Width = source->Width;
    result->Height = source->Height;
    result->PixelsRGBA8.resize(source->PixelsRGBA8.size());
    constexpr std::size_t copyBlockSize = 1024U * 1024U;
    for (std::size_t offset = 0; offset < source->PixelsRGBA8.size();
         offset += copyBlockSize)
    {
        if (cancellation.IsCancellationRequested())
        {
            return {};
        }
        const std::size_t count =
            std::min(copyBlockSize, source->PixelsRGBA8.size() - offset);
        std::copy_n(
            source->PixelsRGBA8.begin() + static_cast<std::ptrdiff_t>(offset),
            count,
            result->PixelsRGBA8.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    cache.emplace(source, result);
    return result;
}

[[nodiscard]] Renderer::MaterialData ToRendererMaterial(
    const Material* source,
    RendererImageCache& imageCache,
    const CancellationToken& cancellation)
{
    Renderer::MaterialData result;
    result.BaseColor = {0.72f, 0.76f, 0.82f, 1.0f};
    if (source == nullptr)
    {
        return result;
    }

    result.BaseColor = {
        source->Diffuse.R, source->Diffuse.G, source->Diffuse.B, source->Opacity};
    // MTL exporters commonly emit a zero Ka for physically based assets. The
    // realtime renderer has no indirect-light transport, so use Kd only when
    // Ka carries no authored color signal.
    constexpr float ambientEpsilon = 1.0e-4f;
    const bool hasAmbient =
        std::abs(source->Ambient.R) > ambientEpsilon ||
        std::abs(source->Ambient.G) > ambientEpsilon ||
        std::abs(source->Ambient.B) > ambientEpsilon;
    const Color& ambient = hasAmbient ? source->Ambient : source->Diffuse;
    result.AmbientColor = {
        ambient.R, ambient.G, ambient.B, source->Ambient.A};
    result.SpecularColor = {
        source->Specular.R, source->Specular.G, source->Specular.B, source->Specular.A};
    result.EmissiveColor = {
        source->Emissive.R, source->Emissive.G, source->Emissive.B, source->Emissive.A};
    result.Shininess = std::max(source->Shininess, 1.0f);
    result.SubsurfaceWeight = source->SubsurfaceWeight;
    result.Refractive = source->Refractive;
    result.Roughness = source->Roughness;
    result.Metallic = source->Metallic;
    result.DielectricReflectance=source->DielectricReflectance;
    result.AlphaCutoff =
        source->OpacityImage && source->OpacityImage->IsValid() ? 0.5f : 0.0f;
    result.DiffuseImage = ToRendererImage(
        source->DiffuseImage, imageCache, cancellation);
    result.MetallicRoughnessImage = ToRendererImage(source->MetallicRoughnessImage, imageCache, cancellation);
    result.NormalImage = ToRendererImage(
        source->NormalImage, imageCache, cancellation);
    result.OpacityImage = ToRendererImage(
        source->OpacityImage, imageCache, cancellation);
    return result;
}

[[nodiscard]] Renderer::RenderMode ToRendererMode(FillMode mode)
{
    switch (mode)
    {
    case FillMode::Solid: return Renderer::RenderMode::Solid;
    case FillMode::Wireframe: return Renderer::RenderMode::Wireframe;
    case FillMode::SolidWireframe: return Renderer::RenderMode::SolidWireframe;
    }
    return Renderer::RenderMode::Solid;
}

[[nodiscard]] Rasterizer::FillMode ToRasterMode(FillMode mode)
{
    switch (mode)
    {
    case FillMode::Solid: return Rasterizer::FillMode::Solid;
    case FillMode::Wireframe: return Rasterizer::FillMode::Wireframe;
    case FillMode::SolidWireframe: return Rasterizer::FillMode::SolidWireframe;
    }
    return Rasterizer::FillMode::Solid;
}

[[nodiscard]] Maths::Matrix4 OrthographicBoundsRH(
    float left,
    float right,
    float bottom,
    float top,
    float nearPlane,
    float farPlane)
{
    Maths::Matrix4 result = Maths::Matrix4::Zero();
    result(0, 0) = 2.0f / (right - left);
    result(0, 3) = -(right + left) / (right - left);
    result(1, 1) = 2.0f / (top - bottom);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 2) = 1.0f / (nearPlane - farPlane);
    result(2, 3) = nearPlane / (nearPlane - farPlane);
    result(3, 3) = 1.0f;
    return result;
}

[[nodiscard]] Maths::Matrix4 PerspectiveOffCenterRH(
    float left,
    float right,
    float bottom,
    float top,
    float nearPlane,
    float farPlane)
{
    Maths::Matrix4 result = Maths::Matrix4::Zero();
    result(0, 0) = 2.0f * nearPlane / (right - left);
    result(0, 2) = (right + left) / (right - left);
    result(1, 1) = 2.0f * nearPlane / (top - bottom);
    result(1, 2) = (top + bottom) / (top - bottom);
    result(2, 2) = farPlane / (nearPlane - farPlane);
    result(2, 3) = farPlane * nearPlane / (nearPlane - farPlane);
    result(3, 2) = -1.0f;
    return result;
}

struct ShadowFrame
{
    Maths::Matrix4 ViewProjection;
    bool Valid = false;
    float DepthBias = 0.01f;
    float SlopeScaledBias = 0.0f;
};

using MeshBoundsCache=std::unordered_map<std::shared_ptr<const MeshAsset>,Maths::AABB>;

[[nodiscard]] ShadowFrame BuildShadowFrame(
    const SceneSnapshot& scene,
    bool useRealtimeRendererDirection = false,MeshBoundsCache* cache=nullptr)
{
    Maths::AABB bounds;
    MeshBoundsCache active;
    for (const SceneNode& node : scene.Nodes)
    {
        if (!node.Visible || !node.Mesh) continue;
        const Maths::Transform transform = ToMath(node.LocalTransform);
        if (cache)
        {
            auto [entry,inserted]=active.try_emplace(node.Mesh);
            if (inserted)
            {
                const auto previous=cache->find(node.Mesh);
                if (previous!=cache->end()) entry->second=previous->second;
                else for (const auto& vertex:node.Mesh->Vertices) entry->second.Expand(ToMath(vertex.Position));
            }
            // Transform the eight corners once per instance, including rotated trees.
            const auto& box=entry->second;
            if (box.IsValid()) for (int i=0;i<8;++i)
                bounds.Expand(transform.TransformPoint({(i&1)?box.Max.x:box.Min.x,(i&2)?box.Max.y:box.Min.y,(i&4)?box.Max.z:box.Min.z}));
            continue;
        }
        for (const Vertex& vertex : node.Mesh->Vertices)
        {
            bounds.Expand(transform.TransformPoint(ToMath(vertex.Position)));
        }
    }
    if (cache) cache->swap(active);
    const auto lightIterator = std::find_if(
        scene.Lights.begin(),
        scene.Lights.end(),
        [](const Light& light) { return light.CastShadows; });
    if (!bounds.IsValid() || lightIterator == scene.Lights.end()) return {};

    const Light& light = *lightIterator;
    const Maths::Vector3 center = bounds.Center();
    const Maths::Vector3 sceneExtents = bounds.Extents();
    const float sceneRadius = std::max(sceneExtents.Length(), 1.0f);
    // Legacy realtime backends still use one canonical directional light.
    // DX11 uses the scene light's emission direction so direct lighting and
    // the shadow projection remain in the same coordinate frame.
    Maths::Vector3 direction = useRealtimeRendererDirection
        ? Maths::Vector3{-0.45f, -1.0f, -0.35f}.Normalized()
        : ToMath(light.Direction).Normalized();
    if (!useRealtimeRendererDirection && light.Type == LightType::Point)
    {
        direction = (center - ToMath(light.Position)).Normalized();
    }
    if (direction.LengthSquared() <= 1.0e-8f) direction = {0.0f, -1.0f, 0.0f};
    const bool finiteLight =
        !useRealtimeRendererDirection && light.Type != LightType::Directional;
    const Maths::Vector3 eye = finiteLight
        ? ToMath(light.Position)
        : center - direction * (sceneRadius + 1.0f);
    const Maths::Vector3 up = std::fabs(direction.Dot(Maths::Vector3::UnitY())) > 0.95f
        ? Maths::Vector3::UnitX()
        : Maths::Vector3::UnitY();
    const Maths::Matrix4 view = Maths::Matrix4::LookAtRH(eye, eye + direction, up);
    const std::array<Maths::Vector3, 8> corners = {
        Maths::Vector3{bounds.Min.x, bounds.Min.y, bounds.Min.z},
        Maths::Vector3{bounds.Max.x, bounds.Min.y, bounds.Min.z},
        Maths::Vector3{bounds.Min.x, bounds.Max.y, bounds.Min.z},
        Maths::Vector3{bounds.Max.x, bounds.Max.y, bounds.Min.z},
        Maths::Vector3{bounds.Min.x, bounds.Min.y, bounds.Max.z},
        Maths::Vector3{bounds.Max.x, bounds.Min.y, bounds.Max.z},
        Maths::Vector3{bounds.Min.x, bounds.Max.y, bounds.Max.z},
        Maths::Vector3{bounds.Max.x, bounds.Max.y, bounds.Max.z}};
    constexpr float minimumNearPlane = 0.01f;
    constexpr float minimumDepthPadding = 0.05f;
    if (finiteLight)
    {
        float minimumSlopeX = std::numeric_limits<float>::infinity();
        float maximumSlopeX = -std::numeric_limits<float>::infinity();
        float minimumSlopeY = std::numeric_limits<float>::infinity();
        float maximumSlopeY = -std::numeric_limits<float>::infinity();
        float minimumDepth = std::numeric_limits<float>::infinity();
        float maximumDepth = 0.0f;
        std::size_t frontCornerCount = 0;
        bool hasCornerBehindEmitter = false;
        for (const Maths::Vector3& corner : corners)
        {
            const Maths::Vector3 lightCorner = view.TransformPoint(corner);
            const float depth = -lightCorner.z;
            if (depth <= minimumNearPlane)
            {
                hasCornerBehindEmitter = true;
                continue;
            }
            minimumSlopeX = std::min(minimumSlopeX, lightCorner.x / depth);
            maximumSlopeX = std::max(maximumSlopeX, lightCorner.x / depth);
            minimumSlopeY = std::min(minimumSlopeY, lightCorner.y / depth);
            maximumSlopeY = std::max(maximumSlopeY, lightCorner.y / depth);
            minimumDepth = std::min(minimumDepth, depth);
            maximumDepth = std::max(maximumDepth, depth);
            ++frontCornerCount;
        }
        if (frontCornerCount == 0)
        {
            return {};
        }
        const float slopeWidth = maximumSlopeX - minimumSlopeX;
        const float slopeHeight = maximumSlopeY - minimumSlopeY;
        const float paddingSlopeX = std::max(slopeWidth * 0.02f, 1.0e-4f);
        const float paddingSlopeY = std::max(slopeHeight * 0.02f, 1.0e-4f);
        const float nearPlane = hasCornerBehindEmitter
            ? minimumNearPlane
            : std::max(minimumNearPlane, minimumDepth * 0.95f);
        const float farPlane = std::max(
            nearPlane + 0.1f,
            maximumDepth * 1.05f);
        const Maths::Matrix4 projection = PerspectiveOffCenterRH(
            (minimumSlopeX - paddingSlopeX) * nearPlane,
            (maximumSlopeX + paddingSlopeX) * nearPlane,
            (minimumSlopeY - paddingSlopeY) * nearPlane,
            (maximumSlopeY + paddingSlopeY) * nearPlane,
            nearPlane,
            farPlane);
        // Perspective shadow depths cluster near one. The directional-light
        // bias would erase nearby occluders, including nearby mesh surfaces.
        // Three texel slopes cover both axes of the 3x3 PCF footprint.
        return {projection * view, true, 0.0001f, 3.0f};
    }

    Maths::AABB lightBounds;
    for (const Maths::Vector3& corner : corners)
    {
        lightBounds.Expand(view.TransformPoint(corner));
    }
    if (!lightBounds.IsValid()) return {};

    const float width = lightBounds.Max.x - lightBounds.Min.x;
    const float height = lightBounds.Max.y - lightBounds.Min.y;
    const float depth = lightBounds.Max.z - lightBounds.Min.z;
    const float paddingX = std::max(width * 0.02f, 0.01f);
    const float paddingY = std::max(height * 0.02f, 0.01f);
    const float paddingDepth = std::max(depth * 0.05f, minimumDepthPadding);
    const float nearPlane = std::max(
        minimumNearPlane,
        -lightBounds.Max.z - paddingDepth);
    const float farPlane = std::max(
        nearPlane + 0.1f,
        -lightBounds.Min.z + paddingDepth);
    const Maths::Matrix4 projection = OrthographicBoundsRH(
        lightBounds.Min.x - paddingX,
        lightBounds.Max.x + paddingX,
        lightBounds.Min.y - paddingY,
        lightBounds.Max.y + paddingY,
        nearPlane,
        farPlane);
    return {projection * view, true};
}

[[nodiscard]] Rasterizer::Float4 ToRaster(const Maths::Vector4& value)
{
    return {value.x, value.y, value.z, value.w};
}

using RasterTextureCache =
    std::unordered_map<const void*, std::uint32_t>;

template<class Image>
[[nodiscard]] std::uint32_t AddRasterTexture(
    const std::shared_ptr<const Image>& source,
    Rasterizer::RasterScene& scene,
    RasterTextureCache& cache,
    bool srgb = false)
{
    if (!source || !source->IsValid())
    {
        return Rasterizer::InvalidRasterTextureIndex;
    }
    const auto found = cache.find(source.get());
    if (found != cache.end())
    {
        return found->second;
    }
    if (scene.Textures.size() >= Rasterizer::InvalidRasterTextureIndex)
    {
        return Rasterizer::InvalidRasterTextureIndex;
    }

    const std::uint32_t index =
        static_cast<std::uint32_t>(scene.Textures.size());
    Rasterizer::RasterTexture texture;
    texture.Width = source->Width;
    texture.Height = source->Height;
    texture.SRGB = srgb;
    // The aliasing shared pointer retains the ImageAsset while exposing its
    // immutable pixel vector directly to the synchronous raster backends.
    texture.PixelsRGBA8 =
        std::shared_ptr<const std::vector<std::uint8_t>>(
            source,
            &source->PixelsRGBA8);
    if (!texture.IsValid())
    {
        return Rasterizer::InvalidRasterTextureIndex;
    }
    scene.Textures.push_back(std::move(texture));
    cache.emplace(source.get(), index);
    return index;
}

struct RasterSceneBuild
{
    Rasterizer::RasterScene Scene;
    ShadowFrame Shadow;
    bool WasCancelled = false;
    std::uint32_t FrontLightmap = Rasterizer::InvalidRasterTextureIndex;
    std::uint32_t BackLightmap = Rasterizer::InvalidRasterTextureIndex;
    std::string LightmapStatus;
    std::uint32_t SkyTexture = Rasterizer::InvalidRasterTextureIndex;
    std::uint32_t SkyDiffuseTexture = Rasterizer::InvalidRasterTextureIndex;
};

RasterSceneBuild BuildSponzaRasterScene(const SceneSnapshot&, const ViewportSettings&,
    const CancellationToken&);

[[nodiscard]] bool UpdateRasterProjection(
    Rasterizer::RasterScene& rasterScene,
    const SceneSnapshot& scene,
    const ViewportSettings& settings,
    const CancellationToken& cancellation)
{
    const float aspect = static_cast<float>(settings.Width) / static_cast<float>(settings.Height);
    const Maths::Matrix4 view = Maths::Matrix4::LookAtRH(
        ToMath(scene.SceneCamera.Position),
        ToMath(scene.SceneCamera.Target),
        ToMath(scene.SceneCamera.Up));
    const Maths::Matrix4 projection = Maths::Matrix4::PerspectiveRH(
        scene.SceneCamera.VerticalFieldOfViewDegrees * Pi / 180.0f,
        aspect, 0.05f, 1000.0f);
    const Maths::Matrix4 viewProjection = projection * view;
    for (std::size_t index = 0; index < rasterScene.Triangles.size(); ++index)
    {
        if ((index & 0x3fffU) == 0U && cancellation.IsCancellationRequested()) return false;
        for (Rasterizer::RasterVertex& vertex : rasterScene.Triangles[index].Vertices)
        {
            vertex.ClipPosition = ToRaster(viewProjection * Maths::Vector4(
                vertex.Position.X, vertex.Position.Y, vertex.Position.Z, 1.0f));
        }
    }
    return !cancellation.IsCancellationRequested();
}

[[nodiscard]] RasterSceneBuild BuildRasterScene(
    const SceneSnapshot& scene,
    const ViewportSettings& settings,
    const CancellationToken& cancellation,
    bool clipSpace)
{
    RasterSceneBuild result;
    result.Shadow = clipSpace ? ShadowFrame{} : BuildShadowFrame(scene);

    result.Scene.Triangles.reserve(scene.TriangleCountAtLoad);
    RasterTextureCache textureCache;

    for (const SceneNode& node : scene.Nodes)
    {
        if (cancellation.IsCancellationRequested())
        {
            result.WasCancelled = true;
            return result;
        }
        if (!node.Visible || !node.Mesh) continue;
        const MeshAsset& mesh = *node.Mesh;
        const Maths::Transform transform = ToMath(node.LocalTransform);
        for (std::size_t firstIndex = 0; firstIndex + 2 < mesh.Indices.size(); firstIndex += 3)
        {
            if (firstIndex % (4096U * 3U) == 0 && cancellation.IsCancellationRequested())
            {
                result.WasCancelled = true;
                return result;
            }
            const std::uint32_t indices[3] = {
                mesh.Indices[firstIndex],
                mesh.Indices[firstIndex + 1],
                mesh.Indices[firstIndex + 2]};
            if (indices[0] >= mesh.Vertices.size() ||
                indices[1] >= mesh.Vertices.size() ||
                indices[2] >= mesh.Vertices.size())
            {
                continue;
            }
            const Material* material = MaterialForTriangle(scene, node, firstIndex);
            Rasterizer::RasterTriangle triangle;
            triangle.Selected = settings.SelectedNode == node.Id;
            if (material != nullptr)
            {
                triangle.Material.BaseColor = {
                    material->Diffuse.R, material->Diffuse.G, material->Diffuse.B, material->Opacity};
                triangle.Material.SpecularColor = {
                    material->Specular.R, material->Specular.G, material->Specular.B, 1.0f};
                triangle.Material.EmissiveColor = {
                    material->Emissive.R, material->Emissive.G, material->Emissive.B, 1.0f};
                triangle.Material.Shininess = std::max(material->Shininess, 1.0f);
                triangle.Material.DiffuseTextureIndex = AddRasterTexture(
                    material->DiffuseImage,
                    result.Scene,
                    textureCache, true);
                triangle.Material.NormalTextureIndex = AddRasterTexture(
                    material->NormalImage, result.Scene, textureCache);
                triangle.Material.SubsurfaceWeight = material->SubsurfaceWeight;
                triangle.Material.OpacityTextureIndex = AddRasterTexture(
                    material->OpacityImage,
                    result.Scene,
                    textureCache);
                triangle.Material.AlphaCutoff =
                    triangle.Material.OpacityTextureIndex !=
                            Rasterizer::InvalidRasterTextureIndex
                        ? 0.5f
                        : 0.0f;
            }
            for (std::size_t corner = 0; corner < 3; ++corner)
            {
                const Vertex& source = mesh.Vertices[indices[corner]];
                const Maths::Vector3 worldPosition = transform.TransformPoint(ToMath(source.Position));
                const Maths::Vector3 worldNormal = TransformNormal(transform, source.Normal);
                Rasterizer::RasterVertex& destination = triangle.Vertices[corner];
                destination.Position = {worldPosition.x, worldPosition.y, worldPosition.z};
                destination.Normal = {worldNormal.x, worldNormal.y, worldNormal.z};
                destination.TexCoord = {source.TexCoord.X, source.TexCoord.Y};
                destination.VertexColor = {
                    Maths::SRGBToLinear(source.VertexColor.R),
                    Maths::SRGBToLinear(source.VertexColor.G),
                    Maths::SRGBToLinear(source.VertexColor.B),
                    source.VertexColor.A};
            }
            result.Scene.Triangles.push_back(triangle);
        }
    }
    if (clipSpace)
        result.WasCancelled = !UpdateRasterProjection(result.Scene, scene, settings, cancellation);
    return result;
}

struct RasterSceneCache
{
    Rasterizer::Detail::ParallelExecutor OutputWorkers;
    const RasterSceneBuild& Get(
        const RenderRequest& request,
        const CancellationToken& cancellation,
        bool clipSpace = false)
    {
        const SceneSnapshot& scene = *request.Scene;
        const ViewportSettings& settings = request.Settings;
        // Camera changes only reproject cached world-space vertices. Geometry,
        // materials, and textures survive camera, viewport, and effect changes.
        const bool rebuild = !Valid || Demo != scene.Demo || Revision != scene.Revision ||
            TransformRevision != scene.TransformRevision ||
            ClipSpace != clipSpace || SelectedNode != settings.SelectedNode;
        const bool reproject = clipSpace && (CameraRevision != scene.CameraRevision ||
            Width != settings.Width || Height != settings.Height);
        const bool lightmaps = !clipSpace && scene.Demo == DemoId::SponzaAtrium &&
            Techniques::UsesLightmaps(settings.Techniques.GlobalIllumination);
        if (rebuild || reproject || lightmaps != Lightmaps)
        {
            Valid = false;
            if (rebuild || lightmaps != Lightmaps)
            {
                Built = std::make_shared<RasterSceneBuild>(!clipSpace && scene.Demo == DemoId::SponzaAtrium ?
                    BuildSponzaRasterScene(scene, settings, cancellation) :
                    BuildRasterScene(scene, settings, cancellation, clipSpace));
                MaterialMips = false;
            }
            else
                Built->WasCancelled = !UpdateRasterProjection(Built->Scene, scene, settings, cancellation);
            if (!Built->WasCancelled && !cancellation.IsCancellationRequested())
            {
                Demo = scene.Demo;
                Revision = scene.Revision;
                TransformRevision = scene.TransformRevision;
                CameraRevision = scene.CameraRevision;
                Width = settings.Width;
                Height = settings.Height;
                SelectedNode = settings.SelectedNode;
                ClipSpace = clipSpace;
                Lightmaps = lightmaps;
                Valid = true;
            }
        }
        const bool filter = !clipSpace && settings.Techniques.AnisotropicFiltering;
        if (Valid && (rebuild || filter != MaterialMips))
        {
            Valid = false;
            for (auto& texture : Built->Scene.Textures)
            {
                texture.MipPixelsRGBA8.reset();
                if (!filter || !texture.IsValid() || texture.Irradiance || (texture.Width == 1 && texture.Height == 1)) continue;
                Renderer::TextureDesc desc;
                desc.Width = texture.Width;
                desc.Height = texture.Height;
                desc.ColorSpace = texture.SRGB ? Renderer::TextureColorSpace::SRGB : Renderer::TextureColorSpace::Linear;
                desc.MipLevels = Renderer::MaterialMipLevels(desc);
                const auto mips = Renderer::BuildMaterialMipChain(desc, *texture.PixelsRGBA8);
                texture.MipPixelsRGBA8 = std::make_shared<const std::vector<std::uint8_t>>(
                    mips.begin() + texture.PixelsRGBA8->size(), mips.end());
            }
            MaterialMips = filter;
            Valid = true;
        }
        return *Built;
    }

    bool MaterialMips = false, Lightmaps = false;
    bool Valid = false;
    bool ClipSpace = false;
    DemoId Demo = DemoId::Triangle;
    std::uint64_t Revision = 0;
    std::uint64_t CameraRevision = 0;
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    SceneNodeId SelectedNode = 0;
    std::shared_ptr<RasterSceneBuild> Built;
    std::uint64_t TransformRevision = 0;
};

[[nodiscard]] Image ToEditorImage(const Rasterizer::RenderTarget& target, Rasterizer::Detail::ParallelExecutor& workers)
{
    Image result;
    result.Width = target.Width();
    result.Height = target.Height();
    if (target.GpuColor) { result.GpuColor=target.GpuColor; return result; }
    const auto& colors=target.Colors();
    result.PixelsRGBA8.resize(colors.size()*4);
    constexpr std::size_t blockSize=4096;
    workers.Run(static_cast<std::uint32_t>((colors.size()+blockSize-1)/blockSize),0,[&](std::uint32_t block) {
        const std::size_t first=std::size_t(block)*blockSize,last=std::min(first+blockSize,colors.size());
        for(std::size_t pixel=first;pixel<last;++pixel)
        {
            const auto& color=colors[pixel];
            auto* destination=result.PixelsRGBA8.data()+4*pixel;
            destination[0]=Maths::LinearToSRGB8(color.R);destination[1]=Maths::LinearToSRGB8(color.G);
            destination[2]=Maths::LinearToSRGB8(color.B);destination[3]=Maths::LinearToUNorm8(color.A);
        }
    });
    return result;
}

struct LightSelection
{
    std::array<std::size_t, Renderer::MaximumSceneLights> SceneIndices{};
    std::size_t Count = 0;
    std::int32_t ShadowLightIndex = -1;
};

[[nodiscard]] LightSelection SelectLights(const SceneSnapshot& scene)
{
    LightSelection result;
    result.Count = std::min<std::size_t>(
        scene.Lights.size(), Renderer::MaximumSceneLights);
    for (std::size_t index = 0; index < result.Count; ++index)
    {
        result.SceneIndices[index] = index;
    }

    const auto shadowLight = std::find_if(
        scene.Lights.begin(),
        scene.Lights.end(),
        [](const Light& light) { return light.CastShadows; });
    if (shadowLight == scene.Lights.end()) return result;

    const std::size_t shadowSceneIndex = static_cast<std::size_t>(
        std::distance(scene.Lights.begin(), shadowLight));
    auto selectedShadowLight = std::find(
        result.SceneIndices.begin(),
        result.SceneIndices.begin() + static_cast<std::ptrdiff_t>(result.Count),
        shadowSceneIndex);
    if (selectedShadowLight ==
        result.SceneIndices.begin() + static_cast<std::ptrdiff_t>(result.Count))
    {
        result.SceneIndices[result.Count - 1U] = shadowSceneIndex;
        selectedShadowLight = result.SceneIndices.begin() +
            static_cast<std::ptrdiff_t>(result.Count - 1U);
    }
    result.ShadowLightIndex = static_cast<std::int32_t>(
        std::distance(result.SceneIndices.begin(), selectedShadowLight));
    return result;
}

[[nodiscard]] Maths::Vector3 NormalizedLightDirection(const Light& source)
{
    Maths::Vector3 direction = ToMath(source.Direction).Normalized();
    if (direction.LengthSquared() <= 1.0e-8f)
    {
        direction = {0.0f, -1.0f, 0.0f};
    }
    return direction;
}

[[nodiscard]] Rasterizer::RasterLight ToRasterLight(const Light& source)
{
    Rasterizer::RasterLight result;
    result.Position = {source.Position.X, source.Position.Y, source.Position.Z};
    result.LightColor = {
        source.LightColor.R,
        source.LightColor.G,
        source.LightColor.B,
        source.LightColor.A};
    result.Intensity = source.Enabled ? source.Intensity : 0;
    result.Range = source.Range;
    result.AreaSize = {source.AreaSize.X, source.AreaSize.Y};
    result.CastShadows = source.CastShadows;

    const Maths::Vector3 direction = NormalizedLightDirection(source);
    switch (source.Type)
    {
    case LightType::Directional:
        result.Type = Rasterizer::RasterLightType::Directional;
        result.Direction = {-direction.x, -direction.y, -direction.z};
        break;
    case LightType::Point:
        result.Type = Rasterizer::RasterLightType::Point;
        break;
    case LightType::Area:
        result.Type = Rasterizer::RasterLightType::RectArea;
        result.Direction = {direction.x, direction.y, direction.z};
        break;
    }
    return result;
}

[[nodiscard]] RenderResult RenderRasterImage(
    const RenderRequest& request,
    const CancellationToken& cancellation,
    Rasterizer::SoftwareRasterizer& rasterizer,
    RasterSceneCache& sceneCache)
{
    if (!request.Scene) return Failed("The render request has no scene");
    if (request.Settings.Width == 0 || request.Settings.Height == 0)
    {
        return Failed("The render request has invalid viewport dimensions");
    }
    if (cancellation.IsCancellationRequested()) return Cancelled();

    const RasterSceneBuild& built = sceneCache.Get(request, cancellation);
    if (built.WasCancelled || cancellation.IsCancellationRequested()) return Cancelled();

    try
    {
        Rasterizer::RasterSettings rasterSettings;
        rasterSettings.ImmutableScene = std::shared_ptr<const Rasterizer::RasterScene>(sceneCache.Built, &built.Scene);
        rasterSettings.GpuPresentation = request.Settings.GpuPresentation && rasterizer.SelectedBackend()!=Rasterizer::RasterBackend::CPU;
        rasterSettings.Mode = ToRasterMode(request.Settings.Mode);
        rasterSettings.InputSpace = Rasterizer::VertexInputSpace::Local;
        rasterSettings.DepthRange = Rasterizer::ClipDepthRange::ZeroToOne;
        const LightSelection lightSelection = SelectLights(*request.Scene);
        rasterSettings.Shading = Rasterizer::ShadingModel::Phong;
        rasterSettings.TextureMapping = request.Settings.Techniques.TextureMapping;
        rasterSettings.NormalMapping = request.Settings.Techniques.NormalMapping;
        rasterSettings.AlphaTesting = request.Settings.Techniques.AlphaTesting;
        rasterSettings.AnisotropicFiltering = request.Settings.Techniques.AnisotropicFiltering;
        if (request.Scene->Demo == DemoId::SponzaAtrium)
        {
            const auto& techniques = request.Settings.Techniques;
            const auto& hdr = techniques.PostProcessing.HDR;
            rasterSettings.ACESColor = hdr.ToneMapping == Techniques::ToneMappingMode::ACESFitted;
            rasterSettings.Exposure = std::exp2(hdr.ExposureEV);
            rasterSettings.Contrast = hdr.ColorGrading ? hdr.Contrast : 1;
            rasterSettings.Saturation = hdr.ColorGrading ? hdr.Saturation : 1;
            rasterSettings.FrontLightmap = built.FrontLightmap;
            rasterSettings.BackLightmap = built.BackLightmap;
            rasterSettings.LightmapIntensity = techniques.GlobalIllumination.Intensity;
            // Prebaked irradiance replaces the constant ambient term in the native path.
            if (built.FrontLightmap != Rasterizer::InvalidRasterTextureIndex)
                rasterSettings.AmbientStrength = 0;
            rasterSettings.SkyTexture = built.SkyTexture;
            rasterSettings.SkyDiffuseTexture = built.SkyDiffuseTexture;
            rasterSettings.SkyIntensity = techniques.Reflections.Skybox ? techniques.Reflections.SkyboxIntensity : 0;
            rasterSettings.SkyDiffuseIntensity = 0.96f * techniques.Reflections.DiffuseIBL;
        }
        const auto& diffusion = request.Settings.Techniques.Subsurface;
        if (Techniques::UsesSubsurfaceScattering(diffusion))
        {
            const float amount = diffusion.Strength * std::min(diffusion.Distance * 4.0f, 1.0f);
            rasterSettings.SubsurfaceScattering = {diffusion.ChannelRadius[0] * amount,
                diffusion.ChannelRadius[1] * amount, diffusion.ChannelRadius[2] * amount};
        }
        const float aspect = static_cast<float>(request.Settings.Width) /
                             static_cast<float>(request.Settings.Height);
        const Maths::Matrix4 view = Maths::Matrix4::LookAtRH(
            ToMath(request.Scene->SceneCamera.Position),
            ToMath(request.Scene->SceneCamera.Target),
            ToMath(request.Scene->SceneCamera.Up));
        const Maths::Matrix4 projection = Maths::Matrix4::PerspectiveRH(
            request.Scene->SceneCamera.VerticalFieldOfViewDegrees * Pi / 180.0f,
            aspect,
            0.05f,
            request.Scene->World ? 10000.0f : 1000.0f);
        const ShadowFrame& shadowFrame = built.Shadow;
        rasterSettings.View = ToRasterMatrix(view);
        rasterSettings.Projection = ToRasterMatrix(projection);
        rasterSettings.LightViewProjection = ToRasterMatrix(
            shadowFrame.Valid ? shadowFrame.ViewProjection : Maths::Matrix4::Identity());
        rasterSettings.CameraPosition = {
            request.Scene->SceneCamera.Position.X,
            request.Scene->SceneCamera.Position.Y,
            request.Scene->SceneCamera.Position.Z};
        static_assert(
            Renderer::MaximumSceneLights <= Rasterizer::MaximumRasterLights,
            "Raster light capacity must cover the shared editor light selection");
        for (std::size_t lightIndex = 0; lightIndex < lightSelection.Count; ++lightIndex)
        {
            rasterSettings.Lights[lightIndex] = ToRasterLight(
                request.Scene->Lights[lightSelection.SceneIndices[lightIndex]]);
        }
        rasterSettings.LightCount = static_cast<std::uint32_t>(lightSelection.Count);
        rasterSettings.ShadowLightIndex = lightSelection.ShadowLightIndex;
        // Pure topology does not sample lighting or the shadow map.
        rasterSettings.ShadowsEnabled = request.Settings.ShadowsEnabled && shadowFrame.Valid &&
            request.Settings.Mode != FillMode::Wireframe;
        rasterSettings.AutoGenerateShadowMap = rasterSettings.ShadowsEnabled;
        rasterSettings.ShadowMapSize = std::clamp(
            std::min(request.Settings.Width, request.Settings.Height), 64U, 1024U);
        rasterSettings.ShadowBias = shadowFrame.DepthBias;
        rasterSettings.ShadowSlopeScaledBias = shadowFrame.SlopeScaledBias;
        if (request.Scene->Demo == DemoId::SponzaAtrium)
        {
            rasterSettings.ShadowMapSize = std::clamp(
                std::max(request.Settings.Width, request.Settings.Height), 256U, 2048U);
            rasterSettings.ShadowPCFRadius = request.Settings.Techniques.ShadowFilter.Mode == Techniques::ShadowFilterMode::Hard ?
                0 : request.Settings.Techniques.ShadowFilter.Radius;
            rasterSettings.ShadowBias = 0.00035f;
            rasterSettings.ShadowReceiverSlopeBias = 0.0006f;
        }
        rasterSettings.OutlineEnabled = request.Settings.OutlineEnabled && request.Settings.SelectedNode!=0;
        Rasterizer::RenderTarget target(request.Settings.Width, request.Settings.Height, !rasterSettings.GpuPresentation);
        const Rasterizer::RasterStatus status = rasterizer.Render(
            built.Scene,
            rasterSettings,
            target,
            Rasterizer::RasterCancellationToken(cancellation.SharedState()));
        if (status == Rasterizer::RasterStatus::Cancelled || cancellation.IsCancellationRequested())
        {
            return Cancelled();
        }

        RenderResult result;
        result.Status = RenderStatus::Completed;
        result.Output = ToEditorImage(target,sceneCache.OutputWorkers);
        result.BakedLightingStatus = built.LightmapStatus;
        const auto statistics=rasterizer.LastStatistics();
        result.Geometry.Available=true;
        result.Geometry.RasterWorkerThreads=statistics.WorkerThreads;
        result.Geometry.RasterComputeGroups=statistics.ComputeGroups;
        result.Geometry.RasterComputeInvocations=statistics.ComputeInvocations;
        result.Geometry.RasterPreparationMilliseconds=statistics.TransformMilliseconds;
        result.Geometry.RasterExecutionMilliseconds=statistics.RasterMilliseconds;
        result.Geometry.RasterExecutionPath=statistics.ExecutionPath;
        return result;
    }
    catch (const std::exception& exception)
    {
        return Failed(exception.what());
    }
}

[[nodiscard]] RenderResult ApplyRayPresentationOverlay(
    const RenderRequest& request,
    const CancellationToken& cancellation,
    RasterSceneCache& sceneCache,
    RenderResult base)
{
    if (base.Status != RenderStatus::Completed || !base.Output.IsValid())
    {
        return base;
    }
    const bool needsWireframe = request.Settings.Mode == FillMode::SolidWireframe;
    const bool needsSelection = request.Settings.OutlineEnabled && request.Settings.SelectedNode != 0;
    if (!needsWireframe && !needsSelection)
    {
        return base;
    }

    const RasterSceneBuild& built = sceneCache.Get(request, cancellation, true);
    if (built.WasCancelled || cancellation.IsCancellationRequested()) return Cancelled();
    if (built.Scene.Triangles.empty())
    {
        return base;
    }

    try
    {
        Rasterizer::RasterSettings overlaySettings;
        overlaySettings.Mode = needsWireframe
            ? Rasterizer::FillMode::SolidWireframe
            : Rasterizer::FillMode::Solid;
        overlaySettings.ClearColor = {0.0f, 0.0f, 0.0f, 0.0f};
        overlaySettings.WireframeColor = needsWireframe
            ? Rasterizer::WireframeDisplayLinear
            : Rasterizer::Color{0.0f, 0.0f, 0.0f, 0.0f};
        overlaySettings.OutlineColor = Rasterizer::OutlineDisplayLinear;
        overlaySettings.OutlineEnabled = needsSelection;
        Rasterizer::RenderTarget target(request.Settings.Width, request.Settings.Height);
        target.Clear(overlaySettings.ClearColor);
        Rasterizer::CpuRasterResources resources;
        Rasterizer::RasterStatistics statistics;
        const std::atomic<bool> backendCancellation{false};
        const Rasterizer::RasterStatus status = Rasterizer::SoftwareRasterCore::RenderCpuTiled(
            built.Scene,
            overlaySettings,
            target,
            Rasterizer::RasterCancellationToken(cancellation.SharedState()),
            backendCancellation,
            8,
            resources,
            statistics,
            Rasterizer::CpuRasterPass::PresentationOverlay);
        if (status == Rasterizer::RasterStatus::Cancelled) return Cancelled();
        if (cancellation.IsCancellationRequested()) return Cancelled();

        const Image overlay = ToEditorImage(target,sceneCache.OutputWorkers);
        if(base.Output.GpuColor)
        {
            // Composite the topology mask at presentation without reading ray color.
            base.Output.Overlay=std::make_shared<Image>(overlay);return base;
        }
        for (std::size_t pixel = 0; pixel + 3 < base.Output.PixelsRGBA8.size(); pixel += 4)
        {
            const float alpha = overlay.PixelsRGBA8[pixel + 3] / 255.0f;
            if (alpha <= 0.0f) continue;
            for (std::size_t channel = 0; channel < 3; ++channel)
            {
                const float source = static_cast<float>(overlay.PixelsRGBA8[pixel + channel]);
                const float destination = static_cast<float>(base.Output.PixelsRGBA8[pixel + channel]);
                base.Output.PixelsRGBA8[pixel + channel] = static_cast<std::uint8_t>(
                    std::clamp(source * alpha + destination * (1.0f - alpha), 0.0f, 255.0f));
            }
            base.Output.PixelsRGBA8[pixel + 3] = 255;
        }
        return base;
    }
    catch (const std::exception& exception)
    {
        return Failed(exception.what());
    }
}

[[nodiscard]] Renderer::Vertex ToRendererVertex(const Vertex& source)
{
    Renderer::Vertex result;
    result.Position = {source.Position.X, source.Position.Y, source.Position.Z};
    result.Normal = {source.Normal.X, source.Normal.Y, source.Normal.Z};
    result.Color = {
        source.VertexColor.R,
        source.VertexColor.G,
        source.VertexColor.B,
        source.VertexColor.A};
    result.TexCoord = {source.TexCoord.X, source.TexCoord.Y};
    return result;
}

Renderer::MeshData RendererMeshInstance(const SceneNode& node)
{
    Renderer::MeshData mesh;
    mesh.Id="Node."+std::to_string(node.Id);
    mesh.SelectionGroup=node.Id;
    mesh.Visible=node.Visible;
    mesh.PlanarReceiver=node.Name.find("Water tile ")==0;
    mesh.ReorderForInstancing=node.Name.find("Pine ")==0;
    mesh.Vegetation=node.Name.find("Grass patch ")==0?1:mesh.ReorderForInstancing?2:0;
    const bool lakebed=node.Name.find("Lakebed ")==0,weather=node.Name=="Weather";
    mesh.ManagedLod=mesh.Vegetation!=0 || mesh.PlanarReceiver || lakebed;
    mesh.DynamicVertices=weather || mesh.PlanarReceiver || lakebed;
    mesh.CastShadow=!mesh.PlanarReceiver && !weather;
    mesh.Color={1,1,1,1};
    mesh.WorldTransform=ToRendererMatrix(ToMath(node.LocalTransform).ToMatrix());
    return mesh;
}

[[nodiscard]] bool AppendFullMaterialMesh(
    const SceneNode& node,
    const std::vector<std::size_t>& triangleOrdinals,
    std::size_t sectionOrdinal,
    const Material* material,
    const CancellationToken& cancellation,
    RendererImageCache& imageCache,
    std::vector<std::uint32_t>& vertexRemap,
    std::vector<std::uint32_t>& touchedVertices,
    std::vector<Renderer::MeshData>& result)
{
    const MeshAsset& sourceMesh = *node.Mesh;
    if (triangleOrdinals.empty())
    {
        return true;
    }

    auto mesh=RendererMeshInstance(node);
    mesh.Id+=".Section."+std::to_string(sectionOrdinal);
    mesh.Indices.reserve(triangleOrdinals.size() * 3U);
    mesh.Vertices.reserve(std::min(
        triangleOrdinals.size() * 3U,
        sourceMesh.Vertices.size()));
    touchedVertices.clear();

    for (std::size_t ordinalIndex = 0;
         ordinalIndex < triangleOrdinals.size();
         ++ordinalIndex)
    {
        if ((ordinalIndex & 0x3fffU) == 0U &&
            cancellation.IsCancellationRequested())
        {
            return false;
        }
        const std::size_t triangle = triangleOrdinals[ordinalIndex] * 3U;
        const std::array<std::uint32_t, 3> sourceIndices = {
            sourceMesh.Indices[triangle],
            sourceMesh.Indices[triangle + 1U],
            sourceMesh.Indices[triangle + 2U]};
        if (std::any_of(
                sourceIndices.begin(),
                sourceIndices.end(),
                [&sourceMesh](std::uint32_t index)
                {
                    return index >= sourceMesh.Vertices.size();
                }))
        {
            continue;
        }

        for (const std::uint32_t sourceIndex : sourceIndices)
        {
            std::uint32_t& remapped = vertexRemap[sourceIndex];
            if (remapped == std::numeric_limits<std::uint32_t>::max())
            {
                remapped = static_cast<std::uint32_t>(mesh.Vertices.size());
                mesh.Vertices.push_back(ToRendererVertex(sourceMesh.Vertices[sourceIndex]));
                touchedVertices.push_back(sourceIndex);
            }
            mesh.Indices.push_back(remapped);
        }
    }

    for (const std::uint32_t sourceIndex : touchedVertices)
    {
        vertexRemap[sourceIndex] = std::numeric_limits<std::uint32_t>::max();
    }
    if (mesh.Indices.empty())
    {
        return true;
    }

    mesh.Material = ToRendererMaterial(material, imageCache, cancellation);
    if (cancellation.IsCancellationRequested())
    {
        return false;
    }
    result.push_back(std::move(mesh));
    return true;
}

[[nodiscard]] std::vector<Renderer::MeshData> BuildRendererMeshes(
    const SceneSnapshot& scene,
    const CancellationToken& cancellation,
    bool fullMaterial, RendererImageCache* persistentImages=nullptr)
{
    std::vector<Renderer::MeshData> result;
    std::size_t meshCount = 0;
    for (const SceneNode& node : scene.Nodes)
    {
        if (!node.Mesh || node.Mesh->Vertices.empty() || node.Mesh->Indices.empty()) continue;
        meshCount += fullMaterial
            ? node.Mesh->Sections.size() + 1U
            : 1U;
    }
    result.reserve(meshCount);
    RendererImageCache localImages;
    auto& imageCache=persistentImages?*persistentImages:localImages;
    for (const SceneNode& node : scene.Nodes)
    {
        if (cancellation.IsCancellationRequested()) return {};
        if (!node.Mesh || node.Mesh->Vertices.empty() || node.Mesh->Indices.empty()) continue;

        if (fullMaterial)
        {
            // Generated single-section grids already have native triangle-list
            // topology. Keep their vertex order across deformation updates.
            const auto& src=*node.Mesh;
            if (scene.World && src.Sections.size()==1 && src.Sections.front().FirstIndex==0 &&
                src.Sections.front().IndexCount==src.Indices.size() && src.Indices.size()%3==0 &&
                std::all_of(src.Indices.begin(),src.Indices.end(),[&](auto i){return i<src.Vertices.size();}))
            {
                auto mesh=RendererMeshInstance(node);
                mesh.Id+=".Section.0";
                mesh.Indices=src.Indices;
                mesh.Vertices.reserve(src.Vertices.size());
                std::transform(src.Vertices.begin(),src.Vertices.end(),std::back_inserter(mesh.Vertices),ToRendererVertex);
                mesh.Material=ToRendererMaterial(FindMaterial(scene,node.MaterialOverride.empty()?src.Sections.front().MaterialName:node.MaterialOverride),
                    imageCache,cancellation);
                result.push_back(std::move(mesh));
                continue;
            }
            std::vector<std::uint32_t> vertexRemap(
                node.Mesh->Vertices.size(),
                std::numeric_limits<std::uint32_t>::max());
            std::vector<std::uint32_t> touchedVertices;
            const std::size_t triangleCount = node.Mesh->Indices.size() / 3U;
            const std::size_t fallbackSection = node.Mesh->Sections.size();
            std::vector<std::size_t> triangleSections(
                triangleCount,
                fallbackSection);
            std::vector<bool> triangleAssigned(triangleCount, false);
            for (std::size_t sectionIndex = 0;
                 sectionIndex < node.Mesh->Sections.size();
                 ++sectionIndex)
            {
                const MeshSection& section = node.Mesh->Sections[sectionIndex];
                const std::size_t begin = std::min<std::size_t>(
                    section.FirstIndex,
                    node.Mesh->Indices.size());
                const std::size_t available = std::min<std::size_t>(
                    section.IndexCount,
                    node.Mesh->Indices.size() - begin);
                const std::size_t end = begin + available;
                // Only complete original triangles belong to a malformed,
                // non-aligned range. Earlier sections retain ownership when
                // ranges overlap.
                const std::size_t firstTriangle =
                    begin / 3U + (begin % 3U == 0U ? 0U : 1U);
                const std::size_t onePastLastTriangle = end / 3U;
                for (std::size_t triangle = firstTriangle;
                     triangle < onePastLastTriangle;
                     ++triangle)
                {
                    if ((triangle & 0x3fffU) == 0U &&
                        cancellation.IsCancellationRequested())
                    {
                        return {};
                    }
                    if (!triangleAssigned[triangle])
                    {
                        triangleAssigned[triangle] = true;
                        triangleSections[triangle] = sectionIndex;
                    }
                }
            }

            std::vector<std::vector<std::size_t>> sectionTriangles(
                node.Mesh->Sections.size() + 1U);
            for (std::size_t triangle = 0; triangle < triangleCount; ++triangle)
            {
                if ((triangle & 0x3fffU) == 0U &&
                    cancellation.IsCancellationRequested())
                {
                    return {};
                }
                sectionTriangles[triangleSections[triangle]].push_back(triangle);
            }
            for (std::size_t sectionIndex = 0;
                 sectionIndex < sectionTriangles.size();
                 ++sectionIndex)
            {
                const Material* material = !node.MaterialOverride.empty() ? FindMaterial(scene,node.MaterialOverride) : sectionIndex < node.Mesh->Sections.size()
                    ? FindMaterial(
                        scene,
                        node.Mesh->Sections[sectionIndex].MaterialName)
                    : (scene.Materials.empty() ? nullptr : &scene.Materials.front());
                if (!AppendFullMaterialMesh(
                        node,
                        sectionTriangles[sectionIndex],
                        sectionIndex,
                        material,
                        cancellation,
                        imageCache,
                        vertexRemap,
                        touchedVertices,
                        result))
                {
                    return {};
                }
            }
            continue;
        }

        auto mesh=RendererMeshInstance(node);
        mesh.Indices = node.Mesh->Indices;
        mesh.Vertices.reserve(node.Mesh->Vertices.size());
        constexpr Color defaultColor{0.72f, 0.76f, 0.82f, 1.0f};
        for (const Vertex& vertex : node.Mesh->Vertices)
        {
            if ((mesh.Vertices.size() & 0x3fffU) == 0 &&
                cancellation.IsCancellationRequested())
            {
                return {};
            }
            mesh.Vertices.push_back(ToRendererVertex(vertex));
        }

        // The OBJ loader emits independent vertices for each triangle. Color
        // those vertices from their material section so a single Renderer mesh
        // preserves Sponza multi-material surfaces without splitting
        // the node identifier used by picking and silhouette outlines.
        for (const MeshSection& section : node.Mesh->Sections)
        {
            const Material* material = FindMaterial(scene, node.MaterialOverride.empty()?section.MaterialName:node.MaterialOverride);
            const Color color = material != nullptr ? material->Diffuse : defaultColor;
            const std::size_t sectionBegin = std::min<std::size_t>(
                section.FirstIndex, mesh.Indices.size());
            const std::size_t sectionEnd = sectionBegin + std::min<std::size_t>(
                section.IndexCount, mesh.Indices.size() - sectionBegin);
            for (std::size_t index = sectionBegin; index < sectionEnd; ++index)
            {
                if ((index & 0x3fffU) == 0 && cancellation.IsCancellationRequested())
                {
                    return {};
                }
                const std::uint32_t vertexIndex = mesh.Indices[index];
                if (vertexIndex < mesh.Vertices.size())
                {
                    const Color surfaceColor =
                        VertexSurfaceColor(material, node.Mesh->Vertices[vertexIndex]);
                    mesh.Vertices[vertexIndex].Color = {
                        surfaceColor.R * color.R,
                        surfaceColor.G * color.G,
                        surfaceColor.B * color.B,
                        surfaceColor.A * color.A};
                }
            }
        }
        result.push_back(std::move(mesh));
    }
    return result;
}

std::uint32_t AddRasterRadiance(Rasterizer::RasterScene& scene, unsigned width, unsigned height,
    const std::vector<Renderer::Float4>& linear)
{
    if (linear.empty()) return Rasterizer::InvalidRasterTextureIndex;
    // Preserve HDR irradiance in the packed software texture table. Decode
    // RGBE before bilinear filtering so the exponent never interpolates.
    auto pixels = std::make_shared<std::vector<std::uint8_t>>(linear.size() * 4);
    for (std::size_t i = 0; i < linear.size(); ++i)
    {
        const auto& color = linear[i];
        const float maximum = std::max({color.X, color.Y, color.Z});
        if (color.W <= 0 || maximum <= 1.0e-12f) continue;
        int exponent;
        std::frexp(maximum, &exponent);
        const float scale = std::ldexp(255.0f, -exponent);
        auto* pixel = pixels->data() + 4 * i;
        pixel[0] = static_cast<std::uint8_t>(std::clamp(color.X * scale + 0.5f, 0.0f, 255.0f));
        pixel[1] = static_cast<std::uint8_t>(std::clamp(color.Y * scale + 0.5f, 0.0f, 255.0f));
        pixel[2] = static_cast<std::uint8_t>(std::clamp(color.Z * scale + 0.5f, 0.0f, 255.0f));
        pixel[3] = static_cast<std::uint8_t>(exponent + 128);
    }
    const auto index = static_cast<std::uint32_t>(scene.Textures.size());
    Rasterizer::RasterTexture texture;
    texture.Width = width;
    texture.Height = height;
    texture.PixelsRGBA8 = std::move(pixels);
    texture.Irradiance = true;
    scene.Textures.push_back(std::move(texture));
    return index;
}

std::vector<Renderer::Float4> BuildRasterSky(const Renderer::EnvironmentRadiance& source,
    unsigned width, unsigned height, bool diffuse)
{
    using Maths::Vector3;
    const unsigned footprint = diffuse ? std::min(8U, source.Resolution) : 1U;
    const unsigned side = source.Resolution / footprint;
    std::array<std::vector<Vector3>, Renderer::EnvironmentFaceCount> faces;
    for (unsigned face = 0; face < faces.size(); ++face)
    {
        auto& values = faces[face];
        values.resize(std::size_t(side) * side);
        for (unsigned y = 0; y < side; ++y)
            for (unsigned x = 0; x < side; ++x)
            {
                Vector3 sum{};
                for (unsigned j = 0; j < footprint; ++j)
                    for (unsigned i = 0; i < footprint; ++i)
                    {
                        const auto sample=source.Texel(Renderer::EnvironmentFace(face),x*footprint+i,y*footprint+j);
                        sum += Vector3(sample.X,sample.Y,sample.Z);
                    }
                values[std::size_t(y) * side + x] = sum / float(footprint * footprint);
            }
    }
    auto sample = [&](const Vector3& direction) {
        Renderer::EnvironmentLookup lookup;
        if (!Renderer::LocateEnvironmentDirection({direction.x, direction.y, direction.z}, lookup)) return Vector3{};
        const float u = lookup.UV.X * side - 0.5f, v = lookup.UV.Y * side - 0.5f;
        const int x = int(std::floor(u)), y = int(std::floor(v));
        const auto& pixels = faces[static_cast<unsigned>(lookup.Face)];
        auto texel = [&](int column, int row) {
            return pixels[std::size_t(std::clamp(row, 0, int(side) - 1)) * side + std::clamp(column, 0, int(side) - 1)];
        };
        const float dx = u - x, dy = v - y;
        return (texel(x, y) * (1 - dx) + texel(x + 1, y) * dx) * (1 - dy) +
            (texel(x, y + 1) * (1 - dx) + texel(x + 1, y + 1) * dx) * dy;
    };
    std::vector<Renderer::Float4> result(std::size_t(width) * height);
    for (unsigned y = 0; y < height; ++y)
    {
        const float polar = Pi * (y + 0.5f) / height;
        for (unsigned x = 0; x < width; ++x)
        {
            const float phi = 2 * Pi * ((x + 0.5f) / width - 0.5f);
            const Vector3 direction(std::sin(polar)*std::cos(phi), std::cos(polar), std::sin(polar)*std::sin(phi));
            auto radiance = diffuse ? Vector3{} : sample(direction);
            if (diffuse)
            {
                const Vector3 axis = std::abs(direction.y) < 0.99f ? Vector3(0,1,0) : Vector3(1,0,0);
                const auto tangent = axis.Cross(direction).Normalized();
                const auto bitangent = direction.Cross(tangent);
                for (unsigned ray = 0; ray < 16; ++ray)
                {
                    const float radius = std::sqrt((ray + 0.5f) / 16.0f), azimuth = ray * 2.39996323f;
                    radiance += sample(tangent * (radius * std::cos(azimuth)) + bitangent * (radius * std::sin(azimuth)) +
                        direction * std::sqrt(1.0f - radius * radius)) * (1.0f / 16.0f);
                }
            }
            result[std::size_t(y) * width + x] = {radiance.x, radiance.y, radiance.z, 1};
        }
    }
    return result;
}

RasterSceneBuild BuildSponzaRasterScene(const SceneSnapshot& scene, const ViewportSettings& settings,
    const CancellationToken& cancellation)
{
    RasterSceneBuild result;
    result.Shadow = BuildShadowFrame(scene);
    if (scene.Environment)
    {
        result.SkyTexture = AddRasterRadiance(result.Scene, 512, 256, BuildRasterSky(*scene.Environment, 512, 256, false));
        result.SkyDiffuseTexture = AddRasterRadiance(result.Scene, 128, 64, BuildRasterSky(*scene.Environment, 128, 64, true));
    }
    auto meshes = BuildRendererMeshes(scene, cancellation, true);
    std::sort(meshes.begin(), meshes.end(), [](const auto& a, const auto& b) { return a.Id < b.Id; });
    std::shared_ptr<const Renderer::LightmapBakeResult> atlas;
    if (Techniques::UsesLightmaps(settings.Techniques.GlobalIllumination))
    {
        std::vector<Renderer::LightmapSourceInstance> instances;
        instances.reserve(meshes.size());
        for (const auto& mesh : meshes)
        {
            if (cancellation.IsCancellationRequested()) { result.WasCancelled = true; return result; }
            auto geometry = std::make_shared<Renderer::LightmapSourceGeometry>();
            geometry->Vertices = mesh.Vertices;
            Renderer::MeshLodChain chain;
            std::string error;
            if (!Renderer::BuildMeshLods(mesh.Vertices, mesh.Indices, {}, chain, error))
            {
                result.LightmapStatus = "Failed: " + error;
                break;
            }
            for (auto& level : chain.Levels) geometry->Levels.push_back(std::move(level.Indices));
            instances.push_back({std::move(geometry), mesh.WorldTransform, mesh.Color, mesh.Material, mesh.Visible, mesh.CastShadow});
        }
        if (instances.size() == meshes.size())
        {
            const auto request = Renderer::MakeLightmapBakeRequest(std::move(instances), settings.Techniques, {}, true);
            atlas = Renderer::LoadLightmapAsset(scene.ActiveBakedLightingDirectory() / "irradiance.evlightmap", request);
            result.LightmapStatus = atlas ? "Loaded baked resource" : "Failed: prebaked lightmap does not match the scene";
        }
    }
    std::vector<const Renderer::LightmapReceiverRange*> ranges(meshes.size(), nullptr);
    if (atlas)
    {
        for (const auto& range : atlas->Receivers->Ranges)
            if (range.LodLevel == 0 && range.InstanceIndex < meshes.size() && meshes[range.InstanceIndex].Visible &&
                range.VertexCount == meshes[range.InstanceIndex].Indices.size()) ranges[range.InstanceIndex] = &range;

        if (std::any_of(ranges.begin(), ranges.end(), [](const auto* range) { return range != nullptr; }))
        {
            const auto& bake = *atlas->Irradiance;
            const unsigned side = bake.Atlas->Settings.Resolution;
            result.FrontLightmap = AddRasterRadiance(result.Scene, side, side, bake.FrontIrradiance);
            result.BackLightmap = AddRasterRadiance(result.Scene, side, side, bake.BackIrradiance);
        }
        else result.LightmapStatus = "Failed: prebaked lightmap has no matching receivers";
    }
    RasterTextureCache textures;
    for (std::size_t m = 0; m < meshes.size(); ++m)
    {
        if (cancellation.IsCancellationRequested()) { result.WasCancelled = true; return result; }
        const auto& mesh = meshes[m];
        if (!mesh.Visible) continue;
        const auto& source = mesh.Material;
        const auto color = [](Renderer::Float4 v) { return Rasterizer::Color{v.X, v.Y, v.Z, v.W}; };
        Rasterizer::RasterTriangle triangle;
        triangle.Selected = settings.SelectedNode == mesh.SelectionGroup;
        triangle.Material.BaseColor = color(source.BaseColor);
        triangle.Material.SpecularColor = color(source.SpecularColor);
        triangle.Material.EmissiveColor = color(source.EmissiveColor);
        triangle.Material.Shininess = source.Shininess;
        triangle.Material.AlphaCutoff = source.AlphaCutoff;
        triangle.Material.DiffuseTextureIndex = AddRasterTexture(source.DiffuseImage, result.Scene, textures, true);
        triangle.Material.NormalTextureIndex = AddRasterTexture(source.NormalImage, result.Scene, textures);
        triangle.Material.OpacityTextureIndex = AddRasterTexture(source.OpacityImage, result.Scene, textures);
        triangle.Material.ReceivesLightmap = ranges[m] != nullptr;
        const auto world = ToRasterMatrix(Maths::Matrix4{mesh.WorldTransform.Values});
        for (std::size_t i = 0; i < mesh.Indices.size(); i += 3)
        {
            for (unsigned corner = 0; corner < 3; ++corner)
            {
                const auto& vertex = mesh.Vertices[mesh.Indices[i + corner]];
                auto& out = triangle.Vertices[corner];
                const auto position = world * Rasterizer::Float4{vertex.Position.X, vertex.Position.Y, vertex.Position.Z, 1};
                const auto normal = world * Rasterizer::Float4{vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z, 0};
                out.Position = {position.X, position.Y, position.Z};
                out.Normal = {normal.X, normal.Y, normal.Z};
                out.TexCoord = {vertex.TexCoord.X, vertex.TexCoord.Y};
                out.VertexColor = {vertex.Color.X * mesh.Color.X, vertex.Color.Y * mesh.Color.Y,
                    vertex.Color.Z * mesh.Color.Z, vertex.Color.W * mesh.Color.W};
                if (ranges[m])
                {
                    const auto uv = atlas->Receivers->Vertices[ranges[m]->FirstVertex + i + corner].AtlasUV;
                    out.LightmapUV = {uv.X, uv.Y};
                }
            }
            result.Scene.Triangles.push_back(triangle);
        }
    }
    return result;
}


[[nodiscard]] Renderer::SceneLight ToRendererLight(const Light& source)
{
    Renderer::SceneLight result;
    result.Position = {source.Position.X, source.Position.Y, source.Position.Z};
    result.Color = {
        source.LightColor.R, source.LightColor.G, source.LightColor.B};
    result.Intensity = source.Enabled ? source.Intensity : 0;
    result.Range = source.Range;
    result.AreaSize = {source.AreaSize.X, source.AreaSize.Y};
    result.CastShadows = source.CastShadows;

    const Maths::Vector3 direction = NormalizedLightDirection(source);
    switch (source.Type)
    {
    case LightType::Directional:
        result.Type = Renderer::LightType::Directional;
        result.Direction = {-direction.x, -direction.y, -direction.z};
        break;
    case LightType::Point:
        result.Type = Renderer::LightType::Point;
        break;
    case LightType::Area:
        result.Type = Renderer::LightType::RectArea;
        result.Direction = {direction.x, direction.y, direction.z};
        break;
    }
    return result;
}

struct RealtimeLightCacheKey
{
    DemoId Demo = DemoId::Triangle;
    std::uint64_t Revision = 0;
    std::uint64_t LightingRevision = 0;

    friend bool operator==(const RealtimeLightCacheKey& a, const RealtimeLightCacheKey& b)
    { return a.Demo == b.Demo && a.Revision == b.Revision && a.LightingRevision == b.LightingRevision; }
};

[[nodiscard]] std::shared_ptr<const Renderer::SceneLightCollection> BuildRendererSceneLights(
    const SceneSnapshot& scene, std::string& error)
{
    if (scene.Lights.size() > Renderer::MaximumForwardPlusLights)
    { error = "Scene lights exceed the native light-collection capacity"; return {}; }
    std::vector<Renderer::SceneLight> lights;
    lights.reserve(scene.Lights.size());
    std::int32_t shadowLightIndex = -1;
    for (const auto& light : scene.Lights)
    {
        if (shadowLightIndex < 0 && light.CastShadows) shadowLightIndex = static_cast<std::int32_t>(lights.size());
        lights.push_back(ToRendererLight(light));
    }
    return Renderer::SceneLightCollection::Create(std::move(lights), shadowLightIndex, error);
}

[[nodiscard]] Renderer::LightingState BuildRendererLighting(
    const SceneSnapshot& scene, const std::shared_ptr<const Renderer::SceneLightCollection>& lights)
{
    Renderer::LightingState result;
    result.CameraPosition = {
        scene.SceneCamera.Position.X,
        scene.SceneCamera.Position.Y,
        scene.SceneCamera.Position.Z};
    result.AmbientColor = {1.0f, 1.0f, 1.0f};
    result.AmbientStrength = scene.World ? 0.0f : 0.08f;
    result.SceneLights = lights;
    lights->WriteLegacySubset(result);
    return result;
}

struct RealtimeCacheKey
{
    double AnimationTime=0;
    DemoId Demo = DemoId::Triangle;
    std::uint64_t Revision = 0;
    std::uint64_t CameraRevision = 0;
    std::uint64_t LightingRevision = 0;
    std::uint64_t TransformRevision = 0;
    std::uint64_t HistoryEpoch = 0;
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    FillMode Mode = FillMode::Solid;
    Techniques::RenderBuffer Buffer=Techniques::RenderBuffer::Color;
    bool ShadowsEnabled = false;
    bool OutlineEnabled = false;
    bool GpuPresentation = false;
    SceneNodeId SelectedNode = 0;
    Techniques::Settings Techniques;

    friend bool operator==(const RealtimeCacheKey& left, const RealtimeCacheKey& right)
    {
        return left.AnimationTime==right.AnimationTime && left.Demo == right.Demo &&
               left.Revision == right.Revision &&
               left.CameraRevision == right.CameraRevision &&
               left.LightingRevision == right.LightingRevision &&
               left.TransformRevision == right.TransformRevision &&
               left.HistoryEpoch == right.HistoryEpoch &&
               left.Width == right.Width &&
               left.Height == right.Height &&
               left.Mode == right.Mode && left.Buffer==right.Buffer &&
               left.ShadowsEnabled == right.ShadowsEnabled &&
               left.OutlineEnabled == right.OutlineEnabled && left.GpuPresentation == right.GpuPresentation &&
               left.SelectedNode == right.SelectedNode &&
               left.Techniques == right.Techniques;
    }
};

[[nodiscard]] RealtimeCacheKey MakeCacheKey(const RenderRequest& request)
{
    RealtimeCacheKey key;
    const auto& effects=request.Settings.Techniques;
    if ((Techniques::UsesHairFur(effects.Hair) && effects.Hair.Animate) ||
        (effects.VolumetricFog.Enabled && effects.VolumetricFog.MediumMode==Techniques::VolumeMediumMode::Clouds && effects.VolumetricFog.Clouds.Animate))
        key.AnimationTime=request.Scene->SampleTimeSeconds;
    key.Demo = request.Scene->Demo;
    key.Revision = request.Scene->Revision;
    key.CameraRevision = request.Scene->CameraRevision;
    key.LightingRevision = request.Scene->LightingRevision;
    key.TransformRevision = request.Scene->TransformRevision;
    key.HistoryEpoch = request.HistoryEpoch;
    key.Width = request.Settings.Width;
    key.Height = request.Settings.Height;
    key.Mode = request.Settings.Mode;
    key.Buffer=request.Settings.Buffer;
    key.ShadowsEnabled = request.Settings.ShadowsEnabled;
    key.OutlineEnabled = request.Settings.OutlineEnabled;
    key.GpuPresentation = request.Settings.GpuPresentation;
    key.SelectedNode = request.Settings.SelectedNode;
    key.Techniques = request.Settings.Techniques;
    return key;
}

struct RealtimeSceneCacheKey
{
    DemoId Demo = DemoId::Triangle;
    std::uint64_t Revision = 0, DynamicRevision = 0;
    Renderer::RHIBackend Backend = Renderer::RHIBackend::Vulkan;
    bool FullMaterial = false;

    friend bool operator==(
        const RealtimeSceneCacheKey& left,
        const RealtimeSceneCacheKey& right)
    {
        return left.Demo == right.Demo &&
               left.Revision == right.Revision && left.DynamicRevision == right.DynamicRevision &&
               left.Backend == right.Backend &&
               left.FullMaterial == right.FullMaterial;
    }
};

[[nodiscard]] RealtimeSceneCacheKey MakeSceneCacheKey(
    const RenderRequest& request,
    Renderer::RHIBackend backend,
    bool fullMaterial)
{
    RealtimeSceneCacheKey key;
    key.Demo = request.Scene->Demo;
    key.Revision = request.Scene->Revision;
    key.DynamicRevision=request.Scene->DynamicRevision;
    key.Backend = backend;
    key.FullMaterial = fullMaterial;
    return key;
}

struct RealtimeBackendContext
{
    explicit RealtimeBackendContext(Renderer::RHIBackend backend)
        : Backend(backend)
        , Thread(RendererInstance)
        , ThreadStarted(Thread.Start())
    {
    }

    ~RealtimeBackendContext()
    {
        if (Thread.IsRunning())
        {
            Thread.Enqueue([](Renderer::Renderer& renderer) { renderer.Shutdown(); });
            Thread.Flush();
            Thread.Stop(true);
        }
    }

    const Renderer::RHIBackend Backend;
    Renderer::Renderer RendererInstance;
    Renderer::RHIThread Thread;
    bool ThreadStarted = false;
    std::shared_ptr<std::mutex> RenderMutex=std::make_shared<std::mutex>();
};

class RealtimeRenderAdapter final : public IRenderAdapter
{
public:
    RealtimeRenderAdapter(
        BackendDescriptor descriptor,
        Renderer::RHIBackend backend)
        : Descriptor_(std::move(descriptor)), Context_(backend)
    {
    }

    ~RealtimeRenderAdapter() override = default;

    [[nodiscard]] std::uint32_t QueryMultisampleCounts() const override { return MultisampleCounts_.load(); }
    [[nodiscard]] std::uint32_t QueryWeightedTransparencySampleCounts() const override { return WeightedTransparencyCounts_.load(); }
    [[nodiscard]] bool QueryNaniteHardwareScheduling() const override { return NaniteHardwareDeviceSupport_.load()==DeviceSupportAvailable; }
    [[nodiscard]] bool QueryLumenHardwareRayTracing() const override { return LumenHardwareDeviceSupport_.load()==DeviceSupportAvailable; }
    [[nodiscard]] bool QueryMaterialAnisotropicFiltering() const override { return MaterialAnisotropicFiltering_.load(); }
    [[nodiscard]] std::uint64_t QuerySceneStorageBlockBytes() const override { return SceneStorageBlockBytes_.load(); }

    [[nodiscard]] Techniques::Support QueryTechniqueSupport(Techniques::Technique technique) const override
    {
        const auto implementation = IRenderAdapter::QueryTechniqueSupport(technique);
        if (technique == Techniques::Technique::Hybrid && implementation.Available() && HybridDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed Hybrid rendering's raster and scene-ray inputs."};
        if (technique == Techniques::Technique::VisibilityBuffer && implementation.Available() && VisibilityDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed visibility storage and integer/float targets."};
        if (technique == Techniques::Technique::GPUDriven && implementation.Available() && GpuDrivenDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed GPU instance compaction and indexed indirect drawing."};
        if (technique == Techniques::Technique::ForwardPlus && implementation.Available() && ForwardPlusDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed compute light lists and material storage buffers."};
        if (technique == Techniques::Technique::ClusteredForward && implementation.Available() && ClusteredDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed clustered assignment and material storage buffers."};
        if (technique == Techniques::Technique::TiledDeferred && implementation.Available() && TiledDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed G-buffer targets, tiled assignment and material storage buffers."};
        if (technique == Techniques::Technique::WeightedBlendedOIT && implementation.Available() && QueryWeightedTransparencySampleCounts() == 0)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed blendable float transparency targets."};
        if (technique == Techniques::Technique::SSGI && implementation.Available() && GIDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed floating material, velocity and private GI history targets."};
        if (technique == Techniques::Technique::MeshShaders && implementation.Available() && MeshShaderDeviceSupport_.load()!=DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend,"The device has not confirmed native mesh stages, required limits and runtime shader compilation."};
        if (technique == Techniques::Technique::Nanite && implementation.Available() && NaniteDeviceSupport_.load()!=DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend,"The device has not confirmed Nanite decode, hybrid coverage and visibility materials."};
        if (technique == Techniques::Technique::VirtualGeometry && implementation.Available() && VirtualGeometryDeviceSupport_.load()!=DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend,"The device has not confirmed virtual geometry selection, page caches and drawable decoding."};
        if (technique == Techniques::Technique::Tessellation && implementation.Available() && TessellationDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed PN vertex/index compute, storage ranges and coherent readback."};
        if (technique == Techniques::Technique::DisplacementMapping && implementation.Available() && DisplacementDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed displacement compute, storage ranges and coherent geometry readback."};
        if (technique == Techniques::Technique::ParallaxOcclusionMapping && implementation.Available() && ParallaxDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed POM material blocks, floating attachments and position invariance."};
        if (technique == Techniques::Technique::TemporalEffects && implementation.Available() && ExposureDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed HDR and renderable FP32 metering targets."};
        if (technique == Techniques::Technique::TSR && implementation.Available() && TemporalSuperResolutionDeviceSupport_.load()!=DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend,"The device has not confirmed TSR compute, fragment storage and HDR image support."};
        if (technique == Techniques::Technique::VirtualShadowMaps && implementation.Available() && VirtualShadowDeviceSupport_.load()!=DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend,"The device has not confirmed virtual shadow demand compute, storage and native page sampling."};
        if (technique == Techniques::Technique::TAA && implementation.Available() && TemporalDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed renderable color/depth history targets."};
        if (technique == Techniques::Technique::VolumetricFog && implementation.Available() && FogDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed floating fog targets and texture bindings."};
        if ((technique == Techniques::Technique::VolumetricLighting || technique == Techniques::Technique::ParticipatingMedia ||
            technique == Techniques::Technique::VolumetricClouds) && implementation.Available() &&
            (VolumeLightDeviceSupport_.load() != DeviceSupportAvailable || FogDeviceSupport_.load() != DeviceSupportAvailable))
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed FP32 volume transport targets."};
        if (technique == Techniques::Technique::MSAA && implementation.Available() && QueryMultisampleCounts() == 1)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed compatible multisample attachments."};
        if (technique == Techniques::Technique::VSM && implementation.Available() && VSMDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed floating shadow moment targets."};
        if (technique == Techniques::Technique::ContactShadows && implementation.Available() && ContactDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed contact-shadow compute and visibility storage."};
        if (technique == Techniques::Technique::PathTracing && implementation.Available() && PathDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed FP32 path targets and complete scene storage."};
        if (technique==Techniques::Technique::Lumen && implementation.Available() && LumenDeviceSupport_.load()!=DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend,"Lumen compute/material capabilities are pending or unavailable on this device."};
        if (technique == Techniques::Technique::VoxelGI && implementation.Available() && VoxelDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed sampled FP32 voxel storage and its shader/texture interface."};
        if (technique == Techniques::Technique::RayTracedGI && implementation.Available() && RayGIDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed diffuse ray scene/material storage and sampling limits."};
        if (technique == Techniques::Technique::RayTracedReflections && implementation.Available() && RayReflectionDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed ray reflection scene/material storage and sampling limits."};
        if (technique == Techniques::Technique::RayTracedAO && implementation.Available() && RayAODeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed four fragment scene storage buffers."};
        if (technique == Techniques::Technique::OIT && implementation.Available() && PeelingDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed floating OIT targets and invariant vertex positions."};
        if (technique == Techniques::Technique::LightShafts && implementation.Available() && LightShaftDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "The device has not confirmed floating color render/blit targets for light shafts."};
        if ((technique != Techniques::Technique::Deferred && technique != Techniques::Technique::SSAO &&
            technique != Techniques::Technique::HBAO && technique != Techniques::Technique::SAO &&
            technique != Techniques::Technique::GTAO && technique != Techniques::Technique::RayTracedAO && technique != Techniques::Technique::SSR &&
            technique != Techniques::Technique::Lumen && technique != Techniques::Technique::VoxelGI && technique != Techniques::Technique::RayTracedReflections && technique != Techniques::Technique::RayTracedGI &&
            technique != Techniques::Technique::CubemapReflections && technique != Techniques::Technique::EnvironmentReflections &&
            technique != Techniques::Technique::PlanarReflections && technique != Techniques::Technique::BakedGI &&
            technique != Techniques::Technique::LightProbes && technique != Techniques::Technique::Lightmaps &&
            technique != Techniques::Technique::Refraction && technique != Techniques::Technique::SubsurfaceScattering && technique != Techniques::Technique::HairFur) ||
            !implementation.Available()) return implementation;
        const int device = DeferredDeviceSupport_.load();
        if (device == DeviceSupportUnknown)
            return {Techniques::Availability::UnsupportedBackend, "Waiting for the renderer's device capability query."};
        if (device == DeviceSupportUnavailable)
            return {Techniques::Availability::UnsupportedBackend, "This device lacks the floating material targets required by Deferred, AO and reflections."};
        if (technique == Techniques::Technique::SAO && SaoDeviceSupport_.load() != DeviceSupportAvailable)
            return {Techniques::Availability::UnsupportedBackend, "This device has not confirmed sampled/renderable R32F depth hierarchy support."};
        return implementation;
    }

    [[nodiscard]] BackendDescriptor Descriptor() const override
    {
        return Descriptor_;
    }

    [[nodiscard]] RenderResult Render(
        const RenderRequest& request,
        const CancellationToken& cancellation) override
    {
        if (!request.Scene) return Failed("The realtime render request has no scene");
        if (cancellation.IsCancellationRequested()) return Cancelled();
        const RealtimeCacheKey key = MakeCacheKey(request);
        {
            std::lock_guard<std::mutex> lock(CacheMutex_);
            if (!request.Settings.ContinuousRendering && CacheValid_ && key == CacheKey_ && !CachedResult_.NeedsTemporalFrame) return CachedResult_;
        }
        if (!Context_.ThreadStarted || !Context_.Thread.IsRunning())
        {
            return Failed("The Renderer RHI thread could not be started");
        }
        std::lock_guard<std::mutex> renderLock(*Context_.RenderMutex);
        struct ReleaseContext
        {
            RealtimeBackendContext& Context;
            ~ReleaseContext()
            {
                if(Context.Backend!=Renderer::RHIBackend::OpenGLES)return;
                try
                {
                    auto done=std::make_shared<std::promise<void>>();auto ready=done->get_future();
                    if(Context.Thread.Enqueue([done](Renderer::Renderer& renderer){renderer.ReleasePresentationContext();done->set_value();}))ready.get();
                }
                catch(...) { /* Teardown still owns the native context if the queue has stopped. */ }
            }
        } releaseContext{Context_};
        if (cancellation.IsCancellationRequested()) return Cancelled();
        const Renderer::RHIBackend backend = Context_.Backend;
        // Camera and rigid-object motion do not change scene-light contents.
        // The queued RHI work captures shared immutable storage, not a vector.
        const RealtimeLightCacheKey lightKey {request.Scene->Demo, request.Scene->Revision, request.Scene->LightingRevision};
        if (!SceneLights_ || !(lightKey == LightCacheKey_))
        {
            std::string error;
            auto lights = BuildRendererSceneLights(*request.Scene, error);
            if (!lights) return Failed(error);
            SceneLights_ = std::move(lights);
            LightCacheKey_ = lightKey;
        }
        const bool fullMaterial =
            backend == Renderer::RHIBackend::DX11 ||
            backend == Renderer::RHIBackend::DX12 ||
            backend == Renderer::RHIBackend::OpenGLES ||
            backend == Renderer::RHIBackend::Vulkan ||
            backend == Renderer::RHIBackend::Metal;
        const RealtimeSceneCacheKey sceneKey =
            MakeSceneCacheKey(request, backend, fullMaterial);
        if (!request.Scene->World) WorldShadowBounds_.clear();

        struct ConfigureResult
        {
            bool Success = false;
            bool Cancelled = false;
            bool SceneInvalidated = false;
            int DeferredDeviceSupport = DeviceSupportUnknown;
            int SaoDeviceSupport = DeviceSupportUnknown;
            int TemporalDeviceSupport = DeviceSupportUnknown;
            int TemporalSuperResolutionDeviceSupport=DeviceSupportUnknown;
            int VirtualShadowDeviceSupport=DeviceSupportUnknown;
            int ExposureDeviceSupport = DeviceSupportUnknown;
            int ParallaxDeviceSupport = DeviceSupportUnknown;
            int DisplacementDeviceSupport = DeviceSupportUnknown;
            int TessellationDeviceSupport = DeviceSupportUnknown;
            int VirtualGeometryDeviceSupport = DeviceSupportUnknown;
            int NaniteDeviceSupport = DeviceSupportUnknown;
            int MeshShaderDeviceSupport = DeviceSupportUnknown;
            std::uint64_t SceneStorageBlockBytes = 0;
            bool MaterialAnisotropicFiltering = false;
            int NaniteHardwareDeviceSupport=DeviceSupportUnknown;
            int LumenHardwareDeviceSupport=DeviceSupportUnknown;
            int GIDeviceSupport = DeviceSupportUnknown;
            int FogDeviceSupport = DeviceSupportUnknown;
            int VolumeLightDeviceSupport = DeviceSupportUnknown;
            int LightShaftDeviceSupport = DeviceSupportUnknown;
            int PeelingDeviceSupport = DeviceSupportUnknown;
            int RayAODeviceSupport = DeviceSupportUnknown;
            int RayReflectionDeviceSupport = DeviceSupportUnknown;
            int RayGIDeviceSupport = DeviceSupportUnknown;
            int VoxelDeviceSupport = DeviceSupportUnknown;
            int LumenDeviceSupport=DeviceSupportUnknown;
            int PathDeviceSupport = DeviceSupportUnknown;
            int GpuDrivenDeviceSupport = DeviceSupportUnknown;
            int VisibilityDeviceSupport = DeviceSupportUnknown;
            int HybridDeviceSupport = DeviceSupportUnknown;
            int VSMDeviceSupport = DeviceSupportUnknown;
            int ContactDeviceSupport = DeviceSupportUnknown;
            int ForwardPlusDeviceSupport = DeviceSupportUnknown;
            int ClusteredDeviceSupport = DeviceSupportUnknown;
            int TiledDeviceSupport = DeviceSupportUnknown;
            std::uint32_t MultisampleCounts = 0;
            std::uint32_t WeightedTransparencySampleCounts = 0;
            std::string Error;
        };
        const bool incrementalWorld=request.Scene->World && UploadedSceneValid_ && SceneCacheMeshes_ &&
            UploadedSceneKey_.Demo==sceneKey.Demo && UploadedSceneKey_.Revision==sceneKey.Revision &&
            UploadedSceneKey_.Backend==sceneKey.Backend;
        const auto previousMeshes=SceneCacheMeshes_;
        std::unordered_set<SceneNodeId> dynamicNodes;
        SceneSnapshot changedScene;
        if (incrementalWorld)
        {
            changedScene.Materials=request.Scene->Materials;
            changedScene.World=request.Scene->World;
            for (const auto& node:request.Scene->Nodes)
            {
                const auto previous=SceneCacheSources_.find(node.Id);
                if (previous==SceneCacheSources_.end() || previous->second!=node.Mesh)
                { dynamicNodes.insert(node.Id); changedScene.Nodes.push_back(node); }
            }
        }
        std::shared_ptr<const MeshList> meshes;
        bool convertedSceneCandidate = false;
        if (SceneCacheValid_ && sceneKey == SceneCacheKey_) meshes = SceneCacheMeshes_;
        else
        {
            // Deforming lakebed vertices retain immutable procedural textures.
            // Stable image identity keeps native texture and vertex allocations resident.
            if (!incrementalWorld) SceneImageCache_.clear();
            auto converted=BuildRendererMeshes(incrementalWorld?changedScene:*request.Scene,cancellation,fullMaterial,&SceneImageCache_);
            auto next=std::make_shared<MeshList>();
            next->reserve(incrementalWorld?std::max(converted.size(),previousMeshes->size()):converted.size());
            for (auto& mesh:converted) next->push_back(std::make_shared<Renderer::MeshData>(std::move(mesh)));
            if (incrementalWorld)
            {
                std::unordered_set<SceneNodeId> present;
                for (const auto& node:request.Scene->Nodes) present.insert(node.Id);
                for (const auto& mesh:*previousMeshes)
                    if (present.count(mesh->SelectionGroup) && !dynamicNodes.count(mesh->SelectionGroup)) next->push_back(mesh);
            }
            meshes=std::move(next);
            convertedSceneCandidate = true;
        }
        if (cancellation.IsCancellationRequested()) return Cancelled();
        const bool uploadScene =
            !UploadedSceneValid_ || !(sceneKey == UploadedSceneKey_);
        const std::shared_ptr<std::promise<ConfigureResult>> configured =
            std::make_shared<std::promise<ConfigureResult>>();
        std::future<ConfigureResult> configureFuture = configured->get_future();
        const ViewportSettings settings = request.Settings;
        const float aspect = static_cast<float>(settings.Width) / static_cast<float>(settings.Height);
        const Maths::Matrix4 view = Maths::Matrix4::LookAtRH(
            ToMath(request.Scene->SceneCamera.Position),
            ToMath(request.Scene->SceneCamera.Target),
            ToMath(request.Scene->SceneCamera.Up));
        const Maths::Matrix4 projection = Maths::Matrix4::PerspectiveRH(
            request.Scene->SceneCamera.VerticalFieldOfViewDegrees * Pi / 180.0f,
            aspect,
            0.05f,
            request.Scene->World ? 10000.0f : 1000.0f);
        const Renderer::LightingState lighting = BuildRendererLighting(*request.Scene, SceneLights_);
        // Local-space vertex/index buffers remain resident during rigid motion.
        // Material sections share their source node's current transform.
        std::unordered_map<SceneNodeId, Renderer::Matrix4> worldTransforms;
        for (const SceneNode& node : request.Scene->Nodes)
            worldTransforms.emplace(node.Id, ToRendererMatrix(ToMath(node.LocalTransform).ToMatrix()));
        ShadowFrame shadowFrame;
        const bool requestedPaths = Techniques::UsesPathTracing(settings.Techniques.Pipeline) && PathDeviceSupport_.load() == DeviceSupportAvailable;
        if (!requestedPaths && settings.ShadowsEnabled && settings.Mode != FillMode::Wireframe)
        {
            if (!ShadowCacheValid_ || !(sceneKey == ShadowCacheKey_) ||
                ShadowTransformRevision_ != request.Scene->TransformRevision ||
                ShadowLightingRevision_ != request.Scene->LightingRevision)
            {
                CachedShadow_ = BuildShadowFrame(*request.Scene,!fullMaterial,request.Scene->World?&WorldShadowBounds_:nullptr);
                ShadowCacheKey_ = sceneKey;
                ShadowTransformRevision_ = request.Scene->TransformRevision;
                ShadowLightingRevision_ = request.Scene->LightingRevision;
                ShadowCacheValid_ = true;
            }
            shadowFrame = CachedShadow_;
        }
        const Renderer::Matrix4 lightViewProjection = ToRendererMatrix(
            shadowFrame.Valid ? shadowFrame.ViewProjection : Maths::Matrix4::Identity());
        const bool enqueued = Context_.Thread.Enqueue(
            [meshes, configured, settings, backend, cameraView = ToRendererMatrix(view), cameraProjection = ToRendererMatrix(projection), lightViewProjection, lighting,
             hasShadowFrame = shadowFrame.Valid, uploadScene, cancellation, incrementalWorld, previousMeshes, dynamicNodes,
             worldTransforms = std::move(worldTransforms), sampleTime = request.Scene->SampleTimeSeconds,
             triangleCount=request.Scene->TriangleCountAtLoad,
             world=request.Scene->World, historyEpoch = request.HistoryEpoch, bakedLightingDirectory=request.Scene->ActiveBakedLightingDirectory(),
             environment=request.Scene->Environment, volume=request.Scene->Volume](
                Renderer::Renderer& renderer)
            {
                ConfigureResult result;
                try
                {
                    if (cancellation.IsCancellationRequested())
                    {
                        result.Cancelled = true;
                        configured->set_value(std::move(result));
                        return;
                    }
                    bool rendererRecreated = false;
                    const int shadowMapSize = static_cast<int>(std::clamp(
                        std::max(settings.Width, settings.Height), 256U, 2048U));
                    const auto initializeRenderer = [&]()
                    {
                        Renderer::RendererCreateInfo createInfo;
                        createInfo.Backend = backend;
                        createInfo.Width = static_cast<int>(settings.Width);
                        createInfo.Height = static_cast<int>(settings.Height);
                        createInfo.AutomaticViewProjection = false;
                        // Runtime backend switching can create an independent
                        // DX12 presentation device. Keep the process-global
                        // debug layer opt-in so it cannot reset that device.
                        createInfo.CaptureDiagnosticBuffers = false;
                        createInfo.ShadowMapSize = shadowMapSize;
                        return renderer.Initialize(createInfo);
                    };
                    if (!renderer.IsInitialized())
                    {
                        result.SceneInvalidated = true;
                        if (!initializeRenderer())
                        {
                            result.Error = renderer.GetLastError();
                            configured->set_value(std::move(result));
                            return;
                        }
                        rendererRecreated = true;
                    }
                    else if (!renderer.Resize(
                                 static_cast<int>(settings.Width),
                                 static_cast<int>(settings.Height),
                                 shadowMapSize))
                    {
                        renderer.Shutdown();
                        result.SceneInvalidated = true;
                        if (!initializeRenderer())
                        {
                            result.Error = renderer.GetLastError();
                            configured->set_value(std::move(result));
                            return;
                        }
                        rendererRecreated = true;
                    }

                    const auto capabilities=renderer.GetCapabilities();
                    const auto deviceSupport=[](bool available) { return available?DeviceSupportAvailable:DeviceSupportUnavailable; };
                    result.MultisampleCounts = capabilities.SupportedSampleCounts;
                    result.WeightedTransparencySampleCounts = capabilities.WeightedTransparencySampleCounts;
                    result.DeferredDeviceSupport = deviceSupport(capabilities.DeferredShading);
                    result.SaoDeviceSupport = deviceSupport(capabilities.ScalableAmbientObscurance);
                    result.ExposureDeviceSupport = deviceSupport(capabilities.TemporalEffects);
                    result.ParallaxDeviceSupport = deviceSupport(capabilities.ParallaxOcclusion);
                    result.DisplacementDeviceSupport = deviceSupport(capabilities.GeometryDisplacement);
                    result.TessellationDeviceSupport = deviceSupport(capabilities.GeometryTessellation);
                    result.VirtualGeometryDeviceSupport = deviceSupport(capabilities.GeometryVirtualization);
                    result.NaniteDeviceSupport = deviceSupport(capabilities.NaniteRendering);
                    result.MeshShaderDeviceSupport = deviceSupport(capabilities.MeshShaders);
                    result.SceneStorageBlockBytes = capabilities.SceneStorageBlockBytes;
                    result.MaterialAnisotropicFiltering = capabilities.MaterialAnisotropicFiltering;
                    result.NaniteHardwareDeviceSupport = deviceSupport(capabilities.NaniteHardwareScheduling);
                    result.LumenHardwareDeviceSupport = deviceSupport(capabilities.LumenHardwareRayTracing);
                    result.TemporalSuperResolutionDeviceSupport = deviceSupport(capabilities.TemporalSuperResolution);
                    result.VirtualShadowDeviceSupport = deviceSupport(capabilities.VirtualShadowMaps);
                    result.TemporalDeviceSupport = deviceSupport(capabilities.TemporalAntiAliasing);
                    result.GIDeviceSupport = deviceSupport(capabilities.ScreenSpaceGlobalIllumination);
                    result.FogDeviceSupport = deviceSupport(capabilities.VolumetricFog);
                    result.VolumeLightDeviceSupport = deviceSupport(capabilities.VolumetricLighting);
                    result.VSMDeviceSupport = deviceSupport(capabilities.VarianceShadowMaps);
                    result.ContactDeviceSupport = deviceSupport(capabilities.ContactShadows);
                    result.ForwardPlusDeviceSupport = deviceSupport(capabilities.ForwardPlusRendering);
                    result.ClusteredDeviceSupport = deviceSupport(capabilities.ClusteredForwardRendering);
                    result.TiledDeviceSupport = deviceSupport(capabilities.TiledDeferredRendering);
                    result.PeelingDeviceSupport = deviceSupport(capabilities.DepthPeeling);
                    result.RayAODeviceSupport = deviceSupport(capabilities.RayTracedAmbientOcclusion);
                    result.RayReflectionDeviceSupport = deviceSupport(capabilities.RayTracedReflections);
                    result.RayGIDeviceSupport = deviceSupport(capabilities.RayTracedGI);
                    result.VoxelDeviceSupport = deviceSupport(capabilities.VoxelGI);
                    result.LumenDeviceSupport = deviceSupport(capabilities.LumenGI);
                    result.PathDeviceSupport = deviceSupport(capabilities.PathTracing);
                    result.GpuDrivenDeviceSupport = deviceSupport(capabilities.GPUDrivenRendering);
                    result.VisibilityDeviceSupport = deviceSupport(capabilities.VisibilityBufferRendering);
                    result.HybridDeviceSupport = deviceSupport(capabilities.HybridRendering);
                    result.LightShaftDeviceSupport = deviceSupport(capabilities.LightShafts);
                    const bool replaceScene = (uploadScene && !incrementalWorld) || rendererRecreated;
                    if (replaceScene)
                    {
                        renderer.ClearScene();
                        result.SceneInvalidated = true;
                    }
                    renderer.SetGpuPresentationEnabled(settings.GpuPresentation);
                    renderer.SetAutomaticViewProjection(false);
                    renderer.SetCameraMatrices(cameraView, cameraProjection);
                    renderer.SetFrameTime(sampleTime, historyEpoch);
                    const auto wind=world?world->Settings:BigWorldSettings{};
                    const float windAngle=wind.WindDirection*3.14159265f/180;
                    renderer.SetVegetation(world?world->VegetationTime:0,
                        {std::cos(windAngle),std::sin(windAngle),0,wind.WindSpeed*0.045f},
                        {wind.GustWavelength,wind.GustStrength,wind.LeafFlutter,0});
                    renderer.SetLightViewProjection(lightViewProjection);
                    renderer.SetLighting(lighting);
                    // A newly selected backend has not published its limits to
                    // the UI yet. Resolve scene storage here as well, before the
                    // first frame can allocate an oversized world-space table.
                    auto sceneSettings=settings.Techniques;
                    // From inside the lake the water mesh resolves the refracted
                    // sky and total internal reflection at the interface.
                    if (world && world->CameraUnderwater) sceneSettings.Reflections.Planar.Enabled=false;
                    if (capabilities.SceneStorageBlockBytes)
                        sceneSettings=Renderer::ConstrainSceneTriangleStorage(sceneSettings,triangleCount,capabilities.SceneStorageBlockBytes);
                    renderer.SetTechniqueSettings(sceneSettings);
                    renderer.SetRenderBuffer(settings.Buffer);
                    renderer.SetBakedLightingDirectory(bakedLightingDirectory);
                    renderer.SetEnvironmentRadiance(environment);
                    renderer.SetDensityVolume(volume);
                    const bool paths = Techniques::UsesPathTracing(renderer.GetTechniqueSettings().Pipeline);
                    renderer.SetRenderMode(paths ? Renderer::RenderMode::Solid : ToRendererMode(settings.Mode));
                    renderer.SetHairSelectionGroup(settings.SelectedNode);
                    renderer.SetShadowsEnabled(settings.ShadowsEnabled && (paths || hasShadowFrame));
                    if (replaceScene)
                    {
                        for (const auto& mesh : *meshes)
                        {
                            if (cancellation.IsCancellationRequested())
                            {
                                result.Cancelled = true;
                                configured->set_value(std::move(result));
                                return;
                            }
                            if (!renderer.AddMesh(*mesh))
                            {
                                result.Error = renderer.GetLastError();
                                configured->set_value(std::move(result));
                                return;
                            }
                        }
                    }
                    if (incrementalWorld && uploadScene && !replaceScene)
                    {
                        // Mark a partial update invalid until the complete batch succeeds.
                        result.SceneInvalidated=true;
                        std::unordered_set<std::string> oldIds,newIds;
                        for (const auto& previous:*previousMeshes) oldIds.insert(previous->Id);
                        for (const auto& current:*meshes) newIds.insert(current->Id);
                        for (const auto& previous:*previousMeshes)
                            if (!newIds.count(previous->Id) && !renderer.RemoveMesh(previous->Id)) throw std::runtime_error(renderer.GetLastError());
                        for (const auto& current:*meshes)
                            if (dynamicNodes.count(current->SelectionGroup))
                                if (!(oldIds.count(current->Id)?renderer.UpdateMesh(*current):renderer.AddMesh(*current)))
                                    throw std::runtime_error(renderer.GetLastError());
                    }
                    for (const auto& mesh : *meshes)
                    {
                        if (cancellation.IsCancellationRequested())
                        {
                            result.Cancelled = true;
                            configured->set_value(std::move(result));
                            return;
                        }
                        Renderer::OutlineSettings outline;
                        const auto transform = worldTransforms.find(mesh->SelectionGroup);
                        if (transform == worldTransforms.end() ||
                            !renderer.SetWorldTransform(mesh->Id, transform->second))
                        {
                            result.SceneInvalidated = true;
                            result.Error = "Resident mesh has no valid source-node transform";
                            configured->set_value(std::move(result));
                            return;
                        }
                        outline.Enabled = settings.Buffer==Techniques::RenderBuffer::Color && settings.OutlineEnabled &&
                                          settings.SelectedNode != 0 &&
                                          mesh->SelectionGroup == settings.SelectedNode;
                        if (!renderer.SetOutline(mesh->Id, outline))
                        {
                            // A missing mesh means the tracked resident scene
                            // no longer matches the Renderer instance.
                            result.SceneInvalidated = true;
                            result.Error = renderer.GetLastError();
                            configured->set_value(std::move(result));
                            return;
                        }
                    }
                    result.Success = true;
                }
                catch (const std::exception& exception)
                {
                    result.Error = exception.what();
                }
                catch (...)
                {
                    result.Error = "Unknown exception while configuring the realtime renderer";
                }
                configured->set_value(std::move(result));
            });
        if (!enqueued) return Failed("Could not enqueue the realtime scene on the RHI thread");
        const ConfigureResult configureResult = configureFuture.get();
        if (configureResult.MultisampleCounts)
        {
            MultisampleCounts_.store(configureResult.MultisampleCounts);
            WeightedTransparencyCounts_.store(configureResult.WeightedTransparencySampleCounts);
        }
        if (configureResult.DeferredDeviceSupport != DeviceSupportUnknown)
            DeferredDeviceSupport_.store(configureResult.DeferredDeviceSupport);
        if (configureResult.SaoDeviceSupport != DeviceSupportUnknown)
            SaoDeviceSupport_.store(configureResult.SaoDeviceSupport);
        if (configureResult.ExposureDeviceSupport != DeviceSupportUnknown) ExposureDeviceSupport_.store(configureResult.ExposureDeviceSupport);
        if (configureResult.ParallaxDeviceSupport != DeviceSupportUnknown) ParallaxDeviceSupport_.store(configureResult.ParallaxDeviceSupport);
        if (configureResult.DisplacementDeviceSupport != DeviceSupportUnknown) DisplacementDeviceSupport_.store(configureResult.DisplacementDeviceSupport);
        if (configureResult.TessellationDeviceSupport != DeviceSupportUnknown) TessellationDeviceSupport_.store(configureResult.TessellationDeviceSupport);
        if (configureResult.NaniteDeviceSupport != DeviceSupportUnknown) NaniteDeviceSupport_.store(configureResult.NaniteDeviceSupport);
        if (configureResult.MeshShaderDeviceSupport != DeviceSupportUnknown) MeshShaderDeviceSupport_.store(configureResult.MeshShaderDeviceSupport);
        if (configureResult.Success)
        {
            SceneStorageBlockBytes_.store(configureResult.SceneStorageBlockBytes);
            MaterialAnisotropicFiltering_.store(configureResult.MaterialAnisotropicFiltering);
        }
        if (configureResult.NaniteHardwareDeviceSupport!=DeviceSupportUnknown) NaniteHardwareDeviceSupport_.store(configureResult.NaniteHardwareDeviceSupport);
        if (configureResult.LumenHardwareDeviceSupport!=DeviceSupportUnknown) LumenHardwareDeviceSupport_.store(configureResult.LumenHardwareDeviceSupport);
        if (configureResult.VirtualGeometryDeviceSupport != DeviceSupportUnknown) VirtualGeometryDeviceSupport_.store(configureResult.VirtualGeometryDeviceSupport);
        if (configureResult.TemporalSuperResolutionDeviceSupport!=DeviceSupportUnknown) TemporalSuperResolutionDeviceSupport_.store(configureResult.TemporalSuperResolutionDeviceSupport);
        if (configureResult.VirtualShadowDeviceSupport!=DeviceSupportUnknown) VirtualShadowDeviceSupport_.store(configureResult.VirtualShadowDeviceSupport);
        if (configureResult.TemporalDeviceSupport != DeviceSupportUnknown)
            TemporalDeviceSupport_.store(configureResult.TemporalDeviceSupport);
        if (configureResult.GIDeviceSupport != DeviceSupportUnknown)
            GIDeviceSupport_.store(configureResult.GIDeviceSupport);
        if (configureResult.FogDeviceSupport != DeviceSupportUnknown)
            FogDeviceSupport_.store(configureResult.FogDeviceSupport);
        if (configureResult.VolumeLightDeviceSupport != DeviceSupportUnknown)
            VolumeLightDeviceSupport_.store(configureResult.VolumeLightDeviceSupport);
        if (configureResult.VSMDeviceSupport != DeviceSupportUnknown)
            VSMDeviceSupport_.store(configureResult.VSMDeviceSupport);
        if (configureResult.ContactDeviceSupport != DeviceSupportUnknown)
            ContactDeviceSupport_.store(configureResult.ContactDeviceSupport);
        if (configureResult.ForwardPlusDeviceSupport != DeviceSupportUnknown)
            ForwardPlusDeviceSupport_.store(configureResult.ForwardPlusDeviceSupport);
        if (configureResult.ClusteredDeviceSupport != DeviceSupportUnknown)
            ClusteredDeviceSupport_.store(configureResult.ClusteredDeviceSupport);
        if (configureResult.TiledDeviceSupport != DeviceSupportUnknown)
            TiledDeviceSupport_.store(configureResult.TiledDeviceSupport);
        if (configureResult.PeelingDeviceSupport != DeviceSupportUnknown)
            PeelingDeviceSupport_.store(configureResult.PeelingDeviceSupport);
        if (configureResult.PathDeviceSupport != DeviceSupportUnknown)
            PathDeviceSupport_.store(configureResult.PathDeviceSupport);
        if (configureResult.HybridDeviceSupport != DeviceSupportUnknown)
            HybridDeviceSupport_.store(configureResult.HybridDeviceSupport);
        if (configureResult.VisibilityDeviceSupport != DeviceSupportUnknown)
            VisibilityDeviceSupport_.store(configureResult.VisibilityDeviceSupport);
        if (configureResult.GpuDrivenDeviceSupport != DeviceSupportUnknown)
            GpuDrivenDeviceSupport_.store(configureResult.GpuDrivenDeviceSupport);
        if (configureResult.VoxelDeviceSupport != DeviceSupportUnknown) VoxelDeviceSupport_.store(configureResult.VoxelDeviceSupport);
        if (configureResult.LumenDeviceSupport!=DeviceSupportUnknown) LumenDeviceSupport_.store(configureResult.LumenDeviceSupport);
        if (configureResult.RayGIDeviceSupport != DeviceSupportUnknown)
            RayGIDeviceSupport_.store(configureResult.RayGIDeviceSupport);
        if (configureResult.RayReflectionDeviceSupport != DeviceSupportUnknown)
            RayReflectionDeviceSupport_.store(configureResult.RayReflectionDeviceSupport);
        if (configureResult.RayAODeviceSupport != DeviceSupportUnknown)
            RayAODeviceSupport_.store(configureResult.RayAODeviceSupport);
        if (configureResult.LightShaftDeviceSupport != DeviceSupportUnknown)
            LightShaftDeviceSupport_.store(configureResult.LightShaftDeviceSupport);
        if (configureResult.Success)
        {
            UploadedSceneKey_ = sceneKey;
            UploadedSceneValid_ = true;
            if (convertedSceneCandidate)
            {
                SceneCacheKey_ = sceneKey;
                SceneCacheMeshes_ = meshes;
                SceneCacheSources_.clear();
                if (request.Scene->World) for (const auto& node:request.Scene->Nodes) SceneCacheSources_.emplace(node.Id,node.Mesh);
                SceneCacheValid_ = true;
            }
        }
        else
        {
            // Rapid settings changes can cancel before touching the resident
            // scene. Only force a re-upload after initialization, ClearScene,
            // or an observed missing mesh invalidated that scene.
            if (configureResult.SceneInvalidated)
            {
                UploadedSceneValid_ = false;
            }
            std::lock_guard<std::mutex> lock(CacheMutex_);
            CacheValid_ = false;
        }
        if (configureResult.Cancelled || cancellation.IsCancellationRequested())
        {
            return Cancelled();
        }
        if (!configureResult.Success)
        {
            return Failed(configureResult.Error.empty()
                ? "Realtime renderer scene configuration failed"
                : configureResult.Error);
        }
        if (cancellation.IsCancellationRequested()) return Cancelled();
        std::future<bool> frame = Context_.Thread.EnqueueFrame();
        const bool frameSucceeded = frame.get();
        if (!frameSucceeded)
        {
            UploadedSceneValid_ = false;
            {
                std::lock_guard<std::mutex> lock(CacheMutex_);
                CacheValid_ = false;
            }
            const std::shared_ptr<std::promise<std::string>> recovered =
                std::make_shared<std::promise<std::string>>();
            std::future<std::string> recoveryFuture = recovered->get_future();
            if (!Context_.Thread.Enqueue(
                    [recovered](Renderer::Renderer& renderer)
                    {
                        const std::string error = renderer.GetLastError();
                        renderer.Shutdown();
                        recovered->set_value(error);
                    }))
            {
                return Failed("The realtime RHI failed and could not be reset");
            }
            const std::string error = recoveryFuture.get();
            return Failed(error.empty()
                ? "The realtime RHI failed to render the frame"
                : error);
        }
        if (cancellation.IsCancellationRequested()) return Cancelled();
        struct ReadbackResult
        {
            std::string BakedLightingStatus;
            std::string VoxelGIStatus;
            std::string LumenGIStatus;
            std::string HairFurStatus;
            bool HairGeometryPending = false;
            Image Output;
            std::string Error;
            bool NeedsTemporalFrame = false;
            std::uint32_t PathTracingSamples = 0;
            GeometryStatistics Geometry;
        };
        const std::shared_ptr<std::promise<ReadbackResult>> readback =
            std::make_shared<std::promise<ReadbackResult>>();
        std::future<ReadbackResult> readbackFuture = readback->get_future();
        if (!Context_.Thread.Enqueue(
                [readback,contextMutex=Context_.RenderMutex](Renderer::Renderer& renderer)
                {
                    ReadbackResult result;
                    try
                    {
                        auto capture = renderer.TakeLastFrameCapture();
                        if(capture.GpuContext && capture.GpuContext->Backend==Renderer::RHIBackend::OpenGLES)
                        {
                            auto context=std::make_shared<Renderer::GpuImage>(*capture.GpuContext);
                            // The EGL context migrates between the render and UI threads under this mutex.
                            context->Owner=contextMutex;capture.GpuContext=std::move(context);
                        }
                        result.Output = ToEditorImage(capture);
                        result.Error = renderer.GetLastError();
                        result.NeedsTemporalFrame = capture.NeedsTemporalFrame;
                        result.PathTracingSamples = capture.Statistics.PathTracingSampleCount;
                        result.BakedLightingStatus = std::move(capture.BakedLightingStatus);
                        result.VoxelGIStatus = std::move(capture.VoxelGIStatus);
                        result.LumenGIStatus=std::move(capture.LumenGIStatus);
                        result.HairFurStatus = std::move(capture.HairFurStatus);
                        result.HairGeometryPending = capture.HairGeometryPending;
                        const auto& statistics = capture.Statistics;
                        result.Geometry = {statistics.LodReducedMeshCount, true, statistics.DrawCallCount, statistics.InstancedDrawCallCount,
                            statistics.IndirectDrawCallCount, statistics.GpuCandidateInstanceCount,
                            statistics.VisibilityRasterDrawCount, statistics.VisibilityMaterialDrawCount, statistics.HybridRenderingPassCount, statistics.SubsurfacePassCount,
                            static_cast<std::uint32_t>(capture.VoxelGIPasses.size()),statistics.HairFurPassCount,statistics.ParallaxMaterialDrawCount,statistics.DisplacedMeshCount,statistics.DisplacedVertexCount,statistics.DisplacementPreparationDispatches,statistics.DisplacementGeneration,statistics.TessellatedMeshCount,statistics.TessellatedVertexCount,statistics.TessellationPreparationDispatches,statistics.GeometryPreparationDispatches,statistics.PreparedGeometryGeneration,statistics.VirtualMeshCount,statistics.VirtualNodeCount,statistics.VirtualResidentPages,statistics.VirtualPendingPages,statistics.VirtualUploadedPages,statistics.VirtualCopiedPages,statistics.VirtualBootstrapPages,statistics.VirtualUploadedBytes,statistics.VirtualCopiedBytes,statistics.VirtualPoolBytes,statistics.VirtualGeometryGeneration,statistics.NaniteMeshCount,statistics.NaniteSelectedClusters,statistics.NanitePendingPages,statistics.NaniteUploadedPages,statistics.NaniteGeometryAssemblyCount,statistics.NaniteRasterExportCount,statistics.NaniteOcclusionQueries,statistics.NaniteOcclusionPhases,statistics.NaniteGeometryGeneration,statistics.NaniteMaterialBinCount};
                        result.Geometry.NaniteReadbackBytes=statistics.NaniteReadbackBytes;
                        result.Geometry.NaniteHostRebuiltVertices=statistics.NaniteHostRebuiltVertices;
                        result.Geometry.NaniteScenePageBudget=statistics.NaniteScenePageBudget;
                        result.Geometry.NaniteFamilyCount=statistics.NaniteFamilyCount;
                        result.Geometry.NaniteViewCount=statistics.NaniteViewCount;
                        result.Geometry.NaniteBudgetBlockedGroups=statistics.NaniteBudgetBlockedGroups;
                        result.Geometry.NaniteCookRequests=statistics.NaniteCookRequests;
                        result.Geometry.NaniteRootPages=statistics.NaniteRootPages;
                        result.Geometry.NaniteAllocatedPages=statistics.NaniteAllocatedPages;
                        result.Geometry.NaniteCookSourceBytes=statistics.NaniteCookSourceBytes;
                        result.Geometry.NanitePreparationMicroseconds=statistics.NanitePreparationMicroseconds;
                        result.Geometry.NaniteBudgetLimited=statistics.NaniteBudgetLimited;
                        result.Geometry.GpuFrameMilliseconds=statistics.GpuFrameMilliseconds;
                        result.Geometry.GpuFrameTimingAvailable=statistics.GpuFrameTimingAvailable;
                        result.Geometry.MeshShaderDispatches=statistics.MeshShaderDispatchCount;
                        result.Geometry.NaniteHardwareDispatches=statistics.NaniteHardwareDispatchCount;
                        result.Geometry.LumenHardwareRaySlots=statistics.LumenHardwareRayCount;
                        result.Geometry.RenderWidth=capture.RenderWidth?capture.RenderWidth:capture.Width;
                        result.Geometry.RenderHeight=capture.RenderHeight?capture.RenderHeight:capture.Height;
                        result.Geometry.TSRPasses=statistics.TemporalSuperResolutionPassCount;
                        result.Geometry.TSRSerial=statistics.TemporalSuperResolutionSerial;
                        result.Geometry.TSRHistoryValid=statistics.TemporalSuperResolutionHistoryValid;
                        result.Geometry.TSRForegroundPixels=statistics.TemporalSuperResolutionForegroundPixels;
                        result.Geometry.TSRValidMotionPixels=statistics.TemporalSuperResolutionValidMotionPixels;
                        result.Geometry.TSRAccumulatedPixels=statistics.TemporalSuperResolutionAccumulatedPixels;
                        result.Geometry.VirtualShadowSerial=statistics.VirtualShadowSerial;
                        result.Geometry.VirtualShadowRequestedPages=statistics.VirtualShadowRequestedPages;
                        result.Geometry.VirtualShadowResidentPages=statistics.VirtualShadowResidentPages;
                        result.Geometry.VirtualShadowRenderedPages=statistics.VirtualShadowRenderedPages;
                        result.Geometry.VirtualShadowCopiedPages=statistics.VirtualShadowCopiedPages;
                        result.Geometry.VirtualShadowDrawCalls=statistics.VirtualShadowDrawCalls;
                    }
                    catch (const std::exception& exception)
                    {
                        result.Error = exception.what();
                    }
                    catch (...)
                    {
                        result.Error = "Unknown exception while reading the realtime framebuffer";
                    }
                    readback->set_value(std::move(result));
                }))
        {
            return Failed("Could not enqueue the realtime RHI readback");
        }
        ReadbackResult readbackResult = readbackFuture.get();
        if (cancellation.IsCancellationRequested()) return Cancelled();
        RenderResult preview;
        preview.Status = RenderStatus::Completed;
        preview.Output = std::move(readbackResult.Output);
        preview.NeedsTemporalFrame = readbackResult.NeedsTemporalFrame;
        preview.PathTracingSamples = readbackResult.PathTracingSamples;
        preview.BakedLightingStatus = std::move(readbackResult.BakedLightingStatus);
        preview.VoxelGIStatus = std::move(readbackResult.VoxelGIStatus);
        preview.LumenGIStatus=std::move(readbackResult.LumenGIStatus);
        preview.HairFurStatus = std::move(readbackResult.HairFurStatus);
        preview.HairGeometryPending = readbackResult.HairGeometryPending;
        preview.Geometry = readbackResult.Geometry;
        if (!preview.Output.IsValid())
        {
            return Failed("The realtime RHI returned an invalid color framebuffer");
        }
        if (!request.Settings.ContinuousRendering && !preview.Output.GpuColor && preview.Status == RenderStatus::Completed)
        {
            std::lock_guard<std::mutex> lock(CacheMutex_);
            CacheKey_ = key;
            CachedResult_ = preview;
            CacheValid_ = true;
        }
        else if (request.Settings.ContinuousRendering || preview.Output.GpuColor)
        {
            std::lock_guard<std::mutex> lock(CacheMutex_);
            CachedResult_={};CacheValid_ = false;
        }
        return preview;
    }

private:
    BackendDescriptor Descriptor_;
    RealtimeBackendContext Context_;
    // Published from the render worker; GUI reads never call GL or block on it.
    static constexpr int DeviceSupportUnknown = 0;
    static constexpr int DeviceSupportAvailable = 1;
    static constexpr int DeviceSupportUnavailable = 2;
    std::atomic<int> DeferredDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> SaoDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> TemporalDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> TemporalSuperResolutionDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> VirtualShadowDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> ExposureDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> ParallaxDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> DisplacementDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> TessellationDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> VirtualGeometryDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> NaniteDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> MeshShaderDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<std::uint64_t> SceneStorageBlockBytes_ {0};
    std::atomic<bool> MaterialAnisotropicFiltering_ {false};
    std::atomic<int> NaniteHardwareDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> LumenHardwareDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> GIDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> FogDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> VolumeLightDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> LightShaftDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> PeelingDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> RayAODeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> RayReflectionDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> RayGIDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> VoxelDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> LumenDeviceSupport_{DeviceSupportUnknown};
    std::atomic<int> PathDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> GpuDrivenDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> VisibilityDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> HybridDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> VSMDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> ContactDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> ForwardPlusDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> ClusteredDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<int> TiledDeviceSupport_ {DeviceSupportUnknown};
    std::atomic<std::uint32_t> MultisampleCounts_ {1};
    std::atomic<std::uint32_t> WeightedTransparencyCounts_ {0};
    std::mutex CacheMutex_;
    bool CacheValid_ = false;
    RealtimeCacheKey CacheKey_;
    RenderResult CachedResult_;
    bool SceneCacheValid_ = false;
    RealtimeSceneCacheKey SceneCacheKey_;
    using MeshList=std::vector<std::shared_ptr<const Renderer::MeshData>>;
    std::shared_ptr<const MeshList> SceneCacheMeshes_;
    std::unordered_map<SceneNodeId,std::shared_ptr<const MeshAsset>> SceneCacheSources_;
    RendererImageCache SceneImageCache_;
    RealtimeLightCacheKey LightCacheKey_;
    std::shared_ptr<const Renderer::SceneLightCollection> SceneLights_;
    bool UploadedSceneValid_ = false;
    RealtimeSceneCacheKey UploadedSceneKey_;
    bool ShadowCacheValid_ = false;
    RealtimeSceneCacheKey ShadowCacheKey_;
    ShadowFrame CachedShadow_;
    MeshBoundsCache WorldShadowBounds_;
    std::uint64_t ShadowTransformRevision_ = 0;
    std::uint64_t ShadowLightingRevision_ = 0;
};

class RasterRenderAdapter final : public IRenderAdapter
{
public:
    RasterRenderAdapter(BackendDescriptor descriptor, Rasterizer::RasterBackend backend)
        : Descriptor_(std::move(descriptor)), Rasterizer_(backend)
    {
    }

    [[nodiscard]] BackendDescriptor Descriptor() const override { return Descriptor_; }

    [[nodiscard]] bool QueryMaterialAnisotropicFiltering() const override { return IsAvailable(); }

    [[nodiscard]] bool IsAvailable() const override
    {
        return Rasterizer_.Capabilities().NativeExecutionAvailable;
    }

    [[nodiscard]] RenderResult Render(
        const RenderRequest& request,
        const CancellationToken& cancellation) override
    {
        return RenderRasterImage(request, cancellation, Rasterizer_, SceneCache_);
    }

private:
    BackendDescriptor Descriptor_;
    Rasterizer::SoftwareRasterizer Rasterizer_;
    RasterSceneCache SceneCache_;
};

[[nodiscard]] std::uint32_t RayMaterialIndex(
    const SceneSnapshot& source,
    const SceneNode& node,
    std::size_t firstIndex)
{
    const Material* material = MaterialForTriangle(source, node, firstIndex);
    if (material == nullptr) return 0;
    return static_cast<std::uint32_t>(material - source.Materials.data());
}

using RayTextureCache =
    std::unordered_map<const ImageAsset*, std::uint32_t>;

[[nodiscard]] std::uint32_t AddRayTexture(
    const std::shared_ptr<const ImageAsset>& source,
    RayTracing::Scene& scene,
    RayTextureCache& cache)
{
    if (!source || !source->IsValid())
    {
        return RayTracing::InvalidRayTextureIndex;
    }
    const auto found = cache.find(source.get());
    if (found != cache.end())
    {
        return found->second;
    }
    if (scene.Textures.size() >= RayTracing::InvalidRayTextureIndex)
    {
        return RayTracing::InvalidRayTextureIndex;
    }

    const std::uint32_t index =
        static_cast<std::uint32_t>(scene.Textures.size());
    RayTracing::RayTexture texture;
    texture.Width = source->Width;
    texture.Height = source->Height;
    // Keep the ImageAsset alive while exposing its immutable pixel vector
    // directly to every ray backend without an intermediate copy.
    texture.PixelsRGBA8 =
        std::shared_ptr<const std::vector<std::uint8_t>>(
            source,
            &source->PixelsRGBA8);
    if (!texture.IsValid())
    {
        return RayTracing::InvalidRayTextureIndex;
    }
    scene.Textures.push_back(std::move(texture));
    cache.emplace(source.get(), index);
    return index;
}

[[nodiscard]] std::shared_ptr<const RayTracing::Scene> BuildRayScene(
    const SceneSnapshot& source,
    const CancellationToken& cancellation)
{
    auto scene = std::make_shared<RayTracing::Scene>();
    RayTextureCache textureCache;
    if (source.Materials.empty())
    {
        scene->Materials.emplace_back();
    }
    else
    {
        scene->Materials.reserve(source.Materials.size());
        for (const Material& sourceMaterial : source.Materials)
        {
            RayTracing::Material material;
            material.Albedo = {
                sourceMaterial.Diffuse.R,
                sourceMaterial.Diffuse.G,
                sourceMaterial.Diffuse.B};
            constexpr float ambientEpsilon = 1.0e-4f;
            const bool hasAmbient =
                std::abs(sourceMaterial.Ambient.R) > ambientEpsilon ||
                std::abs(sourceMaterial.Ambient.G) > ambientEpsilon ||
                std::abs(sourceMaterial.Ambient.B) > ambientEpsilon;
            const Color& ambient =
                hasAmbient ? sourceMaterial.Ambient : sourceMaterial.Diffuse;
            material.AmbientColor = {
                ambient.R,
                ambient.G,
                ambient.B};
            material.SpecularColor = {
                sourceMaterial.Specular.R,
                sourceMaterial.Specular.G,
                sourceMaterial.Specular.B};
            material.Emission = {
                sourceMaterial.Emissive.R,
                sourceMaterial.Emissive.G,
                sourceMaterial.Emissive.B};
            material.Shininess = std::max(sourceMaterial.Shininess, 1.0f);
            // MTL Ks/Ns describe a local Phong highlight, not mirror transport.
            material.Reflectivity = 0.0f;
            material.DiffuseTextureIndex = AddRayTexture(
                sourceMaterial.DiffuseImage,
                *scene,
                textureCache);
            scene->Materials.push_back(material);
        }
    }

    scene->Triangles.reserve(source.TriangleCountAtLoad);
    for (const SceneNode& node : source.Nodes)
    {
        if (cancellation.IsCancellationRequested()) return nullptr;
        if (!node.Visible || !node.Mesh) continue;
        const MeshAsset& mesh = *node.Mesh;
        const Maths::Transform transform = ToMath(node.LocalTransform);
        for (std::size_t firstIndex = 0; firstIndex + 2 < mesh.Indices.size(); firstIndex += 3)
        {
            if (firstIndex % (4096U * 3U) == 0 && cancellation.IsCancellationRequested())
            {
                return nullptr;
            }
            const std::uint32_t i0 = mesh.Indices[firstIndex];
            const std::uint32_t i1 = mesh.Indices[firstIndex + 1];
            const std::uint32_t i2 = mesh.Indices[firstIndex + 2];
            if (i0 >= mesh.Vertices.size() || i1 >= mesh.Vertices.size() || i2 >= mesh.Vertices.size())
            {
                continue;
            }
            const Vertex& v0 = mesh.Vertices[i0];
            const Vertex& v1 = mesh.Vertices[i1];
            const Vertex& v2 = mesh.Vertices[i2];
            const Maths::Vector3 p0 = transform.TransformPoint(ToMath(v0.Position));
            const Maths::Vector3 p1 = transform.TransformPoint(ToMath(v1.Position));
            const Maths::Vector3 p2 = transform.TransformPoint(ToMath(v2.Position));
            if ((p1 - p0).Cross(p2 - p0).LengthSquared() <= 1.0e-16f)
            {
                continue;
            }
            RayTracing::Triangle triangle;
            triangle.Position0 = {p0.x, p0.y, p0.z};
            triangle.Position1 = {p1.x, p1.y, p1.z};
            triangle.Position2 = {p2.x, p2.y, p2.z};
            const Maths::Vector3 n0 = TransformNormal(transform, v0.Normal);
            const Maths::Vector3 n1 = TransformNormal(transform, v1.Normal);
            const Maths::Vector3 n2 = TransformNormal(transform, v2.Normal);
            triangle.Normal0 = {n0.x, n0.y, n0.z};
            triangle.Normal1 = {n1.x, n1.y, n1.z};
            triangle.Normal2 = {n2.x, n2.y, n2.z};
            triangle.Color0 = {
                Maths::SRGBToLinear(v0.VertexColor.R),
                Maths::SRGBToLinear(v0.VertexColor.G),
                Maths::SRGBToLinear(v0.VertexColor.B)};
            triangle.Color1 = {
                Maths::SRGBToLinear(v1.VertexColor.R),
                Maths::SRGBToLinear(v1.VertexColor.G),
                Maths::SRGBToLinear(v1.VertexColor.B)};
            triangle.Color2 = {
                Maths::SRGBToLinear(v2.VertexColor.R),
                Maths::SRGBToLinear(v2.VertexColor.G),
                Maths::SRGBToLinear(v2.VertexColor.B)};
            triangle.TexCoord0 = {v0.TexCoord.X, v0.TexCoord.Y};
            triangle.TexCoord1 = {v1.TexCoord.X, v1.TexCoord.Y};
            triangle.TexCoord2 = {v2.TexCoord.X, v2.TexCoord.Y};
            triangle.SmoothNormals = true;
            triangle.MaterialIndex = RayMaterialIndex(source, node, firstIndex);
            scene->Triangles.push_back(triangle);
        }
    }

    const LightSelection lightSelection = SelectLights(source);
    for (std::size_t lightIndex = 0; lightIndex < lightSelection.Count; ++lightIndex)
    {
        const Light& sourceLight = source.Lights[lightSelection.SceneIndices[lightIndex]];
        const bool castsShadows =
            static_cast<std::int32_t>(lightIndex) == lightSelection.ShadowLightIndex;
        if (sourceLight.Type == LightType::Directional)
        {
            const Maths::Vector3 directionToLight =
                -NormalizedLightDirection(sourceLight);
            RayTracing::DirectionalLight light;
            light.DirectionToLight = {
                directionToLight.x,
                directionToLight.y,
                directionToLight.z};
            light.Color = {
                sourceLight.LightColor.R,
                sourceLight.LightColor.G,
                sourceLight.LightColor.B};
            light.Intensity = sourceLight.Enabled ? sourceLight.Intensity : 0;
            light.CastShadows = castsShadows;
            scene->DirectionalLights.push_back(light);
        }
        else if (sourceLight.Type == LightType::Point)
        {
            RayTracing::PointLight light;
            light.Position = {
                sourceLight.Position.X,
                sourceLight.Position.Y,
                sourceLight.Position.Z};
            light.Color = {
                sourceLight.LightColor.R,
                sourceLight.LightColor.G,
                sourceLight.LightColor.B};
            light.Intensity = sourceLight.Enabled ? sourceLight.Intensity : 0;
            light.Range = sourceLight.Range;
            light.CastShadows = castsShadows;
            scene->PointLights.push_back(light);
        }
        else if (sourceLight.Type == LightType::Area)
        {
            const Maths::Vector3 direction = NormalizedLightDirection(sourceLight);
            RayTracing::RectAreaLight light;
            light.Position = {
                sourceLight.Position.X,
                sourceLight.Position.Y,
                sourceLight.Position.Z};
            light.Direction = {
                direction.x,
                direction.y,
                direction.z};
            light.Color = {
                sourceLight.LightColor.R,
                sourceLight.LightColor.G,
                sourceLight.LightColor.B};
            light.Intensity = sourceLight.Enabled ? sourceLight.Intensity : 0;
            light.Range = sourceLight.Range;
            light.Size = {
                sourceLight.AreaSize.X,
                sourceLight.AreaSize.Y};
            light.CastShadows = castsShadows;
            scene->RectAreaLights.push_back(light);
        }
    }
    return scene;
}

class RayRenderAdapter final : public IRenderAdapter
{
public:
    RayRenderAdapter(
        BackendDescriptor descriptor,
        std::shared_ptr<RayTracing::IRayTracingRHI> rhi)
        : Descriptor_(std::move(descriptor))
        , RHI_(std::move(rhi))
        , PresentationRasterizer_(Rasterizer::RasterBackend::CPU)
    {
    }

    [[nodiscard]] BackendDescriptor Descriptor() const override { return Descriptor_; }

    [[nodiscard]] bool IsAvailable() const override
    {
        return RHI_ != nullptr && RHI_->GetBackendInfo().IsAvailable;
    }

    [[nodiscard]] RenderResult Render(
        const RenderRequest& request,
        const CancellationToken& cancellation) override
    {
        if (!request.Scene) return Failed("The ray tracing request has no scene");
        if (!RHI_) return Failed("The selected ray tracing backend is unavailable");
        if (cancellation.IsCancellationRequested()) return Cancelled();
        if (request.Settings.Mode == FillMode::Wireframe)
        {
            const RayTracing::BackendInfo backend = RHI_->GetBackendInfo();
            if (!backend.IsAvailable)
            {
                return Failed(backend.Description.empty()
                    ? "The selected ray tracing backend is unavailable"
                    : backend.Description);
            }
            // Wireframe is a topology presentation mode. Render it directly
            // instead of tracing and then discarding a complete radiance frame.
            return RenderRasterImage(request, cancellation, PresentationRasterizer_, PresentationSceneCache_);
        }
        std::shared_ptr<const RayTracing::Scene> scene;
        {
            std::lock_guard<std::mutex> lock(SceneCacheMutex_);
            if (CachedScene_ && CachedDemo_ == request.Scene->Demo &&
                CachedRevision_ == request.Scene->Revision &&
                CachedTransformRevision_ == request.Scene->TransformRevision &&
                CachedLightingRevision_ == request.Scene->LightingRevision)
            {
                scene = CachedScene_;
            }
        }
        if (!scene)
        {
            scene = BuildRayScene(*request.Scene, cancellation);
            if (scene && !cancellation.IsCancellationRequested())
            {
                std::lock_guard<std::mutex> lock(SceneCacheMutex_);
                CachedDemo_ = request.Scene->Demo;
                CachedRevision_ = request.Scene->Revision;
                CachedTransformRevision_ = request.Scene->TransformRevision;
                CachedLightingRevision_ = request.Scene->LightingRevision;
                CachedScene_ = scene;
            }
        }
        if (!scene || cancellation.IsCancellationRequested()) return Cancelled();

        RayTracing::Camera camera;
        camera.Position = {
            request.Scene->SceneCamera.Position.X,
            request.Scene->SceneCamera.Position.Y,
            request.Scene->SceneCamera.Position.Z};
        camera.Target = {
            request.Scene->SceneCamera.Target.X,
            request.Scene->SceneCamera.Target.Y,
            request.Scene->SceneCamera.Target.Z};
        camera.Up = {
            request.Scene->SceneCamera.Up.X,
            request.Scene->SceneCamera.Up.Y,
            request.Scene->SceneCamera.Up.Z};
        camera.VerticalFieldOfViewDegrees = request.Scene->SceneCamera.VerticalFieldOfViewDegrees;

        RayTracing::RenderSettings settings;
        settings.Width = request.Settings.Width;
        settings.Height = request.Settings.Height;
        settings.SamplesPerPixel = 1;
        settings.MaxReflectionBounces = 2;
        settings.ShadowsEnabled = request.Settings.ShadowsEnabled;
        settings.ShadowStrength = 0.55f;
        settings.GpuPresentation=request.Settings.GpuPresentation;
        RayTracing::RenderJob job = RHI_->RenderAsync(scene, camera, settings);
        while (!job.IsReady())
        {
            if (cancellation.IsCancellationRequested()) job.Cancel();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const RayTracing::RenderResult rayResult = job.GetResult();
        RenderResult result;
        if (rayResult.Status == RayTracing::RenderStatus::Cancelled ||
            cancellation.IsCancellationRequested())
        {
            result.Status = RenderStatus::Cancelled;
        }
        else if (rayResult.Status == RayTracing::RenderStatus::Failed)
        {
            result.Status = RenderStatus::Failed;
            result.Error = rayResult.Error;
        }
        else
        {
            result.Status = RenderStatus::Completed;
            result.Output.GpuColor=rayResult.GpuColor;
            result.Output.Width = rayResult.GpuColor?request.Settings.Width:rayResult.Output.GetWidth();
            result.Output.Height = rayResult.GpuColor?request.Settings.Height:rayResult.Output.GetHeight();
            result.Output.PixelsRGBA8 = rayResult.Output.GetPixelsRGBA8();
            result = ApplyRayPresentationOverlay(
                request, cancellation, PresentationSceneCache_, std::move(result));
        }
        return result;
    }

private:
    BackendDescriptor Descriptor_;
    std::shared_ptr<RayTracing::IRayTracingRHI> RHI_;
    Rasterizer::SoftwareRasterizer PresentationRasterizer_;
    RasterSceneCache PresentationSceneCache_;
    std::mutex SceneCacheMutex_;
    DemoId CachedDemo_ = DemoId::Triangle;
    std::uint64_t CachedRevision_ = 0;
    std::uint64_t CachedTransformRevision_ = 0;
    std::uint64_t CachedLightingRevision_ = 0;
    std::shared_ptr<const RayTracing::Scene> CachedScene_;
};

[[nodiscard]] Renderer::RHIBackend RealtimeBackend(const std::string& id)
{
    if (id == "renderer.dx11") return Renderer::RHIBackend::DX11;
    if (id == "renderer.dx12") return Renderer::RHIBackend::DX12;
    if (id == "renderer.metal") return Renderer::RHIBackend::Metal;
    if (id == "renderer.opengles") return Renderer::RHIBackend::OpenGLES;
    return Renderer::RHIBackend::Vulkan;
}

[[nodiscard]] Rasterizer::RasterBackend RasterBackend(const std::string& id)
{
    if (id == "rasterizer.cuda") return Rasterizer::RasterBackend::CUDA;
    if (id == "rasterizer.compute") return Rasterizer::RasterBackend::ComputeShader;
    if (id == "rasterizer.metal") return Rasterizer::RasterBackend::Metal;
    return Rasterizer::RasterBackend::CPU;
}

struct CachedPickMesh
{
    std::weak_ptr<const MeshAsset> Asset;
    std::shared_ptr<Maths::TriMesh> Mesh;
};

struct PickCache
{
    std::mutex Mutex;
    std::unordered_map<const MeshAsset*, CachedPickMesh> Meshes;
    float LastAspect = 16.0f / 9.0f;
};

[[nodiscard]] std::shared_ptr<Maths::TriMesh> CachedTriMesh(
    PickCache& cache,
    const std::shared_ptr<const MeshAsset>& asset)
{
    std::lock_guard<std::mutex> lock(cache.Mutex);
    const auto found = cache.Meshes.find(asset.get());
    if (found != cache.Meshes.end() && found->second.Asset.lock() == asset)
    {
        return found->second.Mesh;
    }

    std::vector<Maths::Vector3> vertices;
    vertices.reserve(asset->Vertices.size());
    for (const Vertex& vertex : asset->Vertices)
    {
        vertices.push_back(ToMath(vertex.Position));
    }
    auto mesh = std::make_shared<Maths::TriMesh>();
    std::string error;
    if (!mesh->SetGeometry(std::move(vertices), asset->Indices, &error) ||
        !mesh->BuildAcceleration() ||
        !mesh->GetAcceleration().IsBuilt())
    {
        return nullptr;
    }
    cache.Meshes[asset.get()] = {asset, mesh};
    if (cache.Meshes.size() > 128)
    {
        for (auto iterator = cache.Meshes.begin(); iterator != cache.Meshes.end();)
        {
            iterator = iterator->second.Asset.expired()
                ? cache.Meshes.erase(iterator)
                : std::next(iterator);
        }
    }
    return mesh;
}

[[nodiscard]] float PickAspect(PickCache& cache, const PickRequest& request)
{
    float aspect = 0.0f;
    if (request.ViewportWidth > 0.0f && request.ViewportHeight > 0.0f)
    {
        aspect = request.ViewportWidth / request.ViewportHeight;
    }
    else if (request.NormalizedX > 1.0e-6f && request.NormalizedY > 1.0e-6f)
    {
        const float width = request.ViewportX / request.NormalizedX;
        const float height = request.ViewportY / request.NormalizedY;
        if (width > 0.0f && height > 0.0f) aspect = width / height;
    }
    std::lock_guard<std::mutex> lock(cache.Mutex);
    if (std::isfinite(aspect) && aspect > 0.05f && aspect < 20.0f)
    {
        cache.LastAspect = aspect;
    }
    return cache.LastAspect;
}

} // namespace

bool CookNaniteAssets(const SceneSnapshot& scene,std::size_t& families,std::string& error)
{
    // Reuse the editor's section conversion and demo framing. Offline cooking
    // must produce exactly the keys requested by the interactive renderer.
    CancellationSource cancellation;
    auto meshes=BuildRendererMeshes(scene,cancellation.Token(),true);
    std::vector<std::vector<const Renderer::MeshData*>> groups;
    for (const auto& mesh:meshes)
    {
        auto found=std::find_if(groups.begin(),groups.end(),[&](const auto& group)
        {return mesh.SelectionGroup && group.front()->SelectionGroup==mesh.SelectionGroup;});
        if (found==groups.end()) { groups.push_back({}); found=std::prev(groups.end()); }
        found->push_back(&mesh);
    }
    families=0;
    for (const auto& group:groups)
    {
        std::vector<Renderer::NaniteAssetSection> sections;
        for (const auto* mesh:group) sections.push_back({&mesh->Vertices,&mesh->Indices,Renderer::HashMeshGeometry(mesh->Vertices,mesh->Indices)});
        std::sort(sections.begin(),sections.end(),Renderer::NaniteAssetSectionLess);
        sections.erase(std::unique(sections.begin(),sections.end(),[](const auto& a,const auto& b)
        {return Renderer::EqualMeshGeometry(*a.Vertices,*a.Indices,*b.Vertices,*b.Indices);}),sections.end());
        Renderer::NaniteAssetInput input;
        if (!Renderer::AssembleNaniteAsset(sections,input,error)) return false;
        Renderer::NaniteHierarchy hierarchy; Renderer::NanitePages pages;
        if (!Renderer::LoadOrBuildNaniteAsset(input.Vertices,input.Indices,input.Materials,Techniques::DefaultNanitePhysicalPages,hierarchy,pages,error)) return false;
        ++families;
    }
    if (!families) { error="The selected demo has no Nanite geometry"; return false; }
    error.clear(); return true;
}

bool CookBakedLightingAssets(const SceneSnapshot& scene,const Techniques::Settings& settings,std::string& error)
{
    const auto directory=scene.ActiveBakedLightingDirectory();
    if (directory.empty()) { error="The scene has no baked lighting resource directory"; return false; }
    CancellationSource cancellation;
    auto meshes=BuildRendererMeshes(scene,cancellation.Token(),true);
    std::sort(meshes.begin(),meshes.end(),[](const auto& a,const auto& b) { return a.Id<b.Id; });
    const auto lights=BuildRendererSceneLights(scene,error); if (!lights) return false;
    Renderer::PreparedAreaLighting area;
    if (!area.Prepare(BuildRendererLighting(scene,lights),settings.AreaLights,error)) return false;
    const auto& lighting=area.GetLighting();
    std::vector<Renderer::SceneRayMeshView> views;
    std::vector<Renderer::LightmapSourceInstance> instances;
    std::cout<<"Preparing Sponza lightmap LODs and UV2 receivers...\n"<<std::flush;
    for (const auto& mesh:meshes)
    {
        views.emplace_back(mesh);
        auto geometry=std::make_shared<Renderer::LightmapSourceGeometry>(); geometry->Vertices=mesh.Vertices;
        Renderer::MeshLodChain chain;
        if (!Renderer::BuildMeshLods(mesh.Vertices,mesh.Indices,{},chain,error)) return false;
        for (auto& level:chain.Levels) geometry->Levels.push_back(std::move(level.Indices));
        Renderer::LightmapSourceInstance instance; instance.Geometry=std::move(geometry); instance.World=mesh.WorldTransform;
        instance.Color=mesh.Color; instance.Material=mesh.Material; instance.Visible=mesh.Visible; instance.CastShadow=mesh.CastShadow;
        instances.push_back(std::move(instance));
    }
    auto request=Renderer::MakeLightmapBakeRequest(std::move(instances),settings,lighting,true);
    auto receivers=std::make_shared<Renderer::LightmapReceivers>();
    if (!Renderer::BuildLightmapReceivers(request.Instances,request.Materials,request.Atlas,*receivers,error)) return false;
    std::cout<<"Baking Sponza lightmap ("<<request.Atlas.Resolution<<" x "<<request.Atlas.Resolution<<", "<<request.Sampling.RayCount<<" rays per hemisphere)...\n"<<std::flush;
    auto irradiance=std::make_shared<Renderer::BakedLightmap>();
    if (!Renderer::BakeLightmap(receivers->Atlas,*receivers->TransportScene,lighting,request.Sampling,*irradiance,error)) return false;
    Renderer::LightmapBakeResult result; result.Receivers=receivers; result.Irradiance=irradiance;
    if (!Renderer::SaveLightmapAsset(directory/"irradiance.evlightmap",request,result,error)) return false;
    std::cout<<"Baking Sponza light probes...\n"<<std::flush;
    Renderer::BakedProbeCache probes;
    if (!probes.Update(views,1,settings,lighting,true,error) || !probes.Samples()) return false;
    return Renderer::SaveProbeAsset(directory/"irradiance.evprobes",
        Renderer::ProbeAssetKey(views,settings,lighting,true),*probes.Samples(),error);
}

void RegisterModuleAdapters(EditorCore& editor)
{
    for (const BackendDescriptor& descriptor : StandardBackendCatalog())
    {
        if (!IsBackendSupportedOnPlatform(descriptor, CurrentHostPlatform()))
        {
            continue;
        }
        std::shared_ptr<IRenderAdapter> adapter;
        if (descriptor.Id == "renderer.dxr")
        {
            adapter = std::make_shared<RayRenderAdapter>(
                descriptor,
                std::make_shared<RayTracing::DxrRayTracingRHI>());
        }
        else if (descriptor.Domain == RenderDomain::RealtimeRenderer)
        {
            adapter = std::make_shared<RealtimeRenderAdapter>(
                descriptor, RealtimeBackend(descriptor.Id));
        }
        else if (descriptor.Domain == RenderDomain::SoftwareRasterizer)
        {
            adapter = std::make_shared<RasterRenderAdapter>(
                descriptor, RasterBackend(descriptor.Id));
        }
        if (adapter->IsAvailable())
        {
            editor.RegisterRenderAdapter(std::move(adapter));
        }
    }
    const std::vector<BackendDescriptor> available = editor.Backends();
    const auto selected = std::find_if(
        available.begin(),
        available.end(),
        [&editor](const BackendDescriptor& descriptor)
        {
            return descriptor.Id == editor.SelectedBackend();
        });
    if (selected == available.end() && !available.empty())
    {
        editor.SelectBackend(available.front().Id);
    }
}

PickHandler CreatePickHandler()
{
    const std::shared_ptr<PickCache> cache = std::make_shared<PickCache>();
    return [cache](const SceneSnapshot& scene, const PickRequest& request)
    {
        const float aspect = PickAspect(*cache, request);
        const float halfHeight = std::tan(
            scene.SceneCamera.VerticalFieldOfViewDegrees * Pi / 360.0f);
        const float screenX = (std::clamp(request.NormalizedX, 0.0f, 1.0f) * 2.0f - 1.0f)
            * halfHeight * aspect;
        const float screenY = (1.0f - std::clamp(request.NormalizedY, 0.0f, 1.0f) * 2.0f)
            * halfHeight;
        const Maths::Vector3 origin = ToMath(scene.SceneCamera.Position);
        const Maths::Vector3 forward =
            (ToMath(scene.SceneCamera.Target) - origin).Normalized();
        const Maths::Vector3 right = forward.Cross(ToMath(scene.SceneCamera.Up)).Normalized();
        const Maths::Vector3 up = right.Cross(forward).Normalized();
        if (forward.LengthSquared() <= 1.0e-8f || right.LengthSquared() <= 1.0e-8f)
        {
            return SceneNodeId{0};
        }
        const Maths::Ray worldRay(origin, forward + right * screenX + up * screenY);

        float closestDistance = std::numeric_limits<float>::infinity();
        SceneNodeId closestNode = 0;
        for (const SceneNode& node : scene.Nodes)
        {
            if (!node.Visible || !node.Selectable || !node.Mesh) continue;
            const std::shared_ptr<Maths::TriMesh> mesh = CachedTriMesh(*cache, node.Mesh);
            if (!mesh) continue;
            const Maths::Transform transform = ToMath(node.LocalTransform);
            const Maths::Vector3 localOrigin = transform.InverseTransformPoint(worldRay.Origin);
            const Maths::Vector3 localDirection = transform.InverseTransformVector(worldRay.Direction);
            if (localDirection.LengthSquared() <= 1.0e-12f) continue;

            Maths::TriMeshHit hit;
            if (!mesh->RayCast(Maths::Ray(localOrigin, localDirection), hit)) continue;
            const Maths::Vector3 worldHit = transform.TransformPoint(hit.Position);
            const float worldDistance = (worldHit - worldRay.Origin).Dot(worldRay.Direction);
            if (worldDistance >= 0.0f && worldDistance < closestDistance)
            {
                closestDistance = worldDistance;
                closestNode = node.Id;
            }
        }
        return closestNode;
    };
}

} // namespace Evans::Editor
