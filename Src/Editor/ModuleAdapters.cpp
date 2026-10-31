#include "Editor/ModuleAdapters.h"

#include "Maths/EvansMaths.h"
#include "Rasterizer/Rasterizer.hpp"
#include "Rasterizer/SoftwareRasterCore.hpp"
#include "Renderer/DxrRayTracingRHI.h"
#include "Renderer/EvansRenderer.h"

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

[[nodiscard]] Image ToEditorImage(const Renderer::FrameCapture& capture)
{
    Image result;
    if (!capture.HasColorOutput()) return result;
    result.Width = capture.Width;
    result.Height = capture.Height;
    result.PixelsRGBA8 = capture.ColorRGBA8;
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
    const MeshAsset& mesh,
    std::size_t firstIndex)
{
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
    std::unordered_map<const ImageAsset*, std::shared_ptr<const Renderer::ImageData>>;

[[nodiscard]] std::shared_ptr<const Renderer::ImageData> ToRendererImage(
    const std::shared_ptr<const ImageAsset>& source,
    RendererImageCache& cache,
    const CancellationToken& cancellation)
{
    if (!source || !source->IsValid() || cancellation.IsCancellationRequested())
    {
        return {};
    }
    const auto found = cache.find(source.get());
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
    cache.emplace(source.get(), result);
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
    result.AlphaCutoff =
        source->OpacityImage && source->OpacityImage->IsValid() ? 0.5f : 0.0f;
    result.DiffuseImage = ToRendererImage(
        source->DiffuseImage, imageCache, cancellation);
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

[[nodiscard]] ShadowFrame BuildShadowFrame(
    const SceneSnapshot& scene,
    bool useRealtimeRendererDirection = false)
{
    Maths::AABB bounds;
    for (const SceneNode& node : scene.Nodes)
    {
        if (!node.Visible || !node.Mesh) continue;
        const Maths::Transform transform = ToMath(node.LocalTransform);
        for (const Vertex& vertex : node.Mesh->Vertices)
        {
            bounds.Expand(transform.TransformPoint(ToMath(vertex.Position)));
        }
    }
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
    std::unordered_map<const ImageAsset*, std::uint32_t>;

[[nodiscard]] std::uint32_t AddRasterTexture(
    const std::shared_ptr<const ImageAsset>& source,
    Rasterizer::RasterScene& scene,
    RasterTextureCache& cache)
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
};

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

    std::size_t triangleCount = 0;
    for (const SceneNode& node : scene.Nodes)
    {
        if (node.Visible && node.Mesh) triangleCount += node.Mesh->Indices.size() / 3;
    }
    result.Scene.Triangles.reserve(triangleCount);
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
            const Material* material = MaterialForTriangle(scene, mesh, firstIndex);
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
                    textureCache);
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
            ClipSpace != clipSpace || SelectedNode != settings.SelectedNode;
        const bool reproject = clipSpace && (CameraRevision != scene.CameraRevision ||
            Width != settings.Width || Height != settings.Height);
        if (rebuild || reproject)
        {
            Valid = false;
            if (rebuild)
                Built = BuildRasterScene(scene, settings, cancellation, clipSpace);
            else
                Built.WasCancelled = !UpdateRasterProjection(Built.Scene, scene, settings, cancellation);
            if (!Built.WasCancelled && !cancellation.IsCancellationRequested())
            {
                Demo = scene.Demo;
                Revision = scene.Revision;
                CameraRevision = scene.CameraRevision;
                Width = settings.Width;
                Height = settings.Height;
                SelectedNode = settings.SelectedNode;
                ClipSpace = clipSpace;
                Valid = true;
            }
        }
        return Built;
    }

    bool Valid = false;
    bool ClipSpace = false;
    DemoId Demo = DemoId::Triangle;
    std::uint64_t Revision = 0;
    std::uint64_t CameraRevision = 0;
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    SceneNodeId SelectedNode = 0;
    RasterSceneBuild Built;
};

[[nodiscard]] Image ToEditorImage(const Rasterizer::RenderTarget& target)
{
    Image result;
    result.Width = target.Width();
    result.Height = target.Height();
    const std::vector<std::uint32_t> packed = target.ToRGBA8();
    result.PixelsRGBA8.resize(packed.size() * 4);
    for (std::size_t pixel = 0; pixel < packed.size(); ++pixel)
    {
        result.PixelsRGBA8[pixel * 4] = static_cast<std::uint8_t>(packed[pixel] & 0xffU);
        result.PixelsRGBA8[pixel * 4 + 1] = static_cast<std::uint8_t>((packed[pixel] >> 8U) & 0xffU);
        result.PixelsRGBA8[pixel * 4 + 2] = static_cast<std::uint8_t>((packed[pixel] >> 16U) & 0xffU);
        result.PixelsRGBA8[pixel * 4 + 3] = static_cast<std::uint8_t>((packed[pixel] >> 24U) & 0xffU);
    }
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
    result.Intensity = source.Intensity;
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
        rasterSettings.Mode = ToRasterMode(request.Settings.Mode);
        rasterSettings.InputSpace = Rasterizer::VertexInputSpace::Local;
        rasterSettings.DepthRange = Rasterizer::ClipDepthRange::ZeroToOne;
        const LightSelection lightSelection = SelectLights(*request.Scene);
        rasterSettings.Shading = Rasterizer::ShadingModel::Phong;
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
            1000.0f);
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
        rasterSettings.OutlineEnabled = request.Settings.OutlineEnabled;
        Rasterizer::RenderTarget target(request.Settings.Width, request.Settings.Height);
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
        result.Output = ToEditorImage(target);
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

        const Image overlay = ToEditorImage(target);
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

    Renderer::MeshData mesh;
    mesh.Id = "Node." + std::to_string(node.Id) +
              ".Section." + std::to_string(sectionOrdinal);
    mesh.SelectionGroup = node.Id;
    mesh.Visible = node.Visible;
    mesh.Color = {1.0f, 1.0f, 1.0f, 1.0f};
    mesh.WorldTransform = ToRendererMatrix(ToMath(node.LocalTransform).ToMatrix());
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
    bool fullMaterial)
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
    RendererImageCache imageCache;
    for (const SceneNode& node : scene.Nodes)
    {
        if (cancellation.IsCancellationRequested()) return {};
        if (!node.Mesh || node.Mesh->Vertices.empty() || node.Mesh->Indices.empty()) continue;

        if (fullMaterial)
        {
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
                const Material* material = sectionIndex < node.Mesh->Sections.size()
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

        Renderer::MeshData mesh;
        mesh.Id = "Node." + std::to_string(node.Id);
        mesh.SelectionGroup = node.Id;
        mesh.Visible = node.Visible;
        mesh.Indices = node.Mesh->Indices;
        mesh.Vertices.reserve(node.Mesh->Vertices.size());
        constexpr Color defaultColor{0.72f, 0.76f, 0.82f, 1.0f};
        mesh.Color = {1.0f, 1.0f, 1.0f, 1.0f};
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
            const Material* material = FindMaterial(scene, section.MaterialName);
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
        mesh.WorldTransform = ToRendererMatrix(ToMath(node.LocalTransform).ToMatrix());
        result.push_back(std::move(mesh));
    }
    return result;
}

[[nodiscard]] Renderer::SceneLight ToRendererLight(const Light& source)
{
    Renderer::SceneLight result;
    result.Position = {source.Position.X, source.Position.Y, source.Position.Z};
    result.Color = {
        source.LightColor.R, source.LightColor.G, source.LightColor.B};
    result.Intensity = source.Intensity;
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

[[nodiscard]] Renderer::LightingState BuildRendererLighting(
    const SceneSnapshot& scene)
{
    Renderer::LightingState result;
    result.CameraPosition = {
        scene.SceneCamera.Position.X,
        scene.SceneCamera.Position.Y,
        scene.SceneCamera.Position.Z};
    result.AmbientColor = {1.0f, 1.0f, 1.0f};
    result.AmbientStrength = 0.08f;

    const LightSelection lightSelection = SelectLights(scene);
    for (std::size_t lightIndex = 0; lightIndex < lightSelection.Count; ++lightIndex)
    {
        const std::size_t sceneIndex = lightSelection.SceneIndices[lightIndex];
        result.Lights[lightIndex] = ToRendererLight(scene.Lights[sceneIndex]);
    }
    result.LightCount = static_cast<std::uint32_t>(lightSelection.Count);
    result.ShadowLightIndex = lightSelection.ShadowLightIndex;
    return result;
}

struct RealtimeCacheKey
{
    DemoId Demo = DemoId::Triangle;
    std::uint64_t Revision = 0;
    std::uint64_t CameraRevision = 0;
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    FillMode Mode = FillMode::Solid;
    bool ShadowsEnabled = false;
    bool OutlineEnabled = false;
    SceneNodeId SelectedNode = 0;

    friend bool operator==(const RealtimeCacheKey& left, const RealtimeCacheKey& right)
    {
        return left.Demo == right.Demo &&
               left.Revision == right.Revision &&
               left.CameraRevision == right.CameraRevision &&
               left.Width == right.Width &&
               left.Height == right.Height &&
               left.Mode == right.Mode &&
               left.ShadowsEnabled == right.ShadowsEnabled &&
               left.OutlineEnabled == right.OutlineEnabled &&
               left.SelectedNode == right.SelectedNode;
    }
};

[[nodiscard]] RealtimeCacheKey MakeCacheKey(const RenderRequest& request)
{
    RealtimeCacheKey key;
    key.Demo = request.Scene->Demo;
    key.Revision = request.Scene->Revision;
    key.CameraRevision = request.Scene->CameraRevision;
    key.Width = request.Settings.Width;
    key.Height = request.Settings.Height;
    key.Mode = request.Settings.Mode;
    key.ShadowsEnabled = request.Settings.ShadowsEnabled;
    key.OutlineEnabled = request.Settings.OutlineEnabled;
    key.SelectedNode = request.Settings.SelectedNode;
    return key;
}

struct RealtimeSceneCacheKey
{
    DemoId Demo = DemoId::Triangle;
    std::uint64_t Revision = 0;
    Renderer::RHIBackend Backend = Renderer::RHIBackend::Vulkan;
    bool FullMaterial = false;

    friend bool operator==(
        const RealtimeSceneCacheKey& left,
        const RealtimeSceneCacheKey& right)
    {
        return left.Demo == right.Demo &&
               left.Revision == right.Revision &&
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
    std::mutex RenderMutex;
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
            if (CacheValid_ && key == CacheKey_) return CachedResult_;
        }
        if (!Context_.ThreadStarted || !Context_.Thread.IsRunning())
        {
            return Failed("The Renderer RHI thread could not be started");
        }
        std::lock_guard<std::mutex> renderLock(Context_.RenderMutex);
        if (cancellation.IsCancellationRequested()) return Cancelled();
        const Renderer::RHIBackend backend = Context_.Backend;
        const bool fullMaterial =
            backend == Renderer::RHIBackend::DX11 ||
            backend == Renderer::RHIBackend::DX12 ||
            backend == Renderer::RHIBackend::OpenGLES ||
            backend == Renderer::RHIBackend::Vulkan ||
            backend == Renderer::RHIBackend::Metal;
        const RealtimeSceneCacheKey sceneKey =
            MakeSceneCacheKey(request, backend, fullMaterial);

        struct ConfigureResult
        {
            bool Success = false;
            bool Cancelled = false;
            bool SceneInvalidated = false;
            std::string Error;
        };
        std::shared_ptr<const std::vector<Renderer::MeshData>> meshes;
        bool convertedSceneCandidate = false;
        if (SceneCacheValid_ && sceneKey == SceneCacheKey_)
        {
            meshes = SceneCacheMeshes_;
        }
        else
        {
            meshes = std::make_shared<std::vector<Renderer::MeshData>>(
                BuildRendererMeshes(
                    *request.Scene,
                    cancellation,
                    fullMaterial));
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
            1000.0f);
        const Renderer::Matrix4 viewProjection = ToRendererMatrix(projection * view);
        const Renderer::LightingState lighting = BuildRendererLighting(*request.Scene);
        ShadowFrame shadowFrame;
        if (settings.ShadowsEnabled && settings.Mode != FillMode::Wireframe)
        {
            if (!ShadowCacheValid_ || !(sceneKey == ShadowCacheKey_))
            {
                CachedShadow_ = BuildShadowFrame(*request.Scene, !fullMaterial);
                ShadowCacheKey_ = sceneKey;
                ShadowCacheValid_ = true;
            }
            shadowFrame = CachedShadow_;
        }
        const Renderer::Matrix4 lightViewProjection = ToRendererMatrix(
            shadowFrame.Valid ? shadowFrame.ViewProjection : Maths::Matrix4::Identity());
        const bool enqueued = Context_.Thread.Enqueue(
            [meshes, configured, settings, backend, viewProjection, lightViewProjection, lighting,
             hasShadowFrame = shadowFrame.Valid, uploadScene, cancellation](
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
                        createInfo.EnableValidation = false;
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

                    const bool replaceScene = uploadScene || rendererRecreated;
                    if (replaceScene)
                    {
                        renderer.ClearScene();
                        result.SceneInvalidated = true;
                    }
                    renderer.SetAutomaticViewProjection(false);
                    renderer.SetViewProjection(viewProjection);
                    renderer.SetLightViewProjection(lightViewProjection);
                    renderer.SetLighting(lighting);
                    renderer.SetRenderMode(ToRendererMode(settings.Mode));
                    renderer.SetShadowsEnabled(settings.ShadowsEnabled && hasShadowFrame);
                    if (replaceScene)
                    {
                        for (const Renderer::MeshData& mesh : *meshes)
                        {
                            if (cancellation.IsCancellationRequested())
                            {
                                result.Cancelled = true;
                                configured->set_value(std::move(result));
                                return;
                            }
                            if (!renderer.AddMesh(mesh))
                            {
                                result.Error = renderer.GetLastError();
                                configured->set_value(std::move(result));
                                return;
                            }
                        }
                    }
                    for (const Renderer::MeshData& mesh : *meshes)
                    {
                        if (cancellation.IsCancellationRequested())
                        {
                            result.Cancelled = true;
                            configured->set_value(std::move(result));
                            return;
                        }
                        Renderer::OutlineSettings outline;
                        outline.Enabled = settings.OutlineEnabled &&
                                          settings.SelectedNode != 0 &&
                                          mesh.SelectionGroup == settings.SelectedNode;
                        if (!renderer.SetOutline(mesh.Id, outline))
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
        if (configureResult.Success)
        {
            UploadedSceneKey_ = sceneKey;
            UploadedSceneValid_ = true;
            if (convertedSceneCandidate)
            {
                SceneCacheKey_ = sceneKey;
                SceneCacheMeshes_ = meshes;
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
            Image Output;
            std::string Error;
        };
        const std::shared_ptr<std::promise<ReadbackResult>> readback =
            std::make_shared<std::promise<ReadbackResult>>();
        std::future<ReadbackResult> readbackFuture = readback->get_future();
        if (!Context_.Thread.Enqueue(
                [readback](Renderer::Renderer& renderer)
                {
                    ReadbackResult result;
                    try
                    {
                        result.Output = ToEditorImage(
                            renderer.GetLastFrameCaptureView());
                        result.Error = renderer.GetLastError();
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
        if (!preview.Output.IsValid())
        {
            return Failed("The realtime RHI returned an invalid color framebuffer");
        }
        if (preview.Status == RenderStatus::Completed)
        {
            std::lock_guard<std::mutex> lock(CacheMutex_);
            CacheKey_ = key;
            CachedResult_ = preview;
            CacheValid_ = true;
        }
        return preview;
    }

private:
    BackendDescriptor Descriptor_;
    RealtimeBackendContext Context_;
    std::mutex CacheMutex_;
    bool CacheValid_ = false;
    RealtimeCacheKey CacheKey_;
    RenderResult CachedResult_;
    bool SceneCacheValid_ = false;
    RealtimeSceneCacheKey SceneCacheKey_;
    std::shared_ptr<const std::vector<Renderer::MeshData>> SceneCacheMeshes_;
    bool UploadedSceneValid_ = false;
    RealtimeSceneCacheKey UploadedSceneKey_;
    bool ShadowCacheValid_ = false;
    RealtimeSceneCacheKey ShadowCacheKey_;
    ShadowFrame CachedShadow_;
};

class RasterRenderAdapter final : public IRenderAdapter
{
public:
    RasterRenderAdapter(BackendDescriptor descriptor, Rasterizer::RasterBackend backend)
        : Descriptor_(std::move(descriptor)), Rasterizer_(backend)
    {
    }

    [[nodiscard]] BackendDescriptor Descriptor() const override { return Descriptor_; }

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
    const MeshAsset& mesh,
    std::size_t firstIndex)
{
    const Material* material = MaterialForTriangle(source, mesh, firstIndex);
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

    std::size_t triangleCount = 0;
    for (const SceneNode& node : source.Nodes)
    {
        if (node.Visible && node.Mesh) triangleCount += node.Mesh->Indices.size() / 3;
    }
    scene->Triangles.reserve(triangleCount);
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
            triangle.MaterialIndex = RayMaterialIndex(source, mesh, firstIndex);
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
            light.Intensity = sourceLight.Intensity;
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
            light.Intensity = sourceLight.Intensity;
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
            light.Intensity = sourceLight.Intensity;
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
                CachedRevision_ == request.Scene->Revision)
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
            result.Output.Width = rayResult.Output.GetWidth();
            result.Output.Height = rayResult.Output.GetHeight();
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
