#include "Techniques/LightRange.h"
#include "Rasterizer/GpuRasterCommon.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace Evans::Rasterizer::Detail
{
namespace
{

void CopyColor(float (&destination)[4], const Color& source) noexcept
{
    destination[0] = source.R;
    destination[1] = source.G;
    destination[2] = source.B;
    destination[3] = source.A;
}

void CopyMatrix(float (&destination)[16], const Matrix4& source) noexcept
{
    std::copy(source.Values.begin(), source.Values.end(), destination);
}

void CopyLight(GpuRasterLight& destination, const RasterLight& source) noexcept
{
    destination.PositionType[0] = source.Position.X;
    destination.PositionType[1] = source.Position.Y;
    destination.PositionType[2] = source.Position.Z;
    destination.PositionType[3] = static_cast<float>(source.Type);
    destination.DirectionIntensity[0] = source.Direction.X;
    destination.DirectionIntensity[1] = source.Direction.Y;
    destination.DirectionIntensity[2] = source.Direction.Z;
    destination.DirectionIntensity[3] = source.Intensity;
    destination.ColorCastShadows[0] = source.LightColor.R;
    destination.ColorCastShadows[1] = source.LightColor.G;
    destination.ColorCastShadows[2] = source.LightColor.B;
    destination.ColorCastShadows[3] = source.CastShadows ? 1.0f : 0.0f;
    destination.AreaSizePadding[0] = source.AreaSize.X;
    destination.AreaSizePadding[1] = source.AreaSize.Y;
    destination.AreaSizePadding[2] = Techniques::CanonicalLightRange(source.Range);
}

[[nodiscard]] Float4 TransformForBinning(
    const RasterVertex& vertex,
    const RasterSettings& settings,
    bool lightPass) noexcept
{
    if (settings.InputSpace == VertexInputSpace::Clip)
    {
        return lightPass ? vertex.ShadowClipPosition : vertex.ClipPosition;
    }
    const Float4 local{
        vertex.Position.X,
        vertex.Position.Y,
        vertex.Position.Z,
        1.0f};
    const Float4 world = settings.World * local;
    return lightPass
               ? settings.LightViewProjection * world
               : settings.Projection * (settings.View * world);
}

[[nodiscard]] std::uint32_t TextureTexelCount(const RasterScene& scene)
{
    if (scene.Textures.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::overflow_error("GPU raster texture table is too large");
    }

    std::uint64_t texelCount = 0;
    for (const RasterTexture& texture : scene.Textures)
    {
        if (!texture.IsValid())
        {
            continue;
        }
        texelCount += static_cast<std::uint64_t>(texture.Width) * texture.Height;
        if (texture.MipPixelsRGBA8) texelCount += texture.MipPixelsRGBA8->size() / 4;
        if (texelCount > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error("GPU raster texture atlas is too large");
        }
    }
    return static_cast<std::uint32_t>(texelCount);
}

} // namespace

bool IsGpuRasterCancelled(
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation) noexcept
{
    return externalCancellation.IsCancellationRequested() ||
           backendCancellation.load(std::memory_order_relaxed);
}

std::vector<GpuRasterTriangle> BuildGpuRasterTriangles(
    const RasterScene& scene)
{
    std::vector<GpuRasterTriangle> result(scene.Triangles.size());
    for (std::size_t triangleIndex = 0;
         triangleIndex < scene.Triangles.size();
         ++triangleIndex)
    {
        const RasterTriangle& source = scene.Triangles[triangleIndex];
        GpuRasterTriangle& destination = result[triangleIndex];
        for (std::size_t vertexIndex = 0; vertexIndex < 3; ++vertexIndex)
        {
            const RasterVertex& vertex = source.Vertices[vertexIndex];
            GpuRasterVertex& gpuVertex = destination.Vertices[vertexIndex];
            const float clip[] = {
                vertex.ClipPosition.X,
                vertex.ClipPosition.Y,
                vertex.ClipPosition.Z,
                vertex.ClipPosition.W};
            const float color[] = {
                vertex.VertexColor.R,
                vertex.VertexColor.G,
                vertex.VertexColor.B,
                vertex.VertexColor.A};
            const float position[] = {
                vertex.Position.X,
                vertex.Position.Y,
                vertex.Position.Z,
                1.0f};
            const float normal[] = {
                vertex.Normal.X,
                vertex.Normal.Y,
                vertex.Normal.Z,
                0.0f};
            const float shadowClip[] = {
                vertex.ShadowClipPosition.X,
                vertex.ShadowClipPosition.Y,
                vertex.ShadowClipPosition.Z,
                vertex.ShadowClipPosition.W};
            std::copy(std::begin(clip), std::end(clip), gpuVertex.Clip);
            std::copy(std::begin(color), std::end(color), gpuVertex.Color);
            std::copy(
                std::begin(position),
                std::end(position),
                gpuVertex.Position);
            std::copy(std::begin(normal), std::end(normal), gpuVertex.Normal);
            std::copy(
                std::begin(shadowClip),
                std::end(shadowClip),
                gpuVertex.ShadowClip);
            gpuVertex.TexCoord[0] = vertex.TexCoord.X;
            gpuVertex.TexCoord[1] = vertex.TexCoord.Y;
            gpuVertex.TexCoord[2] = vertex.LightmapUV.X;
            gpuVertex.TexCoord[3] = vertex.LightmapUV.Y;
        }
        CopyColor(destination.BaseColor, source.Material.BaseColor);
        CopyColor(destination.SpecularColor, source.Material.SpecularColor);
        CopyColor(destination.EmissiveColor, source.Material.EmissiveColor);
        destination.Shininess = source.Material.Shininess;
        destination.AlphaCutoff = source.Material.AlphaCutoff;
        destination.DiffuseTextureIndex =
            source.Material.DiffuseTextureIndex;
        destination.OpacityTextureIndex =
            source.Material.OpacityTextureIndex;
        destination.Selected = source.Selected ? 1U : 0U;
        destination.NormalTextureIndex = source.Material.NormalTextureIndex;
        destination.SubsurfaceWeight = source.Material.SubsurfaceWeight;
        destination.ReceivesLightmap = source.Material.ReceivesLightmap ? 1U : 0U;
    }
    return result;
}

GpuRasterTextureData BuildGpuRasterTextures(const RasterScene& scene)
{
    GpuRasterTextureData result;
    const std::uint32_t texelCount = TextureTexelCount(scene);
    result.Descriptors.resize(scene.Textures.size());
    result.TexelsRGBA8.reserve(texelCount);

    auto pack = [&](GpuRasterTexture& descriptor, const std::vector<std::uint8_t>& bytes,
                    std::size_t offset, std::uint32_t width, std::uint32_t height) {
        const std::size_t texels = std::size_t(width) * height;
        if (offset > bytes.size() || texels > (bytes.size() - offset) / 4)
            throw std::invalid_argument("GPU raster texture mip data is incomplete");
        if (texels > std::numeric_limits<std::uint32_t>::max() ||
            result.TexelsRGBA8.size() > std::numeric_limits<std::uint32_t>::max() - texels)
            throw std::overflow_error("GPU raster texture atlas exceeds 32-bit indexing");
        descriptor = {static_cast<std::uint32_t>(result.TexelsRGBA8.size()), width, height, 0};
        for (std::size_t texel = 0; texel < texels; ++texel)
        {
            const auto* rgba = bytes.data() + offset + texel * 4;
            result.TexelsRGBA8.push_back(static_cast<std::uint32_t>(rgba[0]) |
                (static_cast<std::uint32_t>(rgba[1]) << 8U) | (static_cast<std::uint32_t>(rgba[2]) << 16U) |
                (static_cast<std::uint32_t>(rgba[3]) << 24U));
        }
    };
    for (std::size_t index = 0; index < scene.Textures.size(); ++index)
    {
        const auto& texture = scene.Textures[index];
        if (!texture.IsValid()) continue;
        pack(result.Descriptors[index], *texture.PixelsRGBA8, 0, texture.Width, texture.Height);
        if (!texture.MipPixelsRGBA8 || (texture.Width == 1 && texture.Height == 1)) continue;
        result.Descriptors[index].FirstMip = static_cast<std::uint32_t>(result.Descriptors.size());
        std::uint32_t width = texture.Width, height = texture.Height;
        std::size_t offset = 0;
        while (width > 1 || height > 1)
        {
            width = std::max(1U, width / 2);
            height = std::max(1U, height / 2);
            result.Descriptors.emplace_back();
            pack(result.Descriptors.back(), *texture.MipPixelsRGBA8, offset, width, height);
            offset += std::size_t(width) * height * 4;
        }
    }
    return result;
}

const GpuRasterTextureData& GpuRasterTextureCache::Get(const RasterScene& scene)
{
    bool changed=Generation_==0 || Sources_.size()!=scene.Textures.size();
    for(std::size_t i=0;!changed && i<Sources_.size();++i)
        changed=Sources_[i].Width!=scene.Textures[i].Width || Sources_[i].Height!=scene.Textures[i].Height ||
            Sources_[i].PixelsRGBA8!=scene.Textures[i].PixelsRGBA8 ||
            Sources_[i].MipPixelsRGBA8!=scene.Textures[i].MipPixelsRGBA8;
    if(changed)
    {
        auto data=BuildGpuRasterTextures(scene);
        Sources_=scene.Textures;Data_=std::move(data);++Generation_;
    }
    return Data_;
}

GpuRasterConstants BuildGpuRasterConstants(
    const RasterScene& scene,
    const RasterSettings& settings,
    const RenderTarget& target,
    bool shadowsEnabled,
    std::uint32_t shadowWidth,
    std::uint32_t shadowHeight)
{
    GpuRasterConstants constants;
    constants.MaterialOptions = (settings.TextureMapping ? 1U : 0U) |
        (settings.NormalMapping ? 2U : 0U) | (settings.AlphaTesting ? 4U : 0U) |
        (settings.AnisotropicFiltering ? 8U : 0U);
    constants.FrontLightmap = settings.FrontLightmap;
    constants.BackLightmap = settings.BackLightmap;
    constants.ACESColor = settings.ACESColor ? 1U : 0U;
    constants.DisplayParameters[0] = settings.Exposure;
    constants.DisplayParameters[1] = settings.Contrast;
    constants.DisplayParameters[2] = settings.Saturation;
    constants.DisplayParameters[3] = settings.LightmapIntensity;
    constants.SkyTexture = settings.SkyTexture;
    constants.SkyDiffuseTexture = settings.SkyDiffuseTexture;
    constants.SkyIntensity = settings.SkyIntensity;
    constants.SkyDiffuseIntensity = settings.SkyDiffuseIntensity;
    constants.SubsurfaceScattering[0] = settings.SubsurfaceScattering.X;
    constants.SubsurfaceScattering[1] = settings.SubsurfaceScattering.Y;
    constants.SubsurfaceScattering[2] = settings.SubsurfaceScattering.Z;
    CopyMatrix(constants.World, settings.World);
    CopyMatrix(constants.View, settings.View);
    CopyMatrix(constants.Projection, settings.Projection);
    CopyMatrix(constants.LightViewProjection, settings.LightViewProjection);
    constants.Width = target.Width();
    constants.Height = target.Height();
    constants.TriangleCount =
        static_cast<std::uint32_t>(scene.Triangles.size());
    constants.InputSpace = static_cast<std::uint32_t>(settings.InputSpace);
    constants.DepthRange = static_cast<std::uint32_t>(settings.DepthRange);
    constants.FillModeValue = static_cast<std::uint32_t>(settings.Mode);
    constants.ShadingValue = static_cast<std::uint32_t>(settings.Shading);
    constants.BackFaceCulling = settings.BackFaceCulling ? 1U : 0U;
    constants.FrontFaceCounterClockwise =
        settings.FrontFaceCounterClockwise ? 1U : 0U;
    constants.OutlineEnabled = (settings.OutlineEnabled ? 1U : 0U) | (settings.GpuPresentation && settings.OutlineEnabled ? 2U : 0U);
    constants.ShadowsEnabled = shadowsEnabled ? 1U : 0U;
    constants.ShadowWidth = shadowWidth;
    constants.ShadowHeight = shadowHeight;
    constants.TilesX = (target.Width() + 15U) / 16U;
    constants.WireframeWidth = settings.WireframeWidth;
    constants.OutlineWidth = settings.OutlineWidth;
    constants.ShadowBias = settings.ShadowBias;
    constants.ShadowRasterBias[0] = std::max(settings.ShadowSlopeScaledBias, 0.0f);
    constants.ShadowRasterBias[2] = std::max(settings.ShadowReceiverSlopeBias, 0.0f);
    constants.ShadowStrength = settings.ShadowStrength;
    constants.DepthTest = settings.DepthTest ? 1U : 0U;
    constants.ShadowPCFRadius = settings.ShadowPCFRadius;
    CopyColor(constants.ClearColor, settings.ClearColor);
    CopyColor(constants.WireframeColor, settings.WireframeColor);
    CopyColor(constants.OutlineColor, settings.OutlineColor);
    constants.CameraPosition[0] = settings.CameraPosition.X;
    constants.CameraPosition[1] = settings.CameraPosition.Y;
    constants.CameraPosition[2] = settings.CameraPosition.Z;
    constants.CameraPosition[3] = 1.0f;
    constants.LightDirectionIntensity[0] = settings.Light.DirectionToLight.X;
    constants.LightDirectionIntensity[1] = settings.Light.DirectionToLight.Y;
    constants.LightDirectionIntensity[2] = settings.Light.DirectionToLight.Z;
    constants.LightDirectionIntensity[3] = settings.Light.Intensity;
    constants.LightColorAmbientStrength[0] = settings.Light.LightColor.R;
    constants.LightColorAmbientStrength[1] = settings.Light.LightColor.G;
    constants.LightColorAmbientStrength[2] = settings.Light.LightColor.B;
    constants.LightColorAmbientStrength[3] = settings.AmbientStrength;
    CopyColor(constants.AmbientColor, settings.AmbientColor);
    constants.LightCount = EffectiveRasterLightCount(settings);
    constants.ShadowLightIndex =
        settings.ShadowLightIndex >= 0 &&
                static_cast<std::uint32_t>(settings.ShadowLightIndex) < constants.LightCount
            ? settings.ShadowLightIndex
            : -1;
    for (std::uint32_t index = 0; index < constants.LightCount; ++index)
    {
        CopyLight(constants.Lights[index], EffectiveRasterLight(settings, index));
    }
    constants.TextureCount = static_cast<std::uint32_t>(scene.Textures.size());
    for (const auto& texture : scene.Textures)
        if (texture.IsValid() && texture.MipPixelsRGBA8)
            for (auto extent = std::max(texture.Width, texture.Height); extent > 1; extent /= 2) ++constants.TextureCount;
    constants.TextureTexelCount = TextureTexelCount(scene);
    return constants;
}

GpuRasterTileBins BuildGpuRasterTileBins(
    const RasterScene& scene,
    const RasterSettings& settings,
    std::uint32_t width,
    std::uint32_t height,
    bool lightPass,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation)
{
    constexpr std::uint32_t TileSize = 16U;
    struct Bounds
    {
        int MinimumX = 0;
        int MinimumY = 0;
        int MaximumX = -1;
        int MaximumY = -1;
        bool Valid = false;
    };

    GpuRasterTileBins result;
    result.TilesX = (width + TileSize - 1U) / TileSize;
    result.TilesY = (height + TileSize - 1U) / TileSize;
    const std::uint32_t tileCount = result.TilesX * result.TilesY;
    std::vector<std::uint32_t> counts(tileCount, 0U);
    std::vector<Bounds> triangleBounds(scene.Triangles.size());
    for (std::size_t triangleIndex = 0;
         triangleIndex < scene.Triangles.size();
         ++triangleIndex)
    {
        if ((triangleIndex & 255U) == 0U &&
            IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            result.Cancelled = true;
            return result;
        }

        float screenX[3]{};
        float screenY[3]{};
        bool valid = true;
        for (std::size_t vertexIndex = 0; vertexIndex < 3; ++vertexIndex)
        {
            const Float4 clip = TransformForBinning(
                scene.Triangles[triangleIndex].Vertices[vertexIndex],
                settings,
                lightPass);
            if (!std::isfinite(clip.X) || !std::isfinite(clip.Y) ||
                !std::isfinite(clip.W) || clip.W <= 1.0e-6f)
            {
                valid = false;
                break;
            }
            const float inverseW = 1.0f / clip.W;
            screenX[vertexIndex] =
                (clip.X * inverseW * 0.5f + 0.5f) *
                static_cast<float>(width);
            screenY[vertexIndex] =
                (0.5f - clip.Y * inverseW * 0.5f) *
                static_cast<float>(height);
        }
        if (!valid)
        {
            continue;
        }

        const float padding = !lightPass && settings.Mode != FillMode::Solid
            ? std::max(settings.WireframeWidth, 1.0f) * 0.5f : 0.0f;
        Bounds bounds;
        bounds.MinimumX = std::max(
            0,
            static_cast<int>(std::floor(
                std::min({screenX[0], screenX[1], screenX[2]}) - padding)));
        bounds.MinimumY = std::max(
            0,
            static_cast<int>(std::floor(
                std::min({screenY[0], screenY[1], screenY[2]}) - padding)));
        bounds.MaximumX = std::min(
            static_cast<int>(width) - 1,
            static_cast<int>(std::ceil(
                std::max({screenX[0], screenX[1], screenX[2]}) + padding)));
        bounds.MaximumY = std::min(
            static_cast<int>(height) - 1,
            static_cast<int>(std::ceil(
                std::max({screenY[0], screenY[1], screenY[2]}) + padding)));
        bounds.Valid =
            bounds.MinimumX <= bounds.MaximumX &&
            bounds.MinimumY <= bounds.MaximumY;
        triangleBounds[triangleIndex] = bounds;
        if (!bounds.Valid)
        {
            continue;
        }

        ++result.BinnedTriangles;
        for (std::uint32_t tileY =
                 static_cast<std::uint32_t>(bounds.MinimumY) / TileSize;
             tileY <=
             static_cast<std::uint32_t>(bounds.MaximumY) / TileSize;
             ++tileY)
        {
            for (std::uint32_t tileX =
                     static_cast<std::uint32_t>(bounds.MinimumX) / TileSize;
                 tileX <=
                 static_cast<std::uint32_t>(bounds.MaximumX) / TileSize;
                 ++tileX)
            {
                ++counts[tileY * result.TilesX + tileX];
            }
        }
    }

    result.Offsets.resize(static_cast<std::size_t>(tileCount) + 1U, 0U);
    for (std::uint32_t tile = 0; tile < tileCount; ++tile)
    {
        result.MaximumListLength =
            std::max(result.MaximumListLength, counts[tile]);
        result.Offsets[tile + 1U] = result.Offsets[tile] + counts[tile];
    }
    result.TriangleIndices.resize(result.Offsets.back());
    std::vector<std::uint32_t> writeOffsets = result.Offsets;
    for (std::uint32_t triangleIndex = 0;
         triangleIndex <
         static_cast<std::uint32_t>(triangleBounds.size());
         ++triangleIndex)
    {
        if ((triangleIndex & 255U) == 0U &&
            IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            result.Cancelled = true;
            return result;
        }

        const Bounds& bounds = triangleBounds[triangleIndex];
        if (!bounds.Valid)
        {
            continue;
        }
        for (std::uint32_t tileY =
                 static_cast<std::uint32_t>(bounds.MinimumY) / TileSize;
             tileY <=
             static_cast<std::uint32_t>(bounds.MaximumY) / TileSize;
             ++tileY)
        {
            for (std::uint32_t tileX =
                     static_cast<std::uint32_t>(bounds.MinimumX) / TileSize;
                 tileX <=
                 static_cast<std::uint32_t>(bounds.MaximumX) / TileSize;
                 ++tileX)
            {
                const std::uint32_t tile =
                    tileY * result.TilesX + tileX;
                result.TriangleIndices[writeOffsets[tile]++] =
                    triangleIndex;
            }
        }
    }
    return result;
}

} // namespace Evans::Rasterizer::Detail
