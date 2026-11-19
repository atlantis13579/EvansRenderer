#pragma once

#include "Rasterizer/IRasterRHI.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Evans::Rasterizer::Detail
{

// Shared host-side ABI used by native compute raster backends.
struct alignas(16) GpuRasterVertex
{
    float Clip[4]{};
    float Color[4]{};
    float Position[4]{};
    float Normal[4]{};
    float ShadowClip[4]{};
    float TexCoord[4]{};
};

struct alignas(16) GpuRasterTriangle
{
    GpuRasterVertex Vertices[3];
    float BaseColor[4]{};
    float SpecularColor[4]{};
    float EmissiveColor[4]{};
    float Shininess = 1.0f;
    float AlphaCutoff = 0.0f;
    std::uint32_t DiffuseTextureIndex = InvalidRasterTextureIndex;
    std::uint32_t OpacityTextureIndex = InvalidRasterTextureIndex;
    std::uint32_t Selected = 0;
    std::uint32_t NormalTextureIndex = InvalidRasterTextureIndex;
    float SubsurfaceWeight = 0;
    std::uint32_t ReceivesLightmap = 0;
};

struct alignas(16) GpuRasterTexture
{
    std::uint32_t OffsetTexels = 0;
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    std::uint32_t FirstMip = 0;
};

struct GpuRasterTextureData
{
    std::vector<GpuRasterTexture> Descriptors;
    // Each uint stores RGBA8 with red in the least-significant byte.
    std::vector<std::uint32_t> TexelsRGBA8;
};

struct alignas(16) GpuRasterLight
{
    float PositionType[4]{};
    float DirectionIntensity[4]{};
    float ColorCastShadows[4]{};
    float AreaSizePadding[4]{};
};

struct alignas(16) GpuRasterConstants
{
    float World[16]{};
    float View[16]{};
    float Projection[16]{};
    float LightViewProjection[16]{};
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    std::uint32_t TriangleCount = 0;
    std::uint32_t InputSpace = 0;
    std::uint32_t DepthRange = 0;
    std::uint32_t FillModeValue = 0;
    std::uint32_t ShadingValue = 0;
    std::uint32_t BackFaceCulling = 0;
    std::uint32_t FrontFaceCounterClockwise = 0;
    std::uint32_t OutlineEnabled = 0;
    std::uint32_t ShadowsEnabled = 0;
    std::uint32_t ShadowWidth = 0;
    std::uint32_t ShadowHeight = 0;
    std::uint32_t TilesX = 0;
    float WireframeWidth = 1.0f;
    float OutlineWidth = 1.0f;
    float ShadowBias = 0.001f;
    float ShadowStrength = 0.55f;
    std::uint32_t DepthTest = 1;
    std::uint32_t ShadowPCFRadius = 1;
    float ClearColor[4]{};
    float WireframeColor[4]{};
    float OutlineColor[4]{};
    float CameraPosition[4]{};
    float LightDirectionIntensity[4]{};
    float LightColorAmbientStrength[4]{};
    float AmbientColor[4]{};
    std::uint32_t LightCount = 0;
    std::int32_t ShadowLightIndex = -1;
    std::uint32_t TextureCount = 0;
    std::uint32_t TextureTexelCount = 0;
    GpuRasterLight Lights[MaximumRasterLights]{};
    float ShadowRasterBias[4]{};
    std::uint32_t MaterialOptions = 0;
    std::uint32_t FrontLightmap = InvalidRasterTextureIndex;
    std::uint32_t BackLightmap = InvalidRasterTextureIndex;
    std::uint32_t ACESColor = 0;
    float DisplayParameters[4]{}; // Exposure, contrast, saturation, lightmap intensity.
    std::uint32_t SkyTexture = InvalidRasterTextureIndex, SkyDiffuseTexture = InvalidRasterTextureIndex;
    float SkyIntensity = 0, SkyDiffuseIntensity = 0;
    float SubsurfaceScattering[4]{};
};

static_assert(sizeof(GpuRasterVertex) == 96, "GPU raster vertex ABI mismatch");
static_assert(sizeof(GpuRasterTriangle) == 368, "GPU raster triangle ABI mismatch");
static_assert(sizeof(GpuRasterTexture) == 16, "GPU raster texture ABI mismatch");
static_assert(sizeof(GpuRasterLight) == 64, "GPU raster light ABI mismatch");
static_assert(sizeof(GpuRasterConstants) == 800, "GPU raster constants ABI mismatch");
static_assert(
    alignof(GpuRasterVertex) == 16 &&
        alignof(GpuRasterTriangle) == 16 &&
        alignof(GpuRasterTexture) == 16 &&
        alignof(GpuRasterLight) == 16 &&
        alignof(GpuRasterConstants) == 16,
    "GPU raster buffer alignment mismatch");
static_assert(
    offsetof(GpuRasterLight, DirectionIntensity) == 16 &&
        offsetof(GpuRasterLight, ColorCastShadows) == 32 &&
        offsetof(GpuRasterLight, AreaSizePadding) == 48,
    "GPU raster light field offsets mismatch");
static_assert(
    offsetof(GpuRasterVertex, Color) == 16 &&
        offsetof(GpuRasterVertex, Position) == 32 &&
        offsetof(GpuRasterVertex, Normal) == 48 &&
        offsetof(GpuRasterVertex, ShadowClip) == 64 &&
        offsetof(GpuRasterVertex, TexCoord) == 80,
    "GPU raster vertex field offsets mismatch");
static_assert(
    offsetof(GpuRasterTriangle, BaseColor) == 288 &&
        offsetof(GpuRasterTriangle, SpecularColor) == 304 &&
        offsetof(GpuRasterTriangle, EmissiveColor) == 320 &&
        offsetof(GpuRasterTriangle, Shininess) == 336 &&
        offsetof(GpuRasterTriangle, AlphaCutoff) == 340 &&
        offsetof(GpuRasterTriangle, DiffuseTextureIndex) == 344 &&
        offsetof(GpuRasterTriangle, OpacityTextureIndex) == 348 &&
        offsetof(GpuRasterTriangle, Selected) == 352,
    "GPU raster triangle field offsets mismatch");
static_assert(
    offsetof(GpuRasterConstants, Width) == 256 &&
        offsetof(GpuRasterConstants, ShadowHeight) == 304 &&
        offsetof(GpuRasterConstants, ShadowBias) == 320 &&
        offsetof(GpuRasterConstants, ClearColor) == 336 &&
        offsetof(GpuRasterConstants, AmbientColor) == 432 &&
        offsetof(GpuRasterConstants, LightCount) == 448 &&
        offsetof(GpuRasterConstants, TextureCount) == 456 &&
        offsetof(GpuRasterConstants, TextureTexelCount) == 460 &&
        offsetof(GpuRasterConstants, Lights) == 464 &&
        offsetof(GpuRasterConstants, ShadowRasterBias) == 720,
    "GPU raster constant field offsets mismatch");

// Texture pixels are immutable shared storage. Retain identity and dimensions
// so camera changes do not repack or re-upload the material atlas.
class GpuRasterTextureCache
{
public:
    const GpuRasterTextureData& Get(const RasterScene& scene);
    std::uint64_t Generation() const { return Generation_; }
    void Reset() { Sources_.clear(); Data_={}; Generation_=0; }
private:
    std::vector<RasterTexture> Sources_;
    GpuRasterTextureData Data_;
    std::uint64_t Generation_=0;
};

struct GpuRasterTileBins
{
    std::vector<std::uint32_t> Offsets;
    std::vector<std::uint32_t> TriangleIndices;
    std::uint32_t TilesX = 0;
    std::uint32_t TilesY = 0;
    std::uint32_t MaximumListLength = 0;
    std::uint64_t BinnedTriangles = 0;
    bool Cancelled = false;
};

[[nodiscard]] bool IsGpuRasterCancelled(
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation) noexcept;

[[nodiscard]] std::vector<GpuRasterTriangle> BuildGpuRasterTriangles(
    const RasterScene& scene);

[[nodiscard]] GpuRasterTextureData BuildGpuRasterTextures(
    const RasterScene& scene);

[[nodiscard]] GpuRasterConstants BuildGpuRasterConstants(
    const RasterScene& scene,
    const RasterSettings& settings,
    const RenderTarget& target,
    bool shadowsEnabled,
    std::uint32_t shadowWidth,
    std::uint32_t shadowHeight);

[[nodiscard]] GpuRasterTileBins BuildGpuRasterTileBins(
    const RasterScene& scene,
    const RasterSettings& settings,
    std::uint32_t width,
    std::uint32_t height,
    bool lightPass,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation);

} // namespace Evans::Rasterizer::Detail
