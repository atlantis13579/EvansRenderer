#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Evans::Rasterizer
{

struct Float2
{
    float X = 0.0f;
    float Y = 0.0f;
};

struct Float3
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
};

struct Float4
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
    float W = 1.0f;
};

struct Color
{
    float R = 0.0f;
    float G = 0.0f;
    float B = 0.0f;
    float A = 1.0f;

    [[nodiscard]] Color Clamped() const noexcept
    {
        return {
            std::clamp(R, 0.0f, 1.0f),
            std::clamp(G, 0.0f, 1.0f),
            std::clamp(B, 0.0f, 1.0f),
            std::clamp(A, 0.0f, 1.0f)};
    }
};

// Raster targets store linear RGB. These constants preserve the editor's
// authored display-sRGB wireframe and selection colors after output encoding.
inline constexpr Color WireframeDisplayLinear{
    0.013411749f, 0.638283193f, 1.0f, 1.0f};
inline constexpr Color OutlineDisplayLinear{
    1.0f, 0.263273418f, 0.003935940f, 1.0f};

// Row-major matrix that multiplies column vectors on the right.
struct Matrix4
{
    std::array<float, 16> Values{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};

    [[nodiscard]] float& operator()(std::size_t row, std::size_t column) noexcept
    {
        return Values[row * 4U + column];
    }

    [[nodiscard]] float operator()(std::size_t row, std::size_t column) const noexcept
    {
        return Values[row * 4U + column];
    }

    [[nodiscard]] Matrix4 operator*(const Matrix4& right) const noexcept;
    [[nodiscard]] Float4 operator*(const Float4& vector) const noexcept;

    [[nodiscard]] static Matrix4 Identity() noexcept { return {}; }
    [[nodiscard]] static Matrix4 Translation(const Float3& translation) noexcept;
    [[nodiscard]] static Matrix4 Scale(const Float3& scale) noexcept;
    [[nodiscard]] static Matrix4 RotationY(float radians) noexcept;
    [[nodiscard]] static Matrix4 PerspectiveRH(
        float verticalFieldOfViewRadians,
        float aspectRatio,
        float nearPlane,
        float farPlane);
    [[nodiscard]] static Matrix4 LookAtRH(
        const Float3& eye,
        const Float3& target,
        const Float3& up);
};

struct RasterVertex
{
    // Legacy clip-space fields remain first for source compatibility with Editor adapters.
    Float4 ClipPosition;
    // Linear RGB with straight linear alpha.
    Color VertexColor{1.0f, 1.0f, 1.0f, 1.0f};
    Float4 ShadowClipPosition;

    // Local-space fields are transformed by World, View, and Projection when enabled.
    Float3 Position;
    Float3 Normal{0.0f, 0.0f, 1.0f};
    Float2 TexCoord;
};

inline constexpr std::uint32_t InvalidRasterTextureIndex =
    std::numeric_limits<std::uint32_t>::max();

struct RasterTexture
{
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    // Immutable RGBA8 texels in top-left row order. Diffuse RGB is display-sRGB;
    // alpha and opacity red are linear UNORM. Sampling is bilinear repeat with
    // a V flip, and invalid texture references sample opaque white. Shared
    // ownership lets adapters alias an image without copying its pixel storage.
    std::shared_ptr<const std::vector<std::uint8_t>> PixelsRGBA8;

    [[nodiscard]] bool IsValid() const noexcept
    {
        if (Width == 0 || Height == 0 || !PixelsRGBA8)
        {
            return false;
        }
        constexpr std::size_t componentsPerPixel = 4U;
        const std::size_t width = static_cast<std::size_t>(Width);
        const std::size_t height = static_cast<std::size_t>(Height);
        if (height > std::numeric_limits<std::size_t>::max() / width)
        {
            return false;
        }
        const std::size_t pixelCount = width * height;
        return pixelCount <=
                   std::numeric_limits<std::size_t>::max() / componentsPerPixel &&
               PixelsRGBA8->size() == pixelCount * componentsPerPixel;
    }
};

struct RasterMaterial
{
    // Material colors are linear RGB with straight linear alpha.
    Color BaseColor{1.0f, 1.0f, 1.0f, 1.0f};
    Color SpecularColor{0.25f, 0.25f, 0.25f, 1.0f};
    Color EmissiveColor{0.0f, 0.0f, 0.0f, 1.0f};
    float Shininess = 32.0f;
    // Fragment alpha is saturated vertex * base * diffuse * opacity. A positive
    // cutoff discards fragments below the threshold before color/depth writes.
    float AlphaCutoff = 0.0f;
    std::uint32_t DiffuseTextureIndex = InvalidRasterTextureIndex;
    std::uint32_t OpacityTextureIndex = InvalidRasterTextureIndex;
};

struct RasterTriangle
{
    RasterVertex Vertices[3];
    bool Selected = false;
    RasterMaterial Material;
};

struct RasterScene
{
    std::vector<RasterTriangle> Triangles;
    // Materials reference this table by index so texels are shared by every
    // triangle using the same image.
    std::vector<RasterTexture> Textures;
};

struct ShadowMap
{
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    std::vector<float> Depth;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return Width > 0 && Height > 0 &&
               Depth.size() == static_cast<std::size_t>(Width) * Height;
    }

    [[nodiscard]] float At(std::uint32_t x, std::uint32_t y) const
    {
        return Depth.at(static_cast<std::size_t>(y) * Width + x);
    }
};

enum class FillMode
{
    Solid,
    Wireframe,
    SolidWireframe
};

enum class VertexInputSpace
{
    Clip,
    Local
};

enum class ClipDepthRange
{
    ZeroToOne,
    MinusOneToOne
};

enum class ShadingModel
{
    Unlit,
    Lambert,
    Phong
};

struct DirectionalLight
{
    // Normalized direction from a shaded point towards the light.
    Float3 DirectionToLight{-0.4f, 0.8f, 0.45f};
    Color LightColor{1.0f, 1.0f, 1.0f, 1.0f};
    float Intensity = 1.0f;
};

inline constexpr std::uint32_t MaximumRasterLights = 4;
inline constexpr std::uint32_t LegacyRasterLightCount =
    std::numeric_limits<std::uint32_t>::max();

enum class RasterLightType : std::uint32_t
{
    Directional,
    Point,
    RectArea
};

struct RasterLight
{
    RasterLightType Type = RasterLightType::Directional;
    Float3 Position;
    // Directional lights store the direction from the surface to the light.
    // Rect-area lights store the direction emitted by the front face.
    Float3 Direction{-0.4f, 0.8f, 0.45f};
    Color LightColor{1.0f, 1.0f, 1.0f, 1.0f};
    float Intensity = 1.0f;
    Float2 AreaSize{1.0f, 1.0f};
    bool CastShadows = true;
};

struct RasterSettings
{
    FillMode Mode = FillMode::Solid;
    VertexInputSpace InputSpace = VertexInputSpace::Clip;
    ClipDepthRange DepthRange = ClipDepthRange::ZeroToOne;
    ShadingModel Shading = ShadingModel::Unlit;

    Matrix4 World;
    Matrix4 View;
    Matrix4 Projection;
    Matrix4 LightViewProjection;
    Float3 CameraPosition{0.0f, 0.0f, 5.0f};
    DirectionalLight Light;
    Color AmbientColor{1.0f, 1.0f, 1.0f, 1.0f};
    float AmbientStrength = 0.08f;

    // All raster colors are linear RGB with straight linear alpha.
    Color ClearColor{0.02f, 0.02f, 0.025f, 1.0f};
    Color WireframeColor = WireframeDisplayLinear;
    Color OutlineColor = OutlineDisplayLinear;
    bool DepthTest = true;
    bool BackFaceCulling = false;
    bool FrontFaceCounterClockwise = true;
    bool ShadowsEnabled = false;
    bool AutoGenerateShadowMap = false;
    bool OutlineEnabled = true;
    float ShadowBias = 0.001f;
    float ShadowSlopeScaledBias = 0.0f;
    float ShadowStrength = 0.55f;
    std::uint32_t ShadowMapSize = 512;
    std::uint32_t ShadowPCFRadius = 1;
    float WireframeWidth = 1.0f;
    float OutlineWidth = 3.0f;
    const ShadowMap* SceneShadowMap = nullptr;

    // LegacyRasterLightCount preserves the original DirectionalLight field.
    // Any other value explicitly enables the typed light array, including zero.
    std::array<RasterLight, MaximumRasterLights> Lights;
    std::uint32_t LightCount = LegacyRasterLightCount;
    std::int32_t ShadowLightIndex = 0;
};

[[nodiscard]] inline std::uint32_t EffectiveRasterLightCount(
    const RasterSettings& settings) noexcept
{
    return settings.LightCount == LegacyRasterLightCount
        ? 1U
        : std::min(settings.LightCount, MaximumRasterLights);
}

[[nodiscard]] inline RasterLight EffectiveRasterLight(
    const RasterSettings& settings,
    std::uint32_t index) noexcept
{
    if (settings.LightCount != LegacyRasterLightCount)
    {
        return index < MaximumRasterLights ? settings.Lights[index] : RasterLight{};
    }
    RasterLight result;
    result.Type = RasterLightType::Directional;
    result.Direction = settings.Light.DirectionToLight;
    result.LightColor = settings.Light.LightColor;
    result.Intensity = settings.Light.Intensity;
    return result;
}

class RasterCancellationToken
{
public:
    RasterCancellationToken() = default;
    // This constructor lets an owning application bridge its cancellation state
    // without introducing a polling thread or an Editor dependency.
    explicit RasterCancellationToken(std::shared_ptr<std::atomic<bool>> state)
        : State_(std::move(state))
    {
    }
    [[nodiscard]] bool IsCancellationRequested() const noexcept
    {
        return State_ != nullptr && State_->load(std::memory_order_relaxed);
    }

private:
    std::shared_ptr<std::atomic<bool>> State_;
    friend class RasterCancellationSource;
};

class RasterCancellationSource
{
public:
    RasterCancellationSource()
        : State_(std::make_shared<std::atomic<bool>>(false))
    {
    }

    [[nodiscard]] RasterCancellationToken Token() const
    {
        return RasterCancellationToken(State_);
    }

    void Cancel() const noexcept { State_->store(true, std::memory_order_relaxed); }

private:
    std::shared_ptr<std::atomic<bool>> State_;
};

enum class RasterStatus
{
    Completed,
    Cancelled
};

struct RasterStatistics
{
    std::uint64_t InputTriangles = 0;
    std::uint64_t PostClipTriangles = 0;
    std::uint64_t ShadedFragments = 0;
    std::uint64_t DepthRejectedFragments = 0;
    std::uint64_t DispatchedTiles = 0;
    std::uint32_t WorkerThreads = 0;
    double TransformMilliseconds = 0.0;
    double RasterMilliseconds = 0.0;
    std::string ExecutionPath;
};

class RenderTarget
{
public:
    RenderTarget() = default;
    RenderTarget(std::uint32_t width, std::uint32_t height) { Resize(width, height); }

    void Resize(std::uint32_t width, std::uint32_t height)
    {
        if (width == 0 || height == 0)
        {
            throw std::invalid_argument("A render target must have non-zero dimensions");
        }
        Width_ = width;
        Height_ = height;
        Colors_.resize(static_cast<std::size_t>(width) * height);
        Depth_.resize(static_cast<std::size_t>(width) * height);
        Clear({}, std::numeric_limits<float>::infinity());
    }

    void Clear(const Color& color, float depth = std::numeric_limits<float>::infinity())
    {
        std::fill(Colors_.begin(), Colors_.end(), color);
        std::fill(Depth_.begin(), Depth_.end(), depth);
    }

    [[nodiscard]] bool IsValid() const noexcept
    {
        return Width_ > 0 && Height_ > 0 &&
               Colors_.size() == static_cast<std::size_t>(Width_) * Height_ &&
               Depth_.size() == Colors_.size();
    }
    [[nodiscard]] std::uint32_t Width() const noexcept { return Width_; }
    [[nodiscard]] std::uint32_t Height() const noexcept { return Height_; }
    [[nodiscard]] const Color& ColorAt(std::uint32_t x, std::uint32_t y) const
    {
        return Colors_.at(Index(x, y));
    }
    [[nodiscard]] float DepthAt(std::uint32_t x, std::uint32_t y) const
    {
        return Depth_.at(Index(x, y));
    }
    [[nodiscard]] const std::vector<Color>& Colors() const noexcept { return Colors_; }
    [[nodiscard]] const std::vector<float>& Depths() const noexcept { return Depth_; }
    // Mutable buffer access is intended for RHI implementations and image interop.
    [[nodiscard]] std::vector<Color>& MutableColors() noexcept { return Colors_; }
    [[nodiscard]] std::vector<float>& MutableDepths() noexcept { return Depth_; }
    // Encodes linear RGB as display-sRGB and alpha as straight UNORM8.
    // Pixels retain the top-left row order used by the editor image contract.
    [[nodiscard]] std::vector<std::uint32_t> ToRGBA8() const;

private:
    friend class SoftwareRasterCore;
    [[nodiscard]] std::size_t Index(std::uint32_t x, std::uint32_t y) const
    {
        if (x >= Width_ || y >= Height_)
        {
            throw std::out_of_range("Render target pixel coordinates are out of range");
        }
        return static_cast<std::size_t>(y) * Width_ + x;
    }

    std::uint32_t Width_ = 0;
    std::uint32_t Height_ = 0;
    std::vector<Color> Colors_;
    std::vector<float> Depth_;
};

} // namespace Evans::Rasterizer
