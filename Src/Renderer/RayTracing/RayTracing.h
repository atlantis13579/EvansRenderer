#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32) && !defined(EVANS_RENDERER_STATIC)
#  if defined(EVANS_RENDERER_EXPORTS)
#    define EVANS_RENDERER_API __declspec(dllexport)
#  else
#    define EVANS_RENDERER_API __declspec(dllimport)
#  endif
#else
#  define EVANS_RENDERER_API
#endif

namespace Evans::Renderer { struct GpuImage; }

namespace Evans
{
namespace RayTracing
{
    struct Float2
    {
        float X = 0.0f;
        float Y = 0.0f;

        Float2() = default;
        Float2(float x, float y) : X(x), Y(y) {}
    };

    struct Float3
    {
        float X = 0.0f;
        float Y = 0.0f;
        float Z = 0.0f;

        Float3() = default;
        Float3(float x, float y, float z) : X(x), Y(y), Z(z) {}

        Float3 operator-() const { return Float3(-X, -Y, -Z); }
        Float3 operator+(const Float3& rhs) const { return Float3(X + rhs.X, Y + rhs.Y, Z + rhs.Z); }
        Float3 operator-(const Float3& rhs) const { return Float3(X - rhs.X, Y - rhs.Y, Z - rhs.Z); }
        Float3 operator*(float scalar) const { return Float3(X * scalar, Y * scalar, Z * scalar); }
        Float3 operator/(float scalar) const { return Float3(X / scalar, Y / scalar, Z / scalar); }
    };

    inline Float3 operator*(float scalar, const Float3& value) { return value * scalar; }
    EVANS_RENDERER_API float Dot(const Float3& lhs, const Float3& rhs);
    EVANS_RENDERER_API Float3 Cross(const Float3& lhs, const Float3& rhs);
    EVANS_RENDERER_API Float3 Normalize(const Float3& value);

    struct Ray
    {
        Float3 Origin;
        Float3 Direction = Float3(0.0f, 0.0f, -1.0f);
        float MinDistance = 1.0e-4f;
        float MaxDistance = 1.0e30f;
    };

    inline constexpr uint32_t InvalidRayTextureIndex =
        std::numeric_limits<uint32_t>::max();

    struct RayTexture
    {
        uint32_t Width = 0;
        uint32_t Height = 0;
        // Immutable RGBA8 texels in top-left row order. Diffuse RGB uses the
        // exact display-sRGB transfer and alpha is linear UNORM. Sampling is
        // bilinear repeat with a V flip; invalid references sample opaque
        // white. Shared ownership lets adapters alias storage without copying.
        std::shared_ptr<const std::vector<uint8_t>> PixelsRGBA8;

        [[nodiscard]] bool IsValid() const noexcept
        {
            if (Width == 0 || Height == 0 || !PixelsRGBA8)
            {
                return false;
            }
            constexpr size_t ComponentsPerPixel = 4U;
            const size_t width = static_cast<size_t>(Width);
            const size_t height = static_cast<size_t>(Height);
            if (height > std::numeric_limits<size_t>::max() / width)
            {
                return false;
            }
            const size_t pixelCount = width * height;
            return pixelCount <=
                       std::numeric_limits<size_t>::max() / ComponentsPerPixel &&
                   PixelsRGBA8->size() == pixelCount * ComponentsPerPixel;
        }
    };

    struct Material
    {
        Float3 Albedo = Float3(0.8f, 0.8f, 0.8f);
        Float3 AmbientColor = Float3(0.8f, 0.8f, 0.8f);
        Float3 SpecularColor = Float3(0.0f, 0.0f, 0.0f);
        Float3 Emission = Float3(0.0f, 0.0f, 0.0f);
        float Shininess = 1.0f;
        float Reflectivity = 0.0f;
        uint32_t DiffuseTextureIndex = InvalidRayTextureIndex;
    };

    struct Triangle
    {
        Float3 Position0;
        Float3 Position1;
        Float3 Position2;
        Float3 Normal0;
        Float3 Normal1;
        Float3 Normal2;
        Float3 Color0 = Float3(1.0f, 1.0f, 1.0f);
        Float3 Color1 = Float3(1.0f, 1.0f, 1.0f);
        Float3 Color2 = Float3(1.0f, 1.0f, 1.0f);
        Float2 TexCoord0;
        Float2 TexCoord1;
        Float2 TexCoord2;
        uint32_t MaterialIndex = 0;
        bool SmoothNormals = false;
    };

    struct Sphere
    {
        Float3 Center;
        float Radius = 1.0f;
        uint32_t MaterialIndex = 0;
    };

    struct DirectionalLight
    {
        // A normalized vector from the shaded point towards the light.
        Float3 DirectionToLight = Normalize(Float3(-0.5f, 1.0f, 0.4f));
        Float3 Color = Float3(1.0f, 1.0f, 1.0f);
        float Intensity = 1.0f;
        bool CastShadows = true;
    };

    struct PointLight
    {
        Float3 Position;
        Float3 Color = Float3(1.0f, 1.0f, 1.0f);
        float Intensity = 1.0f;
        bool CastShadows = true;
        float Range = 0.0f; // Zero keeps inverse-square lighting unbounded.
    };

    struct RectAreaLight
    {
        Float3 Position;
        // Normalized direction in which the front face emits light.
        Float3 Direction = Float3(0.0f, -1.0f, 0.0f);
        Float3 Color = Float3(1.0f, 1.0f, 1.0f);
        float Intensity = 1.0f;
        // Full rectangle width and height in world units.
        Float2 Size = Float2(1.0f, 1.0f);
        bool CastShadows = true;
        float Range = 0.0f; // Radius measured from each emitter sample.
    };

    struct Scene
    {
        std::vector<Material> Materials;
        // Materials reference this table by index so texels are shared by all
        // triangles that use the same image.
        std::vector<RayTexture> Textures;
        std::vector<Triangle> Triangles;
        std::vector<Sphere> Spheres;
        std::vector<DirectionalLight> DirectionalLights;
        std::vector<PointLight> PointLights;
        std::vector<RectAreaLight> RectAreaLights;
        // Linear environment colors. Equal defaults match the realtime clear.
        Float3 BackgroundTop = Float3(0.02f, 0.02f, 0.025f);
        Float3 BackgroundBottom = Float3(0.02f, 0.02f, 0.025f);
        float AmbientIntensity = 0.08f;
    };

    struct Camera
    {
        Float3 Position = Float3(0.0f, 0.0f, 5.0f);
        Float3 Target = Float3(0.0f, 0.0f, 0.0f);
        Float3 Up = Float3(0.0f, 1.0f, 0.0f);
        float VerticalFieldOfViewDegrees = 60.0f;
    };

    struct RenderSettings
    {
        uint32_t Width = 640;
        uint32_t Height = 360;
        uint32_t SamplesPerPixel = 1;
        uint32_t MaxReflectionBounces = 1;
        // Zero selects the hardware concurrency reported by the standard library.
        uint32_t WorkerCount = 0;
        bool ShadowsEnabled = true;
        bool GpuPresentation = false;
        // Fraction of direct lighting removed when a shadow ray is occluded.
        float ShadowStrength = 1.0f;
        // Encodes linear radiance as exact display-sRGB when enabled.
        bool GammaCorrection = true;
    };

    class EVANS_RENDERER_API Image
    {
    public:
        Image() = default;
        Image(uint32_t width, uint32_t height);

        uint32_t GetWidth() const { return m_Width; }
        uint32_t GetHeight() const { return m_Height; }
        bool IsEmpty() const { return m_Pixels.empty(); }
        // Top-left RGBA8. RGB is display-sRGB when GammaCorrection is enabled.
        const std::vector<uint8_t>& GetPixelsRGBA8() const { return m_Pixels; }
        std::vector<uint8_t>& GetPixelsRGBA8() { return m_Pixels; }
        const uint8_t* GetPixel(uint32_t x, uint32_t y) const;
        bool SavePPM(const std::string& path, std::string* error = nullptr) const;

    private:
        uint32_t m_Width = 0;
        uint32_t m_Height = 0;
        std::vector<uint8_t> m_Pixels;
    };

    enum class BackendType : uint8_t
    {
        DXR
    };

    struct BackendInfo
    {
        BackendType Type = BackendType::DXR;
        std::string Name;
        std::string Description;
        bool IsNative = false;
        bool IsHardwareAccelerated = false;
        bool IsAvailable = true;
    };

    enum class RenderStatus : uint8_t
    {
        Completed,
        Cancelled,
        Failed
    };

    struct RenderStatistics
    {
        uint64_t PrimaryRays = 0;
        uint64_t ShadowRays = 0;
        uint64_t ReflectionRays = 0;
        uint64_t PrimitiveTests = 0;
        uint64_t CompletedPixels = 0;
        double BuildMilliseconds = 0.0;
        double RenderMilliseconds = 0.0;
    };

    struct RenderResult
    {
        RenderStatus Status = RenderStatus::Failed;
        Image Output;
        RenderStatistics Statistics;
        std::shared_ptr<const Renderer::GpuImage> GpuColor;
        std::string Error;
    };

    class EVANS_RENDERER_API CancellationToken
    {
    public:
        CancellationToken() = default;
        bool IsCancellationRequested() const;

    private:
        explicit CancellationToken(std::shared_ptr<std::atomic<bool>> state) : m_State(std::move(state)) {}
        std::shared_ptr<std::atomic<bool>> m_State;
        friend class CancellationSource;
    };

    class EVANS_RENDERER_API CancellationSource
    {
    public:
        CancellationSource();
        CancellationToken GetToken() const;
        void Cancel() const;
        bool IsCancellationRequested() const;

    private:
        std::shared_ptr<std::atomic<bool>> m_State;
    };

    class EVANS_RENDERER_API RenderJob
    {
    public:
        // Backend queues use this constructor to bind a future to its cancellation state.
        RenderJob(CancellationSource cancellation, std::future<RenderResult> future);
        RenderJob(RenderJob&&) noexcept;
        RenderJob& operator=(RenderJob&&) noexcept;
        ~RenderJob();

        RenderJob(const RenderJob&) = delete;
        RenderJob& operator=(const RenderJob&) = delete;

        void Cancel() const;
        bool IsReady() const;
        void Wait() const;
        RenderResult GetResult();

    private:
        CancellationSource m_Cancellation;
        std::future<RenderResult> m_Future;
    };

    class EVANS_RENDERER_API IRayTracingRHI
    {
    public:
        virtual ~IRayTracingRHI() = default;
        virtual BackendInfo GetBackendInfo() const = 0;
        virtual RenderResult Render(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            const CancellationToken& cancellation = CancellationToken()) const = 0;
        virtual RenderJob RenderAsync(Scene scene, Camera camera, RenderSettings settings) const = 0;
        virtual RenderJob RenderAsync(
            std::shared_ptr<const Scene> scene,
            Camera camera,
            RenderSettings settings) const = 0;
        virtual void WaitIdle() const = 0;
    };

}
}
