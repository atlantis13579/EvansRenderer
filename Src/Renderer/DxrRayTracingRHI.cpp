#include "DxrRayTracingRHI.h"

#include "Renderer/RayTracingBackendCommon.h"
#include "DxrShaderBytecode.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  include <Windows.h>
#  include <d3d12.h>
#  include <dxcapi.h>
#  include <dxgi1_6.h>
#  include <wrl/client.h>
#endif

namespace Evans
{
namespace RayTracing
{
    namespace
    {
#if defined(_WIN32)
        using Microsoft::WRL::ComPtr;

        constexpr std::uint32_t SphereLatitudeSegments = 12u;
        constexpr std::uint32_t SphereLongitudeSegments = 24u;
        constexpr std::uint64_t SphereTriangleCount =
            static_cast<std::uint64_t>(SphereLongitudeSegments) * (SphereLatitudeSegments - 1u) * 2u;
        constexpr std::uint32_t DirectionalLightType = 0;
        constexpr std::uint32_t PointLightType = 1;
        constexpr std::uint32_t RectAreaLightType = 2;
        constexpr DWORD GpuWaitTimeoutMilliseconds = 5000;

        constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        struct DynamicModule
        {
            DynamicModule() = default;
            ~DynamicModule()
            {
                if (Handle) FreeLibrary(Handle);
            }

            DynamicModule(const DynamicModule&) = delete;
            DynamicModule& operator=(const DynamicModule&) = delete;

            bool Load(const wchar_t* name)
            {
                Handle = LoadLibraryW(name);
                return Handle != nullptr;
            }

            template <typename Function>
            Function Get(const char* name) const
            {
                return reinterpret_cast<Function>(GetProcAddress(Handle, name));
            }

            HMODULE Handle = nullptr;
        };

        struct EventHandle
        {
            ~EventHandle()
            {
                if (Handle) CloseHandle(Handle);
            }

            EventHandle(const EventHandle&) = delete;
            EventHandle& operator=(const EventHandle&) = delete;
            EventHandle() = default;

            HANDLE Handle = nullptr;
        };

        struct GpuVertex
        {
            float X;
            float Y;
            float Z;
            float Padding;
        };

        struct GpuTriangle
        {
            float Normal0[4];
            float Normal1[4];
            float Normal2[4];
            float Color0[4];
            float Color1[4];
            float Color2[4];
            float TexCoord01[4];
            float TexCoord2[4];
            uint32_t MaterialIndex;
            uint32_t SmoothNormals;
            uint32_t HasTexCoord;
            uint32_t Padding;
        };

        struct GpuMaterial
        {
            float AlbedoReflectivity[4];
            float AmbientShininess[4];
            float Specular[4];
            float Emission[4];
            uint32_t DiffuseTextureIndex;
            uint32_t Padding[3];
        };

        struct GpuLight
        {
            float Position[3];
            uint32_t Type;
            float Direction[3];
            float Intensity;
            float Color[3];
            uint32_t CastShadows;
            float Width;
            float Height;
            uint32_t Padding0;
            uint32_t Padding1;
        };

        struct GpuConstants
        {
            float CameraPosition[3];
            uint32_t Width;
            float CameraForward[3];
            uint32_t Height;
            float CameraRight[3];
            uint32_t SamplesPerPixel;
            float CameraUp[3];
            uint32_t MaxReflectionBounces;
            float HalfWidth;
            float HalfHeight;
            uint32_t MaterialCount;
            uint32_t LightCount;
            float BackgroundTop[3];
            float AmbientIntensity;
            float BackgroundBottom[3];
            float ShadowStrength;
            uint32_t GammaCorrection;
            uint32_t PrimitiveCount;
            uint32_t TextureCount;
            uint32_t TextureTexelCount;
        };

        static_assert(sizeof(GpuVertex) == 16, "DXR vertex layout must match HLSL.");
        static_assert(sizeof(GpuTriangle) == 144, "DXR triangle layout must match HLSL.");
        static_assert(offsetof(GpuTriangle, TexCoord01) == 96, "DXR triangle field layout must match HLSL.");
        static_assert(offsetof(GpuTriangle, MaterialIndex) == 128, "DXR triangle field layout must match HLSL.");
        static_assert(sizeof(GpuMaterial) == 80, "DXR material layout must match HLSL.");
        static_assert(
            offsetof(GpuMaterial, AmbientShininess) == 16
                && offsetof(GpuMaterial, Specular) == 32
                && offsetof(GpuMaterial, Emission) == 48
                && offsetof(GpuMaterial, DiffuseTextureIndex) == 64,
            "DXR material field layout must match HLSL.");
        static_assert(sizeof(GpuLight) == 64, "DXR light layout must match HLSL.");
        static_assert(
            offsetof(GpuLight, Type) == 12
                && offsetof(GpuLight, Direction) == 16
                && offsetof(GpuLight, Intensity) == 28
                && offsetof(GpuLight, Color) == 32
                && offsetof(GpuLight, CastShadows) == 44
                && offsetof(GpuLight, Width) == 48,
            "DXR light field layout must match HLSL.");
        static_assert(
            sizeof(GpuConstants) == 128
                && offsetof(GpuConstants, ShadowStrength) == 108,
            "DXR constant layout must match HLSL.");

        bool IsFinite(const Float3& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
        }

        bool IsNonNegative(const Float3& value)
        {
            return value.X >= 0.0f && value.Y >= 0.0f && value.Z >= 0.0f;
        }

        bool CheckedMultiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result)
        {
            if (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)
            {
                return false;
            }

            result = left * right;
            return true;
        }

        bool CheckedAdd(std::uint64_t left, std::uint64_t right, std::uint64_t& result)
        {
            if (right > std::numeric_limits<std::uint64_t>::max() - left)
            {
                return false;
            }

            result = left + right;
            return true;
        }

        bool ValidateDxrInput(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            std::string& error)
        {
            if (settings.Width == 0u || settings.Height == 0u)
            {
                error = "Render dimensions must be non-zero.";
                return false;
            }

            constexpr std::uint32_t MaxRenderDimension = 16384u;
            if (settings.Width > MaxRenderDimension || settings.Height > MaxRenderDimension)
            {
                error = "Render dimensions exceed the supported limit.";
                return false;
            }

            if (settings.SamplesPerPixel == 0u || settings.SamplesPerPixel > 4096u)
            {
                error = "Samples per pixel must be in the range [1, 4096].";
                return false;
            }

            if (settings.MaxReflectionBounces > 32u)
            {
                error = "Reflection bounce count exceeds the supported limit.";
                return false;
            }

            if (!std::isfinite(settings.ShadowStrength) ||
                settings.ShadowStrength < 0.0f || settings.ShadowStrength > 1.0f)
            {
                error = "Shadow strength must be finite and in the range [0, 1].";
                return false;
            }

            if (!IsFinite(camera.Position) || !IsFinite(camera.Target) || !IsFinite(camera.Up) ||
                !std::isfinite(camera.VerticalFieldOfViewDegrees))
            {
                error = "Camera parameters must be finite.";
                return false;
            }

            if (camera.VerticalFieldOfViewDegrees <= 1.0f || camera.VerticalFieldOfViewDegrees >= 179.0f)
            {
                error = "Camera vertical field of view must be in the range (1, 179) degrees.";
                return false;
            }

            const Float3 cameraForward = {
                camera.Target.X - camera.Position.X,
                camera.Target.Y - camera.Position.Y,
                camera.Target.Z - camera.Position.Z};
            const float forwardLengthSquared = cameraForward.X * cameraForward.X +
                cameraForward.Y * cameraForward.Y + cameraForward.Z * cameraForward.Z;
            const float upLengthSquared = camera.Up.X * camera.Up.X + camera.Up.Y * camera.Up.Y +
                camera.Up.Z * camera.Up.Z;
            if (!IsFinite(cameraForward) || !std::isfinite(forwardLengthSquared) ||
                !std::isfinite(upLengthSquared) || forwardLengthSquared < 1.0e-10f ||
                upLengthSquared < 1.0e-10f)
            {
                error = "Camera direction and up vectors must be finite and non-zero.";
                return false;
            }

            const float inverseForwardLength = 1.0f / std::sqrt(forwardLengthSquared);
            const float inverseUpLength = 1.0f / std::sqrt(upLengthSquared);
            const Float3 normalizedForward = {
                cameraForward.X * inverseForwardLength,
                cameraForward.Y * inverseForwardLength,
                cameraForward.Z * inverseForwardLength};
            const Float3 normalizedUp = {
                camera.Up.X * inverseUpLength,
                camera.Up.Y * inverseUpLength,
                camera.Up.Z * inverseUpLength};
            const Float3 cameraRight = {
                normalizedForward.Y * normalizedUp.Z - normalizedForward.Z * normalizedUp.Y,
                normalizedForward.Z * normalizedUp.X - normalizedForward.X * normalizedUp.Z,
                normalizedForward.X * normalizedUp.Y - normalizedForward.Y * normalizedUp.X};
            const float rightLengthSquared = cameraRight.X * cameraRight.X + cameraRight.Y * cameraRight.Y +
                cameraRight.Z * cameraRight.Z;
            if (!std::isfinite(rightLengthSquared) || rightLengthSquared < 1.0e-10f)
            {
                error = "Camera direction and up vectors must not be parallel.";
                return false;
            }

            if (scene.Materials.empty() && (!scene.Triangles.empty() || !scene.Spheres.empty()))
            {
                error = "Scene geometry requires at least one material.";
                return false;
            }

            if (!IsFinite(scene.BackgroundBottom) || !IsFinite(scene.BackgroundTop) ||
                !std::isfinite(scene.AmbientIntensity) || scene.AmbientIntensity < 0.0f)
            {
                error = "Scene environment parameters must be finite and non-negative.";
                return false;
            }

            const std::uint64_t materialCount = static_cast<std::uint64_t>(scene.Materials.size());
            std::uint64_t lightCount =
                static_cast<std::uint64_t>(scene.DirectionalLights.size());
            if (!CheckedAdd(
                    lightCount,
                    static_cast<std::uint64_t>(scene.PointLights.size()),
                    lightCount)
                || !CheckedAdd(
                    lightCount,
                    static_cast<std::uint64_t>(scene.RectAreaLights.size()),
                    lightCount)
                || materialCount > std::numeric_limits<std::uint32_t>::max()
                || lightCount > std::numeric_limits<std::uint32_t>::max())
            {
                error = "DXR material or light count exceeds 32-bit shader indexing limits.";
                return false;
            }

            std::uint64_t expandedSphereTriangleCount = 0u;
            std::uint64_t totalTriangleCount = 0u;
            std::uint64_t vertexCount = 0u;
            if (!CheckedMultiply(
                    static_cast<std::uint64_t>(scene.Spheres.size()),
                    SphereTriangleCount,
                    expandedSphereTriangleCount) ||
                !CheckedAdd(
                    static_cast<std::uint64_t>(scene.Triangles.size()),
                    expandedSphereTriangleCount,
                    totalTriangleCount) ||
                !CheckedMultiply(totalTriangleCount, 3u, vertexCount) ||
                vertexCount > std::numeric_limits<std::uint32_t>::max())
            {
                error = "DXR geometry count exceeds 32-bit acceleration structure indexing limits.";
                return false;
            }

            std::uint64_t vertexBytes = 0u;
            std::uint64_t triangleBytes = 0u;
            std::uint64_t materialBytes = 0u;
            std::uint64_t lightBytes = 0u;
            if (!CheckedMultiply(vertexCount, sizeof(GpuVertex), vertexBytes) ||
                !CheckedMultiply(totalTriangleCount, sizeof(GpuTriangle), triangleBytes) ||
                !CheckedMultiply(materialCount, sizeof(GpuMaterial), materialBytes) ||
                !CheckedMultiply(lightCount, sizeof(GpuLight), lightBytes) ||
                vertexBytes > std::numeric_limits<std::size_t>::max() ||
                triangleBytes > std::numeric_limits<std::size_t>::max() ||
                materialBytes > std::numeric_limits<std::size_t>::max() ||
                lightBytes > std::numeric_limits<std::size_t>::max())
            {
                error = "DXR scene buffer size overflow.";
                return false;
            }

            std::uint64_t pixelCount = 0u;
            std::uint64_t outputBytes = 0u;
            std::uint64_t primaryRayCount = 0u;
            std::uint64_t maximumPathRayCount = 0u;
            if (!CheckedMultiply(settings.Width, settings.Height, pixelCount) ||
                !CheckedMultiply(pixelCount, sizeof(std::uint32_t), outputBytes) ||
                !CheckedMultiply(pixelCount, settings.SamplesPerPixel, primaryRayCount) ||
                !CheckedMultiply(primaryRayCount, settings.MaxReflectionBounces + 1u, maximumPathRayCount) ||
                outputBytes > std::numeric_limits<std::size_t>::max())
            {
                error = "DXR output or dispatch size overflow.";
                return false;
            }

            for (const Material& material : scene.Materials)
            {
                if (!IsFinite(material.Albedo) || !IsFinite(material.AmbientColor) ||
                    !IsFinite(material.SpecularColor) || !IsFinite(material.Emission) ||
                    !std::isfinite(material.Shininess) ||
                    !std::isfinite(material.Reflectivity) ||
                    !IsNonNegative(material.AmbientColor) ||
                    !IsNonNegative(material.SpecularColor) ||
                    material.Shininess < 0.0f)
                {
                    error = "Material parameters must be finite; ambient, specular, and shininess must be non-negative.";
                    return false;
                }
            }

            for (const Triangle& triangle : scene.Triangles)
            {
                if (triangle.MaterialIndex >= scene.Materials.size())
                {
                    error = "Triangle material index is out of range.";
                    return false;
                }

                if (!IsFinite(triangle.Position0) || !IsFinite(triangle.Position1) ||
                    !IsFinite(triangle.Position2) || !IsFinite(triangle.Normal0) ||
                    !IsFinite(triangle.Normal1) || !IsFinite(triangle.Normal2) ||
                    !IsFinite(triangle.Color0) || !IsFinite(triangle.Color1) ||
                    !IsFinite(triangle.Color2) ||
                    !std::isfinite(triangle.TexCoord0.X) ||
                    !std::isfinite(triangle.TexCoord0.Y) ||
                    !std::isfinite(triangle.TexCoord1.X) ||
                    !std::isfinite(triangle.TexCoord1.Y) ||
                    !std::isfinite(triangle.TexCoord2.X) ||
                    !std::isfinite(triangle.TexCoord2.Y))
                {
                    error = "Triangle geometry must be finite.";
                    return false;
                }

                const Float3 edge1 = {
                    triangle.Position1.X - triangle.Position0.X,
                    triangle.Position1.Y - triangle.Position0.Y,
                    triangle.Position1.Z - triangle.Position0.Z};
                const Float3 edge2 = {
                    triangle.Position2.X - triangle.Position0.X,
                    triangle.Position2.Y - triangle.Position0.Y,
                    triangle.Position2.Z - triangle.Position0.Z};
                const Float3 areaVector = {
                    edge1.Y * edge2.Z - edge1.Z * edge2.Y,
                    edge1.Z * edge2.X - edge1.X * edge2.Z,
                    edge1.X * edge2.Y - edge1.Y * edge2.X};
                const float edge1LengthSquared = edge1.X * edge1.X + edge1.Y * edge1.Y + edge1.Z * edge1.Z;
                const float edge2LengthSquared = edge2.X * edge2.X + edge2.Y * edge2.Y + edge2.Z * edge2.Z;
                const float areaSquared = areaVector.X * areaVector.X + areaVector.Y * areaVector.Y +
                    areaVector.Z * areaVector.Z;
                if (!IsFinite(edge1) || !IsFinite(edge2) || !IsFinite(areaVector) ||
                    !std::isfinite(edge1LengthSquared) || !std::isfinite(edge2LengthSquared) ||
                    !std::isfinite(areaSquared) || edge1LengthSquared < 1.0e-16f ||
                    edge2LengthSquared < 1.0e-16f || areaSquared < 1.0e-16f)
                {
                    error = "Triangle geometry must be non-degenerate and within finite float range.";
                    return false;
                }
            }

            for (const Sphere& sphere : scene.Spheres)
            {
                if (sphere.MaterialIndex >= scene.Materials.size())
                {
                    error = "Sphere material index is out of range.";
                    return false;
                }

                if (!IsFinite(sphere.Center) || !std::isfinite(sphere.Radius) || sphere.Radius <= 0.0f)
                {
                    error = "Sphere center and radius must be finite, and radius must be positive.";
                    return false;
                }

                const Float3 minimum = {
                    sphere.Center.X - sphere.Radius,
                    sphere.Center.Y - sphere.Radius,
                    sphere.Center.Z - sphere.Radius};
                const Float3 maximum = {
                    sphere.Center.X + sphere.Radius,
                    sphere.Center.Y + sphere.Radius,
                    sphere.Center.Z + sphere.Radius};
                if (!IsFinite(minimum) || !IsFinite(maximum))
                {
                    error = "Sphere bounds exceed finite float range.";
                    return false;
                }
            }

            for (const DirectionalLight& light : scene.DirectionalLights)
            {
                if (!IsFinite(light.DirectionToLight) || !IsFinite(light.Color) ||
                    !std::isfinite(light.Intensity) ||
                    light.Intensity < 0.0f)
                {
                    error = "Directional light parameters must be finite and intensity must be non-negative.";
                    return false;
                }

                const float directionLengthSquared = light.DirectionToLight.X * light.DirectionToLight.X +
                    light.DirectionToLight.Y * light.DirectionToLight.Y +
                    light.DirectionToLight.Z * light.DirectionToLight.Z;
                if (!std::isfinite(directionLengthSquared) || directionLengthSquared < 1.0e-12f)
                {
                    error = "Directional light direction must be finite and non-zero.";
                    return false;
                }
            }
            for (const PointLight& light : scene.PointLights)
            {
                if (!IsFinite(light.Position) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity)
                    || light.Intensity < 0.0f)
                {
                    error =
                        "Point light parameters must be finite and intensity "
                        "must be non-negative.";
                    return false;
                }
            }
            for (const RectAreaLight& light : scene.RectAreaLights)
            {
                const float directionLengthSquared =
                    light.Direction.X * light.Direction.X
                    + light.Direction.Y * light.Direction.Y
                    + light.Direction.Z * light.Direction.Z;
                if (!IsFinite(light.Position) || !IsFinite(light.Direction)
                    || !IsFinite(light.Color)
                    || !std::isfinite(light.Size.X)
                    || !std::isfinite(light.Size.Y)
                    || !std::isfinite(light.Intensity)
                    || light.Intensity < 0.0f
                    || !std::isfinite(directionLengthSquared)
                    || directionLengthSquared < 1.0e-12f
                    || light.Size.X <= 0.0f || light.Size.Y <= 0.0f)
                {
                    error =
                        "Rect area light parameters must be finite with a "
                        "non-zero direction, positive size, and non-negative "
                        "intensity.";
                    return false;
                }
                const Float3 emissionDirection = Normalize(light.Direction);
                const Float3 helper =
                    std::fabs(emissionDirection.Y) < 0.999f
                    ? Float3(0.0f, 1.0f, 0.0f)
                    : Float3(1.0f, 0.0f, 0.0f);
                const Float3 tangent =
                    Normalize(Cross(helper, emissionDirection));
                const Float3 bitangent =
                    Normalize(Cross(emissionDirection, tangent));
                const Float3 halfTangent =
                    tangent * (light.Size.X * 0.5f);
                const Float3 halfBitangent =
                    bitangent * (light.Size.Y * 0.5f);
                if (!IsFinite(light.Position + halfTangent + halfBitangent)
                    || !IsFinite(light.Position + halfTangent - halfBitangent)
                    || !IsFinite(light.Position - halfTangent + halfBitangent)
                    || !IsFinite(light.Position - halfTangent - halfBitangent))
                {
                    error = "Rect area light bounds exceed finite float range.";
                    return false;
                }
            }

            return true;
        }

        constexpr const char* DxrShaderSource = R"DXR(
static const uint DirectionalLightType = 0u;
static const uint PointLightType = 1u;
static const uint RectAreaLightType = 2u;
static const float RayEpsilon = 1.0e-4;

struct TriangleData
{
    float4 Normal0;
    float4 Normal1;
    float4 Normal2;
    float4 Color0;
    float4 Color1;
    float4 Color2;
    float4 TexCoord01;
    float4 TexCoord2;
    uint MaterialIndex;
    uint SmoothNormals;
    uint HasTexCoord;
    uint Padding;
};

struct MaterialData
{
    float4 AlbedoReflectivity;
    float4 AmbientShininess;
    float4 Specular;
    float4 Emission;
    uint DiffuseTextureIndex;
    uint3 Padding;
};

struct RayTextureData
{
    uint OffsetTexels;
    uint Width;
    uint Height;
    uint Padding;
};

struct LightData
{
    float3 Position;
    uint Type;
    float3 Direction;
    float Intensity;
    float3 Color;
    uint CastShadows;
    float Width;
    float Height;
    uint Padding0;
    uint Padding1;
};

struct Payload
{
    float Distance;
    float3 Normal;
    float3 Color;
    uint MaterialIndex;
    uint Hit;
    float2 TexCoord;
    uint HasTexCoord;
};

RaytracingAccelerationStructure SceneAcceleration : register(t0);
RWStructuredBuffer<uint> OutputPixels : register(u0);
RWStructuredBuffer<uint> Statistics : register(u1);
StructuredBuffer<float4> Vertices : register(t1);
StructuredBuffer<TriangleData> Triangles : register(t2);
StructuredBuffer<MaterialData> Materials : register(t3);
StructuredBuffer<LightData> Lights : register(t4);
StructuredBuffer<RayTextureData> Textures : register(t5);
StructuredBuffer<uint> TextureTexels : register(t6);

cbuffer Globals : register(b0)
{
    float3 CameraPosition;
    uint Width;
    float3 CameraForward;
    uint Height;
    float3 CameraRight;
    uint SamplesPerPixel;
    float3 CameraUp;
    uint MaxReflectionBounces;
    float HalfWidth;
    float HalfHeight;
    uint MaterialCount;
    uint LightCount;
    float3 BackgroundTop;
    float AmbientIntensity;
    float3 BackgroundBottom;
    float ShadowStrength;
    uint GammaCorrection;
    uint PrimitiveCount;
    uint TextureCount;
    uint TextureTexelCount;
};

uint Hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352d;
    value ^= value >> 15;
    value *= 0x846ca68b;
    value ^= value >> 16;
    return value;
}

float Random01(uint seed)
{
    return float(Hash(seed) & 0x00ffffff) / float(0x01000000);
}

float3 Background(float3 direction)
{
    float blend = saturate(direction.y * 0.5 + 0.5);
    return lerp(BackgroundBottom, BackgroundTop, blend);
}

float3 NormalizeSafe(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12
        ? value * rsqrt(lengthSquared)
        : fallback;
}

float3 LinearToSRGB(float3 color)
{
    color = saturate(color);
    float3 low = color * 12.92;
    float3 high = 1.055 * pow(color, 1.0 / 2.4) - 0.055;
    return float3(
        color.r <= 0.0031308 ? low.r : high.r,
        color.g <= 0.0031308 ? low.g : high.g,
        color.b <= 0.0031308 ? low.b : high.b);
}

float SRGBToLinear(float value)
{
    return value <= 0.04045
        ? value / 12.92
        : pow((value + 0.055) / 1.055, 2.4);
}

int WrapTextureCoordinate(int value, uint size)
{
    int signedSize = int(size);
    int wrapped = value % signedSize;
    return wrapped < 0 ? wrapped + signedSize : wrapped;
}

float3 LoadDiffuseTexel(RayTextureData descriptor, int x, int y)
{
    uint offset = descriptor.OffsetTexels
        + uint(WrapTextureCoordinate(y, descriptor.Height)) * descriptor.Width
        + uint(WrapTextureCoordinate(x, descriptor.Width));
    uint packed = TextureTexels[offset];
    return float3(
        SRGBToLinear(float(packed & 0xffu) / 255.0),
        SRGBToLinear(float((packed >> 8u) & 0xffu) / 255.0),
        SRGBToLinear(float((packed >> 16u) & 0xffu) / 255.0));
}

float3 SampleDiffuse(uint textureIndex, float2 texCoord, uint hasTexCoord)
{
    if (hasTexCoord == 0u || textureIndex >= TextureCount)
    {
        return 1.0.xxx;
    }
    RayTextureData descriptor = Textures[textureIndex];
    if (descriptor.Width == 0u || descriptor.Height == 0u)
    {
        return 1.0.xxx;
    }
    if (descriptor.Height > TextureTexelCount / descriptor.Width)
    {
        return 1.0.xxx;
    }
    uint pixelCount = descriptor.Width * descriptor.Height;
    if (descriptor.OffsetTexels > TextureTexelCount
        || pixelCount > TextureTexelCount - descriptor.OffsetTexels)
    {
        return 1.0.xxx;
    }
    float x = frac(texCoord.x) * float(descriptor.Width) - 0.5;
    float y = frac(1.0 - texCoord.y) * float(descriptor.Height) - 0.5;
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    float amountX = x - floor(x);
    float amountY = y - floor(y);
    float3 top = lerp(
        LoadDiffuseTexel(descriptor, x0, y0),
        LoadDiffuseTexel(descriptor, x0 + 1, y0),
        amountX);
    float3 bottom = lerp(
        LoadDiffuseTexel(descriptor, x0, y0 + 1),
        LoadDiffuseTexel(descriptor, x0 + 1, y0 + 1),
        amountX);
    return lerp(top, bottom, amountY);
}

Payload CastRay(float3 origin, float3 direction, float minimumDistance, float maximumDistance)
{
    Payload payload;
    payload.Distance = maximumDistance;
    payload.Normal = 0.0;
    payload.Color = 1.0;
    payload.MaterialIndex = 0;
    payload.Hit = 0;
    payload.TexCoord = 0.0;
    payload.HasTexCoord = 0;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = minimumDistance;
    ray.TMax = maximumDistance;
    TraceRay(
        SceneAcceleration,
        RAY_FLAG_FORCE_OPAQUE,
        0xff,
        0,
        1,
        0,
        ray,
        payload);
    return payload;
}

void AccumulateLightSample(
    float3 lightDirection,
    float3 incidentRadiance,
    float visibility,
    float3 surfaceNormal,
    float3 viewDirection,
    float shininess,
    inout float3 diffuseLight,
    inout float3 specularLight)
{
    float directionLengthSquared = dot(lightDirection, lightDirection);
    if (directionLengthSquared <= 1.0e-12)
    {
        return;
    }
    float3 direction = lightDirection * rsqrt(directionLengthSquared);
    float normalToLight = max(dot(surfaceNormal, direction), 0.0);
    if (normalToLight <= 0.0)
    {
        return;
    }
    float3 visibleRadiance = incidentRadiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    float3 halfVector = NormalizeSafe(
        direction + viewDirection,
        surfaceNormal);
    float specularAmount = pow(
        max(dot(surfaceNormal, halfVector), 0.0),
        max(shininess, 1.0));
    specularLight += visibleRadiance * specularAmount;
}

void AccumulateFiniteLight(
    LightData light,
    float3 lightPosition,
    float emitterWeight,
    bool areaEmitter,
    float3 position,
    float3 surfaceNormal,
    float3 viewDirection,
    float shininess,
    inout float3 diffuseLight,
    inout float3 specularLight)
{
    if (light.Intensity <= 0.0 || emitterWeight <= 0.0)
    {
        return;
    }
    float3 toLight = lightPosition - position;
    float distanceSquared = dot(toLight, toLight);
    if (!isfinite(distanceSquared)
        || distanceSquared <= RayEpsilon * RayEpsilon)
    {
        return;
    }
    float distance = sqrt(distanceSquared);
    float3 directionToLight = toLight / distance;
    float surfaceCosine = max(dot(surfaceNormal, directionToLight), 0.0);
    if (surfaceCosine <= 0.0)
    {
        return;
    }
    float emitterCosine = 1.0;
    if (areaEmitter)
    {
        emitterCosine =
            max(dot(light.Direction, -directionToLight), 0.0);
        if (emitterCosine <= 0.0)
        {
            return;
        }
    }

    float visibility = 1.0;
    if (ShadowStrength > 0.0 && light.CastShadows != 0u)
    {
        float maximumDistance = distance - RayEpsilon;
        if (maximumDistance > RayEpsilon)
        {
            InterlockedAdd(Statistics[1], 1);
            Payload shadow = CastRay(
                position + surfaceNormal * RayEpsilon,
                directionToLight,
                RayEpsilon,
                maximumDistance);
            if (shadow.Hit != 0)
            {
                visibility = 1.0 - ShadowStrength;
            }
        }
    }
    float3 incidentRadiance =
        max(light.Color, 0.0)
        * (max(light.Intensity, 0.0)
            * emitterCosine
            * emitterWeight
            / distanceSquared);
    AccumulateLightSample(
        directionToLight,
        incidentRadiance,
        visibility,
        surfaceNormal,
        viewDirection,
        shininess,
        diffuseLight,
        specularLight);
}

[shader("miss")]
void Miss(inout Payload payload)
{
    payload.Hit = 0;
}

[shader("closesthit")]
void ClosestHit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    uint primitive = PrimitiveIndex();
    TriangleData triangleData = Triangles[primitive];
    float barycentric0 = 1.0 - attributes.barycentrics.x - attributes.barycentrics.y;
    float3 normal;
    if (triangleData.SmoothNormals != 0)
    {
        normal = triangleData.Normal0.xyz * barycentric0
            + triangleData.Normal1.xyz * attributes.barycentrics.x
            + triangleData.Normal2.xyz * attributes.barycentrics.y;
    }
    else
    {
        uint firstVertex = primitive * 3;
        float3 edge1 = Vertices[firstVertex + 1].xyz - Vertices[firstVertex].xyz;
        float3 edge2 = Vertices[firstVertex + 2].xyz - Vertices[firstVertex].xyz;
        normal = cross(edge1, edge2);
    }
    if (dot(normal, normal) < 1.0e-12)
    {
        uint firstVertex = primitive * 3;
        normal = cross(
            Vertices[firstVertex + 1].xyz - Vertices[firstVertex].xyz,
            Vertices[firstVertex + 2].xyz - Vertices[firstVertex].xyz);
    }
    normal = normalize(normal);
    if (dot(normal, WorldRayDirection()) > 0.0) normal = -normal;
    payload.Distance = RayTCurrent();
    payload.Normal = normal;
    payload.Color = triangleData.Color0.xyz * barycentric0
        + triangleData.Color1.xyz * attributes.barycentrics.x
        + triangleData.Color2.xyz * attributes.barycentrics.y;
    payload.MaterialIndex = triangleData.MaterialIndex;
    payload.TexCoord = triangleData.TexCoord01.xy * barycentric0
        + triangleData.TexCoord01.zw * attributes.barycentrics.x
        + triangleData.TexCoord2.xy * attributes.barycentrics.y;
    payload.HasTexCoord = triangleData.HasTexCoord;
    payload.Hit = 1;
}

)DXR"
R"DXR(
[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= Width || pixel.y >= Height) return;

    float3 accumulated = 0.0;
    for (uint sample = 0; sample < SamplesPerPixel; ++sample)
    {
        InterlockedAdd(Statistics[0], 1);
        float jitterX = 0.5;
        float jitterY = 0.5;
        if (SamplesPerPixel > 1)
        {
            uint seed = (pixel.y * Width + pixel.x) * 9781 + sample * 6271;
            jitterX = Random01(seed);
            jitterY = Random01(seed ^ 0x68bc21eb);
        }
        float u = (float(pixel.x) + jitterX) / float(Width);
        float v = (float(pixel.y) + jitterY) / float(Height);
        float screenX = (2.0 * u - 1.0) * HalfWidth;
        float screenY = (1.0 - 2.0 * v) * HalfHeight;
        float3 origin = CameraPosition;
        float3 direction = normalize(
            CameraForward + CameraRight * screenX + CameraUp * screenY);
        float3 throughput = 1.0;
        float3 radiance = 0.0;

        for (uint bounce = 0; bounce <= MaxReflectionBounces; ++bounce)
        {
            if (PrimitiveCount == 0)
            {
                radiance += throughput * Background(direction);
                break;
            }
            Payload payload = CastRay(
                origin,
                direction,
                RayEpsilon,
                1.0e30);
            if (payload.Hit == 0)
            {
                radiance += throughput * Background(direction);
                break;
            }

            MaterialData material = Materials[min(payload.MaterialIndex, MaterialCount - 1)];
            float3 surface = max(payload.Color, 0.0)
                * max(
                    SampleDiffuse(
                        material.DiffuseTextureIndex,
                        payload.TexCoord,
                        payload.HasTexCoord),
                    0.0);
            float3 position = origin + direction * payload.Distance;
            float3 viewDirection = NormalizeSafe(-direction, payload.Normal);
            float3 diffuseLight = 0.0;
            float3 specularLight = 0.0;
            for (uint lightIndex = 0; lightIndex < LightCount; ++lightIndex)
            {
                LightData light = Lights[lightIndex];
                if (light.Type == DirectionalLightType)
                {
                    float3 directionToLight = light.Direction;
                    float cosine = max(0.0, dot(payload.Normal, directionToLight));
                    if (cosine <= 0.0 || light.Intensity <= 0.0)
                    {
                        continue;
                    }
                    float visibility = 1.0;
                    if (ShadowStrength > 0.0 && light.CastShadows != 0u)
                    {
                        InterlockedAdd(Statistics[1], 1);
                        float3 shadowNormal =
                            dot(payload.Normal, directionToLight) >= 0.0
                                ? payload.Normal
                                : -payload.Normal;
                        Payload shadow = CastRay(
                            position + shadowNormal * RayEpsilon,
                            directionToLight,
                            RayEpsilon,
                            1.0e30);
                        if (shadow.Hit != 0)
                        {
                            visibility = 1.0 - ShadowStrength;
                        }
                    }
                    AccumulateLightSample(
                        directionToLight,
                        max(light.Color, 0.0) * max(light.Intensity, 0.0),
                        visibility,
                        payload.Normal,
                        viewDirection,
                        material.AmbientShininess.w,
                        diffuseLight,
                        specularLight);
                }
                else if (light.Type == PointLightType)
                {
                    AccumulateFiniteLight(
                        light,
                        light.Position,
                        1.0,
                        false,
                        position,
                        payload.Normal,
                        viewDirection,
                        material.AmbientShininess.w,
                        diffuseLight,
                        specularLight);
                }
                else if (light.Type == RectAreaLightType)
                {
                    float3 emissionDirection = light.Direction;
                    float3 helper =
                        abs(emissionDirection.y) < 0.999
                        ? float3(0.0, 1.0, 0.0)
                        : float3(1.0, 0.0, 0.0);
                    float3 tangent =
                        normalize(cross(helper, emissionDirection));
                    float3 bitangent =
                        normalize(cross(emissionDirection, tangent));
                    [unroll]
                    for (uint sampleIndex = 0u; sampleIndex < 4u; ++sampleIndex)
                    {
                        float offsetX =
                            (sampleIndex & 1u) != 0u ? 0.25 : -0.25;
                        float offsetY =
                            (sampleIndex & 2u) != 0u ? 0.25 : -0.25;
                        float3 samplePosition =
                            light.Position
                            + tangent * (offsetX * light.Width)
                            + bitangent * (offsetY * light.Height);
                        AccumulateFiniteLight(
                            light,
                            samplePosition,
                            0.25,
                            true,
                            position,
                            payload.Normal,
                            viewDirection,
                            material.AmbientShininess.w,
                            diffuseLight,
                            specularLight);
                    }
                }
            }

            float3 direct =
                surface
                    * max(material.AmbientShininess.xyz, 0.0)
                    * max(AmbientIntensity, 0.0)
                + surface
                    * max(material.AlbedoReflectivity.xyz, 0.0)
                    * diffuseLight
                + max(material.Specular.xyz, 0.0) * specularLight
                + max(material.Emission.xyz, 0.0);
            float reflectivity = saturate(material.AlbedoReflectivity.w);
            radiance += throughput * direct * (1.0 - reflectivity);
            if (reflectivity <= 0.0 || bounce == MaxReflectionBounces) break;
            throughput *= reflectivity;
            float3 reflectedDirection =
                normalize(reflect(direction, payload.Normal));
            float3 reflectionNormal =
                dot(payload.Normal, reflectedDirection) >= 0.0
                    ? payload.Normal
                    : -payload.Normal;
            origin = position + reflectionNormal * RayEpsilon;
            direction = reflectedDirection;
            InterlockedAdd(Statistics[2], 1);
        }
        accumulated += radiance;
    }

    float3 color = max(accumulated / float(SamplesPerPixel), 0.0);
    color /= max(1.0, max(color.r, max(color.g, color.b)));
    if (GammaCorrection != 0) color = LinearToSRGB(color);
    uint3 encoded = uint3(color * 255.0 + 0.5);
    OutputPixels[pixel.y * Width + pixel.x] =
        encoded.x | (encoded.y << 8) | (encoded.z << 16) | 0xff000000;
    InterlockedAdd(Statistics[3], 1);
}
)DXR";

        std::string HResultMessage(const char* operation, HRESULT result)
        {
            std::ostringstream stream;
            stream << operation << " failed with HRESULT 0x" << std::hex
                << static_cast<uint32_t>(result) << '.';
            return stream.str();
        }

        std::string Narrow(const wchar_t* value)
        {
            if (!value || !*value) return {};
            const int required = WideCharToMultiByte(
                CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
            if (required <= 1) return {};
            std::string result(static_cast<size_t>(required), '\0');
            WideCharToMultiByte(
                CP_UTF8, 0, value, -1, result.data(), required, nullptr, nullptr);
            result.pop_back();
            return result;
        }

        D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type)
        {
            D3D12_HEAP_PROPERTIES properties{};
            properties.Type = type;
            properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            properties.CreationNodeMask = 1;
            properties.VisibleNodeMask = 1;
            return properties;
        }

        D3D12_RESOURCE_DESC BufferDescription(uint64_t size, D3D12_RESOURCE_FLAGS flags)
        {
            D3D12_RESOURCE_DESC description{};
            description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            description.Width = std::max<uint64_t>(size, 4);
            description.Height = 1;
            description.DepthOrArraySize = 1;
            description.MipLevels = 1;
            description.Format = DXGI_FORMAT_UNKNOWN;
            description.SampleDesc.Count = 1;
            description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            description.Flags = flags;
            return description;
        }

        D3D12_RESOURCE_BARRIER TransitionBarrier(
            ID3D12Resource* resource,
            D3D12_RESOURCE_STATES before,
            D3D12_RESOURCE_STATES after)
        {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = resource;
            barrier.Transition.StateBefore = before;
            barrier.Transition.StateAfter = after;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            return barrier;
        }

        D3D12_RESOURCE_BARRIER UavBarrier(ID3D12Resource* resource)
        {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = resource;
            return barrier;
        }

        void StoreFloat3(float destination[3], const Float3& value)
        {
            destination[0] = value.X;
            destination[1] = value.Y;
            destination[2] = value.Z;
        }

        void StoreFloat4(float destination[4], const Float3& value, float w)
        {
            destination[0] = value.X;
            destination[1] = value.Y;
            destination[2] = value.Z;
            destination[3] = w;
        }
#endif
    }

    struct DxrRayTracingRHI::Impl
    {
        Impl()
        {
#if defined(_WIN32)
            Initialize();
#else
            AvailabilityError = "DXR is available only on Windows.";
#endif
        }

        RenderJob Submit(
            const Scene* scene,
            std::shared_ptr<const Scene> owner,
            Camera camera,
            RenderSettings settings,
            CancellationToken external = CancellationToken()) const
        {
            return RHIQueue.Submit(
                [this, scene, owner = std::move(owner), camera, settings, external]
                (const CancellationToken& queued) mutable
                {
                    (void)owner;
                    if (!scene)
                    {
                        return Detail::MakeFailedResult("The DXR submission has no scene.");
                    }
                    const Detail::CancellationProbe cancellation =
                        Detail::CombineCancellation(queued, external);
                    if (cancellation()) return Detail::MakeCancelledResult();
                    if (!Available)
                    {
                        return Detail::MakeFailedResult(
                            std::string("DXR is unavailable: ") + AvailabilityError);
                    }
                    ++DispatchRaysSequence;
#if defined(_WIN32)
                    return RenderNative(*scene, camera, settings, cancellation);
#else
                    return Detail::MakeFailedResult("DXR is available only on Windows.");
#endif
                });
        }

#if defined(_WIN32)
        using CreateFactoryFunction = HRESULT(WINAPI*)(UINT, REFIID, void**);
        using CreateDeviceFunction = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
        using SerializeRootSignatureFunction = HRESULT(WINAPI*)(
            const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);

        bool Initialize()
        {
            if (!DxgiModule.Load(L"dxgi.dll"))
            {
                AvailabilityError = "dxgi.dll could not be loaded.";
                return false;
            }
            if (!D3D12Module.Load(L"d3d12.dll"))
            {
                AvailabilityError = "d3d12.dll could not be loaded.";
                return false;
            }

            const auto createFactory = DxgiModule.Get<CreateFactoryFunction>("CreateDXGIFactory2");
            CreateDevice = D3D12Module.Get<CreateDeviceFunction>("D3D12CreateDevice");
            SerializeRootSignature =
                D3D12Module.Get<SerializeRootSignatureFunction>("D3D12SerializeRootSignature");
            if (!createFactory || !CreateDevice || !SerializeRootSignature)
            {
                AvailabilityError = "Required DXGI or D3D12 entry points are missing.";
                return false;
            }

            HRESULT result = createFactory(0, IID_PPV_ARGS(&Factory));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("CreateDXGIFactory2", result);
                return false;
            }

            for (UINT adapterIndex = 0; ; ++adapterIndex)
            {
                ComPtr<IDXGIAdapter1> candidate;
                if (Factory->EnumAdapters1(adapterIndex, &candidate) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 description{};
                candidate->GetDesc1(&description);
                if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) continue;

                ComPtr<ID3D12Device> baseDevice;
                result = CreateDevice(
                    candidate.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&baseDevice));
                if (FAILED(result)) continue;
                ComPtr<ID3D12Device5> rayTracingDevice;
                if (FAILED(baseDevice.As(&rayTracingDevice))) continue;

                D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
                if (FAILED(rayTracingDevice->CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options))))
                {
                    continue;
                }
                if (options.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) continue;

                Adapter = std::move(candidate);
                Device = std::move(rayTracingDevice);
                AdapterName = Narrow(description.Description);
                RayTracingTier = options.RaytracingTier;
                break;
            }

            if (!Device)
            {
                AvailabilityError = "No hardware adapter exposes DirectX Raytracing.";
                return false;
            }

            D3D12_COMMAND_QUEUE_DESC queueDescription{};
            queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            result = Device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&CommandQueue));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("ID3D12Device::CreateCommandQueue", result);
                return false;
            }
            result = Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Fence));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("ID3D12Device::CreateFence", result);
                return false;
            }
            FenceEvent.Handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!FenceEvent.Handle)
            {
                AvailabilityError = "The DXR fence event could not be created.";
                return false;
            }

            if (!CreateGlobalRootSignature()) return false;
            if (!CreateRayTracingStateObject()) return false;
            Available = true;
            return true;
        }

        bool CreateGlobalRootSignature()
        {
            std::array<D3D12_ROOT_PARAMETER, 10> parameters{};
            const std::array<D3D12_ROOT_PARAMETER_TYPE, 10> types = {
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_UAV,
                D3D12_ROOT_PARAMETER_TYPE_UAV,
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_CBV};
            const std::array<UINT, 10> registers = {
                0, 0, 1, 1, 2, 3, 4, 5, 6, 0};
            for (size_t index = 0; index < parameters.size(); ++index)
            {
                parameters[index].ParameterType = types[index];
                parameters[index].Descriptor.ShaderRegister = registers[index];
                parameters[index].Descriptor.RegisterSpace = 0;
                parameters[index].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            }

            D3D12_ROOT_SIGNATURE_DESC description{};
            description.NumParameters = static_cast<UINT>(parameters.size());
            description.pParameters = parameters.data();
            description.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
            ComPtr<ID3DBlob> serialized;
            ComPtr<ID3DBlob> errors;
            HRESULT result = SerializeRootSignature(
                &description, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
            if (FAILED(result))
            {
                AvailabilityError = errors && errors->GetBufferPointer()
                    ? std::string(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize())
                    : HResultMessage("D3D12SerializeRootSignature", result);
                return false;
            }
            result = Device->CreateRootSignature(
                0,
                serialized->GetBufferPointer(),
                serialized->GetBufferSize(),
                IID_PPV_ARGS(&RootSignature));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("ID3D12Device::CreateRootSignature", result);
                return false;
            }
            return true;
        }

        bool CompileShaderLibrary(ComPtr<IDxcBlob>& library)
        {
            if (!DxcModule.Load(L"dxcompiler.dll"))
            {
                AvailabilityError =
                    "dxcompiler.dll is required to compile the embedded DXR shader library.";
                return false;
            }
            const auto createInstance = DxcModule.Get<DxcCreateInstanceProc>("DxcCreateInstance");
            if (!createInstance)
            {
                AvailabilityError = "DxcCreateInstance is missing from dxcompiler.dll.";
                return false;
            }

            ComPtr<IDxcUtils> utils;
            ComPtr<IDxcCompiler3> compiler;
            HRESULT result = createInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("DxcCreateInstance(IDxcUtils)", result);
                return false;
            }
            result = createInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("DxcCreateInstance(IDxcCompiler3)", result);
                return false;
            }

            DxcBuffer source{};
            source.Ptr = DxrShaderSource;
            source.Size = std::strlen(DxrShaderSource);
            source.Encoding = DXC_CP_UTF8;
            const wchar_t* arguments[] = {
                L"-T", L"lib_6_3", L"-HV", L"2021", L"-O3", L"-Qstrip_debug"};
            ComPtr<IDxcResult> compilation;
            result = compiler->Compile(
                &source,
                arguments,
                static_cast<UINT32>(std::size(arguments)),
                nullptr,
                IID_PPV_ARGS(&compilation));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("IDxcCompiler3::Compile", result);
                return false;
            }

            HRESULT compilationStatus = E_FAIL;
            compilation->GetStatus(&compilationStatus);
            if (FAILED(compilationStatus))
            {
                ComPtr<IDxcBlobUtf8> diagnostics;
                compilation->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&diagnostics), nullptr);
                AvailabilityError = diagnostics && diagnostics->GetStringLength() != 0
                    ? std::string(diagnostics->GetStringPointer(), diagnostics->GetStringLength())
                    : HResultMessage("DXR shader compilation", compilationStatus);
                return false;
            }
            result = compilation->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&library), nullptr);
            if (FAILED(result) || !library)
            {
                AvailabilityError = HResultMessage("IDxcResult::GetOutput", result);
                return false;
            }
            return true;
        }

        bool CreateRayTracingStateObject()
        {
            const void* shaderBytecode = Detail::DxrShaderBytecode.data();
            size_t shaderBytecodeSize = Detail::DxrShaderBytecode.size();
            ComPtr<IDxcBlob> recompiledShaderLibrary;
            wchar_t recompileValue[2]{};
            if (GetEnvironmentVariableW(
                    L"EVANS_DXR_RECOMPILE_SHADERS",
                    recompileValue,
                    static_cast<DWORD>(std::size(recompileValue))) != 0
                && recompileValue[0] == L'1')
            {
                if (CompileShaderLibrary(recompiledShaderLibrary))
                {
                    shaderBytecode = recompiledShaderLibrary->GetBufferPointer();
                    shaderBytecodeSize = recompiledShaderLibrary->GetBufferSize();
                }
                else
                {
                    ShaderCompilerDiagnostics = AvailabilityError;
                    AvailabilityError.clear();
                }
            }

            D3D12_EXPORT_DESC libraryExports[] = {
                {L"RayGen", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"Miss", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"ClosestHit", nullptr, D3D12_EXPORT_FLAG_NONE}};
            D3D12_DXIL_LIBRARY_DESC libraryDescription{};
            libraryDescription.DXILLibrary.pShaderBytecode = shaderBytecode;
            libraryDescription.DXILLibrary.BytecodeLength = shaderBytecodeSize;
            libraryDescription.NumExports = static_cast<UINT>(std::size(libraryExports));
            libraryDescription.pExports = libraryExports;

            D3D12_HIT_GROUP_DESC hitGroup{};
            hitGroup.HitGroupExport = L"HitGroup";
            hitGroup.ClosestHitShaderImport = L"ClosestHit";
            hitGroup.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;

            D3D12_RAYTRACING_SHADER_CONFIG shaderConfig{};
            shaderConfig.MaxPayloadSizeInBytes = 48;
            shaderConfig.MaxAttributeSizeInBytes = D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES;

            D3D12_GLOBAL_ROOT_SIGNATURE globalRoot{};
            globalRoot.pGlobalRootSignature = RootSignature.Get();

            D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig{};
            pipelineConfig.MaxTraceRecursionDepth = 1;

            std::array<D3D12_STATE_SUBOBJECT, 6> subobjects{};
            subobjects[0] = {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &libraryDescription};
            subobjects[1] = {D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hitGroup};
            subobjects[2] = {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderConfig};
            const wchar_t* associatedExports[] = {L"RayGen", L"Miss", L"HitGroup"};
            D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION association{};
            association.pSubobjectToAssociate = &subobjects[2];
            association.NumExports = static_cast<UINT>(std::size(associatedExports));
            association.pExports = associatedExports;
            subobjects[3] = {D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION, &association};
            subobjects[4] = {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &globalRoot};
            subobjects[5] = {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipelineConfig};

            D3D12_STATE_OBJECT_DESC description{};
            description.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
            description.NumSubobjects = static_cast<UINT>(subobjects.size());
            description.pSubobjects = subobjects.data();
            HRESULT result = Device->CreateStateObject(&description, IID_PPV_ARGS(&StateObject));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("ID3D12Device5::CreateStateObject", result);
                return false;
            }
            result = StateObject.As(&StateObjectProperties);
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("ID3D12StateObject::QueryInterface", result);
                return false;
            }

            const void* rayGeneration = StateObjectProperties->GetShaderIdentifier(L"RayGen");
            const void* miss = StateObjectProperties->GetShaderIdentifier(L"Miss");
            const void* hit = StateObjectProperties->GetShaderIdentifier(L"HitGroup");
            if (!rayGeneration || !miss || !hit)
            {
                AvailabilityError = "The DXR state object did not expose all shader identifiers.";
                return false;
            }
            std::memcpy(ShaderIdentifiers.data(), rayGeneration, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            std::memcpy(
                ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
                miss,
                D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            std::memcpy(
                ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * 2,
                hit,
                D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            return true;
        }

        bool CreateBuffer(
            uint64_t size,
            D3D12_HEAP_TYPE heapType,
            D3D12_RESOURCE_STATES initialState,
            D3D12_RESOURCE_FLAGS flags,
            ComPtr<ID3D12Resource>& resource,
            std::string& error) const
        {
            const D3D12_HEAP_PROPERTIES heap = HeapProperties(heapType);
            const D3D12_RESOURCE_DESC description = BufferDescription(size, flags);
            const HRESULT result = Device->CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &description,
                initialState,
                nullptr,
                IID_PPV_ARGS(&resource));
            if (FAILED(result))
            {
                error = HResultMessage("ID3D12Device::CreateCommittedResource", result);
                return false;
            }
            return true;
        }

        bool CreateUploadBuffer(
            const void* data,
            uint64_t size,
            ComPtr<ID3D12Resource>& resource,
            std::string& error) const
        {
            if (!CreateBuffer(
                std::max<uint64_t>(size, 4),
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                D3D12_RESOURCE_FLAG_NONE,
                resource,
                error))
            {
                return false;
            }
            if (data && size != 0)
            {
                void* destination = nullptr;
                const D3D12_RANGE noRead{0, 0};
                const HRESULT result = resource->Map(0, &noRead, &destination);
                if (FAILED(result))
                {
                    error = HResultMessage("ID3D12Resource::Map", result);
                    return false;
                }
                std::memcpy(destination, data, static_cast<size_t>(size));
                resource->Unmap(0, nullptr);
            }
            return true;
        }

        bool ExecuteAndWait(ID3D12CommandList* commandList, std::string& error) const
        {
            ID3D12CommandList* commandLists[] = {commandList};
            CommandQueue->ExecuteCommandLists(1, commandLists);
            const uint64_t value = ++FenceValue;
            HRESULT result = CommandQueue->Signal(Fence.Get(), value);
            if (FAILED(result))
            {
                error = HResultMessage("ID3D12CommandQueue::Signal", result);
                return false;
            }
            if (Fence->GetCompletedValue() < value)
            {
                result = Fence->SetEventOnCompletion(value, FenceEvent.Handle);
                if (FAILED(result))
                {
                    error = HResultMessage("ID3D12Fence::SetEventOnCompletion", result);
                    return false;
                }
                const DWORD waitResult =
                    WaitForSingleObject(FenceEvent.Handle, GpuWaitTimeoutMilliseconds);
                if (waitResult != WAIT_OBJECT_0)
                {
                    const HRESULT removedReason = Device->GetDeviceRemovedReason();
                    error = FAILED(removedReason)
                        ? HResultMessage("DXR device removal", removedReason)
                        : waitResult == WAIT_TIMEOUT
                        ? "Waiting for the DXR command queue timed out."
                        : "Waiting for the DXR command queue failed.";
                    return false;
                }
            }
            return true;
        }

        void AppendTriangle(
            const Float3& position0,
            const Float3& position1,
            const Float3& position2,
            const Float3& normal0,
            const Float3& normal1,
            const Float3& normal2,
            const Float3& color0,
            const Float3& color1,
            const Float3& color2,
            uint32_t materialIndex,
            bool smoothNormals,
            std::vector<GpuVertex>& vertices,
            std::vector<GpuTriangle>& triangles,
            bool hasTexCoord = false,
            const Float2& texCoord0 = Float2(),
            const Float2& texCoord1 = Float2(),
            const Float2& texCoord2 = Float2()) const
        {
            vertices.push_back({position0.X, position0.Y, position0.Z, 0.0f});
            vertices.push_back({position1.X, position1.Y, position1.Z, 0.0f});
            vertices.push_back({position2.X, position2.Y, position2.Z, 0.0f});
            GpuTriangle triangle{};
            StoreFloat4(triangle.Normal0, normal0, 0.0f);
            StoreFloat4(triangle.Normal1, normal1, 0.0f);
            StoreFloat4(triangle.Normal2, normal2, 0.0f);
            StoreFloat4(triangle.Color0, color0, 1.0f);
            StoreFloat4(triangle.Color1, color1, 1.0f);
            StoreFloat4(triangle.Color2, color2, 1.0f);
            triangle.TexCoord01[0] = texCoord0.X;
            triangle.TexCoord01[1] = texCoord0.Y;
            triangle.TexCoord01[2] = texCoord1.X;
            triangle.TexCoord01[3] = texCoord1.Y;
            triangle.TexCoord2[0] = texCoord2.X;
            triangle.TexCoord2[1] = texCoord2.Y;
            triangle.MaterialIndex = materialIndex;
            triangle.SmoothNormals = smoothNormals ? 1u : 0u;
            triangle.HasTexCoord = hasTexCoord ? 1u : 0u;
            triangles.push_back(triangle);
        }

        bool BuildSceneData(
            const Scene& scene,
            std::vector<GpuVertex>& vertices,
            std::vector<GpuTriangle>& triangles,
            std::vector<GpuMaterial>& materials,
            std::vector<GpuLight>& lights,
            std::string& error) const
        {
            constexpr float Pi = 3.14159265358979323846f;
            const uint64_t sphereTriangles = static_cast<uint64_t>(scene.Spheres.size())
                * SphereTriangleCount;
            const uint64_t totalTriangles = scene.Triangles.size() + sphereTriangles;
            if (scene.Materials.size() > std::numeric_limits<uint32_t>::max()
                || scene.DirectionalLights.size() > std::numeric_limits<uint32_t>::max()
                || totalTriangles > std::numeric_limits<uint32_t>::max() / 3u)
            {
                error = "The scene expands beyond the DXR 32-bit vertex limit.";
                return false;
            }
            vertices.reserve(static_cast<size_t>(totalTriangles * 3));
            triangles.reserve(static_cast<size_t>(totalTriangles));

            for (const Triangle& source : scene.Triangles)
            {
                AppendTriangle(
                    source.Position0,
                    source.Position1,
                    source.Position2,
                    source.Normal0,
                    source.Normal1,
                    source.Normal2,
                    source.Color0,
                    source.Color1,
                    source.Color2,
                    source.MaterialIndex,
                    source.SmoothNormals,
                    vertices,
                    triangles,
                    true,
                    source.TexCoord0,
                    source.TexCoord1,
                    source.TexCoord2);
            }

            for (const Sphere& sphere : scene.Spheres)
            {
                for (uint32_t latitude = 0; latitude < SphereLatitudeSegments; ++latitude)
                {
                    const float theta0 = -Pi * 0.5f
                        + Pi * static_cast<float>(latitude) / SphereLatitudeSegments;
                    const float theta1 = -Pi * 0.5f
                        + Pi * static_cast<float>(latitude + 1) / SphereLatitudeSegments;
                    for (uint32_t longitude = 0; longitude < SphereLongitudeSegments; ++longitude)
                    {
                        const float phi0 = 2.0f * Pi * static_cast<float>(longitude) / SphereLongitudeSegments;
                        const float phi1 = 2.0f * Pi * static_cast<float>(longitude + 1) / SphereLongitudeSegments;
                        const auto sphereNormal = [](float theta, float phi)
                        {
                            const float cosine = std::cos(theta);
                            return Float3(cosine * std::cos(phi), std::sin(theta), cosine * std::sin(phi));
                        };
                        const Float3 normal00 = sphereNormal(theta0, phi0);
                        const Float3 normal01 = sphereNormal(theta0, phi1);
                        const Float3 normal10 = sphereNormal(theta1, phi0);
                        const Float3 normal11 = sphereNormal(theta1, phi1);
                        const Float3 position00 = sphere.Center + normal00 * sphere.Radius;
                        const Float3 position01 = sphere.Center + normal01 * sphere.Radius;
                        const Float3 position10 = sphere.Center + normal10 * sphere.Radius;
                        const Float3 position11 = sphere.Center + normal11 * sphere.Radius;
                        if (latitude != 0)
                        {
                            AppendTriangle(
                                position00, position11, position01,
                                normal00, normal11, normal01,
                                Float3(1.0f, 1.0f, 1.0f),
                                Float3(1.0f, 1.0f, 1.0f),
                                Float3(1.0f, 1.0f, 1.0f),
                                sphere.MaterialIndex, true, vertices, triangles);
                        }
                        if (latitude + 1 != SphereLatitudeSegments)
                        {
                            AppendTriangle(
                                position00, position10, position11,
                                normal00, normal10, normal11,
                                Float3(1.0f, 1.0f, 1.0f),
                                Float3(1.0f, 1.0f, 1.0f),
                                Float3(1.0f, 1.0f, 1.0f),
                                sphere.MaterialIndex, true, vertices, triangles);
                        }
                    }
                }
            }

            materials.reserve(scene.Materials.size());
            for (const Material& source : scene.Materials)
            {
                GpuMaterial material{};
                StoreFloat4(material.AlbedoReflectivity, source.Albedo, source.Reflectivity);
                StoreFloat4(material.AmbientShininess, source.AmbientColor, source.Shininess);
                StoreFloat4(material.Specular, source.SpecularColor, 0.0f);
                StoreFloat4(material.Emission, source.Emission, 0.0f);
                material.DiffuseTextureIndex = source.DiffuseTextureIndex;
                materials.push_back(material);
            }
            lights.reserve(
                scene.DirectionalLights.size()
                + scene.PointLights.size()
                + scene.RectAreaLights.size());
            for (const DirectionalLight& source : scene.DirectionalLights)
            {
                GpuLight light{};
                light.Type = DirectionalLightType;
                StoreFloat3(light.Direction, Normalize(source.DirectionToLight));
                light.Intensity = source.Intensity;
                StoreFloat3(light.Color, source.Color);
                light.CastShadows = source.CastShadows ? 1u : 0u;
                lights.push_back(light);
            }
            for (const PointLight& source : scene.PointLights)
            {
                GpuLight light{};
                StoreFloat3(light.Position, source.Position);
                light.Type = PointLightType;
                light.Intensity = source.Intensity;
                StoreFloat3(light.Color, source.Color);
                light.CastShadows = source.CastShadows ? 1u : 0u;
                lights.push_back(light);
            }
            for (const RectAreaLight& source : scene.RectAreaLights)
            {
                GpuLight light{};
                StoreFloat3(light.Position, source.Position);
                light.Type = RectAreaLightType;
                StoreFloat3(light.Direction, Normalize(source.Direction));
                light.Intensity = source.Intensity;
                StoreFloat3(light.Color, source.Color);
                light.CastShadows = source.CastShadows ? 1u : 0u;
                light.Width = source.Size.X;
                light.Height = source.Size.Y;
                lights.push_back(light);
            }
            return true;
        }

        RenderResult RenderNative(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            const Detail::CancellationProbe& cancellation) const
        {
            RenderResult renderResult;
            if (!ValidateDxrInput(scene, camera, settings, renderResult.Error))
            {
                return renderResult;
            }
            if (cancellation()) return Detail::MakeCancelledResult();

            std::vector<GpuVertex> vertices;
            std::vector<GpuTriangle> triangles;
            std::vector<GpuMaterial> materials;
            std::vector<GpuLight> lights;
            Detail::PackedRayTextureAtlas textureAtlas;
            if (!BuildSceneData(scene, vertices, triangles, materials, lights, renderResult.Error))
            {
                return renderResult;
            }
            if (!Detail::BuildRayTextureAtlas(
                    scene,
                    textureAtlas,
                    renderResult.Error))
            {
                return renderResult;
            }

            ComPtr<ID3D12Resource> vertexBuffer;
            ComPtr<ID3D12Resource> triangleBuffer;
            ComPtr<ID3D12Resource> materialBuffer;
            ComPtr<ID3D12Resource> lightBuffer;
            ComPtr<ID3D12Resource> textureDescriptorBuffer;
            ComPtr<ID3D12Resource> textureTexelBuffer;
            if (!CreateUploadBuffer(
                    vertices.data(), vertices.size() * sizeof(GpuVertex), vertexBuffer, renderResult.Error)
                || !CreateUploadBuffer(
                    triangles.data(), triangles.size() * sizeof(GpuTriangle), triangleBuffer, renderResult.Error)
                || !CreateUploadBuffer(
                    materials.data(), materials.size() * sizeof(GpuMaterial), materialBuffer, renderResult.Error)
                || !CreateUploadBuffer(
                    lights.data(), lights.size() * sizeof(GpuLight), lightBuffer, renderResult.Error)
                || !CreateUploadBuffer(
                    textureAtlas.Descriptors.data(),
                    textureAtlas.Descriptors.size()
                        * sizeof(Detail::PackedRayTextureDescriptor),
                    textureDescriptorBuffer,
                    renderResult.Error)
                || !CreateUploadBuffer(
                    textureAtlas.Texels.data(),
                    textureAtlas.Texels.size() * sizeof(std::uint32_t),
                    textureTexelBuffer,
                    renderResult.Error))
            {
                return renderResult;
            }

            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList> baseCommandList;
            ComPtr<ID3D12GraphicsCommandList4> commandList;
            HRESULT result = Device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (FAILED(result))
            {
                renderResult.Error = HResultMessage("ID3D12Device::CreateCommandAllocator", result);
                return renderResult;
            }
            result = Device->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocator.Get(),
                nullptr,
                IID_PPV_ARGS(&baseCommandList));
            if (SUCCEEDED(result)) result = baseCommandList.As(&commandList);
            if (FAILED(result))
            {
                renderResult.Error = HResultMessage("ID3D12Device::CreateCommandList", result);
                return renderResult;
            }

            ComPtr<ID3D12Resource> bottomLevel;
            ComPtr<ID3D12Resource> bottomScratch;
            ComPtr<ID3D12Resource> topLevel;
            ComPtr<ID3D12Resource> topScratch;
            ComPtr<ID3D12Resource> instanceBuffer;
            D3D12_GPU_VIRTUAL_ADDRESS topLevelAddress = 0;
            const auto buildBegin = std::chrono::steady_clock::now();
            if (!triangles.empty())
            {
                D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
                geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
                geometry.Triangles.VertexBuffer.StartAddress = vertexBuffer->GetGPUVirtualAddress();
                geometry.Triangles.VertexBuffer.StrideInBytes = sizeof(GpuVertex);
                geometry.Triangles.VertexCount = static_cast<UINT>(vertices.size());
                geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
                geometry.Triangles.IndexFormat = DXGI_FORMAT_UNKNOWN;

                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottomInputs{};
                bottomInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
                bottomInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
                bottomInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
                bottomInputs.NumDescs = 1;
                bottomInputs.pGeometryDescs = &geometry;
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomInfo{};
                Device->GetRaytracingAccelerationStructurePrebuildInfo(&bottomInputs, &bottomInfo);
                if (bottomInfo.ResultDataMaxSizeInBytes == 0)
                {
                    renderResult.Error = "DXR returned an empty BLAS prebuild description.";
                    return renderResult;
                }
                if (!CreateBuffer(
                        AlignUp(bottomInfo.ScratchDataSizeInBytes, 256),
                        D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        bottomScratch,
                        renderResult.Error)
                    || !CreateBuffer(
                        AlignUp(bottomInfo.ResultDataMaxSizeInBytes, 256),
                        D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        bottomLevel,
                        renderResult.Error))
                {
                    return renderResult;
                }

                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomBuild{};
                bottomBuild.Inputs = bottomInputs;
                bottomBuild.ScratchAccelerationStructureData = bottomScratch->GetGPUVirtualAddress();
                bottomBuild.DestAccelerationStructureData = bottomLevel->GetGPUVirtualAddress();
                commandList->BuildRaytracingAccelerationStructure(&bottomBuild, 0, nullptr);
                D3D12_RESOURCE_BARRIER bottomBarrier = UavBarrier(bottomLevel.Get());
                commandList->ResourceBarrier(1, &bottomBarrier);

                D3D12_RAYTRACING_INSTANCE_DESC instance{};
                instance.Transform[0][0] = 1.0f;
                instance.Transform[1][1] = 1.0f;
                instance.Transform[2][2] = 1.0f;
                instance.InstanceMask = 0xff;
                instance.AccelerationStructure = bottomLevel->GetGPUVirtualAddress();
                if (!CreateUploadBuffer(
                    &instance, sizeof(instance), instanceBuffer, renderResult.Error))
                {
                    return renderResult;
                }

                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS topInputs{};
                topInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
                topInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
                topInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
                topInputs.NumDescs = 1;
                topInputs.InstanceDescs = instanceBuffer->GetGPUVirtualAddress();
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topInfo{};
                Device->GetRaytracingAccelerationStructurePrebuildInfo(&topInputs, &topInfo);
                if (!CreateBuffer(
                        AlignUp(topInfo.ScratchDataSizeInBytes, 256),
                        D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        topScratch,
                        renderResult.Error)
                    || !CreateBuffer(
                        AlignUp(topInfo.ResultDataMaxSizeInBytes, 256),
                        D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        topLevel,
                        renderResult.Error))
                {
                    return renderResult;
                }
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topBuild{};
                topBuild.Inputs = topInputs;
                topBuild.ScratchAccelerationStructureData = topScratch->GetGPUVirtualAddress();
                topBuild.DestAccelerationStructureData = topLevel->GetGPUVirtualAddress();
                commandList->BuildRaytracingAccelerationStructure(&topBuild, 0, nullptr);
                D3D12_RESOURCE_BARRIER topBarrier = UavBarrier(topLevel.Get());
                commandList->ResourceBarrier(1, &topBarrier);
                topLevelAddress = topLevel->GetGPUVirtualAddress();
            }
            result = commandList->Close();
            if (FAILED(result) || !ExecuteAndWait(commandList.Get(), renderResult.Error))
            {
                if (renderResult.Error.empty())
                {
                    renderResult.Error = HResultMessage("ID3D12GraphicsCommandList::Close", result);
                }
                return renderResult;
            }
            const auto buildEnd = std::chrono::steady_clock::now();
            if (cancellation()) return Detail::MakeCancelledResult();

            result = allocator->Reset();
            if (SUCCEEDED(result)) result = commandList->Reset(allocator.Get(), nullptr);
            if (FAILED(result))
            {
                renderResult.Error = HResultMessage("Resetting the DXR command list", result);
                return renderResult;
            }

            const uint64_t pixelCount = static_cast<uint64_t>(settings.Width) * settings.Height;
            const uint64_t outputSize = pixelCount * sizeof(uint32_t);
            constexpr uint64_t StatisticsSize = sizeof(uint32_t) * 4;
            ComPtr<ID3D12Resource> outputBuffer;
            ComPtr<ID3D12Resource> outputReadback;
            ComPtr<ID3D12Resource> statisticsBuffer;
            ComPtr<ID3D12Resource> statisticsReadback;
            ComPtr<ID3D12Resource> zeroUpload;
            const std::array<uint8_t, StatisticsSize> zeroData{};
            if (!CreateBuffer(
                    outputSize,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    outputBuffer,
                    renderResult.Error)
                || !CreateBuffer(
                    outputSize,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE,
                    outputReadback,
                    renderResult.Error)
                || !CreateBuffer(
                    StatisticsSize,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    statisticsBuffer,
                    renderResult.Error)
                || !CreateBuffer(
                    StatisticsSize,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE,
                    statisticsReadback,
                    renderResult.Error)
                || !CreateUploadBuffer(
                    zeroData.data(), zeroData.size(), zeroUpload, renderResult.Error))
            {
                return renderResult;
            }

            GpuConstants constants{};
            const Float3 forward = Normalize(camera.Target - camera.Position);
            const Float3 right = Normalize(Cross(forward, camera.Up));
            const Float3 up = Normalize(Cross(right, forward));
            constexpr float Pi = 3.14159265358979323846f;
            const float halfHeight = std::tan(camera.VerticalFieldOfViewDegrees * Pi / 360.0f);
            StoreFloat3(constants.CameraPosition, camera.Position);
            StoreFloat3(constants.CameraForward, forward);
            StoreFloat3(constants.CameraRight, right);
            StoreFloat3(constants.CameraUp, up);
            constants.Width = settings.Width;
            constants.Height = settings.Height;
            constants.SamplesPerPixel = settings.SamplesPerPixel;
            constants.MaxReflectionBounces = settings.MaxReflectionBounces;
            constants.HalfHeight = halfHeight;
            constants.HalfWidth = halfHeight * static_cast<float>(settings.Width) / settings.Height;
            constants.MaterialCount = static_cast<uint32_t>(materials.size());
            constants.LightCount = static_cast<uint32_t>(lights.size());
            StoreFloat3(constants.BackgroundTop, scene.BackgroundTop);
            constants.AmbientIntensity = scene.AmbientIntensity;
            StoreFloat3(constants.BackgroundBottom, scene.BackgroundBottom);
            constants.ShadowStrength = settings.ShadowsEnabled
                ? std::clamp(settings.ShadowStrength, 0.0f, 1.0f)
                : 0.0f;
            constants.GammaCorrection = settings.GammaCorrection ? 1u : 0u;
            constants.PrimitiveCount = static_cast<uint32_t>(triangles.size());
            constants.TextureCount = static_cast<uint32_t>(
                textureAtlas.Descriptors.size());
            constants.TextureTexelCount = static_cast<uint32_t>(
                textureAtlas.Texels.size());
            std::array<uint8_t, 256> constantData{};
            std::memcpy(constantData.data(), &constants, sizeof(constants));
            ComPtr<ID3D12Resource> constantBuffer;
            if (!CreateUploadBuffer(
                constantData.data(), constantData.size(), constantBuffer, renderResult.Error))
            {
                return renderResult;
            }

            constexpr uint64_t ShaderRecordSize = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
            std::array<uint8_t, ShaderRecordSize * 3> shaderTableData{};
            for (uint32_t record = 0; record < 3; ++record)
            {
                std::memcpy(
                    shaderTableData.data() + record * ShaderRecordSize,
                    ShaderIdentifiers.data() + record * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
                    D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            }
            ComPtr<ID3D12Resource> shaderTable;
            if (!CreateUploadBuffer(
                shaderTableData.data(), shaderTableData.size(), shaderTable, renderResult.Error))
            {
                return renderResult;
            }

            commandList->CopyBufferRegion(statisticsBuffer.Get(), 0, zeroUpload.Get(), 0, StatisticsSize);
            D3D12_RESOURCE_BARRIER statisticsTransition = TransitionBarrier(
                statisticsBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            commandList->ResourceBarrier(1, &statisticsTransition);
            commandList->SetPipelineState1(StateObject.Get());
            commandList->SetComputeRootSignature(RootSignature.Get());
            commandList->SetComputeRootShaderResourceView(0, topLevelAddress);
            commandList->SetComputeRootUnorderedAccessView(1, outputBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootUnorderedAccessView(2, statisticsBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(3, vertexBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(4, triangleBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(5, materialBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(6, lightBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(
                7,
                textureDescriptorBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(
                8,
                textureTexelBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootConstantBufferView(9, constantBuffer->GetGPUVirtualAddress());

            D3D12_DISPATCH_RAYS_DESC dispatch{};
            const D3D12_GPU_VIRTUAL_ADDRESS shaderTableAddress = shaderTable->GetGPUVirtualAddress();
            dispatch.RayGenerationShaderRecord.StartAddress = shaderTableAddress;
            dispatch.RayGenerationShaderRecord.SizeInBytes = ShaderRecordSize;
            dispatch.MissShaderTable.StartAddress = shaderTableAddress + ShaderRecordSize;
            dispatch.MissShaderTable.SizeInBytes = ShaderRecordSize;
            dispatch.MissShaderTable.StrideInBytes = ShaderRecordSize;
            dispatch.HitGroupTable.StartAddress = shaderTableAddress + ShaderRecordSize * 2;
            dispatch.HitGroupTable.SizeInBytes = ShaderRecordSize;
            dispatch.HitGroupTable.StrideInBytes = ShaderRecordSize;
            dispatch.Width = settings.Width;
            dispatch.Height = settings.Height;
            dispatch.Depth = 1;
            const auto renderBegin = std::chrono::steady_clock::now();
            commandList->DispatchRays(&dispatch);

            D3D12_RESOURCE_BARRIER dispatchBarriers[] = {
                UavBarrier(outputBuffer.Get()),
                UavBarrier(statisticsBuffer.Get()),
                TransitionBarrier(
                    outputBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE),
                TransitionBarrier(
                    statisticsBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE)};
            commandList->ResourceBarrier(static_cast<UINT>(std::size(dispatchBarriers)), dispatchBarriers);
            commandList->CopyResource(outputReadback.Get(), outputBuffer.Get());
            commandList->CopyResource(statisticsReadback.Get(), statisticsBuffer.Get());
            result = commandList->Close();
            if (FAILED(result) || !ExecuteAndWait(commandList.Get(), renderResult.Error))
            {
                if (renderResult.Error.empty())
                {
                    renderResult.Error = HResultMessage("ID3D12GraphicsCommandList::Close", result);
                }
                return renderResult;
            }
            const auto renderEnd = std::chrono::steady_clock::now();

            renderResult.Output = Image(settings.Width, settings.Height);
            void* mappedOutput = nullptr;
            const D3D12_RANGE outputReadRange{0, static_cast<SIZE_T>(outputSize)};
            result = outputReadback->Map(0, &outputReadRange, &mappedOutput);
            if (FAILED(result))
            {
                renderResult.Output = Image();
                renderResult.Error = HResultMessage("Mapping the DXR output readback", result);
                return renderResult;
            }
            std::memcpy(
                renderResult.Output.GetPixelsRGBA8().data(),
                mappedOutput,
                static_cast<size_t>(outputSize));
            outputReadback->Unmap(0, nullptr);

            void* mappedStatisticsData = nullptr;
            const D3D12_RANGE statisticsReadRange{0, StatisticsSize};
            result = statisticsReadback->Map(0, &statisticsReadRange, &mappedStatisticsData);
            if (FAILED(result))
            {
                renderResult.Output = Image();
                renderResult.Error = HResultMessage("Mapping the DXR statistics readback", result);
                return renderResult;
            }
            const auto* mappedStatistics = static_cast<const uint32_t*>(mappedStatisticsData);
            renderResult.Statistics.PrimaryRays = pixelCount * settings.SamplesPerPixel;
            renderResult.Statistics.ShadowRays = mappedStatistics[1];
            renderResult.Statistics.ReflectionRays = mappedStatistics[2];
            renderResult.Statistics.CompletedPixels = mappedStatistics[3];
            renderResult.Statistics.PrimitiveTests = 0;
            statisticsReadback->Unmap(0, nullptr);
            renderResult.Statistics.BuildMilliseconds =
                std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count();
            renderResult.Statistics.RenderMilliseconds =
                std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count();
            renderResult.Status = cancellation() ? RenderStatus::Cancelled : RenderStatus::Completed;
            return renderResult;
        }

        DynamicModule DxgiModule;
        DynamicModule D3D12Module;
        DynamicModule DxcModule;
        EventHandle FenceEvent;
        CreateDeviceFunction CreateDevice = nullptr;
        SerializeRootSignatureFunction SerializeRootSignature = nullptr;
        ComPtr<IDXGIFactory4> Factory;
        ComPtr<IDXGIAdapter1> Adapter;
        ComPtr<ID3D12Device5> Device;
        ComPtr<ID3D12CommandQueue> CommandQueue;
        ComPtr<ID3D12Fence> Fence;
        ComPtr<ID3D12RootSignature> RootSignature;
        ComPtr<ID3D12StateObject> StateObject;
        ComPtr<ID3D12StateObjectProperties> StateObjectProperties;
        std::array<uint8_t, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * 3> ShaderIdentifiers{};
        D3D12_RAYTRACING_TIER RayTracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
        mutable uint64_t FenceValue = 0;
        std::string AdapterName;
        std::string ShaderCompilerDiagnostics;
#endif
        bool Available = false;
        std::string AvailabilityError;
        mutable uint64_t DispatchRaysSequence = 0;
        mutable Detail::PersistentRenderQueue RHIQueue;
    };

    DxrRayTracingRHI::DxrRayTracingRHI()
        : m_Impl(std::make_unique<Impl>())
    {
    }

    DxrRayTracingRHI::~DxrRayTracingRHI() = default;

    BackendInfo DxrRayTracingRHI::GetBackendInfo() const
    {
        BackendInfo info;
        info.Type = BackendType::DXR;
        info.Name = "DXR";
        info.IsNative = m_Impl->Available;
        info.IsHardwareAccelerated = m_Impl->Available;
        info.IsAvailable = m_Impl->Available;
        info.Description = m_Impl->Available
#if defined(_WIN32)
            ? std::string("Native D3D12 DXR backend on ") + m_Impl->AdapterName
#else
            ? "Native D3D12 DXR backend"
#endif
            : std::string("Unavailable: ") + m_Impl->AvailabilityError;
        return info;
    }

    RenderResult DxrRayTracingRHI::Render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const CancellationToken& cancellation) const
    {
        return m_Impl->Submit(&scene, nullptr, camera, settings, cancellation).GetResult();
    }

    RenderJob DxrRayTracingRHI::RenderAsync(
        Scene scene,
        Camera camera,
        RenderSettings settings) const
    {
        return RenderAsync(std::make_shared<Scene>(std::move(scene)), camera, settings);
    }

    RenderJob DxrRayTracingRHI::RenderAsync(
        std::shared_ptr<const Scene> scene,
        Camera camera,
        RenderSettings settings) const
    {
        const Scene* pointer = scene.get();
        return m_Impl->Submit(pointer, std::move(scene), camera, settings);
    }

    void DxrRayTracingRHI::WaitIdle() const
    {
        m_Impl->RHIQueue.WaitIdle();
    }
}
}
