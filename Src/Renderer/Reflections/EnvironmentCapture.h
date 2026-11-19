#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    struct EnvironmentRadiance;

    inline constexpr std::size_t EnvironmentFaceCount = 6;
    enum class EnvironmentFace : std::uint32_t { PositiveX, NegativeX, PositiveY, NegativeY, PositiveZ, NegativeZ };

    // Native cube axes use +Z forward. Keep capture cameras right-handed by
    // reflecting world Z at the lookup boundary. These names describe native
    // texture faces: PositiveZ captures world -Z, not world +Z.
    struct EnvironmentCaptureFace
    {
        Float3 Forward;
        Float3 Up;
        Matrix4 View;
        Matrix4 ViewProjection;
    };
    struct EnvironmentCaptureFrame
    {
        std::shared_ptr<const EnvironmentRadiance> Source;
        Float3 Position;
        float NearPlane = 0;
        float FarPlane = 0;
        std::uint32_t Resolution = 0;
        std::uint32_t MipCount = 0;
        std::array<EnvironmentCaptureFace, EnvironmentFaceCount> Faces {};
    };
    struct EnvironmentCaptureBounds
    {
        Float3 Minimum;
        Float3 Maximum;
    };
    struct EnvironmentLookup
    {
        EnvironmentFace Face = EnvironmentFace::PositiveX;
        Float2 UV; // Native cube face coordinates, not a raster API's viewport origin.
    };

    [[nodiscard]] bool BuildEnvironmentCaptureFrame(Float3 position, const EnvironmentCaptureBounds& bounds,
        std::uint32_t resolution, EnvironmentCaptureFrame& output, std::string& error);
    [[nodiscard]] Float3 WorldToEnvironmentDirection(Float3 worldDirection) noexcept;
    [[nodiscard]] bool LocateEnvironmentDirection(Float3 worldDirection, EnvironmentLookup& output) noexcept;
    [[nodiscard]] bool EnvironmentFaceDirection(EnvironmentFace face, Float2 uv, Float3& output) noexcept;

    // Existing geometry shaders use canonical clip Y up and [0,1] depth. ES
    // rasterizes a cube face with inverted clip Y because cube T must agree with
    // native storage rows. Its capture raster state must reverse front winding.
    // Vulkan's ordinary geometry shader already adapts viewport Y; do not flip it twice.
    [[nodiscard]] Matrix4 EnvironmentRasterProjection(const EnvironmentCaptureFace& face, RHIBackend backend) noexcept;
    [[nodiscard]] bool EnvironmentReversesWinding(RHIBackend backend) noexcept;
}
