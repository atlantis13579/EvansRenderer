#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    enum class PlanarCaptureStatus { Ready, CameraOnPlane, NoVisibleHalfSpace };

    // One mirror plane and one frozen main camera. Matrices are canonical Y-up,
    // forward [0,1] depth; native raster adaptation happens in each backend.
    struct PlanarCaptureFrame
    {
        Matrix4 MainViewProjection;
        Matrix4 InverseMainViewProjection;
        Matrix4 ReflectionTransform;
        Matrix4 ViewProjection; // Unclipped projective lookup, including receiver points.
        Matrix4 ClippedViewProjection; // Oblique near plane for capture geometry only.
        Float4 ReceiverPlane; // Unit normal, oriented toward the original camera.
        Float4 ClipPlane; // ReceiverPlane offset by the capture bias.
        Float3 Position; // Reflected lighting/view position.
        std::uint32_t MainWidth = 0, MainHeight = 0;
        std::uint32_t Width = 0, Height = 0, MipCount = 0;
        PlanarCaptureStatus Status = PlanarCaptureStatus::CameraOnPlane;
        bool Active() const noexcept { return Status == PlanarCaptureStatus::Ready; }
    };

    // Invalid input leaves output untouched. A valid but degenerate view returns
    // an inactive frame: its native capture should clear black and skip geometry.
    [[nodiscard]] bool BuildPlanarCaptureFrame(const Matrix4& mainViewProjection, Float3 cameraPosition,
        int width, int height, const Techniques::PlanarReflectionSettings& settings,
        PlanarCaptureFrame& output, std::string& error);
    [[nodiscard]] bool ProjectPlanarReceiver(Float3 worldPosition, const PlanarCaptureFrame& frame, Float2& topLeftUV) noexcept;
}
