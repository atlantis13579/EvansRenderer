#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    // Shared light-axis metric for PCSS blocker distances and VSM moments.
    // Orthographic distances start at the near plane; perspective distances
    // start at the projector. Width/height describe the near-plane rectangle.
    struct ShadowProjection
    {
        bool Perspective = false;
        double NearDistance = 0.0, FarDistance = 0.0;
        double Width = 0.0, Height = 0.0;
        double ClipWScale = 1.0; // removes homogeneous scaling from the input matrix
    };

    // Finite, rectilinear, increasing [0,1] depth; rigid views and off-center
    // perspective are supported. Oblique/sheared projections are rejected.
    // Distance metrics must survive conversion to the native float interface.
    [[nodiscard]] bool BuildShadowProjection(const Matrix4& lightViewProjection,
        ShadowProjection& projection, std::string& error);
    [[nodiscard]] bool IsValidShadowProjection(const ShadowProjection& projection);
    [[nodiscard]] double ShadowLinearDepth(const ShadowProjection& projection, double deviceDepth);

    // Normalized linear depth = clip.z * X + clip.w * Y + Z (W is reserved).
    // Evaluate at the vertex and perspective-correctly interpolate the result.
    // Using un-divided clip coordinates avoids reconstructing moments from a
    // quantized perspective depth texture. Receivers use the identical encoding.
    [[nodiscard]] bool BuildShadowDepthEncoding(const ShadowProjection& projection,
        Float4& encoding, std::string& error);
}
