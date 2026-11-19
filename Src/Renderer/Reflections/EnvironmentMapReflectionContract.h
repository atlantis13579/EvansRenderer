#pragma once

#include "Maths/Matrix4.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // Latitude-longitude environment mapping stores incident radiance in a 2:1
    // image. Upload an HDR panorama or capture the scene into six faces, convert to
    // longitude/colatitude, then generate ordinary image mips before lookup.
    // The world convention is Y up: u = atan2(z,x)/(2*pi)+1/2, v = acos(y)/pi.
    // Longitude wraps at -X; latitude clamps at the poles. At a pole longitude
    // is arbitrary. Conversion reverses world Z only at the native cube lookup.
    // PBRT, 3rd ed., "Infinite Area Lights" describes this parameterization:
    // https://www.pbr-book.org/3ed-2018/Light_Sources/Infinite_Area_Lights
    // This is a specular probe estimate: image mips are not solid-angle-weighted
    // GGX convolution, and the captured view retains local parallax error.
    // The capture producer is shared with cubemaps; conversion and lookup are
    // independent technique shaders. SSR, cube and lat-long occupy one GUI slot.
    struct EnvironmentMapLayout
    {
        std::uint32_t Width = 0, Height = 0, MipCount = 0;
    };
    inline EnvironmentMapLayout BuildEnvironmentMapLayout(std::uint32_t faceResolution)
    {
        constexpr std::uint32_t MinimumFaceResolution = 16, MaximumFaceResolution = 1024;
        constexpr std::uint32_t LongitudeFaceSpans = 4, LatitudeFaceSpans = 2;
        if (faceResolution < MinimumFaceResolution || faceResolution > MaximumFaceResolution ||
            (faceResolution & (faceResolution - 1))) return {};
        EnvironmentMapLayout layout{faceResolution * LongitudeFaceSpans, faceResolution * LatitudeFaceSpans, 0};
        for (auto extent = layout.Width; extent; extent >>= 1) ++layout.MipCount;
        return layout;
    }

    struct EnvironmentMapReflectionConstants
    {
        Matrix4 InverseViewProjection;
        Float4 Viewport; // destination extent XY, reciprocal extent ZW
        Float4 Sampling; // reflection intensity, last lat-long mip, reserved ZW
        ShaderUInt4 Options; // lookup/composite phase, shading model, reserved Z, planar receiver mask
        Float4 ReceiverPlane;
        Float4 ReceiverMinimum; // xyz bounds, w plane tolerance
        Float4 ReceiverMaximum; // xyz bounds, w minimum normal cosine
    };
    static_assert(sizeof(EnvironmentMapReflectionConstants) == 160, "Environment map uniform layout changed");

    inline bool BuildEnvironmentMapReflectionConstants(const ReflectionDesc& desc,
        const Techniques::ReflectionSettings& settings, Techniques::ShadingModel shading,
        int width, int height, std::uint32_t mipCount, EnvironmentMapReflectionConstants& output)
    {
        if (width <= 0 || height <= 0 || !mipCount) return false;
        for (float value : desc.ViewProjection.Values) if (!std::isfinite(value)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        EnvironmentMapReflectionConstants constants{};
        std::copy(inverse.Data(), inverse.Data() + 16, constants.InverseViewProjection.Values.begin());
        for (float value : constants.InverseViewProjection.Values) if (!std::isfinite(value)) return false;
        constants.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
        const auto sanitized=Techniques::SanitizeReflectionSettings(settings);
        constants.Sampling = {sanitized.EnvironmentIntensity, float(mipCount - 1), 0, 0};
        constants.Options = {0, static_cast<std::uint32_t>(shading), 0, 0};
        const auto& receiver=sanitized.Planar;
        constants.Options.W=receiver.Enabled ? (receiver.BoundedReceiver ? 2u : 1u) : 0u;
        constants.ReceiverPlane={receiver.Plane[0],receiver.Plane[1],receiver.Plane[2],receiver.Plane[3]};
        constants.ReceiverMinimum={receiver.ReceiverMinimum[0],receiver.ReceiverMinimum[1],receiver.ReceiverMinimum[2],receiver.ReceiverTolerance};
        constants.ReceiverMaximum={receiver.ReceiverMaximum[0],receiver.ReceiverMaximum[1],receiver.ReceiverMaximum[2],receiver.MinimumNormalCosine};
        output = constants;
        return true;
    }
}
