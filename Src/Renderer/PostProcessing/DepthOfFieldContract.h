#pragma once

#include "Maths/Matrix4.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // Thin-lens signed CoC radius: width/(2*sensorWidth) * f^2/(N*(focus-f))
    // * (1-focus/eyeZ). Negative radii describe foreground. Perspective inverse
    // eye Z is affine in hardware depth, so shaders need only a scale and bias.
    // Recover the lens origin from inverseVP*(0,0,1,0), not the last draw's camera
    // or a lighting position that may be stale after a custom projection change.
    // Sensor width and the actual frustum determine focal length. World scale
    // converts the sensor to scene units; focus distance is axial, not radial.
    // Orthographic cameras retain their pinhole image. Singular/non-finite
    // cameras fail explicitly. Oblique depth clipping is rejected: its inverse
    // eye Z depends on screen position and needs a larger CoC contract.
    // Clear depth uses the infinity limit even with a
    // finite far plane. Focus is constrained to at least 1% beyond focal length.
    // See Demers, "Depth of Field: A Survey of Techniques", GPU Gems (2004), and
    // Hammon, "Practical Post-Process Depth of Field", GPU Gems 3 (2007).
    // https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field
    // Opaque color -> disk gather -> RGB composite -> sharp transparency -> HDR
    // resolve. Neighbor-based background fill cannot recover hidden surfaces.
    struct DepthOfFieldConstants
    {
        Float4 ViewportRadius; // inverse width/height, maximum pixel radius, reserved
        Float4 CoCScaleBias{0, 0, 0, 0}; // depth slope/intercept, infinity radius, perspective enabled
        ShaderUInt4 Options; // gather/composite identifier, reserved
    };
    static_assert(sizeof(DepthOfFieldConstants) == 48, "Depth of field uses three aligned uniform vectors");
    static_assert(offsetof(DepthOfFieldConstants, Options) == 32, "Depth of field stage offset changed");

    inline bool BuildDepthOfFieldConstants(const DepthOfFieldDesc& desc,
        const Techniques::DepthOfFieldSettings& requested, int width, int height,
        Techniques::DepthOfFieldPassKind pass, DepthOfFieldConstants& result)
    {
        if (width <= 0 || height <= 0) return false;
        const auto settings = Techniques::SanitizeDepthOfFieldSettings(requested);
        result = {};
        result.ViewportRadius = {1.0f / width, 1.0f / height, settings.MaxRadiusPixels, 0.0f};
        result.Options.X = static_cast<std::uint32_t>(pass);
        for (float value : desc.ViewProjection.Values) if (!std::isfinite(value)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        for (std::size_t i = 0; i < 16; ++i) if (!std::isfinite(inverse.Data()[i])) return false;
        // An affine input has constant clip W. Gaussian elimination can leave
        // tiny perspective residuals in its inverse after a rotated view.
        if (desc.ViewProjection.Values[12] == 0.0f && desc.ViewProjection.Values[13] == 0.0f &&
            desc.ViewProjection.Values[14] == 0.0f) return true;
        constexpr float DepthProjectionTolerance = 1.0e-6f;
        const float depthScale = std::max(std::abs(inverse(3, 2)), std::abs(inverse(3, 3)));
        if (std::abs(inverse(3, 0)) > depthScale * DepthProjectionTolerance ||
            std::abs(inverse(3, 1)) > depthScale * DepthProjectionTolerance) return false;
        if (inverse(3, 2) == 0.0f) return false;

        using Vector = std::array<double, 3>;
        const auto subtract = [](const Vector& a, const Vector& b)
            { return Vector{a[0] - b[0], a[1] - b[1], a[2] - b[2]}; };
        const auto dot = [](const Vector& a, const Vector& b)
            { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
        const auto normalize = [&](Vector& a)
        {
            const double scale = std::max({std::abs(a[0]), std::abs(a[1]), std::abs(a[2])});
            if (!(scale > 0.0) || !std::isfinite(scale)) return false;
            for (double& value : a) value /= scale;
            const double length = std::sqrt(dot(a, a));
            for (double& value : a) value /= length;
            return true;
        };
        const auto unproject = [&](double x, double y, double z, Vector& point)
        {
            const double w = inverse(3, 0) * x + inverse(3, 1) * y + inverse(3, 2) * z + inverse(3, 3);
            if (w == 0.0 || !std::isfinite(w)) return false;
            for (std::size_t row = 0; row < 3; ++row)
            {
                point[row] = (inverse(row, 0) * x + inverse(row, 1) * y + inverse(row, 2) * z + inverse(row, 3)) / w;
                if (!std::isfinite(point[row])) return false;
            }
            return true;
        };
        Vector camera{}, nearPoint{}, rightPoint{}, upPoint{}, interiorPoint{};
        for (std::size_t row = 0; row < 3; ++row)
        {
            camera[row] = static_cast<double>(inverse(row, 2)) / inverse(3, 2);
            if (!std::isfinite(camera[row])) return false;
        }
        constexpr double InteriorClipDepth = 0.5;
        if (!unproject(0, 0, 0, nearPoint) || !unproject(1, 0, 0, rightPoint) ||
            !unproject(0, 1, 0, upPoint) || !unproject(0, 0, InteriorClipDepth, interiorPoint)) return false;
        auto right = subtract(rightPoint, nearPoint);
        auto up = subtract(upPoint, nearPoint);
        const double nearWidth = std::sqrt(dot(right, right)) * 2.0;
        if (!normalize(right) || !normalize(up)) return false;
        Vector forward{right[1] * up[2] - right[2] * up[1],
            right[2] * up[0] - right[0] * up[2], right[0] * up[1] - right[1] * up[0]};
        if (!normalize(forward)) return false;
        if (dot(forward, subtract(interiorPoint, nearPoint)) < 0)
            for (double& value : forward) value = -value;
        const double nearZ = dot(subtract(nearPoint, camera), forward);
        const double interiorZ = dot(subtract(interiorPoint, camera), forward);
        if (!(nearZ > 0 && interiorZ > nearZ && nearWidth > 0)) return false;
        constexpr double MillimetersPerMeter = 1000.0;
        constexpr double MinimumFocusSeparationRatio = 0.01;
        const double sensorWidth = settings.SensorWidthMillimeters / (MillimetersPerMeter * settings.MetersPerWorldUnit);
        const double focalLength = sensorWidth * nearZ / nearWidth;
        const double focus = std::max(static_cast<double>(settings.FocusDistance), focalLength * (1.0 + MinimumFocusSeparationRatio));
        const double radiusAtInfinity = width * focalLength * focalLength /
            (2.0 * sensorWidth * settings.FNumber * (focus - focalLength));
        const double inverseDepthSlope = (1.0 / interiorZ - 1.0 / nearZ) / InteriorClipDepth;
        result.CoCScaleBias = {static_cast<float>(-radiusAtInfinity * focus * inverseDepthSlope),
            static_cast<float>(radiusAtInfinity * (1.0 - focus / nearZ)),
            static_cast<float>(radiusAtInfinity), settings.Enabled ? 1.0f : 0.0f};
        return std::isfinite(result.CoCScaleBias.X) && std::isfinite(result.CoCScaleBias.Y) &&
            std::isfinite(result.CoCScaleBias.Z);
    }
}
