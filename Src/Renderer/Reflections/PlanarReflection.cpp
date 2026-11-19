#include "Renderer/Reflections/PlanarReflection.h"
#include "Maths/Matrix4.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    // Planar reflection dependencies and native execution:
    // - One world-space plane, the main camera, and existing opaque/cutout meshes.
    // - Reflect the camera with H = I - 2*n*n^T and translation -2*d*n.
    //   Multiplying main VP by H changes winding; XOR capture winding with the
    //   existing mirrored-world-transform test instead of disabling back culling.
    // - Render a private HDR/depth view, clip away the mirror and its far side,
    //   then area-filter ordinary mips, retaining odd edge pixels through 1x1.
    //   No authored environment texture is needed.
    // - Reuse direct/ambient/emissive material shading without recursive effects,
    //   shadows, transparency or screen effects. Those extensions remain separate.
    // - A receiver must lie near the plane and have an aligned normal. Reconstruct
    //   its position from main depth, project into the mirror view, then add
    //   Fresnel-weighted radiance after AO and before blur/sharp transparency.
    // - Forward needs material capture; Deferred reuses its G-buffer. MSAA uses
    //   resolved opaque inputs and the existing processed-color restore bridge.
    // - Planar, SSR, cube and latitude-longitude are alternative primary methods.
    // DX11/DX12/OpenGL ES/Vulkan supply capture/lookup and GUI activation; Metal is pending.
    //
    // Eric Lengyel, "Oblique View Frustum Depth Projection and Clipping" (2005):
    // https://terathon.com/lengyel/Lengyel-Oblique.pdf
    // Here clip Z is [0,1], so row 2 is the scaled plane, without subtracting row 3
    // or multiplying by OpenGL's depth-range factor of two. Inverting the full VP
    // also supports asymmetric, orthographic and infinite-far projections. Keep
    // far corners homogeneous: infinite projections have corner W=0.
    namespace
    {
        constexpr double MinimumEyePlaneDistance = 1e-5;
        constexpr double MinimumCornerDistance = 1e-8;
        constexpr double MinimumProjectiveW = 1e-7;
        bool Finite(Float3 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z); }
        bool Finite(const Matrix4& matrix)
        {
            return std::all_of(matrix.Values.begin(), matrix.Values.end(), [](float v) { return std::isfinite(v); });
        }
        std::array<double, 4> Transform(const Matrix4& matrix, const std::array<double, 4>& vector)
        {
            std::array<double, 4> result {};
            for (std::size_t row = 0; row < 4; ++row)
                for (std::size_t column = 0; column < 4; ++column)
                    result[row] += matrix.Values[row * 4 + column] * vector[column];
            return result;
        }
        Matrix4 Multiply(const Matrix4& a, const Matrix4& b)
        {
            Matrix4 result;
            for (std::size_t row = 0; row < 4; ++row)
                for (std::size_t column = 0; column < 4; ++column)
                {
                    double value = 0;
                    for (std::size_t inner = 0; inner < 4; ++inner)
                        value += double(a.Values[row * 4 + inner]) * b.Values[inner * 4 + column];
                    result.Values[row * 4 + column] = static_cast<float>(value);
                }
            return result;
        }
        bool Invert(const Matrix4& source, Matrix4& output)
        {
            Maths::Matrix4 inverse;
            if (!Maths::Matrix4(source.Values).Inverse(inverse)) return false;
            std::copy(inverse.Data(), inverse.Data() + output.Values.size(), output.Values.begin());
            return Finite(output);
        }
    }

    bool BuildPlanarCaptureFrame(const Matrix4& mainViewProjection, Float3 cameraPosition,
        int width, int height, const Techniques::PlanarReflectionSettings& settings,
        PlanarCaptureFrame& output, std::string& error)
    {
        const auto fail = [&](const char* message) { error = message; return false; };
        if (width <= 0 || height <= 0 || !Finite(cameraPosition) || !Finite(mainViewProjection))
            return fail("Planar capture requires finite camera data and positive view dimensions");
        const auto& p = settings.Plane;
        const double length = std::hypot(double(p[0]), double(p[1]), double(p[2]));
        if (!std::isfinite(length) || length <= 0 || !std::isfinite(p[3]) ||
            !std::isfinite(settings.ClipBias) || settings.ClipBias < 0)
            return fail("Planar capture requires a finite nonzero plane and nonnegative clipping bias");
        const auto sanitized = Techniques::SanitizePlanarReflectionSettings(settings);
        std::array<double, 4> plane;
        for (std::size_t index = 0; index < plane.size(); ++index)
        {
            plane[index] = p[index] / length;
            if (!std::isfinite(static_cast<float>(plane[index]))) return fail("Normalized plane exceeds finite float storage");
        }
        PlanarCaptureFrame candidate;
        candidate.MainViewProjection = mainViewProjection;
        if (!Invert(mainViewProjection, candidate.InverseMainViewProjection))
            return fail("Planar capture requires an invertible main camera");
        const auto divisor = sanitized.ResolutionDivisor;
        candidate.MainWidth = static_cast<std::uint32_t>(width);
        candidate.MainHeight = static_cast<std::uint32_t>(height);
        candidate.Width = (static_cast<std::uint32_t>(width) + divisor - 1) / divisor;
        candidate.Height = (static_cast<std::uint32_t>(height) + divisor - 1) / divisor;
        for (auto extent = std::max(candidate.Width, candidate.Height); extent; extent >>= 1) ++candidate.MipCount;
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
                candidate.ReflectionTransform.Values[row * 4 + column] = float((row == column ? 1.0 : 0.0) - 2 * plane[row] * plane[column]);
            candidate.ReflectionTransform.Values[row * 4 + 3] = float(-2 * plane[3] * plane[row]);
        }
        const auto reflectedEye = Transform(candidate.ReflectionTransform, {cameraPosition.X, cameraPosition.Y, cameraPosition.Z, 1});
        candidate.Position = {float(reflectedEye[0]), float(reflectedEye[1]), float(reflectedEye[2])};
        candidate.ViewProjection = Multiply(mainViewProjection, candidate.ReflectionTransform);
        if (!Finite(candidate.ReflectionTransform) || !Finite(candidate.Position) || !Finite(candidate.ViewProjection))
            return fail("Mirrored camera exceeds finite float storage");
        candidate.ClippedViewProjection = candidate.ViewProjection;
        const double eyeDistance = plane[0] * cameraPosition.X + plane[1] * cameraPosition.Y + plane[2] * cameraPosition.Z + plane[3];
        if (eyeDistance < 0) for (auto& value : plane) value = -value;
        candidate.ReceiverPlane = {float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3])};
        plane[3] -= sanitized.ClipBias;
        candidate.ClipPlane = {float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3])};
        if (std::abs(eyeDistance) > MinimumEyePlaneDistance)
        {
            // H is its own inverse. This avoids inverting an already rounded VP*H.
            const Matrix4 inverseMirror = Multiply(candidate.ReflectionTransform, candidate.InverseMainViewProjection);
            if (!Finite(inverseMirror)) return fail("Inverse mirror projection exceeds finite float storage");
            double farthest = 0;
            for (double x : {-1.0, 1.0}) for (double y : {-1.0, 1.0})
            {
                const auto corner = Transform(inverseMirror, {x, y, 1, 1});
                double distance = 0;
                for (std::size_t index = 0; index < plane.size(); ++index) distance += plane[index] * corner[index];
                farthest = std::max(farthest, distance);
            }
            candidate.Status = PlanarCaptureStatus::NoVisibleHalfSpace;
            if (farthest > MinimumCornerDistance)
            {
                for (std::size_t column = 0; column < 4; ++column)
                    candidate.ClippedViewProjection.Values[8 + column] = float(plane[column] / farthest);
                if (!Finite(candidate.ClippedViewProjection)) return fail("Oblique capture projection exceeds finite float storage");
                candidate.Status = PlanarCaptureStatus::Ready;
            }
        }
        output = candidate;
        error.clear();
        return true;
    }

    bool ProjectPlanarReceiver(Float3 worldPosition, const PlanarCaptureFrame& frame, Float2& topLeftUV) noexcept
    {
        if (!frame.Active() || !Finite(worldPosition)) return false;
        const auto clip = Transform(frame.ViewProjection, {worldPosition.X, worldPosition.Y, worldPosition.Z, 1});
        if (!std::all_of(clip.begin(), clip.end(), [](double v) { return std::isfinite(v); }) || clip[3] <= MinimumProjectiveW) return false;
        const double u = (clip[0] / clip[3] + 1) / 2, v = (1 - clip[1] / clip[3]) / 2;
        if (u < 0 || u > 1 || v < 0 || v > 1) return false;
        // Receiver points may be behind the biased capture near plane. Their XY
        // lookup remains valid; applying the capture depth test here erases mirrors.
        topLeftUV = {float(u), float(v)};
        return true;
    }
}
