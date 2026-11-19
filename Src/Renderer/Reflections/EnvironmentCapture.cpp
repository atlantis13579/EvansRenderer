#include "Renderer/Reflections/EnvironmentCapture.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    // Greene, "Environment Mapping and Other Applications of World Projections"
    // (IEEE Computer Graphics and Applications, 1986): six perspective views
    // represent directional radiance around one probe. Local objects retain
    // parallax error when a different receiver samples the same probe.
    // Native face selection/orientation:
    // https://docs.vulkan.org/spec/latest/chapters/textures.html#textures-cubemapfaceselection
    // https://github.khronos.org/KTX-Specification/ktxspec.v2.html#_facecount
    //
    // Producer contract (all five native APIs; Metal awaits Apple validation):
    // - Reuse existing opaque/cutout meshes, textures, ambient/direct lights and emission.
    // - Six single-sample linear RGBA16F faces share a private depth attachment.
    // - Complete all faces before generating the full mip chain or sampling it.
    // - Capture excludes shadows, recursive reflections, transparency, AO and post effects.
    // - Cubemap and latitude/longitude environment lookup are alternative primary
    //   reflection methods alongside SSR; their composition follows opaque AO.
    // - A material prepass serves Forward; Deferred reuses its material targets.
    //   MSAA resolves receiver inputs, then restores reflected color for transparency.
    // - Frame settings freeze the probe; cancelled work cannot publish a partial cube.
    // Mip averaging is a basic roughness approximation, not GGX convolution.
    // Authored linear HDR faces bypass geometry capture. All five native APIs
    // upload these faces before filtering or converting to latitude-longitude.
    // Metal compilation and GPU execution still require validation on Apple hardware.
    namespace
    {
        constexpr std::uint32_t MinimumFaceResolution = 16;
        constexpr std::uint32_t MaximumFaceResolution = 1024;
        constexpr double MinimumCaptureDistance = 1e-3;
        constexpr double MinimumNearPlane = 1e-6;
        constexpr double NearToFarRatio = 1e-4;
        constexpr double FarPaddingFraction = 0.01;

        struct Vector
        {
            double X, Y, Z;
        };
        Vector Expand(Float3 value) { return {value.X, value.Y, value.Z}; }
        Vector Subtract(Vector a, Vector b) { return {a.X - b.X, a.Y - b.Y, a.Z - b.Z}; }
        double Dot(Vector a, Vector b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
        Vector Cross(Vector a, Vector b) { return {a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X}; }
        bool Finite(Float3 value) { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z); }
        bool Finite(const Matrix4& value)
        {
            return std::all_of(value.Values.begin(), value.Values.end(), [](float v) { return std::isfinite(v); });
        }
        Matrix4 CameraView(Float3 position, Float3 forward, Float3 up)
        {
            const auto f = Expand(forward), u = Expand(up), eye = Expand(position);
            const auto right = Cross(f, u);
            Matrix4 view;
            view.Values = {float(right.X), float(right.Y), float(right.Z), float(-Dot(right, eye)),
                up.X, up.Y, up.Z, float(-Dot(u, eye)),
                -forward.X, -forward.Y, -forward.Z, float(Dot(f, eye)),
                0, 0, 0, 1};
            return view;
        }
        Matrix4 ProjectView(const Matrix4& view, double nearPlane, double farPlane)
        {
            // A square 90-degree perspective has unit focal scales. Compute Z
            // in double before packing; W is positive camera-forward distance.
            Matrix4 result = view;
            const double scale = farPlane / (nearPlane - farPlane);
            const double bias = farPlane * nearPlane / (nearPlane - farPlane);
            for (std::size_t column = 0; column < 4; ++column)
            {
                result.Values[8 + column] = float(scale * view.Values[8 + column] + (column == 3 ? bias : 0));
                result.Values[12 + column] = -view.Values[8 + column];
            }
            return result;
        }
    }

    bool BuildEnvironmentCaptureFrame(Float3 position, const EnvironmentCaptureBounds& bounds,
        std::uint32_t resolution, EnvironmentCaptureFrame& output, std::string& error)
    {
        const auto fail = [&](const char* message) { error = message; return false; };
        if (!Finite(position) || !Finite(bounds.Minimum) || !Finite(bounds.Maximum))
            return fail("Environment capture position and bounds must be finite");
        if (bounds.Minimum.X > bounds.Maximum.X || bounds.Minimum.Y > bounds.Maximum.Y || bounds.Minimum.Z > bounds.Maximum.Z)
            return fail("Environment capture bounds are inverted");
        if (resolution < MinimumFaceResolution || resolution > MaximumFaceResolution || (resolution & (resolution - 1)) != 0)
            return fail("Environment face resolution must be a power of two between 16 and 1024");

        double farthestSquared = 0;
        for (std::uint32_t corner = 0; corner < 8; ++corner)
        {
            const Vector point {corner & 1 ? bounds.Maximum.X : bounds.Minimum.X,
                corner & 2 ? bounds.Maximum.Y : bounds.Minimum.Y, corner & 4 ? bounds.Maximum.Z : bounds.Minimum.Z};
            const auto displacement = Subtract(point, Expand(position));
            farthestSquared = std::max(farthestSquared, Dot(displacement, displacement));
        }
        const double farPlane = std::max(MinimumCaptureDistance, std::sqrt(farthestSquared) * (1 + FarPaddingFraction));
        const double nearPlane = std::max(MinimumNearPlane, farPlane * NearToFarRatio);
        EnvironmentCaptureFrame candidate;
        candidate.Position = position;
        candidate.NearPlane = float(nearPlane); candidate.FarPlane = float(farPlane);
        candidate.Resolution = resolution;
        for (auto extent = resolution; extent != 0; extent >>= 1) ++candidate.MipCount;
        if (!std::isfinite(candidate.FarPlane) || candidate.NearPlane >= candidate.FarPlane)
            return fail("Environment capture depth range exceeds finite float storage");
        // World-Z reflection aligns cube storage with right-handed scene views.
        constexpr std::array<Float3, EnvironmentFaceCount> forwards {{{1, 0, 0}, {-1, 0, 0},
            {0, 1, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}}};
        constexpr std::array<Float3, EnvironmentFaceCount> ups {{{0, 1, 0}, {0, 1, 0},
            {0, 0, 1}, {0, 0, -1}, {0, 1, 0}, {0, 1, 0}}};
        for (std::size_t index = 0; index < candidate.Faces.size(); ++index)
        {
            auto& face = candidate.Faces[index];
            face.Forward = forwards[index]; face.Up = ups[index];
            face.View = CameraView(position, face.Forward, face.Up);
            face.ViewProjection = ProjectView(face.View, candidate.NearPlane, candidate.FarPlane);
            if (!Finite(face.View) || !Finite(face.ViewProjection)) return fail("Environment capture camera is not representable");
        }
        output = candidate;
        error.clear();
        return true;
    }

    Float3 WorldToEnvironmentDirection(Float3 worldDirection) noexcept
    {
        return {worldDirection.X, worldDirection.Y, -worldDirection.Z};
    }

    bool LocateEnvironmentDirection(Float3 worldDirection, EnvironmentLookup& output) noexcept
    {
        if (!Finite(worldDirection)) return false;
        const auto cube = WorldToEnvironmentDirection(worldDirection);
        const double x = cube.X, y = cube.Y, z = cube.Z;
        const double ax = std::abs(x), ay = std::abs(y), az = std::abs(z);
        double horizontal = 0, vertical = 0, major = 0;
        EnvironmentLookup candidate;
        // X then Y then Z gives deterministic CPU ties; hardware may select an
        // adjacent face at an exact edge. Both endpoints represent the same ray.
        if (ax >= ay && ax >= az)
        {
            candidate.Face = x >= 0 ? EnvironmentFace::PositiveX : EnvironmentFace::NegativeX;
            horizontal = x >= 0 ? -z : z; vertical = -y; major = ax;
        }
        else if (ay >= az)
        {
            candidate.Face = y >= 0 ? EnvironmentFace::PositiveY : EnvironmentFace::NegativeY;
            horizontal = x; vertical = y >= 0 ? z : -z; major = ay;
        }
        else
        {
            candidate.Face = z >= 0 ? EnvironmentFace::PositiveZ : EnvironmentFace::NegativeZ;
            horizontal = z >= 0 ? x : -x; vertical = -y; major = az;
        }
        if (major == 0) return false;
        candidate.UV = {float((horizontal / major + 1) * 0.5), float((vertical / major + 1) * 0.5)};
        output = candidate;
        return true;
    }

    bool EnvironmentFaceDirection(EnvironmentFace face, Float2 uv, Float3& output) noexcept
    {
        if (!std::isfinite(uv.X) || !std::isfinite(uv.Y) || uv.X < 0 || uv.X > 1 || uv.Y < 0 || uv.Y > 1) return false;
        const double s = double(uv.X) * 2 - 1, t = double(uv.Y) * 2 - 1;
        Vector cube {};
        switch (face)
        {
        case EnvironmentFace::PositiveX: cube = {1, -t, -s}; break;
        case EnvironmentFace::NegativeX: cube = {-1, -t, s}; break;
        case EnvironmentFace::PositiveY: cube = {s, 1, t}; break;
        case EnvironmentFace::NegativeY: cube = {s, -1, -t}; break;
        case EnvironmentFace::PositiveZ: cube = {s, -t, 1}; break;
        case EnvironmentFace::NegativeZ: cube = {-s, -t, -1}; break;
        default: return false;
        }
        const double inverseLength = 1 / std::sqrt(Dot(cube, cube));
        output = {float(cube.X * inverseLength), float(cube.Y * inverseLength), float(-cube.Z * inverseLength)};
        return true;
    }

    Matrix4 EnvironmentRasterProjection(const EnvironmentCaptureFace& face, RHIBackend backend) noexcept
    {
        auto projection = face.ViewProjection;
        if (EnvironmentReversesWinding(backend))
            for (std::size_t column = 0; column < 4; ++column) projection.Values[4 + column] *= -1;
        return projection;
    }

    bool EnvironmentReversesWinding(RHIBackend backend) noexcept { return backend == RHIBackend::OpenGLES; }
}
