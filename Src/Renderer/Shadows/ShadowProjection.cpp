#include "Renderer/Shadows/ShadowProjection.h"

#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        constexpr double OrthogonalityTolerance = 1.0e-5;
        struct Vector { double X, Y, Z; };
        double Dot(Vector a, Vector b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
        Vector Scale(Vector a, double scale) { return {a.X * scale, a.Y * scale, a.Z * scale}; }
        Vector Subtract(Vector a, Vector b) { return {a.X - b.X, a.Y - b.Y, a.Z - b.Z}; }
        double Length(Vector v) { return std::sqrt(Dot(v, v)); }
        bool Orthogonal(Vector a, Vector b)
        { return std::abs(Dot(a, b)) <= OrthogonalityTolerance * Length(a) * Length(b); }
        bool NativeDistance(double value)
        {
            return std::isfinite(value) && value >= std::numeric_limits<float>::min() &&
                value <= std::numeric_limits<float>::max();
        }
    }

    bool IsValidShadowProjection(const ShadowProjection& projection)
    {
        return NativeDistance(projection.Width) && NativeDistance(projection.Height) &&
            std::isfinite(projection.NearDistance) && projection.NearDistance >= 0.0 &&
            NativeDistance(projection.FarDistance) && projection.FarDistance > projection.NearDistance &&
            float(projection.FarDistance) > float(projection.NearDistance) &&
            std::isfinite(projection.ClipWScale) && projection.ClipWScale > 0.0 &&
            (projection.Perspective ? NativeDistance(projection.NearDistance) : projection.NearDistance == 0.0);
    }

    bool BuildShadowProjection(const Matrix4& matrix, ShadowProjection& projection, std::string& error)
    {
        projection = {}; error.clear();
        const auto fail = [&](const char* reason) { error = reason; return false; };
        for (float value : matrix.Values)
            if (!std::isfinite(value)) return fail("Shadow projection contains nonfinite coefficients");
        const auto row = [&](int index) -> Vector
        { return {matrix.Values[index * 4], matrix.Values[index * 4 + 1], matrix.Values[index * 4 + 2]}; };
        const double perspectiveScale = Length(row(3));
        ShadowProjection candidate;
        Vector right, up, forward;
        if (perspectiveScale == 0.0)
        {
            if (!(matrix.Values[15] > 0.0f)) return fail("Orthographic shadow clip W must be positive");
            candidate.ClipWScale = matrix.Values[15];
            const double scale = 1.0 / candidate.ClipWScale;
            right = Scale(row(0), scale); up = Scale(row(1), scale); forward = Scale(row(2), scale);
            const double depthScale = Length(forward);
            if (!std::isfinite(depthScale) || depthScale <= 0.0)
                return fail("Orthographic shadow depth interval is degenerate");
            candidate.FarDistance = 1.0 / depthScale;
            candidate.Width = 2.0 / Length(right); candidate.Height = 2.0 / Length(up);
        }
        else
        {
            // Remove homogeneous scale: clip W becomes axial distance d, and
            // device depth is A + B/d. Perspective offsets occupy X/Y only.
            candidate.Perspective = true; candidate.ClipWScale = perspectiveScale;
            forward = Scale(row(3), 1.0 / perspectiveScale);
            const auto depthRow = Scale(row(2), 1.0 / perspectiveScale);
            const double a = Dot(depthRow, forward);
            const double b = (matrix.Values[11] - a * matrix.Values[15]) / perspectiveScale;
            if (!(a > 1.0 && b < 0.0) ||
                Length(Subtract(depthRow, Scale(forward, a))) > OrthogonalityTolerance * Length(depthRow))
                return fail("Shadow metric requires finite increasing perspective depth without oblique clipping");
            candidate.NearDistance = -b / a;
            candidate.FarDistance = -b / (a - 1.0);
            right = Scale(row(0), 1.0 / perspectiveScale); up = Scale(row(1), 1.0 / perspectiveScale);
            right = Subtract(right, Scale(forward, Dot(right, forward)));
            up = Subtract(up, Scale(forward, Dot(up, forward)));
            candidate.Width = 2.0 * candidate.NearDistance / Length(right);
            candidate.Height = 2.0 * candidate.NearDistance / Length(up);
        }
        if (!IsValidShadowProjection(candidate) || !Orthogonal(right, up) ||
            !Orthogonal(right, forward) || !Orthogonal(up, forward))
            return fail("Shadow metric requires orthogonal axes and nondegenerate native-float distances");
        projection = candidate;
        return true;
    }

    double ShadowLinearDepth(const ShadowProjection& projection, double depth)
    {
        // Reciprocal interpolation avoids the near*far product in perspective.
        return projection.Perspective ? 1.0 / ((1.0 - depth) / projection.NearDistance + depth / projection.FarDistance) :
            depth * projection.FarDistance;
    }

    bool BuildShadowDepthEncoding(const ShadowProjection& projection, Float4& encoding, std::string& error)
    {
        encoding = {0, 0, 0, 0}; error.clear();
        if (!IsValidShadowProjection(projection))
        { error = "Linear shadow depth requires a valid projection metric"; return false; }
        const double interval = projection.FarDistance - projection.NearDistance;
        const double clipZ = projection.Perspective ? 0.0 : 1.0 / projection.ClipWScale;
        const double clipW = projection.Perspective ? (1.0 / projection.ClipWScale) / interval : 0.0;
        const double offset = projection.Perspective ? -projection.NearDistance / interval : 0.0;
        for (double value : {clipZ, clipW, offset})
            if (value != 0.0 && !NativeDistance(std::abs(value)))
            { error = "Linear shadow depth encoding is outside the native-float range"; return false; }
        encoding = {float(clipZ), float(clipW), float(offset), 0};
        return true;
    }
}
