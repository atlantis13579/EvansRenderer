#include "Renderer/Geometry/FrustumCulling.h"

#include <array>
#include <cmath>

namespace Evans::Renderer
{
    // Conservative homogeneous AABB test. Clip coordinates avoid dividing by W,
    // so a box crossing the eye or near plane is retained. Reject only when all
    // corners lie strictly outside the same plane. The renderer uses 0 <= Z <= W
    // internally; backends convert that convention at their shader boundary.
    // See Akenine-Moller et al., Real-Time Rendering, "Frustum Culling".
    bool IntersectsFrustum(const Float3& minimum, const Float3& maximum,
        const Matrix4& world, const Matrix4& viewProjection)
    {
        constexpr std::size_t PlaneCount = 6;
        constexpr unsigned CornerCount = 8;
        constexpr float ClipBoundaryTolerance = 1.0e-5f;
        std::array<bool, PlaneCount> allOutside;
        allOutside.fill(true);
        const auto transform = [](const Matrix4& matrix, const Float4& point)
        {
            const auto& m = matrix.Values;
            return Float4{
                m[0] * point.X + m[1] * point.Y + m[2] * point.Z + m[3] * point.W,
                m[4] * point.X + m[5] * point.Y + m[6] * point.Z + m[7] * point.W,
                m[8] * point.X + m[9] * point.Y + m[10] * point.Z + m[11] * point.W,
                m[12] * point.X + m[13] * point.Y + m[14] * point.Z + m[15] * point.W};
        };
        for (unsigned corner = 0; corner < CornerCount; ++corner)
        {
            const Float4 local{
                (corner & 1) ? maximum.X : minimum.X,
                (corner & 2) ? maximum.Y : minimum.Y,
                (corner & 4) ? maximum.Z : minimum.Z, 1.0f};
            const Float4 clip = transform(viewProjection, transform(world, local));
            const std::array<float, PlaneCount> distances{
                clip.W + clip.X, clip.W - clip.X, clip.W + clip.Y,
                clip.W - clip.Y, clip.Z, clip.W - clip.Z};
            for (std::size_t plane = 0; plane < PlaneCount; ++plane)
            {
                // Invalid geometry must not cause an unsafe visibility rejection.
                if (!std::isfinite(distances[plane])) return true;
                allOutside[plane] = allOutside[plane] &&
                    distances[plane] < -ClipBoundaryTolerance;
            }
        }
        for (bool outside : allOutside) if (outside) return false;
        return true;
    }
}
