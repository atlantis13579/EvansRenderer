#include "Renderer/Shadows/CascadedShadowMap.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    // Zhang, Sun and Nyman, "Parallel-Split Shadow Maps on Programmable GPUs",
    // GPU Gems 3, chapter 10 (2007): practical linear/logarithmic depth splits.
    // https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus
    // https://learn.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps
    // Representation and filtering are separate: this producer fits directional
    // cascades; depth generation and PCF/PCSS sampling consume the resulting data.
    // Sphere fitting fixes XY scale under camera rotation, texel snapping fixes
    // XY translation, and overlapping depth intervals provide seam-free blending.
    // Caster bounds extend depth toward the light without expanding receiver XY.
    namespace
    {
        constexpr double MatrixTolerance = 1.0e-4;
        constexpr double MinimumHomogeneousW = 1.0e-12;
        constexpr double MinimumSpan = 1.0e-8;
        constexpr double ParallelUpThreshold = 0.95;
        constexpr double RadiusQuantizationSteps = 4096.0;
        constexpr double DepthPaddingFraction = 0.01;
        constexpr double SnapAndFilterMargin = 1.0; // half a texel each for snapping and sampling

        struct Vector
        {
            double X = 0, Y = 0, Z = 0;
            Vector operator+(Vector b) const { return {X + b.X, Y + b.Y, Z + b.Z}; }
            Vector operator-(Vector b) const { return {X - b.X, Y - b.Y, Z - b.Z}; }
            Vector operator*(double scale) const { return {X * scale, Y * scale, Z * scale}; }
        };
        double Dot(Vector a, Vector b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
        Vector Cross(Vector a, Vector b) { return {a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X}; }
        double Length(Vector value) { return std::sqrt(Dot(value, value)); }
        bool Finite(Vector value) { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z); }
        Vector Expand(Float3 value) { return {value.X, value.Y, value.Z}; }
        Float3 Pack(Vector value) { return {float(value.X), float(value.Y), float(value.Z)}; }
        Vector Row(const Matrix4& matrix, int row)
        {
            return {matrix.Values[row * 4], matrix.Values[row * 4 + 1], matrix.Values[row * 4 + 2]};
        }

        bool Inverse(const Matrix4& matrix, double inverse[4][4])
        {
            // CPU fitting uses double intermediates. Normalizing the projective
            // matrix makes inversion independent of a uniform homogeneous scale.
            double scale = 0;
            for (float value : matrix.Values)
            {
                if (!std::isfinite(value)) return false;
                scale = std::max(scale, std::abs(double(value)));
            }
            if (scale == 0) return false;
            double augmented[4][8] {};
            for (int row = 0; row < 4; ++row)
            {
                for (int column = 0; column < 4; ++column) augmented[row][column] = matrix.Values[row * 4 + column] / scale;
                augmented[row][row + 4] = 1;
            }
            for (int column = 0; column < 4; ++column)
            {
                int pivot = column;
                for (int row = column + 1; row < 4; ++row)
                    if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column])) pivot = row;
                if (std::abs(augmented[pivot][column]) < MinimumHomogeneousW) return false;
                for (int entry = 0; entry < 8; ++entry) std::swap(augmented[pivot][entry], augmented[column][entry]);
                const double divisor = augmented[column][column];
                for (double& value : augmented[column]) value /= divisor;
                for (int row = 0; row < 4; ++row)
                {
                    if (row == column) continue;
                    const double factor = augmented[row][column];
                    for (int entry = 0; entry < 8; ++entry) augmented[row][entry] -= factor * augmented[column][entry];
                }
            }
            for (int row = 0; row < 4; ++row)
                for (int column = 0; column < 4; ++column) inverse[row][column] = augmented[row][column + 4];
            return true;
        }

        bool Unproject(const double inverse[4][4], double x, double y, double z, Vector& output)
        {
            const double w = inverse[3][0] * x + inverse[3][1] * y + inverse[3][2] * z + inverse[3][3];
            if (!std::isfinite(w) || std::abs(w) <= MinimumHomogeneousW) return false;
            output = {(inverse[0][0] * x + inverse[0][1] * y + inverse[0][2] * z + inverse[0][3]) / w,
                (inverse[1][0] * x + inverse[1][1] * y + inverse[1][2] * z + inverse[1][3]) / w,
                (inverse[2][0] * x + inverse[2][1] * y + inverse[2][2] * z + inverse[2][3]) / w};
            return Finite(output);
        }

        bool RigidView(const Matrix4& view)
        {
            for (float value : view.Values) if (!std::isfinite(value)) return false;
            if (Length(Row(view, 3)) > MatrixTolerance || std::abs(view.Values[15] - 1) > MatrixTolerance) return false;
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    if (std::abs(Dot(Row(view, row), Row(view, column)) - (row == column ? 1 : 0)) > MatrixTolerance) return false;
            return Dot(Cross(Row(view, 0), Row(view, 1)), Row(view, 2)) > 1 - MatrixTolerance;
        }
        Vector ToWorld(const Matrix4& view, Vector point)
        {
            return Row(view, 0) * (point.X - view.Values[3]) + Row(view, 1) * (point.Y - view.Values[7]) +
                Row(view, 2) * (point.Z - view.Values[11]);
        }
    }

    bool BuildCascadedShadowFrame(const CascadedShadowCamera& camera, const SceneLight& light,
        const std::vector<ShadowCasterBounds>& casters, const CascadedShadowSettings& requested,
        int mapResolution, float filterRadiusTexels, CascadedShadowFrame& output, std::string& error)
    {
        output = {};
        error.clear();
        const auto fail = [&](const char* message) { error = message; return false; };
        const auto settings = SanitizeCascadedShadowSettings(requested);
        const double margin = double(filterRadiusTexels) + SnapAndFilterMargin;
        if (!std::isfinite(filterRadiusTexels) || filterRadiusTexels < 0 || mapResolution <= 2 * margin)
            return fail("CSM resolution must leave receiver texels inside the sampling guard band");
        if (light.Type != LightType::Directional || !Finite(Expand(light.Direction)) || Length(Expand(light.Direction)) <= MinimumSpan)
            return fail("CSM requires a finite directional light vector");
        if (!RigidView(camera.View)) return fail("CSM camera view must be a finite rigid right-handed transform");
        double inverse[4][4] {};
        if (!Inverse(camera.Projection, inverse)) return fail("CSM camera projection must be finite and invertible");
        const bool perspective = Length(Row(camera.Projection, 3)) > 0;
        std::array<Vector, 4> nearCorners {}, middleCorners {};
        double nearDistance = 0, cameraFar = 0;
        bool finiteFar = true;
        for (int corner = 0; corner < 4; ++corner)
        {
            const double x = (corner & 1) ? 1 : -1, y = (corner & 2) ? 1 : -1;
            if (!Unproject(inverse, x, y, 0, nearCorners[corner]) || !Unproject(inverse, x, y, 0.5, middleCorners[corner]))
                return fail("CSM cannot reconstruct the near and middle camera planes");
            if (corner == 0) nearDistance = -nearCorners[corner].Z;
            if (std::abs(-nearCorners[corner].Z - nearDistance) > MatrixTolerance * std::max(1.0, std::abs(nearDistance)) ||
                -middleCorners[corner].Z <= nearDistance)
                return fail("CSM requires parallel, forward zero-to-one camera depth planes");
            Vector farCorner;
            const bool hasFar = Unproject(inverse, x, y, 1, farCorner);
            if (corner == 0) { finiteFar = hasFar; cameraFar = hasFar ? -farCorner.Z : 0; }
            if (hasFar != finiteFar || (hasFar && std::abs(-farCorner.Z - cameraFar) > MatrixTolerance * std::max(1.0, std::abs(cameraFar))))
                return fail("CSM camera far plane is not parallel to its near plane");
        }
        if (nearDistance < 0 || (perspective && nearDistance <= MinimumSpan) || (finiteFar && cameraFar <= nearDistance))
            return fail("CSM requires a nonnegative near distance and forward camera depth");
        if (!finiteFar && settings.MaximumDistance <= nearDistance)
            return fail("An infinite-far CSM camera requires an explicit finite shadow distance");
        const double farDistance = settings.MaximumDistance > 0 ?
            (finiteFar ? std::min(cameraFar, double(settings.MaximumDistance)) : settings.MaximumDistance) : cameraFar;
        if (farDistance <= nearDistance + MinimumSpan) return fail("CSM shadow distance must exceed the camera near plane");
        for (const auto& caster : casters)
            if (!Finite(Expand(caster.Minimum)) || !Finite(Expand(caster.Maximum)) || caster.Minimum.X > caster.Maximum.X ||
                caster.Minimum.Y > caster.Maximum.Y || caster.Minimum.Z > caster.Maximum.Z)
                return fail("CSM caster bounds must be finite ordered world-space boxes");

        CascadedShadowFrame frame;
        frame.CascadeCount = settings.CascadeCount;
        frame.NearDistance = float(nearDistance); frame.FarDistance = float(farDistance); frame.Perspective = perspective;
        if (!std::isfinite(frame.FarDistance) || frame.FarDistance <= frame.NearDistance)
            return fail("CSM camera depth interval exceeds float precision");
        const Vector backward = Row(camera.View, 2);
        frame.EyeDepthPlane = {float(-backward.X), float(-backward.Y), float(-backward.Z), -camera.View.Values[11]};
        const Vector forward = Expand(light.Direction) * (-1 / Length(Expand(light.Direction)));
        const Vector upSeed = std::abs(forward.Y) > ParallelUpThreshold ? Vector{1, 0, 0} : Vector{0, 1, 0};
        const Vector cross = Cross(forward, upSeed);
        const Vector right = cross * (1 / Length(cross)), up = Cross(right, forward);
        double previousSplit = nearDistance;
        for (std::uint32_t index = 0; index < frame.CascadeCount; ++index)
        {
            auto& cascade = frame.Cascades[index];
            const double fraction = double(index + 1) / frame.CascadeCount;
            const double uniform = nearDistance + (farDistance - nearDistance) * fraction;
            const double logarithmic = perspective ? nearDistance * std::pow(farDistance / nearDistance, fraction) : uniform;
            const double split = index + 1 == frame.CascadeCount ? farDistance : uniform + (logarithmic - uniform) * settings.SplitLambda;
            cascade.FitNearDistance = index == 0 ? frame.NearDistance : frame.Cascades[index - 1].BlendStartDistance;
            cascade.FarDistance = float(split);
            cascade.BlendStartDistance = float(split - (split - previousSplit) * settings.BlendFraction);
            if (cascade.FarDistance <= float(previousSplit))
                return fail("CSM split intervals collapse at float precision");
            previousSplit = split;
            std::array<Vector, 8> localCorners {}, worldCorners {};
            Vector localCenter;
            for (int corner = 0; corner < 8; ++corner)
            {
                const int ray = corner % 4;
                const double distance = corner < 4 ? cascade.FitNearDistance : cascade.FarDistance;
                const Vector direction = middleCorners[ray] - nearCorners[ray];
                localCorners[corner] = nearCorners[ray] + direction * ((distance + nearCorners[ray].Z) / -direction.Z);
                localCenter = localCenter + localCorners[corner] * (1.0 / localCorners.size());
                worldCorners[corner] = ToWorld(camera.View, localCorners[corner]);
                cascade.ReceiverCorners[corner] = Pack(worldCorners[corner]);
            }
            double radius = 0;
            for (Vector corner : localCorners) radius = std::max(radius, Length(corner - localCenter));
            const double quantum = farDistance / RadiusQuantizationSteps;
            radius = std::ceil(radius / quantum) * quantum;
            const double extent = radius * mapResolution / (mapResolution - 2 * margin);
            const double texel = 2 * extent / mapResolution;
            if (!std::isfinite(texel) || texel <= MinimumSpan) return fail("CSM receiver footprint has no finite texel scale");
            const Vector center = ToWorld(camera.View, localCenter);
            const double centerX = std::round(Dot(center, right) / texel) * texel;
            const double centerY = std::round(Dot(center, up) / texel) * texel;
            double minimumZ = std::numeric_limits<double>::infinity(), maximumZ = -minimumZ;
            for (Vector corner : worldCorners) { minimumZ = std::min(minimumZ, Dot(corner, forward)); maximumZ = std::max(maximumZ, Dot(corner, forward)); }
            for (const auto& caster : casters)
            {
                double minX = std::numeric_limits<double>::infinity(), minY = minX, minZ = minX;
                double maxX = -minX, maxY = -minX;
                for (int corner = 0; corner < 8; ++corner)
                {
                    const Vector world {(corner & 1) ? caster.Maximum.X : caster.Minimum.X,
                        (corner & 2) ? caster.Maximum.Y : caster.Minimum.Y, (corner & 4) ? caster.Maximum.Z : caster.Minimum.Z};
                    minX = std::min(minX, Dot(world, right)); maxX = std::max(maxX, Dot(world, right));
                    minY = std::min(minY, Dot(world, up)); maxY = std::max(maxY, Dot(world, up));
                    minZ = std::min(minZ, Dot(world, forward));
                }
                if (maxX >= centerX - extent && minX <= centerX + extent && maxY >= centerY - extent && minY <= centerY + extent)
                    minimumZ = std::min(minimumZ, minZ);
            }
            const double padding = std::max((maximumZ - minimumZ) * DepthPaddingFraction, texel);
            minimumZ -= padding; maximumZ += padding;
            const double depthScale = 1 / (maximumZ - minimumZ);
            cascade.WorldUnitsPerTexel = float(texel);
            cascade.LightDepthMinimum = float(minimumZ); cascade.LightDepthMaximum = float(maximumZ);
            if (!std::isfinite(cascade.WorldUnitsPerTexel) || !std::isfinite(cascade.LightDepthMinimum) ||
                !std::isfinite(cascade.LightDepthMaximum) || cascade.LightDepthMaximum <= cascade.LightDepthMinimum)
                return fail("CSM light depth interval exceeds float precision");
            cascade.LightViewProjection.Values = {float(right.X / extent), float(right.Y / extent), float(right.Z / extent), float(-centerX / extent),
                float(up.X / extent), float(up.Y / extent), float(up.Z / extent), float(-centerY / extent),
                float(forward.X * depthScale), float(forward.Y * depthScale), float(forward.Z * depthScale), float(-minimumZ * depthScale),
                0, 0, 0, 1};
            for (float value : cascade.LightViewProjection.Values) if (!std::isfinite(value)) return fail("CSM projection exceeds float precision");
            for (Float3 corner : cascade.ReceiverCorners) if (!Finite(Expand(corner))) return fail("CSM receiver position exceeds float precision");
        }
        output = frame;
        return true;
    }

    ShadowCascadeSelection SelectShadowCascade(const CascadedShadowFrame& frame, float eyeDistance)
    {
        ShadowCascadeSelection selection;
        if (!std::isfinite(eyeDistance) || frame.CascadeCount > MaximumShadowCascades ||
            eyeDistance < frame.NearDistance || eyeDistance > frame.FarDistance) return selection;
        for (std::uint32_t index = 0; index < frame.CascadeCount; ++index)
        {
            const auto& cascade = frame.Cascades[index];
            if (eyeDistance > cascade.FarDistance) continue;
            selection.Primary = index;
            const float width = cascade.FarDistance - cascade.BlendStartDistance;
            if (width > 0 && eyeDistance >= cascade.BlendStartDistance)
            {
                selection.Secondary = index + 1 < frame.CascadeCount ? index + 1 : MaximumShadowCascades;
                selection.Blend = std::clamp((eyeDistance - cascade.BlendStartDistance) / width, 0.0f, 1.0f);
            }
            return selection;
        }
        return selection;
    }
}
