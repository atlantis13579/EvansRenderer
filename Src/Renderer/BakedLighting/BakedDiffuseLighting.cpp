#include "Renderer/BakedLighting/BakedDiffuseLighting.h"
#include "Renderer/Lighting/AreaLightSampling.h"
#include "Techniques/LightRange.h"
#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    // One nonrecursive diffuse bounce. PBR retains Lambert normalization and
    // dielectric/metallic energy removal; legacy shading retains its diffuse
    // scale. Prepared rectangles use area-weighted radiance quadrature; authored
    // legacy rectangles retain normalized four-sample lighting.
    // Ambient, specular reflection and previous GI are deliberately excluded.
    namespace
    {
        using Maths::Vector3;
        constexpr double Pi = 3.14159265358979323846;
        constexpr float DielectricReflectance = 0.04f;
        constexpr float LightDistanceSquaredFloor = 1.0e-6f;
        constexpr float VectorSquaredEpsilon = 1.0e-16f;

        Vector3 Vector(Float3 v) { return {v.X, v.Y, v.Z}; }
        Float3 Native(Vector3 v) { return {v.x, v.y, v.z}; }
        Vector3 Nonnegative(Vector3 v) { return Vector3::Max(v, Vector3::Zero()); }
        Vector3 Unit(Vector3 v, Vector3 fallback)
        { return v.LengthSquared() > VectorSquaredEpsilon ? v.Normalized() : fallback; }

    }

    Maths::Vector3 EvaluateBakedDiffuse(const SceneRayQuery& scene, const SceneRayHit& hit,
        const std::vector<SceneLight>& lights, const BakedDiffuseLightingSettings& settings, std::uint64_t& shadowQueries)
    {
        const auto normal = Vector(hit.Normal), position = Vector(hit.Position);
        const auto geometric = Vector(hit.GeometricNormal);
        auto diffuse = Vector(hit.Diffuse);
        if (scene.UsesPBR()) diffuse *= (1.0f - DielectricReflectance) * (1.0f - hit.Metallic) / static_cast<float>(Pi);
        Vector3 illumination;
        auto accumulate = [&](Vector3 direction, float distance, Vector3 radiance, bool castsShadow) {
            const float cosine = std::max(normal.Dot(direction), 0.0f);
            if (cosine <= 0 || radiance == Vector3::Zero()) return;
            if (settings.Shadows && castsShadow)
            {
                // Use the geometric hemisphere for the offset; a strongly
                // perturbed normal must not start the shadow ray inside a wall.
                const auto origin = position + geometric * (geometric.Dot(direction) >= 0 ? settings.SurfaceBias : -settings.SurfaceBias);
                const auto toTarget = position + direction * distance - origin;
                const float targetDistance = toTarget.Length();
                SceneRayHit blocker;
                ++shadowQueries;
                if (scene.Trace(Native(origin), Native(toTarget), settings.SurfaceBias,
                    std::max(settings.SurfaceBias, targetDistance - settings.SurfaceBias), blocker, true)) return;
            }
            illumination += radiance * cosine;
        };
        for (const auto& light : lights)
        {
            const auto emission = Nonnegative(Vector(light.Color)) * std::max(light.Intensity, 0.0f);
            if (light.Type == LightType::Directional)
            {
                // Shadow extent follows the actual scene, not the secondary ray
                // truncation distance; remote blockers still cast shadows.
                const auto& bounds = scene.GetBounds();
                const float shadowDistance = (bounds.Max - bounds.Min).Length() + settings.SurfaceBias * 2;
                accumulate(Unit(Vector(light.Direction), Vector3::UnitY()), shadowDistance, emission, light.CastShadows);
                continue;
            }
            auto localSample = [&](Vector3 emitter, Vector3 emitterNormal, float weight, bool rectangle) {
                const auto offset = emitter - position;
                const float squaredDistance = offset.LengthSquared();
                if (squaredDistance <= VectorSquaredEpsilon) return;
                const float distance = std::sqrt(squaredDistance);
                const auto direction = offset / distance;
                const float emitterCosine = rectangle ? std::max(emitterNormal.Dot(-direction), 0.0f) : 1.0f;
                const float window = Techniques::LightRangeWindow(offset.x, offset.y, offset.z, light.Range);
                accumulate(direction, distance, emission * (weight * emitterCosine * window /
                    std::max(squaredDistance, LightDistanceSquaredFloor)), light.CastShadows);
            };
            if (light.Type == LightType::Point) { localSample(Vector(light.Position), {}, 1, false); continue; }
            const auto quadrature = BuildRectangleEmitterQuadrature(light);
            for (std::uint32_t y = 0; y < quadrature.SamplesPerAxis; ++y)
                for (std::uint32_t x = 0; x < quadrature.SamplesPerAxis; ++x)
                    localSample(quadrature.Position(x, y), quadrature.Normal, quadrature.Weight, true);
        }
        return Vector(hit.Emission) + diffuse * illumination;
    }
}
