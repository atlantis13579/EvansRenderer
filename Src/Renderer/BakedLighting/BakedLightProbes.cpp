#include "Renderer/BakedLighting/BakedLightProbes.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include "Renderer/BakedLighting/BakedDiffuseLighting.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    /*
    Baked diffuse probe sample producer
    ----------------------------------
    Uniform Fibonacci directions sample the incident radiance field at each
    regular-grid probe. A closest surface contributes emission plus shadowed
    direct diffuse reflection; miss/backface samples are black. This is one
    diffuse bounce, not an iterative DDGI solver or specular path tracer.

    The intended representation is L2 irradiance SH plus directional distance
    moments. SH projection convolves bands with pi, 2*pi/3, pi/4; the separate
    visibility moments prevent ordinary trilinear weights from freely leaking
    light through walls. Independent native shaders project and gather this data.
    Ramamoorthi and Hanrahan, "An Efficient Representation for Irradiance
    Environment Maps" (SIGGRAPH 2001): https://graphics.stanford.edu/papers/envmap/
    Majercik et al., "Dynamic Diffuse Global Illumination with Ray-Traced
    Irradiance Fields" (JCGT 2019): https://jcgt.org/published/0008/02/01/

    Geometry queries include off-screen surfaces and every scene light, with
    per-emitter shadow rays. Rectangle emission retains the project's existing
    four-sample normalized-intensity convention, not a new area-light model.
    PBR uses Lambert diffuse with dielectric F0 energy removal and metallic
    suppression; legacy modes retain their unnormalized Phong diffuse scale.
    Ambient light is excluded to avoid manufacturing an extra ambient bounce.
    No recursion, alpha transmission, microfacet reflection or NPR stylization
    is baked. Mip-zero sampling and finite ray counts can alias small features.

    Backface rejection assumes consistently oriented closed geometry. It is a
    conservative inside-solid heuristic; thin/open surfaces can reject useful
    probes. There is no automatic probe relocation. Miss distance is finite so
    subsequent moment projection never squares infinity.
    */
    namespace
    {
        using Maths::Vector3;
        constexpr double Pi = 3.14159265358979323846;
        constexpr float HalfSample = 0.5f;

        Vector3 Vector(Float3 v) { return {v.X, v.Y, v.Z}; }
        Float3 Native(Vector3 v) { return {v.x, v.y, v.z}; }
        bool Finite(Vector3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
        Vector3 Nonnegative(Vector3 v) { return Vector3::Max(v, Vector3::Zero()); }

    }

    bool BakeLightProbeSamples(const SceneRayQuery& scene, const LightingState& lighting,
        const BakedProbeSettings& settings, BakedProbeSamples& output, std::string& error)
    {
        error.clear();
        const auto minimum = Vector(settings.Minimum), maximum = Vector(settings.Maximum);
        const auto extent = maximum - minimum;
        if (!Finite(minimum) || !Finite(maximum) || !Finite(extent) || extent.x <= 0 || extent.y <= 0 || extent.z <= 0 ||
            !std::isfinite(extent.LengthSquared()) ||
            !std::all_of(settings.Counts.begin(), settings.Counts.end(), [](auto count) { return count >= MinimumProbeGridAxis && count <= MaximumProbeGridAxis; }) ||
            settings.RayCount < MinimumProbeRayCount || settings.RayCount > MaximumProbeRayCount ||
            !std::isfinite(settings.MaximumDistance) || settings.MaximumDistance <= 0 ||
            !std::isfinite(settings.MaximumDistance * settings.MaximumDistance) ||
            !std::isfinite(settings.SurfaceBias) || settings.SurfaceBias <= 0 || settings.SurfaceBias >= settings.MaximumDistance ||
            !std::isfinite(settings.MaximumBackfaceFraction) || settings.MaximumBackfaceFraction < 0 || settings.MaximumBackfaceFraction > 1)
        { error = "Invalid probe bounds, grid dimensions, sampling distance or bias"; return false; }

        auto lights = lighting.SceneLights;
        if (!lights)
        {
            if (lighting.LightCount > MaximumSceneLights) { error = "Inline probe lighting exceeds its light capacity"; return false; }
            lights = SceneLightCollection::Create({lighting.Lights.begin(), lighting.Lights.begin() + lighting.LightCount}, -1, error);
            if (!lights) return false;
        }
        BakedProbeSamples result;
        result.Settings = settings;
        result.Texels.resize(static_cast<std::size_t>(result.Width()) * result.Height(), Float4{0,0,0,0});
        const double goldenAngle = Pi * (3.0 - std::sqrt(5.0));
        const float solidAngle = static_cast<float>(4.0 * Pi / settings.RayCount);
        for (std::uint32_t ray = 0; ray < settings.RayCount; ++ray)
        {
            const double z = 1.0 - 2.0 * (ray + double(HalfSample)) / settings.RayCount;
            const double radius = std::sqrt(std::max(0.0, 1.0 - z*z)), angle = goldenAngle * ray;
            result.Texels[ray + ProbeMetadataBorder] = {static_cast<float>(radius*std::cos(angle)),
                static_cast<float>(radius*std::sin(angle)), static_cast<float>(z), solidAngle};
        }
        for (std::uint32_t probe = 0; probe < result.ProbeCount(); ++probe)
        {
            const auto x = probe % settings.Counts[0], y = (probe / settings.Counts[0]) % settings.Counts[1];
            const auto z = probe / (settings.Counts[0] * settings.Counts[1]);
            const Vector3 fraction {float(x)/float(settings.Counts[0]-1), float(y)/float(settings.Counts[1]-1), float(z)/float(settings.Counts[2]-1)};
            const auto position = minimum + extent*fraction;
            const auto row = static_cast<std::size_t>(probe + ProbeMetadataBorder) * result.Width();
            std::uint32_t backfaces = 0;
            for (std::uint32_t ray = 0; ray < settings.RayCount; ++ray)
            {
                const auto direction = result.Texels[ray + ProbeMetadataBorder];
                SceneRayHit hit;
                Float4 sample {0,0,0,settings.MaximumDistance};
                if (scene.Trace(Native(position), {direction.X, direction.Y, direction.Z}, settings.SurfaceBias, settings.MaximumDistance, hit))
                {
                    ++result.SurfaceHitCount;
                    sample.W = hit.Distance;
                    if (!hit.FrontFace) ++backfaces;
                    else
                    {
                        auto radiance = EvaluateBakedDiffuse(scene, hit, lights->GetLights(),
                            {settings.SurfaceBias, settings.Shadows}, result.ShadowQueryCount);
                        if (!Finite(radiance)) { error = "Probe radiance overflow; reduce material or emitter intensity"; return false; }
                        radiance = Vector3::Min(Nonnegative(radiance), Vector3(MaximumBakedRadiance));
                        sample.X = radiance.x; sample.Y = radiance.y; sample.Z = radiance.z;
                    }
                }
                result.Texels[row + ray + ProbeMetadataBorder] = sample;
            }
            const bool active = float(backfaces) / settings.RayCount <= settings.MaximumBackfaceFraction;
            result.Texels[row] = {position.x, position.y, position.z, active ? 1.0f : 0.0f};
            if (active) ++result.ActiveProbeCount;
        }
        output = std::move(result);
        return true;
    }
}
