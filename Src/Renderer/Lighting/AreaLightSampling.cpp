#include "Renderer/Lighting/AreaLightSampling.h"

#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    // Surface-area integration, using deterministic stratum centers rather than
    // random samples. See PBRT 4e, "Area Lights":
    // https://pbr-book.org/4ed/Light_Sources/Area_Lights
    // This is numerical quadrature, not an LTC fit or an analytic GGX integral.
    namespace
    {
        constexpr double Pi = 3.14159265358979323846;
        constexpr float HalfSample = 0.5f;
        constexpr float VectorSquaredEpsilon = 1.0e-16f;
        constexpr float ParallelAxisLimit = 0.999f;
        constexpr std::uint32_t LegacySamplesPerAxis = 2;
        using Maths::Vector3;
        Vector3 Vector(Float3 v) noexcept { return {v.X, v.Y, v.Z}; }
        Vector3 Unit(Vector3 v, Vector3 fallback) noexcept
        {
            // Scale before normalizing so a finite authored direction cannot
            // overflow its squared length. Zero directions retain the fallback.
            const float scale = std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)});
            if (scale <= 0) return fallback;
            v /= scale;
            return v.LengthSquared() > VectorSquaredEpsilon ? v.Normalized() : fallback;
        }
        Vector3 LegacyUnit(Vector3 v, Vector3 fallback) noexcept
        { return v.LengthSquared() > VectorSquaredEpsilon ? v.Normalized() : fallback; }
    }

    Maths::Vector3 RectangleEmitterQuadrature::Position(std::uint32_t x, std::uint32_t y) const noexcept
    {
        const float u = (static_cast<float>(x) + HalfSample) / SamplesPerAxis - HalfSample;
        const float v = (static_cast<float>(y) + HalfSample) / SamplesPerAxis - HalfSample;
        return Center + Tangent * (u * Size.X) + Bitangent * (v * Size.Y);
    }

    RectangleEmitterQuadrature BuildRectangleEmitterQuadrature(const SceneLight& light) noexcept
    {
        RectangleEmitterQuadrature result;
        result.Center = Vector(light.Position); result.Size = light.AreaSize;
        const bool physical = light.AreaSampleAxisCount != 0;
        const auto normalize = physical ? Unit : LegacyUnit;
        result.Normal = normalize(Vector(light.Direction), -Vector3::UnitY());
        const auto helper = std::abs(result.Normal.y) < ParallelAxisLimit ? Vector3::UnitY() : Vector3::UnitX();
        result.Tangent = normalize(helper.Cross(result.Normal), Vector3::UnitX());
        result.Bitangent = result.Normal.Cross(result.Tangent);
        result.SamplesPerAxis = !physical ? LegacySamplesPerAxis :
            std::clamp(light.AreaSampleAxisCount, Techniques::MinimumAreaLightSamplesPerAxis, Techniques::MaximumAreaLightSamplesPerAxis);
        const double measure = !physical ? 1.0 : std::abs(double(light.AreaSize.X) * light.AreaSize.Y);
        result.Weight = static_cast<float>(measure / (result.SamplesPerAxis * result.SamplesPerAxis));
        return result;
    }

    bool PreparedAreaLighting::Prepare(const LightingState& authored, Techniques::AreaLightSettings settings, std::string& error)
    {
        error.clear(); settings = Techniques::SanitizeAreaLightSettings(settings);
        if (!settings.Enabled)
        {
            Reset(); Lighting = authored;
            if (Lighting.SceneLights) Lighting.SceneLights->WriteLegacySubset(Lighting);
            return true;
        }
        if (!HasPrevious || !(settings == PreviousSettings) || !SameSceneLights(authored, PreviousAuthored))
        {
            std::vector<SceneLight> lights = authored.SceneLights ? authored.SceneLights->GetLights() :
                std::vector<SceneLight>(authored.Lights.begin(), authored.Lights.begin() + std::min(authored.LightCount, MaximumSceneLights));
            const auto shadow = authored.SceneLights ? authored.SceneLights->GetShadowLightIndex() : authored.ShadowLightIndex;
            for (std::size_t index = 0; index < lights.size(); ++index)
            {
                auto& light = lights[index];
                if (light.AreaSampleAxisCount != 0)
                {
                    error = "Area lighting requires authored lights, not an already prepared transport snapshot";
                    Reset(); return false;
                }
                if (light.Type != LightType::RectArea) continue;
                const double area = std::abs(double(light.AreaSize.X) * light.AreaSize.Y);
                if (!std::isfinite(area) || area > std::numeric_limits<float>::max() || !std::isfinite(light.Intensity))
                {
                    error = "Area light " + std::to_string(index) + " has non-finite or unrepresentable radiometric data";
                    Reset(); return false;
                }
                light.AreaSize = {std::abs(light.AreaSize.X), std::abs(light.AreaSize.Y)};
                double radiance = std::max(double(light.Intensity), 0.0);
                if (area == 0) radiance = 0;
                else if (settings.EnergyUnit == Techniques::AreaLightEnergyUnit::Power) radiance /= Pi * area;
                if (radiance > std::numeric_limits<float>::max())
                {
                    error = "Area light " + std::to_string(index) + " power requires unrepresentable emitted radiance";
                    Reset(); return false;
                }
                light.Intensity = static_cast<float>(radiance);
                light.AreaSampleAxisCount = settings.SamplesPerAxis;
            }
            auto prepared = SceneLightCollection::Create(std::move(lights), shadow, error);
            if (!prepared) { Reset(); return false; }
            PreparedLights = std::move(prepared); PreviousAuthored = authored; PreviousSettings = settings; HasPrevious = true;
        }
        Lighting = authored; Lighting.SceneLights = PreparedLights;
        PreparedLights->WriteLegacySubset(Lighting);
        return true;
    }

    void PreparedAreaLighting::Reset() noexcept
    { Lighting = {}; PreviousAuthored = {}; PreviousSettings = {}; PreparedLights.reset(); HasPrevious = false; }
}
