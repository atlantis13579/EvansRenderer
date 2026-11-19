#include "Renderer/Lighting/SceneLightCollection.h"
#include "Techniques/AreaLights.h"

#include <limits>
#include <utility>

namespace Evans::Renderer
{
    namespace
    {
        bool Finite(Float3 value) noexcept
        { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z); }

        bool Same(Float3 a, Float3 b) noexcept
        { return a.X == b.X && a.Y == b.Y && a.Z == b.Z; }

        bool Same(const SceneLight& a, const SceneLight& b) noexcept
        {
            return a.Type == b.Type && a.Range == b.Range && a.Intensity == b.Intensity &&
                a.CastShadows == b.CastShadows && Same(a.Position, b.Position) &&
                Same(a.Direction, b.Direction) && Same(a.Color, b.Color) &&
                a.AreaSize.X == b.AreaSize.X && a.AreaSize.Y == b.AreaSize.Y && a.AreaSampleAxisCount == b.AreaSampleAxisCount;
        }
    }

    std::shared_ptr<const SceneLightCollection> SceneLightCollection::Create(
        std::vector<SceneLight> lights, std::int32_t shadowLightIndex, std::string& error)
    {
        error.clear();
        if (lights.size() > MaximumForwardPlusLights)
        { error = "Scene light collection exceeds the Forward+ storage capacity"; return {}; }
        if (shadowLightIndex < -1 || (shadowLightIndex >= 0 &&
            (static_cast<std::size_t>(shadowLightIndex) >= lights.size() || !lights[shadowLightIndex].CastShadows)))
        { error = "Scene shadow-light index must select a shadow caster or be -1"; return {}; }

        auto result = std::shared_ptr<SceneLightCollection>(new SceneLightCollection);
        result->Lights = std::move(lights);
        result->ShadowLightIndex = shadowLightIndex;
        result->ShaderLights.reserve(result->Lights.size());
        result->Bounds.reserve(result->Lights.size());
        for (std::size_t index = 0; index < result->Lights.size(); ++index)
        {
            auto& light = result->Lights[index];
            if (light.Type > LightType::RectArea || !Finite(light.Position) || !Finite(light.Direction) ||
                !Finite(light.Color) || !std::isfinite(light.Intensity) ||
                !std::isfinite(light.AreaSize.X) || !std::isfinite(light.AreaSize.Y))
            { error = "Scene light " + std::to_string(index) + " has an invalid type or non-finite shading data"; return {}; }
            light.Range = Techniques::CanonicalLightRange(light.Range);
            if (light.AreaSampleAxisCount != 0 && (light.Type != LightType::RectArea ||
                light.AreaSampleAxisCount < Techniques::MinimumAreaLightSamplesPerAxis ||
                light.AreaSampleAxisCount > Techniques::MaximumAreaLightSamplesPerAxis ||
                std::abs(double(light.AreaSize.X) * light.AreaSize.Y) > std::numeric_limits<float>::max()))
            { error = "Scene light has an invalid physical area sampling contract"; return {}; }
            ForwardPlusLightBounds bounds;
            if (!BuildForwardPlusLightBounds(light, bounds, error)) return {};
            result->ShaderLights.push_back(BuildShaderLight(light));
            result->Bounds.push_back(bounds);
        }
        return result;
    }

    void SceneLightCollection::WriteLegacySubset(LightingState& lighting) const noexcept
    {
        lighting.Lights = {};
        lighting.LightCount = static_cast<std::uint32_t>(std::min<std::size_t>(Lights.size(), MaximumSceneLights));
        lighting.ShadowLightIndex = ShadowLightIndex;
        for (std::uint32_t index = 0; index < lighting.LightCount; ++index) lighting.Lights[index] = Lights[index];
        if (ShadowLightIndex >= static_cast<std::int32_t>(lighting.LightCount))
        {
            const auto slot = lighting.LightCount - 1;
            lighting.Lights[slot] = Lights[ShadowLightIndex];
            lighting.ShadowLightIndex = static_cast<std::int32_t>(slot);
        }
    }

    bool SameSceneLights(const LightingState& a, const LightingState& b) noexcept
    {
        if (a.SceneLights && a.SceneLights == b.SceneLights) return true;
        const auto countA = a.SceneLights ? a.SceneLights->GetLights().size() : std::min(a.LightCount, MaximumSceneLights);
        const auto countB = b.SceneLights ? b.SceneLights->GetLights().size() : std::min(b.LightCount, MaximumSceneLights);
        const auto shadowA = a.SceneLights ? a.SceneLights->GetShadowLightIndex() : a.ShadowLightIndex;
        const auto shadowB = b.SceneLights ? b.SceneLights->GetShadowLightIndex() : b.ShadowLightIndex;
        if (countA != countB || shadowA != shadowB) return false;
        for (std::size_t index = 0; index < countA; ++index)
        {
            const auto& lightA = a.SceneLights ? a.SceneLights->GetLights()[index] : a.Lights[index];
            const auto& lightB = b.SceneLights ? b.SceneLights->GetLights()[index] : b.Lights[index];
            if (!Same(lightA, lightB)) return false;
        }
        return true;
    }
}
