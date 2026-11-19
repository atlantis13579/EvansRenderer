#pragma once

#include "Renderer/Lighting/ForwardPlus.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    // One immutable scene-order index space for lighting, support bounds and
    // tile-list indices. Publish a new collection when lights change; queued
    // draws keep their old shared snapshot alive until they finish.
    class SceneLightCollection final
    {
    public:
        // Range is canonicalized before packing or bounding. Invalid light data,
        // shadow selection or capacity is rejected without publishing a prefix.
        [[nodiscard]] static std::shared_ptr<const SceneLightCollection> Create(
            std::vector<SceneLight> lights, std::int32_t shadowLightIndex, std::string& error);

        [[nodiscard]] const std::vector<SceneLight>& GetLights() const noexcept { return Lights; }
        [[nodiscard]] const std::vector<ShaderLight>& GetShaderLights() const noexcept { return ShaderLights; }
        [[nodiscard]] const std::vector<ForwardPlusLightBounds>& GetBounds() const noexcept { return Bounds; }
        [[nodiscard]] std::int32_t GetShadowLightIndex() const noexcept { return ShadowLightIndex; }

        // Existing consumers retain the first four lights, replacing the last
        // slot if needed to retain the explicitly selected shadow caster. The
        // full collection and its shadow index are never reordered or truncated.
        void WriteLegacySubset(LightingState& lighting) const noexcept;

    private:
        SceneLightCollection() = default;
        std::vector<SceneLight> Lights;
        std::vector<ShaderLight> ShaderLights;
        std::vector<ForwardPlusLightBounds> Bounds;
        std::int32_t ShadowLightIndex = -1;
    };

    // Value equality also accepts equivalent inline/snapshot representations.
    // Pointer identity is a fast path, not the definition of lighting equality.
    [[nodiscard]] bool SameSceneLights(const LightingState& a, const LightingState& b) noexcept;
}
