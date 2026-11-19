#pragma once

#include "Renderer/Lighting/SceneLightCollection.h"

namespace Evans::Renderer
{
    struct HairLightSample
    {
        Float4 PositionKind; // Position, or zero for a directional source; kind matches LightType.
        Float4 DirectionRange; // Surface-to-light direction or rectangle outward normal; world range.
        Float4 Radiance; // Nonnegative emitted RGB, including rectangle quadrature weight.
    };
    static_assert(sizeof(HairLightSample)==48,"Hair light storage ABI changed");
    struct HairLightSamples
    {
        std::vector<HairLightSample> Samples;
        std::uint32_t ShadowBegin=0,ShadowEnd=0; // Half-open range belonging to the selected opaque shadow caster.
        // Retain the authoritative source index space. Its shadow caster may
        // lie beyond the four compatibility lights or expand to a rectangle grid.
        std::shared_ptr<const SceneLightCollection> SourceLights;
    };
    // Expand the complete scene-order light collection. Rectangles use the same
    // authored/physical area quadrature as the other scene-light consumers.
    // The native fiber evaluators retain their own direction, range, emitter
    // cosine and inverse-square calculations. Failure preserves output.
    bool BuildHairLightSamples(const LightingState& lighting,HairLightSamples& output,std::string& error);
    class PreparedHairLighting
    {
    public:
        bool Prepare(const LightingState& lighting,std::string& error);
        const std::shared_ptr<const HairLightSamples>& Samples() const noexcept { return Output; }
        std::uint64_t BuildCount() const noexcept { return Builds; }
        void Reset() noexcept { Output.reset(); Previous={}; Builds=0; }
    private:
        LightingState Previous;
        std::shared_ptr<const HairLightSamples> Output;
        std::uint64_t Builds=0;
    };
}
