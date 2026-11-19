#pragma once

#include "Renderer/Lighting/SceneLightCollection.h"
#include "Maths/Vector3.h"
#include "Techniques/AreaLights.h"

namespace Evans::Renderer
{
    // Midpoint quadrature of an emitting rectangle. Weight is dA for physical
    // radiance, or the legacy normalized weight when AreaSampleAxisCount is zero.
    // Weighted incident light is Le * Weight * max(dot(normal, -wi), 0) / distance^2;
    // the receiver supplies its own BRDF, cosine, range window and visibility.
    struct RectangleEmitterQuadrature
    {
        Maths::Vector3 Center, Normal, Tangent, Bitangent;
        Float2 Size;
        std::uint32_t SamplesPerAxis = 2;
        float Weight = 0.25f;

        [[nodiscard]] Maths::Vector3 Position(std::uint32_t x, std::uint32_t y) const noexcept;
    };

    // Input is a validated rectangle from SceneLightCollection or the preparer.
    [[nodiscard]] RectangleEmitterQuadrature BuildRectangleEmitterQuadrature(const SceneLight& light) noexcept;

    // Retains authored intensities and publishes a separate immutable transport
    // snapshot. Camera and ambient edits reuse the light buffers; energy units,
    // quality and emitter edits invalidate both direct and baked lighting keys.
    // The caller must pass authored lights, never a previous prepared result.
    class PreparedAreaLighting
    {
    public:
        [[nodiscard]] bool Prepare(const LightingState& authored, Techniques::AreaLightSettings settings, std::string& error);
        [[nodiscard]] const LightingState& GetLighting() const noexcept { return Lighting; }
        void Reset() noexcept;

    private:
        LightingState Lighting, PreviousAuthored;
        Techniques::AreaLightSettings PreviousSettings;
        std::shared_ptr<const SceneLightCollection> PreparedLights;
        bool HasPrevious = false;
    };
}
