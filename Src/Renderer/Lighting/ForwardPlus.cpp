#include "Renderer/Lighting/ForwardPlus.h"
#include "Techniques/LightRange.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        // Harada, McKee and Yang, "Forward+: Bringing Deferred Lighting to the
        // Next Level" (2012), https://takahiroharada.github.io/forward%2B/.
        // Depth -> tile light assignment -> material lighting. The camera-depth
        // prerequisite may reuse an AO/reflection material pass; contact visibility
        // reads the same depth independently. Neither list generation nor its
        // depth-only prerequisite needs HDR, normals, history or new art assets.
        //
        // Forward+, Forward and Deferred are pipeline alternatives. Pixel-tile
        // lists cannot feed Gouraud's vertex evaluation: that combination needs
        // an explicit GUI compatibility rule, not a silent pixel-shading rewrite.
        // Transparent receivers use a separate full-depth list, including weighted
        // OIT. Reflection captures need lists from their own camera, or a full
        // light loop. They must never reuse the main camera's depth-restricted list.
        // Volume integration also needs its own assignment or a full-light loop:
        // scattering points can lie in front of the opaque tile depth interval.
        constexpr std::uint32_t ListsPerTile = 2;
    }

    bool BuildForwardPlusLightBounds(const SceneLight& light, ForwardPlusLightBounds& result, std::string& error)
    {
        result = {}; error.clear();
        if (light.Type > LightType::RectArea)
        { error = "Forward+ support requires a known scene-light type"; return false; }
        const float range = Techniques::CanonicalLightRange(light.Range);
        if (light.Type == LightType::Directional || range == 0)
        { result.Unbounded = true; return true; }
        if (!std::isfinite(light.Position.X) || !std::isfinite(light.Position.Y) || !std::isfinite(light.Position.Z) ||
            (light.Type == LightType::RectArea && (!std::isfinite(light.AreaSize.X) || !std::isfinite(light.AreaSize.Y))))
        { error = "Forward+ finite light support requires finite position and emitter dimensions"; return false; }
        const double emitterRadius = light.Type == LightType::RectArea ?
            std::hypot(double(light.AreaSize.X), double(light.AreaSize.Y)) / 2 : 0;
        const double centreMagnitude = std::max({std::abs(double(light.Position.X)),
            std::abs(double(light.Position.Y)), std::abs(double(light.Position.Z))});
        // The full rectangle encloses every quadrature point, independent of
        // orientation. Expand for float basis construction, position arithmetic
        // and window evaluation, then round outward. Extreme scales safely
        // become unbounded rather than creating an undersized/infinite sphere.
        constexpr double SupportRoundoff = 64 * std::numeric_limits<float>::epsilon();
        const double radius = range + emitterRadius + SupportRoundoff * (centreMagnitude + range + emitterRadius) +
            64 * double(std::numeric_limits<float>::min());
        if (radius >= std::numeric_limits<float>::max()) { result.Unbounded = true; return true; }
        const float rounded = std::nextafter(float(radius), std::numeric_limits<float>::infinity());
        if (!std::isfinite(rounded)) { result.Unbounded = true; return true; }
        result.Center = light.Position; result.Radius = rounded;
        return true;
    }

    bool BuildForwardPlusGrid(std::uint32_t width, std::uint32_t height, std::uint32_t samples,
        std::size_t lightCount, const ForwardPlusParameters& parameters, ForwardPlusGrid& result, std::string& error)
    {
        result = {}; error.clear();
        if (!width || !height || (samples != 1 && samples != 2 && samples != 4 && samples != 8) ||
            !parameters.TileSize || parameters.TileSize > MaximumForwardPlusTileSize || !parameters.TileCapacity ||
            parameters.TileCapacity > MaximumForwardPlusTileCapacity || lightCount > MaximumForwardPlusLights)
        { error = "Forward+ dimensions, sample count, light count and tile parameters must be bounded"; return false; }
        ForwardPlusGrid candidate;
        candidate.Width = width; candidate.Height = height; candidate.Samples = samples;
        candidate.LightCount = std::uint32_t(lightCount);
        candidate.TileSize = parameters.TileSize; candidate.TileCapacity = parameters.TileCapacity;
        candidate.TilesX = width / parameters.TileSize + (width % parameters.TileSize != 0 ? 1u : 0u);
        candidate.TilesY = height / parameters.TileSize + (height % parameters.TileSize != 0 ? 1u : 0u);
        const auto tiles = std::uint64_t(candidate.TilesX) * candidate.TilesY;
        constexpr auto MaximumBufferBytes = std::numeric_limits<std::uint32_t>::max();
        // Bound before multiplying so neither byte counts nor shader addresses
        // wrap. Native backends may impose a smaller allocation limit.
        if (tiles > MaximumBufferBytes / sizeof(ForwardPlusTile) ||
            tiles > MaximumBufferBytes / (ListsPerTile * parameters.TileCapacity * sizeof(std::uint32_t)))
        { error = "Forward+ list storage exceeds native 32-bit buffer addressing"; return false; }
        candidate.TileCount = std::uint32_t(tiles);
        candidate.IndexCount = candidate.TileCount * ListsPerTile * candidate.TileCapacity;
        result = candidate;
        return true;
    }
}
