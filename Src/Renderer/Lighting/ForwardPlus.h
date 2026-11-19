#pragma once

#include "Renderer/Types.h"

namespace Evans::Renderer
{
    // Host allocation and synchronization can serve both tiled pipelines. Each
    // producer instance owns one immutable choice of independently maintained
    // shader source; switching the frame pipeline never reuses another program.
    enum class TiledLightShader { ForwardPlus, TiledDeferred };
    inline constexpr std::uint32_t DefaultForwardPlusTileSize = 16;
    inline constexpr std::uint32_t MaximumForwardPlusTileSize = 32;
    inline constexpr std::uint32_t MaximumForwardPlusLights = 4096;
    inline constexpr std::uint32_t MaximumForwardPlusTileCapacity = 256;
    struct ForwardPlusParameters
    {
        std::uint32_t TileSize = DefaultForwardPlusTileSize;
        std::uint32_t TileCapacity = 128;
    };
    // A finite sphere must enclose every point with nonzero light contribution.
    // Inverse-square lights without an explicit cutoff, and directional lights,
    // are unbounded. A zero-radius sphere is distinct from an unbounded light.
    // Bounds describe support, not an artistic approximation based on brightness.
    struct ForwardPlusLightBounds
    {
        Float3 Center;
        float Radius = 0;
        bool Unbounded = false;
    };

    // Derive support from the SAME range used by surface and volume lighting.
    // Rectangles expand the sample radius by the emitter's half diagonal.
    [[nodiscard]] bool BuildForwardPlusLightBounds(const SceneLight& light,
        ForwardPlusLightBounds& result, std::string& error);
    struct ForwardPlusGrid
    {
        std::uint32_t Width = 0, Height = 0, Samples = 0, LightCount = 0;
        std::uint32_t TileSize = 0, TileCapacity = 0, TilesX = 0, TilesY = 0;
        std::uint32_t TileCount = 0, IndexCount = 0;
    };
    struct ForwardPlusTile
    {
        std::uint32_t OpaqueOffset = 0, OpaqueCount = 0;
        std::uint32_t TransparentOffset = 0, TransparentCount = 0;
    };
    static_assert(sizeof(ForwardPlusTile) == 16, "Forward+ tile header ABI changed");
    // Fixed-capacity index slices give deterministic, disjoint GPU writes. Count
    // is the TOTAL accepted count; only min(count, capacity) indices are stored.
    // When count > capacity, the lighting consumer MUST evaluate all scene lights.
    // Truncating that loop silently drops light and is not a valid overflow policy.
    [[nodiscard]] bool BuildForwardPlusGrid(std::uint32_t width, std::uint32_t height,
        std::uint32_t samples, std::size_t lightCount, const ForwardPlusParameters& parameters,
        ForwardPlusGrid& result, std::string& error);
}
