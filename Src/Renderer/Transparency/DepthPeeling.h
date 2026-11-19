#pragma once

#include "Renderer/Types.h"
#include "Techniques/OrderIndependentTransparency.h"

namespace Evans::Renderer
{
    inline constexpr float PeelingClearDepth = 1;

    // Capture uses two D32 images. A layer is shaded with the same raster
    // positions and depth EQUAL into transparent-black RGBA using source-over.
    // Accumulation must also check capture depth: empty D32=1 pixels can admit
    // far-plane fragments during that replay and their color must be ignored.
    enum class DepthPeelingStage { Idle, NeedCapture, Capturing, NeedShading, Shading, NeedAccumulation, NeedComposition, Complete };
    enum class DepthPeelingEvent { BeginCapture, EndCapture, BeginShading, EndShading, Accumulate, Compose };
    struct DepthPeelingStep
    {
        std::uint32_t Layer = 0, DepthWrite = 0, DepthRead = 1, AccumulationRead = 0, AccumulationWrite = 1;
        bool FirstLayer = true, OverflowProbe = false;
    };
    class DepthPeelingSequence
    {
    public:
        [[nodiscard]] bool Begin(std::uint32_t maximumLayers, std::string& error);
        [[nodiscard]] bool Advance(DepthPeelingEvent event, std::string& error);
        [[nodiscard]] DepthPeelingStep Step() const noexcept;
        [[nodiscard]] DepthPeelingStage Stage() const noexcept { return Current; }
        void Abort() noexcept;

    private:
        DepthPeelingStage Current = DepthPeelingStage::Idle;
        std::uint32_t MaximumLayers = 0, Layer = 0;
    };
}
