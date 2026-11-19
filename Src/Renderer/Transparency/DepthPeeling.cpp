#include "Renderer/Transparency/DepthPeeling.h"

namespace Evans::Renderer
{
    namespace
    {
        bool LayerBudget(std::uint32_t value)
        { return value >= Techniques::MinimumPeelingLayers && value <= Techniques::MaximumPeelingLayers; }
    }

    bool DepthPeelingSequence::Begin(std::uint32_t maximumLayers,std::string& error)
    {
        error.clear();
        if (!LayerBudget(maximumLayers)) { error = "Invalid depth-peeling layer budget"; return false; }
        if (Current != DepthPeelingStage::Idle && Current != DepthPeelingStage::Complete)
        { error = "An active depth-peeling sequence must complete or abort before restarting"; return false; }
        MaximumLayers = maximumLayers; Layer = 0; Current = DepthPeelingStage::NeedCapture; return true;
    }

    bool DepthPeelingSequence::Advance(DepthPeelingEvent event,std::string& error)
    {
        error.clear();
        using S = DepthPeelingStage; using E = DepthPeelingEvent;
        if (event == E::BeginCapture && Current == S::NeedCapture) Current = S::Capturing;
        else if (event == E::EndCapture && Current == S::Capturing)
            Current = Layer == MaximumLayers ? S::NeedComposition : S::NeedShading;
        else if (event == E::BeginShading && Current == S::NeedShading) Current = S::Shading;
        else if (event == E::EndShading && Current == S::Shading) Current = S::NeedAccumulation;
        else if (event == E::Accumulate && Current == S::NeedAccumulation) { ++Layer; Current = S::NeedCapture; }
        else if (event == E::Compose && Current == S::NeedComposition) Current = S::Complete;
        else { error = "Depth-peeling capture, shading, accumulation, overflow probe and composition are out of order"; return false; }
        return true;
    }

    DepthPeelingStep DepthPeelingSequence::Step() const noexcept
    {
        DepthPeelingStep result; result.Layer = Layer; result.FirstLayer = Layer == 0;
        result.OverflowProbe = MaximumLayers != 0 && Layer == MaximumLayers;
        result.DepthWrite = result.AccumulationRead = Layer%2;
        result.DepthRead = result.AccumulationWrite = 1-result.DepthWrite; return result;
    }

    void DepthPeelingSequence::Abort() noexcept
    { Current = DepthPeelingStage::Idle; Layer = MaximumLayers = 0; }
}
