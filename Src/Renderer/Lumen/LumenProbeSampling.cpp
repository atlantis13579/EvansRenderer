#include "Renderer/Lumen/LumenProbeSampling.h"
#include "Maths/Matrix4.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace Evans::Renderer
{
    bool ValidateLumenProbeSampling(unsigned count,unsigned rays,float horizon,std::string& error,unsigned layers)
    {
        if (!count || count>LumenMaximumProbes || rays<LumenMinimumProbeRays || rays>LumenMaximumProbeRays || (rays&(rays-1)) ||
            !layers || layers>LumenMaximumWorldDepthLayers || std::uint64_t(count)*rays*layers>LumenDistanceMaximumRays || !std::isfinite(horizon) || horizon<=0)
        { error="Lumen probes need bounded positions, power-of-two ray counts and a positive finite horizon"; return false; }
        error.clear(); return true;
    }
    bool BuildLumenScreenProbeFrame(const LumenScreenProbePlacement& placement,const LumenProbeRaySet& sampling,
        LumenScreenProbeFrame& output,std::string& error)
    {
        // Wright et al., "Lumen", SIGGRAPH 2022: screen-space radiance caching.
        // Reserve a uniform anchor followed by breadth-first quadtree slots.
        // GPU placement adds an anchor only for a receiver plane its ancestors
        // cannot represent. Stable slots avoid CPU readback and cross-tile atomics;
        // inactive slots retain invalid source metadata through trace/integration.
        // This bounded hierarchy coarsens the whole grid if its worst-case
        // ray/storage budget cannot fit. It never silently drops edge tiles.
        constexpr unsigned MaximumDimension=16384,MinimumTile=4,MaximumTile=256;
        constexpr float HomogeneousMinimum=1e-7f;
        if (!placement.Enabled || !placement.Width || !placement.Height || placement.Width>MaximumDimension || placement.Height>MaximumDimension ||
            placement.TileSide<MinimumTile || placement.TileSide>MaximumTile || (placement.TileSide&(placement.TileSide-1)) || !sampling.Origins.empty() || sampling.DirectionalSide || sampling.DepthLayers!=1 ||
            placement.AdaptiveLevels<1 || placement.AdaptiveLevels>Techniques::MaximumLumenAdaptiveProbeLevels ||
            !std::isfinite(placement.MinimumNormalCosine) || placement.MinimumNormalCosine<0 || placement.MinimumNormalCosine>=1 ||
            !std::isfinite(placement.PlaneTolerance) || placement.PlaneTolerance<0 || !std::isfinite(placement.RelativePlaneTolerance) ||
            placement.RelativePlaneTolerance<0 || placement.RelativePlaneTolerance>1 ||
            !std::isfinite(placement.NormalBias) || placement.NormalBias<0 || !std::isfinite(placement.MinimumDistance) ||
            placement.MinimumDistance<0 || placement.MinimumDistance>=sampling.MaximumDistance || !ValidateLumenProbeSampling(1,sampling.RaysPerProbe,sampling.MaximumDistance,error))
        { error="Lumen screen probes require a bounded viewport, power-of-two tiles and finite ray offsets"; return false; }
        Maths::Matrix4 inverse;
        for (auto value:placement.ViewProjection.Values) if (!std::isfinite(value)) { error="Nonfinite Lumen screen camera"; return false; }
        if (!Maths::Matrix4(placement.ViewProjection.Values).Inverse(inverse)) { error="Singular Lumen screen camera"; return false; }
        LumenScreenProbeFrame result;
        std::copy(inverse.Data(),inverse.Data()+16,result.InverseViewProjection.Values.begin());
        for (auto value:result.InverseViewProjection.Values) if (!std::isfinite(value)) { error="Nonfinite inverse Lumen screen camera"; return false; }
        auto tile=placement.TileSide; unsigned columns=0,rows=0;
        const auto levels=placement.Adaptive?placement.AdaptiveLevels:0;
        const auto extraSlots=LumenScreenProbeSlots(levels)-1;
        const auto budget=std::min<std::size_t>(LumenMaximumProbes,LumenDistanceMaximumRays/sampling.RaysPerProbe)/(1+extraSlots);
        do { columns=(placement.Width+tile-1)/tile; rows=(placement.Height+tile-1)/tile; if (std::uint64_t(columns)*rows<=budget) break; tile*=2; } while (tile<=MaximumTile);
        if (tile>MaximumTile) { error="Lumen screen probe budget cannot cover this viewport"; return false; }
        const float coordinateLimit=std::numeric_limits<float>::max()/4-sampling.MaximumDistance-placement.NormalBias;
        if (!std::isfinite(coordinateLimit) || coordinateLimit<=0) { error="Lumen screen ray horizon exceeds representable coordinates"; return false; }
        result.Viewport={placement.Width,placement.Height,tile,columns}; result.Grid={rows,columns*rows*(1+extraSlots),levels,extraSlots};
        result.Parameters={placement.NormalBias,placement.MinimumDistance,coordinateLimit,HomogeneousMinimum};
        result.Adaptive={placement.MinimumNormalCosine,placement.PlaneTolerance,placement.RelativePlaneTolerance,32*std::numeric_limits<float>::epsilon()};
        output=result; error.clear(); return true;
    }
    bool ValidateLumenProbeRaySet(const LumenProbeRaySet& probes,std::string& error)
    {
        if (probes.Origins.size()>LumenMaximumProbes || !ValidateLumenProbeSampling(unsigned(probes.Origins.size()),probes.RaysPerProbe,probes.MaximumDistance,error,probes.DepthLayers)) return false;
        if (probes.DepthLayers!=1 && !probes.DirectionalSide) { error="Multiple depth layers require a fixed directional world atlas"; return false; }
        if (probes.DirectionalSide && (probes.DirectionalSide<4 || probes.DirectionalSide>32 ||
            (probes.DirectionalSide&(probes.DirectionalSide-1)) || probes.DirectionalSide*probes.DirectionalSide!=probes.RaysPerProbe || probes.Sequence))
        { error="Directional world probes require a fixed 4..32 square atlas and zero sequence"; return false; }
        std::unordered_set<unsigned> ids;
        constexpr double CoordinateLimit=double(std::numeric_limits<float>::max())/4;
        for (const auto& origin:probes.Origins)
        {
            const auto& p=origin.PositionMinimum; const auto& id=origin.Metadata;
            if (!std::isfinite(p.X) || !std::isfinite(p.Y) || !std::isfinite(p.Z) || !std::isfinite(p.W) || p.W<0 || p.W>=probes.MaximumDistance ||
                double(std::max({std::abs(p.X),std::abs(p.Y),std::abs(p.Z)}))+probes.MaximumDistance>CoordinateLimit ||
                id.X==LumenDistanceInvalid || id.Y || id.Z || id.W || !ids.insert(id.X).second)
            { error="Lumen probe origins need unique identifiers and representable world trace intervals"; return false; }
        }
        error.clear(); return true;
    }
}
