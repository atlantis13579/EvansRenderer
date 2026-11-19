#include "Renderer/Lumen/LumenProbeGather.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    bool BuildLumenProbeGatherFrame(const LumenProbeInfo& source,const LumenDistanceSceneInfo& scene,const LumenProbeGatherSettings& settings,
        Techniques::ShadingModel shading,LumenProbeGatherFrame& output,std::string& error)
    {
        const auto& screen=source.Source.Screen; const auto& viewport=screen.Viewport; const auto& grid=screen.Grid;
        LumenProbeIntegrationFrame integration;
        if (!source.Source.Probes.Origins.empty() || !BuildLumenProbeIntegrationFrame(source.Source,{source.SkyRadiance,false,source.Environment},integration,error) ||
            !viewport.X || !viewport.Y || !viewport.Z || !viewport.W || !grid.X || !grid.Y || !ValidLumenScreenProbeLayout(grid) ||
            viewport.W!=(std::uint64_t(viewport.X)+viewport.Z-1)/viewport.Z || grid.X!=(std::uint64_t(viewport.Y)+viewport.Z-1)/viewport.Z ||
            std::uint64_t(viewport.W)*grid.X*(grid.W+1)!=grid.Y)
        { error="Lumen screen gather requires a complete native screen-probe grid"; return false; }
        if (!std::isfinite(settings.Intensity) || settings.Intensity<0 || !std::isfinite(settings.MinimumNormalCosine) ||
            settings.MinimumNormalCosine<0 || settings.MinimumNormalCosine>=1 || !std::isfinite(settings.PlaneTolerance) || settings.PlaneTolerance<0 ||
            !std::isfinite(settings.RelativePlaneTolerance) || settings.RelativePlaneTolerance<0 || settings.RelativePlaneTolerance>1)
        { error="Lumen gather requires finite nonnegative energy and bounded surface tolerances"; return false; }
        if (unsigned(shading)>unsigned(Techniques::ShadingModel::Toon)) { error="Unknown Lumen receiver shading model"; return false; }
        for (auto value:screen.InverseViewProjection.Values) if (!std::isfinite(value)) { error="Nonfinite Lumen gather camera"; return false; }
        const auto& expected=source.Source.Identity.Occlusion;
        if (expected.Epoch!=scene.Identity.Epoch || expected.Serial!=scene.Identity.Serial || expected.Geometry!=scene.Identity.Geometry ||
            !settings.Visibility.MaximumSteps || settings.Visibility.MaximumSteps>512 || !std::isfinite(settings.Visibility.NormalBias) ||
            settings.Visibility.NormalBias<0 || settings.Visibility.NormalBias>1000 ||
            std::uint64_t(viewport.X)*viewport.Y>LumenProbeVisibilityMaximumBytes/(sizeof(std::uint32_t)*LumenProbeVisibilityWords(screen)))
        { error="Lumen probe visibility requires the exact occluder version, bounded work and finite normal bias"; return false; }
        // Visibility links use all GI surfaces, independently of direct-light
        // shadow-caster flags. Unknown SDF intervals cannot certify a clear link.
        LumenDistanceTraceUpload trace; trace.Rays={{{0,0,0,0},{0,0,1,1}}}; trace.MaximumSteps=settings.Visibility.MaximumSteps;
        LumenDistanceTraceFrame traceFrame; if (!BuildLumenDistanceTraceFrame(scene,trace,traceFrame,error)) return false;
        traceFrame.Counts.X=viewport.X*viewport.Y;
        constexpr float MinimumNormalSquared=1e-12f,MinimumWeight=1e-6f,HalfFloatMaximum=65504,DielectricReflectance=0.04f;
        LumenProbeGatherFrame result; result.Trace=traceFrame; result.Screen=screen;
        result.Appearance={settings.Intensity,settings.MinimumNormalCosine,settings.PlaneTolerance,settings.RelativePlaneTolerance};
        result.Numeric={MinimumNormalSquared,MinimumWeight,HalfFloatMaximum,DielectricReflectance}; result.Model={unsigned(shading),settings.Visibility.Enabled?1u:0u,0,0};
        result.Visibility={std::max(screen.Parameters.X,settings.Visibility.NormalBias),32*std::numeric_limits<float>::epsilon(),0,0};
        output=result; error.clear(); return true;
    }
}
