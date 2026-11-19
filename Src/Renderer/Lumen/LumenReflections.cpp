#include "Renderer/Lumen/LumenReflections.h"
#include "Maths/Matrix4.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    bool BuildLumenReflectionRayFrame(const LumenScreenProbePlacement& screen,const LumenProbeRaySet& sampling,
        const LumenReflectionRays& settings,LumenReflectionRayFrame& output,std::string& error)
    {
        constexpr unsigned MaximumViewport=16384,MaximumDivisor=64;
        constexpr float MinimumHomogeneous=1e-7f,MinimumNormalSquared=1e-12f,MinimumGGXAlpha=0.001f;
        if (!screen.Enabled || screen.Adaptive || !screen.Width || !screen.Height || screen.Width>MaximumViewport || screen.Height>MaximumViewport ||
            !settings.Enabled || (settings.ResolutionDivisor!=1 && settings.ResolutionDivisor!=2 && settings.ResolutionDivisor!=4) ||
            !std::isfinite(settings.MaximumRoughness) || settings.MaximumRoughness<=0 || settings.MaximumRoughness>1 ||
            !std::isfinite(screen.NormalBias) || screen.NormalBias<0 || screen.MinimumDistance!=0 ||
            !sampling.Origins.empty() || sampling.DirectionalSide || sampling.DepthLayers!=1 || !std::isfinite(sampling.MaximumDistance) || sampling.MaximumDistance<=0)
        { error="Lumen reflection rays require a finite screen camera, exclusive GPU sampling and bounded roughness/resolution"; return false; }
        for (float value:screen.ViewProjection.Values) if (!std::isfinite(value)) { error="Nonfinite Lumen reflection camera"; return false; }
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(screen.ViewProjection.Values).Inverse(inverse)) { error="Singular Lumen reflection camera"; return false; }
        LumenReflectionRayFrame frame;
        std::copy(inverse.Data(),inverse.Data()+16,frame.Inverse.Values.begin());
        for (float value:frame.Inverse.Values) if (!std::isfinite(value)) { error="Nonfinite Lumen reflection inverse"; return false; }
        auto divisor=settings.ResolutionDivisor; unsigned columns=0,rows=0;
        for (;;divisor*=2)
        {
            if (divisor>MaximumDivisor) { error="Lumen reflection resolution cannot fit its ray budget"; return false; }
            columns=(screen.Width+divisor-1)/divisor; rows=(screen.Height+divisor-1)/divisor;
            if (std::uint64_t(columns)*rows<=LumenReflectionMaximumRays) break;
        }
        const float coordinateLimit=std::numeric_limits<float>::max()/4-sampling.MaximumDistance-screen.NormalBias;
        if (!std::isfinite(coordinateLimit) || coordinateLimit<=0) { error="Lumen reflection interval exceeds representable coordinates"; return false; }
        frame.Viewport={screen.Width,screen.Height,divisor,columns};
        frame.Work={rows,columns*rows,sampling.Sequence,LumenDistanceTraceThreads*LumenDistanceTraceColumns};
        frame.Trace={sampling.MaximumDistance,screen.NormalBias,settings.MaximumRoughness,MinimumGGXAlpha};
        frame.Numeric={coordinateLimit,MinimumHomogeneous,MinimumNormalSquared,0}; output=frame; error.clear(); return true;
    }
    bool BuildLumenReflectionResolveFrame(const LumenReflectionRayFrame& rays,const LumenReflectionFilter& filter,
        Float3 sky,Techniques::ShadingModel shading,LumenReflectionResolveFrame& output,std::string& error,const LumenEnvironmentLobe& environment)
    {
        constexpr unsigned MaximumRadius=2,MaximumPixels=8*1024*1024,MaximumViewport=16384,MaximumDivisor=64;
        constexpr float MinimumCosine=1e-5f,HalfMaximum=65504,DielectricF0=0.04f;
        const auto unit=[](float x) { return std::isfinite(x) && x>=0 && x<=1; };
        const auto nonnegative=[](float x) { return std::isfinite(x) && x>=0; };
        const auto width=rays.Viewport.X,height=rays.Viewport.Y,divisor=rays.Viewport.Z;
        if (!width || !height || width>MaximumViewport || height>MaximumViewport || std::uint64_t(width)*height>MaximumPixels ||
            !divisor || divisor>MaximumDivisor || (divisor&(divisor-1)) || rays.Viewport.W!=(width+divisor-1)/divisor ||
            rays.Work.X!=(height+divisor-1)/divisor || rays.Work.Y!=rays.Viewport.W*rays.Work.X || rays.Work.Y>LumenReflectionMaximumRays ||
            filter.Radius>MaximumRadius || !unit(filter.NormalCosine) || !nonnegative(filter.PlaneTolerance) ||
            !unit(filter.RelativePlaneTolerance) || !unit(filter.RoughnessTolerance) ||
            !nonnegative(sky.X) || !nonnegative(sky.Y) || !nonnegative(sky.Z) || sky.X>HalfMaximum || sky.Y>HalfMaximum || sky.Z>HalfMaximum ||
            !std::isfinite(rays.Trace.X) || rays.Trace.X<=0 || !nonnegative(rays.Trace.Y) ||
            !std::isfinite(rays.Trace.Z) || rays.Trace.Z<=0 || rays.Trace.Z>1 ||
            !std::isfinite(rays.Trace.W) || rays.Trace.W<=0 || rays.Trace.W>1 ||
            !std::isfinite(rays.Numeric.X) || rays.Numeric.X<=0 || !std::isfinite(rays.Numeric.Y) || rays.Numeric.Y<=0 ||
            !std::isfinite(rays.Numeric.Z) || rays.Numeric.Z<=0 || unsigned(shading)>unsigned(Techniques::ShadingModel::Toon))
        { error="Lumen reflection reconstruction requires bounded rays, material filtering and finite nonnegative sky"; return false; }
        for (float value:rays.Inverse.Values) if (!std::isfinite(value))
        { error="Lumen reflection reconstruction has a nonfinite inverse camera"; return false; }
        Maths::Matrix4 camera;
        if (!Maths::Matrix4(rays.Inverse.Values).Inverse(camera))
        { error="Lumen reflection reconstruction has a singular inverse camera"; return false; }
        LumenReflectionResolveFrame result;
        if (!BuildLumenEnvironmentFrame(environment,result.Environment,error)) return false;
        result.Rays=rays;
        result.Acceptance={filter.NormalCosine,filter.PlaneTolerance,filter.RelativePlaneTolerance,filter.RoughnessTolerance};
        result.Filter={float(filter.Radius),MinimumCosine,HalfMaximum,DielectricF0}; result.Sky={sky.X,sky.Y,sky.Z,0};
        result.Dispatch={unsigned(shading),LumenDistanceTraceThreads*LumenDistanceTraceColumns,width*height,0};
        output=result; error.clear(); return true;
    }
}
