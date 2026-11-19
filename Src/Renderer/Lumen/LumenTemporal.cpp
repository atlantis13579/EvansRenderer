#include "Renderer/Lumen/LumenTemporal.h"

#include <algorithm>
#include <cmath>

namespace Evans::Renderer
{
    bool BuildLumenTemporalFrame(const GlobalIlluminationDesc& camera,const Techniques::GlobalIlluminationSettings& settings,
        int width,int height,bool historyValid,LumenTemporalFrame& output,std::string& error)
    {
        if (width<=0 || height<=0 || std::uint64_t(width)*height>LumenTemporalMaximumBytes/sizeof(LumenTemporalPixel) ||
            !FrameHistory::IsFinite(camera.ViewProjection) || !FrameHistory::IsFinite(camera.PreviousViewProjection))
        { error="Lumen temporal reconstruction requires finite cameras and bounded history storage"; return false; }
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(camera.ViewProjection.Values).Inverse(inverse))
        { error="Lumen temporal reconstruction requires an invertible camera"; return false; }
        MotionBlurDesc previous; previous.ViewProjection=camera.PreviousViewProjection; MotionBlurConstants geometry;
        if (!BuildMotionBlurConstants(previous,{},width,height,geometry))
        { error="Lumen temporal reconstruction requires a finite previous optical-depth plane"; return false; }
        const auto& temporal=settings.Lumen.Temporal;
        const auto finite=[](float value) { return std::isfinite(value); };
        if (!finite(settings.Intensity) || settings.Intensity<0 || !finite(temporal.HistoryWeight) || temporal.HistoryWeight<0 || temporal.HistoryWeight>0.98f ||
            !finite(temporal.DepthTolerance) || temporal.DepthTolerance<0 || temporal.DepthTolerance>1 ||
            !finite(temporal.NormalThreshold) || temporal.NormalThreshold<0 || temporal.NormalThreshold>1 ||
            !finite(temporal.PlaneTolerance) || temporal.PlaneTolerance<0 || !finite(temporal.ClipTolerance) || temporal.ClipTolerance<0 || temporal.ClipTolerance>1 ||
            !temporal.MaximumFrames || temporal.MaximumFrames>64)
        { error="Lumen temporal parameters must be finite and bounded"; return false; }
        LumenTemporalFrame result{}; std::copy(inverse.Data(),inverse.Data()+16,result.InverseViewProjection.Values.begin());
        result.PreviousInverseViewProjection=geometry.InverseViewProjection; result.PreviousDepthPlane=geometry.DepthPlane;
        result.Extent={unsigned(width),unsigned(height),unsigned(width)*unsigned(height),LumenTemporalThreads*LumenTemporalColumns};
        result.Temporal={temporal.HistoryWeight,temporal.DepthTolerance,temporal.NormalThreshold,temporal.PlaneTolerance};
        result.Clip={temporal.ClipTolerance,float(temporal.MaximumFrames),0,0};
        result.Jitter={camera.Sampling.JitterPixels.X-camera.Sampling.PreviousJitterPixels.X,
            camera.Sampling.JitterPixels.Y-camera.Sampling.PreviousJitterPixels.Y,settings.Intensity,historyValid && camera.Sampling.HistoryValid?1.0f:0.0f};
        constexpr float AbsoluteDepthTolerance=0.001f,MinimumClipExtent=0.001f,HomogeneousFloor=1e-7f,HalfMaximum=65504;
        result.Numeric={AbsoluteDepthTolerance,MinimumClipExtent,HomogeneousFloor,HalfMaximum};
        if (!FrameHistory::IsFinite(result.InverseViewProjection) || !finite(result.Jitter.X) || !finite(result.Jitter.Y))
        { error="Lumen temporal camera reconstruction or jitter delta overflowed"; return false; }
        output=result; error.clear(); return true;
    }
    bool BuildLumenReflectionTemporalFrame(const ReflectionDesc& camera,const Techniques::ReflectionSettings& settings,
        int width,int height,bool historyValid,LumenTemporalFrame& output,std::string& error)
    {
        Techniques::GlobalIlluminationSettings temporal;
        temporal.Intensity=settings.Intensity; temporal.Lumen.Temporal=settings.Lumen.Temporal;
        const GlobalIlluminationDesc view{camera.ViewProjection,camera.PreviousViewProjection,camera.Sampling};
        return BuildLumenTemporalFrame(view,temporal,width,height,historyValid && settings.Lumen.Temporal.Enabled,output,error);
    }
}
