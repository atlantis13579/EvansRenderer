#include "Renderer/Lumen/LumenTranslucency.h"
#include "Maths/Matrix4.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Evans::Renderer
{
    bool BuildLumenTranslucencyFrame(const Matrix4& camera,unsigned width,unsigned height,
        const Techniques::GlobalIlluminationSettings& settings,LumenTranslucencyFrame& output,
        LumenRadianceTraceUpload& trace,std::string& error)
    {
        constexpr unsigned MaximumViewport=16384,MaximumAtlasDimension=2048;
        constexpr float Pi=3.14159265358979323846f,MinimumW=1e-7f,MinimumNormalSquared=1e-12f;
        if (!width || !height || width>MaximumViewport || height>MaximumViewport ||
            !std::isfinite(settings.MaximumDistance) || settings.MaximumDistance<=0 ||
            !std::isfinite(settings.Intensity) || settings.Intensity<0)
        { error="Lumen translucency requires a bounded viewport and finite positive horizon"; return false; }
        for (float value:camera.Values) if (!std::isfinite(value))
        { error="Nonfinite Lumen translucency camera"; return false; }
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(camera.Values).Inverse(inverse))
        { error="Singular Lumen translucency camera"; return false; }
        for (unsigned i=0;i<16;++i) if (!std::isfinite(inverse.Data()[i]))
        { error="Nonfinite inverse Lumen translucency camera"; return false; }
        const auto sanitized=Techniques::SanitizeLumenSettings(settings.Lumen);
        const auto& volume=sanitized.Translucency;
        auto tile=volume.TileSide; unsigned columns=0,rows=0;
        do
        {
            columns=(width+tile-1)/tile; rows=(height+tile-1)/tile;
            if (std::uint64_t(columns)*rows*volume.SliceCount<=LumenTranslucencyMaximumProbes &&
                columns*LumenProbeCoefficients<=MaximumAtlasDimension && rows*volume.SliceCount<=MaximumAtlasDimension) break;
            tile*=2;
        } while (tile<=MaximumViewport);
        LumenTranslucencyFrame frame; frame.ViewProjection=camera;
        std::copy(inverse.Data(),inverse.Data()+16,frame.Inverse.Values.begin());
        frame.Grid={columns,rows,volume.SliceCount,LumenTranslucencyActive|(volume.VisibilitySupport?LumenTranslucencyVisibilitySupport:0)};
        frame.Distribution={settings.MaximumDistance,volume.DepthExponent,settings.Intensity,1/(4*Pi)};
        frame.Atlas={float(columns*LumenProbeCoefficients),float(rows*volume.SliceCount),MinimumW,MinimumNormalSquared};
        LumenRadianceTraceUpload result;
        result.Probes.RaysPerProbe=volume.RaysPerProbe; result.Probes.MaximumDistance=settings.MaximumDistance;
        result.Trace.MaximumSteps=settings.MaximumSteps;
        result.Probes.Origins.reserve(std::size_t(columns)*rows*volume.SliceCount);
        // Cell centers are placed on the CPU; angular ray generation, transport,
        // integration and packing remain native GPU work, without a readback.
        for (unsigned z=0;z<volume.SliceCount;++z) for (unsigned y=0;y<rows;++y) for (unsigned x=0;x<columns;++x)
        {
            const float nx=2*(float(x)+0.5f)/float(columns)-1,ny=1-2*(float(y)+0.5f)/float(rows);
            const auto nearH=inverse*Maths::Vector4(nx,ny,0,1),farH=inverse*Maths::Vector4(nx,ny,1,1);
            if (std::abs(nearH.w)<=MinimumW || std::abs(farH.w)<=MinimumW)
            { error="Lumen translucency requires finite near and far camera planes"; return false; }
            const auto nearPoint=nearH.XYZ()/nearH.w,farPoint=farH.XYZ()/farH.w;
            const auto axis=farPoint-nearPoint; const float length=axis.Length();
            if (!std::isfinite(length) || length<=MinimumW)
            { error="Degenerate Lumen translucency frustum ray"; return false; }
            const float fraction=std::pow((float(z)+0.5f)/float(volume.SliceCount),volume.DepthExponent);
            const auto point=nearPoint+axis*(fraction*std::min(settings.MaximumDistance,length)/length);
            LumenProbeOrigin origin; origin.PositionMinimum={point.x,point.y,point.z,0};
            origin.Metadata={unsigned(result.Probes.Origins.size()),0,0,0}; result.Probes.Origins.push_back(origin);
        }
        if (!ValidateLumenProbeRaySet(result.Probes,error)) return false;
        output=frame; trace=std::move(result); error.clear(); return true;
    }
    bool BuildLumenTranslucencySupportFrame(const LumenDistanceSceneInfo& scene,const LumenRadianceTraceUpload& upload,
        LumenDistanceTraceFrame& output,std::string& error)
    {
        if (upload.Screen.Enabled || upload.Reflections.Enabled || upload.Hybrid.Enabled || !upload.Trace.Rays.empty() ||
            upload.Probes.DirectionalSide || upload.Probes.DepthLayers!=1 || upload.Probes.Origins.size()>LumenTranslucencyMaximumProbes ||
            !ValidateLumenProbeRaySet(upload.Probes,error))
        { error="Translucent visibility support requires bounded world-volume probes"; return false; }
        for (const auto& probe:upload.Probes.Origins) if (probe.PositionMinimum.W!=0)
        { error="Translucent support cannot certify transport that skips a probe's origin interval"; return false; }
        auto trace=upload.Trace; trace.Rays={{{0,0,0,0},{0,0,1,1}}}; LumenDistanceTraceFrame frame;
        if (!BuildLumenDistanceTraceFrame(scene,trace,frame,error)) return false;
        constexpr float CoordinateRoundoff=32*std::numeric_limits<float>::epsilon();
        frame.Counts.X=unsigned(upload.Probes.Origins.size());
        frame.Parameters={upload.Trace.HitToleranceCells,CoordinateRoundoff,upload.Probes.MaximumDistance,0};
        output=frame; error.clear(); return true;
    }
}
