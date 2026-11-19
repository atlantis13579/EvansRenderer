#include "Renderer/Lumen/LumenProbeIntegration.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace Evans::Renderer
{
    // Ramamoorthi and Hanrahan, "An Efficient Representation for Irradiance
    // Environment Maps", SIGGRAPH 2001: https://graphics.stanford.edu/papers/envmap/
    // Project incoming radiance with equal solid-angle weights, then convolve
    // SH bands 0/1/2 by pi, 2*pi/3, pi/4. BRDF albedo/pi belongs to the eventual
    // receiver, not this irradiance cache. Truncation can ring; spatial gathering
    // must not mistake these nine coefficients for directional visibility.
    namespace
    {
        std::atomic<std::uint64_t> NextProbe{1};
        bool Finite(Float4 v) { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z) && std::isfinite(v.W); }
        bool Near(float a,double b) { return std::abs(double(a)-b)<=0.0001*std::max(1.0,std::abs(b)); }
    }
    bool BuildLumenProbeIntegrationFrame(const LumenRadianceInfo& source,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationFrame& output,std::string& error)
    {
        // Directional caches keep their angular atlas and distance provenance.
        // The existing SH projection kernel consumes rotated Hammersley rays.
        if (source.Probes.DirectionalSide) { error="Directional world caches require directional reconstruction, not Hammersley SH integration"; return false; }
        const auto& id=source.Identity; const auto& sky=upload.SkyRadiance;
        if (!id.Lighting || !id.Surface.Epoch || !id.Surface.Serial || !id.Surface.Geometry || !id.Surface.Material || !id.Occlusion.Serial ||
            id.Surface.Epoch!=id.Occlusion.Epoch || id.Surface.Geometry!=id.Occlusion.Geometry ||
            !(source.Screen.Grid.Y?ValidateLumenProbeSampling(LumenRadianceProbeCount(source),source.Probes.RaysPerProbe,source.Probes.MaximumDistance,error):ValidateLumenProbeRaySet(source.Probes,error)) ||
            source.RayCount!=LumenRadianceProbeCount(source)*source.Probes.RaysPerProbe || (source.Hybrid && !source.Screen.Grid.Y))
        { error="Lumen irradiance needs a complete generated-probe ray source and matching scene identities"; return false; }
        if (!std::isfinite(sky.X) || !std::isfinite(sky.Y) || !std::isfinite(sky.Z) || sky.X<0 || sky.Y<0 || sky.Z<0)
        { error="Lumen probe sky radiance must be finite and nonnegative"; return false; }
        constexpr float Pi=3.14159265358979323846f; LumenProbeIntegrationFrame result;
        if (!BuildLumenEnvironmentFrame(upload.Environment,result.Environment,error)) return false;
        result.Counts={LumenRadianceProbeCount(source),source.Probes.RaysPerProbe,source.Probes.Sequence,LumenProbeThreads*LumenProbeColumns};
        result.SkySolidAngle={sky.X,sky.Y,sky.Z,4*Pi/source.Probes.RaysPerProbe}; output=result; error.clear(); return true;
    }
    bool ValidateLumenProbeIrradiance(const LumenProbeInfo& info,const std::vector<LumenProbeIrradiance>& probes,std::string& error)
    {
        LumenProbeIntegrationFrame frame;
        if (!BuildLumenProbeIntegrationFrame(info.Source,{info.SkyRadiance,false,info.Environment},frame,error) || probes.size()!=LumenRadianceProbeCount(info.Source))
        { error="Lumen irradiance feedback does not match its source probes"; return false; }
        for (unsigned index=0;index<probes.size();++index)
        {
            const auto& probe=probes[index]; const auto& t=probe.TraceCounts; const auto& c=probe.CacheCounts; const auto& m=probe.Metadata; const auto& w=probe.Coverage;
            const auto rays=info.Source.Probes.RaysPerProbe; const auto status=LumenProbeStatus(m.Y);
            if (m.X!=(info.Source.Screen.Grid.Y?index:info.Source.Probes.Origins[index].Metadata.X) || m.Z!=rays || m.W!=info.Source.Probes.Sequence || m.Y<1 || m.Y>3 ||
                std::uint64_t(t.X)+t.Y+t.Z+t.W!=rays || (!info.Source.WorldCache && t.W) || std::uint64_t(c.X)+c.Y+c.Z+c.W!=std::uint64_t(t.X)+t.W || !Finite(w) ||
                w.X<0 || w.X>1 || w.Y<0 || w.Y>1 || w.Z<0 || w.Z>1 || w.W<0 ||
                (w.W>info.Source.Probes.MaximumDistance && !Near(w.W,info.Source.Probes.MaximumDistance)))
            {
                std::ostringstream detail; detail<<std::setprecision(9)<<"Lumen probe feedback lost identity, accounting or finite coverage at "<<index
                    <<"; metadata "<<m.X<<','<<m.Y<<','<<m.Z<<','<<m.W<<"; trace "<<t.X<<','<<t.Y<<','<<t.Z<<','<<t.W
                    <<"; cache "<<c.X<<','<<c.Y<<','<<c.Z<<','<<c.W<<"; coverage "<<w.X<<','<<w.Y<<','<<w.Z<<','<<w.W
                    <<"; cascades "<<info.Source.WorldCacheCascades;
                error=detail.str(); return false;
            }
            for (const auto& coefficient:probe.Coefficients)
                if (!Finite(coefficient) || coefficient.W || (status==LumenProbeStatus::Invalid && (coefficient.X || coefficient.Y || coefficient.Z)))
                { error="Lumen probe coefficients are nonfinite or invalid irradiance remained usable"; return false; }
            if (status==LumenProbeStatus::Invalid)
            { if (w.X || w.Y || w.Z || w.W) { error="Invalid Lumen probe retained usable confidence"; return false; } }
            else if (c.W || (status==LumenProbeStatus::Complete)!=(t.Z==0 && c.Y==0 && c.Z==0) ||
                !Near(w.X,double(t.Y+c.X)/rays) || !Near(w.Z,double(t.Y)/rays) || (!(t.X+t.W) && w.Y) || (!t.X && w.W))
            { error="Lumen probe confidence disagrees with unresolved transport"; return false; }
        }
        error.clear(); return true;
    }
    LumenProbeHandle NewLumenProbeHandle() noexcept
    {
        auto value=NextProbe.load(std::memory_order_relaxed);
        while (value && value!=UINT64_MAX) if (NextProbe.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return {value};
        return {};
    }
}
