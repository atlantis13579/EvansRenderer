#pragma once

#include "Renderer/Lumen/LumenProbeIntegration.h"
#include <algorithm>

namespace Evans::Renderer
{
    // A translucent caster can carry a blend flag in a depth-only shadow pass.
    // Only main-camera color draws establish the transparency ordering barrier.
    inline bool LumenHasTransparentMainDraws(const FrameCapture& frame)
    {
        for (const auto& pass:frame.Passes)
        {
            if (pass.Description.Type!=PassType::Color || pass.Description.Target) continue;
            for (const auto& draw:pass.DrawCalls) if (draw.Pipeline.BlendEnabled) return true;
        }
        return false;
    }

    // Same-frame consumers may use pending handles, but readback stays outside
    // the frame. Only a successful present publishes them to later frames.
    // Native backends separately retain command resources until GPU completion;
    // cancelling logical publication must never free submitted GPU work.
    class LumenFrameTransaction
    {
    public:
        static constexpr unsigned MaximumChains=64;
        static constexpr std::size_t MaximumFrameRays=LumenDistanceMaximumRays*2; // Opaque diffuse, specular and bounded translucency.
        static constexpr std::size_t MaximumFrameProbes=LumenMaximumProbes*2; // Separate opaque and translucent irradiance grids.
        bool CanTrace(unsigned count,std::string& error) const
        {
            if (!count || count>LumenDistanceMaximumRays || count>MaximumFrameRays-Rays || Radiance.size()>=MaximumChains)
            { error="Lumen frame exceeds its ray or command budget"; return false; }
            return true;
        }
        bool CanIntegrate(unsigned count,std::string& error) const
        {
            if (!count || count>LumenMaximumProbes || count>MaximumFrameProbes-Probes || Irradiance.size()>=MaximumChains)
            { error="Lumen frame exceeds its irradiance-probe or command budget"; return false; }
            return true;
        }
        void Record(LumenRadianceHandle handle,unsigned count) { Radiance.push_back(handle); Rays+=count; }
        void Record(LumenProbeHandle handle,unsigned count) { Irradiance.push_back(handle); Probes+=count; }
        bool CanGather(LumenProbeHandle probe,LumenRadianceHandle source,std::string& error) const
        {
            if (Gathered || std::find(Irradiance.begin(),Irradiance.end(),probe)==Irradiance.end() ||
                std::find(Radiance.begin(),Radiance.end(),source)==Radiance.end())
            { error="Lumen gather requires this frame's probe/ray results and may compose only once"; return false; }
            return true;
        }
        void Gather() noexcept { Gathered=true; }
        bool HasGathered() const noexcept { return Gathered; }
        template<class RadianceMap,class ProbeMap>
        void Finish(bool publish,RadianceMap& radiance,ProbeMap& probes)
        {
            if (!publish)
            {
                for (auto handle:Irradiance) probes.erase(handle.Value);
                for (auto handle:Radiance) radiance.erase(handle.Value);
            }
            Radiance.clear(); Irradiance.clear(); Rays=Probes=0; Gathered=false;
        }
    private:
        std::vector<LumenRadianceHandle> Radiance;
        std::vector<LumenProbeHandle> Irradiance;
        std::size_t Rays=0,Probes=0;
        bool Gathered=false;
    };
}
