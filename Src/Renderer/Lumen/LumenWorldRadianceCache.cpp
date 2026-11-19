#include "Renderer/Lumen/LumenWorldRadianceCache.h"
#include "Renderer/IRHI.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>

namespace Evans::Renderer
{
    // Wright et al., "Lumen: Real-time Global Illumination in Unreal Engine 5",
    // SIGGRAPH 2022: world probes cache directional radiance beyond a near sphere.
    // Interpolation must follow a local visibility trace covering that sphere
    // plus one lattice-cell diagonal. A diffuse SH cache cannot replace these
    // angular samples: parallax reconstruction needs their directions and depth.
    // This producer keeps native surface/miss/unresolved provenance, not invented
    // sky for missing data. A later consumer supplies sky only for proven misses.
    bool LumenWorldCacheSettings::operator==(const LumenWorldCacheSettings& other) const noexcept
    {
        return std::tie(FirstProbe.X,FirstProbe.Y,FirstProbe.Z,CellSize,Dimensions,DirectionalSide,DepthLayers,MaximumSteps,MinimumDistance,MaximumDistance,HitToleranceCells,StepScale,CascadeCount,CameraPosition.X,CameraPosition.Y,CameraPosition.Z,ScrollWithCamera,ProbesPerUpdate)==
            std::tie(other.FirstProbe.X,other.FirstProbe.Y,other.FirstProbe.Z,other.CellSize,other.Dimensions,other.DirectionalSide,other.DepthLayers,other.MaximumSteps,
                other.MinimumDistance,other.MaximumDistance,other.HitToleranceCells,other.StepScale,other.CascadeCount,other.CameraPosition.X,other.CameraPosition.Y,other.CameraPosition.Z,other.ScrollWithCamera,other.ProbesPerUpdate);
    }
    static bool BuildWorldLattice(const LumenWorldCacheSettings& s,LumenWorldCachePlan& output,std::string& error)
    {
        static_assert(Techniques::LumenWorldRadianceMaximumRays==LumenDistanceMaximumRays,"GUI world-cache ray budget changed");
        constexpr unsigned MinimumAxis=2,MaximumAxis=16,MinimumSide=4,MaximumSide=32;
        const auto finite=[](float x) { return std::isfinite(x); };
        if (!finite(s.CellSize) || s.CellSize<=0 || !finite(1/s.CellSize) || !finite(s.FirstProbe.X) || !finite(s.FirstProbe.Y) || !finite(s.FirstProbe.Z) ||
            !finite(s.MinimumDistance) || !finite(s.MaximumDistance) || s.MaximumDistance<=s.MinimumDistance ||
            s.DirectionalSide<MinimumSide || s.DirectionalSide>MaximumSide || (s.DirectionalSide&(s.DirectionalSide-1)) ||
            !s.MaximumSteps || s.MaximumSteps>LumenDistanceMaximumTraceSteps || !finite(s.HitToleranceCells) || s.HitToleranceCells<0 || s.HitToleranceCells>=2 ||
            !finite(s.StepScale) || s.StepScale<=0 || s.StepScale>1)
        { error="World radiance caches require finite lattice, square angular atlas and bounded tracing parameters"; return false; }
        std::uint64_t count=1;
        for (auto axis:s.Dimensions)
        { if (axis<MinimumAxis || axis>MaximumAxis) { error="World radiance-cache dimensions must be between 2 and 16"; return false; } count*=axis; }
        const auto rays=s.DirectionalSide*s.DirectionalSide;
        if (!ValidateLumenProbeSampling(unsigned(count),rays,s.MaximumDistance,error,s.DepthLayers)) return false;
        const double diagonal=std::sqrt(3.0)*s.CellSize;
        const float origin[]{s.FirstProbe.X,s.FirstProbe.Y,s.FirstProbe.Z};
        double coordinateScale=0;
        for (unsigned axis=0;axis<3;++axis)
            coordinateScale=std::max({coordinateScale,std::abs(double(origin[axis])),std::abs(double(origin[axis])+double(s.Dimensions[axis]-1)*s.CellSize)});
        constexpr double CoordinateRoundoff=32*std::numeric_limits<float>::epsilon();
        const double cellBound=diagonal+CoordinateRoundoff*coordinateScale;
        const float local=std::nextafter(float(double(s.MinimumDistance)+cellBound),std::numeric_limits<float>::infinity());
        if (s.MinimumDistance<cellBound || !finite(local) || local>=s.MaximumDistance)
        { error="World probes need a near sphere covering a cell diagonal and a farther trace horizon"; return false; }
        LumenWorldCachePlan result;
        result.Layout.OriginCell={s.FirstProbe.X,s.FirstProbe.Y,s.FirstProbe.Z,s.CellSize};
        result.Layout.Dimensions={s.Dimensions[0],s.Dimensions[1],s.Dimensions[2],s.DirectionalSide};
        result.Layout.Distances={s.MinimumDistance,local,s.MaximumDistance,1/s.CellSize};
        auto& sampling=result.Trace.Probes; sampling.RaysPerProbe=rays; sampling.DirectionalSide=s.DirectionalSide; sampling.DepthLayers=s.DepthLayers; sampling.MaximumDistance=s.MaximumDistance;
        for (unsigned axis=0;axis<3;++axis)
        {
            float previous=origin[axis];
            for (unsigned coordinate=1;coordinate<s.Dimensions[axis];++coordinate)
            {
                const double exact=double(origin[axis])+double(coordinate)*s.CellSize; const float value=float(exact);
                constexpr double MaximumCellError=1.0/1024;
                if (!finite(value) || value<=previous || std::abs(double(value)-exact)>double(s.CellSize)*MaximumCellError)
                { error="World radiance-cache lattice loses its spacing at these coordinates"; return false; }
                previous=value;
            }
        }
        sampling.Origins.reserve(std::size_t(count));
        for (unsigned z=0;z<s.Dimensions[2];++z) for (unsigned y=0;y<s.Dimensions[1];++y) for (unsigned x=0;x<s.Dimensions[0];++x)
        {
            LumenProbeOrigin probe; probe.PositionMinimum={float(double(s.FirstProbe.X)+double(x)*s.CellSize),float(double(s.FirstProbe.Y)+double(y)*s.CellSize),
                float(double(s.FirstProbe.Z)+double(z)*s.CellSize),s.MinimumDistance};
            probe.Metadata.X=unsigned(sampling.Origins.size()); sampling.Origins.push_back(probe);
        }
        if (!ValidateLumenProbeRaySet(sampling,error)) return false;
        result.Trace.Trace.MaximumSteps=s.MaximumSteps; result.Trace.Trace.HitToleranceCells=s.HitToleranceCells; result.Trace.Trace.StepScale=s.StepScale;
        output=std::move(result); error.clear(); return true;
    }
    bool BuildLumenWorldCachePlan(const LumenWorldCacheSettings& settings,LumenWorldCachePlan& output,std::string& error)
    {
        if (!settings.CascadeCount || settings.CascadeCount>LumenWorldMaximumCascades)
        { error="World radiance caches require one to four nested cascades"; return false; }
        if (settings.ProbesPerUpdate>LumenMaximumProbes) { error="World cache update budget exceeds probe capacity"; return false; }
        auto finest=settings;
        const double camera[]{settings.CameraPosition.X,settings.CameraPosition.Y,settings.CameraPosition.Z};
        if (settings.ScrollWithCamera)
        {
            if (!std::isfinite(settings.CellSize) || settings.CellSize<=0 || !std::isfinite(camera[0]) || !std::isfinite(camera[1]) || !std::isfinite(camera[2]))
            { error="Scrolling world caches require a finite camera and positive spacing"; return false; }
            float origin[3];
            for (unsigned axis=0;axis<3;++axis)
                origin[axis]=float((std::floor(camera[axis]/settings.CellSize)-(double(settings.Dimensions[axis])-1)*0.5)*settings.CellSize);
            finest.FirstProbe={origin[0],origin[1],origin[2]};
        }
        LumenWorldCachePlan result; if (!BuildWorldLattice(finest,result,error)) return false;
        auto& probes=result.Trace.Probes; const auto perCascade=unsigned(probes.Origins.size());
        if (!ValidateLumenProbeSampling(perCascade*settings.CascadeCount,probes.RaysPerProbe,probes.MaximumDistance,error,probes.DepthLayers)) return false;
        const double first[]{finest.FirstProbe.X,finest.FirstProbe.Y,finest.FirstProbe.Z}; double center[3];
        for (unsigned axis=0;axis<3;++axis) center[axis]=first[axis]+double(settings.Dimensions[axis]-1)*settings.CellSize*0.5;
        // Each scrolling octave snaps independently, preserving coarse probes
        // during finer cell crossings. Fixed test lattices remain concentric.
        // All levels share one immutable generation and far trace horizon.
        for (unsigned level=1;level<settings.CascadeCount;++level)
        {
            auto coarse=settings; coarse.CascadeCount=1; coarse.CellSize=std::ldexp(settings.CellSize,int(level));
            coarse.MinimumDistance=std::ldexp(settings.MinimumDistance,int(level)); float origin[3];
            for (unsigned axis=0;axis<3;++axis)
            {
                const double anchor=settings.ScrollWithCamera?std::floor(camera[axis]/coarse.CellSize)*coarse.CellSize:center[axis];
                origin[axis]=float(anchor-double(settings.Dimensions[axis]-1)*coarse.CellSize*0.5);
            }
            coarse.FirstProbe={origin[0],origin[1],origin[2]}; LumenWorldCachePlan lattice;
            if (!BuildWorldLattice(coarse,lattice,error)) return false;
            result.Coarser.push_back(lattice.Layout);
            for (auto probe:lattice.Trace.Probes.Origins) { probe.Metadata.X=unsigned(probes.Origins.size()); probes.Origins.push_back(probe); }
        }
        output=std::move(result); error.clear(); return true;
    }
    bool BuildLumenWorldLookupFrame(const LumenRadianceInfo& cached,const LumenRadianceTraceUpload& query,
        const LumenRadianceInput& input,LumenWorldLookupFrame& output,std::string& error)
    {
        const auto& binding=query.WorldCache; const auto& layout=binding.Layout;
        if (!binding.Radiance || binding.Coarser.size()>=LumenWorldMaximumCascades || binding.AngularFilterRadius>Techniques::LumenWorldMaximumFilterRadius ||
            !std::isfinite(binding.BlendCells) || binding.BlendCells<0 || binding.BlendCells>2 ||
            !cached.Probes.DirectionalSide || cached.HairTransport || cached.WorldCache || cached.WorldCacheCascades || cached.Hybrid || cached.Screen.Grid.Y ||
            !input.ProbeTrace.Sampling.X || query.Probes.DirectionalSide || query.Probes.DepthLayers!=1 || !std::isfinite(binding.SkyRadiance.X) ||
            !std::isfinite(binding.SkyRadiance.Y) || !std::isfinite(binding.SkyRadiance.Z) ||
            binding.SkyRadiance.X<0 || binding.SkyRadiance.Y<0 || binding.SkyRadiance.Z<0)
        { error="World lookup requires an immutable directional source, probe queries and nonnegative sky"; return false; }
        LumenWorldCacheSettings settings; settings.FirstProbe={layout.OriginCell.X,layout.OriginCell.Y,layout.OriginCell.Z};
        settings.CellSize=layout.OriginCell.W; settings.Dimensions={layout.Dimensions.X,layout.Dimensions.Y,layout.Dimensions.Z};
        settings.DirectionalSide=layout.Dimensions.W; settings.DepthLayers=cached.Probes.DepthLayers; settings.MinimumDistance=layout.Distances.X; settings.MaximumDistance=layout.Distances.Z;
        const auto cascades=unsigned(binding.Coarser.size()+1);
        LumenWorldCachePlan plan;
        // Residency may scroll each octave independently. Verify every supplied
        // lattice against its retained origins instead of assuming one center.
        for (unsigned level=0;level<cascades;++level)
        {
            const auto& grid=level?binding.Coarser[level-1]:layout; auto lattice=settings;
            lattice.FirstProbe={grid.OriginCell.X,grid.OriginCell.Y,grid.OriginCell.Z};
            lattice.CellSize=std::ldexp(settings.CellSize,int(level)); lattice.MinimumDistance=std::ldexp(settings.MinimumDistance,int(level));
            LumenWorldCachePlan built; if (!BuildWorldLattice(lattice,built,error)) return false;
            if (std::memcmp(&built.Layout,&grid,sizeof(grid))) { error="World cache cascade metadata changed"; return false; }
            if (!level) plan=std::move(built);
            else for (auto probe:built.Trace.Probes.Origins) { probe.Metadata.X=unsigned(plan.Trace.Probes.Origins.size()); plan.Trace.Probes.Origins.push_back(probe); }
        }
        const auto& expected=plan.Trace.Probes;
        if (!ValidateLumenProbeRaySet(expected,error)) return false;
        if (cached.Probes.DirectionalSide!=expected.DirectionalSide ||
            cached.Probes.RaysPerProbe!=expected.RaysPerProbe || cached.Probes.Sequence || cached.Probes.MaximumDistance!=expected.MaximumDistance || query.Probes.MaximumDistance!=expected.MaximumDistance ||
            cached.Probes.Origins.size()!=expected.Origins.size() || cached.RayCount!=expected.Origins.size()*expected.RaysPerProbe*expected.DepthLayers ||
            std::memcmp(cached.Probes.Origins.data(),expected.Origins.data(),expected.Origins.size()*sizeof(LumenProbeOrigin)))
        { error="World lookup layout does not identify the resident directional lattice"; return false; }
        LumenWorldLookupFrame result; if (!BuildLumenEnvironmentFrame(binding.Environment,result.Environment,error)) return false; result.Layout=layout; std::copy(binding.Coarser.begin(),binding.Coarser.end(),result.Coarser.begin());
        result.Work={input.Trace.Counts.X,query.Probes.RaysPerProbe,query.Probes.Sequence,LumenRadianceThreads*LumenRadianceColumns};
        result.Sky={binding.SkyRadiance.X,binding.SkyRadiance.Y,binding.SkyRadiance.Z,binding.Parallax?1.0f:0.0f};
        constexpr float ProjectionGuard=32*std::numeric_limits<float>::epsilon();
        result.Query={query.Probes.MaximumDistance,query.Hybrid.Enabled?1.0f:0.0f,ProjectionGuard,float(cascades)};
        result.Transition={binding.BlendCells,float(binding.AngularFilterRadius),float(expected.DepthLayers),0};
        output=result; error.clear(); return true;
    }
    LumenWorldRadianceCache::~LumenWorldRadianceCache() { Reset(); }
    namespace
    {
        bool SameWorldSampling(LumenWorldCacheSettings a,LumenWorldCacheSettings b)
        {
            a.FirstProbe=b.FirstProbe={}; a.CameraPosition=b.CameraPosition={}; a.ProbesPerUpdate=b.ProbesPerUpdate=0;
            return a==b;
        }
        using OriginKey=std::array<std::uint32_t,4>;
        OriginKey WorldOriginKey(const LumenProbeOrigin& origin)
        { OriginKey key; std::memcpy(key.data(),&origin.PositionMinimum,sizeof(Float4)); return key; }
    }
    bool LumenWorldRadianceCache::CanUseSnapshot(LumenLightingHandle source,const LumenWorldCacheSettings& settings) const
    { return Published.Radiance && Published.Lighting==source && SameWorldSampling(PublishedSettings,settings); }
    bool LumenWorldRadianceCache::Request(LumenLightingHandle source,const LumenWorldCacheSettings& settings,std::string& error)
    {
        if (!source) { error="World radiance-cache source is absent"; return false; }
        LumenWorldCachePlan candidate; if (!BuildLumenWorldCachePlan(settings,candidate,error)) return false;
        const bool compatible=Published.Radiance && Published.Lighting==source && SameWorldSampling(PublishedSettings,settings);
        const auto& origins=candidate.Trace.Probes.Origins;
        if (compatible && origins.size()==Published.Info.Probes.Origins.size() &&
            !std::memcmp(origins.data(),Published.Info.Probes.Origins.data(),origins.size()*sizeof(LumenProbeOrigin)))
        { Cancel(); PublishedSettings=settings; error.clear(); return true; }
        if (State==LumenWorldCacheState::Queued && RequestedSource==source && SameWorldSampling(RequestedSettings,settings))
        { LatestSettings=settings; RequestedSettings.ProbesPerUpdate=settings.ProbesPerUpdate; error.clear(); return true; }
        Cancel(); Pending=std::move(candidate); RequestedSource=source; RequestedSettings=LatestSettings=settings;
        Progress={unsigned(Pending.Trace.Probes.Origins.size()),0,0,0}; ProbeSources.resize(Progress.TotalProbes);
        std::map<OriginKey,unsigned> previous;
        if (compatible) for (unsigned probe=0;probe<Published.Info.Probes.Origins.size();++probe)
            previous.emplace(WorldOriginKey(Published.Info.Probes.Origins[probe]),probe);
        for (unsigned probe=0;probe<Progress.TotalProbes;++probe)
        {
            const auto found=previous.find(WorldOriginKey(Pending.Trace.Probes.Origins[probe]));
            if (found!=previous.end()) { ProbeSources[probe]={Published.Radiance,found->second,probe,1}; ++Progress.ReusedProbes; }
            else Dirty.push_back(probe);
        }
        Failure.clear(); State=LumenWorldCacheState::Queued; error.clear(); return true;
    }
    bool LumenWorldRadianceCache::Advance(std::string& error)
    {
        if (State==LumenWorldCacheState::Failed) { error=Failure; return false; }
        if (State!=LumenWorldCacheState::Queued) { Progress.LastUpdateProbes=0; error.clear(); return true; }
        if (Generation==std::numeric_limits<std::uint64_t>::max())
        { Failure=error="World radiance-cache generation space exhausted"; State=LumenWorldCacheState::Failed; return false; }
        const auto remaining=unsigned(Dirty.size())-Progress.TracedProbes;
        const auto count=RequestedSettings.ProbesPerUpdate?std::min(remaining,RequestedSettings.ProbesPerUpdate):remaining;
        Progress.LastUpdateProbes=0; LumenRadianceTraceResult assembled;
        if (count)
        {
            auto upload=Pending.Trace; upload.Trace.HairTransport=false; upload.Probes.Origins.clear(); upload.Probes.Origins.reserve(count);
            for (unsigned i=0;i<count;++i) upload.Probes.Origins.push_back(Pending.Trace.Probes.Origins[Dirty[Progress.TracedProbes+i]]);
            LumenRadianceTraceResult batch;
            if (!Rhi.TraceLumenRadiance(RequestedSource,upload,batch))
            { Failure=error=Rhi.GetLastError(); State=LumenWorldCacheState::Failed; return false; }
            Staged.push_back(batch.Handle);
            for (unsigned i=0;i<count;++i)
            {
                const auto destination=Dirty[Progress.TracedProbes+i]; ProbeSources[destination]={batch.Handle,i,destination,1};
            }
            Progress.TracedProbes+=count; Progress.LastUpdateProbes=count;
            if (count==Progress.TotalProbes) assembled=std::move(batch); // A full first batch already has the final packing.
        }
        if (Progress.PendingProbes()) { error.clear(); return true; }
        LumenRadianceAssembly assembly; assembly.Probes=Pending.Trace.Probes;
        for (const auto& probe:ProbeSources)
        {
            if (!assembly.Copies.empty())
            {
                auto& back=assembly.Copies.back();
                if (back.Source==probe.Source && back.SourceProbe+back.ProbeCount==probe.SourceProbe)
                { ++back.ProbeCount; continue; }
            }
            assembly.Copies.push_back(probe);
        }
        if (!assembled.Handle && !Rhi.AssembleLumenRadiance(assembly,assembled))
        { Failure=error=Rhi.GetLastError(); State=LumenWorldCacheState::Failed; return false; }
        LumenWorldCacheSnapshot snapshot; snapshot.Lighting=RequestedSource; snapshot.Radiance=assembled.Handle; snapshot.Layout=Pending.Layout;
        snapshot.Coarser=std::move(Pending.Coarser); snapshot.Info=std::move(assembled.Info); snapshot.Generation=++Generation; snapshot.Statistics=Progress;
        const auto previous=Published.Radiance; Published=std::move(snapshot); PublishedSettings=RequestedSettings;
        Staged.erase(std::remove(Staged.begin(),Staged.end(),Published.Radiance),Staged.end());
        const auto latest=LatestSettings; const auto source=RequestedSource;
        Cancel(); if (previous) Rhi.DestroyLumenRadiance(previous);
        // Queue the latest camera target after publication. This also keeps an
        // otherwise idle editor refreshing until coalesced motion is satisfied.
        return Request(source,latest,error);
    }
    void LumenWorldRadianceCache::ReleaseStaged()
    { for (auto handle:Staged) Rhi.DestroyLumenRadiance(handle); Staged.clear(); }
    void LumenWorldRadianceCache::Cancel()
    {
        ReleaseStaged(); ProbeSources.clear(); Dirty.clear(); Pending={}; RequestedSource={}; LatestSettings={}; Failure.clear();
        Progress=Published.Statistics; State=Published.Radiance?LumenWorldCacheState::Ready:LumenWorldCacheState::Empty;
    }
    void LumenWorldRadianceCache::Reset()
    {
        ReleaseStaged(); if (Published.Radiance) Rhi.DestroyLumenRadiance(Published.Radiance);
        Published={}; Cancel();
    }
}
