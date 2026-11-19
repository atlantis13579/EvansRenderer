#pragma once

#include "Renderer/Lumen/LumenRadianceTrace.h"

namespace Evans::Renderer
{
    class IRHI;
    struct LumenWorldCacheSettings
    {
        Float3 FirstProbe{-1,-1,-1};
        Float3 CameraPosition{};
        bool ScrollWithCamera=false;
        unsigned ProbesPerUpdate=0; // Zero completes the pending generation in one Advance.
        float CellSize=1;
        std::array<unsigned,3> Dimensions{4,4,4};
        unsigned DirectionalSide=8,MaximumSteps=256,CascadeCount=1,DepthLayers=1;
        float MinimumDistance=2,MaximumDistance=25,HitToleranceCells=0.1f,StepScale=0.9f;
        bool operator==(const LumenWorldCacheSettings& other) const noexcept;
    };
    struct LumenWorldCachePlan { LumenWorldCacheLayout Layout; std::vector<LumenWorldCacheLayout> Coarser; LumenRadianceTraceUpload Trace; };
    bool BuildLumenWorldCachePlan(const LumenWorldCacheSettings& settings,LumenWorldCachePlan& output,std::string& error);

    enum class LumenWorldCacheState { Empty,Queued,Ready,Failed };
    struct LumenWorldCacheStatistics
    {
        unsigned TotalProbes=0,ReusedProbes=0,TracedProbes=0,LastUpdateProbes=0;
        unsigned PendingProbes() const noexcept { return TotalProbes-ReusedProbes-TracedProbes; }
    };
    struct LumenWorldCacheSnapshot
    {
        LumenLightingHandle Lighting;
        LumenRadianceHandle Radiance;
        LumenWorldCacheLayout Layout;
        std::vector<LumenWorldCacheLayout> Coarser;
        LumenRadianceInfo Info;
        std::uint64_t Generation=0;
        LumenWorldCacheStatistics Statistics;
    };
    // Closed-frame, render-thread ownership, like LumenSceneCache. The source
    // handle fixes geometry, material and direct-light revisions. A completed
    // replacement is published atomically; a rejected request, cancelled job or
    // failed native build leaves the prior immutable allocation intact.
    // Snapshot handles are borrowed until the next successful Advance/Reset.
    // A pending camera move may retain the old bounded layout for the same
    // lighting source. Queries outside it still use the full SDF trace. Camera
    // requests coalesce while a generation builds, preventing update starvation.
    class LumenWorldRadianceCache
    {
    public:
        explicit LumenWorldRadianceCache(IRHI& rhi):Rhi(rhi) {}
        ~LumenWorldRadianceCache();
        LumenWorldRadianceCache(const LumenWorldRadianceCache&)=delete;
        LumenWorldRadianceCache& operator=(const LumenWorldRadianceCache&)=delete;
        bool Request(LumenLightingHandle source,const LumenWorldCacheSettings& settings,std::string& error);
        bool Advance(std::string& error);
        void Cancel();
        void Reset();
        bool CanUseSnapshot(LumenLightingHandle,const LumenWorldCacheSettings&) const;
        bool Current() const noexcept { return State==LumenWorldCacheState::Ready; }
        LumenWorldCacheState GetState() const noexcept { return State; }
        const LumenWorldCacheSnapshot& Snapshot() const noexcept { return Published; }
        const LumenWorldCacheStatistics& Statistics() const noexcept { return Progress; }
    private:
        IRHI& Rhi;
        LumenWorldCacheState State=LumenWorldCacheState::Empty;
        LumenWorldCacheSnapshot Published;
        LumenLightingHandle RequestedSource;
        LumenWorldCacheSettings RequestedSettings,PublishedSettings,LatestSettings;
        LumenWorldCachePlan Pending;
        std::vector<LumenRadianceCopy> ProbeSources;
        std::vector<unsigned> Dirty;
        std::vector<LumenRadianceHandle> Staged;
        LumenWorldCacheStatistics Progress;
        void ReleaseStaged();
        std::uint64_t Generation=0;
        std::string Failure;
    };
}
