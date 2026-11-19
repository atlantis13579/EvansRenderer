#pragma once

#include "Renderer/Lumen/LumenSceneCache.h"

namespace Evans::Renderer
{
    struct LumenSourceGeometry
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
    };
    struct LumenSourceInstance
    {
        std::uint64_t Key=0;
        std::shared_ptr<const LumenSourceGeometry> Geometry;
        MaterialData Material;
        Float4 Color{1,1,1,1};
        Matrix4 World;
        LumenSurfaceDraw Capture;
        bool CastShadow=true;
    };
    enum class LumenRenderState { Idle,Preparing,Distance,Capture,Lighting,Ready,Failed };
    struct LumenRenderStatus
    {
        LumenRenderState State=LumenRenderState::Idle;
        std::uint64_t Generation=0,PreparedMeshes=0;
        unsigned Cards=0,Instances=0,CardSpatialSplits=0;
        // Local mesh samples, counted once per shared proxy, independent of instance scale.
        double SampledCardCoverage=0,SampledTexelCoverage=0,UncheckedTexelCoverage=0;
        bool CardCoverageBudgetExhausted=false;
        LumenLightingHandle Lighting;
        LumenDistanceSceneHandle Distance;
        unsigned CompletedBounces=0,TargetBounces=0;
        LumenLightingStatistics LightingTiles;
        LumenCaptureStatistics CaptureCards;
        bool HasLighting() const { return bool(Lighting); }
        std::string Error;
        bool Current() const { return State==LumenRenderState::Ready && bool(Lighting); }
        bool Pending() const { return State!=LumenRenderState::Idle && State!=LumenRenderState::Ready && State!=LumenRenderState::Failed; }
    };
    // CPU mesh proxies run on one owned, cancellable worker. Native allocations
    // and the staged scene transaction stay on the render thread between frames.
    // Pose-only edits coalesce behind a frozen capture. Published transport may
    // lag rigid motion; geometry/material/caster changes invalidate its reuse.
    class LumenRenderCache
    {
    public:
        explicit LumenRenderCache(IRHI& rhi);
        ~LumenRenderCache();
        LumenRenderCache(const LumenRenderCache&)=delete;
        LumenRenderCache& operator=(const LumenRenderCache&)=delete;
        void Request(std::vector<LumenSourceInstance> instances,std::uint64_t revision,
            const Techniques::Settings& settings,const LightingState& lighting,bool shadows,
            std::shared_ptr<const HairStrandGeometry> hair={},std::shared_ptr<const HairStrandGeometry> hairBase={});
        void Advance();
        void Cancel();
        LumenRenderStatus Status() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> State;
    };
}
