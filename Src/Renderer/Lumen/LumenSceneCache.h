#pragma once

#include "Renderer/Lumen/LumenScene.h"

namespace Evans::Renderer
{
    class IRHI;
    enum class LumenSceneUpdateState { Idle,Distance,Capture,Lighting,Ready,Failed };
    struct LumenSceneCacheStatus
    {
        LumenSceneUpdateState State=LumenSceneUpdateState::Idle;
        std::uint64_t RequestedGeneration=0,PublishedGeneration=0;
        std::uint64_t DistanceUploads=0,MaterialCaptures=0,Relights=0;
        LumenDistanceSceneHandle Distance;
        LumenSurfaceHandle Surface;
        LumenLightingHandle Lighting;
        unsigned CompletedBounces=0,TargetBounces=0;
        LumenLightingStatistics LightingTiles;
        LumenCaptureStatistics CaptureCards;
        bool Building=false; // A frozen scene transaction has not published yet.
        bool Usable=false; // Published lighting belongs to the requested geometry/material scene.
        std::string Error;
        bool Current() const { return State==LumenSceneUpdateState::Ready && RequestedGeneration==PublishedGeneration; }
    };

    // Render-thread coordinator for bounded, immutable scene transactions.
    // Distance upload remains a whole stage. Positive capture/lighting budgets
    // submit asynchronous card/tile slices; zero retains synchronous compatibility.
    // Each indirect iteration reads the previous complete atlas; only the final
    // bounce publishes. Request owns the seed (caller-supplied history is rejected).
    // All calls, including destruction, require a closed frame and a live RHI.
    // Draw buffers/textures remain caller-owned until capture completes.
    // Status handles are borrowed until the next Advance/Reset/destruction;
    // Usable permits older lighting of the exact same scene during a light edit;
    // Current additionally requires the latest requested lighting generation.
    class LumenSceneCache
    {
    public:
        explicit LumenSceneCache(IRHI& rhi);
        ~LumenSceneCache();
        LumenSceneCache(const LumenSceneCache&)=delete;
        LumenSceneCache& operator=(const LumenSceneCache&)=delete;
        bool Request(std::shared_ptr<const LumenScenePlan> scene,LumenSurfaceLightingUpload lighting,std::string& error,unsigned indirectBounces=0);
        bool Advance(unsigned stageBudget=1,unsigned lightingTileBudget=0,unsigned captureCardBudget=0);
        void Cancel();
        void Reset();
        LumenSceneCacheStatus Status() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> State;
    };
}
