#pragma once

#include "Renderer/BakedLighting/LightmapReceivers.h"
#include <filesystem>

namespace Evans::Renderer
{
    struct LightmapBakeRequest
    {
        std::vector<LightmapSourceInstance> Instances;
        // Must advance for geometry, LOD topology, transforms, visibility,
        // instance colors, material values/images or CastShadow changes.
        std::uint64_t SceneRevision = 0;
        Techniques::Settings Materials;
        LightmapAtlasSettings Atlas;
        BakedLightmapSettings Sampling;
        LightingState Lighting;
        // A nonempty archive selects immutable prebaked lighting. Missing or
        // incompatible files fail visibly and never invoke the live baker.
        std::filesystem::path Archive;
    };

    struct LightmapBakeResult
    {
        std::uint64_t Generation = 0;
        std::shared_ptr<const LightmapReceivers> Receivers;
        std::shared_ptr<const BakedLightmap> Irradiance;
        bool LoadedFromAsset = false;
    };

    enum class LightmapBakeState : std::uint8_t { Idle, Queued, Preparing, Baking, Ready, Failed };

    struct LightmapBakeStatus
    {
        LightmapBakeState State = LightmapBakeState::Idle;
        std::uint64_t RequestedGeneration = 0, ReceiverBuildCount = 0, BakeCount = 0;
        std::shared_ptr<const LightmapBakeResult> Published;
        std::string Error;
        // An older completed bake remains inspectable while work is pending,
        // but must not be bound to a different scene/material/light request.
        [[nodiscard]] bool Current() const noexcept
        { return State == LightmapBakeState::Ready && Published && Published->Generation == RequestedGeneration; }
    };

    // One lazily started worker, one pending request. Superseding requests cancel
    // active work and replace the pending slot, never create detached threads.
    // Shader compilation and all GPU operations remain on the render thread.
    // Atlas/query builders are cancellation boundaries; ray integration checks
    // cancellation per ray. Destruction joins the owned worker.
    class BakedLightmapCache final
    {
    public:
        BakedLightmapCache();
        ~BakedLightmapCache();
        BakedLightmapCache(const BakedLightmapCache&) = delete;
        BakedLightmapCache& operator=(const BakedLightmapCache&) = delete;

        // Equivalent requests reuse a generation, including a completed failure
        // until input changes. Intensity, camera, post effects and selected LOD
        // are not bake inputs. Source snapshots must be immutable after request.
        std::uint64_t Request(LightmapBakeRequest request);
        [[nodiscard]] LightmapBakeStatus Status() const;
        void Cancel(); // Retains the last completed bake for a later equivalent request.
        void Reset();  // Also releases published CPU data and resets statistics.

    private:
        struct Impl;
        std::unique_ptr<Impl> State;
    };
}
