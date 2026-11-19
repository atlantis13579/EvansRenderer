#include "Renderer/BakedLighting/BakedLightmapCache.h"
#include "Renderer/BakedLighting/BakedLightingAssets.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>

namespace Evans::Renderer
{
    namespace
    {
        struct BakeKey
        {
            std::uint64_t Revision;
            Techniques::Settings Materials;
            LightmapAtlasSettings Atlas;
            BakedLightmapSettings Sampling;
            LightingState Lighting;
            std::filesystem::path Archive;
        };
        bool SameReceivers(const BakeKey& a, const BakeKey& b)
        {
            const auto& x = a.Materials; const auto& y = b.Materials;
            return a.Revision == b.Revision && x.TextureMapping == y.TextureMapping && x.NormalMapping == y.NormalMapping &&
                x.AlphaTesting == y.AlphaTesting && x.AlphaBlending == y.AlphaBlending && x.Refraction.Enabled==y.Refraction.Enabled && x.Shading == y.Shading && x.Metallic == y.Metallic &&
                a.Atlas.Resolution == b.Atlas.Resolution && a.Atlas.Padding == b.Atlas.Padding &&
                a.Atlas.MaximumNormalAngleDegrees == b.Atlas.MaximumNormalAngleDegrees &&
                a.Atlas.MaximumChartTriangles == b.Atlas.MaximumChartTriangles;
        }
        bool SameBake(const BakeKey& a, const BakeKey& b)
        {
            if (a.Archive!=b.Archive || !SameReceivers(a,b)) return false;
            if (!a.Archive.empty()) return true;
            const auto& x = a.Sampling; const auto& y = b.Sampling;
            return x.RayCount == y.RayCount && x.MaximumDistance == y.MaximumDistance &&
                x.SurfaceBias == y.SurfaceBias && x.Shadows == y.Shadows && x.BakeBackFaces == y.BakeBackFaces && SameSceneLights(a.Lighting,b.Lighting);
        }
    }

    struct BakedLightmapCache::Impl
    {
        struct Job
        {
            BakeKey Key;
            std::uint64_t Generation = 0;
            std::vector<LightmapSourceInstance> Instances;
            std::shared_ptr<std::atomic<bool>> Cancelled = std::make_shared<std::atomic<bool>>(false);
            std::shared_ptr<const LightmapReceivers> Prepared;
        };
        mutable std::mutex Mutex;
        std::condition_variable Wake;
        std::thread Worker;
        bool Stopping = false;
        std::shared_ptr<Job> Pending;
        std::shared_ptr<std::atomic<bool>> ActiveCancellation;
        std::optional<BakeKey> RequestedKey, PublishedKey;
        LightmapBakeStatus Progress;

        ~Impl()
        {
            {
                std::lock_guard<std::mutex> lock(Mutex); Stopping = true;
                if (ActiveCancellation) ActiveCancellation->store(true,std::memory_order_relaxed);
                Pending.reset();
            }
            Wake.notify_one();
            if (Worker.joinable()) Worker.join();
        }

        void Run()
        {
            for (;;)
            {
                std::shared_ptr<Job> job;
                {
                    std::unique_lock<std::mutex> lock(Mutex);
                    Wake.wait(lock,[&] { return Stopping || Pending; });
                    if (Stopping) return;
                    job = std::move(Pending); ActiveCancellation = job->Cancelled;
                    Progress.State = LightmapBakeState::Preparing;
                }
                auto cancelled = [&] { return job->Cancelled->load(std::memory_order_relaxed); };
                std::string error;
                std::shared_ptr<const LightmapBakeResult> completed;
                try
                {
                    if (!job->Key.Archive.empty() && !cancelled())
                    {
                        LightmapBakeRequest request; request.Instances=job->Instances; request.Materials=job->Key.Materials;
                        request.Atlas=job->Key.Atlas; request.Sampling=job->Key.Sampling; request.Lighting=job->Key.Lighting;
                        if (auto loaded=LoadLightmapAsset(job->Key.Archive,request))
                        {
                            auto result=std::make_shared<LightmapBakeResult>(*loaded); result->Generation=job->Generation;
                            result->LoadedFromAsset=true; completed=std::move(result);
                        }
                        else error="Prebaked lightmap is missing, invalid or incompatible: "+job->Key.Archive.u8string();
                    }
                    auto receivers = job->Prepared;
                    if (job->Key.Archive.empty() && !completed && !receivers && !cancelled())
                    {
                        auto prepared = std::make_shared<LightmapReceivers>();
                        if (BuildLightmapReceivers(job->Instances,job->Key.Materials,job->Key.Atlas,*prepared,error,cancelled))
                        {
                            receivers = std::move(prepared);
                            std::lock_guard<std::mutex> lock(Mutex);
                            if (job->Generation == Progress.RequestedGeneration && !cancelled()) ++Progress.ReceiverBuildCount;
                        }
                    }
                    // Native streams and immutable queries now own all needed
                    // data; release copied source topology before the long bake.
                    job->Instances.clear();
                    if (job->Key.Archive.empty() && !completed && receivers && !cancelled())
                    {
                        {
                            std::lock_guard<std::mutex> lock(Mutex);
                            if (job->Generation == Progress.RequestedGeneration && !cancelled()) Progress.State = LightmapBakeState::Baking;
                        }
                        auto irradiance = std::make_shared<BakedLightmap>();
                        if (BakeLightmap(receivers->Atlas,*receivers->TransportScene,job->Key.Lighting,job->Key.Sampling,*irradiance,error,cancelled))
                        {
                            auto result = std::make_shared<LightmapBakeResult>();
                            result->Generation = job->Generation; result->Receivers = std::move(receivers); result->Irradiance = std::move(irradiance);
                            completed = std::move(result);
                        }
                    }
                }
                catch (const std::exception& exception) { error = std::string("Lightmap worker failed: ")+exception.what(); }
                catch (...) { error = "Lightmap worker failed with an unknown exception"; }
                {
                    std::lock_guard<std::mutex> lock(Mutex);
                    if (ActiveCancellation == job->Cancelled) ActiveCancellation.reset();
                    if (Stopping) return;
                    if (job->Generation != Progress.RequestedGeneration || cancelled()) continue;
                    if (completed)
                    {
                        Progress.Published = std::move(completed); PublishedKey = job->Key;
                        Progress.State = LightmapBakeState::Ready; Progress.Error.clear();
                        if (!Progress.Published->LoadedFromAsset) ++Progress.BakeCount;
                    }
                    else
                    {
                        Progress.State = LightmapBakeState::Failed;
                        Progress.Error = error.empty() ? "Lightmap worker did not produce a complete bake" : std::move(error);
                    }
                }
            }
        }

        void CancelLocked()
        {
            if (ActiveCancellation) ActiveCancellation->store(true,std::memory_order_relaxed);
            Pending.reset(); RequestedKey.reset(); ++Progress.RequestedGeneration;
            Progress.State = LightmapBakeState::Idle; Progress.Error.clear();
        }
    };

    BakedLightmapCache::BakedLightmapCache() : State(std::make_unique<Impl>()) {}
    BakedLightmapCache::~BakedLightmapCache() = default;

    std::uint64_t BakedLightmapCache::Request(LightmapBakeRequest request)
    {
        BakeKey key {request.SceneRevision,Techniques::SanitizeSettings(request.Materials),request.Atlas,request.Sampling,request.Lighting,request.Archive};
        std::lock_guard<std::mutex> lock(State->Mutex);
        if (State->RequestedKey && SameBake(*State->RequestedKey,key)) return State->Progress.RequestedGeneration;
        State->CancelLocked(); State->RequestedKey = key;
        const auto generation = State->Progress.RequestedGeneration;
        if (State->PublishedKey && State->Progress.Published && SameBake(*State->PublishedKey,key))
        {
            auto reused = std::make_shared<LightmapBakeResult>(*State->Progress.Published); reused->Generation = generation;
            State->Progress.Published = std::move(reused); State->Progress.State = LightmapBakeState::Ready;
            return generation;
        }
        auto job = std::make_shared<Impl::Job>(); job->Key = std::move(key); job->Generation = generation;
        job->Instances = std::move(request.Instances);
        if (State->PublishedKey && State->Progress.Published && State->Progress.Published->Receivers->TransportScene &&
            SameReceivers(*State->PublishedKey,job->Key))
            job->Prepared = State->Progress.Published->Receivers;
        State->Pending = std::move(job); State->Progress.State = LightmapBakeState::Queued;
        if (!State->Worker.joinable())
        {
            try { State->Worker = std::thread([impl = State.get()] { impl->Run(); }); }
            catch (const std::exception& exception)
            {
                State->Pending.reset(); State->Progress.State = LightmapBakeState::Failed;
                State->Progress.Error = std::string("Cannot start lightmap worker: ")+exception.what();
                return generation;
            }
        }
        State->Wake.notify_one();
        return generation;
    }

    LightmapBakeStatus BakedLightmapCache::Status() const
    { std::lock_guard<std::mutex> lock(State->Mutex); return State->Progress; }

    void BakedLightmapCache::Cancel()
    { std::lock_guard<std::mutex> lock(State->Mutex); State->CancelLocked(); }

    void BakedLightmapCache::Reset()
    {
        std::lock_guard<std::mutex> lock(State->Mutex); State->CancelLocked();
        State->PublishedKey.reset(); State->Progress.Published.reset();
        State->Progress.BakeCount = State->Progress.ReceiverBuildCount = 0;
    }
}
