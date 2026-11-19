#include "Renderer/VoxelGI/VoxelRadianceCache.h"
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
        struct UpdateKey
        {
            std::uint64_t Revision=0;
            Techniques::Settings Materials;
            Techniques::VoxelGISettings Grid;
            LightingState Lighting;
            bool Shadows=true;
        };
        bool SameSource(const UpdateKey& a,const UpdateKey& b)
        {
            const auto& x=a.Materials; const auto& y=b.Materials;
            return a.Revision==b.Revision && x.TextureMapping==y.TextureMapping && x.NormalMapping==y.NormalMapping &&
                x.AlphaTesting==y.AlphaTesting && x.AlphaBlending==y.AlphaBlending && x.Shading==y.Shading && x.Metallic==y.Metallic;
        }
        bool SameGrid(const UpdateKey& a,const UpdateKey& b)
        {
            return SameSource(a,b) && a.Grid.Resolution==b.Grid.Resolution && a.Grid.AutomaticBounds==b.Grid.AutomaticBounds &&
                (a.Grid.AutomaticBounds || (a.Grid.BoundsMinimum==b.Grid.BoundsMinimum && a.Grid.BoundsMaximum==b.Grid.BoundsMaximum));
        }
        bool SameRadiance(const UpdateKey& a,const UpdateKey& b)
        { return SameGrid(a,b) && a.Shadows==b.Shadows && SameSceneLights(a.Lighting,b.Lighting); }
    }
    struct VoxelRadianceCache::Impl
    {
        struct Job
        {
            UpdateKey Key;
            std::uint64_t Generation=0;
            std::vector<VoxelSourceInstance> Instances;
            std::shared_ptr<std::atomic<bool>> Cancelled=std::make_shared<std::atomic<bool>>(false);
        };
        mutable std::mutex Mutex;
        std::condition_variable Wake;
        std::thread Worker;
        bool Stopping=false;
        std::shared_ptr<Job> Pending;
        std::shared_ptr<std::atomic<bool>> ActiveCancellation;
        std::optional<UpdateKey> RequestedKey,QueryKey,GeometryKey,PublishedKey;
        std::shared_ptr<const SceneRayQuery> Query;
        std::shared_ptr<const VoxelizedScene> Geometry;
        VoxelRadianceStatus Progress;

        ~Impl()
        {
            {
                std::lock_guard<std::mutex> lock(Mutex); Stopping=true;
                if (ActiveCancellation) ActiveCancellation->store(true,std::memory_order_relaxed);
                Pending.reset();
            }
            Wake.notify_one(); if (Worker.joinable()) Worker.join();
        }
        void Run()
        {
            for (;;)
            {
                std::shared_ptr<Job> job;
                std::shared_ptr<const SceneRayQuery> query;
                std::shared_ptr<const VoxelizedScene> geometry;
                {
                    std::unique_lock<std::mutex> lock(Mutex);
                    Wake.wait(lock,[&]{return Stopping || Pending;}); if (Stopping) return;
                    job=std::move(Pending); ActiveCancellation=job->Cancelled;
                    if (QueryKey && SameSource(*QueryKey,job->Key)) query=Query;
                    if (GeometryKey && SameGrid(*GeometryKey,job->Key)) geometry=Geometry;
                    Progress.State=VoxelUpdateState::Preparing;
                }
                const auto cancelled=[&]{return job->Cancelled->load(std::memory_order_relaxed);};
                std::string error; std::shared_ptr<const VoxelRadianceResult> completed;
                try
                {
                    if (!query && !cancelled())
                    {
                        std::vector<SceneRayMeshView> views; views.reserve(job->Instances.size()); bool valid=true;
                        for (const auto& instance:job->Instances)
                        {
                            if (!instance.Geometry) { error="Voxel source instance has no immutable geometry"; valid=false; break; }
                            SceneRayMeshView view; view.Vertices=&instance.Geometry->Vertices; view.Indices=&instance.Geometry->Indices;
                            view.World=instance.World; view.Color=instance.Color; view.Material=instance.Material;
                            view.Visible=instance.Visible; view.CastShadow=instance.CastShadow; views.push_back(std::move(view));
                        }
                        if (valid) query=SceneRayQuery::Create(views,job->Key.Materials,error);
                        if (query)
                        {
                            std::lock_guard<std::mutex> lock(Mutex);
                            if (job->Generation==Progress.RequestedGeneration && !cancelled())
                            { Query=query; QueryKey=job->Key; ++Progress.QueryBuildCount; }
                        }
                    }
                    job->Instances.clear(); // The immutable query now owns the accepted source.
                    if (query && !geometry && !cancelled())
                    {
                        {
                            std::lock_guard<std::mutex> lock(Mutex);
                            if (job->Generation==Progress.RequestedGeneration && !cancelled()) Progress.State=VoxelUpdateState::Voxelizing;
                        }
                        auto candidate=std::make_shared<VoxelizedScene>();
                        if (VoxelizeScene(query,job->Key.Grid,*candidate,error,cancelled))
                        {
                            geometry=std::move(candidate);
                            std::lock_guard<std::mutex> lock(Mutex);
                            if (job->Generation==Progress.RequestedGeneration && !cancelled())
                            { Geometry=geometry; GeometryKey=job->Key; ++Progress.VoxelBuildCount; }
                        }
                    }
                    if (geometry && !cancelled())
                    {
                        {
                            std::lock_guard<std::mutex> lock(Mutex);
                            if (job->Generation==Progress.RequestedGeneration && !cancelled()) Progress.State=VoxelUpdateState::Injecting;
                        }
                        auto radiance=std::make_shared<VoxelRadianceVolume>();
                        if (InjectVoxelRadiance(*geometry,job->Key.Lighting,job->Key.Shadows,*radiance,error,cancelled))
                        {
                            auto result=std::make_shared<VoxelRadianceResult>(); result->Generation=job->Generation;
                            result->Geometry=std::move(geometry); result->Radiance=std::move(radiance); completed=std::move(result);
                        }
                    }
                }
                catch (const std::exception& exception) { error=std::string("Voxel worker failed: ")+exception.what(); }
                catch (...) { error="Voxel worker failed with an unknown exception"; }
                {
                    std::lock_guard<std::mutex> lock(Mutex);
                    if (ActiveCancellation==job->Cancelled) ActiveCancellation.reset();
                    if (Stopping) return;
                    if (job->Generation!=Progress.RequestedGeneration || cancelled()) continue;
                    if (completed)
                    {
                        Progress.Published=std::move(completed); PublishedKey=job->Key;
                        Progress.State=VoxelUpdateState::Ready; Progress.Error.clear(); ++Progress.InjectionCount;
                    }
                    else
                    {
                        Progress.State=VoxelUpdateState::Failed;
                        Progress.Error=error.empty()?"Voxel worker did not produce a complete volume":std::move(error);
                    }
                }
            }
        }
        void CancelLocked()
        {
            if (ActiveCancellation) ActiveCancellation->store(true,std::memory_order_relaxed);
            Pending.reset(); RequestedKey.reset(); ++Progress.RequestedGeneration;
            Progress.State=VoxelUpdateState::Idle; Progress.Error.clear();
        }
    };
    VoxelRadianceCache::VoxelRadianceCache():State(std::make_unique<Impl>()) {}
    VoxelRadianceCache::~VoxelRadianceCache()=default;
    std::uint64_t VoxelRadianceCache::Request(VoxelRadianceRequest request)
    {
        UpdateKey key{request.SceneRevision,Techniques::SanitizeSettings(request.Materials),Techniques::SanitizeVoxelGISettings(request.Grid),request.Lighting,request.Shadows};
        std::lock_guard<std::mutex> lock(State->Mutex);
        if (State->RequestedKey && SameRadiance(*State->RequestedKey,key)) return State->Progress.RequestedGeneration;
        State->CancelLocked(); State->RequestedKey=key; const auto generation=State->Progress.RequestedGeneration;
        if (State->PublishedKey && State->Progress.Published && SameRadiance(*State->PublishedKey,key))
        {
            auto result=std::make_shared<VoxelRadianceResult>(*State->Progress.Published); result->Generation=generation;
            State->Progress.Published=std::move(result); State->Progress.State=VoxelUpdateState::Ready; return generation;
        }
        auto job=std::make_shared<Impl::Job>(); job->Key=std::move(key); job->Generation=generation; job->Instances=std::move(request.Instances);
        State->Pending=std::move(job); State->Progress.State=VoxelUpdateState::Queued;
        if (!State->Worker.joinable())
        {
            try { State->Worker=std::thread([impl=State.get()]{impl->Run();}); }
            catch (const std::exception& exception)
            {
                State->Pending.reset(); State->Progress.State=VoxelUpdateState::Failed;
                State->Progress.Error=std::string("Cannot start voxel worker: ")+exception.what(); return generation;
            }
        }
        State->Wake.notify_one(); return generation;
    }
    VoxelRadianceStatus VoxelRadianceCache::Status() const
    { std::lock_guard<std::mutex> lock(State->Mutex); return State->Progress; }
    void VoxelRadianceCache::Cancel()
    { std::lock_guard<std::mutex> lock(State->Mutex); State->CancelLocked(); }
    void VoxelRadianceCache::Reset()
    {
        std::lock_guard<std::mutex> lock(State->Mutex); State->CancelLocked();
        State->QueryKey.reset(); State->GeometryKey.reset(); State->PublishedKey.reset(); State->Query.reset(); State->Geometry.reset();
        State->Progress.Published.reset(); State->Progress.QueryBuildCount=State->Progress.VoxelBuildCount=State->Progress.InjectionCount=0;
    }
}
