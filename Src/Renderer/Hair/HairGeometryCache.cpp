#include "Renderer/Hair/HairGeometryCache.h"

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
        struct GroomKey
        {
            std::uint64_t Revision=0,Selection=0;
            Techniques::Settings Materials;
            Techniques::HairFurSettings Shape;
        };
        bool SameSource(const GroomKey& a,const GroomKey& b)
        {
            const auto& x=a.Materials; const auto& y=b.Materials;
            return a.Revision==b.Revision && a.Selection==b.Selection && a.Shape.UseSelectedObject==b.Shape.UseSelectedObject &&
                x.TextureMapping==y.TextureMapping && x.NormalMapping==y.NormalMapping && x.AlphaTesting==y.AlphaTesting &&
                x.AlphaBlending==y.AlphaBlending && x.Shading==y.Shading && x.Metallic==y.Metallic;
        }
        bool SameGroom(const GroomKey& a,const GroomKey& b)
        {
            const auto& x=a.Shape; const auto& y=b.Shape;
            return SameSource(a,b) && x.StrandCount==y.StrandCount && x.Segments==y.Segments && x.Seed==y.Seed &&
                x.LengthFraction==y.LengthFraction && x.RadiusRatio==y.RadiusRatio && x.Bend==y.Bend &&
                x.LengthVariation==y.LengthVariation && x.RootOffset==y.RootOffset;
        }
    }
    struct HairGeometryCache::Impl
    {
        struct Job
        {
            GroomKey Key;
            std::uint64_t Generation=0;
            std::vector<HairSourceInstance> Instances;
            std::shared_ptr<std::atomic<bool>> Cancelled=std::make_shared<std::atomic<bool>>(false);
        };
        mutable std::mutex Mutex;
        std::condition_variable Wake;
        std::thread Worker;
        bool Stopping=false;
        std::shared_ptr<Job> Pending;
        std::shared_ptr<std::atomic<bool>> ActiveCancellation;
        std::optional<GroomKey> RequestedKey,QueryKey,PublishedKey;
        std::shared_ptr<const SceneRayQuery> Query;
        HairGeometryStatus Progress;
        ~Impl()
        {
            {
                std::lock_guard<std::mutex> lock(Mutex); Stopping=true;
                if (ActiveCancellation) ActiveCancellation->store(true,std::memory_order_relaxed);
                Pending.reset();
            }
            Wake.notify_one(); if (Worker.joinable()) Worker.join();
        }
        void CancelLocked()
        {
            if (ActiveCancellation) ActiveCancellation->store(true,std::memory_order_relaxed);
            Pending.reset(); RequestedKey.reset(); ++Progress.RequestedGeneration;
            Progress.State=HairUpdateState::Idle; Progress.Error.clear();
        }
        void Run()
        {
            for (;;)
            {
                std::shared_ptr<Job> job; std::shared_ptr<const SceneRayQuery> query;
                {
                    std::unique_lock<std::mutex> lock(Mutex); Wake.wait(lock,[&]{return Stopping || Pending;});
                    if (Stopping) return;
                    job=std::move(Pending); ActiveCancellation=job->Cancelled;
                    if (QueryKey && SameSource(*QueryKey,job->Key)) query=Query;
                    Progress.State=HairUpdateState::Preparing;
                }
                const auto cancelled=[&]{return job->Cancelled->load(std::memory_order_relaxed);};
                std::string error; std::shared_ptr<const HairStrandGeometry> groom;
                try
                {
                    if (!query && !cancelled() && !job->Instances.empty())
                    {
                        std::vector<SceneRayMeshView> views; views.reserve(job->Instances.size()); bool valid=true;
                        for (const auto& instance:job->Instances)
                        {
                            if (!instance.Geometry) { error="Hair source instance has no immutable geometry"; valid=false; break; }
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
                    if (!cancelled() && (query || job->Instances.empty()))
                    {
                        {
                            std::lock_guard<std::mutex> lock(Mutex);
                            if (job->Generation==Progress.RequestedGeneration && !cancelled()) Progress.State=HairUpdateState::Growing;
                        }
                        auto candidate=std::make_shared<HairStrandGeometry>();
                        if (!query || BuildHairStrands(*query,job->Key.Shape,*candidate,error,cancelled)) groom=std::move(candidate);
                    }
                }
                catch (const std::exception& exception) { error=std::string("Hair geometry worker failed: ")+exception.what(); }
                catch (...) { error="Hair geometry worker failed with an unknown exception"; }
                {
                    std::lock_guard<std::mutex> lock(Mutex);
                    if (Stopping) return;
                    if (cancelled() || job->Generation!=Progress.RequestedGeneration) continue;
                    if (groom)
                    {
                        auto result=std::make_shared<HairGeometryResult>(); result->Generation=job->Generation; result->Geometry=std::move(groom);
                        Progress.Published=std::move(result); PublishedKey=job->Key; ++Progress.GroomBuildCount;
                        Progress.State=HairUpdateState::Ready; Progress.Error.clear();
                    }
                    else
                    { Progress.State=HairUpdateState::Failed; Progress.Error=error.empty()?"Hair worker produced no complete geometry":std::move(error); }
                }
            }
        }
    };
    HairGeometryCache::HairGeometryCache():State(std::make_unique<Impl>()) {}
    HairGeometryCache::~HairGeometryCache()=default;
    std::uint64_t HairGeometryCache::Request(HairGeometryRequest request)
    {
        GroomKey key{request.SceneRevision,request.Groom.UseSelectedObject?request.SelectionGroup:0,Techniques::SanitizeSettings(request.Materials),Techniques::SanitizeHairFurSettings(request.Groom)};
        std::lock_guard<std::mutex> lock(State->Mutex);
        if (State->RequestedKey && SameGroom(*State->RequestedKey,key)) return State->Progress.RequestedGeneration;
        State->CancelLocked(); State->RequestedKey=key; const auto generation=State->Progress.RequestedGeneration;
        if (State->PublishedKey && State->Progress.Published && SameGroom(*State->PublishedKey,key))
        {
            auto result=std::make_shared<HairGeometryResult>(*State->Progress.Published); result->Generation=generation;
            State->Progress.Published=std::move(result); State->Progress.State=HairUpdateState::Ready; return generation;
        }
        auto job=std::make_shared<Impl::Job>(); job->Key=std::move(key); job->Generation=generation; job->Instances=std::move(request.Instances);
        State->Pending=std::move(job); State->Progress.State=HairUpdateState::Queued;
        if (!State->Worker.joinable())
        {
            try { State->Worker=std::thread([impl=State.get()]{impl->Run();}); }
            catch (const std::exception& exception)
            { State->Pending.reset(); State->Progress.State=HairUpdateState::Failed; State->Progress.Error=std::string("Cannot start hair worker: ")+exception.what(); return generation; }
        }
        State->Wake.notify_one(); return generation;
    }
    HairGeometryStatus HairGeometryCache::Status() const
    { std::lock_guard<std::mutex> lock(State->Mutex); return State->Progress; }
    void HairGeometryCache::Cancel()
    { std::lock_guard<std::mutex> lock(State->Mutex); State->CancelLocked(); }
    void HairGeometryCache::Reset()
    {
        std::lock_guard<std::mutex> lock(State->Mutex); State->CancelLocked();
        State->Query.reset(); State->QueryKey.reset(); State->PublishedKey.reset(); State->Progress.Published.reset();
        State->Progress.QueryBuildCount=State->Progress.GroomBuildCount=0;
    }
}
