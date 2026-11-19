#include "Renderer/Lumen/LumenSceneCache.h"
#include "Renderer/IRHI.h"

#include <cstring>
#include <tuple>

namespace Evans::Renderer
{
    namespace
    {
        bool Same(Float3 a,Float3 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z; }
        bool Same(Float4 a,Float4 b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z && a.W==b.W; }
        bool Same(LumenSurfaceIdentity a,LumenSurfaceIdentity b)
        { return a.Epoch==b.Epoch && a.Serial==b.Serial && a.Geometry==b.Geometry && a.Material==b.Material; }
        bool Same(LumenDistanceSceneIdentity a,LumenDistanceSceneIdentity b)
        { return a.Epoch==b.Epoch && a.Serial==b.Serial && a.Geometry==b.Geometry; }
        bool SameSurface(const LumenSurfaceUpload& a,const LumenSurfaceUpload& b)
        {
            if (a.CardSide!=b.CardSide || a.Cards.size()!=b.Cards.size() || a.Draws.size()!=b.Draws.size()) return false;
            for (std::size_t i=0;i<a.Cards.size();++i)
            {
                const auto& x=a.Cards[i]; const auto& y=b.Cards[i];
                if (x.ViewProjection.Values!=y.ViewProjection.Values || !Same(x.ViewOrigin,y.ViewOrigin) ||
                    x.FirstDraw!=y.FirstDraw || x.DrawCount!=y.DrawCount || x.ProjectionGroup!=y.ProjectionGroup) return false;
            }
            for (std::size_t i=0;i<a.Draws.size();++i)
            {
                const auto& x=a.Draws[i]; const auto& y=b.Draws[i];
                if (x.World.Values!=y.World.Values || !Same(x.BaseColor,y.BaseColor) || !Same(x.Emission,y.Emission) ||
                    std::tie(x.Vertices.Value,x.Indices.Value,x.FirstIndex,x.IndexCount,x.Roughness,x.Metallic,x.Cutoff,x.Diffuse.Value,x.Normal.Value,x.Opacity.Value,x.BackFaceCulling)!=
                    std::tie(y.Vertices.Value,y.Indices.Value,y.FirstIndex,y.IndexCount,y.Roughness,y.Metallic,y.Cutoff,y.Diffuse.Value,y.Normal.Value,y.Opacity.Value,y.BackFaceCulling)) return false;
            }
            return true;
        }
        struct Job
        {
            std::uint64_t Generation=0;
            unsigned IndirectBounces=0;
            std::shared_ptr<const LumenScenePlan> Scene;
            LumenSurfaceLightingUpload Lighting;
            LumenSurfaceLightingInput Input;
        };
        bool SameLight(const Job& a,const Job& b)
        {
            return a.IndirectBounces==b.IndirectBounces && a.Lighting.LightingRevision==b.Lighting.LightingRevision && a.Input.Words==b.Input.Words &&
                !std::memcmp(&a.Input.Frame,&b.Input.Frame,sizeof(a.Input.Frame));
        }
        bool SameScene(const Job& a,const Job& b)
        {
            return Same(a.Scene->Surface.Identity,b.Scene->Surface.Identity) && Same(a.Scene->Distance.Info.Identity,b.Scene->Distance.Info.Identity);
        }
        // Internal references allow a new material version to reuse geometry,
        // and a light edit to reuse both. Publication swaps all three at once.
        struct DistanceResource
        {
            IRHI& Rhi; LumenDistanceSceneHandle Handle;
            ~DistanceResource() { if (Handle) Rhi.DestroyLumenDistanceScene(Handle); }
        };
        struct SurfaceResource
        {
            IRHI& Rhi; LumenSurfaceHandle Handle;
            ~SurfaceResource() { if (Handle) Rhi.DestroyLumenSurface(Handle); }
        };
        struct LightingResource
        {
            IRHI& Rhi; LumenLightingHandle Handle;
            ~LightingResource() { if (Handle) Rhi.DestroyLumenLighting(Handle); }
        };
        struct Version
        {
            std::shared_ptr<const Job> Request;
            std::shared_ptr<DistanceResource> Distance;
            std::shared_ptr<SurfaceResource> Surface;
            std::shared_ptr<LightingResource> Lighting;
            unsigned CompletedBounces=0;
        };
    }
    struct LumenSceneCache::Impl
    {
        explicit Impl(IRHI& rhi):Rhi(rhi) {}
        IRHI& Rhi;
        std::shared_ptr<const Job> Requested;
        Version Published,Candidate;
        LumenSceneCacheStatus Progress;
        LumenLightingTask LightingTask;
        LumenCaptureTask CaptureTask;
        void CancelNativeUpdates()
        { if (LightingTask) { Rhi.CancelLumenLightingUpdate(LightingTask); LightingTask={}; }
          if (CaptureTask) { Rhi.CancelLumenCaptureUpdate(CaptureTask); CaptureTask={}; } }
        ~Impl() { CancelNativeUpdates(); }
    };
    LumenSceneCache::LumenSceneCache(IRHI& rhi):State(std::make_unique<Impl>(rhi)) {}
    LumenSceneCache::~LumenSceneCache()=default;
    bool LumenSceneCache::Request(std::shared_ptr<const LumenScenePlan> scene,LumenSurfaceLightingUpload lighting,std::string& error,unsigned indirectBounces)
    {
        if (!scene) { error="Lumen scene cache requires an immutable scene plan"; return false; }
        if (lighting.PreviousLighting || indirectBounces>LumenMaximumIndirectBounces || (indirectBounces && !lighting.IndirectRays))
        { error="Lumen scene scheduling owns the direct seed and requires bounded indirect iterations with samples"; return false; }
        auto request=std::make_shared<Job>(); request->IndirectBounces=indirectBounces; request->Scene=std::move(scene); auto& plan=*request->Scene;
        if (!ValidateLumenScenePlan(plan,error)) return false;
        lighting.CardInstances=plan.CardInstances; lighting.Readback=false; request->Lighting=std::move(lighting);
        if (!BuildLumenSurfaceLightingInput(plan.Surface.Identity,plan.Layout,plan.Surface.Cards,plan.Distance.Info,request->Lighting,request->Input,error)) return false;
        for (const auto& previous:{State->Requested,State->Published.Request,State->Candidate.Request}) if (previous)
        {
            const auto& old=*previous->Scene;
            if (old.Surface.Identity.Epoch==plan.Surface.Identity.Epoch && old.Surface.Identity.Geometry==plan.Surface.Identity.Geometry)
            {
                bool same=old.Parts.size()==plan.Parts.size();
                for (std::size_t i=0;same && i<old.Parts.size();++i)
                {
                    const auto& x=old.Parts[i]; const auto& y=plan.Parts[i];
                    same=x.InstanceKey==y.InstanceKey && x.Part==y.Part && x.Range.FirstCard==y.Range.FirstCard &&
                        x.Range.CardCount==y.Range.CardCount && x.Range.DistanceInstance==y.Range.DistanceInstance;
                }
                if (!same) { error="Lumen scene correspondence changed without advancing geometry identity"; return false; }
            }
            if ((Same(old.Surface.Identity,plan.Surface.Identity) && !SameSurface(old.Surface,plan.Surface)) ||
                (Same(old.Distance.Info.Identity,plan.Distance.Info.Identity) && old.Distance.Words!=plan.Distance.Words))
            { error="Lumen scene data changed without advancing its material or geometry/occlusion identity"; return false; }
            if (SameScene(*request,*previous) && request->Lighting.LightingRevision==previous->Lighting.LightingRevision && !SameLight(*request,*previous))
            { error="Lumen lighting parameters changed without advancing their identity"; return false; }
        }
        if (State->Requested && SameScene(*request,*State->Requested) && SameLight(*request,*State->Requested)) { error.clear(); return true; }
        if (State->Progress.RequestedGeneration==UINT64_MAX) { error="Lumen scene generation exhausted"; return false; }
        request->Generation=++State->Progress.RequestedGeneration; State->Requested=std::move(request);
        State->Progress.State=LumenSceneUpdateState::Distance; State->Progress.Error.clear(); error.clear(); return true;
    }
    bool LumenSceneCache::Advance(unsigned stageBudget,unsigned lightingTileBudget,unsigned captureCardBudget)
    {
        auto& state=*State; auto& status=state.Progress;
        if (!stageBudget) return status.State!=LumenSceneUpdateState::Failed;
        if (!state.Requested) { state.Candidate={}; return true; }
        if (status.State==LumenSceneUpdateState::Failed || status.State==LumenSceneUpdateState::Ready) return status.State==LumenSceneUpdateState::Ready;
        // Coalesce light-only edits while a frozen generation finishes. Repeated
        // edits must not restart the first tile forever. Scene edits cancel it.
        const bool finishLighting=(state.LightingTask || state.CaptureTask || state.Candidate.Lighting || lightingTileBudget || captureCardBudget) &&
            state.Candidate.Request && SameScene(*state.Candidate.Request,*state.Requested);
        if (state.Candidate.Request!=state.Requested && !finishLighting)
        {
            state.CancelNativeUpdates(); status.LightingTiles={}; status.CaptureCards={};
            state.Candidate={}; state.Candidate.Request=state.Requested;
            if (state.Published.Request)
            {
                const auto& old=*state.Published.Request; const auto& next=*state.Requested;
                if (Same(old.Scene->Distance.Info.Identity,next.Scene->Distance.Info.Identity)) state.Candidate.Distance=state.Published.Distance;
                if (Same(old.Scene->Surface.Identity,next.Scene->Surface.Identity)) state.Candidate.Surface=state.Published.Surface;
                if (SameScene(old,next) && SameLight(old,next))
                { state.Candidate.Lighting=state.Published.Lighting; state.Candidate.CompletedBounces=state.Published.CompletedBounces; }
            }
        }
        auto fail=[&]() { status.State=LumenSceneUpdateState::Failed; status.Error=state.Rhi.GetLastError(); state.CancelNativeUpdates(); state.Candidate={}; return false; };
        auto& version=state.Candidate; const auto& job=*version.Request;
        if (!version.Distance)
        {
            auto resource=std::shared_ptr<DistanceResource>(new DistanceResource{state.Rhi,{}});
            if (!state.Rhi.CreateLumenDistanceScene(job.Scene->Distance,resource->Handle)) return fail();
            version.Distance=std::move(resource); ++status.DistanceUploads; --stageBudget;
        }
        status.State=LumenSceneUpdateState::Capture;
        if (!version.Surface)
        {
            if (!stageBudget) return true;
            auto upload=job.Scene->Surface; upload.Readback=false; LumenSurfaceResult result;
            if (captureCardBudget || state.CaptureTask)
            {
                if (!state.CaptureTask && !state.Rhi.BeginLumenCaptureUpdate(upload,state.CaptureTask)) return fail();
                bool ready=false;
                if (!state.Rhi.AdvanceLumenCaptureUpdate(state.CaptureTask,captureCardBudget,ready,result,status.CaptureCards)) return fail();
                if (!ready) return true;
                state.CaptureTask={};
            }
            else if (!state.Rhi.CaptureLumenSurface(upload,result)) return fail();
            version.Surface=std::shared_ptr<SurfaceResource>(new SurfaceResource{state.Rhi,result.Handle}); ++status.MaterialCaptures; --stageBudget;
        }
        status.State=LumenSceneUpdateState::Lighting;
        while (!version.Lighting || version.CompletedBounces<job.IndirectBounces)
        {
            if (!stageBudget) return true;
            auto upload=job.Lighting;
            upload.PreviousLighting=version.Lighting?version.Lighting->Handle:LumenLightingHandle{};
            upload.IndirectSequence+=version.CompletedBounces;
            LumenSurfaceLightingResult result;
            if (lightingTileBudget || state.LightingTask)
            {
                if (!state.LightingTask && !state.Rhi.BeginLumenLightingUpdate(version.Surface->Handle,version.Distance->Handle,upload,state.LightingTask)) return fail();
                bool ready=false;
                if (!state.Rhi.AdvanceLumenLightingUpdate(state.LightingTask,lightingTileBudget,ready,result,status.LightingTiles)) return fail();
                if (!ready) return true;
                state.LightingTask={};
            }
            else if (!state.Rhi.RelightLumenSurface(version.Surface->Handle,version.Distance->Handle,upload,result)) return fail();
            version.Lighting=std::shared_ptr<LightingResource>(new LightingResource{state.Rhi,result.Handle});
            version.CompletedBounces=result.Info.Bounce; ++status.Relights; --stageBudget;
        }
        status.PublishedGeneration=job.Generation;
        state.Published=std::move(version); version={};
        status.State=status.PublishedGeneration==status.RequestedGeneration?LumenSceneUpdateState::Ready:LumenSceneUpdateState::Lighting;
        return true;
    }
    void LumenSceneCache::Cancel()
    { State->CancelNativeUpdates(); State->Candidate={}; State->Requested.reset(); State->Progress.State=LumenSceneUpdateState::Idle; State->Progress.LightingTiles={}; State->Progress.CaptureCards={}; State->Progress.Error.clear(); }
    void LumenSceneCache::Reset()
    { State->CancelNativeUpdates(); State->Candidate={}; State->Published={}; State->Requested.reset(); State->Progress={}; }
    LumenSceneCacheStatus LumenSceneCache::Status() const
    {
        auto result=State->Progress; const auto& published=State->Published; result.Building=bool(State->Candidate.Request);
        const auto& active=State->Candidate.Request?State->Candidate:published;
        result.CompletedBounces=active.CompletedBounces; result.TargetBounces=active.Request?active.Request->IndirectBounces:0;
        if (published.Distance) result.Distance=published.Distance->Handle;
        if (published.Surface) result.Surface=published.Surface->Handle;
        if (published.Lighting) result.Lighting=published.Lighting->Handle;
        result.Usable=State->Requested && published.Request && published.Lighting && SameScene(*State->Requested,*published.Request);
        return result;
    }
}
