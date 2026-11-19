#include "Renderer/Nanite/NaniteStream.h"
#include "Renderer/Nanite/NaniteAsset.h"
#include "Renderer/IRHI.h"

#include <optional>

namespace Evans::Renderer
{
    namespace
    {
        constexpr float AdmittedPagePriority=1;
        struct CacheLease
        {
            IRHI& RHI; NaniteCacheHandle Handle;
            explicit CacheLease(IRHI& rhi):RHI(rhi) {}
            ~CacheLease() { if (Handle) RHI.DestroyNaniteCache(Handle); }
        };
        struct GeometryLease
        {
            IRHI& RHI; NaniteGeometryBuffers Buffers;
            explicit GeometryLease(IRHI& rhi):RHI(rhi) {}
            ~GeometryLease()
            {
                if (Buffers.VertexBuffer) RHI.DestroyBuffer(Buffers.VertexBuffer);
                if (Buffers.IndexBuffer) RHI.DestroyBuffer(Buffers.IndexBuffer);
            }
        };
        struct PendingTarget
        {
            VirtualGeometryResidency Residency;
            std::shared_ptr<CacheLease> Cache;
            std::vector<std::uint32_t> Pages;
        };
        unsigned Missing(const PendingTarget& target)
        {
            unsigned count=0;
            for (auto page:target.Pages) count+=target.Residency.PageSlots()[page]==NaniteInvalid;
            return count;
        }
        bool NativeResult(IRHI& rhi,bool result,std::string& error)
        { if (!result) error=std::string(rhi.GetName())+": "+rhi.GetLastError(); return result; }
        bool ResidentPagesReady(const NanitePages& pages,const VirtualGeometryResidency& residency,bool& ready,std::string& error)
        {
            std::vector<unsigned> wanted;
            for (const auto& slot:residency.Slots()) if (slot.Page!=NaniteInvalid) wanted.push_back(slot.Page);
            return PrefetchNanitePages(pages,wanted,ready,error);
        }
    }
    struct NaniteStreamVersion
    {
        VirtualGeometryResidency Residency;
        std::shared_ptr<CacheLease> Cache;
        std::unique_ptr<GeometryLease> Geometry;
        NaniteGpuUpload Upload;
        NaniteResidentCut Feedback;
    };
    namespace
    {
        bool Decode(IRHI& rhi,const NaniteHierarchy& h,const NanitePages& pages,const std::vector<LodView>& views,
            NaniteStreamVersion& version,std::string& error)
        {
            auto geometry=std::make_unique<GeometryLease>(rhi);
            if (!NativeResult(rhi,rhi.CreateCachedNaniteGeometry(version.Cache->Handle,version.Upload,geometry->Buffers),error) ||
                !AdmitNaniteGpuFeedback(h,pages,version.Residency,views,version.Upload,geometry->Buffers.Readback.State,version.Feedback,error)) return false;
            version.Geometry=std::move(geometry); return true;
        }
    }
    struct NanitePendingDecode
    {
        IRHI& RHI; NaniteGeometryTask Task;
        std::shared_ptr<NaniteStreamVersion> Version;
        std::vector<LodView> Views; bool PublishesPool=false;
        explicit NanitePendingDecode(IRHI& rhi):RHI(rhi) {}
        ~NanitePendingDecode() { if (Task) RHI.CancelNaniteGeometry(Task); }
    };
    struct NanitePendingPages
    {
        IRHI& RHI; NaniteCacheTask Task; VirtualGeometryResidency Residency;
        explicit NanitePendingPages(IRHI& rhi):RHI(rhi) {}
        ~NanitePendingPages() { if (Task) RHI.CancelNaniteCache(Task); }
    };
    struct NaniteStream::State
    {
        std::shared_ptr<NaniteStreamVersion> Active;
        std::optional<PendingTarget> Target;
        std::unique_ptr<NanitePendingDecode> Producer;
        std::unique_ptr<NanitePendingPages> PageCopy;
        std::uint64_t LastFrame=0;
    };
    struct NaniteStream::Bootstrap
    {
        IRHI& RHI; std::shared_ptr<NaniteStreamVersion> Version;
        std::vector<LodView> Views; unsigned RequestedSlots=0;
        NaniteCacheTask Pages; NaniteGeometryTask Decode;
        explicit Bootstrap(IRHI& rhi):RHI(rhi) {}
        ~Bootstrap() { if (Decode) RHI.CancelNaniteGeometry(Decode); if (Pages) RHI.CancelNaniteCache(Pages); }
    };
    NaniteStream::NaniteStream(IRHI& rhi):m_RHI(rhi) {}
    NaniteStream::~NaniteStream()=default;
    bool NaniteStream::Ready() const noexcept { return bool(m_State); }
    bool NaniteStream::Pending() const noexcept { return m_Bootstrap || (m_State && (m_State->Target || m_State->Producer || m_State->PageCopy)); }
    bool NaniteStream::Producing() const noexcept
    { return (m_Bootstrap && (m_Bootstrap->Pages || m_Bootstrap->Decode)) || (m_State && (m_State->Producer || m_State->PageCopy)); }
    const VirtualGeometryResidency& NaniteStream::Residency() const { return m_State->Active->Residency; }
    const NaniteGeometryBuffers& NaniteStream::Geometry() const { return m_State->Active->Geometry->Buffers; }
    const NaniteGpuUpload& NaniteStream::Upload() const { return m_State->Active->Upload; }
    const NaniteResidentCut& NaniteStream::Feedback() const { return m_State->Active->Feedback; }
    NaniteStreamSnapshot NaniteStream::Snapshot() const noexcept
    { NaniteStreamSnapshot snapshot; if (m_State) snapshot.m_Version=m_State->Active; return snapshot; }
    const VirtualGeometryResidency& NaniteStreamSnapshot::Residency() const { return m_Version->Residency; }
    const NaniteGeometryBuffers& NaniteStreamSnapshot::Geometry() const { return m_Version->Geometry->Buffers; }
    const NaniteGpuUpload& NaniteStreamSnapshot::Upload() const { return m_Version->Upload; }
    const NaniteResidentCut& NaniteStreamSnapshot::Feedback() const { return m_Version->Feedback; }
    void NaniteStream::CancelPending() { m_Bootstrap.reset(); if (m_State) { m_State->Producer.reset(); m_State->PageCopy.reset(); m_State->Target.reset(); } }

    bool NaniteStream::Initialize(const NaniteHierarchy& h,const NanitePages& pages,unsigned physicalSlots,
        const std::vector<LodView>& views,double pixels,const GeometryPreparationLimits& limits,NaniteStreamingStatistics& statistics,std::string& error)
    {
        if (Ready() || m_Bootstrap) { error="Nanite stream already has an active or initializing source"; return false; }
        auto candidate=std::make_unique<State>(); candidate->Active=std::make_shared<NaniteStreamVersion>(); auto& active=*candidate->Active;
        if (!CreateNaniteResidency(h,pages,physicalSlots,active.Residency,error) ||
            !PrepareNaniteGpu(h,pages,active.Residency,views,pixels,limits,active.Upload,error)) return false;
        NaniteStreamingStatistics result; active.Cache=std::make_shared<CacheLease>(m_RHI);
        if (!NativeResult(m_RHI,m_RHI.PrepareNaniteCache(active.Upload,{},active.Cache->Handle,result.Native),error) ||
            !Decode(m_RHI,h,pages,views,active,error)) return false;
        result.PoolPublished=result.CutChanged=true;
        result.ReadbackBytes=active.Geometry->Buffers.ReadbackByteCount();
        m_State=std::move(candidate); statistics=result; error.clear(); return true;
    }

    bool NaniteStream::InitializeAsync(const NaniteHierarchy& h,const NanitePages& pages,unsigned physicalSlots,
        const std::vector<LodView>& views,double pixels,const GeometryPreparationLimits& limits,NaniteStreamingStatistics& statistics,std::string& error)
    {
        if (Ready()) { error="Nanite stream already has an active source"; return false; }
        NaniteStreamingStatistics result;
        if (!m_Bootstrap)
        {
            auto pending=std::make_unique<Bootstrap>(m_RHI); pending->RequestedSlots=physicalSlots;
            pending->Version=std::make_shared<NaniteStreamVersion>(); auto& version=*pending->Version;
            if (!CreateNaniteResidency(h,pages,physicalSlots,version.Residency,error)) return false;
            version.Cache=std::make_shared<CacheLease>(m_RHI);
            m_Bootstrap=std::move(pending);
        }
        auto& pending=*m_Bootstrap; auto& version=*pending.Version;
        if (pages.Signature!=version.Residency.Source() || h.Signature!=pages.HierarchySource || physicalSlots!=pending.RequestedSlots)
        { error="Nanite bootstrap source or capacity changed; cancel it before replacement"; return false; }
        bool diskReady=false;
        if (!ResidentPagesReady(pages,version.Residency,diskReady,error)) return false;
        if (!diskReady)
        {
            result.PageReadPending=true; result.PendingPages=unsigned(pages.PinnedPages.size());
            statistics=result; error.clear(); return true;
        }
        NaniteGpuUpload requested;
        if (!PrepareNaniteGpu(h,pages,version.Residency,views,pixels,limits,requested,error)) return false;
        if (!version.Cache->Handle && !pending.Pages)
        {
            version.Upload=std::move(requested);
            if (!NativeResult(m_RHI,m_RHI.SubmitNaniteCache(version.Upload,{},pending.Pages,result.Native),error)) return false;
            result.PageCopyPending=true; result.PendingPages=unsigned(pages.PinnedPages.size());
            statistics=result; error.clear(); return true;
        }
        if (pending.Pages)
        {
            bool ready=false;
            if (!NativeResult(m_RHI,m_RHI.PollNaniteCache(pending.Pages,ready,version.Cache->Handle),error))
            {
                if (m_RHI.CancelNaniteCache(pending.Pages)) { pending.Pages={}; m_Bootstrap.reset(); }
                return false;
            }
            if (!ready)
            {
                result.PageCopyPending=true; result.PendingPages=unsigned(pages.PinnedPages.size());
                statistics=result; error.clear(); return true;
            }
            pending.Pages={};
        }
        if (!pending.Decode)
        {
            // Page upload is reusable after a failed decode. Capture the current
            // view only when submitting its immutable producer transaction.
            version.Upload=std::move(requested); pending.Views=views;
            if (!NativeResult(m_RHI,m_RHI.SubmitNaniteGeometry(version.Cache->Handle,version.Upload,pending.Decode),error)) return false;
            result.ProducerPending=true; statistics=result; error.clear(); return true;
        }
        auto geometry=std::make_unique<GeometryLease>(m_RHI); bool ready=false;
        if (!NativeResult(m_RHI,m_RHI.PollNaniteGeometry(pending.Decode,ready,geometry->Buffers),error))
        {
            if (m_RHI.CancelNaniteGeometry(pending.Decode)) pending.Decode={};
            return false;
        }
        if (!ready) { result.ProducerPending=true; statistics=result; error.clear(); return true; }
        pending.Decode={};
        if (!AdmitNaniteGpuFeedback(h,pages,version.Residency,pending.Views,version.Upload,geometry->Buffers.Readback.State,version.Feedback,error)) return false;
        result.ReadbackBytes=geometry->Buffers.ReadbackByteCount();
        version.Geometry=std::move(geometry); auto state=std::make_unique<State>(); state->Active=pending.Version;
        result.PoolPublished=result.CutChanged=true; result.ProducerPending=version.Upload.Signature!=requested.Signature;
        m_State=std::move(state); m_Bootstrap.reset(); statistics=result; error.clear(); return true;
    }

    bool NaniteStream::Update(const NaniteHierarchy& h,const NanitePages& pages,const std::vector<LodView>& views,
        double pixels,const GeometryPreparationLimits& limits,std::uint64_t frame,unsigned maximumUploads,
        NaniteStreamingStatistics& statistics,std::string& error)
    {
        if (!Ready() || pages.Signature!=m_State->Active->Residency.Source() || h.Signature!=pages.HierarchySource)
        { error="Nanite stream has no active source or dataset identity changed"; return false; }
        auto& state=*m_State;
        if (state.Producer || state.PageCopy) { error="Cancel asynchronous Nanite production before synchronous updates"; return false; }
        if (frame<state.LastFrame || (state.Target && frame<state.Target->Residency.Frame()))
        { error="Nanite stream frame regressed"; return false; }
        auto candidate=std::make_shared<NaniteStreamVersion>(); candidate->Residency=state.Active->Residency; candidate->Cache=state.Active->Cache;
        const auto& previous=state.Active->Geometry->Buffers.Selection.Clusters;
        if (!PrepareNaniteGpu(h,pages,candidate->Residency,views,pixels,limits,candidate->Upload,error) ||
            !Decode(m_RHI,h,pages,views,*candidate,error)) return false;
        if (!state.Target)
        {
            // GPU error feedback determines demand; this CPU admission check
            // proves that the complete ancestor/group bundle fits the cache.
            VirtualGeometryResidencyPlan full;
            if (!PlanNaniteResidency(h,pages,candidate->Residency,candidate->Feedback,frame,full,error)) return false;
            PendingTarget target{candidate->Residency,candidate->Cache,candidate->Feedback.TargetPages};
            if (Missing(target)) state.Target=std::move(target);
        }
        NaniteStreamingStatistics result; result.ReadbackBytes=candidate->Geometry->Buffers.ReadbackByteCount();
        if (state.Target)
        {
            auto& target=*state.Target;
            if (maximumUploads && Missing(target))
            {
                std::vector<VirtualGeometryPageRequest> requests;
                for (auto page:target.Pages) requests.push_back({page,AdmittedPagePriority});
                VirtualGeometryResidencyPlan step;
                if (!PlanVirtualGeometryResidency(target.Residency,requests,frame,maximumUploads,step,error)) return false;
                auto staged=target.Residency;
                if (!CommitVirtualGeometryResidency(staged,std::move(step),error)) return false;
                NaniteGpuUpload upload;
                if (!PrepareNaniteGpu(h,pages,staged,views,pixels,limits,upload,error)) return false;
                auto cache=std::make_shared<CacheLease>(m_RHI);
                if (!NativeResult(m_RHI,m_RHI.PrepareNaniteCache(upload,target.Cache->Handle,cache->Handle,result.Native),error)) return false;
                target.Residency=std::move(staged); target.Cache=std::move(cache);
            }
            result.PendingPages=Missing(target);
            if (!result.PendingPages)
            {
                auto complete=std::make_shared<NaniteStreamVersion>(); complete->Residency=target.Residency; complete->Cache=target.Cache;
                if (!PrepareNaniteGpu(h,pages,complete->Residency,views,pixels,limits,complete->Upload,error) ||
                    !Decode(m_RHI,h,pages,views,*complete,error)) return false;
                result.ReadbackBytes+=complete->Geometry->Buffers.ReadbackByteCount();
                candidate=std::move(complete); state.Target.reset(); result.PoolPublished=true;
            }
        }
        result.CutChanged=previous!=candidate->Geometry->Buffers.Selection.Clusters;
        state.Active=std::move(candidate); state.LastFrame=frame;
        statistics=result; error.clear(); return true;
    }
    bool NaniteStream::UpdateAsync(const NaniteHierarchy& h,const NanitePages& pages,const std::vector<LodView>& views,
        double pixels,const GeometryPreparationLimits& limits,std::uint64_t frame,unsigned maximumUploads,
        NaniteStreamingStatistics& statistics,std::string& error)
    {
        if (!Ready() || pages.Signature!=m_State->Active->Residency.Source() || h.Signature!=pages.HierarchySource)
        { error="Nanite stream has no active source or dataset identity changed"; return false; }
        auto& state=*m_State;
        if (frame<state.LastFrame || (state.Target && frame<state.Target->Residency.Frame()))
        { error="Nanite stream frame regressed"; return false; }
        // Validate new inputs before consuming an already completed transaction.
        NaniteStreamingStatistics result; bool diskReady=false;
        if (!ResidentPagesReady(pages,state.Active->Residency,diskReady,error)) return false;
        if (!diskReady)
        {
            result.PageReadPending=true; state.LastFrame=frame; statistics=result; error.clear(); return true;
        }
        NaniteGpuUpload requested;
        if (!PrepareNaniteGpu(h,pages,state.Active->Residency,views,pixels,limits,requested,error)) return false;
        if (state.PageCopy)
        {
            auto& pending=*state.PageCopy; auto cache=std::make_shared<CacheLease>(m_RHI); bool ready=false;
            if (!NativeResult(m_RHI,m_RHI.PollNaniteCache(pending.Task,ready,cache->Handle),error))
            {
                if (m_RHI.CancelNaniteCache(pending.Task)) { pending.Task={}; state.PageCopy.reset(); }
                return false;
            }
            if (!ready)
            {
                result.PageCopyPending=true; result.PendingPages=Missing(*state.Target);
                state.LastFrame=frame; statistics=result; error.clear(); return true;
            }
            pending.Task={}; state.Target->Residency=std::move(pending.Residency); state.Target->Cache=std::move(cache);
            state.PageCopy.reset();
        }
        if (state.Producer)
        {
            auto& producer=*state.Producer; auto geometry=std::make_unique<GeometryLease>(m_RHI); bool ready=false;
            if (!NativeResult(m_RHI,m_RHI.PollNaniteGeometry(producer.Task,ready,geometry->Buffers),error))
            {
                // A closed-frame/thread precondition failure must not lose the
                // task. Completed invalid output can be cancelled and retried.
                if (m_RHI.CancelNaniteGeometry(producer.Task)) { producer.Task={}; state.Producer.reset(); }
                return false;
            }
            if (ready)
            {
                producer.Task={}; auto candidate=producer.Version;
                if (!AdmitNaniteGpuFeedback(h,pages,candidate->Residency,producer.Views,candidate->Upload,geometry->Buffers.Readback.State,candidate->Feedback,error))
                { state.Producer.reset(); return false; }
                result.CutChanged=state.Active->Geometry->Buffers.Selection.Clusters!=geometry->Buffers.Selection.Clusters;
                result.PoolPublished=producer.PublishesPool; candidate->Geometry=std::move(geometry);
                result.ReadbackBytes=candidate->Geometry->Buffers.ReadbackByteCount();
                state.Active=std::move(candidate); state.Producer.reset(); if (result.PoolPublished) state.Target.reset();
                // A moving camera may already require another request. Keep the
                // editor rendering until the latest view has been evaluated.
                result.ProducerPending=state.Active->Upload.Signature!=requested.Signature;
            }
            else result.ProducerPending=true;
            result.PendingPages=state.Target?Missing(*state.Target):0;
            state.LastFrame=frame; statistics=result; error.clear(); return true;
        }
        if (!state.Target)
        {
            VirtualGeometryResidencyPlan full;
            if (!PlanNaniteResidency(h,pages,state.Active->Residency,state.Active->Feedback,frame,full,error)) return false;
            PendingTarget target{state.Active->Residency,state.Active->Cache,state.Active->Feedback.TargetPages};
            if (Missing(target)) state.Target=std::move(target);
        }
        auto candidate=std::make_shared<NaniteStreamVersion>(); bool targetReady=false;
        if (state.Target)
        {
            auto& target=*state.Target;
            if (maximumUploads && Missing(target))
            {
                std::vector<VirtualGeometryPageRequest> requests; for (auto page:target.Pages) requests.push_back({page,AdmittedPagePriority});
                VirtualGeometryResidencyPlan step;
                if (!PlanVirtualGeometryResidency(target.Residency,requests,frame,maximumUploads,step,error)) return false;
                auto staged=target.Residency; if (!CommitVirtualGeometryResidency(staged,std::move(step),error)) return false;
                if (!ResidentPagesReady(pages,staged,diskReady,error)) return false;
                if (!diskReady)
                {
                    result.PageReadPending=true; result.PendingPages=Missing(target);
                    state.LastFrame=frame; statistics=result; error.clear(); return true;
                }
                NaniteGpuUpload upload; if (!PrepareNaniteGpu(h,pages,staged,views,pixels,limits,upload,error)) return false;
                // Account uploads when submitted, but admit the new mapping only
                // after the GPU copy completes. The previous pool stays immutable.
                auto pending=std::make_unique<NanitePendingPages>(m_RHI); pending->Residency=std::move(staged);
                if (!NativeResult(m_RHI,m_RHI.SubmitNaniteCache(upload,target.Cache->Handle,pending->Task,result.Native),error)) return false;
                state.PageCopy=std::move(pending); result.PageCopyPending=true; result.PendingPages=Missing(target);
                state.LastFrame=frame; statistics=result; error.clear(); return true;
            }
            result.PendingPages=Missing(target); targetReady=result.PendingPages==0;
            if (targetReady)
            {
                candidate->Residency=target.Residency; candidate->Cache=target.Cache;
                if (!PrepareNaniteGpu(h,pages,candidate->Residency,views,pixels,limits,candidate->Upload,error)) return false;
            }
        }
        if (!targetReady)
        {
            // Finish an admitted bundle before another view-only request, unless
            // uploads are paused. This avoids continuous camera-motion starvation.
            if ((state.Target && maximumUploads) || requested.Signature==state.Active->Upload.Signature)
            { state.LastFrame=frame; statistics=result; error.clear(); return true; }
            candidate->Residency=state.Active->Residency; candidate->Cache=state.Active->Cache; candidate->Upload=std::move(requested);
        }
        auto producer=std::make_unique<NanitePendingDecode>(m_RHI); producer->Version=std::move(candidate); producer->Views=views; producer->PublishesPool=targetReady;
        if (!NativeResult(m_RHI,m_RHI.SubmitNaniteGeometry(producer->Version->Cache->Handle,producer->Version->Upload,producer->Task),error)) return false;
        state.Producer=std::move(producer); state.LastFrame=frame; result.ProducerPending=true;
        statistics=result; error.clear(); return true;
    }

}
