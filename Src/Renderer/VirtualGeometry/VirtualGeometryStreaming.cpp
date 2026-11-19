#include "Renderer/VirtualGeometry/VirtualGeometryStreaming.h"
#include "Renderer/IRHI.h"

#include <optional>

namespace Evans::Renderer
{
    namespace
    {
        constexpr float StreamDemandPriority=1;
        struct CacheLease
        {
            IRHI& RHI;
            VirtualGeometryCacheHandle Handle;
            explicit CacheLease(IRHI& rhi):RHI(rhi) {}
            ~CacheLease() { if (Handle) RHI.DestroyVirtualGeometryCache(Handle); }
        };
        struct GeometryLease
        {
            IRHI& RHI;
            VirtualGeometryBuffers Buffers;
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
        std::uint32_t Missing(const PendingTarget& target)
        {
            std::uint32_t count=0;
            for (auto page:target.Pages) count+=target.Residency.PageSlots()[page]==VirtualGeometryInvalid;
            return count;
        }
        bool NativeResult(IRHI& rhi,bool result,std::string& error)
        {
            if (!result) error=std::string(rhi.GetName())+": "+rhi.GetLastError();
            return result;
        }
    }

    struct VirtualGeometryStreamVersion
    {
        VirtualGeometryResidency Residency;
        std::shared_ptr<CacheLease> Cache;
        std::unique_ptr<GeometryLease> Geometry;
        VirtualGeometryGpuUpload Upload;
        VirtualGeometryCut Feedback;
    };

    namespace
    {
        bool Decode(IRHI& rhi,const VirtualGeometryHierarchy& h,VirtualGeometryStreamVersion& version,std::string& error)
        {
            auto geometry=std::make_unique<GeometryLease>(rhi);
            if (!NativeResult(rhi,rhi.CreateCachedVirtualGeometry(version.Cache->Handle,version.Upload,geometry->Buffers),error) ||
                !AdmitVirtualGeometryGpuFeedback(h,version.Residency,version.Upload,geometry->Buffers.Readback.State,version.Feedback,error)) return false;
            version.Geometry=std::move(geometry); return true;
        }
    }

    struct VirtualGeometryStream::State
    {
        std::shared_ptr<VirtualGeometryStreamVersion> Active;
        std::optional<PendingTarget> Target;
        std::uint64_t LastFrame=0;
    };

    VirtualGeometryStream::VirtualGeometryStream(IRHI& rhi):m_RHI(rhi) {}
    VirtualGeometryStream::~VirtualGeometryStream()=default;
    bool VirtualGeometryStream::Ready() const noexcept { return bool(m_State); }
    bool VirtualGeometryStream::Pending() const noexcept { return m_State && bool(m_State->Target); }
    const VirtualGeometryResidency& VirtualGeometryStream::Residency() const { return m_State->Active->Residency; }
    const VirtualGeometryBuffers& VirtualGeometryStream::Geometry() const { return m_State->Active->Geometry->Buffers; }
    const VirtualGeometryGpuUpload& VirtualGeometryStream::Upload() const { return m_State->Active->Upload; }
    const VirtualGeometryCut& VirtualGeometryStream::Feedback() const { return m_State->Active->Feedback; }
    VirtualGeometryStreamSnapshot VirtualGeometryStream::Snapshot() const noexcept
    {
        VirtualGeometryStreamSnapshot snapshot;
        if (m_State) snapshot.m_Version=m_State->Active;
        return snapshot;
    }
    const VirtualGeometryResidency& VirtualGeometryStreamSnapshot::Residency() const { return m_Version->Residency; }
    const VirtualGeometryBuffers& VirtualGeometryStreamSnapshot::Geometry() const { return m_Version->Geometry->Buffers; }
    const VirtualGeometryGpuUpload& VirtualGeometryStreamSnapshot::Upload() const { return m_Version->Upload; }
    const VirtualGeometryCut& VirtualGeometryStreamSnapshot::Feedback() const { return m_Version->Feedback; }
    void VirtualGeometryStream::CancelPending() { if (m_State) m_State->Target.reset(); }

    bool VirtualGeometryStream::Initialize(const VirtualGeometryHierarchy& h,std::uint32_t physicalSlots,
        const std::vector<LodView>& views,const VirtualGeometryCutSettings& settings,const GeometryPreparationLimits& limits,
        VirtualGeometryStreamingStatistics& statistics,std::string& error)
    {
        if (Ready()) { error="Virtual stream already has an active source"; return false; }
        if (!ValidateVirtualGeometryHierarchy(h,error)) return false;
        auto candidate=std::make_unique<State>(); candidate->Active=std::make_shared<VirtualGeometryStreamVersion>(); auto& active=*candidate->Active;
        if (!CreateVirtualGeometryResidency(h.Signature,static_cast<unsigned>(h.Data.Pages.size()),physicalSlots,
                VirtualGeometryPinnedPages(h),active.Residency,error) ||
            !PrepareVirtualGeometryGpu(h,active.Residency,views,settings,{},limits,active.Upload,error)) return false;
        VirtualGeometryStreamingStatistics result;
        active.Cache=std::make_shared<CacheLease>(m_RHI);
        if (!NativeResult(m_RHI,m_RHI.PrepareVirtualGeometryCache(active.Upload,{},active.Cache->Handle,result.Native),error) ||
            !Decode(m_RHI,h,active,error)) return false;
        result.PoolPublished=result.CutChanged=true;
        m_State=std::move(candidate); statistics=result; error.clear(); return true;
    }

    bool VirtualGeometryStream::Update(const VirtualGeometryHierarchy& h,const std::vector<LodView>& views,
        const VirtualGeometryCutSettings& settings,const GeometryPreparationLimits& limits,std::uint64_t frame,
        std::uint32_t maximumUploads,VirtualGeometryStreamingStatistics& statistics,std::string& error)
    {
        if (!Ready() || h.Signature!=m_State->Active->Residency.Source())
        { error="Virtual stream has no active source or hierarchy identity changed"; return false; }
        auto& state=*m_State;
        if (frame<state.LastFrame || (state.Target && frame<state.Target->Residency.Frame()))
        { error="Virtual stream frame regressed"; return false; }

        // Refresh selection from the published cache for current views, even
        // while a different pool is being filled. No CPU selector is involved.
        auto candidate=std::make_shared<VirtualGeometryStreamVersion>(); candidate->Residency=state.Active->Residency; candidate->Cache=state.Active->Cache;
        const auto& previous=state.Active->Geometry->Buffers.Selection.Nodes;
        if (!PrepareVirtualGeometryGpu(h,candidate->Residency,views,settings,previous,limits,candidate->Upload,error) ||
            !Decode(m_RHI,h,*candidate,error)) return false;
        if (!state.Target)
        {
            // Validate complete target coverage/capacity once against the active
            // GPU feedback. Subsequent steps protect this whole admitted bundle.
            VirtualGeometryResidencyPlan full;
            if (!PlanVirtualGeometryCutResidency(h,candidate->Residency,candidate->Feedback,frame,full,error)) return false;
            PendingTarget target{candidate->Residency,candidate->Cache,candidate->Feedback.TargetPages};
            if (Missing(target)) state.Target=std::move(target);
        }

        VirtualGeometryStreamingStatistics result;
        if (state.Target)
        {
            auto& target=*state.Target;
            if (maximumUploads && Missing(target))
            {
                std::vector<VirtualGeometryPageRequest> requests;
                for (auto page:target.Pages) requests.push_back({page,StreamDemandPriority});
                VirtualGeometryResidencyPlan step;
                if (!PlanVirtualGeometryResidency(target.Residency,requests,frame,maximumUploads,step,error)) return false;
                auto staged=target.Residency;
                if (!CommitVirtualGeometryResidency(staged,std::move(step),error)) return false;
                VirtualGeometryGpuUpload upload;
                if (!PrepareVirtualGeometryGpu(h,staged,views,settings,previous,limits,upload,error)) return false;
                auto cache=std::make_shared<CacheLease>(m_RHI);
                if (!NativeResult(m_RHI,m_RHI.PrepareVirtualGeometryCache(upload,target.Cache->Handle,cache->Handle,result.Native),error)) return false;
                target.Residency=std::move(staged); target.Cache=std::move(cache);
            }
            result.PendingPages=Missing(target);
            if (!result.PendingPages)
            {
                auto complete=std::make_shared<VirtualGeometryStreamVersion>(); complete->Residency=target.Residency; complete->Cache=target.Cache;
                if (!PrepareVirtualGeometryGpu(h,complete->Residency,views,settings,previous,limits,complete->Upload,error) ||
                    !Decode(m_RHI,h,*complete,error)) return false;
                candidate=std::move(complete); state.Target.reset(); result.PoolPublished=true;
            }
        }
        result.CutChanged=previous!=candidate->Geometry->Buffers.Selection.Nodes;
        state.Active=std::move(candidate); state.LastFrame=frame;
        statistics=result; error.clear(); return true;
    }
}
