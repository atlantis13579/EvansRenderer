#include "Renderer/VirtualGeometry/VirtualGeometryCache.h"

#include <atomic>

namespace Evans::Renderer
{
    namespace
    {
        std::atomic<std::uint64_t> NextCacheHandle{1};
        std::uint64_t Pair(std::uint32_t low,std::uint32_t high) { return low|(std::uint64_t(high)<<32); }
        std::uint64_t Content(const VirtualGeometryPage& page)
        {
            constexpr std::uint64_t Seed=14695981039346656037ull,Prime=1099511628211ull;
            auto hash=Seed; for (auto word:page.Words) { hash^=word; hash*=Prime; } return hash;
        }
        bool Same(const ShaderUInt4& a,const ShaderUInt4& b) { return a.X==b.X && a.Y==b.Y && a.Z==b.Z && a.W==b.W; }
        bool Snapshot(const VirtualGeometryGpuUpload& u,VirtualGeometryCacheInfo& info,std::string& error)
        {
            if (!ValidateVirtualGeometryGpuUpload(u,error)) return false;
            info.Source=u.Source; info.Epoch=Pair(u.Constants.Identity.X,u.Constants.Identity.Y); info.Revision=Pair(u.Constants.Identity.Z,u.Constants.Identity.W);
            info.PageTable=u.PageTable; info.Slots.resize(u.PhysicalPages.size());
            for (unsigned page=0;page<u.PageTable.size();++page)
            { const auto& entry=u.PageTable[page]; if (entry.X!=VirtualGeometryInvalid) info.Slots[entry.X]={page,entry.Y,Content(u.PhysicalPages[entry.X])}; }
            return true;
        }
    }
    VirtualGeometryCacheHandle NewVirtualGeometryCacheHandle() noexcept
    {
        auto value=NextCacheHandle.load(std::memory_order_relaxed);
        while (value!=UINT64_MAX) if (NextCacheHandle.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return {value};
        return {};
    }
    bool BuildVirtualGeometryCachePlan(const VirtualGeometryCacheInfo* previous,const VirtualGeometryGpuUpload& u,VirtualGeometryCachePlan& output,std::string& error)
    {
        VirtualGeometryCachePlan plan; if (!Snapshot(u,plan.Next,error)) return false; const auto& next=plan.Next;
        if (previous && (previous->Source!=next.Source || previous->Epoch!=next.Epoch || previous->Revision>next.Revision || previous->PageTable.size()!=next.PageTable.size() || previous->Slots.size()!=next.Slots.size()))
        { error="Native virtual cache update changed source/epoch/capacity or regressed revision"; return false; }
        for (unsigned slot=0;slot<next.Slots.size();++slot)
        {
            const auto& to=next.Slots[slot]; const auto* from=previous?&previous->Slots[slot]:nullptr;
            const bool unchanged=from && from->Page==to.Page && from->Generation==to.Generation;
            if (unchanged && from->Content!=to.Content) { error="Native virtual page contents changed without a slot generation"; return false; }
            if (to.Page==VirtualGeometryInvalid)
            { if (from && from->Page!=VirtualGeometryInvalid) { error="Native virtual cache unexpectedly discarded an occupied slot"; return false; } continue; }
            if (unchanged)
            {
                if (!plan.Copies.empty() && plan.Copies.back().FirstSlot+plan.Copies.back().SlotCount==slot) ++plan.Copies.back().SlotCount;
                else plan.Copies.push_back({slot,1});
                ++plan.Statistics.CopiedPages;
            }
            else
            {
                if (from && (from->Generation==UINT32_MAX || to.Generation!=from->Generation+1)) { error="Native virtual slot replacement has an invalid generation"; return false; }
                plan.UploadSlots.push_back(slot);
            }
        }
        if (previous && previous->Revision==next.Revision)
        {
            bool same=plan.UploadSlots.empty();
            for (unsigned page=0;page<next.PageTable.size();++page) same=same && Same(previous->PageTable[page],next.PageTable[page]);
            if (!same) { error="Native virtual mapping changed without a cache revision"; return false; }
        }
        plan.ReusePool=previous && plan.UploadSlots.empty();
        if (plan.ReusePool) { plan.Copies.clear(); plan.Statistics.CopiedPages=0; }
        plan.Statistics.UploadedPages=static_cast<unsigned>(plan.UploadSlots.size()); plan.Statistics.UploadedBytes=std::uint64_t(plan.Statistics.UploadedPages)*VirtualGeometryPageBytes;
        plan.Statistics.CopiedBytes=std::uint64_t(plan.Statistics.CopiedPages)*VirtualGeometryPageBytes;
        plan.Statistics.AllocatedBytes=plan.ReusePool?0:std::uint64_t(next.Slots.size())*VirtualGeometryPageBytes;
        output=std::move(plan); error.clear(); return true;
    }
    bool MatchVirtualGeometryCache(const VirtualGeometryCacheInfo& cache,const VirtualGeometryGpuUpload& u,std::string& error)
    {
        VirtualGeometryCachePlan plan;
        if (!BuildVirtualGeometryCachePlan(&cache,u,plan,error)) return false;
        if (plan.Next.Revision!=cache.Revision || !plan.ReusePool) { error="Native virtual cache handle names another mapping revision"; return false; }
        error.clear(); return true;
    }
}
