#include "Renderer/VirtualGeometry/VirtualGeometryResidency.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <tuple>

namespace Evans::Renderer
{
    namespace
    {
        constexpr auto MaximumPhysicalPages=std::numeric_limits<std::uint32_t>::max()/VirtualGeometryPageBytes;
        constexpr auto MaximumVirtualPages=std::numeric_limits<std::uint32_t>::max()/VirtualGeometryPageWords;
        std::atomic<std::uint64_t> NextEpoch{1};
        bool NewEpoch(std::uint64_t& epoch)
        {
            auto value=NextEpoch.load(std::memory_order_relaxed);
            while (value!=std::numeric_limits<std::uint64_t>::max())
                if (NextEpoch.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) { epoch=value; return true; }
            return false;
        }
        bool Add(std::uint64_t previous,std::uint32_t delta,std::uint64_t& result)
        {
            if (delta>std::numeric_limits<std::uint64_t>::max()-previous) return false;
            result=previous+delta; return true;
        }
    }
    bool CreateVirtualGeometryResidency(std::uint64_t source,std::uint32_t pageCount,std::uint32_t physicalSlots,
        const std::vector<std::uint32_t>& pinnedPages,VirtualGeometryResidency& output,std::string& error)
    {
        if (!pageCount || pageCount>MaximumVirtualPages || physicalSlots>pageCount || physicalSlots>MaximumPhysicalPages)
        { error="Virtual residency dimensions exceed page/address bounds"; return false; }
        auto pinned=pinnedPages; std::sort(pinned.begin(),pinned.end());
        if (pinned.size()>physicalSlots || (!pinned.empty() && pinned.back()>=pageCount) || std::adjacent_find(pinned.begin(),pinned.end())!=pinned.end())
        { error="Pinned virtual pages exceed capacity, repeat or reference missing pages"; return false; }
        VirtualGeometryResidency candidate; candidate.m_Source=source; candidate.m_Revision=1; candidate.m_Pinned=std::move(pinned);
        if (!NewEpoch(candidate.m_Epoch)) { error="Virtual residency epoch space is exhausted"; return false; }
        candidate.m_PageSlots.assign(pageCount,VirtualGeometryInvalid); candidate.m_Slots.resize(physicalSlots);
        for (std::uint32_t slot=0;slot<candidate.m_Pinned.size();++slot)
        {
            const auto page=candidate.m_Pinned[slot]; candidate.m_PageSlots[page]=slot;
            candidate.m_Slots[slot]={page,1,0};
        }
        if (!ValidateVirtualGeometryResidency(candidate,error)) return false;
        output=std::move(candidate); error.clear(); return true;
    }
    bool ValidateVirtualGeometryResidency(const VirtualGeometryResidency& state,std::string& error)
    {
        if (!state.m_Epoch || !state.m_Revision || state.m_PageSlots.empty() || state.m_PageSlots.size()>MaximumVirtualPages ||
            state.m_Slots.size()>MaximumPhysicalPages || state.m_Slots.size()>state.m_PageSlots.size() ||
            state.m_Pinned.size()>state.m_Slots.size() || !std::is_sorted(state.m_Pinned.begin(),state.m_Pinned.end()) ||
            std::adjacent_find(state.m_Pinned.begin(),state.m_Pinned.end())!=state.m_Pinned.end())
        { error="Virtual residency has invalid dimensions, epoch or pin ordering"; return false; }
        for (std::size_t page=0;page<state.m_PageSlots.size();++page)
        {
            const auto slot=state.m_PageSlots[page];
            if (slot!=VirtualGeometryInvalid && (slot>=state.m_Slots.size() || state.m_Slots[slot].Page!=page))
            { error="Virtual page table is not reciprocal with physical slots"; return false; }
        }
        for (std::size_t slot=0;slot<state.m_Slots.size();++slot)
        {
            const auto& physical=state.m_Slots[slot];
            if (physical.Page==VirtualGeometryInvalid)
            { if (physical.Generation || physical.LastUsedFrame) { error="An empty virtual slot retains generation/age"; return false; } }
            else if (physical.Page>=state.m_PageSlots.size() || state.m_PageSlots[physical.Page]!=slot || !physical.Generation || physical.LastUsedFrame>state.m_Frame)
            { error="Physical virtual-geometry slot identity or age is invalid"; return false; }
        }
        for (auto page:state.m_Pinned) if (page>=state.m_PageSlots.size() || state.m_PageSlots[page]==VirtualGeometryInvalid)
        { error="A pinned coarse page is not resident"; return false; }
        error.clear(); return true;
    }
    bool PlanVirtualGeometryResidency(const VirtualGeometryResidency& current,const std::vector<VirtualGeometryPageRequest>& requests,
        std::uint64_t frame,std::uint32_t maximumUploads,VirtualGeometryResidencyPlan& output,std::string& error)
    {
        if (!ValidateVirtualGeometryResidency(current,error)) return false;
        if (frame<current.m_Frame || current.m_Revision==std::numeric_limits<std::uint64_t>::max())
        { error="Virtual residency frame regressed or revision space is exhausted"; return false; }
        auto unique=requests;
        for (const auto& request:unique) if (request.Page>=current.m_PageSlots.size() || !std::isfinite(request.Priority) || request.Priority<0)
        { error="Virtual page request has an invalid ID or priority"; return false; }
        std::sort(unique.begin(),unique.end(),[](const auto& a,const auto& b){return a.Page!=b.Page?a.Page<b.Page:a.Priority>b.Priority;});
        unique.erase(std::unique(unique.begin(),unique.end(),[](const auto& a,const auto& b){return a.Page==b.Page;}),unique.end());
        std::sort(unique.begin(),unique.end(),[](const auto& a,const auto& b){return a.Priority!=b.Priority?a.Priority>b.Priority:a.Page<b.Page;});
        VirtualGeometryResidencyPlan candidate; candidate.m_BaseEpoch=current.m_Epoch; candidate.m_BaseRevision=current.m_Revision;
        candidate.m_Next=current; auto& next=candidate.m_Next; next.m_Frame=frame; ++next.m_Revision;
        std::vector<bool> keep(current.m_PageSlots.size()); for (auto page:current.m_Pinned) keep[page]=true;
        std::size_t selected=0; const std::size_t capacity=current.m_Slots.size()-current.m_Pinned.size();
        for (const auto& request:unique)
        {
            const auto slot=current.m_PageSlots[request.Page];
            if (slot==VirtualGeometryInvalid) ++candidate.m_Statistics.Misses;
            else { ++candidate.m_Statistics.Hits; next.m_Slots[slot].LastUsedFrame=frame; }
            if (!keep[request.Page] && selected<capacity) { keep[request.Page]=true; ++selected; }
        }
        // Sort victims once, including free slots first. This avoids a full
        // capacity scan for every upload in a large streaming burst.
        std::vector<std::uint32_t> victims;
        for (std::uint32_t slot=0;slot<next.m_Slots.size();++slot)
        { const auto page=next.m_Slots[slot].Page; if (page==VirtualGeometryInvalid || !keep[page]) victims.push_back(slot); }
        std::sort(victims.begin(),victims.end(),[&](auto a,auto b)
        {
            const auto& x=next.m_Slots[a]; const auto& y=next.m_Slots[b];
            return std::make_tuple(x.Page!=VirtualGeometryInvalid,x.LastUsedFrame,x.Page,a)<std::make_tuple(y.Page!=VirtualGeometryInvalid,y.LastUsedFrame,y.Page,b);
        });
        std::size_t victim=0;
        for (const auto& request:unique)
        {
            if (!keep[request.Page] || next.m_PageSlots[request.Page]!=VirtualGeometryInvalid || candidate.m_Uploads.size()>=maximumUploads) continue;
            if (victim>=victims.size()) { error="Virtual working-set allocation lost a replacement slot"; return false; }
            const auto slot=victims[victim++]; auto& physical=next.m_Slots[slot];
            if (physical.Generation==std::numeric_limits<std::uint32_t>::max()) { error="Virtual physical-slot generation space is exhausted"; return false; }
            candidate.m_Uploads.push_back({request.Page,slot,physical.Page,physical.Generation+1});
            if (physical.Page!=VirtualGeometryInvalid) { next.m_PageSlots[physical.Page]=VirtualGeometryInvalid; ++candidate.m_Statistics.Evictions; }
            physical={request.Page,physical.Generation+1,frame}; next.m_PageSlots[request.Page]=slot;
        }
        for (const auto& request:unique) if (next.m_PageSlots[request.Page]==VirtualGeometryInvalid) candidate.m_Deferred.push_back(request.Page);
        auto& statistics=candidate.m_Statistics; statistics.Uploads=static_cast<std::uint32_t>(candidate.m_Uploads.size()); statistics.Deferred=static_cast<std::uint32_t>(candidate.m_Deferred.size());
        for (const auto& slot:next.m_Slots) statistics.Resident+=slot.Page!=VirtualGeometryInvalid;
        if (!Add(current.m_Counters.Hits,statistics.Hits,next.m_Counters.Hits) || !Add(current.m_Counters.Misses,statistics.Misses,next.m_Counters.Misses) ||
            !Add(current.m_Counters.Uploads,statistics.Uploads,next.m_Counters.Uploads) || !Add(current.m_Counters.Evictions,statistics.Evictions,next.m_Counters.Evictions))
        { error="Virtual residency counters overflowed"; return false; }
        if (!ValidateVirtualGeometryResidency(next,error)) return false;
        candidate.m_Prepared=true; output=std::move(candidate); error.clear(); return true;
    }
    bool CommitVirtualGeometryResidency(VirtualGeometryResidency& current,VirtualGeometryResidencyPlan&& plan,std::string& error)
    {
        if (!plan.m_Prepared || plan.m_BaseEpoch!=current.m_Epoch || plan.m_BaseRevision!=current.m_Revision ||
            plan.m_Next.m_Source!=current.m_Source || plan.m_Next.m_Epoch!=current.m_Epoch || plan.m_Next.m_Revision!=current.m_Revision+1 ||
            plan.m_Next.m_Frame<current.m_Frame || plan.m_Next.m_Pinned!=current.m_Pinned || plan.m_Next.m_PageSlots.size()!=current.m_PageSlots.size() ||
            plan.m_Next.m_Slots.size()!=current.m_Slots.size())
        { error="Virtual residency plan is unprepared, stale or belongs to another source/epoch"; return false; }
        if (!ValidateVirtualGeometryResidency(plan.m_Next,error)) return false;
        current=std::move(plan.m_Next); plan.m_Prepared=false; error.clear(); return true;
    }
}
