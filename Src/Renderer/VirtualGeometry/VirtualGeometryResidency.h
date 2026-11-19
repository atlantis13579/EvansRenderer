#pragma once

#include "Renderer/VirtualGeometry/VirtualGeometryPages.h"

namespace Evans::Renderer
{
    struct VirtualGeometryPageRequest { std::uint32_t Page=0; float Priority=0; };
    struct VirtualGeometryPhysicalPage
    {
        std::uint32_t Page=VirtualGeometryInvalid,Generation=0;
        std::uint64_t LastUsedFrame=0;
    };
    struct VirtualGeometryPageUpload
    {
        std::uint32_t Page=0,Slot=0,PreviousPage=VirtualGeometryInvalid,Generation=0;
    };
    struct VirtualGeometryResidencyStatistics
    {
        std::uint32_t Hits=0,Misses=0,Uploads=0,Evictions=0,Deferred=0,Resident=0;
    };
    struct VirtualGeometryResidencyCounters { std::uint64_t Hits=0,Misses=0,Uploads=0,Evictions=0; };
    class VirtualGeometryResidencyPlan;
    class VirtualGeometryResidency
    {
    public:
        [[nodiscard]] std::uint64_t Source() const noexcept { return m_Source; }
        [[nodiscard]] std::uint64_t Epoch() const noexcept { return m_Epoch; }
        [[nodiscard]] std::uint64_t Revision() const noexcept { return m_Revision; }
        [[nodiscard]] std::uint64_t Frame() const noexcept { return m_Frame; }
        [[nodiscard]] const std::vector<std::uint32_t>& PageSlots() const noexcept { return m_PageSlots; }
        [[nodiscard]] const std::vector<std::uint32_t>& PinnedPages() const noexcept { return m_Pinned; }
        [[nodiscard]] const std::vector<VirtualGeometryPhysicalPage>& Slots() const noexcept { return m_Slots; }
        [[nodiscard]] const VirtualGeometryResidencyCounters& Counters() const noexcept { return m_Counters; }
    private:
        std::uint64_t m_Source=0,m_Epoch=0,m_Revision=0,m_Frame=0;
        std::vector<std::uint32_t> m_PageSlots,m_Pinned;
        std::vector<VirtualGeometryPhysicalPage> m_Slots;
        VirtualGeometryResidencyCounters m_Counters;
        friend bool CreateVirtualGeometryResidency(std::uint64_t,std::uint32_t,std::uint32_t,const std::vector<std::uint32_t>&,VirtualGeometryResidency&,std::string&);
        friend bool ValidateVirtualGeometryResidency(const VirtualGeometryResidency&,std::string&);
        friend bool PlanVirtualGeometryResidency(const VirtualGeometryResidency&,const std::vector<VirtualGeometryPageRequest>&,std::uint64_t,std::uint32_t,VirtualGeometryResidencyPlan&,std::string&);
        friend bool CommitVirtualGeometryResidency(VirtualGeometryResidency&,VirtualGeometryResidencyPlan&&,std::string&);
    };
    class VirtualGeometryResidencyPlan
    {
    public:
        [[nodiscard]] bool Prepared() const noexcept { return m_Prepared; }
        [[nodiscard]] const VirtualGeometryResidency& Next() const noexcept { return m_Next; }
        [[nodiscard]] const std::vector<VirtualGeometryPageUpload>& Uploads() const noexcept { return m_Uploads; }
        [[nodiscard]] const std::vector<std::uint32_t>& DeferredPages() const noexcept { return m_Deferred; }
        [[nodiscard]] const VirtualGeometryResidencyStatistics& Statistics() const noexcept { return m_Statistics; }
    private:
        bool m_Prepared=false;
        std::uint64_t m_BaseEpoch=0,m_BaseRevision=0;
        VirtualGeometryResidency m_Next;
        std::vector<VirtualGeometryPageUpload> m_Uploads;
        std::vector<std::uint32_t> m_Deferred;
        VirtualGeometryResidencyStatistics m_Statistics;
        friend bool PlanVirtualGeometryResidency(const VirtualGeometryResidency&,const std::vector<VirtualGeometryPageRequest>&,std::uint64_t,std::uint32_t,VirtualGeometryResidencyPlan&,std::string&);
        friend bool CommitVirtualGeometryResidency(VirtualGeometryResidency&,VirtualGeometryResidencyPlan&&,std::string&);
    };
    // The CPU planner owns no GPU storage. Prepare an immutable replacement
    // pool/page table and wait for native upload/selection completion before
    // Commit. Abandoned or stale plans never mutate the published CPU mapping.
    // Root/coarse pages are pinned; priority chooses the working set, then LRU
    // chooses reusable slots outside that set. Generation tags reject stale
    // physical contents. The epoch also rejects plans across same-source resets.
    // Create reserves logical pinned slots; a native executor must upload those
    // initial pages before publishing the cache. Demand counters exclude them.
    [[nodiscard]] bool CreateVirtualGeometryResidency(std::uint64_t source,std::uint32_t pageCount,std::uint32_t physicalSlots,
        const std::vector<std::uint32_t>& pinnedPages,VirtualGeometryResidency& output,std::string& error);
    [[nodiscard]] bool ValidateVirtualGeometryResidency(const VirtualGeometryResidency& state,std::string& error);
    [[nodiscard]] bool PlanVirtualGeometryResidency(const VirtualGeometryResidency& current,
        const std::vector<VirtualGeometryPageRequest>& requests,std::uint64_t frame,std::uint32_t maximumUploads,
        VirtualGeometryResidencyPlan& output,std::string& error);
    [[nodiscard]] bool CommitVirtualGeometryResidency(VirtualGeometryResidency& current,VirtualGeometryResidencyPlan&& plan,std::string& error);
}
