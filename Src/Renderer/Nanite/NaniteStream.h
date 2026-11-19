#pragma once

#include "Renderer/Nanite/NaniteCache.h"

#include <memory>

namespace Evans::Renderer
{
    class IRHI;
    struct NaniteStreamVersion;
    struct NaniteStreamingStatistics
    {
        NaniteCacheStatistics Native;
        std::uint32_t PendingPages=0;
        std::uint64_t ReadbackBytes=0;
        bool PoolPublished=false,CutChanged=false,ProducerPending=false,PageCopyPending=false,PageReadPending=false;
    };
    // A snapshot leases immutable native geometry/cache versions. Keep every
    // family's old snapshot until a scene transaction succeeds. All leases must
    // be released on the owning RHI thread before the RHI is destroyed.
    class NaniteStreamSnapshot
    {
    public:
        NaniteStreamSnapshot()=default;
        [[nodiscard]] explicit operator bool() const noexcept { return bool(m_Version); }
        [[nodiscard]] const VirtualGeometryResidency& Residency() const;
        [[nodiscard]] const NaniteGeometryBuffers& Geometry() const;
        [[nodiscard]] const NaniteGpuUpload& Upload() const;
        [[nodiscard]] const NaniteResidentCut& Feedback() const;
    private:
        friend class NaniteStream;
        std::shared_ptr<const NaniteStreamVersion> m_Version;
    };
    // Root bootstrap publishes complete coverage. Synchronous updates refresh the
    // active GPU cut, then uploads a bounded number of pages into a private pool.
    // Only a complete dependency-closed target can replace that active pool;
    // partial child bundles and failed decodes retain the old drawable version.
    // Finish a pending target before accepting another to avoid camera-motion
    // starvation. CancelPending explicitly abandons the private candidate.
    // Hierarchy/pages are immutable; source changes require a fresh stream.
    // Call on the owning RHI thread between frames; release streams/snapshots
    // before RHI shutdown, including when cancelling a pending target.
    class NaniteStream
    {
    public:
        explicit NaniteStream(IRHI&);
        ~NaniteStream();
        NaniteStream(const NaniteStream&)=delete;
        NaniteStream& operator=(const NaniteStream&)=delete;
        [[nodiscard]] bool Initialize(const NaniteHierarchy&,const NanitePages&,std::uint32_t physicalSlots,
            const std::vector<LodView>&,double maximumErrorPixels,const GeometryPreparationLimits&,
            NaniteStreamingStatistics&,std::string& error);
        // Repeated calls advance root upload, decode and atomic publication.
        // Ready/Snapshot stay empty until complete. Changing source/capacity
        // requires CancelPending; current views/limits are validated on every call.
        [[nodiscard]] bool InitializeAsync(const NaniteHierarchy&,const NanitePages&,std::uint32_t physicalSlots,
            const std::vector<LodView>&,double maximumErrorPixels,const GeometryPreparationLimits&,
            NaniteStreamingStatistics&,std::string& error);
        [[nodiscard]] bool Update(const NaniteHierarchy&,const NanitePages&,const std::vector<LodView>&,
            double maximumErrorPixels,const GeometryPreparationLimits&,std::uint64_t frame,std::uint32_t maximumUploads,
            NaniteStreamingStatistics&,std::string& error);
        // Nonblocking page copies and selection/decode retain the published cut.
        // A page step is admitted only after completion; complete dependency-closed
        // targets take priority over camera cuts. Initialize is the synchronous oracle.
        [[nodiscard]] bool UpdateAsync(const NaniteHierarchy&,const NanitePages&,const std::vector<LodView>&,
            double maximumErrorPixels,const GeometryPreparationLimits&,std::uint64_t frame,std::uint32_t maximumUploads,
            NaniteStreamingStatistics&,std::string& error);
        void CancelPending();
        [[nodiscard]] bool Ready() const noexcept;
        [[nodiscard]] bool Pending() const noexcept;
        [[nodiscard]] bool Producing() const noexcept; // A native task awaits consumption.
        [[nodiscard]] NaniteStreamSnapshot Snapshot() const noexcept;
        // Accessors require Ready(); snapshots require a nonempty lease.
        [[nodiscard]] const VirtualGeometryResidency& Residency() const;
        [[nodiscard]] const NaniteGeometryBuffers& Geometry() const;
        [[nodiscard]] const NaniteGpuUpload& Upload() const;
        [[nodiscard]] const NaniteResidentCut& Feedback() const;
    private:
        struct State;
        struct Bootstrap;
        IRHI& m_RHI;
        std::unique_ptr<State> m_State;
        std::unique_ptr<Bootstrap> m_Bootstrap;
    };
}
