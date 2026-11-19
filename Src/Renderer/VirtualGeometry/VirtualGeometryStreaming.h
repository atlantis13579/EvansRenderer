#pragma once

#include "Renderer/VirtualGeometry/VirtualGeometryCache.h"

#include <memory>

namespace Evans::Renderer
{
    class IRHI;
    struct VirtualGeometryStreamVersion;

    struct VirtualGeometryStreamingStatistics
    {
        VirtualGeometryCacheStatistics Native;
        std::uint32_t PendingPages=0;
        bool PoolPublished=false,CutChanged=false;
    };

    // A renderer can hold every family's previous version until a whole scene
    // update succeeds. Copying this lease retains native pools and drawable
    // buffers, without copying page payloads or GPU readback. Release all leases
    // on the owning RHI thread, between frames and before RHI shutdown.
    class VirtualGeometryStreamSnapshot
    {
    public:
        VirtualGeometryStreamSnapshot()=default;
        [[nodiscard]] explicit operator bool() const noexcept { return bool(m_Version); }
        // Accessors require a nonempty snapshot.
        [[nodiscard]] const VirtualGeometryResidency& Residency() const;
        [[nodiscard]] const VirtualGeometryBuffers& Geometry() const;
        [[nodiscard]] const VirtualGeometryGpuUpload& Upload() const;
        [[nodiscard]] const VirtualGeometryCut& Feedback() const;
    private:
        friend class VirtualGeometryStream;
        std::shared_ptr<const VirtualGeometryStreamVersion> m_Version;
    };

    // Synchronous native submissions, bounded uploads per Update. A private
    // candidate may replace pages needed by the current cut: the active pool and
    // drawable allocations stay alive until the entire admitted target decodes.
    // Finishing a target before accepting another prevents camera motion from
    // continually restarting uploads. CancelPending explicitly abandons it.
    //
    // Initialize uploads all pinned roots before publishing the first drawable;
    // this bootstrap is outside the per-update detail budget. The immutable
    // hierarchy and RHI must outlive use of the stream. Call between frames,
    // destroy before RHI shutdown, and replace the stream for a new source/epoch.
    class VirtualGeometryStream
    {
    public:
        explicit VirtualGeometryStream(IRHI& rhi);
        ~VirtualGeometryStream();
        VirtualGeometryStream(const VirtualGeometryStream&)=delete;
        VirtualGeometryStream& operator=(const VirtualGeometryStream&)=delete;

        [[nodiscard]] bool Initialize(const VirtualGeometryHierarchy& hierarchy,std::uint32_t physicalSlots,
            const std::vector<LodView>& views,const VirtualGeometryCutSettings& settings,
            const GeometryPreparationLimits& limits,VirtualGeometryStreamingStatistics& statistics,std::string& error);
        [[nodiscard]] bool Update(const VirtualGeometryHierarchy& hierarchy,const std::vector<LodView>& views,
            const VirtualGeometryCutSettings& settings,const GeometryPreparationLimits& limits,
            std::uint64_t frame,std::uint32_t maximumUploads,VirtualGeometryStreamingStatistics& statistics,std::string& error);
        void CancelPending();

        [[nodiscard]] bool Ready() const noexcept;
        [[nodiscard]] bool Pending() const noexcept;
        [[nodiscard]] VirtualGeometryStreamSnapshot Snapshot() const noexcept;
        // Accessors require Ready(). Buffers/readback remain owned by the stream.
        [[nodiscard]] const VirtualGeometryResidency& Residency() const;
        [[nodiscard]] const VirtualGeometryBuffers& Geometry() const;
        [[nodiscard]] const VirtualGeometryGpuUpload& Upload() const;
        [[nodiscard]] const VirtualGeometryCut& Feedback() const;

    private:
        struct State;
        IRHI& m_RHI;
        std::unique_ptr<State> m_State;
    };
}
