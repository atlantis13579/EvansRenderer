#pragma once

#include "Renderer/VirtualGeometry/VirtualGeometryGpu.h"

namespace Evans::Renderer
{
    struct VirtualGeometryCacheHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(VirtualGeometryCacheHandle other) const noexcept { return Value==other.Value; }
        bool operator!=(VirtualGeometryCacheHandle other) const noexcept { return !(*this==other); }
    };
    struct VirtualGeometryCacheSlot
    {
        std::uint32_t Page=VirtualGeometryInvalid,Generation=0;
        std::uint64_t Content=0; // deterministic word identity, not a security hash
    };
    struct VirtualGeometryCacheInfo
    {
        std::uint64_t Source=0,Epoch=0,Revision=0;
        std::vector<ShaderUInt4> PageTable;
        std::vector<VirtualGeometryCacheSlot> Slots;
    };
    struct VirtualGeometryCacheCopy { std::uint32_t FirstSlot=0,SlotCount=0; };
    struct VirtualGeometryCacheStatistics
    {
        std::uint32_t UploadedPages=0,CopiedPages=0;
        std::uint64_t UploadedBytes=0,CopiedBytes=0,AllocatedBytes=0;
    };
    struct VirtualGeometryCachePlan
    {
        VirtualGeometryCacheInfo Next;
        std::vector<std::uint32_t> UploadSlots;
        std::vector<VirtualGeometryCacheCopy> Copies;
        VirtualGeometryCacheStatistics Statistics;
        bool ReusePool=false;
    };
    // Native cache handles name immutable pool versions. Build a candidate by
    // GPU-copying unchanged resident runs and uploading changed pages only. An
    // unchanged mapping may share its native pool across independent handles.
    // Publish CPU residency and the chosen native handle together after decoding
    // succeeds. Releasing an abandoned candidate never changes the previous pool.
    [[nodiscard]] bool BuildVirtualGeometryCachePlan(const VirtualGeometryCacheInfo* previous,const VirtualGeometryGpuUpload& upload,
        VirtualGeometryCachePlan& output,std::string& error);
    [[nodiscard]] bool MatchVirtualGeometryCache(const VirtualGeometryCacheInfo& cache,const VirtualGeometryGpuUpload& upload,std::string& error);
    [[nodiscard]] VirtualGeometryCacheHandle NewVirtualGeometryCacheHandle() noexcept;
}
