#pragma once

#include "Renderer/Nanite/NaniteGpu.h"

namespace Evans::Renderer
{
    struct NaniteCacheHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
        bool operator==(NaniteCacheHandle other) const noexcept { return Value==other.Value; }
        bool operator!=(NaniteCacheHandle other) const noexcept { return !(*this==other); }
    };
    struct NaniteCacheTask
    {
        std::uint64_t Value=0;
        explicit operator bool() const noexcept { return Value!=0; }
    };
    struct NaniteCacheSlot
    {
        std::uint32_t Page=NaniteInvalid,Generation=0;
        std::uint64_t Content=0; // Word identity, not an authentication hash.
    };
    struct NaniteCacheInfo
    {
        std::uint64_t Source=0,Epoch=0,Revision=0;
        std::vector<ShaderUInt4> PageTable;
        std::vector<NaniteCacheSlot> Slots;
    };
    struct NaniteCacheCopy { std::uint32_t FirstSlot=0,SlotCount=0; };
    struct NaniteCacheStatistics
    {
        std::uint32_t UploadedPages=0,CopiedPages=0;
        std::uint64_t UploadedBytes=0,CopiedBytes=0,AllocatedBytes=0;
    };
    struct NaniteCachePlan
    {
        NaniteCacheInfo Next;
        std::vector<std::uint32_t> UploadSlots;
        std::vector<NaniteCacheCopy> Copies;
        NaniteCacheStatistics Statistics;
        bool ReusePool=false;
        std::size_t ReservedBytes=0; // Conservative lifetime charge for async pool copies.
    };
    // Immutable pool versions retain old cuts while candidate bundles arrive.
    // Unchanged slots use GPU copies; only changed compressed pages cross the
    // host boundary. Native identity prefixes are part of the copied stride.
    // An unchanged mapping shares its pool, with independent handle lifetimes.
    [[nodiscard]] bool BuildNaniteCachePlan(const NaniteCacheInfo* previous,const NaniteGpuUpload&,
        NaniteCachePlan&,std::string& error);
    [[nodiscard]] bool MatchNaniteCache(const NaniteCacheInfo&,const NaniteGpuUpload&,std::string& error);
    [[nodiscard]] NaniteCacheHandle NewNaniteCacheHandle() noexcept;
}
