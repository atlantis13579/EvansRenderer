#pragma once

#include "Renderer/Nanite/NaniteRaster.h"

namespace Evans::Renderer
{
    inline constexpr unsigned NaniteHardwareThreads=64,NaniteHardwareColumns=64,NaniteHardwareBinWords=8;
    inline constexpr std::size_t NaniteHardwareMaximumBytes=128ull*1024*1024;
    struct NaniteHardwareConstants
    {
        ShaderUInt4 Counts; // Candidate triangles, source ranges, bins, pixels.
        ShaderUInt4 Layout; // Raster route offset, priority map offset, compact list offset, dispatch columns.
        ShaderUInt4 Images; // Width, height, optional ES integer queue image width, reserved.
    };
    struct NaniteHardwareDrawConstants
    {
        NaniteRasterDrawConstants Raster;
        ShaderUInt4 Schedule; // Bin, priority map offset, candidate count, optional ES queue image width.
    };
    static_assert(sizeof(NaniteHardwareConstants)==48 && sizeof(NaniteHardwareDrawConstants)==128,"Nanite hardware schedule ABI changed");
    struct NaniteHardwareBin
    {
        NaniteHardwareDrawConstants Constants;
        TextureHandle Diffuse,Opacity;
        bool Cull=false,Mirrored=false;
    };
    struct NaniteHardwareWork
    {
        NaniteHardwareConstants Constants;
        std::vector<ShaderUInt4> Ranges; // First triangle, count, bin, original submission priority begin.
        std::vector<NaniteHardwareBin> Bins;
        unsigned ArgumentWords=0,QueueWords=0;
        bool Active() const { return !Bins.empty(); }
    };
    // GPU compaction can reorder triangles and nonadjacent material bins. A
    // depth epoch followed by equal-depth atomic-min ownership preserves the
    // original submission priority independently of that ordering. The resolve
    // maps the priority back to a stable triangle ID before compute coverage.
    // This is a portable two-epoch substitute for a packed 64-bit depth/ID atomic.
    [[nodiscard]] bool BuildNaniteHardwareWork(const VisibilityBufferDesc&,unsigned width,unsigned height,
        NaniteHardwareWork&,std::string&);
}
