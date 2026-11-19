#pragma once

#include "Renderer/Pipelines/VisibilityBuffer.h"

namespace Evans::Renderer
{
    inline constexpr unsigned NaniteMaterialTileSize=8, NaniteMaterialThreads=64;
    inline constexpr unsigned NaniteMaterialBinWords=8, NaniteMaterialQuadVertices=6;
    inline constexpr std::size_t NaniteMaterialMaximumBytes=128u*1024u*1024u;
    struct NaniteMaterialConstants
    {
        ShaderUInt4 Counts; // Image width/height, triangle count, source-material table length.
        ShaderUInt4 Layout; // Bin count, tile columns/rows, temporary-list offset (words).
        ShaderUInt4 Storage; // Padded pixel capacity, compact-list offset, logical output words, optional native tile-image width.
    };
    static_assert(sizeof(NaniteMaterialConstants)==48);
    struct NaniteMaterialWork
    {
        NaniteMaterialConstants Constants;
        std::vector<unsigned> MaterialBins, DrawBins, Representatives;
        unsigned OutputWords=0;
        bool Active() const { return !Representatives.empty(); }
        ShaderUInt4 Resolve(unsigned draw) const
        { return Active()?ShaderUInt4{DrawBins[draw],Constants.Layout.Y,NaniteMaterialTileSize,1}:ShaderUInt4{}; }
    };
    // CPU establishes material identity only. Four native kernels clear headers,
    // classify visible tiles, allocate compact ranges and scatter tile IDs.
    // Each bin begins with the four non-indexed indirect-draw words, followed by
    // its absolute tile-list offset, scatter cursor and two reserved words.
    // Wihlidal, "Nanite GPU Driven Materials", GDC 2024:
    // https://media.gdcvault.com/gdc2024/Slides/GDC%2Bslide%2Bpresentations/Nanite%2BGPU%2BDriven%2BMaterials.pdf
    [[nodiscard]] bool BuildNaniteMaterialWork(const VisibilityBufferDesc& frame,int width,int height,
        NaniteMaterialWork& output,std::string& error);
}
