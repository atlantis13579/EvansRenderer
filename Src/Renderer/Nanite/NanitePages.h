#pragma once

#include "Renderer/Nanite/NaniteHierarchy.h"

#include <array>
#include <memory>

namespace Evans::Renderer
{
    class NanitePageSource;
    inline constexpr std::uint32_t NanitePageBytes=64*1024,NanitePageWords=NanitePageBytes/sizeof(std::uint32_t);
    inline constexpr std::uint32_t NanitePageHeaderWords=8,NaniteVertexChannels=12;
    inline constexpr std::uint32_t NaniteClusterHeaderWords=8+3*NaniteVertexChannels;
    inline constexpr std::uint32_t NanitePageMagic=std::uint32_t('E')|(std::uint32_t('N')<<8)|(std::uint32_t('P')<<16)|(std::uint32_t('G')<<24);
    inline constexpr std::uint32_t NanitePageVersion=1,NanitePagePinned=1;
    enum class NanitePageWord : unsigned { Magic,Version,UsedWords,ClusterCount,Flags,PrecisionExponent,Page,Reserved };
    enum class NaniteClusterWord : unsigned
    {
        Cluster,VertexCount,IndexCount,Material,GeneratingGroup,ParentGroup,WordCount,IndexBitOffset,
        Bases,Widths=Bases+NaniteVertexChannels,Offsets=Widths+NaniteVertexChannels
    };
    struct NanitePage { std::array<std::uint32_t,NanitePageWords> Words{}; };
    struct NaniteClusterLocation
    {
        std::uint32_t Page=0,Word=0,FirstSourceVertex=0,VertexCount=0;
    };
    struct NaniteGroupPages { std::uint32_t First=0,Count=0; };
    struct NanitePageSettings
    {
        // Signed coordinates share a power-of-two grid centered at the object
        // origin. More magnitude bits reduce the position quantization error.
        unsigned PositionPrecisionBits=24;
        std::size_t MaximumPages=16*1024;
    };
    struct NanitePages
    {
        std::vector<NanitePage> Pages;
        std::shared_ptr<NanitePageSource> Source; // Cooked payloads need not stay in RAM.
        std::vector<NaniteClusterLocation> Clusters; // Indexed by hierarchy cluster ID.
        std::vector<NaniteGroupPages> Groups; // Complete child bundles, consecutive pages.
        std::vector<std::uint32_t> PinnedPages,SourceVertices; // Source keys are CPU diagnostics.
        std::int32_t PositionExponent=0;
        double MaximumQuantizationDisplacement=0;
        std::uint64_t HierarchySource=0,Signature=0;
        std::uint64_t PayloadBytes=0,UncompressedBytes=0;
        [[nodiscard]] std::uint32_t PageCount() const noexcept;
    };
    struct NaniteDecodedCluster
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices,SourceVertices;
        std::uint32_t Material=0;
    };

    // Karis et al., SIGGRAPH 2021, "A Deep Dive into Nanite Virtualized Geometry",
    // slides 123-130: fixed pages, complete group parts, pinned roots and a shared
    // quantization grid. This independent format bit-packs local indices and
    // integer positions. Other vertex channels use lossless XOR bit packing.
    // A group may span pages, but a cluster never does. Parent-group membership
    // determines packing; generating-group membership determines LOD error.
    [[nodiscard]] bool BuildNanitePages(const NaniteHierarchy& hierarchy,const NanitePageSettings& settings,
        NanitePages& output,std::string& error);
    [[nodiscard]] bool ValidateNanitePages(const NaniteHierarchy& hierarchy,const NanitePages& data,std::string& error);
    // Local payload validation; full dataset validation precedes publication.
    [[nodiscard]] bool DecodeNaniteCluster(const NanitePages& data,std::uint32_t cluster,
        NaniteDecodedCluster& output,std::string& error);
}
