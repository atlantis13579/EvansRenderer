#pragma once

#include "Renderer/Nanite/NanitePages.h"

#include <filesystem>
#include <memory>

namespace Evans::Renderer
{
    struct NaniteAssetSection
    {
        const std::vector<Vertex>* Vertices=nullptr;
        const std::vector<std::uint32_t>* Indices=nullptr;
        std::uint64_t GeometryHash=0;
    };
    struct NaniteAssetInput
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices,Materials;
    };
    // Stable content ordering makes offline and interactive section assembly
    // produce the same asset key, independent of allocation addresses.
    [[nodiscard]] bool NaniteAssetSectionLess(const NaniteAssetSection&,const NaniteAssetSection&);
    [[nodiscard]] bool AssembleNaniteAsset(const std::vector<NaniteAssetSection>&,NaniteAssetInput&,std::string& error);

    // Cooked metadata stays resident; fixed-size payload pages are fetched into
    // a bounded host LRU before the existing native cache transaction begins.
    // The disk worker never touches the RHI. A failed read cannot publish a new
    // GPU page mapping, so the previous complete cut remains drawable.
    class NanitePageSource
    {
    public:
        virtual ~NanitePageSource()=default;
        [[nodiscard]] virtual std::uint32_t Count() const noexcept=0;
        [[nodiscard]] virtual bool Read(std::uint32_t,NanitePage&,std::string& error) const=0;
        [[nodiscard]] virtual bool Prefetch(const std::vector<std::uint32_t>&,bool& ready,std::string& error)=0;
        [[nodiscard]] virtual bool Matches(const NaniteHierarchy&,const NanitePages&,std::string& error) const=0;
    };

    // The key covers source attributes, topology, material sections and the
    // cooker revision. Each archive stores explicit metadata and per-page hashes;
    // truncation, stale sources and corrupt pages are rejected before GPU upload.
    [[nodiscard]] std::uint64_t NaniteAssetKey(const std::vector<Vertex>&,
        const std::vector<std::uint32_t>& indices,const std::vector<std::uint32_t>& materials);
    [[nodiscard]] bool WriteNaniteAsset(const std::filesystem::path&,std::uint64_t key,
        const NaniteHierarchy&,const NanitePages&,std::string& error);
    [[nodiscard]] bool OpenNaniteAsset(const std::filesystem::path&,std::uint64_t key,
        std::uint32_t hostPages,NaniteHierarchy&,NanitePages&,std::string& error);
    [[nodiscard]] bool LoadOrBuildNaniteAsset(const std::vector<Vertex>&,
        const std::vector<std::uint32_t>& indices,const std::vector<std::uint32_t>& materials,
        std::uint32_t physicalPages,NaniteHierarchy&,NanitePages&,std::string& error);
    [[nodiscard]] bool ReadNanitePage(const NanitePages&,std::uint32_t,NanitePage&,std::string& error);
    [[nodiscard]] bool PrefetchNanitePages(const NanitePages&,const std::vector<std::uint32_t>&,
        bool& ready,std::string& error);

    // Cancelled scene generations release their request without waiting for a
    // cooker. Workers own source copies and never retain MeshResource pointers.
    // Cooking and page I/O use separate shared queues to avoid disk starvation.
    class NaniteAssetRequest
    {
    public:
        // Busy admission succeeds with admitted=false and makes no source copy.
        // The shared worker assembles accepted sections; a single source larger
        // than the queued-input cap is an error rather than an endless retry.
        [[nodiscard]] static bool TryStart(const std::vector<NaniteAssetSection>&,std::uint32_t physicalPages,
            std::unique_ptr<NaniteAssetRequest>& output,bool& admitted,std::string& error);
        ~NaniteAssetRequest();
        NaniteAssetRequest(const NaniteAssetRequest&)=delete;
        NaniteAssetRequest& operator=(const NaniteAssetRequest&)=delete;
        [[nodiscard]] bool Poll(bool& ready,NaniteHierarchy&,NanitePages&,std::string& error);
    private:
        struct State;
        explicit NaniteAssetRequest(std::shared_ptr<State> state):m_State(std::move(state)) {}
        std::shared_ptr<State> m_State;
    };
    inline constexpr std::uint32_t NaniteAssetMaximumRequests=4;
    inline constexpr std::uint64_t NaniteAssetMaximumQueuedSourceBytes=128ull*1024*1024;
    struct NaniteAssetQueueStatistics { std::uint32_t Requests=0; std::uint64_t SourceBytes=0; };
    [[nodiscard]] NaniteAssetQueueStatistics GetNaniteAssetQueueStatistics();
}
