#pragma once

#include "Renderer/Nanite/NaniteVisibility.h"

namespace Evans::Renderer
{
    struct NaniteDepthHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const { return Value!=0; }
        friend bool operator==(NaniteDepthHandle a,NaniteDepthHandle b) { return a.Value==b.Value; }
    };
    struct NaniteVisibilityHandle
    {
        std::uint64_t Value=0;
        explicit operator bool() const { return Value!=0; }
        friend bool operator==(NaniteVisibilityHandle a,NaniteVisibilityHandle b) { return a.Value==b.Value; }
    };
    struct NaniteDepthInfo
    {
        std::uint64_t Frame=0;
        unsigned Width=0,Height=0,Words=0;
        bool Published=false;
    };
    struct NaniteFrameVisibilityInfo
    {
        std::uint64_t Frame=0;
        unsigned QueryCount=0;
        NaniteVisibilityPhase Phase=NaniteVisibilityPhase::Main;
        bool Published=false;
    };
    inline constexpr unsigned NaniteMaximumDepthCaptures=8,NaniteMaximumVisibilityBatches=8;
    inline constexpr std::size_t NaniteMaximumFrameVisibilityBytes=512ull*1024*1024;

    // Frame commands own native resources through submission completion.
    // Logical handles are separate: current-frame consumers can use pending
    // results, but only a successful Present makes them history/readback inputs.
    // The original immutable main result supplies post-pass classifications on
    // the GPU; callers cannot change the selected cut between the two phases.
    class NaniteVisibilityTransaction
    {
    public:
        [[nodiscard]] bool CanAllocate(std::size_t bytes,bool depth,std::string& error) const;
        std::uint64_t Frame();
        std::uint64_t CurrentFrame() const { return Serial; }
        void Record(NaniteDepthHandle handle,std::size_t bytes) { Depth.push_back(handle); Bytes+=bytes; }
        void Record(NaniteVisibilityHandle handle,std::size_t bytes) { Visibility.push_back(handle); Bytes+=bytes; }
        template<class DepthMap,class VisibilityMap>
        void Finish(bool publish,DepthMap& depths,VisibilityMap& visibility)
        {
            for (auto handle:Depth)
            { const auto found=depths.find(handle.Value); if (found!=depths.end()) { if (publish) found->second->Info.Published=true; else depths.erase(found); } }
            for (auto handle:Visibility)
            { const auto found=visibility.find(handle.Value); if (found!=visibility.end()) { if (publish) found->second->Info.Published=true; else visibility.erase(found); } }
            Depth.clear(); Visibility.clear(); Bytes=0; Serial=0;
        }
    private:
        std::uint64_t Serial=0;
        std::size_t Bytes=0;
        std::vector<NaniteDepthHandle> Depth;
        std::vector<NaniteVisibilityHandle> Visibility;
    };
    std::uint64_t NewNaniteVisibilityIdentity();
    [[nodiscard]] bool BuildNaniteDepthWork(unsigned width,unsigned height,NaniteVisibilityWork&,std::string&);
    [[nodiscard]] bool BuildNaniteFrameCull(const std::vector<NaniteVisibilityQuery>&,unsigned width,unsigned height,
        NaniteVisibilityPhase,NaniteVisibilityWork&,std::string&);
    // A primary depth clear establishes a defined image even when no object was
    // drawn. Captures never substitute a shadow/capture target for camera depth.
    bool NanitePrimaryDepthInitialized(const std::vector<PassRecord>&);
    [[nodiscard]] bool ReadNaniteDepthWords(const NaniteDepthInfo&,const std::vector<unsigned>&,std::vector<float>&,std::string&);
    [[nodiscard]] bool ReadNaniteDecisionWords(const NaniteFrameVisibilityInfo&,const std::vector<unsigned>&,std::vector<NaniteVisibilityState>&,std::string&);
}
