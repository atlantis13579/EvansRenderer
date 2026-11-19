#include "Renderer/Nanite/NaniteFrameVisibility.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cmath>

namespace Evans::Renderer
{
    std::uint64_t NewNaniteVisibilityIdentity()
    {
        static std::atomic<std::uint64_t> next{1};
        return next.fetch_add(1,std::memory_order_relaxed);
    }
    std::uint64_t NaniteVisibilityTransaction::Frame()
    { if (!Serial) Serial=NewNaniteVisibilityIdentity(); return Serial; }
    bool NaniteVisibilityTransaction::CanAllocate(std::size_t bytes,bool depth,std::string& error) const
    {
        if (bytes>NaniteMaximumFrameVisibilityBytes-Bytes ||
            (depth?Depth.size()>=NaniteMaximumDepthCaptures:Visibility.size()>=NaniteMaximumVisibilityBatches))
        { error="Nanite frame visibility exceeds its memory or command budget"; return false; }
        return true;
    }
    // Two-pass occlusion tests the current cut against previous depth, then
    // revisits deferred clusters against this frame's main-pass depth. Maximum
    // depth reduction preserves holes; uncertain projections remain visible.
    // Karis, Stubbe and Wihlidal, "A Deep Dive into Nanite Virtualized Geometry"
    // (SIGGRAPH 2021), two-pass occlusion.
    bool BuildNaniteDepthWork(unsigned width,unsigned height,NaniteVisibilityWork& output,std::string& error)
    {
        if (!width || !height || width>NaniteVisibilityMaximumSide || height>NaniteVisibilityMaximumSide)
        { error="Invalid native Nanite depth dimensions"; return false; }
        NaniteVisibilityWork work; work.Input={0};
        unsigned w=width,h=height,previousOffset=0,previousWidth=0,previousHeight=0,level=0;
        for (;;)
        {
            NaniteVisibilityDispatch job; job.Pass=level?0:2;
            job.Constants.Counts={0,width,height,0}; job.Constants.Source={previousOffset,previousWidth,previousHeight,level};
            const auto groups=(w*h+NaniteVisibilityThreads-1)/NaniteVisibilityThreads;
            job.GroupsX=std::min(groups,NaniteVisibilityColumns); job.GroupsY=(groups+job.GroupsX-1)/job.GroupsX;
            job.Constants.Target={work.PyramidWords,w,h,job.GroupsX}; work.Dispatches.push_back(job);
            previousOffset=work.PyramidWords; previousWidth=w; previousHeight=h; work.PyramidWords+=w*h; ++level;
            if (w==1 && h==1) break; w=(w+1)/2; h=(h+1)/2;
        }
        work.OutputWords=work.PyramidWords;
        for (auto& job:work.Dispatches) job.Constants.Layout={0,work.PyramidWords,level,0};
        output=std::move(work); error.clear(); return true;
    }
    bool BuildNaniteFrameCull(const std::vector<NaniteVisibilityQuery>& queries,unsigned width,unsigned height,
        NaniteVisibilityPhase phase,NaniteVisibilityWork& output,std::string& error)
    {
        if (queries.empty() || !width || !height || width>NaniteVisibilityMaximumSide || height>NaniteVisibilityMaximumSide || unsigned(phase)>1)
        { error="Invalid native Nanite culling dimensions or phase"; return false; }
        // Main classifications remain GPU-owned; validate the immutable query
        // records directly without building a synthetic depth upload or dispatch.
        if (!ValidateNaniteVisibilityQueries(queries,error)) return false;
        NaniteVisibilityWork work; work.Input.resize(queries.size()*sizeof(NaniteVisibilityQuery)/sizeof(unsigned));
        std::memcpy(work.Input.data(),queries.data(),work.Input.size()*sizeof(unsigned)); work.OutputWords=unsigned(queries.size());
        NaniteVisibilityDispatch job; job.Pass=3;
        const auto groups=(unsigned(queries.size())+NaniteVisibilityThreads-1)/NaniteVisibilityThreads;
        job.GroupsX=std::min(groups,NaniteVisibilityColumns); job.GroupsY=(groups+job.GroupsX-1)/job.GroupsX;
        job.Constants.Counts={unsigned(queries.size()),width,height,unsigned(phase)}; job.Constants.Target.W=job.GroupsX;
        work.Dispatches.push_back(job); output=std::move(work); error.clear(); return true;
    }
    bool NanitePrimaryDepthInitialized(const std::vector<PassRecord>& passes)
    {
        return std::any_of(passes.begin(),passes.end(),[](const auto& pass)
        {
            const auto& d=pass.Description;
            return !d.Target && !d.DepthPeelingLayer && d.ClearDepthEnabled && (d.Type==PassType::Color || d.Type==PassType::GBuffer);
        });
    }
    bool ReadNaniteDepthWords(const NaniteDepthInfo& info,const std::vector<unsigned>& words,std::vector<float>& output,std::string& error)
    {
        if (!info.Published || words.size()!=info.Words) { error="Nanite depth feedback is pending or has invalid dimensions"; return false; }
        std::vector<float> result(words.size()); std::memcpy(result.data(),words.data(),words.size()*sizeof(unsigned));
        for (float value:result) if (!std::isfinite(value) || value<0 || value>1) { error="Invalid native Nanite depth value"; return false; }
        output=std::move(result); error.clear(); return true;
    }
    bool ReadNaniteDecisionWords(const NaniteFrameVisibilityInfo& info,const std::vector<unsigned>& words,std::vector<NaniteVisibilityState>& output,std::string& error)
    {
        if (!info.Published || words.size()!=info.QueryCount) { error="Nanite visibility feedback is pending or has invalid dimensions"; return false; }
        std::vector<NaniteVisibilityState> result; result.reserve(words.size());
        for (unsigned word:words)
        {
            if (word<1 || word>5 || (info.Phase==NaniteVisibilityPhase::Main?word>3:word==3))
            { error="Invalid native Nanite visibility transition"; return false; }
            result.push_back(NaniteVisibilityState(word));
        }
        output=std::move(result); error.clear(); return true;
    }
}
