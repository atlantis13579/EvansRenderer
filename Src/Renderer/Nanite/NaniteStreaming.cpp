#include "Renderer/Nanite/NaniteStreaming.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace Evans::Renderer
{
    namespace
    {
        bool Compatible(const NaniteHierarchy& hierarchy,const NanitePages& pages,const VirtualGeometryResidency& residency,std::string& error)
        {
            if (!ValidateNanitePages(hierarchy,pages,error) || !ValidateVirtualGeometryResidency(residency,error)) return false;
            if (residency.Source()!=pages.Signature || residency.PageSlots().size()!=pages.PageCount() || residency.PinnedPages()!=pages.PinnedPages ||
                residency.Slots().size()>std::numeric_limits<unsigned>::max()/NanitePageBytes)
            { error="Nanite residency belongs to another dataset or exceeds physical addressing"; return false; }
            return true;
        }
        bool AncestorsRefined(const NaniteHierarchy& hierarchy,unsigned group,const std::vector<bool>& refined)
        {
            for (auto parent:hierarchy.Groups[group].Parents)
            {
                const auto consumer=hierarchy.Clusters[parent].ParentGroup;
                if (consumer!=NaniteInvalid && !refined[consumer]) return false;
            }
            return true;
        }
        std::vector<unsigned> Clusters(const NaniteHierarchy& hierarchy,const std::vector<bool>& refined)
        {
            std::vector<unsigned> result;
            for (unsigned id=0;id<hierarchy.Clusters.size();++id)
            {
                const auto& cluster=hierarchy.Clusters[id];
                if ((cluster.ParentGroup==NaniteInvalid || refined[cluster.ParentGroup]) &&
                    (cluster.GeneratingGroup==NaniteInvalid || !refined[cluster.GeneratingGroup])) result.push_back(id);
            }
            return result;
        }
        float Priority(double pixels)
        { return float(std::min(pixels,double(std::numeric_limits<float>::max()))); }
    }

    bool CreateNaniteResidency(const NaniteHierarchy& hierarchy,const NanitePages& pages,std::uint32_t physicalSlots,
        VirtualGeometryResidency& output,std::string& error)
    {
        if (!ValidateNanitePages(hierarchy,pages,error)) return false;
        if (physicalSlots>std::numeric_limits<unsigned>::max()/NanitePageBytes)
        { error="Nanite physical page pool exceeds native addressing"; return false; }
        return CreateVirtualGeometryResidency(pages.Signature,pages.PageCount(),physicalSlots,pages.PinnedPages,output,error);
    }

    bool SelectNaniteResidentCutFromErrors(const NaniteHierarchy& hierarchy,const NanitePages& pages,const VirtualGeometryResidency& residency,
        const std::vector<LodView>& views,const std::vector<double>& evaluated,double maximumErrorPixels,NaniteResidentCut& output,std::string& error)
    {
        if (!Compatible(hierarchy,pages,residency,error)) return false;
        if (views.empty() || !std::isfinite(maximumErrorPixels) || maximumErrorPixels<0)
        { error="Nanite resident cut requires views and a finite nonnegative error budget"; return false; }
        for (const auto& view:views)
        {
            if (!view.Width || !view.Height) { error="Nanite resident cut view has no pixel extent"; return false; }
            for (const auto* matrix:{&view.World,&view.ViewProjection}) for (auto value:matrix->Values)
                if (!std::isfinite(value)) { error="Nanite resident cut transform is not finite"; return false; }
        }
        const auto projectQuantizationError=[&](Float3 minimum,Float3 maximum)
        {
            const double sum=pages.MaximumQuantizationDisplacement;
            const double bound=sum>0?std::nextafter(sum,std::numeric_limits<double>::infinity()):0;
            double pixels=0;
            for (auto view:views) { view.BoundsMinimum=minimum; view.BoundsMaximum=maximum; pixels=std::max(pixels,ProjectLodError(view,bound)); }
            return pixels;
        };
        NaniteResidentCut cut; cut.Source=pages.Signature; cut.CacheEpoch=residency.Epoch(); cut.CacheRevision=residency.Revision();
        if (evaluated.size()!=hierarchy.Groups.size()) { error="Nanite evaluated errors have the wrong group count"; return false; }
        cut.GroupErrorsPixels=evaluated;
        for (unsigned id=0;id<hierarchy.Groups.size();++id)
        {
            const auto& group=hierarchy.Groups[id]; const auto pixels=evaluated[id];
            if (std::isnan(pixels) || pixels<0) { error="Nanite evaluated error is invalid"; return false; }
            for (auto child:group.Children)
            {
                const auto producer=hierarchy.Clusters[child].GeneratingGroup;
                if (producer!=NaniteInvalid && pixels<evaluated[producer])
                { error="Nanite evaluated errors are not monotonic"; return false; }
            }
        }
        std::vector<unsigned> order(hierarchy.Groups.size()); std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](auto a,auto b) {
            return cut.GroupErrorsPixels[a]!=cut.GroupErrorsPixels[b]?cut.GroupErrorsPixels[a]>cut.GroupErrorsPixels[b]:a>b; });
        // Monotonic errors put every consuming ancestor before its producers.
        // Descending IDs resolve equal-error ties in the same DAG order.
        std::vector<bool> residentRefined(order.size()),targetRefined(order.size()),targetPages(pages.PageCount());
        std::vector<float> priorities(pages.PageCount()); std::size_t targetCount=pages.PinnedPages.size();
        for (auto page:pages.PinnedPages) targetPages[page]=true;
        for (auto id:order)
        {
            if (cut.GroupErrorsPixels[id]<=maximumErrorPixels) continue;
            const auto& range=pages.Groups[id]; bool complete=true; std::size_t newPages=0;
            for (unsigned page=range.First;page<range.First+range.Count;++page)
            { complete=complete && residency.PageSlots()[page]!=VirtualGeometryInvalid; newPages+=!targetPages[page]; }
            if (AncestorsRefined(hierarchy,id,residentRefined))
            {
                if (complete) residentRefined[id]=true; else ++cut.MissingGroups;
            }
            if (!AncestorsRefined(hierarchy,id,targetRefined) || targetCount+newPages>residency.Slots().size())
            { ++cut.BudgetBlockedGroups; continue; }
            targetRefined[id]=true; targetCount+=newPages;
            for (unsigned page=range.First;page<range.First+range.Count;++page)
            { targetPages[page]=true; priorities[page]=std::max(priorities[page],Priority(cut.GroupErrorsPixels[id])); }
        }
        cut.Clusters=Clusters(hierarchy,residentRefined); cut.TargetClusters=Clusters(hierarchy,targetRefined);
        for (unsigned id=0;id<order.size();++id)
        { if (residentRefined[id]) cut.RefinedGroups.push_back(id); if (targetRefined[id]) cut.TargetGroups.push_back(id); }
        for (unsigned page=0;page<targetPages.size();++page) if (targetPages[page])
        { cut.TargetPages.push_back(page); cut.Requests.push_back({page,priorities[page]}); }
        for (auto id:cut.Clusters)
        {
            const auto& cluster=hierarchy.Clusters[id];
            if (residency.PageSlots()[pages.Clusters[id].Page]==VirtualGeometryInvalid)
            { error="Nanite resident cut references an unavailable payload"; return false; }
            cut.TriangleCount+=cluster.Indices.size()/3;
            const auto own=cluster.GeneratingGroup==NaniteInvalid?projectQuantizationError(cluster.BoundsMinimum,cluster.BoundsMaximum):cut.GroupErrorsPixels[cluster.GeneratingGroup];
            cut.MaximumSelectedErrorPixels=std::max(cut.MaximumSelectedErrorPixels,own);
        }
        if (cut.Clusters.empty() || cut.TargetClusters.empty()) { error="Nanite resident cut lost root coverage"; return false; }
        output=std::move(cut); error.clear(); return true;
    }

    bool PlanNaniteResidency(const NaniteHierarchy& hierarchy,const NanitePages& pages,const VirtualGeometryResidency& residency,
        const NaniteResidentCut& cut,std::uint64_t frame,VirtualGeometryResidencyPlan& output,std::string& error)
    {
        if (!Compatible(hierarchy,pages,residency,error)) return false;
        const auto fail=[&](const char* message) { error=message; return false; };
        if (cut.Source!=pages.Signature || cut.CacheEpoch!=residency.Epoch() || cut.CacheRevision!=residency.Revision() ||
            cut.TargetPages.size()!=cut.Requests.size()) return fail("Nanite streaming feedback is stale or inconsistent");
        std::vector<bool> refined(hierarchy.Groups.size()),needed(pages.PageCount());
        for (auto page:pages.PinnedPages) needed[page]=true;
        unsigned previous=NaniteInvalid;
        for (auto group:cut.TargetGroups)
        {
            if (group>=refined.size() || (previous!=NaniteInvalid && group<=previous)) return fail("Nanite target group IDs are invalid or repeated");
            previous=group; refined[group]=true; const auto& range=pages.Groups[group];
            for (unsigned page=range.First;page<range.First+range.Count;++page) needed[page]=true;
        }
        for (auto group:cut.TargetGroups) if (!AncestorsRefined(hierarchy,group,refined)) return fail("Nanite target omits an ancestor page dependency");
        std::vector<unsigned> targetPages; for (unsigned page=0;page<needed.size();++page) if (needed[page]) targetPages.push_back(page);
        if (targetPages!=cut.TargetPages || targetPages.size()>residency.Slots().size() || Clusters(hierarchy,refined)!=cut.TargetClusters)
            return fail("Nanite target pages or clusters do not describe a complete admitted cut");
        for (unsigned at=0;at<targetPages.size();++at)
            if (cut.Requests[at].Page!=targetPages[at] || !std::isfinite(cut.Requests[at].Priority) || cut.Requests[at].Priority<0)
                return fail("Nanite target page requests are inconsistent");
        VirtualGeometryResidencyPlan candidate;
        if (!PlanVirtualGeometryResidency(residency,cut.Requests,frame,std::numeric_limits<unsigned>::max(),candidate,error)) return false;
        if (!candidate.DeferredPages().empty()) return fail("Nanite target transaction left an admitted page unresolved");
        output=std::move(candidate); error.clear(); return true;
    }
}
