#pragma once

#include "Renderer/Nanite/NanitePages.h"
#include "Renderer/VirtualGeometry/VirtualGeometryResidency.h"

namespace Evans::Renderer
{
    struct NaniteResidentCut
    {
        std::vector<std::uint32_t> Clusters,RefinedGroups;
        std::vector<std::uint32_t> TargetClusters,TargetGroups,TargetPages;
        std::vector<VirtualGeometryPageRequest> Requests;
        std::vector<double> GroupErrorsPixels;
        std::uint64_t Source=0,CacheEpoch=0,CacheRevision=0,TriangleCount=0;
        std::uint32_t MissingGroups=0,BudgetBlockedGroups=0;
        double MaximumSelectedErrorPixels=0;
    };

    [[nodiscard]] bool CreateNaniteResidency(const NaniteHierarchy& hierarchy,const NanitePages& pages,
        std::uint32_t physicalSlots,VirtualGeometryResidency& output,std::string& error);
    // Admission of externally evaluated monotonic group errors. Native callers
    // validate state/source identity and verify the resulting resident cut.
    [[nodiscard]] bool SelectNaniteResidentCutFromErrors(const NaniteHierarchy&,const NanitePages&,
        const VirtualGeometryResidency&,const std::vector<LodView>&,const std::vector<double>& groupErrors,
        double maximumErrorPixels,NaniteResidentCut&,std::string& error);
    // Prepare the complete admitted working set. A native uploader must keep
    // the active physical pool alive until every upload in this transaction
    // finishes; only then may it publish/commit the candidate page table.
    [[nodiscard]] bool PlanNaniteResidency(const NaniteHierarchy& hierarchy,const NanitePages& pages,
        const VirtualGeometryResidency& residency,const NaniteResidentCut& cut,std::uint64_t frame,
        VirtualGeometryResidencyPlan& output,std::string& error);
}
