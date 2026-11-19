#pragma once

#include "Renderer/Geometry/LevelOfDetail.h"
#include "Renderer/VirtualGeometry/VirtualGeometryResidency.h"

namespace Evans::Renderer
{
    struct VirtualGeometryHierarchySettings
    {
        VirtualGeometryPageLimits Pages;
        LodBuildSettings Simplification; // one simplification level per parent; MaximumLevels is ignored
        std::uint32_t ChildrenPerNode=4,MaximumDepth=12;
        std::size_t MaximumNodes=256*1024;
    };
    struct VirtualGeometryNode
    {
        std::uint32_t Parent=VirtualGeometryInvalid,Material=0;
        std::uint32_t FirstChild=0,ChildCount=0,FirstCluster=0,ClusterCount=0,FirstPage=0,PageCount=0;
        // Range in SourceTriangles: the exact original region owned by this node.
        std::uint32_t FirstTriangle=0,TriangleCount=0,RepresentationTriangles=0;
        Float3 BoundsMinimum,BoundsMaximum; // original region, including removed vertices
        double MaximumDisplacement=0;
    };
    struct VirtualGeometryHierarchy
    {
        VirtualGeometryPages Data;
        std::vector<VirtualGeometryNode> Nodes;
        std::vector<std::uint32_t> Children,Clusters,Pages,Roots,SourceTriangles;
        std::uint64_t Signature=0; // includes page identity and all selection metadata
    };
    struct VirtualGeometryCutSettings
    {
        double MaximumErrorPixels=1,HysteresisFraction=0.15;
    };
    struct VirtualGeometryCut
    {
        std::vector<std::uint32_t> Nodes,Clusters;
        std::vector<std::uint32_t> TargetNodes,TargetPages;
        std::uint64_t Source=0,CacheEpoch=0,CacheRevision=0;
        // Includes current resident demand and the next refinement/coarsening
        // frontier. Already resident pages must remain visible to the cache.
        std::vector<VirtualGeometryPageRequest> Requests;
        std::uint32_t MissingDetailNodes=0,BudgetBlockedNodes=0,OverBudgetNodes=0;
        double MaximumSelectedErrorPixels=0;
    };
    struct VirtualGeometryTransition { std::uint32_t Node=0; double Priority=0; bool Refine=false; };

    // A material-separated forest combines adjacent leaf regions. Endpoint QEM
    // keeps every region boundary and global coincident seam fixed. Parent error
    // is max(child error) + the local contraction bound. Unchanged representations
    // reuse child clusters; root-referenced payloads are pinned before packing.
    // This is an independent hierarchy, not Unreal Engine's Nanite format.
    [[nodiscard]] bool BuildVirtualGeometryHierarchy(const std::vector<Vertex>& vertices,
        const std::vector<std::uint32_t>& indices,const std::vector<std::uint32_t>& triangleMaterials,
        const VirtualGeometryHierarchySettings& settings,VirtualGeometryHierarchy& output,std::string& error);
    [[nodiscard]] bool ValidateVirtualGeometryHierarchy(const VirtualGeometryHierarchy& hierarchy,std::string& error);
    [[nodiscard]] std::vector<std::uint32_t> VirtualGeometryPinnedPages(const VirtualGeometryHierarchy& hierarchy);

    // Admit the resident cut and disjoint transitions selected by native feedback.
    [[nodiscard]] bool AdmitVirtualGeometryTransitions(const VirtualGeometryHierarchy& hierarchy,const VirtualGeometryResidency& residency,
        VirtualGeometryCut current,const std::vector<VirtualGeometryTransition>& transitions,VirtualGeometryCut& output,std::string& error);
    // Plan the complete admitted target. Execute uploads into a separate candidate
    // pool, optionally over several frames, then commit once all uploads complete.
    // A per-frame upload budget must not publish partial target transactions.
    [[nodiscard]] bool PlanVirtualGeometryCutResidency(const VirtualGeometryHierarchy& hierarchy,
        const VirtualGeometryResidency& residency,const VirtualGeometryCut& cut,std::uint64_t frame,
        VirtualGeometryResidencyPlan& output,std::string& error);
}
