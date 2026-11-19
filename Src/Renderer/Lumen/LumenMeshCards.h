#pragma once

#include "Renderer/RayTracing/SceneRayQuery.h"

#include <functional>

namespace Evans::Renderer
{
    inline constexpr unsigned LumenCardDirectionCount=6;
    inline constexpr std::uint32_t LumenUnassignedCard=UINT32_MAX;
    struct LumenCardBuildSettings
    {
        unsigned Resolution=32,MaximumCards=12,MaximumRayLayers=32,CaptureResolution=32;
        bool RefineCoverage=true;
        float TargetCoverage=0.95f;
        std::size_t MaximumCoverageRays=512*1024;
        std::size_t MaximumSurfels=256*1024;
        float MinimumFacingCosine=0.1f;
    };
    struct LumenCardSurfel
    {
        Float3 Position,Normal;
        std::uint32_t Triangle=0,Direction=0,PixelX=0,PixelY=0,Card=LumenUnassignedCard;
        // Distances from this direction's common outer capture plane. Moving
        // the plane anywhere in this interval leaves the sampled surface visible.
        float MinimumPlane=0,MaximumPlane=0,SurfaceDepth=0;
        double Weight=0;
        bool TexelCoverageKnown=false,TexelCovered=false;
    };
    struct LumenMeshCard
    {
        Float3 Origin,Right,Up,Forward;
        // Origin is the center of the near plane. XY are half extents, Z is
        // the full capture depth. Right x Up = -Forward; depth increases [0,1].
        Float3 Extent;
        unsigned Direction=0,SurfelCount=0;
        double Weight=0;
    };
    struct LumenMeshCards
    {
        std::vector<LumenMeshCard> Cards;
        std::vector<LumenCardSurfel> Surfels;
        std::uint64_t TracedRays=0,TracedHits=0,TruncatedRays=0;
        // Assigned capture intervals and matching finite texels are separate measures.
        double TotalWeight=0,CoveredWeight=0,ResolvedWeight=0,UncheckedWeight=0;
        std::uint64_t CoverageRays=0;
        unsigned SpatialSplits=0;
        bool CoverageBudgetExhausted=false;
    };
    using LumenCardCancellation=std::function<bool()>;

    // Automatic six-direction surfel parameterization for the future surface
    // cache. Input coordinates belong to a mesh or a set of material sections;
    // instance placement remains separate. Existing alpha rules are respected.
    // The CPU prepares capture views, not final GI or a substitute GPU renderer.
    // Failure/cancellation preserves output. Uncovered and ray-budget-limited
    // surfaces remain explicit; finite ray grids cannot prove mesh coverage.
    bool BuildLumenMeshCards(const SceneRayQuery& mesh,const LumenCardBuildSettings& settings,
        LumenMeshCards& output,std::string& error,const LumenCardCancellation& cancelled={});
    // Accept a capture produced by the builder, with positive finite extents.
    Matrix4 LumenCardViewProjection(const LumenMeshCard& card) noexcept;
}
