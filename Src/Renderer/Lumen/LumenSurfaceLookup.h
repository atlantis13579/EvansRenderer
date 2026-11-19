#pragma once

#include "Renderer/Lumen/LumenSurfaceCache.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenLookupThreads=64,LumenLookupColumns=1024,LumenLookupMaximumCandidates=64;
    inline constexpr std::size_t LumenLookupMaximumQueries=1024*1024;
    inline constexpr unsigned LumenLookupMissingCard=UINT32_MAX;
    struct alignas(16) LumenLookupCard
    {
        Matrix4 Projection,InverseProjection; // Both operate in capture-relative world coordinates.
        Float4 Origin,BackwardScale; // Unit direction toward the capture plane; relative-coordinate scale in W.
        ShaderUInt4 Tile; // Atlas tile XY, projection group, original card index.
    };
    static_assert(sizeof(LumenLookupCard)==176,"Lumen lookup card is eleven vectors");
    struct LumenRadianceMapping
    {
        std::vector<std::uint32_t> Words;
        unsigned CardVectorOffset=0;
    };
    // Instance headers reference compact card-index lists; aligned lookup cards
    // follow them. Both hit lighting and radiosity use the same ownership map.
    bool BuildLumenRadianceMapping(const LumenSurfaceLayout& layout,const std::vector<LumenSurfaceCapture>& captures,
        unsigned instanceCount,const std::vector<unsigned>& cardInstances,LumenRadianceMapping& output,std::string& error);
    struct alignas(16) LumenSurfaceQuery
    {
        Float4 PositionRadius{0,0,0,0}; // World position and permitted world-space plane distance.
        Float4 NormalMinimumCosine{0,1,0,0.5f}; // Unit geometric normal and minimum material-normal agreement.
        ShaderUInt4 Range; // First card, count, reserved ZW; one mesh instance's cards only.
    };
    struct LumenSurfaceLookupUpload
    {
        std::vector<LumenSurfaceQuery> Queries;
        float MinimumFacingCosine=0.1f;
    };
    struct alignas(16) LumenSurfaceSample
    {
        Float4 AlbedoMetallic{0,0,0,0},NormalCoverage{0,0,0,0},EmissionRoughness{0,0,0,0};
        ShaderUInt4 Metadata{LumenLookupMissingCard,0,0,0}; // Dominant card, contributing direction count, valid flag, reserved W.
    };
    static_assert(sizeof(LumenSurfaceQuery)==48 && sizeof(LumenSurfaceSample)==64,"Lumen surface lookup query/result ABI changed");
    struct alignas(16) LumenSurfaceLookupFrame
    {
        ShaderUInt4 Counts,Atlas; // Queries/side/columns/tile stride; width/height/dispatch stride/cards.
        Float4 Parameters; // Facing threshold, relative arithmetic error, world-position rounding, reserved W.
    };
    static_assert(sizeof(LumenSurfaceLookupFrame)==48,"Lumen surface lookup frame ABI changed");
    struct LumenSurfaceLookupResult
    {
        LumenSurfaceIdentity Identity;
        std::vector<LumenSurfaceSample> Samples;
    };

    // Surface-cache lookup follows the direction-weighted, depth-gated atlas
    // interpolation described by Wright et al., "Lumen" (SIGGRAPH 2022).
    // This teaching implementation selects the nearest accepted layer in each
    // projection group before blending directions. Capture remains independent
    // of the camera; uncovered samples never borrow unrelated screen pixels.
    bool BuildLumenLookupCards(const LumenSurfaceLayout& layout,const std::vector<LumenSurfaceCapture>& captures,
        std::vector<LumenLookupCard>& output,std::string& error);
    bool BuildLumenSurfaceLookupFrame(const LumenSurfaceLayout& layout,const LumenSurfaceLookupUpload& upload,
        LumenSurfaceLookupFrame& output,std::string& error);
    bool ValidateLumenSurfaceLookupResult(const LumenSurfaceIdentity& identity,const LumenSurfaceLookupUpload& upload,
        const LumenSurfaceLookupResult& result,std::string& error);
}
