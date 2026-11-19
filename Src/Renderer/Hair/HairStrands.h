#pragma once

#include "Renderer/RayTracing/SceneRayQuery.h"
#include "Techniques/HairFur.h"
#include <functional>

namespace Evans::Renderer
{
    struct HairVertex
    {
        Float3 Position,Tangent;
        Float4 Reflectance,Emission; // Linear source material; no baked illumination.
        Float4 Ribbon; // Signed radius, cross-section h, curve height, negative W suppresses shadow casting.
    };
    static_assert(sizeof(HairVertex)==72 && offsetof(HairVertex,Ribbon)==56,"Hair vertex ABI changed");
    struct HairStrandGeometry
    {
        std::vector<HairVertex> Vertices;
        std::vector<std::uint32_t> Indices;
        std::vector<std::uint32_t> VertexSources; // CPU-only source instance for rigid posing.
        Float3 BoundsMinimum,BoundsMaximum;
        float GrowthLength=0,RootRadius=0;
        std::uint32_t CandidateRoots=0,StrandCount=0,RejectedRoots=0;
    };
    // The caller supplies an opaque/cutout query of the desired existing meshes.
    // Seed candidates use world-area-weighted triangles and uniform barycentrics;
    // rejected cutouts remain gaps. The query's outward-hemisphere normals define quadratic
    // centerlines. Native vertex stages orient each ribbon toward the camera.
    // Camera, lighting, scattering roughness and tint do not alter this producer.
    // Output publication is transactional, including cancellation and bad bounds.
    // Native integration must freeze this geometry across camera depth, shadow,
    // color and velocity replays. Fiber color/depth precedes opaque MSAA resolve;
    // an invalid material-normal marker keeps mesh-only SSS/GI receivers from
    // relighting fibers. Camera-depth, shadow, capture and presented-history
    // velocity replays use the RHI; renderer scheduling is a separate stage.
    bool BuildHairStrands(const SceneRayQuery& scene,const Techniques::HairFurSettings& settings,
        HairStrandGeometry& output,std::string& error,const std::function<bool()>& cancelled={});
}
