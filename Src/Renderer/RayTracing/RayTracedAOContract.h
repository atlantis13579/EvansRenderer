#pragma once

#include "Renderer/Lighting/AmbientOcclusionContract.h"
#include "Renderer/RayTracing/SceneRayUpload.h"
#include "Techniques/RayTracedAO.h"

namespace Evans::Renderer
{
    // Finite-radius, cosine-weighted scene visibility. Each API independently
    // walks the uploaded BVH and evaluates cutout coverage at the ray hit.
    // This is native shader BVH traversal, not a screen-space depth test or a
    // CPU AO image. It does not require a hardware ray-query extension.
    // Estimate -> horizontal/vertical bilateral -> lost-ambient composition.
    // Raster textures retain the AO slot order; node/triangle/material/coverage
    // storage is separate. Empty scenes bind valid dummy storage with zero counts.
    struct RayTracedAOConstants
    {
        AmbientOcclusionConstants Surface;
        ShaderUInt4 SceneCounts; // Nodes, triangles, materials, scalar texels.
        ShaderUInt4 Sampling; // Ray count, reserved.
    };
    static_assert(sizeof(RayTracedAOConstants)==208 && offsetof(RayTracedAOConstants,SceneCounts)==176 &&
        offsetof(RayTracedAOConstants,Sampling)==192,"Ray-traced AO uniform ABI changed");

    inline bool BuildRayTracedAOConstants(const AmbientOcclusionDesc& desc,Techniques::AmbientOcclusionSettings surface,
        Techniques::RayTracedAOSettings sampling,const RaySceneUpload& scene,int width,int height,
        Techniques::AmbientOcclusionPassKind pass,RayTracedAOConstants& output)
    {
        output={};
        if (static_cast<unsigned>(pass)>static_cast<unsigned>(Techniques::AmbientOcclusionPassKind::Composite) ||
            scene.Nodes.size()>UINT32_MAX || scene.Triangles.size()>UINT32_MAX ||
            scene.Materials.size()>UINT32_MAX || scene.CoverageTexels.size()>UINT32_MAX) return false;
        // Reuse projection/material layout and world-space radius/bias handling;
        // GTAO supplies optical camera Z for the bilateral filter, including
        // asymmetric projections, and retains a world-space origin bias.
        surface.Mode=Techniques::AmbientOcclusionMode::GTAO;
        if (!BuildAmbientOcclusionConstants(desc,surface,width,height,pass,output.Surface)) return false;
        output.SceneCounts={static_cast<std::uint32_t>(scene.Nodes.size()),static_cast<std::uint32_t>(scene.Triangles.size()),
            static_cast<std::uint32_t>(scene.Materials.size()),static_cast<std::uint32_t>(scene.CoverageTexels.size())};
        output.Sampling={Techniques::SanitizeRayTracedAOSettings(sampling).RayCount,0,0,0}; return true;
    }

    // All AO estimators share the surface prefix. Native backends can reserve
    // one packet size while each independent program consumes its own layout.
    inline bool BuildAmbientOcclusionFrameConstants(const AmbientOcclusionDesc& desc,
        Techniques::AmbientOcclusionSettings settings,const RaySceneUpload* scene,int width,int height,
        Techniques::AmbientOcclusionPassKind pass,RayTracedAOConstants& output)
    {
        output={};
        if (settings.Mode==Techniques::AmbientOcclusionMode::RayTraced)
            return scene && BuildRayTracedAOConstants(desc,settings,settings.RayTracing,*scene,width,height,pass,output);
        return BuildAmbientOcclusionConstants(desc,settings,width,height,pass,output.Surface);
    }
}
