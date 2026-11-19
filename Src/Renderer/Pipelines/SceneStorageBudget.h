#pragma once

#include "Renderer/Pipelines/VisibilityBuffer.h"
#include "Renderer/RayTracing/SceneRayShadingUpload.h"

namespace Evans::Renderer
{
    // These paths expand instances into world-space triangle tables. Check the
    // largest block before building it; a binary BVH needs at most two nodes
    // per triangle. Material/texture upload limits remain backend checks.
    [[nodiscard]] inline bool SceneTriangleStorageFits(Techniques::Technique technique,
        std::uint64_t triangles, std::uint64_t maximumBlockBytes)
    {
        using Techniques::Technique;
        if (!maximumBlockBytes) return true;
        std::size_t stride=0;
        switch (technique)
        {
        case Technique::VisibilityBuffer: case Technique::MeshShaders:
            stride=sizeof(VisibilityTriangle); break;
        case Technique::RayTracedAO: case Technique::RayTracedReflections:
        case Technique::RayTracedGI: case Technique::PathTracing: case Technique::Hybrid:
            stride=std::max({sizeof(RaySceneTriangle),2*sizeof(RaySceneNode),sizeof(RaySceneSurface)}); break;
        default: return true;
        }
        return triangles<=maximumBlockBytes/stride;
    }

    [[nodiscard]] inline Techniques::Settings ConstrainSceneTriangleStorage(Techniques::Settings settings,
        std::uint64_t triangles, std::uint64_t maximumBlockBytes)
    {
        using namespace Techniques;
        if (!SceneTriangleStorageFits(Technique::VisibilityBuffer,triangles,maximumBlockBytes))
        {
            if (settings.Pipeline==RenderingPipeline::VisibilityBuffer) settings.Pipeline=RenderingPipeline::Deferred;
            settings.MeshShaders=false;
        }
        if (!SceneTriangleStorageFits(Technique::PathTracing,triangles,maximumBlockBytes))
        {
            if (settings.Pipeline==RenderingPipeline::PathTracing || settings.Pipeline==RenderingPipeline::Hybrid)
                settings.Pipeline=RenderingPipeline::Deferred;
            if (settings.AmbientOcclusion.Mode==AmbientOcclusionMode::RayTraced) settings.AmbientOcclusion.Mode=AmbientOcclusionMode::None;
            if (settings.Reflections.Mode==ReflectionMode::RayTraced) settings.Reflections.Mode=ReflectionMode::None;
            if (settings.GlobalIllumination.Mode==GlobalIlluminationMode::RayTraced) settings.GlobalIllumination.Mode=GlobalIlluminationMode::None;
        }
        return settings;
    }
}
