#pragma once

#include "Renderer/RayTracing/SceneRayShadingUpload.h"
#include "Renderer/Lighting/SceneLightCollection.h"

namespace Evans::Renderer
{
    // A frame retains the matched geometry/material pair and the complete light
    // collection until command completion or cancellation. Lighting changes do
    // not rebuild scene geometry, and visibility-only AO uses Geometry directly.
    struct RayRadianceFrame
    {
        std::shared_ptr<const RaySceneUpload> Geometry;
        std::shared_ptr<const RaySceneShadingUpload> Shading;
        std::shared_ptr<const SceneLightCollection> Lights;
        LightingState Lighting;
    };
    inline constexpr std::size_t RayRadianceBufferCount=RaySceneShadingBufferCount+1;
    inline std::array<RaySceneBufferData,RayRadianceBufferCount> GetRayRadianceBufferData(const RayRadianceFrame& frame)
    {
        const auto shading=GetRaySceneShadingBufferData(*frame.Shading);
        return {shading[0],shading[1],shading[2],RaySceneData(frame.Lights->GetShaderLights())};
    }
}
