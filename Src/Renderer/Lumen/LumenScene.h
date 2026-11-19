#pragma once

#include "Renderer/Lumen/LumenMeshRepresentation.h"
#include "Renderer/Lumen/LumenSurfaceLighting.h"

namespace Evans::Renderer
{
    struct LumenSceneIdentity
    {
        // Geometry includes placement, visibility, partition and card layout.
        // Occlusion additionally changes when an instance's cast-shadow flag changes.
        std::uint64_t Epoch=1,Geometry=1,Material=1,Occlusion=1;
    };
    struct LumenSceneInstance
    {
        std::uint64_t Key=0; // Stable caller identity, independent of vector order.
        std::shared_ptr<const LumenMeshRepresentation> Mesh;
        Matrix4 World;
        std::vector<LumenSurfaceDraw> Draws;
        bool Visible=true,CastShadow=true;
    };
    struct LumenSceneBudget
    {
        unsigned MaximumCards=LumenSurfaceMaximumCards,MaximumInstances=LumenDistanceMaximumInstances;
        std::size_t MaximumAtlasTexels=LumenSurfaceMaximumTexels,MaximumDistanceWords=LumenDistanceMaximumWords;
        std::uint64_t MaximumCaptureTriangles=64*1024*1024;
    };
    struct LumenSceneUpload
    {
        LumenSceneIdentity Identity;
        unsigned CardSide=16;
        LumenSceneBudget Budget;
        std::vector<LumenSceneInstance> Instances;
        std::shared_ptr<const LumenHairVolume> Hair;
        bool HardwareRayTracing=false; // Requires matching immutable per-part triangle exports.
    };
    struct LumenSceneMeshDemand
    {
        std::uint64_t Key=0;
        unsigned Instances=1,MaximumParts=64;
        std::size_t Triangles=0;
    };
    struct LumenScenePart
    {
        std::uint64_t InstanceKey=0;
        unsigned Part=0;
        LumenMeshPartRange Range;
    };
    struct LumenScenePlan
    {
        LumenSurfaceUpload Surface;
        LumenSurfaceLayout Layout;
        LumenDistanceSceneData Distance;
        std::vector<LumenScenePart> Parts;
        std::vector<unsigned> CardInstances;
        std::uint64_t CaptureTriangles=0;
    };

    // Assemble the entire transport scene, independent of camera LOD/frustum.
    // Distance UserId addresses Parts; that table restores stable instance and
    // compact-part identities. Shared immutable fields are packed only once.
    // Exceeding a budget fails atomically; no object silently becomes a GI miss.
    bool BuildLumenScenePlan(const LumenSceneUpload& upload,LumenScenePlan& output,std::string& error);
    bool ValidateLumenScenePlan(const LumenScenePlan& scene,std::string& error);
    // Reserve at least one domain for every source, then distribute additional
    // domains by triangle load per part. Repeated instances charge every atlas
    // placement but only one shared field. Component grouping preserves all
    // source triangles; reducing proxy domains can reduce card/SDF accuracy.
    // Final assembly still checks measured distance bytes and capture work.
    bool AllocateLumenSceneMeshParts(const std::vector<LumenSceneMeshDemand>& meshes,const LumenMeshBuildSettings& settings,
        unsigned cardSide,const LumenSceneBudget& budget,std::vector<LumenMeshBuildSettings>& output,std::string& error);
}
