#pragma once

#include "Renderer/Lumen/LumenMeshPartition.h"
#include "Renderer/Lumen/LumenHardwareRayTracing.h"
#include "Renderer/Lumen/LumenDistanceScene.h"
#include "Renderer/Lumen/LumenSurfaceCache.h"

namespace Evans::Renderer
{
    struct LumenMeshBuildSettings
    {
        LumenMeshPartitionSettings Partition;
        LumenCardBuildSettings Cards;
        LumenDistanceBuildSettings Distance;
        bool HardwareRayTracing=false; // Retain exact triangle/coverage geometry alongside SDFs.
        std::size_t MaximumSurfels=2*1024*1024,MaximumDistanceBytes=32*1024*1024;
    };
    struct LumenMeshRepresentation
    {
        LumenMeshPartition Geometry;
        std::vector<LumenMeshCards> Cards;
        // A null field explicitly denotes a part with zero geometric area.
        std::vector<std::shared_ptr<const LumenDistanceField>> Fields;
        std::vector<std::shared_ptr<const LumenHardwareMesh>> Hardware; // Empty unless explicitly requested.
        std::size_t SurfelCount=0,DistanceBytes=0;
        unsigned CardCount=0;
    };
    struct LumenMeshPartRange
    {
        unsigned FirstCard=0,CardCount=0,DistanceInstance=LumenDistanceInvalid;
    };
    struct LumenMeshInstanceData
    {
        LumenSurfaceUpload Surface;
        LumenDistanceSceneData Distance;
        std::vector<LumenMeshPartRange> Parts;
    };
    struct LumenMeshPlacement
    {
        LumenSurfaceUpload Surface;
        LumenDistanceSceneUpload Distance;
        std::vector<LumenMeshPartRange> Parts;
    };

    // One local immutable proxy per source geometry/material configuration.
    // Instance transforms are deliberately excluded from this preprocessing;
    // source textures and cutout behavior are retained in each part's query.
    bool BuildLumenMeshRepresentation(const SceneRayMeshView& source,const Techniques::Settings& techniques,
        const LumenMeshBuildSettings& settings,LumenMeshRepresentation& output,std::string& error,
        const LumenPartitionCancellation& cancelled={});
    // Native capture draws must correspond to compact parts in source order.
    // Returned SDF UserId values address Parts; card ranges never cross parts.
    // This packs one mesh instance for the existing five-API native contracts.
    bool BuildLumenMeshInstance(const LumenMeshRepresentation& mesh,const Matrix4& world,
        const std::vector<LumenSurfaceDraw>& draws,unsigned cardSide,LumenMeshInstanceData& output,std::string& error);
    // Unpacked placement lets a scene share fields across repeated instances
    // before building one BVH. Empty placements do not reserve atlas tiles.
    bool BuildLumenMeshPlacement(const LumenMeshRepresentation& mesh,const Matrix4& world,
        const std::vector<LumenSurfaceDraw>& draws,unsigned cardSide,LumenMeshPlacement& output,std::string& error);
}
