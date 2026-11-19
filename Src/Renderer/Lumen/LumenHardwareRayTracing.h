#pragma once

#include "Renderer/Lumen/LumenDistanceScene.h"
#include "Renderer/RayTracing/SceneRayUpload.h"

namespace Evans::Renderer
{
    class SceneRayQuery;
    inline constexpr std::size_t LumenHardwareMaximumWords=128*1024*1024;
    inline constexpr unsigned LumenHardwareInstanceWords=16,LumenHardwareTriangleWords=24,LumenHardwareMaterialWords=12;

    // Local-space triangles and immutable coverage images. The native BLAS
    // primitive ordinal addresses this array directly, independent of source
    // triangle numbering and of the distance field's voxel resolution.
    struct LumenHardwareMesh
    {
        std::vector<RaySceneTriangle> Triangles;
        std::vector<RaySceneMaterial> Materials;
        std::vector<float> CoverageTexels;
        Float3 Minimum,Maximum;
    };
    struct LumenHardwarePlacement
    {
        std::shared_ptr<const LumenHardwareMesh> Mesh;
        Matrix4 World;
        unsigned DistanceInstance=0;
        bool CastShadow=true;
    };
    struct LumenHardwareInstance
    {
        Matrix4 World;
        unsigned Mesh=0;
        bool CastShadow=true;
    };
    struct LumenHardwareGeometry
    {
        unsigned Triangles=0,Materials=0,Coverage=0; // Absolute uint-word addresses.
    };
    struct alignas(16) LumenHardwareGPUInstance
    {
        ShaderUInt4 Storage; // Triangle/material/coverage word addresses; triangle count.
        std::array<Float4,3> Normal; // Positively scaled inverse-transpose rows, W=0.
    };
    static_assert(sizeof(LumenHardwareGPUInstance)==LumenHardwareInstanceWords*sizeof(std::uint32_t) &&
        sizeof(RaySceneTriangle)==LumenHardwareTriangleWords*sizeof(std::uint32_t) &&
        sizeof(RaySceneMaterial)==LumenHardwareMaterialWords*sizeof(std::uint32_t),"Lumen hardware geometry word ABI changed");
    struct LumenHardwareScene
    {
        LumenDistanceSceneIdentity Identity;
        std::vector<std::shared_ptr<const LumenHardwareMesh>> Meshes;
        std::vector<LumenHardwareGeometry> Geometry;
        // Native instance custom IDs must equal this index, which is also the
        // distance-instance/card-ownership index. Repeated instances share BLAS.
        std::vector<LumenHardwareInstance> Instances;
        std::vector<std::uint32_t> Words;
    };

    // Export after the material-aware local query accepts the source geometry.
    // Coverage keeps independently filtered diffuse alpha and opacity red.
    bool BuildLumenHardwareMesh(const SceneRayQuery&,LumenHardwareMesh&,std::string& error);
    // These snapshots describe native build/query inputs; constructing one does
    // not execute a ray query. Backends own BLAS/TLAS and capability fallback.
    // Placements may arrive out of order, but their distance-instance IDs must
    // cover [0,N) exactly. CPU/native resources retain one immutable snapshot;
    // changing a mesh or pose produces a new generation, never an in-place edit.
    // All builders preserve output on failure.
    bool BuildLumenHardwareScene(const std::vector<LumenHardwarePlacement>&,LumenDistanceSceneIdentity,
        LumenHardwareScene&,std::string& error);
    bool ValidateLumenHardwareScene(const LumenHardwareScene&,std::string& error);
}
