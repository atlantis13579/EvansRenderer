#pragma once

#include "Renderer/VoxelGI/VoxelRadiance.h"

namespace Evans::Renderer
{
    struct VoxelSourceGeometry
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
    };
    struct VoxelSourceInstance
    {
        std::shared_ptr<const VoxelSourceGeometry> Geometry;
        Matrix4 World;
        Float4 Color={1,1,1,1};
        MaterialData Material;
        bool Visible=true,CastShadow=true;
    };
    struct VoxelRadianceRequest
    {
        // Immutable shared topology avoids a vertex copy for every instance or
        // every lighting edit. Advance SceneRevision for all source changes.
        std::vector<VoxelSourceInstance> Instances;
        std::uint64_t SceneRevision=0;
        Techniques::Settings Materials;
        Techniques::VoxelGISettings Grid;
        LightingState Lighting;
        bool Shadows=true;
    };
    struct VoxelRadianceResult
    {
        std::uint64_t Generation=0;
        std::shared_ptr<const VoxelizedScene> Geometry;
        std::shared_ptr<const VoxelRadianceVolume> Radiance;
    };
    enum class VoxelUpdateState : std::uint8_t { Idle,Queued,Preparing,Voxelizing,Injecting,Ready,Failed };
    struct VoxelRadianceStatus
    {
        VoxelUpdateState State=VoxelUpdateState::Idle;
        std::uint64_t RequestedGeneration=0,QueryBuildCount=0,VoxelBuildCount=0,InjectionCount=0;
        std::shared_ptr<const VoxelRadianceResult> Published;
        std::string Error;
        bool Current() const noexcept
        { return State==VoxelUpdateState::Ready && Published && Published->Generation==RequestedGeneration; }
    };
    // A single owned worker and a replaceable pending slot. Cancellation is
    // checked while clipping/injecting and before each publication. Native GPU
    // resources and shader compilation stay on the render thread. Destruction
    // cancels and joins; no detached work retains the renderer.
    class VoxelRadianceCache final
    {
    public:
        VoxelRadianceCache();
        ~VoxelRadianceCache();
        VoxelRadianceCache(const VoxelRadianceCache&)=delete;
        VoxelRadianceCache& operator=(const VoxelRadianceCache&)=delete;
        // Display/camera/gather controls do not change the request identity.
        // Failed equivalent requests are latched until an input changes.
        std::uint64_t Request(VoxelRadianceRequest request);
        VoxelRadianceStatus Status() const;
        void Cancel(); // Keep reusable query, geometry and last completed volume.
        void Reset();  // Also release cached producer data and reset counters.
    private:
        struct Impl;
        std::unique_ptr<Impl> State;
    };
}
