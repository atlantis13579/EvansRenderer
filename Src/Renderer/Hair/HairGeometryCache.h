#pragma once

#include "Renderer/Hair/HairStrands.h"

namespace Evans::Renderer
{
    struct HairSourceGeometry
    {
        std::vector<Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
    };
    struct HairSourceInstance
    {
        std::shared_ptr<const HairSourceGeometry> Geometry;
        Matrix4 World;
        Float4 Color{1,1,1,1};
        MaterialData Material;
        bool Visible=true,CastShadow=true;
    };
    struct HairGeometryRequest
    {
        std::vector<HairSourceInstance> Instances;
        std::uint64_t SceneRevision=0,SelectionGroup=0;
        Techniques::Settings Materials;
        Techniques::HairFurSettings Groom;
    };
    enum class HairUpdateState : std::uint8_t { Idle,Queued,Preparing,Growing,Ready,Failed };
    struct HairGeometryResult
    {
        std::uint64_t Generation=0;
        std::shared_ptr<const HairStrandGeometry> Geometry;
    };
    struct HairGeometryStatus
    {
        HairUpdateState State=HairUpdateState::Idle;
        std::uint64_t RequestedGeneration=0,QueryBuildCount=0,GroomBuildCount=0;
        std::shared_ptr<const HairGeometryResult> Published;
        std::string Error;
        bool Current() const noexcept
        { return State==HairUpdateState::Ready && Published && Published->Generation==RequestedGeneration; }
        bool Pending() const noexcept
        { return State==HairUpdateState::Queued || State==HairUpdateState::Preparing || State==HairUpdateState::Growing; }
    };
    // Camera, lights and optical controls do not rebuild roots. One owned worker
    // coalesces geometry edits, reuses the immutable material query and publishes
    // only the current generation. Native resources stay on the render thread.
    class HairGeometryCache final
    {
    public:
        HairGeometryCache();
        ~HairGeometryCache();
        HairGeometryCache(const HairGeometryCache&)=delete;
        HairGeometryCache& operator=(const HairGeometryCache&)=delete;
        std::uint64_t Request(HairGeometryRequest request);
        HairGeometryStatus Status() const;
        void Cancel();
        void Reset();
    private:
        struct Impl;
        std::unique_ptr<Impl> State;
    };
}
