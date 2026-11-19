#pragma once

#include "Renderer/RayTracing/SceneRayQuery.h"
#include "Techniques/VoxelGlobalIllumination.h"

#include <functional>

namespace Evans::Renderer
{
    inline constexpr std::uint32_t VoxelAtlasWidth=1024;
    struct VoxelGrid
    {
        Float3 Origin;
        float CellSize=1;
        std::uint32_t Resolution=0;
        std::uint32_t CellCount() const noexcept { return Resolution*Resolution*Resolution; }
    };
    struct VoxelSurfaceSample
    {
        std::uint32_t FaceCell=0;
        SceneRayHit Surface;
        float FrontFraction=0;
    };
    struct VoxelizedScene
    {
        VoxelGrid Grid;
        std::shared_ptr<const SceneRayQuery> Query;
        // +X,-X,+Y,-Y,+Z,-Z face-major directional coverage. Only front-facing
        // material representatives need lighting samples; backfaces still occlude.
        std::vector<float> Opacity;
        std::vector<VoxelSurfaceSample> Samples;
        std::uint64_t TriangleCellTests=0,ClippedTriangles=0,AcceptedSamples=0;
    };
    struct VoxelMipLevel { std::uint32_t Dimension=0,Offset=0,FaceStride=0,Reserved=0; };
    static_assert(sizeof(VoxelMipLevel)==16,"Voxel mip metadata ABI changed");
    struct VoxelRadianceVolume
    {
        VoxelGrid Grid;
        std::array<VoxelMipLevel,Techniques::MaximumVoxelMipLevels> Levels{};
        std::uint32_t MipCount=0,Width=VoxelAtlasWidth,Height=0;
        std::uint64_t ShadowQueries=0;
        // RGBA32F upload. RGB is radiance premultiplied by directional coverage.
        // Each mip stores six dense cubic faces; integer addressing avoids 2D
        // slice bleed. Native APIs implement their own 3D/LOD reconstruction.
        std::vector<Float4> Texels;
    };
    using VoxelCancellation=std::function<bool()>;

    // Clip existing triangles to cubic cells, sample existing material/cutout
    // rules, then aggregate one material representative per outgoing cell face.
    // Geometry/material edits rebuild this immutable producer; camera motion does
    // not. The query owns the geometry used by subsequent shadow reinjection.
    bool VoxelizeScene(std::shared_ptr<const SceneRayQuery> scene,const Techniques::VoxelGISettings& settings,
        VoxelizedScene& output,std::string& error,const VoxelCancellation& cancelled={});
    // Light edits reinject direct diffuse + emission without revoxelization.
    // Six directional mips composite front/back children before transverse
    // averaging. Isotropic averaging would turn a complete thin wall translucent.
    // Crassin et al., "Interactive Indirect Illumination Using Voxel Cone Tracing" (2011):
    // https://research.nvidia.com/labs/rtr/publication/crassin2011givoxels/
    bool InjectVoxelRadiance(const VoxelizedScene& scene,const LightingState& lighting,bool shadows,
        VoxelRadianceVolume& output,std::string& error,const VoxelCancellation& cancelled={});
}
