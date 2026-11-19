#pragma once

#include "Renderer/Lighting/AmbientOcclusionContract.h"
#include "Renderer/VoxelGI/VoxelRadiance.h"

namespace Evans::Renderer
{
    struct VoxelGIFrame
    {
        std::shared_ptr<const VoxelRadianceVolume> Volume;
        Techniques::VoxelGISettings Settings;
        Techniques::ShadingModel Shading=Techniques::ShadingModel::Phong;
        float Intensity=1,MaximumDistance=10;
    };
    inline constexpr std::array<Techniques::VoxelGIPass,4> VoxelGIPasses{
        Techniques::VoxelGIPass::ConeTrace,Techniques::VoxelGIPass::Horizontal,
        Techniques::VoxelGIPass::Vertical,Techniques::VoxelGIPass::Composite};
    struct VoxelGIConstants
    {
        Matrix4 InverseViewProjection;
        Float4 Viewport;
        Float4 GridOriginCell;
        Float4 Trace; // Maximum distance, tangent of cone half-angle, step scale, normal bias in cells.
        Float4 Filter; // Optical-depth sigma in cells, normal exponent, intensity, reserved.
        Float4 CameraForward;
        ShaderUInt4 Options; // Pass, base resolution, mip count, cosine-quadrature cone count.
        ShaderUInt4 Atlas; // Row width, row count, step budget, shading model.
        std::array<VoxelMipLevel,Techniques::MaximumVoxelMipLevels> Levels;
    };
    static_assert(sizeof(VoxelGIConstants)==288 && offsetof(VoxelGIConstants,Levels)==176,"Voxel GI native ABI changed");

    // Trace and horizontal filtering carry incoming diffuse radiance. Vertical
    // filtering applies the center material and caps additive HDR headroom.
    // Composite binds only that signal and writes RGB with additive blending;
    // its unused scene-color descriptor must not alias the active attachment.
    inline bool BuildVoxelGIConstants(const Matrix4& camera,const Techniques::VoxelGISettings& requested,
        const VoxelRadianceVolume& volume,float intensity,float maximumDistance,Techniques::ShadingModel shading,
        int width,int height,Techniques::VoxelGIPass pass,VoxelGIConstants& output)
    {
        output={};
        const auto resolution=volume.Grid.Resolution;
        if (pass>Techniques::VoxelGIPass::Composite || resolution<Techniques::MinimumVoxelResolution || resolution>Techniques::MaximumVoxelResolution ||
            (resolution&(resolution-1))!=0 || !(volume.Grid.CellSize>0) ||
            !std::isfinite(volume.Grid.CellSize) || !volume.MipCount || volume.MipCount>volume.Levels.size() ||
            (1u<<(volume.MipCount-1))!=resolution ||
            !volume.Width || !volume.Height || volume.Texels.size()!=std::size_t(volume.Width)*volume.Height ||
            !std::isfinite(intensity) || intensity<0) return false;
        const auto controls=Techniques::SanitizeVoxelGISettings(requested);
        for (unsigned axis=0;axis<3;++axis)
        {
            const float origin=axis==0?volume.Grid.Origin.X:axis==1?volume.Grid.Origin.Y:volume.Grid.Origin.Z;
            if (!std::isfinite(origin) || !std::isfinite(origin+volume.Grid.CellSize*resolution)) return false;
        }
        for (std::uint32_t mip=0;mip<volume.MipCount;++mip)
        {
            const auto level=volume.Levels[mip];
            if (level.Dimension!=(volume.Grid.Resolution>>mip) || level.FaceStride!=level.Dimension*level.Dimension*level.Dimension ||
                std::size_t(level.Offset)+std::size_t(level.FaceStride)*Techniques::VoxelDirectionCount>volume.Texels.size()) return false;
        }
        Techniques::AmbientOcclusionSettings projection; projection.Mode=Techniques::AmbientOcclusionMode::GTAO;
        AmbientOcclusionConstants view;
        if (!BuildAmbientOcclusionConstants({camera},projection,width,height,Techniques::AmbientOcclusionPassKind::Estimate,view)) return false;
        constexpr float Pi=3.14159265358979323846f,DegreesPerHalfCircle=180;
        const float distance=controls.AutomaticDistance ? volume.Grid.CellSize*volume.Grid.Resolution*std::sqrt(3.0f) : maximumDistance;
        if (!std::isfinite(distance) || distance<=0) return false;
        output.InverseViewProjection=view.InverseViewProjection; output.CameraForward=view.ForwardIntensity;
        output.Viewport={float(width),float(height),1.0f/width,1.0f/height};
        output.GridOriginCell={volume.Grid.Origin.X,volume.Grid.Origin.Y,volume.Grid.Origin.Z,volume.Grid.CellSize};
        output.Trace={distance,std::tan(controls.ConeApertureDegrees*Pi/(2*DegreesPerHalfCircle)),controls.StepScale,controls.NormalBiasCells};
        constexpr float MaximumIntensity=16;
        output.Filter={controls.DepthSigmaCells,controls.NormalPower,std::min(intensity,MaximumIntensity),0};
        output.Options={static_cast<unsigned>(pass),volume.Grid.Resolution,volume.MipCount,controls.ConeCount};
        output.Atlas={volume.Width,volume.Height,controls.MaximumSteps,static_cast<unsigned>(shading)}; output.Levels=volume.Levels;
        return true;
    }
}
