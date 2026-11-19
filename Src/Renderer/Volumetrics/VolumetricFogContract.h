#pragma once

#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/FrameHistory.h"
#include "Maths/Matrix4.h"

namespace Evans::Renderer
{
    // Frustum slices are packed horizontally in an RGBA16F 2D atlas. Each texel
    // stores cumulative in-scattering RGB and scalar transmittance A, starting at the
    // camera near plane. Endpoint zero is implicit identity transport (0,0,0,1).
    // Participating media uses two native row planes: RGB in-scattering followed
    // by RGB transmittance. Both share logical grid coordinates and depth slices.
    // Ambient height fog integrates each prefix directly. Direct lighting and
    // bounded media use fragment passes for local transport and its prefix scan.
    struct VolumetricFogConstants
    {
        Matrix4 InverseViewProjection;
        Float4 Medium; // extinction, base height, height falloff, maximum distance
        Float4 SourceRadiance;
        Float4 ViewportGrid; // viewport XY, logical grid XY
        ShaderUInt4 Options; // integrate/composite, slices, samples per slice, mode (disabled/scalar/RGB)
        Float4 WaterBounds; // surface height, bottom height, radius, extinction (zero outside water)
        Float4 WaterSun, WaterIrradiance, WaterWaveA, WaterWaveB;
    };
    // The froxel injection shaders consume only the original grid prefix.
    inline constexpr std::size_t FogGridConstantBytes=offsetof(VolumetricFogConstants,WaterBounds);
    static_assert(FogGridConstantBytes==128 && sizeof(VolumetricFogConstants)==208,"Fog uniform layout changed");
    static_assert(offsetof(VolumetricFogConstants, Options) == 112, "Fog options offset changed");

    struct FogGrid
    {
        int Width = 0, Height = 0, Slices = 0, Planes = 1;
        int AtlasWidth() const noexcept { return Width * Slices; }
        int AtlasHeight() const noexcept { return Height * Planes; }
    };
    inline FogGrid BuildFogGrid(int width, int height, std::uint32_t slices, int maximumTextureSize, bool rgbTransport = false,
        int boundedTilePixels=4)
    {
        if (width <= 0 || height <= 0 || slices == 0 || slices > 64 || maximumTextureSize < static_cast<int>(slices)) return {};
        const int planes = rgbTransport ? 2 : 1;
        if (maximumTextureSize < planes) return {};
        // Authored grids need finer screen sampling to retain bounded cloud silhouettes.
        const int tile = rgbTransport ? std::max(1,boundedTilePixels) : Techniques::FogTilePixels;
        return {std::min(1 + (width - 1) / tile, maximumTextureSize / static_cast<int>(slices)),
            std::min(1 + (height - 1) / tile, maximumTextureSize / planes), static_cast<int>(slices), planes};
    }

    inline FogGrid BuildFogGrid(int width,int height,const Techniques::VolumetricFogSettings& settings,int maximumTextureSize)
    {
        auto grid=BuildFogGrid(width,height,settings.SliceCount,maximumTextureSize,Techniques::UsesBoundedMedium(settings),
            static_cast<int>(settings.BoundedTilePixels));
        // Water integrates to scene depth in the composite pass. An empty
        // atmosphere needs just one identity texel per depth slice and plane.
        if (grid.Width>0 && settings.MediumMode==Techniques::VolumeMediumMode::Clouds && settings.Underwater.Extinction>0 &&
            settings.Clouds.Medium.Density==0 && (settings.AtmosphereMode==0 || settings.Extinction==0))
            grid.Width=grid.Height=1;
        return grid;
    }

    inline bool BuildVolumetricFogConstants(const VolumetricFogDesc& desc,
        const Techniques::VolumetricFogSettings& requested, int width, int height,
        int maximumTextureSize, VolumetricFogConstants& output)
    {
        const auto settings = Techniques::SanitizeVolumetricFogSettings(requested);
        const auto grid = BuildFogGrid(width,height,settings,maximumTextureSize);
        if (grid.Width == 0 || !FrameHistory::IsFinite(desc.ViewProjection)) return false;
        Maths::Matrix4 inverse;
        if (!Maths::Matrix4(desc.ViewProjection.Values).Inverse(inverse)) return false;
        output = {};
        std::copy(inverse.Data(), inverse.Data() + 16, output.InverseViewProjection.Values.begin());
        if (!FrameHistory::IsFinite(output.InverseViewProjection)) return false;
        // Finite perspective or affine frusta only: a finite ray endpoint is
        // necessary for distance clamping. Reject projective poles in the box.
        constexpr double MinimumHomogeneousW = 1.0e-8;
        const auto& m = output.InverseViewProjection.Values;
        double sign = 0;
        for (int z = 0; z < 2; ++z)
            for (int y : {-1, 1}) for (int x : {-1, 1})
            {
                const double w = double(m[12]) * x + double(m[13]) * y + double(m[14]) * z + m[15];
                if (!std::isfinite(w) || std::abs(w) < MinimumHomogeneousW || (sign != 0 && sign * w < 0)) return false;
                sign = w;
            }
        output.Medium = {settings.Extinction, settings.BaseHeight, settings.HeightFalloff, settings.MaximumDistance};
        output.SourceRadiance = {settings.Radiance[0], settings.Radiance[1], settings.Radiance[2], static_cast<float>(settings.AtmosphereMode)};
        output.ViewportGrid = {static_cast<float>(width), static_cast<float>(height),
            static_cast<float>(grid.Width), static_cast<float>(grid.Height)};
        // Zero disables surface lookup. One uses scalar height-fog transport;
        // two stores cumulative S.rgb and T.rgb in the first/second native atlas planes.
        output.Options = {0, settings.SliceCount, settings.SamplesPerSlice,
            Techniques::UsesBoundedMedium(settings) ? 2u : Techniques::UsesVolumetricFog(settings) ? 1u : 0u};
        const auto& w=settings.Underwater;
        output.WaterBounds={w.SurfaceHeight,w.BottomHeight,w.Radius,w.Extinction};
        output.WaterSun={w.ToSun[0],w.ToSun[1],w.ToSun[2],0};
        output.WaterIrradiance={w.Irradiance[0],w.Irradiance[1],w.Irradiance[2],0};
        output.WaterWaveA={w.WaveA[0],w.WaveA[1],w.WaveA[2],w.WaveA[3]};
        output.WaterWaveB={w.WaveB[0],w.WaveB[1],w.WaveB[2],w.WaveB[3]};
        return true;
    }
}
