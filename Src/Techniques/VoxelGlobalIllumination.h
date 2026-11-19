#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace Evans::Techniques
{
    inline constexpr std::uint32_t MinimumVoxelResolution=8,MaximumVoxelResolution=64;
    inline constexpr std::uint32_t MaximumVoxelMipLevels=7,VoxelDirectionCount=6;
    struct VoxelGISettings
    {
        bool AutomaticBounds=true,AutomaticDistance=true,Shadows=true;
        std::array<float,3> BoundsMinimum{-1,-1,-1},BoundsMaximum{1,1,1};
        std::uint32_t Resolution=32,ConeCount=6,MaximumSteps=64;
        float ConeApertureDegrees=60,StepScale=0.5f,NormalBiasCells=1.5f;
        float DepthSigmaCells=1,NormalPower=8;
    };
    inline VoxelGISettings SanitizeVoxelGISettings(VoxelGISettings value) noexcept
    {
        const VoxelGISettings defaults;
        const auto finite=[](float x,float fallback){return std::isfinite(x)?x:fallback;};
        value.Resolution=std::clamp(value.Resolution,MinimumVoxelResolution,MaximumVoxelResolution);
        std::uint32_t lower=MinimumVoxelResolution;
        while (lower*2<=value.Resolution) lower*=2;
        value.Resolution=value.Resolution-lower<(lower*2-value.Resolution) ? lower : std::min(lower*2,MaximumVoxelResolution);
        value.ConeCount=std::clamp(value.ConeCount,1u,16u); value.MaximumSteps=std::clamp(value.MaximumSteps,1u,256u);
        value.ConeApertureDegrees=std::clamp(finite(value.ConeApertureDegrees,defaults.ConeApertureDegrees),1.0f,120.0f);
        value.StepScale=std::clamp(finite(value.StepScale,defaults.StepScale),0.125f,1.0f);
        value.NormalBiasCells=std::clamp(finite(value.NormalBiasCells,defaults.NormalBiasCells),0.0f,8.0f);
        value.DepthSigmaCells=std::clamp(finite(value.DepthSigmaCells,defaults.DepthSigmaCells),0.01f,16.0f);
        value.NormalPower=std::clamp(finite(value.NormalPower,defaults.NormalPower),0.0f,64.0f);
        for (std::size_t axis=0;axis<value.BoundsMinimum.size();++axis)
        {
            value.BoundsMinimum[axis]=finite(value.BoundsMinimum[axis],defaults.BoundsMinimum[axis]);
            value.BoundsMaximum[axis]=finite(value.BoundsMaximum[axis],defaults.BoundsMaximum[axis]);
            if (!(value.BoundsMaximum[axis]>value.BoundsMinimum[axis]))
            { value.BoundsMinimum[axis]=defaults.BoundsMinimum[axis]; value.BoundsMaximum[axis]=defaults.BoundsMaximum[axis]; }
        }
        return value;
    }
    inline bool operator==(const VoxelGISettings& a,const VoxelGISettings& b) noexcept
    {
        return a.AutomaticBounds==b.AutomaticBounds && a.AutomaticDistance==b.AutomaticDistance && a.Shadows==b.Shadows &&
            a.BoundsMinimum==b.BoundsMinimum && a.BoundsMaximum==b.BoundsMaximum && a.Resolution==b.Resolution &&
            a.ConeCount==b.ConeCount && a.MaximumSteps==b.MaximumSteps && a.ConeApertureDegrees==b.ConeApertureDegrees &&
            a.StepScale==b.StepScale && a.NormalBiasCells==b.NormalBiasCells && a.DepthSigmaCells==b.DepthSigmaCells && a.NormalPower==b.NormalPower;
    }
    enum class VoxelGIPass : std::uint32_t { ConeTrace,Horizontal,Vertical,Composite };
}
