#pragma once

#include "Renderer/Lumen/LumenProbeIntegration.h"

namespace Evans::Renderer
{
    struct LumenProbeGatherSettings
    {
        float Intensity=1,MinimumNormalCosine=0.8f;
        float PlaneTolerance=0.01f,RelativePlaneTolerance=0.05f;
        Techniques::LumenProbeVisibilitySettings Visibility;
    };
    struct alignas(16) LumenProbeGatherFrame
    {
        LumenDistanceTraceFrame Trace; // The exact immutable scene used by the probe rays.
        LumenScreenProbeFrame Screen;
        Float4 Appearance; // Intensity, normal agreement, absolute/relative plane tolerance.
        Float4 Numeric; // Normal-length floor, weight floor, half-float maximum, dielectric F0.
        ShaderUInt4 Model; // Shading model, visibility enabled, indirect-only output; reserved W.
        Float4 Visibility; // World normal bias, coordinate roundoff; reserved ZW.
    };
    static_assert(sizeof(LumenProbeGatherFrame)==272,"Lumen screen gather ABI changed");
    // Wright et al., "Lumen", SIGGRAPH 2022: screen radiance-cache interpolation
    // rejects unrelated receiver planes. L2 irradiance is reconstructed at the
    // receiving normal, then multiplied by diffuse albedo/pi. Geometric weights
    // are normalized; missing ray energy inside each probe is never renormalized.
    // Optional mesh-SDF segment visibility rejects occluded or unresolved probe
    // links before weight normalization. It does not renormalize missing SH energy.
    inline constexpr std::size_t LumenProbeVisibilityMaximumBytes=128u*1024u*1024u;
    inline unsigned LumenProbeVisibilityWords(const LumenScreenProbeFrame& frame) noexcept
    {
        constexpr unsigned NeighborTiles=4,WordBits=32;
        return (NeighborTiles*(frame.Grid.W+1)+WordBits-1)/WordBits;
    }
    inline std::size_t LumenProbeVisibilityBytes(const LumenProbeGatherFrame& frame)
    { return std::size_t(frame.Screen.Viewport.X)*frame.Screen.Viewport.Y*LumenProbeVisibilityWords(frame.Screen)*sizeof(std::uint32_t); }
    bool BuildLumenProbeGatherFrame(const LumenProbeInfo& source,const LumenDistanceSceneInfo& scene,const LumenProbeGatherSettings& settings,
        Techniques::ShadingModel shading,LumenProbeGatherFrame& output,std::string& error);
}
