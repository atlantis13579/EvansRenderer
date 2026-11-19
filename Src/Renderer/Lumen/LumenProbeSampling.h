#pragma once

#include "Renderer/Lumen/LumenDistanceScene.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenMaximumProbes=4096,LumenMinimumProbeRays=16,LumenMaximumProbeRays=1024,LumenAdaptiveProbeChildren=4;
    inline constexpr unsigned LumenMaximumWorldDepthLayers=Techniques::MaximumLumenWorldDepthLayers;
    inline unsigned LumenScreenProbeSlots(unsigned levels) noexcept
    {
        unsigned slots=1,children=1;
        for (unsigned level=0;level<levels;++level) { children*=LumenAdaptiveProbeChildren; slots+=children; }
        return slots;
    }
    inline bool ValidLumenScreenProbeLayout(const ShaderUInt4& grid) noexcept
    { return grid.Z<=Techniques::MaximumLumenAdaptiveProbeLevels && grid.W==LumenScreenProbeSlots(grid.Z)-1; }
    struct alignas(16) LumenProbeOrigin
    {
        Float4 PositionMinimum; // World position and minimum trace distance.
        ShaderUInt4 Metadata; // ID; screen-source validity (zero is valid); selected pixel XY. CPU origins reserve YZW.
    };
    struct LumenProbeRaySet
    {
        std::vector<LumenProbeOrigin> Origins;
        unsigned RaysPerProbe=128,Sequence=0;
        // Zero selects rotated Hammersley sampling. A nonzero power-of-two side
        // selects a fixed equal-area square atlas for directional world caches.
        unsigned DirectionalSide=0,DepthLayers=1;
        // Directional records are probe-major, then depth-major, then direction.
        // Keeping a probe contiguous makes scrolling copies include every layer.
        float MaximumDistance=1000;
    };
    struct LumenScreenProbePlacement
    {
        bool Enabled=false,Adaptive=false;
        float MinimumNormalCosine=0.8f,PlaneTolerance=0.01f,RelativePlaneTolerance=0.05f;
        Matrix4 ViewProjection;
        unsigned Width=0,Height=0,TileSide=16,AdaptiveLevels=1;
        float NormalBias=0.01f,MinimumDistance=0;
    };
    struct alignas(16) LumenScreenProbeFrame
    {
        Matrix4 InverseViewProjection;
        ShaderUInt4 Viewport{}; // Width, height, effective tile side, grid width.
        ShaderUInt4 Grid{}; // Grid height, total reserved probes, quadtree depth, additional slots per tile.
        Float4 Parameters{}; // Normal bias, minimum distance, coordinate limit, minimum homogeneous W.
        Float4 Adaptive{}; // Minimum normal cosine, absolute/relative plane tolerance, coordinate roundoff.
    };
    static_assert(sizeof(LumenScreenProbeFrame)==128,"Lumen screen placement ABI changed");
    bool ValidateLumenProbeSampling(unsigned count,unsigned rays,float horizon,std::string& error,unsigned layers=1);
    bool BuildLumenScreenProbeFrame(const LumenScreenProbePlacement& placement,const LumenProbeRaySet& sampling,
        LumenScreenProbeFrame& output,std::string& error);
    struct alignas(16) LumenProbeTraceFrame
    {
        LumenDistanceTraceFrame Trace;
        ShaderUInt4 Sampling; // Rays per probe, probe count, sequence, availability bits: screen=1, world cache=2.
        Float4 Range; // Maximum world distance, atlas side (zero is Hammersley), depth layers, coordinate roundoff.
    };
    static_assert(sizeof(LumenProbeOrigin)==32 && sizeof(LumenProbeTraceFrame)==112,"Lumen probe sampling ABI changed");
    bool ValidateLumenProbeRaySet(const LumenProbeRaySet& probes,std::string& error);
}
