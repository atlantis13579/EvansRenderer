#pragma once

#include "Renderer/Lumen/LumenProbeSampling.h"
#include "Renderer/Lumen/LumenEnvironment.h"

namespace Evans::Renderer
{
    struct LumenReflectionRays
    {
        bool Enabled=false;
        unsigned ResolutionDivisor=2;
        float MaximumRoughness=0.8f;
    };
    struct alignas(16) LumenReflectionRayFrame
    {
        Matrix4 Inverse;
        ShaderUInt4 Viewport; // Width, height, effective resolution divisor, trace columns.
        ShaderUInt4 Work; // Trace rows, ray count, sampling sequence, dispatch row stride.
        Float4 Trace; // Maximum distance, normal bias, maximum roughness, minimum GGX alpha.
        Float4 Numeric; // Coordinate limit, homogeneous floor, squared normal floor, reserved.
    };
    static_assert(sizeof(LumenReflectionRayFrame)==128,"Lumen reflection ray ABI changed");
    inline constexpr unsigned LumenReflectionMaximumRays=LumenDistanceMaximumRays/2;
    inline constexpr unsigned LumenGeneratedReflectionRays=1,LumenReflectionScreenFirst=2;

    // Heitz, "Sampling the GGX Distribution of Visible Normals", JCGT 2018.
    // https://jcgt.org/published/0007/04/01/
    // One visible-normal sample per trace pixel yields a reflected world ray.
    // Native generation reads the current G-buffer; no CPU rays or new assets.
    // Zero-length rays encode background/back-facing/roughness-rejected pixels.
    bool BuildLumenReflectionRayFrame(const LumenScreenProbePlacement& screen,const LumenProbeRaySet& sampling,
        const LumenReflectionRays& settings,LumenReflectionRayFrame& output,std::string& error);

    struct LumenReflectionFilter
    {
        unsigned Radius=1; // Trace-grid cells, bounded to a five-by-five neighborhood.
        float NormalCosine=0.9f,PlaneTolerance=0.02f,RelativePlaneTolerance=0.01f,RoughnessTolerance=0.1f;
    };
    struct alignas(16) LumenReflectionResolveFrame
    {
        LumenReflectionRayFrame Rays;
        Float4 Acceptance; // Normal cosine, absolute/relative plane tolerance, roughness tolerance.
        Float4 Filter; // Neighborhood radius, cosine floor, binary16 maximum, dielectric F0.
        Float4 Sky; // Explicit miss radiance; reserved W.
        ShaderUInt4 Dispatch; // Shading model, dispatch row stride, output pixel count, reserved.
        LumenEnvironmentFrame Environment;
    };
    static_assert(sizeof(LumenReflectionResolveFrame)==224,"Lumen reflection reconstruction ABI changed");
    // Reconstruct the VNDF estimate F * G2 / G1(view), not unweighted hit color.
    // Unknown cache transport and below-horizon samples remain in the denominator.
    // Alpha distinguishes complete (1), partial (0.5), and absent (0) estimates;
    // partial energy is displayed but cannot recover missing energy from history.
    bool BuildLumenReflectionResolveFrame(const LumenReflectionRayFrame& rays,const LumenReflectionFilter& filter,
        Float3 sky,Techniques::ShadingModel shading,LumenReflectionResolveFrame& output,std::string& error,
        const LumenEnvironmentLobe& environment={});
}
