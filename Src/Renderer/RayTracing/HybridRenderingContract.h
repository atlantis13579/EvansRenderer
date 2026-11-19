#pragma once

#include "Renderer/RayTracing/RayTracedReflectionContract.h"
#include "Techniques/HybridRendering.h"

namespace Evans::Renderer
{
    inline constexpr std::array<Techniques::HybridRenderingPass, 4> HybridRenderingPasses {
        Techniques::HybridRenderingPass::Trace, Techniques::HybridRenderingPass::Horizontal,
        Techniques::HybridRenderingPass::Vertical, Techniques::HybridRenderingPass::Composite};
    inline Techniques::ReflectionPassKind HybridRenderingLogPass(Techniques::HybridRenderingPass pass)
    {
        constexpr std::array<Techniques::ReflectionPassKind, 4> kinds {
            Techniques::ReflectionPassKind::Trace, Techniques::ReflectionPassKind::BilateralHorizontal,
            Techniques::ReflectionPassKind::BilateralVertical, Techniques::ReflectionPassKind::Composite};
        return kinds[static_cast<std::size_t>(pass)];
    }

    // The ray-scene prefix shares the existing material/light storage ABI. Each
    // native HybridRendering source owns screen tracing, BVH fallback, filtering
    // and composition; no shader source is included or translated at runtime.
    struct HybridRenderingConstants
    {
        Matrix4 InverseViewProjection;
        Float4 Viewport, Trace, CameraForward, Ambient, Material;
        ShaderUInt4 SceneCounts, ShadingCounts, Options, Lighting;
        Matrix4 ViewProjection;
        Float4 ScreenTrace; // World thickness, pixel stride, edge fade pixels, reserved.
        ShaderUInt4 HybridOptions; // Screen intersections, scene fallback, maximum steps, reserved.
    };
    static_assert(sizeof(HybridRenderingConstants) == 304 && offsetof(HybridRenderingConstants, ViewProjection) == 208 &&
        offsetof(HybridRenderingConstants, HybridOptions) == 288, "Hybrid rendering uniform ABI changed");

    inline bool BuildHybridRenderingConstants(const ReflectionDesc& camera, const Techniques::Settings& settings,
        Techniques::HybridRenderingSettings hybrid, const RayRadianceFrame& frame, int width, int height,
        Techniques::HybridRenderingPass pass, HybridRenderingConstants& output)
    {
        output = {};
        if (pass > Techniques::HybridRenderingPass::Composite || !frame.Geometry || !frame.Shading || !frame.Lights) return false;
        RayTracedReflectionConstants ray;
        if (!BuildRayTracedReflectionConstants(camera, settings, settings.Reflections.RayTracing,
            *frame.Geometry, *frame.Shading, frame.Lighting, *frame.Lights, width, height,
            static_cast<Techniques::RayReflectionPass>(pass), ray)) return false;
        const auto reflection = Techniques::SanitizeReflectionSettings(settings.Reflections);
        output = {ray.InverseViewProjection, ray.Viewport, ray.Trace, ray.CameraForward, ray.Ambient, ray.Material,
            ray.SceneCounts, ray.ShadingCounts, ray.Options, ray.Lighting, camera.ViewProjection,
            {reflection.Thickness, reflection.StridePixels, reflection.EdgeFadePixels, 0},
            {hybrid.ScreenSpace ? 1u : 0u, hybrid.RayFallback ? 1u : 0u, reflection.MaximumSteps, 0}};
        return true;
    }
}
