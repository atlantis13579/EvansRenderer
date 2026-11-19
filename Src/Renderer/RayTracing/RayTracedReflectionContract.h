#pragma once

#include "Renderer/Lighting/AmbientOcclusionContract.h"
#include "Renderer/RayTracing/RaySceneRadianceFrame.h"
#include "Techniques/RayTracedReflections.h"

#include <limits>

namespace Evans::Renderer
{
    inline constexpr std::array<Techniques::RayReflectionPass,4> RayReflectionPasses {
        Techniques::RayReflectionPass::Trace,Techniques::RayReflectionPass::Horizontal,
        Techniques::RayReflectionPass::Vertical,Techniques::RayReflectionPass::Composite};
    inline Techniques::ReflectionPassKind RayReflectionLogPass(Techniques::RayReflectionPass pass)
    {
        using Pass=Techniques::ReflectionPassKind;
        constexpr std::array<Pass,4> kinds {Pass::Trace,Pass::BilateralHorizontal,Pass::BilateralVertical,Pass::Composite};
        return kinds[static_cast<std::size_t>(pass)];
    }
    // Single-bounce reflection: GGX visible-normal sampling, closest-hit material
    // evaluation and direct lighting with optional visibility rays. The full
    // scene-order light buffer is shared with Forward+, never its legacy subset.
    // Heitz, "Sampling the GGX Distribution of Visible Normals", JCGT 2018.
    // https://jcgt.org/published/0007/04/01/
    struct RayTracedReflectionConstants
    {
        Matrix4 InverseViewProjection;
        Float4 Viewport; // Extent XY, reciprocal extent ZW.
        Float4 Trace; // Maximum ray distance, origin bias, intensity, roughness cutoff.
        Float4 CameraForward; // Optical camera Z, full-scene directional shadow extent.
        Float4 Ambient; // Ambient RGB times strength; W reserved.
        Float4 Material; // Secondary-hit roughness, metallic, toon bands, reserved.
        ShaderUInt4 SceneCounts; // Nodes, triangles, coverage materials, scalar texels.
        ShaderUInt4 ShadingCounts; // Surfaces, surface materials, RGBA texels, lights.
        ShaderUInt4 Options; // Pass, ray count, shading model, surface flags.
        ShaderUInt4 Lighting; // Trace direct visibility, remaining components reserved.
    };
    static_assert(sizeof(RayTracedReflectionConstants)==208 && offsetof(RayTracedReflectionConstants,SceneCounts)==144 &&
        offsetof(RayTracedReflectionConstants,Options)==176,"Ray reflection constant ABI changed");

    inline bool BuildRayTracedReflectionConstants(const ReflectionDesc& camera,const Techniques::Settings& requested,
        Techniques::RayTracedReflectionSettings sampling,const RaySceneUpload& scene,const RaySceneShadingUpload& shading,
        const LightingState& lighting,const SceneLightCollection& lights,int width,int height,
        Techniques::RayReflectionPass pass,RayTracedReflectionConstants& output)
    {
        if (pass>Techniques::RayReflectionPass::Composite || scene.Nodes.size()>UINT32_MAX ||
            scene.Triangles.size()>UINT32_MAX || scene.Materials.size()>UINT32_MAX || scene.CoverageTexels.size()>UINT32_MAX ||
            shading.Surfaces.size()!=scene.Triangles.size() || shading.Materials.size()!=scene.Materials.size() ||
            shading.Texels.size()>UINT32_MAX || lights.GetShaderLights().size()>UINT32_MAX ||
            (shading.Flags&~RaySceneFlatNormals)) return false;
        Techniques::AmbientOcclusionSettings optical; optical.Mode=Techniques::AmbientOcclusionMode::GTAO;
        AmbientOcclusionConstants projection;
        if (!BuildAmbientOcclusionConstants({camera.ViewProjection},optical,width,height,
            Techniques::AmbientOcclusionPassKind::Estimate,projection)) return false;
        const auto reflection=Techniques::SanitizeReflectionSettings(requested.Reflections);
        const auto settings=Techniques::SanitizeSettings(requested);
        sampling=Techniques::SanitizeRayTracedReflectionSettings(sampling);
        RayTracedReflectionConstants candidate;
        candidate.InverseViewProjection=projection.InverseViewProjection;
        candidate.Viewport={float(width),float(height),1.0f/width,1.0f/height};
        candidate.Trace={reflection.MaximumDistance,reflection.NormalBias,reflection.Intensity,reflection.MaximumRoughness};
        double diagonal=0;
        if (!scene.Nodes.empty())
        {
            const auto& root=scene.Nodes.front();
            diagonal=std::hypot(double(root.Maximum.X)-root.Minimum.X,double(root.Maximum.Y)-root.Minimum.Y,
                double(root.Maximum.Z)-root.Minimum.Z);
        }
        const double shadowExtent=diagonal+double(reflection.NormalBias)*2;
        if (!std::isfinite(shadowExtent) || shadowExtent>std::numeric_limits<float>::max()) return false;
        candidate.CameraForward=projection.ForwardIntensity; candidate.CameraForward.W=float(shadowExtent);
        for (const float component:{lighting.AmbientColor.X,lighting.AmbientColor.Y,lighting.AmbientColor.Z,lighting.AmbientStrength})
            if (!std::isfinite(component)) return false;
        const float ambientStrength=std::max(lighting.AmbientStrength,0.0f);
        candidate.Ambient={std::max(lighting.AmbientColor.X,0.0f)*ambientStrength,std::max(lighting.AmbientColor.Y,0.0f)*ambientStrength,
            std::max(lighting.AmbientColor.Z,0.0f)*ambientStrength,0};
        for (const float component:{candidate.Ambient.X,candidate.Ambient.Y,candidate.Ambient.Z})
            if (!std::isfinite(component)) return false;
        candidate.Material={settings.Roughness,settings.Metallic,float(settings.ToonBands),0};
        candidate.SceneCounts={std::uint32_t(scene.Nodes.size()),std::uint32_t(scene.Triangles.size()),
            std::uint32_t(scene.Materials.size()),std::uint32_t(scene.CoverageTexels.size())};
        candidate.ShadingCounts={std::uint32_t(shading.Surfaces.size()),std::uint32_t(shading.Materials.size()),
            std::uint32_t(shading.Texels.size()),std::uint32_t(lights.GetShaderLights().size())};
        candidate.Options={std::uint32_t(pass),sampling.RayCount,std::uint32_t(settings.Shading),shading.Flags};
        candidate.Lighting={sampling.Shadows ? 1u : 0u,0,0,0};
        output=candidate; return true;
    }
}
