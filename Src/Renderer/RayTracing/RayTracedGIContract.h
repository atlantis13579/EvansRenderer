#pragma once

#include "Renderer/Lighting/AmbientOcclusionContract.h"
#include "Renderer/RayTracing/RaySceneRadianceFrame.h"
#include "Techniques/RayTracedGI.h"

#include <limits>

namespace Evans::Renderer
{
    inline constexpr std::array<Techniques::RayGIPass,4> RayGIPasses {
        Techniques::RayGIPass::Trace,Techniques::RayGIPass::Horizontal,Techniques::RayGIPass::Vertical,Techniques::RayGIPass::Composite};
    // Bounded diffuse paths: each hit adds emission and explicit direct lighting,
    // then continues with a cosine-weighted Lambertian sample. Its BRDF*cos/pdf
    // cancels to diffuse reflectance. PBR diffuse paths use (1-metallic)*(1-F0);
    // explicit hit lighting retains the selected Phong/Toon/GGX closure.
    // Existing ambient light is not transported recursively. Misses are black.
    // Two spatial bilateral passes preserve depth, normal and albedo boundaries.
    // PBRT 4e, Diffuse Reflection and A Simple Path Tracer:
    // https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection
    // https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Simple_Path_Tracer
    //
    // Six screen images: depth, normal/roughness, diffuse/opacity, ambient/metallic,
    // previous estimate and opaque HDR. Scene storage is shared with reflections;
    // estimate/filter targets and pass constants belong exclusively to diffuse GI.
    struct RayTracedGIConstants
    {
        Matrix4 InverseViewProjection;
        Float4 Viewport;
        Float4 Trace; // Maximum segment distance, surface bias, intensity, reserved.
        Float4 CameraForward; // Optical axis XYZ, directional visibility extent W.
        Float4 Reserved;
        Float4 Material; // Secondary roughness, metallic, toon bands, reserved.
        ShaderUInt4 SceneCounts;
        ShaderUInt4 ShadingCounts;
        ShaderUInt4 Options; // Phase, path count, shading model, surface flags.
        ShaderUInt4 Lighting; // Direct visibility enabled, diffuse bounces, reserved.
    };
    static_assert(sizeof(RayTracedGIConstants)==208 && offsetof(RayTracedGIConstants,SceneCounts)==144 &&
        offsetof(RayTracedGIConstants,Lighting)==192,"Ray GI uniform ABI changed");

    inline bool BuildRayTracedGIConstants(const GlobalIlluminationDesc& camera,const Techniques::Settings& requested,
        Techniques::RayTracedGISettings sampling,const RayRadianceFrame& frame,int width,int height,
        Techniques::RayGIPass pass,RayTracedGIConstants& output)
    {
        if (!frame.Geometry || !frame.Shading || !frame.Lights || pass>Techniques::RayGIPass::Composite) return false;
        const auto& scene=*frame.Geometry; const auto& shading=*frame.Shading;
        if (scene.Nodes.size()>UINT32_MAX || scene.Triangles.size()>UINT32_MAX || scene.Materials.size()>UINT32_MAX ||
            scene.CoverageTexels.size()>UINT32_MAX || shading.Surfaces.size()!=scene.Triangles.size() ||
            shading.Materials.size()!=scene.Materials.size() || shading.Texels.size()>UINT32_MAX ||
            frame.Lights->GetShaderLights().size()>UINT32_MAX || (shading.Flags&~RaySceneFlatNormals)) return false;
        Techniques::AmbientOcclusionSettings optical; optical.Mode=Techniques::AmbientOcclusionMode::GTAO;
        AmbientOcclusionConstants projection;
        if (!BuildAmbientOcclusionConstants({camera.ViewProjection},optical,width,height,
            Techniques::AmbientOcclusionPassKind::Estimate,projection)) return false;
        const auto settings=Techniques::SanitizeSettings(requested);
        auto gi=Techniques::SanitizeGlobalIlluminationSettings(requested.GlobalIllumination);
        // Screen-space slab thickness is inactive for geometric scene rays.
        gi.NormalBias=std::min(gi.NormalBias,gi.MaximumDistance);
        sampling=Techniques::SanitizeRayTracedGISettings(sampling);
        double diagonal=0;
        if (!scene.Nodes.empty())
        {
            const auto& root=scene.Nodes.front();
            diagonal=std::hypot(double(root.Maximum.X)-root.Minimum.X,double(root.Maximum.Y)-root.Minimum.Y,
                double(root.Maximum.Z)-root.Minimum.Z);
        }
        const double extent=diagonal+double(gi.NormalBias)*2;
        if (!std::isfinite(extent) || extent>std::numeric_limits<float>::max()) return false;
        RayTracedGIConstants candidate{};
        candidate.InverseViewProjection=projection.InverseViewProjection;
        candidate.Viewport={float(width),float(height),1.0f/width,1.0f/height};
        candidate.Trace={gi.MaximumDistance,gi.NormalBias,gi.Intensity,0};
        candidate.CameraForward=projection.ForwardIntensity; candidate.CameraForward.W=float(extent);
        candidate.Material={settings.Roughness,settings.Metallic,float(settings.ToonBands),0};
        candidate.SceneCounts={std::uint32_t(scene.Nodes.size()),std::uint32_t(scene.Triangles.size()),
            std::uint32_t(scene.Materials.size()),std::uint32_t(scene.CoverageTexels.size())};
        candidate.ShadingCounts={std::uint32_t(shading.Surfaces.size()),std::uint32_t(shading.Materials.size()),
            std::uint32_t(shading.Texels.size()),std::uint32_t(frame.Lights->GetShaderLights().size())};
        candidate.Options={std::uint32_t(pass),sampling.RayCount,std::uint32_t(settings.Shading),shading.Flags};
        candidate.Lighting={sampling.Shadows ? 1u : 0u,sampling.BounceCount,0,0};
        output=candidate; return true;
    }
}
