#pragma once

#include "Renderer/RayTracing/SceneRayUpload.h"

namespace Evans::Renderer
{
    inline constexpr std::uint32_t RaySceneFlatNormals=1;

    // Indexed by the visibility BVH's triangle order, not source mesh order.
    // Normals retain their inverse-transpose magnitude until interpolation.
    // Authored vertex RGB is decoded before interpolation, like raster vertices.
    struct alignas(16) RaySceneSurface
    {
        Float4 NormalA {0,0,0,0},NormalB {0,0,0,0},NormalC {0,0,0,0};
        Float4 ColorA {0,0,0,0},ColorB {0,0,0,0},ColorC {0,0,0,0};
    };
    struct alignas(16) RaySceneSurfaceMaterial
    {
        Float4 BaseColor; // Object tint times authored linear Kd, including alpha.
        Float4 AmbientColor {0,0,0,0}; // Authored linear Ka; W is reserved.
        Float4 SpecularShininess; // Nonnegative linear Ks and Ns >= 1.
        Float4 Emission {0,0,0,0}; // Linear emitted radiance; W is dielectric IOR, or zero for opaque/thin surfaces.
        // Float4 texel offset, width, height, presence. Diffuse RGB is linear;
        // normal RGB is UNORM. Both use bilinear repeat with one OBJ-V flip.
        std::array<std::uint32_t,4> DiffuseImage{},NormalImage{};
    };
    struct RaySceneShadingUpload
    {
        std::vector<RaySceneSurface> Surfaces;
        std::vector<RaySceneSurfaceMaterial> Materials;
        std::vector<Float4> Texels;
        std::uint32_t Flags=0;
    };
    static_assert(sizeof(RaySceneSurface)==96 && offsetof(RaySceneSurface,ColorA)==48,"Ray surface ABI changed");
    static_assert(sizeof(RaySceneSurfaceMaterial)==96 && offsetof(RaySceneSurfaceMaterial,DiffuseImage)==64,
        "Ray surface material ABI changed");

    // Visibility and shading are published together by ExportRadiance. Keeping
    // shading optional avoids RGB atlas and corner storage for visibility rays.
    [[nodiscard]] bool ValidateRaySceneShadingUpload(const RaySceneUpload& visibility,
        const RaySceneShadingUpload& shading,std::string& error);
    inline constexpr std::size_t RaySceneShadingBufferCount=3;
    inline std::array<RaySceneBufferData,RaySceneShadingBufferCount> GetRaySceneShadingBufferData(const RaySceneShadingUpload& scene)
    {
        return {RaySceneData(scene.Surfaces),RaySceneData(scene.Materials),RaySceneData(scene.Texels)};
    }
}
