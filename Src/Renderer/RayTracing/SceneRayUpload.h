#pragma once

#include "Renderer/Types.h"
#include <algorithm>

namespace Evans::Renderer
{
    inline constexpr std::uint32_t RaySceneImagePresent=1;
    inline constexpr std::uint32_t RaySceneShadowCaster=1;
    inline constexpr std::uint32_t RaySceneInvalidIndex=UINT32_MAX;

    // A stackless binary BVH. Internal nodes enter Links.X; rejected nodes and
    // completed leaves continue at Links.W. The root's escape is Nodes.size().
    // A leaf owns Links.Z consecutive triangles starting at Links.Y. Forward
    // links bound traversal by node count without a fixed shader stack limit.
    struct alignas(16) RaySceneNode
    {
        Float4 Minimum,Maximum;
        std::array<std::uint32_t,4> Links{};
    };
    struct alignas(16) RaySceneTriangle
    {
        Float4 A,B,C; // World position XYZ, authored vertex opacity W.
        Float4 UVAB,UVC; // Original OBJ UVs; the sampler flips V once.
        // Material, query triangle, source mesh, cast-shadow flags.
        std::array<std::uint32_t,4> Metadata{};
    };
    struct alignas(16) RaySceneMaterial
    {
        // Scalar float atlas offset, width, height, presence flag. Diffuse alpha
        // and opacity red remain separate: filter first, then multiply. Packing
        // their product would change bilinear coverage at cutout boundaries.
        std::array<std::uint32_t,4> DiffuseAlpha{},OpacityRed{};
        Float4 Coverage; // Object/material alpha, cutoff, reserved, reserved.
    };
    struct RaySceneUpload
    {
        std::vector<RaySceneNode> Nodes;
        std::vector<RaySceneTriangle> Triangles;
        std::vector<RaySceneMaterial> Materials;
        std::vector<float> CoverageTexels;
    };
    static_assert(sizeof(RaySceneNode)==48 && offsetof(RaySceneNode,Links)==32,"Ray node ABI changed");
    static_assert(sizeof(RaySceneTriangle)==96 && offsetof(RaySceneTriangle,Metadata)==80,"Ray triangle ABI changed");
    static_assert(sizeof(RaySceneMaterial)==48 && offsetof(RaySceneMaterial,Coverage)==32,"Ray material ABI changed");

    // Validate before publishing immutable buffers. Backend upload paths must
    // additionally check each byte size against their device's storage limits.
    // Shared triangle/coverage validation also serves native hardware geometry,
    // which replaces the software BVH while keeping identical cutout semantics.
    [[nodiscard]] bool ValidateRaySceneGeometry(const std::vector<RaySceneTriangle>&,
        const std::vector<RaySceneMaterial>&,const std::vector<float>&,std::string& error);
    [[nodiscard]] bool ValidateRaySceneUpload(const RaySceneUpload& scene,std::string& error);

    inline constexpr std::size_t RaySceneBufferCount=4;
    struct RaySceneBufferData
    {
        const void* Data=nullptr;
        std::size_t Bytes=0;
        std::uint32_t Stride=0;
    };
    // Buffer order is part of every native shader ABI. Even empty arrays get a
    // valid zero element; logical counts remain zero in the frame constants.
    template<class T> RaySceneBufferData RaySceneData(const std::vector<T>& values)
    {
        static const T empty{};
        return {values.empty()?&empty:values.data(),
            std::max(values.size(),std::size_t{1})*sizeof(T),sizeof(T)};
    }
    inline std::array<RaySceneBufferData,RaySceneBufferCount> GetRaySceneBufferData(const RaySceneUpload& scene)
    {
        return {RaySceneData(scene.Nodes),RaySceneData(scene.Triangles),
            RaySceneData(scene.Materials),RaySceneData(scene.CoverageTexels)};
    }
}
