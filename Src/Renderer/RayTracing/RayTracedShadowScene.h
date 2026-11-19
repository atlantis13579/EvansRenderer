#pragma once

#include "Renderer/RayTracing/SceneRayUpload.h"
#include "Techniques/RayTracedShadows.h"

#include <array>

namespace Evans::Renderer
{
    // A sampled RGBA32UI image exposes the existing immutable, stackless BVH to
    // material and volume shaders. This also works in GLES 3.0 fragment stages
    // without competing with light-list storage blocks. Integer texel fetches
    // preserve positions, escape links and independently filtered cutout planes.
    // Each API owns its texture upload and independently authored query shader.
    using RayShadowTexel=std::array<std::uint32_t,4>;
    inline constexpr unsigned RayShadowHeaderTexels=4,RayShadowNodeTexels=3;
    inline constexpr unsigned RayShadowTriangleTexels=6,RayShadowMaterialTexels=3;
    inline constexpr unsigned RayShadowScalarLanes=4,RayShadowImageVersion=1;
    inline constexpr unsigned RayShadowMaximumTextureSide=16384;
    inline constexpr std::size_t RayShadowMaximumImageBytes=256*1024*1024;
    static_assert(sizeof(RayShadowTexel)==16 && sizeof(RaySceneNode)==RayShadowNodeTexels*sizeof(RayShadowTexel) &&
        sizeof(RaySceneTriangle)==RayShadowTriangleTexels*sizeof(RayShadowTexel) &&
        sizeof(RaySceneMaterial)==RayShadowMaterialTexels*sizeof(RayShadowTexel),"Ray-shadow image ABI changed");

    struct RayShadowFrame
    {
        bool Enabled=false;
        LightType PrimaryLight=LightType::Directional;
        Techniques::RayTracedShadowSettings Settings;
    };
    struct RayShadowImage
    {
        unsigned Width=0,Height=0;
        std::vector<RayShadowTexel> Texels;
    };
    // Header: enabled/node/triangle/material counts; four payload offsets;
    // normal bias/start bias/horizon/reserved float bits; light type/version/
    // scalar coverage count/reserved. Origins and light endpoints remain world
    // space; only the selected primary light uses this shadow representation.
    bool BuildRayShadowHeader(const RaySceneUpload&,const RayShadowFrame&,
        std::array<RayShadowTexel,RayShadowHeaderTexels>&,std::string& error);
    // Failure preserves the previous image. Padding and empty scenes contain
    // valid zero texels, so disabled bindings never need an invalid texture.
    bool BuildRayShadowImage(const RaySceneUpload&,const RayShadowFrame&,unsigned maximumTextureSide,
        RayShadowImage& output,std::string& error);
}
