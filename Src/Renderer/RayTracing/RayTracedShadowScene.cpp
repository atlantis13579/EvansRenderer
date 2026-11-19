#include "Renderer/RayTracing/RayTracedShadowScene.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        std::uint32_t FloatWord(float value)
        { std::uint32_t word=0; std::memcpy(&word,&value,sizeof(word)); return word; }
        bool CountsFit(const RaySceneUpload& scene)
        {
            return scene.Nodes.size()<=UINT32_MAX && scene.Triangles.size()<=UINT32_MAX &&
                scene.Materials.size()<=UINT32_MAX && scene.CoverageTexels.size()<=UINT32_MAX;
        }
    }
    bool BuildRayShadowHeader(const RaySceneUpload& scene,const RayShadowFrame& frame,
        std::array<RayShadowTexel,RayShadowHeaderTexels>& output,std::string& error)
    {
        const auto& settings=frame.Settings;
        const auto bias=[](float x) { return std::isfinite(x) && x>=0 && x<=Techniques::MaximumRayShadowBias; };
        if (!CountsFit(scene) || frame.PrimaryLight>LightType::RectArea ||
            !std::isfinite(settings.MaximumDistance) || settings.MaximumDistance<Techniques::MinimumRayShadowDistance ||
            settings.MaximumDistance>Techniques::MaximumRayShadowDistance || !bias(settings.NormalBias) || !bias(settings.StartBias) ||
            double(settings.NormalBias)+settings.StartBias>=settings.MaximumDistance)
        { error="Ray shadows require bounded scene counts, a known light and finite world-space intervals"; return false; }
        const std::uint64_t nodes=RayShadowHeaderTexels,triangles=nodes+scene.Nodes.size()*RayShadowNodeTexels,
            materials=triangles+scene.Triangles.size()*RayShadowTriangleTexels,
            coverage=materials+scene.Materials.size()*RayShadowMaterialTexels;
        if (coverage>UINT32_MAX)
        { error="Ray-shadow payload offsets exceed their native integer address space"; return false; }
        std::array<RayShadowTexel,RayShadowHeaderTexels> header{};
        header[0]={frame.Enabled?1u:0u,unsigned(scene.Nodes.size()),unsigned(scene.Triangles.size()),unsigned(scene.Materials.size())};
        header[1]={unsigned(nodes),unsigned(triangles),unsigned(materials),unsigned(coverage)};
        header[2]={FloatWord(settings.NormalBias),FloatWord(settings.StartBias),FloatWord(settings.MaximumDistance),0};
        header[3]={unsigned(frame.PrimaryLight),RayShadowImageVersion,unsigned(scene.CoverageTexels.size()),0};
        output=header; error.clear(); return true;
    }
    bool BuildRayShadowImage(const RaySceneUpload& scene,const RayShadowFrame& frame,unsigned maximumTextureSide,
        RayShadowImage& output,std::string& error)
    {
        if (!ValidateRaySceneUpload(scene,error)) return false;
        std::array<RayShadowTexel,RayShadowHeaderTexels> header;
        if (!BuildRayShadowHeader(scene,frame,header,error)) return false;
        const std::uint64_t texels=std::uint64_t(header[1][3])+
            (scene.CoverageTexels.size()+RayShadowScalarLanes-1)/RayShadowScalarLanes;
        maximumTextureSide=std::min(maximumTextureSide,RayShadowMaximumTextureSide);
        if (maximumTextureSide<RayShadowHeaderTexels || texels>std::uint64_t(maximumTextureSide)*maximumTextureSide ||
            texels>RayShadowMaximumImageBytes/sizeof(RayShadowTexel))
        { error="Ray-shadow scene exceeds the sampled integer image or memory budget"; return false; }
        unsigned side=unsigned(std::ceil(std::sqrt(double(texels))));
        while (std::uint64_t(side)*side<texels) ++side;
        RayShadowImage candidate;
        candidate.Width=std::min(maximumTextureSide,std::max(side,RayShadowHeaderTexels));
        candidate.Height=unsigned((texels+candidate.Width-1)/candidate.Width);
        const auto padded=std::size_t(candidate.Width)*candidate.Height;
        if (padded>RayShadowMaximumImageBytes/sizeof(RayShadowTexel))
        { error="Ray-shadow row padding exceeds the image memory budget"; return false; }
        candidate.Texels.resize(padded);
        std::copy(header.begin(),header.end(),candidate.Texels.begin());
        const auto copy=[&](unsigned offset,const void* source,std::size_t bytes)
        { if (bytes) std::memcpy(candidate.Texels.data()+offset,source,bytes); };
        copy(header[1][0],scene.Nodes.data(),scene.Nodes.size()*sizeof(RaySceneNode));
        copy(header[1][1],scene.Triangles.data(),scene.Triangles.size()*sizeof(RaySceneTriangle));
        copy(header[1][2],scene.Materials.data(),scene.Materials.size()*sizeof(RaySceneMaterial));
        copy(header[1][3],scene.CoverageTexels.data(),scene.CoverageTexels.size()*sizeof(float));
        output=std::move(candidate); error.clear(); return true;
    }
}
