#pragma once
#include "Renderer/BakedLighting/BakedLightmapCache.h"

namespace Evans::Renderer
{
struct EnvironmentRadiance;
enum class ArvoBakeBackend { CUDA, CPU };
struct ArvoBakeOptions
{
    std::filesystem::path Executable;
    std::filesystem::path ProjectRoot;
    std::filesystem::path Directory;
    uint32_t Samples=128,Bounces=3,Threads=0,Seed=1;
    bool BackFaces=true;
    ArvoBakeBackend Backend=ArvoBakeBackend::CUDA;
    std::shared_ptr<const EnvironmentRadiance> Environment;
    float EnvironmentScale=0;
    float SurfaceBias=0; // Zero selects a scene-relative offset.
};
// Only scene serialization, process ownership and result import live in Evans.
bool BakeWithArvo(const LightmapBakeRequest&,const ArvoBakeOptions&,
    std::string& error,const std::function<bool()>& cancelled={},
    const std::function<void(const std::string&)>& status={});
}
