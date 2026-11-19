#pragma once
#include "Renderer/BakedLighting/BakedLightProbes.h"
#include <filesystem>

namespace Evans::Renderer
{
    // Geometry/material edits rebuild the immutable query; light/grid edits only
    // rebake samples. Camera, display, intensity and gather controls do neither.
    class BakedProbeCache
    {
    public:
        bool Update(const std::vector<SceneRayMeshView>& meshes, std::uint64_t sceneRevision,
            const Techniques::Settings& settings, const LightingState& lighting, bool shadows, std::string& error,
            const std::filesystem::path& archive = {});
        const std::shared_ptr<const BakedProbeSamples>& Samples() const noexcept { return Output; }
        std::uint64_t BakeCount() const noexcept { return Bakes; }
        std::uint64_t QueryBuildCount() const noexcept { return QueryBuilds; }
        bool LoadedFromAsset() const noexcept { return Loaded; }
    private:
        std::shared_ptr<const SceneRayQuery> Query;
        std::shared_ptr<const BakedProbeSamples> Output;
        std::uint64_t Revision = 0, Bakes = 0, QueryBuilds = 0;
        Techniques::Settings MaterialSettings;
        LightingState Lighting;
        Techniques::LightProbeSettings RequestedProbes;
        std::filesystem::path Archive;
        bool Loaded = false;
    };
}
