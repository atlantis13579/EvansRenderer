#pragma once

#include "Renderer/BakedLighting/BakedLightmapCache.h"
#include "Renderer/BakedLighting/BakedLightProbes.h"
#include <filesystem>

namespace Evans::Renderer
{
    // The cooker and runtime use the same ordered source instances and bake
    // parameters. Camera, exposure and indirect intensity are not bake inputs.
    [[nodiscard]] LightmapBakeRequest MakeLightmapBakeRequest(std::vector<LightmapSourceInstance> instances,
        const Techniques::Settings& settings, const LightingState& lighting, bool shadows, float maximumDistance = 0);
    [[nodiscard]] std::uint64_t LightmapAssetKey(const LightmapBakeRequest& request);
    [[nodiscard]] std::uint64_t LightmapReceiverKey(const LightmapBakeRequest& request);
    [[nodiscard]] std::uint64_t ProbeAssetKey(const std::vector<SceneRayMeshView>& meshes,
        const Techniques::Settings& settings, const LightingState& lighting, bool shadows);

    // Versioned, bounded little-endian archives contain native-independent
    // irradiance/receiver streams or raw probe rays. Native shaders still own
    // lightmap evaluation and SH/moment projection. Lightmaps retain the baked
    // lighting when realtime lights/materials change; receiver topology and
    // transforms must still match. Missing or incompatible archives return null.
    [[nodiscard]] std::shared_ptr<const LightmapBakeResult> LoadLightmapAsset(const std::filesystem::path& path,
        const LightmapBakeRequest& request);
    [[nodiscard]] std::shared_ptr<const BakedProbeSamples> LoadProbeAsset(const std::filesystem::path& path, std::uint64_t key);
    [[nodiscard]] bool SaveLightmapAsset(const std::filesystem::path& path, const LightmapBakeRequest& request,
        const LightmapBakeResult& result, std::string& error);
    [[nodiscard]] bool SaveProbeAsset(const std::filesystem::path& path, std::uint64_t key,
        const BakedProbeSamples& samples, std::string& error);
}
