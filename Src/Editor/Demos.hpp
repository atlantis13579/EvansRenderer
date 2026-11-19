#pragma once

#include "Editor/Resources.hpp"

#include <filesystem>
#include <string_view>
#include <vector>

namespace Evans::Editor
{

struct DemoDescriptor
{
    DemoId Id = DemoId::Triangle;
    std::string_view Name;
    std::string_view Description;
};

class DemoCatalog
{
public:
    explicit DemoCatalog(std::filesystem::path contentRoot);

    [[nodiscard]] static const std::vector<DemoDescriptor>& Descriptors();
    [[nodiscard]] static bool SupportsBackend(DemoId demo,Techniques::Backend backend);
    [[nodiscard]] SceneSnapshot Load(DemoId demo) const;
    void Update(SceneSnapshot& scene, float deltaSeconds, float teapotSpeedDegrees) const;
    static void ApplyDragonAppearance(Techniques::Settings& settings);
    static void ApplyHairAppearance(const SceneSnapshot& scene, Techniques::Settings& settings);
    static void ConfigureSponzaLighting(SceneSnapshot& scene);
    static void ApplySponzaAppearance(Techniques::Settings& settings);
    [[nodiscard]] static bool SupportsSponzaLightmaps(const Techniques::Settings& settings) noexcept;
    [[nodiscard]] static bool IsVolumeDemo(DemoId demo) noexcept;
    static void ApplyVolumeAppearance(const SceneSnapshot& scene, Techniques::Settings& settings);

private:
    [[nodiscard]] SceneSnapshot LoadTriangle() const;
    [[nodiscard]] SceneSnapshot LoadDragon() const;
    [[nodiscard]] SceneSnapshot LoadHair() const;
    [[nodiscard]] SceneSnapshot LoadVolume(DemoId demo) const;
    [[nodiscard]] SceneSnapshot LoadModelDemo(
        DemoId demo,
        const std::filesystem::path& relativePath) const;

    std::filesystem::path ContentRoot_;
    std::shared_ptr<const Renderer::EnvironmentRadiance> SkyEnvironment_;
};

} // namespace Evans::Editor
