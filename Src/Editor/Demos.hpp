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
    [[nodiscard]] SceneSnapshot Load(DemoId demo) const;
    void Update(SceneSnapshot& scene, float deltaSeconds, float teapotSpeedDegrees) const;

private:
    [[nodiscard]] SceneSnapshot LoadTriangle() const;
    [[nodiscard]] SceneSnapshot LoadModelDemo(
        DemoId demo,
        const std::filesystem::path& relativePath) const;

    std::filesystem::path ContentRoot_;
};

} // namespace Evans::Editor
