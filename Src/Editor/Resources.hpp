#pragma once

#include "Editor/Types.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Evans::Editor
{

class ResourceParseError : public std::runtime_error
{
public:
    ResourceParseError(std::size_t line, const std::string& message);
    [[nodiscard]] std::size_t Line() const noexcept { return Line_; }

private:
    std::size_t Line_ = 0;
};

struct ObjDocument
{
    std::vector<MeshAsset> Meshes;
    std::vector<std::string> MaterialLibraries;
    std::vector<std::string> Warnings;
};

struct LoadedModel
{
    ObjDocument Geometry;
    std::vector<Material> Materials;
};

class ResourceParser
{
public:
    [[nodiscard]] static ObjDocument ParseObj(std::string_view source);
    [[nodiscard]] static std::vector<Material> ParseMtl(std::string_view source);
};

class FileResourceLoader
{
public:
    [[nodiscard]] static std::string ReadTextFile(const std::filesystem::path& path);
    [[nodiscard]] static std::shared_ptr<const ImageAsset> LoadImage(
        const std::filesystem::path& path);
    [[nodiscard]] static LoadedModel LoadObj(const std::filesystem::path& path);
};

} // namespace Evans::Editor
