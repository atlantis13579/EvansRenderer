#include "Editor/ContentRoot.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace Evans::Editor
{
namespace
{

[[nodiscard]] std::filesystem::path ExecutablePath()
{
#if defined(_WIN32)
    std::vector<wchar_t> buffer(260);
    for (;;)
    {
        const DWORD length = GetModuleFileNameW(
            nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) return {};
        if (length < buffer.size()) return std::filesystem::path(buffer.data());
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) return {};
    return std::filesystem::path(buffer.data());
#else
    std::error_code error;
    return std::filesystem::read_symlink("/proc/self/exe", error);
#endif
}

[[nodiscard]] bool HasDemoAssets(const std::filesystem::path& contentRoot)
{
    std::error_code error;
    return std::filesystem::is_regular_file(
               contentRoot / "Models/UtahTeapot/teapot.obj", error) &&
           std::filesystem::is_regular_file(
               contentRoot / "Scenes/Sponza/sponza.obj", error) &&
           std::filesystem::is_regular_file(
               contentRoot / "Models/StanfordDragon/dragon.obj", error);
}

[[nodiscard]] std::filesystem::path FindAbove(std::filesystem::path directory)
{
    while (!directory.empty())
    {
        const std::filesystem::path contentRoot = directory / "Contents";
        if (HasDemoAssets(contentRoot)) return contentRoot;
        const std::filesystem::path parent = directory.parent_path();
        if (parent == directory) break;
        directory = parent;
    }
    return {};
}

} // namespace

std::filesystem::path ResolveContentRoot()
{
    const std::filesystem::path executable = ExecutablePath();
    if (const std::filesystem::path found = FindAbove(executable.parent_path());
        !found.empty()) return found;

    std::error_code error;
    const std::filesystem::path workingDirectory = std::filesystem::current_path(error);
    if (!error)
    {
        if (const std::filesystem::path found = FindAbove(workingDirectory);
            !found.empty()) return found;
    }

    throw std::runtime_error(
        "Cannot find Contents with the Teapot, Sponza and Dragon demo assets near the "
        "executable or working directory: " + executable.string());
}

} // namespace Evans::Editor
