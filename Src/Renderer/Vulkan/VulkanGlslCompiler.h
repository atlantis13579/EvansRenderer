#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Evans::Renderer
{
    enum class VulkanShaderStage : std::uint32_t { Vertex, Fragment, Compute, Mesh, MeshFragment, Task };

    // glslang shares the renderer's runtime configuration and caller-owned output.
    bool CompileVulkanGlsl(const std::string& source, const std::string& preamble,
        VulkanShaderStage stage, std::vector<std::uint32_t>& words,
        std::string& diagnostics) noexcept;
}
