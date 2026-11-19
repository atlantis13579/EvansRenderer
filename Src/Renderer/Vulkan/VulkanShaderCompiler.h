#pragma once

#include "Renderer/Vulkan/VulkanABI.h"
#include "Renderer/Vulkan/VulkanGlslCompiler.h"

#include <string>
#include <vector>

namespace Evans::Renderer
{
    enum class VulkanShaderDomain { Technique = 0, Editor = 2 };
    struct VulkanShaderProgram
    {
        const char* File;
        VulkanShaderStage Stage;
        const char* Definitions = ""; // Semicolon-separated NAME=VALUE options.
        VulkanShaderDomain Domain = VulkanShaderDomain::Technique;
    };

    // Source text, stage, options and compiler identity select local bytecode.
    // Changed inputs miss the cache; failed compiles never reuse stale output.
    bool CompileVulkanShaderText(const VulkanShaderProgram& program, const std::string& source,
        std::vector<std::uint32_t>& words, std::string& error);
    bool CompileVulkanShader(const VulkanShaderProgram& program,
        std::vector<std::uint32_t>& words, std::string& error);
    bool CreateVulkanShaderModule(VulkanABI::VkDevice device, VulkanABI::PFN_vkCreateShaderModule create,
        const VulkanShaderProgram& program, VulkanABI::VkShaderModule& module, std::string& error);
}
