#include "Renderer/Vulkan/VulkanShaderCompiler.h"
#include "Renderer/ShaderPreparationProgress.h"
#include "Renderer/ShaderCache.h"
#include <cstring>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace Evans::Renderer
{
    namespace
    {
        struct CachedProgram { std::string Source; std::vector<std::uint32_t> Words; };
        std::mutex CacheMutex;
        std::unordered_map<std::string, CachedProgram> Programs;
        std::string ProgramName(const VulkanShaderProgram& program)
        { return program.File ? program.File : "<unnamed GLSL>"; }
        std::string Options(const VulkanShaderProgram& program)
        { return program.Definitions ? program.Definitions : ""; }
    }

    bool CompileVulkanShaderText(const VulkanShaderProgram& program, const std::string& source,
        std::vector<std::uint32_t>& words, std::string& error)
    {
        words.clear(); error.clear();
        ShaderPreparationScope::SourceLoaded("Vulkan", ProgramName(program).c_str());
        const auto options = Options(program);
        const auto key = ProgramName(program) + '\n' + std::to_string(static_cast<unsigned>(program.Domain)) + '\n' +
            std::to_string(static_cast<unsigned>(program.Stage)) + '\n' + options;
        std::lock_guard<std::mutex> lock(CacheMutex);
        const auto existing = Programs.find(key);
        if (existing != Programs.end() && existing->second.Source == source)
        { words = existing->second.Words; return true; }
        // Each file remains an independent shader. Only declared compile options
        // become a preamble; no shader-source concatenation or cross-API input.
        std::string preamble, definition; std::istringstream definitions(options);
        while (std::getline(definitions, definition, ';'))
        {
            if (definition.empty()) continue;
            const auto equals = definition.find('=');
            preamble += "#define " + definition.substr(0, equals) + ' ' +
                (equals == std::string::npos ? "1" : definition.substr(equals + 1)) + '\n';
        }
        std::string diagnostics;
        ShaderCache::Key diskKey("vulkan-spirv-v2");
        diskKey.Add(EVANS_VULKAN_COMPILER_ID).Add(source).Add(preamble).Value(program.Stage);
        const auto digest=diskKey.Finish();std::vector<std::uint8_t> artifact;
        if(ShaderCache::Read("vulkan-shader",digest,artifact) && artifact.size()>=20 && artifact.size()%4==0)
        {
            std::uint32_t magic=0;std::memcpy(&magic,artifact.data(),sizeof(magic));
            if(magic==0x07230203)
            {
                words.resize(artifact.size()/4);std::memcpy(words.data(),artifact.data(),artifact.size());
                Programs[key]={source,words};return true;
            }
        }
        if (!CompileVulkanGlsl(source, preamble, program.Stage, words, diagnostics))
        {
            Programs.erase(key);
            error = ProgramName(program) + " [" + options + "]:\n" +
                (diagnostics.empty() ? "GLSL compilation failed" : diagnostics);
            return false;
        }
        Programs[key] = {source, words};
        ShaderCache::Write("vulkan-shader",digest,words.data(),words.size()*sizeof(std::uint32_t));
        return true;
    }

    bool CompileVulkanShader(const VulkanShaderProgram& program,
        std::vector<std::uint32_t>& words, std::string& error)
    {
        words.clear(); error.clear();
        if (!program.File || !*program.File) { error = "Vulkan shader has no GLSL filename"; return false; }
#if defined(EVANS_SOURCE_DIRECTORY)
        const std::filesystem::path root = EVANS_SOURCE_DIRECTORY;
#else
        const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
#endif
        std::filesystem::path directory;
        switch (program.Domain)
        {
        case VulkanShaderDomain::Technique: directory = root / "Shaders/Vulkan"; break;
        case VulkanShaderDomain::Editor: directory = root / "Editor/Shaders"; break;
        default: error = "Unknown Vulkan GLSL source domain"; return false;
        }
        const auto path = directory / program.File;
        std::ifstream input(path, std::ios::binary);
        if (!input) { error = "Cannot read Vulkan GLSL: " + path.string(); return false; }
        const std::string source {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        return CompileVulkanShaderText(program, source, words, error);
    }

    bool CreateVulkanShaderModule(VulkanABI::VkDevice device, VulkanABI::PFN_vkCreateShaderModule create,
        const VulkanShaderProgram& program, VulkanABI::VkShaderModule& module, std::string& error)
    {
        module = 0;
        if (!device || !create) { error = "Cannot compile a Vulkan module without its device entry point"; return false; }
        std::vector<std::uint32_t> words;
        if (!CompileVulkanShader(program, words, error)) return false;
        VulkanABI::VkShaderModuleCreateInfo info {};
        info.sType = VulkanABI::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = words.size() * sizeof(std::uint32_t); info.pCode = words.data();
        const auto result = create(device, &info, nullptr, &module);
        if (result == VulkanABI::VK_SUCCESS) return true;
        module = 0; error = "vkCreateShaderModule(" + ProgramName(program) + ") failed: " + std::to_string(result);
        return false;
    }
}
