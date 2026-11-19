#include "Renderer/Vulkan/VulkanGlslCompiler.h"

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Evans::Renderer
{
    namespace
    {
        // The Khronos front end parses this API's GLSL and emits SPIR-V
        // directly into memory. No files, command-line compiler or bytecode packing.
        // https://github.com/KhronosGroup/glslang#c-functional-interface-new
        struct CompilerProcess
        {
            std::mutex Mutex;
            bool Ready = glslang_initialize_process() != 0;
            ~CompilerProcess() { if (Ready) glslang_finalize_process(); }
        };
        std::string Log(const char* message) { return message ? message : ""; }
    }

    bool CompileVulkanGlsl(const std::string& source, const std::string& preamble,
        VulkanShaderStage stage, std::vector<std::uint32_t>& words,
        std::string& diagnostics) noexcept
    {
        words.clear(); diagnostics.clear();
        try
        {
            static CompilerProcess process;
            std::lock_guard<std::mutex> lock(process.Mutex);
            if (source.empty() || !process.Ready)
            { diagnostics = "GLSL source is empty or compiler initialization failed"; return false; }
            glslang_stage_t nativeStage;
            switch (stage)
            {
            case VulkanShaderStage::Vertex: nativeStage = GLSLANG_STAGE_VERTEX; break;
            case VulkanShaderStage::Fragment:
            case VulkanShaderStage::MeshFragment: nativeStage = GLSLANG_STAGE_FRAGMENT; break;
            case VulkanShaderStage::Compute: nativeStage = GLSLANG_STAGE_COMPUTE; break;
            case VulkanShaderStage::Mesh: nativeStage = GLSLANG_STAGE_MESH; break;
            case VulkanShaderStage::Task: nativeStage = GLSLANG_STAGE_TASK; break;
            default: diagnostics = "Unknown Vulkan shader stage"; return false;
            }
            // Mesh primitive fragment inputs also carry SPV_EXT_mesh_shader.
            const bool meshEnvironment=stage==VulkanShaderStage::Mesh || stage==VulkanShaderStage::MeshFragment || stage==VulkanShaderStage::Task;
            const bool rayEnvironment=preamble.find("#define EVANS_LUMEN_HARDWARE_RAYS ")!=std::string::npos;
            constexpr int DefaultGlslVersion = 450;
            const auto rules = GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT;
            glslang_input_t input {};
            input.language = GLSLANG_SOURCE_GLSL; input.stage = nativeStage;
            input.client = GLSLANG_CLIENT_VULKAN; input.client_version = (meshEnvironment||rayEnvironment)?GLSLANG_TARGET_VULKAN_1_2:GLSLANG_TARGET_VULKAN_1_0;
            input.target_language = GLSLANG_TARGET_SPV; input.target_language_version = meshEnvironment?GLSLANG_TARGET_SPV_1_5:rayEnvironment?GLSLANG_TARGET_SPV_1_4:GLSLANG_TARGET_SPV_1_0;
            input.code = source.c_str(); input.default_version = DefaultGlslVersion; input.default_profile = GLSLANG_NO_PROFILE;
            input.messages = static_cast<glslang_messages_t>(rules); input.resource = glslang_default_resource();
            std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)> shader(glslang_shader_create(&input), glslang_shader_delete);
            if (!shader) { diagnostics = "Cannot allocate GLSL shader"; return false; }
            glslang_shader_set_preamble(shader.get(), preamble.c_str());
            if (!glslang_shader_preprocess(shader.get(), &input) || !glslang_shader_parse(shader.get(), &input))
            {
                diagnostics = Log(glslang_shader_get_info_log(shader.get())) + Log(glslang_shader_get_info_debug_log(shader.get()));
                return false;
            }
            std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)> program(glslang_program_create(), glslang_program_delete);
            if (!program) { diagnostics = "Cannot allocate GLSL program"; return false; }
            glslang_program_add_shader(program.get(), shader.get());
            if (!glslang_program_link(program.get(), rules))
            {
                diagnostics = Log(glslang_program_get_info_log(program.get())) + Log(glslang_program_get_info_debug_log(program.get()));
                return false;
            }
            glslang_spv_options_t options {};
            options.optimize_size = true; options.validate = true;
            glslang_program_SPIRV_generate_with_options(program.get(), nativeStage, &options);
            diagnostics = Log(glslang_program_SPIRV_get_messages(program.get()));
            const auto count = glslang_program_SPIRV_get_size(program.get());
            if (!count || diagnostics.find("ERROR") != std::string::npos || diagnostics.find("error:") != std::string::npos)
            { if (diagnostics.empty()) diagnostics = "GLSL compilation produced no Vulkan code"; return false; }
            words.resize(count); glslang_program_SPIRV_get(program.get(), words.data());
            return true;
        }
        catch (...)
        {
            words.clear();
            diagnostics = "GLSL compiler failed while allocating or processing the source";
            return false;
        }
    }

} // namespace Evans::Renderer
