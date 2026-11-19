#include "Renderer/Vulkan/VulkanPipelineCache.h"
#include "Renderer/Vulkan/VulkanContactShadows.h"
#include "Renderer/Vulkan/VulkanShaderPrograms.h"
#include <cstring>

namespace Evans::Renderer
{
    using namespace VulkanABI;
    struct VulkanContactShadows::Impl
    {
        static constexpr std::uint32_t GroupEdge = 8;
        struct Constants { ContactShadowConstants Trace; std::array<Float4, 8> Offsets; };
        static_assert(sizeof(Constants) == 336 && offsetof(Constants, Offsets) == 208, "Vulkan contact UBO layout changed");
        VkDevice Device = nullptr;
        VkPhysicalDeviceMemoryProperties Memory {};
        VkBuffer Uniform = 0;
        VkDeviceMemory UniformMemory = 0, ImageMemory = 0;
        VkImage Image = 0;
        VkImageView View = 0;
        VkSampler Sampler = 0;
        VkDescriptorPool Pool = 0;
        VkDescriptorSetLayout SetLayout = 0;
        VkDescriptorSet Set = 0;
        VkPipelineLayout Layout = 0;
        std::array<VkPipeline, 2> Pipelines {};
        VkImageLayout ImageLayout = VK_IMAGE_LAYOUT_UNDEFINED, FrameStartLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint32_t Width = 0, Height = 0, Samples = 0;
        bool Open = false, Recorded = false, Ready = false;
        std::string Error;
#define CONTACT_FUNCTIONS(X) \
        X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) \
        X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(CreateImage) X(DestroyImage) \
        X(GetImageMemoryRequirements) X(BindImageMemory) X(CreateImageView) X(DestroyImageView) \
        X(CreateSampler) X(DestroySampler) X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
        X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
        X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateShaderModule) X(DestroyShaderModule) \
        X(CreateComputePipelines) X(DestroyPipeline) X(CmdPipelineBarrier) X(CmdBindPipeline) \
        X(CmdBindDescriptorSets) X(CmdDispatch)
#define DECLARE(name) PFN_vk##name name = nullptr;
        CONTACT_FUNCTIONS(DECLARE)
#undef DECLARE
        bool Fail(const std::string& message) { Error = message; return false; }
        bool Check(VkResult result, const char* action)
        { return result == VK_SUCCESS || Fail(std::string("Vulkan contact ") + action + " failed: " + std::to_string(result)); }
        bool Allocate(const VkMemoryRequirements& requirement, VkFlags flags, VkDeviceMemory& destination)
        {
            for (std::uint32_t type = 0; type < Memory.memoryTypeCount; ++type)
                if ((requirement.memoryTypeBits & (1u << type)) && (Memory.memoryTypes[type].propertyFlags & flags) == flags)
                {
                    VkMemoryAllocateInfo info {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                    info.allocationSize = requirement.size; info.memoryTypeIndex = type;
                    return Check(AllocateMemory(Device, &info, nullptr, &destination), "allocate memory");
                }
            return Fail("Vulkan contact memory type is unavailable");
        }
        bool Initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory, PFN_vkGetDeviceProcAddr load)
        {
            if (!device || !load) return Fail("Vulkan contact initialization requires a device and loader");
            Device = device; Memory = memory;
#define LOAD(name) name = reinterpret_cast<PFN_vk##name>(load(Device, "vk" #name)); if (!name) return Fail("Missing vk" #name);
            CONTACT_FUNCTIONS(LOAD)
#undef LOAD
#undef CONTACT_FUNCTIONS
            VkBufferCreateInfo uniform {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            uniform.size = sizeof(Constants); uniform.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            if (!Check(CreateBuffer(Device, &uniform, nullptr, &Uniform), "create uniform buffer")) return false;
            VkMemoryRequirements requirement {}; GetBufferMemoryRequirements(Device, Uniform, &requirement);
            if (!Allocate(requirement, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, UniformMemory) ||
                !Check(BindBufferMemory(Device, Uniform, UniformMemory, 0), "bind uniform memory")) return false;
            VkSamplerCreateInfo sampler {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sampler.maxAnisotropy = 1;
            if (!Check(CreateSampler(Device, &sampler, nullptr, &Sampler), "create depth sampler")) return false;
            const VkDescriptorSetLayoutBinding bindings[] {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
            VkDescriptorSetLayoutCreateInfo set {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            set.bindingCount = 3; set.pBindings = bindings;
            if (!Check(CreateDescriptorSetLayout(Device, &set, nullptr, &SetLayout), "create descriptor layout")) return false;
            const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
            VkDescriptorPoolCreateInfo pool {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pool.maxSets = 1; pool.poolSizeCount = 3; pool.pPoolSizes = sizes;
            if (!Check(CreateDescriptorPool(Device, &pool, nullptr, &Pool), "create descriptor pool")) return false;
            VkDescriptorSetAllocateInfo allocation {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocation.descriptorPool = Pool; allocation.descriptorSetCount = 1; allocation.pSetLayouts = &SetLayout;
            if (!Check(AllocateDescriptorSets(Device, &allocation, &Set), "allocate descriptors")) return false;
            VkPipelineLayoutCreateInfo layout {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout.setLayoutCount = 1; layout.pSetLayouts = &SetLayout;
            if (!Check(CreatePipelineLayout(Device, &layout, nullptr, &Layout), "create pipeline layout")) return false;
            for (unsigned multisample = 0; multisample < Pipelines.size(); ++multisample)
            {
                VkShaderModule module = 0;
                if (!CreateVulkanShaderModule(Device, CreateShaderModule, multisample ? VulkanContactComputeMS : VulkanContactCompute, module, Error)) return false;
                VkComputePipelineCreateInfo pipeline {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pipeline.stage.module = module; pipeline.stage.pName = "main";
                pipeline.layout = Layout; pipeline.basePipelineIndex = -1;
                const bool created = Check(Evans::Renderer::CreateCachedVulkanComputePipelines(CreateComputePipelines,Device,0, 1, &pipeline, nullptr, &Pipelines[multisample]), "create compute pipeline");
                DestroyShaderModule(Device, module, nullptr);
                if (!created) return false;
            }
            return true;
        }
        void ReleaseImage()
        {
            if (View) DestroyImageView(Device, View, nullptr);
            if (Image) DestroyImage(Device, Image, nullptr);
            if (ImageMemory) FreeMemory(Device, ImageMemory, nullptr);
            View = Image = ImageMemory = 0; Width = Height = Samples = 0;
            ImageLayout = FrameStartLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
        bool EnsureImage(const VulkanContactDepth& input)
        {
            if (Image && Width == input.Width && Height == input.Height && Samples == input.Samples) return true;
            // The owner has fenced prior work before BeginFrame. A newly allocated
            // image has no executed layout even when a recording is later aborted.
            ReleaseImage();
            VkImageCreateInfo image {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            image.imageType = VK_IMAGE_TYPE_2D; image.format = VK_FORMAT_R32_SFLOAT;
            image.extent = {input.Width, input.Height, 1}; image.mipLevels = 1; image.arrayLayers = input.Samples; image.samples = 1;
            image.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (!Check(CreateImage(Device, &image, nullptr, &Image), "create visibility array")) return false;
            VkMemoryRequirements requirement {}; GetImageMemoryRequirements(Device, Image, &requirement);
            if (!Allocate(requirement, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ImageMemory) ||
                !Check(BindImageMemory(Device, Image, ImageMemory, 0), "bind visibility memory")) return false;
            VkImageViewCreateInfo view {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = Image; view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; view.format = image.format;
            view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, input.Samples};
            if (!Check(CreateImageView(Device, &view, nullptr, &View), "create visibility view")) return false;
            Width = input.Width; Height = input.Height; Samples = input.Samples;
            return true;
        }
        bool Record(VkCommandBuffer commands, const VulkanContactDepth& input, VkImageLayout& depthLayout,
            const ContactShadowDispatch& trace)
        {
            Ready = false; Error.clear();
            if (!Device || !Open || Recorded || !commands || !input.Image || !input.View || input.Format != VK_FORMAT_D32_SFLOAT)
                return Fail("Vulkan contact recording requires one open frame, a command buffer and actual D32 depth");
            if (depthLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
                depthLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL && depthLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                return Fail("Vulkan contact input must have completed depth attachment or shader-read usage");
            Constants constants {};
            if (!BuildContactShadowConstants(trace, input.Width, input.Height, input.Samples, constants.Trace, Error)) return false;
            for (unsigned sample = 0; sample < input.Samples; ++sample)
            {
                const auto offset = input.SampleOffsets[sample];
                if (!std::isfinite(offset.X) || !std::isfinite(offset.Y) || offset.X < -0.5f || offset.X >= 0.5f || offset.Y < -0.5f || offset.Y >= 0.5f)
                    return Fail("Vulkan contact sample offsets must lie inside the pixel");
                constants.Offsets[sample] = {offset.X, offset.Y, 0, 0};
            }
            if (!EnsureImage(input)) return false;
            void* mapped = nullptr;
            if (!Check(MapMemory(Device, UniformMemory, 0, sizeof(constants), 0, &mapped), "map constants")) return false;
            std::memcpy(mapped, &constants, sizeof(constants)); UnmapMemory(Device, UniformMemory);
            const VkDescriptorBufferInfo uniform {Uniform, 0, sizeof(constants)};
            const VkDescriptorImageInfo depth {Sampler, input.View, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
            const VkDescriptorImageInfo output {0, View, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet writes[3] {};
            for (unsigned binding = 0; binding < 3; ++binding)
            { writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[binding].dstSet = Set; writes[binding].dstBinding = binding; writes[binding].descriptorCount = 1; }
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[0].pBufferInfo = &uniform;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[1].pImageInfo = &depth;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[2].pImageInfo = &output;
            UpdateDescriptorSets(Device, 3, writes, 0, nullptr);
            // The first barrier publishes early/late depth writes to compute.
            // Reuse waits for all previous readers before overwriting the array.
            // https://docs.vulkan.org/guide/latest/synchronization_examples.html
            VkImageMemoryBarrier barriers[2] {};
            for (auto& barrier : barriers)
            {
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            }
            barriers[0].image = input.Image; barriers[0].oldLayout = depthLayout; barriers[0].newLayout = depth.imageLayout;
            barriers[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
            barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barriers[0].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            barriers[1].image = Image; barriers[1].oldLayout = ImageLayout; barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[1].srcAccessMask = ImageLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 :
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            barriers[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, Samples};
            const VkFlags readers = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
            CmdPipelineBarrier(commands, readers | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
            depthLayout = depth.imageLayout; ImageLayout = VK_IMAGE_LAYOUT_GENERAL;
            CmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, Pipelines[Samples > 1 ? 1 : 0]);
            CmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, Layout, 0, 1, &Set, 0, nullptr);
            CmdDispatch(commands, (Width + GroupEdge - 1) / GroupEdge, (Height + GroupEdge - 1) / GroupEdge, Samples);
            auto& visibility = barriers[1]; visibility.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            visibility.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; visibility.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            CmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, readers, 0, 0, nullptr, 0, nullptr, 1, &visibility);
            Recorded = Ready = true; return true;
        }
        void Shutdown()
        {
            ReleaseImage();
            if (Uniform) DestroyBuffer(Device, Uniform, nullptr);
            if (UniformMemory) FreeMemory(Device, UniformMemory, nullptr);
            if (Sampler) DestroySampler(Device, Sampler, nullptr);
            for (auto pipeline : Pipelines) if (pipeline) DestroyPipeline(Device, pipeline, nullptr);
            if (Pool) DestroyDescriptorPool(Device, Pool, nullptr);
            if (Layout) DestroyPipelineLayout(Device, Layout, nullptr);
            if (SetLayout) DestroyDescriptorSetLayout(Device, SetLayout, nullptr);
        }
    };
    VulkanContactShadows::VulkanContactShadows() : m_Impl(std::make_unique<Impl>()) {}
    VulkanContactShadows::~VulkanContactShadows() { m_Impl->Shutdown(); }
    bool VulkanContactShadows::Initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory, PFN_vkGetDeviceProcAddr load, std::string& error)
    {
        Shutdown(); const bool valid = m_Impl->Initialize(device, memory, load); error = m_Impl->Error;
        if (!valid) Shutdown();
        return valid;
    }
    void VulkanContactShadows::Shutdown() { m_Impl->Shutdown(); m_Impl = std::make_unique<Impl>(); }
    void VulkanContactShadows::BeginFrame()
    { m_Impl->Open = true; m_Impl->Recorded = m_Impl->Ready = false; m_Impl->FrameStartLayout = m_Impl->ImageLayout; }
    void VulkanContactShadows::AbortFrame()
    { m_Impl->Open = m_Impl->Recorded = m_Impl->Ready = false; m_Impl->ImageLayout = m_Impl->FrameStartLayout; }
    bool VulkanContactShadows::Record(VkCommandBuffer commands, const VulkanContactDepth& input,
        VkImageLayout& depthLayout, const ContactShadowDispatch& trace, std::string& error)
    { const bool valid = m_Impl->Record(commands, input, depthLayout, trace); error = m_Impl->Error; return valid; }
    VkImage VulkanContactShadows::Result() const { return m_Impl->Ready ? m_Impl->Image : 0; }
    VkImageView VulkanContactShadows::ResultView() const { return m_Impl->Ready ? m_Impl->View : 0; }
}
