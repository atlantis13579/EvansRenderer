#pragma once
#include "Renderer/Lighting/ForwardPlusContract.h"
#include "Renderer/Vulkan/VulkanDeviceProperties.h"
#include <memory>

namespace Evans::Renderer
{
    struct VulkanForwardPlusDepth
    {
        VulkanABI::VkImage Image = 0;
        VulkanABI::VkImageView View = 0;
        VulkanABI::VkFormat Format = VulkanABI::VK_FORMAT_D32_SFLOAT;
        std::uint32_t Width = 0, Height = 0, Samples = 1;
    };
    // Caller supplies actual base-mip D32 sampled depth and keeps its image/view,
    // command buffer and owning device alive through completion. All commands
    // and consumers use the same graphics/compute queue, outside render passes.
    // Submit the recorded work AND its consumers with SubmissionFence(), then
    // call MarkSubmitted immediately after successful vkQueueSubmit. This class
    // owns that fence; callers must never reset, destroy or otherwise signal it.
    // BeginFrame rejects pending work before mutating descriptors or storage.
    // AbortFrame requires discarding the unsubmitted command buffer; its owner
    // restores the depth-layout snapshot. Compute bindings/push constants change.
    class VulkanForwardPlus
    {
    public:
        explicit VulkanForwardPlus(TiledLightShader shader = TiledLightShader::ForwardPlus);
        ~VulkanForwardPlus(); // Waits for registered submissions; device stays alive.
        VulkanForwardPlus(const VulkanForwardPlus&) = delete;
        VulkanForwardPlus& operator=(const VulkanForwardPlus&) = delete;
        bool Initialize(VulkanABI::VkDevice device, const VulkanABI::VkPhysicalDeviceMemoryProperties& memory,
            const VulkanABI::PhysicalDeviceLimitsPrefix& limits, VulkanABI::PFN_vkGetDeviceProcAddr load, std::string& error);
        bool BeginFrame(std::string& error);
        bool AbortFrame(std::string& error);
        bool MarkSubmitted(std::string& error);
        // Alternatively track the frame owner's fence. It must cover assignment
        // and every consumer, and remain alive and unreset until BeginFrame or
        // Shutdown releases the completed submission. This class never owns it.
        bool MarkSubmitted(VulkanABI::VkFence completionFence, std::string& error);
        bool Shutdown(std::string& error);
        VulkanABI::VkFence SubmissionFence() const;
        bool Record(VulkanABI::VkCommandBuffer commands, const VulkanForwardPlusDepth& depth,
            VulkanABI::VkImageLayout& layout, const Matrix4& camera,
            const std::vector<ForwardPlusLightBounds>& lights, const ForwardPlusParameters& parameters, std::string& error);
        VulkanABI::VkDescriptorBufferInfo Headers() const;
        VulkanABI::VkDescriptorBufferInfo Indices() const;
        ForwardPlusGrid Grid() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
