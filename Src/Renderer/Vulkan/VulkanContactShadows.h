#pragma once

#include "Renderer/Shadows/ContactShadowContract.h"
#include "Renderer/Vulkan/VulkanABI.h"
#include <memory>

namespace Evans::Renderer
{
    struct VulkanContactDepth
    {
        VulkanABI::VkImage Image = 0;
        VulkanABI::VkImageView View = 0; // depth-only 2D view of D32_SFLOAT, with SAMPLED usage
        VulkanABI::VkFormat Format = VulkanABI::VK_FORMAT_D32_SFLOAT;
        std::uint32_t Width = 0, Height = 0, Samples = 1;
        // Actual fixed raster sample pattern, top-left pixel-centred offsets. Keep the
        // sample index ordering; never use a resolved depth or reorder samples.
        std::array<Float2, 8> SampleOffsets {};
    };
    // Records a single compute producer outside a raster pass, on a compute-capable
    // command buffer. Input/output uses stay in the same queue family. The owner must
    // wait for previous GPU use before BeginFrame, resize or destruction, retain
    // input depth through submission, and restore its depth layout on abort.
    // AbortFrame is only for discarded command buffers, never submitted work.
    // The result stays in GENERAL, with shader/transfer reads made visible. It
    // has one R32F layer per depth sample; consumers must preserve this layout.
    // Compute pipeline/descriptors are overwritten; graphics state is separate.
    class VulkanContactShadows
    {
    public:
        VulkanContactShadows();
        ~VulkanContactShadows();
        VulkanContactShadows(const VulkanContactShadows&) = delete;
        VulkanContactShadows& operator=(const VulkanContactShadows&) = delete;
        bool Initialize(VulkanABI::VkDevice device, const VulkanABI::VkPhysicalDeviceMemoryProperties& memory,
            VulkanABI::PFN_vkGetDeviceProcAddr load, std::string& error);
        void Shutdown();
        void BeginFrame();
        void AbortFrame();
        bool Record(VulkanABI::VkCommandBuffer commands, const VulkanContactDepth& input,
            VulkanABI::VkImageLayout& depthLayout, const ContactShadowDispatch& trace, std::string& error);
        VulkanABI::VkImage Result() const;
        VulkanABI::VkImageView ResultView() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
