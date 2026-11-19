#pragma once
#include "Renderer/Lighting/ClusteredForwardContract.h"
#include "Renderer/Vulkan/VulkanDeviceProperties.h"
#include <memory>

namespace Evans::Renderer
{
    // One dense cluster assignment per frame on the owning graphics/compute
    // queue. Submit assignment and all consumers with SubmissionFence(), then
    // MarkSubmitted; alternatively register the frame owner's completion fence.
    // BeginFrame rejects pending work before changing descriptors or uploads.
    // AbortFrame requires discarding unsubmitted command buffers first.
    class VulkanClusteredForward
    {
    public:
        VulkanClusteredForward();
        ~VulkanClusteredForward(); // Waits for registered submissions; device stays alive.
        VulkanClusteredForward(const VulkanClusteredForward&) = delete;
        VulkanClusteredForward& operator=(const VulkanClusteredForward&) = delete;
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
        bool Record(VulkanABI::VkCommandBuffer commands, std::uint32_t width, std::uint32_t height, const Matrix4& camera,
            const std::vector<ForwardPlusLightBounds>& lights, const ClusteredForwardParameters& parameters, std::string& error);
        VulkanABI::VkDescriptorBufferInfo Headers() const;
        VulkanABI::VkDescriptorBufferInfo Indices() const;
        ClusteredForwardGrid Grid() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
