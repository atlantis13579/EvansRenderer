#pragma once
#include "Renderer/Vulkan/VulkanABI.h"
namespace Evans::Renderer
{
    void InitializeVulkanPipelineCache(VulkanABI::VkDevice device,VulkanABI::VkPhysicalDevice physical,
        VulkanABI::PFN_vkGetPhysicalDeviceProperties properties,VulkanABI::PFN_vkGetDeviceProcAddr address);
    void DestroyVulkanPipelineCache(VulkanABI::VkDevice device);
    VulkanABI::VkResult CreateCachedVulkanGraphicsPipelines(VulkanABI::PFN_vkCreateGraphicsPipelines create,
        VulkanABI::VkDevice device,VulkanABI::VkPipelineCache cache,std::uint32_t count,
        const VulkanABI::VkGraphicsPipelineCreateInfo* descriptions,const void* allocator,VulkanABI::VkPipeline* pipelines);
    VulkanABI::VkResult CreateCachedVulkanComputePipelines(VulkanABI::PFN_vkCreateComputePipelines create,
        VulkanABI::VkDevice device,VulkanABI::VkPipelineCache cache,std::uint32_t count,
        const VulkanABI::VkComputePipelineCreateInfo* descriptions,const void* allocator,VulkanABI::VkPipeline* pipelines);
}
