#pragma once
#include "Renderer/Vulkan/VulkanABI.h"

namespace Evans::Renderer::VulkanABI
{
// Vulkan 1.0 ABI prefix through standardSampleLocations. The driver still writes into
// the oversized aligned properties buffer below, never into this partial struct.
// Typed fields preserve native alignment, avoiding opaque byte offsets for limits.
// https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceLimits.html
struct PhysicalDeviceLimitsPrefix
{
    std::uint32_t maxImageDimension1D, maxImageDimension2D, maxImageDimension3D, maxImageDimensionCube;
    std::uint32_t maxImageArrayLayers, maxTexelBufferElements, maxUniformBufferRange, maxStorageBufferRange;
    std::uint32_t maxPushConstantsSize, maxMemoryAllocationCount, maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity, sparseAddressSpaceSize;
    std::uint32_t maxBoundDescriptorSets, maxPerStageDescriptorSamplers, maxPerStageDescriptorUniformBuffers;
    std::uint32_t maxPerStageDescriptorStorageBuffers, maxPerStageDescriptorSampledImages;
    std::uint32_t maxPerStageDescriptorStorageImages, maxPerStageDescriptorInputAttachments, maxPerStageResources;
    std::uint32_t maxDescriptorSetSamplers, maxDescriptorSetUniformBuffers, maxDescriptorSetUniformBuffersDynamic;
    std::uint32_t maxDescriptorSetStorageBuffers, maxDescriptorSetStorageBuffersDynamic, maxDescriptorSetSampledImages;
    std::uint32_t maxDescriptorSetStorageImages, maxDescriptorSetInputAttachments;
    std::uint32_t maxVertexInputAttributes, maxVertexInputBindings, maxVertexInputAttributeOffset;
    std::uint32_t maxVertexInputBindingStride, maxVertexOutputComponents;
    std::uint32_t maxTessellationGenerationLevel, maxTessellationPatchSize;
    std::uint32_t maxTessellationControlPerVertexInputComponents, maxTessellationControlPerVertexOutputComponents;
    std::uint32_t maxTessellationControlPerPatchOutputComponents, maxTessellationControlTotalOutputComponents;
    std::uint32_t maxTessellationEvaluationInputComponents, maxTessellationEvaluationOutputComponents;
    std::uint32_t maxGeometryShaderInvocations, maxGeometryInputComponents, maxGeometryOutputComponents;
    std::uint32_t maxGeometryOutputVertices, maxGeometryTotalOutputComponents;
    std::uint32_t maxFragmentInputComponents, maxFragmentOutputAttachments, maxFragmentDualSrcAttachments;
    std::uint32_t maxFragmentCombinedOutputResources, maxComputeSharedMemorySize, maxComputeWorkGroupCount[3];
    std::uint32_t maxComputeWorkGroupInvocations, maxComputeWorkGroupSize[3];
    std::uint32_t subPixelPrecisionBits, subTexelPrecisionBits, mipmapPrecisionBits;
    std::uint32_t maxDrawIndexedIndexValue, maxDrawIndirectCount;
    float maxSamplerLodBias, maxSamplerAnisotropy;
    std::uint32_t maxViewports, maxViewportDimensions[2];
    float viewportBoundsRange[2];
    std::uint32_t viewportSubPixelBits;
    std::size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment, minUniformBufferOffsetAlignment, minStorageBufferOffsetAlignment;
    std::int32_t minTexelOffset;
    std::uint32_t maxTexelOffset;
    std::int32_t minTexelGatherOffset;
    std::uint32_t maxTexelGatherOffset;
    float minInterpolationOffset, maxInterpolationOffset;
    std::uint32_t subPixelInterpolationOffsetBits, maxFramebufferWidth, maxFramebufferHeight, maxFramebufferLayers;
    VkFlags framebufferColorSampleCounts, framebufferDepthSampleCounts, framebufferStencilSampleCounts;
    VkFlags framebufferNoAttachmentsSampleCounts;
    std::uint32_t maxColorAttachments;
    VkFlags sampledImageColorSampleCounts, sampledImageIntegerSampleCounts, sampledImageDepthSampleCounts;
    VkFlags sampledImageStencilSampleCounts, storageImageSampleCounts;
    std::uint32_t maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    std::uint32_t maxClipDistances, maxCullDistances, maxCombinedClipAndCullDistances, discreteQueuePriorities;
    float pointSizeRange[2], lineWidthRange[2], pointSizeGranularity, lineWidthGranularity;
    VkBool32 strictLines, standardSampleLocations;
};

struct PhysicalDevicePropertiesPrefix
{
    std::uint32_t ApiVersion;
    std::uint32_t DriverVersion;
    std::uint32_t VendorId;
    std::uint32_t DeviceId;
    VkPhysicalDeviceType DeviceType;
    char DeviceName[256];
    std::uint8_t PipelineCacheUuid[16];
    PhysicalDeviceLimitsPrefix Limits;
};
static_assert(offsetof(PhysicalDevicePropertiesPrefix, Limits) == 296, "Vulkan properties ABI changed");

}
