#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define EVANS_VK_CALL __stdcall
#else
#define EVANS_VK_CALL
#endif

namespace Evans::Renderer::VulkanABI
{
using VkFlags = std::uint32_t;
using VkBool32 = std::uint32_t;
using VkDeviceSize = std::uint64_t;
using VkSampleMask = std::uint32_t;
using VkResult = std::int32_t;
using VkStructureType = std::int32_t;
using VkPhysicalDeviceType = std::int32_t;
using VkFormat = std::int32_t;
using VkImageType = std::int32_t;
using VkImageViewType = std::int32_t;
using VkImageTiling = std::int32_t;
using VkImageLayout = std::int32_t;
using VkSharingMode = std::int32_t;
using VkDescriptorType = std::int32_t;
using VkPipelineBindPoint = std::int32_t;
using VkCommandBufferLevel = std::int32_t;
using VkSubpassContents = std::int32_t;
using VkPrimitiveTopology = std::int32_t;
using VkPolygonMode = std::int32_t;
using VkFrontFace = std::int32_t;
using VkCompareOp = std::int32_t;
using VkStencilOp = std::int32_t;
using VkLogicOp = std::int32_t;
using VkBlendFactor = std::int32_t;
using VkBlendOp = std::int32_t;
using VkDynamicState = std::int32_t;
using VkVertexInputRate = std::int32_t;
using VkIndexType = std::int32_t;
using VkAttachmentLoadOp = std::int32_t;
using VkAttachmentStoreOp = std::int32_t;
using VkFilter = std::int32_t;
using VkSamplerMipmapMode = std::int32_t;
using VkSamplerAddressMode = std::int32_t;
using VkBorderColor = std::int32_t;
using VkComponentSwizzle = std::int32_t;

struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkCommandBuffer_T;
using VkInstance = VkInstance_T*;
using VkPhysicalDevice = VkPhysicalDevice_T*;
using VkDevice = VkDevice_T*;
using VkQueue = VkQueue_T*;
using VkCommandBuffer = VkCommandBuffer_T*;
using VkDeviceMemory = std::uint64_t;
using VkBuffer = std::uint64_t;
using VkBufferView = std::uint64_t;
using VkImage = std::uint64_t;
using VkImageView = std::uint64_t;
using VkShaderModule = std::uint64_t;
using VkPipeline = std::uint64_t;
using VkPipelineLayout = std::uint64_t;
using VkRenderPass = std::uint64_t;
using VkFramebuffer = std::uint64_t;
using VkCommandPool = std::uint64_t;
using VkFence = std::uint64_t;
using VkDescriptorSetLayout = std::uint64_t;
using VkDescriptorPool = std::uint64_t;
using VkDescriptorSet = std::uint64_t;
using VkSampler = std::uint64_t;
using VkPipelineCache = std::uint64_t;

constexpr VkBool32 VK_FALSE_VALUE = 0;
constexpr VkBool32 VK_TRUE_VALUE = 1;
constexpr VkResult VK_SUCCESS = 0;
constexpr VkResult VK_NOT_READY = 1;
constexpr std::uint32_t VK_QUEUE_FAMILY_IGNORED = 0xFFFFFFFFu;
constexpr std::uint32_t VK_SUBPASS_EXTERNAL = 0xFFFFFFFFu;
constexpr VkDeviceSize VK_WHOLE_SIZE = ~VkDeviceSize{0};

constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
constexpr VkStructureType VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
constexpr VkStructureType VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
constexpr VkStructureType VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
constexpr VkStructureType VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
constexpr VkStructureType VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO = 14;
constexpr VkStructureType VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15;
constexpr VkStructureType VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO = 19;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO = 20;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO = 22;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO = 23;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO = 24;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO = 25;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO = 26;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO = 27;
constexpr VkStructureType VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO = 28;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
constexpr VkStructureType VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO = 31;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
constexpr VkStructureType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
constexpr VkStructureType VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO = 37;
constexpr VkStructureType VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO = 38;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
constexpr VkStructureType VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO = 43;
constexpr VkStructureType VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER = 45;

constexpr VkPhysicalDeviceType VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2;
constexpr VkFlags VK_QUEUE_GRAPHICS_BIT = 0x00000001;
constexpr VkFlags VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;
constexpr VkFlags VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001;
constexpr VkFlags VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002;
constexpr VkFlags VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010;
constexpr VkFlags VK_BUFFER_USAGE_INDEX_BUFFER_BIT = 0x00000040;
constexpr VkFlags VK_BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x00000080;
constexpr VkFlags VK_IMAGE_USAGE_TRANSFER_SRC_BIT = 0x00000001;
constexpr VkFlags VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x00000002;
constexpr VkFlags VK_IMAGE_USAGE_SAMPLED_BIT = 0x00000004;
constexpr VkFlags VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT = 0x00000010;
constexpr VkFlags VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000020;
constexpr VkFlags VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001;
constexpr VkFlags VK_IMAGE_ASPECT_DEPTH_BIT = 0x00000002;
constexpr VkFlags VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002;
constexpr VkFlags VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001;
constexpr VkFlags VK_FENCE_CREATE_SIGNALED_BIT = 0x00000001;
constexpr VkFlags VK_SHADER_STAGE_VERTEX_BIT = 0x00000001;
constexpr VkFlags VK_SHADER_STAGE_FRAGMENT_BIT = 0x00000010;
constexpr VkFlags VK_CULL_MODE_NONE = 0;
constexpr VkFlags VK_CULL_MODE_FRONT_BIT = 0x00000001;
constexpr VkFlags VK_COLOR_COMPONENT_R_BIT = 0x00000001;
constexpr VkFlags VK_COLOR_COMPONENT_G_BIT = 0x00000002;
constexpr VkFlags VK_COLOR_COMPONENT_B_BIT = 0x00000004;
constexpr VkFlags VK_COLOR_COMPONENT_A_BIT = 0x00000008;
constexpr VkFlags VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001;
constexpr VkFlags VK_PIPELINE_STAGE_VERTEX_INPUT_BIT = 0x00000004;
constexpr VkFlags VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT = 0x00000080;
constexpr VkFlags VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT = 0x00000100;
constexpr VkFlags VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT = 0x00000200;
constexpr VkFlags VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400;
constexpr VkFlags VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000;
constexpr VkFlags VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x00002000;
constexpr VkFlags VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT = 0x00000004;
constexpr VkFlags VK_ACCESS_SHADER_READ_BIT = 0x00000020;
constexpr VkFlags VK_ACCESS_COLOR_ATTACHMENT_READ_BIT = 0x00000080;
constexpr VkFlags VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT = 0x00000100;
constexpr VkFlags VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x00000200;
constexpr VkFlags VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x00000400;
constexpr VkFlags VK_ACCESS_TRANSFER_READ_BIT = 0x00000800;
constexpr VkFlags VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000;
constexpr VkFlags VK_DEPENDENCY_BY_REGION_BIT = 0x00000001;

constexpr VkFormat VK_FORMAT_R8G8B8A8_UNORM = 37;
constexpr VkFormat VK_FORMAT_R8G8B8A8_SRGB = 43;
constexpr VkFormat VK_FORMAT_R32G32_SFLOAT = 103;
constexpr VkFormat VK_FORMAT_R32G32B32_SFLOAT = 106;
constexpr VkFormat VK_FORMAT_R32G32B32A32_SFLOAT = 109;
constexpr VkFormat VK_FORMAT_D32_SFLOAT = 126;
constexpr VkImageType VK_IMAGE_TYPE_2D = 1;
constexpr VkImageViewType VK_IMAGE_VIEW_TYPE_2D = 1;
constexpr VkImageTiling VK_IMAGE_TILING_OPTIMAL = 0;
constexpr VkImageLayout VK_IMAGE_LAYOUT_UNDEFINED = 0;
constexpr VkImageLayout VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL = 2;
constexpr VkImageLayout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 3;
constexpr VkImageLayout VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL = 5;
constexpr VkImageLayout VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL = 6;
constexpr VkImageLayout VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL = 7;
constexpr VkSharingMode VK_SHARING_MODE_EXCLUSIVE = 0;
constexpr VkDescriptorType VK_DESCRIPTOR_TYPE_SAMPLER = 0;
constexpr VkDescriptorType VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1;
constexpr VkDescriptorType VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE = 2;
constexpr VkDescriptorType VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC = 8;
constexpr VkPipelineBindPoint VK_PIPELINE_BIND_POINT_GRAPHICS = 0;
constexpr VkCommandBufferLevel VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
constexpr VkSubpassContents VK_SUBPASS_CONTENTS_INLINE = 0;
constexpr VkPrimitiveTopology VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3;
constexpr VkPolygonMode VK_POLYGON_MODE_FILL = 0;
constexpr VkPolygonMode VK_POLYGON_MODE_LINE = 1;
constexpr VkFrontFace VK_FRONT_FACE_COUNTER_CLOCKWISE = 0;
constexpr VkFrontFace VK_FRONT_FACE_CLOCKWISE = 1;
constexpr VkCompareOp VK_COMPARE_OP_LESS = 1;
constexpr VkCompareOp VK_COMPARE_OP_LESS_OR_EQUAL = 3;
constexpr VkBlendFactor VK_BLEND_FACTOR_ZERO = 0;
constexpr VkBlendFactor VK_BLEND_FACTOR_ONE = 1;
constexpr VkBlendFactor VK_BLEND_FACTOR_SRC_ALPHA = 6;
constexpr VkBlendFactor VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = 7;
constexpr VkBlendOp VK_BLEND_OP_ADD = 0;
constexpr VkDynamicState VK_DYNAMIC_STATE_VIEWPORT = 0;
constexpr VkDynamicState VK_DYNAMIC_STATE_SCISSOR = 1;
constexpr VkVertexInputRate VK_VERTEX_INPUT_RATE_VERTEX = 0;
constexpr VkIndexType VK_INDEX_TYPE_UINT32 = 1;
constexpr VkAttachmentLoadOp VK_ATTACHMENT_LOAD_OP_LOAD = 0;
constexpr VkAttachmentLoadOp VK_ATTACHMENT_LOAD_OP_CLEAR = 1;
constexpr VkAttachmentStoreOp VK_ATTACHMENT_STORE_OP_STORE = 0;
constexpr VkAttachmentStoreOp VK_ATTACHMENT_STORE_OP_DONT_CARE = 1;
constexpr VkFilter VK_FILTER_NEAREST = 0;
constexpr VkFilter VK_FILTER_LINEAR = 1;
constexpr VkSamplerMipmapMode VK_SAMPLER_MIPMAP_MODE_NEAREST = 0;
constexpr VkSamplerMipmapMode VK_SAMPLER_MIPMAP_MODE_LINEAR = 1;
constexpr VkSamplerAddressMode VK_SAMPLER_ADDRESS_MODE_REPEAT = 0;
constexpr VkSamplerAddressMode VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 2;
constexpr VkBorderColor VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE = 4;
constexpr VkComponentSwizzle VK_COMPONENT_SWIZZLE_IDENTITY = 0;
constexpr std::uint32_t VK_SAMPLE_COUNT_1_BIT = 0x00000001;

constexpr std::uint32_t MakeVersion(std::uint32_t major, std::uint32_t minor, std::uint32_t patch)
{
    return (major << 22U) | (minor << 12U) | patch;
}

struct VkExtent2D { std::uint32_t width; std::uint32_t height; };
struct VkExtent3D { std::uint32_t width; std::uint32_t height; std::uint32_t depth; };
struct VkOffset2D { std::int32_t x; std::int32_t y; };
struct VkOffset3D { std::int32_t x; std::int32_t y; std::int32_t z; };
struct VkRect2D { VkOffset2D offset; VkExtent2D extent; };
struct VkViewport { float x; float y; float width; float height; float minDepth; float maxDepth; };

struct VkApplicationInfo
{
    VkStructureType sType;
    const void* pNext;
    const char* pApplicationName;
    std::uint32_t applicationVersion;
    const char* pEngineName;
    std::uint32_t engineVersion;
    std::uint32_t apiVersion;
};

struct VkInstanceCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    std::uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    std::uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};

struct VkPhysicalDeviceFeatures
{
    VkBool32 values[55];
};

struct VkDeviceQueueCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t queueFamilyIndex;
    std::uint32_t queueCount;
    const float* pQueuePriorities;
};

struct VkDeviceCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    std::uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    std::uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures* pEnabledFeatures;
};

struct VkQueueFamilyProperties
{
    VkFlags queueFlags;
    std::uint32_t queueCount;
    std::uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
};

struct VkMemoryType { VkFlags propertyFlags; std::uint32_t heapIndex; };
struct VkMemoryHeap { VkDeviceSize size; VkFlags flags; };
struct VkPhysicalDeviceMemoryProperties
{
    std::uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    std::uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
};

struct VkMemoryAllocateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    std::uint32_t memoryTypeIndex;
};

struct VkMemoryRequirements
{
    VkDeviceSize size;
    VkDeviceSize alignment;
    std::uint32_t memoryTypeBits;
};

struct VkBufferCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    VkSharingMode sharingMode;
    std::uint32_t queueFamilyIndexCount;
    const std::uint32_t* pQueueFamilyIndices;
};

struct VkImageCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkImageType imageType;
    VkFormat format;
    VkExtent3D extent;
    std::uint32_t mipLevels;
    std::uint32_t arrayLayers;
    std::uint32_t samples;
    VkImageTiling tiling;
    VkFlags usage;
    VkSharingMode sharingMode;
    std::uint32_t queueFamilyIndexCount;
    const std::uint32_t* pQueueFamilyIndices;
    VkImageLayout initialLayout;
};

struct VkComponentMapping
{
    VkComponentSwizzle r;
    VkComponentSwizzle g;
    VkComponentSwizzle b;
    VkComponentSwizzle a;
};

struct VkImageSubresourceRange
{
    VkFlags aspectMask;
    std::uint32_t baseMipLevel;
    std::uint32_t levelCount;
    std::uint32_t baseArrayLayer;
    std::uint32_t layerCount;
};

struct VkImageViewCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkImage image;
    VkImageViewType viewType;
    VkFormat format;
    VkComponentMapping components;
    VkImageSubresourceRange subresourceRange;
};

struct VkImageSubresourceLayers
{
    VkFlags aspectMask;
    std::uint32_t mipLevel;
    std::uint32_t baseArrayLayer;
    std::uint32_t layerCount;
};

struct VkBufferImageCopy
{
    VkDeviceSize bufferOffset;
    std::uint32_t bufferRowLength;
    std::uint32_t bufferImageHeight;
    VkImageSubresourceLayers imageSubresource;
    VkOffset3D imageOffset;
    VkExtent3D imageExtent;
};

struct VkImageMemoryBarrier
{
    VkStructureType sType;
    const void* pNext;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
    VkImageLayout oldLayout;
    VkImageLayout newLayout;
    std::uint32_t srcQueueFamilyIndex;
    std::uint32_t dstQueueFamilyIndex;
    VkImage image;
    VkImageSubresourceRange subresourceRange;
};

struct VkCommandPoolCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkCommandPool commandPool;
    VkCommandBufferLevel level;
    std::uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
};

struct VkFenceCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
};

struct VkSubmitInfo
{
    VkStructureType sType;
    const void* pNext;
    std::uint32_t waitSemaphoreCount;
    const std::uint64_t* pWaitSemaphores;
    const VkFlags* pWaitDstStageMask;
    std::uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    std::uint32_t signalSemaphoreCount;
    const std::uint64_t* pSignalSemaphores;
};

struct VkAttachmentDescription
{
    VkFlags flags;
    VkFormat format;
    std::uint32_t samples;
    VkAttachmentLoadOp loadOp;
    VkAttachmentStoreOp storeOp;
    VkAttachmentLoadOp stencilLoadOp;
    VkAttachmentStoreOp stencilStoreOp;
    VkImageLayout initialLayout;
    VkImageLayout finalLayout;
};

struct VkAttachmentReference { std::uint32_t attachment; VkImageLayout layout; };
struct VkSubpassDescription
{
    VkFlags flags;
    VkPipelineBindPoint pipelineBindPoint;
    std::uint32_t inputAttachmentCount;
    const VkAttachmentReference* pInputAttachments;
    std::uint32_t colorAttachmentCount;
    const VkAttachmentReference* pColorAttachments;
    const VkAttachmentReference* pResolveAttachments;
    const VkAttachmentReference* pDepthStencilAttachment;
    std::uint32_t preserveAttachmentCount;
    const std::uint32_t* pPreserveAttachments;
};

struct VkSubpassDependency
{
    std::uint32_t srcSubpass;
    std::uint32_t dstSubpass;
    VkFlags srcStageMask;
    VkFlags dstStageMask;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
    VkFlags dependencyFlags;
};

struct VkRenderPassCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t attachmentCount;
    const VkAttachmentDescription* pAttachments;
    std::uint32_t subpassCount;
    const VkSubpassDescription* pSubpasses;
    std::uint32_t dependencyCount;
    const VkSubpassDependency* pDependencies;
};

struct VkFramebufferCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkRenderPass renderPass;
    std::uint32_t attachmentCount;
    const VkImageView* pAttachments;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t layers;
};

union VkClearColorValue
{
    float float32[4];
    std::int32_t int32[4];
    std::uint32_t uint32[4];
};
struct VkClearDepthStencilValue { float depth; std::uint32_t stencil; };
union VkClearValue
{
    VkClearColorValue color;
    VkClearDepthStencilValue depthStencil;
};

struct VkRenderPassBeginInfo
{
    VkStructureType sType;
    const void* pNext;
    VkRenderPass renderPass;
    VkFramebuffer framebuffer;
    VkRect2D renderArea;
    std::uint32_t clearValueCount;
    const VkClearValue* pClearValues;
};

struct VkShaderModuleCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::size_t codeSize;
    const std::uint32_t* pCode;
};

struct VkDescriptorSetLayoutBinding
{
    std::uint32_t binding;
    VkDescriptorType descriptorType;
    std::uint32_t descriptorCount;
    VkFlags stageFlags;
    const VkSampler* pImmutableSamplers;
};

struct VkDescriptorSetLayoutCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
};

struct VkDescriptorPoolSize { VkDescriptorType type; std::uint32_t descriptorCount; };
struct VkDescriptorPoolCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t maxSets;
    std::uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
};

struct VkDescriptorSetAllocateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    std::uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
};

struct VkDescriptorImageInfo { VkSampler sampler; VkImageView imageView; VkImageLayout imageLayout; };
struct VkDescriptorBufferInfo { VkBuffer buffer; VkDeviceSize offset; VkDeviceSize range; };
struct VkWriteDescriptorSet
{
    VkStructureType sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    std::uint32_t dstBinding;
    std::uint32_t dstArrayElement;
    std::uint32_t descriptorCount;
    VkDescriptorType descriptorType;
    const VkDescriptorImageInfo* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const VkBufferView* pTexelBufferView;
};

struct VkSamplerCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkFilter magFilter;
    VkFilter minFilter;
    VkSamplerMipmapMode mipmapMode;
    VkSamplerAddressMode addressModeU;
    VkSamplerAddressMode addressModeV;
    VkSamplerAddressMode addressModeW;
    float mipLodBias;
    VkBool32 anisotropyEnable;
    float maxAnisotropy;
    VkBool32 compareEnable;
    VkCompareOp compareOp;
    float minLod;
    float maxLod;
    VkBorderColor borderColor;
    VkBool32 unnormalizedCoordinates;
};

struct VkPipelineLayoutCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    std::uint32_t pushConstantRangeCount;
    const void* pPushConstantRanges;
};

struct VkPipelineShaderStageCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkFlags stage;
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
};

struct VkVertexInputBindingDescription
{
    std::uint32_t binding;
    std::uint32_t stride;
    VkVertexInputRate inputRate;
};
struct VkVertexInputAttributeDescription
{
    std::uint32_t location;
    std::uint32_t binding;
    VkFormat format;
    std::uint32_t offset;
};
struct VkPipelineVertexInputStateCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t vertexBindingDescriptionCount;
    const VkVertexInputBindingDescription* pVertexBindingDescriptions;
    std::uint32_t vertexAttributeDescriptionCount;
    const VkVertexInputAttributeDescription* pVertexAttributeDescriptions;
};
struct VkPipelineInputAssemblyStateCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkPrimitiveTopology topology;
    VkBool32 primitiveRestartEnable;
};
struct VkPipelineViewportStateCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t viewportCount;
    const VkViewport* pViewports;
    std::uint32_t scissorCount;
    const VkRect2D* pScissors;
};
struct VkPipelineRasterizationStateCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkBool32 depthClampEnable;
    VkBool32 rasterizerDiscardEnable;
    VkPolygonMode polygonMode;
    VkFlags cullMode;
    VkFrontFace frontFace;
    VkBool32 depthBiasEnable;
    float depthBiasConstantFactor;
    float depthBiasClamp;
    float depthBiasSlopeFactor;
    float lineWidth;
};
struct VkPipelineMultisampleStateCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t rasterizationSamples;
    VkBool32 sampleShadingEnable;
    float minSampleShading;
    const VkSampleMask* pSampleMask;
    VkBool32 alphaToCoverageEnable;
    VkBool32 alphaToOneEnable;
};
struct VkStencilOpState
{
    VkStencilOp failOp;
    VkStencilOp passOp;
    VkStencilOp depthFailOp;
    VkCompareOp compareOp;
    std::uint32_t compareMask;
    std::uint32_t writeMask;
    std::uint32_t reference;
};
struct VkPipelineDepthStencilStateCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkBool32 depthTestEnable;
    VkBool32 depthWriteEnable;
    VkCompareOp depthCompareOp;
    VkBool32 depthBoundsTestEnable;
    VkBool32 stencilTestEnable;
    VkStencilOpState front;
    VkStencilOpState back;
    float minDepthBounds;
    float maxDepthBounds;
};
struct VkPipelineColorBlendAttachmentState
{
    VkBool32 blendEnable;
    VkBlendFactor srcColorBlendFactor;
    VkBlendFactor dstColorBlendFactor;
    VkBlendOp colorBlendOp;
    VkBlendFactor srcAlphaBlendFactor;
    VkBlendFactor dstAlphaBlendFactor;
    VkBlendOp alphaBlendOp;
    VkFlags colorWriteMask;
};
struct VkPipelineColorBlendStateCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkBool32 logicOpEnable;
    VkLogicOp logicOp;
    std::uint32_t attachmentCount;
    const VkPipelineColorBlendAttachmentState* pAttachments;
    float blendConstants[4];
};
struct VkPipelineDynamicStateCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t dynamicStateCount;
    const VkDynamicState* pDynamicStates;
};
struct VkGraphicsPipelineCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t stageCount;
    const VkPipelineShaderStageCreateInfo* pStages;
    const VkPipelineVertexInputStateCreateInfo* pVertexInputState;
    const VkPipelineInputAssemblyStateCreateInfo* pInputAssemblyState;
    const void* pTessellationState;
    const VkPipelineViewportStateCreateInfo* pViewportState;
    const VkPipelineRasterizationStateCreateInfo* pRasterizationState;
    const VkPipelineMultisampleStateCreateInfo* pMultisampleState;
    const VkPipelineDepthStencilStateCreateInfo* pDepthStencilState;
    const VkPipelineColorBlendStateCreateInfo* pColorBlendState;
    const VkPipelineDynamicStateCreateInfo* pDynamicState;
    VkPipelineLayout layout;
    VkRenderPass renderPass;
    std::uint32_t subpass;
    VkPipeline basePipelineHandle;
    std::int32_t basePipelineIndex;
};

using PFN_vkVoidFunction = void(EVANS_VK_CALL*)();
using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction(EVANS_VK_CALL*)(VkInstance, const char*);
using PFN_vkGetDeviceProcAddr = PFN_vkVoidFunction(EVANS_VK_CALL*)(VkDevice, const char*);
using PFN_vkCreateInstance = VkResult(EVANS_VK_CALL*)(const VkInstanceCreateInfo*, const void*, VkInstance*);
using PFN_vkDestroyInstance = void(EVANS_VK_CALL*)(VkInstance, const void*);
using PFN_vkEnumeratePhysicalDevices = VkResult(EVANS_VK_CALL*)(VkInstance, std::uint32_t*, VkPhysicalDevice*);
using PFN_vkGetPhysicalDeviceProperties = void(EVANS_VK_CALL*)(VkPhysicalDevice, void*);
using PFN_vkGetPhysicalDeviceFeatures = void(EVANS_VK_CALL*)(VkPhysicalDevice, VkPhysicalDeviceFeatures*);
using PFN_vkGetPhysicalDeviceQueueFamilyProperties = void(EVANS_VK_CALL*)(VkPhysicalDevice, std::uint32_t*, VkQueueFamilyProperties*);
using PFN_vkGetPhysicalDeviceMemoryProperties = void(EVANS_VK_CALL*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
using PFN_vkCreateDevice = VkResult(EVANS_VK_CALL*)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
using PFN_vkDestroyDevice = void(EVANS_VK_CALL*)(VkDevice, const void*);
using PFN_vkGetDeviceQueue = void(EVANS_VK_CALL*)(VkDevice, std::uint32_t, std::uint32_t, VkQueue*);
using PFN_vkDeviceWaitIdle = VkResult(EVANS_VK_CALL*)(VkDevice);
using PFN_vkCreateBuffer = VkResult(EVANS_VK_CALL*)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
using PFN_vkDestroyBuffer = void(EVANS_VK_CALL*)(VkDevice, VkBuffer, const void*);
using PFN_vkGetBufferMemoryRequirements = void(EVANS_VK_CALL*)(VkDevice, VkBuffer, VkMemoryRequirements*);
using PFN_vkAllocateMemory = VkResult(EVANS_VK_CALL*)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
using PFN_vkFreeMemory = void(EVANS_VK_CALL*)(VkDevice, VkDeviceMemory, const void*);
using PFN_vkBindBufferMemory = VkResult(EVANS_VK_CALL*)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
using PFN_vkMapMemory = VkResult(EVANS_VK_CALL*)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
using PFN_vkUnmapMemory = void(EVANS_VK_CALL*)(VkDevice, VkDeviceMemory);
using PFN_vkCreateImage = VkResult(EVANS_VK_CALL*)(VkDevice, const VkImageCreateInfo*, const void*, VkImage*);
using PFN_vkDestroyImage = void(EVANS_VK_CALL*)(VkDevice, VkImage, const void*);
using PFN_vkGetImageMemoryRequirements = void(EVANS_VK_CALL*)(VkDevice, VkImage, VkMemoryRequirements*);
using PFN_vkBindImageMemory = VkResult(EVANS_VK_CALL*)(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize);
using PFN_vkCreateImageView = VkResult(EVANS_VK_CALL*)(VkDevice, const VkImageViewCreateInfo*, const void*, VkImageView*);
using PFN_vkDestroyImageView = void(EVANS_VK_CALL*)(VkDevice, VkImageView, const void*);
using PFN_vkCreateSampler = VkResult(EVANS_VK_CALL*)(VkDevice, const VkSamplerCreateInfo*, const void*, VkSampler*);
using PFN_vkDestroySampler = void(EVANS_VK_CALL*)(VkDevice, VkSampler, const void*);
using PFN_vkCreateRenderPass = VkResult(EVANS_VK_CALL*)(VkDevice, const VkRenderPassCreateInfo*, const void*, VkRenderPass*);
using PFN_vkDestroyRenderPass = void(EVANS_VK_CALL*)(VkDevice, VkRenderPass, const void*);
using PFN_vkCreateFramebuffer = VkResult(EVANS_VK_CALL*)(VkDevice, const VkFramebufferCreateInfo*, const void*, VkFramebuffer*);
using PFN_vkDestroyFramebuffer = void(EVANS_VK_CALL*)(VkDevice, VkFramebuffer, const void*);
using PFN_vkCreateShaderModule = VkResult(EVANS_VK_CALL*)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
using PFN_vkDestroyShaderModule = void(EVANS_VK_CALL*)(VkDevice, VkShaderModule, const void*);
using PFN_vkCreateDescriptorSetLayout = VkResult(EVANS_VK_CALL*)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
using PFN_vkDestroyDescriptorSetLayout = void(EVANS_VK_CALL*)(VkDevice, VkDescriptorSetLayout, const void*);
using PFN_vkCreateDescriptorPool = VkResult(EVANS_VK_CALL*)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
using PFN_vkDestroyDescriptorPool = void(EVANS_VK_CALL*)(VkDevice, VkDescriptorPool, const void*);
using PFN_vkAllocateDescriptorSets = VkResult(EVANS_VK_CALL*)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
using PFN_vkUpdateDescriptorSets = void(EVANS_VK_CALL*)(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t, const void*);
using PFN_vkCreatePipelineLayout = VkResult(EVANS_VK_CALL*)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
using PFN_vkDestroyPipelineLayout = void(EVANS_VK_CALL*)(VkDevice, VkPipelineLayout, const void*);
using PFN_vkCreateGraphicsPipelines = VkResult(EVANS_VK_CALL*)(VkDevice, VkPipelineCache, std::uint32_t, const VkGraphicsPipelineCreateInfo*, const void*, VkPipeline*);
using PFN_vkDestroyPipeline = void(EVANS_VK_CALL*)(VkDevice, VkPipeline, const void*);
using PFN_vkCreateCommandPool = VkResult(EVANS_VK_CALL*)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
using PFN_vkDestroyCommandPool = void(EVANS_VK_CALL*)(VkDevice, VkCommandPool, const void*);
using PFN_vkResetCommandPool = VkResult(EVANS_VK_CALL*)(VkDevice, VkCommandPool, VkFlags);
using PFN_vkAllocateCommandBuffers = VkResult(EVANS_VK_CALL*)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
using PFN_vkBeginCommandBuffer = VkResult(EVANS_VK_CALL*)(VkCommandBuffer, const VkCommandBufferBeginInfo*);
using PFN_vkEndCommandBuffer = VkResult(EVANS_VK_CALL*)(VkCommandBuffer);
using PFN_vkCreateFence = VkResult(EVANS_VK_CALL*)(VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
using PFN_vkDestroyFence = void(EVANS_VK_CALL*)(VkDevice, VkFence, const void*);
using PFN_vkWaitForFences = VkResult(EVANS_VK_CALL*)(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t);
using PFN_vkResetFences = VkResult(EVANS_VK_CALL*)(VkDevice, std::uint32_t, const VkFence*);
using PFN_vkQueueSubmit = VkResult(EVANS_VK_CALL*)(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence);
using PFN_vkCmdPipelineBarrier = void(EVANS_VK_CALL*)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, std::uint32_t, const void*, std::uint32_t, const void*, std::uint32_t, const VkImageMemoryBarrier*);
using PFN_vkCmdCopyBufferToImage = void(EVANS_VK_CALL*)(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t, const VkBufferImageCopy*);
using PFN_vkCmdCopyImageToBuffer = void(EVANS_VK_CALL*)(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t, const VkBufferImageCopy*);
using PFN_vkCmdBeginRenderPass = void(EVANS_VK_CALL*)(VkCommandBuffer, const VkRenderPassBeginInfo*, VkSubpassContents);
using PFN_vkCmdEndRenderPass = void(EVANS_VK_CALL*)(VkCommandBuffer);
using PFN_vkCmdSetViewport = void(EVANS_VK_CALL*)(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*);
using PFN_vkCmdSetScissor = void(EVANS_VK_CALL*)(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*);
using PFN_vkCmdBindPipeline = void(EVANS_VK_CALL*)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
using PFN_vkCmdBindDescriptorSets = void(EVANS_VK_CALL*)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t, const VkDescriptorSet*, std::uint32_t, const std::uint32_t*);
using PFN_vkCmdBindVertexBuffers = void(EVANS_VK_CALL*)(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*, const VkDeviceSize*);
using PFN_vkCmdBindIndexBuffer = void(EVANS_VK_CALL*)(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType);
using PFN_vkCmdDrawIndexed = void(EVANS_VK_CALL*)(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t, std::uint32_t);
}
