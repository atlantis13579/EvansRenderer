#pragma once

// The Editor extends Evans's header-only Vulkan ABI with Win32 WSI declarations.
// No Vulkan SDK headers or import libraries are required by the presenter.

#include "Renderer/Vulkan/VulkanABI.h"

#include <cstdint>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace Evans::Editor::VulkanPresenterABI
{
using namespace Evans::Renderer::VulkanABI;

using VkSurfaceKHR = std::uint64_t;
using VkSwapchainKHR = std::uint64_t;
using VkSemaphore = std::uint64_t;
using VkColorSpaceKHR = std::int32_t;
using VkPresentModeKHR = std::int32_t;
using VkSurfaceTransformFlagBitsKHR = std::uint32_t;
using VkCompositeAlphaFlagBitsKHR = std::uint32_t;

constexpr VkResult VK_SUBOPTIMAL_KHR = 1000001003;
constexpr VkResult VK_ERROR_OUT_OF_DATE_KHR = -1000001004;
constexpr VkResult VK_ERROR_SURFACE_LOST_KHR = -1000000000;
constexpr VkStructureType VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9;
constexpr VkStructureType VK_STRUCTURE_TYPE_PRESENT_INFO_KHR = 1000001001;
constexpr VkStructureType VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR = 1000001000;
constexpr VkStructureType VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR = 1000009000;
constexpr VkImageLayout VK_IMAGE_LAYOUT_PRESENT_SRC_KHR = 1000001002;
constexpr VkFormat VK_FORMAT_R32G32_SFLOAT = 103;
constexpr VkFormat VK_FORMAT_B8G8R8A8_UNORM = 44;
constexpr VkColorSpaceKHR VK_COLOR_SPACE_SRGB_NONLINEAR_KHR = 0;
constexpr VkPresentModeKHR VK_PRESENT_MODE_IMMEDIATE_KHR = 0;
constexpr VkPresentModeKHR VK_PRESENT_MODE_MAILBOX_KHR = 1;
constexpr VkPresentModeKHR VK_PRESENT_MODE_FIFO_KHR = 2;
constexpr VkSurfaceTransformFlagBitsKHR VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR = 0x00000001U;
constexpr VkCompositeAlphaFlagBitsKHR VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR = 0x00000001U;
constexpr VkCompositeAlphaFlagBitsKHR VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR = 0x00000002U;
constexpr VkCompositeAlphaFlagBitsKHR VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR = 0x00000004U;
constexpr VkCompositeAlphaFlagBitsKHR VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR = 0x00000008U;
constexpr VkBlendFactor VK_BLEND_FACTOR_ZERO = 0;
constexpr VkBlendFactor VK_BLEND_FACTOR_ONE = 1;
constexpr VkBlendFactor VK_BLEND_FACTOR_SRC_ALPHA = 6;
constexpr VkBlendFactor VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = 7;
constexpr VkBlendOp VK_BLEND_OP_ADD = 0;
constexpr VkCompareOp VK_COMPARE_OP_ALWAYS = 7;
constexpr VkAttachmentLoadOp VK_ATTACHMENT_LOAD_OP_DONT_CARE = 2;

constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR = 1000059000;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT = 1000275000;
constexpr VkStructureType VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT = 1000275001;

struct VkExtensionProperties
{
    char extensionName[256];
    std::uint32_t specVersion;
};

struct VkPhysicalDeviceFeatures2KHR
{
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceFeatures features;
};

struct VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT
{
    VkStructureType sType;
    void* pNext;
    VkBool32 swapchainMaintenance1;
};

struct VkSwapchainPresentFenceInfoEXT
{
    VkStructureType sType;
    const void* pNext;
    std::uint32_t swapchainCount;
    const VkFence* pFences;
};

using PFN_vkEnumerateInstanceExtensionProperties = VkResult(EVANS_VK_CALL*)(
    const char*, std::uint32_t*, VkExtensionProperties*);
using PFN_vkEnumerateDeviceExtensionProperties = VkResult(EVANS_VK_CALL*)(
    VkPhysicalDevice, const char*, std::uint32_t*, VkExtensionProperties*);
using PFN_vkGetPhysicalDeviceFeatures2KHR = void(EVANS_VK_CALL*)(
    VkPhysicalDevice, VkPhysicalDeviceFeatures2KHR*);

struct VkWin32SurfaceCreateInfoKHR
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    HINSTANCE hinstance;
    HWND hwnd;
};

struct VkSurfaceCapabilitiesKHR
{
    std::uint32_t minImageCount;
    std::uint32_t maxImageCount;
    VkExtent2D currentExtent;
    VkExtent2D minImageExtent;
    VkExtent2D maxImageExtent;
    std::uint32_t maxImageArrayLayers;
    VkFlags supportedTransforms;
    VkSurfaceTransformFlagBitsKHR currentTransform;
    VkFlags supportedCompositeAlpha;
    VkFlags supportedUsageFlags;
};

struct VkSurfaceFormatKHR
{
    VkFormat format;
    VkColorSpaceKHR colorSpace;
};

struct VkSwapchainCreateInfoKHR
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkSurfaceKHR surface;
    std::uint32_t minImageCount;
    VkFormat imageFormat;
    VkColorSpaceKHR imageColorSpace;
    VkExtent2D imageExtent;
    std::uint32_t imageArrayLayers;
    VkFlags imageUsage;
    VkSharingMode imageSharingMode;
    std::uint32_t queueFamilyIndexCount;
    const std::uint32_t* pQueueFamilyIndices;
    VkSurfaceTransformFlagBitsKHR preTransform;
    VkCompositeAlphaFlagBitsKHR compositeAlpha;
    VkPresentModeKHR presentMode;
    VkBool32 clipped;
    VkSwapchainKHR oldSwapchain;
};

struct VkPresentInfoKHR
{
    VkStructureType sType;
    const void* pNext;
    std::uint32_t waitSemaphoreCount;
    const VkSemaphore* pWaitSemaphores;
    std::uint32_t swapchainCount;
    const VkSwapchainKHR* pSwapchains;
    const std::uint32_t* pImageIndices;
    VkResult* pResults;
};

struct VkSemaphoreCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
};

struct VkPushConstantRange
{
    VkFlags stageFlags;
    std::uint32_t offset;
    std::uint32_t size;
};

#if defined(_WIN64)
static_assert(sizeof(VkWin32SurfaceCreateInfoKHR) == 40,
    "Win32 Vulkan surface ABI mismatch.");
static_assert(sizeof(VkSurfaceCapabilitiesKHR) == 52,
    "Vulkan surface capabilities ABI mismatch.");
static_assert(sizeof(VkSurfaceFormatKHR) == 8,
    "Vulkan surface format ABI mismatch.");
static_assert(sizeof(VkSwapchainCreateInfoKHR) == 104,
    "Vulkan swapchain ABI mismatch.");
static_assert(sizeof(VkPresentInfoKHR) == 64,
    "Vulkan present ABI mismatch.");
static_assert(sizeof(VkSemaphoreCreateInfo) == 24,
    "Vulkan semaphore ABI mismatch.");
static_assert(sizeof(VkPushConstantRange) == 12,
    "Vulkan push constant range ABI mismatch.");
#endif

using PFN_vkCreateWin32SurfaceKHR = VkResult(EVANS_VK_CALL*)(
    VkInstance, const VkWin32SurfaceCreateInfoKHR*, const void*, VkSurfaceKHR*);
using PFN_vkDestroySurfaceKHR = void(EVANS_VK_CALL*)(VkInstance, VkSurfaceKHR, const void*);
using PFN_vkGetPhysicalDeviceSurfaceSupportKHR = VkResult(EVANS_VK_CALL*)(
    VkPhysicalDevice, std::uint32_t, VkSurfaceKHR, VkBool32*);
using PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = VkResult(EVANS_VK_CALL*)(
    VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR*);
using PFN_vkGetPhysicalDeviceSurfaceFormatsKHR = VkResult(EVANS_VK_CALL*)(
    VkPhysicalDevice, VkSurfaceKHR, std::uint32_t*, VkSurfaceFormatKHR*);
using PFN_vkGetPhysicalDeviceSurfacePresentModesKHR = VkResult(EVANS_VK_CALL*)(
    VkPhysicalDevice, VkSurfaceKHR, std::uint32_t*, VkPresentModeKHR*);
using PFN_vkCreateSwapchainKHR = VkResult(EVANS_VK_CALL*)(
    VkDevice, const VkSwapchainCreateInfoKHR*, const void*, VkSwapchainKHR*);
using PFN_vkDestroySwapchainKHR = void(EVANS_VK_CALL*)(VkDevice, VkSwapchainKHR, const void*);
using PFN_vkGetSwapchainImagesKHR = VkResult(EVANS_VK_CALL*)(
    VkDevice, VkSwapchainKHR, std::uint32_t*, VkImage*);
using PFN_vkAcquireNextImageKHR = VkResult(EVANS_VK_CALL*)(
    VkDevice, VkSwapchainKHR, std::uint64_t, VkSemaphore, VkFence, std::uint32_t*);
using PFN_vkQueuePresentKHR = VkResult(EVANS_VK_CALL*)(VkQueue, const VkPresentInfoKHR*);
using PFN_vkCreateSemaphore = VkResult(EVANS_VK_CALL*)(
    VkDevice, const VkSemaphoreCreateInfo*, const void*, VkSemaphore*);
using PFN_vkDestroySemaphore = void(EVANS_VK_CALL*)(VkDevice, VkSemaphore, const void*);
using PFN_vkCmdDraw = void(EVANS_VK_CALL*)(
    VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);
using PFN_vkCmdPushConstants = void(EVANS_VK_CALL*)(
    VkCommandBuffer, VkPipelineLayout, VkFlags, std::uint32_t, std::uint32_t, const void*);

} // namespace Evans::Editor::VulkanPresenterABI
