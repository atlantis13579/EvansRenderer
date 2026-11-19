#include "Renderer/Types.h"
#include "Renderer/Vulkan/VulkanExternalMemory.h"
#include "Renderer/Vulkan/VulkanPipelineCache.h"
#include "Editor/EditorPresenterVulkan.h"

#if defined(_WIN32)

#include "Editor/ImguiRenderVulkan.h"
#include "Editor/VulkanPresenterABI.h"
#include "Renderer/Vulkan/VulkanShaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace Evans::Editor
{
using namespace VulkanPresenterABI;

namespace
{
constexpr std::uint64_t GpuWaitTimeout = 5'000'000'000ULL;
constexpr std::uint64_t InvalidRevision = ~std::uint64_t{0};

struct PushConstants
{
    float Scale[2];
    float Translate[2];
};

[[nodiscard]] std::string VulkanError(const char* operation, VkResult result)
{
    std::ostringstream stream;
    stream << operation << " failed with VkResult " << result;
    return stream.str();
}
}

struct EditorPresenterVulkan::Impl
{
    struct Buffer
    {
        VkBuffer Handle = 0;
        VkDeviceMemory Memory = 0;
        VkDeviceSize Size = 0;
    };

    struct Texture
    {
        VkImage Handle = 0;
        VkDeviceMemory Memory = 0;
        VkImageView View = 0;
        VkImageLayout Layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint32_t Width = 0;
        std::uint32_t Height = 0;
    };

    ~Impl()
    {
        Shutdown();
    }

    bool Fail(std::string message)
    {
        LastError = std::move(message);
        return false;
    }

    bool Check(VkResult result, const char* operation)
    {
        return result == VK_SUCCESS || Fail(VulkanError(operation, result));
    }

    template <typename Function>
    bool LoadGlobal(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetInstanceProcAddress(nullptr, name));
        return destination != nullptr || Fail(std::string("Missing Vulkan global function: ") + name);
    }

    template <typename Function>
    bool LoadInstance(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetInstanceProcAddress(Instance, name));
        return destination != nullptr || Fail(std::string("Missing Vulkan instance function: ") + name);
    }

    template <typename Function>
    bool LoadDevice(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetDeviceProcAddress(Device, name));
        return destination != nullptr || Fail(std::string("Missing Vulkan device function: ") + name);
    }

    bool LoadRuntime()
    {
        VulkanModule = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (VulkanModule == nullptr)
        {
            return Fail("Vulkan presentation is unavailable: vulkan-1.dll was not found.");
        }
        GetInstanceProcAddress = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            GetProcAddress(VulkanModule, "vkGetInstanceProcAddr"));
        if (GetInstanceProcAddress == nullptr)
        {
            return Fail("Vulkan presentation is unavailable: vkGetInstanceProcAddr is missing.");
        }
        return LoadGlobal(CreateInstanceFunction, "vkCreateInstance") &&
            LoadGlobal(EnumerateInstanceExtensions, "vkEnumerateInstanceExtensionProperties");
    }

    bool RecreateSignaledFrameFence()
    {
        if (FrameFence != 0 && DestroyFenceFunction != nullptr)
        {
            DestroyFenceFunction(Device, FrameFence, nullptr);
            FrameFence = 0;
        }
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        return Check(
            CreateFenceFunction(Device, &fence, nullptr, &FrameFence),
            "vkCreateFence");
    }

    bool LoadInstanceFunctions()
    {
        return LoadInstance(DestroyInstanceFunction, "vkDestroyInstance") &&
            LoadInstance(EnumerateDeviceExtensions, "vkEnumerateDeviceExtensionProperties") &&
            LoadInstance(EnumeratePhysicalDevices, "vkEnumeratePhysicalDevices") &&
            LoadInstance(GetPhysicalDeviceQueueFamilyProperties,
                "vkGetPhysicalDeviceQueueFamilyProperties") &&
            LoadInstance(GetPhysicalDeviceMemoryProperties,
                "vkGetPhysicalDeviceMemoryProperties") &&
            LoadInstance(GetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties") &&
            LoadInstance(CreateDeviceFunction, "vkCreateDevice") &&
            LoadInstance(GetDeviceProcAddress, "vkGetDeviceProcAddr") &&
            LoadInstance(CreateWin32Surface, "vkCreateWin32SurfaceKHR") &&
            LoadInstance(DestroySurface, "vkDestroySurfaceKHR") &&
            LoadInstance(GetSurfaceSupport, "vkGetPhysicalDeviceSurfaceSupportKHR") &&
            LoadInstance(GetSurfaceCapabilities, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR") &&
            LoadInstance(GetSurfaceFormats, "vkGetPhysicalDeviceSurfaceFormatsKHR") &&
            LoadInstance(GetSurfacePresentModes, "vkGetPhysicalDeviceSurfacePresentModesKHR");
    }

    bool LoadDeviceFunctions()
    {
        return LoadDevice(DestroyDeviceFunction, "vkDestroyDevice") &&
            LoadDevice(GetDeviceQueue, "vkGetDeviceQueue") &&
            LoadDevice(DeviceWaitIdle, "vkDeviceWaitIdle") &&
            LoadDevice(CreateSwapchain, "vkCreateSwapchainKHR") &&
            LoadDevice(DestroySwapchain, "vkDestroySwapchainKHR") &&
            LoadDevice(GetSwapchainImages, "vkGetSwapchainImagesKHR") &&
            LoadDevice(AcquireNextImage, "vkAcquireNextImageKHR") &&
            LoadDevice(QueuePresent, "vkQueuePresentKHR") &&
            LoadDevice(CreateSemaphoreFunction, "vkCreateSemaphore") &&
            LoadDevice(DestroySemaphoreFunction, "vkDestroySemaphore") &&
            LoadDevice(CreateFenceFunction, "vkCreateFence") &&
            LoadDevice(DestroyFenceFunction, "vkDestroyFence") &&
            LoadDevice(WaitForFences, "vkWaitForFences") &&
            LoadDevice(ResetFences, "vkResetFences") &&
            LoadDevice(CreateCommandPoolFunction, "vkCreateCommandPool") &&
            LoadDevice(DestroyCommandPoolFunction, "vkDestroyCommandPool") &&
            LoadDevice(ResetCommandPool, "vkResetCommandPool") &&
            LoadDevice(AllocateCommandBuffers, "vkAllocateCommandBuffers") &&
            LoadDevice(BeginCommandBuffer, "vkBeginCommandBuffer") &&
            LoadDevice(EndCommandBuffer, "vkEndCommandBuffer") &&
            LoadDevice(QueueSubmit, "vkQueueSubmit") &&
            LoadDevice(CreateBufferFunction, "vkCreateBuffer") &&
            LoadDevice(DestroyBufferFunction, "vkDestroyBuffer") &&
            LoadDevice(GetBufferMemoryRequirements, "vkGetBufferMemoryRequirements") &&
            LoadDevice(AllocateMemory, "vkAllocateMemory") &&
            LoadDevice(FreeMemory, "vkFreeMemory") &&
            LoadDevice(BindBufferMemory, "vkBindBufferMemory") &&
            LoadDevice(MapMemory, "vkMapMemory") &&
            LoadDevice(UnmapMemory, "vkUnmapMemory") &&
            LoadDevice(CreateImageFunction, "vkCreateImage") &&
            LoadDevice(DestroyImageFunction, "vkDestroyImage") &&
            LoadDevice(GetImageMemoryRequirements, "vkGetImageMemoryRequirements") &&
            LoadDevice(BindImageMemory, "vkBindImageMemory") &&
            LoadDevice(CreateImageViewFunction, "vkCreateImageView") &&
            LoadDevice(DestroyImageViewFunction, "vkDestroyImageView") &&
            LoadDevice(CreateSamplerFunction, "vkCreateSampler") &&
            LoadDevice(DestroySamplerFunction, "vkDestroySampler") &&
            LoadDevice(CreateDescriptorSetLayoutFunction, "vkCreateDescriptorSetLayout") &&
            LoadDevice(DestroyDescriptorSetLayoutFunction, "vkDestroyDescriptorSetLayout") &&
            LoadDevice(CreateDescriptorPoolFunction, "vkCreateDescriptorPool") &&
            LoadDevice(DestroyDescriptorPoolFunction, "vkDestroyDescriptorPool") &&
            LoadDevice(AllocateDescriptorSets, "vkAllocateDescriptorSets") &&
            LoadDevice(UpdateDescriptorSets, "vkUpdateDescriptorSets") &&
            LoadDevice(CreateRenderPassFunction, "vkCreateRenderPass") &&
            LoadDevice(DestroyRenderPassFunction, "vkDestroyRenderPass") &&
            LoadDevice(CreateFramebufferFunction, "vkCreateFramebuffer") &&
            LoadDevice(DestroyFramebufferFunction, "vkDestroyFramebuffer") &&
            LoadDevice(CreateShaderModuleFunction, "vkCreateShaderModule") &&
            LoadDevice(DestroyShaderModuleFunction, "vkDestroyShaderModule") &&
            LoadDevice(CreatePipelineLayoutFunction, "vkCreatePipelineLayout") &&
            LoadDevice(DestroyPipelineLayoutFunction, "vkDestroyPipelineLayout") &&
            LoadDevice(CreateGraphicsPipelines, "vkCreateGraphicsPipelines") &&
            LoadDevice(DestroyPipelineFunction, "vkDestroyPipeline") &&
            LoadDevice(CmdPipelineBarrier, "vkCmdPipelineBarrier") &&
            LoadDevice(CmdCopyImage, "vkCmdCopyImage") &&
            LoadDevice(CmdCopyBufferToImage, "vkCmdCopyBufferToImage") &&
            LoadDevice(CmdBeginRenderPass, "vkCmdBeginRenderPass") &&
            LoadDevice(CmdEndRenderPass, "vkCmdEndRenderPass") &&
            LoadDevice(CmdSetViewport, "vkCmdSetViewport") &&
            LoadDevice(CmdSetScissor, "vkCmdSetScissor") &&
            LoadDevice(CmdBindPipeline, "vkCmdBindPipeline") &&
            LoadDevice(CmdBindDescriptorSets, "vkCmdBindDescriptorSets") &&
            LoadDevice(CmdBindVertexBuffers, "vkCmdBindVertexBuffers") &&
            LoadDevice(CmdDraw, "vkCmdDraw") &&
            LoadDevice(CmdPushConstants, "vkCmdPushConstants");
    }

    bool CreateInstanceAndSurface(HWND window)
    {
        VkApplicationInfo application{};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.pApplicationName = "Evans Editor";
        application.applicationVersion = MakeVersion(1, 0, 0);
        application.pEngineName = "Evans";
        application.engineVersion = MakeVersion(1, 0, 0);
        application.apiVersion = MakeVersion(1, 0, 0);
        std::vector<const char*> extensions = {"VK_KHR_surface", "VK_KHR_win32_surface"};
        std::uint32_t extensionCount = 0;
        if (!Check(EnumerateInstanceExtensions(nullptr, &extensionCount, nullptr),
                "vkEnumerateInstanceExtensionProperties(count)")) return false;
        std::vector<VkExtensionProperties> available(extensionCount);
        if (!Check(EnumerateInstanceExtensions(nullptr, &extensionCount, available.data()),
                "vkEnumerateInstanceExtensionProperties")) return false;
        const auto hasExtension = [&](const char* name) {
            return std::any_of(available.begin(), available.end(), [&](const auto& extension) {
                return std::strcmp(extension.extensionName, name) == 0;
            });
        };
        MaintenanceInstanceEnabled = hasExtension("VK_EXT_surface_maintenance1") &&
            hasExtension("VK_KHR_get_surface_capabilities2") &&
            hasExtension("VK_KHR_get_physical_device_properties2");
        if (MaintenanceInstanceEnabled)
        {
            extensions.push_back("VK_EXT_surface_maintenance1");
            extensions.push_back("VK_KHR_get_surface_capabilities2");
            extensions.push_back("VK_KHR_get_physical_device_properties2");
        }
        VkInstanceCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create.pApplicationInfo = &application;
        create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
        create.ppEnabledExtensionNames = extensions.data();
        if (!Check(CreateInstanceFunction(&create, nullptr, &Instance), "vkCreateInstance"))
        {
            return false;
        }
        if (!LoadInstanceFunctions())
        {
            return false;
        }
        VkWin32SurfaceCreateInfoKHR surfaceCreate{};
        surfaceCreate.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        surfaceCreate.hinstance = GetModuleHandleW(nullptr);
        surfaceCreate.hwnd = window;
        return Check(CreateWin32Surface(Instance, &surfaceCreate, nullptr, &Surface),
            "vkCreateWin32SurfaceKHR");
    }

    bool SelectPhysicalDevice()
    {
        std::uint32_t deviceCount = 0;
        if (!Check(EnumeratePhysicalDevices(Instance, &deviceCount, nullptr),
                "vkEnumeratePhysicalDevices(count)"))
        {
            return false;
        }
        if (deviceCount == 0)
        {
            return Fail("No Vulkan physical device is available.");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        if (!Check(EnumeratePhysicalDevices(Instance, &deviceCount, devices.data()),
                "vkEnumeratePhysicalDevices"))
        {
            return false;
        }
        for (VkPhysicalDevice candidate : devices)
        {
            std::uint32_t queueCount = 0;
            GetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, nullptr);
            std::vector<VkQueueFamilyProperties> queues(queueCount);
            GetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, queues.data());
            for (std::uint32_t queueIndex = 0; queueIndex < queueCount; ++queueIndex)
            {
                VkBool32 presentSupported = VK_FALSE_VALUE;
                if ((queues[queueIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U &&
                    GetSurfaceSupport(candidate, queueIndex, Surface, &presentSupported) == VK_SUCCESS &&
                    presentSupported == VK_TRUE_VALUE)
                {
                    PhysicalDevice = candidate;
                    QueueFamily = queueIndex;
                    GetPhysicalDeviceMemoryProperties(PhysicalDevice, &MemoryProperties);
                    return true;
                }
            }
        }
        return Fail("No Vulkan queue family supports both graphics and this Win32 surface.");
    }

    bool CreateDeviceAndQueue()
    {
        constexpr float priority = 1.0f;
        VkDeviceQueueCreateInfo queueCreate{};
        queueCreate.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreate.queueFamilyIndex = QueueFamily;
        queueCreate.queueCount = 1;
        queueCreate.pQueuePriorities = &priority;
        std::vector<const char*> extensions = {"VK_KHR_swapchain","VK_KHR_timeline_semaphore","VK_KHR_external_semaphore_win32","VK_KHR_external_semaphore","VK_KHR_external_memory_win32","VK_KHR_external_memory","VK_KHR_dedicated_allocation","VK_KHR_get_memory_requirements2"};
        VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT maintenance{};
        maintenance.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;
        if (MaintenanceInstanceEnabled)
        {
            std::uint32_t count = 0;
            if (!Check(EnumerateDeviceExtensions(PhysicalDevice, nullptr, &count, nullptr),
                    "vkEnumerateDeviceExtensionProperties(count)")) return false;
            std::vector<VkExtensionProperties> available(count);
            if (!Check(EnumerateDeviceExtensions(PhysicalDevice, nullptr, &count, available.data()),
                    "vkEnumerateDeviceExtensionProperties")) return false;
            if (std::any_of(available.begin(), available.end(), [](const auto& extension) {
                    return std::strcmp(extension.extensionName, "VK_EXT_swapchain_maintenance1") == 0;
                }))
            {
                PFN_vkGetPhysicalDeviceFeatures2KHR getFeatures = nullptr;
                if (!LoadInstance(getFeatures, "vkGetPhysicalDeviceFeatures2KHR")) return false;
                VkPhysicalDeviceFeatures2KHR features{};
                features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
                features.pNext = &maintenance;
                getFeatures(PhysicalDevice, &features);
                PresentFencesEnabled = maintenance.swapchainMaintenance1 != 0;
                if (PresentFencesEnabled) extensions.push_back("VK_EXT_swapchain_maintenance1");
            }
        }
        VkDeviceCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create.queueCreateInfoCount = 1;
        create.pQueueCreateInfos = &queueCreate;
        create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
        create.ppEnabledExtensionNames = extensions.data();
        TimelineSemaphoreFeatures timeline;timeline.pNext=PresentFencesEnabled?&maintenance:nullptr;create.pNext=&timeline;
        if (!Check(CreateDeviceFunction(PhysicalDevice, &create, nullptr, &Device),
                "vkCreateDevice"))
        {
            return false;
        }
        if (!LoadDeviceFunctions())
        {
            return false;
        }
        Renderer::InitializeVulkanPipelineCache(Device,PhysicalDevice,GetPhysicalDeviceProperties,GetDeviceProcAddress);
        GetDeviceQueue(Device, QueueFamily, 0, &Queue);
        return Queue != nullptr || Fail("vkGetDeviceQueue returned a null graphics queue.");
    }

    bool CreateCommandState()
    {
        VkCommandPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = QueueFamily;
        if (!Check(CreateCommandPoolFunction(Device, &pool, nullptr, &CommandPool),
                "vkCreateCommandPool"))
        {
            return false;
        }
        VkCommandBufferAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool = CommandPool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (!Check(AllocateCommandBuffers(Device, &allocate, &CommandBuffer),
                "vkAllocateCommandBuffers"))
        {
            return false;
        }
        if (!RecreateSignaledFrameFence())
        {
            return false;
        }
        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (!Check(CreateSemaphoreFunction(Device, &semaphore, nullptr, &ImageAvailable),
                   "vkCreateSemaphore(image available)")) return false;
        VkFenceCreateInfo acquireFence{};
        acquireFence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (!Check(CreateFenceFunction(Device, &acquireFence, nullptr, &AcquireFence),
                "vkCreateFence(acquire)")) return false;
        if (PresentFencesEnabled)
        {
            VkFenceCreateInfo fence{};
            fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if (!Check(CreateFenceFunction(Device, &fence, nullptr, &PresentFence),
                    "vkCreateFence(presentation)")) return false;
        }
        return true;
    }

    [[nodiscard]] std::uint32_t FindMemoryType(std::uint32_t bits, VkFlags flags) const
    {
        for (std::uint32_t index = 0; index < MemoryProperties.memoryTypeCount; ++index)
        {
            if ((bits & (1U << index)) != 0U &&
                (MemoryProperties.memoryTypes[index].propertyFlags & flags) == flags)
            {
                return index;
            }
        }
        return std::numeric_limits<std::uint32_t>::max();
    }

    bool CreateBuffer(VkDeviceSize size, VkFlags usage, VkFlags memoryFlags, Buffer& output)
    {
        DestroyBuffer(output);
        VkBufferCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create.size = std::max<VkDeviceSize>(size, 4U);
        create.usage = usage;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!Check(CreateBufferFunction(Device, &create, nullptr, &output.Handle),
                "vkCreateBuffer"))
        {
            return false;
        }
        VkMemoryRequirements requirements{};
        GetBufferMemoryRequirements(Device, output.Handle, &requirements);
        const std::uint32_t memoryType = FindMemoryType(requirements.memoryTypeBits, memoryFlags);
        if (memoryType == std::numeric_limits<std::uint32_t>::max())
        {
            DestroyBuffer(output);
            return Fail("Vulkan could not find a compatible buffer memory type.");
        }
        VkMemoryAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memoryType;
        if (!Check(AllocateMemory(Device, &allocate, nullptr, &output.Memory),
                "vkAllocateMemory(buffer)") ||
            !Check(BindBufferMemory(Device, output.Handle, output.Memory, 0),
                "vkBindBufferMemory"))
        {
            DestroyBuffer(output);
            return false;
        }
        output.Size = create.size;
        return true;
    }

    void DestroyBuffer(Buffer& buffer) noexcept
    {
        if (buffer.Handle != 0 && DestroyBufferFunction != nullptr)
        {
            DestroyBufferFunction(Device, buffer.Handle, nullptr);
        }
        if (buffer.Memory != 0 && FreeMemory != nullptr)
        {
            FreeMemory(Device, buffer.Memory, nullptr);
        }
        buffer = {};
    }

    bool EnsureBuffer(
        Buffer& buffer, VkDeviceSize size, VkFlags usage, VkFlags memoryFlags)
    {
        if (buffer.Handle != 0 && buffer.Size >= size)
        {
            return true;
        }
        const VkDeviceSize capacity = std::max<VkDeviceSize>(size, buffer.Size * 2U);
        return CreateBuffer(capacity, usage, memoryFlags, buffer);
    }

    bool CreateTexture(std::uint32_t width, std::uint32_t height, Texture& output)
    {
        if (width == 0 || height == 0)
        {
            return Fail("Vulkan editor textures require positive dimensions.");
        }
        DestroyTexture(output);
        VkImageCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        create.imageType = VK_IMAGE_TYPE_2D;
        create.format = VK_FORMAT_R8G8B8A8_UNORM;
        create.extent = {width, height, 1};
        create.mipLevels = 1;
        create.arrayLayers = 1;
        create.samples = VK_SAMPLE_COUNT_1_BIT;
        create.tiling = VK_IMAGE_TILING_OPTIMAL;
        create.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!Check(CreateImageFunction(Device, &create, nullptr, &output.Handle),
                "vkCreateImage(editor texture)"))
        {
            return false;
        }
        VkMemoryRequirements requirements{};
        GetImageMemoryRequirements(Device, output.Handle, &requirements);
        const std::uint32_t memoryType = FindMemoryType(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memoryType == std::numeric_limits<std::uint32_t>::max())
        {
            DestroyTexture(output);
            return Fail("Vulkan could not find device-local texture memory.");
        }
        VkMemoryAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memoryType;
        if (!Check(AllocateMemory(Device, &allocate, nullptr, &output.Memory),
                "vkAllocateMemory(editor texture)") ||
            !Check(BindImageMemory(Device, output.Handle, output.Memory, 0),
                "vkBindImageMemory(editor texture)"))
        {
            DestroyTexture(output);
            return false;
        }
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = output.Handle;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R8G8B8A8_UNORM;
        view.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!Check(CreateImageViewFunction(Device, &view, nullptr, &output.View),
                "vkCreateImageView(editor texture)"))
        {
            DestroyTexture(output);
            return false;
        }
        output.Width = width;
        output.Height = height;
        output.Layout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    }

    void DestroyTexture(Texture& texture) noexcept
    {
        if (texture.View != 0 && DestroyImageViewFunction != nullptr)
        {
            DestroyImageViewFunction(Device, texture.View, nullptr);
        }
        if (texture.Handle != 0 && DestroyImageFunction != nullptr)
        {
            DestroyImageFunction(Device, texture.Handle, nullptr);
        }
        if (texture.Memory != 0 && FreeMemory != nullptr)
        {
            FreeMemory(Device, texture.Memory, nullptr);
        }
        texture = {};
    }

    void TransitionTexture(Texture& texture, VkImageLayout layout)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = texture.Layout;
        barrier.newLayout = layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = texture.Handle;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkFlags destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (texture.Layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        if (layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        else
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        CmdPipelineBarrier(CommandBuffer, sourceStage, destinationStage, 0,
            0, nullptr, 0, nullptr, 1, &barrier);
        texture.Layout = layout;
    }

    bool UploadTexture(Texture& texture, const void* pixels, std::size_t byteCount)
    {
        const std::uint64_t expectedByteCount = static_cast<std::uint64_t>(texture.Width) *
            texture.Height * 4U;
        if (texture.Handle == 0 || pixels == nullptr || byteCount != expectedByteCount)
        {
            return Fail("Vulkan editor texture upload data is invalid.");
        }
        if (!EnsureBuffer(UploadBuffer, static_cast<VkDeviceSize>(byteCount),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        {
            return false;
        }
        void* mapped = nullptr;
        if (!Check(MapMemory(Device, UploadBuffer.Memory, 0, byteCount, 0, &mapped),
                "vkMapMemory(texture upload)"))
        {
            return false;
        }
        std::memcpy(mapped, pixels, byteCount);
        UnmapMemory(Device, UploadBuffer.Memory);
        TransitionTexture(texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {texture.Width, texture.Height, 1};
        CmdCopyBufferToImage(CommandBuffer, UploadBuffer.Handle, texture.Handle,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        TransitionTexture(texture, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return true;
    }

    bool CreateDescriptorState()
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = 1;
        layout.pBindings = &binding;
        if (!Check(CreateDescriptorSetLayoutFunction(Device, &layout, nullptr,
                &DescriptorSetLayout), "vkCreateDescriptorSetLayout(editor)"))
        {
            return false;
        }
        const VkPushConstantRange pushRange{
            VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<std::uint32_t>(sizeof(PushConstants))};
        VkPipelineLayoutCreateInfo pipelineLayout{};
        pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayout.setLayoutCount = 1;
        pipelineLayout.pSetLayouts = &DescriptorSetLayout;
        pipelineLayout.pushConstantRangeCount = 1;
        pipelineLayout.pPushConstantRanges = &pushRange;
        if (!Check(CreatePipelineLayoutFunction(Device, &pipelineLayout, nullptr,
                &PipelineLayout), "vkCreatePipelineLayout(editor)"))
        {
            return false;
        }
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = 0.0f;
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        if (!Check(CreateSamplerFunction(Device, &sampler, nullptr, &Sampler),
                "vkCreateSampler(editor)"))
        {
            return false;
        }
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3};
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 3;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &DescriptorPool),
                "vkCreateDescriptorPool(editor)"))
        {
            return false;
        }
        const std::array<VkDescriptorSetLayout, 3> layouts = {
            DescriptorSetLayout, DescriptorSetLayout, DescriptorSetLayout};
        std::array<VkDescriptorSet, 3> sets{};
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = DescriptorPool;
        allocate.descriptorSetCount = static_cast<std::uint32_t>(sets.size());
        allocate.pSetLayouts = layouts.data();
        if (!Check(AllocateDescriptorSets(Device, &allocate, sets.data()),
                "vkAllocateDescriptorSets(editor)"))
        {
            return false;
        }
        WhiteDescriptor = sets[0];
        FontDescriptor = sets[1];
        ViewportDescriptor = sets[2];
        return true;
    }

    void UpdateDescriptor(VkDescriptorSet descriptor, VkImageView imageView)
    {
        const VkDescriptorImageInfo image{
            Sampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image;
        UpdateDescriptorSets(Device, 1, &write, 0, nullptr);
    }

    bool BeginImmediateCommands()
    {
        if (!Check(WaitForFences(Device, 1, &FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
                "vkWaitForFences(upload)") ||
            !Check(ResetCommandPool(Device, CommandPool, 0), "vkResetCommandPool(upload)"))
        {
            return false;
        }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        return Check(BeginCommandBuffer(CommandBuffer, &begin), "vkBeginCommandBuffer(upload)");
    }

    bool EndImmediateCommands()
    {
        if (!Check(EndCommandBuffer(CommandBuffer), "vkEndCommandBuffer(upload)"))
        {
            return false;
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &CommandBuffer;
        if (!Check(ResetFences(Device, 1, &FrameFence), "vkResetFences(upload)"))
        {
            return false;
        }
        const VkResult submitResult = QueueSubmit(Queue, 1, &submit, FrameFence);
        if (submitResult != VK_SUCCESS)
        {
            const std::string submitError = VulkanError("vkQueueSubmit(upload)", submitResult);
            (void)RecreateSignaledFrameFence();
            return Fail(submitError);
        }
        return Check(
            WaitForFences(Device, 1, &FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
            "vkWaitForFences(upload completion)");
    }

    bool InitializeTexture(
        Texture& texture, std::uint32_t width, std::uint32_t height,
        const void* pixels, std::size_t byteCount)
    {
        if (!CreateTexture(width, height, texture) || !BeginImmediateCommands())
        {
            return false;
        }
        if (!UploadTexture(texture, pixels, byteCount))
        {
            return false;
        }
        return EndImmediateCommands();
    }

    bool CreateStaticTextures()
    {
        constexpr std::uint32_t white = 0xffffffffU;
        if (!InitializeTexture(WhiteTexture, 1, 1, &white, sizeof(white)))
        {
            return false;
        }
        UpdateDescriptor(WhiteDescriptor, WhiteTexture.View);
        const auto& fontPixels = ImguiRenderer.FontPixels();
        if (fontPixels.empty() || !InitializeTexture(FontTexture,
                static_cast<std::uint32_t>(ImguiRenderer.FontWidth()),
                static_cast<std::uint32_t>(ImguiRenderer.FontHeight()),
                fontPixels.data(), fontPixels.size()))
        {
            return Fail("The Vulkan presenter could not upload the ImGui font atlas.");
        }
        UpdateDescriptor(FontDescriptor, FontTexture.View);
        UpdateDescriptor(ViewportDescriptor, WhiteTexture.View);
        return true;
    }

    bool QuerySurface(
        VkSurfaceCapabilitiesKHR& capabilities,
        std::vector<VkSurfaceFormatKHR>& formats,
        std::vector<VkPresentModeKHR>& modes)
    {
        if (!Check(GetSurfaceCapabilities(PhysicalDevice, Surface, &capabilities),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))
        {
            return false;
        }
        std::uint32_t formatCount = 0;
        if (!Check(GetSurfaceFormats(PhysicalDevice, Surface, &formatCount, nullptr),
                "vkGetPhysicalDeviceSurfaceFormatsKHR(count)"))
        {
            return false;
        }
        if (formatCount == 0)
        {
            return Fail("The Vulkan Win32 surface exposes no formats.");
        }
        formats.resize(formatCount);
        if (!Check(GetSurfaceFormats(PhysicalDevice, Surface, &formatCount, formats.data()),
                "vkGetPhysicalDeviceSurfaceFormatsKHR"))
        {
            return false;
        }
        std::uint32_t modeCount = 0;
        if (!Check(GetSurfacePresentModes(PhysicalDevice, Surface, &modeCount, nullptr),
                "vkGetPhysicalDeviceSurfacePresentModesKHR(count)"))
        {
            return false;
        }
        if (modeCount == 0)
        {
            return Fail("The Vulkan Win32 surface exposes no present modes.");
        }
        modes.resize(modeCount);
        return Check(GetSurfacePresentModes(PhysicalDevice, Surface, &modeCount, modes.data()),
            "vkGetPhysicalDeviceSurfacePresentModesKHR");
    }

    bool WaitForAcquisition()
    {
        if (!AcquisitionPending) return true;
        // A frame can fail after acquisition but before queue submission. Its
        // render fence then cannot protect the acquisition semaphore or images.
        if (!Check(WaitForFences(Device, 1, &AcquireFence, VK_TRUE_VALUE, GpuWaitTimeout),
                "vkWaitForFences(acquire)")) return false;
        AcquisitionPending = false;
        return true;
    }

    bool WaitForPresentation()
    {
        if (!PresentationPending) return true;
        const bool completed = PresentFencesEnabled
            ? Check(WaitForFences(Device, 1, &PresentFence, VK_TRUE_VALUE, GpuWaitTimeout),
                "vkWaitForFences(presentation)")
            : Check(DeviceWaitIdle(Device), "vkDeviceWaitIdle(legacy presentation)");
        if (completed) PresentationPending = false;
        return completed;
    }

    void DestroySwapchainResources() noexcept
    {
        if (Pipeline != 0 && DestroyPipelineFunction != nullptr)
        {
            DestroyPipelineFunction(Device, Pipeline, nullptr);
        }
        Pipeline = 0;
        if (ViewportPipeline != 0 && DestroyPipelineFunction != nullptr)
            DestroyPipelineFunction(Device, ViewportPipeline, nullptr);
        ViewportPipeline = 0;
        if (RenderPass != 0 && DestroyRenderPassFunction != nullptr)
        {
            for (VkFramebuffer framebuffer : Framebuffers)
            {
                if (framebuffer != 0)
                {
                    DestroyFramebufferFunction(Device, framebuffer, nullptr);
                }
            }
            Framebuffers.clear();
            DestroyRenderPassFunction(Device, RenderPass, nullptr);
        }
        RenderPass = 0;
        for (VkImageView view : SwapchainViews)
        {
            if (view != 0 && DestroyImageViewFunction != nullptr)
            {
                DestroyImageViewFunction(Device, view, nullptr);
            }
        }
        SwapchainViews.clear();
        SwapchainImages.clear();
        for (VkSemaphore finished : RenderFinished)
            if (finished != 0 && DestroySemaphoreFunction != nullptr)
                DestroySemaphoreFunction(Device, finished, nullptr);
        RenderFinished.clear();
        if (Swapchain != 0 && DestroySwapchain != nullptr)
        {
            DestroySwapchain(Device, Swapchain, nullptr);
        }
        Swapchain = 0;
    }

    bool CreateRenderPassAndPipeline()
    {
        VkAttachmentDescription attachment{};
        attachment.format = SwapchainFormat;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        const VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorReference;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo renderPass{};
        renderPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPass.attachmentCount = 1;
        renderPass.pAttachments = &attachment;
        renderPass.subpassCount = 1;
        renderPass.pSubpasses = &subpass;
        renderPass.dependencyCount = 1;
        renderPass.pDependencies = &dependency;
        if (!Check(CreateRenderPassFunction(Device, &renderPass, nullptr, &RenderPass),
                "vkCreateRenderPass(editor)"))
        {
            return false;
        }

        VkShaderModule vertexShader = 0;
        VkShaderModule fragmentShader = 0;
        const Renderer::VulkanShaderProgram vertexProgram {"VulkanPresenter.glsl", Renderer::VulkanShaderStage::Vertex,
            "EVANS_VERTEX_STAGE=1", Renderer::VulkanShaderDomain::Editor};
        const Renderer::VulkanShaderProgram fragmentProgram {"VulkanPresenter.glsl", Renderer::VulkanShaderStage::Fragment,
            "", Renderer::VulkanShaderDomain::Editor};
        if (!Renderer::CreateVulkanShaderModule(Device, CreateShaderModuleFunction, vertexProgram, vertexShader, LastError))
            return false;
        if (!Renderer::CreateVulkanShaderModule(Device, CreateShaderModuleFunction, fragmentProgram, fragmentShader, LastError))
        {
            DestroyShaderModuleFunction(Device, vertexShader, nullptr);
            return false;
        }

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertexShader;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragmentShader;
        stages[1].pName = "main";
        const VkVertexInputBindingDescription binding{
            0, static_cast<std::uint32_t>(sizeof(ImguiRenderVulkan::Vertex)),
            VK_VERTEX_INPUT_RATE_VERTEX};
        const std::array<VkVertexInputAttributeDescription, 3> attributes = {{
            {0, 0, VK_FORMAT_R32G32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(ImguiRenderVulkan::Vertex, Position))},
            {1, 0, VK_FORMAT_R32G32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(ImguiRenderVulkan::Vertex, TexCoord))},
            {2, 0, VK_FORMAT_R8G8B8A8_UNORM,
                static_cast<std::uint32_t>(offsetof(ImguiRenderVulkan::Vertex, Color))}}};
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
        depth.minDepthBounds = 0.0f;
        depth.maxDepthBounds = 1.0f;
        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.blendEnable = VK_TRUE_VALUE;
        blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.attachmentCount = 1;
        blend.pAttachments = &blendAttachment;
        const std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkGraphicsPipelineCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        create.stageCount = static_cast<std::uint32_t>(stages.size());
        create.pStages = stages.data();
        create.pVertexInputState = &vertexInput;
        create.pInputAssemblyState = &inputAssembly;
        create.pViewportState = &viewport;
        create.pRasterizationState = &raster;
        create.pMultisampleState = &multisample;
        create.pDepthStencilState = &depth;
        create.pColorBlendState = &blend;
        create.pDynamicState = &dynamic;
        create.layout = PipelineLayout;
        create.renderPass = RenderPass;
        create.subpass = 0;
        VkResult result = CreateGraphicsPipelines(
            Device, 0, 1, &create, nullptr, &Pipeline);
        if (result == VK_SUCCESS)
        {
            // The viewport is a completed color framebuffer, not a UI overlay.
            // Preserve its RGB even when material cutouts left fractional alpha.
            blendAttachment.blendEnable = VK_FALSE_VALUE;
            result = Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0, 1, &create, nullptr, &ViewportPipeline);
        }
        DestroyShaderModuleFunction(Device, fragmentShader, nullptr);
        DestroyShaderModuleFunction(Device, vertexShader, nullptr);
        return Check(result, "vkCreateGraphicsPipelines(editor)");
    }

    bool CreateSwapchainResources(int requestedWidth, int requestedHeight)
    {
        if (requestedWidth <= 0 || requestedHeight <= 0)
        {
            return Fail("The Vulkan presenter cannot create a zero-sized swapchain.");
        }
        if (Device != nullptr)
        {
            if (FrameFence != 0 &&
                !Check(WaitForFences(
                    Device, 1, &FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
                    "vkWaitForFences(swapchain resize)"))
            {
                return false;
            }
        }
        RetireGpuImports();
        if (!WaitForAcquisition() || !WaitForPresentation()) return false;
        DestroySwapchainResources();

        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> modes;
        if (!QuerySurface(capabilities, formats, modes))
        {
            return false;
        }
        const auto preferredFormat = std::find_if(formats.begin(), formats.end(),
            [](const VkSurfaceFormatKHR& format)
            {
                return format.format == VK_FORMAT_R8G8B8A8_UNORM &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            });
        const auto bgraFormat = std::find_if(formats.begin(), formats.end(),
            [](const VkSurfaceFormatKHR& format)
            {
                return format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            });
        const VkSurfaceFormatKHR selectedFormat = preferredFormat != formats.end()
            ? *preferredFormat
            : (bgraFormat != formats.end() ? *bgraFormat : formats.front());
        if (selectedFormat.format != VK_FORMAT_R8G8B8A8_UNORM &&
            selectedFormat.format != VK_FORMAT_B8G8R8A8_UNORM)
        {
            return Fail("The Vulkan surface has no RGBA8 or BGRA8 swapchain format.");
        }
        SwapchainFormat = selectedFormat.format;
        const bool fixedExtent = capabilities.currentExtent.width !=
            std::numeric_limits<std::uint32_t>::max();
        SwapchainExtent = fixedExtent ? capabilities.currentExtent : VkExtent2D{
            std::clamp(static_cast<std::uint32_t>(requestedWidth),
                capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(static_cast<std::uint32_t>(requestedHeight),
                capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
        if (SwapchainExtent.width == 0 || SwapchainExtent.height == 0 ||
            SwapchainExtent.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            SwapchainExtent.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        {
            return Fail("The Vulkan surface returned an invalid swapchain extent.");
        }
        Width = static_cast<int>(SwapchainExtent.width);
        Height = static_cast<int>(SwapchainExtent.height);
        if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0U)
        {
            return Fail("The Vulkan surface must support color attachment usage.");
        }
        std::uint32_t imageCount = std::max(2U, capabilities.minImageCount + 1U);
        if (capabilities.maxImageCount > 0U)
        {
            imageCount = std::min(imageCount, capabilities.maxImageCount);
        }
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end())
        {
            presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
        else if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end())
        {
            presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        }
        VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> alphaModes = {
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
        for (VkCompositeAlphaFlagBitsKHR candidate : alphaModes)
        {
            if ((capabilities.supportedCompositeAlpha & candidate) != 0U)
            {
                compositeAlpha = candidate;
                break;
            }
        }
        VkSwapchainCreateInfoKHR create{};
        create.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create.surface = Surface;
        create.minImageCount = imageCount;
        create.imageFormat = SwapchainFormat;
        create.imageColorSpace = selectedFormat.colorSpace;
        create.imageExtent = SwapchainExtent;
        create.imageArrayLayers = 1;
        create.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        create.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create.preTransform = (capabilities.supportedTransforms &
            VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) != 0U
            ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
            : capabilities.currentTransform;
        create.compositeAlpha = compositeAlpha;
        create.presentMode = presentMode;
        create.clipped = VK_TRUE_VALUE;
        if (!Check(CreateSwapchain(Device, &create, nullptr, &Swapchain),
                "vkCreateSwapchainKHR"))
        {
            return false;
        }
        std::uint32_t swapchainImageCount = 0;
        if (!Check(GetSwapchainImages(Device, Swapchain, &swapchainImageCount, nullptr),
                "vkGetSwapchainImagesKHR(count)"))
        {
            return false;
        }
        if (swapchainImageCount == 0)
        {
            return Fail("The Vulkan swapchain returned no presentable images.");
        }
        SwapchainImages.resize(swapchainImageCount);
        if (!Check(GetSwapchainImages(Device, Swapchain, &swapchainImageCount,
                SwapchainImages.data()), "vkGetSwapchainImagesKHR"))
        {
            return false;
        }
        RenderFinished.resize(swapchainImageCount);
        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (VkSemaphore& finished : RenderFinished)
            if (!Check(CreateSemaphoreFunction(Device, &semaphore, nullptr, &finished),
                    "vkCreateSemaphore(present image)")) return false;
        SwapchainViews.resize(swapchainImageCount);
        for (std::uint32_t index = 0; index < swapchainImageCount; ++index)
        {
            VkImageViewCreateInfo view{};
            view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = SwapchainImages[index];
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = SwapchainFormat;
            view.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (!Check(CreateImageViewFunction(Device, &view, nullptr, &SwapchainViews[index]),
                    "vkCreateImageView(swapchain)"))
            {
                return false;
            }
        }
        if (!CreateRenderPassAndPipeline())
        {
            return false;
        }
        Framebuffers.resize(swapchainImageCount);
        for (std::uint32_t index = 0; index < swapchainImageCount; ++index)
        {
            VkFramebufferCreateInfo framebuffer{};
            framebuffer.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer.renderPass = RenderPass;
            framebuffer.attachmentCount = 1;
            framebuffer.pAttachments = &SwapchainViews[index];
            framebuffer.width = SwapchainExtent.width;
            framebuffer.height = SwapchainExtent.height;
            framebuffer.layers = 1;
            if (!Check(CreateFramebufferFunction(Device, &framebuffer, nullptr,
                    &Framebuffers[index]), "vkCreateFramebuffer(swapchain)"))
            {
                return false;
            }
        }
        return true;
    }

    bool Initialize(HWND window, int width, int height)
    {
        Shutdown();
        if (window == nullptr || width <= 0 || height <= 0)
        {
            return Fail("The Vulkan presenter requires a valid Win32 window and dimensions.");
        }
        Window = window;
        if (!LoadRuntime() || !CreateInstanceAndSurface(window) || !SelectPhysicalDevice() ||
            !CreateDeviceAndQueue() || !CreateCommandState() ||
            !ImguiRenderer.Initialize() || !CreateDescriptorState() ||
            !CreateStaticTextures() || !CreateSwapchainResources(width, height))
        {
            const std::string failure = LastError.empty()
                ? "The Vulkan editor presenter failed to initialize." : LastError;
            Shutdown();
            LastError = failure;
            return false;
        }
        Initialized = true;
        LastError.clear();
        return true;
    }

    std::vector<Texture> GpuImports;
    std::vector<VkSemaphore> GpuReadySemaphores;
    std::vector<std::shared_ptr<const Renderer::GpuImage>> GpuLeases;
    void RetireGpuImports()
    {
        for(auto& texture:GpuImports)DestroyTexture(texture);
        for(auto semaphore:GpuReadySemaphores)DestroySemaphoreFunction(Device,semaphore,nullptr);
        GpuReadySemaphores.clear();GpuImports.clear();GpuLeases.clear();
    }
    bool ImportViewport(const Image& image)
    {
        if(image.GpuColor->Backend!=Renderer::RHIBackend::Vulkan)return Fail("Incompatible GPU viewport backend");
        const auto& gpu=*image.GpuColor;Texture source;
        ExternalImageCreateInfo external;
        VkImageCreateInfo create{};create.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;create.pNext=&external;
        create.imageType=VK_IMAGE_TYPE_2D;create.format=VkFormat(gpu.Format);create.extent={image.Width,image.Height,1};
        create.mipLevels=1;create.arrayLayers=1;create.samples=VK_SAMPLE_COUNT_1_BIT;create.tiling=VK_IMAGE_TILING_OPTIMAL;
        create.usage=VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;create.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        if(!Check(CreateImageFunction(Device,&create,nullptr,&source.Handle),"Create imported Vulkan viewport"))return false;
        DedicatedImageMemoryInfo dedicated;dedicated.image=source.Handle;
        ImportWin32MemoryInfo imported;imported.handle=gpu.SharedHandle;imported.pNext=&dedicated;
        VkMemoryAllocateInfo allocation{};allocation.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;allocation.pNext=&imported;
        allocation.allocationSize=gpu.AllocationBytes;allocation.memoryTypeIndex=gpu.MemoryType;
        if(!Check(AllocateMemory(Device,&allocation,nullptr,&source.Memory),"Import Vulkan viewport memory") ||
            !Check(BindImageMemory(Device,source.Handle,source.Memory,0),"Bind imported viewport")){DestroyTexture(source);return false;}
        VkImageMemoryBarrier acquire{};acquire.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acquire.srcQueueFamilyIndex=ExternalQueueFamily;acquire.dstQueueFamilyIndex=QueueFamily;
        acquire.oldLayout=VK_IMAGE_LAYOUT_GENERAL;acquire.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        acquire.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;acquire.image=source.Handle;acquire.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&acquire);
        TransitionTexture(ViewportTexture,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy copy{};copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.extent={image.Width,image.Height,1};
        CmdCopyImage(CommandBuffer,source.Handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,ViewportTexture.Handle,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
        TransitionTexture(ViewportTexture,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        auto release=acquire;release.srcQueueFamilyIndex=QueueFamily;release.dstQueueFamilyIndex=ExternalQueueFamily;
        release.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;release.newLayout=VK_IMAGE_LAYOUT_GENERAL;
        release.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT;release.dstAccessMask=0;
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,nullptr,0,nullptr,1,&release);
        TimelineSemaphoreType type;VkSemaphore semaphore=0;VkSemaphoreCreateInfo desc{};desc.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;desc.pNext=&type;
        if(!Check(CreateSemaphoreFunction(Device,&desc,nullptr,&semaphore),"Create viewport wait semaphore")){DestroyTexture(source);return false;}
        const auto importSemaphore=reinterpret_cast<ImportSemaphoreWin32Handle>(GetDeviceProcAddress(Device,"vkImportSemaphoreWin32HandleKHR"));
        ImportWin32SemaphoreInfo importedReady;importedReady.semaphore=semaphore;importedReady.handle=gpu.ReadyHandle;
        if(!importSemaphore || !Check(importSemaphore(Device,&importedReady),"Import GPU viewport readiness")){DestroySemaphoreFunction(Device,semaphore,nullptr);DestroyTexture(source);return false;}
        GpuReadySemaphores.push_back(semaphore);
        GpuImports.push_back(source);GpuLeases.push_back(image.GpuColor);return true;
    }

    bool EnsureViewportTexture(const Image& image, std::uint64_t revision)
    {
        if (!image.IsValid())
        {
            ViewportVisible = false;
            return true;
        }
        if (ViewportTexture.Width != image.Width || ViewportTexture.Height != image.Height)
        {
            if (!CreateTexture(image.Width, image.Height, ViewportTexture))
            {
                return false;
            }
            UpdateDescriptor(ViewportDescriptor, ViewportTexture.View);
            UploadedRevision = InvalidRevision;
        }
        if (revision != UploadedRevision)
        {
            if (!(image.GpuColor ? ImportViewport(image) : UploadTexture(ViewportTexture, image.PixelsRGBA8.data(),
                    image.PixelsRGBA8.size())))
            {
                return false;
            }
            UploadedRevision = revision;
        }
        ViewportVisible = true;
        return true;
    }

    bool BeginFrame(
        const Image& image,
        std::uint64_t revision,
        int viewportX,
        int viewportY,
        int viewportWidth,
        int viewportHeight)
    {
        if (!Initialized || Swapchain == 0)
        {
            return Fail("The Vulkan presenter is not initialized.");
        }
        if (FrameActive)
        {
            return Fail("A Vulkan editor frame is already active.");
        }
        if (RecreateSwapchain)
        {
            if (!CreateSwapchainResources(Width, Height))
            {
                return false;
            }
            RecreateSwapchain = false;
        }
        if (!Check(WaitForFences(Device, 1, &FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
                "vkWaitForFences(frame)"))
        {
            return false;
        }
        RetireGpuImports();
        if (!WaitForAcquisition() ||
            !Check(ResetFences(Device, 1, &AcquireFence), "vkResetFences(acquire)")) return false;
        VkResult acquire = AcquireNextImage(
            Device, Swapchain, GpuWaitTimeout, ImageAvailable, AcquireFence, &AcquiredImage);
        if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
        {
            if (!CreateSwapchainResources(Width, Height))
            {
                return false;
            }
            acquire = AcquireNextImage(
                Device, Swapchain, GpuWaitTimeout, ImageAvailable, AcquireFence, &AcquiredImage);
        }
        if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
        {
            return Fail(VulkanError("vkAcquireNextImageKHR", acquire));
        }
        AcquisitionPending = true;
        if (AcquiredImage >= SwapchainImages.size() || AcquiredImage >= Framebuffers.size())
        {
            return Fail("vkAcquireNextImageKHR returned an invalid swapchain image index.");
        }
        RecreateSwapchain = acquire == VK_SUBOPTIMAL_KHR;
        if (!Check(ResetCommandPool(Device, CommandPool, 0), "vkResetCommandPool(frame)"))
        {
            return false;
        }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!Check(BeginCommandBuffer(CommandBuffer, &begin), "vkBeginCommandBuffer(frame)"))
        {
            return false;
        }
        if (!EnsureViewportTexture(image, revision))
        {
            return false;
        }
        ViewportX = viewportX;
        ViewportY = viewportY;
        ViewportWidth = viewportWidth;
        ViewportHeight = viewportHeight;
        ImguiBuilt = false;
        FrameActive = true;
        return true;
    }

    void DrawImgui()
    {
        if (!FrameActive)
        {
            return;
        }
        ImguiRenderer.Build(Width, Height);
        ImguiBuilt = true;
    }

    void SetScissorClipped(int x, int y, int width, int height)
    {
        const int left = std::clamp(x, 0, Width);
        const int top = std::clamp(y, 0, Height);
        const int right = std::clamp(x + width, left, Width);
        const int bottom = std::clamp(y + height, top, Height);
        const VkRect2D scissor{{left, top}, {
            static_cast<std::uint32_t>(right - left),
            static_cast<std::uint32_t>(bottom - top)}};
        CmdSetScissor(CommandBuffer, 0, 1, &scissor);
    }

    void DrawBatch(VkDescriptorSet descriptor, std::uint32_t firstVertex,
        std::uint32_t vertexCount, int x, int y, int width, int height)
    {
        if (vertexCount == 0 || width <= 0 || height <= 0)
        {
            return;
        }
        SetScissorClipped(x, y, width, height);
        CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            PipelineLayout, 0, 1, &descriptor, 0, nullptr);
        CmdDraw(CommandBuffer, vertexCount, 1, firstVertex, 0);
    }

    void TransitionSwapchainForPresent()
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = SwapchainImages[AcquiredImage];
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        CmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    bool FinishFrameCommands()
    {
        if (!FrameActive)
        {
            return Fail("No Vulkan editor frame is active.");
        }
        std::vector<ImguiRenderVulkan::Vertex> vertices;
        bool drawViewport = ViewportVisible && ViewportWidth > 0 && ViewportHeight > 0;
        if (drawViewport)
        {
            const float left = static_cast<float>(ViewportX);
            const float top = static_cast<float>(ViewportY);
            const float right = static_cast<float>(ViewportX + ViewportWidth);
            const float bottom = static_cast<float>(ViewportY + ViewportHeight);
            constexpr std::uint32_t white = 0xffffffffU;
            const std::array<ImguiRenderVulkan::Vertex, 6> viewportVertices = {{
                {{left, top}, {0.0f, 0.0f}, white},
                {{right, top}, {1.0f, 0.0f}, white},
                {{right, bottom}, {1.0f, 1.0f}, white},
                {{left, top}, {0.0f, 0.0f}, white},
                {{right, bottom}, {1.0f, 1.0f}, white},
                {{left, bottom}, {0.0f, 1.0f}, white}}};
            vertices.insert(vertices.end(), viewportVertices.begin(), viewportVertices.end());
        }
        const std::uint32_t imguiBase = static_cast<std::uint32_t>(vertices.size());
        if (ImguiBuilt)
        {
            const auto& imguiVertices = ImguiRenderer.Vertices();
            vertices.insert(vertices.end(), imguiVertices.begin(), imguiVertices.end());
        }
        const VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(vertices.size()) *
            sizeof(ImguiRenderVulkan::Vertex);
        if (!vertices.empty())
        {
            if (!EnsureBuffer(VertexBuffer, vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                return false;
            }
            void* mapped = nullptr;
            if (!Check(MapMemory(Device, VertexBuffer.Memory, 0, vertexBytes, 0, &mapped),
                    "vkMapMemory(editor vertices)"))
            {
                return false;
            }
            std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(vertexBytes));
            UnmapMemory(Device, VertexBuffer.Memory);
        }

        VkClearValue clear{};
        clear.color.float32[0] = 0.035f;
        clear.color.float32[1] = 0.043f;
        clear.color.float32[2] = 0.059f;
        clear.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo pass{};
        pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        pass.renderPass = RenderPass;
        pass.framebuffer = Framebuffers[AcquiredImage];
        pass.renderArea = {{0, 0}, SwapchainExtent};
        pass.clearValueCount = 1;
        pass.pClearValues = &clear;
        CmdBeginRenderPass(CommandBuffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0.0f, 0.0f,
            static_cast<float>(Width), static_cast<float>(Height), 0.0f, 1.0f};
        CmdSetViewport(CommandBuffer, 0, 1, &viewport);
        CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ViewportPipeline);
        // A positive Vulkan viewport maps NDC -1 to the top row, so top-left
        // pixel coordinates use a positive Y scale unlike the D3D presenters.
        const PushConstants constants{{
            2.0f / static_cast<float>(Width),
            2.0f / static_cast<float>(Height)}, {-1.0f, -1.0f}};
        CmdPushConstants(CommandBuffer, PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
            0, static_cast<std::uint32_t>(sizeof(constants)), &constants);
        if (!vertices.empty())
        {
            constexpr VkDeviceSize offset = 0;
            CmdBindVertexBuffers(CommandBuffer, 0, 1, &VertexBuffer.Handle, &offset);
        }
        if (drawViewport)
        {
            DrawBatch(ViewportDescriptor, 0, 6,
                ViewportX, ViewportY, ViewportWidth, ViewportHeight);
        }
        if (ImguiBuilt)
        {
            CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
            for (const ImguiRenderVulkan::Batch& batch : ImguiRenderer.Batches())
            {
                const VkDescriptorSet descriptor =
                    batch.TextureSlot == ImguiRenderVulkan::Texture::Font
                    ? FontDescriptor : WhiteDescriptor;
                DrawBatch(descriptor, imguiBase + batch.FirstVertex, batch.VertexCount,
                    batch.ScissorX, batch.ScissorY,
                    batch.ScissorWidth, batch.ScissorHeight);
            }
        }
        CmdEndRenderPass(CommandBuffer);
        TransitionSwapchainForPresent();
        if (!Check(EndCommandBuffer(CommandBuffer), "vkEndCommandBuffer(frame)"))
        {
            return false;
        }
        constexpr VkFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        std::vector<VkSemaphore> waits{ImageAvailable};waits.insert(waits.end(),GpuReadySemaphores.begin(),GpuReadySemaphores.end());
        std::vector<VkFlags> stages(waits.size(),VK_PIPELINE_STAGE_TRANSFER_BIT);stages[0]=waitStage;
        submit.waitSemaphoreCount = unsigned(waits.size());
        submit.pWaitSemaphores = waits.data();
        submit.pWaitDstStageMask = stages.data();
        std::vector<std::uint64_t> waitValues(waits.size(),1);waitValues[0]=0;const std::uint64_t signalValue=0;
        TimelineSemaphoreSubmit timeline;timeline.waitValueCount=unsigned(waitValues.size());timeline.waitValues=waitValues.data();
        timeline.signalValueCount=1;timeline.signalValues=&signalValue;if(!GpuReadySemaphores.empty())submit.pNext=&timeline;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &CommandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &RenderFinished[AcquiredImage];
        if (!Check(ResetFences(Device, 1, &FrameFence), "vkResetFences(frame)"))
        {
            FrameActive = false;
            return false;
        }
        const VkResult submitResult = QueueSubmit(Queue, 1, &submit, FrameFence);
        if (submitResult != VK_SUCCESS)
        {
            const std::string submitError = VulkanError("vkQueueSubmit(frame)", submitResult);
            (void)RecreateSignaledFrameFence();
            FrameActive = false;
            return Fail(submitError);
        }
        FrameActive = false;
        FrameSubmitted = true;
        return true;
    }

    bool PresentSubmittedFrame()
    {
        if (!FrameSubmitted)
        {
            return Fail("No submitted Vulkan editor frame is available to present.");
        }
        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &RenderFinished[AcquiredImage];
        present.swapchainCount = 1;
        present.pSwapchains = &Swapchain;
        present.pImageIndices = &AcquiredImage;
        VkSwapchainPresentFenceInfoEXT fenceInfo{};
        if (PresentFencesEnabled)
        {
            if (!WaitForPresentation() ||
                !Check(ResetFences(Device, 1, &PresentFence), "vkResetFences(presentation)"))
                return false;
            fenceInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT;
            fenceInfo.swapchainCount = 1;
            fenceInfo.pFences = &PresentFence;
            present.pNext = &fenceInfo;
        }
        const VkResult result = QueuePresent(Queue, &present);
        PresentationPending = result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR ||
            result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR;
        FrameSubmitted = false;
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            RecreateSwapchain = true;
            return true;
        }
        return Check(result, "vkQueuePresentKHR");
    }

    bool EndFrame()
    {
        return FinishFrameCommands() && PresentSubmittedFrame();
    }

    bool Resize(int width, int height)
    {
        if (!Initialized)
        {
            return Fail("The Vulkan presenter is not initialized.");
        }
        if (width <= 0 || height <= 0)
        {
            return true;
        }
        if (width == Width && height == Height)
        {
            return true;
        }
        return CreateSwapchainResources(width, height);
    }

    void Shutdown() noexcept
    {
        if (Abandoned)
        {
            return;
        }
        if (Device != nullptr && FrameFence != 0 && WaitForFences != nullptr)
        {
            const VkResult waitResult =
                WaitForFences(Device, 1, &FrameFence, VK_TRUE_VALUE, GpuWaitTimeout);
            if (waitResult != VK_SUCCESS)
            {
                LastError = VulkanError("vkWaitForFences(shutdown)", waitResult);
                Initialized = false;
                FrameActive = false;
                FrameSubmitted = false;
                Abandoned = true;
                return;
            }
        }
        if (Device != nullptr && (!WaitForAcquisition() || !WaitForPresentation()))
        {
            Initialized = false;
            Abandoned = true;
            return;
        }
        FrameActive = false;
        FrameSubmitted = false;
        DestroySwapchainResources();
        RetireGpuImports();
        DestroyTexture(ViewportTexture);
        DestroyTexture(FontTexture);
        DestroyTexture(WhiteTexture);
        DestroyBuffer(VertexBuffer);
        DestroyBuffer(UploadBuffer);
        if (DescriptorPool != 0 && DestroyDescriptorPoolFunction != nullptr)
        {
            DestroyDescriptorPoolFunction(Device, DescriptorPool, nullptr);
        }
        if (Sampler != 0 && DestroySamplerFunction != nullptr)
        {
            DestroySamplerFunction(Device, Sampler, nullptr);
        }
        if (PipelineLayout != 0 && DestroyPipelineLayoutFunction != nullptr)
        {
            DestroyPipelineLayoutFunction(Device, PipelineLayout, nullptr);
        }
        if (DescriptorSetLayout != 0 && DestroyDescriptorSetLayoutFunction != nullptr)
        {
            DestroyDescriptorSetLayoutFunction(Device, DescriptorSetLayout, nullptr);
        }
        if (ImageAvailable != 0 && DestroySemaphoreFunction != nullptr)
        {
            DestroySemaphoreFunction(Device, ImageAvailable, nullptr);
        }
        if (PresentFence != 0 && DestroyFenceFunction != nullptr)
        {
            DestroyFenceFunction(Device, PresentFence, nullptr);
        }
        if (AcquireFence != 0 && DestroyFenceFunction != nullptr)
            DestroyFenceFunction(Device, AcquireFence, nullptr);
        if (FrameFence != 0 && DestroyFenceFunction != nullptr)
        {
            DestroyFenceFunction(Device, FrameFence, nullptr);
        }
        if (CommandPool != 0 && DestroyCommandPoolFunction != nullptr)
        {
            DestroyCommandPoolFunction(Device, CommandPool, nullptr);
        }
        if (Device != nullptr && DestroyDeviceFunction != nullptr)
        {
            Renderer::DestroyVulkanPipelineCache(Device);
            DestroyDeviceFunction(Device, nullptr);
        }
        if (Surface != 0 && DestroySurface != nullptr)
        {
            DestroySurface(Instance, Surface, nullptr);
        }
        if (Instance != nullptr && DestroyInstanceFunction != nullptr)
        {
            DestroyInstanceFunction(Instance, nullptr);
        }
        if (VulkanModule != nullptr)
        {
            FreeLibrary(VulkanModule);
        }

        Window = nullptr;
        VulkanModule = nullptr;
        Instance = nullptr;
        PhysicalDevice = nullptr;
        Device = nullptr;
        Queue = nullptr;
        Surface = 0;
        Swapchain = 0;
        CommandPool = 0;
        CommandBuffer = nullptr;
        FrameFence = 0;
        ImageAvailable = 0;
        PresentFence = 0;
        AcquireFence = 0;
        AcquisitionPending = false;
        PresentationPending = false;
        PresentFencesEnabled = false;
        MaintenanceInstanceEnabled = false;
        DescriptorSetLayout = 0;
        DescriptorPool = 0;
        PipelineLayout = 0;
        Sampler = 0;
        WhiteDescriptor = 0;
        FontDescriptor = 0;
        ViewportDescriptor = 0;
        Width = 0;
        Height = 0;
        UploadedRevision = InvalidRevision;
        Initialized = false;
        Abandoned = false;
        RecreateSwapchain = false;
    }

    HWND Window = nullptr;
    HMODULE VulkanModule = nullptr;
    VkInstance Instance = nullptr;
    VkPhysicalDevice PhysicalDevice = nullptr;
    VkDevice Device = nullptr;
    VkQueue Queue = nullptr;
    VkSurfaceKHR Surface = 0;
    VkSwapchainKHR Swapchain = 0;
    VkFormat SwapchainFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkExtent2D SwapchainExtent{};
    std::vector<VkImage> SwapchainImages;
    std::vector<VkImageView> SwapchainViews;
    std::vector<VkFramebuffer> Framebuffers;
    VkRenderPass RenderPass = 0;
    VkPipeline Pipeline = 0;
    VkPipeline ViewportPipeline = 0;
    VkFence AcquireFence = 0;
    bool AcquisitionPending = false;
    VkCommandPool CommandPool = 0;
    VkCommandBuffer CommandBuffer = nullptr;
    VkFence FrameFence = 0;
    VkSemaphore ImageAvailable = 0;
    std::vector<VkSemaphore> RenderFinished;
    VkFence PresentFence = 0;
    bool PresentationPending = false;
    bool PresentFencesEnabled = false;
    bool MaintenanceInstanceEnabled = false;
    VkDescriptorSetLayout DescriptorSetLayout = 0;
    VkDescriptorPool DescriptorPool = 0;
    VkPipelineLayout PipelineLayout = 0;
    VkSampler Sampler = 0;
    VkDescriptorSet WhiteDescriptor = 0;
    VkDescriptorSet FontDescriptor = 0;
    VkDescriptorSet ViewportDescriptor = 0;
    VkPhysicalDeviceMemoryProperties MemoryProperties{};
    std::uint32_t QueueFamily = 0;
    std::uint32_t AcquiredImage = 0;
    Buffer UploadBuffer;
    Buffer VertexBuffer;
    Texture WhiteTexture;
    Texture FontTexture;
    Texture ViewportTexture;
    ImguiRenderVulkan ImguiRenderer;
    std::uint64_t UploadedRevision = InvalidRevision;
    int Width = 0;
    int Height = 0;
    int ViewportX = 0;
    int ViewportY = 0;
    int ViewportWidth = 0;
    int ViewportHeight = 0;
    bool Initialized = false;
    bool Abandoned = false;
    bool FrameActive = false;
    bool FrameSubmitted = false;
    bool ViewportVisible = false;
    bool ImguiBuilt = false;
    bool RecreateSwapchain = false;
    std::string LastError;

    PFN_vkGetInstanceProcAddr GetInstanceProcAddress = nullptr;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddress = nullptr;
    PFN_vkCreateInstance CreateInstanceFunction = nullptr;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensions = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensions = nullptr;
    PFN_vkDestroyInstance DestroyInstanceFunction = nullptr;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkCreateDevice CreateDeviceFunction = nullptr;
    PFN_vkDestroyDevice DestroyDeviceFunction = nullptr;
    PFN_vkGetDeviceQueue GetDeviceQueue = nullptr;
    PFN_vkDeviceWaitIdle DeviceWaitIdle = nullptr;
    PFN_vkCreateWin32SurfaceKHR CreateWin32Surface = nullptr;
    PFN_vkDestroySurfaceKHR DestroySurface = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR GetSurfaceSupport = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR GetSurfaceCapabilities = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR GetSurfaceFormats = nullptr;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR GetSurfacePresentModes = nullptr;
    PFN_vkCreateSwapchainKHR CreateSwapchain = nullptr;
    PFN_vkDestroySwapchainKHR DestroySwapchain = nullptr;
    PFN_vkGetSwapchainImagesKHR GetSwapchainImages = nullptr;
    PFN_vkAcquireNextImageKHR AcquireNextImage = nullptr;
    PFN_vkQueuePresentKHR QueuePresent = nullptr;
    PFN_vkCreateSemaphore CreateSemaphoreFunction = nullptr;
    PFN_vkDestroySemaphore DestroySemaphoreFunction = nullptr;
    PFN_vkCreateFence CreateFenceFunction = nullptr;
    PFN_vkDestroyFence DestroyFenceFunction = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkResetFences ResetFences = nullptr;
    PFN_vkCreateCommandPool CreateCommandPoolFunction = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPoolFunction = nullptr;
    PFN_vkResetCommandPool ResetCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkCreateBuffer CreateBufferFunction = nullptr;
    PFN_vkDestroyBuffer DestroyBufferFunction = nullptr;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkBindBufferMemory BindBufferMemory = nullptr;
    PFN_vkMapMemory MapMemory = nullptr;
    PFN_vkUnmapMemory UnmapMemory = nullptr;
    PFN_vkCreateImage CreateImageFunction = nullptr;
    PFN_vkDestroyImage DestroyImageFunction = nullptr;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements = nullptr;
    PFN_vkBindImageMemory BindImageMemory = nullptr;
    PFN_vkCreateImageView CreateImageViewFunction = nullptr;
    PFN_vkDestroyImageView DestroyImageViewFunction = nullptr;
    PFN_vkCreateSampler CreateSamplerFunction = nullptr;
    PFN_vkDestroySampler DestroySamplerFunction = nullptr;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayoutFunction = nullptr;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayoutFunction = nullptr;
    PFN_vkCreateDescriptorPool CreateDescriptorPoolFunction = nullptr;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPoolFunction = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;
    PFN_vkCreateRenderPass CreateRenderPassFunction = nullptr;
    PFN_vkDestroyRenderPass DestroyRenderPassFunction = nullptr;
    PFN_vkCreateFramebuffer CreateFramebufferFunction = nullptr;
    PFN_vkDestroyFramebuffer DestroyFramebufferFunction = nullptr;
    PFN_vkCreateShaderModule CreateShaderModuleFunction = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModuleFunction = nullptr;
    PFN_vkCreatePipelineLayout CreatePipelineLayoutFunction = nullptr;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayoutFunction = nullptr;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines = nullptr;
    PFN_vkDestroyPipeline DestroyPipelineFunction = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyImage CmdCopyImage = nullptr;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage = nullptr;
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass = nullptr;
    PFN_vkCmdEndRenderPass CmdEndRenderPass = nullptr;
    PFN_vkCmdSetViewport CmdSetViewport = nullptr;
    PFN_vkCmdSetScissor CmdSetScissor = nullptr;
    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers = nullptr;
    PFN_vkCmdDraw CmdDraw = nullptr;
    PFN_vkCmdPushConstants CmdPushConstants = nullptr;
};

EditorPresenterVulkan::EditorPresenterVulkan()
    : Impl_(std::make_unique<Impl>())
{
}

EditorPresenterVulkan::~EditorPresenterVulkan() = default;

bool EditorPresenterVulkan::Initialize(void* nativeWindow, int width, int height)
{
    return Impl_->Initialize(static_cast<HWND>(nativeWindow), width, height);
}

void EditorPresenterVulkan::Shutdown() noexcept
{
    Impl_->Shutdown();
}

bool EditorPresenterVulkan::Resize(int width, int height)
{
    return Impl_->Resize(width, height);
}

bool EditorPresenterVulkan::BeginFrame(
    const Image& viewportImage,
    std::uint64_t viewportRevision,
    int viewportX,
    int viewportY,
    int viewportWidth,
    int viewportHeight)
{
    return Impl_->BeginFrame(viewportImage, viewportRevision, viewportX, viewportY,
        viewportWidth, viewportHeight);
}

void EditorPresenterVulkan::DrawImgui()
{
    Impl_->DrawImgui();
}

bool EditorPresenterVulkan::EndFrame()
{
    return Impl_->EndFrame();
}

bool EditorPresenterVulkan::IsInitialized() const noexcept
{
    return Impl_->Initialized;
}

const std::string& EditorPresenterVulkan::LastError() const noexcept
{
    return Impl_->LastError;
}

} // namespace Evans::Editor

#endif
