#include "Renderer/VulkanRHI.h"

#include "Renderer/VulkanABI.h"
#include "Renderer/VulkanFragmentShader.h"
#include "Renderer/VulkanOutlineFragmentShader.h"
#include "Renderer/VulkanShadowFragmentShader.h"
#include "Renderer/VulkanVertexShader.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/RealtimeShader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace Evans::Renderer
{
#if defined(_WIN32)
namespace
{
using namespace VulkanABI;

constexpr std::size_t UniformBufferSize = 16U * 1024U * 1024U;
constexpr std::size_t UniformAlignment = 256U;
constexpr std::uint32_t MaximumFrameDescriptorSets = 8192U;
constexpr std::uint64_t GpuWaitTimeout = 5'000'000'000ULL;
constexpr std::uint32_t FillModeNonSolidFeatureIndex = 13U;

struct PhysicalDevicePropertiesPrefix
{
    std::uint32_t ApiVersion;
    std::uint32_t DriverVersion;
    std::uint32_t VendorId;
    std::uint32_t DeviceId;
    VkPhysicalDeviceType DeviceType;
    char DeviceName[256];
    std::uint8_t PipelineCacheUuid[16];
};

[[nodiscard]] std::size_t AlignUp(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

[[nodiscard]] std::string VulkanError(const char* operation, VkResult result)
{
    std::ostringstream stream;
    stream << operation << " failed with VkResult " << result;
    return stream.str();
}

[[nodiscard]] bool IsValidDepthCapture(const std::vector<float>& depths, bool requireGeometry)
{
    bool hasGeometry = false;
    for (float depth : depths)
    {
        if (!std::isfinite(depth) || depth < 0.0f || depth > 1.0f) return false;
        hasGeometry = hasGeometry || depth < 0.999999f;
    }
    return !depths.empty() && (!requireGeometry || hasGeometry);
}

} // namespace

using namespace VulkanABI;

struct VulkanRHI::Impl
{
    struct RawBuffer
    {
        VkBuffer Buffer = 0;
        VkDeviceMemory Memory = 0;
        void* Mapped = nullptr;
        VkDeviceSize Size = 0;
    };

    struct RawImage
    {
        VkImage Image = 0;
        VkDeviceMemory Memory = 0;
        VkImageView View = 0;
        VkImageLayout Layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkFormat Format = VK_FORMAT_R8G8B8A8_UNORM;
        VkFlags Aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        std::uint32_t Width = 0;
        std::uint32_t Height = 0;
    };

    struct BufferResource
    {
        BufferDesc Description;
        std::vector<std::uint8_t> Bytes;
        RawBuffer Allocation;
    };

    struct TextureResource
    {
        TextureDesc Description;
        RawImage Image;
        RawImage PrivateDepth;
        VkFramebuffer Framebuffer = 0;
    };

    template <typename Function>
    bool LoadGlobal(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetInstanceProcAddress(nullptr, name));
        return destination != nullptr || Fail(std::string("Vulkan global entry point is missing: ") + name);
    }

    template <typename Function>
    bool LoadInstance(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetInstanceProcAddress(Instance, name));
        return destination != nullptr || Fail(std::string("Vulkan instance entry point is missing: ") + name);
    }

    template <typename Function>
    bool LoadDevice(Function& destination, const char* name)
    {
        destination = reinterpret_cast<Function>(GetDeviceProcAddress(Device, name));
        return destination != nullptr || Fail(std::string("Vulkan device entry point is missing: ") + name);
    }

    bool Fail(std::string error)
    {
        LastError = std::move(error);
        return false;
    }

    bool Check(VkResult result, const char* operation)
    {
        return result == VK_SUCCESS || Fail(VulkanError(operation, result));
    }

    bool CheckThread(const char* operation)
    {
        if (!Initialized) return Fail("Vulkan RHI is not initialized");
        if (OwnerThread != std::this_thread::get_id())
        {
            return Fail(std::string(operation) + " must execute on the owning Vulkan RHI thread");
        }
        return true;
    }

    bool LoadRuntime()
    {
        VulkanModule = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (VulkanModule == nullptr) return Fail("Vulkan is unavailable: vulkan-1.dll was not found");
        GetInstanceProcAddress = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            GetProcAddress(VulkanModule, "vkGetInstanceProcAddr"));
        if (GetInstanceProcAddress == nullptr)
            return Fail("Vulkan loader does not export vkGetInstanceProcAddr");
        return LoadGlobal(CreateInstanceFunction, "vkCreateInstance");
    }

    bool LoadInstanceFunctions()
    {
        return LoadInstance(DestroyInstanceFunction, "vkDestroyInstance") &&
               LoadInstance(EnumeratePhysicalDevices, "vkEnumeratePhysicalDevices") &&
               LoadInstance(GetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties") &&
               LoadInstance(GetPhysicalDeviceFeatures, "vkGetPhysicalDeviceFeatures") &&
               LoadInstance(GetPhysicalDeviceQueueFamilyProperties,
                            "vkGetPhysicalDeviceQueueFamilyProperties") &&
               LoadInstance(GetPhysicalDeviceMemoryProperties,
                            "vkGetPhysicalDeviceMemoryProperties") &&
               LoadInstance(CreateDeviceFunction, "vkCreateDevice") &&
               LoadInstance(GetDeviceProcAddress, "vkGetDeviceProcAddr");
    }

    bool LoadDeviceFunctions()
    {
        return LoadDevice(DestroyDeviceFunction, "vkDestroyDevice") &&
               LoadDevice(GetDeviceQueue, "vkGetDeviceQueue") &&
               LoadDevice(DeviceWaitIdle, "vkDeviceWaitIdle") &&
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
               LoadDevice(CreateRenderPassFunction, "vkCreateRenderPass") &&
               LoadDevice(DestroyRenderPassFunction, "vkDestroyRenderPass") &&
               LoadDevice(CreateFramebufferFunction, "vkCreateFramebuffer") &&
               LoadDevice(DestroyFramebufferFunction, "vkDestroyFramebuffer") &&
               LoadDevice(CreateShaderModuleFunction, "vkCreateShaderModule") &&
               LoadDevice(DestroyShaderModuleFunction, "vkDestroyShaderModule") &&
               LoadDevice(CreateDescriptorSetLayoutFunction, "vkCreateDescriptorSetLayout") &&
               LoadDevice(DestroyDescriptorSetLayoutFunction, "vkDestroyDescriptorSetLayout") &&
               LoadDevice(CreateDescriptorPoolFunction, "vkCreateDescriptorPool") &&
               LoadDevice(DestroyDescriptorPoolFunction, "vkDestroyDescriptorPool") &&
               LoadDevice(AllocateDescriptorSets, "vkAllocateDescriptorSets") &&
               LoadDevice(UpdateDescriptorSets, "vkUpdateDescriptorSets") &&
               LoadDevice(CreatePipelineLayoutFunction, "vkCreatePipelineLayout") &&
               LoadDevice(DestroyPipelineLayoutFunction, "vkDestroyPipelineLayout") &&
               LoadDevice(CreateGraphicsPipelines, "vkCreateGraphicsPipelines") &&
               LoadDevice(DestroyPipelineFunction, "vkDestroyPipeline") &&
               LoadDevice(CreateCommandPoolFunction, "vkCreateCommandPool") &&
               LoadDevice(DestroyCommandPoolFunction, "vkDestroyCommandPool") &&
               LoadDevice(ResetCommandPool, "vkResetCommandPool") &&
               LoadDevice(AllocateCommandBuffers, "vkAllocateCommandBuffers") &&
               LoadDevice(BeginCommandBuffer, "vkBeginCommandBuffer") &&
               LoadDevice(EndCommandBuffer, "vkEndCommandBuffer") &&
               LoadDevice(CreateFenceFunction, "vkCreateFence") &&
               LoadDevice(DestroyFenceFunction, "vkDestroyFence") &&
               LoadDevice(WaitForFences, "vkWaitForFences") &&
               LoadDevice(ResetFences, "vkResetFences") &&
               LoadDevice(QueueSubmit, "vkQueueSubmit") &&
               LoadDevice(CmdPipelineBarrier, "vkCmdPipelineBarrier") &&
               LoadDevice(CmdCopyBufferToImage, "vkCmdCopyBufferToImage") &&
               LoadDevice(CmdCopyImageToBuffer, "vkCmdCopyImageToBuffer") &&
               LoadDevice(CmdBeginRenderPass, "vkCmdBeginRenderPass") &&
               LoadDevice(CmdEndRenderPass, "vkCmdEndRenderPass") &&
               LoadDevice(CmdSetViewport, "vkCmdSetViewport") &&
               LoadDevice(CmdSetScissor, "vkCmdSetScissor") &&
               LoadDevice(CmdBindPipeline, "vkCmdBindPipeline") &&
               LoadDevice(CmdBindDescriptorSets, "vkCmdBindDescriptorSets") &&
               LoadDevice(CmdBindVertexBuffers, "vkCmdBindVertexBuffers") &&
               LoadDevice(CmdBindIndexBuffer, "vkCmdBindIndexBuffer") &&
               LoadDevice(CmdDrawIndexed, "vkCmdDrawIndexed");
    }

    bool CreateInstance()
    {
        VkApplicationInfo application{};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.pApplicationName = "Evans";
        application.applicationVersion = MakeVersion(1, 0, 0);
        application.pEngineName = "Evans Renderer";
        application.engineVersion = MakeVersion(1, 0, 0);
        application.apiVersion = MakeVersion(1, 0, 0);
        VkInstanceCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create.pApplicationInfo = &application;
        return Check(CreateInstanceFunction(&create, nullptr, &Instance), "vkCreateInstance");
    }

    bool SelectPhysicalDevice()
    {
        std::uint32_t count = 0;
        if (!Check(EnumeratePhysicalDevices(Instance, &count, nullptr),
                   "vkEnumeratePhysicalDevices(count)") || count == 0)
            return count != 0 || Fail("Vulkan found no physical device");
        std::vector<VkPhysicalDevice> devices(count);
        if (!Check(EnumeratePhysicalDevices(Instance, &count, devices.data()),
                   "vkEnumeratePhysicalDevices")) return false;

        int bestScore = -1;
        for (VkPhysicalDevice candidate : devices)
        {
            VkPhysicalDeviceFeatures features{};
            GetPhysicalDeviceFeatures(candidate, &features);
            if (features.values[FillModeNonSolidFeatureIndex] == VK_FALSE_VALUE) continue;
            std::uint32_t queueCount = 0;
            GetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, nullptr);
            std::vector<VkQueueFamilyProperties> queues(queueCount);
            if (queueCount != 0)
                GetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, queues.data());
            for (std::uint32_t queueIndex = 0; queueIndex < queueCount; ++queueIndex)
            {
                if (queues[queueIndex].queueCount == 0 ||
                    (queues[queueIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) continue;
                alignas(std::max_align_t) std::array<std::byte, 4096> properties{};
                GetPhysicalDeviceProperties(candidate, properties.data());
                const auto* prefix = reinterpret_cast<const PhysicalDevicePropertiesPrefix*>(
                    properties.data());
                const int score = prefix->DeviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 100 : 10;
                if (score > bestScore)
                {
                    bestScore = score;
                    PhysicalDevice = candidate;
                    QueueFamilyIndex = queueIndex;
                    DeviceName = prefix->DeviceName;
                }
                break;
            }
        }
        if (PhysicalDevice == nullptr)
            return Fail("Vulkan found no graphics device with fillModeNonSolid support");
        GetPhysicalDeviceMemoryProperties(PhysicalDevice, &MemoryProperties);
        return true;
    }

    bool CreateLogicalDevice()
    {
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue{};
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = QueueFamilyIndex;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures features{};
        features.values[FillModeNonSolidFeatureIndex] = VK_TRUE_VALUE;
        VkDeviceCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create.queueCreateInfoCount = 1;
        create.pQueueCreateInfos = &queue;
        create.pEnabledFeatures = &features;
        if (!Check(CreateDeviceFunction(PhysicalDevice, &create, nullptr, &Device),
                   "vkCreateDevice")) return false;
        if (!LoadDeviceFunctions()) return false;
        GetDeviceQueue(Device, QueueFamilyIndex, 0, &GraphicsQueue);
        return GraphicsQueue != nullptr || Fail("Vulkan graphics queue is null");
    }

    [[nodiscard]] std::uint32_t FindMemoryType(std::uint32_t mask, VkFlags required) const
    {
        for (std::uint32_t index = 0; index < MemoryProperties.memoryTypeCount; ++index)
        {
            if ((mask & (1U << index)) != 0 &&
                (MemoryProperties.memoryTypes[index].propertyFlags & required) == required)
                return index;
        }
        return std::numeric_limits<std::uint32_t>::max();
    }

    bool CreateRawBuffer(VkDeviceSize size, VkFlags usage, VkFlags properties, RawBuffer& output)
    {
        VkBufferCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create.size = size;
        create.usage = usage;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!Check(CreateBufferFunction(Device, &create, nullptr, &output.Buffer),
                   "vkCreateBuffer")) return false;
        VkMemoryRequirements requirements{};
        GetBufferMemoryRequirements(Device, output.Buffer, &requirements);
        const std::uint32_t memoryType = FindMemoryType(requirements.memoryTypeBits, properties);
        if (memoryType == std::numeric_limits<std::uint32_t>::max())
        {
            DestroyRawBuffer(output);
            return Fail("Vulkan found no compatible buffer memory type");
        }
        VkMemoryAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memoryType;
        if (!Check(AllocateMemory(Device, &allocate, nullptr, &output.Memory),
                   "vkAllocateMemory(buffer)") ||
            !Check(BindBufferMemory(Device, output.Buffer, output.Memory, 0),
                   "vkBindBufferMemory"))
        {
            DestroyRawBuffer(output);
            return false;
        }
        output.Size = size;
        if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
            !Check(MapMemory(Device, output.Memory, 0, size, 0, &output.Mapped), "vkMapMemory"))
        {
            DestroyRawBuffer(output);
            return false;
        }
        return true;
    }

    void DestroyRawBuffer(RawBuffer& buffer)
    {
        if (buffer.Mapped != nullptr && UnmapMemory) UnmapMemory(Device, buffer.Memory);
        if (buffer.Buffer != 0 && DestroyBufferFunction)
            DestroyBufferFunction(Device, buffer.Buffer, nullptr);
        if (buffer.Memory != 0 && FreeMemory) FreeMemory(Device, buffer.Memory, nullptr);
        buffer = {};
    }

    bool CreateRawImage(std::uint32_t width, std::uint32_t height, VkFormat format,
                        VkFlags aspect, VkFlags usage, RawImage& output)
    {
        VkImageCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        create.imageType = VK_IMAGE_TYPE_2D;
        create.format = format;
        create.extent = {width, height, 1};
        create.mipLevels = 1;
        create.arrayLayers = 1;
        create.samples = VK_SAMPLE_COUNT_1_BIT;
        create.tiling = VK_IMAGE_TILING_OPTIMAL;
        create.usage = usage;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!Check(CreateImageFunction(Device, &create, nullptr, &output.Image),
                   "vkCreateImage")) return false;
        VkMemoryRequirements requirements{};
        GetImageMemoryRequirements(Device, output.Image, &requirements);
        const std::uint32_t memoryType = FindMemoryType(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memoryType == std::numeric_limits<std::uint32_t>::max())
        {
            DestroyRawImage(output);
            return Fail("Vulkan found no device-local image memory type");
        }
        VkMemoryAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memoryType;
        if (!Check(AllocateMemory(Device, &allocate, nullptr, &output.Memory),
                   "vkAllocateMemory(image)") ||
            !Check(BindImageMemory(Device, output.Image, output.Memory, 0),
                   "vkBindImageMemory"))
        {
            DestroyRawImage(output);
            return false;
        }
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = output.Image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        view.subresourceRange = {aspect, 0, 1, 0, 1};
        if (!Check(CreateImageViewFunction(Device, &view, nullptr, &output.View),
                   "vkCreateImageView"))
        {
            DestroyRawImage(output);
            return false;
        }
        output.Layout = VK_IMAGE_LAYOUT_UNDEFINED;
        output.Format = format;
        output.Aspect = aspect;
        output.Width = width;
        output.Height = height;
        return true;
    }

    void DestroyRawImage(RawImage& image)
    {
        if (image.View != 0 && DestroyImageViewFunction)
            DestroyImageViewFunction(Device, image.View, nullptr);
        if (image.Image != 0 && DestroyImageFunction)
            DestroyImageFunction(Device, image.Image, nullptr);
        if (image.Memory != 0 && FreeMemory) FreeMemory(Device, image.Memory, nullptr);
        image = {};
    }

    static void LayoutSource(VkImageLayout layout, VkFlags& stage, VkFlags& access)
    {
        switch (layout)
        {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            access = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            access = VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            access = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        default:
            stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            access = 0;
            break;
        }
    }

    static void LayoutDestination(VkImageLayout layout, VkFlags& stage, VkFlags& access)
    {
        LayoutSource(layout, stage, access);
        if (layout == VK_IMAGE_LAYOUT_UNDEFINED)
        {
            stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            access = 0;
        }
    }

    void TransitionImage(VkCommandBuffer commandBuffer, RawImage& image, VkImageLayout layout)
    {
        if (image.Layout == layout) return;
        VkFlags sourceStage = 0;
        VkFlags sourceAccess = 0;
        VkFlags destinationStage = 0;
        VkFlags destinationAccess = 0;
        LayoutSource(image.Layout, sourceStage, sourceAccess);
        LayoutDestination(layout, destinationStage, destinationAccess);
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = sourceAccess;
        barrier.dstAccessMask = destinationAccess;
        barrier.oldLayout = image.Layout;
        barrier.newLayout = layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.Image;
        barrier.subresourceRange = {image.Aspect, 0, 1, 0, 1};
        CmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0,
                           0, nullptr, 0, nullptr, 1, &barrier);
        image.Layout = layout;
    }

    bool CreateCommandState()
    {
        VkCommandPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = QueueFamilyIndex;
        if (!Check(CreateCommandPoolFunction(Device, &pool, nullptr, &CommandPool),
                   "vkCreateCommandPool")) return false;
        VkCommandBufferAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool = CommandPool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (!Check(AllocateCommandBuffers(Device, &allocate, &CommandBuffer),
                   "vkAllocateCommandBuffers")) return false;
        return RecreateSignaledFrameFence();
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

    bool BeginImmediate()
    {
        if (!Check(WaitForFences(Device, 1, &FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
                   "vkWaitForFences(immediate)") ||
            !Check(ResetCommandPool(Device, CommandPool, 0), "vkResetCommandPool(immediate)"))
            return false;
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        return Check(BeginCommandBuffer(CommandBuffer, &begin), "vkBeginCommandBuffer(immediate)");
    }

    bool EndImmediate()
    {
        if (!Check(EndCommandBuffer(CommandBuffer), "vkEndCommandBuffer(immediate)") ||
            !Check(ResetFences(Device, 1, &FrameFence), "vkResetFences(immediate)")) return false;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &CommandBuffer;
        const VkResult submitResult = QueueSubmit(GraphicsQueue, 1, &submit, FrameFence);
        if (submitResult != VK_SUCCESS)
        {
            const std::string submitError = VulkanError("vkQueueSubmit(immediate)", submitResult);
            (void)RecreateSignaledFrameFence();
            return Fail(submitError);
        }
        return Check(WaitForFences(Device, 1, &FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
                     "vkWaitForFences(immediate completion)");
    }

    bool InitializeImage(RawImage& image, const void* data, std::size_t size)
    {
        if (data == nullptr)
        {
            if (!BeginImmediate()) return false;
            TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return EndImmediate();
        }
        RawBuffer staging;
        if (!CreateRawBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging)) return false;
        std::memcpy(staging.Mapped, data, size);
        if (!BeginImmediate())
        {
            DestroyRawBuffer(staging);
            return false;
        }
        TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {image.Aspect, 0, 0, 1};
        copy.imageExtent = {image.Width, image.Height, 1};
        CmdCopyBufferToImage(CommandBuffer, staging.Buffer, image.Image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const bool result = EndImmediate();
        DestroyRawBuffer(staging);
        return result;
    }

    bool CreateColorRenderPass(bool clearColor, bool clearDepth, VkRenderPass& output)
    {
        std::array<VkAttachmentDescription, 2> attachments{};
        attachments[0].format = VK_FORMAT_R8G8B8A8_SRGB;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = clearColor ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = 2;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = clearDepth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = 2;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        const VkAttachmentReference depth{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color;
        subpass.pDepthStencilAttachment = &depth;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = dependency.srcStageMask;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        create.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        create.pAttachments = attachments.data();
        create.subpassCount = 1;
        create.pSubpasses = &subpass;
        create.dependencyCount = 1;
        create.pDependencies = &dependency;
        return Check(CreateRenderPassFunction(Device, &create, nullptr, &output),
                     "vkCreateRenderPass(color)");
    }

    bool CreateShadowRenderPass(bool clearDepth, VkRenderPass& output)
    {
        VkAttachmentDescription attachment{};
        attachment.format = VK_FORMAT_D32_SFLOAT;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = clearDepth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = 2;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        const VkAttachmentReference depth{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pDepthStencilAttachment = &depth;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        create.attachmentCount = 1;
        create.pAttachments = &attachment;
        create.subpassCount = 1;
        create.pSubpasses = &subpass;
        create.dependencyCount = 1;
        create.pDependencies = &dependency;
        return Check(CreateRenderPassFunction(Device, &create, nullptr, &output),
                     "vkCreateRenderPass(shadow)");
    }

    bool CreateRenderPasses()
    {
        for (std::uint32_t mask = 0; mask < ColorRenderPasses.size(); ++mask)
        {
            if (!CreateColorRenderPass((mask & 1U) != 0, (mask & 2U) != 0,
                                       ColorRenderPasses[mask])) return false;
        }
        return CreateShadowRenderPass(false, ShadowRenderPasses[0]) &&
               CreateShadowRenderPass(true, ShadowRenderPasses[1]);
    }

    bool CreateFramebuffer(VkRenderPass renderPass, const VkImageView* attachments,
                           std::uint32_t attachmentCount, std::uint32_t width,
                           std::uint32_t height, VkFramebuffer& output)
    {
        VkFramebufferCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        create.renderPass = renderPass;
        create.attachmentCount = attachmentCount;
        create.pAttachments = attachments;
        create.width = width;
        create.height = height;
        create.layers = 1;
        return Check(CreateFramebufferFunction(Device, &create, nullptr, &output),
                     "vkCreateFramebuffer");
    }

    bool CreateShaderModules()
    {
        VkShaderModuleCreateInfo vertex{};
        vertex.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vertex.codeSize = sizeof(EvansVulkanVertexShader);
        vertex.pCode = EvansVulkanVertexShader;
        if (!Check(CreateShaderModuleFunction(Device, &vertex, nullptr, &VertexShader),
                   "vkCreateShaderModule(vertex)")) return false;
        VkShaderModuleCreateInfo fragment{};
        fragment.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fragment.codeSize = sizeof(EvansVulkanFragmentShader);
        fragment.pCode = EvansVulkanFragmentShader;
        if (!Check(CreateShaderModuleFunction(Device, &fragment, nullptr, &FragmentShader),
                   "vkCreateShaderModule(fragment)")) return false;
        fragment.codeSize = sizeof(EvansVulkanShadowFragmentShader);
        fragment.pCode = EvansVulkanShadowFragmentShader;
        if (!Check(CreateShaderModuleFunction(
                Device, &fragment, nullptr, &ShadowFragmentShader),
                "vkCreateShaderModule(shadow fragment)")) return false;
        fragment.codeSize = sizeof(EvansVulkanOutlineFragmentShader);
        fragment.pCode = EvansVulkanOutlineFragmentShader;
        return Check(CreateShaderModuleFunction(
            Device, &fragment, nullptr, &OutlineFragmentShader),
            "vkCreateShaderModule(outline fragment)");
    }

    bool CreateDescriptorState()
    {
        const RealtimeShaderParseResult& parsedShader =
            BuiltinRealtimeShader();
        std::string contractError;
        if (!parsedShader)
        {
            return Fail(
                "Built-in realtime shader parse failed:\n" +
                parsedShader.ErrorText());
        }
        if (!ValidateRealtimeShaderContract(
                parsedShader.Program,
                sizeof(RealtimeDrawConstants),
                MaximumSceneLights,
                contractError))
        {
            return Fail(contractError);
        }
        const RealtimeShaderProgram& shader = parsedShader.Program;
        const std::array<VkDescriptorSetLayoutBinding, 6> bindings = {{
            {shader.FindResourceByName("DrawConstants")->VulkanBinding,
             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {shader.FindResourceByName("ShadowMap")->VulkanBinding,
             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {shader.FindResourceByName("DiffuseMap")->VulkanBinding,
             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {shader.FindResourceByName("NormalMap")->VulkanBinding,
             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {shader.FindResourceByName("OpacityMap")->VulkanBinding,
             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {shader.FindResourceByName("SurfaceSampler")->VulkanBinding,
             VK_DESCRIPTOR_TYPE_SAMPLER, 1,
             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
        VkDescriptorSetLayoutCreateInfo setLayout{};
        setLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        setLayout.pBindings = bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device, &setLayout, nullptr,
                                                      &DescriptorSetLayout),
                   "vkCreateDescriptorSetLayout")) return false;
        VkPipelineLayoutCreateInfo pipelineLayout{};
        pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayout.setLayoutCount = 1;
        pipelineLayout.pSetLayouts = &DescriptorSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device, &pipelineLayout, nullptr, &PipelineLayout),
                   "vkCreatePipelineLayout")) return false;
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.maxLod = 0.0f;
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        if (!Check(CreateSamplerFunction(Device, &sampler, nullptr, &ShadowSampler),
                   "vkCreateSampler")) return false;
        if (!CreateRawBuffer(UniformBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, UniformBuffer)) return false;
        const std::array<VkDescriptorPoolSize, 3> sizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, MaximumFrameDescriptorSets},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaximumFrameDescriptorSets * 4U},
            {VK_DESCRIPTOR_TYPE_SAMPLER, MaximumFrameDescriptorSets}}};
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = MaximumFrameDescriptorSets;
        pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        pool.pPoolSizes = sizes.data();
        if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &DescriptorPool),
                   "vkCreateDescriptorPool")) return false;
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = DescriptorPool;
        std::vector<VkDescriptorSetLayout> layouts(
            MaximumFrameDescriptorSets, DescriptorSetLayout);
        DescriptorSets.resize(MaximumFrameDescriptorSets);
        allocate.descriptorSetCount = MaximumFrameDescriptorSets;
        allocate.pSetLayouts = layouts.data();
        return Check(AllocateDescriptorSets(Device, &allocate, DescriptorSets.data()),
                     "vkAllocateDescriptorSets(frame)");
    }

    void UpdateDescriptorSet(
        VkDescriptorSet descriptorSet,
        const std::array<VkImageView, 4>& imageViews)
    {
        const VkDescriptorBufferInfo buffer{
            UniformBuffer.Buffer, 0, sizeof(RealtimeDrawConstants)};
        std::array<VkDescriptorImageInfo, 4> images {};
        for (std::size_t index = 0; index < images.size(); ++index)
        {
            images[index] = {
                0,
                imageViews[index],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
        const VkDescriptorImageInfo sampler{
            ShadowSampler, 0, VK_IMAGE_LAYOUT_UNDEFINED};
        std::array<VkWriteDescriptorSet, 6> writes{};
        const RealtimeShaderProgram& shader =
            BuiltinRealtimeShader().Program;
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding =
            shader.FindResourceByName("DrawConstants")->VulkanBinding;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[0].pBufferInfo = &buffer;
        for (std::uint32_t index = 0; index < images.size(); ++index)
        {
            VkWriteDescriptorSet& write = writes[index + 1];
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSet;
            static constexpr std::array<const char*, 4> imageNames = {
                "ShadowMap", "DiffuseMap", "NormalMap", "OpacityMap"};
            write.dstBinding =
                shader.FindResourceByName(imageNames[index])->VulkanBinding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            write.pImageInfo = &images[index];
        }
        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = descriptorSet;
        writes[5].dstBinding =
            shader.FindResourceByName("SurfaceSampler")->VulkanBinding;
        writes[5].descriptorCount = 1;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[5].pImageInfo = &sampler;
        UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(),
                             0, nullptr);
    }

    bool CreateMainTarget(int width, int height)
    {
        DestroyMainTarget();
        const std::uint32_t targetWidth = static_cast<std::uint32_t>(width);
        const std::uint32_t targetHeight = static_cast<std::uint32_t>(height);
        if (!CreateRawImage(targetWidth, targetHeight, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT, MainColor) ||
            !CreateRawImage(targetWidth, targetHeight, VK_FORMAT_D32_SFLOAT,
                            VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT, MainDepth))
        {
            DestroyMainTarget();
            return false;
        }
        const std::array<VkImageView, 2> attachments = {MainColor.View, MainDepth.View};
        if (!CreateFramebuffer(ColorRenderPasses[0], attachments.data(),
                               static_cast<std::uint32_t>(attachments.size()),
                               targetWidth, targetHeight, MainFramebuffer) ||
            !CreateRawBuffer(static_cast<VkDeviceSize>(targetWidth) * targetHeight * 4U,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ColorReadback) ||
            !CreateRawBuffer(static_cast<VkDeviceSize>(targetWidth) * targetHeight * sizeof(float),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, DepthReadback))
        {
            DestroyMainTarget();
            return false;
        }
        return true;
    }

    void DestroyMainTarget()
    {
        if (MainFramebuffer != 0 && DestroyFramebufferFunction)
            DestroyFramebufferFunction(Device, MainFramebuffer, nullptr);
        MainFramebuffer = 0;
        DestroyRawImage(MainColor);
        DestroyRawImage(MainDepth);
        DestroyRawBuffer(ColorReadback);
        DestroyRawBuffer(DepthReadback);
    }

    bool CreateDummyDepth()
    {
        if (!CreateRawImage(1, 1, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            DummyDepth)) return false;
        const float clearDepth = 1.0f;
        if (!InitializeImage(DummyDepth, &clearDepth, sizeof(clearDepth))) return false;
        return true;
    }

    bool CreateDummyColor()
    {
        if (!CreateRawImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            DummyColor)) return false;
        const std::uint32_t white = 0xffffffffU;
        return InitializeImage(DummyColor, &white, sizeof(white));
    }

    bool CreateTextureResource(TextureResource& resource, const void* initialData)
    {
        const bool depth = resource.Description.Format == TextureFormat::Depth32;
        VkFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (resource.Description.RenderTarget)
            usage |= depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                           : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (!CreateRawImage(static_cast<std::uint32_t>(resource.Description.Width),
                            static_cast<std::uint32_t>(resource.Description.Height),
                            depth
                                ? VK_FORMAT_D32_SFLOAT
                                : (resource.Description.RenderTarget ||
                                    resource.Description.ColorSpace ==
                                        TextureColorSpace::SRGB
                                    ? VK_FORMAT_R8G8B8A8_SRGB
                                    : VK_FORMAT_R8G8B8A8_UNORM),
                            depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                            usage, resource.Image)) return false;
        if (initialData != nullptr || !resource.Description.RenderTarget)
        {
            const std::size_t bytes = static_cast<std::size_t>(resource.Description.Width) *
                                      resource.Description.Height * 4U;
            if (!InitializeImage(resource.Image, initialData, bytes)) return false;
        }
        if (!resource.Description.RenderTarget) return true;
        if (depth)
        {
            return CreateFramebuffer(ShadowRenderPasses[0], &resource.Image.View, 1,
                                     resource.Image.Width, resource.Image.Height,
                                     resource.Framebuffer);
        }
        if (!CreateRawImage(resource.Image.Width, resource.Image.Height, VK_FORMAT_D32_SFLOAT,
                            VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                            resource.PrivateDepth)) return false;
        const std::array<VkImageView, 2> attachments = {
            resource.Image.View, resource.PrivateDepth.View};
        return CreateFramebuffer(ColorRenderPasses[0], attachments.data(),
                                 static_cast<std::uint32_t>(attachments.size()),
                                 resource.Image.Width, resource.Image.Height,
                                 resource.Framebuffer);
    }

    void DestroyTextureResource(TextureResource& resource)
    {
        if (resource.Framebuffer != 0 && DestroyFramebufferFunction)
            DestroyFramebufferFunction(Device, resource.Framebuffer, nullptr);
        DestroyRawImage(resource.PrivateDepth);
        DestroyRawImage(resource.Image);
        resource = {};
    }

    [[nodiscard]] std::uint32_t PipelineKey(
        const PipelineState& pipeline,
        PassType pass) const
    {
        return static_cast<std::uint32_t>(pass) |
               (pipeline.Fill == FillMode::Wireframe ? 1U << 2U : 0U) |
               (pipeline.DepthTest ? 1U << 3U : 0U) |
               (pipeline.DepthWrite ? 1U << 4U : 0U) |
               (pipeline.ColorWrite ? 1U << 5U : 0U) |
               (pipeline.OutlineExtrusion ? 1U << 6U : 0U) |
               (pipeline.BlendEnabled ? 1U << 7U : 0U);
    }

    VkPipeline GetPipeline(const PipelineState& pipeline, PassType pass)
    {
        const bool shadow = pass == PassType::Shadow;
        const std::uint32_t key = PipelineKey(pipeline, pass);
        const auto found = Pipelines.find(key);
        if (found != Pipelines.end()) return found->second;

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = VertexShader;
        const RealtimeShaderProgram& shader =
            BuiltinRealtimeShader().Program;
        stages[0].pName = shader.FindStage(
            RealtimeShaderStage::Vertex)->SPIRV.c_str();
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = shadow
            ? ShadowFragmentShader
            : (pass == PassType::Outline
                ? OutlineFragmentShader
                : FragmentShader);
        stages[1].pName = shader.FindStage(
            shadow
                ? RealtimeShaderStage::Shadow
                : (pass == PassType::Outline
                    ? RealtimeShaderStage::Outline
                    : RealtimeShaderStage::Fragment))->SPIRV.c_str();

        const VkVertexInputBindingDescription binding{0, sizeof(Vertex),
                                                       VK_VERTEX_INPUT_RATE_VERTEX};
        const std::array<VkVertexInputAttributeDescription, 4> attributes = {{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, Position))},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, Normal))},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, Color))},
            {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<std::uint32_t>(offsetof(Vertex, TexCoord))}}};
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
        raster.polygonMode = pipeline.Fill == FillMode::Wireframe && !shadow
            ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        raster.cullMode = pipeline.OutlineExtrusion
            ? VK_CULL_MODE_FRONT_BIT : VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.depthBiasEnable = shadow ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        raster.depthBiasConstantFactor = shadow ? static_cast<float>(ShadowDepthBiasUnits) : 0.0f;
        raster.depthBiasClamp = ShadowDepthBiasClamp;
        raster.depthBiasSlopeFactor = shadow ? ShadowSlopeScaledDepthBias : 0.0f;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = pipeline.DepthTest ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        depth.depthWriteEnable = pipeline.DepthWrite ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        depth.depthCompareOp = pipeline.DepthWrite ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_LESS_OR_EQUAL;
        depth.minDepthBounds = 0.0f;
        depth.maxDepthBounds = 1.0f;
        VkPipelineColorBlendAttachmentState colorAttachment{};
        colorAttachment.blendEnable =
            pass == PassType::Color && pipeline.BlendEnabled
            ? VK_TRUE_VALUE
            : VK_FALSE_VALUE;
        colorAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorAttachment.dstColorBlendFactor =
            VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorAttachment.dstAlphaBlendFactor =
            VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        colorAttachment.colorWriteMask = pipeline.ColorWrite
            ? VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            : 0U;
        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.attachmentCount = shadow ? 0U : 1U;
        colorBlend.pAttachments = shadow ? nullptr : &colorAttachment;
        const std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkGraphicsPipelineCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        create.stageCount = 2U;
        create.pStages = stages.data();
        create.pVertexInputState = &vertexInput;
        create.pInputAssemblyState = &inputAssembly;
        create.pViewportState = &viewport;
        create.pRasterizationState = &raster;
        create.pMultisampleState = &multisample;
        create.pDepthStencilState = &depth;
        create.pColorBlendState = &colorBlend;
        create.pDynamicState = &dynamic;
        create.layout = PipelineLayout;
        create.renderPass = shadow ? ShadowRenderPasses[0] : ColorRenderPasses[0];
        create.subpass = 0;
        VkPipeline result = 0;
        if (!Check(CreateGraphicsPipelines(Device, 0, 1, &create, nullptr, &result),
                   "vkCreateGraphicsPipelines")) return 0;
        Pipelines.emplace(key, result);
        return result;
    }

    void CopyImageToReadback(RawImage& image, RawBuffer& readback)
    {
        TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {image.Aspect, 0, 0, 1};
        copy.imageExtent = {image.Width, image.Height, 1};
        CmdCopyImageToBuffer(CommandBuffer, image.Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback.Buffer, 1, &copy);
    }

    bool EnsureShadowReadback(std::uint32_t width, std::uint32_t height)
    {
        const VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * sizeof(float);
        if (ShadowReadback.Buffer != 0 && ShadowReadback.Size >= size) return true;
        DestroyRawBuffer(ShadowReadback);
        return CreateRawBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ShadowReadback);
    }

    void PresentToWindow(const FrameCapture& capture) const
    {
        if (CreateInfo.NativeWindow == nullptr || !capture.HasColorOutput()) return;
        HWND window = static_cast<HWND>(CreateInfo.NativeWindow);
        if (!IsWindow(window)) return;
        std::vector<std::uint8_t> bgra(capture.ColorRGBA8.size());
        for (std::size_t pixel = 0; pixel < capture.ColorRGBA8.size() / 4U; ++pixel)
        {
            bgra[pixel * 4U] = capture.ColorRGBA8[pixel * 4U + 2U];
            bgra[pixel * 4U + 1U] = capture.ColorRGBA8[pixel * 4U + 1U];
            bgra[pixel * 4U + 2U] = capture.ColorRGBA8[pixel * 4U];
            bgra[pixel * 4U + 3U] = 255;
        }
        BITMAPINFO bitmap{};
        bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmap.bmiHeader.biWidth = static_cast<LONG>(capture.Width);
        bitmap.bmiHeader.biHeight = -static_cast<LONG>(capture.Height);
        bitmap.bmiHeader.biPlanes = 1;
        bitmap.bmiHeader.biBitCount = 32;
        bitmap.bmiHeader.biCompression = BI_RGB;
        RECT client{};
        GetClientRect(window, &client);
        HDC dc = GetDC(window);
        if (dc != nullptr)
        {
            StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0,
                          static_cast<int>(capture.Width), static_cast<int>(capture.Height),
                          bgra.data(), &bitmap, DIB_RGB_COLORS, SRCCOPY);
            ReleaseDC(window, dc);
        }
    }

    HMODULE VulkanModule = nullptr;
    VkInstance Instance = nullptr;
    VkPhysicalDevice PhysicalDevice = nullptr;
    VkDevice Device = nullptr;
    VkQueue GraphicsQueue = nullptr;
    VkPhysicalDeviceMemoryProperties MemoryProperties{};
    std::uint32_t QueueFamilyIndex = 0;
    std::string DeviceName;
    VkCommandPool CommandPool = 0;
    VkCommandBuffer CommandBuffer = nullptr;
    VkFence FrameFence = 0;
    std::array<VkRenderPass, 4> ColorRenderPasses{};
    std::array<VkRenderPass, 2> ShadowRenderPasses{};
    VkShaderModule VertexShader = 0;
    VkShaderModule FragmentShader = 0;
    VkShaderModule ShadowFragmentShader = 0;
    VkShaderModule OutlineFragmentShader = 0;
    VkDescriptorSetLayout DescriptorSetLayout = 0;
    VkPipelineLayout PipelineLayout = 0;
    VkDescriptorPool DescriptorPool = 0;
    std::vector<VkDescriptorSet> DescriptorSets;
    std::uint32_t DescriptorSetIndex = 0;
    VkSampler ShadowSampler = 0;
    RawBuffer UniformBuffer;
    std::size_t UniformOffset = 0;
    RawImage DummyDepth;
    RawImage DummyColor;
    RawImage MainColor;
    RawImage MainDepth;
    VkFramebuffer MainFramebuffer = 0;
    RawBuffer ColorReadback;
    RawBuffer DepthReadback;
    RawBuffer ShadowReadback;
    std::unordered_map<std::uint32_t, BufferResource> Buffers;
    std::unordered_map<std::uint32_t, TextureResource> Textures;
    std::unordered_map<std::uint32_t, VkPipeline> Pipelines;
    RHICreateInfo CreateInfo;
    FrameCapture PendingFrame;
    FrameCapture LastFrame;
    PassDesc CurrentPass;
    PipelineState BoundPipeline;
    Matrix4 BoundWorld;
    Matrix4 BoundViewProjection;
    Matrix4 BoundLightViewProjection;
    BufferHandle BoundVertexBuffer;
    BufferHandle BoundIndexBuffer;
    TextureHandle LastShadowTexture;
    RawImage* ActiveColor = nullptr;
    RawImage* ActiveDepth = nullptr;
    VkFramebuffer ActiveFramebuffer = 0;
    std::uint32_t ActiveWidth = 0;
    std::uint32_t ActiveHeight = 0;
    std::string LastError;
    std::thread::id OwnerThread;
    std::uint64_t FrameNumber = 0;
    std::uint32_t NextBuffer = 0x56000001u;
    std::uint32_t NextTexture = 0x56800001u;
    bool Initialized = false;
    bool Abandoned = false;
    bool FrameOpen = false;
    bool PassOpen = false;
    bool PipelineBound = false;
    bool WorldBound = false;
    bool ViewProjectionBound = false;
    bool LightViewProjectionBound = false;

    PFN_vkGetInstanceProcAddr GetInstanceProcAddress = nullptr;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddress = nullptr;
    PFN_vkCreateInstance CreateInstanceFunction = nullptr;
    PFN_vkDestroyInstance DestroyInstanceFunction = nullptr;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceFeatures GetPhysicalDeviceFeatures = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateDevice CreateDeviceFunction = nullptr;
    PFN_vkDestroyDevice DestroyDeviceFunction = nullptr;
    PFN_vkGetDeviceQueue GetDeviceQueue = nullptr;
    PFN_vkDeviceWaitIdle DeviceWaitIdle = nullptr;
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
    PFN_vkCreateRenderPass CreateRenderPassFunction = nullptr;
    PFN_vkDestroyRenderPass DestroyRenderPassFunction = nullptr;
    PFN_vkCreateFramebuffer CreateFramebufferFunction = nullptr;
    PFN_vkDestroyFramebuffer DestroyFramebufferFunction = nullptr;
    PFN_vkCreateShaderModule CreateShaderModuleFunction = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModuleFunction = nullptr;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayoutFunction = nullptr;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayoutFunction = nullptr;
    PFN_vkCreateDescriptorPool CreateDescriptorPoolFunction = nullptr;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPoolFunction = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;
    PFN_vkCreatePipelineLayout CreatePipelineLayoutFunction = nullptr;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayoutFunction = nullptr;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines = nullptr;
    PFN_vkDestroyPipeline DestroyPipelineFunction = nullptr;
    PFN_vkCreateCommandPool CreateCommandPoolFunction = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPoolFunction = nullptr;
    PFN_vkResetCommandPool ResetCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkCreateFence CreateFenceFunction = nullptr;
    PFN_vkDestroyFence DestroyFenceFunction = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkResetFences ResetFences = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage = nullptr;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer = nullptr;
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass = nullptr;
    PFN_vkCmdEndRenderPass CmdEndRenderPass = nullptr;
    PFN_vkCmdSetViewport CmdSetViewport = nullptr;
    PFN_vkCmdSetScissor CmdSetScissor = nullptr;
    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers = nullptr;
    PFN_vkCmdBindIndexBuffer CmdBindIndexBuffer = nullptr;
    PFN_vkCmdDrawIndexed CmdDrawIndexed = nullptr;
};

VulkanRHI::VulkanRHI() : m_Impl(std::make_unique<Impl>()) {}
VulkanRHI::~VulkanRHI() { Shutdown(); }

bool VulkanRHI::Initialize(const RHICreateInfo& createInfo)
{
    if (m_Impl->Abandoned)
        return m_Impl->Fail("Vulkan RHI was abandoned after a GPU timeout");
    if (m_Impl->Initialized) return m_Impl->Fail("Vulkan RHI is already initialized");
    if (createInfo.Width <= 0 || createInfo.Height <= 0)
        return m_Impl->Fail("Vulkan RHI dimensions must be positive");
    if (!m_Impl->LoadRuntime() || !m_Impl->CreateInstance() || !m_Impl->LoadInstanceFunctions() ||
        !m_Impl->SelectPhysicalDevice() || !m_Impl->CreateLogicalDevice() ||
        !m_Impl->CreateCommandState() || !m_Impl->CreateRenderPasses() ||
        !m_Impl->CreateShaderModules() || !m_Impl->CreateDescriptorState() ||
        !m_Impl->CreateMainTarget(createInfo.Width, createInfo.Height) ||
        !m_Impl->CreateDummyDepth() || !m_Impl->CreateDummyColor())
    {
        Shutdown();
        return false;
    }
    m_Impl->CreateInfo = createInfo;
    m_Impl->OwnerThread = std::this_thread::get_id();
    m_Impl->Initialized = true;
    m_Impl->LastError.clear();
    return true;
}

void VulkanRHI::Shutdown()
{
    if (!m_Impl) return;
    if (m_Impl->Abandoned) return;
    if (m_Impl->Device != nullptr && m_Impl->FrameFence != 0 && m_Impl->WaitForFences)
    {
        const VkResult waitResult = m_Impl->WaitForFences(
            m_Impl->Device,
            1,
            &m_Impl->FrameFence,
            VK_TRUE_VALUE,
            GpuWaitTimeout);
        if (waitResult != VK_SUCCESS)
        {
            m_Impl->LastError = VulkanError("vkWaitForFences(shutdown)", waitResult);
            m_Impl->Initialized = false;
            m_Impl->FrameOpen = false;
            m_Impl->PassOpen = false;
            m_Impl->Abandoned = true;
            return;
        }
    }
    if (m_Impl->Device != nullptr)
    {
        for (auto& [key, pipeline] : m_Impl->Pipelines)
        {
            (void)key;
            if (pipeline != 0 && m_Impl->DestroyPipelineFunction)
                m_Impl->DestroyPipelineFunction(m_Impl->Device, pipeline, nullptr);
        }
        for (auto& [handle, resource] : m_Impl->Buffers)
        {
            (void)handle;
            m_Impl->DestroyRawBuffer(resource.Allocation);
        }
        for (auto& [handle, resource] : m_Impl->Textures)
        {
            (void)handle;
            m_Impl->DestroyTextureResource(resource);
        }
        m_Impl->DestroyMainTarget();
        m_Impl->DestroyRawImage(m_Impl->DummyDepth);
        m_Impl->DestroyRawImage(m_Impl->DummyColor);
        m_Impl->DestroyRawBuffer(m_Impl->ShadowReadback);
        m_Impl->DestroyRawBuffer(m_Impl->UniformBuffer);
        if (m_Impl->DescriptorPool != 0 && m_Impl->DestroyDescriptorPoolFunction)
            m_Impl->DestroyDescriptorPoolFunction(m_Impl->Device, m_Impl->DescriptorPool, nullptr);
        if (m_Impl->PipelineLayout != 0 && m_Impl->DestroyPipelineLayoutFunction)
            m_Impl->DestroyPipelineLayoutFunction(m_Impl->Device, m_Impl->PipelineLayout, nullptr);
        if (m_Impl->DescriptorSetLayout != 0 && m_Impl->DestroyDescriptorSetLayoutFunction)
            m_Impl->DestroyDescriptorSetLayoutFunction(
                m_Impl->Device, m_Impl->DescriptorSetLayout, nullptr);
        if (m_Impl->ShadowSampler != 0 && m_Impl->DestroySamplerFunction)
            m_Impl->DestroySamplerFunction(m_Impl->Device, m_Impl->ShadowSampler, nullptr);
        if (m_Impl->VertexShader != 0 && m_Impl->DestroyShaderModuleFunction)
            m_Impl->DestroyShaderModuleFunction(m_Impl->Device, m_Impl->VertexShader, nullptr);
        if (m_Impl->FragmentShader != 0 && m_Impl->DestroyShaderModuleFunction)
            m_Impl->DestroyShaderModuleFunction(m_Impl->Device, m_Impl->FragmentShader, nullptr);
        if (m_Impl->ShadowFragmentShader != 0 && m_Impl->DestroyShaderModuleFunction)
            m_Impl->DestroyShaderModuleFunction(
                m_Impl->Device, m_Impl->ShadowFragmentShader, nullptr);
        if (m_Impl->OutlineFragmentShader != 0 && m_Impl->DestroyShaderModuleFunction)
            m_Impl->DestroyShaderModuleFunction(
                m_Impl->Device, m_Impl->OutlineFragmentShader, nullptr);
        for (VkRenderPass& pass : m_Impl->ColorRenderPasses)
        {
            if (pass != 0 && m_Impl->DestroyRenderPassFunction)
                m_Impl->DestroyRenderPassFunction(m_Impl->Device, pass, nullptr);
            pass = 0;
        }
        for (VkRenderPass& pass : m_Impl->ShadowRenderPasses)
        {
            if (pass != 0 && m_Impl->DestroyRenderPassFunction)
                m_Impl->DestroyRenderPassFunction(m_Impl->Device, pass, nullptr);
            pass = 0;
        }
        if (m_Impl->FrameFence != 0 && m_Impl->DestroyFenceFunction)
            m_Impl->DestroyFenceFunction(m_Impl->Device, m_Impl->FrameFence, nullptr);
        if (m_Impl->CommandPool != 0 && m_Impl->DestroyCommandPoolFunction)
            m_Impl->DestroyCommandPoolFunction(m_Impl->Device, m_Impl->CommandPool, nullptr);
        if (m_Impl->DestroyDeviceFunction)
            m_Impl->DestroyDeviceFunction(m_Impl->Device, nullptr);
    }
    m_Impl->Buffers.clear();
    m_Impl->Textures.clear();
    m_Impl->Pipelines.clear();
    m_Impl->Device = nullptr;
    m_Impl->PhysicalDevice = nullptr;
    m_Impl->GraphicsQueue = nullptr;
    m_Impl->CommandPool = 0;
    m_Impl->CommandBuffer = nullptr;
    m_Impl->FrameFence = 0;
    m_Impl->DescriptorPool = 0;
    m_Impl->DescriptorSets.clear();
    m_Impl->DescriptorSetIndex = 0;
    m_Impl->DescriptorSetLayout = 0;
    m_Impl->PipelineLayout = 0;
    m_Impl->ShadowSampler = 0;
    m_Impl->VertexShader = 0;
    m_Impl->FragmentShader = 0;
    m_Impl->ShadowFragmentShader = 0;
    m_Impl->OutlineFragmentShader = 0;
    if (m_Impl->Instance != nullptr && m_Impl->DestroyInstanceFunction)
        m_Impl->DestroyInstanceFunction(m_Impl->Instance, nullptr);
    m_Impl->Instance = nullptr;
    if (m_Impl->VulkanModule != nullptr) FreeLibrary(m_Impl->VulkanModule);
    m_Impl->VulkanModule = nullptr;
    m_Impl->Initialized = false;
    m_Impl->Abandoned = false;
    m_Impl->FrameOpen = false;
    m_Impl->PassOpen = false;
    m_Impl->OwnerThread = {};
}

bool VulkanRHI::IsInitialized() const { return m_Impl->Initialized; }
RHIBackend VulkanRHI::GetBackend() const { return RHIBackend::Vulkan; }
const char* VulkanRHI::GetName() const { return "Vulkan Native Graphics RHI"; }
RHICapabilities VulkanRHI::GetCapabilities() const
{
    RHICapabilities result;
    result.NativeAPI = m_Impl->Initialized;
    result.Wireframe = true;
    result.ShadowMaps = true;
    result.Outline = true;
    result.AsyncCommandRecording = false;
    result.MaximumTextureSize = m_Impl->Initialized ? 16384U : 0U;
    return result;
}

bool VulkanRHI::Resize(int width, int height)
{
    if (!m_Impl->CheckThread("Resize") || m_Impl->FrameOpen || width <= 0 || height <= 0)
        return m_Impl->Fail("Vulkan cannot resize in its current state");
    if (width == m_Impl->CreateInfo.Width && height == m_Impl->CreateInfo.Height) return true;
    if (!m_Impl->Check(m_Impl->WaitForFences(
            m_Impl->Device, 1, &m_Impl->FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
            "vkWaitForFences(resize)") ||
        !m_Impl->CreateMainTarget(width, height)) return false;
    m_Impl->CreateInfo.Width = width;
    m_Impl->CreateInfo.Height = height;
    return true;
}

BufferHandle VulkanRHI::CreateBuffer(const BufferDesc& desc, const void* initialData)
{
    if (!m_Impl->CheckThread("CreateBuffer") || desc.SizeInBytes == 0)
    {
        m_Impl->Fail("Vulkan buffer description is invalid");
        return {};
    }
    Impl::BufferResource resource;
    resource.Description = desc;
    resource.Bytes.resize(desc.SizeInBytes);
    if (initialData != nullptr) std::memcpy(resource.Bytes.data(), initialData, desc.SizeInBytes);
    VkFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (desc.Usage == BufferUsage::Index) usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    else if (desc.Usage == BufferUsage::Uniform) usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (!m_Impl->CreateRawBuffer(desc.SizeInBytes, usage,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 resource.Allocation)) return {};
    std::memcpy(resource.Allocation.Mapped, resource.Bytes.data(), desc.SizeInBytes);
    const BufferHandle handle{m_Impl->NextBuffer++};
    m_Impl->Buffers.emplace(handle.Value, std::move(resource));
    return handle;
}

bool VulkanRHI::UpdateBuffer(BufferHandle buffer, std::size_t offset,
                             const void* data, std::size_t size)
{
    if (!m_Impl->CheckThread("UpdateBuffer")) return false;
    auto found = m_Impl->Buffers.find(buffer.Value);
    if (found == m_Impl->Buffers.end() || data == nullptr || size == 0 ||
        offset > found->second.Bytes.size() || size > found->second.Bytes.size() - offset)
        return m_Impl->Fail("Vulkan buffer update is invalid");
    std::memcpy(found->second.Bytes.data() + offset, data, size);
    std::memcpy(static_cast<std::uint8_t*>(found->second.Allocation.Mapped) + offset, data, size);
    return true;
}

bool VulkanRHI::DestroyBuffer(BufferHandle buffer)
{
    if (!m_Impl->CheckThread("DestroyBuffer")) return false;
    auto found = m_Impl->Buffers.find(buffer.Value);
    if (!buffer || found == m_Impl->Buffers.end()) return m_Impl->Fail("Vulkan buffer handle is invalid");
    m_Impl->DestroyRawBuffer(found->second.Allocation);
    m_Impl->Buffers.erase(found);
    return true;
}

TextureHandle VulkanRHI::CreateTexture(const TextureDesc& desc, const void* initialData)
{
    if (!m_Impl->CheckThread("CreateTexture") || m_Impl->FrameOpen ||
        desc.Width <= 0 || desc.Height <= 0 || desc.Width > 16384 || desc.Height > 16384)
    {
        m_Impl->Fail("Vulkan texture description or creation state is invalid");
        return {};
    }
    Impl::TextureResource resource;
    resource.Description = desc;
    if (!m_Impl->CreateTextureResource(resource, initialData))
    {
        m_Impl->DestroyTextureResource(resource);
        return {};
    }
    const TextureHandle handle{m_Impl->NextTexture++};
    m_Impl->Textures.emplace(handle.Value, std::move(resource));
    return handle;
}

bool VulkanRHI::DestroyTexture(TextureHandle texture)
{
    if (!m_Impl->CheckThread("DestroyTexture") || m_Impl->FrameOpen)
        return m_Impl->Fail("Vulkan texture cannot be destroyed in the current state");
    auto found = m_Impl->Textures.find(texture.Value);
    if (!texture || found == m_Impl->Textures.end()) return m_Impl->Fail("Vulkan texture handle is invalid");
    m_Impl->DestroyTextureResource(found->second);
    m_Impl->Textures.erase(found);
    if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
    return true;
}

bool VulkanRHI::BeginFrame()
{
    if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
        return m_Impl->Fail("Vulkan frame cannot begin in the current state");
    if (!m_Impl->Check(m_Impl->WaitForFences(m_Impl->Device, 1, &m_Impl->FrameFence,
                                             VK_TRUE_VALUE, GpuWaitTimeout),
                       "vkWaitForFences(frame)") ||
        !m_Impl->Check(m_Impl->ResetCommandPool(m_Impl->Device, m_Impl->CommandPool, 0),
                       "vkResetCommandPool(frame)")) return false;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!m_Impl->Check(m_Impl->BeginCommandBuffer(m_Impl->CommandBuffer, &begin),
                       "vkBeginCommandBuffer(frame)")) return false;
    m_Impl->PendingFrame = {};
    m_Impl->PendingFrame.Backend = RHIBackend::Vulkan;
    m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
    m_Impl->UniformOffset = 0;
    m_Impl->DescriptorSetIndex = 0;
    m_Impl->LastShadowTexture = {};
    m_Impl->FrameOpen = true;
    m_Impl->LastError.clear();
    return true;
}

void VulkanRHI::AbortFrame()
{
    if (!m_Impl || !m_Impl->Initialized)
    {
        return;
    }
    if (m_Impl->OwnerThread != std::this_thread::get_id())
    {
        m_Impl->LastError =
            "AbortFrame must execute on the owning Vulkan RHI thread";
        return;
    }
    if (m_Impl->FrameOpen)
    {
        if (m_Impl->PassOpen)
        {
            m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
        }
        (void)m_Impl->EndCommandBuffer(m_Impl->CommandBuffer);
    }
    m_Impl->PendingFrame = {};
    m_Impl->CurrentPass = {};
    m_Impl->BoundPipeline = {};
    m_Impl->BoundVertexBuffer = {};
    m_Impl->BoundIndexBuffer = {};
    m_Impl->LastShadowTexture = {};
    m_Impl->UniformOffset = 0;
    m_Impl->DescriptorSetIndex = 0;
    m_Impl->FrameOpen = false;
    m_Impl->PassOpen = false;
    m_Impl->PipelineBound = false;
    m_Impl->WorldBound = false;
    m_Impl->ViewProjectionBound = false;
    m_Impl->LightViewProjectionBound = false;
}

bool VulkanRHI::BeginPass(const PassDesc& desc)
{
    if (!m_Impl->CheckThread("BeginPass") || !m_Impl->FrameOpen || m_Impl->PassOpen)
        return m_Impl->Fail("Vulkan pass cannot begin in the current state");
    m_Impl->ActiveColor = &m_Impl->MainColor;
    m_Impl->ActiveDepth = &m_Impl->MainDepth;
    m_Impl->ActiveFramebuffer = m_Impl->MainFramebuffer;
    m_Impl->ActiveWidth = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
    m_Impl->ActiveHeight = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
    if (desc.Target)
    {
        auto target = m_Impl->Textures.find(desc.Target.Value);
        if (target == m_Impl->Textures.end() || target->second.Framebuffer == 0)
            return m_Impl->Fail("Vulkan pass target is invalid or not renderable");
        const bool depthTarget = target->second.Description.Format == TextureFormat::Depth32;
        if ((desc.Type == PassType::Shadow) != depthTarget)
            return m_Impl->Fail("Vulkan pass target format does not match the pass type");
        m_Impl->ActiveFramebuffer = target->second.Framebuffer;
        m_Impl->ActiveWidth = target->second.Image.Width;
        m_Impl->ActiveHeight = target->second.Image.Height;
        if (depthTarget)
        {
            m_Impl->ActiveColor = nullptr;
            m_Impl->ActiveDepth = &target->second.Image;
            m_Impl->LastShadowTexture = desc.Target;
        }
        else
        {
            m_Impl->ActiveColor = &target->second.Image;
            m_Impl->ActiveDepth = &target->second.PrivateDepth;
        }
    }
    else if (desc.Type == PassType::Shadow)
    {
        return m_Impl->Fail("Vulkan shadow pass requires a depth texture");
    }

    const bool shadow = desc.Type == PassType::Shadow;
    if (shadow)
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveDepth,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }
    else
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveColor,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveDepth,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    VkClearValue clearValues[2]{};
    clearValues[0].color.float32[0] = desc.ClearColor.X;
    clearValues[0].color.float32[1] = desc.ClearColor.Y;
    clearValues[0].color.float32[2] = desc.ClearColor.Z;
    clearValues[0].color.float32[3] = desc.ClearColor.W;
    clearValues[shadow ? 0 : 1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = shadow
        ? m_Impl->ShadowRenderPasses[desc.ClearDepthEnabled ? 1U : 0U]
        : m_Impl->ColorRenderPasses[(desc.ClearColorEnabled ? 1U : 0U) |
                                    (desc.ClearDepthEnabled ? 2U : 0U)];
    begin.framebuffer = m_Impl->ActiveFramebuffer;
    begin.renderArea = {{0, 0}, {m_Impl->ActiveWidth, m_Impl->ActiveHeight}};
    begin.clearValueCount = shadow ? 1U : 2U;
    begin.pClearValues = clearValues;
    m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(m_Impl->ActiveWidth),
                              static_cast<float>(m_Impl->ActiveHeight), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, {m_Impl->ActiveWidth, m_Impl->ActiveHeight}};
    m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
    m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &scissor);
    m_Impl->CurrentPass = desc;
    m_Impl->PendingFrame.Passes.push_back({desc, {}});
    m_Impl->PassOpen = true;
    m_Impl->PipelineBound = false;
    m_Impl->WorldBound = false;
    m_Impl->ViewProjectionBound = false;
    m_Impl->LightViewProjectionBound = false;
    m_Impl->BoundVertexBuffer = {};
    m_Impl->BoundIndexBuffer = {};
    return true;
}

bool VulkanRHI::BindPipeline(const PipelineState& pipeline)
{
    if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
        return m_Impl->Fail("Vulkan pipeline must be bound inside a pass");
    if (pipeline.OutlineThickness < 0.0f) return m_Impl->Fail("Vulkan outline thickness is invalid");
    m_Impl->BoundPipeline = pipeline;
    m_Impl->PipelineBound = true;
    return true;
}

bool VulkanRHI::BindWorldTransform(const Matrix4& value)
{
    if (!m_Impl->CheckThread("BindWorldTransform") || !m_Impl->PassOpen) return false;
    m_Impl->BoundWorld = value;
    m_Impl->WorldBound = true;
    return true;
}

bool VulkanRHI::BindViewProjection(const Matrix4& value)
{
    if (!m_Impl->CheckThread("BindViewProjection") || !m_Impl->PassOpen) return false;
    m_Impl->BoundViewProjection = value;
    m_Impl->ViewProjectionBound = true;
    return true;
}

bool VulkanRHI::BindLightViewProjection(const Matrix4& value)
{
    if (!m_Impl->CheckThread("BindLightViewProjection") || !m_Impl->PassOpen) return false;
    m_Impl->BoundLightViewProjection = value;
    m_Impl->LightViewProjectionBound = true;
    return true;
}

bool VulkanRHI::BindVertexBuffer(BufferHandle buffer)
{
    if (!m_Impl->CheckThread("BindVertexBuffer") || !m_Impl->PassOpen) return false;
    const auto found = m_Impl->Buffers.find(buffer.Value);
    if (found == m_Impl->Buffers.end() || found->second.Description.Usage != BufferUsage::Vertex)
        return m_Impl->Fail("Vulkan vertex buffer is invalid");
    m_Impl->BoundVertexBuffer = buffer;
    return true;
}

bool VulkanRHI::BindIndexBuffer(BufferHandle buffer)
{
    if (!m_Impl->CheckThread("BindIndexBuffer") || !m_Impl->PassOpen) return false;
    const auto found = m_Impl->Buffers.find(buffer.Value);
    if (found == m_Impl->Buffers.end() || found->second.Description.Usage != BufferUsage::Index)
        return m_Impl->Fail("Vulkan index buffer is invalid");
    m_Impl->BoundIndexBuffer = buffer;
    return true;
}

bool VulkanRHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount,
                            std::int32_t baseVertex)
{
    if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound ||
        !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound ||
        !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
        return m_Impl->Fail("Vulkan DrawIndexed is missing required state");
    auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
    auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
    if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end() ||
        vertex->second.Bytes.size() % sizeof(Vertex) != 0 ||
        index->second.Bytes.size() % sizeof(std::uint32_t) != 0 || indexCount % 3U != 0)
        return m_Impl->Fail("Vulkan DrawIndexed requires complete triangle-list buffers");
    const std::size_t availableIndices = index->second.Bytes.size() / sizeof(std::uint32_t);
    if (firstIndex > availableIndices || indexCount > availableIndices - firstIndex)
        return m_Impl->Fail("Vulkan DrawIndexed exceeds the index buffer");
    const std::size_t availableVertices = vertex->second.Bytes.size() / sizeof(Vertex);
    for (std::size_t offset = firstIndex;
         offset < static_cast<std::size_t>(firstIndex) + indexCount; ++offset)
    {
        std::uint32_t sourceIndex = 0;
        std::memcpy(&sourceIndex, index->second.Bytes.data() + offset * sizeof(sourceIndex),
                    sizeof(sourceIndex));
        const std::int64_t resolved = static_cast<std::int64_t>(sourceIndex) + baseVertex;
        if (resolved < 0 || static_cast<std::uint64_t>(resolved) >= availableVertices)
            return m_Impl->Fail("Vulkan DrawIndexed references a vertex outside the vertex buffer");
    }

    const std::size_t alignedOffset = AlignUp(m_Impl->UniformOffset, UniformAlignment);
    if (alignedOffset > UniformBufferSize ||
        sizeof(RealtimeDrawConstants) > UniformBufferSize - alignedOffset)
        return m_Impl->Fail("Vulkan per-frame uniform arena is exhausted");

    const bool shadow = m_Impl->CurrentPass.Type == PassType::Shadow;
    std::array<VkImageView, 4> imageViews {
        m_Impl->DummyDepth.View,
        m_Impl->DummyColor.View,
        m_Impl->DummyColor.View,
        m_Impl->DummyColor.View};
    bool shadowAvailable = false;
    if (!shadow && m_Impl->BoundPipeline.SamplesShadowMap &&
        m_Impl->LastShadowTexture)
    {
        const auto texture =
            m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
        if (texture == m_Impl->Textures.end() ||
            texture->second.Description.Format != TextureFormat::Depth32 ||
            texture->second.Image.View == 0)
            return m_Impl->Fail("Vulkan shadow map resource is unavailable");
        imageViews[0] = texture->second.Image.View;
        shadowAvailable = true;
    }
    auto resolveSurfaceTexture = [&](TextureHandle handle, const char* usage,
                                     VkImageView& imageView)
    {
        if (!handle) return true;
        const auto texture = m_Impl->Textures.find(handle.Value);
        if (texture == m_Impl->Textures.end() ||
            texture->second.Description.Format != TextureFormat::RGBA8 ||
            texture->second.Image.View == 0 ||
            texture->second.Image.Layout !=
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            return m_Impl->Fail(
                std::string("Vulkan ") + usage + " texture is invalid");
        }
        imageView = texture->second.Image.View;
        return true;
    };
    if (!resolveSurfaceTexture(
            m_Impl->BoundPipeline.DiffuseTexture, "diffuse", imageViews[1]) ||
        !resolveSurfaceTexture(
            m_Impl->BoundPipeline.NormalTexture, "normal", imageViews[2]) ||
        !resolveSurfaceTexture(
            m_Impl->BoundPipeline.OpacityTexture, "opacity", imageViews[3]))
        return false;

    const RealtimeDrawConstants constants = BuildRealtimeDrawConstants(
        m_Impl->BoundPipeline,
        m_Impl->BoundWorld,
        m_Impl->BoundViewProjection,
        m_Impl->BoundLightViewProjection,
        m_Impl->CurrentPass.Type,
        shadowAvailable,
        m_Impl->BoundPipeline.DiffuseTexture != TextureHandle {},
        m_Impl->BoundPipeline.NormalTexture != TextureHandle {},
        m_Impl->BoundPipeline.OpacityTexture != TextureHandle {});
    std::memcpy(
        static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped) +
            alignedOffset,
        &constants,
        sizeof(constants));
    m_Impl->UniformOffset =
        alignedOffset + sizeof(RealtimeDrawConstants);
    if (m_Impl->DescriptorSetIndex >= m_Impl->DescriptorSets.size())
        return m_Impl->Fail("Vulkan per-frame descriptor arena is exhausted");
    const VkDescriptorSet descriptorSet =
        m_Impl->DescriptorSets[m_Impl->DescriptorSetIndex++];
    m_Impl->UpdateDescriptorSet(descriptorSet, imageViews);

    const VkPipeline pipeline = m_Impl->GetPipeline(
        m_Impl->BoundPipeline, m_Impl->CurrentPass.Type);
    if (pipeline == 0) return false;
    m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(alignedOffset);
    m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  m_Impl->PipelineLayout, 0, 1, &descriptorSet,
                                  1, &dynamicOffset);
    const VkDeviceSize vertexOffset = 0;
    m_Impl->CmdBindVertexBuffers(m_Impl->CommandBuffer, 0, 1,
                                 &vertex->second.Allocation.Buffer, &vertexOffset);
    m_Impl->CmdBindIndexBuffer(m_Impl->CommandBuffer, index->second.Allocation.Buffer,
                               0, VK_INDEX_TYPE_UINT32);
    m_Impl->CmdDrawIndexed(m_Impl->CommandBuffer, indexCount, 1, firstIndex, baseVertex, 0);

    DrawCallRecord record;
    record.VertexBuffer = m_Impl->BoundVertexBuffer;
    record.IndexBuffer = m_Impl->BoundIndexBuffer;
    record.Pipeline = m_Impl->BoundPipeline;
    record.WorldTransform = m_Impl->BoundWorld;
    record.FirstIndex = firstIndex;
    record.IndexCount = indexCount;
    record.BaseVertex = baseVertex;
    m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(record);
    FrameStatistics& statistics = m_Impl->PendingFrame.Statistics;
    ++statistics.DrawCallCount;
    statistics.SubmittedTriangleCount += indexCount / 3U;
    statistics.ShadowDrawCallCount += shadow ? 1U : 0U;
    statistics.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1U : 0U;
    return true;
}

bool VulkanRHI::EndPass()
{
    if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
        return m_Impl->Fail("Vulkan has no open pass");
    m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
    if (m_Impl->CurrentPass.Type == PassType::Shadow)
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveDepth,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    m_Impl->PassOpen = false;
    m_Impl->ActiveColor = nullptr;
    m_Impl->ActiveDepth = nullptr;
    m_Impl->ActiveFramebuffer = 0;
    return true;
}

bool VulkanRHI::Present()
{
    if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
        return m_Impl->Fail("Vulkan frame cannot be presented in the current state");
    const bool captureDepth = IsDepthCaptureRequired() || m_Impl->CreateInfo.CaptureDiagnosticBuffers ||
        m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0;
    m_Impl->CopyImageToReadback(m_Impl->MainColor, m_Impl->ColorReadback);
    if (captureDepth)
    {
        m_Impl->CopyImageToReadback(m_Impl->MainDepth, m_Impl->DepthReadback);
    }
    Impl::TextureResource* shadow = nullptr;
    if (m_Impl->CreateInfo.CaptureDiagnosticBuffers && m_Impl->LastShadowTexture)
    {
        const auto found = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
        if (found != m_Impl->Textures.end())
        {
            shadow = &found->second;
            if (!m_Impl->EnsureShadowReadback(shadow->Image.Width, shadow->Image.Height)) return false;
            m_Impl->CopyImageToReadback(shadow->Image, m_Impl->ShadowReadback);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, shadow->Image,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }
    if (!m_Impl->Check(m_Impl->EndCommandBuffer(m_Impl->CommandBuffer),
                       "vkEndCommandBuffer(frame)") ||
        !m_Impl->Check(m_Impl->ResetFences(m_Impl->Device, 1, &m_Impl->FrameFence),
                       "vkResetFences(frame)")) return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &m_Impl->CommandBuffer;
    const VkResult submitResult =
        m_Impl->QueueSubmit(m_Impl->GraphicsQueue, 1, &submit, m_Impl->FrameFence);
    if (submitResult != VK_SUCCESS)
    {
        const std::string submitError = VulkanError("vkQueueSubmit(frame)", submitResult);
        (void)m_Impl->RecreateSignaledFrameFence();
        return m_Impl->Fail(submitError);
    }
    if (!m_Impl->Check(m_Impl->WaitForFences(
            m_Impl->Device, 1, &m_Impl->FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
            "vkWaitForFences(frame completion)")) return false;

    m_Impl->PendingFrame.Statistics.PassCount =
        static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size());
    m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
    m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
    const std::size_t pixelCount = static_cast<std::size_t>(m_Impl->CreateInfo.Width) *
                                   m_Impl->CreateInfo.Height;
    m_Impl->PendingFrame.ColorRGBA8.resize(pixelCount * 4U);
    std::memcpy(m_Impl->PendingFrame.ColorRGBA8.data(), m_Impl->ColorReadback.Mapped,
                m_Impl->PendingFrame.ColorRGBA8.size());
    if (captureDepth)
    {
        m_Impl->PendingFrame.Depth32.resize(pixelCount);
        std::memcpy(m_Impl->PendingFrame.Depth32.data(), m_Impl->DepthReadback.Mapped,
                    m_Impl->PendingFrame.Depth32.size() * sizeof(float));
    }
    const bool hasDepthWritingGeometry = std::any_of(
        m_Impl->PendingFrame.Passes.begin(), m_Impl->PendingFrame.Passes.end(),
        [](const PassRecord& pass)
        {
            return pass.Description.Type != PassType::Shadow &&
                std::any_of(
                    pass.DrawCalls.begin(),
                    pass.DrawCalls.end(),
                    [](const DrawCallRecord& draw)
                    {
                        return draw.Pipeline.DepthWrite;
                    });
        });
    if (captureDepth &&
        !IsValidDepthCapture(
            m_Impl->PendingFrame.Depth32,
            hasDepthWritingGeometry))
        return m_Impl->Fail("Vulkan main depth readback is invalid");
    if (shadow != nullptr)
    {
        m_Impl->PendingFrame.ShadowWidth = shadow->Image.Width;
        m_Impl->PendingFrame.ShadowHeight = shadow->Image.Height;
        m_Impl->PendingFrame.ShadowDepth32.resize(
            static_cast<std::size_t>(shadow->Image.Width) * shadow->Image.Height);
        std::memcpy(m_Impl->PendingFrame.ShadowDepth32.data(), m_Impl->ShadowReadback.Mapped,
                    m_Impl->PendingFrame.ShadowDepth32.size() * sizeof(float));
        if (!IsValidDepthCapture(
                m_Impl->PendingFrame.ShadowDepth32,
                false))
            return m_Impl->Fail("Vulkan shadow depth readback is invalid");
    }
    m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
    m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
    m_Impl->FrameOpen = false;
    m_Impl->PresentToWindow(m_Impl->LastFrame);
    return true;
}

FrameCapture VulkanRHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
std::string VulkanRHI::GetLastError() const { return m_Impl->LastError; }

#else
struct VulkanRHI::Impl
{
    std::string LastError = "The native Vulkan Renderer backend is currently available only on Windows";
};

VulkanRHI::VulkanRHI() : m_Impl(std::make_unique<Impl>()) {}
VulkanRHI::~VulkanRHI() = default;
bool VulkanRHI::Initialize(const RHICreateInfo&) { return false; }
void VulkanRHI::Shutdown() {}
bool VulkanRHI::IsInitialized() const { return false; }
RHIBackend VulkanRHI::GetBackend() const { return RHIBackend::Vulkan; }
const char* VulkanRHI::GetName() const { return "Vulkan Native Graphics RHI Unavailable"; }
RHICapabilities VulkanRHI::GetCapabilities() const { return {}; }
bool VulkanRHI::Resize(int, int) { return false; }
BufferHandle VulkanRHI::CreateBuffer(const BufferDesc&, const void*) { return {}; }
bool VulkanRHI::UpdateBuffer(BufferHandle, std::size_t, const void*, std::size_t) { return false; }
bool VulkanRHI::DestroyBuffer(BufferHandle) { return false; }
TextureHandle VulkanRHI::CreateTexture(const TextureDesc&, const void*) { return {}; }
bool VulkanRHI::DestroyTexture(TextureHandle) { return false; }
bool VulkanRHI::BeginFrame() { return false; }
void VulkanRHI::AbortFrame() {}
bool VulkanRHI::BeginPass(const PassDesc&) { return false; }
bool VulkanRHI::BindPipeline(const PipelineState&) { return false; }
bool VulkanRHI::BindWorldTransform(const Matrix4&) { return false; }
bool VulkanRHI::BindViewProjection(const Matrix4&) { return false; }
bool VulkanRHI::BindLightViewProjection(const Matrix4&) { return false; }
bool VulkanRHI::BindVertexBuffer(BufferHandle) { return false; }
bool VulkanRHI::BindIndexBuffer(BufferHandle) { return false; }
bool VulkanRHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
bool VulkanRHI::EndPass() { return false; }
bool VulkanRHI::Present() { return false; }
FrameCapture VulkanRHI::GetLastFrameCapture() const { return {}; }
std::string VulkanRHI::GetLastError() const { return m_Impl->LastError; }
#endif
} // namespace Evans::Renderer
