#include "Renderer/Vulkan/VulkanExternalMemory.h"
#include "Renderer/DiagnosticBuffer.h"
#include "Renderer/Vulkan/VulkanPipelineCache.h"
#include "Renderer/Lumen/LumenHairReceiver.h"
#include "Renderer/Lumen/LumenHardwareRayTracing.h"
#include "Renderer/Geometry/MeshGeometry.h"
#include "Renderer/Nanite/NaniteAsync.h"
#include "Renderer/MaterialTechnique.h"
#include "Renderer/Shadows/PCFContract.h"
#include "Renderer/Lumen/LumenGlobalDistanceAsync.h"
#include "Renderer/Lumen/LumenLightingAsync.h"
#include "Renderer/Lumen/LumenCaptureAsync.h"
#include "Renderer/Nanite/NaniteMaterials.h"
#include "Renderer/Nanite/NaniteInstances.h"
#include "Renderer/Nanite/NaniteHardware.h"
#include "Renderer/MeshShaders/MeshShaders.h"
#include "Renderer/Vulkan/VulkanDeviceProperties.h"
#include "Renderer/Vulkan/VulkanRHI.h"
#include "Renderer/MaterialTexture.h"
#include "Renderer/Geometry/ParallaxOcclusionMapping.h"
#include "Renderer/Vulkan/VulkanContactShadows.h"
#include "Renderer/Vulkan/VulkanForwardPlus.h"
#include "Renderer/Vulkan/VulkanClusteredForward.h"
#include "Renderer/Lighting/LightListFrame.h"
#include "Renderer/Pipelines/TiledDeferredContract.h"
#include "Renderer/Geometry/Instancing.h"

#include "Renderer/Vulkan/VulkanABI.h"
#include "Renderer/Vulkan/VulkanRayTracingABI.h"
#include "Renderer/RealtimeShaderContract.h"
#include "Renderer/RealtimeShader.h"
#include "Renderer/TechniqueShader.h"
#include "Renderer/PostProcessing/PostProcessContract.h"
#include "Renderer/PostProcessing/HdrContract.h"
#include "Renderer/Pipelines/DeferredContract.h"
#include "Renderer/RayTracing/RayTracedAOContract.h"
#include "Renderer/Reflections/ReflectionContract.h"
#include "Renderer/Reflections/CubemapReflectionContract.h"
#include "Renderer/Reflections/EnvironmentRadiance.h"
#include "Renderer/Reflections/EnvironmentMapReflectionContract.h"
#include "Renderer/Reflections/PlanarReflectionContract.h"
#include "Renderer/PostProcessing/DepthOfFieldContract.h"
#include "Renderer/PostProcessing/MotionBlurContract.h"
#include "Renderer/PostProcessing/TemporalAAContract.h"
#include "Renderer/Lighting/GlobalIlluminationContract.h"
#include "Renderer/BakedLighting/LightProbeContract.h"
#include "Renderer/Volumetrics/VolumetricFogContract.h"
#include "Renderer/Volumetrics/VolumetricLightingContract.h"
#include "Renderer/Shadows/PCSSContract.h"
#include "Renderer/Shadows/VSMContract.h"
#include "Renderer/Volumetrics/LightShaftContract.h"
#include "Renderer/Transparency/WeightedTransparencyContract.h"
#include "Renderer/Shadows/CascadedShadowContract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include "Renderer/Vulkan/VulkanShaderPrograms.h"
#endif

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
constexpr std::uint32_t SamplerAnisotropyFeatureIndex = 19U;
constexpr std::uint32_t FragmentStoresAndAtomicsFeatureIndex = 26U;
constexpr std::uint32_t SampleRateShadingFeatureIndex = 6U;
constexpr std::uint32_t IndependentBlendFeatureIndex = 3U;
constexpr std::uint32_t SampleCountPipelineShift = 19U;
constexpr VkFlags CullBackFaces = 2U;
constexpr float MaximumHalfFloatRadiance = 65504.0f;
constexpr std::uint32_t CascadeDescriptorSetIndex = 1;
constexpr std::uint32_t InstanceDescriptorSetIndex = 2;
constexpr std::uint32_t FogDescriptorSetIndex = 3;
constexpr std::size_t InstanceUploadPageSize = 64U * 1024U;
constexpr std::size_t CascadeInstanceProgram = InstanceVertexProgramCount;
constexpr std::size_t VulkanInstanceProgramCount = InstanceVertexProgramCount + 1;
static_assert(std::size(VulkanInstanceShaders) == VulkanInstanceProgramCount, "Instance shader table changed");
constexpr std::uint32_t CascadeSamplerBinding = MaximumShadowCascades + 1;
enum class FullscreenBlend { Replace, SubtractAmbient, AddRadiance, SourceOver };
constexpr std::uint32_t CascadedDepthPipelineFlag = 1U << 18U;
static_assert(static_cast<std::uint32_t>(PassType::ForwardPlusDepth) < 16U, "Pass keys overlap the cascade pipeline flag");
constexpr std::uint32_t ForwardPlusStorageFirstBinding = 7, ForwardPlusStorageCount = 3;
constexpr std::size_t ForwardPlusFilterCount = static_cast<std::size_t>(Techniques::ShadowFilterMode::VSM) + 1;
enum ForwardPlusProgram : std::size_t { ForwardPlusSurface, ForwardPlusDepth, ForwardPlusWeighted,
    ForwardPlusContact, ForwardPlusContactMS, ForwardPlusProgramCount };

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

[[nodiscard]] bool IsValidDepthCapture(const std::vector<float>& depths)
{
    // Submission is not coverage: clipping, culling and alpha testing can leave
    // every sample at clear depth even when depth-writing draws were recorded.
    for (float depth : depths)
    {
        if (!std::isfinite(depth) || depth < 0.0f || depth > 1.0f) return false;
    }
    return !depths.empty();
}

} // namespace

using namespace VulkanABI;

struct VulkanRHI::Impl
{
    static constexpr std::uint32_t FogInputCount = 3, FogSetCount = 6;
    static constexpr std::size_t FogDisabledSet = 0, FogIntegrationSet = 1, FogCompositionSet = 2, FogSurfaceSet = 3;
    static constexpr std::size_t LumenVolumeFirstReceiverSet=4;
    static constexpr std::uint32_t LumenVolumeUniformBinding=4,LumenVolumeImageBinding=5,RayShadowImageBinding=6;
    static constexpr std::size_t FogPointSampler = 0, FogLinearSampler = 1;
    static constexpr std::uint32_t VolumeLightInputCount = 6;
    static constexpr std::uint32_t VolumeSceneLightBinding = VolumeLightInputCount + 1;
    static constexpr std::uint32_t VolumeDensityBinding = VolumeSceneLightBinding + 1;
    static constexpr std::uint32_t LightShaftInputCount = 3, LightShaftPhaseCount = 2;
    static constexpr std::uint32_t VolumeLightPassLimit = 7; // Injection plus six scan strides for 64 slices.
    // Exclusive PCSS/VSM receivers append their own bytes to the native UBO
    // prefix. Their equal size preserves descriptor ranges and set layouts.
    using ShadowFilterBytes = std::array<std::uint8_t, sizeof(VSMConstants)>;
    static_assert(sizeof(VSMConstants) == sizeof(PCSSConstants), "Shadow-filter suffix sizes differ");
    struct VolumeLightConstants
    {
        std::array<std::uint8_t,FogGridConstantBytes> Medium;
        VolumetricLightingConstants Lighting;
    };
    static_assert(offsetof(VolumeLightConstants, Lighting) == 128 &&
        offsetof(VolumetricLightingConstants, Shadows) + offsetof(VolumeLightConstants, Lighting) == 832,
        "Vulkan volume-light std140 layout changed");
    struct FilteredDrawConstants { RealtimeDrawConstants Draw; ShadowFilterBytes SoftShadows; };
    struct ForwardPlusDrawConstants { FilteredDrawConstants Surface; LightListSurfaceConstants Selection; };
    static_assert(offsetof(ForwardPlusDrawConstants, Selection) == 800 && sizeof(ForwardPlusDrawConstants) == 832,
        "Vulkan Forward+ draw std140 layout changed");
    struct TiledLightingConstants { ForwardPlusDrawConstants Surface; TiledDeferredReconstruction Reconstruction; };
    static_assert(offsetof(TiledLightingConstants, Reconstruction) == 832 && sizeof(TiledLightingConstants) == 912,
        "Vulkan tiled lighting std140 layout changed");
    struct FilteredDeferredConstants { DeferredConstants Lighting; ShadowFilterBytes SoftShadows; };
    static_assert(offsetof(FilteredDrawConstants, SoftShadows) == 688 && sizeof(FilteredDrawConstants) == 800,
        "Vulkan filtered forward std140 layout changed");
    static_assert(offsetof(FilteredDeferredConstants, SoftShadows) == 448 && sizeof(FilteredDeferredConstants) == 560,
        "Vulkan filtered deferred std140 layout changed");
    static constexpr std::size_t ReflectionConstantBytes = std::max({sizeof(ReflectionConstants), sizeof(PlanarReflectionConstants), sizeof(CubemapReflectionConstants)});
    static constexpr std::uint32_t ReflectionImageCount = 7, ReflectionPhaseCount = 2;
    static constexpr std::uint32_t MaximumEnvironmentMipLevels = 11; // 1024-pixel GUI limit.
    static constexpr std::uint32_t MaximumEnvironmentMapMipLevels = MaximumEnvironmentMipLevels + 2; // Image width is four faces.
    static constexpr std::uint32_t MaximumPlanarMipLevels = std::numeric_limits<std::uint32_t>::digits;
    // All producers and both lookup layers remain referenced by this command
    // buffer. Their descriptor ranges must never overwrite one another.
    static constexpr std::uint32_t PlanarReflectionDescriptorBase=ReflectionPhaseCount;
    static constexpr std::uint32_t SkyboxDescriptorBase=ReflectionPhaseCount*2;
    static constexpr std::uint32_t CubeMipDescriptorBase=ReflectionPhaseCount*3;
    static constexpr std::uint32_t MapMipDescriptorBase=CubeMipDescriptorBase+
        static_cast<std::uint32_t>(EnvironmentFaceCount)*(MaximumEnvironmentMipLevels-1);
    static constexpr std::uint32_t PlanarMipDescriptorBase=MapMipDescriptorBase+MaximumEnvironmentMapMipLevels;
    static constexpr std::uint32_t ReflectionDescriptorCount=PlanarMipDescriptorBase+MaximumPlanarMipLevels-1;
    static constexpr std::uint32_t CubeImageBinding = ReflectionImageCount + 3;
    static constexpr std::uint32_t ParentMipBinding = CubeImageBinding + 1;
    static constexpr std::uint32_t EnvironmentSamplerBinding = ParentMipBinding + 1;
    static constexpr std::uint32_t EnvironmentImageBinding = EnvironmentSamplerBinding + 1;
    static constexpr std::uint32_t EnvironmentParentBinding = EnvironmentImageBinding + 1;
    static constexpr std::array<std::uint32_t, 3> ReflectionSamplerBindings {
        ReflectionImageCount + 1, ReflectionImageCount + 2, EnvironmentSamplerBinding};
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
        std::uint32_t MipLevels = 1;
        std::uint32_t ArrayLayers = 1;
        std::uint32_t Samples = 1;
        VkDeviceSize AllocationBytes=0; std::uint32_t MemoryType=0;
    };

    struct InstanceUploadPage
    {
        RawBuffer Buffer;
        VkDescriptorPool Pool = 0;
        VkDescriptorSet Set = 0;
    };

    static constexpr std::size_t VSMRaw = 0, VSMHorizontal = 1, VSMFiltered = 2;
    static constexpr VkFormat VSMFormat = VK_FORMAT_R32G32_SFLOAT;
    struct VSMMomentMap
    {
        std::array<RawImage, 3> Images {};
        std::array<VkFramebuffer, 3> Framebuffers {};
        VkDescriptorPool DescriptorPool = 0;
        std::array<VkDescriptorSet, 2> FilterSets {};
        RawBuffer Readback;
        bool Ready = false, RawWritten = false;
    };

    enum class SamplePhase : std::size_t { Color, Depth, Material, Velocity, Restore, Count };
    static constexpr std::size_t SamplePhaseCount = static_cast<std::size_t>(SamplePhase::Count);
    static constexpr std::size_t SampleColor = 0, SampleVelocity = 1, SampleMaterialFirst = 2;
    static constexpr std::size_t SampleSceneFB = 0, SampleLightingFB = 1, SampleMaterialFB = 2, SampleVelocityFB = 3;
    struct MultisampleTargets
    {
        std::array<RawImage, SampleMaterialFirst + GBufferAttachmentCount> Images {};
        RawImage Depth;
        std::array<VkFramebuffer, 4> Framebuffers {};
        std::array<VkRenderPass, 4> ColorPasses {}, MaterialPasses {};
        VkRenderPass VelocityPass = 0, LightingPass = 0;
        VkDescriptorSetLayout DescriptorLayout = 0;
        VkDescriptorPool DescriptorPool = 0;
        std::array<VkDescriptorSet, SamplePhaseCount> DescriptorSets {};
        VkPipelineLayout Layout = 0;
        VkSampler Sampler = 0;
        std::array<VkPipeline, SamplePhaseCount> Pipelines {};
        std::array<std::array<VkShaderModule, 2>, SamplePhaseCount> Modules {};
        VkPipeline LightingPipeline = 0;
        std::array<VkShaderModule, 2> LightingModules {};
        VkPipeline PCSSLightingPipeline = 0;
        std::array<VkShaderModule, 2> PCSSLightingModules {};
        VkPipeline VSMLightingPipeline = 0;
        std::array<VkShaderModule, 2> VSMLightingModules {};
    } Multisample;

    struct BufferResource
    {
        BufferDesc Description;
        bool NaniteGeometry=false;
        IndexedDrawRange LastValidRange;
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
        return RecoverVirtualShadowAtlas() && RecoverLumenSurface() && RecoverLumenDistanceScene() && RecoverLumenSurfaceLookup() && RecoverLumenSurfaceLighting() && RecoverLumenRadianceTrace() && RecoverLumenProbes() && RecoverLumenFrameWork() && RecoverNaniteFrameWork();
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
               LoadInstance(GetPhysicalDeviceFormatProperties, "vkGetPhysicalDeviceFormatProperties") &&
               LoadInstance(GetPhysicalDeviceImageFormatProperties, "vkGetPhysicalDeviceImageFormatProperties") &&
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
               LoadDevice(CmdPushConstants, "vkCmdPushConstants") &&
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
               LoadDevice(CmdCopyImage, "vkCmdCopyImage") &&
               LoadDevice(CmdResolveImage, "vkCmdResolveImage") &&
               LoadDevice(CmdBeginRenderPass, "vkCmdBeginRenderPass") &&
               LoadDevice(CmdEndRenderPass, "vkCmdEndRenderPass") &&
               LoadDevice(CmdSetViewport, "vkCmdSetViewport") &&
               LoadDevice(CmdSetScissor, "vkCmdSetScissor") &&
               LoadDevice(CmdBindPipeline, "vkCmdBindPipeline") &&
               LoadDevice(CmdBindDescriptorSets, "vkCmdBindDescriptorSets") &&
               LoadDevice(CmdBindVertexBuffers, "vkCmdBindVertexBuffers") &&
               LoadDevice(CmdBindIndexBuffer, "vkCmdBindIndexBuffer") &&
               LoadDevice(CmdDrawIndexed, "vkCmdDrawIndexed") &&
               LoadDevice(CmdDraw, "vkCmdDraw");
    }

    bool CreateInstance()
    {
        VkApplicationInfo application{};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.pApplicationName = "Evans";
        application.applicationVersion = MakeVersion(1, 0, 0);
        application.pEngineName = "Evans Renderer";
        application.engineVersion = MakeVersion(1, 0, 0);
        const auto version=reinterpret_cast<PFN_vkEnumerateInstanceVersion>(GetInstanceProcAddress(nullptr,"vkEnumerateInstanceVersion"));
        std::uint32_t supported=MakeVersion(1,0,0);
        if (version && version(&supported)!=VK_SUCCESS) supported=MakeVersion(1,0,0);
        InstanceApiVersion=std::min(supported,MakeVersion(1,2,0)); application.apiVersion=InstanceApiVersion;
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
                PhysicalDevicePropertiesPrefix decodedProperties {};
                std::memcpy(&decodedProperties, properties.data(), sizeof(decodedProperties));
                const auto* prefix = &decodedProperties;
                const bool compute = (queues[queueIndex].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
                const int score = (prefix->DeviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 100 : 10) + (compute ? 1 : 0);
                if (score > bestScore)
                {
                    bestScore = score;
                    PhysicalDevice = candidate;
                    QueueFamilyIndex = queueIndex;
                    FrameTimeBits=queues[queueIndex].timestampValidBits;
                    GraphicsQueueHasCompute = compute;
                    DeviceApiVersion=prefix->ApiVersion;
                    DeviceName = prefix->DeviceName;
                    DeviceLimits = prefix->Limits;
                }
            }
        }
        if (PhysicalDevice == nullptr)
            return Fail("Vulkan found no graphics device with fillModeNonSolid support");
        GetPhysicalDeviceMemoryProperties(PhysicalDevice, &MemoryProperties);
        VkPhysicalDeviceFeatures available {};
        GetPhysicalDeviceFeatures(PhysicalDevice, &available);
        AnisotropicFilteringSupported = available.values[SamplerAnisotropyFeatureIndex] != VK_FALSE_VALUE;
        SampleRateShadingSupported = available.values[SampleRateShadingFeatureIndex] != VK_FALSE_VALUE;
        IndependentBlendSupported = available.values[IndependentBlendFeatureIndex] != VK_FALSE_VALUE;
        FragmentStoresAndAtomicsSupported=available.values[FragmentStoresAndAtomicsFeatureIndex]!=VK_FALSE_VALUE;
        SupportedSampleCounts = SampleRateShadingSupported ? Techniques::AllMultisampleCounts : 1;
        // Query the exact sampled attachment usages; color/depth global limits
        // alone do not establish support for all material and velocity formats.
        for (auto format : {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT})
        {
            VkImageFormatProperties limits {};
            const VkFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | (format == VK_FORMAT_D32_SFLOAT
                ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            if (GetPhysicalDeviceImageFormatProperties(PhysicalDevice, format, VK_IMAGE_TYPE_2D,
                VK_IMAGE_TILING_OPTIMAL, usage, 0, &limits) != VK_SUCCESS) SupportedSampleCounts = 1;
            else SupportedSampleCounts &= limits.sampleCounts;
        }
        SupportedSampleCounts |= 1;
        // Both statistics must blend and remain sampleable at the scene sample
        // count. Independent blending is a device feature, not a format property.
        WeightedSampleCounts = IndependentBlendSupported && DeviceLimits.maxColorAttachments >= 2 &&
            DeviceLimits.maxFragmentOutputAttachments >= 2 ? SupportedSampleCounts : 0;
        constexpr VkFlags weightedFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
        for (std::size_t index = 0; index < WeightedFormats.size(); ++index)
        {
            VkFormatProperties properties {};
            GetPhysicalDeviceFormatProperties(PhysicalDevice, WeightedFormats[index], &properties);
            if ((properties.optimalTilingFeatures & weightedFeatures) != weightedFeatures ||
                GetPhysicalDeviceImageFormatProperties(PhysicalDevice, WeightedFormats[index], VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    0, &WeightedLimits[index]) != VK_SUCCESS) WeightedSampleCounts = 0;
            else WeightedSampleCounts &= WeightedLimits[index].sampleCounts;
        }
        VkFormatProperties material{}, depth{};
        GetPhysicalDeviceFormatProperties(PhysicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT, &material);
        GetPhysicalDeviceFormatProperties(PhysicalDevice, VK_FORMAT_D32_SFLOAT, &depth);
        constexpr VkFlags materialFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
        constexpr VkFlags depthFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
        DeferredSupported = DeviceLimits.maxColorAttachments >= GBufferAttachmentCount &&
            DeviceLimits.maxFragmentOutputAttachments >= GBufferAttachmentCount &&
            (material.optimalTilingFeatures & materialFeatures) == materialFeatures &&
            (depth.optimalTilingFeatures & depthFeatures) == depthFeatures;
        VkFormatProperties linearDepth {};
        GetPhysicalDeviceFormatProperties(PhysicalDevice, VK_FORMAT_R32_SFLOAT, &linearDepth);
        VkImageFormatProperties contactLimits {};
        constexpr VkFlags contactFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        constexpr VkFlags contactUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        constexpr std::uint32_t ContactWorkgroupEdge = 8, ContactSampleCapacity = 8;
        ContactSupported = GraphicsQueueHasCompute && DeviceLimits.standardSampleLocations &&
            DeviceLimits.maxComputeWorkGroupInvocations >= ContactWorkgroupEdge * ContactWorkgroupEdge &&
            DeviceLimits.maxComputeWorkGroupSize[0] >= ContactWorkgroupEdge &&
            DeviceLimits.maxComputeWorkGroupSize[1] >= ContactWorkgroupEdge && DeviceLimits.maxPerStageDescriptorStorageImages >= 1 &&
            (depth.optimalTilingFeatures & depthFeatures) == depthFeatures &&
            (linearDepth.optimalTilingFeatures & contactFeatures) == contactFeatures &&
            GetPhysicalDeviceImageFormatProperties(PhysicalDevice, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TYPE_2D,
                VK_IMAGE_TILING_OPTIMAL, contactUsage, 0, &contactLimits) == VK_SUCCESS && contactLimits.maxArrayLayers >= ContactSampleCapacity;
        ForwardPlusSupported = GraphicsQueueHasCompute &&
            DeviceLimits.maxUniformBufferRange >= sizeof(ForwardPlusDrawConstants) &&
            DeviceLimits.maxStorageBufferRange >= MaximumForwardPlusLights * sizeof(ShaderLight) &&
            DeviceLimits.maxPerStageDescriptorStorageBuffers >= ForwardPlusStorageCount &&
            DeviceLimits.maxDescriptorSetStorageBuffers >= ForwardPlusStorageCount &&
            DeviceLimits.maxFragmentCombinedOutputResources >= ForwardPlusStorageCount + 2 &&
            (depth.optimalTilingFeatures & depthFeatures) == depthFeatures;
        constexpr VkFlags linearDepthFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
        // The image-format query validates transfer usages even on Vulkan 1.0,
        // where the later TRANSFER_SRC/DST format-feature bits are unavailable.
        constexpr VkFlags linearDepthUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        SaoSupported = DeferredSupported &&
            (linearDepth.optimalTilingFeatures & linearDepthFeatures) == linearDepthFeatures &&
            GetPhysicalDeviceImageFormatProperties(PhysicalDevice, VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, linearDepthUsage, 0, &SaoImageLimits) == VK_SUCCESS;
        constexpr VkFlags temporalColorUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        constexpr VkFlags temporalDepthUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        // The shaft mask needs one attachment and point reads. The shared HDR
        // tone resolve also samples RGBA16F with the existing linear sampler.
        constexpr VkFlags shaftColorFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
        LightShaftSupported = DeviceLimits.maxBoundDescriptorSets >= 1 &&
            DeviceLimits.maxUniformBufferRange >= sizeof(LightShaftConstants) &&
            DeviceLimits.maxPerStageDescriptorSamplers >= LightShaftInputCount &&
            DeviceLimits.maxPerStageDescriptorSampledImages >= LightShaftInputCount &&
            (material.optimalTilingFeatures & shaftColorFeatures) == shaftColorFeatures &&
            (depth.optimalTilingFeatures & depthFeatures) == depthFeatures &&
            GetPhysicalDeviceImageFormatProperties(PhysicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, temporalColorUsage, 0, &LightShaftImageLimits) == VK_SUCCESS;
        FogSupported = DeviceLimits.maxBoundDescriptorSets > FogDescriptorSetIndex &&
            (material.optimalTilingFeatures & materialFeatures) == materialFeatures &&
            (depth.optimalTilingFeatures & depthFeatures) == depthFeatures &&
            GetPhysicalDeviceImageFormatProperties(PhysicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, temporalColorUsage, 0, &FogImageLimits) == VK_SUCCESS;
        VkFormatProperties volumeTransport {};
        GetPhysicalDeviceFormatProperties(PhysicalDevice, VK_FORMAT_R32G32B32A32_SFLOAT, &volumeTransport);
        constexpr VkFlags volumeFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
        constexpr VkFlags volumeUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        VolumeLightSupported = FogSupported && DeviceLimits.maxUniformBufferRange >= sizeof(VolumeLightConstants) &&
            (volumeTransport.optimalTilingFeatures & volumeFeatures) == volumeFeatures &&
            GetPhysicalDeviceImageFormatProperties(PhysicalDevice, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, volumeUsage, 0, &VolumeLightImageLimits) == VK_SUCCESS;
        VkFormatProperties moments {};
        GetPhysicalDeviceFormatProperties(PhysicalDevice, VSMFormat, &moments);
        const VkFlags momentUsage = volumeUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VSMSupported = DeviceLimits.maxUniformBufferRange >= sizeof(FilteredDrawConstants) &&
            (moments.optimalTilingFeatures & volumeFeatures) == volumeFeatures &&
            GetPhysicalDeviceImageFormatProperties(PhysicalDevice, VSMFormat, VK_IMAGE_TYPE_2D,
                VK_IMAGE_TILING_OPTIMAL, momentUsage, 0, &VSMImageLimits) == VK_SUCCESS;
        TemporalSupported = DeviceLimits.maxColorAttachments >= TemporalAttachmentCount &&
            DeviceLimits.maxFragmentOutputAttachments >= TemporalAttachmentCount &&
            (material.optimalTilingFeatures & materialFeatures) == materialFeatures &&
            (depth.optimalTilingFeatures & depthFeatures) == depthFeatures &&
            (linearDepth.optimalTilingFeatures & linearDepthFeatures) == linearDepthFeatures &&
            GetPhysicalDeviceImageFormatProperties(PhysicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, temporalColorUsage, 0, &TemporalColorLimits) == VK_SUCCESS &&
            GetPhysicalDeviceImageFormatProperties(PhysicalDevice, VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, temporalDepthUsage, 0, &TemporalDepthLimits) == VK_SUCCESS;
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
        features.values[SamplerAnisotropyFeatureIndex] = AnisotropicFilteringSupported ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        features.values[SampleRateShadingFeatureIndex] = SampleRateShadingSupported ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        features.values[IndependentBlendFeatureIndex] = IndependentBlendSupported ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        features.values[FragmentStoresAndAtomicsFeatureIndex]=FragmentStoresAndAtomicsSupported?VK_TRUE_VALUE:VK_FALSE_VALUE;
        VkDeviceCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create.queueCreateInfoCount = 1;
        create.pQueueCreateInfos = &queue;
        create.pEnabledFeatures = &features;
        QueryMeshShaderSupport();
        QueryLumenHardwareSupport();
        std::vector<const char*> extensions {"VK_KHR_timeline_semaphore","VK_KHR_external_semaphore_win32","VK_KHR_external_semaphore","VK_KHR_external_memory_win32","VK_KHR_external_memory","VK_KHR_dedicated_allocation","VK_KHR_get_memory_requirements2"};
        VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
        if (MeshStagesEnabled)
        {
            mesh.meshShader=VK_TRUE_VALUE; mesh.taskShader=MeshTaskStagesEnabled; create.pNext=&mesh;
            extensions.push_back("VK_EXT_mesh_shader"); if (MeshNativeBarycentrics) extensions.push_back("VK_KHR_fragment_shader_barycentric");
            if (MeshNativeBarycentrics) { bary.fragmentShaderBarycentric=VK_TRUE_VALUE; mesh.pNext=&bary; }
        }
        VkPhysicalDeviceRayQueryFeaturesKHR query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,&query};
        VkPhysicalDeviceBufferDeviceAddressFeatures address{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,&acceleration};
        if (LumenHardwareEnabled)
        {
            for (auto extension:{"VK_KHR_acceleration_structure","VK_KHR_deferred_host_operations","VK_KHR_ray_query"}) extensions.push_back(extension);
            query.rayQuery=acceleration.accelerationStructure=address.bufferDeviceAddress=VK_TRUE_VALUE;
            query.pNext=const_cast<void*>(create.pNext); create.pNext=&address;
        }
        TimelineSemaphoreFeatures timeline;timeline.pNext=create.pNext;create.pNext=&timeline;
        create.enabledExtensionCount=unsigned(extensions.size()); create.ppEnabledExtensionNames=extensions.data();
        if (!Check(CreateDeviceFunction(PhysicalDevice, &create, nullptr, &Device),
                   "vkCreateDevice")) return false;
        if (!LoadDeviceFunctions()) return false;
        InitializeVulkanPipelineCache(Device,PhysicalDevice,GetPhysicalDeviceProperties,GetDeviceProcAddress);
        if (MeshStagesEnabled && (!LoadDevice(DrawMeshTasks,"vkCmdDrawMeshTasksEXT") || !LoadDevice(DrawMeshTasksIndirect,"vkCmdDrawMeshTasksIndirectEXT"))) return false;
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
        // GPU cluster compaction copies native index streams without host mirrors.
        create.usage = usage|((usage&VK_BUFFER_USAGE_INDEX_BUFFER_BIT)?VK_BUFFER_USAGE_TRANSFER_SRC_BIT:0);
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!Check(CreateBufferFunction(Device, &create, nullptr, &output.Buffer),
                   "vkCreateBuffer")) return false;
        VkMemoryRequirements requirements{};
        GetBufferMemoryRequirements(Device, output.Buffer, &requirements);
        const auto hostType = FindMemoryType(requirements.memoryTypeBits, properties);
        auto memoryType = hostType;
        // Prefer mapped VRAM for geometry read on every draw. Some discrete
        // devices expose only a small host-visible device-local heap.
        if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (usage & (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT)))
        {
            const auto local = FindMemoryType(requirements.memoryTypeBits, properties | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (local != std::numeric_limits<std::uint32_t>::max()) memoryType = local;
        }
        if (memoryType == std::numeric_limits<std::uint32_t>::max())
        {
            DestroyRawBuffer(output);
            return Fail("Vulkan found no compatible buffer memory type");
        }
        VkMemoryAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memoryType;
        const VkMemoryAllocateFlagsInfo addressFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,nullptr,VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,0};
        if (usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) allocate.pNext=&addressFlags;
        VkResult allocation = AllocateMemory(Device, &allocate, nullptr, &output.Memory);
        if (allocation == VK_ERROR_OUT_OF_DEVICE_MEMORY && memoryType != hostType)
        {
            allocate.memoryTypeIndex = hostType;
            allocation = AllocateMemory(Device, &allocate, nullptr, &output.Memory);
        }
        if (!Check(allocation, "vkAllocateMemory(buffer)") ||
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
                        VkFlags aspect, VkFlags usage, RawImage& output, std::uint32_t mipLevels = 1, std::uint32_t samples = 1,
                        std::uint32_t arrayLayers = 1, bool cube = false, bool external = false)
    {
        VkImageCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ExternalImageCreateInfo externalImage; create.pNext=external?&externalImage:nullptr;
        create.imageType = VK_IMAGE_TYPE_2D;
        create.format = format;
        create.extent = {width, height, 1};
        create.mipLevels = mipLevels;
        create.arrayLayers = arrayLayers;
        create.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        create.samples = samples;
        create.tiling = VK_IMAGE_TILING_OPTIMAL;
        // GPU cluster compaction copies native index streams without host mirrors.
        create.usage = usage|((usage&VK_BUFFER_USAGE_INDEX_BUFFER_BIT)?VK_BUFFER_USAGE_TRANSFER_SRC_BIT:0);
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
        DedicatedImageMemoryInfo dedicated; dedicated.image=output.Image;
        ExportMemoryInfo exportInfo; exportInfo.pNext=&dedicated;
        allocate.pNext=external?&exportInfo:nullptr;
        output.AllocationBytes=requirements.size;output.MemoryType=memoryType;
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
        view.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : (arrayLayers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
        view.format = format;
        view.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        view.subresourceRange = {aspect, 0, mipLevels, 0, arrayLayers};
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
        output.MipLevels = mipLevels;
        output.ArrayLayers = arrayLayers;
        output.Samples = samples;
        return true;
    }

    struct GpuSurface
    {
        RawImage Image;
        std::shared_ptr<GpuImage> Published;
        std::shared_ptr<void> MemoryOwner;
    };
    std::vector<GpuSurface> GpuSurfaces;
    std::uint64_t PendingGpuSignal=0,GpuSignalValue=1;
    TimelineSemaphoreSubmit GpuTimelineSubmit;
    void CompleteGpuSignal()
    {
        if(PendingGpuSignal)
        {
            const auto destroy=reinterpret_cast<DestroyExternalSemaphore>(GetDeviceProcAddress(Device,"vkDestroySemaphore"));
            destroy(Device,PendingGpuSignal,nullptr);PendingGpuSignal=0;
        }
    }
    void AttachGpuSignal(VkSubmitInfo& submit)
    {
        if(PendingGpuSignal)
        {
            GpuTimelineSubmit.signalValueCount=1;GpuTimelineSubmit.signalValues=&GpuSignalValue;
            GpuTimelineSubmit.pNext=submit.pNext;submit.pNext=&GpuTimelineSubmit;
            submit.signalSemaphoreCount=1;submit.pSignalSemaphores=&PendingGpuSignal;
        }
    }

    bool PublishGpuColor(RawImage& source)
    {
        GpuSurface* surface=nullptr;
        for(auto& item:GpuSurfaces) if(item.Published.use_count()==1 && item.Image.Width==source.Width && item.Image.Height==source.Height) {surface=&item;break;}
        if(!surface)
        {
            GpuSurfaces.erase(std::remove_if(GpuSurfaces.begin(),GpuSurfaces.end(),[this](auto& item){
                if(item.Published.use_count()!=1)return false;DestroyRawImage(item.Image);return true;
            }),GpuSurfaces.end());
            GpuSurface item;
            if(!CreateRawImage(source.Width,source.Height,source.Format,VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,item.Image,1,1,1,false,true))return false;
            const auto getHandle=reinterpret_cast<GetMemoryWin32Handle>(GetDeviceProcAddress(Device,"vkGetMemoryWin32HandleKHR"));
            void* handle=nullptr;GetWin32MemoryInfo info;info.memory=item.Image.Memory;
            if(!getHandle || !Check(getHandle(Device,&info,&handle),"Export Vulkan viewport memory")){DestroyRawImage(item.Image);return false;}
            item.Published=std::make_shared<GpuImage>(); item.Published->Backend=RHIBackend::Vulkan;
            item.Published->SharedHandle=handle;
            item.Published->Owner=std::shared_ptr<void>(handle,[](void* h){if(h)CloseHandle(h);});
            item.MemoryOwner=item.Published->Owner;
            item.Published->AllocationBytes=item.Image.AllocationBytes;item.Published->MemoryType=item.Image.MemoryType;item.Published->Format=source.Format;
            GpuSurfaces.push_back(std::move(item));surface=&GpuSurfaces.back();
        }
        if(PendingGpuSignal)
        {
            if(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,0)!=VK_SUCCESS)return Fail("A previous GPU viewport submission has not retired");
            CompleteGpuSignal();
        }
        const auto createSemaphore=reinterpret_cast<CreateExternalSemaphore>(GetDeviceProcAddress(Device,"vkCreateSemaphore"));
        const auto exportSemaphore=reinterpret_cast<GetSemaphoreWin32Handle>(GetDeviceProcAddress(Device,"vkGetSemaphoreWin32HandleKHR"));
        TimelineSemaphoreType timeline;ExportSemaphoreInfo exported;exported.pNext=&timeline;ExternalSemaphoreCreateInfo create;create.pNext=&exported;
        if(!createSemaphore || !exportSemaphore || !Check(createSemaphore(Device,&create,nullptr,&PendingGpuSignal),"Create viewport ready semaphore"))return false;
        GetWin32SemaphoreInfo ready;ready.semaphore=PendingGpuSignal;void* handle=nullptr;
        if(!Check(exportSemaphore(Device,&ready,&handle),"Export viewport readiness")){CompleteGpuSignal();return false;}
        surface->Published->ReadyHandle=handle;
        surface->Published->Owner=std::shared_ptr<void>(handle,[memory=surface->MemoryOwner](void* h){if(h)CloseHandle(h);});
        auto& target=surface->Image;
        if(FrameOpen && !FrameSubmitted)FrameImageLayouts.emplace_back(&target,target.Layout);
        VkImageMemoryBarrier acquire{};acquire.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acquire.srcQueueFamilyIndex=target.Layout==VK_IMAGE_LAYOUT_UNDEFINED?VK_QUEUE_FAMILY_IGNORED:ExternalQueueFamily;
        acquire.dstQueueFamilyIndex=target.Layout==VK_IMAGE_LAYOUT_UNDEFINED?VK_QUEUE_FAMILY_IGNORED:QueueFamilyIndex;
        acquire.oldLayout=target.Layout;acquire.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        acquire.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;acquire.image=target.Image;acquire.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&acquire);
        TransitionImage(CommandBuffer,source,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkImageCopy copy{};copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.extent={source.Width,source.Height,1};
        CmdCopyImage(CommandBuffer,source.Image,source.Layout,target.Image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
        auto release=acquire;release.srcQueueFamilyIndex=QueueFamilyIndex;release.dstQueueFamilyIndex=ExternalQueueFamily;
        release.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;release.newLayout=VK_IMAGE_LAYOUT_GENERAL;
        release.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;release.dstAccessMask=0;
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,nullptr,0,nullptr,1,&release);
        target.Layout=VK_IMAGE_LAYOUT_GENERAL;PendingFrame.GpuColor=surface->Published;return true;
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

    void LayoutSource(VkImageLayout layout, VkFlags& stage, VkFlags& access)
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
            stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | (GraphicsQueueHasCompute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0);
            access = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                (GraphicsQueueHasCompute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0);
            // Read-only depth tests still end with STORE, whose memory access is
            // a depth attachment write. Include it before the next transition.
            // https://docs.vulkan.org/refpages/latest/refpages/source/VkAttachmentStoreOp.html
            access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_SHADER_READ_BIT;
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

    void LayoutDestination(VkImageLayout layout, VkFlags& stage, VkFlags& access)
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
        // Read-only depth still has attachment LOAD/STORE accesses. A contact
        // dispatch may have just sampled it without changing this layout, so
        // publish those uses before the next depth-testing render pass as well.
        if (image.Layout == layout && layout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) return;
        // Recording is speculative until submission. Preserve each image's
        // previous layout so cancellation cannot publish an unexecuted barrier.
        if (FrameOpen && !FrameSubmitted && commandBuffer == CommandBuffer &&
            std::none_of(FrameImageLayouts.begin(), FrameImageLayouts.end(),
                [&image](const auto& entry) { return entry.first == &image; }))
            FrameImageLayouts.emplace_back(&image, image.Layout);
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
        barrier.subresourceRange = {image.Aspect, 0, image.MipLevels, 0, image.ArrayLayers};
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
        if (!ResetForwardPlusAfterDiscard()) return false;
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

    bool InitializeImage(RawImage& image, const void* data, std::size_t size, int mipLevels = 1)
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
        TextureDesc layout;
        layout.Width = static_cast<int>(image.Width);
        layout.Height = static_cast<int>(image.Height);
        layout.MipLevels = mipLevels;
        std::vector<VkBufferImageCopy> copies(mipLevels);
        VisitTextureMips(layout, [&](int level, int width, int height, std::size_t offset) {
            copies[level].bufferOffset = offset;
            copies[level].imageSubresource = {image.Aspect, static_cast<std::uint32_t>(level), 0, 1};
            copies[level].imageExtent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1};
        });
        CmdCopyBufferToImage(CommandBuffer, staging.Buffer, image.Image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(copies.size()), copies.data());
        TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const bool result = EndImmediate();
        DestroyRawBuffer(staging);
        return result;
    }

    bool CreateColorRenderPass(bool clearColor, bool clearDepth, VkRenderPass& output,
        VkFormat format = VK_FORMAT_R8G8B8A8_SRGB, bool readOnlyDepth = false, std::uint32_t samples = 1)
    {
        std::array<VkAttachmentDescription, 2> attachments{};
        attachments[0].format = format;
        attachments[0].samples = samples;
        attachments[0].loadOp = clearColor ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = 2;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = samples;
        attachments[1].loadOp = clearDepth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = 2;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        const VkImageLayout depthLayout = readOnlyDepth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments[1].initialLayout = attachments[1].finalLayout = depthLayout;
        const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        const VkAttachmentReference depth{1, depthLayout};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color;
        subpass.pDepthStencilAttachment = &depth;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = dependency.srcStageMask;
        // Lighting and subsequent transparency keep the same color layout.
        // LOAD/blending still need visibility of the preceding attachment writes.
        dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask = dependency.srcAccessMask | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        // STORE remains a write even when fragment depth writes are disabled.
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

    void DestroyMultisampleTargets(MultisampleTargets& targets)
    {
        if (&targets==&Multisample) { DestroyRayShadowPrograms(); DestroyPCFPrograms(); }
        for (auto framebuffer : targets.Framebuffers)
            if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, framebuffer, nullptr);
        for (auto pipeline : targets.Pipelines)
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
        if (targets.LightingPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, targets.LightingPipeline, nullptr);
        if (targets.PCSSLightingPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, targets.PCSSLightingPipeline, nullptr);
        if (targets.VSMLightingPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, targets.VSMLightingPipeline, nullptr);
        for (auto module : targets.VSMLightingModules)
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
        for (auto module : targets.PCSSLightingModules)
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
        for (auto module : targets.LightingModules)
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
        for (auto& pair : targets.Modules) for (auto module : pair)
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
        for (auto pass : targets.ColorPasses)
            if (pass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, pass, nullptr);
        for (auto pass : targets.MaterialPasses)
            if (pass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, pass, nullptr);
        for (auto pass : {targets.VelocityPass, targets.LightingPass})
            if (pass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, pass, nullptr);
        if (targets.DescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, targets.DescriptorPool, nullptr);
        if (targets.Layout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, targets.Layout, nullptr);
        if (targets.DescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, targets.DescriptorLayout, nullptr);
        if (targets.Sampler && DestroySamplerFunction) DestroySamplerFunction(Device, targets.Sampler, nullptr);
        for (auto& image : targets.Images) DestroyRawImage(image);
        DestroyRawImage(targets.Depth);
        targets = {};
    }

    bool BuildMultisampleTargets(MultisampleTargets& target, bool material)
    {
        const auto width = MainColor.Width, height = MainColor.Height;
        const auto format = HdrActive ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_SRGB;
        constexpr VkFlags ColorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (!CreateRawImage(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, target.Depth, 1, FrameSampleCount) ||
            !CreateRawImage(width, height, format, VK_IMAGE_ASPECT_COLOR_BIT, ColorUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, target.Images[SampleColor], 1, FrameSampleCount)) return false;
        if (VelocityActive && !CreateRawImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT, ColorUsage, target.Images[SampleVelocity], 1, FrameSampleCount)) return false;
        if (material) for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
            if (!CreateRawImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                ColorUsage, target.Images[SampleMaterialFirst + index], 1, FrameSampleCount)) return false;
        for (std::size_t mask = 0; mask < target.ColorPasses.size(); ++mask)
        {
            if (!CreateColorRenderPass((mask & 1U) != 0, (mask & 2U) != 0, target.ColorPasses[mask], format, false, FrameSampleCount)) return false;
            if (material && !CreateDeferredRenderPass((mask & 1U) != 0, (mask & 2U) != 0, target.MaterialPasses[mask], FrameSampleCount)) return false;
        }
        const std::array<VkImageView, 2> scene {target.Images[SampleColor].View, target.Depth.View};
        if (!CreateFramebuffer(target.ColorPasses[0], scene.data(), static_cast<std::uint32_t>(scene.size()), width, height, target.Framebuffers[SampleSceneFB]) ||
            !CreateFullscreenRenderPass(format, target.LightingPass, false, FrameSampleCount) ||
            !CreateFramebuffer(target.LightingPass, scene.data(), 1, width, height, target.Framebuffers[SampleLightingFB])) return false;
        if (material)
        {
            std::array<VkImageView, GBufferAttachmentCount + 1> views {};
            for (std::size_t index = 0; index < GBufferAttachmentCount; ++index) views[index] = target.Images[SampleMaterialFirst + index].View;
            views.back() = target.Depth.View;
            if (!CreateFramebuffer(target.MaterialPasses[0], views.data(), static_cast<std::uint32_t>(views.size()), width, height, target.Framebuffers[SampleMaterialFB])) return false;
        }
        if (VelocityActive)
        {
            const std::array<VkImageView, 2> views {target.Images[SampleVelocity].View, target.Depth.View};
            if (!CreateColorRenderPass(true, false, target.VelocityPass, VK_FORMAT_R16G16B16A16_SFLOAT, true, FrameSampleCount) ||
                !CreateFramebuffer(target.VelocityPass, views.data(), static_cast<std::uint32_t>(views.size()), width, height, target.Framebuffers[SampleVelocityFB])) return false;
        }
        // Each resolve phase owns a stable descriptor set. The two color resolves
        // reuse identical bindings without updating a set referenced by commands.
        constexpr std::uint32_t ImageCount = 9;
        std::array<VkDescriptorSetLayoutBinding, ImageCount + 2> bindings {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        for (std::uint32_t index = 1; index <= ImageCount; ++index)
            bindings[index] = {index, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings.back() = {ImageCount + 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout {};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size()); layout.pBindings = bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device, &layout, nullptr, &target.DescriptorLayout), "vkCreateDescriptorSetLayout(MSAA)")) return false;
        VkPipelineLayoutCreateInfo pipelineLayout {};
        pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayout.setLayoutCount = 1; pipelineLayout.pSetLayouts = &target.DescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device, &pipelineLayout, nullptr, &target.Layout), "vkCreatePipelineLayout(MSAA)")) return false;
        constexpr auto PhaseCount = static_cast<std::uint32_t>(SamplePhaseCount);
        const std::array<VkDescriptorPoolSize, 3> sizes {{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, PhaseCount},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, ImageCount * PhaseCount}, {VK_DESCRIPTOR_TYPE_SAMPLER, PhaseCount}}};
        VkDescriptorPoolCreateInfo pool {};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = PhaseCount; pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size()); pool.pPoolSizes = sizes.data();
        if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &target.DescriptorPool), "vkCreateDescriptorPool(MSAA)")) return false;
        std::array<VkDescriptorSetLayout, SamplePhaseCount> layouts; layouts.fill(target.DescriptorLayout);
        VkDescriptorSetAllocateInfo allocate {};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = target.DescriptorPool; allocate.descriptorSetCount = PhaseCount; allocate.pSetLayouts = layouts.data();
        if (!Check(AllocateDescriptorSets(Device, &allocate, target.DescriptorSets.data()), "vkAllocateDescriptorSets(MSAA)")) return false;
        VkSamplerCreateInfo sampler {};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.minFilter = sampler.magFilter = VK_FILTER_NEAREST;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        return Check(CreateSamplerFunction(Device, &sampler, nullptr, &target.Sampler), "vkCreateSampler(MSAA)");
    }

    bool EnsureMultisampleTargets()
    {
        const bool material = Hair.Active || GIActive || DeferredActive || AmbientOcclusionActive || ReflectionActive;
        const auto format = HdrActive ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_SRGB;
        if (!Multisample.Depth.Image || Multisample.Depth.Width != MainColor.Width || Multisample.Depth.Height != MainColor.Height ||
            Multisample.Depth.Samples != FrameSampleCount || Multisample.Images[SampleColor].Format != format ||
            (material && !Multisample.Framebuffers[SampleMaterialFB]) || (VelocityActive && !Multisample.Framebuffers[SampleVelocityFB]))
        {
            MultisampleTargets candidate;
            if (!BuildMultisampleTargets(candidate, material)) { DestroyMultisampleTargets(candidate); return false; }
            DestroyWeightedTargets();
            DestroyMultisampleTargets(Multisample);
            Multisample = candidate;
        }
        return !DeferredActive || CreateFullscreenPipeline(VulkanDeferredMultisampleShaders, Multisample.LightingModules,
            Multisample.LightingPass, Multisample.LightingPipeline, DeferredPipelineLayout, FullscreenBlend::Replace, false, FrameSampleCount, true) != 0;
    }

    bool DrawSampleResolve(SamplePhase phase)
    {
        const auto index = static_cast<std::size_t>(phase);
        const bool material = phase == SamplePhase::Material, velocity = phase == SamplePhase::Velocity;
        const bool restore = phase == SamplePhase::Restore;
        const bool depthWrite = material || phase == SamplePhase::Depth;
        auto* color = restore ? &Multisample.Images[SampleColor] : (HdrActive ? &HdrImages[0] : &MainColor);
        auto* depth = restore ? &Multisample.Depth : &MainDepth;
        auto framebuffer = restore ? Multisample.Framebuffers[SampleSceneFB] : (HdrActive ? HdrSceneFramebuffer : MainFramebuffer);
        auto pass = restore ? Multisample.ColorPasses[0] : (HdrActive ? HdrColorRenderPasses[0] : ColorRenderPasses[0]);
        if (material) { framebuffer = GBufferFramebuffer; pass = DeferredRenderPasses[0]; }
        if (velocity) { color = &MotionImages[MotionVelocity]; framebuffer = MotionFramebuffers[MotionVelocity]; pass = MotionGeometryRenderPass; }
        if (!CreateFullscreenPipeline(VulkanMSAAShaders[index], Multisample.Modules[index], pass, Multisample.Pipelines[index],
            Multisample.Layout, FullscreenBlend::Replace, false, restore ? FrameSampleCount : 1, false,
            material ? static_cast<std::uint32_t>(GBufferAttachmentCount) : 1, depthWrite, phase != SamplePhase::Depth)) return false;
        std::array<RawImage*, 9> inputs;
        inputs.fill(&Multisample.Images[SampleColor]);
        inputs[1] = &Multisample.Depth;
        for (std::size_t i = 0; i < GBufferAttachmentCount; ++i)
            if (Multisample.Images[SampleMaterialFirst + i].Image) inputs[2 + i] = &Multisample.Images[SampleMaterialFirst + i];
        if (Multisample.Images[SampleVelocity].Image) inputs[7] = &Multisample.Images[SampleVelocity];
        inputs[8] = HdrActive ? &HdrImages[0] : &MainColor;
        for (std::size_t input = 0; input < inputs.size(); ++input)
        {
            const bool sampled = (phase == SamplePhase::Color && input == 0) ||
                ((depthWrite || velocity) && input == 1) || (material && input >= 2 && input <= 6) ||
                (velocity && input == 7) || (restore && input == 8);
            if (sampled) TransitionImage(CommandBuffer, *inputs[input], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        const auto offset = AlignUp(UniformOffset, UniformAlignment);
        const std::array<std::uint32_t, 4> counts {FrameSampleCount, 0, 0, 0};
        if (offset > UniformBufferSize || sizeof(counts) > UniformBufferSize - offset) return Fail("Vulkan MSAA uniform arena is exhausted");
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset, counts.data(), sizeof(counts));
        UniformOffset = offset + sizeof(counts);
        const auto set = Multisample.DescriptorSets[index];
        if (!SampleDescriptorsWritten[index])
        {
            std::array<VkDescriptorImageInfo, 10> images {};
            std::array<VkWriteDescriptorSet, 11> writes {};
            const VkDescriptorBufferInfo buffer {UniformBuffer.Buffer, 0, sizeof(counts)};
            for (std::size_t binding = 0; binding < writes.size(); ++binding)
            {
                auto& write = writes[binding];
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = set; write.dstBinding = static_cast<std::uint32_t>(binding); write.descriptorCount = 1;
                if (!binding) { write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo = &buffer; }
                else
                {
                    auto& image = images[binding - 1];
                    if (binding <= inputs.size())
                    {
                        image.imageView = inputs[binding - 1]->View; image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    }
                    else { image.sampler = Multisample.Sampler; write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; }
                    write.pImageInfo = &image;
                }
            }
            UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            SampleDescriptorsWritten[index] = true;
        }
        if (material) for (auto& image : GBufferImages) TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        else TransitionImage(CommandBuffer, *color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        TransitionImage(CommandBuffer, *depth, velocity ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        std::array<VkClearValue, GBufferAttachmentCount + 1> clear {};
        VkRenderPassBeginInfo begin {};
        begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass = pass; begin.framebuffer = framebuffer;
        begin.renderArea = {{0, 0}, {MainColor.Width, MainColor.Height}};
        begin.clearValueCount = material ? static_cast<std::uint32_t>(clear.size()) : 2; begin.pClearValues = clear.data();
        CmdBeginRenderPass(CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport {0, 0, static_cast<float>(MainColor.Width), static_cast<float>(MainColor.Height), 0, 1};
        CmdSetViewport(CommandBuffer, 0, 1, &viewport); CmdSetScissor(CommandBuffer, 0, 1, &begin.renderArea);
        CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, Multisample.Pipelines[index]);
        const auto dynamicOffset = static_cast<std::uint32_t>(offset);
        CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, Multisample.Layout, 0, 1, &set, 1, &dynamicOffset);
        constexpr std::uint32_t FullscreenTriangleVertices = 3;
        CmdDraw(CommandBuffer, FullscreenTriangleVertices, 1, 0, 0);
        CmdEndRenderPass(CommandBuffer);
        return true;
    }

    bool NeedsSampleColorRestore() const { return SubsurfaceRequested || SubsurfaceResolved || GIActive || AmbientOcclusionActive || ReflectionActive || FogActive || LightShaftActive || TemporalActive || MotionBlurActive || DepthOfFieldActive; }
    bool OpaqueEffectsComplete() const
    {
        return SubsurfaceComplete() && ((!GIActive || GI.Resolved) && (!AmbientOcclusionActive || AmbientOcclusionResolved) && (!ReflectionActive || ReflectionResolved) &&
            (!FogActive || FogResolved) && (!LightShaftActive || LightShaftResolved) && (!TemporalActive || TemporalResolved) && (!MotionBlurActive || MotionBlurResolved) &&
            (!DepthOfFieldActive || DepthOfFieldResolved));
    }

    bool CreateShadowRenderPass(bool clearDepth, VkRenderPass& output, std::uint32_t samples = 1)
    {
        VkAttachmentDescription attachment{};
        attachment.format = VK_FORMAT_D32_SFLOAT;
        attachment.samples = samples;
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

    bool EnsureWeightedSurface(Techniques::ShadingModel model)
    {
        const auto index = ShadingTechniqueIndex(model, FrameAreaLights);
        if (index >= WeightedSurfaceModules.size()) return Fail("Invalid Vulkan weighted shading model");
        if (WeightedSurfaceModules[index]) return true;
        const auto& code = VulkanWeightedSurfaceShaders[index];
        std::vector<std::uint32_t> compiledWords;
        if (!CompileVulkanShader(code, compiledWords, LastError)) return false;
        VkShaderModuleCreateInfo create {};
        create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create.codeSize = compiledWords.size() * sizeof(std::uint32_t); create.pCode = compiledWords.data();
        return Check(CreateShaderModuleFunction(Device, &create, nullptr, &WeightedSurfaceModules[index]),
            "vkCreateShaderModule(weighted surface)");
    }

    void DestroyWeightedTargets()
    {
        if (Weighted.ResolvePipeline) DestroyPipelineFunction(Device, Weighted.ResolvePipeline, nullptr);
        for (auto framebuffer : {Weighted.GeometryFramebuffer, Weighted.ResolveFramebuffer})
            if (framebuffer) DestroyFramebufferFunction(Device, framebuffer, nullptr);
        for (auto renderPass : Weighted.GeometryPasses)
            if (renderPass) DestroyRenderPassFunction(Device, renderPass, nullptr);
        if (Weighted.ResolvePass) DestroyRenderPassFunction(Device, Weighted.ResolvePass, nullptr);
        for (auto& image : Weighted.Images) DestroyRawImage(image);
        Weighted = {};
    }

    bool CreateWeightedGeometryPass(bool clear, VkRenderPass& output)
    {
        // Opaque depth is shared read-only, including its original MSAA samples.
        // Clear and load passes have identical attachment formats/dependencies.
        std::array<VkAttachmentDescription, 3> attachments {};
        for (std::size_t index = 0; index < WeightedFormats.size(); ++index)
        {
            auto& attachment = attachments[index];
            attachment.format = WeightedFormats[index];
            attachment.samples = FrameSampleCount;
            attachment.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        auto& depth = attachments.back();
        depth.format = VK_FORMAT_D32_SFLOAT;
        depth.samples = FrameSampleCount;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        const std::array<VkAttachmentReference, 2> colors {{{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}};
        const VkAttachmentReference depthReference {2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkSubpassDescription subpass {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
        subpass.pColorAttachments = colors.data();
        subpass.pDepthStencilAttachment = &depthReference;
        VkSubpassDependency dependency {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo create {};
        create.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        create.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        create.pAttachments = attachments.data();
        create.subpassCount = 1;
        create.pSubpasses = &subpass;
        create.dependencyCount = 1;
        create.pDependencies = &dependency;
        return Check(CreateRenderPassFunction(Device, &create, nullptr, &output), "vkCreateRenderPass(weighted geometry)");
    }

    bool EnsureWeightedResources()
    {
        if (!HdrActive || !(WeightedSampleCounts & FrameSampleCount))
            return Fail("Vulkan weighted transparency requires HDR and supported blendable float samples");
        const auto width = MainColor.Width, height = MainColor.Height;
        for (const auto& limits : WeightedLimits)
            if (width > limits.maxExtent.width || height > limits.maxExtent.height)
                return Fail("Vulkan weighted statistics exceed the format extent limits");
        auto& scene = FrameSampleCount > 1 ? Multisample.Images[SampleColor] : HdrImages[0];
        auto& depth = FrameSampleCount > 1 ? Multisample.Depth : MainDepth;
        if (!WeightedDescriptorLayout)
        {
            const std::array<VkDescriptorSetLayoutBinding, 2> bindings {{
                {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
            VkDescriptorSetLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount = static_cast<std::uint32_t>(bindings.size());
            create.pBindings = bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device, &create, nullptr, &WeightedDescriptorLayout),
                "vkCreateDescriptorSetLayout(weighted resolve)")) return false;
        }
        if (!WeightedResolveLayout)
        {
            VkPipelineLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            create.setLayoutCount = 1;
            create.pSetLayouts = &WeightedDescriptorLayout;
            if (!Check(CreatePipelineLayoutFunction(Device, &create, nullptr, &WeightedResolveLayout),
                "vkCreatePipelineLayout(weighted resolve)")) return false;
        }
        if (!WeightedDescriptorPool)
        {
            const VkDescriptorPoolSize size {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2};
            VkDescriptorPoolCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            create.maxSets = 1;
            create.poolSizeCount = 1;
            create.pPoolSizes = &size;
            if (!Check(CreateDescriptorPoolFunction(Device, &create, nullptr, &WeightedDescriptorPool),
                "vkCreateDescriptorPool(weighted resolve)")) return false;
        }
        if (!WeightedDescriptorSet)
        {
            VkDescriptorSetAllocateInfo allocate {};
            allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool = WeightedDescriptorPool;
            allocate.descriptorSetCount = 1;
            allocate.pSetLayouts = &WeightedDescriptorLayout;
            if (!Check(AllocateDescriptorSets(Device, &allocate, &WeightedDescriptorSet),
                "vkAllocateDescriptorSets(weighted resolve)")) return false;
        }
        if (!WeightedSampler)
        {
            VkSamplerCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.magFilter = create.minFilter = VK_FILTER_NEAREST;
            create.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            create.maxAnisotropy = 1.0f;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &WeightedSampler),
                "vkCreateSampler(weighted resolve)")) return false;
        }
        if (!Weighted.ResolveFramebuffer || Weighted.Images[0].Width != width || Weighted.Images[0].Height != height ||
            Weighted.Images[0].Samples != FrameSampleCount || Weighted.DepthView != depth.View || Weighted.SceneView != scene.View)
        {
            DestroyWeightedTargets();
            for (std::size_t index = 0; index < WeightedFormats.size(); ++index)
                if (!CreateRawImage(width, height, WeightedFormats[index], VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, Weighted.Images[index], 1, FrameSampleCount)) return false;
            if (!CreateWeightedGeometryPass(false, Weighted.GeometryPasses[0]) ||
                !CreateWeightedGeometryPass(true, Weighted.GeometryPasses[1]) ||
                !CreateFullscreenRenderPass(scene.Format, Weighted.ResolvePass, true, FrameSampleCount)) return false;
            const std::array<VkImageView, 3> views {Weighted.Images[0].View, Weighted.Images[1].View, depth.View};
            if (!CreateFramebuffer(Weighted.GeometryPasses[0], views.data(), static_cast<std::uint32_t>(views.size()),
                width, height, Weighted.GeometryFramebuffer) ||
                !CreateFramebuffer(Weighted.ResolvePass, &scene.View, 1, width, height, Weighted.ResolveFramebuffer)) return false;
            Weighted.DepthView = depth.View;
            Weighted.SceneView = scene.View;
        }
        const auto variant = FrameSampleCount > 1 ? 1 : 0;
        if (!CreateFullscreenPipeline(VulkanWeightedResolveShaders[variant], WeightedResolveModules[variant],
            Weighted.ResolvePass, Weighted.ResolvePipeline, WeightedResolveLayout, FullscreenBlend::SourceOver,
            false, FrameSampleCount, FrameSampleCount > 1)) return false;
        // BeginFrame waits for the fence and resets recorded commands first.
        // These descriptors stay immutable from recording through submission.
        std::array<VkDescriptorImageInfo, 2> images {};
        std::array<VkWriteDescriptorSet, 2> writes {};
        for (std::uint32_t index = 0; index < images.size(); ++index)
        {
            images[index] = {WeightedSampler, Weighted.Images[index].View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = WeightedDescriptorSet;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[index].pImageInfo = &images[index];
        }
        UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    bool CreateShaderModules(Techniques::ShadingModel model = Techniques::ShadingModel::Phong)
    {
        const std::size_t index = ShadingTechniqueIndex(model, FrameAreaLights);
        if (index >= TechniqueModules.size()) return Fail("Invalid Vulkan shading model");
        for (std::size_t stage = 0; stage < TechniqueModules[index].size(); ++stage)
        {
            if (TechniqueModules[index][stage] != 0) continue;
            const VulkanShaderProgram& code = VulkanTechniqueShaders[index][stage];
            std::vector<std::uint32_t> compiledWords;
            if (!CompileVulkanShader(code, compiledWords, LastError)) return false;
            VkShaderModuleCreateInfo descriptor{};
            descriptor.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            descriptor.codeSize = compiledWords.size() * sizeof(std::uint32_t); descriptor.pCode = compiledWords.data();
            if (!Check(CreateShaderModuleFunction(Device, &descriptor, nullptr, &TechniqueModules[index][stage]),
                "vkCreateShaderModule(technique)")) return false;
        }
        return true;
    }

    bool CreateShadowModules()
    {
        for (std::size_t stage = 0; stage < ShadowModules.size(); ++stage)
        {
            if (ShadowModules[stage]) continue;
            std::vector<std::uint32_t> compiledWords;
            if (!CompileVulkanShader(VulkanShadowMappingShaders[stage], compiledWords, LastError)) return false;
            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = compiledWords.size() * sizeof(std::uint32_t); info.pCode = compiledWords.data();
            if (!Check(CreateShaderModuleFunction(Device, &info, nullptr, &ShadowModules[stage]),
                "vkCreateShaderModule(single shadow map)")) return false;
        }
        return true;
    }

    bool EnsureInstanceShader(std::size_t program)
    {
        if (InstanceModules[program]) return true;
        const auto& code = VulkanInstanceShaders[program];
        std::vector<std::uint32_t> compiledWords;
        if (!CompileVulkanShader(code, compiledWords, LastError)) return false;
        VkShaderModuleCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = compiledWords.size() * sizeof(std::uint32_t); info.pCode = compiledWords.data();
        return Check(CreateShaderModuleFunction(Device, &info, nullptr, &InstanceModules[program]),
            "vkCreateShaderModule(instancing)");
    }

    bool CreateInstanceLayout()
    {
        const VkDescriptorSetLayoutBinding binding {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
            VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 1; info.pBindings = &binding;
        return Check(CreateDescriptorSetLayoutFunction(Device, &info, nullptr, &InstanceDescriptorLayout),
            "vkCreateDescriptorSetLayout(instancing)");
    }

    void DestroyInstancePage(InstanceUploadPage& page)
    {
        if (page.Pool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, page.Pool, nullptr);
        DestroyRawBuffer(page.Buffer);
        page = {};
    }

    bool CreateInstancePage(InstanceUploadPage& page)
    {
        if (!CreateRawBuffer(InstanceUploadPageSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, page.Buffer)) return false;
        const VkDescriptorPoolSize size {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1};
        VkDescriptorPoolCreateInfo pool {};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 1; pool.poolSizeCount = 1; pool.pPoolSizes = &size;
        if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &page.Pool), "vkCreateDescriptorPool(instance page)"))
        { DestroyInstancePage(page); return false; }
        VkDescriptorSetAllocateInfo allocate {};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = page.Pool; allocate.descriptorSetCount = 1; allocate.pSetLayouts = &InstanceDescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device, &allocate, &page.Set), "vkAllocateDescriptorSets(instance page)"))
        { DestroyInstancePage(page); return false; }
        const VkDescriptorBufferInfo buffer {page.Buffer.Buffer, 0, sizeof(InstanceConstantBlock)};
        VkWriteDescriptorSet write {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = page.Set; write.dstBinding = 0; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo = &buffer;
        UpdateDescriptorSets(Device, 1, &write, 0, nullptr);
        return true;
    }

    bool UploadInstances(const InstanceConstantBlock& constants)
    {
        if (ActiveGpuBatch)
        {
            const std::uint32_t offset = 0;
            CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout,
                InstanceDescriptorSetIndex, 1, &ActiveGpuBatch->InstanceSet, 1, &offset);
            return true;
        }
        // Each page descriptor is written once. Dynamic offsets select disjoint
        // slices; neither descriptors nor bytes used by recorded draws change.
        // BeginFrame waits for the frame fence before rewinding these cursors.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindDescriptorSets.html
        const auto alignment = std::max<std::size_t>(1, static_cast<std::size_t>(DeviceLimits.minUniformBufferOffsetAlignment));
        InstancePageOffset = AlignUp(InstancePageOffset, alignment);
        if (InstancePageOffset + sizeof(constants) > InstanceUploadPageSize)
        { ++InstancePageIndex; InstancePageOffset = 0; }
        if (InstancePageIndex == InstancePages.size())
        {
            InstanceUploadPage page;
            if (!CreateInstancePage(page)) return false;
            try { InstancePages.push_back(page); }
            catch (...) { DestroyInstancePage(page); throw; }
        }
        auto& page = InstancePages[InstancePageIndex];
        std::memcpy(static_cast<std::uint8_t*>(page.Buffer.Mapped) + InstancePageOffset, &constants, sizeof(constants));
        const auto offset = static_cast<std::uint32_t>(InstancePageOffset);
        CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout,
            InstanceDescriptorSetIndex, 1, &page.Set, 1, &offset);
        InstancePageOffset += sizeof(constants);
        return true;
    }

    void DestroyInstanceState()
    {
        for (auto& page : InstancePages) DestroyInstancePage(page);
        InstancePages.clear(); InstancePageIndex = InstancePageOffset = 0;
        for (auto& module : InstanceModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        if (InstanceDescriptorLayout && DestroyDescriptorSetLayoutFunction)
            DestroyDescriptorSetLayoutFunction(Device, InstanceDescriptorLayout, nullptr);
        InstanceDescriptorLayout = 0;
    }

    bool CreateDescriptorState()
    {
        if (!CreateCascadeDescriptorState() || !CreateInstanceLayout() || !CreateFogDescriptorState()) return false;
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
        const std::array<VkDescriptorSetLayoutBinding, 10> bindings = {{
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
             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {ForwardPlusStorageFirstBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {ForwardPlusStorageFirstBinding + 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {ForwardPlusStorageFirstBinding + 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
        VkDescriptorSetLayoutCreateInfo setLayout{};
        setLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        setLayout.pBindings = bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device, &setLayout, nullptr,
                                                      &DescriptorSetLayout),
                   "vkCreateDescriptorSetLayout")) return false;
        VkPipelineLayoutCreateInfo pipelineLayout{};
        pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        const std::array<VkDescriptorSetLayout, 4> layoutsBySet {DescriptorSetLayout, CascadeDescriptorLayout, InstanceDescriptorLayout, FogDescriptorLayout};
        pipelineLayout.setLayoutCount = static_cast<std::uint32_t>(layoutsBySet.size());
        pipelineLayout.pSetLayouts = layoutsBySet.data();
        // Four descriptor sets already cover material, cascades, instances and
        // fog. One fragment push range holds either POM parameters or OIT weights;
        // their distinct raster passes never read the other technique's payload.
        const VkPushConstantRange transparencyRange {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ParallaxConstants)};
        pipelineLayout.pushConstantRangeCount = 1;
        pipelineLayout.pPushConstantRanges = &transparencyRange;
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
        if (AnisotropicFilteringSupported)
        {
            sampler.anisotropyEnable = VK_TRUE_VALUE;
            sampler.maxAnisotropy = std::min(float(MaterialAnisotropy), DeviceLimits.maxSamplerAnisotropy);
            sampler.maxLod = 1000.0f; // VK_LOD_CLAMP_NONE
            if (!Check(CreateSamplerFunction(Device, &sampler, nullptr, &AnisotropicSampler),
                "vkCreateSampler(anisotropic)")) return false;
        }
        if (!CreateRawBuffer(UniformBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, UniformBuffer)) return false;
        if (!CreateRawBuffer(sizeof(ShaderLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ForwardPlusEmptyStorage)) return false;
        std::memset(ForwardPlusEmptyStorage.Mapped, 0, sizeof(ShaderLight));
        const std::array<VkDescriptorPoolSize, 4> sizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, MaximumFrameDescriptorSets},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaximumFrameDescriptorSets * 5U},
            {VK_DESCRIPTOR_TYPE_SAMPLER, MaximumFrameDescriptorSets},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MaximumFrameDescriptorSets * ForwardPlusStorageCount}}};
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

    bool CreateCascadeDescriptorState()
    {
        // Set 1 is identical for forward and deferred lighting. Two immutable
        // per-frame sets select enabled/disabled CSM without rewriting descriptors
        // already referenced by a recorded draw. Every unused image has a dummy.
        std::array<VkDescriptorSetLayoutBinding, MaximumShadowCascades + 2> bindings {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        for (std::uint32_t index = 0; index < MaximumShadowCascades; ++index)
            bindings[index + 1] = {index + 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings.back() = {CascadeSamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout {};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device, &layout, nullptr, &CascadeDescriptorLayout),
            "vkCreateDescriptorSetLayout(cascades)")) return false;
        constexpr std::uint32_t SetCount = 2;
        const std::array<VkDescriptorPoolSize, 3> sizes {{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SetCount},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaximumShadowCascades * SetCount}, {VK_DESCRIPTOR_TYPE_SAMPLER, SetCount}}};
        VkDescriptorPoolCreateInfo pool {};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = SetCount;
        pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        pool.pPoolSizes = sizes.data();
        if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &CascadeDescriptorPool),
            "vkCreateDescriptorPool(cascades)")) return false;
        const std::array<VkDescriptorSetLayout, SetCount> layouts {CascadeDescriptorLayout, CascadeDescriptorLayout};
        VkDescriptorSetAllocateInfo allocate {};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = CascadeDescriptorPool;
        allocate.descriptorSetCount = SetCount;
        allocate.pSetLayouts = layouts.data();
        if (!Check(AllocateDescriptorSets(Device, &allocate, CascadeDescriptorSets.data()),
            "vkAllocateDescriptorSets(cascades)")) return false;
        VkSamplerCreateInfo sampler {};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        return Check(CreateSamplerFunction(Device, &sampler, nullptr, &CascadeSampler), "vkCreateSampler(cascades)");
    }

    bool PrepareCascadedShadows(const CascadedShadowState& state)
    {
        FrameCascades = {};
        CascadeImages.fill(nullptr);
        const auto count = state.Frame.CascadeCount;
        if (count > MaximumShadowCascades || (count && state.MapResolution <= 0))
            return Fail("Vulkan cascade count or map resolution is invalid");
        for (std::uint32_t index = 0; index < count; ++index)
        {
            const auto map = Textures.find(state.Maps[index].Value);
            if (map == Textures.end() || map->second.Description.Format != TextureFormat::Depth32 ||
                map->second.Description.Width != state.MapResolution || map->second.Description.Height != state.MapResolution ||
                !map->second.Image.View || !map->second.Framebuffer)
                return Fail("Vulkan cascade depth target is invalid");
            CascadeImages[index] = &map->second.Image;
            if (UsesVSM())
            {
                if (!EnsureMomentMap(state.Maps[index])) return false;
                CascadeImages[index] = &VSMMoments.at(state.Maps[index].Value).Images[VSMFiltered];
            }
        }
        if (count)
            for (std::size_t stage = 0; stage < CascadeModules.size(); ++stage)
            {
                if (CascadeModules[stage]) continue;
                std::vector<std::uint32_t> compiledWords;
                if (!CompileVulkanShader(VulkanCascadedShadowShaders[stage], compiledWords, LastError)) return false;
                VkShaderModuleCreateInfo shader {};
                shader.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                shader.codeSize = compiledWords.size() * sizeof(std::uint32_t); shader.pCode = compiledWords.data();
                if (!Check(CreateShaderModuleFunction(Device, &shader, nullptr, &CascadeModules[stage]),
                    "vkCreateShaderModule(cascade depth)")) return false;
            }
        for (std::size_t enabled = 0; enabled < CascadeDescriptorSets.size(); ++enabled)
        {
            const auto constants = enabled && FrameVirtualShadowAtlas ? BuildVirtualShadowSamplingConstants(FrameVirtualShadowAtlas->Info,FrameVirtualShadowSampling,FrameShadowFilter) :
                BuildCascadedShadowConstants(enabled ? state : CascadedShadowState{}, FrameShadowFilter);
            const auto offset = AlignUp(UniformOffset, UniformAlignment);
            if (offset > UniformBufferSize || sizeof(constants) > UniformBufferSize - offset)
                return Fail("Vulkan cascade uniform arena is exhausted");
            std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
            UniformOffset = offset + sizeof(constants);
            const VkDescriptorBufferInfo buffer {UniformBuffer.Buffer, offset, sizeof(constants)};
            std::array<VkDescriptorImageInfo, MaximumShadowCascades + 1> images {};
            std::array<VkWriteDescriptorSet, MaximumShadowCascades + 2> writes {};
            for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
            {
                writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[binding].dstSet = CascadeDescriptorSets[enabled];
                writes[binding].dstBinding = binding;
                writes[binding].descriptorCount = 1;
            }
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &buffer;
            for (std::size_t index = 0; index < MaximumShadowCascades; ++index)
            {
                images[index].imageView = enabled && FrameVirtualShadowAtlas && index<2 ?
                    (index?FrameVirtualShadowAtlas->Directory.View:FrameVirtualShadowAtlas->Depth.View) : enabled && index<count?CascadeImages[index]->View:DummyDepth.View;
                images[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                writes[index + 1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                writes[index + 1].pImageInfo = &images[index];
            }
            images.back().sampler = CascadeSampler;
            writes.back().descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            writes.back().pImageInfo = &images.back();
            UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
        FrameCascades = state;
        return true;
    }

    bool BindCascades(VkPipelineLayout layout, bool enabled)
    {
        const bool active = enabled && (FrameCascades.Frame.CascadeCount > 0 || FrameVirtualShadowAtlas);
        if (active)
            for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index)
            {
                if (UsesVSM() && !MomentImage(FrameCascades.Maps[index], CascadeImages[index])) return false;
                if (CascadeImages[index]->Layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    return Fail("Vulkan cascade must complete its depth pass before lighting");
            }
        const VkDescriptorSet set = CascadeDescriptorSets[active ? 1 : 0];
        CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
            CascadeDescriptorSetIndex, 1, &set, 0, nullptr);
        return true;
    }

    void DestroyCascadeState()
    {
        for (auto& module : CascadeModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        for (auto& readback : CascadeReadbacks) DestroyRawBuffer(readback);
        if (CascadeDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, CascadeDescriptorPool, nullptr);
        if (CascadeDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, CascadeDescriptorLayout, nullptr);
        if (CascadeSampler && DestroySamplerFunction) DestroySamplerFunction(Device, CascadeSampler, nullptr);
        CascadeDescriptorPool = CascadeDescriptorLayout = CascadeSampler = 0;
        CascadeDescriptorSets.fill(0);
        CascadeImages.fill(nullptr);
        FrameCascades = {};
    }

    void UpdateDescriptorSet(
        VkDescriptorSet descriptorSet,
        const std::array<VkImageView, 4>& imageViews,
        VkSampler samplerOverride = 0, std::size_t constantSize = sizeof(RealtimeDrawConstants))
    {
        const VkDescriptorBufferInfo buffer{
            UniformBuffer.Buffer, 0, constantSize};
        std::array<VkDescriptorImageInfo, 4> images {};
        for (std::size_t index = 0; index < images.size(); ++index)
        {
            images[index] = {
                0,
                imageViews[index],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
        const VkDescriptorImageInfo sampler{
            samplerOverride ? samplerOverride : ShadowSampler, 0, VK_IMAGE_LAYOUT_UNDEFINED};
        std::array<VkWriteDescriptorSet, 7> writes{};
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
        // Refresh even unused bindings. A capture/depth draw can reuse a set
        // from last frame before this frame's compute producer resizes its image;
        // leaving that old view here would invalidate the recorded command buffer.
        const VkDescriptorImageInfo contact {0, ContactReceiverActive ? ContactProducer->ResultView() : DummyColor.View,
            ContactReceiverActive ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[6].dstSet = descriptorSet; writes[6].dstBinding = 6; writes[6].descriptorCount = 1;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; writes[6].pImageInfo = &contact;
        UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(),
                             0, nullptr);
        // Unused descriptors are refreshed as well: a previous frame's list
        // allocation may be replaced after an early reflection capture records.
        const VkDescriptorBufferInfo empty {ForwardPlusEmptyStorage.Buffer, 0, sizeof(ShaderLight)};
        std::array<VkWriteDescriptorSet, ForwardPlusStorageCount> storageWrites {};
        for (std::uint32_t slot = 0; slot < ForwardPlusStorageCount; ++slot)
        {
            auto& write = storageWrites[slot]; write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSet; write.dstBinding = ForwardPlusStorageFirstBinding + slot; write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &empty;
        }
        UpdateDescriptorSets(Device, ForwardPlusStorageCount, storageWrites.data(), 0, nullptr);
    }

    bool CreateMainTarget(int width, int height)
    {
        DestroyMainTarget();
        const std::uint32_t targetWidth = static_cast<std::uint32_t>(width);
        const std::uint32_t targetHeight = static_cast<std::uint32_t>(height);
        if (!CreateRawImage(targetWidth, targetHeight, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, MainColor) ||
            !CreateRawImage(targetWidth, targetHeight, VK_FORMAT_D32_SFLOAT,
                            VK_IMAGE_ASPECT_DEPTH_BIT,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, MainDepth))
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
        DestroyContactTargets();
        DestroyWeightedTargets();
        DestroyTemporalTargets();
        DestroyGITargets(); DestroyProbeTargets(); DestroyLightmapTargets(); DestroyRefractionTargets(); DestroyPeelingTargets();
        DestroyMultisampleTargets(Multisample);
        DestroyMotionTargets();
        DestroyAmbientOcclusionTargets();
        DestroyReflectionTargets();
        DestroyDeferredTargets();
        DestroyHdrTargets();
        DestroyPostTargets();
        if (MainFramebuffer != 0 && DestroyFramebufferFunction)
            DestroyFramebufferFunction(Device, MainFramebuffer, nullptr);
        MainFramebuffer = 0;
        DestroyRawImage(MainColor);
        DestroyRawImage(MainDepth);
        DestroyRawBuffer(ColorReadback);
        DestroyRawBuffer(DepthReadback);
    }

    void DestroyPostTargets()
    {
        for (VkFramebuffer& framebuffer : PostFramebuffers)
        {
            if (framebuffer != 0 && DestroyFramebufferFunction)
                DestroyFramebufferFunction(Device, framebuffer, nullptr);
            framebuffer = 0;
        }
        for (RawImage& image : PostImages) DestroyRawImage(image);
        if (PostSampler && DestroySamplerFunction) DestroySamplerFunction(Device, PostSampler, nullptr);
        PostSampler = 0;
    }

    bool CreateFullscreenRenderPass(VkFormat format, VkRenderPass& output, bool loadColor = false, std::uint32_t samples = 1)
    {
        if (output != 0) return true;
        VkAttachmentDescription attachment{};
        attachment.format = format;
        attachment.samples = samples;
        attachment.loadOp = loadColor ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // Keep clear/load variants compatible, including their dependencies.
        // AO loads previously rendered HDR color for reverse-subtract blending.
        dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo descriptor{};
        descriptor.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        descriptor.attachmentCount = 1;
        descriptor.pAttachments = &attachment;
        descriptor.subpassCount = 1;
        descriptor.pSubpasses = &subpass;
        descriptor.dependencyCount = 1;
        descriptor.pDependencies = &dependency;
        return Check(CreateRenderPassFunction(Device, &descriptor, nullptr, &output),
            "vkCreateRenderPass(post-process)");
    }

    bool EnsurePostRenderPass()
    {
        return CreateFullscreenRenderPass(VK_FORMAT_R8G8B8A8_SRGB, PostRenderPass);
    }

    bool EnsurePostTargets()
    {
        if (PostFramebuffers[0] != 0 && PostFramebuffers[1] != 0) return true;
        if (!EnsurePostRenderPass()) return false;
        DestroyPostTargets();
        // Fullscreen color sampling has its own edge policy. The general
        // surface sampler repeats material UVs and is unsuitable for warps.
        VkSamplerCreateInfo sampler {};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.minFilter = sampler.magFilter = VK_FILTER_LINEAR;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(CreateSamplerFunction(Device, &sampler, nullptr, &PostSampler),
            "vkCreateSampler(post-process)")) return false;
        for (std::size_t index = 0; index < PostImages.size(); ++index)
        {
            if (!CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R8G8B8A8_SRGB,
                    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, PostImages[index]) ||
                !CreateFramebuffer(PostRenderPass, &PostImages[index].View, 1,
                    MainColor.Width, MainColor.Height, PostFramebuffers[index]))
            {
                DestroyPostTargets();
                return false;
            }
        }
        return true;
    }

    VkPipeline CreateFullscreenPipeline(const VulkanShaderProgram* codes,
        std::array<VkShaderModule, 2>& modules, VkRenderPass renderPass, VkPipeline& pipeline,
        VkPipelineLayout layout = 0, FullscreenBlend blendMode = FullscreenBlend::Replace, bool preserveAlpha = false,
        std::uint32_t samples = 1, bool sampleShading = false, std::uint32_t colorCount = 1,
        bool writeDepth = false, bool writeColor = true)
    {
        if (pipeline) return pipeline;
        for (std::size_t stage = 0; stage < modules.size(); ++stage)
        {
            if (modules[stage] != 0) continue;
            const auto& code = codes[stage];
            std::vector<std::uint32_t> compiledWords;
            if (!CompileVulkanShader(code, compiledWords, LastError)) return 0;
            VkShaderModuleCreateInfo descriptor{};
            descriptor.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            descriptor.codeSize = compiledWords.size() * sizeof(std::uint32_t); descriptor.pCode = compiledWords.data();
            if (!Check(CreateShaderModuleFunction(Device, &descriptor, nullptr, &modules[stage]),
                "vkCreateShaderModule(post-process)")) return 0;
        }
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        for (std::size_t stage = 0; stage < stages.size(); ++stage)
        {
            stages[stage].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[stage].stage = stage == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[stage].module = modules[stage];
            stages[stage].pName = "main";
        }
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo assembly{};
        assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
        multisample.rasterizationSamples = samples;
        multisample.sampleShadingEnable = sampleShading ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        multisample.minSampleShading = sampleShading ? 1.0f : 0.0f;
        VkPipelineDepthStencilStateCreateInfo depth {};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = depth.depthWriteEnable = writeDepth ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
        depth.maxDepthBounds = 1.0f;
        VkPipelineColorBlendAttachmentState color{};
        color.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (preserveAlpha) color.colorWriteMask &= ~VK_COLOR_COMPONENT_A_BIT;
        if (blendMode != FullscreenBlend::Replace)
        {
            color.blendEnable = VK_TRUE_VALUE;
            color.srcColorBlendFactor = color.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            color.colorBlendOp = blendMode == FullscreenBlend::SubtractAmbient ? VK_BLEND_OP_REVERSE_SUBTRACT : VK_BLEND_OP_ADD;
            color.srcAlphaBlendFactor = color.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            color.alphaBlendOp = VK_BLEND_OP_ADD;
            color.colorWriteMask &= ~VK_COLOR_COMPONENT_A_BIT;
        }
        if (blendMode == FullscreenBlend::SourceOver)
        {
            color.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            color.dstColorBlendFactor = color.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            if (!preserveAlpha) color.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
        }
        VkPipelineColorBlendStateCreateInfo blend{};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        if (!writeColor) color.colorWriteMask = 0;
        std::array<VkPipelineColorBlendAttachmentState, GBufferAttachmentCount> colors;
        colors.fill(color);
        blend.attachmentCount = colorCount;
        blend.pAttachments = colors.data();
        const std::array<VkDynamicState, 2> states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(states.size());
        dynamic.pDynamicStates = states.data();
        VkGraphicsPipelineCreateInfo descriptor{};
        descriptor.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        descriptor.stageCount = static_cast<std::uint32_t>(stages.size());
        descriptor.pStages = stages.data();
        descriptor.pVertexInputState = &vertexInput;
        descriptor.pInputAssemblyState = &assembly;
        descriptor.pViewportState = &viewport;
        descriptor.pRasterizationState = &raster;
        descriptor.pMultisampleState = &multisample;
        descriptor.pColorBlendState = &blend;
        descriptor.pDepthStencilState = &depth;
        descriptor.pDynamicState = &dynamic;
        descriptor.layout = layout ? layout : PipelineLayout;
        descriptor.renderPass = renderPass;
        if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0, 1, &descriptor, nullptr, &pipeline),
            "vkCreateGraphicsPipelines(post-process)")) return 0;
        return pipeline;
    }

    VkPipeline GetPostPipeline(Techniques::PostProcessPassKind pass)
    {
        const std::size_t index = static_cast<std::size_t>(pass);
        if (index >= PostPipelines.size()) { Fail("Invalid Vulkan post-process pass"); return 0; }
        if (!EnsurePostRenderPass()) return 0;
        return CreateFullscreenPipeline(VulkanPostProcessShaders[index], PostModules[index], PostRenderPass, PostPipelines[index]);
    }

    bool ApplyPostProcess(const Techniques::PostProcessSettings& settings, RawImage*& result)
    {
        result = &MainColor;
        const auto plan = Techniques::BuildPostProcessPlan(settings);
        if (plan.empty()) return true;
        if (!EnsurePostTargets()) return false;
        const PostProcessConstants constants = BuildPostProcessConstants(settings, CreateInfo.Width, CreateInfo.Height);
        std::size_t destination = 0;
        for (const auto pass : plan)
        {
            const VkPipeline pipeline = GetPostPipeline(pass);
            if (pipeline == 0) return false;
            const std::size_t offset = AlignUp(UniformOffset, UniformAlignment);
            if (offset > UniformBufferSize || sizeof(RealtimeDrawConstants) > UniformBufferSize - offset)
                return Fail("Vulkan post-process uniform arena is exhausted");
            if (DescriptorSetIndex >= DescriptorSets.size())
                return Fail("Vulkan post-process descriptor arena is exhausted");
            std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
            UniformOffset = offset + sizeof(RealtimeDrawConstants);
            const VkDescriptorSet descriptorSet = DescriptorSets[DescriptorSetIndex++];
            const bool geometryEdge = pass == Techniques::PostProcessPassKind::DepthNormalOutline;
            if (geometryEdge)
            {
                TransitionImage(CommandBuffer, MainDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                TransitionImage(CommandBuffer, GBufferImages[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            UpdateDescriptorSet(descriptorSet, {geometryEdge ? MainDepth.View : DummyDepth.View,
                result->View, geometryEdge ? GBufferImages[1].View : DummyColor.View, DummyColor.View}, PostSampler);
            TransitionImage(CommandBuffer, *result, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            TransitionImage(CommandBuffer, PostImages[destination], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            VkClearValue clear{};
            VkRenderPassBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            begin.renderPass = PostRenderPass;
            begin.framebuffer = PostFramebuffers[destination];
            begin.renderArea = {{0, 0}, {MainColor.Width, MainColor.Height}};
            begin.clearValueCount = 1;
            begin.pClearValues = &clear;
            CmdBeginRenderPass(CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport{0.0f, 0.0f, static_cast<float>(MainColor.Width), static_cast<float>(MainColor.Height), 0.0f, 1.0f};
            const VkRect2D scissor{{0, 0}, {MainColor.Width, MainColor.Height}};
            CmdSetViewport(CommandBuffer, 0, 1, &viewport);
            CmdSetScissor(CommandBuffer, 0, 1, &scissor);
            CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(offset);
            CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &descriptorSet, 1, &dynamicOffset);
            CmdDraw(CommandBuffer, 3, 1, 0, 0);
            CmdEndRenderPass(CommandBuffer);
            result = &PostImages[destination];
            destination ^= 1U;
            PendingFrame.PostProcessingPasses.push_back(pass);
        }
        PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
        return true;
    }

    bool CreateDeferredRenderPass(bool clearColor, bool clearDepth, VkRenderPass& output, std::uint32_t samples = 1)
    {
        if (output) return true;
        std::array<VkAttachmentDescription, GBufferAttachmentCount + 1> attachments {};
        std::array<VkAttachmentReference, GBufferAttachmentCount> colors {};
        for (std::size_t index = 0; index < attachments.size(); ++index)
        {
            const bool depth = index == GBufferAttachmentCount;
            auto& attachment = attachments[index];
            attachment.format = depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
            attachment.samples = samples;
            attachment.loadOp = (depth ? clearDepth : clearColor) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = attachment.finalLayout = depth
                ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            if (!depth) colors[index] = {static_cast<std::uint32_t>(index), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        }
        const VkAttachmentReference depth {static_cast<std::uint32_t>(GBufferAttachmentCount), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
        subpass.pColorAttachments = colors.data();
        subpass.pDepthStencilAttachment = &depth;
        VkSubpassDependency dependency {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask = dependency.srcAccessMask | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo create {};
        create.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        create.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        create.pAttachments = attachments.data();
        create.subpassCount = 1;
        create.pSubpasses = &subpass;
        create.dependencyCount = 1;
        create.pDependencies = &dependency;
        return Check(CreateRenderPassFunction(Device, &create, nullptr, &output), "vkCreateRenderPass(G-buffer)");
    }

    void DestroyDeferredTargets()
    {
        DestroyVisibilityTargets();
        if (GBufferFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, GBufferFramebuffer, nullptr);
        GBufferFramebuffer = 0;
        for (auto& image : GBufferImages) DestroyRawImage(image);
    }

    void DestroyDeferredState()
    {
        DestroyRayShadowPrograms(); DestroyPCFPrograms();
        DestroyDeferredTargets();
        if (PCSSDeferredPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, PCSSDeferredPipeline, nullptr);
        PCSSDeferredPipeline = 0;
        if (VSMDeferredPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, VSMDeferredPipeline, nullptr);
        VSMDeferredPipeline = 0;
        for (auto& family : TiledLighting) for (auto& samples : family) for (auto& state : samples)
        {
            if (state.Pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, state.Pipeline, nullptr);
            for (auto module : state.Modules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            state = {};
        }
        if (DeferredLightingPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, DeferredLightingPipeline, nullptr);
        DeferredLightingPipeline = 0;
        for (auto* modules : {&DeferredGeometryModules, &DeferredLightingModules, &PCSSDeferredModules, &VSMDeferredModules})
            for (auto& module : *modules)
            {
                if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
                module = 0;
            }
        for (auto& pass : DeferredRenderPasses)
        {
            if (pass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, pass, nullptr);
            pass = 0;
        }
        if (DeferredDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, DeferredDescriptorPool, nullptr);
        if (DeferredPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, DeferredPipelineLayout, nullptr);
        if (DeferredDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, DeferredDescriptorLayout, nullptr);
        if (DeferredSampler && DestroySamplerFunction) DestroySamplerFunction(Device, DeferredSampler, nullptr);
        DeferredDescriptorPool = DeferredPipelineLayout = DeferredDescriptorLayout = DeferredDescriptorSet = DeferredSampler = 0;
        DeferredActive = GBufferWritten = DeferredResolved = false;
    }

    bool EnsureDeferredResources()
    {
        if (!DeferredSupported || !HdrActive) return Fail("Vulkan deferred shading requires five sampled floating attachments and sampled depth");
        for (std::size_t mask = 0; mask < DeferredRenderPasses.size(); ++mask)
            if (!CreateDeferredRenderPass((mask & 1U) != 0, (mask & 2U) != 0, DeferredRenderPasses[mask])) return false;
        if (!GBufferFramebuffer)
        {
            std::array<VkImageView, GBufferAttachmentCount + 1> views {};
            for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
            {
                if (!CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, GBufferImages[index]))
                {
                    DestroyDeferredTargets();
                    return false;
                }
                views[index] = GBufferImages[index].View;
            }
            views[GBufferAttachmentCount] = MainDepth.View;
            if (!CreateFramebuffer(DeferredRenderPasses[0], views.data(), static_cast<std::uint32_t>(views.size()),
                MainColor.Width, MainColor.Height, GBufferFramebuffer))
            {
                DestroyDeferredTargets();
                return false;
            }
        }
        for (std::size_t stage = 0; stage < DeferredGeometryModules.size(); ++stage)
        {
            if (DeferredGeometryModules[stage]) continue;
            std::vector<std::uint32_t> compiledWords;
            if (!CompileVulkanShader(VulkanDeferredGeometryShaders[stage], compiledWords, LastError)) return false;
            VkShaderModuleCreateInfo shader {};
            shader.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shader.codeSize = compiledWords.size() * sizeof(std::uint32_t); shader.pCode = compiledWords.data();
            if (!Check(CreateShaderModuleFunction(Device, &shader, nullptr, &DeferredGeometryModules[stage]), "vkCreateShaderModule(G-buffer)")) return false;
        }
        constexpr std::uint32_t ImageCount = static_cast<std::uint32_t>(GBufferAttachmentCount) + 2;
        if (!DeferredDescriptorLayout)
        {
            std::array<VkDescriptorSetLayoutBinding, ImageCount + 3 + ForwardPlusStorageCount> bindings {};
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (std::uint32_t index = 0; index < ImageCount; ++index)
                bindings[index + 1] = {index + 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            bindings[ImageCount + 1] = {ImageCount + 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            bindings[ImageCount + 2] = {ImageCount + 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (std::uint32_t slot = 0; slot < ForwardPlusStorageCount; ++slot)
                bindings[ImageCount + 3 + slot] = {ImageCount + 3 + slot, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount = static_cast<std::uint32_t>(bindings.size());
            create.pBindings = bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device, &create, nullptr, &DeferredDescriptorLayout), "vkCreateDescriptorSetLayout(deferred)")) return false;
        }
        if (!DeferredPipelineLayout)
        {
            VkPipelineLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            const std::array<VkDescriptorSetLayout, 4> layouts {DeferredDescriptorLayout,CascadeDescriptorLayout,InstanceDescriptorLayout,FogDescriptorLayout};
            create.setLayoutCount = static_cast<std::uint32_t>(layouts.size());
            create.pSetLayouts = layouts.data();
            if (!Check(CreatePipelineLayoutFunction(Device, &create, nullptr, &DeferredPipelineLayout), "vkCreatePipelineLayout(deferred)")) return false;
        }
        if (!DeferredSampler)
        {
            // All light-pass inputs use texelFetch, including manually filtered
            // shadows. A nearest sampler avoids requiring D32 linear filtering.
            VkSamplerCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.magFilter = create.minFilter = VK_FILTER_NEAREST;
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &DeferredSampler), "vkCreateSampler(deferred)")) return false;
        }
        if (!DeferredDescriptorPool)
        {
            const std::array<VkDescriptorPoolSize, 4> sizes {{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, ImageCount + 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ForwardPlusStorageCount}}};
            VkDescriptorPoolCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            create.maxSets = 1;
            create.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
            create.pPoolSizes = sizes.data();
            if (!Check(CreateDescriptorPoolFunction(Device, &create, nullptr, &DeferredDescriptorPool), "vkCreateDescriptorPool(deferred)")) return false;
        }
        if (!DeferredDescriptorSet)
        {
            VkDescriptorSetAllocateInfo allocate {};
            allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool = DeferredDescriptorPool;
            allocate.descriptorSetCount = 1;
            allocate.pSetLayouts = &DeferredDescriptorLayout;
            if (!Check(AllocateDescriptorSets(Device, &allocate, &DeferredDescriptorSet), "vkAllocateDescriptorSets(deferred)")) return false;
        }
        return CreateFullscreenPipeline(VulkanDeferredLightingShaders, DeferredLightingModules, HdrRenderPass,
            DeferredLightingPipeline, DeferredPipelineLayout) != 0;
    }

    void DestroyAmbientOcclusionTargets()
    {
        if (SaoScratchFramebuffer && DestroyFramebufferFunction)
            DestroyFramebufferFunction(Device, SaoScratchFramebuffer, nullptr);
        SaoScratchFramebuffer = 0;
        DestroyRawImage(SaoScratch);
        DestroyRawImage(SaoDepthHierarchy);
        for (auto& framebuffer : AmbientOcclusionFramebuffers)
        {
            if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, framebuffer, nullptr);
            framebuffer = 0;
        }
        for (auto& image : AmbientOcclusionImages) DestroyRawImage(image);
    }

    #include "Renderer/Vulkan/VulkanRayScene.inl"
    #include "Renderer/Vulkan/VulkanRayRadiance.inl"
    #include "Renderer/Vulkan/VulkanRayReflections.inl"
    #include "Renderer/Vulkan/VulkanHybridRendering.inl"
    #include "Renderer/Vulkan/VulkanSubsurfaceScattering.inl"
    #include "Renderer/Vulkan/VulkanHairFur.inl"
    #include "Renderer/Vulkan/VulkanTemporalEffects.inl"
    #include "Renderer/Vulkan/VulkanParallaxOcclusionMapping.inl"
    #include "Renderer/Vulkan/VulkanDisplacementMapping.inl"
    #include "Renderer/Vulkan/VulkanTessellation.inl"
    #include "Renderer/VirtualGeometry/Vulkan/VulkanVirtualGeometryCache.inl"
    #include "Renderer/VirtualGeometry/Vulkan/VulkanVirtualGeometry.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteCache.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteGeometry.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteAsync.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteCacheAsync.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteFrameVisibility.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteDraws.inl"
#include "Renderer/Vulkan/VulkanTemporalSuperResolution.inl"
#include "Renderer/VirtualShadowMaps/Vulkan/VulkanVirtualShadowMaps.inl"
#include "Renderer/VirtualShadowMaps/Vulkan/VulkanVirtualShadowAtlas.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenSurface.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenCaptureAsync.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenSurfaceLookup.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenDistanceTrace.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenGlobalDistance.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenGlobalDistanceAsync.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenSurfaceLighting.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenLightingAsync.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenRadianceTrace.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenProbeIntegration.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenProbeGather.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenTranslucency.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenHairReceiver.inl"
#include "Renderer/Vulkan/VulkanRayTracedShadows.inl"
#include "Renderer/Vulkan/VulkanPCF.inl"
#include "Renderer/Vulkan/VulkanMaterials.inl"

    bool UsesMaterialSurface(const PipelineState& pipeline,PassType pass) const
    {
        return pass==PassType::Color && !pipeline.OutlineExtrusion &&
            FrameShadowFilter.Mode==Techniques::ShadowFilterMode::Hard &&
            !FrameVirtualShadowAtlas && !FrameContact.Enabled && !FrameAreaLights && !LightListsActive && !FrameRayShadowImage;
    }
    bool UsesMaterialStage(const PipelineState& pipeline,PassType pass) const
    { return (pass==PassType::GBuffer && !UsesParallaxMaterial(pipeline,pass)) || UsesMaterialSurface(pipeline,pass); }

    bool UsesDedicatedPCF() const
    { return FrameShadowFilter.Mode==Techniques::ShadowFilterMode::PCF && !FrameVirtualShadowAtlas && !FrameContact.Enabled && !FrameAreaLights && !LightListsActive; }
    bool UsesSurfacePCF(const PipelineState& pipeline,PassType pass) const
    { return UsesDedicatedPCF() && !FrameAreaLights && pass==PassType::Color && pipeline.SamplesShadowMap; }

#include "Renderer/Lumen/Vulkan/VulkanLumenTemporal.inl"
#include "Renderer/Lumen/Vulkan/VulkanLumenReflections.inl"
#include "Renderer/VirtualShadowMaps/Vulkan/VulkanVirtualShadowReceivers.inl"
#include "Renderer/VirtualShadowMaps/Vulkan/VulkanVirtualShadowLookup.inl"
    #include "Renderer/Vulkan/VulkanRayGI.inl"
    #include "Renderer/Vulkan/VulkanVoxelGI.inl"
    #include "Renderer/Vulkan/VulkanPathTracing.inl"
    #include "Renderer/Vulkan/VulkanGPUDriven.inl"
    #include "Renderer/Vulkan/VulkanVisibilityBuffer.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteMaterials.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteFrameRaster.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteVisibilityGeometry.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteInstances.inl"
    #include "Renderer/Nanite/Vulkan/VulkanNaniteHardware.inl"
#include "Renderer/MeshShaders/Vulkan/VulkanMeshShaders.inl"
    #include "Renderer/Profiling/Vulkan/VulkanFrameTiming.inl"

    void DestroyAmbientOcclusionState()
    {
        DestroyRayScene(); DestroyRayShadowImage(); DestroyRayShadowPrograms(); DestroyPCFPrograms(); DestroyMaterials();
        DestroyAmbientOcclusionTargets();
        for (auto& pipeline : SsaoPipelines)
        {
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& module : SsaoModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        for (auto& pipeline : HbaoPipelines)
        {
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& module : HbaoModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        for (auto& pipeline : GtaoPipelines)
        {
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& module : GtaoModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        for (auto& pipeline : SaoPipelines)
        {
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& module : SaoModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        if (SaoDepthRenderPass && DestroyRenderPassFunction)
            DestroyRenderPassFunction(Device, SaoDepthRenderPass, nullptr);
        SaoDepthRenderPass = 0;
        if (AmbientOcclusionCompositeRenderPass && DestroyRenderPassFunction)
            DestroyRenderPassFunction(Device, AmbientOcclusionCompositeRenderPass, nullptr);
        AmbientOcclusionCompositeRenderPass = 0;
        AmbientOcclusionActive = AmbientOcclusionResolved = ForwardOpaqueComplete = false;
        FrameAmbientOcclusion = {};
    }

    bool EnsureAmbientOcclusionResources()
    {
        if (!DeferredSupported || !HdrActive || !GBufferFramebuffer)
            return Fail("Vulkan screen-space AO requires sampled material/depth buffers and blendable HDR");
        if (!CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT, AmbientOcclusionCompositeRenderPass, true)) return false;
        for (std::size_t index = 0; index < AmbientOcclusionImages.size(); ++index)
        {
            if (AmbientOcclusionFramebuffers[index]) continue;
            if (!CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    AmbientOcclusionImages[index]) ||
                !CreateFramebuffer(HdrRenderPass, &AmbientOcclusionImages[index].View, 1,
                    MainColor.Width, MainColor.Height, AmbientOcclusionFramebuffers[index]))
            {
                DestroyAmbientOcclusionTargets();
                return false;
            }
        }
        if (FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::RayTraced) return EnsureRayAOState();
        // Each estimator owns independently compiled native shader modules and
        // pipelines. Only the attachment lifetime and execution contract are shared.
        if (FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::SAO)
        {
            if (!EnsureSaoDepthHierarchy()) return false;
            return CreateFullscreenPipeline(VulkanSaoShaders, SaoModules, HdrRenderPass, SaoPipelines[0]) &&
                CreateFullscreenPipeline(VulkanSaoShaders, SaoModules, AmbientOcclusionCompositeRenderPass,
                    SaoPipelines[1], PipelineLayout, FullscreenBlend::SubtractAmbient) &&
                CreateFullscreenPipeline(VulkanSaoShaders, SaoModules, SaoDepthRenderPass, SaoPipelines[2]);
        }
        const bool groundTruth = FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::GTAO;
        const bool horizon = FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::HBAO;
        const auto* shaders = groundTruth ? VulkanGtaoShaders : horizon ? VulkanHbaoShaders : VulkanSsaoShaders;
        auto& modules = groundTruth ? GtaoModules : horizon ? HbaoModules : SsaoModules;
        auto& pipelines = groundTruth ? GtaoPipelines : horizon ? HbaoPipelines : SsaoPipelines;
        return CreateFullscreenPipeline(shaders, modules, HdrRenderPass, pipelines[0]) &&
            CreateFullscreenPipeline(shaders, modules, AmbientOcclusionCompositeRenderPass,
                pipelines[1], PipelineLayout, FullscreenBlend::SubtractAmbient);
    }

    bool EnsureSaoDepthHierarchy()
    {
        const auto mipLevels = AmbientOcclusionDepthMipCount(MainColor.Width, MainColor.Height);
        if (!SaoSupported || mipLevels > SaoImageLimits.maxMipLevels ||
            MainColor.Width > SaoImageLimits.maxExtent.width || MainColor.Height > SaoImageLimits.maxExtent.height)
            return Fail("Vulkan SAO requires a sampled, renderable and transferable R32F depth hierarchy");
        if (SaoScratchFramebuffer && SaoDepthHierarchy.MipLevels == mipLevels) return true;
        if (!CreateFullscreenRenderPass(VK_FORMAT_R32_SFLOAT, SaoDepthRenderPass)) return false;
        // A distinct scratch attachment keeps the whole hierarchy read-only while
        // reducing it. Copies publish one mip at a time without feedback bindings.
        if (!CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                SaoScratch) ||
            !CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                SaoDepthHierarchy, mipLevels) ||
            !CreateFramebuffer(SaoDepthRenderPass, &SaoScratch.View, 1,
                MainColor.Width, MainColor.Height, SaoScratchFramebuffer))
        {
            if (SaoScratchFramebuffer && DestroyFramebufferFunction)
                DestroyFramebufferFunction(Device, SaoScratchFramebuffer, nullptr);
            SaoScratchFramebuffer = 0;
            DestroyRawImage(SaoScratch);
            DestroyRawImage(SaoDepthHierarchy);
            return false;
        }
        return true;
    }

    bool CreateLightShaftState()
    {
        if (!LightShaftDescriptorLayout)
        {
            std::array<VkDescriptorSetLayoutBinding, LightShaftInputCount + 1> bindings {};
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (std::uint32_t index = 1; index < bindings.size(); ++index)
                bindings[index] = {index, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount = static_cast<std::uint32_t>(bindings.size()); create.pBindings = bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device, &create, nullptr, &LightShaftDescriptorLayout), "vkCreateDescriptorSetLayout(light shafts)")) return false;
        }
        if (!LightShaftPipelineLayout)
        {
            VkPipelineLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            create.setLayoutCount = 1; create.pSetLayouts = &LightShaftDescriptorLayout;
            if (!Check(CreatePipelineLayoutFunction(Device, &create, nullptr, &LightShaftPipelineLayout), "vkCreatePipelineLayout(light shafts)")) return false;
        }
        if (!LightShaftDescriptorPool)
        {
            const std::array<VkDescriptorPoolSize, 2> sizes {{
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, LightShaftPhaseCount},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, LightShaftPhaseCount * LightShaftInputCount}}};
            VkDescriptorPoolCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            create.maxSets = LightShaftPhaseCount; create.poolSizeCount = static_cast<std::uint32_t>(sizes.size()); create.pPoolSizes = sizes.data();
            if (!Check(CreateDescriptorPoolFunction(Device, &create, nullptr, &LightShaftDescriptorPool), "vkCreateDescriptorPool(light shafts)")) return false;
        }
        if (!LightShaftSets[0])
        {
            std::array<VkDescriptorSetLayout, LightShaftPhaseCount> layouts; layouts.fill(LightShaftDescriptorLayout);
            VkDescriptorSetAllocateInfo allocate {};
            allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool = LightShaftDescriptorPool;
            allocate.descriptorSetCount = LightShaftPhaseCount; allocate.pSetLayouts = layouts.data();
            if (!Check(AllocateDescriptorSets(Device, &allocate, LightShaftSets.data()), "vkAllocateDescriptorSets(light shafts)")) return false;
        }
        if (!LightShaftSampler)
        {
            VkSamplerCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.minFilter = create.magFilter = VK_FILTER_NEAREST;
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &LightShaftSampler), "vkCreateSampler(light shafts)")) return false;
        }
        return true;
    }

    void DestroyLightShaftTargets()
    {
        if (LightShaftFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, LightShaftFramebuffer, nullptr);
        LightShaftFramebuffer = 0;
        DestroyRawImage(LightShaftMask);
        LightShaftResolved = false;
    }

    void DestroyLightShaftState()
    {
        DestroyLightShaftTargets();
        if (LightShaftPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, LightShaftPipeline, nullptr);
        for (auto module : LightShaftModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
        if (LightShaftSampler && DestroySamplerFunction) DestroySamplerFunction(Device, LightShaftSampler, nullptr);
        if (LightShaftDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, LightShaftDescriptorPool, nullptr);
        if (LightShaftPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, LightShaftPipelineLayout, nullptr);
        if (LightShaftDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, LightShaftDescriptorLayout, nullptr);
        LightShaftPipeline = LightShaftSampler = LightShaftDescriptorPool = LightShaftPipelineLayout = LightShaftDescriptorLayout = 0;
        LightShaftModules = {}; LightShaftSets = {}; LightShaftOffsets = {};
        LightShaftActive = LightShaftSupported = false; FrameLightShafts = {}; LightShaftImageLimits = {};
    }

    bool EnsureLightShaftResources()
    {
        if (!LightShaftSupported || !HdrSceneFramebuffer || MainColor.Width > LightShaftImageLimits.maxExtent.width ||
            MainColor.Height > LightShaftImageLimits.maxExtent.height)
            return Fail("Vulkan light shafts require floating HDR render/sample/transfer targets and sampled depth");
        if (!CreateLightShaftState() || !CreateFullscreenPipeline(VulkanLightShaftShaders, LightShaftModules,
                HdrRenderPass, LightShaftPipeline, LightShaftPipelineLayout)) return false;
        if (LightShaftFramebuffer && LightShaftMask.Width == MainColor.Width && LightShaftMask.Height == MainColor.Height) return true;
        RawImage candidate;
        VkFramebuffer framebuffer = 0;
        if (!CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, candidate) ||
            !CreateFramebuffer(HdrRenderPass, &candidate.View, 1, candidate.Width, candidate.Height, framebuffer))
        { DestroyRawImage(candidate); return false; }
        // Replacement precedes command recording and layout-journal entries.
        DestroyLightShaftTargets(); LightShaftMask = candidate; LightShaftFramebuffer = framebuffer;
        return true;
    }

    bool WriteLightShaftSet(std::size_t phase, const LightShaftConstants& constants,
        const std::array<VkImageView, LightShaftInputCount>& views)
    {
        const auto offset = AlignUp(UniformOffset, UniformAlignment);
        if (offset > UniformBufferSize || sizeof(constants) > UniformBufferSize - offset)
            return Fail("Vulkan light-shaft uniform arena is exhausted");
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
        UniformOffset = offset + sizeof(constants); LightShaftOffsets[phase] = static_cast<std::uint32_t>(offset);
        const VkDescriptorBufferInfo buffer {UniformBuffer.Buffer, 0, sizeof(constants)};
        std::array<VkDescriptorImageInfo, LightShaftInputCount> images {};
        std::array<VkWriteDescriptorSet, LightShaftInputCount + 1> writes {};
        for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
        {
            auto& write = writes[binding]; write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = LightShaftSets[phase]; write.dstBinding = binding; write.descriptorCount = 1;
            write.descriptorType = binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (binding == 0) write.pBufferInfo = &buffer;
            else
            {
                images[binding - 1] = {LightShaftSampler, views[binding - 1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                write.pImageInfo = &images[binding - 1];
            }
        }
        // Separate sets and aligned slices remain immutable until the frame
        // fence completes; composing never rewrites the mask draw's inputs.
        UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    bool CreateFogDescriptorState()
    {
        // Set 3 extends surface pipelines without disturbing material, cascade
        // or instance bindings. The producer uses the same layout at set 0.
        std::array<VkDescriptorSetLayoutBinding, RayShadowImageBinding + 1> bindings {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        for (std::uint32_t index = 1; index < bindings.size(); ++index)
            bindings[index] = {index, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[LumenVolumeUniformBinding]={LumenVolumeUniformBinding,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo layout {};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size()); layout.pBindings = bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device, &layout, nullptr, &FogDescriptorLayout), "vkCreateDescriptorSetLayout(fog)")) return false;
        VkPipelineLayoutCreateInfo pipeline {};
        pipeline.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline.setLayoutCount = 1; pipeline.pSetLayouts = &FogDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device, &pipeline, nullptr, &FogPipelineLayout), "vkCreatePipelineLayout(fog)")) return false;
        const std::array<VkDescriptorPoolSize, 3> sizes {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, FogSetCount},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, FogSetCount},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, FogSetCount * (FogInputCount+2)}}};
        VkDescriptorPoolCreateInfo pool {};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = FogSetCount; pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size()); pool.pPoolSizes = sizes.data();
        if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &FogDescriptorPool), "vkCreateDescriptorPool(fog)")) return false;
        std::array<VkDescriptorSetLayout, FogSetCount> layouts; layouts.fill(FogDescriptorLayout);
        VkDescriptorSetAllocateInfo allocate {};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = FogDescriptorPool; allocate.descriptorSetCount = FogSetCount; allocate.pSetLayouts = layouts.data();
        if (!Check(AllocateDescriptorSets(Device, &allocate, FogDescriptorSets.data()), "vkAllocateDescriptorSets(fog)")) return false;
        for (std::size_t index = 0; index < FogSamplers.size(); ++index)
        {
            VkSamplerCreateInfo sampler {};
            sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler.magFilter = sampler.minFilter = index == FogLinearSampler ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
            sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device, &sampler, nullptr, &FogSamplers[index]), "vkCreateSampler(fog)")) return false;
        }
        return true;
    }

    bool CreateVolumeLightDescriptorState()
    {
        if (!VolumeLightDescriptorLayout)
        {
            std::array<VkDescriptorSetLayoutBinding, VolumeDensityBinding + 1> bindings {};
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (std::uint32_t index = 1; index <= VolumeLightInputCount; ++index)
                bindings[index] = {index, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            bindings[VolumeSceneLightBinding] = {VolumeSceneLightBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            bindings[VolumeDensityBinding] = {VolumeDensityBinding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount = static_cast<std::uint32_t>(bindings.size()); create.pBindings = bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device, &create, nullptr, &VolumeLightDescriptorLayout),
                "vkCreateDescriptorSetLayout(volume lighting)")) return false;
        }
        if (!VolumeLightPipelineLayout)
        {
            VkPipelineLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            const std::array<VkDescriptorSetLayout,4> layouts{VolumeLightDescriptorLayout,CascadeDescriptorLayout,InstanceDescriptorLayout,FogDescriptorLayout};
            create.setLayoutCount=std::uint32_t(layouts.size()); create.pSetLayouts=layouts.data();
            if (!Check(CreatePipelineLayoutFunction(Device, &create, nullptr, &VolumeLightPipelineLayout),
                "vkCreatePipelineLayout(volume lighting)")) return false;
        }
        if (!VolumeLightDescriptorPool)
        {
            const std::array<VkDescriptorPoolSize, 3> sizes {{
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VolumeLightPassLimit},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VolumeLightPassLimit * (VolumeLightInputCount + 1)},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VolumeLightPassLimit}}};
            VkDescriptorPoolCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            create.maxSets = VolumeLightPassLimit; create.poolSizeCount = static_cast<std::uint32_t>(sizes.size()); create.pPoolSizes = sizes.data();
            if (!Check(CreateDescriptorPoolFunction(Device, &create, nullptr, &VolumeLightDescriptorPool),
                "vkCreateDescriptorPool(volume lighting)")) return false;
        }
        if (!VolumeLightSets[0])
        {
            std::array<VkDescriptorSetLayout, VolumeLightPassLimit> layouts; layouts.fill(VolumeLightDescriptorLayout);
            VkDescriptorSetAllocateInfo allocate {};
            allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool = VolumeLightDescriptorPool;
            allocate.descriptorSetCount = VolumeLightPassLimit; allocate.pSetLayouts = layouts.data();
            if (!Check(AllocateDescriptorSets(Device, &allocate, VolumeLightSets.data()), "vkAllocateDescriptorSets(volume lighting)")) return false;
        }
        return true;
    }

    void DestroyVolumeLightTargets()
    {
        for (auto& framebuffer : VolumeLightFramebuffers)
        {
            if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, framebuffer, nullptr);
            framebuffer = 0;
        }
        for (auto& image : VolumeLightImages) DestroyRawImage(image);
    }

    void DestroyVolumeLightState()
    {
        DestroyRayShadowPrograms(); DestroyPCFPrograms();
        for (auto& pipeline : CloudVolumePipelines)
            { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr); pipeline = 0; }
        for (auto& module : CloudVolumeModules)
            { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        for (auto& pipeline : MediaVolumePipelines)
        { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr); pipeline = 0; }
        for (auto& module : MediaVolumeModules)
        { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        if (PCSSVolumePipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, PCSSVolumePipeline, nullptr);
        PCSSVolumePipeline = 0;
        if (VSMVolumePipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, VSMVolumePipeline, nullptr);
        VSMVolumePipeline = 0;
        for (auto& module : VSMVolumeModules)
        { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        for (auto& module : PCSSVolumeModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        DestroyVolumeLightTargets();
        for (auto& pipeline : VolumeLightPipelines)
        {
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& module : VolumeLightModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        if (VolumeLightRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, VolumeLightRenderPass, nullptr);
        if (VolumeLightDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, VolumeLightDescriptorPool, nullptr);
        if (VolumeLightPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, VolumeLightPipelineLayout, nullptr);
        if (VolumeLightDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, VolumeLightDescriptorLayout, nullptr);
        VolumeLightRenderPass = VolumeLightDescriptorPool = VolumeLightPipelineLayout = VolumeLightDescriptorLayout = 0;
        VolumeLightSets.fill(0); FrameVolumeLighting = {};
    }

    bool EnsureVolumeLightResources()
    {
        const bool media = Techniques::UsesBoundedMedium(FrameFog);
        const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
        auto& programs = cloud ? CloudVolumePipelines : media ? MediaVolumePipelines : VolumeLightPipelines;
        auto& modules = cloud ? CloudVolumeModules : media ? MediaVolumeModules : VolumeLightModules;
        const auto& stages = cloud ? VulkanCloudTransportShaders : media ? VulkanMediaTransportShaders : VulkanVolumetricLightingShaders;
        if (!VolumeLightSupported || !FogFramebuffer) return Fail("Vulkan volume lighting requires FP32 transport targets and the cumulative atlas");
        if (!CreateVolumeLightDescriptorState() ||
            !CreateFullscreenRenderPass(VK_FORMAT_R32G32B32A32_SFLOAT, VolumeLightRenderPass) ||
            !CreateFullscreenPipeline(stages, modules, VolumeLightRenderPass,
                programs[0], VolumeLightPipelineLayout) ||
            !CreateFullscreenPipeline(stages, modules, HdrRenderPass,
                programs[1], VolumeLightPipelineLayout)) return false;
        if (VolumeLightFramebuffers[0] && VolumeLightFramebuffers[1]) return true;
        DestroyVolumeLightTargets();
        for (std::size_t index = 0; index < VolumeLightImages.size(); ++index)
        {
            auto& image = VolumeLightImages[index];
            if (!CreateRawImage(FogDimensions.AtlasWidth(), FogDimensions.AtlasHeight(), VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, image) ||
                !CreateFramebuffer(VolumeLightRenderPass, &image.View, 1, image.Width, image.Height, VolumeLightFramebuffers[index]))
            { DestroyVolumeLightTargets(); return false; }
        }
        return true;
    }

    bool GenerateVolumeLighting(const VolumetricFogDesc& requested, const VolumetricFogConstants& medium)
    {
        const bool media = Techniques::UsesBoundedMedium(FrameFog);
        const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
        auto& programs = cloud ? CloudVolumePipelines : media ? MediaVolumePipelines : VolumeLightPipelines;
        auto desc = requested;
        if (LightListsActive && !LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, LastError)) return false;
            if (LightListsActive) desc.Lighting.SceneLights = LightListFrame.Description().Lights;
        const auto shadow = Textures.find(LastShadowTexture.Value);
        if (!FrameRayShadowImage && !FrameVirtualShadowAtlas && !FrameCascades.Frame.CascadeCount)
            desc.SamplesShadowMap = desc.SamplesShadowMap && shadow != Textures.end() && shadow->second.Description.Format == TextureFormat::Depth32;
        VolumeLightConstants constants {};
        std::memcpy(constants.Medium.data(),&medium,constants.Medium.size());
        if (!BuildVolumetricLightingConstants(desc, FrameVolumeLighting, FrameCascades, FrameShadowFilter, constants.Lighting, LightListsActive, &FrameFog,
                FrameVirtualShadowAtlas ? &FrameVirtualShadowAtlas->Info : nullptr, &FrameVirtualShadowSampling,
                {medium.ViewportGrid.Z, medium.ViewportGrid.W}))
            return Fail("Vulkan volume lighting requires finite lights and shadow transforms");
        const bool pcss = FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS;
        const bool vsm = UsesVSM();
        if (!media && (pcss || vsm))
        {
            ShadowFilterBytes filterBytes {};
            if (!BuildShadowFilterData(desc.LightViewProjection, desc.Lighting, desc.SamplesShadowMap, filterBytes)) return false;
            std::memcpy(&constants.Lighting.Shadows, filterBytes.data(), filterBytes.size());
        }
        if (!media && pcss && !CreateFullscreenPipeline(VulkanPCSSScreenShaders[2], PCSSVolumeModules, VolumeLightRenderPass,
            PCSSVolumePipeline, VolumeLightPipelineLayout)) return false;
        if (!media && vsm && !CreateFullscreenPipeline(VulkanVSMScreenShaders[2], VSMVolumeModules, VolumeLightRenderPass,
            VSMVolumePipeline, VolumeLightPipelineLayout)) return false;
        const auto passes = BuildVolumetricLightingPasses(FogDimensions.Slices);
        const auto firstOffset = AlignUp(UniformOffset, UniformAlignment);
        const auto stride = AlignUp(sizeof(constants), UniformAlignment);
        const auto required = stride * passes.size();
        if (passes.empty() || passes.size() > VolumeLightPassLimit || firstOffset > UniformBufferSize || required > UniformBufferSize - firstOffset)
            return Fail("Vulkan volume lighting uniform arena is exhausted");
        RawImage* density=&DummyColor;
        if (desc.Density)
        {
            const auto found=Textures.find(desc.Density->Texture.Value);
            if (found==Textures.end() || !ValidDensityBinding(*desc.Density,found->second.Description))
                return Fail("Vulkan volume density requires a matching linear RGBA8 atlas");
            density=&found->second.Image;
        }
        std::array<RawImage*, VolumeLightInputCount> inputs {&DummyColor, &DummyDepth, &DummyDepth, &DummyDepth, &DummyDepth, &DummyDepth};
        const auto& lightControl = constants.Lighting.Control;
        if (FrameRayShadowImage) {}
        else if (lightControl.Y && FrameVirtualShadowAtlas)
        { inputs[2]=&FrameVirtualShadowAtlas->Depth; inputs[3]=&FrameVirtualShadowAtlas->Directory; }
        else if (lightControl.Y == 1)
        {
            inputs[1] = &shadow->second.Image;
            if (vsm && !MomentImage(LastShadowTexture, inputs[1])) return false;
        }
        else if (lightControl.Y == 2)
            for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index) inputs[index + 2] = CascadeImages[index];
        const VkDescriptorBufferInfo uniform {UniformBuffer.Buffer, 0, sizeof(constants)};
        const VkDescriptorBufferInfo scene {LightListsActive ? ForwardPlusSceneStorage.Buffer : ForwardPlusEmptyStorage.Buffer, 0,
            (LightListsActive ? std::max(constants.Lighting.Control.X, 1u) : 1u) * sizeof(ShaderLight)};
        for (std::size_t index = 0; index < passes.size(); ++index)
        {
            const bool inject = index == 0, final = index + 1 == passes.size();
            constants.Lighting.Control.W = passes[index].PrefixStride;
            const auto offset = firstOffset + index * stride;
            std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
            if (!inject) inputs = {&VolumeLightImages[(index - 1) % VolumeLightImages.size()], &DummyDepth, &DummyDepth, &DummyDepth, &DummyDepth, &DummyDepth};
            std::array<VkDescriptorImageInfo, VolumeLightInputCount + 1> images {};
            std::array<VkWriteDescriptorSet, VolumeDensityBinding + 1> writes {};
            for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
            {
                auto& write = writes[binding];
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = VolumeLightSets[index]; write.dstBinding = binding; write.descriptorCount = 1;
                write.descriptorType = binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC :
                    binding == VolumeSceneLightBinding ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                if (binding == 0) write.pBufferInfo = &uniform;
                else if (binding == VolumeSceneLightBinding) write.pBufferInfo = &scene;
                else
                {
                    const auto slot=binding==VolumeDensityBinding?VolumeLightInputCount:binding-1;
                    auto& image=*(binding==VolumeDensityBinding?density:inputs[slot]);
                    TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    images[slot] = {FogSamplers[FogPointSampler], image.View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                    write.pImageInfo = &images[slot];
                }
            }
            // A distinct set and UBO range per pass preserve every recorded bind.
            // Reuse across frames happens only after the previous submission fence.
            UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            auto& destination = final ? FogAtlas : VolumeLightImages[index % VolumeLightImages.size()];
            TransitionImage(CommandBuffer, destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            VkClearValue clear {};
            VkRenderPassBeginInfo begin {};
            begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            begin.renderPass = final ? HdrRenderPass : VolumeLightRenderPass;
            begin.framebuffer = final ? FogFramebuffer : VolumeLightFramebuffers[index % VolumeLightFramebuffers.size()];
            begin.renderArea = {{0, 0}, {destination.Width, destination.Height}};
            begin.clearValueCount = 1; begin.pClearValues = &clear;
            CmdBeginRenderPass(CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport {0, 0, float(destination.Width), float(destination.Height), 0, 1};
            CmdSetViewport(CommandBuffer, 0, 1, &viewport); CmdSetScissor(CommandBuffer, 0, 1, &begin.renderArea);
            CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                inject && !media && pcss ? PCSSVolumePipeline : inject && !media && vsm ? VSMVolumePipeline : programs[final ? 1 : 0]);
            const auto dynamicOffset = static_cast<std::uint32_t>(offset);
            CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VolumeLightPipelineLayout,
                0, 1, &VolumeLightSets[index], 1, &dynamicOffset);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VolumeLightPipelineLayout,FogDescriptorSetIndex,1,&FogDescriptorSets[FogDisabledSet],1,&FogUniformOffsets[FogDisabledSet]);
            constexpr std::uint32_t FullscreenVertexCount = 3;
            CmdDraw(CommandBuffer, FullscreenVertexCount, 1, 0, 0);
            CmdEndRenderPass(CommandBuffer);
        }
        UniformOffset = firstOffset + required;
        (Techniques::UsesVolumetricClouds(FrameFog) ? PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(FrameFog) ? PendingFrame.Statistics.ParticipatingMediaPassCount : PendingFrame.Statistics.VolumetricLightingPassCount) += static_cast<std::uint32_t>(passes.size());
        return true;
    }

    void DestroyFogTargets()
    {
        DestroyVolumeLightTargets();
        if (FogFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, FogFramebuffer, nullptr);
        FogFramebuffer = 0;
        DestroyRawImage(FogAtlas);
        FogDimensions = {};
        FogResolved = false;
    }

    void DestroyFogState()
    {
        if (CloudFogPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, CloudFogPipeline, nullptr);
        CloudFogPipeline = 0;
        for (auto& module : CloudFogModules)
            { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        if (MediaFogPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, MediaFogPipeline, nullptr);
        MediaFogPipeline = 0;
        for (auto& module : MediaFogModules)
        { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        DestroyVolumeLightState();
        if (FogPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, FogPipeline, nullptr);
        FogPipeline = 0;
        for (auto& module : FogModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        for (auto& sampler : FogSamplers)
        {
            if (sampler && DestroySamplerFunction) DestroySamplerFunction(Device, sampler, nullptr);
            sampler = 0;
        }
        if (FogDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, FogDescriptorPool, nullptr);
        if (FogPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, FogPipelineLayout, nullptr);
        if (FogDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, FogDescriptorLayout, nullptr);
        FogDescriptorPool = FogPipelineLayout = FogDescriptorLayout = 0;
        FogDescriptorSets.fill(0); FogUniformOffsets.fill(0);
        FogActive = FogResolved = false;
    }

    int FogTextureLimit() const
    {
        auto limit = std::min({DeviceLimits.maxImageDimension2D, FogImageLimits.maxExtent.width,
            FogImageLimits.maxExtent.height, DeviceLimits.maxViewportDimensions[0], DeviceLimits.maxViewportDimensions[1],
            DeviceLimits.maxFramebufferWidth, DeviceLimits.maxFramebufferHeight});
        if ((Techniques::UsesVolumetricLighting(FrameVolumeLighting) || Techniques::UsesBoundedMedium(FrameFog)))
            limit = std::min({limit, VolumeLightImageLimits.maxExtent.width, VolumeLightImageLimits.maxExtent.height});
        return static_cast<int>(limit);
    }

    bool EnsureFogResources()
    {
        if (!FogSupported || !HdrSceneFramebuffer || MainColor.Width > FogImageLimits.maxExtent.width ||
            MainColor.Height > FogImageLimits.maxExtent.height)
            return Fail("Vulkan fog requires filtered float targets, sampled depth and HDR transfers");
        const bool media = Techniques::UsesBoundedMedium(FrameFog);
        const bool cloud = Techniques::UsesVolumetricClouds(FrameFog);
        if (!CreateFullscreenPipeline(cloud ? VulkanCloudCompositionShaders : media ? VulkanMediaCompositionShaders : VulkanVolumetricFogShaders,
            cloud ? CloudFogModules : media ? MediaFogModules : FogModules, HdrRenderPass, cloud ? CloudFogPipeline : media ? MediaFogPipeline : FogPipeline, FogPipelineLayout)) return false;
        const auto grid = BuildFogGrid(CreateInfo.Width,CreateInfo.Height,FrameFog,FogTextureLimit());
        if (grid.Width == 0) return Fail("Vulkan fog atlas dimensions exceed the device limits");
        if (FogFramebuffer && FogDimensions.Width == grid.Width && FogDimensions.Height == grid.Height &&
            FogDimensions.Slices == grid.Slices && FogDimensions.Planes == grid.Planes) return true;
        RawImage candidate;
        VkFramebuffer framebuffer = 0;
        if (!CreateRawImage(grid.AtlasWidth(), grid.AtlasHeight(), VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, candidate) ||
            !CreateFramebuffer(HdrRenderPass, &candidate.View, 1, candidate.Width, candidate.Height, framebuffer))
        { DestroyRawImage(candidate); return false; }
        // Resource replacement happens after the frame fence and before any
        // commands or image-layout journal entries reference the new atlas.
        DestroyFogTargets(); FogAtlas = candidate; FogFramebuffer = framebuffer; FogDimensions = grid;
        return true;
    }

    bool WriteFogSet(std::size_t index, const VolumetricFogConstants& constants,
        const std::array<VkImageView, FogInputCount>& views)
    {
        const auto offset=AlignUp(UniformOffset,UniformAlignment),volumeOffset=AlignUp(offset+sizeof(constants),UniformAlignment);
        if (volumeOffset>UniformBufferSize || sizeof(LumenVolumeFrame)>UniformBufferSize-volumeOffset)
            return Fail("Vulkan surface-volume uniform arena is exhausted");
        auto volume=LumenVolumeFrame; const bool lit=FrameGI.Lumen.Translucency.Enabled && index>=LumenVolumeFirstReceiverSet && volume.Grid.W!=0;
        if (!lit) volume.Grid.W=0;
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&constants,sizeof(constants));
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+volumeOffset,&volume,sizeof(volume));
        UniformOffset=volumeOffset+sizeof(volume); FogUniformOffsets[index]=std::uint32_t(offset);
        const VkDescriptorBufferInfo fogInfo{UniformBuffer.Buffer,0,sizeof(constants)},volumeInfo{UniformBuffer.Buffer,volumeOffset,sizeof(volume)};
        std::array<VkDescriptorImageInfo,FogInputCount+2> images{};
        for (unsigned input=0;input<FogInputCount;++input)
            images[input]={FogSamplers[input+1==FogInputCount?FogLinearSampler:FogPointSampler],views[input],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[FogInputCount]={FogSamplers[FogLinearSampler],lit?LumenVolumeAtlas.View:DummyColor.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images.back()=RayShadowImageDescriptor();
        std::array<VkWriteDescriptorSet,RayShadowImageBinding+1> writes{};
        for (unsigned binding=0;binding<writes.size();++binding)
        {
            auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet=FogDescriptorSets[index]; write.dstBinding=binding; write.descriptorCount=1;
            if (binding==0) { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&fogInfo; }
            else if (binding==LumenVolumeUniformBinding) { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&volumeInfo; }
            else { write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&images[binding==RayShadowImageBinding?FogInputCount+1:binding==LumenVolumeImageBinding?FogInputCount:binding-1]; }
        }
        // Producers, disabled surfaces and the two lit-transparent variants own
        // distinct sets. Never rewrite descriptors already referenced this frame.
        UpdateDescriptorSets(Device,std::uint32_t(writes.size()),writes.data(),0,nullptr); return true;
    }

    void DestroyHdrTargets()
    {
        DestroySubsurfaceTargets();
        DestroyExposureTargets();
        DestroyPathTargets();
        DestroyLightShaftTargets();
        DestroyWeightedTargets();
        DestroyFogTargets();
        for (auto& framebuffer : HdrFramebuffers)
        {
            if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, framebuffer, nullptr);
            framebuffer = 0;
        }
        if (HdrSceneFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, HdrSceneFramebuffer, nullptr);
        if (HdrResolveFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, HdrResolveFramebuffer, nullptr);
        HdrSceneFramebuffer = HdrResolveFramebuffer = 0;
        for (auto& image : HdrImages) DestroyRawImage(image);
    }

    bool EnsureHdrTargets()
    {
        if (HdrSceneFramebuffer) return true;
        VkFormatProperties properties{};
        GetPhysicalDeviceFormatProperties(PhysicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT, &properties);
        constexpr VkFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
        if ((properties.optimalTilingFeatures & required) != required)
            return Fail("Vulkan HDR requires sampled, filtered, blendable RGBA16F color attachments");
        for (std::size_t mask = 0; mask < HdrColorRenderPasses.size(); ++mask)
            if (!HdrColorRenderPasses[mask] && !CreateColorRenderPass((mask & 1U) != 0, (mask & 2U) != 0,
                HdrColorRenderPasses[mask], VK_FORMAT_R16G16B16A16_SFLOAT)) return false;
        if (!EnsurePostRenderPass() || !CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT, HdrRenderPass)) return false;
        for (std::size_t index = 0; index < HdrImages.size(); ++index)
        {
            const auto width = index >= 2 ? (MainColor.Width + 1U) / 2U : MainColor.Width;
            const auto height = index >= 2 ? (MainColor.Height + 1U) / 2U : MainColor.Height;
            if (!CreateRawImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    (index == 0 ? VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT : index == 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0), HdrImages[index]) ||
                !CreateFramebuffer(HdrRenderPass, &HdrImages[index].View, 1, width, height, HdrFramebuffers[index]))
            {
                DestroyHdrTargets();
                return false;
            }
        }
        const std::array<VkImageView, 2> attachments{HdrImages[0].View, MainDepth.View};
        if (!CreateFramebuffer(HdrColorRenderPasses[0], attachments.data(), static_cast<std::uint32_t>(attachments.size()),
                MainColor.Width, MainColor.Height, HdrSceneFramebuffer) ||
            !CreateFramebuffer(PostRenderPass, &MainColor.View, 1, MainColor.Width, MainColor.Height, HdrResolveFramebuffer))
        {
            DestroyHdrTargets();
            return false;
        }
        return true;
    }

    void DestroyMotionTargets()
    {
        for (auto& framebuffer : MotionFramebuffers)
        {
            if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, framebuffer, nullptr);
            framebuffer = 0;
        }
        for (auto& image : MotionImages) DestroyRawImage(image);
        DestroyRawBuffer(MotionReadback);
    }

    bool EnsureVelocityResources()
    {
        if (!HdrActive || !HdrSceneFramebuffer)
            return Fail("Vulkan velocity requires the main HDR scene");
        VkFormatProperties depth {};
        GetPhysicalDeviceFormatProperties(PhysicalDevice, VK_FORMAT_D32_SFLOAT, &depth);
        if ((depth.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0)
            return Fail("Vulkan velocity requires sampled D32 depth");
        // The velocity replay loads a read-only depth attachment. Filtering then
        // samples it in depthless passes, with every layout transition journaled
        // by TransitionImage until submission (including cancellation recovery).
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkImageLayout.html
        if (!MotionGeometryRenderPass && !CreateColorRenderPass(true, false, MotionGeometryRenderPass,
                VK_FORMAT_R16G16B16A16_SFLOAT, true)) return false;
        for (std::size_t stage = 0; stage < MotionGeometryModules.size(); ++stage)
        {
            if (MotionGeometryModules[stage]) continue;
            std::vector<std::uint32_t> compiledWords;
            if (!CompileVulkanShader(VulkanMotionBlurGeometryShaders[stage], compiledWords, LastError)) return false;
            VkShaderModuleCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            create.codeSize = compiledWords.size() * sizeof(std::uint32_t); create.pCode = compiledWords.data();
            if (!Check(CreateShaderModuleFunction(Device, &create, nullptr, &MotionGeometryModules[stage]),
                    "vkCreateShaderModule(velocity)")) return false;
        }
        if (MotionFramebuffers[MotionVelocity]) return true;
        if (!CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                MotionImages[MotionVelocity])) return false;
        const std::array<VkImageView, 2> views {MotionImages[MotionVelocity].View, MainDepth.View};
        if (!CreateFramebuffer(MotionGeometryRenderPass, views.data(), static_cast<std::uint32_t>(views.size()),
                MainColor.Width, MainColor.Height, MotionFramebuffers[MotionVelocity]))
        { DestroyMotionTargets(); return false; }
        constexpr std::size_t HalfFloatPixelBytes = 4 * sizeof(std::uint16_t);
        if (CreateInfo.CaptureDiagnosticBuffers && !CreateRawBuffer(
                static_cast<VkDeviceSize>(MainColor.Width) * MainColor.Height * HalfFloatPixelBytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                MotionReadback)) { DestroyMotionTargets(); return false; }
        return true;
    }

    bool EnsureMotionResources()
    {
        if (!EnsureVelocityResources() ||
            !CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT, MotionCompositeRenderPass, true)) return false;
        if (!MotionSampler)
        {
            VkSamplerCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.magFilter = create.minFilter = VK_FILTER_NEAREST;
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &MotionSampler),
                    "vkCreateSampler(motion blur)")) return false;
        }
        if (!CreateFullscreenPipeline(VulkanMotionBlurFilterShaders, MotionFilterModules,
                HdrRenderPass, MotionFilterPipelines[0]) ||
            !CreateFullscreenPipeline(VulkanMotionBlurFilterShaders, MotionFilterModules,
                MotionCompositeRenderPass, MotionFilterPipelines[1], PipelineLayout, FullscreenBlend::Replace, true)) return false;
        if (MotionFramebuffers[MotionNeighborMax]) return true;
        constexpr std::uint32_t tileSize = Techniques::MotionBlurTileSize;
        for (std::size_t index = MotionTileMax; index < MotionImages.size(); ++index)
        {
            const auto width = (MainColor.Width + tileSize - 1) / tileSize;
            const auto height = (MainColor.Height + tileSize - 1) / tileSize;
            if (!CreateRawImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    MotionImages[index])) { DestroyMotionTargets(); return false; }
            if (!CreateFramebuffer(HdrRenderPass, &MotionImages[index].View, 1, width, height, MotionFramebuffers[index]))
            { DestroyMotionTargets(); return false; }
        }
        return true;
    }

    void DestroyTemporalTargets()
    {
        for (auto& framebuffer : TemporalFramebuffers)
        {
            if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, framebuffer, nullptr);
            framebuffer = 0;
        }
        for (auto& image : TemporalColors) DestroyRawImage(image);
        for (auto& image : TemporalDepths) DestroyRawImage(image);
        TemporalHistoryValid = false;
        TemporalHistoryReadIndex = 0;
    }

    void DestroyTemporalState()
    {
        if (TemporalPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, TemporalPipeline, nullptr);
        TemporalPipeline = 0;
        for (auto& module : TemporalModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        for (auto& sampler : TemporalSamplers)
        {
            if (sampler && DestroySamplerFunction) DestroySamplerFunction(Device, sampler, nullptr);
            sampler = 0;
        }
        if (TemporalDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, TemporalDescriptorPool, nullptr);
        if (TemporalPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, TemporalPipelineLayout, nullptr);
        if (TemporalDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, TemporalDescriptorLayout, nullptr);
        if (TemporalRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, TemporalRenderPass, nullptr);
        TemporalDescriptorPool = TemporalDescriptorSet = TemporalPipelineLayout = TemporalDescriptorLayout = TemporalRenderPass = 0;
        TemporalActive = TemporalResolved = false;
    }

    #include "Renderer/Vulkan/VulkanLightmaps.inl"
    #include "Renderer/Vulkan/VulkanRefraction.inl"
    #include "Renderer/Vulkan/VulkanDepthPeeling.inl"

    void DestroyProbeTargets()
    {
        for (auto& framebuffer:ProbeFramebuffers)
        { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,framebuffer,nullptr); framebuffer=0; }
        for (auto& image:ProbeImages) DestroyRawImage(image);
        DestroyRawBuffer(ProbeUpload); UploadedProbeSamples.reset(); Probes.Invalidate();
    }
    void DestroyProbeState()
    {
        for (auto& pipeline:ProbePipelines)
        { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
        for (auto& pair:ProbeModules) for (auto& module:pair)
        { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
        if (ProbeSampler && DestroySamplerFunction) DestroySamplerFunction(Device,ProbeSampler,nullptr);
        if (ProbeDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,ProbeDescriptorPool,nullptr);
        if (ProbePipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,ProbePipelineLayout,nullptr);
        if (ProbeDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,ProbeDescriptorLayout,nullptr);
        ProbeSampler=ProbeDescriptorPool=ProbePipelineLayout=ProbeDescriptorLayout=0;
        ProbeDescriptorSets.fill(0); FrameProbeSamples.reset(); Probes.Invalidate();
    }
    bool EnsureProbeResources()
    {
        if (!FrameProbeSamples || FrameProbeSamples->Texels.empty() || !HdrActive) return Fail("Vulkan probes require HDR and baked samples");
        if (!ProbeDescriptorLayout)
        {
            std::array<VkDescriptorSetLayoutBinding,ProbeInputCount+2> bindings {};
            bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
            for (std::uint32_t slot=1;slot<=ProbeInputCount;++slot) bindings[slot]={slot,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
            bindings.back()={ProbeInputCount+1,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
            VkDescriptorSetLayoutCreateInfo create {}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&ProbeDescriptorLayout),"vkCreateDescriptorSetLayout(probes)")) return false;
        }
        if (!ProbePipelineLayout)
        {
            VkPipelineLayoutCreateInfo create {}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            create.setLayoutCount=1; create.pSetLayouts=&ProbeDescriptorLayout;
            if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&ProbePipelineLayout),"vkCreatePipelineLayout(probes)")) return false;
        }
        if (!ProbeSampler)
        {
            VkSamplerCreateInfo create {}; create.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.magFilter=create.minFilter=VK_FILTER_NEAREST;
            create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device,&create,nullptr,&ProbeSampler),"vkCreateSampler(probes)")) return false;
        }
        if (!ProbeDescriptorPool)
        {
            const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,ProbePhaseCount},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,ProbeInputCount*ProbePhaseCount},{VK_DESCRIPTOR_TYPE_SAMPLER,ProbePhaseCount}};
            VkDescriptorPoolCreateInfo create {}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            create.maxSets=ProbePhaseCount; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
            if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&ProbeDescriptorPool),"vkCreateDescriptorPool(probes)")) return false;
        }
        for (auto& set:ProbeDescriptorSets)
        {
            if (set) continue;
            VkDescriptorSetAllocateInfo allocate {}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool=ProbeDescriptorPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&ProbeDescriptorLayout;
            if (!Check(AllocateDescriptorSets(Device,&allocate,&set),"vkAllocateDescriptorSets(probes)")) return false;
        }
        if (!CreateFullscreenPipeline(VulkanBakedGIShaders,ProbeModules[0],HdrRenderPass,ProbePipelines[0],ProbePipelineLayout) ||
            !CreateFullscreenPipeline(VulkanLightProbeShaders,ProbeModules[1],HdrRenderPass,ProbePipelines[1],ProbePipelineLayout)) return false;
        const auto count=FrameProbeSamples->ProbeCount();
        if (ProbeImages[ProbeRaw].Width!=FrameProbeSamples->Width() || ProbeImages[ProbeSH].Height!=count ||
            ProbeImages[ProbeComposite].Width!=MainColor.Width || ProbeImages[ProbeComposite].Height!=MainColor.Height) DestroyProbeTargets();
        if (!ProbeImages[ProbeRaw].Image)
        {
            const std::uint32_t widths[] {FrameProbeSamples->Width(),ProbeSHCoefficientCount,ProbeMomentAtlasColumns*ProbeMomentTileSize,MainColor.Width};
            const std::uint32_t heights[] {FrameProbeSamples->Height(),count,
                ((count+ProbeMomentAtlasColumns-1)/ProbeMomentAtlasColumns)*ProbeMomentTileSize,MainColor.Height};
            for (std::size_t index=0;index<ProbeImages.size();++index)
            {
                const auto format=index==ProbeRaw?VK_FORMAT_R32G32B32A32_SFLOAT:VK_FORMAT_R16G16B16A16_SFLOAT;
                const auto usage=VK_IMAGE_USAGE_SAMPLED_BIT | (index==ProbeRaw?VK_IMAGE_USAGE_TRANSFER_DST_BIT:
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (index==ProbeComposite?VK_IMAGE_USAGE_TRANSFER_SRC_BIT:0));
                if (!CreateRawImage(widths[index],heights[index],format,VK_IMAGE_ASPECT_COLOR_BIT,usage,ProbeImages[index]) ||
                    (index!=ProbeRaw && !CreateFramebuffer(HdrRenderPass,&ProbeImages[index].View,1,widths[index],heights[index],ProbeFramebuffers[index])))
                { DestroyProbeTargets(); return false; }
            }
            Probes.Invalidate();
        }
        if (UploadedProbeSamples!=FrameProbeSamples)
        {
            DestroyRawBuffer(ProbeUpload);
            const auto size=FrameProbeSamples->Texels.size()*sizeof(Float4);
            if (!CreateRawBuffer(size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,ProbeUpload)) return false;
            std::memcpy(ProbeUpload.Mapped,FrameProbeSamples->Texels.data(),size);
            UploadedProbeSamples=FrameProbeSamples; Probes.Invalidate();
        }
        return true;
    }
    bool ResolveProbes(const GlobalIlluminationDesc& desc)
    {
        LightProbeConstants constants;
        if (!FrameProbeSamples || !BuildLightProbeConstants(desc,*FrameProbeSamples,FrameGI,FrameReflectionShading,
            CreateInfo.Width,CreateInfo.Height,constants)) return Fail("Vulkan probes require finite geometry and sample inputs");
        auto& raw=ProbeImages[ProbeRaw]; auto& scene=HdrImages[0];
        const bool projection=Probes.NeedsProjection(FrameProbeSamples,constants.Visibility.Y);
        if (projection)
        {
            Probes.Invalidate(); TransitionImage(CommandBuffer,raw,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy copy {}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent={raw.Width,raw.Height,1};
            CmdCopyBufferToImage(CommandBuffer,ProbeUpload.Buffer,raw.Image,raw.Layout,1,&copy);
            TransitionImage(CommandBuffer,raw,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        auto draw=[&](ProbePass phase,std::size_t targetIndex) {
            const auto phaseIndex=static_cast<std::uint32_t>(phase); constants.Options.X=phaseIndex;
            const auto offset=AlignUp(UniformOffset,UniformAlignment);
            if (offset>UniformBufferSize || sizeof(constants)>UniformBufferSize-offset) return Fail("Vulkan probe uniform arena is exhausted");
            std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&constants,sizeof(constants)); UniformOffset=offset+sizeof(constants);
            const bool gather=phase==ProbePass::Gather;
            const std::array<RawImage*,ProbeInputCount> inputs {&raw,gather?&ProbeImages[ProbeSH]:&raw,gather?&ProbeImages[ProbeMoments]:&raw,
                gather?&scene:&raw,gather?&MainDepth:&raw,gather?&GBufferImages[1]:&raw,gather?&GBufferImages[0]:&raw,gather?&GBufferImages[3]:&raw};
            std::array<VkDescriptorImageInfo,ProbeInputCount+1> images {};
            for (std::size_t slot=0;slot<inputs.size();++slot)
            {
                TransitionImage(CommandBuffer,*inputs[slot],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                images[slot]={0,inputs[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
            images.back()={ProbeSampler,0,VK_IMAGE_LAYOUT_UNDEFINED};
            const VkDescriptorBufferInfo buffer {UniformBuffer.Buffer,0,sizeof(constants)};
            const auto descriptor=ProbeDescriptorSets[phaseIndex];
            std::array<VkWriteDescriptorSet,ProbeInputCount+2> writes {};
            for (std::uint32_t slot=0;slot<writes.size();++slot)
            {
                auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet=descriptor; write.dstBinding=slot; write.descriptorCount=1;
                write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:slot==ProbeInputCount+1?VK_DESCRIPTOR_TYPE_SAMPLER:VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                if (slot==0) write.pBufferInfo=&buffer; else write.pImageInfo=&images[slot-1];
            }
            // Separate sets are essential: later recorded phases cannot mutate
            // descriptors referenced by earlier projection commands.
            UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
            auto& target=ProbeImages[targetIndex]; TransitionImage(CommandBuffer,target,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            const VkClearValue clear {}; VkRenderPassBeginInfo begin {}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            begin.renderPass=HdrRenderPass; begin.framebuffer=ProbeFramebuffers[targetIndex]; begin.renderArea={{0,0},{target.Width,target.Height}};
            begin.clearValueCount=1; begin.pClearValues=&clear;
            CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport {0,0,float(target.Width),float(target.Height),0,1}; CmdSetViewport(CommandBuffer,0,1,&viewport);
            CmdSetScissor(CommandBuffer,0,1,&begin.renderArea); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,ProbePipelines[gather?1:0]);
            const auto dynamicOffset=static_cast<std::uint32_t>(offset);
            CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,ProbePipelineLayout,0,1,&descriptor,1,&dynamicOffset);
            CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer); return true;
        };
        if (projection)
        {
            if (!draw(ProbePass::ProjectIrradiance,ProbeSH) || !draw(ProbePass::ProjectDistance,ProbeMoments)) return false;
            Probes.Projected(FrameProbeSamples,constants.Visibility.Y);
        }
        if (!draw(ProbePass::Gather,ProbeComposite)) return false;
        auto& composite=ProbeImages[ProbeComposite]; TransitionImage(CommandBuffer,composite,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        TransitionImage(CommandBuffer,scene,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy copy {}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={scene.Width,scene.Height,1};
        CmdCopyImage(CommandBuffer,composite.Image,composite.Layout,scene.Image,scene.Layout,1,&copy);
        Probes.Resolved=GI.Resolved=true; PendingFrame.Statistics.GlobalIlluminationPassCount=projection?3:1; return true;
    }

    bool EnsureGIResources()
    {
        if (!DeferredSupported || !TemporalSupported || !HdrActive || MainColor.Width > TemporalColorLimits.maxExtent.width ||
            MainColor.Height > TemporalColorLimits.maxExtent.height || MainColor.Width > TemporalDepthLimits.maxExtent.width ||
            MainColor.Height > TemporalDepthLimits.maxExtent.height)
            return Fail("Vulkan SSGI requires renderable, sampled color/depth history and transferable HDR color");
        if (!GIRenderPass)
        {
            std::array<VkAttachmentDescription, GIAttachmentCount> attachments {};
            std::array<VkAttachmentReference, GIAttachmentCount> references {};
            for (std::uint32_t index = 0; index < GIAttachmentCount; ++index)
            {
                auto& attachment = attachments[index];
                attachment.format = index == 1 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
                attachment.samples = 1;
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                references[index] = {index, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            }
            VkSubpassDescription subpass {};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = GIAttachmentCount;
            subpass.pColorAttachments = references.data();
            // Explicit image barriers outside the pass supply visibility and
            // transitions; the pass keeps all attachments in the same layout.
            VkRenderPassCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            create.attachmentCount = GIAttachmentCount;
            create.pAttachments = attachments.data();
            create.subpassCount = 1;
            create.pSubpasses = &subpass;
            if (!Check(CreateRenderPassFunction(Device, &create, nullptr, &GIRenderPass), "vkCreateRenderPass(SSGI)")) return false;
        }
        if (!GIDescriptorLayout)
        {
            std::array<VkDescriptorSetLayoutBinding, GIInputCount + 2> bindings {};
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (std::uint32_t i = 1; i <= GIInputCount; ++i)
                bindings[i] = {i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            bindings[GIInputCount + 1] = {GIInputCount + 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount = static_cast<std::uint32_t>(bindings.size()); create.pBindings = bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device, &create, nullptr, &GIDescriptorLayout), "vkCreateDescriptorSetLayout(SSGI)")) return false;
        }
        if (!GIPipelineLayout)
        {
            VkPipelineLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            create.setLayoutCount = 1; create.pSetLayouts = &GIDescriptorLayout;
            if (!Check(CreatePipelineLayoutFunction(Device, &create, nullptr, &GIPipelineLayout), "vkCreatePipelineLayout(SSGI)")) return false;
        }
        if (!GISampler)
        {
            VkSamplerCreateInfo create {}; create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.magFilter = create.minFilter = VK_FILTER_NEAREST;
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &GISampler), "vkCreateSampler(SSGI)")) return false;
        }
        if (!GIDescriptorPool)
        {
            const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, GIPhaseCount},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GIInputCount * GIPhaseCount}, {VK_DESCRIPTOR_TYPE_SAMPLER, GIPhaseCount}};
            VkDescriptorPoolCreateInfo create {}; create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            create.maxSets = GIPhaseCount; create.poolSizeCount = static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes = sizes;
            if (!Check(CreateDescriptorPoolFunction(Device, &create, nullptr, &GIDescriptorPool), "vkCreateDescriptorPool(SSGI)")) return false;
        }
        for (auto& set : GIDescriptorSets)
        {
            if (set) continue;
            VkDescriptorSetAllocateInfo allocate {}; allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool = GIDescriptorPool; allocate.descriptorSetCount = 1; allocate.pSetLayouts = &GIDescriptorLayout;
            if (!Check(AllocateDescriptorSets(Device, &allocate, &set), "vkAllocateDescriptorSets(SSGI)")) return false;
        }
        if (!CreateFullscreenPipeline(VulkanGIShaders, GIModules, HdrRenderPass, GIPipeline, GIPipelineLayout) ||
            !CreateFullscreenPipeline(VulkanGITemporalShaders, GITemporalModules, GIRenderPass, GITemporalPipeline,
                GIPipelineLayout, FullscreenBlend::Replace, false, 1, false, GIAttachmentCount)) return false;
        if (GIFramebuffers[0]) return true;
        for (std::size_t i = 0; i < GIImages.size(); ++i)
        {
            const bool depth = i >= GIDepthFirst && i < GINormalFirst;
            const auto usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                (i == GIComposite ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
            if (!CreateRawImage(MainColor.Width, MainColor.Height, depth ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_ASPECT_COLOR_BIT, usage, GIImages[i])) { DestroyGITargets(); return false; }
        }
        for (std::size_t i = 0; i < GIFramebuffers.size(); ++i)
        {
            const bool history = i >= GIColorFirst;
            const std::size_t slot = history ? i - GIColorFirst : 0;
            const VkImageView views[] {GIImages[i].View, GIImages[GIDepthFirst + slot].View, GIImages[GINormalFirst + slot].View};
            if (!CreateFramebuffer(history ? GIRenderPass : HdrRenderPass, views, history ? GIAttachmentCount : 1,
                MainColor.Width, MainColor.Height, GIFramebuffers[i])) { DestroyGITargets(); return false; }
        }
        GI.Invalidate(); return true;
    }

    void DestroyGITargets()
    {
        for (auto& framebuffer : GIFramebuffers)
        { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, framebuffer, nullptr); framebuffer = 0; }
        for (auto& image : GIImages) DestroyRawImage(image);
        GI.Invalidate();
    }
    void DestroyGIState()
    {
        for (auto* pipeline : {&GIPipeline, &GITemporalPipeline})
        { if (*pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, *pipeline, nullptr); *pipeline = 0; }
        for (auto* modules : {&GIModules, &GITemporalModules})
            for (auto& module : *modules)
            { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        if (GISampler && DestroySamplerFunction) DestroySamplerFunction(Device, GISampler, nullptr);
        if (GIDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, GIDescriptorPool, nullptr);
        if (GIPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, GIPipelineLayout, nullptr);
        if (GIDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, GIDescriptorLayout, nullptr);
        if (GIRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, GIRenderPass, nullptr);
        GISampler = GIDescriptorPool = GIPipelineLayout = GIDescriptorLayout = GIRenderPass = 0;
        GIDescriptorSets.fill(0); GIActive = false; GI.Invalidate();
    }

    bool EnsureTemporalResources()
    {
        if (!TemporalSupported || !HdrActive || MainColor.Width > TemporalColorLimits.maxExtent.width ||
            MainColor.Height > TemporalColorLimits.maxExtent.height || MainColor.Width > TemporalDepthLimits.maxExtent.width ||
            MainColor.Height > TemporalDepthLimits.maxExtent.height)
            return Fail("Vulkan TAA requires renderable, sampled color/depth history and transferable HDR color");
        if (!TemporalRenderPass)
        {
            std::array<VkAttachmentDescription, TemporalAttachmentCount> attachments {};
            std::array<VkAttachmentReference, TemporalAttachmentCount> references {};
            for (std::uint32_t index = 0; index < TemporalAttachmentCount; ++index)
            {
                auto& attachment = attachments[index];
                attachment.format = index == 0 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32_SFLOAT;
                attachment.samples = 1;
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                references[index] = {index, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            }
            VkSubpassDescription subpass {};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = TemporalAttachmentCount;
            subpass.pColorAttachments = references.data();
            // Explicit image barriers outside the pass supply visibility and
            // transitions; the pass keeps both attachments in the same layout.
            VkRenderPassCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            create.attachmentCount = TemporalAttachmentCount;
            create.pAttachments = attachments.data();
            create.subpassCount = 1;
            create.pSubpasses = &subpass;
            if (!Check(CreateRenderPassFunction(Device, &create, nullptr, &TemporalRenderPass), "vkCreateRenderPass(TAA)")) return false;
        }
        if (!TemporalDescriptorLayout)
        {
            std::array<VkDescriptorSetLayoutBinding, TemporalInputCount + 1> bindings {};
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (std::uint32_t index = 1; index < bindings.size(); ++index)
                bindings[index] = {index, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount = static_cast<std::uint32_t>(bindings.size());
            create.pBindings = bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device, &create, nullptr, &TemporalDescriptorLayout), "vkCreateDescriptorSetLayout(TAA)")) return false;
        }
        if (!TemporalPipelineLayout)
        {
            VkPipelineLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            create.setLayoutCount = 1;
            create.pSetLayouts = &TemporalDescriptorLayout;
            if (!Check(CreatePipelineLayoutFunction(Device, &create, nullptr, &TemporalPipelineLayout), "vkCreatePipelineLayout(TAA)")) return false;
        }
        for (std::size_t index = 0; index < TemporalSamplers.size(); ++index)
        {
            if (TemporalSamplers[index]) continue;
            VkSamplerCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.magFilter = create.minFilter = index == TemporalLinearSampler ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &TemporalSamplers[index]), "vkCreateSampler(TAA)")) return false;
        }
        if (!TemporalDescriptorPool)
        {
            const std::array<VkDescriptorPoolSize, 2> sizes {{
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, TemporalInputCount}}};
            VkDescriptorPoolCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            create.maxSets = 1;
            create.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
            create.pPoolSizes = sizes.data();
            if (!Check(CreateDescriptorPoolFunction(Device, &create, nullptr, &TemporalDescriptorPool), "vkCreateDescriptorPool(TAA)")) return false;
        }
        if (!TemporalDescriptorSet)
        {
            VkDescriptorSetAllocateInfo allocate {};
            allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool = TemporalDescriptorPool;
            allocate.descriptorSetCount = 1;
            allocate.pSetLayouts = &TemporalDescriptorLayout;
            if (!Check(AllocateDescriptorSets(Device, &allocate, &TemporalDescriptorSet), "vkAllocateDescriptorSets(TAA)")) return false;
        }
        if (!CreateFullscreenPipeline(VulkanTemporalAAShaders, TemporalModules, TemporalRenderPass, TemporalPipeline,
                TemporalPipelineLayout, FullscreenBlend::Replace, false, 1, false, TemporalAttachmentCount)) return false;
        if (TemporalFramebuffers[0]) return true;
        for (std::size_t index = 0; index < TemporalFramebuffers.size(); ++index)
        {
            if (!CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, TemporalColors[index]) ||
                !CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, TemporalDepths[index]))
            { DestroyTemporalTargets(); return false; }
            const std::array<VkImageView, TemporalAttachmentCount> views {TemporalColors[index].View, TemporalDepths[index].View};
            if (!CreateFramebuffer(TemporalRenderPass, views.data(), TemporalAttachmentCount,
                    MainColor.Width, MainColor.Height, TemporalFramebuffers[index]))
            { DestroyTemporalTargets(); return false; }
        }
        return true;
    }

    void DestroyMotionState()
    {
        for (auto& pipeline : MotionFilterPipelines)
        {
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto* modules : {&MotionGeometryModules, &MotionFilterModules})
            for (auto& module : *modules)
            {
                if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
                module = 0;
            }
        if (MotionGeometryRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, MotionGeometryRenderPass, nullptr);
        if (MotionCompositeRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, MotionCompositeRenderPass, nullptr);
        MotionGeometryRenderPass = MotionCompositeRenderPass = 0;
        if (MotionSampler && DestroySamplerFunction) DestroySamplerFunction(Device, MotionSampler, nullptr);
        MotionSampler = 0;
        MotionBlurActive = VelocityCaptured = MotionBlurResolved = false;
    }

    void ReadMotionVectors()
    {
        // Binary16 transfer preserves signed velocities and fractional depth.
        const auto decode = [](std::uint16_t packed)
        {
            constexpr unsigned MantissaBits = 10, ExponentBias = 15;
            constexpr unsigned MantissaMask = 0x3ff, ExponentMask = 0x1f, SignMask = 0x8000;
            const unsigned exponent = (packed >> MantissaBits) & ExponentMask;
            const unsigned mantissa = packed & MantissaMask;
            const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), 1 - int(ExponentBias + MantissaBits)) :
                exponent == ExponentMask ? (mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) :
                std::ldexp(1.0f + static_cast<float>(mantissa) / float(1U << MantissaBits), int(exponent) - int(ExponentBias));
            return (packed & SignMask) ? -magnitude : magnitude;
        };
        const auto* values = static_cast<const std::uint16_t*>(MotionReadback.Mapped);
        PendingFrame.MotionVectors.resize(static_cast<std::size_t>(MainColor.Width) * MainColor.Height);
        for (std::size_t pixel = 0; pixel < PendingFrame.MotionVectors.size(); ++pixel)
            PendingFrame.MotionVectors[pixel] = DecodeMotionVector({decode(values[pixel * 4]), decode(values[pixel * 4 + 1]),
                decode(values[pixel * 4 + 2]), decode(values[pixel * 4 + 3])});
    }

    struct EnvironmentTargets
    {
        RawImage Cube, Depth, Scratch;
        RawBuffer Upload;
        std::array<VkImageView, EnvironmentFaceCount> FaceViews {};
        std::array<VkFramebuffer, EnvironmentFaceCount> FaceFramebuffers {};
        std::vector<VkImageView> ParentMipViews;
        VkFramebuffer ScratchFramebuffer = 0;
    } Environment;

    void DestroyEnvironmentTargets(EnvironmentTargets& targets)
    {
        if (DestroyFramebufferFunction)
        {
            for (auto framebuffer : targets.FaceFramebuffers)
                if (framebuffer) DestroyFramebufferFunction(Device, framebuffer, nullptr);
            if (targets.ScratchFramebuffer) DestroyFramebufferFunction(Device, targets.ScratchFramebuffer, nullptr);
        }
        if (DestroyImageViewFunction)
        {
            for (auto view : targets.FaceViews) if (view) DestroyImageViewFunction(Device, view, nullptr);
            for (auto view : targets.ParentMipViews) if (view) DestroyImageViewFunction(Device, view, nullptr);
        }
        DestroyRawBuffer(targets.Upload);
        DestroyRawImage(targets.Cube); DestroyRawImage(targets.Depth); DestroyRawImage(targets.Scratch);
        targets = {};
    }

    bool PrepareEnvironmentCapture(const EnvironmentCaptureFrame& frame)
    {
        EnvironmentActive = Techniques::UsesEnvironmentTexture(FrameReflections);
        EnvironmentReady = false; EnvironmentFacesWritten = 0; FrameEnvironment = {};
        if (!EnvironmentActive) return true;
        if (!ValidEnvironmentSource(frame) || (!frame.Source && frame.Resolution != FrameReflections.CaptureResolution) || frame.MipCount == 0 ||
            frame.MipCount > MaximumEnvironmentMipLevels)
            return Fail("Vulkan environment reflections need a matching capture frame before BeginFrame");
        std::uint32_t mipCount = 0;
        for (auto extent = frame.Resolution; extent; extent >>= 1) ++mipCount;
        if (frame.MipCount != mipCount) return Fail("Vulkan capture requires a complete mip chain");
        for (const auto& face : frame.Faces)
            for (float value : face.ViewProjection.Values)
                if (!std::isfinite(value)) return Fail("Vulkan capture projections must be finite");
        if (Environment.Cube.Width != frame.Resolution)
        {
            // BeginFrame has waited for the previous submission. Build a complete
            // replacement before releasing views referenced by older descriptors.
            EnvironmentTargets candidate;
            const auto discard = [&]() { DestroyEnvironmentTargets(candidate); return false; };
            constexpr VkFlags CubeUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            constexpr VkFlags ScratchUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (!CreateRawImage(frame.Resolution, frame.Resolution, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_ASPECT_COLOR_BIT, CubeUsage, candidate.Cube, frame.MipCount, 1,
                    static_cast<std::uint32_t>(EnvironmentFaceCount), true) ||
                !CreateRawImage(frame.Resolution, frame.Resolution, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, candidate.Depth) ||
                !CreateRawImage(frame.Resolution / 2, frame.Resolution / 2, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_ASPECT_COLOR_BIT, ScratchUsage, candidate.Scratch)) return discard();
            VkImageViewCreateInfo view {};
            view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = candidate.Cube.Image; view.format = candidate.Cube.Format;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (std::uint32_t face = 0; face < EnvironmentFaceCount; ++face)
            {
                view.subresourceRange.baseArrayLayer = face;
                if (!Check(CreateImageViewFunction(Device, &view, nullptr, &candidate.FaceViews[face]), "vkCreateImageView(cube face)")) return discard();
                const VkImageView attachments[] {candidate.FaceViews[face], candidate.Depth.View};
                if (!CreateFramebuffer(HdrColorRenderPasses[3], attachments, 2, frame.Resolution, frame.Resolution,
                    candidate.FaceFramebuffers[face])) return discard();
            }
            view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            view.subresourceRange.baseArrayLayer = 0;
            view.subresourceRange.layerCount = static_cast<std::uint32_t>(EnvironmentFaceCount);
            candidate.ParentMipViews.resize(frame.MipCount - 1);
            for (std::uint32_t mip = 0; mip < candidate.ParentMipViews.size(); ++mip)
            {
                view.subresourceRange.baseMipLevel = mip;
                if (!Check(CreateImageViewFunction(Device, &view, nullptr, &candidate.ParentMipViews[mip]), "vkCreateImageView(cube parent mip)")) return discard();
            }
            if (!CreateFramebuffer(HdrRenderPass, &candidate.Scratch.View, 1, candidate.Scratch.Width,
                candidate.Scratch.Height, candidate.ScratchFramebuffer)) return discard();
            DestroyEnvironmentTargets(Environment);
            Environment = std::move(candidate);
        }
        FrameEnvironment = frame;
        return true;
    }

    struct EnvironmentMapTargets
    {
        RawImage Image, Scratch;
        std::vector<VkImageView> ParentMipViews;
        VkFramebuffer BaseFramebuffer = 0, ScratchFramebuffer = 0;
    } EnvironmentMap;

    void DestroyEnvironmentMapTargets(EnvironmentMapTargets& targets)
    {
        if (DestroyFramebufferFunction)
            for (auto framebuffer : {targets.BaseFramebuffer, targets.ScratchFramebuffer})
                if (framebuffer) DestroyFramebufferFunction(Device, framebuffer, nullptr);
        if (DestroyImageViewFunction)
            for (auto view : targets.ParentMipViews)
                if (view) DestroyImageViewFunction(Device, view, nullptr);
        DestroyRawImage(targets.Image); DestroyRawImage(targets.Scratch);
        targets = {};
    }

    bool EnsureEnvironmentMapResources()
    {
        const auto layout = BuildEnvironmentMapLayout(FrameEnvironment.Resolution);
        if (!layout.MipCount || layout.MipCount > MaximumEnvironmentMapMipLevels || layout.Width > DeviceLimits.maxImageDimension2D)
            return Fail("Vulkan environment image exceeds the supported dimensions");
        if (EnvironmentMap.Image.Image && EnvironmentMap.Image.Width == layout.Width) return true;
        EnvironmentMapTargets candidate;
        const auto discard = [&]() { DestroyEnvironmentMapTargets(candidate); return false; };
        constexpr VkFlags ImageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        constexpr VkFlags ScratchUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (!CreateRawImage(layout.Width, layout.Height, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_ASPECT_COLOR_BIT, ImageUsage, candidate.Image, layout.MipCount) ||
            !CreateRawImage(layout.Width / 2, layout.Height / 2, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_ASPECT_COLOR_BIT, ScratchUsage, candidate.Scratch)) return discard();
        candidate.ParentMipViews.resize(layout.MipCount - 1);
        VkImageViewCreateInfo view {};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = candidate.Image.Image; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = candidate.Image.Format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (std::uint32_t mip = 0; mip < candidate.ParentMipViews.size(); ++mip)
        {
            view.subresourceRange.baseMipLevel = mip;
            if (!Check(CreateImageViewFunction(Device, &view, nullptr, &candidate.ParentMipViews[mip]),
                "vkCreateImageView(environment parent mip)")) return discard();
        }
        // The base attachment exposes only mip zero; the full view is for lookup.
        if (!CreateFramebuffer(HdrRenderPass, &candidate.ParentMipViews[0], 1, layout.Width, layout.Height, candidate.BaseFramebuffer) ||
            !CreateFramebuffer(HdrRenderPass, &candidate.Scratch.View, 1, candidate.Scratch.Width, candidate.Scratch.Height,
                candidate.ScratchFramebuffer)) return discard();
        DestroyEnvironmentMapTargets(EnvironmentMap);
        EnvironmentMap = std::move(candidate);
        return true;
    }

    bool UploadEnvironmentMapConstants(const EnvironmentMapReflectionConstants& constants, std::uint32_t& dynamicOffset)
    {
        const auto offset = AlignUp(UniformOffset, UniformAlignment);
        if (offset > UniformBufferSize || ReflectionConstantBytes > UniformBufferSize - offset)
            return Fail("Vulkan environment map constant arena is exhausted");
        auto* destination = static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset;
        std::memset(destination, 0, ReflectionConstantBytes);
        std::memcpy(destination, &constants, sizeof(constants));
        UniformOffset = offset + ReflectionConstantBytes;
        dynamicOffset = static_cast<std::uint32_t>(offset);
        return true;
    }

    bool ConvertEnvironmentMap()
    {
        auto& image = EnvironmentMap.Image;
        auto& scratch = EnvironmentMap.Scratch;
        TransitionImage(CommandBuffer, Environment.Cube, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const auto draw = [&](RawImage& destination, VkFramebuffer framebuffer, std::uint32_t width, std::uint32_t height,
            VkPipeline pipeline, std::uint32_t descriptor, VkImageView parent)
        {
            EnvironmentMapReflectionConstants constants {};
            constants.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
            std::uint32_t offset = 0;
            if (!UploadEnvironmentMapConstants(constants, offset)) return false;
            TransitionImage(CommandBuffer, destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            std::array<VkImageView, ReflectionImageCount> views; views.fill(DummyColor.View);
            UpdateReflectionDescriptor(descriptor, views, Environment.Cube.View, 0, 0, parent);
            VkClearValue clear {};
            VkRenderPassBeginInfo begin {};
            begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            begin.renderPass = HdrRenderPass; begin.framebuffer = framebuffer;
            begin.renderArea = {{0, 0}, {width, height}};
            begin.clearValueCount = 1; begin.pClearValues = &clear;
            CmdBeginRenderPass(CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport {0, 0, float(width), float(height), 0, 1};
            CmdSetViewport(CommandBuffer, 0, 1, &viewport); CmdSetScissor(CommandBuffer, 0, 1, &begin.renderArea);
            CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ReflectionPipelineLayout,
                0, 1, &ReflectionDescriptors[descriptor], 1, &offset);
            CmdDraw(CommandBuffer, 3, 1, 0, 0);
            CmdEndRenderPass(CommandBuffer);
            return true;
        };
        // Each conversion/reduction owns a descriptor and uniform slice. Updating
        // a set referenced by recorded commands would invalidate those commands.
        // Vulkan specification: vkUpdateDescriptorSets, vkCmdCopyImage.
        if (!draw(image, EnvironmentMap.BaseFramebuffer, image.Width, image.Height,
            EnvironmentConversionPipeline, MapMipDescriptorBase, 0)) return false;
        ++PendingFrame.Statistics.EnvironmentConversionPassCount;
        for (std::uint32_t mip = 1; mip < image.MipLevels; ++mip)
        {
            const auto width = std::max(1u, image.Width >> mip), height = std::max(1u, image.Height >> mip);
            TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (!draw(scratch, EnvironmentMap.ScratchFramebuffer, width, height, EnvironmentMipPipeline,
                MapMipDescriptorBase + mip, EnvironmentMap.ParentMipViews[mip - 1])) return false;
            // Separate source/destination images retain one layout for every mip,
            // so the frame cancellation journal restores speculative barriers.
            TransitionImage(CommandBuffer, scratch, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageCopy copy {};
            copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
            copy.extent = {width, height, 1};
            CmdCopyImage(CommandBuffer, scratch.Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image.Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            ++PendingFrame.Statistics.EnvironmentMipPassCount;
        }
        TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return true;
    }

    struct PlanarTargets
    {
        RawImage Image, Depth, Scratch;
        std::vector<VkImageView> MipViews;
        VkFramebuffer CaptureFramebuffer = 0, ScratchFramebuffer = 0;
    } Planar;

    void DestroyPlanarTargets(PlanarTargets& targets)
    {
        if (DestroyFramebufferFunction)
            for (auto framebuffer : {targets.CaptureFramebuffer, targets.ScratchFramebuffer})
                if (framebuffer) DestroyFramebufferFunction(Device, framebuffer, nullptr);
        if (DestroyImageViewFunction)
            for (auto view : targets.MipViews) if (view) DestroyImageViewFunction(Device, view, nullptr);
        DestroyRawImage(targets.Image); DestroyRawImage(targets.Depth); DestroyRawImage(targets.Scratch);
        targets = {};
    }

    bool PreparePlanarCapture(const PlanarCaptureFrame& frame)
    {
        PlanarActive = Techniques::UsesPlanarCapture(FrameReflections);
        PlanarWritten = PlanarReady = false; FramePlanar = {};
        if (!PlanarActive) return true;
        ReflectionDesc desc; desc.ViewProjection = frame.MainViewProjection;
        PlanarReflectionConstants constants;
        if (!BuildPlanarReflectionConstants(desc, frame, FrameReflections, FrameReflectionShading, CreateInfo.Width, CreateInfo.Height, constants))
            return Fail("Vulkan planar capture requires a finite camera and matching main viewport before BeginFrame");
        std::uint32_t expectedMips = 0;
        for (auto extent = std::max(frame.Width, frame.Height); extent; extent >>= 1) ++expectedMips;
        const auto divisor = FrameReflections.Planar.ResolutionDivisor;
        if (frame.MipCount != expectedMips || frame.MipCount > MaximumPlanarMipLevels ||
            frame.Width != (frame.MainWidth + divisor - 1) / divisor || frame.Height != (frame.MainHeight + divisor - 1) / divisor ||
            std::max(frame.Width, frame.Height) > DeviceLimits.maxImageDimension2D)
            return Fail("Vulkan planar capture extent or mip chain is inconsistent or exceeds the device limit");
        for (float value : frame.ClippedViewProjection.Values)
            if (!std::isfinite(value)) return Fail("Vulkan oblique capture projection must be finite");
        if (Planar.Image.Width != frame.Width || Planar.Image.Height != frame.Height)
        {
            PlanarTargets candidate;
            const auto discard = [&]() { DestroyPlanarTargets(candidate); return false; };
            constexpr VkFlags ImageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            constexpr VkFlags ScratchUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (!CreateRawImage(frame.Width, frame.Height, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_ASPECT_COLOR_BIT, ImageUsage, candidate.Image, frame.MipCount) ||
                !CreateRawImage(frame.Width, frame.Height, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, candidate.Depth) ||
                !CreateRawImage(std::max(1u, frame.Width >> 1), std::max(1u, frame.Height >> 1), VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_ASPECT_COLOR_BIT, ScratchUsage, candidate.Scratch)) return discard();
            candidate.MipViews.resize(frame.MipCount);
            VkImageViewCreateInfo view {};
            view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = candidate.Image.Image; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = candidate.Image.Format;
            view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (std::uint32_t mip = 0; mip < frame.MipCount; ++mip)
            {
                view.subresourceRange.baseMipLevel = mip;
                if (!Check(CreateImageViewFunction(Device, &view, nullptr, &candidate.MipViews[mip]), "vkCreateImageView(planar mip)")) return discard();
            }
            const VkImageView attachments[] {candidate.MipViews[0], candidate.Depth.View};
            if (!CreateFramebuffer(HdrColorRenderPasses[3], attachments, 2, frame.Width, frame.Height, candidate.CaptureFramebuffer) ||
                !CreateFramebuffer(HdrRenderPass, &candidate.Scratch.View, 1, candidate.Scratch.Width, candidate.Scratch.Height,
                    candidate.ScratchFramebuffer)) return discard();
            DestroyPlanarTargets(Planar); Planar = std::move(candidate);
        }
        FramePlanar = frame;
        return true;
    }

    bool UploadPlanarConstants(const PlanarReflectionConstants& constants, std::uint32_t& dynamicOffset)
    {
        const auto offset = AlignUp(UniformOffset, UniformAlignment);
        if (offset > UniformBufferSize || sizeof(constants) > UniformBufferSize - offset)
            return Fail("Vulkan planar constant arena is exhausted");
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
        UniformOffset = offset + sizeof(constants);
        dynamicOffset = static_cast<std::uint32_t>(offset);
        return true;
    }

    bool UploadCubeConstants(const CubemapReflectionConstants& constants, std::uint32_t& dynamicOffset)
    {
        const auto offset = AlignUp(UniformOffset, UniformAlignment);
        // The shared descriptor range fits the larger SSR constants as well.
        if (offset > UniformBufferSize || ReflectionConstantBytes > UniformBufferSize - offset)
            return Fail("Vulkan environment constant arena is exhausted");
        auto* destination = static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset;
        std::memset(destination, 0, ReflectionConstantBytes);
        std::memcpy(destination, &constants, sizeof(constants));
        UniformOffset = offset + ReflectionConstantBytes;
        dynamicOffset = static_cast<std::uint32_t>(offset);
        return true;
    }

    void DestroyReflectionTargets()
    {
        DestroyRayReflectionTargets(); DestroyHybridRenderingTargets();
        if (ReflectionFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, ReflectionFramebuffer, nullptr);
        ReflectionFramebuffer = 0;
        DestroyRawImage(ReflectionImage);
    }

    void DestroyReflectionState()
    {
        DestroyPathTracing();
        DestroyRayGI(); DestroyVoxelGI();
        DestroyRayReflections(); DestroyHybridRendering(); DestroySubsurface(); DestroyHairFur(); DestroyExposure(); DestroyParallax(); DestroyDisplacement(); DestroyTessellation(); DestroyVirtualGeometry(); DestroyNaniteFrameVisibility(); DestroyNaniteGeometry(); DestroyTSR(); DestroyVirtualShadowRequests(); DestroyVirtualShadowAtlases(); DestroyLumenProbes(); DestroyLumenRadianceTraces(); DestroyLumenLightingCaches(); DestroyLumenSurfaces(); DestroyLumenSurfaceLookup(); DestroyLumenDistanceScenes();
        DestroyReflectionTargets();
        DestroyEnvironmentTargets(Environment);
        DestroyEnvironmentMapTargets(EnvironmentMap);
        DestroyPlanarTargets(Planar);
        for (auto* pair : {&CubeReflectionPipelines, &EnvironmentMapPipelines, &PlanarPipelines})
            for (auto& pipeline : *pair)
            {
                if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
                pipeline = 0;
            }
        for (auto* pipeline : {&CubeMipPipeline, &EnvironmentConversionPipeline, &EnvironmentMipPipeline, &PlanarMipPipeline})
        {
            if (*pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, *pipeline, nullptr);
            *pipeline = 0;
        }
        for (auto* pair : {&CubeReflectionModules, &CubeMipModules, &EnvironmentMapModules, &EnvironmentConversionModules, &EnvironmentMipModules,
            &PlanarModules, &PlanarMipModules})
            for (auto& module : *pair)
            {
                if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
                module = 0;
            }
        FrameEnvironment = {}; EnvironmentFacesWritten = 0;
        EnvironmentActive = EnvironmentReady = false;
        FramePlanar = {}; PlanarActive = PlanarWritten = PlanarReady = false;
        for (auto& pipeline : ReflectionPipelines)
        {
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& module : ReflectionModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        for (auto& sampler : ReflectionSamplers)
        {
            if (sampler && DestroySamplerFunction) DestroySamplerFunction(Device, sampler, nullptr);
            sampler = 0;
        }
        if (ReflectionCompositeRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, ReflectionCompositeRenderPass, nullptr);
        if (ReflectionDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, ReflectionDescriptorPool, nullptr);
        if (ReflectionPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, ReflectionPipelineLayout, nullptr);
        if (ReflectionDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, ReflectionDescriptorLayout, nullptr);
        ReflectionCompositeRenderPass = ReflectionDescriptorPool = ReflectionPipelineLayout = ReflectionDescriptorLayout = 0;
        ReflectionDescriptors.fill(0);
        ReflectionActive = ReflectionResolved = false;
        FrameReflections = {};
    }

    bool EnsureReflectionResources()
    {
            // Sky lookup and authored cube upload also need the cube mip
            // pipeline when the selected surface-reflection method is None.
            if (FrameReflections.Skybox)
            {
                const auto mode=FrameReflections.Mode;
                FrameReflections.Mode=Techniques::ReflectionMode::Cubemap;
                const bool ready=EnsureReflectionResources(false);
                FrameReflections.Mode=mode;
                if (!ready) return false;
            }

        if (PlanarActive && !EnsureReflectionResources(true)) return false;
        return FrameReflections.Mode==Techniques::ReflectionMode::None || EnsureReflectionResources(false);
    }

    bool EnsureReflectionResources(bool planarPass)
    {
        if (Techniques::UsesLumenReflections(FrameReflections)) return true; // Native resources are sized after the bounded query.
            if (FrameReflections.Mode==Techniques::ReflectionMode::RayTraced)
            return Techniques::UsesHybridRendering(FrameRayRadianceSettings.Pipeline) ? EnsureHybridRenderingResources() : EnsureRayReflectionResources();
        if (!DeferredSupported || !HdrActive || !GBufferFramebuffer)
            return Fail("Vulkan SSR requires sampled depth/materials and blendable HDR");
        if (!ReflectionFramebuffer)
        {
            if (!CreateRawImage(MainColor.Width, MainColor.Height, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, ReflectionImage) ||
                !CreateFramebuffer(HdrRenderPass, &ReflectionImage.View, 1, MainColor.Width, MainColor.Height, ReflectionFramebuffer))
            {
                DestroyReflectionTargets();
                return false;
            }
        }
        for (std::size_t index = 0; index < ReflectionSamplers.size(); ++index)
        {
            if (ReflectionSamplers[index]) continue;
            VkSamplerCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.magFilter = create.minFilter = index == 0 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            create.mipmapMode = index == 0 ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
            // A planar view follows the viewport rather than the cube GUI limit.
            // The clamp sampler must reach its complete chain even beyond 1024px.
            create.maxLod = static_cast<float>((index == 2 ? MaximumEnvironmentMapMipLevels : MaximumPlanarMipLevels) - 1);
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (index == 2) create.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &ReflectionSamplers[index]), "vkCreateSampler(SSR)")) return false;
        }
        if (!ReflectionDescriptorLayout)
        {
            std::array<VkDescriptorSetLayoutBinding, EnvironmentParentBinding + 1> bindings {};
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (std::uint32_t index = 0; index < ReflectionImageCount; ++index)
                bindings[index + 1] = {index + 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (auto binding : ReflectionSamplerBindings)
                bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            for (auto binding : {CubeImageBinding, ParentMipBinding, EnvironmentImageBinding, EnvironmentParentBinding})
                bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount = static_cast<std::uint32_t>(bindings.size()); create.pBindings = bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device, &create, nullptr, &ReflectionDescriptorLayout), "vkCreateDescriptorSetLayout(SSR)")) return false;
        }
        if (!ReflectionPipelineLayout)
        {
            VkPipelineLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            create.setLayoutCount = 1; create.pSetLayouts = &ReflectionDescriptorLayout;
            if (!Check(CreatePipelineLayoutFunction(Device, &create, nullptr, &ReflectionPipelineLayout), "vkCreatePipelineLayout(SSR)")) return false;
        }
        if (!ReflectionDescriptorPool)
        {
            constexpr std::array<VkDescriptorPoolSize, 3> sizes {{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, ReflectionDescriptorCount},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, (ReflectionImageCount + 4) * ReflectionDescriptorCount},
                {VK_DESCRIPTOR_TYPE_SAMPLER, static_cast<std::uint32_t>(ReflectionSamplerBindings.size()) * ReflectionDescriptorCount}}};
            VkDescriptorPoolCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            create.maxSets = ReflectionDescriptorCount;
            create.poolSizeCount = static_cast<std::uint32_t>(sizes.size()); create.pPoolSizes = sizes.data();
            if (!Check(CreateDescriptorPoolFunction(Device, &create, nullptr, &ReflectionDescriptorPool), "vkCreateDescriptorPool(SSR)")) return false;
        }
        if (!ReflectionDescriptors[0])
        {
            std::array<VkDescriptorSetLayout, ReflectionDescriptorCount> layouts; layouts.fill(ReflectionDescriptorLayout);
            VkDescriptorSetAllocateInfo allocate {};
            allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocate.descriptorPool = ReflectionDescriptorPool;
            allocate.descriptorSetCount = ReflectionDescriptorCount; allocate.pSetLayouts = layouts.data();
            if (!Check(AllocateDescriptorSets(Device, &allocate, ReflectionDescriptors.data()), "vkAllocateDescriptorSets(SSR)")) return false;
        }
        if (!CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT, ReflectionCompositeRenderPass, true)) return false;
        if (planarPass)
            return CreateFullscreenPipeline(VulkanPlanarShaders, PlanarModules, HdrRenderPass, PlanarPipelines[0], ReflectionPipelineLayout) &&
                CreateFullscreenPipeline(VulkanPlanarShaders, PlanarModules, ReflectionCompositeRenderPass,
                    PlanarPipelines[1], ReflectionPipelineLayout, FullscreenBlend::AddRadiance) &&
                CreateFullscreenPipeline(VulkanPlanarMipShaders, PlanarMipModules, HdrRenderPass, PlanarMipPipeline, ReflectionPipelineLayout);
        if (FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude)
            return EnsureEnvironmentMapResources() &&
                CreateFullscreenPipeline(VulkanEnvironmentMapShaders, EnvironmentMapModules, HdrRenderPass,
                    EnvironmentMapPipelines[0], ReflectionPipelineLayout) &&
                CreateFullscreenPipeline(VulkanEnvironmentMapShaders, EnvironmentMapModules, ReflectionCompositeRenderPass,
                    EnvironmentMapPipelines[1], ReflectionPipelineLayout, FullscreenBlend::AddRadiance) &&
                CreateFullscreenPipeline(VulkanEnvironmentMapConversionShaders, EnvironmentConversionModules, HdrRenderPass,
                    EnvironmentConversionPipeline, ReflectionPipelineLayout) &&
                CreateFullscreenPipeline(VulkanEnvironmentMapMipShaders, EnvironmentMipModules, HdrRenderPass,
                    EnvironmentMipPipeline, ReflectionPipelineLayout);
        if (FrameReflections.Mode==Techniques::ReflectionMode::Cubemap)
            return CreateFullscreenPipeline(VulkanCubemapShaders, CubeReflectionModules, HdrRenderPass, CubeReflectionPipelines[0], ReflectionPipelineLayout) &&
                CreateFullscreenPipeline(VulkanCubemapShaders, CubeReflectionModules, ReflectionCompositeRenderPass,
                    CubeReflectionPipelines[1], ReflectionPipelineLayout, FullscreenBlend::AddRadiance) &&
                CreateFullscreenPipeline(VulkanCubemapMipShaders, CubeMipModules, HdrRenderPass, CubeMipPipeline, ReflectionPipelineLayout);
        return CreateFullscreenPipeline(VulkanSsrShaders, ReflectionModules, HdrRenderPass, ReflectionPipelines[0], ReflectionPipelineLayout) &&
            CreateFullscreenPipeline(VulkanSsrShaders, ReflectionModules, ReflectionCompositeRenderPass,
                ReflectionPipelines[1], ReflectionPipelineLayout, FullscreenBlend::AddRadiance);
    }

    void UpdateReflectionDescriptor(std::uint32_t phase, const std::array<VkImageView, ReflectionImageCount>& views,
        VkImageView cube = 0, VkImageView parentMip = 0, VkImageView environmentImage = 0, VkImageView environmentParent = 0)
    {
        const VkDescriptorBufferInfo uniform {UniformBuffer.Buffer, 0, ReflectionConstantBytes};
        std::array<VkDescriptorImageInfo, EnvironmentParentBinding> images {};
        for (std::size_t index = 0; index < views.size(); ++index)
            images[index] = {0, views[index], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        for (std::size_t index = 0; index < ReflectionSamplers.size(); ++index)
            images[ReflectionSamplerBindings[index] - 1].sampler = ReflectionSamplers[index];
        images[CubeImageBinding - 1] = {0, cube ? cube : DummyColor.View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[ParentMipBinding - 1] = {0, parentMip ? parentMip : DummyColor.View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[EnvironmentImageBinding - 1] = {0, environmentImage ? environmentImage : DummyColor.View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        images[EnvironmentParentBinding - 1] = {0, environmentParent ? environmentParent : DummyColor.View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        std::array<VkWriteDescriptorSet, EnvironmentParentBinding + 1> writes {};
        for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
        {
            auto& write = writes[binding];
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = ReflectionDescriptors[phase]; write.dstBinding = binding; write.descriptorCount = 1;
            write.descriptorType = binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC :
                std::find(ReflectionSamplerBindings.begin(), ReflectionSamplerBindings.end(), binding) != ReflectionSamplerBindings.end()
                    ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            if (binding == 0) write.pBufferInfo = &uniform;
            else write.pImageInfo = &images[binding - 1];
        }
        UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    bool EnsureDepthOfFieldResources()
    {
        if (!HdrActive || !HdrSceneFramebuffer)
            return Fail("Vulkan depth of field requires the main HDR scene");
        VkFormatProperties depth {};
        GetPhysicalDeviceFormatProperties(PhysicalDevice, VK_FORMAT_D32_SFLOAT, &depth);
        if ((depth.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0)
            return Fail("Vulkan depth of field requires sampled D32 depth");
        if (!DepthOfFieldSampler)
        {
            // Keep the effect independent of MRT resources and D32 filtering.
            VkSamplerCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.magFilter = create.minFilter = VK_FILTER_NEAREST;
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &DepthOfFieldSampler),
                    "vkCreateSampler(depth of field)")) return false;
        }
        return CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT, DepthOfFieldCompositeRenderPass, true) &&
            CreateFullscreenPipeline(VulkanDepthOfFieldShaders, DepthOfFieldModules,
                HdrRenderPass, DepthOfFieldPipelines[0]) &&
            CreateFullscreenPipeline(VulkanDepthOfFieldShaders, DepthOfFieldModules,
                DepthOfFieldCompositeRenderPass, DepthOfFieldPipelines[1], PipelineLayout, FullscreenBlend::Replace, true);
    }

    void DestroyDepthOfFieldState()
    {
        for (auto& pipeline : DepthOfFieldPipelines)
        {
            if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& module : DepthOfFieldModules)
        {
            if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr);
            module = 0;
        }
        if (DepthOfFieldCompositeRenderPass && DestroyRenderPassFunction)
            DestroyRenderPassFunction(Device, DepthOfFieldCompositeRenderPass, nullptr);
        DepthOfFieldCompositeRenderPass = 0;
        if (DepthOfFieldSampler && DestroySamplerFunction)
            DestroySamplerFunction(Device, DepthOfFieldSampler, nullptr);
        DepthOfFieldSampler = 0;
        DepthOfFieldActive = DepthOfFieldResolved = false;
        FramePostProcessing = {};
    }

    bool ApplyHdr(const Techniques::HdrSettings& settings)
    {
        const auto plan = Techniques::BuildHdrPlan(settings);
        if (plan.empty()) return true;
        RawImage* scene = &HdrImages[0];
        using Pass = Techniques::HdrPassKind;
        for (const auto pass : plan)
        {
            const auto index = HdrShaderIndex(pass);
            const VkRenderPass renderPass = pass == Pass::ToneMapping ? PostRenderPass : HdrRenderPass;
            const VkPipeline pipeline = CreateFullscreenPipeline(VulkanHdrShaders[index], HdrModules[index], renderPass, HdrPipelines[index]);
            if (!pipeline) return false;
            RawImage* source = scene;
            std::size_t destinationIndex = scene == &HdrImages[0] ? 1U : 0U;
            RawImage* destination = &HdrImages[destinationIndex];
            VkFramebuffer framebuffer = HdrFramebuffers[destinationIndex];
            const auto resources = DescribeHdrPass(pass);
            constexpr std::size_t FirstHalfTarget = 2;
            if (resources.DestinationHalf >= 0)
            {
                destinationIndex = FirstHalfTarget + resources.DestinationHalf;
                destination = &HdrImages[destinationIndex];
                framebuffer = HdrFramebuffers[destinationIndex];
                if (resources.SourceHalf >= 0) source = &HdrImages[FirstHalfTarget + resources.SourceHalf];
            }
            else if (pass == Pass::ToneMapping)
            {
                destination = &MainColor;
                framebuffer = HdrResolveFramebuffer;
            }
            const HdrConstants constants = BuildHdrConstants(settings, static_cast<int>(source->Width), static_cast<int>(source->Height), pass,
                CreateInfo.Width, CreateInfo.Height);
            const std::size_t offset = AlignUp(UniformOffset, UniformAlignment);
            if (offset > UniformBufferSize || sizeof(RealtimeDrawConstants) > UniformBufferSize - offset ||
                DescriptorSetIndex >= DescriptorSets.size()) return Fail("Vulkan HDR per-frame arena is exhausted");
            std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
            UniformOffset = offset + sizeof(RealtimeDrawConstants);
            const VkDescriptorSet descriptor = DescriptorSets[DescriptorSetIndex++];
            RawImage* bloom = resources.CompositeHalf >= 0 ? &HdrImages[FirstHalfTarget + resources.CompositeHalf] : source;
            TransitionImage(CommandBuffer, *source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            TransitionImage(CommandBuffer, *bloom, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            TransitionImage(CommandBuffer, *destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            UpdateDescriptorSet(descriptor, {DummyDepth.View, source->View, bloom->View, DummyColor.View});
            VkClearValue clear{};
            VkRenderPassBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            begin.renderPass = renderPass;
            begin.framebuffer = framebuffer;
            begin.renderArea = {{0, 0}, {destination->Width, destination->Height}};
            begin.clearValueCount = 1;
            begin.pClearValues = &clear;
            CmdBeginRenderPass(CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport{0.0f, 0.0f, static_cast<float>(destination->Width), static_cast<float>(destination->Height), 0.0f, 1.0f};
            const VkRect2D scissor{{0, 0}, {destination->Width, destination->Height}};
            CmdSetViewport(CommandBuffer, 0, 1, &viewport);
            CmdSetScissor(CommandBuffer, 0, 1, &scissor);
            CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(offset);
            CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &descriptor, 1, &dynamicOffset);
            CmdDraw(CommandBuffer, 3, 1, 0, 0);
            CmdEndRenderPass(CommandBuffer);
            if (resources.UpdatesScene) scene = destination;
            PendingFrame.HDRPasses.push_back(pass);
        }
        PendingFrame.Statistics.PostProcessPassCount += static_cast<std::uint32_t>(plan.size());
        return true;
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
                            usage, resource.Image, resource.Description.MipLevels)) return false;
        if (initialData != nullptr || !resource.Description.RenderTarget)
        {
            const auto& desc = resource.Description;
            if (!InitializeImage(resource.Image, initialData, TextureMipBytes(desc), desc.MipLevels)) return false;
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

    bool UsesVSM() const { return FrameShadowFilter.Mode == Techniques::ShadowFilterMode::VSM; }
    bool UsesSurfaceVSM(const PipelineState& pipeline, PassType pass) const
    { return UsesVSM() && pass == PassType::Color && pipeline.SamplesShadowMap; }

    bool MomentImage(TextureHandle handle, RawImage*& image)
    {
        const auto found = VSMMoments.find(handle.Value);
        if (found == VSMMoments.end() || !found->second.Ready)
            return Fail("Vulkan VSM requires filtered moments from this frame");
        image = &found->second.Images[VSMFiltered];
        return true;
    }

    bool BuildShadowFilterData(const Matrix4& projection, const LightingState& lighting, bool enabled, ShadowFilterBytes& bytes)
    {
        if (FrameShadowFilter.Mode==Techniques::ShadowFilterMode::PCF)
        {
            const auto constants=BuildPCFConstants(FrameShadowFilter,enabled);
            bytes={}; std::memcpy(bytes.data(),&constants,sizeof(constants)); return true;
        }
        if (UsesVSM())
        {
            if (enabled)
            {
                RawImage* image = nullptr;
                if (!FrameCascades.Frame.CascadeCount) { if (!MomentImage(LastShadowTexture, image)) return false; }
                else for (std::uint32_t index = 0; index < FrameCascades.Frame.CascadeCount; ++index)
                    if (!MomentImage(FrameCascades.Maps[index], image)) return false;
            }
            VSMConstants constants;
            if (!BuildVSMConstants(projection, lighting, FrameCascades, FrameShadowFilter.VSM, enabled, constants, LastError)) return false;
            std::memcpy(bytes.data(), &constants, sizeof(constants));
        }
        else
        {
            PCSSConstants constants;
            if (!BuildPCSSConstants(projection, lighting, FrameCascades, FrameShadowFilter.PCSS, enabled, constants, LastError)) return false;
            std::memcpy(bytes.data(), &constants, sizeof(constants));
        }
        return true;
    }

    bool EnsureVSMModule(const VulkanShaderProgram& code, VkShaderModule& module)
    {
        if (module) return true;
        std::vector<std::uint32_t> compiledWords;
        if (!CompileVulkanShader(code, compiledWords, LastError)) return false;
        VkShaderModuleCreateInfo create {};
        create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create.codeSize = compiledWords.size() * sizeof(std::uint32_t); create.pCode = compiledWords.data();
        return Check(CreateShaderModuleFunction(Device, &create, nullptr, &module), "vkCreateShaderModule(VSM)");
    }

    bool EnsureVSMState()
    {
        if (!VSMSupported) return Fail("Vulkan VSM requires sampled RG32F color/transfer targets");
        for (std::size_t clear = 0; clear < VSMCasterPasses.size(); ++clear)
            if (!VSMCasterPasses[clear] && !CreateColorRenderPass(clear != 0, clear != 0, VSMCasterPasses[clear], VSMFormat)) return false;
        if (!VSMFilterPass && !CreateFullscreenRenderPass(VSMFormat, VSMFilterPass)) return false;
        if (!VSMFilterDescriptorLayout)
        {
            const std::array<VkDescriptorSetLayoutBinding, 2> bindings {{{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
            VkDescriptorSetLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            create.bindingCount = static_cast<std::uint32_t>(bindings.size()); create.pBindings = bindings.data();
            if (!Check(CreateDescriptorSetLayoutFunction(Device, &create, nullptr, &VSMFilterDescriptorLayout), "vkCreateDescriptorSetLayout(VSM filter)")) return false;
        }
        if (!VSMFilterLayout)
        {
            VkPipelineLayoutCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            create.setLayoutCount = 1; create.pSetLayouts = &VSMFilterDescriptorLayout;
            if (!Check(CreatePipelineLayoutFunction(Device, &create, nullptr, &VSMFilterLayout), "vkCreatePipelineLayout(VSM filter)")) return false;
        }
        if (!VSMSampler)
        {
            VkSamplerCreateInfo create {};
            create.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            create.minFilter = create.magFilter = VK_FILTER_NEAREST;
            create.addressModeU = create.addressModeV = create.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!Check(CreateSamplerFunction(Device, &create, nullptr, &VSMSampler), "vkCreateSampler(VSM)")) return false;
        }
        return CreateFullscreenPipeline(VulkanVSMFilterShaders, VSMFilterModules, VSMFilterPass, VSMFilterPipeline, VSMFilterLayout) != 0;
    }

    void DestroyMomentMap(VSMMomentMap& map)
    {
        if (map.DescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, map.DescriptorPool, nullptr);
        for (auto framebuffer : map.Framebuffers) if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, framebuffer, nullptr);
        for (auto& image : map.Images) DestroyRawImage(image);
        DestroyRawBuffer(map.Readback);
        map = {};
    }

    bool EnsureMomentMap(TextureHandle handle)
    {
        if (!EnsureVSMState()) return false;
        const auto depth = Textures.find(handle.Value);
        if (depth == Textures.end() || depth->second.Description.Format != TextureFormat::Depth32)
            return Fail("Vulkan moment producer requires a depth texture");
        auto& map = VSMMoments[handle.Value];
        if (map.Images[VSMRaw].Image) return true;
        const auto width = depth->second.Image.Width, height = depth->second.Image.Height;
        if (width > VSMImageLimits.maxExtent.width || height > VSMImageLimits.maxExtent.height)
            return Fail("Vulkan moment map exceeds the RG32F format extent");
        constexpr VkFlags Usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        for (std::size_t index = 0; index < map.Images.size(); ++index)
        {
            if (!CreateRawImage(width, height, VSMFormat, VK_IMAGE_ASPECT_COLOR_BIT, Usage, map.Images[index]))
            { DestroyMomentMap(map); return false; }
            const std::array<VkImageView, 2> attachments {map.Images[index].View, depth->second.Image.View};
            if (!CreateFramebuffer(index == VSMRaw ? VSMCasterPasses[0] : VSMFilterPass, attachments.data(),
                index == VSMRaw ? 2u : 1u, width, height, map.Framebuffers[index]))
            { DestroyMomentMap(map); return false; }
        }
        const std::array<VkDescriptorPoolSize, 2> sizes {{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}}};
        VkDescriptorPoolCreateInfo pool {};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 2; pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size()); pool.pPoolSizes = sizes.data();
        if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &map.DescriptorPool), "vkCreateDescriptorPool(VSM map)"))
        { DestroyMomentMap(map); return false; }
        const std::array<VkDescriptorSetLayout, 2> layouts {VSMFilterDescriptorLayout, VSMFilterDescriptorLayout};
        VkDescriptorSetAllocateInfo allocate {};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = map.DescriptorPool; allocate.descriptorSetCount = 2; allocate.pSetLayouts = layouts.data();
        if (!Check(AllocateDescriptorSets(Device, &allocate, map.FilterSets.data()), "vkAllocateDescriptorSets(VSM map)"))
        { DestroyMomentMap(map); return false; }
        // Image identities never change while commands reference these sets.
        // Each filter draw varies only its immutable dynamic uniform offset.
        const VkDescriptorBufferInfo uniform {UniformBuffer.Buffer, 0, sizeof(VSMConstants)};
        for (std::size_t axis = 0; axis < map.FilterSets.size(); ++axis)
        {
            const VkDescriptorImageInfo image {VSMSampler, map.Images[axis].View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 2> writes {};
            for (std::uint32_t binding = 0; binding < writes.size(); ++binding)
            {
                auto& write = writes[binding]; write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = map.FilterSets[axis]; write.dstBinding = binding; write.descriptorCount = 1;
                write.descriptorType = binding ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            }
            writes[0].pBufferInfo = &uniform; writes[1].pImageInfo = &image;
            UpdateDescriptorSets(Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
        return true;
    }

    bool FilterMoments(TextureHandle handle)
    {
        const auto found = VSMMoments.find(handle.Value);
        if (found == VSMMoments.end()) return Fail("Vulkan VSM caster map is missing");
        auto& map = found->second;
        auto constants = BuildVSMParameters(FrameShadowFilter.VSM);
        for (std::size_t axis = 0; axis < map.FilterSets.size(); ++axis)
        {
            const auto offset = AlignUp(UniformOffset, UniformAlignment);
            if (offset > UniformBufferSize || sizeof(constants) > UniformBufferSize - offset)
                return Fail("Vulkan VSM filter exhausted the uniform arena");
            constants.Control.W = static_cast<std::uint32_t>(axis);
            std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
            UniformOffset = offset + sizeof(constants);
            auto& destination = map.Images[axis + 1];
            TransitionImage(CommandBuffer, map.Images[axis], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            TransitionImage(CommandBuffer, destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            VkClearValue clear {}; clear.color.float32[0] = clear.color.float32[1] = 1;
            VkRenderPassBeginInfo begin {};
            begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            begin.renderPass = VSMFilterPass; begin.framebuffer = map.Framebuffers[axis + 1];
            begin.renderArea = {{0, 0}, {destination.Width, destination.Height}};
            begin.clearValueCount = 1; begin.pClearValues = &clear;
            CmdBeginRenderPass(CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport {0, 0, float(destination.Width), float(destination.Height), 0, 1};
            CmdSetViewport(CommandBuffer, 0, 1, &viewport); CmdSetScissor(CommandBuffer, 0, 1, &begin.renderArea);
            CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VSMFilterPipeline);
            const auto dynamicOffset = static_cast<std::uint32_t>(offset);
            CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VSMFilterLayout, 0, 1, &map.FilterSets[axis], 1, &dynamicOffset);
            constexpr std::uint32_t TriangleVertices = 3; CmdDraw(CommandBuffer, TriangleVertices, 1, 0, 0);
            CmdEndRenderPass(CommandBuffer);
        }
        TransitionImage(CommandBuffer, map.Images[VSMFiltered], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        map.Ready = map.RawWritten = true;
        PendingFrame.Statistics.ShadowMomentFilterPassCount += 2;
        return true;
    }

    void DestroyVSMState()
    {
        for (auto& entry : VSMMoments) DestroyMomentMap(entry.second);
        VSMMoments.clear();
        if (VSMFilterPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, VSMFilterPipeline, nullptr);
        VSMFilterPipeline = 0;
        const auto destroyModules = [&](auto& modules)
        {
            for (auto& module : modules)
            { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        };
        destroyModules(VSMSurfaceModules); destroyModules(VSMCasterModules); destroyModules(VSMFilterModules);
        if (VSMFilterLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, VSMFilterLayout, nullptr);
        if (VSMFilterDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, VSMFilterDescriptorLayout, nullptr);
        if (VSMSampler && DestroySamplerFunction) DestroySamplerFunction(Device, VSMSampler, nullptr);
        for (auto& pass : VSMCasterPasses) { if (pass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, pass, nullptr); pass = 0; }
        if (VSMFilterPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device, VSMFilterPass, nullptr);
        VSMFilterLayout = VSMFilterDescriptorLayout = VSMSampler = VSMFilterPass = 0;
    }

    bool UsesSurfacePCSS(const PipelineState& pipeline, PassType pass) const
    {
        return pass == PassType::Color && pipeline.SamplesShadowMap &&
            FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS;
    }

    bool EnsurePCSSSurface(Techniques::ShadingModel model, bool weighted)
    {
        const auto index = static_cast<std::size_t>(model) + (weighted ? 8u : 0u);
        if (PCSSSurfaceModules[index]) return true;
        const auto& code = VulkanPCSSFragments[index];
        std::vector<std::uint32_t> compiledWords;
        if (!CompileVulkanShader(code, compiledWords, LastError)) return false;
        VkShaderModuleCreateInfo create {};
        create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create.codeSize = compiledWords.size() * sizeof(std::uint32_t); create.pCode = compiledWords.data();
        return Check(CreateShaderModuleFunction(Device, &create, nullptr, &PCSSSurfaceModules[index]),
            "vkCreateShaderModule(PCSS surface)");
    }

    bool UsesForwardPlusMaterial(PassType pass) const
    {
        return LightListsActive && (pass == PassType::Color || pass == PassType::EnvironmentCapture || pass == PassType::PlanarCapture);
    }

    std::size_t ForwardPlusFilter(const PipelineState& pipeline, PassType pass) const
    {
        return pass == PassType::Color && pipeline.SamplesShadowMap ? static_cast<std::size_t>(FrameShadowFilter.Mode) :
            static_cast<std::size_t>(Techniques::ShadowFilterMode::Hard);
    }

    ForwardPlusProgram ForwardPlusVariant(const PipelineState& pipeline, PassType pass) const
    {
        return pass == PassType::ForwardPlusDepth ? ForwardPlusDepth : WeightedPass ? ForwardPlusWeighted :
            UsesContactReceiver(pipeline, pass) ? (FrameSampleCount > 1 ? ForwardPlusContactMS : ForwardPlusContact) : ForwardPlusSurface;
    }

    bool EnsureForwardPlusModule(const VulkanShaderProgram& code, VkShaderModule& module)
    {
        if (module) return true;
        std::vector<std::uint32_t> compiledWords;
        if (!CompileVulkanShader(code, compiledWords, LastError)) return false;
        VkShaderModuleCreateInfo create {};
        create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; create.codeSize = compiledWords.size() * sizeof(std::uint32_t); create.pCode = compiledWords.data();
        return Check(CreateShaderModuleFunction(Device, &create, nullptr, &module), "vkCreateShaderModule(Forward+)");
    }

    bool ResetForwardPlusAfterDiscard()
    {
        // The caller waited for FrameFence and reset the command pool. Release
        // the borrowed completion fence before an immediate upload can reset it.
        if (ForwardPlusSubmitted)
        {
            if (!(ClusteredActive ? ClusteredProducer->BeginFrame(LastError) : TileProducer()->BeginFrame(LastError))) return false;
            ForwardPlusSubmitted = false; ForwardPlusRecordOpen = true;
        }
        if (ForwardPlusRecordOpen && !(ClusteredActive ? ClusteredProducer->AbortFrame(LastError) : TileProducer()->AbortFrame(LastError))) return false;
        ForwardPlusRecordOpen = ForwardPlusRecorded = false;
        return true;
    }

    bool UploadForwardPlusScene()
    {
        if (!ForwardPlusSceneStorage.Buffer && !CreateRawBuffer(MaximumForwardPlusLights * sizeof(ShaderLight),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ForwardPlusSceneStorage)) return false;
        const auto& lights = LightListFrame.Description().Lights->GetShaderLights();
        // Host-coherent writes precede queue submission. All consumers retain
        // these bytes until the same frame fence completes.
        if (lights.empty()) std::memset(ForwardPlusSceneStorage.Mapped, 0, sizeof(ShaderLight));
        else std::memcpy(ForwardPlusSceneStorage.Mapped, lights.data(), lights.size() * sizeof(ShaderLight));
        return true;
    }

    void BindForwardPlusStorage(VkDescriptorSet set, bool mainCamera, std::uint32_t firstBinding = ForwardPlusStorageFirstBinding)
    {
        const VkDescriptorBufferInfo empty {ForwardPlusEmptyStorage.Buffer, 0, sizeof(ShaderLight)};
        const auto count = LightListFrame.Description().Lights->GetShaderLights().size();
        const std::array<VkDescriptorBufferInfo, ForwardPlusStorageCount> buffers {{
            {ForwardPlusSceneStorage.Buffer, 0, std::max(count, std::size_t{1}) * sizeof(ShaderLight)},
            mainCamera ? (ClusteredActive ? ClusteredProducer->Headers() : TileProducer()->Headers()) : empty,
            mainCamera ? (ClusteredActive ? ClusteredProducer->Indices() : TileProducer()->Indices()) : empty}};
        std::array<VkWriteDescriptorSet, ForwardPlusStorageCount> writes {};
        for (std::uint32_t slot = 0; slot < ForwardPlusStorageCount; ++slot)
        {
            auto& write = writes[slot]; write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set; write.dstBinding = firstBinding + slot; write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &buffers[slot];
        }
        // The draw owns this set; captures never reference the main-camera list
        // buffers and no descriptor used by an earlier recorded draw is mutated.
        UpdateDescriptorSets(Device, ForwardPlusStorageCount, writes.data(), 0, nullptr);
    }

    void DestroyForwardPlusState()
    {
        ForwardPlusProducer.reset(); TiledProducer.reset(); ClusteredProducer.reset(); LightListFrame.Reset();
        DestroyRawBuffer(ForwardPlusSceneStorage); DestroyRawBuffer(ForwardPlusEmptyStorage);
        if (ForwardPlusVertexModule) DestroyShaderModuleFunction(Device, ForwardPlusVertexModule, nullptr);
        ForwardPlusVertexModule = 0;
        if (ClusteredVertexModule) DestroyShaderModuleFunction(Device, ClusteredVertexModule, nullptr);
        ClusteredVertexModule = 0;
        if (TiledVertexModule) DestroyShaderModuleFunction(Device, TiledVertexModule, nullptr);
        TiledVertexModule = 0;
        for (auto& family : TiledFragments) for (auto& module : family)
        { if (module) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        for (auto& family : ClusteredFragments) for (auto& module : family)
        { if (module) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        for (auto& family : ForwardPlusFragments) for (auto& module : family)
        { if (module) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        ForwardPlusRecordOpen = ForwardPlusRecorded = ForwardPlusSubmitted = false;
        LightListsActive = ForwardPlusDepthWritten = ForwardPlusSupported = false;
    }

    // Depth production and receivers share the existing material layouts. Each
    // forward draw already owns immutable descriptors; adding one image there
    // keeps the four-set Vulkan 1.0 contract and avoids updating recorded sets.
    bool UsesContactReceiver(const PipelineState& pipeline, PassType pass) const
    {
        return ContactReceiverActive && pass == PassType::Color && !CurrentPass.Target &&
            !pipeline.BlendEnabled && !pipeline.OutlineExtrusion && pipeline.ColorWrite;
    }

    std::size_t ContactFilterFamily() const
    {
        return UsesVSM() ? 2 : FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS ? 1 : 0;
    }

    bool EnsureContactModule(const VulkanShaderProgram& code, VkShaderModule& module)
    {
        if (module) return true;
        std::vector<std::uint32_t> compiledWords;
        if (!CompileVulkanShader(code, compiledWords, LastError)) return false;
        VkShaderModuleCreateInfo create {};
        create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create.codeSize = compiledWords.size() * sizeof(std::uint32_t); create.pCode = compiledWords.data();
        return Check(CreateShaderModuleFunction(Device, &create, nullptr, &module), "vkCreateShaderModule(contact)");
    }

    void BindContactImage(VkDescriptorSet set, std::uint32_t binding)
    {
        const VkDescriptorImageInfo image {0, ContactReceiverActive ? ContactProducer->ResultView() : DummyColor.View,
            ContactReceiverActive ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set; write.dstBinding = binding; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; write.pImageInfo = &image;
        UpdateDescriptorSets(Device, 1, &write, 0, nullptr);
    }

    void DestroyContactTargets()
    {
        if (ContactDepthFramebuffer) DestroyFramebufferFunction(Device, ContactDepthFramebuffer, nullptr);
        ContactDepthFramebuffer = 0;
        if (ContactDepthPass) DestroyRenderPassFunction(Device, ContactDepthPass, nullptr);
        ContactDepthPass = 0;
        for (auto& pass : ContactColorPasses)
        { if (pass) DestroyRenderPassFunction(Device, pass, nullptr); pass = 0; }
    }

    bool EnsureCameraVisibilityTargets()
    {
        if (FrameContact.Enabled && !ContactSupported)
            return Fail("Vulkan contact shadows require compute, sampled storage images and standard sample locations");
        if (FrameContact.Enabled && !ContactProducer)
        {
            auto producer = std::make_unique<VulkanContactShadows>();
            if (!producer->Initialize(Device, MemoryProperties, GetDeviceProcAddress, LastError)) return false;
            ContactProducer = std::move(producer);
        }
        if (FrameContact.Enabled) ContactProducer->BeginFrame();
        auto& depth = FrameSampleCount > 1 ? Multisample.Depth : MainDepth;
        if (!CreateShadowRenderPass(true, ContactDepthPass, FrameSampleCount) ||
            !CreateFramebuffer(ContactDepthPass, &depth.View, 1, depth.Width, depth.Height, ContactDepthFramebuffer)) return false;
        const auto format = HdrActive ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_SRGB;
        for (std::size_t clear = 0; clear < ContactColorPasses.size(); ++clear)
            if (!CreateColorRenderPass(clear != 0, false, ContactColorPasses[clear], format, true, FrameSampleCount)) return false;
        return true;
    }

    void DestroyContactState()
    {
        DestroyContactTargets();
        ContactProducer.reset();
        for (auto& module : ContactDepthModules)
        { if (module) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        for (auto& family : ContactSurfaceModules) for (auto& samples : family) for (auto& module : samples)
        { if (module) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
        for (auto& family : ContactDeferred) for (auto& state : family)
        {
            if (state.Pipeline) DestroyPipelineFunction(Device, state.Pipeline, nullptr);
            for (auto module : state.Modules) if (module) DestroyShaderModuleFunction(Device, module, nullptr);
            state = {};
        }
        FrameContact = {}; ContactDepthWritten = ContactGenerated = ContactReceiverActive = false;
    }

    VkPipeline TiledLightingPipeline()
    {
        const auto filter = static_cast<std::size_t>(FrameShadowFilter.Mode);
        const auto sampleSlot = FrameSampleCount == 8 ? 3 : FrameSampleCount == 4 ? 2 : FrameSampleCount == 2 ? 1 : 0;
        auto& state = TiledLighting[filter][sampleSlot][ContactReceiverActive ? 1 : 0];
        const auto variant = (FrameSampleCount > 1 ? 1 : 0) | (ContactReceiverActive ? 2 : 0);
        return CreateFullscreenPipeline(VulkanTiledLightingShaders[filter][variant], state.Modules,
            FrameSampleCount > 1 ? Multisample.LightingPass : HdrRenderPass, state.Pipeline,
            DeferredPipelineLayout, FullscreenBlend::Replace, false, FrameSampleCount, FrameSampleCount > 1);
    }

    VkPipeline ContactDeferredPipeline()
    {
        const auto family = ContactFilterFamily();
        const auto sampleSlot = FrameSampleCount == 8 ? 3 : FrameSampleCount == 4 ? 2 : FrameSampleCount == 2 ? 1 : 0;
        auto& state = ContactDeferred[family][sampleSlot];
        const auto& source = VulkanContactDeferredShaders[family][FrameSampleCount > 1 ? 1 : 0];
        const VulkanShaderProgram codes[2] {source[0], source[1]};
        return CreateFullscreenPipeline(codes, state.Modules, FrameSampleCount > 1 ? Multisample.LightingPass : HdrRenderPass,
            state.Pipeline, DeferredPipelineLayout, FullscreenBlend::Replace, false, FrameSampleCount, FrameSampleCount > 1);
    }

    [[nodiscard]] std::uint64_t PipelineKey(
        const PipelineState& pipeline,
        PassType pass, bool instanced) const
    {
        constexpr unsigned ForwardPlusFilterKeyShift = 38;
        // PassType no longer fits the old two-bit field: Velocity collided with
        // wireframe Color. Reserve a disjoint high field for every pass value.
        return (static_cast<std::uint32_t>(pass) << 14U) |
               (pipeline.Fill == FillMode::Wireframe ? 1U << 2U : 0U) |
               (pipeline.DepthTest ? 1U << 3U : 0U) |
               (pipeline.DepthWrite ? 1U << 4U : 0U) |
               (pipeline.ColorWrite ? 1U << 5U : 0U) |
               (pipeline.OutlineExtrusion ? 1U << 6U : 0U) |
               (pipeline.BlendEnabled ? 1U << 7U : 0U) |
               (static_cast<std::uint32_t>(ShadingTechniqueIndex(pipeline.Shading, FrameAreaLights)) << 8U) |
               (pipeline.BackFaceCulling ? 1U << 11U : 0U) |
               (RealtimeWorldIsMirrored(BoundWorld) ? 1U << 12U : 0U) |
               (ActiveColor && ActiveColor->Format == VK_FORMAT_R16G16B16A16_SFLOAT ? 1U << 13U : 0U) |
               ((ActiveDepth ? ActiveDepth->Samples : 1U) << SampleCountPipelineShift) |
               (pass == PassType::Shadow && FrameCascades.Frame.CascadeCount ? CascadedDepthPipelineFlag : 0U) |
               (instanced ? std::uint64_t{1} << 32 : 0) |
               (WeightedPass ? std::uint64_t{1} << 33 : 0) |
               (CurrentPass.DepthPeelingLayer ? std::uint64_t{1} << 42 : 0) |
               (UsesParallaxMaterial(pipeline,pass) ? std::uint64_t{1} << 43 : 0) |
               (UsesSurfacePCF(pipeline,pass)?std::uint64_t{1}<<45:0) |
               (UsesMaterialStage(pipeline,pass)?std::uint64_t(unsigned(SelectMaterialTechnique(pipeline,WeightedPass))+1)<<MaterialPipelineKeyShift:0) |
               (FrameRayShadowImage && !FrameAreaLights && pass==PassType::Color && pipeline.SamplesShadowMap && !UsesForwardPlusMaterial(pass) ? std::uint64_t{1}<<44 : 0) |
               (UsesSurfacePCSS(pipeline, pass) ? std::uint64_t{1} << 34 : 0) |
               (UsesVSM() && (pass == PassType::Shadow || UsesSurfaceVSM(pipeline, pass)) ? std::uint64_t{1} << 35 : 0) |
               (UsesContactReceiver(pipeline, pass) ? std::uint64_t{1} << 36 : 0) |
               (UsesForwardPlusMaterial(pass) ? std::uint64_t{1} << 37 : 0) |
               (UsesForwardPlusMaterial(pass) && ClusteredActive ? std::uint64_t{1} << 40 : 0) |
               (UsesForwardPlusMaterial(pass) && TiledActive ? std::uint64_t{1} << 41 : 0) |
               // Unlike the legacy dynamic filter, Forward+ owns separate Hard/PCF modules.
               (UsesForwardPlusMaterial(pass) ? std::uint64_t(ForwardPlusFilter(pipeline, pass)) << ForwardPlusFilterKeyShift : 0);
    }

    VkPipeline GetPipeline(const PipelineState& pipeline, PassType pass, bool instanced = false)
    {
        const bool shadow = pass == PassType::Shadow;
        const bool contactDepth = pass == PassType::ContactDepth;
        const bool forwardPlusDepth = pass == PassType::ForwardPlusDepth;
        const bool forwardPlus = UsesForwardPlusMaterial(pass) || forwardPlusDepth;
        if (forwardPlus && pipeline.Shading == Techniques::ShadingModel::Gouraud)
        { Fail("Vulkan Forward+ pixel lists cannot supply Gouraud vertex lighting"); return 0; }
        const bool contactReceiver = UsesContactReceiver(pipeline, pass);
        const bool vsmCaster = shadow && UsesVSM();
        const auto key = PipelineKey(pipeline, pass, instanced);
        const auto found = Pipelines.find(key);
        if (found != Pipelines.end()) return found->second;

        const bool parallax=UsesParallaxMaterial(pipeline,pass);
        if (parallax && !EnsureParallax()) return 0;
        const bool geometry = pass == PassType::GBuffer;
        const bool velocity = pass == PassType::Velocity;
        const bool cascaded = shadow && FrameCascades.Frame.CascadeCount > 0;
        if (shadow && !cascaded && !CreateShadowModules()) return 0;
        if (!geometry && !velocity && !shadow && !forwardPlus && !CreateShaderModules(pipeline.Shading)) return 0;
        if (WeightedPass && !forwardPlus && !EnsureWeightedSurface(pipeline.Shading)) return 0;
        const auto& modules = TechniqueModules[ShadingTechniqueIndex(pipeline.Shading, FrameAreaLights)];
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = velocity ? MotionGeometryModules[0] : (geometry ? DeferredGeometryModules[0] : modules[0]);
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = velocity ? MotionGeometryModules[1] : (geometry ? DeferredGeometryModules[1] : modules[1]);
        stages[1].pName = "main";
        if (WeightedPass) stages[1].module = WeightedSurfaceModules[ShadingTechniqueIndex(pipeline.Shading, FrameAreaLights)];
        if (UsesSurfacePCSS(pipeline, pass) && !forwardPlus)
        {
            if (!EnsurePCSSSurface(pipeline.Shading, WeightedPass)) return 0;
            stages[1].module = PCSSSurfaceModules[static_cast<std::size_t>(pipeline.Shading) + (WeightedPass ? 8u : 0u)];
        }
        if (shadow)
        {
            stages[0].module = cascaded ? CascadeModules[0] : ShadowModules[0];
            stages[1].module = cascaded ? CascadeModules[1] : ShadowModules[1];
        }
        if (contactDepth)
        {
            for (std::size_t stage = 0; stage < ContactDepthModules.size(); ++stage)
                if (!EnsureContactModule(VulkanContactDepthShaders[stage], ContactDepthModules[stage])) return 0;
            stages[0].module = ContactDepthModules[0]; stages[1].module = ContactDepthModules[1];
        }
        if (forwardPlus)
        {
            const auto filter = ForwardPlusFilter(pipeline, pass); const auto variant = ForwardPlusVariant(pipeline, pass);
            auto& vertex = ClusteredActive ? ClusteredVertexModule : TiledActive ? TiledVertexModule : ForwardPlusVertexModule;
            auto& fragment = (ClusteredActive ? ClusteredFragments : TiledActive ? TiledFragments : ForwardPlusFragments)[filter][variant];
            if (!EnsureForwardPlusModule(ClusteredActive ? VulkanClusteredForwardVertexShader : TiledActive ? VulkanTiledDeferredVertexShader : VulkanForwardPlusVertexShader, vertex) ||
                !EnsureForwardPlusModule(ClusteredActive ? VulkanClusteredForwardFragmentShaders[filter][variant] : TiledActive ? VulkanTiledDeferredFragmentShaders[filter][variant] : VulkanForwardPlusFragmentShaders[filter][variant], fragment)) return 0;
            stages[0].module = vertex; stages[1].module = fragment;
        }
        if (instanced)
        {
            const auto program = velocity ? InstanceVertexProgram::Velocity : shadow ? InstanceVertexProgram::Shadow :
                geometry ? InstanceVertexProgram::Deferred : !contactDepth && pipeline.Shading == Techniques::ShadingModel::Gouraud ?
                InstanceVertexProgram::Gouraud : InstanceVertexProgram::Forward;
            const auto index = cascaded ? CascadeInstanceProgram : static_cast<std::size_t>(program);
            if (!EnsureInstanceShader(index)) return 0;
            stages[0].module = InstanceModules[index];
        }
        if (vsmCaster)
        {
            const std::size_t vertex = instanced ? 1 : 0;
            if (!EnsureVSMModule(VulkanVSMCasterShaders[vertex][0], VSMCasterModules[vertex]) ||
                !EnsureVSMModule(VulkanVSMCasterShaders[vertex][1], VSMCasterModules[2])) return 0;
            stages[0].module = VSMCasterModules[vertex]; stages[1].module = VSMCasterModules[2];
        }
        else if (UsesSurfaceVSM(pipeline, pass) && !forwardPlus)
        {
            constexpr std::size_t WeightedFirst = 10;
            const auto index = static_cast<std::size_t>(pipeline.Shading) + (WeightedPass ? WeightedFirst : 0);
            if (!EnsureVSMModule(VulkanVSMFragments[index], VSMSurfaceModules[index])) return 0;
            stages[1].module = VSMSurfaceModules[index];
        }
        if (contactReceiver && !forwardPlus)
        {
            // Match the pipeline key and uploaded filter suffix. A draw without
            // map sampling needs only the base receiver even if PCSS/VSM was requested.
            const std::size_t family = UsesSurfaceVSM(pipeline, pass) ? 2 : UsesSurfacePCSS(pipeline, pass) ? 1 : 0;
            const auto samples = FrameSampleCount > 1 ? 1u : 0u;
            const auto model = static_cast<std::size_t>(pipeline.Shading);
            auto& module = ContactSurfaceModules[family][samples][model];
            if (!EnsureContactModule(VulkanContactSurfaceShaders[family][samples][model], module)) return 0;
            stages[1].module = module;
        }

        if (UsesMaterialStage(pipeline,pass))
        {
            const auto material=SelectMaterialTechnique(pipeline,WeightedPass);
            const unsigned model=unsigned(pipeline.Shading);
            if (!EnsureMaterialFragment(material,model,WeightedPass,geometry)) return 0;
            stages[1].module=MaterialFragments[MaterialProgramIndex(material,model,WeightedPass,geometry)];
        }
        if (UsesSurfacePCF(pipeline,pass))
        {
            const unsigned model=unsigned(pipeline.Shading);
            if (!EnsurePCFFragment(model,WeightedPass)) return 0;
            stages[1].module=PCFFragments[WeightedPass?8+model:model];
        }
        if (FrameRayShadowImage && !FrameAreaLights && pass==PassType::Color && pipeline.SamplesShadowMap && !forwardPlus)
        {
            const unsigned model=unsigned(pipeline.Shading);
            if (!EnsureRayShadowFragment(model,WeightedPass)) return 0;
            stages[1].module=RayShadowFragments[WeightedPass?8+model:model];
        }
        if (parallax) { stages[0].module=ParallaxModules[0]; stages[1].module=ParallaxModules[1]; }

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
            ? VK_CULL_MODE_FRONT_BIT : (pipeline.BackFaceCulling ? CullBackFaces : VK_CULL_MODE_NONE);
        raster.frontFace = RealtimeWorldIsMirrored(BoundWorld) != (pass == PassType::PlanarCapture)
            ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.depthBiasEnable = shadow ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        raster.depthBiasConstantFactor = shadow ? static_cast<float>(ShadowDepthBiasUnits) : 0.0f;
        raster.depthBiasClamp = ShadowDepthBiasClamp;
        raster.depthBiasSlopeFactor = shadow ? ShadowSlopeScaledDepthBias : 0.0f;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = ActiveDepth ? ActiveDepth->Samples : 1;
        multisample.sampleShadingEnable = contactReceiver && FrameSampleCount > 1 ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        multisample.minSampleShading = 1.0f;
        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = pipeline.DepthTest ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        depth.depthWriteEnable = pipeline.DepthWrite ? VK_TRUE_VALUE : VK_FALSE_VALUE;
        depth.depthCompareOp = CurrentPass.DepthPeelingLayer ? VK_COMPARE_OP_EQUAL : pipeline.DepthWrite ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_LESS_OR_EQUAL;
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
        colorAttachment.colorWriteMask = pipeline.ColorWrite || vsmCaster
            ? VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            : 0U;
        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        std::array<VkPipelineColorBlendAttachmentState, GBufferAttachmentCount> materialBlends;
        materialBlends.fill(colorAttachment);
        colorBlend.attachmentCount = contactDepth || forwardPlusDepth || (shadow && !vsmCaster) ? 0U : (geometry ? static_cast<std::uint32_t>(materialBlends.size()) : 1U);
        colorBlend.pAttachments = contactDepth || forwardPlusDepth || (shadow && !vsmCaster) ? nullptr : (geometry ? materialBlends.data() : &colorAttachment);
        if (WeightedPass)
        {
            // A = sum(C * alpha * w, alpha * w), R = product(1 - alpha).
            auto& sum = materialBlends[0];
            sum.blendEnable = VK_TRUE_VALUE;
            sum.srcColorBlendFactor = sum.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            sum.srcAlphaBlendFactor = sum.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            auto& revealage = materialBlends[1];
            revealage.blendEnable = VK_TRUE_VALUE;
            revealage.srcColorBlendFactor = revealage.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            revealage.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            revealage.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            revealage.colorWriteMask &= VK_COLOR_COMPONENT_R_BIT;
            colorBlend.attachmentCount = 2;
            colorBlend.pAttachments = materialBlends.data();
        }
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
        create.renderPass = velocity ? MotionGeometryRenderPass : (shadow ? ShadowRenderPasses[0] : (geometry ? DeferredRenderPasses[0] :
            ((pass == PassType::EnvironmentCapture || pass == PassType::PlanarCapture || ActiveColor == &HdrImages[0]) ? HdrColorRenderPasses[0] : ColorRenderPasses[0])));
        if (vsmCaster) create.renderPass = VSMCasterPasses[0];
        if (ActiveDepth && ActiveDepth->Samples > 1)
            create.renderPass = velocity ? Multisample.VelocityPass :
                (geometry ? Multisample.MaterialPasses[0] : Multisample.ColorPasses[0]);
        create.subpass = 0;
        if (WeightedPass) create.renderPass = Weighted.GeometryPasses[0];
        if (contactDepth || forwardPlusDepth) create.renderPass = ContactDepthPass;
        if (CurrentPass.DepthPeelingLayer) create.renderPass = HdrColorRenderPasses[0];
        VkPipeline result = 0;
        if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0, 1, &create, nullptr, &result),
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
    std::array<std::array<VkShaderModule, 2>, StandaloneShadingTechniqueCount> TechniqueModules{};
    std::array<VkShaderModule, 13> PCSSSurfaceModules {};
    std::array<VkShaderModule, 15> VSMSurfaceModules {};
    std::array<VkShaderModule, 3> VSMCasterModules {};
    std::array<VkShaderModule, 2> VSMFilterModules {};
    std::array<VkRenderPass, 2> VSMCasterPasses {};
    VkRenderPass VSMFilterPass = 0;
    VkPipeline VSMFilterPipeline = 0;
    VkPipelineLayout VSMFilterLayout = 0;
    VkDescriptorSetLayout VSMFilterDescriptorLayout = 0;
    VkSampler VSMSampler = 0;
    bool VSMSupported = false;
    VkImageFormatProperties VSMImageLimits {};
    std::unordered_map<std::uint32_t, VSMMomentMap> VSMMoments;
    std::array<VkShaderModule, 2> ShadowModules{};
    std::array<VkShaderModule, 2> CascadeModules {};
    VkDescriptorSetLayout CascadeDescriptorLayout = 0;
    VkDescriptorPool CascadeDescriptorPool = 0;
    VkSampler CascadeSampler = 0;
    std::array<VkDescriptorSet, 2> CascadeDescriptorSets {};
    CascadedShadowState FrameCascades {};
    Techniques::ShadowFilterSettings FrameShadowFilter;
    std::array<RawImage*, MaximumShadowCascades> CascadeImages {};
    std::array<RawBuffer, MaximumShadowCascades> CascadeReadbacks {};
    VkDescriptorSetLayout DescriptorSetLayout = 0;
    VkPipelineLayout PipelineLayout = 0;
    VkDescriptorPool DescriptorPool = 0;
    std::vector<VkDescriptorSet> DescriptorSets;
    std::uint32_t DescriptorSetIndex = 0;
    VkSampler ShadowSampler = 0, AnisotropicSampler = 0;
    bool AnisotropicFilteringSupported = false, AnisotropicFiltering = false;
    VkSampler MaterialSampler() const { return AnisotropicFiltering && AnisotropicSampler ? AnisotropicSampler : ShadowSampler; }
    RawBuffer UniformBuffer;
    VkDescriptorSetLayout InstanceDescriptorLayout = 0;
    std::array<VkShaderModule, VulkanInstanceProgramCount> InstanceModules {};
    std::vector<InstanceUploadPage> InstancePages;
    std::size_t InstancePageIndex = 0, InstancePageOffset = 0;
    std::size_t UniformOffset = 0;
    RawImage DummyDepth;
    RawImage DummyColor;
    RawImage MainColor;
    std::array<RawImage, 4> HdrImages{};
    PhysicalDeviceLimitsPrefix DeviceLimits {};
    bool SampleRateShadingSupported = false;
    bool GraphicsQueueHasCompute = false, ContactSupported = false;
    std::unique_ptr<VulkanContactShadows> ContactProducer;
    std::unique_ptr<VulkanForwardPlus> ForwardPlusProducer;
    std::unique_ptr<VulkanForwardPlus> TiledProducer;
    bool TiledActive = false;
    VulkanForwardPlus* TileProducer() const { return TiledActive ? TiledProducer.get() : ForwardPlusProducer.get(); }
    std::unique_ptr<VulkanClusteredForward> ClusteredProducer;
    bool ClusteredActive = false;
    LightListFrameState LightListFrame;
    RawBuffer ForwardPlusSceneStorage, ForwardPlusEmptyStorage;
    bool ForwardPlusSupported = false, LightListsActive = false, ForwardPlusDepthWritten = false;
    bool ForwardPlusRecordOpen = false, ForwardPlusRecorded = false, ForwardPlusSubmitted = false;
    VkShaderModule ForwardPlusVertexModule = 0, ClusteredVertexModule = 0, TiledVertexModule = 0;
    std::array<std::array<VkShaderModule, ForwardPlusProgramCount>, ForwardPlusFilterCount> ForwardPlusFragments {}, ClusteredFragments {}, TiledFragments {};
    Techniques::ContactShadowSettings FrameContact;
    bool ContactDepthWritten = false, ContactGenerated = false, ContactReceiverActive = false;
    VkRenderPass ContactDepthPass = 0;
    VkFramebuffer ContactDepthFramebuffer = 0;
    std::array<VkRenderPass, 2> ContactColorPasses {};
    std::array<VkShaderModule, 2> ContactDepthModules {};
    std::array<std::array<std::array<VkShaderModule, 5>, 2>, 3> ContactSurfaceModules {};
    struct ContactDeferredState { std::array<VkShaderModule, 2> Modules {}; VkPipeline Pipeline = 0; };
    std::array<std::array<ContactDeferredState, 4>, 3> ContactDeferred {};
    std::array<std::array<std::array<ContactDeferredState, 2>, 4>, 4> TiledLighting {};
    bool IndependentBlendSupported = false;
    bool FragmentStoresAndAtomicsSupported=false;
    std::uint32_t WeightedSampleCounts = 0;
    inline static constexpr std::array<VkFormat, 2> WeightedFormats {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32_SFLOAT};
    std::array<VkImageFormatProperties, 2> WeightedLimits {};
    struct WeightedTargets
    {
        std::array<RawImage, 2> Images {};
        std::array<VkRenderPass, 2> GeometryPasses {};
        VkFramebuffer GeometryFramebuffer = 0, ResolveFramebuffer = 0;
        VkRenderPass ResolvePass = 0;
        VkPipeline ResolvePipeline = 0;
        VkImageView DepthView = 0, SceneView = 0;
    } Weighted;
    std::array<VkShaderModule, StandaloneShadingTechniqueCount> WeightedSurfaceModules {};
    std::array<std::array<VkShaderModule, 2>, 2> WeightedResolveModules {};
    VkDescriptorSetLayout WeightedDescriptorLayout = 0;
    VkPipelineLayout WeightedResolveLayout = 0;
    VkDescriptorPool WeightedDescriptorPool = 0;
    VkDescriptorSet WeightedDescriptorSet = 0;
    VkSampler WeightedSampler = 0;
    Techniques::TransparencySettings FrameTransparency {};
    bool WeightedActive = false, WeightedPass = false, WeightedPrepared = false;
    bool WeightedAccumulated = false, WeightedResolved = false;
    std::uint32_t SupportedSampleCounts = 1, FrameSampleCount = 1;
    bool SamplesOpaqueResolved = false, SamplesVelocityResolved = false;
    bool SamplesColorRestored = false, SamplesColorResolved = false, SamplesTransparentWritten = false;
    std::array<bool, 5> SampleDescriptorsWritten {};
    bool DeferredSupported = false;
    bool SaoSupported = false;
    VkImageFormatProperties SaoImageLimits {};
    bool DeferredActive = false;
    bool GBufferWritten = false;
    bool DeferredResolved = false;
    bool AmbientOcclusionActive = false;
    bool AmbientOcclusionResolved = false;
    RawImage ReflectionImage;
    VkFramebuffer ReflectionFramebuffer = 0;
    VkRenderPass ReflectionCompositeRenderPass = 0;
    std::array<VkShaderModule, 2> ReflectionModules {};
    std::array<VkShaderModule, 2> CubeReflectionModules {}, CubeMipModules {};
    std::array<VkPipeline, ReflectionPhaseCount> CubeReflectionPipelines {};
    VkPipeline CubeMipPipeline = 0;
    std::array<VkShaderModule, 2> EnvironmentMapModules {}, EnvironmentConversionModules {}, EnvironmentMipModules {};
    std::array<VkPipeline, ReflectionPhaseCount> EnvironmentMapPipelines {};
    VkPipeline EnvironmentConversionPipeline = 0, EnvironmentMipPipeline = 0;
    std::array<VkShaderModule, 2> PlanarModules {}, PlanarMipModules {};
    std::array<VkPipeline, ReflectionPhaseCount> PlanarPipelines {};
    VkPipeline PlanarMipPipeline = 0;
    PlanarCaptureFrame FramePlanar {};
    bool PlanarActive = false, PlanarWritten = false, PlanarReady = false;
    EnvironmentCaptureFrame FrameEnvironment {};
    std::uint32_t EnvironmentFacesWritten = 0;
    bool EnvironmentActive = false, EnvironmentReady = false;
    std::array<VkPipeline, ReflectionPhaseCount> ReflectionPipelines {};
    VkDescriptorPool ReflectionDescriptorPool = 0;
    VkDescriptorSetLayout ReflectionDescriptorLayout = 0;
    VkPipelineLayout ReflectionPipelineLayout = 0;
    std::array<VkDescriptorSet, ReflectionDescriptorCount> ReflectionDescriptors {};
    std::array<VkSampler, ReflectionSamplerBindings.size()> ReflectionSamplers {}; // Point, clamp trilinear, wrap-U trilinear.
    bool ReflectionActive = false, ReflectionResolved = false;
    Techniques::ReflectionSettings FrameReflections {};
    Techniques::ShadingModel FrameReflectionShading = Techniques::ShadingModel::Phong;
    bool ForwardOpaqueComplete = false;
    Techniques::AmbientOcclusionSettings FrameAmbientOcclusion;
    std::array<RawImage, 2> AmbientOcclusionImages {};
    std::array<VkFramebuffer, 2> AmbientOcclusionFramebuffers {};
    std::array<VkShaderModule, 2> SsaoModules {};
    std::array<VkPipeline, 2> SsaoPipelines {};
    std::array<VkShaderModule, 2> HbaoModules {};
    std::array<VkPipeline, 2> HbaoPipelines {};
    std::array<VkShaderModule, 2> GtaoModules {};
    std::array<VkPipeline, 2> GtaoPipelines {};
    std::array<VkShaderModule, 2> SaoModules {};
    std::array<VkPipeline, 3> SaoPipelines {};
    RawImage SaoDepthHierarchy;
    RawImage SaoScratch;
    VkFramebuffer SaoScratchFramebuffer = 0;
    VkRenderPass SaoDepthRenderPass = 0;
    VkRenderPass AmbientOcclusionCompositeRenderPass = 0;
    Float4 DeferredBackground;
    std::array<RawImage, GBufferAttachmentCount> GBufferImages {};
    VkFramebuffer GBufferFramebuffer = 0;
    std::array<VkRenderPass, 4> DeferredRenderPasses {};
    std::array<VkShaderModule, 2> DeferredGeometryModules {};
    std::array<VkShaderModule, 2> DeferredLightingModules {};
    VkPipeline DeferredLightingPipeline = 0;
    VkPipeline PCSSDeferredPipeline = 0;
    VkPipeline VSMDeferredPipeline = 0;
    std::array<VkShaderModule, 2> VSMDeferredModules {};
    std::array<VkShaderModule, 2> PCSSDeferredModules {};
    VkDescriptorPool DeferredDescriptorPool = 0;
    VkDescriptorSetLayout DeferredDescriptorLayout = 0;
    VkDescriptorSet DeferredDescriptorSet = 0;
    VkPipelineLayout DeferredPipelineLayout = 0;
    VkSampler DeferredSampler = 0;
    std::array<VkFramebuffer, 4> HdrFramebuffers{};
    std::array<VkRenderPass, 4> HdrColorRenderPasses{};
    std::array<VkPipeline, HdrTechniqueCount> HdrPipelines{};
    std::array<std::array<VkShaderModule, 2>, HdrTechniqueCount> HdrModules{};
    VkRenderPass HdrRenderPass = 0;
    VkFramebuffer HdrSceneFramebuffer = 0;
    VkFramebuffer HdrResolveFramebuffer = 0;
    bool HdrActive = false;
    Techniques::PostProcessSettings FramePostProcessing;
    bool DepthOfFieldActive = false;
    bool DepthOfFieldResolved = false;
    std::array<VkShaderModule, 2> DepthOfFieldModules {};
    std::array<VkPipeline, 2> DepthOfFieldPipelines {};
    VkRenderPass DepthOfFieldCompositeRenderPass = 0;
    VkSampler DepthOfFieldSampler = 0;
    static constexpr std::size_t MotionVelocity = 0, MotionTileMax = 1, MotionNeighborMax = 2;
    std::array<RawImage, 3> MotionImages {};
    std::array<VkFramebuffer, 3> MotionFramebuffers {};
    RawBuffer MotionReadback;
    std::array<VkShaderModule, 2> MotionGeometryModules {}, MotionFilterModules {};
    std::array<VkPipeline, 2> MotionFilterPipelines {};
    VkRenderPass MotionGeometryRenderPass = 0, MotionCompositeRenderPass = 0;
    VkSampler MotionSampler = 0;
    bool MotionBlurActive = false, VelocityCaptured = false, MotionBlurResolved = false;
    bool VelocityActive = false, TemporalActive = false, TemporalResolved = false;
    static constexpr std::size_t GIRaw = 0, GIFiltered = 1, GIComposite = 2, GIColorFirst = 3,
        GIDepthFirst = 5, GINormalFirst = 7, GITargetCount = 9;
    static constexpr std::uint32_t GIInputCount = 10, GIAttachmentCount = 3, GIPhaseCount = 4;
    std::array<RawImage, GITargetCount> GIImages {};
    std::array<VkFramebuffer, GIDepthFirst> GIFramebuffers {};
    std::array<VkShaderModule, 2> GIModules {}, GITemporalModules {};
    VkPipeline GIPipeline = 0, GITemporalPipeline = 0;
    VkRenderPass GIRenderPass = 0;
    VkPipelineLayout GIPipelineLayout = 0;
    VkDescriptorSetLayout GIDescriptorLayout = 0;
    VkDescriptorPool GIDescriptorPool = 0;
    std::array<VkDescriptorSet, GIPhaseCount> GIDescriptorSets {};
    VkSampler GISampler = 0;
    static constexpr std::size_t ProbeRaw=0,ProbeSH=1,ProbeMoments=2,ProbeComposite=3;
    static constexpr std::uint32_t ProbeInputCount=8,ProbePhaseCount=3;
    std::array<RawImage,4> ProbeImages {};
    std::array<VkFramebuffer,4> ProbeFramebuffers {};
    std::array<VkPipeline,2> ProbePipelines {};
    std::array<std::array<VkShaderModule,2>,2> ProbeModules {};
    VkPipelineLayout ProbePipelineLayout=0;
    VkDescriptorSetLayout ProbeDescriptorLayout=0;
    VkDescriptorPool ProbeDescriptorPool=0;
    VkSampler ProbeSampler=0;
    std::array<VkDescriptorSet,ProbePhaseCount> ProbeDescriptorSets {};
    RawBuffer ProbeUpload;
    ProbeProjectionState Probes;
    std::shared_ptr<const BakedProbeSamples> FrameProbeSamples,UploadedProbeSamples;
    GlobalIlluminationHistory GI;
    Techniques::GlobalIlluminationSettings FrameGI;
    bool GIActive = false, EarlyTemporalActive = false, ProbeActive = false;
    static constexpr std::uint32_t TemporalAttachmentCount = 2, TemporalInputCount = 5;
    static constexpr std::size_t TemporalPointSampler = 0, TemporalLinearSampler = 1;
    std::array<RawImage, 2> TemporalColors {}, TemporalDepths {};
    std::array<VkFramebuffer, 2> TemporalFramebuffers {};
    std::array<VkShaderModule, 2> TemporalModules {};
    std::array<VkSampler, 2> TemporalSamplers {};
    VkPipeline TemporalPipeline = 0;
    VkRenderPass TemporalRenderPass = 0;
    VkPipelineLayout TemporalPipelineLayout = 0;
    VkDescriptorSetLayout TemporalDescriptorLayout = 0;
    VkDescriptorPool TemporalDescriptorPool = 0;
    VkDescriptorSet TemporalDescriptorSet = 0;
    VkImageFormatProperties TemporalColorLimits {}, TemporalDepthLimits {};
    std::size_t TemporalHistoryReadIndex = 0;
    bool TemporalHistoryValid = false, TemporalSupported = false;
    bool LightShaftActive = false, LightShaftResolved = false, LightShaftSupported = false;
    Techniques::LightShaftSettings FrameLightShafts;
    RawImage LightShaftMask;
    VkImageFormatProperties LightShaftImageLimits {};
    VkFramebuffer LightShaftFramebuffer = 0;
    VkPipeline LightShaftPipeline = 0;
    VkPipelineLayout LightShaftPipelineLayout = 0;
    VkDescriptorSetLayout LightShaftDescriptorLayout = 0;
    VkDescriptorPool LightShaftDescriptorPool = 0;
    VkSampler LightShaftSampler = 0;
    std::array<VkShaderModule, 2> LightShaftModules {};
    std::array<VkDescriptorSet, LightShaftPhaseCount> LightShaftSets {};
    std::array<std::uint32_t, LightShaftPhaseCount> LightShaftOffsets {};
    bool FogActive = false, FogResolved = false, FogSupported = false;
    Techniques::VolumetricFogSettings FrameFog;
    Techniques::VolumetricLightingSettings FrameVolumeLighting;
    bool VolumeLightSupported = false;
    VkImageFormatProperties VolumeLightImageLimits {};
    std::array<RawImage, 2> VolumeLightImages;
    std::array<VkFramebuffer, 2> VolumeLightFramebuffers {};
    VkRenderPass VolumeLightRenderPass = 0;
    std::array<VkPipeline, 2> VolumeLightPipelines {}, MediaVolumePipelines {}, CloudVolumePipelines {};
    VkPipeline PCSSVolumePipeline = 0;
    VkPipeline VSMVolumePipeline = 0;
    std::array<VkShaderModule, 2> VSMVolumeModules {};
    std::array<VkShaderModule, 2> PCSSVolumeModules {};
    std::array<VkShaderModule, 2> VolumeLightModules {}, MediaVolumeModules {}, CloudVolumeModules {};
    VkDescriptorSetLayout VolumeLightDescriptorLayout = 0;
    VkPipelineLayout VolumeLightPipelineLayout = 0;
    VkDescriptorPool VolumeLightDescriptorPool = 0;
    std::array<VkDescriptorSet, VolumeLightPassLimit> VolumeLightSets {};
    FogGrid FogDimensions;
    RawImage FogAtlas;
    VkImageFormatProperties FogImageLimits {};
    VkFramebuffer FogFramebuffer = 0;
    VkPipeline FogPipeline = 0, MediaFogPipeline = 0, CloudFogPipeline = 0;
    VkPipelineLayout FogPipelineLayout = 0;
    VkDescriptorSetLayout FogDescriptorLayout = 0;
    VkDescriptorPool FogDescriptorPool = 0;
    std::array<VkDescriptorSet, FogSetCount> FogDescriptorSets {};
    std::array<std::uint32_t, FogSetCount> FogUniformOffsets {};
    std::array<VkShaderModule, 2> FogModules {}, MediaFogModules {}, CloudFogModules {};
    std::array<VkSampler, 2> FogSamplers {};
    std::vector<std::pair<RawImage*, VkImageLayout>> FrameImageLayouts;
    bool FrameSubmitted = false;
    bool FrameCommandEnded = false;
    std::array<RawImage, 2> PostImages{};
    std::array<VkFramebuffer, 2> PostFramebuffers{};
    std::array<VkPipeline, PostProcessTechniqueCount> PostPipelines{};
    VkSampler PostSampler = 0;
    std::array<std::array<VkShaderModule, 2>, PostProcessTechniqueCount> PostModules{};
    VkRenderPass PostRenderPass = 0;
    RawImage MainDepth;
    VkFramebuffer MainFramebuffer = 0;
    RawBuffer ColorReadback;
    RawBuffer DepthReadback;
    RawBuffer ShadowReadback;
    std::unordered_map<std::uint32_t, BufferResource> Buffers;
    std::unordered_map<std::uint32_t, TextureResource> Textures;
    std::unordered_map<std::uint64_t, VkPipeline> Pipelines;
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
    bool FrameAreaLights = false;
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
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties GetPhysicalDeviceImageFormatProperties = nullptr;
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
    PFN_vkCmdCopyImage CmdCopyImage = nullptr;
    PFN_vkCmdResolveImage CmdResolveImage = nullptr;
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass = nullptr;
    PFN_vkCmdEndRenderPass CmdEndRenderPass = nullptr;
    PFN_vkCmdSetViewport CmdSetViewport = nullptr;
    PFN_vkCmdSetScissor CmdSetScissor = nullptr;
    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants CmdPushConstants = nullptr;
    PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers = nullptr;
    PFN_vkCmdBindIndexBuffer CmdBindIndexBuffer = nullptr;
    PFN_vkCmdDrawIndexed CmdDrawIndexed = nullptr;
    PFN_vkCmdDraw CmdDraw = nullptr;
};

VulkanRHI::VulkanRHI() : m_Impl(std::make_unique<Impl>()) {}
// Unresolved shutdown waits must not destroy private pending command pools.
VulkanRHI::~VulkanRHI() { Shutdown(); if (m_Impl && m_Impl->Abandoned) (void)m_Impl.release(); }

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
    SetEnvironmentCaptureFrame({});
    SetPlanarCaptureFrame({});
    SetCascadedShadowState({});
    if (!m_Impl) return;
    if (m_Impl->Abandoned) return;
    if (!m_Impl->LumenCaptureTasks.Visit([&](auto& work)
        { return !work.InFlight || m_Impl->Check(m_Impl->WaitForFences(m_Impl->Device,1,&work.Completion,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for Lumen captures during shutdown"); }))
    { m_Impl->Initialized=false; m_Impl->FrameOpen=m_Impl->PassOpen=false; m_Impl->Abandoned=true; return; }
    if (!m_Impl->LumenLightingTasks.Visit([&](auto& work)
        { return !work.InFlight || m_Impl->Check(m_Impl->WaitForFences(m_Impl->Device,1,&work.Completion,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for Lumen lighting tiles during shutdown"); }))
    { m_Impl->Initialized=false; m_Impl->FrameOpen=m_Impl->PassOpen=false; m_Impl->Abandoned=true; return; }
    if (!m_Impl->LumenGlobalTasks.Visit([&](auto& work)
        { return !work.InFlight || m_Impl->Check(m_Impl->WaitForFences(m_Impl->Device,1,&work.Completion,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for global distance slices during shutdown"); }))
    { m_Impl->Initialized=false; m_Impl->FrameOpen=m_Impl->PassOpen=false; m_Impl->Abandoned=true; return; }
    if (!m_Impl->NaniteCacheTasks.Visit([&](auto& work)
        { return !work.Completion || m_Impl->Check(m_Impl->WaitForFences(m_Impl->Device,1,&work.Completion,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for Nanite page copies during shutdown"); }))
    { m_Impl->Initialized=false; m_Impl->FrameOpen=m_Impl->PassOpen=false; m_Impl->Abandoned=true; return; }
    if (!m_Impl->NaniteAsyncTasks.Visit([&](auto& work)
        { return m_Impl->Check(m_Impl->WaitForFences(m_Impl->Device,1,&work.Completion,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for Nanite production during shutdown"); }))
    { m_Impl->Initialized=false; m_Impl->FrameOpen=m_Impl->PassOpen=false; m_Impl->Abandoned=true; return; }
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
            // The existing abandoned-device policy retains objects after a GPU
            // timeout. Do not let the producer's RAII destructor free live work.
            (void)m_Impl->ContactProducer.release();
            (void)m_Impl->ForwardPlusProducer.release(); (void)m_Impl->TiledProducer.release(); (void)m_Impl->ClusteredProducer.release();
            return;
        }
    }
    if (m_Impl->Device != nullptr)
    {
        if ((m_Impl->ForwardPlusRecordOpen || m_Impl->ForwardPlusSubmitted) &&
            (!m_Impl->Check(m_Impl->ResetCommandPool(m_Impl->Device, m_Impl->CommandPool, 0), "vkResetCommandPool(Forward+ shutdown)") ||
             !m_Impl->ResetForwardPlusAfterDiscard()))
        {
            m_Impl->Initialized = false; m_Impl->FrameOpen = m_Impl->PassOpen = false; m_Impl->Abandoned = true;
            (void)m_Impl->ContactProducer.release(); (void)m_Impl->ForwardPlusProducer.release(); (void)m_Impl->TiledProducer.release(); (void)m_Impl->ClusteredProducer.release();
            return;
        }
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
        // Moment framebuffers reference the source depth views.
        m_Impl->DestroyVSMState();
        m_Impl->DestroyContactState();
        m_Impl->DestroyForwardPlusState();
        for (auto& [handle, resource] : m_Impl->Textures)
        {
            (void)handle;
            m_Impl->DestroyTextureResource(resource);
        }
        m_Impl->DestroyMainTarget();
        for (auto& module : m_Impl->WeightedSurfaceModules)
        {
            if (module) m_Impl->DestroyShaderModuleFunction(m_Impl->Device, module, nullptr);
            module = 0;
        }
        for (auto& pair : m_Impl->WeightedResolveModules)
            for (auto& module : pair)
            {
                if (module) m_Impl->DestroyShaderModuleFunction(m_Impl->Device, module, nullptr);
                module = 0;
            }
        if (m_Impl->WeightedDescriptorPool) m_Impl->DestroyDescriptorPoolFunction(m_Impl->Device, m_Impl->WeightedDescriptorPool, nullptr);
        if (m_Impl->WeightedResolveLayout) m_Impl->DestroyPipelineLayoutFunction(m_Impl->Device, m_Impl->WeightedResolveLayout, nullptr);
        if (m_Impl->WeightedDescriptorLayout) m_Impl->DestroyDescriptorSetLayoutFunction(m_Impl->Device, m_Impl->WeightedDescriptorLayout, nullptr);
        if (m_Impl->WeightedSampler) m_Impl->DestroySamplerFunction(m_Impl->Device, m_Impl->WeightedSampler, nullptr);
        m_Impl->WeightedDescriptorPool = m_Impl->WeightedResolveLayout = m_Impl->WeightedDescriptorLayout =
            m_Impl->WeightedDescriptorSet = m_Impl->WeightedSampler = 0;
        m_Impl->WeightedActive = m_Impl->WeightedPass = m_Impl->WeightedPrepared = m_Impl->WeightedAccumulated = m_Impl->WeightedResolved = false;
        m_Impl->DestroyAmbientOcclusionState();
        m_Impl->DestroyReflectionState();
        m_Impl->DestroyDeferredState();
        for (auto& module : m_Impl->PCSSSurfaceModules)
        {
            if (module && m_Impl->DestroyShaderModuleFunction) m_Impl->DestroyShaderModuleFunction(m_Impl->Device, module, nullptr);
            module = 0;
        }
        m_Impl->DestroyDepthOfFieldState();
        m_Impl->DestroyMotionState();
        m_Impl->DestroyTemporalState();
        m_Impl->DestroyGIState(); m_Impl->DestroyProbeState(); m_Impl->DestroyLightmapState(); m_Impl->DestroyRefractionState(); m_Impl->DestroyPeelingState();
        m_Impl->DestroyFogState();
        m_Impl->DestroyLightShaftState();
        m_Impl->DestroyCascadeState();
        m_Impl->DestroyGpuDriven();
        m_Impl->DestroyVisibilityBuffer();
        m_Impl->DestroyInstanceState();
        for (auto& pipeline : m_Impl->HdrPipelines)
        {
            if (pipeline && m_Impl->DestroyPipelineFunction) m_Impl->DestroyPipelineFunction(m_Impl->Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& modules : m_Impl->HdrModules)
            for (auto& module : modules)
            {
                if (module && m_Impl->DestroyShaderModuleFunction) m_Impl->DestroyShaderModuleFunction(m_Impl->Device, module, nullptr);
                module = 0;
            }
        for (auto& pass : m_Impl->HdrColorRenderPasses)
        {
            if (pass && m_Impl->DestroyRenderPassFunction) m_Impl->DestroyRenderPassFunction(m_Impl->Device, pass, nullptr);
            pass = 0;
        }
        if (m_Impl->HdrRenderPass && m_Impl->DestroyRenderPassFunction)
            m_Impl->DestroyRenderPassFunction(m_Impl->Device, m_Impl->HdrRenderPass, nullptr);
        m_Impl->HdrRenderPass = 0;
        m_Impl->HdrActive = false;
        for (VkPipeline& pipeline : m_Impl->PostPipelines)
        {
            if (pipeline != 0 && m_Impl->DestroyPipelineFunction)
                m_Impl->DestroyPipelineFunction(m_Impl->Device, pipeline, nullptr);
            pipeline = 0;
        }
        for (auto& modules : m_Impl->PostModules)
            for (VkShaderModule& module : modules)
            {
                if (module != 0 && m_Impl->DestroyShaderModuleFunction)
                    m_Impl->DestroyShaderModuleFunction(m_Impl->Device, module, nullptr);
                module = 0;
            }
        if (m_Impl->PostRenderPass != 0 && m_Impl->DestroyRenderPassFunction)
            m_Impl->DestroyRenderPassFunction(m_Impl->Device, m_Impl->PostRenderPass, nullptr);
        m_Impl->PostRenderPass = 0;
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
        if (m_Impl->AnisotropicSampler && m_Impl->DestroySamplerFunction)
            m_Impl->DestroySamplerFunction(m_Impl->Device, m_Impl->AnisotropicSampler, nullptr);
        if (m_Impl->ShadowSampler != 0 && m_Impl->DestroySamplerFunction)
            m_Impl->DestroySamplerFunction(m_Impl->Device, m_Impl->ShadowSampler, nullptr);
        if (m_Impl->DestroyShaderModuleFunction)
        {
            for (auto& module : m_Impl->ShadowModules)
            {
                if (module) m_Impl->DestroyShaderModuleFunction(m_Impl->Device, module, nullptr);
                module = 0;
            }
            for (auto& modules : m_Impl->TechniqueModules)
                for (VkShaderModule& module : modules)
                {
                    if (module != 0) m_Impl->DestroyShaderModuleFunction(m_Impl->Device, module, nullptr);
                    module = 0;
                }
        }
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
        m_Impl->DestroyFrameTiming();
        DestroyVulkanPipelineCache(m_Impl->Device);
        if (m_Impl->DestroyDeviceFunction)
            m_Impl->CompleteGpuSignal();
            for(auto& item:m_Impl->GpuSurfaces)m_Impl->DestroyRawImage(item.Image);
            m_Impl->GpuSurfaces.clear();
            m_Impl->DestroyDeviceFunction(m_Impl->Device, nullptr);
    }
    m_Impl->Buffers.clear();
    m_Impl->Textures.clear();
    m_Impl->Pipelines.clear();
    m_Impl->FrameImageLayouts.clear();
    m_Impl->FrameSubmitted = false;
    m_Impl->FrameCommandEnded = false;
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
    m_Impl->ShadowSampler = m_Impl->AnisotropicSampler = 0;
    m_Impl->TechniqueModules = {};
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
    result.MaterialAnisotropicFiltering = m_Impl->Initialized && m_Impl->AnisotropicSampler != 0;
    result.TemporalEffects=m_Impl->Initialized && m_Impl->SupportsTemporalExposure();
    result.InstancedRendering = m_Impl->Initialized &&
        m_Impl->DeviceLimits.maxUniformBufferRange >= sizeof(InstanceConstantBlock);
    result.GPUDrivenRendering = result.InstancedRendering && m_Impl->GraphicsQueueHasCompute;
    result.VisibilityBufferRendering = m_Impl->Initialized && m_Impl->SupportsVisibilityBuffer();
    result.Wireframe = true;
    result.ShadowMaps = true;
    result.VarianceShadowMaps = m_Impl->Initialized && m_Impl->VSMSupported;
    result.ContactShadows = m_Impl->Initialized && m_Impl->ContactSupported;
    result.ForwardPlusRendering = m_Impl->Initialized && m_Impl->ForwardPlusSupported;
    result.ClusteredForwardRendering = result.ForwardPlusRendering && m_Impl->DeviceLimits.maxUniformBufferRange >= sizeof(ClusteredForwardConstants);
    result.DeferredShading = m_Impl->Initialized && m_Impl->DeferredSupported;
    result.ParallaxOcclusion=m_Impl->Initialized && result.DeferredShading;
    result.GeometryDisplacement=m_Impl->Initialized && m_Impl->SupportsDisplacement();
        result.GeometryTessellation=m_Impl->Initialized && m_Impl->SupportsTessellation();
        result.GeometryVirtualization=m_Impl->Initialized && m_Impl->SupportsVirtualGeometry();
    result.NaniteRendering=result.VisibilityBufferRendering && result.GeometryVirtualization;
    result.MeshShaders=m_Impl->Initialized && m_Impl->MeshStagesEnabled;
    result.NaniteHardwareScheduling=result.NaniteRendering && m_Impl->FragmentStoresAndAtomicsSupported && m_Impl->DeviceLimits.maxPerStageDescriptorStorageBuffers>=4 && m_Impl->DeviceLimits.maxDescriptorSetStorageBuffers>=4;
        result.VirtualShadowMaps=m_Impl->Initialized && m_Impl->GraphicsQueueHasCompute &&
            m_Impl->DeviceLimits.maxPerStageDescriptorStorageBuffers>=2 &&
            m_Impl->DeviceLimits.maxStorageBufferRange>=VirtualShadowMaximumReceiverPixels*sizeof(float) &&
            m_Impl->DeviceLimits.maxUniformBufferRange>=sizeof(VirtualShadowRequestConstants) &&
            m_Impl->DeviceLimits.maxComputeWorkGroupInvocations>=VirtualShadowGroupWidth &&
            m_Impl->DeviceLimits.maxComputeWorkGroupSize[0]>=VirtualShadowGroupWidth;
    result.TiledDeferredRendering = result.DeferredShading && result.ForwardPlusRendering;
    result.ScalableAmbientObscurance = m_Impl->Initialized && m_Impl->SaoSupported;
    result.TemporalAntiAliasing = m_Impl->Initialized && m_Impl->TemporalSupported;
    result.TemporalSuperResolution=result.TemporalAntiAliasing && m_Impl->GraphicsQueueHasCompute && m_Impl->DeviceLimits.maxPerStageDescriptorStorageBuffers>=4 && m_Impl->DeviceLimits.maxPerStageDescriptorSampledImages>=TSRSceneImageCount && m_Impl->DeviceLimits.maxComputeWorkGroupInvocations>=TSRGroupSide*TSRGroupSide && m_Impl->DeviceLimits.maxComputeWorkGroupSize[0]>=TSRGroupSide && m_Impl->DeviceLimits.maxComputeWorkGroupSize[1]>=TSRGroupSide;
    result.ScreenSpaceGlobalIllumination = result.DeferredShading && result.TemporalAntiAliasing;
    result.BakedLightProbes = result.DeferredShading;
        result.BakedLightmaps = result.DeferredShading;
        result.VoxelGI = result.DeferredShading && m_Impl->Initialized && m_Impl->SupportsVoxelGI();
        result.LumenGI=result.DeferredShading && m_Impl->Initialized && m_Impl->DeviceLimits.maxPerStageDescriptorStorageBuffers>=4;
        result.LumenHardwareRayTracing=result.LumenGI && m_Impl->LumenHardwareEnabled;
    result.Refraction = result.DeferredShading;
        result.RayTracedAmbientOcclusion = result.DeferredShading &&
        m_Impl->DeviceLimits.maxPerStageDescriptorStorageBuffers >= RaySceneBufferCount &&
        m_Impl->DeviceLimits.maxDescriptorSetStorageBuffers >= RaySceneBufferCount &&
        m_Impl->DeviceLimits.maxFragmentCombinedOutputResources >= RaySceneBufferCount+1;
    result.RayTracedReflections = result.DeferredShading &&
        m_Impl->DeviceLimits.maxPerStageDescriptorStorageBuffers >= Impl::RayReflectionStorageCount &&
        m_Impl->DeviceLimits.maxDescriptorSetStorageBuffers >= Impl::RayReflectionStorageCount &&
        m_Impl->DeviceLimits.maxFragmentCombinedOutputResources >= Impl::RayReflectionStorageCount+1;
        result.HybridRendering = result.DeferredShading && result.RayTracedReflections;
    result.RayTracedGI = result.RayTracedReflections;
    result.PathTracing = result.RayTracedReflections && m_Impl->SupportsPathTracing();
    result.VolumetricFog = m_Impl->Initialized && m_Impl->FogSupported;
    result.VolumetricLighting = m_Impl->Initialized && m_Impl->VolumeLightSupported;
        result.DepthPeeling = result.Refraction && result.VolumetricLighting;
    result.LightShafts = m_Impl->Initialized && m_Impl->LightShaftSupported;
    result.WeightedTransparencySampleCounts = m_Impl->Initialized ? m_Impl->WeightedSampleCounts : 0;
    result.SupportedSampleCounts = m_Impl->Initialized ? m_Impl->SupportedSampleCounts : 1;
    result.Outline = true;
    result.AsyncCommandRecording = false;
    result.MaximumTextureSize = m_Impl->Initialized ? m_Impl->DeviceLimits.maxImageDimension2D : 0U;
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
    m_Impl->Hair.ResetHistory();
    m_Impl->ExposureHistory.Invalidate();
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

bool VulkanRHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& candidate,VirtualGeometryCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache updates require a closed frame");
    return m_Impl->PrepareVirtualGeometryCache(upload,previous,candidate,statistics);
}
bool VulkanRHI::SubmitNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheTask& task,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Submit Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->SubmitNaniteCache(upload,previous,task,statistics);
}
bool VulkanRHI::PollNaniteCache(NaniteCacheTask task,bool& ready,NaniteCacheHandle& output)
{
    if (!m_Impl->CheckThread("Poll Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->PollNaniteCache(task,ready,output);
}
bool VulkanRHI::CancelNaniteCache(NaniteCacheTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite page copy")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite page copies require an initialized RHI between frames");
    return m_Impl->CancelNaniteCache(task);
}
bool VulkanRHI::PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& candidate,NaniteCacheStatistics& statistics)
{
    if (!m_Impl->CheckThread("Prepare Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache updates require a closed frame");
    return m_Impl->PrepareNaniteCache(upload,previous,candidate,statistics);
}
bool VulkanRHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy virtual geometry cache")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual cache release requires a closed frame");
    return m_Impl->DestroyVirtualGeometryCache(handle);
}
bool VulkanRHI::DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!m_Impl->CheckThread("Destroy Nanite geometry cache")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite cache release requires a closed frame");
    return m_Impl->DestroyNaniteCache(handle);
}
bool VulkanRHI::SnapshotTSROpaque()
{
    if (!m_Impl->CheckThread("Snapshot TSR opaque")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || m_Impl->PassOpen) return m_Impl->Fail("TSR opaque snapshot requires a closed pass in an initialized frame");
    return m_Impl->SnapshotTSROpaque();
}
bool VulkanRHI::RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& settings,FrameCapture& output)
{
    if (!m_Impl->CheckThread("Render TSR display")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR display work requires an initialized RHI and closed frame");
    return m_Impl->RenderTSRDisplay(handle,settings,output);
}
bool VulkanRHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    if (!m_Impl->CheckThread("BuildVirtualShadowReceiverRequests")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow receivers require an initialized RHI and closed frame");
    return m_Impl->BuildVirtualShadowReceiverRequests(upload,output);
}
bool VulkanRHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    if (!m_Impl->CheckThread("SampleVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow lookup requires an initialized RHI and closed frame");
    return m_Impl->SampleVirtualShadowAtlas(handle,upload,output);
}
bool VulkanRHI::CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!m_Impl->CheckThread("CreateLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->CreateLumenDistanceScene(data,output);
}
bool VulkanRHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    if (!m_Impl->CheckThread("CreateLumenGlobalDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CreateLumenGlobalDistanceScene(source,settings,output,previous);
}

bool VulkanRHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenGlobalDistanceUpdate(source,settings,previous,task);
}

bool VulkanRHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenGlobalDistanceUpdate(task,budget,ready,output,statistics);
}

bool VulkanRHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{
    if (!m_Impl->CheckThread("CancelLumenGlobalDistanceUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Global distance composition requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenGlobalDistanceUpdate(task);
}
bool VulkanRHI::TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->TraceLumenDistanceScene(handle,upload,output);
}
bool VulkanRHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenDistanceScene")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen distance operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenDistanceScene(handle);
}
bool VulkanRHI::RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    if (!m_Impl->CheckThread("RelightLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->RelightLumenSurface(surface,distance,upload,output);
}

bool VulkanRHI::BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenLightingUpdate(surface,distance,upload,task);
}

bool VulkanRHI::AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenLightingUpdate(task,budget,ready,output,statistics);
}

bool VulkanRHI::CancelLumenLightingUpdate(LumenLightingTask task)
{
    if (!m_Impl->CheckThread("CancelLumenLightingUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenLightingUpdate(task);
}
bool VulkanRHI::ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenLighting(handle,output);
}
bool VulkanRHI::DestroyLumenLighting(LumenLightingHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenLighting")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen lighting requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenLighting(handle);
}
bool VulkanRHI::TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("TraceLumenRadiance")) return false;
    if (upload.Probes.DirectionalSide && m_Impl->FrameOpen) return m_Impl->Fail("Directional world cache generation requires a closed frame");
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->TraceLumenRadiance(lighting,upload,output);
}
bool VulkanRHI::AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!m_Impl->CheckThread("AssembleLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("World radiance assembly requires an initialized RHI and closed frame");
    if (!m_Impl->ValidateLumenRecording(false)) return false;
    return m_Impl->AssembleLumenRadiance(assembly,output);
}
bool VulkanRHI::ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbeOrigins")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbeOrigins(handle,output);
}
bool VulkanRHI::ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    if (!m_Impl->CheckThread("ReadLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->ReadLumenRadiance(handle,output);
}
bool VulkanRHI::DestroyLumenRadiance(LumenRadianceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenRadiance")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen radiance requires an initialized RHI and closed frame");
    return m_Impl->DestroyLumenRadiance(handle);
}
bool VulkanRHI::IntegrateLumenProbes(LumenRadianceHandle source,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    if (!m_Impl->CheckThread("IntegrateLumenProbes")) return false;
    if (!IsInitialized()) return m_Impl->Fail("Lumen requires an initialized RHI");
    if (!m_Impl->ValidateLumenRecording(upload.Readback)) return false;
    return m_Impl->IntegrateLumenProbes(source,upload,output);
}
bool VulkanRHI::ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if (!m_Impl->CheckThread("ResolveLumenProbes")) return false;
    if (!IsInitialized() || !m_Impl->FrameOpen || !m_Impl->ValidateLumenRecording(false)) return m_Impl->Fail("Lumen gather requires an open frame outside a graphics pass");
    return m_Impl->ResolveLumenProbes(handle,settings);
}
bool VulkanRHI::ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    if (!m_Impl->CheckThread("ReadLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->ReadLumenProbes(handle,output);
}
bool VulkanRHI::DestroyLumenProbes(LumenProbeHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenProbes")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen probes require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenProbes(handle);
}
bool VulkanRHI::LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    if (!m_Impl->CheckThread("LookupLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface lookup requires an initialized RHI and closed frame");
    return m_Impl->LookupLumenSurface(handle,upload,output);
}
bool VulkanRHI::CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    if (!m_Impl->CheckThread("CaptureLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->CaptureLumenSurface(upload,output);
}
bool VulkanRHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& task)
{
    if (!m_Impl->CheckThread("BeginLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->BeginLumenCaptureUpdate(upload,task);
}
bool VulkanRHI::AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    ready=false;
    if (!m_Impl->CheckThread("AdvanceLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->AdvanceLumenCaptureUpdate(task,budget,ready,output,statistics);
}
bool VulkanRHI::CancelLumenCaptureUpdate(LumenCaptureTask task)
{
    if (!m_Impl->CheckThread("CancelLumenCaptureUpdate")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen capture requires an initialized RHI and closed frame");
    return m_Impl->CancelLumenCaptureUpdate(task);
}
bool VulkanRHI::ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    if (!m_Impl->CheckThread("ReadLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->ReadLumenSurface(handle,output);
}
bool VulkanRHI::DestroyLumenSurface(LumenSurfaceHandle handle)
{
    if (!m_Impl->CheckThread("DestroyLumenSurface")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Lumen surface operations require an initialized RHI and closed frame");
    return m_Impl->DestroyLumenSurface(handle);
}
bool VulkanRHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    if (!m_Impl->CheckThread("RenderVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->RenderVirtualShadowAtlas(upload,previous,output);
}
bool VulkanRHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->ReadVirtualShadowAtlas(handle,output);
}
bool VulkanRHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{
    if (!m_Impl->CheckThread("DestroyVirtualShadowAtlas")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Virtual shadow atlas work requires an initialized RHI and closed frame");
    return m_Impl->DestroyVirtualShadowAtlas(handle);
}
bool VulkanRHI::DestroyTSRHistory(TSRHistoryHandle handle)
{
    if (!m_Impl->CheckThread("Destroy TSR history")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("TSR history release requires an initialized RHI and closed frame");
    return m_Impl->DestroyTSRHistory(handle);
}
bool VulkanRHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle handle,const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached virtual geometry")) return false;
    if (m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateVirtualGeometry(upload,output,handle);
}
bool VulkanRHI::SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& task)
{
    if (!m_Impl->CheckThread("Submit Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->SubmitNaniteGeometry(cache,upload,task);
}
bool VulkanRHI::PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Poll Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->PollNaniteGeometry(task,ready,output);
}
bool VulkanRHI::CancelNaniteGeometry(NaniteGeometryTask task)
{
    if (!m_Impl->CheckThread("Cancel Nanite geometry task")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite production requires an initialized RHI between frames");
    return m_Impl->CancelNaniteGeometry(task);
}
bool VulkanRHI::CreateCachedNaniteGeometry(NaniteCacheHandle handle,const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create cached Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen || !handle) return m_Impl->Fail("Cached geometry requires a closed frame and valid cache handle");
    return m_Impl->CreateNaniteGeometry(upload,output,handle);
}

bool VulkanRHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create virtual geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Virtual geometry must be produced between frames");
    return m_Impl->CreateVirtualGeometry(upload,output);
}
bool VulkanRHI::PrepareNaniteDraws(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDraws") && m_Impl->PrepareNaniteDraws(visibility,draws,output);
}
bool VulkanRHI::PrepareNaniteDrawBins(NaniteVisibilityHandle visibility,const std::vector<NaniteClusterDraw>& draws,NaniteDrawHandle& output)
{
    return m_Impl->CheckThread("PrepareNaniteDrawBins") && m_Impl->PrepareNaniteDraws(visibility,draws,output,true);
}
bool VulkanRHI::DrawNaniteBins(NaniteDrawHandle handle)
{
    if (!m_Impl->CheckThread("DrawNaniteBins") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->NaniteCommandsEnded() ||
        m_Impl->CurrentPass.Target || m_Impl->CurrentPass.DepthPeelingLayer ||
        (m_Impl->CurrentPass.Type!=PassType::Color && m_Impl->CurrentPass.Type!=PassType::GBuffer)) return m_Impl->Fail("Nanite bins require an open primary opaque graphics pass");
    const auto found=m_Impl->NaniteDrawBatches.find(handle.Value);
    if (found==m_Impl->NaniteDrawBatches.end() || !found->second->Binning ||
        std::any_of(found->second->Submitted.begin(),found->second->Submitted.end(),[](bool submitted){return submitted;}))
        return m_Impl->Fail("Nanite bin handle is invalid or already consumed");
    auto& record=*found->second;
    if (record.Phase==NaniteVisibilityPhase::Post && m_Impl->CurrentPass.ClearDepthEnabled) return m_Impl->Fail("Nanite post bins must preserve main depth");
    for (unsigned id=0;id<record.Binning->Bins.size();++id)
    {
        const auto& bin=record.Binning->Bins[id]; const auto& draw=(*record.Draws)[bin.FirstDraw];
        if (!BindPipeline(draw.Pipeline) || !BindWorldTransform(draw.World) || !BindViewProjection(draw.ViewProjection) ||
            !BindLightViewProjection(draw.LightViewProjection) || !BindVertexBuffer(draw.VertexBuffer) || !BindIndexBuffer(draw.IndexBuffer)) return false;
        m_Impl->ActiveNaniteDraw=&record; m_Impl->ActiveNaniteQuery=bin.FirstDraw; m_Impl->ActiveNaniteBin=id;
        struct Reset { Impl& Owner; ~Reset() { Owner.ActiveNaniteDraw=nullptr; } } reset{*m_Impl};
        if (!DrawIndexedGeometry(bin.FirstIndex,bin.IndexCount,0,nullptr)) return false;
        std::fill(record.Submitted.begin()+bin.FirstDraw,record.Submitted.begin()+bin.FirstDraw+bin.DrawCount,true);
    }
    return true;
}
bool VulkanRHI::DrawNaniteCluster(NaniteDrawHandle handle,unsigned query)
{
    if (!m_Impl->CheckThread("DrawNaniteCluster") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->NaniteCommandsEnded() ||
        m_Impl->CurrentPass.Target || m_Impl->CurrentPass.DepthPeelingLayer ||
        (m_Impl->CurrentPass.Type!=PassType::Color && m_Impl->CurrentPass.Type!=PassType::GBuffer)) return m_Impl->Fail("Nanite draw requires an open primary opaque graphics pass");
    const auto found=m_Impl->NaniteDrawBatches.find(handle.Value);
    if (found==m_Impl->NaniteDrawBatches.end() || found->second->Binning || query>=found->second->Draws->size() || found->second->Submitted[query])
        return m_Impl->Fail("Nanite draw handle, query index or duplicate submission is invalid");
    auto& record=*found->second;
    if (record.Phase==NaniteVisibilityPhase::Post && m_Impl->CurrentPass.ClearDepthEnabled)
        return m_Impl->Fail("Nanite post pass must preserve main depth");
    const auto& draw=(*record.Draws)[query];
    if (!BindPipeline(draw.Pipeline) || !BindWorldTransform(draw.World) || !BindViewProjection(draw.ViewProjection) ||
        !BindLightViewProjection(draw.LightViewProjection) || !BindVertexBuffer(draw.VertexBuffer) || !BindIndexBuffer(draw.IndexBuffer)) return false;
    m_Impl->ActiveNaniteDraw=&record; m_Impl->ActiveNaniteQuery=query;
    struct Reset { Impl& Owner; ~Reset() { Owner.ActiveNaniteDraw=nullptr; } } reset{*m_Impl};
    if (!DrawIndexedGeometry(draw.FirstIndex,draw.IndexCount,0,nullptr)) return false;
    record.Submitted[query]=true; return true;
}
bool VulkanRHI::CaptureNaniteDepth(NaniteDepthHandle& output)
{
    if (!m_Impl->CheckThread("CaptureNaniteDepth") || !IsInitialized()) return false;
    return m_Impl->CaptureNaniteDepth(output);
}
bool VulkanRHI::BeginNaniteVisibility(NaniteDepthHandle history,const std::vector<NaniteVisibilityQuery>& queries,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("BeginNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->BeginNaniteVisibility(history,queries,output);
}
bool VulkanRHI::RetestNaniteVisibility(NaniteVisibilityHandle main,NaniteDepthHandle depth,NaniteVisibilityHandle& output)
{
    if (!m_Impl->CheckThread("RetestNaniteVisibility") || !IsInitialized()) return false;
    return m_Impl->RetestNaniteVisibility(main,depth,output);
}
bool VulkanRHI::ReadNaniteDepth(NaniteDepthHandle handle,std::vector<float>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(found->second->Data,words) && ReadNaniteDepthWords(found->second->Info,words,output,m_Impl->LastError);
}
bool VulkanRHI::ReadNaniteVisibility(NaniteVisibilityHandle handle,std::vector<NaniteVisibilityState>& output)
{
    if (!m_Impl->CheckThread("ReadNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    std::vector<unsigned> words; return m_Impl->ReadNaniteFrameBuffer(found->second->Data,words) && ReadNaniteDecisionWords(found->second->Info,words,output,m_Impl->LastError);
}
bool VulkanRHI::DestroyNaniteDepth(NaniteDepthHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteDepth") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDepths.find(handle.Value);
    if (found==m_Impl->NaniteDepths.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDepths.erase(found); return true;
}
bool VulkanRHI::DestroyNaniteVisibility(NaniteVisibilityHandle handle)
{
    if (!m_Impl->CheckThread("DestroyNaniteVisibility") || !IsInitialized()) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Nanite feedback and release require a closed frame");
    const auto found=m_Impl->NaniteDecisions.find(handle.Value);
    if (found==m_Impl->NaniteDecisions.end() || !found->second->Info.Published) return m_Impl->Fail("Nanite handle is invalid or pending");
    m_Impl->NaniteDecisions.erase(found); return true;
}
bool VulkanRHI::CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create Nanite geometry")) return false;
    if (!IsInitialized() || m_Impl->FrameOpen) return m_Impl->Fail("Nanite geometry must be produced between frames");
    return m_Impl->CreateNaniteGeometry(upload,output);
}

bool VulkanRHI::CreateTessellatedGeometry(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    if (!m_Impl->CheckThread("Create tessellated geometry")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Tessellated geometry must be prepared between frames");
    return m_Impl->CreateTessellated(geometry,settings,output);
}

bool VulkanRHI::CreateDisplacedVertexBuffer(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    if (!m_Impl->CheckThread("Create displaced vertex buffer")) return false;
    if (m_Impl->FrameOpen) return m_Impl->Fail("Displaced geometry must be prepared between frames");
    return m_Impl->CreateDisplaced(geometry,settings,output);
}

bool VulkanRHI::UpdateBuffer(BufferHandle buffer, std::size_t offset,
                             const void* data, std::size_t size)
{
    if (!m_Impl->CheckThread("UpdateBuffer")) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
    auto found = m_Impl->Buffers.find(buffer.Value);
    if (found == m_Impl->Buffers.end() || data == nullptr || size == 0 ||
        offset > found->second.Bytes.size() || size > found->second.Bytes.size() - offset)
        return m_Impl->Fail("Vulkan buffer update is invalid");
    if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
    found->second.LastValidRange = {};
    std::memcpy(found->second.Bytes.data() + offset, data, size);
    std::memcpy(static_cast<std::uint8_t*>(found->second.Allocation.Mapped) + offset, data, size);
    return true;
}

bool VulkanRHI::DestroyBuffer(BufferHandle buffer)
{
    if (!m_Impl->CheckThread("DestroyBuffer")) return false;
    if (!m_Impl->RetireLumenCaptureInput(buffer)) return false;
        if (m_Impl->NaniteBufferLocked(buffer)) return m_Impl->Fail("Nanite draw geometry is locked until frame completion");
    auto found = m_Impl->Buffers.find(buffer.Value);
    if (!buffer || found == m_Impl->Buffers.end()) return m_Impl->Fail("Vulkan buffer handle is invalid");
    m_Impl->DestroyRawBuffer(found->second.Allocation);
    m_Impl->Buffers.erase(found);
    return true;
}

TextureHandle VulkanRHI::CreateTexture(const TextureDesc& desc, const void* initialData)
{
    if (!m_Impl->CheckThread("CreateTexture") || m_Impl->FrameOpen ||
        !m_Impl->CheckThread("CreateTexture") || desc.Width <= 0 || desc.Height <= 0 ||
        !ValidTextureMipLevels(desc, initialData))
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
    if (!m_Impl->CheckThread("DestroyTexture")) return false;
    if (!m_Impl->RetireLumenCaptureInput(texture)) return false;
    if (!m_Impl->CheckThread("DestroyTexture") || m_Impl->FrameOpen)
        return m_Impl->Fail("Vulkan texture cannot be destroyed in the current state");
    auto found = m_Impl->Textures.find(texture.Value);
    if (!texture || found == m_Impl->Textures.end()) return m_Impl->Fail("Vulkan texture handle is invalid");
    const auto moments = m_Impl->VSMMoments.find(texture.Value);
    if (moments != m_Impl->VSMMoments.end())
    { m_Impl->DestroyMomentMap(moments->second); m_Impl->VSMMoments.erase(moments); }
    m_Impl->DestroyTextureResource(found->second);
    m_Impl->Textures.erase(found);
    if (m_Impl->LastShadowTexture == texture) m_Impl->LastShadowTexture = {};
    return true;
}

bool VulkanRHI::BeginFrame()
{
    if (!m_Impl->CheckThread("BeginFrame") || m_Impl->FrameOpen)
        return m_Impl->Fail("Vulkan frame cannot begin in the current state");
    if (!m_Impl->CollectLumenGlobalDistance() || !m_Impl->CollectLumenLighting() || !m_Impl->CollectLumenCapture()) return false;
    ResetMotionTransform();
    m_Impl->ExposureHistory.Begin();
    if (!m_Impl->Check(m_Impl->WaitForFences(m_Impl->Device, 1, &m_Impl->FrameFence,
                                             VK_TRUE_VALUE, GpuWaitTimeout),
                       "vkWaitForFences(frame)") ||
        !m_Impl->Check(m_Impl->ResetCommandPool(m_Impl->Device, m_Impl->CommandPool, 0),
                       "vkResetCommandPool(frame)")) return false;
    if (!m_Impl->PrepareRayShadowImage(GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced?GetRayShadowImage():nullptr)) return false;
    m_Impl->NaniteFrameWork.clear();
    if (!m_Impl->ResetForwardPlusAfterDiscard()) return false;
    m_Impl->LightListFrame.Reset(); m_Impl->ForwardPlusDepthWritten = false;
    m_Impl->TiledActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::TiledDeferred;
        m_Impl->ClusteredActive = GetTechniqueSettings().Pipeline == Techniques::RenderingPipeline::ClusteredForward;
    m_Impl->LightListsActive = Techniques::UsesLightLists(GetTechniqueSettings().Pipeline);
    if (m_Impl->LightListsActive && !m_Impl->ForwardPlusSupported)
        return m_Impl->Fail("Vulkan Forward+ requires compute assignment and complete fragment light storage");
    m_Impl->FrameAreaLights = GetTechniqueSettings().AreaLights.Enabled;
        m_Impl->FrameSampleCount = GetTechniqueSettings().SampleCount;
    m_Impl->DestroyContactTargets();
    if (m_Impl->ContactProducer) m_Impl->ContactProducer->BeginFrame();
    m_Impl->FrameContact = GetTechniqueSettings().ContactShadows;
    m_Impl->ContactDepthWritten = m_Impl->ContactGenerated = m_Impl->ContactReceiverActive = false;
    m_Impl->WeightedActive = Techniques::UsesWeightedTransparency(GetTechniqueSettings());
    m_Impl->FrameTransparency = GetTechniqueSettings().Transparency;
    m_Impl->WeightedPass = m_Impl->WeightedPrepared = m_Impl->WeightedAccumulated = m_Impl->WeightedResolved = false;
    m_Impl->SamplesOpaqueResolved = m_Impl->SamplesVelocityResolved = false;
    m_Impl->SamplesColorRestored = m_Impl->SamplesColorResolved = m_Impl->SamplesTransparentWritten = false;
    m_Impl->SampleDescriptorsWritten.fill(false);
    m_Impl->SubsurfaceProducing=m_Impl->SubsurfaceResolved=m_Impl->SubsurfaceHasScene=false;
    m_Impl->FrameSubsurface=GetTechniqueSettings().Subsurface;
    m_Impl->SubsurfaceRequested=Techniques::UsesSubsurfaceScattering(m_Impl->FrameSubsurface);
    m_Impl->FramePostProcessing = GetTechniqueSettings().PostProcessing;
    if (!m_Impl->PrepareTSRScene(GetTSRSceneFrame(),m_Impl->FramePostProcessing)) return false;
    m_Impl->FrameGI = GetTechniqueSettings().GlobalIllumination;
        m_Impl->FrameLumenLighting=GetLumenLighting(); m_Impl->FrameLumenWorldCache=GetLumenWorldCache(); m_Impl->FrameLumenGlobalDistance=GetLumenGlobalDistance();
    m_Impl->FrameVoxelGI=GetVoxelGIFrame();
    if (!m_Impl->Hair.Freeze(GetHairFurFrame(),GetTechniqueSettings(),m_Impl->LastError)) return false;
        if (Techniques::UsesVoxelGI(m_Impl->FrameGI))
        {
            if (!m_Impl->FrameVoxelGI.Volume) return m_Impl->Fail("Voxel GI requires a current immutable radiance publication");
            m_Impl->FrameVoxelGI.Settings=m_Impl->FrameGI.Voxels; m_Impl->FrameVoxelGI.Intensity=m_Impl->FrameGI.Intensity;
            m_Impl->FrameVoxelGI.MaximumDistance=m_Impl->FrameGI.MaximumDistance; m_Impl->FrameVoxelGI.Shading=GetTechniqueSettings().Shading;
        }
        if (m_Impl->FrameVoxelGI.Volume && ((Techniques::UsesGlobalIllumination(m_Impl->FrameGI) && !Techniques::UsesVoxelGI(m_Impl->FrameGI)) ||
            Techniques::UsesPathTracing(GetTechniqueSettings().Pipeline) ||
            !Techniques::RequiresHDR(m_Impl->FramePostProcessing.HDR) || m_Impl->FrameVoxelGI.Shading!=GetTechniqueSettings().Shading))
            return m_Impl->Fail("Voxel GI requires an exclusive HDR raster frame and matching receiver shading");
        m_Impl->GIActive = Techniques::UsesGlobalIllumination(m_Impl->FrameGI) || bool(m_Impl->FrameVoxelGI.Volume);
    m_Impl->EarlyTemporalActive = Techniques::UsesTemporalGI(m_Impl->FrameGI) || Techniques::UsesTemporalReflections(GetTechniqueSettings().Reflections);
    m_Impl->ProbeActive = Techniques::UsesBakedProbes(m_Impl->FrameGI);
    m_Impl->FrameProbeSamples = GetBakedProbeSamples();
    m_Impl->LightmapActive = Techniques::UsesLightmaps(m_Impl->FrameGI);
    m_Impl->FrameLightmaps = GetLightmapFrame();
    m_Impl->FramePeeling=GetDepthPeelingFrame(); m_Impl->PeelingSequence.Abort(); m_Impl->PeelingDrawBindings.clear();
    m_Impl->FrameRefraction = GetRefractionFrame();
    m_Impl->FrameRefractionSettings = GetTechniqueSettings().Refraction;
    m_Impl->RefractionActive = Techniques::UsesRefraction(GetTechniqueSettings());
    m_Impl->RefractionResolved = false;
    m_Impl->FrameLightmapSettings = GetTechniqueSettings();
    m_Impl->PendingUploadedLightmap.reset();
    m_Impl->GI.Begin(); m_Impl->Probes.Begin();
    m_Impl->HdrActive = Techniques::RequiresHDR(m_Impl->FramePostProcessing.HDR);
    m_Impl->DepthOfFieldActive = m_Impl->FramePostProcessing.DepthOfField.Enabled;
    m_Impl->DepthOfFieldResolved = false;
    m_Impl->MotionBlurActive = m_Impl->FramePostProcessing.MotionBlur.Enabled;
    m_Impl->VelocityCaptured = m_Impl->MotionBlurResolved = false;
    m_Impl->TemporalActive = Techniques::UsesTemporalAA(m_Impl->FramePostProcessing);
    m_Impl->VelocityActive = (Techniques::RequiresMotionVectors(m_Impl->FramePostProcessing) || m_Impl->TSRScene.Active) || m_Impl->EarlyTemporalActive;
    m_Impl->TemporalResolved = false;
    m_Impl->FrameFog = GetTechniqueSettings().VolumetricFog;
    m_Impl->FrameVolumeLighting = GetTechniqueSettings().VolumetricLighting;
    m_Impl->FogActive = Techniques::UsesVolumetricFog(m_Impl->FrameFog);
    m_Impl->FogResolved = false;
    m_Impl->FrameLightShafts = GetTechniqueSettings().LightShafts;
    m_Impl->LightShaftActive = Techniques::UsesLightShafts(m_Impl->FrameLightShafts);
    m_Impl->LightShaftResolved = false;
    if (m_Impl->HdrActive && !m_Impl->EnsureHdrTargets()) return false;
    if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI) && !m_Impl->EnsureGIResources()) return false;
    if (m_Impl->ProbeActive && !m_Impl->EnsureProbeResources()) return false;
    if (m_Impl->LightShaftActive && !m_Impl->EnsureLightShaftResources()) return false;
    if (m_Impl->FogActive && !m_Impl->EnsureFogResources()) return false;
    if (m_Impl->FogActive && (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog)) && !m_Impl->EnsureVolumeLightResources()) return false;
    m_Impl->DeferredActive = Techniques::UsesDeferredLighting(GetTechniqueSettings().Pipeline);
    m_Impl->FrameAmbientOcclusion = GetTechniqueSettings().AmbientOcclusion;
    m_Impl->FrameRayScene = GetRayScene();
    m_Impl->FrameRayRadiance = GetRayRadianceFrame();
    m_Impl->FrameRayRadianceSettings = GetTechniqueSettings();
        m_Impl->PathFrame.BeginFrame(Techniques::UsesPathTracing(GetTechniqueSettings().Pipeline));
    m_Impl->FrameReflections = GetTechniqueSettings().Reflections;
    m_Impl->FrameReflectionShading = GetTechniqueSettings().Shading;
    m_Impl->AnisotropicFiltering = GetTechniqueSettings().AnisotropicFiltering;
    if (!m_Impl->PrepareEnvironmentCapture(GetEnvironmentCaptureFrame())) return false;
    if (!m_Impl->PreparePlanarCapture(GetPlanarCaptureFrame())) return false;
    m_Impl->ReflectionActive = Techniques::UsesReflections(m_Impl->FrameReflections);
    m_Impl->ReflectionResolved = false;
    m_Impl->AmbientOcclusionActive = Techniques::UsesAmbientOcclusion(m_Impl->FrameAmbientOcclusion);
    m_Impl->AmbientOcclusionResolved = m_Impl->ForwardOpaqueComplete = false;
    m_Impl->GBufferWritten = m_Impl->DeferredResolved = false;
    m_Impl->DeferredBackground = {};
    if ((m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive) && !m_Impl->EnsureDeferredResources()) return false;
    if (m_Impl->AmbientOcclusionActive && !m_Impl->EnsureAmbientOcclusionResources()) return false;
    if (m_Impl->ReflectionActive && !m_Impl->EnsureReflectionResources()) return false;
    if (Techniques::UsesRayTracedGI(m_Impl->FrameGI) && !m_Impl->EnsureRayGIResources()) return false;
    if (m_Impl->FrameVoxelGI.Volume && !m_Impl->EnsureVoxelGIResources()) return false;
    if (m_Impl->PathFrame.Active && !m_Impl->EnsurePathTracing()) return false;
    if (m_Impl->DepthOfFieldActive && !m_Impl->EnsureDepthOfFieldResources()) return false;
    if ((m_Impl->VelocityActive && !m_Impl->EnsureVelocityResources()) ||
        (m_Impl->TemporalActive && !m_Impl->EnsureTemporalResources()) ||
        (m_Impl->MotionBlurActive && !m_Impl->EnsureMotionResources())) return false;
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->EnsureMultisampleTargets()) return false;
    if (m_Impl->LightmapActive && !m_Impl->EnsureLightmapResources()) return false;
    if (m_Impl->FramePeeling.Enabled && !m_Impl->EnsurePeelingResources()) return false;
    if (m_Impl->RefractionActive && !m_Impl->EnsureRefractionResources()) return false;
    if ((m_Impl->FrameContact.Enabled || (m_Impl->LightListsActive && !m_Impl->ClusteredActive)) && !m_Impl->EnsureCameraVisibilityTargets()) return false;
    if (m_Impl->WeightedActive && !m_Impl->EnsureWeightedResources()) return false;
    if (m_Impl->Hair.Active && !m_Impl->EnsureHairFur()) return false;
    m_Impl->UniformOffset = 0;
    if (!m_Impl->WriteFogSet(Impl::FogDisabledSet, {},
        {m_Impl->DummyColor.View, m_Impl->DummyColor.View, m_Impl->DummyColor.View})) return false;
    m_Impl->FrameShadowFilter = GetTechniqueSettings().ShadowFilter;
    if (GetTechniqueSettings().Shadows==Techniques::ShadowRepresentation::RayTraced) m_Impl->FrameShadowFilter.Mode=Techniques::ShadowFilterMode::Hard;
    for (auto& entry : m_Impl->VSMMoments) entry.second.Ready = entry.second.RawWritten = false;
    if (!m_Impl->PrepareVirtualShadowSampling(GetVirtualShadowSamplingFrame(),GetCascadedShadowState()) ||
            !m_Impl->PrepareCascadedShadows(GetCascadedShadowState())) return false;
    m_Impl->InstancePageIndex = m_Impl->InstancePageOffset = 0;
    m_Impl->GpuCursor = 0; m_Impl->ActiveGpuBatch = nullptr; ++m_Impl->GpuEpoch;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!m_Impl->Check(m_Impl->BeginCommandBuffer(m_Impl->CommandBuffer, &begin),
                       "vkBeginCommandBuffer(frame)")) return false;
    m_Impl->PendingFrame = {};
    m_Impl->PendingFrame.Backend = RHIBackend::Vulkan;
    m_Impl->BeginFrameTiming();
    m_Impl->PendingFrame.Statistics.FrameNumber = m_Impl->FrameNumber + 1;
    m_Impl->PendingFrame.Statistics.RasterSampleCount = m_Impl->FrameSampleCount;
    m_Impl->DescriptorSetIndex = 0;
    m_Impl->LastShadowTexture = {};
    m_Impl->FrameImageLayouts.clear();
    m_Impl->FrameSubmitted = false;
    m_Impl->FrameCommandEnded = false;
    m_Impl->FrameOpen = true;
    m_Impl->LastError.clear();
    if (m_Impl->FramePostProcessing.TemporalEffects.Enabled && !m_Impl->EnsureExposure()) return false;
    return true;
}

void VulkanRHI::AbortFrame()
{
    m_Impl->PeelingSequence.Abort(); m_Impl->FramePeeling={}; m_Impl->PeelingDrawBindings.clear();
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
        m_Impl->PathFrame.Abort();
    if (m_Impl->FrameOpen && !m_Impl->FrameSubmitted && !m_Impl->FrameCommandEnded)
    {
        if (m_Impl->PassOpen)
        {
            m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
        }
        (void)m_Impl->EndCommandBuffer(m_Impl->CommandBuffer);
    }
    if (!m_Impl->FrameSubmitted)
    {
        // Resetting discards commands before a producer may resize its buffers.
        // A failed reset leaves ownership intact for the next successful wait/reset.
        if (m_Impl->ForwardPlusRecordOpen &&
            m_Impl->Check(m_Impl->ResetCommandPool(m_Impl->Device, m_Impl->CommandPool, 0), "vkResetCommandPool(Forward+ abort)"))
            (void)m_Impl->ResetForwardPlusAfterDiscard();
        if (m_Impl->ContactProducer) m_Impl->ContactProducer->AbortFrame();
        for (const auto& entry : m_Impl->FrameImageLayouts)
            entry.first->Layout = entry.second;
        m_Impl->CompleteGpuSignal();
    }
    else if (m_Impl->ContactProducer) m_Impl->ContactProducer->BeginFrame();
    m_Impl->LightListFrame.Reset(); m_Impl->LightListsActive = m_Impl->ForwardPlusDepthWritten = false;
    m_Impl->ContactDepthWritten = m_Impl->ContactGenerated = m_Impl->ContactReceiverActive = false;
    m_Impl->FrameContact = {};
    m_Impl->FrameImageLayouts.clear();
    m_Impl->PendingFrame = {};
    for (auto& entry : m_Impl->VSMMoments) entry.second.Ready = entry.second.RawWritten = false;
    m_Impl->FrameCascades = {};
    m_Impl->CascadeImages.fill(nullptr);
    SetCascadedShadowState({});
    m_Impl->CurrentPass = {};
    m_Impl->BoundPipeline = {};
    m_Impl->BoundVertexBuffer = {};
    m_Impl->BoundIndexBuffer = {};
    m_Impl->LastShadowTexture = {};
    m_Impl->UniformOffset = 0;
    m_Impl->DescriptorSetIndex = 0;
    m_Impl->AbortTSRScene();
    m_Impl->FrameOpen = false;
    m_Impl->FinishLumenFrame(false); m_Impl->FinishNaniteFrame(false);
    m_Impl->PassOpen = false;
    m_Impl->PipelineBound = false;
    m_Impl->WorldBound = false;
    m_Impl->ViewProjectionBound = false;
    m_Impl->LightViewProjectionBound = false;
    m_Impl->AmbientOcclusionResolved = m_Impl->ForwardOpaqueComplete = false;
    m_Impl->GBufferWritten = m_Impl->DeferredResolved = false;
    m_Impl->ReflectionActive = m_Impl->ReflectionResolved = false;
    m_Impl->FrameReflections = {};
    m_Impl->FrameEnvironment = {}; m_Impl->EnvironmentFacesWritten = 0;
    m_Impl->EnvironmentActive = m_Impl->EnvironmentReady = false;
    SetEnvironmentCaptureFrame({});
    m_Impl->FramePlanar = {}; m_Impl->PlanarActive = m_Impl->PlanarWritten = m_Impl->PlanarReady = false;
    SetPlanarCaptureFrame({});
    m_Impl->DepthOfFieldActive = m_Impl->DepthOfFieldResolved = false;
    m_Impl->MotionBlurActive = m_Impl->VelocityCaptured = m_Impl->MotionBlurResolved = false;
    m_Impl->VelocityActive = m_Impl->TemporalActive = m_Impl->TemporalResolved = false;
    m_Impl->GI.Begin(); m_Impl->Probes.Invalidate(); m_Impl->GIActive = m_Impl->EarlyTemporalActive = m_Impl->ProbeActive = false;
    if (m_Impl->PendingUploadedLightmap) m_Impl->UploadedLightmap.reset();
    m_Impl->PendingUploadedLightmap.reset();
    m_Impl->FogActive = m_Impl->FogResolved = false;
    m_Impl->LightShaftActive = m_Impl->LightShaftResolved = false;
    m_Impl->FrameLightShafts = {};
    m_Impl->FrameFog = {};
    m_Impl->FrameVolumeLighting = {};
    m_Impl->FrameTransparency = {};
    m_Impl->WeightedActive = m_Impl->WeightedPass = m_Impl->WeightedPrepared = m_Impl->WeightedAccumulated = m_Impl->WeightedResolved = false;
    m_Impl->SamplesOpaqueResolved = m_Impl->SamplesVelocityResolved = false;
    m_Impl->SamplesColorRestored = m_Impl->SamplesColorResolved = m_Impl->SamplesTransparentWritten = false;
    m_Impl->SampleDescriptorsWritten.fill(false);
}

bool VulkanRHI::BeginPass(const PassDesc& desc)
{
    if (!m_Impl->CheckThread("TSR scene boundary")) return false;
    if (m_Impl->TSRScene.Active && m_Impl->TSRScene.OpaqueReady && !desc.Target && (desc.Type==PassType::GBuffer || (desc.Type==PassType::Color && !desc.DepthPeelingLayer && desc.ClearColorEnabled))) return m_Impl->Fail("Opaque scene writes cannot restart after the TSR snapshot");

    if (m_Impl->FrameOpen && m_Impl->ExposureHistory.Resolved && desc.Type!=PassType::Outline)
        return m_Impl->Fail("Temporal exposure is the last scene-HDR stage before display resolve");
    if (m_Impl->FrameOpen && m_Impl->Hair.Resolved &&
        ((desc.Type==PassType::Color && !desc.Target && desc.ClearColorEnabled) ||
         (desc.ClearDepthEnabled && desc.Type!=PassType::Shadow && desc.Type!=PassType::EnvironmentCapture && desc.Type!=PassType::PlanarCapture)))
        return m_Impl->Fail("Completed hair color and primary depth cannot be cleared by a later pass");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete() && !desc.Target && (desc.Type==PassType::Velocity || desc.Type==PassType::Outline))
        return m_Impl->Fail("Complete hair rasterization before velocity or outline draws");
    if (m_Impl->FrameOpen && m_Impl->Hair.Resolved && !desc.Target && desc.Type==PassType::GBuffer)
        return m_Impl->Fail("Opaque geometry cannot replace completed hair depth");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete() && !desc.Target &&
        (desc.Type==PassType::Color || desc.Type==PassType::Outline))
        return m_Impl->Fail("Complete subsurface diffusion before transparent or outline composition");
    if (!m_Impl->CheckThread("BeginPass") || !m_Impl->FrameOpen || m_Impl->PassOpen)
        return m_Impl->Fail("Vulkan pass cannot begin in the current state");
        if (m_Impl->PathFrame.Active) return m_Impl->Fail("Path tracing owns primary visibility; raster passes are inactive");
    if (desc.Type == PassType::DeferredLighting)
        return m_Impl->Fail("Use DrawDeferredLighting for Vulkan fullscreen lighting");
    const bool mainColor = desc.Type == PassType::Color && !desc.Target;
    const bool contactDepth = desc.Type == PassType::ContactDepth;
    const bool forwardPlusDepth = desc.Type == PassType::ForwardPlusDepth;
    if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Prepared())
        return m_Impl->Fail("Vulkan Forward+ requires complete scene lights before every frame pass");
    if (forwardPlusDepth && (m_Impl->ClusteredActive || m_Impl->TiledActive || !m_Impl->LightListsActive || desc.Target || desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
        m_Impl->ForwardPlusDepthWritten || m_Impl->GBufferWritten || m_Impl->LightListFrame.Assigned() ||
        m_Impl->ForwardOpaqueComplete || m_Impl->ContactGenerated))
        return m_Impl->Fail("Vulkan Forward+ depth must clear opaque camera depth once before assignment and lighting");
    if (m_Impl->LightListsActive && desc.Type == PassType::Color && !desc.Target &&
        (!m_Impl->LightListFrame.Assigned() || (!m_Impl->ClusteredActive && desc.ClearDepthEnabled)))
        return m_Impl->Fail("Vulkan Forward+ lighting requires completed assignment and preserved camera depth");
    if (contactDepth && (!m_Impl->FrameContact.Enabled || desc.Target || !desc.ClearDepthEnabled ||
        m_Impl->ContactDepthWritten || m_Impl->GBufferWritten || m_Impl->ForwardOpaqueComplete || m_Impl->ContactGenerated))
        return m_Impl->Fail("Vulkan contact depth requires one cleared main-camera prepass before lighting");
    if (mainColor && m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
        return m_Impl->Fail("Vulkan main lighting requires current contact visibility");
    if (!ValidateDepthPeelingPass(desc,m_Impl->FramePeeling,m_Impl->PeelingSequence,m_Impl->LastError)) return false;
    const bool weightedPass = mainColor && m_Impl->WeightedActive && m_Impl->WeightedPrepared && !m_Impl->WeightedResolved;
    if (m_Impl->WeightedActive && mainColor && m_Impl->WeightedResolved)
        return m_Impl->Fail("Vulkan main color cannot be redrawn after weighted composition");
    if (m_Impl->WeightedPrepared && !m_Impl->WeightedResolved &&
        (!weightedPass || m_Impl->WeightedAccumulated || desc.ClearColorEnabled || desc.ClearDepthEnabled))
        return m_Impl->Fail("Vulkan prepared transparency requires one depth-preserving main color pass");
    if (m_Impl->WeightedActive && mainColor && !m_Impl->WeightedPrepared &&
        (m_Impl->DeferredResolved || m_Impl->ForwardOpaqueComplete))
        return m_Impl->Fail("Vulkan prepare weighted statistics before transparent geometry");
    const bool environmentCapture = desc.Type == PassType::EnvironmentCapture;
    const bool planarCapture = desc.Type == PassType::PlanarCapture;
    const bool capture = environmentCapture || planarCapture;
    if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && !desc.Target && !capture && desc.Type != PassType::Shadow &&
        (desc.Type == PassType::GBuffer || desc.ClearDepthEnabled))
        return m_Impl->Fail("Vulkan Forward+ lists require immutable camera depth after assignment");
    if (!desc.Target && !capture && m_Impl->ContactGenerated && desc.ClearDepthEnabled)
        return m_Impl->Fail("Vulkan traced camera depth must be preserved");
    if (environmentCapture && (!m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || desc.Target ||
        desc.EnvironmentFaceIndex >= EnvironmentFaceCount || !desc.ClearColorEnabled || !desc.ClearDepthEnabled ||
        m_Impl->ForwardOpaqueComplete || m_Impl->GBufferWritten || (m_Impl->EnvironmentFacesWritten & (1u << desc.EnvironmentFaceIndex))))
        return m_Impl->Fail("Vulkan capture requires six distinct cleared faces before main geometry");
    if (!environmentCapture && m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady && desc.Type != PassType::Shadow)
        return m_Impl->Fail("Vulkan complete capture and mip generation before main geometry");
    if (planarCapture && (!m_Impl->PlanarActive || m_Impl->PlanarWritten || m_Impl->PlanarReady || desc.Target ||
        !desc.ClearColorEnabled || !desc.ClearDepthEnabled || m_Impl->ForwardOpaqueComplete || m_Impl->GBufferWritten))
        return m_Impl->Fail("Vulkan capture the planar view once with cleared private color/depth before main geometry");
    if (!planarCapture && m_Impl->PlanarActive && !m_Impl->PlanarReady && desc.Type != PassType::Shadow && desc.Type != PassType::EnvironmentCapture)
        return m_Impl->Fail("Vulkan complete planar capture and mips before main geometry");
    const bool velocity = desc.Type == PassType::Velocity;
    if (velocity && !m_Impl->EarlyTemporalActive && m_Impl->LightShaftActive && !m_Impl->LightShaftResolved)
        return m_Impl->Fail("Vulkan light shafts must precede velocity replay");
    if (m_Impl->LightShaftResolved && desc.Type != PassType::Shadow &&
        (desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
        return m_Impl->Fail("Vulkan shaft color and its opaque depth must be preserved");
    if (velocity && !m_Impl->EarlyTemporalActive && m_Impl->FogActive && !m_Impl->FogResolved)
        return m_Impl->Fail("Vulkan fog composition must precede velocity replay");
    if (m_Impl->FogResolved && desc.Type != PassType::Shadow &&
        (desc.ClearDepthEnabled || (desc.Type == PassType::Color && desc.ClearColorEnabled)))
        return m_Impl->Fail("Vulkan fogged color and its opaque depth must be preserved");
    const bool multisample = m_Impl->FrameSampleCount > 1 && desc.Type != PassType::Outline && !capture;
    if (multisample && desc.Target && desc.Type != PassType::Shadow)
        return m_Impl->Fail("Vulkan MSAA requires internal sample-count-matched attachments");
    if (m_Impl->FrameSampleCount > 1 && (velocity || desc.Type == PassType::Outline) && !m_Impl->SamplesOpaqueResolved)
        return m_Impl->Fail("Vulkan resolve opaque sample depth before velocity or outline draws");
    if (multisample && m_Impl->SamplesOpaqueResolved && desc.Type == PassType::Color &&
        (desc.ClearColorEnabled || desc.ClearDepthEnabled || (m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored)))
        return m_Impl->Fail("Vulkan transparency must load processed sample color and preserve opaque depth");
    const bool opaqueComplete = m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete;
    if (velocity && (!m_Impl->VelocityActive || desc.Target || !desc.ClearColorEnabled || desc.ClearDepthEnabled ||
        !opaqueComplete || m_Impl->VelocityCaptured || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
        (!m_Impl->EarlyTemporalActive && m_Impl->ReflectionActive && !m_Impl->ReflectionResolved)))
        return m_Impl->Fail("Vulkan velocity requires opaque lighting/AO and a fresh internal target, once before blur");
    if (!desc.Target && m_Impl->VelocityCaptured && desc.ClearDepthEnabled)
        return m_Impl->Fail("Vulkan opaque depth cannot be cleared after velocity capture");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->MotionBlurActive)
    {
        if (opaqueComplete && !m_Impl->MotionBlurResolved)
            return m_Impl->Fail("Vulkan transparent color must follow motion blur");
        if (m_Impl->MotionBlurResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("Vulkan transparent color must load completed scene color and depth");
    }
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->TemporalActive)
    {
        if (opaqueComplete && !m_Impl->TemporalResolved)
            return m_Impl->Fail("Vulkan transparent color must follow temporal reconstruction");
        if (m_Impl->TemporalResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("Vulkan transparent color must load reconstructed scene color and depth");
    }
    if (m_Impl->ReflectionResolved && desc.Type != PassType::Shadow && desc.ClearDepthEnabled)
        return m_Impl->Fail("Vulkan opaque depth cannot be cleared after SSR");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->GIActive)
    {
        if (!m_Impl->GBufferWritten) return m_Impl->Fail("SSGI requires material capture before lighting");
        if (opaqueComplete && !m_Impl->GI.Resolved) return m_Impl->Fail("Transparency must follow SSGI");
        if (m_Impl->GI.Resolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("Scene attachments cannot be cleared after SSGI");
    }
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->ReflectionActive)
    {
        if (!m_Impl->GBufferWritten) return m_Impl->Fail("Vulkan SSR requires materials before opaque lighting");
        if (opaqueComplete && !m_Impl->ReflectionResolved) return m_Impl->Fail("Vulkan transparency must follow SSR");
        if (m_Impl->ReflectionResolved && desc.ClearColorEnabled) return m_Impl->Fail("Vulkan scene color cannot be cleared after SSR");
    }
    const bool geometry = desc.Type == PassType::GBuffer;
    if (!desc.Target && m_Impl->DepthOfFieldResolved &&
        (desc.Type != PassType::Color || desc.ClearColorEnabled || desc.ClearDepthEnabled))
        return m_Impl->Fail("Vulkan only loaded transparent color may follow depth of field");
    if (geometry && (!(m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive) || desc.Target ||
        m_Impl->DeferredResolved || m_Impl->ForwardOpaqueComplete || m_Impl->AmbientOcclusionResolved))
        return m_Impl->Fail("Vulkan material capture requires an unresolved deferred/AO frame and the main target");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->DeferredActive && !m_Impl->DeferredResolved)
        return m_Impl->Fail("Vulkan forward transparency must follow deferred lighting");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->DepthOfFieldActive &&
        (m_Impl->DeferredResolved || m_Impl->ForwardOpaqueComplete) && !m_Impl->DepthOfFieldResolved)
        return m_Impl->Fail("Vulkan transparency must follow depth of field");
    if (desc.Type == PassType::Color && !desc.Target && m_Impl->AmbientOcclusionActive)
    {
        if (!m_Impl->GBufferWritten)
            return m_Impl->Fail("Vulkan screen-space AO requires material capture before forward color");
        if ((m_Impl->DeferredResolved || m_Impl->ForwardOpaqueComplete) && !m_Impl->AmbientOcclusionResolved)
            return m_Impl->Fail("Vulkan transparency must follow ambient occlusion");
        if (m_Impl->AmbientOcclusionResolved && (desc.ClearColorEnabled || desc.ClearDepthEnabled))
            return m_Impl->Fail("Vulkan color/depth cannot be cleared after ambient occlusion");
    }
    m_Impl->ActiveColor = m_Impl->HdrActive ? &m_Impl->HdrImages[0] : &m_Impl->MainColor;
    m_Impl->ActiveDepth = &m_Impl->MainDepth;
    m_Impl->ActiveFramebuffer = m_Impl->HdrActive ? m_Impl->HdrSceneFramebuffer : m_Impl->MainFramebuffer;
    if (geometry) m_Impl->ActiveFramebuffer = m_Impl->GBufferFramebuffer;
    if (velocity)
    {
        m_Impl->ActiveColor = &m_Impl->MotionImages[Impl::MotionVelocity];
        m_Impl->ActiveFramebuffer = m_Impl->MotionFramebuffers[Impl::MotionVelocity];
    }
    if (multisample && !desc.Target)
    {
        m_Impl->ActiveColor = &m_Impl->Multisample.Images[velocity ? Impl::SampleVelocity : Impl::SampleColor];
        m_Impl->ActiveDepth = &m_Impl->Multisample.Depth;
        m_Impl->ActiveFramebuffer = m_Impl->Multisample.Framebuffers[geometry ? Impl::SampleMaterialFB :
            (velocity ? Impl::SampleVelocityFB : Impl::SampleSceneFB)];
    }
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

    if (environmentCapture)
    {
        m_Impl->ActiveColor = &m_Impl->Environment.Cube;
        m_Impl->ActiveDepth = &m_Impl->Environment.Depth;
        m_Impl->ActiveFramebuffer = m_Impl->Environment.FaceFramebuffers[desc.EnvironmentFaceIndex];
        m_Impl->ActiveWidth = m_Impl->ActiveHeight = m_Impl->FrameEnvironment.Resolution;
    }
    else if (planarCapture)
    {
        m_Impl->ActiveColor = &m_Impl->Planar.Image; m_Impl->ActiveDepth = &m_Impl->Planar.Depth;
        m_Impl->ActiveFramebuffer = m_Impl->Planar.CaptureFramebuffer;
        m_Impl->ActiveWidth = m_Impl->FramePlanar.Width; m_Impl->ActiveHeight = m_Impl->FramePlanar.Height;
    }
    if (m_Impl->DepthOfFieldResolved && m_Impl->ActiveDepth == &m_Impl->MainDepth && desc.ClearDepthEnabled)
        return m_Impl->Fail("Vulkan main depth cannot be cleared after depth of field");

    const bool shadow = desc.Type == PassType::Shadow;
    const bool vsmCaster = shadow && m_Impl->UsesVSM();
    if (contactDepth || forwardPlusDepth)
    {
        m_Impl->ActiveColor = nullptr;
        m_Impl->ActiveFramebuffer = m_Impl->ContactDepthFramebuffer;
    }
    if (vsmCaster)
    {
        if (!m_Impl->EnsureMomentMap(desc.Target)) return false;
        auto& map = m_Impl->VSMMoments.at(desc.Target.Value);
        if (!desc.ClearDepthEnabled && !map.RawWritten) return m_Impl->Fail("Vulkan VSM requires an initial caster clear each frame");
        map.Ready = false;
        m_Impl->ActiveColor = &map.Images[Impl::VSMRaw];
        m_Impl->ActiveFramebuffer = map.Framebuffers[Impl::VSMRaw];
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveColor, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    m_Impl->WeightedPass = weightedPass;
    if (weightedPass) m_Impl->ActiveFramebuffer = m_Impl->Weighted.GeometryFramebuffer;
    if (desc.DepthPeelingLayer)
    {
        if (!m_Impl->PeelingSequence.Advance(DepthPeelingEvent::BeginShading,m_Impl->LastError)) return false;
        const auto step=m_Impl->PeelingSequence.Step();
        m_Impl->ActiveColor=&m_Impl->PeelingImages[4]; m_Impl->ActiveDepth=&m_Impl->PeelingImages[step.DepthWrite];
        m_Impl->ActiveFramebuffer=m_Impl->PeelingLayerFB[step.DepthWrite];
    }
    if (shadow || contactDepth || forwardPlusDepth)
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveDepth,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }
    else
    {
        if (weightedPass)
            for (auto& image : m_Impl->Weighted.Images)
                m_Impl->TransitionImage(m_Impl->CommandBuffer, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        else if (geometry)
            for (std::size_t index = 0; index < GBufferAttachmentCount; ++index)
                m_Impl->TransitionImage(m_Impl->CommandBuffer, multisample ? m_Impl->Multisample.Images[Impl::SampleMaterialFirst + index] :
                    m_Impl->GBufferImages[index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        else m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveColor,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveDepth,
                                (!desc.DepthPeelingLayer && (velocity || weightedPass || (mainColor && (m_Impl->ContactGenerated || (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned()))))) ?
                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    VkClearValue clearValues[GBufferAttachmentCount + 1]{};
    clearValues[0].color.float32[0] = desc.ClearColor.X;
    clearValues[0].color.float32[1] = desc.ClearColor.Y;
    clearValues[0].color.float32[2] = desc.ClearColor.Z;
    clearValues[0].color.float32[3] = desc.ClearColor.W;
    if (m_Impl->ActiveColor && m_Impl->ActiveColor->Format == VK_FORMAT_R16G16B16A16_SFLOAT)
        for (std::size_t channel = 0; channel < 3; ++channel)
            clearValues[0].color.float32[channel] = std::clamp(clearValues[0].color.float32[channel], 0.0f, MaximumHalfFloatRadiance);
    clearValues[contactDepth || forwardPlusDepth || (shadow && !vsmCaster) ? 0 : 1].depthStencil = {1.0f, 0};
    if (vsmCaster)
    { clearValues[0] = {}; clearValues[0].color.float32[0] = clearValues[0].color.float32[1] = 1; }
    if (velocity) clearValues[0] = {};
    if (geometry)
    {
        if (desc.ClearColorEnabled)
            m_Impl->DeferredBackground = {clearValues[0].color.float32[0], clearValues[0].color.float32[1],
                clearValues[0].color.float32[2], clearValues[0].color.float32[3]};
        for (auto& value : clearValues) value = {};
        clearValues[GBufferAttachmentCount].depthStencil = {1.0f, 0};
    }
    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = shadow
        ? (vsmCaster ? m_Impl->VSMCasterPasses : m_Impl->ShadowRenderPasses)[desc.ClearDepthEnabled ? 1U : 0U]
        : (geometry ? m_Impl->DeferredRenderPasses :
            ((capture || m_Impl->ActiveColor == &m_Impl->HdrImages[0]) ? m_Impl->HdrColorRenderPasses : m_Impl->ColorRenderPasses))
            [(desc.ClearColorEnabled ? 1U : 0U) | (desc.ClearDepthEnabled ? 2U : 0U)];
    if (velocity) begin.renderPass = m_Impl->MotionGeometryRenderPass;
    if (multisample && !shadow)
        begin.renderPass = velocity ? m_Impl->Multisample.VelocityPass :
            (geometry ? m_Impl->Multisample.MaterialPasses : m_Impl->Multisample.ColorPasses)
            [(desc.ClearColorEnabled ? 1U : 0U) | (desc.ClearDepthEnabled ? 2U : 0U)];
    begin.framebuffer = m_Impl->ActiveFramebuffer;
    if (weightedPass) begin.renderPass = m_Impl->Weighted.GeometryPasses[0];
    else if (mainColor && (m_Impl->ContactGenerated || (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned())))
        begin.renderPass = m_Impl->ContactColorPasses[desc.ClearColorEnabled ? 1 : 0];
    if (contactDepth || forwardPlusDepth) begin.renderPass = m_Impl->ContactDepthPass;
    begin.renderArea = {{0, 0}, {m_Impl->ActiveWidth, m_Impl->ActiveHeight}};
    begin.clearValueCount = contactDepth || forwardPlusDepth || (shadow && !vsmCaster) ? 1U : (geometry ? static_cast<std::uint32_t>(GBufferAttachmentCount) + 1 : 2U);
    begin.pClearValues = clearValues;
    if (weightedPass) begin.clearValueCount = 0;
    if (desc.DepthPeelingLayer) { begin.renderPass=m_Impl->HdrColorRenderPasses[1]; clearValues[0]={}; }
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
    if (!m_Impl->CheckThread("TSR scene boundary")) return false;
    if (m_Impl->TSRScene.Active && m_Impl->TSRScene.OpaqueReady && !m_Impl->CurrentPass.Target && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.DepthPeelingLayer && !pipeline.BlendEnabled) return m_Impl->Fail("Only transparent color draws follow the TSR opaque snapshot");

    if (m_Impl->FrameOpen && m_Impl->Hair.Active && m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target &&
        ((!m_Impl->Hair.Resolved && pipeline.BlendEnabled) || (m_Impl->Hair.Resolved && pipeline.DepthWrite)))
        return m_Impl->Fail("Opaque draws precede hair; transparent draws follow it without replacing depth");
    if (m_Impl->WeightedPass && (!pipeline.BlendEnabled || pipeline.DepthWrite || !pipeline.DepthTest || !pipeline.ColorWrite || pipeline.OutlineExtrusion))
        return m_Impl->Fail("Vulkan weighted geometry requires color accumulation and read-only depth testing");
    if (m_Impl->WeightedActive && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
        pipeline.BlendEnabled != m_Impl->WeightedPass)
        return m_Impl->Fail("Vulkan opaque color precedes weighted preparation; transparent color follows it");
    if (!m_Impl->CheckThread("BindPipeline") || !m_Impl->PassOpen)
        return m_Impl->Fail("Vulkan pipeline must be bound inside a pass");
    if (m_Impl->CurrentPass.Type == PassType::ContactDepth &&
        (!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.ColorWrite || pipeline.BlendEnabled || pipeline.OutlineExtrusion))
        return m_Impl->Fail("Vulkan contact depth requires opaque depth-only geometry");
    if (m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth &&
        (!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.ColorWrite || pipeline.BlendEnabled || pipeline.OutlineExtrusion))
        return m_Impl->Fail("Vulkan Forward+ depth requires opaque depth-only geometry");
    if (!m_Impl->ClusteredActive && m_Impl->LightListFrame.Assigned() && pipeline.DepthWrite &&
        (m_Impl->ActiveDepth == &m_Impl->MainDepth || m_Impl->ActiveDepth == &m_Impl->Multisample.Depth))
        return m_Impl->Fail("Vulkan Forward+ receivers must preserve assigned camera depth");
    if (m_Impl->LightListsActive && m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && !pipeline.DepthTest)
        return m_Impl->Fail("Vulkan Forward+ tile receivers require camera depth testing");
    if (m_Impl->ContactGenerated && pipeline.DepthWrite &&
        (m_Impl->ActiveDepth == &m_Impl->MainDepth || m_Impl->ActiveDepth == &m_Impl->Multisample.Depth))
        return m_Impl->Fail("Vulkan contact receivers must preserve the traced depth");
    if ((m_Impl->CurrentPass.Type == PassType::EnvironmentCapture || m_Impl->CurrentPass.Type == PassType::PlanarCapture) &&
        (!pipeline.DepthTest || !pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite ||
         pipeline.SamplesShadowMap || pipeline.OutlineExtrusion || pipeline.Shading != m_Impl->FrameReflectionShading))
        return m_Impl->Fail("Vulkan capture requires opaque depth and the frame shading model without shadow sampling");
    if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && !m_Impl->FramePlanar.Active())
        return m_Impl->Fail("Vulkan inactive planar capture must clear without drawing geometry");
    if (pipeline.OutlineThickness < 0.0f) return m_Impl->Fail("Vulkan outline thickness is invalid");
    if (m_Impl->LightShaftResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
        return m_Impl->Fail("Vulkan preserve opaque depth after light-shaft composition");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->LightShaftActive &&
        pipeline.BlendEnabled != m_Impl->LightShaftResolved)
        return m_Impl->Fail("Vulkan opaque draws precede light shafts; transparent draws follow them");
    if (m_Impl->FogResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
        return m_Impl->Fail("Vulkan opaque depth is read-only after fog composition");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->FogActive &&
        pipeline.BlendEnabled != m_Impl->FogResolved)
        return m_Impl->Fail("Vulkan opaque draws precede fog; transparent draws follow it");
    if (m_Impl->SamplesOpaqueResolved && m_Impl->ActiveDepth == &m_Impl->Multisample.Depth && pipeline.DepthWrite)
        return m_Impl->Fail("Vulkan original sample depth is read-only after opaque resolve");
    if (m_Impl->VelocityCaptured && (m_Impl->ActiveDepth == &m_Impl->MainDepth || m_Impl->ActiveDepth == &m_Impl->Multisample.Depth) && pipeline.DepthWrite)
        return m_Impl->Fail("Vulkan shared opaque depth is read-only after velocity capture");
    if (m_Impl->CurrentPass.Type == PassType::Velocity &&
        (!pipeline.DepthTest || pipeline.DepthWrite || pipeline.BlendEnabled || !pipeline.ColorWrite || pipeline.OutlineExtrusion))
        return m_Impl->Fail("Vulkan velocity requires opaque coverage and read-only depth testing");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->MotionBlurActive &&
        pipeline.BlendEnabled != m_Impl->MotionBlurResolved)
        return m_Impl->Fail("Vulkan opaque draws precede motion blur; transparent draws follow it");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->TemporalActive &&
        pipeline.BlendEnabled != m_Impl->TemporalResolved)
        return m_Impl->Fail("Vulkan opaque draws precede TAA; transparent draws follow it");
    if (m_Impl->ReflectionResolved && m_Impl->CurrentPass.Type != PassType::Shadow && pipeline.DepthWrite)
        return m_Impl->Fail("Vulkan opaque depth is read-only after SSR");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->GIActive &&
        pipeline.BlendEnabled != m_Impl->GI.Resolved) return m_Impl->Fail("SSGI separates opaque and transparent material draws");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->ReflectionActive &&
        pipeline.BlendEnabled != m_Impl->ReflectionResolved)
        return m_Impl->Fail("Vulkan opaque draws precede SSR; transparent draws follow it");
    if (m_Impl->DepthOfFieldResolved && (m_Impl->ActiveDepth == &m_Impl->MainDepth || m_Impl->ActiveDepth == &m_Impl->Multisample.Depth) && pipeline.DepthWrite)
        return m_Impl->Fail("Vulkan main depth cannot be written after depth of field");
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->DepthOfFieldActive)
    {
        if (m_Impl->DepthOfFieldResolved && (!pipeline.BlendEnabled || pipeline.DepthWrite))
            return m_Impl->Fail("Vulkan only transparent, depth-read-only draws may follow depth of field");
        if (!m_Impl->DepthOfFieldResolved && pipeline.BlendEnabled)
            return m_Impl->Fail("Vulkan transparent draws require depth of field first");
    }
    if (m_Impl->CurrentPass.Type == PassType::GBuffer &&
        (pipeline.BlendEnabled || pipeline.Shading != m_Impl->FrameReflectionShading ||
            (m_Impl->DeferredActive && pipeline.Shading == Techniques::ShadingModel::Gouraud)))
        return m_Impl->Fail("Vulkan G-buffer requires opaque materials and the frame shading model");
    if (m_Impl->CurrentPass.DepthPeelingLayer && !ValidateDepthPeelingPipeline(pipeline,m_Impl->LastError)) return false;
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
    if (m_Impl->CurrentPass.Type == PassType::PlanarCapture && value.Values != m_Impl->FramePlanar.ClippedViewProjection.Values)
        return m_Impl->Fail("Vulkan planar geometry requires the frozen oblique capture projection");
    if (m_Impl->CurrentPass.DepthPeelingLayer && value.Values!=m_Impl->FramePeeling.ViewProjection.Values)
        return m_Impl->Fail("OIT capture and replay require the same raster projection");
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

bool VulkanRHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc& desc, GpuDrivenBatchHandle& handle)
{
    handle = {};
    if (!m_Impl->CheckThread("PrepareGpuDrivenBatch") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        desc.Instances.empty() || !desc.IndexCount)
        return m_Impl->Fail("Vulkan GPU batch preparation requires an open frame outside a graphics pass");
    std::uint32_t index = 0;
    if (!m_Impl->PrepareGpuBatch(desc, index)) return false;
    handle = {m_Impl->GpuEpoch, index};
    ++m_Impl->PendingFrame.Statistics.GpuCullDispatchCount;
    return true;
}

bool VulkanRHI::RenderVisibilityBuffer(const VisibilityBufferDesc& desc)
{
    if (!m_Impl->CheckThread("RenderVisibilityBuffer") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->DeferredActive || m_Impl->GBufferWritten || m_Impl->FrameSampleCount != 1)
        return m_Impl->Fail("Vulkan visibility geometry requires an unresolved single-sample deferred frame");
    if (!ValidateVisibilityFrame(desc, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->LastError) ||
        !m_Impl->PrepareVisibilityBuffer(desc.Scene, desc.Draws.size())) return false;
        if (desc.NaniteRaster && !m_Impl->PrepareNaniteFrameRaster(desc)) return false;
        if (!m_Impl->PrepareNaniteHardware(desc)) return false;
        if (!m_Impl->PrepareMeshShaders(desc)) return false;
        if (!m_Impl->PrepareNaniteInstances(desc)) return false;
        if (!m_Impl->PrepareNaniteMaterials(desc)) return false;
    PassDesc pass; pass.Type = PassType::GBuffer; pass.ClearColor = desc.Background;
    if (!BeginPass(pass) || !m_Impl->DrawVisibilityBuffer(desc)) return false;
    RecordVisibilityDraws(m_Impl->PendingFrame, desc);
    return EndPass();
}

bool VulkanRHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle handle)
{
    if (!m_Impl->CheckThread("DrawGpuDrivenBatch") || !m_Impl->PassOpen || !handle.Frame ||
        handle.Frame != m_Impl->GpuEpoch || handle.Index >= m_Impl->GpuCursor || m_Impl->ActiveGpuBatch)
        return m_Impl->Fail("Vulkan indirect draw has an invalid frame token or pass state");
    auto& batch = m_Impl->GpuBatches[handle.Index];
    const auto& desc = batch.Description;
    if (!GpuDrivenDrawMatches(desc, m_Impl->BoundWorld, m_Impl->BoundViewProjection, m_Impl->BoundPipeline, m_Impl->CurrentPass))
        return m_Impl->Fail("Vulkan indirect draw does not match its prepared instance data");
    m_Impl->ActiveGpuBatch = &batch;
    struct Reset { Impl* State; ~Reset() { State->ActiveGpuBatch = nullptr; } } reset {m_Impl.get()};
    return DrawIndexedGeometry(desc.FirstIndex, desc.IndexCount, desc.BaseVertex, &desc.Instances);
}

bool VulkanRHI::DrawIndexed(std::uint32_t firstIndex, std::uint32_t indexCount,
                            std::int32_t baseVertex)
{
    return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, nullptr);
}

bool VulkanRHI::DrawIndexedInstanced(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
    const std::vector<DrawInstance>& instances)
{
    return DrawIndexedGeometry(firstIndex, indexCount, baseVertex, &instances);
}

bool VulkanRHI::DrawIndexedGeometry(std::uint32_t firstIndex, std::uint32_t indexCount, std::int32_t baseVertex,
    const std::vector<DrawInstance>* instances)
{
    if (m_Impl->PassOpen && m_Impl->Hair.Active && HairFurReplayPass(m_Impl->CurrentPass.Type) &&
        m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Hair replay ends the pass geometry stream");
    if (!m_Impl->CheckThread("DrawIndexed") || !m_Impl->PassOpen || !m_Impl->PipelineBound ||
        !m_Impl->WorldBound || !m_Impl->ViewProjectionBound || !m_Impl->LightViewProjectionBound ||
        !m_Impl->BoundVertexBuffer || !m_Impl->BoundIndexBuffer || indexCount == 0)
        return m_Impl->Fail("Vulkan DrawIndexed is missing required state");
    if (instances && m_Impl->CurrentPass.DepthPeelingLayer) return m_Impl->Fail("OIT replay requires ordinary invariant draws");
    if (instances && UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type))
        return m_Impl->Fail("POM material capture currently requires individual mesh draws");
    std::optional<InstanceConstantBlock> instanceConstants;
    if (instances)
    {
        instanceConstants.emplace();
        if (!GetCapabilities().InstancedRendering || m_Impl->CurrentPass.Type == PassType::Outline ||
            m_Impl->BoundPipeline.OutlineExtrusion || !BuildInstanceConstants(*instances, m_Impl->BoundWorld, *instanceConstants))
            return m_Impl->Fail("Vulkan instancing requires a supported uniform range and a bounded batch with finite transforms and common winding");
        instanceConstants->ViewProjection = m_Impl->BoundViewProjection;
        instanceConstants->Material.X = m_Impl->BoundPipeline.Color.W * m_Impl->BoundPipeline.MaterialOpacity;
    }
    auto vertex = m_Impl->Buffers.find(m_Impl->BoundVertexBuffer.Value);
    auto index = m_Impl->Buffers.find(m_Impl->BoundIndexBuffer.Value);
    if (vertex == m_Impl->Buffers.end() || index == m_Impl->Buffers.end())
        return m_Impl->Fail("Vulkan DrawIndexed requires complete triangle-list buffers");
    if (!ValidateIndexedGeometryDraw(vertex->second.Description,index->second.Description,index->second.Bytes,
        vertex->second.NaniteGeometry && index->second.NaniteGeometry,firstIndex,indexCount,baseVertex,m_Impl->LastError,
        &index->second.LastValidRange)) return false;

    const bool pcss = m_Impl->UsesSurfacePCSS(m_Impl->BoundPipeline, m_Impl->CurrentPass.Type);
    const bool pcf=m_Impl->UsesSurfacePCF(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type);
    const bool materialSurface=m_Impl->UsesMaterialSurface(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type);
    const bool vsm = m_Impl->UsesSurfaceVSM(m_Impl->BoundPipeline, m_Impl->CurrentPass.Type);
    const bool vsmCaster = m_Impl->UsesVSM() && m_Impl->CurrentPass.Type == PassType::Shadow;
    const bool forwardPlusMaterial = m_Impl->UsesForwardPlusMaterial(m_Impl->CurrentPass.Type);
    const bool forwardPlusDepth = m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth;
    const bool forwardPlus = forwardPlusMaterial || forwardPlusDepth;
    const bool mainCamera = m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target;
    if (m_Impl->LightListsActive && (m_Impl->CurrentPass.Type == PassType::ContactDepth || forwardPlusDepth || m_Impl->CurrentPass.Type == PassType::GBuffer) &&
        !m_Impl->LightListFrame.ValidateDraw(m_Impl->BoundPipeline.Lighting, m_Impl->BoundViewProjection, true, m_Impl->LastError)) return false;
    LightListSurfaceConstants selection;
    if (forwardPlusMaterial)
    {
        const auto mode = !mainCamera ? LightListSelection::FullScene : m_Impl->BoundPipeline.BlendEnabled ?
            LightListSelection::Transparent : LightListSelection::Opaque;
        if (!m_Impl->LightListFrame.Surface(m_Impl->BoundPipeline.Lighting, m_Impl->BoundViewProjection, mode, selection, m_Impl->LastError)) return false;
    }
    const std::size_t constantSize = forwardPlus ? sizeof(Impl::ForwardPlusDrawConstants) :
        materialSurface || pcf || pcss || vsm || vsmCaster ? sizeof(Impl::FilteredDrawConstants) : sizeof(RealtimeDrawConstants);
    std::array<std::uint8_t,sizeof(Impl::ForwardPlusDrawConstants)> uniformBytes{};
    std::size_t alignedOffset=0;
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
        if (vsm)
        {
            Impl::RawImage* moments = nullptr;
            if (!m_Impl->MomentImage(m_Impl->LastShadowTexture, moments)) return false;
            imageViews[0] = moments->View;
        }
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
        (bool(m_Impl->FrameRayShadowImage) || shadowAvailable || (m_Impl->FrameVirtualShadowAtlas && m_Impl->BoundPipeline.SamplesShadowMap && m_Impl->CurrentPass.Type==PassType::Color)),
        m_Impl->BoundPipeline.DiffuseTexture != TextureHandle {},
        m_Impl->BoundPipeline.NormalTexture != TextureHandle {},
        m_Impl->BoundPipeline.OpacityTexture != TextureHandle {},
        m_Impl->ActiveColor && m_Impl->ActiveColor->Format == VK_FORMAT_R16G16B16A16_SFLOAT);
    std::memcpy(
        uniformBytes.data(),
        &constants,
        sizeof(constants));
    if (m_Impl->CurrentPass.Type == PassType::Velocity)
    {
        MotionBlurConstants motion {};
        static_assert(sizeof(motion) <= sizeof(RealtimeDrawConstants), "Motion uniforms exceed the descriptor range");
        if (!BuildMotionGeometryConstants(m_Impl->BoundPipeline, m_Impl->BoundWorld, m_Impl->BoundViewProjection,
                GetMotionTransform(), m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
                bool(m_Impl->BoundPipeline.DiffuseTexture), bool(m_Impl->BoundPipeline.OpacityTexture), motion))
            return m_Impl->Fail("Vulkan velocity requires finite current and previous transforms");
        if (instances) motion.Options.W = 1; // Zero previous matrices invalidate only their own objects.
        std::memcpy(uniformBytes.data(), &motion, sizeof(motion));
    }
    if (materialSurface || pcf || pcss || vsm || vsmCaster || forwardPlus)
    {
        Impl::ShadowFilterBytes softShadows {};
        if (vsmCaster)
        {
            auto moments = BuildVSMParameters(m_Impl->FrameShadowFilter.VSM);
            if (!BuildVSMEncoding(m_Impl->BoundViewProjection, moments.DepthEncoding[0], m_Impl->LastError)) return false;
            std::memcpy(softShadows.data(), &moments, sizeof(moments));
        }
        else if (materialSurface)
        {
            const auto kernel=BuildPCFConstants(m_Impl->FrameShadowFilter,shadowAvailable);
            std::memcpy(softShadows.data(),&kernel,sizeof(kernel));
        }
        else if ((pcf || pcss || vsm) && !m_Impl->BuildShadowFilterData(m_Impl->BoundLightViewProjection, m_Impl->BoundPipeline.Lighting,
            shadowAvailable, softShadows)) return false;
        // Each recorded draw owns its aligned prefix and filter-specific suffix.
        std::memcpy(uniformBytes.data() + sizeof(constants),
            softShadows.data(), softShadows.size());
    }
    if (forwardPlus)
        std::memcpy(uniformBytes.data() + offsetof(Impl::ForwardPlusDrawConstants, Selection),
            &selection, sizeof(selection));
    VkDescriptorSet descriptorSet=0;
    std::string peelingKey;
    if (m_Impl->CurrentPass.DepthPeelingLayer)
    {
        peelingKey.assign(reinterpret_cast<const char*>(uniformBytes.data()),constantSize);
        peelingKey.append(reinterpret_cast<const char*>(imageViews.data()),sizeof(imageViews));
        const auto found=m_Impl->PeelingDrawBindings.find(peelingKey);
        if (found!=m_Impl->PeelingDrawBindings.end()) { alignedOffset=found->second.first; descriptorSet=found->second.second; }
    }
    if (!descriptorSet)
    {
        alignedOffset=AlignUp(m_Impl->UniformOffset,UniformAlignment);
        if (alignedOffset>UniformBufferSize || constantSize>UniformBufferSize-alignedOffset)
            return m_Impl->Fail("Vulkan per-frame uniform arena is exhausted");
        if (m_Impl->DescriptorSetIndex>=m_Impl->DescriptorSets.size()) return m_Impl->Fail("Vulkan per-frame descriptor arena is exhausted");
        std::memcpy(static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped)+alignedOffset,uniformBytes.data(),constantSize);
        m_Impl->UniformOffset=alignedOffset+constantSize;
        descriptorSet=m_Impl->DescriptorSets[m_Impl->DescriptorSetIndex++];
        m_Impl->UpdateDescriptorSet(descriptorSet,imageViews,m_Impl->MaterialSampler(),constantSize);
        if (forwardPlusMaterial) m_Impl->BindForwardPlusStorage(descriptorSet,mainCamera);
        if (m_Impl->CurrentPass.DepthPeelingLayer)
            m_Impl->PeelingDrawBindings.emplace(std::move(peelingKey),std::make_pair(std::uint32_t(alignedOffset),descriptorSet));
    }

    const VkPipeline pipeline = m_Impl->GetPipeline(
        m_Impl->BoundPipeline, m_Impl->CurrentPass.Type, instances != nullptr);
    if (pipeline == 0) return false;
    m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(alignedOffset);
    m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  m_Impl->PipelineLayout, 0, 1, &descriptorSet,
                                  1, &dynamicOffset);
    if (!m_Impl->BindCascades(m_Impl->PipelineLayout,
        m_Impl->CurrentPass.Type == PassType::Color && m_Impl->BoundPipeline.SamplesShadowMap)) return false;
    if (instances && !m_Impl->UploadInstances(*instanceConstants)) return false;
    const bool foggedSurface = m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target &&
        m_Impl->BoundPipeline.BlendEnabled && m_Impl->FogActive && m_Impl->FogResolved;
    auto fogSet = foggedSurface ? Impl::FogSurfaceSet : Impl::FogDisabledSet;
    if (m_Impl->CurrentPass.Type==PassType::Color && !m_Impl->CurrentPass.Target && m_Impl->BoundPipeline.BlendEnabled && m_Impl->FrameGI.Lumen.Translucency.Enabled && m_Impl->LumenVolumeFrame.Grid.W!=0)
    {
        const unsigned variant=foggedSurface?1u:0u; fogSet=Impl::LumenVolumeFirstReceiverSet+variant;
        if (!m_Impl->LumenVolumeReceiverWritten[variant])
        {
            VolumetricFogConstants medium{}; const auto dummy=m_Impl->DummyColor.View;
            if (foggedSurface) std::memcpy(&medium,static_cast<const std::uint8_t*>(m_Impl->UniformBuffer.Mapped)+m_Impl->FogUniformOffsets[Impl::FogSurfaceSet],sizeof(medium));
            if (!m_Impl->WriteFogSet(fogSet,medium,{dummy,dummy,foggedSurface?m_Impl->FogAtlas.View:dummy})) return false;
            m_Impl->LumenVolumeReceiverWritten[variant]=true;
        }
    }
    m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->PipelineLayout,
        FogDescriptorSetIndex, 1, &m_Impl->FogDescriptorSets[fogSet], 1, &m_Impl->FogUniformOffsets[fogSet]);
    if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type) && !m_Impl->BindParallaxDraw()) return false;
    const VkDeviceSize vertexOffset = 0;
    if (m_Impl->WeightedPass)
    {
        const auto transparency = BuildWeightedTransparencyConstants(m_Impl->FrameTransparency);
        m_Impl->CmdPushConstants(m_Impl->CommandBuffer, m_Impl->PipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(transparency.Weights), &transparency.Weights);
    }
    m_Impl->CmdBindVertexBuffers(m_Impl->CommandBuffer, 0, 1,
                                 &vertex->second.Allocation.Buffer, &vertexOffset);
    m_Impl->CmdBindIndexBuffer(m_Impl->CommandBuffer,m_Impl->ActiveNaniteDraw && m_Impl->ActiveNaniteDraw->Binning?m_Impl->ActiveNaniteDraw->Arguments->Allocation.Buffer:index->second.Allocation.Buffer,
                               0, VK_INDEX_TYPE_UINT32);
    if (m_Impl->ActiveGpuBatch)
        m_Impl->GpuDrawIndexedIndirect(m_Impl->CommandBuffer, m_Impl->ActiveGpuBatch->Arguments.Buffer, 0, 1, sizeof(GpuDrivenIndexedArguments));
    else if (m_Impl->ActiveNaniteDraw) m_Impl->GpuDrawIndexedIndirect(m_Impl->CommandBuffer,m_Impl->ActiveNaniteDraw->Arguments->Allocation.Buffer,m_Impl->NaniteArgumentOffset(),1,NaniteDrawArgumentWords*sizeof(unsigned));
    else m_Impl->CmdDrawIndexed(m_Impl->CommandBuffer, indexCount,
        instances ? static_cast<std::uint32_t>(instances->size()) : 1, firstIndex, baseVertex, 0);

    if (UsesParallaxMaterial(m_Impl->BoundPipeline,m_Impl->CurrentPass.Type)) ++m_Impl->PendingFrame.Statistics.ParallaxMaterialDrawCount;
    DrawCallRecord record;
    record.VertexBuffer = m_Impl->BoundVertexBuffer;
    record.IndexBuffer = m_Impl->BoundIndexBuffer;
    record.Pipeline = m_Impl->BoundPipeline;
    record.WorldTransform = m_Impl->BoundWorld;
    record.Motion = GetMotionTransform();
    record.FirstIndex = firstIndex;
    record.IndexCount = indexCount;
    record.BaseVertex = baseVertex;
    if (instances) record.Instances = *instances;
    record.GPUDriven = m_Impl->ActiveGpuBatch != nullptr;
        record.NaniteIndirect=m_Impl->ActiveNaniteDraw!=nullptr; record.NaniteQueryIndex=record.NaniteIndirect?m_Impl->ActiveNaniteQuery:0;
    m_Impl->PendingFrame.Passes.back().DrawCalls.push_back(std::move(record));
    FrameStatistics& statistics = m_Impl->PendingFrame.Statistics;
    ++statistics.DrawCallCount;
    statistics.InstancedDrawCallCount += instances ? 1u : 0u;
    if (m_Impl->ActiveGpuBatch)
    {
        ++statistics.IndirectDrawCallCount;
        statistics.GpuCandidateInstanceCount += instances->size();
        statistics.GpuCandidateTriangleCount += std::uint64_t(indexCount / 3) * instances->size();
    }
    else if (m_Impl->ActiveNaniteDraw)
        {
            ++statistics.IndirectDrawCallCount; ++statistics.NaniteIndirectDrawCount;
            statistics.NaniteCandidateTriangles+=indexCount/3;
        }
        else statistics.SubmittedTriangleCount += std::uint64_t(indexCount / 3U) * (instances ? instances->size() : 1);
    statistics.ShadowDrawCallCount += shadow ? 1U : 0U;
    statistics.ContactDepthDrawCallCount += m_Impl->CurrentPass.Type == PassType::ContactDepth ? 1U : 0U;
    statistics.ForwardPlusDepthDrawCallCount += forwardPlusDepth ? 1U : 0U;
    statistics.OutlineDrawCallCount += m_Impl->CurrentPass.Type == PassType::Outline ? 1U : 0U;
    return true;
}

bool VulkanRHI::PrepareLightListFrame(const LightListFrameDesc& desc)
{
    if (!m_Impl->CheckThread("PrepareLightListFrame") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        desc.Layout != (m_Impl->ClusteredActive ? LightListLayout::Clusters : LightListLayout::Tiles) || !m_Impl->LightListsActive || !m_Impl->PendingFrame.Passes.empty())
        return m_Impl->Fail("Vulkan Forward+ scene storage must precede every frame pass");
    if (m_Impl->ClusteredActive && !m_Impl->ClusteredProducer)
    {
        auto producer = std::make_unique<VulkanClusteredForward>();
        if (!producer->Initialize(m_Impl->Device, m_Impl->MemoryProperties, m_Impl->DeviceLimits,
            m_Impl->GetDeviceProcAddress, m_Impl->LastError)) return false;
        m_Impl->ClusteredProducer = std::move(producer);
    }
    if (!m_Impl->ClusteredActive && !m_Impl->TileProducer())
    {
        auto producer = std::make_unique<VulkanForwardPlus>(m_Impl->TiledActive ? TiledLightShader::TiledDeferred : TiledLightShader::ForwardPlus);
        if (!producer->Initialize(m_Impl->Device, m_Impl->MemoryProperties, m_Impl->DeviceLimits,
            m_Impl->GetDeviceProcAddress, m_Impl->LastError)) return false;
        (m_Impl->TiledActive ? m_Impl->TiledProducer : m_Impl->ForwardPlusProducer) = std::move(producer);
    }
    if (!m_Impl->LightListFrame.Prepare(desc, m_Impl->LastError) ||
        !(m_Impl->ClusteredActive ? m_Impl->ClusteredProducer->BeginFrame(m_Impl->LastError) : m_Impl->TileProducer()->BeginFrame(m_Impl->LastError))) return false;
    m_Impl->ForwardPlusRecordOpen = true;
    return m_Impl->UploadForwardPlusScene();
}

bool VulkanRHI::GenerateLightLists()
{
    if (!m_Impl->CheckThread("GenerateLightLists") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->LightListsActive || !m_Impl->LightListFrame.Prepared() || m_Impl->LightListFrame.Assigned() ||
        (!m_Impl->ClusteredActive && !m_Impl->ForwardPlusDepthWritten && !m_Impl->GBufferWritten) || m_Impl->ContactGenerated || m_Impl->ForwardOpaqueComplete)
        return m_Impl->Fail("Vulkan Forward+ assignment requires completed opaque depth, once before visibility and lighting");
    if (m_Impl->ClusteredActive)
    {
        const auto& frame = m_Impl->LightListFrame.Description();
        if (!m_Impl->ClusteredProducer->Record(m_Impl->CommandBuffer, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
            frame.ViewProjection, frame.Lights->GetBounds(), {}, m_Impl->LastError)) return false;
        m_Impl->ForwardPlusRecorded = true;
        if (!m_Impl->LightListFrame.CompleteAssignment(m_Impl->ClusteredProducer->Grid(), m_Impl->LastError)) return false;
        ++m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount;
        return true;
    }
    auto& depth = m_Impl->FrameSampleCount > 1 ? m_Impl->Multisample.Depth : m_Impl->MainDepth;
    VulkanForwardPlusDepth input;
    input.Image = depth.Image; input.View = depth.View; input.Format = depth.Format;
    input.Width = depth.Width; input.Height = depth.Height; input.Samples = depth.Samples;
    if (std::none_of(m_Impl->FrameImageLayouts.begin(), m_Impl->FrameImageLayouts.end(),
        [&depth](const auto& entry) { return entry.first == &depth; }))
        m_Impl->FrameImageLayouts.emplace_back(&depth, depth.Layout);
    const auto& frame = m_Impl->LightListFrame.Description();
    if (!m_Impl->TileProducer()->Record(m_Impl->CommandBuffer, input, depth.Layout, frame.ViewProjection,
        frame.Lights->GetBounds(), {}, m_Impl->LastError)) return false;
    m_Impl->ForwardPlusRecorded = true;
    if (!m_Impl->LightListFrame.CompleteAssignment(m_Impl->TileProducer()->Grid(), m_Impl->LastError)) return false;
    ++(m_Impl->TiledActive ? m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount : m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount);
    return true;
}

bool VulkanRHI::GenerateContactShadows(const ContactShadowDesc& desc)
{
    if (!m_Impl->CheckThread("GenerateContactShadows") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->FrameContact.Enabled || m_Impl->ContactGenerated || m_Impl->ForwardOpaqueComplete || m_Impl->DeferredResolved ||
        (!m_Impl->ContactDepthWritten && !m_Impl->GBufferWritten && !m_Impl->ForwardPlusDepthWritten))
        return m_Impl->Fail("Vulkan contact visibility requires completed opaque depth before lighting, once per frame");
    if (m_Impl->LightListsActive)
    {
        if (!m_Impl->LightListFrame.Assigned())
            return m_Impl->Fail("Vulkan Forward+ contact visibility must follow light assignment");
        if (!m_Impl->LightListFrame.ValidateDraw(desc.Lighting, desc.ViewProjection, true, m_Impl->LastError)) return false;
    }

    auto lighting = desc.Lighting;
    if (lighting.SceneLights) lighting.SceneLights->WriteLegacySubset(lighting);
    if (!m_Impl->Hair.ValidateContactProducer(desc.ViewProjection,lighting,m_Impl->LastError)) return false;
    const auto light = lighting.ShadowLightIndex;
    const bool active = light >= 0 && light < int(std::min(lighting.LightCount, MaximumSceneLights)) &&
        lighting.Lights[light].CastShadows;
    if (active)
    {
        auto& depth = m_Impl->FrameSampleCount > 1 ? m_Impl->Multisample.Depth : m_Impl->MainDepth;
        VulkanContactDepth input;
        input.Image = depth.Image; input.View = depth.View; input.Format = depth.Format;
        input.Width = depth.Width; input.Height = depth.Height; input.Samples = depth.Samples;
        // Vulkan's standardSampleLocations guarantee defines positions and index
        // order. Capability selection requires that guarantee; no vendor pattern
        // is guessed. Coordinates below are the spec's sixteenth-pixel grid.
        // https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-multisampling
        constexpr float SubpixelGrid = 16.0f, PixelCenter = 0.5f;
        std::array<Float2, 8> locations {};
        switch (input.Samples)
        {
        case 1: locations[0] = {8, 8}; break;
        case 2: locations = {{{12, 12}, {4, 4}}}; break;
        case 4: locations = {{{6, 2}, {14, 6}, {2, 10}, {10, 14}}}; break;
        case 8: locations = {{{9, 5}, {7, 11}, {13, 9}, {5, 3}, {3, 13}, {1, 7}, {11, 15}, {15, 1}}}; break;
        default: return m_Impl->Fail("Vulkan contact shadows require a supported standard sample count");
        }
        for (std::uint32_t sample = 0; sample < input.Samples; ++sample)
            input.SampleOffsets[sample] = {locations[sample].X / SubpixelGrid - PixelCenter,
                locations[sample].Y / SubpixelGrid - PixelCenter};
        ContactShadowDispatch dispatch;
        dispatch.ViewProjection = desc.ViewProjection; dispatch.Light = lighting.Lights[light];
        dispatch.MaximumDistance = m_Impl->FrameContact.MaximumDistance;
        dispatch.Parameters.StartBias = m_Impl->FrameContact.StartBias;
        dispatch.Parameters.Thickness = m_Impl->FrameContact.Thickness;
        dispatch.Parameters.MaximumSteps = m_Impl->FrameContact.MaximumSteps;
        if (std::none_of(m_Impl->FrameImageLayouts.begin(), m_Impl->FrameImageLayouts.end(),
            [&depth](const auto& entry) { return entry.first == &depth; }))
            m_Impl->FrameImageLayouts.emplace_back(&depth, depth.Layout);
        if (!m_Impl->ContactProducer->Record(m_Impl->CommandBuffer, input, depth.Layout, dispatch, m_Impl->LastError)) return false;
        ++m_Impl->PendingFrame.Statistics.ContactShadowPassCount;
    }
    m_Impl->ContactReceiverActive = active;
    m_Impl->Hair.RecordContact(desc.ViewProjection,lighting,m_Impl->ContactReceiverActive);
    m_Impl->ContactGenerated = true;
    return true;
}

bool VulkanRHI::DrawHairFurVelocity(const HairFurVelocityDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFurVelocity") || !m_Impl->FrameOpen || !m_Impl->PassOpen || m_Impl->CurrentPass.Type!=PassType::Velocity ||
        !m_Impl->Hair.Active || !m_Impl->Hair.Resolved || !m_Impl->Hair.VelocityRequested || m_Impl->Hair.VelocityResolved ||
        m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size() || !m_Impl->Hair.MatchesPrimaryView(desc.View))
        return m_Impl->Fail("Fiber velocity requires matching completed primary geometry, once at the read-only velocity pass tail");
    if (m_Impl->ViewProjectionBound && desc.View.ViewProjection.Values!=m_Impl->BoundViewProjection.Values)
        return m_Impl->Fail("Fiber velocity projection must match the current mesh velocity pass");
    HairFurConstants constants;
    if (!BuildHairFurVelocityConstants(desc,m_Impl->Hair,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,constants,m_Impl->LastError) ||
        !m_Impl->ReplayHairFur(constants,HairFurRasterPhase::Velocity)) return false;
    m_Impl->Hair.ReplayedPass=m_Impl->PendingFrame.Passes.size(); m_Impl->Hair.VelocityResolved=true;
    ++m_Impl->PendingFrame.Statistics.HairFurVelocityDrawCount; return true;
}

bool VulkanRHI::DrawHairFurInPass(const HairFurDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFurInPass") || !m_Impl->FrameOpen || !m_Impl->PassOpen || !m_Impl->Hair.Active || m_Impl->Hair.Resolved ||
        !HairFurReplayPass(m_Impl->CurrentPass.Type) || m_Impl->CurrentPass.Type==PassType::Velocity || m_Impl->Hair.ReplayedPass==m_Impl->PendingFrame.Passes.size())
        return m_Impl->Fail("Hair replay requires an open depth, shadow or capture pass, once at its geometry tail");
    const auto type=m_Impl->CurrentPass.Type;
    if (type==PassType::PlanarCapture && !m_Impl->FramePlanar.Active()) return m_Impl->Fail("Inactive planar captures must remain empty");
    if (HairFurCameraDepthPass(type) && !m_Impl->Hair.MatchesPrimaryView(desc)) return m_Impl->Fail("Hair camera depth views must match");
    if ((HairFurCameraDepthPass(type) || type==PassType::Shadow) && m_Impl->ViewProjectionBound && desc.ViewProjection.Values!=m_Impl->BoundViewProjection.Values)
        return m_Impl->Fail("Hair replay projection must match the current mesh pass");
    if (type==PassType::EnvironmentCapture || type==PassType::PlanarCapture)
    {
        auto expected=desc;
        if (type==PassType::EnvironmentCapture)
        {
            const auto& face=m_Impl->FrameEnvironment.Faces[m_Impl->CurrentPass.EnvironmentFaceIndex];
            expected.ViewProjection=EnvironmentRasterProjection(face,GetBackend()); expected.Lighting.CameraPosition=m_Impl->FrameEnvironment.Position;
            expected.Forward=face.Forward; expected.Orthographic=false; expected.ClipPlane={0,0,0,0};
        }
        else
        {
            expected.ViewProjection=m_Impl->FramePlanar.ClippedViewProjection; expected.Lighting.CameraPosition=m_Impl->FramePlanar.Position;
            expected.ClipPlane=m_Impl->FramePlanar.ClipPlane;
        }
        if (!SameHairFurRasterView(desc,expected)) return m_Impl->Fail("Hair captures require the frozen reflection camera and clip plane");
    }
    const auto phase=HairFurReplayPhase(type,m_Impl->FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM); HairFurConstants constants;
    if (!BuildHairFurReplayConstants(desc,m_Impl->Hair.Frame,phase,m_Impl->FrameShadowFilter.VSM,constants,m_Impl->LastError) || !m_Impl->ReplayHairFur(constants,phase)) return false;
    m_Impl->Hair.ReplayedPass=m_Impl->PendingFrame.Passes.size();
    if (type==PassType::Shadow) m_Impl->Hair.RecordShadowView(m_Impl->CurrentPass.Target,desc.ViewProjection);
    if (HairFurCameraDepthPass(type)) m_Impl->Hair.RecordPrimaryView(desc);
    if (!m_Impl->Hair.Frame.Geometry->Indices.empty())
    {
        if (type==PassType::Shadow) ++m_Impl->PendingFrame.Statistics.HairFurShadowDrawCount;
        else if (HairFurCameraDepthPass(type)) ++m_Impl->PendingFrame.Statistics.HairFurDepthDrawCount;
        else ++m_Impl->PendingFrame.Statistics.HairFurCaptureDrawCount;
    }
    return true;
}

bool VulkanRHI::DrawHairFur(const HairFurDesc& desc)
{
    if (!m_Impl->CheckThread("DrawHairFur") || !m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->Hair.Active || m_Impl->Hair.Resolved || !m_Impl->GBufferWritten ||
        !(m_Impl->DeferredActive?m_Impl->DeferredResolved:m_Impl->ForwardOpaqueComplete) || m_Impl->SamplesOpaqueResolved || m_Impl->VelocityCaptured ||
        !HairFurPrecedesLaterPasses(m_Impl->PendingFrame))
        return m_Impl->Fail("Hair requires completed opaque lighting, once before MSAA resolution and screen-space effects");
    if (!m_Impl->Hair.MatchesPrimaryView(desc)) return m_Impl->Fail("Hair camera depth and primary surface require identical frozen raster inputs");
    HairFurConstants constants;
    if (!BuildHairFurConstants(desc.ViewProjection,desc.Lighting,m_Impl->Hair.Frame.Settings,*m_Impl->Hair.Frame.Lights,
        desc.Orthographic,desc.Forward,desc.ClipPlane,constants)) return m_Impl->Fail("Hair camera and lighting constants must be finite");
    if (!m_Impl->PrepareHairShadowReceiver(desc,constants) || !m_Impl->RenderHairFur(constants)) return false;
    m_Impl->Hair.RecordPrimaryView(desc); m_Impl->Hair.Resolved=true;
    if (!m_Impl->Hair.Frame.Geometry->Indices.empty()) ++m_Impl->PendingFrame.Statistics.HairFurPassCount;
    return true;
}

bool VulkanRHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc& desc)
{
    if (!m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before subsurface diffusion");
    if (!m_Impl->CheckThread("ApplySubsurfaceScattering") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->DeferredActive || !m_Impl->DeferredResolved || !m_Impl->SubsurfaceHasScene || m_Impl->SubsurfaceResolved || m_Impl->TiledActive ||
        m_Impl->PendingFrame.Statistics.SubsurfacePassCount ||
        (m_Impl->FrameSampleCount>1 && (!m_Impl->SamplesOpaqueResolved || m_Impl->SamplesColorRestored || m_Impl->SamplesTransparentWritten)))
        return m_Impl->Fail("Subsurface diffusion requires resolved opaque scene lighting, once before transparency");
    if (!SubsurfacePrecedesLaterPasses(m_Impl->PendingFrame))
        return m_Impl->Fail("Subsurface diffusion must precede AO, GI, reflections and lens/temporal effects");
    if (m_Impl->SubsurfaceRequested && !(Techniques::SanitizeSubsurfaceScatteringSettings(desc.Settings)==m_Impl->FrameSubsurface))
        return m_Impl->Fail("Subsurface controls must match the immutable frame settings");
    SubsurfaceScatteringConstants constants;
    if (!Techniques::UsesSubsurfaceScattering(desc.Settings) ||
        !BuildSubsurfaceScatteringConstants(m_Impl->SubsurfaceLighting.ViewProjection,desc.Settings,
            m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,Techniques::SubsurfacePass::Horizontal,constants))
        return m_Impl->Fail("Subsurface diffusion requires active finite profile and camera parameters");
    if (!m_Impl->SnapshotSubsurfaceScene()) return false;
    auto source=m_Impl->SubsurfaceLighting; source.Output=DeferredOutputMode::DirectDiffuse;
    m_Impl->SubsurfaceProducing=true;
    struct Reset { bool& Value; ~Reset() { Value=false; } } reset{m_Impl->SubsurfaceProducing};
    const bool produced=DrawDeferredLighting(source); m_Impl->SubsurfaceProducing=false;
    return produced && m_Impl->CaptureSubsurfaceSource() && m_Impl->FilterSubsurface(constants);
}

bool VulkanRHI::DrawDeferredLighting(const DeferredLightingDesc& desc)
{
    const auto lightingDescriptor=m_Impl->SubsurfaceProducing ? m_Impl->SubsurfaceSourceDescriptor : m_Impl->DeferredDescriptorSet;
    if (desc.Output == DeferredOutputMode::DirectDiffuse && m_Impl->TiledActive)
        return m_Impl->Fail("Direct-diffuse output requires ordinary deferred lighting");
    if (m_Impl->FrameContact.Enabled && !m_Impl->ContactGenerated)
        return m_Impl->Fail("Vulkan deferred lighting requires current contact visibility");
    if (!m_Impl->CheckThread("DrawDeferredLighting") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->DeferredActive || !m_Impl->GBufferWritten || (m_Impl->DeferredResolved && !m_Impl->SubsurfaceProducing) || m_Impl->DepthOfFieldResolved ||
        desc.Shading != GetTechniqueSettings().Shading)
        return m_Impl->Fail("Vulkan deferred lighting requires completed geometry and the frame shading model");

    const bool multisample = m_Impl->FrameSampleCount > 1;
    Impl::RawImage* shadow = &m_Impl->DummyDepth;
    if (desc.SamplesShadowMap && m_Impl->LastShadowTexture)
    {
        const auto found = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
        if (found == m_Impl->Textures.end() || found->second.Description.Format != TextureFormat::Depth32)
            return m_Impl->Fail("Vulkan deferred shadow map is unavailable");
        shadow = &found->second.Image;
    }
    const bool pcss = m_Impl->FrameShadowFilter.Mode == Techniques::ShadowFilterMode::PCSS;
    const bool vsm = m_Impl->UsesVSM();
    Impl::FilteredDeferredConstants constants {};
    if (!BuildDeferredConstants(desc, (shadow != &m_Impl->DummyDepth || bool(m_Impl->FrameVirtualShadowAtlas) || bool(m_Impl->FrameRayShadowImage)), m_Impl->DeferredBackground, constants.Lighting))
        return m_Impl->Fail("Vulkan deferred lighting requires an invertible finite camera matrix");
    VkPipeline lightingPipeline = multisample ? m_Impl->Multisample.LightingPipeline : m_Impl->DeferredLightingPipeline;
    if (pcss || vsm)
    {
        if (!m_Impl->BuildShadowFilterData(desc.LightViewProjection, desc.Lighting,
            desc.SamplesShadowMap && shadow != &m_Impl->DummyDepth, constants.SoftShadows)) return false;
        if (vsm && shadow != &m_Impl->DummyDepth && !m_Impl->MomentImage(m_Impl->LastShadowTexture, shadow)) return false;
        if (!m_Impl->TiledActive)
        {
        lightingPipeline = m_Impl->CreateFullscreenPipeline(vsm ? VulkanVSMScreenShaders[multisample ? 1 : 0] : VulkanPCSSScreenShaders[multisample ? 1 : 0],
            vsm ? (multisample ? m_Impl->Multisample.VSMLightingModules : m_Impl->VSMDeferredModules) :
                (multisample ? m_Impl->Multisample.PCSSLightingModules : m_Impl->PCSSDeferredModules),
            multisample ? m_Impl->Multisample.LightingPass : m_Impl->HdrRenderPass,
            vsm ? (multisample ? m_Impl->Multisample.VSMLightingPipeline : m_Impl->VSMDeferredPipeline) :
                (multisample ? m_Impl->Multisample.PCSSLightingPipeline : m_Impl->PCSSDeferredPipeline),
            m_Impl->DeferredPipelineLayout, FullscreenBlend::Replace, false, m_Impl->FrameSampleCount, multisample);
        if (!lightingPipeline) return false;
        }
    }
    if (m_Impl->ContactReceiverActive && !m_Impl->TiledActive)
    {
        lightingPipeline = m_Impl->ContactDeferredPipeline();
        if (!lightingPipeline) return false;
    }
    if (m_Impl->UsesDedicatedPCF())
    {
        if (!m_Impl->BuildShadowFilterData(desc.LightViewProjection,desc.Lighting,desc.SamplesShadowMap && shadow!=&m_Impl->DummyDepth,constants.SoftShadows)) return false;
        lightingPipeline=m_Impl->PCFScreen(multisample?1:0); if (!lightingPipeline) return false;
    }
    if (m_Impl->FrameRayShadowImage && !m_Impl->TiledActive)
    { lightingPipeline=m_Impl->RayShadowScreen(multisample?1:0); if (!lightingPipeline) return false; }
    Impl::TiledLightingConstants tiledConstants {};
    const void* upload = &constants;
    std::size_t uploadSize = sizeof(constants);
    if (m_Impl->TiledActive)
    {
        if (!BuildTiledDeferredConstants(desc, (shadow != &m_Impl->DummyDepth || bool(m_Impl->FrameVirtualShadowAtlas) || bool(m_Impl->FrameRayShadowImage)), m_Impl->DeferredBackground,
                tiledConstants.Surface.Surface.Draw, tiledConstants.Reconstruction) ||
            !m_Impl->LightListFrame.Surface(desc.Lighting, desc.ViewProjection, LightListSelection::Opaque,
                tiledConstants.Surface.Selection, m_Impl->LastError)) return false;
        tiledConstants.Surface.Surface.SoftShadows = constants.SoftShadows;
        upload = &tiledConstants; uploadSize = sizeof(tiledConstants);
        lightingPipeline = m_Impl->TiledLightingPipeline();
        if (!lightingPipeline) return false;
    }
    const std::size_t offset = AlignUp(m_Impl->UniformOffset, UniformAlignment);
    if (offset > UniformBufferSize || uploadSize > UniformBufferSize - offset)
        return m_Impl->Fail("Vulkan deferred uniform arena is exhausted");
    std::memcpy(static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped) + offset, upload, uploadSize);
    m_Impl->UniformOffset = offset + uploadSize;

    // The lighting framebuffer deliberately has no depth attachment. Its sampled
    // opaque depth is restored to attachment layout only for forward transparency.
    constexpr std::size_t ImageCount = GBufferAttachmentCount + 2;
    std::array<Impl::RawImage*, ImageCount> images{};
    for (std::size_t index = 0; index < GBufferAttachmentCount; ++index) images[index] = multisample ? &m_Impl->Multisample.Images[Impl::SampleMaterialFirst + index] : &m_Impl->GBufferImages[index];
    images[GBufferAttachmentCount] = multisample ? &m_Impl->Multisample.Depth : &m_Impl->MainDepth;
    images[GBufferAttachmentCount + 1] = shadow;
    std::array<VkDescriptorImageInfo, ImageCount + 1> imageInfos{};
    std::array<VkWriteDescriptorSet, ImageCount + 2> writes{};
    const VkDescriptorBufferInfo buffer{m_Impl->UniformBuffer.Buffer, 0, uploadSize};
    for (std::size_t index = 0; index < writes.size(); ++index)
    {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = lightingDescriptor;
        writes[index].dstBinding = static_cast<std::uint32_t>(index);
        writes[index].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[0].pBufferInfo = &buffer;
    for (std::size_t index = 0; index < ImageCount; ++index)
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *images[index], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        imageInfos[index].imageView = images[index]->View;
        imageInfos[index].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        writes[index + 1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[index + 1].pImageInfo = &imageInfos[index];
    }
    imageInfos[ImageCount].sampler = m_Impl->DeferredSampler;
    writes.back().descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes.back().pImageInfo = &imageInfos[ImageCount];
    m_Impl->UpdateDescriptorSets(m_Impl->Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    m_Impl->BindContactImage(lightingDescriptor, 9);
    if (m_Impl->TiledActive) m_Impl->BindForwardPlusStorage(lightingDescriptor, true, 10);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, multisample ? m_Impl->Multisample.Images[Impl::SampleColor] : m_Impl->HdrImages[0], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkClearValue clear{};
    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = multisample ? m_Impl->Multisample.LightingPass : m_Impl->HdrRenderPass;
    begin.framebuffer = multisample ? m_Impl->Multisample.Framebuffers[Impl::SampleLightingFB] : m_Impl->HdrFramebuffers[0];
    begin.renderArea = {{0, 0}, {m_Impl->MainColor.Width, m_Impl->MainColor.Height}};
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(m_Impl->MainColor.Width), static_cast<float>(m_Impl->MainColor.Height), 0.0f, 1.0f};
    m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
    m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
    m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lightingPipeline);
    const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(offset);
    m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_Impl->DeferredPipelineLayout, 0, 1, &lightingDescriptor, 1, &dynamicOffset);
    m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,m_Impl->DeferredPipelineLayout,FogDescriptorSetIndex,1,&m_Impl->FogDescriptorSets[Impl::FogDisabledSet],1,&m_Impl->FogUniformOffsets[Impl::FogDisabledSet]);
    if (!m_Impl->BindCascades(m_Impl->DeferredPipelineLayout, desc.SamplesShadowMap))
    {
        m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
        return false;
    }
    m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
    m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);

        if (m_Impl->SubsurfaceProducing)
        { ++m_Impl->PendingFrame.Statistics.SubsurfacePassCount; return true; }
        m_Impl->SubsurfaceLighting=desc;
        m_Impl->SubsurfaceHasScene=desc.Output==DeferredOutputMode::Scene;
    PassDesc pass;
    pass.Type = PassType::DeferredLighting;
    pass.ClearColorEnabled = pass.ClearDepthEnabled = false;
    m_Impl->PendingFrame.Passes.push_back({pass, {}});
    ++m_Impl->PendingFrame.Statistics.DeferredLightingPassCount;
    m_Impl->DeferredResolved = true;
    return true;
}

bool VulkanRHI::EndPass()
{
    if (m_Impl->FrameOpen && m_Impl->PassOpen && m_Impl->Hair.Active && !m_Impl->Hair.Frame.Geometry->Indices.empty() &&
        HairFurReplayPass(m_Impl->CurrentPass.Type) && (m_Impl->CurrentPass.Type!=PassType::PlanarCapture || m_Impl->FramePlanar.Active()) &&
        m_Impl->Hair.ReplayedPass!=m_Impl->PendingFrame.Passes.size()) return m_Impl->Fail("Complete requested hair replay before ending its native geometry pass");
    if (!m_Impl->CheckThread("EndPass") || !m_Impl->PassOpen)
        return m_Impl->Fail("Vulkan has no open pass");
    m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
    if (m_Impl->CurrentPass.DepthPeelingLayer && !m_Impl->PeelingSequence.Advance(DepthPeelingEvent::EndShading,m_Impl->LastError)) return false;
    m_Impl->PassOpen = false;
    if (m_Impl->CurrentPass.Type == PassType::EnvironmentCapture)
    {
        m_Impl->EnvironmentFacesWritten |= 1u << m_Impl->CurrentPass.EnvironmentFaceIndex;
        ++m_Impl->PendingFrame.Statistics.EnvironmentCaptureFaceCount;
    }
    if (m_Impl->CurrentPass.Type == PassType::GBuffer) m_Impl->GBufferWritten = true;
    if (m_Impl->CurrentPass.Type == PassType::ContactDepth) m_Impl->ContactDepthWritten = true;
    if (m_Impl->CurrentPass.Type == PassType::ForwardPlusDepth) m_Impl->ForwardPlusDepthWritten = true;
    if (m_Impl->CurrentPass.Type == PassType::PlanarCapture)
    {
        m_Impl->PlanarWritten = true; ++m_Impl->PendingFrame.Statistics.PlanarCapturePassCount;
    }
    if (m_Impl->CurrentPass.Type == PassType::Velocity) m_Impl->VelocityCaptured = true;
    if (m_Impl->WeightedPass) m_Impl->WeightedAccumulated = true;
    if (m_Impl->CurrentPass.Type == PassType::Color && !m_Impl->CurrentPass.Target)
    {
        if (!m_Impl->DeferredActive) m_Impl->ForwardOpaqueComplete = true;
        if (m_Impl->FrameSampleCount > 1 && m_Impl->SamplesOpaqueResolved)
        {
            m_Impl->SamplesTransparentWritten = true;
            m_Impl->SamplesColorResolved = false;
        }
    }
    if (m_Impl->CurrentPass.Type == PassType::Shadow)
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *m_Impl->ActiveDepth,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (m_Impl->UsesVSM() && !m_Impl->FilterMoments(m_Impl->CurrentPass.Target)) return false;
    }
    m_Impl->PassOpen = false;
    m_Impl->ActiveColor = nullptr;
    m_Impl->ActiveDepth = nullptr;
    m_Impl->ActiveFramebuffer = 0;
    m_Impl->WeightedPass = false;
    return true;
}

bool VulkanRHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
        return m_Impl->Fail("Vulkan ApplyAmbientOcclusion requires resolved multisample inputs");
    if (!m_Impl->CheckThread("ApplyAmbientOcclusion") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->AmbientOcclusionActive || m_Impl->AmbientOcclusionResolved || m_Impl->ReflectionResolved || m_Impl->DepthOfFieldResolved || m_Impl->VelocityCaptured || !m_Impl->GBufferWritten ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete))
        return m_Impl->Fail("Vulkan screen-space AO requires completed material/opaque lighting and runs once before transparency");
    using Pass = Techniques::AmbientOcclusionPassKind;
    const auto passes = Techniques::BuildAmbientOcclusionPlan(m_Impl->FrameAmbientOcclusion,
        m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height);
    RayTracedAOConstants packet{};
    auto& constants=packet.Surface;
    const bool rayTraced=m_Impl->FrameAmbientOcclusion.Mode==Techniques::AmbientOcclusionMode::RayTraced;
    if (!BuildAmbientOcclusionFrameConstants(desc, m_Impl->FrameAmbientOcclusion, m_Impl->FrameRayScene.get(),
        m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, passes.front(), packet))
        return m_Impl->Fail("Vulkan screen-space AO requires a finite, invertible camera with a valid forward direction");

    // Scene depth is sampled without a bound depth attachment. Transparency's
    // BeginPass restores it to attachment layout after ambient subtraction.
    m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->MainDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->GBufferImages[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->GBufferImages[3], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const bool scalable = m_Impl->FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::SAO;
    std::uint32_t destinationMip = 0;
    for (const auto pass : passes)
    {
        const bool composite = pass == Pass::Composite;
        const bool depthStage = pass == Pass::DepthPrepare || pass == Pass::DepthReduce;
        const std::uint32_t width = depthStage ? std::max(1U, m_Impl->MainColor.Width >> destinationMip) : m_Impl->MainColor.Width;
        const std::uint32_t height = depthStage ? std::max(1U, m_Impl->MainColor.Height >> destinationMip) : m_Impl->MainColor.Height;
        const std::size_t destinationIndex = pass == Pass::BilateralHorizontal ? 1U : 0U;
        auto* source = pass == Pass::Estimate || depthStage ? &m_Impl->DummyColor :
            &m_Impl->AmbientOcclusionImages[pass == Pass::BilateralVertical ? 1 : 0];
        auto* destination = depthStage ? &m_Impl->SaoScratch : composite ? &m_Impl->HdrImages[0] :
            &m_Impl->AmbientOcclusionImages[destinationIndex];
        auto* depth = scalable && pass != Pass::DepthPrepare ? &m_Impl->SaoDepthHierarchy : &m_Impl->MainDepth;
        const std::size_t offset = AlignUp(m_Impl->UniformOffset, UniformAlignment);
        // Existing descriptors expose the scene-uniform range. Reserve its full
        // size; AO uses the surface prefix, with scene counts appended for rays.
        if (offset > UniformBufferSize || sizeof(RealtimeDrawConstants) > UniformBufferSize - offset ||
            (!rayTraced && m_Impl->DescriptorSetIndex >= m_Impl->DescriptorSets.size()))
            return m_Impl->Fail("Vulkan screen-space AO per-frame arena is exhausted");
        constants.Options.X = static_cast<std::uint32_t>(pass);
        constants.Options.Y = depthStage ? destinationMip : 0;
        constants.Options.Z = scalable ? m_Impl->SaoDepthHierarchy.MipLevels - 1 : 0;
        std::memcpy(static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped) + offset, &packet, sizeof(packet));
        m_Impl->UniformOffset = offset + sizeof(RealtimeDrawConstants);
        const VkDescriptorSet descriptor = rayTraced ? m_Impl->RayAODescriptors[static_cast<std::size_t>(pass)]
            : m_Impl->DescriptorSets[m_Impl->DescriptorSetIndex++];
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (rayTraced) m_Impl->UpdateRayAODescriptor(descriptor, {depth->View, m_Impl->GBufferImages[1].View,
            m_Impl->GBufferImages[3].View, source->View});
        else m_Impl->UpdateDescriptorSet(descriptor, {depth->View, m_Impl->GBufferImages[1].View,
            m_Impl->GBufferImages[3].View, source->View}, m_Impl->DeferredSampler);

        VkClearValue clear {};
        VkRenderPassBeginInfo begin {};
        begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass = depthStage ? m_Impl->SaoDepthRenderPass : composite ?
            m_Impl->AmbientOcclusionCompositeRenderPass : m_Impl->HdrRenderPass;
        begin.framebuffer = depthStage ? m_Impl->SaoScratchFramebuffer : composite ?
            m_Impl->HdrFramebuffers[0] : m_Impl->AmbientOcclusionFramebuffers[destinationIndex];
        begin.renderArea = {{0, 0}, {width, height}};
        begin.clearValueCount = 1;
        begin.pClearValues = &clear;
        m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport {0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
        m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
        m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
        const auto& pipelines = rayTraced ? m_Impl->RayAOPipelines : m_Impl->FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::GTAO
            ? m_Impl->GtaoPipelines : m_Impl->FrameAmbientOcclusion.Mode == Techniques::AmbientOcclusionMode::HBAO
                ? m_Impl->HbaoPipelines : m_Impl->SsaoPipelines;
        m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            scalable ? m_Impl->SaoPipelines[depthStage ? 2 : composite ? 1 : 0] : pipelines[composite ? 1 : 0]);
        const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(offset);
        m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            rayTraced ? m_Impl->RayAOPipelineLayout : m_Impl->PipelineLayout, 0, 1, &descriptor, 1, &dynamicOffset);
        m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
        m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
        if (depthStage)
        {
            // All levels change layout together, so the existing speculative
            // image journal also restores an interrupted hierarchy build.
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->SaoScratch, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->SaoDepthHierarchy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageCopy copy {};
            copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, destinationMip, 0, 1};
            copy.extent = {width, height, 1};
            m_Impl->CmdCopyImage(m_Impl->CommandBuffer, m_Impl->SaoScratch.Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_Impl->SaoDepthHierarchy.Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->SaoDepthHierarchy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            ++destinationMip;
        }
        m_Impl->PendingFrame.AmbientOcclusionPasses.push_back(pass);
        ++m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount;
    }
    m_Impl->AmbientOcclusionResolved = true;
    return true;
}

bool VulkanRHI::CompleteEnvironmentCapture()
{
    constexpr std::uint32_t CompleteFaces = (1u << EnvironmentFaceCount) - 1u;
    if (!m_Impl->CheckThread("CompleteEnvironmentCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->EnvironmentActive || m_Impl->EnvironmentReady || (!m_Impl->FrameEnvironment.Source && m_Impl->EnvironmentFacesWritten != CompleteFaces))
        return m_Impl->Fail("Vulkan mip reduction requires all six completed capture faces exactly once");
    if (const auto& source=m_Impl->FrameEnvironment.Source)
    {
        auto& environment=m_Impl->Environment;
        const auto faceBytes=static_cast<VkDeviceSize>(source->Resolution)*source->Resolution*4*sizeof(std::uint16_t);
        if (!environment.Upload.Buffer && !m_Impl->CreateRawBuffer(faceBytes*EnvironmentFaceCount,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,environment.Upload)) return false;
        std::array<VkBufferImageCopy,EnvironmentFaceCount> copies{};
        for (std::uint32_t face=0;face<EnvironmentFaceCount;++face)
        {
            std::memcpy(static_cast<std::uint8_t*>(environment.Upload.Mapped)+face*faceBytes,source->FacesRGBA16[face].data(),faceBytes);
            copies[face].bufferOffset=face*faceBytes;
            copies[face].imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,face,1};
            copies[face].imageExtent={source->Resolution,source->Resolution,1};
        }
        m_Impl->TransitionImage(m_Impl->CommandBuffer,environment.Cube,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        m_Impl->CmdCopyBufferToImage(m_Impl->CommandBuffer,environment.Upload.Buffer,environment.Cube.Image,environment.Cube.Layout,
            static_cast<std::uint32_t>(copies.size()),copies.data());
    }
    if (m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude)
    {
        if (!m_Impl->ConvertEnvironmentMap()) return false;
        m_Impl->EnvironmentReady = true;
        return true;
    }
    auto& target = m_Impl->Environment;
    for (std::uint32_t mip = 1; mip < target.Cube.MipLevels; ++mip)
    {
        const auto extent = target.Cube.Width >> mip;
        for (std::uint32_t face = 0; face < EnvironmentFaceCount; ++face)
        {
            m_Impl->TransitionImage(m_Impl->CommandBuffer, target.Cube, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, target.Scratch, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            CubemapReflectionConstants constants {}; constants.Options.Z = face;
            std::uint32_t offset = 0;
            if (!m_Impl->UploadCubeConstants(constants, offset)) return false;
            const auto descriptor = Impl::CubeMipDescriptorBase + (mip - 1) * static_cast<std::uint32_t>(EnvironmentFaceCount) + face;
            std::array<VkImageView, Impl::ReflectionImageCount> views; views.fill(m_Impl->DummyColor.View);
            m_Impl->UpdateReflectionDescriptor(descriptor, views, target.Cube.View, target.ParentMipViews[mip - 1]);
            VkClearValue clear {};
            VkRenderPassBeginInfo begin {};
            begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            begin.renderPass = m_Impl->HdrRenderPass; begin.framebuffer = target.ScratchFramebuffer;
            begin.renderArea = {{0, 0}, {extent, extent}};
            begin.clearValueCount = 1; begin.pClearValues = &clear;
            m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport {0, 0, float(extent), float(extent), 0, 1};
            m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
            m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
            m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->CubeMipPipeline);
            m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->ReflectionPipelineLayout,
                0, 1, &m_Impl->ReflectionDescriptors[descriptor], 1, &offset);
            m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
            m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
            // Separate reduction and copy retain one layout per whole image.
            // The ordinary cancellation journal can therefore restore every barrier.
            m_Impl->TransitionImage(m_Impl->CommandBuffer, target.Scratch, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, target.Cube, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageCopy copy {};
            copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, face, 1};
            copy.extent = {extent, extent, 1};
            m_Impl->CmdCopyImage(m_Impl->CommandBuffer, target.Scratch.Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                target.Cube.Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            ++m_Impl->PendingFrame.Statistics.EnvironmentMipPassCount;
        }
    }
    m_Impl->TransitionImage(m_Impl->CommandBuffer, target.Cube, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_Impl->EnvironmentReady = true;
    return true;
}

bool VulkanRHI::CompletePlanarCapture()
{
    if (!m_Impl->CheckThread("CompletePlanarCapture") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->PlanarActive || !m_Impl->PlanarWritten || m_Impl->PlanarReady)
        return m_Impl->Fail("Vulkan planar mip generation requires one completed capture exactly once per frame");
    auto& image = m_Impl->Planar.Image; auto& scratch = m_Impl->Planar.Scratch;
    for (std::uint32_t mip = 1; mip < image.MipLevels; ++mip)
    {
        const auto width = std::max(1u, image.Width >> mip), height = std::max(1u, image.Height >> mip);
        PlanarReflectionConstants constants {};
        constants.Viewport = {float(width), float(height), 1.0f / width, 1.0f / height};
        std::uint32_t offset = 0;
        if (!m_Impl->UploadPlanarConstants(constants, offset)) return false;
        m_Impl->TransitionImage(m_Impl->CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, scratch, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        // Every draw retains its descriptor and constants until submission.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateDescriptorSets.html
        const auto descriptor = Impl::PlanarMipDescriptorBase + mip - 1;
        std::array<VkImageView, Impl::ReflectionImageCount> views; views.fill(m_Impl->DummyColor.View);
        m_Impl->UpdateReflectionDescriptor(descriptor, views, 0, 0, 0, m_Impl->Planar.MipViews[mip - 1]);
        VkClearValue clear {};
        VkRenderPassBeginInfo begin {};
        begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass = m_Impl->HdrRenderPass; begin.framebuffer = m_Impl->Planar.ScratchFramebuffer;
        begin.renderArea = {{0, 0}, {width, height}}; begin.clearValueCount = 1; begin.pClearValues = &clear;
        m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport {0, 0, float(width), float(height), 0, 1};
        m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
        m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
        m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->PlanarMipPipeline);
        m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->ReflectionPipelineLayout,
            0, 1, &m_Impl->ReflectionDescriptors[descriptor], 1, &offset);
        m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
        m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
        // The separate scratch image permits whole-image layout tracking. All
        // speculative barriers participate in the existing AbortFrame journal.
        m_Impl->TransitionImage(m_Impl->CommandBuffer, scratch, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy copy {};
        copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1}; copy.extent = {width, height, 1};
        m_Impl->CmdCopyImage(m_Impl->CommandBuffer, scratch.Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image.Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        ++m_Impl->PendingFrame.Statistics.PlanarMipPassCount;
    }
    m_Impl->TransitionImage(m_Impl->CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_Impl->PlanarReady = true;
    return true;
}

bool VulkanRHI::ApplyPathTracing(const PathTracingDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyPathTracing") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->PathFrame.Active || m_Impl->PathFrame.Resolved || !m_Impl->HdrActive || m_Impl->ForwardOpaqueComplete)
        return m_Impl->Fail("Path tracing requires an untouched HDR frame with primary-ray visibility");
    return m_Impl->TracePaths(desc);
}

bool VulkanRHI::ApplyGlobalIllumination(const GlobalIlluminationDesc& desc)
{
    if (m_Impl->FrameOpen && m_Impl->EarlyTemporalActive && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ApplyGlobalIllumination") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->GIActive || m_Impl->GI.Resolved || !m_Impl->HdrActive || !m_Impl->GBufferWritten || (m_Impl->EarlyTemporalActive && !m_Impl->VelocityCaptured) ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete) ||
        (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || (m_Impl->EarlyTemporalActive && !m_Impl->SamplesVelocityResolved))) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) || m_Impl->ReflectionResolved ||
        m_Impl->FogResolved || m_Impl->LightShaftResolved || m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved)
        return m_Impl->Fail("Vulkan SSGI requires opaque materials/depth/velocity before reflections and volumes");
    if (Techniques::UsesLumen(m_Impl->FrameGI)) return m_Impl->ApplyLumenGI(desc);
        if (m_Impl->FrameVoxelGI.Volume) return m_Impl->ApplyVoxelGI(desc);
        if (Techniques::UsesRayTracedGI(m_Impl->FrameGI)) return m_Impl->ApplyRayGI(desc);
        if (m_Impl->LightmapActive) return m_Impl->ResolveLightmaps(desc);
    if (m_Impl->ProbeActive) return m_Impl->ResolveProbes(desc);
    GlobalIlluminationConstants constants;
    if (!BuildGlobalIlluminationConstants(desc, m_Impl->FrameGI, m_Impl->FrameReflectionShading,
        m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->GI.Valid, constants))
        return m_Impl->Fail("Vulkan SSGI requires invertible current and previous cameras");
    auto& scene = m_Impl->HdrImages[0]; auto& textures = m_Impl->GIImages;
    const unsigned read = m_Impl->GI.ReadIndex, write = m_Impl->GI.WriteIndex();
    using Phase = Techniques::GlobalIlluminationPassKind;
    for (auto phase : {Phase::Trace, Phase::Spatial, Phase::Temporal, Phase::Composite})
    {
        const auto phaseIndex = static_cast<std::uint32_t>(phase);
        constants.Options.X = phaseIndex;
        const std::size_t offset = AlignUp(m_Impl->UniformOffset, UniformAlignment);
        if (offset > UniformBufferSize || sizeof(constants) > UniformBufferSize - offset)
            return m_Impl->Fail("Vulkan SSGI per-frame uniform arena is exhausted");
        std::memcpy(static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
        m_Impl->UniformOffset = offset + sizeof(constants);
        auto* estimate = phase == Phase::Spatial ? &textures[Impl::GIRaw] : phase == Phase::Temporal ?
            &textures[Impl::GIFiltered] : phase == Phase::Composite ? &textures[Impl::GIColorFirst + write] : &scene;
        const std::array<Impl::RawImage*, Impl::GIInputCount> inputs {&scene, &m_Impl->MainDepth, &m_Impl->GBufferImages[1],
            &m_Impl->GBufferImages[0], &m_Impl->GBufferImages[3], &m_Impl->MotionImages[Impl::MotionVelocity], estimate,
            &textures[Impl::GIColorFirst + read], &textures[Impl::GIDepthFirst + read], &textures[Impl::GINormalFirst + read]};
        std::array<VkDescriptorImageInfo, Impl::GIInputCount + 1> images {};
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            m_Impl->TransitionImage(m_Impl->CommandBuffer, *inputs[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            images[i] = {0, inputs[i]->View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
        images[Impl::GIInputCount].sampler = m_Impl->GISampler;
        const VkDescriptorBufferInfo buffer {m_Impl->UniformBuffer.Buffer, 0, sizeof(constants)};
        std::array<VkWriteDescriptorSet, Impl::GIInputCount + 2> writes {};
        const auto descriptor = m_Impl->GIDescriptorSets[phaseIndex];
        for (std::uint32_t i = 0; i < writes.size(); ++i)
        {
            auto& binding = writes[i]; binding.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            binding.dstSet = descriptor; binding.dstBinding = i; binding.descriptorCount = 1;
            binding.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC :
                i == Impl::GIInputCount + 1 ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            if (i == 0) binding.pBufferInfo = &buffer; else binding.pImageInfo = &images[i - 1];
        }
        // Four stable sets prevent a later phase from changing descriptors used
        // by an earlier recorded draw. BeginFrame waits before reusing any set.
        m_Impl->UpdateDescriptorSets(m_Impl->Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        const auto target = phase == Phase::Trace ? Impl::GIRaw : phase == Phase::Spatial ? Impl::GIFiltered :
            phase == Phase::Temporal ? Impl::GIColorFirst + write : Impl::GIComposite;
        const std::size_t outputIndices[] {target, Impl::GIDepthFirst + write, Impl::GINormalFirst + write};
        const unsigned outputCount = phase == Phase::Temporal ? Impl::GIAttachmentCount : 1;
        for (unsigned i = 0; i < outputCount; ++i)
            m_Impl->TransitionImage(m_Impl->CommandBuffer, textures[outputIndices[i]], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const VkClearValue clears[Impl::GIAttachmentCount] {};
        VkRenderPassBeginInfo begin {}; begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass = phase == Phase::Temporal ? m_Impl->GIRenderPass : m_Impl->HdrRenderPass;
        begin.framebuffer = m_Impl->GIFramebuffers[target]; begin.renderArea = {{0, 0}, {scene.Width, scene.Height}};
        begin.clearValueCount = outputCount; begin.pClearValues = clears;
        m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport {0, 0, float(scene.Width), float(scene.Height), 0, 1};
        m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
        m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
        m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            phase == Phase::Temporal ? m_Impl->GITemporalPipeline : m_Impl->GIPipeline);
        const auto dynamicOffset = static_cast<std::uint32_t>(offset);
        m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_Impl->GIPipelineLayout, 0, 1, &descriptor, 1, &dynamicOffset);
        m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0); m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
    }
    auto& composite = textures[Impl::GIComposite];
    m_Impl->TransitionImage(m_Impl->CommandBuffer, composite, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, scene, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy {}; copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {scene.Width, scene.Height, 1};
    m_Impl->CmdCopyImage(m_Impl->CommandBuffer, composite.Image, composite.Layout, scene.Image, scene.Layout, 1, &copy);
    m_Impl->GI.Resolved = true;
    m_Impl->PendingFrame.Temporal = desc.Sampling;
    m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount = Impl::GIPhaseCount;
    return true;
}

bool VulkanRHI::ApplyReflections(const ReflectionDesc& desc)
{
        if (desc.SkyboxOnly)
        {
            const auto saved=m_Impl->FrameReflections;
            const bool active=m_Impl->ReflectionActive,done=m_Impl->ReflectionResolved,planar=m_Impl->PlanarActive;
            m_Impl->FrameReflections.Mode=Techniques::ReflectionMode::Cubemap;
            m_Impl->FrameReflections.EnvironmentIntensity=0;
            m_Impl->FrameReflections.Planar.Enabled=false;
            m_Impl->ReflectionActive=true; m_Impl->ReflectionResolved=false; m_Impl->PlanarActive=false;
            auto request=desc; request.SkyboxOnly=false; request.DrawSkybox=true;
            const bool success=m_Impl->EnsureReflectionResources(false) && ApplyReflections(request);
            m_Impl->FrameReflections=saved; m_Impl->ReflectionActive=active; m_Impl->ReflectionResolved=done; m_Impl->PlanarActive=planar;
            if (success) m_Impl->PendingFrame.Statistics.SkyboxPassCount+=2;
            return success;
        }

    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ApplyReflections")) return false;
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
        return m_Impl->Fail("Vulkan SSR requires resolved sample radiance, depth and materials");
    if (!m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->ReflectionActive || m_Impl->ReflectionResolved ||
        (m_Impl->EnvironmentActive && !m_Impl->EnvironmentReady) ||
        (m_Impl->PlanarActive && !m_Impl->PlanarReady) ||
        !m_Impl->HdrActive || !m_Impl->GBufferWritten || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
        (m_Impl->GIActive && !m_Impl->GI.Resolved) || m_Impl->DepthOfFieldResolved ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved))
        return m_Impl->Fail("Vulkan SSR requires opaque lighting/materials and AO, once before velocity and transparency");
    if (Techniques::UsesLumenReflections(m_Impl->FrameReflections)) return m_Impl->ApplyLumenReflections(desc);
        if (m_Impl->FrameReflections.Mode==Techniques::ReflectionMode::RayTraced)
        return Techniques::UsesHybridRendering(m_Impl->FrameRayRadianceSettings.Pipeline) ? m_Impl->ApplyHybridRendering(desc) : m_Impl->ApplyRayReflections(desc);
    for (const bool planarPass : {false,true})
    {
        if (planarPass ? !m_Impl->PlanarActive : m_Impl->FrameReflections.Mode==Techniques::ReflectionMode::None) continue;
        ReflectionConstants constants {};
        CubemapReflectionConstants cubeConstants {};
        EnvironmentMapReflectionConstants mapConstants {};
        PlanarReflectionConstants planarConstants {};
        const bool cube = !planarPass && m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::Cubemap;
        const bool latitudeLongitude = !planarPass && m_Impl->FrameReflections.Mode == Techniques::ReflectionMode::LatitudeLongitude;
        const bool planar = planarPass;
        if (!m_Impl->EnvironmentActive && !planar && !BuildReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
            return m_Impl->Fail("Vulkan SSR requires a finite invertible projection");
        if (cube && !BuildCubemapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->FrameEnvironment.MipCount, cubeConstants))
            return m_Impl->Fail("Vulkan cubemap reflections require a finite invertible projection");
        if (latitudeLongitude && !BuildEnvironmentMapReflectionConstants(desc, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, m_Impl->EnvironmentMap.Image.MipLevels, mapConstants))
            return m_Impl->Fail("Vulkan environment map reflections require a finite invertible projection");
        if (planar && !BuildPlanarReflectionConstants(desc, m_Impl->FramePlanar, m_Impl->FrameReflections, m_Impl->FrameReflectionShading,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, planarConstants))
            return m_Impl->Fail("Vulkan planar lookup requires the frozen capture camera and main viewport");
        const auto firstOffset = AlignUp(m_Impl->UniformOffset, UniformAlignment);
        const auto secondOffset = AlignUp(firstOffset + Impl::ReflectionConstantBytes, UniformAlignment);
        if (secondOffset > UniformBufferSize || Impl::ReflectionConstantBytes > UniformBufferSize - secondOffset)
            return m_Impl->Fail("Vulkan SSR uniform arena is exhausted");
        const std::array<std::size_t, Impl::ReflectionPhaseCount> offsets {firstOffset, secondOffset};
        m_Impl->UniformOffset = secondOffset + Impl::ReflectionConstantBytes;
        m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->MainDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        for (std::size_t index = 0; index < GBufferAttachmentCount - 1; ++index)
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->GBufferImages[index], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        // Each stage owns a descriptor set and constant slice for this frame. No
        // descriptor referenced by recorded work is rewritten before submission.
        // Dummy inputs also keep the active attachment out of every sampled binding.
        for (const auto pass : {planar ? Techniques::ReflectionPassKind::PlanarLookup :
            Techniques::UsesEnvironmentCapture(m_Impl->FrameReflections) ? Techniques::ReflectionPassKind::EnvironmentLookup : Techniques::ReflectionPassKind::Trace,
            Techniques::ReflectionPassKind::Composite})
        {
            const bool composite = pass == Techniques::ReflectionPassKind::Composite;
            const std::uint32_t phase = composite ? 1 : 0;
            constants.Options.X = cubeConstants.Options.X = mapConstants.Options.X = planarConstants.Options.X = static_cast<std::uint32_t>(pass);
            auto* uniform = static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped) + offsets[phase];
            std::memcpy(uniform, &constants, sizeof(constants));
            if (cube) std::memcpy(uniform, &cubeConstants, sizeof(cubeConstants));
            if (latitudeLongitude) std::memcpy(uniform, &mapConstants, sizeof(mapConstants));
            if (planar) std::memcpy(uniform, &planarConstants, sizeof(planarConstants));
            auto& source = composite ? m_Impl->ReflectionImage : m_Impl->HdrImages[0];
            auto& destination = composite ? m_Impl->HdrImages[0] : m_Impl->ReflectionImage;
            m_Impl->TransitionImage(m_Impl->CommandBuffer, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            std::array<VkImageView, Impl::ReflectionImageCount> views;
            views.fill(m_Impl->DummyColor.View);
            if (composite) views.back() = source.View;
            else views = {source.View, m_Impl->MainDepth.View, m_Impl->GBufferImages[1].View, m_Impl->GBufferImages[2].View,
                m_Impl->GBufferImages[0].View, m_Impl->GBufferImages[3].View, m_Impl->DummyColor.View};
            const auto descriptor=phase+(desc.DrawSkybox ? Impl::SkyboxDescriptorBase : planar ? Impl::PlanarReflectionDescriptorBase : 0);
            m_Impl->UpdateReflectionDescriptor(descriptor, views, cube ? m_Impl->Environment.Cube.View : 0, 0,
                planar ? m_Impl->Planar.Image.View : latitudeLongitude ? m_Impl->EnvironmentMap.Image.View : 0);
            VkClearValue clear {};
            VkRenderPassBeginInfo begin {};
            begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            begin.renderPass = composite ? m_Impl->ReflectionCompositeRenderPass : m_Impl->HdrRenderPass;
            begin.framebuffer = composite ? m_Impl->HdrFramebuffers[0] : m_Impl->ReflectionFramebuffer;
            begin.renderArea = {{0, 0}, {destination.Width, destination.Height}};
            begin.clearValueCount = 1; begin.pClearValues = &clear;
            m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport {0, 0, static_cast<float>(destination.Width), static_cast<float>(destination.Height), 0, 1};
            m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
            m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
            m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                planar ? m_Impl->PlanarPipelines[phase] : latitudeLongitude ? m_Impl->EnvironmentMapPipelines[phase] :
                    cube ? m_Impl->CubeReflectionPipelines[phase] : m_Impl->ReflectionPipelines[phase]);
            const auto dynamicOffset = static_cast<std::uint32_t>(offsets[phase]);
            m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->ReflectionPipelineLayout,
                0, 1, &m_Impl->ReflectionDescriptors[descriptor], 1, &dynamicOffset);
            m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
            m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
            m_Impl->PendingFrame.ReflectionPasses.push_back(pass);
        }
    }
    m_Impl->ReflectionResolved = true;
    m_Impl->PendingFrame.Statistics.ReflectionPassCount = static_cast<std::uint32_t>(m_Impl->PendingFrame.ReflectionPasses.size());
    return true;
}

bool VulkanRHI::ApplyLightShafts(const LightShaftDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ApplyLightShafts") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->HdrActive || !m_Impl->LightShaftActive || m_Impl->LightShaftResolved || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
        (m_Impl->GIActive && !m_Impl->GI.Resolved) ||
        m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
        (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) || (m_Impl->FogActive && !m_Impl->FogResolved) ||
        (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
        return m_Impl->Fail("Vulkan light shafts require resolved opaque lighting, AO, reflections and fog before temporal effects and transparency");
    LightShaftConstants constants {};
    if (!BuildLightShaftConstants(desc, m_Impl->FrameLightShafts, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
        return m_Impl->Fail("Vulkan light shafts require a finite camera and selected light");
    const auto dummy = m_Impl->DummyColor.View;
    constants.Control.Y = 0;
    if (!m_Impl->WriteLightShaftSet(0, constants, {dummy, m_Impl->MainDepth.View, dummy})) return false;
    constants.Control.Y = 1;
    if (!m_Impl->WriteLightShaftSet(1, constants,
            {m_Impl->HdrImages[0].View, m_Impl->MainDepth.View, m_Impl->LightShaftMask.View})) return false;

    m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->MainDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (std::size_t phase = 0; phase < Impl::LightShaftPhaseCount; ++phase)
    {
        const bool mask = phase == 0;
        auto& target = mask ? m_Impl->LightShaftMask : m_Impl->HdrImages[1];
        if (!mask)
        {
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->LightShaftMask, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->HdrImages[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        m_Impl->TransitionImage(m_Impl->CommandBuffer, target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkClearValue clear {};
        VkRenderPassBeginInfo begin {};
        begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass = m_Impl->HdrRenderPass;
        begin.framebuffer = mask ? m_Impl->LightShaftFramebuffer : m_Impl->HdrFramebuffers[1];
        begin.renderArea = {{0, 0}, {target.Width, target.Height}};
        begin.clearValueCount = 1; begin.pClearValues = &clear;
        m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport {0.0f, 0.0f, static_cast<float>(target.Width), static_cast<float>(target.Height), 0.0f, 1.0f};
        m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
        m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
        m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->LightShaftPipeline);
        m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->LightShaftPipelineLayout,
            0, 1, &m_Impl->LightShaftSets[phase], 1, &m_Impl->LightShaftOffsets[phase]);
        m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
        m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
    }
    // Copy preserves HDR radiance and coverage alpha. Every speculative image
    // transition, including the new mask, belongs to the AbortFrame journal.
    auto& scratch = m_Impl->HdrImages[1]; auto& scene = m_Impl->HdrImages[0];
    m_Impl->TransitionImage(m_Impl->CommandBuffer, scratch, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, scene, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy {};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {scene.Width, scene.Height, 1};
    m_Impl->CmdCopyImage(m_Impl->CommandBuffer, scratch.Image, scratch.Layout, scene.Image, scene.Layout, 1, &copy);
    m_Impl->LightShaftResolved = true;
    m_Impl->PendingFrame.Statistics.LightShaftPassCount += Impl::LightShaftPhaseCount;
    return true;
}

bool VulkanRHI::ApplyVolumetricFog(const VolumetricFogDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->LightShaftResolved) return m_Impl->Fail("Vulkan fog must precede light shafts");
    if (!m_Impl->CheckThread("ApplyVolumetricFog") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->HdrActive || !m_Impl->FogActive || m_Impl->FogResolved || (m_Impl->VelocityCaptured && !m_Impl->EarlyTemporalActive) ||
        (m_Impl->GIActive && !m_Impl->GI.Resolved) ||
        m_Impl->TemporalResolved || m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
        (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) ||
        (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved))
        return m_Impl->Fail("Vulkan fog requires resolved opaque lighting, AO and reflections, once before temporal effects and transparency");
    VolumetricFogConstants constants {};
    if (!BuildVolumetricFogConstants(desc, m_Impl->FrameFog, m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height,
            m_Impl->FogTextureLimit(), constants)) return m_Impl->Fail("Vulkan fog requires a finite invertible frustum");
    const bool lighting = (Techniques::UsesVolumetricLighting(m_Impl->FrameVolumeLighting) || Techniques::UsesBoundedMedium(m_Impl->FrameFog));
    if (lighting && !m_Impl->GenerateVolumeLighting(desc, constants)) return false;
    const auto dummy = m_Impl->DummyColor.View;
    if (!lighting && !m_Impl->WriteFogSet(Impl::FogIntegrationSet, constants, {dummy, dummy, dummy})) return false;
    constants.Options.X = 1;
    if (!m_Impl->WriteFogSet(Impl::FogCompositionSet, constants,
            {m_Impl->HdrImages[0].View, m_Impl->MainDepth.View, m_Impl->FogAtlas.View}) ||
        !m_Impl->WriteFogSet(Impl::FogSurfaceSet, constants, {dummy, dummy, m_Impl->FogAtlas.View})) return false;

    for (std::size_t phase = lighting ? 1U : 0U; phase < 2; ++phase)
    {
        const bool integrate = phase == 0;
        auto& target = integrate ? m_Impl->FogAtlas : m_Impl->HdrImages[1];
        if (!integrate)
        {
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->FogAtlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->HdrImages[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->MainDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        m_Impl->TransitionImage(m_Impl->CommandBuffer, target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkClearValue clear {};
        VkRenderPassBeginInfo begin {};
        begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass = m_Impl->HdrRenderPass;
        begin.framebuffer = integrate ? m_Impl->FogFramebuffer : m_Impl->HdrFramebuffers[1];
        begin.renderArea = {{0, 0}, {target.Width, target.Height}};
        begin.clearValueCount = 1; begin.pClearValues = &clear;
        m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport {0.0f, 0.0f, static_cast<float>(target.Width), static_cast<float>(target.Height), 0.0f, 1.0f};
        m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
        m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
        m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->CloudFogPipeline : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->MediaFogPipeline : m_Impl->FogPipeline);
        const auto set = integrate ? Impl::FogIntegrationSet : Impl::FogCompositionSet;
        m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->FogPipelineLayout,
            0, 1, &m_Impl->FogDescriptorSets[set], 1, &m_Impl->FogUniformOffsets[set]);
        m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
        m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
    }
    // Preserve float radiance and coverage alpha with a same-format image copy.
    // The shared transition journal restores CPU layouts when commands abort;
    // submitted commands retain their layouts until the next fence completes.
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImage.html
    auto& scratch = m_Impl->HdrImages[1]; auto& scene = m_Impl->HdrImages[0];
    m_Impl->TransitionImage(m_Impl->CommandBuffer, scratch, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, scene, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy {};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {scene.Width, scene.Height, 1};
    m_Impl->CmdCopyImage(m_Impl->CommandBuffer, scratch.Image, scratch.Layout, scene.Image, scene.Layout, 1, &copy);
    m_Impl->FogResolved = true;
    (Techniques::UsesVolumetricClouds(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount : Techniques::UsesBoundedMedium(m_Impl->FrameFog) ? m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount : m_Impl->PendingFrame.Statistics.VolumetricFogPassCount) += lighting ? 1 : 2;
    return true;
}

bool VulkanRHI::ApplyDepthOfField(const DepthOfFieldDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
    if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved) return m_Impl->Fail("Vulkan light shafts must precede depth of field");
    if (m_Impl->FogActive && !m_Impl->FogResolved) return m_Impl->Fail("Vulkan fog must precede depth of field");
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesOpaqueResolved)
        return m_Impl->Fail("Vulkan ApplyDepthOfField requires resolved multisample inputs");
    if (!m_Impl->CheckThread("ApplyDepthOfField") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->DepthOfFieldActive || m_Impl->DepthOfFieldResolved ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete) ||
        (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved) ||
        (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved) ||
        (m_Impl->MotionBlurActive && !m_Impl->MotionBlurResolved) ||
        (m_Impl->TemporalActive && !m_Impl->TemporalResolved))
        return m_Impl->Fail("Vulkan depth of field requires completed opaque lighting and AO, once before transparency");

    using Pass = Techniques::DepthOfFieldPassKind;
    const auto& settings = m_Impl->FramePostProcessing.DepthOfField;
    const auto passes = Techniques::BuildDepthOfFieldPlan(settings);
    DepthOfFieldConstants constants {};
    if (!BuildDepthOfFieldConstants(desc, settings, m_Impl->CreateInfo.Width,
            m_Impl->CreateInfo.Height, Pass::Gather, constants))
        return m_Impl->Fail("Vulkan depth of field requires a finite supported camera projection and lens");

    // Neither stage binds the sampled depth as an attachment. The composite
    // loads scene alpha and writes RGB only, then transparency restores depth.
    m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->MainDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (const auto pass : passes)
    {
        const bool composite = pass == Pass::Composite;
        auto& source = m_Impl->HdrImages[composite ? 1 : 0];
        auto& destination = m_Impl->HdrImages[composite ? 0 : 1];
        const std::size_t offset = AlignUp(m_Impl->UniformOffset, UniformAlignment);
        if (offset > UniformBufferSize || sizeof(RealtimeDrawConstants) > UniformBufferSize - offset ||
            m_Impl->DescriptorSetIndex >= m_Impl->DescriptorSets.size())
            return m_Impl->Fail("Vulkan depth of field per-frame arena is exhausted");
        constants.Options.X = static_cast<std::uint32_t>(pass);
        std::memcpy(static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
        m_Impl->UniformOffset = offset + sizeof(RealtimeDrawConstants);
        const VkDescriptorSet descriptor = m_Impl->DescriptorSets[m_Impl->DescriptorSetIndex++];
        m_Impl->TransitionImage(m_Impl->CommandBuffer, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        m_Impl->UpdateDescriptorSet(descriptor,
            {source.View, m_Impl->MainDepth.View, m_Impl->DummyColor.View, m_Impl->DummyColor.View}, m_Impl->DepthOfFieldSampler);

        VkClearValue clear {};
        VkRenderPassBeginInfo begin {};
        begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass = composite ? m_Impl->DepthOfFieldCompositeRenderPass : m_Impl->HdrRenderPass;
        // Load/clear variants are render-pass compatible with these depthless
        // HDR framebuffers. Both stages have disjoint input/output images.
        begin.framebuffer = m_Impl->HdrFramebuffers[composite ? 0 : 1];
        begin.renderArea = {{0, 0}, {destination.Width, destination.Height}};
        begin.clearValueCount = 1;
        begin.pClearValues = &clear;
        m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport {0.0f, 0.0f, static_cast<float>(destination.Width),
            static_cast<float>(destination.Height), 0.0f, 1.0f};
        m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
        m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
        m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_Impl->DepthOfFieldPipelines[composite ? 1 : 0]);
        const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(offset);
        m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_Impl->PipelineLayout, 0, 1, &descriptor, 1, &dynamicOffset);
        m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
        m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
        m_Impl->PendingFrame.DepthOfFieldPasses.push_back(pass);
        ++m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount;
    }
    m_Impl->DepthOfFieldResolved = true;
    return true;
}

bool VulkanRHI::ApplyTemporalEffects(const TemporalEffectsDesc& desc)
{
    if (!m_Impl->CheckThread("ApplyTemporalEffects") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->FramePostProcessing.TemporalEffects.Enabled || !m_Impl->HdrActive || m_Impl->ExposureHistory.Resolved)
        return m_Impl->Fail("Temporal exposure requires completed scene HDR, once before presentation");
    if (!m_Impl->OpaqueEffectsComplete() || !m_Impl->Hair.Complete() || !m_Impl->Hair.VelocityComplete() ||
        !(m_Impl->PathFrame.Active?m_Impl->PathFrame.Resolved:m_Impl->DeferredActive?m_Impl->DeferredResolved:m_Impl->ForwardOpaqueComplete) ||
        (m_Impl->FrameSampleCount>1 && !m_Impl->SamplesColorResolved))
        return m_Impl->Fail("Complete opaque effects and final multisample color before exposure metering");
    return m_Impl->ResolveExposure(desc);
}

bool VulkanRHI::ApplyTemporalAA(const TemporalAADesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
    if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved) return m_Impl->Fail("Vulkan light shafts must precede temporal reconstruction");
    if (m_Impl->FogActive && !m_Impl->FogResolved) return m_Impl->Fail("Vulkan fog must precede temporal reconstruction");
    if (!m_Impl->CheckThread("ApplyTemporalAA") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->HdrActive || !m_Impl->TemporalActive || !m_Impl->VelocityCaptured || m_Impl->TemporalResolved ||
        m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved))
        return m_Impl->Fail("Vulkan TAA requires completed opaque velocity once before blur/transparency");
    TemporalAAConstants constants {};
    if (!BuildTemporalAAConstants(desc, m_Impl->FramePostProcessing.TemporalAA, m_Impl->CreateInfo.Width,
            m_Impl->CreateInfo.Height, m_Impl->TemporalHistoryValid, constants))
        return m_Impl->Fail("Vulkan TAA requires a finite previous camera and jitter");
    const std::size_t offset = AlignUp(m_Impl->UniformOffset, UniformAlignment);
    if (offset > UniformBufferSize || sizeof(constants) > UniformBufferSize - offset)
        return m_Impl->Fail("Vulkan TAA per-frame uniform arena is exhausted");
    std::memcpy(static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
    m_Impl->UniformOffset = offset + sizeof(constants);
    const auto read = m_Impl->TemporalHistoryReadIndex;
    const auto write = 1 - read;
    const std::array<Impl::RawImage*, Impl::TemporalInputCount> inputs {&m_Impl->HdrImages[0], &m_Impl->MainDepth,
        &m_Impl->MotionImages[Impl::MotionVelocity], &m_Impl->TemporalColors[read], &m_Impl->TemporalDepths[read]};
    std::array<VkDescriptorImageInfo, Impl::TemporalInputCount> images {};
    constexpr std::size_t HistoryColorInput = 3;
    for (std::size_t index = 0; index < inputs.size(); ++index)
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, *inputs[index], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        images[index] = {m_Impl->TemporalSamplers[index == HistoryColorInput ? Impl::TemporalLinearSampler : Impl::TemporalPointSampler],
            inputs[index]->View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }
    const VkDescriptorBufferInfo buffer {m_Impl->UniformBuffer.Buffer, 0, sizeof(constants)};
    std::array<VkWriteDescriptorSet, Impl::TemporalInputCount + 1> writes {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_Impl->TemporalDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &buffer;
    for (std::uint32_t index = 0; index < inputs.size(); ++index)
    {
        auto& descriptor = writes[index + 1];
        descriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor.dstSet = m_Impl->TemporalDescriptorSet;
        descriptor.dstBinding = index + 1;
        descriptor.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor.descriptorCount = 1;
        descriptor.pImageInfo = &images[index];
    }
    // BeginFrame waits for the previous submission. This dedicated set is
    // written once for this frame, and no later effect mutates its bindings.
    m_Impl->UpdateDescriptorSets(m_Impl->Device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    auto& destination = m_Impl->TemporalColors[write];
    m_Impl->TransitionImage(m_Impl->CommandBuffer, destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->TemporalDepths[write], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const std::array<VkClearValue, Impl::TemporalAttachmentCount> clears {};
    VkRenderPassBeginInfo begin {};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = m_Impl->TemporalRenderPass;
    begin.framebuffer = m_Impl->TemporalFramebuffers[write];
    begin.renderArea = {{0, 0}, {destination.Width, destination.Height}};
    begin.clearValueCount = static_cast<std::uint32_t>(clears.size());
    begin.pClearValues = clears.data();
    m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport {0.0f, 0.0f, static_cast<float>(destination.Width), static_cast<float>(destination.Height), 0.0f, 1.0f};
    m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
    m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
    m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->TemporalPipeline);
    const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(offset);
    m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_Impl->TemporalPipelineLayout, 0, 1, &m_Impl->TemporalDescriptorSet, 1, &dynamicOffset);
    m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
    m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
    // Preserve opaque history before any later effects. Both images have the
    // same float format and extent; this transfer preserves linear HDR values.
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImage.html
    auto& scene = m_Impl->HdrImages[0];
    m_Impl->TransitionImage(m_Impl->CommandBuffer, destination, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, scene, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy {};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {destination.Width, destination.Height, 1};
    m_Impl->CmdCopyImage(m_Impl->CommandBuffer, destination.Image, destination.Layout, scene.Image, scene.Layout, 1, &copy);
    m_Impl->TemporalResolved = true;
    m_Impl->PendingFrame.Temporal = desc.Sampling;
    ++m_Impl->PendingFrame.Statistics.TemporalAAPassCount;
    return true;
}

bool VulkanRHI::ApplyMotionBlur(const MotionBlurDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
    if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved) return m_Impl->Fail("Vulkan light shafts must precede motion blur");
    if (m_Impl->FogActive && !m_Impl->FogResolved) return m_Impl->Fail("Vulkan fog must precede motion blur");
    if (m_Impl->FrameSampleCount > 1 && !m_Impl->SamplesVelocityResolved)
        return m_Impl->Fail("Vulkan ApplyMotionBlur requires resolved multisample inputs");
    if (!m_Impl->CheckThread("ApplyMotionBlur") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->HdrActive || !m_Impl->MotionBlurActive || !m_Impl->VelocityCaptured ||
        m_Impl->MotionBlurResolved || m_Impl->DepthOfFieldResolved ||
        (m_Impl->TemporalActive && !m_Impl->TemporalResolved) ||
        (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved))
        return m_Impl->Fail("Vulkan motion blur requires completed velocity, once before lens blur/transparency");
    MotionBlurConstants constants {};
    if (!BuildMotionBlurConstants(desc, m_Impl->FramePostProcessing.MotionBlur,
            m_Impl->CreateInfo.Width, m_Impl->CreateInfo.Height, constants))
        return m_Impl->Fail("Vulkan motion blur requires a finite camera and valid sample interval");
    using Pass = Techniques::MotionBlurPassKind;
    const auto passes = Techniques::BuildMotionBlurPlan(m_Impl->FramePostProcessing.MotionBlur);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->MainDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_Impl->TransitionImage(m_Impl->CommandBuffer, m_Impl->MotionImages[Impl::MotionVelocity], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (const auto pass : passes)
    {
        const bool composite = pass == Pass::Composite;
        const bool tiles = pass == Pass::TileMax || pass == Pass::NeighborMax;
        const auto tileIndex = pass == Pass::TileMax ? Impl::MotionTileMax : Impl::MotionNeighborMax;
        auto& destination = tiles ? m_Impl->MotionImages[tileIndex] : m_Impl->HdrImages[composite ? 0 : 1];
        auto& source = m_Impl->HdrImages[composite ? 1 : 0];
        auto& maximum = m_Impl->MotionImages[pass == Pass::NeighborMax ? Impl::MotionTileMax :
            (pass == Pass::Gather ? Impl::MotionNeighborMax : Impl::MotionVelocity)];
        const std::size_t offset = AlignUp(m_Impl->UniformOffset, UniformAlignment);
        if (offset > UniformBufferSize || sizeof(RealtimeDrawConstants) > UniformBufferSize - offset ||
            m_Impl->DescriptorSetIndex >= m_Impl->DescriptorSets.size())
            return m_Impl->Fail("Vulkan motion blur per-frame arena is exhausted");
        constants.Options.X = static_cast<std::uint32_t>(pass);
        std::memcpy(static_cast<std::uint8_t*>(m_Impl->UniformBuffer.Mapped) + offset, &constants, sizeof(constants));
        m_Impl->UniformOffset = offset + sizeof(RealtimeDrawConstants);
        const VkDescriptorSet descriptor = m_Impl->DescriptorSets[m_Impl->DescriptorSetIndex++];
        m_Impl->TransitionImage(m_Impl->CommandBuffer, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, maximum, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_Impl->TransitionImage(m_Impl->CommandBuffer, destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        m_Impl->UpdateDescriptorSet(descriptor, {source.View, m_Impl->MainDepth.View,
            m_Impl->MotionImages[Impl::MotionVelocity].View, maximum.View}, m_Impl->MotionSampler);
        VkClearValue clear {};
        VkRenderPassBeginInfo begin {};
        begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass = composite ? m_Impl->MotionCompositeRenderPass : m_Impl->HdrRenderPass;
        begin.framebuffer = tiles ? m_Impl->MotionFramebuffers[tileIndex] : m_Impl->HdrFramebuffers[composite ? 0 : 1];
        begin.renderArea = {{0, 0}, {destination.Width, destination.Height}};
        begin.clearValueCount = 1;
        begin.pClearValues = &clear;
        m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport {0.0f, 0.0f, static_cast<float>(destination.Width),
            static_cast<float>(destination.Height), 0.0f, 1.0f};
        m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
        m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &begin.renderArea);
        m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_Impl->MotionFilterPipelines[composite ? 1 : 0]);
        const std::uint32_t dynamicOffset = static_cast<std::uint32_t>(offset);
        m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_Impl->PipelineLayout, 0, 1, &descriptor, 1, &dynamicOffset);
        m_Impl->CmdDraw(m_Impl->CommandBuffer, 3, 1, 0, 0);
        m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
        m_Impl->PendingFrame.MotionBlurPasses.push_back(pass);
        ++m_Impl->PendingFrame.Statistics.MotionBlurPassCount;
    }
    m_Impl->MotionBlurResolved = true;
    return true;
}

bool VulkanRHI::ProcessDepthPeeling(DepthPeelingPass pass)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ProcessDepthPeeling") || !m_Impl->FrameOpen || m_Impl->PassOpen || !m_Impl->FramePeeling.Enabled ||
        !m_Impl->OpaqueEffectsComplete() || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete))
        return m_Impl->Fail("OIT requires completed opaque HDR processing outside a raster pass");
    if (!m_Impl->ProcessPeeling(pass)) return false;
    RecordDepthPeelingPass(m_Impl->FramePeeling,pass,m_Impl->PendingFrame.Statistics); return true;
}

bool VulkanRHI::ProcessRefraction(const RefractionDesc& desc)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (!m_Impl->CheckThread("ProcessRefraction") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->RefractionActive || m_Impl->RefractionResolved || m_Impl->FrameSampleCount!=1 ||
        !m_Impl->HdrActive || !m_Impl->OpaqueEffectsComplete() || !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete))
        return m_Impl->Fail("Refraction requires completed single-sample opaque HDR processing and executes once per frame");
    return m_Impl->ResolveRefraction(desc);
}

bool VulkanRHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind pass)
{
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    using Pass = Techniques::WeightedTransparencyPassKind;
    if (!m_Impl->CheckThread("ProcessWeightedTransparency") || !m_Impl->FrameOpen || m_Impl->PassOpen ||
        !m_Impl->WeightedActive || !m_Impl->HdrActive || !m_Impl->OpaqueEffectsComplete() ||
        !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete) ||
        (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved ||
            (m_Impl->NeedsSampleColorRestore() && !m_Impl->SamplesColorRestored))))
        return m_Impl->Fail("Vulkan weighted transparency requires completed opaque processing and restored sample color");
    if (pass == Pass::Prepare)
    {
        if (m_Impl->WeightedPrepared || m_Impl->SamplesTransparentWritten)
            return m_Impl->Fail("Vulkan prepare weighted statistics once before geometry");
    }
    else if (pass == Pass::Resolve)
    {
        if (!m_Impl->WeightedPrepared || !m_Impl->WeightedAccumulated || m_Impl->WeightedResolved)
            return m_Impl->Fail("Vulkan resolve weighted statistics once after geometry");
    }
    else return m_Impl->Fail("Unknown Vulkan weighted transparency stage");

    for (auto& image : m_Impl->Weighted.Images)
        m_Impl->TransitionImage(m_Impl->CommandBuffer, image, pass == Pass::Prepare ?
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    auto& scene = m_Impl->FrameSampleCount > 1 ? m_Impl->Multisample.Images[Impl::SampleColor] : m_Impl->HdrImages[0];
    auto& depth = m_Impl->FrameSampleCount > 1 ? m_Impl->Multisample.Depth : m_Impl->MainDepth;
    std::array<VkClearValue, 2> clears {};
    clears[1].color.float32[0] = 1.0f;
    VkRenderPassBeginInfo begin {};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderArea = {{0, 0}, {scene.Width, scene.Height}};
    if (pass == Pass::Prepare)
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        begin.renderPass = m_Impl->Weighted.GeometryPasses[1];
        begin.framebuffer = m_Impl->Weighted.GeometryFramebuffer;
        begin.clearValueCount = static_cast<std::uint32_t>(clears.size());
        begin.pClearValues = clears.data();
    }
    else
    {
        m_Impl->TransitionImage(m_Impl->CommandBuffer, scene, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        begin.renderPass = m_Impl->Weighted.ResolvePass;
        begin.framebuffer = m_Impl->Weighted.ResolveFramebuffer;
    }
    m_Impl->CmdBeginRenderPass(m_Impl->CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
    if (pass == Pass::Resolve)
    {
        const VkViewport viewport {0, 0, static_cast<float>(scene.Width), static_cast<float>(scene.Height), 0, 1};
        const VkRect2D scissor {{0, 0}, {scene.Width, scene.Height}};
        m_Impl->CmdSetViewport(m_Impl->CommandBuffer, 0, 1, &viewport);
        m_Impl->CmdSetScissor(m_Impl->CommandBuffer, 0, 1, &scissor);
        m_Impl->CmdBindPipeline(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Impl->Weighted.ResolvePipeline);
        m_Impl->CmdBindDescriptorSets(m_Impl->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_Impl->WeightedResolveLayout, 0, 1, &m_Impl->WeightedDescriptorSet, 0, nullptr);
        constexpr std::uint32_t FullscreenVertexCount = 3;
        m_Impl->CmdDraw(m_Impl->CommandBuffer, FullscreenVertexCount, 1, 0, 0);
        m_Impl->WeightedResolved = true;
    }
    else m_Impl->WeightedPrepared = true;
    m_Impl->CmdEndRenderPass(m_Impl->CommandBuffer);
    ++m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount;
    return true;
}

bool VulkanRHI::ProcessMultisampling(Techniques::MultisamplePassKind pass)
{
    if (pass==Techniques::MultisamplePassKind::OpaqueResolve && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before opaque MSAA resolution");
    using Pass = Techniques::MultisamplePassKind;
    using Phase = Impl::SamplePhase;
    if (!m_Impl->CheckThread("ProcessMultisampling") || !m_Impl->FrameOpen || m_Impl->PassOpen || m_Impl->FrameSampleCount <= 1)
        return m_Impl->Fail("Vulkan sample processing requires a multisample frame outside a raster pass");
    const bool material = m_Impl->Hair.Active || m_Impl->GIActive || m_Impl->DeferredActive || m_Impl->AmbientOcclusionActive || m_Impl->ReflectionActive;
    switch (pass)
    {
    case Pass::OpaqueResolve:
        if (m_Impl->SamplesOpaqueResolved || (material && !m_Impl->GBufferWritten) ||
            !(m_Impl->DeferredActive ? m_Impl->DeferredResolved : m_Impl->ForwardOpaqueComplete))
            return m_Impl->Fail("Vulkan resolve opaque samples once after opaque lighting");
        if (!m_Impl->DrawSampleResolve(Phase::Color) || !m_Impl->DrawSampleResolve(material ? Phase::Material : Phase::Depth)) return false;
        m_Impl->SamplesOpaqueResolved = m_Impl->SamplesColorResolved = true;
        break;
    case Pass::VelocityResolve:
        if (!m_Impl->SamplesOpaqueResolved || !m_Impl->VelocityCaptured || m_Impl->SamplesVelocityResolved ||
            m_Impl->TemporalResolved || m_Impl->MotionBlurResolved)
            return m_Impl->Fail("Vulkan resolve sample velocity once before motion blur");
        if (!m_Impl->DrawSampleResolve(Phase::Velocity)) return false;
        m_Impl->SamplesVelocityResolved = true;
        break;
    case Pass::ColorRestore:
        if (!m_Impl->SamplesOpaqueResolved || !m_Impl->NeedsSampleColorRestore() || !m_Impl->OpaqueEffectsComplete() ||
            m_Impl->SamplesColorRestored || m_Impl->SamplesTransparentWritten)
            return m_Impl->Fail("Vulkan restore processed opaque radiance once before sample transparency");
        if (!m_Impl->DrawSampleResolve(Phase::Restore)) return false;
        m_Impl->SamplesColorRestored = true;
        break;
    case Pass::ColorResolve:
        if (m_Impl->WeightedActive && !m_Impl->WeightedResolved)
            return m_Impl->Fail("Vulkan composite weighted transparency before the final sample average");
        if (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesTransparentWritten || m_Impl->SamplesColorResolved)
            return m_Impl->Fail("Vulkan final sample resolve requires transparent composition");
        if (!m_Impl->DrawSampleResolve(Phase::Color)) return false;
        m_Impl->SamplesColorResolved = true;
        break;
    default: return m_Impl->Fail("Unknown Vulkan sample resolve phase");
    }
    ++m_Impl->PendingFrame.Statistics.MultisamplePassCount;
    return true;
}

bool VulkanRHI::Present()
{
    if (!m_Impl->CheckThread("Lumen frame publication")) return false;
    Impl::LumenPresentPublication lumenPublication{*m_Impl};
    Impl::NanitePresentPublication nanitePublication{*m_Impl};
    if (!m_Impl->CheckThread("TSR present boundary")) return false;
    if (m_Impl->FrameOpen && m_Impl->TSRScene.Active && (!m_Impl->TSRScene.OpaqueReady || !m_Impl->VelocityCaptured || (m_Impl->FrameSampleCount>1 && !m_Impl->SamplesVelocityResolved))) return m_Impl->Fail("TSR presentation requires completed opaque snapshot and physical motion");

    if (m_Impl->FrameOpen && m_Impl->FramePostProcessing.TemporalEffects.Enabled && !m_Impl->ExposureHistory.Resolved)
        return m_Impl->Fail("Complete temporal exposure before presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.VelocityComplete()) return m_Impl->Fail("Complete fiber velocity before temporal consumers or presentation");
    if (m_Impl->FrameOpen && !m_Impl->Hair.Complete()) return m_Impl->Fail("Complete hair rasterization before downstream effects or presentation");
    if (m_Impl->FrameOpen && !m_Impl->SubsurfaceComplete()) return m_Impl->Fail("Complete subsurface diffusion before downstream effects or presentation");
    if (m_Impl->FrameOpen && m_Impl->PendingFrame.Statistics.SubsurfacePassCount && !m_Impl->SubsurfaceResolved)
        return m_Impl->Fail("Incomplete subsurface diffusion must be aborted before presentation");
    if (m_Impl->PathFrame.Active && !m_Impl->PathFrame.Resolved) return m_Impl->Fail("Path tracing must resolve before presentation");
        if (m_Impl->GIActive && !m_Impl->GI.Resolved) return m_Impl->Fail("GI must complete before temporal/lens effects and presentation");
    if (m_Impl->FramePeeling.Enabled && m_Impl->PeelingSequence.Stage()!=DepthPeelingStage::Complete)
        return m_Impl->Fail("Present requires completed OIT composition");
    if (m_Impl->RefractionActive && !m_Impl->RefractionResolved)
        return m_Impl->Fail("Present requires completed refraction composition");
    if (m_Impl->WeightedActive && !m_Impl->WeightedResolved)
        return m_Impl->Fail("Vulkan weighted transparency must resolve before presentation");
    if (!m_Impl->CheckThread("Present") || !m_Impl->FrameOpen || m_Impl->PassOpen)
        return m_Impl->Fail("Vulkan frame cannot be presented in the current state");
    if (m_Impl->LightListsActive && !m_Impl->LightListFrame.Assigned())
        return m_Impl->Fail("Vulkan Forward+ frame requires completed light assignment before presentation");
    if (m_Impl->FrameSampleCount > 1 && (!m_Impl->SamplesOpaqueResolved || !m_Impl->SamplesColorResolved))
        return m_Impl->Fail("Vulkan multisample color/depth must resolve before presentation");
    if (m_Impl->DeferredActive && !m_Impl->DeferredResolved)
        return m_Impl->Fail("Vulkan deferred frame requires completed lighting before presentation");
    if (m_Impl->AmbientOcclusionActive && !m_Impl->AmbientOcclusionResolved)
        return m_Impl->Fail("Vulkan screen-space AO frame requires completed ambient occlusion before presentation");
    if (m_Impl->ReflectionActive && !m_Impl->ReflectionResolved)
        return m_Impl->Fail("Vulkan SSR must complete before presentation");
    if (m_Impl->DepthOfFieldActive && !m_Impl->DepthOfFieldResolved)
        return m_Impl->Fail("Vulkan depth of field must resolve before presentation");
    if (m_Impl->MotionBlurActive && !m_Impl->MotionBlurResolved)
        return m_Impl->Fail("Vulkan motion blur must resolve before presentation");
    if (m_Impl->TemporalActive && !m_Impl->TemporalResolved)
        return m_Impl->Fail("Vulkan temporal reconstruction must resolve before presentation");
    if (m_Impl->FogActive && !m_Impl->FogResolved)
        return m_Impl->Fail("Vulkan fog must resolve before presentation");
    if (m_Impl->LightShaftActive && !m_Impl->LightShaftResolved)
        return m_Impl->Fail("Vulkan light shafts must resolve before presentation");
    Impl::RawBuffer diagnosticReadback;
    const int diagnosticIndex=Techniques::MaterialBufferIndex(GetDiagnosticBuffer());
    auto* diagnostic=diagnosticIndex>=0?&m_Impl->GBufferImages[diagnosticIndex]:GetDiagnosticBuffer()==Techniques::RenderBuffer::AmbientOcclusion?&m_Impl->AmbientOcclusionImages[0]:nullptr;
    if (diagnostic && diagnostic->Image)
    {
        const VkDeviceSize bytes=VkDeviceSize(m_Impl->CreateInfo.Width)*m_Impl->CreateInfo.Height*8;
        if (!m_Impl->CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,diagnosticReadback)) return false;
        m_Impl->CopyImageToReadback(*diagnostic,diagnosticReadback);
        m_Impl->TransitionImage(m_Impl->CommandBuffer,*diagnostic,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    const bool captureDepth = IsDepthCaptureRequired() || (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) ||
        (!IsGpuPresentationEnabled() && m_Impl->PendingFrame.Statistics.OutlineDrawCallCount > 0);
    Impl::RawImage* postColor = nullptr;
    const bool gpuPresentation=IsGpuPresentationEnabled() && !captureDepth && !m_Impl->TSRScene.Active && GetDiagnosticBuffer()==Techniques::RenderBuffer::Color;
    if (!m_Impl->TSRScene.Active)
    {
        if(!m_Impl->ApplyHdr(m_Impl->FramePostProcessing.HDR) || !m_Impl->ApplyPostProcess(m_Impl->FramePostProcessing,postColor))return false;
        if(gpuPresentation){if(!m_Impl->PublishGpuColor(*postColor))return false;}
        else m_Impl->CopyImageToReadback(*postColor,m_Impl->ColorReadback);
    }
    const bool captureMotion = m_Impl->VelocityActive && (m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color);
    if (captureMotion)
    {
        if (!m_Impl->MotionReadback.Buffer && !m_Impl->CreateRawBuffer(VkDeviceSize(m_Impl->CreateInfo.Width)*m_Impl->CreateInfo.Height*8,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,m_Impl->MotionReadback)) return false;
        m_Impl->CopyImageToReadback(m_Impl->MotionImages[Impl::MotionVelocity],m_Impl->MotionReadback);
    }
    if (captureDepth)
    {
        m_Impl->CopyImageToReadback(m_Impl->MainDepth, m_Impl->DepthReadback);
    }
    Impl::TextureResource* shadow = nullptr;
    Impl::VSMMomentMap* capturedMoments = nullptr;
    if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color) && m_Impl->LastShadowTexture)
    {
        const auto found = m_Impl->Textures.find(m_Impl->LastShadowTexture.Value);
        if (found != m_Impl->Textures.end())
        {
            shadow = &found->second;
            if (!m_Impl->EnsureShadowReadback(shadow->Image.Width, shadow->Image.Height)) return false;
            m_Impl->CopyImageToReadback(shadow->Image, m_Impl->ShadowReadback);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, shadow->Image,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (m_Impl->UsesVSM())
            {
                Impl::RawImage* image = nullptr;
                if (!m_Impl->MomentImage(m_Impl->LastShadowTexture, image)) return false;
                capturedMoments = &m_Impl->VSMMoments.at(m_Impl->LastShadowTexture.Value);
                const VkDeviceSize bytes = static_cast<VkDeviceSize>(image->Width) * image->Height * sizeof(Float2);
                auto& readback = capturedMoments->Readback;
                if (!readback.Buffer && !m_Impl->CreateRawBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, readback)) return false;
                m_Impl->CopyImageToReadback(*image, readback);
                m_Impl->TransitionImage(m_Impl->CommandBuffer, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }
    if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
    {
        for (std::uint32_t index = 0; index < m_Impl->FrameCascades.Frame.CascadeCount; ++index)
        {
            // Receiver descriptors may refer to RG32 moments. Diagnostics retain
            // the actual D32 caster coverage independently of the selected filter.
            auto& image = m_Impl->Textures.at(m_Impl->FrameCascades.Maps[index].Value).Image;
            auto& readback = m_Impl->CascadeReadbacks[index];
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(image.Width) * image.Height * sizeof(float);
            if (!readback.Buffer || readback.Size < bytes)
            {
                m_Impl->DestroyRawBuffer(readback);
                if (!m_Impl->CreateRawBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, readback)) return false;
            }
            m_Impl->CopyImageToReadback(image, readback);
            m_Impl->TransitionImage(m_Impl->CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }
    m_Impl->EndFrameTiming();
    if (!m_Impl->Check(m_Impl->EndCommandBuffer(m_Impl->CommandBuffer),
                       "vkEndCommandBuffer(frame)")) return false;
    m_Impl->FrameCommandEnded = true;
    if (!m_Impl->Check(m_Impl->ResetFences(m_Impl->Device, 1, &m_Impl->FrameFence),
                       "vkResetFences(frame)")) return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &m_Impl->CommandBuffer;
    m_Impl->AttachGpuSignal(submit);
    const VkResult submitResult =
        m_Impl->QueueSubmit(m_Impl->GraphicsQueue, 1, &submit, m_Impl->FrameFence);
    if (submitResult != VK_SUCCESS)
    {
        const std::string submitError = VulkanError("vkQueueSubmit(frame)", submitResult);
        (void)m_Impl->RecreateSignaledFrameFence();
        return m_Impl->Fail(submitError);
    }
    m_Impl->FrameSubmitted = true;
    if (m_Impl->ForwardPlusRecorded)
    {
        if (!(m_Impl->ClusteredActive ? m_Impl->ClusteredProducer->MarkSubmitted(m_Impl->FrameFence, m_Impl->LastError) :
            m_Impl->TileProducer()->MarkSubmitted(m_Impl->FrameFence, m_Impl->LastError))) return false;
        m_Impl->ForwardPlusSubmitted = true; m_Impl->ForwardPlusRecordOpen = false;
    }
    m_Impl->FrameImageLayouts.clear();
    if (!m_Impl->Check(m_Impl->WaitForFences(
            m_Impl->Device, 1, &m_Impl->FrameFence, VK_TRUE_VALUE, GpuWaitTimeout),
            "vkWaitForFences(frame completion)")) return false;
    m_Impl->CompleteGpuSignal();
    m_Impl->ReadFrameTiming();

    m_Impl->PendingFrame.Statistics.PassCount =
        static_cast<std::uint32_t>(m_Impl->PendingFrame.Passes.size()) +
        m_Impl->PendingFrame.Statistics.PostProcessPassCount +
        m_Impl->PendingFrame.Statistics.AmbientOcclusionPassCount + m_Impl->PendingFrame.Statistics.HairFurPassCount + m_Impl->PendingFrame.Statistics.SubsurfacePassCount + m_Impl->PendingFrame.Statistics.PathTracingPassCount + m_Impl->PendingFrame.Statistics.GlobalIlluminationPassCount + m_Impl->PendingFrame.Statistics.ReflectionPassCount +
        m_Impl->PendingFrame.Statistics.DepthOfFieldPassCount + m_Impl->PendingFrame.Statistics.MotionBlurPassCount +
        m_Impl->PendingFrame.Statistics.TemporalAAPassCount + m_Impl->PendingFrame.Statistics.TemporalEffectsPassCount + m_Impl->PendingFrame.Statistics.VolumetricFogPassCount +
        m_Impl->PendingFrame.Statistics.VolumetricLightingPassCount + m_Impl->PendingFrame.Statistics.ParticipatingMediaPassCount + m_Impl->PendingFrame.Statistics.VolumetricCloudPassCount + m_Impl->PendingFrame.Statistics.LightShaftPassCount +
        m_Impl->PendingFrame.Statistics.MultisamplePassCount + m_Impl->PendingFrame.Statistics.EnvironmentMipPassCount +
        m_Impl->PendingFrame.Statistics.EnvironmentConversionPassCount + m_Impl->PendingFrame.Statistics.PlanarMipPassCount +
        m_Impl->PendingFrame.Statistics.WeightedTransparencyPassCount + m_Impl->PendingFrame.Statistics.DepthPeelingPassCount +
        m_Impl->PendingFrame.Statistics.ShadowMomentFilterPassCount + m_Impl->PendingFrame.Statistics.ContactShadowPassCount +
        m_Impl->PendingFrame.Statistics.ForwardPlusAssignmentCount + m_Impl->PendingFrame.Statistics.ClusteredAssignmentCount + m_Impl->PendingFrame.Statistics.TiledDeferredAssignmentCount;
    m_Impl->PendingFrame.Width = static_cast<std::uint32_t>(m_Impl->CreateInfo.Width);
    m_Impl->PendingFrame.Height = static_cast<std::uint32_t>(m_Impl->CreateInfo.Height);
    const std::size_t pixelCount = static_cast<std::size_t>(m_Impl->CreateInfo.Width) *
                                   m_Impl->CreateInfo.Height;
    if (!gpuPresentation && !m_Impl->TSRScene.Active) { m_Impl->PendingFrame.ColorRGBA8.resize(pixelCount * 4U); std::memcpy(m_Impl->PendingFrame.ColorRGBA8.data(), m_Impl->ColorReadback.Mapped,m_Impl->PendingFrame.ColorRGBA8.size()); }
    if (diagnosticReadback.Buffer)
    {
        DecodeDiagnosticRows(diagnosticReadback.Mapped,std::size_t(m_Impl->CreateInfo.Width)*8,m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height,m_Impl->PendingFrame.DiagnosticPixels);
        m_Impl->DestroyRawBuffer(diagnosticReadback);
    }
    if (captureMotion) m_Impl->ReadMotionVectors();
    if (captureDepth)
    {
        m_Impl->PendingFrame.Depth32.resize(pixelCount);
        std::memcpy(m_Impl->PendingFrame.Depth32.data(), m_Impl->DepthReadback.Mapped,
                    m_Impl->PendingFrame.Depth32.size() * sizeof(float));
    }
    if (captureDepth && !IsValidDepthCapture(m_Impl->PendingFrame.Depth32))
        return m_Impl->Fail("Vulkan main depth readback is invalid");
    if (shadow != nullptr)
    {
        m_Impl->PendingFrame.ShadowWidth = shadow->Image.Width;
        m_Impl->PendingFrame.ShadowHeight = shadow->Image.Height;
        m_Impl->PendingFrame.ShadowDepth32.resize(
            static_cast<std::size_t>(shadow->Image.Width) * shadow->Image.Height);
        std::memcpy(m_Impl->PendingFrame.ShadowDepth32.data(), m_Impl->ShadowReadback.Mapped,
                    m_Impl->PendingFrame.ShadowDepth32.size() * sizeof(float));
        if (!IsValidDepthCapture(m_Impl->PendingFrame.ShadowDepth32))
            return m_Impl->Fail("Vulkan shadow depth readback is invalid");
    }
    if (capturedMoments)
    {
        const auto& image = capturedMoments->Images[Impl::VSMFiltered];
        auto& moments = m_Impl->PendingFrame.ShadowMoments;
        moments.resize(static_cast<std::size_t>(image.Width) * image.Height);
        std::memcpy(moments.data(), capturedMoments->Readback.Mapped, moments.size() * sizeof(Float2));
    }
    if ((m_Impl->CreateInfo.CaptureDiagnosticBuffers || GetDiagnosticBuffer()!=Techniques::RenderBuffer::Color))
    {
        m_Impl->PendingFrame.CascadeDepth32.resize(m_Impl->FrameCascades.Frame.CascadeCount);
        for (std::size_t index = 0; index < m_Impl->PendingFrame.CascadeDepth32.size(); ++index)
        {
            const auto& image = m_Impl->Textures.at(m_Impl->FrameCascades.Maps[index].Value).Image;
            auto& depth = m_Impl->PendingFrame.CascadeDepth32[index];
            depth.resize(static_cast<std::size_t>(image.Width) * image.Height);
            std::memcpy(depth.data(), m_Impl->CascadeReadbacks[index].Mapped, depth.size() * sizeof(float));
            if (!IsValidDepthCapture(depth)) return m_Impl->Fail("Vulkan cascade depth readback is invalid");
        }
    }
    if (!m_Impl->FinishTSRScene()) return false;
    m_Impl->CommitTSRScene();
    m_Impl->Hair.CommitPresent(m_Impl->CreateInfo.Width,m_Impl->CreateInfo.Height);
    m_Impl->ExposureHistory.Commit();
    ApplyDiagnosticCapture(m_Impl->PendingFrame,GetDiagnosticBuffer());
    m_Impl->LastFrame = std::move(m_Impl->PendingFrame);
    m_Impl->FrameNumber = m_Impl->LastFrame.Statistics.FrameNumber;
    m_Impl->FrameOpen = false;
    m_Impl->PresentToWindow(m_Impl->LastFrame);
    // Barriers track submitted GPU state independently of logical history.
    // Readback/presentation failure must not publish the pending image pair.
    if (m_Impl->TemporalResolved)
    {
        m_Impl->TemporalHistoryReadIndex = 1 - m_Impl->TemporalHistoryReadIndex;
        m_Impl->TemporalHistoryValid = true;
    }
    else m_Impl->TemporalHistoryValid = false;
    m_Impl->PathFrame.Commit();
        if (Techniques::UsesScreenSpaceGI(m_Impl->FrameGI)) m_Impl->GI.Commit(); else m_Impl->GI.Invalidate();
    if (m_Impl->ProbeActive) m_Impl->Probes.Commit();
    if (m_Impl->PendingUploadedLightmap) m_Impl->UploadedLightmap=std::move(m_Impl->PendingUploadedLightmap);
    lumenPublication.Published=true; nanitePublication.Published=true;
    return true;
}

FrameCapture VulkanRHI::GetLastFrameCapture() const { return m_Impl->LastFrame; }
FrameCapture VulkanRHI::TakeLastFrameCapture() { return std::move(m_Impl->LastFrame); }
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
    bool VulkanRHI::PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload&,VirtualGeometryCacheHandle,VirtualGeometryCacheHandle&,VirtualGeometryCacheStatistics&) { return false; }
bool VulkanRHI::PrepareNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheHandle&,NaniteCacheStatistics&) { return false; }
bool VulkanRHI::SubmitNaniteCache(const NaniteGpuUpload&,NaniteCacheHandle,NaniteCacheTask&,NaniteCacheStatistics&) { return false; }
bool VulkanRHI::PollNaniteCache(NaniteCacheTask,bool&,NaniteCacheHandle&) { return false; }
bool VulkanRHI::CancelNaniteCache(NaniteCacheTask) { return false; }
    bool VulkanRHI::DestroyVirtualGeometryCache(VirtualGeometryCacheHandle) { return false; }
bool VulkanRHI::DestroyNaniteCache(NaniteCacheHandle) { return false; }
    bool VulkanRHI::BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload&,VirtualShadowRequestResult&) { return false; }
    bool VulkanRHI::SampleVirtualShadowAtlas(VirtualShadowAtlasHandle,const VirtualShadowLookupUpload&,VirtualShadowLookupResult&) { return false; }
    bool VulkanRHI::RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload&,VirtualShadowAtlasHandle,VirtualShadowAtlasResult&) { return false; }
    bool VulkanRHI::ReadVirtualShadowAtlas(VirtualShadowAtlasHandle,std::vector<float>&) { return false; }
    bool VulkanRHI::DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle) { return false; }
    bool VulkanRHI::CreateLumenDistanceScene(const LumenDistanceSceneData&,LumenDistanceSceneHandle&) { return false; }
    bool VulkanRHI::CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle&,LumenDistanceSceneHandle) { return false; }
    bool VulkanRHI::BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle,const LumenGlobalDistanceSettings&,LumenDistanceSceneHandle,LumenGlobalDistanceTask&) { return false; }
    bool VulkanRHI::AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask,unsigned,bool& ready,LumenDistanceSceneHandle&,LumenGlobalDistanceStatistics&) { ready=false; return false; }
    bool VulkanRHI::CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask) { return false; }
    bool VulkanRHI::TraceLumenDistanceScene(LumenDistanceSceneHandle,const LumenDistanceTraceUpload&,LumenDistanceTraceResult&) { return false; }
    bool VulkanRHI::DestroyLumenDistanceScene(LumenDistanceSceneHandle) { return false; }
    bool VulkanRHI::RelightLumenSurface(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenSurfaceLightingResult&) { return false; }
    bool VulkanRHI::BeginLumenLightingUpdate(LumenSurfaceHandle,LumenDistanceSceneHandle,const LumenSurfaceLightingUpload&,LumenLightingTask&) { return false; }
    bool VulkanRHI::AdvanceLumenLightingUpdate(LumenLightingTask,unsigned,bool& ready,LumenSurfaceLightingResult&,LumenLightingStatistics&) { ready=false; return false; }
    bool VulkanRHI::CancelLumenLightingUpdate(LumenLightingTask) { return false; }
    bool VulkanRHI::ReadLumenLighting(LumenLightingHandle,std::vector<LumenLightingTexel>&) { return false; }
    bool VulkanRHI::DestroyLumenLighting(LumenLightingHandle) { return false; }
    bool VulkanRHI::TraceLumenRadiance(LumenLightingHandle,const LumenRadianceTraceUpload&,LumenRadianceTraceResult&) { return false; }
    bool VulkanRHI::ReadLumenRadiance(LumenRadianceHandle,std::vector<LumenRadianceSample>&) { return false; }
    bool VulkanRHI::AssembleLumenRadiance(const LumenRadianceAssembly&,LumenRadianceTraceResult&) { return false; }
    bool VulkanRHI::ReadLumenProbeOrigins(LumenRadianceHandle,std::vector<LumenProbeOrigin>&) { return false; }
    bool VulkanRHI::DestroyLumenRadiance(LumenRadianceHandle) { return false; }
    bool VulkanRHI::IntegrateLumenProbes(LumenRadianceHandle,const LumenProbeIntegrationUpload&,LumenProbeIntegrationResult&) { return false; }
    bool VulkanRHI::ResolveLumenProbes(LumenProbeHandle,const LumenProbeGatherSettings&) { return false; }
    bool VulkanRHI::ReadLumenProbes(LumenProbeHandle,std::vector<LumenProbeIrradiance>&) { return false; }
    bool VulkanRHI::DestroyLumenProbes(LumenProbeHandle) { return false; }
    bool VulkanRHI::LookupLumenSurface(LumenSurfaceHandle,const LumenSurfaceLookupUpload&,LumenSurfaceLookupResult&) { return false; }
    bool VulkanRHI::CaptureLumenSurface(const LumenSurfaceUpload&,LumenSurfaceResult&) { return false; }
    bool VulkanRHI::BeginLumenCaptureUpdate(const LumenSurfaceUpload&,LumenCaptureTask&) { return false; }
    bool VulkanRHI::AdvanceLumenCaptureUpdate(LumenCaptureTask,unsigned,bool& ready,LumenSurfaceResult&,LumenCaptureStatistics&) { ready=false; return false; }
    bool VulkanRHI::CancelLumenCaptureUpdate(LumenCaptureTask) { return false; }
    bool VulkanRHI::ReadLumenSurface(LumenSurfaceHandle,std::vector<LumenSurfaceTexel>&) { return false; }
    bool VulkanRHI::DestroyLumenSurface(LumenSurfaceHandle) { return false; }
    bool VulkanRHI::DestroyTSRHistory(TSRHistoryHandle) { return false; }
    bool VulkanRHI::RenderTSRDisplay(TSRHistoryHandle,const Techniques::PostProcessSettings&,FrameCapture&) { return false; }
    bool VulkanRHI::SnapshotTSROpaque() { return false; }
    bool VulkanRHI::CreateCachedVirtualGeometry(VirtualGeometryCacheHandle,const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool VulkanRHI::CreateCachedNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
bool VulkanRHI::SubmitNaniteGeometry(NaniteCacheHandle,const NaniteGpuUpload&,NaniteGeometryTask&) { return false; }
bool VulkanRHI::PollNaniteGeometry(NaniteGeometryTask,bool&,NaniteGeometryBuffers&) { return false; }
bool VulkanRHI::CancelNaniteGeometry(NaniteGeometryTask) { return false; }
    bool VulkanRHI::CreateVirtualGeometry(const VirtualGeometryGpuUpload&,VirtualGeometryBuffers&) { return false; }
bool VulkanRHI::PrepareNaniteDraws(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool VulkanRHI::PrepareNaniteDrawBins(NaniteVisibilityHandle,const std::vector<NaniteClusterDraw>&,NaniteDrawHandle&) { return false; }
bool VulkanRHI::DrawNaniteBins(NaniteDrawHandle) { return false; }
bool VulkanRHI::DrawNaniteCluster(NaniteDrawHandle,unsigned) { return false; }
bool VulkanRHI::CaptureNaniteDepth(NaniteDepthHandle&) { return false; }
bool VulkanRHI::BeginNaniteVisibility(NaniteDepthHandle,const std::vector<NaniteVisibilityQuery>&,NaniteVisibilityHandle&) { return false; }
bool VulkanRHI::RetestNaniteVisibility(NaniteVisibilityHandle,NaniteDepthHandle,NaniteVisibilityHandle&) { return false; }
bool VulkanRHI::ReadNaniteDepth(NaniteDepthHandle,std::vector<float>&) { return false; }
bool VulkanRHI::ReadNaniteVisibility(NaniteVisibilityHandle,std::vector<NaniteVisibilityState>&) { return false; }
bool VulkanRHI::DestroyNaniteDepth(NaniteDepthHandle) { return false; }
bool VulkanRHI::DestroyNaniteVisibility(NaniteVisibilityHandle) { return false; }
bool VulkanRHI::CreateNaniteGeometry(const NaniteGpuUpload&,NaniteGeometryBuffers&) { return false; }
    bool VulkanRHI::CreateTessellatedGeometry(const PNTessellationGeometry&,const Techniques::TessellationSettings&,TessellatedGeometryBuffers&) { return false; }
    bool VulkanRHI::CreateDisplacedVertexBuffer(const DisplacementGeometry&,const Techniques::DisplacementMappingSettings&,DisplacedVertexBuffer&) { return false; }
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
bool VulkanRHI::PrepareGpuDrivenBatch(const GpuDrivenBatchDesc&, GpuDrivenBatchHandle&) { return false; }
bool VulkanRHI::DrawGpuDrivenBatch(GpuDrivenBatchHandle) { return false; }
bool VulkanRHI::RenderVisibilityBuffer(const VisibilityBufferDesc&) { return false; }
bool VulkanRHI::DrawIndexed(std::uint32_t, std::uint32_t, std::int32_t) { return false; }
bool VulkanRHI::DrawIndexedInstanced(std::uint32_t, std::uint32_t, std::int32_t, const std::vector<DrawInstance>&) { return false; }
bool VulkanRHI::EndPass() { return false; }
bool VulkanRHI::PrepareLightListFrame(const LightListFrameDesc&) { return false; }
bool VulkanRHI::GenerateLightLists() { return false; }
bool VulkanRHI::GenerateContactShadows(const ContactShadowDesc&) { return false; }
bool VulkanRHI::DrawDeferredLighting(const DeferredLightingDesc&) { return false; }
    bool VulkanRHI::ApplySubsurfaceScattering(const SubsurfaceScatteringDesc&) { return false; }
    bool VulkanRHI::DrawHairFur(const HairFurDesc&) { return false; }
    bool VulkanRHI::DrawHairFurInPass(const HairFurDesc&) { return false; }
    bool VulkanRHI::DrawHairFurVelocity(const HairFurVelocityDesc&) { return false; }
bool VulkanRHI::ApplyAmbientOcclusion(const AmbientOcclusionDesc&) { return false; }
bool VulkanRHI::CompleteEnvironmentCapture() { return false; }
bool VulkanRHI::CompletePlanarCapture() { return false; }
bool VulkanRHI::ApplyPathTracing(const PathTracingDesc&) { return false; }
    bool VulkanRHI::ApplyGlobalIllumination(const GlobalIlluminationDesc&) { return false; }
bool VulkanRHI::ApplyReflections(const ReflectionDesc&) { return false; }
bool VulkanRHI::ApplyDepthOfField(const DepthOfFieldDesc&) { return false; }
bool VulkanRHI::ApplyLightShafts(const LightShaftDesc&) { return false; }
bool VulkanRHI::ApplyMotionBlur(const MotionBlurDesc&) { return false; }
bool VulkanRHI::ApplyTemporalAA(const TemporalAADesc&) { return false; }
    bool VulkanRHI::ApplyTemporalEffects(const TemporalEffectsDesc&) { return false; }
bool VulkanRHI::ApplyVolumetricFog(const VolumetricFogDesc&) { return false; }
bool VulkanRHI::ProcessMultisampling(Techniques::MultisamplePassKind) { return false; }
bool VulkanRHI::ProcessWeightedTransparency(Techniques::WeightedTransparencyPassKind) { return false; }
bool VulkanRHI::ProcessDepthPeeling(DepthPeelingPass) { return false; }
bool VulkanRHI::ProcessRefraction(const RefractionDesc&) { return false; }
bool VulkanRHI::Present() { return false; }
FrameCapture VulkanRHI::GetLastFrameCapture() const { return {}; }
FrameCapture VulkanRHI::TakeLastFrameCapture() { return {}; }
std::string VulkanRHI::GetLastError() const { return m_Impl->LastError; }
#endif
} // namespace Evans::Renderer
