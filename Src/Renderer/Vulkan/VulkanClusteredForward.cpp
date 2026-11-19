#include "Renderer/Vulkan/VulkanPipelineCache.h"
#include "Renderer/Vulkan/VulkanClusteredForward.h"
#include "Renderer/Vulkan/VulkanShaderPrograms.h"
#include <cstring>

namespace Evans::Renderer
{
    using namespace VulkanABI;
    struct VulkanClusteredForward::Impl
    {
        enum class Phase { Idle, Open, Recorded, Submitted };
        struct Storage { VkBuffer Buffer = 0; VkDeviceMemory Memory = 0; VkDeviceSize Bytes = 0; };
        static constexpr VkFlags AllCommands = 0x00010000;
        static constexpr std::uint32_t GroupLanes = 64, StorageBlocks = 3;
        // Membership words, lane/block scans and six plane/error pairs; allow padding.
        static constexpr std::uint32_t SharedBytes = 4096;
        VkDevice Device = nullptr;
        VkPhysicalDeviceMemoryProperties Memory {};
        PhysicalDeviceLimitsPrefix Limits {};
        Storage Constants, Bounds, Header, Index;
        VkDescriptorSetLayout SetLayout = 0;
        VkDescriptorPool Pool = 0;
        VkDescriptorSet Set = 0;
        VkPipelineLayout Layout = 0;
        VkFence Fence = 0;
        VkFence CompletionFence = 0;
        std::array<VkPipeline, 1> Pipelines {};
        Phase Frame = Phase::Idle;
        ClusteredForwardGrid CurrentGrid;
        bool Initialized = false, Ready = false;
        std::string Error;
#define FORWARD_FUNCTIONS(X) \
        X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) \
        X(BindBufferMemory) X(MapMemory) X(UnmapMemory) \
        X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreateDescriptorPool) X(DestroyDescriptorPool) \
        X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreatePipelineLayout) X(DestroyPipelineLayout) \
        X(CreateShaderModule) X(DestroyShaderModule) X(CreateComputePipelines) X(DestroyPipeline) \
        X(CmdPipelineBarrier) X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdDispatch) \
        X(CreateFence) X(DestroyFence) X(GetFenceStatus) X(WaitForFences) X(ResetFences)
#define DECLARE(name) PFN_vk##name name = nullptr;
        FORWARD_FUNCTIONS(DECLARE)
#undef DECLARE
        bool Fail(const std::string& message) { Error = message; return false; }
        bool Check(VkResult result, const char* action)
        { return result == VK_SUCCESS || Fail(std::string("Vulkan Clustered Forward ") + action + ": " + std::to_string(result)); }
        bool Completed()
        {
            if (Frame != Phase::Submitted) return true;
            const auto status = GetFenceStatus(Device, CompletionFence);
            if (status == VK_NOT_READY) return Fail("Vulkan Clustered Forward submission is still in flight");
            return Check(status, "completion fence");
        }
        void Release(Storage& storage)
        {
            if (storage.Buffer) DestroyBuffer(Device, storage.Buffer, nullptr);
            if (storage.Memory) FreeMemory(Device, storage.Memory, nullptr);
            storage = {};
        }
        void Release()
        {
            Release(Constants); Release(Bounds); Release(Header); Release(Index);
            for (auto pipeline : Pipelines) if (pipeline) DestroyPipeline(Device, pipeline, nullptr);
            if (Pool) DestroyDescriptorPool(Device, Pool, nullptr);
            if (Layout) DestroyPipelineLayout(Device, Layout, nullptr);
            if (SetLayout) DestroyDescriptorSetLayout(Device, SetLayout, nullptr);
            if (Fence) DestroyFence(Device, Fence, nullptr);
            Pipelines = {}; Pool = Layout = SetLayout = Set = Fence = 0;
            CompletionFence = 0;
            Device = nullptr; Frame = Phase::Idle; Initialized = Ready = false; CurrentGrid = {};
        }
        bool CreateStorage(VkDeviceSize bytes, bool upload, Storage& destination, bool uniform = false)
        {
            Storage candidate; candidate.Bytes = bytes;
            VkBufferCreateInfo info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; info.size = bytes;
            info.usage = (uniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) |
                (upload ? 0 : VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            if (!Check(CreateBuffer(Device, &info, nullptr, &candidate.Buffer), "create storage")) return false;
            VkMemoryRequirements requirements {}; GetBufferMemoryRequirements(Device, candidate.Buffer, &requirements);
            const VkFlags flags = upload ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            std::uint32_t type = 0;
            for (; type < Memory.memoryTypeCount; ++type)
                if ((requirements.memoryTypeBits & (1u << type)) && (Memory.memoryTypes[type].propertyFlags & flags) == flags) break;
            if (type == Memory.memoryTypeCount) { Release(candidate); return Fail("Vulkan Clustered Forward storage memory unavailable"); }
            VkMemoryAllocateInfo allocation {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size; allocation.memoryTypeIndex = type;
            if (!Check(AllocateMemory(Device, &allocation, nullptr, &candidate.Memory), "allocate storage") ||
                !Check(BindBufferMemory(Device, candidate.Buffer, candidate.Memory, 0), "bind storage"))
            { Release(candidate); return false; }
            destination = candidate; return true;
        }
        bool Initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
            const PhysicalDeviceLimitsPrefix& limits, PFN_vkGetDeviceProcAddr load)
        {
            if (!device || !load) return Fail("Vulkan Clustered Forward needs an owning device and loader");
            Device = device; Memory = memory; Limits = limits;
#define LOAD(name) name = reinterpret_cast<PFN_vk##name>(load(Device, "vk" #name)); if (!name) return Fail("Missing vk" #name);
            FORWARD_FUNCTIONS(LOAD)
#undef LOAD
#undef FORWARD_FUNCTIONS
            if (limits.maxComputeWorkGroupInvocations < GroupLanes || limits.maxComputeWorkGroupSize[0] < GroupLanes ||
                limits.maxComputeWorkGroupSize[1] < 1 || limits.maxComputeWorkGroupSize[2] < 1 ||
                limits.maxComputeSharedMemorySize < SharedBytes || limits.maxUniformBufferRange < sizeof(ClusteredForwardConstants) || limits.maxPerStageDescriptorUniformBuffers < 1 ||
                limits.maxPerStageDescriptorStorageBuffers < StorageBlocks || limits.maxDescriptorSetStorageBuffers < StorageBlocks ||
                limits.maxStorageBufferRange < MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds))
                return Fail("Vulkan Clustered Forward device limits are insufficient");
            const VkDescriptorSetLayoutBinding bindings[] {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
            VkDescriptorSetLayoutCreateInfo set {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; set.bindingCount = 4; set.pBindings = bindings;
            if (!Check(CreateDescriptorSetLayout(Device, &set, nullptr, &SetLayout), "descriptor layout")) return false;
            const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, StorageBlocks}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
            VkDescriptorPoolCreateInfo pool {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets = 1; pool.poolSizeCount = 2; pool.pPoolSizes = sizes;
            if (!Check(CreateDescriptorPool(Device, &pool, nullptr, &Pool), "descriptor pool")) return false;
            VkDescriptorSetAllocateInfo allocation {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocation.descriptorPool = Pool; allocation.descriptorSetCount = 1; allocation.pSetLayouts = &SetLayout;
            if (!Check(AllocateDescriptorSets(Device, &allocation, &Set), "descriptor allocation")) return false;
            VkPipelineLayoutCreateInfo layout {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; layout.setLayoutCount = 1; layout.pSetLayouts = &SetLayout;
            if (!Check(CreatePipelineLayout(Device, &layout, nullptr, &Layout), "pipeline layout")) return false;
            VkFenceCreateInfo fence {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (!Check(CreateFence(Device, &fence, nullptr, &Fence), "completion fence")) return false;
            for (unsigned variant = 0; variant < Pipelines.size(); ++variant)
            {
                VkShaderModule module = 0;
                if (!CreateVulkanShaderModule(Device, CreateShaderModule, VulkanClusteredForwardAssignment, module, Error)) return false;
                VkComputePipelineCreateInfo pipeline {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
                pipeline.layout = Layout; pipeline.basePipelineIndex = -1;
                const bool created = Check(Evans::Renderer::CreateCachedVulkanComputePipelines(CreateComputePipelines,Device,0, 1, &pipeline, nullptr, &Pipelines[variant]), "compute pipeline");
                DestroyShaderModule(Device, module, nullptr); if (!created) return false;
            }
            if (!CreateStorage(MaximumForwardPlusLights * sizeof(ForwardPlusGpuBounds), true, Bounds)) return false;
            if (!CreateStorage(sizeof(ClusteredForwardConstants), true, Constants, true)) return false;
            Initialized = true; return true;
        }
        bool Record(VkCommandBuffer commands, std::uint32_t width, std::uint32_t height,
            const Matrix4& camera, const std::vector<ForwardPlusLightBounds>& lights, const ClusteredForwardParameters& parameters)
        {
            Ready = false; Error.clear();
            if (!Initialized || Frame != Phase::Open || !commands)
                return Fail("Vulkan clustered assignment requires an unused open frame and command buffer");
            ClusteredForwardGrid grid; ClusteredForwardConstants constants; std::vector<ForwardPlusGpuBounds> bounds;
            if (!BuildClusteredForwardGrid(width, height, lights.size(), parameters, grid, Error) ||
                !BuildClusteredForwardConstants(camera, grid, constants, Error) || !BuildForwardPlusGpuBounds(lights, bounds, Error)) return false;
            const VkDeviceSize headerBytes = VkDeviceSize(grid.ClusterCount + ClusteredBoundarySlots) * sizeof(ClusteredForwardHeader), indexBytes = VkDeviceSize(grid.IndexCount) * sizeof(std::uint32_t);
            if (grid.TilesX > Limits.maxComputeWorkGroupCount[0] || grid.TilesY > Limits.maxComputeWorkGroupCount[1] ||
                grid.DepthSlices > Limits.maxComputeWorkGroupCount[2] || headerBytes > Limits.maxStorageBufferRange || indexBytes > Limits.maxStorageBufferRange)
                return Fail("Vulkan Clustered Forward image, dispatch or storage extent exceeds device limits");
            if (Header.Bytes != headerBytes || Index.Bytes != indexBytes)
            {
                Storage headers, indices;
                if (!CreateStorage(headerBytes, false, headers) || !CreateStorage(indexBytes, false, indices))
                { Release(headers); Release(indices); return false; }
                Release(Header); Release(Index); Header = headers; Index = indices;
            }
            if (!bounds.empty())
            {
                void* mapped = nullptr; const auto bytes = bounds.size() * sizeof(ForwardPlusGpuBounds);
                if (!Check(MapMemory(Device, Bounds.Memory, 0, bytes, 0, &mapped), "map bounds")) return false;
                std::memcpy(mapped, bounds.data(), bytes); UnmapMemory(Device, Bounds.Memory);
            }
            void* uniform = nullptr;
            if (!Check(MapMemory(Device, Constants.Memory, 0, sizeof(constants), 0, &uniform), "map constants")) return false;
            std::memcpy(uniform, &constants, sizeof(constants)); UnmapMemory(Device, Constants.Memory);
            const VkDescriptorBufferInfo buffers[] {{Constants.Buffer, 0, Constants.Bytes}, {Bounds.Buffer, 0, Bounds.Bytes},
                {Header.Buffer, 0, Header.Bytes}, {Index.Buffer, 0, Index.Bytes}};
            VkWriteDescriptorSet writes[4] {};
            for (unsigned binding = 0; binding < 4; ++binding)
            {
                auto& write = writes[binding]; write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet = Set;
                write.dstBinding = binding; write.descriptorCount = 1;
                write.descriptorType = binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &buffers[binding];
            }
            UpdateDescriptorSets(Device, 4, writes, 0, nullptr);
            // Coherent host writes happen before queue submission (which makes
            // them available). The fence prevents overwriting their previous use.
            // Order all earlier list readers/writers before assignment.
            // https://docs.vulkan.org/guide/latest/synchronization_examples.html
            VkMemoryBarrier reuse {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT};
            CmdPipelineBarrier(commands, AllCommands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &reuse, 0, nullptr, 0, nullptr);
            CmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, Pipelines[0]);
            CmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, Layout, 0, 1, &Set, 0, nullptr);
            CmdDispatch(commands, grid.TilesX, grid.TilesY, grid.DepthSlices);
            VkMemoryBarrier publish {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT};
            CmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &publish, 0, nullptr, 0, nullptr);
            CurrentGrid = grid; Frame = Phase::Recorded; Ready = true; return true;
        }
    };
    VulkanClusteredForward::VulkanClusteredForward() : m_Impl(std::make_unique<Impl>()) {}
    VulkanClusteredForward::~VulkanClusteredForward()
    {
        auto& state = *m_Impl;
        if (state.Frame == Impl::Phase::Submitted)
            state.WaitForFences(state.Device, 1, &state.CompletionFence, VK_TRUE_VALUE, std::numeric_limits<std::uint64_t>::max());
        state.Release();
    }
    bool VulkanClusteredForward::Initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
        const PhysicalDeviceLimitsPrefix& limits, PFN_vkGetDeviceProcAddr load, std::string& error)
    {
        auto& state = *m_Impl; error.clear();
        if (state.Initialized) { if (state.Device == device) return true; error = "Vulkan Clustered Forward already owns another device"; return false; }
        state.Error.clear(); const bool success = state.Initialize(device, memory, limits, load);
        error = state.Error; if (!success) state.Release(); return success;
    }
    bool VulkanClusteredForward::BeginFrame(std::string& error)
    {
        auto& state = *m_Impl; state.Ready = false; error.clear(); state.Error.clear();
        if (!state.Initialized || state.Frame == Impl::Phase::Recorded) { error = "Vulkan Clustered Forward has no device or unsubmitted recording"; return false; }
        if (!state.Completed()) { error = state.Error; return false; }
        if (state.Frame == Impl::Phase::Submitted && state.CompletionFence == state.Fence &&
            !state.Check(state.ResetFences(state.Device, 1, &state.Fence), "reset completed fence"))
        { error = state.Error; return false; }
        state.CompletionFence = 0;
        state.Frame = Impl::Phase::Open; return true;
    }
    bool VulkanClusteredForward::AbortFrame(std::string& error)
    {
        auto& state = *m_Impl; error.clear();
        if (state.Frame == Impl::Phase::Submitted) { error = "Cannot abort a submitted Vulkan Clustered Forward frame"; return false; }
        state.Frame = Impl::Phase::Idle; state.Ready = false; return true;
    }
    bool VulkanClusteredForward::MarkSubmitted(std::string& error)
    { return MarkSubmitted(m_Impl->Fence, error); }
    bool VulkanClusteredForward::MarkSubmitted(VkFence completionFence, std::string& error)
    {
        auto& state = *m_Impl; error.clear();
        if (state.Frame != Impl::Phase::Recorded || !completionFence)
        { error = "Vulkan Clustered Forward requires recorded work and its submission fence"; return false; }
        state.CompletionFence = completionFence;
        state.Frame = Impl::Phase::Submitted; return true;
    }
    bool VulkanClusteredForward::Shutdown(std::string& error)
    {
        auto& state = *m_Impl; error.clear(); state.Error.clear();
        if (state.Frame == Impl::Phase::Recorded) { error = "Discard or submit the Vulkan Clustered Forward recording before shutdown"; return false; }
        if (!state.Completed()) { error = state.Error; return false; }
        state.Release(); return true;
    }
    VkFence VulkanClusteredForward::SubmissionFence() const { return m_Impl->Frame == Impl::Phase::Recorded ? m_Impl->Fence : 0; }
    bool VulkanClusteredForward::Record(VkCommandBuffer commands, std::uint32_t width, std::uint32_t height,
        const Matrix4& camera, const std::vector<ForwardPlusLightBounds>& lights, const ClusteredForwardParameters& parameters, std::string& error)
    { const bool result = m_Impl->Record(commands, width, height, camera, lights, parameters); error = m_Impl->Error; return result; }
    VkDescriptorBufferInfo VulkanClusteredForward::Headers() const
    { return m_Impl->Ready ? VkDescriptorBufferInfo {m_Impl->Header.Buffer, 0, m_Impl->Header.Bytes} : VkDescriptorBufferInfo {}; }
    VkDescriptorBufferInfo VulkanClusteredForward::Indices() const
    { return m_Impl->Ready ? VkDescriptorBufferInfo {m_Impl->Index.Buffer, 0, m_Impl->Index.Bytes} : VkDescriptorBufferInfo {}; }
    ClusteredForwardGrid VulkanClusteredForward::Grid() const { return m_Impl->Ready ? m_Impl->CurrentGrid : ClusteredForwardGrid {}; }
}
