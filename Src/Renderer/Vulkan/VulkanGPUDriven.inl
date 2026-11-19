// Vulkan 1.0 compute and one indirect indexed draw per material batch. This
// requires neither multiDrawIndirect nor drawIndirectFirstInstance features.
// https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndexedIndirect.html
struct GpuBatch
{
    GpuDrivenBatchDesc Description;
    RawBuffer Constants, Input, Output, Arguments;
    VkDescriptorPool Pool = 0;
    VkDescriptorSet ComputeSet = 0, InstanceSet = 0;
};
std::vector<GpuBatch> GpuBatches;
GpuBatch* ActiveGpuBatch = nullptr;
std::size_t GpuCursor = 0;
std::uint64_t GpuEpoch = 0;
VkDescriptorSetLayout GpuCullSetLayout = 0;
VkPipelineLayout GpuCullLayout = 0;
VkPipeline GpuCullPipeline = 0;
PFN_vkCreateComputePipelines GpuCreateComputePipelines = nullptr;
PFN_vkCmdDispatch GpuDispatch = nullptr;
PFN_vkCmdDrawIndexedIndirect GpuDrawIndexedIndirect = nullptr;

void DestroyGpuBatch(GpuBatch& batch)
{
    if (batch.Pool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device, batch.Pool, nullptr);
    for (auto* buffer : {&batch.Constants, &batch.Input, &batch.Output, &batch.Arguments}) DestroyRawBuffer(*buffer);
    batch = {};
}
void DestroyGpuDriven()
{
    ActiveGpuBatch = nullptr; GpuCursor = 0;
    for (auto& batch : GpuBatches) DestroyGpuBatch(batch);
    GpuBatches.clear();
    if (GpuCullPipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device, GpuCullPipeline, nullptr);
    if (GpuCullLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device, GpuCullLayout, nullptr);
    if (GpuCullSetLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device, GpuCullSetLayout, nullptr);
    GpuCullPipeline = 0; GpuCullLayout = 0; GpuCullSetLayout = 0;
}
bool EnsureGpuCull()
{
    if (!GraphicsQueueHasCompute) return Fail("Vulkan GPU-driven rendering requires a graphics/compute queue");
    if (GpuCullPipeline) return true;
    if (!LoadDevice(GpuCreateComputePipelines, "vkCreateComputePipelines") || !LoadDevice(GpuDispatch, "vkCmdDispatch") ||
        !LoadDevice(GpuDrawIndexedIndirect, "vkCmdDrawIndexedIndirect")) return false;
    if (!GpuCullSetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[4] {};
        for (std::uint32_t slot = 0; slot < 4; ++slot)
            bindings[slot] = {slot, slot ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = 4; info.pBindings = bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device, &info, nullptr, &GpuCullSetLayout), "vkCreateDescriptorSetLayout(GPU cull)")) return false;
    }
    if (!GpuCullLayout)
    {
        VkPipelineLayoutCreateInfo info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = 1; info.pSetLayouts = &GpuCullSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device, &info, nullptr, &GpuCullLayout), "vkCreatePipelineLayout(GPU cull)")) return false;
    }
    VkShaderModule module = 0;
    if (!CreateVulkanShaderModule(Device, CreateShaderModuleFunction, VulkanGpuDrivenCull, module, LastError)) return false;
    VkComputePipelineCreateInfo info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
    info.layout = GpuCullLayout; info.basePipelineIndex = -1;
    const auto result = Evans::Renderer::CreateCachedVulkanComputePipelines(GpuCreateComputePipelines,Device,0, 1, &info, nullptr, &GpuCullPipeline);
    DestroyShaderModuleFunction(Device, module, nullptr);
    return Check(result, "vkCreateComputePipelines(GPU cull)");
}
bool CreateGpuBatch(GpuBatch& batch)
{
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(GpuDrivenCullConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, host, batch.Constants) ||
        !CreateRawBuffer(sizeof(InstanceConstants) * MaximumDrawInstances, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, batch.Input) ||
        !CreateRawBuffer(sizeof(InstanceConstantBlock), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, batch.Output) ||
        !CreateRawBuffer(sizeof(GpuDrivenIndexedArguments), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, batch.Arguments)) return false;
    const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1}};
    VkDescriptorPoolCreateInfo pool {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 2; pool.poolSizeCount = 3; pool.pPoolSizes = sizes;
    if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &batch.Pool), "vkCreateDescriptorPool(GPU batch)")) return false;
    const VkDescriptorSetLayout layouts[] {GpuCullSetLayout, InstanceDescriptorLayout};
    VkDescriptorSet sets[2] {};
    VkDescriptorSetAllocateInfo allocation {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = batch.Pool; allocation.descriptorSetCount = 2; allocation.pSetLayouts = layouts;
    if (!Check(AllocateDescriptorSets(Device, &allocation, sets), "vkAllocateDescriptorSets(GPU batch)")) return false;
    batch.ComputeSet = sets[0]; batch.InstanceSet = sets[1];
    const VkDescriptorBufferInfo buffers[] {{batch.Constants.Buffer, 0, batch.Constants.Size}, {batch.Input.Buffer, 0, batch.Input.Size},
        {batch.Output.Buffer, 0, batch.Output.Size}, {batch.Arguments.Buffer, 0, batch.Arguments.Size}, {batch.Output.Buffer, 0, batch.Output.Size}};
    VkWriteDescriptorSet writes[5] {};
    for (std::uint32_t slot = 0; slot < 5; ++slot)
    {
        auto& write = writes[slot]; write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = slot == 4 ? batch.InstanceSet : batch.ComputeSet; write.dstBinding = slot == 4 ? 0 : slot;
        write.descriptorCount = 1; write.pBufferInfo = &buffers[slot];
        write.descriptorType = slot == 4 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC :
            slot == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    UpdateDescriptorSets(Device, 5, writes, 0, nullptr);
    return true;
}
bool PrepareGpuBatch(const GpuDrivenBatchDesc& desc, std::uint32_t& index)
{
    GpuDrivenBatchUpload upload;
    if (!BuildGpuDrivenBatch(desc, upload)) return Fail("Vulkan GPU batch contains invalid bounds, transforms or command data");
    if (!EnsureGpuCull()) return false;
    if (GpuCursor == GpuBatches.size())
    {
        GpuBatch batch;
        if (!CreateGpuBatch(batch)) { DestroyGpuBatch(batch); return false; }
        try { GpuBatches.push_back(std::move(batch)); }
        catch (...) { DestroyGpuBatch(batch); throw; }
    }
    auto& batch = GpuBatches[GpuCursor]; batch.Description = desc;
    std::memcpy(batch.Constants.Mapped, &upload.Constants, sizeof(upload.Constants));
    std::memcpy(batch.Input.Mapped, upload.Candidates.data(), sizeof(upload.Candidates));
    CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, GpuCullPipeline);
    CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, GpuCullLayout, 0, 1, &batch.ComputeSet, 0, nullptr);
    GpuDispatch(CommandBuffer, 1, 1, 1);
    VkBufferMemoryBarrier barriers[2] {};
    for (auto& barrier : barriers)
    {
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }
    barriers[0].buffer = batch.Output.Buffer; barriers[0].size = batch.Output.Size; barriers[0].dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    barriers[1].buffer = batch.Arguments.Buffer; barriers[1].size = batch.Arguments.Size; barriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    CmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, 2, barriers, 0, nullptr);
    index = static_cast<std::uint32_t>(GpuCursor++); return true;
}
