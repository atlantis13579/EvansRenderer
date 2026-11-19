struct VirtualShadowRequestWork
{
    Impl& Owner; std::array<RawBuffer,3> Buffers{}; VkDescriptorPool Pool=0;
    std::shared_ptr<void> NativeDepthOwner;
    explicit VirtualShadowRequestWork(Impl& owner):Owner(owner) {}
    ~VirtualShadowRequestWork() { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); for (auto& buffer:Buffers) Owner.DestroyRawBuffer(buffer); }
};
std::shared_ptr<VirtualShadowRequestWork> VirtualShadowRequestPending;
VkDescriptorSetLayout VirtualShadowRequestSet=0; VkPipelineLayout VirtualShadowRequestLayout=0;
std::array<VkPipeline,2> VirtualShadowRequestPipelines{};
PFN_vkCreateComputePipelines VirtualShadowCreatePipelines=nullptr; PFN_vkCmdDispatch VirtualShadowDispatch=nullptr;
void DestroyVirtualShadowRequests()
{
    VirtualShadowRequestPending.reset();
    for (auto& pipeline:VirtualShadowRequestPipelines) { if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    if (VirtualShadowRequestLayout) DestroyPipelineLayoutFunction(Device,VirtualShadowRequestLayout,nullptr); VirtualShadowRequestLayout=0;
    if (VirtualShadowRequestSet) DestroyDescriptorSetLayoutFunction(Device,VirtualShadowRequestSet,nullptr); VirtualShadowRequestSet=0;
}
bool BuildVirtualShadowRequests(const VirtualShadowRequestUpload& upload,VirtualShadowRequestResult& output,
    const RawBuffer* nativeDepth=nullptr,std::shared_ptr<void> depthOwner={})
{
    if (!nativeDepth && !ValidateVirtualShadowRequestUpload(upload,LastError)) return false;
    VirtualShadowRequestConstants constants; if (!BuildVirtualShadowRequestConstants(upload.Frame,constants,LastError)) return false;
    const VkDeviceSize depthBytes=VkDeviceSize(upload.Frame.Width)*upload.Frame.Height*sizeof(float),requestBytes=VkDeviceSize(constants.Limits.Y)*sizeof(std::uint32_t);
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorStorageBuffers<2 || DeviceLimits.maxStorageBufferRange<depthBytes ||
        DeviceLimits.maxStorageBufferRange<requestBytes || DeviceLimits.maxUniformBufferRange<sizeof(constants) ||
        DeviceLimits.maxComputeWorkGroupInvocations<VirtualShadowGroupWidth || DeviceLimits.maxComputeWorkGroupSize[0]<VirtualShadowGroupWidth)
        return Fail("Vulkan virtual shadow requests exceed compute, storage or uniform limits");
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Recover virtual shadow request fence")) return false;
    VirtualShadowRequestPending.reset();
    if (!LoadDevice(VirtualShadowCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(VirtualShadowDispatch,"vkCmdDispatch")) return false;
    if (!VirtualShadowRequestSet)
    {
        const VkDescriptorSetLayoutBinding bindings[]{{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=3; desc.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&VirtualShadowRequestSet),"Create virtual shadow request layout")) return false;
    }
    if (!VirtualShadowRequestLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&VirtualShadowRequestSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&VirtualShadowRequestLayout),"Create virtual shadow request pipeline layout")) return false;
    }
    for (unsigned stage=0;stage<VirtualShadowRequestPipelines.size();++stage) if (!VirtualShadowRequestPipelines[stage])
    {
        VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanVirtualShadowRequestShaders[stage],module,LastError)) return false;
        VkComputePipelineCreateInfo desc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; desc.layout=VirtualShadowRequestLayout; desc.basePipelineIndex=-1;
        desc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(VirtualShadowCreatePipelines,Device,0,1,&desc,nullptr,&VirtualShadowRequestPipelines[stage]); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create virtual shadow request pipeline")) return false;
    }
    auto work=std::make_shared<VirtualShadowRequestWork>(*this);
    work->NativeDepthOwner=std::move(depthOwner);
    constexpr auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkDeviceSize sizes[]{sizeof(constants),depthBytes,requestBytes};
    for (unsigned slot=0;slot<work->Buffers.size();++slot)
    {
        if (slot==1 && nativeDepth) continue;
        if (!CreateRawBuffer(sizes[slot],slot?VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->Buffers[slot])) return false;
    }
    std::memcpy(work->Buffers[0].Mapped,&constants,sizeof(constants));
    if (!nativeDepth) std::memcpy(work->Buffers[1].Mapped,upload.Depth.data(),static_cast<std::size_t>(depthBytes));
    const VkDescriptorPoolSize poolSizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=poolSizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create virtual shadow request descriptors")) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&VirtualShadowRequestSet;
    VkDescriptorSet set=0; if (!Check(AllocateDescriptorSets(Device,&allocate,&set),"Allocate virtual shadow request descriptors")) return false;
    std::array<VkDescriptorBufferInfo,3> infos{}; std::array<VkWriteDescriptorSet,3> writes{};
    for (unsigned slot=0;slot<writes.size();++slot)
    {
        infos[slot]={slot==1 && nativeDepth?nativeDepth->Buffer:work->Buffers[slot].Buffer,0,sizes[slot]}; writes[slot]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; writes[slot].dstSet=set; writes[slot].dstBinding=slot;
        writes[slot].descriptorCount=1; writes[slot].descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[slot].pBufferInfo=&infos[slot];
    }
    UpdateDescriptorSets(Device,static_cast<unsigned>(writes.size()),writes.data(),0,nullptr);
    const unsigned counts[]{constants.Limits.Y,constants.Extent.X*constants.Extent.Y};
    for (auto count:counts)
    {
        const auto groups=(count+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
        if (std::min(groups,VirtualShadowGroupColumns)>DeviceLimits.maxComputeWorkGroupCount[0] ||
            (groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Virtual shadow demand exceeds Vulkan group dimensions");
    }
    if (!BeginImmediate()) return false; VirtualShadowRequestPending=work;
    std::array<VkBufferMemoryBarrier,2> visible{};
    for (unsigned slot=0;slot<visible.size();++slot)
    {
        auto& barrier=visible[slot]; barrier={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask=slot?VK_ACCESS_SHADER_READ_BIT:VK_ACCESS_UNIFORM_READ_BIT; barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer=slot && nativeDepth?nativeDepth->Buffer:work->Buffers[slot].Buffer; barrier.size=sizes[slot];
        if (slot && nativeDepth) barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    }
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,static_cast<unsigned>(visible.size()),visible.data(),0,nullptr);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,VirtualShadowRequestLayout,0,1,&set,0,nullptr);
    for (unsigned stage=0;stage<VirtualShadowRequestPipelines.size();++stage)
    {
        const auto groups=(counts[stage]+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
        const auto x=std::min(groups,VirtualShadowGroupColumns),y=(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns;
        if (stage)
        {
            VkBufferMemoryBarrier ordered{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; ordered.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; ordered.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
            ordered.srcQueueFamilyIndex=ordered.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; ordered.buffer=work->Buffers[2].Buffer; ordered.size=requestBytes;
            CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&ordered,0,nullptr);
        }
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,VirtualShadowRequestPipelines[stage]); VirtualShadowDispatch(CommandBuffer,x,y,1);
    }
    VkBufferMemoryBarrier completed{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; completed.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; completed.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    completed.srcQueueFamilyIndex=completed.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; completed.buffer=work->Buffers[2].Buffer; completed.size=requestBytes;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&completed,0,nullptr);
    if (!EndImmediate()) return false; VirtualShadowRequestPending.reset();
    VirtualShadowRequestResult result; result.Pages.resize(constants.Limits.Y); std::memcpy(result.Pages.data(),work->Buffers[2].Mapped,static_cast<std::size_t>(requestBytes));
    VirtualShadowLayout layout; if (!BuildVirtualShadowLayout(upload.Frame.Settings,layout,LastError) || !ValidateVirtualShadowRequests(layout,result.Pages,LastError)) return false;
    output=std::move(result); return true;
}
