// Per-frame storage is replaced only after BeginFrame has recovered its fence.
// One immutable descriptor set serves all four native GLSL compute pipelines.
NaniteMaterialWork MaterialPlan;
std::shared_ptr<NaniteFrameBuffer> MaterialMap,MaterialWork;
RawBuffer MaterialUniform;
VkDescriptorSetLayout MaterialSetLayout=0;
VkPipelineLayout MaterialLayout=0;
VkDescriptorPool MaterialPool=0;
VkDescriptorSet MaterialSet=0;
std::array<VkPipeline,4> MaterialKernels{};
void(EVANS_VK_CALL* MaterialDrawIndirect)(VkCommandBuffer,VkBuffer,VkDeviceSize,std::uint32_t,std::uint32_t)=nullptr;
void DestroyNaniteMaterials()
{
    MaterialPlan={}; MaterialMap.reset(); MaterialWork.reset(); DestroyRawBuffer(MaterialUniform);
    for (auto kernel:MaterialKernels) if (kernel) DestroyPipelineFunction(Device,kernel,nullptr); MaterialKernels={};
    if (MaterialPool) DestroyDescriptorPoolFunction(Device,MaterialPool,nullptr);
    if (MaterialLayout) DestroyPipelineLayoutFunction(Device,MaterialLayout,nullptr);
    if (MaterialSetLayout) DestroyDescriptorSetLayoutFunction(Device,MaterialSetLayout,nullptr);
    MaterialPool=MaterialSet=MaterialLayout=MaterialSetLayout=0;
}
bool PrepareNaniteMaterials(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteMaterialWork(frame,CreateInfo.Width,CreateInfo.Height,MaterialPlan,LastError)) return false;
    if (!MaterialPlan.Active()) return true;
    const auto& c=MaterialPlan.Constants;
    if (!GraphicsQueueHasCompute || DeviceLimits.maxComputeWorkGroupInvocations<NaniteMaterialThreads ||
        DeviceLimits.maxComputeWorkGroupSize[0]<NaniteMaterialThreads || DeviceLimits.maxPerStageDescriptorStorageBuffers<3 ||
        c.Layout.Y>DeviceLimits.maxComputeWorkGroupCount[0] || c.Layout.Z>DeviceLimits.maxComputeWorkGroupCount[1] ||
        (c.Layout.X+NaniteMaterialThreads-1)/NaniteMaterialThreads>DeviceLimits.maxComputeWorkGroupCount[0])
        return Fail("Vulkan material scheduling exceeds compute device limits");
    if (!LoadDevice(NaniteGeometryCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(NaniteGeometryDispatch,"vkCmdDispatch") ||
        !LoadDevice(MaterialDrawIndirect,"vkCmdDrawIndirect")) return false;
    if (!CreateNaniteFrameBuffer(unsigned(MaterialPlan.MaterialBins.size()),MaterialPlan.MaterialBins.data(),false,MaterialMap) ||
        !CreateNaniteFrameBuffer(MaterialPlan.OutputWords,nullptr,true,MaterialWork,true)) return false;
    if (!MaterialUniform.Buffer && !CreateRawBuffer(sizeof(NaniteMaterialConstants),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,MaterialUniform)) return false;
    std::memcpy(MaterialUniform.Mapped,&MaterialPlan.Constants,sizeof(NaniteMaterialConstants));
    if (!MaterialSetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[5]{};
        for (unsigned index=0;index<5;++index)
            bindings[index]={index,index==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:index==4?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; info.bindingCount=5; info.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&info,nullptr,&MaterialSetLayout),"Create material schedule descriptors")) return false;
    }
    if (!MaterialLayout)
    {
        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; info.setLayoutCount=1; info.pSetLayouts=&MaterialSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&info,nullptr,&MaterialLayout),"Create material schedule layout")) return false;
    }
    for (unsigned pass=0;pass<MaterialKernels.size();++pass) if (!MaterialKernels[pass])
    {
        VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanNaniteMaterialShaders[pass],module,LastError)) return false;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; info.layout=MaterialLayout; info.basePipelineIndex=-1;
        info.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(NaniteGeometryCreatePipelines,Device,0,1,&info,nullptr,&MaterialKernels[pass]); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create material schedule compute pipeline")) return false;
    }
    if (!MaterialPool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1}};
        VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; info.maxSets=1; info.poolSizeCount=3; info.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&info,nullptr,&MaterialPool),"Create material schedule pool")) return false;
    }
    if (!MaterialSet)
    {
        VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; info.descriptorPool=MaterialPool; info.descriptorSetCount=1; info.pSetLayouts=&MaterialSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&info,&MaterialSet),"Allocate material schedule set")) return false;
    }
    const VkDescriptorBufferInfo buffers[]{{MaterialUniform.Buffer,0,sizeof(NaniteMaterialConstants)},
        {MaterialMap->Allocation.Buffer,0,MaterialMap->Allocation.Size},{VisibilityGeometry.Buffer,0,VisibilityGeometry.Size},
        {MaterialWork->Allocation.Buffer,0,MaterialWork->Allocation.Size}};
    const VkDescriptorImageInfo image{DeferredSampler,VisibilityImages[0].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[5]{};
    for (unsigned index=0;index<5;++index)
    {
        auto& write=writes[index]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=MaterialSet; write.dstBinding=index; write.descriptorCount=1;
        write.descriptorType=index==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:index==4?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (index==4) write.pImageInfo=&image; else write.pBufferInfo=&buffers[index];
    }
    UpdateDescriptorSets(Device,5,writes,0,nullptr); return true;
}
bool ClassifyNaniteMaterials()
{
    if (!MaterialPlan.Active()) return true;
    TransitionImage(CommandBuffer,VisibilityImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,MaterialLayout,0,1,&MaterialSet,0,nullptr);
    for (unsigned pass=0;pass<MaterialKernels.size();++pass)
    {
        const auto& c=MaterialPlan.Constants;
        const unsigned x=pass==0?(c.Layout.X+NaniteMaterialThreads-1)/NaniteMaterialThreads:pass==2?1:c.Layout.Y;
        const unsigned y=pass==0 || pass==2?1:c.Layout.Z;
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,MaterialKernels[pass]); NaniteGeometryDispatch(CommandBuffer,x,y,1);
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer=MaterialWork->Allocation.Buffer; barrier.size=MaterialWork->Allocation.Size;
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    }
    auto& stats=PendingFrame.Statistics; stats.NaniteMaterialDispatchCount+=unsigned(MaterialKernels.size());
    stats.NaniteMaterialBinCount+=unsigned(MaterialPlan.Representatives.size()); stats.VisibilityMaterialDrawCount+=unsigned(MaterialPlan.Representatives.size());
    stats.IndirectDrawCallCount+=unsigned(MaterialPlan.Representatives.size()); return true;
}
