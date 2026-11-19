struct LumenLookupWork
{
    Impl& Owner; RawBuffer Uniform,Cards,Queries,Feedback; VkDescriptorPool Pool=0; VkDescriptorSet Set=0;
    std::shared_ptr<LumenSurfaceRecord> Surface;
    explicit LumenLookupWork(Impl& owner):Owner(owner) {}
    ~LumenLookupWork()
    {
        if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr);
        Owner.DestroyRawBuffer(Uniform); Owner.DestroyRawBuffer(Cards); Owner.DestroyRawBuffer(Queries); Owner.DestroyRawBuffer(Feedback);
    }
};
std::shared_ptr<LumenLookupWork> LumenLookupPending;
VkDescriptorSetLayout LumenLookupSet=0; VkPipelineLayout LumenLookupLayout=0; VkPipeline LumenSurfaceLookup=0; VkSampler LumenLookupSampler=0;
PFN_vkCreateComputePipelines LookupCreateCompute=nullptr; PFN_vkCmdDispatch LookupDispatch=nullptr;
bool RecoverLumenSurfaceLookup()
{
    if (!LumenLookupPending) return true;
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for retained Lumen surface lookup")) return false;
    LumenLookupPending.reset(); return true;
}
void DestroyLumenSurfaceLookup()
{
    LumenLookupPending.reset();
    if (LumenSurfaceLookup) DestroyPipelineFunction(Device,LumenSurfaceLookup,nullptr); LumenSurfaceLookup=0;
    if (LumenLookupLayout) DestroyPipelineLayoutFunction(Device,LumenLookupLayout,nullptr); LumenLookupLayout=0;
    if (LumenLookupSet) DestroyDescriptorSetLayoutFunction(Device,LumenLookupSet,nullptr); LumenLookupSet=0;
    if (LumenLookupSampler) DestroySamplerFunction(Device,LumenLookupSampler,nullptr); LumenLookupSampler=0;
}
bool EnsureLumenSurfaceLookup()
{
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorStorageBuffers<3 || DeviceLimits.maxDescriptorSetStorageBuffers<3 ||
        DeviceLimits.maxPerStageDescriptorSampledImages<LumenSurfacePlanes || DeviceLimits.maxPerStageDescriptorSamplers<LumenSurfacePlanes ||
        DeviceLimits.maxUniformBufferRange<sizeof(LumenSurfaceLookupFrame) || DeviceLimits.maxComputeWorkGroupInvocations<LumenLookupThreads || DeviceLimits.maxComputeWorkGroupSize[0]<LumenLookupThreads)
        return Fail("Vulkan surface lookup exceeds native descriptor/compute limits");
    if (!LoadDevice(LookupCreateCompute,"vkCreateComputePipelines") || !LoadDevice(LookupDispatch,"vkCmdDispatch")) return false;
    if (!LumenLookupSet)
    {
        VkDescriptorSetLayoutBinding bindings[7]{};
        for (unsigned slot=0;slot<7;++slot)
            bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot<=3?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=7; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenLookupSet),"Create Lumen lookup descriptor layout")) return false;
    }
    if (!LumenLookupLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenLookupSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenLookupLayout),"Create Lumen lookup pipeline layout")) return false;
    }
    if (!LumenLookupSampler)
    {
        VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; create.magFilter=create.minFilter=VK_FILTER_NEAREST;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; create.maxAnisotropy=1;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&LumenLookupSampler),"Create Lumen lookup sampler")) return false;
    }
    if (LumenSurfaceLookup) return true;
    VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenSurfaceLookupShader,module,LastError)) return false;
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenLookupLayout; create.basePipelineIndex=-1;
    create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(LookupCreateCompute,Device,0,1,&create,nullptr,&LumenSurfaceLookup); DestroyShaderModuleFunction(Device,module,nullptr);
    return Check(status,"Create Lumen surface lookup pipeline");
}
void LumenLookupBarrier(const RawBuffer& buffer,VkFlags source,VkFlags destination,VkFlags sourceStage,VkFlags destinationStage)
{
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=source; barrier.dstAccessMask=destination;
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=buffer.Buffer; barrier.size=buffer.Size;
    CmdPipelineBarrier(CommandBuffer,sourceStage,destinationStage,0,0,nullptr,1,&barrier,0,nullptr);
}
bool LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Unknown Lumen surface lookup handle");
    LumenSurfaceLookupFrame frame; std::vector<LumenLookupCard> cards;
    if (!BuildLumenSurfaceLookupFrame(found->second->Layout,upload,frame,LastError) || !BuildLumenLookupCards(found->second->Layout,found->second->Captures,cards,LastError) || !EnsureLumenSurfaceLookup()) return false;
    const auto bytes=upload.Queries.size()*sizeof(LumenSurfaceSample),cardBytes=cards.size()*sizeof(LumenLookupCard),queryBytes=upload.Queries.size()*sizeof(LumenSurfaceQuery);
    if (std::max(bytes,cardBytes)>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan Lumen lookup exceeds storage limits");
    const auto groups=unsigned((upload.Queries.size()+LumenLookupThreads-1)/LumenLookupThreads);
    const unsigned x=std::min(groups,LumenLookupColumns),y=(groups+LumenLookupColumns-1)/LumenLookupColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan Lumen lookup exceeds dispatch limits");
    auto work=std::make_shared<LumenLookupWork>(*this); work->Surface=found->second;
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(frame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->Uniform) || !CreateRawBuffer(cardBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Cards) ||
        !CreateRawBuffer(queryBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Queries) || !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Feedback)) return false;
    std::memcpy(work->Uniform.Mapped,&frame,sizeof(frame)); std::memcpy(work->Cards.Mapped,cards.data(),cardBytes); std::memcpy(work->Queries.Mapped,upload.Queries.data(),queryBytes);
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,LumenSurfacePlanes},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Lumen lookup descriptor pool")) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenLookupSet;
    if (!Check(AllocateDescriptorSets(Device,&allocate,&work->Set),"Allocate Lumen lookup descriptor set")) return false;
    const RawBuffer* buffers[]{&work->Uniform,&work->Cards,&work->Queries,&work->Feedback}; VkDescriptorBufferInfo bufferInfos[4]{}; VkDescriptorImageInfo imageInfos[LumenSurfacePlanes]{};
    for (unsigned index=0;index<4;++index) bufferInfos[index]={buffers[index]->Buffer,0,buffers[index]->Size};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) imageInfos[plane]={LumenLookupSampler,work->Surface->Images[plane].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[7]{};
    for (unsigned slot=0;slot<7;++slot)
    {
        auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->Set; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot<=3?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (slot>=1 && slot<=3) write.pImageInfo=&imageInfos[slot-1]; else write.pBufferInfo=&bufferInfos[slot?slot-3:0];
    }
    UpdateDescriptorSets(Device,7,writes,0,nullptr);
    LumenSurfaceLookupResult result; result.Identity=work->Surface->Identity; result.Samples.resize(upload.Queries.size());
    if (!BeginImmediate()) return false;
    LumenLookupBarrier(work->Uniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    for (const auto* buffer:{&work->Cards,&work->Queries}) LumenLookupBarrier(*buffer,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenSurfaceLookup); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenLookupLayout,0,1,&work->Set,0,nullptr);
    LookupDispatch(CommandBuffer,x,y,1); LumenLookupBarrier(work->Feedback,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
    if (!Check(EndCommandBuffer(CommandBuffer),"End Lumen lookup commands") || !Check(ResetFences(Device,1,&FrameFence),"Reset Lumen lookup fence")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&CommandBuffer;
    const auto status=QueueSubmit(GraphicsQueue,1,&submit,FrameFence);
    if (status!=VK_SUCCESS) { (void)RecreateSignaledFrameFence(); return Check(status,"Submit Lumen lookup commands"); }
    LumenLookupPending=work; if (!RecoverLumenSurfaceLookup()) return false;
    std::memcpy(result.Samples.data(),work->Feedback.Mapped,bytes);
    if (!ValidateLumenSurfaceLookupResult(work->Surface->Identity,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
