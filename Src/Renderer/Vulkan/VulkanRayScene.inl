// Four immutable storage buffers and four per-pass descriptor sets. A descriptor
// is never rewritten while an earlier AO draw in this frame references it.
static constexpr std::uint32_t RayAOTextureCount=4, RayAOPassCount=4;
static constexpr std::uint32_t RayAOSamplerBinding=RayAOTextureCount+1;
static constexpr std::uint32_t RayAOStorageBinding=RayAOSamplerBinding+1;
std::shared_ptr<const RaySceneUpload> FrameRayScene, UploadedRayScene;
std::array<RawBuffer,RaySceneBufferCount> RaySceneBuffers{};
VkDescriptorSetLayout RayAODescriptorLayout=0;
VkPipelineLayout RayAOPipelineLayout=0;
VkDescriptorPool RayAODescriptorPool=0;
std::array<VkDescriptorSet,RayAOPassCount> RayAODescriptors{};
std::array<VkPipeline,2> RayAOPipelines{};
std::array<VkShaderModule,2> RayAOModules{};

void DestroyRayScene()
{
    DestroyRayRadiance();
    for (auto& buffer:RaySceneBuffers) DestroyRawBuffer(buffer);
    for (auto pipeline:RayAOPipelines) if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr);
    for (auto module:RayAOModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    if (RayAODescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,RayAODescriptorPool,nullptr);
    if (RayAOPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,RayAOPipelineLayout,nullptr);
    if (RayAODescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,RayAODescriptorLayout,nullptr);
    RayAOPipelines={}; RayAOModules={}; RayAODescriptors={};
    RayAODescriptorPool=RayAOPipelineLayout=RayAODescriptorLayout=0;
    FrameRayScene.reset(); UploadedRayScene.reset();
}

bool EnsureRayScene()
{
    if (!FrameRayScene) return Fail("Vulkan ray AO requires a scene snapshot");
    if (FrameRayScene==UploadedRayScene) return true;
    if (!ValidateRaySceneUpload(*FrameRayScene,LastError)) return false;
    if (DeviceLimits.maxPerStageDescriptorStorageBuffers<RaySceneBufferCount ||
        DeviceLimits.maxDescriptorSetStorageBuffers<RaySceneBufferCount ||
        DeviceLimits.maxFragmentCombinedOutputResources<RaySceneBufferCount+1)
        return Fail("Vulkan ray AO requires four fragment storage buffers and one color output");
    const auto data=GetRaySceneBufferData(*FrameRayScene);
    for (const auto& buffer:data) if (buffer.Bytes>DeviceLimits.maxStorageBufferRange)
        return Fail("Vulkan ray scene exceeds maxStorageBufferRange");
    std::array<RawBuffer,RaySceneBufferCount> buffers{};
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        // Coherent immutable storage needs no per-frame transfer or rollback.
        if (!CreateRawBuffer(data[slot].Bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,buffers[slot]))
        {
            for (auto& buffer:buffers) DestroyRawBuffer(buffer); return false;
        }
        std::memcpy(buffers[slot].Mapped,data[slot].Data,data[slot].Bytes);
    }
    for (auto& buffer:RaySceneBuffers) DestroyRawBuffer(buffer);
    RaySceneBuffers=buffers; UploadedRayScene=FrameRayScene; return true;
}

bool EnsureRayAOState()
{
    if (!EnsureRayScene()) return false;
    if (!RayAODescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,RayAOStorageBinding+RaySceneBufferCount> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=1;slot<=RayAOTextureCount;++slot)
            bindings[slot]={slot,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        bindings[RayAOSamplerBinding]={RayAOSamplerBinding,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=RayAOStorageBinding;slot<bindings.size();++slot)
            bindings[slot]={slot,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&RayAODescriptorLayout),"vkCreateDescriptorSetLayout(ray AO)")) return false;
    }
    if (!RayAOPipelineLayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create.setLayoutCount=1; create.pSetLayouts=&RayAODescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&RayAOPipelineLayout),"vkCreatePipelineLayout(ray AO)")) return false;
    }
    if (!RayAODescriptorPool)
    {
        const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,RayAOPassCount},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,RayAOTextureCount*RayAOPassCount},
            {VK_DESCRIPTOR_TYPE_SAMPLER,RayAOPassCount},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,static_cast<std::uint32_t>(RaySceneBufferCount)*RayAOPassCount}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=RayAOPassCount; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&RayAODescriptorPool),"vkCreateDescriptorPool(ray AO)")) return false;
    }
    for (auto& descriptor:RayAODescriptors)
    {
        if (descriptor) continue;
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=RayAODescriptorPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&RayAODescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&descriptor),"vkAllocateDescriptorSets(ray AO)")) return false;
    }
    return CreateFullscreenPipeline(VulkanRayTracedAOShaders,RayAOModules,HdrRenderPass,RayAOPipelines[0],RayAOPipelineLayout) &&
        CreateFullscreenPipeline(VulkanRayTracedAOShaders,RayAOModules,AmbientOcclusionCompositeRenderPass,
            RayAOPipelines[1],RayAOPipelineLayout,FullscreenBlend::SubtractAmbient);
}

void UpdateRayAODescriptor(VkDescriptorSet descriptor,const std::array<VkImageView,RayAOTextureCount>& views)
{
    std::array<VkDescriptorBufferInfo,RaySceneBufferCount+1> buffers{};
    buffers[0]={UniformBuffer.Buffer,0,sizeof(RayTracedAOConstants)};
    for (std::size_t slot=0;slot<RaySceneBufferCount;++slot)
        buffers[slot+1]={RaySceneBuffers[slot].Buffer,0,RaySceneBuffers[slot].Size};
    std::array<VkDescriptorImageInfo,RayAOTextureCount+1> images{};
    std::array<VkWriteDescriptorSet,RayAOStorageBinding+RaySceneBufferCount> writes{};
    for (std::uint32_t binding=0;binding<writes.size();++binding)
    {
        auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=descriptor; write.dstBinding=binding; write.descriptorCount=1;
        if (binding==0)
        { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&buffers[0]; }
        else if (binding<=RayAOTextureCount)
        {
            images[binding-1]={0,views[binding-1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; write.pImageInfo=&images[binding-1];
        }
        else if (binding==RayAOSamplerBinding)
        {
            images.back().sampler=DeferredSampler;
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER; write.pImageInfo=&images.back();
        }
        else
        { write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo=&buffers[binding-RayAOStorageBinding+1]; }
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
}
