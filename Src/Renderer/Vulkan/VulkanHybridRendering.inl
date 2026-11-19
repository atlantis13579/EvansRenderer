// The GLSL source is compiled at runtime. Each pass owns a descriptor set so
// later ping-pong bindings cannot change descriptors referenced by earlier work.
static constexpr std::uint32_t HybridRenderingTextureCount=7,HybridRenderingPassCount=4;
static constexpr std::uint32_t HybridRenderingSamplerBinding=HybridRenderingTextureCount+1;
static constexpr std::uint32_t HybridRenderingStorageBinding=HybridRenderingSamplerBinding+1;
static constexpr std::uint32_t HybridRenderingStorageSlots=RaySceneBufferCount+RayRadianceBufferCount;
static constexpr std::uint32_t HybridRenderingStorageCount=HybridRenderingStorageSlots-1;
static constexpr std::uint32_t HybridRenderingAtlasBinding=HybridRenderingStorageBinding+RaySceneBufferCount+RayRadianceAtlasSlot;
std::array<RawImage,2> HybridRenderingImages{};
std::array<VkFramebuffer,2> HybridRenderingFramebuffers{};
VkDescriptorSetLayout HybridRenderingDescriptorLayout=0;
VkPipelineLayout HybridRenderingPipelineLayout=0;
VkDescriptorPool HybridRenderingDescriptorPool=0;
std::array<VkDescriptorSet,HybridRenderingPassCount> HybridRenderingDescriptors{};
std::array<VkPipeline,2> HybridRenderingPipelines{};
std::array<VkShaderModule,2> HybridRenderingModules{};

void DestroyHybridRenderingTargets()
{
    for (auto& framebuffer:HybridRenderingFramebuffers)
    { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,framebuffer,nullptr); framebuffer=0; }
    for (auto& image:HybridRenderingImages) DestroyRawImage(image);
}
void DestroyHybridRendering()
{
    DestroyHybridRenderingTargets();
    for (auto pipeline:HybridRenderingPipelines) if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr);
    for (auto module:HybridRenderingModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    if (HybridRenderingDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,HybridRenderingDescriptorPool,nullptr);
    if (HybridRenderingPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,HybridRenderingPipelineLayout,nullptr);
    if (HybridRenderingDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,HybridRenderingDescriptorLayout,nullptr);
    HybridRenderingPipelines={}; HybridRenderingModules={}; HybridRenderingDescriptors={};
    HybridRenderingDescriptorPool=HybridRenderingPipelineLayout=HybridRenderingDescriptorLayout=0;
}

bool EnsureHybridRenderingResources()
{
    if (!EnsureRayRadiance()) return false;
    if (!CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT,ReflectionCompositeRenderPass,true)) return false;
    if (HybridRenderingImages[0].Width!=MainColor.Width || HybridRenderingImages[0].Height!=MainColor.Height)
        DestroyHybridRenderingTargets();
    for (std::size_t index=0;index<HybridRenderingImages.size();++index)
    {
        if (HybridRenderingFramebuffers[index]) continue;
        if (!CreateRawImage(MainColor.Width,MainColor.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,HybridRenderingImages[index]) ||
            !CreateFramebuffer(HdrRenderPass,&HybridRenderingImages[index].View,1,MainColor.Width,MainColor.Height,HybridRenderingFramebuffers[index]))
        { DestroyHybridRenderingTargets(); return false; }
    }
    if (!HybridRenderingDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,HybridRenderingStorageBinding+HybridRenderingStorageSlots> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=1;slot<=HybridRenderingTextureCount;++slot)
            bindings[slot]={slot,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        bindings[HybridRenderingSamplerBinding]={HybridRenderingSamplerBinding,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=HybridRenderingStorageBinding;slot<bindings.size();++slot)
            bindings[slot]={slot,slot==HybridRenderingAtlasBinding ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&HybridRenderingDescriptorLayout),"vkCreateDescriptorSetLayout(hybrid reflection)")) return false;
    }
    if (!HybridRenderingPipelineLayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create.setLayoutCount=1; create.pSetLayouts=&HybridRenderingDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&HybridRenderingPipelineLayout),"vkCreatePipelineLayout(hybrid reflection)")) return false;
    }
    if (!HybridRenderingDescriptorPool)
    {
        const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,HybridRenderingPassCount},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,(HybridRenderingTextureCount+1)*HybridRenderingPassCount},
            {VK_DESCRIPTOR_TYPE_SAMPLER,HybridRenderingPassCount},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,HybridRenderingStorageCount*HybridRenderingPassCount}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=HybridRenderingPassCount; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&HybridRenderingDescriptorPool),"vkCreateDescriptorPool(hybrid reflection)")) return false;
    }
    for (auto& descriptor:HybridRenderingDescriptors) if (!descriptor)
    {
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=HybridRenderingDescriptorPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&HybridRenderingDescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&descriptor),"vkAllocateDescriptorSets(hybrid reflection)")) return false;
    }
    return CreateFullscreenPipeline(VulkanHybridRenderingShaders,HybridRenderingModules,HdrRenderPass,HybridRenderingPipelines[0],HybridRenderingPipelineLayout) &&
        CreateFullscreenPipeline(VulkanHybridRenderingShaders,HybridRenderingModules,ReflectionCompositeRenderPass,
            HybridRenderingPipelines[1],HybridRenderingPipelineLayout,FullscreenBlend::AddRadiance,true);
}
void UpdateHybridRenderingDescriptor(VkDescriptorSet descriptor,const std::array<VkImageView,HybridRenderingTextureCount>& views)
{
    std::array<VkDescriptorBufferInfo,HybridRenderingStorageSlots+1> buffers{};
    buffers[0]={UniformBuffer.Buffer,0,sizeof(HybridRenderingConstants)};
    for (std::size_t slot=0;slot<RaySceneBufferCount;++slot)
        buffers[slot+1]={RaySceneBuffers[slot].Buffer,0,RaySceneBuffers[slot].Size};
    for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot)
        buffers[slot+1+RaySceneBufferCount]={RayRadianceBuffers[slot].Buffer,0,RayRadianceBuffers[slot].Size};
    std::array<VkDescriptorImageInfo,HybridRenderingTextureCount+2> images{};
    std::array<VkWriteDescriptorSet,HybridRenderingStorageBinding+HybridRenderingStorageSlots> writes{};
    for (std::uint32_t binding=0;binding<writes.size();++binding)
    {
        auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=descriptor; write.dstBinding=binding; write.descriptorCount=1;
        if (binding==0)
        { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&buffers[0]; }
        else if (binding<=HybridRenderingTextureCount)
        {
            images[binding-1]={0,views[binding-1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; write.pImageInfo=&images[binding-1];
        }
        else if (binding==HybridRenderingSamplerBinding)
        {
            images.back().sampler=DeferredSampler;
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER; write.pImageInfo=&images.back();
        }
        else if (binding==HybridRenderingAtlasBinding)
        {
            images[HybridRenderingTextureCount]={0,RayRadianceColorAtlas.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; write.pImageInfo=&images[HybridRenderingTextureCount];
        }
        else
        { write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo=&buffers[binding-HybridRenderingStorageBinding+1]; }
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
}
bool ApplyHybridRendering(const ReflectionDesc& desc)
{
    if (!EnsureHybridRenderingResources()) return false;
    HybridRenderingConstants constants;
    const auto& frame=FrameRayRadiance;
    if (!BuildHybridRenderingConstants(desc,FrameRayRadianceSettings,FrameRayRadianceSettings.Hybrid,frame,CreateInfo.Width,CreateInfo.Height,
        Techniques::HybridRenderingPass::Trace,constants)) return Fail("Vulkan hybrid reflection frame is invalid");
    std::array<std::uint32_t,HybridRenderingPassCount> offsets{};
    auto end=UniformOffset;
    for (auto& offset:offsets)
    {
        end=AlignUp(end,UniformAlignment);
        if (end>UniformBufferSize || sizeof(constants)>UniformBufferSize-end)
            return Fail("Vulkan hybrid reflection uniform arena is full");
        offset=static_cast<std::uint32_t>(end); end+=sizeof(constants);
    }
    UniformOffset=end;
    TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (auto& image:GBufferImages) TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (const auto pass:HybridRenderingPasses)
    {
        const auto index=static_cast<std::size_t>(pass);
        const bool composite=pass==Techniques::HybridRenderingPass::Composite;
        auto& previous=index ? HybridRenderingImages[(index-1)%HybridRenderingImages.size()] : DummyColor;
        auto& destination=composite ? HdrImages[0] : HybridRenderingImages[index%HybridRenderingImages.size()];
        TransitionImage(CommandBuffer,previous,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const std::array<VkImageView,HybridRenderingTextureCount> views=composite ?
            std::array<VkImageView,HybridRenderingTextureCount>{DummyColor.View,DummyColor.View,DummyColor.View,DummyColor.View,DummyColor.View,previous.View,DummyColor.View} :
            std::array<VkImageView,HybridRenderingTextureCount>{MainDepth.View,GBufferImages[1].View,GBufferImages[2].View,GBufferImages[0].View,GBufferImages[3].View,previous.View,HdrImages[0].View};
        UpdateHybridRenderingDescriptor(HybridRenderingDescriptors[index],views);
        constants.Options.X=static_cast<std::uint32_t>(pass);
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offsets[index],&constants,sizeof(constants));
        VkClearValue clear{}; VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass=composite ? ReflectionCompositeRenderPass : HdrRenderPass;
        begin.framebuffer=composite ? HdrFramebuffers[0] : HybridRenderingFramebuffers[index%HybridRenderingImages.size()];
        begin.renderArea={{0,0},{destination.Width,destination.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
        CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,static_cast<float>(destination.Width),static_cast<float>(destination.Height),0,1};
        CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,HybridRenderingPipelines[composite ? 1 : 0]);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,HybridRenderingPipelineLayout,0,1,&HybridRenderingDescriptors[index],1,&offsets[index]);
        CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
        PendingFrame.ReflectionPasses.push_back(HybridRenderingLogPass(pass)); ++PendingFrame.Statistics.HybridRenderingPassCount;
    }
    ReflectionResolved=true;
    PendingFrame.Statistics.ReflectionPassCount=static_cast<std::uint32_t>(PendingFrame.ReflectionPasses.size()); return true;
}
