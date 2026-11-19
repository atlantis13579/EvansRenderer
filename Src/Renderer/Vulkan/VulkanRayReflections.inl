// The GLSL source is compiled at runtime. Each pass owns a descriptor set so
// later ping-pong bindings cannot change descriptors referenced by earlier work.
static constexpr std::uint32_t RayReflectionTextureCount=7,RayReflectionPassCount=4;
static constexpr std::uint32_t RayReflectionSamplerBinding=RayReflectionTextureCount+1;
static constexpr std::uint32_t RayReflectionStorageBinding=RayReflectionSamplerBinding+1;
static constexpr std::uint32_t RayReflectionStorageSlots=RaySceneBufferCount+RayRadianceBufferCount;
static constexpr std::uint32_t RayReflectionStorageCount=RayReflectionStorageSlots-1;
static constexpr std::uint32_t RayReflectionAtlasBinding=RayReflectionStorageBinding+RaySceneBufferCount+RayRadianceAtlasSlot;
std::array<RawImage,2> RayReflectionImages{};
std::array<VkFramebuffer,2> RayReflectionFramebuffers{};
VkDescriptorSetLayout RayReflectionDescriptorLayout=0;
VkPipelineLayout RayReflectionPipelineLayout=0;
VkDescriptorPool RayReflectionDescriptorPool=0;
std::array<VkDescriptorSet,RayReflectionPassCount> RayReflectionDescriptors{};
std::array<VkPipeline,2> RayReflectionPipelines{};
std::array<VkShaderModule,2> RayReflectionModules{};

void DestroyRayReflectionTargets()
{
    for (auto& framebuffer:RayReflectionFramebuffers)
    { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,framebuffer,nullptr); framebuffer=0; }
    for (auto& image:RayReflectionImages) DestroyRawImage(image);
}
void DestroyRayReflections()
{
    DestroyRayReflectionTargets();
    for (auto pipeline:RayReflectionPipelines) if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr);
    for (auto module:RayReflectionModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    if (RayReflectionDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,RayReflectionDescriptorPool,nullptr);
    if (RayReflectionPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,RayReflectionPipelineLayout,nullptr);
    if (RayReflectionDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,RayReflectionDescriptorLayout,nullptr);
    RayReflectionPipelines={}; RayReflectionModules={}; RayReflectionDescriptors={};
    RayReflectionDescriptorPool=RayReflectionPipelineLayout=RayReflectionDescriptorLayout=0;
}

bool EnsureRayReflectionResources()
{
    if (!EnsureRayRadiance()) return false;
    if (!CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT,ReflectionCompositeRenderPass,true)) return false;
    if (RayReflectionImages[0].Width!=MainColor.Width || RayReflectionImages[0].Height!=MainColor.Height)
        DestroyRayReflectionTargets();
    for (std::size_t index=0;index<RayReflectionImages.size();++index)
    {
        if (RayReflectionFramebuffers[index]) continue;
        if (!CreateRawImage(MainColor.Width,MainColor.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,RayReflectionImages[index]) ||
            !CreateFramebuffer(HdrRenderPass,&RayReflectionImages[index].View,1,MainColor.Width,MainColor.Height,RayReflectionFramebuffers[index]))
        { DestroyRayReflectionTargets(); return false; }
    }
    if (!RayReflectionDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,RayReflectionStorageBinding+RayReflectionStorageSlots> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=1;slot<=RayReflectionTextureCount;++slot)
            bindings[slot]={slot,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        bindings[RayReflectionSamplerBinding]={RayReflectionSamplerBinding,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=RayReflectionStorageBinding;slot<bindings.size();++slot)
            bindings[slot]={slot,slot==RayReflectionAtlasBinding ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&RayReflectionDescriptorLayout),"vkCreateDescriptorSetLayout(ray reflection)")) return false;
    }
    if (!RayReflectionPipelineLayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create.setLayoutCount=1; create.pSetLayouts=&RayReflectionDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&RayReflectionPipelineLayout),"vkCreatePipelineLayout(ray reflection)")) return false;
    }
    if (!RayReflectionDescriptorPool)
    {
        const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,RayReflectionPassCount},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,(RayReflectionTextureCount+1)*RayReflectionPassCount},
            {VK_DESCRIPTOR_TYPE_SAMPLER,RayReflectionPassCount},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,RayReflectionStorageCount*RayReflectionPassCount}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=RayReflectionPassCount; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&RayReflectionDescriptorPool),"vkCreateDescriptorPool(ray reflection)")) return false;
    }
    for (auto& descriptor:RayReflectionDescriptors) if (!descriptor)
    {
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=RayReflectionDescriptorPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&RayReflectionDescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&descriptor),"vkAllocateDescriptorSets(ray reflection)")) return false;
    }
    return CreateFullscreenPipeline(VulkanRayTracedReflectionShaders,RayReflectionModules,HdrRenderPass,RayReflectionPipelines[0],RayReflectionPipelineLayout) &&
        CreateFullscreenPipeline(VulkanRayTracedReflectionShaders,RayReflectionModules,ReflectionCompositeRenderPass,
            RayReflectionPipelines[1],RayReflectionPipelineLayout,FullscreenBlend::AddRadiance,true);
}
void UpdateRayReflectionDescriptor(VkDescriptorSet descriptor,const std::array<VkImageView,RayReflectionTextureCount>& views)
{
    std::array<VkDescriptorBufferInfo,RayReflectionStorageSlots+1> buffers{};
    buffers[0]={UniformBuffer.Buffer,0,sizeof(RayTracedReflectionConstants)};
    for (std::size_t slot=0;slot<RaySceneBufferCount;++slot)
        buffers[slot+1]={RaySceneBuffers[slot].Buffer,0,RaySceneBuffers[slot].Size};
    for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot)
        buffers[slot+1+RaySceneBufferCount]={RayRadianceBuffers[slot].Buffer,0,RayRadianceBuffers[slot].Size};
    std::array<VkDescriptorImageInfo,RayReflectionTextureCount+2> images{};
    std::array<VkWriteDescriptorSet,RayReflectionStorageBinding+RayReflectionStorageSlots> writes{};
    for (std::uint32_t binding=0;binding<writes.size();++binding)
    {
        auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=descriptor; write.dstBinding=binding; write.descriptorCount=1;
        if (binding==0)
        { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&buffers[0]; }
        else if (binding<=RayReflectionTextureCount)
        {
            images[binding-1]={0,views[binding-1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; write.pImageInfo=&images[binding-1];
        }
        else if (binding==RayReflectionSamplerBinding)
        {
            images.back().sampler=DeferredSampler;
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER; write.pImageInfo=&images.back();
        }
        else if (binding==RayReflectionAtlasBinding)
        {
            images[RayReflectionTextureCount]={0,RayRadianceColorAtlas.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; write.pImageInfo=&images[RayReflectionTextureCount];
        }
        else
        { write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo=&buffers[binding-RayReflectionStorageBinding+1]; }
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
}
bool ApplyRayReflections(const ReflectionDesc& desc)
{
    if (!EnsureRayReflectionResources()) return false;
    RayTracedReflectionConstants constants;
    const auto& frame=FrameRayRadiance;
    if (!BuildRayTracedReflectionConstants(desc,FrameRayRadianceSettings,FrameReflections.RayTracing,
        *frame.Geometry,*frame.Shading,frame.Lighting,*frame.Lights,CreateInfo.Width,CreateInfo.Height,
        Techniques::RayReflectionPass::Trace,constants)) return Fail("Vulkan ray reflection frame is invalid");
    std::array<std::uint32_t,RayReflectionPassCount> offsets{};
    auto end=UniformOffset;
    for (auto& offset:offsets)
    {
        end=AlignUp(end,UniformAlignment);
        if (end>UniformBufferSize || sizeof(constants)>UniformBufferSize-end)
            return Fail("Vulkan ray reflection uniform arena is full");
        offset=static_cast<std::uint32_t>(end); end+=sizeof(constants);
    }
    UniformOffset=end;
    TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (auto& image:GBufferImages) TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (const auto pass:RayReflectionPasses)
    {
        const auto index=static_cast<std::size_t>(pass);
        const bool composite=pass==Techniques::RayReflectionPass::Composite;
        auto& previous=index ? RayReflectionImages[(index-1)%RayReflectionImages.size()] : DummyColor;
        auto& destination=composite ? HdrImages[0] : RayReflectionImages[index%RayReflectionImages.size()];
        TransitionImage(CommandBuffer,previous,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const std::array<VkImageView,RayReflectionTextureCount> views=composite ?
            std::array<VkImageView,RayReflectionTextureCount>{DummyColor.View,DummyColor.View,DummyColor.View,DummyColor.View,DummyColor.View,previous.View,DummyColor.View} :
            std::array<VkImageView,RayReflectionTextureCount>{MainDepth.View,GBufferImages[1].View,GBufferImages[2].View,GBufferImages[0].View,GBufferImages[3].View,previous.View,HdrImages[0].View};
        UpdateRayReflectionDescriptor(RayReflectionDescriptors[index],views);
        constants.Options.X=static_cast<std::uint32_t>(pass);
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offsets[index],&constants,sizeof(constants));
        VkClearValue clear{}; VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass=composite ? ReflectionCompositeRenderPass : HdrRenderPass;
        begin.framebuffer=composite ? HdrFramebuffers[0] : RayReflectionFramebuffers[index%RayReflectionImages.size()];
        begin.renderArea={{0,0},{destination.Width,destination.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
        CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,static_cast<float>(destination.Width),static_cast<float>(destination.Height),0,1};
        CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RayReflectionPipelines[composite ? 1 : 0]);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RayReflectionPipelineLayout,0,1,&RayReflectionDescriptors[index],1,&offsets[index]);
        CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
        PendingFrame.ReflectionPasses.push_back(RayReflectionLogPass(pass));
    }
    ReflectionResolved=true;
    PendingFrame.Statistics.ReflectionPassCount=static_cast<std::uint32_t>(PendingFrame.ReflectionPasses.size()); return true;
}
