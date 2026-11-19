// The GLSL source is compiled at runtime. Each pass owns a descriptor set so
// later ping-pong bindings cannot change descriptors referenced by earlier work.
static constexpr std::uint32_t RayGITextureCount=6,RayGIPassCount=4;
static constexpr std::uint32_t RayGISamplerBinding=RayGITextureCount+1;
static constexpr std::uint32_t RayGIStorageBinding=RayGISamplerBinding+1;
static constexpr std::uint32_t RayGIStorageSlots=RaySceneBufferCount+RayRadianceBufferCount;
static constexpr std::uint32_t RayGIStorageCount=RayGIStorageSlots-1;
static constexpr std::uint32_t RayGIAtlasBinding=RayGIStorageBinding+RaySceneBufferCount+RayRadianceAtlasSlot;
std::array<RawImage,2> RayGIImages{};
std::array<VkFramebuffer,2> RayGIFramebuffers{};
VkRenderPass RayGICompositeRenderPass=0;
VkDescriptorSetLayout RayGIDescriptorLayout=0;
VkPipelineLayout RayGIPipelineLayout=0;
VkDescriptorPool RayGIDescriptorPool=0;
std::array<VkDescriptorSet,RayGIPassCount> RayGIDescriptors{};
std::array<VkPipeline,2> RayGIPipelines{};
std::array<VkShaderModule,2> RayGIModules{};

void DestroyRayGITargets()
{
    for (auto& framebuffer:RayGIFramebuffers)
    { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,framebuffer,nullptr); framebuffer=0; }
    for (auto& image:RayGIImages) DestroyRawImage(image);
}
void DestroyRayGI()
{
    DestroyRayGITargets();
    for (auto pipeline:RayGIPipelines) if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr);
    for (auto module:RayGIModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    if (RayGIDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,RayGIDescriptorPool,nullptr);
    if (RayGIPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,RayGIPipelineLayout,nullptr);
    if (RayGIDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,RayGIDescriptorLayout,nullptr);
    if (RayGICompositeRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device,RayGICompositeRenderPass,nullptr);
    RayGICompositeRenderPass=0;
    RayGIPipelines={}; RayGIModules={}; RayGIDescriptors={};
    RayGIDescriptorPool=RayGIPipelineLayout=RayGIDescriptorLayout=0;
}

bool EnsureRayGIResources()
{
    if (!EnsureRayRadiance()) return false;
    if (!CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT,RayGICompositeRenderPass,true)) return false;
    if (RayGIImages[0].Width!=MainColor.Width || RayGIImages[0].Height!=MainColor.Height)
        DestroyRayGITargets();
    for (std::size_t index=0;index<RayGIImages.size();++index)
    {
        if (RayGIFramebuffers[index]) continue;
        if (!CreateRawImage(MainColor.Width,MainColor.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,RayGIImages[index]) ||
            !CreateFramebuffer(HdrRenderPass,&RayGIImages[index].View,1,MainColor.Width,MainColor.Height,RayGIFramebuffers[index]))
        { DestroyRayGITargets(); return false; }
    }
    if (!RayGIDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,RayGIStorageBinding+RayGIStorageSlots> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=1;slot<=RayGITextureCount;++slot)
            bindings[slot]={slot,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        bindings[RayGISamplerBinding]={RayGISamplerBinding,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=RayGIStorageBinding;slot<bindings.size();++slot)
            bindings[slot]={slot,slot==RayGIAtlasBinding ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&RayGIDescriptorLayout),"vkCreateDescriptorSetLayout(ray GI)")) return false;
    }
    if (!RayGIPipelineLayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create.setLayoutCount=1; create.pSetLayouts=&RayGIDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&RayGIPipelineLayout),"vkCreatePipelineLayout(ray GI)")) return false;
    }
    if (!RayGIDescriptorPool)
    {
        const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,RayGIPassCount},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,(RayGITextureCount+1)*RayGIPassCount},
            {VK_DESCRIPTOR_TYPE_SAMPLER,RayGIPassCount},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,RayGIStorageCount*RayGIPassCount}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=RayGIPassCount; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&RayGIDescriptorPool),"vkCreateDescriptorPool(ray GI)")) return false;
    }
    for (auto& descriptor:RayGIDescriptors) if (!descriptor)
    {
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=RayGIDescriptorPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&RayGIDescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&descriptor),"vkAllocateDescriptorSets(ray GI)")) return false;
    }
    return CreateFullscreenPipeline(VulkanRayTracedGIShaders,RayGIModules,HdrRenderPass,RayGIPipelines[0],RayGIPipelineLayout) &&
        CreateFullscreenPipeline(VulkanRayTracedGIShaders,RayGIModules,RayGICompositeRenderPass,
            RayGIPipelines[1],RayGIPipelineLayout,FullscreenBlend::AddRadiance,true);
}
void UpdateRayGIDescriptor(VkDescriptorSet descriptor,const std::array<VkImageView,RayGITextureCount>& views)
{
    std::array<VkDescriptorBufferInfo,RayGIStorageSlots+1> buffers{};
    buffers[0]={UniformBuffer.Buffer,0,sizeof(RayTracedGIConstants)};
    for (std::size_t slot=0;slot<RaySceneBufferCount;++slot)
        buffers[slot+1]={RaySceneBuffers[slot].Buffer,0,RaySceneBuffers[slot].Size};
    for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot)
        buffers[slot+1+RaySceneBufferCount]={RayRadianceBuffers[slot].Buffer,0,RayRadianceBuffers[slot].Size};
    std::array<VkDescriptorImageInfo,RayGITextureCount+2> images{};
    std::array<VkWriteDescriptorSet,RayGIStorageBinding+RayGIStorageSlots> writes{};
    for (std::uint32_t binding=0;binding<writes.size();++binding)
    {
        auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=descriptor; write.dstBinding=binding; write.descriptorCount=1;
        if (binding==0)
        { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&buffers[0]; }
        else if (binding<=RayGITextureCount)
        {
            images[binding-1]={0,views[binding-1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; write.pImageInfo=&images[binding-1];
        }
        else if (binding==RayGISamplerBinding)
        {
            images.back().sampler=DeferredSampler;
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER; write.pImageInfo=&images.back();
        }
        else if (binding==RayGIAtlasBinding)
        {
            images[RayGITextureCount]={0,RayRadianceColorAtlas.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; write.pImageInfo=&images[RayGITextureCount];
        }
        else
        { write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo=&buffers[binding-RayGIStorageBinding+1]; }
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
}
bool ApplyRayGI(const GlobalIlluminationDesc& desc)
{
    if (!EnsureRayGIResources()) return false;
    RayTracedGIConstants constants;
    const auto& frame=FrameRayRadiance;
    if (!BuildRayTracedGIConstants(desc,FrameRayRadianceSettings,FrameGI.RayTracing,frame,
        CreateInfo.Width,CreateInfo.Height,Techniques::RayGIPass::Trace,constants)) return Fail("Vulkan ray GI frame is invalid");
    std::array<std::uint32_t,RayGIPassCount> offsets{};
    auto end=UniformOffset;
    for (auto& offset:offsets)
    {
        end=AlignUp(end,UniformAlignment);
        if (end>UniformBufferSize || sizeof(constants)>UniformBufferSize-end)
            return Fail("Vulkan ray GI uniform arena is full");
        offset=static_cast<std::uint32_t>(end); end+=sizeof(constants);
    }
    UniformOffset=end;
    TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (auto& image:GBufferImages) TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (const auto pass:RayGIPasses)
    {
        const auto index=static_cast<std::size_t>(pass);
        const bool composite=pass==Techniques::RayGIPass::Composite;
        auto& previous=index ? RayGIImages[(index-1)%RayGIImages.size()] : DummyColor;
        auto& destination=composite ? HdrImages[0] : RayGIImages[index%RayGIImages.size()];
        TransitionImage(CommandBuffer,previous,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const std::array<VkImageView,RayGITextureCount> views=composite ?
            std::array<VkImageView,RayGITextureCount>{DummyColor.View,DummyColor.View,DummyColor.View,DummyColor.View,previous.View,DummyColor.View} :
            std::array<VkImageView,RayGITextureCount>{MainDepth.View,GBufferImages[1].View,GBufferImages[0].View,GBufferImages[3].View,previous.View,HdrImages[0].View};
        UpdateRayGIDescriptor(RayGIDescriptors[index],views);
        constants.Options.X=static_cast<std::uint32_t>(pass);
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offsets[index],&constants,sizeof(constants));
        VkClearValue clear{}; VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass=composite ? RayGICompositeRenderPass : HdrRenderPass;
        begin.framebuffer=composite ? HdrFramebuffers[0] : RayGIFramebuffers[index%RayGIImages.size()];
        begin.renderArea={{0,0},{destination.Width,destination.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
        CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,static_cast<float>(destination.Width),static_cast<float>(destination.Height),0,1};
        CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RayGIPipelines[composite ? 1 : 0]);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RayGIPipelineLayout,0,1,&RayGIDescriptors[index],1,&offsets[index]);
        CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
        PendingFrame.RayGIPasses.push_back(pass);
    }
    GI.Resolved=true;
    PendingFrame.Statistics.GlobalIlluminationPassCount=static_cast<std::uint32_t>(PendingFrame.RayGIPasses.size()); return true;
}
