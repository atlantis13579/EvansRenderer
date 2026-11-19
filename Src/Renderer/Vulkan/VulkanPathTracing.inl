// Trace and composition use distinct descriptor sets: updating a later pass
// cannot change resources referenced by an earlier recorded draw. Image layout
// changes use the frame's existing AbortFrame journal.
PathTracingFrameState PathFrame;
std::array<RawImage,2> PathMeans{};
RawImage PathDepth;
std::array<VkFramebuffer,2> PathFramebuffers{};
VkFramebuffer PathComposeFramebuffer=0;
std::array<VkRenderPass,2> PathRenderPasses{};
std::array<VkPipeline,2> PathPipelines{};
std::array<std::array<VkShaderModule,2>,2> PathModules{};
std::array<VkDescriptorSet,2> PathDescriptors{};
VkDescriptorSetLayout PathDescriptorLayout=0;
VkPipelineLayout PathPipelineLayout=0;
VkDescriptorPool PathDescriptorPool=0;
VkSampler PathSampler=0;
static constexpr std::uint32_t PathTextureCount=2,PathSamplerBinding=3,PathStorageBinding=4;
static constexpr std::uint32_t PathStorageSlots=RaySceneBufferCount+RayRadianceBufferCount;
static constexpr std::uint32_t PathAtlasBinding=PathStorageBinding+RaySceneBufferCount+RayRadianceAtlasSlot;

bool SupportsPathTracing() const
{
    if (!PhysicalDevice || !GetPhysicalDeviceFormatProperties || DeviceLimits.maxColorAttachments<2 ||
        DeviceLimits.maxPerStageDescriptorStorageBuffers<PathStorageSlots-1 ||
        DeviceLimits.maxDescriptorSetStorageBuffers<PathStorageSlots-1 ||
        DeviceLimits.maxFragmentCombinedOutputResources<PathStorageSlots+1) return false;
    constexpr VkFlags required=VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    for (auto format:{VK_FORMAT_R32G32B32A32_SFLOAT,VK_FORMAT_R32_SFLOAT})
    {
        VkFormatProperties properties{}; GetPhysicalDeviceFormatProperties(PhysicalDevice,format,&properties);
        if ((properties.optimalTilingFeatures&required)!=required) return false;
    }
    return true;
}
void DestroyPathTargets()
{
    for (auto& framebuffer:PathFramebuffers)
    { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,framebuffer,nullptr); framebuffer=0; }
    if (PathComposeFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,PathComposeFramebuffer,nullptr);
    PathComposeFramebuffer=0;
    for (auto& image:PathMeans) DestroyRawImage(image);
    DestroyRawImage(PathDepth); PathFrame.History.Invalidate();
}
void DestroyPathTracing()
{
    DestroyPathTargets();
    for (auto pipeline:PathPipelines) if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr);
    for (auto& modules:PathModules) for (auto module:modules)
        if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    if (PathDescriptorPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,PathDescriptorPool,nullptr);
    if (PathPipelineLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,PathPipelineLayout,nullptr);
    if (PathDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,PathDescriptorLayout,nullptr);
    if (PathSampler && DestroySamplerFunction) DestroySamplerFunction(Device,PathSampler,nullptr);
    for (auto pass:PathRenderPasses) if (pass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device,pass,nullptr);
    PathPipelines={}; PathModules={}; PathDescriptors={}; PathRenderPasses={};
    PathDescriptorPool=PathPipelineLayout=PathDescriptorLayout=PathSampler=0; PathFrame={};
}
bool EnsurePathRenderPasses()
{
    for (unsigned phase=0;phase<2;++phase)
    {
        if (PathRenderPasses[phase]) continue;
        std::array<VkAttachmentDescription,2> attachments{};
        for (unsigned slot=0;slot<attachments.size();++slot)
        {
            auto& attachment=attachments[slot];
            attachment.format=phase ? (slot ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT) :
                (slot ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT);
            attachment.samples=VK_SAMPLE_COUNT_1_BIT;
            attachment.loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout=attachment.finalLayout=phase && slot ?
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        const VkAttachmentReference colors[]{{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},{1,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
        const VkAttachmentReference depth{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount=phase ? 1 : 2; subpass.pColorAttachments=colors;
        subpass.pDepthStencilAttachment=phase ? &depth : nullptr;
        VkRenderPassCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        create.attachmentCount=static_cast<std::uint32_t>(attachments.size()); create.pAttachments=attachments.data();
        create.subpassCount=1; create.pSubpasses=&subpass;
        if (!Check(CreateRenderPassFunction(Device,&create,nullptr,&PathRenderPasses[phase]),"vkCreateRenderPass(path tracing)")) return false;
    }
    return true;
}
bool EnsurePathTracing()
{
    if (!SupportsPathTracing()) return Fail("Vulkan path tracing requires FP32 attachments and scene storage buffers");
    if (!EnsureRayRadiance() || !EnsurePathRenderPasses()) return false;
    if (PathDepth.Width!=MainColor.Width || PathDepth.Height!=MainColor.Height) DestroyPathTargets();
    constexpr VkFlags usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!PathDepth.Image && !CreateRawImage(MainColor.Width,MainColor.Height,VK_FORMAT_R32_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,usage,PathDepth)) return false;
    for (std::size_t index=0;index<PathMeans.size();++index)
    {
        if (PathFramebuffers[index]) continue;
        if (!PathMeans[index].Image && !CreateRawImage(MainColor.Width,MainColor.Height,VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT,usage,PathMeans[index])) return false;
        const VkImageView views[]{PathMeans[index].View,PathDepth.View};
        if (!CreateFramebuffer(PathRenderPasses[0],views,2,MainColor.Width,MainColor.Height,PathFramebuffers[index])) return false;
    }
    if (!PathComposeFramebuffer)
    {
        const VkImageView views[]{HdrImages[0].View,MainDepth.View};
        if (!CreateFramebuffer(PathRenderPasses[1],views,2,MainColor.Width,MainColor.Height,PathComposeFramebuffer)) return false;
    }
    if (!PathSampler)
    {
        VkSamplerCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        create.magFilter=create.minFilter=VK_FILTER_NEAREST;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&PathSampler),"vkCreateSampler(path tracing)")) return false;
    }
    if (!PathDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,PathStorageBinding+PathStorageSlots> bindings{};
        for (std::uint32_t slot=0;slot<bindings.size();++slot)
        {
            const auto type=slot==0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC :
                slot==PathSamplerBinding ? VK_DESCRIPTOR_TYPE_SAMPLER :
                slot<PathStorageBinding || slot==PathAtlasBinding ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[slot]={slot,type,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        }
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&PathDescriptorLayout),"vkCreateDescriptorSetLayout(path tracing)")) return false;
    }
    if (!PathPipelineLayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create.setLayoutCount=1; create.pSetLayouts=&PathDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&PathPipelineLayout),"vkCreatePipelineLayout(path tracing)")) return false;
    }
    if (!PathDescriptorPool)
    {
        constexpr std::uint32_t phases=2;
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,phases},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,(PathTextureCount+1)*phases},
            {VK_DESCRIPTOR_TYPE_SAMPLER,phases},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,(PathStorageSlots-1)*phases}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=phases; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&PathDescriptorPool),"vkCreateDescriptorPool(path tracing)")) return false;
    }
    for (auto& descriptor:PathDescriptors) if (!descriptor)
    {
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=PathDescriptorPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&PathDescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&descriptor),"vkAllocateDescriptorSets(path tracing)")) return false;
    }
    return CreateFullscreenPipeline(VulkanPathTracingShaders,PathModules[0],PathRenderPasses[0],PathPipelines[0],PathPipelineLayout,
        FullscreenBlend::Replace,false,1,false,2) &&
        CreateFullscreenPipeline(VulkanPathTracingComposeShaders,PathModules[1],PathRenderPasses[1],PathPipelines[1],PathPipelineLayout,
            FullscreenBlend::Replace,false,1,false,1,true);
}
void UpdatePathDescriptor(VkDescriptorSet descriptor,VkImageView mean,VkImageView depth)
{
    std::array<VkDescriptorBufferInfo,PathStorageSlots+1> buffers{};
    buffers[0]={UniformBuffer.Buffer,0,sizeof(PathTracingConstants)};
    for (std::size_t slot=0;slot<RaySceneBufferCount;++slot) buffers[slot+1]={RaySceneBuffers[slot].Buffer,0,RaySceneBuffers[slot].Size};
    for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot)
        buffers[slot+1+RaySceneBufferCount]={RayRadianceBuffers[slot].Buffer,0,RayRadianceBuffers[slot].Size};
    const VkDescriptorImageInfo images[]{{0,mean,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},{0,depth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {PathSampler,0,VK_IMAGE_LAYOUT_UNDEFINED},{0,RayRadianceColorAtlas.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    std::array<VkWriteDescriptorSet,PathStorageBinding+PathStorageSlots> writes{};
    for (std::uint32_t binding=0;binding<writes.size();++binding)
    {
        auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=descriptor; write.dstBinding=binding; write.descriptorCount=1;
        if (binding==0) { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&buffers[0]; }
        else if (binding<PathStorageBinding || binding==PathAtlasBinding)
        {
            write.descriptorType=binding==PathSamplerBinding ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            write.pImageInfo=&images[binding==PathAtlasBinding ? PathSamplerBinding : binding-1];
        }
        else { write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo=&buffers[binding-PathStorageBinding+1]; }
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
}
bool TracePaths(const PathTracingDesc& desc)
{
    PathTracingConstants constants;
    if (!PathFrame.Prepare(desc,FrameRayRadianceSettings,FrameRayRadiance,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("Vulkan path frame requires matched scene inputs and a finite invertible camera");
    const auto plan=PathFrame.History.GetPlan();
    const auto offset=AlignUp(UniformOffset,UniformAlignment);
    if (offset>UniformBufferSize || sizeof(constants)>UniformBufferSize-offset) return Fail("Vulkan path uniform arena is full");
    std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&constants,sizeof(constants));
    UniformOffset=offset+sizeof(constants); const auto dynamicOffset=static_cast<std::uint32_t>(offset);
    for (unsigned phase=0;phase<2;++phase)
    {
        auto& mean=phase ? PathMeans[plan.WriteIndex] : (plan.PreviousSamples ? PathMeans[plan.ReadIndex] : DummyColor);
        auto& destination=phase ? HdrImages[0] : PathMeans[plan.WriteIndex];
        TransitionImage(CommandBuffer,mean,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        TransitionImage(CommandBuffer,PathDepth,phase ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (phase) TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        UpdatePathDescriptor(PathDescriptors[phase],mean.View,phase ? PathDepth.View : DummyColor.View);
        VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass=PathRenderPasses[phase]; begin.framebuffer=phase ? PathComposeFramebuffer : PathFramebuffers[plan.WriteIndex];
        begin.renderArea={{0,0},{MainColor.Width,MainColor.Height}};
        CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,float(MainColor.Width),float(MainColor.Height),0,1};
        CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,PathPipelines[phase]);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,PathPipelineLayout,0,1,&PathDescriptors[phase],1,&dynamicOffset);
        CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    }
    ForwardOpaqueComplete=true; return PathFrame.Resolve(PendingFrame.Statistics);
}
