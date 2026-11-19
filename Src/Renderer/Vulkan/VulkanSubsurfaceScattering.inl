// Four private HDR images and three immutable-per-frame descriptor sets. The
// second deferred-lighting descriptor never rewrites the scene draw's set.
enum SubsurfaceImage : std::size_t { SubsurfaceScene, SubsurfaceSource, SubsurfaceHorizontal, SubsurfaceVertical, SubsurfaceImageCount };
static constexpr std::uint32_t SubsurfaceInputCount=5,SubsurfaceFilterDraws=3;
std::array<RawImage,SubsurfaceImageCount> SubsurfaceImages{};
std::array<VkFramebuffer,SubsurfaceImageCount> SubsurfaceFramebuffers{};
VkDescriptorSetLayout SubsurfaceDescriptorLayout=0;
VkPipelineLayout SubsurfaceLayout=0;
VkDescriptorPool SubsurfacePool=0,SubsurfaceSourcePool=0;
std::array<VkDescriptorSet,SubsurfaceFilterDraws> SubsurfaceDescriptors{};
VkDescriptorSet SubsurfaceSourceDescriptor=0;
VkPipeline SubsurfacePipeline=0;
std::array<VkShaderModule,2> SubsurfaceModules{};
bool SubsurfaceProducing=false,SubsurfaceResolved=false,SubsurfaceHasScene=false;
DeferredLightingDesc SubsurfaceLighting;
Techniques::SubsurfaceScatteringSettings FrameSubsurface;
bool SubsurfaceRequested=false;
bool SubsurfaceComplete() const { return !SubsurfaceRequested || SubsurfaceResolved; }

void DestroySubsurfaceTargets()
{
    for (auto& framebuffer:SubsurfaceFramebuffers)
    { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,framebuffer,nullptr); framebuffer=0; }
    for (auto& image:SubsurfaceImages) DestroyRawImage(image);
}
void DestroySubsurface()
{
    DestroySubsurfaceTargets();
    if (SubsurfacePipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,SubsurfacePipeline,nullptr);
    for (auto module:SubsurfaceModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    for (auto pool:{SubsurfacePool,SubsurfaceSourcePool}) if (pool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,pool,nullptr);
    if (SubsurfaceLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,SubsurfaceLayout,nullptr);
    if (SubsurfaceDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,SubsurfaceDescriptorLayout,nullptr);
    SubsurfacePipeline=SubsurfacePool=SubsurfaceSourcePool=SubsurfaceLayout=SubsurfaceDescriptorLayout=SubsurfaceSourceDescriptor=0;
    SubsurfaceModules={}; SubsurfaceDescriptors={};
}
bool EnsureSubsurface()
{
    if (SubsurfaceImages[0].Width!=MainColor.Width || SubsurfaceImages[0].Height!=MainColor.Height) DestroySubsurfaceTargets();
    for (std::size_t index=0;index<SubsurfaceImageCount;++index)
    {
        if (SubsurfaceFramebuffers[index]) continue;
        constexpr VkFlags Usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (!CreateRawImage(MainColor.Width,MainColor.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,Usage,SubsurfaceImages[index]) ||
            !CreateFramebuffer(HdrRenderPass,&SubsurfaceImages[index].View,1,MainColor.Width,MainColor.Height,SubsurfaceFramebuffers[index])) return false;
    }
    if (!SubsurfaceDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,SubsurfaceInputCount+1> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t binding=1;binding<bindings.size();++binding)
            bindings[binding]={binding,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&SubsurfaceDescriptorLayout),"vkCreateDescriptorSetLayout(subsurface)")) return false;
    }
    if (!SubsurfaceLayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create.setLayoutCount=1; create.pSetLayouts=&SubsurfaceDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&SubsurfaceLayout),"vkCreatePipelineLayout(subsurface)")) return false;
    }
    if (!SubsurfacePool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,SubsurfaceFilterDraws},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,SubsurfaceFilterDraws*SubsurfaceInputCount}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=SubsurfaceFilterDraws; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&SubsurfacePool),"vkCreateDescriptorPool(subsurface)")) return false;
    }
    for (auto& descriptor:SubsurfaceDescriptors) if (!descriptor)
    {
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=SubsurfacePool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&SubsurfaceDescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&descriptor),"vkAllocateDescriptorSets(subsurface)")) return false;
    }
    if (!SubsurfaceSourcePool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,static_cast<std::uint32_t>(GBufferAttachmentCount)+3},
            {VK_DESCRIPTOR_TYPE_SAMPLER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,ForwardPlusStorageCount}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=1; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&SubsurfaceSourcePool),"vkCreateDescriptorPool(subsurface light producer)")) return false;
    }
    if (!SubsurfaceSourceDescriptor)
    {
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=SubsurfaceSourcePool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&DeferredDescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&SubsurfaceSourceDescriptor),"vkAllocateDescriptorSets(subsurface light producer)")) return false;
    }
    return CreateFullscreenPipeline(VulkanSubsurfaceScatteringShaders,SubsurfaceModules,HdrRenderPass,SubsurfacePipeline,SubsurfaceLayout);
}
void SubsurfaceCopy(RawImage& source,RawImage& destination)
{
    TransitionImage(CommandBuffer,source,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={source.Width,source.Height,1};
    CmdCopyImage(CommandBuffer,source.Image,source.Layout,destination.Image,destination.Layout,1,&copy);
}
bool SnapshotSubsurfaceScene()
{
    if (!EnsureSubsurface()) return false;
    SubsurfaceCopy(HdrImages[0],SubsurfaceImages[SubsurfaceScene]); return true;
}
bool CaptureSubsurfaceSource()
{
    if (FrameSampleCount<=1) SubsurfaceCopy(HdrImages[0],SubsurfaceImages[SubsurfaceSource]);
    else
    {
        auto& source=Multisample.Images[SampleColor]; auto& destination=SubsurfaceImages[SubsurfaceSource];
        TransitionImage(CommandBuffer,source,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageResolve resolve{}; resolve.srcSubresource=resolve.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; resolve.extent={source.Width,source.Height,1};
        CmdResolveImage(CommandBuffer,source.Image,source.Layout,destination.Image,destination.Layout,1,&resolve);
    }
    return true;
}
bool FilterSubsurface(SubsurfaceScatteringConstants constants)
{
    std::array<std::uint32_t,SubsurfaceFilterDraws> offsets{}; auto end=UniformOffset;
    for (auto& offset:offsets)
    {
        end=AlignUp(end,UniformAlignment);
        if (end>UniformBufferSize || sizeof(constants)>UniformBufferSize-end) return Fail("Vulkan subsurface uniform arena is full");
        offset=static_cast<std::uint32_t>(end); end+=sizeof(constants);
    }
    UniformOffset=end;
    TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,GBufferImages[1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,SubsurfaceImages[SubsurfaceScene],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,SubsurfaceImages[SubsurfaceSource],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    std::size_t draw=0;
    for (auto pass:{Techniques::SubsurfacePass::Horizontal,Techniques::SubsurfacePass::Vertical,Techniques::SubsurfacePass::Composite})
    {
        const bool horizontal=pass==Techniques::SubsurfacePass::Horizontal,composite=pass==Techniques::SubsurfacePass::Composite;
        auto& source=SubsurfaceImages[horizontal ? SubsurfaceSource : composite ? SubsurfaceVertical : SubsurfaceHorizontal];
        auto& destination=composite ? HdrImages[0] : SubsurfaceImages[horizontal ? SubsurfaceHorizontal : SubsurfaceVertical];
        TransitionImage(CommandBuffer,source,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const std::array<VkImageView,SubsurfaceInputCount> views{source.View,MainDepth.View,GBufferImages[1].View,SubsurfaceImages[SubsurfaceScene].View,SubsurfaceImages[SubsurfaceSource].View};
        VkDescriptorBufferInfo buffer{UniformBuffer.Buffer,0,sizeof(constants)};
        std::array<VkDescriptorImageInfo,SubsurfaceInputCount> images{};
        std::array<VkWriteDescriptorSet,SubsurfaceInputCount+1> writes{};
        for (std::uint32_t binding=0;binding<writes.size();++binding)
        {
            auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=SubsurfaceDescriptors[draw]; write.dstBinding=binding; write.descriptorCount=1;
            if (!binding) { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&buffer; }
            else { images[binding-1]={DeferredSampler,views[binding-1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&images[binding-1]; }
        }
        UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
        constants.Options.X=static_cast<unsigned>(pass); std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offsets[draw],&constants,sizeof(constants));
        VkClearValue clear{}; VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO; begin.renderPass=HdrRenderPass;
        begin.framebuffer=composite ? HdrFramebuffers[0] : SubsurfaceFramebuffers[horizontal ? SubsurfaceHorizontal : SubsurfaceVertical];
        begin.renderArea={{0,0},{destination.Width,destination.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
        CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,float(destination.Width),float(destination.Height),0,1}; CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,SubsurfacePipeline);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,SubsurfaceLayout,0,1,&SubsurfaceDescriptors[draw],1,&offsets[draw]);
        CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer); ++draw; ++PendingFrame.Statistics.SubsurfacePassCount;
    }
    SubsurfaceResolved=true; return true;
}
