bool SupportsTemporalExposure() const
{
    if (!PhysicalDevice || !GetPhysicalDeviceFormatProperties) return false;
    VkFormatProperties support{}; GetPhysicalDeviceFormatProperties(PhysicalDevice,VK_FORMAT_R32G32B32A32_SFLOAT,&support);
    constexpr VkFlags required=VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    return (support.optimalTilingFeatures&required)==required;
}
// Every reduction draw owns its descriptor set and UBO slice. Rewriting one
// shared set would make all recorded levels sample the final level's input.
TemporalExposureHistory ExposureHistory;
struct ExposureSurface { RawImage Image; VkFramebuffer Framebuffer=0; };
std::vector<ExposureSurface> ExposureLevels;
std::array<ExposureSurface,2> ExposureValues{};
VkRenderPass ExposureRenderPass=0;
VkDescriptorSetLayout ExposureSetLayout=0;
VkPipelineLayout ExposureLayout=0;
VkDescriptorPool ExposurePool=0;
VkSampler ExposureSampler=0;
std::vector<VkDescriptorSet> ExposureSets;
std::array<VkPipeline,2> ExposurePipelines{};
std::array<VkShaderModule,2> ExposureModules{};
int ExposureWidth=0,ExposureHeight=0;
void DestroyExposureTargets()
{
    const auto release=[&](ExposureSurface& surface)
    {
        if (surface.Framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,surface.Framebuffer,nullptr);
        DestroyRawImage(surface.Image); surface={};
    };
    for (auto& image:ExposureLevels) release(image);
    for (auto& image:ExposureValues) release(image);
    ExposureLevels.clear(); ExposureWidth=ExposureHeight=0; ExposureHistory.Invalidate();
}
void DestroyExposure()
{
    DestroyExposureTargets();
    for (auto pipeline:ExposurePipelines) if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr);
    for (auto module:ExposureModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    if (ExposurePool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,ExposurePool,nullptr);
    if (ExposureLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,ExposureLayout,nullptr);
    if (ExposureSetLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,ExposureSetLayout,nullptr);
    if (ExposureRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device,ExposureRenderPass,nullptr);
    if (ExposureSampler && DestroySamplerFunction) DestroySamplerFunction(Device,ExposureSampler,nullptr);
    ExposurePool=ExposureLayout=ExposureSetLayout=ExposureRenderPass=ExposureSampler=0; ExposureSets.clear(); ExposurePipelines={}; ExposureModules={};
}
bool EnsureExposure()
{
    if (!ExposureRenderPass && !CreateFullscreenRenderPass(VK_FORMAT_R32G32B32A32_SFLOAT,ExposureRenderPass)) return false;
    if (!ExposureSetLayout)
    {
        const VkDescriptorSetLayoutBinding bindings[]{{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=static_cast<unsigned>(std::size(bindings)); create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&ExposureSetLayout),"Create exposure descriptor layout")) return false;
    }
    if (!ExposureLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&ExposureSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&ExposureLayout),"Create exposure pipeline layout")) return false;
    }
    if (!ExposureSampler)
    {
        VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; create.magFilter=create.minFilter=VK_FILTER_NEAREST;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&ExposureSampler),"Create exposure point sampler")) return false;
    }
    constexpr unsigned MaximumDraws=std::numeric_limits<int>::digits+2; // ceil-half levels plus adaptation/composite
    if (!ExposurePool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,MaximumDraws},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,MaximumDraws*2}};
        VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; create.maxSets=MaximumDraws; create.poolSizeCount=static_cast<unsigned>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&ExposurePool),"Create exposure descriptor pool")) return false;
    }
    while (ExposureSets.size()<MaximumDraws)
    {
        VkDescriptorSet set=0; VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool=ExposurePool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&ExposureSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&set),"Allocate exposure descriptor")) return false;
        ExposureSets.push_back(set);
    }
    if (!CreateFullscreenPipeline(VulkanTemporalEffectsShaders,ExposureModules,ExposureRenderPass,ExposurePipelines[0],ExposureLayout) ||
        !CreateFullscreenPipeline(VulkanTemporalEffectsShaders,ExposureModules,HdrRenderPass,ExposurePipelines[1],ExposureLayout)) return false;
    if (ExposureWidth==CreateInfo.Width && ExposureHeight==CreateInfo.Height) return true;
    DestroyExposureTargets(); const auto plan=BuildExposureReductionPlan(CreateInfo.Width,CreateInfo.Height); ExposureLevels.resize(plan.size());
    const auto create=[&](ExposureSurface& surface,int width,int height)
    {
        return CreateRawImage(width,height,VK_FORMAT_R32G32B32A32_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,surface.Image) &&
            CreateFramebuffer(ExposureRenderPass,&surface.Image.View,1,width,height,surface.Framebuffer);
    };
    for (std::size_t index=0;index<plan.size();++index) if (!create(ExposureLevels[index],plan[index].Width,plan[index].Height)) return false;
    for (auto& value:ExposureValues) if (!create(value,1,1)) return false;
    ExposureWidth=CreateInfo.Width; ExposureHeight=CreateInfo.Height; return true;
}
bool ResolveExposure(const TemporalEffectsDesc& desc)
{
    TemporalEffectsConstants constants;
    if (!ExposureHistory.Prepare(desc,FramePostProcessing.TemporalEffects,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("Exposure requires finite scene time and positive dimensions");
    std::size_t drawIndex=0;
    const auto draw=[&](RawImage& source,RawImage& previous,RawImage& destination,VkFramebuffer framebuffer,Techniques::TemporalEffectsPassKind phase)
    {
        const auto offset=AlignUp(UniformOffset,UniformAlignment);
        if (offset>UniformBufferSize || sizeof(constants)>UniformBufferSize-offset || drawIndex>=ExposureSets.size()) return Fail("Exposure draw storage is exhausted");
        constants.PhaseExtentHistory.X=static_cast<unsigned>(phase); constants.PhaseExtentHistory.Y=source.Width; constants.PhaseExtentHistory.Z=source.Height;
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&constants,sizeof(constants)); UniformOffset=offset+sizeof(constants);
        TransitionImage(CommandBuffer,source,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); TransitionImage(CommandBuffer,previous,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const VkDescriptorBufferInfo buffer{UniformBuffer.Buffer,0,sizeof(constants)};
        const VkDescriptorImageInfo inputs[]{{ExposureSampler,source.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},{ExposureSampler,previous.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        std::array<VkWriteDescriptorSet,3> writes{}; const auto set=ExposureSets[drawIndex++];
        for (unsigned binding=0;binding<writes.size();++binding)
        {
            auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=set; write.dstBinding=binding; write.descriptorCount=1;
            if (!binding) { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&buffer; }
            else { write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&inputs[binding-1]; }
        }
        UpdateDescriptorSets(Device,static_cast<unsigned>(writes.size()),writes.data(),0,nullptr);
        const bool composite=phase==Techniques::TemporalEffectsPassKind::Composite;
        VkClearValue clear{}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        begin.renderPass=composite?HdrRenderPass:ExposureRenderPass; begin.framebuffer=framebuffer; begin.renderArea={{0,0},{destination.Width,destination.Height}};
        begin.clearValueCount=1; begin.pClearValues=&clear; CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,float(destination.Width),float(destination.Height),0,1}; CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,ExposurePipelines[composite?1:0]); const auto dynamicOffset=static_cast<std::uint32_t>(offset);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,ExposureLayout,0,1,&set,1,&dynamicOffset);
        CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer); ++PendingFrame.Statistics.TemporalEffectsPassCount; return true;
    };
    auto* source=&HdrImages[0];
    for (std::size_t index=0;index<ExposureLevels.size();++index)
    {
        auto& level=ExposureLevels[index];
        if (!draw(*source,*source,level.Image,level.Framebuffer,index?Techniques::TemporalEffectsPassKind::Reduce:Techniques::TemporalEffectsPassKind::Meter)) return false;
        source=&level.Image;
    }
    auto& current=ExposureValues[1-ExposureHistory.ReadIndex()];
    if (!draw(*source,constants.PhaseExtentHistory.W?ExposureValues[ExposureHistory.ReadIndex()].Image:*source,current.Image,current.Framebuffer,Techniques::TemporalEffectsPassKind::Adapt) ||
        !draw(HdrImages[0],current.Image,HdrImages[1],HdrFramebuffers[1],Techniques::TemporalEffectsPassKind::Composite)) return false;
    SubsurfaceCopy(HdrImages[1],HdrImages[0]); ExposureHistory.Resolved=true; return true;
}
