// GLSL remains runtime source. Completed CPU volumes upload through an immediate
// transaction before frame recording. Four descriptor sets keep draws immutable.
VoxelGIFrame FrameVoxelGI;
std::shared_ptr<const VoxelRadianceVolume> UploadedVoxelVolume;
RawImage VoxelVolume;
std::array<RawImage,2> VoxelGIImages{};
std::array<VkFramebuffer,2> VoxelGIFramebuffers{};
VkRenderPass VoxelGICompositeRenderPass=0;
VkDescriptorSetLayout VoxelGIDescriptorLayout=0;
VkPipelineLayout VoxelGILayout=0;
VkDescriptorPool VoxelGIPool=0;
VkSampler VoxelGISampler=0;
std::array<VkDescriptorSet,4> VoxelGIDescriptors{};
std::array<VkPipeline,2> VoxelGIPipelines{};
std::array<VkShaderModule,2> VoxelGIModules{};
static constexpr std::uint32_t VoxelGIInputCount=7,VoxelGIDrawCount=4;
bool SupportsVoxelGI() const
{
    if (!Device || !GetPhysicalDeviceFormatProperties || !GetPhysicalDeviceImageFormatProperties) return false;
    VkFormatProperties format{}; VkImageFormatProperties image{};
    GetPhysicalDeviceFormatProperties(PhysicalDevice,VK_FORMAT_R32G32B32A32_SFLOAT,&format);
    return (format.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)!=0 &&
        DeviceLimits.maxPerStageDescriptorSampledImages>=VoxelGIInputCount && DeviceLimits.maxPerStageDescriptorSamplers>=VoxelGIInputCount &&
        DeviceLimits.maxDescriptorSetSampledImages>=VoxelGIInputCount && DeviceLimits.maxDescriptorSetSamplers>=VoxelGIInputCount &&
        DeviceLimits.maxUniformBufferRange>=sizeof(VoxelGIConstants) &&
        GetPhysicalDeviceImageFormatProperties(PhysicalDevice,VK_FORMAT_R32G32B32A32_SFLOAT,VK_IMAGE_TYPE_2D,VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,0,&image)==VK_SUCCESS && image.maxExtent.width>=2048 && image.maxExtent.height>=2048;
}
void DestroyVoxelGITargets()
{
    for (auto& framebuffer:VoxelGIFramebuffers)
    { if (framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,framebuffer,nullptr); framebuffer=0; }
    for (auto& image:VoxelGIImages) DestroyRawImage(image);
}
void DestroyVoxelGI()
{
    DestroyVoxelGITargets(); DestroyRawImage(VoxelVolume); FrameVoxelGI={}; UploadedVoxelVolume.reset();
    for (auto pipeline:VoxelGIPipelines) if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr);
    for (auto module:VoxelGIModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    if (VoxelGIPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,VoxelGIPool,nullptr);
    if (VoxelGILayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,VoxelGILayout,nullptr);
    if (VoxelGIDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,VoxelGIDescriptorLayout,nullptr);
    if (VoxelGICompositeRenderPass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device,VoxelGICompositeRenderPass,nullptr);
    if (VoxelGISampler && DestroySamplerFunction) DestroySamplerFunction(Device,VoxelGISampler,nullptr);
    VoxelGIPool=VoxelGILayout=VoxelGIDescriptorLayout=VoxelGICompositeRenderPass=VoxelGISampler=0;
    VoxelGIPipelines={}; VoxelGIModules={}; VoxelGIDescriptors={};
}
bool EnsureVoxelGIResources()
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants({},frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("Vulkan voxel GI publication is invalid");
    if (frame.Volume!=UploadedVoxelVolume)
    {
        const auto& volume=*frame.Volume;
        if (volume.Width>DeviceLimits.maxImageDimension2D || volume.Height>DeviceLimits.maxImageDimension2D) return Fail("Vulkan voxel atlas exceeds maxImageDimension2D");
        RawImage image; RawBuffer staging; const auto bytes=volume.Texels.size()*sizeof(Float4);
        if (!CreateRawImage(volume.Width,volume.Height,VK_FORMAT_R32G32B32A32_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,image) ||
            !CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,staging))
        { DestroyRawImage(image); DestroyRawBuffer(staging); return false; }
        std::memcpy(staging.Mapped,volume.Texels.data(),bytes); bool uploaded=BeginImmediate();
        if (uploaded)
        {
            TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy copy{}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent={image.Width,image.Height,1};
            CmdCopyBufferToImage(CommandBuffer,staging.Buffer,image.Image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
            TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); uploaded=EndImmediate();
        }
        DestroyRawBuffer(staging);
        if (!uploaded) { DestroyRawImage(image); return false; }
        DestroyRawImage(VoxelVolume); VoxelVolume=image; UploadedVoxelVolume=frame.Volume;
    }
    if (!CreateFullscreenRenderPass(VK_FORMAT_R16G16B16A16_SFLOAT,VoxelGICompositeRenderPass,true)) return false;
    if (VoxelGIImages[0].Width!=MainColor.Width || VoxelGIImages[0].Height!=MainColor.Height) DestroyVoxelGITargets();
    for (std::size_t index=0;index<VoxelGIImages.size();++index) if (!VoxelGIFramebuffers[index])
    {
        if (!CreateRawImage(MainColor.Width,MainColor.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,VoxelGIImages[index]) ||
            !CreateFramebuffer(HdrRenderPass,&VoxelGIImages[index].View,1,MainColor.Width,MainColor.Height,VoxelGIFramebuffers[index]))
        { DestroyVoxelGITargets(); return false; }
    }
    if (!VoxelGISampler)
    {
        VkSamplerCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        create.magFilter=create.minFilter=VK_FILTER_NEAREST; create.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&VoxelGISampler),"vkCreateSampler(voxel GI)")) return false;
    }
    if (!VoxelGIDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,VoxelGIInputCount+1> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t slot=1;slot<bindings.size();++slot) bindings[slot]={slot,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create.bindingCount=static_cast<std::uint32_t>(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&VoxelGIDescriptorLayout),"vkCreateDescriptorSetLayout(voxel GI)")) return false;
    }
    if (!VoxelGILayout)
    {
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create.setLayoutCount=1; create.pSetLayouts=&VoxelGIDescriptorLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&VoxelGILayout),"vkCreatePipelineLayout(voxel GI)")) return false;
    }
    if (!VoxelGIPool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,VoxelGIDrawCount},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,VoxelGIDrawCount*VoxelGIInputCount}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create.maxSets=VoxelGIDrawCount; create.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&VoxelGIPool),"vkCreateDescriptorPool(voxel GI)")) return false;
    }
    for (auto& descriptor:VoxelGIDescriptors) if (!descriptor)
    {
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=VoxelGIPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&VoxelGIDescriptorLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&descriptor),"vkAllocateDescriptorSets(voxel GI)")) return false;
    }
    return CreateFullscreenPipeline(VulkanVoxelGIShaders,VoxelGIModules,HdrRenderPass,VoxelGIPipelines[0],VoxelGILayout) &&
        CreateFullscreenPipeline(VulkanVoxelGIShaders,VoxelGIModules,VoxelGICompositeRenderPass,VoxelGIPipelines[1],VoxelGILayout,FullscreenBlend::AddRadiance,true);
}
void UpdateVoxelGIDescriptor(VkDescriptorSet descriptor,const std::array<VkImageView,VoxelGIInputCount>& views)
{
    VkDescriptorBufferInfo buffer{UniformBuffer.Buffer,0,sizeof(VoxelGIConstants)};
    std::array<VkDescriptorImageInfo,VoxelGIInputCount> images{};
    std::array<VkWriteDescriptorSet,VoxelGIInputCount+1> writes{};
    for (std::uint32_t slot=0;slot<writes.size();++slot)
    {
        auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=descriptor; write.dstBinding=slot; write.descriptorCount=1;
        if (!slot) { write.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; write.pBufferInfo=&buffer; }
        else
        {
            images[slot-1]={VoxelGISampler,views[slot-1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&images[slot-1];
        }
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
}
bool ApplyVoxelGI(const GlobalIlluminationDesc& desc)
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants(desc.ViewProjection,frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("Vulkan voxel GI camera is invalid");
    std::array<std::uint32_t,VoxelGIDrawCount> offsets{}; auto end=UniformOffset;
    for (auto& offset:offsets)
    {
        end=AlignUp(end,UniformAlignment);
        if (end>UniformBufferSize || sizeof(constants)>UniformBufferSize-end) return Fail("Vulkan voxel GI uniform arena exhausted");
        offset=static_cast<std::uint32_t>(end); end+=sizeof(constants);
    }
    UniformOffset=end; TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (auto& image:GBufferImages) TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (const auto pass:VoxelGIPasses)
    {
        const auto index=static_cast<std::size_t>(pass); const bool composite=pass==Techniques::VoxelGIPass::Composite;
        auto& previous=index ? VoxelGIImages[(index-1)%2]:DummyColor;
        auto& destination=composite ? HdrImages[0]:VoxelGIImages[index%2];
        TransitionImage(CommandBuffer,previous,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); TransitionImage(CommandBuffer,destination,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const std::array<VkImageView,VoxelGIInputCount> inputs=composite ?
            std::array<VkImageView,VoxelGIInputCount>{DummyColor.View,DummyColor.View,DummyColor.View,DummyColor.View,DummyColor.View,previous.View,DummyColor.View} :
            std::array<VkImageView,VoxelGIInputCount>{MainDepth.View,GBufferImages[1].View,GBufferImages[0].View,GBufferImages[3].View,VoxelVolume.View,previous.View,HdrImages[0].View};
        UpdateVoxelGIDescriptor(VoxelGIDescriptors[index],inputs); constants.Options.X=static_cast<std::uint32_t>(pass);
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offsets[index],&constants,sizeof(constants));
        VkClearValue clear{}; VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass=composite ? VoxelGICompositeRenderPass:HdrRenderPass;
        begin.framebuffer=composite ? HdrFramebuffers[0]:VoxelGIFramebuffers[index%2];
        begin.renderArea={{0,0},{destination.Width,destination.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
        CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,float(destination.Width),float(destination.Height),0,1}; CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VoxelGIPipelines[composite ? 1:0]);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VoxelGILayout,0,1,&VoxelGIDescriptors[index],1,&offsets[index]);
        CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
        PendingFrame.VoxelGIPasses.push_back(pass); ++PendingFrame.Statistics.GlobalIlluminationPassCount;
    }
    GI.Resolved=true; return true;
}
