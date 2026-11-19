// VulkanRHI::Impl members. Each capture draw owns a descriptor set. Image
// transitions use the frame's speculative-layout journal, including private D32.
std::array<RawImage,4> RefractionImages{};
RawBuffer RefractionVertices,RefractionIndices;
VkRenderPass RefractionCapturePass=0;
VkFramebuffer RefractionCaptureFB=0,RefractionCompositeFB=0;
VkDescriptorSetLayout RefractionDescriptorLayout=0;
VkPipelineLayout RefractionLayout=0;
VkDescriptorPool RefractionPool=0;
VkSampler RefractionSampler=0;
std::vector<VkDescriptorSet> RefractionSets;
std::array<VkShaderModule,4> RefractionModules{};
std::array<VkPipeline,5> RefractionPipelines{};
RefractionFrame FrameRefraction;
Techniques::RefractionSettings FrameRefractionSettings;
std::shared_ptr<const RefractionGeometry> UploadedRefraction;
bool RefractionActive=false,RefractionResolved=false;

void DestroyRefractionTargets()
{
    for (auto* framebuffer:{&RefractionCaptureFB,&RefractionCompositeFB})
    { if (*framebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,*framebuffer,nullptr); *framebuffer=0; }
    for (auto& image:RefractionImages) DestroyRawImage(image);
}
void DestroyRefractionState()
{
    DestroyRefractionTargets(); DestroyRawBuffer(RefractionVertices); DestroyRawBuffer(RefractionIndices);
    for (auto& pipeline:RefractionPipelines) { if (pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    for (auto& module:RefractionModules) { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    if (RefractionCapturePass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device,RefractionCapturePass,nullptr);
    if (RefractionSampler && DestroySamplerFunction) DestroySamplerFunction(Device,RefractionSampler,nullptr);
    if (RefractionPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,RefractionPool,nullptr);
    if (RefractionLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,RefractionLayout,nullptr);
    if (RefractionDescriptorLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,RefractionDescriptorLayout,nullptr);
    RefractionCapturePass=RefractionSampler=RefractionPool=RefractionLayout=RefractionDescriptorLayout=0;
    RefractionSets.clear(); UploadedRefraction.reset(); FrameRefraction={}; RefractionActive=RefractionResolved=false;
}

bool CreateRefractionPipeline(std::uint32_t variant)
{
    auto& output=RefractionPipelines[variant]; if (output) return true;
    const bool composite=variant==4;
    for (std::size_t index=0;index<RefractionModules.size();++index)
        if (!RefractionModules[index] && !CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanRefractionShaders[index],RefractionModules[index],LastError)) return false;
    std::array<VkPipelineShaderStageCreateInfo,2> stages{};
    for (std::size_t index=0;index<stages.size();++index)
    {
        stages[index].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[index].stage=index==0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[index].module=RefractionModules[index+(composite ? 2 : 0)]; stages[index].pName="main";
    }
    const VkVertexInputBindingDescription binding{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[]{
        {0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,Position)}, {1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,Normal)},
        {2,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(Vertex,Color)}, {3,0,VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex,TexCoord)}};
    VkPipelineVertexInputStateCreateInfo input{}; input.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (!composite) { input.vertexBindingDescriptionCount=1; input.pVertexBindingDescriptions=&binding; input.vertexAttributeDescriptionCount=std::uint32_t(std::size(attributes)); input.pVertexAttributeDescriptions=attributes; }
    VkPipelineInputAssemblyStateCreateInfo assembly{}; assembly.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{}; viewport.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; viewport.viewportCount=viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{}; raster.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode=VK_POLYGON_MODE_FILL; raster.lineWidth=1; raster.cullMode=!composite && (variant&1) ? CullBackFaces : VK_CULL_MODE_NONE;
    raster.frontFace=(variant&2) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPipelineMultisampleStateCreateInfo samples{}; samples.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO; samples.rasterizationSamples=1;
    VkPipelineDepthStencilStateCreateInfo depth{}; depth.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable=depth.depthWriteEnable=composite ? VK_FALSE_VALUE : VK_TRUE_VALUE; depth.depthCompareOp=VK_COMPARE_OP_LESS; depth.maxDepthBounds=1;
    std::array<VkPipelineColorBlendAttachmentState,2> attachments{};
    for (auto& attachment:attachments) attachment.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{}; blend.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; blend.attachmentCount=composite ? 1 : 2; blend.pAttachments=attachments.data();
    const VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{}; dynamic.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO; dynamic.dynamicStateCount=2; dynamic.pDynamicStates=dynamicStates;
    VkGraphicsPipelineCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; create.stageCount=2; create.pStages=stages.data();
    create.pVertexInputState=&input; create.pInputAssemblyState=&assembly; create.pViewportState=&viewport; create.pRasterizationState=&raster;
    create.pMultisampleState=&samples; create.pDepthStencilState=&depth; create.pColorBlendState=&blend; create.pDynamicState=&dynamic;
    create.layout=RefractionLayout; create.renderPass=composite ? HdrRenderPass : RefractionCapturePass;
    return Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&create,nullptr,&output),"vkCreateGraphicsPipelines(refraction)");
}

bool EnsureRefractionResources()
{
    if (!FrameRefraction.Geometry || FrameRefraction.Draws.empty()) return true;
    if (!RefractionDescriptorLayout)
    {
        std::array<VkDescriptorSetLayoutBinding,10> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        bindings[1]={1,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        for (std::uint32_t index=2;index<bindings.size();++index) bindings[index]={index,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; create.bindingCount=std::uint32_t(bindings.size()); create.pBindings=bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&RefractionDescriptorLayout),"vkCreateDescriptorSetLayout(refraction)")) return false;
    }
    if (!RefractionLayout)
    {
        const std::array<VkDescriptorSetLayout,4> layouts{RefractionDescriptorLayout,CascadeDescriptorLayout,InstanceDescriptorLayout,FogDescriptorLayout};
        VkPipelineLayoutCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; create.setLayoutCount=unsigned(layouts.size()); create.pSetLayouts=layouts.data();
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&RefractionLayout),"vkCreatePipelineLayout(refraction)")) return false;
    }
    if (!RefractionSampler)
    {
        VkSamplerCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        create.magFilter=create.minFilter=VK_FILTER_LINEAR; create.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&RefractionSampler),"vkCreateSampler(refraction)")) return false;
    }
    const auto required=FrameRefraction.Draws.size()+1;
    if (required>UINT32_MAX/8) return Fail("Vulkan refraction descriptor capacity exceeded");
    if (RefractionSets.size()<required)
    {
        if (RefractionPool) DestroyDescriptorPoolFunction(Device,RefractionPool,nullptr);
        RefractionPool=0; RefractionSets.clear(); const auto count=std::uint32_t(required);
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,count},{VK_DESCRIPTOR_TYPE_SAMPLER,count},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,count*8}};
        VkDescriptorPoolCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; create.maxSets=count; create.poolSizeCount=3; create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&RefractionPool),"vkCreateDescriptorPool(refraction)")) return false;
        std::vector<VkDescriptorSetLayout> layouts(count,RefractionDescriptorLayout); RefractionSets.resize(count);
        VkDescriptorSetAllocateInfo allocate{}; allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool=RefractionPool; allocate.descriptorSetCount=count; allocate.pSetLayouts=layouts.data();
        if (!Check(AllocateDescriptorSets(Device,&allocate,RefractionSets.data()),"vkAllocateDescriptorSets(refraction)")) { RefractionSets.clear(); return false; }
    }
    if (!RefractionCapturePass)
    {
        std::array<VkAttachmentDescription,3> attachments{};
        for (unsigned index=0;index<attachments.size();++index)
        {
            auto& attachment=attachments[index]; attachment.format=index==2 ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT; attachment.samples=1;
            attachment.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout=attachment.finalLayout=index==2 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        const VkAttachmentReference colors[]{{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},{1,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
        const VkAttachmentReference depth{2,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS; subpass.colorAttachmentCount=2; subpass.pColorAttachments=colors; subpass.pDepthStencilAttachment=&depth;
        VkSubpassDependency dependency{}; dependency.srcSubpass=VK_SUBPASS_EXTERNAL; dependency.dstSubpass=0;
        dependency.srcStageMask=dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask=dependency.srcAccessMask|VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT; dependency.dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO; create.attachmentCount=3; create.pAttachments=attachments.data();
        create.subpassCount=1; create.pSubpasses=&subpass; create.dependencyCount=1; create.pDependencies=&dependency;
        if (!Check(CreateRenderPassFunction(Device,&create,nullptr,&RefractionCapturePass),"vkCreateRenderPass(refraction)")) return false;
    }
    if (RefractionImages[0].Width!=MainColor.Width || RefractionImages[0].Height!=MainColor.Height) DestroyRefractionTargets();
    if (!RefractionCaptureFB)
    {
        for (unsigned index=0;index<RefractionImages.size();++index)
        {
            const bool depth=index==3;
            const auto usage=(depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)|VK_IMAGE_USAGE_SAMPLED_BIT|
                (index==2 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
            if (!CreateRawImage(MainColor.Width,MainColor.Height,depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT,
                depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,usage,RefractionImages[index])) { DestroyRefractionTargets(); return false; }
        }
        const VkImageView views[]{RefractionImages[0].View,RefractionImages[1].View,RefractionImages[3].View};
        if (!CreateFramebuffer(RefractionCapturePass,views,3,MainColor.Width,MainColor.Height,RefractionCaptureFB) ||
            !CreateFramebuffer(HdrRenderPass,&RefractionImages[2].View,1,MainColor.Width,MainColor.Height,RefractionCompositeFB)) { DestroyRefractionTargets(); return false; }
    }
    for (std::uint32_t index=0;index<RefractionPipelines.size();++index) if (!CreateRefractionPipeline(index)) return false;
    if (UploadedRefraction!=FrameRefraction.Geometry)
    {
        const auto& geometry=*FrameRefraction.Geometry;
        if (geometry.Vertices.empty() || geometry.Indices.empty()) return Fail("Vulkan refraction geometry is empty");
        UploadedRefraction.reset(); DestroyRawBuffer(RefractionVertices); DestroyRawBuffer(RefractionIndices);
        const auto vertexBytes=geometry.Vertices.size()*sizeof(Vertex),indexBytes=geometry.Indices.size()*sizeof(std::uint32_t);
        constexpr VkFlags Memory=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (!CreateRawBuffer(vertexBytes,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,Memory,RefractionVertices) ||
            !CreateRawBuffer(indexBytes,VK_BUFFER_USAGE_INDEX_BUFFER_BIT,Memory,RefractionIndices)) return false;
        std::memcpy(RefractionVertices.Mapped,geometry.Vertices.data(),vertexBytes); std::memcpy(RefractionIndices.Mapped,geometry.Indices.data(),indexBytes);
        UploadedRefraction=FrameRefraction.Geometry;
    }
    return true;
}

bool ResolveRefraction(const RefractionDesc& desc)
{
    if (!FrameRefraction.Geometry || FrameRefraction.Draws.empty()) { RefractionResolved=true; return true; }
    if (UploadedRefraction!=FrameRefraction.Geometry) return Fail("Vulkan refraction snapshot changed during a frame");
    RefractionConstants base;
    if (!BuildRefractionConstants(desc.ViewProjection,desc.Environment,FrameRefractionSettings,MainColor.Width,MainColor.Height,base)) return Fail("Vulkan refraction projection is invalid");
    auto writeSet=[&](std::size_t index,const RefractionConstants& value,const std::array<RawImage*,8>& inputs,std::uint32_t& dynamicOffset) {
        const auto offset=AlignUp(UniformOffset,UniformAlignment);
        if (offset>UniformBufferSize || sizeof(value)>UniformBufferSize-offset) return Fail("Vulkan refraction uniform arena is exhausted");
        std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&value,sizeof(value)); UniformOffset=offset+sizeof(value); dynamicOffset=std::uint32_t(offset);
        const VkDescriptorBufferInfo buffer{UniformBuffer.Buffer,0,sizeof(value)};
        std::array<VkDescriptorImageInfo,9> images{}; images[0]={RefractionSampler,0,VK_IMAGE_LAYOUT_UNDEFINED};
        for (std::size_t slot=0;slot<inputs.size();++slot)
        { TransitionImage(CommandBuffer,*inputs[slot],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); images[slot+1]={0,inputs[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; }
        std::array<VkWriteDescriptorSet,10> writes{};
        for (std::uint32_t slot=0;slot<writes.size();++slot)
        {
            auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=RefractionSets[index]; write.dstBinding=slot; write.descriptorCount=1;
            write.descriptorType=slot==0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : slot==1 ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            if (slot==0) write.pBufferInfo=&buffer; else write.pImageInfo=&images[slot-1];
        }
        UpdateDescriptorSets(Device,std::uint32_t(writes.size()),writes.data(),0,nullptr); return true;
    };
    auto texture=[&](TextureHandle handle)->RawImage* {
        if (!handle) return &DummyColor; const auto found=Textures.find(handle.Value);
        return found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? &found->second.Image : nullptr;
    };
    std::vector<std::uint32_t> offsets(FrameRefraction.Draws.size()); std::vector<RefractionGeometryRange> ranges(FrameRefraction.Draws.size());
    for (std::size_t index=0;index<FrameRefraction.Draws.size();++index)
    {
        const auto& draw=FrameRefraction.Draws[index]; RefractionConstants value;
        if (!BuildRefractionDrawConstants(FrameRefraction,draw,base,value,ranges[index])) return Fail("Vulkan refraction draw is invalid");
        auto* diffuse=texture(draw.DiffuseTexture); auto* normal=texture(draw.NormalTexture); auto* opacity=texture(draw.OpacityTexture);
        if (!diffuse || !normal || !opacity) return Fail("Vulkan refraction material texture is invalid");
        if (!writeSet(index,value,{&MainDepth,diffuse,normal,opacity,&DummyColor,&MainDepth,&DummyColor,&DummyColor},offsets[index])) return false;
    }
    for (auto index:{0,1}) TransitionImage(CommandBuffer,RefractionImages[index],VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    TransitionImage(CommandBuffer,RefractionImages[3],VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    std::array<VkClearValue,3> clear{}; clear[2].depthStencil={1,0};
    VkRenderPassBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO; begin.renderPass=RefractionCapturePass; begin.framebuffer=RefractionCaptureFB;
    begin.renderArea={{0,0},{MainColor.Width,MainColor.Height}}; begin.clearValueCount=3; begin.pClearValues=clear.data();
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0,0,float(MainColor.Width),float(MainColor.Height),0,1}; CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
    const VkDeviceSize offset=0; CmdBindVertexBuffers(CommandBuffer,0,1,&RefractionVertices.Buffer,&offset); CmdBindIndexBuffer(CommandBuffer,RefractionIndices.Buffer,0,VK_INDEX_TYPE_UINT32);
    for (std::size_t index=0;index<FrameRefraction.Draws.size();++index)
    {
        const auto& draw=FrameRefraction.Draws[index]; const auto& range=ranges[index]; if (!range.IndexCount) continue;
        const auto variant=(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RefractionPipelines[variant]);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RefractionLayout,0,1,&RefractionSets[index],1,&offsets[index]);
        CmdDrawIndexed(CommandBuffer,range.IndexCount,1,range.FirstIndex,0,0);
    }
    CmdEndRenderPass(CommandBuffer);
    auto& scene=HdrImages[0]; auto& composite=RefractionImages[2]; std::uint32_t compositionOffset=0; const auto index=FrameRefraction.Draws.size();
    if (!writeSet(index,base,{&MainDepth,&DummyColor,&DummyColor,&DummyColor,&scene,&RefractionImages[3],&RefractionImages[0],&RefractionImages[1]},compositionOffset)) return false;
    auto volumeSet=FogDisabledSet;
    if (FrameGI.Lumen.Translucency.Enabled && LumenVolumeFrame.Grid.W!=0)
    {
        volumeSet=LumenVolumeFirstReceiverSet;
        if (!LumenVolumeReceiverWritten[0])
        {
            if (!WriteFogSet(volumeSet,{},{DummyColor.View,DummyColor.View,DummyColor.View})) return false;
            LumenVolumeReceiverWritten[0]=true;
        }
    }
    TransitionImage(CommandBuffer,composite,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin.renderPass=HdrRenderPass; begin.framebuffer=RefractionCompositeFB; begin.clearValueCount=1;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RefractionPipelines[4]);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RefractionLayout,0,1,&RefractionSets[index],1,&compositionOffset);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,RefractionLayout,FogDescriptorSetIndex,1,&FogDescriptorSets[volumeSet],1,&FogUniformOffsets[volumeSet]);
    CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    TransitionImage(CommandBuffer,composite,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,scene,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={scene.Width,scene.Height,1};
    CmdCopyImage(CommandBuffer,composite.Image,composite.Layout,scene.Image,scene.Layout,1,&copy);
    RefractionResolved=true; RecordRefractionDraws(FrameRefraction,PendingFrame.Statistics); return true;
}
