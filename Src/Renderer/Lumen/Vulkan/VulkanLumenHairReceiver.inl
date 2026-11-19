RawImage LumenHairIndirect,LumenHairSamples;
VkFramebuffer LumenHairFramebuffer=0;
VkImageView LumenHairDepthView=0;
VkDescriptorSetLayout LumenHairSetLayout=0;
VkPipelineLayout LumenHairLayout=0;
VkDescriptorPool LumenHairPool=0;
std::array<VkDescriptorSet,2> LumenHairSets{};
VkShaderModule LumenHairFragment=0;
std::array<VkShaderModule,2> LumenHairCompositeModules{};
VkPipeline LumenHairComposite=0;
struct LumenHairRaster { VkRenderPass Pass=0; VkPipeline Pipeline=0; };
std::unordered_map<unsigned,LumenHairRaster> LumenHairPipelines;
void DestroyLumenHairReceiver()
{
    if (LumenHairFramebuffer) DestroyFramebufferFunction(Device,LumenHairFramebuffer,nullptr);
    for (auto entry:LumenHairPipelines) { if (entry.second.Pipeline) DestroyPipelineFunction(Device,entry.second.Pipeline,nullptr); if (entry.second.Pass) DestroyRenderPassFunction(Device,entry.second.Pass,nullptr); }
    if (LumenHairComposite) DestroyPipelineFunction(Device,LumenHairComposite,nullptr);
    if (LumenHairFragment) DestroyShaderModuleFunction(Device,LumenHairFragment,nullptr);
    for (auto module:LumenHairCompositeModules) if (module) DestroyShaderModuleFunction(Device,module,nullptr);
    if (LumenHairPool) DestroyDescriptorPoolFunction(Device,LumenHairPool,nullptr);
    if (LumenHairLayout) DestroyPipelineLayoutFunction(Device,LumenHairLayout,nullptr);
    if (LumenHairSetLayout) DestroyDescriptorSetLayoutFunction(Device,LumenHairSetLayout,nullptr);
    DestroyRawImage(LumenHairIndirect); DestroyRawImage(LumenHairSamples); LumenHairPipelines.clear(); LumenHairSets={}; LumenHairCompositeModules={};
    LumenHairFramebuffer=LumenHairDepthView=LumenHairSetLayout=LumenHairLayout=LumenHairPool=LumenHairFragment=LumenHairComposite=0;
}
bool EnsureLumenHairReceiver()
{
    if (!LumenHairSetLayout)
    {
        const VkDescriptorSetLayoutBinding bindings[]{
            {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=4; desc.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&LumenHairSetLayout),"Create fiber irradiance descriptors")) return false;
    }
    if (!LumenHairLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&LumenHairSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&LumenHairLayout),"Create fiber irradiance layout")) return false;
    }
    if (!LumenHairPool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,4},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,4}};
        VkDescriptorPoolCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; desc.maxSets=2; desc.poolSizeCount=2; desc.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&desc,nullptr,&LumenHairPool),"Create fiber irradiance pool")) return false;
        const VkDescriptorSetLayout layouts[]{LumenHairSetLayout,LumenHairSetLayout}; VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool=LumenHairPool; allocate.descriptorSetCount=2; allocate.pSetLayouts=layouts;
        if (!Check(AllocateDescriptorSets(Device,&allocate,LumenHairSets.data()),"Allocate fiber irradiance sets")) return false;
    }
    if (!LumenHairFragment)
    { if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenHairIndirectShader,LumenHairFragment,LastError)) return false; }
    auto& raster=LumenHairPipelines[FrameSampleCount];
    if (!raster.Pass)
    {
        VkAttachmentDescription attachments[2]{};
        for (unsigned i=0;i<2;++i)
        {
            auto& a=attachments[i]; a.format=i?VK_FORMAT_D32_SFLOAT:VK_FORMAT_R16G16B16A16_SFLOAT; a.samples=FrameSampleCount;
            a.loadOp=i?VK_ATTACHMENT_LOAD_OP_LOAD:VK_ATTACHMENT_LOAD_OP_CLEAR; a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            a.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; a.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout=a.finalLayout=i?VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        const VkAttachmentReference color{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},depth{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS; subpass.colorAttachmentCount=1; subpass.pColorAttachments=&color; subpass.pDepthStencilAttachment=&depth;
        VkSubpassDependency dependency{}; dependency.srcSubpass=VK_SUBPASS_EXTERNAL; dependency.dstSubpass=0;
        dependency.srcStageMask=dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; dependency.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT; dependency.dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo desc{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO}; desc.attachmentCount=2; desc.pAttachments=attachments; desc.subpassCount=1; desc.pSubpasses=&subpass; desc.dependencyCount=1; desc.pDependencies=&dependency;
        if (!Check(CreateRenderPassFunction(Device,&desc,nullptr,&raster.Pass),"Create read-only fiber irradiance pass")) return false;
    }
    if (!raster.Pipeline)
    {
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (unsigned i=0;i<2;++i) { stages[i].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[i].stage=i?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT; stages[i].module=i?LumenHairFragment:HairModules[0]; stages[i].pName="main"; }
        const VkVertexInputBindingDescription binding{0,sizeof(HairVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription attributes[]{
            {0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(HairVertex,Position)},{1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(HairVertex,Tangent)},
            {2,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(HairVertex,Reflectance)},{3,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(HairVertex,Emission)},
            {4,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(HairVertex,Ribbon)}};
        VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO}; vertex.vertexBindingDescriptionCount=1; vertex.pVertexBindingDescriptions=&binding; vertex.vertexAttributeDescriptionCount=5; vertex.pVertexAttributeDescriptions=attributes;
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; viewport.viewportCount=viewport.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo drawing{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; drawing.polygonMode=VK_POLYGON_MODE_FILL; drawing.cullMode=VK_CULL_MODE_NONE; drawing.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; samples.rasterizationSamples=FrameSampleCount;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; depth.depthTestEnable=1; depth.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL; depth.maxDepthBounds=1;
        VkPipelineColorBlendAttachmentState color{}; color.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; blend.attachmentCount=1; blend.pAttachments=&color;
        const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dynamic.dynamicStateCount=2; dynamic.pDynamicStates=states;
        VkGraphicsPipelineCreateInfo desc{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; desc.stageCount=2; desc.pStages=stages; desc.pVertexInputState=&vertex; desc.pInputAssemblyState=&assembly; desc.pViewportState=&viewport; desc.pRasterizationState=&drawing; desc.pMultisampleState=&samples; desc.pDepthStencilState=&depth; desc.pColorBlendState=&blend; desc.pDynamicState=&dynamic; desc.layout=LumenHairLayout; desc.renderPass=raster.Pass;
        if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&desc,nullptr,&raster.Pipeline),"Create fiber irradiance pipeline")) return false;
    }
    if (!CreateFullscreenPipeline(VulkanLumenHairCompositeShaders,LumenHairCompositeModules,HdrRenderPass,LumenHairComposite,LumenHairLayout)) return false;
    const auto& depth=FrameSampleCount>1?Multisample.Depth:MainDepth;
    if (LumenHairIndirect.Image && LumenHairIndirect.Width==unsigned(CreateInfo.Width) && LumenHairIndirect.Height==unsigned(CreateInfo.Height) && LumenHairDepthView==depth.View &&
        (FrameSampleCount==1 || LumenHairSamples.Samples==FrameSampleCount)) return true;
    RawImage indirect,samples; VkFramebuffer framebuffer=0;
    const auto cleanup=[&]() { DestroyRawImage(indirect); DestroyRawImage(samples); };
    if (!CreateRawImage(CreateInfo.Width,CreateInfo.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,indirect)) return false;
    if (FrameSampleCount>1 && !CreateRawImage(CreateInfo.Width,CreateInfo.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,samples,1,FrameSampleCount)) { cleanup(); return false; }
    const VkImageView views[]{FrameSampleCount>1?samples.View:indirect.View,depth.View};
    if (!CreateFramebuffer(raster.Pass,views,2,CreateInfo.Width,CreateInfo.Height,framebuffer)) { cleanup(); return false; }
    if (LumenHairFramebuffer) DestroyFramebufferFunction(Device,LumenHairFramebuffer,nullptr); DestroyRawImage(LumenHairIndirect); DestroyRawImage(LumenHairSamples);
    LumenHairIndirect=indirect; LumenHairSamples=samples; LumenHairFramebuffer=framebuffer; LumenHairDepthView=depth.View; return true;
}
bool ResolveLumenHairReceiver()
{
    if (!Hair.LumenReceiver) return true;
    if (!Hair.Resolved || !Hair.HasPrimaryView || !LumenVolumeFrame.Grid.W) return Fail("Fiber irradiance requires the primary surface and current SH volume");
    LumenHairReceiverFrame frame; HairFurConstants geometry;
    if (!BuildLumenHairReceiverFrame(LumenVolumeFrame,Hair.Frame.Settings,Hair.PrimaryView.Lighting,Hair.PrimaryView.ClipPlane,frame,LastError) ||
        !BuildHairFurReplayConstants(Hair.PrimaryView,Hair.Frame,HairFurRasterPhase::Surface,{},geometry,LastError) || !EnsureLumenHairReceiver()) return false;
    const auto vertexOffset=AlignUp(UniformOffset,UniformAlignment),fragmentOffset=AlignUp(vertexOffset+sizeof(geometry),UniformAlignment);
    if (fragmentOffset>UniformBufferSize || sizeof(frame)>UniformBufferSize-fragmentOffset) return Fail("Fiber irradiance exhausted frame uniforms");
    std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+vertexOffset,&geometry,sizeof(geometry)); std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+fragmentOffset,&frame,sizeof(frame)); UniformOffset=fragmentOffset+sizeof(frame);
    TransitionImage(CommandBuffer,LumenVolumeAtlas,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const VkDescriptorBufferInfo buffers[]{{UniformBuffer.Buffer,0,sizeof(geometry)},{UniformBuffer.Buffer,0,sizeof(frame)}};
    for (unsigned set=0;set<2;++set)
    {
        const VkDescriptorImageInfo images[]{{LumenGatherSampler,set?LumenHairIndirect.View:LumenVolumeAtlas.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},{LumenGatherSampler,HdrImages[0].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        VkWriteDescriptorSet writes[4]{};
        for (unsigned i=0;i<4;++i) { auto& w=writes[i]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=LumenHairSets[set]; w.dstBinding=i; w.descriptorCount=1; w.descriptorType=i<2?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; if (i<2) w.pBufferInfo=&buffers[i]; else w.pImageInfo=&images[i-2]; }
        UpdateDescriptorSets(Device,4,writes,0,nullptr);
    }
    auto& target=FrameSampleCount>1?LumenHairSamples:LumenHairIndirect; auto& depth=FrameSampleCount>1?Multisample.Depth:MainDepth;
    TransitionImage(CommandBuffer,target,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); TransitionImage(CommandBuffer,depth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    const VkClearValue clear{}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; begin.renderPass=LumenHairPipelines.at(FrameSampleCount).Pass; begin.framebuffer=LumenHairFramebuffer; begin.renderArea={{0,0},{unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)}}; begin.clearValueCount=1; begin.pClearValues=&clear;
    const VkViewport viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const std::uint32_t offsets[]{std::uint32_t(vertexOffset),std::uint32_t(fragmentOffset)};
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenHairPipelines.at(FrameSampleCount).Pipeline);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenHairLayout,0,1,&LumenHairSets[0],2,offsets);
    const VkDeviceSize zero=0; CmdBindVertexBuffers(CommandBuffer,0,1,&HairVertices.Buffer,&zero); CmdBindIndexBuffer(CommandBuffer,HairIndices.Buffer,0,VK_INDEX_TYPE_UINT32); CmdDrawIndexed(CommandBuffer,unsigned(Hair.Frame.Geometry->Indices.size()),1,0,0,0); CmdEndRenderPass(CommandBuffer);
    if (FrameSampleCount>1)
    {
        TransitionImage(CommandBuffer,target,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,LumenHairIndirect,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageResolve resolve{}; resolve.srcSubresource=resolve.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; resolve.extent={unsigned(CreateInfo.Width),unsigned(CreateInfo.Height),1};
        CmdResolveImage(CommandBuffer,target.Image,target.Layout,LumenHairIndirect.Image,LumenHairIndirect.Layout,1,&resolve);
    }
    TransitionImage(CommandBuffer,LumenHairIndirect,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); TransitionImage(CommandBuffer,LumenGatherTarget,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin.renderPass=HdrRenderPass; begin.framebuffer=LumenGatherFramebuffer; CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenHairComposite); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenHairLayout,0,1,&LumenHairSets[1],2,offsets); CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    TransitionImage(CommandBuffer,LumenGatherTarget,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={unsigned(CreateInfo.Width),unsigned(CreateInfo.Height),1}; CmdCopyImage(CommandBuffer,LumenGatherTarget.Image,LumenGatherTarget.Layout,HdrImages[0].Image,HdrImages[0].Layout,1,&copy);
    ++PendingFrame.Statistics.LumenHairReceiverPassCount; ++PendingFrame.Statistics.GlobalIlluminationPassCount; return true;
}
