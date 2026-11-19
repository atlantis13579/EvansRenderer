// GLSL is compiled in memory. The surface owns its load/store render pass;
// replays borrow compatible passes. Every draw gets a distinct dynamic UBO
// slice so later camera constants cannot alter an earlier recorded view.
HairFurExecution Hair;
HairFurFrame UploadedHair;
RawBuffer HairVertices,HairIndices,HairLights;
VkDescriptorSetLayout HairSetLayout=0;
VkPipelineLayout HairLayout=0;
VkDescriptorPool HairPool=0;
VkDescriptorSet HairSet=0,HairSurfaceSet=0;
std::array<RawImage*,HairShadowMapCount> HairShadowImages{};
void WriteHairDescriptors(VkDescriptorSet set,bool surface)
{
    const VkDescriptorBufferInfo buffers[]{{UniformBuffer.Buffer,0,sizeof(HairFurConstants)},{HairLights.Buffer,0,HairLights.Size}};
    std::array<VkDescriptorImageInfo,HairShadowMapCount> images{};
    std::array<VkWriteDescriptorSet,2+HairShadowMapCount> writes{};
    for (std::uint32_t binding=0;binding<writes.size();++binding)
    {
        auto& write=writes[binding]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=set;
        write.dstBinding=binding; write.descriptorCount=1;
        if (binding<2)
        {
            write.descriptorType=binding?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            write.pBufferInfo=&buffers[binding];
        }
        else
        {
            const auto slot=binding-2; auto* image=surface?HairShadowImages[slot]:&DummyDepth;
            images[slot]={DeferredSampler,image->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&images[slot];
        }
    }
    UpdateDescriptorSets(Device,static_cast<std::uint32_t>(writes.size()),writes.data(),0,nullptr);
    if (surface && Hair.UsesContactReceiver())
    {
        const VkDescriptorImageInfo contact{DeferredSampler,ContactProducer->ResultView(),VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=set; write.dstBinding=7;
        write.descriptorCount=1; write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&contact;
        UpdateDescriptorSets(Device,1,&write,0,nullptr);
    }
}
bool PrepareHairShadowReceiver(const HairFurDesc& desc,HairFurConstants& constants)
{
    SetHairFurAnimation(Hair.Frame,Hair.Frame.SampleTimeSeconds,constants);
    constants.MotionOptions.Y=Hair.LumenReceiver?1u:0u;
    if (!BuildHairContactControl(desc.ViewProjection,*Hair.Frame.Lights,FrameContact.Enabled && !Hair.Frame.Geometry->Indices.empty(),
        Hair.Contact,ContactGenerated && ContactReceiverActive && ContactProducer && ContactProducer->ResultView(),FrameSampleCount,constants.ContactControl,LastError)) return false;
    const bool moments=FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM;
    HairShadowImages.fill(moments?&DummyColor:&DummyDepth);
    std::array<HairShadowMapReadiness,HairShadowMapCount> maps{};
    const bool requested=desc.SamplesShadowMap && !Hair.Frame.Geometry->Indices.empty();
    if (requested && Hair.Frame.Lights->ShadowBegin!=Hair.Frame.Lights->ShadowEnd)
    {
        const auto count=FrameCascades.Frame.CascadeCount,first=count?1u:0u;
        for (unsigned slot=first;slot<first+(count?count:1u) && slot<HairShadowMapCount;++slot)
        {
            const auto handle=slot?FrameCascades.Maps[slot-1]:LastShadowTexture;
            const auto& projection=slot?FrameCascades.Frame.Cascades[slot-1].LightViewProjection:desc.LightViewProjection;
            auto found=Textures.find(handle.Value);
            if (found==Textures.end() || found->second.Description.Format!=TextureFormat::Depth32) continue;
            auto& texture=found->second;
            maps[slot]={std::uint32_t(texture.Description.Width),std::uint32_t(texture.Description.Height),Hair.HasShadowView(handle,projection),false};
            if (!maps[slot].DepthReady) continue;
            HairShadowImages[slot]=&texture.Image;
            if (moments)
            {
                if (!MomentImage(handle,HairShadowImages[slot])) return false;
                maps[slot].MomentsReady=true;
            }
        }
    }
    if (!BuildHairShadowReceiverConstants({desc.LightViewProjection,requested},*Hair.Frame.Lights,FrameCascades,FrameShadowFilter,maps,constants.Shadows,LastError)) return false;
    for (auto* image:HairShadowImages) TransitionImage(CommandBuffer,*image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // Replay descriptors may already be referenced by recorded depth/capture
    // commands. Publish the receiver images only to the unused surface set.
    WriteHairDescriptors(HairSurfaceSet,true); return true;
}
std::array<VkShaderModule,8> HairModules{};
struct HairRasterPipeline { VkRenderPass Pass=0; VkPipeline Pipeline=0; bool OwnsPass=false; };
std::array<std::unordered_map<std::uint32_t,HairRasterPipeline>,HairFurPhaseCount> HairPipelines;
VkFramebuffer HairFramebuffer=0;
void DestroyHairFur()
{
    if (HairFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device,HairFramebuffer,nullptr);
    for (auto& pipelines:HairPipelines) for (auto& entry:pipelines)
    {
        if (entry.second.Pipeline && DestroyPipelineFunction) DestroyPipelineFunction(Device,entry.second.Pipeline,nullptr);
        if (entry.second.OwnsPass && entry.second.Pass && DestroyRenderPassFunction) DestroyRenderPassFunction(Device,entry.second.Pass,nullptr);
    }
    for (auto module:HairModules) if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr);
    if (HairPool && DestroyDescriptorPoolFunction) DestroyDescriptorPoolFunction(Device,HairPool,nullptr);
    if (HairLayout && DestroyPipelineLayoutFunction) DestroyPipelineLayoutFunction(Device,HairLayout,nullptr);
    if (HairSetLayout && DestroyDescriptorSetLayoutFunction) DestroyDescriptorSetLayoutFunction(Device,HairSetLayout,nullptr);
    for (auto* buffer:{&HairVertices,&HairIndices,&HairLights}) DestroyRawBuffer(*buffer);
    Hair={}; UploadedHair={}; for (auto& pipelines:HairPipelines) pipelines.clear(); HairModules={}; HairShadowImages={}; HairFramebuffer=HairPool=HairLayout=HairSetLayout=HairSet=HairSurfaceSet=0;
}
bool EnsureHairFur()
{
    const auto& frame=Hair.Frame;
    if (UploadedHair.Geometry!=frame.Geometry || UploadedHair.Lights!=frame.Lights)
    {
        if (!ValidateHairFurPublication(frame,LastError)) return false;
        RawBuffer vertices,indices,lights;
        const auto cleanup=[&](){DestroyRawBuffer(vertices); DestroyRawBuffer(indices); DestroyRawBuffer(lights);};
        const auto create=[&](const void* data,std::size_t bytes,VkFlags usage,RawBuffer& buffer)
        {
            if (usage==VK_BUFFER_USAGE_STORAGE_BUFFER_BIT && bytes>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan hair lighting exceeds maxStorageBufferRange");
            if (!CreateRawBuffer(bytes,usage,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,buffer)) return false;
            std::memcpy(buffer.Mapped,data,bytes); return true;
        };
        if (UploadedHair.Geometry!=frame.Geometry && !frame.Geometry->Indices.empty())
        {
            if (!create(frame.Geometry->Vertices.data(),frame.Geometry->Vertices.size()*sizeof(HairVertex),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,vertices) ||
                !create(frame.Geometry->Indices.data(),frame.Geometry->Indices.size()*sizeof(std::uint32_t),VK_BUFFER_USAGE_INDEX_BUFFER_BIT,indices)) { cleanup(); return false; }
        }
        if (UploadedHair.Lights!=frame.Lights)
        {
            const HairLightSample empty{}; const auto& samples=frame.Lights->Samples;
            if (!create(samples.empty()?&empty:samples.data(),std::max(samples.size(),std::size_t{1})*sizeof(HairLightSample),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,lights)) { cleanup(); return false; }
        }
        if (UploadedHair.Geometry!=frame.Geometry)
        { DestroyRawBuffer(HairVertices); DestroyRawBuffer(HairIndices); HairVertices=vertices; HairIndices=indices; }
        if (UploadedHair.Lights!=frame.Lights) { DestroyRawBuffer(HairLights); HairLights=lights; }
        UploadedHair=frame;
    }
    if (!HairSetLayout)
    {
        const VkDescriptorSetLayoutBinding bindings[]{
            {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {4,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {5,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {6,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {7,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=static_cast<std::uint32_t>(std::size(bindings)); desc.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&HairSetLayout),"vkCreateDescriptorSetLayout(hair)")) return false;
    }
    if (!HairLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&HairSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&HairLayout),"vkCreatePipelineLayout(hair)")) return false;
    }
    if (!HairPool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,2},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2*(HairShadowMapCount+1)}};
        VkDescriptorPoolCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; desc.maxSets=2; desc.poolSizeCount=static_cast<std::uint32_t>(std::size(sizes)); desc.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&desc,nullptr,&HairPool),"vkCreateDescriptorPool(hair)")) return false;
    }
    if (!HairSet)
    {
        VkDescriptorSetAllocateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; desc.descriptorPool=HairPool; desc.descriptorSetCount=1; desc.pSetLayouts=&HairSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&desc,&HairSet),"vkAllocateDescriptorSets(hair)")) return false;
    }
    if (!HairSurfaceSet)
    {
        VkDescriptorSetAllocateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; desc.descriptorPool=HairPool; desc.descriptorSetCount=1; desc.pSetLayouts=&HairSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&desc,&HairSurfaceSet),"vkAllocateDescriptorSets(hair surface)")) return false;
    }
    HairShadowImages.fill(&DummyDepth); WriteHairDescriptors(HairSet,false); WriteHairDescriptors(HairSurfaceSet,true);
    if (!EnsureHairPipeline(HairFurRasterPhase::Surface,FrameSampleCount,0)) return false;
    if (HairFramebuffer) DestroyFramebufferFunction(Device,HairFramebuffer,nullptr); HairFramebuffer=0;
    const VkImageView views[]{FrameSampleCount>1?Multisample.Images[SampleColor].View:HdrImages[0].View,
        FrameSampleCount>1?Multisample.Images[SampleMaterialFirst+1].View:GBufferImages[1].View,FrameSampleCount>1?Multisample.Depth.View:MainDepth.View};
    return CreateFramebuffer(HairPipelines[HairFurPhaseIndex(HairFurRasterPhase::Surface)].at(FrameSampleCount).Pass,views,3,MainColor.Width,MainColor.Height,HairFramebuffer);
}
bool EnsureHairPipeline(HairFurRasterPhase phase,std::uint32_t sampleCount,VkRenderPass pass)
{
    const bool contact=phase==HairFurRasterPhase::Surface && Hair.UsesContactReceiver();
    auto& pipeline=HairPipelines[HairFurPhaseIndex(phase)][HairFurPipelineKey(phase,sampleCount,contact)];
    if (phase!=HairFurRasterPhase::Surface && !pipeline.Pass) pipeline.Pass=pass;
    if (phase!=HairFurRasterPhase::Surface && !pipeline.Pass) return Fail("Vulkan hair replay has no compatible render pass");
    if (!pipeline.Pass)
    {
        VkAttachmentDescription attachments[3]{};
        for (std::size_t index=0;index<std::size(attachments);++index)
        {
            auto& attachment=attachments[index]; const bool depth=index==2;
            attachment.format=depth?VK_FORMAT_D32_SFLOAT:VK_FORMAT_R16G16B16A16_SFLOAT; attachment.samples=sampleCount;
            attachment.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD; attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout=attachment.finalLayout=depth?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        const VkAttachmentReference colors[]{{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},{1,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
        const VkAttachmentReference depth{2,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS; subpass.colorAttachmentCount=2; subpass.pColorAttachments=colors; subpass.pDepthStencilAttachment=&depth;
        VkSubpassDependency dependency{}; dependency.srcSubpass=VK_SUBPASS_EXTERNAL; dependency.dstSubpass=0;
        dependency.srcStageMask=dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask=dependency.srcAccessMask|VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        dependency.dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo desc{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO}; desc.attachmentCount=3; desc.pAttachments=attachments;
        desc.subpassCount=1; desc.pSubpasses=&subpass; desc.dependencyCount=1; desc.pDependencies=&dependency;
        if (!Check(CreateRenderPassFunction(Device,&desc,nullptr,&pipeline.Pass),"vkCreateRenderPass(hair)")) return false;
        pipeline.OwnsPass=true;
    }
    if (!pipeline.Pipeline)
    {
        const auto fragment=phase==HairFurRasterPhase::Velocity?5u:phase==HairFurRasterPhase::Surface?(contact?(sampleCount>1?7u:6u):1u):phase==HairFurRasterPhase::Capture?4u:phase==HairFurRasterPhase::ShadowMoments?3u:2u;
        // Retain each compiled module, but do not compile unused replay variants
        // before the first visible groom.
        const VulkanShaderProgram programs[]{VulkanHairFurShaders[0],VulkanHairFurShaders[1],VulkanHairFurDepthShaders[1],VulkanHairFurMomentShaders[1],VulkanHairFurCaptureShaders[1],VulkanHairFurVelocityShaders[1],VulkanHairFurContactShaders[0],VulkanHairFurContactShaders[1]};
        for (const auto stage:{0u,fragment})
            if (!HairModules[stage] && !CreateVulkanShaderModule(Device,CreateShaderModuleFunction,programs[stage],HairModules[stage],LastError)) return false;
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (std::size_t stage=0;stage<2;++stage)
        { stages[stage].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[stage].stage=stage?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT; stages[stage].module=HairModules[stage?fragment:0]; stages[stage].pName="main"; }
        const VkVertexInputBindingDescription binding{0,sizeof(HairVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription attributes[]{
            {0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(HairVertex,Position)},{1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(HairVertex,Tangent)},
            {2,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(HairVertex,Reflectance)},{3,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(HairVertex,Emission)},
            {4,0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(HairVertex,Ribbon)}};
        VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO}; vertices.vertexBindingDescriptionCount=1; vertices.pVertexBindingDescriptions=&binding;
        vertices.vertexAttributeDescriptionCount=static_cast<std::uint32_t>(std::size(attributes)); vertices.pVertexAttributeDescriptions=attributes;
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; viewport.viewportCount=viewport.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; raster.polygonMode=VK_POLYGON_MODE_FILL; raster.cullMode=VK_CULL_MODE_NONE; raster.lineWidth=1;
        if (phase==HairFurRasterPhase::ShadowDepth || phase==HairFurRasterPhase::ShadowMoments)
        { raster.depthBiasEnable=1; raster.depthBiasConstantFactor=float(ShadowDepthBiasUnits); raster.depthBiasClamp=ShadowDepthBiasClamp; raster.depthBiasSlopeFactor=ShadowSlopeScaledDepthBias; }
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; samples.rasterizationSamples=sampleCount;
        samples.sampleShadingEnable=contact && sampleCount>1?VK_TRUE_VALUE:VK_FALSE_VALUE; samples.minSampleShading=1;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; depth.depthTestEnable=1; depth.depthWriteEnable=phase==HairFurRasterPhase::Velocity?0:1; depth.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL; depth.maxDepthBounds=1;
        VkPipelineColorBlendAttachmentState colors[2]{}; for (auto& color:colors) color.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; blend.attachmentCount=phase==HairFurRasterPhase::Surface?2:(phase==HairFurRasterPhase::Capture || phase==HairFurRasterPhase::ShadowMoments || phase==HairFurRasterPhase::Velocity)?1:0; blend.pAttachments=colors;
        const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dynamic.dynamicStateCount=2; dynamic.pDynamicStates=states;
        VkGraphicsPipelineCreateInfo desc{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; desc.stageCount=2; desc.pStages=stages; desc.pVertexInputState=&vertices; desc.pInputAssemblyState=&assembly;
        desc.pViewportState=&viewport; desc.pRasterizationState=&raster; desc.pMultisampleState=&samples; desc.pDepthStencilState=&depth; desc.pColorBlendState=&blend; desc.pDynamicState=&dynamic;
        desc.layout=HairLayout; desc.renderPass=pipeline.Pass;
        if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&desc,nullptr,&pipeline.Pipeline),"vkCreateGraphicsPipelines(hair)")) return false;
    }
    return true;
}
bool DrawHairGeometry(const HairFurConstants& constants,HairFurRasterPhase phase,std::uint32_t sampleCount)
{
    if (Hair.Frame.Geometry->Indices.empty()) return true;
    const auto first=AlignUp(UniformOffset,UniformAlignment);
    if (first>UniformBufferSize || sizeof(constants)>UniformBufferSize-first) return Fail("Vulkan hair uniform arena is full");
    const auto offset=static_cast<std::uint32_t>(first); UniformOffset=first+sizeof(constants);
    std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&constants,sizeof(constants));
    const auto& pipeline=HairPipelines[HairFurPhaseIndex(phase)].at(HairFurPipelineKey(phase,sampleCount,Hair.UsesContactReceiver()));
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline.Pipeline);
    const auto set=phase==HairFurRasterPhase::Surface?HairSurfaceSet:HairSet;
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,HairLayout,0,1,&set,1,&offset);
    const VkDeviceSize vertexOffset=0; CmdBindVertexBuffers(CommandBuffer,0,1,&HairVertices.Buffer,&vertexOffset);
    CmdBindIndexBuffer(CommandBuffer,HairIndices.Buffer,0,VK_INDEX_TYPE_UINT32);
    CmdDrawIndexed(CommandBuffer,static_cast<std::uint32_t>(Hair.Frame.Geometry->Indices.size()),1,0,0,0); return true;
}
bool RenderHairFur(const HairFurConstants& constants)
{
    if (!EnsureHairPipeline(HairFurRasterPhase::Surface,FrameSampleCount,0)) return false;
    auto& color=FrameSampleCount>1?Multisample.Images[SampleColor]:HdrImages[0];
    auto& normal=FrameSampleCount>1?Multisample.Images[SampleMaterialFirst+1]:GBufferImages[1];
    auto& depth=FrameSampleCount>1?Multisample.Depth:MainDepth;
    TransitionImage(CommandBuffer,color,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    TransitionImage(CommandBuffer,normal,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    TransitionImage(CommandBuffer,depth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    const auto& pipeline=HairPipelines[HairFurPhaseIndex(HairFurRasterPhase::Surface)].at(HairFurPipelineKey(HairFurRasterPhase::Surface,FrameSampleCount,Hair.UsesContactReceiver()));
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; begin.renderPass=pipeline.Pass; begin.framebuffer=HairFramebuffer; begin.renderArea={{0,0},{MainColor.Width,MainColor.Height}};
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0,0,float(MainColor.Width),float(MainColor.Height),0,1}; CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
    const bool result=DrawHairGeometry(constants,HairFurRasterPhase::Surface,FrameSampleCount); CmdEndRenderPass(CommandBuffer); return result;
}
bool ReplayHairFur(const HairFurConstants& constants,HairFurRasterPhase phase)
{
    const auto samples=ActiveDepth->Samples;
    const VkRenderPass pass=phase==HairFurRasterPhase::Velocity?(FrameSampleCount>1?Multisample.VelocityPass:MotionGeometryRenderPass):phase==HairFurRasterPhase::CameraDepth?ContactDepthPass:phase==HairFurRasterPhase::ShadowDepth?ShadowRenderPasses[0]:
        phase==HairFurRasterPhase::ShadowMoments?VSMCasterPasses[0]:HdrColorRenderPasses[0];
    return EnsureHairPipeline(phase,samples,pass) && DrawHairGeometry(constants,phase,samples);
}
