LumenTemporalHistory LumenHistory;
std::array<RawBuffer,2> LumenTemporalBuffers;
VkDescriptorSetLayout LumenTemporalSetLayout=0;
VkPipelineLayout LumenTemporalLayout=0;
VkDescriptorPool LumenTemporalPool=0;
std::array<VkDescriptorSet,2> LumenTemporalSets{};
VkPipeline LumenTemporalCompute=0,LumenTemporalComposite=0;
std::array<VkShaderModule,2> LumenTemporalModules{};
void DestroyLumenTemporal()
{
    for (auto& buffer:LumenTemporalBuffers) DestroyRawBuffer(buffer);
    if (LumenTemporalCompute) DestroyPipelineFunction(Device,LumenTemporalCompute,nullptr); LumenTemporalCompute=0;
    if (LumenTemporalComposite) DestroyPipelineFunction(Device,LumenTemporalComposite,nullptr); LumenTemporalComposite=0;
    for (auto& module:LumenTemporalModules) { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    if (LumenTemporalPool) DestroyDescriptorPoolFunction(Device,LumenTemporalPool,nullptr); LumenTemporalPool=0; LumenTemporalSets={};
    if (LumenTemporalLayout) DestroyPipelineLayoutFunction(Device,LumenTemporalLayout,nullptr); LumenTemporalLayout=0;
    if (LumenTemporalSetLayout) DestroyDescriptorSetLayoutFunction(Device,LumenTemporalSetLayout,nullptr); LumenTemporalSetLayout=0;
    LumenHistory.Invalidate();
}
bool EnsureLumenTemporal(std::size_t bytes)
{
    if (LumenTemporalBuffers[0].Size!=bytes)
    {
        std::array<RawBuffer,2> buffers;
        for (auto& buffer:buffers)
            if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,buffer))
            { for (auto& candidate:buffers) DestroyRawBuffer(candidate); return false; }
        for (auto& buffer:LumenTemporalBuffers) DestroyRawBuffer(buffer); LumenTemporalBuffers=buffers; LumenHistory.Invalidate();
    }
    if (!LumenTemporalSetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[7]{};
        for (unsigned slot=0;slot<std::size(bindings);++slot)
        {
            const auto stages=VK_SHADER_STAGE_COMPUTE_BIT|((slot==0 || slot==1 || slot==5)?VK_SHADER_STAGE_FRAGMENT_BIT:0);
            bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:slot<5?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,stages,nullptr};
        }
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=unsigned(std::size(bindings)); create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenTemporalSetLayout),"Create Lumen temporal descriptors")) return false;
    }
    if (!LumenTemporalLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenTemporalSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenTemporalLayout),"Create Lumen temporal pipeline layout")) return false;
    }
    if (!LumenTemporalPool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,2},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,8},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}};
        VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; create.maxSets=2; create.poolSizeCount=unsigned(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&LumenTemporalPool),"Create Lumen temporal descriptor pool")) return false;
    }
    for (auto& set:LumenTemporalSets) if (!set)
    {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=LumenTemporalPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenTemporalSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&set),"Allocate Lumen temporal descriptors")) return false;
    }
    if (!LumenTemporalCompute)
    {
        constexpr VulkanShaderProgram program{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_TEMPORAL_STAGE=1"}; VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,program,module,LastError)) return false;
        VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenTemporalLayout; create.basePipelineIndex=-1;
        create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenTemporalCompute); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create Lumen temporal compute pipeline")) return false;
    }
    constexpr VulkanShaderProgram shaders[]{{"Lumen.glsl",VulkanShaderStage::Vertex,"EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE=1;EVANS_VERTEX_STAGE=1"},
        {"Lumen.glsl",VulkanShaderStage::Fragment,"EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE=1"}};
    return CreateFullscreenPipeline(shaders,LumenTemporalModules,HdrRenderPass,LumenTemporalComposite,LumenTemporalLayout);
}
bool ResolveLumenTemporal(const GlobalIlluminationDesc& camera)
{
    const LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    LumenTemporalFrame frame;
    if (!BuildLumenTemporalFrame(camera,FrameGI,CreateInfo.Width,CreateInfo.Height,LumenHistory.CanReuse(key),frame,LastError) ||
        !EnsureLumenTemporal(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    const auto offset=AlignUp(UniformOffset,UniformAlignment);
    if (offset>UniformBufferSize || sizeof(frame)>UniformBufferSize-offset) return Fail("Lumen temporal exhausted the frame uniform arena");
    std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&frame,sizeof(frame)); UniformOffset=offset+sizeof(frame);
    const auto read=LumenHistory.ReadIndex(),write=LumenHistory.WriteIndex();
    LumenDistanceBarrier(LumenTemporalBuffers[read],VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(LumenTemporalBuffers[write],VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    RawImage* inputs[]{&LumenGatherTarget,&MainDepth,&GBufferImages[1],&MotionImages[MotionVelocity]}; VkDescriptorImageInfo images[4]{};
    for (unsigned slot=0;slot<std::size(inputs);++slot)
    { TransitionImage(CommandBuffer,*inputs[slot],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); images[slot]={LumenGatherSampler,inputs[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; }
    TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const VkDescriptorImageInfo scene{LumenGatherSampler,HdrImages[0].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo uniform{UniformBuffer.Buffer,0,sizeof(frame)},histories[]{{LumenTemporalBuffers[read].Buffer,0,LumenTemporalBuffers[read].Size},
        {LumenTemporalBuffers[write].Buffer,0,LumenTemporalBuffers[write].Size}};
    // Separate sets preserve the compute's old history binding while the later
    // fragment pass reads the pending slot. Neither set is edited in flight.
    for (unsigned pass=0;pass<LumenTemporalSets.size();++pass)
    {
        VkWriteDescriptorSet writes[7]{};
        for (unsigned slot=0;slot<std::size(writes);++slot)
        { auto& w=writes[slot]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=LumenTemporalSets[pass]; w.dstBinding=slot; w.descriptorCount=1;
          w.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:slot<5?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          if (slot==0) w.pBufferInfo=&uniform; else if (slot<5) w.pImageInfo=pass && slot==1?&scene:&images[slot-1]; else w.pBufferInfo=&histories[slot==5?pass:1-pass]; }
        UpdateDescriptorSets(Device,unsigned(std::size(writes)),writes,0,nullptr);
    }
    const auto dynamicOffset=std::uint32_t(offset); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenTemporalCompute);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenTemporalLayout,0,1,&LumenTemporalSets[0],1,&dynamicOffset);
    const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    DistanceDispatch(CommandBuffer,std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    LumenDistanceBarrier(LumenTemporalBuffers[write],VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    TransitionImage(CommandBuffer,LumenGatherTarget,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const VkClearValue clear{}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; begin.renderPass=HdrRenderPass; begin.framebuffer=LumenGatherFramebuffer;
    begin.renderArea={{0,0},{LumenGatherTarget.Width,LumenGatherTarget.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); const VkViewport viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1};
    CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenTemporalComposite);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenTemporalLayout,0,1,&LumenTemporalSets[1],1,&dynamicOffset); CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    TransitionImage(CommandBuffer,LumenGatherTarget,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={LumenGatherTarget.Width,LumenGatherTarget.Height,1};
    CmdCopyImage(CommandBuffer,LumenGatherTarget.Image,LumenGatherTarget.Layout,HdrImages[0].Image,HdrImages[0].Layout,1,&copy);
    LumenHistory.Record(key); PendingFrame.Temporal=camera.Sampling; PendingFrame.Statistics.GlobalIlluminationPassCount+=2; return true;
}
