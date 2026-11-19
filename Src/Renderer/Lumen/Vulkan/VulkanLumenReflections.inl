LumenRadianceHandle LumenReflectionRays;
RawBuffer LumenReflectionEstimate;
RawImage LumenReflectionCompositeTarget;
VkFramebuffer LumenReflectionFramebuffer=0;
VkSampler LumenReflectionSampler=0;
VkDescriptorSetLayout LumenReflectionResolveSetLayout=0;
VkPipelineLayout LumenReflectionResolveLayout=0;
VkDescriptorPool LumenReflectionResolvePool=0;
VkDescriptorSet LumenReflectionResolveSet=0;
VkPipeline LumenReflectionResolvePipeline=0;
bool EnsureLumenReflectionTarget()
{
    if (!LumenReflectionSampler)
    {
        VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; create.magFilter=create.minFilter=VK_FILTER_NEAREST;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&LumenReflectionSampler),"Create Lumen reflection sampler")) return false;
    }
    if (LumenReflectionCompositeTarget.Image && LumenReflectionCompositeTarget.Width==unsigned(CreateInfo.Width) && LumenReflectionCompositeTarget.Height==unsigned(CreateInfo.Height)) return true;
    RawImage image; VkFramebuffer framebuffer=0;
    if (!CreateRawImage(CreateInfo.Width,CreateInfo.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,image) ||
        !CreateFramebuffer(HdrRenderPass,&image.View,1,image.Width,image.Height,framebuffer))
    { DestroyRawImage(image); return false; }
    if (LumenReflectionFramebuffer) DestroyFramebufferFunction(Device,LumenReflectionFramebuffer,nullptr);
    DestroyRawImage(LumenReflectionCompositeTarget); LumenReflectionCompositeTarget=image; LumenReflectionFramebuffer=framebuffer; return true;
}
LumenTemporalHistory LumenSpecularHistory;
std::array<RawBuffer,2> LumenSpecularBuffers;
VkDescriptorSetLayout LumenSpecularSetLayout=0;
VkPipelineLayout LumenSpecularLayout=0;
VkDescriptorPool LumenSpecularPool=0;
std::array<VkDescriptorSet,2> LumenSpecularSets{};
VkPipeline LumenSpecularCompute=0,LumenSpecularComposite=0;
std::array<VkShaderModule,2> LumenSpecularModules{};
void DestroyLumenSpecular()
{
    for (auto& buffer:LumenSpecularBuffers) DestroyRawBuffer(buffer);
    if (LumenSpecularCompute) DestroyPipelineFunction(Device,LumenSpecularCompute,nullptr); LumenSpecularCompute=0;
    if (LumenSpecularComposite) DestroyPipelineFunction(Device,LumenSpecularComposite,nullptr); LumenSpecularComposite=0;
    for (auto& module:LumenSpecularModules) { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    if (LumenSpecularPool) DestroyDescriptorPoolFunction(Device,LumenSpecularPool,nullptr); LumenSpecularPool=0; LumenSpecularSets={};
    if (LumenSpecularLayout) DestroyPipelineLayoutFunction(Device,LumenSpecularLayout,nullptr); LumenSpecularLayout=0;
    if (LumenSpecularSetLayout) DestroyDescriptorSetLayoutFunction(Device,LumenSpecularSetLayout,nullptr); LumenSpecularSetLayout=0;
    LumenSpecularHistory.Invalidate();
}
bool EnsureLumenSpecular(std::size_t bytes)
{
    if (!EnsureLumenReflectionTarget()) return false;
    if (LumenSpecularBuffers[0].Size!=bytes)
    {
        std::array<RawBuffer,2> buffers;
        for (auto& buffer:buffers)
            if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,buffer))
            { for (auto& candidate:buffers) DestroyRawBuffer(candidate); return false; }
        for (auto& buffer:LumenSpecularBuffers) DestroyRawBuffer(buffer); LumenSpecularBuffers=buffers; LumenSpecularHistory.Invalidate();
    }
    if (!LumenSpecularSetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[8]{};
        for (unsigned slot=0;slot<std::size(bindings);++slot)
        {
            const auto stages=VK_SHADER_STAGE_COMPUTE_BIT|((slot==0 || slot==1 || slot==5)?VK_SHADER_STAGE_FRAGMENT_BIT:0);
            bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:slot<5?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,stages,nullptr};
        }
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=unsigned(std::size(bindings)); create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenSpecularSetLayout),"Create Lumen specular temporal descriptors")) return false;
    }
    if (!LumenSpecularLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenSpecularSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenSpecularLayout),"Create Lumen specular temporal pipeline layout")) return false;
    }
    if (!LumenSpecularPool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,2},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,8},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6}};
        VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; create.maxSets=2; create.poolSizeCount=unsigned(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&LumenSpecularPool),"Create Lumen specular temporal descriptor pool")) return false;
    }
    for (auto& set:LumenSpecularSets) if (!set)
    {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=LumenSpecularPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenSpecularSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&set),"Allocate Lumen specular temporal descriptors")) return false;
    }
    if (!LumenSpecularCompute)
    {
        constexpr VulkanShaderProgram program{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_TEMPORAL_STAGE=1;EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE=1"}; VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,program,module,LastError)) return false;
        VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenSpecularLayout; create.basePipelineIndex=-1;
        create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenSpecularCompute); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create Lumen specular temporal compute pipeline")) return false;
    }
    constexpr VulkanShaderProgram shaders[]{{"Lumen.glsl",VulkanShaderStage::Vertex,"EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE=1;EVANS_VERTEX_STAGE=1"},
        {"Lumen.glsl",VulkanShaderStage::Fragment,"EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE=1"}};
    return CreateFullscreenPipeline(shaders,LumenSpecularModules,HdrRenderPass,LumenSpecularComposite,LumenSpecularLayout);
}
bool ResolveLumenSpecular(const ReflectionDesc& camera)
{
    LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    key.Reflections=FrameReflections; key.Reflected=true; LumenTemporalFrame frame;
    if (!BuildLumenReflectionTemporalFrame(camera,FrameReflections,CreateInfo.Width,CreateInfo.Height,LumenSpecularHistory.CanReuse(key),frame,LastError) ||
        !EnsureLumenSpecular(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    const auto offset=AlignUp(UniformOffset,UniformAlignment);
    if (offset>UniformBufferSize || sizeof(frame)>UniformBufferSize-offset) return Fail("Lumen specular temporal exhausted the frame uniform arena");
    std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&frame,sizeof(frame)); UniformOffset=offset+sizeof(frame);
    const auto read=LumenSpecularHistory.ReadIndex(),write=LumenSpecularHistory.WriteIndex();
    LumenDistanceBarrier(LumenSpecularBuffers[read],VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(LumenSpecularBuffers[write],VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    RawImage* inputs[]{&HdrImages[0],&MainDepth,&GBufferImages[1],MotionImages[MotionVelocity].View?&MotionImages[MotionVelocity]:&GBufferImages[1]}; VkDescriptorImageInfo images[4]{};
    for (unsigned slot=0;slot<std::size(inputs);++slot)
    { TransitionImage(CommandBuffer,*inputs[slot],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); images[slot]={LumenReflectionSampler,inputs[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; }
    TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const VkDescriptorImageInfo scene{LumenReflectionSampler,HdrImages[0].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo uniform{UniformBuffer.Buffer,0,sizeof(frame)},histories[]{{LumenSpecularBuffers[read].Buffer,0,LumenSpecularBuffers[read].Size},
        {LumenSpecularBuffers[write].Buffer,0,LumenSpecularBuffers[write].Size}};
    const VkDescriptorBufferInfo estimate{LumenReflectionEstimate.Buffer,0,LumenReflectionEstimate.Size};
    // Separate sets preserve the compute's old history binding while the later
    // fragment pass reads the pending slot. Neither set is edited in flight.
    for (unsigned pass=0;pass<LumenSpecularSets.size();++pass)
    {
        VkWriteDescriptorSet writes[8]{};
        for (unsigned slot=0;slot<std::size(writes);++slot)
        { auto& w=writes[slot]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=LumenSpecularSets[pass]; w.dstBinding=slot; w.descriptorCount=1;
          w.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:slot<5?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          if (slot==0) w.pBufferInfo=&uniform; else if (slot<5) w.pImageInfo=pass && slot==1?&scene:&images[slot-1]; else w.pBufferInfo=slot==7?&estimate:&histories[slot==5?pass:1-pass]; }
        UpdateDescriptorSets(Device,unsigned(std::size(writes)),writes,0,nullptr);
    }
    const auto dynamicOffset=std::uint32_t(offset); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenSpecularCompute);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenSpecularLayout,0,1,&LumenSpecularSets[0],1,&dynamicOffset);
    const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    DistanceDispatch(CommandBuffer,std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    LumenDistanceBarrier(LumenSpecularBuffers[write],VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    TransitionImage(CommandBuffer,LumenReflectionCompositeTarget,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const VkClearValue clear{}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; begin.renderPass=HdrRenderPass; begin.framebuffer=LumenReflectionFramebuffer;
    begin.renderArea={{0,0},{LumenReflectionCompositeTarget.Width,LumenReflectionCompositeTarget.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); const VkViewport viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1};
    CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenSpecularComposite);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenSpecularLayout,0,1,&LumenSpecularSets[1],1,&dynamicOffset); CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    TransitionImage(CommandBuffer,LumenReflectionCompositeTarget,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={LumenReflectionCompositeTarget.Width,LumenReflectionCompositeTarget.Height,1};
    CmdCopyImage(CommandBuffer,LumenReflectionCompositeTarget.Image,LumenReflectionCompositeTarget.Layout,HdrImages[0].Image,HdrImages[0].Layout,1,&copy);
    LumenSpecularHistory.Record(key); PendingFrame.Temporal=camera.Sampling; PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::LumenTemporal);
    PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::Composite); PendingFrame.Statistics.ReflectionPassCount+=2; return true;
}

void FinishLumenReflections(bool publish)
{
    LumenSpecularHistory.Finish(publish);
    if (LumenReflectionRays) LumenRadianceTraces.erase(LumenReflectionRays.Value); LumenReflectionRays={};
}
void DestroyLumenReflections()
{
    FinishLumenReflections(false); DestroyLumenSpecular(); DestroyRawBuffer(LumenReflectionEstimate);
    if (LumenReflectionFramebuffer) DestroyFramebufferFunction(Device,LumenReflectionFramebuffer,nullptr); LumenReflectionFramebuffer=0;
    DestroyRawImage(LumenReflectionCompositeTarget);
    if (LumenReflectionSampler) DestroySamplerFunction(Device,LumenReflectionSampler,nullptr); LumenReflectionSampler=0;
    if (LumenReflectionResolvePipeline) DestroyPipelineFunction(Device,LumenReflectionResolvePipeline,nullptr); LumenReflectionResolvePipeline=0;
    if (LumenReflectionResolvePool) DestroyDescriptorPoolFunction(Device,LumenReflectionResolvePool,nullptr); LumenReflectionResolvePool=LumenReflectionResolveSet=0;
    if (LumenReflectionResolveLayout) DestroyPipelineLayoutFunction(Device,LumenReflectionResolveLayout,nullptr); LumenReflectionResolveLayout=0;
    if (LumenReflectionResolveSetLayout) DestroyDescriptorSetLayoutFunction(Device,LumenReflectionResolveSetLayout,nullptr); LumenReflectionResolveSetLayout=0;
}
bool EnsureLumenReflectionResolve(std::size_t bytes)
{
    if (!EnsureLumenReflectionTarget()) return false;
    if (LumenReflectionEstimate.Size!=bytes)
    {
        RawBuffer buffer; if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,buffer)) return false;
        DestroyRawBuffer(LumenReflectionEstimate); LumenReflectionEstimate=buffer;
    }
    if (!LumenReflectionResolveSetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[9]{};
        for (unsigned slot=0;slot<std::size(bindings);++slot) bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            slot<6?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=unsigned(std::size(bindings)); create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenReflectionResolveSetLayout),"Create Lumen reflection resolve descriptors")) return false;
    }
    if (!LumenReflectionResolveLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenReflectionResolveSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenReflectionResolveLayout),"Create Lumen reflection resolve layout")) return false;
    }
    if (!LumenReflectionResolvePool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,5},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
        VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; create.maxSets=1; create.poolSizeCount=unsigned(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&LumenReflectionResolvePool),"Create Lumen reflection resolve pool")) return false;
    }
    if (!LumenReflectionResolveSet)
    {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=LumenReflectionResolvePool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenReflectionResolveSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&LumenReflectionResolveSet),"Allocate Lumen reflection resolve descriptors")) return false;
    }
    if (LumenReflectionResolvePipeline) return true;
    VkShaderModule module=0;
    if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenReflectionResolveShader,module,LastError)) return false;
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenReflectionResolveLayout; create.basePipelineIndex=-1;
    create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenReflectionResolvePipeline); DestroyShaderModuleFunction(Device,module,nullptr);
    return Check(status,"Create Lumen reflection resolve pipeline");
}
bool ApplyLumenReflections(const ReflectionDesc& camera)
{
    if (!FrameLumenLighting || LumenReflectionRays) return Fail("Lumen reflections require one consumer of the current lighting cache");
    if (Techniques::UsesTemporalReflections(FrameReflections) && (!VelocityCaptured || (FrameSampleCount>1 && !SamplesVelocityResolved)))
        return Fail("Temporal Lumen reflections require this frame's resolved velocity before tracing");
    LumenRadianceTraceUpload query;
    if (!BuildLumenReflectionQuery(camera,FrameReflections,CreateInfo.Width,CreateInfo.Height,query,LastError)) return false;
    if (FrameGI.Lumen.GlobalDistance.Enabled)
    { query.GlobalDistance=FrameLumenGlobalDistance; query.Trace.GlobalNearDistance=FrameGI.Lumen.GlobalDistance.NearScale*FrameReflections.MaximumDistance; query.Trace.GlobalMaximumSteps=FrameGI.Lumen.GlobalDistance.MaximumSteps; }
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,query,rays)) return false; LumenReflectionRays=rays.Handle;
    const auto source=LumenRadianceTraces.find(rays.Handle.Value); if (source==LumenRadianceTraces.end()) return Fail("Lumen reflection source disappeared");
    const auto& filter=FrameReflections.Lumen; const auto& sky=FrameGI.Lumen.SkyRadiance; LumenReflectionResolveFrame frame;
    if (!BuildLumenReflectionResolveFrame(rays.Info.Reflection,{filter.FilterRadius,filter.NormalCosine,filter.PlaneTolerance,filter.RelativePlaneTolerance,filter.RoughnessTolerance},
        {sky[0],sky[1],sky[2]},FrameReflectionShading,frame,LastError,MakeLumenEnvironmentLobe(FrameGI.Lumen)) || !EnsureLumenReflectionResolve(std::size_t(frame.Dispatch.Z)*sizeof(Float4))) return false;
    const auto offset=AlignUp(UniformOffset,UniformAlignment);
    if (offset>UniformBufferSize || sizeof(frame)>UniformBufferSize-offset) return Fail("Lumen reflection reconstruction exhausted the uniform arena");
    std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&frame,sizeof(frame)); UniformOffset=offset+sizeof(frame);
    RawImage* textures[]{&MainDepth,&GBufferImages[1],&GBufferImages[0],&GBufferImages[3],&GBufferImages[2]}; VkDescriptorImageInfo images[5]{};
    for (unsigned slot=0;slot<std::size(textures);++slot)
    { TransitionImage(CommandBuffer,*textures[slot],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); images[slot]={LumenReflectionSampler,textures[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; }
    const VkDescriptorBufferInfo uniform{UniformBuffer.Buffer,0,sizeof(frame)},buffers[]{
        {source->second->Origins.Buffer,0,source->second->Origins.Size},{source->second->Buffer.Buffer,0,source->second->Buffer.Size},{LumenReflectionEstimate.Buffer,0,LumenReflectionEstimate.Size}};
    VkWriteDescriptorSet writes[9]{};
    for (unsigned slot=0;slot<std::size(writes);++slot)
    {
        auto& w=writes[slot]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=LumenReflectionResolveSet; w.dstBinding=slot; w.descriptorCount=1;
        w.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:slot<6?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (slot==0) w.pBufferInfo=&uniform; else if (slot<6) w.pImageInfo=&images[slot-1]; else w.pBufferInfo=&buffers[slot-6];
    }
    UpdateDescriptorSets(Device,unsigned(std::size(writes)),writes,0,nullptr);
    LumenDistanceBarrier(LumenReflectionEstimate,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const auto dynamicOffset=std::uint32_t(offset); CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenReflectionResolvePipeline);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenReflectionResolveLayout,0,1,&LumenReflectionResolveSet,1,&dynamicOffset);
    const auto groups=(frame.Dispatch.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    DistanceDispatch(CommandBuffer,std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    LumenDistanceBarrier(LumenReflectionEstimate,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::Trace); PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::LumenReconstruction);
    PendingFrame.Statistics.ReflectionPassCount+=2;
    if (!ResolveLumenSpecular(camera)) return false; ReflectionResolved=true; return true;
}
