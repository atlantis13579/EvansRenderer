LumenTranslucencyFrame LumenVolumeFrame;
std::array<bool,2> LumenVolumeReceiverWritten{};
LumenRadianceHandle LumenVolumeRays;
LumenProbeHandle LumenVolumeProbes;
void ReleaseLumenTranslucencyResults()
{
    if (LumenVolumeProbes) LumenProbes.erase(LumenVolumeProbes.Value);
    if (LumenVolumeRays) LumenRadianceTraces.erase(LumenVolumeRays.Value);
    LumenVolumeProbes={}; LumenVolumeRays={}; LumenVolumeFrame.Grid.W=0; LumenVolumeReceiverWritten={};
}

RawImage LumenVolumeAtlas;
VkFramebuffer LumenVolumeFramebuffer=0;
VkDescriptorSetLayout LumenVolumeSetLayout=0;
VkDescriptorPool LumenVolumePool=0;
VkDescriptorSet LumenVolumeSet=0;
VkPipelineLayout LumenVolumeLayout=0;
VkPipeline LumenVolumePipeline=0;
std::array<VkShaderModule,2> LumenVolumeModules{};
RawBuffer LumenVolumeSupport,LumenVolumeSupportUniform;
VkPipeline LumenVolumeSupportPipeline=0;
VkDescriptorPool LumenVolumeSupportPool=0;
VkDescriptorSet LumenVolumeSupportSet=0;
void DestroyLumenTranslucency()
{
    ReleaseLumenTranslucencyResults();
    if (LumenVolumeFramebuffer) DestroyFramebufferFunction(Device,LumenVolumeFramebuffer,nullptr); LumenVolumeFramebuffer=0;
    DestroyRawImage(LumenVolumeAtlas);
    if (LumenVolumePipeline) DestroyPipelineFunction(Device,LumenVolumePipeline,nullptr); LumenVolumePipeline=0;
    for (auto& module:LumenVolumeModules) { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    if (LumenVolumePool) DestroyDescriptorPoolFunction(Device,LumenVolumePool,nullptr); LumenVolumePool=LumenVolumeSet=0;
    if (LumenVolumeLayout) DestroyPipelineLayoutFunction(Device,LumenVolumeLayout,nullptr); LumenVolumeLayout=0;
    if (LumenVolumeSetLayout) DestroyDescriptorSetLayoutFunction(Device,LumenVolumeSetLayout,nullptr); LumenVolumeSetLayout=0;
    if (LumenVolumeSupportPipeline) DestroyPipelineFunction(Device,LumenVolumeSupportPipeline,nullptr); LumenVolumeSupportPipeline=0;
    if (LumenVolumeSupportPool) DestroyDescriptorPoolFunction(Device,LumenVolumeSupportPool,nullptr); LumenVolumeSupportPool=LumenVolumeSupportSet=0;
    DestroyRawBuffer(LumenVolumeSupport); DestroyRawBuffer(LumenVolumeSupportUniform);
}
bool EnsureLumenTranslucency()
{
    if (!LumenVolumeSetLayout)
    {
        const VkDescriptorSetLayoutBinding bindings[]{
            {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=unsigned(std::size(bindings)); desc.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&LumenVolumeSetLayout),"Create Lumen volume descriptors")) return false;
    }
    if (!LumenVolumeLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&LumenVolumeSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&LumenVolumeLayout),"Create Lumen volume layout")) return false;
    }
    if (!LumenVolumePool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};
        VkDescriptorPoolCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; desc.maxSets=1; desc.poolSizeCount=unsigned(std::size(sizes)); desc.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&desc,nullptr,&LumenVolumePool),"Create Lumen volume pool")) return false;
    }
    if (!LumenVolumeSet)
    {
        VkDescriptorSetAllocateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; desc.descriptorPool=LumenVolumePool; desc.descriptorSetCount=1; desc.pSetLayouts=&LumenVolumeSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&desc,&LumenVolumeSet),"Allocate Lumen volume set")) return false;
    }
    constexpr VulkanShaderProgram shaders[]{
        {"LumenTranslucency.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
        {"LumenTranslucency.glsl",VulkanShaderStage::Fragment}};
    if (!CreateFullscreenPipeline(shaders,LumenVolumeModules,HdrRenderPass,LumenVolumePipeline,LumenVolumeLayout)) return false;
    const unsigned width=unsigned(LumenVolumeFrame.Atlas.X),height=unsigned(LumenVolumeFrame.Atlas.Y);
    if (LumenVolumeAtlas.Image && LumenVolumeAtlas.Width==width && LumenVolumeAtlas.Height==height) return true;
    RawImage image; VkFramebuffer framebuffer=0;
    if (!CreateRawImage(width,height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,image) ||
        !CreateFramebuffer(HdrRenderPass,&image.View,1,width,height,framebuffer)) { DestroyRawImage(image); return false; }
    if (LumenVolumeFramebuffer) DestroyFramebufferFunction(Device,LumenVolumeFramebuffer,nullptr);
    DestroyRawImage(LumenVolumeAtlas); LumenVolumeAtlas=image; LumenVolumeFramebuffer=framebuffer; return true;
}
bool PrepareLumenVolumeSupport(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    const auto source=LumenRadianceTraces.find(probes.SourceHandle.Value);
    if (source==LumenRadianceTraces.end() || !source->second->TraceDistance) return Fail("Vulkan Lumen volume lost its immutable distance scene");
    LumenDistanceTraceFrame frame;
    if (!BuildLumenTranslucencySupportFrame(source->second->TraceDistance->Info,upload,frame,LastError) || !EnsureLumenDistanceTrace()) return false;
    if (!LumenVolumeSupportPipeline)
    {
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenVolumeSupportShader,module,LastError)) return false;
        VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenDistanceLayout; create.basePipelineIndex=-1;
        create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenVolumeSupportPipeline); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create Lumen volume visibility pipeline")) return false;
    }
    if (!LumenVolumeSupportPool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
        VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; create.maxSets=1; create.poolSizeCount=unsigned(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&LumenVolumeSupportPool),"Create Lumen volume visibility pool")) return false;
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=LumenVolumeSupportPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenDistanceSet;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&LumenVolumeSupportSet),"Allocate Lumen volume visibility set")) return false;
    }
    DestroyRawBuffer(LumenVolumeSupport); DestroyRawBuffer(LumenVolumeSupportUniform);
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(frame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,LumenVolumeSupportUniform) ||
        !CreateRawBuffer(std::size_t(frame.Counts.X)*sizeof(float),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,LumenVolumeSupport)) return false;
    std::memcpy(LumenVolumeSupportUniform.Mapped,&frame,sizeof(frame));
    LumenDistanceBarrier(LumenVolumeSupportUniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const auto& scene=source->second->TraceDistance->Scene; const auto& origins=source->second->Origins;
    const VkDescriptorBufferInfo buffers[]{{LumenVolumeSupportUniform.Buffer,0,sizeof(frame)},{scene.Buffer,0,scene.Size},{origins.Buffer,0,origins.Size},{LumenVolumeSupport.Buffer,0,LumenVolumeSupport.Size}};
    VkWriteDescriptorSet writes[4]{};
    for (unsigned slot=0;slot<std::size(writes);++slot)
    { auto& w=writes[slot]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=LumenVolumeSupportSet; w.dstBinding=slot; w.descriptorCount=1;
      w.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w.pBufferInfo=&buffers[slot]; }
    UpdateDescriptorSets(Device,unsigned(std::size(writes)),writes,0,nullptr);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenVolumeSupportPipeline);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenDistanceLayout,0,1,&LumenVolumeSupportSet,0,nullptr);
    DistanceDispatch(CommandBuffer,(frame.Counts.X+LumenRadianceThreads-1)/LumenRadianceThreads,1,1);
    LumenDistanceBarrier(LumenVolumeSupport,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT); return true;
}
bool PackLumenTranslucency(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    if (!PrepareLumenVolumeSupport(probes,upload) || !EnsureLumenTranslucency()) return false;
    const auto offset=AlignUp(UniformOffset,UniformAlignment);
    if (offset>UniformBufferSize || sizeof(LumenVolumeFrame)>UniformBufferSize-offset) return Fail("Lumen volume exhausted frame uniforms");
    std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&LumenVolumeFrame,sizeof(LumenVolumeFrame)); UniformOffset=offset+sizeof(LumenVolumeFrame);
    LumenDistanceBarrier(probes.Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    const VkDescriptorBufferInfo buffers[]{{UniformBuffer.Buffer,0,sizeof(LumenVolumeFrame)},{probes.Buffer.Buffer,0,probes.Buffer.Size},{LumenVolumeSupport.Buffer,0,LumenVolumeSupport.Size}};
    VkWriteDescriptorSet writes[3]{};
    for (unsigned slot=0;slot<std::size(writes);++slot)
    { auto& w=writes[slot]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=LumenVolumeSet; w.dstBinding=slot; w.descriptorCount=1;
      w.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; w.pBufferInfo=&buffers[slot]; }
    UpdateDescriptorSets(Device,unsigned(std::size(writes)),writes,0,nullptr);
    TransitionImage(CommandBuffer,LumenVolumeAtlas,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const VkClearValue clear{}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass=HdrRenderPass; begin.framebuffer=LumenVolumeFramebuffer; begin.renderArea={{0,0},{LumenVolumeAtlas.Width,LumenVolumeAtlas.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0,0,LumenVolumeFrame.Atlas.X,LumenVolumeFrame.Atlas.Y,0,1}; CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenVolumePipeline); const auto dynamicOffset=std::uint32_t(offset);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenVolumeLayout,0,1,&LumenVolumeSet,1,&dynamicOffset);
    CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    TransitionImage(CommandBuffer,LumenVolumeAtlas,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); return true;
}
bool PrepareLumenTranslucency(const GlobalIlluminationDesc& desc)
{
    const auto& settings=FrameGI.Lumen;
    if ((!desc.Translucency || !settings.Translucency.Enabled) && !Hair.LumenReceiver) return true;
    LumenRadianceTraceUpload trace;
    if (!BuildLumenTranslucencyFrame(desc.ViewProjection,CreateInfo.Width,CreateInfo.Height,FrameGI,LumenVolumeFrame,trace,LastError)) return false;
    if (settings.GlobalDistance.Enabled)
    { trace.GlobalDistance=FrameLumenGlobalDistance; trace.Trace.GlobalNearDistance=settings.GlobalDistance.NearScale*FrameGI.MaximumDistance; trace.Trace.GlobalMaximumSteps=settings.GlobalDistance.MaximumSteps; }
    if (settings.WorldCache.Enabled) trace.WorldCache=FrameLumenWorldCache;
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,trace,rays)) return false; LumenVolumeRays=rays.Handle;
    const auto& sky=settings.SkyRadiance; LumenProbeIntegrationResult probes;
    if (!IntegrateLumenProbes(rays.Handle,{{sky[0],sky[1],sky[2]},false,MakeLumenEnvironmentLobe(settings)},probes)) return false; LumenVolumeProbes=probes.Handle;
    if (!PackLumenTranslucency(*LumenProbes.at(probes.Handle.Value),trace)) return false;
    ++PendingFrame.Statistics.GlobalIlluminationPassCount; PendingFrame.Statistics.LumenTranslucencyProbeCount=unsigned(trace.Probes.Origins.size()); return true;
}
