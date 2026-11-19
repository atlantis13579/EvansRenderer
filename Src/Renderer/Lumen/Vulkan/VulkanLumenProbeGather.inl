LumenLightingHandle FrameLumenLighting;
LumenWorldCacheBinding FrameLumenWorldCache;
LumenDistanceSceneHandle FrameLumenGlobalDistance;
LumenRadianceHandle LumenGIRays;
LumenProbeHandle LumenGIProbes;
bool LumenGatherIndirect=false;
RawImage LumenGatherTarget;
VkFramebuffer LumenGatherFramebuffer=0;
VkDescriptorSetLayout LumenGatherSetLayout=0;
VkDescriptorPool LumenGatherPool=0;
VkDescriptorSet LumenGatherSet=0;
VkPipelineLayout LumenGatherLayout=0;
VkPipeline LumenGatherPipeline=0;
VkSampler LumenGatherSampler=0;
RawBuffer LumenVisibilityBuffer;
VkDescriptorSetLayout LumenVisibilitySetLayout=0;
VkDescriptorSet LumenVisibilitySet=0;
VkPipelineLayout LumenVisibilityLayout=0;
VkPipeline LumenVisibilityPipeline=0;
std::array<VkShaderModule,2> LumenGatherModules{};
void DestroyLumenGather()
{
    ReleaseLumenGIResults(); DestroyLumenTemporal(); DestroyLumenReflections(); DestroyLumenTranslucency(); DestroyLumenHairReceiver(); LumenGatherIndirect=false; FrameLumenLighting={}; FrameLumenWorldCache={}; FrameLumenGlobalDistance={};
    DestroyRawBuffer(LumenVisibilityBuffer);
    if (LumenVisibilityPipeline) DestroyPipelineFunction(Device,LumenVisibilityPipeline,nullptr); LumenVisibilityPipeline=0;
    if (LumenVisibilityLayout) DestroyPipelineLayoutFunction(Device,LumenVisibilityLayout,nullptr); LumenVisibilityLayout=0;
    if (LumenVisibilitySetLayout) DestroyDescriptorSetLayoutFunction(Device,LumenVisibilitySetLayout,nullptr); LumenVisibilitySetLayout=0; LumenVisibilitySet=0;
    if (LumenGatherFramebuffer) DestroyFramebufferFunction(Device,LumenGatherFramebuffer,nullptr); LumenGatherFramebuffer=0;
    DestroyRawImage(LumenGatherTarget);
    if (LumenGatherPipeline) DestroyPipelineFunction(Device,LumenGatherPipeline,nullptr); LumenGatherPipeline=0;
    for (auto& module:LumenGatherModules) { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    if (LumenGatherPool) DestroyDescriptorPoolFunction(Device,LumenGatherPool,nullptr); LumenGatherPool=LumenGatherSet=0;
    if (LumenGatherLayout) DestroyPipelineLayoutFunction(Device,LumenGatherLayout,nullptr); LumenGatherLayout=0;
    if (LumenGatherSetLayout) DestroyDescriptorSetLayoutFunction(Device,LumenGatherSetLayout,nullptr); LumenGatherSetLayout=0;
    if (LumenGatherSampler) DestroySamplerFunction(Device,LumenGatherSampler,nullptr); LumenGatherSampler=0;
}
bool EnsureLumenGather(const LumenProbeGatherFrame& frame)
{
    if (DeviceLimits.maxPerStageDescriptorStorageBuffers<3 || DeviceLimits.maxDescriptorSetStorageBuffers<3 || DeviceLimits.maxPerStageDescriptorSampledImages<5 ||
        DeviceLimits.maxPerStageDescriptorSamplers<5 || DeviceLimits.maxUniformBufferRange<sizeof(LumenProbeGatherFrame))
        return Fail("Vulkan Lumen gather exceeds fragment descriptor limits");
    if (!LumenGatherSetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[9]{};
        for (unsigned slot=0;slot<std::size(bindings);++slot) bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            slot<6?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=unsigned(std::size(bindings)); create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenGatherSetLayout),"Create Lumen gather descriptor layout")) return false;
    }
    if (!LumenGatherLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenGatherSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenGatherLayout),"Create Lumen gather pipeline layout")) return false;
    }
    if (!LumenGatherSampler)
    {
        VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; create.magFilter=create.minFilter=VK_FILTER_NEAREST;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&LumenGatherSampler),"Create Lumen gather sampler")) return false;
    }
    if (!LumenVisibilitySetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[6]{};
        for (unsigned slot=0;slot<std::size(bindings);++slot) bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            slot<4?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=unsigned(std::size(bindings)); create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenVisibilitySetLayout),"Create Lumen visibility descriptors")) return false;
    }
    if (!LumenVisibilityLayout)
    {
        if (!CreateLumenQueryLayout(LumenVisibilitySetLayout,LumenVisibilityLayout)) return false;
    }
    if (!LumenVisibilityPipeline)
    {
        constexpr VulkanShaderProgram program{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_GATHER_VISIBILITY_STAGE=1"};
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,program,module,LastError)) return false;
        VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenVisibilityLayout; create.basePipelineIndex=-1;
        create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenVisibilityPipeline); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create Lumen visibility pipeline")) return false;
    }
    if (!LumenGatherPool)
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,2},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,7},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6}};
        VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; create.maxSets=2; create.poolSizeCount=unsigned(std::size(sizes)); create.pPoolSizes=sizes;
        if (!Check(CreateDescriptorPoolFunction(Device,&create,nullptr,&LumenGatherPool),"Create Lumen gather descriptor pool")) return false;
    }
    if (!LumenGatherSet)
    {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=LumenGatherPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenGatherSetLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&LumenGatherSet),"Allocate Lumen gather descriptor set")) return false;
    }
    if (!LumenVisibilitySet)
    {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=LumenGatherPool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenVisibilitySetLayout;
        if (!Check(AllocateDescriptorSets(Device,&allocate,&LumenVisibilitySet),"Allocate Lumen visibility set")) return false;
    }
    const auto bytes=LumenProbeVisibilityBytes(frame);
    if (LumenVisibilityBuffer.Size!=bytes)
    {
        RawBuffer mask; if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,mask)) return false;
        DestroyRawBuffer(LumenVisibilityBuffer); LumenVisibilityBuffer=mask;
    }
    constexpr VulkanShaderProgram shaders[]{
        {"Lumen.glsl",VulkanShaderStage::Vertex,"EVANS_LUMEN_PROBE_GATHER_STAGE=1;EVANS_VERTEX_STAGE=1"},
        {"Lumen.glsl",VulkanShaderStage::Fragment,"EVANS_LUMEN_PROBE_GATHER_STAGE=1"}};
    if (!CreateFullscreenPipeline(shaders,LumenGatherModules,HdrRenderPass,LumenGatherPipeline,LumenGatherLayout)) return false;
    if (LumenGatherTarget.Image && LumenGatherTarget.Width==unsigned(CreateInfo.Width) && LumenGatherTarget.Height==unsigned(CreateInfo.Height)) return true;
    RawImage target; VkFramebuffer framebuffer=0;
    if (!CreateRawImage(CreateInfo.Width,CreateInfo.Height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,target) ||
        !CreateFramebuffer(HdrRenderPass,&target.View,1,target.Width,target.Height,framebuffer))
    { DestroyRawImage(target); return false; }
    if (LumenGatherFramebuffer) DestroyFramebufferFunction(Device,LumenGatherFramebuffer,nullptr);
    DestroyRawImage(LumenGatherTarget); LumenGatherTarget=target; LumenGatherFramebuffer=framebuffer; return true;
}
bool ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if ((GIActive && (!Techniques::UsesLumen(FrameGI) || GI.Resolved)) || !HdrActive || !GBufferWritten || !(DeferredActive?DeferredResolved:ForwardOpaqueComplete) ||
        (FrameSampleCount>1 && (!SamplesOpaqueResolved || SamplesColorRestored || SamplesTransparentWritten)) ||
        (AmbientOcclusionActive && !AmbientOcclusionResolved) || ReflectionResolved || FogResolved || LightShaftResolved || TemporalResolved ||
        MotionBlurResolved || DepthOfFieldResolved || RefractionResolved || !Hair.Complete() || !SubsurfaceComplete())
        return Fail("Lumen gather requires resolved opaque HDR materials before reflections, volumes and transparency");
    if (LumenHasTransparentMainDraws(PendingFrame)) return Fail("Lumen gather must precede transparent draws");
    const auto found=LumenProbes.find(handle.Value); if (found==LumenProbes.end()) return Fail("Unknown Lumen gather probes");
    const auto& record=*found->second; if (!FrameLumen.CanGather(handle,record.SourceHandle,LastError)) return false;
    const auto source=LumenRadianceTraces.find(record.SourceHandle.Value); if (source==LumenRadianceTraces.end()) return Fail("Lumen gather source was released");
    LumenProbeGatherFrame frame; if (!BuildLumenProbeGatherFrame(record.Info,source->second->Lighting->Distance->Info,settings,FrameReflectionShading,frame,LastError) || !EnsureLumenGather(frame)) return false;
    const auto& distance=*source->second->Lighting->Distance;
    if (settings.Visibility.Enabled && distance.Hardware && !EnsureLumenHardwarePipeline(LumenHardwareProgram::Visibility,LumenVisibilityLayout)) return false;
    frame.Model.Z=LumenGatherIndirect?1u:0u;
    const auto offset=AlignUp(UniformOffset,UniformAlignment);
    if (offset>UniformBufferSize || sizeof(frame)>UniformBufferSize-offset) return Fail("Lumen gather exhausted the frame uniform arena");
    std::memcpy(static_cast<std::uint8_t*>(UniformBuffer.Mapped)+offset,&frame,sizeof(frame)); UniformOffset=offset+sizeof(frame);
    const RawBuffer* buffers[]{&source->second->Origins,&record.Buffer}; VkBufferMemoryBarrier barriers[2]{};
    for (unsigned i=0;i<2;++i)
    { auto& b=barriers[i]; b.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
      b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.buffer=buffers[i]->Buffer; b.size=buffers[i]->Size; }
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,2,barriers,0,nullptr);
    RawImage* inputs[]{&HdrImages[0],&MainDepth,&GBufferImages[1],&GBufferImages[0],&GBufferImages[3]}; VkDescriptorImageInfo images[5]{};
    for (unsigned slot=0;slot<std::size(inputs);++slot)
    { TransitionImage(CommandBuffer,*inputs[slot],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); images[slot]={LumenGatherSampler,inputs[slot]->View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; }
    if (settings.Visibility.Enabled)
    {
        LumenDistanceBarrier(source->second->Origins,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        LumenDistanceBarrier(LumenVisibilityBuffer,VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        const auto& scene=source->second->Lighting->Distance->Scene;
        const VkDescriptorBufferInfo info[]{{UniformBuffer.Buffer,0,sizeof(frame)},{scene.Buffer,0,scene.Size},{buffers[0]->Buffer,0,buffers[0]->Size},
            {LumenVisibilityBuffer.Buffer,0,LumenVisibilityBuffer.Size}};
        const VkDescriptorImageInfo textures[]{images[1],images[2]}; VkWriteDescriptorSet updates[6]{};
        for (unsigned slot=0;slot<std::size(updates);++slot)
        { auto& w=updates[slot]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=LumenVisibilitySet; w.dstBinding=slot; w.descriptorCount=1;
          w.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:slot<4?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
          if (slot<4) w.pBufferInfo=&info[slot]; else w.pImageInfo=&textures[slot-4]; }
        UpdateDescriptorSets(Device,unsigned(std::size(updates)),updates,0,nullptr);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,distance.Hardware?LumenHardwarePipelines[unsigned(LumenHardwareProgram::Visibility)]:LumenVisibilityPipeline);
        BindLumenHardware(CommandBuffer,LumenVisibilityLayout,distance); const auto dynamicOffset=std::uint32_t(offset);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenVisibilityLayout,0,1,&LumenVisibilitySet,1,&dynamicOffset);
        const auto groups=(frame.Trace.Counts.X+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
        DistanceDispatch(CommandBuffer,std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
        LumenDistanceBarrier(LumenVisibilityBuffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
    const VkDescriptorBufferInfo infos[]{{UniformBuffer.Buffer,0,sizeof(frame)},{buffers[0]->Buffer,0,buffers[0]->Size},{buffers[1]->Buffer,0,buffers[1]->Size},{LumenVisibilityBuffer.Buffer,0,LumenVisibilityBuffer.Size}};
    VkWriteDescriptorSet writes[9]{};
    for (unsigned slot=0;slot<std::size(writes);++slot)
    { auto& w=writes[slot]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=LumenGatherSet; w.dstBinding=slot; w.descriptorCount=1;
      w.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:slot<6?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      if (slot==0) w.pBufferInfo=&infos[0]; else if (slot<6) w.pImageInfo=&images[slot-1]; else w.pBufferInfo=&infos[slot-5]; }
    // One gather is allowed per frame. BeginFrame waits for the preceding submit,
    // so this descriptor set is never rewritten while a draw still references it.
    UpdateDescriptorSets(Device,unsigned(std::size(writes)),writes,0,nullptr);
    TransitionImage(CommandBuffer,LumenGatherTarget,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const VkClearValue clear{}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass=HdrRenderPass; begin.framebuffer=LumenGatherFramebuffer; begin.renderArea={{0,0},{LumenGatherTarget.Width,LumenGatherTarget.Height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE); const VkViewport viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1};
    CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&begin.renderArea);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenGatherPipeline); const auto dynamicOffset=std::uint32_t(offset);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenGatherLayout,0,1,&LumenGatherSet,1,&dynamicOffset); CmdDraw(CommandBuffer,3,1,0,0); CmdEndRenderPass(CommandBuffer);
    if (!LumenGatherIndirect)
    {
    TransitionImage(CommandBuffer,LumenGatherTarget,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{}; copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.extent={LumenGatherTarget.Width,LumenGatherTarget.Height,1};
    CmdCopyImage(CommandBuffer,LumenGatherTarget.Image,LumenGatherTarget.Layout,HdrImages[0].Image,HdrImages[0].Layout,1,&copy);
    }
    FrameLumen.Gather(); ++PendingFrame.Statistics.GlobalIlluminationPassCount; return true;
}

// The ordinary GI pass owns these transient results; diagnostic APIs keep their
// explicit handles. FinishLumenFrame releases only this automatically owned pair.
bool ApplyLumenGI(const GlobalIlluminationDesc& desc)
{
    if (!FrameLumenLighting || LumenGIRays || LumenGIProbes) return Fail("Lumen requires a current surface cache and one frame consumer");
    const auto& settings=FrameGI.Lumen; LumenRadianceTraceUpload trace;
    trace.Screen.Enabled=true; trace.Screen.ViewProjection=desc.ViewProjection; trace.Screen.Width=CreateInfo.Width; trace.Screen.Height=CreateInfo.Height;
    trace.Screen.TileSide=settings.TileSide; trace.Screen.NormalBias=FrameGI.NormalBias;
    trace.Screen.Adaptive=settings.AdaptiveProbes; trace.Screen.AdaptiveLevels=settings.AdaptiveProbeLevels; trace.Screen.MinimumNormalCosine=FrameGI.NormalThreshold; trace.Screen.PlaneTolerance=FrameGI.DepthTolerance;
    trace.Probes.Sequence=settings.Temporal.Enabled?desc.Sampling.SampleIndex:0; trace.Probes.RaysPerProbe=settings.RaysPerProbe; trace.Probes.MaximumDistance=FrameGI.MaximumDistance;
    if (settings.GlobalDistance.Enabled)
    { trace.GlobalDistance=FrameLumenGlobalDistance; trace.Trace.GlobalNearDistance=settings.GlobalDistance.NearScale*FrameGI.MaximumDistance; trace.Trace.GlobalMaximumSteps=settings.GlobalDistance.MaximumSteps; }
    trace.Trace.MaximumSteps=FrameGI.MaximumSteps; trace.Hybrid=settings.ScreenTrace; if (settings.WorldCache.Enabled) trace.WorldCache=FrameLumenWorldCache;
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,trace,rays)) return false; LumenGIRays=rays.Handle;
    LumenProbeIntegrationResult probes; const auto& sky=settings.SkyRadiance;
    if (!IntegrateLumenProbes(rays.Handle,{{sky[0],sky[1],sky[2]},false,MakeLumenEnvironmentLobe(settings)},probes)) return false; LumenGIProbes=probes.Handle;
    LumenProbeGatherSettings gather; gather.Intensity=FrameGI.Intensity; gather.MinimumNormalCosine=FrameGI.NormalThreshold; gather.PlaneTolerance=FrameGI.DepthTolerance; gather.Visibility=settings.ProbeVisibility;
    LumenGatherIndirect=settings.Temporal.Enabled;
    const bool gathered=ResolveLumenProbes(probes.Handle,gather); LumenGatherIndirect=false;
    if (!gathered || (settings.Temporal.Enabled && !ResolveLumenTemporal(desc))) return false;
    if (!PrepareLumenTranslucency(desc) || !ResolveLumenHairReceiver()) return false;
    GI.Resolved=true; return true;
}
void ReleaseLumenGIResults()
{
    ReleaseLumenTranslucencyResults();
    if (LumenGIProbes) LumenProbes.erase(LumenGIProbes.Value);
    if (LumenGIRays) LumenRadianceTraces.erase(LumenGIRays.Value);
    LumenGIProbes={}; LumenGIRays={};
}
