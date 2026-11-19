struct LumenRadianceRecord
{
    std::shared_ptr<LumenDistanceRecord> TraceDistance;
    std::shared_ptr<LumenRadianceRecord> WorldCache;
    Impl& Owner; LumenRadianceInfo Info; RawBuffer Buffer,Origins; std::shared_ptr<LumenLightingRecord> Lighting;
    explicit LumenRadianceRecord(Impl& owner):Owner(owner) {}
    ~LumenRadianceRecord() { Owner.DestroyRawBuffer(Buffer); Owner.DestroyRawBuffer(Origins); }
};
struct LumenRadianceWork
{
    std::vector<std::shared_ptr<LumenRadianceRecord>> CopySources;
    Impl& Owner; RawBuffer TraceUniform,ResolveUniform,ScreenUniform,HybridUniform,WorldUniform,Mapping,Hits,Readback; VkDescriptorPool Pool=0; VkDescriptorSet Sets[4]{},WorldSet=0;
    RawBuffer Pyramid; std::array<RawBuffer,LumenScreenDepthLevels> DepthUniforms{}; std::array<VkDescriptorSet,LumenScreenDepthLevels> DepthSets{};
    std::shared_ptr<LumenRadianceRecord> Record;
    explicit LumenRadianceWork(Impl& owner):Owner(owner) {}
    ~LumenRadianceWork()
    {
        if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr);
        Owner.DestroyRawBuffer(TraceUniform); Owner.DestroyRawBuffer(ResolveUniform); Owner.DestroyRawBuffer(ScreenUniform);
        Owner.DestroyRawBuffer(WorldUniform); Owner.DestroyRawBuffer(HybridUniform); Owner.DestroyRawBuffer(Pyramid);
        for (auto& buffer:DepthUniforms) Owner.DestroyRawBuffer(buffer);
        Owner.DestroyRawBuffer(Mapping); Owner.DestroyRawBuffer(Hits); Owner.DestroyRawBuffer(Readback);
    }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenRadianceRecord>> LumenRadianceTraces;
LumenFrameTransaction FrameLumen;
std::shared_ptr<LumenRadianceWork> LumenRadiancePending;
std::vector<std::shared_ptr<LumenRadianceWork>> LumenFrameRadianceWork;
VkDescriptorSetLayout LumenRadianceSet=0,LumenScreenSet=0; VkPipelineLayout LumenRadianceLayout=0,LumenScreenLayout=0; VkPipeline LumenRadianceTrace=0,LumenRadianceResolve=0,LumenProbeTrace=0,LumenScreenPlacement=0,LumenReflectionRayShader=0;
VkDescriptorSetLayout LumenDepthSet=0; VkPipelineLayout LumenDepthLayout=0; VkPipeline LumenScreenDepth=0;
VkDescriptorSetLayout LumenHybridSet=0; VkPipelineLayout LumenHybridLayout=0; VkPipeline LumenHybridTrace=0;
VkDescriptorSetLayout LumenWorldSet=0; VkPipelineLayout LumenWorldLayout=0; VkPipeline LumenWorldLookup=0;
bool EnsureLumenWorldLookup()
{
    if (!LumenWorldSet)
    {
        VkDescriptorSetLayoutBinding bindings[5]{};
        for (unsigned slot=0;slot<5;++slot) bindings[slot]={slot,slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=5; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenWorldSet),"Create Lumen world lookup descriptors")) return false;
    }
    if (!LumenWorldLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenWorldSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenWorldLayout),"Create Lumen world lookup layout")) return false;
    }
    if (LumenWorldLookup) return true;
    VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenWorldLookupShader,module,LastError)) return false;
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenWorldLayout; create.basePipelineIndex=-1;
    create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenWorldLookup); DestroyShaderModuleFunction(Device,module,nullptr);
    return Check(status,"Create Lumen world lookup pipeline");
}
bool RecoverLumenRadianceTrace()
{
    if (!LumenRadiancePending) return true;
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for retained Lumen radiance work")) return false;
    LumenRadiancePending.reset(); return true;
}
void DestroyLumenRadianceTraces()
{
    if (LumenWorldLookup) DestroyPipelineFunction(Device,LumenWorldLookup,nullptr); LumenWorldLookup=0;
    if (LumenWorldLayout) DestroyPipelineLayoutFunction(Device,LumenWorldLayout,nullptr); LumenWorldLayout=0;
    if (LumenWorldSet) DestroyDescriptorSetLayoutFunction(Device,LumenWorldSet,nullptr); LumenWorldSet=0;
    LumenRadiancePending.reset(); LumenFrameRadianceWork.clear(); LumenRadianceTraces.clear();
    if (LumenScreenDepth) DestroyPipelineFunction(Device,LumenScreenDepth,nullptr); LumenScreenDepth=0;
    if (LumenDepthLayout) DestroyPipelineLayoutFunction(Device,LumenDepthLayout,nullptr); LumenDepthLayout=0;
    if (LumenDepthSet) DestroyDescriptorSetLayoutFunction(Device,LumenDepthSet,nullptr); LumenDepthSet=0;
    if (LumenHybridTrace) DestroyPipelineFunction(Device,LumenHybridTrace,nullptr); LumenHybridTrace=0;
    if (LumenHybridLayout) DestroyPipelineLayoutFunction(Device,LumenHybridLayout,nullptr); LumenHybridLayout=0;
    if (LumenHybridSet) DestroyDescriptorSetLayoutFunction(Device,LumenHybridSet,nullptr); LumenHybridSet=0;
    if (LumenReflectionRayShader) DestroyPipelineFunction(Device,LumenReflectionRayShader,nullptr); LumenReflectionRayShader=0;
    if (LumenScreenPlacement) DestroyPipelineFunction(Device,LumenScreenPlacement,nullptr); LumenScreenPlacement=0;
    if (LumenScreenLayout) DestroyPipelineLayoutFunction(Device,LumenScreenLayout,nullptr); LumenScreenLayout=0;
    if (LumenScreenSet) DestroyDescriptorSetLayoutFunction(Device,LumenScreenSet,nullptr); LumenScreenSet=0;
    if (LumenProbeTrace) DestroyPipelineFunction(Device,LumenProbeTrace,nullptr); LumenProbeTrace=0;
    if (LumenRadianceTrace) DestroyPipelineFunction(Device,LumenRadianceTrace,nullptr); LumenRadianceTrace=0;
    if (LumenRadianceResolve) DestroyPipelineFunction(Device,LumenRadianceResolve,nullptr); LumenRadianceResolve=0;
    if (LumenRadianceLayout) DestroyPipelineLayoutFunction(Device,LumenRadianceLayout,nullptr); LumenRadianceLayout=0;
    if (LumenRadianceSet) DestroyDescriptorSetLayoutFunction(Device,LumenRadianceSet,nullptr); LumenRadianceSet=0;
}
bool EnsureLumenRadianceTrace(bool hardware=false)
{
    if (!EnsureLumenDistanceTrace()) return false;
    if (DeviceLimits.maxPerStageDescriptorStorageBuffers<4 || DeviceLimits.maxDescriptorSetStorageBuffers<4 ||
        DeviceLimits.maxPerStageDescriptorSampledImages<3 || DeviceLimits.maxPerStageDescriptorSamplers<3 || DeviceLimits.maxUniformBufferRange<sizeof(LumenScreenTraceFrame))
        return Fail("Vulkan Lumen radiance exceeds descriptor limits");
    if (!LumenDepthSet)
    {
        const VkDescriptorSetLayoutBinding bindings[]{{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=3; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenDepthSet),"Create Lumen screen depth descriptors")) return false;
    }
    if (!LumenDepthLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenDepthSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenDepthLayout),"Create Lumen screen depth layout")) return false;
    }
    if (!LumenHybridSet)
    {
        VkDescriptorSetLayoutBinding bindings[8]{};
        for (unsigned slot=0;slot<8;++slot) bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:(slot<4 || slot==7)?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=8; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenHybridSet),"Create Lumen hybrid descriptors")) return false;
    }
    if (!LumenHybridLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenHybridSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenHybridLayout),"Create Lumen hybrid layout")) return false;
    }
    if (!LumenScreenSet)
    {
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (unsigned slot=0;slot<4;++slot) bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot==3?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=4; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenScreenSet),"Create Lumen screen descriptors")) return false;
    }
    if (!LumenScreenLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenScreenSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenScreenLayout),"Create Lumen screen pipeline layout")) return false;
    }
    if (!LumenRadianceSet)
    {
        VkDescriptorSetLayoutBinding bindings[6]{};
        for (unsigned slot=0;slot<6;++slot) bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot==1?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=6; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenRadianceSet),"Create Lumen radiance descriptors")) return false;
    }
    if (!LumenRadianceLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenRadianceSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenRadianceLayout),"Create Lumen radiance pipeline layout")) return false;
    }
    const VulkanShaderProgram programs[]{VulkanLumenRadianceTraceShader,VulkanLumenRadianceResolveShader,VulkanLumenProbeTraceShader,VulkanLumenScreenProbeShader,VulkanLumenScreenTraceShader,VulkanLumenScreenDepthShader,VulkanLumenReflectionRayShader};
    VkPipeline* targets[]{&LumenRadianceTrace,&LumenRadianceResolve,&LumenProbeTrace,&LumenScreenPlacement,&LumenHybridTrace,&LumenScreenDepth,&LumenReflectionRayShader}; const VkPipelineLayout layouts[]{LumenDistanceLayout,LumenRadianceLayout,LumenDistanceLayout,LumenScreenLayout,LumenHybridLayout,LumenDepthLayout,LumenScreenLayout};
    for (unsigned stage=0;stage<std::size(programs);++stage)
    {
        if (*targets[stage]) continue;
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,programs[stage],module,LastError)) return false;
        VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=layouts[stage]; create.basePipelineIndex=-1;
        create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,targets[stage]); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create Lumen radiance pipeline")) return false;
    }
    return !hardware || (EnsureLumenHardwarePipeline(LumenHardwareProgram::Radiance,LumenDistanceLayout) && EnsureLumenHardwarePipeline(LumenHardwareProgram::Probes,LumenDistanceLayout));
}
bool SubmitLumenRadianceWork(const std::shared_ptr<LumenRadianceWork>& work)
{
    if (FrameOpen) { LumenFrameRadianceWork.push_back(work); return true; }
    if (!Check(EndCommandBuffer(CommandBuffer),"End Lumen radiance commands") || !Check(ResetFences(Device,1,&FrameFence),"Reset Lumen radiance fence")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&CommandBuffer;
    const auto status=QueueSubmit(GraphicsQueue,1,&submit,FrameFence);
    if (status!=VK_SUCCESS) { (void)RecreateSignaledFrameFence(); return Check(status,"Submit Lumen radiance work"); }
    LumenRadiancePending=work; return RecoverLumenRadianceTrace();
}
bool ReadLumenOriginRecord(const std::shared_ptr<LumenRadianceRecord>& record,std::vector<LumenProbeOrigin>& output)
{
    if (!LumenRadianceProbeCount(record->Info)) return Fail("This Lumen trace has no probe-origin buffer");
    auto work=std::make_shared<LumenRadianceWork>(*this); work->Record=record; const auto bytes=std::size_t(LumenRadianceProbeCount(record->Info))*sizeof(LumenProbeOrigin);
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,host,work->Readback) || !BeginImmediate()) return false;
    LumenDistanceBarrier(record->Origins,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferCopy copy{0,0,bytes}; DistanceCopyBuffer(CommandBuffer,record->Origins.Buffer,work->Readback.Buffer,1,&copy);
    LumenDistanceBarrier(record->Origins,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->Readback,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
    if (!SubmitLumenRadianceWork(work)) return false;
    std::vector<LumenProbeOrigin> samples(LumenRadianceProbeCount(record->Info)); std::memcpy(samples.data(),work->Readback.Mapped,bytes);
    if (!ValidateLumenRadianceOrigins(record->Info,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool ReadLumenRadianceRecord(const std::shared_ptr<LumenRadianceRecord>& record,std::vector<LumenRadianceSample>& output)
{
    auto work=std::make_shared<LumenRadianceWork>(*this); work->Record=record; const auto bytes=std::size_t(record->Info.RayCount)*sizeof(LumenRadianceSample);
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,host,work->Readback) || !BeginImmediate()) return false;
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferCopy copy{0,0,bytes}; DistanceCopyBuffer(CommandBuffer,record->Buffer.Buffer,work->Readback.Buffer,1,&copy);
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->Readback,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
    if (!SubmitLumenRadianceWork(work)) return false;
    std::vector<LumenRadianceSample> samples(record->Info.RayCount); std::memcpy(samples.data(),work->Readback.Mapped,bytes);
    if (!ValidateLumenRadianceSamples(record->Info,record->Lighting->Distance->Info,record->Lighting->CardInstances,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (upload.Hybrid.Enabled && (!FrameOpen || (!upload.Reflections.Enabled && (GI.Resolved || FrameLumen.HasGathered())) || !HdrActive || !GBufferWritten || !(DeferredActive?DeferredResolved:ForwardOpaqueComplete) ||
        (FrameSampleCount>1 && (!SamplesOpaqueResolved || SamplesColorRestored || SamplesTransparentWritten)) ||
        (AmbientOcclusionActive && !AmbientOcclusionResolved) || ReflectionResolved || FogResolved || LightShaftResolved || TemporalResolved ||
        MotionBlurResolved || DepthOfFieldResolved || RefractionResolved || !Hair.Complete() || !SubsurfaceComplete()))
        return Fail("Lumen hybrid tracing requires resolved opaque HDR before secondary composition");
    if (upload.Hybrid.Enabled && LumenHasTransparentMainDraws(PendingFrame)) return Fail("Lumen hybrid tracing must precede transparent draws");
    if (upload.Screen.Enabled && (!FrameOpen || !GBufferWritten || (FrameSampleCount>1 && !SamplesOpaqueResolved) ||
        upload.Screen.Width!=unsigned(CreateInfo.Width) || upload.Screen.Height!=unsigned(CreateInfo.Height)))
        return Fail("Lumen screen placement needs this frame's resolved G-buffer and matching viewport");
    const auto found=LumenLightingCaches.find(lighting.Value); if (found==LumenLightingCaches.end()) return Fail("Unknown Lumen radiance source lighting");
    const auto source=found->second; LumenRadianceInput input;
    auto distance=source->Distance;
    if (upload.GlobalDistance)
    {
        const auto global=LumenDistanceScenes.find(upload.GlobalDistance.Value);
        if (global==LumenDistanceScenes.end() || global->second->Base!=source->Distance) return Fail("Global distance scene has a different or expired mesh source");
        distance=global->second;
    }
    if (!BuildLumenRadianceInput(source->Info,distance->Info,source->Surface->Captures,source->CardInstances,upload,input,LastError) || (FrameOpen && !FrameLumen.CanTrace(input.Resolve.Counts.X,LastError)) || !EnsureLumenRadianceTrace(bool(distance->Hardware))) return false;
    std::shared_ptr<LumenRadianceRecord> worldCache; LumenWorldLookupFrame worldFrame{};
    if (upload.WorldCache.Radiance)
    {
        const auto cache=LumenRadianceTraces.find(upload.WorldCache.Radiance.Value);
        if (cache==LumenRadianceTraces.end() || cache->second->Lighting!=source) return Fail("World radiance cache has a different or expired lighting source");
        worldCache=cache->second;
        if (!BuildLumenWorldLookupFrame(worldCache->Info,upload,input,worldFrame,LastError) || !EnsureLumenWorldLookup()) return false;
    }
    const auto sources=LumenRadianceSources(upload,input),traceFrame=LumenRadianceFrameData(input);
    const auto count=std::size_t(input.Resolve.Counts.X),bytes=count*sizeof(LumenRadianceSample),mapBytes=input.Words.size()*sizeof(std::uint32_t);
    if (std::max({bytes,mapBytes,input.Depth.Bytes})>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan Lumen radiance exceeds storage range");
    const auto groups=unsigned((count+LumenRadianceThreads-1)/LumenRadianceThreads),x=std::min(groups,LumenRadianceColumns),y=(groups+LumenRadianceColumns-1)/LumenRadianceColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan Lumen radiance exceeds dispatch limits");
    auto work=std::make_shared<LumenRadianceWork>(*this); auto record=std::make_shared<LumenRadianceRecord>(*this); work->Record=record;
    record->Info=MakeLumenRadianceInfo(source->Info.Identity,upload,input,worldCache?unsigned(worldFrame.Query.W):0); record->Info.HardwareTracing=bool(distance->Hardware); record->Lighting=source; record->WorldCache=worldCache; record->TraceDistance=distance;
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(traceFrame.Bytes,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->TraceUniform) ||
        !CreateRawBuffer(sizeof(input.Resolve),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->ResolveUniform) ||
        !CreateRawBuffer(sources.Bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,upload.Screen.Enabled?VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:host,record->Origins) ||
        !CreateRawBuffer(mapBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Mapping) ||
        !CreateRawBuffer(count*sizeof(LumenDistanceHit),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,work->Hits) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,record->Buffer)) return false;
    std::memcpy(work->TraceUniform.Mapped,traceFrame.Data,traceFrame.Bytes); std::memcpy(work->ResolveUniform.Mapped,&input.Resolve,sizeof(input.Resolve));
    if (!upload.Screen.Enabled) std::memcpy(record->Origins.Mapped,sources.Data,sources.Bytes); std::memcpy(work->Mapping.Mapped,input.Words.data(),mapBytes);
    const auto levels=unsigned(input.Depth.Levels.size());
    if (levels)
    {
        if (!CreateRawBuffer(input.Depth.Bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,work->Pyramid)) return false;
        for (unsigned level=0;level<levels;++level)
        {
            const auto& frame=input.Depth.Levels[level]; const auto levelGroups=(frame.Destination.X*frame.Destination.Y+LumenRadianceThreads-1)/LumenRadianceThreads;
            if (std::min(levelGroups,LumenRadianceColumns)>DeviceLimits.maxComputeWorkGroupCount[0] || (levelGroups+LumenRadianceColumns-1)/LumenRadianceColumns>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan Lumen depth exceeds dispatch limits");
            if (!CreateRawBuffer(sizeof(frame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->DepthUniforms[level])) return false;
            std::memcpy(work->DepthUniforms[level].Mapped,&frame,sizeof(frame));
        }
    }
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,5+levels},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,16+levels},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,6+levels}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=5+levels; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Lumen radiance descriptor pool")) return false;
    const VkDescriptorSetLayout layouts[]{LumenDistanceSet,LumenRadianceSet,LumenScreenSet,LumenHybridSet}; VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=upload.Hybrid.Enabled?4:upload.Screen.Enabled?3:2; allocate.pSetLayouts=layouts;
    if (!Check(AllocateDescriptorSets(Device,&allocate,work->Sets),"Allocate Lumen radiance descriptors")) return false;
    for (unsigned level=0;level<levels;++level)
    {
        VkDescriptorSetAllocateInfo depthAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; depthAllocate.descriptorPool=work->Pool;
        depthAllocate.descriptorSetCount=1; depthAllocate.pSetLayouts=&LumenDepthSet;
        if (!Check(AllocateDescriptorSets(Device,&depthAllocate,&work->DepthSets[level]),"Allocate Lumen depth descriptors")) return false;
        const VkDescriptorBufferInfo uniform{work->DepthUniforms[level].Buffer,0,sizeof(LumenScreenDepthFrame)},tree{work->Pyramid.Buffer,0,work->Pyramid.Size};
        const VkDescriptorImageInfo image{LumenLightingSampler,MainDepth.View,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL}; VkWriteDescriptorSet depthWrites[3]{};
        for (unsigned slot=0;slot<3;++slot)
        {
            auto& write=depthWrites[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->DepthSets[level]; write.dstBinding=slot; write.descriptorCount=1;
            write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot==1?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            if (slot==1) write.pImageInfo=&image; else write.pBufferInfo=slot==0?&uniform:&tree;
        }
        UpdateDescriptorSets(Device,3,depthWrites,0,nullptr);
    }
    const RawBuffer* traceBuffers[]{&work->TraceUniform,&distance->Scene,&record->Origins,&work->Hits};
    const RawBuffer* resolveBuffers[]{&work->ResolveUniform,&work->Hits,&source->Buffer,&work->Mapping,&record->Buffer};
    VkDescriptorBufferInfo traceInfos[4]{},resolveInfos[5]{};
    for (unsigned i=0;i<4;++i) traceInfos[i]={traceBuffers[i]->Buffer,0,traceBuffers[i]->Size};
    for (unsigned i=0;i<5;++i) resolveInfos[i]={resolveBuffers[i]->Buffer,0,resolveBuffers[i]->Size};
    const VkDescriptorImageInfo normal{LumenLightingSampler,source->Surface->Images[1].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; VkWriteDescriptorSet writes[10]{};
    for (unsigned slot=0;slot<4;++slot)
    {
        auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->Sets[0]; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&traceInfos[slot];
    }
    for (unsigned slot=0;slot<6;++slot)
    {
        auto& write=writes[4+slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->Sets[1]; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot==1?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (slot==1) write.pImageInfo=&normal; else write.pBufferInfo=&resolveInfos[slot?slot-1:0];
    }
    UpdateDescriptorSets(Device,10,writes,0,nullptr);
    if (worldCache)
    {
        if (!CreateRawBuffer(sizeof(worldFrame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->WorldUniform)) return false;
        std::memcpy(work->WorldUniform.Mapped,&worldFrame,sizeof(worldFrame));
        VkDescriptorSetAllocateInfo worldAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; worldAllocate.descriptorPool=work->Pool;
        worldAllocate.descriptorSetCount=1; worldAllocate.pSetLayouts=&LumenWorldSet;
        if (!Check(AllocateDescriptorSets(Device,&worldAllocate,&work->WorldSet),"Allocate Lumen world lookup descriptors")) return false;
        const RawBuffer* buffers[]{&work->WorldUniform,&worldCache->Buffer,&record->Origins,&work->Hits,&record->Buffer};
        VkDescriptorBufferInfo infos[5]{}; VkWriteDescriptorSet worldWrites[5]{};
        for (unsigned slot=0;slot<5;++slot)
        {
            infos[slot]={buffers[slot]->Buffer,0,buffers[slot]->Size}; auto& write=worldWrites[slot];
            write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->WorldSet; write.dstBinding=slot; write.descriptorCount=1;
            write.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&infos[slot];
        }
        UpdateDescriptorSets(Device,5,worldWrites,0,nullptr);
    }
    // Complete fallible allocations before recording any frame commands.
    if (upload.Hybrid.Enabled)
    {
        if (!CreateRawBuffer(sizeof(input.Hybrid),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->HybridUniform)) return false;
        std::memcpy(work->HybridUniform.Mapped,&input.Hybrid,sizeof(input.Hybrid));
    }
    if (upload.Screen.Enabled)
    {
        const auto placement=LumenRadiancePlacementData(input);
        if (!CreateRawBuffer(placement.Bytes,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->ScreenUniform)) return false;
        std::memcpy(work->ScreenUniform.Mapped,placement.Data,placement.Bytes);
        const VkDescriptorBufferInfo screenBuffers[]{{work->ScreenUniform.Buffer,0,work->ScreenUniform.Size},{record->Origins.Buffer,0,record->Origins.Size}};
        const VkDescriptorImageInfo screenImages[]{{LumenLightingSampler,MainDepth.View,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
            {LumenLightingSampler,GBufferImages[1].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        VkWriteDescriptorSet screenWrites[4]{};
        for (unsigned slot=0;slot<4;++slot)
        {
            auto& write=screenWrites[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->Sets[2]; write.dstBinding=slot; write.descriptorCount=1;
            write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot==3?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (slot==0 || slot==3) write.pBufferInfo=&screenBuffers[slot==3?1:0]; else write.pImageInfo=&screenImages[slot-1];
        }
        UpdateDescriptorSets(Device,4,screenWrites,0,nullptr);
        TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL); TransitionImage(CommandBuffer,GBufferImages[1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        LumenDistanceBarrier(work->ScreenUniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,upload.Reflections.Enabled?LumenReflectionRayShader:LumenScreenPlacement); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenScreenLayout,0,1,&work->Sets[2],0,nullptr);
        DistanceDispatch(CommandBuffer,(input.Screen.Grid.Y+LumenRadianceThreads-1)/LumenRadianceThreads,1,1);
        LumenDistanceBarrier(record->Origins,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    if (upload.Hybrid.Enabled)
    {
        const RawBuffer* buffers[]{&work->HybridUniform,&record->Origins,&work->Hits,&record->Buffer,levels?&work->Pyramid:&record->Origins}; VkDescriptorBufferInfo infos[5]{};
        for (unsigned i=0;i<5;++i) infos[i]={buffers[i]->Buffer,0,buffers[i]->Size};
        TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        TransitionImage(CommandBuffer,GBufferImages[1],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); TransitionImage(CommandBuffer,HdrImages[0],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const VkDescriptorImageInfo images[]{{LumenLightingSampler,MainDepth.View,MainDepth.Layout},
            {LumenLightingSampler,GBufferImages[1].View,GBufferImages[1].Layout},{LumenLightingSampler,HdrImages[0].View,HdrImages[0].Layout}};
        VkWriteDescriptorSet hybridWrites[8]{};
        for (unsigned slot=0;slot<8;++slot)
        {
            auto& write=hybridWrites[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->Sets[3]; write.dstBinding=slot; write.descriptorCount=1;
            write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:(slot<4 || slot==7)?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (slot<4 || slot==7) write.pBufferInfo=&infos[slot==7?4:slot]; else write.pImageInfo=&images[slot-4];
        }
        UpdateDescriptorSets(Device,8,hybridWrites,0,nullptr);
        for (unsigned level=0;level<levels;++level)
        {
            LumenDistanceBarrier(work->DepthUniforms[level],VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenScreenDepth); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenDepthLayout,0,1,&work->DepthSets[level],0,nullptr);
            const auto& extent=input.Depth.Levels[level].Destination; const auto levelGroups=(extent.X*extent.Y+LumenRadianceThreads-1)/LumenRadianceThreads;
            DistanceDispatch(CommandBuffer,std::min(levelGroups,LumenRadianceColumns),(levelGroups+LumenRadianceColumns-1)/LumenRadianceColumns,1);
            LumenDistanceBarrier(work->Pyramid,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        LumenDistanceBarrier(work->HybridUniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenHybridTrace); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenHybridLayout,0,1,&work->Sets[3],0,nullptr);
        DistanceDispatch(CommandBuffer,x,y,1);
        LumenDistanceBarrier(work->Hits,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        LumenDistanceBarrier(record->Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    if (!FrameOpen && !BeginImmediate()) return false;
    LumenDistanceBarrier(work->TraceUniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->ResolveUniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!upload.Screen.Enabled) LumenDistanceBarrier(record->Origins,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->Mapping,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (worldCache)
    {
        LumenDistanceBarrier(work->WorldUniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        LumenDistanceBarrier(worldCache->Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenWorldLookup); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenWorldLayout,0,1,&work->WorldSet,0,nullptr);
        DistanceDispatch(CommandBuffer,x,y,1);
        LumenDistanceBarrier(work->Hits,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        LumenDistanceBarrier(record->Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,distance->Hardware?LumenHardwarePipelines[unsigned(input.ProbeTrace.Sampling.X?LumenHardwareProgram::Probes:LumenHardwareProgram::Radiance)]:input.ProbeTrace.Sampling.X?LumenProbeTrace:LumenRadianceTrace);
    BindLumenHardware(CommandBuffer,LumenDistanceLayout,*distance); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenDistanceLayout,0,1,&work->Sets[0],0,nullptr); DistanceDispatch(CommandBuffer,x,y,1);
    LumenDistanceBarrier(work->Hits,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenRadianceResolve); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenRadianceLayout,0,1,&work->Sets[1],0,nullptr); DistanceDispatch(CommandBuffer,x,y,1);
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!SubmitLumenRadianceWork(work)) return false;
    if (FrameOpen && distance->Hardware) PendingFrame.Statistics.LumenHardwareRayCount+=input.Resolve.Counts.X;
    LumenRadianceTraceResult result; result.Info=record->Info; if (upload.Readback && !ReadLumenRadianceRecord(record,result.Samples)) return false;
    result.Handle=NewLumenRadianceHandle(); if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,result.Info.RayCount);
    output=std::move(result); return true;
}
bool AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!ValidateLumenRadianceAssembly(assembly,LastError) || !RecoverLumenRadianceTrace()) return false;
    auto work=std::make_shared<LumenRadianceWork>(*this);
    for (const auto& range:assembly.Copies)
    {
        const auto source=LumenRadianceTraces.find(range.Source.Value);
        if (source==LumenRadianceTraces.end()) return Fail("World cache assembly source is unavailable");
        if (!ValidateLumenRadianceCopy(source->second->Info,assembly,range,LastError)) return false;
        if (!work->CopySources.empty() && source->second->Lighting!=work->CopySources.front()->Lighting) return Fail("World cache assembly has mixed scene revisions");
        work->CopySources.push_back(source->second);
    }
    auto record=std::make_shared<LumenRadianceRecord>(*this); work->Record=record;
    record->Lighting=work->CopySources.front()->Lighting; record->Info=work->CopySources.front()->Info;
    record->Info.Probes=assembly.Probes; record->Info.RayCount=unsigned(assembly.Probes.Origins.size())*assembly.Probes.RaysPerProbe*assembly.Probes.DepthLayers;
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(assembly.Probes.Origins.size()*sizeof(LumenProbeOrigin),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,host,record->Origins) ||
        !CreateRawBuffer(std::size_t(record->Info.RayCount)*sizeof(LumenRadianceSample),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,record->Buffer)) return false;
    std::memcpy(record->Origins.Mapped,assembly.Probes.Origins.data(),assembly.Probes.Origins.size()*sizeof(LumenProbeOrigin));
    if (!BeginImmediate()) return false;
    LumenDistanceBarrier(record->Origins,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const auto probeBytes=VkDeviceSize(assembly.Probes.RaysPerProbe)*assembly.Probes.DepthLayers*sizeof(LumenRadianceSample);
    for (unsigned i=0;i<assembly.Copies.size();++i)
    {
        const auto& range=assembly.Copies[i]; const auto& source=work->CopySources[i]->Buffer;
        LumenDistanceBarrier(source,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferCopy region{range.SourceProbe*probeBytes,range.DestinationProbe*probeBytes,range.ProbeCount*probeBytes};
        DistanceCopyBuffer(CommandBuffer,source.Buffer,record->Buffer.Buffer,1,&region);
        LumenDistanceBarrier(source,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!SubmitLumenRadianceWork(work)) return false;
    LumenRadianceTraceResult result; result.Info=record->Info; result.Handle=NewLumenRadianceHandle();
    if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    const auto found=LumenRadianceTraces.find(handle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen radiance trace");
    if (!LumenRadianceProbeCount(found->second->Info)) return Fail("Lumen trace has no probe origins");
    return ReadLumenOriginRecord(found->second,output);
}
bool ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    const auto found=LumenRadianceTraces.find(handle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen radiance trace");
    return ReadLumenRadianceRecord(found->second,output);
}
bool DestroyLumenRadiance(LumenRadianceHandle handle) { return LumenRadianceTraces.erase(handle.Value)!=0 || Fail("Unknown Lumen radiance trace"); }
