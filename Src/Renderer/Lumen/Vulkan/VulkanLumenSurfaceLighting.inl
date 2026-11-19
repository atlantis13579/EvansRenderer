struct LumenLightingRecord
{
    Impl& Owner; std::vector<unsigned> CardInstances; LumenLightingInfo Info; RawBuffer Buffer;
    std::shared_ptr<LumenSurfaceRecord> Surface; std::shared_ptr<LumenDistanceRecord> Distance;
    explicit LumenLightingRecord(Impl& owner):Owner(owner) {}
    ~LumenLightingRecord() { Owner.DestroyRawBuffer(Buffer); }
};
struct LumenLightingWork
{
    Impl& Owner; RawBuffer Uniform,Input,Readback; VkDescriptorPool Pool=0; VkDescriptorSet Set=0;
    std::shared_ptr<LumenLightingRecord> Record,Previous;
    explicit LumenLightingWork(Impl& owner):Owner(owner) {}
    ~LumenLightingWork()
    { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Uniform); Owner.DestroyRawBuffer(Input); Owner.DestroyRawBuffer(Readback); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenLightingRecord>> LumenLightingCaches;
std::shared_ptr<LumenLightingWork> LumenLightingPending;
VkDescriptorSetLayout LumenLightingSet=0; VkPipelineLayout LumenLightingLayout=0; VkPipeline LumenSurfaceLighting=0,LumenSurfaceRadiosity=0; VkSampler LumenLightingSampler=0;
PFN_vkCreateComputePipelines LightingCreateCompute=nullptr; PFN_vkCmdDispatch LightingDispatch=nullptr; PFN_vkCmdCopyBuffer LightingCopyBuffer=nullptr;
bool RecoverLumenSurfaceLighting()
{
    if (!LumenLightingPending) return true;
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for retained Lumen lighting")) return false;
    LumenLightingPending.reset(); return true;
}
void DestroyLumenLightingCaches()
{ DestroyLumenLightingAsync();
    LumenLightingPending.reset(); LumenLightingCaches.clear();
    if (LumenSurfaceLighting) DestroyPipelineFunction(Device,LumenSurfaceLighting,nullptr); LumenSurfaceLighting=0;
    if (LumenSurfaceRadiosity) DestroyPipelineFunction(Device,LumenSurfaceRadiosity,nullptr); LumenSurfaceRadiosity=0;
    if (LumenLightingLayout) DestroyPipelineLayoutFunction(Device,LumenLightingLayout,nullptr); LumenLightingLayout=0;
    if (LumenLightingSet) DestroyDescriptorSetLayoutFunction(Device,LumenLightingSet,nullptr); LumenLightingSet=0;
    if (LumenLightingSampler) DestroySamplerFunction(Device,LumenLightingSampler,nullptr); LumenLightingSampler=0;
}
bool EnsureLumenSurfaceLighting(bool indirect=false,bool hardware=false)
{
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorStorageBuffers<4 || DeviceLimits.maxDescriptorSetStorageBuffers<4 ||
        DeviceLimits.maxPerStageDescriptorSampledImages<LumenSurfacePlanes || DeviceLimits.maxPerStageDescriptorSamplers<LumenSurfacePlanes ||
        DeviceLimits.maxUniformBufferRange<sizeof(LumenSurfaceLightingFrame) || DeviceLimits.maxComputeWorkGroupInvocations<LumenLightingThreads || DeviceLimits.maxComputeWorkGroupSize[0]<LumenLightingThreads)
        return Fail("Vulkan Lumen lighting exceeds native descriptor/compute limits");
    if (!LoadDevice(LightingCreateCompute,"vkCreateComputePipelines") || !LoadDevice(LightingDispatch,"vkCmdDispatch") || !LoadDevice(LightingCopyBuffer,"vkCmdCopyBuffer")) return false;
    if (!LumenLightingSet)
    {
        VkDescriptorSetLayoutBinding bindings[8]{};
        for (unsigned slot=0;slot<8;++slot) bindings[slot]={slot,slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot<=3?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=8; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenLightingSet),"Create Lumen lighting descriptors")) return false;
    }
    if (!LumenLightingLayout)
    {
        if (!CreateLumenQueryLayout(LumenLightingSet,LumenLightingLayout)) return false;
    }
    if (!LumenLightingSampler)
    {
        VkSamplerCreateInfo create{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; create.magFilter=create.minFilter=VK_FILTER_NEAREST;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; create.maxAnisotropy=1;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&LumenLightingSampler),"Create Lumen lighting sampler")) return false;
    }
    if (hardware) return EnsureLumenHardwarePipeline(indirect?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct,LumenLightingLayout);
    auto& pipeline=indirect?LumenSurfaceRadiosity:LumenSurfaceLighting;
    if (pipeline) return true;
    VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,indirect?VulkanLumenSurfaceRadiosityShader:VulkanLumenSurfaceLightingShader,module,LastError)) return false;
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenLightingLayout; create.basePipelineIndex=-1;
    create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(LightingCreateCompute,Device,0,1,&create,nullptr,&pipeline); DestroyShaderModuleFunction(Device,module,nullptr);
    return Check(status,"Create Lumen lighting pipeline");
}
bool SubmitLumenLightingWork(const std::shared_ptr<LumenLightingWork>& work)
{
    if (!Check(EndCommandBuffer(CommandBuffer),"End Lumen lighting commands") || !Check(ResetFences(Device,1,&FrameFence),"Reset Lumen lighting fence")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&CommandBuffer;
    const auto status=QueueSubmit(GraphicsQueue,1,&submit,FrameFence);
    if (status!=VK_SUCCESS) { (void)RecreateSignaledFrameFence(); return Check(status,"Submit Lumen lighting commands"); }
    LumenLightingPending=work; return RecoverLumenSurfaceLighting();
}
bool SelectLumenPreviousLighting(const LumenSurfaceLightingUpload& upload,const std::shared_ptr<LumenSurfaceRecord>& surface,
    const std::shared_ptr<LumenDistanceRecord>& distance,std::shared_ptr<LumenLightingRecord>& output)
{
    if (!upload.PreviousLighting) return true;
    const auto found=LumenLightingCaches.find(upload.PreviousLighting.Value);
    if (found==LumenLightingCaches.end() || found->second->Surface!=surface || found->second->Distance!=distance || found->second->CardInstances!=upload.CardInstances)
        return Fail("Lumen radiosity source is expired or belongs to different native inputs");
    if (!ValidateLumenRadiositySource(surface->Identity,distance->Info.Identity,upload,found->second->Info,LastError)) return false;
    output=found->second; return true;
}
bool ReadLumenLightingRecord(const std::shared_ptr<LumenLightingRecord>& record,std::vector<LumenLightingTexel>& output)
{
    auto work=std::make_shared<LumenLightingWork>(*this); work->Record=record;
    const auto count=std::size_t(record->Info.Layout.Width)*record->Info.Layout.Height,bytes=count*sizeof(LumenLightingTexel);
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,host,work->Readback) || !BeginImmediate()) return false;
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferCopy copy{0,0,bytes}; LightingCopyBuffer(CommandBuffer,record->Buffer.Buffer,work->Readback.Buffer,1,&copy);
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->Readback,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
    if (!SubmitLumenLightingWork(work)) return false;
    std::vector<LumenLightingTexel> result(count); std::memcpy(result.data(),work->Readback.Mapped,bytes);
    if (!ValidateLumenLightingTexels(record->Info,result,LastError)) return false; output=std::move(result); return true;
}
bool RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    const auto source=LumenSurfaces.find(surface.Value); const auto occluders=LumenDistanceScenes.find(distance.Value);
    if (source==LumenSurfaces.end() || occluders==LumenDistanceScenes.end()) return Fail("Lumen relighting source handles are invalid or expired");
    std::shared_ptr<LumenLightingRecord> previous;
    if (!SelectLumenPreviousLighting(upload,source->second,occluders->second,previous)) return false;
    LumenSurfaceLightingInput input;
    if (!BuildLumenSurfaceLightingInput(source->second->Identity,source->second->Layout,source->second->Captures,occluders->second->Info,upload,input,LastError) || !EnsureLumenSurfaceLighting(bool(previous),bool(occluders->second->Hardware))) return false;
    const auto count=std::size_t(source->second->Layout.Width)*source->second->Layout.Height,bytes=count*sizeof(LumenLightingTexel),inputBytes=input.Words.size()*sizeof(std::uint32_t);
    if (std::max(bytes,inputBytes)>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan Lumen lighting exceeds storage range");
    const auto groups=LumenLightingGroups(input.Frame),x=std::min(groups,LumenLightingColumns),y=(groups+LumenLightingColumns-1)/LumenLightingColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan Lumen lighting exceeds dispatch limits");
    auto work=std::make_shared<LumenLightingWork>(*this); auto record=std::make_shared<LumenLightingRecord>(*this); work->Record=record; work->Previous=previous;
    record->Surface=source->second; record->Distance=occluders->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{source->second->Identity,occluders->second->Info.Identity,upload.LightingRevision},source->second->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=previous->Info.Bounce+1; }
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(input.Frame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->Uniform) || !CreateRawBuffer(inputBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Input) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,record->Buffer)) return false;
    std::memcpy(work->Uniform.Mapped,&input.Frame,sizeof(input.Frame)); std::memcpy(work->Input.Mapped,input.Words.data(),inputBytes);
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,LumenSurfacePlanes},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Lumen lighting descriptor pool")) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenLightingSet;
    if (!Check(AllocateDescriptorSets(Device,&allocate,&work->Set),"Allocate Lumen lighting descriptors")) return false;
    const RawBuffer* buffers[]{&work->Uniform,&record->Distance->Scene,&work->Input,&record->Buffer,previous?&previous->Buffer:nullptr}; VkDescriptorBufferInfo bufferInfos[5]{}; VkDescriptorImageInfo imageInfos[LumenSurfacePlanes]{};
    for (unsigned i=0;i<(previous?5u:4u);++i) bufferInfos[i]={buffers[i]->Buffer,0,buffers[i]->Size};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) imageInfos[plane]={LumenLightingSampler,record->Surface->Images[plane].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[8]{}; const unsigned writeCount=previous?8:7;
    for (unsigned slot=0;slot<writeCount;++slot)
    {
        auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->Set; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot<=3?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (slot>=1 && slot<=3) write.pImageInfo=&imageInfos[slot-1]; else write.pBufferInfo=&bufferInfos[slot?slot-3:0];
    }
    UpdateDescriptorSets(Device,writeCount,writes,0,nullptr); if (!BeginImmediate()) return false;
    LumenDistanceBarrier(work->Uniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->Input,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,record->Distance->Hardware?LumenHardwarePipelines[unsigned(previous?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct)]:previous?LumenSurfaceRadiosity:LumenSurfaceLighting);
    BindLumenHardware(CommandBuffer,LumenLightingLayout,*record->Distance); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenLightingLayout,0,1,&work->Set,0,nullptr);
    LightingDispatch(CommandBuffer,x,y,1);
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!SubmitLumenLightingWork(work)) return false;
    LumenSurfaceLightingResult result; result.Info=record->Info;
    if (upload.Readback && !ReadLumenLightingRecord(record,result.Texels)) return false;
    result.Handle=NewLumenLightingHandle(); if (!result.Handle) return Fail("Lumen lighting handle space exhausted");
    LumenLightingCaches.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    const auto found=LumenLightingCaches.find(handle.Value); if (found==LumenLightingCaches.end()) return Fail("Unknown Lumen lighting cache");
    return ReadLumenLightingRecord(found->second,output);
}
bool DestroyLumenLighting(LumenLightingHandle handle)
{ return LumenLightingCaches.erase(handle.Value)!=0 || Fail("Unknown Lumen lighting cache"); }
