struct LumenProbeRecord
{
    Impl& Owner; LumenProbeInfo Info; LumenRadianceHandle SourceHandle; RawBuffer Buffer;
    explicit LumenProbeRecord(Impl& owner):Owner(owner) {}
    ~LumenProbeRecord() { Owner.DestroyRawBuffer(Buffer); }
};
struct LumenProbeWork
{
    Impl& Owner; RawBuffer Uniform,Readback; VkDescriptorPool Pool=0; VkDescriptorSet Set=0;
    std::shared_ptr<LumenProbeRecord> Record; std::shared_ptr<LumenRadianceRecord> Source;
    explicit LumenProbeWork(Impl& owner):Owner(owner) {}
    ~LumenProbeWork()
    {
        if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr);
        Owner.DestroyRawBuffer(Uniform); Owner.DestroyRawBuffer(Readback);
    }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenProbeRecord>> LumenProbes;
std::shared_ptr<LumenProbeWork> LumenProbePending;
std::vector<std::shared_ptr<LumenProbeWork>> LumenFrameProbeWork;
VkDescriptorSetLayout LumenProbeSet=0; VkPipelineLayout LumenProbeLayout=0; VkPipeline LumenProbeIntegrate=0;
bool RecoverLumenProbes()
{
    if (!LumenProbePending) return true;
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for retained Lumen probe work")) return false;
    LumenProbePending.reset(); return true;
}
void DestroyLumenProbes()
{ DestroyLumenGather(); FrameLumen.Finish(false,LumenRadianceTraces,LumenProbes);
    LumenProbePending.reset(); LumenFrameProbeWork.clear(); LumenProbes.clear();
    if (LumenProbeIntegrate) DestroyPipelineFunction(Device,LumenProbeIntegrate,nullptr); LumenProbeIntegrate=0;
    if (LumenProbeLayout) DestroyPipelineLayoutFunction(Device,LumenProbeLayout,nullptr); LumenProbeLayout=0;
    if (LumenProbeSet) DestroyDescriptorSetLayoutFunction(Device,LumenProbeSet,nullptr); LumenProbeSet=0;
}
bool EnsureLumenProbeIntegration()
{
    if (!EnsureLumenDistanceTrace()) return false;
    if (DeviceLimits.maxPerStageDescriptorStorageBuffers<3 || DeviceLimits.maxDescriptorSetStorageBuffers<3 || DeviceLimits.maxUniformBufferRange<sizeof(LumenProbeIntegrationFrame))
        return Fail("Vulkan Lumen probe integration exceeds descriptor limits");
    if (!LumenProbeSet)
    {
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (unsigned slot=0;slot<4;++slot) bindings[slot]={slot,slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=4; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenProbeSet),"Create Lumen probe descriptors")) return false;
    }
    if (!LumenProbeLayout)
    {
        VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=1; create.pSetLayouts=&LumenProbeSet;
        if (!Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&LumenProbeLayout),"Create Lumen probe pipeline layout")) return false;
    }
    if (LumenProbeIntegrate) return true;
    VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanLumenProbeIntegrateShader,module,LastError)) return false;
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=LumenProbeLayout; create.basePipelineIndex=-1;
    create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&LumenProbeIntegrate); DestroyShaderModuleFunction(Device,module,nullptr);
    return Check(status,"Create Lumen probe integration pipeline");
}
bool SubmitLumenProbeWork(const std::shared_ptr<LumenProbeWork>& work)
{
    if (FrameOpen) { LumenFrameProbeWork.push_back(work); return true; }
    if (!Check(EndCommandBuffer(CommandBuffer),"End Lumen probe commands") || !Check(ResetFences(Device,1,&FrameFence),"Reset Lumen probe fence")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&CommandBuffer;
    const auto status=QueueSubmit(GraphicsQueue,1,&submit,FrameFence);
    if (status!=VK_SUCCESS) { (void)RecreateSignaledFrameFence(); return Check(status,"Submit Lumen probe work"); }
    LumenProbePending=work; return RecoverLumenProbes();
}
bool ReadLumenProbeRecord(const std::shared_ptr<LumenProbeRecord>& record,std::vector<LumenProbeIrradiance>& output)
{
    auto work=std::make_shared<LumenProbeWork>(*this); work->Record=record; const auto count=std::size_t(LumenRadianceProbeCount(record->Info.Source)),bytes=count*sizeof(LumenProbeIrradiance);
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,host,work->Readback) || !BeginImmediate()) return false;
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferCopy copy{0,0,bytes}; DistanceCopyBuffer(CommandBuffer,record->Buffer.Buffer,work->Readback.Buffer,1,&copy);
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    LumenDistanceBarrier(work->Readback,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
    if (!SubmitLumenProbeWork(work)) return false;
    std::vector<LumenProbeIrradiance> probes(count); std::memcpy(probes.data(),work->Readback.Mapped,bytes);
    if (!ValidateLumenProbeIrradiance(record->Info,probes,LastError)) return false;
    output=std::move(probes); return true;
}
bool IntegrateLumenProbes(LumenRadianceHandle sourceHandle,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    const auto found=LumenRadianceTraces.find(sourceHandle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen probe radiance source");
    const auto source=found->second; LumenProbeIntegrationFrame frame;
    if (!BuildLumenProbeIntegrationFrame(source->Info,upload,frame,LastError) || (FrameOpen && !FrameLumen.CanIntegrate(frame.Counts.X,LastError)) || !EnsureLumenProbeIntegration()) return false;
    const auto bytes=std::size_t(frame.Counts.X)*sizeof(LumenProbeIrradiance);
    if (bytes>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan Lumen probes exceed storage range");
    const auto groups=(frame.Counts.X+LumenProbeThreads-1)/LumenProbeThreads;
    if (groups>DeviceLimits.maxComputeWorkGroupCount[0]) return Fail("Vulkan Lumen probes exceed dispatch limit");
    auto work=std::make_shared<LumenProbeWork>(*this); auto record=std::make_shared<LumenProbeRecord>(*this); work->Record=record; work->Source=source;
    record->Info={source->Info,upload.SkyRadiance,upload.Environment}; record->SourceHandle=sourceHandle;
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(frame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->Uniform) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,record->Buffer)) return false;
    std::memcpy(work->Uniform.Mapped,&frame,sizeof(frame));
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Lumen probe descriptor pool")) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenProbeSet;
    if (!Check(AllocateDescriptorSets(Device,&allocate,&work->Set),"Allocate Lumen probe descriptors")) return false;
    const RawBuffer* buffers[]{&work->Uniform,&source->Buffer,&source->Origins,&record->Buffer}; VkDescriptorBufferInfo infos[4]{}; VkWriteDescriptorSet writes[4]{};
    for (unsigned slot=0;slot<4;++slot)
    {
        infos[slot]={buffers[slot]->Buffer,0,buffers[slot]->Size}; auto& write=writes[slot];
        write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->Set; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&infos[slot];
    }
    UpdateDescriptorSets(Device,4,writes,0,nullptr); if (!FrameOpen && !BeginImmediate()) return false;
    LumenDistanceBarrier(work->Uniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenProbeIntegrate); CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,LumenProbeLayout,0,1,&work->Set,0,nullptr); DistanceDispatch(CommandBuffer,groups,1,1);
    LumenDistanceBarrier(record->Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!SubmitLumenProbeWork(work)) return false;
    LumenProbeIntegrationResult result; result.Info=record->Info; if (upload.Readback && !ReadLumenProbeRecord(record,result.Probes)) return false;
    result.Handle=NewLumenProbeHandle(); if (!result.Handle) return Fail("Lumen probe handle space exhausted");
    LumenProbes.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,frame.Counts.X);
    output=std::move(result); return true;
}
bool ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    const auto found=LumenProbes.find(handle.Value); if (found==LumenProbes.end()) return Fail("Unknown Lumen probe cache");
    return ReadLumenProbeRecord(found->second,output);
}
bool DestroyLumenProbes(LumenProbeHandle handle) { return LumenProbes.erase(handle.Value)!=0 || Fail("Unknown Lumen probe cache"); }

// Handles are usable by later GPU work in this frame, then either published or
// cancelled together. Native command ownership is independent of publication.
bool LumenCommandsEnded() const { return FrameSubmitted || FrameCommandEnded; }
bool ValidateLumenRecording(bool readback)
{
    if (PassOpen || (FrameOpen && (readback || LumenCommandsEnded())))
        return Fail("Lumen frame work requires an open command stream, no graphics pass and no CPU readback");
    return true;
}
void FinishLumenFrame(bool publish)
{
    LumenHistory.Finish(publish); FinishLumenReflections(publish);
    FrameLumen.Finish(publish,LumenRadianceTraces,LumenProbes);
    ReleaseLumenGIResults();
    if (publish || !FrameSubmitted) { LumenFrameProbeWork.clear(); LumenFrameRadianceWork.clear(); }
}
struct LumenPresentPublication
{
    Impl& Owner; bool Published=false;
    ~LumenPresentPublication()
    { if (Published || !Owner.FrameOpen || Owner.LumenCommandsEnded()) Owner.FinishLumenFrame(Published); }
};
bool RecoverLumenFrameWork()
{
    if (FrameOpen || (LumenFrameProbeWork.empty() && LumenFrameRadianceWork.empty())) return true;
    if (FrameSubmitted && !Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Wait for retained Lumen frame work")) return false;
    LumenFrameProbeWork.clear(); LumenFrameRadianceWork.clear(); return true;
}
