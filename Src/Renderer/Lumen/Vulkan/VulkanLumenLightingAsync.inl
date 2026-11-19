struct LumenLightingSliceWork : LumenLightingJob
{
    Impl& Owner; std::shared_ptr<LumenLightingRecord> Record,Previous; std::shared_ptr<LumenLightingWork> Resources;
    VkCommandPool CommandsPool=0; VkCommandBuffer Commands=nullptr; VkFence Completion=0;
    explicit LumenLightingSliceWork(Impl& owner):Owner(owner) {}
    void ReleaseCommands()
    {
        if (Completion) Owner.DestroyFenceFunction(Owner.Device,Completion,nullptr); Completion=0;
        if (CommandsPool) Owner.DestroyCommandPoolFunction(Owner.Device,CommandsPool,nullptr); CommandsPool=0; Commands=nullptr;
    }
    ~LumenLightingSliceWork() { ReleaseCommands(); }
};
PFN_vkGetFenceStatus LightingGetFenceStatus=nullptr;
LumenLightingQueue<LumenLightingSliceWork> LumenLightingTasks;
bool CollectLumenLighting()
{ return LumenLightingTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenLighting(work,ready); }); }
bool CancelLumenLightingUpdate(LumenLightingTask task)
{ return (LumenLightingTasks.Cancel(task) || Fail("Invalid Lumen lighting task")) && CollectLumenLighting(); }
void DestroyLumenLightingAsync() { LumenLightingTasks.Clear(); }
bool ProbeLumenLighting(LumenLightingSliceWork& work,bool& ready)
{
    const auto status=LightingGetFenceStatus(Device,work.Completion); ready=status==VK_SUCCESS;
    return status==VK_NOT_READY || Check(status,"Poll lighting slice completion");
}
bool BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& output)
{
    if (!LoadDevice(LightingGetFenceStatus,"vkGetFenceStatus") || !CollectLumenLighting()) return false;
    const auto source=LumenSurfaces.find(surface.Value); const auto occluders=LumenDistanceScenes.find(distance.Value);
    if (source==LumenSurfaces.end() || occluders==LumenDistanceScenes.end()) return Fail("Lumen relighting source handles are invalid or expired");
    if (!EnsureLumenSurfaceLighting(bool(upload.PreviousLighting),bool(occluders->second->Hardware))) return false;
    auto job=std::make_shared<LumenLightingSliceWork>(*this);
    if (!SelectLumenPreviousLighting(upload,source->second,occluders->second,job->Previous)) return false;
    if (!job->Prepare(source->second->Identity,source->second->Layout,source->second->Captures,occluders->second->Info,upload,LastError) || !LumenLightingTasks.CanSubmit(*job,LastError)) return false;
    const auto& input=job->Input;
    const auto count=std::size_t(source->second->Layout.Width)*source->second->Layout.Height,bytes=count*sizeof(LumenLightingTexel),inputBytes=input.Words.size()*sizeof(std::uint32_t);
    if (std::max(bytes,inputBytes)>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan Lumen lighting exceeds storage range");
    const auto groups=LumenLightingGroups(input.Frame),x=std::min(groups,LumenLightingColumns),y=(groups+LumenLightingColumns-1)/LumenLightingColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan Lumen lighting exceeds dispatch limits");
    auto work=std::make_shared<LumenLightingWork>(*this); auto record=std::make_shared<LumenLightingRecord>(*this); work->Record=record; work->Previous=job->Previous; job->Record=record; job->Resources=work;
    record->Surface=source->second; record->Distance=occluders->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{source->second->Identity,occluders->second->Info.Identity,upload.LightingRevision},source->second->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (job->Previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=job->Previous->Info.Bounce+1; }
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(sizeof(input.Frame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,host,work->Uniform) || !CreateRawBuffer(inputBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Input) ||
        !CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,record->Buffer)) return false;
    std::memcpy(work->Uniform.Mapped,&input.Frame,sizeof(input.Frame)); std::memcpy(work->Input.Mapped,input.Words.data(),inputBytes);
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,LumenSurfacePlanes},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Lumen lighting descriptor pool")) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&LumenLightingSet;
    if (!Check(AllocateDescriptorSets(Device,&allocate,&work->Set),"Allocate Lumen lighting descriptors")) return false;
    const RawBuffer* buffers[]{&work->Uniform,&record->Distance->Scene,&work->Input,&record->Buffer,job->Previous?&job->Previous->Buffer:nullptr}; VkDescriptorBufferInfo bufferInfos[5]{}; VkDescriptorImageInfo imageInfos[LumenSurfacePlanes]{};
    for (unsigned i=0;i<(job->Previous?5u:4u);++i) bufferInfos[i]={buffers[i]->Buffer,0,buffers[i]->Size};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) imageInfos[plane]={LumenLightingSampler,record->Surface->Images[plane].View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[8]{}; const unsigned writeCount=job->Previous?8:7;
    for (unsigned slot=0;slot<writeCount;++slot)
    {
        auto& write=writes[slot]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=work->Set; write.dstBinding=slot; write.descriptorCount=1;
        write.descriptorType=slot==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:slot<=3?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (slot>=1 && slot<=3) write.pImageInfo=&imageInfos[slot-1]; else write.pBufferInfo=&bufferInfos[slot?slot-3:0];
    }
    UpdateDescriptorSets(Device,writeCount,writes,0,nullptr);
    const auto task=LumenLightingTasks.Insert(job); if (!task) return Fail("Lumen lighting task space exhausted"); output=task; return true;
}
bool SubmitLumenLightingSlice(LumenLightingSliceWork& work,const LumenSurfaceLightingFrame& frame)
{
    const auto groups=LumenLightingGroups(frame),x=std::min(groups,LumenLightingColumns),y=(groups+LumenLightingColumns-1)/LumenLightingColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Lighting slice exceeds Vulkan dispatch limits");
    work.ReleaseCommands(); auto& resources=*work.Resources; std::memcpy(resources.Uniform.Mapped,&frame,sizeof(frame));
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pool.queueFamilyIndex=QueueFamilyIndex;
    if (!Check(CreateCommandPoolFunction(Device,&pool,nullptr,&work.CommandsPool),"Create lighting slice command pool")) return false;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; allocate.commandPool=work.CommandsPool; allocate.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount=1;
    if (!Check(AllocateCommandBuffers(Device,&allocate,&work.Commands),"Allocate lighting slice commands")) return false;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; if (!Check(CreateFenceFunction(Device,&fence,nullptr,&work.Completion),"Create lighting slice fence")) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!Check(BeginCommandBuffer(work.Commands,&begin),"Begin lighting slice")) return false;
    const auto barrier=[&](const RawBuffer& buffer,VkFlags before,VkFlags after,VkFlags sourceStage,VkFlags targetStage)
    {
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; b.srcAccessMask=before; b.dstAccessMask=after;
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.buffer=buffer.Buffer; b.size=buffer.Size;
        CmdPipelineBarrier(work.Commands,sourceStage,targetStage,0,0,nullptr,1,&b,0,nullptr);
    };
    barrier(resources.Uniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!work.Initialized) barrier(resources.Input,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    else barrier(work.Record->Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CmdBindPipeline(work.Commands,VK_PIPELINE_BIND_POINT_COMPUTE,work.Record->Distance->Hardware?LumenHardwarePipelines[unsigned(work.Previous?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct)]:work.Previous?LumenSurfaceRadiosity:LumenSurfaceLighting);
    BindLumenHardware(work.Commands,LumenLightingLayout,*work.Record->Distance);
    CmdBindDescriptorSets(work.Commands,VK_PIPELINE_BIND_POINT_COMPUTE,LumenLightingLayout,0,1,&resources.Set,0,nullptr);
    LightingDispatch(work.Commands,x,y,1);
    barrier(work.Record->Buffer,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!Check(EndCommandBuffer(work.Commands),"End lighting slice")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&work.Commands;
    if (!Check(QueueSubmit(GraphicsQueue,1,&submit,work.Completion),"Submit lighting slice")) return false;
    work.InFlight=true; return true;
}
bool AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics)
{
    if (!CollectLumenLighting()) return false;
    return LumenLightingTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenLighting(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenLightingSlice(work,frame); },
        [&](auto& work,LumenSurfaceLightingResult& result)
        {
            result.Handle=NewLumenLightingHandle(); if (!result.Handle) return Fail("Lumen lighting handle space exhausted");
            result.Info=work.Record->Info; LumenLightingCaches.emplace(result.Handle.Value,work.Record); return true;
        });
}
