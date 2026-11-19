struct LumenGlobalWork : LumenGlobalDistanceJob
{
    Impl& Owner; std::shared_ptr<LumenDistanceRecord> Record,Previous;
    RawBuffer Uniform,Dirty; VkDescriptorPool Descriptors=0; VkDescriptorSet Set=0;
    VkCommandPool CommandsPool=0; VkCommandBuffer Commands=nullptr; VkFence Completion=0;
    explicit LumenGlobalWork(Impl& owner):Owner(owner) {}
    void ReleaseCommands()
    {
        if (Completion) Owner.DestroyFenceFunction(Owner.Device,Completion,nullptr); Completion=0;
        if (CommandsPool) Owner.DestroyCommandPoolFunction(Owner.Device,CommandsPool,nullptr); CommandsPool=0; Commands=nullptr;
    }
    ~LumenGlobalWork()
    {
        ReleaseCommands();
        if (Descriptors) Owner.DestroyDescriptorPoolFunction(Owner.Device,Descriptors,nullptr);
        Owner.DestroyRawBuffer(Uniform); Owner.DestroyRawBuffer(Dirty);
    }
};
LumenGlobalDistanceQueue<LumenGlobalWork> LumenGlobalTasks;
PFN_vkGetFenceStatus GlobalGetFenceStatus=nullptr;
void DestroyLumenGlobalDistanceAsync() { LumenGlobalTasks.Clear(); }
bool ProbeLumenGlobalDistance(LumenGlobalWork& work,bool& ready)
{
    const auto status=GlobalGetFenceStatus(Device,work.Completion); ready=status==VK_SUCCESS;
    return status==VK_NOT_READY || Check(status,"Poll global distance completion");
}
bool CollectLumenGlobalDistance()
{ return LumenGlobalTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenGlobalDistance(work,ready); }); }
bool CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{ return (LumenGlobalTasks.Cancel(task) || Fail("Invalid global distance task")) && CollectLumenGlobalDistance(); }
bool BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& output)
{
    if (!LoadDevice(GlobalGetFenceStatus,"vkGetFenceStatus") || !CollectLumenGlobalDistance() || !EnsureLumenGlobalDistance()) return false;
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source");
    auto work=std::make_shared<LumenGlobalWork>(*this);
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second) return Fail("Global distance reuse crosses source generations");
        work->Previous=old->second;
    }
    if (!work->Prepare(found->second->Info,settings,work->Previous?work->Previous->Info.Global.get():nullptr,LastError) || !LumenGlobalTasks.CanSubmit(*work,LastError)) return false;
    const auto bytes=std::size_t(work->Plan->WordCount)*sizeof(std::uint32_t);
    if (bytes>DeviceLimits.maxStorageBufferRange) return Fail("Global distance scene exceeds Vulkan storage range");
    work->Record=std::make_shared<LumenDistanceRecord>(*this); auto& record=*work->Record;
    record.Base=found->second; record.Info=record.Base->Info; record.Info.Global=work->Plan;
    constexpr VkFlags host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,record.Scene) ||
        !CreateRawBuffer(sizeof(LumenGlobalDistanceFrame),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,host,work->Uniform) ||
        !CreateRawBuffer(std::max<std::size_t>(1,work->Update.Compose.size())*sizeof(std::uint32_t),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,host,work->Dirty)) return false;
    if (!work->Update.Compose.empty()) std::memcpy(work->Dirty.Mapped,work->Update.Compose.data(),work->Update.Compose.size()*sizeof(std::uint32_t));
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Descriptors),"Create global slice descriptors")) return false;
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=work->Descriptors; allocation.descriptorSetCount=1; allocation.pSetLayouts=&LumenGlobalSet;
    if (!Check(AllocateDescriptorSets(Device,&allocation,&work->Set),"Allocate global slice descriptors")) return false;
    const RawBuffer* buffers[]{&work->Uniform,&record.Base->Scene,&work->Dirty,&record.Scene}; VkDescriptorBufferInfo infos[4]{}; VkWriteDescriptorSet writes[4]{};
    for (unsigned i=0;i<4;++i)
    {
        infos[i]={buffers[i]->Buffer,0,buffers[i]->Size}; auto& write=writes[i]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet=work->Set; write.dstBinding=i; write.descriptorCount=1; write.descriptorType=i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&infos[i];
    }
    UpdateDescriptorSets(Device,4,writes,0,nullptr);
    const auto task=LumenGlobalTasks.Insert(work); if (!task) return Fail("Global task space exhausted"); output=task; return true;
}
bool SubmitLumenGlobalDistanceSlice(LumenGlobalWork& work,const LumenGlobalDistanceFrame& frame)
{
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    const unsigned x=std::min(groups,LumenDistanceTraceColumns),y=(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns;
    if (x>DeviceLimits.maxComputeWorkGroupCount[0] || y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Global slice exceeds Vulkan dispatch limits");
    // A prior slice's fence has retired before reusing the mapped constants.
    work.ReleaseCommands(); std::memcpy(work.Uniform.Mapped,&frame,sizeof(frame));
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pool.queueFamilyIndex=QueueFamilyIndex;
    if (!Check(CreateCommandPoolFunction(Device,&pool,nullptr,&work.CommandsPool),"Create global slice command pool")) return false;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; allocate.commandPool=work.CommandsPool; allocate.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount=1;
    if (!Check(AllocateCommandBuffers(Device,&allocate,&work.Commands),"Allocate global slice commands")) return false;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; if (!Check(CreateFenceFunction(Device,&fence,nullptr,&work.Completion),"Create global slice fence")) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!Check(BeginCommandBuffer(work.Commands,&begin),"Begin global slice")) return false;
    const auto barrier=[&](const RawBuffer& buffer,VkFlags before,VkFlags after,VkFlags sourceStage,VkFlags targetStage)
    {
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; b.srcAccessMask=before; b.dstAccessMask=after;
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.buffer=buffer.Buffer; b.size=buffer.Size;
        CmdPipelineBarrier(work.Commands,sourceStage,targetStage,0,0,nullptr,1,&b,0,nullptr);
    };
    auto& record=*work.Record;
    barrier(work.Uniform,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT);
    if (!work.Initialized)
    {
        barrier(work.Dirty,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        barrier(record.Base->Scene,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferCopy prefix{0,0,std::max(1u,record.Info.Counts.W)*sizeof(std::uint32_t)};
        DistanceCopyBuffer(work.Commands,record.Base->Scene.Buffer,record.Scene.Buffer,1,&prefix);
        const VkBufferCopy header{offsetof(LumenGlobalDistanceFrame,Header),std::size_t(frame.Header.Layout.W)*sizeof(std::uint32_t),sizeof(LumenGlobalDistanceHeader)};
        DistanceCopyBuffer(work.Commands,work.Uniform.Buffer,record.Scene.Buffer,1,&header);
        barrier(record.Base->Scene,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        if (!work.Update.Copies.empty())
        {
            barrier(work.Previous->Scene,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
            std::vector<VkBufferCopy> ranges; ranges.reserve(work.Update.Copies.size());
            for (const auto& copy:work.Update.Copies) ranges.push_back({std::size_t(copy.SourceWord)*sizeof(std::uint32_t),
                std::size_t(copy.DestinationWord)*sizeof(std::uint32_t),std::size_t(copy.WordCount)*sizeof(std::uint32_t)});
            DistanceCopyBuffer(work.Commands,work.Previous->Scene.Buffer,record.Scene.Buffer,unsigned(ranges.size()),ranges.data());
            barrier(work.Previous->Scene,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
    }
    barrier(record.Scene,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    CmdBindPipeline(work.Commands,VK_PIPELINE_BIND_POINT_COMPUTE,LumenGlobalCompose);
    CmdBindDescriptorSets(work.Commands,VK_PIPELINE_BIND_POINT_COMPUTE,LumenGlobalLayout,0,1,&work.Set,0,nullptr);
    if (groups) DistanceDispatch(work.Commands,x,y,1);
    barrier(record.Scene,VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!Check(EndCommandBuffer(work.Commands),"End global slice")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&work.Commands;
    if (!Check(QueueSubmit(GraphicsQueue,1,&submit,work.Completion),"Submit global distance slice")) return false;
    work.InFlight=true; return true;
}
bool AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    if (!CollectLumenGlobalDistance()) return false;
    return LumenGlobalTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenGlobalDistance(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenGlobalDistanceSlice(work,frame); },
        [&](auto& work,LumenDistanceSceneHandle& handle)
        {
            handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Global scene handle space exhausted");
            LumenDistanceScenes.emplace(handle.Value,work.Record); return true;
        });
}
