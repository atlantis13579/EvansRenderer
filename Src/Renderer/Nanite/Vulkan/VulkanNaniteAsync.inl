// Each job owns its pool, descriptor sets, uniform slices and completion fence.
// Coherent host mappings are read only after vkGetFenceStatus succeeds. Pending
// command buffers are never reset by another frame or a cancelled producer.
PFN_vkGetFenceStatus NaniteGetFenceStatus=nullptr;
struct NaniteAsyncWork
{
    Impl& Owner; NaniteGpuUpload Upload; NaniteAsyncPlan Plan;
    std::array<RawBuffer,7> Storage{}; std::shared_ptr<NaniteCachePool> Borrowed;
    VkCommandPool CommandsPool=0; VkCommandBuffer Commands=nullptr; VkDescriptorPool Descriptors=0; VkFence Completion=0;
    explicit NaniteAsyncWork(Impl& owner):Owner(owner) {}
    ~NaniteAsyncWork()
    {
        if (Completion) Owner.DestroyFenceFunction(Owner.Device,Completion,nullptr);
        if (CommandsPool) Owner.DestroyCommandPoolFunction(Owner.Device,CommandsPool,nullptr);
        if (Descriptors) Owner.DestroyDescriptorPoolFunction(Owner.Device,Descriptors,nullptr);
        for (unsigned i=0;i<Storage.size();++i) if (i!=3 || !Borrowed) Owner.DestroyRawBuffer(Storage[i]);
    }
};
NaniteAsyncQueue<NaniteAsyncWork> NaniteAsyncTasks;
void DestroyNaniteAsync() { NaniteAsyncTasks.Clear(); }
bool ProbeNaniteAsync(NaniteAsyncWork& work,bool& ready)
{
    const auto status=NaniteGetFenceStatus(Device,work.Completion); ready=status==VK_SUCCESS;
    return status==VK_NOT_READY || Check(status,"Poll Nanite completion fence");
}
bool CollectNaniteAsync() { return NaniteAsyncTasks.Collect([&](auto& work,bool& ready){return ProbeNaniteAsync(work,ready);}); }
bool CancelNaniteGeometry(NaniteGeometryTask task)
{ return (NaniteAsyncTasks.Cancel(task) || Fail("Invalid Nanite task")) && CollectNaniteAsync(); }
bool SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& output)
{
    if (!LoadDevice(NaniteGetFenceStatus,"vkGetFenceStatus")) return false;
    auto work=std::make_shared<NaniteAsyncWork>(*this);
    if (!CollectNaniteAsync() || !BuildNaniteAsyncPlan(upload,work->Plan,LastError) || !NaniteAsyncTasks.CanSubmit(work->Plan,LastError) || !EnsureNaniteGeometry()) return false;
    if (cache) { auto found=NaniteCaches.find(cache.Value); if (found==NaniteCaches.end()) return Fail("Invalid Nanite cache"); if (!MatchNaniteCache(found->second.Info,upload,LastError)) return false; work->Borrowed=found->second.Pool; }
    for (const auto& job:work->Plan.Dispatches) if (job.GroupsX>DeviceLimits.maxComputeWorkGroupCount[0] || job.GroupsY>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Nanite async dispatch exceeds Vulkan limits");
    work->Upload=upload; const unsigned count=unsigned(work->Plan.Dispatches.size());
    const auto alignment=std::max<VkDeviceSize>(1,DeviceLimits.minUniformBufferOffsetAlignment);
    const auto stride=(sizeof(NaniteGpuConstants)+alignment-1)/alignment*alignment;
    work->Plan.ReservedBytes+=std::size_t(stride*count);
    if (!NaniteAsyncTasks.CanSubmit(work->Plan,LastError)) return false;
    const void* data[]{nullptr,upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data(),nullptr,nullptr};
    const std::size_t sizes[]{std::size_t(stride*count),upload.Metadata.size()*sizeof(unsigned),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage),upload.InitialState.size()*sizeof(unsigned),work->Plan.VertexBytes,work->Plan.IndexBytes};
    const auto host=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (unsigned i=0;i<work->Storage.size();++i)
    {
        if (i==3 && work->Borrowed) { work->Storage[i]=*work->Borrowed; continue; }
        if (i && sizes[i]>DeviceLimits.maxStorageBufferRange) return Fail("Nanite async storage exceeds Vulkan limits");
        VkFlags usage=i?VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (i==NaniteVertices) usage|=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; if (i==NaniteIndices) usage|=VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        const auto memory=i>=NaniteVertices && !upload.ReadbackGeometry?VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:host;
        if (!CreateRawBuffer(sizes[i],usage,memory,work->Storage[i])) return false;
        if (data[i]) std::memcpy(work->Storage[i].Mapped,data[i],sizes[i]);
    }
    const VkDescriptorPoolSize poolSizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,count},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,count*unsigned(NaniteStorageCount)}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=count; pool.poolSizeCount=2; pool.pPoolSizes=poolSizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Descriptors),"Create async Nanite descriptors")) return false;
    std::vector<VkDescriptorSetLayout> layouts(count,NaniteGeometrySetLayout); std::vector<VkDescriptorSet> sets(count);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Descriptors; allocate.descriptorSetCount=count; allocate.pSetLayouts=layouts.data();
    if (!Check(AllocateDescriptorSets(Device,&allocate,sets.data()),"Allocate async Nanite descriptor sets")) return false;
    for (unsigned stage=0;stage<count;++stage)
    {
        std::memcpy(static_cast<std::uint8_t*>(work->Storage[0].Mapped)+stage*stride,&work->Plan.Dispatches[stage].Constants,sizeof(NaniteGpuConstants));
        std::array<VkDescriptorBufferInfo,7> buffers{}; std::array<VkWriteDescriptorSet,7> writes{};
        for (unsigned i=0;i<writes.size();++i)
        {
            buffers[i]={work->Storage[i].Buffer,i?0:stage*stride,i?work->Storage[i].Size:sizeof(NaniteGpuConstants)};
            auto& write=writes[i]; write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=sets[stage]; write.dstBinding=i; write.descriptorCount=1;
            write.descriptorType=i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&buffers[i];
        }
        UpdateDescriptorSets(Device,unsigned(writes.size()),writes.data(),0,nullptr);
    }
    VkCommandPoolCreateInfo commandPool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; commandPool.queueFamilyIndex=QueueFamilyIndex;
    if (!Check(CreateCommandPoolFunction(Device,&commandPool,nullptr,&work->CommandsPool),"Create async Nanite command pool")) return false;
    VkCommandBufferAllocateInfo commands{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; commands.commandPool=work->CommandsPool; commands.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; commands.commandBufferCount=1;
    if (!Check(AllocateCommandBuffers(Device,&commands,&work->Commands),"Allocate async Nanite commands")) return false;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; if (!Check(CreateFenceFunction(Device,&fence,nullptr,&work->Completion),"Create async Nanite fence")) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!Check(BeginCommandBuffer(work->Commands,&begin),"Begin async Nanite commands")) return false;
    std::array<VkBufferMemoryBarrier,7> hostWrites{};
    for (unsigned i=0;i<hostWrites.size();++i)
    {
        auto& b=hostWrites[i]; b.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; b.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_UNIFORM_READ_BIT;
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.buffer=work->Storage[i].Buffer; b.size=work->Storage[i].Size;
    }
    CmdPipelineBarrier(work->Commands,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,unsigned(hostWrites.size()),hostWrites.data(),0,nullptr);
    for (unsigned stage=0;stage<count;++stage)
    {
        const auto& job=work->Plan.Dispatches[stage]; CmdBindDescriptorSets(work->Commands,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteGeometryLayout,0,1,&sets[stage],0,nullptr);
        CmdBindPipeline(work->Commands,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteGeometryPipelines[unsigned(job.Pass)]); NaniteGeometryDispatch(work->Commands,job.GroupsX,job.GroupsY,1);
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.buffer=work->Storage[NaniteState].Buffer; b.size=sizes[NaniteState];
        CmdPipelineBarrier(work->Commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&b,0,nullptr);
    }
    std::array<VkBufferMemoryBarrier,3> visible{};
    for (unsigned i=0;i<visible.size();++i)
    {
        auto& b=visible[i]; b.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask=((i==0 || upload.ReadbackGeometry)?VK_ACCESS_HOST_READ_BIT:0)|VK_ACCESS_SHADER_READ_BIT|(i==1?VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT:i==2?VK_ACCESS_INDEX_READ_BIT:0);
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.buffer=work->Storage[NaniteState+i].Buffer; b.size=sizes[NaniteState+i];
    }
    CmdPipelineBarrier(work->Commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,unsigned(visible.size()),visible.data(),0,nullptr);
    if (!Check(EndCommandBuffer(work->Commands),"End async Nanite commands")) return false;
    const auto task=NaniteAsyncTasks.Insert(work); VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&work->Commands;
    if (!Check(QueueSubmit(GraphicsQueue,1,&submit,work->Completion),"Submit async Nanite commands")) { NaniteAsyncTasks.Cancel(task); return false; }
    output=task; return true;
}
bool PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    ready=false; auto work=NaniteAsyncTasks.Find(task); if (!work) return Fail("Invalid Nanite task");
    bool complete=false; if (!ProbeNaniteAsync(*work,complete)) return false; if (!complete) return true;
    NaniteGeometryBuffers result; result.Readback.State.resize(work->Upload.InitialState.size());
    std::memcpy(result.Readback.State.data(),work->Storage[NaniteState].Mapped,result.Readback.State.size()*sizeof(unsigned));
    if (!ValidateNaniteGpuState(work->Upload,result.Readback.State,result.Selection,LastError)) return false;
    const auto vertexBytes=std::size_t(result.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(result.Selection.IndexCount)*sizeof(unsigned);
    if (work->Upload.ReadbackGeometry)
    {
        result.Readback.Vertices.resize(result.Selection.VertexCount); result.Readback.Indices.resize(result.Selection.IndexCount);
        std::memcpy(result.Readback.Vertices.data(),work->Storage[NaniteVertices].Mapped,vertexBytes); std::memcpy(result.Readback.Indices.data(),work->Storage[NaniteIndices].Mapped,indexBytes);
        if (!ValidateNaniteGpuReadback(work->Upload,result.Readback,result.Selection,LastError)) return false;
    }
    std::array<BufferResource,2> resources; const void* mirrors[]{result.Readback.Vertices.data(),result.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite async vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite async indices"};
    for (unsigned i=0;i<2;++i)
    { auto& resource=resources[i]; resource.NaniteGeometry=true; resource.Allocation=work->Storage[NaniteVertices+i]; if (work->Upload.ReadbackGeometry) { resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[i],resource.Bytes.size()); } }
    result.VertexBuffer={NextBuffer++}; result.IndexBuffer={NextBuffer++}; result.DispatchCount=unsigned(work->Plan.Dispatches.size());
    try { Buffers.emplace(result.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(result.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(result.VertexBuffer.Value); throw; }
    work->Storage[NaniteVertices]={}; work->Storage[NaniteIndices]={}; NaniteAsyncTasks.Erase(task); output=std::move(result); ready=true; return true;
}
