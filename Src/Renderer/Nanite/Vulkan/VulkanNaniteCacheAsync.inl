// Each immutable pool update owns its command pool and fence. Host patches and
// unchanged-page transfer copies are made visible to subsequent shader/copy reads.
// A zero-wait fence probe gates publication; cancelled submissions retain resources.
struct NaniteCacheWork
{
    Impl& Owner; NaniteCachePlan Plan; std::shared_ptr<NaniteCachePool> Source,Pool;
    VkCommandPool CommandsPool=0; VkCommandBuffer Commands=nullptr; VkFence Completion=0;
    explicit NaniteCacheWork(Impl& owner):Owner(owner) {}
    ~NaniteCacheWork()
    {
        if (Completion) Owner.DestroyFenceFunction(Owner.Device,Completion,nullptr);
        if (CommandsPool) Owner.DestroyCommandPoolFunction(Owner.Device,CommandsPool,nullptr);
    }
};
NaniteAsyncQueue<NaniteCacheWork,NaniteCacheTask> NaniteCacheTasks;
void DestroyNaniteCacheAsync() { NaniteCacheTasks.Clear(); }
bool ProbeNaniteCache(NaniteCacheWork& work,bool& ready)
{
    if (!work.Completion) { ready=true; return true; }
    const auto status=NaniteGetFenceStatus(Device,work.Completion); ready=status==VK_SUCCESS;
    return status==VK_NOT_READY || Check(status,"Poll Nanite page fence");
}
bool CollectNaniteCache() { return NaniteCacheTasks.Collect([&](auto& work,bool& ready){return ProbeNaniteCache(work,ready);}); }
bool CancelNaniteCache(NaniteCacheTask task)
{ return (NaniteCacheTasks.Cancel(task) || Fail("Invalid Nanite page task")) && CollectNaniteCache(); }
bool SubmitNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheTask& output,NaniteCacheStatistics& statistics)
{
    if (!LoadDevice(NaniteGetFenceStatus,"vkGetFenceStatus") || !LoadDevice(NaniteCopyBuffer,"vkCmdCopyBuffer")) return false;
    const NaniteNativeCache* old=nullptr;
    if (previous) { auto found=NaniteCaches.find(previous.Value); if (found==NaniteCaches.end()) return Fail("Invalid Nanite source pool"); old=&found->second; }
    auto work=std::make_shared<NaniteCacheWork>(*this);
    if (!CollectNaniteCache() || !BuildNaniteCachePlan(old?&old->Info:nullptr,upload,work->Plan,LastError) || !NaniteCacheTasks.CanSubmit(work->Plan,LastError)) return false;
    if (!SupportsNaniteGeometry() || std::uint64_t(upload.Constants.Limits.Y)*sizeof(unsigned)>std::uint64_t(DeviceLimits.maxStorageBufferRange)) return Fail("Vulkan Nanite page pool exceeds storage limits");
    if (old) work->Source=old->Pool; const auto& plan=work->Plan;
    if (plan.ReusePool) work->Pool=work->Source;
    else
    {
        work->Pool=std::shared_ptr<NaniteCachePool>(new RawBuffer{},[this](auto* buffer){DestroyRawBuffer(*buffer); delete buffer;});
        if (!CreateRawBuffer(plan.Statistics.AllocatedBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,*work->Pool)) return false;
        for (auto slot:plan.UploadSlots) std::memcpy(static_cast<std::uint8_t*>(work->Pool->Mapped)+std::size_t(slot)*NaniteGpuPageBytes,&upload.PhysicalPages[slot],NaniteGpuPageBytes);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pool.queueFamilyIndex=QueueFamilyIndex;
        if (!Check(CreateCommandPoolFunction(Device,&pool,nullptr,&work->CommandsPool),"Create Nanite page command pool")) return false;
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; allocation.commandPool=work->CommandsPool; allocation.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocation.commandBufferCount=1;
        if (!Check(AllocateCommandBuffers(Device,&allocation,&work->Commands),"Allocate Nanite page commands")) return false;
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; if (!Check(CreateFenceFunction(Device,&fence,nullptr,&work->Completion),"Create Nanite page fence")) return false;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!Check(BeginCommandBuffer(work->Commands,&begin),"Begin Nanite page commands")) return false;
        if (!plan.Copies.empty())
        {
            VkBufferMemoryBarrier source{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; source.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_HOST_WRITE_BIT; source.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            source.srcQueueFamilyIndex=source.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; source.buffer=work->Source->Buffer; source.size=work->Source->Size;
            CmdPipelineBarrier(work->Commands,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&source,0,nullptr);
            std::vector<VkBufferCopy> copies;
            for (const auto& run:plan.Copies) { const auto offset=std::uint64_t(run.FirstSlot)*NaniteGpuPageBytes; copies.push_back({offset,offset,std::uint64_t(run.SlotCount)*NaniteGpuPageBytes}); }
            NaniteCopyBuffer(work->Commands,work->Source->Buffer,work->Pool->Buffer,unsigned(copies.size()),copies.data());
        }
        VkBufferMemoryBarrier visible{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; visible.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
        visible.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT;
        visible.srcQueueFamilyIndex=visible.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; visible.buffer=work->Pool->Buffer; visible.size=work->Pool->Size;
        CmdPipelineBarrier(work->Commands,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&visible,0,nullptr);
        if (!Check(EndCommandBuffer(work->Commands),"End Nanite page commands")) return false;
    }
    const auto task=NaniteCacheTasks.Insert(work);
    if (work->Commands)
    {
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&work->Commands;
        if (!Check(QueueSubmit(GraphicsQueue,1,&submit,work->Completion),"Submit Nanite page copies")) { NaniteCacheTasks.Cancel(task); return false; }
    }
    output=task; statistics=plan.Statistics; return true;
}
bool PollNaniteCache(NaniteCacheTask task,bool& ready,NaniteCacheHandle& output)
{
    ready=false; auto work=NaniteCacheTasks.Find(task); if (!work) return Fail("Invalid Nanite page task");
    bool complete=false; if (!ProbeNaniteCache(*work,complete)) return false; if (!complete) return true;
    const auto handle=NewNaniteCacheHandle(); if (!handle) return Fail("Nanite cache handle space is exhausted");
    NaniteCaches.emplace(handle.Value,NaniteNativeCache{work->Plan.Next,work->Pool});
    NaniteCacheTasks.Erase(task); output=handle; ready=true; return true;
}
