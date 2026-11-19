// Shared candidate pages are patched before commit; tracked blit hazards order
// retained source copies. Publication checks command status without a host wait.
struct NaniteCacheWork
{
    NaniteCachePlan Plan; std::shared_ptr<NaniteCachePool> Source,Pool;
    ObjCObject Completion=nullptr;
    ~NaniteCacheWork() { ReleaseObject(Completion); }
};
NaniteAsyncQueue<NaniteCacheWork,NaniteCacheTask> NaniteCacheTasks;
void DestroyNaniteCacheAsync()
{
    NaniteCacheTasks.Visit([](auto& work){if (work.Completion) Send<void>(work.Completion,"waitUntilCompleted"); return true;});
    NaniteCacheTasks.Clear();
}
bool ProbeNaniteCache(NaniteCacheWork& work,bool& ready)
{
    if (!work.Completion) { ready=true; return true; }
    const auto status=Send<NSUInteger>(work.Completion,"status");
    if (status==CommandBufferStatusError) return Fail("Metal Nanite page copy failed: "+ErrorDescription(Send<ObjCObject>(work.Completion,"error")));
    ready=status==NaniteCommandCompleted; return true;
}
bool CollectNaniteCache() { return NaniteCacheTasks.Collect([&](auto& work,bool& ready){return ProbeNaniteCache(work,ready);}); }
bool CancelNaniteCache(NaniteCacheTask task)
{ return (NaniteCacheTasks.Cancel(task) || Fail("Invalid Nanite page task")) && CollectNaniteCache(); }
bool SubmitNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheTask& output,NaniteCacheStatistics& statistics)
{
    const NaniteNativeCache* old=nullptr;
    if (previous) { auto found=NaniteCaches.find(previous.Value); if (found==NaniteCaches.end()) return Fail("Invalid Nanite source pool"); old=&found->second; }
    auto work=std::make_shared<NaniteCacheWork>();
    if (!CollectNaniteCache() || !BuildNaniteCachePlan(old?&old->Info:nullptr,upload,work->Plan,LastError) || !NaniteCacheTasks.CanSubmit(work->Plan,LastError)) return false;
    if (old) work->Source=old->Pool; const auto& plan=work->Plan;
    if (plan.ReusePool) work->Pool=work->Source;
    else
    {
        work->Pool=std::shared_ptr<NaniteCachePool>(new NaniteCachePool{},[](auto* pool){ReleaseObject(pool->Buffer); delete pool;});
        work->Pool->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",NSUInteger(plan.Statistics.AllocatedBytes),NSUInteger(0));
        if (!work->Pool->Buffer) return Fail("Allocate Metal async Nanite page pool failed");
        auto* mapped=Send<void*>(work->Pool->Buffer,"contents"); if (!mapped) return Fail("Metal Nanite page pool is not shared");
        for (auto slot:plan.UploadSlots) std::memcpy(static_cast<std::uint8_t*>(mapped)+std::size_t(slot)*NaniteGpuPageBytes,&upload.PhysicalPages[slot],NaniteGpuPageBytes);
        if (!plan.Copies.empty())
        {
            const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Create Metal Nanite page commands failed");
            work->Completion=Send<ObjCObject>(command,"retain");
            const auto encoder=Send<ObjCObject>(command,"blitCommandEncoder"); if (!encoder) return Fail("Create Metal Nanite page blit encoder failed");
            for (const auto& run:plan.Copies)
            {
                const auto offset=NSUInteger(std::uint64_t(run.FirstSlot)*NaniteGpuPageBytes),bytes=NSUInteger(std::uint64_t(run.SlotCount)*NaniteGpuPageBytes);
                Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(encoder,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",work->Source->Buffer,offset,work->Pool->Buffer,offset,bytes);
            }
            Send<void>(encoder,"endEncoding");
        }
    }
    const auto task=NaniteCacheTasks.Insert(work); if (work->Completion) Send<void>(work->Completion,"commit");
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
