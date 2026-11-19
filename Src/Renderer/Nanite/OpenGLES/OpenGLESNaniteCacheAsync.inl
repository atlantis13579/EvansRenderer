// Ordered immutable pool copies use an ES fence instead of completion mapping.
// Restore generic bindings on submission; retain native pools until GPU retirement.
struct NaniteCacheWork
{
    Impl& Owner; NaniteCachePlan Plan; std::shared_ptr<NaniteCachePool> Source,Pool;
    void* Completion=nullptr;
    explicit NaniteCacheWork(Impl& owner):Owner(owner) {}
    ~NaniteCacheWork() { if (Completion) Owner.NaniteDeleteSync(Completion); }
};
NaniteAsyncQueue<NaniteCacheWork,NaniteCacheTask> NaniteCacheTasks;
void DestroyNaniteCacheAsync() { NaniteCacheTasks.Clear(); }
bool ProbeNaniteCache(NaniteCacheWork& work,bool& ready)
{
    if (!work.Completion) { ready=true; return true; }
    const auto status=NaniteClientWaitSync(work.Completion,0,0);
    ready=status==NaniteAlreadySignaled || status==NaniteConditionSatisfied;
    return ready || status==NaniteTimeoutExpired || Fail("ES Nanite page completion query failed");
}
bool CollectNaniteCache() { return NaniteCacheTasks.Collect([&](auto& work,bool& ready){return ProbeNaniteCache(work,ready);}); }
bool CancelNaniteCache(NaniteCacheTask task)
{ return (NaniteCacheTasks.Cancel(task) || Fail("Invalid Nanite page task")) && CollectNaniteCache(); }
bool SubmitNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheTask& output,NaniteCacheStatistics& statistics)
{
    if (!LoadGL(NaniteFenceSync,"glFenceSync") || !LoadGL(NaniteClientWaitSync,"glClientWaitSync") || !LoadGL(NaniteDeleteSync,"glDeleteSync") ||
        !LoadGL(NaniteFlush,"glFlush") || !LoadGL(NaniteCopyBuffer,"glCopyBufferSubData")) return false;
    const NaniteNativeCache* old=nullptr;
    if (previous) { auto found=NaniteCaches.find(previous.Value); if (found==NaniteCaches.end()) return Fail("Invalid Nanite source pool"); old=&found->second; }
    auto work=std::make_shared<NaniteCacheWork>(*this);
    if (!CollectNaniteCache() || !BuildNaniteCachePlan(old?&old->Info:nullptr,upload,work->Plan,LastError) || !NaniteCacheTasks.CanSubmit(work->Plan,LastError)) return false;
    if (!NaniteGeometrySupported || std::uint64_t(upload.Constants.Limits.Y)*sizeof(unsigned)>std::uint64_t(NaniteMaximumBlock)) return Fail("ES Nanite page pool exceeds storage limits");
    if (old) work->Source=old->Pool; const auto& plan=work->Plan;
    if (plan.ReusePool) work->Pool=work->Source;
    else
    {
        struct Bindings
        {
            Impl& Owner; GLint Read=0,Write=0;
            explicit Bindings(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(NaniteCopyRead,&Read); Owner.GlGetIntegerv(NaniteCopyWrite,&Write); }
            ~Bindings() { Owner.GlBindBuffer(NaniteCopyRead,GLuint(Read)); Owner.GlBindBuffer(NaniteCopyWrite,GLuint(Write)); }
        } bindings(*this);
        work->Pool=std::shared_ptr<NaniteCachePool>(new NaniteCachePool{},[this](auto* pool){if (pool->Buffer) GlDeleteBuffers(1,&pool->Buffer); delete pool;});
        GlGenBuffers(1,&work->Pool->Buffer); GlBindBuffer(NaniteCopyWrite,work->Pool->Buffer);
        GlBufferData(NaniteCopyWrite,GLsizeiptr(plan.Statistics.AllocatedBytes),nullptr,GL_STATIC_DRAW);
        if (!CheckGL("Allocate async Nanite page pool")) return false;
        for (auto slot:plan.UploadSlots) GlBufferSubData(NaniteCopyWrite,GLintptr(std::uint64_t(slot)*NaniteGpuPageBytes),NaniteGpuPageBytes,&upload.PhysicalPages[slot]);
        if (!plan.Copies.empty())
        {
            GlBindBuffer(NaniteCopyRead,work->Source->Buffer);
            for (const auto& run:plan.Copies)
            {
                const auto offset=GLintptr(std::uint64_t(run.FirstSlot)*NaniteGpuPageBytes);
                NaniteCopyBuffer(NaniteCopyRead,NaniteCopyWrite,offset,offset,GLsizeiptr(std::uint64_t(run.SlotCount)*NaniteGpuPageBytes));
            }
        }
        GpuMemoryBarrier(GpuStorageBarrier|NaniteBufferBarrier);
        if (!CheckGL("Submit async Nanite page copies")) return false;
        work->Completion=NaniteFenceSync(NaniteSyncGpuCommandsComplete,0);
        if (!work->Completion) return Fail("Create ES Nanite page fence failed"); NaniteFlush();
    }
    const auto task=NaniteCacheTasks.Insert(work); output=task; statistics=plan.Statistics; return true;
}
bool PollNaniteCache(NaniteCacheTask task,bool& ready,NaniteCacheHandle& output)
{
    ready=false; auto work=NaniteCacheTasks.Find(task); if (!work) return Fail("Invalid Nanite page task");
    bool complete=false; if (!ProbeNaniteCache(*work,complete)) return false; if (!complete) return true;
    const auto handle=NewNaniteCacheHandle(); if (!handle) return Fail("Nanite cache handle space is exhausted");
    NaniteCaches.emplace(handle.Value,NaniteNativeCache{work->Plan.Next,work->Pool});
    NaniteCacheTasks.Erase(task); output=handle; ready=true; return true;
}
