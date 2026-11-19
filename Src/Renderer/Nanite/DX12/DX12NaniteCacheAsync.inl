// Private command resources and a queue fence retire immutable page-copy jobs.
// Buffers decay to COMMON after each direct-queue submission. The old pool is
// read-only and remains retained if its public handle is released mid-copy.
struct NaniteCacheWork
{
    NaniteCachePlan Plan; std::shared_ptr<NaniteCachePool> Source,Pool;
    ComPtr<ID3D12Resource> Patches;
    ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands;
    ComPtr<ID3D12Fence> Completion;
};
NaniteAsyncQueue<NaniteCacheWork,NaniteCacheTask> NaniteCacheTasks;
void DestroyNaniteCacheAsync() { NaniteCacheTasks.Clear(); }
bool ProbeNaniteCache(NaniteCacheWork& work,bool& ready)
{
    if (!work.Completion) { ready=true; return true; }
    const auto value=work.Completion->GetCompletedValue();
    if (value==UINT64_MAX) return FailHRESULT("Poll Nanite page device removal",Device->GetDeviceRemovedReason());
    ready=value>=1; return true;
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
    work->Pool=plan.ReusePool?work->Source:std::make_shared<NaniteCachePool>();
    if (!plan.ReusePool)
    {
        if (!CreateBufferResource(std::size_t(plan.Statistics.AllocatedBytes),D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,work->Pool->Resource) ||
            !CreateBufferResource(std::size_t(plan.Statistics.UploadedBytes),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work->Patches)) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; auto status=work->Patches->Map(0,&noRead,&mapped);
        if (FAILED(status)) return FailHRESULT("Map async Nanite page patches",status);
        for (std::size_t patch=0;patch<plan.UploadSlots.size();++patch)
            std::memcpy(static_cast<std::uint8_t*>(mapped)+patch*NaniteGpuPageBytes,&upload.PhysicalPages[plan.UploadSlots[patch]],NaniteGpuPageBytes);
        work->Patches->Unmap(0,nullptr);
        status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work->Allocator.GetAddressOf()));
        if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work->Allocator.Get(),nullptr,IID_PPV_ARGS(work->Commands.GetAddressOf()));
        if (SUCCEEDED(status)) status=Device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(work->Completion.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create async Nanite page command resources",status);
        for (const auto& run:plan.Copies)
        {
            const auto offset=std::uint64_t(run.FirstSlot)*NaniteGpuPageBytes;
            work->Commands->CopyBufferRegion(work->Pool->Resource.Get(),offset,work->Source->Resource.Get(),offset,std::uint64_t(run.SlotCount)*NaniteGpuPageBytes);
        }
        for (std::size_t patch=0;patch<plan.UploadSlots.size();++patch)
            work->Commands->CopyBufferRegion(work->Pool->Resource.Get(),std::uint64_t(plan.UploadSlots[patch])*NaniteGpuPageBytes,work->Patches.Get(),patch*NaniteGpuPageBytes,NaniteGpuPageBytes);
        status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close async Nanite page commands",status);
    }
    const auto task=NaniteCacheTasks.Insert(work);
    if (work->Commands)
    {
        ID3D12CommandList* commands[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,commands);
        const auto status=CommandQueue->Signal(work->Completion.Get(),1);
        if (FAILED(status)) { NaniteCacheTasks.Cancel(task); return FailHRESULT("Signal Nanite page completion",status); }
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
