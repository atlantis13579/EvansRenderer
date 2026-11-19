// Ordered page copies and updates end in an event query, without a staging read.
// Each task owns both pool versions until completion, including after cancellation.
struct NaniteCacheWork
{
    NaniteCachePlan Plan; std::shared_ptr<NaniteCachePool> Source,Pool;
    ComPtr<ID3D11Query> Completion;
};
NaniteAsyncQueue<NaniteCacheWork,NaniteCacheTask> NaniteCacheTasks;
void DestroyNaniteCacheAsync() { NaniteCacheTasks.Clear(); }
bool ProbeNaniteCache(NaniteCacheWork& work,bool& ready)
{
    if (!work.Completion) { ready=true; return true; }
    BOOL complete=FALSE; const auto status=Context->GetData(work.Completion.Get(),&complete,sizeof(complete),D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (FAILED(status)) return Fail(HResultMessage("Poll Nanite page copy",status));
    ready=status==S_OK && complete; return true;
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
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(plan.Statistics.AllocatedBytes); desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        auto status=Device->CreateBuffer(&desc,nullptr,work->Pool->Buffer.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Allocate async Nanite pool",status));
        D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32_TYPELESS; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;
        view.BufferEx.NumElements=desc.ByteWidth/sizeof(unsigned); view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
        status=Device->CreateShaderResourceView(work->Pool->Buffer.Get(),&view,work->Pool->View.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create async Nanite pool view",status));
        const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0}; status=Device->CreateQuery(&query,work->Completion.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Nanite page completion query",status));
        for (const auto& run:plan.Copies)
        {
            const D3D11_BOX region{run.FirstSlot*NaniteGpuPageBytes,0,0,(run.FirstSlot+run.SlotCount)*NaniteGpuPageBytes,1,1};
            Context->CopySubresourceRegion(work->Pool->Buffer.Get(),0,region.left,0,0,work->Source->Buffer.Get(),0,&region);
        }
        for (auto slot:plan.UploadSlots)
        {
            const D3D11_BOX region{slot*NaniteGpuPageBytes,0,0,(slot+1)*NaniteGpuPageBytes,1,1};
            Context->UpdateSubresource(work->Pool->Buffer.Get(),0,&region,&upload.PhysicalPages[slot],0,0);
        }
        Context->End(work->Completion.Get()); Context->Flush();
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
