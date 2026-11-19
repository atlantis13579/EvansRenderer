// Native pool versions are immutable. Copy unchanged resident runs on the direct
// queue and stage changed pages only. Pending copy resources survive a failed
// fence wait. Recovery precedes scratch reuse; retained references protect
// source pools even when their public handles have already been released.
struct NaniteCachePool { ComPtr<ID3D12Resource> Resource; };
struct NaniteNativeCache { NaniteCacheInfo Info; std::shared_ptr<NaniteCachePool> Pool; };
std::unordered_map<std::uint64_t,NaniteNativeCache> NaniteCaches;
std::shared_ptr<NaniteCachePool> NanitePendingCache,NanitePendingCacheSource;
ComPtr<ID3D12Resource> NaniteCacheUpload;
void DestroyNaniteCaches()
{ NaniteCaches.clear(); NanitePendingCache.reset(); NanitePendingCacheSource.reset(); NaniteCacheUpload.Reset(); }
bool DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!handle || !NaniteCaches.erase(handle.Value)) return Fail("DX12 Nanite cache handle is invalid or belongs to another RHI");
    return true;
}
bool PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& output,NaniteCacheStatistics& statistics)
{
    const NaniteNativeCache* old=nullptr;
    if (previous) { const auto found=NaniteCaches.find(previous.Value); if (found==NaniteCaches.end()) return Fail("DX12 previous Nanite cache handle is invalid"); old=&found->second; }
    NaniteCachePlan plan; if (!BuildNaniteCachePlan(old?&old->Info:nullptr,upload,plan,LastError)) return false;
    if (!RecoverSubmittedWork("Prepare Nanite cache") || !WaitForGPU() || !EnsureNaniteGeometry()) return false;
    NanitePendingCache.reset(); NanitePendingCacheSource.reset(); NaniteCacheUpload.Reset();
    auto pool=plan.ReusePool?old->Pool:std::make_shared<NaniteCachePool>();
    if (!plan.ReusePool)
    {
        if (!CreateBufferResource(static_cast<std::size_t>(plan.Statistics.AllocatedBytes),D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,pool->Resource) ||
            !CreateBufferResource(static_cast<std::size_t>(plan.Statistics.UploadedBytes),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,NaniteCacheUpload)) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; auto result=NaniteCacheUpload->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map Nanite cache patches",result);
        for (std::size_t patch=0;patch<plan.UploadSlots.size();++patch) std::memcpy(static_cast<std::uint8_t*>(mapped)+patch*NaniteGpuPageBytes,&upload.PhysicalPages[plan.UploadSlots[patch]],NaniteGpuPageBytes);
        NaniteCacheUpload->Unmap(0,nullptr); NanitePendingCache=pool; NanitePendingCacheSource=old?old->Pool:nullptr;
        result=NaniteGeometryAllocator->Reset(); if (SUCCEEDED(result)) result=NaniteGeometryCommands->Reset(NaniteGeometryAllocator.Get(),nullptr);
        if (FAILED(result)) return FailHRESULT("Reset Nanite cache recording",result);
        for (const auto& run:plan.Copies)
        { const auto offset=std::uint64_t(run.FirstSlot)*NaniteGpuPageBytes; NaniteGeometryCommands->CopyBufferRegion(pool->Resource.Get(),offset,old->Pool->Resource.Get(),offset,std::uint64_t(run.SlotCount)*NaniteGpuPageBytes); }
        for (std::size_t patch=0;patch<plan.UploadSlots.size();++patch)
            NaniteGeometryCommands->CopyBufferRegion(pool->Resource.Get(),std::uint64_t(plan.UploadSlots[patch])*NaniteGpuPageBytes,NaniteCacheUpload.Get(),patch*NaniteGpuPageBytes,NaniteGpuPageBytes);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={pool->Resource.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}; NaniteGeometryCommands->ResourceBarrier(1,&barrier);
        result=NaniteGeometryCommands->Close(); if (FAILED(result)) return FailHRESULT("Close Nanite cache recording",result);
        ID3D12CommandList* lists[]{NaniteGeometryCommands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
        if (!SignalAndWait()) return false;
        NanitePendingCache.reset(); NanitePendingCacheSource.reset(); NaniteCacheUpload.Reset();
    }
    const auto handle=NewNaniteCacheHandle(); if (!handle) return Fail("Nanite cache handle space is exhausted");
    NaniteCaches.emplace(handle.Value,NaniteNativeCache{std::move(plan.Next),std::move(pool)}); output=handle; statistics=plan.Statistics; return true;
}
