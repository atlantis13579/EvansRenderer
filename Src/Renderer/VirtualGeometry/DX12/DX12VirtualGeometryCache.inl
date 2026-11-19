// Native pool versions are immutable. Copy unchanged resident runs on the direct
// queue and stage changed pages only. Pending copy resources survive a failed
// fence wait; recovery completes before allocator reuse or cache release.
struct VirtualGeometryCachePool { ComPtr<ID3D12Resource> Resource; };
struct VirtualGeometryNativeCache { VirtualGeometryCacheInfo Info; std::shared_ptr<VirtualGeometryCachePool> Pool; };
std::unordered_map<std::uint64_t,VirtualGeometryNativeCache> VirtualGeometryCaches;
std::shared_ptr<VirtualGeometryCachePool> VirtualGeometryPendingCache;
ComPtr<ID3D12Resource> VirtualGeometryCacheUpload;
void DestroyVirtualGeometryCaches()
{ VirtualGeometryCaches.clear(); VirtualGeometryPendingCache.reset(); VirtualGeometryCacheUpload.Reset(); }
bool DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!RecoverSubmittedWork("Release virtual cache") || !WaitForGPU()) return false;
    if (!handle || !VirtualGeometryCaches.erase(handle.Value)) return Fail("DX12 virtual cache handle is invalid or belongs to another RHI");
    return true;
}
bool PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& output,VirtualGeometryCacheStatistics& statistics)
{
    const VirtualGeometryNativeCache* old=nullptr;
    if (previous) { const auto found=VirtualGeometryCaches.find(previous.Value); if (found==VirtualGeometryCaches.end()) return Fail("DX12 previous virtual cache handle is invalid"); old=&found->second; }
    VirtualGeometryCachePlan plan; if (!BuildVirtualGeometryCachePlan(old?&old->Info:nullptr,upload,plan,LastError)) return false;
    if (!RecoverSubmittedWork("Prepare virtual cache") || !WaitForGPU() || !EnsureVirtualGeometry()) return false;
    VirtualGeometryPendingCache.reset(); VirtualGeometryCacheUpload.Reset();
    auto pool=plan.ReusePool?old->Pool:std::make_shared<VirtualGeometryCachePool>();
    if (!plan.ReusePool)
    {
        if (!CreateBufferResource(static_cast<std::size_t>(plan.Statistics.AllocatedBytes),D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,pool->Resource) ||
            !CreateBufferResource(static_cast<std::size_t>(plan.Statistics.UploadedBytes),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,VirtualGeometryCacheUpload)) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; auto result=VirtualGeometryCacheUpload->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map virtual cache patches",result);
        for (std::size_t patch=0;patch<plan.UploadSlots.size();++patch) std::memcpy(static_cast<std::uint8_t*>(mapped)+patch*VirtualGeometryPageBytes,&upload.PhysicalPages[plan.UploadSlots[patch]],VirtualGeometryPageBytes);
        VirtualGeometryCacheUpload->Unmap(0,nullptr); VirtualGeometryPendingCache=pool;
        result=VirtualGeometryAllocator->Reset(); if (SUCCEEDED(result)) result=VirtualGeometryCommands->Reset(VirtualGeometryAllocator.Get(),nullptr);
        if (FAILED(result)) return FailHRESULT("Reset virtual cache recording",result);
        for (const auto& run:plan.Copies)
        { const auto offset=std::uint64_t(run.FirstSlot)*VirtualGeometryPageBytes; VirtualGeometryCommands->CopyBufferRegion(pool->Resource.Get(),offset,old->Pool->Resource.Get(),offset,std::uint64_t(run.SlotCount)*VirtualGeometryPageBytes); }
        for (std::size_t patch=0;patch<plan.UploadSlots.size();++patch)
            VirtualGeometryCommands->CopyBufferRegion(pool->Resource.Get(),std::uint64_t(plan.UploadSlots[patch])*VirtualGeometryPageBytes,VirtualGeometryCacheUpload.Get(),patch*VirtualGeometryPageBytes,VirtualGeometryPageBytes);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={pool->Resource.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE}; VirtualGeometryCommands->ResourceBarrier(1,&barrier);
        result=VirtualGeometryCommands->Close(); if (FAILED(result)) return FailHRESULT("Close virtual cache recording",result);
        ID3D12CommandList* lists[]{VirtualGeometryCommands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
        if (!SignalAndWait()) return false;
        VirtualGeometryPendingCache.reset(); VirtualGeometryCacheUpload.Reset();
    }
    const auto handle=NewVirtualGeometryCacheHandle(); if (!handle) return Fail("Virtual cache handle space is exhausted");
    VirtualGeometryCaches.emplace(handle.Value,VirtualGeometryNativeCache{std::move(plan.Next),std::move(pool)}); output=handle; statistics=plan.Statistics; return true;
}
