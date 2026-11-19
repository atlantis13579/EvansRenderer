// Immutable shared-buffer pools combine CPU patches and blit copies. Tracked
// hazards and command completion protect old versions and candidate publication.
struct VirtualGeometryCachePool { ObjCObject Buffer=nullptr; };
struct VirtualGeometryNativeCache { VirtualGeometryCacheInfo Info; std::shared_ptr<VirtualGeometryCachePool> Pool; };
std::unordered_map<std::uint64_t,VirtualGeometryNativeCache> VirtualGeometryCaches;
void DestroyVirtualGeometryCaches() { VirtualGeometryCaches.clear(); }
bool DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!handle || !VirtualGeometryCaches.erase(handle.Value)) return Fail("Metal virtual cache handle is invalid or belongs to another RHI");
    return true;
}
bool PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& output,VirtualGeometryCacheStatistics& statistics)
{
    const VirtualGeometryNativeCache* old=nullptr;
    if (previous) { const auto found=VirtualGeometryCaches.find(previous.Value); if (found==VirtualGeometryCaches.end()) return Fail("Metal previous virtual cache handle is invalid"); old=&found->second; }
    VirtualGeometryCachePlan plan; if (!BuildVirtualGeometryCachePlan(old?&old->Info:nullptr,upload,plan,LastError)) return false;
    std::shared_ptr<VirtualGeometryCachePool> pool;
    if (plan.ReusePool) pool=old->Pool;
    else
    {
        pool=std::shared_ptr<VirtualGeometryCachePool>(new VirtualGeometryCachePool{},[](VirtualGeometryCachePool* item){ReleaseObject(item->Buffer); delete item;});
        pool->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",static_cast<NSUInteger>(plan.Statistics.AllocatedBytes),static_cast<NSUInteger>(0));
        if (!pool->Buffer) return Fail("Metal virtual cache allocation failed"); auto* mapped=Send<void*>(pool->Buffer,"contents"); if (!mapped) return Fail("Metal virtual cache is not host-visible");
        for (auto slot:plan.UploadSlots) std::memcpy(static_cast<std::uint8_t*>(mapped)+std::size_t(slot)*VirtualGeometryPageBytes,&upload.PhysicalPages[slot],VirtualGeometryPageBytes);
        if (!plan.Copies.empty())
        {
            ObjCObject command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal virtual cache command buffer failed");
            ObjCObject encoder=Send<ObjCObject>(command,"blitCommandEncoder"); if (!encoder) return Fail("Metal virtual cache blit encoder failed");
            for (const auto& run:plan.Copies)
            {
                const auto offset=static_cast<NSUInteger>(std::uint64_t(run.FirstSlot)*VirtualGeometryPageBytes),bytes=static_cast<NSUInteger>(std::uint64_t(run.SlotCount)*VirtualGeometryPageBytes);
                Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(encoder,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",old->Pool->Buffer,offset,pool->Buffer,offset,bytes);
            }
            Send<void>(encoder,"endEncoding"); Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
            if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal virtual cache blit failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
        }
    }
    const auto handle=NewVirtualGeometryCacheHandle(); if (!handle) return Fail("Virtual cache handle space is exhausted");
    VirtualGeometryCaches.emplace(handle.Value,VirtualGeometryNativeCache{std::move(plan.Next),std::move(pool)}); output=handle; statistics=plan.Statistics; return true;
}
