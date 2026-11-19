// Immutable shared-buffer pools combine CPU patches and blit copies. Tracked
// hazards and command completion protect old versions and candidate publication.
struct NaniteCachePool { ObjCObject Buffer=nullptr; };
struct NaniteNativeCache { NaniteCacheInfo Info; std::shared_ptr<NaniteCachePool> Pool; };
std::unordered_map<std::uint64_t,NaniteNativeCache> NaniteCaches;
void DestroyNaniteCaches() { NaniteCaches.clear(); }
bool DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!handle || !NaniteCaches.erase(handle.Value)) return Fail("Metal Nanite cache handle is invalid or belongs to another RHI");
    return true;
}
bool PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& output,NaniteCacheStatistics& statistics)
{
    const NaniteNativeCache* old=nullptr;
    if (previous) { const auto found=NaniteCaches.find(previous.Value); if (found==NaniteCaches.end()) return Fail("Metal previous Nanite cache handle is invalid"); old=&found->second; }
    NaniteCachePlan plan; if (!BuildNaniteCachePlan(old?&old->Info:nullptr,upload,plan,LastError)) return false;
    std::shared_ptr<NaniteCachePool> pool;
    if (plan.ReusePool) pool=old->Pool;
    else
    {
        pool=std::shared_ptr<NaniteCachePool>(new NaniteCachePool{},[](NaniteCachePool* item){ReleaseObject(item->Buffer); delete item;});
        pool->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",static_cast<NSUInteger>(plan.Statistics.AllocatedBytes),static_cast<NSUInteger>(0));
        if (!pool->Buffer) return Fail("Metal Nanite cache allocation failed"); auto* mapped=Send<void*>(pool->Buffer,"contents"); if (!mapped) return Fail("Metal Nanite cache is not host-visible");
        for (auto slot:plan.UploadSlots) std::memcpy(static_cast<std::uint8_t*>(mapped)+std::size_t(slot)*NaniteGpuPageBytes,&upload.PhysicalPages[slot],NaniteGpuPageBytes);
        if (!plan.Copies.empty())
        {
            ObjCObject command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Nanite cache command buffer failed");
            ObjCObject encoder=Send<ObjCObject>(command,"blitCommandEncoder"); if (!encoder) return Fail("Metal Nanite cache blit encoder failed");
            for (const auto& run:plan.Copies)
            {
                const auto offset=static_cast<NSUInteger>(std::uint64_t(run.FirstSlot)*NaniteGpuPageBytes),bytes=static_cast<NSUInteger>(std::uint64_t(run.SlotCount)*NaniteGpuPageBytes);
                Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(encoder,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",old->Pool->Buffer,offset,pool->Buffer,offset,bytes);
            }
            Send<void>(encoder,"endEncoding"); Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
            if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal Nanite cache blit failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
        }
    }
    const auto handle=NewNaniteCacheHandle(); if (!handle) return Fail("Nanite cache handle space is exhausted");
    NaniteCaches.emplace(handle.Value,NaniteNativeCache{std::move(plan.Next),std::move(pool)}); output=handle; statistics=plan.Statistics; return true;
}
