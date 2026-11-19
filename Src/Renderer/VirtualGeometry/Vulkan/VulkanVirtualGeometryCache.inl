// Candidate pools combine coherent host patches with GPU copies of unchanged
// resident runs. Transfer/host writes become visible to both compute and later
// copies. Shared ownership keeps borrowed pools alive through failed waits.
using VirtualGeometryCachePool=RawBuffer;
struct VirtualGeometryNativeCache { VirtualGeometryCacheInfo Info; std::shared_ptr<VirtualGeometryCachePool> Pool; };
std::unordered_map<std::uint64_t,VirtualGeometryNativeCache> VirtualGeometryCaches;
std::shared_ptr<VirtualGeometryCachePool> VirtualGeometryPendingCache;
PFN_vkCmdCopyBuffer VirtualGeometryCopyBuffer=nullptr;
void DestroyVirtualGeometryCaches() { VirtualGeometryCaches.clear(); VirtualGeometryPendingCache.reset(); }
bool DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Release virtual cache fence")) return false;
    if (!handle || !VirtualGeometryCaches.erase(handle.Value)) return Fail("Vulkan virtual cache handle is invalid or belongs to another RHI");
    return true;
}
bool PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& output,VirtualGeometryCacheStatistics& statistics)
{
    const VirtualGeometryNativeCache* old=nullptr;
    if (previous) { const auto found=VirtualGeometryCaches.find(previous.Value); if (found==VirtualGeometryCaches.end()) return Fail("Vulkan previous virtual cache handle is invalid"); old=&found->second; }
    VirtualGeometryCachePlan plan; if (!BuildVirtualGeometryCachePlan(old?&old->Info:nullptr,upload,plan,LastError)) return false;
    if (!SupportsVirtualGeometry() || upload.Constants.Limits.Z*sizeof(std::uint32_t)>std::uint64_t(DeviceLimits.maxStorageBufferRange)) return Fail("Vulkan virtual cache exceeds compute/storage limits");
    if (!Check(WaitForFences(Device,1,&FrameFence,VK_TRUE_VALUE,GpuWaitTimeout),"Recover virtual cache copy") || !LoadDevice(VirtualGeometryCopyBuffer,"vkCmdCopyBuffer")) return false;
    VirtualGeometryPendingCache.reset(); std::shared_ptr<VirtualGeometryCachePool> pool;
    if (plan.ReusePool) pool=old->Pool;
    else
    {
        pool=std::shared_ptr<VirtualGeometryCachePool>(new RawBuffer{},[this](RawBuffer* buffer){DestroyRawBuffer(*buffer); delete buffer;});
        if (!CreateRawBuffer(plan.Statistics.AllocatedBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,*pool)) return false;
        for (auto slot:plan.UploadSlots) std::memcpy(static_cast<std::uint8_t*>(pool->Mapped)+std::size_t(slot)*VirtualGeometryPageBytes,&upload.PhysicalPages[slot],VirtualGeometryPageBytes);
        VirtualGeometryPendingCache=pool; if (!BeginImmediate()) return false;
        if (!plan.Copies.empty())
        {
            VkBufferMemoryBarrier source{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; source.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_HOST_WRITE_BIT; source.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            source.srcQueueFamilyIndex=source.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; source.buffer=old->Pool->Buffer; source.size=old->Pool->Size;
            CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&source,0,nullptr);
            std::vector<VkBufferCopy> copies;
            for (const auto& run:plan.Copies) { const auto offset=std::uint64_t(run.FirstSlot)*VirtualGeometryPageBytes; copies.push_back({offset,offset,std::uint64_t(run.SlotCount)*VirtualGeometryPageBytes}); }
            VirtualGeometryCopyBuffer(CommandBuffer,old->Pool->Buffer,pool->Buffer,static_cast<unsigned>(copies.size()),copies.data());
        }
        VkBufferMemoryBarrier visible{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; visible.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_HOST_WRITE_BIT; visible.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT;
        visible.srcQueueFamilyIndex=visible.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; visible.buffer=pool->Buffer; visible.size=pool->Size;
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&visible,0,nullptr);
        if (!EndImmediate()) return false; VirtualGeometryPendingCache.reset();
    }
    const auto handle=NewVirtualGeometryCacheHandle(); if (!handle) return Fail("Virtual cache handle space is exhausted");
    VirtualGeometryCaches.emplace(handle.Value,VirtualGeometryNativeCache{std::move(plan.Next),std::move(pool)}); output=handle; statistics=plan.Statistics; return true;
}
