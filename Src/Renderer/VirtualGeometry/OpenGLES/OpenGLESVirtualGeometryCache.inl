// Copy resident runs on the GPU and upload changed pages into a separate buffer.
// Generic copy/storage bindings are restored. Independent handles may share a
// pool; releasing either handle never mutates the other's native allocation.
static constexpr GLenum VirtualCopyRead=0x8F36,VirtualCopyWrite=0x8F37;
struct VirtualGeometryCachePool { GLuint Buffer=0; };
struct VirtualGeometryNativeCache { VirtualGeometryCacheInfo Info; std::shared_ptr<VirtualGeometryCachePool> Pool; };
std::unordered_map<std::uint64_t,VirtualGeometryNativeCache> VirtualGeometryCaches;
void(EVANS_GL_CALL* VirtualCopyBuffer)(GLenum,GLenum,GLintptr,GLintptr,GLsizeiptr)=nullptr;
void DestroyVirtualGeometryCaches() { VirtualGeometryCaches.clear(); }
bool DestroyVirtualGeometryCache(VirtualGeometryCacheHandle handle)
{
    if (!handle || !VirtualGeometryCaches.erase(handle.Value)) return Fail("ES virtual cache handle is invalid or belongs to another RHI");
    return true;
}
bool PrepareVirtualGeometryCache(const VirtualGeometryGpuUpload& upload,VirtualGeometryCacheHandle previous,VirtualGeometryCacheHandle& output,VirtualGeometryCacheStatistics& statistics)
{
    const VirtualGeometryNativeCache* old=nullptr;
    if (previous) { const auto found=VirtualGeometryCaches.find(previous.Value); if (found==VirtualGeometryCaches.end()) return Fail("ES previous virtual cache handle is invalid"); old=&found->second; }
    VirtualGeometryCachePlan plan; if (!BuildVirtualGeometryCachePlan(old?&old->Info:nullptr,upload,plan,LastError)) return false;
    if (!VirtualGeometrySupported || std::uint64_t(upload.Constants.Limits.Z)*sizeof(std::uint32_t)>static_cast<std::uint64_t>(VirtualMaximumBlock)) return Fail("ES virtual cache exceeds compute/storage limits");
    if (!LoadGL(VirtualCopyBuffer,"glCopyBufferSubData")) return false;
    std::shared_ptr<VirtualGeometryCachePool> pool;
    if (plan.ReusePool) pool=old->Pool;
    else
    {
        struct Restore
        {
            Impl& Owner; GLint Read=0,Write=0,Storage=0;
            explicit Restore(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(VirtualCopyRead,&Read); Owner.GlGetIntegerv(VirtualCopyWrite,&Write); Owner.GlGetIntegerv(GpuStorageBinding,&Storage); }
            ~Restore() { Owner.GlBindBuffer(VirtualCopyRead,static_cast<GLuint>(Read)); Owner.GlBindBuffer(VirtualCopyWrite,static_cast<GLuint>(Write)); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,static_cast<GLuint>(Storage)); }
        } restore(*this);
        pool=std::shared_ptr<VirtualGeometryCachePool>(new VirtualGeometryCachePool{},[this](VirtualGeometryCachePool* item){if (item->Buffer) GlDeleteBuffers(1,&item->Buffer); delete item;});
        GlGenBuffers(1,&pool->Buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,pool->Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(plan.Statistics.AllocatedBytes),nullptr,GL_STATIC_DRAW);
        for (auto slot:plan.UploadSlots) GlBufferSubData(GL_SHADER_STORAGE_BUFFER,static_cast<GLintptr>(std::uint64_t(slot)*VirtualGeometryPageBytes),VirtualGeometryPageBytes,&upload.PhysicalPages[slot]);
        if (!plan.Copies.empty())
        {
            GlBindBuffer(VirtualCopyRead,old->Pool->Buffer); GlBindBuffer(VirtualCopyWrite,pool->Buffer);
            for (const auto& run:plan.Copies)
            { const auto offset=static_cast<GLintptr>(std::uint64_t(run.FirstSlot)*VirtualGeometryPageBytes); VirtualCopyBuffer(VirtualCopyRead,VirtualCopyWrite,offset,offset,static_cast<GLsizeiptr>(std::uint64_t(run.SlotCount)*VirtualGeometryPageBytes)); }
        }
        GpuMemoryBarrier(GpuStorageBarrier|VirtualBufferBarrier); if (!CheckGL("Copy virtual cache pages")) return false;
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,pool->Buffer); const auto* mapped=VirtualMapBuffer(GL_SHADER_STORAGE_BUFFER,0,sizeof(std::uint32_t),VirtualMapRead);
        if (!mapped) return Fail("ES virtual cache completion mapping failed"); const bool valid=VirtualUnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=0;
        if (!CheckGL("Complete virtual cache update") || !valid) return Fail("ES virtual cache completion read invalidated");
    }
    const auto handle=NewVirtualGeometryCacheHandle(); if (!handle) return Fail("Virtual cache handle space is exhausted");
    VirtualGeometryCaches.emplace(handle.Value,VirtualGeometryNativeCache{std::move(plan.Next),std::move(pool)}); output=handle; statistics=plan.Statistics; return true;
}
