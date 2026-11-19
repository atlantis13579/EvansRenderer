// Copy resident runs on the GPU and upload changed pages into a separate buffer.
// Generic copy/storage bindings are restored. Independent handles may share a
// pool; releasing either handle never mutates the other's native allocation.
static constexpr GLenum NaniteCopyRead=0x8F36,NaniteCopyWrite=0x8F37;
struct NaniteCachePool { GLuint Buffer=0; };
struct NaniteNativeCache { NaniteCacheInfo Info; std::shared_ptr<NaniteCachePool> Pool; };
std::unordered_map<std::uint64_t,NaniteNativeCache> NaniteCaches;
void(EVANS_GL_CALL* NaniteCopyBuffer)(GLenum,GLenum,GLintptr,GLintptr,GLsizeiptr)=nullptr;
void DestroyNaniteCaches() { NaniteCaches.clear(); }
bool DestroyNaniteCache(NaniteCacheHandle handle)
{
    if (!handle || !NaniteCaches.erase(handle.Value)) return Fail("ES Nanite cache handle is invalid or belongs to another RHI");
    return true;
}
bool PrepareNaniteCache(const NaniteGpuUpload& upload,NaniteCacheHandle previous,NaniteCacheHandle& output,NaniteCacheStatistics& statistics)
{
    const NaniteNativeCache* old=nullptr;
    if (previous) { const auto found=NaniteCaches.find(previous.Value); if (found==NaniteCaches.end()) return Fail("ES previous Nanite cache handle is invalid"); old=&found->second; }
    NaniteCachePlan plan; if (!BuildNaniteCachePlan(old?&old->Info:nullptr,upload,plan,LastError)) return false;
    if (!NaniteGeometrySupported || std::uint64_t(upload.Constants.Limits.Y)*sizeof(std::uint32_t)>static_cast<std::uint64_t>(NaniteMaximumBlock)) return Fail("ES Nanite cache exceeds compute/storage limits");
    if (!LoadGL(NaniteCopyBuffer,"glCopyBufferSubData")) return false;
    std::shared_ptr<NaniteCachePool> pool;
    if (plan.ReusePool) pool=old->Pool;
    else
    {
        struct Restore
        {
            Impl& Owner; GLint Read=0,Write=0,Storage=0;
            explicit Restore(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(NaniteCopyRead,&Read); Owner.GlGetIntegerv(NaniteCopyWrite,&Write); Owner.GlGetIntegerv(GpuStorageBinding,&Storage); }
            ~Restore() { Owner.GlBindBuffer(NaniteCopyRead,static_cast<GLuint>(Read)); Owner.GlBindBuffer(NaniteCopyWrite,static_cast<GLuint>(Write)); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,static_cast<GLuint>(Storage)); }
        } restore(*this);
        pool=std::shared_ptr<NaniteCachePool>(new NaniteCachePool{},[this](NaniteCachePool* item){if (item->Buffer) GlDeleteBuffers(1,&item->Buffer); delete item;});
        GlGenBuffers(1,&pool->Buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,pool->Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(plan.Statistics.AllocatedBytes),nullptr,GL_STATIC_DRAW);
        for (auto slot:plan.UploadSlots) GlBufferSubData(GL_SHADER_STORAGE_BUFFER,static_cast<GLintptr>(std::uint64_t(slot)*NaniteGpuPageBytes),NaniteGpuPageBytes,&upload.PhysicalPages[slot]);
        if (!plan.Copies.empty())
        {
            GlBindBuffer(NaniteCopyRead,old->Pool->Buffer); GlBindBuffer(NaniteCopyWrite,pool->Buffer);
            for (const auto& run:plan.Copies)
            { const auto offset=static_cast<GLintptr>(std::uint64_t(run.FirstSlot)*NaniteGpuPageBytes); NaniteCopyBuffer(NaniteCopyRead,NaniteCopyWrite,offset,offset,static_cast<GLsizeiptr>(std::uint64_t(run.SlotCount)*NaniteGpuPageBytes)); }
        }
        GpuMemoryBarrier(GpuStorageBarrier|NaniteBufferBarrier); if (!CheckGL("Copy Nanite cache pages")) return false;
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,pool->Buffer); const auto* mapped=NaniteMapBuffer(GL_SHADER_STORAGE_BUFFER,0,sizeof(std::uint32_t),NaniteMapRead);
        if (!mapped) return Fail("ES Nanite cache completion mapping failed"); const bool valid=NaniteUnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=0;
        if (!CheckGL("Complete Nanite cache update") || !valid) return Fail("ES Nanite cache completion read invalidated");
    }
    const auto handle=NewNaniteCacheHandle(); if (!handle) return Fail("Nanite cache handle space is exhausted");
    NaniteCaches.emplace(handle.Value,NaniteNativeCache{std::move(plan.Next),std::move(pool)}); output=handle; statistics=plan.Statistics; return true;
}
