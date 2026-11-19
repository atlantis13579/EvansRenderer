struct NaniteFrameBuffer
{
    Impl& Owner; GLuint Buffer=0; unsigned Words=0;
    explicit NaniteFrameBuffer(Impl& owner):Owner(owner) {}
    ~NaniteFrameBuffer() { if (Buffer) Owner.GlDeleteBuffers(1,&Buffer); }
};
struct NaniteDepthRecord { NaniteDepthInfo Info; std::shared_ptr<NaniteFrameBuffer> Data; };
struct NaniteVisibilityRecord
{
    NaniteFrameVisibilityInfo Info; std::uint64_t ConstantAlignmentPadding=0; NaniteVisibilityConstants Constants;
    std::shared_ptr<NaniteFrameBuffer> Queries,Data;
    std::shared_ptr<const std::vector<NaniteVisibilityQuery>> QueryRecords;
    std::shared_ptr<const std::vector<NaniteClusterDraw>> Draws;
};
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteDepthRecord>> NaniteDepths;
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteVisibilityRecord>> NaniteDecisions;
NaniteVisibilityTransaction FrameNanite;
std::array<GLuint,7> NaniteFrameShaders{}; GLuint NaniteFrameUniform=0;
std::shared_ptr<NaniteFrameBuffer> NaniteFrameDummy,NaniteFrameMainDummy;
void(EVANS_GL_CALL* NaniteBindSampler)(GLuint,GLuint)=nullptr;
static constexpr GLenum NaniteActiveTexture=0x84E0,NaniteTextureBinding=0x8069,NaniteSamplerBinding=0x8919,NaniteDrawBinding=0x8CA6,NaniteReadBinding=0x8CAA;
static constexpr GLbitfield NaniteTextureFetchBarrier=0x00000008,NaniteFramebufferBarrier=0x00000400;
bool NaniteCommandsEnded() const { return false; }
bool ValidateNaniteRecording(bool rasterInternal=false)
{ return (FrameOpen && (!PassOpen || (rasterInternal && CurrentPass.Type==PassType::GBuffer))) || Fail("Nanite GPU frame work requires an open frame between graphics passes"); }
void FinishNaniteFrame(bool publish) { FrameNanite.Finish(publish,NaniteDepths,NaniteDecisions); FinishNaniteDraws(); }
struct NanitePresentPublication
{
    Impl& Owner; bool Published=false;
    ~NanitePresentPublication() { if (Published || !Owner.FrameOpen) Owner.FinishNaniteFrame(Published); }
};
void DestroyNaniteFrameVisibility()
{
    FinishNaniteFrame(false); NaniteDepths.clear(); NaniteDecisions.clear(); NaniteFrameDummy.reset(); NaniteFrameMainDummy.reset();
    if (NaniteFrameUniform) GlDeleteBuffers(1,&NaniteFrameUniform); NaniteFrameUniform=0;
    for (auto& shader:NaniteFrameShaders) { if (shader && GlDeleteProgram) GlDeleteProgram(shader); shader=0; }
}
bool CreateNaniteFrameBuffer(unsigned words,const void* initial,bool,std::shared_ptr<NaniteFrameBuffer>& output)
{
    if (std::size_t(words)*sizeof(unsigned)>std::size_t(NaniteMaximumBlock)) return Fail("Nanite frame buffer exceeds ES storage range");
    auto buffer=std::make_shared<NaniteFrameBuffer>(*this); buffer->Words=words; GLint previous=0; GlGetIntegerv(GpuStorageBinding,&previous);
    GlGenBuffers(1,&buffer->Buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer->Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(words)*sizeof(unsigned),initial,GL_STATIC_DRAW);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GLuint(previous)); if (!CheckGL("Allocate Nanite frame buffer")) return false;
    output=std::move(buffer); return true;
}
bool EnsureNaniteFrameVisibility()
{
    if (!NaniteGeometrySupported || !LoadGL(NaniteBindSampler,"glBindSampler")) return Fail("ES Nanite frame visibility requires compute and sampler binding");
    if (!NaniteFrameDummy) { const unsigned clear=0; if (!CreateNaniteFrameBuffer(1,&clear,false,NaniteFrameDummy)) return false; }
    // ANGLE lowers SSBOs to UAVs even for readonly blocks. Active bindings must
    // not alias the same allocation when history and main feedback are absent.
    if (!NaniteFrameMainDummy) { const unsigned clear=0; if (!CreateNaniteFrameBuffer(1,&clear,false,NaniteFrameMainDummy)) return false; }
    if (!NaniteFrameUniform)
    {
        GLint previous=0; GlGetIntegerv(GpuUniformBinding,&previous); GlGenBuffers(1,&NaniteFrameUniform); GlBindBuffer(GL_UNIFORM_BUFFER,NaniteFrameUniform);
        GlBufferData(GL_UNIFORM_BUFFER,sizeof(NaniteVisibilityConstants),nullptr,GL_STATIC_DRAW); GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(previous));
        if (!CheckGL("Allocate Nanite frame constants")) return false;
    }
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","NaniteVisibility.glsl",source,LastError)) return false;
    const auto versionEnd=source.find('\n'); if (versionEnd==std::string::npos) return Fail("Nanite frame shader has no version");
    constexpr unsigned passes[]{0,2,3,0};
    for (unsigned pass=0;pass<NaniteFrameShaders.size();++pass) if (!NaniteFrameShaders[pass])
    {
        auto text=source;
        if (pass>=3)
        {
            if (!LoadTechniqueShaderSource("OpenGLES","NaniteDraws.glsl",text,LastError)) return false;
            if (pass>=4) text.insert(text.find('\n')+1,"#define EVANS_NANITE_BIN_PASS "+std::to_string(pass-4)+"\n");
        }
        else text.insert(versionEnd+1,"#define EVANS_NANITE_VISIBILITY_PASS "+std::to_string(passes[pass])+"\n");
        const auto program=CompileComputeProgram(text); if(!program)return false;
        NaniteFrameShaders[pass]=program;
    }
    return true;
}
struct NaniteFrameBindings
{
    Impl& Owner; std::array<GLint,5> Storage{}; std::array<std::int64_t,5> Starts{},Sizes{};
    GLuint StorageSlots=4;
    GLint Uniform=0,GenericUniform=0,GenericStorage=0,Program=0,ActiveTexture=0,Texture=0,Sampler=0,Read=0,Draw=0;
    std::int64_t UniformStart=0,UniformSize=0;
    explicit NaniteFrameBindings(Impl& owner):Owner(owner)
    {
        StorageSlots=Owner.VisibilitySplitStorage?5:4;
        for (GLuint slot=0;slot<StorageSlots;++slot)
        { Owner.GpuGetIntegerIndexed(GpuStorageBinding,slot,&Storage[slot]); Owner.NaniteGetRange(NaniteStorageStart,slot,&Starts[slot]); Owner.NaniteGetRange(NaniteStorageSize,slot,&Sizes[slot]); }
        Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform); Owner.NaniteGetRange(NaniteUniformStart,0,&UniformStart); Owner.NaniteGetRange(NaniteUniformSize,0,&UniformSize);
        Owner.GlGetIntegerv(GpuUniformBinding,&GenericUniform); Owner.GlGetIntegerv(GpuStorageBinding,&GenericStorage); Owner.GlGetIntegerv(GpuCurrentProgram,&Program);
        Owner.GlGetIntegerv(NaniteActiveTexture,&ActiveTexture); Owner.GlActiveTexture(GL_TEXTURE0); Owner.GlGetIntegerv(NaniteTextureBinding,&Texture); Owner.GlGetIntegerv(NaniteSamplerBinding,&Sampler);
        Owner.GlGetIntegerv(NaniteReadBinding,&Read); Owner.GlGetIntegerv(NaniteDrawBinding,&Draw);
    }
    ~NaniteFrameBindings()
    {
        for (GLuint slot=0;slot<StorageSlots;++slot)
            if (Storage[slot] && Sizes[slot]>0) Owner.NaniteBindRange(GL_SHADER_STORAGE_BUFFER,slot,GLuint(Storage[slot]),GLintptr(Starts[slot]),GLsizeiptr(Sizes[slot]));
            else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,GLuint(Storage[slot]));
        if (Uniform && UniformSize>0) Owner.NaniteBindRange(GL_UNIFORM_BUFFER,0,GLuint(Uniform),GLintptr(UniformStart),GLsizeiptr(UniformSize));
        else Owner.GlBindBufferBase(GL_UNIFORM_BUFFER,0,GLuint(Uniform));
        Owner.GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(GenericUniform)); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GLuint(GenericStorage)); Owner.GlUseProgram(GLuint(Program));
        Owner.GlActiveTexture(GL_TEXTURE0); Owner.GlBindTexture(GL_TEXTURE_2D,GLuint(Texture)); Owner.NaniteBindSampler(0,GLuint(Sampler)); Owner.GlActiveTexture(GLenum(ActiveTexture));
        Owner.GlBindFramebuffer(GL_READ_FRAMEBUFFER,GLuint(Read)); Owner.GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,GLuint(Draw));
    }
};
bool DispatchNaniteFrame(unsigned pass,const NaniteVisibilityConstants& constants,unsigned groupsX,unsigned groupsY,
    GLuint source,GLuint depth,GLuint main,GLuint output,GLuint camera)
{
    if (groupsX>unsigned(NaniteMaximumGroupsX) || groupsY>unsigned(NaniteMaximumGroupsY)) return Fail("ES Nanite frame dispatch exceeds device capacity");
    NaniteFrameBindings restore(*this); GlBindFramebuffer(GL_FRAMEBUFFER,0);
    GlBindBuffer(GL_UNIFORM_BUFFER,NaniteFrameUniform); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants); GlBindBufferBase(GL_UNIFORM_BUFFER,0,NaniteFrameUniform);
    const GLuint buffers[]{source,output,depth?depth:NaniteFrameDummy->Buffer,
        main && main!=NaniteFrameDummy->Buffer?main:NaniteFrameMainDummy->Buffer};
    for (unsigned slot=0;slot<4;++slot) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,buffers[slot]);
    GlUseProgram(NaniteFrameShaders[pass]);
    if (camera)
    { GlActiveTexture(GL_TEXTURE0); NaniteBindSampler(0,0); GlBindTexture(GL_TEXTURE_2D,camera); GpuMemoryBarrier(NaniteFramebufferBarrier|NaniteTextureFetchBarrier); }
    GpuDispatch(groupsX,groupsY,1); GpuMemoryBarrier(GpuStorageBarrier); return CheckGL("Dispatch Nanite GPU frame visibility");
}
bool CaptureNaniteDepth(NaniteDepthHandle& output,bool rasterInternal=false)
{
    NaniteVisibilityWork work;
    if (!ValidateNaniteRecording(rasterInternal) || FrameSampleCount!=1 || !DepthTexture || !NanitePrimaryDepthInitialized(PendingFrame.Passes)) return Fail("Nanite depth capture requires initialized single-sample camera depth");
    if (!BuildNaniteDepthWork(CreateInfo.Width,CreateInfo.Height,work,LastError) || !FrameNanite.CanAllocate(std::size_t(work.OutputWords)*sizeof(unsigned),true,LastError) || !EnsureNaniteFrameVisibility()) return false;
    auto record=std::make_shared<NaniteDepthRecord>();
    if (!CreateNaniteFrameBuffer(work.OutputWords,0,true,record->Data)) return false;
    record->Info={FrameNanite.Frame(),unsigned(CreateInfo.Width),unsigned(CreateInfo.Height),work.OutputWords,false};
    for (const auto& job:work.Dispatches)
        if (!DispatchNaniteFrame(job.Pass==2?1:0,job.Constants,job.GroupsX,job.GroupsY,NaniteFrameDummy->Buffer,0,0,record->Data->Buffer,job.Pass==2?DepthTexture:0)) return false;
    NaniteDepthHandle handle{NewNaniteVisibilityIdentity()}; NaniteDepths.emplace(handle.Value,record); FrameNanite.Record(handle,std::size_t(work.OutputWords)*sizeof(unsigned)); output=handle; return true;
}
bool BeginNaniteVisibility(NaniteDepthHandle history,const std::vector<NaniteVisibilityQuery>& queries,NaniteVisibilityHandle& output)
{
    if (!ValidateNaniteRecording()) return false;
    std::shared_ptr<NaniteDepthRecord> depth;
    if (history) { const auto found=NaniteDepths.find(history.Value); if (found==NaniteDepths.end() || !found->second->Info.Published) return Fail("Nanite history must be a committed depth snapshot"); depth=found->second; }
    NaniteVisibilityWork work; if (!BuildNaniteFrameCull(queries,depth?depth->Info.Width:CreateInfo.Width,depth?depth->Info.Height:CreateInfo.Height,NaniteVisibilityPhase::Main,work,LastError)) return false;
    const auto bytes=(work.Input.size()+work.OutputWords)*sizeof(unsigned)+queries.size()*sizeof(NaniteVisibilityQuery);
    if (!FrameNanite.CanAllocate(bytes,false,LastError) || !EnsureNaniteFrameVisibility()) return false;
    auto record=std::make_shared<NaniteVisibilityRecord>();
    if (!CreateNaniteFrameBuffer(unsigned(work.Input.size()),work.Input.data(),false,record->Queries) || !CreateNaniteFrameBuffer(work.OutputWords,0,true,record->Data)) return false;
    const auto& job=work.Dispatches.front(); record->Constants=job.Constants; record->Constants.Layout.W=depth?1:0;
    record->Info={FrameNanite.Frame(),work.OutputWords,NaniteVisibilityPhase::Main,false};
    record->QueryRecords=std::make_shared<const std::vector<NaniteVisibilityQuery>>(queries);
    if (!DispatchNaniteFrame(2,record->Constants,job.GroupsX,job.GroupsY,record->Queries->Buffer,depth?depth->Data->Buffer:NaniteFrameDummy->Buffer,NaniteFrameDummy->Buffer,record->Data->Buffer,0)) return false;
    NaniteVisibilityHandle handle{NewNaniteVisibilityIdentity()}; NaniteDecisions.emplace(handle.Value,record); FrameNanite.Record(handle,bytes); output=handle; return true;
}
bool RetestNaniteVisibility(NaniteVisibilityHandle main,NaniteDepthHandle camera,NaniteVisibilityHandle& output,bool rasterInternal=false)
{
    if (!ValidateNaniteRecording(rasterInternal)) return false;
    const auto m=NaniteDecisions.find(main.Value); const auto d=NaniteDepths.find(camera.Value);
    if (m==NaniteDecisions.end() || d==NaniteDepths.end() || m->second->Info.Frame!=FrameNanite.CurrentFrame() || d->second->Info.Frame!=FrameNanite.CurrentFrame() ||
        m->second->Info.Phase!=NaniteVisibilityPhase::Main || m->second->Info.Published || d->second->Info.Published) return Fail("Nanite retest requires this frame's main decisions and camera depth");
    if (m->second->Draws && !NaniteMainDrawsComplete(m->second->Draws)) return Fail("Submit all main Nanite candidates before retesting");
    const auto bytes=std::size_t(m->second->Info.QueryCount)*sizeof(unsigned); if (!FrameNanite.CanAllocate(bytes,false,LastError)) return false;
    auto record=std::make_shared<NaniteVisibilityRecord>(); record->Info=m->second->Info; record->Info.Phase=NaniteVisibilityPhase::Post; record->Queries=m->second->Queries; record->QueryRecords=m->second->QueryRecords; record->Draws=m->second->Draws;
    if (!CreateNaniteFrameBuffer(record->Info.QueryCount,0,true,record->Data)) return false;
    record->Constants=m->second->Constants; record->Constants.Counts.Y=d->second->Info.Width; record->Constants.Counts.Z=d->second->Info.Height; record->Constants.Counts.W=1; record->Constants.Layout.W=1;
    const unsigned groups=(record->Info.QueryCount+NaniteVisibilityThreads-1)/NaniteVisibilityThreads,gx=record->Constants.Target.W;
    if (!DispatchNaniteFrame(2,record->Constants,gx,(groups+gx-1)/gx,record->Queries->Buffer,d->second->Data->Buffer,m->second->Data->Buffer,record->Data->Buffer,0)) return false;
    NaniteVisibilityHandle handle{NewNaniteVisibilityIdentity()}; NaniteDecisions.emplace(handle.Value,record); FrameNanite.Record(handle,bytes); output=handle; return true;
}

bool ReadNaniteFrameBuffer(const NaniteFrameBuffer& buffer,std::vector<unsigned>& words)
{
    GLint previous=0; GlGetIntegerv(GpuStorageBinding,&previous); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer.Buffer);
    GpuMemoryBarrier(GpuStorageBarrier|NaniteBufferBarrier);
    const void* mapped=NaniteMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(buffer.Words)*sizeof(unsigned),NaniteMapRead);
    if (!mapped) { GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GLuint(previous)); return Fail("ES Nanite frame mapping failed"); }
    words.resize(buffer.Words); std::memcpy(words.data(),mapped,std::size_t(buffer.Words)*sizeof(unsigned)); const bool valid=NaniteUnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=0;
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GLuint(previous)); return CheckGL("Read Nanite frame feedback") && (valid || Fail("ES Nanite frame mapping invalidated"));
}
