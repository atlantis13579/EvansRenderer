void*(EVANS_GL_CALL* GlobalFenceSync)(GLenum,GLbitfield)=nullptr;
GLenum(EVANS_GL_CALL* GlobalClientWaitSync)(void*,GLbitfield,std::uint64_t)=nullptr;
void(EVANS_GL_CALL* GlobalDeleteSync)(void*)=nullptr;
void(EVANS_GL_CALL* GlobalFlush)()=nullptr;
static constexpr GLenum GlobalSyncComplete=0x9117,GlobalAlreadySignaled=0x911A,GlobalTimeoutExpired=0x911B,GlobalConditionSatisfied=0x911C;
struct LumenGlobalWork : LumenGlobalDistanceJob
{
    Impl& Owner; std::shared_ptr<LumenDistanceRecord> Record,Previous;
    GLuint Uniform=0,Dirty=0; void* Completion=nullptr;
    explicit LumenGlobalWork(Impl& owner):Owner(owner) {}
    ~LumenGlobalWork()
    {
        if (Completion) Owner.GlobalDeleteSync(Completion);
        if (Uniform) Owner.GlDeleteBuffers(1,&Uniform); if (Dirty) Owner.GlDeleteBuffers(1,&Dirty);
    }
};
LumenGlobalDistanceQueue<LumenGlobalWork> LumenGlobalTasks;
void DestroyLumenGlobalDistanceAsync() { LumenGlobalTasks.Clear(); }
bool ProbeLumenGlobalDistance(LumenGlobalWork& work,bool& ready)
{
    const auto status=GlobalClientWaitSync(work.Completion,0,0);
    ready=status==GlobalAlreadySignaled || status==GlobalConditionSatisfied;
    return ready || status==GlobalTimeoutExpired || Fail("ES global distance fence query failed");
}
bool CollectLumenGlobalDistance()
{ return LumenGlobalTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenGlobalDistance(work,ready); }); }
bool CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{ return (LumenGlobalTasks.Cancel(task) || Fail("Invalid global distance task")) && CollectLumenGlobalDistance(); }
bool BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& output)
{
    if (!LoadGL(GlobalFenceSync,"glFenceSync") || !LoadGL(GlobalClientWaitSync,"glClientWaitSync") || !LoadGL(GlobalDeleteSync,"glDeleteSync") ||
        !LoadGL(GlobalFlush,"glFlush") || !CollectLumenGlobalDistance() || !EnsureLumenGlobalDistance()) return false;
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source");
    auto work=std::make_shared<LumenGlobalWork>(*this);
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second) return Fail("Global distance reuse crosses source generations");
        work->Previous=old->second;
    }
    if (!work->Prepare(found->second->Info,settings,work->Previous?work->Previous->Info.Global.get():nullptr,LastError) || !LumenGlobalTasks.CanSubmit(*work,LastError)) return false;
    const auto bytes=std::size_t(work->Plan->WordCount)*sizeof(std::uint32_t); std::int64_t maximum=0; DistanceGetInteger64(DistanceMaxStorageSize,&maximum);
    if (std::uint64_t(bytes)>std::uint64_t(maximum)) return Fail("Global distance scene exceeds ES storage range");
    work->Record=std::make_shared<LumenDistanceRecord>(*this); auto& record=*work->Record;
    record.Base=found->second; record.Info=record.Base->Info; record.Info.Global=work->Plan;
    LumenDistanceState state(*this);
    GlGenBuffers(1,&record.Buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record.Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),nullptr,GL_STATIC_DRAW);
    GlGenBuffers(1,&work->Dirty); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work->Dirty); const std::uint32_t empty=0;
    GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(std::max<std::size_t>(1,work->Update.Compose.size())*sizeof(std::uint32_t)),work->Update.Compose.empty()?&empty:work->Update.Compose.data(),GL_STATIC_DRAW);
    GlGenBuffers(1,&work->Uniform); GlBindBuffer(GL_UNIFORM_BUFFER,work->Uniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(LumenGlobalDistanceFrame),nullptr,GL_DYNAMIC_DRAW);
    if (!CheckGL("Allocate asynchronous global field")) return false;
    const auto task=LumenGlobalTasks.Insert(work); if (!task) return Fail("Global task space exhausted"); output=task; return true;
}
bool SubmitLumenGlobalDistanceSlice(LumenGlobalWork& work,const LumenGlobalDistanceFrame& frame)
{
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    const unsigned x=std::min(groups,LumenDistanceTraceColumns),y=(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns;
    GLint maxX=0,maxY=0; GpuGetIntegerIndexed(DistanceMaxGroups,0,&maxX); GpuGetIntegerIndexed(DistanceMaxGroups,1,&maxY);
    if (x>unsigned(maxX) || y>unsigned(maxY)) return Fail("Global slice exceeds ES dispatch limits");
    constexpr GLenum CopyRead=0x8F36,CopyWrite=0x8F37;
    struct CopyBindings
    {
        Impl& Owner; GLint Read=0,Write=0;
        explicit CopyBindings(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(CopyRead,&Read); Owner.GlGetIntegerv(CopyWrite,&Write); }
        ~CopyBindings() { Owner.GlBindBuffer(CopyRead,GLuint(Read)); Owner.GlBindBuffer(CopyWrite,GLuint(Write)); }
    } bindings(*this);
    LumenDistanceState state(*this); auto& record=*work.Record;
    if (!work.Initialized)
    {
        GpuMemoryBarrier(DistanceBufferUpdateBarrier); GlBindBuffer(CopyWrite,record.Buffer); GlBindBuffer(CopyRead,record.Base->Buffer);
        GlobalCopyBuffer(CopyRead,CopyWrite,0,0,std::max(1u,record.Info.Counts.W)*sizeof(std::uint32_t));
        GlBufferSubData(CopyWrite,GLintptr(frame.Header.Layout.W)*sizeof(std::uint32_t),sizeof(LumenGlobalDistanceHeader),&frame.Header);
        if (!work.Update.Copies.empty())
        {
            GlBindBuffer(CopyRead,work.Previous->Buffer);
            for (const auto& copy:work.Update.Copies) GlobalCopyBuffer(CopyRead,CopyWrite,GLintptr(copy.SourceWord)*sizeof(std::uint32_t),
                GLintptr(copy.DestinationWord)*sizeof(std::uint32_t),GLsizeiptr(copy.WordCount)*sizeof(std::uint32_t));
        }
    }
    GlBindBuffer(GL_UNIFORM_BUFFER,work.Uniform); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(frame),&frame);
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Uniform); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,record.Base->Buffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,work.Dirty); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,record.Buffer);
    GlUseProgram(LumenGlobalProgram); if (groups) GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier|DistanceBufferUpdateBarrier);
    if (work.Completion) GlobalDeleteSync(work.Completion);
    work.Completion=GlobalFenceSync(GlobalSyncComplete,0);
    if (!work.Completion) return Fail("Create global distance completion fence failed");
    work.InFlight=true; GlobalFlush(); return CheckGL("Submit asynchronous global slice");
}
bool AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    if (!CollectLumenGlobalDistance()) return false;
    return LumenGlobalTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenGlobalDistance(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenGlobalDistanceSlice(work,frame); },
        [&](auto& work,LumenDistanceSceneHandle& handle)
        {
            handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Global scene handle space exhausted");
            LumenDistanceScenes.emplace(handle.Value,work.Record); return true;
        });
}
