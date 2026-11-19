GLuint LumenGlobalProgram=0;
void(EVANS_GL_CALL* GlobalCopyBuffer)(GLenum,GLenum,GLintptr,GLintptr,GLsizeiptr)=nullptr;
void(EVANS_GL_CALL* GlobalFinish)()=nullptr;
void DestroyLumenGlobalDistance() { DestroyLumenGlobalDistanceAsync(); if (LumenGlobalProgram) GlDeleteProgram(LumenGlobalProgram); LumenGlobalProgram=0; }
bool EnsureLumenGlobalDistance()
{
    if (!EnsureLumenDistanceTrace() || !LoadGL(GlobalCopyBuffer,"glCopyBufferSubData") || !LoadGL(GlobalFinish,"glFinish")) return false;
    if (LumenGlobalProgram) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","LumenGlobalDistance.glsl",source,LastError)) return false;
    const auto program=CompileComputeProgram(source); if(!program)return false;
    LumenGlobalProgram=program; return true;
}
bool CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source scene");
    std::shared_ptr<LumenDistanceRecord> retained;
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second)
            return Fail("Global distance reuse requires the exact immutable source scene");
        retained=old->second;
    }
    auto plan=std::make_shared<LumenGlobalDistancePlan>();
    if (!BuildLumenGlobalDistancePlan(found->second->Info,settings,*plan,LastError) || !EnsureLumenGlobalDistance()) return false;
    LumenGlobalDistanceUpdate update;
    if (!BuildLumenGlobalDistanceUpdate(found->second->Info,*plan,retained?retained->Info.Global.get():nullptr,update,LastError)) return false;
    auto frame=plan->Frame; frame.Dispatch={0,unsigned(update.Compose.size()),0,0};
    const auto bytes=std::size_t(plan->WordCount)*sizeof(std::uint32_t); std::int64_t maximum=0; DistanceGetInteger64(DistanceMaxStorageSize,&maximum);
    if (std::uint64_t(bytes)>std::uint64_t(maximum)) return Fail("Global distance scene exceeds ES storage range");
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    const unsigned x=std::min(groups,LumenDistanceTraceColumns),y=(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns;
    GLint maxX=0,maxY=0; GpuGetIntegerIndexed(DistanceMaxGroups,0,&maxX); GpuGetIntegerIndexed(DistanceMaxGroups,1,&maxY);
    if (x>unsigned(maxX) || y>unsigned(maxY)) return Fail("Global distance dispatch exceeds ES group limits");
    auto candidate=std::make_shared<LumenDistanceRecord>(*this); candidate->Base=found->second; candidate->Info=candidate->Base->Info; candidate->Info.Global=plan;
    constexpr GLenum CopyRead=0x8F36,CopyWrite=0x8F37;
    struct Resources
    {
        Impl& Owner; GLint Read=0,Write=0; GLuint Uniform=0,Dirty=0;
        explicit Resources(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(CopyRead,&Read); Owner.GlGetIntegerv(CopyWrite,&Write); }
        ~Resources() { Owner.GlBindBuffer(CopyRead,GLuint(Read)); Owner.GlBindBuffer(CopyWrite,GLuint(Write)); if (Uniform) Owner.GlDeleteBuffers(1,&Uniform); if (Dirty) Owner.GlDeleteBuffers(1,&Dirty); }
    } resources(*this);
    LumenDistanceState state(*this);
    GlGenBuffers(1,&candidate->Buffer); GlBindBuffer(CopyWrite,candidate->Buffer); GlBufferData(CopyWrite,GLsizeiptr(bytes),nullptr,GL_STATIC_DRAW);
    GpuMemoryBarrier(DistanceBufferUpdateBarrier);
    if (!update.Copies.empty())
    {
        GlBindBuffer(CopyRead,retained->Buffer);
        for (const auto& copy:update.Copies) GlobalCopyBuffer(CopyRead,CopyWrite,GLintptr(copy.SourceWord)*sizeof(std::uint32_t),
            GLintptr(copy.DestinationWord)*sizeof(std::uint32_t),GLsizeiptr(copy.WordCount)*sizeof(std::uint32_t));
    }
    GlBindBuffer(CopyRead,candidate->Base->Buffer);
    GlobalCopyBuffer(CopyRead,CopyWrite,0,0,std::max(1u,candidate->Info.Counts.W)*sizeof(std::uint32_t));
    GlBufferSubData(CopyWrite,plan->Frame.Header.Layout.W*sizeof(std::uint32_t),sizeof(LumenGlobalDistanceHeader),&plan->Frame.Header);
    GlGenBuffers(1,&resources.Uniform); GlBindBuffer(GL_UNIFORM_BUFFER,resources.Uniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW);
    GlGenBuffers(1,&resources.Dirty); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,resources.Dirty);
    const std::uint32_t empty=0;
    GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(std::max<std::size_t>(1,update.Compose.size())*sizeof(std::uint32_t)),update.Compose.empty()?&empty:update.Compose.data(),GL_STATIC_DRAW);
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,resources.Uniform); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,candidate->Base->Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,resources.Dirty); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,candidate->Buffer);
    if (!CheckGL("Allocate global distance scene")) return false;
    GlUseProgram(LumenGlobalProgram); if (groups) GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier|DistanceBufferUpdateBarrier); GlobalFinish();
    if (!CheckGL("Compose global distance scene")) return false;
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,std::move(candidate)); output=handle; return true;
}
