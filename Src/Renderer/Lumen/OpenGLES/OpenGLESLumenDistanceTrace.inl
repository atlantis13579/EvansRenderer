struct LumenDistanceRecord
{
    Impl& Owner; LumenDistanceSceneInfo Info; std::shared_ptr<LumenDistanceRecord> Base; GLuint Buffer=0;
    explicit LumenDistanceRecord(Impl& owner):Owner(owner) {}
    ~LumenDistanceRecord() { if (Buffer) Owner.GlDeleteBuffers(1,&Buffer); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenDistanceRecord>> LumenDistanceScenes;
GLuint LumenDistanceProgram=0;
void*(EVANS_GL_CALL* DistanceMapBuffer)(GLenum,GLintptr,GLsizeiptr,GLbitfield)=nullptr;
GLboolean(EVANS_GL_CALL* DistanceUnmapBuffer)(GLenum)=nullptr;
void(EVANS_GL_CALL* DistanceGetRange)(GLenum,GLuint,std::int64_t*)=nullptr;
void(EVANS_GL_CALL* DistanceGetInteger64)(GLenum,std::int64_t*)=nullptr;
void(EVANS_GL_CALL* DistanceBindRange)(GLenum,GLuint,GLuint,GLintptr,GLsizeiptr)=nullptr;
static constexpr GLenum DistanceMaxStorageSize=0x90DE,DistanceMaxComputeStorage=0x90DB,DistanceMaxGroups=0x91BE;
static constexpr GLenum DistanceUniformStart=0x8A29,DistanceUniformSize=0x8A2A,DistanceStorageStart=0x90D4,DistanceStorageSize=0x90D5;
static constexpr GLbitfield DistanceBufferUpdateBarrier=0x0200,DistanceReadAccess=0x0001;
void DestroyLumenDistanceScenes()
{ LumenDistanceScenes.clear(); DestroyLumenGlobalDistance(); if (LumenDistanceProgram) GlDeleteProgram(LumenDistanceProgram); LumenDistanceProgram=0; }
bool EnsureLumenDistanceTrace()
{
    GLint major=0,minor=0; GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    if (major<3 || (major==3 && minor<1)) return Fail("Lumen distance tracing requires OpenGL ES 3.1");
    if (!LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier") || !LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") ||
        !LoadGL(DistanceMapBuffer,"glMapBufferRange") || !LoadGL(DistanceUnmapBuffer,"glUnmapBuffer") || !LoadGL(DistanceGetRange,"glGetInteger64i_v") ||
        !LoadGL(DistanceGetInteger64,"glGetInteger64v") || !LoadGL(DistanceBindRange,"glBindBufferRange")) return false;
    GLint bindings=0; GlGetIntegerv(DistanceMaxComputeStorage,&bindings); if (bindings<3) return Fail("Lumen distance tracing requires three compute storage bindings");
    if (LumenDistanceProgram) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
    const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen distance shader has no GLSL version");
    source.insert(version+1,"#define EVANS_LUMEN_DISTANCE_TRACE_STAGE 1\n");
    const auto program=CompileComputeProgram(source); if(!program)return false;
    LumenDistanceProgram=program; return true;
}
struct LumenDistanceState
{
    Impl& Owner; GLint Program=0,GenericUniform=0,GenericStorage=0,Uniform=0; std::array<GLint,3> Storage{};
    std::int64_t UniformStart=0,UniformSize=0; std::array<std::int64_t,3> StorageStart{},StorageSize{};
    explicit LumenDistanceState(Impl& owner):Owner(owner)
    {
        Owner.GlGetIntegerv(GpuCurrentProgram,&Program); Owner.GlGetIntegerv(GpuUniformBinding,&GenericUniform); Owner.GlGetIntegerv(GpuStorageBinding,&GenericStorage);
        Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform); Owner.DistanceGetRange(DistanceUniformStart,0,&UniformStart); Owner.DistanceGetRange(DistanceUniformSize,0,&UniformSize);
        for (unsigned slot=0;slot<Storage.size();++slot)
        { Owner.GpuGetIntegerIndexed(GpuStorageBinding,slot,&Storage[slot]); Owner.DistanceGetRange(DistanceStorageStart,slot,&StorageStart[slot]); Owner.DistanceGetRange(DistanceStorageSize,slot,&StorageSize[slot]); }
    }
    ~LumenDistanceState()
    {
        Owner.GlUseProgram(Program);
        if (Uniform && UniformSize) Owner.DistanceBindRange(GL_UNIFORM_BUFFER,0,Uniform,GLintptr(UniformStart),GLsizeiptr(UniformSize));
        else Owner.GlBindBufferBase(GL_UNIFORM_BUFFER,0,Uniform);
        for (unsigned slot=0;slot<Storage.size();++slot)
            if (Storage[slot] && StorageSize[slot]) Owner.DistanceBindRange(GL_SHADER_STORAGE_BUFFER,slot,Storage[slot],GLintptr(StorageStart[slot]),GLsizeiptr(StorageSize[slot]));
            else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,Storage[slot]);
        Owner.GlBindBuffer(GL_UNIFORM_BUFFER,GenericUniform); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GenericStorage);
    }
};
bool CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!ValidateLumenDistanceScene(data,LastError) || !EnsureLumenDistanceTrace()) return false;
    const std::uint32_t empty=0; const auto bytes=std::max<std::size_t>(1,data.Words.size())*sizeof(std::uint32_t);
    std::int64_t maximum=0; DistanceGetInteger64(DistanceMaxStorageSize,&maximum);
    if (std::uint64_t(bytes)>std::uint64_t(maximum)) return Fail("Lumen distance scene exceeds ES storage size");
    auto candidate=std::make_shared<LumenDistanceRecord>(*this); candidate->Info=data.Info;
    GLint previous=0; GlGetIntegerv(GpuStorageBinding,&previous); GlGenBuffers(1,&candidate->Buffer);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,candidate->Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),data.Words.empty()?&empty:data.Words.data(),GL_STATIC_DRAW);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,previous); if (!CheckGL("Upload immutable Lumen distance scene")) return false;
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,std::move(candidate)); output=handle; return true;
}
bool TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    const auto found=LumenDistanceScenes.find(handle.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown Lumen distance scene");
    LumenDistanceTraceFrame frame; if (!BuildLumenDistanceTraceFrame(found->second->Info,upload,frame,LastError) || !EnsureLumenDistanceTrace()) return false;
    const auto bytes=upload.Rays.size()*sizeof(LumenDistanceHit); std::int64_t maximum=0; DistanceGetInteger64(DistanceMaxStorageSize,&maximum);
    if (std::uint64_t(bytes)>std::uint64_t(maximum)) return Fail("Lumen distance feedback exceeds ES storage size");
    const auto groups=unsigned((upload.Rays.size()+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads);
    const unsigned x=std::min(groups,LumenDistanceTraceColumns),y=(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns;
    GLint maxX=0,maxY=0; GpuGetIntegerIndexed(DistanceMaxGroups,0,&maxX); GpuGetIntegerIndexed(DistanceMaxGroups,1,&maxY);
    if (x>unsigned(maxX) || y>unsigned(maxY)) return Fail("Lumen distance dispatch exceeds ES group counts");
    struct BuffersOwner { Impl& Owner; std::array<GLuint,3> Values{}; ~BuffersOwner() { Owner.GlDeleteBuffers(3,Values.data()); } } buffers{*this};
    LumenDistanceState state(*this); GlGenBuffers(3,buffers.Values.data());
    GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Values[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Values[0]);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers.Values[1]); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(upload.Rays.size()*sizeof(LumenDistanceRay)),upload.Rays.data(),GL_STATIC_DRAW);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,found->second->Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,buffers.Values[1]);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers.Values[2]); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),nullptr,GL_STATIC_DRAW); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,buffers.Values[2]);
    if (!CheckGL("Allocate Lumen distance queries")) return false;
    GlUseProgram(LumenDistanceProgram); GpuDispatch(x,y,1); GpuMemoryBarrier(DistanceBufferUpdateBarrier);
    if (!CheckGL("Dispatch Lumen distance rays")) return false;
    LumenDistanceTraceResult result; result.Identity=found->second->Info.Identity; result.Hits.resize(upload.Rays.size());
    const auto* mapped=DistanceMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),DistanceReadAccess);
    if (!mapped) return Fail("Map Lumen distance feedback failed");
    std::memcpy(result.Hits.data(),mapped,bytes); const auto valid=DistanceUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (!valid || !CheckGL("Read Lumen distance feedback")) return Fail("Lumen distance feedback became invalid");
    if (!ValidateLumenDistanceTraceResult(found->second->Info,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
bool DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{ return LumenDistanceScenes.erase(handle.Value)!=0 || Fail("Unknown Lumen distance scene"); }
