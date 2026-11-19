// ES fence polling always uses timeout zero. A separate binding scope restores
// the caller's indexed ranges immediately, not when a delayed job is destroyed.
static constexpr GLenum NaniteSyncGpuCommandsComplete=0x9117,NaniteAlreadySignaled=0x911A,NaniteTimeoutExpired=0x911B,NaniteConditionSatisfied=0x911C;
void*(EVANS_GL_CALL* NaniteFenceSync)(GLenum,GLbitfield)=nullptr;
GLenum(EVANS_GL_CALL* NaniteClientWaitSync)(void*,GLbitfield,std::uint64_t)=nullptr;
void(EVANS_GL_CALL* NaniteDeleteSync)(void*)=nullptr;
void(EVANS_GL_CALL* NaniteFlush)()=nullptr;
struct NaniteAsyncWork
{
    Impl& Owner; NaniteGpuUpload Upload; NaniteAsyncPlan Plan;
    std::array<GLuint,7> Buffers{}; std::shared_ptr<NaniteCachePool> Borrowed; void* Completion=nullptr;
    explicit NaniteAsyncWork(Impl& owner):Owner(owner) {}
    ~NaniteAsyncWork()
    { if (Completion) Owner.NaniteDeleteSync(Completion); if (Borrowed) Buffers[3]=0; Owner.GlDeleteBuffers(GLsizei(Buffers.size()),Buffers.data()); }
};
struct NaniteAsyncBindings
{
    Impl& Owner; std::array<GLint,6> Storage{}; std::array<std::int64_t,6> Starts{},Sizes{};
    GLint Uniform=0,GenericUniform=0,GenericStorage=0,Program=0; std::int64_t UniformStart=0,UniformSize=0;
    explicit NaniteAsyncBindings(Impl& owner):Owner(owner)
    {
        for (GLuint i=0;i<Storage.size();++i)
        { Owner.GpuGetIntegerIndexed(GpuStorageBinding,i,&Storage[i]); Owner.NaniteGetRange(NaniteStorageStart,i,&Starts[i]); Owner.NaniteGetRange(NaniteStorageSize,i,&Sizes[i]); }
        Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform); Owner.NaniteGetRange(NaniteUniformStart,0,&UniformStart); Owner.NaniteGetRange(NaniteUniformSize,0,&UniformSize);
        Owner.GlGetIntegerv(GpuUniformBinding,&GenericUniform); Owner.GlGetIntegerv(GpuStorageBinding,&GenericStorage); Owner.GlGetIntegerv(GpuCurrentProgram,&Program);
    }
    ~NaniteAsyncBindings()
    {
        for (GLuint i=0;i<Storage.size();++i)
            if (Storage[i] && Sizes[i]>0) Owner.NaniteBindRange(GL_SHADER_STORAGE_BUFFER,i,GLuint(Storage[i]),GLintptr(Starts[i]),GLsizeiptr(Sizes[i]));
            else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,i,GLuint(Storage[i]));
        if (Uniform && UniformSize>0) Owner.NaniteBindRange(GL_UNIFORM_BUFFER,0,GLuint(Uniform),GLintptr(UniformStart),GLsizeiptr(UniformSize));
        else Owner.GlBindBufferBase(GL_UNIFORM_BUFFER,0,GLuint(Uniform));
        Owner.GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(GenericUniform)); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GLuint(GenericStorage)); Owner.GlUseProgram(GLuint(Program));
    }
};
NaniteAsyncQueue<NaniteAsyncWork> NaniteAsyncTasks;
GLuint NaniteWireProgram=0,NaniteWireUniform=0;
void DestroyNaniteAsync()
{
    NaniteAsyncTasks.Clear();
    if (NaniteWireProgram) GlDeleteProgram(NaniteWireProgram);
    if (NaniteWireUniform) GlDeleteBuffers(1,&NaniteWireUniform);
    NaniteWireProgram=NaniteWireUniform=0;
}
bool PrepareNaniteWireIndices(GLuint source,unsigned first,unsigned count)
{
    if (!count) return true;
    const auto bytes=std::uint64_t(count)*2*sizeof(unsigned);
    if (bytes>std::uint64_t(NaniteMaximumBlock)) return Fail("Nanite wire indices exceed the ES storage block budget");
    if (!NaniteWireProgram)
    {
        std::string code; if (!LoadTechniqueShaderSource("OpenGLES","NaniteGeometry.glsl",code,LastError)) return false;
        const auto line=code.find('\n'); if (line==std::string::npos) return Fail("Nanite edge shader has no version directive");
        code.insert(line+1,"#define EVANS_NANITE_WIRE_EDGES 1\n");
        const auto program=CompileComputeProgram(code); if(!program)return false;
        NaniteWireProgram=program;
    }
    NaniteAsyncBindings restore(*this);
    if (!NaniteWireUniform) GlGenBuffers(1,&NaniteWireUniform);
    const auto triangles=count/static_cast<unsigned>(VisibilityTriangleCorners);
    const auto groups=(triangles+NaniteGpuWidth-1)/NaniteGpuWidth;
    const auto x=std::min(groups,unsigned(NaniteMaximumGroupsX)),y=(groups+x-1)/x;
    if (y>unsigned(NaniteMaximumGroupsY)) return Fail("Nanite wire dispatch exceeds ES workgroup limits");
    const ShaderUInt4 constants{first,count,x*NaniteGpuWidth,0};
    GlBindBuffer(GL_UNIFORM_BUFFER,NaniteWireUniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(constants),&constants,GL_DYNAMIC_DRAW);
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,NaniteWireUniform);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,ScratchIndexBuffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),nullptr,GL_DYNAMIC_DRAW);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,source); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,ScratchIndexBuffer);
    GlUseProgram(NaniteWireProgram); GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier|NaniteIndexBarrier);
    return CheckGL("Generate Nanite wire indices");
}
bool ProbeNaniteAsync(NaniteAsyncWork& work,bool& ready)
{
    const auto status=NaniteClientWaitSync(work.Completion,0,0);
    ready=status==NaniteAlreadySignaled || status==NaniteConditionSatisfied;
    return ready || status==NaniteTimeoutExpired || Fail("ES Nanite completion query failed");
}
bool CollectNaniteAsync() { return NaniteAsyncTasks.Collect([&](auto& work,bool& ready){return ProbeNaniteAsync(work,ready);}); }
bool CancelNaniteGeometry(NaniteGeometryTask task)
{ return (NaniteAsyncTasks.Cancel(task) || Fail("Invalid Nanite task")) && CollectNaniteAsync(); }
bool SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& output)
{
    if (!LoadGL(NaniteFenceSync,"glFenceSync") || !LoadGL(NaniteClientWaitSync,"glClientWaitSync") || !LoadGL(NaniteDeleteSync,"glDeleteSync") || !LoadGL(NaniteFlush,"glFlush")) return false;
    auto work=std::make_shared<NaniteAsyncWork>(*this);
    if (!CollectNaniteAsync() || !BuildNaniteAsyncPlan(upload,work->Plan,LastError) || !NaniteAsyncTasks.CanSubmit(work->Plan,LastError) || !EnsureNaniteGeometry()) return false;
    if (cache) { auto found=NaniteCaches.find(cache.Value); if (found==NaniteCaches.end()) return Fail("Invalid Nanite cache"); if (!MatchNaniteCache(found->second.Info,upload,LastError)) return false; work->Borrowed=found->second.Pool; }
    for (const auto& job:work->Plan.Dispatches) if (job.GroupsX>unsigned(NaniteMaximumGroupsX) || job.GroupsY>unsigned(NaniteMaximumGroupsY)) return Fail("Nanite async dispatch exceeds ES limits");
    work->Upload=upload;
    const void* data[]{nullptr,upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data(),nullptr,nullptr};
    const std::size_t sizes[]{sizeof(NaniteGpuConstants),upload.Metadata.size()*sizeof(unsigned),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage),upload.InitialState.size()*sizeof(unsigned),work->Plan.VertexBytes,work->Plan.IndexBytes};
    for (unsigned i=1;i<7;++i) if (sizes[i]>std::size_t(NaniteMaximumBlock)) return Fail("Nanite async storage exceeds ES limits");
    NaniteAsyncBindings restore(*this); GlGenBuffers(GLsizei(work->Buffers.size()),work->Buffers.data());
    for (GLuint i=0;i<work->Buffers.size();++i)
    {
        const GLenum target=i?GL_SHADER_STORAGE_BUFFER:GL_UNIFORM_BUFFER;
        if (i==3 && work->Borrowed) { GlDeleteBuffers(1,&work->Buffers[i]); work->Buffers[i]=work->Borrowed->Buffer; }
        else { GlBindBuffer(target,work->Buffers[i]); GlBufferData(target,GLsizeiptr(sizes[i]),data[i],GL_STATIC_DRAW); }
        GlBindBufferBase(target,i?i-1:0,work->Buffers[i]);
    }
    if (!CheckGL("Allocate async Nanite buffers")) return false;
    const auto task=NaniteAsyncTasks.Insert(work);
    for (const auto& job:work->Plan.Dispatches)
    {
        GlBindBuffer(GL_UNIFORM_BUFFER,work->Buffers[0]); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(job.Constants),&job.Constants);
        GlUseProgram(NaniteGeometryPrograms[unsigned(job.Pass)]); GpuDispatch(job.GroupsX,job.GroupsY,1); GpuMemoryBarrier(GpuStorageBarrier);
    }
    GpuMemoryBarrier(GpuStorageBarrier|NaniteVertexBarrier|NaniteIndexBarrier|NaniteBufferBarrier);
    work->Completion=NaniteFenceSync(NaniteSyncGpuCommandsComplete,0); NaniteFlush();
    if (!work->Completion || !CheckGL("Submit async Nanite geometry")) { NaniteAsyncTasks.Erase(task); return false; }
    output=task; return true;
}
bool PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    ready=false; auto work=NaniteAsyncTasks.Find(task); if (!work) return Fail("Invalid Nanite task");
    bool complete=false; if (!ProbeNaniteAsync(*work,complete)) return false; if (!complete) return true;
    NaniteAsyncBindings restore(*this);
    const auto read=[&](unsigned i,void* target,std::size_t bytes)
    {
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work->Buffers[i+4]); const void* mapped=NaniteMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),NaniteMapRead);
        if (!mapped) return Fail("Map completed ES Nanite result failed"); std::memcpy(target,mapped,bytes);
        return NaniteUnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=0 && CheckGL("Read completed Nanite result");
    };
    NaniteGeometryBuffers result; result.Readback.State.resize(work->Upload.InitialState.size());
    if (!read(0,result.Readback.State.data(),result.Readback.State.size()*sizeof(unsigned)) || !ValidateNaniteGpuState(work->Upload,result.Readback.State,result.Selection,LastError)) return false;
    const auto vertexBytes=std::size_t(result.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(result.Selection.IndexCount)*sizeof(unsigned);
    if (work->Upload.ReadbackGeometry)
    {
        result.Readback.Vertices.resize(result.Selection.VertexCount); result.Readback.Indices.resize(result.Selection.IndexCount);
        if (!read(1,result.Readback.Vertices.data(),vertexBytes) || !read(2,result.Readback.Indices.data(),indexBytes) || !ValidateNaniteGpuReadback(work->Upload,result.Readback,result.Selection,LastError)) return false;
    }
    std::array<BufferResource,2> resources; const void* mirrors[]{result.Readback.Vertices.data(),result.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite async vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite async indices"};
    for (unsigned i=0;i<2;++i)
    { auto& resource=resources[i]; resource.NaniteGeometry=true; resource.Buffer=work->Buffers[i+5]; if (work->Upload.ReadbackGeometry) { resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[i],resource.Bytes.size()); } }
    result.VertexBuffer={NextBuffer++}; result.IndexBuffer={NextBuffer++}; result.DispatchCount=unsigned(work->Plan.Dispatches.size());
    try { Buffers.emplace(result.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(result.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(result.VertexBuffer.Value); throw; }
    work->Buffers[5]=work->Buffers[6]=0; NaniteAsyncTasks.Erase(task); output=std::move(result); ready=true; return true;
}
