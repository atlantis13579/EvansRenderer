// ES 3.1 uses UBO 0 and six separate SSBO bindings. Preserve exact external
// ranges/program bindings on every exit. GPU prefix feedback sizes the drawable
// buffers; mapped bytes are authoritative GPU results, not replacement geometry.
static constexpr GLenum VirtualComputeBlocks=0x90DB,VirtualGroupCount=0x91BE,VirtualGroupSize=0x91BF,VirtualInvocations=0x90EB;
static constexpr GLenum VirtualStorageStart=0x90D4,VirtualStorageSize=0x90D5,VirtualUniformStart=0x8A29,VirtualUniformSize=0x8A2A;
static constexpr GLbitfield VirtualVertexBarrier=0x00000001,VirtualIndexBarrier=0x00000002,VirtualBufferBarrier=0x00000200,VirtualMapRead=0x00000001;
bool VirtualGeometrySupported=false;
GLint VirtualMaximumBlock=0,VirtualMaximumGroupsX=0,VirtualMaximumGroupsY=0;
std::array<GLuint,6> VirtualGeometryPrograms{};
void*(EVANS_GL_CALL* VirtualMapBuffer)(GLenum,GLintptr,GLsizeiptr,GLbitfield)=nullptr;
GLboolean(EVANS_GL_CALL* VirtualUnmapBuffer)(GLenum)=nullptr;
void(EVANS_GL_CALL* VirtualGetRange)(GLenum,GLuint,std::int64_t*)=nullptr;
void(EVANS_GL_CALL* VirtualBindRange)(GLenum,GLuint,GLuint,GLintptr,GLsizeiptr)=nullptr;
bool ProbeVirtualGeometry()
{
    GLint major=0,minor=0,blocks=0,bindings=0,invocations=0,width=0;
    GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor); if (major<3 || (major==3 && minor<1)) return false;
    if (!LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier") || !LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") ||
        !LoadGL(VirtualMapBuffer,"glMapBufferRange") || !LoadGL(VirtualUnmapBuffer,"glUnmapBuffer") || !LoadGL(VirtualGetRange,"glGetInteger64i_v") || !LoadGL(VirtualBindRange,"glBindBufferRange")) return false;
    GlGetIntegerv(VirtualComputeBlocks,&blocks); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&VirtualMaximumBlock);
    GlGetIntegerv(VirtualInvocations,&invocations); GpuGetIntegerIndexed(VirtualGroupSize,0,&width);
    GpuGetIntegerIndexed(VirtualGroupCount,0,&VirtualMaximumGroupsX); GpuGetIntegerIndexed(VirtualGroupCount,1,&VirtualMaximumGroupsY);
    return GlGetError()==GL_NO_ERROR && blocks>=6 && bindings>=6 && invocations>=VirtualGeometryGroupWidth && width>=VirtualGeometryGroupWidth && VirtualMaximumBlock>0 && VirtualMaximumGroupsX>0 && VirtualMaximumGroupsY>0;
}
void DestroyVirtualGeometry()
{ DestroyVirtualGeometryCaches(); for (auto& program:VirtualGeometryPrograms) { if (program && GlDeleteProgram) GlDeleteProgram(program); program=0; } }
bool EnsureVirtualGeometry()
{
    if (!VirtualGeometrySupported) return Fail("ES virtual geometry requires compute and six storage blocks");
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","VirtualGeometry.glsl",source,LastError)) return false;
    const auto versionEnd=source.find('\n'); if (versionEnd==std::string::npos) return Fail("ES virtual geometry has no version line");
    for (std::size_t stage=0;stage<VirtualGeometryPrograms.size();++stage) if (!VirtualGeometryPrograms[stage])
    {
        auto text=source; text.insert(versionEnd+1,"#define EVANS_VIRTUAL_GEOMETRY_PASS "+std::to_string(stage)+"\n");
        const auto program=CompileComputeProgram(text); if(!program)return false;
        VirtualGeometryPrograms[stage]=program;
    }
    return true;
}
bool CreateVirtualGeometry(const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output,VirtualGeometryCacheHandle cache={})
{
    std::vector<VirtualGeometryGpuDispatch> jobs;
    if (!BuildVirtualGeometryGpuSchedule(upload,jobs,LastError) || !EnsureVirtualGeometry()) return false;
    const VirtualGeometryNativeCache* cached=nullptr;
    if (cache) { const auto found=VirtualGeometryCaches.find(cache.Value); if (found==VirtualGeometryCaches.end()) return Fail("ES virtual cache handle is invalid"); cached=&found->second; if (!MatchVirtualGeometryCache(cached->Info,upload,LastError)) return false; }
    for (const auto& job:jobs) if (job.GroupsX>static_cast<unsigned>(VirtualMaximumGroupsX) || job.GroupsY>static_cast<unsigned>(VirtualMaximumGroupsY)) return Fail("Virtual geometry exceeds ES group dimensions");
    const std::size_t stateBytes=upload.InitialState.size()*sizeof(std::uint32_t);
    const void* data[]{nullptr,upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data(),nullptr,nullptr};
    const std::size_t sizes[]{sizeof(VirtualGeometryGpuConstants),upload.Metadata.size()*sizeof(std::uint32_t),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(VirtualGeometryPage),stateBytes,sizeof(std::uint32_t),sizeof(std::uint32_t)};
    for (std::size_t slot=1;slot<7;++slot) if (sizes[slot]>static_cast<std::size_t>(VirtualMaximumBlock)) return Fail("Virtual geometry exceeds ES storage range");
    struct Work
    {
        Impl& Owner; std::array<GLuint,7> Buffers{}; std::array<GLint,6> Storage{}; std::array<std::int64_t,6> Starts{},Sizes{};
        bool BorrowedPool=false;
        GLint Uniform=0,GenericUniform=0,GenericStorage=0,Program=0; std::int64_t UniformStart=0,UniformSize=0;
        explicit Work(Impl& owner):Owner(owner)
        {
            for (GLuint slot=0;slot<Storage.size();++slot)
            { Owner.GpuGetIntegerIndexed(GpuStorageBinding,slot,&Storage[slot]); Owner.VirtualGetRange(VirtualStorageStart,slot,&Starts[slot]); Owner.VirtualGetRange(VirtualStorageSize,slot,&Sizes[slot]); }
            Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform); Owner.VirtualGetRange(VirtualUniformStart,0,&UniformStart); Owner.VirtualGetRange(VirtualUniformSize,0,&UniformSize);
            Owner.GlGetIntegerv(GpuUniformBinding,&GenericUniform); Owner.GlGetIntegerv(GpuStorageBinding,&GenericStorage); Owner.GlGetIntegerv(GpuCurrentProgram,&Program);
        }
        ~Work()
        {
            for (GLuint slot=0;slot<Storage.size();++slot)
                if (Storage[slot] && Sizes[slot]>0) Owner.VirtualBindRange(GL_SHADER_STORAGE_BUFFER,slot,static_cast<GLuint>(Storage[slot]),static_cast<GLintptr>(Starts[slot]),static_cast<GLsizeiptr>(Sizes[slot]));
                else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,static_cast<GLuint>(Storage[slot]));
            if (Uniform && UniformSize>0) Owner.VirtualBindRange(GL_UNIFORM_BUFFER,0,static_cast<GLuint>(Uniform),static_cast<GLintptr>(UniformStart),static_cast<GLsizeiptr>(UniformSize));
            else Owner.GlBindBufferBase(GL_UNIFORM_BUFFER,0,static_cast<GLuint>(Uniform));
            Owner.GlBindBuffer(GL_UNIFORM_BUFFER,static_cast<GLuint>(GenericUniform)); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,static_cast<GLuint>(GenericStorage)); Owner.GlUseProgram(static_cast<GLuint>(Program));
            if (BorrowedPool) Buffers[3]=0;
            Owner.GlDeleteBuffers(static_cast<GLsizei>(Buffers.size()),Buffers.data());
        }
    } work(*this);
    GlGenBuffers(static_cast<GLsizei>(work.Buffers.size()),work.Buffers.data());
    for (GLuint slot=0;slot<work.Buffers.size();++slot)
    {
        if (slot==3 && cached) { GlDeleteBuffers(1,&work.Buffers[slot]); work.Buffers[slot]=cached->Pool->Buffer; work.BorrowedPool=true; GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot-1,work.Buffers[slot]); continue; }
        const GLenum target=slot?GL_SHADER_STORAGE_BUFFER:GL_UNIFORM_BUFFER;
        GlBindBuffer(target,work.Buffers[slot]); GlBufferData(target,static_cast<GLsizeiptr>(sizes[slot]),data[slot],GL_STATIC_DRAW); GlBindBufferBase(target,slot?slot-1:0,work.Buffers[slot]);
    }
    if (!CheckGL("Allocate virtual geometry storage")) return false;
    auto dispatch=[&](const VirtualGeometryGpuDispatch& job,const VirtualGeometryGpuConstants& constants)
    {
        GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[0]); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants);
        GlUseProgram(VirtualGeometryPrograms[static_cast<unsigned>(job.Pass)]); GpuDispatch(job.GroupsX,job.GroupsY,1); GpuMemoryBarrier(GpuStorageBarrier);
    };
    auto read=[&](unsigned slot,std::size_t bytes,void* destination)
    {
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work.Buffers[slot]); const void* mapped=VirtualMapBuffer(GL_SHADER_STORAGE_BUFFER,0,static_cast<GLsizeiptr>(bytes),VirtualMapRead);
        if (!mapped) return Fail("ES virtual geometry mapping failed"); std::memcpy(destination,mapped,bytes);
        const bool valid=VirtualUnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=0;
        return CheckGL("Read virtual geometry") && (valid || Fail("ES virtual geometry readback invalidated"));
    };
    for (std::size_t stage=0;stage+1<jobs.size();++stage) dispatch(jobs[stage],jobs[stage].Constants);
    GpuMemoryBarrier(GpuStorageBarrier|VirtualBufferBarrier); if (!CheckGL("Dispatch virtual selection")) return false;
    VirtualGeometryBuffers candidate; candidate.Readback.State.resize(upload.InitialState.size());
    if (!read(4,stateBytes,candidate.Readback.State.data()) || !ValidateVirtualGeometryGpuState(upload,candidate.Readback.State,candidate.Selection,LastError)) return false;
    const std::size_t vertexBytes=std::size_t(candidate.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(candidate.Selection.IndexCount)*sizeof(std::uint32_t);
    if (vertexBytes>static_cast<std::size_t>(VirtualMaximumBlock) || indexBytes>static_cast<std::size_t>(VirtualMaximumBlock)) return Fail("Virtual drawable streams exceed ES storage range");
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work.Buffers[5]); GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(vertexBytes),nullptr,GL_STATIC_DRAW);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,4,work.Buffers[5]);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work.Buffers[6]); GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(indexBytes),nullptr,GL_STATIC_DRAW);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,5,work.Buffers[6]);
    auto constants=jobs.back().Constants; constants.Capacity.X=candidate.Selection.VertexCount; constants.Capacity.Y=candidate.Selection.IndexCount; dispatch(jobs.back(),constants);
    GpuMemoryBarrier(GpuStorageBarrier|VirtualVertexBarrier|VirtualIndexBarrier|VirtualBufferBarrier); if (!CheckGL("Decode virtual pages")) return false;
    candidate.Readback.Vertices.resize(candidate.Selection.VertexCount); candidate.Readback.Indices.resize(candidate.Selection.IndexCount);
    if (!read(4,stateBytes,candidate.Readback.State.data()) || !read(5,vertexBytes,candidate.Readback.Vertices.data()) || !read(6,indexBytes,candidate.Readback.Indices.data()) ||
        !ValidateVirtualGeometryGpuReadback(upload,candidate.Readback,candidate.Selection,LastError)) return false;
    std::array<BufferResource,2> resources; const void* mirrors[]{candidate.Readback.Vertices.data(),candidate.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Virtual geometry vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Virtual geometry indices"};
    for (std::size_t stream=0;stream<resources.size();++stream)
    { auto& resource=resources[stream]; resource.Buffer=work.Buffers[5+stream]; resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[stream],resource.Bytes.size()); }
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(jobs.size());
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    work.Buffers[5]=work.Buffers[6]=0; output=std::move(candidate); return true;
}
