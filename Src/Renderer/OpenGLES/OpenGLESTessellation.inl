// ES keeps uniform and storage binding namespaces separate: UBO 0 and SSBOs
// 0..7. Exact indexed ranges, generic bindings and the previous program are
// restored on every exit. Mapping waits for the authoritative GPU streams.
static constexpr GLenum TessellationComputeBlocks=0x90DB,TessellationGroupCount=0x91BE,TessellationGroupSize=0x91BF,TessellationInvocations=0x90EB;
static constexpr GLenum TessellationStorageStart=0x90D4,TessellationStorageSize=0x90D5,TessellationUniformStart=0x8A29,TessellationUniformSize=0x8A2A;
static constexpr GLbitfield TessellationVertexBarrier=0x00000001,TessellationIndexBarrier=0x00000002,TessellationBufferBarrier=0x00000200,TessellationMapRead=0x00000001;
bool TessellationSupported=false;
GLint TessellationMaximumBlock=0,TessellationMaximumGroups=0;
std::array<GLuint,3> TessellationPrograms{};
void*(EVANS_GL_CALL* TessellationMapBuffer)(GLenum,GLintptr,GLsizeiptr,GLbitfield)=nullptr;
GLboolean(EVANS_GL_CALL* TessellationUnmapBuffer)(GLenum)=nullptr;
void(EVANS_GL_CALL* TessellationGetRange)(GLenum,GLuint,std::int64_t*)=nullptr;
void(EVANS_GL_CALL* TessellationBindRange)(GLenum,GLuint,GLuint,GLintptr,GLsizeiptr)=nullptr;
bool ProbeTessellation()
{
    GLint major=0,minor=0,blocks=0,bindings=0,invocations=0,width=0;
    GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    if (major<3 || (major==3 && minor<1)) return false;
    if (!LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier") || !LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") ||
        !LoadGL(TessellationMapBuffer,"glMapBufferRange") || !LoadGL(TessellationUnmapBuffer,"glUnmapBuffer") ||
        !LoadGL(TessellationGetRange,"glGetInteger64i_v") || !LoadGL(TessellationBindRange,"glBindBufferRange")) return false;
    GlGetIntegerv(TessellationComputeBlocks,&blocks); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&TessellationMaximumBlock); GlGetIntegerv(TessellationInvocations,&invocations);
    GpuGetIntegerIndexed(TessellationGroupCount,0,&TessellationMaximumGroups); GpuGetIntegerIndexed(TessellationGroupSize,0,&width);
    return GlGetError()==GL_NO_ERROR && blocks>=8 && bindings>=8 && TessellationMaximumBlock>=sizeof(Vertex) &&
        invocations>=TessellationThreadCount && width>=TessellationThreadCount && TessellationMaximumGroups>0;
}
void DestroyTessellation()
{ for (auto& program:TessellationPrograms) { if (program && GlDeleteProgram) GlDeleteProgram(program); program=0; } }
bool EnsureTessellation()
{
    if (!TessellationSupported) return Fail("ES PN tessellation requires compute and eight storage blocks/bindings");
    if (TessellationPrograms[0] && TessellationPrograms[1] && TessellationPrograms[2]) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Tessellation.glsl",source,LastError)) return false;
    const auto versionEnd=source.find('\n'); if (versionEnd==std::string::npos) return Fail("ES PN source is missing its version line");
    for (std::size_t phase=0;phase<TessellationPrograms.size();++phase) if (!TessellationPrograms[phase])
    {
        auto nativeSource=source; nativeSource.insert(versionEnd+1,"#define EVANS_TESSELLATION_PASS "+std::to_string(phase)+"\n");
        const auto program=CompileComputeProgram(nativeSource); if(!program)return false;
        TessellationPrograms[phase]=program;
    }
    return true;
}
bool CreateTessellated(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    TessellationConstants constants;
    if (!ValidateTessellationUpload(geometry,settings,constants,LastError) || !EnsureTessellation()) return false;
    const std::size_t vertexBytes=std::size_t(constants.Outputs.X)*sizeof(Vertex),indexBytes=std::size_t(constants.Outputs.Y)*sizeof(std::uint32_t);
    const void* data[]{&constants,geometry.Vertices.data(),geometry.Patches.data(),geometry.PatchEdges.data(),geometry.AttributeEdges.data(),geometry.GeometricEdges.data(),nullptr,nullptr,nullptr};
    const std::size_t sizes[]{sizeof(constants),geometry.Vertices.size()*sizeof(Vertex),geometry.Patches.size()*sizeof(ShaderUInt4),geometry.PatchEdges.size()*sizeof(ShaderUInt4),
        geometry.AttributeEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*2*sizeof(Float4),vertexBytes,indexBytes};
    for (std::size_t slot=1;slot<9;++slot) if (sizes[slot]>static_cast<std::size_t>(TessellationMaximumBlock)) return Fail("PN storage exceeds the ES storage-block range");
    const unsigned counts[]{constants.Inputs.W,constants.Outputs.X,constants.Outputs.Y/3};
    for (auto count:counts) if ((count+TessellationThreadCount-1)/TessellationThreadCount>static_cast<unsigned>(TessellationMaximumGroups)) return Fail("PN dispatch exceeds the ES workgroup limit");
    struct Work
    {
        Impl& Owner; std::array<GLuint,9> Buffers{}; std::array<GLint,8> Storage{};
        std::array<std::int64_t,8> Starts{},Sizes{}; std::int64_t UniformStart=0,UniformSize=0;
        GLint Uniform=0,GenericUniform=0,GenericStorage=0,Program=0;
        explicit Work(Impl& owner):Owner(owner)
        {
            for (GLuint slot=0;slot<Storage.size();++slot)
            {
                Owner.GpuGetIntegerIndexed(GpuStorageBinding,slot,&Storage[slot]);
                Owner.TessellationGetRange(TessellationStorageStart,slot,&Starts[slot]); Owner.TessellationGetRange(TessellationStorageSize,slot,&Sizes[slot]);
            }
            Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform);
            Owner.TessellationGetRange(TessellationUniformStart,0,&UniformStart); Owner.TessellationGetRange(TessellationUniformSize,0,&UniformSize);
            Owner.GlGetIntegerv(GpuUniformBinding,&GenericUniform); Owner.GlGetIntegerv(GpuStorageBinding,&GenericStorage); Owner.GlGetIntegerv(GpuCurrentProgram,&Program);
        }
        ~Work()
        {
            for (GLuint slot=0;slot<Storage.size();++slot)
                if (Storage[slot] && Sizes[slot]>0)
                    Owner.TessellationBindRange(GL_SHADER_STORAGE_BUFFER,slot,static_cast<GLuint>(Storage[slot]),static_cast<GLintptr>(Starts[slot]),static_cast<GLsizeiptr>(Sizes[slot]));
                else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,static_cast<GLuint>(Storage[slot]));
            if (Uniform && UniformSize>0) Owner.TessellationBindRange(GL_UNIFORM_BUFFER,0,static_cast<GLuint>(Uniform),static_cast<GLintptr>(UniformStart),static_cast<GLsizeiptr>(UniformSize));
            else Owner.GlBindBufferBase(GL_UNIFORM_BUFFER,0,static_cast<GLuint>(Uniform));
            Owner.GlBindBuffer(GL_UNIFORM_BUFFER,static_cast<GLuint>(GenericUniform)); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,static_cast<GLuint>(GenericStorage));
            Owner.GlUseProgram(static_cast<GLuint>(Program)); Owner.GlDeleteBuffers(static_cast<GLsizei>(Buffers.size()),Buffers.data());
        }
    } work(*this);
    GlGenBuffers(static_cast<GLsizei>(work.Buffers.size()),work.Buffers.data());
    for (GLuint slot=0;slot<work.Buffers.size();++slot)
    {
        const GLenum target=slot?GL_SHADER_STORAGE_BUFFER:GL_UNIFORM_BUFFER;
        GlBindBuffer(target,work.Buffers[slot]); GlBufferData(target,static_cast<GLsizeiptr>(sizes[slot]),data[slot],GL_STATIC_DRAW);
        GlBindBufferBase(target,slot?slot-1:0,work.Buffers[slot]);
    }
    if (!CheckGL("Allocate PN storage")) return false;
    for (std::size_t phase=0;phase<TessellationPrograms.size();++phase)
    {
        GlUseProgram(TessellationPrograms[phase]); GpuDispatch((counts[phase]+TessellationThreadCount-1)/TessellationThreadCount,1,1);
        if (phase==0) GpuMemoryBarrier(GpuStorageBarrier);
    }
    GpuMemoryBarrier(GpuStorageBarrier|TessellationVertexBarrier|TessellationIndexBarrier|TessellationBufferBarrier);
    if (!CheckGL("Dispatch PN geometry generation")) return false;
    TessellatedGeometryBuffers candidate; candidate.Vertices.resize(constants.Outputs.X); candidate.Indices.resize(constants.Outputs.Y);
    std::array<BufferResource,2> resources;
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"PN tessellated vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"PN tessellated indices"};
    void* mirrors[]{candidate.Vertices.data(),candidate.Indices.data()};
    for (std::size_t stream=0;stream<resources.size();++stream)
    {
        auto& resource=resources[stream]; const auto bytes=resource.Description.SizeInBytes; resource.Bytes.resize(bytes);
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work.Buffers[7+stream]);
        const void* mapped=TessellationMapBuffer(GL_SHADER_STORAGE_BUFFER,0,static_cast<GLsizeiptr>(bytes),TessellationMapRead);
        if (!mapped) return Fail("ES PN output mapping failed");
        std::memcpy(mirrors[stream],mapped,bytes); std::memcpy(resource.Bytes.data(),mapped,bytes);
        const bool valid=TessellationUnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=0;
        if (!CheckGL("Read PN output stream") || !valid) return Fail("ES PN readback failed or was invalidated");
        resource.Buffer=work.Buffers[7+stream];
    }
    if (!ValidateTriangleGeometry(candidate.Vertices,candidate.Indices,LastError)) return false;
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=3;
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    work.Buffers[7]=work.Buffers[8]=0; output=std::move(candidate); return true;
}
