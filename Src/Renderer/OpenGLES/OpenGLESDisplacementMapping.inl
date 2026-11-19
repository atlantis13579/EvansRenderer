// ES preparation restores indexed storage/uniform bindings as well as generic
// bindings. Later cached surface programs must not inherit compute scratch.
static constexpr GLenum DisplacementComputeBlocks=0x90DB;
static constexpr GLbitfield DisplacementVertexBarrier=0x00000001,DisplacementReadbackBarrier=0x00000200,DisplacementMapRead=0x00000001;
bool DisplacementSupported=false;
GLint DisplacementMaximumBlock=0;
std::array<GLuint,2> DisplacementPrograms{};
void*(EVANS_GL_CALL* DisplacementMapBuffer)(GLenum,GLintptr,GLsizeiptr,GLbitfield)=nullptr;
GLboolean(EVANS_GL_CALL* DisplacementUnmapBuffer)(GLenum)=nullptr;
void(EVANS_GL_CALL* DisplacementGetRange)(GLenum,GLuint,std::int64_t*)=nullptr;
void(EVANS_GL_CALL* DisplacementBindRange)(GLenum,GLuint,GLuint,GLintptr,GLsizeiptr)=nullptr;
static constexpr GLenum DisplacementStorageStart=0x90D4,DisplacementStorageSize=0x90D5,DisplacementUniformStart=0x8A29,DisplacementUniformSize=0x8A2A;
bool ProbeDisplacement()
{
    GLint major=0,minor=0,blocks=0,bindings=0;
    GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    if (major<3 || (major==3 && minor<1)) return false;
    GlGetIntegerv(DisplacementComputeBlocks,&blocks); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&DisplacementMaximumBlock);
    return GlGetError()==GL_NO_ERROR && blocks>=6 && bindings>=7 && DisplacementMaximumBlock>=sizeof(Vertex) &&
        LoadGL(GpuDispatch,"glDispatchCompute") && LoadGL(GpuMemoryBarrier,"glMemoryBarrier") && LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") &&
        LoadGL(DisplacementMapBuffer,"glMapBufferRange") && LoadGL(DisplacementUnmapBuffer,"glUnmapBuffer") &&
        LoadGL(DisplacementGetRange,"glGetInteger64i_v") && LoadGL(DisplacementBindRange,"glBindBufferRange");
}
void DestroyDisplacement()
{ for (auto& program:DisplacementPrograms) { if (program && GlDeleteProgram) GlDeleteProgram(program); program=0; } }
bool EnsureDisplacement()
{
    if (!DisplacementSupported) return Fail("ES displacement requires compute, six storage blocks and seven storage binding points");
    if (DisplacementPrograms[0] && DisplacementPrograms[1]) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","DisplacementMapping.glsl",source,LastError)) return false;
    const auto versionEnd=source.find('\n'); if (versionEnd==std::string::npos) return Fail("ES displacement source is missing its version line");
    for (std::size_t phase=0;phase<DisplacementPrograms.size();++phase) if (!DisplacementPrograms[phase])
    {
        auto nativeSource=source; nativeSource.insert(versionEnd+1,"#define EVANS_DISPLACEMENT_PASS "+std::to_string(phase)+"\n");
        const auto program=CompileComputeProgram(nativeSource); if(!program)return false;
        DisplacementPrograms[phase]=program;
    }
    return true;
}
bool CreateDisplaced(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    DisplacementConstants constants;
    if (!ValidateDisplacementUpload(geometry,settings,constants,LastError) || !EnsureDisplacement()) return false;
    const std::size_t vertexBytes=geometry.Vertices.size()*sizeof(Vertex);
    const void* data[]{&constants,geometry.Vertices.data(),geometry.Directions.data(),geometry.Indices.data(),geometry.Neighborhoods.data(),geometry.AdjacentTriangles.data(),nullptr};
    const std::size_t sizes[]{sizeof(constants),vertexBytes,geometry.Directions.size()*sizeof(Float4),geometry.Indices.size()*sizeof(std::uint32_t),
        geometry.Neighborhoods.size()*sizeof(ShaderUInt4),geometry.AdjacentTriangles.size()*sizeof(std::uint32_t),vertexBytes};
    for (std::size_t slot=1;slot<7;++slot)
        if (sizes[slot]>static_cast<std::size_t>(DisplacementMaximumBlock)) return Fail("Displacement allocation exceeds the ES storage-block range");
    struct Work
    {
        Impl& Owner; std::array<GLuint,7> Buffers{}; std::array<GLint,6> Storage{};
        std::array<std::int64_t,6> Starts{},Sizes{}; std::int64_t UniformStart=0,UniformSize=0;
        GLint Uniform=0,GenericUniform=0,GenericStorage=0,Program=0;
        explicit Work(Impl& owner):Owner(owner)
        {
            for (GLuint slot=1;slot<7;++slot)
            {
                Owner.GpuGetIntegerIndexed(GpuStorageBinding,slot,&Storage[slot-1]);
                Owner.DisplacementGetRange(DisplacementStorageStart,slot,&Starts[slot-1]); Owner.DisplacementGetRange(DisplacementStorageSize,slot,&Sizes[slot-1]);
            }
            Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform);
            Owner.DisplacementGetRange(DisplacementUniformStart,0,&UniformStart); Owner.DisplacementGetRange(DisplacementUniformSize,0,&UniformSize);
            Owner.GlGetIntegerv(GpuUniformBinding,&GenericUniform); Owner.GlGetIntegerv(GpuStorageBinding,&GenericStorage); Owner.GlGetIntegerv(GpuCurrentProgram,&Program);
        }
        ~Work()
        {
            for (GLuint slot=1;slot<7;++slot)
                if (Storage[slot-1] && Sizes[slot-1]>0)
                    Owner.DisplacementBindRange(GL_SHADER_STORAGE_BUFFER,slot,static_cast<GLuint>(Storage[slot-1]),static_cast<GLintptr>(Starts[slot-1]),static_cast<GLsizeiptr>(Sizes[slot-1]));
                else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,static_cast<GLuint>(Storage[slot-1]));
            if (Uniform && UniformSize>0) Owner.DisplacementBindRange(GL_UNIFORM_BUFFER,0,static_cast<GLuint>(Uniform),static_cast<GLintptr>(UniformStart),static_cast<GLsizeiptr>(UniformSize));
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
        GlBindBufferBase(target,slot,work.Buffers[slot]);
    }
    if (!CheckGL("Allocate displacement storage")) return false;
    BufferResource resource; resource.Description={BufferUsage::Vertex,vertexBytes,false,"Displaced vertices"}; resource.Bytes.resize(vertexBytes);
    DisplacedVertexBuffer candidate; candidate.Vertices.resize(geometry.Vertices.size());
    const GLuint groups=static_cast<GLuint>((geometry.Vertices.size()+DisplacementThreadCount-1)/DisplacementThreadCount);
    GlUseProgram(DisplacementPrograms[0]); GpuDispatch(groups,1,1); GpuMemoryBarrier(GpuStorageBarrier);
    GlUseProgram(DisplacementPrograms[1]); GpuDispatch(groups,1,1);
    GpuMemoryBarrier(GpuStorageBarrier|DisplacementVertexBarrier|DisplacementReadbackBarrier);
    if (!CheckGL("Dispatch displacement and normal reconstruction")) return false;
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work.Buffers[6]);
    const void* mapped=DisplacementMapBuffer(GL_SHADER_STORAGE_BUFFER,0,static_cast<GLsizeiptr>(vertexBytes),DisplacementMapRead);
    if (!mapped) return Fail("ES displacement readback mapping failed");
    std::memcpy(candidate.Vertices.data(),mapped,vertexBytes); std::memcpy(resource.Bytes.data(),mapped,vertexBytes);
    const bool valid=DisplacementUnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=0;
    if (!CheckGL("Read displacement output")) return false;
    if (!valid) return Fail("ES displacement readback was invalidated");
    if (!ValidateTriangleGeometry(candidate.Vertices,geometry.Indices,LastError)) return false;
    resource.Buffer=work.Buffers[6]; candidate.Buffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(DisplacementPrograms.size());
    Buffers.emplace(candidate.Buffer.Value,std::move(resource)); work.Buffers[6]=0; output=std::move(candidate); return true;
}
