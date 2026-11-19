GLuint LumenSurfaceLookup=0;
void(EVANS_GL_CALL* LookupGetInteger64)(GLenum,std::int64_t*)=nullptr;
static constexpr GLenum LookupMaxStorageSize=0x90DE,LookupMaxComputeStorage=0x90DB,LookupMaxGroups=0x91BE;
void DestroyLumenSurfaceLookup() { if (LumenSurfaceLookup) GlDeleteProgram(LumenSurfaceLookup); LumenSurfaceLookup=0; }
bool EnsureLumenSurfaceLookup()
{
    if (!EnsureLumenSurface() || !LoadGL(LookupGetInteger64,"glGetInteger64v")) return false;
    GLint count=0; GlGetIntegerv(LookupMaxComputeStorage,&count); if (count<3) return Fail("Lumen surface lookup needs three compute storage bindings");
    if (LumenSurfaceLookup) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
    const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen lookup shader has no version directive");
    source.insert(version+1,"#define EVANS_LUMEN_LOOKUP_STAGE 1\n"); const auto program=CompileComputeProgram(source); if(!program)return false;
    LumenSurfaceLookup=program; return true;
}
// Capture's state guard already preserves texture units and storage slot zero.
// Lookup additionally uses two indexed storage ranges; restore them before the
// capture guard restores the caller's generic buffer binding.
struct LumenLookupStorageState
{
    Impl& Owner; std::array<GLint,3> Buffers{}; std::array<std::int64_t,3> Starts{},Sizes{};
    explicit LumenLookupStorageState(Impl& owner):Owner(owner)
    {
        for (unsigned index=0;index<Buffers.size();++index)
        { Owner.GpuGetIntegerIndexed(GpuStorageBinding,index+1,&Buffers[index]); Owner.LumenGetRange(LumenStorageStart,index+1,&Starts[index]); Owner.LumenGetRange(LumenStorageSize,index+1,&Sizes[index]); }
    }
    ~LumenLookupStorageState()
    {
        for (unsigned index=0;index<Buffers.size();++index)
            if (Buffers[index] && Sizes[index]) Owner.LumenBindRange(GL_SHADER_STORAGE_BUFFER,index+1,Buffers[index],GLintptr(Starts[index]),GLsizeiptr(Sizes[index]));
            else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,index+1,Buffers[index]);
    }
};
bool LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Unknown Lumen surface lookup handle");
    const auto& record=*found->second; LumenSurfaceLookupFrame frame; std::vector<LumenLookupCard> cards;
    if (!BuildLumenSurfaceLookupFrame(record.Layout,upload,frame,LastError) || !BuildLumenLookupCards(record.Layout,record.Captures,cards,LastError) || !EnsureLumenSurfaceLookup()) return false;
    const auto bytes=upload.Queries.size()*sizeof(LumenSurfaceSample),cardBytes=cards.size()*sizeof(LumenLookupCard);
    std::int64_t maximum=0; LookupGetInteger64(LookupMaxStorageSize,&maximum);
    if (maximum<=0 || std::max(bytes,cardBytes)>std::uint64_t(maximum)) return Fail("Lumen surface lookup exceeds ES storage limits");
    const auto groups=unsigned((upload.Queries.size()+LumenLookupThreads-1)/LumenLookupThreads);
    const unsigned x=std::min(groups,LumenLookupColumns),y=(groups+LumenLookupColumns-1)/LumenLookupColumns;
    GLint maxX=0,maxY=0; GpuGetIntegerIndexed(LookupMaxGroups,0,&maxX); GpuGetIntegerIndexed(LookupMaxGroups,1,&maxY);
    if (x>unsigned(maxX) || y>unsigned(maxY)) return Fail("Lumen surface lookup exceeds ES dispatch limits");
    struct BuffersOwner { Impl& Owner; std::array<GLuint,4> Values{}; ~BuffersOwner() { Owner.GlDeleteBuffers(4,Values.data()); } } buffers{*this};
    LumenCaptureState state(*this); LumenLookupStorageState storageState(*this); GlGenBuffers(4,buffers.Values.data());
    GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Values[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Values[0]);
    const void* sources[]{cards.data(),upload.Queries.data(),nullptr};
    const std::size_t sizes[]{cardBytes,upload.Queries.size()*sizeof(LumenSurfaceQuery),bytes};
    for (unsigned slot=0;slot<3;++slot)
    { GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers.Values[slot+1]); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(sizes[slot]),sources[slot],GL_STATIC_DRAW); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,buffers.Values[slot+1]); }
    GlUseProgram(LumenSurfaceLookup); const char* names[]{"LookupAlbedo","LookupNormal","LookupEmission"};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        GlActiveTexture(GL_TEXTURE0+plane); LumenBindSampler(plane,0); GlBindTexture(GL_TEXTURE_2D,record.Images[plane]);
        GlUniform1i(GlGetUniformLocation(LumenSurfaceLookup,names[plane]),GLint(plane));
    }
    if (!CheckGL("Bind Lumen surface lookup")) return false;
    GpuDispatch(x,y,1); GpuMemoryBarrier(LumenBufferBarrier); if (!CheckGL("Dispatch Lumen surface lookup")) return false;
    LumenSurfaceLookupResult result; result.Identity=record.Identity; result.Samples.resize(upload.Queries.size());
    const auto* mapped=LumenMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),LumenMapRead); if (!mapped) return Fail("Map Lumen surface lookup failed");
    std::memcpy(result.Samples.data(),mapped,bytes); const auto valid=LumenUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (!valid || !CheckGL("Read Lumen surface lookup")) return Fail("Lumen surface lookup feedback became invalid");
    if (!ValidateLumenSurfaceLookupResult(record.Identity,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
