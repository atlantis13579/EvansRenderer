std::array<GLuint,2> VirtualShadowRequestPrograms{};
bool VirtualShadowSupported=false;
bool ProbeVirtualShadows()
{
    GLint major=0,minor=0,blocks=0,bindings=0,bytes=0,invocations=0,width=0,groupsX=0,groupsY=0;
    GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    if (major<3 || (major==3 && minor<1)) return false;
    if (!LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier") || !LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") ||
        !LoadGL(VirtualMapBuffer,"glMapBufferRange") || !LoadGL(VirtualUnmapBuffer,"glUnmapBuffer") ||
        !LoadGL(VirtualGetRange,"glGetInteger64i_v") || !LoadGL(VirtualBindRange,"glBindBufferRange")) return false;
    GlGetIntegerv(VirtualComputeBlocks,&blocks); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&bytes); GlGetIntegerv(VirtualInvocations,&invocations);
    GpuGetIntegerIndexed(VirtualGroupSize,0,&width); GpuGetIntegerIndexed(VirtualGroupCount,0,&groupsX); GpuGetIntegerIndexed(VirtualGroupCount,1,&groupsY);
    return GlGetError()==GL_NO_ERROR && blocks>=2 && bindings>=2 && bytes>=GLint(VirtualShadowMaximumReceiverPixels*sizeof(float)) &&
        invocations>=GLint(VirtualShadowGroupWidth) && width>=GLint(VirtualShadowGroupWidth) && groupsX>=GLint(VirtualShadowGroupColumns) && groupsY>0;
}
void DestroyVirtualShadowRequests()
{ for (auto& program:VirtualShadowRequestPrograms) { if (program && GlDeleteProgram) GlDeleteProgram(program); program=0; } }
bool BuildVirtualShadowRequests(const VirtualShadowRequestUpload& upload,VirtualShadowRequestResult& output,GLuint nativeDepth=0)
{
    if (!nativeDepth && !ValidateVirtualShadowRequestUpload(upload,LastError)) return false;
    VirtualShadowRequestConstants constants;
    if (!BuildVirtualShadowRequestConstants(upload.Frame,constants,LastError)) return false;
    GLint major=0,minor=0; GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    if (major<3 || (major==3 && minor<1)) return Fail("Virtual shadow demand requires ES 3.1 compute/storage buffers");
    if (!LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier") || !LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") ||
        !LoadGL(VirtualMapBuffer,"glMapBufferRange") || !LoadGL(VirtualUnmapBuffer,"glUnmapBuffer") || !LoadGL(VirtualGetRange,"glGetInteger64i_v") || !LoadGL(VirtualBindRange,"glBindBufferRange")) return false;
    GLint blockBytes=0,blocks=0,invocations=0,groupWidth=0,maxGroupsX=0,maxGroupsY=0;
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&blockBytes); GlGetIntegerv(VirtualComputeBlocks,&blocks); GlGetIntegerv(VirtualInvocations,&invocations);
    GpuGetIntegerIndexed(VirtualGroupSize,0,&groupWidth); GpuGetIntegerIndexed(VirtualGroupCount,0,&maxGroupsX); GpuGetIntegerIndexed(VirtualGroupCount,1,&maxGroupsY);
    const auto depthBytes=std::size_t(upload.Frame.Width)*upload.Frame.Height*sizeof(float),requestBytes=std::size_t(constants.Limits.Y)*sizeof(std::uint32_t);
    if (!CheckGL("Probe virtual shadow demand") || blocks<2 || invocations<int(VirtualShadowGroupWidth) || groupWidth<int(VirtualShadowGroupWidth) ||
        maxGroupsX<=0 || maxGroupsY<=0 || blockBytes<=0 || depthBytes>std::size_t(blockBytes) || requestBytes>std::size_t(blockBytes)) return Fail("ES virtual shadow demand exceeds compute/storage limits");
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","VirtualShadowMaps.glsl",source,LastError)) return false;
    const auto versionEnd=source.find('\n'); if (versionEnd==std::string::npos) return Fail("Virtual shadow demand has no GLSL version line");
    for (unsigned phase=0;phase<VirtualShadowRequestPrograms.size();++phase) if (!VirtualShadowRequestPrograms[phase])
    {
        auto text=source; text.insert(versionEnd+1,"#define EVANS_VIRTUAL_SHADOW_PASS "+std::to_string(phase)+"\n");
        const auto program=CompileComputeProgram(text); if(!program)return false;
        VirtualShadowRequestPrograms[phase]=program;
    }
    struct Work
    {
        Impl& Owner; std::array<GLuint,3> Buffers{}; std::array<GLint,2> Storage{}; std::array<std::int64_t,2> Starts{},Sizes{};
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
            Owner.GlDeleteBuffers(static_cast<GLsizei>(Buffers.size()),Buffers.data());
        }
    } work(*this);
    GlGenBuffers(static_cast<GLsizei>(work.Buffers.size()),work.Buffers.data());
    const void* data[]{&constants,upload.Depth.data(),nullptr}; const std::size_t sizes[]{sizeof(constants),depthBytes,requestBytes};
    for (GLuint slot=0;slot<work.Buffers.size();++slot)
    {
        if (slot==1 && nativeDepth) { GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,nativeDepth); continue; }
        const GLenum target=slot?GL_SHADER_STORAGE_BUFFER:GL_UNIFORM_BUFFER;
        GlBindBuffer(target,work.Buffers[slot]); GlBufferData(target,static_cast<GLsizeiptr>(sizes[slot]),data[slot],GL_STATIC_DRAW); GlBindBufferBase(target,slot?slot-1:0,work.Buffers[slot]);
    }
    if (!CheckGL("Allocate virtual shadow request storage")) return false;
    const unsigned counts[]{constants.Limits.Y,constants.Extent.X*constants.Extent.Y};
    for (unsigned phase=0;phase<VirtualShadowRequestPrograms.size();++phase)
    {
        const auto groups=(counts[phase]+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
        const auto x=std::min(groups,VirtualShadowGroupColumns),y=(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns;
        if (x>unsigned(maxGroupsX) || y>unsigned(maxGroupsY)) return Fail("ES virtual shadow request dispatch exceeds device group dimensions");
        GlUseProgram(VirtualShadowRequestPrograms[phase]); GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier|VirtualBufferBarrier);
    }
    if (!CheckGL("Execute virtual shadow request marking")) return false;
    VirtualShadowRequestResult result; result.Pages.resize(constants.Limits.Y); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work.Buffers[2]);
    const auto* mapped=VirtualMapBuffer(GL_SHADER_STORAGE_BUFFER,0,static_cast<GLsizeiptr>(requestBytes),VirtualMapRead);
    if (!mapped) return Fail("ES virtual shadow request feedback is not readable"); std::memcpy(result.Pages.data(),mapped,requestBytes);
    if (!VirtualUnmapBuffer(GL_SHADER_STORAGE_BUFFER) || !CheckGL("Read virtual shadow request feedback")) return Fail("ES virtual shadow request feedback was invalidated");
    VirtualShadowLayout layout; if (!BuildVirtualShadowLayout(upload.Frame.Settings,layout,LastError) || !ValidateVirtualShadowRequests(layout,result.Pages,LastError)) return false;
    output=std::move(result); return true;
}
