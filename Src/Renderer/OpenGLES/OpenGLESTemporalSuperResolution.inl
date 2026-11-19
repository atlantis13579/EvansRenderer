// ES owns four storage bindings and UBO zero, restoring exact prior ranges and
// program state. Mapping the two GPU-written buffers completes the transaction.
struct TSRNativeHistory
{
    Impl& Owner; TSRHistoryInfo Info; std::array<GLuint,2> Buffers{};
    GLuint Color=0,Framebuffer=0;
    explicit TSRNativeHistory(Impl& owner):Owner(owner) {}
    ~TSRNativeHistory() { if (Framebuffer) Owner.GlDeleteFramebuffers(1,&Framebuffer); if (Color) Owner.GlDeleteTextures(1,&Color); Owner.GlDeleteBuffers(static_cast<GLsizei>(Buffers.size()),Buffers.data()); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<TSRNativeHistory>> TSRHistories;
TSRSceneState TSRScene;
bool TSRSupported=false;
void ProbeTSRSupport()
{
    GLint major=0,minor=0,blocks=0,fragmentBlocks=0,bindings=0,invocations=0;
    GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    TSRSupported=false; if (major<3 || (major==3 && minor<1)) return;
    GlGetIntegerv(VirtualComputeBlocks,&blocks); GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,&fragmentBlocks); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings); GlGetIntegerv(VirtualInvocations,&invocations);
    TSRSupported=HalfFloatTargetsSupported && blocks>=4 && fragmentBlocks>=1 && bindings>=4 && invocations>=int(TSRGroupSide*TSRGroupSide) && CheckGL("Probe TSR support");
}
GLuint TSROpaque=0,TSROpaqueFBO=0;
unsigned TSROpaqueWidth=0,TSROpaqueHeight=0;
GLuint TSRProgram=0;
GLuint TSRExportProgram=0;
GLuint TSRPackProgram=0;
GLboolean(EVANS_GL_CALL* TSRIsEnabled)(GLenum)=nullptr;
void(EVANS_GL_CALL* TSRGetBoolean)(GLenum,GLboolean*)=nullptr;
void(EVANS_GL_CALL* TSRGetTexLevel)(GLenum,GLint,GLenum,GLint*)=nullptr;
static constexpr GLenum TSRDrawFramebufferBinding=0x8CA6,TSRReadFramebufferBinding=0x8CAA,TSRTextureBinding=0x8069;
static constexpr GLenum TSRViewport=0x0BA2,TSRColorMask=0x0C23,TSRScissorTest=0x0C11,TSRVertexArrayBinding=0x85B5;
static constexpr GLenum TSRPixelPackBuffer=0x88EB,TSRPixelPackBinding=0x88ED;
static constexpr GLenum TSRTextureWidth=0x1000,TSRTextureHeight=0x1001,TSRActiveTexture=0x84E0;
static constexpr GLenum TSRPixelUnpackBuffer=0x88EC,TSRPixelUnpackBinding=0x88EF;
void*(EVANS_GL_CALL* TSRMap)(GLenum,GLintptr,GLsizeiptr,GLbitfield)=nullptr;
GLboolean(EVANS_GL_CALL* TSRUnmap)(GLenum)=nullptr;
void(EVANS_GL_CALL* TSRGetRange)(GLenum,GLuint,std::int64_t*)=nullptr;
void(EVANS_GL_CALL* TSRBindRange)(GLenum,GLuint,GLuint,GLintptr,GLsizeiptr)=nullptr;
GLint TSRMaximumBlock=0,TSRMaximumGroups[2]{};
void DestroyTSR() { TSRScene.Reset(); if (TSROpaqueFBO) GlDeleteFramebuffers(1,&TSROpaqueFBO); if (TSROpaque) GlDeleteTextures(1,&TSROpaque); TSROpaque=TSROpaqueFBO=TSROpaqueWidth=TSROpaqueHeight=0; TSRHistories.clear(); if (TSRProgram && GlDeleteProgram) GlDeleteProgram(TSRProgram); if (TSRExportProgram && GlDeleteProgram) GlDeleteProgram(TSRExportProgram); if (TSRPackProgram && GlDeleteProgram) GlDeleteProgram(TSRPackProgram); TSRProgram=TSRExportProgram=TSRPackProgram=0; }
bool DestroyTSRHistory(TSRHistoryHandle handle)
{ return (handle && TSRHistories.erase(handle.Value)) || Fail("ES TSR history handle is invalid or belongs to another RHI"); }
bool ExportTSR(TSRNativeHistory& record,TSRReadback& result,bool readback=true)
{
    if (!TSRExportProgram)
    {
        GLint blocks=0; GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,&blocks);
        if (!CheckGL("Probe TSR export") || blocks<1) return Fail("ES TSR HDR export requires a fragment storage block");
        std::string source; if (!LoadTechniqueShaderSource("OpenGLES","TSR.glsl",source,LastError)) return false;
        auto vertexSource=source,fragmentSource=source;
        vertexSource.insert(vertexSource.find('\n')+1,"#define EVANS_TSR_EXPORT_VERTEX_STAGE 1\n");
        fragmentSource.insert(fragmentSource.find('\n')+1,"#define EVANS_TSR_EXPORT_FRAGMENT_STAGE 1\n");
        const auto program=CompileProgramPair(vertexSource,fragmentSource); if(!program)return false;
        TSRExportProgram=program;
    }
    struct Restore
    {
        Impl& Owner; GLint Draw=0,Read=0,Texture=0,Pack=0,Array=0,PixelPack=0,Viewport[4]{}; bool Depth=false,Blend=false,Cull=false,Scissor=false; GLboolean Mask[4]{};
        explicit Restore(Impl& owner):Owner(owner)
        {
            Owner.GlGetIntegerv(TSRDrawFramebufferBinding,&Draw); Owner.GlGetIntegerv(TSRReadFramebufferBinding,&Read);
            Owner.GlGetIntegerv(TSRTextureBinding,&Texture); Owner.GlGetIntegerv(GL_PACK_ALIGNMENT,&Pack); Owner.GlGetIntegerv(TSRViewport,Viewport);
            Owner.GlGetIntegerv(TSRVertexArrayBinding,&Array); Owner.GlGetIntegerv(TSRPixelPackBinding,&PixelPack);
            Depth=Owner.TSRIsEnabled(GL_DEPTH_TEST)!=0; Blend=Owner.TSRIsEnabled(GL_BLEND)!=0; Cull=Owner.TSRIsEnabled(GL_CULL_FACE)!=0; Scissor=Owner.TSRIsEnabled(TSRScissorTest)!=0;
            Owner.TSRGetBoolean(TSRColorMask,Mask);
        }
        ~Restore()
        {
            Owner.GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,Draw); Owner.GlBindFramebuffer(GL_READ_FRAMEBUFFER,Read); Owner.GlBindTexture(GL_TEXTURE_2D,Texture); Owner.GlPixelStorei(GL_PACK_ALIGNMENT,Pack);
            Owner.GlBindVertexArray(Array); Owner.GlBindBuffer(TSRPixelPackBuffer,PixelPack);
            Owner.GlViewport(Viewport[0],Viewport[1],Viewport[2],Viewport[3]); Owner.GlColorMask(Mask[0],Mask[1],Mask[2],Mask[3]);
            const GLenum states[]{GL_DEPTH_TEST,GL_BLEND,GL_CULL_FACE,TSRScissorTest}; const bool enabled[]{Depth,Blend,Cull,Scissor};
            for (unsigned i=0;i<4;++i) if (enabled[i]) Owner.GlEnable(states[i]); else Owner.GlDisable(states[i]);
        }
    } restore(*this);
    GlGenTextures(1,&record.Color); GlBindTexture(GL_TEXTURE_2D,record.Color);
    GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,record.Info.Width,record.Info.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    GlGenFramebuffers(1,&record.Framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,record.Framebuffer);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,record.Color,0);
    if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE) return Fail("ES TSR HDR export framebuffer is incomplete");
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(record.Info.Width,record.Info.Height,0,constants,LastError)) return false;
    GLuint uniform=0; GlGenBuffers(1,&uniform); GlBindBuffer(GL_UNIFORM_BUFFER,uniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(constants),&constants,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,uniform);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,record.Buffers[0]); GlUseProgram(TSRExportProgram);
    for (auto state:{GL_DEPTH_TEST,GL_BLEND,GL_CULL_FACE,TSRScissorTest}) GlDisable(state);
    GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlViewport(0,0,record.Info.Width,record.Info.Height); GlBindVertexArray(VertexArray); GlDrawArrays(GL_TRIANGLES,0,3);
    if(!readback){GlDeleteBuffers(1,&uniform);return CheckGL("Export TSR HDR");}
    std::vector<Float4> bottomUp(std::size_t(record.Info.Width)*record.Info.Height); GlPixelStorei(GL_PACK_ALIGNMENT,1);
    GlBindBuffer(TSRPixelPackBuffer,0); GlReadPixels(0,0,record.Info.Width,record.Info.Height,GL_RGBA,GL_FLOAT,bottomUp.data()); GlDeleteBuffers(1,&uniform);
    if (!CheckGL("Export TSR HDR image")) return false;
    result.PresentationHDR.resize(bottomUp.size());
    for (unsigned y=0;y<record.Info.Height;++y) std::copy_n(bottomUp.data()+std::size_t(record.Info.Height-1-y)*record.Info.Width,record.Info.Width,result.PresentationHDR.data()+std::size_t(y)*record.Info.Width);
    return true;
}
bool EnsureTSR()
{
    if (!LoadGL(TSRIsEnabled,"glIsEnabled") || !LoadGL(TSRGetBoolean,"glGetBooleanv") || !LoadGL(TSRGetTexLevel,"glGetTexLevelParameteriv")) return false;
    if (TSRProgram && TSRPackProgram) return true;
    GLint major=0,minor=0,blocks=0,bindings=0,invocations=0,sides[2]{};
    GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    if (major<3 || (major==3 && minor<1)) return Fail("TSR requires ES 3.1 compute");
    if (!LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier") || !LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") ||
        !LoadGL(TSRMap,"glMapBufferRange") || !LoadGL(TSRUnmap,"glUnmapBuffer") || !LoadGL(TSRGetRange,"glGetInteger64i_v") || !LoadGL(TSRBindRange,"glBindBufferRange")) return false;
    GlGetIntegerv(VirtualComputeBlocks,&blocks); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&TSRMaximumBlock); GlGetIntegerv(VirtualInvocations,&invocations);
    for (unsigned axis=0;axis<2;++axis) { GpuGetIntegerIndexed(VirtualGroupSize,axis,&sides[axis]); GpuGetIntegerIndexed(VirtualGroupCount,axis,&TSRMaximumGroups[axis]); }
    if (!CheckGL("Probe TSR compute") || blocks<4 || bindings<4 || invocations<TSRGroupSide*TSRGroupSide || sides[0]<TSRGroupSide || sides[1]<TSRGroupSide || TSRMaximumBlock<=0)
        return Fail("ES TSR requires four storage blocks and its reconstruction workgroup");
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","TSR.glsl",source,LastError)) return false;
    const auto compile=[&](const std::string& code,GLuint& target)
    {
        if (target) return true;
        const auto program=CompileComputeProgram(code); if(!program)return false;
        target=program; return true;
    };
    auto packSource=source; packSource.insert(packSource.find('\n')+1,"#define EVANS_TSR_PACK_STAGE 1\n");
    return compile(source,TSRProgram) && compile(packSource,TSRPackProgram);
}
struct TSRNativeImages { std::array<GLuint,TSRSceneImageCount> Textures{}; std::uint32_t Flags=0; };
bool PackTSRImages(const TSRFrame& frame,const TSRNativeImages& images,GLuint destination)
{
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(frame.InputWidth,frame.InputHeight,images.Flags,constants,LastError)) return false;
    struct Restore
    {
        Impl& Owner; GLint Active=0; std::array<GLint,TSRSceneImageCount> Textures{}; GLuint Uniform=0;
        explicit Restore(Impl& owner):Owner(owner)
        {
            Owner.GlGetIntegerv(TSRActiveTexture,&Active);
            for (unsigned i=0;i<TSRSceneImageCount;++i) { Owner.GlActiveTexture(GL_TEXTURE0+i); Owner.GlGetIntegerv(TSRTextureBinding,&Textures[i]); }
        }
        ~Restore()
        {
            for (unsigned i=0;i<TSRSceneImageCount;++i) { Owner.GlActiveTexture(GL_TEXTURE0+i); Owner.GlBindTexture(GL_TEXTURE_2D,Textures[i]); }
            Owner.GlActiveTexture(Active); if (Uniform) Owner.GlDeleteBuffers(1,&Uniform);
        }
    } restore(*this);
    const char* names[]{"FinalRadiance","OpaqueRadiance","RenderMotion","RenderDepth","RenderOutline","AdditionalReactive"};
    GlUseProgram(TSRPackProgram);
    for (unsigned image=0;image<TSRSceneImageCount;++image)
    {
        const auto texture=TSRImageRequired(image,images.Flags)?images.Textures[image]:images.Textures[TSRSceneFinal];
        if (!texture) return Fail("ES TSR is missing a required scene image");
        GlActiveTexture(GL_TEXTURE0+image); GlBindTexture(GL_TEXTURE_2D,texture); GLint width=0,height=0;
        TSRGetTexLevel(GL_TEXTURE_2D,0,TSRTextureWidth,&width); TSRGetTexLevel(GL_TEXTURE_2D,0,TSRTextureHeight,&height);
        if (width!=int(frame.InputWidth) || height!=int(frame.InputHeight)) return Fail("ES TSR requires resolved render-sized level-zero images");
        const auto location=GlGetUniformLocation(TSRPackProgram,names[image]); if (location>=0) GlUniform1i(location,static_cast<GLint>(image));
    }
    GlGenBuffers(1,&restore.Uniform); GlBindBuffer(GL_UNIFORM_BUFFER,restore.Uniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(constants),&constants,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,restore.Uniform);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,destination);
    GpuDispatch((frame.InputWidth+TSRGroupSide-1)/TSRGroupSide,(frame.InputHeight+TSRGroupSide-1)/TSRGroupSide,1); GpuMemoryBarrier(GpuStorageBarrier);
    return CheckGL("Pack TSR scene images");
}
bool ReconstructTSRImages(const TSRFrame& frame,const TSRNativeImages& images,TSRHistoryHandle previous,TSRResult& output)
{
    std::shared_ptr<TSRNativeHistory> old;
    if (previous) { const auto found=TSRHistories.find(previous.Value); if (found==TSRHistories.end()) return Fail("ES previous TSR history is invalid"); old=found->second; }
    TSRConstants constants; if (!BuildTSRConstants(frame,old?&old->Info:nullptr,constants,LastError) || !EnsureTSR()) return false;
    const auto groupsX=(constants.Extent.Z+TSRGroupSide-1)/TSRGroupSide,groupsY=(constants.Extent.W+TSRGroupSide-1)/TSRGroupSide;
    if (groupsX>static_cast<unsigned>(TSRMaximumGroups[0]) || groupsY>static_cast<unsigned>(TSRMaximumGroups[1])) return Fail("TSR exceeds ES group dimensions");
    const auto bytes=std::size_t(frame.OutputWidth)*frame.OutputHeight*TSRHistoryVectors*sizeof(Float4);
    const std::array<Float4,TSRHistoryVectors> empty{}; const void* data[]{&constants,nullptr,empty.data()}; const std::size_t sizes[]{sizeof(constants),std::size_t(frame.InputWidth)*frame.InputHeight*(TSRInputVectors*sizeof(Float4)),sizeof(empty)};
    if (bytes>static_cast<std::size_t>(TSRMaximumBlock) || sizes[1]>static_cast<std::size_t>(TSRMaximumBlock)) return Fail("TSR exceeds ES storage range");
    struct Work
    {
        Impl& Owner; std::array<GLuint,3> Buffers{}; std::array<GLint,4> Storage{}; std::array<std::int64_t,4> Starts{},Sizes{};
        GLint Uniform=0,GenericUniform=0,GenericStorage=0,Program=0; std::int64_t UniformStart=0,UniformSize=0;
        explicit Work(Impl& owner):Owner(owner)
        {
            for (GLuint slot=0;slot<Storage.size();++slot) { Owner.GpuGetIntegerIndexed(GpuStorageBinding,slot,&Storage[slot]); Owner.TSRGetRange(VirtualStorageStart,slot,&Starts[slot]); Owner.TSRGetRange(VirtualStorageSize,slot,&Sizes[slot]); }
            Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform); Owner.TSRGetRange(VirtualUniformStart,0,&UniformStart); Owner.TSRGetRange(VirtualUniformSize,0,&UniformSize);
            Owner.GlGetIntegerv(GpuUniformBinding,&GenericUniform); Owner.GlGetIntegerv(GpuStorageBinding,&GenericStorage); Owner.GlGetIntegerv(GpuCurrentProgram,&Program);
        }
        ~Work()
        {
            for (GLuint slot=0;slot<Storage.size();++slot)
                if (Storage[slot] && Sizes[slot]>0) Owner.TSRBindRange(GL_SHADER_STORAGE_BUFFER,slot,static_cast<GLuint>(Storage[slot]),static_cast<GLintptr>(Starts[slot]),static_cast<GLsizeiptr>(Sizes[slot]));
                else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,static_cast<GLuint>(Storage[slot]));
            if (Uniform && UniformSize>0) Owner.TSRBindRange(GL_UNIFORM_BUFFER,0,static_cast<GLuint>(Uniform),static_cast<GLintptr>(UniformStart),static_cast<GLsizeiptr>(UniformSize));
            else Owner.GlBindBufferBase(GL_UNIFORM_BUFFER,0,static_cast<GLuint>(Uniform));
            Owner.GlBindBuffer(GL_UNIFORM_BUFFER,static_cast<GLuint>(GenericUniform)); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,static_cast<GLuint>(GenericStorage)); Owner.GlUseProgram(static_cast<GLuint>(Program));
            Owner.GlDeleteBuffers(static_cast<GLsizei>(Buffers.size()),Buffers.data());
        }
    } work(*this);
    GlGenBuffers(static_cast<GLsizei>(work.Buffers.size()),work.Buffers.data());
    for (unsigned slot=0;slot<work.Buffers.size();++slot)
    { const auto target=slot?GL_SHADER_STORAGE_BUFFER:GL_UNIFORM_BUFFER; GlBindBuffer(target,work.Buffers[slot]); GlBufferData(target,static_cast<GLsizeiptr>(sizes[slot]),data[slot],GL_STATIC_DRAW); }
    if (!PackTSRImages(frame,images,work.Buffers[1])) return false;
    auto record=std::make_shared<TSRNativeHistory>(*this); record->Info=TSRHistoryIdentity(frame); GlGenBuffers(static_cast<GLsizei>(record->Buffers.size()),record->Buffers.data());
    for (auto buffer:record->Buffers) { GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(bytes),nullptr,GL_STATIC_DRAW); }
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[0]);
    const GLuint bindings[]{work.Buffers[1],constants.Options.X?old->Buffers[0]:work.Buffers[2],record->Buffers[0],record->Buffers[1]};
    for (GLuint slot=0;slot<4;++slot) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,bindings[slot]);
    GlUseProgram(TSRProgram); GpuDispatch(groupsX,groupsY,1); GpuMemoryBarrier(GpuStorageBarrier|VirtualBufferBarrier);
    if (!CheckGL("Dispatch TSR reconstruction")) return false;
    TSRResult candidate; candidate.HistoryUsed=constants.Options.X!=0; candidate.Info=record->Info; std::vector<Float4>* mirrors[]{&candidate.Readback.History,&candidate.Readback.Auxiliary};
    if(frame.CaptureDiagnostics)for (unsigned slot=0;slot<record->Buffers.size();++slot)
    {
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record->Buffers[slot]); const void* mapped=TSRMap(GL_SHADER_STORAGE_BUFFER,0,static_cast<GLsizeiptr>(bytes),VirtualMapRead);
        if (!mapped) return Fail("ES TSR readback mapping failed"); mirrors[slot]->resize(bytes/sizeof(Float4)); std::memcpy(mirrors[slot]->data(),mapped,bytes);
        if (!TSRUnmap(GL_SHADER_STORAGE_BUFFER) || !CheckGL("Read TSR reconstruction")) return Fail("ES TSR readback was invalidated");
    }
    if (!ExportTSR(*record,candidate.Readback,frame.CaptureDiagnostics) || (frame.CaptureDiagnostics && !ValidateTSRReadback(frame,candidate.Readback,LastError))) return false;
    candidate.Handle=NewTSRHistoryHandle(); if (!candidate.Handle) return Fail("TSR history handle space is exhausted");
    TSRHistories.emplace(candidate.Handle.Value,std::move(record)); output=std::move(candidate); return true;
}
bool PrepareTSRScene(const TSRSceneFrame& requested,const Techniques::PostProcessSettings& post)
{
    const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value);
    const auto found=TSRHistories.find(TSRScene.Committed.Value);
    return TSRScene.Begin(requested,CreateInfo.Width,CreateInfo.Height,post,found==TSRHistories.end()?nullptr:&found->second->Info,LastError);
}
void AbortTSRScene() { const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value); }
void CommitTSRScene() { const auto old=TSRScene.Commit(); if (old) TSRHistories.erase(old.Value); }
bool RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& requested,FrameCapture& output)
{
    const auto found=TSRHistories.find(handle.Value); if (found==TSRHistories.end()) return Fail("ES TSR display history is invalid");
    Techniques::PostProcessSettings settings; if (!BuildTSRDisplaySettings(requested,settings,LastError)) return false;
    const auto history=found->second;
    struct Targets
    {
        Impl& Owner; int Width=0,Height=0; bool Hdr=false; FrameCapture Capture;
        GLuint Color=0,Framebuffer=0,Depth=0; std::array<GLuint,4> HDR{},HdrFBO{}; std::array<GLuint,2> Post{},PostFBO{};
        explicit Targets(Impl& owner):Owner(owner) {}
        ~Targets()
        {
            Owner.GlDeleteFramebuffers(static_cast<GLsizei>(HdrFBO.size()),HdrFBO.data()); Owner.GlDeleteFramebuffers(static_cast<GLsizei>(PostFBO.size()),PostFBO.data());
            if (Framebuffer) Owner.GlDeleteFramebuffers(1,&Framebuffer); Owner.GlDeleteTextures(static_cast<GLsizei>(HDR.size()),HDR.data()); Owner.GlDeleteTextures(static_cast<GLsizei>(Post.size()),Post.data()); if (Color) Owner.GlDeleteTextures(1,&Color);
        }
        void Swap()
        {
            std::swap(Width,Owner.CreateInfo.Width); std::swap(Height,Owner.CreateInfo.Height); std::swap(Hdr,Owner.HdrActive); std::swap(Capture,Owner.PendingFrame);
            std::swap(Color,Owner.ColorTexture); std::swap(Framebuffer,Owner.MainFramebuffer); std::swap(Depth,Owner.DepthTexture); HDR.swap(Owner.HdrTextures); HdrFBO.swap(Owner.HdrFramebuffers); Post.swap(Owner.PostTextures); PostFBO.swap(Owner.PostFramebuffers);
        }
    } targets(*this);
    targets.Width=history->Info.Width; targets.Height=history->Info.Height; targets.Hdr=true;
    GlGenTextures(1,&targets.Color); GlBindTexture(GL_TEXTURE_2D,targets.Color); ConfigureTexture();
    GlTexImage2D(GL_TEXTURE_2D,0,GL_SRGB8_ALPHA8,targets.Width,targets.Height,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    GlGenFramebuffers(1,&targets.Framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,targets.Framebuffer); GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,targets.Color,0);
    if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create TSR display color")) return Fail("ES TSR display framebuffer is incomplete");
    {
        struct Restore { Targets& Data; explicit Restore(Targets& data):Data(data){Data.Swap();} ~Restore(){Data.Swap();} } restore(targets);
        // Depth is deliberately absent from presentation-only framebuffers.
        if (!EnsureHdrTargets()) return false;
        GlBindFramebuffer(GL_READ_FRAMEBUFFER,history->Framebuffer); GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,HdrFramebuffers[0]);
        GlBlitFramebuffer(0,0,CreateInfo.Width,CreateInfo.Height,0,0,CreateInfo.Width,CreateInfo.Height,GL_COLOR_BUFFER_BIT,GL_NEAREST);
        GLuint result=MainFramebuffer;
        if (!CheckGL("Copy immutable TSR display source") || !ApplyHdr(settings.HDR) || !ApplyPostProcess(settings,result)) return false;
        if(TSRScene.Active && !TSRScene.Frame.CaptureDiagnostics){if(!PublishGpuColor(result))return false;}
        else if(!ReadColor(result,PendingFrame.ColorRGBA8))return false;
        PendingFrame.Width=history->Info.Width; PendingFrame.Height=history->Info.Height; PendingFrame.Statistics.PassCount=PendingFrame.Statistics.PostProcessPassCount;
    }
    output=std::move(targets.Capture); return true;
}
bool SnapshotTSROpaque()
{
    if (!TSRScene.Active || TSRScene.OpaqueReady || !OpaqueEffectsComplete() || !(DeferredActive?DeferredResolved:ForwardColorWritten) || (FrameSampleCount>1 && !SamplesOpaqueResolved)) return Fail("TSR snapshot requires completed opaque effects exactly once");
    if (!EnsureTSR()) return false;
    struct Restore
    {
        Impl& Owner; GLint Read=0,Draw=0,Texture=0,Unpack=0; bool Scissor=false;
        explicit Restore(Impl& owner):Owner(owner){Owner.GlGetIntegerv(TSRReadFramebufferBinding,&Read);Owner.GlGetIntegerv(TSRDrawFramebufferBinding,&Draw);Owner.GlGetIntegerv(TSRTextureBinding,&Texture);Owner.GlGetIntegerv(TSRPixelUnpackBinding,&Unpack);Scissor=Owner.TSRIsEnabled(TSRScissorTest)!=0;Owner.GlDisable(TSRScissorTest);Owner.GlBindBuffer(TSRPixelUnpackBuffer,0);}
        ~Restore(){Owner.GlBindFramebuffer(GL_READ_FRAMEBUFFER,Read);Owner.GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,Draw);Owner.GlBindTexture(GL_TEXTURE_2D,Texture);Owner.GlBindBuffer(TSRPixelUnpackBuffer,Unpack);if(Scissor)Owner.GlEnable(TSRScissorTest);}
    } restore(*this);
    if (TSROpaqueWidth!=unsigned(CreateInfo.Width) || TSROpaqueHeight!=unsigned(CreateInfo.Height))
    {
        if (TSROpaqueFBO) GlDeleteFramebuffers(1,&TSROpaqueFBO); if (TSROpaque) GlDeleteTextures(1,&TSROpaque); TSROpaqueFBO=TSROpaque=0;
        GlGenTextures(1,&TSROpaque); GlBindTexture(GL_TEXTURE_2D,TSROpaque); GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        GlGenFramebuffers(1,&TSROpaqueFBO); GlBindFramebuffer(GL_FRAMEBUFFER,TSROpaqueFBO); GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,TSROpaque,0);
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE) return Fail("ES TSR opaque snapshot framebuffer is incomplete"); TSROpaqueWidth=CreateInfo.Width; TSROpaqueHeight=CreateInfo.Height;
    }
    GlBindFramebuffer(GL_READ_FRAMEBUFFER,HdrFramebuffers[0]); GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,TSROpaqueFBO); GlBlitFramebuffer(0,0,CreateInfo.Width,CreateInfo.Height,0,0,CreateInfo.Width,CreateInfo.Height,GL_COLOR_BUFFER_BIT,GL_NEAREST);
    if (!CheckGL("Snapshot TSR opaque HDR")) return false; TSRScene.OpaqueReady=true; return true;
}
bool FinishTSRScene()
{
    if (!TSRScene.Active) return true;
    TSRScene.MotionReady=VelocityCaptured && (FrameSampleCount==1 || SamplesVelocityResolved);
    if (!TSRScene.OpaqueReady || !TSRScene.MotionReady) return Fail("TSR presentation requires opaque snapshot and resolved physical motion");
    struct Restore { bool& Value; explicit Restore(bool& value):Value(value){Value=false;} ~Restore(){Value=true;} } restore(FrameOpen);
    TSRNativeImages images; images.Flags=TSRTextureMotion; images.Textures[TSRSceneFinal]=HdrTextures[0]; images.Textures[TSRSceneOpaque]=TSROpaque; images.Textures[TSRSceneMotion]=MotionTextures[MotionVelocity]; images.Textures[TSRSceneDepth]=DepthTexture;
    TSRResult candidate; if (!ReconstructTSRImages(TSRScene.Frame,images,TSRScene.Committed,candidate)) return false; TSRScene.Pending=candidate.Handle;
    FrameCapture display; return RenderTSRDisplay(candidate.Handle,FramePostProcessSettings,display) && ComposeTSRSceneCapture(TSRScene.Frame,candidate,std::move(display),PendingFrame,LastError);
}
