// An integer image carries the queue between ES compute and vertex pulling.
// Depth and ownership are separate epochs: equal-depth ties retain submission priority.
NaniteHardwareWork HardwarePlan;
std::shared_ptr<NaniteFrameBuffer> HardwareRanges,HardwareArguments;
GLuint HardwareUniform=0,HardwareQueueImage=0,HardwareFramebuffer=0,HardwareOwnerImage=0;
unsigned HardwareImageWidth=0,HardwareImageHeight=0,HardwareOwnerWidth=0,HardwareOwnerHeight=0;
static constexpr unsigned HardwareEmptyOwner=1u<<24; // Every smaller positive integer is exact in float32.
static_assert(NaniteRasterMaximumTriangles*2+1<HardwareEmptyOwner,"Hardware ownership keys exceed float32 integer precision");
std::array<GLuint,4> HardwareKernels{};
std::array<GLuint,3> HardwareGraphics{};
void(EVANS_GL_CALL* HardwareDrawIndirect)(GLenum,const void*)=nullptr;
void(EVANS_GL_CALL* HardwareTextureStorage)(GLenum,GLsizei,GLenum,GLsizei,GLsizei)=nullptr;
void(EVANS_GL_CALL* HardwareBindImage)(GLuint,GLuint,GLint,GLboolean,GLint,GLenum,GLenum)=nullptr;
void DestroyNaniteHardware()
{
    HardwarePlan={}; HardwareRanges.reset(); HardwareArguments.reset();
    for (auto program:HardwareKernels) if (program && GlDeleteProgram) GlDeleteProgram(program); HardwareKernels={};
    for (auto program:HardwareGraphics) if (program && GlDeleteProgram) GlDeleteProgram(program); HardwareGraphics={};
    if (HardwareUniform && GlDeleteBuffers) GlDeleteBuffers(1,&HardwareUniform); HardwareUniform=0;
    if (HardwareQueueImage && GlDeleteTextures) GlDeleteTextures(1,&HardwareQueueImage); HardwareQueueImage=0;
    if (HardwareFramebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1,&HardwareFramebuffer); HardwareFramebuffer=0;
    if (HardwareOwnerImage && GlDeleteTextures) GlDeleteTextures(1,&HardwareOwnerImage); HardwareOwnerImage=0;
    HardwareOwnerWidth=HardwareOwnerHeight=0;
    HardwareImageWidth=HardwareImageHeight=0;
}
bool SupportsNaniteHardwareBins() const
{
    const auto* bytes=GlGetString(GL_EXTENSIONS);
    const std::string extensions=" "+std::string(bytes?reinterpret_cast<const char*>(bytes):"")+" ";
    return FloatDepthTargetSupported && extensions.find(" GL_EXT_float_blend ")!=std::string::npos;
}
bool PrepareNaniteHardware(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteHardwareWork(frame,CreateInfo.Width,CreateInfo.Height,HardwarePlan,LastError)) return false;
    if (!HardwarePlan.Active()) return true;
    if (!SupportsNaniteHardwareBins()) return Fail("ES hardware ownership requires R32F rendering and EXT_float_blend");
    if (!LoadGL(HardwareDrawIndirect,"glDrawArraysIndirect") || !LoadGL(HardwareTextureStorage,"glTexStorage2D") || !LoadGL(HardwareBindImage,"glBindImageTexture")) return false;
    const unsigned width=std::min(unsigned(MaximumTextureSize),HardwarePlan.QueueWords),height=(HardwarePlan.QueueWords+width-1)/width;
    const auto imageWords=std::size_t(width)*height;
    const auto otherWords=HardwarePlan.ArgumentWords+HardwarePlan.Constants.Counts.W+HardwarePlan.Ranges.size()*4;
    const unsigned items=std::max(HardwarePlan.Constants.Counts.X,std::max(HardwarePlan.Constants.Counts.W,HardwarePlan.ArgumentWords));
    const unsigned rows=(items+NaniteHardwareThreads*NaniteHardwareColumns-1)/(NaniteHardwareThreads*NaniteHardwareColumns);
    if (height>unsigned(MaximumTextureSize) || (imageWords+otherWords)*sizeof(unsigned)>NaniteHardwareMaximumBytes ||
        NaniteHardwareColumns>unsigned(NaniteMaximumGroupsX) || rows>unsigned(NaniteMaximumGroupsY)) return Fail("ES hardware schedule exceeds image/dispatch limits");
    HardwarePlan.Constants.Images.Z=width;
    for (auto& bin:HardwarePlan.Bins) bin.Constants.Schedule.W=width;
    if (!CreateNaniteFrameBuffer(unsigned(HardwarePlan.Ranges.size()*4),HardwarePlan.Ranges.data(),false,HardwareRanges) ||
        !CreateNaniteFrameBuffer(HardwarePlan.ArgumentWords,nullptr,true,HardwareArguments)) return false;
    GLint texture=0,uniform=0; GlGetIntegerv(VisibilityTexture2DBinding,&texture); GlGetIntegerv(GpuUniformBinding,&uniform);
    if (HardwareImageWidth!=width || HardwareImageHeight!=height)
    {
        if (HardwareQueueImage) GlDeleteTextures(1,&HardwareQueueImage); GlGenTextures(1,&HardwareQueueImage);
        GlBindTexture(GL_TEXTURE_2D,HardwareQueueImage); ConfigureDepthTexture(); HardwareTextureStorage(GL_TEXTURE_2D,1,VisibilityR32UI,width,height);
        GlBindTexture(GL_TEXTURE_2D,GLuint(texture)); HardwareImageWidth=width; HardwareImageHeight=height;
    }
    if (HardwareOwnerWidth!=unsigned(CreateInfo.Width) || HardwareOwnerHeight!=unsigned(CreateInfo.Height))
    {
        if (HardwareOwnerImage) GlDeleteTextures(1,&HardwareOwnerImage); GlGenTextures(1,&HardwareOwnerImage);
        GlBindTexture(GL_TEXTURE_2D,HardwareOwnerImage); ConfigureDepthTexture(); HardwareTextureStorage(GL_TEXTURE_2D,1,GL_R32F,CreateInfo.Width,CreateInfo.Height);
        GlBindTexture(GL_TEXTURE_2D,GLuint(texture)); HardwareOwnerWidth=CreateInfo.Width; HardwareOwnerHeight=CreateInfo.Height;
    }
    if (!HardwareUniform) GlGenBuffers(1,&HardwareUniform);
    GlBindBuffer(GL_UNIFORM_BUFFER,HardwareUniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(NaniteHardwareDrawConstants),nullptr,GL_DYNAMIC_DRAW);
    GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(uniform));
    if (!CheckGL("Prepare hardware schedule storage")) { HardwareImageWidth=HardwareImageHeight=HardwareOwnerWidth=HardwareOwnerHeight=0; return false; }
    GLint previousDraw=0,previousRead=0; GlGetIntegerv(NaniteDrawBinding,&previousDraw); GlGetIntegerv(NaniteReadBinding,&previousRead);
    if (!HardwareFramebuffer) GlGenFramebuffers(1,&HardwareFramebuffer);
    GlBindFramebuffer(GL_FRAMEBUFFER,HardwareFramebuffer);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,HardwareOwnerImage,0);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,DepthTexture,0);
    const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment); GlReadBuffer(GL_COLOR_ATTACHMENT0);
    const auto complete=GlCheckFramebufferStatus(GL_FRAMEBUFFER);
    GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,GLuint(previousDraw)); GlBindFramebuffer(GL_READ_FRAMEBUFFER,GLuint(previousRead));
    if (complete!=GL_FRAMEBUFFER_COMPLETE) return Fail("Incomplete ES hardware coverage framebuffer");
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","NaniteHardware.glsl",source,LastError)) return false;
    for (unsigned pass=0;pass<HardwareKernels.size();++pass) if (!HardwareKernels[pass])
    {
        auto code=source; code.insert(code.find('\n')+1,"#define EVANS_HARDWARE_COMPUTE "+std::to_string(pass)+"\n");
        const auto program=CompileComputeProgram(code); if(!program)return false;
        HardwareKernels[pass]=program;
    }
    for (unsigned pass=0;pass<HardwareGraphics.size();++pass) if (!HardwareGraphics[pass])
    {
        const auto defines=VisibilityShaderDefines()+"#define EVANS_HARDWARE_GRAPHICS 1\n#define EVANS_HARDWARE_PASS "+std::to_string(pass)+"\n";
        HardwareGraphics[pass]=CompileStageProgram("NaniteHardware.glsl",defines.c_str()); if (!HardwareGraphics[pass]) return false;
    }
    return true;
}
bool DrawNaniteHardware(const VisibilityBufferDesc&)
{
    if (!HardwarePlan.Active()) return true;
    NaniteFrameBindings bindings(*this);
    constexpr GLenum DrawIndirectBinding=0x8F43,ImageName=0x8F3A,ImageLevel=0x8F3B,ImageLayered=0x8F3C,ImageLayer=0x8F3D,ImageAccess=0x8F3E,ImageFormat=0x906E,WriteOnly=0x88B9;
    constexpr GLbitfield ImageBarrier=0x00000020;
    GLint indirect=0,priorImage[6]{},textures[3]{},samplers[3]{};
    GlGetIntegerv(DrawIndirectBinding,&indirect);
    const GLenum fields[]{ImageName,ImageLevel,ImageLayered,ImageLayer,ImageAccess,ImageFormat};
    for (unsigned field=0;field<6;++field) VisibilityGetIndexed(fields[field],0,&priorImage[field]);
    for (GLuint unit=1;unit<=3;++unit)
    { GlActiveTexture(GL_TEXTURE0+unit); GlGetIntegerv(NaniteTextureBinding,&textures[unit-1]); GlGetIntegerv(NaniteSamplerBinding,&samplers[unit-1]); }
    const auto restore=[&]
    {
        GlDisable(GL_BLEND); GlBlendEquation(GL_FUNC_ADD); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
        GlBindBuffer(GpuDrawIndirectBuffer,GLuint(indirect));
        HardwareBindImage(0,GLuint(priorImage[0]),priorImage[1],GLboolean(priorImage[2]),priorImage[3],GLenum(priorImage[4]),GLenum(priorImage[5]));
        for (GLuint unit=1;unit<=3;++unit)
        { GlActiveTexture(GL_TEXTURE0+unit); GlBindTexture(GL_TEXTURE_2D,GLuint(textures[unit-1])); NaniteBindSampler(unit,GLuint(samplers[unit-1])); }
    };
    struct Restore { const decltype(restore)& Function; ~Restore() { Function(); } } state{restore};
    GlBindFramebuffer(GL_FRAMEBUFFER,0); UnbindSampledTextures();
    HardwareBindImage(0,HardwareQueueImage,0,GL_FALSE_VALUE,0,WriteOnly,VisibilityR32UI);
    GlBindBuffer(GL_UNIFORM_BUFFER,HardwareUniform); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(HardwarePlan.Constants),&HardwarePlan.Constants);
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,HardwareUniform);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,NaniteRasterOutput->Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,HardwareArguments->Buffer);
    const auto& counts=HardwarePlan.Constants.Counts;
    for (unsigned pass=0;pass<HardwareKernels.size();++pass)
    {
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,HardwareRanges->Buffer);
        const unsigned items=pass==0?counts.Z:counts.X;
        const unsigned rows=(items+NaniteHardwareThreads*NaniteHardwareColumns-1)/(NaniteHardwareThreads*NaniteHardwareColumns);
        GlUseProgram(HardwareKernels[pass]); GpuDispatch(pass==2?1:NaniteHardwareColumns,pass==2?1:std::max(1u,rows),1);
        GpuMemoryBarrier(GpuStorageBarrier|ImageBarrier);
    }
    GpuMemoryBarrier(GpuStorageBarrier|ImageBarrier|GpuCommandBarrier|NaniteTextureFetchBarrier);
    HardwareBindImage(0,0,0,GL_FALSE_VALUE,0,WriteOnly,VisibilityR32UI);
    BindVisibilityGeometry(0); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,NaniteRasterInput->Buffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,0); GlBindBuffer(GpuDrawIndirectBuffer,HardwareArguments->Buffer);
    GlActiveTexture(GL_TEXTURE0+2); NaniteBindSampler(2,0); GlBindTexture(GL_TEXTURE_2D,HardwareQueueImage);
    GlBindFramebuffer(GL_FRAMEBUFFER,HardwareFramebuffer); GlBindVertexArray(VisibilityVertexArray); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height);
    // Float32 represents every priority/orientation key exactly. MIN blending
    // provides the same ordering as atomicMin without fragment SSBO side effects
    // on ES translation layers. Only the queue and headers use compute storage.
    GlDisable(GL_BLEND); GlEnable(GL_DEPTH_TEST); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
    const GLfloat emptyOwner[]{float(HardwareEmptyOwner),0,0,0}; GlClearBufferfv(GL_COLOR,0,emptyOwner);
    GlColorMask(GL_FALSE_VALUE,GL_FALSE_VALUE,GL_FALSE_VALUE,GL_FALSE_VALUE);
    const auto bind=[&](const NaniteHardwareBin& bin,unsigned pass)
    {
        const auto program=HardwareGraphics[pass]; GlUseProgram(program);
        GlBindBuffer(GL_UNIFORM_BUFFER,HardwareUniform); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(bin.Constants),&bin.Constants);
        GlUniform1i(GlGetUniformLocation(program,"HardwareQueue"),2);
        GlUniform1i(GlGetUniformLocation(program,"HardwareOwners"),3);
        const TextureHandle handles[]{bin.Diffuse,bin.Opacity}; const char* names[]{"RasterDiffuse","RasterOpacity"};
        for (GLuint unit=0;unit<2;++unit)
        {
            GLuint texture=0;
            if (handles[unit])
            { const auto found=Textures.find(handles[unit].Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Invalid ES hardware coverage texture"); texture=found->second.Texture; }
            GlActiveTexture(GL_TEXTURE0+unit); NaniteBindSampler(unit,0); GlBindTexture(GL_TEXTURE_2D,texture); GlUniform1i(GlGetUniformLocation(program,names[unit]),unit);
        }
        return true;
    };
    for (unsigned pass=0;pass<2;++pass)
    {
        if (pass) { constexpr GLenum MinimumBlend=0x8007; GlEnable(GL_BLEND); GlBlendEquation(MinimumBlend); }
        GlColorMask(pass?GL_TRUE_VALUE:GL_FALSE_VALUE,pass?GL_TRUE_VALUE:GL_FALSE_VALUE,pass?GL_TRUE_VALUE:GL_FALSE_VALUE,pass?GL_TRUE_VALUE:GL_FALSE_VALUE);
        GlDepthMask(pass?GL_FALSE_VALUE:GL_TRUE_VALUE); GlDepthFunc(pass?GL_EQUAL:GL_LESS);
        for (unsigned index=0;index<HardwarePlan.Bins.size();++index)
        {
            const auto& bin=HardwarePlan.Bins[index]; if (!bind(bin,pass)) return false;
            if (bin.Cull) GlEnable(GL_CULL_FACE); else GlDisable(GL_CULL_FACE);
            GlCullFace(GL_BACK); GlFrontFace(bin.Mirrored?GL_CW:GL_CCW);
            HardwareDrawIndirect(GL_TRIANGLES,reinterpret_cast<const void*>(std::uintptr_t(index*NaniteHardwareBinWords*sizeof(unsigned))));
        }
    }
    GlBindFramebuffer(GL_FRAMEBUFFER,VisibilityFramebuffer); GlDisable(GL_BLEND); GlBlendEquation(GL_FUNC_ADD);
    GpuMemoryBarrier(NaniteFramebufferBarrier|NaniteTextureFetchBarrier);
    GlActiveTexture(GL_TEXTURE0+3); NaniteBindSampler(3,0); GlBindTexture(GL_TEXTURE_2D,HardwareOwnerImage);
    GpuMemoryBarrier(GpuStorageBarrier); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
    GlDepthMask(GL_FALSE_VALUE); GlDepthFunc(GL_ALWAYS); GlDisable(GL_CULL_FACE);
    if (!bind(HardwarePlan.Bins.front(),2)) return false; GlDrawArrays(GL_TRIANGLES,0,3);
    auto& stats=PendingFrame.Statistics; stats.NaniteHardwareBinCount+=unsigned(HardwarePlan.Bins.size()); stats.NaniteHardwareDispatchCount+=unsigned(HardwareKernels.size());
    stats.NaniteHardwareIndirectCount+=unsigned(HardwarePlan.Bins.size()*2); stats.IndirectDrawCallCount+=unsigned(HardwarePlan.Bins.size()*2);
    return CheckGL("Compact and rasterize Nanite hardware bins");
}
