// Separate snapshots make every draw's attachment distinct from its samplers.
// The standard framebuffer blit averages the sample-frequency diffuse seed.
enum SubsurfaceImage : std::size_t { SubsurfaceScene, SubsurfaceSource, SubsurfaceHorizontal, SubsurfaceVertical, SubsurfaceImageCount };
std::array<GLuint,SubsurfaceImageCount> SubsurfaceImages{},SubsurfaceFramebuffers{};
GLuint SubsurfaceProgram=0,SubsurfaceUniforms=0;
int SubsurfaceWidth=0,SubsurfaceHeight=0;
bool SubsurfaceProducing=false,SubsurfaceResolved=false,SubsurfaceHasScene=false;
DeferredLightingDesc SubsurfaceLighting;
Techniques::SubsurfaceScatteringSettings FrameSubsurface;
bool SubsurfaceRequested=false;
bool SubsurfaceComplete() const { return !SubsurfaceRequested || SubsurfaceResolved; }

void DestroySubsurfaceTargets()
{
    if (GlDeleteFramebuffers) GlDeleteFramebuffers(static_cast<GLsizei>(SubsurfaceImageCount),SubsurfaceFramebuffers.data());
    if (GlDeleteTextures) GlDeleteTextures(static_cast<GLsizei>(SubsurfaceImageCount),SubsurfaceImages.data());
    SubsurfaceImages={}; SubsurfaceFramebuffers={}; SubsurfaceWidth=SubsurfaceHeight=0;
}
void DestroySubsurface()
{
    DestroySubsurfaceTargets();
    if (GlDeleteProgram && SubsurfaceProgram) GlDeleteProgram(SubsurfaceProgram);
    if (GlDeleteBuffers && SubsurfaceUniforms) GlDeleteBuffers(1,&SubsurfaceUniforms);
    SubsurfaceProgram=SubsurfaceUniforms=0;
}
bool EnsureSubsurface()
{
    if (!SubsurfaceProgram) SubsurfaceProgram=CompileStageProgram("SubsurfaceScattering.glsl");
    if (!SubsurfaceProgram) return false;
    if (!SubsurfaceUniforms)
    {
        GlGenBuffers(1,&SubsurfaceUniforms); GlBindBuffer(GL_UNIFORM_BUFFER,SubsurfaceUniforms);
        GlBufferData(GL_UNIFORM_BUFFER,sizeof(SubsurfaceScatteringConstants),nullptr,GL_DYNAMIC_DRAW);
    }
    if (SubsurfaceWidth==CreateInfo.Width && SubsurfaceHeight==CreateInfo.Height) return true;
    DestroySubsurfaceTargets(); UnbindSampledTextures();
    GlGenTextures(static_cast<GLsizei>(SubsurfaceImageCount),SubsurfaceImages.data());
    GlGenFramebuffers(static_cast<GLsizei>(SubsurfaceImageCount),SubsurfaceFramebuffers.data());
    bool complete=true;
    for (std::size_t index=0;index<SubsurfaceImageCount;++index)
    {
        GlBindTexture(GL_TEXTURE_2D,SubsurfaceImages[index]); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
        GlBindFramebuffer(GL_FRAMEBUFFER,SubsurfaceFramebuffers[index]);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,SubsurfaceImages[index],0);
        const GLenum color=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&color);
        complete=complete && GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    }
    GlBindTexture(GL_TEXTURE_2D,0); GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!complete || !CheckGL("Create subsurface targets")) { DestroySubsurfaceTargets(); return Fail("OpenGL ES subsurface framebuffer is incomplete"); }
    SubsurfaceWidth=CreateInfo.Width; SubsurfaceHeight=CreateInfo.Height; return true;
}
bool SubsurfaceCopy(GLuint source,GLuint destination)
{
    UnbindSampledTextures(); GlBindFramebuffer(GL_READ_FRAMEBUFFER,source); GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,destination);
    GlBlitFramebuffer(0,0,CreateInfo.Width,CreateInfo.Height,0,0,CreateInfo.Width,CreateInfo.Height,GL_COLOR_BUFFER_BIT,GL_NEAREST);
    GlBindFramebuffer(GL_FRAMEBUFFER,0); return CheckGL("Copy subsurface radiance");
}
bool SnapshotSubsurfaceScene()
{ return EnsureSubsurface() && SubsurfaceCopy(LightingFramebuffer,SubsurfaceFramebuffers[SubsurfaceScene]); }
bool CaptureSubsurfaceSource()
{ return SubsurfaceCopy(FrameSampleCount>1 ? Multisample.Framebuffers[SampleLightingFBO] : LightingFramebuffer,SubsurfaceFramebuffers[SubsurfaceSource]); }
bool FilterSubsurface(SubsurfaceScatteringConstants constants)
{
    GlUseProgram(SubsurfaceProgram); GlBindVertexArray(VertexArray);
    GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlDisable(GL_BLEND); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE);
    GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDepthMask(GL_FALSE_VALUE);
    GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
    constexpr std::array<const char*,5> samplers{"DiffuseSignal","SceneDepth","NormalImage","SceneColor","OriginalDiffuse"};
    for (std::size_t slot=0;slot<samplers.size();++slot) GlUniform1i(GlGetUniformLocation(SubsurfaceProgram,samplers[slot]),static_cast<GLint>(slot));
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,SubsurfaceUniforms);
    for (auto pass:{Techniques::SubsurfacePass::Horizontal,Techniques::SubsurfacePass::Vertical,Techniques::SubsurfacePass::Composite})
    {
        const bool horizontal=pass==Techniques::SubsurfacePass::Horizontal,composite=pass==Techniques::SubsurfacePass::Composite;
        UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,composite ? LightingFramebuffer : SubsurfaceFramebuffers[horizontal ? SubsurfaceHorizontal : SubsurfaceVertical]);
        constants.Options.X=static_cast<unsigned>(pass); GlBindBuffer(GL_UNIFORM_BUFFER,SubsurfaceUniforms);
        GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants);
        const std::array<GLuint,5> inputs{SubsurfaceImages[horizontal ? SubsurfaceSource : composite ? SubsurfaceVertical : SubsurfaceHorizontal],
            DepthTexture,GBufferTextures[1],SubsurfaceImages[SubsurfaceScene],SubsurfaceImages[SubsurfaceSource]};
        for (std::size_t slot=0;slot<inputs.size();++slot)
        { GlActiveTexture(GL_TEXTURE0+static_cast<GLenum>(slot)); GlBindTexture(GL_TEXTURE_2D,inputs[slot]); }
        GlDrawArrays(GL_TRIANGLES,0,3);
        if (!CheckGL("Draw subsurface diffusion")) { UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0); return false; }
        ++PendingFrame.Statistics.SubsurfacePassCount;
    }
    UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindBufferBase(GL_UNIFORM_BUFFER,0,0);
    GlDepthMask(GL_TRUE_VALUE); SubsurfaceResolved=true; return true;
}
