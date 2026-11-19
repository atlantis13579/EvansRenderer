bool ExposureSupported=false;
bool ProbeTemporalExposure()
{
    GLuint texture=0,framebuffer=0; GlGenTextures(1,&texture); GlBindTexture(GL_TEXTURE_2D,texture); ConfigureDepthTexture();
    GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,1,1,0,GL_RGBA,GL_FLOAT,nullptr);
    GlGenFramebuffers(1,&framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0);
    const GLenum color=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&color);
    const bool complete=GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    const bool clean=GlGetError()==GL_NO_ERROR; GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindTexture(GL_TEXTURE_2D,0);
    if (framebuffer) GlDeleteFramebuffers(1,&framebuffer); if (texture) GlDeleteTextures(1,&texture);
    return complete && clean;
}
TemporalExposureHistory ExposureHistory;
struct ExposureSurface { GLuint Texture=0,Framebuffer=0; int Width=0,Height=0; };
std::vector<ExposureSurface> ExposureLevels;
std::array<ExposureSurface,2> ExposureValues{};
GLuint ExposureProgram=0,ExposureUniforms=0;
int ExposureWidth=0,ExposureHeight=0;
void DestroyExposureTargets()
{
    const auto release=[&](ExposureSurface& image)
    {
        if (image.Framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1,&image.Framebuffer);
        if (image.Texture && GlDeleteTextures) GlDeleteTextures(1,&image.Texture);
        image={};
    };
    for (auto& image:ExposureLevels) release(image);
    for (auto& image:ExposureValues) release(image);
    ExposureLevels.clear(); ExposureWidth=ExposureHeight=0; ExposureHistory.Invalidate();
}
void DestroyExposure()
{
    DestroyExposureTargets();
    if (ExposureProgram && GlDeleteProgram) GlDeleteProgram(ExposureProgram);
    if (ExposureUniforms && GlDeleteBuffers) GlDeleteBuffers(1,&ExposureUniforms);
    ExposureProgram=ExposureUniforms=0;
}
bool EnsureExposure()
{
    if (!ExposureProgram) ExposureProgram=CompileStageProgram("TemporalEffects.glsl");
    if (!ExposureProgram) return false;
    if (!ExposureUniforms)
    {
        GlGenBuffers(1,&ExposureUniforms); GlBindBuffer(GL_UNIFORM_BUFFER,ExposureUniforms);
        GlBufferData(GL_UNIFORM_BUFFER,sizeof(TemporalEffectsConstants),nullptr,GL_DYNAMIC_DRAW);
    }
    if (ExposureWidth==CreateInfo.Width && ExposureHeight==CreateInfo.Height) return true;
    DestroyExposureTargets(); UnbindSampledTextures();
    const auto plan=BuildExposureReductionPlan(CreateInfo.Width,CreateInfo.Height); ExposureLevels.resize(plan.size());
    const auto create=[&](ExposureSurface& image,int width,int height)
    {
        image.Width=width; image.Height=height; GlGenTextures(1,&image.Texture); GlBindTexture(GL_TEXTURE_2D,image.Texture); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,width,height,0,GL_RGBA,GL_FLOAT,nullptr);
        GlGenFramebuffers(1,&image.Framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,image.Framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,image.Texture,0);
        const GLenum color=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&color);
        return GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    };
    bool complete=true;
    for (std::size_t index=0;index<plan.size();++index) complete=create(ExposureLevels[index],plan[index].Width,plan[index].Height) && complete;
    for (auto& value:ExposureValues) complete=create(value,1,1) && complete;
    GlBindTexture(GL_TEXTURE_2D,0); GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!complete || !CheckGL("Create exposure FP32 targets")) { DestroyExposureTargets(); return Fail("OpenGL ES exposure requires renderable RGBA32F"); }
    ExposureWidth=CreateInfo.Width; ExposureHeight=CreateInfo.Height; return true;
}
bool ResolveExposure(const TemporalEffectsDesc& desc)
{
    TemporalEffectsConstants constants;
    if (!ExposureHistory.Prepare(desc,FramePostProcessSettings.TemporalEffects,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("Exposure requires finite scene time and positive dimensions");
    GlUseProgram(ExposureProgram); GlBindVertexArray(VertexArray); GlDisable(GL_BLEND); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE);
    GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDepthMask(GL_FALSE_VALUE);
    GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlBindBufferBase(GL_UNIFORM_BUFFER,0,ExposureUniforms);
    const auto draw=[&](const ExposureSurface& source,const ExposureSurface& previous,const ExposureSurface& destination,Techniques::TemporalEffectsPassKind phase)
    {
        UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,destination.Framebuffer); GlViewport(0,0,destination.Width,destination.Height);
        constants.PhaseExtentHistory.X=static_cast<unsigned>(phase); constants.PhaseExtentHistory.Y=source.Width; constants.PhaseExtentHistory.Z=source.Height;
        GlBindBuffer(GL_UNIFORM_BUFFER,ExposureUniforms); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants);
        GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,source.Texture);
        GlActiveTexture(GL_TEXTURE0+1); GlBindTexture(GL_TEXTURE_2D,previous.Texture);
        GlDrawArrays(GL_TRIANGLES,0,3); ++PendingFrame.Statistics.TemporalEffectsPassCount;
    };
    ExposureSurface scene{HdrTextures[0],HdrFramebuffers[0],CreateInfo.Width,CreateInfo.Height}; const auto* source=&scene;
    for (std::size_t index=0;index<ExposureLevels.size();++index)
    { draw(*source,*source,ExposureLevels[index],index?Techniques::TemporalEffectsPassKind::Reduce:Techniques::TemporalEffectsPassKind::Meter); source=&ExposureLevels[index]; }
    const auto& current=ExposureValues[1-ExposureHistory.ReadIndex()];
    draw(*source,constants.PhaseExtentHistory.W?ExposureValues[ExposureHistory.ReadIndex()]:*source,current,Techniques::TemporalEffectsPassKind::Adapt);
    const ExposureSurface scratch{HdrTextures[1],HdrFramebuffers[1],CreateInfo.Width,CreateInfo.Height};
    draw(scene,current,scratch,Techniques::TemporalEffectsPassKind::Composite);
    UnbindSampledTextures(); GlBindFramebuffer(GL_READ_FRAMEBUFFER,scratch.Framebuffer); GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,scene.Framebuffer);
    GlBlitFramebuffer(0,0,CreateInfo.Width,CreateInfo.Height,0,0,CreateInfo.Width,CreateInfo.Height,GL_COLOR_BUFFER_BIT,GL_NEAREST);
    GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!CheckGL("Reduce/adapt/apply temporal exposure")) return false;
    ExposureHistory.Resolved=true; return true;
}
