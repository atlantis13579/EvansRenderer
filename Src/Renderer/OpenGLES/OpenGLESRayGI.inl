// Seven fragment SSBOs are optional in ES. Visibility is shared with ray AO;
// material and light snapshots are replaced independently after complete uploads.
std::array<GLuint,2> RayGIImages{},RayGIFramebuffers{};
GLuint RayGIProgram=0;
int RayGIWidth=0,RayGIHeight=0;

void DestroyRayGITargets()
{
    if (GlDeleteFramebuffers) GlDeleteFramebuffers(2,RayGIFramebuffers.data());
    if (GlDeleteTextures) GlDeleteTextures(2,RayGIImages.data());
    RayGIFramebuffers={}; RayGIImages={}; RayGIWidth=RayGIHeight=0;
}
void DestroyRayGI()
{
    DestroyRayGITargets();
    if (RayGIProgram && GlDeleteProgram) GlDeleteProgram(RayGIProgram);
    RayGIProgram=0;
}

bool EnsureRayGIResources()
{
    if (!EnsureRayRadiance()) return false;
    if (!RayGIProgram) RayGIProgram=CompileStageProgram("RayTracedGI.glsl");
    if (!RayGIProgram) return false;
    if (RayGIWidth==CreateInfo.Width && RayGIHeight==CreateInfo.Height) return true;
    DestroyRayGITargets(); UnbindSampledTextures();
    GlGenTextures(2,RayGIImages.data()); GlGenFramebuffers(2,RayGIFramebuffers.data());
    bool complete=true;
    for (std::size_t index=0;index<RayGIImages.size();++index)
    {
        GlBindTexture(GL_TEXTURE_2D,RayGIImages[index]); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
        GlBindFramebuffer(GL_FRAMEBUFFER,RayGIFramebuffers[index]);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,RayGIImages[index],0);
        const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment);
        complete=complete && GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    }
    GlBindTexture(GL_TEXTURE_2D,0); GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!CheckGL("Create ray GI targets") || !complete)
    { DestroyRayGITargets(); return Fail("OpenGL ES ray GI framebuffer is incomplete"); }
    RayGIWidth=CreateInfo.Width; RayGIHeight=CreateInfo.Height; return true;
}
bool ApplyRayGI(const GlobalIlluminationDesc& desc)
{
    if (!EnsureRayGIResources()) return false;
    RayTracedGIConstants constants;
    const auto& frame=FrameRayRadiance;
    if (!BuildRayTracedGIConstants(desc,FrameRayRadianceSettings,FrameGI.RayTracing,frame,
        CreateInfo.Width,CreateInfo.Height,Techniques::RayGIPass::Trace,constants)) return Fail("OpenGL ES ray GI frame is invalid");
    const auto location=[&](const char* name) { return GlGetUniformLocation(RayGIProgram,name); };
    const auto uint4=[&](const char* name,const ShaderUInt4& value)
    { GlUniform4ui(location(name),value.X,value.Y,value.Z,value.W); };
    const auto restore=[&]()
    {
        UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0);
        for (GLuint slot=0;slot<RaySceneBufferCount+RayRadianceBufferCount;++slot)
            GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,0);
        GlDisable(GL_BLEND); GlBlendEquation(GL_FUNC_ADD);
        GlBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
        GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlDepthMask(GL_TRUE_VALUE);
    };
    UnbindSampledTextures(); GlUseProgram(RayGIProgram);
    const auto inverse=ToColumnMajor(constants.InverseViewProjection);
    GlUniformMatrix4fv(location("WorldFromClip"),1,GL_FALSE_VALUE,inverse.data());
    UploadFloat4(location("FrameSize"),constants.Viewport); UploadFloat4(location("RayParameters"),constants.Trace);
    UploadFloat4(location("EyeForward"),constants.CameraForward);
    UploadFloat4(location("MaterialParameters"),constants.Material);
    uint4("GeometryCounts",constants.SceneCounts); uint4("SurfaceCounts",constants.ShadingCounts); uint4("LightControl",constants.Lighting);
    constexpr std::array<const char*,6> samplers{
        "DepthImage","NormalImage","DiffuseImage","AmbientImage","PreviousRadiance","OpaqueColor"};
    GlUniform1i(location("ColorAtlas"),static_cast<GLint>(samplers.size()));
    for (std::size_t slot=0;slot<samplers.size();++slot) GlUniform1i(location(samplers[slot]),static_cast<GLint>(slot));
    for (GLuint slot=0;slot<RaySceneBufferCount;++slot) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,RaySceneBuffers[slot]);
    for (GLuint slot=0;slot<RayRadianceBufferCount;++slot)
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot+RaySceneBufferCount,RayRadianceBuffers[slot]);
    GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE);
    GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDepthMask(GL_FALSE_VALUE); GlBindVertexArray(VertexArray);
    for (const auto pass:RayGIPasses)
    {
        const auto index=static_cast<std::size_t>(pass);
        const bool composite=pass==Techniques::RayGIPass::Composite;
        const GLuint previous=index ? RayGIImages[(index-1)%RayGIImages.size()] : 0;
        UnbindSampledTextures();
        GlBindFramebuffer(GL_FRAMEBUFFER,composite ? LightingFramebuffer : RayGIFramebuffers[index%RayGIImages.size()]);
        GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,composite ? GL_FALSE_VALUE : GL_TRUE_VALUE);
        if (composite) { GlEnable(GL_BLEND); GlBlendEquation(GL_FUNC_ADD); GlBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE); }
        else GlDisable(GL_BLEND);
        constants.Options.X=static_cast<GLuint>(pass); uint4("Control",constants.Options);
        const std::array<GLuint,6> textures=composite ? std::array<GLuint,6>{0,0,0,0,previous,0} :
            std::array<GLuint,6>{DepthTexture,GBufferTextures[1],GBufferTextures[0],GBufferTextures[3],previous,HdrTextures[0]};
        for (std::size_t slot=0;slot<textures.size();++slot)
        { GlActiveTexture(GL_TEXTURE0+static_cast<GLenum>(slot)); GlBindTexture(GL_TEXTURE_2D,textures[slot]); }
        GlActiveTexture(GL_TEXTURE0+static_cast<GLenum>(textures.size())); GlBindTexture(GL_TEXTURE_2D,RayRadianceColorAtlas);
        GlDrawArrays(GL_TRIANGLES,0,3);
        if (!CheckGL("Draw ray GI pass")) { restore(); return false; }
        PendingFrame.RayGIPasses.push_back(pass);
    }
    restore(); GI.Resolved=true;
    PendingFrame.Statistics.GlobalIlluminationPassCount=static_cast<std::uint32_t>(PendingFrame.RayGIPasses.size()); return true;
}
