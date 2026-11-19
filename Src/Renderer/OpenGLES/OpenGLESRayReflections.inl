// Seven fragment SSBOs are optional in ES. Visibility is shared with ray AO;
// material and light snapshots are replaced independently after complete uploads.
std::array<GLuint,2> RayReflectionImages{},RayReflectionFramebuffers{};
GLuint RayReflectionProgram=0;
int RayReflectionWidth=0,RayReflectionHeight=0;

void DestroyRayReflectionTargets()
{
    if (GlDeleteFramebuffers) GlDeleteFramebuffers(2,RayReflectionFramebuffers.data());
    if (GlDeleteTextures) GlDeleteTextures(2,RayReflectionImages.data());
    RayReflectionFramebuffers={}; RayReflectionImages={}; RayReflectionWidth=RayReflectionHeight=0;
}
void DestroyRayReflections()
{
    DestroyRayReflectionTargets();
    if (RayReflectionProgram && GlDeleteProgram) GlDeleteProgram(RayReflectionProgram);
    RayReflectionProgram=0;
}

bool EnsureRayReflectionResources()
{
    if (!EnsureRayRadiance()) return false;
    if (!RayReflectionProgram) RayReflectionProgram=CompileStageProgram("RayTracedReflections.glsl");
    if (!RayReflectionProgram) return false;
    if (RayReflectionWidth==CreateInfo.Width && RayReflectionHeight==CreateInfo.Height) return true;
    DestroyRayReflectionTargets(); UnbindSampledTextures();
    GlGenTextures(2,RayReflectionImages.data()); GlGenFramebuffers(2,RayReflectionFramebuffers.data());
    bool complete=true;
    for (std::size_t index=0;index<RayReflectionImages.size();++index)
    {
        GlBindTexture(GL_TEXTURE_2D,RayReflectionImages[index]); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
        GlBindFramebuffer(GL_FRAMEBUFFER,RayReflectionFramebuffers[index]);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,RayReflectionImages[index],0);
        const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment);
        complete=complete && GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    }
    GlBindTexture(GL_TEXTURE_2D,0); GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!CheckGL("Create ray reflection targets") || !complete)
    { DestroyRayReflectionTargets(); return Fail("OpenGL ES ray reflection framebuffer is incomplete"); }
    RayReflectionWidth=CreateInfo.Width; RayReflectionHeight=CreateInfo.Height; return true;
}
bool ApplyRayReflections(const ReflectionDesc& desc)
{
    if (!EnsureRayReflectionResources()) return false;
    RayTracedReflectionConstants constants;
    const auto& frame=FrameRayRadiance;
    if (!BuildRayTracedReflectionConstants(desc,FrameRayRadianceSettings,FrameReflections.RayTracing,
        *frame.Geometry,*frame.Shading,frame.Lighting,*frame.Lights,CreateInfo.Width,CreateInfo.Height,
        Techniques::RayReflectionPass::Trace,constants)) return Fail("OpenGL ES ray reflection frame is invalid");
    const auto location=[&](const char* name) { return GlGetUniformLocation(RayReflectionProgram,name); };
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
    UnbindSampledTextures(); GlUseProgram(RayReflectionProgram);
    const auto inverse=ToColumnMajor(constants.InverseViewProjection);
    GlUniformMatrix4fv(location("WorldFromClip"),1,GL_FALSE_VALUE,inverse.data());
    UploadFloat4(location("FrameSize"),constants.Viewport); UploadFloat4(location("RayParameters"),constants.Trace);
    UploadFloat4(location("EyeForward"),constants.CameraForward); UploadFloat4(location("AmbientLight"),constants.Ambient);
    UploadFloat4(location("MaterialParameters"),constants.Material);
    uint4("GeometryCounts",constants.SceneCounts); uint4("SurfaceCounts",constants.ShadingCounts); uint4("LightControl",constants.Lighting);
    constexpr std::array<const char*,7> samplers{
        "DepthImage","NormalImage","SpecularImage","DiffuseImage","AmbientImage","PreviousRadiance","OpaqueColor"};
    GlUniform1i(location("ColorAtlas"),static_cast<GLint>(samplers.size()));
    for (std::size_t slot=0;slot<samplers.size();++slot) GlUniform1i(location(samplers[slot]),static_cast<GLint>(slot));
    for (GLuint slot=0;slot<RaySceneBufferCount;++slot) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,RaySceneBuffers[slot]);
    for (GLuint slot=0;slot<RayRadianceBufferCount;++slot)
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot+RaySceneBufferCount,RayRadianceBuffers[slot]);
    GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE);
    GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDepthMask(GL_FALSE_VALUE); GlBindVertexArray(VertexArray);
    for (const auto pass:RayReflectionPasses)
    {
        const auto index=static_cast<std::size_t>(pass);
        const bool composite=pass==Techniques::RayReflectionPass::Composite;
        const GLuint previous=index ? RayReflectionImages[(index-1)%RayReflectionImages.size()] : 0;
        UnbindSampledTextures();
        GlBindFramebuffer(GL_FRAMEBUFFER,composite ? LightingFramebuffer : RayReflectionFramebuffers[index%RayReflectionImages.size()]);
        GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,composite ? GL_FALSE_VALUE : GL_TRUE_VALUE);
        if (composite) { GlEnable(GL_BLEND); GlBlendEquation(GL_FUNC_ADD); GlBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE); }
        else GlDisable(GL_BLEND);
        constants.Options.X=static_cast<GLuint>(pass); uint4("Control",constants.Options);
        const std::array<GLuint,7> textures=composite ? std::array<GLuint,7>{0,0,0,0,0,previous,0} :
            std::array<GLuint,7>{DepthTexture,GBufferTextures[1],GBufferTextures[2],GBufferTextures[0],GBufferTextures[3],previous,HdrTextures[0]};
        for (std::size_t slot=0;slot<textures.size();++slot)
        { GlActiveTexture(GL_TEXTURE0+static_cast<GLenum>(slot)); GlBindTexture(GL_TEXTURE_2D,textures[slot]); }
        GlActiveTexture(GL_TEXTURE0+static_cast<GLenum>(textures.size())); GlBindTexture(GL_TEXTURE_2D,RayRadianceColorAtlas);
        GlDrawArrays(GL_TRIANGLES,0,3);
        if (!CheckGL("Draw ray reflection pass")) { restore(); return false; }
        PendingFrame.ReflectionPasses.push_back(RayReflectionLogPass(pass));
    }
    restore(); ReflectionResolved=true;
    PendingFrame.Statistics.ReflectionPassCount=static_cast<std::uint32_t>(PendingFrame.ReflectionPasses.size()); return true;
}
