// FP32 running means are separate from the display HDR image. A cancelled frame
// may overwrite the uncommitted slot, but never the last published mean.
PathTracingFrameState PathFrame;
std::array<GLuint,2> PathMeans{},PathFramebuffers{},PathPrograms{};
GLuint PathDepth=0,PathComposeFramebuffer=0;
int PathWidth=0,PathHeight=0;
bool PathTargetsSupported=false;

bool ProbePathTracingSupport()
{
    if (!HalfFloatTargetsSupported || !FloatDepthTargetSupported || MaximumDrawBuffers<2 || MaximumColorAttachments<2) return false;
    // Weighted targets have the same RGBA32F + R32F attachment formats, without
    // enabling blending. Path tracing does not require EXT_float_blend.
    const auto previous=LastError; WeightedTargets probe;
    const bool supported=CreateWeightedTargets(probe,1,1,1);
    DestroyWeightedTargets(probe); LastError=previous; return supported;
}
void DestroyPathTargets()
{
    if (GlDeleteFramebuffers)
    {
        GlDeleteFramebuffers(2,PathFramebuffers.data());
        if (PathComposeFramebuffer) GlDeleteFramebuffers(1,&PathComposeFramebuffer);
    }
    if (GlDeleteTextures)
    {
        GlDeleteTextures(2,PathMeans.data());
        if (PathDepth) GlDeleteTextures(1,&PathDepth);
    }
    PathMeans={}; PathFramebuffers={}; PathDepth=PathComposeFramebuffer=0;
    PathWidth=PathHeight=0; PathFrame.History.Invalidate();
}
void DestroyPathTracing()
{
    DestroyPathTargets();
    for (auto program:PathPrograms) if (program && GlDeleteProgram) GlDeleteProgram(program);
    PathPrograms={}; PathFrame={};
}
bool EnsurePathTracing()
{
    if (!PathTargetsSupported) return Fail("OpenGL ES path tracing requires FP32 render/load targets");
    if (!EnsureRayRadiance()) return false;
    for (std::size_t phase=0;phase<PathPrograms.size();++phase)
    {
        if (!PathPrograms[phase]) PathPrograms[phase]=CompileStageProgram("PathTracing.glsl",phase ? "#define EVANS_PATH_COMPOSE 1\n" : "");
        if (!PathPrograms[phase]) return false;
    }
    if (PathWidth==CreateInfo.Width && PathHeight==CreateInfo.Height) return true;
    DestroyPathTargets(); UnbindSampledTextures(); GlActiveTexture(GL_TEXTURE0);
    GlGenTextures(2,PathMeans.data()); GlGenTextures(1,&PathDepth);
    GlGenFramebuffers(2,PathFramebuffers.data()); GlGenFramebuffers(1,&PathComposeFramebuffer);
    GlBindTexture(GL_TEXTURE_2D,PathDepth); ConfigureDepthTexture();
    GlTexImage2D(GL_TEXTURE_2D,0,GL_R32F,CreateInfo.Width,CreateInfo.Height,0,GL_RED,GL_FLOAT,nullptr);
    const GLenum attachments[]{GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT0+1};
    bool complete=true;
    for (std::size_t index=0;index<PathMeans.size();++index)
    {
        GlBindTexture(GL_TEXTURE_2D,PathMeans[index]); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_FLOAT,nullptr);
        GlBindFramebuffer(GL_FRAMEBUFFER,PathFramebuffers[index]);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,attachments[0],GL_TEXTURE_2D,PathMeans[index],0);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,attachments[1],GL_TEXTURE_2D,PathDepth,0);
        GlDrawBuffers(2,attachments);
        complete=complete && GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    }
    GlBindFramebuffer(GL_FRAMEBUFFER,PathComposeFramebuffer);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,HdrTextures[0],0);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,DepthTexture,0);
    GlDrawBuffers(1,attachments);
    complete=complete && GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    GlBindTexture(GL_TEXTURE_2D,0); GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!CheckGL("Create path tracing targets") || !complete)
    { DestroyPathTargets(); return Fail("OpenGL ES path tracing framebuffer is incomplete"); }
    PathWidth=CreateInfo.Width; PathHeight=CreateInfo.Height; return true;
}
bool TracePaths(const PathTracingDesc& desc)
{
    PathTracingConstants constants;
    if (!PathFrame.Prepare(desc,FrameRayRadianceSettings,FrameRayRadiance,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("OpenGL ES path frame requires matched scene inputs and a finite invertible camera");
    const auto plan=PathFrame.History.GetPlan();
    const auto restore=[&]()
    {
        UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0);
        for (GLuint slot=0;slot<RaySceneBufferCount+RayRadianceBufferCount;++slot) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,0);
        GlDepthMask(GL_TRUE_VALUE); GlDepthFunc(GL_LESS);
    };
    GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlDisable(GL_BLEND); GlDisable(GL_CULL_FACE);
    GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlBindVertexArray(VertexArray);
    GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
    for (GLuint slot=0;slot<RaySceneBufferCount;++slot) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,RaySceneBuffers[slot]);
    for (GLuint slot=0;slot<RayRadianceBufferCount;++slot)
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot+RaySceneBufferCount,RayRadianceBuffers[slot]);
    for (unsigned phase=0;phase<2;++phase)
    {
        UnbindSampledTextures(); GlUseProgram(PathPrograms[phase]);
        const auto location=[&](const char* name) { return GlGetUniformLocation(PathPrograms[phase],name); };
        const auto uint4=[&](const char* name,const ShaderUInt4& value) { GlUniform4ui(location(name),value.X,value.Y,value.Z,value.W); };
        const auto inverse=ToColumnMajor(constants.InverseViewProjection),projection=ToColumnMajor(constants.ViewProjection);
        GlUniformMatrix4fv(location("WorldFromClip"),1,GL_FALSE_VALUE,inverse.data());
        GlUniformMatrix4fv(location("ClipFromWorld"),1,GL_FALSE_VALUE,projection.data());
        UploadFloat4(location("FrameSize"),constants.Viewport); UploadFloat4(location("RayParameters"),constants.Trace);
        UploadFloat4(location("BackgroundColor"),constants.Background); UploadFloat4(location("MaterialParameters"),constants.Material);
        uint4("GeometryCounts",constants.SceneCounts); uint4("SurfaceCounts",constants.ShadingCounts);
        uint4("Samples",constants.Sampling); uint4("Control",constants.Options);
        GlUniform1i(location("CommittedMean"),0); GlUniform1i(location("PathDepth"),1); GlUniform1i(location("ColorAtlas"),2);
        const GLuint textures[]{phase ? PathMeans[plan.WriteIndex] : (plan.PreviousSamples ? PathMeans[plan.ReadIndex] : 0),
            phase ? PathDepth : 0,RayRadianceColorAtlas};
        GlBindFramebuffer(GL_FRAMEBUFFER,phase ? PathComposeFramebuffer : PathFramebuffers[plan.WriteIndex]);
        if (phase) { GlEnable(GL_DEPTH_TEST); GlDepthFunc(GL_ALWAYS); GlDepthMask(GL_TRUE_VALUE); }
        else { GlDisable(GL_DEPTH_TEST); GlDepthMask(GL_FALSE_VALUE); }
        for (GLuint slot=0;slot<std::size(textures);++slot)
        { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,textures[slot]); }
        GlDrawArrays(GL_TRIANGLES,0,3);
        if (!CheckGL("Draw path tracing pass")) { restore(); return false; }
    }
    restore(); ForwardColorWritten=true; return PathFrame.Resolve(PendingFrame.Statistics);
}
