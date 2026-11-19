LumenRadianceHandle LumenReflectionRays;
GLuint LumenReflectionEstimate=0,LumenReflectionResolveProgram=0,LumenReflectionTexture=0,LumenReflectionFramebuffer=0;
int LumenReflectionWidth=0,LumenReflectionHeight=0;
std::size_t LumenReflectionEstimateBytes=0;
bool EnsureLumenReflectionTarget()
{
    if (LumenReflectionTexture && LumenReflectionWidth==CreateInfo.Width && LumenReflectionHeight==CreateInfo.Height) return true;
    GLuint texture=0,framebuffer=0; GlGenTextures(1,&texture); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,texture); ConfigureDepthTexture();
    GlBindBuffer(TSRPixelUnpackBuffer,0); GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
    GlGenFramebuffers(1,&framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0); const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment);
    const bool complete=GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    if (!CheckGL("Create Lumen reflection target") || !complete)
    { GlDeleteFramebuffers(1,&framebuffer); GlDeleteTextures(1,&texture); return Fail("Lumen reflection target allocation failed"); }
    if (LumenReflectionFramebuffer) GlDeleteFramebuffers(1,&LumenReflectionFramebuffer);
    if (LumenReflectionTexture) GlDeleteTextures(1,&LumenReflectionTexture);
    LumenReflectionTexture=texture; LumenReflectionFramebuffer=framebuffer; LumenReflectionWidth=CreateInfo.Width; LumenReflectionHeight=CreateInfo.Height; return true;
}
LumenTemporalHistory LumenSpecularHistory;
std::array<GLuint,2> LumenSpecularBuffers{};
GLuint LumenSpecularCompute=0,LumenSpecularComposite=0;
std::size_t LumenSpecularBytes=0;
void DestroyLumenSpecular()
{
    if (LumenSpecularBuffers[0] || LumenSpecularBuffers[1]) GlDeleteBuffers(GLsizei(LumenSpecularBuffers.size()),LumenSpecularBuffers.data()); LumenSpecularBuffers={}; LumenSpecularBytes=0;
    if (LumenSpecularCompute) GlDeleteProgram(LumenSpecularCompute); if (LumenSpecularComposite) GlDeleteProgram(LumenSpecularComposite);
    LumenSpecularCompute=LumenSpecularComposite=0; LumenSpecularHistory.Invalidate();
}
bool EnsureLumenSpecular(std::size_t bytes)
{
    if (!EnsureLumenReflectionTarget()) return false;
    if (LumenSpecularBytes!=bytes)
    {
        std::array<GLuint,2> buffers{}; GlGenBuffers(GLsizei(buffers.size()),buffers.data());
        for (auto buffer:buffers) { GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,bytes,nullptr,GL_DYNAMIC_DRAW); }
        if (!CheckGL("Allocate Lumen specular temporal history")) { GlDeleteBuffers(GLsizei(buffers.size()),buffers.data()); return false; }
        GlDeleteBuffers(GLsizei(LumenSpecularBuffers.size()),LumenSpecularBuffers.data()); LumenSpecularBuffers=buffers; LumenSpecularBytes=bytes; LumenSpecularHistory.Invalidate();
    }
    if (!LumenSpecularComposite) LumenSpecularComposite=CompileStageProgram("Lumen.glsl","#define EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE 1\n");
    if (!LumenSpecularComposite) return false;
    if (!LumenSpecularCompute)
    {
        std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
        const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen specular temporal shader has no version directive");
        source.insert(version+1,"#define EVANS_LUMEN_TEMPORAL_STAGE 1\n#define EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE 1\n"); const auto program=CompileComputeProgram(source); if(!program)return false;
        LumenSpecularCompute=program;
    }
    return true;
}
bool ResolveLumenSpecular(const ReflectionDesc& camera)
{
    LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    key.Reflections=FrameReflections; key.Reflected=true; LumenTemporalFrame frame;
    if (!BuildLumenReflectionTemporalFrame(camera,FrameReflections,CreateInfo.Width,CreateInfo.Height,LumenSpecularHistory.CanReuse(key),frame,LastError)) return false;
    LumenCaptureState state(*this); LumenRadianceStorageState storage(*this); LumenCaptureBuffers constants(*this);
    if (!EnsureLumenSpecular(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    GlGenBuffers(1,constants.Buffers.data()); GlBindBuffer(GL_UNIFORM_BUFFER,constants.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW);
    if (!CheckGL("Allocate Lumen specular temporal constants")) return false;
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,constants.Buffers[0]); const auto read=LumenSpecularHistory.ReadIndex(),write=LumenSpecularHistory.WriteIndex();
    GlBindFramebuffer(GL_FRAMEBUFFER,0); GlUseProgram(LumenSpecularCompute);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,LumenSpecularBuffers[read]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,LumenSpecularBuffers[write]);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,LumenReflectionEstimate);
    const GLuint inputs[]{HdrTextures[0],DepthTexture,GBufferTextures[1],MotionTextures[MotionVelocity]?MotionTextures[MotionVelocity]:GBufferTextures[1]};
    const char* names[]{"TemporalCurrent","TemporalDepth","TemporalNormals","TemporalMotion"};
    for (unsigned slot=0;slot<std::size(inputs);++slot)
    { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,inputs[slot]); LumenBindSampler(slot,0); GlUniform1i(GlGetUniformLocation(LumenSpecularCompute,names[slot]),slot); }
    GpuMemoryBarrier(GpuStorageBarrier); const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    GpuDispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1); GpuMemoryBarrier(GpuStorageBarrier);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,0); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,LumenSpecularBuffers[write]);
    GlUseProgram(LumenSpecularComposite); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]); GlUniform1i(GlGetUniformLocation(LumenSpecularComposite,"TemporalCurrent"),0);
    for (auto capability:LumenCaptureState::Switches) GlDisable(capability);
    GlDepthMask(GL_FALSE_VALUE); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlBindVertexArray(VertexArray);
    GlBindFramebuffer(GL_FRAMEBUFFER,LumenReflectionFramebuffer); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlDrawArrays(GL_TRIANGLES,0,3);
    for (unsigned slot=0;slot<std::size(inputs);++slot) { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,0); }
    GlReadBuffer(GL_COLOR_ATTACHMENT0); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]);
    GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,CreateInfo.Width,CreateInfo.Height);
    if (!CheckGL("Resolve Lumen specular temporal reflection")) return false;
    LumenSpecularHistory.Record(key); PendingFrame.Temporal=camera.Sampling; PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::LumenTemporal);
    PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::Composite); PendingFrame.Statistics.ReflectionPassCount+=2; return true;
}

void FinishLumenReflections(bool publish)
{
    LumenSpecularHistory.Finish(publish);
    if (LumenReflectionRays) LumenRadianceTraces.erase(LumenReflectionRays.Value); LumenReflectionRays={};
}
void DestroyLumenReflections()
{
    FinishLumenReflections(false); DestroyLumenSpecular();
    if (LumenReflectionResolveProgram) GlDeleteProgram(LumenReflectionResolveProgram);
    if (LumenReflectionEstimate) GlDeleteBuffers(1,&LumenReflectionEstimate);
    if (LumenReflectionFramebuffer) GlDeleteFramebuffers(1,&LumenReflectionFramebuffer);
    if (LumenReflectionTexture) GlDeleteTextures(1,&LumenReflectionTexture);
    LumenReflectionEstimate=LumenReflectionResolveProgram=LumenReflectionTexture=LumenReflectionFramebuffer=0;
    LumenReflectionEstimateBytes=0; LumenReflectionWidth=LumenReflectionHeight=0;
}
bool EnsureLumenReflectionResolve(std::size_t bytes)
{
    if (LumenReflectionEstimateBytes!=bytes)
    {
        GLuint buffer=0; GlGenBuffers(1,&buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,bytes,nullptr,GL_DYNAMIC_DRAW);
        if (!CheckGL("Allocate Lumen reflection estimate")) { GlDeleteBuffers(1,&buffer); return false; }
        if (LumenReflectionEstimate) GlDeleteBuffers(1,&LumenReflectionEstimate); LumenReflectionEstimate=buffer; LumenReflectionEstimateBytes=bytes;
    }
    if (LumenReflectionResolveProgram) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
    const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen reflection shader has no version directive");
    source.insert(version+1,"#define EVANS_LUMEN_REFLECTION_RESOLVE_STAGE 1\n");
    const auto program=CompileComputeProgram(source); if(!program)return false;
    LumenReflectionResolveProgram=program; return true;
}
bool ApplyLumenReflections(const ReflectionDesc& camera)
{
    if (!FrameLumenLighting || LumenReflectionRays) return Fail("Lumen reflections require one consumer of the current lighting cache");
    if (Techniques::UsesTemporalReflections(FrameReflections) && (!VelocityCaptured || (FrameSampleCount>1 && !SamplesVelocityResolved)))
        return Fail("Temporal Lumen reflections require this frame's resolved velocity before tracing");
    LumenRadianceTraceUpload query;
    if (!BuildLumenReflectionQuery(camera,FrameReflections,CreateInfo.Width,CreateInfo.Height,query,LastError)) return false;
    if (FrameGI.Lumen.GlobalDistance.Enabled)
    { query.GlobalDistance=FrameLumenGlobalDistance; query.Trace.GlobalNearDistance=FrameGI.Lumen.GlobalDistance.NearScale*FrameReflections.MaximumDistance; query.Trace.GlobalMaximumSteps=FrameGI.Lumen.GlobalDistance.MaximumSteps; }
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,query,rays)) return false; LumenReflectionRays=rays.Handle;
    const auto source=LumenRadianceTraces.find(rays.Handle.Value); if (source==LumenRadianceTraces.end()) return Fail("Lumen reflection source disappeared");
    const auto& filter=FrameReflections.Lumen; const auto& sky=FrameGI.Lumen.SkyRadiance; LumenReflectionResolveFrame frame;
    if (!BuildLumenReflectionResolveFrame(rays.Info.Reflection,{filter.FilterRadius,filter.NormalCosine,filter.PlaneTolerance,filter.RelativePlaneTolerance,filter.RoughnessTolerance},
        {sky[0],sky[1],sky[2]},FrameReflectionShading,frame,LastError,MakeLumenEnvironmentLobe(FrameGI.Lumen))) return false;
    LumenCaptureState state(*this); LumenRadianceStorageState storage(*this); LumenCaptureBuffers constants(*this);
    if (!EnsureLumenReflectionResolve(std::size_t(frame.Dispatch.Z)*sizeof(Float4))) return false;
    GlGenBuffers(1,constants.Buffers.data()); GlBindBuffer(GL_UNIFORM_BUFFER,constants.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW);
    if (!CheckGL("Allocate Lumen reflection constants")) return false;
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,constants.Buffers[0]); GlBindFramebuffer(GL_FRAMEBUFFER,0); GlUseProgram(LumenReflectionResolveProgram);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,source->second->Origins); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,source->second->Buffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,LumenReflectionEstimate);
    const GLuint images[]{DepthTexture,GBufferTextures[1],GBufferTextures[0],GBufferTextures[3],GBufferTextures[2]};
    const char* names[]{"ReconstructionDepth","ReconstructionNormal","ReconstructionDiffuse","ReconstructionMetal","ReconstructionSpecular"};
    for (unsigned slot=0;slot<std::size(images);++slot)
    { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,images[slot]); LumenBindSampler(slot,0); GlUniform1i(GlGetUniformLocation(LumenReflectionResolveProgram,names[slot]),slot); }
    GpuMemoryBarrier(GpuStorageBarrier); const auto groups=(frame.Dispatch.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    GpuDispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1); GpuMemoryBarrier(GpuStorageBarrier);
    if (!CheckGL("Reconstruct Lumen reflection estimate")) return false;
    PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::Trace); PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::LumenReconstruction);
    PendingFrame.Statistics.ReflectionPassCount+=2;
    if (!ResolveLumenSpecular(camera)) return false; ReflectionResolved=true; return true;
}
