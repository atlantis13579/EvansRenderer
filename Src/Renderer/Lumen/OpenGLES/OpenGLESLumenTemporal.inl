LumenTemporalHistory LumenHistory;
std::array<GLuint,2> LumenTemporalBuffers{};
GLuint LumenTemporalCompute=0,LumenTemporalComposite=0;
std::size_t LumenTemporalBytes=0;
void DestroyLumenTemporal()
{
    if (LumenTemporalBuffers[0] || LumenTemporalBuffers[1]) GlDeleteBuffers(GLsizei(LumenTemporalBuffers.size()),LumenTemporalBuffers.data()); LumenTemporalBuffers={}; LumenTemporalBytes=0;
    if (LumenTemporalCompute) GlDeleteProgram(LumenTemporalCompute); if (LumenTemporalComposite) GlDeleteProgram(LumenTemporalComposite);
    LumenTemporalCompute=LumenTemporalComposite=0; LumenHistory.Invalidate();
}
bool EnsureLumenTemporal(std::size_t bytes)
{
    if (LumenTemporalBytes!=bytes)
    {
        std::array<GLuint,2> buffers{}; GlGenBuffers(GLsizei(buffers.size()),buffers.data());
        for (auto buffer:buffers) { GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,bytes,nullptr,GL_DYNAMIC_DRAW); }
        if (!CheckGL("Allocate Lumen temporal history")) { GlDeleteBuffers(GLsizei(buffers.size()),buffers.data()); return false; }
        GlDeleteBuffers(GLsizei(LumenTemporalBuffers.size()),LumenTemporalBuffers.data()); LumenTemporalBuffers=buffers; LumenTemporalBytes=bytes; LumenHistory.Invalidate();
    }
    if (!LumenTemporalComposite) LumenTemporalComposite=CompileStageProgram("Lumen.glsl","#define EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE 1\n");
    if (!LumenTemporalComposite) return false;
    if (!LumenTemporalCompute)
    {
        std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
        const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen temporal shader has no version directive");
        source.insert(version+1,"#define EVANS_LUMEN_TEMPORAL_STAGE 1\n"); const auto program=CompileComputeProgram(source); if(!program)return false;
        LumenTemporalCompute=program;
    }
    return true;
}
bool ResolveLumenTemporal(const GlobalIlluminationDesc& camera)
{
    const LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    LumenTemporalFrame frame;
    if (!BuildLumenTemporalFrame(camera,FrameGI,CreateInfo.Width,CreateInfo.Height,LumenHistory.CanReuse(key),frame,LastError)) return false;
    LumenCaptureState state(*this); LumenRadianceStorageState storage(*this); LumenCaptureBuffers constants(*this);
    if (!EnsureLumenTemporal(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    GlGenBuffers(1,constants.Buffers.data()); GlBindBuffer(GL_UNIFORM_BUFFER,constants.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW);
    if (!CheckGL("Allocate Lumen temporal constants")) return false;
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,constants.Buffers[0]); const auto read=LumenHistory.ReadIndex(),write=LumenHistory.WriteIndex();
    GlBindFramebuffer(GL_FRAMEBUFFER,0); GlUseProgram(LumenTemporalCompute);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,LumenTemporalBuffers[read]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,LumenTemporalBuffers[write]);
    const GLuint inputs[]{LumenGatherTexture,DepthTexture,GBufferTextures[1],MotionTextures[MotionVelocity]};
    const char* names[]{"TemporalCurrent","TemporalDepth","TemporalNormals","TemporalMotion"};
    for (unsigned slot=0;slot<std::size(inputs);++slot)
    { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,inputs[slot]); LumenBindSampler(slot,0); GlUniform1i(GlGetUniformLocation(LumenTemporalCompute,names[slot]),slot); }
    GpuMemoryBarrier(GpuStorageBarrier); const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    GpuDispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1); GpuMemoryBarrier(GpuStorageBarrier);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,0); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,LumenTemporalBuffers[write]);
    GlUseProgram(LumenTemporalComposite); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]); GlUniform1i(GlGetUniformLocation(LumenTemporalComposite,"TemporalCurrent"),0);
    for (auto capability:LumenCaptureState::Switches) GlDisable(capability);
    GlDepthMask(GL_FALSE_VALUE); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlBindVertexArray(VertexArray);
    GlBindFramebuffer(GL_FRAMEBUFFER,LumenGatherFramebuffer); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlDrawArrays(GL_TRIANGLES,0,3);
    for (unsigned slot=0;slot<std::size(inputs);++slot) { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,0); }
    GlReadBuffer(GL_COLOR_ATTACHMENT0); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]);
    GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,CreateInfo.Width,CreateInfo.Height);
    if (!CheckGL("Resolve Lumen temporal diffuse")) return false;
    LumenHistory.Record(key); PendingFrame.Temporal=camera.Sampling; PendingFrame.Statistics.GlobalIlluminationPassCount+=2; return true;
}
