LumenTranslucencyFrame LumenVolumeFrame;
LumenRadianceHandle LumenVolumeRays;
LumenProbeHandle LumenVolumeProbes;
void ReleaseLumenTranslucencyResults()
{
    if (LumenVolumeProbes) LumenProbes.erase(LumenVolumeProbes.Value);
    if (LumenVolumeRays) LumenRadianceTraces.erase(LumenVolumeRays.Value);
    LumenVolumeProbes={}; LumenVolumeRays={}; LumenVolumeFrame.Grid.W=0;
}

GLuint LumenVolumeAtlas=0,LumenVolumeFramebuffer=0,LumenVolumeProgram=0;
GLuint LumenVolumeSupportProgram=0,LumenVolumeSupport=0;
unsigned LumenVolumeWidth=0,LumenVolumeHeight=0;
void DestroyLumenTranslucency()
{
    ReleaseLumenTranslucencyResults();
    if (LumenVolumeProgram) GlDeleteProgram(LumenVolumeProgram);
    if (LumenVolumeFramebuffer) GlDeleteFramebuffers(1,&LumenVolumeFramebuffer);
    if (LumenVolumeAtlas) GlDeleteTextures(1,&LumenVolumeAtlas);
    if (LumenVolumeSupportProgram) GlDeleteProgram(LumenVolumeSupportProgram);
    if (LumenVolumeSupport) GlDeleteBuffers(1,&LumenVolumeSupport);
    LumenVolumeProgram=LumenVolumeFramebuffer=LumenVolumeAtlas=0; LumenVolumeWidth=LumenVolumeHeight=0;
    LumenVolumeSupportProgram=LumenVolumeSupport=0;
}
bool PrepareLumenVolumeSupport(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    const auto source=LumenRadianceTraces.find(probes.SourceHandle.Value);
    if (source==LumenRadianceTraces.end() || !source->second->TraceDistance) return Fail("Lumen volume visibility lost its frozen distance scene");
    LumenDistanceTraceFrame frame;
    if (!BuildLumenTranslucencySupportFrame(source->second->TraceDistance->Info,upload,frame,LastError) || !EnsureLumenDistanceTrace()) return false;
    if (!LumenVolumeSupportProgram)
    {
        std::string text; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",text,LastError)) return false;
        const auto version=text.find('\n'); if (version==std::string::npos) return Fail("Lumen volume shader has no version directive");
        text.insert(version+1,"#define EVANS_LUMEN_VOLUME_VISIBILITY_STAGE 1\n"); const auto program=CompileComputeProgram(text); if(!program)return false;
        LumenVolumeSupportProgram=program;
    }
    LumenCaptureBuffers work(*this); GlGenBuffers(1,work.Buffers.data()); GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW);
    if (!LumenVolumeSupport) GlGenBuffers(1,&LumenVolumeSupport);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,LumenVolumeSupport); GlBufferData(GL_SHADER_STORAGE_BUFFER,std::size_t(frame.Counts.X)*sizeof(float),nullptr,GL_STATIC_DRAW);
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[0]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,source->second->TraceDistance->Buffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,source->second->Origins); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,LumenVolumeSupport);
    GlUseProgram(LumenVolumeSupportProgram); GpuDispatch((frame.Counts.X+LumenRadianceThreads-1)/LumenRadianceThreads,1,1); GpuMemoryBarrier(GpuStorageBarrier);
    return CheckGL("Compute Lumen volume visibility bounds");
}
bool PackLumenTranslucency(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    LumenCaptureState state(*this); LumenRadianceStorageState storage(*this); LumenCaptureBuffers work(*this);
    if (!PrepareLumenVolumeSupport(probes,upload)) return false;
    if (!LumenVolumeProgram) LumenVolumeProgram=CompileStageProgram("LumenTranslucency.glsl","");
    if (!LumenVolumeProgram) return false;
    const unsigned width=unsigned(LumenVolumeFrame.Atlas.X),height=unsigned(LumenVolumeFrame.Atlas.Y);
    if (!LumenVolumeAtlas || LumenVolumeWidth!=width || LumenVolumeHeight!=height)
    {
        GLuint texture=0,framebuffer=0; GlGenTextures(1,&texture); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,texture); ConfigureDepthTexture();
        GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        GlBindBuffer(TSRPixelUnpackBuffer,0); GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,width,height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
        GlGenFramebuffers(1,&framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0); const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment);
        const bool complete=GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
        if (!CheckGL("Create Lumen translucent volume") || !complete)
        { GlDeleteFramebuffers(1,&framebuffer); GlDeleteTextures(1,&texture); return Fail("Lumen translucent volume allocation failed"); }
        if (LumenVolumeFramebuffer) GlDeleteFramebuffers(1,&LumenVolumeFramebuffer);
        if (LumenVolumeAtlas) GlDeleteTextures(1,&LumenVolumeAtlas);
        LumenVolumeAtlas=texture; LumenVolumeFramebuffer=framebuffer; LumenVolumeWidth=width; LumenVolumeHeight=height;
    }
    GlGenBuffers(1,work.Buffers.data()); GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(LumenVolumeFrame),&LumenVolumeFrame,GL_STATIC_DRAW);
    if (!CheckGL("Create Lumen translucent constants")) return false;
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[0]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,probes.Buffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,LumenVolumeSupport);
    for (auto capability:LumenCaptureState::Switches) GlDisable(capability);
    GlDepthMask(GL_FALSE_VALUE); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
    GlBindVertexArray(VertexArray); GlUseProgram(LumenVolumeProgram); GlBindFramebuffer(GL_FRAMEBUFFER,LumenVolumeFramebuffer); GlViewport(0,0,width,height);
    GpuMemoryBarrier(GpuStorageBarrier); GlDrawArrays(GL_TRIANGLES,0,3); return CheckGL("Pack Lumen translucent irradiance");
}
bool PrepareLumenTranslucency(const GlobalIlluminationDesc& desc)
{
    const auto& settings=FrameGI.Lumen;
    if ((!desc.Translucency || !settings.Translucency.Enabled) && !Hair.LumenReceiver) return true;
    LumenRadianceTraceUpload trace;
    if (!BuildLumenTranslucencyFrame(desc.ViewProjection,CreateInfo.Width,CreateInfo.Height,FrameGI,LumenVolumeFrame,trace,LastError)) return false;
    if (settings.GlobalDistance.Enabled)
    { trace.GlobalDistance=FrameLumenGlobalDistance; trace.Trace.GlobalNearDistance=settings.GlobalDistance.NearScale*FrameGI.MaximumDistance; trace.Trace.GlobalMaximumSteps=settings.GlobalDistance.MaximumSteps; }
    if (settings.WorldCache.Enabled) trace.WorldCache=FrameLumenWorldCache;
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,trace,rays)) return false; LumenVolumeRays=rays.Handle;
    const auto& sky=settings.SkyRadiance; LumenProbeIntegrationResult probes;
    if (!IntegrateLumenProbes(rays.Handle,{{sky[0],sky[1],sky[2]},false,MakeLumenEnvironmentLobe(settings)},probes)) return false; LumenVolumeProbes=probes.Handle;
    if (!PackLumenTranslucency(*LumenProbes.at(probes.Handle.Value),trace)) return false;
    ++PendingFrame.Statistics.GlobalIlluminationPassCount; PendingFrame.Statistics.LumenTranslucencyProbeCount=unsigned(trace.Probes.Origins.size()); return true;
}

static constexpr GLuint LumenVolumeTextureUnit=9;
void BindLumenTranslucency(GLuint program,bool active)
{
    const auto& frame=LumenVolumeFrame; const bool enabled=active && FrameGI.Lumen.Translucency.Enabled && frame.Grid.W!=0;
    const auto location=[&](const char* name) { return GlGetUniformLocation(program,name); };
    GlUniform4ui(location("LumenVolumeCells"),frame.Grid.X,frame.Grid.Y,frame.Grid.Z,enabled?frame.Grid.W:0);
    GlUniform1i(location("LumenVolumeSH"),LumenVolumeTextureUnit);
    if (enabled)
    {
        const auto camera=ToColumnMajor(frame.ViewProjection),inverse=ToColumnMajor(frame.Inverse);
        GlUniformMatrix4fv(location("LumenVolumeCamera"),1,GL_FALSE_VALUE,camera.data());
        GlUniformMatrix4fv(location("LumenVolumeInverse"),1,GL_FALSE_VALUE,inverse.data());
        UploadFloat4(location("LumenVolumeDepth"),frame.Distribution); UploadFloat4(location("LumenVolumeExtent"),frame.Atlas);
    }
    GlActiveTexture(GL_TEXTURE0+LumenVolumeTextureUnit); GlBindTexture(GL_TEXTURE_2D,enabled?LumenVolumeAtlas:0);
    if (LumenBindSampler) LumenBindSampler(LumenVolumeTextureUnit,0);
}
