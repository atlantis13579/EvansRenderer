LumenLightingHandle FrameLumenLighting;
LumenWorldCacheBinding FrameLumenWorldCache;
LumenDistanceSceneHandle FrameLumenGlobalDistance;
LumenRadianceHandle LumenGIRays;
LumenProbeHandle LumenGIProbes;
bool LumenGatherIndirect=false;
bool LumenGatherSupported=false;
GLuint LumenGatherProgram=0,LumenGatherTexture=0,LumenGatherFramebuffer=0;
GLuint LumenVisibilityProgram=0,LumenVisibilityBuffer=0;
int LumenGatherWidth=0,LumenGatherHeight=0;
std::size_t LumenVisibilityBytes=0;
void DestroyLumenGather()
{
    LumenVisibilityBytes=0;
    ReleaseLumenGIResults(); DestroyLumenTemporal(); DestroyLumenReflections(); DestroyLumenTranslucency(); DestroyLumenHairReceiver(); LumenGatherIndirect=false; FrameLumenLighting={}; FrameLumenWorldCache={}; FrameLumenGlobalDistance={};
    if (LumenVisibilityProgram) GlDeleteProgram(LumenVisibilityProgram);
    if (LumenVisibilityBuffer) GlDeleteBuffers(1,&LumenVisibilityBuffer);
    LumenVisibilityProgram=LumenVisibilityBuffer=0;
    if (LumenGatherProgram) GlDeleteProgram(LumenGatherProgram);
    if (LumenGatherFramebuffer) GlDeleteFramebuffers(1,&LumenGatherFramebuffer);
    if (LumenGatherTexture) GlDeleteTextures(1,&LumenGatherTexture);
    LumenGatherProgram=LumenGatherFramebuffer=LumenGatherTexture=0; LumenGatherWidth=LumenGatherHeight=0;
}
bool EnsureLumenGather(const LumenProbeGatherFrame& frame)
{
    constexpr GLenum MaxFragmentStorageBlocks=0x90DA; // GLES 3.1 GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS.
    GLint capacity=0; GlGetIntegerv(MaxFragmentStorageBlocks,&capacity);
    if (capacity<3) return Fail("Lumen gather needs three fragment shader storage blocks");
    if (!LumenGatherProgram) LumenGatherProgram=CompileStageProgram("Lumen.glsl","#define EVANS_LUMEN_PROBE_GATHER_STAGE 1\n");
    if (!LumenGatherProgram) return false;
    if (!LumenVisibilityProgram)
    {
        std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
        const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen visibility shader has no version directive");
        source.insert(version+1,"#define EVANS_LUMEN_GATHER_VISIBILITY_STAGE 1\n");
        const auto program=CompileComputeProgram(source); if(!program)return false;
        LumenVisibilityProgram=program;
    }
    if (LumenGatherTexture && LumenGatherWidth==CreateInfo.Width && LumenGatherHeight==CreateInfo.Height && LumenVisibilityBytes==LumenProbeVisibilityBytes(frame)) return true;
    GLuint texture=0,framebuffer=0; GlGenTextures(1,&texture); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,texture); ConfigureDepthTexture();
    GlBindBuffer(TSRPixelUnpackBuffer,0); GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
    GlGenFramebuffers(1,&framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0); const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment);
    const bool complete=GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    if (!CheckGL("Create Lumen gather target") || !complete)
    { GlDeleteFramebuffers(1,&framebuffer); GlDeleteTextures(1,&texture); return Fail("Lumen gather target allocation failed"); }
    GLuint mask=0; GlGenBuffers(1,&mask); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,mask);
    GlBufferData(GL_SHADER_STORAGE_BUFFER,LumenProbeVisibilityBytes(frame),nullptr,GL_DYNAMIC_DRAW);
    if (!CheckGL("Allocate Lumen probe visibility")) { GlDeleteBuffers(1,&mask); GlDeleteFramebuffers(1,&framebuffer); GlDeleteTextures(1,&texture); return false; }
    if (LumenVisibilityBuffer) GlDeleteBuffers(1,&LumenVisibilityBuffer); LumenVisibilityBuffer=mask; LumenVisibilityBytes=LumenProbeVisibilityBytes(frame);
    if (LumenGatherFramebuffer) GlDeleteFramebuffers(1,&LumenGatherFramebuffer);
    if (LumenGatherTexture) GlDeleteTextures(1,&LumenGatherTexture);
    LumenGatherTexture=texture; LumenGatherFramebuffer=framebuffer; LumenGatherWidth=CreateInfo.Width; LumenGatherHeight=CreateInfo.Height; return true;
}
bool ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if ((GIActive && (!Techniques::UsesLumen(FrameGI) || GI.Resolved)) || !HdrActive || !GBufferWritten || !(DeferredActive?DeferredResolved:ForwardColorWritten) ||
        (FrameSampleCount>1 && (!SamplesOpaqueResolved || SamplesColorRestored || SamplesTransparentWritten)) ||
        (AmbientOcclusionActive && !AmbientOcclusionResolved) || ReflectionResolved || FogResolved || LightShaftResolved || TemporalResolved ||
        MotionBlurResolved || DepthOfFieldResolved || RefractionResolved || !Hair.Complete() || !SubsurfaceComplete())
        return Fail("Lumen gather requires resolved opaque HDR materials before reflections, volumes and transparency");
    if (LumenHasTransparentMainDraws(PendingFrame)) return Fail("Lumen gather must precede transparent draws");
    const auto found=LumenProbes.find(handle.Value); if (found==LumenProbes.end()) return Fail("Unknown Lumen gather probes");
    const auto& record=*found->second; if (!FrameLumen.CanGather(handle,record.SourceHandle,LastError)) return false;
    const auto source=LumenRadianceTraces.find(record.SourceHandle.Value); if (source==LumenRadianceTraces.end()) return Fail("Lumen gather source was released");
    LumenProbeGatherFrame frame; if (!BuildLumenProbeGatherFrame(record.Info,source->second->Lighting->Distance->Info,settings,FrameReflectionShading,frame,LastError)) return false;
    frame.Model.Z=LumenGatherIndirect?1u:0u;
    LumenCaptureState state(*this); LumenRadianceStorageState storageState(*this); LumenCaptureBuffers work(*this);
    if (!EnsureLumenGather(frame)) return false;
    GlGenBuffers(1,work.Buffers.data()); GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_STATIC_DRAW);
    if (!CheckGL("Allocate Lumen gather constants")) return false;
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[0]);
    if (settings.Visibility.Enabled)
    {
        GlBindFramebuffer(GL_FRAMEBUFFER,0); GlUseProgram(LumenVisibilityProgram);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,source->second->Lighting->Distance->Buffer);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,source->second->Origins); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,LumenVisibilityBuffer);
        const GLuint textures[]{DepthTexture,GBufferTextures[1]}; const char* names[]{"LinkDepth","LinkNormals"};
        for (unsigned slot=0;slot<std::size(textures);++slot)
        { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,textures[slot]); LumenBindSampler(slot,0); GlUniform1i(GlGetUniformLocation(LumenVisibilityProgram,names[slot]),slot); }
        GpuMemoryBarrier(GpuStorageBarrier);
        const auto groups=(frame.Trace.Counts.X+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
        GpuDispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
        GpuMemoryBarrier(GpuStorageBarrier);
    }
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,LumenVisibilityBuffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,source->second->Origins); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,record.Buffer);
    for (auto capability:LumenCaptureState::Switches) GlDisable(capability);
    GlDepthMask(GL_FALSE_VALUE); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlBindVertexArray(VertexArray);
    GlUseProgram(LumenGatherProgram); GlBindFramebuffer(GL_FRAMEBUFFER,LumenGatherFramebuffer); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height);
    const GLuint inputs[]{HdrTextures[0],DepthTexture,GBufferTextures[1],GBufferTextures[0],GBufferTextures[3]};
    const char* names[]{"GatherScene","GatherDepth","GatherNormal","GatherDiffuse","GatherAmbient"};
    for (unsigned slot=0;slot<std::size(inputs);++slot)
    { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,inputs[slot]); LumenBindSampler(slot,0); GlUniform1i(GlGetUniformLocation(LumenGatherProgram,names[slot]),slot); }
    GpuMemoryBarrier(GpuStorageBarrier); GlDrawArrays(GL_TRIANGLES,0,3);
    for (unsigned slot=0;slot<std::size(inputs);++slot) { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,0); }
    if (!LumenGatherIndirect)
    {
        GlReadBuffer(GL_COLOR_ATTACHMENT0); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]);
        GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,CreateInfo.Width,CreateInfo.Height);
    }
    if (!CheckGL("Gather Lumen screen irradiance")) return false;
    FrameLumen.Gather(); ++PendingFrame.Statistics.GlobalIlluminationPassCount; return true;
}

// The ordinary GI pass owns these transient results; diagnostic APIs keep their
// explicit handles. FinishLumenFrame releases only this automatically owned pair.
bool ApplyLumenGI(const GlobalIlluminationDesc& desc)
{
    if (!FrameLumenLighting || LumenGIRays || LumenGIProbes) return Fail("Lumen requires a current surface cache and one frame consumer");
    const auto& settings=FrameGI.Lumen; LumenRadianceTraceUpload trace;
    trace.Screen.Enabled=true; trace.Screen.ViewProjection=desc.ViewProjection; trace.Screen.Width=CreateInfo.Width; trace.Screen.Height=CreateInfo.Height;
    trace.Screen.TileSide=settings.TileSide; trace.Screen.NormalBias=FrameGI.NormalBias;
    trace.Screen.Adaptive=settings.AdaptiveProbes; trace.Screen.AdaptiveLevels=settings.AdaptiveProbeLevels; trace.Screen.MinimumNormalCosine=FrameGI.NormalThreshold; trace.Screen.PlaneTolerance=FrameGI.DepthTolerance;
    trace.Probes.Sequence=settings.Temporal.Enabled?desc.Sampling.SampleIndex:0; trace.Probes.RaysPerProbe=settings.RaysPerProbe; trace.Probes.MaximumDistance=FrameGI.MaximumDistance;
    if (settings.GlobalDistance.Enabled)
    { trace.GlobalDistance=FrameLumenGlobalDistance; trace.Trace.GlobalNearDistance=settings.GlobalDistance.NearScale*FrameGI.MaximumDistance; trace.Trace.GlobalMaximumSteps=settings.GlobalDistance.MaximumSteps; }
    trace.Trace.MaximumSteps=FrameGI.MaximumSteps; trace.Hybrid=settings.ScreenTrace; if (settings.WorldCache.Enabled) trace.WorldCache=FrameLumenWorldCache;
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,trace,rays)) return false; LumenGIRays=rays.Handle;
    LumenProbeIntegrationResult probes; const auto& sky=settings.SkyRadiance;
    if (!IntegrateLumenProbes(rays.Handle,{{sky[0],sky[1],sky[2]},false,MakeLumenEnvironmentLobe(settings)},probes)) return false; LumenGIProbes=probes.Handle;
    LumenProbeGatherSettings gather; gather.Intensity=FrameGI.Intensity; gather.MinimumNormalCosine=FrameGI.NormalThreshold; gather.PlaneTolerance=FrameGI.DepthTolerance; gather.Visibility=settings.ProbeVisibility;
    LumenGatherIndirect=settings.Temporal.Enabled;
    const bool gathered=ResolveLumenProbes(probes.Handle,gather); LumenGatherIndirect=false;
    if (!gathered || (settings.Temporal.Enabled && !ResolveLumenTemporal(desc))) return false;
    if (!PrepareLumenTranslucency(desc) || !ResolveLumenHairReceiver()) return false;
    GI.Resolved=true; return true;
}
void ReleaseLumenGIResults()
{
    ReleaseLumenTranslucencyResults();
    if (LumenGIProbes) LumenProbes.erase(LumenGIProbes.Value);
    if (LumenGIRays) LumenRadianceTraces.erase(LumenGIRays.Value);
    LumenGIProbes={}; LumenGIRays={};
}
