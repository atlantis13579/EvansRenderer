struct LumenRadianceRecord
{
    std::shared_ptr<LumenDistanceRecord> TraceDistance;
    std::shared_ptr<LumenRadianceRecord> WorldCache;
    Impl& Owner; LumenRadianceInfo Info; GLuint Buffer=0,Origins=0; std::shared_ptr<LumenLightingRecord> Lighting;
    explicit LumenRadianceRecord(Impl& owner):Owner(owner) {}
    ~LumenRadianceRecord() { if (Buffer) Owner.GlDeleteBuffers(1,&Buffer); if (Origins) Owner.GlDeleteBuffers(1,&Origins); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenRadianceRecord>> LumenRadianceTraces;
LumenFrameTransaction FrameLumen;
GLuint LumenRadianceTrace=0,LumenRadianceResolve=0,LumenProbeTrace=0,LumenScreenPlacement=0,LumenHybridTrace=0,LumenScreenDepth=0,LumenReflectionRayShader=0;
GLuint LumenWorldLookup=0;
void(EVANS_GL_CALL* LumenCopyRadiance)(GLenum,GLenum,GLintptr,GLintptr,GLsizeiptr)=nullptr;
bool EnsureLumenWorldLookup()
{
    if (LumenWorldLookup) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","LumenWorldCache.glsl",source,LastError)) return false;
    const auto program=CompileComputeProgram(source); if(!program)return false;
    LumenWorldLookup=program; return true;
}
void DestroyLumenRadianceTraces()
{
    if (LumenWorldLookup) GlDeleteProgram(LumenWorldLookup); LumenWorldLookup=0;
    LumenRadianceTraces.clear(); if (LumenRadianceTrace) GlDeleteProgram(LumenRadianceTrace); if (LumenRadianceResolve) GlDeleteProgram(LumenRadianceResolve);
    if (LumenProbeTrace) GlDeleteProgram(LumenProbeTrace); if (LumenScreenPlacement) GlDeleteProgram(LumenScreenPlacement); if (LumenHybridTrace) GlDeleteProgram(LumenHybridTrace); if (LumenScreenDepth) GlDeleteProgram(LumenScreenDepth); if (LumenReflectionRayShader) GlDeleteProgram(LumenReflectionRayShader); LumenRadianceTrace=LumenRadianceResolve=LumenProbeTrace=LumenScreenPlacement=LumenHybridTrace=LumenScreenDepth=LumenReflectionRayShader=0;
}
bool EnsureLumenRadianceTrace()
{
    if (!EnsureLumenSurface() || !EnsureLumenDistanceTrace()) return false;
    GLint bindings=0; GlGetIntegerv(DistanceMaxComputeStorage,&bindings); if (bindings<4) return Fail("Lumen radiance resolve needs four compute storage bindings");
    if (LumenRadianceTrace && LumenRadianceResolve && LumenProbeTrace && LumenScreenPlacement && LumenHybridTrace && LumenScreenDepth && LumenReflectionRayShader) return true;
    const char* stages[]{"EVANS_LUMEN_RADIANCE_TRACE_STAGE","EVANS_LUMEN_RADIANCE_RESOLVE_STAGE","EVANS_LUMEN_PROBE_TRACE_STAGE","EVANS_LUMEN_SCREEN_PROBES_STAGE","EVANS_LUMEN_SCREEN_TRACE_STAGE","EVANS_LUMEN_SCREEN_DEPTH_STAGE","EVANS_LUMEN_REFLECTION_RAYS_STAGE"}; GLuint* targets[]{&LumenRadianceTrace,&LumenRadianceResolve,&LumenProbeTrace,&LumenScreenPlacement,&LumenHybridTrace,&LumenScreenDepth,&LumenReflectionRayShader};
    for (unsigned stage=0;stage<std::size(stages);++stage)
    {
        if (*targets[stage]) continue;
        std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
        const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen radiance shader has no version directive");
        source.insert(version+1,std::string("#define ")+stages[stage]+" 1\n"); const auto program=CompileComputeProgram(source); if(!program)return false;
        *targets[stage]=program;
    }
    return true;
}
struct LumenRadianceStorageState
{
    Impl& Owner; std::array<GLint,3> Buffers{}; std::array<std::int64_t,3> Starts{},Sizes{};
    explicit LumenRadianceStorageState(Impl& owner):Owner(owner)
    {
        for (unsigned i=0;i<Buffers.size();++i)
        { Owner.GpuGetIntegerIndexed(GpuStorageBinding,i+1,&Buffers[i]); Owner.LumenGetRange(LumenStorageStart,i+1,&Starts[i]); Owner.LumenGetRange(LumenStorageSize,i+1,&Sizes[i]); }
    }
    ~LumenRadianceStorageState()
    {
        for (unsigned i=0;i<Buffers.size();++i)
            if (Buffers[i] && Sizes[i]) Owner.LumenBindRange(GL_SHADER_STORAGE_BUFFER,i+1,Buffers[i],GLintptr(Starts[i]),GLsizeiptr(Sizes[i]));
            else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,i+1,Buffers[i]);
    }
};
bool ReadLumenOriginRecord(const LumenRadianceRecord& record,std::vector<LumenProbeOrigin>& output)
{
    if (!LumenRadianceProbeCount(record.Info)) return Fail("This Lumen trace has no probe-origin buffer");
    struct Binding
    {
        Impl& Owner; GLint Previous=0;
        explicit Binding(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(GpuStorageBinding,&Previous); }
        ~Binding() { Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,Previous); }
    } binding(*this);
    const auto bytes=std::size_t(LumenRadianceProbeCount(record.Info))*sizeof(LumenProbeOrigin); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record.Origins);
    std::vector<LumenProbeOrigin> samples(LumenRadianceProbeCount(record.Info)); const auto* mapped=LumenMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),LumenMapRead);
    if (!mapped) return Fail("Map Lumen radiance readback failed");
    std::memcpy(samples.data(),mapped,bytes); const auto valid=LumenUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (!valid || !CheckGL("Read Lumen ray radiance")) return Fail("Lumen radiance readback became invalid");
    if (!ValidateLumenRadianceOrigins(record.Info,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool ReadLumenRadianceRecord(const LumenRadianceRecord& record,std::vector<LumenRadianceSample>& output)
{
    struct Binding
    {
        Impl& Owner; GLint Previous=0;
        explicit Binding(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(GpuStorageBinding,&Previous); }
        ~Binding() { Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,Previous); }
    } binding(*this);
    const auto bytes=std::size_t(record.Info.RayCount)*sizeof(LumenRadianceSample); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record.Buffer);
    std::vector<LumenRadianceSample> samples(record.Info.RayCount); const auto* mapped=LumenMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),LumenMapRead);
    if (!mapped) return Fail("Map Lumen radiance readback failed");
    std::memcpy(samples.data(),mapped,bytes); const auto valid=LumenUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (!valid || !CheckGL("Read Lumen ray radiance")) return Fail("Lumen radiance readback became invalid");
    if (!ValidateLumenRadianceSamples(record.Info,record.Lighting->Distance->Info,record.Lighting->CardInstances,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool TraceLumenRadiance(LumenLightingHandle lighting,const LumenRadianceTraceUpload& upload,LumenRadianceTraceResult& output)
{
    if (upload.Hybrid.Enabled && (!FrameOpen || (!upload.Reflections.Enabled && (GI.Resolved || FrameLumen.HasGathered())) || !HdrActive || !GBufferWritten || !(DeferredActive?DeferredResolved:ForwardColorWritten) ||
        (FrameSampleCount>1 && (!SamplesOpaqueResolved || SamplesColorRestored || SamplesTransparentWritten)) ||
        (AmbientOcclusionActive && !AmbientOcclusionResolved) || ReflectionResolved || FogResolved || LightShaftResolved || TemporalResolved ||
        MotionBlurResolved || DepthOfFieldResolved || RefractionResolved || !Hair.Complete() || !SubsurfaceComplete()))
        return Fail("Lumen hybrid tracing requires resolved opaque HDR before secondary composition");
    if (upload.Hybrid.Enabled && LumenHasTransparentMainDraws(PendingFrame)) return Fail("Lumen hybrid tracing must precede transparent draws");
    if (upload.Screen.Enabled && (!FrameOpen || !GBufferWritten || (FrameSampleCount>1 && !SamplesOpaqueResolved) ||
        upload.Screen.Width!=unsigned(CreateInfo.Width) || upload.Screen.Height!=unsigned(CreateInfo.Height)))
        return Fail("Lumen screen placement needs this frame's resolved G-buffer and matching viewport");
    const auto found=LumenLightingCaches.find(lighting.Value); if (found==LumenLightingCaches.end()) return Fail("Unknown Lumen radiance source lighting");
    const auto source=found->second; LumenRadianceInput input;
    auto distance=source->Distance;
    if (upload.GlobalDistance)
    {
        const auto global=LumenDistanceScenes.find(upload.GlobalDistance.Value);
        if (global==LumenDistanceScenes.end() || global->second->Base!=source->Distance) return Fail("Global distance scene has a different or expired mesh source");
        distance=global->second;
    }
    if (!BuildLumenRadianceInput(source->Info,distance->Info,source->Surface->Captures,source->CardInstances,upload,input,LastError) || (FrameOpen && !FrameLumen.CanTrace(input.Resolve.Counts.X,LastError)) || !EnsureLumenRadianceTrace()) return false;
    std::shared_ptr<LumenRadianceRecord> worldCache; LumenWorldLookupFrame worldFrame{};
    if (upload.WorldCache.Radiance)
    {
        const auto cache=LumenRadianceTraces.find(upload.WorldCache.Radiance.Value);
        if (cache==LumenRadianceTraces.end() || cache->second->Lighting!=source) return Fail("World radiance cache has a different or expired lighting source");
        worldCache=cache->second;
        if (!BuildLumenWorldLookupFrame(worldCache->Info,upload,input,worldFrame,LastError) || !EnsureLumenWorldLookup()) return false;
    }
    const auto sources=LumenRadianceSources(upload,input),traceFrame=LumenRadianceFrameData(input);
    const auto count=std::size_t(input.Resolve.Counts.X),bytes=count*sizeof(LumenRadianceSample),mapBytes=input.Words.size()*sizeof(std::uint32_t);
    std::int64_t maximum=0; DistanceGetInteger64(DistanceMaxStorageSize,&maximum);
    if (maximum<=0 || std::max({bytes,mapBytes,input.Depth.Bytes})>std::uint64_t(maximum)) return Fail("Lumen radiance exceeds ES storage size");
    const auto groups=unsigned((count+LumenRadianceThreads-1)/LumenRadianceThreads),x=std::min(groups,LumenRadianceColumns),y=(groups+LumenRadianceColumns-1)/LumenRadianceColumns;
    GLint maxX=0,maxY=0; GpuGetIntegerIndexed(DistanceMaxGroups,0,&maxX); GpuGetIntegerIndexed(DistanceMaxGroups,1,&maxY);
    if (x>unsigned(maxX) || y>unsigned(maxY)) return Fail("Lumen radiance exceeds ES dispatch limits");
    for (const auto& level:input.Depth.Levels)
    {
        const auto levelGroups=(level.Destination.X*level.Destination.Y+LumenRadianceThreads-1)/LumenRadianceThreads;
        if (std::min(levelGroups,LumenRadianceColumns)>unsigned(maxX) || (levelGroups+LumenRadianceColumns-1)/LumenRadianceColumns>unsigned(maxY)) return Fail("Lumen depth exceeds ES dispatch limits");
    }
    auto record=std::make_shared<LumenRadianceRecord>(*this); record->Info=MakeLumenRadianceInfo(source->Info.Identity,upload,input,worldCache?unsigned(worldFrame.Query.W):0); record->Lighting=source; record->WorldCache=worldCache; record->TraceDistance=distance;
    struct Temporary { Impl& Owner; std::array<GLuint,10> Buffers{}; ~Temporary() { Owner.GlDeleteBuffers(GLsizei(Buffers.size()),Buffers.data()); } } work{*this};
    LumenCaptureState state(*this); LumenRadianceStorageState storageState(*this); GlGenBuffers(GLsizei(work.Buffers.size()),work.Buffers.data());
    const void* frames[]{traceFrame.Data,&input.Resolve}; const std::size_t frameSizes[]{traceFrame.Bytes,sizeof(input.Resolve)};
    for (unsigned i=0;i<2;++i) { GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[i]); GlBufferData(GL_UNIFORM_BUFFER,GLsizeiptr(frameSizes[i]),frames[i],GL_STATIC_DRAW); }
    const void* data[]{sources.Data,input.Words.data(),nullptr}; const std::size_t sizes[]{sources.Bytes,mapBytes,count*sizeof(LumenDistanceHit)};
    for (unsigned i=0;i<3;++i) { GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work.Buffers[i+2]); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(sizes[i]),data[i],GL_STATIC_DRAW); }
    GlGenBuffers(1,&record->Buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record->Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),nullptr,GL_STATIC_DRAW);
    record->Origins=work.Buffers[2]; work.Buffers[2]=0;
    if (upload.Hybrid.Enabled)
    {
        // ANGLE's storage blocks cannot alias even when both GLSL declarations
        // are readonly. Linear traversal keeps its own inactive tree binding.
        const std::uint32_t empty=0; GlBindBuffer(GL_SHADER_STORAGE_BUFFER,work.Buffers[7]);
        GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(std::max(input.Depth.Bytes,sizeof(empty))),input.Depth.Bytes?nullptr:&empty,GL_DYNAMIC_DRAW);
    }
    if (worldCache)
    { GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[9]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(worldFrame),&worldFrame,GL_STATIC_DRAW); }
    if (!CheckGL("Allocate Lumen radiance chain")) return false;
    if (upload.Screen.Enabled)
    {
        const auto placement=LumenRadiancePlacementData(input); const auto program=upload.Reflections.Enabled?LumenReflectionRayShader:LumenScreenPlacement;
        GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[5]); GlBufferData(GL_UNIFORM_BUFFER,GLsizeiptr(placement.Bytes),placement.Data,GL_STATIC_DRAW);
        GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[5]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,record->Origins); GlUseProgram(program);
        GlActiveTexture(GL_TEXTURE0); LumenBindSampler(0,0); GlBindTexture(GL_TEXTURE_2D,DepthTexture);
        GlActiveTexture(GL_TEXTURE0+1); LumenBindSampler(1,0); GlBindTexture(GL_TEXTURE_2D,GBufferTextures[1]);
        GlUniform1i(GlGetUniformLocation(program,upload.Reflections.Enabled?"SpecularDepth":"PlacementDepth"),0); GlUniform1i(GlGetUniformLocation(program,upload.Reflections.Enabled?"SpecularNormal":"PlacementNormal"),1);
        GpuDispatch((input.Screen.Grid.Y+LumenRadianceThreads-1)/LumenRadianceThreads,1,1); GpuMemoryBarrier(GpuStorageBarrier);
        if (!CheckGL("Place Lumen screen probes")) return false;
    }
    if (upload.Hybrid.Enabled)
    {
        GlBindFramebuffer(GL_FRAMEBUFFER,0);
        if (input.Depth.Bytes)
        {
            GlUseProgram(LumenScreenDepth); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,work.Buffers[7]);
            GlActiveTexture(GL_TEXTURE0); LumenBindSampler(0,0); GlBindTexture(GL_TEXTURE_2D,DepthTexture);
            GlUniform1i(GlGetUniformLocation(LumenScreenDepth,"PyramidDepth"),0);
            for (const auto& level:input.Depth.Levels)
            {
                GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[8]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(level),&level,GL_DYNAMIC_DRAW);
                GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[8]);
                const auto levelGroups=(level.Destination.X*level.Destination.Y+LumenRadianceThreads-1)/LumenRadianceThreads;
                GpuDispatch(std::min(levelGroups,LumenRadianceColumns),(levelGroups+LumenRadianceColumns-1)/LumenRadianceColumns,1); GpuMemoryBarrier(GpuStorageBarrier);
            }
            if (!CheckGL("Build Lumen screen depth hierarchy")) return false;
        }
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,work.Buffers[7]);
        GlBindBuffer(GL_UNIFORM_BUFFER,work.Buffers[6]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(input.Hybrid),&input.Hybrid,GL_STATIC_DRAW);
        GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[6]); GlUseProgram(LumenHybridTrace);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,record->Origins); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,work.Buffers[4]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,record->Buffer);
        const GLuint textures[]{DepthTexture,GBufferTextures[1],HdrTextures[0]}; const char* names[]{"HybridDepth","HybridNormals","HybridColor"};
        for (unsigned slot=0;slot<std::size(textures);++slot)
        { GlActiveTexture(GL_TEXTURE0+slot); LumenBindSampler(slot,0); GlBindTexture(GL_TEXTURE_2D,textures[slot]); GlUniform1i(GlGetUniformLocation(LumenHybridTrace,names[slot]),slot); }
        GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier);
        if (!CheckGL("Trace Lumen screen radiance")) return false;
    }
    if (worldCache)
    {
        GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[9]); GlUseProgram(LumenWorldLookup);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,worldCache->Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,record->Origins);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,work.Buffers[4]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,record->Buffer);
        GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier); if (!CheckGL("Reconstruct Lumen world radiance")) return false;
    }
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[0]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,distance->Buffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,record->Origins); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,work.Buffers[4]); GlUseProgram(input.ProbeTrace.Sampling.X?LumenProbeTrace:LumenRadianceTrace);
    GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier); if (!CheckGL("Trace Lumen radiance geometry")) return false;
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Buffers[1]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,work.Buffers[4]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,source->Buffer);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,work.Buffers[3]); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,record->Buffer); GlUseProgram(LumenRadianceResolve);
    GlActiveTexture(GL_TEXTURE0); LumenBindSampler(0,0); GlBindTexture(GL_TEXTURE_2D,source->Surface->Images[1]); GlUniform1i(GlGetUniformLocation(LumenRadianceResolve,"RadianceNormal"),0);
    GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier|LumenBufferBarrier); if (!CheckGL("Resolve Lumen ray radiance")) return false;
    LumenRadianceTraceResult result; result.Info=record->Info;
    if (upload.Readback) { if (!ReadLumenRadianceRecord(*record,result.Samples)) return false; }
    else if (!FrameOpen) { LumenFinish(); if (!CheckGL("Complete Lumen ray radiance")) return false; }
    result.Handle=NewLumenRadianceHandle(); if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,result.Info.RayCount);
    output=std::move(result); return true;
}
bool AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!ValidateLumenRadianceAssembly(assembly,LastError) || !LoadGL(LumenCopyRadiance,"glCopyBufferSubData")) return false;
    std::vector<std::shared_ptr<LumenRadianceRecord>> inputs;
    for (const auto& copy:assembly.Copies)
    {
        const auto source=LumenRadianceTraces.find(copy.Source.Value);
        if (source==LumenRadianceTraces.end()) return Fail("World probe copy has an expired source");
        if (!ValidateLumenRadianceCopy(source->second->Info,assembly,copy,LastError)) return false;
        if (!inputs.empty() && inputs.front()->Lighting!=source->second->Lighting) return Fail("World probe copy mixes lighting versions");
        inputs.push_back(source->second);
    }
    constexpr GLenum CopyRead=0x8F36,CopyWrite=0x8F37;
    struct Bindings
    {
        Impl& Owner; GLint Read=0,Write=0;
        explicit Bindings(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(CopyRead,&Read); Owner.GlGetIntegerv(CopyWrite,&Write); }
        ~Bindings() { Owner.GlBindBuffer(CopyRead,GLuint(Read)); Owner.GlBindBuffer(CopyWrite,GLuint(Write)); }
    } saved(*this);
    auto record=std::make_shared<LumenRadianceRecord>(*this); record->Lighting=inputs.front()->Lighting;
    record->Info=inputs.front()->Info; record->Info.Probes=assembly.Probes;
    record->Info.RayCount=unsigned(assembly.Probes.Origins.size())*assembly.Probes.RaysPerProbe*assembly.Probes.DepthLayers;
    GlGenBuffers(1,&record->Origins); GlBindBuffer(CopyWrite,record->Origins);
    GlBufferData(CopyWrite,assembly.Probes.Origins.size()*sizeof(LumenProbeOrigin),assembly.Probes.Origins.data(),GL_STATIC_DRAW);
    GlGenBuffers(1,&record->Buffer); GlBindBuffer(CopyWrite,record->Buffer);
    GlBufferData(CopyWrite,std::size_t(record->Info.RayCount)*sizeof(LumenRadianceSample),nullptr,GL_STATIC_DRAW);
    GpuMemoryBarrier(LumenBufferBarrier);
    const auto probeBytes=std::size_t(assembly.Probes.RaysPerProbe)*assembly.Probes.DepthLayers*sizeof(LumenRadianceSample);
    for (unsigned i=0;i<assembly.Copies.size();++i)
    {
        const auto& copy=assembly.Copies[i]; GlBindBuffer(CopyRead,inputs[i]->Buffer);
        LumenCopyRadiance(CopyRead,CopyWrite,copy.SourceProbe*probeBytes,copy.DestinationProbe*probeBytes,copy.ProbeCount*probeBytes);
    }
    GpuMemoryBarrier(GpuStorageBarrier|LumenBufferBarrier); LumenFinish();
    if (!CheckGL("Assemble directional world radiance")) return false;
    LumenRadianceTraceResult result; result.Info=record->Info; result.Handle=NewLumenRadianceHandle();
    if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenProbeOrigins(LumenRadianceHandle handle,std::vector<LumenProbeOrigin>& output)
{
    const auto found=LumenRadianceTraces.find(handle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen radiance trace");
    if (!LumenRadianceProbeCount(found->second->Info)) return Fail("Lumen trace has no probe origins");
    return ReadLumenOriginRecord(*found->second,output);
}
bool ReadLumenRadiance(LumenRadianceHandle handle,std::vector<LumenRadianceSample>& output)
{
    const auto found=LumenRadianceTraces.find(handle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen radiance trace");
    return ReadLumenRadianceRecord(*found->second,output);
}
bool DestroyLumenRadiance(LumenRadianceHandle handle) { return LumenRadianceTraces.erase(handle.Value)!=0 || Fail("Unknown Lumen radiance trace"); }
