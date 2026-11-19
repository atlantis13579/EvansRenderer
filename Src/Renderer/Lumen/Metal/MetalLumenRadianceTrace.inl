struct LumenRadianceRecord
{
    std::shared_ptr<LumenDistanceRecord> TraceDistance;
    std::shared_ptr<LumenRadianceRecord> WorldCache;
    LumenRadianceInfo Info; ObjCObject Buffer=nullptr,Origins=nullptr; std::shared_ptr<LumenLightingRecord> Lighting;
    ~LumenRadianceRecord() { ReleaseObject(Buffer); ReleaseObject(Origins); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenRadianceRecord>> LumenRadianceTraces;
LumenFrameTransaction FrameLumen;
ObjCObject LumenRadianceTrace=nullptr,LumenRadianceResolve=nullptr,LumenProbeTrace=nullptr,LumenScreenPlacement=nullptr,LumenHybridTrace=nullptr,LumenScreenDepth=nullptr,LumenReflectionRayShader=nullptr;
ObjCObject LumenWorldLookup=nullptr;
bool EnsureLumenWorldLookup()
{
    if (LumenWorldLookup) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","LumenWorldCache.metal",source,LastError)) return false;
    struct Resources { ObjCObject Options=nullptr,Library=nullptr,Function=nullptr;
        ~Resources() { ReleaseObject(Function); ReleaseObject(Library); ReleaseObject(Options); } } work;
    work.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(work.Options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr; work.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),work.Options,&error);
    if (!work.Library) return Fail("Metal Lumen world lookup compilation failed: "+ErrorDescription(error));
    work.Function=Send<ObjCObject,ObjCObject>(work.Library,"newFunctionWithName:",MakeString("lumenWorldLookup"));
    if (!work.Function) return Fail("Metal Lumen world lookup kernel is unavailable");
    LumenWorldLookup=Evans::Renderer::CreateCachedMetalComputePipeline(Device,work.Function,&error);
    if (!LumenWorldLookup) return Fail("Metal Lumen world lookup pipeline failed: "+ErrorDescription(error));
    if (Send<NSUInteger>(LumenWorldLookup,"maxTotalThreadsPerThreadgroup")<LumenRadianceThreads)
    { ReleaseObject(LumenWorldLookup); LumenWorldLookup=nullptr; return Fail("Metal Lumen world lookup exceeds threadgroup capacity"); }
    return true;
}
void DestroyLumenRadianceTraces()
{ ReleaseObject(LumenWorldLookup); LumenWorldLookup=nullptr; LumenRadianceTraces.clear(); ReleaseObject(LumenRadianceTrace); ReleaseObject(LumenRadianceResolve); ReleaseObject(LumenProbeTrace); ReleaseObject(LumenScreenPlacement); ReleaseObject(LumenHybridTrace); ReleaseObject(LumenScreenDepth); ReleaseObject(LumenReflectionRayShader); LumenRadianceTrace=LumenRadianceResolve=LumenProbeTrace=LumenScreenPlacement=LumenHybridTrace=LumenScreenDepth=LumenReflectionRayShader=nullptr; }
bool EnsureLumenRadianceTrace(bool hardware=false)
{
    const auto nativeReady=[&]() { return !hardware || (EnsureLumenHardwarePipeline(LumenHardwareProgram::Radiance) && EnsureLumenHardwarePipeline(LumenHardwareProgram::Probes)); };

    if (LumenRadianceTrace && LumenRadianceResolve && LumenProbeTrace && LumenScreenPlacement && LumenHybridTrace && LumenScreenDepth && LumenReflectionRayShader) return nativeReady();
    std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
    struct Compilation
    {
        ObjCObject Options=nullptr,Library=nullptr,Functions[6]{};
        ~Compilation() { for (auto function:Functions) ReleaseObject(function); ReleaseObject(Library); ReleaseObject(Options); }
    } compilation;
    compilation.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(compilation.Options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr; compilation.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),compilation.Options,&error);
    if (!compilation.Library) return Fail("Metal Lumen radiance compilation failed: "+ErrorDescription(error));
    const char* names[]{"lumenTraceRadiance","lumenResolveRadiance","lumenTraceProbes","lumenPlaceScreenProbes","lumenTraceScreen","lumenBuildScreenDepth","lumenReflectionRays"}; ObjCObject* targets[]{&LumenRadianceTrace,&LumenRadianceResolve,&LumenProbeTrace,&LumenScreenPlacement,&LumenHybridTrace,&LumenScreenDepth,&LumenReflectionRayShader};
    for (unsigned stage=0;stage<std::size(names);++stage)
    {
        compilation.Functions[stage]=Send<ObjCObject,ObjCObject>(compilation.Library,"newFunctionWithName:",MakeString(names[stage]));
        if (!compilation.Functions[stage]) return Fail("Metal Lumen radiance kernel is unavailable");
        ReleaseObject(*targets[stage]); *targets[stage]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,compilation.Functions[stage],&error);
        if (!*targets[stage]) return Fail("Metal Lumen radiance pipeline failed: "+ErrorDescription(error));
        if (Send<NSUInteger>(*targets[stage],"maxTotalThreadsPerThreadgroup")<LumenRadianceThreads)
        { ReleaseObject(*targets[stage]); *targets[stage]=nullptr; return Fail("Metal Lumen radiance exceeds threadgroup capacity"); }
    }
    return nativeReady();
}
bool ReadLumenOriginRecord(const LumenRadianceRecord& record,std::vector<LumenProbeOrigin>& output)
{
    if (!LumenRadianceProbeCount(record.Info)) return Fail("This Lumen trace has no probe-origin buffer");
    const auto* contents=Send<void*>(record.Origins,"contents"); if (!contents) return Fail("Metal Lumen radiance buffer is unavailable");
    std::vector<LumenProbeOrigin> samples(LumenRadianceProbeCount(record.Info)); std::memcpy(samples.data(),contents,samples.size()*sizeof(LumenProbeOrigin));
    if (!ValidateLumenRadianceOrigins(record.Info,samples,LastError)) return false;
    output=std::move(samples); return true;
}
bool ReadLumenRadianceRecord(const LumenRadianceRecord& record,std::vector<LumenRadianceSample>& output)
{
    const auto* contents=Send<void*>(record.Buffer,"contents"); if (!contents) return Fail("Metal Lumen radiance buffer is unavailable");
    std::vector<LumenRadianceSample> samples(record.Info.RayCount); std::memcpy(samples.data(),contents,samples.size()*sizeof(LumenRadianceSample));
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
    if (!BuildLumenRadianceInput(source->Info,distance->Info,source->Surface->Captures,source->CardInstances,upload,input,LastError) || (FrameOpen && !FrameLumen.CanTrace(input.Resolve.Counts.X,LastError)) || !EnsureLumenRadianceTrace(bool(distance->Hardware))) return false;
    std::shared_ptr<LumenRadianceRecord> worldCache; LumenWorldLookupFrame worldFrame{};
    if (upload.WorldCache.Radiance)
    {
        const auto cache=LumenRadianceTraces.find(upload.WorldCache.Radiance.Value);
        if (cache==LumenRadianceTraces.end() || cache->second->Lighting!=source) return Fail("World radiance cache has a different or expired lighting source");
        worldCache=cache->second;
        if (!BuildLumenWorldLookupFrame(worldCache->Info,upload,input,worldFrame,LastError) || !EnsureLumenWorldLookup()) return false;
    }
    auto record=std::make_shared<LumenRadianceRecord>(); record->Info=MakeLumenRadianceInfo(source->Info.Identity,upload,input,worldCache?unsigned(worldFrame.Query.W):0); record->Info.HardwareTracing=bool(distance->Hardware); record->Lighting=source; record->WorldCache=worldCache; record->TraceDistance=distance;
    struct Resources
    {
        ObjCObject Mapping=nullptr,Hits=nullptr,Pyramid=nullptr;
        ~Resources() { ReleaseObject(Mapping); ReleaseObject(Hits); ReleaseObject(Pyramid); }
    } work;
    const auto count=std::size_t(input.Resolve.Counts.X); const auto sources=LumenRadianceSources(upload,input),traceFrame=LumenRadianceFrameData(input);
    record->Origins=upload.Screen.Enabled?Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",sources.Bytes,0):
        Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",sources.Data,sources.Bytes,0);
    work.Mapping=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",input.Words.data(),input.Words.size()*sizeof(std::uint32_t),0);
    work.Hits=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",count*sizeof(LumenDistanceHit),0);
    record->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",count*sizeof(LumenRadianceSample),0);
    if (!record->Origins || !work.Mapping || !work.Hits || !record->Buffer || !Send<void*>(record->Buffer,"contents")) return Fail("Metal Lumen radiance allocation failed");
    if (input.Depth.Bytes)
    { work.Pyramid=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",input.Depth.Bytes,0); if (!work.Pyramid) return Fail("Metal Lumen screen depth allocation failed"); }
    const auto command=FrameOpen?CommandBuffer:Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen radiance command buffer is unavailable");
    if (upload.Screen.Enabled)
    {
        const auto placement=Send<ObjCObject>(command,"computeCommandEncoder"); if (!placement) return Fail("Metal Lumen screen encoder is unavailable");
        Send<void,ObjCObject>(placement,"setComputePipelineState:",upload.Reflections.Enabled?LumenReflectionRayShader:LumenScreenPlacement);
        const auto parameters=LumenRadiancePlacementData(input);
        Send<void,const void*,NSUInteger,NSUInteger>(placement,"setBytes:length:atIndex:",parameters.Data,parameters.Bytes,0);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(placement,"setBuffer:offset:atIndex:",record->Origins,0,1);
        Send<void,ObjCObject,NSUInteger>(placement,"setTexture:atIndex:",DepthTexture,0); Send<void,ObjCObject,NSUInteger>(placement,"setTexture:atIndex:",GBufferTextures[1],1);
        Send<void,MTLSize,MTLSize>(placement,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{(input.Screen.Grid.Y+LumenRadianceThreads-1)/LumenRadianceThreads,1,1},MTLSize{LumenRadianceThreads,1,1});
        Send<void>(placement,"endEncoding");
    }
    const auto groups=unsigned((count+LumenRadianceThreads-1)/LumenRadianceThreads);
    const MTLSize grid{std::min(groups,LumenRadianceColumns),(groups+LumenRadianceColumns-1)/LumenRadianceColumns,1},threads{LumenRadianceThreads,1,1};
    if (upload.Hybrid.Enabled)
    {
        for (const auto& level:input.Depth.Levels)
        {
            const auto reduction=Send<ObjCObject>(command,"computeCommandEncoder"); if (!reduction) return Fail("Metal Lumen depth encoder is unavailable");
            Send<void,ObjCObject>(reduction,"setComputePipelineState:",LumenScreenDepth);
            Send<void,const void*,NSUInteger,NSUInteger>(reduction,"setBytes:length:atIndex:",&level,sizeof(level),0);
            Send<void,ObjCObject,NSUInteger,NSUInteger>(reduction,"setBuffer:offset:atIndex:",work.Pyramid,0,1);
            Send<void,ObjCObject,NSUInteger>(reduction,"setTexture:atIndex:",DepthTexture,0);
            const auto levelGroups=(level.Destination.X*level.Destination.Y+LumenRadianceThreads-1)/LumenRadianceThreads;
            Send<void,MTLSize,MTLSize>(reduction,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{std::min(levelGroups,LumenRadianceColumns),(levelGroups+LumenRadianceColumns-1)/LumenRadianceColumns,1},threads);
            Send<void>(reduction,"endEncoding");
        }
        const auto hybrid=Send<ObjCObject>(command,"computeCommandEncoder"); if (!hybrid) return Fail("Metal Lumen screen trace encoder is unavailable");
        Send<void,ObjCObject>(hybrid,"setComputePipelineState:",LumenHybridTrace);
        Send<void,const void*,NSUInteger,NSUInteger>(hybrid,"setBytes:length:atIndex:",&input.Hybrid,sizeof(input.Hybrid),0);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(hybrid,"setBuffer:offset:atIndex:",record->Origins,0,1);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(hybrid,"setBuffer:offset:atIndex:",work.Hits,0,2); Send<void,ObjCObject,NSUInteger,NSUInteger>(hybrid,"setBuffer:offset:atIndex:",record->Buffer,0,3);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(hybrid,"setBuffer:offset:atIndex:",work.Pyramid?work.Pyramid:record->Origins,0,4);
        const ObjCObject textures[]{DepthTexture,GBufferTextures[1],HdrTextures[0]};
        for (NSUInteger slot=0;slot<std::size(textures);++slot) Send<void,ObjCObject,NSUInteger>(hybrid,"setTexture:atIndex:",textures[slot],slot);
        Send<void,MTLSize,MTLSize>(hybrid,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(hybrid,"endEncoding");
    }
    if (worldCache)
    {
        const auto lookup=Send<ObjCObject>(command,"computeCommandEncoder"); if (!lookup) return Fail("Metal Lumen world lookup encoder is unavailable");
        Send<void,ObjCObject>(lookup,"setComputePipelineState:",LumenWorldLookup);
        Send<void,const void*,NSUInteger,NSUInteger>(lookup,"setBytes:length:atIndex:",&worldFrame,sizeof(worldFrame),0);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(lookup,"setBuffer:offset:atIndex:",worldCache->Buffer,0,1);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(lookup,"setBuffer:offset:atIndex:",record->Origins,0,2);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(lookup,"setBuffer:offset:atIndex:",work.Hits,0,3);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(lookup,"setBuffer:offset:atIndex:",record->Buffer,0,4);
        Send<void,MTLSize,MTLSize>(lookup,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(lookup,"endEncoding");
    }
    auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen radiance trace encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",distance->Hardware?LumenHardwarePipelines[unsigned(input.ProbeTrace.Sampling.X?LumenHardwareProgram::Probes:LumenHardwareProgram::Radiance)]:input.ProbeTrace.Sampling.X?LumenProbeTrace:LumenRadianceTrace);
    BindLumenHardware(encoder,*distance); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",traceFrame.Data,traceFrame.Bytes,0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",distance->Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record->Origins,0,2); Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Hits,0,3);
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    // A new encoder establishes the tracked write/read dependency on Hits.
    encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen radiance resolve encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",LumenRadianceResolve); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&input.Resolve,sizeof(input.Resolve),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Hits,0,1); Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",source->Buffer,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Mapping,0,3); Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record->Buffer,0,4);
    Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",source->Surface->Images[1],0); Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    if (!FrameOpen)
    {
        Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
        if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal Lumen radiance submission failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    }
    if (FrameOpen && distance->Hardware) PendingFrame.Statistics.LumenHardwareRayCount+=input.Resolve.Counts.X;
    LumenRadianceTraceResult result; result.Info=record->Info; if (upload.Readback && !ReadLumenRadianceRecord(*record,result.Samples)) return false;
    result.Handle=NewLumenRadianceHandle(); if (!result.Handle) return Fail("Lumen radiance handle space exhausted");
    LumenRadianceTraces.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,result.Info.RayCount);
    output=std::move(result); return true;
}
bool AssembleLumenRadiance(const LumenRadianceAssembly& assembly,LumenRadianceTraceResult& output)
{
    if (!ValidateLumenRadianceAssembly(assembly,LastError)) return false;
    std::vector<std::shared_ptr<LumenRadianceRecord>> sources;
    for (const auto& span:assembly.Copies)
    {
        const auto found=LumenRadianceTraces.find(span.Source.Value);
        if (found==LumenRadianceTraces.end()) return Fail("Metal world radiance source expired");
        if (!ValidateLumenRadianceCopy(found->second->Info,assembly,span,LastError)) return false;
        if (!sources.empty() && sources.front()->Lighting!=found->second->Lighting) return Fail("Metal world radiance sources have different lighting");
        sources.push_back(found->second);
    }
    auto record=std::make_shared<LumenRadianceRecord>(); record->Lighting=sources.front()->Lighting;
    record->Info=sources.front()->Info; record->Info.Probes=assembly.Probes;
    record->Info.RayCount=unsigned(assembly.Probes.Origins.size())*assembly.Probes.RaysPerProbe*assembly.Probes.DepthLayers;
    record->Origins=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",assembly.Probes.Origins.data(),assembly.Probes.Origins.size()*sizeof(LumenProbeOrigin),0);
    record->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::size_t(record->Info.RayCount)*sizeof(LumenRadianceSample),0);
    if (!record->Origins || !record->Buffer) return Fail("Metal world radiance assembly allocation failed");
    auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal world radiance command buffer is unavailable");
    auto blit=Send<ObjCObject>(command,"blitCommandEncoder"); if (!blit) return Fail("Metal world radiance blit encoder is unavailable");
    const auto probeBytes=NSUInteger(assembly.Probes.RaysPerProbe)*assembly.Probes.DepthLayers*sizeof(LumenRadianceSample);
    for (unsigned i=0;i<assembly.Copies.size();++i)
    {
        const auto& span=assembly.Copies[i];
        Send<void,ObjCObject,NSUInteger,ObjCObject,NSUInteger,NSUInteger>(blit,"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
            sources[i]->Buffer,span.SourceProbe*probeBytes,record->Buffer,span.DestinationProbe*probeBytes,span.ProbeCount*probeBytes);
    }
    Send<void>(blit,"endEncoding"); Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal world radiance assembly failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
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
