LumenLightingHandle FrameLumenLighting;
LumenWorldCacheBinding FrameLumenWorldCache;
LumenDistanceSceneHandle FrameLumenGlobalDistance;
LumenRadianceHandle LumenGIRays;
LumenProbeHandle LumenGIProbes;
bool LumenGatherIndirect=false;
ObjCObject LumenGatherPipeline=nullptr,LumenGatherTarget=nullptr;
ObjCObject LumenVisibilityPipeline=nullptr,LumenVisibilityBuffer=nullptr;
int LumenGatherWidth=0,LumenGatherHeight=0;
std::size_t LumenVisibilityBytes=0;
void DestroyLumenGather()
{
    LumenVisibilityBytes=0;
    ReleaseLumenGIResults(); DestroyLumenTemporal(); DestroyLumenReflections(); DestroyLumenTranslucency(); DestroyLumenHairReceiver(); LumenGatherIndirect=false; FrameLumenLighting={}; FrameLumenWorldCache={}; FrameLumenGlobalDistance={};
    ReleaseObject(LumenGatherPipeline); ReleaseObject(LumenGatherTarget); LumenGatherPipeline=LumenGatherTarget=nullptr;
    ReleaseObject(LumenVisibilityPipeline); ReleaseObject(LumenVisibilityBuffer); LumenVisibilityPipeline=LumenVisibilityBuffer=nullptr;
    LumenGatherWidth=LumenGatherHeight=0;
}
bool EnsureLumenGather(const LumenProbeGatherFrame& frame)
{
    if (!LumenGatherPipeline) LumenGatherPipeline=CreateTechniquePipeline("Lumen.metal","lumenGatherVertex","lumenGatherFragment",1,{PixelFormatRGBA16Float},false);
    if (!LumenGatherPipeline) return false;
    if (!LumenVisibilityPipeline)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
        const auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        ObjCObject error=nullptr; const auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options); if (!library) return Fail("Metal Lumen visibility compilation failed: "+ErrorDescription(error));
        const auto function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("lumenProbeVisibility")); ReleaseObject(library);
        if (!function) return Fail("Metal Lumen visibility kernel is unavailable");
        LumenVisibilityPipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error); ReleaseObject(function);
        if (!LumenVisibilityPipeline) return Fail("Metal Lumen visibility pipeline failed: "+ErrorDescription(error));
        if (Send<NSUInteger>(LumenVisibilityPipeline,"maxTotalThreadsPerThreadgroup")<LumenDistanceTraceThreads)
        { ReleaseObject(LumenVisibilityPipeline); LumenVisibilityPipeline=nullptr; return Fail("Metal Lumen visibility exceeds threadgroup capacity"); }
    }
    if (LumenGatherTarget && LumenGatherWidth==CreateInfo.Width && LumenGatherHeight==CreateInfo.Height && LumenVisibilityBytes==LumenProbeVisibilityBytes(frame)) return true;
    const auto target=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,TextureUsageShaderRead|TextureUsageRenderTarget,"Evans Lumen screen gather");
    if (!target) return false;
    const auto mask=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",LumenProbeVisibilityBytes(frame),0);
    if (!mask) { ReleaseObject(target); return Fail("Metal Lumen visibility allocation failed"); }
    ReleaseObject(LumenVisibilityBuffer); LumenVisibilityBuffer=mask; LumenVisibilityBytes=LumenProbeVisibilityBytes(frame);
    ReleaseObject(LumenGatherTarget); LumenGatherTarget=target; LumenGatherWidth=CreateInfo.Width; LumenGatherHeight=CreateInfo.Height; return true;
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
    LumenProbeGatherFrame frame; if (!BuildLumenProbeGatherFrame(record.Info,source->second->Lighting->Distance->Info,settings,FrameReflectionShading,frame,LastError) || !EnsureLumenGather(frame)) return false;
    const auto& distance=*source->second->Lighting->Distance;
    if (settings.Visibility.Enabled && distance.Hardware && !EnsureLumenHardwarePipeline(LumenHardwareProgram::Visibility)) return false;
    frame.Model.Z=LumenGatherIndirect?1u:0u;
    if (settings.Visibility.Enabled)
    {
        const auto compute=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!compute) return Fail("Metal Lumen visibility encoder is unavailable");
        Send<void,ObjCObject>(compute,"setComputePipelineState:",distance.Hardware?LumenHardwarePipelines[unsigned(LumenHardwareProgram::Visibility)]:LumenVisibilityPipeline);
        BindLumenHardware(compute,distance);
        Send<void,const void*,NSUInteger,NSUInteger>(compute,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(compute,"setBuffer:offset:atIndex:",source->second->Lighting->Distance->Buffer,0,1);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(compute,"setBuffer:offset:atIndex:",source->second->Origins,0,2);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(compute,"setBuffer:offset:atIndex:",LumenVisibilityBuffer,0,3);
        Send<void,ObjCObject,NSUInteger>(compute,"setTexture:atIndex:",DepthTexture,0); Send<void,ObjCObject,NSUInteger>(compute,"setTexture:atIndex:",GBufferTextures[1],1);
        const auto groups=(frame.Trace.Counts.X+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
        const MTLSize grid{std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1},threads{LumenDistanceTraceThreads,1,1};
        Send<void,MTLSize,MTLSize>(compute,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(compute,"endEncoding");
    }
    // A fresh render encoder orders the tracked visibility write before sampling.
    const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    const auto attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(attachment,"setTexture:",LumenGatherTarget); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore);
    const auto encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal Lumen gather encoder is unavailable");
    Send<void,MTLViewport>(encoder,"setViewport:",MTLViewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1});
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",LumenGatherPipeline); Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
    Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",source->second->Origins,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",record.Buffer,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",LumenVisibilityBuffer,0,3);
    const ObjCObject inputs[]{HdrTextures[0],DepthTexture,GBufferTextures[1],GBufferTextures[0],GBufferTextures[3]};
    for (NSUInteger slot=0;slot<std::size(inputs);++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",inputs[slot],slot);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); Send<void>(encoder,"endEncoding");
    if (!LumenGatherIndirect)
    {
        const auto blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal Lumen gather copy encoder is unavailable");
        const MTLOrigin origin{0,0,0}; const MTLSize size{NSUInteger(CreateInfo.Width),NSUInteger(CreateInfo.Height),1};
        Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
            LumenGatherTarget,0,0,origin,size,HdrTextures[0],0,0,origin); Send<void>(blit,"endEncoding");
    }
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
