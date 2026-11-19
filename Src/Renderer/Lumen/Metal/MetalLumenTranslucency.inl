LumenTranslucencyFrame LumenVolumeFrame;
LumenRadianceHandle LumenVolumeRays;
LumenProbeHandle LumenVolumeProbes;
void ReleaseLumenTranslucencyResults()
{
    if (LumenVolumeProbes) LumenProbes.erase(LumenVolumeProbes.Value);
    if (LumenVolumeRays) LumenRadianceTraces.erase(LumenVolumeRays.Value);
    LumenVolumeProbes={}; LumenVolumeRays={}; LumenVolumeFrame.Grid.W=0;
}

ObjCObject LumenVolumeAtlas=nullptr,LumenVolumePipeline=nullptr;
ObjCObject LumenVolumeSupportPipeline=nullptr,LumenVolumeSupport=nullptr;
unsigned LumenVolumeWidth=0,LumenVolumeHeight=0;
void DestroyLumenTranslucency()
{
    ReleaseLumenTranslucencyResults(); ReleaseObject(LumenVolumeAtlas); ReleaseObject(LumenVolumePipeline);
    LumenVolumeAtlas=LumenVolumePipeline=nullptr; LumenVolumeWidth=LumenVolumeHeight=0;
    ReleaseObject(LumenVolumeSupportPipeline); ReleaseObject(LumenVolumeSupport); LumenVolumeSupportPipeline=LumenVolumeSupport=nullptr;
}
bool PrepareLumenVolumeSupport(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    const auto source=LumenRadianceTraces.find(probes.SourceHandle.Value);
    if (source==LumenRadianceTraces.end() || !source->second->TraceDistance) return Fail("Metal Lumen volume lost its frozen distance source");
    LumenDistanceTraceFrame frame;
    if (!BuildLumenTranslucencySupportFrame(source->second->TraceDistance->Info,upload,frame,LastError)) return false;
    if (!LumenVolumeSupportPipeline)
    {
        std::string text; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",text,LastError)) return false;
        struct Compilation { ObjCObject Options=nullptr,Library=nullptr,Function=nullptr;
            ~Compilation() { ReleaseObject(Function); ReleaseObject(Library); ReleaseObject(Options); } } code;
        code.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(code.Options,"setFastMathEnabled:",ObjCNo);
        ObjCObject error=nullptr; code.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(text.c_str()),code.Options,&error);
        if (!code.Library) return Fail("Metal Lumen volume compilation failed: "+ErrorDescription(error));
        code.Function=Send<ObjCObject,ObjCObject>(code.Library,"newFunctionWithName:",MakeString("lumenVolumeSupport"));
        if (!code.Function) return Fail("Metal Lumen volume visibility kernel is unavailable");
        LumenVolumeSupportPipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,code.Function,&error);
        if (!LumenVolumeSupportPipeline) return Fail("Metal Lumen volume visibility pipeline failed: "+ErrorDescription(error));
        if (Send<NSUInteger>(LumenVolumeSupportPipeline,"maxTotalThreadsPerThreadgroup")<LumenRadianceThreads)
        { ReleaseObject(LumenVolumeSupportPipeline); LumenVolumeSupportPipeline=nullptr; return Fail("Metal Lumen volume visibility exceeds threadgroup capacity"); }
    }
    ReleaseObject(LumenVolumeSupport); LumenVolumeSupport=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::size_t(frame.Counts.X)*sizeof(float),0);
    if (!LumenVolumeSupport) return Fail("Metal Lumen volume visibility allocation failed");
    const auto encoder=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen volume compute encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",LumenVolumeSupportPipeline); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",source->second->TraceDistance->Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",source->second->Origins,0,2); Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",LumenVolumeSupport,0,3);
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{(frame.Counts.X+LumenRadianceThreads-1)/LumenRadianceThreads,1,1},MTLSize{LumenRadianceThreads,1,1});
    Send<void>(encoder,"endEncoding"); return true;
}
bool PackLumenTranslucency(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    if (!PrepareLumenVolumeSupport(probes,upload)) return false;
    if (!LumenVolumePipeline) LumenVolumePipeline=CreateTechniquePipeline("LumenTranslucency.metal","lumenVolumeVertex","lumenVolumeFragment",1,{PixelFormatRGBA16Float},false);
    if (!LumenVolumePipeline) return false;
    const unsigned width=unsigned(LumenVolumeFrame.Atlas.X),height=unsigned(LumenVolumeFrame.Atlas.Y);
    if (!LumenVolumeAtlas || LumenVolumeWidth!=width || LumenVolumeHeight!=height)
    {
        const auto image=CreateNativeTexture(width,height,PixelFormatRGBA16Float,TextureUsageShaderRead|TextureUsageRenderTarget,"Evans Lumen translucency volume");
        if (!image) return false;
        ReleaseObject(LumenVolumeAtlas); LumenVolumeAtlas=image; LumenVolumeWidth=width; LumenVolumeHeight=height;
    }
    const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    const auto attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(attachment,"setTexture:",LumenVolumeAtlas); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore);
    const auto encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal Lumen translucency encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",LumenVolumePipeline);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&LumenVolumeFrame,sizeof(LumenVolumeFrame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",probes.Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",LumenVolumeSupport,0,2);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
    Send<void>(encoder,"endEncoding"); return true;
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
