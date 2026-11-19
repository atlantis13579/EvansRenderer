LumenRadianceHandle LumenReflectionRays;
ObjCObject LumenReflectionEstimate=nullptr,LumenReflectionCompositeTarget=nullptr,LumenReflectionResolvePipeline=nullptr;
std::size_t LumenReflectionEstimateBytes=0;
int LumenReflectionWidth=0,LumenReflectionHeight=0;
bool EnsureLumenReflectionTarget()
{
    if (LumenReflectionCompositeTarget && LumenReflectionWidth==CreateInfo.Width && LumenReflectionHeight==CreateInfo.Height) return true;
    const auto target=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,TextureUsageShaderRead|TextureUsageRenderTarget,"Evans Lumen reflection composition");
    if (!target) return false;
    ReleaseObject(LumenReflectionCompositeTarget); LumenReflectionCompositeTarget=target; LumenReflectionWidth=CreateInfo.Width; LumenReflectionHeight=CreateInfo.Height; return true;
}
LumenTemporalHistory LumenSpecularHistory;
std::array<ObjCObject,2> LumenSpecularBuffers{};
ObjCObject LumenSpecularCompute=nullptr,LumenSpecularComposite=nullptr;
std::size_t LumenSpecularBytes=0;
void DestroyLumenSpecular()
{
    for (auto& buffer:LumenSpecularBuffers) { ReleaseObject(buffer); buffer=nullptr; } LumenSpecularBytes=0;
    ReleaseObject(LumenSpecularCompute); ReleaseObject(LumenSpecularComposite); LumenSpecularCompute=LumenSpecularComposite=nullptr; LumenSpecularHistory.Invalidate();
}
bool EnsureLumenSpecular(std::size_t bytes)
{
    if (!EnsureLumenReflectionTarget()) return false;
    if (LumenSpecularBytes!=bytes)
    {
        std::array<ObjCObject,2> buffers{};
        for (auto& buffer:buffers)
        {
            buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
            if (!buffer) { for (auto candidate:buffers) ReleaseObject(candidate); return Fail("Metal Lumen specular temporal history allocation failed"); }
        }
        for (auto buffer:LumenSpecularBuffers) ReleaseObject(buffer); LumenSpecularBuffers=buffers; LumenSpecularBytes=bytes; LumenSpecularHistory.Invalidate();
    }
    if (!LumenSpecularComposite) LumenSpecularComposite=CreateTechniquePipeline("Lumen.metal","lumenTemporalVertex","lumenTemporalComposite",1,{PixelFormatRGBA16Float},false);
    if (!LumenSpecularComposite) return false;
    if (!LumenSpecularCompute)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
        const auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        ObjCObject error=nullptr; const auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options); if (!library) return Fail("Metal Lumen specular temporal compilation failed: "+ErrorDescription(error));
        const auto function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("lumenReflectionTemporal")); ReleaseObject(library);
        if (!function) return Fail("Metal Lumen specular temporal kernel is unavailable");
        LumenSpecularCompute=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error); ReleaseObject(function);
        if (!LumenSpecularCompute) return Fail("Metal Lumen specular temporal pipeline failed: "+ErrorDescription(error));
        if (Send<NSUInteger>(LumenSpecularCompute,"maxTotalThreadsPerThreadgroup")<LumenTemporalThreads)
        { ReleaseObject(LumenSpecularCompute); LumenSpecularCompute=nullptr; return Fail("Metal Lumen specular temporal exceeds threadgroup capacity"); }
    }
    return true;
}
bool ResolveLumenSpecular(const ReflectionDesc& camera)
{
    LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    key.Reflections=FrameReflections; key.Reflected=true; LumenTemporalFrame frame;
    if (!BuildLumenReflectionTemporalFrame(camera,FrameReflections,CreateInfo.Width,CreateInfo.Height,LumenSpecularHistory.CanReuse(key),frame,LastError) ||
        !EnsureLumenSpecular(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    const auto read=LumenSpecularHistory.ReadIndex(),write=LumenSpecularHistory.WriteIndex();
    const auto compute=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!compute) return Fail("Metal Lumen specular temporal encoder is unavailable");
    Send<void,ObjCObject>(compute,"setComputePipelineState:",LumenSpecularCompute); Send<void,const void*,NSUInteger,NSUInteger>(compute,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(compute,"setBuffer:offset:atIndex:",LumenSpecularBuffers[read],0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(compute,"setBuffer:offset:atIndex:",LumenSpecularBuffers[write],0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(compute,"setBuffer:offset:atIndex:",LumenReflectionEstimate,0,3);
    const ObjCObject inputs[]{HdrTextures[0],DepthTexture,GBufferTextures[1],MotionTextures[MotionVelocity]?MotionTextures[MotionVelocity]:GBufferTextures[1]};
    for (NSUInteger slot=0;slot<std::size(inputs);++slot) Send<void,ObjCObject,NSUInteger>(compute,"setTexture:atIndex:",inputs[slot],slot);
    const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    const MTLSize grid{std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1},threads{LumenTemporalThreads,1,1};
    Send<void,MTLSize,MTLSize>(compute,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(compute,"endEncoding");
    const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    const auto attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(attachment,"setTexture:",LumenReflectionCompositeTarget); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore);
    const auto render=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor); if (!render) return Fail("Metal Lumen specular temporal composition is unavailable");
    Send<void,ObjCObject>(render,"setRenderPipelineState:",LumenSpecularComposite); Send<void,NSUInteger>(render,"setCullMode:",CullModeNone); Send<void,NSUInteger>(render,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,MTLViewport>(render,"setViewport:",MTLViewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1});
    Send<void,const void*,NSUInteger,NSUInteger>(render,"setFragmentBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(render,"setFragmentBuffer:offset:atIndex:",LumenSpecularBuffers[write],0,1); Send<void,ObjCObject,NSUInteger>(render,"setFragmentTexture:atIndex:",HdrTextures[0],0);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(render,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); Send<void>(render,"endEncoding");
    const auto blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal Lumen specular temporal copy is unavailable");
    const MTLOrigin origin{0,0,0}; const MTLSize size{NSUInteger(CreateInfo.Width),NSUInteger(CreateInfo.Height),1};
    Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
        LumenReflectionCompositeTarget,0,0,origin,size,HdrTextures[0],0,0,origin); Send<void>(blit,"endEncoding");
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
    ReleaseObject(LumenReflectionEstimate); ReleaseObject(LumenReflectionCompositeTarget); ReleaseObject(LumenReflectionResolvePipeline);
    LumenReflectionEstimate=LumenReflectionCompositeTarget=LumenReflectionResolvePipeline=nullptr;
    LumenReflectionEstimateBytes=0; LumenReflectionWidth=LumenReflectionHeight=0;
}
bool EnsureLumenReflectionResolve(std::size_t bytes)
{
    if (LumenReflectionEstimateBytes!=bytes)
    {
        const auto buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
        if (!buffer) return Fail("Metal Lumen reflection estimate allocation failed");
        ReleaseObject(LumenReflectionEstimate); LumenReflectionEstimate=buffer; LumenReflectionEstimateBytes=bytes;
    }
    if (LumenReflectionResolvePipeline) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
    const auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr; const auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options); if (!library) return Fail("Metal Lumen reflection reconstruction compilation failed: "+ErrorDescription(error));
    const auto function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("lumenReflectionResolve")); ReleaseObject(library);
    if (!function) return Fail("Metal Lumen reflection reconstruction kernel is unavailable");
    LumenReflectionResolvePipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error); ReleaseObject(function);
    if (!LumenReflectionResolvePipeline) return Fail("Metal Lumen reflection reconstruction pipeline failed: "+ErrorDescription(error));
    if (Send<NSUInteger>(LumenReflectionResolvePipeline,"maxTotalThreadsPerThreadgroup")<LumenTemporalThreads)
    { ReleaseObject(LumenReflectionResolvePipeline); LumenReflectionResolvePipeline=nullptr; return Fail("Metal Lumen reflection reconstruction exceeds threadgroup capacity"); }
    return true;
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
        {sky[0],sky[1],sky[2]},FrameReflectionShading,frame,LastError,MakeLumenEnvironmentLobe(FrameGI.Lumen)) || !EnsureLumenReflectionResolve(std::size_t(frame.Dispatch.Z)*sizeof(Float4))) return false;
    const auto encoder=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen reflection reconstruction encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",LumenReflectionResolvePipeline);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",source->second->Origins,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",source->second->Buffer,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",LumenReflectionEstimate,0,3);
    const ObjCObject images[]{DepthTexture,GBufferTextures[1],GBufferTextures[0],GBufferTextures[3],GBufferTextures[2]};
    for (NSUInteger slot=0;slot<std::size(images);++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",images[slot],slot);
    const auto groups=(frame.Dispatch.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    const MTLSize grid{std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1},threads{LumenTemporalThreads,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::Trace); PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::LumenReconstruction);
    PendingFrame.Statistics.ReflectionPassCount+=2;
    if (!ResolveLumenSpecular(camera)) return false; ReflectionResolved=true; return true;
}
