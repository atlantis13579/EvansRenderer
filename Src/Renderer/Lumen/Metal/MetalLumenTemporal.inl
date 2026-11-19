LumenTemporalHistory LumenHistory;
std::array<ObjCObject,2> LumenTemporalBuffers{};
ObjCObject LumenTemporalCompute=nullptr,LumenTemporalComposite=nullptr;
std::size_t LumenTemporalBytes=0;
void DestroyLumenTemporal()
{
    for (auto& buffer:LumenTemporalBuffers) { ReleaseObject(buffer); buffer=nullptr; } LumenTemporalBytes=0;
    ReleaseObject(LumenTemporalCompute); ReleaseObject(LumenTemporalComposite); LumenTemporalCompute=LumenTemporalComposite=nullptr; LumenHistory.Invalidate();
}
bool EnsureLumenTemporal(std::size_t bytes)
{
    if (LumenTemporalBytes!=bytes)
    {
        std::array<ObjCObject,2> buffers{};
        for (auto& buffer:buffers)
        {
            buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
            if (!buffer) { for (auto candidate:buffers) ReleaseObject(candidate); return Fail("Metal Lumen temporal history allocation failed"); }
        }
        for (auto buffer:LumenTemporalBuffers) ReleaseObject(buffer); LumenTemporalBuffers=buffers; LumenTemporalBytes=bytes; LumenHistory.Invalidate();
    }
    if (!LumenTemporalComposite) LumenTemporalComposite=CreateTechniquePipeline("Lumen.metal","lumenTemporalVertex","lumenTemporalComposite",1,{PixelFormatRGBA16Float},false);
    if (!LumenTemporalComposite) return false;
    if (!LumenTemporalCompute)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
        const auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        ObjCObject error=nullptr; const auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options); if (!library) return Fail("Metal Lumen temporal compilation failed: "+ErrorDescription(error));
        const auto function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("lumenTemporalResolve")); ReleaseObject(library);
        if (!function) return Fail("Metal Lumen temporal kernel is unavailable");
        LumenTemporalCompute=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error); ReleaseObject(function);
        if (!LumenTemporalCompute) return Fail("Metal Lumen temporal pipeline failed: "+ErrorDescription(error));
        if (Send<NSUInteger>(LumenTemporalCompute,"maxTotalThreadsPerThreadgroup")<LumenTemporalThreads)
        { ReleaseObject(LumenTemporalCompute); LumenTemporalCompute=nullptr; return Fail("Metal Lumen temporal exceeds threadgroup capacity"); }
    }
    return true;
}
bool ResolveLumenTemporal(const GlobalIlluminationDesc& camera)
{
    const LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    LumenTemporalFrame frame;
    if (!BuildLumenTemporalFrame(camera,FrameGI,CreateInfo.Width,CreateInfo.Height,LumenHistory.CanReuse(key),frame,LastError) ||
        !EnsureLumenTemporal(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    const auto read=LumenHistory.ReadIndex(),write=LumenHistory.WriteIndex();
    const auto compute=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!compute) return Fail("Metal Lumen temporal encoder is unavailable");
    Send<void,ObjCObject>(compute,"setComputePipelineState:",LumenTemporalCompute); Send<void,const void*,NSUInteger,NSUInteger>(compute,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(compute,"setBuffer:offset:atIndex:",LumenTemporalBuffers[read],0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(compute,"setBuffer:offset:atIndex:",LumenTemporalBuffers[write],0,2);
    const ObjCObject inputs[]{LumenGatherTarget,DepthTexture,GBufferTextures[1],MotionTextures[MotionVelocity]};
    for (NSUInteger slot=0;slot<std::size(inputs);++slot) Send<void,ObjCObject,NSUInteger>(compute,"setTexture:atIndex:",inputs[slot],slot);
    const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    const MTLSize grid{std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1},threads{LumenTemporalThreads,1,1};
    Send<void,MTLSize,MTLSize>(compute,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(compute,"endEncoding");
    const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    const auto attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(attachment,"setTexture:",LumenGatherTarget); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore);
    const auto render=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor); if (!render) return Fail("Metal Lumen temporal composition is unavailable");
    Send<void,ObjCObject>(render,"setRenderPipelineState:",LumenTemporalComposite); Send<void,NSUInteger>(render,"setCullMode:",CullModeNone); Send<void,NSUInteger>(render,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,MTLViewport>(render,"setViewport:",MTLViewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1});
    Send<void,const void*,NSUInteger,NSUInteger>(render,"setFragmentBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(render,"setFragmentBuffer:offset:atIndex:",LumenTemporalBuffers[write],0,1); Send<void,ObjCObject,NSUInteger>(render,"setFragmentTexture:atIndex:",HdrTextures[0],0);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(render,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); Send<void>(render,"endEncoding");
    const auto blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal Lumen temporal copy is unavailable");
    const MTLOrigin origin{0,0,0}; const MTLSize size{NSUInteger(CreateInfo.Width),NSUInteger(CreateInfo.Height),1};
    Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
        LumenGatherTarget,0,0,origin,size,HdrTextures[0],0,0,origin); Send<void>(blit,"endEncoding");
    LumenHistory.Record(key); PendingFrame.Temporal=camera.Sampling; PendingFrame.Statistics.GlobalIlluminationPassCount+=2; return true;
}
