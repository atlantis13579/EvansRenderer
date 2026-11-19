ObjCObject LumenHairIndirect=nullptr,LumenHairSamples=nullptr,LumenHairComposite=nullptr;
std::unordered_map<unsigned,ObjCObject> LumenHairPipelines;
int LumenHairWidth=0,LumenHairHeight=0;
unsigned LumenHairSampleCount=0;
void DestroyLumenHairReceiver()
{
    ReleaseObject(LumenHairIndirect); ReleaseObject(LumenHairSamples); ReleaseObject(LumenHairComposite);
    for (auto entry:LumenHairPipelines) ReleaseObject(entry.second); LumenHairPipelines.clear();
    LumenHairIndirect=LumenHairSamples=LumenHairComposite=nullptr; LumenHairWidth=LumenHairHeight=0; LumenHairSampleCount=0;
}
bool EnsureLumenHairReceiver()
{
    if (!LumenHairPipelines.count(FrameSampleCount))
    {
        std::string vertexText,pixelText;
        if (!LoadTechniqueShaderSource("Metal","HairFur.metal",vertexText,LastError) || !LoadTechniqueShaderSource("Metal","LumenHair.metal",pixelText,LastError)) return false;
        struct Compilation
        {
            ObjCObject Options=nullptr,VertexLibrary=nullptr,PixelLibrary=nullptr,Vertex=nullptr,Pixel=nullptr,Descriptor=nullptr;
            ~Compilation() { ReleaseObject(Descriptor); ReleaseObject(Pixel); ReleaseObject(Vertex); ReleaseObject(PixelLibrary); ReleaseObject(VertexLibrary); ReleaseObject(Options); }
        } code;
        code.Options=PositionInvariantCompileOptions(); if (!SupportsPositionInvariance(code.Options)) return Fail("Fiber irradiance replay needs invariant vertex compilation");
        Send<void,ObjCBool>(code.Options,"setFastMathEnabled:",ObjCNo); ObjCObject error=nullptr;
        code.VertexLibrary=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(vertexText.c_str()),code.Options,&error);
        if (!code.VertexLibrary) return Fail("Metal fiber irradiance vertex compilation failed: "+ErrorDescription(error));
        code.PixelLibrary=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(pixelText.c_str()),code.Options,&error);
        if (!code.PixelLibrary) return Fail("Metal fiber irradiance fragment compilation failed: "+ErrorDescription(error));
        code.Vertex=Send<ObjCObject,ObjCObject>(code.VertexLibrary,"newFunctionWithName:",MakeString("VSMain"));
        code.Pixel=Send<ObjCObject,ObjCObject>(code.PixelLibrary,"newFunctionWithName:",MakeString("lumenFiberIndirect"));
        code.Descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
        Send<void,ObjCObject>(code.Descriptor,"setVertexFunction:",code.Vertex); Send<void,ObjCObject>(code.Descriptor,"setFragmentFunction:",code.Pixel);
        Send<void,NSUInteger>(code.Descriptor,"setSampleCount:",FrameSampleCount); Send<void,NSUInteger>(code.Descriptor,"setDepthAttachmentPixelFormat:",PixelFormatDepth32Float);
        const auto attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(code.Descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
        Send<void,NSUInteger>(attachment,"setPixelFormat:",PixelFormatRGBA16Float); Send<void,NSUInteger>(attachment,"setWriteMask:",ColorWriteMaskAll); Send<void,ObjCBool>(attachment,"setBlendingEnabled:",ObjCNo);
        const auto pipeline=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,code.Descriptor,&error);
        if (!pipeline) return Fail("Metal fiber irradiance pipeline failed: "+ErrorDescription(error)); LumenHairPipelines.emplace(FrameSampleCount,pipeline);
    }
    if (!LumenHairComposite) LumenHairComposite=CreateTechniquePipeline("LumenHair.metal","lumenFiberCompositeVertex","lumenFiberComposite",1,{PixelFormatRGBA16Float},false);
    if (!LumenHairComposite) return false;
    if (LumenHairIndirect && LumenHairWidth==CreateInfo.Width && LumenHairHeight==CreateInfo.Height && LumenHairSampleCount==FrameSampleCount) return true;
    const auto indirect=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,TextureUsageShaderRead|TextureUsageRenderTarget,"Evans fiber indirect lighting");
    if (!indirect) return false;
    const auto samples=FrameSampleCount>1?CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,TextureUsageRenderTarget,"Evans multisample fiber irradiance",1,FrameSampleCount):nullptr;
    if (FrameSampleCount>1 && !samples) { ReleaseObject(indirect); return false; }
    ReleaseObject(LumenHairIndirect); ReleaseObject(LumenHairSamples); LumenHairIndirect=indirect; LumenHairSamples=samples;
    LumenHairWidth=CreateInfo.Width; LumenHairHeight=CreateInfo.Height; LumenHairSampleCount=FrameSampleCount; return true;
}
bool ResolveLumenHairReceiver()
{
    if (!Hair.LumenReceiver) return true;
    if (!Hair.Resolved || !Hair.HasPrimaryView || !LumenVolumeFrame.Grid.W) return Fail("Fiber irradiance needs its completed primary surface and current SH volume");
    LumenHairReceiverFrame frame; HairFurConstants geometry;
    if (!BuildLumenHairReceiverFrame(LumenVolumeFrame,Hair.Frame.Settings,Hair.PrimaryView.Lighting,Hair.PrimaryView.ClipPlane,frame,LastError) ||
        !BuildHairFurReplayConstants(Hair.PrimaryView,Hair.Frame,HairFurRasterPhase::Surface,{},geometry,LastError) || !EnsureLumenHairReceiver()) return false;
    const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    const auto attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(attachment,"setTexture:",FrameSampleCount>1?LumenHairSamples:LumenHairIndirect);
    Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,MTLClearColor>(attachment,"setClearColor:",MTLClearColor{0,0,0,0});
    Send<void,NSUInteger>(attachment,"setStoreAction:",FrameSampleCount>1?StoreActionMultisampleResolve:StoreActionStore);
    if (FrameSampleCount>1) Send<void,ObjCObject>(attachment,"setResolveTexture:",LumenHairIndirect);
    const auto depth=Send<ObjCObject>(descriptor,"depthAttachment");
    Send<void,ObjCObject>(depth,"setTexture:",FrameSampleCount>1?Multisample.Depth:DepthTexture); Send<void,NSUInteger>(depth,"setLoadAction:",LoadActionLoad); Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore);
    const auto encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal fiber irradiance encoder is unavailable");
    const MTLViewport viewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1}; Send<void,MTLViewport>(encoder,"setViewport:",viewport);
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",LumenHairPipelines.at(FrameSampleCount)); Send<void,ObjCObject>(encoder,"setDepthStencilState:",HairVelocityDepth);
    Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone); Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",HairVertices,0,0); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&geometry,sizeof(geometry),1);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&frame,sizeof(frame),0); Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",LumenVolumeAtlas,0);
    Send<void,NSUInteger,NSUInteger,NSUInteger,ObjCObject,NSUInteger>(encoder,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",PrimitiveTypeTriangle,Hair.Frame.Geometry->Indices.size(),IndexTypeUInt32,HairIndices,0); Send<void>(encoder,"endEncoding");
    const auto compositeDescriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    const auto color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(compositeDescriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(color,"setTexture:",LumenGatherTarget); Send<void,NSUInteger>(color,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(color,"setStoreAction:",StoreActionStore);
    const auto composite=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",compositeDescriptor);
    if (!composite) return Fail("Metal fiber irradiance composition encoder is unavailable");
    Send<void,MTLViewport>(composite,"setViewport:",viewport); Send<void,ObjCObject>(composite,"setRenderPipelineState:",LumenHairComposite); Send<void,NSUInteger>(composite,"setCullMode:",CullModeNone); Send<void,NSUInteger>(composite,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,ObjCObject,NSUInteger>(composite,"setFragmentTexture:atIndex:",LumenHairIndirect,0); Send<void,ObjCObject,NSUInteger>(composite,"setFragmentTexture:atIndex:",HdrTextures[0],1);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(composite,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); Send<void>(composite,"endEncoding");
    const auto blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal fiber irradiance copy encoder is unavailable");
    const MTLOrigin origin{0,0,0}; const MTLSize size{NSUInteger(CreateInfo.Width),NSUInteger(CreateInfo.Height),1};
    Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",LumenGatherTarget,0,0,origin,size,HdrTextures[0],0,0,origin); Send<void>(blit,"endEncoding");
    ++PendingFrame.Statistics.LumenHairReceiverPassCount; ++PendingFrame.Statistics.GlobalIlluminationPassCount; return true;
}
