// Metal owns its FP32 accumulation textures and native pipeline states. The
// shared history transaction publishes a new mean only after frame completion.
PathTracingFrameState PathFrame;
std::array<ObjCObject,2> PathMeans{},PathPipelines{};
ObjCObject PathDepth=nullptr,PathDepthWriteState=nullptr;
int PathWidth=0,PathHeight=0;

void DestroyPathTargets()
{
    for (auto& image:PathMeans) { ReleaseObject(image); image=nullptr; }
    ReleaseObject(PathDepth); PathDepth=nullptr; PathWidth=PathHeight=0;
    PathFrame.History.Invalidate();
}
void DestroyPathTracing()
{
    DestroyPathTargets();
    for (auto& pipeline:PathPipelines) { ReleaseObject(pipeline); pipeline=nullptr; }
    ReleaseObject(PathDepthWriteState); PathDepthWriteState=nullptr; PathFrame={};
}
bool EnsurePathTracing()
{
    if (!EnsureRayRadiance()) return false;
    if (PathWidth!=CreateInfo.Width || PathHeight!=CreateInfo.Height)
    {
        DestroyPathTargets();
        for (auto& image:PathMeans) image=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA32Float,
            TextureUsageShaderRead|TextureUsageRenderTarget,"Evans path tracing mean");
        PathDepth=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatR32Float,
            TextureUsageShaderRead|TextureUsageRenderTarget,"Evans path tracing center depth");
        if (!PathMeans[0] || !PathMeans[1] || !PathDepth)
        { DestroyPathTargets(); return Fail("Metal path target allocation failed"); }
        PathWidth=CreateInfo.Width; PathHeight=CreateInfo.Height;
    }
    if (!PathDepthWriteState) PathDepthWriteState=CreateDepthState(CompareFunctionAlways,true);
    if (!PathDepthWriteState) return false;
    if (PathPipelines[0] && PathPipelines[1]) return true;
    std::string source;
    if (!LoadTechniqueShaderSource("Metal","PathTracing.metal",source,LastError)) return false;
    ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new");
    Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options);
    if (!library) return Fail("Metal path tracing compilation failed: "+ErrorDescription(error));
    ObjCObject vertex=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("VSMain"));
    std::array<ObjCObject,2> pipelines{};
    if (vertex) for (std::size_t phase=0;phase<pipelines.size();++phase)
    {
        ObjCObject fragment=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString(phase ? "PSCompose" : "PSMain"));
        if (!fragment) break;
        ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
        Send<void,ObjCObject>(descriptor,"setVertexFunction:",vertex);
        Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
        ObjCObject attachments=Send<ObjCObject>(descriptor,"colorAttachments");
        const NSUInteger count=phase ? 1 : 2;
        for (NSUInteger slot=0;slot<count;++slot)
        {
            ObjCObject color=Send<ObjCObject,NSUInteger>(attachments,"objectAtIndexedSubscript:",slot);
            Send<void,NSUInteger>(color,"setPixelFormat:",phase ? PixelFormatRGBA16Float : (slot ? PixelFormatR32Float : PixelFormatRGBA32Float));
            Send<void,NSUInteger>(color,"setWriteMask:",ColorWriteMaskAll);
            Send<void,ObjCBool>(color,"setBlendingEnabled:",ObjCNo);
        }
        if (phase) Send<void,NSUInteger>(descriptor,"setDepthAttachmentPixelFormat:",PixelFormatDepth32Float);
        error=nullptr;
        pipelines[phase]=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
        ReleaseObject(descriptor); ReleaseObject(fragment);
        if (!pipelines[phase]) break;
    }
    ReleaseObject(vertex); ReleaseObject(library);
    if (!pipelines[0] || !pipelines[1])
    { for (auto pipeline:pipelines) ReleaseObject(pipeline); return Fail("Metal path tracing pipeline failed: "+ErrorDescription(error)); }
    for (auto pipeline:PathPipelines) ReleaseObject(pipeline);
    PathPipelines=pipelines; return true;
}
bool TracePaths(const PathTracingDesc& desc)
{
    PathTracingConstants constants;
    if (!PathFrame.Prepare(desc,FrameRayRadianceSettings,FrameRayRadiance,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("Metal path frame requires matched scene inputs and a finite invertible camera");
    const auto plan=PathFrame.History.GetPlan();
    const MTLViewport viewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1};
    for (unsigned phase=0;phase<2;++phase)
    {
        ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
        ObjCObject attachments=Send<ObjCObject>(descriptor,"colorAttachments");
        const NSUInteger count=phase ? 1 : 2;
        for (NSUInteger slot=0;slot<count;++slot)
        {
            ObjCObject color=Send<ObjCObject,NSUInteger>(attachments,"objectAtIndexedSubscript:",slot);
            Send<void,ObjCObject>(color,"setTexture:",phase ? HdrTextures[0] : (slot ? PathDepth : PathMeans[plan.WriteIndex]));
            Send<void,NSUInteger>(color,"setLoadAction:",LoadActionClear);
            Send<void,NSUInteger>(color,"setStoreAction:",StoreActionStore);
        }
        if (phase)
        {
            ObjCObject depth=Send<ObjCObject>(descriptor,"depthAttachment");
            Send<void,ObjCObject>(depth,"setTexture:",DepthTexture);
            Send<void,NSUInteger>(depth,"setLoadAction:",LoadActionClear);
            Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore);
        }
        ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
        if (!encoder) return Fail("Metal path tracing encoder allocation failed");
        Send<void,MTLViewport>(encoder,"setViewport:",viewport);
        Send<void,ObjCObject>(encoder,"setRenderPipelineState:",PathPipelines[phase]);
        Send<void,ObjCObject>(encoder,"setDepthStencilState:",phase ? PathDepthWriteState : DepthDisabledState);
        Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
        Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",phase ? PathMeans[plan.WriteIndex] :
            (plan.PreviousSamples ? PathMeans[plan.ReadIndex] : DummyColorTexture),0);
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",phase ? PathDepth : DummyColorTexture,1);
        for (std::size_t slot=0;slot<RaySceneBufferCount;++slot)
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",RaySceneBuffers[slot],0,slot+1);
        for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot)
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",RayRadianceBuffers[slot],0,slot+1+RaySceneBufferCount);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
        Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
        Send<void>(encoder,"endEncoding");
    }
    ForwardColorWritten=true; return PathFrame.Resolve(PendingFrame.Statistics);
}
