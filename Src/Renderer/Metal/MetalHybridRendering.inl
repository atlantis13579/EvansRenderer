// Each encoder snapshots its constants. Immutable shared buffers remain valid
// until command-buffer completion; cancelling a frame does not publish history.
std::array<ObjCObject,2> HybridRenderingImages{},HybridRenderingPipelines{};
int HybridRenderingWidth=0,HybridRenderingHeight=0;

void DestroyHybridRendering()
{
    for (auto& image:HybridRenderingImages) { ReleaseObject(image); image=nullptr; }
    for (auto& pipeline:HybridRenderingPipelines) { ReleaseObject(pipeline); pipeline=nullptr; }
    HybridRenderingWidth=HybridRenderingHeight=0;
}

bool EnsureHybridRenderingResources()
{
    if (!EnsureRayRadiance()) return false;
    if (HybridRenderingWidth!=CreateInfo.Width || HybridRenderingHeight!=CreateInfo.Height)
    {
        std::array<ObjCObject,2> images{};
        for (auto& image:images) image=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,
            TextureUsageShaderRead | TextureUsageRenderTarget,"Evans Metal hybrid reflection radiance");
        if (!images[0] || !images[1])
        { for (auto image:images) ReleaseObject(image); return Fail("Metal hybrid reflection target allocation failed"); }
        for (auto image:HybridRenderingImages) ReleaseObject(image);
        HybridRenderingImages=images; HybridRenderingWidth=CreateInfo.Width; HybridRenderingHeight=CreateInfo.Height;
    }
    if (HybridRenderingPipelines[0] && HybridRenderingPipelines[1]) return true;
    std::string source;
    if (!LoadTechniqueShaderSource("Metal","HybridRendering.metal",source,LastError)) return false;
    ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new");
    Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options);
    if (!library) return Fail("Metal hybrid reflection compilation failed: "+ErrorDescription(error));
    ObjCObject vertex=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("VSMain"));
    ObjCObject fragment=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("PSMain"));
    std::array<ObjCObject,2> pipelines{};
    if (vertex && fragment) for (std::size_t index=0;index<pipelines.size();++index)
    {
        const bool composite=index==1;
        ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
        Send<void,ObjCObject>(descriptor,"setVertexFunction:",vertex);
        Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
        ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
        Send<void,NSUInteger>(color,"setPixelFormat:",PixelFormatRGBA16Float);
        Send<void,NSUInteger>(color,"setWriteMask:",composite ? ColorWriteMaskRGB : ColorWriteMaskAll);
        Send<void,ObjCBool>(color,"setBlendingEnabled:",composite ? ObjCYes : ObjCNo);
        if (composite)
        {
            Send<void,NSUInteger>(color,"setSourceRGBBlendFactor:",BlendFactorOne);
            Send<void,NSUInteger>(color,"setDestinationRGBBlendFactor:",BlendFactorOne);
            Send<void,NSUInteger>(color,"setRgbBlendOperation:",BlendOperationAdd);
        }
        error=nullptr;
        pipelines[index]=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
        ReleaseObject(descriptor);
        if (!pipelines[index]) break;
    }
    ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
    if (!pipelines[0] || !pipelines[1])
    { for (auto pipeline:pipelines) ReleaseObject(pipeline); return Fail("Metal hybrid reflection pipeline failed: "+ErrorDescription(error)); }
    for (auto pipeline:HybridRenderingPipelines) ReleaseObject(pipeline);
    HybridRenderingPipelines=pipelines; return true;
}
bool ApplyHybridRendering(const ReflectionDesc& desc)
{
    if (!EnsureHybridRenderingResources()) return false;
    HybridRenderingConstants constants;
    const auto& frame=FrameRayRadiance;
    if (!BuildHybridRenderingConstants(desc,FrameRayRadianceSettings,FrameRayRadianceSettings.Hybrid,frame,CreateInfo.Width,CreateInfo.Height,
        Techniques::HybridRenderingPass::Trace,constants)) return Fail("Metal hybrid reflection frame is invalid");
    const MTLViewport viewport{0,0,static_cast<double>(CreateInfo.Width),static_cast<double>(CreateInfo.Height),0,1};
    for (const auto pass:HybridRenderingPasses)
    {
        const auto index=static_cast<std::size_t>(pass);
        const bool composite=pass==Techniques::HybridRenderingPass::Composite;
        const ObjCObject previous=index ? HybridRenderingImages[(index-1)%HybridRenderingImages.size()] : DummyColorTexture;
        ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
        ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
        Send<void,ObjCObject>(color,"setTexture:",composite ? HdrTextures[0] : HybridRenderingImages[index%HybridRenderingImages.size()]);
        Send<void,NSUInteger>(color,"setLoadAction:",composite ? LoadActionLoad : LoadActionClear);
        Send<void,NSUInteger>(color,"setStoreAction:",StoreActionStore);
        ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
        if (!encoder) return Fail("Metal hybrid reflection encoder allocation failed");
        Send<void,MTLViewport>(encoder,"setViewport:",viewport);
        Send<void,ObjCObject>(encoder,"setRenderPipelineState:",HybridRenderingPipelines[composite ? 1 : 0]);
        Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
        Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
        const std::array<ObjCObject,7> textures=composite ?
            std::array<ObjCObject,7>{DummyShadowTexture,DummyColorTexture,DummyColorTexture,DummyColorTexture,DummyColorTexture,previous,DummyColorTexture} :
            std::array<ObjCObject,7>{DepthTexture,GBufferTextures[1],GBufferTextures[2],GBufferTextures[0],GBufferTextures[3],previous,HdrTextures[0]};
        for (std::size_t slot=0;slot<textures.size();++slot)
            Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",textures[slot],slot);
        for (std::size_t slot=0;slot<RaySceneBufferCount;++slot)
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",RaySceneBuffers[slot],0,slot+1);
        for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot)
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",RayRadianceBuffers[slot],0,slot+1+RaySceneBufferCount);
        constants.Options.X=static_cast<std::uint32_t>(pass);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
        Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
        Send<void>(encoder,"endEncoding"); PendingFrame.ReflectionPasses.push_back(HybridRenderingLogPass(pass)); ++PendingFrame.Statistics.HybridRenderingPassCount;
    }
    ReflectionResolved=true;
    PendingFrame.Statistics.ReflectionPassCount=static_cast<std::uint32_t>(PendingFrame.ReflectionPasses.size()); return true;
}
