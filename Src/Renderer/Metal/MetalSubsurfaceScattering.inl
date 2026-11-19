// Encoders preserve the full opaque scene and average the diffuse seed before
// diffusion. Every fragment draw snapshots its constants with setFragmentBytes.
enum SubsurfaceImage : std::size_t { SubsurfaceScene, SubsurfaceSource, SubsurfaceHorizontal, SubsurfaceVertical, SubsurfaceImageCount };
std::array<ObjCObject,SubsurfaceImageCount> SubsurfaceImages{};
ObjCObject SubsurfacePipeline=nullptr;
int SubsurfaceWidth=0,SubsurfaceHeight=0;
bool SubsurfaceProducing=false,SubsurfaceResolved=false,SubsurfaceHasScene=false;
DeferredLightingDesc SubsurfaceLighting;
Techniques::SubsurfaceScatteringSettings FrameSubsurface;
bool SubsurfaceRequested=false;
bool SubsurfaceComplete() const { return !SubsurfaceRequested || SubsurfaceResolved; }

void DestroySubsurface()
{
    for (auto& image:SubsurfaceImages) { ReleaseObject(image); image=nullptr; }
    ReleaseObject(SubsurfacePipeline); SubsurfacePipeline=nullptr; SubsurfaceWidth=SubsurfaceHeight=0;
}
bool EnsureSubsurface()
{
    if (SubsurfaceWidth!=CreateInfo.Width || SubsurfaceHeight!=CreateInfo.Height)
    {
        std::array<ObjCObject,SubsurfaceImageCount> images{};
        for (auto& image:images) image=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,
            TextureUsageShaderRead | TextureUsageRenderTarget,"Evans Metal subsurface radiance");
        if (std::any_of(images.begin(),images.end(),[](auto image){return !image;}))
        { for (auto image:images) ReleaseObject(image); return Fail("Metal subsurface target allocation failed"); }
        for (auto image:SubsurfaceImages) ReleaseObject(image);
        SubsurfaceImages=images; SubsurfaceWidth=CreateInfo.Width; SubsurfaceHeight=CreateInfo.Height;
    }
    if (SubsurfacePipeline) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","SubsurfaceScattering.metal",source,LastError)) return false;
    ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new");
    Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo); ObjCObject error=nullptr;
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options);
    if (!library) return Fail("Metal subsurface compilation failed: "+ErrorDescription(error));
    ObjCObject vertex=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("VSMain"));
    ObjCObject fragment=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("PSMain"));
    ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
    Send<void,ObjCObject>(descriptor,"setVertexFunction:",vertex); Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
    ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,NSUInteger>(color,"setPixelFormat:",PixelFormatRGBA16Float); Send<void,NSUInteger>(color,"setWriteMask:",ColorWriteMaskAll);
    Send<void,ObjCBool>(color,"setBlendingEnabled:",ObjCNo);
    if (vertex && fragment) SubsurfacePipeline=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
    ReleaseObject(descriptor); ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
    return SubsurfacePipeline || Fail("Metal subsurface pipeline failed: "+ErrorDescription(error));
}
bool SubsurfaceCopy(ObjCObject source,ObjCObject destination)
{
    ObjCObject blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder");
    if (!blit) return Fail("Metal subsurface copy encoder allocation failed");
    const MTLOrigin origin{0,0,0}; const MTLSize size{static_cast<NSUInteger>(CreateInfo.Width),static_cast<NSUInteger>(CreateInfo.Height),1};
    Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
        source,0,0,origin,size,destination,0,0,origin);
    Send<void>(blit,"endEncoding"); return true;
}
bool SnapshotSubsurfaceScene()
{ return EnsureSubsurface() && SubsurfaceCopy(HdrTextures[0],SubsurfaceImages[SubsurfaceScene]); }
bool CaptureSubsurfaceSource()
{
    if (FrameSampleCount<=1) return SubsurfaceCopy(HdrTextures[0],SubsurfaceImages[SubsurfaceSource]);
    ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(color,"setTexture:",Multisample.Colors[SampleColor]);
    Send<void,ObjCObject>(color,"setResolveTexture:",SubsurfaceImages[SubsurfaceSource]);
    Send<void,NSUInteger>(color,"setLoadAction:",LoadActionLoad); Send<void,NSUInteger>(color,"setStoreAction:",StoreActionMultisampleResolve);
    ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal subsurface resolve encoder allocation failed");
    Send<void>(encoder,"endEncoding"); return true;
}
bool FilterSubsurface(SubsurfaceScatteringConstants constants)
{
    const MTLViewport viewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1};
    for (auto pass:{Techniques::SubsurfacePass::Horizontal,Techniques::SubsurfacePass::Vertical,Techniques::SubsurfacePass::Composite})
    {
        const bool horizontal=pass==Techniques::SubsurfacePass::Horizontal,composite=pass==Techniques::SubsurfacePass::Composite;
        ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
        ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
        Send<void,ObjCObject>(color,"setTexture:",composite ? HdrTextures[0] : SubsurfaceImages[horizontal ? SubsurfaceHorizontal : SubsurfaceVertical]);
        Send<void,NSUInteger>(color,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(color,"setStoreAction:",StoreActionStore);
        ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
        if (!encoder) return Fail("Metal subsurface diffusion encoder allocation failed");
        Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,ObjCObject>(encoder,"setRenderPipelineState:",SubsurfacePipeline);
        Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone); Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
        const std::array<ObjCObject,5> textures{SubsurfaceImages[horizontal ? SubsurfaceSource : composite ? SubsurfaceVertical : SubsurfaceHorizontal],
            DepthTexture,GBufferTextures[1],SubsurfaceImages[SubsurfaceScene],SubsurfaceImages[SubsurfaceSource]};
        for (std::size_t slot=0;slot<textures.size();++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",textures[slot],slot);
        constants.Options.X=static_cast<unsigned>(pass);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
        Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
        Send<void>(encoder,"endEncoding"); ++PendingFrame.Statistics.SubsurfacePassCount;
    }
    SubsurfaceResolved=true; return true;
}
