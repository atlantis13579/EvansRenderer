// MetalRHI::Impl members. All textures, pipeline states and encoders belong to
// this native backend; the worker provides only immutable CPU bake/UV2 data.
static constexpr std::size_t LightmapFront=0,LightmapBack=1,LightmapCovered=2,LightmapResolved=3,LightmapComposite=4;
std::array<ObjCObject,5> LightmapTextures{};
std::array<ObjCObject,2> LightmapSurfacePipelines{};
ObjCObject LightmapCompositePipeline=nullptr,LightmapVertices=nullptr,LightmapDepthEqual=nullptr;
std::shared_ptr<const LightmapBakeResult> UploadedLightmap;
LightmapFrame FrameLightmaps;
Techniques::Settings FrameLightmapSettings;
bool LightmapActive=false;
int LightmapWidth=0,LightmapHeight=0;
std::uint32_t LightmapSamples=0,LightmapPipelineSamples=0;

void DestroyLightmapTargets(bool complete=false)
{
    for (std::size_t index=LightmapCovered;index<LightmapTextures.size();++index)
    { ReleaseObject(LightmapTextures[index]); LightmapTextures[index]=nullptr; }
    LightmapWidth=LightmapHeight=0; LightmapSamples=0;
    if (!complete) return;
    for (auto index:{LightmapFront,LightmapBack}) { ReleaseObject(LightmapTextures[index]); LightmapTextures[index]=nullptr; }
    for (auto& pipeline:LightmapSurfacePipelines) { ReleaseObject(pipeline); pipeline=nullptr; }
    ReleaseObject(LightmapCompositePipeline); ReleaseObject(LightmapVertices); ReleaseObject(LightmapDepthEqual);
    LightmapCompositePipeline=LightmapVertices=LightmapDepthEqual=nullptr; LightmapPipelineSamples=0;
    UploadedLightmap.reset(); FrameLightmaps={}; LightmapActive=false;
}

bool EnsureLightmapResources()
{
    if (!FrameLightmaps.Source || FrameLightmaps.Draws.empty()) return true;
    const auto& source=*FrameLightmaps.Source;
    if (!source.Receivers || !source.Irradiance || !source.Irradiance->Atlas || source.Receivers->Vertices.empty())
        return Fail("Metal lightmaps require a complete receiver and bake snapshot");
    if (LightmapPipelineSamples!=FrameSampleCount)
    {
        for (auto& pipeline:LightmapSurfacePipelines) { ReleaseObject(pipeline); pipeline=nullptr; }
        LightmapPipelineSamples=FrameSampleCount;
    }
    auto& surface=LightmapSurfacePipelines[FrameLightmaps.KeepFirstSurface ? 1 : 0];
    if (!surface) surface=CreateTechniquePipeline("Lightmaps.metal","VSMain","PSMain",FrameSampleCount,
        {PixelFormatRGBA16Float},true,ColorWriteMaskAll,false,nullptr,FrameLightmaps.KeepFirstSurface);
    if (!LightmapCompositePipeline) LightmapCompositePipeline=CreateTechniquePipeline("Lightmaps.metal","VSMain","PSMain",1,{PixelFormatRGBA16Float},false);
    if (!LightmapDepthEqual) LightmapDepthEqual=CreateDepthState(CompareFunctionEqual,false);
    if (!surface || !LightmapCompositePipeline || !LightmapDepthEqual) return false;
    if (UploadedLightmap!=FrameLightmaps.Source)
    {
        const auto& bake=*source.Irradiance; const auto side=bake.Atlas->Settings.Resolution;
        if (bake.FrontIrradiance.size()!=static_cast<std::size_t>(side)*side ||
            (!bake.BackIrradiance.empty() && bake.BackIrradiance.size()!=bake.FrontIrradiance.size())) return Fail("Metal lightmap upload size is invalid");
        ObjCObject front=CreateNativeTexture(side,side,PixelFormatRGBA32Float,TextureUsageShaderRead,"Lightmap front irradiance");
        ObjCObject back=CreateNativeTexture(side,side,PixelFormatRGBA32Float,TextureUsageShaderRead,"Lightmap back irradiance");
        ObjCObject vertices=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",
            source.Receivers->Vertices.data(),source.Receivers->Vertices.size()*sizeof(LightmapReceiverVertex),0);
        if (!front || !back || !vertices || !UploadTexture(front,side,side,sizeof(Float4),bake.FrontIrradiance.data()) ||
            !UploadTexture(back,side,side,sizeof(Float4),bake.BackIrradiance.empty() ? bake.FrontIrradiance.data() : bake.BackIrradiance.data()))
        { ReleaseObject(front); ReleaseObject(back); ReleaseObject(vertices); return Fail("Metal failed to upload the complete lightmap snapshot"); }
        ReleaseObject(LightmapTextures[LightmapFront]); ReleaseObject(LightmapTextures[LightmapBack]); ReleaseObject(LightmapVertices);
        LightmapTextures[LightmapFront]=front; LightmapTextures[LightmapBack]=back; LightmapVertices=vertices; UploadedLightmap=FrameLightmaps.Source;
    }
    if (LightmapWidth!=CreateInfo.Width || LightmapHeight!=CreateInfo.Height || LightmapSamples!=FrameSampleCount) DestroyLightmapTargets();
    for (std::size_t index=LightmapCovered;index<LightmapTextures.size();++index)
    {
        if (LightmapTextures[index]) continue;
        LightmapTextures[index]=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,
            TextureUsageRenderTarget|TextureUsageShaderRead,"Lightmap contribution",1,index==LightmapCovered ? FrameSampleCount : 1);
        if (!LightmapTextures[index]) { DestroyLightmapTargets(); return false; }
    }
    LightmapWidth=CreateInfo.Width; LightmapHeight=CreateInfo.Height; LightmapSamples=FrameSampleCount; return true;
}

bool ResolveLightmaps(const GlobalIlluminationDesc& desc)
{
    if (!FrameLightmaps.Source || FrameLightmaps.Draws.empty()) { GI.Resolved=true; return true; }
    if (UploadedLightmap!=FrameLightmaps.Source) return Fail("Metal lightmap snapshot changed during a frame");
    ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    ObjCObject attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(attachment,"setTexture:",LightmapTextures[LightmapCovered]);
    Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear);
    Send<void,MTLClearColor>(attachment,"setClearColor:",MTLClearColor{});
    Send<void,NSUInteger>(attachment,"setStoreAction:",FrameSampleCount>1 ? StoreActionMultisampleResolve : StoreActionStore);
    if (FrameSampleCount>1) Send<void,ObjCObject>(attachment,"setResolveTexture:",LightmapTextures[LightmapResolved]);
    ObjCObject depth=Send<ObjCObject>(descriptor,"depthAttachment");
    Send<void,ObjCObject>(depth,"setTexture:",FrameSampleCount>1 ? Multisample.Depth : DepthTexture);
    Send<void,NSUInteger>(depth,"setLoadAction:",LoadActionLoad); Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore);
    ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal failed to create lightmap receiver encoder");
    Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentSamplerState:atIndex:",MaterialSampler(),0);
    const MTLViewport viewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1};
    Send<void,MTLViewport>(encoder,"setViewport:",viewport);
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",LightmapSurfacePipelines[FrameLightmaps.KeepFirstSurface ? 1 : 0]);
    Send<void,ObjCObject>(encoder,"setDepthStencilState:",LightmapDepthEqual);
    Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",LightmapVertices,0,0);
    LightmapConstants constants;
    for (const auto& draw:FrameLightmaps.Draws)
    {
        if (!BuildLightmapConstants(FrameLightmaps,draw,desc.ViewProjection,FrameGI.Intensity,FrameLightmapSettings.Shading,FrameLightmapSettings.Metallic,constants))
        { Send<void>(encoder,"endEncoding"); return Fail("Metal lightmap receiver range or transform is invalid"); }
        auto texture=[&](TextureHandle handle) -> ObjCObject {
            const auto found=Textures.find(handle.Value);
            return handle && found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? found->second.Texture : nullptr;
        };
        const auto diffuse=texture(draw.DiffuseTexture),opacity=texture(draw.OpacityTexture);
        if ((draw.DiffuseTexture && !diffuse) || (draw.OpacityTexture && !opacity))
        { Send<void>(encoder,"endEncoding"); return Fail("Metal lightmap material texture is invalid"); }
        Send<void,NSUInteger>(encoder,"setCullMode:",draw.BackFaceCulling ? CullModeBack : CullModeNone);
        Send<void,NSUInteger>(encoder,"setFrontFacingWinding:",RealtimeWorldIsMirrored(draw.World) ? WindingClockwise : WindingCounterClockwise);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&constants,sizeof(constants),1);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
        const ObjCObject inputs[]{LightmapTextures[LightmapFront],LightmapTextures[LightmapBack],diffuse,opacity,DummyColorTexture,DummyColorTexture};
        for (NSUInteger slot=0;slot<std::size(inputs);++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",inputs[slot],slot);
        Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,draw.FirstVertex,draw.VertexCount);
    }
    Send<void>(encoder,"endEncoding");
    descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(attachment,"setTexture:",LightmapTextures[LightmapComposite]);
    Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore);
    encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal failed to create lightmap composition encoder");
    Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,ObjCObject>(encoder,"setRenderPipelineState:",LightmapCompositePipeline);
    Send<void,ObjCObject>(encoder,"setDepthStencilState:",DepthDisabledState); Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",LightmapVertices,0,0);
    constants.Options.W=static_cast<unsigned>(LightmapPass::Composite);
    constants.Options.X=AmbientOcclusionResolved ? 1u : 0u;
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&constants,sizeof(constants),1);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
    Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentSamplerState:atIndex:",SurfaceSampler,0);
    const ObjCObject inputs[]{AmbientOcclusionResolved ? AmbientOcclusionTextures[0] : DummyColorTexture,
        LightmapTextures[LightmapBack],DummyColorTexture,DummyColorTexture,
        HdrTextures[0],LightmapTextures[FrameSampleCount>1 ? LightmapResolved : LightmapCovered]};
    for (NSUInteger slot=0;slot<std::size(inputs);++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",inputs[slot],slot);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
    Send<void>(encoder,"endEncoding");
    ObjCObject blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder");
    if (!blit) return Fail("Metal failed to create lightmap composition copy encoder");
    const MTLOrigin origin{0,0,0}; const MTLSize extent{static_cast<NSUInteger>(CreateInfo.Width),static_cast<NSUInteger>(CreateInfo.Height),1};
    Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
        LightmapTextures[LightmapComposite],0,0,origin,extent,HdrTextures[0],0,0,origin);
    Send<void>(blit,"endEncoding"); GI.Resolved=true; PendingFrame.Statistics.GlobalIlluminationPassCount=2;
    RecordLightmapDraws(FrameLightmaps,PendingFrame.Statistics); return true;
}
