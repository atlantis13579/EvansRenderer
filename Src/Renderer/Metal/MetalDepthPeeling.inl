// MetalRHI::Impl members. Tracked textures synchronize between render encoders;
// sampled previous layers never overlap the current attachment. Capture and
// replay libraries use the same invariant position compilation options.
DepthPeelingFrame FramePeeling;
DepthPeelingSequence PeelingSequence;
std::array<ObjCObject,2> PeelingDepth{},PeelingAccumulation{};
std::array<ObjCObject,3> PeelingLibraries{},PeelingPipelines{};
ObjCObject PeelingLayer=nullptr,PeelingEqualState=nullptr;
int PeelingWidth=0,PeelingHeight=0;
void DestroyPeelingTargets(bool complete=false)
{
    for (auto* array:{&PeelingDepth,&PeelingAccumulation})
        for (auto& texture:*array) { ReleaseObject(texture); texture=nullptr; }
    ReleaseObject(PeelingLayer); PeelingLayer=nullptr; PeelingWidth=PeelingHeight=0;
    if (!complete) return;
    for (auto& pipeline:PeelingPipelines) { ReleaseObject(pipeline); pipeline=nullptr; }
    for (auto& library:PeelingLibraries) { ReleaseObject(library); library=nullptr; }
    ReleaseObject(PeelingEqualState); PeelingEqualState=nullptr; PeelingSequence.Abort(); FramePeeling={};
}
bool EnsurePeelingResources()
{
    if (!ValidateDepthPeelingFrame(FramePeeling,LastError)) return false;
    if (!HdrActive || FrameSampleCount!=1 || WeightedActive)
        return Fail("Depth peeling requires single-sample HDR without weighted accumulation");
    const auto options=PositionInvariantCompileOptions();
    const bool invariant=SupportsPositionInvariance(options); ReleaseObject(options);
    if (!invariant) return Fail("Metal OIT requires guaranteed vertex position invariance (macOS 11 / iOS 14 or newer)");
    for (unsigned phase=0;phase<3;++phase)
    {
        if (!PeelingLibraries[phase])
        {
            std::string source; if (!LoadTechniqueShaderSource("Metal","OIT.metal",source,LastError)) return false;
            if (phase<2) source.insert(0,phase==0 ? "#define EVANS_OIT_CAPTURE 1\n" : "#define EVANS_OIT_ACCUMULATE 1\n");
            ObjCObject error=nullptr; PeelingLibraries[phase]=CompilePositionInvariantLibrary(Device,source,&error);
            if (!PeelingLibraries[phase]) return Fail("Metal OIT compilation failed: "+ErrorDescription(error));
        }
        if (!PeelingPipelines[phase]) PeelingPipelines[phase]=CreateTechniquePipeline("OIT.metal","VSMain","PSMain",1,
            phase==0 ? std::vector<NSUInteger>{} : std::vector<NSUInteger>{phase==1 ? PixelFormatRGBA32Float : PixelFormatRGBA16Float},
            phase==0,ColorWriteMaskAll,false,PeelingLibraries[phase]);
        if (!PeelingPipelines[phase]) return false;
    }
    if (!PeelingEqualState) PeelingEqualState=CreateDepthState(CompareFunctionEqual,false);
    if (!PeelingEqualState) return false;
    if (PeelingWidth!=CreateInfo.Width || PeelingHeight!=CreateInfo.Height) DestroyPeelingTargets();
    for (unsigned index=0;index<2;++index)
    {
        if (!PeelingDepth[index]) PeelingDepth[index]=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatDepth32Float,
            TextureUsageRenderTarget|TextureUsageShaderRead,"OIT captured depth");
        if (!PeelingAccumulation[index]) PeelingAccumulation[index]=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA32Float,
            TextureUsageRenderTarget|TextureUsageShaderRead,"OIT accumulated radiance and transmittance");
        if (!PeelingDepth[index] || !PeelingAccumulation[index]) return false;
    }
    if (!PeelingLayer) PeelingLayer=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,
        TextureUsageRenderTarget|TextureUsageShaderRead,"OIT shaded layer");
    if (!PeelingLayer) return false;
    PeelingWidth=CreateInfo.Width; PeelingHeight=CreateInfo.Height;
    return PeelingSequence.Begin(FramePeeling.Settings.LayerCount,LastError);
}
void PeelingColorAttachment(ObjCObject descriptor,NSUInteger slot,ObjCObject texture,MTLClearColor clear)
{
    const auto attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",slot);
    Send<void,ObjCObject>(attachment,"setTexture:",texture); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear);
    Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore); Send<void,MTLClearColor>(attachment,"setClearColor:",clear);
}
void PeelingDepthAttachment(ObjCObject descriptor,ObjCObject texture)
{
    const auto attachment=Send<ObjCObject>(descriptor,"depthAttachment"); Send<void,ObjCObject>(attachment,"setTexture:",texture);
    Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore);
    Send<void,double>(attachment,"setClearDepth:",PeelingClearDepth);
}
bool ProcessPeeling(DepthPeelingPass pass)
{
    const auto step=PeelingSequence.Step(); DepthPeelingConstants base;
    if (!BuildDepthPeelingConstants(FramePeeling.ViewProjection,CreateInfo.Width,CreateInfo.Height,step.FirstLayer,FramePeeling.Settings.ShowOverflow,base))
        return Fail("Invalid Metal OIT frame constants");
    const MTLViewport viewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1};
    if (pass==DepthPeelingPass::Capture)
    {
        if (!PeelingSequence.Advance(DepthPeelingEvent::BeginCapture,LastError)) return false;
        if (step.FirstLayer)
        {
            const auto clear=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
            for (NSUInteger index=0;index<2;++index) PeelingColorAttachment(clear,index,PeelingAccumulation[index],MTLClearColor{0,0,0,1});
            PeelingDepthAttachment(clear,PeelingDepth[step.DepthRead]);
            const auto encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",clear);
            if (!encoder) return Fail("Metal OIT clear encoder failed");
            Send<void>(encoder,"endEncoding");
        }
        const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
        PeelingDepthAttachment(descriptor,PeelingDepth[step.DepthWrite]);
        const auto encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
        if (!encoder) return Fail("Metal OIT capture encoder failed");
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentSamplerState:atIndex:",MaterialSampler(),0);
        Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,ObjCObject>(encoder,"setRenderPipelineState:",PeelingPipelines[0]);
        Send<void,ObjCObject>(encoder,"setDepthStencilState:",DepthWriteState); Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
        auto texture=[&](TextureHandle handle)->ObjCObject {
            const auto found=Textures.find(handle.Value);
            return handle && found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? found->second.Texture : nullptr;
        };
        for (const auto& draw:FramePeeling.Draws)
        {
            if (!draw.IndexCount) continue;
            const auto vertex=Buffers.find(draw.VertexBuffer.Value),index=Buffers.find(draw.IndexBuffer.Value);
            if (vertex==Buffers.end() || index==Buffers.end()) { Send<void>(encoder,"endEncoding"); return Fail("Invalid Metal OIT buffers"); }
            if (!ValidateDepthPeelingBuffers(draw,vertex->second.Description,index->second.Description,vertex->second.Bytes.size(),index->second.Bytes,LastError,vertex->second.NaniteGeometry && index->second.NaniteGeometry))
            { Send<void>(encoder,"endEncoding"); return false; }
            DepthPeelingConstants value;
            if (!BuildDepthPeelingDrawConstants(base,draw.World,draw.Opacity,draw.AlphaCutoff,bool(draw.DiffuseTexture),bool(draw.OpacityTexture),value))
            { Send<void>(encoder,"endEncoding"); return Fail("Invalid Metal OIT draw constants"); }
            const auto diffuse=texture(draw.DiffuseTexture),opacity=texture(draw.OpacityTexture);
            if ((draw.DiffuseTexture && !diffuse) || (draw.OpacityTexture && !opacity)) { Send<void>(encoder,"endEncoding"); return Fail("Invalid Metal OIT material texture"); }
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",vertex->second.Buffer,0,0);
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&value,sizeof(value),1);
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&value,sizeof(value),0);
            Send<void,NSUInteger>(encoder,"setCullMode:",draw.BackFaceCulling ? CullModeBack : CullModeNone);
            Send<void,NSUInteger>(encoder,"setFrontFacingWinding:",RealtimeWorldIsMirrored(draw.World) ? WindingClockwise : WindingCounterClockwise);
            const ObjCObject images[]{DepthTexture,PeelingDepth[step.DepthRead],diffuse ? diffuse : DummyColorTexture,opacity ? opacity : DummyColorTexture};
            for (NSUInteger slot=0;slot<std::size(images);++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",images[slot],slot);
            Send<void,NSUInteger,NSUInteger,NSUInteger,ObjCObject,NSUInteger,NSUInteger,NSInteger,NSUInteger>(encoder,
                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                PrimitiveTypeTriangle,draw.IndexCount,IndexTypeUInt32,index->second.Buffer,std::size_t(draw.FirstIndex)*sizeof(std::uint32_t),1,draw.BaseVertex,0);
        }
        Send<void>(encoder,"endEncoding"); return PeelingSequence.Advance(DepthPeelingEvent::EndCapture,LastError);
    }
    const bool compose=pass==DepthPeelingPass::Compose;
    if ((!compose && PeelingSequence.Stage()!=DepthPeelingStage::NeedAccumulation) ||
        (compose && PeelingSequence.Stage()!=DepthPeelingStage::NeedComposition)) return Fail("Metal OIT resolve is out of order");
    const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    PeelingColorAttachment(descriptor,0,compose ? PeelingLayer : PeelingAccumulation[step.AccumulationWrite],MTLClearColor{});
    const auto encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal OIT accumulation/composition encoder failed");
    Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,ObjCObject>(encoder,"setRenderPipelineState:",PeelingPipelines[compose ? 2 : 1]);
    Send<void,ObjCObject>(encoder,"setDepthStencilState:",DepthDisabledState); Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&base,sizeof(base),0);
    const ObjCObject images[]{compose ? DummyColorTexture : PeelingLayer,PeelingAccumulation[step.AccumulationRead],HdrTextures[0],PeelingDepth[step.DepthWrite]};
    for (NSUInteger slot=0;slot<std::size(images);++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",images[slot],slot+4);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); Send<void>(encoder,"endEncoding");
    if (compose)
    {
        const auto blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal OIT copy encoder failed");
        const MTLOrigin origin{0,0,0}; const MTLSize extent{static_cast<NSUInteger>(CreateInfo.Width),static_cast<NSUInteger>(CreateInfo.Height),1};
        Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
            PeelingLayer,0,0,origin,extent,HdrTextures[0],0,0,origin);
        Send<void>(blit,"endEncoding");
    }
    return PeelingSequence.Advance(compose ? DepthPeelingEvent::Compose : DepthPeelingEvent::Accumulate,LastError);
}
