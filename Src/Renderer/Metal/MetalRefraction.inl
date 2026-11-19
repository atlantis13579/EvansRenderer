// MetalRHI::Impl members. Libraries are compiled independently for capture and
// composition. Resources remain native Metal objects; no shader translation.
std::array<ObjCObject,4> RefractionTextures{}; // Normal, tint, composite, depth.
std::array<ObjCObject,2> RefractionLibraries{},RefractionPipelines{};
ObjCObject RefractionVertices=nullptr,RefractionIndices=nullptr;
int RefractionWidth=0,RefractionHeight=0;
RefractionFrame FrameRefraction;
Techniques::RefractionSettings FrameRefractionSettings;
std::shared_ptr<const RefractionGeometry> UploadedRefraction;
bool RefractionActive=false,RefractionResolved=false;

void DestroyRefractionTargets(bool complete=false)
{
    for (auto& texture:RefractionTextures) { ReleaseObject(texture); texture=nullptr; }
    RefractionWidth=RefractionHeight=0;
    if (!complete) return;
    for (auto& pipeline:RefractionPipelines) { ReleaseObject(pipeline); pipeline=nullptr; }
    for (auto& library:RefractionLibraries) { ReleaseObject(library); library=nullptr; }
    ReleaseObject(RefractionVertices); ReleaseObject(RefractionIndices); RefractionVertices=RefractionIndices=nullptr;
    UploadedRefraction.reset(); FrameRefraction={}; RefractionActive=RefractionResolved=false;
}

bool EnsureRefractionResources()
{
    if (!FrameRefraction.Geometry || FrameRefraction.Draws.empty()) return true;
    for (unsigned phase=0;phase<RefractionLibraries.size();++phase)
    {
        if (!RefractionLibraries[phase])
        {
            std::string source; if (!LoadTechniqueShaderSource("Metal","Refraction.metal",source,LastError)) return false;
            if (phase==0) source.insert(0,"#define EVANS_REFRACTION_CAPTURE 1\n");
            ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr;
            Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
            RefractionLibraries[phase]=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
            ReleaseObject(options);
            if (!RefractionLibraries[phase]) return Fail("Metal refraction compilation failed: "+ErrorDescription(error));
        }
        if (!RefractionPipelines[phase]) RefractionPipelines[phase]=CreateTechniquePipeline("Refraction.metal","VSMain","PSMain",1,
            phase==0 ? std::vector<NSUInteger>{PixelFormatRGBA16Float,PixelFormatRGBA16Float} : std::vector<NSUInteger>{PixelFormatRGBA16Float},
            phase==0,ColorWriteMaskAll,false,RefractionLibraries[phase]);
        if (!RefractionPipelines[phase]) return false;
    }
    if (UploadedRefraction!=FrameRefraction.Geometry)
    {
        const auto& geometry=*FrameRefraction.Geometry;
        if (geometry.Vertices.empty() || geometry.Indices.empty()) return Fail("Metal refraction geometry is empty");
        ObjCObject vertices=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",
            geometry.Vertices.data(),geometry.Vertices.size()*sizeof(Vertex),0);
        ObjCObject indices=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",
            geometry.Indices.data(),geometry.Indices.size()*sizeof(std::uint32_t),0);
        if (!vertices || !indices) { ReleaseObject(vertices); ReleaseObject(indices); return Fail("Metal refraction geometry allocation failed"); }
        ReleaseObject(RefractionVertices); ReleaseObject(RefractionIndices);
        RefractionVertices=vertices; RefractionIndices=indices; UploadedRefraction=FrameRefraction.Geometry;
    }
    if (RefractionWidth!=CreateInfo.Width || RefractionHeight!=CreateInfo.Height) DestroyRefractionTargets();
    for (unsigned index=0;index<RefractionTextures.size();++index)
    {
        if (!RefractionTextures[index]) RefractionTextures[index]=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,
            index==3 ? PixelFormatDepth32Float : PixelFormatRGBA16Float,TextureUsageRenderTarget|TextureUsageShaderRead,"Refraction target");
        if (!RefractionTextures[index]) return false;
    }
    RefractionWidth=CreateInfo.Width; RefractionHeight=CreateInfo.Height; return true;
}

bool ResolveRefraction(const RefractionDesc& desc)
{
    if (!FrameRefraction.Geometry || FrameRefraction.Draws.empty()) { RefractionResolved=true; return true; }
    if (!EnsureRefractionResources()) return false;
    RefractionConstants base;
    if (!BuildRefractionConstants(desc.ViewProjection,desc.Environment,FrameRefractionSettings,CreateInfo.Width,CreateInfo.Height,base)) return Fail("Metal refraction projection is invalid");
    ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    auto colorAttachment=[&](ObjCObject pass,NSUInteger slot,ObjCObject texture) {
        ObjCObject attachment=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(pass,"colorAttachments"),"objectAtIndexedSubscript:",slot);
        Send<void,ObjCObject>(attachment,"setTexture:",texture); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear);
        Send<void,MTLClearColor>(attachment,"setClearColor:",MTLClearColor{}); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore);
    };
    colorAttachment(descriptor,0,RefractionTextures[0]); colorAttachment(descriptor,1,RefractionTextures[1]);
    ObjCObject depth=Send<ObjCObject>(descriptor,"depthAttachment");
    Send<void,ObjCObject>(depth,"setTexture:",RefractionTextures[3]); Send<void,NSUInteger>(depth,"setLoadAction:",LoadActionClear);
    Send<void,double>(depth,"setClearDepth:",1); Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore);
    ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal refraction capture encoder failed");
    const MTLViewport viewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1};
    Send<void,MTLViewport>(encoder,"setViewport:",viewport);
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",RefractionPipelines[0]); Send<void,ObjCObject>(encoder,"setDepthStencilState:",DepthWriteState);
    Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",RefractionVertices,0,0);
    for (const auto& draw:FrameRefraction.Draws)
    {
        RefractionConstants value; RefractionGeometryRange range;
        if (!BuildRefractionDrawConstants(FrameRefraction,draw,base,value,range)) { Send<void>(encoder,"endEncoding"); return Fail("Metal refraction draw is invalid"); }
        if (!range.IndexCount) continue;
        auto texture=[&](TextureHandle handle) -> ObjCObject {
            const auto found=Textures.find(handle.Value);
            return handle && found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? found->second.Texture : nullptr;
        };
        const auto diffuse=texture(draw.DiffuseTexture),normal=texture(draw.NormalTexture),opacity=texture(draw.OpacityTexture);
        if ((draw.DiffuseTexture && !diffuse) || (draw.NormalTexture && !normal) || (draw.OpacityTexture && !opacity))
        { Send<void>(encoder,"endEncoding"); return Fail("Metal refraction material texture is invalid"); }
        Send<void,NSUInteger>(encoder,"setCullMode:",draw.BackFaceCulling ? CullModeBack : CullModeNone);
        Send<void,NSUInteger>(encoder,"setFrontFacingWinding:",RealtimeWorldIsMirrored(draw.World) ? WindingClockwise : WindingCounterClockwise);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&value,sizeof(value),1);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&value,sizeof(value),0);
        const ObjCObject inputs[]{DepthTexture,diffuse ? diffuse : DummyColorTexture,normal ? normal : DummyColorTexture,opacity ? opacity : DummyColorTexture};
        for (NSUInteger slot=0;slot<std::size(inputs);++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",inputs[slot],slot);
        Send<void,NSUInteger,NSUInteger,NSUInteger,ObjCObject,NSUInteger>(encoder,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
            PrimitiveTypeTriangle,range.IndexCount,IndexTypeUInt32,RefractionIndices,std::size_t(range.FirstIndex)*sizeof(std::uint32_t));
    }
    Send<void>(encoder,"endEncoding"); descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    colorAttachment(descriptor,0,RefractionTextures[2]); encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal refraction composition encoder failed");
    Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",RefractionPipelines[1]); Send<void,ObjCObject>(encoder,"setDepthStencilState:",DepthDisabledState);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&base,sizeof(base),0);
    constexpr NSUInteger VolumeUniformSlot=9,VolumeImageSlot=19;
    auto volume=LumenVolumeFrame; if (!FrameGI.Lumen.Translucency.Enabled) volume.Grid.W=0;
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&volume,sizeof(volume),VolumeUniformSlot);
    Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",volume.Grid.W!=0?LumenVolumeAtlas:DummyColorTexture,VolumeImageSlot);
    const ObjCObject inputs[]{DepthTexture,DummyColorTexture,DummyColorTexture,DummyColorTexture,HdrTextures[0],RefractionTextures[3],RefractionTextures[0],RefractionTextures[1]};
    for (NSUInteger slot=0;slot<std::size(inputs);++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",inputs[slot],slot);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); Send<void>(encoder,"endEncoding");
    ObjCObject blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal refraction copy encoder failed");
    const MTLOrigin origin{0,0,0}; const MTLSize extent{static_cast<NSUInteger>(CreateInfo.Width),static_cast<NSUInteger>(CreateInfo.Height),1};
    Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
        RefractionTextures[2],0,0,origin,extent,HdrTextures[0],0,0,origin);
    Send<void>(blit,"endEncoding"); RefractionResolved=true; RecordRefractionDraws(FrameRefraction,PendingFrame.Statistics); return true;
}
