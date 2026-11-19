struct VirtualShadowAtlasRecord
{
    VirtualShadowSnapshot Info; ObjCObject Depth=nullptr,Directory=nullptr;
    ~VirtualShadowAtlasRecord() { ReleaseObject(Depth); ReleaseObject(Directory); }
};
struct VirtualAtlasScissor { NSUInteger X,Y,Width,Height; };
std::unordered_map<std::uint64_t,std::shared_ptr<VirtualShadowAtlasRecord>> VirtualShadowAtlases;
ObjCObject VirtualAtlasRaster=nullptr,VirtualAtlasCopy=nullptr,VirtualAtlasRead=nullptr;
ObjCObject VirtualAtlasDepthState=nullptr,VirtualAtlasCopyState=nullptr;
void DestroyVirtualShadowAtlases()
{
    DestroyVirtualShadowLookup();
    VirtualShadowAtlases.clear();
    for (auto* object:{&VirtualAtlasRaster,&VirtualAtlasCopy,&VirtualAtlasRead,&VirtualAtlasDepthState,&VirtualAtlasCopyState})
    { ReleaseObject(*object); *object=nullptr; }
}
bool EnsureVirtualShadowAtlas()
{
    if (VirtualAtlasRaster && VirtualAtlasCopy && VirtualAtlasRead && VirtualAtlasDepthState && VirtualAtlasCopyState) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","VirtualShadowMaps.metal",source,LastError)) return false;
    struct Objects
    {
        ObjCObject Options=nullptr,Library=nullptr;
        ~Objects() { ReleaseObject(Library); ReleaseObject(Options); }
    } objects;
    objects.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(objects.Options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    objects.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),objects.Options,&error);
    if (!objects.Library) return Fail("Metal virtual atlas compilation failed: "+ErrorDescription(error));
    const char* vertices[]{"virtualAtlasDepthVertex","virtualAtlasCopyVertex"};
    const char* fragments[]{"virtualAtlasDepthFragment","virtualAtlasCopyFragment"};
    ObjCObject* pipelines[]{&VirtualAtlasRaster,&VirtualAtlasCopy};
    for (unsigned stage=0;stage<2;++stage) if (!*pipelines[stage])
    {
        const auto vertex=Send<ObjCObject,ObjCObject>(objects.Library,"newFunctionWithName:",MakeString(vertices[stage]));
        const auto fragment=Send<ObjCObject,ObjCObject>(objects.Library,"newFunctionWithName:",MakeString(fragments[stage]));
        const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
        Send<void,ObjCObject>(descriptor,"setVertexFunction:",vertex); Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
        Send<void,NSUInteger>(descriptor,"setDepthAttachmentPixelFormat:",PixelFormatDepth32Float); Send<void,NSUInteger>(descriptor,"setSampleCount:",1);
        if (vertex && fragment) *pipelines[stage]=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
        ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(descriptor);
        if (!*pipelines[stage]) return Fail("Metal virtual atlas render pipeline failed: "+ErrorDescription(error));
    }
    if (!VirtualAtlasRead)
    {
        const auto function=Send<ObjCObject,ObjCObject>(objects.Library,"newFunctionWithName:",MakeString("virtualAtlasRead"));
        if (function) VirtualAtlasRead=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!VirtualAtlasRead || Send<NSUInteger>(VirtualAtlasRead,"maxTotalThreadsPerThreadgroup")<VirtualShadowGroupWidth)
            return Fail("Metal virtual atlas read pipeline is unavailable: "+ErrorDescription(error));
    }
    if (!VirtualAtlasDepthState) VirtualAtlasDepthState=CreateDepthState(CompareFunctionLessEqual,true);
    if (!VirtualAtlasCopyState) VirtualAtlasCopyState=CreateDepthState(CompareFunctionAlways,true);
    return VirtualAtlasDepthState && VirtualAtlasCopyState;
}
bool EncodeVirtualDepthPack(ObjCObject command,ObjCObject depth,unsigned width,unsigned height,ObjCObject& feedback)
{
    const auto bytes=std::size_t(width)*height*sizeof(float);
    feedback=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
    if (!feedback || !Send<void*>(feedback,"contents")) return Fail("Metal virtual atlas feedback allocation failed");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal virtual atlas read encoder is unavailable");
    VirtualShadowRasterConstants constants; constants.Extent={width,height,VirtualShadowGroupColumns*VirtualShadowGroupWidth,0};
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",VirtualAtlasRead);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",feedback,0,1);
    Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",depth,0);
    const auto count=std::size_t(width)*height;
    const auto groups=static_cast<unsigned>((count+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth);
    const MTLSize grid{std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1},threads{VirtualShadowGroupWidth,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding"); return true;
}
bool EncodeVirtualAtlasRead(ObjCObject command,const VirtualShadowAtlasRecord& record,ObjCObject& feedback)
{ return EncodeVirtualDepthPack(command,record.Depth,record.Info.Layout.AtlasWidth,record.Info.Layout.AtlasHeight,feedback); }
bool CompleteVirtualAtlas(ObjCObject command,const VirtualShadowAtlasRecord& record,ObjCObject feedback,std::vector<float>* output)
{
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError)
        return Fail("Metal virtual atlas submission failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    if (output)
    {
        std::vector<float> result(std::size_t(record.Info.Layout.AtlasWidth)*record.Info.Layout.AtlasHeight);
        std::memcpy(result.data(),Send<void*>(feedback,"contents"),result.size()*sizeof(float));
        if (!ValidateVirtualShadowAtlasDepth(record.Info,result,LastError)) return false; *output=std::move(result);
    }
    return true;
}
bool ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("Metal virtual atlas handle is invalid");
    if (!EnsureVirtualShadowAtlas()) return false;
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal virtual atlas command buffer is unavailable");
    struct Feedback { ObjCObject Buffer=nullptr; ~Feedback(){ReleaseObject(Buffer);} } feedback;
    if (!EncodeVirtualAtlasRead(command,*found->second,feedback.Buffer)) return false;
    return CompleteVirtualAtlas(command,*found->second,feedback.Buffer,&output);
}
bool DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{ return (handle && VirtualShadowAtlases.erase(handle.Value)) || Fail("Metal virtual atlas handle is invalid or already released"); }
bool RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    std::shared_ptr<VirtualShadowAtlasRecord> parent;
    if (previous) { const auto found=VirtualShadowAtlases.find(previous.Value); if (found==VirtualShadowAtlases.end()) return Fail("Metal virtual atlas parent is invalid"); parent=found->second; }
    VirtualShadowPlan plan;
    if (!BuildVirtualShadowPlan(upload.Identity,upload.Settings,upload.Requests,upload.Invalidations,parent?&parent->Info:nullptr,plan,LastError) || !EnsureVirtualShadowAtlas()) return false;
    for (const auto& caster:upload.Casters)
    {
        const auto vertex=Buffers.find(caster.Vertices.Value),index=Buffers.find(caster.Indices.Value);
        if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Virtual atlas caster buffers are invalid");
        if (!ValidateVirtualShadowCaster(caster,vertex->second.Description,index->second.Description,index->second.Bytes,vertex->second.NaniteGeometry && index->second.NaniteGeometry,LastError)) return false;
        for (auto texture:{caster.Diffuse,caster.Opacity}) if (texture)
        { const auto image=Textures.find(texture.Value); if (image==Textures.end() || image->second.Description.Format!=TextureFormat::RGBA8) return Fail("Virtual atlas coverage requires existing color textures"); }
    }
    auto record=std::make_shared<VirtualShadowAtlasRecord>(); record->Info=plan.Candidate; const auto& layout=record->Info.Layout;
    const auto descriptor=Send<ObjCObject,NSUInteger,NSUInteger,NSUInteger,ObjCBool>(GetClass("MTLTextureDescriptor"),
        "texture2DDescriptorWithPixelFormat:width:height:mipmapped:",PixelFormatDepth32Float,layout.AtlasWidth,layout.AtlasHeight,ObjCNo);
    Send<void,NSUInteger>(descriptor,"setStorageMode:",StorageModePrivate);
    Send<void,NSUInteger>(descriptor,"setUsage:",TextureUsageRenderTarget|TextureUsageShaderRead);
    record->Depth=Send<ObjCObject,ObjCObject>(Device,"newTextureWithDescriptor:",descriptor);
    if (!record->Depth) return Fail("Metal virtual depth atlas allocation failed");
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal virtual atlas command buffer is unavailable");
    const auto pass=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    const auto attachment=Send<ObjCObject>(pass,"depthAttachment"); Send<void,ObjCObject>(attachment,"setTexture:",record->Depth);
    Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore); Send<void,double>(attachment,"setClearDepth:",1);
    const auto encoder=Send<ObjCObject,ObjCObject>(command,"renderCommandEncoderWithDescriptor:",pass);
    if (!encoder) return Fail("Metal virtual atlas depth encoder is unavailable");
    struct EndEncoder { ObjCObject Encoder; ~EndEncoder(){if(Encoder) Send<void>(Encoder,"endEncoding");} } end{encoder};
    Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",VirtualAtlasCopy); Send<void,ObjCObject>(encoder,"setDepthStencilState:",VirtualAtlasCopyState);
    Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
    const auto stride=layout.PageSide+VirtualShadowPageBorder*2;
    const auto frame=[&](const VirtualShadowRasterConstants& constants)
    {
        const MTLViewport viewport{constants.Origins.X,constants.Origins.Y,double(stride),double(stride),0,1};
        const VirtualAtlasScissor scissor{NSUInteger(constants.Origins.X),NSUInteger(constants.Origins.Y),stride,stride};
        Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,VirtualAtlasScissor>(encoder,"setScissorRect:",scissor);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&constants,sizeof(constants),1);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),1);
    };
    if (parent) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",parent->Depth,2);
    for (const auto& copy:plan.Copies)
    { frame(BuildVirtualShadowCopyConstants(parent->Info.Layout,layout,copy)); Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); }
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",VirtualAtlasRaster); Send<void,ObjCObject>(encoder,"setDepthStencilState:",VirtualAtlasDepthState);
    Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentSamplerState:atIndex:",SurfaceSampler,0);
    VirtualShadowAtlasResult result;
    for (const auto& tile:plan.Dirty) for (const auto& caster:upload.Casters)
    {
        VirtualShadowRasterConstants constants; if (!BuildVirtualShadowRasterConstants(record->Info,tile,caster,constants,LastError)) return false; frame(constants);
        Send<void,NSUInteger>(encoder,"setCullMode:",caster.BackFaceCulling?CullModeBack:CullModeNone);
        Send<void,NSUInteger>(encoder,"setFrontFacingWinding:",RealtimeWorldIsMirrored(caster.World)?WindingClockwise:WindingCounterClockwise);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",Buffers.at(caster.Vertices.Value).Buffer,0,0);
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",caster.Diffuse?Textures.at(caster.Diffuse.Value).Texture:nullptr,0);
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",caster.Opacity?Textures.at(caster.Opacity.Value).Texture:nullptr,1);
        Send<void,NSUInteger,NSUInteger,NSUInteger,ObjCObject,NSUInteger>(encoder,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
            PrimitiveTypeTriangle,caster.IndexCount,IndexTypeUInt32,Buffers.at(caster.Indices.Value).Buffer,NSUInteger(caster.FirstIndex)*sizeof(std::uint32_t));
        ++result.DrawCalls;
    }
    Send<void>(encoder,"endEncoding"); end.Encoder=nullptr;
    struct Feedback { ObjCObject Buffer=nullptr; ~Feedback(){ReleaseObject(Buffer);} } feedback;
    if (upload.Readback && !EncodeVirtualAtlasRead(command,*record,feedback.Buffer)) return false;
    if (!CompleteVirtualAtlas(command,*record,feedback.Buffer,upload.Readback?&result.Depth:nullptr)) return false;
    std::vector<unsigned> rendered; for (const auto& tile:plan.Dirty) rendered.push_back(tile.Page);
    if (!CompleteVirtualShadowPlan(plan,rendered,record->Info,LastError)) return false;
    if (!CreateVirtualShadowDirectory(*record)) return false;
    result.Handle=NewVirtualShadowAtlasHandle(); if (!result.Handle) return Fail("Virtual atlas identity space is exhausted");
    result.Info=record->Info; result.CopiedPages=static_cast<unsigned>(plan.Copies.size()); result.RenderedPages=static_cast<unsigned>(plan.Dirty.size());
    VirtualShadowAtlases.emplace(result.Handle.Value,record); output=std::move(result); return true;
}
