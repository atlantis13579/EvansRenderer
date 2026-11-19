struct LumenSurfaceRecord
{
    LumenSurfaceIdentity Identity; LumenSurfaceLayout Layout;
    std::vector<LumenSurfaceCapture> Captures;
    std::array<ObjCObject,LumenSurfacePlanes> Images{}; ObjCObject Depth=nullptr;
    ~LumenSurfaceRecord() { for (auto image:Images) ReleaseObject(image); ReleaseObject(Depth); }
};
struct LumenCaptureScissor { NSUInteger X,Y,Width,Height; };
std::unordered_map<std::uint64_t,std::shared_ptr<LumenSurfaceRecord>> LumenSurfaces;
ObjCObject LumenCapturePipeline=nullptr,LumenMaterialRead=nullptr,LumenCaptureDepth=nullptr;
void DestroyLumenSurfaces()
{
    DestroyLumenCaptureAsync();
    LumenSurfaces.clear();
    for (auto* object:{&LumenCapturePipeline,&LumenMaterialRead,&LumenCaptureDepth}) { ReleaseObject(*object); *object=nullptr; }
}
bool EnsureLumenSurface()
{
    if (LumenCapturePipeline && LumenMaterialRead && LumenCaptureDepth) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
    struct Objects
    {
        ObjCObject Options=nullptr,Library=nullptr;
        ~Objects() { ReleaseObject(Options); ReleaseObject(Library); }
    } objects;
    objects.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(objects.Options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    objects.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),objects.Options,&error);
    if (!objects.Library) return Fail("Metal Lumen material compilation failed: "+ErrorDescription(error));
    if (!LumenCapturePipeline)
    {
        const auto vertex=Send<ObjCObject,ObjCObject>(objects.Library,"newFunctionWithName:",MakeString("lumenCaptureVertex"));
        const auto fragment=Send<ObjCObject,ObjCObject>(objects.Library,"newFunctionWithName:",MakeString("lumenCaptureFragment"));
        const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
        Send<void,ObjCObject>(descriptor,"setVertexFunction:",vertex); Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
        Send<void,NSUInteger>(descriptor,"setDepthAttachmentPixelFormat:",PixelFormatDepth32Float); Send<void,NSUInteger>(descriptor,"setSampleCount:",1);
        const auto colors=Send<ObjCObject>(descriptor,"colorAttachments");
        for (NSUInteger plane=0;plane<LumenSurfacePlanes;++plane)
        {
            const auto target=Send<ObjCObject,NSUInteger>(colors,"objectAtIndexedSubscript:",plane);
            Send<void,NSUInteger>(target,"setPixelFormat:",PixelFormatRGBA32Float); Send<void,ObjCBool>(target,"setBlendingEnabled:",ObjCNo);
        }
        if (vertex && fragment) LumenCapturePipeline=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
        ReleaseObject(descriptor); ReleaseObject(vertex); ReleaseObject(fragment);
        if (!LumenCapturePipeline) return Fail("Metal Lumen capture pipeline failed: "+ErrorDescription(error));
    }
    if (!LumenMaterialRead)
    {
        const auto function=Send<ObjCObject,ObjCObject>(objects.Library,"newFunctionWithName:",MakeString("lumenReadMaterials"));
        if (function) LumenMaterialRead=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!LumenMaterialRead || Send<NSUInteger>(LumenMaterialRead,"maxTotalThreadsPerThreadgroup")<LumenSurfaceReadThreads)
            return Fail("Metal Lumen material export pipeline is unavailable: "+ErrorDescription(error));
    }
    if (!LumenCaptureDepth) LumenCaptureDepth=CreateDepthState(CompareFunctionLessEqual,true);
    return LumenCaptureDepth!=nullptr;
}
bool EncodeLumenRead(ObjCObject command,const LumenSurfaceRecord& record,ObjCObject& feedback)
{
    const auto count=std::size_t(record.Layout.Width)*record.Layout.Height,bytes=count*sizeof(LumenSurfaceTexel);
    feedback=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
    if (!feedback || !Send<void*>(feedback,"contents")) return Fail("Metal Lumen material feedback allocation failed");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen material export encoder is unavailable");
    const ShaderUInt4 extent{record.Layout.Width,record.Layout.Height,LumenSurfaceReadColumns*LumenSurfaceReadThreads,0};
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",LumenMaterialRead);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&extent,sizeof(extent),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",feedback,0,1);
    for (NSUInteger plane=0;plane<LumenSurfacePlanes;++plane) Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",record.Images[plane],plane);
    const auto groups=unsigned((count+LumenSurfaceReadThreads-1)/LumenSurfaceReadThreads);
    const MTLSize grid{std::min(groups,LumenSurfaceReadColumns),(groups+LumenSurfaceReadColumns-1)/LumenSurfaceReadColumns,1},threads{LumenSurfaceReadThreads,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding"); return true;
}
bool CompleteLumenCapture(ObjCObject command,const LumenSurfaceRecord& record,ObjCObject feedback,std::vector<LumenSurfaceTexel>* output)
{
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError)
        return Fail("Metal Lumen material submission failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    if (output)
    {
        std::vector<LumenSurfaceTexel> result(std::size_t(record.Layout.Width)*record.Layout.Height);
        std::memcpy(result.data(),Send<void*>(feedback,"contents"),result.size()*sizeof(LumenSurfaceTexel));
        if (!ValidateLumenSurfaceTexels(record.Layout,result,LastError)) return false; *output=std::move(result);
    }
    return true;
}
bool CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    LumenSurfaceLayout layout; if (!ValidateLumenSurfaceUpload(upload,layout,LastError) || !EnsureLumenSurface()) return false;
    for (const auto& draw:upload.Draws)
    {
        const auto vertices=Buffers.find(draw.Vertices.Value),indices=Buffers.find(draw.Indices.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Lumen material draw has invalid buffers");
        if (!ValidateLumenSurfaceDraw(draw,vertices->second.Description,indices->second.Description,indices->second.Bytes,LastError)) return false;
        for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture)
        { const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Lumen material texture is invalid"); }
    }
    auto record=std::make_shared<LumenSurfaceRecord>(); record->Layout=layout; record->Identity=upload.Identity; record->Captures=upload.Cards;
    for (auto& image:record->Images)
    {
        image=CreateNativeTexture(layout.Width,layout.Height,PixelFormatRGBA32Float,TextureUsageRenderTarget|TextureUsageShaderRead,"Lumen material capture");
        if (!image) return false;
    }
    record->Depth=CreateNativeTexture(layout.Width,layout.Height,PixelFormatDepth32Float,TextureUsageRenderTarget,"Lumen capture depth"); if (!record->Depth) return false;
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen capture command buffer is unavailable");
    const auto pass=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor"); const auto colors=Send<ObjCObject>(pass,"colorAttachments");
    for (NSUInteger plane=0;plane<LumenSurfacePlanes;++plane)
    {
        const auto attachment=Send<ObjCObject,NSUInteger>(colors,"objectAtIndexedSubscript:",plane);
        Send<void,ObjCObject>(attachment,"setTexture:",record->Images[plane]); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear);
        Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore); const MTLClearColor clear{0,0,0,plane?1.0:0.0}; Send<void,MTLClearColor>(attachment,"setClearColor:",clear);
    }
    const auto depth=Send<ObjCObject>(pass,"depthAttachment"); Send<void,ObjCObject>(depth,"setTexture:",record->Depth);
    Send<void,NSUInteger>(depth,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore); Send<void,double>(depth,"setClearDepth:",1);
    const auto encoder=Send<ObjCObject,ObjCObject>(command,"renderCommandEncoderWithDescriptor:",pass); if (!encoder) return Fail("Metal Lumen material encoder is unavailable");
    struct EndEncoder { ObjCObject Encoder; ~EndEncoder(){if(Encoder) Send<void>(Encoder,"endEncoding");} } end{encoder};
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",LumenCapturePipeline); Send<void,ObjCObject>(encoder,"setDepthStencilState:",LumenCaptureDepth);
    Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill); Send<void,float,float,float>(encoder,"setDepthBias:slopeScale:clamp:",0,0,0);
    Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentSamplerState:atIndex:",SurfaceSampler,0);
    const unsigned stride=layout.CardSide+LumenSurfaceBorder*2; LumenSurfaceResult result; result.Layout=layout; result.Identity=upload.Identity;
    for (unsigned card=0;card<upload.Cards.size();++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=(card/layout.Columns)*stride;
        const MTLViewport viewport{double(x),double(y),double(stride),double(stride),0,1}; const LumenCaptureScissor scissor{x,y,stride,stride};
        Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,LumenCaptureScissor>(encoder,"setScissorRect:",scissor);
        const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; LumenSurfaceConstants constants; if (!BuildLumenSurfaceConstants(layout,capture,draw,constants,LastError)) return false;
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&constants,sizeof(constants),1);
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),1);
            Send<void,NSUInteger>(encoder,"setCullMode:",draw.BackFaceCulling?CullModeBack:CullModeNone);
            Send<void,NSUInteger>(encoder,"setFrontFacingWinding:",RealtimeWorldIsMirrored(draw.World)?WindingClockwise:WindingCounterClockwise);
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",Buffers.at(draw.Vertices.Value).Buffer,0,0);
            const TextureHandle textures[]{draw.Diffuse,draw.Normal,draw.Opacity};
            for (NSUInteger slot=0;slot<LumenSurfacePlanes;++slot)
                Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",textures[slot]?Textures.at(textures[slot].Value).Texture:nullptr,slot);
            Send<void,NSUInteger,NSUInteger,NSUInteger,ObjCObject,NSUInteger>(encoder,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
                PrimitiveTypeTriangle,draw.IndexCount,IndexTypeUInt32,Buffers.at(draw.Indices.Value).Buffer,NSUInteger(draw.FirstIndex)*sizeof(std::uint32_t)); ++result.DrawCalls;
        }
    }
    Send<void>(encoder,"endEncoding"); end.Encoder=nullptr;
    struct Feedback { ObjCObject Buffer=nullptr; ~Feedback(){ReleaseObject(Buffer);} } feedback;
    if (upload.Readback && !EncodeLumenRead(command,*record,feedback.Buffer)) return false;
    if (!CompleteLumenCapture(command,*record,feedback.Buffer,upload.Readback?&result.Texels:nullptr)) return false;
    result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space is exhausted");
    LumenSurfaces.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Lumen surface handle is invalid or expired");
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen export command buffer is unavailable");
    struct Feedback { ObjCObject Buffer=nullptr; ~Feedback(){ReleaseObject(Buffer);} } feedback;
    return EncodeLumenRead(command,*found->second,feedback.Buffer) && CompleteLumenCapture(command,*found->second,feedback.Buffer,&output);
}
bool DestroyLumenSurface(LumenSurfaceHandle handle)
{ return (handle && LumenSurfaces.erase(handle.Value)) || Fail("Lumen surface handle is invalid or already released"); }
