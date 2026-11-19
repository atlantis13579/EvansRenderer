struct LumenCaptureWork : LumenCaptureJob
{
    std::shared_ptr<LumenSurfaceRecord> Record; ObjCObject Completion=nullptr;
    ~LumenCaptureWork() { ReleaseObject(Completion); }
};
LumenCaptureQueue<LumenCaptureWork> LumenCaptureTasks;
bool CollectLumenCapture()
{ return LumenCaptureTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenCapture(work,ready); }); }
bool CancelLumenCaptureUpdate(LumenCaptureTask task)
{ return (LumenCaptureTasks.Cancel(task) || Fail("Invalid Lumen capture task")) && CollectLumenCapture(); }
template<class Handle> bool RetireLumenCaptureInput(Handle handle)
{
    if (!handle || LumenCaptureTasks.Empty()) return true;
    // Source deletion/mutation is exceptional. Drain only submissions borrowing
    // this resource; ordinary frame advances never wait for their capture work.
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    if (!LumenCaptureTasks.Visit([&](auto& work)
    {
        if (!work.Uses(handle)) return true;
        if (FrameOpen) return Fail("Lumen capture inputs cannot change inside a render frame");
        work.Cancelled=true;
        while (work.InFlight)
        {
            bool ready=false; if (!ProbeLumenCapture(work,ready)) return false;
            if (ready) { work.InFlight=false; break; }
            if (std::chrono::steady_clock::now()>=deadline) return Fail("Retiring Lumen capture input timed out");
            std::this_thread::yield();
        }
        return true;
    })) return false;
    return CollectLumenCapture();
}
void DestroyLumenCaptureAsync()
{
    LumenCaptureTasks.Visit([](auto& work) { if (work.InFlight) Send<void>(work.Completion,"waitUntilCompleted"); return true; });
    LumenCaptureTasks.Clear();
}
bool ProbeLumenCapture(LumenCaptureWork& work,bool& ready)
{
    const auto status=Send<NSUInteger>(work.Completion,"status");
    if (status==CommandBufferStatusError) return Fail("Metal capture slice failed: "+ErrorDescription(Send<ObjCObject>(work.Completion,"error")));
    constexpr NSUInteger CompletedStatus=4; ready=status==CompletedStatus; return true;
}
bool BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& output)
{
    if (!CollectLumenCapture() || !EnsureLumenSurface()) return false;
    auto job=std::make_shared<LumenCaptureWork>();
    if (!job->Prepare(upload,LastError) || !LumenCaptureTasks.CanSubmit(*job,LastError)) return false;
    const auto& layout=job->Layout;
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
    job->Record=record;
    const auto task=LumenCaptureTasks.Insert(job); if (!task) return Fail("Lumen capture task space exhausted"); output=task; return true;
}
bool SubmitLumenCaptureSlice(LumenCaptureWork& work,const LumenCaptureJob::SliceFrame& slice)
{
    const auto& upload=work.Upload; const auto& layout=work.Layout; const auto& record=work.Record;
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen capture command buffer is unavailable");
    const auto pass=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor"); const auto colors=Send<ObjCObject>(pass,"colorAttachments");
    for (NSUInteger plane=0;plane<LumenSurfacePlanes;++plane)
    {
        const auto attachment=Send<ObjCObject,NSUInteger>(colors,"objectAtIndexedSubscript:",plane);
        Send<void,ObjCObject>(attachment,"setTexture:",record->Images[plane]); Send<void,NSUInteger>(attachment,"setLoadAction:",work.Initialized?LoadActionLoad:LoadActionClear);
        Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore); const MTLClearColor clear{0,0,0,plane?1.0:0.0}; Send<void,MTLClearColor>(attachment,"setClearColor:",clear);
    }
    const auto depth=Send<ObjCObject>(pass,"depthAttachment"); Send<void,ObjCObject>(depth,"setTexture:",record->Depth);
    Send<void,NSUInteger>(depth,"setLoadAction:",work.Initialized?LoadActionLoad:LoadActionClear); Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore); Send<void,double>(depth,"setClearDepth:",1);
    const auto encoder=Send<ObjCObject,ObjCObject>(command,"renderCommandEncoderWithDescriptor:",pass); if (!encoder) return Fail("Metal Lumen material encoder is unavailable");
    struct EndEncoder { ObjCObject Encoder; ~EndEncoder(){if(Encoder) Send<void>(Encoder,"endEncoding");} } end{encoder};
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",LumenCapturePipeline); Send<void,ObjCObject>(encoder,"setDepthStencilState:",LumenCaptureDepth);
    Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill); Send<void,float,float,float>(encoder,"setDepthBias:slopeScale:clamp:",0,0,0);
    Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentSamplerState:atIndex:",SurfaceSampler,0);
    const unsigned stride=layout.CardSide+LumenSurfaceBorder*2;
    for (unsigned card=slice.Dispatch.X;card<slice.Dispatch.X+slice.Dispatch.Y;++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=(card/layout.Columns)*stride;
        const MTLViewport viewport{double(x),double(y),double(stride),double(stride),0,1}; const LumenCaptureScissor scissor{x,y,stride,stride};
        Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,LumenCaptureScissor>(encoder,"setScissorRect:",scissor);
        const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; const auto& constants=work.Constants[work.FirstConstant[card]+index-capture.FirstDraw];
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&constants,sizeof(constants),1);
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),1);
            Send<void,NSUInteger>(encoder,"setCullMode:",draw.BackFaceCulling?CullModeBack:CullModeNone);
            Send<void,NSUInteger>(encoder,"setFrontFacingWinding:",RealtimeWorldIsMirrored(draw.World)?WindingClockwise:WindingCounterClockwise);
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",Buffers.at(draw.Vertices.Value).Buffer,0,0);
            const TextureHandle textures[]{draw.Diffuse,draw.Normal,draw.Opacity};
            for (NSUInteger slot=0;slot<LumenSurfacePlanes;++slot)
                Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",textures[slot]?Textures.at(textures[slot].Value).Texture:nullptr,slot);
            Send<void,NSUInteger,NSUInteger,NSUInteger,ObjCObject,NSUInteger>(encoder,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
                PrimitiveTypeTriangle,draw.IndexCount,IndexTypeUInt32,Buffers.at(draw.Indices.Value).Buffer,NSUInteger(draw.FirstIndex)*sizeof(std::uint32_t));
        }
    }
    Send<void>(encoder,"endEncoding"); end.Encoder=nullptr;
    ReleaseObject(work.Completion); work.Completion=Send<ObjCObject>(command,"retain"); work.InFlight=true; Send<void>(command,"commit"); return true;
}
bool AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    if (!CollectLumenCapture()) return false;
    return LumenCaptureTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenCapture(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenCaptureSlice(work,frame); },
        [&](auto& work,LumenSurfaceResult& result)
        {
            result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space exhausted");
            result.Layout=work.Layout; result.Identity=work.Upload.Identity; result.DrawCalls=work.Statistics.DrawCalls;
            LumenSurfaces.emplace(result.Handle.Value,work.Record); return true;
        });
}
