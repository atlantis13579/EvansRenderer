NaniteFrameRasterWork NaniteRasterPlan;
std::shared_ptr<NaniteFrameBuffer> NaniteRasterDecisions;
NaniteDepthHandle NaniteRasterMainDepth;
NaniteVisibilityHandle NaniteRasterPost;
bool PrepareNaniteRasterVisibility(const VisibilityBufferDesc& frame)
{
    // Previous frame resources are retired only after the frame fence/abort.
    NaniteDepths.erase(NaniteRasterMainDepth.Value); NaniteDecisions.erase(NaniteRasterPost.Value);
    NaniteRasterMainDepth={}; NaniteRasterPost={}; NaniteRasterDecisions.reset();
    if (!NaniteFrameDummy) { const unsigned zero=0; if (!CreateNaniteFrameBuffer(1,&zero,false,NaniteFrameDummy)) return false; }
    NaniteRasterDecisions=NaniteFrameDummy;
    if (!frame.NaniteVisibility) return true;
    const auto found=NaniteDecisions.find(frame.NaniteVisibility.Value);
    if (found==NaniteDecisions.end() || found->second->Info.Published || found->second->Info.Phase!=NaniteVisibilityPhase::Main ||
        found->second->Info.Frame!=FrameNanite.CurrentFrame() || found->second->Info.QueryCount!=frame.NaniteClusters.size() || found->second->Draws)
        return Fail("Nanite raster requires matching unconsumed main-phase cluster decisions");
    NaniteRasterDecisions=found->second->Data; return true;
}
bool RetestNaniteRaster(const VisibilityBufferDesc& frame)
{
    if (!CaptureNaniteDepth(NaniteRasterMainDepth,true) || !RetestNaniteVisibility(frame.NaniteVisibility,NaniteRasterMainDepth,NaniteRasterPost,true)) return false;
    NaniteRasterDecisions=NaniteDecisions.at(NaniteRasterPost.Value)->Data;
    for (auto& job:NaniteRasterPlan.Dispatches) { job.Constants.Raster.Quality.Z=unsigned(NaniteVisibilityState::PostVisible); job.Constants.Raster.Quality.W|=NaniteRasterPreserveMain; }
    return true;
}
std::shared_ptr<NaniteFrameBuffer> NaniteRasterInput,NaniteRasterOutput;
ObjCObject NaniteRasterFrameLibrary=nullptr,NaniteRasterExportDepth=nullptr;
std::array<ObjCObject,6> NaniteRasterFramePipelines{};
std::array<ObjCObject,2> NaniteRasterGraphics{};
ObjCObject NaniteRasterPostDepth=nullptr;
void DestroyNaniteFrameRaster()
{
    DestroyNaniteHardware();
    NaniteRasterDecisions.reset(); NaniteRasterMainDepth={}; NaniteRasterPost={};
    NaniteRasterInput.reset(); NaniteRasterOutput.reset(); NaniteRasterPlan={};
    for (auto& p:NaniteRasterFramePipelines) { ReleaseObject(p); p=nullptr; }
    for (auto& p:NaniteRasterGraphics) { ReleaseObject(p); p=nullptr; }
    ReleaseObject(NaniteRasterPostDepth); NaniteRasterPostDepth=nullptr;
    ReleaseObject(NaniteRasterFrameLibrary); ReleaseObject(NaniteRasterExportDepth); NaniteRasterFrameLibrary=NaniteRasterExportDepth=nullptr;
}
bool PrepareNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (!PrepareNaniteRasterVisibility(frame)) return false;
    NaniteFrameRasterWork plan; if (!BuildNaniteFrameRasterWork(frame,CreateInfo.Width,CreateInfo.Height,plan,LastError)) return false;
    if (!NaniteRasterFrameLibrary)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","NaniteRaster.metal",source,LastError)) return false;
        auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        ObjCObject error=nullptr;
        NaniteRasterFrameLibrary=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options); if (!NaniteRasterFrameLibrary) return Fail("Compile Metal Nanite frame raster: "+ErrorDescription(error));
    }
    constexpr const char* entries[]{"EvansNaniteSeedDepth","EvansNaniteRaster","EvansNaniteSeedOwner","EvansNaniteClearInputs","EvansNaniteProjectInputs","EvansNaniteCaptureInputs"};
    for (unsigned i=0;i<NaniteRasterFramePipelines.size();++i) if (!NaniteRasterFramePipelines[i])
    {
        auto function=Send<ObjCObject,ObjCObject>(NaniteRasterFrameLibrary,"newFunctionWithName:",MakeString(entries[i])); ObjCObject error=nullptr;
        if (function) NaniteRasterFramePipelines[i]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!NaniteRasterFramePipelines[i] || Send<NSUInteger>(NaniteRasterFramePipelines[i],"maxTotalThreadsPerThreadgroup")<NaniteRasterThreads)
            return Fail("Create Metal Nanite frame raster kernel: "+ErrorDescription(error));
    }
    if (!NaniteRasterGraphics[0]) NaniteRasterGraphics[0]=CreateTechniquePipeline("NaniteRaster.metal","EvansNaniteHardwareVertex","EvansNaniteHardwareFragment",1,
        {VisibilityIntegerFormat,PixelFormatRG32Float},true,ColorWriteMaskAll,false,NaniteRasterFrameLibrary);
    if (!NaniteRasterGraphics[1]) NaniteRasterGraphics[1]=CreateTechniquePipeline("NaniteRaster.metal","EvansNaniteExportVertex","EvansNaniteExportFragment",1,
        {VisibilityIntegerFormat,PixelFormatRG32Float},true,ColorWriteMaskAll,false,NaniteRasterFrameLibrary);
    if (!NaniteRasterExportDepth) NaniteRasterExportDepth=CreateDepthState(CompareFunctionAlways,true);
    if (!NaniteRasterPostDepth) NaniteRasterPostDepth=CreateDepthState(CompareFunctionLessEqual,true);
    if (!NaniteRasterGraphics[0] || !NaniteRasterGraphics[1] || !NaniteRasterExportDepth || !NaniteRasterPostDepth) return false;
    std::shared_ptr<NaniteFrameBuffer> input,output;
    if (!CreateNaniteFrameBuffer(plan.InputWords,nullptr,true,input) || !CreateNaniteFrameBuffer(plan.OutputWords,nullptr,true,output)) return false;
    NaniteRasterInput=std::move(input); NaniteRasterOutput=std::move(output); NaniteRasterPlan=std::move(plan); return true;
}
bool DispatchNaniteRasterFrame(const NaniteFrameRasterDispatch& job)
{
    auto encoder=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!encoder) return Fail("Create Metal Nanite raster encoder");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",NaniteRasterFramePipelines[job.Pass]);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&job.Constants,
        job.Pass<3?sizeof(NaniteRasterConstants):sizeof(NaniteFrameRasterConstants),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",job.Pass<3?NaniteRasterInput->Buffer:VisibilityGeometry,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",job.Pass<3?NaniteRasterOutput->Buffer:NaniteRasterInput->Buffer,0,2);
    if (job.Pass==4) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",NaniteRasterDecisions?NaniteRasterDecisions->Buffer:NaniteRasterInput->Buffer,0,3);
    if (job.Pass==5)
    { Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",DepthTexture,0); Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",VisibilityImages[0],1); }
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{job.GroupsX,job.GroupsY,1},MTLSize{NaniteRasterThreads,1,1});
    Send<void>(encoder,"endEncoding"); ++PendingFrame.Statistics.NaniteRasterDispatchCount; return true;
}
bool DrawNaniteRasterPhase(const VisibilityBufferDesc& frame)
{
    if (frame.NaniteVisibility) ++PendingFrame.Statistics.NaniteOcclusionPhases;
    const auto begin=[&](bool exporting)
    {
        auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
        auto colors=Send<ObjCObject>(descriptor,"colorAttachments");
        for (NSUInteger i=0;i<2;++i)
        {
            auto attachment=Send<ObjCObject,NSUInteger>(colors,"objectAtIndexedSubscript:",i);
            Send<void,ObjCObject>(attachment,"setTexture:",VisibilityImages[i]); Send<void,NSUInteger>(attachment,"setLoadAction:",NaniteRasterPost?LoadActionLoad:LoadActionClear);
            Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore); Send<void,MTLClearColor>(attachment,"setClearColor:",MTLClearColor{0,0,0,0});
        }
        auto depth=Send<ObjCObject>(descriptor,"depthAttachment"); Send<void,ObjCObject>(depth,"setTexture:",DepthTexture);
        Send<void,NSUInteger>(depth,"setLoadAction:",LoadActionLoad); Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore);
        RenderEncoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
        if (!RenderEncoder) return Fail("Create Metal Nanite visibility encoder");
        Send<void,MTLViewport>(RenderEncoder,"setViewport:",MTLViewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1});
        Send<void,ObjCObject>(RenderEncoder,"setRenderPipelineState:",NaniteRasterGraphics[exporting?1:0]);
        Send<void,ObjCObject>(RenderEncoder,"setDepthStencilState:",exporting?NaniteRasterExportDepth:NaniteRasterPost?NaniteRasterPostDepth:DepthWriteState);
        Send<void,NSUInteger>(RenderEncoder,"setTriangleFillMode:",TriangleFillModeFill);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setVertexBuffer:offset:atIndex:",NaniteRasterInput->Buffer,0,2);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setVertexBuffer:offset:atIndex:",NaniteRasterOutput->Buffer,0,3);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setFragmentBuffer:offset:atIndex:",VisibilityGeometry,0,1);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setFragmentBuffer:offset:atIndex:",NaniteRasterInput->Buffer,0,2);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setFragmentBuffer:offset:atIndex:",NaniteRasterOutput->Buffer,0,3);
        Send<void,ObjCObject,NSUInteger>(RenderEncoder,"setFragmentSamplerState:atIndex:",SurfaceSampler,0); return true;
    };
    const auto bind=[&](const VisibilityMaterialDraw* draw)
    {
        NaniteRasterDrawConstants constants;
        if (!BuildNaniteRasterDrawConstants(frame,draw,CreateInfo.Width,CreateInfo.Height,constants)) return Fail("Invalid Metal Nanite raster constants");
        if (NaniteRasterPost) constants.Raster.Quality.W|=NaniteRasterPreserveMain;
        Send<void,const void*,NSUInteger,NSUInteger>(RenderEncoder,"setVertexBytes:length:atIndex:",&constants,sizeof(constants),0);
        Send<void,const void*,NSUInteger,NSUInteger>(RenderEncoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
        ObjCObject images[]{DummyColorTexture,DummyColorTexture};
        if (draw)
        {
            const TextureHandle handles[]{draw->Pipeline.DiffuseTexture,draw->Pipeline.OpacityTexture};
            for (unsigned i=0;i<2;++i) if (handles[i])
            { const auto found=Textures.find(handles[i].Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Invalid Metal Nanite material image"); images[i]=found->second.Texture; }
        }
        for (NSUInteger i=0;i<2;++i) Send<void,ObjCObject,NSUInteger>(RenderEncoder,"setFragmentTexture:atIndex:",images[i],i);
        return true;
    };
    for (unsigned i=0;i<NaniteRasterPlan.HardwareBoundary;++i) if (!DispatchNaniteRasterFrame(NaniteRasterPlan.Dispatches[i])) return false;
    if (frame.GpuHardware)
    { if (!DrawNaniteHardware(frame)) return false; }
    else
    {
        if (!begin(false)) return false;
        for (std::size_t order=0;order<frame.Draws.size();++order)
        {
            const auto& draw=frame.Draws[NaniteRasterPost?frame.Draws.size()-1-order:order];
            if (!bind(&draw)) return false; const auto& range=frame.Scene->Draws[draw.Range];
            Send<void,NSUInteger>(RenderEncoder,"setCullMode:",draw.Pipeline.BackFaceCulling?CullModeBack:CullModeNone);
            Send<void,NSUInteger>(RenderEncoder,"setFrontFacingWinding:",range.Mirrored?WindingClockwise:WindingCounterClockwise);
            Send<void,NSUInteger,NSUInteger,NSUInteger>(RenderEncoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,range.TriangleCount*VisibilityTriangleCorners);
        }
        Send<void>(RenderEncoder,"endEncoding"); RenderEncoder=nullptr;
    }
    for (unsigned i=NaniteRasterPlan.HardwareBoundary;i<NaniteRasterPlan.Dispatches.size();++i) if (!DispatchNaniteRasterFrame(NaniteRasterPlan.Dispatches[i])) return false;
    if (!begin(true) || !bind(nullptr)) return false;
    Send<void,NSUInteger>(RenderEncoder,"setCullMode:",CullModeNone);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(RenderEncoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
    Send<void>(RenderEncoder,"endEncoding"); RenderEncoder=nullptr; ++PendingFrame.Statistics.NaniteRasterExportCount; return true;
}

bool DrawNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (frame.Scene->HasNativeGeometry() && !(frame.GpuInstances?DrawNaniteInstances():DrawNaniteVisibilityGeometry())) return false;
    PendingFrame.Statistics.NaniteOcclusionQueries+=unsigned(frame.NaniteClusters.size());
    if (!DrawNaniteRasterPhase(frame)) return false;
    if (frame.NaniteVisibility)
    {
        if (!RetestNaniteRaster(frame)) return false;
        if (!DrawNaniteRasterPhase(frame)) return false;
    }
    return true;
}
