// Shared tracked buffers are immutable history versions. Completion and numeric
// validation gate publication; abandoning a candidate never changes its parent.
struct TSRNativeHistory
{
    TSRHistoryInfo Info; std::array<ObjCObject,2> Buffers{};
    ObjCObject Color=nullptr;
    ~TSRNativeHistory() { ReleaseObject(Color); for (auto buffer:Buffers) ReleaseObject(buffer); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<TSRNativeHistory>> TSRHistories;
TSRSceneState TSRScene;
ObjCObject TSROpaque=nullptr;
ObjCObject TSRPipeline=nullptr;
ObjCObject TSRPackPipeline=nullptr;
ObjCObject TSRExportPipeline=nullptr;
void DestroyTSR() { TSRScene.Reset(); ReleaseObject(TSROpaque); TSROpaque=nullptr; TSRHistories.clear(); ReleaseObject(TSRPipeline); ReleaseObject(TSRPackPipeline); ReleaseObject(TSRExportPipeline); TSRPipeline=TSRPackPipeline=TSRExportPipeline=nullptr; }
bool DestroyTSRHistory(TSRHistoryHandle handle)
{ return (handle && TSRHistories.erase(handle.Value)) || Fail("Metal TSR history handle is invalid or belongs to another RHI"); }
bool ExportTSR(TSRNativeHistory& record,TSRReadback& result,bool readback=true)
{
    if (!TSRExportPipeline)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","TSR.metal",source,LastError)) return false;
        ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr; Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        const auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error); ReleaseObject(options);
        if (!library) return Fail("Metal TSR export compilation failed: "+ErrorDescription(error));
        const auto vertex=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("tsrExportVertex"));
        const auto fragment=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("tsrExportFragment"));
        if (!vertex || !fragment) { ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library); return Fail("Metal TSR export entry points are missing"); }
        const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
        Send<void,ObjCObject>(descriptor,"setVertexFunction:",vertex); Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
        const auto attachments=Send<ObjCObject>(descriptor,"colorAttachments"),attachment=Send<ObjCObject,NSUInteger>(attachments,"objectAtIndexedSubscript:",0);
        Send<void,NSUInteger>(attachment,"setPixelFormat:",PixelFormatRGBA16Float); Send<void,NSUInteger>(attachment,"setWriteMask:",ColorWriteMaskAll); Send<void,ObjCBool>(attachment,"setBlendingEnabled:",ObjCNo);
        TSRExportPipeline=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
        ReleaseObject(descriptor); ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
        if (!TSRExportPipeline) return Fail("Metal TSR export pipeline failed: "+ErrorDescription(error));
    }
    record.Color=CreateNativeTexture(record.Info.Width,record.Info.Height,PixelFormatRGBA16Float,TextureUsageShaderRead|TextureUsageRenderTarget,"Evans TSR immutable HDR export");
    if (!record.Color) return false;
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(record.Info.Width,record.Info.Height,0,constants,LastError)) return false;
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal TSR export command is unavailable");
    const auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor"),attachments=Send<ObjCObject>(descriptor,"colorAttachments"),attachment=Send<ObjCObject,NSUInteger>(attachments,"objectAtIndexedSubscript:",0);
    Send<void,ObjCObject>(attachment,"setTexture:",record.Color); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore);
    const auto encoder=Send<ObjCObject,ObjCObject>(command,"renderCommandEncoderWithDescriptor:",descriptor); if (!encoder) return Fail("Metal TSR export encoder is unavailable");
    const MTLViewport viewport{0,0,double(record.Info.Width),double(record.Info.Height),0,1}; Send<void,MTLViewport>(encoder,"setViewport:",viewport);
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",TSRExportPipeline); Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",record.Buffers[0],0,1);
    Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); Send<void>(encoder,"endEncoding");
    if(!readback){Send<void>(command,"commit");Send<void>(command,"waitUntilCompleted");return CommandSucceeded(command,"Metal TSR HDR export");}
    const auto blit=Send<ObjCObject>(command,"blitCommandEncoder"); if (!blit) return Fail("Metal TSR HDR readback encoder is unavailable");
    ObjCObject readbackBuffer=nullptr; std::size_t rowPitch=0;
    const bool enqueued=EnqueueReadback(blit,record.Color,record.Info.Width,record.Info.Height,4*sizeof(std::uint16_t),readbackBuffer,rowPitch); Send<void>(blit,"endEncoding");
    if (!enqueued) { ReleaseObject(readbackBuffer); return false; }
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (!CommandSucceeded(command,"Metal TSR HDR export")) { ReleaseObject(readbackBuffer); return false; }
    const auto* start=static_cast<const std::uint8_t*>(Send<void*>(readbackBuffer,"contents")); if (!start) { ReleaseObject(readbackBuffer); return Fail("Metal TSR HDR readback is unavailable"); }
    result.PresentationHDR.resize(std::size_t(record.Info.Width)*record.Info.Height);
    for (unsigned y=0;y<record.Info.Height;++y)
    {
        const auto* row=reinterpret_cast<const std::uint16_t*>(start+std::size_t(y)*rowPitch);
        for (unsigned x=0;x<record.Info.Width;++x) result.PresentationHDR[std::size_t(y)*record.Info.Width+x]=DecodeTSRHalfPixel(row+x*4);
    }
    ReleaseObject(readbackBuffer); return true;
}
struct TSRNativeImages { std::array<ObjCObject,TSRSceneImageCount> Images{}; std::uint32_t Flags=0; };
bool PackTSRImages(ObjCObject command,const TSRFrame& frame,const TSRNativeImages& images,ObjCObject packet)
{
    TSRTextureConstants constants; if (!BuildTSRTextureConstants(frame.InputWidth,frame.InputHeight,images.Flags,constants,LastError)) return false;
    for (unsigned image=0;image<TSRSceneImageCount;++image) if (TSRImageRequired(image,images.Flags))
    {
        const auto texture=images.Images[image];
        if (!texture || Send<ObjCObject>(texture,"device")!=Device || Send<NSUInteger>(texture,"width")!=frame.InputWidth || Send<NSUInteger>(texture,"height")!=frame.InputHeight ||
            Send<NSUInteger>(texture,"sampleCount")!=1 || Send<NSUInteger>(texture,"textureType")!=TextureType2D)
            return Fail("Metal TSR requires resolved render-sized images on this device");
    }
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal TSR image pack encoder failed");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",TSRPackPipeline); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",packet,0,1);
    for (NSUInteger image=0;image<TSRSceneImageCount;++image) Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",TSRImageRequired(unsigned(image),images.Flags)?images.Images[image]:images.Images[TSRSceneFinal],image);
    const MTLSize groups{(frame.InputWidth+TSRGroupSide-1)/TSRGroupSide,(frame.InputHeight+TSRGroupSide-1)/TSRGroupSide,1},threads{TSRGroupSide,TSRGroupSide,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",groups,threads); Send<void>(encoder,"endEncoding"); return true;
}
bool ReconstructTSRImages(const TSRFrame& frame,const TSRNativeImages& images,TSRHistoryHandle previous,TSRResult& output)
{
    std::shared_ptr<TSRNativeHistory> old;
    if (previous) { const auto found=TSRHistories.find(previous.Value); if (found==TSRHistories.end()) return Fail("Metal previous TSR history is invalid"); old=found->second; }
    TSRConstants constants; if (!BuildTSRConstants(frame,old?&old->Info:nullptr,constants,LastError)) return false;
    if (!TSRPipeline || !TSRPackPipeline)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","TSR.metal",source,LastError)) return false;
        ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr; Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error); ReleaseObject(options);
        if (!library) return Fail("Metal TSR compilation failed: "+ErrorDescription(error));
        ReleaseObject(TSRPipeline); ReleaseObject(TSRPackPipeline); TSRPipeline=TSRPackPipeline=nullptr;
        ObjCObject function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("tsrReconstruct"));
        if (function) TSRPipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function); function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("tsrPack"));
        if (function) TSRPackPipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function); ReleaseObject(library);
        if (!TSRPipeline || !TSRPackPipeline) return Fail("Metal TSR pipeline failed: "+ErrorDescription(error));
        if (Send<NSUInteger>(TSRPipeline,"maxTotalThreadsPerThreadgroup")<TSRGroupSide*TSRGroupSide || Send<NSUInteger>(TSRPackPipeline,"maxTotalThreadsPerThreadgroup")<TSRGroupSide*TSRGroupSide) { DestroyTSR(); return Fail("Metal TSR workgroup exceeds device capacity"); }
    }
    struct Work { std::array<ObjCObject,2> Buffers{}; ~Work(){for (auto buffer:Buffers) ReleaseObject(buffer);} } work;
    const std::array<Float4,TSRHistoryVectors> empty{}; const void* data[]{nullptr,empty.data()}; const std::size_t sizes[]{std::size_t(frame.InputWidth)*frame.InputHeight*(TSRInputVectors*sizeof(Float4)),sizeof(empty)};
    for (unsigned slot=0;slot<work.Buffers.size();++slot)
    {
        const NSUInteger options=!frame.CaptureDiagnostics && slot==0 ? StorageModePrivate<<4 : 0;
        work.Buffers[slot]=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",sizes[slot],options);
        if(!work.Buffers[slot])return Fail("Metal TSR source allocation failed");
        if(data[slot])
        {
            void* mapped=Send<void*>(work.Buffers[slot],"contents");if(!mapped)return Fail("Metal TSR constants are not host-visible");
            std::memcpy(mapped,data[slot],sizes[slot]);
        }
    }
    auto record=std::make_shared<TSRNativeHistory>(); record->Info=TSRHistoryIdentity(frame); const auto bytes=std::size_t(record->Info.Width)*record->Info.Height*TSRHistoryVectors*sizeof(Float4);
    for (auto& buffer:record->Buffers)
    { buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,frame.CaptureDiagnostics?0:StorageModePrivate<<4); if (!buffer || (frame.CaptureDiagnostics && !Send<void*>(buffer,"contents"))) return Fail("Metal TSR output allocation failed"); }
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal TSR command buffer failed");
    if (!PackTSRImages(command,frame,images,work.Buffers[0])) return false;
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal TSR encoder failed");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",TSRPipeline); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),0);
    const ObjCObject bindings[]{work.Buffers[0],constants.Options.X?old->Buffers[0]:work.Buffers[1],record->Buffers[0],record->Buffers[1]};
    for (NSUInteger slot=0;slot<4;++slot) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",bindings[slot],0,slot+1);
    const MTLSize groups{(constants.Extent.Z+TSRGroupSide-1)/TSRGroupSide,(constants.Extent.W+TSRGroupSide-1)/TSRGroupSide,1},threads{TSRGroupSide,TSRGroupSide,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",groups,threads); Send<void>(encoder,"endEncoding");
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal TSR command failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    TSRResult candidate; candidate.HistoryUsed=constants.Options.X!=0; candidate.Info=record->Info; if(frame.CaptureDiagnostics){candidate.Readback.History.resize(bytes/sizeof(Float4)); candidate.Readback.Auxiliary.resize(bytes/sizeof(Float4));
    std::memcpy(candidate.Readback.History.data(),Send<void*>(record->Buffers[0],"contents"),bytes); std::memcpy(candidate.Readback.Auxiliary.data(),Send<void*>(record->Buffers[1],"contents"),bytes);
}
    if (!ExportTSR(*record,candidate.Readback,frame.CaptureDiagnostics) || (frame.CaptureDiagnostics && !ValidateTSRReadback(frame,candidate.Readback,LastError))) return false;
    candidate.Handle=NewTSRHistoryHandle(); if (!candidate.Handle) return Fail("TSR history handle space is exhausted");
    TSRHistories.emplace(candidate.Handle.Value,std::move(record)); output=std::move(candidate); return true;
}
bool PrepareTSRScene(const TSRSceneFrame& requested,const Techniques::PostProcessSettings& post)
{
    const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value);
    const auto found=TSRHistories.find(TSRScene.Committed.Value);
    return TSRScene.Begin(requested,CreateInfo.Width,CreateInfo.Height,post,found==TSRHistories.end()?nullptr:&found->second->Info,LastError);
}
void AbortTSRScene() { const auto discarded=TSRScene.Abort(); if (discarded) TSRHistories.erase(discarded.Value); }
void CommitTSRScene() { const auto old=TSRScene.Commit(); if (old) TSRHistories.erase(old.Value); }
bool RenderTSRDisplay(TSRHistoryHandle handle,const Techniques::PostProcessSettings& requested,FrameCapture& output)
{
    const auto found=TSRHistories.find(handle.Value); if (found==TSRHistories.end()) return Fail("Metal TSR display history is invalid");
    Techniques::PostProcessSettings settings; if (!BuildTSRDisplaySettings(requested,settings,LastError)) return false;
    const auto history=found->second;
    struct Targets
    {
        Impl& Owner; int Width=0,Height=0; FrameCapture Capture; ObjCObject Color=nullptr,Commands=nullptr,Readback=nullptr;
        std::array<ObjCObject,4> HDR{}; std::array<ObjCObject,2> Post{};
        explicit Targets(Impl& owner):Owner(owner) {}
        ~Targets() { ReleaseObject(Color); ReleaseObject(Readback); for (auto image:HDR) ReleaseObject(image); for (auto image:Post) ReleaseObject(image); }
        void Swap()
        {
            std::swap(Width,Owner.CreateInfo.Width); std::swap(Height,Owner.CreateInfo.Height); std::swap(Capture,Owner.PendingFrame); std::swap(Color,Owner.ColorTexture); std::swap(Commands,Owner.CommandBuffer); HDR.swap(Owner.HdrTextures); Post.swap(Owner.PostTextures);
        }
    } targets(*this);
    targets.Width=history->Info.Width; targets.Height=history->Info.Height;
    targets.Color=CreateNativeTexture(targets.Width,targets.Height,PixelFormatRGBA8UnormSRGB,TextureUsageShaderRead|TextureUsageRenderTarget,"Evans TSR display output");
    targets.Commands=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!targets.Color || !targets.Commands) return Fail("Metal TSR display allocation failed");
    {
        struct Restore { Targets& Data; explicit Restore(Targets& data):Data(data){Data.Swap();} ~Restore(){Data.Swap();} } restore(targets);
        if (!EnsureHdrTargets()) return false;
        auto blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal TSR display source copy failed");
        const MTLOrigin origin{0,0,0}; const MTLSize size{history->Info.Width,history->Info.Height,1};
        Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",history->Color,0,0,origin,size,HdrTextures[0],0,0,origin); Send<void>(blit,"endEncoding");
        ObjCObject color=ColorTexture; if (!ApplyHdr(settings.HDR) || !ApplyPostProcess(settings,color)) return false;
        blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal TSR display readback encoder failed");
        std::size_t pitch=0; const bool enqueued=TSRScene.Active && !TSRScene.Frame.CaptureDiagnostics ? PublishGpuColor(blit,color) : EnqueueReadback(blit,color,CreateInfo.Width,CreateInfo.Height,4,targets.Readback,pitch); Send<void>(blit,"endEncoding"); if (!enqueued) return false;
        Send<void>(CommandBuffer,"commit"); Send<void>(CommandBuffer,"waitUntilCompleted"); if (!CommandSucceeded(CommandBuffer,"Metal TSR display")) return false;
        if(!(TSRScene.Active && !TSRScene.Frame.CaptureDiagnostics)){
        const auto rowBytes=std::size_t(CreateInfo.Width)*4; PendingFrame.ColorRGBA8.resize(rowBytes*CreateInfo.Height);
        const auto* bytes=static_cast<const std::uint8_t*>(Send<void*>(targets.Readback,"contents")); if (!bytes) return Fail("Metal TSR display readback is unavailable");
        for (int row=0;row<CreateInfo.Height;++row) std::memcpy(PendingFrame.ColorRGBA8.data()+std::size_t(row)*rowBytes,bytes+std::size_t(row)*pitch,rowBytes);
        }
        PendingFrame.Width=history->Info.Width; PendingFrame.Height=history->Info.Height; PendingFrame.Statistics.PassCount=PendingFrame.Statistics.PostProcessPassCount;
    }
    output=std::move(targets.Capture); return true;
}
bool SnapshotTSROpaque()
{
    if (!TSRScene.Active || TSRScene.OpaqueReady || !OpaqueEffectsComplete() || !(DeferredActive?DeferredResolved:ForwardColorWritten) || (FrameSampleCount>1 && !SamplesOpaqueResolved)) return Fail("TSR snapshot requires completed opaque effects exactly once");
    if (!TSROpaque || Send<NSUInteger>(TSROpaque,"width")!=NSUInteger(CreateInfo.Width) || Send<NSUInteger>(TSROpaque,"height")!=NSUInteger(CreateInfo.Height))
    { ReleaseObject(TSROpaque); TSROpaque=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,TextureUsageShaderRead,"Evans TSR opaque snapshot"); if (!TSROpaque) return false; }
    const auto blit=Send<ObjCObject>(CommandBuffer,"blitCommandEncoder"); if (!blit) return Fail("Metal TSR opaque snapshot encoder failed");
    const MTLOrigin origin{0,0,0}; const MTLSize size{NSUInteger(CreateInfo.Width),NSUInteger(CreateInfo.Height),1};
    Send<void,ObjCObject,NSUInteger,NSUInteger,MTLOrigin,MTLSize,ObjCObject,NSUInteger,NSUInteger,MTLOrigin>(blit,
        "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",HdrTextures[0],0,0,origin,size,TSROpaque,0,0,origin); Send<void>(blit,"endEncoding"); TSRScene.OpaqueReady=true; return true;
}
bool FinishTSRScene()
{
    if (!TSRScene.Active) return true;
    TSRScene.MotionReady=VelocityCaptured && (FrameSampleCount==1 || SamplesVelocityResolved);
    if (!TSRScene.OpaqueReady || !TSRScene.MotionReady) return Fail("TSR presentation requires opaque snapshot and resolved physical motion");
    struct Restore { bool& Value; explicit Restore(bool& value):Value(value){Value=false;} ~Restore(){Value=true;} } restore(FrameOpen);
    TSRNativeImages images; images.Flags=TSRTextureMotion; images.Images[TSRSceneFinal]=HdrTextures[0]; images.Images[TSRSceneOpaque]=TSROpaque; images.Images[TSRSceneMotion]=MotionTextures[MotionVelocity]; images.Images[TSRSceneDepth]=DepthTexture;
    TSRResult candidate; if (!ReconstructTSRImages(TSRScene.Frame,images,TSRScene.Committed,candidate)) return false; TSRScene.Pending=candidate.Handle;
    FrameCapture display; return RenderTSRDisplay(candidate.Handle,FramePostProcessSettings,display) && ComposeTSRSceneCapture(TSRScene.Frame,candidate,std::move(display),PendingFrame,LastError);
}
