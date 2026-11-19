NaniteHardwareWork HardwarePlan;
std::shared_ptr<NaniteFrameBuffer> HardwareRanges,HardwareArguments,HardwareQueue,HardwareOwners;
ObjCObject HardwareLibrary=nullptr;
std::array<ObjCObject,4> HardwareKernels{};
std::array<ObjCObject,3> HardwareGraphics{},HardwareDepth{};
void DestroyNaniteHardware()
{
    HardwarePlan={}; HardwareRanges.reset(); HardwareArguments.reset(); HardwareQueue.reset(); HardwareOwners.reset();
    for (auto item:HardwareKernels) ReleaseObject(item); HardwareKernels={};
    for (auto item:HardwareGraphics) ReleaseObject(item); HardwareGraphics={};
    for (auto item:HardwareDepth) ReleaseObject(item); HardwareDepth={}; ReleaseObject(HardwareLibrary); HardwareLibrary=nullptr;
}
bool PrepareNaniteHardware(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteHardwareWork(frame,CreateInfo.Width,CreateInfo.Height,HardwarePlan,LastError)) return false;
    if (!HardwarePlan.Active()) return true;
    if (!CreateNaniteFrameBuffer(unsigned(HardwarePlan.Ranges.size()*sizeof(ShaderUInt4)/sizeof(unsigned)),HardwarePlan.Ranges.data(),false,HardwareRanges) ||
        !CreateNaniteFrameBuffer(HardwarePlan.ArgumentWords,nullptr,true,HardwareArguments) || !CreateNaniteFrameBuffer(HardwarePlan.QueueWords,nullptr,true,HardwareQueue) ||
        !CreateNaniteFrameBuffer(HardwarePlan.Constants.Counts.W,nullptr,true,HardwareOwners)) return false;
    if (!HardwareLibrary)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","NaniteHardware.metal",source,LastError)) return false;
        auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo); ObjCObject error=nullptr;
        HardwareLibrary=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options); if (!HardwareLibrary) return Fail("Compile Metal hardware scheduling: "+ErrorDescription(error));
    }
    constexpr const char* kernels[]{"EvansHardwareClear","EvansHardwareCount","EvansHardwareScan","EvansHardwareScatter"};
    for (unsigned pass=0;pass<HardwareKernels.size();++pass) if (!HardwareKernels[pass])
    {
        auto function=Send<ObjCObject,ObjCObject>(HardwareLibrary,"newFunctionWithName:",MakeString(kernels[pass])); ObjCObject error=nullptr;
        if (function) HardwareKernels[pass]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error); ReleaseObject(function);
        if (!HardwareKernels[pass] || Send<NSUInteger>(HardwareKernels[pass],"maxTotalThreadsPerThreadgroup")<(pass==2?1:NaniteHardwareThreads))
            return Fail("Create Metal hardware queue kernel: "+ErrorDescription(error));
    }
    constexpr const char* pixels[]{"EvansHardwareDepth","EvansHardwareOwner","EvansHardwareResolve"};
    for (unsigned phase=0;phase<HardwareGraphics.size();++phase)
    {
        if (!HardwareGraphics[phase]) HardwareGraphics[phase]=CreateTechniquePipeline("NaniteHardware.metal",phase==2?"EvansHardwareResolveVertex":"EvansHardwareQueueVertex",pixels[phase],1,
            {VisibilityIntegerFormat,PixelFormatRG32Float},true,phase==2?ColorWriteMaskAll:0,false,HardwareLibrary);
        if (!HardwareDepth[phase]) HardwareDepth[phase]=CreateDepthState(phase==0?CompareFunctionLess:phase==1?CompareFunctionEqual:CompareFunctionAlways,phase==0);
        if (!HardwareGraphics[phase] || !HardwareDepth[phase]) return false;
    }
    return true;
}
bool DrawNaniteHardware(const VisibilityBufferDesc& frame)
{
    if (!HardwarePlan.Active()) return true;
    for (unsigned pass=0;pass<HardwareKernels.size();++pass)
    {
        auto encoder=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!encoder) return Fail("Create Metal hardware scheduling encoder");
        Send<void,ObjCObject>(encoder,"setComputePipelineState:",HardwareKernels[pass]);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&HardwarePlan.Constants,sizeof(NaniteHardwareConstants),0);
        const ObjCObject buffers[]{HardwareRanges->Buffer,NaniteRasterOutput->Buffer,HardwareArguments->Buffer,HardwareQueue->Buffer,HardwareOwners->Buffer};
        for (NSUInteger i=0;i<5;++i) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",buffers[i],0,i+1);
        const auto& c=HardwarePlan.Constants; const unsigned work=pass==0?std::max(c.Counts.Z,c.Counts.W):c.Counts.X,groups=(work+NaniteHardwareThreads-1)/NaniteHardwareThreads;
        Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",
            MTLSize{pass==2?1:std::min(groups,NaniteHardwareColumns),pass==2?1:(groups+NaniteHardwareColumns-1)/NaniteHardwareColumns,1},MTLSize{pass==2?1:NaniteHardwareThreads,1,1});
        Send<void>(encoder,"endEncoding");
    }
    // Tracked encoder boundaries publish queue/argument storage, then depth,
    // then atomic ownership. Only the final resolve writes visibility colors.
    for (unsigned phase=0;phase<HardwareGraphics.size();++phase)
    {
        auto descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor"); auto colors=Send<ObjCObject>(descriptor,"colorAttachments");
        for (NSUInteger i=0;i<2;++i)
        {
            auto attachment=Send<ObjCObject,NSUInteger>(colors,"objectAtIndexedSubscript:",i); Send<void,ObjCObject>(attachment,"setTexture:",VisibilityImages[i]);
            Send<void,NSUInteger>(attachment,"setLoadAction:",phase==0&&!NaniteRasterPost?LoadActionClear:LoadActionLoad);
            Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore); Send<void,MTLClearColor>(attachment,"setClearColor:",MTLClearColor{0,0,0,0});
        }
        auto depth=Send<ObjCObject>(descriptor,"depthAttachment"); Send<void,ObjCObject>(depth,"setTexture:",DepthTexture);
        Send<void,NSUInteger>(depth,"setLoadAction:",LoadActionLoad); Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore);
        auto encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor); if (!encoder) return Fail("Create Metal hardware ownership encoder");
        Send<void,MTLViewport>(encoder,"setViewport:",MTLViewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1});
        Send<void,ObjCObject>(encoder,"setRenderPipelineState:",frame.MeshShaders && phase<2?MeshPipelines[phase+1]:HardwareGraphics[phase]); Send<void,ObjCObject>(encoder,"setDepthStencilState:",HardwareDepth[phase]);
        Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
        const ObjCObject buffers[]{VisibilityGeometry,NaniteRasterInput->Buffer,HardwareQueue->Buffer,HardwareOwners->Buffer};
        for (NSUInteger i=0;i<4;++i)
        {
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",buffers[i],0,i+1);
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",buffers[i],0,i+1);
            if (frame.MeshShaders && phase<2) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setMeshBuffer:offset:atIndex:",buffers[i],0,i+1);
        }
        if (frame.MeshShaders && phase<2)
        {
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setMeshBuffer:offset:atIndex:",HardwareArguments->Buffer,0,5);
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setObjectBuffer:offset:atIndex:",HardwareArguments->Buffer,0,5);
        }
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentSamplerState:atIndex:",SurfaceSampler,0);
        const unsigned count=phase==2?1:unsigned(HardwarePlan.Bins.size());
        for (unsigned bin=0;bin<count;++bin)
        {
            const auto& item=HardwarePlan.Bins[bin];
            if (frame.MeshShaders && phase<2)
            {
                Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setMeshBytes:length:atIndex:",&item.Constants,sizeof(item.Constants),0);
                Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setObjectBytes:length:atIndex:",&item.Constants,sizeof(item.Constants),0);
            }
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&item.Constants,sizeof(item.Constants),0);
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&item.Constants,sizeof(item.Constants),0);
            Send<void,NSUInteger>(encoder,"setCullMode:",phase!=2 && item.Cull?CullModeBack:CullModeNone);
            Send<void,NSUInteger>(encoder,"setFrontFacingWinding:",item.Mirrored?WindingClockwise:WindingCounterClockwise);
            const TextureHandle handles[]{item.Diffuse,item.Opacity};
            for (NSUInteger i=0;i<2;++i)
            {
                ObjCObject texture=DummyColorTexture;
                if (handles[i])
                {
                    const auto found=Textures.find(handles[i].Value);
                    if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8)
                    { Send<void>(encoder,"endEncoding"); return Fail("Invalid Metal hardware coverage texture"); }
                    texture=found->second.Texture;
                }
                Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",texture,i);
            }
            if (phase==2) Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
            else if (frame.MeshShaders) DrawNaniteMeshBin(encoder);
            else Send<void,NSUInteger,ObjCObject,NSUInteger>(encoder,"drawPrimitives:indirectBuffer:indirectBufferOffset:",PrimitiveTypeTriangle,HardwareArguments->Buffer,bin*NaniteHardwareBinWords*sizeof(unsigned));
        }
        Send<void>(encoder,"endEncoding");
    }
    auto& stats=PendingFrame.Statistics; stats.NaniteHardwareBinCount+=unsigned(HardwarePlan.Bins.size()); stats.NaniteHardwareDispatchCount+=unsigned(HardwareKernels.size());
    const auto dispatches=unsigned(HardwarePlan.Bins.size()*2);
    if (frame.MeshShaders)
    {
        stats.MeshShaderDispatchCount+=dispatches;
        stats.MeshShaderObjectDispatchCount+=dispatches;
    }
    if (!frame.MeshShaders) { stats.NaniteHardwareIndirectCount+=dispatches; stats.IndirectDrawCallCount+=dispatches; }
    return true;
}
