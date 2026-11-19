std::array<ObjCObject,2> VirtualShadowRequestPipelines{};
void DestroyVirtualShadowRequests()
{ for (auto& pipeline:VirtualShadowRequestPipelines) { ReleaseObject(pipeline); pipeline=nullptr; } }
bool BuildVirtualShadowRequests(const VirtualShadowRequestUpload& upload,VirtualShadowRequestResult& output,ObjCObject nativeDepth=nullptr)
{
    if (!nativeDepth && !ValidateVirtualShadowRequestUpload(upload,LastError)) return false;
    VirtualShadowRequestConstants constants; if (!BuildVirtualShadowRequestConstants(upload.Frame,constants,LastError)) return false;
    if (!VirtualShadowRequestPipelines[0] || !VirtualShadowRequestPipelines[1])
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","VirtualShadowMaps.metal",source,LastError)) return false;
        ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr;
        Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        const auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error); ReleaseObject(options);
        if (!library) return Fail("Metal virtual shadow requests failed compilation: "+ErrorDescription(error));
        constexpr const char* names[]{"virtualShadowClear","virtualShadowMark"};
        for (unsigned stage=0;stage<VirtualShadowRequestPipelines.size();++stage) if (!VirtualShadowRequestPipelines[stage])
        {
            const auto function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString(names[stage]));
            if (function) VirtualShadowRequestPipelines[stage]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
            ReleaseObject(function);
        }
        ReleaseObject(library);
        for (const auto pipeline:VirtualShadowRequestPipelines)
            if (!pipeline || Send<NSUInteger>(pipeline,"maxTotalThreadsPerThreadgroup")<VirtualShadowGroupWidth)
            { DestroyVirtualShadowRequests(); return Fail("Metal virtual shadow request pipeline is unavailable or has insufficient threads: "+ErrorDescription(error)); }
    }
    struct Work { std::array<ObjCObject,2> Buffers{}; ~Work(){for (auto buffer:Buffers) ReleaseObject(buffer);} } work;
    const std::size_t sizes[]{std::size_t(upload.Frame.Width)*upload.Frame.Height*sizeof(float),std::size_t(constants.Limits.Y)*sizeof(std::uint32_t)};
    for (unsigned slot=0;slot<work.Buffers.size();++slot)
    {
        if (!slot && nativeDepth) { work.Buffers[slot]=Send<ObjCObject>(nativeDepth,"retain"); continue; }
        work.Buffers[slot]=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",sizes[slot],0);
        if (!work.Buffers[slot] || !Send<void*>(work.Buffers[slot],"contents")) return Fail("Metal virtual shadow request buffer is not host-visible");
    }
    if (!nativeDepth) std::memcpy(Send<void*>(work.Buffers[0],"contents"),upload.Depth.data(),sizes[0]);
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal virtual shadow request command buffer is unavailable");
    const unsigned counts[]{constants.Limits.Y,constants.Extent.X*constants.Extent.Y};
    for (unsigned stage=0;stage<VirtualShadowRequestPipelines.size();++stage)
    {
        const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal virtual shadow request encoder is unavailable");
        Send<void,ObjCObject>(encoder,"setComputePipelineState:",VirtualShadowRequestPipelines[stage]);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),0);
        for (NSUInteger slot=0;slot<work.Buffers.size();++slot) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Buffers[slot],0,slot+1);
        const auto groups=(counts[stage]+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
        const MTLSize grid{std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1},threads{VirtualShadowGroupWidth,1,1};
        Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    }
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal virtual shadow request dispatch failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    VirtualShadowRequestResult result; result.Pages.resize(constants.Limits.Y); std::memcpy(result.Pages.data(),Send<void*>(work.Buffers[1],"contents"),sizes[1]);
    VirtualShadowLayout layout; if (!BuildVirtualShadowLayout(upload.Frame.Settings,layout,LastError) || !ValidateVirtualShadowRequests(layout,result.Pages,LastError)) return false;
    output=std::move(result); return true;
}
