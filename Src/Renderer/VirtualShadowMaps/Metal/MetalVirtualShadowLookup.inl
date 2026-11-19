// The frame retains the immutable atlas until subsequent native work can safely
// replace its bindings. Releasing a public handle cannot invalidate encoded draws.
std::shared_ptr<VirtualShadowAtlasRecord> FrameVirtualShadowAtlas;
VirtualShadowSamplingFrame FrameVirtualShadowSampling;
bool PrepareVirtualShadowSampling(const VirtualShadowSamplingFrame& frame,const CascadedShadowState& cascades)
{
    if (!ValidateVirtualShadowSamplingFrame(frame,LastError)) return false;
    std::shared_ptr<VirtualShadowAtlasRecord> candidate;
    if (frame.Atlas)
    {
        if (cascades.Frame.CascadeCount) return Fail("Virtual and cascaded shadow representations are mutually exclusive");
        if (FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::Hard && FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::PCF)
            return Fail("Virtual scene sampling currently requires hard or PCF filtering");
        const auto found=VirtualShadowAtlases.find(frame.Atlas.Value);
        if (found==VirtualShadowAtlases.end()) return Fail("Virtual scene sampling requires a current atlas handle");
        candidate=found->second;
    }
    FrameVirtualShadowAtlas=std::move(candidate); FrameVirtualShadowSampling=frame; return true;
}
ObjCObject VirtualShadowLookupPipeline=nullptr;
void DestroyVirtualShadowLookup() { FrameVirtualShadowAtlas.reset(); FrameVirtualShadowSampling={}; ReleaseObject(VirtualShadowLookupPipeline); VirtualShadowLookupPipeline=nullptr; }
bool CreateVirtualShadowDirectory(VirtualShadowAtlasRecord& record)
{
    VirtualShadowDirectory directory; if (!BuildVirtualShadowDirectory(record.Info,directory,LastError)) return false;
    const auto descriptor=Send<ObjCObject,NSUInteger,NSUInteger,NSUInteger,ObjCBool>(GetClass("MTLTextureDescriptor"),
        "texture2DDescriptorWithPixelFormat:width:height:mipmapped:",PixelFormatDepth32Float,directory.Width,directory.Height,ObjCNo);
    Send<void,NSUInteger>(descriptor,"setStorageMode:",StorageModePrivate); Send<void,NSUInteger>(descriptor,"setUsage:",TextureUsageShaderRead);
    record.Directory=Send<ObjCObject,ObjCObject>(Device,"newTextureWithDescriptor:",descriptor);
    return (record.Directory || Fail("Metal virtual directory allocation failed")) &&
        UploadTexture(record.Directory,int(directory.Width),int(directory.Height),sizeof(float),directory.Depth.data());
}
bool SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("Virtual lookup atlas handle is invalid");
    const auto record=found->second; VirtualShadowLookupConstants constants;
    if (!BuildVirtualShadowLookupConstants(record->Info,upload,constants,LastError)) return false;
    if (!VirtualShadowLookupPipeline)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","VirtualShadowMaps.metal",source,LastError)) return false;
        ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr; Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        const auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error); ReleaseObject(options);
        if (!library) return Fail("Metal virtual lookup compilation failed: "+ErrorDescription(error));
        const auto function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("virtualShadowLookup"));
        if (function) VirtualShadowLookupPipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function); ReleaseObject(library);
        if (!VirtualShadowLookupPipeline) return Fail("Metal virtual lookup pipeline is unavailable: "+ErrorDescription(error));
        if (Send<NSUInteger>(VirtualShadowLookupPipeline,"maxTotalThreadsPerThreadgroup")<VirtualShadowGroupWidth)
        { DestroyVirtualShadowLookup(); return Fail("Metal virtual lookup has insufficient thread capacity"); }
    }
    struct Work { ObjCObject Queries=nullptr,Results=nullptr; ~Work(){ReleaseObject(Queries); ReleaseObject(Results);} } work;
    const auto bytes=upload.Samples.size()*sizeof(Float4);
    work.Queries=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
    work.Results=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
    if (!work.Queries || !work.Results || !Send<void*>(work.Queries,"contents") || !Send<void*>(work.Results,"contents")) return Fail("Metal virtual lookup storage is unavailable");
    std::memcpy(Send<void*>(work.Queries,"contents"),upload.Samples.data(),bytes);
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal virtual lookup commands are unavailable");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal virtual lookup encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",VirtualShadowLookupPipeline);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Queries,0,1); Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Results,0,2);
    Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",record->Depth,0); Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",record->Directory,1);
    const auto groups=(constants.Filtering.Z+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
    const MTLSize grid{std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1},threads{VirtualShadowGroupWidth,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted"); if (!CommandSucceeded(command,"Metal virtual lookup failed")) return false;
    VirtualShadowLookupResult result; result.Samples.resize(upload.Samples.size()); std::memcpy(result.Samples.data(),Send<void*>(work.Results,"contents"),bytes);
    if (!ValidateVirtualShadowLookupResult(record->Info,upload,result,LastError)) return false; output=std::move(result); return true;
}
