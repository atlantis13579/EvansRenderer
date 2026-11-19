// Default tracked Metal buffers synchronize compute writes with the subsequent
// render encoder. Each batch owns its compacted constants and indirect command;
// Present waits for completion before BeginFrame rewinds the batch pool.
struct GpuBatch
{
    GpuDrivenBatchDesc Description;
    ObjCObject Input = nullptr, Output = nullptr, Arguments = nullptr;
};
std::vector<GpuBatch> GpuBatches;
GpuBatch* ActiveGpuBatch = nullptr;
std::size_t GpuCursor = 0;
std::uint64_t GpuEpoch = 0;
ObjCObject GpuCullPipeline = nullptr;

void DestroyGpuBatch(GpuBatch& batch)
{
    ReleaseObject(batch.Input); ReleaseObject(batch.Output); ReleaseObject(batch.Arguments);
    batch = {};
}
void DestroyGpuDriven()
{
    ActiveGpuBatch = nullptr; GpuCursor = 0;
    for (auto& batch : GpuBatches) DestroyGpuBatch(batch);
    GpuBatches.clear(); ReleaseObject(GpuCullPipeline); GpuCullPipeline = nullptr;
}
bool EnsureGpuCull()
{
    if (GpuCullPipeline) return true;
    std::string source;
    if (!LoadTechniqueShaderSource("Metal", "GPUDrivenRendering.metal", source, LastError)) return false;
    ObjCObject options = Send<ObjCObject>(GetClass("MTLCompileOptions"), "new");
    Send<void, ObjCBool>(options, "setFastMathEnabled:", ObjCNo);
    ObjCObject error = nullptr;
    ObjCObject library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options);
    if (!library) return Fail("Metal GPU visibility compilation failed: " + ErrorDescription(error));
    ObjCObject function = Send<ObjCObject, ObjCObject>(library, "newFunctionWithName:", MakeString("EvansGpuDrivenCull"));
    if (function) GpuCullPipeline = Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
    ReleaseObject(function); ReleaseObject(library);
    if (!GpuCullPipeline) return Fail("Metal GPU visibility pipeline failed: " + ErrorDescription(error));
    if (Send<NSUInteger>(GpuCullPipeline, "maxTotalThreadsPerThreadgroup") < MaximumDrawInstances)
    { ReleaseObject(GpuCullPipeline); GpuCullPipeline = nullptr; return Fail("Metal GPU visibility group exceeds device capacity"); }
    return true;
}
bool PrepareGpuBatch(const GpuDrivenBatchDesc& desc, std::uint32_t& index)
{
    GpuDrivenBatchUpload upload;
    if (!BuildGpuDrivenBatch(desc, upload)) return Fail("Metal GPU batch contains invalid bounds, transforms or command data");
    if (!CommandBuffer || RenderEncoder) return Fail("Metal GPU visibility requires a command buffer outside a render encoder");
    if (!EnsureGpuCull()) return false;
    if (GpuCursor == GpuBatches.size())
    {
        GpuBatch batch;
        batch.Input = Send<ObjCObject, NSUInteger, NSUInteger>(Device, "newBufferWithLength:options:", sizeof(upload.Candidates), 0);
        batch.Output = Send<ObjCObject, NSUInteger, NSUInteger>(Device, "newBufferWithLength:options:", sizeof(InstanceConstantBlock), 0);
        batch.Arguments = Send<ObjCObject, NSUInteger, NSUInteger>(Device, "newBufferWithLength:options:", sizeof(GpuDrivenIndexedArguments), 0);
        if (!batch.Input || !batch.Output || !batch.Arguments)
        { DestroyGpuBatch(batch); return Fail("Metal GPU batch allocation failed"); }
        try { GpuBatches.push_back(std::move(batch)); }
        catch (...) { DestroyGpuBatch(batch); throw; }
    }
    auto& batch = GpuBatches[GpuCursor]; batch.Description = desc;
    void* mapped = Send<void*>(batch.Input, "contents");
    if (!mapped) return Fail("Metal GPU candidate upload is not host-visible");
    std::memcpy(mapped, upload.Candidates.data(), sizeof(upload.Candidates));
    ObjCObject encoder = Send<ObjCObject>(CommandBuffer, "computeCommandEncoder");
    if (!encoder) return Fail("Metal GPU visibility encoder creation failed");
    Send<void, ObjCObject>(encoder, "setComputePipelineState:", GpuCullPipeline);
    Send<void, const void*, NSUInteger, NSUInteger>(encoder, "setBytes:length:atIndex:", &upload.Constants, sizeof(upload.Constants), 0);
    Send<void, ObjCObject, NSUInteger, NSUInteger>(encoder, "setBuffer:offset:atIndex:", batch.Input, 0, 1);
    Send<void, ObjCObject, NSUInteger, NSUInteger>(encoder, "setBuffer:offset:atIndex:", batch.Output, 0, 2);
    Send<void, ObjCObject, NSUInteger, NSUInteger>(encoder, "setBuffer:offset:atIndex:", batch.Arguments, 0, 3);
    const MTLSize groups {1, 1, 1}, threads {MaximumDrawInstances, 1, 1};
    Send<void, MTLSize, MTLSize>(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", groups, threads);
    Send<void>(encoder, "endEncoding");
    index = static_cast<std::uint32_t>(GpuCursor++); return true;
}
