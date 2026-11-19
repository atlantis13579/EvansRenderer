// Shared hazard-tracked Metal buffers keep native output and host mirrors coherent.
// Encoder boundaries order compute phases; command completion gates both allocation
// from prefix counts and publication. Interval projection requires fast math off.
std::array<ObjCObject,5> NaniteGeometryPipelines{};
void DestroyNaniteGeometry()
{ DestroyNaniteAsync(); DestroyNaniteCacheAsync(); DestroyNaniteCaches(); for (auto& pipeline:NaniteGeometryPipelines) { ReleaseObject(pipeline); pipeline=nullptr; } }
bool EnsureNaniteGeometry()
{
    if (std::all_of(NaniteGeometryPipelines.begin(),NaniteGeometryPipelines.end(),[](auto pipeline){return pipeline!=nullptr;})) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","NaniteGeometry.metal",source,LastError)) return false;
    ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr; Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options); if (!library) return Fail("Metal Nanite geometry compilation failed: "+ErrorDescription(error));
    constexpr const char* entries[]{"EvansNaniteEvaluate","EvansNaniteRefine","EvansNaniteClusters","EvansNaniteBlocks","EvansNaniteDecode"};
    for (std::size_t stage=0;stage<NaniteGeometryPipelines.size();++stage) if (!NaniteGeometryPipelines[stage])
    {
        ObjCObject function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString(entries[stage]));
        if (function) NaniteGeometryPipelines[stage]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!NaniteGeometryPipelines[stage]) { ReleaseObject(library); return Fail("Metal Nanite geometry pipeline failed: "+ErrorDescription(error)); }
        if (Send<NSUInteger>(NaniteGeometryPipelines[stage],"maxTotalThreadsPerThreadgroup")<NaniteGpuWidth)
        { ReleaseObject(library); DestroyNaniteGeometry(); return Fail("Metal Nanite geometry workgroup exceeds device capacity"); }
    }
    ReleaseObject(library); return true;
}
bool CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output,NaniteCacheHandle cache={})
{
    std::vector<NaniteGpuDispatch> jobs;
    if (!BuildNaniteGpuSchedule(upload,jobs,LastError) || !EnsureNaniteGeometry()) return false;
    const NaniteNativeCache* cached=nullptr;
    if (cache) { const auto found=NaniteCaches.find(cache.Value); if (found==NaniteCaches.end()) return Fail("Metal Nanite cache handle is invalid"); cached=&found->second; if (!MatchNaniteCache(cached->Info,upload,LastError)) return false; }
    const std::size_t stateBytes=upload.InitialState.size()*sizeof(std::uint32_t);
    const void* inputs[]{upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data(),nullptr,nullptr};
    const std::size_t sizes[]{upload.Metadata.size()*sizeof(std::uint32_t),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage),stateBytes,sizeof(std::uint32_t),sizeof(std::uint32_t)};
    struct Work { std::array<ObjCObject,6> Buffers{}; ~Work(){for (auto buffer:Buffers) ReleaseObject(buffer);} } work;
    auto allocate=[&](std::size_t slot,std::size_t bytes,const void* data)
    {
        ReleaseObject(work.Buffers[slot]); work.Buffers[slot]=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,static_cast<NSUInteger>(0));
        if (!work.Buffers[slot]) return Fail("Metal Nanite geometry allocation failed");
        void* mapped=Send<void*>(work.Buffers[slot],"contents"); if (!mapped) return Fail("Metal Nanite geometry allocation is not host-visible");
        if (data) std::memcpy(mapped,data,bytes); return true;
    };
    for (std::size_t slot=0;slot<work.Buffers.size();++slot)
        if (slot==2 && cached) work.Buffers[slot]=Send<ObjCObject>(cached->Pool->Buffer,"retain"); else if (!allocate(slot,sizes[slot],inputs[slot])) return false;
    auto encode=[&](ObjCObject command,const NaniteGpuDispatch& job,const NaniteGpuConstants& constants)
    {
        ObjCObject encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Nanite geometry encoder failed");
        Send<void,ObjCObject>(encoder,"setComputePipelineState:",NaniteGeometryPipelines[static_cast<unsigned>(job.Pass)]);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),0);
        for (NSUInteger slot=0;slot<work.Buffers.size();++slot) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Buffers[slot],0,slot+1);
        const MTLSize groups{job.GroupsX,job.GroupsY,1},threads{NaniteGpuWidth,1,1};
        Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",groups,threads); Send<void>(encoder,"endEncoding"); return true;
    };
    auto submit=[&](ObjCObject command)
    {
        Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
        return Send<NSUInteger>(command,"status")!=CommandBufferStatusError || Fail("Metal Nanite geometry command failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    };
    ObjCObject command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Nanite geometry command buffer failed");
    for (std::size_t stage=0;stage+1<jobs.size();++stage) if (!encode(command,jobs[stage],jobs[stage].Constants)) return false;
    if (!submit(command)) return false;
    NaniteGeometryBuffers candidate; candidate.Readback.State.resize(upload.InitialState.size());
    std::memcpy(candidate.Readback.State.data(),Send<void*>(work.Buffers[3],"contents"),stateBytes);
    if (!ValidateNaniteGpuState(upload,candidate.Readback.State,candidate.Selection,LastError)) return false;
    const std::size_t vertexBytes=std::size_t(candidate.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(candidate.Selection.IndexCount)*sizeof(std::uint32_t);
    if (!allocate(4,vertexBytes,nullptr) || !allocate(5,indexBytes,nullptr)) return false;
    auto constants=jobs.back().Constants; constants.Capacity.X=candidate.Selection.VertexCount; constants.Capacity.Y=candidate.Selection.IndexCount;
    command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Nanite decode command buffer failed");
    if (!encode(command,jobs.back(),constants) || !submit(command)) return false;
    candidate.Readback.Vertices.resize(candidate.Selection.VertexCount); candidate.Readback.Indices.resize(candidate.Selection.IndexCount);
    std::memcpy(candidate.Readback.State.data(),Send<void*>(work.Buffers[3],"contents"),stateBytes);
    std::memcpy(candidate.Readback.Vertices.data(),Send<void*>(work.Buffers[4],"contents"),vertexBytes); std::memcpy(candidate.Readback.Indices.data(),Send<void*>(work.Buffers[5],"contents"),indexBytes);
    if (!ValidateNaniteGpuReadback(upload,candidate.Readback,candidate.Selection,LastError)) return false;
    std::array<BufferResource,2> resources; const void* mirrors[]{candidate.Readback.Vertices.data(),candidate.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite geometry vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite geometry indices"};
    for (std::size_t stream=0;stream<resources.size();++stream)
    { auto& resource=resources[stream]; resource.NaniteGeometry=true; resource.Buffer=work.Buffers[4+stream]; resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[stream],resource.Bytes.size()); }
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(jobs.size());
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    work.Buffers[4]=work.Buffers[5]=nullptr; output=std::move(candidate); return true;
}
