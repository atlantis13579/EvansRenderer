// Retained shared buffers and command buffers form an immutable production job.
// Encoder boundaries publish scan dependencies. Host reads are deferred until
// command status reports completion; ordinary polls never waitUntilCompleted.
static constexpr NSUInteger NaniteCommandCompleted=4;
struct NaniteAsyncWork
{
    NaniteGpuUpload Upload; NaniteAsyncPlan Plan;
    std::array<ObjCObject,6> Buffers{}; ObjCObject Completion=nullptr;
    ~NaniteAsyncWork() { ReleaseObject(Completion); for (auto buffer:Buffers) ReleaseObject(buffer); }
};
NaniteAsyncQueue<NaniteAsyncWork> NaniteAsyncTasks;
void DestroyNaniteAsync()
{
    NaniteAsyncTasks.Visit([](auto& work){Send<void>(work.Completion,"waitUntilCompleted"); return true;});
    NaniteAsyncTasks.Clear();
}
bool ProbeNaniteAsync(NaniteAsyncWork& work,bool& ready)
{
    const auto status=Send<NSUInteger>(work.Completion,"status");
    if (status==CommandBufferStatusError) return Fail("Metal Nanite production failed: "+ErrorDescription(Send<ObjCObject>(work.Completion,"error")));
    ready=status==NaniteCommandCompleted; return true;
}
bool CollectNaniteAsync() { return NaniteAsyncTasks.Collect([&](auto& work,bool& ready){return ProbeNaniteAsync(work,ready);}); }
bool CancelNaniteGeometry(NaniteGeometryTask task)
{ return (NaniteAsyncTasks.Cancel(task) || Fail("Invalid Nanite task")) && CollectNaniteAsync(); }
bool SubmitNaniteGeometry(NaniteCacheHandle cache,const NaniteGpuUpload& upload,NaniteGeometryTask& output)
{
    auto work=std::make_shared<NaniteAsyncWork>();
    if (!CollectNaniteAsync() || !BuildNaniteAsyncPlan(upload,work->Plan,LastError) || !NaniteAsyncTasks.CanSubmit(work->Plan,LastError) || !EnsureNaniteGeometry()) return false;
    const NaniteNativeCache* cached=nullptr;
    if (cache) { auto found=NaniteCaches.find(cache.Value); if (found==NaniteCaches.end()) return Fail("Invalid Nanite cache"); cached=&found->second; if (!MatchNaniteCache(cached->Info,upload,LastError)) return false; }
    work->Upload=upload;
    const void* data[]{upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data(),nullptr,nullptr};
    const std::size_t sizes[]{upload.Metadata.size()*sizeof(unsigned),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage),upload.InitialState.size()*sizeof(unsigned),work->Plan.VertexBytes,work->Plan.IndexBytes};
    for (unsigned i=0;i<work->Buffers.size();++i)
    {
        if (i==2 && cached) { work->Buffers[i]=Send<ObjCObject>(cached->Pool->Buffer,"retain"); continue; }
        work->Buffers[i]=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",sizes[i],i>=4 && !upload.ReadbackGeometry?(StorageModePrivate<<ResourceStorageModeShift):static_cast<NSUInteger>(0));
        if (!work->Buffers[i]) return Fail("Allocate Metal async Nanite buffer failed");
        if (data[i])
        {
            void* mapped=Send<void*>(work->Buffers[i],"contents"); if (!mapped) return Fail("Metal async Nanite input is not shared");
            std::memcpy(mapped,data[i],sizes[i]);
        }
    }
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Create Metal async Nanite command failed");
    work->Completion=Send<ObjCObject>(command,"retain");
    for (const auto& job:work->Plan.Dispatches)
    {
        const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Create Metal async Nanite encoder failed");
        Send<void,ObjCObject>(encoder,"setComputePipelineState:",NaniteGeometryPipelines[unsigned(job.Pass)]);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&job.Constants,sizeof(job.Constants),0);
        for (NSUInteger i=0;i<work->Buffers.size();++i) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work->Buffers[i],0,i+1);
        Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{job.GroupsX,job.GroupsY,1},MTLSize{NaniteGpuWidth,1,1}); Send<void>(encoder,"endEncoding");
    }
    const auto task=NaniteAsyncTasks.Insert(work); Send<void>(command,"commit"); output=task; return true;
}
bool PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    ready=false; auto work=NaniteAsyncTasks.Find(task); if (!work) return Fail("Invalid Nanite task");
    bool complete=false; if (!ProbeNaniteAsync(*work,complete)) return false; if (!complete) return true;
    NaniteGeometryBuffers result; result.Readback.State.resize(work->Upload.InitialState.size());
    std::memcpy(result.Readback.State.data(),Send<void*>(work->Buffers[3],"contents"),result.Readback.State.size()*sizeof(unsigned));
    if (!ValidateNaniteGpuState(work->Upload,result.Readback.State,result.Selection,LastError)) return false;
    const auto vertexBytes=std::size_t(result.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(result.Selection.IndexCount)*sizeof(unsigned);
    if (work->Upload.ReadbackGeometry)
    {
        result.Readback.Vertices.resize(result.Selection.VertexCount); result.Readback.Indices.resize(result.Selection.IndexCount);
        std::memcpy(result.Readback.Vertices.data(),Send<void*>(work->Buffers[4],"contents"),vertexBytes); std::memcpy(result.Readback.Indices.data(),Send<void*>(work->Buffers[5],"contents"),indexBytes);
        if (!ValidateNaniteGpuReadback(work->Upload,result.Readback,result.Selection,LastError)) return false;
    }
    std::array<BufferResource,2> resources; const void* mirrors[]{result.Readback.Vertices.data(),result.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite async vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite async indices"};
    for (unsigned i=0;i<2;++i)
    { auto& resource=resources[i]; resource.NaniteGeometry=true; resource.Buffer=work->Buffers[i+4]; if (work->Upload.ReadbackGeometry) { resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[i],resource.Bytes.size()); } }
    result.VertexBuffer={NextBuffer++}; result.IndexBuffer={NextBuffer++}; result.DispatchCount=unsigned(work->Plan.Dispatches.size());
    try { Buffers.emplace(result.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(result.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(result.VertexBuffer.Value); throw; }
    work->Buffers[4]=work->Buffers[5]=nullptr; NaniteAsyncTasks.Erase(task); output=std::move(result); ready=true; return true;
}
