// Dedicated allocators, command lists and fences permit overlapping immutable
// production jobs. No frame fence is reset and no prefix count is read mid-job.
struct NaniteAsyncWork
{
    NaniteGpuUpload Upload; NaniteAsyncPlan Plan;
    std::array<ComPtr<ID3D12Resource>,10> Storage;
    ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands;
    ComPtr<ID3D12Fence> Completion;
};
NaniteAsyncQueue<NaniteAsyncWork> NaniteAsyncTasks;
void DestroyNaniteAsync() { NaniteAsyncTasks.Clear(); }
bool ProbeNaniteAsync(NaniteAsyncWork& work,bool& ready)
{
    const auto value=work.Completion->GetCompletedValue();
    if (value==UINT64_MAX) return FailHRESULT("Poll Nanite device removal",Device->GetDeviceRemovedReason());
    ready=value>=1; return true;
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
    const void* inputs[]{upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data()};
    const std::size_t sizes[]{upload.Metadata.size()*sizeof(unsigned),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage),upload.InitialState.size()*sizeof(unsigned)};
    for (unsigned i=0;i<4;++i)
    {
        if (i==2 && cached) { work->Storage[i]=cached->Pool->Resource; continue; }
        if (!CreateBufferResource(sizes[i],D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work->Storage[i])) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto status=work->Storage[i]->Map(0,&noRead,&mapped);
        if (FAILED(status)) return FailHRESULT("Map async Nanite input",status); std::memcpy(mapped,inputs[i],sizes[i]); work->Storage[i]->Unmap(0,nullptr);
    }
    const std::size_t bytes[]{sizes[3],work->Plan.VertexBytes,work->Plan.IndexBytes};
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    for (unsigned i=0;i<3;++i)
    {
        auto desc=BufferResourceDesc(bytes[i]); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,i==0?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(work->Storage[NaniteState+i].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Allocate async Nanite output",status);
        if ((i==0 || upload.ReadbackGeometry) && !CreateBufferResource(bytes[i],D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Storage[NaniteStateReadback+i])) return false;
    }
    auto status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work->Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work->Allocator.Get(),nullptr,IID_PPV_ARGS(work->Commands.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(work->Completion.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create async Nanite command resources",status);
    auto* commands=work->Commands.Get(); commands->SetComputeRootSignature(NaniteGeometryRoot.Get());
    for (unsigned i=0;i<3;++i)
    { commands->SetComputeRootShaderResourceView(i+1,work->Storage[i]->GetGPUVirtualAddress()); commands->SetComputeRootUnorderedAccessView(i+4,work->Storage[NaniteState+i]->GetGPUVirtualAddress()); }
    const auto transition=[&](unsigned slot,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
    { D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition={work->Storage[slot].Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; commands->ResourceBarrier(1,&b); };
    commands->CopyResource(work->Storage[NaniteState].Get(),work->Storage[3].Get()); transition(NaniteState,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    for (const auto& job:work->Plan.Dispatches)
    {
        commands->SetComputeRoot32BitConstants(0,sizeof(job.Constants)/sizeof(unsigned),&job.Constants,0); commands->SetPipelineState(NaniteGeometryPipelines[unsigned(job.Pass)].Get());
        commands->Dispatch(job.GroupsX,job.GroupsY,1);
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource=work->Storage[NaniteState].Get(); commands->ResourceBarrier(1,&b);
    }
    for (unsigned i=0;i<3;++i)
    {
        if (i==0 || upload.ReadbackGeometry)
        {
            transition(NaniteState+i,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->CopyResource(work->Storage[NaniteStateReadback+i].Get(),work->Storage[NaniteState+i].Get());
        }
        else transition(NaniteState+i,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
    }
    status=commands->Close(); if (FAILED(status)) return FailHRESULT("Close async Nanite commands",status);
    const auto task=NaniteAsyncTasks.Insert(work); ID3D12CommandList* lists[]{commands}; CommandQueue->ExecuteCommandLists(1,lists);
    status=CommandQueue->Signal(work->Completion.Get(),1);
    if (FAILED(status)) { NaniteAsyncTasks.Cancel(task); return FailHRESULT("Signal async Nanite completion",status); }
    output=task; return true;
}
bool PollNaniteGeometry(NaniteGeometryTask task,bool& ready,NaniteGeometryBuffers& output)
{
    ready=false; auto work=NaniteAsyncTasks.Find(task); if (!work) return Fail("Invalid Nanite task");
    bool complete=false; if (!ProbeNaniteAsync(*work,complete)) return false; if (!complete) return true;
    const auto read=[&](unsigned i,void* target,std::size_t bytes)
    {
        auto& buffer=work->Storage[NaniteStateReadback+i]; void* mapped=nullptr; const D3D12_RANGE range{0,bytes}; const auto status=buffer->Map(0,&range,&mapped);
        if (FAILED(status)) return FailHRESULT("Map completed Nanite result",status);
        std::memcpy(target,mapped,bytes); const D3D12_RANGE noWrite{0,0}; buffer->Unmap(0,&noWrite); return true;
    };
    NaniteGeometryBuffers result; result.Readback.State.resize(work->Upload.InitialState.size());
    if (!read(0,result.Readback.State.data(),result.Readback.State.size()*sizeof(unsigned)) || !ValidateNaniteGpuState(work->Upload,result.Readback.State,result.Selection,LastError)) return false;
    const auto vertexBytes=std::size_t(result.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(result.Selection.IndexCount)*sizeof(unsigned);
    if (work->Upload.ReadbackGeometry)
    {
        result.Readback.Vertices.resize(result.Selection.VertexCount); result.Readback.Indices.resize(result.Selection.IndexCount);
        if (!read(1,result.Readback.Vertices.data(),vertexBytes) || !read(2,result.Readback.Indices.data(),indexBytes) || !ValidateNaniteGpuReadback(work->Upload,result.Readback,result.Selection,LastError)) return false;
    }
    std::array<BufferResource,2> resources; const void* mirrors[]{result.Readback.Vertices.data(),result.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite async vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite async indices"};
    for (unsigned i=0;i<2;++i)
    {
        auto& resource=resources[i]; resource.NaniteGeometry=true; resource.Resource=work->Storage[NaniteVertices+i]; resource.State=D3D12_RESOURCE_STATE_COMMON;
        if (work->Upload.ReadbackGeometry) { resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[i],resource.Bytes.size()); }
        if (!i) resource.VertexView={resource.Resource->GetGPUVirtualAddress(),UINT(vertexBytes),sizeof(Vertex)};
        else resource.IndexView={resource.Resource->GetGPUVirtualAddress(),UINT(indexBytes),DXGI_FORMAT_R32_UINT};
    }
    result.VertexBuffer={NextBuffer++}; result.IndexBuffer={NextBuffer++}; result.DispatchCount=unsigned(work->Plan.Dispatches.size());
    try { Buffers.emplace(result.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(result.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(result.VertexBuffer.Value); throw; }
    NaniteAsyncTasks.Erase(task); output=std::move(result); ready=true; return true;
}
