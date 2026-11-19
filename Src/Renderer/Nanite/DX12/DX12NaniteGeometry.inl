// Dedicated direct-queue submissions retain scratch until fence recovery. Prefix
// readback precedes exact UAV allocation; decode publishes the original GPU buffers.
// Buffers decay to COMMON after ExecuteCommandLists, including explicit barriers:
// https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12
ComPtr<ID3D12RootSignature> NaniteGeometryRoot;
std::array<ComPtr<ID3D12PipelineState>,5> NaniteGeometryPipelines;
ComPtr<ID3D12CommandAllocator> NaniteGeometryAllocator;
ComPtr<ID3D12GraphicsCommandList> NaniteGeometryCommands;
std::array<ComPtr<ID3D12Resource>,10> NaniteGeometryScratch;
static constexpr std::size_t NaniteState=4,NaniteVertices=5,NaniteIndices=6,NaniteStateReadback=7,NaniteVertexReadback=8,NaniteIndexReadback=9;
void DestroyNaniteGeometry()
{
    DestroyNaniteAsync(); DestroyNaniteCacheAsync(); DestroyNaniteCaches();

    for (auto& buffer:NaniteGeometryScratch) buffer.Reset(); NaniteGeometryCommands.Reset(); NaniteGeometryAllocator.Reset();
    for (auto& pipeline:NaniteGeometryPipelines) pipeline.Reset(); NaniteGeometryRoot.Reset();
}
bool EnsureNaniteGeometry()
{
    if (!NaniteGeometryRoot)
    {
        std::array<D3D12_ROOT_PARAMETER,7> parameters{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants={0,0,sizeof(NaniteGpuConstants)/sizeof(std::uint32_t)};
        for (UINT slot=0;slot<3;++slot)
        {
            parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[slot+1].Descriptor.ShaderRegister=slot;
            parameters[slot+4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[slot+4].Descriptor.ShaderRegister=slot;
        }
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=static_cast<UINT>(parameters.size()); desc.pParameters=parameters.data(); ComPtr<ID3DBlob> code,errors;
        auto result=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf()); if (FAILED(result)) return FailHRESULT("Serialize Nanite geometry root",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(NaniteGeometryRoot.GetAddressOf())); if (FAILED(result)) return FailHRESULT("Create Nanite geometry root",result);
    }
    if (!LoadTechniqueShaderSource("DX12","NaniteGeometry.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSEvaluate","CSRefine","CSClusters","CSBlocks","CSDecode"};
    for (std::size_t stage=0;stage<NaniteGeometryPipelines.size();++stage) if (!NaniteGeometryPipelines[stage])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[stage],"cs_5_0",code,nullptr,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=NaniteGeometryRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto result=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(NaniteGeometryPipelines[stage].GetAddressOf())); if (FAILED(result)) return FailHRESULT("Create Nanite geometry pipeline",result);
    }
    if (!NaniteGeometryAllocator)
    { const auto result=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(NaniteGeometryAllocator.GetAddressOf())); if (FAILED(result)) return FailHRESULT("Create Nanite allocator",result); }
    if (!NaniteGeometryCommands)
    {
        auto result=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,NaniteGeometryAllocator.Get(),nullptr,IID_PPV_ARGS(NaniteGeometryCommands.GetAddressOf()));
        if (SUCCEEDED(result)) result=NaniteGeometryCommands->Close(); if (FAILED(result)) return FailHRESULT("Create Nanite command list",result);
    }
    return true;
}
bool CreateNaniteGeometry(const NaniteGpuUpload& upload,NaniteGeometryBuffers& output,NaniteCacheHandle cache={})
{
    std::vector<NaniteGpuDispatch> jobs;
    if (!BuildNaniteGpuSchedule(upload,jobs,LastError) || !RecoverSubmittedWork("Create Nanite geometry") || !WaitForGPU() || !EnsureNaniteGeometry()) return false;
    const NaniteNativeCache* cached=nullptr;
    if (cache) { const auto found=NaniteCaches.find(cache.Value); if (found==NaniteCaches.end()) return Fail("DX12 Nanite cache handle is invalid"); cached=&found->second; if (!MatchNaniteCache(cached->Info,upload,LastError)) return false; }
    for (auto& buffer:NaniteGeometryScratch) buffer.Reset();
    const void* inputs[]{upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data()};
    const std::size_t sizes[]{upload.Metadata.size()*sizeof(std::uint32_t),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(NaniteGpuPhysicalPage),upload.InitialState.size()*sizeof(std::uint32_t)};
    for (std::size_t slot=0;slot<4;++slot)
    {
        if (slot==2 && cached) { NaniteGeometryScratch[slot]=cached->Pool->Resource; continue; }
        if (!CreateBufferResource(sizes[slot],D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,NaniteGeometryScratch[slot])) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto result=NaniteGeometryScratch[slot]->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map Nanite source",result); std::memcpy(mapped,inputs[slot],sizes[slot]); NaniteGeometryScratch[slot]->Unmap(0,nullptr);
    }
    auto uav=[&](std::size_t slot,std::size_t bytes,D3D12_RESOURCE_STATES state)
    {
        NaniteGeometryScratch[slot].Reset(); const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const auto result=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,IID_PPV_ARGS(NaniteGeometryScratch[slot].GetAddressOf()));
        return SUCCEEDED(result) || FailHRESULT("Create Nanite UAV",result);
    };
    if (!uav(NaniteState,sizes[3],D3D12_RESOURCE_STATE_COPY_DEST) || !uav(NaniteVertices,sizeof(std::uint32_t),D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
        !uav(NaniteIndices,sizeof(std::uint32_t),D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
        !CreateBufferResource(sizes[3],D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,NaniteGeometryScratch[NaniteStateReadback])) return false;
    auto begin=[&]()
    {
        auto result=NaniteGeometryAllocator->Reset(); if (SUCCEEDED(result)) result=NaniteGeometryCommands->Reset(NaniteGeometryAllocator.Get(),nullptr);
        if (FAILED(result)) return FailHRESULT("Reset Nanite recording",result);
        NaniteGeometryCommands->SetComputeRootSignature(NaniteGeometryRoot.Get());
        for (UINT slot=0;slot<3;++slot)
        { NaniteGeometryCommands->SetComputeRootShaderResourceView(slot+1,NaniteGeometryScratch[slot]->GetGPUVirtualAddress()); NaniteGeometryCommands->SetComputeRootUnorderedAccessView(slot+4,NaniteGeometryScratch[slot+NaniteState]->GetGPUVirtualAddress()); }
        return true;
    };
    auto transition=[&](std::size_t slot,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
    { D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; barrier.Transition={NaniteGeometryScratch[slot].Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; NaniteGeometryCommands->ResourceBarrier(1,&barrier); };
    auto dispatch=[&](const NaniteGpuDispatch& job,const NaniteGpuConstants& constants)
    {
        NaniteGeometryCommands->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0);
        NaniteGeometryCommands->SetPipelineState(NaniteGeometryPipelines[static_cast<unsigned>(job.Pass)].Get()); NaniteGeometryCommands->Dispatch(job.GroupsX,job.GroupsY,1);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=NaniteGeometryScratch[NaniteState].Get(); NaniteGeometryCommands->ResourceBarrier(1,&barrier);
    };
    auto submit=[&]()
    {
        const auto result=NaniteGeometryCommands->Close(); if (FAILED(result)) return FailHRESULT("Close Nanite recording",result);
        ID3D12CommandList* lists[]{NaniteGeometryCommands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
        return SignalAndWait();
    };
    auto read=[&](std::size_t slot,std::size_t bytes,void* destination)
    {
        void* mapped=nullptr; const D3D12_RANGE range{0,bytes}; const auto result=NaniteGeometryScratch[slot]->Map(0,&range,&mapped);
        if (FAILED(result)) return FailHRESULT("Map Nanite readback",result); std::memcpy(destination,mapped,bytes); const D3D12_RANGE noWrite{0,0}; NaniteGeometryScratch[slot]->Unmap(0,&noWrite); return true;
    };
    if (!begin()) return false;
    NaniteGeometryCommands->CopyResource(NaniteGeometryScratch[NaniteState].Get(),NaniteGeometryScratch[3].Get()); transition(NaniteState,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    for (std::size_t stage=0;stage+1<jobs.size();++stage) dispatch(jobs[stage],jobs[stage].Constants);
    transition(NaniteState,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    NaniteGeometryCommands->CopyResource(NaniteGeometryScratch[NaniteStateReadback].Get(),NaniteGeometryScratch[NaniteState].Get());
    if (!submit()) return false;
    NaniteGeometryBuffers candidate; candidate.Readback.State.resize(upload.InitialState.size());
    if (!read(NaniteStateReadback,sizes[3],candidate.Readback.State.data()) || !ValidateNaniteGpuState(upload,candidate.Readback.State,candidate.Selection,LastError)) return false;
    const std::size_t vertexBytes=std::size_t(candidate.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(candidate.Selection.IndexCount)*sizeof(std::uint32_t);
    if (!uav(NaniteVertices,vertexBytes,D3D12_RESOURCE_STATE_UNORDERED_ACCESS) || !uav(NaniteIndices,indexBytes,D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
        !CreateBufferResource(vertexBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,NaniteGeometryScratch[NaniteVertexReadback]) ||
        !CreateBufferResource(indexBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,NaniteGeometryScratch[NaniteIndexReadback]) || !begin()) return false;
    transition(NaniteState,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto constants=jobs.back().Constants; constants.Capacity.X=candidate.Selection.VertexCount; constants.Capacity.Y=candidate.Selection.IndexCount; dispatch(jobs.back(),constants);
    for (std::size_t stream=0;stream<3;++stream)
    {
        const auto slot=NaniteState+stream; transition(slot,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        NaniteGeometryCommands->CopyResource(NaniteGeometryScratch[NaniteStateReadback+stream].Get(),NaniteGeometryScratch[slot].Get());
        if (stream) transition(slot,D3D12_RESOURCE_STATE_COPY_SOURCE,stream==1?D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER:D3D12_RESOURCE_STATE_INDEX_BUFFER);
    }
    if (!submit()) return false;
    candidate.Readback.Vertices.resize(candidate.Selection.VertexCount); candidate.Readback.Indices.resize(candidate.Selection.IndexCount);
    if (!read(NaniteStateReadback,sizes[3],candidate.Readback.State.data()) || !read(NaniteVertexReadback,vertexBytes,candidate.Readback.Vertices.data()) ||
        !read(NaniteIndexReadback,indexBytes,candidate.Readback.Indices.data()) || !ValidateNaniteGpuReadback(upload,candidate.Readback,candidate.Selection,LastError)) return false;
    std::array<BufferResource,2> resources; const void* mirrors[]{candidate.Readback.Vertices.data(),candidate.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Nanite geometry vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Nanite geometry indices"};
    for (std::size_t stream=0;stream<resources.size();++stream)
    {
        auto& resource=resources[stream]; resource.NaniteGeometry=true; resource.Resource=NaniteGeometryScratch[NaniteVertices+stream]; resource.State=D3D12_RESOURCE_STATE_COMMON;
        resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[stream],resource.Bytes.size());
        if (!stream) resource.VertexView={resource.Resource->GetGPUVirtualAddress(),static_cast<UINT>(vertexBytes),sizeof(Vertex)};
        else resource.IndexView={resource.Resource->GetGPUVirtualAddress(),static_cast<UINT>(indexBytes),DXGI_FORMAT_R32_UINT};
    }
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(jobs.size());
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    output=std::move(candidate); for (auto& buffer:NaniteGeometryScratch) buffer.Reset(); return true;
}
