// Dedicated direct-queue submissions retain scratch until fence recovery. Prefix
// readback precedes exact UAV allocation; decode publishes the original GPU buffers.
// Buffers decay to COMMON after ExecuteCommandLists, including explicit barriers:
// https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12
ComPtr<ID3D12RootSignature> VirtualGeometryRoot;
std::array<ComPtr<ID3D12PipelineState>,6> VirtualGeometryPipelines;
ComPtr<ID3D12CommandAllocator> VirtualGeometryAllocator;
ComPtr<ID3D12GraphicsCommandList> VirtualGeometryCommands;
std::array<ComPtr<ID3D12Resource>,10> VirtualGeometryScratch;
static constexpr std::size_t VirtualState=4,VirtualVertices=5,VirtualIndices=6,VirtualStateReadback=7,VirtualVertexReadback=8,VirtualIndexReadback=9;
void DestroyVirtualGeometry()
{
    DestroyVirtualGeometryCaches();
    for (auto& buffer:VirtualGeometryScratch) buffer.Reset(); VirtualGeometryCommands.Reset(); VirtualGeometryAllocator.Reset();
    for (auto& pipeline:VirtualGeometryPipelines) pipeline.Reset(); VirtualGeometryRoot.Reset();
}
bool EnsureVirtualGeometry()
{
    if (!VirtualGeometryRoot)
    {
        std::array<D3D12_ROOT_PARAMETER,7> parameters{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants={0,0,sizeof(VirtualGeometryGpuConstants)/sizeof(std::uint32_t)};
        for (UINT slot=0;slot<3;++slot)
        {
            parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[slot+1].Descriptor.ShaderRegister=slot;
            parameters[slot+4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[slot+4].Descriptor.ShaderRegister=slot;
        }
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=static_cast<UINT>(parameters.size()); desc.pParameters=parameters.data(); ComPtr<ID3DBlob> code,errors;
        auto result=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf()); if (FAILED(result)) return FailHRESULT("Serialize virtual geometry root",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(VirtualGeometryRoot.GetAddressOf())); if (FAILED(result)) return FailHRESULT("Create virtual geometry root",result);
    }
    if (!LoadTechniqueShaderSource("DX12","VirtualGeometry.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSEvaluate","CSPropagate","CSSelect","CSClusters","CSBlocks","CSDecode"};
    for (std::size_t stage=0;stage<VirtualGeometryPipelines.size();++stage) if (!VirtualGeometryPipelines[stage])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[stage],"cs_5_0",code,nullptr,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=VirtualGeometryRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto result=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(VirtualGeometryPipelines[stage].GetAddressOf())); if (FAILED(result)) return FailHRESULT("Create virtual geometry pipeline",result);
    }
    if (!VirtualGeometryAllocator)
    { const auto result=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(VirtualGeometryAllocator.GetAddressOf())); if (FAILED(result)) return FailHRESULT("Create virtual allocator",result); }
    if (!VirtualGeometryCommands)
    {
        auto result=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,VirtualGeometryAllocator.Get(),nullptr,IID_PPV_ARGS(VirtualGeometryCommands.GetAddressOf()));
        if (SUCCEEDED(result)) result=VirtualGeometryCommands->Close(); if (FAILED(result)) return FailHRESULT("Create virtual command list",result);
    }
    return true;
}
bool CreateVirtualGeometry(const VirtualGeometryGpuUpload& upload,VirtualGeometryBuffers& output,VirtualGeometryCacheHandle cache={})
{
    std::vector<VirtualGeometryGpuDispatch> jobs;
    if (!BuildVirtualGeometryGpuSchedule(upload,jobs,LastError) || !RecoverSubmittedWork("Create virtual geometry") || !WaitForGPU() || !EnsureVirtualGeometry()) return false;
    const VirtualGeometryNativeCache* cached=nullptr;
    if (cache) { const auto found=VirtualGeometryCaches.find(cache.Value); if (found==VirtualGeometryCaches.end()) return Fail("DX12 virtual cache handle is invalid"); cached=&found->second; if (!MatchVirtualGeometryCache(cached->Info,upload,LastError)) return false; }
    for (auto& buffer:VirtualGeometryScratch) buffer.Reset();
    const void* inputs[]{upload.Metadata.data(),upload.PageTable.data(),upload.PhysicalPages.data(),upload.InitialState.data()};
    const std::size_t sizes[]{upload.Metadata.size()*sizeof(std::uint32_t),upload.PageTable.size()*sizeof(ShaderUInt4),upload.PhysicalPages.size()*sizeof(VirtualGeometryPage),upload.InitialState.size()*sizeof(std::uint32_t)};
    for (std::size_t slot=0;slot<4;++slot)
    {
        if (slot==2 && cached) { VirtualGeometryScratch[slot]=cached->Pool->Resource; continue; }
        if (!CreateBufferResource(sizes[slot],D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,VirtualGeometryScratch[slot])) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto result=VirtualGeometryScratch[slot]->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map virtual source",result); std::memcpy(mapped,inputs[slot],sizes[slot]); VirtualGeometryScratch[slot]->Unmap(0,nullptr);
    }
    auto uav=[&](std::size_t slot,std::size_t bytes,D3D12_RESOURCE_STATES state)
    {
        VirtualGeometryScratch[slot].Reset(); const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const auto result=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,IID_PPV_ARGS(VirtualGeometryScratch[slot].GetAddressOf()));
        return SUCCEEDED(result) || FailHRESULT("Create virtual UAV",result);
    };
    if (!uav(VirtualState,sizes[3],D3D12_RESOURCE_STATE_COPY_DEST) || !uav(VirtualVertices,sizeof(std::uint32_t),D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
        !uav(VirtualIndices,sizeof(std::uint32_t),D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
        !CreateBufferResource(sizes[3],D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,VirtualGeometryScratch[VirtualStateReadback])) return false;
    auto begin=[&]()
    {
        auto result=VirtualGeometryAllocator->Reset(); if (SUCCEEDED(result)) result=VirtualGeometryCommands->Reset(VirtualGeometryAllocator.Get(),nullptr);
        if (FAILED(result)) return FailHRESULT("Reset virtual recording",result);
        VirtualGeometryCommands->SetComputeRootSignature(VirtualGeometryRoot.Get());
        for (UINT slot=0;slot<3;++slot)
        { VirtualGeometryCommands->SetComputeRootShaderResourceView(slot+1,VirtualGeometryScratch[slot]->GetGPUVirtualAddress()); VirtualGeometryCommands->SetComputeRootUnorderedAccessView(slot+4,VirtualGeometryScratch[slot+VirtualState]->GetGPUVirtualAddress()); }
        return true;
    };
    auto transition=[&](std::size_t slot,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
    { D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; barrier.Transition={VirtualGeometryScratch[slot].Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; VirtualGeometryCommands->ResourceBarrier(1,&barrier); };
    auto dispatch=[&](const VirtualGeometryGpuDispatch& job,const VirtualGeometryGpuConstants& constants)
    {
        VirtualGeometryCommands->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0);
        VirtualGeometryCommands->SetPipelineState(VirtualGeometryPipelines[static_cast<unsigned>(job.Pass)].Get()); VirtualGeometryCommands->Dispatch(job.GroupsX,job.GroupsY,1);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=VirtualGeometryScratch[VirtualState].Get(); VirtualGeometryCommands->ResourceBarrier(1,&barrier);
    };
    auto submit=[&]()
    {
        const auto result=VirtualGeometryCommands->Close(); if (FAILED(result)) return FailHRESULT("Close virtual recording",result);
        ID3D12CommandList* lists[]{VirtualGeometryCommands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
        return SignalAndWait();
    };
    auto read=[&](std::size_t slot,std::size_t bytes,void* destination)
    {
        void* mapped=nullptr; const D3D12_RANGE range{0,bytes}; const auto result=VirtualGeometryScratch[slot]->Map(0,&range,&mapped);
        if (FAILED(result)) return FailHRESULT("Map virtual readback",result); std::memcpy(destination,mapped,bytes); const D3D12_RANGE noWrite{0,0}; VirtualGeometryScratch[slot]->Unmap(0,&noWrite); return true;
    };
    if (!begin()) return false;
    VirtualGeometryCommands->CopyResource(VirtualGeometryScratch[VirtualState].Get(),VirtualGeometryScratch[3].Get()); transition(VirtualState,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    for (std::size_t stage=0;stage+1<jobs.size();++stage) dispatch(jobs[stage],jobs[stage].Constants);
    transition(VirtualState,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    VirtualGeometryCommands->CopyResource(VirtualGeometryScratch[VirtualStateReadback].Get(),VirtualGeometryScratch[VirtualState].Get());
    if (!submit()) return false;
    VirtualGeometryBuffers candidate; candidate.Readback.State.resize(upload.InitialState.size());
    if (!read(VirtualStateReadback,sizes[3],candidate.Readback.State.data()) || !ValidateVirtualGeometryGpuState(upload,candidate.Readback.State,candidate.Selection,LastError)) return false;
    const std::size_t vertexBytes=std::size_t(candidate.Selection.VertexCount)*sizeof(Vertex),indexBytes=std::size_t(candidate.Selection.IndexCount)*sizeof(std::uint32_t);
    if (!uav(VirtualVertices,vertexBytes,D3D12_RESOURCE_STATE_UNORDERED_ACCESS) || !uav(VirtualIndices,indexBytes,D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
        !CreateBufferResource(vertexBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,VirtualGeometryScratch[VirtualVertexReadback]) ||
        !CreateBufferResource(indexBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,VirtualGeometryScratch[VirtualIndexReadback]) || !begin()) return false;
    transition(VirtualState,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto constants=jobs.back().Constants; constants.Capacity.X=candidate.Selection.VertexCount; constants.Capacity.Y=candidate.Selection.IndexCount; dispatch(jobs.back(),constants);
    for (std::size_t stream=0;stream<3;++stream)
    {
        const auto slot=VirtualState+stream; transition(slot,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        VirtualGeometryCommands->CopyResource(VirtualGeometryScratch[VirtualStateReadback+stream].Get(),VirtualGeometryScratch[slot].Get());
        if (stream) transition(slot,D3D12_RESOURCE_STATE_COPY_SOURCE,stream==1?D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER:D3D12_RESOURCE_STATE_INDEX_BUFFER);
    }
    if (!submit()) return false;
    candidate.Readback.Vertices.resize(candidate.Selection.VertexCount); candidate.Readback.Indices.resize(candidate.Selection.IndexCount);
    if (!read(VirtualStateReadback,sizes[3],candidate.Readback.State.data()) || !read(VirtualVertexReadback,vertexBytes,candidate.Readback.Vertices.data()) ||
        !read(VirtualIndexReadback,indexBytes,candidate.Readback.Indices.data()) || !ValidateVirtualGeometryGpuReadback(upload,candidate.Readback,candidate.Selection,LastError)) return false;
    std::array<BufferResource,2> resources; const void* mirrors[]{candidate.Readback.Vertices.data(),candidate.Readback.Indices.data()};
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"Virtual geometry vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"Virtual geometry indices"};
    for (std::size_t stream=0;stream<resources.size();++stream)
    {
        auto& resource=resources[stream]; resource.Resource=VirtualGeometryScratch[VirtualVertices+stream]; resource.State=D3D12_RESOURCE_STATE_COMMON;
        resource.Bytes.resize(resource.Description.SizeInBytes); std::memcpy(resource.Bytes.data(),mirrors[stream],resource.Bytes.size());
        if (!stream) resource.VertexView={resource.Resource->GetGPUVirtualAddress(),static_cast<UINT>(vertexBytes),sizeof(Vertex)};
        else resource.IndexView={resource.Resource->GetGPUVirtualAddress(),static_cast<UINT>(indexBytes),DXGI_FORMAT_R32_UINT};
    }
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(jobs.size());
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    output=std::move(candidate); for (auto& buffer:VirtualGeometryScratch) buffer.Reset(); return true;
}
