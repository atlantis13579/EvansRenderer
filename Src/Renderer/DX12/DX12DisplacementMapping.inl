// A dedicated direct command list prepares geometry between frames. Scratch
// resources survive a failed fence wait; the next job must recover queue work
// before replacing them. Successful results retain the actual UAV allocation.
ComPtr<ID3D12RootSignature> DisplacementRoot;
std::array<ComPtr<ID3D12PipelineState>,2> DisplacementPipelines;
ComPtr<ID3D12CommandAllocator> DisplacementAllocator;
ComPtr<ID3D12GraphicsCommandList> DisplacementCommands;
std::array<ComPtr<ID3D12Resource>,7> DisplacementScratch;
static constexpr std::size_t DisplacementOutput=5,DisplacementReadback=6;
void DestroyDisplacement()
{
    for (auto& buffer:DisplacementScratch) buffer.Reset();
    DisplacementCommands.Reset(); DisplacementAllocator.Reset();
    for (auto& pipeline:DisplacementPipelines) pipeline.Reset(); DisplacementRoot.Reset();
}
bool EnsureDisplacement()
{
    if (!DisplacementRoot)
    {
        D3D12_ROOT_PARAMETER parameters[7]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants={0,0,sizeof(DisplacementConstants)/sizeof(std::uint32_t)};
        for (UINT input=0;input<5;++input)
        { parameters[input+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[input+1].Descriptor.ShaderRegister=input; }
        parameters[6].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC description{}; description.NumParameters=7; description.pParameters=parameters;
        ComPtr<ID3DBlob> serialized,errors;
        auto result=SerializeRootSignatureFunction(&description,D3D_ROOT_SIGNATURE_VERSION_1,serialized.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("Serialize displacement root signature",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(DisplacementRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("Create displacement root signature",result);
    }
    if (!DisplacementPipelines[0] || !DisplacementPipelines[1])
    {
        if (!LoadTechniqueShaderSource("DX12","DisplacementMapping.hlsl",ShaderCode,LastError)) return false;
        constexpr const char* entries[]{"CSDisplace","CSNormals"};
        for (std::size_t phase=0;phase<DisplacementPipelines.size();++phase) if (!DisplacementPipelines[phase])
        {
            ComPtr<ID3DBlob> code; if (!CompileShader(entries[phase],"cs_5_0",code,nullptr,true)) return false;
            D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=DisplacementRoot.Get();
            pipeline.CS={code->GetBufferPointer(),code->GetBufferSize()};
            const auto result=Evans::Renderer::CreateCachedComputePipelineState(Device,&pipeline,IID_PPV_ARGS(DisplacementPipelines[phase].GetAddressOf()));
            if (FAILED(result)) return FailHRESULT("Create displacement compute pipeline",result);
        }
    }
    if (!DisplacementAllocator)
    {
        const auto result=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(DisplacementAllocator.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("Create displacement command allocator",result);
    }
    if (!DisplacementCommands)
    {
        auto result=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,DisplacementAllocator.Get(),nullptr,IID_PPV_ARGS(DisplacementCommands.GetAddressOf()));
        if (SUCCEEDED(result)) result=DisplacementCommands->Close();
        if (FAILED(result)) return FailHRESULT("Create displacement command list",result);
    }
    return true;
}
bool CreateDisplaced(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    DisplacementConstants constants;
    if (!ValidateDisplacementUpload(geometry,settings,constants,LastError) || !RecoverSubmittedWork("Create displaced geometry") ||
        !WaitForGPU() || !EnsureDisplacement()) return false;
    for (auto& buffer:DisplacementScratch) buffer.Reset();
    const void* data[]{geometry.Vertices.data(),geometry.Directions.data(),geometry.Indices.data(),geometry.Neighborhoods.data(),geometry.AdjacentTriangles.data()};
    const std::size_t sizes[]{geometry.Vertices.size()*sizeof(Vertex),geometry.Directions.size()*sizeof(Float4),geometry.Indices.size()*sizeof(std::uint32_t),
        geometry.Neighborhoods.size()*sizeof(ShaderUInt4),geometry.AdjacentTriangles.size()*sizeof(std::uint32_t)};
    for (std::size_t slot=0;slot<5;++slot)
    {
        if (!CreateBufferResource(sizes[slot],D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,DisplacementScratch[slot])) return false;
        void* mapped=nullptr; D3D12_RANGE noRead{0,0}; const auto result=DisplacementScratch[slot]->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map displacement source upload",result);
        std::memcpy(mapped,data[slot],sizes[slot]); DisplacementScratch[slot]->Unmap(0,nullptr);
    }
    auto vertexDescription=BufferResourceDesc(sizes[0]); vertexDescription.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    auto result=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&vertexDescription,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,
        IID_PPV_ARGS(DisplacementScratch[DisplacementOutput].GetAddressOf()));
    if (FAILED(result)) return FailHRESULT("Create displacement vertex/UAV allocation",result);
    if (!CreateBufferResource(sizes[0],D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,DisplacementScratch[DisplacementReadback])) return false;
    BufferResource resource; resource.Description={BufferUsage::Vertex,sizes[0],false,"Displaced vertices"}; resource.Bytes.resize(sizes[0]);
    DisplacedVertexBuffer candidate; candidate.Vertices.resize(geometry.Vertices.size());
    result=DisplacementAllocator->Reset(); if (SUCCEEDED(result)) result=DisplacementCommands->Reset(DisplacementAllocator.Get(),nullptr);
    if (FAILED(result)) return FailHRESULT("Reset displacement command recording",result);
    DisplacementCommands->SetComputeRootSignature(DisplacementRoot.Get());
    DisplacementCommands->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0);
    for (UINT slot=0;slot<5;++slot) DisplacementCommands->SetComputeRootShaderResourceView(slot+1,DisplacementScratch[slot]->GetGPUVirtualAddress());
    auto* destination=DisplacementScratch[DisplacementOutput].Get();
    DisplacementCommands->SetComputeRootUnorderedAccessView(6,destination->GetGPUVirtualAddress());
    const UINT groups=static_cast<UINT>((geometry.Vertices.size()+DisplacementThreadCount-1)/DisplacementThreadCount);
    DisplacementCommands->SetPipelineState(DisplacementPipelines[0].Get()); DisplacementCommands->Dispatch(groups,1,1);
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=destination;
    DisplacementCommands->ResourceBarrier(1,&barrier);
    DisplacementCommands->SetPipelineState(DisplacementPipelines[1].Get()); DisplacementCommands->Dispatch(groups,1,1);
    barrier={}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={destination,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
    DisplacementCommands->ResourceBarrier(1,&barrier);
    DisplacementCommands->CopyResource(DisplacementScratch[DisplacementReadback].Get(),destination);
    barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE; barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    DisplacementCommands->ResourceBarrier(1,&barrier); result=DisplacementCommands->Close();
    if (FAILED(result)) return FailHRESULT("Close displacement command list",result);
    ID3D12CommandList* lists[]{DisplacementCommands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false;
    void* mapped=nullptr; const D3D12_RANGE read{0,sizes[0]}; result=DisplacementScratch[DisplacementReadback]->Map(0,&read,&mapped);
    if (FAILED(result)) return FailHRESULT("Map displacement readback",result);
    std::memcpy(candidate.Vertices.data(),mapped,sizes[0]); std::memcpy(resource.Bytes.data(),mapped,sizes[0]);
    const D3D12_RANGE noWrite{0,0}; DisplacementScratch[DisplacementReadback]->Unmap(0,&noWrite);
    if (!ValidateTriangleGeometry(candidate.Vertices,geometry.Indices,LastError)) return false;
    resource.Resource=std::move(DisplacementScratch[DisplacementOutput]);
    // Buffers decay to COMMON after execution completes on the direct queue.
    resource.State=D3D12_RESOURCE_STATE_COMMON;
    resource.VertexView={resource.Resource->GetGPUVirtualAddress(),static_cast<UINT>(sizes[0]),sizeof(Vertex)};
    candidate.Buffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(DisplacementPipelines.size());
    Buffers.emplace(candidate.Buffer.Value,std::move(resource)); output=std::move(candidate);
    for (auto& buffer:DisplacementScratch) buffer.Reset(); return true;
}
