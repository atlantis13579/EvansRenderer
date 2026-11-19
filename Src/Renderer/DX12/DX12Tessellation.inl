// Dedicated direct-queue recording keeps geometry generation outside frame
// recording. Persistent scratch survives a failed wait; queue recovery precedes
// reuse. Published buffers are the actual UAV allocations, not CPU re-uploads.
ComPtr<ID3D12RootSignature> TessellationRoot;
std::array<ComPtr<ID3D12PipelineState>,3> TessellationPipelines;
ComPtr<ID3D12CommandAllocator> TessellationAllocator;
ComPtr<ID3D12GraphicsCommandList> TessellationCommands;
std::array<ComPtr<ID3D12Resource>,10> TessellationScratch;
static constexpr std::size_t TessellationControls=5,TessellationVertices=6,TessellationIndices=7,TessellationReadback=8;
void DestroyTessellation()
{
    for (auto& buffer:TessellationScratch) buffer.Reset();
    TessellationCommands.Reset(); TessellationAllocator.Reset();
    for (auto& pipeline:TessellationPipelines) pipeline.Reset(); TessellationRoot.Reset();
}
bool EnsureTessellation()
{
    if (!TessellationRoot)
    {
        D3D12_ROOT_PARAMETER parameters[9]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants={0,0,sizeof(TessellationConstants)/sizeof(std::uint32_t)};
        for (UINT slot=0;slot<5;++slot)
        { parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[slot+1].Descriptor.ShaderRegister=slot; }
        for (UINT slot=0;slot<3;++slot)
        { parameters[slot+6].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[slot+6].Descriptor.ShaderRegister=slot; }
        D3D12_ROOT_SIGNATURE_DESC description{}; description.NumParameters=9; description.pParameters=parameters;
        ComPtr<ID3DBlob> serialized,errors;
        auto result=SerializeRootSignatureFunction(&description,D3D_ROOT_SIGNATURE_VERSION_1,serialized.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("Serialize PN root signature",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(TessellationRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("Create PN root signature",result);
    }
    if (!TessellationPipelines[0] || !TessellationPipelines[1] || !TessellationPipelines[2])
    {
        if (!LoadTechniqueShaderSource("DX12","Tessellation.hlsl",ShaderCode,LastError)) return false;
        constexpr const char* entries[]{"CSEdges","CSVertices","CSIndices"};
        for (std::size_t phase=0;phase<TessellationPipelines.size();++phase) if (!TessellationPipelines[phase])
        {
            ComPtr<ID3DBlob> code; if (!CompileShader(entries[phase],"cs_5_0",code,nullptr,true)) return false;
            D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=TessellationRoot.Get();
            pipeline.CS={code->GetBufferPointer(),code->GetBufferSize()};
            const auto result=Evans::Renderer::CreateCachedComputePipelineState(Device,&pipeline,IID_PPV_ARGS(TessellationPipelines[phase].GetAddressOf()));
            if (FAILED(result)) return FailHRESULT("Create PN compute pipeline",result);
        }
    }
    if (!TessellationAllocator)
    {
        const auto result=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(TessellationAllocator.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("Create PN command allocator",result);
    }
    if (!TessellationCommands)
    {
        auto result=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,TessellationAllocator.Get(),nullptr,IID_PPV_ARGS(TessellationCommands.GetAddressOf()));
        if (SUCCEEDED(result)) result=TessellationCommands->Close();
        if (FAILED(result)) return FailHRESULT("Create PN command list",result);
    }
    return true;
}
bool CreateTessellated(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    TessellationConstants constants;
    if (!ValidateTessellationUpload(geometry,settings,constants,LastError) || !RecoverSubmittedWork("Create PN geometry") || !WaitForGPU() || !EnsureTessellation()) return false;
    for (auto& buffer:TessellationScratch) buffer.Reset();
    const void* data[]{geometry.Vertices.data(),geometry.Patches.data(),geometry.PatchEdges.data(),geometry.AttributeEdges.data(),geometry.GeometricEdges.data()};
    const std::size_t sizes[]{geometry.Vertices.size()*sizeof(Vertex),geometry.Patches.size()*sizeof(ShaderUInt4),geometry.PatchEdges.size()*sizeof(ShaderUInt4),
        geometry.AttributeEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*2*sizeof(Float4),
        std::size_t(constants.Outputs.X)*sizeof(Vertex),std::size_t(constants.Outputs.Y)*sizeof(std::uint32_t)};
    for (std::size_t slot=0;slot<5;++slot)
    {
        if (!CreateBufferResource(sizes[slot],D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,TessellationScratch[slot])) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto result=TessellationScratch[slot]->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map PN source upload",result);
        std::memcpy(mapped,data[slot],sizes[slot]); TessellationScratch[slot]->Unmap(0,nullptr);
    }
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    for (std::size_t slot=TessellationControls;slot<=TessellationIndices;++slot)
    {
        auto description=BufferResourceDesc(sizes[slot]); description.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const auto result=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&description,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(TessellationScratch[slot].GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("Create PN UAV allocation",result);
    }
    for (std::size_t stream=0;stream<2;++stream)
        if (!CreateBufferResource(sizes[TessellationVertices+stream],D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,TessellationScratch[TessellationReadback+stream])) return false;
    auto result=TessellationAllocator->Reset(); if (SUCCEEDED(result)) result=TessellationCommands->Reset(TessellationAllocator.Get(),nullptr);
    if (FAILED(result)) return FailHRESULT("Reset PN command recording",result);
    TessellationCommands->SetComputeRootSignature(TessellationRoot.Get());
    TessellationCommands->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0);
    for (UINT slot=0;slot<5;++slot) TessellationCommands->SetComputeRootShaderResourceView(slot+1,TessellationScratch[slot]->GetGPUVirtualAddress());
    for (UINT slot=0;slot<3;++slot) TessellationCommands->SetComputeRootUnorderedAccessView(slot+6,TessellationScratch[slot+TessellationControls]->GetGPUVirtualAddress());
    const unsigned counts[]{constants.Inputs.W,constants.Outputs.X,constants.Outputs.Y/3};
    for (std::size_t phase=0;phase<TessellationPipelines.size();++phase)
    {
        TessellationCommands->SetPipelineState(TessellationPipelines[phase].Get()); TessellationCommands->Dispatch((counts[phase]+TessellationThreadCount-1)/TessellationThreadCount,1,1);
        if (phase==0)
        {
            D3D12_RESOURCE_BARRIER dependency{}; dependency.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; dependency.UAV.pResource=TessellationScratch[TessellationControls].Get();
            TessellationCommands->ResourceBarrier(1,&dependency);
        }
    }
    std::array<BufferResource,2> resources;
    resources[0].Description={BufferUsage::Vertex,sizes[TessellationVertices],false,"PN tessellated vertices"};
    resources[1].Description={BufferUsage::Index,sizes[TessellationIndices],false,"PN tessellated indices"};
    for (std::size_t stream=0;stream<resources.size();++stream)
    {
        auto* destination=TessellationScratch[TessellationVertices+stream].Get();
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={destination,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
        TessellationCommands->ResourceBarrier(1,&barrier); TessellationCommands->CopyResource(TessellationScratch[TessellationReadback+stream].Get(),destination);
        barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE; barrier.Transition.StateAfter=stream?D3D12_RESOURCE_STATE_INDEX_BUFFER:D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        TessellationCommands->ResourceBarrier(1,&barrier);
    }
    result=TessellationCommands->Close(); if (FAILED(result)) return FailHRESULT("Close PN command list",result);
    ID3D12CommandList* lists[]{TessellationCommands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false;
    TessellatedGeometryBuffers candidate; candidate.Vertices.resize(constants.Outputs.X); candidate.Indices.resize(constants.Outputs.Y);
    void* mirrors[]{candidate.Vertices.data(),candidate.Indices.data()};
    for (std::size_t stream=0;stream<resources.size();++stream)
    {
        auto& resource=resources[stream]; const auto bytes=resource.Description.SizeInBytes; resource.Bytes.resize(bytes);
        void* mapped=nullptr; const D3D12_RANGE read{0,bytes}; result=TessellationScratch[TessellationReadback+stream]->Map(0,&read,&mapped);
        if (FAILED(result)) return FailHRESULT("Map PN stream readback",result);
        std::memcpy(mirrors[stream],mapped,bytes); std::memcpy(resource.Bytes.data(),mapped,bytes);
        const D3D12_RANGE noWrite{0,0}; TessellationScratch[TessellationReadback+stream]->Unmap(0,&noWrite);
        resource.Resource=TessellationScratch[TessellationVertices+stream]; resource.State=D3D12_RESOURCE_STATE_COMMON;
        if (!stream) resource.VertexView={resource.Resource->GetGPUVirtualAddress(),static_cast<UINT>(bytes),sizeof(Vertex)};
        else resource.IndexView={resource.Resource->GetGPUVirtualAddress(),static_cast<UINT>(bytes),DXGI_FORMAT_R32_UINT};
    }
    if (!ValidateTriangleGeometry(candidate.Vertices,candidate.Indices,LastError)) return false;
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=3;
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    output=std::move(candidate); for (auto& buffer:TessellationScratch) buffer.Reset(); return true;
}
