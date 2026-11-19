struct LumenLookupWork
{
    ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands; ComPtr<ID3D12DescriptorHeap> Views;
    ComPtr<ID3D12Resource> Cards,Queries,Feedback,Readback; std::shared_ptr<LumenSurfaceRecord> Surface;
};
ComPtr<ID3D12RootSignature> LumenLookupRoot; ComPtr<ID3D12PipelineState> LumenSurfaceLookup;
std::shared_ptr<LumenLookupWork> LumenLookupPending;
void DestroyLumenSurfaceLookup() { LumenLookupPending.reset(); LumenSurfaceLookup.Reset(); LumenLookupRoot.Reset(); }
bool EnsureLumenSurfaceLookup()
{
    if (!LumenLookupRoot)
    {
        D3D12_DESCRIPTOR_RANGE range{}; range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors=LumenSurfacePlanes;
        D3D12_ROOT_PARAMETER parameters[5]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.Num32BitValues=sizeof(LumenSurfaceLookupFrame)/sizeof(std::uint32_t);
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&range};
        parameters[2].ParameterType=parameters[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=3; parameters[3].Descriptor.ShaderRegister=4;
        parameters[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=5; desc.pParameters=parameters; ComPtr<ID3DBlob> binary,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen lookup root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenLookupRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen lookup root",status);
    }
    if (LumenSurfaceLookup) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_LOOKUP_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSLookup","cs_5_0",binary,macros,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenLookupRoot.Get(); desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(LumenSurfaceLookup.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen surface lookup pipeline",status);
}
bool LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Unknown Lumen surface lookup handle");
    LumenSurfaceLookupFrame frame; std::vector<LumenLookupCard> cards;
    if (!BuildLumenSurfaceLookupFrame(found->second->Layout,upload,frame,LastError) || !BuildLumenLookupCards(found->second->Layout,found->second->Captures,cards,LastError) ||
        !RecoverSubmittedWork("Lookup Lumen surface") || !EnsureLumenSurfaceLookup()) return false;
    LumenLookupPending.reset(); auto work=std::make_shared<LumenLookupWork>(); work->Surface=found->second;
    const auto copyInput=[&](const void* data,std::size_t bytes,ComPtr<ID3D12Resource>& resource)
    {
        if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,resource)) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto status=resource->Map(0,&noRead,&mapped);
        if (FAILED(status)) return FailHRESULT("Map Lumen lookup input",status);
        std::memcpy(mapped,data,bytes); resource->Unmap(0,nullptr); return true;
    };
    if (!copyInput(cards.data(),cards.size()*sizeof(LumenLookupCard),work->Cards) || !copyInput(upload.Queries.data(),upload.Queries.size()*sizeof(LumenSurfaceQuery),work->Queries)) return false;
    const auto bytes=upload.Queries.size()*sizeof(LumenSurfaceSample); const auto properties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(work->Feedback.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen lookup feedback",status);
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Readback)) return false;
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=LumenSurfacePlanes; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(work->Views.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen lookup atlas views",status);
    auto cpu=work->Views->GetCPUDescriptorHandleForHeapStart(); const auto increment=Device->GetDescriptorHandleIncrementSize(heap.Type);
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(work->Surface->Images[plane].Get(),&view,cpu); cpu.ptr+=increment;
    }
    status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work->Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work->Allocator.Get(),nullptr,IID_PPV_ARGS(work->Commands.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen lookup commands",status);
    work->Commands->SetPipelineState(LumenSurfaceLookup.Get()); work->Commands->SetComputeRootSignature(LumenLookupRoot.Get());
    ID3D12DescriptorHeap* heaps[]{work->Views.Get()}; work->Commands->SetDescriptorHeaps(1,heaps);
    work->Commands->SetComputeRoot32BitConstants(0,sizeof(frame)/sizeof(std::uint32_t),&frame,0);
    work->Commands->SetComputeRootDescriptorTable(1,work->Views->GetGPUDescriptorHandleForHeapStart());
    work->Commands->SetComputeRootShaderResourceView(2,work->Cards->GetGPUVirtualAddress()); work->Commands->SetComputeRootShaderResourceView(3,work->Queries->GetGPUVirtualAddress());
    work->Commands->SetComputeRootUnorderedAccessView(4,work->Feedback->GetGPUVirtualAddress());
    const auto groups=unsigned((upload.Queries.size()+LumenLookupThreads-1)/LumenLookupThreads);
    work->Commands->Dispatch(std::min(groups,LumenLookupColumns),(groups+LumenLookupColumns-1)/LumenLookupColumns,1);
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={work->Feedback.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
    work->Commands->ResourceBarrier(1,&barrier); work->Commands->CopyBufferRegion(work->Readback.Get(),0,work->Feedback.Get(),0,bytes);
    status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close Lumen lookup commands",status);
    LumenSurfaceLookupResult result; result.Identity=work->Surface->Identity; result.Samples.resize(upload.Queries.size());
    LumenLookupPending=work; ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false; LumenLookupPending.reset();
    void* mapped=nullptr; const D3D12_RANGE read{0,bytes}; status=work->Readback->Map(0,&read,&mapped);
    if (FAILED(status)) return FailHRESULT("Read Lumen lookup feedback",status);
    std::memcpy(result.Samples.data(),mapped,bytes); const D3D12_RANGE noWrite{0,0}; work->Readback->Unmap(0,&noWrite);
    if (!ValidateLumenSurfaceLookupResult(work->Surface->Identity,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
