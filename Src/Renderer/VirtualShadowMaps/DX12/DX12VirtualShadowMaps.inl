// Each submission retains its bounded descriptor table and buffers until fence
// completion. A failed wait keeps them alive for recovery rather than publishing
// a partial CPU page directory or dropping resources still used by the queue.
struct VirtualShadowRequestWork
{
    ComPtr<ID3D12CommandAllocator> Allocator;
    ComPtr<ID3D12GraphicsCommandList> Commands;
    ComPtr<ID3D12DescriptorHeap> Views;
    ComPtr<ID3D12Resource> Depth,Demand,Readback;
};
std::shared_ptr<VirtualShadowRequestWork> VirtualShadowRequestPending;
ComPtr<ID3D12RootSignature> VirtualShadowRequestRoot;
std::array<ComPtr<ID3D12PipelineState>,2> VirtualShadowRequestPipelines;
void DestroyVirtualShadowRequests()
{ VirtualShadowRequestPending.reset(); VirtualShadowRequestRoot.Reset(); for (auto& pipeline:VirtualShadowRequestPipelines) pipeline.Reset(); }
bool BuildVirtualShadowRequests(const VirtualShadowRequestUpload& upload,VirtualShadowRequestResult& output,ID3D12Resource* nativeDepth=nullptr)
{
    if ((!nativeDepth && !ValidateVirtualShadowRequestUpload(upload,LastError)) || !RecoverSubmittedWork("Virtual shadow demand") || !WaitForGPU()) return false;
    VirtualShadowRequestPending.reset(); VirtualShadowRequestConstants constants;
    if (!BuildVirtualShadowRequestConstants(upload.Frame,constants,LastError)) return false;
    if (!VirtualShadowRequestRoot)
    {
        D3D12_DESCRIPTOR_RANGE ranges[2]{}; ranges[0].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[1].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        D3D12_ROOT_PARAMETER parameters[3]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants={0,0,sizeof(constants)/sizeof(std::uint32_t)};
        for (unsigned table=0;table<2;++table)
        { ranges[table].NumDescriptors=1; parameters[table+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[table+1].DescriptorTable={1,&ranges[table]}; }
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=3; desc.pParameters=parameters; ComPtr<ID3DBlob> binary,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize virtual shadow request root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(VirtualShadowRequestRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create virtual shadow request root",status);
    }
    if (!LoadTechniqueShaderSource("DX12","VirtualShadowMaps.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSClear","CSMark"};
    for (unsigned stage=0;stage<VirtualShadowRequestPipelines.size();++stage) if (!VirtualShadowRequestPipelines[stage])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[stage],"cs_5_0",code,nullptr,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=VirtualShadowRequestRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(VirtualShadowRequestPipelines[stage].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create virtual shadow request pipeline",status);
    }
    auto work=std::make_shared<VirtualShadowRequestWork>();
    const auto depthCount=std::size_t(upload.Frame.Width)*upload.Frame.Height;
    const auto outputBytes=std::size_t(constants.Limits.Y)*sizeof(std::uint32_t);
    work->Depth=nativeDepth;
    if ((!nativeDepth && !CreateBufferResource(depthCount*sizeof(float),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work->Depth)) ||
        !CreateBufferResource(outputBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Readback)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; HRESULT status=S_OK;
    if (!nativeDepth)
    {
        status=work->Depth->Map(0,&noRead,&mapped);
        if (FAILED(status)) return FailHRESULT("Map virtual receiver upload",status);
        std::memcpy(mapped,upload.Depth.data(),depthCount*sizeof(float)); work->Depth->Unmap(0,nullptr);
    }
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); auto desc=BufferResourceDesc(outputBytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(work->Demand.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create virtual shadow demand buffer",status);
    D3D12_DESCRIPTOR_HEAP_DESC viewHeap{}; viewHeap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; viewHeap.NumDescriptors=2; viewHeap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    status=Device->CreateDescriptorHeap(&viewHeap,IID_PPV_ARGS(work->Views.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create virtual shadow demand views",status);
    const auto increment=Device->GetDescriptorHandleIncrementSize(viewHeap.Type); auto cpu=work->Views->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC source{}; source.ViewDimension=D3D12_SRV_DIMENSION_BUFFER; source.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    source.Buffer.NumElements=static_cast<UINT>(depthCount); source.Buffer.StructureByteStride=sizeof(float); Device->CreateShaderResourceView(work->Depth.Get(),&source,cpu); cpu.ptr+=increment;
    D3D12_UNORDERED_ACCESS_VIEW_DESC target{}; target.ViewDimension=D3D12_UAV_DIMENSION_BUFFER; target.Buffer.NumElements=constants.Limits.Y; target.Buffer.StructureByteStride=sizeof(std::uint32_t);
    Device->CreateUnorderedAccessView(work->Demand.Get(),nullptr,&target,cpu);
    status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work->Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work->Allocator.Get(),nullptr,IID_PPV_ARGS(work->Commands.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create virtual shadow demand commands",status);
    auto* commands=work->Commands.Get(); auto* views=work->Views.Get(); auto gpu=views->GetGPUDescriptorHandleForHeapStart();
    commands->SetDescriptorHeaps(1,&views); commands->SetComputeRootSignature(VirtualShadowRequestRoot.Get());
    commands->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0); commands->SetComputeRootDescriptorTable(1,gpu); gpu.ptr+=increment; commands->SetComputeRootDescriptorTable(2,gpu);
    const unsigned counts[]{constants.Limits.Y,constants.Extent.X*constants.Extent.Y};
    for (unsigned stage=0;stage<VirtualShadowRequestPipelines.size();++stage)
    {
        if (stage) { D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=work->Demand.Get(); commands->ResourceBarrier(1,&barrier); }
        const auto groups=(counts[stage]+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
        commands->SetPipelineState(VirtualShadowRequestPipelines[stage].Get()); commands->Dispatch(std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1);
    }
    D3D12_RESOURCE_BARRIER completed{}; completed.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    completed.Transition={work->Demand.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE}; commands->ResourceBarrier(1,&completed);
    commands->CopyResource(work->Readback.Get(),work->Demand.Get()); status=commands->Close(); if (FAILED(status)) return FailHRESULT("Close virtual shadow demand",status);
    VirtualShadowRequestPending=work; ID3D12CommandList* lists[]{commands}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false;
    VirtualShadowRequestPending.reset(); const D3D12_RANGE read{0,outputBytes}; status=work->Readback->Map(0,&read,&mapped);
    if (FAILED(status)) return FailHRESULT("Map virtual shadow demand",status);
    VirtualShadowRequestResult result; result.Pages.resize(constants.Limits.Y); std::memcpy(result.Pages.data(),mapped,outputBytes); const D3D12_RANGE noWrite{0,0}; work->Readback->Unmap(0,&noWrite);
    VirtualShadowLayout layout; if (!BuildVirtualShadowLayout(upload.Frame.Settings,layout,LastError) || !ValidateVirtualShadowRequests(layout,result.Pages,LastError)) return false;
    output=std::move(result); return true;
}
