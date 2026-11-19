ComPtr<ID3D12RootSignature> LumenGlobalRoot;
ComPtr<ID3D12PipelineState> LumenGlobalCompose;
void DestroyLumenGlobalDistance() { DestroyLumenGlobalDistanceAsync(); LumenGlobalCompose.Reset(); LumenGlobalRoot.Reset(); }
bool EnsureLumenGlobalDistance()
{
    if (!LumenGlobalRoot)
    {
        D3D12_ROOT_PARAMETER parameters[4]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[1].ParameterType=parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=1;
        parameters[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=4; desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize global distance root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenGlobalRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create global distance root",status);
    }
    if (LumenGlobalCompose) return true;
    if (!LoadTechniqueShaderSource("DX12","LumenGlobalDistance.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> binary; if (!CompileShader("CSCompose","cs_5_0",binary,nullptr,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenGlobalRoot.Get(); desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(LumenGlobalCompose.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create global distance composition",status);
}
bool CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source scene");
    std::shared_ptr<LumenDistanceRecord> retained;
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second)
            return Fail("Global distance reuse requires the exact immutable source scene");
        retained=old->second;
    }
    auto plan=std::make_shared<LumenGlobalDistancePlan>();
    if (!BuildLumenGlobalDistancePlan(found->second->Info,settings,*plan,LastError) || !RecoverSubmittedWork("Compose global distance scene") || !EnsureLumenGlobalDistance()) return false;
    LumenGlobalDistanceUpdate update;
    if (!BuildLumenGlobalDistanceUpdate(found->second->Info,*plan,retained?retained->Info.Global.get():nullptr,update,LastError)) return false;
    auto frame=plan->Frame; frame.Dispatch={0,unsigned(update.Compose.size()),0,0};
    LumenDistancePending.reset(); auto work=std::make_shared<LumenDistanceWork>(); work->Record=std::make_shared<LumenDistanceRecord>();
    auto& record=*work->Record; record.Base=found->second; record.Info=record.Base->Info; record.Info.Global=plan; record.Hardware=record.Base->Hardware;
    work->CopySource=retained;
    std::vector<std::uint8_t> upload(sizeof(frame)+std::max<std::size_t>(1,update.Compose.size())*sizeof(std::uint32_t));
    std::memcpy(upload.data(),&frame,sizeof(frame));
    if (!update.Compose.empty()) std::memcpy(upload.data()+sizeof(frame),update.Compose.data(),update.Compose.size()*sizeof(std::uint32_t));
    if (!BeginLumenDistanceWork(*work,upload.data(),upload.size())) return false;
    auto desc=BufferResourceDesc(std::size_t(plan->WordCount)*sizeof(std::uint32_t)); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    const auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(record.Scene.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Allocate global distance scene",status);
    auto* commands=work->Commands.Get();
    const auto transition=[&](ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; commands->ResourceBarrier(1,&barrier);
    };
    // Buffer resources from completed submissions have decayed to COMMON.
    transition(record.Base->Scene.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
    commands->CopyBufferRegion(record.Scene.Get(),0,record.Base->Scene.Get(),0,std::max(1u,record.Info.Counts.W)*sizeof(std::uint32_t));
    commands->CopyBufferRegion(record.Scene.Get(),plan->Frame.Header.Layout.W*sizeof(std::uint32_t),work->Upload.Get(),offsetof(LumenGlobalDistanceFrame,Header),sizeof(LumenGlobalDistanceHeader));
    if (!update.Copies.empty())
    {
        transition(retained->Scene.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        for (const auto& copy:update.Copies)
            commands->CopyBufferRegion(record.Scene.Get(),std::size_t(copy.DestinationWord)*sizeof(std::uint32_t),retained->Scene.Get(),
                std::size_t(copy.SourceWord)*sizeof(std::uint32_t),std::size_t(copy.WordCount)*sizeof(std::uint32_t));
        transition(retained->Scene.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    transition(record.Base->Scene.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(record.Scene.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    commands->SetPipelineState(LumenGlobalCompose.Get()); commands->SetComputeRootSignature(LumenGlobalRoot.Get());
    commands->SetComputeRootConstantBufferView(0,work->Upload->GetGPUVirtualAddress());
    commands->SetComputeRootShaderResourceView(1,record.Base->Scene->GetGPUVirtualAddress());
    commands->SetComputeRootShaderResourceView(2,work->Upload->GetGPUVirtualAddress()+sizeof(frame));
    commands->SetComputeRootUnorderedAccessView(3,record.Scene->GetGPUVirtualAddress());
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    if (groups) commands->Dispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
    transition(record.Scene.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenDistanceWork(work)) return false;
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,work->Record); output=handle; return true;
}
