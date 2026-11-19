struct LumenHardwareRecord { ComPtr<ID3D12Resource> Top,Words; std::vector<ComPtr<ID3D12Resource>> Bottom; };
struct LumenDistanceRecord { LumenDistanceSceneInfo Info; std::shared_ptr<LumenDistanceRecord> Base; ComPtr<ID3D12Resource> Scene; std::shared_ptr<LumenHardwareRecord> Hardware; };
struct LumenDistanceWork
{
    ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands;
    ComPtr<ID3D12Resource> Upload,Feedback,Readback; std::shared_ptr<LumenDistanceRecord> Record,CopySource;
    std::vector<ComPtr<ID3D12Resource>> HardwareInputs; // Retained through the build fence, including failed waits.
};
#include "Renderer/Lumen/DX12/DX12LumenHardwareRayTracing.inl"
std::unordered_map<std::uint64_t,std::shared_ptr<LumenDistanceRecord>> LumenDistanceScenes;
std::shared_ptr<LumenDistanceWork> LumenDistancePending;
ComPtr<ID3D12RootSignature> LumenDistanceRoot; ComPtr<ID3D12PipelineState> LumenDistanceTrace;
void DestroyLumenDistanceScenes()
{ LumenDistancePending.reset(); LumenDistanceScenes.clear(); DestroyLumenGlobalDistance(); LumenHardwarePipelines={}; LumenDistanceTrace.Reset(); LumenDistanceRoot.Reset(); }
bool EnsureLumenDistanceTrace(bool hardware=false)
{
    if (!LumenDistanceRoot)
    {
        D3D12_ROOT_PARAMETER parameters[6]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants.Num32BitValues=sizeof(LumenDistanceTraceFrame)/sizeof(std::uint32_t);
        parameters[1].ParameterType=parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=1;
        parameters[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        LumenHardwareRoots(parameters+4);
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=6; desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen distance root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenDistanceRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen distance root",status);
    }
    if (hardware) return EnsureLumenHardwarePipeline(LumenHardwareProgram::Distance,LumenDistanceRoot.Get());
    if (LumenDistanceTrace) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_DISTANCE_TRACE_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSTraceDistance","cs_5_0",binary,macros,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenDistanceRoot.Get(); desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(LumenDistanceTrace.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen distance compute pipeline",status);
}
bool BeginLumenDistanceWork(LumenDistanceWork& work,const void* bytes,std::size_t size)
{
    auto status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work.Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work.Allocator.Get(),nullptr,IID_PPV_ARGS(work.Commands.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen distance command list",status);
    if (!CreateBufferResource(size,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work.Upload)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; status=work.Upload->Map(0,&noRead,&mapped);
    if (FAILED(status)) return FailHRESULT("Map Lumen distance upload",status);
    std::memcpy(mapped,bytes,size); work.Upload->Unmap(0,nullptr); return true;
}
bool SubmitLumenDistanceWork(const std::shared_ptr<LumenDistanceWork>& work)
{
    const auto status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close Lumen distance commands",status);
    // Keep every referenced allocation alive if a fence wait fails after submit.
    LumenDistancePending=work; ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false;
    LumenDistancePending.reset(); return true;
}
bool CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!ValidateLumenDistanceScene(data,LastError) || !RecoverSubmittedWork("Upload Lumen distance scene") || !EnsureLumenDistanceTrace()) return false;
    LumenDistancePending.reset(); auto work=std::make_shared<LumenDistanceWork>(); work->Record=std::make_shared<LumenDistanceRecord>(); work->Record->Info=data.Info;
    const std::uint32_t empty=0; const auto bytes=std::max<std::size_t>(1,data.Words.size())*sizeof(std::uint32_t);
    if (!BeginLumenDistanceWork(*work,data.Words.empty()?&empty:data.Words.data(),bytes) ||
        !CreateBufferResource(bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST,work->Record->Scene)) return false;
    work->Commands->CopyBufferRegion(work->Record->Scene.Get(),0,work->Upload.Get(),0,bytes);
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={work->Record->Scene.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    work->Commands->ResourceBarrier(1,&barrier);
    if (!BuildLumenHardwareAcceleration(*work) || !SubmitLumenDistanceWork(work)) return false;
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,work->Record); output=handle; return true;
}
bool TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    const auto found=LumenDistanceScenes.find(handle.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown Lumen distance scene");
    LumenDistanceTraceFrame frame; if (!BuildLumenDistanceTraceFrame(found->second->Info,upload,frame,LastError) ||
        !RecoverSubmittedWork("Trace Lumen distance scene") || !EnsureLumenDistanceTrace(bool(found->second->Hardware))) return false;
    LumenDistancePending.reset(); auto work=std::make_shared<LumenDistanceWork>(); work->Record=found->second;
    if (!BeginLumenDistanceWork(*work,upload.Rays.data(),upload.Rays.size()*sizeof(LumenDistanceRay))) return false;
    const auto bytes=upload.Rays.size()*sizeof(LumenDistanceHit); const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(work->Feedback.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen distance feedback",status);
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Readback)) return false;
    work->Commands->SetPipelineState((work->Record->Hardware?LumenHardwarePipelines[unsigned(LumenHardwareProgram::Distance)]:LumenDistanceTrace).Get()); work->Commands->SetComputeRootSignature(LumenDistanceRoot.Get());
    work->Commands->SetComputeRoot32BitConstants(0,sizeof(frame)/sizeof(std::uint32_t),&frame,0);
    // Completed buffer submissions decay to COMMON; read-only SRV use promotes
    // the immutable scene without changing its contents or requiring a heap.
    work->Commands->SetComputeRootShaderResourceView(1,work->Record->Scene->GetGPUVirtualAddress());
    work->Commands->SetComputeRootShaderResourceView(2,work->Upload->GetGPUVirtualAddress());
    work->Commands->SetComputeRootUnorderedAccessView(3,work->Feedback->GetGPUVirtualAddress());
    BindLumenHardware(work->Commands.Get(),*work->Record,4);
    const auto groups=unsigned((upload.Rays.size()+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads);
    work->Commands->Dispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={work->Feedback.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
    work->Commands->ResourceBarrier(1,&barrier); work->Commands->CopyBufferRegion(work->Readback.Get(),0,work->Feedback.Get(),0,bytes);
    LumenDistanceTraceResult result; result.Identity=work->Record->Info.Identity; result.Hits.resize(upload.Rays.size()); result.HardwareTracing=bool(work->Record->Hardware);
    if (!SubmitLumenDistanceWork(work)) return false;
    void* mapped=nullptr; const D3D12_RANGE range{0,bytes}; status=work->Readback->Map(0,&range,&mapped);
    if (FAILED(status)) return FailHRESULT("Map Lumen distance feedback",status);
    std::memcpy(result.Hits.data(),mapped,bytes); const D3D12_RANGE noWrite{0,0}; work->Readback->Unmap(0,&noWrite);
    if (!ValidateLumenDistanceTraceResult(work->Record->Info,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
bool DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{
    if (!RecoverSubmittedWork("Release Lumen distance scene")) return false;
    LumenDistancePending.reset(); return LumenDistanceScenes.erase(handle.Value)!=0 || Fail("Unknown Lumen distance scene");
}
