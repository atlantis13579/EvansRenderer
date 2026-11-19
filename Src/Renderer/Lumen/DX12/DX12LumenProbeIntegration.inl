struct LumenProbeRecord { LumenProbeInfo Info; LumenRadianceHandle SourceHandle; ComPtr<ID3D12Resource> Buffer; };
struct LumenProbeWork
{
    ComPtr<ID3D12CommandAllocator> Allocator; ComPtr<ID3D12GraphicsCommandList> Commands; ComPtr<ID3D12Resource> Readback;
    std::shared_ptr<LumenProbeRecord> Record; std::shared_ptr<LumenRadianceRecord> Source;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenProbeRecord>> LumenProbes;
std::shared_ptr<LumenProbeWork> LumenProbePending;
std::vector<std::shared_ptr<LumenProbeWork>> LumenFrameProbeWork;
ComPtr<ID3D12RootSignature> LumenProbeRoot; ComPtr<ID3D12PipelineState> LumenProbeIntegrate;
void DestroyLumenProbes() { DestroyLumenGather(); FrameLumen.Finish(false,LumenRadianceTraces,LumenProbes); LumenProbePending.reset(); LumenFrameProbeWork.clear(); LumenProbes.clear(); LumenProbeIntegrate.Reset(); LumenProbeRoot.Reset(); }
bool EnsureLumenProbeIntegration()
{
    if (!LumenProbeRoot)
    {
        D3D12_ROOT_PARAMETER parameters[4]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.Num32BitValues=sizeof(LumenProbeIntegrationFrame)/sizeof(std::uint32_t);
        parameters[1].ParameterType=parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=1;
        parameters[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=4; desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen probe root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenProbeRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen probe root",status);
    }
    if (LumenProbeIntegrate) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_PROBE_INTEGRATE_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSIntegrateProbes","cs_5_0",binary,macros,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenProbeRoot.Get(); desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(LumenProbeIntegrate.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen probe integration",status);
}
bool BeginLumenProbeWork(LumenProbeWork& work)
{
    if (FrameOpen) { work.Commands=CommandList; return true; }
    auto status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work.Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work.Allocator.Get(),nullptr,IID_PPV_ARGS(work.Commands.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen probe commands",status);
}
bool SubmitLumenProbeWork(const std::shared_ptr<LumenProbeWork>& work)
{
    if (FrameOpen) { LumenFrameProbeWork.push_back(work); return true; }
    const auto status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close Lumen probe commands",status);
    LumenProbePending=work; ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false; LumenProbePending.reset(); return true;
}
void LumenProbeBarrier(LumenProbeWork& work,ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; work.Commands->ResourceBarrier(1,&barrier);
}
bool ReadLumenProbeRecord(const std::shared_ptr<LumenProbeRecord>& record,std::vector<LumenProbeIrradiance>& output)
{
    if (!RecoverSubmittedWork("Read Lumen probes")) return false; LumenProbePending.reset(); auto work=std::make_shared<LumenProbeWork>(); work->Record=record;
    const auto count=std::size_t(LumenRadianceProbeCount(record->Info.Source)),bytes=count*sizeof(LumenProbeIrradiance);
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Readback) || !BeginLumenProbeWork(*work)) return false;
    LumenProbeBarrier(*work,record->Buffer.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
    work->Commands->CopyBufferRegion(work->Readback.Get(),0,record->Buffer.Get(),0,bytes);
    LumenProbeBarrier(*work,record->Buffer.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenProbeWork(work)) return false;
    void* mapped=nullptr; const D3D12_RANGE read{0,bytes}; const auto status=work->Readback->Map(0,&read,&mapped);
    if (FAILED(status)) return FailHRESULT("Read Lumen probe staging",status);
    std::vector<LumenProbeIrradiance> probes(count); std::memcpy(probes.data(),mapped,bytes); const D3D12_RANGE noWrite{0,0}; work->Readback->Unmap(0,&noWrite);
    if (!ValidateLumenProbeIrradiance(record->Info,probes,LastError)) return false; output=std::move(probes); return true;
}
bool IntegrateLumenProbes(LumenRadianceHandle sourceHandle,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    const auto found=LumenRadianceTraces.find(sourceHandle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen probe ray source");
    const auto source=found->second; LumenProbeIntegrationFrame frame;
    if (!BuildLumenProbeIntegrationFrame(source->Info,upload,frame,LastError) || !RecoverSubmittedWork("Integrate Lumen probes") || (FrameOpen && !FrameLumen.CanIntegrate(frame.Counts.X,LastError)) || !EnsureLumenProbeIntegration()) return false;
    LumenProbePending.reset(); auto work=std::make_shared<LumenProbeWork>(); auto record=std::make_shared<LumenProbeRecord>();
    work->Source=source; work->Record=record; record->Info={source->Info,upload.SkyRadiance,upload.Environment}; record->SourceHandle=sourceHandle;
    if (!CreateLumenRadianceOutput(frame.Counts.X*sizeof(LumenProbeIrradiance),record->Buffer) || !BeginLumenProbeWork(*work)) return false;
    work->Commands->SetPipelineState(LumenProbeIntegrate.Get()); work->Commands->SetComputeRootSignature(LumenProbeRoot.Get());
    work->Commands->SetComputeRoot32BitConstants(0,sizeof(frame)/sizeof(std::uint32_t),&frame,0);
    work->Commands->SetComputeRootShaderResourceView(1,source->Buffer->GetGPUVirtualAddress()); work->Commands->SetComputeRootShaderResourceView(2,source->Origins->GetGPUVirtualAddress());
    work->Commands->SetComputeRootUnorderedAccessView(3,record->Buffer->GetGPUVirtualAddress()); work->Commands->Dispatch((frame.Counts.X+LumenProbeThreads-1)/LumenProbeThreads,1,1);
    LumenProbeBarrier(*work,record->Buffer.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!SubmitLumenProbeWork(work)) return false;
    LumenProbeIntegrationResult result; result.Info=record->Info; if (upload.Readback && !ReadLumenProbeRecord(record,result.Probes)) return false;
    result.Handle=NewLumenProbeHandle(); if (!result.Handle) return Fail("Lumen probe handle space exhausted");
    LumenProbes.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,frame.Counts.X);
    output=std::move(result); return true;
}
bool ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    const auto found=LumenProbes.find(handle.Value); if (found==LumenProbes.end()) return Fail("Unknown Lumen irradiance probe cache");
    return ReadLumenProbeRecord(found->second,output);
}
bool DestroyLumenProbes(LumenProbeHandle handle) { return LumenProbes.erase(handle.Value)!=0 || Fail("Unknown Lumen irradiance probe cache"); }

// Handles are usable by later GPU work in this frame, then either published or
// cancelled together. Native command ownership is independent of publication.
bool LumenCommandsEnded() const { return FrameWasSubmitted || SubmittedWorkPending; }
bool ValidateLumenRecording(bool readback)
{
    if (PassOpen || (FrameOpen && (readback || LumenCommandsEnded())))
        return Fail("Lumen frame work requires an open command stream, no graphics pass and no CPU readback");
    return true;
}
void FinishLumenFrame(bool publish)
{
    LumenHistory.Finish(publish); FinishLumenReflections(publish);
    FrameLumen.Finish(publish,LumenRadianceTraces,LumenProbes);
    ReleaseLumenGIResults();
    if (!SubmittedWorkPending) { LumenFrameProbeWork.clear(); LumenFrameRadianceWork.clear(); }
}
struct LumenPresentPublication
{
    Impl& Owner; bool Published=false;
    ~LumenPresentPublication()
    { if (Published || !Owner.FrameOpen || Owner.LumenCommandsEnded()) Owner.FinishLumenFrame(Published); }
};
