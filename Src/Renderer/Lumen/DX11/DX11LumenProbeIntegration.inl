struct LumenProbeRecord
{
    LumenProbeInfo Info; LumenRadianceHandle SourceHandle; ComPtr<ID3D11Buffer> Buffer; ComPtr<ID3D11ShaderResourceView> View;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenProbeRecord>> LumenProbes;
ComPtr<ID3D11ComputeShader> LumenProbeIntegrate;
void DestroyLumenProbes() { DestroyLumenGather(); FrameLumen.Finish(false,LumenRadianceTraces,LumenProbes); LumenProbes.clear(); LumenProbeIntegrate.Reset(); }
bool EnsureLumenProbeIntegration()
{
    if (LumenProbeIntegrate) return true;
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_PROBE_INTEGRATE_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSIntegrateProbes","cs_5_0",binary,macros,true)) return false;
    const auto status=Device->CreateComputeShader(binary->GetBufferPointer(),binary->GetBufferSize(),nullptr,LumenProbeIntegrate.GetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create Lumen probe integration",status));
}
bool ReadLumenProbeRecord(const LumenProbeRecord& record,std::vector<LumenProbeIrradiance>& output)
{
    const auto count=std::size_t(LumenRadianceProbeCount(record.Info.Source)),bytes=count*sizeof(LumenProbeIrradiance); D3D11_BUFFER_DESC desc{};
    desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ; ComPtr<ID3D11Buffer> staging;
    auto status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen probe staging",status));
    Context->CopyResource(staging.Get(),record.Buffer.Get()); D3D11_MAPPED_SUBRESOURCE mapped{};
    status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped); if (FAILED(status)) return Fail(HResultMessage("Read Lumen probes",status));
    std::vector<LumenProbeIrradiance> probes(count); std::memcpy(probes.data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    if (!ValidateLumenProbeIrradiance(record.Info,probes,LastError)) return false;
    output=std::move(probes); return true;
}
bool IntegrateLumenProbes(LumenRadianceHandle sourceHandle,const LumenProbeIntegrationUpload& upload,LumenProbeIntegrationResult& output)
{
    const auto found=LumenRadianceTraces.find(sourceHandle.Value); if (found==LumenRadianceTraces.end()) return Fail("Unknown Lumen probe ray source");
    const auto source=found->second; LumenProbeIntegrationFrame frame;
    if (!BuildLumenProbeIntegrationFrame(source->Info,upload,frame,LastError) || (FrameOpen && !FrameLumen.CanIntegrate(frame.Counts.X,LastError)) || !EnsureLumenProbeIntegration()) return false;
    auto record=std::make_shared<LumenProbeRecord>(); record->Info={source->Info,upload.SkyRadiance,upload.Environment}; record->SourceHandle=sourceHandle;
    ComPtr<ID3D11Buffer> uniform; ComPtr<ID3D11UnorderedAccessView> target;
    if (!CreateLumenRadianceBuffer(frame.Counts.X*sizeof(LumenProbeIrradiance),sizeof(LumenProbeIrradiance),nullptr,record->Buffer,record->View,&target)) return false;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(frame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&frame; auto status=Device->CreateBuffer(&desc,&data,uniform.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen probe frame",status));
    ComPtr<ID3D11Query> completion;
    if (!upload.Readback && !FrameOpen) { const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0}; status=Device->CreateQuery(&query,completion.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen probe completion",status)); }
    auto* constants=uniform.Get(); auto* destination=target.Get(); ID3D11ShaderResourceView* views[]{source->View.Get(),source->OriginView.Get()};
    Context->CSSetShader(LumenProbeIntegrate.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&constants);
    Context->CSSetShaderResources(0,2,views); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr);
    Context->Dispatch((frame.Counts.X+LumenProbeThreads-1)/LumenProbeThreads,1,1);
    ID3D11ShaderResourceView* empty[2]{}; destination=nullptr; constants=nullptr;
    Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr); Context->CSSetShaderResources(0,2,empty); Context->CSSetConstantBuffers(0,1,&constants); Context->CSSetShader(nullptr,nullptr,0);
    LumenProbeIntegrationResult result; result.Info=record->Info;
    if (upload.Readback) { if (!ReadLumenProbeRecord(*record,result.Probes)) return false; }
    else if (!FrameOpen)
    {
        Context->End(completion.Get()); Context->Flush(); constexpr ULONGLONG TimeoutMilliseconds=10000; const auto deadline=GetTickCount64()+TimeoutMilliseconds;
        while ((status=Context->GetData(completion.Get(),nullptr,0,0))==S_FALSE)
        { if (GetTickCount64()>=deadline) return Fail("Lumen probe integration timed out"); std::this_thread::yield(); }
        if (FAILED(status)) return Fail(HResultMessage("Complete Lumen probes",status));
    }
    result.Handle=NewLumenProbeHandle(); if (!result.Handle) return Fail("Lumen probe handle space exhausted");
    LumenProbes.emplace(result.Handle.Value,std::move(record));
    if (FrameOpen) FrameLumen.Record(result.Handle,frame.Counts.X);
    output=std::move(result); return true;
}
bool ReadLumenProbes(LumenProbeHandle handle,std::vector<LumenProbeIrradiance>& output)
{
    const auto found=LumenProbes.find(handle.Value); if (found==LumenProbes.end()) return Fail("Unknown Lumen irradiance probe cache");
    return ReadLumenProbeRecord(*found->second,output);
}
bool DestroyLumenProbes(LumenProbeHandle handle) { return LumenProbes.erase(handle.Value)!=0 || Fail("Unknown Lumen irradiance probe cache"); }

// Handles are usable by later GPU work in this frame, then either published or
// cancelled together. Native command ownership is independent of publication.
bool LumenCommandsEnded() const { return false; }
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

}
struct LumenPresentPublication
{
    Impl& Owner; bool Published=false;
    ~LumenPresentPublication()
    { if (Published || !Owner.FrameOpen || Owner.LumenCommandsEnded()) Owner.FinishLumenFrame(Published); }
};
