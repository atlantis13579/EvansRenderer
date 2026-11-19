struct NaniteFrameBuffer
{
    ComPtr<ID3D11Buffer> Buffer; ComPtr<ID3D11ShaderResourceView> View; ComPtr<ID3D11UnorderedAccessView> Target;
    unsigned Words=0;
};
struct NaniteDepthRecord { NaniteDepthInfo Info; std::shared_ptr<NaniteFrameBuffer> Data; };
struct NaniteVisibilityRecord
{
    NaniteFrameVisibilityInfo Info; std::uint64_t ConstantAlignmentPadding=0; NaniteVisibilityConstants Constants;
    std::shared_ptr<NaniteFrameBuffer> Queries,Data;
    std::shared_ptr<const std::vector<NaniteVisibilityQuery>> QueryRecords;
    std::shared_ptr<const std::vector<NaniteClusterDraw>> Draws;
};
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteDepthRecord>> NaniteDepths;
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteVisibilityRecord>> NaniteDecisions;
NaniteVisibilityTransaction FrameNanite;
std::array<ComPtr<ID3D11ComputeShader>,7> NaniteFrameShaders; // Reduce, native depth capture, native cull, indirect arguments.
ComPtr<ID3D11Buffer> NaniteFrameUniform;
std::shared_ptr<NaniteFrameBuffer> NaniteFrameDummy;
bool NaniteCommandsEnded() const { return false; }
bool ValidateNaniteRecording(bool rasterInternal=false)
{ return (FrameOpen && (!PassOpen || (rasterInternal && CurrentPass.Type==PassType::GBuffer))) || Fail("Nanite GPU frame work requires an open frame between graphics passes"); }
void FinishNaniteFrame(bool publish) { FrameNanite.Finish(publish,NaniteDepths,NaniteDecisions); FinishNaniteDraws(); }
struct NanitePresentPublication
{
    Impl& Owner; bool Published=false;
    ~NanitePresentPublication() { if (Published || !Owner.FrameOpen) Owner.FinishNaniteFrame(Published); }
};
void DestroyNaniteFrameVisibility()
{
    FinishNaniteFrame(false); NaniteDepths.clear(); NaniteDecisions.clear(); NaniteFrameDummy.reset(); NaniteFrameUniform.Reset();
    for (auto& shader:NaniteFrameShaders) shader.Reset();
}
bool CreateNaniteFrameBuffer(unsigned words,const void* initial,bool writable,std::shared_ptr<NaniteFrameBuffer>& output,bool indirect=false,bool indexed=false)
{
    auto buffer=std::make_shared<NaniteFrameBuffer>(); buffer->Words=words;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=words*sizeof(unsigned); desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|(writable?D3D11_BIND_UNORDERED_ACCESS:0)|(indexed?D3D11_BIND_INDEX_BUFFER:0); desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS|(indirect?D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS:0);
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=initial;
    auto status=Device->CreateBuffer(&desc,initial?&data:nullptr,buffer->Buffer.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame buffer",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32_TYPELESS; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;
    view.BufferEx.NumElements=words; view.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
    status=Device->CreateShaderResourceView(buffer->Buffer.Get(),&view,buffer->View.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame SRV",status));
    if (writable)
    {
        D3D11_UNORDERED_ACCESS_VIEW_DESC target{}; target.Format=DXGI_FORMAT_R32_TYPELESS; target.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
        target.Buffer.NumElements=words; target.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
        status=Device->CreateUnorderedAccessView(buffer->Buffer.Get(),&target,buffer->Target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame UAV",status));
    }
    output=std::move(buffer); return true;
}
bool EnsureNaniteFrameVisibility()
{
    if (!NaniteFrameDummy) { const unsigned clear=0; if (!CreateNaniteFrameBuffer(1,&clear,false,NaniteFrameDummy)) return false; }
    if (!NaniteFrameUniform)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(NaniteVisibilityConstants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        const auto status=Device->CreateBuffer(&desc,nullptr,NaniteFrameUniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame constants",status));
    }
    if (!LoadTechniqueShaderSource("DX11","NaniteVisibility.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSReduce","CSCapture","CSCull","CSBuildDraws","CSCountBins","CSScanBins","CSScatterBins"}; constexpr const char* definitions[]{nullptr,"EVANS_NANITE_CAPTURE_DEPTH","EVANS_NANITE_FRAME_CULL",nullptr,nullptr,nullptr,nullptr};
    for (unsigned pass=0;pass<NaniteFrameShaders.size();++pass) if (!NaniteFrameShaders[pass])
    {
        const D3D_SHADER_MACRO macros[]{{definitions[pass],"1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> code;
        if (pass>=3 && !LoadTechniqueShaderSource("DX11","NaniteDraws.hlsl",ShaderCode,LastError)) return false;
        if (!CompileShader(entries[pass],"cs_5_0",code,definitions[pass]?macros:nullptr,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,NaniteFrameShaders[pass].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame kernel",status));
    }
    return true;
}
void DispatchNaniteFrame(unsigned pass,const NaniteVisibilityConstants& constants,unsigned groupsX,unsigned groupsY,
    ID3D11ShaderResourceView* source,ID3D11ShaderResourceView* depth,ID3D11ShaderResourceView* main,ID3D11UnorderedAccessView* output)
{
    Context->OMSetRenderTargets(0,nullptr,nullptr);
    ID3D11ShaderResourceView* inputs[]{source,depth,main}; auto* uniform=NaniteFrameUniform.Get();
    Context->UpdateSubresource(uniform,0,nullptr,&constants,0,0); Context->CSSetConstantBuffers(0,1,&uniform);
    Context->CSSetShaderResources(0,3,inputs); Context->CSSetUnorderedAccessViews(0,1,&output,nullptr);
    Context->CSSetShader(NaniteFrameShaders[pass].Get(),nullptr,0); Context->Dispatch(groupsX,groupsY,1);
    inputs[0]=inputs[1]=inputs[2]=nullptr; output=nullptr; uniform=nullptr;
    Context->CSSetShaderResources(0,3,inputs); Context->CSSetUnorderedAccessViews(0,1,&output,nullptr); Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0);
}
bool CaptureNaniteDepth(NaniteDepthHandle& output,bool rasterInternal=false)
{
    NaniteVisibilityWork work;
    if (!ValidateNaniteRecording(rasterInternal) || FrameSampleCount!=1 || !DepthShaderView || !NanitePrimaryDepthInitialized(PendingFrame.Passes)) return Fail("Nanite depth capture requires initialized single-sample camera depth");
    if (!BuildNaniteDepthWork(CreateInfo.Width,CreateInfo.Height,work,LastError) || !FrameNanite.CanAllocate(std::size_t(work.OutputWords)*sizeof(unsigned),true,LastError) || !EnsureNaniteFrameVisibility()) return false;
    auto record=std::make_shared<NaniteDepthRecord>();
    if (!CreateNaniteFrameBuffer(work.OutputWords,nullptr,true,record->Data)) return false;
    record->Info={FrameNanite.Frame(),unsigned(CreateInfo.Width),unsigned(CreateInfo.Height),work.OutputWords,false};
    for (const auto& job:work.Dispatches)
        DispatchNaniteFrame(job.Pass==2?1:0,job.Constants,job.GroupsX,job.GroupsY,job.Pass==2?DepthShaderView.Get():NaniteFrameDummy->View.Get(),nullptr,nullptr,record->Data->Target.Get());
    NaniteDepthHandle handle{NewNaniteVisibilityIdentity()}; NaniteDepths.emplace(handle.Value,record); FrameNanite.Record(handle,std::size_t(work.OutputWords)*sizeof(unsigned)); output=handle; return true;
}
bool BeginNaniteVisibility(NaniteDepthHandle history,const std::vector<NaniteVisibilityQuery>& queries,NaniteVisibilityHandle& output)
{
    if (!ValidateNaniteRecording()) return false;
    std::shared_ptr<NaniteDepthRecord> depth;
    if (history) { const auto found=NaniteDepths.find(history.Value); if (found==NaniteDepths.end() || !found->second->Info.Published) return Fail("Nanite history must be a committed depth snapshot"); depth=found->second; }
    NaniteVisibilityWork work; if (!BuildNaniteFrameCull(queries,depth?depth->Info.Width:CreateInfo.Width,depth?depth->Info.Height:CreateInfo.Height,NaniteVisibilityPhase::Main,work,LastError)) return false;
    const auto bytes=(work.Input.size()+work.OutputWords)*sizeof(unsigned)+queries.size()*sizeof(NaniteVisibilityQuery);
    if (!FrameNanite.CanAllocate(bytes,false,LastError) || !EnsureNaniteFrameVisibility()) return false;
    auto record=std::make_shared<NaniteVisibilityRecord>();
    if (!CreateNaniteFrameBuffer(unsigned(work.Input.size()),work.Input.data(),false,record->Queries) || !CreateNaniteFrameBuffer(work.OutputWords,nullptr,true,record->Data)) return false;
    const auto& job=work.Dispatches.front(); record->Constants=job.Constants; record->Constants.Layout.W=depth?1:0;
    record->Info={FrameNanite.Frame(),work.OutputWords,NaniteVisibilityPhase::Main,false};
    record->QueryRecords=std::make_shared<const std::vector<NaniteVisibilityQuery>>(queries);
    DispatchNaniteFrame(2,record->Constants,job.GroupsX,job.GroupsY,record->Queries->View.Get(),depth?depth->Data->View.Get():NaniteFrameDummy->View.Get(),NaniteFrameDummy->View.Get(),record->Data->Target.Get());
    NaniteVisibilityHandle handle{NewNaniteVisibilityIdentity()}; NaniteDecisions.emplace(handle.Value,record); FrameNanite.Record(handle,bytes); output=handle; return true;
}
bool RetestNaniteVisibility(NaniteVisibilityHandle main,NaniteDepthHandle camera,NaniteVisibilityHandle& output,bool rasterInternal=false)
{
    if (!ValidateNaniteRecording(rasterInternal)) return false;
    const auto m=NaniteDecisions.find(main.Value); const auto d=NaniteDepths.find(camera.Value);
    if (m==NaniteDecisions.end() || d==NaniteDepths.end() || m->second->Info.Frame!=FrameNanite.CurrentFrame() || d->second->Info.Frame!=FrameNanite.CurrentFrame() ||
        m->second->Info.Phase!=NaniteVisibilityPhase::Main || m->second->Info.Published || d->second->Info.Published) return Fail("Nanite retest requires this frame's main decisions and camera depth");
    if (m->second->Draws && !NaniteMainDrawsComplete(m->second->Draws)) return Fail("Submit all main Nanite candidates before retesting");
    const auto bytes=std::size_t(m->second->Info.QueryCount)*sizeof(unsigned); if (!FrameNanite.CanAllocate(bytes,false,LastError)) return false;
    auto record=std::make_shared<NaniteVisibilityRecord>(); record->Info=m->second->Info; record->Info.Phase=NaniteVisibilityPhase::Post; record->Queries=m->second->Queries; record->QueryRecords=m->second->QueryRecords; record->Draws=m->second->Draws;
    if (!CreateNaniteFrameBuffer(record->Info.QueryCount,nullptr,true,record->Data)) return false;
    record->Constants=m->second->Constants; record->Constants.Counts.Y=d->second->Info.Width; record->Constants.Counts.Z=d->second->Info.Height; record->Constants.Counts.W=1; record->Constants.Layout.W=1;
    const unsigned groups=(record->Info.QueryCount+NaniteVisibilityThreads-1)/NaniteVisibilityThreads,gx=record->Constants.Target.W;
    DispatchNaniteFrame(2,record->Constants,gx,(groups+gx-1)/gx,record->Queries->View.Get(),d->second->Data->View.Get(),m->second->Data->View.Get(),record->Data->Target.Get());
    NaniteVisibilityHandle handle{NewNaniteVisibilityIdentity()}; NaniteDecisions.emplace(handle.Value,record); FrameNanite.Record(handle,bytes); output=handle; return true;
}
bool ReadNaniteFrameBuffer(const NaniteFrameBuffer& buffer,std::vector<unsigned>& words)
{
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=buffer.Words*sizeof(unsigned); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> staging; auto status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Nanite frame readback",status));
    Context->CopyResource(staging.Get(),buffer.Buffer.Get()); D3D11_MAPPED_SUBRESOURCE mapped{}; status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped);
    if (FAILED(status)) return Fail(HResultMessage("Read Nanite frame feedback",status));
    words.resize(buffer.Words); std::memcpy(words.data(),mapped.pData,desc.ByteWidth); Context->Unmap(staging.Get(),0); return true;
}
