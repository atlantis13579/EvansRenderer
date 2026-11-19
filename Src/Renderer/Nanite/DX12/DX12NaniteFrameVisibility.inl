struct NaniteReadbackResources
{
    ComPtr<ID3D12CommandAllocator> Allocator;
    ComPtr<ID3D12GraphicsCommandList> Commands;
    ComPtr<ID3D12Resource> Input,Output,Readback;
};
struct NaniteFrameBuffer
{
    ComPtr<ID3D12Resource> Buffer; unsigned Words=0;
    D3D12_RESOURCE_STATES State=D3D12_RESOURCE_STATE_COMMON;
};
struct NaniteDepthRecord { NaniteDepthInfo Info; std::shared_ptr<NaniteFrameBuffer> Data; };
struct NaniteVisibilityRecord
{
    NaniteFrameVisibilityInfo Info; std::uint64_t ConstantAlignmentPadding=0; NaniteVisibilityConstants Constants;
    std::shared_ptr<NaniteFrameBuffer> Queries,Data;
    std::shared_ptr<const std::vector<NaniteVisibilityQuery>> QueryRecords;
    std::shared_ptr<const std::vector<NaniteClusterDraw>> Draws;
};
struct NaniteFrameCommandResources
{
    ComPtr<ID3D12DescriptorHeap> Views; ComPtr<ID3D12Resource> Camera;
    std::array<std::shared_ptr<NaniteFrameBuffer>,4> Buffers;
};
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteDepthRecord>> NaniteDepths;
std::unordered_map<std::uint64_t,std::shared_ptr<NaniteVisibilityRecord>> NaniteDecisions;
NaniteVisibilityTransaction FrameNanite;
std::array<ComPtr<ID3D12RootSignature>,7> NaniteFrameRoots;
std::array<ComPtr<ID3D12PipelineState>,7> NaniteFrameShaders;
std::vector<std::shared_ptr<NaniteFrameCommandResources>> NaniteFrameWork;
std::shared_ptr<NaniteReadbackResources> NaniteFrameReadback;
std::shared_ptr<NaniteFrameBuffer> NaniteFrameDummy;
bool NaniteCommandsEnded() const { return FrameWasSubmitted || SubmittedWorkPending; }
bool ValidateNaniteRecording(bool rasterInternal=false)
{ return (FrameOpen && (!PassOpen || (rasterInternal && CurrentPass.Type==PassType::GBuffer)) && !NaniteCommandsEnded()) || Fail("Nanite GPU work requires an open command stream between graphics passes"); }
void FinishNaniteFrame(bool publish)
{ FrameNanite.Finish(publish,NaniteDepths,NaniteDecisions); FinishNaniteDraws(); if (!SubmittedWorkPending) NaniteFrameWork.clear(); }
struct NanitePresentPublication
{
    Impl& Owner; bool Published=false;
    ~NanitePresentPublication() { if (Published || !Owner.FrameOpen || Owner.NaniteCommandsEnded()) Owner.FinishNaniteFrame(Published); }
};
void DestroyNaniteFrameVisibility()
{
    FinishNaniteFrame(false); NaniteFrameReadback.reset(); NaniteFrameWork.clear(); NaniteDepths.clear(); NaniteDecisions.clear(); NaniteFrameDummy.reset();
    for (auto& shader:NaniteFrameShaders) shader.Reset(); for (auto& root:NaniteFrameRoots) root.Reset();
}
bool CreateNaniteFrameBuffer(unsigned words,const void* initial,bool writable,std::shared_ptr<NaniteFrameBuffer>& output)
{
    auto buffer=std::make_shared<NaniteFrameBuffer>(); buffer->Words=words; const auto bytes=std::size_t(words)*sizeof(unsigned);
    if (writable)
    {
        auto desc=BufferResourceDesc(bytes); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
        const auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(buffer->Buffer.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Nanite frame output",status);
    }
    else
    {
        if (!initial || !CreateBufferResource(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,buffer->Buffer)) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto status=buffer->Buffer->Map(0,&noRead,&mapped);
        if (FAILED(status)) return FailHRESULT("Map Nanite frame input",status); std::memcpy(mapped,initial,bytes); buffer->Buffer->Unmap(0,nullptr);
        buffer->State=D3D12_RESOURCE_STATE_GENERIC_READ;
    }
    output=std::move(buffer); return true;
}
bool EnsureNaniteFrameVisibility()
{
    if (!NaniteFrameDummy) { const unsigned clear=0; if (!CreateNaniteFrameBuffer(1,&clear,false,NaniteFrameDummy)) return false; }
    if (!LoadTechniqueShaderSource("DX12","NaniteVisibility.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSReduce","CSCapture","CSCull","CSBuildDraws","CSCountBins","CSScanBins","CSScatterBins"}; constexpr const char* definitions[]{nullptr,"EVANS_NANITE_CAPTURE_DEPTH","EVANS_NANITE_FRAME_CULL",nullptr,nullptr,nullptr,nullptr};
    for (unsigned pass=0;pass<NaniteFrameShaders.size();++pass)
    {
        if (!NaniteFrameRoots[pass])
        {
            D3D12_ROOT_PARAMETER parameters[5]{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            parameters[0].Constants.Num32BitValues=sizeof(NaniteVisibilityConstants)/sizeof(unsigned);
            parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
            parameters[3].ParameterType=parameters[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[3].Descriptor.ShaderRegister=1; parameters[4].Descriptor.ShaderRegister=2;
            D3D12_DESCRIPTOR_RANGE range{}; range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors=1;
            if (pass==1) { parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[1].DescriptorTable={1,&range}; }
            D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=pass>=2?5:3; desc.pParameters=parameters; ComPtr<ID3DBlob> code,errors;
            auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf());
            if (FAILED(status)) return FailHRESULT("Serialize Nanite frame root",status);
            status=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(NaniteFrameRoots[pass].GetAddressOf()));
            if (FAILED(status)) return FailHRESULT("Create Nanite frame root",status);
        }
        if (!NaniteFrameShaders[pass])
        {
            const D3D_SHADER_MACRO macros[]{{definitions[pass],"1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> code;
            if (pass>=3 && !LoadTechniqueShaderSource("DX12","NaniteDraws.hlsl",ShaderCode,LastError)) return false;
            if (!CompileShader(entries[pass],"cs_5_0",code,definitions[pass]?macros:nullptr,true)) return false;
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=NaniteFrameRoots[pass].Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
            const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(NaniteFrameShaders[pass].GetAddressOf()));
            if (FAILED(status)) return FailHRESULT("Create Nanite frame kernel",status);
        }
    }
    return true;
}
bool DispatchNaniteFrame(unsigned pass,const NaniteVisibilityConstants& constants,unsigned groupsX,unsigned groupsY,
    const std::shared_ptr<NaniteFrameBuffer>& source,const std::shared_ptr<NaniteFrameBuffer>& depth,
    const std::shared_ptr<NaniteFrameBuffer>& main,const std::shared_ptr<NaniteFrameBuffer>& output,bool camera)
{
    auto work=std::make_shared<NaniteFrameCommandResources>(); work->Buffers={source,depth,main,output};
    if (camera)
    {
        work->Camera=MainDepth; D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=1; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        const auto status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(work->Views.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Nanite frame depth view heap",status);
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32_FLOAT; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(MainDepth.Get(),&view,work->Views->GetCPUDescriptorHandleForHeapStart());
    }
    NaniteFrameWork.push_back(work); // Root addresses and descriptors outlive recording and failed waits.
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    const auto originalDepth=MainDepthState;
    if (camera) Transition(MainDepth.Get(),MainDepthState,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Transition(output->Buffer.Get(),output->State,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CommandList->SetComputeRootSignature(NaniteFrameRoots[pass].Get()); CommandList->SetPipelineState(NaniteFrameShaders[pass].Get());
    CommandList->SetComputeRoot32BitConstants(0,sizeof(constants)/sizeof(unsigned),&constants,0);
    if (camera)
    { ID3D12DescriptorHeap* heaps[]{work->Views.Get()}; CommandList->SetDescriptorHeaps(1,heaps); CommandList->SetComputeRootDescriptorTable(1,work->Views->GetGPUDescriptorHandleForHeapStart()); }
    else CommandList->SetComputeRootShaderResourceView(1,source->Buffer->GetGPUVirtualAddress());
    CommandList->SetComputeRootUnorderedAccessView(2,output->Buffer->GetGPUVirtualAddress());
    if (pass>=2)
    { CommandList->SetComputeRootShaderResourceView(3,depth->Buffer->GetGPUVirtualAddress()); CommandList->SetComputeRootShaderResourceView(4,main->Buffer->GetGPUVirtualAddress()); }
    CommandList->Dispatch(groupsX,groupsY,1); D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=output->Buffer.Get(); CommandList->ResourceBarrier(1,&barrier);
    // Immutable outputs become readable by later frame work. Across completed
    // submissions buffers decay to COMMON and can be implicitly promoted again.
    Transition(output->Buffer.Get(),output->State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (camera)
    { Transition(MainDepth.Get(),MainDepthState,originalDepth); ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps); }
    return true;
}
bool CaptureNaniteDepth(NaniteDepthHandle& output,bool rasterInternal=false)
{
    NaniteVisibilityWork work;
    if (!ValidateNaniteRecording(rasterInternal) || FrameSampleCount!=1 || !MainDepth || !NanitePrimaryDepthInitialized(PendingFrame.Passes)) return Fail("Nanite depth capture requires initialized single-sample camera depth");
    if (!BuildNaniteDepthWork(CreateInfo.Width,CreateInfo.Height,work,LastError) || !FrameNanite.CanAllocate(std::size_t(work.OutputWords)*sizeof(unsigned),true,LastError) || !EnsureNaniteFrameVisibility()) return false;
    auto record=std::make_shared<NaniteDepthRecord>();
    if (!CreateNaniteFrameBuffer(work.OutputWords,nullptr,true,record->Data)) return false;
    record->Info={FrameNanite.Frame(),unsigned(CreateInfo.Width),unsigned(CreateInfo.Height),work.OutputWords,false};
    for (const auto& job:work.Dispatches)
        if (!DispatchNaniteFrame(job.Pass==2?1:0,job.Constants,job.GroupsX,job.GroupsY,NaniteFrameDummy,nullptr,nullptr,record->Data,job.Pass==2)) return false;
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
    if (!DispatchNaniteFrame(2,record->Constants,job.GroupsX,job.GroupsY,record->Queries,depth?depth->Data:NaniteFrameDummy,NaniteFrameDummy,record->Data,false)) return false;
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
    if (!DispatchNaniteFrame(2,record->Constants,gx,(groups+gx-1)/gx,record->Queries,d->second->Data,m->second->Data,record->Data,false)) return false;
    NaniteVisibilityHandle handle{NewNaniteVisibilityIdentity()}; NaniteDecisions.emplace(handle.Value,record); FrameNanite.Record(handle,bytes); output=handle; return true;
}
bool ReadNaniteFrameBuffer(const NaniteFrameBuffer& buffer,std::vector<unsigned>& words)
{
    if (!RecoverSubmittedWork("Read Nanite frame result")) return false; NaniteFrameReadback.reset();
    auto work=std::make_shared<NaniteReadbackResources>(); work->Input=buffer.Buffer; const auto bytes=std::size_t(buffer.Words)*sizeof(unsigned);
    if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST,work->Readback)) return false;
    auto status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(work->Allocator.GetAddressOf()));
    if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,work->Allocator.Get(),nullptr,IID_PPV_ARGS(work->Commands.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Nanite frame readback commands",status);
    // Closed-frame immutable buffers have decayed to COMMON; copy-source
    // promotion and decay leave their next graphics-frame use unchanged.
    work->Commands->CopyBufferRegion(work->Readback.Get(),0,buffer.Buffer.Get(),0,bytes); status=work->Commands->Close();
    if (FAILED(status)) return FailHRESULT("Close Nanite frame readback commands",status);
    NaniteFrameReadback=work; ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists); SubmittedWorkPending=true;
    if (!SignalAndWait()) return false; NaniteFrameReadback.reset();
    void* mapped=nullptr; const D3D12_RANGE range{0,bytes}; status=work->Readback->Map(0,&range,&mapped);
    if (FAILED(status)) return FailHRESULT("Map Nanite frame feedback",status);
    words.resize(buffer.Words); std::memcpy(words.data(),mapped,bytes); const D3D12_RANGE noWrite{0,0}; work->Readback->Unmap(0,&noWrite); return true;
}
