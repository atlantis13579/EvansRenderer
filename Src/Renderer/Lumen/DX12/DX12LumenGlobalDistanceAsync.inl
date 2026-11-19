struct LumenGlobalWork : LumenGlobalDistanceJob
{
    std::shared_ptr<LumenDistanceRecord> Record,Previous;
    std::shared_ptr<LumenDistanceWork> Submission;
    ComPtr<ID3D12Resource> Dirty;
    ComPtr<ID3D12Fence> Completion;
    std::uint64_t FenceValue=0;
    bool Unfenced=false;
};
LumenGlobalDistanceQueue<LumenGlobalWork> LumenGlobalTasks;
void DestroyLumenGlobalDistanceAsync() { LumenGlobalTasks.Clear(); }
bool ProbeLumenGlobalDistance(LumenGlobalWork& work,bool& ready)
{
    if (work.Unfenced) return Fail("Global distance submission has no completion fence");
    const auto completed=work.Completion->GetCompletedValue();
    if (completed==UINT64_MAX) return FailHRESULT("Poll global distance device removal",Device->GetDeviceRemovedReason());
    ready=completed>=work.FenceValue; return true;
}
bool CollectLumenGlobalDistance()
{ return LumenGlobalTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenGlobalDistance(work,ready); }); }
bool CancelLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task)
{ return (LumenGlobalTasks.Cancel(task) || Fail("Invalid global distance task")) && CollectLumenGlobalDistance(); }
bool BeginLumenGlobalDistanceUpdate(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle previous,LumenGlobalDistanceTask& output)
{
    if (!CollectLumenGlobalDistance() || !EnsureLumenGlobalDistance()) return false;
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source");
    auto work=std::make_shared<LumenGlobalWork>();
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second) return Fail("Global distance reuse crosses source generations");
        work->Previous=old->second;
    }
    if (!work->Prepare(found->second->Info,settings,work->Previous?work->Previous->Info.Global.get():nullptr,LastError) || !LumenGlobalTasks.CanSubmit(*work,LastError)) return false;
    work->Record=std::make_shared<LumenDistanceRecord>(); auto& record=*work->Record;
    record.Base=found->second; record.Info=record.Base->Info; record.Info.Global=work->Plan;
    auto desc=BufferResourceDesc(std::size_t(work->Plan->WordCount)*sizeof(std::uint32_t)); desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    auto status=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(record.Scene.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Allocate asynchronous global field",status);
    const auto dirtyBytes=std::max<std::size_t>(1,work->Update.Compose.size())*sizeof(std::uint32_t);
    if (!CreateBufferResource(dirtyBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work->Dirty)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; status=work->Dirty->Map(0,&noRead,&mapped);
    if (FAILED(status)) return FailHRESULT("Map global dirty list",status);
    if (!work->Update.Compose.empty()) std::memcpy(mapped,work->Update.Compose.data(),work->Update.Compose.size()*sizeof(std::uint32_t));
    work->Dirty->Unmap(0,nullptr);
    status=Device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(work->Completion.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create global distance completion fence",status);
    const auto task=LumenGlobalTasks.Insert(work); if (!task) return Fail("Global task space exhausted"); output=task; return true;
}
bool SubmitLumenGlobalDistanceSlice(LumenGlobalWork& work,const LumenGlobalDistanceFrame& frame)
{
    auto submission=std::make_shared<LumenDistanceWork>(); submission->Record=work.Record; submission->CopySource=work.Previous;
    if (!BeginLumenDistanceWork(*submission,&frame,sizeof(frame))) return false;
    auto& record=*work.Record; auto* commands=submission->Commands.Get();
    const auto transition=[&](ID3D12Resource* buffer,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={buffer,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; commands->ResourceBarrier(1,&barrier);
    };
    if (!work.Initialized)
    {
        transition(record.Base->Scene.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        commands->CopyBufferRegion(record.Scene.Get(),0,record.Base->Scene.Get(),0,std::max(1u,record.Info.Counts.W)*sizeof(std::uint32_t));
        commands->CopyBufferRegion(record.Scene.Get(),std::size_t(frame.Header.Layout.W)*sizeof(std::uint32_t),submission->Upload.Get(),offsetof(LumenGlobalDistanceFrame,Header),sizeof(LumenGlobalDistanceHeader));
        transition(record.Base->Scene.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (!work.Update.Copies.empty())
        {
            transition(work.Previous->Scene.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
            for (const auto& copy:work.Update.Copies) commands->CopyBufferRegion(record.Scene.Get(),std::size_t(copy.DestinationWord)*sizeof(std::uint32_t),
                work.Previous->Scene.Get(),std::size_t(copy.SourceWord)*sizeof(std::uint32_t),std::size_t(copy.WordCount)*sizeof(std::uint32_t));
            transition(work.Previous->Scene.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
    }
    // Completed buffer submissions decay to COMMON. Private candidates have no
    // frame readers; only this task can write the next bounded slice.
    transition(record.Scene.Get(),work.Initialized?D3D12_RESOURCE_STATE_COMMON:D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    commands->SetPipelineState(LumenGlobalCompose.Get()); commands->SetComputeRootSignature(LumenGlobalRoot.Get());
    commands->SetComputeRootConstantBufferView(0,submission->Upload->GetGPUVirtualAddress());
    commands->SetComputeRootShaderResourceView(1,record.Base->Scene->GetGPUVirtualAddress()); commands->SetComputeRootShaderResourceView(2,work.Dirty->GetGPUVirtualAddress());
    commands->SetComputeRootUnorderedAccessView(3,record.Scene->GetGPUVirtualAddress());
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    if (groups) commands->Dispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
    transition(record.Scene.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto status=commands->Close(); if (FAILED(status)) return FailHRESULT("Close global distance slice",status);
    work.Submission=std::move(submission); work.InFlight=true; ++work.FenceValue;
    ID3D12CommandList* lists[]{commands}; CommandQueue->ExecuteCommandLists(1,lists);
    status=CommandQueue->Signal(work.Completion.Get(),work.FenceValue);
    if (FAILED(status)) { work.Unfenced=true; return FailHRESULT("Signal global distance slice",status); }
    return true;
}
bool AdvanceLumenGlobalDistanceUpdate(LumenGlobalDistanceTask task,unsigned budget,bool& ready,LumenDistanceSceneHandle& output,LumenGlobalDistanceStatistics& statistics)
{
    if (!CollectLumenGlobalDistance()) return false;
    return LumenGlobalTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenGlobalDistance(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenGlobalDistanceSlice(work,frame); },
        [&](auto& work,LumenDistanceSceneHandle& handle)
        {
            handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Global scene handle space exhausted");
            LumenDistanceScenes.emplace(handle.Value,work.Record); return true;
        });
}
