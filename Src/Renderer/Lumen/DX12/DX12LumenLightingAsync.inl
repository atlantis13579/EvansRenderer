struct LumenLightingSliceWork : LumenLightingJob
{
    std::shared_ptr<LumenLightingRecord> Record,Previous;
    std::shared_ptr<LumenLightingWork> Submission;
    ComPtr<ID3D12Fence> Completion;
    std::uint64_t FenceValue=0; bool Unfenced=false;
};
LumenLightingQueue<LumenLightingSliceWork> LumenLightingTasks;
bool CollectLumenLighting()
{ return LumenLightingTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenLighting(work,ready); }); }
bool CancelLumenLightingUpdate(LumenLightingTask task)
{ return (LumenLightingTasks.Cancel(task) || Fail("Invalid Lumen lighting task")) && CollectLumenLighting(); }
void DestroyLumenLightingAsync() { LumenLightingTasks.Clear(); }
bool ProbeLumenLighting(LumenLightingSliceWork& work,bool& ready)
{
    if (work.Unfenced) return Fail("Lumen lighting submission has no completion fence");
    const auto completed=work.Completion->GetCompletedValue();
    if (completed==UINT64_MAX) return FailHRESULT("Poll Lumen lighting device removal",Device->GetDeviceRemovedReason());
    ready=completed>=work.FenceValue; return true;
}
bool BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& output)
{
    if (!CollectLumenLighting()) return false;
    const auto source=LumenSurfaces.find(surface.Value); const auto occluders=LumenDistanceScenes.find(distance.Value);
    if (source==LumenSurfaces.end() || occluders==LumenDistanceScenes.end()) return Fail("Lumen relighting source handles are invalid or expired");
    if (!EnsureLumenSurfaceLighting(bool(upload.PreviousLighting),bool(occluders->second->Hardware))) return false;
    auto job=std::make_shared<LumenLightingSliceWork>();
    if (!SelectLumenPreviousLighting(upload,source->second,occluders->second,job->Previous)) return false;
    if (!job->Prepare(source->second->Identity,source->second->Layout,source->second->Captures,occluders->second->Info,upload,LastError) || !LumenLightingTasks.CanSubmit(*job,LastError)) return false;
    const auto& input=job->Input;
    auto work=std::make_shared<LumenLightingWork>(); auto record=std::make_shared<LumenLightingRecord>(); work->Record=record; job->Record=record; job->Submission=work;
    record->Surface=source->second; record->Distance=occluders->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{source->second->Identity,occluders->second->Info.Identity,upload.LightingRevision},source->second->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (job->Previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=job->Previous->Info.Bounce+1; }
    const auto inputBytes=input.Words.size()*sizeof(std::uint32_t);
    if (!CreateBufferResource(inputBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,work->Input)) return false;
    void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; auto status=work->Input->Map(0,&noRead,&mapped);
    if (FAILED(status)) return FailHRESULT("Upload Lumen lighting inputs",status);
    std::memcpy(mapped,input.Words.data(),inputBytes); work->Input->Unmap(0,nullptr);
    const auto count=std::size_t(record->Info.Layout.Width)*record->Info.Layout.Height; auto desc=BufferResourceDesc(count*sizeof(LumenLightingTexel));
    desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; const auto heapProperties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    status=Device->CreateCommittedResource(&heapProperties,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(record->Buffer.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create resident Lumen radiance",status);
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=LumenSurfacePlanes; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(work->Views.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen lighting views",status);
    auto cpu=work->Views->GetCPUDescriptorHandleForHeapStart(); const auto increment=Device->GetDescriptorHandleIncrementSize(heap.Type);
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(record->Surface->Images[plane].Get(),&view,cpu); cpu.ptr+=increment;
    }
    status=Device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(job->Completion.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create lighting slice fence",status);
    const auto task=LumenLightingTasks.Insert(job); if (!task) return Fail("Lumen lighting task space exhausted"); output=task; return true;
}
bool SubmitLumenLightingSlice(LumenLightingSliceWork& job,const LumenSurfaceLightingFrame& frame)
{
    // The previous private fence has retired; descriptors and input remain immutable.
    auto work=std::make_shared<LumenLightingWork>(); work->Record=job.Record; work->Previous=job.Previous;
    work->Input=job.Submission->Input; work->Views=job.Submission->Views; const auto& record=job.Record;
    if (!UploadLumenLightingFrame(*work,frame) || !BeginLumenLightingWork(*work)) return false;
    if (job.Initialized) LumenLightingTransition(*work,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    work->Commands->SetPipelineState((record->Distance->Hardware?LumenHardwarePipelines[unsigned(work->Previous?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct)]:work->Previous?LumenSurfaceRadiosity:LumenSurfaceLighting).Get()); work->Commands->SetComputeRootSignature(LumenLightingRoot.Get());
    ID3D12DescriptorHeap* heaps[]{work->Views.Get()}; work->Commands->SetDescriptorHeaps(1,heaps);
    work->Commands->SetComputeRootConstantBufferView(0,work->Uniform->GetGPUVirtualAddress());
    work->Commands->SetComputeRootDescriptorTable(1,work->Views->GetGPUDescriptorHandleForHeapStart());
    work->Commands->SetComputeRootShaderResourceView(2,record->Distance->Scene->GetGPUVirtualAddress()); work->Commands->SetComputeRootShaderResourceView(3,work->Input->GetGPUVirtualAddress());
    work->Commands->SetComputeRootUnorderedAccessView(4,record->Buffer->GetGPUVirtualAddress());
    if (work->Previous) work->Commands->SetComputeRootShaderResourceView(5,work->Previous->Buffer->GetGPUVirtualAddress());
    BindLumenHardware(work->Commands.Get(),*record->Distance,6);
    const auto groups=LumenLightingGroups(frame);
    work->Commands->Dispatch(std::min(groups,LumenLightingColumns),(groups+LumenLightingColumns-1)/LumenLightingColumns,1);
    LumenLightingTransition(*work,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close lighting slice",status);
    job.Submission=work; job.InFlight=true; ++job.FenceValue;
    ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists);
    status=CommandQueue->Signal(job.Completion.Get(),job.FenceValue);
    if (FAILED(status)) { job.Unfenced=true; return FailHRESULT("Signal lighting slice",status); }
    return true;
}
bool AdvanceLumenLightingUpdate(LumenLightingTask task,unsigned budget,bool& ready,LumenSurfaceLightingResult& output,LumenLightingStatistics& statistics)
{
    if (!CollectLumenLighting()) return false;
    return LumenLightingTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenLighting(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenLightingSlice(work,frame); },
        [&](auto& work,LumenSurfaceLightingResult& result)
        {
            result.Handle=NewLumenLightingHandle(); if (!result.Handle) return Fail("Lumen lighting handle space exhausted");
            result.Info=work.Record->Info; LumenLightingCaches.emplace(result.Handle.Value,work.Record); return true;
        });
}
