struct LumenCaptureWork : LumenCaptureJob
{
    std::shared_ptr<LumenSurfaceRecord> Record; std::shared_ptr<LumenSurfaceWork> Submission;
    ComPtr<ID3D12Fence> Completion; std::uint64_t FenceValue=0; bool Unfenced=false;
};
LumenCaptureQueue<LumenCaptureWork> LumenCaptureTasks;
bool CollectLumenCapture()
{ return LumenCaptureTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenCapture(work,ready); }); }
bool CancelLumenCaptureUpdate(LumenCaptureTask task)
{ return (LumenCaptureTasks.Cancel(task) || Fail("Invalid Lumen capture task")) && CollectLumenCapture(); }
template<class Handle> bool RetireLumenCaptureInput(Handle handle)
{
    if (!handle || LumenCaptureTasks.Empty()) return true;
    // Source deletion/mutation is exceptional. Drain only submissions borrowing
    // this resource; ordinary frame advances never wait for their capture work.
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    if (!LumenCaptureTasks.Visit([&](auto& work)
    {
        if (!work.Uses(handle)) return true;
        if (FrameOpen) return Fail("Lumen capture inputs cannot change inside a render frame");
        work.Cancelled=true;
        while (work.InFlight)
        {
            bool ready=false; if (!ProbeLumenCapture(work,ready)) return false;
            if (ready) { work.InFlight=false; break; }
            if (std::chrono::steady_clock::now()>=deadline) return Fail("Retiring Lumen capture input timed out");
            std::this_thread::yield();
        }
        return true;
    })) return false;
    return CollectLumenCapture();
}
void DestroyLumenCaptureAsync() { LumenCaptureTasks.Clear(); }
bool ProbeLumenCapture(LumenCaptureWork& work,bool& ready)
{
    if (work.Unfenced) return Fail("Lumen capture has no completion fence");
    const auto completed=work.Completion->GetCompletedValue();
    if (completed==UINT64_MAX) return FailHRESULT("Poll capture device removal",Device->GetDeviceRemovedReason());
    ready=completed>=work.FenceValue; return true;
}
bool BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& output)
{
    if (!CollectLumenCapture() || !EnsureLumenSurface()) return false;
    auto job=std::make_shared<LumenCaptureWork>();
    if (!job->Prepare(upload,LastError) || !LumenCaptureTasks.CanSubmit(*job,LastError)) return false;
    const auto& layout=job->Layout;
    for (const auto& draw:upload.Draws)
    {
        const auto vertices=Buffers.find(draw.Vertices.Value),indices=Buffers.find(draw.Indices.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Lumen material draw has invalid buffers");
        if (!ValidateLumenSurfaceDraw(draw,vertices->second.Description,indices->second.Description,indices->second.Bytes,LastError)) return false;
        for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture)
        { const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Lumen material texture is invalid"); }
    }
    auto work=std::make_shared<LumenSurfaceWork>(); work->Candidate=std::make_shared<LumenSurfaceRecord>();
    job->Record=work->Candidate; job->Submission=work; auto& record=*job->Record;
    record.Layout=layout; record.Identity=upload.Identity; record.Captures=upload.Cards;
    std::vector<std::uint8_t> constants((job->Constants.size()+1)*LumenConstantStride);
    for (unsigned i=0;i<job->Constants.size();++i) std::memcpy(constants.data()+(i+1)*LumenConstantStride,&job->Constants[i],sizeof(LumenSurfaceConstants));
    if (!CreateLumenWork(*work,unsigned(upload.Draws.size())+1,constants,false)) return false;
    const auto properties=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); D3D12_RESOURCE_DESC image{}; image.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    image.Width=layout.Width; image.Height=layout.Height; image.DepthOrArraySize=1; image.MipLevels=1; image.SampleDesc.Count=1;
    image.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; image.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV; heap.NumDescriptors=LumenSurfacePlanes;
    auto status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(record.Targets.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen color target descriptors",status);
    auto rtv=record.Targets->GetCPUDescriptorHandleForHeapStart(); const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE targets[LumenSurfacePlanes]; const float clear[LumenSurfacePlanes][4]{{0,0,0,0},{0,0,0,1},{0,0,0,1}};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        D3D12_CLEAR_VALUE value{}; value.Format=image.Format; std::copy(clear[plane],clear[plane]+4,value.Color);
        status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&image,D3D12_RESOURCE_STATE_RENDER_TARGET,&value,IID_PPV_ARGS(record.Images[plane].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create immutable Lumen material target",status);
        targets[plane]=rtv; Device->CreateRenderTargetView(record.Images[plane].Get(),nullptr,rtv); rtv.ptr+=increment;
    }
    image.Format=DXGI_FORMAT_D32_FLOAT; image.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL; D3D12_CLEAR_VALUE depthClear{}; depthClear.Format=image.Format; depthClear.DepthStencil.Depth=1;
    status=Device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&image,D3D12_RESOURCE_STATE_DEPTH_WRITE,&depthClear,IID_PPV_ARGS(record.Depth.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen capture depth",status);
    heap={}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV; heap.NumDescriptors=1;
    status=Device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(record.DepthView.GetAddressOf())); if (FAILED(status)) return FailHRESULT("Create Lumen depth descriptor",status);
    const auto depthView=record.DepthView->GetCPUDescriptorHandleForHeapStart(); Device->CreateDepthStencilView(record.Depth.Get(),nullptr,depthView);
    for (unsigned index=0;index<upload.Draws.size();++index)
    {
        const auto& draw=upload.Draws[index];
        std::array<ID3D12Resource*,LumenSurfacePlanes> sources{}; std::array<DXGI_FORMAT,LumenSurfacePlanes> formats;
        const TextureHandle handles[]{draw.Diffuse,draw.Normal,draw.Opacity};
        for (unsigned slot=0;slot<LumenSurfacePlanes;++slot)
        {
            formats[slot]=DXGI_FORMAT_R8G8B8A8_UNORM;
            if (handles[slot]) { const auto& texture=Textures.at(handles[slot].Value); sources[slot]=texture.Resource.Get();
                if (texture.Description.ColorSpace==TextureColorSpace::SRGB) formats[slot]=DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; }
        }
        LumenViews(*work,index+1,sources,formats);
    }
    status=Device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(job->Completion.GetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create capture slice fence",status);
    const auto task=LumenCaptureTasks.Insert(job); if (!task) return Fail("Lumen capture task space exhausted"); output=task; return true;
}
bool SubmitLumenCaptureSlice(LumenCaptureWork& job,const LumenCaptureJob::SliceFrame& slice)
{
    const auto& upload=job.Upload; const auto& layout=job.Layout; auto& record=*job.Record;
    auto work=job.Submission;
    if (job.Initialized)
    {
        auto next=std::make_shared<LumenSurfaceWork>(); next->Candidate=job.Record; next->Constants=work->Constants; next->Views=work->Views;
        auto status=Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(next->Allocator.GetAddressOf()));
        if (SUCCEEDED(status)) status=Device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,next->Allocator.Get(),nullptr,IID_PPV_ARGS(next->Commands.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create capture slice commands",status); work=std::move(next);
    }
    for (unsigned card=slice.Dispatch.X;card<slice.Dispatch.X+slice.Dispatch.Y;++card)
    {
        const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<capture.FirstDraw+capture.DrawCount;++index)
            if (!PrepareLumenDraw(*work,upload.Draws[index])) return false;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE targets[LumenSurfacePlanes]; auto rtv=record.Targets->GetCPUDescriptorHandleForHeapStart();
    const auto increment=Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    for (auto& target:targets) { target=rtv; rtv.ptr+=increment; }
    const auto depthView=record.DepthView->GetCPUDescriptorHandleForHeapStart();
    const float clear[LumenSurfacePlanes][4]{{0,0,0,0},{0,0,0,1},{0,0,0,1}};
    auto* commands=work->Commands.Get(); auto* views=work->Views.Get(); commands->SetDescriptorHeaps(1,&views);
    commands->SetGraphicsRootSignature(LumenSurfaceRoot.Get()); commands->OMSetRenderTargets(LumenSurfacePlanes,targets,FALSE,&depthView);
    if (!job.Initialized) for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) commands->ClearRenderTargetView(targets[plane],clear[plane],0,nullptr);
    if (!job.Initialized) commands->ClearDepthStencilView(depthView,D3D12_CLEAR_FLAG_DEPTH,1,0,0,nullptr); commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const unsigned stride=layout.CardSide+LumenSurfaceBorder*2; unsigned constantSlot=1+job.FirstConstant[slice.Dispatch.X];
    for (unsigned card=slice.Dispatch.X;card<slice.Dispatch.X+slice.Dispatch.Y;++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=(card/layout.Columns)*stride;
        const D3D12_VIEWPORT viewport{float(x),float(y),float(stride),float(stride),0,1}; const D3D12_RECT scissor{LONG(x),LONG(y),LONG(x+stride),LONG(y+stride)};
        commands->RSSetViewports(1,&viewport); commands->RSSetScissorRects(1,&scissor); const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; const unsigned raster=(draw.BackFaceCulling?2u:0u)+(RealtimeWorldIsMirrored(draw.World)?1u:0u);
            commands->SetPipelineState(LumenCapturePipelines[raster].Get()); commands->SetGraphicsRootConstantBufferView(0,work->Constants->GetGPUVirtualAddress()+std::uint64_t(constantSlot++)*LumenConstantStride);
            BindLumenViews(*work,index+1,false); commands->IASetVertexBuffers(0,1,&Buffers.at(draw.Vertices.Value).VertexView); commands->IASetIndexBuffer(&Buffers.at(draw.Indices.Value).IndexView);
            commands->DrawIndexedInstanced(draw.IndexCount,1,draw.FirstIndex,0,0);
        }
    }
    commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    if (slice.Dispatch.X+slice.Dispatch.Y==layout.CardCount) for (auto& texture:record.Images) LumenBarrier(*work,texture.Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,LumenSampleState);
    auto status=work->Commands->Close(); if (FAILED(status)) return FailHRESULT("Close capture slice",status);
    job.Submission=work; job.InFlight=true; ++job.FenceValue;
    ID3D12CommandList* lists[]{work->Commands.Get()}; CommandQueue->ExecuteCommandLists(1,lists);
    for (const auto& entry:work->BufferStates) { auto& resource=Buffers.at(entry.first); resource.State=D3D12_RESOURCE_STATE_COMMON; resource.Dirty=false; }
    for (const auto& entry:work->TextureStates) { auto& resource=Textures.at(entry.first); resource.State=entry.second; resource.Dirty=false; }
    status=CommandQueue->Signal(job.Completion.Get(),job.FenceValue);
    if (FAILED(status)) { job.Unfenced=true; return FailHRESULT("Signal capture slice",status); } return true;
}
bool AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    if (!CollectLumenCapture()) return false;
    return LumenCaptureTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenCapture(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenCaptureSlice(work,frame); },
        [&](auto& work,LumenSurfaceResult& result)
        {
            result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space exhausted");
            result.Layout=work.Layout; result.Identity=work.Upload.Identity; result.DrawCalls=work.Statistics.DrawCalls;
            LumenSurfaces.emplace(result.Handle.Value,work.Record); return true;
        });
}
