// Timestamp pairs bracket the submitted frame command list, including GPU
// uploads and diagnostic copies. CPU preparation, fence waits and presentation
// are outside this interval. D3D12 timestamps in one list share a stable clock:
// https://microsoft.github.io/DirectX-Specs/d3d/CountersAndQueries.html
ComPtr<ID3D12QueryHeap> FrameTimeQueries;
ComPtr<ID3D12Resource> FrameTimeReadback;
UINT64 FrameTimeFrequency=0;
bool FrameTimeChecked=false,FrameTimeActive=false;
void DestroyFrameTiming()
{
    FrameTimeReadback.Reset(); FrameTimeQueries.Reset(); FrameTimeFrequency=0;
    FrameTimeChecked=FrameTimeActive=false;
}
void BeginFrameTiming()
{
    constexpr UINT QueryCount=2; constexpr UINT64 ResultBytes=QueryCount*sizeof(UINT64);
    if (!FrameTimeChecked)
    {
        FrameTimeChecked=true;
        D3D12_QUERY_HEAP_DESC queries{}; queries.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; queries.Count=QueryCount;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type=D3D12_HEAP_TYPE_READBACK; heap.CreationNodeMask=heap.VisibleNodeMask=1;
        D3D12_RESOURCE_DESC buffer{}; buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; buffer.Width=ResultBytes;
        buffer.Height=buffer.DepthOrArraySize=buffer.MipLevels=1; buffer.SampleDesc.Count=1; buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(CommandQueue->GetTimestampFrequency(&FrameTimeFrequency)) || !FrameTimeFrequency ||
            FAILED(Device->CreateQueryHeap(&queries,IID_PPV_ARGS(FrameTimeQueries.GetAddressOf()))) ||
            FAILED(Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,IID_PPV_ARGS(FrameTimeReadback.GetAddressOf()))))
        { FrameTimeQueries.Reset(); FrameTimeReadback.Reset(); FrameTimeFrequency=0; }
    }
    FrameTimeActive=FrameTimeQueries && FrameTimeReadback && FrameTimeFrequency;
    if (FrameTimeActive) CommandList->EndQuery(FrameTimeQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
}
void EndFrameTiming()
{
    if (!FrameTimeActive) return;
    CommandList->EndQuery(FrameTimeQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);
    CommandList->ResolveQueryData(FrameTimeQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,FrameTimeReadback.Get(),0);
}
void ReadFrameTiming()
{
    if (!FrameTimeActive) return;
    UINT64 ticks[2]{}; void* mapped=nullptr; const D3D12_RANGE read{0,sizeof(ticks)},written{0,0};
    if (FAILED(FrameTimeReadback->Map(0,&read,&mapped))) return;
    std::memcpy(ticks,mapped,sizeof(ticks)); FrameTimeReadback->Unmap(0,&written);
    constexpr double MillisecondsPerSecond=1000;
    const double elapsed=double(ticks[1]-ticks[0])*MillisecondsPerSecond/double(FrameTimeFrequency);
    if (ticks[1]>=ticks[0] && std::isfinite(elapsed) && elapsed>=0)
    { PendingFrame.Statistics.GpuFrameMilliseconds=elapsed; PendingFrame.Statistics.GpuFrameTimingAvailable=true; }
}
