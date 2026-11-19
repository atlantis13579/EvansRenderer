struct LumenGlobalWork : LumenGlobalDistanceJob
{
    std::shared_ptr<LumenDistanceRecord> Record,Previous;
    ComPtr<ID3D11Buffer> Uniform,Dirty;
    ComPtr<ID3D11ShaderResourceView> DirtyView;
    ComPtr<ID3D11UnorderedAccessView> Target;
    ComPtr<ID3D11Query> Completion;
};
LumenGlobalDistanceQueue<LumenGlobalWork> LumenGlobalTasks;
void DestroyLumenGlobalDistanceAsync() { LumenGlobalTasks.Clear(); }
bool ProbeLumenGlobalDistance(LumenGlobalWork& work,bool& ready)
{
    BOOL complete=FALSE;
    const auto status=Context->GetData(work.Completion.Get(),&complete,sizeof(complete),D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (FAILED(status)) return Fail(HResultMessage("Poll global distance completion",status));
    ready=status==S_OK && complete; return true;
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
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=work->Plan->WordCount*sizeof(std::uint32_t); desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(std::uint32_t);
    auto status=Device->CreateBuffer(&desc,nullptr,record.Scene.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Allocate asynchronous global field",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; srv.Buffer.NumElements=work->Plan->WordCount;
    status=Device->CreateShaderResourceView(record.Scene.Get(),&srv,record.View.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View asynchronous global field",status));
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=work->Plan->WordCount;
    status=Device->CreateUnorderedAccessView(record.Scene.Get(),&uav,work->Target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Write asynchronous global field",status));
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.Usage=D3D11_USAGE_IMMUTABLE;
    desc.ByteWidth=UINT(std::max<std::size_t>(1,work->Update.Compose.size())*sizeof(std::uint32_t));
    const std::uint32_t empty=0; D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=work->Update.Compose.empty()?&empty:work->Update.Compose.data();
    status=Device->CreateBuffer(&desc,&initial,work->Dirty.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Upload global dirty list",status));
    srv.Buffer.NumElements=desc.ByteWidth/sizeof(std::uint32_t);
    status=Device->CreateShaderResourceView(work->Dirty.Get(),&srv,work->DirtyView.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View global dirty list",status));
    desc={}; desc.ByteWidth=sizeof(LumenGlobalDistanceFrame); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    status=Device->CreateBuffer(&desc,nullptr,work->Uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Allocate global slice constants",status));
    const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0}; status=Device->CreateQuery(&query,work->Completion.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create global completion query",status));
    const auto task=LumenGlobalTasks.Insert(work); if (!task) return Fail("Global task space exhausted"); output=task; return true;
}
bool SubmitLumenGlobalDistanceSlice(LumenGlobalWork& work,const LumenGlobalDistanceFrame& frame)
{
    auto& record=*work.Record;
    if (!work.Initialized)
    {
        const D3D11_BOX prefix{0,0,0,std::max(1u,record.Info.Counts.W)*sizeof(std::uint32_t),1,1};
        Context->CopySubresourceRegion(record.Scene.Get(),0,0,0,0,record.Base->Scene.Get(),0,&prefix);
        const UINT offset=frame.Header.Layout.W*sizeof(std::uint32_t);
        const D3D11_BOX header{offset,0,0,offset+sizeof(LumenGlobalDistanceHeader),1,1};
        Context->UpdateSubresource(record.Scene.Get(),0,&header,&frame.Header,0,0);
        for (const auto& copy:work.Update.Copies)
        {
            const UINT first=copy.SourceWord*sizeof(std::uint32_t);
            const D3D11_BOX range{first,0,0,first+copy.WordCount*sizeof(std::uint32_t),1,1};
            Context->CopySubresourceRegion(record.Scene.Get(),0,copy.DestinationWord*sizeof(std::uint32_t),0,0,work.Previous->Scene.Get(),0,&range);
        }
    }
    Context->UpdateSubresource(work.Uniform.Get(),0,nullptr,&frame,0,0);
    auto* uniform=work.Uniform.Get(); auto* target=work.Target.Get(); ID3D11ShaderResourceView* inputs[]{record.Base->View.Get(),work.DirtyView.Get()};
    Context->CSSetShader(LumenGlobalCompose.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&uniform);
    Context->CSSetShaderResources(0,2,inputs); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    if (groups) Context->Dispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
    uniform=nullptr; target=nullptr; inputs[0]=inputs[1]=nullptr;
    Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShaderResources(0,2,inputs); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetShader(nullptr,nullptr,0);
    Context->End(work.Completion.Get()); work.InFlight=true; Context->Flush(); return true;
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
