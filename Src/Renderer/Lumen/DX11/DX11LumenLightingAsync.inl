// Tile slices write a private atlas; only a retired final event publishes it.
struct LumenLightingSliceWork : LumenLightingJob
{
    std::shared_ptr<LumenLightingRecord> Record,Previous;
    ComPtr<ID3D11Buffer> Uniform,Data;
    ComPtr<ID3D11ShaderResourceView> DataView;
    ComPtr<ID3D11UnorderedAccessView> Target;
    ComPtr<ID3D11Query> Completion;
};
LumenLightingQueue<LumenLightingSliceWork> LumenLightingTasks;
bool CollectLumenLighting()
{ return LumenLightingTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenLighting(work,ready); }); }
bool CancelLumenLightingUpdate(LumenLightingTask task)
{ return (LumenLightingTasks.Cancel(task) || Fail("Invalid Lumen lighting task")) && CollectLumenLighting(); }
void DestroyLumenLightingAsync() { LumenLightingTasks.Clear(); }
bool ProbeLumenLighting(LumenLightingSliceWork& work,bool& ready)
{
    BOOL complete=FALSE; const auto status=Context->GetData(work.Completion.Get(),&complete,sizeof(complete),D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (FAILED(status)) return Fail(HResultMessage("Poll Lumen lighting completion",status));
    ready=status==S_OK && complete; return true;
}
bool BeginLumenLightingUpdate(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenLightingTask& output)
{
    if (!CollectLumenLighting() || !EnsureLumenSurfaceLighting(bool(upload.PreviousLighting))) return false;
    const auto source=LumenSurfaces.find(surface.Value); const auto occluders=LumenDistanceScenes.find(distance.Value);
    if (source==LumenSurfaces.end() || occluders==LumenDistanceScenes.end()) return Fail("Lumen relighting source handles are invalid or expired");
    auto job=std::make_shared<LumenLightingSliceWork>();
    if (!SelectLumenPreviousLighting(upload,source->second,occluders->second,job->Previous)) return false;
    if (!job->Prepare(source->second->Identity,source->second->Layout,source->second->Captures,occluders->second->Info,upload,LastError) || !LumenLightingTasks.CanSubmit(*job,LastError)) return false;
    const auto& input=job->Input;
    auto record=std::make_shared<LumenLightingRecord>(); record->Surface=source->second; record->Distance=occluders->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{source->second->Identity,occluders->second->Info.Identity,upload.LightingRevision},source->second->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (job->Previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=job->Previous->Info.Bounce+1; }
    job->Record=record; auto& constants=job->Uniform; auto& data=job->Data; auto& dataView=job->DataView; auto& target=job->Target;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(input.Frame); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=&input.Frame; auto status=Device->CreateBuffer(&desc,&initial,constants.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen lighting frame",status));
    desc.ByteWidth=UINT(input.Words.size()*sizeof(std::uint32_t)); desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.Usage=D3D11_USAGE_IMMUTABLE;
    desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(ShaderUInt4); initial.pSysMem=input.Words.data();
    status=Device->CreateBuffer(&desc,&initial,data.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen lighting emitters/cards",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; srv.Buffer.NumElements=desc.ByteWidth/desc.StructureByteStride;
    status=Device->CreateShaderResourceView(data.Get(),&srv,dataView.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View Lumen lighting input",status));
    const auto count=std::size_t(record->Info.Layout.Width)*record->Info.Layout.Height;
    desc.ByteWidth=UINT(count*sizeof(LumenLightingTexel)); desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS; desc.StructureByteStride=sizeof(LumenLightingTexel);
    status=Device->CreateBuffer(&desc,nullptr,record->Buffer.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create resident Lumen radiance",status));
    srv.Buffer.NumElements=UINT(count); status=Device->CreateShaderResourceView(record->Buffer.Get(),&srv,record->View.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("View resident Lumen radiance",status));
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=UINT(count);
    status=Device->CreateUnorderedAccessView(record->Buffer.Get(),&uav,target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Write resident Lumen radiance",status));
    const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0}; status=Device->CreateQuery(&query,job->Completion.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create lighting slice event",status));
    const auto task=LumenLightingTasks.Insert(job); if (!task) return Fail("Lumen lighting task space exhausted"); output=task; return true;
}
bool SubmitLumenLightingSlice(LumenLightingSliceWork& work,const LumenSurfaceLightingFrame& frame)
{
    const auto& record=work.Record; Context->UpdateSubresource(work.Uniform.Get(),0,nullptr,&frame,0,0);
    auto* uniform=work.Uniform.Get(); auto* destination=work.Target.Get();
    ID3D11ShaderResourceView* views[]{record->Surface->Views[0].Get(),record->Surface->Views[1].Get(),record->Surface->Views[2].Get(),record->Distance->View.Get(),work.DataView.Get(),work.Previous?work.Previous->View.Get():nullptr};
    Context->CSSetShader((work.Previous?LumenSurfaceRadiosity:LumenSurfaceLighting).Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&uniform);
    Context->CSSetShaderResources(0,6,views); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr);
    const auto groups=LumenLightingGroups(frame);
    Context->Dispatch(std::min(groups,LumenLightingColumns),(groups+LumenLightingColumns-1)/LumenLightingColumns,1);
    ID3D11ShaderResourceView* empty[6]{}; destination=nullptr; uniform=nullptr;
    Context->CSSetShaderResources(0,6,empty); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr);
    Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0);
    Context->End(work.Completion.Get()); work.InFlight=true; Context->Flush(); return true;
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
