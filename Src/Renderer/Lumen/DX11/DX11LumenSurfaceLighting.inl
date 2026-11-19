struct LumenLightingRecord
{
    std::vector<unsigned> CardInstances; LumenLightingInfo Info;
    std::shared_ptr<LumenSurfaceRecord> Surface;
    std::shared_ptr<LumenDistanceRecord> Distance;
    ComPtr<ID3D11Buffer> Buffer;
    ComPtr<ID3D11ShaderResourceView> View;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenLightingRecord>> LumenLightingCaches;
ComPtr<ID3D11ComputeShader> LumenSurfaceLighting,LumenSurfaceRadiosity;
void DestroyLumenLightingCaches() { DestroyLumenLightingAsync(); LumenLightingCaches.clear(); LumenSurfaceLighting.Reset(); LumenSurfaceRadiosity.Reset(); }
bool EnsureLumenSurfaceLighting(bool indirect=false)
{
    auto& pipeline=indirect?LumenSurfaceRadiosity:LumenSurfaceLighting;
    if (pipeline) return true;
    if (Device->GetFeatureLevel()<D3D_FEATURE_LEVEL_11_0) return Fail("Lumen relighting requires D3D11 compute");
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_LIGHTING_STAGE","1"},{indirect?"EVANS_LUMEN_RADIOSITY":nullptr,"1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSLightSurface","cs_5_0",binary,macros,true)) return false;
    const auto status=Device->CreateComputeShader(binary->GetBufferPointer(),binary->GetBufferSize(),nullptr,pipeline.GetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create Lumen lighting shader",status));
}
bool SelectLumenPreviousLighting(const LumenSurfaceLightingUpload& upload,const std::shared_ptr<LumenSurfaceRecord>& surface,
    const std::shared_ptr<LumenDistanceRecord>& distance,std::shared_ptr<LumenLightingRecord>& output)
{
    if (!upload.PreviousLighting) return true;
    const auto found=LumenLightingCaches.find(upload.PreviousLighting.Value);
    if (found==LumenLightingCaches.end() || found->second->Surface!=surface || found->second->Distance!=distance || found->second->CardInstances!=upload.CardInstances)
        return Fail("Lumen radiosity source is expired or belongs to different native inputs");
    if (!ValidateLumenRadiositySource(surface->Identity,distance->Info.Identity,upload,found->second->Info,LastError)) return false;
    output=found->second; return true;
}
bool ReadLumenLightingRecord(const LumenLightingRecord& record,std::vector<LumenLightingTexel>& output)
{
    const auto count=std::size_t(record.Info.Layout.Width)*record.Info.Layout.Height,bytes=count*sizeof(LumenLightingTexel);
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> staging; auto status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen lighting staging",status));
    Context->CopyResource(staging.Get(),record.Buffer.Get()); D3D11_MAPPED_SUBRESOURCE mapped{};
    status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped); if (FAILED(status)) return Fail(HResultMessage("Read Lumen lighting",status));
    std::vector<LumenLightingTexel> result(count); std::memcpy(result.data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    if (!ValidateLumenLightingTexels(record.Info,result,LastError)) return false; output=std::move(result); return true;
}
bool RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    const auto source=LumenSurfaces.find(surface.Value); const auto occluders=LumenDistanceScenes.find(distance.Value);
    if (source==LumenSurfaces.end() || occluders==LumenDistanceScenes.end()) return Fail("Lumen relighting source handles are invalid or expired");
    std::shared_ptr<LumenLightingRecord> previous;
    if (!SelectLumenPreviousLighting(upload,source->second,occluders->second,previous)) return false;
    LumenSurfaceLightingInput input;
    if (!BuildLumenSurfaceLightingInput(source->second->Identity,source->second->Layout,source->second->Captures,occluders->second->Info,upload,input,LastError) || !EnsureLumenSurfaceLighting(bool(previous))) return false;
    auto record=std::make_shared<LumenLightingRecord>(); record->Surface=source->second; record->Distance=occluders->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{source->second->Identity,occluders->second->Info.Identity,upload.LightingRevision},source->second->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=previous->Info.Bounce+1; }
    ComPtr<ID3D11Buffer> constants,data; ComPtr<ID3D11ShaderResourceView> dataView; ComPtr<ID3D11UnorderedAccessView> target;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(input.Frame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=&input.Frame; auto status=Device->CreateBuffer(&desc,&initial,constants.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen lighting frame",status));
    desc.ByteWidth=UINT(input.Words.size()*sizeof(std::uint32_t)); desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
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
    ComPtr<ID3D11Query> completion;
    if (!upload.Readback)
    { const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT,0}; status=Device->CreateQuery(&query,completion.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen lighting completion query",status)); }
    auto* uniform=constants.Get(); auto* destination=target.Get();
    ID3D11ShaderResourceView* views[]{record->Surface->Views[0].Get(),record->Surface->Views[1].Get(),record->Surface->Views[2].Get(),record->Distance->View.Get(),dataView.Get(),previous?previous->View.Get():nullptr};
    Context->CSSetShader((previous?LumenSurfaceRadiosity:LumenSurfaceLighting).Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&uniform);
    Context->CSSetShaderResources(0,6,views); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr);
    const auto groups=LumenLightingGroups(input.Frame);
    Context->Dispatch(std::min(groups,LumenLightingColumns),(groups+LumenLightingColumns-1)/LumenLightingColumns,1);
    ID3D11ShaderResourceView* empty[6]{}; destination=nullptr; uniform=nullptr;
    Context->CSSetShaderResources(0,6,empty); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr);
    Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0);
    LumenSurfaceLightingResult result; result.Info=record->Info;
    if (upload.Readback) { if (!ReadLumenLightingRecord(*record,result.Texels)) return false; }
    else
    {
        Context->End(completion.Get()); Context->Flush(); constexpr ULONGLONG TimeoutMilliseconds=10000;
        const auto deadline=GetTickCount64()+TimeoutMilliseconds;
        while ((status=Context->GetData(completion.Get(),nullptr,0,0))==S_FALSE)
        { if (GetTickCount64()>=deadline) return Fail("Lumen relighting timed out"); std::this_thread::yield(); }
        if (FAILED(status)) return Fail(HResultMessage("Complete Lumen relighting",status));
    }
    result.Handle=NewLumenLightingHandle(); if (!result.Handle) return Fail("Lumen lighting handle space exhausted");
    LumenLightingCaches.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    const auto found=LumenLightingCaches.find(handle.Value); if (found==LumenLightingCaches.end()) return Fail("Unknown Lumen lighting cache");
    return ReadLumenLightingRecord(*found->second,output);
}
bool DestroyLumenLighting(LumenLightingHandle handle)
{ return LumenLightingCaches.erase(handle.Value)!=0 || Fail("Unknown Lumen lighting cache"); }
