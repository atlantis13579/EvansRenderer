struct LumenDistanceRecord
{
    LumenDistanceSceneInfo Info; std::shared_ptr<LumenDistanceRecord> Base;
    ComPtr<ID3D11Buffer> Scene; ComPtr<ID3D11ShaderResourceView> View;
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenDistanceRecord>> LumenDistanceScenes;
ComPtr<ID3D11ComputeShader> LumenDistanceTrace;
void DestroyLumenDistanceScenes() { LumenDistanceScenes.clear(); DestroyLumenGlobalDistance(); LumenDistanceTrace.Reset(); }
bool EnsureLumenDistanceTrace()
{
    if (LumenDistanceTrace) return true;
    if (Device->GetFeatureLevel()<D3D_FEATURE_LEVEL_11_0) return Fail("Lumen distance tracing requires D3D11 compute");
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_DISTANCE_TRACE_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSTraceDistance","cs_5_0",binary,macros,true)) return false;
    const auto status=Device->CreateComputeShader(binary->GetBufferPointer(),binary->GetBufferSize(),nullptr,LumenDistanceTrace.GetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create Lumen distance compute shader",status));
}
bool CreateLumenDistanceScene(const LumenDistanceSceneData& data,LumenDistanceSceneHandle& output)
{
    if (!ValidateLumenDistanceScene(data,LastError) || !EnsureLumenDistanceTrace()) return false;
    auto candidate=std::make_shared<LumenDistanceRecord>(); candidate->Info=data.Info;
    const std::uint32_t empty=0; const auto count=std::max<std::size_t>(1,data.Words.size());
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(count*sizeof(std::uint32_t)); desc.Usage=D3D11_USAGE_IMMUTABLE;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(std::uint32_t);
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data.Words.empty()?&empty:data.Words.data();
    auto status=Device->CreateBuffer(&desc,&initial,candidate->Scene.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Upload immutable Lumen distance scene",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; view.Buffer.NumElements=UINT(count);
    status=Device->CreateShaderResourceView(candidate->Scene.Get(),&view,candidate->View.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("View Lumen distance scene",status));
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,std::move(candidate)); output=handle; return true;
}
bool TraceLumenDistanceScene(LumenDistanceSceneHandle handle,const LumenDistanceTraceUpload& upload,LumenDistanceTraceResult& output)
{
    const auto found=LumenDistanceScenes.find(handle.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown Lumen distance scene");
    const auto& record=*found->second; LumenDistanceTraceFrame frame;
    if (!BuildLumenDistanceTraceFrame(record.Info,upload,frame,LastError) || !EnsureLumenDistanceTrace()) return false;
    const auto count=upload.Rays.size(),bytes=count*sizeof(LumenDistanceHit);
    ComPtr<ID3D11Buffer> queries,uniform,feedback,staging; ComPtr<ID3D11ShaderResourceView> input; ComPtr<ID3D11UnorderedAccessView> target;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(count*sizeof(LumenDistanceRay)); desc.Usage=D3D11_USAGE_IMMUTABLE;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(LumenDistanceRay);
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=upload.Rays.data(); auto status=Device->CreateBuffer(&desc,&initial,queries.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen distance rays",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; srv.Buffer.NumElements=UINT(count);
    status=Device->CreateShaderResourceView(queries.Get(),&srv,input.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View Lumen distance rays",status));
    desc={}; desc.ByteWidth=sizeof(frame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER; initial.pSysMem=&frame;
    status=Device->CreateBuffer(&desc,&initial,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen distance constants",status));
    desc={}; desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(LumenDistanceHit);
    status=Device->CreateBuffer(&desc,nullptr,feedback.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen distance feedback",status));
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=UINT(count);
    status=Device->CreateUnorderedAccessView(feedback.Get(),&uav,target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View Lumen distance feedback",status));
    desc={}; desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen distance readback",status));
    LumenDistanceTraceResult result; result.Identity=record.Info.Identity; result.Hits.resize(count);
    ID3D11ShaderResourceView* views[]{record.View.Get(),input.Get()}; auto* constants=uniform.Get(); auto* destination=target.Get();
    Context->CSSetShader(LumenDistanceTrace.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&constants);
    Context->CSSetShaderResources(0,2,views); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr);
    const auto groups=unsigned((count+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads);
    Context->Dispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
    ID3D11ShaderResourceView* empty[2]{}; constants=nullptr; destination=nullptr;
    Context->CSSetShaderResources(0,2,empty); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr); Context->CSSetConstantBuffers(0,1,&constants); Context->CSSetShader(nullptr,nullptr,0);
    Context->CopyResource(staging.Get(),feedback.Get()); D3D11_MAPPED_SUBRESOURCE mapped{};
    status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped); if (FAILED(status)) return Fail(HResultMessage("Read Lumen distance feedback",status));
    std::memcpy(result.Hits.data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    if (!ValidateLumenDistanceTraceResult(record.Info,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
bool DestroyLumenDistanceScene(LumenDistanceSceneHandle handle)
{ return LumenDistanceScenes.erase(handle.Value)!=0 || Fail("Unknown Lumen distance scene"); }
