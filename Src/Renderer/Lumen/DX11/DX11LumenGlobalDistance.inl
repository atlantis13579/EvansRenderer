ComPtr<ID3D11ComputeShader> LumenGlobalCompose;
void DestroyLumenGlobalDistance() { DestroyLumenGlobalDistanceAsync(); LumenGlobalCompose.Reset(); }
bool EnsureLumenGlobalDistance()
{
    if (LumenGlobalCompose) return true;
    if (!EnsureLumenDistanceTrace() || !LoadTechniqueShaderSource("DX11","LumenGlobalDistance.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> code; if (!CompileShader("CSCompose","cs_5_0",code,nullptr,true)) return false;
    const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,LumenGlobalCompose.GetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create global distance composition",status));
}
bool CreateLumenGlobalDistanceScene(LumenDistanceSceneHandle source,const LumenGlobalDistanceSettings& settings,LumenDistanceSceneHandle& output,LumenDistanceSceneHandle previous)
{
    const auto found=LumenDistanceScenes.find(source.Value); if (found==LumenDistanceScenes.end()) return Fail("Unknown global distance source scene");
    std::shared_ptr<LumenDistanceRecord> retained;
    if (previous)
    {
        const auto old=LumenDistanceScenes.find(previous.Value);
        if (old==LumenDistanceScenes.end() || !old->second->Info.Global || old->second->Base!=found->second)
            return Fail("Global distance reuse requires the exact immutable source scene");
        retained=old->second;
    }
    auto plan=std::make_shared<LumenGlobalDistancePlan>();
    if (!BuildLumenGlobalDistancePlan(found->second->Info,settings,*plan,LastError) || !EnsureLumenGlobalDistance()) return false;
    LumenGlobalDistanceUpdate update;
    if (!BuildLumenGlobalDistanceUpdate(found->second->Info,*plan,retained?retained->Info.Global.get():nullptr,update,LastError)) return false;
    auto frame=plan->Frame; frame.Dispatch={0,unsigned(update.Compose.size()),0,0};
    auto candidate=std::make_shared<LumenDistanceRecord>(); candidate->Base=found->second; candidate->Info=found->second->Info; candidate->Info.Global=plan;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=plan->WordCount*sizeof(std::uint32_t); desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(std::uint32_t);
    auto status=Device->CreateBuffer(&desc,nullptr,candidate->Scene.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Allocate global distance scene",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; view.Buffer.NumElements=plan->WordCount;
    status=Device->CreateShaderResourceView(candidate->Scene.Get(),&view,candidate->View.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View global distance scene",status));
    ComPtr<ID3D11UnorderedAccessView> target; D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=plan->WordCount;
    status=Device->CreateUnorderedAccessView(candidate->Scene.Get(),&uav,target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Write global distance scene",status));
    ComPtr<ID3D11Buffer> constants; desc={}; desc.ByteWidth=sizeof(plan->Frame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=&frame;
    status=Device->CreateBuffer(&desc,&initial,constants.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create global distance constants",status));
    ComPtr<ID3D11Buffer> dirty; ComPtr<ID3D11ShaderResourceView> dirtyView;
    const std::uint32_t empty=0;
    desc={}; desc.ByteWidth=UINT(std::max<std::size_t>(1,update.Compose.size())*sizeof(std::uint32_t)); desc.Usage=D3D11_USAGE_IMMUTABLE;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(std::uint32_t);
    initial.pSysMem=update.Compose.empty()?&empty:update.Compose.data();
    status=Device->CreateBuffer(&desc,&initial,dirty.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Upload global distance dirty voxels",status));
    view.Buffer.NumElements=desc.ByteWidth/sizeof(std::uint32_t);
    status=Device->CreateShaderResourceView(dirty.Get(),&view,dirtyView.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View global distance dirty voxels",status));
    for (const auto& copy:update.Copies)
    {
        const UINT first=copy.SourceWord*sizeof(std::uint32_t);
        D3D11_BOX region{first,0,0,first+copy.WordCount*sizeof(std::uint32_t),1,1};
        Context->CopySubresourceRegion(candidate->Scene.Get(),0,copy.DestinationWord*sizeof(std::uint32_t),0,0,retained->Scene.Get(),0,&region);
    }
    D3D11_BOX prefix{0,0,0,std::max(1u,candidate->Info.Counts.W)*sizeof(std::uint32_t),1,1};
    Context->CopySubresourceRegion(candidate->Scene.Get(),0,0,0,0,candidate->Base->Scene.Get(),0,&prefix);
    const UINT headerOffset=plan->Frame.Header.Layout.W*sizeof(std::uint32_t);
    D3D11_BOX header{headerOffset,0,0,headerOffset+sizeof(LumenGlobalDistanceHeader),1,1};
    Context->UpdateSubresource(candidate->Scene.Get(),0,&header,&plan->Frame.Header,0,0);
    auto* uniform=constants.Get(); ID3D11ShaderResourceView* inputs[]{candidate->Base->View.Get(),dirtyView.Get()}; auto* destination=target.Get();
    Context->CSSetShader(LumenGlobalCompose.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&uniform);
    Context->CSSetShaderResources(0,2,inputs); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr);
    const unsigned groups=(frame.Dispatch.Y+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
    if (groups) Context->Dispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
    uniform=nullptr; inputs[0]=inputs[1]=nullptr; destination=nullptr;
    Context->CSSetShaderResources(0,2,inputs); Context->CSSetUnorderedAccessViews(0,1,&destination,nullptr); Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0);
    const auto handle=NewLumenDistanceSceneHandle(); if (!handle) return Fail("Lumen distance handle space exhausted");
    LumenDistanceScenes.emplace(handle.Value,std::move(candidate)); output=handle; return true;
}
