std::array<ComPtr<ID3D11ComputeShader>,2> VirtualShadowRequestShaders;
void DestroyVirtualShadowRequests() { for (auto& shader:VirtualShadowRequestShaders) shader.Reset(); }
bool BuildVirtualShadowRequests(const VirtualShadowRequestUpload& upload,VirtualShadowRequestResult& output,ID3D11Buffer* nativeDepth=nullptr)
{
    if (!nativeDepth && !ValidateVirtualShadowRequestUpload(upload,LastError)) return false;
    VirtualShadowRequestConstants constants;
    if (!BuildVirtualShadowRequestConstants(upload.Frame,constants,LastError)) return false;
    if (Device->GetFeatureLevel()<D3D_FEATURE_LEVEL_11_0) return Fail("Virtual shadow demand requires D3D11 compute and structured atomics");
    if (!LoadTechniqueShaderSource("DX11","VirtualShadowMaps.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSClear","CSMark"};
    for (unsigned stage=0;stage<VirtualShadowRequestShaders.size();++stage) if (!VirtualShadowRequestShaders[stage])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[stage],"cs_5_0",code,nullptr,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,VirtualShadowRequestShaders[stage].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create virtual shadow request shader",status));
    }
    struct Bindings
    {
        ID3D11DeviceContext* Context;
        ~Bindings()
        {
            ID3D11ShaderResourceView* source=nullptr; ID3D11UnorderedAccessView* target=nullptr; ID3D11Buffer* uniform=nullptr;
            Context->CSSetShaderResources(0,1,&source); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
            Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShader(nullptr,nullptr,0);
        }
    } bindings{Context.Get()};
    ComPtr<ID3D11Buffer> depth,demand,uniform,staging; ComPtr<ID3D11ShaderResourceView> source; ComPtr<ID3D11UnorderedAccessView> target;
    const auto depthCount=std::size_t(upload.Frame.Width)*upload.Frame.Height;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(depthCount*sizeof(float)); desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(float);
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=upload.Depth.data();
    HRESULT status=S_OK; depth=nativeDepth;
    if (!depth) status=Device->CreateBuffer(&desc,&initial,depth.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create shadow receiver depth",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; view.Buffer.NumElements=static_cast<UINT>(depthCount);
    status=Device->CreateShaderResourceView(depth.Get(),&view,source.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View shadow receiver depth",status));
    desc.ByteWidth=constants.Limits.Y*sizeof(std::uint32_t); desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
    status=Device->CreateBuffer(&desc,nullptr,demand.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create shadow page requests",status));
    D3D11_UNORDERED_ACCESS_VIEW_DESC unordered{}; unordered.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; unordered.Buffer.NumElements=constants.Limits.Y;
    status=Device->CreateUnorderedAccessView(demand.Get(),&unordered,target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View shadow page requests",status));
    desc={}; desc.ByteWidth=sizeof(constants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER; initial.pSysMem=&constants;
    status=Device->CreateBuffer(&desc,&initial,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create shadow request constants",status));
    auto* cb=uniform.Get(); auto* srv=source.Get(); auto* uav=target.Get();
    Context->CSSetConstantBuffers(0,1,&cb); Context->CSSetShaderResources(0,1,&srv); Context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);
    const unsigned counts[]{constants.Limits.Y,constants.Extent.X*constants.Extent.Y};
    for (unsigned stage=0;stage<VirtualShadowRequestShaders.size();++stage)
    {
        const auto groups=(counts[stage]+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
        Context->CSSetShader(VirtualShadowRequestShaders[stage].Get(),nullptr,0);
        Context->Dispatch(std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1);
    }
    desc={}; desc.ByteWidth=constants.Limits.Y*sizeof(std::uint32_t); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create shadow demand readback",status));
    Context->CopyResource(staging.Get(),demand.Get()); D3D11_MAPPED_SUBRESOURCE mapped{};
    status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped); if (FAILED(status)) return Fail(HResultMessage("Map shadow demand",status));
    VirtualShadowRequestResult result; result.Pages.resize(constants.Limits.Y); std::memcpy(result.Pages.data(),mapped.pData,desc.ByteWidth); Context->Unmap(staging.Get(),0);
    VirtualShadowLayout layout; if (!BuildVirtualShadowLayout(upload.Frame.Settings,layout,LastError) || !ValidateVirtualShadowRequests(layout,result.Pages,LastError)) return false;
    output=std::move(result); return true;
}
