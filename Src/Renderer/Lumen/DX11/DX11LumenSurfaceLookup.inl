ComPtr<ID3D11ComputeShader> LumenSurfaceLookup;
void DestroyLumenSurfaceLookup() { LumenSurfaceLookup.Reset(); }
bool EnsureLumenSurfaceLookup()
{
    if (LumenSurfaceLookup) return true;
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_LOOKUP_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSLookup","cs_5_0",binary,macros,true)) return false;
    const auto status=Device->CreateComputeShader(binary->GetBufferPointer(),binary->GetBufferSize(),nullptr,LumenSurfaceLookup.GetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create Lumen surface lookup shader",status));
}
bool LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Unknown Lumen surface lookup handle");
    const auto& record=*found->second; LumenSurfaceLookupFrame frame; std::vector<LumenLookupCard> cards;
    if (!BuildLumenSurfaceLookupFrame(record.Layout,upload,frame,LastError) || !BuildLumenLookupCards(record.Layout,record.Captures,cards,LastError) || !EnsureLumenSurfaceLookup()) return false;
    ComPtr<ID3D11Buffer> cardBuffer,queries,uniform,feedback,readback; ComPtr<ID3D11ShaderResourceView> cardView,queryView; ComPtr<ID3D11UnorderedAccessView> destination;
    const auto structured=[&](const void* data,std::size_t count,unsigned stride,ComPtr<ID3D11Buffer>& buffer,ComPtr<ID3D11ShaderResourceView>& view)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=UINT(count*stride); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=stride; D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data;
        auto status=Device->CreateBuffer(&desc,&initial,buffer.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen lookup input",status));
        D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; srv.Buffer.NumElements=UINT(count);
        status=Device->CreateShaderResourceView(buffer.Get(),&srv,view.GetAddressOf()); return SUCCEEDED(status) || Fail(HResultMessage("View Lumen lookup input",status));
    };
    if (!structured(cards.data(),cards.size(),sizeof(LumenLookupCard),cardBuffer,cardView) ||
        !structured(upload.Queries.data(),upload.Queries.size(),sizeof(LumenSurfaceQuery),queries,queryView)) return false;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(frame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=&frame; auto status=Device->CreateBuffer(&desc,&initial,uniform.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen lookup frame",status));
    const auto bytes=upload.Queries.size()*sizeof(LumenSurfaceSample);
    desc={}; desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(LumenSurfaceSample);
    status=Device->CreateBuffer(&desc,nullptr,feedback.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen lookup feedback",status));
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=UINT(upload.Queries.size());
    status=Device->CreateUnorderedAccessView(feedback.Get(),&uav,destination.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View Lumen lookup feedback",status));
    desc={}; desc.ByteWidth=UINT(bytes); desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    status=Device->CreateBuffer(&desc,nullptr,readback.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen lookup staging",status));
    LumenSurfaceLookupResult result; result.Identity=record.Identity; result.Samples.resize(upload.Queries.size());
    ID3D11ShaderResourceView* inputs[]{record.Views[0].Get(),record.Views[1].Get(),record.Views[2].Get(),cardView.Get(),queryView.Get()};
    auto* constants=uniform.Get(); auto* target=destination.Get(); Context->CSSetShader(LumenSurfaceLookup.Get(),nullptr,0);
    Context->CSSetConstantBuffers(0,1,&constants); Context->CSSetShaderResources(0,5,inputs); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
    const auto groups=unsigned((upload.Queries.size()+LumenLookupThreads-1)/LumenLookupThreads);
    Context->Dispatch(std::min(groups,LumenLookupColumns),(groups+LumenLookupColumns-1)/LumenLookupColumns,1);
    ID3D11ShaderResourceView* empty[5]{}; constants=nullptr; target=nullptr;
    Context->CSSetShaderResources(0,5,empty); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetConstantBuffers(0,1,&constants); Context->CSSetShader(nullptr,nullptr,0);
    Context->CopyResource(readback.Get(),feedback.Get()); D3D11_MAPPED_SUBRESOURCE mapped{};
    status=Context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped); if (FAILED(status)) return Fail(HResultMessage("Read Lumen surface lookup",status));
    std::memcpy(result.Samples.data(),mapped.pData,bytes); Context->Unmap(readback.Get(),0);
    if (!ValidateLumenSurfaceLookupResult(record.Identity,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
