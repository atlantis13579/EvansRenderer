// The frame retains the immutable atlas until subsequent native work can safely
// replace its bindings. Releasing a public handle cannot invalidate encoded draws.
std::shared_ptr<VirtualShadowAtlasRecord> FrameVirtualShadowAtlas;
VirtualShadowSamplingFrame FrameVirtualShadowSampling;
bool PrepareVirtualShadowSampling(const VirtualShadowSamplingFrame& frame,const CascadedShadowState& cascades)
{
    if (!ValidateVirtualShadowSamplingFrame(frame,LastError)) return false;
    std::shared_ptr<VirtualShadowAtlasRecord> candidate;
    if (frame.Atlas)
    {
        if (cascades.Frame.CascadeCount) return Fail("Virtual and cascaded shadow representations are mutually exclusive");
        if (FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::Hard && FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::PCF)
            return Fail("Virtual scene sampling currently requires hard or PCF filtering");
        const auto found=VirtualShadowAtlases.find(frame.Atlas.Value);
        if (found==VirtualShadowAtlases.end()) return Fail("Virtual scene sampling requires a current atlas handle");
        candidate=found->second;
    }
    FrameVirtualShadowAtlas=std::move(candidate); FrameVirtualShadowSampling=frame; return true;
}
ComPtr<ID3D11ComputeShader> VirtualShadowLookupShader;
void DestroyVirtualShadowLookup() { FrameVirtualShadowAtlas.reset(); FrameVirtualShadowSampling={}; VirtualShadowLookupShader.Reset(); }
bool CreateVirtualShadowDirectory(VirtualShadowAtlasRecord& record)
{
    VirtualShadowDirectory directory; if (!BuildVirtualShadowDirectory(record.Info,directory,LastError)) return false;
    D3D11_TEXTURE2D_DESC desc{}; desc.Width=directory.Width; desc.Height=directory.Height; desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
    desc.Format=DXGI_FORMAT_R32_FLOAT; desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=directory.Depth.data(); data.SysMemPitch=directory.Width*sizeof(float);
    auto status=Device->CreateTexture2D(&desc,&data,record.Directory.GetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateShaderResourceView(record.Directory.Get(),nullptr,record.DirectoryView.GetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Upload immutable virtual page directory",status));
}
bool SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("Virtual lookup atlas handle is invalid");
    const auto record=found->second; VirtualShadowLookupConstants constants;
    if (!BuildVirtualShadowLookupConstants(record->Info,upload,constants,LastError)) return false;
    if (!VirtualShadowLookupShader)
    {
        if (!LoadTechniqueShaderSource("DX11","VirtualShadowMaps.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO definitions[]{{"EVANS_VIRTUAL_SHADOW_LOOKUP_STAGE","1"},{nullptr,nullptr}};
        ComPtr<ID3DBlob> code; if (!CompileShader("CSLookup","cs_5_0",code,definitions,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,VirtualShadowLookupShader.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create virtual shadow lookup",status));
    }
    ComPtr<ID3D11Buffer> queries,results,uniform,staging; ComPtr<ID3D11ShaderResourceView> source; ComPtr<ID3D11UnorderedAccessView> target;
    const auto bytes=static_cast<UINT>(upload.Samples.size()*sizeof(Float4));
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=bytes; desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=sizeof(Float4);
    D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=upload.Samples.data();
    auto status=Device->CreateBuffer(&desc,&initial,queries.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create virtual lookup queries",status));
    D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; srv.Buffer.NumElements=static_cast<UINT>(upload.Samples.size());
    status=Device->CreateShaderResourceView(queries.Get(),&srv,source.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View virtual lookup queries",status));
    desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS; status=Device->CreateBuffer(&desc,nullptr,results.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create virtual lookup feedback",status));
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements=static_cast<UINT>(upload.Samples.size());
    status=Device->CreateUnorderedAccessView(results.Get(),&uav,target.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("View virtual lookup feedback",status));
    desc={}; desc.ByteWidth=sizeof(constants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER; initial.pSysMem=&constants;
    status=Device->CreateBuffer(&desc,&initial,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create virtual lookup constants",status));
    struct Cleanup
    {
        ID3D11DeviceContext* Context;
        ~Cleanup() { ID3D11ShaderResourceView* empty[3]{}; ID3D11UnorderedAccessView* uav=nullptr; Context->CSSetShaderResources(0,3,empty); Context->CSSetUnorderedAccessViews(0,1,&uav,nullptr); Context->CSSetShader(nullptr,nullptr,0); }
    } cleanup{Context.Get()};
    ID3D11ShaderResourceView* views[]{record->View.Get(),record->DirectoryView.Get(),source.Get()}; auto* cb=uniform.Get(); auto* result=target.Get();
    Context->CSSetShaderResources(0,3,views); Context->CSSetConstantBuffers(0,1,&cb); Context->CSSetUnorderedAccessViews(0,1,&result,nullptr); Context->CSSetShader(VirtualShadowLookupShader.Get(),nullptr,0);
    const auto groups=(constants.Filtering.Z+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
    Context->Dispatch(std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1);
    desc={}; desc.ByteWidth=bytes; desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    status=Device->CreateBuffer(&desc,nullptr,staging.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create lookup readback",status));
    Context->CopyResource(staging.Get(),results.Get()); D3D11_MAPPED_SUBRESOURCE mapped{};
    status=Context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped); if (FAILED(status)) return Fail(HResultMessage("Read virtual lookup",status));
    VirtualShadowLookupResult feedback; feedback.Samples.resize(upload.Samples.size()); std::memcpy(feedback.Samples.data(),mapped.pData,bytes); Context->Unmap(staging.Get(),0);
    if (!ValidateVirtualShadowLookupResult(record->Info,upload,feedback,LastError)) return false; output=std::move(feedback); return true;
}
