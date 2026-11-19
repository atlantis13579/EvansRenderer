// Frame-owned raw storage feeds native indirect draws after UAV unbinding.
NaniteMaterialWork MaterialPlan;
std::shared_ptr<NaniteFrameBuffer> MaterialMap,MaterialWork;
ComPtr<ID3D11Buffer> MaterialUniform;
std::array<ComPtr<ID3D11ComputeShader>,4> MaterialKernels;
void DestroyNaniteMaterials()
{
    MaterialPlan={}; MaterialMap.reset(); MaterialWork.reset(); MaterialUniform.Reset(); MaterialKernels={};
}
bool PrepareNaniteMaterials(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteMaterialWork(frame,CreateInfo.Width,CreateInfo.Height,MaterialPlan,LastError)) return false;
    if (!MaterialPlan.Active()) return true;
    if (!CreateNaniteFrameBuffer(unsigned(MaterialPlan.MaterialBins.size()),MaterialPlan.MaterialBins.data(),false,MaterialMap) ||
        !CreateNaniteFrameBuffer(MaterialPlan.OutputWords,nullptr,true,MaterialWork,true)) return false;
    if (!MaterialUniform)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(NaniteMaterialConstants); desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        const auto status=Device->CreateBuffer(&desc,nullptr,MaterialUniform.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create material schedule constants",status));
    }
    if (std::all_of(MaterialKernels.begin(),MaterialKernels.end(),[](const auto& kernel){return bool(kernel);})) return true;
    if (!LoadTechniqueShaderSource("DX11","NaniteMaterials.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSClear","CSClassify","CSAllocate","CSScatter"};
    for (unsigned pass=0;pass<MaterialKernels.size();++pass) if (!MaterialKernels[pass])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[pass],"cs_5_0",code,nullptr,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,MaterialKernels[pass].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create material schedule kernel",status));
    }
    return true;
}
bool ClassifyNaniteMaterials()
{
    if (!MaterialPlan.Active()) return true;
    Context->OMSetRenderTargets(0,nullptr,nullptr);
    ID3D11ShaderResourceView* empty[8]{};
    Context->VSSetShaderResources(0,8,empty); Context->PSSetShaderResources(0,8,empty);
    ID3D11Buffer* uniform=MaterialUniform.Get(); Context->UpdateSubresource(uniform,0,nullptr,&MaterialPlan.Constants,0,0);
    Context->CSSetConstantBuffers(0,1,&uniform);
    ID3D11ShaderResourceView* inputs[]{MaterialMap->View.Get(),VisibilityGeometryView.Get(),VisibilityImages[0].View.Get()};
    Context->CSSetShaderResources(0,3,inputs);
    ID3D11UnorderedAccessView* target=MaterialWork->Target.Get(); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
    for (unsigned pass=0;pass<MaterialKernels.size();++pass)
    {
        const auto& c=MaterialPlan.Constants;
        const unsigned x=pass==0?(c.Layout.X+NaniteMaterialThreads-1)/NaniteMaterialThreads:pass==2?1:c.Layout.Y;
        const unsigned y=pass==0 || pass==2?1:c.Layout.Z;
        Context->CSSetShader(MaterialKernels[pass].Get(),nullptr,0); Context->Dispatch(x,y,1);
    }
    target=nullptr; Context->CSSetUnorderedAccessViews(0,1,&target,nullptr); Context->CSSetShaderResources(0,3,empty); Context->CSSetShader(nullptr,nullptr,0);
    auto& stats=PendingFrame.Statistics; stats.NaniteMaterialDispatchCount+=unsigned(MaterialKernels.size());
    stats.NaniteMaterialBinCount+=unsigned(MaterialPlan.Representatives.size()); stats.VisibilityMaterialDrawCount+=unsigned(MaterialPlan.Representatives.size());
    stats.IndirectDrawCallCount+=unsigned(MaterialPlan.Representatives.size()); return true;
}
