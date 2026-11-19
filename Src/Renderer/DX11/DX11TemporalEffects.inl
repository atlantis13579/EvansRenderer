bool SupportsTemporalExposure() const
{
    if (!Device) return false;
    UINT flags=0; constexpr UINT required=D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_SHADER_LOAD;
    return SUCCEEDED(Device->CheckFormatSupport(DXGI_FORMAT_R32G32B32A32_FLOAT,&flags)) && (flags&required)==required;
}
// FP32 luminance reduction and presentation-transactional scalar EV history.
// Composite writes the existing HDR scratch, then copies back before bloom.
TemporalExposureHistory ExposureHistory;
std::vector<HdrSurface> ExposureLevels;
std::array<HdrSurface,2> ExposureValues;
ComPtr<ID3D11VertexShader> ExposureVertex;
ComPtr<ID3D11PixelShader> ExposurePixel;
ComPtr<ID3D11Buffer> ExposureUniforms;
int ExposureWidth=0,ExposureHeight=0;

void DestroyExposure()
{
    ExposureLevels.clear(); ExposureValues={}; ExposureVertex.Reset(); ExposurePixel.Reset(); ExposureUniforms.Reset();
    ExposureHistory={}; ExposureWidth=ExposureHeight=0;
}
bool CreateExposureSurface(HdrSurface& surface,int width,int height)
{
    if (surface.Texture) return true;
    HdrSurface candidate;
    D3D11_TEXTURE2D_DESC desc{}; desc.Width=width; desc.Height=height; desc.MipLevels=desc.ArraySize=1;
    desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; desc.SampleDesc.Count=1;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_RENDER_TARGET;
    auto result=Device->CreateTexture2D(&desc,nullptr,candidate.Texture.GetAddressOf());
    if (SUCCEEDED(result)) result=Device->CreateRenderTargetView(candidate.Texture.Get(),nullptr,candidate.Target.GetAddressOf());
    if (SUCCEEDED(result)) result=Device->CreateShaderResourceView(candidate.Texture.Get(),nullptr,candidate.View.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("Create exposure FP32 surface",result));
    candidate.Width=width; candidate.Height=height; surface=std::move(candidate); return true;
}
bool EnsureExposure()
{
    if (ExposureWidth!=CreateInfo.Width || ExposureHeight!=CreateInfo.Height)
    {
        ExposureLevels.clear(); ExposureValues={}; ExposureHistory.Invalidate();
        ExposureWidth=CreateInfo.Width; ExposureHeight=CreateInfo.Height;
    }
    const auto plan=BuildExposureReductionPlan(CreateInfo.Width,CreateInfo.Height); ExposureLevels.resize(plan.size());
    for (std::size_t index=0;index<plan.size();++index)
        if (!CreateExposureSurface(ExposureLevels[index],plan[index].Width,plan[index].Height)) return false;
    for (auto& value:ExposureValues) if (!CreateExposureSurface(value,1,1)) return false;
    if (ExposureVertex && ExposurePixel && ExposureUniforms) return true;
    if (!LoadTechniqueShaderSource("DX11","TemporalEffects.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSMain","vs_5_0",vertex) || !CompileShader("PSMain","ps_5_0",pixel)) return false;
    auto result=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,ExposureVertex.ReleaseAndGetAddressOf());
    if (SUCCEEDED(result)) result=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,ExposurePixel.ReleaseAndGetAddressOf());
    D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(TemporalEffectsConstants); buffer.Usage=D3D11_USAGE_DYNAMIC;
    buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    if (SUCCEEDED(result)) result=Device->CreateBuffer(&buffer,nullptr,ExposureUniforms.ReleaseAndGetAddressOf());
    return SUCCEEDED(result) || Fail(HResultMessage("Create temporal exposure shaders/constants",result));
}
bool ResolveExposure(const TemporalEffectsDesc& desc)
{
    TemporalEffectsConstants constants;
    if (!ExposureHistory.Prepare(desc,FramePostProcessing.TemporalEffects,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("Exposure requires finite scene time and positive dimensions");
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->VSSetShader(ExposureVertex.Get(),nullptr,0); Context->PSSetShader(ExposurePixel.Get(),nullptr,0);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0);
    Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    const auto draw=[&](HdrSurface& source,HdrSurface& previous,HdrSurface& destination,Techniques::TemporalEffectsPassKind phase)
    {
        Context->OMSetRenderTargets(0,nullptr,nullptr); ID3D11ShaderResourceView* empty[2]{}; Context->PSSetShaderResources(0,2,empty);
        constants.PhaseExtentHistory.X=static_cast<unsigned>(phase); constants.PhaseExtentHistory.Y=source.Width; constants.PhaseExtentHistory.Z=source.Height;
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (FAILED(Context->Map(ExposureUniforms.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) return Fail("Map temporal exposure constants failed");
        std::memcpy(mapped.pData,&constants,sizeof(constants)); Context->Unmap(ExposureUniforms.Get(),0);
        auto* buffer=ExposureUniforms.Get(); Context->PSSetConstantBuffers(0,1,&buffer);
        ID3D11ShaderResourceView* inputs[]{source.View.Get(),previous.View.Get()}; Context->PSSetShaderResources(0,2,inputs);
        auto* target=destination.Target.Get(); Context->OMSetRenderTargets(1,&target,nullptr);
        const D3D11_VIEWPORT viewport{0,0,float(destination.Width),float(destination.Height),0,1}; Context->RSSetViewports(1,&viewport);
        Context->Draw(3,0); ++PendingFrame.Statistics.TemporalEffectsPassCount; return true;
    };
    auto* source=&HdrSurfaces[HdrScene];
    for (std::size_t index=0;index<ExposureLevels.size();++index)
    {
        auto& destination=ExposureLevels[index];
        if (!draw(*source,*source,destination,index?Techniques::TemporalEffectsPassKind::Reduce:Techniques::TemporalEffectsPassKind::Meter)) return false;
        source=&destination;
    }
    auto& current=ExposureValues[1-ExposureHistory.ReadIndex()];
    auto& previous=constants.PhaseExtentHistory.W?ExposureValues[ExposureHistory.ReadIndex()]:*source;
    if (!draw(*source,previous,current,Techniques::TemporalEffectsPassKind::Adapt) ||
        !draw(HdrSurfaces[HdrScene],current,HdrSurfaces[HdrScratch],Techniques::TemporalEffectsPassKind::Composite)) return false;
    Context->OMSetRenderTargets(0,nullptr,nullptr); ID3D11ShaderResourceView* empty[2]{}; Context->PSSetShaderResources(0,2,empty);
    Context->CopyResource(HdrSurfaces[HdrScene].Texture.Get(),HdrSurfaces[HdrScratch].Texture.Get());
    ExposureHistory.Resolved=true; return true;
}
