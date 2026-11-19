// Preserve opaque HDR before reusing the lighting target for the diffuse seed.
// MSAA is averaged only after sample-frequency light evaluation. The final
// replacement reads the saved scene, so no draw samples its own attachment.
enum SubsurfaceImage : std::size_t { SubsurfaceScene, SubsurfaceSource, SubsurfaceHorizontal, SubsurfaceVertical, SubsurfaceImageCount };
std::array<HdrSurface,SubsurfaceImageCount> SubsurfaceImages;
ComPtr<ID3D11VertexShader> SubsurfaceVertex;
ComPtr<ID3D11PixelShader> SubsurfacePixel;
ComPtr<ID3D11Buffer> SubsurfaceUniforms;
int SubsurfaceWidth=0,SubsurfaceHeight=0;
bool SubsurfaceProducing=false,SubsurfaceResolved=false,SubsurfaceHasScene=false;
DeferredLightingDesc SubsurfaceLighting;
Techniques::SubsurfaceScatteringSettings FrameSubsurface;
bool SubsurfaceRequested=false;
bool SubsurfaceComplete() const { return !SubsurfaceRequested || SubsurfaceResolved; }

void DestroySubsurface()
{
    SubsurfaceImages={}; SubsurfaceVertex.Reset(); SubsurfacePixel.Reset(); SubsurfaceUniforms.Reset();
    SubsurfaceWidth=SubsurfaceHeight=0;
}
bool EnsureSubsurface()
{
    if (SubsurfaceWidth!=CreateInfo.Width || SubsurfaceHeight!=CreateInfo.Height)
    { SubsurfaceImages={}; SubsurfaceWidth=CreateInfo.Width; SubsurfaceHeight=CreateInfo.Height; }
    for (auto& image:SubsurfaceImages) if (!CreateHdrSurface(image,CreateInfo.Width,CreateInfo.Height)) return false;
    if (SubsurfaceVertex && SubsurfacePixel && SubsurfaceUniforms) return true;
    if (!LoadTechniqueShaderSource("DX11","SubsurfaceScattering.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,nullptr,true)) return false;
    auto result=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,SubsurfaceVertex.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(subsurface)",result));
    result=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,SubsurfacePixel.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(subsurface)",result));
    D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(SubsurfaceScatteringConstants); buffer.Usage=D3D11_USAGE_DYNAMIC;
    buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    result=Device->CreateBuffer(&buffer,nullptr,SubsurfaceUniforms.ReleaseAndGetAddressOf());
    return SUCCEEDED(result) || Fail(HResultMessage("CreateBuffer(subsurface)",result));
}
void SubsurfaceUnbind()
{
    ID3D11ShaderResourceView* empty[16]{}; Context->PSSetShaderResources(0,16,empty);
    Context->OMSetRenderTargets(0,nullptr,nullptr);
}
bool SnapshotSubsurfaceScene()
{
    if (!EnsureSubsurface()) return false;
    SubsurfaceUnbind(); Context->CopyResource(SubsurfaceImages[SubsurfaceScene].Texture.Get(),HdrSurfaces[HdrScene].Texture.Get());
    return true;
}
bool CaptureSubsurfaceSource()
{
    SubsurfaceUnbind();
    if (FrameSampleCount>1)
        Context->ResolveSubresource(SubsurfaceImages[SubsurfaceSource].Texture.Get(),0,Multisampling.Color.Texture.Get(),0,DXGI_FORMAT_R16G16B16A16_FLOAT);
    else Context->CopyResource(SubsurfaceImages[SubsurfaceSource].Texture.Get(),HdrSurfaces[HdrScene].Texture.Get());
    return true;
}
bool FilterSubsurface(SubsurfaceScatteringConstants constants)
{
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->VSSetShader(SubsurfaceVertex.Get(),nullptr,0); Context->PSSetShader(SubsurfacePixel.Get(),nullptr,0);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0);
    Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    const D3D11_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    ID3D11Buffer* buffer=SubsurfaceUniforms.Get(); Context->PSSetConstantBuffers(0,1,&buffer);
    for (auto pass:{Techniques::SubsurfacePass::Horizontal,Techniques::SubsurfacePass::Vertical,Techniques::SubsurfacePass::Composite})
    {
        const auto phase=static_cast<unsigned>(pass); const bool composite=pass==Techniques::SubsurfacePass::Composite;
        SubsurfaceUnbind(); constants.Options.X=phase;
        D3D11_MAPPED_SUBRESOURCE mapped{}; const auto result=Context->Map(buffer,0,D3D11_MAP_WRITE_DISCARD,0,&mapped);
        if (FAILED(result)) return Fail(HResultMessage("Map(subsurface)",result));
        std::memcpy(mapped.pData,&constants,sizeof(constants)); Context->Unmap(buffer,0);
        auto& source=SubsurfaceImages[phase==1 ? SubsurfaceSource : phase==2 ? SubsurfaceHorizontal : SubsurfaceVertical];
        auto& destination=composite ? HdrSurfaces[HdrScene] : SubsurfaceImages[phase==1 ? SubsurfaceHorizontal : SubsurfaceVertical];
        ID3D11ShaderResourceView* inputs[]{source.View.Get(),DepthShaderView.Get(),GBufferSurfaces[1].View.Get(),
            SubsurfaceImages[SubsurfaceScene].View.Get(),SubsurfaceImages[SubsurfaceSource].View.Get()};
        auto* target=destination.Target.Get(); Context->OMSetRenderTargets(1,&target,nullptr);
        Context->PSSetShaderResources(0,static_cast<UINT>(std::size(inputs)),inputs); Context->Draw(3,0);
        ++PendingFrame.Statistics.SubsurfacePassCount;
    }
    SubsurfaceUnbind(); SubsurfaceResolved=true; return true;
}
