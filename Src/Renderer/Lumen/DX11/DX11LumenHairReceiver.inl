// One read-only-depth ribbon replay writes the private indirect component.
// Overwrite preserves the primary draw's last-writer rule at equal depths.
HdrSurface LumenHairIndirect;
ComPtr<ID3D11Texture2D> LumenHairSamples;
ComPtr<ID3D11RenderTargetView> LumenHairSampleTarget;
ComPtr<ID3D11PixelShader> LumenHairPixel,LumenHairComposite;
ComPtr<ID3D11VertexShader> LumenHairCompositeVertex;
ComPtr<ID3D11Buffer> LumenHairUniform;
unsigned LumenHairSampleCount=0;
int LumenHairWidth=0,LumenHairHeight=0;
void DestroyLumenHairReceiver()
{
    LumenHairIndirect={}; LumenHairSamples.Reset(); LumenHairSampleTarget.Reset();
    LumenHairPixel.Reset(); LumenHairComposite.Reset(); LumenHairCompositeVertex.Reset(); LumenHairUniform.Reset();
    LumenHairSampleCount=0; LumenHairWidth=LumenHairHeight=0;
}
bool EnsureLumenHairReceiver()
{
    if (!CreateHdrSurface(LumenHairIndirect,CreateInfo.Width,CreateInfo.Height)) return false;
    if (LumenHairWidth!=CreateInfo.Width || LumenHairHeight!=CreateInfo.Height || LumenHairSampleCount!=FrameSampleCount)
    {
        ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11RenderTargetView> target;
        if (FrameSampleCount>1)
        {
            D3D11_TEXTURE2D_DESC desc{}; desc.Width=CreateInfo.Width; desc.Height=CreateInfo.Height; desc.MipLevels=desc.ArraySize=1;
            desc.Format=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count=FrameSampleCount; desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_RENDER_TARGET;
            auto status=Device->CreateTexture2D(&desc,nullptr,texture.GetAddressOf());
            if (SUCCEEDED(status)) status=Device->CreateRenderTargetView(texture.Get(),nullptr,target.GetAddressOf());
            if (FAILED(status)) return Fail(HResultMessage("Create multisample fiber irradiance",status));
        }
        LumenHairSamples=texture; LumenHairSampleTarget=target; LumenHairWidth=CreateInfo.Width; LumenHairHeight=CreateInfo.Height; LumenHairSampleCount=FrameSampleCount;
    }
    if (LumenHairPixel && LumenHairComposite && LumenHairCompositeVertex && LumenHairUniform) return true;
    if (!LoadTechniqueShaderSource("DX11","LumenHair.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> indirect,vertex,composite;
    if (!CompileShader("PSIndirect","ps_5_0",indirect,nullptr,true) || !CompileShader("VSCompose","vs_5_0",vertex,nullptr,true) ||
        !CompileShader("PSCompose","ps_5_0",composite,nullptr,true)) return false;
    auto status=Device->CreatePixelShader(indirect->GetBufferPointer(),indirect->GetBufferSize(),nullptr,LumenHairPixel.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,LumenHairCompositeVertex.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreatePixelShader(composite->GetBufferPointer(),composite->GetBufferSize(),nullptr,LumenHairComposite.ReleaseAndGetAddressOf());
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(LumenHairReceiverFrame); desc.Usage=D3D11_USAGE_DYNAMIC; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER; desc.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    if (SUCCEEDED(status)) status=Device->CreateBuffer(&desc,nullptr,LumenHairUniform.ReleaseAndGetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create fiber irradiance programs",status));
}
bool ResolveLumenHairReceiver()
{
    if (!Hair.LumenReceiver) return true;
    if (!Hair.Resolved || !Hair.HasPrimaryView || !LumenVolumeFrame.Grid.W) return Fail("Fiber irradiance requires the completed primary surface and current SH volume");
    LumenHairReceiverFrame frame; HairFurConstants geometry;
    if (!BuildLumenHairReceiverFrame(LumenVolumeFrame,Hair.Frame.Settings,Hair.PrimaryView.Lighting,Hair.PrimaryView.ClipPlane,frame,LastError) ||
        !BuildHairFurReplayConstants(Hair.PrimaryView,Hair.Frame,HairFurRasterPhase::Surface,{},geometry,LastError) || !EnsureLumenHairReceiver()) return false;
    const auto upload=[&](ID3D11Buffer* buffer,const void* data,std::size_t bytes) {
        D3D11_MAPPED_SUBRESOURCE mapped{}; const auto status=Context->Map(buffer,0,D3D11_MAP_WRITE_DISCARD,0,&mapped);
        if (FAILED(status)) return Fail(HResultMessage("Upload fiber irradiance constants",status));
        std::memcpy(mapped.pData,data,bytes); Context->Unmap(buffer,0); return true;
    };
    if (!upload(HairUniforms.Get(),&geometry,sizeof(geometry)) || !upload(LumenHairUniform.Get(),&frame,sizeof(frame))) return false;
    ID3D11ShaderResourceView* empty[32]{}; Context->PSSetShaderResources(0,32,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    auto* target=FrameSampleCount>1?LumenHairSampleTarget.Get():LumenHairIndirect.Target.Get();
    const float clear[4]{}; Context->ClearRenderTargetView(target,clear);
    Context->OMSetRenderTargets(1,&target,FrameSampleCount>1?Multisampling.DepthTarget.Get():DepthTarget.Get());
    Context->OMSetDepthStencilState(HairVelocityDepth.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX); Context->RSSetState(HairRaster.Get());
    const D3D11_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    Context->IASetInputLayout(HairLayout.Get()); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const UINT stride=sizeof(HairVertex),offset=0; auto* vertices=HairVertices.Get(); Context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
    Context->IASetIndexBuffer(HairIndices.Get(),DXGI_FORMAT_R32_UINT,0); Context->VSSetShader(HairVertexProgram.Get(),nullptr,0); Context->PSSetShader(LumenHairPixel.Get(),nullptr,0);
    auto* vertexUniform=HairUniforms.Get(); auto* fragmentUniform=LumenHairUniform.Get(); auto* volume=LumenVolumeAtlas.View.Get();
    Context->VSSetConstantBuffers(0,1,&vertexUniform); Context->PSSetConstantBuffers(1,1,&fragmentUniform); Context->PSSetShaderResources(0,1,&volume);
    Context->DrawIndexed(UINT(Hair.Frame.Geometry->Indices.size()),0,0); Context->PSSetShaderResources(0,32,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    if (FrameSampleCount>1) Context->ResolveSubresource(LumenHairIndirect.Texture.Get(),0,LumenHairSamples.Get(),0,DXGI_FORMAT_R16G16B16A16_FLOAT);
    // Opaque MSAA color has already been resolved. Add only the resolved fiber
    // component; retain all preceding screen-space work and scene alpha.
    target=LumenGatherTarget.Target.Get(); Context->OMSetRenderTargets(1,&target,nullptr); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0); Context->RSSetState(SolidRasterState.Get());
    Context->IASetInputLayout(nullptr); Context->VSSetShader(LumenHairCompositeVertex.Get(),nullptr,0); Context->PSSetShader(LumenHairComposite.Get(),nullptr,0);
    ID3D11ShaderResourceView* sources[]{LumenHairIndirect.View.Get(),HdrSurfaces[HdrScene].View.Get()}; Context->PSSetShaderResources(0,2,sources); Context->Draw(3,0);
    Context->PSSetShaderResources(0,32,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    Context->CopyResource(HdrSurfaces[HdrScene].Texture.Get(),LumenGatherTarget.Texture.Get());
    ++PendingFrame.Statistics.LumenHairReceiverPassCount; ++PendingFrame.Statistics.GlobalIlluminationPassCount; return true;
}
