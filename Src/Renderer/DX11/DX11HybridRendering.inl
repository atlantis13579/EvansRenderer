// DX11 scene/material buffers are immutable snapshots. Only lighting storage
// changes for a light edit; the shared visibility buffers also serve ray AO.
std::array<HdrSurface,2> HybridRenderingImages;
int HybridRenderingWidth=0,HybridRenderingHeight=0;
ComPtr<ID3D11VertexShader> HybridRenderingVertex;
ComPtr<ID3D11PixelShader> HybridRenderingPixel;
ComPtr<ID3D11Buffer> HybridRenderingUniforms;
ComPtr<ID3D11BlendState> HybridRenderingBlend;

void DestroyHybridRendering()
{
    HybridRenderingImages={};
    HybridRenderingVertex.Reset(); HybridRenderingPixel.Reset(); HybridRenderingUniforms.Reset(); HybridRenderingBlend.Reset();
    HybridRenderingWidth=HybridRenderingHeight=0;
}

bool EnsureHybridRenderingResources()
{
    if (!EnsureRayRadiance()) return false;
    if (HybridRenderingWidth!=CreateInfo.Width || HybridRenderingHeight!=CreateInfo.Height)
    { HybridRenderingImages={}; HybridRenderingWidth=CreateInfo.Width; HybridRenderingHeight=CreateInfo.Height; }
    for (auto& image:HybridRenderingImages) if (!CreateHdrSurface(image,CreateInfo.Width,CreateInfo.Height)) return false;
    if (HybridRenderingVertex && HybridRenderingPixel && HybridRenderingUniforms && HybridRenderingBlend) return true;
    if (!LoadTechniqueShaderSource("DX11","HybridRendering.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,nullptr,true)) return false;
    auto result=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,HybridRenderingVertex.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(hybrid reflections)",result));
    result=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,HybridRenderingPixel.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(hybrid reflections)",result));
    D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(HybridRenderingConstants);
    buffer.Usage=D3D11_USAGE_DYNAMIC; buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    result=Device->CreateBuffer(&buffer,nullptr,HybridRenderingUniforms.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(hybrid reflection constants)",result));
    D3D11_BLEND_DESC blend{}; auto& color=blend.RenderTarget[0]; color.BlendEnable=TRUE;
    color.SrcBlend=color.DestBlend=D3D11_BLEND_ONE; color.BlendOp=color.BlendOpAlpha=D3D11_BLEND_OP_ADD;
    color.SrcBlendAlpha=D3D11_BLEND_ZERO; color.DestBlendAlpha=D3D11_BLEND_ONE;
    color.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_RED|D3D11_COLOR_WRITE_ENABLE_GREEN|D3D11_COLOR_WRITE_ENABLE_BLUE;
    result=Device->CreateBlendState(&blend,HybridRenderingBlend.ReleaseAndGetAddressOf());
    return SUCCEEDED(result) || Fail(HResultMessage("CreateBlendState(hybrid reflections)",result));
}
bool ApplyHybridRendering(const ReflectionDesc& desc)
{
    const auto& frame=FrameRayRadiance; HybridRenderingConstants constants;
    if (!BuildHybridRenderingConstants(desc,FrameRayRadianceSettings,FrameRayRadianceSettings.Hybrid,frame,CreateInfo.Width,CreateInfo.Height,Techniques::HybridRenderingPass::Trace,constants))
        return Fail("DX11 hybrid reflections require finite camera, material and lighting inputs");
    constexpr UINT TextureCount=7,InputCount=TextureCount+RaySceneBufferCount+RayRadianceBufferCount;
    ID3D11ShaderResourceView* empty[InputCount]{};
    Context->OMSetRenderTargets(0,nullptr,nullptr); Context->PSSetShaderResources(0,InputCount,empty);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->VSSetShader(HybridRenderingVertex.Get(),nullptr,0); Context->PSSetShader(HybridRenderingPixel.Get(),nullptr,0);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0);
    const D3D11_VIEWPORT viewport {0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    ID3D11Buffer* buffer=HybridRenderingUniforms.Get(); Context->PSSetConstantBuffers(0,1,&buffer);
    for (const auto pass:HybridRenderingPasses)
    {
        const auto index=static_cast<UINT>(pass); const bool composite=pass==Techniques::HybridRenderingPass::Composite;
        constants.Options.X=index;
        D3D11_MAPPED_SUBRESOURCE mapped{};
        const auto result=Context->Map(buffer,0,D3D11_MAP_WRITE_DISCARD,0,&mapped);
        if (FAILED(result)) return Fail(HResultMessage("Map(hybrid reflection constants)",result));
        std::memcpy(mapped.pData,&constants,sizeof(constants)); Context->Unmap(buffer,0);
        Context->PSSetShaderResources(0,InputCount,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
        ID3D11RenderTargetView* target=composite ? HdrSurfaces[HdrScene].Target.Get() : HybridRenderingImages[index%2].Target.Get();
        Context->OMSetRenderTargets(1,&target,nullptr);
        Context->OMSetBlendState(composite ? HybridRenderingBlend.Get() : nullptr,nullptr,UINT_MAX);
        ID3D11ShaderResourceView* inputs[InputCount]{};
        if (index) inputs[5]=HybridRenderingImages[(index-1)%2].View.Get();
        if (!composite)
        {
            inputs[0]=DepthShaderView.Get(); inputs[1]=GBufferSurfaces[1].View.Get(); inputs[2]=GBufferSurfaces[2].View.Get();
            inputs[3]=GBufferSurfaces[0].View.Get(); inputs[4]=GBufferSurfaces[3].View.Get(); inputs[6]=HdrSurfaces[HdrScene].View.Get();
        }
        for (std::size_t slot=0;slot<RaySceneBufferCount;++slot) inputs[TextureCount+slot]=RaySceneViews[slot].Get();
        for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot) inputs[TextureCount+RaySceneBufferCount+slot]=RayRadianceViews[slot].Get();
        Context->PSSetShaderResources(0,InputCount,inputs); Context->Draw(3,0);
        PendingFrame.ReflectionPasses.push_back(HybridRenderingLogPass(pass)); ++PendingFrame.Statistics.ReflectionPassCount; ++PendingFrame.Statistics.HybridRenderingPassCount;
    }
    Context->PSSetShaderResources(0,InputCount,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    Context->OMSetBlendState(nullptr,nullptr,UINT_MAX); ReflectionApplied=true; return true;
}
