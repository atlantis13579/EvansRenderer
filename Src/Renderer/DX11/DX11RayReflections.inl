// DX11 scene/material buffers are immutable snapshots. Only lighting storage
// changes for a light edit; the shared visibility buffers also serve ray AO.
std::array<HdrSurface,2> RayReflectionImages;
int RayReflectionWidth=0,RayReflectionHeight=0;
ComPtr<ID3D11VertexShader> RayReflectionVertex;
ComPtr<ID3D11PixelShader> RayReflectionPixel;
ComPtr<ID3D11Buffer> RayReflectionConstants;
ComPtr<ID3D11BlendState> RayReflectionBlend;

void DestroyRayReflections()
{
    RayReflectionImages={};
    RayReflectionVertex.Reset(); RayReflectionPixel.Reset(); RayReflectionConstants.Reset(); RayReflectionBlend.Reset();
    RayReflectionWidth=RayReflectionHeight=0;
}

bool EnsureRayReflectionResources()
{
    if (!EnsureRayRadiance()) return false;
    if (RayReflectionWidth!=CreateInfo.Width || RayReflectionHeight!=CreateInfo.Height)
    { RayReflectionImages={}; RayReflectionWidth=CreateInfo.Width; RayReflectionHeight=CreateInfo.Height; }
    for (auto& image:RayReflectionImages) if (!CreateHdrSurface(image,CreateInfo.Width,CreateInfo.Height)) return false;
    if (RayReflectionVertex && RayReflectionPixel && RayReflectionConstants && RayReflectionBlend) return true;
    if (!LoadTechniqueShaderSource("DX11","RayTracedReflections.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,nullptr,true)) return false;
    auto result=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,RayReflectionVertex.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(ray reflections)",result));
    result=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,RayReflectionPixel.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(ray reflections)",result));
    D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(RayTracedReflectionConstants);
    buffer.Usage=D3D11_USAGE_DYNAMIC; buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    result=Device->CreateBuffer(&buffer,nullptr,RayReflectionConstants.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(ray reflection constants)",result));
    D3D11_BLEND_DESC blend{}; auto& color=blend.RenderTarget[0]; color.BlendEnable=TRUE;
    color.SrcBlend=color.DestBlend=D3D11_BLEND_ONE; color.BlendOp=color.BlendOpAlpha=D3D11_BLEND_OP_ADD;
    color.SrcBlendAlpha=D3D11_BLEND_ZERO; color.DestBlendAlpha=D3D11_BLEND_ONE;
    color.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_RED|D3D11_COLOR_WRITE_ENABLE_GREEN|D3D11_COLOR_WRITE_ENABLE_BLUE;
    result=Device->CreateBlendState(&blend,RayReflectionBlend.ReleaseAndGetAddressOf());
    return SUCCEEDED(result) || Fail(HResultMessage("CreateBlendState(ray reflections)",result));
}
bool ApplyRayReflections(const ReflectionDesc& desc)
{
    const auto& frame=FrameRayRadiance; RayTracedReflectionConstants constants;
    if (!BuildRayTracedReflectionConstants(desc,FrameRayRadianceSettings,FrameReflections.RayTracing,*frame.Geometry,*frame.Shading,
        frame.Lighting,*frame.Lights,CreateInfo.Width,CreateInfo.Height,Techniques::RayReflectionPass::Trace,constants))
        return Fail("DX11 ray reflections require finite camera, material and lighting inputs");
    constexpr UINT TextureCount=7,InputCount=TextureCount+RaySceneBufferCount+RayRadianceBufferCount;
    ID3D11ShaderResourceView* empty[InputCount]{};
    Context->OMSetRenderTargets(0,nullptr,nullptr); Context->PSSetShaderResources(0,InputCount,empty);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->VSSetShader(RayReflectionVertex.Get(),nullptr,0); Context->PSSetShader(RayReflectionPixel.Get(),nullptr,0);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0);
    const D3D11_VIEWPORT viewport {0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    ID3D11Buffer* buffer=RayReflectionConstants.Get(); Context->PSSetConstantBuffers(0,1,&buffer);
    for (const auto pass:RayReflectionPasses)
    {
        const auto index=static_cast<UINT>(pass); const bool composite=pass==Techniques::RayReflectionPass::Composite;
        constants.Options.X=index;
        D3D11_MAPPED_SUBRESOURCE mapped{};
        const auto result=Context->Map(buffer,0,D3D11_MAP_WRITE_DISCARD,0,&mapped);
        if (FAILED(result)) return Fail(HResultMessage("Map(ray reflection constants)",result));
        std::memcpy(mapped.pData,&constants,sizeof(constants)); Context->Unmap(buffer,0);
        Context->PSSetShaderResources(0,InputCount,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
        ID3D11RenderTargetView* target=composite ? HdrSurfaces[HdrScene].Target.Get() : RayReflectionImages[index%2].Target.Get();
        Context->OMSetRenderTargets(1,&target,nullptr);
        Context->OMSetBlendState(composite ? RayReflectionBlend.Get() : nullptr,nullptr,UINT_MAX);
        ID3D11ShaderResourceView* inputs[InputCount]{};
        if (index) inputs[5]=RayReflectionImages[(index-1)%2].View.Get();
        if (!composite)
        {
            inputs[0]=DepthShaderView.Get(); inputs[1]=GBufferSurfaces[1].View.Get(); inputs[2]=GBufferSurfaces[2].View.Get();
            inputs[3]=GBufferSurfaces[0].View.Get(); inputs[4]=GBufferSurfaces[3].View.Get(); inputs[6]=HdrSurfaces[HdrScene].View.Get();
        }
        for (std::size_t slot=0;slot<RaySceneBufferCount;++slot) inputs[TextureCount+slot]=RaySceneViews[slot].Get();
        for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot) inputs[TextureCount+RaySceneBufferCount+slot]=RayRadianceViews[slot].Get();
        Context->PSSetShaderResources(0,InputCount,inputs); Context->Draw(3,0);
        PendingFrame.ReflectionPasses.push_back(RayReflectionLogPass(pass)); ++PendingFrame.Statistics.ReflectionPassCount;
    }
    Context->PSSetShaderResources(0,InputCount,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    Context->OMSetBlendState(nullptr,nullptr,UINT_MAX); ReflectionApplied=true; return true;
}
