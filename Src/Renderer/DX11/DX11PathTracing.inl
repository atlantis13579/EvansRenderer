// FP32 accumulation is separate from the display HDR target. The pending mean
// and center depth are never sampled while bound as render targets.
PathTracingFrameState PathFrame;
std::array<HdrSurface,2> PathMeans;
HdrSurface PathDepth;
ComPtr<ID3D11VertexShader> PathVertex;
std::array<ComPtr<ID3D11PixelShader>,2> PathPixels;
ComPtr<ID3D11Buffer> PathConstants;
ComPtr<ID3D11DepthStencilState> PathDepthWrite;

bool SupportsPathTracing() const
{
    if (!Device) return false;
    constexpr UINT required=D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_SHADER_LOAD;
    for (auto format:{DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32_FLOAT})
    { UINT support=0; if (FAILED(Device->CheckFormatSupport(format,&support)) || (support&required)!=required) return false; }
    return true;
}
void DestroyPathTracing()
{
    PathMeans={}; PathDepth={}; PathVertex.Reset(); PathPixels={}; PathConstants.Reset(); PathDepthWrite.Reset();
    PathFrame={};
}
bool CreatePathTarget(HdrSurface& surface,DXGI_FORMAT format)
{
    if (surface.Texture) return true;
    HdrSurface candidate;
    D3D11_TEXTURE2D_DESC create{}; create.Width=CreateInfo.Width; create.Height=CreateInfo.Height;
    create.MipLevels=create.ArraySize=create.SampleDesc.Count=1; create.Format=format;
    create.Usage=D3D11_USAGE_DEFAULT; create.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
    auto result=Device->CreateTexture2D(&create,nullptr,candidate.Texture.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateTexture2D(path tracing)",result));
    result=Device->CreateRenderTargetView(candidate.Texture.Get(),nullptr,candidate.Target.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateRenderTargetView(path tracing)",result));
    result=Device->CreateShaderResourceView(candidate.Texture.Get(),nullptr,candidate.View.GetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateShaderResourceView(path tracing)",result));
    candidate.Width=CreateInfo.Width; candidate.Height=CreateInfo.Height; surface=std::move(candidate); return true;
}
bool EnsurePathTracing()
{
    if (!SupportsPathTracing()) return Fail("DX11 path tracing requires FP32 render/load targets");
    if (!EnsureRayRadiance()) return false;
    if (PathDepth.Width!=CreateInfo.Width || PathDepth.Height!=CreateInfo.Height)
    { PathMeans={}; PathDepth={}; PathFrame.History.Invalidate(); }
    for (auto& target:PathMeans) if (!CreatePathTarget(target,DXGI_FORMAT_R32G32B32A32_FLOAT)) return false;
    if (!CreatePathTarget(PathDepth,DXGI_FORMAT_R32_FLOAT)) return false;
    if (PathVertex && PathPixels[0] && PathPixels[1] && PathConstants && PathDepthWrite) return true;
    if (!LoadTechniqueShaderSource("DX11","PathTracing.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true)) return false;
    auto result=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,PathVertex.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateVertexShader(path tracing)",result));
    for (std::size_t phase=0;phase<PathPixels.size();++phase)
    {
        ComPtr<ID3DBlob> pixel;
        if (!CompileShader(phase ? "PSCompose" : "PSMain","ps_5_0",pixel,nullptr,true)) return false;
        result=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,PathPixels[phase].ReleaseAndGetAddressOf());
        if (FAILED(result)) return Fail(HResultMessage("CreatePixelShader(path tracing)",result));
    }
    D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(PathTracingConstants); buffer.Usage=D3D11_USAGE_DYNAMIC;
    buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    result=Device->CreateBuffer(&buffer,nullptr,PathConstants.ReleaseAndGetAddressOf());
    if (FAILED(result)) return Fail(HResultMessage("CreateBuffer(path tracing)",result));
    D3D11_DEPTH_STENCIL_DESC depth{}; depth.DepthEnable=TRUE;
    depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL; depth.DepthFunc=D3D11_COMPARISON_ALWAYS;
    result=Device->CreateDepthStencilState(&depth,PathDepthWrite.ReleaseAndGetAddressOf());
    return SUCCEEDED(result) || Fail(HResultMessage("CreateDepthStencilState(path composition)",result));
}
bool TracePaths(const PathTracingDesc& desc)
{
    PathTracingConstants constants;
    if (!PathFrame.Prepare(desc,FrameRayRadianceSettings,FrameRayRadiance,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("DX11 path frame requires matched scene inputs and a finite invertible camera");
    const auto plan=PathFrame.History.GetPlan();
    D3D11_MAPPED_SUBRESOURCE mapped{};
    const auto result=Context->Map(PathConstants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped);
    if (FAILED(result)) return Fail(HResultMessage("Map(path tracing)",result));
    std::memcpy(mapped.pData,&constants,sizeof(constants)); Context->Unmap(PathConstants.Get(),0);
    constexpr UINT ImageCount=2,InputCount=ImageCount+RaySceneBufferCount+RayRadianceBufferCount;
    ID3D11ShaderResourceView* empty[InputCount]{};
    Context->OMSetRenderTargets(0,nullptr,nullptr); Context->PSSetShaderResources(0,InputCount,empty);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->VSSetShader(PathVertex.Get(),nullptr,0); Context->RSSetState(SolidRasterState.Get());
    Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    const D3D11_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    ID3D11Buffer* buffer=PathConstants.Get(); Context->PSSetConstantBuffers(0,1,&buffer);
    for (unsigned phase=0;phase<2;++phase)
    {
        Context->PSSetShaderResources(0,InputCount,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
        ID3D11ShaderResourceView* inputs[InputCount]{};
        inputs[0]=phase ? PathMeans[plan.WriteIndex].View.Get() : plan.PreviousSamples ? PathMeans[plan.ReadIndex].View.Get() : nullptr;
        inputs[1]=phase ? PathDepth.View.Get() : nullptr;
        for (std::size_t slot=0;slot<RaySceneBufferCount;++slot) inputs[ImageCount+slot]=RaySceneViews[slot].Get();
        for (std::size_t slot=0;slot<RayRadianceBufferCount;++slot) inputs[ImageCount+RaySceneBufferCount+slot]=RayRadianceViews[slot].Get();
        Context->PSSetShaderResources(0,InputCount,inputs); Context->PSSetShader(PathPixels[phase].Get(),nullptr,0);
        Context->OMSetDepthStencilState(phase ? PathDepthWrite.Get() : DepthDisabledState.Get(),0);
        ID3D11RenderTargetView* targets[]{phase ? HdrSurfaces[HdrScene].Target.Get() : PathMeans[plan.WriteIndex].Target.Get(),PathDepth.Target.Get()};
        Context->OMSetRenderTargets(phase ? 1 : 2,targets,phase ? DepthTarget.Get() : nullptr); Context->Draw(3,0);
    }
    Context->OMSetRenderTargets(0,nullptr,nullptr); Context->PSSetShaderResources(0,InputCount,empty);
    Context->OMSetDepthStencilState(DepthWriteState.Get(),0);
    SceneColorWritten=true; return PathFrame.Resolve(PendingFrame.Statistics);
}
