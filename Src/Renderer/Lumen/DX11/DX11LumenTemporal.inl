LumenTemporalHistory LumenHistory;
std::array<ComPtr<ID3D11Buffer>,2> LumenTemporalBuffers;
std::array<ComPtr<ID3D11ShaderResourceView>,2> LumenTemporalViews;
std::array<ComPtr<ID3D11UnorderedAccessView>,2> LumenTemporalTargets;
ComPtr<ID3D11ComputeShader> LumenTemporalCompute;
ComPtr<ID3D11VertexShader> LumenTemporalVertex;
ComPtr<ID3D11PixelShader> LumenTemporalComposite;
std::size_t LumenTemporalBytes=0;
void DestroyLumenTemporal()
{
    LumenTemporalBuffers={}; LumenTemporalViews={}; LumenTemporalTargets={}; LumenTemporalBytes=0;
    LumenTemporalCompute.Reset(); LumenTemporalVertex.Reset(); LumenTemporalComposite.Reset(); LumenHistory.Invalidate();
}
bool EnsureLumenTemporal(std::size_t bytes)
{
    if (LumenTemporalBytes!=bytes)
    {
        std::array<ComPtr<ID3D11Buffer>,2> buffers; std::array<ComPtr<ID3D11ShaderResourceView>,2> views;
        std::array<ComPtr<ID3D11UnorderedAccessView>,2> targets;
        for (unsigned i=0;i<buffers.size();++i)
            if (!CreateLumenRadianceBuffer(bytes,sizeof(LumenTemporalPixel),nullptr,buffers[i],views[i],&targets[i])) return false;
        LumenTemporalBuffers=buffers; LumenTemporalViews=views; LumenTemporalTargets=targets; LumenTemporalBytes=bytes; LumenHistory.Invalidate();
    }
    if (LumenTemporalCompute && LumenTemporalVertex && LumenTemporalComposite) return true;
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO compute[]{{"EVANS_LUMEN_TEMPORAL_STAGE","1"},{nullptr,nullptr}},composite[]{{"EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE","1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> cs,vs,ps;
    if (!CompileShader("CSTemporal","cs_5_0",cs,compute,true) || !CompileShader("VSTemporal","vs_5_0",vs,composite,true) ||
        !CompileShader("PSTemporalComposite","ps_5_0",ps,composite,true)) return false;
    auto status=Device->CreateComputeShader(cs->GetBufferPointer(),cs->GetBufferSize(),nullptr,LumenTemporalCompute.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,LumenTemporalVertex.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,LumenTemporalComposite.ReleaseAndGetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create Lumen temporal shaders",status));
}
bool ResolveLumenTemporal(const GlobalIlluminationDesc& camera)
{
    const LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    LumenTemporalFrame frame;
    if (!BuildLumenTemporalFrame(camera,FrameGI,CreateInfo.Width,CreateInfo.Height,LumenHistory.CanReuse(key),frame,LastError) ||
        !EnsureLumenTemporal(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(frame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&frame; ComPtr<ID3D11Buffer> constants;
    const auto status=Device->CreateBuffer(&desc,&data,constants.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen temporal constants",status));
    const auto read=LumenHistory.ReadIndex(),write=LumenHistory.WriteIndex(); Context->OMSetRenderTargets(0,nullptr,nullptr);
    auto* uniform=constants.Get(); Context->CSSetConstantBuffers(0,1,&uniform);
    ID3D11ShaderResourceView* inputs[]{LumenGatherTarget.View.Get(),DepthShaderView.Get(),GBufferSurfaces[1].View.Get(),MotionSurfaces[MotionVelocity].View.Get(),LumenTemporalViews[read].Get()};
    Context->CSSetShaderResources(0,UINT(std::size(inputs)),inputs); auto* output=LumenTemporalTargets[write].Get();
    Context->CSSetUnorderedAccessViews(0,1,&output,nullptr); Context->CSSetShader(LumenTemporalCompute.Get(),nullptr,0);
    const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    Context->Dispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    ID3D11UnorderedAccessView* emptyTarget=nullptr; ID3D11ShaderResourceView* empty[std::size(inputs)]{};
    Context->CSSetUnorderedAccessViews(0,1,&emptyTarget,nullptr); Context->CSSetShaderResources(0,UINT(std::size(empty)),empty); Context->CSSetShader(nullptr,nullptr,0);
    auto* target=LumenGatherTarget.Target.Get(); Context->OMSetRenderTargets(1,&target,nullptr);
    ID3D11ShaderResourceView* composite[]{HdrSurfaces[HdrScene].View.Get(),nullptr,nullptr,nullptr,LumenTemporalViews[write].Get()};
    Context->PSSetShaderResources(0,UINT(std::size(composite)),composite); Context->PSSetConstantBuffers(0,1,&uniform);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    const D3D11_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    Context->VSSetShader(LumenTemporalVertex.Get(),nullptr,0); Context->PSSetShader(LumenTemporalComposite.Get(),nullptr,0); Context->Draw(3,0);
    Context->PSSetShaderResources(0,UINT(std::size(empty)),empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    Context->CopyResource(HdrSurfaces[HdrScene].Texture.Get(),LumenGatherTarget.Texture.Get());
    LumenHistory.Record(key); PendingFrame.Temporal=camera.Sampling; PendingFrame.Statistics.GlobalIlluminationPassCount+=2; return true;
}
