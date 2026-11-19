static constexpr UINT LumenVolumeConstantSlot=8,LumenVolumeTextureSlot=19,LumenVolumeSamplerSlot=4;
LumenTranslucencyFrame LumenVolumeFrame;
LumenRadianceHandle LumenVolumeRays;
LumenProbeHandle LumenVolumeProbes;
HdrSurface LumenVolumeAtlas;
ComPtr<ID3D11VertexShader> LumenVolumeVertex;
ComPtr<ID3D11PixelShader> LumenVolumePixel;
ComPtr<ID3D11Buffer> LumenVolumeUniform;
ComPtr<ID3D11ComputeShader> LumenVolumeSupportCompute;
ComPtr<ID3D11Buffer> LumenVolumeSupport;
ComPtr<ID3D11ShaderResourceView> LumenVolumeSupportView;
void ReleaseLumenTranslucencyResults()
{
    if (LumenVolumeProbes) LumenProbes.erase(LumenVolumeProbes.Value);
    if (LumenVolumeRays) LumenRadianceTraces.erase(LumenVolumeRays.Value);
    LumenVolumeProbes={}; LumenVolumeRays={}; LumenVolumeFrame.Grid.W=0;
}
void DestroyLumenTranslucency()
{
    ReleaseLumenTranslucencyResults(); LumenVolumeAtlas={};
    LumenVolumeVertex.Reset(); LumenVolumePixel.Reset(); LumenVolumeUniform.Reset();
    LumenVolumeSupportCompute.Reset(); LumenVolumeSupport.Reset(); LumenVolumeSupportView.Reset();
}
bool PrepareLumenVolumeSupport(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    const auto source=LumenRadianceTraces.find(probes.SourceHandle.Value);
    if (source==LumenRadianceTraces.end() || !source->second->TraceDistance) return Fail("Lumen volume support lost its immutable distance source");
    LumenDistanceTraceFrame frame;
    if (!BuildLumenTranslucencySupportFrame(source->second->TraceDistance->Info,upload,frame,LastError)) return false;
    if (!LumenVolumeSupportCompute)
    {
        if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO options[]{{"EVANS_LUMEN_VOLUME_VISIBILITY_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> code;
        if (!CompileShader("CSVolumeSupport","cs_5_0",code,options,true)) return false;
        const auto status=Device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,LumenVolumeSupportCompute.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Lumen volume visibility shader",status));
    }
    LumenVolumeSupport.Reset(); LumenVolumeSupportView.Reset(); ComPtr<ID3D11UnorderedAccessView> target;
    if (!CreateLumenRadianceBuffer(std::size_t(frame.Counts.X)*sizeof(float),sizeof(float),nullptr,LumenVolumeSupport,LumenVolumeSupportView,&target)) return false;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(frame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&frame; ComPtr<ID3D11Buffer> uniform;
    const auto status=Device->CreateBuffer(&desc,&data,uniform.GetAddressOf()); if (FAILED(status)) return Fail(HResultMessage("Create Lumen volume visibility constants",status));
    auto* parameters=uniform.Get(); auto* output=target.Get();
    ID3D11ShaderResourceView* inputs[]{source->second->TraceDistance->View.Get(),source->second->OriginView.Get()};
    Context->CSSetShader(LumenVolumeSupportCompute.Get(),nullptr,0); Context->CSSetConstantBuffers(0,1,&parameters);
    Context->CSSetShaderResources(0,2,inputs); Context->CSSetUnorderedAccessViews(0,1,&output,nullptr);
    Context->Dispatch((frame.Counts.X+LumenRadianceThreads-1)/LumenRadianceThreads,1,1);
    parameters=nullptr; output=nullptr; ID3D11ShaderResourceView* empty[2]{};
    Context->CSSetUnorderedAccessViews(0,1,&output,nullptr); Context->CSSetShaderResources(0,2,empty); Context->CSSetConstantBuffers(0,1,&parameters); Context->CSSetShader(nullptr,nullptr,0);
    return true;
}
bool PackLumenTranslucency(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    if (!PrepareLumenVolumeSupport(probes,upload)) return false;
    if (!CreateHdrSurface(LumenVolumeAtlas,int(LumenVolumeFrame.Atlas.X),int(LumenVolumeFrame.Atlas.Y))) return false;
    if (!LumenVolumeVertex || !LumenVolumePixel)
    {
        if (!LoadTechniqueShaderSource("DX11","LumenTranslucency.hlsl",ShaderCode,LastError)) return false;
        ComPtr<ID3DBlob> vertex,pixel;
        if (!CompileShader("VSVolume","vs_5_0",vertex,nullptr,true) || !CompileShader("PSVolume","ps_5_0",pixel,nullptr,true)) return false;
        auto status=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,LumenVolumeVertex.ReleaseAndGetAddressOf());
        if (SUCCEEDED(status)) status=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,LumenVolumePixel.ReleaseAndGetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Lumen translucent volume shaders",status));
    }
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(LumenVolumeFrame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&LumenVolumeFrame;
    const auto status=Device->CreateBuffer(&desc,&data,LumenVolumeUniform.ReleaseAndGetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen translucent volume constants",status));
    // The atlas is no longer a sampled input when it becomes this frame's target.
    ID3D11ShaderResourceView* empty=nullptr; Context->PSSetShaderResources(LumenVolumeTextureSlot,1,&empty);
    auto* target=LumenVolumeAtlas.Target.Get(); Context->OMSetRenderTargets(1,&target,nullptr);
    ID3D11ShaderResourceView* sources[]{probes.View.Get(),LumenVolumeSupportView.Get()}; Context->PSSetShaderResources(0,2,sources);
    auto* uniform=LumenVolumeUniform.Get(); Context->PSSetConstantBuffers(0,1,&uniform);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    const D3D11_VIEWPORT viewport{0,0,LumenVolumeFrame.Atlas.X,LumenVolumeFrame.Atlas.Y,0,1}; Context->RSSetViewports(1,&viewport);
    Context->VSSetShader(LumenVolumeVertex.Get(),nullptr,0); Context->PSSetShader(LumenVolumePixel.Get(),nullptr,0); Context->Draw(3,0);
    ID3D11ShaderResourceView* noSources[2]{}; Context->PSSetShaderResources(0,2,noSources); Context->OMSetRenderTargets(0,nullptr,nullptr); return true;
}
bool PrepareLumenTranslucency(const GlobalIlluminationDesc& desc)
{
    const auto& settings=FrameGI.Lumen;
    if ((!desc.Translucency || !settings.Translucency.Enabled) && !Hair.LumenReceiver) return true;
    LumenRadianceTraceUpload trace;
    if (!BuildLumenTranslucencyFrame(desc.ViewProjection,CreateInfo.Width,CreateInfo.Height,FrameGI,LumenVolumeFrame,trace,LastError)) return false;
    if (settings.GlobalDistance.Enabled)
    { trace.GlobalDistance=FrameLumenGlobalDistance; trace.Trace.GlobalNearDistance=settings.GlobalDistance.NearScale*FrameGI.MaximumDistance; trace.Trace.GlobalMaximumSteps=settings.GlobalDistance.MaximumSteps; }
    if (settings.WorldCache.Enabled) trace.WorldCache=FrameLumenWorldCache;
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,trace,rays)) return false; LumenVolumeRays=rays.Handle;
    const auto& sky=settings.SkyRadiance; LumenProbeIntegrationResult probes;
    if (!IntegrateLumenProbes(rays.Handle,{{sky[0],sky[1],sky[2]},false,MakeLumenEnvironmentLobe(settings)},probes)) return false; LumenVolumeProbes=probes.Handle;
    if (!PackLumenTranslucency(*LumenProbes.at(probes.Handle.Value),trace)) return false;
    ++PendingFrame.Statistics.GlobalIlluminationPassCount; PendingFrame.Statistics.LumenTranslucencyProbeCount=unsigned(trace.Probes.Origins.size()); return true;
}
