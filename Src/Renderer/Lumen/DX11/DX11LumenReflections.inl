// Reflection history has its own storage and presentation transaction. Diffuse
// GI and specular reflection may share lighting but never overwrite each other.
LumenTemporalHistory LumenReflectionHistory;
LumenRadianceHandle LumenReflectionRays;
ComPtr<ID3D11Buffer> LumenReflectionEstimate;
ComPtr<ID3D11ShaderResourceView> LumenReflectionEstimateView;
ComPtr<ID3D11UnorderedAccessView> LumenReflectionEstimateTarget;
std::array<ComPtr<ID3D11Buffer>,2> LumenReflectionHistoryBuffers;
std::array<ComPtr<ID3D11ShaderResourceView>,2> LumenReflectionHistoryViews;
std::array<ComPtr<ID3D11UnorderedAccessView>,2> LumenReflectionHistoryTargets;
ComPtr<ID3D11ComputeShader> LumenReflectionResolveShader,LumenReflectionTemporalShader;
ComPtr<ID3D11VertexShader> LumenReflectionCompositeVertex;
ComPtr<ID3D11PixelShader> LumenReflectionCompositePixel;
HdrSurface LumenReflectionCompositeTarget;
std::size_t LumenReflectionBytes=0;
void FinishLumenReflections(bool publish)
{
    LumenReflectionHistory.Finish(publish);
    if (LumenReflectionRays) LumenRadianceTraces.erase(LumenReflectionRays.Value);
    LumenReflectionRays={};
}
void DestroyLumenReflections()
{
    FinishLumenReflections(false); LumenReflectionHistory.Invalidate();
    LumenReflectionEstimate.Reset(); LumenReflectionEstimateView.Reset(); LumenReflectionEstimateTarget.Reset();
    LumenReflectionHistoryBuffers={}; LumenReflectionHistoryViews={}; LumenReflectionHistoryTargets={}; LumenReflectionBytes=0;
    LumenReflectionResolveShader.Reset(); LumenReflectionTemporalShader.Reset();
    LumenReflectionCompositeVertex.Reset(); LumenReflectionCompositePixel.Reset(); LumenReflectionCompositeTarget={};
}
bool EnsureLumenReflections(std::size_t bytes)
{
    if (!CreateHdrSurface(LumenReflectionCompositeTarget,CreateInfo.Width,CreateInfo.Height)) return false;
    if (LumenReflectionBytes!=bytes)
    {
        ComPtr<ID3D11Buffer> estimate; ComPtr<ID3D11ShaderResourceView> view; ComPtr<ID3D11UnorderedAccessView> target;
        std::array<ComPtr<ID3D11Buffer>,2> buffers; std::array<ComPtr<ID3D11ShaderResourceView>,2> views; std::array<ComPtr<ID3D11UnorderedAccessView>,2> targets;
        if (!CreateLumenRadianceBuffer(bytes,sizeof(Float4),nullptr,estimate,view,&target)) return false;
        for (unsigned slot=0;slot<buffers.size();++slot)
            if (!CreateLumenRadianceBuffer(bytes,sizeof(LumenTemporalPixel),nullptr,buffers[slot],views[slot],&targets[slot])) return false;
        LumenReflectionEstimate=estimate; LumenReflectionEstimateView=view; LumenReflectionEstimateTarget=target;
        LumenReflectionHistoryBuffers=buffers; LumenReflectionHistoryViews=views; LumenReflectionHistoryTargets=targets;
        LumenReflectionBytes=bytes; LumenReflectionHistory.Invalidate();
    }
    if (LumenReflectionResolveShader && LumenReflectionTemporalShader && LumenReflectionCompositeVertex && LumenReflectionCompositePixel) return true;
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO resolve[]{{"EVANS_LUMEN_REFLECTION_RESOLVE_STAGE","1"},{nullptr,nullptr}},
        temporal[]{{"EVANS_LUMEN_TEMPORAL_STAGE","1"},{"EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE","1"},{nullptr,nullptr}},
        composite[]{{"EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE","1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> spatial,history,vertex,pixel;
    if (!CompileShader("CSReflectionResolve","cs_5_0",spatial,resolve,true) || !CompileShader("CSTemporal","cs_5_0",history,temporal,true) ||
        !CompileShader("VSTemporal","vs_5_0",vertex,composite,true) || !CompileShader("PSTemporalComposite","ps_5_0",pixel,composite,true)) return false;
    auto status=Device->CreateComputeShader(spatial->GetBufferPointer(),spatial->GetBufferSize(),nullptr,LumenReflectionResolveShader.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateComputeShader(history->GetBufferPointer(),history->GetBufferSize(),nullptr,LumenReflectionTemporalShader.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,LumenReflectionCompositeVertex.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,LumenReflectionCompositePixel.ReleaseAndGetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create Lumen reflection reconstruction shaders",status));
}
bool ApplyLumenReflections(const ReflectionDesc& camera)
{
    if (!FrameLumenLighting || LumenReflectionRays) return Fail("Lumen reflections require one consumer of the current lighting cache");
    if (Techniques::UsesTemporalReflections(FrameReflections) && (!VelocityCaptured || (FrameSampleCount>1 && !SamplesVelocityResolved)))
        return Fail("Temporal Lumen reflections require this frame's resolved velocity before tracing");
    LumenRadianceTraceUpload query;
    if (!BuildLumenReflectionQuery(camera,FrameReflections,CreateInfo.Width,CreateInfo.Height,query,LastError)) return false;
    if (FrameGI.Lumen.GlobalDistance.Enabled)
    { query.GlobalDistance=FrameLumenGlobalDistance; query.Trace.GlobalNearDistance=FrameGI.Lumen.GlobalDistance.NearScale*FrameReflections.MaximumDistance; query.Trace.GlobalMaximumSteps=FrameGI.Lumen.GlobalDistance.MaximumSteps; }
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,query,rays)) return false; LumenReflectionRays=rays.Handle;
    const auto source=LumenRadianceTraces.find(rays.Handle.Value); if (source==LumenRadianceTraces.end()) return Fail("Lumen reflection source disappeared");
    const auto& filter=FrameReflections.Lumen; const auto& sky=FrameGI.Lumen.SkyRadiance; LumenReflectionResolveFrame reconstruction;
    if (!BuildLumenReflectionResolveFrame(rays.Info.Reflection,{filter.FilterRadius,filter.NormalCosine,filter.PlaneTolerance,filter.RelativePlaneTolerance,filter.RoughnessTolerance},
        {sky[0],sky[1],sky[2]},FrameReflectionShading,reconstruction,LastError,MakeLumenEnvironmentLobe(FrameGI.Lumen))) return false;
    LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    key.Reflections=FrameReflections; key.Reflected=true; LumenTemporalFrame history;
    if (!BuildLumenReflectionTemporalFrame(camera,FrameReflections,CreateInfo.Width,CreateInfo.Height,LumenReflectionHistory.CanReuse(key),history,LastError) ||
        !EnsureLumenReflections(std::size_t(history.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    ComPtr<ID3D11Buffer> constants[2];
    const void* data[]{&reconstruction,&history}; const UINT sizes[]{sizeof(reconstruction),sizeof(history)};
    for (unsigned slot=0;slot<std::size(constants);++slot)
    {
        D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizes[slot]; desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data[slot]; const auto status=Device->CreateBuffer(&desc,&initial,constants[slot].GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Lumen reflection frame constants",status));
    }
    Context->OMSetRenderTargets(0,nullptr,nullptr);
    ID3D11ShaderResourceView* inputs[]{DepthShaderView.Get(),GBufferSurfaces[1].View.Get(),GBufferSurfaces[0].View.Get(),GBufferSurfaces[3].View.Get(),GBufferSurfaces[2].View.Get(),source->second->OriginView.Get(),source->second->View.Get()};
    ID3D11ShaderResourceView* empty[std::size(inputs)]{}; ID3D11UnorderedAccessView* emptyTarget=nullptr;
    auto* uniform=constants[0].Get(); auto* target=LumenReflectionEstimateTarget.Get();
    Context->CSSetConstantBuffers(0,1,&uniform); Context->CSSetShaderResources(0,UINT(std::size(inputs)),inputs); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
    const auto groups=(history.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    Context->CSSetShader(LumenReflectionResolveShader.Get(),nullptr,0);
    Context->Dispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    Context->CSSetShaderResources(0,UINT(std::size(empty)),empty); Context->CSSetUnorderedAccessViews(0,1,&emptyTarget,nullptr);
    const auto read=LumenReflectionHistory.ReadIndex(),write=LumenReflectionHistory.WriteIndex();
    ID3D11ShaderResourceView* temporal[]{HdrSurfaces[HdrScene].View.Get(),DepthShaderView.Get(),GBufferSurfaces[1].View.Get(),
        MotionSurfaces[MotionVelocity].View?MotionSurfaces[MotionVelocity].View.Get():GBufferSurfaces[1].View.Get(),LumenReflectionHistoryViews[read].Get(),LumenReflectionEstimateView.Get()};
    uniform=constants[1].Get(); target=LumenReflectionHistoryTargets[write].Get(); Context->CSSetConstantBuffers(0,1,&uniform);
    Context->CSSetShaderResources(0,UINT(std::size(temporal)),temporal); Context->CSSetUnorderedAccessViews(0,1,&target,nullptr);
    Context->CSSetShader(LumenReflectionTemporalShader.Get(),nullptr,0);
    Context->Dispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    Context->CSSetShaderResources(0,UINT(std::size(empty)),empty); Context->CSSetUnorderedAccessViews(0,1,&emptyTarget,nullptr); Context->CSSetShader(nullptr,nullptr,0);
    auto* renderTarget=LumenReflectionCompositeTarget.Target.Get(); Context->OMSetRenderTargets(1,&renderTarget,nullptr);
    ID3D11ShaderResourceView* composite[]{HdrSurfaces[HdrScene].View.Get(),nullptr,nullptr,nullptr,LumenReflectionHistoryViews[write].Get()};
    Context->PSSetShaderResources(0,UINT(std::size(composite)),composite); Context->PSSetConstantBuffers(0,1,&uniform);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    const D3D11_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    Context->VSSetShader(LumenReflectionCompositeVertex.Get(),nullptr,0); Context->PSSetShader(LumenReflectionCompositePixel.Get(),nullptr,0); Context->Draw(3,0);
    Context->PSSetShaderResources(0,UINT(std::size(empty)),empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    Context->CopyResource(HdrSurfaces[HdrScene].Texture.Get(),LumenReflectionCompositeTarget.Texture.Get());
    LumenReflectionHistory.Record(key); PendingFrame.Temporal=camera.Sampling;
    for (auto pass:{Techniques::ReflectionPassKind::Trace,Techniques::ReflectionPassKind::LumenReconstruction,Techniques::ReflectionPassKind::LumenTemporal,Techniques::ReflectionPassKind::Composite})
    { PendingFrame.ReflectionPasses.push_back(pass); ++PendingFrame.Statistics.ReflectionPassCount; }
    ReflectionApplied=true; return true;
}
