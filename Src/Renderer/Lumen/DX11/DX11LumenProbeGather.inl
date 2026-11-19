LumenLightingHandle FrameLumenLighting;
LumenWorldCacheBinding FrameLumenWorldCache;
LumenDistanceSceneHandle FrameLumenGlobalDistance;
LumenRadianceHandle LumenGIRays;
LumenProbeHandle LumenGIProbes;
bool LumenGatherIndirect=false;
HdrSurface LumenGatherTarget;
ComPtr<ID3D11Buffer> LumenVisibilityBuffer;
ComPtr<ID3D11ShaderResourceView> LumenVisibilityView;
ComPtr<ID3D11UnorderedAccessView> LumenVisibilityTarget;
ComPtr<ID3D11ComputeShader> LumenVisibilityShader;
std::size_t LumenVisibilityBytes=0;
ComPtr<ID3D11VertexShader> LumenGatherVertex;
ComPtr<ID3D11PixelShader> LumenGatherPixel;
void DestroyLumenGather() { ReleaseLumenGIResults(); DestroyLumenTemporal(); DestroyLumenReflections(); DestroyLumenTranslucency(); DestroyLumenHairReceiver(); LumenGatherIndirect=false; FrameLumenLighting={}; FrameLumenWorldCache={}; FrameLumenGlobalDistance={}; LumenGatherTarget={}; LumenGatherVertex.Reset(); LumenGatherPixel.Reset(); LumenVisibilityBuffer.Reset(); LumenVisibilityView.Reset(); LumenVisibilityTarget.Reset(); LumenVisibilityShader.Reset(); LumenVisibilityBytes=0; }
bool EnsureLumenGather(const LumenProbeGatherFrame& frame)
{
    if (!CreateHdrSurface(LumenGatherTarget,CreateInfo.Width,CreateInfo.Height)) return false;
    const auto bytes=LumenProbeVisibilityBytes(frame);
    if (LumenVisibilityBytes!=bytes)
    {
        ComPtr<ID3D11Buffer> buffer; ComPtr<ID3D11ShaderResourceView> view; ComPtr<ID3D11UnorderedAccessView> target;
        if (!CreateLumenRadianceBuffer(bytes,sizeof(std::uint32_t),nullptr,buffer,view,&target)) return false;
        LumenVisibilityBuffer=buffer; LumenVisibilityView=view; LumenVisibilityTarget=target; LumenVisibilityBytes=bytes;
    }
    if (!LumenVisibilityShader)
    {
        if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_GATHER_VISIBILITY_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
        if (!CompileShader("CSProbeVisibility","cs_5_0",binary,macros,true)) return false;
        const auto status=Device->CreateComputeShader(binary->GetBufferPointer(),binary->GetBufferSize(),nullptr,LumenVisibilityShader.GetAddressOf());
        if (FAILED(status)) return Fail(HResultMessage("Create Lumen probe visibility shader",status));
    }
    if (LumenGatherVertex && LumenGatherPixel) return true;
    if (!LoadTechniqueShaderSource("DX11","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_PROBE_GATHER_STAGE","1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSGather","vs_5_0",vertex,macros,true) || !CompileShader("PSGather","ps_5_0",pixel,macros,true)) return false;
    auto status=Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,LumenGatherVertex.ReleaseAndGetAddressOf());
    if (SUCCEEDED(status)) status=Device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,LumenGatherPixel.ReleaseAndGetAddressOf());
    return SUCCEEDED(status) || Fail(HResultMessage("Create Lumen gather shaders",status));
}
bool ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if ((GIActive && (!Techniques::UsesLumen(FrameGI) || GI.Resolved)) || !HdrSceneActive || !SceneColorWritten || !GBufferWritten ||
        (FrameSampleCount>1 && (!SamplesOpaqueResolved || SamplesColorRestored || SamplesTransparentWritten)) ||
        (DeferredActive && !DeferredResolved) || (AmbientOcclusionActive && !AmbientOcclusionApplied) ||
        ReflectionApplied || VolumetricFog.Resolved || LightShafts.Resolved || TemporalAA.Resolved || MotionBlurResolved || DepthOfFieldResolved ||
        RefractionResolved || !Hair.Complete() || !SubsurfaceComplete())
        return Fail("Lumen gather requires resolved opaque HDR materials before reflections, volumes and transparency");
    if (LumenHasTransparentMainDraws(PendingFrame)) return Fail("Lumen gather must precede transparent draws");
    const auto found=LumenProbes.find(handle.Value); if (found==LumenProbes.end()) return Fail("Unknown Lumen gather probes");
    const auto& record=*found->second; if (!FrameLumen.CanGather(handle,record.SourceHandle,LastError)) return false;
    const auto source=LumenRadianceTraces.find(record.SourceHandle.Value); if (source==LumenRadianceTraces.end()) return Fail("Lumen gather source was released");
    LumenProbeGatherFrame frame;
    if (!BuildLumenProbeGatherFrame(record.Info,source->second->Lighting->Distance->Info,settings,FrameReflectionShading,frame,LastError) || !EnsureLumenGather(frame)) return false;
    frame.Model.Z=LumenGatherIndirect?1u:0u;
    D3D11_BUFFER_DESC desc{}; desc.ByteWidth=sizeof(frame); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA data{}; data.pSysMem=&frame; ComPtr<ID3D11Buffer> uniform;
    const auto status=Device->CreateBuffer(&desc,&data,uniform.GetAddressOf());
    if (FAILED(status)) return Fail(HResultMessage("Create Lumen gather constants",status));
    Context->OMSetRenderTargets(0,nullptr,nullptr);
    if (settings.Visibility.Enabled)
    {
        auto* constants=uniform.Get(); Context->CSSetConstantBuffers(0,1,&constants);
        ID3D11ShaderResourceView* views[]{source->second->Lighting->Distance->View.Get(),source->second->OriginView.Get(),DepthShaderView.Get(),GBufferSurfaces[1].View.Get()};
        Context->CSSetShaderResources(0,UINT(std::size(views)),views);
        auto* output=LumenVisibilityTarget.Get(); Context->CSSetUnorderedAccessViews(0,1,&output,nullptr);
        Context->CSSetShader(LumenVisibilityShader.Get(),nullptr,0);
        const auto groups=(frame.Trace.Counts.X+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
        Context->Dispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
        ID3D11UnorderedAccessView* emptyTarget=nullptr; Context->CSSetUnorderedAccessViews(0,1,&emptyTarget,nullptr);
        ID3D11ShaderResourceView* emptyViews[std::size(views)]{}; Context->CSSetShaderResources(0,UINT(std::size(views)),emptyViews);
        Context->CSSetShader(nullptr,nullptr,0);
    }
    auto* target=LumenGatherTarget.Target.Get(); Context->OMSetRenderTargets(1,&target,nullptr);
    ID3D11ShaderResourceView* inputs[]{HdrSurfaces[HdrScene].View.Get(),DepthShaderView.Get(),GBufferSurfaces[1].View.Get(),
        GBufferSurfaces[0].View.Get(),GBufferSurfaces[3].View.Get(),source->second->OriginView.Get(),record.View.Get(),LumenVisibilityView.Get()};
    Context->PSSetShaderResources(0,UINT(std::size(inputs)),inputs);
    auto* constants=uniform.Get(); Context->PSSetConstantBuffers(0,1,&constants);
    Context->IASetInputLayout(nullptr); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->RSSetState(SolidRasterState.Get()); Context->OMSetDepthStencilState(DepthDisabledState.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    const D3D11_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    Context->VSSetShader(LumenGatherVertex.Get(),nullptr,0); Context->PSSetShader(LumenGatherPixel.Get(),nullptr,0); Context->Draw(3,0);
    ID3D11ShaderResourceView* empty[std::size(inputs)]{}; Context->PSSetShaderResources(0,UINT(std::size(empty)),empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    if (!LumenGatherIndirect) Context->CopyResource(HdrSurfaces[HdrScene].Texture.Get(),LumenGatherTarget.Texture.Get());
    FrameLumen.Gather(); ++PendingFrame.Statistics.GlobalIlluminationPassCount; return true;
}

// The ordinary GI pass owns these transient results; diagnostic APIs keep their
// explicit handles. FinishLumenFrame releases only this automatically owned pair.
bool ApplyLumenGI(const GlobalIlluminationDesc& desc)
{
    if (!FrameLumenLighting || LumenGIRays || LumenGIProbes) return Fail("Lumen requires a current surface cache and one frame consumer");
    const auto& settings=FrameGI.Lumen; LumenRadianceTraceUpload trace;
    trace.Screen.Enabled=true; trace.Screen.ViewProjection=desc.ViewProjection; trace.Screen.Width=CreateInfo.Width; trace.Screen.Height=CreateInfo.Height;
    trace.Screen.TileSide=settings.TileSide; trace.Screen.NormalBias=FrameGI.NormalBias;
    trace.Screen.Adaptive=settings.AdaptiveProbes; trace.Screen.AdaptiveLevels=settings.AdaptiveProbeLevels; trace.Screen.MinimumNormalCosine=FrameGI.NormalThreshold; trace.Screen.PlaneTolerance=FrameGI.DepthTolerance;
    trace.Probes.Sequence=settings.Temporal.Enabled?desc.Sampling.SampleIndex:0; trace.Probes.RaysPerProbe=settings.RaysPerProbe; trace.Probes.MaximumDistance=FrameGI.MaximumDistance;
    if (settings.GlobalDistance.Enabled)
    { trace.GlobalDistance=FrameLumenGlobalDistance; trace.Trace.GlobalNearDistance=settings.GlobalDistance.NearScale*FrameGI.MaximumDistance; trace.Trace.GlobalMaximumSteps=settings.GlobalDistance.MaximumSteps; }
    trace.Trace.MaximumSteps=FrameGI.MaximumSteps; trace.Hybrid=settings.ScreenTrace; if (settings.WorldCache.Enabled) trace.WorldCache=FrameLumenWorldCache;
    LumenRadianceTraceResult rays; if (!TraceLumenRadiance(FrameLumenLighting,trace,rays)) return false; LumenGIRays=rays.Handle;
    LumenProbeIntegrationResult probes; const auto& sky=settings.SkyRadiance;
    if (!IntegrateLumenProbes(rays.Handle,{{sky[0],sky[1],sky[2]},false,MakeLumenEnvironmentLobe(settings)},probes)) return false; LumenGIProbes=probes.Handle;
    LumenProbeGatherSettings gather; gather.Intensity=FrameGI.Intensity; gather.MinimumNormalCosine=FrameGI.NormalThreshold; gather.PlaneTolerance=FrameGI.DepthTolerance; gather.Visibility=settings.ProbeVisibility;
    LumenGatherIndirect=settings.Temporal.Enabled;
    const bool gathered=ResolveLumenProbes(probes.Handle,gather); LumenGatherIndirect=false;
    if (!gathered || (settings.Temporal.Enabled && !ResolveLumenTemporal(desc))) return false;
    if (!PrepareLumenTranslucency(desc) || !ResolveLumenHairReceiver()) return false;
    GI.Resolved=true; return true;
}
void ReleaseLumenGIResults()
{
    ReleaseLumenTranslucencyResults();
    if (LumenGIProbes) LumenProbes.erase(LumenGIProbes.Value);
    if (LumenGIRays) LumenRadianceTraces.erase(LumenGIRays.Value);
    LumenGIProbes={}; LumenGIRays={};
}
