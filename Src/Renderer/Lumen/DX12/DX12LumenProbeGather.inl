LumenLightingHandle FrameLumenLighting;
LumenWorldCacheBinding FrameLumenWorldCache;
LumenDistanceSceneHandle FrameLumenGlobalDistance;
LumenRadianceHandle LumenGIRays;
LumenProbeHandle LumenGIProbes;
bool LumenGatherIndirect=false;
HdrSurface LumenGatherTarget;
ComPtr<ID3D12Resource> LumenVisibilityBuffer;
D3D12_RESOURCE_STATES LumenVisibilityState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS,LumenVisibilityFrameState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
std::size_t LumenVisibilityBytes=0;
ComPtr<ID3D12RootSignature> LumenVisibilityRoot;
ComPtr<ID3D12PipelineState> LumenVisibilityPipeline;
ComPtr<ID3D12RootSignature> LumenGatherRoot;
ComPtr<ID3D12PipelineState> LumenGatherPipeline;
void DestroyLumenGather() { ReleaseLumenGIResults(); DestroyLumenTemporal(); DestroyLumenReflections(); DestroyLumenTranslucency(); DestroyLumenHairReceiver(); LumenGatherIndirect=false; FrameLumenLighting={}; FrameLumenWorldCache={}; FrameLumenGlobalDistance={}; LumenGatherTarget={}; LumenGatherRoot.Reset(); LumenGatherPipeline.Reset(); LumenVisibilityBuffer.Reset(); LumenVisibilityRoot.Reset(); LumenVisibilityPipeline.Reset(); LumenVisibilityBytes=0; }
bool EnsureLumenGather(const LumenProbeGatherFrame& frame)
{
    if (!CreateHdrSurface(LumenGatherTarget,CreateInfo.Width,CreateInfo.Height)) return false;
    const auto bytes=LumenProbeVisibilityBytes(frame);
    if (LumenVisibilityBytes!=bytes)
    {
        ComPtr<ID3D12Resource> buffer; if (!CreateLumenRadianceOutput(bytes,buffer)) return false;
        LumenVisibilityBuffer=buffer; LumenVisibilityBytes=bytes; LumenVisibilityState=LumenVisibilityFrameState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    if (!LumenVisibilityRoot)
    {
        D3D12_ROOT_PARAMETER parameters[8]{}; D3D12_DESCRIPTOR_RANGE ranges[2]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        for (UINT slot=1;slot<3;++slot) { parameters[slot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[slot].Descriptor={slot-1,0}; }
        for (UINT slot=0;slot<2;++slot)
        {
            ranges[slot]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,slot+2,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[slot+3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[slot+3].DescriptorTable={1,&ranges[slot]};
        }
        parameters[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        LumenHardwareRoots(parameters+6);
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(std::size(parameters)); desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen probe visibility root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenVisibilityRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen probe visibility root",status);
    }
    if (!LumenVisibilityPipeline)
    {
        if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_GATHER_VISIBILITY_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
        if (!CompileShader("CSProbeVisibility","cs_5_0",binary,macros,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenVisibilityRoot.Get(); desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(LumenVisibilityPipeline.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen probe visibility pipeline",status);
    }
    if (!LumenGatherRoot)
    {
        D3D12_DESCRIPTOR_RANGE ranges[5]{}; D3D12_ROOT_PARAMETER parameters[9]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        for (UINT slot=0;slot<5;++slot)
        {
            ranges[slot]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,slot,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[slot+1].DescriptorTable={1,&ranges[slot]};
        }
        for (UINT slot=6;slot<9;++slot) { parameters[slot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[slot].Descriptor={slot-1,0}; }
        for (auto& parameter:parameters) parameter.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(std::size(parameters)); desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen gather root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenGatherRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen gather root",status);
    }
    if (LumenGatherPipeline) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_PROBE_GATHER_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSGather","vs_5_0",vertex,macros,true) || !CompileShader("PSGather","ps_5_0",pixel,macros,true)) return false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenGatherRoot.Get();
    desc.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; desc.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
    auto& blend=desc.BlendState.RenderTarget[0]; blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
    blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask=UINT_MAX; desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    desc.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK; desc.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
    auto& stencil=desc.DepthStencilState.FrontFace; stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; desc.DepthStencilState.BackFace=stencil;
    desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.NumRenderTargets=1; desc.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count=1;
    const auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(LumenGatherPipeline.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen gather pipeline",status);
}
bool ResolveLumenProbes(LumenProbeHandle handle,const LumenProbeGatherSettings& settings)
{
    if ((GIActive && (!Techniques::UsesLumen(FrameGI) || GI.Resolved)) || !HdrSceneActive || !SceneColorWritten || !GBufferWritten ||
        (FrameSampleCount>1 && (!SamplesOpaqueResolved || SamplesColorRestored || SamplesTransparentWritten)) ||
        (DeferredActive && !DeferredResolved) || (AmbientOcclusionActive && !AmbientOcclusionApplied) || ReflectionApplied ||
        FogResolved || LightShaftResolved || TemporalResolved || MotionBlurResolved || DepthOfFieldResolved || RefractionResolved || !Hair.Complete() || !SubsurfaceComplete())
        return Fail("Lumen gather requires resolved opaque HDR materials before reflections, volumes and transparency");
    if (LumenHasTransparentMainDraws(PendingFrame)) return Fail("Lumen gather must precede transparent draws");
    const auto found=LumenProbes.find(handle.Value); if (found==LumenProbes.end()) return Fail("Unknown Lumen gather probes");
    const auto& record=*found->second; if (!FrameLumen.CanGather(handle,record.SourceHandle,LastError)) return false;
    const auto source=LumenRadianceTraces.find(record.SourceHandle.Value); if (source==LumenRadianceTraces.end()) return Fail("Lumen gather source was released");
    LumenProbeGatherFrame frame; if (!BuildLumenProbeGatherFrame(record.Info,source->second->Lighting->Distance->Info,settings,FrameReflectionShading,frame,LastError) || !EnsureLumenGather(frame)) return false;
    const auto& distance=*source->second->Lighting->Distance;
    if (settings.Visibility.Enabled && distance.Hardware && !EnsureLumenHardwarePipeline(LumenHardwareProgram::Visibility,LumenVisibilityRoot.Get())) return false;
    frame.Model.Z=LumenGatherIndirect?1u:0u;
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(frame)>ConstantBufferCapacity-offset) return Fail("Lumen gather exhausted the frame constant arena");
    std::memcpy(ConstantData+offset,&frame,sizeof(frame)); ConstantOffset=offset+sizeof(frame);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    if (settings.Visibility.Enabled)
    {
        constexpr auto computeRead=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        Transition(MainDepth.Get(),MainDepthState,computeRead); Transition(GBufferSurfaces[1].Resource.Get(),GBufferSurfaces[1].State,computeRead);
        Transition(LumenVisibilityBuffer.Get(),LumenVisibilityState,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps);
        CommandList->SetComputeRootSignature(LumenVisibilityRoot.Get()); CommandList->SetPipelineState((distance.Hardware?LumenHardwarePipelines[unsigned(LumenHardwareProgram::Visibility)]:LumenVisibilityPipeline).Get());
        CommandList->SetComputeRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
        CommandList->SetComputeRootShaderResourceView(1,source->second->Lighting->Distance->Scene->GetGPUVirtualAddress());
        CommandList->SetComputeRootShaderResourceView(2,source->second->Origins->GetGPUVirtualAddress());
        CommandList->SetComputeRootDescriptorTable(3,SRVArena.Gpu(MainDepthSRV)); CommandList->SetComputeRootDescriptorTable(4,SRVArena.Gpu(GBufferSurfaces[1].SRV));
        CommandList->SetComputeRootUnorderedAccessView(5,LumenVisibilityBuffer->GetGPUVirtualAddress());
        BindLumenHardware(CommandList.Get(),distance,6);
        const auto groups=(frame.Trace.Counts.X+LumenDistanceTraceThreads-1)/LumenDistanceTraceThreads;
        CommandList->Dispatch(std::min(groups,LumenDistanceTraceColumns),(groups+LumenDistanceTraceColumns-1)/LumenDistanceTraceColumns,1);
    }
    Transition(LumenVisibilityBuffer.Get(),LumenVisibilityState,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    auto& scene=HdrSurfaces[HdrScene]; constexpr auto sampled=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    Transition(scene.Resource.Get(),scene.State,sampled); Transition(MainDepth.Get(),MainDepthState,sampled);
    for (auto index:{0,1,3}) Transition(GBufferSurfaces[index].Resource.Get(),GBufferSurfaces[index].State,sampled);
    // These immutable buffers end their producer passes in non-pixel read state.
    // Restore it after the fragment consumer for later integration/readback work.
    D3D12_RESOURCE_BARRIER barriers[2]{}; ID3D12Resource* buffers[]{source->second->Origins.Get(),record.Buffer.Get()};
    for (UINT i=0;i<2;++i) { barriers[i].Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; barriers[i].Transition={buffers[i],D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,sampled}; }
    CommandList->ResourceBarrier(2,barriers);
    Transition(LumenGatherTarget.Resource.Get(),LumenGatherTarget.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps);
    CommandList->SetGraphicsRootSignature(LumenGatherRoot.Get()); CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    const UINT views[]{scene.SRV,MainDepthSRV,GBufferSurfaces[1].SRV,GBufferSurfaces[0].SRV,GBufferSurfaces[3].SRV};
    for (UINT slot=0;slot<std::size(views);++slot) CommandList->SetGraphicsRootDescriptorTable(slot+1,SRVArena.Gpu(views[slot]));
    CommandList->SetGraphicsRootShaderResourceView(6,buffers[0]->GetGPUVirtualAddress()); CommandList->SetGraphicsRootShaderResourceView(7,buffers[1]->GetGPUVirtualAddress());
    CommandList->SetGraphicsRootShaderResourceView(8,LumenVisibilityBuffer->GetGPUVirtualAddress());
    const auto target=RTVArena.Cpu(LumenGatherTarget.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const D3D12_RECT scissor{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    CommandList->SetPipelineState(LumenGatherPipeline.Get()); CommandList->DrawInstanced(3,1,0,0); CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    for (auto& barrier:barriers) std::swap(barrier.Transition.StateBefore,barrier.Transition.StateAfter); CommandList->ResourceBarrier(2,barriers);
    if (!LumenGatherIndirect)
    {
        Transition(LumenGatherTarget.Resource.Get(),LumenGatherTarget.State,D3D12_RESOURCE_STATE_COPY_SOURCE); Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST);
        CommandList->CopyResource(scene.Resource.Get(),LumenGatherTarget.Resource.Get());
    }
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
