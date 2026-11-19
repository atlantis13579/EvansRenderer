LumenRadianceHandle LumenReflectionRays;
HdrSurface LumenReflectionCompositeTarget;
ComPtr<ID3D12Resource> LumenReflectionEstimate;
D3D12_RESOURCE_STATES LumenReflectionEstimateState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS,LumenReflectionEstimateFrameState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
std::size_t LumenReflectionEstimateBytes=0;
ComPtr<ID3D12RootSignature> LumenReflectionResolveRoot;
ComPtr<ID3D12PipelineState> LumenReflectionResolvePipeline;
LumenTemporalHistory LumenSpecularHistory;
std::array<ComPtr<ID3D12Resource>,2> LumenSpecularBuffers;
std::array<D3D12_RESOURCE_STATES,2> LumenSpecularStates{D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
std::array<D3D12_RESOURCE_STATES,2> LumenSpecularFrameStates=LumenSpecularStates;
ComPtr<ID3D12RootSignature> LumenSpecularRoot;
ComPtr<ID3D12PipelineState> LumenSpecularCompute,LumenSpecularComposite;
std::size_t LumenSpecularBytes=0;
void DestroyLumenSpecular()
{
    LumenSpecularBuffers={}; LumenSpecularBytes=0; LumenSpecularRoot.Reset(); LumenSpecularCompute.Reset(); LumenSpecularComposite.Reset(); LumenSpecularHistory.Invalidate();
}
bool EnsureLumenSpecular(std::size_t bytes)
{
    if (!CreateHdrSurface(LumenReflectionCompositeTarget,CreateInfo.Width,CreateInfo.Height)) return false;
    if (LumenSpecularBytes!=bytes)
    {
        std::array<ComPtr<ID3D12Resource>,2> buffers;
        for (auto& buffer:buffers) if (!CreateLumenRadianceOutput(bytes,buffer)) return false;
        LumenSpecularBuffers=buffers; LumenSpecularBytes=bytes; LumenSpecularStates.fill(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        LumenSpecularFrameStates=LumenSpecularStates; LumenSpecularHistory.Invalidate();
    }
    if (!LumenSpecularRoot)
    {
        D3D12_ROOT_PARAMETER parameters[8]{}; D3D12_DESCRIPTOR_RANGE ranges[4]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        for (UINT slot=0;slot<std::size(ranges);++slot)
        { ranges[slot]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,slot,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
          parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[slot+1].DescriptorTable={1,&ranges[slot]}; }
        parameters[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[5].Descriptor={4,0}; parameters[6].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[7].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[7].Descriptor={5,0};
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(std::size(parameters)); desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen specular temporal root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenSpecularRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen specular temporal root",status);
    }
    if (LumenSpecularCompute && LumenSpecularComposite) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO compute[]{{"EVANS_LUMEN_TEMPORAL_STAGE","1"},{"EVANS_LUMEN_REFLECTION_TEMPORAL_STAGE","1"},{nullptr,nullptr}},composite[]{{"EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE","1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> cs,vs,ps;
    if (!CompileShader("CSTemporal","cs_5_0",cs,compute,true) || !CompileShader("VSTemporal","vs_5_0",vs,composite,true) || !CompileShader("PSTemporalComposite","ps_5_0",ps,composite,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc{}; computeDesc.pRootSignature=LumenSpecularRoot.Get(); computeDesc.CS={cs->GetBufferPointer(),cs->GetBufferSize()};
    auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&computeDesc,IID_PPV_ARGS(LumenSpecularCompute.ReleaseAndGetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen specular temporal compute pipeline",status);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenSpecularRoot.Get(); desc.VS={vs->GetBufferPointer(),vs->GetBufferSize()}; desc.PS={ps->GetBufferPointer(),ps->GetBufferSize()};
    auto& blend=desc.BlendState.RenderTarget[0]; blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
    blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask=UINT_MAX; desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    desc.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK; desc.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
    auto& stencil=desc.DepthStencilState.FrontFace; stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP; stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    desc.DepthStencilState.BackFace=stencil; desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.NumRenderTargets=1; desc.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count=1;
    status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(LumenSpecularComposite.ReleaseAndGetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen specular temporal composite pipeline",status);
}
bool ResolveLumenSpecular(const ReflectionDesc& camera)
{
    LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    key.Reflections=FrameReflections; key.Reflected=true; LumenTemporalFrame frame;
    if (!BuildLumenReflectionTemporalFrame(camera,FrameReflections,CreateInfo.Width,CreateInfo.Height,LumenSpecularHistory.CanReuse(key),frame,LastError) ||
        !EnsureLumenSpecular(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(frame)>ConstantBufferCapacity-offset) return Fail("Lumen specular temporal exhausted the frame constant arena");
    std::memcpy(ConstantData+offset,&frame,sizeof(frame)); ConstantOffset=offset+sizeof(frame);
    const auto read=LumenSpecularHistory.ReadIndex(),write=LumenSpecularHistory.WriteIndex(); CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    constexpr auto computeRead=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    Transition(HdrSurfaces[HdrScene].Resource.Get(),HdrSurfaces[HdrScene].State,computeRead);
    Transition(LumenReflectionEstimate.Get(),LumenReflectionEstimateState,computeRead); Transition(MainDepth.Get(),MainDepthState,computeRead);
    Transition(GBufferSurfaces[1].Resource.Get(),GBufferSurfaces[1].State,computeRead); if (MotionSurfaces[MotionVelocity].Resource) Transition(MotionSurfaces[MotionVelocity].Resource.Get(),MotionSurfaces[MotionVelocity].State,computeRead);
    Transition(LumenSpecularBuffers[read].Get(),LumenSpecularStates[read],computeRead); Transition(LumenSpecularBuffers[write].Get(),LumenSpecularStates[write],D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps); CommandList->SetComputeRootSignature(LumenSpecularRoot.Get());
    CommandList->SetComputeRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    const UINT inputs[]{HdrSurfaces[HdrScene].SRV,MainDepthSRV,GBufferSurfaces[1].SRV,MotionSurfaces[MotionVelocity].Resource?MotionSurfaces[MotionVelocity].SRV:GBufferSurfaces[1].SRV};
    for (UINT slot=0;slot<std::size(inputs);++slot) CommandList->SetComputeRootDescriptorTable(slot+1,SRVArena.Gpu(inputs[slot]));
    CommandList->SetComputeRootShaderResourceView(5,LumenSpecularBuffers[read]->GetGPUVirtualAddress()); CommandList->SetComputeRootUnorderedAccessView(6,LumenSpecularBuffers[write]->GetGPUVirtualAddress());
    CommandList->SetComputeRootShaderResourceView(7,LumenReflectionEstimate->GetGPUVirtualAddress());
    CommandList->SetPipelineState(LumenSpecularCompute.Get()); const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    CommandList->Dispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    Transition(LumenSpecularBuffers[write].Get(),LumenSpecularStates[write],D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    auto& scene=HdrSurfaces[HdrScene]; Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(LumenReflectionCompositeTarget.Resource.Get(),LumenReflectionCompositeTarget.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    CommandList->SetGraphicsRootSignature(LumenSpecularRoot.Get()); CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    CommandList->SetGraphicsRootDescriptorTable(1,SRVArena.Gpu(scene.SRV)); CommandList->SetGraphicsRootShaderResourceView(5,LumenSpecularBuffers[write]->GetGPUVirtualAddress());
    const auto target=RTVArena.Cpu(LumenReflectionCompositeTarget.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const D3D12_RECT scissor{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    CommandList->SetPipelineState(LumenSpecularComposite.Get()); CommandList->DrawInstanced(3,1,0,0); CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(LumenReflectionCompositeTarget.Resource.Get(),LumenReflectionCompositeTarget.State,D3D12_RESOURCE_STATE_COPY_SOURCE); Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST);
    CommandList->CopyResource(scene.Resource.Get(),LumenReflectionCompositeTarget.Resource.Get());
    LumenSpecularHistory.Record(key); PendingFrame.Temporal=camera.Sampling; PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::LumenTemporal);
    PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::Composite); PendingFrame.Statistics.ReflectionPassCount+=2; return true;
}

void FinishLumenReflections(bool publish)
{
    LumenSpecularHistory.Finish(publish);
    if (LumenReflectionRays) LumenRadianceTraces.erase(LumenReflectionRays.Value); LumenReflectionRays={};
}
void DestroyLumenReflections()
{
    FinishLumenReflections(false); DestroyLumenSpecular(); LumenReflectionCompositeTarget={};
    LumenReflectionEstimate.Reset(); LumenReflectionEstimateBytes=0; LumenReflectionResolveRoot.Reset(); LumenReflectionResolvePipeline.Reset();
}
bool EnsureLumenReflectionResolve(std::size_t bytes)
{
    if (LumenReflectionEstimateBytes!=bytes)
    {
        ComPtr<ID3D12Resource> buffer; if (!CreateLumenRadianceOutput(bytes,buffer)) return false;
        LumenReflectionEstimate=buffer; LumenReflectionEstimateBytes=bytes;
        LumenReflectionEstimateState=LumenReflectionEstimateFrameState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    if (!LumenReflectionResolveRoot)
    {
        D3D12_ROOT_PARAMETER parameters[9]{}; D3D12_DESCRIPTOR_RANGE ranges[5]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        for (UINT slot=0;slot<std::size(ranges);++slot)
        {
            ranges[slot]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,slot,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[slot+1].DescriptorTable={1,&ranges[slot]};
        }
        parameters[6].ParameterType=parameters[7].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[6].Descriptor={5,0}; parameters[7].Descriptor={6,0}; parameters[8].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(std::size(parameters)); desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen reflection reconstruction root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenReflectionResolveRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen reflection reconstruction root",status);
    }
    if (LumenReflectionResolvePipeline) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO macros[]{{"EVANS_LUMEN_REFLECTION_RESOLVE_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> binary;
    if (!CompileShader("CSReflectionResolve","cs_5_0",binary,macros,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenReflectionResolveRoot.Get(); desc.CS={binary->GetBufferPointer(),binary->GetBufferSize()};
    const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(LumenReflectionResolvePipeline.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen reflection reconstruction pipeline",status);
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
    const auto& filter=FrameReflections.Lumen; const auto& sky=FrameGI.Lumen.SkyRadiance; LumenReflectionResolveFrame frame;
    if (!BuildLumenReflectionResolveFrame(rays.Info.Reflection,{filter.FilterRadius,filter.NormalCosine,filter.PlaneTolerance,filter.RelativePlaneTolerance,filter.RoughnessTolerance},
        {sky[0],sky[1],sky[2]},FrameReflectionShading,frame,LastError,MakeLumenEnvironmentLobe(FrameGI.Lumen)) || !EnsureLumenReflectionResolve(std::size_t(frame.Dispatch.Z)*sizeof(Float4))) return false;
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(frame)>ConstantBufferCapacity-offset) return Fail("Lumen reflection reconstruction exhausted the constant arena");
    std::memcpy(ConstantData+offset,&frame,sizeof(frame)); ConstantOffset=offset+sizeof(frame);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); constexpr auto sampled=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    Transition(MainDepth.Get(),MainDepthState,sampled); for (auto& surface:GBufferSurfaces) Transition(surface.Resource.Get(),surface.State,sampled);
    Transition(LumenReflectionEstimate.Get(),LumenReflectionEstimateState,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps);
    CommandList->SetComputeRootSignature(LumenReflectionResolveRoot.Get()); CommandList->SetPipelineState(LumenReflectionResolvePipeline.Get());
    CommandList->SetComputeRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    const UINT views[]{MainDepthSRV,GBufferSurfaces[1].SRV,GBufferSurfaces[0].SRV,GBufferSurfaces[3].SRV,GBufferSurfaces[2].SRV};
    for (UINT slot=0;slot<std::size(views);++slot) CommandList->SetComputeRootDescriptorTable(slot+1,SRVArena.Gpu(views[slot]));
    CommandList->SetComputeRootShaderResourceView(6,source->second->Origins->GetGPUVirtualAddress());
    CommandList->SetComputeRootShaderResourceView(7,source->second->Buffer->GetGPUVirtualAddress());
    CommandList->SetComputeRootUnorderedAccessView(8,LumenReflectionEstimate->GetGPUVirtualAddress());
    const auto groups=(frame.Dispatch.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    CommandList->Dispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::Trace); PendingFrame.ReflectionPasses.push_back(Techniques::ReflectionPassKind::LumenReconstruction);
    PendingFrame.Statistics.ReflectionPassCount+=2;
    if (!ResolveLumenSpecular(camera)) return false; ReflectionApplied=true; return true;
}
