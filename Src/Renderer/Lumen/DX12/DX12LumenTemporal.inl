LumenTemporalHistory LumenHistory;
std::array<ComPtr<ID3D12Resource>,2> LumenTemporalBuffers;
std::array<D3D12_RESOURCE_STATES,2> LumenTemporalStates{D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
std::array<D3D12_RESOURCE_STATES,2> LumenTemporalFrameStates=LumenTemporalStates;
ComPtr<ID3D12RootSignature> LumenTemporalRoot;
ComPtr<ID3D12PipelineState> LumenTemporalCompute,LumenTemporalComposite;
std::size_t LumenTemporalBytes=0;
void DestroyLumenTemporal()
{
    LumenTemporalBuffers={}; LumenTemporalBytes=0; LumenTemporalRoot.Reset(); LumenTemporalCompute.Reset(); LumenTemporalComposite.Reset(); LumenHistory.Invalidate();
}
bool EnsureLumenTemporal(std::size_t bytes)
{
    if (LumenTemporalBytes!=bytes)
    {
        std::array<ComPtr<ID3D12Resource>,2> buffers;
        for (auto& buffer:buffers) if (!CreateLumenRadianceOutput(bytes,buffer)) return false;
        LumenTemporalBuffers=buffers; LumenTemporalBytes=bytes; LumenTemporalStates.fill(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        LumenTemporalFrameStates=LumenTemporalStates; LumenHistory.Invalidate();
    }
    if (!LumenTemporalRoot)
    {
        D3D12_ROOT_PARAMETER parameters[7]{}; D3D12_DESCRIPTOR_RANGE ranges[4]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        for (UINT slot=0;slot<std::size(ranges);++slot)
        { ranges[slot]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,slot,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
          parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[slot+1].DescriptorTable={1,&ranges[slot]}; }
        parameters[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[5].Descriptor={4,0}; parameters[6].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(std::size(parameters)); desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen temporal root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenTemporalRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen temporal root",status);
    }
    if (LumenTemporalCompute && LumenTemporalComposite) return true;
    if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
    const D3D_SHADER_MACRO compute[]{{"EVANS_LUMEN_TEMPORAL_STAGE","1"},{nullptr,nullptr}},composite[]{{"EVANS_LUMEN_TEMPORAL_COMPOSITE_STAGE","1"},{nullptr,nullptr}};
    ComPtr<ID3DBlob> cs,vs,ps;
    if (!CompileShader("CSTemporal","cs_5_0",cs,compute,true) || !CompileShader("VSTemporal","vs_5_0",vs,composite,true) || !CompileShader("PSTemporalComposite","ps_5_0",ps,composite,true)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc{}; computeDesc.pRootSignature=LumenTemporalRoot.Get(); computeDesc.CS={cs->GetBufferPointer(),cs->GetBufferSize()};
    auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&computeDesc,IID_PPV_ARGS(LumenTemporalCompute.ReleaseAndGetAddressOf()));
    if (FAILED(status)) return FailHRESULT("Create Lumen temporal compute pipeline",status);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenTemporalRoot.Get(); desc.VS={vs->GetBufferPointer(),vs->GetBufferSize()}; desc.PS={ps->GetBufferPointer(),ps->GetBufferSize()};
    auto& blend=desc.BlendState.RenderTarget[0]; blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
    blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask=UINT_MAX; desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    desc.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK; desc.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
    auto& stencil=desc.DepthStencilState.FrontFace; stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP; stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    desc.DepthStencilState.BackFace=stencil; desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.NumRenderTargets=1; desc.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count=1;
    status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(LumenTemporalComposite.ReleaseAndGetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen temporal composite pipeline",status);
}
bool ResolveLumenTemporal(const GlobalIlluminationDesc& camera)
{
    const LumenTemporalKey key{FrameLumenLighting,FrameGI,FrameReflectionShading,unsigned(CreateInfo.Width),unsigned(CreateInfo.Height)};
    LumenTemporalFrame frame;
    if (!BuildLumenTemporalFrame(camera,FrameGI,CreateInfo.Width,CreateInfo.Height,LumenHistory.CanReuse(key),frame,LastError) ||
        !EnsureLumenTemporal(std::size_t(frame.Extent.Z)*sizeof(LumenTemporalPixel))) return false;
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(frame)>ConstantBufferCapacity-offset) return Fail("Lumen temporal exhausted the frame constant arena");
    std::memcpy(ConstantData+offset,&frame,sizeof(frame)); ConstantOffset=offset+sizeof(frame);
    const auto read=LumenHistory.ReadIndex(),write=LumenHistory.WriteIndex(); CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    constexpr auto computeRead=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    Transition(LumenGatherTarget.Resource.Get(),LumenGatherTarget.State,computeRead); Transition(MainDepth.Get(),MainDepthState,computeRead);
    Transition(GBufferSurfaces[1].Resource.Get(),GBufferSurfaces[1].State,computeRead); Transition(MotionSurfaces[MotionVelocity].Resource.Get(),MotionSurfaces[MotionVelocity].State,computeRead);
    Transition(LumenTemporalBuffers[read].Get(),LumenTemporalStates[read],computeRead); Transition(LumenTemporalBuffers[write].Get(),LumenTemporalStates[write],D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps); CommandList->SetComputeRootSignature(LumenTemporalRoot.Get());
    CommandList->SetComputeRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    const UINT inputs[]{LumenGatherTarget.SRV,MainDepthSRV,GBufferSurfaces[1].SRV,MotionSurfaces[MotionVelocity].SRV};
    for (UINT slot=0;slot<std::size(inputs);++slot) CommandList->SetComputeRootDescriptorTable(slot+1,SRVArena.Gpu(inputs[slot]));
    CommandList->SetComputeRootShaderResourceView(5,LumenTemporalBuffers[read]->GetGPUVirtualAddress()); CommandList->SetComputeRootUnorderedAccessView(6,LumenTemporalBuffers[write]->GetGPUVirtualAddress());
    CommandList->SetPipelineState(LumenTemporalCompute.Get()); const auto groups=(frame.Extent.Z+LumenTemporalThreads-1)/LumenTemporalThreads;
    CommandList->Dispatch(std::min(groups,LumenTemporalColumns),(groups+LumenTemporalColumns-1)/LumenTemporalColumns,1);
    Transition(LumenTemporalBuffers[write].Get(),LumenTemporalStates[write],D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    auto& scene=HdrSurfaces[HdrScene]; Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(LumenGatherTarget.Resource.Get(),LumenGatherTarget.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    CommandList->SetGraphicsRootSignature(LumenTemporalRoot.Get()); CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    CommandList->SetGraphicsRootDescriptorTable(1,SRVArena.Gpu(scene.SRV)); CommandList->SetGraphicsRootShaderResourceView(5,LumenTemporalBuffers[write]->GetGPUVirtualAddress());
    const auto target=RTVArena.Cpu(LumenGatherTarget.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const D3D12_RECT scissor{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    CommandList->SetPipelineState(LumenTemporalComposite.Get()); CommandList->DrawInstanced(3,1,0,0); CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(LumenGatherTarget.Resource.Get(),LumenGatherTarget.State,D3D12_RESOURCE_STATE_COPY_SOURCE); Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST);
    CommandList->CopyResource(scene.Resource.Get(),LumenGatherTarget.Resource.Get());
    LumenHistory.Record(key); PendingFrame.Temporal=camera.Sampling; PendingFrame.Statistics.GlobalIlluminationPassCount+=2; return true;
}
