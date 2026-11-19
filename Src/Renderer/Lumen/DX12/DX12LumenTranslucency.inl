LumenTranslucencyFrame LumenVolumeFrame;
LumenRadianceHandle LumenVolumeRays;
LumenProbeHandle LumenVolumeProbes;
void ReleaseLumenTranslucencyResults()
{
    if (LumenVolumeProbes) LumenProbes.erase(LumenVolumeProbes.Value);
    if (LumenVolumeRays) LumenRadianceTraces.erase(LumenVolumeRays.Value);
    LumenVolumeProbes={}; LumenVolumeRays={}; LumenVolumeFrame.Grid.W=0;
}
HdrSurface LumenVolumeAtlas;
ComPtr<ID3D12RootSignature> LumenVolumeRoot;
ComPtr<ID3D12PipelineState> LumenVolumePipeline;
ComPtr<ID3D12PipelineState> LumenVolumeSupportPipeline;
ComPtr<ID3D12Resource> LumenVolumeSupport;
void DestroyLumenTranslucency()
{
    ReleaseLumenTranslucencyResults(); LumenVolumeAtlas={}; LumenVolumeRoot.Reset(); LumenVolumePipeline.Reset();
    LumenVolumeSupportPipeline.Reset(); LumenVolumeSupport.Reset();
}
bool EnsureLumenTranslucency()
{
    if (!CreateHdrSurface(LumenVolumeAtlas,int(LumenVolumeFrame.Atlas.X),int(LumenVolumeFrame.Atlas.Y))) return false;
    if (!LumenVolumeRoot)
    {
        D3D12_ROOT_PARAMETER parameters[3]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[2].Descriptor.ShaderRegister=1;
        for (auto& parameter:parameters) parameter.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(std::size(parameters)); desc.pParameters=parameters;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize Lumen translucency root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenVolumeRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen translucency root",status);
    }
    if (LumenVolumePipeline) return true;
    if (!LoadTechniqueShaderSource("DX12","LumenTranslucency.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSPack","vs_5_0",vertex,nullptr,true) || !CompileShader("PSPack","ps_5_0",pixel,nullptr,true)) return false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenVolumeRoot.Get();
    desc.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; desc.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
    auto& blend=desc.BlendState.RenderTarget[0]; blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
    blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask=UINT_MAX; desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    desc.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK; desc.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
    auto& stencil=desc.DepthStencilState.FrontFace; stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; desc.DepthStencilState.BackFace=stencil;
    desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.NumRenderTargets=1; desc.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count=1;
    const auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(LumenVolumePipeline.GetAddressOf()));
    return SUCCEEDED(status) || FailHRESULT("Create Lumen translucency pipeline",status);
}
bool PrepareLumenVolumeSupport(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    const auto source=LumenRadianceTraces.find(probes.SourceHandle.Value);
    if (source==LumenRadianceTraces.end() || !source->second->TraceDistance) return Fail("Lumen volume support lost its immutable distance scene");
    LumenDistanceTraceFrame frame;
    if (!BuildLumenTranslucencySupportFrame(source->second->TraceDistance->Info,upload,frame,LastError) || !EnsureLumenDistanceTrace()) return false;
    if (!LumenVolumeSupportPipeline)
    {
        if (!LoadTechniqueShaderSource("DX12","Lumen.hlsl",ShaderCode,LastError)) return false;
        const D3D_SHADER_MACRO options[]{{"EVANS_LUMEN_VOLUME_VISIBILITY_STAGE","1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> code;
        if (!CompileShader("CSVolumeSupport","cs_5_0",code,options,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenDistanceRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(LumenVolumeSupportPipeline.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create Lumen volume visibility pipeline",status);
    }
    LumenVolumeSupport.Reset(); if (!CreateLumenRadianceOutput(std::size_t(frame.Counts.X)*sizeof(float),LumenVolumeSupport)) return false;
    CommandList->SetComputeRootSignature(LumenDistanceRoot.Get()); CommandList->SetPipelineState(LumenVolumeSupportPipeline.Get());
    CommandList->SetComputeRoot32BitConstants(0,sizeof(frame)/sizeof(unsigned),&frame,0);
    CommandList->SetComputeRootShaderResourceView(1,source->second->TraceDistance->Scene->GetGPUVirtualAddress());
    CommandList->SetComputeRootShaderResourceView(2,source->second->Origins->GetGPUVirtualAddress());
    CommandList->SetComputeRootUnorderedAccessView(3,LumenVolumeSupport->GetGPUVirtualAddress());
    CommandList->Dispatch((frame.Counts.X+LumenRadianceThreads-1)/LumenRadianceThreads,1,1);
    auto state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS; Transition(LumenVolumeSupport.Get(),state,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); return true;
}
bool PackLumenTranslucency(const LumenProbeRecord& probes,const LumenRadianceTraceUpload& upload)
{
    if (!PrepareLumenVolumeSupport(probes,upload) || !EnsureLumenTranslucency()) return false;
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(LumenVolumeFrame)>ConstantBufferCapacity-offset) return Fail("Lumen translucency exhausted the frame constant arena");
    std::memcpy(ConstantData+offset,&LumenVolumeFrame,sizeof(LumenVolumeFrame)); ConstantOffset=offset+sizeof(LumenVolumeFrame);
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={probes.Buffer.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
    CommandList->ResourceBarrier(1,&barrier);
    Transition(LumenVolumeAtlas.Resource.Get(),LumenVolumeAtlas.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    const auto target=RTVArena.Cpu(LumenVolumeAtlas.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
    CommandList->SetGraphicsRootSignature(LumenVolumeRoot.Get()); CommandList->SetPipelineState(LumenVolumePipeline.Get());
    CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    CommandList->SetGraphicsRootShaderResourceView(1,probes.Buffer->GetGPUVirtualAddress());
    CommandList->SetGraphicsRootShaderResourceView(2,LumenVolumeSupport->GetGPUVirtualAddress());
    const D3D12_VIEWPORT viewport{0,0,LumenVolumeFrame.Atlas.X,LumenVolumeFrame.Atlas.Y,0,1};
    const D3D12_RECT scissor{0,0,LONG(LumenVolumeFrame.Atlas.X),LONG(LumenVolumeFrame.Atlas.Y)};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor);
    CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); CommandList->DrawInstanced(3,1,0,0);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(LumenVolumeAtlas.Resource.Get(),LumenVolumeAtlas.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    std::swap(barrier.Transition.StateBefore,barrier.Transition.StateAfter); CommandList->ResourceBarrier(1,&barrier); return true;
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

bool BindLumenTranslucency(UINT uniformRoot=ForwardLumenVolumeRoot,UINT imageRoot=ForwardLumenVolumeImageRoot,bool refractive=false)
{
    const bool active=FrameGI.Lumen.Translucency.Enabled && LumenVolumeFrame.Grid.W!=0 && (refractive || (CurrentPass.Type==PassType::Color && !CurrentPass.Target && BoundPipeline.BlendEnabled));
    auto frame=LumenVolumeFrame; if (!active) frame.Grid.W=0;
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(frame)>ConstantBufferCapacity-offset) return Fail("Lumen receiver exhausted frame constants");
    std::memcpy(ConstantData+offset,&frame,sizeof(frame)); ConstantOffset=offset+sizeof(frame);
    CommandList->SetGraphicsRootConstantBufferView(uniformRoot,ConstantBuffer->GetGPUVirtualAddress()+offset);
    CommandList->SetGraphicsRootDescriptorTable(imageRoot,SRVArena.Gpu(active?LumenVolumeAtlas.SRV:NullColorSRV)); return true;
}
