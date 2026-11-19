bool SupportsTemporalExposure() const
{
    if (!Device) return false;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{DXGI_FORMAT_R32G32B32A32_FLOAT};
    constexpr UINT required=D3D12_FORMAT_SUPPORT1_RENDER_TARGET|D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
    return SUCCEEDED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT,&support,sizeof(support))) && (support.Support1&required)==required;
}
TemporalExposureHistory ExposureHistory;
std::vector<HdrSurface> ExposureLevels;
std::array<HdrSurface,2> ExposureValues;
ComPtr<ID3D12RootSignature> ExposureRoot;
std::array<ComPtr<ID3D12PipelineState>,2> ExposurePipelines;
int ExposureWidth=0,ExposureHeight=0;
void DestroyExposureTargets()
{
    const auto release=[&](HdrSurface& image){SRVArena.Release(image.SRV); RTVArena.Release(image.RTV); image={};};
    for (auto& image:ExposureLevels) release(image);
    for (auto& image:ExposureValues) release(image);
    ExposureLevels.clear(); ExposureWidth=ExposureHeight=0; ExposureHistory.Invalidate();
}
void DestroyExposure()
{ DestroyExposureTargets(); ExposureRoot.Reset(); ExposurePipelines={}; }
void SnapshotExposureStates(bool restore)
{
    const auto update=[&](HdrSurface& image){if (restore) image.State=image.FrameStartState; else image.FrameStartState=image.State;};
    for (auto& image:ExposureLevels) update(image);
    for (auto& image:ExposureValues) update(image);
}
bool EnsureExposure()
{
    if (ExposureWidth!=CreateInfo.Width || ExposureHeight!=CreateInfo.Height) DestroyExposureTargets();
    const auto plan=BuildExposureReductionPlan(CreateInfo.Width,CreateInfo.Height); ExposureLevels.resize(plan.size());
    for (std::size_t index=0;index<plan.size();++index)
        if (!CreateHdrSurface(ExposureLevels[index],plan[index].Width,plan[index].Height,DXGI_FORMAT_R32G32B32A32_FLOAT)) return false;
    for (auto& value:ExposureValues) if (!CreateHdrSurface(value,1,1,DXGI_FORMAT_R32G32B32A32_FLOAT)) return false;
    ExposureWidth=CreateInfo.Width; ExposureHeight=CreateInfo.Height;
    if (!ExposureRoot)
    {
        constexpr UINT InputCount=2;
        std::array<D3D12_DESCRIPTOR_RANGE,InputCount> ranges{};
        std::array<D3D12_ROOT_PARAMETER,InputCount+1> parameters{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants={0,0,sizeof(TemporalEffectsConstants)/sizeof(std::uint32_t)};
        parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        for (UINT slot=0;slot<InputCount;++slot)
        {
            ranges[slot].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[slot].NumDescriptors=1; ranges[slot].BaseShaderRegister=slot;
            ranges[slot].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[slot+1].DescriptorTable={1,&ranges[slot]}; parameters[slot+1].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_ROOT_SIGNATURE_DESC signature{}; signature.NumParameters=static_cast<UINT>(parameters.size()); signature.pParameters=parameters.data();
        signature.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> serialized,errors;
        auto result=SerializeRootSignatureFunction(&signature,D3D_ROOT_SIGNATURE_VERSION_1,serialized.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("Serialize temporal exposure root",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(ExposureRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("Create temporal exposure root",result);
    }
    if (ExposurePipelines[0] && ExposurePipelines[1]) return true;
    if (!LoadTechniqueShaderSource("DX12","TemporalEffects.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel; if (!CompileShader("VSMain","vs_5_0",vertex) || !CompileShader("PSMain","ps_5_0",pixel)) return false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=ExposureRoot.Get();
    pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
    auto& blend=pipeline.BlendState.RenderTarget[0]; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
    blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP;
    pipeline.SampleMask=UINT_MAX; pipeline.SampleDesc.Count=1; pipeline.NumRenderTargets=1;
    pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; pipeline.RasterizerState.DepthClipEnable=TRUE;
    pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    pipeline.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK; pipeline.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
    pipeline.DepthStencilState.FrontFace.StencilFailOp=pipeline.DepthStencilState.FrontFace.StencilDepthFailOp=pipeline.DepthStencilState.FrontFace.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    pipeline.DepthStencilState.FrontFace.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; pipeline.DepthStencilState.BackFace=pipeline.DepthStencilState.FrontFace;
    pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    for (unsigned index=0;index<ExposurePipelines.size();++index)
    {
        pipeline.RTVFormats[0]=index?DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_R32G32B32A32_FLOAT;
        const auto result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(ExposurePipelines[index].ReleaseAndGetAddressOf()));
        if (FAILED(result)) return FailHRESULT("Create temporal exposure pipeline",result);
    }
    return true;
}
bool ResolveExposure(const TemporalEffectsDesc& desc)
{
    TemporalEffectsConstants constants;
    if (!ExposureHistory.Prepare(desc,FramePostProcessing.TemporalEffects,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("Exposure requires finite scene time and positive dimensions");
    CommandList->SetGraphicsRootSignature(ExposureRoot.Get()); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const auto draw=[&](HdrSurface& source,HdrSurface& previous,HdrSurface& destination,Techniques::TemporalEffectsPassKind phase)
    {
        CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
        Transition(source.Resource.Get(),source.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        Transition(previous.Resource.Get(),previous.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        Transition(destination.Resource.Get(),destination.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
        constants.PhaseExtentHistory.X=static_cast<unsigned>(phase); constants.PhaseExtentHistory.Y=source.Width; constants.PhaseExtentHistory.Z=source.Height;
        CommandList->SetGraphicsRoot32BitConstants(0,sizeof(constants)/sizeof(std::uint32_t),&constants,0);
        CommandList->SetGraphicsRootDescriptorTable(1,SRVArena.Gpu(source.SRV)); CommandList->SetGraphicsRootDescriptorTable(2,SRVArena.Gpu(previous.SRV));
        const auto target=RTVArena.Cpu(destination.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
        const D3D12_VIEWPORT viewport{0,0,float(destination.Width),float(destination.Height),0,1}; const D3D12_RECT scissor{0,0,destination.Width,destination.Height};
        CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor);
        CommandList->SetPipelineState(ExposurePipelines[phase==Techniques::TemporalEffectsPassKind::Composite?1:0].Get());
        CommandList->DrawInstanced(3,1,0,0); ++PendingFrame.Statistics.TemporalEffectsPassCount;
    };
    auto* source=&HdrSurfaces[HdrScene];
    for (std::size_t index=0;index<ExposureLevels.size();++index)
    { draw(*source,*source,ExposureLevels[index],index?Techniques::TemporalEffectsPassKind::Reduce:Techniques::TemporalEffectsPassKind::Meter); source=&ExposureLevels[index]; }
    auto& current=ExposureValues[1-ExposureHistory.ReadIndex()];
    draw(*source,constants.PhaseExtentHistory.W?ExposureValues[ExposureHistory.ReadIndex()]:*source,current,Techniques::TemporalEffectsPassKind::Adapt);
    auto& scene=HdrSurfaces[HdrScene]; auto& scratch=HdrSurfaces[HdrScratch];
    draw(scene,current,scratch,Techniques::TemporalEffectsPassKind::Composite);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(scratch.Resource.Get(),scratch.State,D3D12_RESOURCE_STATE_COPY_SOURCE); Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST);
    CommandList->CopyResource(scene.Resource.Get(),scratch.Resource.Get()); ExposureHistory.Resolved=true; return true;
}
