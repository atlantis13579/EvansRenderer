// One immutable SRV per image and a CBV slice per draw. Resource states join
// the ordinary frame journal, including frames cancelled after seed production.
enum SubsurfaceImage : std::size_t { SubsurfaceScene, SubsurfaceSource, SubsurfaceHorizontal, SubsurfaceVertical, SubsurfaceImageCount };
std::array<HdrSurface,SubsurfaceImageCount> SubsurfaceImages;
ComPtr<ID3D12RootSignature> SubsurfaceRoot;
ComPtr<ID3D12PipelineState> SubsurfacePipeline;
static constexpr UINT SubsurfaceInputCount=5;
bool SubsurfaceProducing=false,SubsurfaceResolved=false,SubsurfaceHasScene=false;
DeferredLightingDesc SubsurfaceLighting;
Techniques::SubsurfaceScatteringSettings FrameSubsurface;
bool SubsurfaceRequested=false;
bool SubsurfaceComplete() const { return !SubsurfaceRequested || SubsurfaceResolved; }

void DestroySubsurface()
{
    for (auto& image:SubsurfaceImages) { SRVArena.Release(image.SRV); RTVArena.Release(image.RTV); image={}; }
    SubsurfacePipeline.Reset(); SubsurfaceRoot.Reset();
}
bool EnsureSubsurface()
{
    for (auto& image:SubsurfaceImages) if (!CreateHdrSurface(image,CreateInfo.Width,CreateInfo.Height)) return false;
    if (!SubsurfaceRoot)
    {
        std::array<D3D12_DESCRIPTOR_RANGE,SubsurfaceInputCount> ranges{};
        std::array<D3D12_ROOT_PARAMETER,SubsurfaceInputCount+1> parameters{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[0].Descriptor.ShaderRegister=0;
        parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        for (UINT slot=0;slot<ranges.size();++slot)
        {
            ranges[slot].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[slot].NumDescriptors=1;
            ranges[slot].BaseShaderRegister=slot; ranges[slot].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            auto& parameter=parameters[slot+1]; parameter.ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameter.DescriptorTable.NumDescriptorRanges=1; parameter.DescriptorTable.pDescriptorRanges=&ranges[slot];
            parameter.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_ROOT_SIGNATURE_DESC signature{}; signature.NumParameters=static_cast<UINT>(parameters.size());
        signature.pParameters=parameters.data(); signature.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> serialized,errors;
        auto result=SerializeRootSignatureFunction(&signature,D3D_ROOT_SIGNATURE_VERSION_1,serialized.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(subsurface)",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(SubsurfaceRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(subsurface)",result);
    }
    if (SubsurfacePipeline) return true;
    if (!LoadTechniqueShaderSource("DX12","SubsurfaceScattering.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,nullptr,true)) return false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=SubsurfaceRoot.Get();
    pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
    auto& blend=pipeline.BlendState.RenderTarget[0]; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    pipeline.SampleMask=UINT_MAX; pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;
    pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; pipeline.RasterizerState.DepthClipEnable=TRUE;
    blend.SrcBlend=D3D12_BLEND_ONE; blend.DestBlend=D3D12_BLEND_ZERO; blend.BlendOp=D3D12_BLEND_OP_ADD;
    blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlendAlpha=D3D12_BLEND_ZERO; blend.BlendOpAlpha=D3D12_BLEND_OP_ADD;
    blend.LogicOp=D3D12_LOGIC_OP_NOOP;
    pipeline.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK;
    pipeline.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
    pipeline.DepthStencilState.FrontFace.StencilFailOp=pipeline.DepthStencilState.FrontFace.StencilDepthFailOp=
        pipeline.DepthStencilState.FrontFace.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    pipeline.DepthStencilState.FrontFace.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    pipeline.DepthStencilState.BackFace=pipeline.DepthStencilState.FrontFace;
    pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; pipeline.NumRenderTargets=1;
    pipeline.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; pipeline.SampleDesc.Count=1;
    return SUCCEEDED(Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(SubsurfacePipeline.GetAddressOf()))) || Fail("DX12 subsurface pipeline creation failed");
}
bool SnapshotSubsurfaceScene()
{
    if (!EnsureSubsurface()) return false;
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    auto& scene=HdrSurfaces[HdrScene]; auto& saved=SubsurfaceImages[SubsurfaceScene];
    Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(saved.Resource.Get(),saved.State,D3D12_RESOURCE_STATE_COPY_DEST);
    CommandList->CopyResource(saved.Resource.Get(),scene.Resource.Get()); return true;
}
bool CaptureSubsurfaceSource()
{
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    auto& source=FrameSampleCount>1 ? MultisampleSurfaces[MultisampleColor] : HdrSurfaces[HdrScene];
    auto& destination=SubsurfaceImages[SubsurfaceSource];
    if (FrameSampleCount>1)
    {
        Transition(source.Resource.Get(),source.State,D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        Transition(destination.Resource.Get(),destination.State,D3D12_RESOURCE_STATE_RESOLVE_DEST);
        CommandList->ResolveSubresource(destination.Resource.Get(),0,source.Resource.Get(),0,DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    else
    {
        Transition(source.Resource.Get(),source.State,D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(destination.Resource.Get(),destination.State,D3D12_RESOURCE_STATE_COPY_DEST);
        CommandList->CopyResource(destination.Resource.Get(),source.Resource.Get());
    }
    return true;
}
bool FilterSubsurface(SubsurfaceScatteringConstants constants)
{
    const auto stride=AlignUp(sizeof(constants),D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    const auto first=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    constexpr unsigned FilterDrawCount=3;
    if (first>ConstantBufferCapacity || FilterDrawCount*stride>ConstantBufferCapacity-first) return Fail("DX12 subsurface constant arena is full");
    Transition(MainDepth.Get(),MainDepthState,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(GBufferSurfaces[1].Resource.Get(),GBufferSurfaces[1].State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    for (auto& image:SubsurfaceImages) Transition(image.Resource.Get(),image.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    CommandList->SetGraphicsRootSignature(SubsurfaceRoot.Get()); CommandList->SetPipelineState(SubsurfacePipeline.Get());
    CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1};
    const D3D12_RECT rectangle{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&rectangle);
    unsigned draw=0;
    for (auto pass:{Techniques::SubsurfacePass::Horizontal,Techniques::SubsurfacePass::Vertical,Techniques::SubsurfacePass::Composite})
    {
        const bool horizontal=pass==Techniques::SubsurfacePass::Horizontal,composite=pass==Techniques::SubsurfacePass::Composite;
        auto& source=SubsurfaceImages[horizontal ? SubsurfaceSource : composite ? SubsurfaceVertical : SubsurfaceHorizontal];
        auto& destination=composite ? HdrSurfaces[HdrScene] : SubsurfaceImages[horizontal ? SubsurfaceHorizontal : SubsurfaceVertical];
        CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
        Transition(source.Resource.Get(),source.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        Transition(destination.Resource.Get(),destination.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
        const std::array<UINT,SubsurfaceInputCount> inputs{source.SRV,MainDepthSRV,GBufferSurfaces[1].SRV,SubsurfaceImages[SubsurfaceScene].SRV,SubsurfaceImages[SubsurfaceSource].SRV};
        for (UINT slot=0;slot<inputs.size();++slot) CommandList->SetGraphicsRootDescriptorTable(slot+1,SRVArena.Gpu(inputs[slot]));
        constants.Options.X=static_cast<unsigned>(pass); const auto offset=first+draw++*stride;
        std::memcpy(ConstantData+offset,&constants,sizeof(constants)); ConstantOffset=offset+sizeof(constants);
        CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
        const auto target=RTVArena.Cpu(destination.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
        CommandList->DrawInstanced(3,1,0,0); ++PendingFrame.Statistics.SubsurfacePassCount;
    }
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); SubsurfaceResolved=true; return true;
}
