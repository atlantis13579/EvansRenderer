// Per-pass CBV slices and persistent SRV slots avoid rewriting descriptors
// referenced by recorded draws. Shared visibility storage remains immutable.
std::array<HdrSurface,2> HybridRenderingImages;
ComPtr<ID3D12RootSignature> HybridRenderingRoot;
std::array<ComPtr<ID3D12PipelineState>,2> HybridRenderingPipelines;
static constexpr UINT HybridRenderingTextureCount=7;
static constexpr UINT HybridRenderingInputCount=HybridRenderingTextureCount+RaySceneBufferCount+RayRadianceBufferCount;

void DestroyHybridRendering()
{
    for (auto& image:HybridRenderingImages) { SRVArena.Release(image.SRV); RTVArena.Release(image.RTV); image={}; }
    HybridRenderingPipelines={}; HybridRenderingRoot.Reset();
}

bool EnsureHybridRenderingResources()
{
    if (!EnsureRayRadiance()) return false;
    for (auto& image:HybridRenderingImages) if (!CreateHdrSurface(image,CreateInfo.Width,CreateInfo.Height)) return false;
    if (!HybridRenderingRoot)
    {
        std::array<D3D12_DESCRIPTOR_RANGE,HybridRenderingInputCount> ranges{};
        std::array<D3D12_ROOT_PARAMETER,HybridRenderingInputCount+1> parameters{};
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
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(hybrid reflections)",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(HybridRenderingRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(hybrid reflections)",result);
    }
    if (HybridRenderingPipelines[0] && HybridRenderingPipelines[1]) return true;
    if (!LoadTechniqueShaderSource("DX12","HybridRendering.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,nullptr,true)) return false;
    for (std::size_t phase=0;phase<HybridRenderingPipelines.size();++phase)
    {
        if (HybridRenderingPipelines[phase]) continue;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=HybridRenderingRoot.Get();
        pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
        auto& blend=pipeline.BlendState.RenderTarget[0]; blend.BlendEnable=phase!=0;
        blend.SrcBlend=D3D12_BLEND_ONE; blend.DestBlend=phase ? D3D12_BLEND_ONE : D3D12_BLEND_ZERO;
        blend.SrcBlendAlpha=D3D12_BLEND_ZERO; blend.DestBlendAlpha=D3D12_BLEND_ONE;
        blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP;
        blend.RenderTargetWriteMask=phase ? D3D12_COLOR_WRITE_ENABLE_RED|D3D12_COLOR_WRITE_ENABLE_GREEN|D3D12_COLOR_WRITE_ENABLE_BLUE : D3D12_COLOR_WRITE_ENABLE_ALL;
        pipeline.SampleMask=UINT_MAX; pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;
        pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; pipeline.RasterizerState.DepthClipEnable=TRUE;
        pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        pipeline.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK;
        pipeline.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
        pipeline.DepthStencilState.FrontFace.StencilFailOp=pipeline.DepthStencilState.FrontFace.StencilDepthFailOp=
            pipeline.DepthStencilState.FrontFace.StencilPassOp=D3D12_STENCIL_OP_KEEP;
        pipeline.DepthStencilState.FrontFace.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        pipeline.DepthStencilState.BackFace=pipeline.DepthStencilState.FrontFace;
        pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pipeline.NumRenderTargets=1; pipeline.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; pipeline.SampleDesc.Count=1;
        const auto result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(HybridRenderingPipelines[phase].GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(hybrid reflections)",result);
    }
    return true;
}
bool ApplyHybridRendering(const ReflectionDesc& desc)
{
    const auto& frame=FrameRayRadiance; HybridRenderingConstants constants;
    if (!BuildHybridRenderingConstants(desc,FrameRayRadianceSettings,FrameRayRadianceSettings.Hybrid,frame,CreateInfo.Width,CreateInfo.Height,Techniques::HybridRenderingPass::Trace,constants))
        return Fail("DX12 hybrid reflections require finite camera, material and lighting inputs");
    const auto first=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    const auto stride=AlignUp(sizeof(constants),D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    const auto required=(HybridRenderingPasses.size()-1)*stride+sizeof(constants);
    if (first>ConstantBufferCapacity || required>ConstantBufferCapacity-first) return Fail("DX12 hybrid reflection constant arena exhausted");
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(MainDepth.Get(),MainDepthState,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    for (auto& image:GBufferSurfaces) Transition(image.Resource.Get(),image.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    CommandList->SetGraphicsRootSignature(HybridRenderingRoot.Get()); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VIEWPORT viewport {0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1};
    const D3D12_RECT rectangle {0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&rectangle);
    for (UINT slot=0;slot<RaySceneBufferCount;++slot)
        CommandList->SetGraphicsRootDescriptorTable(1+HybridRenderingTextureCount+slot,SRVArena.Gpu(RaySceneViews[slot]));
    for (UINT slot=0;slot<RayRadianceBufferCount;++slot)
        CommandList->SetGraphicsRootDescriptorTable(1+HybridRenderingTextureCount+RaySceneBufferCount+slot,SRVArena.Gpu(RayRadianceViews[slot]));
    for (const auto pass:HybridRenderingPasses)
    {
        const auto index=static_cast<UINT>(pass); const bool composite=pass==Techniques::HybridRenderingPass::Composite;
        auto& destination=composite ? HdrSurfaces[HdrScene] : HybridRenderingImages[index%2];
        CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
        if (index) Transition(HybridRenderingImages[(index-1)%2].Resource.Get(),HybridRenderingImages[(index-1)%2].State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        if (!composite) Transition(HdrSurfaces[HdrScene].Resource.Get(),HdrSurfaces[HdrScene].State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        Transition(destination.Resource.Get(),destination.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
        const std::array<UINT,HybridRenderingTextureCount> textures {MainDepthSRV,GBufferSurfaces[1].SRV,GBufferSurfaces[2].SRV,
            GBufferSurfaces[0].SRV,GBufferSurfaces[3].SRV,index ? HybridRenderingImages[(index-1)%2].SRV : NullColorSRV,
            composite ? NullColorSRV : HdrSurfaces[HdrScene].SRV};
        for (UINT slot=0;slot<textures.size();++slot) CommandList->SetGraphicsRootDescriptorTable(slot+1,SRVArena.Gpu(textures[slot]));
        const auto offset=first+index*stride; constants.Options.X=index;
        std::memcpy(ConstantData+offset,&constants,sizeof(constants)); ConstantOffset=offset+sizeof(constants);
        CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
        const auto target=RTVArena.Cpu(destination.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
        CommandList->SetPipelineState(HybridRenderingPipelines[composite ? 1 : 0].Get()); CommandList->DrawInstanced(3,1,0,0);
        PendingFrame.ReflectionPasses.push_back(HybridRenderingLogPass(pass)); ++PendingFrame.Statistics.ReflectionPassCount; ++PendingFrame.Statistics.HybridRenderingPassCount;
    }
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); ReflectionApplied=true; return true;
}
