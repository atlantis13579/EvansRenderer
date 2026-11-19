// Descriptor slots remain stable while commands are in flight. Image states
// are journaled by BeginFrame/AbortFrame independently of committed sample counts.
PathTracingFrameState PathFrame;
std::array<HdrSurface,2> PathMeans;
HdrSurface PathDepth;
ComPtr<ID3D12RootSignature> PathRoot;
std::array<ComPtr<ID3D12PipelineState>,2> PathPipelines;
static constexpr UINT PathImageCount=2,PathInputCount=PathImageCount+RaySceneBufferCount+RayRadianceBufferCount;

bool SupportsPathTracing() const
{
    if (!Device) return false;
    constexpr auto required=D3D12_FORMAT_SUPPORT1_RENDER_TARGET|D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
    for (auto format:{DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32_FLOAT})
    {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        if (FAILED(Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT,&support,sizeof(support))) ||
            (support.Support1&required)!=required) return false;
    }
    return true;
}
void DestroyPathTracing()
{
    for (auto& image:PathMeans) { SRVArena.Release(image.SRV); RTVArena.Release(image.RTV); image={}; }
    SRVArena.Release(PathDepth.SRV); RTVArena.Release(PathDepth.RTV); PathDepth={};
    PathPipelines={}; PathRoot.Reset(); PathFrame={};
}
bool EnsurePathTracing()
{
    if (!SupportsPathTracing()) return Fail("DX12 path tracing requires FP32 render/load targets");
    if (!EnsureRayRadiance()) return false;
    if (PathDepth.Width!=CreateInfo.Width || PathDepth.Height!=CreateInfo.Height) PathFrame.History.Invalidate();
    for (auto& image:PathMeans)
        if (!CreateHdrSurface(image,CreateInfo.Width,CreateInfo.Height,DXGI_FORMAT_R32G32B32A32_FLOAT)) return false;
    if (!CreateHdrSurface(PathDepth,CreateInfo.Width,CreateInfo.Height,DXGI_FORMAT_R32_FLOAT)) return false;
    if (!PathRoot)
    {
        std::array<D3D12_DESCRIPTOR_RANGE,PathInputCount> ranges{};
        std::array<D3D12_ROOT_PARAMETER,PathInputCount+1> parameters{};
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
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(path tracing)",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(PathRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(path tracing)",result);
    }
    if (PathPipelines[0] && PathPipelines[1]) return true;
    if (!LoadTechniqueShaderSource("DX12","PathTracing.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true)) return false;
    for (std::size_t phase=0;phase<PathPipelines.size();++phase)
    {
        if (PathPipelines[phase]) continue;
        ComPtr<ID3DBlob> pixel;
        if (!CompileShader(phase ? "PSCompose" : "PSMain","ps_5_0",pixel,nullptr,true)) return false;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=PathRoot.Get();
        pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
        auto& blend=pipeline.BlendState.RenderTarget[0]; blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE;
        blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO; blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD;
        blend.LogicOp=D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
        pipeline.SampleMask=UINT_MAX; pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;
        pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; pipeline.RasterizerState.DepthClipEnable=TRUE;
        pipeline.DepthStencilState.DepthEnable=phase!=0;
        pipeline.DepthStencilState.DepthWriteMask=phase ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        pipeline.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK;
        pipeline.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
        pipeline.DepthStencilState.FrontFace.StencilFailOp=pipeline.DepthStencilState.FrontFace.StencilDepthFailOp=
            pipeline.DepthStencilState.FrontFace.StencilPassOp=D3D12_STENCIL_OP_KEEP;
        pipeline.DepthStencilState.FrontFace.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        pipeline.DepthStencilState.BackFace=pipeline.DepthStencilState.FrontFace;
        pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pipeline.NumRenderTargets=phase ? 1 : 2;
        pipeline.RTVFormats[0]=phase ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
        if (!phase) pipeline.RTVFormats[1]=DXGI_FORMAT_R32_FLOAT;
        pipeline.DSVFormat=phase ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN; pipeline.SampleDesc.Count=1;
        const auto result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(PathPipelines[phase].GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(path tracing)",result);
    }
    return true;
}
bool TracePaths(const PathTracingDesc& desc)
{
    PathTracingConstants constants;
    if (!PathFrame.Prepare(desc,FrameRayRadianceSettings,FrameRayRadiance,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("DX12 path frame requires matched scene inputs and a finite invertible camera");
    const auto plan=PathFrame.History.GetPlan();
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(constants)>ConstantBufferCapacity-offset) return Fail("DX12 path constant arena exhausted");
    std::memcpy(ConstantData+offset,&constants,sizeof(constants)); ConstantOffset=offset+sizeof(constants);
    CommandList->SetGraphicsRootSignature(PathRoot.Get()); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1};
    const D3D12_RECT scissor{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor);
    for (UINT slot=0;slot<RaySceneBufferCount;++slot)
        CommandList->SetGraphicsRootDescriptorTable(1+PathImageCount+slot,SRVArena.Gpu(RaySceneViews[slot]));
    for (UINT slot=0;slot<RayRadianceBufferCount;++slot)
        CommandList->SetGraphicsRootDescriptorTable(1+PathImageCount+RaySceneBufferCount+slot,SRVArena.Gpu(RayRadianceViews[slot]));
    for (unsigned phase=0;phase<2;++phase)
    {
        CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
        auto& mean=PathMeans[phase ? plan.WriteIndex : plan.ReadIndex];
        Transition(mean.Resource.Get(),mean.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        auto& destination=phase ? HdrSurfaces[HdrScene] : PathMeans[plan.WriteIndex];
        Transition(destination.Resource.Get(),destination.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
        Transition(PathDepth.Resource.Get(),PathDepth.State,phase ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_RENDER_TARGET);
        CommandList->SetGraphicsRootDescriptorTable(1,SRVArena.Gpu(phase || plan.PreviousSamples ? mean.SRV : NullColorSRV));
        CommandList->SetGraphicsRootDescriptorTable(2,SRVArena.Gpu(phase ? PathDepth.SRV : NullColorSRV));
        const D3D12_CPU_DESCRIPTOR_HANDLE targets[]{RTVArena.Cpu(destination.RTV),RTVArena.Cpu(PathDepth.RTV)};
        const auto depth=DSVArena.Cpu(MainDepthDSV);
        if (phase) Transition(MainDepth.Get(),MainDepthState,D3D12_RESOURCE_STATE_DEPTH_WRITE);
        CommandList->OMSetRenderTargets(phase ? 1 : 2,targets,FALSE,phase ? &depth : nullptr);
        CommandList->SetPipelineState(PathPipelines[phase].Get()); CommandList->DrawInstanced(3,1,0,0);
    }
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    SceneColorWritten=true; return PathFrame.Resolve(PendingFrame.Statistics);
}
