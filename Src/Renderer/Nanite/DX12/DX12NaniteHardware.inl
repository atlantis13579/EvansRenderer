NaniteHardwareWork HardwarePlan;
std::shared_ptr<NaniteFrameBuffer> HardwareRanges,HardwareArguments,HardwareQueue,HardwareOwners;
ComPtr<ID3D12RootSignature> HardwareComputeRoot,HardwareGraphicsRoot;
ComPtr<ID3D12CommandSignature> HardwareDrawSignature;
std::array<ComPtr<ID3D12PipelineState>,4> HardwareKernels;
std::array<ComPtr<ID3DBlob>,2> HardwareVertices;
std::array<ComPtr<ID3DBlob>,3> HardwarePixels;
std::unordered_map<unsigned,ComPtr<ID3D12PipelineState>> HardwarePipelines;
void DestroyNaniteHardware()
{
    HardwarePlan={}; HardwareRanges.reset(); HardwareArguments.reset(); HardwareQueue.reset(); HardwareOwners.reset();
    HardwareComputeRoot.Reset(); HardwareGraphicsRoot.Reset(); HardwareDrawSignature.Reset(); HardwareKernels={}; HardwareVertices={}; HardwarePixels={}; HardwarePipelines.clear();
}
bool PrepareNaniteHardware(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteHardwareWork(frame,CreateInfo.Width,CreateInfo.Height,HardwarePlan,LastError)) return false;
    if (!HardwarePlan.Active()) return true;
    if (!CreateNaniteFrameBuffer(unsigned(HardwarePlan.Ranges.size()*sizeof(ShaderUInt4)/sizeof(unsigned)),HardwarePlan.Ranges.data(),false,HardwareRanges) ||
        !CreateNaniteFrameBuffer(HardwarePlan.ArgumentWords,nullptr,true,HardwareArguments) ||
        !CreateNaniteFrameBuffer(HardwarePlan.QueueWords,nullptr,true,HardwareQueue) ||
        !CreateNaniteFrameBuffer(HardwarePlan.Constants.Counts.W,nullptr,true,HardwareOwners)) return false;
    if (!HardwareComputeRoot)
    {
        D3D12_ROOT_PARAMETER p[6]{}; p[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[0].Constants.Num32BitValues=sizeof(NaniteHardwareConstants)/sizeof(unsigned);
        for (unsigned i=1;i<=2;++i) { p[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; p[i].Descriptor.ShaderRegister=i-1; }
        for (unsigned i=3;i<6;++i) { p[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV; p[i].Descriptor.ShaderRegister=i-3; }
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=6; desc.pParameters=p; ComPtr<ID3DBlob> code,errors;
        auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf());
        if (SUCCEEDED(status)) status=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(HardwareComputeRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create hardware schedule compute root",status);
    }
    if (!HardwareGraphicsRoot)
    {
        D3D12_ROOT_PARAMETER p[9]{}; p[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[0].Constants.Num32BitValues=sizeof(NaniteHardwareDrawConstants)/sizeof(unsigned);
        for (unsigned i=1;i<=4;++i) { p[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; p[i].Descriptor.ShaderRegister=i==4?5:i-1; }
        p[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        p[8].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; p[8].Descriptor.ShaderRegister=6;
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        for (unsigned i=0;i<2;++i)
        { ranges[i]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,i+3,0,0}; p[i+6].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[i+6].DescriptorTable={1,&ranges[i]}; p[i+6].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL; }
        D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR; sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS; sampler.MaxLOD=D3D12_FLOAT32_MAX; sampler.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=9; desc.pParameters=p; desc.NumStaticSamplers=1; desc.pStaticSamplers=&sampler;
        ComPtr<ID3DBlob> code,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),errors.GetAddressOf());
        if (SUCCEEDED(status)) status=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(HardwareGraphicsRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create hardware ownership graphics root",status);
    }
    if (!HardwareDrawSignature)
    {
        D3D12_INDIRECT_ARGUMENT_DESC argument{}; argument.Type=D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
        D3D12_COMMAND_SIGNATURE_DESC desc{}; desc.ByteStride=sizeof(D3D12_DRAW_ARGUMENTS); desc.NumArgumentDescs=1; desc.pArgumentDescs=&argument;
        const auto status=Device->CreateCommandSignature(&desc,nullptr,IID_PPV_ARGS(HardwareDrawSignature.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create hardware bin draw signature",status);
    }
    if (HardwareKernels.back() && HardwareVertices.back() && HardwarePixels.back()) return true;
    if (!LoadTechniqueShaderSource("DX12","NaniteHardware.hlsl",ShaderCode,LastError)) return false;
    constexpr const char* entries[]{"CSClear","CSCount","CSScan","CSScatter"};
    for (unsigned pass=0;pass<HardwareKernels.size();++pass) if (!HardwareKernels[pass])
    {
        ComPtr<ID3DBlob> code; if (!CompileShader(entries[pass],"cs_5_0",code,nullptr,true)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=HardwareComputeRoot.Get(); desc.CS={code->GetBufferPointer(),code->GetBufferSize()};
        const auto status=Evans::Renderer::CreateCachedComputePipelineState(Device,&desc,IID_PPV_ARGS(HardwareKernels[pass].GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create hardware queue pipeline",status);
    }
    const D3D_SHADER_MACRO graphics[]{{"EVANS_HARDWARE_GRAPHICS","1"},{nullptr,nullptr}};
    constexpr const char* vertices[]{"VSQueue","VSResolve"}; constexpr const char* pixels[]{"PSDepth","PSOwner","PSResolve"};
    for (unsigned i=0;i<HardwareVertices.size();++i) if (!HardwareVertices[i] && !CompileShader(vertices[i],"vs_5_0",HardwareVertices[i],graphics,true)) return false;
    for (unsigned i=0;i<HardwarePixels.size();++i) if (!HardwarePixels[i] && !CompileShader(pixels[i],"ps_5_0",HardwarePixels[i],graphics,true)) return false;
    return true;
}
ID3D12PipelineState* NaniteHardwarePipeline(unsigned phase,bool cull,bool mirrored)
{
    const unsigned key=(phase<<2)|unsigned(cull)|(unsigned(mirrored)<<1); const auto found=HardwarePipelines.find(key);
    if (found!=HardwarePipelines.end()) return found->second.Get();
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=HardwareGraphicsRoot.Get(); const auto& vertex=HardwareVertices[phase==2?1:0]; const auto& pixel=HardwarePixels[phase];
    desc.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; desc.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()}; desc.NumRenderTargets=2; desc.BlendState.IndependentBlendEnable=TRUE;
    for (unsigned i=0;i<2;++i)
    {
        auto& blend=desc.BlendState.RenderTarget[i]; blend.RenderTargetWriteMask=phase==2?D3D12_COLOR_WRITE_ENABLE_ALL:0;
        blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
        blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; desc.RTVFormats[i]=i?DXGI_FORMAT_R32G32_FLOAT:DXGI_FORMAT_R32_UINT;
    }
    desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.CullMode=cull&&phase!=2?D3D12_CULL_MODE_BACK:D3D12_CULL_MODE_NONE;
    desc.RasterizerState.FrontCounterClockwise=mirrored?FALSE:TRUE; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthEnable=TRUE; desc.DepthStencilState.DepthWriteMask=phase?D3D12_DEPTH_WRITE_MASK_ZERO:D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc=phase==0?D3D12_COMPARISON_FUNC_LESS:phase==1?D3D12_COMPARISON_FUNC_EQUAL:D3D12_COMPARISON_FUNC_ALWAYS;
    auto& stencil=desc.DepthStencilState.FrontFace; stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; desc.DepthStencilState.BackFace=stencil;
    desc.DSVFormat=DXGI_FORMAT_D32_FLOAT; desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.SampleDesc.Count=1; desc.SampleMask=UINT_MAX;
    ComPtr<ID3D12PipelineState> pipeline; const auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(pipeline.GetAddressOf()));
    if (FAILED(status)) { FailHRESULT("Create hardware ownership pipeline",status); return nullptr; }
    auto* result=pipeline.Get(); HardwarePipelines.emplace(key,std::move(pipeline)); return result;
}
bool DrawNaniteHardware(const VisibilityBufferDesc& frame)
{
    if (!HardwarePlan.Active()) return true;
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); CommandList->SetComputeRootSignature(HardwareComputeRoot.Get());
    CommandList->SetComputeRoot32BitConstants(0,sizeof(NaniteHardwareConstants)/sizeof(unsigned),&HardwarePlan.Constants,0);
    CommandList->SetComputeRootShaderResourceView(1,HardwareRanges->Buffer->GetGPUVirtualAddress()); CommandList->SetComputeRootShaderResourceView(2,NaniteRasterOutput->Buffer->GetGPUVirtualAddress());
    const std::shared_ptr<NaniteFrameBuffer> outputs[]{HardwareArguments,HardwareQueue,HardwareOwners};
    for (unsigned i=0;i<3;++i) { Transition(outputs[i]->Buffer.Get(),outputs[i]->State,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); CommandList->SetComputeRootUnorderedAccessView(i+3,outputs[i]->Buffer->GetGPUVirtualAddress()); }
    for (unsigned pass=0;pass<HardwareKernels.size();++pass)
    {
        const auto& c=HardwarePlan.Constants; const unsigned work=pass==0?std::max(c.Counts.Z,c.Counts.W):c.Counts.X,groups=(work+NaniteHardwareThreads-1)/NaniteHardwareThreads;
        CommandList->SetPipelineState(HardwareKernels[pass].Get()); CommandList->Dispatch(pass==2?1:std::min(groups,NaniteHardwareColumns),pass==2?1:(groups+NaniteHardwareColumns-1)/NaniteHardwareColumns,1);
        D3D12_RESOURCE_BARRIER barriers[3]{};
        for (unsigned i=0;i<3;++i) { barriers[i].Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barriers[i].UAV.pResource=outputs[i]->Buffer.Get(); }
        CommandList->ResourceBarrier(3,barriers);
    }
    if (frame.MeshShaders) BuildMeshDispatchArguments();
    else Transition(HardwareArguments->Buffer.Get(),HardwareArguments->State,D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    Transition(HardwareQueue->Buffer.Get(),HardwareQueue->State,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    D3D12_CPU_DESCRIPTOR_HANDLE targets[]{RTVArena.Cpu(VisibilityImages[0].RTV),RTVArena.Cpu(VisibilityImages[1].RTV)}; const auto depth=DSVArena.Cpu(MainDepthDSV);
    CommandList->OMSetRenderTargets(2,targets,FALSE,&depth); CommandList->SetGraphicsRootSignature(HardwareGraphicsRoot.Get()); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12DescriptorHeap* heaps[]{SRVArena.Heap.Get()}; CommandList->SetDescriptorHeaps(1,heaps);
    CommandList->SetGraphicsRootShaderResourceView(1,VisibilityGeometry->GetGPUVirtualAddress()); CommandList->SetGraphicsRootShaderResourceView(2,NaniteRasterInput->Buffer->GetGPUVirtualAddress());
    CommandList->SetGraphicsRootShaderResourceView(3,HardwareQueue->Buffer->GetGPUVirtualAddress()); CommandList->SetGraphicsRootUnorderedAccessView(5,HardwareOwners->Buffer->GetGPUVirtualAddress());
    if (frame.MeshShaders) CommandList->SetGraphicsRootShaderResourceView(8,HardwareArguments->Buffer->GetGPUVirtualAddress());
    const auto bind=[&](const NaniteHardwareBin& bin)
    {
        CommandList->SetGraphicsRoot32BitConstants(0,sizeof(NaniteHardwareDrawConstants)/sizeof(unsigned),&bin.Constants,0);
        const TextureHandle textures[]{bin.Diffuse,bin.Opacity};
        for (unsigned i=0;i<2;++i)
        {
            UINT descriptor=NullColorSRV;
            if (textures[i])
            {
                const auto found=Textures.find(textures[i].Value);
                if (found==Textures.end() || found->second.SRV==InvalidDescriptor || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Invalid hardware coverage texture");
                Transition(found->second.Resource.Get(),found->second.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); descriptor=found->second.SRV;
            }
            CommandList->SetGraphicsRootDescriptorTable(i+6,SRVArena.Gpu(descriptor));
        }
        return true;
    };
    for (unsigned phase=0;phase<2;++phase) for (unsigned bin=0;bin<HardwarePlan.Bins.size();++bin)
    {
        const auto& item=HardwarePlan.Bins[bin]; PipelineState state; state.BackFaceCulling=item.Cull;
        auto* pipeline=frame.MeshShaders?MeshPipeline(state,item.Mirrored,true,phase):NaniteHardwarePipeline(phase,item.Cull,item.Mirrored);
        if (!pipeline || !bind(item)) return false; CommandList->SetPipelineState(pipeline);
        if (frame.MeshShaders) CommandList->ExecuteIndirect(MeshDispatchSignature.Get(),1,MeshDispatchArguments->Buffer.Get(),bin*sizeof(MeshShaderGrid),nullptr,0);
        else CommandList->ExecuteIndirect(HardwareDrawSignature.Get(),1,HardwareArguments->Buffer.Get(),bin*NaniteHardwareBinWords*sizeof(unsigned),nullptr,0);
    }
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; barrier.UAV.pResource=HardwareOwners->Buffer.Get(); CommandList->ResourceBarrier(1,&barrier);
    Transition(HardwareOwners->Buffer.Get(),HardwareOwners->State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    CommandList->SetGraphicsRootShaderResourceView(4,HardwareOwners->Buffer->GetGPUVirtualAddress()); auto* resolve=NaniteHardwarePipeline(2,false,false);
    if (!resolve || !bind(HardwarePlan.Bins.front())) return false; CommandList->SetPipelineState(resolve); CommandList->DrawInstanced(3,1,0,0);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    auto& stats=PendingFrame.Statistics; stats.NaniteHardwareBinCount+=unsigned(HardwarePlan.Bins.size()); stats.NaniteHardwareDispatchCount+=unsigned(HardwareKernels.size());
    const auto dispatches=unsigned(HardwarePlan.Bins.size()*2);
    if (frame.MeshShaders)
    {
        stats.MeshShaderDispatchCount+=dispatches;
        stats.MeshShaderIndirectCount+=dispatches; ++stats.MeshShaderArgumentDispatchCount;
    }
    { stats.NaniteHardwareIndirectCount+=dispatches; stats.IndirectDrawCallCount+=dispatches; }
    return true;
}
