HdrSurface LumenHairIndirect,LumenHairSamples;
ComPtr<ID3D12RootSignature> LumenHairRoot;
ComPtr<ID3D12PipelineState> LumenHairComposite;
std::unordered_map<UINT,ComPtr<ID3D12PipelineState>> LumenHairPipelines;
void DestroyLumenHairReceiver()
{
    for (auto* image:{&LumenHairIndirect,&LumenHairSamples}) { SRVArena.Release(image->SRV); RTVArena.Release(image->RTV); *image={}; }
    LumenHairRoot.Reset(); LumenHairComposite.Reset(); LumenHairPipelines.clear();
}
bool EnsureLumenHairReceiver()
{
    const float clear[4]{};
    if (!CreateHdrSurface(LumenHairIndirect,CreateInfo.Width,CreateInfo.Height,DXGI_FORMAT_R16G16B16A16_FLOAT,1,clear) ||
        (FrameSampleCount>1 && !CreateHdrSurface(LumenHairSamples,CreateInfo.Width,CreateInfo.Height,DXGI_FORMAT_R16G16B16A16_FLOAT,FrameSampleCount,clear))) return false;
    if (!LumenHairRoot)
    {
        D3D12_DESCRIPTOR_RANGE ranges[2]{}; D3D12_ROOT_PARAMETER parameters[4]{};
        for (UINT i=0;i<2;++i)
        {
            parameters[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[i].Descriptor.ShaderRegister=i;
            parameters[i].ShaderVisibility=i?D3D12_SHADER_VISIBILITY_PIXEL:D3D12_SHADER_VISIBILITY_VERTEX;
            ranges[i]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,i,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            parameters[i+2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[i+2].DescriptorTable={1,&ranges[i]}; parameters[i+2].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=4; desc.pParameters=parameters; desc.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> binary,errors; auto status=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,binary.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(status)) return FailHRESULT("Serialize fiber irradiance root",status);
        status=Evans::Renderer::CreateCachedRootSignature(Device,0,binary->GetBufferPointer(),binary->GetBufferSize(),IID_PPV_ARGS(LumenHairRoot.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create fiber irradiance root",status);
    }
    if (LumenHairPipelines.count(FrameSampleCount) && LumenHairComposite) return true;
    ComPtr<ID3DBlob> ribbon,indirect,vertex,composite;
    if (!LoadTechniqueShaderSource("DX12","HairFur.hlsl",ShaderCode,LastError) || !CompileShader("VSMain","vs_5_0",ribbon,nullptr,true) ||
        !LoadTechniqueShaderSource("DX12","LumenHair.hlsl",ShaderCode,LastError) || !CompileShader("PSIndirect","ps_5_0",indirect,nullptr,true) ||
        !CompileShader("VSCompose","vs_5_0",vertex,nullptr,true) || !CompileShader("PSCompose","ps_5_0",composite,nullptr,true)) return false;
    const D3D12_INPUT_ELEMENT_DESC elements[]{
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(HairVertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(HairVertex,Tangent),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Reflectance),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",1,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Emission),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Ribbon),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{}; desc.pRootSignature=LumenHairRoot.Get();
    desc.VS={ribbon->GetBufferPointer(),ribbon->GetBufferSize()}; desc.PS={indirect->GetBufferPointer(),indirect->GetBufferSize()}; desc.InputLayout={elements,UINT(std::size(elements))};
    auto& blend=desc.BlendState.RenderTarget[0]; blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
    blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask=UINT_MAX; desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE; desc.RasterizerState.DepthClipEnable=TRUE;
    desc.DepthStencilState.DepthEnable=TRUE; desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_LESS_EQUAL;
    desc.DepthStencilState.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK; desc.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
    auto& stencil=desc.DepthStencilState.FrontFace; stencil.StencilFailOp=stencil.StencilDepthFailOp=stencil.StencilPassOp=D3D12_STENCIL_OP_KEEP; stencil.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; desc.DepthStencilState.BackFace=stencil;
    desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; desc.NumRenderTargets=1; desc.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.DSVFormat=DXGI_FORMAT_D32_FLOAT; desc.SampleDesc.Count=FrameSampleCount;
    if (!LumenHairPipelines.count(FrameSampleCount))
    {
        ComPtr<ID3D12PipelineState> pipeline; const auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(pipeline.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create fiber irradiance pipeline",status);
        LumenHairPipelines.emplace(FrameSampleCount,std::move(pipeline));
    }
    if (!LumenHairComposite)
    {
        desc.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; desc.PS={composite->GetBufferPointer(),composite->GetBufferSize()}; desc.InputLayout={}; desc.SampleDesc.Count=1; desc.DepthStencilState.DepthEnable=FALSE; desc.DSVFormat=DXGI_FORMAT_UNKNOWN;
        const auto status=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&desc,IID_PPV_ARGS(LumenHairComposite.GetAddressOf()));
        if (FAILED(status)) return FailHRESULT("Create fiber irradiance composition",status);
    }
    return true;
}
bool ResolveLumenHairReceiver()
{
    if (!Hair.LumenReceiver) return true;
    if (!Hair.Resolved || !Hair.HasPrimaryView || !LumenVolumeFrame.Grid.W) return Fail("Fiber irradiance requires a completed primary surface and current SH volume");
    LumenHairReceiverFrame frame; HairFurConstants geometry;
    if (!BuildLumenHairReceiverFrame(LumenVolumeFrame,Hair.Frame.Settings,Hair.PrimaryView.Lighting,Hair.PrimaryView.ClipPlane,frame,LastError) ||
        !BuildHairFurReplayConstants(Hair.PrimaryView,Hair.Frame,HairFurRasterPhase::Surface,{},geometry,LastError) || !EnsureLumenHairReceiver()) return false;
    const auto vertexOffset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT),fragmentOffset=AlignUp(vertexOffset+sizeof(geometry),D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (fragmentOffset>ConstantBufferCapacity || sizeof(frame)>ConstantBufferCapacity-fragmentOffset) return Fail("Fiber irradiance exhausted frame constants");
    std::memcpy(ConstantData+vertexOffset,&geometry,sizeof(geometry)); std::memcpy(ConstantData+fragmentOffset,&frame,sizeof(frame)); ConstantOffset=fragmentOffset+sizeof(frame);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    auto& target=FrameSampleCount>1?LumenHairSamples:LumenHairIndirect;
    Transition(target.Resource.Get(),target.State,D3D12_RESOURCE_STATE_RENDER_TARGET); Transition(LumenVolumeAtlas.Resource.Get(),LumenVolumeAtlas.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    auto* depth=FrameSampleCount>1?MultisampleDepth.Resource.Get():MainDepth.Get(); auto& depthState=FrameSampleCount>1?MultisampleDepth.State:MainDepthState;
    Transition(depth,depthState,D3D12_RESOURCE_STATE_DEPTH_READ);
    const auto rtv=RTVArena.Cpu(target.RTV),dsv=DSVArena.Cpu(FrameSampleCount>1?MultisampleDepthReadDSV:MainDepthReadOnlyDSV); const float clear[4]{};
    CommandList->ClearRenderTargetView(rtv,clear,0,nullptr); CommandList->OMSetRenderTargets(1,&rtv,FALSE,&dsv);
    CommandList->SetGraphicsRootSignature(LumenHairRoot.Get()); CommandList->SetPipelineState(LumenHairPipelines.at(FrameSampleCount).Get());
    CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+vertexOffset); CommandList->SetGraphicsRootConstantBufferView(1,ConstantBuffer->GetGPUVirtualAddress()+fragmentOffset);
    CommandList->SetGraphicsRootDescriptorTable(2,SRVArena.Gpu(LumenVolumeAtlas.SRV));
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const D3D12_RECT rectangle{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&rectangle); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vertices{HairVertices->GetGPUVirtualAddress(),UINT(Hair.Frame.Geometry->Vertices.size()*sizeof(HairVertex)),sizeof(HairVertex)};
    const D3D12_INDEX_BUFFER_VIEW indices{HairIndices->GetGPUVirtualAddress(),UINT(Hair.Frame.Geometry->Indices.size()*sizeof(unsigned)),DXGI_FORMAT_R32_UINT};
    CommandList->IASetVertexBuffers(0,1,&vertices); CommandList->IASetIndexBuffer(&indices); CommandList->DrawIndexedInstanced(UINT(Hair.Frame.Geometry->Indices.size()),1,0,0,0); CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    if (FrameSampleCount>1)
    {
        Transition(target.Resource.Get(),target.State,D3D12_RESOURCE_STATE_RESOLVE_SOURCE); Transition(LumenHairIndirect.Resource.Get(),LumenHairIndirect.State,D3D12_RESOURCE_STATE_RESOLVE_DEST);
        CommandList->ResolveSubresource(LumenHairIndirect.Resource.Get(),0,target.Resource.Get(),0,DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    Transition(LumenHairIndirect.Resource.Get(),LumenHairIndirect.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); Transition(HdrSurfaces[HdrScene].Resource.Get(),HdrSurfaces[HdrScene].State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    Transition(LumenGatherTarget.Resource.Get(),LumenGatherTarget.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    const auto composition=RTVArena.Cpu(LumenGatherTarget.RTV); CommandList->OMSetRenderTargets(1,&composition,FALSE,nullptr); CommandList->SetPipelineState(LumenHairComposite.Get());
    CommandList->SetGraphicsRootDescriptorTable(2,SRVArena.Gpu(LumenHairIndirect.SRV)); CommandList->SetGraphicsRootDescriptorTable(3,SRVArena.Gpu(HdrSurfaces[HdrScene].SRV)); CommandList->DrawInstanced(3,1,0,0); CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(LumenGatherTarget.Resource.Get(),LumenGatherTarget.State,D3D12_RESOURCE_STATE_COPY_SOURCE); Transition(HdrSurfaces[HdrScene].Resource.Get(),HdrSurfaces[HdrScene].State,D3D12_RESOURCE_STATE_COPY_DEST);
    CommandList->CopyResource(HdrSurfaces[HdrScene].Resource.Get(),LumenGatherTarget.Resource.Get());
    ++PendingFrame.Statistics.LumenHairReceiverPassCount; ++PendingFrame.Statistics.GlobalIlluminationPassCount; return true;
}
