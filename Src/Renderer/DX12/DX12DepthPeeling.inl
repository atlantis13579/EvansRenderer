// DX12RHI::Impl members. Resource states are checkpointed before recording and
// restored on abort. Uniform memory is reserved before recording repeated draws.
std::array<HdrSurface,5> PeelingImages; // D32 pair, RGBA32F pair, RGBA16F shaded layer.
std::array<UINT,2> PeelingDSV{InvalidDescriptor,InvalidDescriptor};
static constexpr UINT DepthPeelingSamplerRoot = 9;
ComPtr<ID3D12RootSignature> PeelingRoot;
std::array<ComPtr<ID3D12PipelineState>,6> PeelingPipelines;
DepthPeelingFrame FramePeeling;
DepthPeelingSequence PeelingSequence;
void ResetPeeling(bool complete=false)
{
    for (auto& image:PeelingImages) { image.Resource.Reset(); image.State=image.FrameStartState=D3D12_RESOURCE_STATE_RENDER_TARGET; }
    PeelingSequence.Abort(); FramePeeling={};
    if (!complete) return;
    PeelingImages={}; PeelingDSV={InvalidDescriptor,InvalidDescriptor}; PeelingRoot.Reset(); PeelingPipelines={};
}

bool EnsurePeelingResources()
{
    if (!ValidateDepthPeelingFrame(FramePeeling,LastError)) return false;
    if (!HdrSceneActive || FrameSampleCount!=1 || WeightedActive)
        return Fail("Depth peeling requires single-sample HDR without weighted accumulation");
    for (unsigned index=2;index<PeelingImages.size();++index)
    {
        const float clear[4]{0,0,0,index==4 ? 0.0f : 1.0f};
        if (!CreateHdrSurface(PeelingImages[index],CreateInfo.Width,CreateInfo.Height,
            index==4 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT,1,clear)) return false;
    }
    for (unsigned index=0;index<2;++index)
    {
        auto& depth=PeelingImages[index];
        if (depth.Resource && depth.Width==CreateInfo.Width && depth.Height==CreateInfo.Height) continue;
        if (depth.SRV==InvalidDescriptor) depth.SRV=SRVArena.Allocate();
        if (PeelingDSV[index]==InvalidDescriptor) PeelingDSV[index]=DSVArena.Allocate();
        if (depth.SRV==InvalidDescriptor || PeelingDSV[index]==InvalidDescriptor) return Fail("DX12 OIT depth descriptor exhaustion");
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width=CreateInfo.Width; desc.Height=CreateInfo.Height; desc.DepthOrArraySize=desc.MipLevels=1; desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_R32_TYPELESS; desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); D3D12_CLEAR_VALUE value{}; value.Format=DXGI_FORMAT_D32_FLOAT; value.DepthStencil.Depth=PeelingClearDepth;
        ComPtr<ID3D12Resource> resource;
        const auto result=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_DEPTH_WRITE,&value,IID_PPV_ARGS(resource.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateCommittedResource(OIT depth)",result);
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv{}; dsv.Format=DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D;
        Device->CreateDepthStencilView(resource.Get(),&dsv,DSVArena.Cpu(PeelingDSV[index]));
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format=DXGI_FORMAT_R32_FLOAT; srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srv.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(resource.Get(),&srv,SRVArena.Cpu(depth.SRV));
        depth.Resource=std::move(resource); depth.Width=CreateInfo.Width; depth.Height=CreateInfo.Height;
        depth.State=depth.FrameStartState=D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }
    if (!PeelingRoot)
    {
        std::array<D3D12_DESCRIPTOR_RANGE,8> ranges{}; std::array<D3D12_ROOT_PARAMETER, DepthPeelingSamplerRoot + 1> parameters{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[0].Descriptor={0,0}; parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        for (UINT slot=0;slot<ranges.size();++slot)
        {
            ranges[slot].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[slot].NumDescriptors=1; ranges[slot].BaseShaderRegister=slot;
            ranges[slot].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[slot+1].DescriptorTable={1,&ranges[slot]}; parameters[slot+1].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_DESCRIPTOR_RANGE materialSamplerRange{};
        DescribeMaterialSampler(parameters[DepthPeelingSamplerRoot], materialSamplerRange);
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(parameters.size()); desc.pParameters=parameters.data();
        desc.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> code,error; auto result=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),error.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(OIT)",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(PeelingRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(OIT)",result);
    }
    if (!PeelingPipelines[0] || !PeelingPipelines[5])
    {
        if (!LoadTechniqueShaderSource("DX12","OIT.hlsl",ShaderCode,LastError)) return false;
        std::array<ComPtr<ID3DBlob>,6> stages;
        for (unsigned phase=0;phase<3;++phase)
        {
            const D3D_SHADER_MACRO definitions[]{{phase==0 ? "EVANS_OIT_CAPTURE" : "EVANS_OIT_ACCUMULATE","1"},{nullptr,nullptr}};
            if (!CompileShader("VSMain","vs_5_0",stages[phase*2],phase<2 ? definitions : nullptr,true) ||
                !CompileShader("PSMain","ps_5_0",stages[phase*2+1],phase<2 ? definitions : nullptr,true)) return false;
        }
        const D3D12_INPUT_ELEMENT_DESC attributes[]{
            {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Normal),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(Vertex,Color),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(Vertex,TexCoord),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
        for (UINT variant=0;variant<PeelingPipelines.size();++variant)
        {
            if (PeelingPipelines[variant]) continue; const bool composite=variant>=4; const unsigned shader=composite ? (variant-3)*2 : 0;
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=PeelingRoot.Get();
            pipeline.VS={stages[shader]->GetBufferPointer(),stages[shader]->GetBufferSize()}; pipeline.PS={stages[shader+1]->GetBufferPointer(),stages[shader+1]->GetBufferSize()};
            if (!composite) pipeline.InputLayout={attributes,UINT(std::size(attributes))};
            pipeline.NumRenderTargets=composite ? 1 : 0;
            for (UINT index=0;index<pipeline.NumRenderTargets;++index)
            {
                pipeline.RTVFormats[index]=variant==4 ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT; auto& blend=pipeline.BlendState.RenderTarget[index];
                blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
                blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; blend.LogicOp=D3D12_LOGIC_OP_NOOP; blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
            }
            pipeline.SampleMask=UINT_MAX; pipeline.SampleDesc.Count=1; pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; pipeline.RasterizerState.CullMode=!composite && (variant&1) ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
            pipeline.RasterizerState.FrontCounterClockwise=(variant&2)==0; pipeline.RasterizerState.DepthClipEnable=TRUE;
            pipeline.DepthStencilState.DepthEnable=!composite; pipeline.DepthStencilState.DepthWriteMask=composite ? D3D12_DEPTH_WRITE_MASK_ZERO : D3D12_DEPTH_WRITE_MASK_ALL;
            pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_LESS; pipeline.DepthStencilState.StencilReadMask=pipeline.DepthStencilState.StencilWriteMask=D3D12_DEFAULT_STENCIL_READ_MASK;
            pipeline.DepthStencilState.FrontFace={D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_COMPARISON_FUNC_ALWAYS}; pipeline.DepthStencilState.BackFace=pipeline.DepthStencilState.FrontFace;
            pipeline.DSVFormat=composite ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_D32_FLOAT;
            const auto result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(PeelingPipelines[variant].GetAddressOf()));
            if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(OIT)",result);
        }
    }
    // Reserve complete per-layer capture and material uniforms before recording;
    // the ordinary four-megabyte frame allocation must not cap the layer budget.
    const auto aligned=[](std::size_t bytes) { return AlignUp(bytes,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT); };
    const std::size_t perDraw=aligned(sizeof(RealtimeDrawConstants))+aligned(sizeof(DepthPeelingConstants))+
        aligned(std::max(sizeof(PCSSConstants),sizeof(VSMConstants)))+aligned(sizeof(CascadedShadowConstants))+
        aligned(sizeof(VolumetricFogConstants))+2*aligned(sizeof(ShaderUInt4)); // Contact and light-list selection.
    const std::size_t repeats=FramePeeling.Settings.LayerCount+1;
    if (FramePeeling.Draws.size()>(SIZE_MAX-DefaultConstantBufferCapacity)/perDraw/repeats-1)
        return Fail("DX12 OIT uniform reservation overflow");
    const auto capacity=DefaultConstantBufferCapacity+(FramePeeling.Draws.size()+1)*repeats*perDraw;
    if (capacity>ConstantBufferCapacity)
    {
        ComPtr<ID3D12Resource> resource;
        if (!CreateBufferResource(capacity,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,resource)) return false;
        void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto result=resource->Map(0,&noRead,&mapped);
        if (FAILED(result)) return FailHRESULT("Map(OIT frame uniforms)",result);
        if (ConstantBuffer && ConstantData) ConstantBuffer->Unmap(0,nullptr);
        ConstantBuffer=std::move(resource); ConstantData=static_cast<std::uint8_t*>(mapped); ConstantBufferCapacity=capacity;
    }
    return PeelingSequence.Begin(FramePeeling.Settings.LayerCount,LastError);
}

bool BeginPeelingLayer(D3D12_CPU_DESCRIPTOR_HANDLE& color,D3D12_CPU_DESCRIPTOR_HANDLE& depth)
{
    if (!PeelingSequence.Advance(DepthPeelingEvent::BeginShading,LastError)) return false;
    auto& layer=PeelingImages[4]; auto& captured=PeelingImages[PeelingSequence.Step().DepthWrite];
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    Transition(layer.Resource.Get(),layer.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    Transition(captured.Resource.Get(),captured.State,D3D12_RESOURCE_STATE_DEPTH_WRITE);
    color=RTVArena.Cpu(layer.RTV); depth=DSVArena.Cpu(PeelingDSV[PeelingSequence.Step().DepthWrite]);
    const float clear[4]{}; CommandList->ClearRenderTargetView(color,clear,0,nullptr); return true;
}
bool ProcessPeeling(DepthPeelingPass pass)
{
    const auto step=PeelingSequence.Step(); DepthPeelingConstants base;
    if (!BuildDepthPeelingConstants(FramePeeling.ViewProjection,CreateInfo.Width,CreateInfo.Height,step.FirstLayer,FramePeeling.Settings.ShowOverflow,base))
        return Fail("Invalid DX12 OIT frame constants");
    auto constants=[&](const DepthPeelingConstants& value) {
        const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        if (offset>ConstantBufferCapacity || sizeof(value)>ConstantBufferCapacity-offset) return Fail("DX12 OIT exhausted frame uniforms");
        std::memcpy(ConstantData+offset,&value,sizeof(value)); ConstantOffset=offset+sizeof(value);
        CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset); return true;
    };
    constexpr auto sampled=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    auto bind=[&](const std::array<UINT,8>& inputs) { for (UINT slot=0;slot<inputs.size();++slot) CommandList->SetGraphicsRootDescriptorTable(slot+1,SRVArena.Gpu(inputs[slot])); };
    auto texture=[&](TextureHandle handle,UINT& descriptor) {
        descriptor=NullColorSRV; if (!handle) return true; const auto found=Textures.find(handle.Value);
        if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8 || !found->second.Resource) return Fail("Invalid DX12 OIT material texture");
        Transition(found->second.Resource.Get(),found->second.State,sampled); descriptor=found->second.SRV; return true;
    };
    const auto capture=pass==DepthPeelingPass::Capture,compose=pass==DepthPeelingPass::Compose;
    if (capture)
    {
        if (!PeelingSequence.Advance(DepthPeelingEvent::BeginCapture,LastError)) return false;
    }
    else if ((!compose && PeelingSequence.Stage()!=DepthPeelingStage::NeedAccumulation) ||
        (compose && PeelingSequence.Stage()!=DepthPeelingStage::NeedComposition)) return Fail("DX12 OIT resolve is out of order");
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    CommandList->SetGraphicsRootSignature(PeelingRoot.Get());
    BindMaterialSampler(DepthPeelingSamplerRoot); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const D3D12_RECT scissor{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor);
    if (capture)
    {
        if (step.FirstLayer)
        {
            const float initial[4]{0,0,0,1};
            for (unsigned index=0;index<2;++index)
            {
                auto& depth=PeelingImages[index]; Transition(depth.Resource.Get(),depth.State,D3D12_RESOURCE_STATE_DEPTH_WRITE);
                CommandList->ClearDepthStencilView(DSVArena.Cpu(PeelingDSV[index]),D3D12_CLEAR_FLAG_DEPTH,PeelingClearDepth,0,0,nullptr);
                auto& accumulated=PeelingImages[index+2]; Transition(accumulated.Resource.Get(),accumulated.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
                CommandList->ClearRenderTargetView(RTVArena.Cpu(accumulated.RTV),initial,0,nullptr);
            }
        }
        auto& previous=PeelingImages[step.DepthRead]; auto& depth=PeelingImages[step.DepthWrite];
        Transition(MainDepth.Get(),MainDepthState,sampled); Transition(previous.Resource.Get(),previous.State,sampled);
        Transition(depth.Resource.Get(),depth.State,D3D12_RESOURCE_STATE_DEPTH_WRITE);
        const auto target=DSVArena.Cpu(PeelingDSV[step.DepthWrite]);
        CommandList->ClearDepthStencilView(target,D3D12_CLEAR_FLAG_DEPTH,PeelingClearDepth,0,0,nullptr);
        CommandList->OMSetRenderTargets(0,nullptr,FALSE,&target);
        for (const auto& draw:FramePeeling.Draws)
        {
            if (!draw.IndexCount) continue;
            const auto vertex=Buffers.find(draw.VertexBuffer.Value),index=Buffers.find(draw.IndexBuffer.Value);
            if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Invalid DX12 OIT buffers");
            if (!ValidateDepthPeelingBuffers(draw,vertex->second.Description,index->second.Description,vertex->second.Bytes.size(),index->second.Bytes,LastError,vertex->second.NaniteGeometry && index->second.NaniteGeometry)) return false;
            DepthPeelingConstants value;
            if (!BuildDepthPeelingDrawConstants(base,draw.World,draw.Opacity,draw.AlphaCutoff,bool(draw.DiffuseTexture),bool(draw.OpacityTexture),value))
                return Fail("Invalid DX12 OIT draw constants");
            UINT diffuse,opacity; if (!texture(draw.DiffuseTexture,diffuse) || !texture(draw.OpacityTexture,opacity) || !constants(value)) return false;
            bind({MainDepthSRV,previous.SRV,diffuse,opacity,NullColorSRV,NullColorSRV,NullColorSRV,MainDepthSRV});
            Transition(vertex->second.Resource.Get(),vertex->second.State,D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            Transition(index->second.Resource.Get(),index->second.State,D3D12_RESOURCE_STATE_INDEX_BUFFER);
            CommandList->IASetVertexBuffers(0,1,&vertex->second.VertexView); CommandList->IASetIndexBuffer(&index->second.IndexView);
            CommandList->SetPipelineState(PeelingPipelines[(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u)].Get());
            CommandList->DrawIndexedInstanced(draw.IndexCount,1,draw.FirstIndex,draw.BaseVertex,0);
        }
        CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); return PeelingSequence.Advance(DepthPeelingEvent::EndCapture,LastError);
    }
    auto& layer=PeelingImages[4]; auto& accumulated=PeelingImages[step.AccumulationRead+2];
    auto& depth=PeelingImages[step.DepthWrite]; auto& scene=HdrSurfaces[HdrScene];
    auto& output=compose ? layer : PeelingImages[step.AccumulationWrite+2];
    Transition(accumulated.Resource.Get(),accumulated.State,sampled); Transition(depth.Resource.Get(),depth.State,sampled);
    Transition(scene.Resource.Get(),scene.State,sampled);
    if (!compose) Transition(layer.Resource.Get(),layer.State,sampled);
    Transition(output.Resource.Get(),output.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    const auto target=RTVArena.Cpu(output.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
    if (!constants(base)) return false;
    bind({NullColorSRV,NullColorSRV,NullColorSRV,NullColorSRV,compose ? NullColorSRV : layer.SRV,accumulated.SRV,scene.SRV,depth.SRV});
    CommandList->SetPipelineState(PeelingPipelines[compose ? 5 : 4].Get()); CommandList->DrawInstanced(3,1,0,0);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    if (compose)
    {
        Transition(layer.Resource.Get(),layer.State,D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST); CommandList->CopyResource(scene.Resource.Get(),layer.Resource.Get());
    }
    return PeelingSequence.Advance(compose ? DepthPeelingEvent::Compose : DepthPeelingEvent::Accumulate,LastError);
}
