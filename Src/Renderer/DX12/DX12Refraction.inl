// DX12RHI::Impl members. Geometry uploads are CPU-visible immutable snapshots;
// image states participate in the same begin/abort checkpoint as other effects.
std::array<HdrSurface,4> RefractionImages; // Normal, tint, composition, D32.
UINT RefractionDSV=InvalidDescriptor;
ComPtr<ID3D12Resource> RefractionVertices,RefractionIndices;
D3D12_VERTEX_BUFFER_VIEW RefractionVertexView{};
D3D12_INDEX_BUFFER_VIEW RefractionIndexView{};
ComPtr<ID3D12RootSignature> RefractionRoot;
std::array<ComPtr<ID3D12PipelineState>,5> RefractionPipelines;
RefractionFrame FrameRefraction;
Techniques::RefractionSettings FrameRefractionSettings;
std::shared_ptr<const RefractionGeometry> UploadedRefraction;
bool RefractionActive=false,RefractionResolved=false;

void ResetRefraction(bool complete=false)
{
    for (auto& image:RefractionImages) { image.Resource.Reset(); image.State=image.FrameStartState=D3D12_RESOURCE_STATE_RENDER_TARGET; }
    RefractionVertices.Reset(); RefractionIndices.Reset(); RefractionVertexView={}; RefractionIndexView={}; UploadedRefraction.reset(); FrameRefraction={};
    if (!complete) return;
    RefractionImages={}; RefractionDSV=InvalidDescriptor; RefractionRoot.Reset(); RefractionPipelines={}; RefractionActive=RefractionResolved=false;
}

bool EnsureRefractionResources()
{
    if (!FrameRefraction.Geometry || FrameRefraction.Draws.empty()) return true;
    const float clear[4]{};
    for (unsigned index=0;index<3;++index)
        if (!CreateHdrSurface(RefractionImages[index],CreateInfo.Width,CreateInfo.Height,DXGI_FORMAT_R16G16B16A16_FLOAT,1,clear)) return false;
    auto& depth=RefractionImages[3];
    if (!depth.Resource || depth.Width!=CreateInfo.Width || depth.Height!=CreateInfo.Height)
    {
        if (depth.SRV==InvalidDescriptor) depth.SRV=SRVArena.Allocate();
        if (RefractionDSV==InvalidDescriptor) RefractionDSV=DSVArena.Allocate();
        if (depth.SRV==InvalidDescriptor || RefractionDSV==InvalidDescriptor) return Fail("DX12 refraction depth descriptor exhaustion");
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width=CreateInfo.Width; desc.Height=CreateInfo.Height; desc.DepthOrArraySize=desc.MipLevels=1; desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_R32_TYPELESS; desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        const auto heap=HeapProperties(D3D12_HEAP_TYPE_DEFAULT); D3D12_CLEAR_VALUE value{}; value.Format=DXGI_FORMAT_D32_FLOAT; value.DepthStencil.Depth=1;
        ComPtr<ID3D12Resource> resource;
        auto result=Device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_DEPTH_WRITE,&value,IID_PPV_ARGS(resource.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateCommittedResource(refraction depth)",result);
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv{}; dsv.Format=DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D;
        Device->CreateDepthStencilView(resource.Get(),&dsv,DSVArena.Cpu(RefractionDSV));
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format=DXGI_FORMAT_R32_FLOAT; srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srv.Texture2D.MipLevels=1;
        Device->CreateShaderResourceView(resource.Get(),&srv,SRVArena.Cpu(depth.SRV));
        depth.Resource=std::move(resource); depth.Width=CreateInfo.Width; depth.Height=CreateInfo.Height;
        depth.State=depth.FrameStartState=D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }
    if (UploadedRefraction!=FrameRefraction.Geometry)
    {
        const auto& geometry=*FrameRefraction.Geometry;
        const auto vertexBytes=geometry.Vertices.size()*sizeof(Vertex),indexBytes=geometry.Indices.size()*sizeof(std::uint32_t);
        if (!vertexBytes || !indexBytes || vertexBytes>UINT_MAX || indexBytes>UINT_MAX) return Fail("DX12 refraction geometry size is invalid");
        ComPtr<ID3D12Resource> vertices,indices;
        auto upload=[&](std::size_t bytes,const void* data,ComPtr<ID3D12Resource>& output) {
            if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,output)) return false;
            void* mapped=nullptr; const D3D12_RANGE noRead{0,0}; const auto result=output->Map(0,&noRead,&mapped);
            if (FAILED(result)) return FailHRESULT("Map(refraction geometry)",result);
            std::memcpy(mapped,data,bytes); output->Unmap(0,nullptr); return true;
        };
        if (!upload(vertexBytes,geometry.Vertices.data(),vertices) || !upload(indexBytes,geometry.Indices.data(),indices)) return false;
        RefractionVertices=std::move(vertices); RefractionIndices=std::move(indices);
        RefractionVertexView={RefractionVertices->GetGPUVirtualAddress(),static_cast<UINT>(vertexBytes),sizeof(Vertex)};
        RefractionIndexView={RefractionIndices->GetGPUVirtualAddress(),static_cast<UINT>(indexBytes),DXGI_FORMAT_R32_UINT}; UploadedRefraction=FrameRefraction.Geometry;
    }
    if (!RefractionRoot)
    {
        std::array<D3D12_DESCRIPTOR_RANGE,8> ranges{}; std::array<D3D12_ROOT_PARAMETER,11> parameters{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[0].Descriptor={0,0}; parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        for (UINT slot=0;slot<ranges.size();++slot)
        {
            ranges[slot].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[slot].NumDescriptors=1; ranges[slot].BaseShaderRegister=slot;
            ranges[slot].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            parameters[slot+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[slot+1].DescriptorTable={1,&ranges[slot]}; parameters[slot+1].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        constexpr UINT VolumeUniformRoot=9,VolumeImageRoot=10;
        parameters[VolumeUniformRoot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[VolumeUniformRoot].Descriptor={LumenVolumeConstantRegister,0}; parameters[VolumeUniformRoot].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_DESCRIPTOR_RANGE volumeRange{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,LumenVolumeImageRegister,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
        parameters[VolumeImageRoot].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[VolumeImageRoot].DescriptorTable={1,&volumeRange}; parameters[VolumeImageRoot].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS; sampler.MaxAnisotropy=1; sampler.MaxLOD=D3D12_FLOAT32_MAX; sampler.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(parameters.size()); desc.pParameters=parameters.data();
        std::array<D3D12_STATIC_SAMPLER_DESC,2> samplers{sampler,sampler}; samplers[1].ShaderRegister=LumenVolumeSamplerRegister;
        samplers[1].AddressU=samplers[1].AddressV=samplers[1].AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        desc.NumStaticSamplers=UINT(samplers.size()); desc.pStaticSamplers=samplers.data(); desc.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> code,error; auto result=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,code.GetAddressOf(),error.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(refraction)",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(RefractionRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(refraction)",result);
    }
    if (!RefractionPipelines[0] || !RefractionPipelines[4])
    {
        if (!LoadTechniqueShaderSource("DX12","Refraction.hlsl",ShaderCode,LastError)) return false;
        std::array<ComPtr<ID3DBlob>,4> stages;
        const D3D_SHADER_MACRO definitions[]{{"EVANS_REFRACTION_CAPTURE","1"},{nullptr,nullptr}};
        if (!CompileShader("VSMain","vs_5_0",stages[0],definitions,true) || !CompileShader("PSMain","ps_5_0",stages[1],definitions,true) ||
            !CompileShader("VSMain","vs_5_0",stages[2],nullptr,true) || !CompileShader("PSMain","ps_5_0",stages[3],nullptr,true)) return false;
        const D3D12_INPUT_ELEMENT_DESC attributes[]{
            {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Normal),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(Vertex,Color),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
            {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(Vertex,TexCoord),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
        for (UINT variant=0;variant<RefractionPipelines.size();++variant)
        {
            if (RefractionPipelines[variant]) continue; const bool composite=variant==4; const unsigned shader=composite ? 2 : 0;
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=RefractionRoot.Get();
            pipeline.VS={stages[shader]->GetBufferPointer(),stages[shader]->GetBufferSize()}; pipeline.PS={stages[shader+1]->GetBufferPointer(),stages[shader+1]->GetBufferSize()};
            if (!composite) pipeline.InputLayout={attributes,UINT(std::size(attributes))};
            pipeline.NumRenderTargets=composite ? 1 : 2;
            for (UINT index=0;index<pipeline.NumRenderTargets;++index)
            {
                pipeline.RTVFormats[index]=DXGI_FORMAT_R16G16B16A16_FLOAT; auto& blend=pipeline.BlendState.RenderTarget[index];
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
            const auto result=Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(RefractionPipelines[variant].GetAddressOf()));
            if (FAILED(result)) return FailHRESULT("CreateGraphicsPipelineState(refraction)",result);
        }
    }
    return true;
}

bool ResolveRefraction(const RefractionDesc& desc)
{
    if (!FrameRefraction.Geometry || FrameRefraction.Draws.empty()) { RefractionResolved=true; return true; }
    if (UploadedRefraction!=FrameRefraction.Geometry) return Fail("DX12 refraction snapshot changed during a frame");
    RefractionConstants base;
    if (!BuildRefractionConstants(desc.ViewProjection,desc.Environment,FrameRefractionSettings,CreateInfo.Width,CreateInfo.Height,base)) return Fail("DX12 refraction projection is invalid");
    auto constants=[&](const RefractionConstants& value) {
        const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        if (offset>ConstantBufferCapacity || sizeof(value)>ConstantBufferCapacity-offset) return Fail("DX12 refraction exhausted frame constants");
        std::memcpy(ConstantData+offset,&value,sizeof(value)); ConstantOffset=offset+sizeof(value);
        CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset); return true;
    };
    constexpr auto sampled=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    auto bind=[&](const std::array<UINT,8>& inputs) { for (UINT slot=0;slot<inputs.size();++slot) CommandList->SetGraphicsRootDescriptorTable(slot+1,SRVArena.Gpu(inputs[slot])); };
    auto texture=[&](TextureHandle handle,UINT& descriptor) {
        descriptor=NullColorSRV; if (!handle) return true; const auto found=Textures.find(handle.Value);
        if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8 || !found->second.Resource) return Fail("DX12 refraction material texture is invalid");
        Transition(found->second.Resource.Get(),found->second.State,sampled); descriptor=found->second.SRV; return true;
    };
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); Transition(MainDepth.Get(),MainDepthState,sampled);
    for (unsigned index=0;index<2;++index) Transition(RefractionImages[index].Resource.Get(),RefractionImages[index].State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    auto& depth=RefractionImages[3]; Transition(depth.Resource.Get(),depth.State,D3D12_RESOURCE_STATE_DEPTH_WRITE);
    const D3D12_CPU_DESCRIPTOR_HANDLE targets[]{RTVArena.Cpu(RefractionImages[0].RTV),RTVArena.Cpu(RefractionImages[1].RTV)};
    const auto depthTarget=DSVArena.Cpu(RefractionDSV); const float clear[4]{};
    for (auto target:targets) CommandList->ClearRenderTargetView(target,clear,0,nullptr);
    CommandList->ClearDepthStencilView(depthTarget,D3D12_CLEAR_FLAG_DEPTH,1,0,0,nullptr); CommandList->OMSetRenderTargets(2,targets,FALSE,&depthTarget);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const D3D12_RECT scissor{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor); CommandList->SetGraphicsRootSignature(RefractionRoot.Get());
    CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); CommandList->IASetVertexBuffers(0,1,&RefractionVertexView); CommandList->IASetIndexBuffer(&RefractionIndexView);
    for (const auto& draw:FrameRefraction.Draws)
    {
        RefractionConstants value; RefractionGeometryRange range;
        if (!BuildRefractionDrawConstants(FrameRefraction,draw,base,value,range)) return Fail("DX12 refraction draw is invalid");
        if (!range.IndexCount) continue; UINT diffuse,normal,opacity;
        if (!texture(draw.DiffuseTexture,diffuse) || !texture(draw.NormalTexture,normal) || !texture(draw.OpacityTexture,opacity) || !constants(value)) return false;
        bind({MainDepthSRV,diffuse,normal,opacity,NullColorSRV,MainDepthSRV,NullColorSRV,NullColorSRV});
        CommandList->SetPipelineState(RefractionPipelines[(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u)].Get());
        CommandList->DrawIndexedInstanced(range.IndexCount,1,range.FirstIndex,0,0);
    }
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    for (auto index:{0,1,3}) Transition(RefractionImages[index].Resource.Get(),RefractionImages[index].State,sampled);
    auto& scene=HdrSurfaces[HdrScene]; auto& composite=RefractionImages[2]; Transition(scene.Resource.Get(),scene.State,sampled);
    Transition(composite.Resource.Get(),composite.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    const auto target=RTVArena.Cpu(composite.RTV); CommandList->OMSetRenderTargets(1,&target,FALSE,nullptr);
    constexpr UINT VolumeUniformRoot=9,VolumeImageRoot=10;
    if (!constants(base) || !BindLumenTranslucency(VolumeUniformRoot,VolumeImageRoot,true)) return false;
    bind({MainDepthSRV,NullColorSRV,NullColorSRV,NullColorSRV,scene.SRV,depth.SRV,RefractionImages[0].SRV,RefractionImages[1].SRV});
    CommandList->SetPipelineState(RefractionPipelines[4].Get()); CommandList->DrawInstanced(3,1,0,0);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); Transition(composite.Resource.Get(),composite.State,D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(scene.Resource.Get(),scene.State,D3D12_RESOURCE_STATE_COPY_DEST); CommandList->CopyResource(scene.Resource.Get(),composite.Resource.Get());
    RefractionResolved=true; RecordRefractionDraws(FrameRefraction,PendingFrame.Statistics); return true;
}
