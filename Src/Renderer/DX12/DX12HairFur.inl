// Upload-heap vertex/index/emitter storage is immutable throughout recording.
// Root buffer descriptors avoid descriptor rewrites after a draw is recorded.
HairFurExecution Hair;
HairFurFrame UploadedHair;
std::array<UINT,HairShadowMapCount> HairShadowDescriptors{};
bool PrepareHairShadowReceiver(const HairFurDesc& desc,HairFurConstants& constants)
{
    SetHairFurAnimation(Hair.Frame,Hair.Frame.SampleTimeSeconds,constants);
    constants.MotionOptions.Y=Hair.LumenReceiver?1u:0u;
    if (!BuildHairContactControl(desc.ViewProjection,*Hair.Frame.Lights,FrameContact.Enabled && !Hair.Frame.Geometry->Indices.empty(),
        Hair.Contact,ContactGenerated && ContactReceiverActive && ContactShadows.Result(),FrameSampleCount,constants.ContactControl,LastError)) return false;
    const bool moments=FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM;
    HairShadowDescriptors.fill(moments?NullColorSRV:NullShadowSRV);
    std::array<HairShadowMapReadiness,HairShadowMapCount> maps{};
    const bool requested=desc.SamplesShadowMap && !Hair.Frame.Geometry->Indices.empty();
    if (requested && Hair.Frame.Lights->ShadowBegin!=Hair.Frame.Lights->ShadowEnd)
    {
        const auto count=FrameCascades.Frame.CascadeCount,first=count?1u:0u;
        for (unsigned slot=first;slot<first+(count?count:1u) && slot<HairShadowMapCount;++slot)
        {
            const auto handle=slot?FrameCascades.Maps[slot-1]:LastShadowTexture;
            const auto& projection=slot?FrameCascades.Frame.Cascades[slot-1].LightViewProjection:desc.LightViewProjection;
            auto found=Textures.find(handle.Value);
            if (found==Textures.end() || found->second.Description.Format!=TextureFormat::Depth32) continue;
            auto& texture=found->second;
            maps[slot]={std::uint32_t(texture.Description.Width),std::uint32_t(texture.Description.Height),Hair.HasShadowView(handle,projection),false};
            if (!maps[slot].DepthReady) continue;
            if (moments)
            {
                if (!VSMMomentDescriptor(handle,HairShadowDescriptors[slot])) return false;
                maps[slot].MomentsReady=true;
            }
            else
            {
                Transition(texture.Resource.Get(),texture.State,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                HairShadowDescriptors[slot]=texture.SRV;
            }
        }
    }
    return BuildHairShadowReceiverConstants({desc.LightViewProjection,requested},*Hair.Frame.Lights,FrameCascades,FrameShadowFilter,maps,constants.Shadows,LastError);
}
ComPtr<ID3D12Resource> HairVertices,HairIndices,HairLights;
ComPtr<ID3D12RootSignature> HairRoot;
std::array<std::unordered_map<UINT,ComPtr<ID3D12PipelineState>>,HairFurPhaseCount> HairPipelines;
void DestroyHairFur()
{
    Hair={}; UploadedHair={}; HairVertices.Reset(); HairIndices.Reset(); HairLights.Reset(); HairRoot.Reset(); for (auto& pipelines:HairPipelines) pipelines.clear();
}
bool EnsureHairFur()
{
    const auto& frame=Hair.Frame;
    if (UploadedHair.Geometry!=frame.Geometry || UploadedHair.Lights!=frame.Lights)
    {
        if (!ValidateHairFurPublication(frame,LastError)) return false;
        ComPtr<ID3D12Resource> vertices=HairVertices,indices=HairIndices,lights=HairLights;
        const auto create=[&](const void* data,std::size_t bytes,ComPtr<ID3D12Resource>& buffer)
        {
            if (bytes>UINT_MAX) return Fail("DX12 hair buffer exceeds native view limits");
            ComPtr<ID3D12Resource> created;
            if (!CreateBufferResource(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ,created)) return false;
            void* mapped=nullptr; const D3D12_RANGE noRead{0,0};
            if (FAILED(created->Map(0,&noRead,&mapped))) return Fail("DX12 hair upload mapping failed");
            std::memcpy(mapped,data,bytes); const D3D12_RANGE written{0,bytes}; created->Unmap(0,&written); buffer=created; return true;
        };
        if (UploadedHair.Geometry!=frame.Geometry)
        {
            vertices.Reset(); indices.Reset(); const auto& geometry=*frame.Geometry;
            if (!geometry.Indices.empty() && (!create(geometry.Vertices.data(),geometry.Vertices.size()*sizeof(HairVertex),vertices) ||
                !create(geometry.Indices.data(),geometry.Indices.size()*sizeof(std::uint32_t),indices))) return false;
        }
        if (UploadedHair.Lights!=frame.Lights)
        {
            const HairLightSample empty{}; const auto& samples=frame.Lights->Samples;
            if (!create(samples.empty()?&empty:samples.data(),std::max(samples.size(),std::size_t{1})*sizeof(HairLightSample),lights)) return false;
        }
        HairVertices=vertices; HairIndices=indices; HairLights=lights; UploadedHair=frame;
    }
    if (!HairRoot)
    {
        D3D12_ROOT_PARAMETER parameters[3+HairShadowMapCount]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[0].Descriptor.ShaderRegister=0; parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[1].Descriptor.ShaderRegister=0; parameters[1].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_DESCRIPTOR_RANGE ranges[HairShadowMapCount+1]{};
        for (UINT slot=0;slot<HairShadowMapCount+1;++slot)
        {
            ranges[slot]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,slot+1,0,D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            auto& parameter=parameters[slot+2]; parameter.ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameter.DescriptorTable={1,&ranges[slot]}; parameter.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=static_cast<UINT>(std::size(parameters)); desc.pParameters=parameters;
        desc.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> serialized,errors;
        auto result=SerializeRootSignatureFunction(&desc,D3D_ROOT_SIGNATURE_VERSION_1,serialized.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("SerializeRootSignature(hair)",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(HairRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("CreateRootSignature(hair)",result);
    }
    return true;
}
bool EnsureHairPipeline(HairFurRasterPhase phase,UINT sampleCount)
{
    auto& pipelines=HairPipelines[HairFurPhaseIndex(phase)];
    const bool contact=phase==HairFurRasterPhase::Surface && Hair.UsesContactReceiver();
    const auto key=HairFurPipelineKey(phase,sampleCount,contact);
    if (pipelines.count(key)) return true;
    if (!LoadTechniqueShaderSource("DX12","HairFur.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex,pixel;
    const char* definition=phase==HairFurRasterPhase::Velocity?"EVANS_HAIR_VELOCITY":phase==HairFurRasterPhase::Surface?(contact?"EVANS_HAIR_CONTACT_RECEIVER":nullptr):phase==HairFurRasterPhase::Capture?"EVANS_HAIR_CAPTURE":phase==HairFurRasterPhase::ShadowMoments?"EVANS_HAIR_VSM_SHADOW":"EVANS_HAIR_DEPTH_ONLY";
    const D3D_SHADER_MACRO macros[]{{definition,"1"},{nullptr,nullptr}};
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) || !CompileShader("PSMain","ps_5_0",pixel,definition?macros:nullptr,true)) return false;
    const D3D12_INPUT_ELEMENT_DESC elements[]{
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(HairVertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(HairVertex,Tangent),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Reflectance),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",1,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Emission),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Ribbon),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=HairRoot.Get();
    pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()}; pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
    pipeline.InputLayout={elements,static_cast<UINT>(std::size(elements))};
    pipeline.SampleMask=UINT_MAX; pipeline.SampleDesc.Count=sampleCount;
    pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;
    pipeline.RasterizerState.DepthClipEnable=TRUE; pipeline.RasterizerState.MultisampleEnable=TRUE;
    if (phase==HairFurRasterPhase::ShadowDepth || phase==HairFurRasterPhase::ShadowMoments)
    { pipeline.RasterizerState.DepthBias=ShadowDepthBiasUnits; pipeline.RasterizerState.DepthBiasClamp=ShadowDepthBiasClamp; pipeline.RasterizerState.SlopeScaledDepthBias=ShadowSlopeScaledDepthBias; }
    for (auto& color:pipeline.BlendState.RenderTarget)
    {
        color.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL; color.SrcBlend=color.SrcBlendAlpha=D3D12_BLEND_ONE;
        color.DestBlend=color.DestBlendAlpha=D3D12_BLEND_ZERO; color.BlendOp=color.BlendOpAlpha=D3D12_BLEND_OP_ADD; color.LogicOp=D3D12_LOGIC_OP_NOOP;
    }
    auto& depth=pipeline.DepthStencilState; depth.DepthEnable=TRUE; depth.DepthWriteMask=phase==HairFurRasterPhase::Velocity?D3D12_DEPTH_WRITE_MASK_ZERO:D3D12_DEPTH_WRITE_MASK_ALL; depth.DepthFunc=D3D12_COMPARISON_FUNC_LESS_EQUAL;
    depth.StencilReadMask=D3D12_DEFAULT_STENCIL_READ_MASK; depth.StencilWriteMask=D3D12_DEFAULT_STENCIL_WRITE_MASK;
    depth.FrontFace.StencilFailOp=depth.FrontFace.StencilDepthFailOp=depth.FrontFace.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    depth.FrontFace.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; depth.BackFace=depth.FrontFace;
    pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline.NumRenderTargets=phase==HairFurRasterPhase::Surface?2:(phase==HairFurRasterPhase::Capture || phase==HairFurRasterPhase::ShadowMoments || phase==HairFurRasterPhase::Velocity)?1:0;
    for (UINT target=0;target<pipeline.NumRenderTargets;++target) pipeline.RTVFormats[target]=phase==HairFurRasterPhase::ShadowMoments?DXGI_FORMAT_R32G32_FLOAT:DXGI_FORMAT_R16G16B16A16_FLOAT;
    pipeline.DSVFormat=DXGI_FORMAT_D32_FLOAT;
    ComPtr<ID3D12PipelineState> created;
    if (FAILED(Evans::Renderer::CreateCachedGraphicsPipelineState(Device,&pipeline,IID_PPV_ARGS(created.GetAddressOf())))) return Fail("DX12 hair pipeline creation failed");
    pipelines.emplace(key,created); return true;
}
bool DrawHairGeometry(const HairFurConstants& constants,HairFurRasterPhase phase,UINT sampleCount)
{
    if (Hair.Frame.Geometry->Indices.empty()) return true;
    // Replay phases are compiled on demand, including after a sample-count change.
    if (!EnsureHairPipeline(phase,sampleCount)) return false;
    const auto offset=AlignUp(ConstantOffset,D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset>ConstantBufferCapacity || sizeof(constants)>ConstantBufferCapacity-offset) return Fail("DX12 hair constant arena is full");
    std::memcpy(ConstantData+offset,&constants,sizeof(constants)); ConstantOffset=offset+sizeof(constants);
    CommandList->SetGraphicsRootSignature(HairRoot.Get()); CommandList->SetPipelineState(HairPipelines[HairFurPhaseIndex(phase)].at(HairFurPipelineKey(phase,sampleCount,Hair.UsesContactReceiver())).Get());
    CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+offset);
    CommandList->SetGraphicsRootShaderResourceView(1,HairLights->GetGPUVirtualAddress());
    for (UINT slot=0;slot<HairShadowMapCount;++slot)
        CommandList->SetGraphicsRootDescriptorTable(slot+2,SRVArena.Gpu(phase==HairFurRasterPhase::Surface?HairShadowDescriptors[slot]:NullShadowSRV));
    CommandList->SetGraphicsRootDescriptorTable(2+HairShadowMapCount,SRVArena.Gpu(phase==HairFurRasterPhase::Surface && constants.ContactControl.X?ContactDescriptor:NullArraySRV));
    const D3D12_VERTEX_BUFFER_VIEW vertices{HairVertices->GetGPUVirtualAddress(),static_cast<UINT>(Hair.Frame.Geometry->Vertices.size()*sizeof(HairVertex)),sizeof(HairVertex)};
    const D3D12_INDEX_BUFFER_VIEW indices{HairIndices->GetGPUVirtualAddress(),static_cast<UINT>(Hair.Frame.Geometry->Indices.size()*sizeof(std::uint32_t)),DXGI_FORMAT_R32_UINT};
    CommandList->IASetVertexBuffers(0,1,&vertices); CommandList->IASetIndexBuffer(&indices); CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    CommandList->DrawIndexedInstanced(static_cast<UINT>(Hair.Frame.Geometry->Indices.size()),1,0,0,0); return true;
}
bool RenderHairFur(const HairFurConstants& constants)
{
    if (!EnsureHairPipeline(HairFurRasterPhase::Surface,FrameSampleCount)) return false;
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    auto& color=FrameSampleCount>1?MultisampleSurfaces[MultisampleColor]:HdrSurfaces[HdrScene];
    auto& normal=FrameSampleCount>1?MultisampleSurfaces[MultisampleMaterialFirst+1]:GBufferSurfaces[1];
    Transition(color.Resource.Get(),color.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    Transition(normal.Resource.Get(),normal.State,D3D12_RESOURCE_STATE_RENDER_TARGET);
    Transition(FrameSampleCount>1?MultisampleDepth.Resource.Get():MainDepth.Get(),FrameSampleCount>1?MultisampleDepth.State:MainDepthState,D3D12_RESOURCE_STATE_DEPTH_WRITE);
    const D3D12_CPU_DESCRIPTOR_HANDLE targets[]{RTVArena.Cpu(color.RTV),RTVArena.Cpu(normal.RTV)};
    const auto depth=DSVArena.Cpu(FrameSampleCount>1?MultisampleDepthDSV:MainDepthDSV);
    CommandList->OMSetRenderTargets(static_cast<UINT>(std::size(targets)),targets,FALSE,&depth);
    const D3D12_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; const D3D12_RECT scissor{0,0,CreateInfo.Width,CreateInfo.Height};
    CommandList->RSSetViewports(1,&viewport); CommandList->RSSetScissorRects(1,&scissor);
    const bool result=DrawHairGeometry(constants,HairFurRasterPhase::Surface,FrameSampleCount);
    CommandList->OMSetRenderTargets(0,nullptr,FALSE,nullptr); return result;
}
bool ReplayHairFur(const HairFurConstants& constants,HairFurRasterPhase phase) { return DrawHairGeometry(constants,phase,ActiveSampleCount); }
